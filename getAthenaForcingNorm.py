import yt
import numpy as np
from mpi4py import MPI
import time
import pickle
import sys
import h5py
import os

#RES = int(sys.argv[3])
ID = sys.argv[1]
origRES = int(sys.argv[2])

comm  = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def shrink(origField):
    """
    shrinks a field by a factor of two using simple volume average
    """
    
    origRes = origField.shape[-1]
    
    destField = np.zeros((origRes/2,origRes/2,origRes/2),dtype=origField.dtype)    
    
    destField = (origField[::2,::2,::2] + origField[::2,1::2,::2] + 
        origField[::2,::2,1::2] + origField[::2,1::2,1::2] +
        origField[1::2,::2,::2] + origField[1::2,1::2,::2] + 
        origField[1::2,::2,1::2] + origField[1::2,1::2,1::2])/8.
    
    return destField

def readOneFieldWithOneProc(dd,fieldName,origRES,loadYT,loadHDF,numShrinks=0,order='C'):

    if rank == 0:
        
        if loadYT:
            data = np.array(dd[fieldName],dtype=np.float64)
        elif loadHDF:
            Filename = dd + fieldName + "-" + str(origRES)
            if os.path.isfile(Filename + '.h5'):
                Filename = Filename + '.h5'
            elif os.path.isfile(Filename + '.hdf5'):
                Filename + '.hdf5'
            else:
                print("Can't find corresponding hdf file for %s" % Filename)
                sys.exit(1)

            if order == 'C':
                data = np.float64(np.resize(
                    h5py.File(Filename, 'r')[fieldName],(origRES,origRES,origRES))) 

            elif order == 'F':
                data = np.float64(np.resize(
                    h5py.File(Filename, 'r')[fieldName],(origRES,origRES,origRES))).T 
            else:
                print("Wrong storage layout: order =  %s" % order)
                sys.exit(1)

            
        while numShrinks > 0:
            print("Reducing data from %d^3 to " % (data.shape[-1])),
            data = shrink(data)
            print("%d^3 done." % (data.shape[-1]))
            numShrinks -= 1

	if data.shape[-1] % size != 0:
	    print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
	    sys.exit(1)

    	finalRES = comm.bcast(data.shape[-1])
    else:
    	finalRES = comm.bcast(None)
        data = np.empty(finalRES**3,dtype=np.float64)
    
    outdata = np.empty((finalRES/size,finalRES,finalRES),dtype=np.float64)
    comm.Scatter([data,MPI.DOUBLE], [outdata, MPI.DOUBLE])

    if rank == 0:
    	print("Field %s read and distributed." % fieldName)

    return outdata

def readOneFieldWithNProc(dd,fieldName,origRES,loadYT,loadHDF,numShrinks=0,order='C'):

    Filename = dd + fieldName + "-" + str(origRES)
    if os.path.isfile(Filename + '.h5'):
        Filename += '.h5'
    elif os.path.isfile(Filename + '.hdf5'):
        Filename += '.hdf5'
    else:
        print("Can't find corresponding hdf file for %s" % Filename)
        sys.exit(1)

    h5Data = h5py.File(Filename, 'r')[fieldName]

    chunkSize = origRES/size
    startIdx = rank * chunkSize
    endIdx = (rank + 1) * chunkSize
    if endIdx == origRES:
        endIdx = None
    
    if order == 'C':
        data = np.float64(h5Data[0,startIdx:endIdx,:,:]) 
    elif order == 'F':
        data = np.float64(h5Data[0,:,:,startIdx:endIdx].T)
    else:
        print("Get your order straight!")
        sys.exit(1)
        

    if rank == 0:
    	print("Field %s read" % fieldName)

    return np.ascontiguousarray(data)

def readAllFieldsWithOneProc(loadPath, loadYT = False, loadHDF = False, origRES = None,
                             rhoField = None, velFields = None, magFields = None,
                             numShrinks=0,order='C'):
    
    if rank == 0:
        if loadYT:
            ds = yt.load(loadPath)
            dd = ds.h.covering_grid(level=0, left_edge=[0,0.0,0.0],dims=ds.domain_dimensions)
            RES = ds.domain_dimensions[0]
            
        elif loadHDF:
            dd = loadPath
            
    else:
        dd = None                 
        
    origRES = comm.bcast(origRES)
    finalRES = origRES / 2**numShrinks
    FinalShape = (finalRES//size,finalRES,finalRES)   
    
    if rhoField is not None:
        rho = readOneFieldWithOneProc(dd,rhoField,origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        rho = np.ones(FinalShape,dtype=np.float64) 
    
    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = readOneFieldWithOneProc(dd,velFields[0],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        U[1] = readOneFieldWithOneProc(dd,velFields[1],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        U[2] = readOneFieldWithOneProc(dd,velFields[2],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        U = None
        
    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = readOneFieldWithOneProc(dd,magFields[0],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        B[1] = readOneFieldWithOneProc(dd,magFields[1],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        B[2] = readOneFieldWithOneProc(dd,magFields[2],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        B = None
        
    return finalRES, rho, U, B

def readAllFieldsWithNProc(loadPath, loadYT = False, loadHDF = False, origRES = None,
                             rhoField = None, velFields = None, magFields = None,
                             accFields = None, numShrinks=0,order='C'):
    
    if not loadHDF:
        print("Sorry MPI loading with yt not supported")
        sys.exit(1)
    if numShrinks > 0:
        print("Sorry shrinking not tested yet")
        sys.exit(1)
	
    if origRES % size != 0:
	print("Data cannot be split evenly among processes. Abort (for now) - fix me!")
	sys.exit(1)
        
    finalRES = origRES / 2**numShrinks
    FinalShape = (finalRES//size,finalRES,finalRES)   
    
    if rhoField is not None:
        rho = readOneFieldWithNProc(loadPath,rhoField,origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        rho = np.ones(FinalShape,dtype=np.float64) 
    
    if velFields is not None:
        U = np.zeros((3,) + FinalShape,dtype=np.float64)
        U[0] = readOneFieldWithNProc(loadPath,velFields[0],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        U[1] = readOneFieldWithNProc(loadPath,velFields[1],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        U[2] = readOneFieldWithNProc(loadPath,velFields[2],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        U = None
        
    if magFields is not None:
        B = np.zeros((3,) + FinalShape,dtype=np.float64)  
        B[0] = readOneFieldWithNProc(loadPath,magFields[0],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        B[1] = readOneFieldWithNProc(loadPath,magFields[1],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        B[2] = readOneFieldWithNProc(loadPath,magFields[2],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        B = None
    
    if accFields is not None:
        Acc = np.zeros((3,) + FinalShape,dtype=np.float64)  
        Acc[0] = readOneFieldWithNProc(loadPath,accFields[0],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        Acc[1] = readOneFieldWithNProc(loadPath,accFields[1],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
        Acc[2] = readOneFieldWithNProc(loadPath,accFields[2],origRES,loadYT,loadHDF,numShrinks=numShrinks,order=order)
    else:
        Acc = None
        
    # CAREFUL assuming isothermal EOS here with c_s = 1 -> P = rho in code units
    return finalRES, rho, U, B, Acc, rho


        

rhoField = "density"
velFields = ["velocity_x","velocity_y","velocity_z"]
accFields = ['DV1','DV2','DV3']
loadPath =  ID + "/"
loadYT = False
loadHDF = True
order = 'F'

RES, rho, U , B, Acc, P = readAllFieldsWithNProc(loadPath,#"/mnt/scratch/gretephi/inv-cascade/MHD/128/id0/Turb.0060.vtk",
                                      loadYT = loadYT,
                                      loadHDF = loadHDF,
                                      origRES = origRES,
                                      rhoField = rhoField,
                                      velFields = velFields,
                                      accFields = accFields,
                                      order=order
                                      )

comm.Barrier()
    

if rank == 0:
    Quantities = ConfigObj() 
    Quantities.filename = str(ID).zfill(4) + "-flowQuantities-" + str(RES) + ".txt"

    for quantName in ["Mean","RMS"]:
        if not Quantities.has_key(quantName):
            Quantities[quantName] = {}

    dedt = Quantities["Mean"]["dedt"]
    dt = Quantities["Mean"]["dt"]

    dedt = comm.bcast(dedt)
    dt = comm.bcast(dt)

else:
    dedt = comm.bcast()
    dt = comm.bcast()



t0 = comm.allreduce(np.sum(rho))
t1 = comm.allreduce(np.sum(rho * Acc[0]))
t2 = comm.allreduce(np.sum(rho * Acc[1]))
t3 = comm.allreduce(np.sum(rho * Acc[2]))


Acc[0] -= t1/t0
Acc[1] -= t2/t0
Acc[2] -= t3/t0

t1 = comm.allreduce(np.sum(rho * (Acc[0]**2. + Acc[1]**2. + Acc[2]**2.)))
t2 = comm.allreduce(np.sum(rho*(U[0] * Acc[0] + U[1]*Acc[1] + U[2]*Acc[2])))


dvol = 1./1024.**3.
de = dedt * dt
aa = 0.5 * t1
aa = np.max([aa,1.e-20])
b = t2
c = -de/dvol
            
if (b >= 0.):
    s = (2.*c)/(b + np.sqrt(b*b - 4.0*aa*c))
else:
    s = (-b + np.sqrt(b*b - 4.0*aa*c))/(2.0*aa)

if rank == 0:
    print("s = %g" % s)
    Quantities["Mean"]["s"] = s
    Quantities.write()



# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 ai
