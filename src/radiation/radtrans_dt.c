#include "../copyright.h"
/*==============================================================================
 * FILE: radtrans_dt.c
 *
 * PURPOSE: Computes timestep using CFL condition, for radiation transfer
 *   used with the operator split update to the energy equation.
 *   With MPI parallel jobs, finds minimum dt across all processors.
 *   Function returns minimum radiation dt.  Modeled after diff_dt()
 *   function.
 *
 * CONTAINS PUBLIC FUNCTIONS: 
 *   radtrans_dt()  - computes dt
 *============================================================================*/

#include <stdio.h>
#include <math.h>
#include "../defs.h"
#include "../athena.h"
#include "../globals.h"
#include "prototypes.h"
#include "../prototypes.h"

#ifdef RADIATION_TRANSFER
/*----------------------------------------------------------------------------*/
/* diff_dt:  */

Real radtrans_dt(DomainS *pD)
{
  int irefine, ir;
  Real dtmin_radtrans=(HUGE_NUMBER);
  Real dxmin,qa;
  int i,j,k,nl,nd;
  int il,iu,jl,ju,kl,ku;
  Real nu_rad, chi;
  Real qb;
  Real nu_con;
  GridS *pG=(pD->Grid);
  RadGridS *pRG=(pD->RadGrid);
  int ig,jg,kg,ioff,joff,koff;
#ifdef MPI_PARALLEL
  double my_dt, dt;
  int ierr;
#endif

/* Calculate minimum dx.  Always given by Grid on highest level of refinement */

  dxmin = pD->dx[0];
  if (pD->Nx[1] > 1) dxmin = MIN( dxmin, (pD->dx[1]) );
  if (pD->Nx[2] > 1) dxmin = MIN( dxmin, (pD->dx[2]) );

  qa = (dxmin*dxmin) / (4.0 * PI * PI);
  nu_con = 16.0 * PI * Gamma_1/ R_ideal;

  if (pG->Nx[0] > 1) {
    ioff = nghost - 1;
  } else ioff = 0;
  if (pG->Nx[1] > 1) {
    joff = nghost - 1; 
  } else joff = 0; 
  if (pG->Nx[2] > 1) {
    koff = nghost - 1;
  } else koff = 0;
  
  il = pRG->is, iu = pRG->ie;
  jl = pRG->js, ju = pRG->je;
  kl = pRG->ks, ku = pRG->ke;
  
  for (k=kl; k<=ku; k++) {
    kg = k + koff;
    for (j=jl; j<=ju; j++) {
      jg = j + joff;
      for (i=il; i<=iu; i++) {
	ig = i + ioff;
	chi = pRG->R[k][j][i][0].chi;
	nu_rad = nu_con * pRG->R[k][j][i][0].B * chi / 
                 (pG->tgas[kg][jg][ig] * pG->U[kg][jg][ig].d);
	dtmin_radtrans = MIN(dtmin_radtrans,(CourNo/nu_rad));			 
      }}}  

/* Find minimum timestep over all processors */
#ifdef MPI_PARALLEL
  my_dt = dtmin_radtrans;
  ierr = MPI_Allreduce(&my_dt, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  dtmin_radtrans = dt;
#endif /* MPI_PARALLEL */

  return dtmin_radtrans;
}

#endif /* RADIATION_TRANSFER */
