#ifndef FULL_RADIATION_TRANSFER_PROTOTYPES_H
#define FULL_RADIATION_TRANSFER_PROTOTYPES_H 
#include "../copyright.h"
/*==============================================================================
 * FILE: prototypes.h
 *
 * PURPOSE: Prototypes for all public functions in the /src/fullradiation dir
 *============================================================================*/

#include <stdarg.h>
#include "../athena.h"
#include "../defs.h"
#include "../config.h"

#ifdef FULL_RADIATION_TRANSFER

/* init_fullradiation.c */

VDFun_t init_fullradiation(MeshS *pM);
void fullradiation_destruct(MeshS *pM);


/* hydro_to_rad.c */

void hydro_to_fullrad(DomainS *pD);


/* bvals_fullrad.c */
void bvals_fullrad(DomainS *pD);
void bvals_fullrad_init(MeshS *pM);
void bvals_fullrad_destruct();
void bvals_fullrad_trans_fun(DomainS *pD, enum BCDirection dir, VRGIFun_t prob_bc);

/* utils_fullrad.c */
/* Update the momentums of specific intensity for each cell */

void UpdateRT(DomainS *pD);

/* FullRT_flux.c */

/* piece linear flux */
void flux_PLM(const Real dt, const Real ds, const Real vel, Real imu[3], Real imhalf[1]);
void flux_PPM(const Real dt, const Real ds, const Real vel, Real imu[5], Real imhalf[1]);
void lrstate_PPM(Real imu[5], Real iLeft[1], Real iRight[1]);

int permutation(int i, int j, int k, int **pl, int np);

/* fullRT_2d.c */
void fullRT_2d_init(RadGridS *pRG);
void fullRT_2d_destruct(void);

void fullRT_2d(DomainS *pD);

/* output_spec.c */
void output_spec(MeshS *pM);

#endif /* RADIATION_TRANSFER */
#endif /* RADIATION_TRANSFER_PROTOTYPES_H */
