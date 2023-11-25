//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos.cpp
//! \brief implements common functions in class EquationOfState

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "eos.hpp"

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitiveTest(AthenaArray<Real> &cons,
//!   const AthenaArray<Real> &prim_old, const FaceField &b,
//!   AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!   int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Just for test. cons(IEN) only contains e_int + e_k even if it is MHD

void EquationOfState::ConservedToPrimitiveTest(
    const AthenaArray<Real> &cons, const AthenaArray<Real> &bcc,
    int il, int iu, int jl, int ju, int kl, int ku) {
  int nbad_d = 0, nbad_p = 0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real u_d  = cons(IDN,k,j,i);
        Real u_m1 = cons(IM1,k,j,i);
        Real u_m2 = cons(IM2,k,j,i);
        Real u_m3 = cons(IM3,k,j,i);
        Real u_e  = cons(IEN,k,j,i);
        Real e_mag = 0.0;
        if (MAGNETIC_FIELDS_ENABLED) {
          Real bcc1 = bcc(IB1,k,j,i);
          Real bcc2 = bcc(IB2,k,j,i);
          Real bcc3 = bcc(IB3,k,j,i);
          e_mag = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
        }

        Real w_d, w_vx, w_vy, w_vz, w_p, dp;
        bool dfloor_used = false, pfloor_used = false;
        SingleConservativeToPrimitiveMHD(u_d, u_m1, u_m2, u_m3, u_e,
                                         w_d, w_vx, w_vy, w_vz, w_p,
                                         dp, dfloor_used, pfloor_used, e_mag);
        fofc_(k,j,i) = dfloor_used || pfloor_used;
        if (dfloor_used) nbad_d++;
        if (pfloor_used) nbad_p++;
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::SingleConservativeToPrimitive(
//!  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e,
//!  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
//!  Real &dp, bool &dfloor_used, bool &pfloor_used)
//! \brief Converts single conserved variable into primitive variable in hydro.
//!        Checks floor needs
void EquationOfState::SingleConservativeToPrimitiveHydro(
  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e,
  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
  Real &dp, bool &dfloor_used, bool &pfloor_used)  {
  // apply density floor, without changing momentum or energy
  if (u_d < density_floor_) {
    u_d = density_floor_;
    dfloor_used = true;
  }

  w_d = u_d;

  Real di = 1.0/u_d;
  w_vx = u_m1*di;
  w_vy = u_m2*di;
  w_vz = u_m3*di;

  if (NON_BAROTROPIC_EOS) {
    Real gm1 = gamma_ - 1.0;
    Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    w_p = gm1*(u_e - e_k);

    // apply pressure floor, correct total energy
    if (w_p < pressure_floor_) {
      dp = pressure_floor_ - w_p;
      w_p = pressure_floor_;
      u_e = w_p/gm1 + e_k;
      pfloor_used = true;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::SingleConservativeToPrimitive(
//!  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e, Real emag,
//!  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
//!  Real &dp, bool &dfloor_used, bool &pfloor_used)
//! \brief Converts single conserved variable into primitive variable in hydro.
//!        Checks floor needs
void EquationOfState::SingleConservativeToPrimitiveMHD(
  Real &u_d, Real &u_m1, Real &u_m2, Real &u_m3, Real &u_e,
  Real &w_d, Real &w_vx, Real &w_vy, Real &w_vz, Real &w_p,
  Real &dp, bool &dfloor_used, bool &pfloor_used, const Real e_mag)  {
  // apply density floor, without changing momentum or energy
  if (u_d < density_floor_) {
    u_d = density_floor_;
    dfloor_used = true;
  }

  w_d = u_d;

  Real di = 1.0/u_d;
  w_vx = u_m1*di;
  w_vy = u_m2*di;
  w_vz = u_m3*di;

  if (NON_BAROTROPIC_EOS) {
    Real gm1 = gamma_ - 1.0;
    Real e_k = 0.5*di*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    w_p = gm1*(u_e - e_k - e_mag);

    // apply pressure floor, correct total energy
    if (w_p < pressure_floor_) {
      dp = pressure_floor_ - w_p;
      w_p = pressure_floor_;
      u_e = w_p/gm1 + e_k + e_mag;
      pfloor_used = true;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::NeighborAveragingConserved
//! \brief Calculate neighbor averaged values for all hydro variables
void EquationOfState::NeighborAveragingConserved(
  const AthenaArray<Real> &cons, const AthenaArray<Real> &bcc,
  AthenaArray<Real> &cons_avg, AthenaArray<Real> &prim_avg,
  int k, int j, int i,
  int il, int iu, int jl, int ju, int kl, int ku) {
  cons_avg.ZeroClear();
  Real vol_neighbors = 0.0;
  int nvars = NHYDRO;

  if (NON_BAROTROPIC_EOS) nvars--;

  int koff[] = {1,-1,0,0,0,0};
  int joff[] = {0,0,1,-1,0,0};
  int ioff[] = {0,0,0,0,1,-1};

  // calculate averaged density and momentum
  for (int idx=0; idx<6; ++idx) {
    int k0=k+koff[idx];
    int j0=j+joff[idx];
    int i0=i+ioff[idx];
    // skip idices outside mesh block
    if ((i0<il) || (i0>iu) || (j0<jl) || (j0>ju) || (k0<kl) || (k0>ku)) continue;

    // check if neighbor is good
    if (!nbavg_d_(k0,j0,i0)) {
      Real vol = 1.0; // will change to real cell volume
      vol_neighbors += vol;
      // sum density and momentum
      for (int n=0; n<nvars; ++n)
        cons_avg(n) += cons(n,k0,j0,i0)*vol;
    }
  }
  // assign means
  for (int n=0; n<nvars; ++n) cons_avg(n) = cons_avg(n)/vol_neighbors;
  // apply floor if there is no good neighbors
  cons_avg(IDN) = vol_neighbors == 0.0 ? density_floor_ : cons_avg(IDN);
  cons_avg(IM1) = vol_neighbors == 0.0 ? cons(IM1,k,j,i)/cons(IDN,k,j,i)
               * density_floor_ : cons_avg(IM1);
  cons_avg(IM2) = vol_neighbors == 0.0 ? cons(IM2,k,j,i)/cons(IDN,k,j,i)
               * density_floor_ : cons_avg(IM2);
  cons_avg(IM3) = vol_neighbors == 0.0 ? cons(IM3,k,j,i)/cons(IDN,k,j,i)
               * density_floor_ : cons_avg(IM3);
  // calculate new kinetic energy
  Real di = 1.0/cons_avg(IDN);
  prim_avg(IDN) = cons_avg(IDN);
  prim_avg(IVX) = di*cons_avg(IM1);
  prim_avg(IVY) = di*cons_avg(IM2);
  prim_avg(IVZ) = di*cons_avg(IM3);

  // get averaged internal energy
  if (NON_BAROTROPIC_EOS) {
    Real gm1 = gamma_ - 1.0;
    Real e_k = 0.5*di*(SQR(cons_avg(IM1))+SQR(cons_avg(IM2))+SQR(cons_avg(IM3)));
    Real eint_avg;
    NeighborAveragingEint(cons, bcc, eint_avg, k, j, i, il, iu, jl, ju, kl, ku);
    cons_avg(IEN) = eint_avg + e_k;
    prim_avg(IPR) = eint_avg*gm1;
    if (MAGNETIC_FIELDS_ENABLED) {
      const Real bcc1 = bcc(IB1,k,j,i);
      const Real bcc2 = bcc(IB2,k,j,i);
      const Real bcc3 = bcc(IB3,k,j,i);
      Real e_mag = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
      cons_avg(IEN) += e_mag;
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::NeighborAveragingEint
//! \brief Calculate neighbor averaged values for internal energy
void EquationOfState::NeighborAveragingEint(const AthenaArray<Real> &cons,
  const AthenaArray<Real> &bcc, Real &eint_avg, int k, int j, int i,
  int il, int iu, int jl, int ju, int kl, int ku) {
  Real gm1 = gamma_ - 1.0;
  Real vol_neighbors = 0.0;
  Real q_neighbors = 0.0;

  int koff[] = {1,-1,0,0,0,0};
  int joff[] = {0,0,1,-1,0,0};
  int ioff[] = {0,0,0,0,1,-1};

  for (int idx=0; idx<6; ++idx) {
    int k0=k+koff[idx];
    int j0=j+joff[idx];
    int i0=i+ioff[idx];
    // skip idices outside mesh block
    if ((i0<il) || (i0>iu) || (j0<jl) || (j0>ju) || (k0<kl) || (k0 >ku)) continue;

    if (!nbavg_p_(k0,j0,i0)) {
      // calculate internal energy only if the neighboring cell is good
      Real nu_d  = cons(IDN,k0,j0,i0);
      Real nu_m1 = cons(IM1,k0,j0,i0);
      Real nu_m2 = cons(IM2,k0,j0,i0);
      Real nu_m3 = cons(IM3,k0,j0,i0);
      Real nu_e  = cons(IEN,k0,j0,i0);
      Real ndi = 1.0/nu_d;
      Real ne_k = 0.5*ndi*(SQR(nu_m1) + SQR(nu_m2) + SQR(nu_m3));
      Real neint = nu_e - ne_k;

      if (MAGNETIC_FIELDS_ENABLED) {
        const Real bcc1 = bcc(IB1,k0,j0,i0);
        const Real bcc2 = bcc(IB2,k0,j0,i0);
        const Real bcc3 = bcc(IB3,k0,j0,i0);
        Real ne_mag = 0.5*(SQR(bcc1) + SQR(bcc2) + SQR(bcc3));
        neint -= ne_mag;
      }

      Real vol = 1.0; // will change to real cell volume
      vol_neighbors += vol;
      q_neighbors += neint;
    }
  }

  // apply floor if there is no good neighbors
  eint_avg = vol_neighbors>0 ? q_neighbors/vol_neighbors : pressure_floor_/gm1;

  return;
}
