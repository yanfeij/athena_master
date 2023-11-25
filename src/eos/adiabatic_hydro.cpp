//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file adiabatic_hydro.cpp
//! \brief implements functions in class EquationOfState for adiabatic hydrodynamics`

// C headers

// C++ headers
#include <cmath>   // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "eos.hpp"

// EquationOfState constructor

EquationOfState::EquationOfState(MeshBlock *pmb, ParameterInput *pin) :
    pmy_block_(pmb),
    neighbor_flooring_{pin->GetOrAddBoolean("hydro", "neighbor_flooring", false)},
    gamma_{pin->GetReal("hydro", "gamma")},
    density_floor_{pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024*float_min))},
    pressure_floor_{pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024*float_min))},
    scalar_floor_{pin->GetOrAddReal("hydro", "sfloor", std::sqrt(1024*float_min))} {

      if (pmb->phydro->fofc_enabled)
        fofc_.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
      if (neighbor_flooring_) {
        nbavg_d_.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
        nbavg_p_.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1);
      }

    }

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ConservedToPrimitive(AthenaArray<Real> &cons,
//!          const AthenaArray<Real> &prim_old, const FaceField &b,
//!          AthenaArray<Real> &prim, AthenaArray<Real> &bcc, Coordinates *pco,
//!          int il, int iu, int jl, int ju, int kl, int ku)
//! \brief Converts conserved into primitive variables in adiabatic hydro.

void EquationOfState::ConservedToPrimitive(
    AthenaArray<Real> &cons, const AthenaArray<Real> &prim_old, const FaceField &b,
    AthenaArray<Real> &prim, AthenaArray<Real> &bcc,
    Coordinates *pco, int il, int iu, int jl, int ju, int kl, int ku) {
  Real gm1 = GetGamma() - 1.0;
  int nbad_d = 0, nbad_p = 0;

  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        Real w_d, w_vx, w_vy, w_vz, w_p, dp;
        bool dfloor_used = false, pfloor_used = false;
        SingleConservativeToPrimitiveHydro(u_d, u_m1, u_m2, u_m3, u_e,
                                           w_d, w_vx, w_vy, w_vz, w_p,
                                           dp, dfloor_used, pfloor_used);
        // update counter, reset conserved if floor was used
        if (neighbor_flooring_) {
          nbavg_d_(k,j,i) = dfloor_used;
          nbavg_p_(k,j,i) = pfloor_used;
        }

        if (dfloor_used) {
          cons(IDN,k,j,i) = u_d;
          nbad_d++;
        }
        if (pfloor_used) {
          if (bookkeeping) efloor(k,j,i) += beta*dp/gm1;
          cons(IEN,k,j,i) = u_e;
          nbad_p++;
        }
        // update primitives
        // update primitives
        prim(IDN,k,j,i) = w_d;
        prim(IVX,k,j,i) = w_vx;
        prim(IVY,k,j,i) = w_vy;
        prim(IVZ,k,j,i) = w_vz;
        prim(IPR,k,j,i) = w_p;

      }
    }
  }


  // apply neighbor averaging
  if (neighbor_flooring_) {
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          if (nbavg_d_(k,j,i)) {
            // if density is bad
            Real eint_prev = prim(IPR,k,j,i)/gm1;
            AthenaArray<Real> cons_avg(NHYDRO), prim_avg(NHYDRO);
            NeighborAveragingConserved(cons,bcc,cons_avg,prim_avg,
                                       k,j,i,il,iu,jl,ju,kl,ku);
            for (int n=0; n<NHYDRO; ++n) {
              cons(n,k,j,i) = cons_avg(n);
              prim(n,k,j,i) = prim_avg(n);
            }
            if (bookkeeping) efloor(k,j,i) += (prim_avg(IPR)/gm1 - eint_prev)*beta;
          } else if (nbavg_p_(k,j,i)) {
            // this only handless the pressure floor case
            Real u_d  = cons(IDN,k,j,i);
            Real u_m1 = cons(IM1,k,j,i);
            Real u_m2 = cons(IM2,k,j,i);
            Real u_m3 = cons(IM3,k,j,i);
            Real u_e  = cons(IEN,k,j,i);

            Real e_k = 0.5*(SQR(u_m1) + SQR(u_m2) + SQR(u_m3))/u_d;
            Real eint_prev = u_e - e_k;
            Real eint_avg;
            NeighborAveragingEint(cons,bcc,eint_avg,k,j,i,il,iu,jl,ju,kl,ku);
            if (bookkeeping) efloor(k,j,i) += (eint_avg - eint_prev)*beta;
            cons(IEN,k,j,i) = eint_avg + e_k;
            prim(IPR,k,j,i) = eint_avg*gm1;
          }
        }
      }
    }
  }

  // updated number of bad cells in the mesh block
  // to be used elsewhere for diagnosing purposes
  pmy_block_->nbad_d = nbad_d;
  pmy_block_->nbad_p = nbad_p;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::PrimitiveToConserved(const AthenaArray<Real> &prim,
//!          const AthenaArray<Real> &bc, AthenaArray<Real> &cons, Coordinates *pco,
//!          int il, int iu, int jl, int ju, int kl, int ku);
//! \brief Converts primitive variables into conservative variables

void EquationOfState::PrimitiveToConserved(
    const AthenaArray<Real> &prim, const AthenaArray<Real> &bc,
    AthenaArray<Real> &cons, Coordinates *pco,
    int il, int iu, int jl, int ju, int kl, int ku) {
  Real igm1 = 1.0/(GetGamma() - 1.0);
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        Real& u_d  = cons(IDN,k,j,i);
        Real& u_m1 = cons(IM1,k,j,i);
        Real& u_m2 = cons(IM2,k,j,i);
        Real& u_m3 = cons(IM3,k,j,i);
        Real& u_e  = cons(IEN,k,j,i);

        const Real& w_d  = prim(IDN,k,j,i);
        const Real& w_vx = prim(IVX,k,j,i);
        const Real& w_vy = prim(IVY,k,j,i);
        const Real& w_vz = prim(IVZ,k,j,i);
        const Real& w_p  = prim(IPR,k,j,i);

        u_d = w_d;
        u_m1 = w_vx*w_d;
        u_m2 = w_vy*w_d;
        u_m3 = w_vz*w_d;
        u_e = w_p*igm1 + 0.5*w_d*(SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
//! \brief returns adiabatic sound speed given vector of primitive variables
Real EquationOfState::SoundSpeed(const Real prim[NHYDRO]) {
  return std::sqrt(gamma_*prim[IPR]/prim[IDN]);
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j,
//!                                                 =int i)
//! \brief Apply density and pressure floors to reconstructed L/R cell interface states
void EquationOfState::ApplyPrimitiveFloors(AthenaArray<Real> &prim, int k, int j, int i) {
  Real& w_d  = prim(IDN,i);
  Real& w_p  = prim(IPR,i);

  // apply (prim) density floor
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // apply pressure floor
  w_p = (w_p > pressure_floor_) ?  w_p : pressure_floor_;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void EquationOfState::ApplyPrimitiveConservedFloors(AthenaArray<Real> &prim,
//!           AthenaArray<Real> &cons, FaceField &b, int k, int j, int i) {
//! \brief Apply pressure (prim) floor and correct energy (cons) (typically after W(U))
void EquationOfState::ApplyPrimitiveConservedFloors(
    AthenaArray<Real> &prim, AthenaArray<Real> &cons, AthenaArray<Real> &bcc,
    int k, int j, int i) {
  Real gm1 = GetGamma() - 1.0;
  Real& w_d  = prim(IDN,k,j,i);
  Real& w_p  = prim(IPR,k,j,i);

  Real& u_d  = cons(IDN,k,j,i);
  Real& u_e  = cons(IEN,k,j,i);
  // apply (prim) density floor, without changing momentum or energy
  w_d = (w_d > density_floor_) ?  w_d : density_floor_;
  // ensure cons density matches
  u_d = w_d;

  Real e_k = 0.5*w_d*(SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i))
                      + SQR(prim(IVZ,k,j,i)));
  // apply pressure floor, correct total energy
  u_e = (w_p > pressure_floor_) ?
        u_e : ((pressure_floor_/gm1) + e_k);
  w_p = (w_p > pressure_floor_) ?
        w_p : pressure_floor_;

  return;
}
