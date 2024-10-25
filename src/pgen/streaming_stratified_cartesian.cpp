//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  PRIVATE FUNCTION PROTOTYPES:
//  - ran2() - random number generator from NR
//
//  REFERENCE: Hawley, J. F. & Balbus, S. A., ApJ 400, 595-609 (1992).*/
//
//======================================================================================

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // cout, endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp" // ran2()

#if NON_BAROTROPIC_EOS
#error "This problem generator requires isothermal equation of state!"
#endif

namespace {
Real amp, nwx, nwy, nwz, sigma_g0; // amplitude, Wavenumbers
Real etaVk; // The amplitude of pressure gradient force
Real gm1, iso_cs;
Real x1size, x2size, x3size;
Real Omega0, qshear;
Real pslope;
Real user_dt;
Real initial_D2G[NDUSTFLUIDS];
Real Stokes_number[NDUSTFLUIDS];
Real kappap, kappap2, AN(0.0), BN(0.0), Psi(0.0), Kai0;
Real Hg, Hdust[NDUSTFLUIDS], Hratio[NDUSTFLUIDS];
//Real Kpar, kx, ky, kz, omg_osi, s_grow;

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void VerticalGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void InertialForces(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);
Real UserTimeStep(MeshBlock *pmb);
Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout);

void LowerZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void UpperZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh);
} // namespace

//======================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Init the Mesh properties //======================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  x1size = mesh_size.x1max - mesh_size.x1min;
  x2size = mesh_size.x2max - mesh_size.x2min;
  x3size = mesh_size.x3max - mesh_size.x3min;

  // initialize global variables
  amp      = pin->GetReal("problem",         "amp");
  sigma_g0 = pin->GetOrAddReal("problem",    "sigma_g0", 1.0);
  nwx      = pin->GetOrAddInteger("problem", "nwx",      1);
  nwy      = pin->GetOrAddInteger("problem", "nwy",      1);
  nwz      = pin->GetOrAddInteger("problem", "nwz",      1);
  etaVk    = pin->GetReal("problem",         "etaVk");
  iso_cs   = pin->GetReal("hydro",           "iso_sound_speed");

  user_dt = pin->GetOrAddReal("time", "user_dt", 0.0);

  Omega0 = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);
  qshear = pin->GetOrAddReal("orbital_advection", "qshear", 0.0);


  kappap  = 2.0*(2.0 - qshear);
  kappap2 = SQR(kappap);
  Kai0    = 2.0*etaVk*iso_cs;
  Hg      = iso_cs/Omega0;

  if (NDUSTFLUIDS > 0) {

    for (int n=0; n<NDUSTFLUIDS; ++n) {
      // Dust to gas ratio && dust stokes numbers
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
      Hdust[n]         = Hg*Hratio[n];

      AN += (initial_D2G[n] * Stokes_number[n])/(1.0 + kappap2 * SQR(Stokes_number[n]));
      BN += (initial_D2G[n])/(1.0 + kappap2 * SQR(Stokes_number[n]));
    }

    AN *= kappap2;
    BN += 1.0;
    Psi = 1.0/(SQR(AN) + kappap2*SQR(BN));

    EnrollUserDustStoppingTime(MyStoppingTime);
  }

  EnrollUserExplicitSourceFunction(MySource);

  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(UserTimeStep);

  if (mesh_bcs[BoundaryFace::inner_x1] != GetBoundaryFlag("periodic")) {
    std::stringstream msg;
    msg << "The boundary condition of x1 direction must be periodic!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_bcs[BoundaryFace::outer_x1] != GetBoundaryFlag("periodic")) {
    std::stringstream msg;
    msg << "The boundary condition of x1 direction must be periodic!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, LowerZ);

  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, UpperZ);

  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, DustFluidsRatioMaximum, "RatioMax", UserHistoryOperation::max);

  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief
void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "This problem file must be setup in the cartesian coordinate!" << std::endl;
    ATHENA_ERROR(msg);
  }

  if ((block_size.nx2 == 1) || (block_size.nx3 > 1)) {
    std::stringstream msg;
    msg << "This problem file must be setup in 2D!" << std::endl;
    ATHENA_ERROR(msg);
  }

  std::int64_t iseed = -1 - gid;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x_dis = pcoord->x1v(i);
        Real z_dis = pcoord->x2v(j);
        Real y_dis = pcoord->x3v(k);

        Real K_vel    = qshear*Omega0*x_dis;
        Real gas_vel1 = AN*Kai0*Psi;
        Real gas_vel2 = 0.0;
        Real gas_vel3 = -K_vel - 0.5*kappap2*BN*Kai0*Psi;

        Real delta_gas_vel1 = amp*iso_cs*(ran2(&iseed) - 0.5);
        Real delta_gas_vel2 = amp*iso_cs*(ran2(&iseed) - 0.5);
        Real delta_gas_vel3 = amp*iso_cs*(ran2(&iseed) - 0.5);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        //gas_dens = sigma_g0/(std::sqrt(TWO_PI)*Hg)*std::exp(-SQR(z_dis)/(2.0*SQR(Hg)));
        gas_dens = sigma_g0;
        gas_mom1 = gas_dens * (gas_vel1 + delta_gas_vel1);
        gas_mom2 = gas_dens * (gas_vel2 + delta_gas_vel2);
        gas_mom3 = gas_dens * (gas_vel3 + delta_gas_vel3);

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real sigma_d0  = initial_D2G[dust_id]*sigma_g0;
            Real dust_vel1 = (gas_vel1 + 2.0*Stokes_number[dust_id]*(gas_vel3 + K_vel))/(1.0 + kappap2*SQR(Stokes_number[dust_id]));
            Real dust_vel2 = 0.0;
            Real dust_vel3 = -K_vel + ((gas_vel3 + K_vel) - (2.0 - qshear)*Stokes_number[dust_id]*gas_vel1)/(1.0 + kappap2*SQR(Stokes_number[dust_id]));

            Real delta_dust_vel1 = amp*iso_cs*(ran2(&iseed)-0.5);
            Real delta_dust_vel2 = amp*iso_cs*(ran2(&iseed)-0.5);
            Real delta_dust_vel3 = amp*iso_cs*(ran2(&iseed)-0.5);

            Real &dust_dens = pdustfluids->df_cons(rho_id, k, j, i);
            Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
            Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
            Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

            //dust_dens = sigma_d0/(std::sqrt(TWO_PI)*Hdust[n])*std::exp(-SQR(z_dis)/(2.0*SQR(Hdust[n])));
            dust_dens = sigma_d0;
            dust_mom1 = dust_dens * (dust_vel1 + delta_dust_vel1);
            dust_mom2 = dust_dens * (dust_vel2 + delta_dust_vel2);
            dust_mom3 = dust_dens * (dust_vel3 + delta_dust_vel3);
          }
        }
      }
    }
  }
  return;
}


namespace {
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  InertialForces(pmb,   time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  PressureGradient(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  //VerticalGravity(pmb,  time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void PressureGradient(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_rho  = prim(IDN, k, j, i);
        Real press_gra       = gas_rho*Kai0*Omega0*dt;
        Real &gas_mom1       = cons(IM1, k, j, i);
        gas_mom1            += press_gra;
      }
    }
  }
  return;
}


void VerticalGravity(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int nc1 = pmb->ncells1;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      AthenaArray<Real> vert_gravity(nc1);
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        Real vertical_dis;
        vertical_dis = pmb->pcoord->x2v(j);
        vert_gravity(i) = -SQR(Omega0)*vertical_dis;

        const Real &gas_rho = prim(IDN, k, j, i);
        Real &gas_mom2      = cons(IM2, k, j, i);
        Real &gas_mom3      = cons(IM3, k, j, i);

        gas_mom2 += gas_rho*vert_gravity(i)*dt;
      }

      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          const Real &dust_rho = prim_df(rho_id, k, j, i);
          Real &dust_mom2      = cons_df(v2_id,  k, j, i);
          Real &dust_mom3      = cons_df(v3_id,  k, j, i);

          dust_mom2 += dust_rho*vert_gravity(i)*dt;
        }
      }
    }
  }
  return;
}


void InertialForces(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int nc1 = pmb->ncells1;
  Hydro *ph = pmb->phydro;

  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        const Real &gas_rho = prim(IDN, k, j, i);
        const Real &xc      = pmb->pcoord->x1v(i);
        const Real qO2      = qshear*SQR(Omega0);
        const Real mom1     = gas_rho*prim(IM1, k, j, i);

        Real &gas_mom1  = cons(IM1, k, j, i);
        Real &gas_mom3  = cons(IM3, k, j, i);
        gas_mom1       += 2.0*dt*(Omega0*(gas_rho*prim(IM3, k, j, i))+qO2*gas_rho*xc);
        gas_mom3       -= 2.0*dt*Omega0*mom1;

        if (NON_BAROTROPIC_EOS) {
          const Real phic = qO2*SQR(xc);
          const Real phil = qO2*SQR(pmb->pcoord->x1f(i));
          const Real phir = qO2*SQR(pmb->pcoord->x1f(i+1));

          Real &gas_erg  = cons(IEN, k, j, i);
          gas_erg       += dt*(ph->flux[X1DIR](IDN, k, j, i)*(phic-phil)+
                               ph->flux[X1DIR](IDN, k, j, i+1)*(phir-phic))/pmb->pcoord->dx1f(i);
        }
      }

      for (int n=0; n<NDUSTFLUIDS; ++n) {
        int dust_id = n;
        int rho_id  = 4*dust_id;
        int v1_id   = rho_id + 1;
        int v2_id   = rho_id + 2;
        int v3_id   = rho_id + 3;
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          const Real &dust_rho = prim_df(rho_id, k, j, i);
          const Real &xc       = pmb->pcoord->x1v(i);
          const Real qO2       = qshear*SQR(Omega0);
          const Real mom1      = dust_rho*prim_df(v1_id, k, j, i);

          Real &dust_mom1  = cons_df(v1_id, k, j, i);
          Real &dust_mom3  = cons_df(v3_id, k, j, i);
          dust_mom1       += 2.0*dt*(Omega0*(dust_rho*prim_df(v3_id, k, j, i)) + qO2*dust_rho*xc);
          dust_mom3       -= 2.0*dt*Omega0*mom1;
        }
      }
    }
  }
  return;
}



void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {

    Real inv_Omega = 1.0/Omega0;

    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            Real &st_time = stopping_time(dust_id, k, j, i);
            st_time       = Stokes_number[dust_id]*inv_Omega;
          }
        }
      }
    }
  return;
}


Real UserTimeStep(MeshBlock *pmb) {

  Real min_user_dt = user_dt;
  return min_user_dt;
}


void LowerZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  Real inv_sqrt_2PIHg = 1.0/(std::sqrt(TWO_PI)*Hg);
  Real inv_2square_Hg = 1.0/(2.0*SQR(Hg));

  Real inv_sqrt_2PIHd[NDUSTFLUIDS];
  Real inv_2square_Hd[NDUSTFLUIDS];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    inv_sqrt_2PIHd[dust_id] = 1.0/(std::sqrt(TWO_PI)*Hdust[dust_id]);
    inv_2square_Hd[dust_id] = 1.0/(2.0*SQR(Hdust[dust_id]));
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real x_dis = pco->x1v(i);
        Real z_dis = pco->x2v(jl-j);
        Real y_dis = pco->x3v(k);

        Real &gas_rho_ac  = prim(IDN, k, jl, i);
        Real &gas_vel1_ac = prim(IM1, k, jl, i);
        Real &gas_vel2_ac = prim(IM2, k, jl, i);
        Real &gas_vel3_ac = prim(IM3, k, jl, i);

        Real &gas_rho_gh  = prim(IDN, k, jl-j, i);
        Real &gas_vel1_gh = prim(IM1, k, jl-j, i);
        Real &gas_vel2_gh = prim(IM2, k, jl-j, i);
        Real &gas_vel3_gh = prim(IM3, k, jl-j, i);

        gas_rho_gh  = sigma_g0*inv_sqrt_2PIHg*std::exp(-SQR(z_dis)*inv_2square_Hg);
        gas_vel1_gh = gas_vel1_ac;
        gas_vel3_gh = gas_vel3_ac;
        gas_vel2_gh = (gas_vel2_ac < 0.0) ? gas_vel2_ac : 0.0;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real sigma_d0  = initial_D2G[dust_id]*sigma_g0;

            Real &dust_rho_ac  = prim_df(rho_id, k, jl, i);
            Real &dust_vel1_ac = prim_df(v1_id,  k, jl, i);
            Real &dust_vel2_ac = prim_df(v2_id,  k, jl, i);
            Real &dust_vel3_ac = prim_df(v3_id,  k, jl, i);

            Real &dust_rho_gh  = prim_df(rho_id, k, jl-j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, jl-j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, jl-j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, jl-j, i);

            dust_rho_gh  = sigma_d0*inv_sqrt_2PIHd[dust_id]*std::exp(-SQR(z_dis)*inv_2square_Hd[dust_id]);
            dust_vel1_gh = dust_vel1_ac;
            dust_vel3_gh = dust_vel3_ac;
            dust_vel2_gh = (dust_vel2_ac < 0.0) ? dust_vel2_ac : 0.0;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values

void UpperZ(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df, FaceField &b,
  Real time, Real dt, int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  Real inv_sqrt_2PIHg = 1.0/(std::sqrt(TWO_PI)*Hg);
  Real inv_2square_Hg = 1.0/(2.0*SQR(Hg));

  Real inv_sqrt_2PIHd[NDUSTFLUIDS];
  Real inv_2square_Hd[NDUSTFLUIDS];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    inv_sqrt_2PIHd[dust_id] = 1.0/(std::sqrt(TWO_PI)*Hdust[dust_id]);
    inv_2square_Hd[dust_id] = 1.0/(2.0*SQR(Hdust[dust_id]));
  }

  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=il; i<=iu; ++i) {
        Real x_dis = pco->x1v(i);
        Real z_dis = pco->x2v(ju+j);
        Real y_dis = pco->x3v(k);

        Real &gas_rho_ac  = prim(IDN, k, ju, i);
        Real &gas_vel1_ac = prim(IM1, k, ju, i);
        Real &gas_vel2_ac = prim(IM2, k, ju, i);
        Real &gas_vel3_ac = prim(IM3, k, ju, i);

        Real &gas_rho_gh  = prim(IDN, k, ju+j, i);
        Real &gas_vel1_gh = prim(IM1, k, ju+j, i);
        Real &gas_vel2_gh = prim(IM2, k, ju+j, i);
        Real &gas_vel3_gh = prim(IM3, k, ju+j, i);

        gas_rho_gh  = sigma_g0*inv_sqrt_2PIHg*std::exp(-SQR(z_dis)*inv_2square_Hg);
        gas_vel1_gh = gas_vel1_ac;
        gas_vel3_gh = gas_vel3_ac;
        gas_vel2_gh = (gas_vel2_ac > 0.0) ? gas_vel2_ac : 0.0;

        if (NDUSTFLUIDS > 0) {
          for (int n=0; n<NDUSTFLUIDS; ++n) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real sigma_d0 = initial_D2G[dust_id]*sigma_g0;

            Real &dust_rho_ac  = prim_df(rho_id, k, ju, i);
            Real &dust_vel1_ac = prim_df(v1_id,  k, ju, i);
            Real &dust_vel2_ac = prim_df(v2_id,  k, ju, i);
            Real &dust_vel3_ac = prim_df(v3_id,  k, ju, i);

            Real &dust_rho_gh  = prim_df(rho_id, k, ju+j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, ju+j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, ju+j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, ju+j, i);

            dust_rho_gh  = sigma_d0*inv_sqrt_2PIHd[dust_id]*std::exp(-SQR(z_dis)*inv_2square_Hd[dust_id]);
            dust_vel1_gh = dust_vel1_ac;
            dust_vel3_gh = dust_vel3_ac;
            dust_vel2_gh = (dust_vel2_ac > 0.0) ? dust_vel2_ac : 0.0;
          }
        }
      }
    }
  }
  return;
}


Real DustFluidsRatioMaximum(MeshBlock *pmb, int iout) {
  Real ratio_maximum = 0.0;

  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &df_prim = pmb->pdustfluids->df_prim;
  AthenaArray<Real> &w       = pmb->phydro->w;

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real &gas_rho  = w(IDN,k,j,i);
        //for (int n=0; n<NDUSTFLUIDS; n++) {
        int dust_id     = IDN;
        int rho_id      = 4*dust_id;
        Real &dust_rho  = df_prim(rho_id, k, j, i);
        ratio_maximum   = std::max(ratio_maximum, dust_rho/gas_rho);
        //}
      }
    }
  }
  return ratio_maximum;
}

}

