//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//  \brief Initializes stratified Keplerian accretion disk in both cylindrical and
//  spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <cstdlib>    // srand
#include <cstring>    // strcmp()
#include <fstream>
#include <iostream>   // endl
#include <limits>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../dustfluids/dustfluids.hpp"
#include "../orbital_advection/orbital_advection.hpp"
#include "../utils/utils.hpp" // ran2()

namespace {
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k);
Real PoverRho(const Real rad, const Real phi, const Real z);
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope);
Real DenProfileCyl_Gas(const Real rad, const Real phi, const Real z);
Real VelProfileCyl_Gas(const Real rad, const Real phi, const Real z);
Real DenProfileCyl_Dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio);
Real VelProfileCyl_Dust(const Real rad, const Real phi, const Real z);
void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void DustVelProfileCyl_NSH(const Real Ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3);
void Keplerian_interpolate(const Real r_active, const Real r_ghost, const Real vphi_active,
    Real &vphi_ghost);
void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
    const Real slope, Real &rho_ghost);
// problem parameters which are useful to make global to this file
//
Real Keplerian_velocity(const Real rad);
Real Delta_gas_vr(const Real vk,    const Real SN, const Real QN,    const Real Psi);
Real Delta_gas_vphi(const Real vk,  const Real SN, const Real QN,    const Real Psi);
Real Delta_dust_vr(const Real ts,   const Real vk, const Real d_vgr, const Real d_vgphi);
Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi);

// User Sources
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);
void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s);

// User Stopping time
void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time_array);

Real gm0, r0, rho0, dslope, p0_over_r0, pslope, gamma_gas, dfloor, dffloor, user_dt, amp;
Real tau_relax, rs, gmp, rad_planet, phi_planet, t0pot, omega_p, Bump_flag, A0, dwidth, rn, rand_amp, dust_dens_slope;
Real x1min, x1max, tau_damping, damping_rate;
Real radius_inner_damping, radius_outer_damping, inner_ratio_region, outer_ratio_region,
     inner_width_damping, outer_width_damping;
Real Omega0;

Real SN_const(0.0), QN_const(0.0), Psi_const(0.0);

Real initial_D2G[NDUSTFLUIDS], Stokes_number[NDUSTFLUIDS], Hratio[NDUSTFLUIDS];
Real eta_gas, beta_gas, ks_gas;
bool Damping_Flag;
//AthenaArray<Real> omega_dyn, R_func, inv_damping_tau;

// User defined time step
Real MyTimeStep(MeshBlock *pmb);
} // namespace

// User-defined boundary conditions for disk simulations
void InnerX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

void InnerX2_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);
void OuterX2_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Get parameters for gravitatonal potential of central point mass
  gm0 = pin->GetOrAddReal("problem", "GM", 0.0);
  r0  = pin->GetOrAddReal("problem", "r0", 1.0);
  amp = pin->GetReal("problem", "amp");

  user_dt = pin->GetOrAddReal("time", "user_dt", 0.0);

  // Get parameters for initial density and velocity
  rho0            = pin->GetReal("problem", "rho0");
  dslope          = pin->GetOrAddReal("problem", "dslope", 0.0);
  dust_dens_slope = pin->GetOrAddReal("problem", "dust_dens_slope", 0.0);

  // The parameters of the amplitude of random perturbation on the radial velocity
  rand_amp     = pin->GetOrAddReal("problem", "random_vel_r_amp", 0.0);
  Damping_Flag = pin->GetBoolean("problem", "Damping_Flag");

  // The parameters of damping zones
  x1min = pin->GetReal("mesh", "x1min");
  x1max = pin->GetReal("mesh", "x1max");

  //ratio of the orbital periods between the edge of the wave-killing zone and the corresponding edge of the mesh
  inner_ratio_region = pin->GetOrAddReal("problem", "inner_dampingregion_ratio", 2.5);
  outer_ratio_region = pin->GetOrAddReal("problem", "outer_dampingregion_ratio", 1.2);

  radius_inner_damping = x1min*pow(inner_ratio_region, 2./3.);
  radius_outer_damping = x1max*pow(outer_ratio_region, -2./3.);

  inner_width_damping = radius_inner_damping - x1min;
  outer_width_damping = x1max - radius_outer_damping;

  // The normalized wave damping timescale, in unit of dynamical timescale.
  damping_rate = pin->GetOrAddReal("problem", "damping_rate", 1.0);

  Omega0 = pin->GetOrAddReal("orbital_advection", "Omega0", 0.0);

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      initial_D2G[n]   = pin->GetReal("dust", "initial_D2G_" + std::to_string(n+1));
      Stokes_number[n] = pin->GetReal("dust", "Stokes_number_" + std::to_string(n+1));
      Hratio[n]        = pin->GetReal("dust", "Hratio_" + std::to_string(n+1));
    }
  }

  // Get parameters of initial pressure and cooling parameters
  if (NON_BAROTROPIC_EOS) {
    p0_over_r0 = pin->GetOrAddReal("problem", "p0_over_r0", 0.0025);
    pslope     = pin->GetReal("problem", "pslope");
    gamma_gas  = pin->GetReal("hydro", "gamma");
  } else {
    p0_over_r0 = SQR(pin->GetReal("hydro", "iso_sound_speed"));
  }

  Real float_min = std::numeric_limits<float>::min();
  dfloor         = pin->GetOrAddReal("hydro", "dfloor", (1024*(float_min)));
  dffloor        = pin->GetOrAddReal("dust",  "dffloor", (1024*(float_min)));

  eta_gas  = 0.5*p0_over_r0*(pslope + dslope);
  beta_gas = std::sqrt(1.0 + 2.0*eta_gas);
  ks_gas   = 0.5 * beta_gas;

  // Dust to gas ratio && dust stopping time
  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++) {
      SN_const += (initial_D2G[n])/(1.0 + SQR(Stokes_number[n]));
      QN_const += (initial_D2G[n]*Stokes_number[n])/(1.0 + SQR(Stokes_number[n]));
    }
    Psi_const = 1.0/((SN_const + beta_gas)*(SN_const + 2.0*ks_gas) + SQR(QN_const));
  }

  EnrollUserDustStoppingTime(MyStoppingTime);

  // Enroll damping zone and local isothermal equation of state
  EnrollUserExplicitSourceFunction(MySource);

  // Enroll user-defined boundary condition
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, InnerX1_NSH);

  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OuterX1_NSH);

  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, InnerX2_NSH);

  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user"))
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, OuterX2_NSH);

  // Enroll user-defined time step
  if (user_dt > 0.0)
    EnrollUserTimeStepFunction(MyTimeStep);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Initializes Keplerian accretion disk.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  std::int64_t iseed = -1 - gid;
  Real rad(0.0), phi(0.0), z(0.0);
  Real x1, x2, x3;

  Real igm1 = 1.0/(gamma_gas - 1.0);
  // Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    x3 = pcoord->x3v(k);
    for (int j=js; j<=je; ++j) {
      x2 = pcoord->x2v(j);
      for (int i=is; i<=ie; ++i) {
        Real gas_vel1(0.0), gas_vel2(0.0), gas_vel3(0.0);
        x1 = pcoord->x1v(i);

        Real &gas_dens = phydro->u(IDN, k, j, i);
        Real &gas_mom1 = phydro->u(IM1, k, j, i);
        Real &gas_mom2 = phydro->u(IM2, k, j, i);
        Real &gas_mom3 = phydro->u(IM3, k, j, i);

        GetCylCoord(pcoord, rad, phi, z, i, j, k);
        Real Vel_K = Keplerian_velocity(rad);

        Real delta_gas_vel1 = amp*std::sqrt(PoverRho(rad, phi, z))*(ran2(&iseed) - 0.5);
        Real delta_gas_vel2 = amp*std::sqrt(PoverRho(rad, phi, z))*(ran2(&iseed) - 0.5);
        Real delta_gas_vel3 = amp*std::sqrt(PoverRho(rad, phi, z))*(ran2(&iseed) - 0.5);

        gas_dens         = DenProfileCyl_Gas(rad, phi, z);
        Real inv_gas_den = 1.0/gas_dens;

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, gas_vel1, gas_vel2, gas_vel3);
        if (porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
          gas_vel2 -= Vel_K;
        if (porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
          gas_vel3 -= Vel_K;

        gas_mom1 = gas_dens * (gas_vel1 + delta_gas_vel1);
        gas_mom2 = gas_dens * (gas_vel2 + delta_gas_vel2);
        gas_mom3 = gas_dens * (gas_vel3 + delta_gas_vel3);

        if (NON_BAROTROPIC_EOS) {
          Real &gas_erg  = phydro->u(IEN, k, j, i);
          Real p_over_r  = PoverRho(rad, phi, z);
          gas_erg        = p_over_r * gas_dens * igm1;
          gas_erg       += 0.5 * (SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_den;
        }

        for (int n=0; n<NDUSTFLUIDS; n++) {
          Real dust_vel1(0.0), dust_vel2(0.0), dust_vel3(0.0);
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;

          Real delta_dust_vel1 = amp*std::sqrt(PoverRho(rad, phi, z))*(ran2(&iseed) - 0.5);
          Real delta_dust_vel2 = amp*std::sqrt(PoverRho(rad, phi, z))*(ran2(&iseed) - 0.5);
          Real delta_dust_vel3 = amp*std::sqrt(PoverRho(rad, phi, z))*(ran2(&iseed) - 0.5);

          Real &dust_dens = pdustfluids->df_cons(rho_id, k, j, i);
          Real &dust_mom1 = pdustfluids->df_cons(v1_id,  k, j, i);
          Real &dust_mom2 = pdustfluids->df_cons(v2_id,  k, j, i);
          Real &dust_mom3 = pdustfluids->df_cons(v3_id,  k, j, i);

          dust_dens = DenProfileCyl_Dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);

          DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z, dust_vel1, dust_vel2, dust_vel3);
          if (porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
            dust_vel2 -= Vel_K;
          if (porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
            dust_vel3 -= Vel_K;

          dust_mom1 = dust_dens * (dust_vel1 + delta_dust_vel1);
          dust_mom2 = dust_dens * (dust_vel2 + delta_dust_vel2);
          dust_mom3 = dust_dens * (dust_vel3 + delta_dust_vel3);
        }

      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
void MySource(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s)
{
  if (Damping_Flag) {
    InnerWavedamping(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
    OuterWavedamping(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);
  }

  if (NON_BAROTROPIC_EOS)
    LocalIsothermalEOS(pmb, time, dt, prim, prim_df, prim_s, bcc, cons, cons_df, cons_s);

  return;
}


void MyStoppingTime(MeshBlock *pmb, const Real time, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time) {

  Real inv_sqrt_gm0 = 1.0/std::sqrt(gm0);

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real rad, phi, z;
          Real &st_time = stopping_time(dust_id, k, j, i);

          //Constant Stokes number in disk problems
          GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
          st_time = Stokes_number[dust_id]*std::pow(rad, 1.5)*inv_sqrt_gm0;
        }
      }
    }
  }
  return;
}


Real MyTimeStep(MeshBlock *pmb)
{
  Real min_user_dt = user_dt;
  return min_user_dt;
}


// Wavedamping function
void InnerWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1           = 1.0/(gamma_gas - 1.0);
  Real inv_inner_damp = 1.0/inner_width_damping;

  AthenaArray<Real> Vel_K, omega_dyn, R_func, inv_damping_tau;
  Vel_K.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        Vel_K(i) = Keplerian_velocity(rad);

        if (rad <= radius_inner_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad*rad*rad));
          //Real omega_dyn   = std::sqrt(gm0);
          R_func(i)          = SQR((rad - radius_inner_damping)*inv_inner_damp);
          inv_damping_tau(i) = damping_rate*omega_dyn(i);

          Real gas_rho_0 = DenProfileCyl_Gas(rad, phi, z);
          Real gas_vel1_0, gas_vel2_0, gas_vel3_0;
          GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, gas_vel1_0, gas_vel2_0, gas_vel3_0);

          if (pmb->porb->orbital_advection_defined)
            gas_vel3_0 -= Vel_K(i);

          Real &gas_dens       = cons(IDN, k, j, i);
          Real &gas_mom1       = cons(IM1, k, j, i);
          Real &gas_mom2       = cons(IM2, k, j, i);
          Real &gas_mom3       = cons(IM3, k, j, i);
          Real inv_den_gas_ori = 1.0/gas_dens;

          Real gas_vel1 = gas_mom1*inv_den_gas_ori;
          Real gas_vel2 = gas_mom2*inv_den_gas_ori;
          Real gas_vel3 = gas_mom3*inv_den_gas_ori;

          Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

          gas_dens += delta_gas_dens;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;

          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;

          //if (NON_BAROTROPIC_EOS) {
            //Real gas_erg_0  = PoverRho(rad, phi, z)*gas_rho_0;
            //gas_erg_0      += 0.5*(SQR(gas_rho_0*gas_vel1_0) + SQR(gas_rho_0*gas_vel2_0) + SQR(gas_rho_0*gas_vel3_0))/gas_rho_0;
            //Real &gas_erg   = cons(IEN, k, j, i);
            //gas_erg         = gas_erg*alpha_ori + gas_erg_0*alpha_dam;
          //}
        }
      }

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
            if (rad >= x1min && rad <= radius_inner_damping) {
              //Real omega_dyn   = std::sqrt(gm0/(rad*rad*rad));
              //Real R_func      = SQR((rad - radius_inner_damping)*inv_inner_damp);
              //Real inv_damping_tau = damping_rate*omega_dyn;

              Real dust_rho_0 = DenProfileCyl_Dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real dust_vel1_0, dust_vel2_0, dust_vel3_0;
              DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z,
                dust_vel1_0, dust_vel2_0, dust_vel3_0);
              if (pmb->porb->orbital_advection_defined)
                dust_vel3_0 -= Vel_K(i);

              Real &dust_dens       = cons_df(rho_id, k, j, i);
              Real &dust_mom1       = cons_df(v1_id,  k, j, i);
              Real &dust_mom2       = cons_df(v2_id,  k, j, i);
              Real &dust_mom3       = cons_df(v3_id,  k, j, i);
              Real inv_den_dust_ori = 1.0/dust_dens;

              Real dust_vel1 = dust_mom1*inv_den_dust_ori;
              Real dust_vel2 = dust_mom2*inv_den_dust_ori;
              Real dust_vel3 = dust_mom3*inv_den_dust_ori;

              Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func(i)*inv_damping_tau(i)*dt;
              Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
              Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
              Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

              dust_dens += delta_dust_dens;
              dust_vel1 += delta_dust_vel1;
              dust_vel2 += delta_dust_vel2;
              dust_vel3 += delta_dust_vel3;

              dust_mom1 = dust_dens*dust_vel1;
              dust_mom2 = dust_dens*dust_vel2;
              dust_mom3 = dust_dens*dust_vel3;
            }
          }
        }
      }
    }
  }
  return;
}


void OuterWavedamping(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  int is  = pmb->is; int ie = pmb->ie;
  int js  = pmb->js; int je = pmb->je;
  int ks  = pmb->ks; int ke = pmb->ke;
  int nc1 = pmb->ncells1;

  Real igm1           = 1.0/(gamma_gas - 1.0);
  Real inv_outer_damp = 1.0/outer_width_damping;

  AthenaArray<Real> Vel_K, omega_dyn, R_func, inv_damping_tau;
  Vel_K.NewAthenaArray(nc1);
  omega_dyn.NewAthenaArray(nc1);
  R_func.NewAthenaArray(nc1);
  inv_damping_tau.NewAthenaArray(nc1);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);
        Vel_K(i) = Keplerian_velocity(rad);
        if (rad >= radius_outer_damping) {
          // See de Val-Borro et al. 2006 & 2007
          omega_dyn(i)       = std::sqrt(gm0/(rad*rad*rad));
          //Real omega_dyn   = std::sqrt(gm0);
          R_func(i)          = SQR((rad - radius_outer_damping)*inv_outer_damp);
          inv_damping_tau(i) = damping_rate*omega_dyn(i);

          Real gas_rho_0 = DenProfileCyl_Gas(rad, phi, z);
          Real gas_vel1_0, gas_vel2_0, gas_vel3_0;
          GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad, phi, z, gas_vel1_0, gas_vel2_0, gas_vel3_0);
          if (pmb->porb->orbital_advection_defined)
            gas_vel3_0 -= Vel_K(i);

          Real &gas_dens       = cons(IDN, k, j, i);
          Real &gas_mom1       = cons(IM1, k, j, i);
          Real &gas_mom2       = cons(IM2, k, j, i);
          Real &gas_mom3       = cons(IM3, k, j, i);
          Real inv_den_gas_ori = 1.0/gas_dens;

          Real gas_vel1 = gas_mom1*inv_den_gas_ori;
          Real gas_vel2 = gas_mom2*inv_den_gas_ori;
          Real gas_vel3 = gas_mom3*inv_den_gas_ori;

          Real delta_gas_dens = (gas_rho_0  - gas_dens)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel1 = (gas_vel1_0 - gas_vel1)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel2 = (gas_vel2_0 - gas_vel2)*R_func(i)*inv_damping_tau(i)*dt;
          Real delta_gas_vel3 = (gas_vel3_0 - gas_vel3)*R_func(i)*inv_damping_tau(i)*dt;

          gas_dens += delta_gas_dens;
          gas_vel1 += delta_gas_vel1;
          gas_vel2 += delta_gas_vel2;
          gas_vel3 += delta_gas_vel3;

          gas_mom1 = gas_dens*gas_vel1;
          gas_mom2 = gas_dens*gas_vel2;
          gas_mom3 = gas_dens*gas_vel3;
        }
      }

          //if (NON_BAROTROPIC_EOS) {
            //Real gas_erg_0  = PoverRho(rad, phi, z)*gas_rho_0;
            //gas_erg_0      += 0.5*(SQR(gas_rho_0*gas_vel1_0) + SQR(gas_rho_0*gas_vel2_0) + SQR(gas_rho_0*gas_vel3_0))/gas_rho_0;
            //Real &gas_erg   = cons(IEN, k, j, i);
            //gas_erg         = gas_erg*alpha_ori + gas_erg_0*alpha_dam;
          //}

      if (NDUSTFLUIDS > 0) {
        for (int n=0; n<NDUSTFLUIDS; ++n) {
          int dust_id = n;
          int rho_id  = 4*dust_id;
          int v1_id   = rho_id + 1;
          int v2_id   = rho_id + 2;
          int v3_id   = rho_id + 3;
#pragma omp simd
          for (int i=is; i<=ie; ++i) {
            Real rad, phi, z;
            GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

            if (rad <= x1max && rad >= radius_outer_damping) {
              Real dust_rho_0 = DenProfileCyl_Dust(rad, phi, z, initial_D2G[dust_id], Hratio[dust_id]);
              Real dust_vel1_0, dust_vel2_0, dust_vel3_0;
              DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad, phi, z,
                dust_vel1_0, dust_vel2_0, dust_vel3_0);
              if (pmb->porb->orbital_advection_defined)
                dust_vel3_0 -= Vel_K(i);

              Real &dust_dens       = cons_df(rho_id, k, j, i);
              Real &dust_mom1       = cons_df(v1_id,  k, j, i);
              Real &dust_mom2       = cons_df(v2_id,  k, j, i);
              Real &dust_mom3       = cons_df(v3_id,  k, j, i);
              Real inv_den_dust_ori = 1.0/dust_dens;

              Real dust_vel1 = dust_mom1*inv_den_dust_ori;
              Real dust_vel2 = dust_mom2*inv_den_dust_ori;
              Real dust_vel3 = dust_mom3*inv_den_dust_ori;

              Real delta_dust_dens = (dust_rho_0  - dust_dens)*R_func(i)*inv_damping_tau(i)*dt;
              Real delta_dust_vel1 = (dust_vel1_0 - dust_vel1)*R_func(i)*inv_damping_tau(i)*dt;
              Real delta_dust_vel2 = (dust_vel2_0 - dust_vel2)*R_func(i)*inv_damping_tau(i)*dt;
              Real delta_dust_vel3 = (dust_vel3_0 - dust_vel3)*R_func(i)*inv_damping_tau(i)*dt;

              dust_dens += delta_dust_dens;
              dust_vel1 += delta_dust_vel1;
              dust_vel2 += delta_dust_vel2;
              dust_vel3 += delta_dust_vel3;

              dust_mom1 = dust_dens*dust_vel1;
              dust_mom2 = dust_dens*dust_vel2;
              dust_mom3 = dust_dens*dust_vel3;
            }
          }
        }
      }
    }
  }
  return;
}


void LocalIsothermalEOS(MeshBlock *pmb, const Real time, const Real dt, const AthenaArray<Real> &prim,
    const AthenaArray<Real> &prim_df, const AthenaArray<Real> &prim_s, const AthenaArray<Real> &bcc,
    AthenaArray<Real> &cons, AthenaArray<Real> &cons_df, AthenaArray<Real> &cons_s) {

  // Local Isothermal equation of state
  int is = pmb->is; int ie = pmb->ie;
  int js = pmb->js; int je = pmb->je;
  int ks = pmb->ks; int ke = pmb->ke;

  //Real inv_gamma = 1.0/gamma_gas;
  Real igm1 = 1.0/(gamma_gas - 1.0);
  for (int k=ks; k<=ke; ++k) { // include ghost zone
    for (int j=js; j<=je; ++j) { // prim, cons
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rad, phi, z;
        GetCylCoord(pmb->pcoord, rad, phi, z, i, j, k);

        Real &gas_dens = cons(IDN, k, j, i);
        Real &gas_mom1 = cons(IM1, k, j, i);
        Real &gas_mom2 = cons(IM2, k, j, i);
        Real &gas_mom3 = cons(IM3, k, j, i);
        Real &gas_erg  = cons(IEN, k, j, i);

        Real inv_gas_den = 1.0/gas_dens;
        Real press       = PoverRho(rad, phi, z)*gas_dens;
        gas_erg          = press*igm1 + 0.5*(SQR(gas_mom1) + SQR(gas_mom2) + SQR(gas_mom3))*inv_gas_den;
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
void GetCylCoord(Coordinates *pco,Real &rad,Real &phi,Real &z,int i,int j,int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(k);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}

//----------------------------------------------------------------------------------------
Real RadialD2G(const Real rad, const Real initial_dust2gas, const Real slope)
{
  Real dust2gas = initial_dust2gas*std::pow(rad/r0, slope);
  return dust2gas;
}


Real DenProfileCyl_Gas(const Real rad, const Real phi, const Real z) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  //Real dentem = denmid*std::exp(gm0/p_over_r*(1./std::sqrt(SQR(rad))-1./rad));
  den = dentem;
  return std::max(den,dfloor);
}


Real DenProfileCyl_Dust(const Real rad, const Real phi, const Real z, const Real den_ratio, const Real H_ratio) {
  Real den;
  Real p_over_r = p0_over_r0;
  if (NON_BAROTROPIC_EOS) p_over_r = PoverRho(rad, phi, z);
  Real denmid = den_ratio*rho0*std::pow(rad/r0,dslope);
  Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
  //Real dentem = denmid*std::exp(gm0/(SQR(H_ratio)*p_over_r)*(1./std::sqrt(SQR(rad))-1./rad));
  den         = dentem;
  return std::max(den,dffloor);
}


Real PoverRho(const Real rad, const Real phi, const Real z) {
  Real poverr;
  poverr = p0_over_r0*std::pow(rad/r0, pslope);
  return poverr;
}


//----------------------------------------------------------------------------------------
Real VelProfileCyl_Gas(const Real rad, const Real phi, const Real z) {
  Real p_over_r = PoverRho(rad, phi, z);
  //Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             //- pslope*rad/std::sqrt(rad*rad+z*z);
  Real vel = (dslope+pslope)*p_over_r/(gm0/rad) + (1.0+pslope)
             - pslope*rad/std::sqrt(rad*rad);
  vel = std::sqrt(gm0/rad)*std::sqrt(vel) - rad*Omega0;
  return vel;
}


Real VelProfileCyl_Dust(const Real rad, const Real phi, const Real z) {
  //Real dis = std::sqrt(SQR(rad) + SQR(z));
  Real dis = rad;
  Real vel = std::sqrt(gm0/dis) - rad*Omega0;
  return vel;
}


void GasVelProfileCyl_NSH(const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {
  Real vel_Keplerian  = Keplerian_velocity(rad);
  Real vel            = beta_gas*vel_Keplerian;

  Real delta_gas_vr   = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = delta_gas_vr;
    v2 = vel + delta_gas_vphi;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = delta_gas_vr;
    v2 = 0.0;
    v3 = vel + delta_gas_vphi;
  }
  return;
}


void DustVelProfileCyl_NSH(const Real ts, const Real SN, const Real QN, const Real Psi,
    const Real rad, const Real phi, const Real z, Real &v1, Real &v2, Real &v3) {

  Real vel_Keplerian   = Keplerian_velocity(rad);
  Real delta_gas_vr    = Delta_gas_vr(vel_Keplerian,   SN, QN, Psi);
  Real delta_gas_vphi  = Delta_gas_vphi(vel_Keplerian, SN, QN, Psi);

  Real delta_dust_vr   = Delta_dust_vr(ts,   vel_Keplerian, delta_gas_vr, delta_gas_vphi);
  Real delta_dust_vphi = Delta_dust_vphi(ts, vel_Keplerian, delta_gas_vr, delta_gas_vphi);

  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    v1 = delta_dust_vr;
    v2 = vel_Keplerian+delta_dust_vphi;
    v3 = 0.0;
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    v1 = delta_dust_vr;
    v2 = 0.0;
    v3 = vel_Keplerian+delta_dust_vphi;
  }
  return;
}


Real Keplerian_velocity(const Real rad) {
  Real vk = std::sqrt(gm0/rad);
  return vk;
}


Real Delta_gas_vr(const Real vk, const Real SN, const Real QN, const Real Psi) {
  Real d_g_vr = -2.0*beta_gas*QN*Psi*(beta_gas - 1.0)*vk;
  return d_g_vr;
}


Real Delta_gas_vphi(const Real vk, const Real SN, const Real QN, const Real Psi) {
  Real d_g_vphi = -1.0*((SN + 2.0*ks_gas)*SN + SQR(QN))*Psi*(beta_gas - 1.0)*vk;
  return d_g_vphi;
}


Real Delta_dust_vr(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi) {
  Real d_d_vr = (2.0*ts*(beta_gas - 1.0)*vk)/(1.0+SQR(ts)) + ((d_vgr + 2.0*ts*d_vgphi)/(1.0+SQR(ts)));
  return d_d_vr;
}


Real Delta_dust_vphi(const Real ts, const Real vk, const Real d_vgr, const Real d_vgphi) {
  Real d_d_vphi = ((beta_gas - 1.0)*vk)/(1.0+SQR(ts)) + ((2.0*d_vgphi - ts*d_vgr)/(2.0+2.0*SQR(ts)));
  return d_d_vphi;
}


void Keplerian_interpolate(const Real r_active, const Real r_ghost,
    const Real vphi_active, Real &vphi_ghost) {
  vphi_ghost = vphi_active*std::sqrt(r_active/r_ghost);
  return;
}


void Density_interpolate(const Real r_active, const Real r_ghost, const Real rho_active,
    const Real slope, Real &rho_ghost) {
  rho_ghost = rho_active * std::pow(r_ghost/r_active, slope);
  return;
}


void Vr_interpolate_inner_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //vr_ghost = vr_active <= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}


void Vr_interpolate_outer_nomatter(const Real r_active, const Real r_ghost, const Real sigma_active,
    const Real sigma_ghost, const Real vr_active, Real &vr_ghost) {
  //if (sigma_active < TINY_NUMBER)
    //vr_ghost = vr_active >= 0.0 ? ((sigma_active+TINY_NUMBER)*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  //else
    //vr_ghost = vr_active >= 0.0 ? (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost) : 0.0;
  vr_ghost = (sigma_active*r_active*vr_active)/(sigma_ghost*r_ghost);
  return;
}

} // namespace


void InnerX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        Real x1 = pco->x1v(i);
        Real rad_gh, phi_gh, z_gh;
        Real vel1_gh, vel2_gh, vel3_gh;
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, il-i, j, k);

        Real Vel_K          = Keplerian_velocity(rad_gh);
        Real &gas_rho_gh    = prim(IDN, k, j, il-i);
        Real &gas_vel1_gh   = prim(IM1, k, j, il-i);
        Real &gas_vel2_gh   = prim(IM2, k, j, il-i);
        Real &gas_vel3_gh   = prim(IM3, k, j, il-i);
        gas_rho_gh          = DenProfileCyl_Gas(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          SN += (initial_D2G[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
          QN += (initial_D2G[dust_id]*Stokes_number[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, vel1_gh, vel2_gh, vel3_gh);

        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
          vel2_gh -= Vel_K;
        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
          vel3_gh -= Vel_K;
        gas_vel1_gh = vel1_gh;
        gas_vel2_gh = vel2_gh;
        gas_vel3_gh = vel3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, j, il-i);
          gas_pre_gh       = PoverRho(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real df_vel1_gh, df_vel2_gh, df_vel3_gh;
            Real &dust_rho_gh  = prim_df(rho_id, k, j, il-i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, j, il-i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, j, il-i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, j, il-i);

            dust_rho_gh = DenProfileCyl_Dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_vel1_gh, df_vel2_gh, df_vel3_gh);

            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
              df_vel2_gh -= Vel_K;
            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
              df_vel3_gh -= Vel_K;

            dust_vel1_gh = df_vel1_gh;
            dust_vel2_gh = df_vel2_gh;
            dust_vel3_gh = df_vel3_gh;
          }
        }

      }
    }
  }
  return;
}


void OuterX1_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pco->x3v(k);
    for (int j=jl; j<=ju; ++j) {
      Real x2 = pco->x2v(j);
      for (int i=1; i<=ngh; ++i) {
        Real x1 = pco->x1v(i);
        Real rad_gh, phi_gh, z_gh;
        Real vel1_gh, vel2_gh, vel3_gh;
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, iu+i, j, k);

        Real Vel_K          = Keplerian_velocity(rad_gh);
        Real &gas_rho_gh    = prim(IDN, k, j, iu+i);
        Real &gas_vel1_gh   = prim(IM1, k, j, iu+i);
        Real &gas_vel2_gh   = prim(IM2, k, j, iu+i);
        Real &gas_vel3_gh   = prim(IM3, k, j, iu+i);
        gas_rho_gh          = DenProfileCyl_Gas(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          SN += (initial_D2G[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
          QN += (initial_D2G[dust_id]*Stokes_number[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, vel1_gh, vel2_gh, vel3_gh);

        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
          vel2_gh -= Vel_K;
        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
          vel3_gh -= Vel_K;
        gas_vel1_gh = vel1_gh;
        gas_vel2_gh = vel2_gh;
        gas_vel3_gh = vel3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, j, iu+i);
          gas_pre_gh       = PoverRho(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real df_vel1_gh, df_vel2_gh, df_vel3_gh;
            Real &dust_rho_gh  = prim_df(rho_id, k, j, iu+i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, j, iu+i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, j, iu+i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, j, iu+i);

            dust_rho_gh = DenProfileCyl_Dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_vel1_gh, df_vel2_gh, df_vel3_gh);

            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
              df_vel2_gh -= Vel_K;
            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
              df_vel3_gh -= Vel_K;

            dust_vel1_gh = df_vel1_gh;
            dust_vel2_gh = df_vel2_gh;
            dust_vel3_gh = df_vel3_gh;
          }
        }

      }
    }
  }
  return;
}


void InnerX2_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pco->x3v(k);
    for (int j=1; j<=ngh; ++j) {
      Real x2 = pco->x2v(j);
      for (int i=il; i<=iu; ++i) {
        Real x1 = pco->x1v(i);
        Real rad_gh, phi_gh, z_gh;
        Real vel1_gh, vel2_gh, vel3_gh;
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, i, jl-j, k);

        Real Vel_K          = Keplerian_velocity(rad_gh);
        Real &gas_rho_gh    = prim(IDN, k, jl-j, i);
        Real &gas_vel1_gh   = prim(IM1, k, jl-j, i);
        Real &gas_vel2_gh   = prim(IM2, k, jl-j, i);
        Real &gas_vel3_gh   = prim(IM3, k, jl-j, i);
        gas_rho_gh          = DenProfileCyl_Gas(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          SN += (initial_D2G[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
          QN += (initial_D2G[dust_id]*Stokes_number[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, vel1_gh, vel2_gh, vel3_gh);

        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
          vel2_gh -= Vel_K;
        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
          vel3_gh -= Vel_K;
        gas_vel1_gh = vel1_gh;
        gas_vel2_gh = vel2_gh;
        gas_vel3_gh = vel3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, jl-j, i);
          gas_pre_gh       = PoverRho(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real df_vel1_gh, df_vel2_gh, df_vel3_gh;
            Real &dust_rho_gh  = prim_df(rho_id, k, jl-j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, jl-j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, jl-j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, jl-j, i);

            dust_rho_gh = DenProfileCyl_Dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_vel1_gh, df_vel2_gh, df_vel3_gh);

            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
              df_vel2_gh -= Vel_K;
            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
              df_vel3_gh -= Vel_K;

            dust_vel1_gh = df_vel1_gh;
            dust_vel2_gh = df_vel2_gh;
            dust_vel3_gh = df_vel3_gh;
          }
        }
      }
    }
  }
  return;
}


void OuterX2_NSH(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, AthenaArray<Real> &prim_df,
                  FaceField &b, Real time, Real dt,
                  int il, int iu, int jl, int ju, int kl, int ku, int ngh) {

  for (int k=kl; k<=ku; ++k) {
    Real x3 = pco->x3v(k);
    for (int j=1; j<=ngh; ++j) {
      Real x2 = pco->x2v(j);
        for (int i=il; i<=iu; ++i) {
        Real x1 = pco->x1v(i);
        Real rad_gh, phi_gh, z_gh;
        Real vel1_gh, vel2_gh, vel3_gh;
        GetCylCoord(pco, rad_gh, phi_gh, z_gh, i, ju+j, k);

        Real Vel_K          = Keplerian_velocity(rad_gh);
        Real &gas_rho_gh    = prim(IDN, k, ju+j, i);
        Real &gas_vel1_gh   = prim(IM1, k, ju+j, i);
        Real &gas_vel2_gh   = prim(IM2, k, ju+j, i);
        Real &gas_vel3_gh   = prim(IM3, k, ju+j, i);
        gas_rho_gh          = DenProfileCyl_Gas(rad_gh, phi_gh, z_gh);
        Real inv_gas_rho_gh = 1.0/gas_rho_gh;

        Real SN(0.0), QN(0.0), Psi(0.0);
        for (int n=0; n<NDUSTFLUIDS; n++) {
          int dust_id = n;
          SN += (initial_D2G[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
          QN += (initial_D2G[dust_id]*Stokes_number[dust_id])/(1.0 + SQR(Stokes_number[dust_id]));
        }
        Psi = 1.0/((SN + beta_gas)*(SN + 2.0*ks_gas) + SQR(QN));

        GasVelProfileCyl_NSH(SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh, vel1_gh, vel2_gh, vel3_gh);

        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
          vel2_gh -= Vel_K;
        if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
          vel3_gh -= Vel_K;
        gas_vel1_gh = vel1_gh;
        gas_vel2_gh = vel2_gh;
        gas_vel3_gh = vel3_gh;

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre_gh = prim(IPR, k, ju+j, i);
          gas_pre_gh       = PoverRho(rad_gh, phi_gh, z_gh)*gas_rho_gh;
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real df_vel1_gh, df_vel2_gh, df_vel3_gh;
            Real &dust_rho_gh  = prim_df(rho_id, k, ju+j, i);
            Real &dust_vel1_gh = prim_df(v1_id,  k, ju+j, i);
            Real &dust_vel2_gh = prim_df(v2_id,  k, ju+j, i);
            Real &dust_vel3_gh = prim_df(v3_id,  k, ju+j, i);

            dust_rho_gh = DenProfileCyl_Dust(rad_gh, phi_gh, z_gh, initial_D2G[dust_id], Hratio[dust_id]);
            DustVelProfileCyl_NSH(Stokes_number[dust_id], SN_const, QN_const, Psi_const, rad_gh, phi_gh, z_gh,
                df_vel1_gh, df_vel2_gh, df_vel3_gh);

            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0))
              df_vel2_gh -= Vel_K;
            if (pmb->porb->orbital_advection_defined && (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0))
              df_vel3_gh -= Vel_K;

            dust_vel1_gh = df_vel1_gh;
            dust_vel2_gh = df_vel2_gh;
            dust_vel3_gh = df_vel3_gh;
          }
        }

      }
    }
  }
  return;
}
