//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file default_pgen.cpp
//  \brief Provides default (empty) versions of all functions in problem generator files
//  This means user does not have to implement these functions if they are not needed.
//
// The attribute "weak" is used to ensure the loader selects the user-defined version of
// functions rather than the default version given here.
//
// The attribute "alias" may be used with the "weak" functions (in non-defining
// declarations) in order to have them refer to common no-operation function definition in
// the same translation unit. Target function must be specified by mangled name unless C
// linkage is specified.
//
// This functionality is not in either the C nor the C++ standard. These GNU extensions
// are largely supported by LLVM, Intel, IBM, but may affect portability for some
// architecutres and compilers. In such cases, simply define all 6 of the below class
// functions in every pgen/*.cpp file (without any function attributes).

// C headers

// C++ headers
#include <algorithm>  // min, max
#include <cmath>      // sqrt()
#include <cstdio>     // fopen(), fprintf(), freopen()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../reconstruct/reconstruction.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

// problem parameters which are useful to make global to this file
namespace {
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real d0, p0, u0, bx0, by0, bz0, dby, dbz;
Real delta_rho_gas_real, delta_rho_gas_imag;
Real delta_vel_gas_real, delta_vel_gas_imag;
//Real delta_rho_d_1_real, delta_rho_d_1_imag;
//Real delta_vel_d_1_real, delta_vel_d_1_imag;
//Real delta_rho_d_2_real, delta_rho_d_2_imag;
//Real delta_vel_d_2_real, delta_vel_d_2_imag;
//Real delta_rho_d_3_real, delta_rho_d_3_imag;
//Real delta_vel_d_3_real, delta_vel_d_3_imag;
//Real delta_rho_d_4_real, delta_rho_d_4_imag;
//Real delta_vel_d_4_real, delta_vel_d_4_imag;
Real user_dt;
Real amp, lambda, k_par; // amplitude, Wavelength, 2*PI/wavelength
Real gam, gm1, iso_cs, vflow;
Real MyTimeStep(MeshBlock *pmb);
} // namespace

// 3x members of Mesh class:

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // read global parameters
  d0      = pin->GetOrAddReal("problem", "d0",      1.0);
  user_dt = pin->GetOrAddReal("time",    "user_dt", 1.375e-2);
  amp     = pin->GetReal("problem",      "amp");
  vflow   = pin->GetOrAddReal("problem", "vflow",   0.0);
  iso_cs  = pin->GetReal("hydro",        "iso_sound_speed");

  delta_rho_gas_real = pin->GetReal("problem", "delta_rho_gas_real");
  delta_rho_gas_imag = pin->GetReal("problem", "delta_rho_gas_imag");
  delta_vel_gas_real = pin->GetReal("problem", "delta_vel_gas_real");
  delta_vel_gas_imag = pin->GetReal("problem", "delta_vel_gas_imag");

  // initialize global variables
  if (NON_BAROTROPIC_EOS) {
    gam = pin->GetReal("hydro", "gamma");
    gm1 = (gam - 1.0);
  }

  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and exactly one
  // wavelength along each grid direction
  Real x1size = mesh_size.x1max - mesh_size.x1min;
  Real x2size = mesh_size.x2max - mesh_size.x2min;
  Real x3size = mesh_size.x3max - mesh_size.x3min;

  Real x1 = x1size;
  Real x2 = 0.0;
  Real x3 = 0.0;

  // For lambda choose the smaller of the 3
  lambda = x1;

  // Initialize k_parallel
  k_par = 2.0*(PI)/lambda;

  // Compute eigenvectors, where the quantities u0 and bx0 are parallel to the
  // wavevector, and v0,w0,by0,bz0 are perpendicular.
  //d0      = 1.0;
  //p0      = 0.0;
  u0      = vflow;
  Real v0 = 0.0;
  Real w0 = 0.0;
  Real h0 = 0.0;

  //if (NON_BAROTROPIC_EOS) {
  p0 = gam*SQR(iso_cs)*d0;
    //h0 = ((p0/gm1 + 0.5*d0*(u0*u0 + v0*v0 + w0*w0)) + p0)/d0;
  //}

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//  \brief Function called once every time step for user-defined work.
//========================================================================================

//void MeshBlock::UserWorkInLoop() {
  //// Local Isothermal equation of state
  //Real igm1 = 1.0/(gam - 1.0);

  //for (int k=ks; k<=ke; ++k) {
    //for (int j=js; j<=je; ++j) {
      //for (int i=is; i<=ie; ++i) {

        //Real &gas_rho = phydro->w(IDN, k, j, i);
        //Real &gas_v1  = phydro->w(IVX, k, j, i);
        //Real &gas_v2  = phydro->w(IVY, k, j, i);
        //Real &gas_v3  = phydro->w(IVZ, k, j, i);

        //Real &gas_den = phydro->u(IDN, k, j, i);
        //Real &gas_m1  = phydro->u(IM1, k, j, i);
        //Real &gas_m2  = phydro->u(IM2, k, j, i);
        //Real &gas_m3  = phydro->u(IM3, k, j, i);

        //if (NON_BAROTROPIC_EOS) {
          //Real &gas_pre = phydro->w(IPR, k, j, i);
          //gas_pre       = SQR(iso_cs)*gas_rho;
          //Real &gas_erg = phydro->u(IEN, k, j, i);
          //gas_erg       = gas_pre*igm1 + 0.5*(SQR(gas_m1)+SQR(gas_m2)+SQR(gas_m3))/gas_den;
        //}

      //}
    //}
  //}

  //return;
//}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Function called after main loop is finished for user-defined work.
//========================================================================================

//void __attribute__((weak)) Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  //// do nothing
  //return;
//}

// 4x members of MeshBlock class:

////========================================================================================
////! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
////  \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
////  used to initialize variables which are global to other functions in this file.
////  Called in MeshBlock constructor before ProblemGenerator.
////========================================================================================

//void __attribute__((weak)) MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  //// do nothing
  //return;
//}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Initialize the magnetic fields.  Note wavevector, eigenvectors, and other variables
  // are set in InitUserMeshData
  // initialize conserved variables
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x  = pcoord->x1v(i);
        Real sn = std::sin(k_par*x);
        Real cn = std::cos(k_par*x);

        Real &gas_den = phydro->u(IDN, k, j, i);
        Real &gas_m1  = phydro->u(IM1, k, j, i);
        Real &gas_m2  = phydro->u(IM2, k, j, i);
        Real &gas_m3  = phydro->u(IM3, k, j, i);

        Real delta_rho = amp*d0*(cn*delta_rho_gas_real     - sn*delta_rho_gas_imag);
        Real delta_vel = amp*iso_cs*(cn*delta_vel_gas_real - sn*delta_vel_gas_imag);
        gas_den        = d0 + delta_rho;
        gas_m1         = d0*(u0 + delta_vel);
        gas_m2         = 0.0;
        gas_m3         = 0.0;
        Real delta_pre = amp*gam*SQR(iso_cs)*d0*(cn*delta_rho_gas_real - sn*delta_rho_gas_imag);

        //Real &gas_pre = phydro->w(IPR, k, j, i);
        //gas_pre       = SQR(gas_den) * gas_den;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = p0/gm1 + 0.5*d0*u0*u0 + delta_pre;
          //Real e0 = p0/gm1 + 0.5*d0*u0*u0 + amp*sn*rem[4][wave_flag];
          //phydro->u(IEN,k,j,i) = (gam*SQR(iso_cs)*(d0+delta_rho))/gm1 + 0.5*(d0+delta_rho)*SQR(u0+delta_vel);
          //phydro->u(IEN,k,j,i) = (SQR(iso_cs)*(d0+delta_rho)) + 0.5*(d0+delta_rho)*SQR(u0+delta_vel);
        }

      }
    }
  }
  return;
}

void MeshBlock::UserWorkInLoop() {
  // Local Isothermal equation of state
  Real igm1 = 1.0/(gam - 1.0);

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

        Real &gas_rho = phydro->w(IDN, k, j, i);
        //Real &gas_v1  = phydro->w(IVX, k, j, i);
        //Real &gas_v2  = phydro->w(IVY, k, j, i);
        //Real &gas_v3  = phydro->w(IVZ, k, j, i);

        //Real &gas_den = phydro->u(IDN, k, j, i);
        //Real &gas_m1  = phydro->u(IM1, k, j, i);
        //Real &gas_m2  = phydro->u(IM2, k, j, i);
        //Real &gas_m3  = phydro->u(IM3, k, j, i);

        if (NON_BAROTROPIC_EOS) {
          Real &gas_pre = phydro->w(IPR, k, j, i);
          gas_pre       = gam*SQR(iso_cs)*gas_rho;
          //Real &gas_erg = phydro->u(IEN, k, j, i);
          //gas_erg       = gas_pre*igm1 + 0.5*(SQR(gas_m1)+SQR(gas_m2)+SQR(gas_m3))/gas_den;
          phydro->u(IEN,k,j,i) = gas_pre/gm1 + 0.5*d0*u0*u0;
        }

      }
    }
  }

  return;
}

