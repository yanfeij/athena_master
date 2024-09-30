//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_wave.c
//  \brief Linear wave problem generator for 1D/2D/3D problems.
//
// In 1D, the problem is setup along one of the three coordinate axes (specified by
// setting [ang_2,ang_3] = 0.0 or PI/2 in the input file).  In 2D/3D this routine
// automatically sets the wavevector along the domain diagonal.
//========================================================================================

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
#include "../dustfluids/dustfluids.hpp"
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

namespace {
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real d0, p0, u0, bx0, by0, bz0, dby, dbz;
Real user_dt;
int wave_flag;
Real ang_2, ang_3; // Rotation angles about the y and z' axis
bool ang_2_vert, ang_3_vert; // Switches to set ang_2 and/or ang_3 to pi/2
Real sin_a2, cos_a2, sin_a3, cos_a3;
Real amp, lambda, k_par; // amplitude, Wavelength, 2*PI/wavelength
Real gam,gm1,iso_cs,vflow;
Real ev[NWAVE], rem[NWAVE][NWAVE], lem[NWAVE][NWAVE];

// functions to compute vector potential to initialize the solution
Real A1(const Real x1, const Real x2, const Real x3);
Real A2(const Real x1, const Real x2, const Real x3);
Real A3(const Real x1, const Real x2, const Real x3);

// function to compute eigenvectors of linear waves
void Eigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                 const Real h, const Real b1, const Real b2, const Real b3,
                 const Real x, const Real y, Real eigenvalues[(NWAVE)],
                 Real right_eigenmatrix[(NWAVE)][(NWAVE)],
                 Real left_eigenmatrix[(NWAVE)][(NWAVE)]);

Real MaxV2(MeshBlock *pmb, int iout);

AthenaArray<Real> initial_D2G(NDUSTFLUIDS);
Real MyTimeStep(MeshBlock *pmb);
} // namespace

// AMR refinement condition
int RefinementCondition(MeshBlock *pmb);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // read global parameters
  user_dt   = pin->GetOrAddReal("problem", "user_dt", 1e-1);
  wave_flag = pin->GetInteger("problem", "wave_flag");
  amp = pin->GetReal("problem", "amp");
  vflow = pin->GetOrAddReal("problem", "vflow", 0.0);
  ang_2 = pin->GetOrAddReal("problem", "ang_2", -999.9);
  ang_3 = pin->GetOrAddReal("problem", "ang_3", -999.9);

  ang_2_vert = pin->GetOrAddBoolean("problem", "ang_2_vert", false);
  ang_3_vert = pin->GetOrAddBoolean("problem", "ang_3_vert", false);
  iso_cs = pin->GetReal("hydro", "iso_sound_speed");

  if (NDUSTFLUIDS > 0) {
    for (int n=0; n<NDUSTFLUIDS; n++)
      initial_D2G(n) = pin->GetOrAddReal("dust", "Initial_D2G_" + std::to_string(n+1), 0.01);
  }

  // initialize global variables
  if (NON_BAROTROPIC_EOS) {
    gam   = pin->GetReal("hydro", "gamma");
    gm1 = (gam - 1.0);
  } else {
    iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  }

  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and exactly one
  // wavelength along each grid direction
  Real x1size = mesh_size.x1max - mesh_size.x1min;
  Real x2size = mesh_size.x2max - mesh_size.x2min;
  Real x3size = mesh_size.x3max - mesh_size.x3min;

  // User should never input -999.9 in angles
  if (ang_3 == -999.9) ang_3 = std::atan(x1size/x2size);
  sin_a3 = std::sin(ang_3);
  cos_a3 = std::cos(ang_3);

  // Override ang_3 input and hardcode vertical (along x2 axis) wavevector
  if (ang_3_vert) {
    sin_a3 = 1.0;
    cos_a3 = 0.0;
    ang_3 = 0.5*M_PI;
  }

  if (ang_2 == -999.9) ang_2 = std::atan(0.5*(x1size*cos_a3 + x2size*sin_a3)/x3size);
  sin_a2 = std::sin(ang_2);
  cos_a2 = std::cos(ang_2);

  // Override ang_2 input and hardcode vertical (along x3 axis) wavevector
  if (ang_2_vert) {
    sin_a2 = 1.0;
    cos_a2 = 0.0;
    ang_2 = 0.5*M_PI;
  }

  Real x1 = x1size*cos_a2*cos_a3;
  Real x2 = x2size*cos_a2*sin_a3;
  Real x3 = x3size*sin_a2;

  // For lambda choose the smaller of the 3
  lambda = x1;
  if (f2 && ang_3 != 0.0) lambda = std::min(lambda,x2);
  if (f3 && ang_2 != 0.0) lambda = std::min(lambda,x3);

  // If cos_a2 or cos_a3 = 0, need to override lambda
  if (ang_3_vert)
    lambda = x2;
  if (ang_2_vert)
    lambda = x3;

  // Initialize k_parallel
  k_par = 2.0*(PI)/lambda;

  // Compute eigenvectors, where the quantities u0 and bx0 are parallel to the
  // wavevector, and v0,w0,by0,bz0 are perpendicular.
  d0 = 1.0;
  p0 = 0.0;
  u0 = vflow;
  Real v0 = 0.0;
  Real w0 = 0.0;
  bx0 = 1.0;
  by0 = std::sqrt(2.0);
  bz0 = 0.5;
  Real xfact = 0.0;
  Real yfact = 1.0;
  Real h0 = 0.0;

  if (NON_BAROTROPIC_EOS) {
    p0 = 1.0/gam;
    h0 = ((p0/gm1 + 0.5*d0*(u0*u0 + v0*v0+w0*w0)) + p0)/d0;
  }

  Eigensystem(d0, u0, v0, w0, h0, bx0, by0, bz0, xfact, yfact, ev, rem, lem);

  if (pin->GetOrAddBoolean("problem", "test", false) && ncycle==0) {
    // reinterpret tlim as the number of orbital periods
    Real ntlim = lambda/std::abs(ev[wave_flag])*tlim;
    tlim = ntlim;
    pin->SetReal("time", "tlim", ntlim);
  }

  if (adaptive)
    EnrollUserRefinementCondition(RefinementCondition);

  // primarily used for tests of decaying linear waves (might conditionally enroll):
  AllocateUserHistoryOutput(1);
  EnrollUserHistoryOutput(0, MaxV2, "max-v2", UserHistoryOperation::max);
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Compute L1 error in linear waves and output to file
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  if (!pin->GetOrAddBoolean("problem", "compute_error", false)) return;

  // Initialize errors to zero
  Real l1_err[NHYDRO+NFIELD]{}, max_err[NHYDRO+NFIELD]{};

  MeshBlock *pmb = pblock;
  while (pmb != nullptr) {
    BoundaryValues *pbval = pmb->pbval;
    int il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je,
        kl = pmb->ks, ku = pmb->ke;
    // adjust loop limits for fourth order error calculation
    //------------------------------------------------
    if (pmb->precon->correct_err) {
      // Expand loop limits on all sides by one
      if (pbval->nblevel[1][1][0] != -1) il -= 1;
      if (pbval->nblevel[1][1][2] != -1) iu += 1;
      if (pbval->nblevel[1][0][1] != -1) jl -= 1;
      if (pbval->nblevel[1][2][1] != -1) ju += 1;
      if (pbval->nblevel[0][1][1] != -1) kl -= 1;
      if (pbval->nblevel[2][1][1] != -1) ku += 1;
    }
    // Save analytic solution of conserved variables in 4D scratch array
    AthenaArray<Real> cons_;
    // Even for MHD, there are only cell-centered mesh variables
    int ncells4 = NHYDRO + NFIELD;
    int nl = 0;
    int nu = ncells4 - 1;
    cons_.NewAthenaArray(ncells4, pmb->ncells3, pmb->ncells2, pmb->ncells1);

    //  Compute errors at cell centers
    for (int k=kl; k<=ku; k++) {
      for (int j=jl; j<=ju; j++) {
        for (int i=il; i<=iu; i++) {
          Real x = cos_a2*(pmb->pcoord->x1v(i)*cos_a3 + pmb->pcoord->x2v(j)*sin_a3)
                   + pmb->pcoord->x3v(k)*sin_a2;
          Real sn = std::sin(k_par*x);

          Real d1 = d0 + amp*sn*rem[0][wave_flag];
          Real mx = d0*vflow + amp*sn*rem[1][wave_flag];
          Real my = amp*sn*rem[2][wave_flag];
          Real mz = amp*sn*rem[3][wave_flag];
          Real m1 = mx*cos_a2*cos_a3 - my*sin_a3 - mz*sin_a2*cos_a3;
          Real m2 = mx*cos_a2*sin_a3 + my*cos_a3 - mz*sin_a2*sin_a3;
          Real m3 = mx*sin_a2                    + mz*cos_a2;

          // Store analytic solution at cell-centers
          cons_(IDN,k,j,i) = d1;
          cons_(IM1,k,j,i) = m1;
          cons_(IM2,k,j,i) = m2;
          cons_(IM3,k,j,i) = m3;

          if (NON_BAROTROPIC_EOS) {
            Real e0 = p0/gm1 + 0.5*d0*u0*u0 + amp*sn*rem[4][wave_flag];
            cons_(IEN,k,j,i) = e0;
          }
        }
      }
    }
    // begin fourth-order error correction
    // -------------------------------
    if (pmb->precon->correct_err) {
      // Restore loop limits to real cells only
      il = pmb->is, iu = pmb->ie, jl = pmb->js, ju = pmb->je, kl = pmb->ks, ku = pmb->ke;

      // Compute and store Laplacian of cell-centered conserved variables, Hydro and Bcc
      AthenaArray<Real> delta_cons_;
      delta_cons_.NewAthenaArray(ncells4, pmb->ncells3, pmb->ncells2, pmb->ncells1);
      pmb->pcoord->Laplacian(cons_, delta_cons_, il, iu, jl, ju, kl, ku, nl, nu);

      // TODO(felker): assuming uniform mesh with dx1f=dx2f=dx3f, so this factors out
      // TODO(felker): also, this may need to be dx1v, since Laplacian is cell-centered
      Real h = pmb->pcoord->dx1f(il);  // pco->dx1f(i); inside loop
      Real C = (h*h)/24.0;

      // Compute fourth-order approximation to cell-averaged conserved variables
      for (int n=nl; n<=nu; ++n) {
        for (int k=kl; k<=ku; ++k) {
          for (int j=jl; j<=ju; ++j) {
            for (int i=il; i<=iu; ++i) {
              cons_(n,k,j,i) = cons_(n,k,j,i) + C*delta_cons_(n,k,j,i);
            }
          }
        }
      }
    } // end if (pmb->precon->correct_err)
    // ------- end fourth-order error calculation

    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          // Load cell-averaged <U>, either midpoint approx. or fourth-order approx
          Real d1 = cons_(IDN,k,j,i);
          Real m1 = cons_(IM1,k,j,i);
          Real m2 = cons_(IM2,k,j,i);
          Real m3 = cons_(IM3,k,j,i);
          // Weight l1 error by cell volume
          Real vol = pmb->pcoord->GetCellVolume(k, j, i);

          l1_err[IDN] += std::abs(d1 - pmb->phydro->u(IDN,k,j,i))*vol;
          max_err[IDN] = std::max(
              static_cast<Real>(std::abs(d1 - pmb->phydro->u(IDN,k,j,i))),
              max_err[IDN]);
          l1_err[IM1] += std::abs(m1 - pmb->phydro->u(IM1,k,j,i))*vol;
          l1_err[IM2] += std::abs(m2 - pmb->phydro->u(IM2,k,j,i))*vol;
          l1_err[IM3] += std::abs(m3 - pmb->phydro->u(IM3,k,j,i))*vol;
          max_err[IM1] = std::max(
              static_cast<Real>(std::abs(m1 - pmb->phydro->u(IM1,k,j,i))),
              max_err[IM1]);
          max_err[IM2] = std::max(
              static_cast<Real>(std::abs(m2 - pmb->phydro->u(IM2,k,j,i))),
              max_err[IM2]);
          max_err[IM3] = std::max(
              static_cast<Real>(std::abs(m3 - pmb->phydro->u(IM3,k,j,i))),
              max_err[IM3]);

          if (NON_BAROTROPIC_EOS) {
            Real e0 = cons_(IEN,k,j,i);
            l1_err[IEN] += std::abs(e0 - pmb->phydro->u(IEN,k,j,i))*vol;
            max_err[IEN] = std::max(
                static_cast<Real>(std::abs(e0-pmb->phydro->u(IEN,k,j,i))),
                max_err[IEN]);
          }

        }
      }
    }
    pmb = pmb->next;
  }
  Real rms_err = 0.0, max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, (NHYDRO+NFIELD), MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, (NHYDRO+NFIELD), MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, (NHYDRO+NFIELD), MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, (NHYDRO+NFIELD), MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    // normalize errors by number of cells
    Real vol= (mesh_size.x1max - mesh_size.x1min)*(mesh_size.x2max - mesh_size.x2min)
              *(mesh_size.x3max - mesh_size.x3min);
    for (int i=0; i<(NHYDRO+NFIELD); ++i) l1_err[i] = l1_err[i]/vol;
    // compute rms error
    for (int i=0; i<(NHYDRO+NFIELD); ++i) {
      rms_err += SQR(l1_err[i]);
      max_max_over_l1 = std::max(max_max_over_l1, (max_err[i]/l1_err[i]));
    }
    rms_err = std::sqrt(rms_err);

    // open output file and write out errors
    std::string fname;
    fname.assign("linearwave-errors.dat");
    std::stringstream msg;
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop"
            << std::endl << "Error output file could not be opened" <<std::endl;
        ATHENA_ERROR(msg);
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop"
            << std::endl << "Error output file could not be opened" <<std::endl;
        ATHENA_ERROR(msg);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
      std::fprintf(pfile, "RMS-L1-Error  d_L1  M1_L1  M2_L1  M3_L1  E_L1 ");
      std::fprintf(pfile, "  Largest-Max/L1  d_max  M1_max  M2_max  M3_max  E_max ");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx1, mesh_size.nx2);
    std::fprintf(pfile, "  %d  %d", mesh_size.nx3, ncycle);
    std::fprintf(pfile, "  %e  %e", rms_err, l1_err[IDN]);
    std::fprintf(pfile, "  %e  %e  %e", l1_err[IM1], l1_err[IM2], l1_err[IM3]);
    if (NON_BAROTROPIC_EOS)
      std::fprintf(pfile, "  %e", l1_err[IEN]);
    std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err[IDN]);
    std::fprintf(pfile, "%e  %e  %e", max_err[IM1], max_err[IM2], max_err[IM3]);
    if (NON_BAROTROPIC_EOS)
      std::fprintf(pfile, "  %e", max_err[IEN]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Linear wave problem generator for 1D/2D/3D problems.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Initialize the magnetic fields.  Note wavevector, eigenvectors, and other variables
  // are set in InitUserMeshData
  // initialize conserved variables
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        Real x = cos_a2*(pcoord->x1v(i)*cos_a3 + pcoord->x2v(j)*sin_a3) +
                 pcoord->x3v(k)*sin_a2;
        Real sn = std::sin(k_par*x);
        phydro->u(IDN,k,j,i) = d0 + amp*sn*rem[0][wave_flag];
        Real mx = d0*vflow + amp*sn*rem[1][wave_flag];
        Real my = amp*sn*rem[2][wave_flag];
        Real mz = amp*sn*rem[3][wave_flag];

        phydro->u(IM1,k,j,i) = mx*cos_a2*cos_a3 - my*sin_a3 - mz*sin_a2*cos_a3;
        phydro->u(IM2,k,j,i) = mx*cos_a2*sin_a3 + my*cos_a3 - mz*sin_a2*sin_a3;
        phydro->u(IM3,k,j,i) = mx*sin_a2                    + mz*cos_a2;

        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = p0/gm1 + 0.5*d0*u0*u0 + amp*sn*rem[4][wave_flag];
        }

        if (NDUSTFLUIDS > 0) {
          for (int n = 0; n<NDUSTFLUIDS; n++) {
            int dust_id = n;
            int rho_id  = 4*dust_id;
            int v1_id   = rho_id + 1;
            int v2_id   = rho_id + 2;
            int v3_id   = rho_id + 3;

            Real &dust_den = pdustfluids->df_cons(rho_id, k, j, i);
            Real &dust_m1  = pdustfluids->df_cons(v1_id,  k, j, i);
            Real &dust_m2  = pdustfluids->df_cons(v2_id,  k, j, i);
            Real &dust_m3  = pdustfluids->df_cons(v3_id,  k, j, i);

            dust_den = initial_D2G(dust_id) * phydro->u(IDN, k, j, i);
            dust_m1  = initial_D2G(dust_id) * phydro->u(IM1, k, j, i);
            dust_m2  = initial_D2G(dust_id) * phydro->u(IM1, k, j, i);
            dust_m3  = initial_D2G(dust_id) * phydro->u(IM1, k, j, i);

            //dust_m1  = 0.0;
            //dust_m2  = 0.0;
            //dust_m3  = 0.0;
          }
        }

      }
    }
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//  \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//  Az are functions of x and y alone.

Real A1(const Real x1, const Real x2, const Real x3) {
  Real x =  x1*cos_a2*cos_a3 + x2*cos_a2*sin_a3 + x3*sin_a2;
  Real y = -x1*sin_a3        + x2*cos_a3;
  Real Ay =  bz0*x - (dbz/k_par)*std::cos(k_par*(x));
  Real Az = -by0*x + (dby/k_par)*std::cos(k_par*(x)) + bx0*y;

  return -Ay*sin_a3 - Az*sin_a2*cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//  \brief A2: 2-component of vector potential

Real A2(const Real x1, const Real x2, const Real x3) {
  Real x =  x1*cos_a2*cos_a3 + x2*cos_a2*sin_a3 + x3*sin_a2;
  Real y = -x1*sin_a3        + x2*cos_a3;
  Real Ay =  bz0*x - (dbz/k_par)*std::cos(k_par*(x));
  Real Az = -by0*x + (dby/k_par)*std::cos(k_par*(x)) + bx0*y;

  return Ay*cos_a3 - Az*sin_a2*sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential

Real A3(const Real x1, const Real x2, const Real x3) {
  Real x =  x1*cos_a2*cos_a3 + x2*cos_a2*sin_a3 + x3*sin_a2;
  Real y = -x1*sin_a3        + x2*cos_a3;
  Real Az = -by0*x + (dby/k_par)*std::cos(k_par*(x)) + bx0*y;

  return Az*cos_a2;
}

//----------------------------------------------------------------------------------------
//! \fn void Eigensystem()
//  \brief computes eigenvectors of linear waves

void Eigensystem(const Real d, const Real v1, const Real v2, const Real v3,
                 const Real h, const Real b1, const Real b2, const Real b3,
                 const Real x, const Real y, Real eigenvalues[(NWAVE)],
                 Real right_eigenmatrix[(NWAVE)][(NWAVE)],
                 Real left_eigenmatrix[(NWAVE)][(NWAVE)]) {
    //--- Adiabatic Hydrodynamics ---
    if (NON_BAROTROPIC_EOS) {
      Real vsq = v1*v1 + v2*v2 + v3*v3;
      Real asq = gm1*std::max((h-0.5*vsq), TINY_NUMBER);
      Real a = std::sqrt(asq);

      // Compute eigenvalues (eq. B2)
      eigenvalues[0] = v1 - a;
      eigenvalues[1] = v1;
      eigenvalues[2] = v1;
      eigenvalues[3] = v1;
      eigenvalues[4] = v1 + a;

      // Right-eigenvectors, stored as COLUMNS (eq. B3)
      right_eigenmatrix[0][0] = 1.0;
      right_eigenmatrix[1][0] = v1 - a;
      right_eigenmatrix[2][0] = v2;
      right_eigenmatrix[3][0] = v3;
      right_eigenmatrix[4][0] = h - v1*a;

      right_eigenmatrix[0][1] = 0.0;
      right_eigenmatrix[1][1] = 0.0;
      right_eigenmatrix[2][1] = 1.0;
      right_eigenmatrix[3][1] = 0.0;
      right_eigenmatrix[4][1] = v2;

      right_eigenmatrix[0][2] = 0.0;
      right_eigenmatrix[1][2] = 0.0;
      right_eigenmatrix[2][2] = 0.0;
      right_eigenmatrix[3][2] = 1.0;
      right_eigenmatrix[4][2] = v3;

      right_eigenmatrix[0][3] = 1.0;
      right_eigenmatrix[1][3] = v1;
      right_eigenmatrix[2][3] = v2;
      right_eigenmatrix[3][3] = v3;
      right_eigenmatrix[4][3] = 0.5*vsq;

      right_eigenmatrix[0][4] = 1.0;
      right_eigenmatrix[1][4] = v1 + a;
      right_eigenmatrix[2][4] = v2;
      right_eigenmatrix[3][4] = v3;
      right_eigenmatrix[4][4] = h + v1*a;

      // Left-eigenvectors, stored as ROWS (eq. B4)
      Real na = 0.5/asq;
      left_eigenmatrix[0][0] = na*(0.5*gm1*vsq + v1*a);
      left_eigenmatrix[0][1] = -na*(gm1*v1 + a);
      left_eigenmatrix[0][2] = -na*gm1*v2;
      left_eigenmatrix[0][3] = -na*gm1*v3;
      left_eigenmatrix[0][4] = na*gm1;

      left_eigenmatrix[1][0] = -v2;
      left_eigenmatrix[1][1] = 0.0;
      left_eigenmatrix[1][2] = 1.0;
      left_eigenmatrix[1][3] = 0.0;
      left_eigenmatrix[1][4] = 0.0;

      left_eigenmatrix[2][0] = -v3;
      left_eigenmatrix[2][1] = 0.0;
      left_eigenmatrix[2][2] = 0.0;
      left_eigenmatrix[2][3] = 1.0;
      left_eigenmatrix[2][4] = 0.0;

      Real qa = gm1/asq;
      left_eigenmatrix[3][0] = 1.0 - na*gm1*vsq;
      left_eigenmatrix[3][1] = qa*v1;
      left_eigenmatrix[3][2] = qa*v2;
      left_eigenmatrix[3][3] = qa*v3;
      left_eigenmatrix[3][4] = -qa;

      left_eigenmatrix[4][0] = na*(0.5*gm1*vsq - v1*a);
      left_eigenmatrix[4][1] = -na*(gm1*v1 - a);
      left_eigenmatrix[4][2] = left_eigenmatrix[0][2];
      left_eigenmatrix[4][3] = left_eigenmatrix[0][3];
      left_eigenmatrix[4][4] = left_eigenmatrix[0][4];

      //--- Isothermal Hydrodynamics ---

    } else {
      // Compute eigenvalues (eq. B6)
      eigenvalues[0] = v1 - iso_cs;
      eigenvalues[1] = v1;
      eigenvalues[2] = v1;
      eigenvalues[3] = v1 + iso_cs;

      // Right-eigenvectors, stored as COLUMNS (eq. B3)
      right_eigenmatrix[0][0] = 1.0;
      right_eigenmatrix[1][0] = v1 - iso_cs;
      right_eigenmatrix[2][0] = v2;
      right_eigenmatrix[3][0] = v3;

      right_eigenmatrix[0][1] = 0.0;
      right_eigenmatrix[1][1] = 0.0;
      right_eigenmatrix[2][1] = 1.0;
      right_eigenmatrix[3][1] = 0.0;

      right_eigenmatrix[0][2] = 0.0;
      right_eigenmatrix[1][2] = 0.0;
      right_eigenmatrix[2][2] = 0.0;
      right_eigenmatrix[3][2] = 1.0;

      right_eigenmatrix[0][3] = 1.0;
      right_eigenmatrix[1][3] = v1 + iso_cs;
      right_eigenmatrix[2][3] = v2;
      right_eigenmatrix[3][3] = v3;

      // Left-eigenvectors, stored as ROWS (eq. B7)

      left_eigenmatrix[0][0] = 0.5*(1.0 + v1/iso_cs);
      left_eigenmatrix[0][1] = -0.5/iso_cs;
      left_eigenmatrix[0][2] = 0.0;
      left_eigenmatrix[0][3] = 0.0;

      left_eigenmatrix[1][0] = -v2;
      left_eigenmatrix[1][1] = 0.0;
      left_eigenmatrix[1][2] = 1.0;
      left_eigenmatrix[1][3] = 0.0;

      left_eigenmatrix[2][0] = -v3;
      left_eigenmatrix[2][1] = 0.0;
      left_eigenmatrix[2][2] = 0.0;
      left_eigenmatrix[2][3] = 1.0;

      left_eigenmatrix[3][0] = 0.5*(1.0 - v1/iso_cs);
      left_eigenmatrix[3][1] = 0.5/iso_cs;
      left_eigenmatrix[3][2] = 0.0;
      left_eigenmatrix[3][3] = 0.0;
    }
}

Real MaxV2(MeshBlock *pmb, int iout) {
  Real max_v2 = 0.0;
  int is = pmb->is, ie = pmb->ie, js = pmb->js, je = pmb->je, ks = pmb->ks, ke = pmb->ke;
  AthenaArray<Real> &w = pmb->phydro->w;
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        max_v2 = std::max(std::abs(w(IVY,k,j,i)), max_v2);
      }
    }
  }
  return max_v2;
}
} // namespace

// refinement condition: density curvature
int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  Real dmax = 0.0, dmin = 2.0*d0;  // max and min densities
  for (int k=pmb->ks; k<=pmb->ke; k++) {
    for (int j=pmb->js; j<=pmb->je; j++) {
      for (int i=pmb->is; i<=pmb->ie; i++) {
        if (w(IDN,k,j,i) > dmax) dmax = w(IDN,k,j,i);
        if (w(IDN,k,j,i) < dmin) dmin = w(IDN,k,j,i);
      }
    }
  }
  // refine : delta rho > 0.9*amp
  if (dmax-d0 > 0.9*amp*rem[0][wave_flag]) return 1;
  //  Real a=std::max(dmax-d0,d0-dmin);
  //  if (a > 0.9*amp*rem[0][wave_flag]) return 1;
  // derefinement: else
  return -1;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  AllocateUserOutputVariables(1);
  return;
}

void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++)
        user_out_var(0,k,j,i) = phydro->w(IDN,k,j,i)-d0;
    }
  }
  return;
}

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
