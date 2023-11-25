//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file first_order_flux_correction.cpp
//! \brief Computes divergence of the Hydro fluxes and
//! adds that to a temporary conserved variable register
//! then replace flux to the first order fluxes if it is bad

// C headers

// C++ headers
#include <algorithm>  // std::binary_search
#include <vector>     // std::vector

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../mesh/mesh.hpp"
#include "hydro.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//----------------------------------------------------------------------------------------
//! \fn  void Hydro::AddFluxDivergence
//! \brief Adds flux divergence to weighted average of conservative variables from
//! previous step(s) of time integrator algorithm

// TODO(felker): consider combining with PassiveScalars implementation + (see 57cfe28b)
// (may rename to AddPhysicalFluxDivergence or AddQuantityFluxDivergence to explicitly
// distinguish from CoordTerms)
// (may rename to AddHydroFluxDivergence and AddScalarsFluxDivergence, if
// the implementations remain completely independent / no inheritance is
// used)
void Hydro::FirstOrderFluxCorrection(Real delta, Real gam0, Real gam1, Real beta) {
  MeshBlock *pmb = pmy_block;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;
  int il, iu, jl, ju, kl, ku;
  // estimate updated conserved quantites and flag bad cells
  // assume second order integrator
  Real beta_dt = beta*pmb->pmy_mesh->dt;

  // estimate next step conserved quantities
  for (int n=0; n<NHYDRO; ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          // the integration step consists of two steps:
          // u1 = u1 + delta*u
          // u = gam0*u + gam1*u1 + beta*dt*F(u)
          // This line handles the update except term F(u) term:
          // utest = gam0*u + gam1*(u1+delta*u)
          utest_(n,k,j,i) = gam0*u(n,k,j,i) + gam1*(u1(n,k,j,i)+delta*u(n,k,j,i));
        }
      }
    }
  }

  AddFluxDivergence(beta_dt, utest_);

#if MAGNETIC_FIELDS_ENABLED

  Field *pf = pmb->pfield;
  Coordinates *pco = pmb->pcoord;
  AthenaArray<Real> &bcc_ = pf->bcc;

  AthenaArray<Real> &e3x1_ = pmb->pfield->e3_x1f, &e2x1_ = pmb->pfield->e2_x1f;
  AthenaArray<Real> &e1x2_ = pmb->pfield->e1_x2f, &e3x2_ = pmb->pfield->e3_x2f;
  AthenaArray<Real> &e1x3_ = pmb->pfield->e1_x3f, &e2x3_ = pmb->pfield->e2_x3f;
  // assuming cartesian
  Real dtodx1 = beta_dt/pco->dx1v(is);
  Real dtodx2 = beta_dt/pco->dx2v(js);
  Real dtodx3 = beta_dt/pco->dx3v(ks);
  pf->CalculateCellCenteredField(pf->b1,bcctest_,pco,is,ie,js,je,ks,ke);
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real b1old = bcctest_(IB1,k,j,i);
        Real b2old = bcctest_(IB2,k,j,i);
        Real b3old = bcctest_(IB3,k,j,i);
        bcctest_(IB1,k,j,i) = gam0*bcc_(IB1,k,j,i) + gam1*(b1old + delta*bcc_(IB1,k,j,i));
        bcctest_(IB2,k,j,i) = gam0*bcc_(IB2,k,j,i) + gam1*(b2old + delta*bcc_(IB2,k,j,i));
        bcctest_(IB3,k,j,i) = gam0*bcc_(IB3,k,j,i) + gam1*(b3old + delta*bcc_(IB3,k,j,i));

        bcctest_(IB2,k,j,i) += dtodx1*(e3x1_(k,j,i+1) - e3x1_(k,j,i));
        bcctest_(IB3,k,j,i) -= dtodx1*(e2x1_(k,j,i+1) - e2x1_(k,j,i));
        if (pmb->pmy_mesh->f2) {
          bcctest_(IB1,k,j,i) -= dtodx2*(e3x2_(k,j+1,i) - e3x2_(k,j,i));
          bcctest_(IB3,k,j,i) += dtodx2*(e1x2_(k,j+1,i) - e1x2_(k,j,i));
        }
        if (pmb->pmy_mesh->f3) {
          bcctest_(IB1,k,j,i) += dtodx3*(e2x3_(k+1,j,i) - e2x3_(k,j,i));
          bcctest_(IB2,k,j,i) -= dtodx3*(e1x3_(k+1,j,i) - e1x3_(k,j,i));
        }
      }
    }
  }
#endif

  // test only active zones
  // utest_(IEN) must be e_int + e_k excluding e_mag even if MHD
  pmb->peos->ConservedToPrimitiveTest(utest_, bcctest_, is, ie, js, je, ks, ke);

  // now replace fluxes with first-order fluxes
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (pmb->peos->fofc_(k,j,i)) {
          #if MAGNETIC_FIELDS_ENABLED
            ApplyFOFC_MHD(i,j,k);
          #else
            ApplyFOFC_Hydro(i,j,k);
          #endif
          // diffusion fluxes needs to be added
          if (!STS_ENABLED) AddDiffusionFluxesSingleCell(i,j,k);
        }
      }
    }
  }

  return;
}


#if MAGNETIC_FIELDS_ENABLED
//----------------------------------------------------------------------------------------
//! \fn void Hydro::ApplyFOFC_MHD
//! \brief ApplyFOFC for MHD
void Hydro::ApplyFOFC_MHD(int i, int j, int k) {
  MeshBlock *pmb = pmy_block;
  // apply FOFC
  Real wim1[NWAVE],wi[NWAVE],wip1[NWAVE],flx[NWAVE];
  AthenaArray<Real> &x1flux = flux[X1DIR];
  AthenaArray<Real> &x2flux = flux[X2DIR];
  AthenaArray<Real> &x3flux = flux[X3DIR];

  Real bxi;
  FaceField &b_ = pmb->pfield->b;
  AthenaArray<Real> &bcc_ = pmb->pfield->bcc;
  AthenaArray<Real> &e3x1_ = pmb->pfield->e3_x1f, &e2x1_ = pmb->pfield->e2_x1f;
  AthenaArray<Real> &e1x2_ = pmb->pfield->e1_x2f, &e3x2_ = pmb->pfield->e3_x2f;
  AthenaArray<Real> &e1x3_ = pmb->pfield->e1_x3f, &e2x3_ = pmb->pfield->e2_x3f;

  // first order i-1 state
  wim1[IDN] = w(IDN,k,j,i-1);
  wim1[IVX] = w(IVX,k,j,i-1);
  wim1[IVY] = w(IVY,k,j,i-1);
  wim1[IVZ] = w(IVZ,k,j,i-1);
  if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k,j,i-1);
  wim1[IBY] = bcc_(IB2,k,j,i-1);
  wim1[IBZ] = bcc_(IB3,k,j,i-1);

  // first order i state
  wi[IDN] = w(IDN,k,j,i);
  wi[IVX] = w(IVX,k,j,i);
  wi[IVY] = w(IVY,k,j,i);
  wi[IVZ] = w(IVZ,k,j,i);
  if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);
  wi[IBY] = bcc_(IB2,k,j,i);
  wi[IBZ] = bcc_(IB3,k,j,i);

  // first order i+1 state
  wip1[IDN] = w(IDN,k,j,i+1);
  wip1[IVX] = w(IVX,k,j,i+1);
  wip1[IVY] = w(IVY,k,j,i+1);
  wip1[IVZ] = w(IVZ,k,j,i+1);
  if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k,j,i+1);
  wip1[IBY] = bcc_(IB2,k,j,i+1);
  wip1[IBZ] = bcc_(IB3,k,j,i+1);

  // compute LLF flux at i
  bxi = b_.x1f(k,j,i);
  SingleStateLLF_MHD(wim1, wi, bxi, flx);

  // replace fluxes at i
  x1flux(IDN,k,j,i) = flx[IDN];
  x1flux(IM1,k,j,i) = flx[IVX];
  x1flux(IM2,k,j,i) = flx[IVY];
  x1flux(IM3,k,j,i) = flx[IVZ];
  if (NON_BAROTROPIC_EOS) x1flux(IEN,k,j,i) = flx[IEN];
  e3x1_(k,j,i) = -flx[IBY];
  e2x1_(k,j,i) = flx[IBZ];

  // compute LLF flux at i+1
  bxi = b_.x1f(k,j,i+1);
  SingleStateLLF_MHD(wi, wip1, bxi, flx);

  // replace fluxes at i+1
  x1flux(IDN,k,j,i+1) = flx[IDN];
  x1flux(IM1,k,j,i+1) = flx[IVX];
  x1flux(IM2,k,j,i+1) = flx[IVY];
  x1flux(IM3,k,j,i+1) = flx[IVZ];
  if (NON_BAROTROPIC_EOS) x1flux(IEN,k,j,i+1) = flx[IEN];
  e3x1_(k,j,i+1) = -flx[IBY];
  e2x1_(k,j,i+1) = flx[IBZ];

  if (pmb->pmy_mesh->f2) {
    // 2D
    // first order j-1 state
    wim1[IDN] = w(IDN,k,j-1,i);
    wim1[IVX] = w(IVY,k,j-1,i);
    wim1[IVY] = w(IVZ,k,j-1,i);
    wim1[IVZ] = w(IVX,k,j-1,i);
    if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k,j-1,i);
    wim1[IBY] = bcc_(IB3,k,j-1,i);
    wim1[IBZ] = bcc_(IB1,k,j-1,i);

    // first order j state
    wi[IDN] = w(IDN,k,j,i);
    wi[IVX] = w(IVY,k,j,i);
    wi[IVY] = w(IVZ,k,j,i);
    wi[IVZ] = w(IVX,k,j,i);
    if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);
    wi[IBY] = bcc_(IB3,k,j,i);
    wi[IBZ] = bcc_(IB1,k,j,i);

    // x2-flux at j+1
    // first order j+1 state
    wip1[IDN] = w(IDN,k,j+1,i);
    wip1[IVX] = w(IVY,k,j+1,i);
    wip1[IVY] = w(IVZ,k,j+1,i);
    wip1[IVZ] = w(IVX,k,j+1,i);
    if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k,j+1,i);
    wip1[IBY] = bcc_(IB3,k,j+1,i);
    wip1[IBZ] = bcc_(IB1,k,j+1,i);

    // compute LLF flux at j
    bxi = b_.x2f(k,j,i);
    SingleStateLLF_MHD(wim1, wi, bxi, flx);

    // replace fluxes at j
    x2flux(IDN,k,j,i) = flx[IDN];
    x2flux(IM2,k,j,i) = flx[IVX];
    x2flux(IM3,k,j,i) = flx[IVY];
    x2flux(IM1,k,j,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x2flux(IEN,k,j,i) = flx[IEN];
    e1x2_(k,j,i) = -flx[IBY];
    e3x2_(k,j,i) = flx[IBZ];

    // compute LLF flux at j+1
    bxi = b_.x2f(k,j+1,i);
    SingleStateLLF_MHD(wi, wip1, bxi, flx);

    // replace fluxes at j+1
    x2flux(IDN,k,j+1,i) = flx[IDN];
    x2flux(IM2,k,j+1,i) = flx[IVX];
    x2flux(IM3,k,j+1,i) = flx[IVY];
    x2flux(IM1,k,j+1,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x2flux(IEN,k,j+1,i) = flx[IEN];
    e1x2_(k,j+1,i) = -flx[IBY];
    e3x2_(k,j+1,i) = flx[IBZ];
  }

  if (pmb->pmy_mesh->f3) {
    // 3D
    // first order k-1 state
    wim1[IDN] = w(IDN,k-1,j,i);
    wim1[IVX] = w(IVZ,k-1,j,i);
    wim1[IVY] = w(IVX,k-1,j,i);
    wim1[IVZ] = w(IVY,k-1,j,i);
    if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k-1,j,i);
    wim1[IBY] = bcc_(IB1,k-1,j,i);
    wim1[IBZ] = bcc_(IB2,k-1,j,i);

    // first order k state
    wi[IDN] = w(IDN,k,j,i);
    wi[IVX] = w(IVZ,k,j,i);
    wi[IVY] = w(IVX,k,j,i);
    wi[IVZ] = w(IVY,k,j,i);
    if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);
    wi[IBY] = bcc_(IB1,k,j,i);
    wi[IBZ] = bcc_(IB2,k,j,i);

    // first order k+1 state
    wip1[IDN] = w(IDN,k+1,j,i);
    wip1[IVX] = w(IVZ,k+1,j,i);
    wip1[IVY] = w(IVX,k+1,j,i);
    wip1[IVZ] = w(IVY,k+1,j,i);
    if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k+1,j,i);
    wip1[IBY] = bcc_(IB1,k+1,j,i);
    wip1[IBZ] = bcc_(IB2,k+1,j,i);

    // compute LLF flux at k
    bxi = b_.x3f(k,j,i);
    SingleStateLLF_MHD(wim1, wi, bxi, flx);

    // replace fluxes at k
    x3flux(IDN,k,j,i) = flx[IDN];
    x3flux(IM3,k,j,i) = flx[IVX];
    x3flux(IM1,k,j,i) = flx[IVY];
    x3flux(IM2,k,j,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x3flux(IEN,k,j,i) = flx[IEN];
    e2x3_(k,j,i) = -flx[IBY];
    e1x3_(k,j,i) = flx[IBZ];

    // compute LLF flux at k+1
    bxi = b_.x3f(k+1,j,i);
    SingleStateLLF_MHD(wi, wip1, bxi, flx);

    // replace fluxes at k+1
    x3flux(IDN,k+1,j,i) = flx[IDN];
    x3flux(IM3,k+1,j,i) = flx[IVX];
    x3flux(IM1,k+1,j,i) = flx[IVY];
    x3flux(IM2,k+1,j,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x3flux(IEN,k+1,j,i) = flx[IEN];
    e2x3_(k+1,j,i) = -flx[IBY];
    e1x3_(k+1,j,i) = flx[IBZ];
  }
}
#else
//----------------------------------------------------------------------------------------
//! \fn void Hydro::ApplyFOFC_Hydro
//! \brief ApplyFOFC for hydro
void Hydro::ApplyFOFC_Hydro(int i, int j, int k) {
  MeshBlock *pmb = pmy_block;

  // apply FOFC
  Real wim1[NWAVE],wi[NWAVE],wip1[NWAVE],flx[NWAVE];
  AthenaArray<Real> &x1flux = flux[X1DIR];
  AthenaArray<Real> &x2flux = flux[X2DIR];
  AthenaArray<Real> &x3flux = flux[X3DIR];

  // first order i-1 state
  wim1[IDN] = w(IDN,k,j,i-1);
  wim1[IVX] = w(IVX,k,j,i-1);
  wim1[IVY] = w(IVY,k,j,i-1);
  wim1[IVZ] = w(IVZ,k,j,i-1);
  if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k,j,i-1);

  // first order i state
  wi[IDN] = w(IDN,k,j,i);
  wi[IVX] = w(IVX,k,j,i);
  wi[IVY] = w(IVY,k,j,i);
  wi[IVZ] = w(IVZ,k,j,i);
  if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);

  // first order i+1 state
  wip1[IDN] = w(IDN,k,j,i+1);
  wip1[IVX] = w(IVX,k,j,i+1);
  wip1[IVY] = w(IVY,k,j,i+1);
  wip1[IVZ] = w(IVZ,k,j,i+1);
  if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k,j,i+1);

  // compute LLF flux at i
  SingleStateLLF_Hydro(wim1, wi, flx);

  // replace fluxes at i
  x1flux(IDN,k,j,i) = flx[IDN];
  x1flux(IM1,k,j,i) = flx[IVX];
  x1flux(IM2,k,j,i) = flx[IVY];
  x1flux(IM3,k,j,i) = flx[IVZ];
  if (NON_BAROTROPIC_EOS) x1flux(IEN,k,j,i) = flx[IEN];

  // compute LLF flux at i+1
  SingleStateLLF_Hydro(wi, wip1, flx);

  // replace fluxes at i+1
  x1flux(IDN,k,j,i+1) = flx[IDN];
  x1flux(IM1,k,j,i+1) = flx[IVX];
  x1flux(IM2,k,j,i+1) = flx[IVY];
  x1flux(IM3,k,j,i+1) = flx[IVZ];
  if (NON_BAROTROPIC_EOS) x1flux(IEN,k,j,i+1) = flx[IEN];

  if (pmb->pmy_mesh->f2) {
    // 2D
    // first order j-1 state
    wim1[IDN] = w(IDN,k,j-1,i);
    wim1[IVX] = w(IVY,k,j-1,i);
    wim1[IVY] = w(IVZ,k,j-1,i);
    wim1[IVZ] = w(IVX,k,j-1,i);
    if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k,j-1,i);

    // first order j state
    wi[IDN] = w(IDN,k,j,i);
    wi[IVX] = w(IVY,k,j,i);
    wi[IVY] = w(IVZ,k,j,i);
    wi[IVZ] = w(IVX,k,j,i);
    if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);

    // x2-flux at j+1
    // first order j+1 state
    wip1[IDN] = w(IDN,k,j+1,i);
    wip1[IVX] = w(IVY,k,j+1,i);
    wip1[IVY] = w(IVZ,k,j+1,i);
    wip1[IVZ] = w(IVX,k,j+1,i);
    if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k,j+1,i);

    // compute LLF flux at j
    SingleStateLLF_Hydro(wim1, wi, flx);

    // replace fluxes at j
    x2flux(IDN,k,j,i) = flx[IDN];
    x2flux(IM2,k,j,i) = flx[IVX];
    x2flux(IM3,k,j,i) = flx[IVY];
    x2flux(IM1,k,j,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x2flux(IEN,k,j,i) = flx[IEN];

    // compute LLF flux at j+1
    SingleStateLLF_Hydro(wi, wip1, flx);

    // replace fluxes at j+1
    x2flux(IDN,k,j+1,i) = flx[IDN];
    x2flux(IM2,k,j+1,i) = flx[IVX];
    x2flux(IM3,k,j+1,i) = flx[IVY];
    x2flux(IM1,k,j+1,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x2flux(IEN,k,j+1,i) = flx[IEN];
  }

  if (pmb->pmy_mesh->f3) {
    // 3D
    // first order k-1 state
    wim1[IDN] = w(IDN,k-1,j,i);
    wim1[IVX] = w(IVZ,k-1,j,i);
    wim1[IVY] = w(IVX,k-1,j,i);
    wim1[IVZ] = w(IVY,k-1,j,i);
    if (NON_BAROTROPIC_EOS) wim1[IPR] = w(IPR,k-1,j,i);

    // first order k state
    wi[IDN] = w(IDN,k,j,i);
    wi[IVX] = w(IVZ,k,j,i);
    wi[IVY] = w(IVX,k,j,i);
    wi[IVZ] = w(IVY,k,j,i);
    if (NON_BAROTROPIC_EOS) wi[IPR] = w(IPR,k,j,i);

    // first order k+1 state
    wip1[IDN] = w(IDN,k+1,j,i);
    wip1[IVX] = w(IVZ,k+1,j,i);
    wip1[IVY] = w(IVX,k+1,j,i);
    wip1[IVZ] = w(IVY,k+1,j,i);
    if (NON_BAROTROPIC_EOS) wip1[IPR] = w(IPR,k+1,j,i);

    // compute LLF flux at k
    SingleStateLLF_Hydro(wim1, wi, flx);

    // replace fluxes at k
    x3flux(IDN,k,j,i) = flx[IDN];
    x3flux(IM3,k,j,i) = flx[IVX];
    x3flux(IM1,k,j,i) = flx[IVY];
    x3flux(IM2,k,j,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x3flux(IEN,k,j,i) = flx[IEN];

    // compute LLF flux at k+1
    SingleStateLLF_Hydro(wi, wip1, flx);

    // replace fluxes at k+1
    x3flux(IDN,k+1,j,i) = flx[IDN];
    x3flux(IM3,k+1,j,i) = flx[IVX];
    x3flux(IM1,k+1,j,i) = flx[IVY];
    x3flux(IM2,k+1,j,i) = flx[IVZ];
    if (NON_BAROTROPIC_EOS) x3flux(IEN,k+1,j,i) = flx[IEN];
  }
}
#endif
