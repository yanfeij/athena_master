#ifndef BUFFER_UTILS_HPP
#define BUFFER_UTILS_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file buffer_utils.hpp
//  \brief prototypes of utility functions to pack/unpack buffers
//======================================================================================
#include "../athena.hpp"
#include "../athena_arrays.hpp"

namespace BufferUtility
{
int Pack4DData(AthenaArray<Real> &src, Real *buf, int sn, int en,
               int si, int ei, int sj, int ej, int sk, int ek);
void Unpack4DData(Real *buf, AthenaArray<Real> &dst, int sn, int en,
                  int si, int ei, int sj, int ej, int sk, int ek);
int Pack3DData(AthenaArray<Real> &src, Real *buf,
               int si, int ei, int sj, int ej, int sk, int ek);
void Unpack3DData(Real *buf, AthenaArray<Real> &dst,
                  int si, int ei, int sj, int ej, int sk, int ek);
}
#endif // BUFFER_UTILS_HPP