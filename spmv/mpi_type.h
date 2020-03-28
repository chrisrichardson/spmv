// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <mpi.h>

#pragma once

namespace spmv
{
template <typename T>
inline MPI_Datatype mpi_type();

template <>
inline MPI_Datatype mpi_type<float>()
{
  return MPI_FLOAT;
}
template <>
inline MPI_Datatype mpi_type<std::complex<float>>()
{
  return MPI_C_FLOAT_COMPLEX;
}
template <>
inline MPI_Datatype mpi_type<double>()
{
  return MPI_DOUBLE;
}
template <>
inline MPI_Datatype mpi_type<std::complex<double>>()
{
  return MPI_DOUBLE_COMPLEX;
}
} 