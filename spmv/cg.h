// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#pragma once
#include <memory>
#include <mpi.h>

namespace spmv
{

template <typename T>
class Matrix;

template <typename T>
class Vector;

/// @brief Solve **A.x=b** iteratively with Conjugate Gradient
///
/// Input
/// @param comm MPI communicator
/// @param A LHS matrix
/// @param b RHS vector
/// @param max_its Maximum iteration count
/// @param rtol Relative tolerance
///
/// @return tuple of result **x** and number of iterations
///
std::tuple<Vector<double>, int> cg(MPI_Comm comm, const Matrix<double>& A,
                                   Vector<double>& b, int max_its, double rtol);

} // namespace spmv
