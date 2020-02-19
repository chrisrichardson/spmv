// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>

#ifdef EIGEN_USE_MKL_ALL
#include "mkl_sparsematrix.h"
#endif

namespace spmv
{

#ifdef EIGEN_USE_MKL_ALL
using SparseMatrix = MKLSparseMatrix;
#else
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
#endif

class L2GMap;

// Solve A.x=b iteratively with Conjugate Gradient
//
// Input
// @param comm MPI comm
// @param A LHS matrix
// @param l2g Local-to-global map
// @param b RHS vector
// @param max_its Maximum iteration count
// @param rtol Relative tolerance
//
// @return tuple of result and number of iterations
//
std::tuple<Eigen::VectorXd, int>
cg(MPI_Comm comm, SparseMatrix A,
   const std::shared_ptr<const L2GMap> l2g,
   const Eigen::Ref<const Eigen::VectorXd>& b, int max_its, double rtol);

} // namespace spmv
