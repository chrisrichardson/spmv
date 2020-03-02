// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include "L2GMap.h"
#include "cg.h"

//-----------------------------------------------------------------------------
std::tuple<Eigen::VectorXd, int> spmv::cg(
    MPI_Comm comm,
    const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
    const std::shared_ptr<const spmv::L2GMap> l2g,
    const Eigen::Ref<const Eigen::VectorXd>& b, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  int* outer = const_cast<int*>(A.outerIndexPtr());
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, A.rows(), A.cols(), outer, outer + 1,
      const_cast<int*>(A.innerIndexPtr()), const_cast<double*>(A.valuePtr()));
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  status = mkl_sparse_optimize(A_mkl);
  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  struct matrix_descr mat_desc;
  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;
#endif

  int M = A.rows();

  // Residual vector
  Eigen::VectorXd r(M);
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(l2g->local_size(true));
  Eigen::VectorXd psp(l2g->local_size(true));
  Eigen::Map<Eigen::VectorXd> p(psp.data(), M);

  // Assign to dense part of sparse vector
  x.setZero();
  r = b; // b - A * x0
  p = r;

  double rnorm = r.squaredNorm();
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG

  double rnorm_old = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // y = A.p
    l2g->update(psp.data());

#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                    psp.data(), 0.0, y.data());
#else
    y = A * psp;
#endif

    // Calculate alpha = r.r/p.y
    double pdoty = p.dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = rnorm_old / pdoty_sum;

    // Update x and r
    x.head(M) += alpha * p;
    r -= alpha * y;

    // Update rnorm
    rnorm = r.squaredNorm();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p
    p = p * beta + r;

    if (rnorm_new / rnorm0 < rtol)
      break;
  }

#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_destroy(A_mkl);
#endif

  return {x, k};
}
