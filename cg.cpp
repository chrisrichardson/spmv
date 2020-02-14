#include "cg.h"
#include "L2GMap.h"
#include <iostream>

//-----------------------------------------------------------------------------
// Untested CG solver
std::tuple<Eigen::VectorXd, int>
cg(MPI_Comm comm,
   Eigen::Ref<Eigen::SparseMatrix<double, Eigen::RowMajor>> A,
   const std::shared_ptr<const L2GMap> l2g,
   const Eigen::Ref<const Eigen::VectorXd>& b,
   int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, A.rows(), A.cols(), A.outerIndexPtr(),
      A.outerIndexPtr() + 1, A.innerIndexPtr(), A.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(A_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

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
  Eigen::VectorXd psp(l2g->local_size());
  Eigen::Map<Eigen::VectorXd> p(psp.data(), M);

  p = b;
  l2g->update(psp.data());
#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                  psp.data(), 0.0, y.data());
#else
  y = A * psp;
#endif
  r = b - y;

  // Assign to dense part of sparse vector
  p = r;

  double rnorm = r.squaredNorm();
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  Eigen::VectorXd x(M);
  x.setZero();

  double rnorm_old = rnorm0;
  for (int k = 0; k < kmax; ++k)
  {
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
    x += alpha * p;
    r -= alpha * y;

    // Update rnorm
    rnorm = r.squaredNorm();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p
    p *= beta;
    p += r;

    if (rnorm_new/rnorm0 < rtol)
      return {x, k};
  }
  return {x, kmax};
}
