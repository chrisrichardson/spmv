#include "cg.h"
#include "L2GMap.h"
#include <iostream>

//-----------------------------------------------------------------------------
// Untested CG solver
std::tuple<Eigen::VectorXd, int>
cg(MPI_Comm comm,
   const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
   const std::shared_ptr<const L2GMap> l2g,
   const Eigen::Ref<const Eigen::VectorXd>& b)
{
  // Max iterations
  int kmax = 10000;

  // Residual tolerance
  double rtol = 1e-10;

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
  psp.head(M) = b;
  l2g->update(psp.data());
#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                  psp.data(), 0.0, y.data());
#else
  y = A * psp;
#endif
  r = b - y;

  // Assign to dense part of sparse vector
  psp.head(M) = r;

  double rnorm = r.squaredNorm();
  double rnorm_old;
  MPI_Allreduce(&rnorm, &rnorm_old, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  Eigen::VectorXd x(M);
  x.setZero();

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

    // Update x and r
    double pdoty = psp.head(M).dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    double alpha = rnorm_old / pdoty_sum;
    x += alpha * psp.head(M);
    r -= alpha * y;

    // Update p
    rnorm = r.squaredNorm();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    psp.head(M) *= beta;
    psp.head(M) += r;

    if (rnorm_new < rtol)
      return {x, k};

    rnorm_old = rnorm_new;
  }
  return {x, kmax};
}
