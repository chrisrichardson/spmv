#include "MKL.h"

#define sparse_CHECK(x)                                                        \
  if (x != SPARSE_STATUS_SUCCESS)                                              \
  throw std::runtime_error(#x " failed")

OperatorMKL::OperatorMKL(Eigen::SparseMatrix<double, Eigen::RowMajor>& A)
{
  sparse_CHECK(mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, A.rows(), A.cols(), A.outerIndexPtr(),
      A.outerIndexPtr() + 1, A.innerIndexPtr(), A.valuePtr()));

  sparse_CHECK(mkl_sparse_optimize(A_mkl));

  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;
}

Eigen::VectorXd OperatorMKL::apply(Eigen::VectorXd& psp) const
{
  Eigen::VectorXd q(psp.size());
  mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                  psp.data(), 0.0, q.data());
  return q;
}