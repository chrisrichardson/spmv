

#include "Matrix.h"
#include <iostream>

using namespace spmv;

Matrix::Matrix(Eigen::SparseMatrix<double, Eigen::RowMajor> A,
               std::shared_ptr<spmv::L2GMap> col_map)
    : _matA(A), _col_map(col_map)
{
  // _row_map = std::make_shared<L2GMap>(comm, _matA.rows(), {});

#ifdef EIGEN_USE_MKL_ALL
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, _matA.rows(), _matA.cols(),
      _matA.outerIndexPtr(), _matA.outerIndexPtr() + 1, _matA.innerIndexPtr(),
      _matA.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(A_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;
#endif
}
//-----------------------------------------------------------------------------
Eigen::VectorXd Matrix::operator*(const Eigen::VectorXd& b) const
{
#ifdef EIGEN_USE_MKL_ALL
  Eigen::VectorXd y(_matA.rows());
  mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                  b.data(), 0.0, y.data());

  return y;
#else
  return _matA * b;
#endif
}
//-----------------------------------------------------------------------------
Eigen::VectorXd Matrix::transpmult(const Eigen::VectorXd& b) const
{
#ifdef EIGEN_USE_MKL_ALL
  Eigen::VectorXd y(_matA.cols());
  mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, A_mkl, mat_desc, b.data(),
                  0.0, y.data());

  return y;
#else
  return _matA.transpose() * b;
#endif
}
//-----------------------------------------------------------------------------
