#pragma once
#include <mkl.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class OperatorMKL {
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;

public:
  OperatorMKL(Eigen::SparseMatrix<double, Eigen::RowMajor>& A) {
    sparse_status_t status = mkl_sparse_d_create_csr(
        &A_mkl, SPARSE_INDEX_BASE_ZERO, A.rows(), A.cols(), A.outerIndexPtr(),
        A.outerIndexPtr() + 1, A.innerIndexPtr(), A.valuePtr());
    assert(status == SPARSE_STATUS_SUCCESS);

    status = mkl_sparse_optimize(A_mkl);
    assert(status == SPARSE_STATUS_SUCCESS);

    if (status != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Could not create MKL matrix");

    mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    mat_desc.diag = SPARSE_DIAG_NON_UNIT;
   }

   Eigen::VectorXd operator*(Eigen::VectorXd &psp) const {
    Eigen::VectorXd q(psp.size());
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                    psp.data(), 0.0, q.data());
    return q;
   }
};