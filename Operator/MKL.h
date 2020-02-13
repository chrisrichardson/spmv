#pragma once
#include "Operator.h"
#include <mkl.h>

class OperatorMKL : public Operator
{
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;

public:
  OperatorMKL(Eigen::SparseMatrix<double, Eigen::RowMajor>& A);
  Eigen::VectorXd apply(Eigen::VectorXd& psp) const;
};