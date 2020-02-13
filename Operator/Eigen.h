#pragma once
#include "Operator.h"

class OperatorEigen : public Operator
{
  Eigen::SparseMatrix<double, Eigen::RowMajor> A;

public:
  OperatorEigen(Eigen::SparseMatrix<double, Eigen::RowMajor>& _A) : A(_A) {}

  Eigen::VectorXd apply(Eigen::VectorXd& psp) const { return A * psp; }
};