#pragma once
#include "Operator.h"
#include <cusparse_v2.h>

class OperatorCUDA : public Operator {
  cusparseHandle_t handle;
  cusparseSpMatDescr_t spmat;
  double *alpha, *beta;
  void *scratch;

  cusparseDnVecDescr_t vecX, vecY;
  double *xdata, *ydata;

public:
  OperatorCUDA(Eigen::SparseMatrix<double, Eigen::RowMajor>& A);
  ~OperatorCUDA();

  Eigen::VectorXd apply(Eigen::VectorXd &psp) const;
};