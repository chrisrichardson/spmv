#pragma once
#include "Operator.h"
#include <cusparse_v2.h>

class OperatorCUDA : public Operator
{
  cusparseHandle_t handle;
  cusparseSpMatDescr_t spmat;
  double* value;
  int *inner, *outer;

  double *alpha, *beta;
  void* scratch;

  cusparseDnVecDescr_t vecX, vecY;
  double *xdata, *ydata;

  int nnz, rows, cols;

public:
  OperatorCUDA(Eigen::SparseMatrix<double, Eigen::RowMajor>& A);
  ~OperatorCUDA();

  Eigen::VectorXd apply(Eigen::VectorXd& psp) const;
};