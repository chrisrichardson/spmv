// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "jacobi.h"
#include "L2GMap.h"
#include "Matrix.h"
#include <iostream>

// Jacobi iteration
// x = x + D-1 (b - A x)

double spmv::jacobi(const spmv::Matrix<double>& A, Eigen::VectorXd& x,
                    const Eigen::VectorXd& b)
{
  A.col_map()->update(x.data());
  Eigen::VectorXd diag = A.mat().diagonal();
  Eigen::VectorXd r = b - A * x;
  x.head(b.size()) += (r.array() / diag.array()).matrix();
  return r.norm();
}
