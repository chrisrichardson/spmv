// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "jacobi.h"
#include "L2GMap.h"
#include <iostream>

// Jacobi iteration
// x = x + D-1 (b - A x)

void spmv::jacobi(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
                  std::shared_ptr<const spmv::L2GMap> l2g,
                  Eigen::Ref<Eigen::VectorXd> x,
                  const Eigen::Ref<const Eigen::VectorXd>& b,
                  const Eigen::Ref<const Eigen::VectorXd>& D)
{
  l2g->update(x.data());
  Eigen::VectorXd r = b - A * x;
  x.head(b.size()) += (r.array() * D.array()).matrix();
}
