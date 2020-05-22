// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#include "cg.h"
#include "L2GMap.h"
#include "Matrix.h"
#include <iomanip>
#include <iostream>

#include <numeric>
#include <algorithm>
#include <functional>

template<typename X>
auto squaredNorm(X x) -> typename X::value_type {
  return std::transform_reduce(x.data(), x.data() + x.size(), x.data(), typename X::value_type());
}

//-----------------------------------------------------------------------------
std::tuple<Eigen::VectorXd, int>
spmv::cg(MPI_Comm comm, const spmv::Matrix<double>& A,
         const Eigen::Ref<const Eigen::VectorXd>& b, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  std::shared_ptr<const spmv::L2GMap> col_l2g = A.col_map();
  std::shared_ptr<const spmv::L2GMap> row_l2g = A.row_map();

  // Check the row map is unghosted
  if (row_l2g->local_size(true) != row_l2g->local_size(false))
    throw std::runtime_error("spmv::cg - Error: A.row_map() has ghost entries");

  int M = row_l2g->local_size(false);

  if (b.rows() != M)
    throw std::runtime_error("spmv::cg - Error: b.rows() != A.rows()");

  // Residual vector
  Eigen::VectorXd r(M);
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(col_l2g->local_size(true));
  Eigen::VectorXd p(col_l2g->local_size(true));
  p.setZero();

  // Assign to dense part of sparse vector
  x.setZero();
  r = b; // b - A * x0
  p.head(M) = r;

  double rnorm = squaredNorm(r);
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm_old = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // y = A.p
    col_l2g->update(p.data());
    y = A * p;

    // Calculate alpha = r.r/p.y
    //double pdoty = p.head(M).dot(y);
    double pdoty = std::transform_reduce(
      p.data(), p.data() + M, y.data(),
      0.0, std::plus<double>(), std::multiplies<double>());
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = rnorm_old / pdoty_sum;

    // Update x and r
    //x.head(M) += alpha * p.head(M);
    std::transform(
      x.data(), x.data() + M, p.data(), x.data(),
      [alpha](auto x, auto p) { return x + alpha * p; });
    r -= alpha * y;

    // Update rnorm
    rnorm = squaredNorm(r);
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p
    //p.head(M) = p.head(M) * beta + r;
    std::transform(
      p.data(), p.data() + M, r.data(), p.data(),
      [beta](auto p, auto r) { return p*beta + r; });

    if (rnorm_new / rnorm0 < rtol2)
      break;
  }

  return std::make_tuple(std::move(x), k);
}
