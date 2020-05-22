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

  auto rnorm = std::make_shared<double>(squaredNorm(r));
  auto rnorm0 = std::make_shared<double>();
  MPI_Allreduce(rnorm.get(), rnorm0.get(), 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  auto rnorm_old = std::make_shared<double>(*rnorm0);
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // y = A.p
    col_l2g->update(p.data());
    y = A * p;

    //// Calculate alpha = r.r/p.y
    //double pdoty = p.head(M).dot(y);
    auto pdoty = std::make_shared<double>(std::transform_reduce(
      p.data(), p.data() + M, y.data(),
      0.0, std::plus<double>(), std::multiplies<double>()));
    auto pdoty_sum = std::make_shared<double>();
    MPI_Allreduce(pdoty.get(), pdoty_sum.get(), 1, MPI_DOUBLE, MPI_SUM, comm);
    auto alpha = std::make_shared<double>(*rnorm_old / *pdoty_sum);
    //double alpha = rnorm_old / pdoty_sum;

    // Update x and r
    //x.head(M) += alpha * p.head(M);
    //r -= *alpha * y;
    std::transform(
      x.data(), x.data() + M, p.data(), x.data(),
      [alpha](auto x, auto p) { return x + *alpha * p; });
    std::transform(
      r.data(), r.data() + r.size(), y.data(), r.data(),
      [alpha](auto r, auto y) { return r - *alpha * y; });

    // Update rnorm
    *rnorm = squaredNorm(r);
    auto rnorm_new = std::make_shared<double>();
    MPI_Allreduce(rnorm.get(), rnorm_new.get(), 1, MPI_DOUBLE, MPI_SUM, comm);
    auto beta = std::make_shared<double>(*rnorm_new / *rnorm_old);
    *rnorm_old = *rnorm_new;

    // Update p
    //p.head(M) = p.head(M) * beta + r;
    std::transform(
      p.data(), p.data() + M, r.data(), p.data(),
      [beta](auto p, auto r) { return *beta*p + r; });

    if (*rnorm_new / *rnorm0 < rtol2)
      break;
  }

  return std::make_tuple(std::move(x), k);
}
