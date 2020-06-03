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

using std::begin;
double* begin(Eigen::VectorXd &x) {
  return x.data();
}
using std::end;
double* end(Eigen::VectorXd &x) {
  return x.data() + x.size();
}
using std::size;
std::size_t size(Eigen::VectorXd &x) {
  return x.size();
}

template <typename X, typename Y>
std::shared_ptr<typename X::value_type> dot_product(X x, Y y) {
  return std::make_shared<typename X::value_type>(
    size(x) <= size(y) ?
    std::transform_reduce(begin(x), end(x), begin(y), typename X::value_type()) :
    std::transform_reduce(begin(y), end(y), begin(x), typename X::value_type())
  );
}
template<typename X>
std::shared_ptr<typename X::value_type> squaredNorm(X x) {
  return dot_product(x, x);
}


namespace mpi {

std::vector<double> allreduce(MPI_Comm const &comm, std::vector<double> const &x, MPI_Op const &op) {
  std::vector<double> x_global(x.size());
  MPI_Allreduce(x.data(), x_global.data(), x.size(), MPI_DOUBLE, op, comm);
  return x_global;
}
Eigen::VectorXd allreduce(MPI_Comm const &comm, Eigen::VectorXd const &x, MPI_Op const &op) {
  Eigen::VectorXd x_global(x.size());
  MPI_Allreduce(x.data(), x_global.data(), x.size(), MPI_DOUBLE, op, comm);
  return x_global;
}
std::shared_ptr<double> allreduce(MPI_Comm const &comm, std::shared_ptr<double> const &x, MPI_Op const &op) {
  auto x_global = std::make_shared<double>();
  MPI_Allreduce(x.get(), x_global.get(), 1, MPI_DOUBLE, op, comm);
  return x_global;
}

template<typename T>
T sum(MPI_Comm const &comm, T const &x) {
  return allreduce(comm, x, MPI_SUM);
}
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

  auto rnorm0 = mpi::sum(comm, squaredNorm(r));
  auto rnorm = std::make_shared<double>(*rnorm0);

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
    auto pdoty = mpi::sum(comm, dot_product(p, y));
    auto alpha = std::make_shared<double>(*rnorm_old / *pdoty);

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
    auto rnorm = mpi::sum(comm, squaredNorm(r));
    auto beta = std::make_shared<double>(*rnorm / *rnorm_old);
    *rnorm_old = *rnorm;

    // Update p
    //p.head(M) = p.head(M) * beta + r;
    std::transform(
      p.data(), p.data() + M, r.data(), p.data(),
      [beta](auto p, auto r) { return *beta*p + r; });

    if (*rnorm / *rnorm0 < rtol2)
      break;
  }

  return std::make_tuple(std::move(x), k);
}