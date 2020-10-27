// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include "cg.h"
#include <iomanip>
#include <iostream>

#include "L2GMap.h"
#include "Matrix.h"
#include "Vector.h"

//-----------------------------------------------------------------------------
std::tuple<spmv::Vector<double>, int> spmv::cg(MPI_Comm comm,
                                               const spmv::Matrix<double>& A,
                                               spmv::Vector<double>& b,
                                               int max_its, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  std::shared_ptr<const spmv::L2GMap> col_l2g = A.col_map();
  std::shared_ptr<const spmv::L2GMap> row_l2g = A.row_map();

  // Check the row map is unghosted
  if (row_l2g->num_ghosts() > 0)
    throw std::runtime_error("spmv::cg - Error: A.row_map() has ghost entries");

  int M = row_l2g->local_size();

  if (b.local_size() != M)
    throw std::runtime_error("spmv::cg - Error: b.rows() != A.rows()");

  // Residual vector
  auto y = b.duplicate();
  auto x = b.duplicate();

  x.set_zero();

  // Assign to dense part of sparse vector
  // TODO: Create copy assignment - Default creates shallow copy due to
  // shared_ptr.
  auto r = b.copy(); // b - A * x0
  auto p = r.copy();

  double rnorm = r.norm();
  double rnorm0 = rnorm;

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm_old = rnorm;
  int k = 0;
  while (k < max_its)
  {
    ++k;

    // y = A.p
    p.update();
    y = A * p;

    // Calculate alpha = r.r/p.y
    double pdoty = p.dot(y);
    double alpha = rnorm_old / pdoty;

    // Update x and r
    x += p * alpha;
    r -= y * alpha;

    // Update rnorm
    rnorm = r.norm();
    double beta = rnorm / rnorm_old;
    rnorm_old = rnorm;

    // Update p
    p = p * beta + r;

    std::cout << rnorm << std::endl;

    if (rnorm / rnorm0 < rtol2)
      break;
  }

  return std::make_tuple(std::move(x), k);
}
