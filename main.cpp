// Copyright (C) 2018 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <set>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "DistributedVector.h"

std::vector<index_type> owner_ranges(MPI_Comm comm, index_type N)
{
  // Work out ranges for all processes
  int size;
  MPI_Comm_size(comm, &size);

  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<index_type> ranges;
  for (int rank = 0; rank != (size + 1); ++rank)
  {
    if (rank < r)
      ranges.push_back(rank * (n + 1));
    else
      ranges.push_back(rank * n + r);
  }

  return ranges;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "# rank = " << rank << "/" << size << "\n";

  // Make a square Matrix divided evenly across cores
  int N = 1000;

  std::vector<index_type> ranges = owner_ranges(MPI_COMM_WORLD, N);

  index_type r0 = ranges[rank];
  index_type r1 = ranges[rank + 1];
  int M = r1 - r0;

  std::cout << "# " << r0 << "-" << r1 << " \n";

  // Local part of the matrix
  // Must be RowMajor and compressed
  Eigen::SparseMatrix<double, Eigen::RowMajor> A(M, N);

  // Set up A
  for (unsigned int i = 0; i < M; ++i)
  {
    if ((r0 + i) == 0)
      A.insert(i, i) = 1.0;
    else if ((r0 + i) == (N - 1))
      A.insert(i, i) = 1.0;
    else
    {
      A.insert(i, r0 + i - 1) = 1.0;
      A.insert(i, r0 + i) = -2.0;
      A.insert(i, r0 + i + 1) = 1.0;
    }
  }
  A.makeCompressed();

  // Make distributed vector - this is the only
  // one that needs to be 'sparse'
  DistributedVector psp;
  psp.setup(MPI_COMM_WORLD, A, ranges);
  auto p = psp.vec();

  // RHS vector
  Eigen::VectorXd b(M);

  // Set up values
  for (unsigned int i = 0; i != M; ++i)
  {
    double z = (double)(i + r0) / double(N);
    b[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  // Residual vector
  Eigen::VectorXd r(M);
  r = b;
  // Assign to dense part of sparse vector
  p = r;
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(M);

  // Iterations of CG
  for (unsigned int k = 0; k != 500; ++k)
  {
    double rnorm = r.squaredNorm();

    // y = A.p
    psp.update(MPI_COMM_WORLD);
    y = A * psp.spvec();

    // Update x and r
    double alpha = rnorm / p.dot(y);
    x += alpha * p;
    r -= alpha * y;

    // Update p
    double beta = r.squaredNorm() / rnorm;
    p *= beta;
    p += r;
    //    std::cerr << k << ":" << rnorm << "\n";
  }

  // Output
  std::stringstream s;
  for (unsigned int i = 0; i != M; ++i)
    s << x[i] << "\n";

  for (unsigned int i = 0; i != size; ++i)
  {
    if (i == rank)
      std::cout << s.str() << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
