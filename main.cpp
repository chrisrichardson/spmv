// Copyright (C) 2018 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "DistributedVector.h"

//-----------------------------------------------------------------------------
// Untested CG solver
void cg(const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
        const Eigen::Ref<const Eigen::VectorXd>& b)
{
  int M = A.rows();

  DistributedVector psp(MPI_COMM_WORLD, A);
  auto p = psp.vec();

  // Residual vector
  Eigen::VectorXd r(M);
  r = b;
  // Assign to dense part of sparse vector
  p = r;
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(M);

  double rnorm = r.squaredNorm();
  double rnorm_sum1;
  MPI_Allreduce(&rnorm, &rnorm_sum1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Iterations of CG
  for (int k = 0; k < 500; ++k)
  {
    // y = A.p
    psp.update();
    y = A * psp.spvec();

    // Update x and r
    double pdoty = p.dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double alpha = rnorm_sum1 / pdoty_sum;
    x += alpha * p;
    r -= alpha * y;

    // Update p
    rnorm = r.squaredNorm();
    double rnorm_sum2;
    MPI_Allreduce(&rnorm, &rnorm_sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double beta = rnorm_sum2 / rnorm_sum1;
    rnorm_sum1 = rnorm_sum2;

    p *= beta;
    p += r;
    std::cerr << k << ":" << rnorm << "\n";
  }
}
//-----------------------------------------------------------------------------
// Divide size into N ~equal chunks
std::vector<index_type> owner_ranges(std::int64_t size, index_type N)
{
  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<index_type> ranges;
  for (int rank = 0; rank < (size + 1); ++rank)
  {
    if (rank < r)
      ranges.push_back(rank * (n + 1));
    else
      ranges.push_back(rank * n + r);
  }

  return ranges;
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "# rank = " << rank << "/" << size << "\n";

  // Make a square Matrix divided evenly across cores
  int N = 5000;

  std::vector<index_type> ranges = owner_ranges(size, N);

  index_type r0 = ranges[rank];
  index_type r1 = ranges[rank + 1];
  int M = r1 - r0;

  std::cout << "# " << r0 << "-" << r1 << " \n";

  // Local part of the matrix
  // Must be RowMajor and compressed
  Eigen::SparseMatrix<double, Eigen::RowMajor> A(M, N);

  // Set up A
  // Add entries on all local rows
  // Using [local_row, global_column] indexing
  double gamma = 0.1;
  for (int i = 0; i < M; ++i)
  {
    // Global column diagonal index
    int c0 = r0 + i;
    // Special case for very first and last global rows
    if (c0 == 0)
      A.insert(i, c0) = 1.0;
    else if (c0 == (N - 1))
      A.insert(i, c0) = 1.0;
    else
    {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - 2.0 * gamma;
      A.insert(i, c0 + 1) = gamma;
    }
  }
  A.makeCompressed();

  // Make distributed vector - this is the only
  // one that needs to be 'sparse'
  auto psp = std::make_shared<DistributedVector>(MPI_COMM_WORLD, A);
  auto p = psp->vec();

  // Set up values
  for (int i = 0; i < M; ++i)
  {
    double z = (double)(i + r0) / double(N);
    p[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  // Apply matrix a few times

  auto start = std::chrono::system_clock::now();

  // Temporary variable
  Eigen::VectorXd q;
  for (int i = 0; i < 10000; ++i)
  {
    psp->update();
    q = A * psp->spvec();
    p = q;
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end - start;

  double pnorm = p.squaredNorm();
  double pnorm_sum;
  MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0)
  {
    std::cout << "time = " << diff.count() << "s.\n";
    std::cout << "norm = " << pnorm_sum << "\n";
  }

  // // Output
  // std::stringstream s;
  // s << rank << " [";
  // for (int i = 0; i < M; ++i)
  //   s << p[i] << " ";
  // s << "]";

  // for (int i = 0; i < size; ++i)
  // {
  //   if (i == rank)
  //     std::cout << s.str() << "\n";
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Destroy here before MPI_Finalize, because it holds a comm
  psp.reset();

  MPI_Finalize();
  return 0;
}
