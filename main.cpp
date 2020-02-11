// Copyright (C) 2018 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include "DistributedVector.h"
#include "read_petsc.h"

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
Eigen::SparseMatrix<double, Eigen::RowMajor> create_A(MPI_Comm comm, int N)
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Make a square Matrix divided evenly across cores
  std::vector<index_type> ranges = owner_ranges(mpi_size, N);

  index_type r0 = ranges[mpi_rank];
  index_type r1 = ranges[mpi_rank + 1];
  int M = r1 - r0;

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
    {
      A.insert(i, c0) = 1.0 - gamma;
      A.insert(i, c0 + 1) = gamma;
    }
    else if (c0 == (N - 1))
    {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - gamma;
    }
    else
    {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - 2.0 * gamma;
      A.insert(i, c0 + 1) = gamma;
    }
  }
  A.makeCompressed();

  return A;
}
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
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();

  // Either create a simple 1D stencil
  auto A = create_A(MPI_COMM_WORLD, 50000 * mpi_size);

  // Or read file created with "-ksp_view_mat binary" option
  //  auto A = read_petsc_binary(MPI_COMM_WORLD, "binaryoutput");

  // Get local range from number of rows in A
  std::vector<int> nrows_all(mpi_size);
  std::vector<index_type> ranges = {0};
  int nrows = A.rows();
  MPI_Allgather(&nrows, 1, MPI_INT, nrows_all.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  for (int i = 0; i < mpi_size; ++i)
    ranges.push_back(ranges.back() + nrows_all[i]);

  int N = ranges.back();
  int M = A.rows();
  int r0 = ranges[mpi_rank];

#ifdef EIGEN_USE_MKL_ALL
  // Remap columns to local indexing for MKL
  std::map<int, int> global_to_local;
  std::vector<MKL_INT> columns(A.outerIndexPtr()[M]);
  for (std::size_t i = 0; i < columns.size(); ++i)
  {
    int global_index = A.innerIndexPtr()[i];
    global_to_local.insert({global_index, 0});
  }

  int lc = 0;
  for (auto& q : global_to_local)
    q.second = lc++;

  for (std::size_t i = 0; i < columns.size(); ++i)
  {
    int global_index = A.innerIndexPtr()[i];
    columns[i] = global_to_local[global_index];
  }

  sparse_matrix_t A_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, M, N, A.outerIndexPtr(),
      A.outerIndexPtr() + 1, columns.data(), A.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(A_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

  struct matrix_descr mat_desc;
  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;

#endif

  auto timer_end = std::chrono::system_clock::now();
  //    timings["0.MatCreate"] += (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

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

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Apply matrix a few times
  int n_apply = 10000;
  if (mpi_rank == 0)
    std::cout << "Applying matrix " << n_apply << " times\n";

  // Temporary variable
  Eigen::VectorXd q(p.size());
  for (int i = 0; i < n_apply; ++i)
  {
    timer_start = std::chrono::system_clock::now();
    psp->update();
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                    psp->spvec().valuePtr(), 0.0, q.data());
#else
    q = A * psp->spvec();
#endif
    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    p = q;
    timer_end = std::chrono::system_clock::now();
    timings["4.Copy"] += (timer_end - timer_start);
  }

  double pnorm = p.squaredNorm();
  double pnorm_sum;
  MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0)
  {
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";
    std::chrono::duration<double> total_time;
    for (auto q : timings)
    {
      std::string pad(16 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q.second.count() << "\n";
      total_time += q.second;
    }
    std::cout << "[Total]           " << total_time.count() << "\n";
    std::cout << "----------------------------\n";
    std::cout << "norm = " << pnorm_sum << "\n";
  }

  // Destroy here before MPI_Finalize, because it holds a comm
  psp.reset();

  MPI_Finalize();
  return 0;
}
