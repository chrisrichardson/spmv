// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
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

#include "CreateA.h"
#include "read_petsc.h"

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
  // auto [A, l2g] = create_A(MPI_COMM_WORLD, 50000);

  // Or read file created with "-ksp_view_mat binary" option
  auto [A, l2g] = read_petsc_binary(MPI_COMM_WORLD, "binaryoutput");

  // Get local and global sizes
  std::int64_t M = A.rows();
  std::int64_t N = l2g->global_size();

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, A.rows(), A.cols(), A.outerIndexPtr(),
      A.outerIndexPtr() + 1, A.innerIndexPtr(), A.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(A_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  struct matrix_descr mat_desc;
  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;
#endif

  auto timer_end = std::chrono::system_clock::now();
  //    timings["0.MatCreate"] += (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  // Vector with extra space for ghosts at end
  Eigen::VectorXd psp(l2g->local_size());

  // Set up values in local range
  int r0 = l2g->global_offset();
  for (int i = 0; i < M; ++i)
  {
    double z = (double)(i + r0) / double(N);
    psp[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Apply matrix a few times
  int n_apply = 1000;
  if (mpi_rank == 0)
    std::cout << "Applying matrix " << n_apply << " times\n";

  // Temporary variable
  Eigen::VectorXd q(M);
  for (int i = 0; i < n_apply; ++i)
  {
    timer_start = std::chrono::system_clock::now();
    l2g->update(psp.data());
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                    psp.data(), 0.0, q.data());
#else
    q = A * psp;
#endif
    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    psp.head(M) = q;
    timer_end = std::chrono::system_clock::now();
    timings["4.Copy"] += (timer_end - timer_start);
  }

  double pnorm = psp.head(M).squaredNorm();
  double pnorm_sum;
  MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";

  std::chrono::duration<double> total_time
      = std::chrono::duration<double>::zero();
  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::string pad(16 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                << "\n";
    }
    total_time += q.second;
  }

  double total_local = total_time.count(), total_min, total_max;
  MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&total_local, &total_min, 1, MPI_DOUBLE, MPI_MIN, 0,
             MPI_COMM_WORLD);
  if (mpi_rank == 0)
  {
    std::cout << "[Total]           " << total_min << '\t' << total_max << "\n";
    std::cout << "----------------------------\n";
    std::cout << "norm = " << pnorm_sum << "\n";
  }

  // Need to destroy L2G here before MPI_Finalize, because it holds a comm
  l2g.reset();

  MPI_Finalize();
  return 0;
}
