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
#include "CreateA.h"

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

  int nc_local = 0;
  for (auto& q : global_to_local)
    q.second = nc_local++;

  for (std::size_t i = 0; i < columns.size(); ++i)
  {
    int global_index = A.innerIndexPtr()[i];
    columns[i] = global_to_local[global_index];
  }

  sparse_matrix_t A_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(&A_mkl, SPARSE_INDEX_BASE_ZERO, M, nc_local, A.outerIndexPtr(),
                                                   A.outerIndexPtr() + 1, columns.data(), A.valuePtr());
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
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";

  std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();
  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::string pad(16 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max << "\n";
    }
    total_time += q.second;
  }

  double total_local = total_time.count(), total_min, total_max;
  MPI_Reduce(&total_local, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&total_local, &total_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  if (mpi_rank == 0)
  {
    std::cout << "[Total]           " << total_min << '\t' << total_max << "\n";
    std::cout << "----------------------------\n";
    std::cout << "norm = " << pnorm_sum << "\n";
  }

  // Destroy here before MPI_Finalize, because it holds a comm
  psp.reset();

  MPI_Finalize();
  return 0;
}
