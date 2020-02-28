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
#include "L2GMap.h"
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
  // Read in a PETSc binary format matrix
  auto [R, l2g] = spmv::read_petsc_binary(MPI_COMM_WORLD, "R4.dat");
  auto q = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, "b4.dat");

  // Get local and global sizes
  std::int64_t M = R.rows();
  std::int64_t N = l2g->global_size();

  std::cout << "Vector = " << q.size() << " " << M << "\n";

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t R_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(
      &R_mkl, SPARSE_INDEX_BASE_ZERO, R.rows(), R.cols(), R.outerIndexPtr(),
      R.outerIndexPtr() + 1, R.innerIndexPtr(), R.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(R_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  struct matrix_descr mat_desc;
  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;
#endif
  auto timer_end = std::chrono::system_clock::now();
  timings["0.PetscRead"] += (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  // Vector in "column space" with extra space for ghosts at end
  Eigen::VectorXd psp(l2g->local_size(true));

  // Set up values in local range (column space)
  int r0 = l2g->global_offset();
  for (int i = 0; i < l2g->local_size(false); ++i)
  {
    double z = (double)(i + r0) / double(N);
    psp[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Apply matrix
  if (mpi_rank == 0)
    std::cout << "Applying matrix\n";

  double pnorm_sum, qnorm_sum;
  for (int i = 0; i < 10; ++i)
  {
    // Restrict
    timer_start = std::chrono::system_clock::now();
#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, R_mkl, mat_desc, q.data(),
                    0.0, psp.data());
#else
    psp = R.transpose() * q;
#endif
    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    l2g->reverse_update(psp.data());
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    Eigen::Map<Eigen::VectorXd> p(psp.data(), l2g->local_size(false));
    double pnorm = p.squaredNorm();
    MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Prolongate
    timer_start = std::chrono::system_clock::now();
    l2g->update(psp.data());
    timer_end = std::chrono::system_clock::now();
    timings["2.SparseUpdate"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, R_mkl, mat_desc,
                    psp.data(), 0.0, q.data());
#else
    q = R * psp;
#endif
    timer_end = std::chrono::system_clock::now();
    timings["3.SpMV"] += (timer_end - timer_start);

    double qnorm = q.squaredNorm();
    MPI_Allreduce(&qnorm, &qnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  if (mpi_rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";

  std::chrono::duration<double> total_time
      = std::chrono::duration<double>::zero();
  for (auto q : timings)
    total_time += q.second;
  timings["Total"] = total_time;

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
  }

  if (mpi_rank == 0)
  {
    std::cout << "----------------------------\n";
    std::cout << "norm q = " << qnorm_sum << "\n";
    std::cout << "norm p = " << pnorm_sum << "\n";
  }

  // Need to destroy L2G here before MPI_Finalize, because it holds a comm
  l2g.reset();

  MPI_Finalize();
  return 0;
}
