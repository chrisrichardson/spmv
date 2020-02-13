// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include "CreateR.h"
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

  // Create a restriction matrix
  auto [R, l2g_col, l2g_row] = create_R(MPI_COMM_WORLD, 60000);

  // Get local and global sizes
  std::int64_t M = R.rows();
  std::int64_t N = l2g_row->global_size();

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
  //    timings["0.MatCreate"] += (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  // Vector with extra space for ghosts at end
  Eigen::VectorXd psp(l2g_col->local_size());
  psp.setZero();

  // Set up values in local range
  int c0 = l2g_col->global_offset();
  for (int i = 0; i < psp.size(); ++i)
  {
    double z = (double)(i + c0) / double(N * 2);
    psp[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }
  double pnorm = psp.squaredNorm();
  double pnorm_sum;
  MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  timer_end = std::chrono::system_clock::now();
  timings["1.VecCreate"] += (timer_end - timer_start);

  // Temporary variable
  Eigen::VectorXd q(M);
  l2g_col->update(psp.data());
#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, R_mkl, mat_desc,
                  psp.data(), 0.0, q.data());
#else
  q = R * psp;
#endif

  double qnorm = q.squaredNorm();
  double qnorm_sum;
  MPI_Allreduce(&qnorm, &qnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1.0, R_mkl, mat_desc, q.data(),
                  0.0, psp.data());
#else
  psp = R.transpose() * q;
#endif
  l2g_col->reverse_update(psp.data());

  std::ofstream file;
  file.open("out" + std::to_string(mpi_rank) + ".dat");
  file << psp << "\n";
  file.close();

  if (mpi_rank == 0)
  {
    std::cout << "norm = " << pnorm_sum << " " << qnorm_sum << "\n";
  }

  // Need to destroy L2G here before MPI_Finalize, because it holds a comm
  l2g_col.reset();
  l2g_row.reset();

  MPI_Finalize();
  return 0;
}
