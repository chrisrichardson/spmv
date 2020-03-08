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
#include "util.h"

void restrict_main()
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();
  // Read in a PETSc binary format matrix

  auto R = spmv::read_petsc_binary(MPI_COMM_WORLD, "R4.dat");

  auto A = spmv::read_petsc_binary(MPI_COMM_WORLD, "A4.dat");
  auto l2g_A = A.col_map();
  std::cout << "A = " << A.rows() << "x" << A.col_map()->local_size(true)
            << "\n";

  // Remap row map to column map...
  spmv::Matrix B = remap_mat(MPI_COMM_WORLD, l2g_A, R);
  auto l2g_B = B.col_map();

  std::cout << "B=" << B.rows() << "x" << B.col_map()->local_size(true) << "\n";

  spmv::Matrix C(A.mat() * B.mat(), B.col_map());

  std::cout << "C = " << C.rows() << " x" << C.col_map()->local_size(true)
            << "\n";

  std::cout << "R.mat = " << R.mat().rows() << "x" << R.mat().cols() << "\n";

  auto Q = spmv::Matrix(
      R.mat().leftCols(R.col_map()->local_size(false)).transpose() * C.mat(),
      C.col_map());

  std::cout << "Q.mat = " << Q.rows() << "x" << Q.col_map()->local_size(true)
            << "\n";

  int qrows = Q.rows();
  int qrows_sum;

  MPI_Allreduce(&qrows, &qrows_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  std::cout << qrows_sum << "x" << Q.col_map()->global_size() << " "
            << Q.mat().nonZeros() << "\n";

  auto b4 = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, "b4.dat");

  // Get global size
  auto l2g_R = R.col_map();
  std::int64_t N = l2g_R->global_size();

  auto timer_end = std::chrono::system_clock::now();
  timings["0.PetscRead"] += (timer_end - timer_start);

  if (mpi_rank == 0)
    std::cout << "Creating vector of size " << N << "\n";

  // Vector in "column space" with extra space for ghosts at end
  Eigen::VectorXd b3sp(l2g_B->local_size(true));

  // Restrict
  timer_start = std::chrono::system_clock::now();
  b3sp = B.transpmult(b4);
  l2g_B->reverse_update(b3sp.data());

  auto w = Q * b3sp;
  std::cout << Q.col_map()->local_size(true) << " = " << b3sp.rows() << "\n";

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
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  restrict_main();

  MPI_Finalize();
  return 0;
}
