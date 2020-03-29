// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include "CreateA.h"
#include <spmv/jacobi.h>
#include <spmv/spmv.h>

void jacobi_main(int argc, char** argv)
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();

  std::string argv1;
  if (argc == 2)
    argv1 = argv[1];
  else
    throw std::runtime_error("Use with filename");

  auto A
      = spmv::read_petsc_binary(MPI_COMM_WORLD, "petsc_mat" + argv1 + ".dat");
  auto b = spmv::read_petsc_binary_vector(MPI_COMM_WORLD,
                                          "petsc_vec" + argv1 + ".dat");
  // Get local and global sizes
  auto l2g = A.col_map();
  std::int64_t N = l2g->global_size();

  if (mpi_rank == 0)
    std::cout << "Global vec size = " << N << "\n";

  auto timer_end = std::chrono::system_clock::now();
  timings["0.ReadPetsc"] += (timer_end - timer_start);

  int num_its = 10000;
  double rtol = 1e-6;

  // Eigen::VectorXd D = spmv::extract_diagonal(A.mat()).cwiseInverse();
  Eigen::VectorXd x(l2g->local_size(true));

  timer_start = std::chrono::system_clock::now();
  double rnorm = 2 * rtol;
  int i = 0;
  while (rnorm > rtol and i++ < num_its)
    rnorm = spmv::jacobi(A, x, b);
  timer_end = std::chrono::system_clock::now();
  timings["1.Solve"] += (timer_end - timer_start);

  // Get norm on local part of vector
  double xnorm = x.head(l2g->local_size(false)).squaredNorm();
  double xnorm_sum;
  MPI_Allreduce(&xnorm, &xnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Test result
  l2g->update(x.data());

  Eigen::VectorXd r = A * x - b;
  rnorm = r.squaredNorm();
  double rnorm_sum;
  MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0)
  {
    std::cout << "r.norm = " << std::sqrt(rnorm_sum) << "\n";
    std::cout << "x.norm = " << std::sqrt(xnorm_sum) << " in " << i
              << " iterations\n";
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";
  }

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
    std::cout << "----------------------------\n";
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  jacobi_main(argc, argv);

  MPI_Finalize();
  return 0;
}
