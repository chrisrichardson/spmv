// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT
#include <CL/sycl.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include <spmv/L2GMap.h>
#include <spmv/Matrix.h>
#include <spmv/cg.h>
#include <spmv/read_petsc.h>

int cg_main(int argc, char** argv)
{
  // Turn off profiling
  MPI_Pcontrol(0);

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

  auto A = spmv::read_petsc_binary(MPI_COMM_WORLD, "petsc_mat_" + argv1);
  std::shared_ptr<spmv::L2GMap> l2g = A.col_map();

  auto b_data
      = spmv::read_petsc_binary_vector(MPI_COMM_WORLD, "petsc_vec_" + argv1);
  std::int32_t ls = l2g->local_size() + l2g->num_ghosts();
  b_data.resize(ls);
  spmv::Vector<double> b(b_data, l2g);

  // Get local and global sizes
  std::int64_t N = l2g->global_size();

  if (mpi_rank == 0)
    std::cout << "Global vec size = " << N << "\n";

  auto timer_end = std::chrono::system_clock::now();
  timings["0.ReadPetsc"] += (timer_end - timer_start);

  int max_its = 10000;
  double rtol = 1e-10;

  // Turn on profiling for solver only
  MPI_Pcontrol(1);
  timer_start = std::chrono::system_clock::now();
  auto [x, num_its] = spmv::cg(MPI_COMM_WORLD, A, b, max_its, rtol);
  timer_end = std::chrono::system_clock::now();
  timings["1.Solve"] += (timer_end - timer_start);
  MPI_Pcontrol(0);

  // Get norm on local part of vector
  double xnorm = x.norm();

  // Test result
  x.update();
  auto r = A * x - b;
  double rnom = r.norm();

  if (mpi_rank == 0)
  {
    std::cout << "r.norm = " << std::sqrt(rnom) << "\n";
    std::cout << "x.norm = " << std::sqrt(xnorm) << " in " << num_its
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

  return 0;
}
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  cg_main(argc, argv);

  MPI_Finalize();
  return 0;
}
