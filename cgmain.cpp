// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iostream>
#include <memory>
#include <mpi.h>

#include "CreateA.h"
#include "L2GMap.h"
#include "cg.h"
#include "read_petsc.h"

#include "cuda_check.h"

//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  // Turn off profiling
  MPI_Pcontrol(0);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

#ifdef HAVE_CUDA
  // Get number of local gpus
  int ngpus;
  cuda_CHECK(cudaGetDeviceCount(&ngpus));

  MPI_Comm split_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &split_comm);
  int nlocal;
  MPI_Comm_size(split_comm, &nlocal);
  std::cout << "nlocal = " << nlocal << "\n";
  std::cout << "ngpus = " << ngpus << "\n";

  int local_rank;
  MPI_Comm_rank(split_comm, &local_rank);

  // Hack
  cuda_CHECK(cudaSetDevice(local_rank));
#endif

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  auto timer_start = std::chrono::system_clock::now();
  // Either create a simple 1D stencil
  std::string argv1;
  if (argc == 2)
    argv1 = argv[1];
  else
    throw std::runtime_error("Use with filename");

  std::string cores = std::to_string(mpi_size);
  auto [A, l2g]
      = spmv::read_petsc_binary(MPI_COMM_WORLD, "petsc_mat" + argv1 + ".dat");
  auto b = spmv::read_petsc_binary_vector(MPI_COMM_WORLD,
                                          "petsc_vec" + argv1 + ".dat");
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
  auto [x, num_its] = spmv::cg(MPI_COMM_WORLD, A, l2g, b, max_its, rtol);
  timer_end = std::chrono::system_clock::now();
  timings["0.Solve(CPU)"] += (timer_end - timer_start);
  timings["0.Solve"] += (timer_end - timer_start);
  MPI_Pcontrol(0);

  double xnorm = x.squaredNorm();
  double xnorm_sum;
  MPI_Allreduce(&xnorm, &xnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0)
    std::cout << "x.norm = " << std::sqrt(xnorm_sum) << " in " << num_its
              << "\n";

#ifdef HAVE_CUDA
  timer_start = std::chrono::system_clock::now();
  std::tie(x, num_its)
      = spmv::cg_cuda(MPI_COMM_WORLD, A, l2g, b, max_its, rtol);
  timer_end = std::chrono::system_clock::now();
  timings["0.Solve(CUDA)"] += (timer_end - timer_start);

  xnorm = x.squaredNorm();
  MPI_Allreduce(&xnorm, &xnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Test result - prepare ghosted vector
  Eigen::VectorXd xsp(l2g->local_size(true));
  xsp.head(A.rows()) = x;
  l2g->update(xsp.data());

  Eigen::VectorXd r = A * xsp - b;
  double rnorm = r.squaredNorm();
  double rnorm_sum;
  MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0)
  {
    std::cout << "r.norm = " << std::sqrt(rnorm_sum) << "\n";
    std::cout << "x.norm = " << std::sqrt(xnorm_sum) << " in " << num_its
              << " iterations\n";
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";
  }

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
  }

  // Need to destroy L2G here before MPI_Finalize, because it holds a comm
  l2g.reset();
  MPI_Finalize();
  return 0;
}
