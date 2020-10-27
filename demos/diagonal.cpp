// Copyright (C) 2020 Igor Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>
#include <mkl_sycl.hpp>

#include <mpi.h>
#include <numeric>
#include <vector>

#include <spmv/spmv.h>

// Create diagonal matrix and multiply by vector.
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  // Problem data
  std::size_t size = 50000;
  std::size_t n_apply = 100;
  double diag = 4.;

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  // Create local to global map (diagonal matrix, no  ghosts)
  auto map = std::make_shared<spmv::L2GMap>(comm, size);

  // Create diagonal csr matrix data
  std::vector<double> data(size, diag);
  std::vector<std::int32_t> indptr(size + 1, 1.);
  std::iota(indptr.begin(), indptr.end(), 0);
  std::vector<std::int32_t> indices(size, 1.);
  std::iota(indices.begin(), indices.end(), 0);

  // Create Matrix
  spmv::Matrix<double> A(data, indptr, indices, map, map);

  // Create vector data
  std::vector<double> b_data(size, 1.);
  spmv::Vector<double> b(b_data, map);

  if (mpi_rank == 0)
    std::cout << "Applying diagonal matrix to vector " << n_apply << " times\n";

  std::vector<double> c_data(size, 0.0);
  spmv::Vector<double> c(c_data, map);

  for (std::size_t i = 0; i < n_apply; i++)
  {
    auto timer_start = std::chrono::system_clock::now();
    c = A * b;
    auto timer_end = std::chrono::system_clock::now();
    timings["1.Mult w/ Alloc"] += (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    A.mult(b, c);
    timer_end = std::chrono::system_clock::now();
    timings["2.Mult w/o Alloc"] += (timer_end - timer_start);
  }

  if (mpi_rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";

  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::string pad(32 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                << "\n";
    }
  }

  double norm = c.norm();
  if (mpi_rank == 0)
  {
    std::cout << "\nComp Norm: " << norm << std::endl;
    std::cout << "Exct Norm: " << diag * std::sqrt((double)mpi_size * size)
              << std::endl;
  }

  return 0;
}