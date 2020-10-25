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

  int mpi_rank, mpi_size;
  MPI_Comm_rank(comm, &mpi_rank);
  MPI_Comm_size(comm, &mpi_size);

  // Problem data
  std::size_t size = 1000;
  double diag = 4.;

  // Create local to global map (diagonal matrix, no  ghosts)
  auto map
      = std::make_shared<spmv::L2GMap>(comm, size, std::vector<std::int64_t>());

  // Create data csr matrix
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

  spmv::Vector<double> c = A * b;

  double norm = c.norm();
  if (mpi_rank == 0)
  {
    std::cout << "Comp Norm: " << norm << std::endl;
    std::cout << "Exct Norm: " << diag * std::sqrt((double)mpi_size * size)
              << std::endl;
  }

  return 0;
}