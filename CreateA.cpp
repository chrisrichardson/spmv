// Copyright (C) 2018-2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CreateA.h"
#include "L2GMap.h"
#include <Eigen/Sparse>
#include <memory>
#include <set>

//-----------------------------------------------------------------------------
// Divide size into N ~equal chunks
std::vector<index_type> owner_ranges(std::int64_t size, index_type N)
{
  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<index_type> ranges;
  for (int rank = 0; rank < (size + 1); ++rank)
  {
    if (rank < r)
      ranges.push_back(rank * (n + 1));
    else
      ranges.push_back(rank * n + r);
  }

  return ranges;
}
//-----------------------------------------------------------------------------
spmv::Matrix create_A(MPI_Comm comm, int N)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);

  // Make a square Matrix divided evenly across cores
  std::vector<index_type> ranges = owner_ranges(mpi_size, N);

  index_type r0 = ranges[mpi_rank];
  index_type r1 = ranges[mpi_rank + 1];
  int M = r1 - r0;

  // Local part of the matrix
  // Must be RowMajor and compressed
  Eigen::SparseMatrix<double, Eigen::RowMajor> A(M, N);

  // Set up A
  // Add entries on all local rows
  // Using [local_row, global_column] indexing
  double gamma = 0.1;
  for (int i = 0; i < M; ++i)
  {
    // Global column diagonal index
    int c0 = r0 + i;
    // Special case for very first and last global rows
    if (c0 == 0)
    {
      A.insert(i, c0) = 1.0 - gamma;
      A.insert(i, c0 + 1) = gamma;
    }
    else if (c0 == (N - 1))
    {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - gamma;
    }
    else
    {
      A.insert(i, c0 - 1) = gamma;
      A.insert(i, c0) = 1.0 - 2.0 * gamma;
      A.insert(i, c0 + 1) = gamma;
    }
  }
  A.makeCompressed();

  // Remap columns to local indexing
  std::set<index_type> ghost_indices;
  std::int32_t nnz = A.outerIndexPtr()[M];
  for (std::int32_t i = 0; i < nnz; ++i)
  {
    index_type global_index = A.innerIndexPtr()[i];
    if (global_index < r0 or global_index >= r1)
      ghost_indices.insert(global_index);
  }

  std::vector<std::int64_t> ghosts(ghost_indices.begin(), ghost_indices.end());
  auto l2g = std::make_shared<spmv::L2GMap>(comm, ranges, ghosts);

  // Rebuild A using local indices
  Eigen::SparseMatrix<double, Eigen::RowMajor> Alocal(M, M + ghosts.size());
  std::vector<Eigen::Triplet<double>> vals;
  index_type* Aouter = A.outerIndexPtr();
  index_type* Ainner = A.innerIndexPtr();
  double* Aval = A.valuePtr();

  for (index_type row = 0; row < M; ++row)
  {
    for (index_type j = Aouter[row]; j < Aouter[row + 1]; ++j)
    {
      index_type col = l2g->global_to_local(Ainner[j]);
      double val = Aval[j];
      vals.push_back(Eigen::Triplet<double>(row, col, val));
    }
  }
  Alocal.setFromTriplets(vals.begin(), vals.end());

  return spmv::Matrix(Alocal, l2g);
}
//-----------------------------------------------------------------------------
