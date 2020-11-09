// Copyright (C) 2018-2020 Chris Richardson, Igor Baratta
// SPDX-License-Identifier:    MIT

#include "CreateA.h"

#include <memory>
#include <set>
#include <spmv/L2GMap.h>

#include <spmv/utils.h>

//-----------------------------------------------------------------------------
spmv::Matrix<double> create_A(MPI_Comm comm, int N)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);

  // Make a square Matrix divided evenly across cores
  std::vector<std::int64_t> ranges = spmv::owner_ranges(mpi_size, N);

  std::int64_t r0 = ranges[mpi_rank];
  std::int64_t r1 = ranges[mpi_rank + 1];
  int M = r1 - r0;

  // Local part of the matrix, COO format
  std::vector<double> coo_data;
  std::vector<std::int32_t> coo_row;
  std::vector<std::int32_t> coo_col;

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
      coo_data.push_back(1.0 - gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0);

      coo_data.push_back(gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0 + 1);
    }
    else if (c0 == (N - 1))
    {
      coo_data.push_back(gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0 - 1);

      coo_data.push_back(1.0 - gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0);
    }
    else
    {
      coo_data.push_back(1.0 - gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0);

      coo_data.push_back(1.0 - 2.0 * gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0);

      coo_data.push_back(gamma);
      coo_row.push_back(i);
      coo_col.push_back(c0 + 1);
    }
  }

  // Remap columns to local indexing
  std::set<std::int64_t> ghost_indices;
  std::int32_t nnz = coo_col.size();
  for (std::int32_t i = 0; i < nnz; ++i)
  {
    std::int32_t global_index = coo_col[i];
    if (global_index < r0 or global_index >= r1)
      ghost_indices.insert(global_index);
  }

  std::vector<std::int64_t> ghosts(ghost_indices.begin(), ghost_indices.end());
  auto col_l2g = std::make_shared<spmv::L2GMap>(comm, M, ghosts);
  auto row_l2g
      = std::make_shared<spmv::L2GMap>(comm, M, std::vector<std::int64_t>());

  // Rebuild A using local indices
  for (auto& col : coo_col)
  {
    col = col_l2g->global_to_local(col);
  }
  auto [data, indptr, indices] = spmv::coo_to_csr<double>(
      row_l2g->local_size(), col_l2g->local_size() + col_l2g->num_ghosts(),
      coo_data.size(), coo_row, coo_col, coo_data);

  return spmv::Matrix<double>(data, indptr, indices, col_l2g, row_l2g);
}
//-----------------------------------------------------------------------------
