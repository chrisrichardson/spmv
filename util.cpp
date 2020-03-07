// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "util.h"
#include "L2GMap.h"
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

using namespace spmv;
//-----------------------------------------------------------------------------
Eigen::VectorXd
spmv::extract_diagonal(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
{
  const int* inner = mat.innerIndexPtr();
  const int* outer = mat.outerIndexPtr();
  const double* val = mat.valuePtr();
  Eigen::VectorXd result(mat.rows());
  result.setZero();

  for (int i = 0; i < mat.rows(); ++i)
  {
    for (int j = outer[i]; j < outer[i + 1]; ++j)
    {
      if (inner[j] == i)
        result[i] = val[j];
    }
  }

  return result;
}
//-----------------------------------------------------------------------------
std::vector<int>
diagonal_block_nnz(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
{
  const int* inner = mat.innerIndexPtr();
  const int* outer = mat.outerIndexPtr();
  const int rows = mat.rows();
  const int cols = rows;

  std::vector<int> innernnz(rows, 0);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = outer[i]; j < outer[i + 1]; ++j)
    {
      if (inner[j] < cols)
        ++innernnz[i];
    }
  }

  return innernnz;
}
//-----------------------------------------------------------------------------
spmv::Matrix spmv::remap_mat(MPI_Comm comm,
                             std::shared_ptr<const spmv::L2GMap> row_map,
                             const spmv::Matrix& A)
{
  // Takes a SparseMatrix A, and fetches the ghost rows in row_map and
  // appends them to A, creating a new SparseMatrix B

  Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>> B = A.mat();
  std::shared_ptr<const spmv::L2GMap> col_map = A.col_map();

  if (B.rows() != row_map->local_size(false))
    throw std::runtime_error(
        "Cannot use L2G row map which is not compliant with matrix.\n"
        "The matrix must have the same number of rows as the local size "
        "(unghosted) of the row map.");

  // First fetch the nnz on each new row
  std::vector<int> nnz(row_map->local_size(true), -1);
  const int* Bouter = B.outerIndexPtr();
  const int* Binner = B.innerIndexPtr();

  if (B.isCompressed())
  {
    for (int i = 0; i < B.rows(); ++i)
      nnz[i] = Bouter[i + 1] - Bouter[i];
  }
  else
  {
    throw std::runtime_error("Must be compressed");
    const int* innernnz = B.innerNonZeroPtr();
    std::copy(innernnz, innernnz + B.rows(), nnz.begin());
  }

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Send/receive nnz for ghost rows in row_map
  row_map->update(nnz.data());

  std::stringstream s;
  s << rank << "] (" << row_map->ghosts().size() << ")\n";

  s << "Got nnz data for ghosts: [";
  for (int i = row_map->local_size(false); i < row_map->local_size(true); ++i)
    s << nnz[i] << " ";
  s << "]\n";

  // Fetch ghost row column indices (global)
  std::vector<std::int64_t> global_index_send;
  std::vector<std::int64_t> global_index_recv;

  std::vector<int> send_offset = {0};
  std::vector<int> send_count;
  const std::vector<int>& owned_count = row_map->num_owned_per_neighbour();
  const std::vector<int>& indexbuf = row_map->indexbuf();

  // Convert local to global using Column L2GMap
  std::int64_t col_global_offset = col_map->global_offset();
  int col_local_size = col_map->local_size(false);
  const std::vector<std::int64_t>& col_ghosts = col_map->ghosts();
  for (std::size_t i = 0; i < owned_count.size(); ++i)
  {
    int count = 0;
    for (int j = 0; j < owned_count[i]; ++j)
    {
      count += nnz[indexbuf[j]];
      for (int k = Bouter[indexbuf[j]]; k < Bouter[indexbuf[j] + 1]; ++k)
      {
        std::int64_t global_index;
        if (Binner[k] < col_local_size)
          global_index = Binner[k] + col_global_offset;
        else
          global_index = col_ghosts[Binner[k] - col_local_size];
        global_index_send.push_back(global_index);
      }
    }
    send_count.push_back(count);
    send_offset.push_back(send_offset.back() + count);
  }
  assert((int)global_index_send.size() == send_offset.back());

  std::vector<int> recv_count;
  std::vector<int> recv_offset = {0};
  const std::vector<int>& ghost_count = row_map->num_ghosts_per_neighbour();
  for (std::size_t i = 0; i < ghost_count.size(); ++i)
  {
    int count = 0;
    for (int j = 0; j < ghost_count[i]; ++j)
      count += nnz[row_map->local_size(false) + j];
    recv_count.push_back(count);
    recv_offset.push_back(recv_offset.back() + count);
  }
  global_index_recv.resize(recv_offset.back());

  MPI_Neighbor_alltoallv(
      global_index_send.data(), send_count.data(), send_offset.data(),
      MPI_INT64_T, global_index_recv.data(), recv_count.data(),
      recv_offset.data(), MPI_INT64_T, row_map->neighbour_comm());

  // Get a list of all ghost indices in column space, including new ones.
  std::set<std::int64_t> column_ghost_set(col_ghosts.begin(), col_ghosts.end());
  for (std::int64_t q : global_index_recv)
  {
    if (q < col_global_offset or q >= (col_global_offset + col_local_size))
      column_ghost_set.insert(q);
  }
  std::vector<std::int64_t> new_col_ghosts(column_ghost_set.begin(),
                                           column_ghost_set.end());

  auto new_col_map
      = std::make_shared<spmv::L2GMap>(comm, col_map->ranges(), new_col_ghosts);

  // Prepare new data for W
  Eigen::SparseMatrix<double, Eigen::RowMajor> W(row_map->local_size(true),
                                                 new_col_map->local_size(true));

  s << "W.rows() = " << W.rows() << " \n";
  s << "W.cols() = " << W.cols() << " \n";

  std::cout << s.str();

  return spmv::Matrix(W, new_col_map);
}
