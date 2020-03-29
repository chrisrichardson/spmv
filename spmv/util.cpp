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
spmv::Matrix<double>
spmv::remap_mat(MPI_Comm comm, std::shared_ptr<const spmv::L2GMap> row_map,
                const spmv::Matrix<double>& mat)
{
  // Takes a SparseMatrix A, and fetches the ghost rows in row_map and
  // appends them to A, creating a new SparseMatrix B

  const Eigen::SparseMatrix<double, Eigen::RowMajor>& A = mat.mat();
  std::shared_ptr<const spmv::L2GMap> col_map = mat.col_map();

  if (A.rows() != row_map->local_size(false))
    throw std::runtime_error(
        "Cannot use L2G row map which is not compliant with matrix.\n"
        "The matrix must have the same number of rows as the local size "
        "(unghosted) of the row map.");

  // First fetch the nnz on each new row
  std::vector<int> nnz(row_map->local_size(true), -1);
  const int* Aouter = A.outerIndexPtr();
  const int* Ainner = A.innerIndexPtr();
  const double* Aval = A.valuePtr();

  if (A.isCompressed())
  {
    for (int i = 0; i < A.rows(); ++i)
      nnz[i] = Aouter[i + 1] - Aouter[i];
  }
  else
  {
    throw std::runtime_error("Must be compressed");
    const int* innernnz = A.innerNonZeroPtr();
    std::copy(innernnz, innernnz + A.rows(), nnz.begin());
  }

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Send/receive nnz for ghost rows in row_map
  row_map->update(nnz.data());

  std::stringstream s;
  s << rank << "] (" << row_map->ghosts().size() << ")\n";

  int nnz_sum = 0;
  for (int i = row_map->local_size(false); i < row_map->local_size(true); ++i)
  {
    nnz_sum += nnz[i];
  }
  s << "nnz_sum = " << nnz_sum << "\n";

  std::cout << s.str();

  // Fetch ghost row column indices (global)
  std::vector<std::int64_t> global_index_send;
  std::vector<std::int64_t> global_index_recv;
  std::vector<double> global_value_send;
  std::vector<double> global_value_recv;

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
      for (int k = Aouter[indexbuf[j]]; k < Aouter[indexbuf[j] + 1]; ++k)
      {
        std::int64_t global_index;
        if (Ainner[k] < col_local_size)
          global_index = Ainner[k] + col_global_offset;
        else
          global_index = col_ghosts[Ainner[k] - col_local_size];
        global_index_send.push_back(global_index);
      }
    }
    send_count.push_back(count);
    send_offset.push_back(send_offset.back() + count);
  }
  assert((int)global_index_send.size() == send_offset.back());
  global_value_send.resize(send_offset.back());

  std::vector<int> recv_count;
  std::vector<int> recv_offset = {0};
  const std::vector<int>& ghost_count = row_map->num_ghosts_per_neighbour();
  int c = 0;
  for (std::size_t i = 0; i < ghost_count.size(); ++i)
  {
    int count = 0;
    for (int j = 0; j < ghost_count[i]; ++j)
    {
      count += nnz[row_map->local_size(false) + c];
      ++c;
    }

    recv_count.push_back(count);
    recv_offset.push_back(recv_offset.back() + count);
  }
  global_index_recv.resize(recv_offset.back());
  global_value_recv.resize(recv_offset.back());
  assert(recv_offset.back() == nnz_sum);

  MPI_Neighbor_alltoallv(
      global_index_send.data(), send_count.data(), send_offset.data(),
      MPI_INT64_T, global_index_recv.data(), recv_count.data(),
      recv_offset.data(), MPI_INT64_T, row_map->neighbour_comm());

  MPI_Neighbor_alltoallv(
      global_value_send.data(), send_count.data(), send_offset.data(),
      MPI_DOUBLE, global_value_recv.data(), recv_count.data(),
      recv_offset.data(), MPI_DOUBLE, row_map->neighbour_comm());

  // Get a list of all ghost indices in column space, including new ones.
  std::set<std::int64_t> column_ghost_set(col_ghosts.begin(), col_ghosts.end());
  for (std::int64_t q : global_index_recv)
  {
    if (q < col_global_offset or q >= (col_global_offset + col_local_size))
      column_ghost_set.insert(q);
  }
  std::vector<std::int64_t> new_col_ghosts(column_ghost_set.begin(),
                                           column_ghost_set.end());

  // Find old ghost in new col space
  std::vector<std::int32_t> map_old_to_new;
  for (std::int64_t old_ghost : col_ghosts)
    map_old_to_new.push_back(
        std::find(new_col_ghosts.begin(), new_col_ghosts.end(), old_ghost)
        - new_col_ghosts.begin());

  auto new_col_map = std::make_shared<spmv::L2GMap>(
      comm, col_map->local_size(false), new_col_ghosts);

  auto new_row_map = std::make_shared<spmv::L2GMap>(
      comm, row_map->local_size(false), std::vector<std::int64_t>());

  // Prepare new data for B
  Eigen::SparseMatrix<double, Eigen::RowMajor> B(row_map->local_size(true),
                                                 new_col_map->local_size(true));

  std::vector<Eigen::Triplet<double>> vals;

  // Copy existing rows from A to B
  for (int row = 0; row < A.rows(); ++row)
  {
    for (int j = Aouter[row]; j < Aouter[row + 1]; ++j)
    {
      int col;
      if (Ainner[j] >= col_local_size)
        col = map_old_to_new[Ainner[j] - col_local_size] + col_local_size;
      else
        col = Ainner[j];

      assert(col < B.cols());
      assert(row < B.rows());
      assert(row >= 0 and col >= 0);
      vals.push_back(Eigen::Triplet<double>(row, col, Aval[j]));
    }
  }
  c = 0;
  for (int row = row_map->local_size(false); row < row_map->local_size(true);
       ++row)
  {
    for (int j = 0; j < nnz[row]; ++j)
    {
      int col = new_col_map->global_to_local(global_index_recv[c]);
      assert(col < B.cols());
      assert(row < B.rows());
      assert(row >= 0 and col >= 0);
      vals.push_back(Eigen::Triplet<double>(row, col, global_value_recv[c]));
      ++c;
    }
  }
  assert(c == nnz_sum);

  B.setFromTriplets(vals.begin(), vals.end());

  s << "B.rows() = " << B.rows() << " \n";
  s << "B.cols() = " << B.cols() << " \n";
  s << new_col_map->local_size(false) << ", " << new_col_map->local_size(true)
    << "\n";

  std::cout << s.str();

  return spmv::Matrix<double>(B, new_col_map, new_row_map);
}
