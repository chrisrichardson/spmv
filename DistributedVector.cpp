// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedVector.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <vector>

//-----------------------------------------------------------------------------
DistributedVector::DistributedVector(
    MPI_Comm comm, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A)
{
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  // Get local range from number of rows in A
  std::vector<int> nrows_all(mpi_size);
  std::vector<index_type> ranges = {0};
  int nrows = A.rows();
  MPI_Allgather(&nrows, 1, MPI_INT, nrows_all.data(), 1, MPI_INT, comm);
  for (int i = 0; i < mpi_size; ++i)
    ranges.push_back(ranges.back() + nrows_all[i]);

  index_type r0 = ranges[mpi_rank];
  index_type r1 = ranges[mpi_rank + 1];
  index_type N = ranges.back();
  _local_size = r1 - r0;
  assert(_local_size = nrows);

  std::stringstream s;
  s << "# local_size[" << mpi_rank << "] = " << _local_size << "/" << N
            << "\n";
  std::cout << s.str();
  
  _xsp.resize(N);

  // Look for all columns with non-zeros - insert 1.0 (a non zero)
  const index_type nmax = *(A.outerIndexPtr() + A.rows());
  std::set<index_type> valid_cols;
  for (auto ptr = A.innerIndexPtr(); ptr != A.innerIndexPtr() + nmax; ++ptr)
    valid_cols.insert(*ptr);

  // Insert all local rows/cols - (in case any got missed...?)
  for (index_type i = r0; i < r1; ++i)
    valid_cols.insert(i);

  for (index_type q : valid_cols)
    _xsp.coeffRef(q) = 1.0;

  // Set to zero without removing non-zeros (!)
  _xsp *= 0;

  // Finished filling

  // Get indices of filled values from xsp in each range "what this process
  // needs" - and send indices to each process

  // Calculate NNZ in each range

  std::vector<int> neighbours;
  _i0 = -1;

  index_type* dptr = _xsp.innerIndexPtr();
  index_type* dptr_end = dptr + _xsp.nonZeros();
  index_type* d0 = dptr;
  for (std::size_t i = 0; i < ranges.size(); ++i)
  {
    index_type* d1 = std::lower_bound(d0, dptr_end, ranges[i]);

    if ((int)i == mpi_rank)
      _i0 = d1 - dptr;

    if (d1 > d0 and (int)(i - 1) != mpi_rank)
    {
      neighbours.push_back(i - 1);
      _send_count.push_back(d1 - d0);
    }

    d0 = d1;
  }
  assert(_i0 != -1);

  const int neighbour_size = neighbours.size();

  MPI_Dist_graph_create_adjacent(comm, neighbours.size(), neighbours.data(),
                                 MPI_UNWEIGHTED, neighbours.size(),
                                 neighbours.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &_neighbour_comm);

  // Send NNZs by Alltoall - these will be the receive counts for incoming
  // index/values
  _recv_count.resize(neighbour_size);
  MPI_Neighbor_alltoall(_send_count.data(), 1, MPI_INT, _recv_count.data(), 1,
                        MPI_INT, _neighbour_comm);

  _send_offset = {0};
  for (int c : _send_count)
    _send_offset.push_back(_send_offset.back() + c);
  for (int i = 0; i < neighbour_size; ++i)
    if (neighbours[i] > mpi_rank)
      _send_offset[i] += _local_size;

  _recv_offset = {0};
  for (int c : _recv_count)
    _recv_offset.push_back(_recv_offset.back() + c);
  int count = _recv_offset.back();

  _indexbuf.resize(count);
  _send_data.resize(count);

  // Send global indices to remote processes that own them
  int err = MPI_Neighbor_alltoallv(
      _xsp.innerIndexPtr(), _send_count.data(), _send_offset.data(), MPI_INT,
      _indexbuf.data(), _recv_count.data(), _recv_offset.data(), MPI_INT,
      _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");

  // Should be in own range
  for (index_type i : _indexbuf)
    assert(i >= r0 and i < r1);
}
//-----------------------------------------------------------------------------
DistributedVector::~DistributedVector() { MPI_Comm_free(&_neighbour_comm); }
//-----------------------------------------------------------------------------
Eigen::Map<Eigen::VectorXd> DistributedVector::vec()
{
  return Eigen::Map<Eigen::VectorXd>(_xsp.valuePtr() + _i0, _local_size);
}
//-----------------------------------------------------------------------------
Eigen::SparseVector<double>& DistributedVector::spvec() { return _xsp; }
//-----------------------------------------------------------------------------
void DistributedVector::update()
{
  // Get data from global indices to send to other processes
  for (std::size_t i = 0; i < _indexbuf.size(); ++i)
    _send_data[i] = _xsp.coeffRef(_indexbuf[i]);

  // Send actual values - NB meaning of _send and _recv count/offset is reversed
  int err = MPI_Neighbor_alltoallv(
      _send_data.data(), _recv_count.data(), _recv_offset.data(), MPI_DOUBLE,
      _xsp.valuePtr(), _send_count.data(), _send_offset.data(), MPI_DOUBLE,
      _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");
}
//-----------------------------------------------------------------------------
