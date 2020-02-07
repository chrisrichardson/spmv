// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedVector.h"
#include <iostream>

DistributedVector::DistributedVector(
    MPI_Comm comm, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
    std::vector<index_type>& ranges)
{
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  index_type r0 = ranges[rank];
  index_type r1 = ranges[rank + 1];
  index_type N = ranges.back();
  _local_size = r1 - r0;
  assert(_local_size = A.rows());

  std::cout << "# local_size[" << rank << "] = " << _local_size << "/" << N
            << "\n";
  _xsp.resize(N);

  // Look for all columns with non-zeros - insert 1.0 (a non zero)
  const index_type nmax = *(A.outerIndexPtr() + A.rows());
  for (auto ptr = A.innerIndexPtr(); ptr != A.innerIndexPtr() + nmax; ++ptr)
    _xsp.coeffRef(*ptr) = 1.0;

  // Insert all local rows/cols - (in case any got missed...?)
  for (index_type i = r0; i != r1; ++i)
    _xsp.coeffRef(i) = 1.0;

  // finished filling

  index_type* dptr = _xsp.innerIndexPtr();
  index_type* dptr_end = dptr + _xsp.nonZeros();

  // Find index of r0 - location of dense x within _xsp
  index_type* i0_ptr = std::find(dptr, dptr_end, r0);
  assert(i0_ptr != dptr_end);
  _i0 = i0_ptr - dptr;

  // Get indices of filled values from xsp in each range "what this process
  // needs" - and send indices to each process

  // Calculate NNZ in each range
  _send_count.resize(mpi_size, 0);
  int r = 0;
  for (index_type* d = dptr; d != dptr_end; ++d)
  {
    while (*d >= ranges[r + 1])
      ++r;
    ++_send_count[r];
  }

  // Send NNZs by Alltoall - these will be the receive counts for incoming
  // index/values
  _recv_count.resize(mpi_size);
  MPI_Alltoall(_send_count.data(), 1, MPI_INT, _recv_count.data(), 1, MPI_INT,
               comm);

  _send_offset.resize(mpi_size, 0);
  for (unsigned int i = 1; i != mpi_size; ++i)
    _send_offset[i] = _send_offset[i - 1] + _send_count[i - 1];

  // No need to send data to self, but keep offsets in place for send
  _recv_count[rank] = 0;
  _send_count[rank] = 0;

  _recv_offset = {0};
  for (int c : _recv_count)
    _recv_offset.push_back(_recv_offset.back() + c);
  unsigned int count = _recv_offset.back();

  _indexbuf.resize(count);
  _send_data.resize(count);

  // Send global indices to remote processes that own them
  int err = MPI_Alltoallv(
      _xsp.innerIndexPtr(), _send_count.data(), _send_offset.data(), MPI_INT,
      _indexbuf.data(), _recv_count.data(), _recv_offset.data(), MPI_INT, comm);

  // Should be in own range
  for (index_type i : _indexbuf)
    assert(i >= r0 and i < r1);
}
//-----------------------------------------------------------------------------
Eigen::Map<Eigen::VectorXd> DistributedVector::vec()
{
  return Eigen::Map<Eigen::VectorXd>(_xsp.valuePtr() + _i0, _local_size);
}
//-----------------------------------------------------------------------------
Eigen::SparseVector<double>& DistributedVector::spvec() { return _xsp; }
//-----------------------------------------------------------------------------
void DistributedVector::update(MPI_Comm comm)
{
  // Get data from global indices to send to other processes
  for (index_type i = 0; i != _indexbuf.size(); ++i)
    _send_data[i] = _xsp.coeffRef(_indexbuf[i]);

  // Send actual values - NB meaning of _send and _recv count/offset is reversed
  int err = MPI_Alltoallv(_send_data.data(), _recv_count.data(),
                          _recv_offset.data(), MPI_DOUBLE, _xsp.valuePtr(),
                          _send_count.data(), _send_offset.data(), MPI_DOUBLE,
                          comm);
}
//-----------------------------------------------------------------------------
