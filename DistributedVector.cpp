// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "DistributedVector.h"
#include <iostream>

DistributedVector::DistributedVector() {}
//-----------------------------------------------------------------------------
Eigen::Map<Eigen::VectorXd> DistributedVector::vec()
{
  return Eigen::Map<Eigen::VectorXd>(_xsp.valuePtr() + _i0, _local_size);
}
//-----------------------------------------------------------------------------
Eigen::SparseVector<double>& DistributedVector::spvec() { return _xsp; }
//-----------------------------------------------------------------------------
void DistributedVector::setup(
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

  std::cout << "# local_size[" << rank << "] = " << _local_size << "/" << N
            << "\n";
  _xsp.resize(N);

  // Insert all local rows/cols
  for (index_type i = r0; i != r1; ++i)
    _xsp.coeffRef(i) = 1.0;

  // Look for any other columns with non-zeros
  const index_type nmax = *(A.outerIndexPtr() + A.rows());
  for (auto ptr = A.innerIndexPtr(); ptr != A.innerIndexPtr() + nmax; ++ptr)
    _xsp.coeffRef(*ptr) = 1.0;

  // Find index of r0 - location of x within _xsp
  index_type* iptr = _xsp.innerIndexPtr();
  while (*iptr != r0)
    ++iptr;
  _i0 = iptr - _xsp.innerIndexPtr();

  // Get indices of filled values from xsp in each range "what this process
  // needs" - and send indices to each process

  // Calculate NNZ in each range
  _counts[1].resize(mpi_size, 0);
  int* dptr = _xsp.innerIndexPtr();
  int* dptr_end = dptr + _xsp.nonZeros();
  int r = 0;
  for (auto d = dptr; d != dptr_end; ++d)
  {
    while (*d >= ranges[r + 1])
      ++r;
    ++_counts[1][r];
  }

  // Send NNZs by Alltoall - these will be the receive counts for incoming
  // index/values
  _counts[0].resize(mpi_size);
  MPI_Alltoall(_counts[1].data(), 1, MPI_INT, _counts[0].data(), 1, MPI_INT,
               comm);

  _offsets[1].resize(mpi_size, 0);
  for (unsigned int i = 1; i != mpi_size; ++i)
    _offsets[1][i] = _offsets[1][i - 1] + _counts[1][i - 1];

  // No need to send data to self, but keep offsets in place for [1]
  _counts[0][rank] = 0;
  _counts[1][rank] = 0;

  _offsets[0].resize(mpi_size, 0);
  for (unsigned int i = 1; i != mpi_size; ++i)
    _offsets[0][i] = _offsets[0][i - 1] + _counts[0][i - 1];
  unsigned int count = _offsets[0].back() + _counts[0].back();

  _indexbuf.resize(count);
  _send_data.resize(count);

  int err = MPI_Alltoallv(_xsp.innerIndexPtr(), _counts[1].data(),
                          _offsets[1].data(), MPI_INT, _indexbuf.data(),
                          _counts[0].data(), _offsets[0].data(), MPI_INT, comm);
}
//-----------------------------------------------------------------------------
void DistributedVector::update(MPI_Comm comm)
{
  for (index_type i = 0; i != _indexbuf.size(); ++i)
    _send_data[i] = _xsp.coeffRef(_indexbuf[i]);

  // Send actual values
  int err = MPI_Alltoallv(
      _send_data.data(), _counts[0].data(), _offsets[0].data(), MPI_DOUBLE,
      _xsp.valuePtr(), _counts[1].data(), _offsets[1].data(), MPI_DOUBLE, comm);
}
//-----------------------------------------------------------------------------
