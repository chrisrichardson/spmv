// Copyright (C) 2018 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <iostream>
#include <cmath>
#include <mpi.h>
#include <set>

#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::SparseMatrix<double>::StorageIndex index_type;

class DistributedVector
{
public:

  DistributedVector()
  {}

  // Local "dense" portion of sparse vector
  Eigen::Map<Eigen::VectorXd> vec()
  {
    return Eigen::Map<Eigen::VectorXd>(_xsp.valuePtr() + _i0, _local_size);
  }

  Eigen::SparseVector<double>& spvec()
  {
    return _xsp;
  }

  // Set up communication pattern for A.x by querying columns of A for non-zeros
  // and sending send-pattern to remotes
  void setup(MPI_Comm comm, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, std::vector<index_type>& ranges)
  {

    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &rank);
    index_type r0 = ranges[rank];
    index_type r1 = ranges[rank+1];
    index_type N = ranges.back();
    _local_size = r1 - r0;

    std::cout << "# local_size[" << rank << "] = " << _local_size << "/" << N << "\n";
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

    // Get indices of filled values from xsp in each range "what this process needs" - and send indices to each process

    // Calculate NNZ in each range
    _counts[1].resize(mpi_size, 0);
    int *dptr = _xsp.innerIndexPtr();
    int *dptr_end = dptr + _xsp.nonZeros();
    int r = 0;
    for (auto d = dptr; d != dptr_end; ++d)
      {
	while (*d >= ranges[r + 1])
	  ++r;
	++_counts[1][r];
      }

    // Send NNZs by Alltoall - these will be the receive counts for incoming index/values
    _counts[0].resize(mpi_size);
    MPI_Alltoall(_counts[1].data(), 1, MPI_INT, _counts[0].data(), 1, MPI_INT,  comm);

    _offsets[1].resize(mpi_size, 0);
    for (unsigned int i = 1; i != mpi_size; ++i)
	_offsets[1][i] = _offsets[1][i-1] + _counts[1][i-1];

    // No need to send data to self, but keep offsets in place for [1]
    _counts[0][rank] = 0;
    _counts[1][rank] = 0;

    _offsets[0].resize(mpi_size, 0);
    for (unsigned int i = 1; i != mpi_size; ++i)
      _offsets[0][i] = _offsets[0][i-1] + _counts[0][i-1];
    unsigned int count = _offsets[0].back() + _counts[0].back();

    _indexbuf.resize(count);
    _send_data.resize(count);

    int err = MPI_Alltoallv(_xsp.innerIndexPtr(), _counts[1].data(),
			    _offsets[1].data(), MPI_INT,
			    _indexbuf.data(), _counts[0].data(),
			    _offsets[0].data(), MPI_INT, comm);
  }

  void update(MPI_Comm comm)
  {
    for (index_type i = 0; i != _indexbuf.size(); ++i)
      _send_data[i] = _xsp.coeffRef(_indexbuf[i]);

    // Send actual values
    int err = MPI_Alltoallv(_send_data.data(), _counts[0].data(), _offsets[0].data(), MPI_DOUBLE,
                              _xsp.valuePtr(), _counts[1].data(), _offsets[1].data(), MPI_DOUBLE, comm);
  }

private:
  // Actual data
  Eigen::SparseVector<double> _xsp;

  // Indices, counts and offsets for communication
  std::vector<index_type> _indexbuf;
  std::vector<index_type> _counts[2];
  std::vector<index_type> _offsets[2];

  // Data buffer for sending to remotes
  std::vector<double> _send_data;

  // Address and size of "local" entries in sparse vector
  index_type _i0;
  index_type _local_size;

  // MPI rank
  int rank;
};


std::vector<index_type> owner_ranges(MPI_Comm comm, index_type N)
{
  // Work out ranges for all processes
  int size;
  MPI_Comm_size(comm, &size);

  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<index_type> ranges;
  for (int rank = 0; rank != (size+1); ++rank)
  {
    if (rank < r)
      ranges.push_back(rank*(n + 1));
    else
      ranges.push_back(rank*n + r);
  }

  return ranges;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << "# rank = " << rank << "/" << size <<  "\n";

  // Make a square Matrix divided evenly across cores
  int N = 1000;

  std::vector<index_type> ranges = owner_ranges(MPI_COMM_WORLD, N);

  index_type r0 = ranges[rank];
  index_type r1 = ranges[rank + 1];
  int M = r1 - r0;

  std::cout << "# " << r0 << "-" << r1 <<" \n";

  // Local part of the matrix
  // Must be RowMajor and compressed
  Eigen::SparseMatrix<double, Eigen::RowMajor> A(M, N);

  // Set up A
  for (unsigned int i = 0; i < M; ++i)
  {
    if ((r0 + i) == 0)
      A.insert(i, i) = 1.0;
    else if ((r0 + i) == (N-1))
      A.insert(i, i) = 1.0;
    else
    {
      A.insert(i, r0 + i - 1) = 1.0;
      A.insert(i, r0 + i)    = -2.0;
      A.insert(i, r0 + i + 1) = 1.0;
    }

  }
  A.makeCompressed();

  // Make distributed vector - this is the only
  // one that needs to be 'sparse'
  DistributedVector psp;
  psp.setup(MPI_COMM_WORLD, A, ranges);
  auto p = psp.vec();

  // RHS vector
  Eigen::VectorXd b(M);

  // Set up values
  for (unsigned int i = 0; i != M; ++i)
  {
    double z = (double)(i+r0)/double(N);
    b[i] = exp(-10*pow(5*(z-0.5), 2.0));
  }

  // Residual vector
  Eigen::VectorXd r(M);
  r = b;
  // Assign to dense part of sparse vector
  p = r;
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(M);

  // Iterations of CG
  for (unsigned int k = 0; k != 500; ++k)
  {
    double rnorm = r.squaredNorm();

    // y = A.p
    psp.update(MPI_COMM_WORLD);
    y = A*psp.spvec();

    // Update x and r
    double alpha = rnorm/p.dot(y);
    x += alpha*p;
    r -= alpha*y;

    // Update p
    double beta = r.squaredNorm()/rnorm;
    p *= beta;
    p += r;
    std::cerr << k << ":" << rnorm << "\n";
  }

  // Output
  std::stringstream s;
  for (unsigned int i = 0; i != M; ++i)
    s << x[i] << "\n";

  for (unsigned int i = 0; i != size; ++i)
  {
    if (i == rank)
      std::cout << s.str() << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
