#include "CreateA.h"
#include "DistributedVector.h" //for index_type

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

Eigen::SparseMatrix<double, Eigen::RowMajor> create_A(MPI_Comm comm, int N)
{
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

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

  return A;
}
//-----------------------------------------------------------------------------