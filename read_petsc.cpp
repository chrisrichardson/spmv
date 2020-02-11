// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "read_petsc.h"
#include "L2GMap.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

// Divide size into N ~equal chunks
std::vector<std::int32_t> owner_ranges(int size, std::int64_t N)
{
  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<std::int32_t> ranges;
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
std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
           std::shared_ptr<L2GMap>>
read_petsc_binary(MPI_Comm comm, std::string filename)
{
  Eigen::SparseMatrix<double, Eigen::RowMajor> A;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::map<std::int32_t, std::int32_t> col_indices;
  std::int64_t nrows_local;
  std::vector<std::int32_t> ranges;

  std::ifstream file(filename.c_str(),
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open())
  {
    // Get first 4 ints from file
    std::vector<char> memblock(16);
    file.seekg(0, std::ios::beg);
    file.read(memblock.data(), 16);

    char* ptr = memblock.data();
    for (int i = 0; i < 4; ++i)
    {
      std::swap(*ptr, *(ptr + 3));
      std::swap(*(ptr + 1), *(ptr + 2));
      ptr += 4;
    }

    std::int32_t* int_data = (std::int32_t*)(memblock.data());
    int id = int_data[0];
    if (id != 1211216)
      throw std::runtime_error("Bad signature in PETSc Matrix file");

    int nrows = int_data[1];
    int ncols = int_data[2];
    int nnz_tot = int_data[3];
    ranges = owner_ranges(mpi_size, nrows);

    if (mpi_rank == 0)
      std::cout << "Read file: " << filename << ": " << nrows << "x" << ncols
                << " = " << nnz_tot << "\n";

    nrows_local = ranges[mpi_rank + 1] - ranges[mpi_rank];

    // Reset memory block and read nnz per row for all rows
    memblock.resize(nrows * 4);
    ptr = memblock.data();
    file.read(memblock.data(), nrows * 4);
    std::vector<std::int32_t> nnz(nrows);
    std::int64_t nnz_sum = 0;
    for (int i = 0; i < nrows; ++i)
    {
      std::swap(*ptr, *(ptr + 3));
      std::swap(*(ptr + 1), *(ptr + 2));
      nnz[i] = *((std::int32_t*)ptr);
      nnz_sum += nnz[i];
      ptr += 4;
    }
    assert(nnz_sum == nnz_tot);

    // Get offset and size for data
    std::int64_t nnz_offset = 0;
    std::int64_t nnz_size = 0;
    for (std::int64_t i = 0; i < ranges[mpi_rank]; ++i)
      nnz_offset += nnz[i];
    for (std::int64_t i = ranges[mpi_rank]; i < ranges[mpi_rank + 1]; ++i)
      nnz_size += nnz[i];

    std::streampos value_data_pos
        = file.tellg() + (std::streampos)(nnz_tot * 4 + nnz_offset * 8);

    // Read col indices for each row
    memblock.resize(nnz_size * 4);
    ptr = memblock.data();
    file.seekg(nnz_offset * 4, std::ios::cur);
    file.read(memblock.data(), nnz_size * 4);

    std::int32_t c = 0;
    // Assuming square matrix... fill local indices first
    // FIXME: rectangular matrix
    for (std::int64_t row = ranges[mpi_rank]; row < ranges[mpi_rank + 1]; ++row)
    {
      col_indices.insert({row, c});
      ++c;
    }

    // Map other columns
    for (std::int64_t row = ranges[mpi_rank]; row < ranges[mpi_rank + 1]; ++row)
    {
      for (std::int64_t j = 0; j < nnz[row]; ++j)
      {
        std::swap(*ptr, *(ptr + 3));
        std::swap(*(ptr + 1), *(ptr + 2));
        if (col_indices.insert({*((std::int32_t*)ptr), c}).second)
          ++c;
        ptr += 4;
      }
    }

    A.resize(nrows_local, col_indices.size());

    // Read values
    std::vector<char> valuedata(nnz_size * 8);
    file.seekg(value_data_pos, std::ios::beg);
    file.read(valuedata.data(), nnz_size * 8);
    file.close();

    // Pointer to values
    char* vptr = valuedata.data();
    ptr = memblock.data();
    for (std::int64_t row = ranges[mpi_rank]; row < ranges[mpi_rank + 1]; ++row)
    {
      for (std::int64_t j = 0; j < nnz[row]; ++j)
      {
        std::swap(*vptr, *(vptr + 7));
        std::swap(*(vptr + 1), *(vptr + 6));
        std::swap(*(vptr + 2), *(vptr + 5));
        std::swap(*(vptr + 3), *(vptr + 4));
        double val = *((double*)vptr);
        vptr += 8;

        // Look up column local index
        std::int32_t col = col_indices[*((std::int32_t*)ptr)];
        ptr += 4;

        A.insert(row - ranges[mpi_rank], col) = val;
      }
    }
  }
  else
    throw std::runtime_error("Could not open file");

  A.makeCompressed();

  std::vector<index_type> ghosts(col_indices.size() - nrows_local);
  for (auto& q : col_indices)
    if (q.first < ranges[mpi_rank] or q.first >= ranges[mpi_rank + 1])
      ghosts[q.second - nrows_local] = q.first;

  auto l2g = std::make_shared<L2GMap>(comm, ranges, ghosts);

  return {A, l2g};
}
