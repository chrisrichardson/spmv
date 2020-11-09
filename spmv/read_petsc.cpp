// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "L2GMap.h"
#include "read_petsc.h"
#include "utils.h"

//-----------------------------------------------------------------------------
spmv::Matrix<double> spmv::read_petsc_binary(MPI_Comm comm,
                                             std::string filename)
{
  // COO data structure
  std::vector<double> coo_data;
  std::vector<std::int32_t> coo_row;
  std::vector<std::int32_t> coo_col;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::map<std::int32_t, std::int32_t> col_indices;
  std::vector<std::int64_t> row_ranges, col_ranges;
  std::int64_t ncols_local, nrows_local;

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
    row_ranges = owner_ranges(mpi_size, nrows);
    col_ranges = owner_ranges(mpi_size, ncols);

    if (mpi_rank == 0)
      std::cout << "Read file: " << filename << ": " << nrows << "x" << ncols
                << " = " << nnz_tot << "\n";

    nrows_local = row_ranges[mpi_rank + 1] - row_ranges[mpi_rank];
    ncols_local = col_ranges[mpi_rank + 1] - col_ranges[mpi_rank];

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
    for (std::int64_t i = 0; i < row_ranges[mpi_rank]; ++i)
      nnz_offset += nnz[i];
    for (std::int64_t i = row_ranges[mpi_rank]; i < row_ranges[mpi_rank + 1];
         ++i)
      nnz_size += nnz[i];

    std::streampos value_data_pos
        = file.tellg() + (std::streampos)(nnz_tot * 4 + nnz_offset * 8);

    // Read col indices for each row
    memblock.resize(nnz_size * 4);
    ptr = memblock.data();
    file.seekg(nnz_offset * 4, std::ios::cur);
    file.read(memblock.data(), nnz_size * 4);

    std::int32_t c = 0;
    for (std::int64_t col = col_ranges[mpi_rank];
         col < col_ranges[mpi_rank + 1]; ++col)
    {
      col_indices.insert({col, c});
      ++c;
    }

    // Map other columns
    for (std::int64_t row = row_ranges[mpi_rank];
         row < row_ranges[mpi_rank + 1]; ++row)
    {
      for (std::int64_t j = 0; j < nnz[row]; ++j)
      {
        std::swap(*ptr, *(ptr + 3));
        std::swap(*(ptr + 1), *(ptr + 2));
        col_indices.insert({*((std::int32_t*)ptr), -1});
        ptr += 4;
      }
    }
    // Ensure they are labelled in ascending order
    for (auto& q : col_indices)
      if (q.second == -1)
      {
        q.second = c;
        ++c;
      }

    // Read values
    std::vector<char> valuedata(nnz_size * 8);
    file.seekg(value_data_pos, std::ios::beg);
    file.read(valuedata.data(), nnz_size * 8);
    file.close();

    // Pointer to values
    char* vptr = valuedata.data();
    ptr = memblock.data();
    for (std::int64_t row = row_ranges[mpi_rank];
         row < row_ranges[mpi_rank + 1]; ++row)
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
        coo_data.push_back(val);
        coo_row.push_back(row - row_ranges[mpi_rank]);
        coo_col.push_back(col);
      }
    }
  }
  else
    throw std::runtime_error("Could not open file");

  std::vector<std::int64_t> ghosts(col_indices.size() - ncols_local);
  for (auto& q : col_indices)
    if (q.first < col_ranges[mpi_rank] or q.first >= col_ranges[mpi_rank + 1])
      ghosts[q.second - ncols_local] = q.first;

  auto col_map = std::make_shared<spmv::L2GMap>(comm, ncols_local, ghosts);
  auto row_map = std::make_shared<spmv::L2GMap>(comm, nrows_local);

  auto [data, indptr, indices]
      = coo_to_csr<double>(nrows_local, col_indices.size(), coo_data.size(),
                           coo_row, coo_col, coo_data);

  for (std::int32_t i = 0; i < 20; i++)
    std::cout << indptr[i] << " ";

    return spmv::Matrix<double>(data, indptr, indices, col_map, row_map);
}
//-----------------------------------------------------------------------------
std::vector<double> spmv::read_petsc_binary_vector(MPI_Comm comm,
                                                   std::string filename)
{
  std::vector<double> vec;

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::ifstream file(filename.c_str(),
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open())
  {
    // Get first 2 ints from file
    std::vector<char> memblock(8);
    file.seekg(0, std::ios::beg);
    file.read(memblock.data(), 8);

    char* ptr = memblock.data();
    for (int i = 0; i < 2; ++i)
    {
      std::swap(*ptr, *(ptr + 3));
      std::swap(*(ptr + 1), *(ptr + 2));
      ptr += 4;
    }

    std::int32_t* int_data = (std::int32_t*)(memblock.data());
    int id = int_data[0];
    if (id != 1211214)
      throw std::runtime_error("Bad signature in PETSc Vector file");

    int nrows = int_data[1];
    std::vector<std::int64_t> ranges = owner_ranges(mpi_size, nrows);

    if (mpi_rank == 0)
      std::cout << "Read vector file: " << filename << ": " << nrows << "\n";

    std::int64_t nrows_local = ranges[mpi_rank + 1] - ranges[mpi_rank];

    vec.resize(nrows_local);

    std::streampos value_data_pos
        = file.tellg() + (std::streampos)(ranges[mpi_rank] * 8);

    // Read values
    std::vector<char> valuedata(nrows_local * 8);
    file.seekg(value_data_pos, std::ios::beg);
    file.read(valuedata.data(), nrows_local * 8);
    file.close();

    // Pointer to values
    char* vptr = valuedata.data();

    for (std::int64_t row = ranges[mpi_rank]; row < ranges[mpi_rank + 1]; ++row)
    {
      std::swap(*vptr, *(vptr + 7));
      std::swap(*(vptr + 1), *(vptr + 6));
      std::swap(*(vptr + 2), *(vptr + 5));
      std::swap(*(vptr + 3), *(vptr + 4));
      double val = *((double*)vptr);
      vptr += 8;
      vec[row - ranges[mpi_rank]] = val;
    }
  }
  else
    throw std::runtime_error("Could not open file");

  return vec;
}
//-----------------------------------------------------------------------------
