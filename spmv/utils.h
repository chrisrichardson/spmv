// Copyright (C) 2020 Igor Baratta (ia397@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <cstdint>
#include <numeric>
#include <vector>

namespace spmv
{

// Divide size into N ~equal chunks
std::vector<std::int64_t> owner_ranges(int size, std::int64_t N)
{
  // Compute number of items per process and remainder
  const std::int64_t n = N / size;
  const std::int64_t r = N % size;

  // Compute local range
  std::vector<std::int64_t> ranges;
  for (int rank = 0; rank < (size + 1); ++rank)
  {
    if (rank < r)
      ranges.push_back(rank * (n + 1));
    else
      ranges.push_back(rank * n + r);
  }

  return ranges;
}

/// Simple csr data sctructure used for structured binding in C++ 17
template <typename T>
struct csr_data
{
  std::vector<T> data;
  std::vector<std::int32_t> indptr;
  std::vector<std::int32_t> indices;
};

template <class T>
csr_data<T>
coo_to_csr(const std::int32_t nrows, const std::int32_t ncols,
           const std::int32_t nnz, std::vector<std::int32_t> coo_row,
           std::vector<std::int32_t> coo_col, std::vector<T> coo_data)
{
  std::vector<T> data(nnz);
  std::vector<std::int32_t> indptr(nrows + 1);
  std::vector<std::int32_t> indices(nnz);

  // Compute number of non-zero entries per row of A
  std::vector<std::int32_t> nnz_row(nrows);
  for (std::int32_t i = 0; i < nnz; i++)
    nnz_row[coo_row[i]]++;

  std::exclusive_scan(nnz_row.begin(), nnz_row.end(), indptr.begin(), 0);
  indptr[nrows] = nnz;

  std::fill(nnz_row.begin(), nnz_row.end(), 0);
  for (std::int32_t i = 0; i < nnz; i++)
  {
    std::int32_t row = coo_row[i];
    std::int32_t pos = nnz_row[row] + indptr[row];

    data[pos] = coo_data[i];
    indices[pos] = coo_col[i];

    nnz_row[row]++;
  }

  return {data, indptr, indices};
}

} // namespace spmv