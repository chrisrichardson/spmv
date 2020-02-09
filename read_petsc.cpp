#include "read_petsc.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

Eigen::SparseMatrix<double, Eigen::RowMajor>
read_petsc_binary(MPI_Comm comm, std::string filename)
{
  std::streampos size;
  Eigen::SparseMatrix<double, Eigen::RowMajor> A;

  std::ifstream file(filename.c_str(),
                     std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    std::vector<char> memblock(size);
    file.seekg(0, std::ios::beg);
    file.read(memblock.data(), size);
    file.close();
    char* ptr = memblock.data();
    for (int i = 0; i < 4; ++i)
    {
      std::swap(*ptr, *(ptr + 3));
      std::swap(*(ptr + 1), *(ptr + 2));
      ptr += 4;
    }

    std::int32_t* int_data = (std::int32_t*)(memblock.data());
    int id = int_data[0];
    int nrows = int_data[1];
    int ncols = int_data[2];
    int nnz_tot = int_data[3];
    std::cout << nrows << "x" << ncols << " = " << nnz_tot << "\n";
    A.resize(nrows, ncols);

    std::vector<std::int32_t> nnz(nrows);
    int nnz_sum = 0;
    for (int i = 0; i < nrows; ++i)
    {
      std::swap(*ptr, *(ptr + 3));
      std::swap(*(ptr + 1), *(ptr + 2));
      nnz[i] = *((std::int32_t*)ptr);
      nnz_sum += nnz[i];
      ptr += 4;
    }
    assert(nnz_sum == nnz_tot);

    // Pointer to values
    char* vptr = ptr + 4 * nnz_tot;

    std::vector<std::int32_t> cols;
    std::vector<double> vals;
    for (int row = 0; row < nrows; ++row)
    {
      for (int j = 0; j < nnz[row]; ++j)
      {
        std::swap(*ptr, *(ptr + 3));
        std::swap(*(ptr + 1), *(ptr + 2));
        std::int32_t col = *((std::int32_t*)ptr);
        cols.push_back(col);
        ptr += 4;

        std::swap(*vptr, *(vptr + 7));
        std::swap(*(vptr + 1), *(vptr + 6));
        std::swap(*(vptr + 2), *(vptr + 5));
        std::swap(*(vptr + 3), *(vptr + 4));
        double val = *((double*)vptr);
        vals.push_back(val);
        vptr += 8;

        A.insert(row, col) = val;
      }
    }
    std::cout << cols.back() << "\n" << cols.size() << "\n";

    std::cout << vals[0] << "\n";
  }
  else
    throw std::runtime_error("Could not open file");

  A.makeCompressed();
  return A;
}
