// Copyright (C) 2020 Chris Richardson, Igor Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <memory>
#include <mkl_sycl.hpp>
#include <mpi.h>
#include <vector>

#include "L2GMap.h"
#include "Vector.h"

#pragma once

/// Simple Distributed Sparse Linear Algebra Library
namespace spmv
{

class L2GMap;

template <typename ScalarType>
class Matrix
{
  /// CSR Matrix with row and column maps.
public:
  /// Create sparse matrix from buffer (data may be already on device memory)
  Matrix(std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> row_ptr,
         std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> col_ind,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// Create sparse matrix from vectors (data is still on the host memory)
  Matrix(std::vector<ScalarType>& data, std::vector<std::int32_t>& row_ptr,
         std::vector<std::int32_t>& col_ind,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// Destructor (destroys MKL structs)
  ~Matrix();

  /// MatVec operator for A x
  spmv::Vector<ScalarType> operator*(spmv::Vector<ScalarType>& x) const;

  /// MatVec operator for b = A x
  void mult(spmv::Vector<ScalarType>& x, spmv::Vector<ScalarType>& b) const;

  /// MatVec operator for b = A^T x
  void transpmult(spmv::Vector<ScalarType>& x,
                  spmv::Vector<ScalarType>& b) const;

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<L2GMap> col_map() const { return _col_map; }

private:
  void mkl_init();

  mutable cl::sycl::queue _q;
  oneapi::mkl::sparse::matrix_handle_t A_onemkl;

  /// CSR format data buffer of the matrix
  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;

  /// CSR format row pointer buffer of the matrix
  std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> _indptr;

  /// CSR format column indices buffer of the matrix
  std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> _indices;

  /// Shape of the local sparse matrix
  std::array<std::size_t, 2> _shape;

  ///  Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<spmv::L2GMap> _col_map;
  std::shared_ptr<spmv::L2GMap> _row_map;
};
} // namespace spmv
