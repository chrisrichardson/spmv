// Copyright (C) 2020 Chris Richardson, Igor Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <memory>
#include <mpi.h>
#include <oneapi/mkl.hpp>
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
  /// Create distributed sparse matrix from buffers and local-to-global maps
  /// (data might be on the device memory)
  Matrix(std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> row_ptr,
         std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> col_ind,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// Create distributed sparse matrix from vectors and local-to-global maps
  /// (data is still on the host memory)
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

  /// Set queue to submit work (indirectly defines the device in each rank)
  void set_queue(cl::sycl::queue q) { _q = q; }

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
