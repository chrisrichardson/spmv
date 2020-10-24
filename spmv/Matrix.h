// Copyright (C) 2020 Chris Richardson, Igor Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>
#include <memory>
#include <mkl_sycl.hpp>
#include <mpi.h>
#include <vector>

#include "Vector.h"

#pragma once

/// Simple Distributed Sparse Linear Algebra Library
namespace spmv
{

class L2GMap;

template <typename ScalarType, typename IndType>
class Matrix
{
  /// Matrix with row and column maps.
public:
  Matrix(std::array<std::size_t, 2> shape,
         std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<cl::sycl::buffer<IndType, 1>> indptr,
         std::shared_ptr<cl::sycl::buffer<IndType, 1>> indices,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// Destructor (destroys MKL structs, if using MKL)
  ~Matrix();

  /// MatVec operator for A x
  spmv::Vector<ScalarType> operator*(spmv::Vector<ScalarType>& b) const
  {
    std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> y(
        cl::sycl::range<1>{_shape[0]});
    oneapi::mkl::sparse::gemv(_q, oneapi::mkl::transpose::nontrans, 1.0,
                              A_onemkl, b.getLocalData(), 0.0, *y);
    return spmv::Vector<ScalarType>(y, _row_map);
  };

  /// MatVec operator for A^T x
  spmv::Vector<ScalarType>
  transpmult(const cl::sycl::buffer<ScalarType, 1>& b) const
  {
    std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> y(
        cl::sycl::range<1>{_shape[0]});
    oneapi::mkl::sparse::gemv(_q, oneapi::mkl::transpose::trans, 1.0, A_onemkl,
                              b.getLocalData(), 0.0, *y);
    return spmv::Vector<ScalarType>(y, _row_map);
  };

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<const L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

private:
  void mkl_init();

  mutable cl::sycl::queue _q;
  oneapi::mkl::sparse::matrix_handle_t A_onemkl;

  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;
  std::shared_ptr<cl::sycl::buffer<IndType, 1>> _indptr;
  std::shared_ptr<cl::sycl::buffer<IndType, 1>> _indices;

  std::array<std::size_t, 2> _shape;

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<spmv::L2GMap> _col_map;
  std::shared_ptr<spmv::L2GMap> _row_map;
};
} // namespace spmv
