// Copyright (C) 2020 Chris Richardson, Igor Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>
#include <memory>
#include <mkl_sycl.hpp>
#include <mpi.h>
#include <vector>

#include <Eigen/Sparse>

#include "Vector.h"

#pragma once

/// Simple Distributed Sparse Linear Algebra Library
namespace spmv
{

class L2GMap;

template <typename ScalarType>
class Matrix
{
  /// Matrix with row and column maps.
public:
  Matrix(std::array<std::size_t, 2> shape,
         std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> row_ptr,
         std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> col_ind,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  Matrix(std::vector<ScalarType>& data, std::vector<std::int32_t>& row_ptr,
         std::vector<std::int32_t>& col_ind,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  /// Destructor (destroys MKL structs)
  ~Matrix();

  /// MatVec operator for A x
  spmv::Vector<ScalarType> operator*(spmv::Vector<ScalarType>& b) const
  {
    std::size_t ls = _row_map->local_size() + _row_map->num_ghosts();
    auto y = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(
        cl::sycl::range<1>{ls});

    oneapi::mkl::sparse::gemv(_q, oneapi::mkl::transpose::nontrans, 1.0,
                              A_onemkl, b.getLocalData(), 0.0, *y);
    return spmv::Vector<ScalarType>(y, _row_map);
  };

  /// MatVec operator for A^T x
  spmv::Vector<ScalarType> transpmult(spmv::Vector<ScalarType>& b) const
  {
    auto y = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(
        cl::sycl::range<1>{_shape[0]});
    oneapi::mkl::sparse::gemv(_q, oneapi::mkl::transpose::trans, 1.0, A_onemkl,
                              b.getLocalData(), 0.0, *y);
    return spmv::Vector<ScalarType>(y, _row_map);
  };

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<L2GMap> col_map() const { return _col_map; }

private:
  void mkl_init()
  {
    oneapi::mkl::sparse::init_matrix_handle(&A_onemkl);
    oneapi::mkl::sparse::set_csr_data(A_onemkl, _shape[0], _shape[1],
                                      oneapi::mkl::index_base::zero, *_indptr,
                                      *_indices, *_data);
  }

  mutable cl::sycl::queue _q;
  oneapi::mkl::sparse::matrix_handle_t A_onemkl;

  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;
  std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> _indptr;
  std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> _indices;

  std::array<std::size_t, 2> _shape;

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<spmv::L2GMap> _col_map;
  std::shared_ptr<spmv::L2GMap> _row_map;
};
} // namespace spmv
