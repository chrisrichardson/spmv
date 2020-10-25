// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    MIT

#include "Matrix.h"
#include "L2GMap.h"
#include "mpi_type.h"
#include <iostream>
#include <numeric>
#include <set>

using namespace spmv;

template <typename ScalarType>
Matrix<ScalarType>::Matrix(
    std::array<std::size_t, 2> shape,
    std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
    std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> indptr,
    std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> indices,
    std::shared_ptr<spmv::L2GMap> col_map,
    std::shared_ptr<spmv::L2GMap> row_map)
    : _data(data), _indptr(indptr), _indices(indices), _shape(shape),
      _col_map(col_map), _row_map(row_map)
{
  sycl::default_selector device_selector;
  _q = sycl::queue(device_selector);
  std::cout << "Running on "
            << _q.get_device().get_info<sycl::info::device::name>() << "\n";

  mkl_init();
}

template <typename ScalarType>
Matrix<ScalarType>::Matrix(std::vector<ScalarType>& data,
                           std::vector<std::int32_t>& row_ptr,
                           std::vector<std::int32_t>& col_ind,
                           std::shared_ptr<spmv::L2GMap> col_map,
                           std::shared_ptr<spmv::L2GMap> row_map)
    : _col_map(col_map), _row_map(row_map)
{
  sycl::default_selector device_selector;
  _q = sycl::queue(device_selector);

  _data = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(data);
  _indptr = std::make_shared<cl::sycl::buffer<std::int32_t, 1>>(row_ptr);
  _indices = std::make_shared<cl::sycl::buffer<std::int32_t, 1>>(col_ind);
  
  _shape[0] = row_map->local_size() + row_map->num_ghosts();
  _shape[1] = col_map->local_size() + col_map->num_ghosts();

  mkl_init();
}

template <typename ScalarType>
Matrix<ScalarType>::~Matrix()
{
  oneapi::mkl::sparse::release_matrix_handle(&A_onemkl);
}

// Explicit instantiation
template class spmv::Matrix<float>;
template class spmv::Matrix<double>;
// template class spmv::Matrix<std::complex<float>>;
// template class spmv::Matrix<std::complex<double>>;
