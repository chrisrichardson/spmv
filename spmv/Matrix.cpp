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
Matrix<ScalarType>::Matrix(Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> A,
                           std::shared_ptr<spmv::L2GMap> col_map,
                           std::shared_ptr<spmv::L2GMap> row_map)
{
  _col_map = col_map;
  _row_map = row_map;

  _shape[0] = A.rows();
  _shape[1] = A.cols();

  sycl::default_selector device_selector;
  _q = sycl::queue(device_selector);
  std::cout << "Running on "
            << _q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Get indptr buffer
  std::int32_t* Aouter = A.outerIndexPtr();
  std::vector<std::int32_t> indptr(A.rows());
  std::memcpy(Aouter, indptr.data(), sizeof(std::int32_t) * indptr.size());
  _indptr = std::make_shared<cl::sycl::buffer<std::int32_t, 1>>(indptr);

  // Get indices buffer
  std::int32_t* Ainner = A.innerIndexPtr();
  std::vector<std::int32_t> indices(A.nonZeros());
  std::memcpy(Ainner, indices.data(), sizeof(std::int32_t) * indices.size());
  _indices = std::make_shared<cl::sycl::buffer<std::int32_t, 1>>(indices);

  // Get data buffer
  ScalarType* Aptr = A.valuePtr();
  std::vector<double> data(A.nonZeros());
  std::memcpy(Aptr, data.data(), sizeof(ScalarType) * data.size());
  _data = std::make_shared<cl::sycl::buffer<double, 1>>(data);

  cl::sycl::buffer<double, 1> norm{1};
  oneapi::mkl::blas::nrm2(_q, A.nonZeros(), *_data, 1, norm);

  auto acc = norm.template get_access<cl::sycl::access::mode::read>();
  double result = acc[0];
  std::cout << Aptr[1] << std::endl;

  mkl_init();
}

template <typename ScalarType>
Matrix<ScalarType>::~Matrix()
{
  oneapi::mkl::sparse::release_matrix_handle(&A_onemkl);
}


// Explicit instantiation
// template class spmv::Matrix<float>;
template class spmv::Matrix<double>;
// template class spmv::Matrix<std::complex<float>>;
// template class spmv::Matrix<std::complex<double>>;
