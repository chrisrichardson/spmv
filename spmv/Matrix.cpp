// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk), Jeffrey Salmond
// and Igor Baratta
// SPDX-License-Identifier:    MIT

#include <numeric>

#include "oneapi/mkl.hpp"

#include "Matrix.h"
#include "mpi_type.h"

using namespace spmv;

//-----------------------------------------------------------------------------
template <typename ScalarType>
Matrix<ScalarType>::Matrix(
    std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
    std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> indptr,
    std::shared_ptr<cl::sycl::buffer<std::int32_t, 1>> indices,
    std::shared_ptr<spmv::L2GMap> col_map,
    std::shared_ptr<spmv::L2GMap> row_map)
    : _data(data), _indptr(indptr), _indices(indices), _col_map(col_map),
      _row_map(row_map)
{
  sycl::cpu_selector device_selector;
  _q = sycl::queue(device_selector);

  _shape[0] = row_map->local_size() + row_map->num_ghosts();
  _shape[1] = col_map->local_size() + col_map->num_ghosts();

  mkl_init();
}

//-----------------------------------------------------------------------------
template <typename ScalarType>
Matrix<ScalarType>::Matrix(std::vector<ScalarType>& data,
                           std::vector<std::int32_t>& row_ptr,
                           std::vector<std::int32_t>& col_ind,
                           std::shared_ptr<spmv::L2GMap> col_map,
                           std::shared_ptr<spmv::L2GMap> row_map)
    : _col_map(col_map), _row_map(row_map)
{
  sycl::cpu_selector device_selector;
  _q = sycl::queue(device_selector);

  _data = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(data);
  _indptr = std::make_shared<cl::sycl::buffer<std::int32_t, 1>>(row_ptr);
  _indices = std::make_shared<cl::sycl::buffer<std::int32_t, 1>>(col_ind);

  _shape[0] = row_map->local_size() + row_map->num_ghosts();
  _shape[1] = col_map->local_size() + col_map->num_ghosts();

  mkl_init();
}
//-----------------------------------------------------------------------------
template <typename ScalarType>
void Matrix<ScalarType>::mult(spmv::Vector<ScalarType>& x,
                              spmv::Vector<ScalarType>& b) const
{
  oneapi::mkl::sparse::gemv(_q, oneapi::mkl::transpose::nontrans, 1.0, A_onemkl,
                            x.get_local_buffer(), 0.0, b.get_local_buffer());
};
//-----------------------------------------------------------------------------
template <typename ScalarType>
void Matrix<ScalarType>::transpmult(spmv::Vector<ScalarType>& x,
                                    spmv::Vector<ScalarType>& b) const
{
  oneapi::mkl::sparse::gemv(_q, oneapi::mkl::transpose::trans, 1.0, A_onemkl,
                            x.get_local_buffer(), 0.0, b.get_local_buffer());
};
//-----------------------------------------------------------------------------
template <typename ScalarType>
spmv::Vector<ScalarType>
Matrix<ScalarType>::operator*(spmv::Vector<ScalarType>& x) const
{
  std::size_t ls = _row_map->local_size() + _row_map->num_ghosts();
  auto buffer = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(
      cl::sycl::range<1>(ls));
  spmv::Vector<ScalarType> b(buffer, _row_map);
  this->mult(x, b);
  return b;
};
//-----------------------------------------------------------------------------
template <typename ScalarType>
Matrix<ScalarType>::~Matrix()
{
  oneapi::mkl::sparse::release_matrix_handle(&A_onemkl);
}
//-----------------------------------------------------------------------------
template <typename ScalarType>
void Matrix<ScalarType>::mkl_init()
{
  oneapi::mkl::sparse::init_matrix_handle(&A_onemkl);
  oneapi::mkl::sparse::set_csr_data(A_onemkl, _shape[0], _shape[1],
                                    oneapi::mkl::index_base::zero, *_indptr,
                                    *_indices, *_data);
  oneapi::mkl::sparse::optimize_gemv(_q, oneapi::mkl::transpose::nontrans,
                                     A_onemkl);
}
//----------------------------------------------------------------------------
// Explicit instantiation
template class spmv::Matrix<float>;
template class spmv::Matrix<double>;
template class spmv::Matrix<std::complex<float>>;
template class spmv::Matrix<std::complex<double>>;
