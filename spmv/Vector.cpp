// Copyright (C) 2020 Igor Baratta (ia397@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <mkl_sycl.hpp>
#include <numeric>

#include "Vector.h"
#include "mpi_type.h"

using namespace spmv;

//-----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>::Vector(
    std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
    std::shared_ptr<spmv::L2GMap> map)
    : _data(data), _map(map)
{
  cl::sycl::default_selector device_selector;
  _q = sycl::queue(device_selector);

  std::size_t ls = _map->local_size() + _map->num_ghosts();
  if (ls != _data->get_size())
    std::runtime_error("Size mismatch");
}

//-----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>::Vector(std::vector<ScalarType>& vec,
                           std::shared_ptr<spmv::L2GMap> map)
    : _map(map)
{
  cl::sycl::default_selector device_selector;
  _q = sycl::queue(device_selector);
  _data = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(vec);

  std::size_t ls = _map->local_size() + _map->num_ghosts();
  if (ls != _data->get_size())
    std::runtime_error("Size mismatch");
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator*=(ScalarType alpha)
{
  std::size_t ls = _map->local_size() + _map->num_ghosts();
  oneapi::mkl::blas::scal(_q, ls, alpha, *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator+=(Vector& other)
{
  oneapi::mkl::blas::axpy(_q, local_size(), 1.0, other.getLocalData(), 1,
                          *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator-=(Vector& other)
{
  oneapi::mkl::blas::axpy(_q, local_size(), -1.0, other.getLocalData(), 1,
                          *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator+(Vector& other)
{
  auto buffer = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(*_data);
  spmv::Vector<ScalarType> out(buffer, _map);
  return out += other;
}

//----------------------------------------------------------------------------
template <typename ScalarType>
double Vector<ScalarType>::dot(spmv::Vector<ScalarType>& y) const
{
  cl::sycl::buffer<ScalarType, 1> res{1};
  oneapi::mkl::blas::dot(_q, _map->local_size(), *_data, 1, y.getLocalData(), 1,
                         res);
  auto acc = res.template get_access<cl::sycl::access::mode::read>();
  MPI_Comm comm = _map->comm();

  double local_sum = acc[0];
  double global_sum{-1.0};

  auto data_type = spmv::mpi_type<double>();
  MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, comm);
  return global_sum;
};
//----------------------------------------------------------------------------
template <>
double
Vector<std::complex<double>>::dot(spmv::Vector<std::complex<double>>& y) const
{
  cl::sycl::buffer<std::complex<double>, 1> res{1};
  oneapi::mkl::blas::dotc(_q, _map->local_size(), *_data, 1, y.getLocalData(),
                          1, res);
  auto acc = res.template get_access<cl::sycl::access::mode::read>();
  MPI_Comm comm = _map->comm();

  double local_sum = acc[0].real();
  double global_sum{-1.0};

  auto data_type = spmv::mpi_type<double>();
  MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, comm);
  return global_sum;
};
//----------------------------------------------------------------------------
template <>
double
Vector<std::complex<float>>::dot(spmv::Vector<std::complex<float>>& y) const
{
  cl::sycl::buffer<std::complex<float>, 1> res{1};
  oneapi::mkl::blas::dotc(_q, _map->local_size(), *_data, 1, y.getLocalData(),
                          1, res);
  auto acc = res.template get_access<cl::sycl::access::mode::read>();
  MPI_Comm comm = _map->comm();

  double local_sum = acc[0].real();
  double global_sum{-1.0};

  auto data_type = spmv::mpi_type<double>();
  MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, comm);
  return global_sum;
};
//----------------------------------------------------------------------------
// Explicit instantiation
template class spmv::Vector<float>;
template class spmv::Vector<double>;
template class spmv::Vector<std::complex<float>>;
template class spmv::Vector<std::complex<double>>;