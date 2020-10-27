// Copyright (C) 2020 Igor Baratta (ia397@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <mkl_sycl.hpp>
#include <numeric>

#include "Vector.h"
#include "mpi_type.h"

using namespace spmv;

template <typename ScalarType>
class SetZero;

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
//-----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType> Vector<ScalarType>::duplicate() const
{
  std::size_t ls = _map->local_size() + _map->num_ghosts();
  auto buffer = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(
      cl::sycl::range<1>{ls});
  return spmv::Vector<ScalarType>(buffer, _map);
}
//-----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType> Vector<ScalarType>::copy() const
{
  std::size_t ls = _map->local_size() + _map->num_ghosts();
  auto buffer = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(
      cl::sycl::range<1>{ls});
  oneapi::mkl::blas::copy(_q, ls, *_data, 1, *buffer, 1);
  return spmv::Vector<ScalarType>(buffer, _map);
}
//----------------------------------------------------------------------------
template <typename ScalarType>
void Vector<ScalarType>::set_zero()
{
  _q.submit([&](cl::sycl::handler& cgh) {
    std::size_t ls = _map->local_size() + _map->num_ghosts();
    auto acc
        = _data->template get_access<sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<SetZero<ScalarType>>(
        cl::sycl::range<1>{ls}, [=](cl::sycl::id<1> i) { acc[i] = 0.; });
  });
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
Vector<ScalarType> Vector<ScalarType>::operator*(ScalarType alpha) const
{
  auto out = this->copy();
  out *= alpha;
  return out;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator+=(Vector& other)
{
  oneapi::mkl::blas::axpy(_q, local_size(), 1.0, other.get_local_buffer(), 1,
                          *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator+=(Vector&& other)
{
  oneapi::mkl::blas::axpy(_q, local_size(), 1.0, other.get_local_buffer(), 1,
                          *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType> Vector<ScalarType>::operator+(Vector& other) const
{
  auto out = this->copy();
  out += other;
  return out;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType> Vector<ScalarType>::operator+(Vector&& other) const
{
  auto out = this->copy();
  out += other;
  return out;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator-=(Vector& other)
{
  oneapi::mkl::blas::axpy(_q, local_size(), -1.0, other.get_local_buffer(), 1,
                          *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType>& Vector<ScalarType>::operator-=(Vector&& other)
{
  oneapi::mkl::blas::axpy(_q, local_size(), -1.0, other.get_local_buffer(), 1,
                          *_data, 1);
  return *this;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
Vector<ScalarType> Vector<ScalarType>::operator-(Vector& other) const
{
  auto out = this->copy();
  out -= other;
  return out;
}
//----------------------------------------------------------------------------
template <typename ScalarType>
double Vector<ScalarType>::dot(spmv::Vector<ScalarType>& y)
{
  cl::sycl::buffer<ScalarType, 1> res{1};
  oneapi::mkl::blas::dot(_q, _map->local_size(), *_data, 1,
                         y.get_local_buffer(), 1, res);
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
double Vector<std::complex<double>>::dot(spmv::Vector<std::complex<double>>& y)
{
  cl::sycl::buffer<std::complex<double>, 1> res{1};
  oneapi::mkl::blas::dotc(_q, _map->local_size(), *_data, 1,
                          y.get_local_buffer(), 1, res);
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
double Vector<std::complex<float>>::dot(spmv::Vector<std::complex<float>>& y)
{
  cl::sycl::buffer<std::complex<float>, 1> res{1};
  oneapi::mkl::blas::dotc(_q, _map->local_size(), *_data, 1,
                          y.get_local_buffer(), 1, res);
  auto acc = res.template get_access<cl::sycl::access::mode::read>();
  MPI_Comm comm = _map->comm();

  double local_sum = acc[0].real();
  double global_sum{-1.0};

  auto data_type = spmv::mpi_type<double>();
  MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, comm);
  return global_sum;
};
//----------------------------------------------------------------------------
template <typename ScalarType>
void Vector<ScalarType>::update()
{
  auto p_buffer = this->get_local_buffer();
  auto pacc = p_buffer.template get_access<cl::sycl::access::mode::read>();
  _map->update(static_cast<ScalarType*>(pacc.get_pointer()));
}
//----------------------------------------------------------------------------
// Explicit instantiation
template class spmv::Vector<float>;
template class spmv::Vector<double>;
template class spmv::Vector<std::complex<float>>;
template class spmv::Vector<std::complex<double>>;