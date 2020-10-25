// Copyright (C) 2020 Igor Baratta (ia397@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>
#include <memory>
#include <mkl_sycl.hpp>

#include "L2GMap.h"
#include "mpi_type.h"

namespace spmv
{
template <typename ScalarType,
          typename DeviceSelector = cl::sycl::default_selector>
class Vector
{
public:
  Vector(std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<spmv::L2GMap> map)
      : _data(data), _map(map)
  {
    DeviceSelector device_selector;
    _q = sycl::queue(device_selector);

    if (_map->local_size() != _data->get_size())
      std::runtime_error("Size mismatch");
  };

  Vector(std::vector<ScalarType>& vec, std::shared_ptr<spmv::L2GMap> map)
      : _map(map)
  {
    DeviceSelector device_selector;
    _q = sycl::queue(device_selector);
    _data = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(vec);
  }

  std::int32_t local_size() const { return _map->local_size(); }

  cl::sycl::buffer<ScalarType, 1>& getLocalData() { return *_data; };

  /// Computes the product of a vector by a scalar.
  Vector& operator*=(ScalarType alpha)
  {
    std::size_t ls = _map->local_size() + _map->num_ghosts();
    oneapi::mkl::blas::scal(_q, ls, alpha, *_data, 1);
    return *this;
  }

  Vector& operator+=(Vector& other)
  {
    oneapi::mkl::blas::axpy(_q, local_size(), 1.0, other.getLocalData(), 1,
                            *_data, 1);
    return *this;
  }

  Vector& operator-=(Vector& other)
  {
    oneapi::mkl::blas::axpy(_q, local_size(), -1.0, other.getLocalData(), 1,
                            *_data, 1);
    return *this;
  }

  Vector& operator+(Vector& other)
  {
    auto buffer = std::make_shared<cl::sycl::buffer<ScalarType, 1>>(*_data);
    spmv::Vector<ScalarType> out(buffer, _map);
    return out += other;
  }

  double dot(spmv::Vector<ScalarType>& y)
  {
    cl::sycl::buffer<ScalarType, 1> res{1};
    oneapi::mkl::blas::dot(_q, _map->local_size(), *_data, 1, y.getLocalData(),
                           1, res);
    auto acc = res.template get_access<cl::sycl::access::mode::read>();
    MPI_Comm comm = _map->comm();
    double local_sum = acc[0];
    double global_sum{-1.0};

    auto data_type = spmv::mpi_type<ScalarType>();
    MPI_Allreduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, comm);
    return global_sum;
  };

  /// Computes the vector 2 norm (Euclidean norm).
  double norm() { return std::sqrt(this->dot(*this)); }

private:
  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;
  std::shared_ptr<spmv::L2GMap> _map;
  mutable cl::sycl::queue _q;
};
} // namespace spmv