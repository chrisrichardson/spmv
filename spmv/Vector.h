// Copyright (C) 2020 Igor Baratta (ia397@cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>
#include <memory>

#include "L2GMap.h"
#include "mpi_type.h"

namespace spmv
{
template <typename ScalarType>
class Vector
{
public:
  /// Create Distributed Vector using data buffers and local-to-global map (data
  /// might be on device memory)
  Vector(std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<spmv::L2GMap> map);

  /// Create Distributed Vector using data on host and a local-to-global map
  Vector(std::vector<ScalarType>& vec, std::shared_ptr<spmv::L2GMap> map);

  std::int32_t local_size() const { return _map->local_size(); }

  cl::sycl::buffer<ScalarType, 1>& getLocalData() { return *_data; };

  /// Computes the product of the vector by a scalar.
  Vector& operator*=(ScalarType alpha);

  Vector& operator+=(Vector& other);

  Vector& operator-=(Vector& other);

  Vector& operator+(Vector& other);

  double dot(spmv::Vector<ScalarType>& y) const;

  /// Computes the vector 2 norm (Euclidean norm).
  double norm() { return std::sqrt(this->dot(*this)); }

private:
  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;
  std::shared_ptr<spmv::L2GMap> _map;
  mutable cl::sycl::queue _q;
};
} // namespace spmv