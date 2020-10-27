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
  /// might be already on device memory)
  Vector(std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> data,
         std::shared_ptr<spmv::L2GMap> map);

  /// Create Distributed Vector using data on host and a local-to-global map
  Vector(std::vector<ScalarType>& vec, std::shared_ptr<spmv::L2GMap> map);

  /// Set queue to submit work (indirectly defines the device in each rank)
  void set_queue(cl::sycl::queue q) { _q = q; }

  std::int32_t local_size() const { return _map->local_size(); }

  cl::sycl::buffer<ScalarType, 1>& get_local_buffer() { return *_data; };

  /// Computes the product of the vector by a scalar.
  Vector& operator*=(ScalarType alpha);

  Vector& operator+=(Vector<ScalarType>& other);
  Vector& operator+=(Vector<ScalarType>&& other);
  Vector& operator-=(Vector<ScalarType>& other);
  Vector& operator-=(Vector<ScalarType>&& other);

  /// Computes the product of the vector by a scalar.
  Vector operator*(ScalarType alpha) const;

  /// Applies binary sum operator to each element of tow Vectors (a = b + c)
  Vector operator+(Vector<ScalarType>& other) const;

  /// Applies binary sum operator to each element of tow Vectors (a = b + c)
  Vector operator+(Vector<ScalarType>&& other) const;

  /// Applies substraction operator to each element of tow Vectors (a = b - c)
  Vector operator-(Vector<ScalarType>& other) const;

  /// Creates a new vector of the same type and dimension as this.
  Vector duplicate() const;

  /// Creates a new vector and copy the content of the current vector.
  Vector copy() const;

  /// Sets all elements of this vector to zero.
  void set_zero();

  /// Computes the dot product.
  double dot(spmv::Vector<ScalarType>& y);

  /// Computes the vector 2 norm (Euclidean norm).
  double norm() { return std::sqrt(this->dot(*this)); }

  /// Updates ghosts values, from global representation to local.
  void update();

private:
  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;
  std::shared_ptr<spmv::L2GMap> _map;
  mutable cl::sycl::queue _q;
};
} // namespace spmv