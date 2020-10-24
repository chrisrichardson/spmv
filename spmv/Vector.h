#include <CL/sycl.hpp>
#include <memory>

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
    if (_map->local_size() != data->size())
      std::runtime_error("Size mismatch");
  };

  std::int32_t local_size() const { return _map->local_size(); }

  cl::sycl::buffer<ScalarType, 1>& getLocalData() { return *_data; };

  Vector& operator+=(const Vector& other)
  {
    oneapi::mkl::blas::axpy(_q, local_size(), 1.0, other.getLocalData(), 1,
                            *_data, 1);
  }

  double dot(spmv::Vector<ScalarType>& y)
  {
    cl::sycl::buffer<ScalarType, 1> res{1};
    oneapi::mkl::blas::dot(_q, _map->local_size(), *_data, 1, y.getLocalData(),
                           1, res);
    auto acc = res.template get_access<cl::sycl::access::mode::read>();
    MPI_Comm comm = _map->comm();
    double local_sum = acc[0];
    double global_sum;

    auto data_type = spmv::mpi_type<ScalarType>();
    MPI_Reduce(&local_sum, &global_sum, 1, data_type, MPI_SUM, 0, comm);
    return global_sum;
  };

private:
  std::shared_ptr<cl::sycl::buffer<ScalarType, 1>> _data;
  std::shared_ptr<spmv::L2GMap> _map;
  mutable cl::sycl::queue _q;
};
} // namespace spmv