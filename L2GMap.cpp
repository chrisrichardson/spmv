// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "L2GMap.h"
#include "cuda_check.h"
#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#ifdef HAVE_CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif

#include "shuffle_kernel.h"

using namespace spmv;

namespace
{
template <typename T>
MPI_Datatype mpi_type();

template <>
MPI_Datatype mpi_type<float>()
{
  return MPI_FLOAT;
}
template <>
MPI_Datatype mpi_type<std::complex<float>>()
{
  return MPI_C_FLOAT_COMPLEX;
}
template <>
MPI_Datatype mpi_type<double>()
{
  return MPI_DOUBLE;
}
template <>
MPI_Datatype mpi_type<std::complex<double>>()
{
  return MPI_DOUBLE_COMPLEX;
}
} // namespace
//-----------------------------------------------------------------------------
L2GMap::L2GMap(MPI_Comm comm, const std::vector<index_type>& ranges,
               const std::vector<index_type>& ghosts)
    : _ranges(ranges), _ghosts(ghosts)
{
  int mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &_mpi_rank);

  const std::int64_t r0 = _ranges[_mpi_rank];
  const std::int64_t r1 = _ranges[_mpi_rank + 1];
  const index_type local_size = r1 - r0;

  // Make sure ghosts are in order
  std::sort(_ghosts.begin(), _ghosts.end());

  // Get count on each process and local index
  std::vector<std::int32_t> ghost_count(mpi_size);
  for (std::size_t i = 0; i < _ghosts.size(); ++i)
  {
    const index_type idx = _ghosts[i];

    if (idx >= r0 and idx < r1)
      throw std::runtime_error("Ghost index in local range");
    _global_to_local.insert({idx, local_size + i});

    auto it = std::upper_bound(_ranges.begin(), _ranges.end(), idx);
    assert(it != _ranges.end());
    const int p = it - _ranges.begin() - 1;
    ++ghost_count[p];
  }

  std::vector<int> neighbours;
  for (std::size_t i = 0; i < ghost_count.size(); ++i)
  {
    const std::int32_t c = ghost_count[i];
    if (c > 0)
    {
      neighbours.push_back(i);
      _send_count.push_back(c);
    }
  }

  const int neighbour_size = neighbours.size();
  MPI_Dist_graph_create_adjacent(comm, neighbours.size(), neighbours.data(),
                                 MPI_UNWEIGHTED, neighbours.size(),
                                 neighbours.data(), MPI_UNWEIGHTED,
                                 MPI_INFO_NULL, false, &_neighbour_comm);

  // Send NNZs by Alltoall - these will be the receive counts for incoming
  // index/values
  _recv_count.resize(neighbour_size);
  MPI_Neighbor_alltoall(_send_count.data(), 1, MPI_INT, _recv_count.data(), 1,
                        MPI_INT, _neighbour_comm);

  _send_offset = {0};
  for (int c : _send_count)
    _send_offset.push_back(_send_offset.back() + c);

  _recv_offset = {0};
  for (int c : _recv_count)
    _recv_offset.push_back(_recv_offset.back() + c);
  int count = _recv_offset.back();

  _indexbuf.resize(count);

  // Send global indices to remote processes that own them
  int err = MPI_Neighbor_alltoallv(
      _ghosts.data(), _send_count.data(), _send_offset.data(), MPI_INT,
      _indexbuf.data(), _recv_count.data(), _recv_offset.data(), MPI_INT,
      _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");

  // Should be in own range, subtract off _r0
  for (index_type& i : _indexbuf)
  {
    assert(i >= r0 and i < r1);
    i -= r0;
  }

  // Add local_range onto _send_offset (ghosts will be at end of range)
  for (index_type& s : _send_offset)
    s += local_size;

#ifdef HAVE_CUDA
  cuda_CHECK(cudaMalloc(&_indexbuf_d, sizeof(int) * _indexbuf.size()));
  cuda_CHECK(cudaMemcpy(_indexbuf_d, _indexbuf.data(), sizeof(int) * _indexbuf.size(),
             cudaMemcpyHostToDevice));

  cuda_CHECK(cudaMalloc(&_databuf_d, sizeof(double) * _indexbuf.size()));
#endif
}
//-----------------------------------------------------------------------------
L2GMap::~L2GMap() { MPI_Comm_free(&_neighbour_comm); }
//-----------------------------------------------------------------------------
#ifdef HAVE_CUDA
template <typename T>
void L2GMap::update(thrust::device_ptr<T> &vec_data) const
{
  // In serial, nothing to do
  if (_indexbuf.size() == 0)
    return;

  MPI_Datatype data_type = mpi_type<T>();

  do_shuffle(_databuf_d, thrust::raw_pointer_cast(vec_data), _indexbuf_d, _indexbuf.size());

  int err = MPI_Neighbor_alltoallv(
      _databuf_d, _recv_count.data(), _recv_offset.data(), data_type, thrust::raw_pointer_cast(vec_data),
      _send_count.data(), _send_offset.data(), data_type, _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");
}
#endif

template <typename T>
void L2GMap::update_cpu(T *vec_data) const {
  // In serial, nothing to do
  if (_indexbuf.size() == 0)
    return;

  MPI_Datatype data_type = mpi_type<T>();

  std::vector<T> buf(_indexbuf.size());
  for (std::size_t i = 0; i < _indexbuf.size(); ++i)
    buf[i] = vec_data[_indexbuf[i]];

  // Send actual values - NB meaning of _send and _recv count/offset is
  // reversed
  int err = MPI_Neighbor_alltoallv(
      buf.data(), _recv_count.data(), _recv_offset.data(), data_type, vec_data,
      _send_count.data(), _send_offset.data(), data_type, _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");
}
template <typename T>
void L2GMap::update(std::vector<T> &vec_data) const
{
  update_cpu(vec_data.data());
}
void L2GMap::update(Eigen::VectorXd &vec_data) const
{
  update_cpu(vec_data.data());
}

//-----------------------------------------------------------------------------
template <typename T>
void L2GMap::reverse_update(T* vec_data) const
{
  MPI_Datatype data_type = mpi_type<T>();
  // Send values from ghost region of vector to remotes
  // accumulating in local vector.
  std::vector<T> databuf(_indexbuf.size());
  int err = MPI_Neighbor_alltoallv(
      vec_data, _send_count.data(), _send_offset.data(), data_type,
      databuf.data(), _recv_count.data(), _recv_offset.data(), data_type,
      _neighbour_comm);
  if (err != MPI_SUCCESS)
    throw std::runtime_error("MPI failure");

  for (std::size_t i = 0; i < _indexbuf.size(); ++i)
    vec_data[_indexbuf[i]] += databuf[i];
}
//-----------------------------------------------------------------------------
index_type L2GMap::global_to_local(index_type i) const
{
  const std::int64_t r0 = _ranges[_mpi_rank];
  const std::int64_t r1 = _ranges[_mpi_rank + 1];

  if (i >= r0 and i < r1)
    return (i - r0);
  else
  {
    auto it = _global_to_local.find(i);
    assert(it != _global_to_local.end());
    return it->second;
  }
}
//-----------------------------------------------------------------------------
std::int32_t L2GMap::local_size(bool ghosted) const
{
  if (ghosted)
    return (_ranges[_mpi_rank + 1] - _ranges[_mpi_rank] + _ghosts.size());
  else
    return (_ranges[_mpi_rank + 1] - _ranges[_mpi_rank]);
}
//-----------------------------------------------------------------------------
std::int64_t L2GMap::global_size() const { return _ranges.back(); }
//-----------------------------------------------------------------------------
std::int64_t L2GMap::global_offset() const { return _ranges[_mpi_rank]; }
//-----------------------------------------------------------------------------
// Explicit instantiation
template void spmv::L2GMap::update(std::vector<double>&) const;
#ifdef HAVE_CUDA
template void spmv::L2GMap::update(thrust::device_ptr<double>&) const;
#endif

template void spmv::L2GMap::reverse_update<double>(double*) const;
// template void spmv::L2GMap::reverse_update<float>(float*) const;
// template void
// spmv::L2GMap::reverse_update<std::complex<float>>(std::complex<float>*)
// const; template void
// spmv::L2GMap::reverse_update<std::complex<double>>(std::complex<double>*)
// const;
