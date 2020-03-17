// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <vector>

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#pragma once

typedef Eigen::SparseMatrix<double>::StorageIndex index_type;

namespace spmv
{

class L2GMap
{
public:
  /// L2GMap (Local to Global Map)
  /// @param comm MPI Comm
  /// @param ranges Local range on each process
  /// @param ghosts Ghost indices, owned by other processes
  L2GMap(MPI_Comm comm, const std::vector<index_type>& ranges,
         const std::vector<index_type>& ghosts);

  // Destructor destroys neighbour comm
  ~L2GMap();

  // Disable copying (may cause problems with held neighbour comm)
  L2GMap(const L2GMap& p) = delete;
  L2GMap& operator=(const L2GMap& p) = delete;

  /// Local size
  /// @param ghosted - if set, return the full local size including ghost
  /// entries
  /// @return number of entries in local map
  std::int32_t local_size(bool ghosted) const;

  /// Global size
  /// @return global size of L2GMap
  std::int64_t global_size() const;

  // Global offset on this process
  std::int64_t global_offset() const;

  // Convert a global index to local
  index_type global_to_local(index_type i) const;

  // Ghost update - should be done each time *before* matvec
  template <typename T>
  void update(std::vector<T> &) const;
  void update(Eigen::VectorXd &) const;
  template <typename T>
  void update_cpu(T*) const;
  #ifdef HAVE_CUDA
  template <typename T>
  void update(thrust::device_ptr<T> &) const;
  #endif

  // Update the other way, ghost -> local.
  template <typename T>
  void reverse_update(T* vec_data) const;

private:
  // Ownership ranges for all processes on global comm
  std::vector<index_type> _ranges;

  // Cached mpi rank on global comm
  // Local range is _ranges[_mpi_rank] -> _ranges[_mpi_rank + 1]
  std::int32_t _mpi_rank;

  // Forward and reverse maps for ghosts
  std::map<index_type, index_type> _global_to_local;
  std::vector<index_type> _ghosts;

  // Indices, counts and offsets for communication
  std::vector<index_type> _indexbuf;
#ifdef HAVE_CUDA
  int *_indexbuf_d;
  double *_databuf_d;
#endif
  std::vector<index_type> _send_count;
  std::vector<index_type> _recv_count;
  std::vector<index_type> _send_offset;
  std::vector<index_type> _recv_offset;

  MPI_Comm _neighbour_comm;
};

} // namespace spmv
