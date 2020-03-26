// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <vector>

#pragma once

namespace spmv
{

class L2GMap
{
public:
  /// L2GMap (Local to Global Map)
  /// @param comm MPI Comm
  /// @param local_size Local size
  /// @param ghosts Ghost indices, owned by other processes
  L2GMap(MPI_Comm comm, std::int64_t local_size,
         const std::vector<std::int64_t>& ghosts);

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
  std::int32_t global_to_local(std::int64_t i) const;

  // Ghost update - should be done each time *before* matvec
  template <typename T>
  void update(T* vec_data) const;

  // Update the other way, ghost -> local.
  template <typename T>
  void reverse_update(T* vec_data) const;

private:
  // Ownership ranges for all processes on global comm
  std::vector<std::int64_t> _ranges;

  // Cached mpi rank on global comm
  // Local range is _ranges[_mpi_rank] -> _ranges[_mpi_rank + 1]
  std::int32_t _mpi_rank;

  // Forward and reverse maps for ghosts
  std::map<std::int64_t, std::int32_t> _global_to_local;
  std::vector<std::int64_t> _ghosts;

  // Indices, counts and offsets for communication
  std::vector<std::int32_t> _indexbuf;
  std::vector<std::int32_t> _send_count;
  std::vector<std::int32_t> _recv_count;
  std::vector<std::int32_t> _send_offset;
  std::vector<std::int32_t> _recv_offset;

  MPI_Comm _neighbour_comm;
};

} // namespace spmv
