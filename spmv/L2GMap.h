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
/// Local to Global Map
/// Maps from the local indices on the current process to global indices across
/// all processes. The local process owns a contiguous set of the global
/// indices, starting at "global_offset". Any indices which are not owned appear
/// as "ghost entries" at the end of the local range.
{
public:
  /// L2GMap (Local to Global Map)
  /// @param comm MPI Comm
  /// @param local_size Local size
  /// @param ghosts Ghost indices, owned by other processes
  /// Ghosts must be sorted in ascending order.
  L2GMap(MPI_Comm comm, std::int64_t local_size,
         const std::vector<std::int64_t>& ghosts);

  // Destructor destroys neighbour comm
  ~L2GMap();

  // Disable copying (may cause problems with held neighbour comm)
  L2GMap(const L2GMap& p) = delete;
  L2GMap& operator=(const L2GMap& p) = delete;

  /// Local size
  /// @param ghosted - if set, return the full local size including ghost
  /// entries, otherwise, just the number of local, owned entries.
  /// @return number of entries in local map
  std::int32_t local_size(bool ghosted) const;

  /// Global size
  /// @return global size of L2GMap
  std::int64_t global_size() const;

  /// Global offset on this process
  /// @return Global index of first local index
  std::int64_t global_offset() const;

  /// Convert a global index to local
  /// @param i Global Index
  /// @return Local index
  std::int32_t global_to_local(std::int64_t i) const;

  /// Ghost update. Copies values from remote indices to the local process.
  /// This should be applied to a vector *before* a MatVec operation, if the
  /// Matrix has column ghosts.
  /// @param vec_data Pointer to vector data
  template <typename T>
  void update(T* vec_data) const;

  /// Reverse update. Sends ghost values to their owners, where they are
  /// accumulated at the local index. This should be applied to the result
  /// *after* a MatVec operation, if the Matrix has row ghosts.
  /// @param vec_data Pointer to vector data
  template <typename T>
  void reverse_update(T* vec_data) const;

  /// Access the ghost indices
  const std::vector<std::int64_t>& ghosts() const { return _ghosts; }

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
