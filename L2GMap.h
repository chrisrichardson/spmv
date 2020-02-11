// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <vector>

#pragma once

typedef Eigen::SparseMatrix<double>::StorageIndex index_type;

class L2GMap
{
public:
  // Constructor
  L2GMap(MPI_Comm comm, const std::vector<index_type>& ranges,
         const std::vector<index_type>& ghosts);

  // Destructor destroys neighbour comm
  ~L2GMap();

  // Total local size (including ghosts)
  std::int32_t local_size() const;

  // Global size
  std::int64_t global_size() const;

  // Global offset on this process
  std::int64_t global_offset() const;

  // Convert a global index to local
  index_type global_to_local(index_type i) const;

  // Ghost update - should be done each time *before* matvec
  void update(double* vec_data) const;

  // Update the other way, ghost -> local.
  void reverse_update(double* vec_data) const;

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
  std::vector<index_type> _send_count;
  std::vector<index_type> _recv_count;
  std::vector<index_type> _send_offset;
  std::vector<index_type> _recv_offset;

  // Temporary data buffer for sending/receiving in local range
  // This is just scratch space, so marking as mutable
  // Or could change update to be non-const
  mutable std::vector<double> _databuf;

  MPI_Comm _neighbour_comm;
};
