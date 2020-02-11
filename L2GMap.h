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

  // Ghost update - should be done each time *before* matvec
  void update(double* vec_data);

  // Update the other way, ghost -> local.
  void reverse_update(double* vec_data);

private:

  // Ownership ranges for all processes
  std::vector<index_type> _ranges;

  // Indices, counts and offsets for communication
  std::vector<index_type> _indexbuf;
  std::vector<index_type> _send_count;
  std::vector<index_type> _recv_count;
  std::vector<index_type> _send_offset;
  std::vector<index_type> _recv_offset;

  // Temporary data buffer for sending/receiving in local range
  std::vector<double> _databuf;

  MPI_Comm _neighbour_comm;
};
