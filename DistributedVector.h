// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>
#include <vector>

#pragma once

typedef Eigen::SparseMatrix<double>::StorageIndex index_type;

class DistributedVector
{
public:
  DistributedVector(MPI_Comm comm,
                    const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);

  // Destructor destroys neighbour comm
  ~DistributedVector();

  // Local "dense" portion of sparse vector
  Eigen::Map<Eigen::VectorXd> vec();

  // Eigen SparseVector using global indexing
  Eigen::SparseVector<double>& spvec();

  // Ghost update - should be done each time *before* matvec
  void update();

private:
  // Actual data
  Eigen::SparseVector<double> _xsp;

  // Indices, counts and offsets for communication
  std::vector<index_type> _indexbuf;
  std::vector<index_type> _send_count;
  std::vector<index_type> _recv_count;
  std::vector<index_type> _send_offset;
  std::vector<index_type> _recv_offset;

  // Data buffer for sending to remotes
  std::vector<double> _send_data;

  // Address and size of "local" entries in sparse vector
  index_type _i0;
  index_type _local_size;

  MPI_Comm _neighbour_comm;
};
