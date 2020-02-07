// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include<Eigen/Sparse>
#include<Eigen/Dense>
#include <mpi.h>

#pragma once

typedef Eigen::SparseMatrix<double>::StorageIndex index_type;

class DistributedVector
{
public:
  DistributedVector();

  // Local "dense" portion of sparse vector
  Eigen::Map<Eigen::VectorXd> vec();

  // Eigen SparseVector
  Eigen::SparseVector<double>& spvec();

  // Set up communication pattern for A.x by querying columns of A for non-zeros
  // and sending send-pattern to remotes
  void setup(MPI_Comm comm,
             const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
             std::vector<index_type>& ranges);

  // Ghost update - should be done each time *before* matvec
  void update(MPI_Comm comm);

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

  // MPI rank
  int rank;
};
