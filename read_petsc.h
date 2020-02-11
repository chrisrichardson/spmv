// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Sparse>
#include <mpi.h>
#include <string>

// Read a binary PETSc matrix file (32-bit)
Eigen::SparseMatrix<double, Eigen::RowMajor>
read_petsc_binary(MPI_Comm comm, std::string filename);
