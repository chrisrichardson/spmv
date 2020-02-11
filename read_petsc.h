// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Sparse>
#include <mpi.h>
#include <string>

// Read a binary PETSc matrix file (32-bit)
// Create a suitable file with petsc option "-ksp_view_mat binary"
Eigen::SparseMatrix<double, Eigen::RowMajor>
read_petsc_binary_matrix(MPI_Comm comm, std::string filename);

// Read a binary PETSc vector file and distribute
// Create a suitable file with petsc option "-ksp_view_rhs binary"
Eigen::VectorXd read_petsc_binary_vector(MPI_Comm comm, std::string filename);
