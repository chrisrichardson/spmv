#pragma once
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::SparseMatrix<double, Eigen::RowMajor> create_A(MPI_Comm comm, int N);