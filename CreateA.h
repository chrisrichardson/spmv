#pragma once

#include "L2GMap.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>

std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
           std::shared_ptr<L2GMap>>
create_A(MPI_Comm comm, int N);
