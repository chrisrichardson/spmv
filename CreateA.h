#pragma once

#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>

namespace spmv
{
class L2GMap;
}

/// Create a simple matrix for testing
std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
           std::shared_ptr<spmv::L2GMap>>
create_A(MPI_Comm comm, int N);
