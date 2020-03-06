#pragma once

#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>
#include "Matrix.h"

namespace spmv
{
class L2GMap;
}

/// Create a simple matrix for testing
spmv::Matrix create_A(MPI_Comm comm, int N);
