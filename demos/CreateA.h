// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>

#include <memory>
#include <mpi.h>
#include <spmv/Matrix.h>

/// Create a simple matrix for testing
spmv::Matrix<double> create_A(MPI_Comm comm, int N);
