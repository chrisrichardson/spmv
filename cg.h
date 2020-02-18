#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>

class L2GMap;

// Solve A.x=b iteratively with Conjugate Gradient
//
// Input
// @param comm MPI comm
// @param A LHS matrix
// @param l2g Local-to-global map
// @param b RHS vector
// @param max_its Maximum iteration count
// @param rtol Relative tolerance
//
// @return tuple of result and number of iterations
//
std::tuple<Eigen::VectorXd, int>
cg(MPI_Comm comm, Eigen::Ref<Eigen::SparseMatrix<double, Eigen::RowMajor>> A,
   const std::shared_ptr<const L2GMap> l2g,
   const Eigen::Ref<const Eigen::VectorXd>& b, int max_its, double rtol);
