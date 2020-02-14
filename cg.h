#pragma once
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

class L2GMap;

// Solve A.x=b iteratively with Conjugate Gradient
std::tuple<Eigen::VectorXd, int> cg(MPI_Comm comm,
                                    Eigen::Ref<Eigen::SparseMatrix<double, Eigen::RowMajor>> A,
                   const std::shared_ptr<const L2GMap> l2g,
                   const Eigen::Ref<const Eigen::VectorXd>& b);
