#pragma once
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

void cg(const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
        const Eigen::Ref<const Eigen::VectorXd>& b);