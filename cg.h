#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <mpi.h>

void cg(const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
        const Eigen::Ref<const Eigen::VectorXd>& b);