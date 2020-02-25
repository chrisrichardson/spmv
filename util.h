
#include<vector>
#include<Eigen/Dense>
#include<Eigen/Sparse>

Eigen::VectorXd
extract_diagonal(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);

std::vector<int>
diagonal_block_nnz(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);
