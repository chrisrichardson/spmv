
#include<vector>
#include<Eigen/Dense>
#include<Eigen/Sparse>

Eigen::VectorXd
extract_diagonal(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);

std::vector<int>
diagonal_block_nnz(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);
