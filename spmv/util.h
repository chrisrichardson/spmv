
#include "Matrix.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>
#include <vector>

namespace spmv
{
class L2GMap;

Eigen::VectorXd
extract_diagonal(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);

std::vector<int>
diagonal_block_nnz(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat);

spmv::Matrix<double> remap_mat(MPI_Comm comm,
                               std::shared_ptr<const spmv::L2GMap> row_map,
                               const spmv::Matrix<double>& A);

} // namespace spmv
