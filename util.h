
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

std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>,
           std::shared_ptr<spmv::L2GMap>>
remap_mat(MPI_Comm comm, std::shared_ptr<spmv::L2GMap> row_map,
          Eigen::Ref<Eigen::SparseMatrix<double, Eigen::RowMajor>> A,
          std::shared_ptr<spmv::L2GMap> col_map);

} // namespace spmv
