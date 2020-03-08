#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#pragma once

namespace spmv
{
class L2GMap;

class Matrix
{
public:
  Matrix(Eigen::SparseMatrix<double, Eigen::RowMajor> A,
         std::shared_ptr<const spmv::L2GMap> col_map);

  ~Matrix()
  {
#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_destroy(A_mkl);
#endif
  }

  // Multiply operator
  Eigen::VectorXd operator*(const Eigen::VectorXd& b) const;

  // Transpose multiply
  Eigen::VectorXd transpmult(const Eigen::VectorXd& b) const;

  // Column Local-to-Global map
  std::shared_ptr<const L2GMap> col_map() const
  {
    return _col_map;
  }

  // Number of local rows
  int rows() const
  {
    return _matA.rows();
  }

  // Underlying matrix
  Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>> mat() const
  {
    return _matA;
  }

private:

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;
#endif

  Eigen::SparseMatrix<double, Eigen::RowMajor> _matA;

  std::shared_ptr<const spmv::L2GMap> _col_map;
};
} // namespace spmv
