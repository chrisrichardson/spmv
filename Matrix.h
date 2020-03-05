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
         std::shared_ptr<spmv::L2GMap> col_map);

  ~Matrix()
  {
#ifdef EIGEN_USE_MKL_ALL
  mkl_sparse_destroy(A_mkl);
#endif
  }


  // operator
  Eigen::VectorXd operator*(const Eigen::VectorXd& b) const;

  std::shared_ptr<const L2GMap> row_map() const
  {
    return _row_map;
  }

    std::shared_ptr<const L2GMap> col_map() const
  {
    return _col_map;
  }

  int rows() const
  {
    return _matA.rows();
  }


private:

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;
#endif

  Eigen::SparseMatrix<double, Eigen::RowMajor> _matA;

  std::shared_ptr<spmv::L2GMap> _row_map;
  std::shared_ptr<spmv::L2GMap> _col_map;
};
} // namespace spmv
