// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <mpi.h>

#pragma once

namespace spmv
{
class L2GMap;

class Matrix
{
public:
  Matrix(Eigen::SparseMatrix<double, Eigen::RowMajor> A,
         std::shared_ptr<spmv::L2GMap> col_map,
         std::shared_ptr<spmv::L2GMap> row_map);

  ~Matrix()
  {
#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_destroy(A_mkl);
#endif
  }

  // operator
  Eigen::VectorXd operator*(const Eigen::VectorXd& b) const;

  Eigen::VectorXd transpmult(const Eigen::VectorXd& b) const;

  std::shared_ptr<const L2GMap> row_map() const { return _row_map; }

  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

  int rows() const { return _matA.rows(); }

  static Matrix
  create_matrix(MPI_Comm comm,
                const Eigen::SparseMatrix<double, Eigen::RowMajor> mat,
                std::int64_t nrows_local, std::int64_t ncols_local,
                std::vector<std::int64_t> row_ghosts,
                std::vector<std::int64_t> col_ghosts);

private:
#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;
#endif

  Eigen::SparseMatrix<double, Eigen::RowMajor> _matA;

  std::shared_ptr<spmv::L2GMap> _col_map;
  std::shared_ptr<spmv::L2GMap> _row_map;
};
} // namespace spmv
