// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include <mpi.h>

#pragma once

namespace spmv
{
class L2GMap;

class Matrix
{
public:
  /// Matrix with row and column maps.
  /// This constructor just copies in the data. To build a Matrix with no ghost
  /// rows, use Matrix::create_matrix instead.
  Matrix(Eigen::SparseMatrix<double, Eigen::RowMajor> A,
         std::shared_ptr<const spmv::L2GMap> col_map,
         std::shared_ptr<const spmv::L2GMap> row_map);

  ~Matrix()
  {
#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_destroy(A_mkl);
#endif
  }

  /// MatVec operator for A x
  Eigen::VectorXd operator*(const Eigen::VectorXd& b) const;

  /// MatVec operator for A^T x
  Eigen::VectorXd transpmult(const Eigen::VectorXd& b) const;

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<const L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

  /// Access the underlying local sparse matrix
  Eigen::SparseMatrix<double, Eigen::RowMajor>& mat() { return _matA; }
  const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat() const
  {
    return _matA;
  }

  /// Create a matrix from an Eigen::SparseMatrix and row and column mappings
  /// FIXME: should this function really be in the constructor?
  static Matrix
  create_matrix(MPI_Comm comm,
                const Eigen::SparseMatrix<double, Eigen::RowMajor> mat,
                std::int64_t nrows_local, std::int64_t ncols_local,
                std::vector<std::int64_t> row_ghosts,
                std::vector<std::int64_t> col_ghosts);

private:
// MKL pointers to Eigen data
#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;
#endif

  // Storage for Matrix
  Eigen::SparseMatrix<double, Eigen::RowMajor> _matA;

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<const spmv::L2GMap> _col_map;
  std::shared_ptr<const spmv::L2GMap> _row_map;
};
} // namespace spmv
