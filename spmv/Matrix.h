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

/// Simple Distributed Sparse Linear Algebra Library
namespace spmv
{

class L2GMap;

template <typename T>
class Matrix
{
  /// Matrix with row and column maps.
public:
  /// This constructor just copies in the data. To build a Matrix with no ghost
  /// rows, use Matrix::create_matrix instead.
  Matrix(Eigen::SparseMatrix<T, Eigen::RowMajor> A,
         std::shared_ptr<const spmv::L2GMap> col_map,
         std::shared_ptr<const spmv::L2GMap> row_map);

  /// Destructor (destroys MKL structs, if using MKL)
  ~Matrix();

  /// MatVec operator for A x
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator*(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// MatVec operator for A^T x
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  transpmult(const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) const;

  /// Row mapping (local-to-global). Usually, there will not be ghost rows.
  std::shared_ptr<const L2GMap> row_map() const { return _row_map; }

  /// Column mapping (local-to-global)
  std::shared_ptr<const L2GMap> col_map() const { return _col_map; }

  /// Access the underlying local sparse matrix
  Eigen::SparseMatrix<T, Eigen::RowMajor>& mat() { return _matA; }
  const Eigen::SparseMatrix<T, Eigen::RowMajor>& mat() const { return _matA; }

  /// Create a matrix from an Eigen::SparseMatrix and row and column mappings
  /// FIXME: should this function really be in the constructor?
  static Matrix<T>
  create_matrix(MPI_Comm comm,
                const Eigen::SparseMatrix<T, Eigen::RowMajor> mat,
                std::int64_t nrows_local, std::int64_t ncols_local,
                std::vector<std::int64_t> row_ghosts,
                std::vector<std::int64_t> col_ghosts);

private:
// MKL pointers to Eigen data
#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  struct matrix_descr mat_desc;
  void mkl_init();
#endif

  // Storage for Matrix
  Eigen::SparseMatrix<T, Eigen::RowMajor> _matA;

  // Column and Row maps: usually _row_map will not have ghosts.
  std::shared_ptr<const spmv::L2GMap> _col_map;
  std::shared_ptr<const spmv::L2GMap> _row_map;
};
} // namespace spmv
