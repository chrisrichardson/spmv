#include <Eigen/Core>
#include <Eigen/Dense>

class MKLSparseMatrix;
using Eigen::SparseMatrix;

namespace Eigen {
namespace internal {
  // MKLSparseMatrix looks-like a SparseMatrix, so let's inherits its traits:
  template<>
  struct traits<MKLSparseMatrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
  {};
}
}

// Example of a matrix-free wrapper from a user type to Eigen's compatible type
// For the sake of simplicity, this example simply wrap a Eigen::SparseMatrix.
class MKLSparseMatrix : public Eigen::EigenBase<MKLSparseMatrix> {
public:
  // Required typedefs, constants, and method:
  typedef double Scalar;
  typedef double RealScalar;
  typedef int StorageIndex;
  enum {
    ColsAtCompileTime = Eigen::Dynamic,
    MaxColsAtCompileTime = Eigen::Dynamic,
    IsRowMajor = true
  };

  Index rows() const { return _rows; }
  Index cols() const { return _cols; }
  Index nonZero() const { return _nnz; }

  template<typename Rhs>
  Eigen::Product<MKLSparseMatrix,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MKLSparseMatrix,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }

  MKLSparseMatrix(const Eigen::SparseMatrix<Scalar, Eigen::RowMajor> &mat) : 
  _rows(mat.rows()), _cols(mat.cols()), _nnz(mat.nonZeros()) {
    auto &mat_ = const_cast<Eigen::SparseMatrix<Scalar, Eigen::RowMajor>&>(mat);

    sparse_status_t status = mkl_sparse_d_create_csr(
        &mkl, SPARSE_INDEX_BASE_ZERO, rows(), cols(), 
        mat_.outerIndexPtr(), mat_.outerIndexPtr() + 1,
        mat_.innerIndexPtr(), mat_.valuePtr());
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("mkl sparse create failed");

/*
    status = mkl_sparse_optimize(mkl);
    if (status != SPARSE_STATUS_SUCCESS)
      throw std::runtime_error("mkl sparse optimise failed");
*/

    desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    desc.diag = SPARSE_DIAG_NON_UNIT;
  }
  ~MKLSparseMatrix() {
    mkl_sparse_destroy(mkl);
  }

  sparse_matrix_t mkl;
  struct matrix_descr desc;
private:
  const StorageIndex _rows, _cols, _nnz;
};


// Implementation of MKLSparseMatrix * Eigen::DenseVector though a specialization of internal::generic_product_impl:
namespace Eigen {
namespace internal {

  template<typename Rhs>
  struct generic_product_impl<MKLSparseMatrix, Rhs, SparseShape, DenseShape, GemvProduct> // GEMV stands for matrix-vector
  : generic_product_impl_base<MKLSparseMatrix,Rhs,generic_product_impl<MKLSparseMatrix,Rhs> >
  {
    typedef typename Product<MKLSparseMatrix,Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& y, const MKLSparseMatrix& A, const Rhs& x, const Scalar& alpha)
    {
      auto _x = const_cast<Scalar*>(x.data());
      auto _y = const_cast<Scalar*>(y.data());

      sparse_status_t status = mkl_sparse_d_mv(
          SPARSE_OPERATION_NON_TRANSPOSE, alpha, A.mkl, A.desc, _x, 0.0, _y);
      if (status != SPARSE_STATUS_SUCCESS)
        throw std::runtime_error("mkl sparse mv failed");
    }
  };

}
}