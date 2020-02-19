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

  Index rows() const { return mp_mat.rows(); }
  Index cols() const { return mp_mat.cols(); }

  template<typename Rhs>
  Eigen::Product<MKLSparseMatrix,Rhs,Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
    return Eigen::Product<MKLSparseMatrix,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
  }

  MKLSparseMatrix(Eigen::SparseMatrix<double> &mat) : mp_mat(mat) {
    mkl_sparse_d_create_csr(
        &mkl, SPARSE_INDEX_BASE_ZERO, rows(), cols(), mp_mat.outerIndexPtr(),
        mp_mat.outerIndexPtr() + 1, mp_mat.innerIndexPtr(), mp_mat.valuePtr());

    mkl_sparse_optimize(mkl);

    desc.type = SPARSE_MATRIX_TYPE_GENERAL;
    desc.diag = SPARSE_DIAG_NON_UNIT;
  }

  sparse_matrix_t mkl;
  struct matrix_descr desc;
private:
  Eigen::SparseMatrix<double> &mp_mat;
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
    static void scaleAndAddTo(Dest& x, const MKLSparseMatrix& A, const Rhs& y, const Scalar& alpha)
    {
        assert(alpha==Scalar(1) && "scaling is not implemented");
        EIGEN_ONLY_USED_FOR_DEBUG(alpha);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A.mkl, A.desc,
                        const_cast<Scalar*>(x.data()), 0.0, 
                        const_cast<Scalar*>(y.data()));
    }
  };

}
}