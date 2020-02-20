
#include <Eigen/Dense>
#include <Eigen/Sparse>

Eigen::VectorXd
extract_diagonal(Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
{
  const int* inner = mat.innerIndexPtr();
  const int* outer = mat.outerIndexPtr();
  const double* val = mat.valuePtr();
  Eigen::VectorXd result(mat.rows());
  result.setZero();

  for (int i = 0; i < mat.rows(); ++i)
  {
    for (int j = outer[i]; j < outer[i + 1]; ++j)
    {
      if (inner[j] == i)
        result[i] = val[j];
    }
  }

  return result;
}
