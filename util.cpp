
#include "util.h"

//-----------------------------------------------------------------------------
Eigen::VectorXd
extract_diagonal(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
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
//-----------------------------------------------------------------------------
std::vector<int>
diagonal_block_nnz(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat)
{
  const int* inner = mat.innerIndexPtr();
  const int* outer = mat.outerIndexPtr();
  const int rows = mat.rows();
  const int cols = rows;

  std::vector<int> innernnz(rows, 0);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = outer[i]; j < outer[i + 1]; ++j)
    {
      if (inner[j] < cols)
        ++innernnz[i];
    }
  }

  return innernnz;
}
//-----------------------------------------------------------------------------
