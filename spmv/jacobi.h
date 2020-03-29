#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

namespace spmv
{
template <typename T>
class Matrix;

double jacobi(const spmv::Matrix<double>& A, Eigen::VectorXd& x,
              const Eigen::VectorXd& b);
} // namespace spmv
