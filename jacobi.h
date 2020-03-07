#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

namespace spmv
{
class Matrix;


 void jacobi(const spmv::Matrix& A,
             Eigen::Ref<Eigen::VectorXd> x,
             const Eigen::Ref<const Eigen::VectorXd>& b,
             const Eigen::Ref<const Eigen::VectorXd>& D);
} // namespace spmv
