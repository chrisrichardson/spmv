#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>

namespace spmv
{
class L2GMap;

void jacobi(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
            std::shared_ptr<const L2GMap> l2g, Eigen::Ref<Eigen::VectorXd> x,
            const Eigen::Ref<const Eigen::VectorXd>& b,
            const Eigen::Ref<const Eigen::VectorXd>& D);
} // namespace spmv
