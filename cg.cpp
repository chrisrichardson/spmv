#include "cg.h"
#include "L2GMap.h"
#include <iostream>

//-----------------------------------------------------------------------------
// Untested CG solver
Eigen::VectorXd
cg(MPI_Comm comm,
   const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
   const std::shared_ptr<const L2GMap> l2g,
   const Eigen::Ref<const Eigen::VectorXd>& b)
{
  int M = A.rows();

  Eigen::VectorXd psp(l2g->local_size());

  // Residual vector
  Eigen::VectorXd r(M);
  r = b;
  // Assign to dense part of sparse vector
  psp.head(M) = r;
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(M);

  double rnorm = r.squaredNorm();
  double rnorm_sum1;
  MPI_Allreduce(&rnorm, &rnorm_sum1, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  for (int k = 0; k < 500; ++k)
  {
    // y = A.p
    l2g->update(psp.data());
    y = A * psp;

    // Update x and r
    double pdoty = psp.head(M).dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    double alpha = rnorm_sum1 / pdoty_sum;
    x += alpha * psp.head(M);
    r -= alpha * y;

    // Update p
    rnorm = r.squaredNorm();
    double rnorm_sum2;
    MPI_Allreduce(&rnorm, &rnorm_sum2, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_sum2 / rnorm_sum1;
    rnorm_sum1 = rnorm_sum2;

    psp.head(M) *= beta;
    psp.head(M) += r;
    std::cerr << k << ":" << rnorm << "\n";
  }
  return x;
}
