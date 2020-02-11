#include "cg.h"
#include "DistributedVector.h"
#include <iostream>

//-----------------------------------------------------------------------------
// Untested CG solver
void cg(const Eigen::Ref<const Eigen::SparseMatrix<double, Eigen::RowMajor>>& A,
        const Eigen::Ref<const Eigen::VectorXd>& b)
{
  int M = A.rows();

  DistributedVector psp(MPI_COMM_WORLD, A);
  auto p = psp.vec();

  // Residual vector
  Eigen::VectorXd r(M);
  r = b;
  // Assign to dense part of sparse vector
  p = r;
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(M);

  double rnorm = r.squaredNorm();
  double rnorm_sum1;
  MPI_Allreduce(&rnorm, &rnorm_sum1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Iterations of CG
  for (int k = 0; k < 500; ++k)
  {
    // y = A.p
    psp.update();
    y = A * psp.spvec();

    // Update x and r
    double pdoty = p.dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double alpha = rnorm_sum1 / pdoty_sum;
    x += alpha * p;
    r -= alpha * y;

    // Update p
    rnorm = r.squaredNorm();
    double rnorm_sum2;
    MPI_Allreduce(&rnorm, &rnorm_sum2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double beta = rnorm_sum2 / rnorm_sum1;
    rnorm_sum1 = rnorm_sum2;

    p *= beta;
    p += r;
    std::cerr << k << ":" << rnorm << "\n";
  }
}