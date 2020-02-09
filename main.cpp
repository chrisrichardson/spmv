// Copyright (C) 2018 Chris Richardson (chris@bpi.cam.ac.uk)
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <chrono>
#include <iostream>
#include <memory>

#ifdef HAS_MKL
#include <mkl.h>
#endif

#include <mpi.h>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "DistributedVector.h"
#include "read_petsc.h"

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
//-----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Read file created with "-ksp_view_mat binary" option
  auto A = read_petsc_binary(MPI_COMM_WORLD, "binaryoutput");

  // Get local range from number of rows in A
  std::vector<int> nrows_all(mpi_size);
  std::vector<index_type> ranges = {0};
  int nrows = A.rows();
  MPI_Allgather(&nrows, 1, MPI_INT, nrows_all.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
  for (int i = 0; i < mpi_size; ++i)
    ranges.push_back(ranges.back() + nrows_all[i]);

  int N = ranges.back();
  int M = A.rows();
  int r0 = ranges[mpi_rank];

  std::cout << "# rank = " << mpi_rank << "/" << mpi_size << "\n";

#ifdef HAS_MKL
  // Remap columns to local indexing for MKL
  std::map<int, int> global_to_local;
  std::vector<MKL_INT> columns(A.outerIndexPtr()[M]);
  for (std::size_t i = 0; i < columns.size(); ++i)
  {
    int global_index = A.innerIndexPtr()[i];
    global_to_local.insert({global_index, 0});
  }

  int lc = 0;
  for (auto& q : global_to_local)
    q.second = lc++;

  for (std::size_t i = 0; i < columns.size(); ++i)
  {
    int global_index = A.innerIndexPtr()[i];
    columns[i] = global_to_local[global_index];
  }

  sparse_matrix_t A_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, M, N, A.outerIndexPtr(),
      A.outerIndexPtr() + 1, columns.data(), A.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(A_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

  struct matrix_descr mat_desc;
  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;

#endif

  // Make distributed vector - this is the only
  // one that needs to be 'sparse'
  auto psp = std::make_shared<DistributedVector>(MPI_COMM_WORLD, A);
  auto p = psp->vec();

  // Set up values
  for (int i = 0; i < M; ++i)
  {
    double z = (double)(i + r0) / double(N);
    p[i] = exp(-10 * pow(5 * (z - 0.5), 2.0));
  }

  // Apply matrix a few times

  auto start = std::chrono::system_clock::now();

  // Temporary variable

  Eigen::VectorXd q(p.size());
  for (int i = 0; i < 100; ++i)
  {
    psp->update();

#ifdef HAS_MKL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                    psp->spvec().valuePtr(), 0.0, q.data());
#else
    q = A * psp->spvec();
#endif

    p = q;
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end - start;

  double pnorm = p.squaredNorm();
  double pnorm_sum;
  MPI_Allreduce(&pnorm, &pnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  if (mpi_rank == 0)
  {
    std::cout << "time = " << diff.count() << "s.\n";
    std::cout << "norm = " << pnorm_sum << "\n";
  }

  // // Output
  // std::stringstream s;
  // s << rank << " [";
  // for (int i = 0; i < M; ++i)
  //   s << p[i] << " ";
  // s << "]";

  // for (int i = 0; i < size; ++i)
  // {
  //   if (i == rank)
  //     std::cout << s.str() << "\n";
  //   MPI_Barrier(MPI_COMM_WORLD);
  // }

  // Destroy here before MPI_Finalize, because it holds a comm
  psp.reset();

  MPI_Finalize();
  return 0;
}
