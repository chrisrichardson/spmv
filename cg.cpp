// Copyright (C) 2020 Chris Richardson (chris@bpi.cam.ac.uk) and Jeffrey Salmond
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef EIGEN_USE_MKL_ALL
#include <mkl.h>
#endif

#include <iostream>

#include "L2GMap.h"
#include "cg.h"

#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#endif

//-----------------------------------------------------------------------------
std::tuple<Eigen::VectorXd, int>
spmv::cg(MPI_Comm comm,
         Eigen::Ref<Eigen::SparseMatrix<double, Eigen::RowMajor>> A,
         const std::shared_ptr<const spmv::L2GMap> l2g,
         const Eigen::Ref<const Eigen::VectorXd>& b, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

#ifdef EIGEN_USE_MKL_ALL
  sparse_matrix_t A_mkl;
  sparse_status_t status = mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, A.rows(), A.cols(), A.outerIndexPtr(),
      A.outerIndexPtr() + 1, A.innerIndexPtr(), A.valuePtr());
  assert(status == SPARSE_STATUS_SUCCESS);

  status = mkl_sparse_optimize(A_mkl);
  assert(status == SPARSE_STATUS_SUCCESS);

  if (status != SPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create MKL matrix");

  struct matrix_descr mat_desc;
  mat_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  mat_desc.diag = SPARSE_DIAG_NON_UNIT;
#endif

  int M = A.rows();

  // Residual vector
  Eigen::VectorXd r(M);
  Eigen::VectorXd y(M);
  Eigen::VectorXd x(M);
  Eigen::VectorXd psp(l2g->local_size(true));
  Eigen::Map<Eigen::VectorXd> p(psp.data(), M);

  // Assign to dense part of sparse vector
  r = b;
  p = b;
  x.setZero();

  double rnorm = r.squaredNorm();
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  double rnorm_old = rnorm0;
  for (int k = 0; k < kmax; ++k)
  {
    // y = A.p
    l2g->update(psp.data());

#ifdef EIGEN_USE_MKL_ALL
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, mat_desc,
                    psp.data(), 0.0, y.data());
#else
    y = A * psp;
#endif

    // Calculate alpha = r.r/p.y
    double pdoty = p.dot(y);
    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = rnorm_old / pdoty_sum;

    // Update x and r
    x += alpha * p;
    r -= alpha * y;

    // Update rnorm
    rnorm = r.squaredNorm();
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p
    p = p * beta + r;

    if (rnorm_new / rnorm0 < rtol)
      return {x, k};
  }
  return {x, kmax};
}

#ifdef HAVE_CUDA
//-----------------------------------------------------------------------------
#define cuda_CHECK(x)                                                          \
  if (x != cudaSuccess)                                                        \
  throw std::runtime_error(#x " failed")
#define cusparse_CHECK(x)                                                      \
  if (x != CUSPARSE_STATUS_SUCCESS)                                            \
  throw std::runtime_error(#x " failed")
//-----------------------------------------------------------------------------
std::tuple<Eigen::VectorXd, int>
spmv::cg_cuda(MPI_Comm comm,
              Eigen::Ref<Eigen::SparseMatrix<double, Eigen::RowMajor>> A,
              const std::shared_ptr<const spmv::L2GMap> l2g,
              const Eigen::Ref<const Eigen::VectorXd>& b, int kmax, double rtol)
{
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);

  // Initialise cuBLAS
  cublasHandle_t blas_handle;
  cublasCreate(&blas_handle);

  // Copy A over to gpu
  cusparseHandle_t handle;
  cusparseSpMatDescr_t spmat;
  double* value;
  int *inner, *outer;

  cusparse_CHECK(cusparseCreate(&handle));
  cusparse_CHECK(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));

  int nnz = A.nonZeros();
  int rows = A.rows();
  int cols = A.cols();

  cuda_CHECK(cudaMalloc(&value, nnz * sizeof(double)));
  cuda_CHECK(cudaMalloc(&inner, nnz * sizeof(int)));
  cuda_CHECK(cudaMalloc(&outer, (rows + 1) * sizeof(int)));

  cuda_CHECK(cudaMemcpy(value, A.valuePtr(), nnz * sizeof(double),
                        cudaMemcpyHostToDevice));
  cuda_CHECK(cudaMemcpy(inner, A.innerIndexPtr(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  cuda_CHECK(cudaMemcpy(outer, A.outerIndexPtr(), (rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));

  cusparse_CHECK(cusparseCreateCsr(&spmat, rows, cols, nnz, outer, inner, value,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  double* alpha_d;
  double* beta_d;
  cuda_CHECK(cudaMalloc(&alpha_d, sizeof(double)));
  cuda_CHECK(cudaMalloc(&beta_d, sizeof(double)));

  double alpha_h = 1, beta_h = 0;
  cuda_CHECK(
      cudaMemcpy(alpha_d, &alpha_h, sizeof(double), cudaMemcpyHostToDevice));
  cuda_CHECK(
      cudaMemcpy(beta_d, &beta_h, sizeof(double), cudaMemcpyHostToDevice));

  cusparseDnVecDescr_t vecY;
  double* y;
  cuda_CHECK(cudaMalloc(&y, rows * sizeof(double)));
  cusparse_CHECK(cusparseCreateDnVec(&vecY, rows, y, CUDA_R_64F));

  // Allocate p in GPU
  cusparseDnVecDescr_t vecP;
  double* psp;
  cudaMalloc(&psp, cols * sizeof(double));
  cusparse_CHECK(cusparseCreateDnVec(&vecP, cols, psp, CUDA_R_64F));

  // Solution vector x
  double* x;
  double zero = 0.0;
  cuda_CHECK(cudaMalloc(&x, rows * sizeof(double)));
  cublasDscal(blas_handle, rows, &zero, x, 1);

  // allocate scratch space
  size_t bufsize;
  void* scratch;
  cusparse_CHECK(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha_d, spmat, vecP, beta_d,
      vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufsize));
  cuda_CHECK(cudaMalloc(&scratch, bufsize));

  // Residual vector, r = b
  double* r;
  cuda_CHECK(cudaMalloc(&r, rows * sizeof(double)));
  cuda_CHECK(cudaMemcpy(r, b.data(), b.size() * sizeof(double),
                        cudaMemcpyHostToDevice));

  // p = r;
  cuda_CHECK(
      cudaMemcpy(psp, r, rows * sizeof(double), cudaMemcpyDeviceToDevice));

  double rnorm; // = r.squaredNorm();
  cublasDdot(blas_handle, rows, r, 1, r, 1, &rnorm);
  double rnorm0;
  MPI_Allreduce(&rnorm, &rnorm0, 1, MPI_DOUBLE, MPI_SUM, comm);

  // Iterations of CG
  double rnorm_old = rnorm0;
  for (int k = 0; k < kmax; ++k)
  {
    // y = A.p
    l2g->update(psp);
    cusparse_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                alpha_d, spmat, vecP, beta_d, vecY, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, scratch));

    // Calculate alpha = r.r/p.y
    double pdoty; // = p.dot(y);
    cublasDdot(blas_handle, rows, psp, 1, y, 1, &pdoty);

    double pdoty_sum;
    MPI_Allreduce(&pdoty, &pdoty_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
    double alpha = rnorm_old / pdoty_sum;

    // Update x and r
    //    x += alpha * p;
    cublasDaxpy(blas_handle, rows, &alpha, psp, 1, x, 1);

    //    r -= alpha * y;
    alpha = -alpha;
    cublasDaxpy(blas_handle, rows, &alpha, y, 1, r, 1);

    // Update rnorm
    //    rnorm = r.squaredNorm();
    cublasDdot(blas_handle, rows, r, 1, r, 1, &rnorm);
    double rnorm_new;
    MPI_Allreduce(&rnorm, &rnorm_new, 1, MPI_DOUBLE, MPI_SUM, comm);
    double beta = rnorm_new / rnorm_old;
    rnorm_old = rnorm_new;

    // Update p
    //    p *= beta;
    //    p += r;
    double one = 1.0;
    cublasDscal(blas_handle, rows, &beta, psp, 1);
    cublasDaxpy(blas_handle, rows, &one, r, 1, psp, 1);

    if (rnorm_new / rnorm0 < rtol)
    {
      Eigen::Matrix<double, Eigen::Dynamic, 1> x_eigen(rows);
      cuda_CHECK(cudaMemcpy(x_eigen.data(), x, rows * sizeof(double),
                            cudaMemcpyDeviceToHost));
      return {x_eigen, k};
    }
  }
  Eigen::Matrix<double, Eigen::Dynamic, 1> x_eigen(rows);
  cuda_CHECK(cudaMemcpy(x_eigen.data(), x, rows * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return {x_eigen, kmax};
}
#endif
