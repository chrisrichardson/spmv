#include "CUDA.h"
#include <cuda_runtime.h>
#include <library_types.h>

#define cuda_CHECK(x)if(x != cudaSuccess) throw std::runtime_error(#x " failed")
#define cusparse_CHECK(x) if(x != CUSPARSE_STATUS_SUCCESS) throw std::runtime_error(#x " failed")

OperatorCUDA::OperatorCUDA(Eigen::SparseMatrix<double, Eigen::RowMajor>& A) {
  cusparse_CHECK(cusparseCreate(&handle));
  cusparse_CHECK(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));

  // move all the crap to the GPU
  cuda_CHECK(cudaMalloc(&value, A.nonZeros() * sizeof(double)));
  cuda_CHECK(cudaMalloc(&inner, A.nonZeros() * sizeof(int)));
  cuda_CHECK(cudaMalloc(&outer, (A.rows() + 1) * sizeof(int)));

  cuda_CHECK(cudaMemcpy(value, A.valuePtr(), A.nonZeros() * sizeof(double), cudaMemcpyHostToDevice));
  cuda_CHECK(cudaMemcpy(inner, A.innerIndexPtr(), A.nonZeros() * sizeof(int), cudaMemcpyHostToDevice));
  cuda_CHECK(cudaMemcpy(outer, A.outerIndexPtr(), (A.rows() + 1) * sizeof(int), cudaMemcpyHostToDevice));

  cusparse_CHECK(cusparseCreateCsr(&spmat, A.rows(), A.cols(), A.nonZeros(),
                                   outer, inner, value, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F));
  // move constants????
  cuda_CHECK(cudaMalloc(&alpha, sizeof(double)));
  cuda_CHECK(cudaMalloc(&beta, sizeof(double)));

  double alpha_h = 1, beta_h = 0;
  cuda_CHECK(cudaMemcpy(alpha, &alpha_h, sizeof(double), cudaMemcpyHostToDevice));
  cuda_CHECK(cudaMemcpy(beta, &beta_h, sizeof(double), cudaMemcpyHostToDevice));

  // create vector descriptors
  cuda_CHECK(cudaMalloc(&xdata, A.rows() * sizeof(double)));
  cusparse_CHECK(cusparseCreateDnVec(&vecX, A.rows(), xdata, CUDA_R_64F));

  cuda_CHECK(cudaMalloc(&ydata, A.cols() * sizeof(double)));
  cusparse_CHECK(cusparseCreateDnVec(&vecY, A.cols(), ydata, CUDA_R_64F));

  // allocate scratch space
  size_t bufsize;
  cusparse_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, alpha,
                                spmat, vecX, beta, vecY, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, &bufsize));
  cuda_CHECK(cudaMalloc(&scratch, bufsize));
}

OperatorCUDA::~OperatorCUDA() {
    cudaFree(alpha);
    cudaFree(beta);
    cudaFree(scratch);
    cusparseDestroyDnVec(vecX); cudaFree(xdata);
    cusparseDestroyDnVec(vecY); cudaFree(ydata);
    cudaFree(value);
    cudaFree(inner);
    cudaFree(outer);
    cusparseDestroySpMat(spmat);
    cusparseDestroy(handle);
}

Eigen::VectorXd OperatorCUDA::apply(Eigen::VectorXd &psp) const {
    cuda_CHECK(cudaMemcpy(ydata, psp.data(), psp.size()*sizeof(double), cudaMemcpyHostToDevice));

    cusparse_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha, spmat, vecX, beta, vecY, CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, scratch));

    Eigen::VectorXd q(psp.size());
    cuda_CHECK(cudaMemcpy(q.data(), ydata, q.size()*sizeof(double), cudaMemcpyDeviceToHost));
    return q;
}
