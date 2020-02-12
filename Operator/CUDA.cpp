#include "CUDA.h"
#include <cuda_runtime.h>
#include <library_types.h>


OperatorCUDA::OperatorCUDA(Eigen::SparseMatrix<double, Eigen::RowMajor>& A) {
    cusparseStatus_t status;
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not initialize cusparse");

    //move all the crap to the GPU
    status = cusparseCreateCsr(&spmat, A.rows(), A.cols(), A.nonZeros(),
    A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(),
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not create cusparse CSR matrix");

    //move constants????
    cudaMalloc(&alpha, sizeof(double));
    cudaMalloc(&beta, sizeof(double));

    double alpha_h = 1, beta_h = 1;
    cudaMemcpy(alpha, &alpha_h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(beta, &beta_h, sizeof(double), cudaMemcpyHostToDevice);

    //create vector descriptors
    cudaMalloc(&xdata, A.rows()*sizeof(double));
    cusparseCreateDnVec(&vecX, A.rows(), xdata, CUDA_R_64F);

    cudaMalloc(&ydata, A.cols()*sizeof(double));
    cusparseCreateDnVec(&vecX, A.cols(), ydata, CUDA_R_64F);

    //allocate scratch space
    size_t bufsize;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    alpha, spmat, vecX, beta, vecY, CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufsize);
    if (status != CUSPARSE_STATUS_SUCCESS)
    throw std::runtime_error("Could not get cusparse SpMV buffer size");
    cudaMalloc(&scratch, bufsize);
}

OperatorCUDA::~OperatorCUDA() {
    cusparseDestroySpMat(spmat);
    cusparseDestroy(handle);
}

Eigen::VectorXd OperatorCUDA::apply(Eigen::VectorXd &psp) const {
    cudaMemcpy(ydata, psp.data(), psp.size()*sizeof(double), cudaMemcpyHostToDevice);

    cusparseStatus_t status;
    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha, spmat, vecX, beta, vecY, CUDA_R_64F, 
        CUSPARSE_MV_ALG_DEFAULT, scratch);
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error("Could not perform cusparse SpMV");

    Eigen::VectorXd q(psp.size());
    cudaMemcpy(q.data(), ydata, q.size()*sizeof(double), cudaMemcpyDeviceToHost);
    return q;
}