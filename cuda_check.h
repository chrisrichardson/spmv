
#pragma once
#ifdef HAVE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#define cuda_CHECK(x)                                                   \
  if (x != cudaSuccess)                                                 \
    throw std::runtime_error(#x " failed")
#define cusparse_CHECK(x)                                               \
  if (x != CUSPARSE_STATUS_SUCCESS)                                     \
    throw std::runtime_error(#x " failed")
#define cublas_CHECK(x)                                                 \
  if (x != CUBLAS_STATUS_SUCCESS)                                       \
  throw std::runtime_error(#x " failed")

#endif
