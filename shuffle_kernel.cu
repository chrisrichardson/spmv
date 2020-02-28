#include "shuffle_kernel.h"

__global__ void shuffle_kernel(double *databuf, double *vec_data, int *_indexbuf, int indexbuf_sz) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < indexbuf_sz)
        return;

    databuf[i] = vec_data[_indexbuf[i]];
}

void do_shuffle(double *databuf, double *vec_data, int *indexbuf, int n) {

  int block_size = 16;
  int blocks = std::ceil(n/double(block_size));

  shuffle_kernel<<<blocks, block_size>>>(databuf, vec_data, indexbuf, n);
}