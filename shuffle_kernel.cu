#include "shuffle_kernel.h"

__global__ void shuffle_kernel(double *databuf, double *vec_data, int *_indexbuf, int indexbuf_sz) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i > indexbuf_sz)
        return;

    databuf[i] = vec_data[_indexbuf[i]];
}

void do_shuffle(double *databuf, double *vec_data, const std::vector<int>& _indexbuf) {

  int block_size = 16;
  int blocks = std::ceil(_indexbuf.size()/double(block_size));

  int *_indexbuf_d;
  cudaMalloc(&_indexbuf_d, sizeof(int)*_indexbuf.size());
  cudaMemcpy(_indexbuf_d, _indexbuf.data(), sizeof(int)*_indexbuf.size(), cudaMemcpyHostToDevice);

  shuffle_kernel<<<blocks, block_size>>>(databuf, vec_data, _indexbuf_d, _indexbuf.size());
}