#include "activation.cuh"
#include "error_checking.cuh"

__global__ void reluKernel(float* input,
                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        input[idx] = fmaxf(0, input[idx]);
    }
}

void ReLU::operator()(float* d_output) {
    int total_size = _batch_size * _output_size;
    int threadsPerBlock = 256;
    int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    reluKernel<<<numBlocks, threadsPerBlock>>>(d_output, total_size);
    CHECK_LAST_CUDA_ERROR();
}
