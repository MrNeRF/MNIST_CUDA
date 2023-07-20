#include "activation.cuh"
#include "error_checking.cuh"

__global__ void ReLuKernel(float* input,
                           const int size);

__global__ void LogSoftmaxKernel(const int output_size,
                                 const int batch_size,
                                 float* values);

void ReLU::operator()(const int output_size,
                      const int batch_size,
                      float* d_value) {
    const int total_size = batch_size * output_size;
    const int threadsPerBlock = 256;
    const int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    ReLuKernel<<<numBlocks, threadsPerBlock>>>(d_value, total_size);
    CHECK_LAST_CUDA_ERROR();
}

void LogSoftMax::operator()(const int output_size,
                            const int batch_size,
                            float* d_value) {
    const int total_size = batch_size * output_size;
    const int threadsPerBlock = 256;
    const int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    //__global__ void LogSoftmaxBatkh(const float* input, float* output, const int outputSize, const int batchSize) {
    LogSoftmaxKernel<<<numBlocks, threadsPerBlock>>>(d_value, batch_size, output_size);
    CHECK_LAST_CUDA_ERROR();
}

__global__ void ReLuKernel(float* input,
                           const int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        input[idx] = fmaxf(0, input[idx]);
    }
}

__global__ void LogSoftmaxKernel(const int output_size,
                                 const int batch_size,
                                 float* values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
        return;

    // Calculate max and sum for each batch
    // if we get to big numbers, we can get inf or nan
    // That's why we subtract the max from each element
    // This is called the log-sum-exp trick and is used to avoid overflow

    float maxInput = -INFINITY;
    for (int j = 0; j < output_size; ++j) {
        maxInput = fmaxf(maxInput, values[idx * output_size + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < output_size; ++j) {
        // Attention! This has issues with overflow if input is too big since it gets exp(input)
        // TODO: Implement a more stable version
        sum += expf(values[idx * output_size + j] - maxInput);
    }

    // Calculate log softmax for each element in the batch that this thread should process
    for (int j = 0; j < output_size; ++j) {
        float softmax = expf(values[idx * output_size + j] - maxInput) / sum;
        values[idx * output_size + j] = logf(softmax + 1e-8f);
    }
}