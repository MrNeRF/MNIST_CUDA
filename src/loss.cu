#include "error_checking.cuh"
#include "loss.cuh"
#include <cuda_runtime.h>

__global__ void CrossEntropyLossKernel(const float* predictions,
                                       const int* labels,
                                       float* loss,
                                       const int numClasses,
                                       const int batchSize);

float CrossEntropyLoss(const float* d_predictions,
                       const int* d_labels,
                       const int numClasses,
                       const int batchSize) {
    float loss;
    float* d_loss;
    const int threadsPerBlock = 50;
    const int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, sizeof(float))); // Allocate device memory for loss
    CHECK_CUDA_ERROR(cudaMemset(d_loss, 0, sizeof(float)));

    CrossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions,
                                                               d_labels,
                                                               d_loss,
                                                               numClasses,
                                                               batchSize);

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaFree(d_loss));

    return loss / batchSize;
}

__global__ void CrossEntropyLossKernel(const float* predictions,
                                       const int* labels,
                                       float* loss,
                                       const int numClasses,
                                       const int batchSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    int label = labels[idx];
    float prediction = predictions[idx * numClasses + label];
    atomicAdd(loss, -prediction);
}