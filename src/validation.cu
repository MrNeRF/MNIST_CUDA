#include "error_checking.cuh"
#include "validation.cuh"
#include <vector>

__global__ void CorrectPredictionKernel(const int* predictions,
                                        const int* labels,
                                        int* d_correctCount,
                                        int batchSize);

__global__ void ArgMaxKernel(const float* predictions,
                             int* outputLabels,
                             const int numClasses,
                             const int batchSize);

float ComputeAccuracy(const int* d_predictions, const int* d_labels, int batchSize) {
    int* d_correctCount;
    cudaMalloc((void**)&d_correctCount, sizeof(int));
    cudaMemset(d_correctCount, 0, sizeof(int));

    CorrectPredictionKernel<<<(batchSize + 255) / 256, 256>>>(d_predictions,
                                                              d_labels,
                                                              d_correctCount,
                                                              batchSize);

    int correctCount;
    cudaMemcpy(&correctCount, d_correctCount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_correctCount);

    return (float)correctCount / batchSize;
}

void ArgMax(const float* d_predictions, int* d_labels, int numClasses, int batch_size) {

    ArgMaxKernel<<<(batch_size + 255) / 256, 256>>>(d_predictions, d_labels, 10, batch_size);
}

__global__ void CorrectPredictionKernel(const int* predictions,
                                        const int* labels,
                                        int* d_correctCount,
                                        int batchSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batchSize) {
        if (predictions[i] == labels[i]) {
            atomicAdd(d_correctCount, 1);
        }
    }
}

__global__ void ArgMaxKernel(const float* predictions,
                             int* outputLabels,
                             const int numClasses,
                             const int batchSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    int predictedClass = 0;
    float maxProb = predictions[idx * numClasses];

    for (int i = 1; i < numClasses; ++i) {
        float prob = predictions[idx * numClasses + i];
        if (prob > maxProb) {
            predictedClass = i;
            maxProb = prob;
        }
    }

    outputLabels[idx] = predictedClass;
}