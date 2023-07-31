#include "error_checking.cuh"
#include "loss.cuh"
#include <cuda_runtime.h>

__global__ void ComputeDzLastLayerKernel(const float* log_predictions,
                                         const int* labels,
                                         float* dZ,
                                         const int numClasses,
                                         const int batchSize);

__global__ void CrossEntropyLossKernel(const float* predictions,
                                       const int* labels,
                                       float* loss,
                                       const int numClasses,
                                       const int batchSize);

__global__ void LogSoftmaxCrossEntropyLossKernel(const float* values,
                                                 float* predictions,
                                                 const int* labels,
                                                 float* loss,
                                                 const int output_size,
                                                 const int batch_size);

CrossEntropyLoss::CrossEntropyLoss(const int num_classes, const int batch_size) : _num_classes(num_classes),
                                                                                  _batch_size(batch_size) {
    CHECK_CUDA_ERROR(cudaMalloc(&_d_dZ, sizeof(float) * num_classes * batch_size));
    CHECK_CUDA_ERROR(cudaMalloc(&_d_loss, sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&_d_predictions, sizeof(float) * num_classes * batch_size));
}

CrossEntropyLoss::~CrossEntropyLoss() {
    CHECK_CUDA_ERROR(cudaFree(_d_dZ));
    CHECK_CUDA_ERROR(cudaFree(_d_loss));
    CHECK_CUDA_ERROR(cudaFree(_d_predictions));
}

float CrossEntropyLoss::operator()(const float* d_values, const int* d_labels) {
    float loss;
    const int threadsPerBlock = 50;
    const int blocksPerGrid = (_batch_size + threadsPerBlock - 1) / threadsPerBlock;
    CHECK_CUDA_ERROR(cudaMemset(_d_loss, 0, sizeof(float)));

    LogSoftmaxCrossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock>>>(d_values,
                                                                         _d_predictions,
                                                                         d_labels,
                                                                         _d_loss,
                                                                         _num_classes,
                                                                         _batch_size);

    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaMemcpy(&loss, _d_loss, sizeof(float), cudaMemcpyDeviceToHost));

    return loss / _batch_size;
}

float* CrossEntropyLoss::Backward(const int* d_labels) {

    const int threadsPerBlock = 50;
    const int blocksPerGrid = (_batch_size + threadsPerBlock - 1) / threadsPerBlock;

    ComputeDzLastLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(_d_predictions,
                                                                 d_labels,
                                                                 _d_dZ,
                                                                 _num_classes,
                                                                 _batch_size);

    CHECK_LAST_CUDA_ERROR();
    return _d_dZ;
}

__global__ void CrossEntropyLossKernel(const float* predictions,
                                       const int* labels,
                                       float* loss,
                                       const int numClasses,
                                       const int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
        return;

    const int label = labels[idx];
    const float prediction = predictions[idx * numClasses + label];
    atomicAdd(loss, -prediction);
}

__global__ void LogSoftmaxCrossEntropyLossKernel(const float* values,
                                                 float* predictions,
                                                 const int* labels,
                                                 float* loss,
                                                 const int output_size,
                                                 const int batch_size) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batch_size)
        return;

    float maxInput = -INFINITY;
    for (int j = 0; j < output_size; ++j) {
        maxInput = fmaxf(maxInput, values[idx * output_size + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < output_size; ++j) {
        sum += expf(values[idx * output_size + j] - maxInput);
    }

    const int label = labels[idx];
    float log_softmax = 0.0f;

    for (int j = 0; j < output_size; ++j) {
        float softmax = expf(values[idx * output_size + j] - maxInput) / sum;
        predictions[idx * output_size + j] = logf(softmax + 1e-8f);

        if (j == label) {
            log_softmax = predictions[idx * output_size + j];
        }
    }

    atomicAdd(loss, -log_softmax);
}

__global__ void ComputeDzLastLayerKernel(const float* log_predictions, const int* labels, float* dZ, const int numClasses, const int batchSize) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    for (int i = 0; i < numClasses; i++) {
        const float softmax_output = expf(log_predictions[idx * numClasses + i]); // convert log_softmax to softmax
        dZ[idx * numClasses + i] = softmax_output - (i == labels[idx] ? 1.0f : 0.0f);
    }
}