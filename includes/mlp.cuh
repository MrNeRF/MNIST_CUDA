#pragma once

#include "error_checking.cuh"
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

struct DenseLayer {
    int inputSize;
    int outputSize;

    float* weights; // GPU pointer
    float* biases;  // GPU pointer

    float* m_weights; // GPU pointer
    float* v_weights; // GPU pointer
    float* m_biases;  // GPU pointer
    float* v_biases;  // GPU pointer
    float beta1;
    float beta2;
    float epsilon;
    int t;

    DenseLayer(int inputSize, int outputSize)
        : inputSize(inputSize),
          outputSize(outputSize) {
        // Allocate GPU memory for weights and biases
        CHECK_CUDA_ERROR(cudaMalloc((void**)&weights, inputSize * outputSize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&biases, outputSize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_weights, inputSize * outputSize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&v_weights, inputSize * outputSize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&m_biases, outputSize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&v_biases, outputSize * sizeof(float)));
        beta1 = 0.9f;
        beta2 = 0.999f;
        epsilon = 1e-8f;
        t = 0;
    }

    ~DenseLayer() {
        // Deallocate GPU memory
        CHECK_CUDA_ERROR(cudaFree(weights));
        CHECK_CUDA_ERROR(cudaFree(biases));
        CHECK_CUDA_ERROR(cudaFree(m_weights));
        CHECK_CUDA_ERROR(cudaFree(v_weights));
        CHECK_CUDA_ERROR(cudaFree(m_biases));
        CHECK_CUDA_ERROR(cudaFree(v_biases));
    }
};

void InitializeLayer(DenseLayer& layer) {
    // Allocate memory on the host
    float* h_weights = new float[layer.inputSize * layer.outputSize];
    float* h_biases = new float[layer.outputSize];

    // Initialize weights and biases on the host
    float stddev = sqrtf(2.0f / layer.inputSize); // He initialization
    for (int i = 0; i < layer.inputSize * layer.outputSize; i++) {
        h_weights[i] = stddev * (rand() / (RAND_MAX + 1.0f));
    }

    for (int i = 0; i < layer.outputSize; i++) {
        h_biases[i] = 0.0f;
    }

    // Copy initialized weights and biases to the device
    CHECK_CUDA_ERROR(cudaMemcpy(layer.weights, h_weights, layer.inputSize * layer.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(layer.biases, h_biases, layer.outputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Free host memory
    delete[] h_weights;
    delete[] h_biases;
}

__global__ void CrossEntropyLoss(const float* predictions, const int* labels, float* loss, const int numClasses, const int batchSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    int label = labels[idx];
    float prediction = predictions[idx * numClasses + label];
    atomicAdd(loss, -prediction);
}

__global__ void LogSoftmaxBatch(const float* input, float* output, const int outputSize, const int batchSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    // Calculate max and sum for each batch
    // if we get to big numbers, we can get inf or nan
    // That's why we subtract the max from each element
    // This is called the log-sum-exp trick and is used to avoid overflow
    float maxInput = -INFINITY;
    for (int j = 0; j < outputSize; ++j) {
        maxInput = fmaxf(maxInput, input[idx * outputSize + j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < outputSize; ++j) {
        // Attention! This has issues with overflow if input is too big since it gets exp(input)
        // TODO: Implement a more stable version
        sum += expf(input[idx * outputSize + j] - maxInput);
    }

    // Calculate log softmax for each element in the batch that this thread should process
    for (int j = 0; j < outputSize; ++j) {
        float softmax = expf(input[idx * outputSize + j] - maxInput) / sum;
        output[idx * outputSize + j] = logf(softmax + 1e-8f);
    }
}

__global__ void ReLU(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        input[idx] = fmaxf(0, input[idx]);
    }
}

__global__ void addBias(float* output, float* biases, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;

    if (row < rows && col < cols) {
        output[idx] += biases[col];
    }
}
// Dense layer forward propagation for batches
// in summary: output = W * input + b
void ForwardPropagation(const DenseLayer& layer, const float* input, float* output, const bool relu, const int batchSize) {
    // Create a cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status;

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    CHECK_LAST_CUDA_ERROR();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                layer.outputSize, batchSize, layer.inputSize,
                &alpha,
                layer.weights, layer.inputSize,
                input, layer.inputSize,
                &beta,
                output, layer.outputSize);

    CHECK_LAST_CUDA_ERROR();

    // Add biases to each output in the batch := output + biases
    // float* d_ones; // Vector of ones
    // CHECK_CUDA_ERROR(cudaMalloc(&d_ones, batchSize * sizeof(float)));
    // CHECK_CUDA_ERROR(cudaMemset(d_ones, 5.f, batchSize * sizeof(float)));

    // float* d_biases; // Vector of ones
    // CHECK_CUDA_ERROR(cudaMalloc(&d_biases, layer.outputSize * sizeof(float)));
    // CHECK_CUDA_ERROR(cudaMemset(d_biases, 5.f, layer.outputSize * sizeof(float)));

    // // // output = output + alpha * ones * layer.biases
    // cublasSger(handle,
    //            batchSize, layer.outputSize,
    //            &alpha,
    //            d_biases, 1,
    //            d_ones, 1,
    //            output, batchSize);
    addBias<<<(batchSize * layer.outputSize + 255) / 256, 256>>>(output, layer.biases, batchSize, layer.outputSize);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    CHECK_LAST_CUDA_ERROR();

    // cudaFree(d_biases);
    // cudaFree(d_ones);

    // Apply ReLU activation function to each output in the batch
    if (relu) {
        ReLU<<<(batchSize * layer.outputSize + 255) / 256, 256>>>(output, batchSize * layer.outputSize);
        CHECK_LAST_CUDA_ERROR();
    }

    // Destroy the handle
    cublasDestroy(handle);
    CHECK_LAST_CUDA_ERROR();
}

// M = batchSize, N = outputSize, K = inputSize
__global__ void BackpropagationKernel(const float* dZ_next,
                                      const float* A_prev,
                                      const float* W,
                                      float* dW,
                                      float* db,
                                      float* dZ,
                                      const float* A,
                                      const int M,
                                      const int N,
                                      const int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate weights derivative
    if (idx < N * K) {
        const int n = idx / K;
        const int k = idx % K;

        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            sum += dZ_next[m * N + n] * A_prev[m * K + k];
        }
        dW[idx] = sum / M;
    }

    // Calculate biases derivative
    if (idx < N) {
        float sum = 0.0f;
        // iterate over all batches and sum the gradients
        // db dimensions are 1 x K
        for (int m = 0; m < M; ++m) {
            sum += dZ_next[m * N + idx];
        }
        db[idx] = sum / M;
    }

    if (dZ && idx < K) {
        for (int m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += dZ_next[m * N + n] * W[n * K + idx];
            }

            // Apply the derivative of ReLU activation function
            if (A) {
                sum = A[m * K + idx] > 0 ? sum : 0;
            }

            dZ[m * K + idx] = sum;
        }
    }
}

__global__ void InitCurandState(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void DropoutKernel(float* input, float* mask, float dropout_rate, int size, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        curandState localState = states[idx];
        float random_value = curand_uniform(&localState);
        states[idx] = localState;

        mask[idx] = random_value < dropout_rate ? 0.0f : 1.0f;
        input[idx] *= mask[idx];
    }
}

__global__ void ApplyMaskKernel(float* dA, float* mask, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        dA[idx] *= mask[idx];
    }
}

__global__ void AdamUpdateKernel(float* params, float* d_params, float* m, float* v, int size, float lr_t, float beta1, float beta2, float epsilon) {
    // calculate the index for the weight/bias
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // only execute if the index is within the size of the weights/biases
    if (idx < size) {
        // compute the new moving average of the gradient
        m[idx] = beta1 * m[idx] + (1 - beta1) * d_params[idx];

        // compute the new moving average of the squared gradient
        v[idx] = beta2 * v[idx] + (1 - beta2) * d_params[idx] * d_params[idx];

        // update the weights/biases
        params[idx] -= lr_t * m[idx] / (sqrt(v[idx]) + epsilon);
    }
}

void performAdamOptimization(DenseLayer& layer, float* dW, float* db, float lr, float beta1, float beta2, float epsilon) {
    // increment the timestep
    layer.t += 1;

    // calculate the learning rate adjustment
    float lr_t = lr * sqrt(1 - pow(beta2, layer.t)) / (1 - pow(beta1, layer.t));

    // launch the CUDA kernel to do the Adam update
    int numBlocks = (layer.inputSize * layer.outputSize + 255) / 256;
    AdamUpdateKernel<<<numBlocks, 256>>>(layer.weights, dW, layer.m_weights, layer.v_weights, layer.inputSize * layer.outputSize, lr_t, beta1, beta2, epsilon);
    cudaDeviceSynchronize(); // synchronize here

    numBlocks = (layer.outputSize + 255) / 256;
    AdamUpdateKernel<<<numBlocks, 256>>>(layer.biases, db, layer.m_biases, layer.v_biases, layer.outputSize, lr_t, beta1, beta2, epsilon);
    cudaDeviceSynchronize(); // and here
}

__global__ void GradientDescentUpdateKernel(float* params, const float* gradients, const int size, const float learning_rate) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * gradients[idx];
    }
}

void ComputeGradients(DenseLayer& layer,
                      float* derivative_Loss_Z_nextLayer,
                      float* activations_previousLayer,
                      float* derivative_Loss_Weights,
                      float* derivative_Loss_Biases,
                      float* derivative_Loss_Activations_previousLayer,
                      const int batchSize,
                      const float* activations_currentLayer) {
    int batchSize_dim = batchSize;
    int outputSize_dim = layer.outputSize;
    int inputSize_dim = layer.inputSize;

    // Compute gradients
    BackpropagationKernel<<<(batchSize_dim * outputSize_dim * inputSize_dim + 255) / 256, 256>>>(derivative_Loss_Z_nextLayer, activations_previousLayer, layer.weights, derivative_Loss_Weights, derivative_Loss_Biases, derivative_Loss_Activations_previousLayer, activations_currentLayer, batchSize_dim, outputSize_dim, inputSize_dim);
    CHECK_LAST_CUDA_ERROR();
}

void UpdateWeightsAndBiases(DenseLayer& layer, float* derivative_Loss_Weights, float* derivative_Loss_Biases, const float learningRate, const float threshold) {
    // Add the gradient clipping
    // Perform Gradient Descent optimization
    int numBlocks = (layer.inputSize * layer.outputSize + 255) / 256;
    GradientDescentUpdateKernel<<<numBlocks, 256>>>(layer.weights, derivative_Loss_Weights, layer.inputSize * layer.outputSize, learningRate);
    CHECK_LAST_CUDA_ERROR();

    numBlocks = (layer.outputSize + 255) / 256;
    GradientDescentUpdateKernel<<<numBlocks, 256>>>(layer.biases, derivative_Loss_Biases, layer.outputSize, learningRate);
    CHECK_LAST_CUDA_ERROR();
}

__global__ void ComputeDzLastLayer(const float* log_predictions, const int* labels, float* dZ, const int numClasses, const int batchSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= batchSize)
        return;

    for (int i = 0; i < numClasses; i++) {
        float softmax_output = expf(log_predictions[idx * numClasses + i]); // convert log_softmax to softmax
        dZ[idx * numClasses + i] = softmax_output - (i == labels[idx] ? 1.0f : 0.0f);
    }
}

__global__ void CorrectPredictionKernel(int* predictions, int* labels, int* d_correctCount, int batchSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < batchSize) {
        if (predictions[i] == labels[i]) {
            atomicAdd(d_correctCount, 1);
        }
    }
}

float ComputeAccuracy(int* predictions, int* labels, int batchSize) {
    int* d_correctCount;
    cudaMalloc((void**)&d_correctCount, sizeof(int));
    cudaMemset(d_correctCount, 0, sizeof(int));

    CorrectPredictionKernel<<<(batchSize + 255) / 256, 256>>>(predictions, labels, d_correctCount, batchSize);

    int correctCount;
    cudaMemcpy(&correctCount, d_correctCount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_correctCount);

    return (float)correctCount / batchSize;
}

__global__ void ArgMaxKernel(float* predictions, int* outputLabels, int numClasses, int batchSize) {
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