#pragma once

#include "error_checking.cuh"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// This will be later reused. We know leave this as is
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

void InitializeLayer_OLD(DenseLayer& layer) {
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