#pragma once

#include "error_checking.cuh"
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// This will be later reused. We know leave this as is

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