#include "activation.cuh"
#include "error_checking.cuh"
#include "linear_layer.cuh"
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

//__global__ void addBias(float* output, float* biases, int rows, int cols) {
__global__ void addBias(const float* biases, const int rows, const int cols, float* output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = idx / cols;
    const int col = idx % cols;

    if (row < rows && col < cols) {
        output[idx] += biases[col];
    }
}

LinearLayer::LinearLayer(uint32_t input_size, uint32_t output_size) : _h_input_size(input_size),
                                                                      _h_output_size(output_size) {
    cudaMalloc(&_d_weights, sizeof(float) * input_size * output_size);
    cudaMalloc(&_d_bias, sizeof(float) * output_size);
    cudaMalloc(&_d_output, sizeof(float) * output_size);
}

LinearLayer::~LinearLayer() {
    cudaFree(_d_weights);
    cudaFree(_d_bias);
    cudaFree(_d_output);
}

void LinearLayer::initWeightsAndBias() {
    // Allocate memory on the host
    std::vector<float> weights(_h_input_size * _h_output_size);
    std::vector<float> biases(_h_output_size);

    // Initialize weights and biases on the host
    float stddev = sqrtf(2.0f / _h_input_size); // He initialization
    for (int i = 0; i < _h_input_size * _h_output_size; i++) {
        weights[i] = stddev * (rand() / (RAND_MAX + 1.0f));
    }

    for (int i = 0; i < _h_output_size; i++) {
        biases[i] = 0.0f;
    }

    // Copy initialized weights and biases to the device
    CHECK_CUDA_ERROR(cudaMemcpy(_d_weights, weights.data(), _h_input_size * _h_output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(_d_bias, biases.data(), _h_output_size * sizeof(float), cudaMemcpyHostToDevice));
}

void LinearLayer::Forward(const float* d_input) {
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
                _h_output_size, batchSize, _h_input_size,
                &alpha,
                _d_weights, _h_input_size,
                d_input, _h_input_size,
                &beta,
                _d_output, _h_output_size);

    CHECK_LAST_CUDA_ERROR();

    addBias<<<(batchSize * _h_output_size + 255) / 256, 256>>>(_d_bias, batchSize, _h_output_size, _d_output);

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