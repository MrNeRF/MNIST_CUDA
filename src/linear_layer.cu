#include "error_checking.cuh"
#include "linear_layer.cuh"
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void AddBiasKernel(const float* biases,
                              const int rows,
                              const int cols,
                              float* output);

void initWeightsAndBias(float* d_weights, float* d_bias);

LinearLayer::LinearLayer(int batch_size, int input_size, int output_size) : _h_batch_size(batch_size),
                                                                            _h_input_size(input_size),
                                                                            _h_output_size(output_size) {
    cudaMalloc(&_d_weights, sizeof(float) * input_size * output_size);
    cudaMalloc(&_d_bias, sizeof(float) * output_size);
    cudaMalloc(&_d_output, sizeof(float) * output_size);
    initWeightsAndBias();
}

LinearLayer::~LinearLayer() {
    cudaFree(_d_weights);
    cudaFree(_d_bias);
    cudaFree(_d_output);
}

void LinearLayer::Forward(const float* d_input, std::unique_ptr<Activation> activation) {
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
    // Destroy the handle
    cublasDestroy(handle);
    CHECK_LAST_CUDA_ERROR();

    AddBiasKernel<<<(batchSize * _h_output_size + 255) / 256, 256>>>(_d_bias,
                                                                     batchSize,
                                                                     _h_output_size,
                                                                     _d_output);

    CHECK_LAST_CUDA_ERROR();

    if (activation != nullptr) {
        activation->Forward(_d_output);
    }
}

std::vector<float> LinearLayer::GetWeightsCPU() const {
    std::vector<float> weights(_h_input_size * _h_output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(weights.data(), _d_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return weights;
}
std::vector<float> LineraLayer::GetBiasCPU() const {
    std::vector<float> bias(_h_output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data(), _d_bias, bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return bias;
}

std::vector<float> GetOutputCPU() const {
    std::vector<float> output(_h_output_size * batchSize);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), _d_output, output.size() sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

void initWeightsAndBias(float* d_weights, float* d_bias) {
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

__global__ void AddBiasKernel(const float* biases,
                              const int rows,
                              const int cols,
                              float* output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = idx / cols;
    const int col = idx % cols;

    if (row < rows && col < cols) {
        output[idx] += biases[col];
    }
}
