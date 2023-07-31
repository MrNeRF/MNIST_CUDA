#include "error_checking.cuh"
#include "linear_layer.cuh"
#include <cstdlib>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <vector>

__global__ void AddBiasKernel(const float* __restrict__ biases,
                              const int rows,
                              const int cols,
                              float* __restrict__ output);

__global__ void BackpropagationKernel(const float* __restrict__ dZ_next,
                                      const float* __restrict__ A_prev,
                                      const float* __restrict__ W,
                                      float* __restrict__ dW,
                                      float* __restrict__ db,
                                      float* __restrict__ dZ,
                                      const int M,
                                      const int N,
                                      const int K);

__global__ void GradientDescentUpdateWeightsBiasesKernel(float* __restrict__ weights,
                                                         const float* __restrict__ weight_gradients,
                                                         const int weight_size,
                                                         float* __restrict__ biases,
                                                         const float* __restrict__ bias_gradients,
                                                         const int bias_size,
                                                         const float learning_rate);

void initWeightsAndBias(float* d_weights, float* d_bias, int input_size, int output_size);

LinearLayer::LinearLayer(int batch_size, int input_size, int output_size) : _h_batch_size(batch_size),
                                                                            _h_input_size(input_size),
                                                                            _h_output_size(output_size) {
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_weights, sizeof(float) * _h_input_size * _h_output_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_bias, sizeof(float) * _h_output_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_output, sizeof(float) * _h_output_size * _h_batch_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_dZ, sizeof(float) * _h_input_size * _h_batch_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_dW, sizeof(float) * _h_input_size * _h_output_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_dB, sizeof(float) * _h_output_size));
    initWeightsAndBias(_d_weights, _d_bias, _h_input_size, _h_output_size);
}

LinearLayer::~LinearLayer() {
    CHECK_CUDA_ERROR(cudaFree(_d_weights));
    CHECK_CUDA_ERROR(cudaFree(_d_bias));
    CHECK_CUDA_ERROR(cudaFree(_d_output));
    CHECK_CUDA_ERROR(cudaFree(_d_dZ));
    CHECK_CUDA_ERROR(cudaFree(_d_dW));
    CHECK_CUDA_ERROR(cudaFree(_d_dB));
}

const float* LinearLayer::Forward(const float* d_input, std::unique_ptr<Activation> activation) {
    CHECK_CUDA_ERROR(cudaMemset(_d_dW, 0, sizeof(float) * _h_input_size * _h_output_size));
    CHECK_CUDA_ERROR(cudaMemset(_d_dB, 0, sizeof(float) * _h_output_size));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    using ColumMajor = cutlass::layout::ColumnMajor;
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor,
                                                    float, ColumMajor,
                                                    float, ColumMajor>;

    CutlassGemm::Arguments args({_h_output_size, _h_batch_size, _h_input_size},
                                {_d_weights, _h_input_size},
                                {d_input, _h_input_size},
                                {_d_output, _h_output_size},
                                {_d_output, _h_output_size},
                                {alpha, beta});

    CutlassGemm gemm_op;

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to perform cutlass gemm" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    AddBiasKernel<<<(_h_batch_size * _h_output_size + 255) / 256, 256>>>(_d_bias,
                                                                         _h_batch_size,
                                                                         _h_output_size,
                                                                         _d_output);
    CHECK_LAST_CUDA_ERROR();

    if (activation != nullptr) {
        (*activation)(_h_batch_size, _h_output_size, _d_output);
    }

    return _d_output;
}

const float* LinearLayer::Backward(const float* d_dZ, const float* d_activation_prev_layer) {
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (_h_batch_size * _h_output_size * _h_input_size + threadsPerBlock - 1) / threadsPerBlock;
    // TODO: d_dZ comes from ouside. I think it should be inside.
    // _d_dz is here allocated but should be outside. This is the output. Not ideal but ok for now.

    BackpropagationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_dZ,
                                                              d_activation_prev_layer,
                                                              _d_weights,
                                                              _d_dW,
                                                              _d_dB,
                                                              _d_dZ,
                                                              _h_batch_size,
                                                              _h_output_size,
                                                              _h_input_size);
    CHECK_LAST_CUDA_ERROR();
    return _d_dZ;
}

void LinearLayer::Update(float learning_rate, cudaStream_t stream) {
    const int threadsPerBlock = 256;
    const int blocksPerGridWeights = (_h_output_size * _h_input_size + threadsPerBlock - 1) / threadsPerBlock;
    GradientDescentUpdateWeightsBiasesKernel<<<blocksPerGridWeights, threadsPerBlock, 0, stream>>>(_d_weights,
                                                                                                   _d_dW,
                                                                                                   _h_input_size * _h_output_size,
                                                                                                   _d_bias,
                                                                                                   _d_dB,
                                                                                                   _h_output_size,
                                                                                                   learning_rate);
    CHECK_LAST_CUDA_ERROR();
}

std::vector<float> LinearLayer::GetWeightsCPU() const {
    std::vector<float> weights(_h_input_size * _h_output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(weights.data(), _d_weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return weights;
}

std::vector<float> LinearLayer::GetBiasCPU() const {
    std::vector<float> bias(_h_output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(bias.data(), _d_bias, bias.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return bias;
}

std::vector<float> LinearLayer::GetOutputCPU() const {
    std::vector<float> output(_h_output_size * _h_batch_size);
    CHECK_CUDA_ERROR(cudaMemcpy(output.data(), _d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
}

void initWeightsAndBias(float* d_weights, float* d_bias, int input_size, int output_size) {
    // Allocate memory on the host
    std::vector<float> weights(input_size * output_size);
    std::vector<float> biases(output_size);

    // Initialize weights and biases on the host
    float stddev = sqrtf(2.0f / input_size); // He initialization
    for (int i = 0; i < input_size * output_size; i++) {
        weights[i] = stddev * (rand() / (RAND_MAX + 1.0f));
    }

    for (int i = 0; i < output_size; i++) {
        biases[i] = 0.0f;
    }

    // Copy initialized weights and biases to the device
    CHECK_CUDA_ERROR(cudaMemcpy(d_weights, weights.data(), input_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_bias, biases.data(), output_size * sizeof(float), cudaMemcpyHostToDevice));
}

void LinearLayer::SetWeightsFromCPU(const float* weights) {
    CHECK_CUDA_ERROR(cudaMemcpy(_d_weights, weights, _h_output_size * _h_input_size * sizeof(float), cudaMemcpyHostToDevice));
}

void LinearLayer::SetBiasFromCPU(const float* bias) {
    CHECK_CUDA_ERROR(cudaMemcpy(_d_bias, bias, _h_output_size * sizeof(float), cudaMemcpyHostToDevice));
}

std::vector<float> LinearLayer::GetWeightGradientsCPU() const {
    std::vector<float> dW(_h_input_size * _h_output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(dW.data(), _d_dW, dW.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return dW;
}

std::vector<float> LinearLayer::GetBiasGradientsCPU() const {
    std::vector<float> dB(_h_output_size);
    CHECK_CUDA_ERROR(cudaMemcpy(dB.data(), _d_dB, dB.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return dB;
}

__global__ void AddBiasKernel(const float* __restrict__ biases,
                              const int rows,
                              const int cols,
                              float* __restrict__ output) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = idx / cols;
    const int col = idx % cols;

    if (row < rows && col < cols) {
        output[idx] += biases[col];
    }
}

// M = batchSize, N = outputSize, K = inputSize
__global__ void BackpropagationKernel(const float* __restrict__ dZ_next,
                                      const float* __restrict__ A_prev,
                                      const float* __restrict__ W,
                                      float* __restrict__ dW,
                                      float* __restrict__ db,
                                      float* __restrict__ dZ,
                                      const int M,
                                      const int N,
                                      const int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate weights derivative
    if (idx < N * K) {
        const int n = idx / K;
        const int k = idx % K;

        float sum = 0.0f;

#pragma unroll
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
#pragma unroll
        for (int m = 0; m < M; ++m) {
            sum += dZ_next[m * N + idx];
        }
        db[idx] = sum / M;
    }

    if (idx < K) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (int n = 0; n < N; ++n) {
                sum += dZ_next[m * N + n] * W[n * K + idx];
            }

            // Apply the derivative of ReLU activation function
            sum = A_prev[m * K + idx] > 0 ? sum : 0.f;

            dZ[m * K + idx] = sum;
        }
    }
}

__global__ void GradientDescentUpdateWeightsBiasesKernel(float* __restrict__ weights,
                                                         const float* __restrict__ weight_gradients,
                                                         const int weight_size,
                                                         float* __restrict__ biases,
                                                         const float* __restrict__ bias_gradients,
                                                         const int bias_size,
                                                         const float learning_rate) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < weight_size) {
        weights[idx] -= learning_rate * weight_gradients[idx];
    }
    if (idx < bias_size) {
        biases[idx] -= learning_rate * bias_gradients[idx];
    }
}