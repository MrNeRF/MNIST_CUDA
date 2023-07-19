#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "mlp.cuh"
#include "stb_image_write.h"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <vector>

// CPU reference implementation
void ComputeDzLastLayerCPU(const std::vector<float>& predictions, const std::vector<int>& labels, std::vector<float>& dZ, int numClasses, int batchSize) {
    for (int idx = 0; idx < batchSize; ++idx) {
        for (int i = 0; i < numClasses; ++i) {
            ASSERT_TRUE(idx * numClasses + i < dZ.size());
            ASSERT_TRUE(idx * numClasses + i < predictions.size());
            ASSERT_TRUE(idx < labels.size());
            float softmax_output = std::exp(predictions[idx * numClasses + i]); // convert log_softmax to softmax
            dZ[idx * numClasses + i] = softmax_output - (i == labels[idx] ? 1.0f : 0.0f);
        }
    }
}

TEST(ComputeDzLastLayerTest, CompareCPUAndGPUResults) {
    // Initialize test inputs
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    std::uniform_int_distribution<int> int_dis(0, 9);

    int numClasses = 10;
    int batchSize = 32;

    std::vector<float> predictions(batchSize * numClasses, 0.0);
    std::vector<int> labels(batchSize, 0);

    // Generate random predictions and labels
    for (int i = 0; i < batchSize; ++i) {
        float sum = 0;
        for (int j = 0; j < numClasses; ++j) {
            float rand_val = dis(gen);
            predictions[i * numClasses + j] = rand_val;
            sum += rand_val;
        }

        // Ensure that each set of predictions adds up to 1.0
        for (int j = 0; j < numClasses; ++j) {
            predictions[i * numClasses + j] /= sum;
        }

        // Generate random labels
        labels[i] = int_dis(gen);
    }

    // Run CPU version
    std::vector<float> dZ_cpu(batchSize * numClasses);
    ComputeDzLastLayerCPU(predictions, labels, dZ_cpu, numClasses, batchSize);

    // Prepare device memory
    float* d_predictions;
    int* d_labels;
    float* d_dZ;
    cudaMalloc(&d_predictions, predictions.size() * sizeof(float));
    cudaMalloc(&d_labels, labels.size() * sizeof(int));
    cudaMalloc(&d_dZ, dZ_cpu.size() * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_predictions, predictions.data(), predictions.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Run GPU version
    ComputeDzLastLayer<<<(batchSize + 255) / 256, 256>>>(d_predictions, d_labels, d_dZ, numClasses, batchSize);

    // Copy output data to host
    std::vector<float> dZ_gpu(dZ_cpu.size());
    cudaMemcpy(dZ_gpu.data(), d_dZ, dZ_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU outputs
    for (int i = 0; i < dZ_cpu.size(); ++i) {
        ASSERT_NEAR(dZ_cpu[i], dZ_gpu[i], 1e-6);
    }

    // Free device memory
    cudaFree(d_predictions);
    cudaFree(d_labels);
    cudaFree(d_dZ);
}

// CPU reference implementation
void LogSoftmaxBatchCPU(const std::vector<float>& input, std::vector<float>& output, const int outputSize, const int batchSize) {
    for (int i = 0; i < batchSize; ++i) {
        float maxInput = *std::max_element(input.begin() + i * outputSize, input.begin() + (i + 1) * outputSize);
        ASSERT_TRUE(std::isnan(maxInput) == false);
        float sum = 0.0f;
        for (int j = 0; j < outputSize; ++j) {
            ASSERT_TRUE(i * outputSize + j < input.size());
            sum += std::exp(input[i * outputSize + j] - maxInput);
        }
        for (int j = 0; j < outputSize; ++j) {
            ASSERT_TRUE(i * outputSize + j < output.size());
            ASSERT_TRUE(i * outputSize + j < input.size());
            output[i * outputSize + j] = std::log(std::exp(input[i * outputSize + j] - maxInput) / sum + 1e-8);
        }
    }
}

TEST(SoftmaxBatchTest, BasicTest) {
    const int outputSize = 10;
    const int batchSize = 32;
    std::vector<float> input(outputSize * batchSize);
    std::vector<float> output_cpu(outputSize * batchSize);
    std::vector<float> output_gpu(outputSize * batchSize);

    // Initialize input with some random values
    std::generate(input.begin(), input.end(), []() { return rand() / (float)RAND_MAX; });

    // Run CPU version
    LogSoftmaxBatchCPU(input, output_cpu, outputSize, batchSize);

    // Prepare device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output, outputSize * batchSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU version
    LogSoftmaxBatch<<<(batchSize + 255) / 256, 256>>>(d_input, d_output, outputSize, batchSize);

    // Copy output data to host
    cudaMemcpy(output_gpu.data(), d_output, output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU outputs
    for (int i = 0; i < batchSize; ++i) {
        float cpu_sum = 0.0f;
        float gpu_sum = 0.0f;
        for (int j = 0; j < outputSize; ++j) {
            cpu_sum += output_cpu[i * outputSize + j];
            gpu_sum += output_gpu[i * outputSize + j];
        }
        EXPECT_NEAR(cpu_sum, gpu_sum, 1e-4);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

float CrossEntropyLossCPU(const std::vector<float>& predictions, const std::vector<int>& labels, const int numClasses, const int batchSize) {
    float loss = 0.0f;
    for (int i = 0; i < batchSize; ++i) {
        auto index = static_cast<size_t>(i * numClasses + labels[i]);
        bool isTrue = index < predictions.size();
        if (!isTrue) {
            std::cout << "index: " << index << " predictions.size(): " << predictions.size() << std::endl;
        }
        // because predictions are already in log domain, we do not need to apply logarithm again
        loss -= predictions[index];
    }
    return loss / batchSize;
}

// Test
TEST(CrossEntropyLossTest, BasicTest) {
    const int numClasses = 10;
    const int batchSize = 32;
    std::vector<float> inputs(numClasses * batchSize);
    std::vector<float> predictions(numClasses * batchSize);
    std::vector<int> labels(batchSize);

    // Initialize inputs with some random values
    std::generate(inputs.begin(), inputs.end(), []() { return rand() / (float)RAND_MAX; });

    // Initialize labels with some random values
    std::generate(labels.begin(), labels.end(), [numClasses]() { return rand() % numClasses; });

    // Compute softmax on CPU
    LogSoftmaxBatchCPU(inputs, predictions, numClasses, batchSize);

    // Run CPU version of CrossEntropyLoss
    float loss_cpu = CrossEntropyLossCPU(predictions, labels, numClasses, batchSize);

    // Prepare device memory
    float* d_inputs;
    float* d_predictions;
    int* d_labels;
    float* d_loss;
    cudaMalloc(&d_inputs, inputs.size() * sizeof(float));
    cudaMalloc(&d_predictions, predictions.size() * sizeof(float));
    cudaMalloc(&d_labels, labels.size() * sizeof(int));
    cudaMalloc(&d_loss, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_inputs, inputs.data(), inputs.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Compute softmax on GPU
    LogSoftmaxBatch<<<(batchSize + 255) / 256, 256>>>(d_inputs, d_predictions, numClasses, batchSize);

    // Initialize loss to zero
    float initial_loss = 0.0f;
    cudaMemcpy(d_loss, &initial_loss, sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU version of CrossEntropyLoss
    CrossEntropyLoss<<<(batchSize + 255) / 256, 256>>>(d_predictions, d_labels, d_loss, numClasses, batchSize);

    // Copy output data to host
    float loss_gpu;
    cudaMemcpy(&loss_gpu, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU outputs
    // CrossEntropyLoss must be divided by batch size
    EXPECT_NEAR(loss_cpu, loss_gpu / batchSize, 1e-4);

    // Free device memory
    cudaFree(d_inputs);
    cudaFree(d_predictions);
    cudaFree(d_labels);
    cudaFree(d_loss);
}

void ForwardPropagationCPU(const std::vector<float>& weights, const std::vector<float>& biases, const std::vector<float>& input, std::vector<float>& output, const bool relu, int batchSize) {
    int N = biases.size();      // output size
    int K = weights.size() / N; // input size
    std::cout << "N: " << N << " K: " << K << " batchSize: " << batchSize << "output.size: " << output.size() << "image.size " << input.size() << std::endl;
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                ASSERT_TRUE(i * K + k < input.size());
                ASSERT_TRUE(j * K + k < weights.size());
                sum += input[i * K + k] * weights[j * K + k];
            }
            output[i * N + j] = sum + biases[j];
            if (relu) {
                ASSERT_TRUE(i * N + j < output.size());
                output[i * N + j] = std::max(0.0f, output[i * N + j]);
            }
        }
    }
}

void ForwardPropagationFunc(int batchsize) {
    int batchSize = 32;
    DenseLayer layer(28 * 28, 50);
    InitializeLayer(layer);

    const bool activateRelu = true;
    std::vector<float> input(batchSize * 28 * 28, 0.f);
    std::generate(input.begin(), input.end(), []() { return rand() / (float)RAND_MAX; });

    // Prepare device memory
    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output, layer.outputSize * batchSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU version
    ForwardPropagation(layer, d_input, d_output, activateRelu, batchSize);

    // Copy output data to host
    std::vector<float> output_gpu(layer.outputSize * batchSize);
    cudaMemcpy(output_gpu.data(), d_output, output_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy weights and biases from GPU to host
    std::vector<float> weights(layer.inputSize * layer.outputSize);
    std::vector<float> biases(layer.outputSize);
    cudaMemcpy(weights.data(), layer.weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases.data(), layer.biases, biases.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CPU version
    std::vector<float> output_cpu(layer.outputSize * batchSize);
    ForwardPropagationCPU(weights, biases, input, output_cpu, activateRelu, batchSize);

    // Compare CPU and GPU outputs
    for (int i = 0; i < output_cpu.size(); ++i) {
        EXPECT_NEAR(output_cpu[i], output_gpu[i], 1e-4);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

TEST(ForwardPropagationTest_Batch32, BasicTest) {
    ForwardPropagationFunc(32);
}

TEST(ForwardPropagationTest_Batch1, BasicTest) {
    ForwardPropagationFunc(1);
}

TEST(ForwardPropagationTest_Batch64, BasicTest) {
    ForwardPropagationFunc(64);
}

TEST(ComputeDzLastLayerAfterForwardPassTest, BasicTest) {
    int batchSize = 32;
    DenseLayer layer1(28 * 28, 50);
    DenseLayer layer2(50, 50);
    DenseLayer layer3(50, 10);
    InitializeLayer(layer1);
    InitializeLayer(layer2);
    InitializeLayer(layer3);

    std::vector<float> input(batchSize * 28 * 28);
    std::generate(input.begin(), input.end(), []() { return rand() / (float)RAND_MAX; });

    std::vector<int> labels(batchSize);
    std::generate(labels.begin(), labels.end(), []() { return rand() % 10; }); // assuming 10 classes

    // Prepare device memory
    float* d_input;
    float* d_output1;
    float* d_output2;
    float* d_output3;
    int* d_labels;
    float* d_loss;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output1, layer1.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_output2, layer2.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_output3, layer3.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_labels, labels.size() * sizeof(int));
    cudaMalloc(&d_loss, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Run GPU version
    ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
    ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
    ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);

    LogSoftmaxBatch<<<(batchSize + 255) / 256, 256>>>(d_output3, d_output3, layer3.outputSize, batchSize);
    CrossEntropyLoss<<<(batchSize + 255) / 256, 256>>>(d_output3, d_labels, d_loss, layer3.outputSize, batchSize);

    // Copy output data to host
    float loss_gpu;
    cudaMemcpy(&loss_gpu, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Copy weights and biases from GPU to host
    std::vector<float> weights1(layer1.inputSize * layer1.outputSize);
    std::vector<float> biases1(layer1.outputSize);
    cudaMemcpy(weights1.data(), layer1.weights, weights1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases1.data(), layer1.biases, biases1.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> weights2(layer2.inputSize * layer2.outputSize);
    std::vector<float> biases2(layer2.outputSize);
    cudaMemcpy(weights2.data(), layer2.weights, weights2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases2.data(), layer2.biases, biases2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> weights3(layer3.inputSize * layer3.outputSize);
    std::vector<float> biases3(layer3.outputSize);
    cudaMemcpy(weights3.data(), layer3.weights, weights3.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases3.data(), layer3.biases, biases3.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CPU version
    std::vector<float> output1_cpu(layer1.outputSize * batchSize);
    std::vector<float> output2_cpu(layer2.outputSize * batchSize);
    std::vector<float> output3_cpu(layer3.outputSize * batchSize);
    std::vector<float> softmax_output_cpu(layer3.outputSize * batchSize);

    ForwardPropagationCPU(weights1, biases1, input, output1_cpu, true, batchSize);
    ForwardPropagationCPU(weights2, biases2, output1_cpu, output2_cpu, true, batchSize);
    ForwardPropagationCPU(weights3, biases3, output2_cpu, output3_cpu, false, batchSize);
    LogSoftmaxBatchCPU(output3_cpu, softmax_output_cpu, layer3.outputSize, batchSize);
    float loss = CrossEntropyLossCPU(softmax_output_cpu, labels, layer3.outputSize, batchSize);
    ASSERT_TRUE(std::abs(loss - loss_gpu / batchSize) < 1e-4 && loss > 0.0f);
    // Free device memory

    // Compare CPU and GPU outputs
    std::vector<float> output3_gpu(layer3.outputSize * batchSize);
    cudaMemcpy(output3_gpu.data(), d_output3, output3_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // Initialize test inputs

    // Run CPU version
    std::vector<float> dZ_cpu(layer3.outputSize * batchSize);
    ComputeDzLastLayerCPU(softmax_output_cpu, labels, dZ_cpu, layer3.outputSize, batchSize);

    // Prepare device memory
    // Run GPU version
    float* d_dZ;
    cudaMalloc(&d_dZ, dZ_cpu.size() * sizeof(float));
    ComputeDzLastLayer<<<(batchSize + 255) / 256, 256>>>(d_output3, d_labels, d_dZ, layer3.outputSize, batchSize);

    // Copy output data to host
    std::vector<float> dZ_gpu(dZ_cpu.size());
    cudaMemcpy(dZ_gpu.data(), d_dZ, dZ_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare CPU and GPU outputs
    for (int i = 0; i < dZ_cpu.size(); ++i) {
        EXPECT_NEAR(dZ_cpu[i], dZ_gpu[i], 1e-2);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_dZ);
}

void BackpropagationKernelCPU(const std::vector<float>& dZ_next, const std::vector<float>& A_prev, const std::vector<float>& W, std::vector<float>& dW, std::vector<float>& db, std::vector<float>& dZ, int M, int N, int K) {
    // Calculate dW
    std::cout << "K: " << K << " M: " << M << " N: " << N << " A_prev.size() " << A_prev.size() << " dZ_next.size() " << dZ_next.size() << " dW.size() " << dW.size() << "\n"
              << std::endl;
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float sum = 0.0f;
            for (int m = 0; m < M; ++m) {
                ASSERT_TRUE(m * N + n < dZ_next.size());
                ASSERT_TRUE(m * K + k < A_prev.size());
                sum += dZ_next[m * N + n] * A_prev[m * K + k];
            }
            ASSERT_TRUE(n * K + k < dW.size());
            dW[n * K + k] = sum / M;
        }
    }

    // Calculate db
    std::cout << "K: " << K << " M: " << M << " N: " << N << " dZ_next.size() " << dZ_next.size() << " db.size() " << db.size() << "\n"
              << std::endl;
    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int m = 0; m < M; ++m) {
            ASSERT_TRUE(m * N + n < dZ_next.size());
            sum += dZ_next[m * N + n];
        }
        ASSERT_TRUE(sum / M < 1e+10);
        ASSERT_TRUE(sum / M > -1e+10);
        ASSERT_TRUE(std::isfinite(sum / M));
        db[n] = sum / M;
    }

    // Calculate dZ
    if (dZ.size() > 0) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < K; ++k) {
                float sum = 0.0f;
                for (int n = 0; n < N; ++n) {
                    sum += dZ_next[m * N + n] * W[n * K + k];
                }
                dZ[m * K + k] = sum;
            }
        }
    }
}

TEST(BackpropagationTest, BasicTest) {
    int batchSize = 32;
    DenseLayer layer1(28 * 28, 50);
    DenseLayer layer2(50, 50);
    DenseLayer layer3(50, 10);
    InitializeLayer(layer1);
    InitializeLayer(layer2);
    InitializeLayer(layer3);

    std::vector<float> input(batchSize * 28 * 28);
    std::generate(input.begin(), input.end(), []() { return rand() / (float)RAND_MAX; });

    std::vector<int> labels(batchSize);
    std::generate(labels.begin(), labels.end(), []() { return rand() % 10; }); // assuming 10 classes

    // Prepare device memory
    float* d_input;
    float* d_output1;
    float* d_output2;
    float* d_output3;
    int* d_labels;
    float* d_loss;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output1, layer1.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_output2, layer2.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_output3, layer3.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_labels, labels.size() * sizeof(int));
    cudaMalloc(&d_loss, sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Run GPU version
    ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
    ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
    ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);

    LogSoftmaxBatch<<<(batchSize + 255) / 256, 256>>>(d_output3, d_output3, layer3.outputSize, batchSize);
    CrossEntropyLoss<<<(batchSize + 255) / 256, 256>>>(d_output3, d_labels, d_loss, layer3.outputSize, batchSize);

    // Copy output data to host
    float loss_gpu;
    cudaMemcpy(&loss_gpu, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Copy weights and biases from GPU to host
    std::vector<float> weights1(layer1.inputSize * layer1.outputSize);
    std::vector<float> biases1(layer1.outputSize);
    cudaMemcpy(weights1.data(), layer1.weights, weights1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases1.data(), layer1.biases, biases1.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> weights2(layer2.inputSize * layer2.outputSize);
    std::vector<float> biases2(layer2.outputSize);
    cudaMemcpy(weights2.data(), layer2.weights, weights2.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases2.data(), layer2.biases, biases2.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> weights3(layer3.inputSize * layer3.outputSize);
    std::vector<float> biases3(layer3.outputSize);
    cudaMemcpy(weights3.data(), layer3.weights, weights3.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases3.data(), layer3.biases, biases3.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CPU version
    std::vector<float> output1_cpu(layer1.outputSize * batchSize);
    std::vector<float> output2_cpu(layer2.outputSize * batchSize);
    std::vector<float> output3_cpu(layer3.outputSize * batchSize);
    std::vector<float> softmax_output_cpu(layer3.outputSize * batchSize);

    ForwardPropagationCPU(weights1, biases1, input, output1_cpu, true, batchSize);
    ForwardPropagationCPU(weights2, biases2, output1_cpu, output2_cpu, true, batchSize);
    ForwardPropagationCPU(weights3, biases3, output2_cpu, output3_cpu, false, batchSize);
    LogSoftmaxBatchCPU(output3_cpu, softmax_output_cpu, layer3.outputSize, batchSize);
    float loss = CrossEntropyLossCPU(softmax_output_cpu, labels, layer3.outputSize, batchSize);
    EXPECT_NEAR(loss, loss_gpu / batchSize, 1e-3);

    // Compare CPU and GPU outputs
    std::vector<float> output3_gpu(layer3.outputSize * batchSize);
    cudaMemcpy(output3_gpu.data(), d_output3, output3_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // Initialize test inputs

    // Run CPU version
    std::vector<float> dZ_cpu(layer3.outputSize * batchSize);
    ComputeDzLastLayerCPU(softmax_output_cpu, labels, dZ_cpu, layer3.outputSize, batchSize);

    // Prepare device memory
    // Run GPU version
    float* d_dZ;
    cudaMalloc(&d_dZ, dZ_cpu.size() * sizeof(float));
    ComputeDzLastLayer<<<(batchSize + 255) / 256, 256>>>(d_output3, d_labels, d_dZ, layer3.outputSize, batchSize);

    std::vector<float> dZ_gpu(dZ_cpu.size());
    cudaMemcpy(dZ_gpu.data(), d_dZ, dZ_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dZ_cpu.size(); ++i) {
        EXPECT_NEAR(dZ_cpu[i], dZ_gpu[i], 1e-4);
    }

    float* d_dW;
    float* d_db;
    cudaMalloc(&d_dW, weights3.size() * sizeof(float));
    cudaMalloc(&d_db, biases3.size() * sizeof(float));

    ComputeGradients(layer3, d_dZ, d_output2, d_dW, d_db, nullptr, batchSize, nullptr);

    // // Copy output data to host
    std::vector<float> dW_gpu(weights3.size());
    std::vector<float> db_gpu(biases3.size());
    cudaMemcpy(dW_gpu.data(), d_dW, dW_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(db_gpu.data(), d_db, db_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // // Run CPU version
    std::vector<float> dW_cpu(weights3.size());
    std::vector<float> db_cpu(biases3.size());

    std::vector<float> dZ_next_cpu;
    BackpropagationKernelCPU(dZ_cpu, output2_cpu, weights3, dW_cpu, db_cpu, dZ_next_cpu, batchSize, layer3.outputSize, layer3.inputSize);

    for (int i = 0; i < dW_cpu.size(); ++i) {
        EXPECT_NEAR(dW_cpu[i], dW_gpu[i], 0.1);
    }

    for (int i = 0; i < db_cpu.size(); ++i) {
        EXPECT_NEAR(db_cpu[i], db_gpu[i], 0.1);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_dZ);
    cudaFree(d_dW);
    cudaFree(d_db);
}

TEST(ForwardMatmulTest, BasicTest) {
    int batchSize = 5;
    DenseLayer layer1(4, 3);

    // batchSize * layer1.inputSize
    std::vector<float> input{0.1f, 0.1f, 0.1f, 0.1f,
                             0.2f, 0.2f, 0.2f, 0.2f,
                             0.3f, 0.3f, 0.3f, 0.3f,
                             0.4f, 0.4f, 0.4f, 0.4f,
                             0.5f, 0.5f, 0.5f, 0.5f};
    {
        float h_weights[12] = {-1.f, 2.f, 3.f, 4.f,
                               5.f, -6.f, 7.f, 8.f,
                               9.f, 10.f, -11.f, 12.f};
        float h_biases[3] = {1.f, 2.f, 3.f};

        // Copy initialized weights and biases to the device
        CHECK_CUDA_ERROR(cudaMemcpy(layer1.weights, h_weights, layer1.inputSize * layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(layer1.biases, h_biases, layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Prepare device memory
    float* d_input;
    float* d_output1;
    // float* d_output2;
    // float* d_output3;
    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output1, batchSize * layer1.outputSize * sizeof(float));
    // cudaMalloc(&d_output2, layer2.outputSize * batchSize * sizeof(float));
    // cudaMalloc(&d_output3, layer3.outputSize * batchSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU version
    ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
    // ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
    // ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);

    std::vector<float> output1_gpu(layer1.outputSize * batchSize);
    cudaMemcpy(output1_gpu.data(), d_output1, output1_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < layer1.outputSize; ++j) {
            std::cout << output1_gpu[i * layer1.outputSize + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Copy weights and biases from GPU to host
    std::vector<float> weights(layer1.inputSize * layer1.outputSize);
    std::vector<float> biases(layer1.outputSize);
    cudaMemcpy(weights.data(), layer1.weights, weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases.data(), layer1.biases, biases.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CPU version
    std::vector<float> output1_cpu(layer1.outputSize * batchSize);
    ForwardPropagationCPU(weights, biases, input, output1_cpu, true, batchSize);

    for (int i = 0; i < output1_cpu.size(); ++i) {
        EXPECT_NEAR(output1_cpu[i], output1_gpu[i], 1e-4);
    }

    // Copy output data to host
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output1);
    // cudaFree(d_output2);
    // cudaFree(d_output3);
}

TEST(ForwardPassLossLibtorch, BasicTest) {
    const int batchSize = 5;
    const int inputDim1 = 4;
    const int outputDim1 = 3;
    const int inputDim2 = 3;
    const int outputDim2 = 3;
    const int inputDim3 = 3;
    const int outputDim3 = 2;
    const int numberOfClasses = 2;

    DenseLayer layer1(inputDim1, outputDim1);
    DenseLayer layer2(inputDim2, outputDim2);
    DenseLayer layer3(inputDim3, outputDim3);

    std::vector<int> labels{0, 1, 0, 1, 0};
    std::vector<float> input{0.1f, 0.1f, 0.1f, 0.1f,
                             0.2f, 0.2f, 0.2f, 0.2f,
                             0.3f, 0.3f, 0.3f, 0.3f,
                             0.4f, 0.4f, 0.4f, 0.4f,
                             0.5f, 0.5f, 0.5f, 0.5f};

    float h_weights1[12] = {-.1f, .2f, .2f, .2f,
                            .5f, -.6f, .7f, .8f,
                            .9f, .10f, -.11f, .12f};
    float h_biases1[3] = {.1f, .2f, .3f};

    float h_weights2[9] = {-.1f, .2f, .3f,
                           .4f, .5f, -.6f,
                           .7f, .8f, .9f};
    float h_biases2[3] = {.4f, .2f, .3f};

    float h_weights3[6] = {-.1f, .2f, .3f,
                           .4f, .5f, -.6f};
    float h_biases3[2] = {.1f, .2f};

    // Copy initialized weights and biases to the device
    CHECK_CUDA_ERROR(cudaMemcpy(layer1.weights, h_weights1, layer1.inputSize * layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(layer1.biases, h_biases1, layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(layer2.weights, h_weights2, layer2.inputSize * layer2.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(layer2.biases, h_biases2, layer2.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(layer3.weights, h_weights3, layer3.inputSize * layer3.outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(layer3.biases, h_biases3, layer3.outputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Prepare device memory
    float* d_input;
    float* d_output1;
    float* d_output2;
    float* d_output3;
    float* d_softmax_output;
    float* d_loss;
    int* d_labels;
    float *d_dZ2, *d_dZ3, *d_dZ1;
    float* derivative_Loss_Weights3;
    float* derivative_Loss_Biases3;
    float* derivative_Loss_Weights2;
    float* derivative_Loss_Biases2;
    float* derivative_Loss_Weights1;
    float* derivative_Loss_Biases1;

    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_output1, batchSize * layer1.outputSize * sizeof(float));
    cudaMalloc(&d_output2, layer2.outputSize * batchSize * sizeof(float));
    cudaMalloc(&d_output3, layer3.outputSize * batchSize * sizeof(float));
    cudaMalloc((void**)&d_softmax_output, numberOfClasses * batchSize * sizeof(float));
    cudaMalloc((void**)&d_loss, sizeof(float));
    cudaMalloc(&d_labels, labels.size() * sizeof(int));
    cudaMalloc((void**)&d_dZ3, numberOfClasses * batchSize * sizeof(float));
    cudaMalloc((void**)&d_dZ2, layer2.outputSize * batchSize * sizeof(float));
    cudaMalloc((void**)&d_dZ1, layer1.outputSize * batchSize * sizeof(float));
    cudaMalloc((void**)&derivative_Loss_Weights3, layer3.inputSize * layer3.outputSize * sizeof(float));
    cudaMalloc((void**)&derivative_Loss_Biases3, layer3.outputSize * sizeof(float));
    cudaMalloc((void**)&derivative_Loss_Weights2, layer2.inputSize * layer2.outputSize * sizeof(float));
    cudaMalloc((void**)&derivative_Loss_Biases2, layer2.outputSize * sizeof(float));
    cudaMalloc((void**)&derivative_Loss_Weights1, layer1.inputSize * layer1.outputSize * sizeof(float));
    cudaMalloc((void**)&derivative_Loss_Biases1, layer1.outputSize * sizeof(float));
    cudaMemset(derivative_Loss_Weights3, 0, layer3.inputSize * layer3.outputSize * sizeof(float));
    cudaMemset(derivative_Loss_Biases3, 0, layer3.outputSize * sizeof(float));
    cudaMemset(derivative_Loss_Weights2, 0, layer2.inputSize * layer2.outputSize * sizeof(float));
    cudaMemset(derivative_Loss_Biases2, 0, layer2.outputSize * sizeof(float));
    cudaMemset(derivative_Loss_Weights1, 0, layer1.inputSize * layer1.outputSize * sizeof(float));
    cudaMemset(derivative_Loss_Biases1, 0, layer1.outputSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Run GPU version
    ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
    ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
    ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);
    std::vector<float> output3_cpu(layer3.outputSize * batchSize);
    cudaMemcpy(output3_cpu.data(), d_output3, output3_cpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
    LogSoftmaxBatch<<<(batchSize + 255) / 50, 50>>>(d_output3, d_softmax_output, numberOfClasses, batchSize);
    float my_loss;
    CrossEntropyLoss<<<(batchSize + 255) / 256, 256>>>(d_softmax_output, d_labels, d_loss, numberOfClasses, batchSize);
    cudaMemcpy(&my_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    my_loss /= batchSize; // Only CUDA

    ComputeDzLastLayer<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_dZ3, numberOfClasses, batchSize);
    std::vector<float> d_dZ3_cpu(numberOfClasses * batchSize);
    cudaMemcpy(d_dZ3_cpu.data(), d_dZ3, d_dZ3_cpu.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Perform backpropagation
    ComputeGradients(layer3, d_dZ3, d_output2, derivative_Loss_Weights3, derivative_Loss_Biases3, d_dZ2, batchSize, d_output2);
    ComputeGradients(layer2, d_dZ2, d_output1, derivative_Loss_Weights2, derivative_Loss_Biases2, d_dZ1, batchSize, d_output2);
    ComputeGradients(layer1, d_dZ1, d_input, derivative_Loss_Weights1, derivative_Loss_Biases1, nullptr, batchSize, nullptr);

    UpdateWeightsAndBiases(layer3, derivative_Loss_Weights3, derivative_Loss_Biases3, 0.01 /*learning rate*/, 1.0 /*threshold*/);
    UpdateWeightsAndBiases(layer2, derivative_Loss_Weights2, derivative_Loss_Biases2, 0.01 /*learning rate*/, 1.0 /*threshold*/);
    UpdateWeightsAndBiases(layer1, derivative_Loss_Weights1, derivative_Loss_Biases1, 0.01 /*learning rate*/, 1.0 /*threshold*/);

    // libtorch implementation
    struct Net : torch::nn::Module {
        Net(int inputDim1, int outputDim1, int inputDim2, int outputDim2, int inputDim3, int outputDim3) {
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", torch::nn::Linear(inputDim1, outputDim1));
            fc2 = register_module("fc2", torch::nn::Linear(inputDim2, outputDim2));
            fc3 = register_module("fc3", torch::nn::Linear(inputDim3, outputDim3));
        }

        // Implement the Net's algorithm.
        torch::Tensor forward(torch::Tensor x) {
            // Use one of many tensor manipulation functions.
            x = torch::relu(fc1->forward(x));
            x = torch::relu(fc2->forward(x));
            x = fc3->forward(x);

            return x;
        }

        // Use one of many "standard library" modules.
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    };

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto inputTensor = torch::from_blob(input.data(), {5, 4}, options).requires_grad_(true);
    auto weightTensor1 = torch::from_blob(h_weights1, {3, 4}, options).requires_grad_(true);
    auto biasTensor1 = torch::from_blob(h_biases1, {3}, options).requires_grad_(true);
    auto weightTensor2 = torch::from_blob(h_weights2, {3, 3}, options).requires_grad_(true);
    auto biasTensor2 = torch::from_blob(h_biases2, {3}, options).requires_grad_(true);
    auto weightTensor3 = torch::from_blob(h_weights3, {2, 3}, options).requires_grad_(true);
    auto biasTensor3 = torch::from_blob(h_biases3, {2}, options).requires_grad_(true);

    auto net = std::make_shared<Net>(4, 3,
                                     3, 3,
                                     3, 2);

    net->fc1->weight = weightTensor1;
    net->fc1->bias = biasTensor1;
    net->fc2->weight = weightTensor2;
    net->fc2->bias = biasTensor2;
    net->fc3->weight = weightTensor3;
    net->fc3->bias = biasTensor3;

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
    optimizer.zero_grad();
    auto prediction = net->forward(inputTensor);

    // This step is necessary because torch is picky about the input type
    // and the labels are int64_t. If you insert std::vector<int>, the loss computation explodes.
    std::vector<int64_t> labels_torch_long(labels.begin(), labels.end());

    auto tensorLables = torch::from_blob(labels_torch_long.data(), {5}, torch::TensorOptions().dtype(torch::kLong));
    auto loss = torch::nn::functional::cross_entropy(prediction, tensorLables);
    loss.backward();
    optimizer.step();

    std::vector<float> h_l3_weight_grads(layer3.inputSize * layer3.outputSize);
    cudaMemcpy(h_l3_weight_grads.data(), derivative_Loss_Weights3, layer3.inputSize * layer3.outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> h_l3_bias_grads(layer3.outputSize);
    cudaMemcpy(h_l3_bias_grads.data(), derivative_Loss_Biases3, layer3.outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    bool isGradientWeightLayer3Defined = net->fc3->weight.grad().defined();
    ASSERT_TRUE(isGradientWeightLayer3Defined);
    if (isGradientWeightLayer3Defined) {
        for (int i = 0; i < layer3.inputSize * layer3.outputSize; ++i) {
            EXPECT_NEAR(net->fc3->weight.grad().view(-1)[i].item<float>(), h_l3_weight_grads[i], 1e-4);
        }
    }

    bool isGradientBiasLayer3Defined = net->fc3->bias.grad().defined();
    ASSERT_TRUE(isGradientBiasLayer3Defined);
    if (isGradientBiasLayer3Defined) {
        for (int i = 0; i < layer3.outputSize; ++i) {
            EXPECT_NEAR(net->fc3->bias.grad().view(-1)[i].item<float>(), h_l3_bias_grads[i], 1e-4);
        }
    }
    // Compare Updated Weights
    bool isUpdatedWeightsLayer3Defined = net->fc3->weight.defined();
    ASSERT_TRUE(isUpdatedWeightsLayer3Defined);
    if (isUpdatedWeightsLayer3Defined) {
        auto updated_weights = net->fc3->weight.detach().cpu();
        auto weights_data = updated_weights.view(-1).data_ptr<float>();
        std::vector<float> h_updated_weights(layer3.inputSize * layer3.outputSize);
        cudaMemcpy(h_updated_weights.data(), layer3.weights, h_updated_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < h_updated_weights.size(); ++i) {
            EXPECT_NEAR(weights_data[i], h_updated_weights[i], 1e-2);
        }
    }

    // Compare Updated Biases
    bool isUpdatedBiasesLayer3Defined = net->fc3->bias.defined();
    ASSERT_TRUE(isUpdatedBiasesLayer3Defined);
    if (isUpdatedBiasesLayer3Defined) {
        auto updated_biases = net->fc3->bias.detach().cpu();
        auto biases_data = updated_biases.view(-1).data_ptr<float>();
        std::vector<float> updated_biases_data(layer3.outputSize);
        cudaMemcpy(updated_biases_data.data(), layer3.biases, updated_biases_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < layer3.outputSize; ++i) {
            EXPECT_NEAR(biases_data[i], updated_biases_data[i], 1e-2);
        }
    }

    // std::cout << "Libtorch layer 2 weight gradients" << std::endl;
    // std::cout << net->fc2->weight.grad() << std::endl;
    // std::vector<float> h_l2_weight_grads(layer2.inputSize * layer2.outputSize);
    // cudaMemcpy(h_l2_weight_grads.data(), derivative_Loss_Weights2, layer2.inputSize * layer2.outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "My layer 2 weight gradients" << std::endl;

    // for (int i = 0; i < h_l2_weight_grads.size(); ++i) {
    //     std::cout << h_l2_weight_grads[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "Libtorch layer 1 weight gradients" << std::endl;
    // std::cout << net->fc1->weight.grad() << std::endl;
    // std::vector<float> h_l1_weight_grads(layer1.inputSize * layer1.outputSize);
    // cudaMemcpy(h_l1_weight_grads.data(), derivative_Loss_Weights1, layer1.inputSize * layer1.outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "My layer 1 weight gradients" << std::endl;
    // for (int i = 0; i < h_l1_weight_grads.size(); ++i) {
    //     std::cout << h_l1_weight_grads[i] << " ";
    // }
    // bool isGradientWeightLayer2Defined = net->fc2->weight.grad().defined();
    // ASSERT_TRUE(isGradientWeightLayer2Defined);
    // if (isGradientWeightLayer2Defined) {
    //     for (int i = 0; i < layer2.inputSize * layer2.outputSize; ++i) {
    //         EXPECT_NEAR(net->fc2->weight.grad().view(-1)[i].item<float>(), derivative_Loss_Weights2[i], 1e-4);
    //     }
    // }

    // bool isGradientBiasLayer2Defined = net->fc2->bias.grad().defined();
    // ASSERT_TRUE(isGradientBiasLayer2Defined);
    // if (isGradientBiasLayer2Defined) {
    //     for (int i = 0; i < layer2.outputSize; ++i) {
    //         EXPECT_NEAR(net->fc2->bias.grad().view(-1)[i].item<float>(), derivative_Loss_Biases2[i], 1e-4);
    //     }
    // }

    // bool isGradientWeightLayer1Defined = net->fc1->weight.grad().defined();
    // ASSERT_TRUE(isGradientWeightLayer1Defined);
    // if (isGradientWeightLayer1Defined) {
    //     for (int i = 0; i < layer1.inputSize * layer1.outputSize; ++i) {
    //         EXPECT_NEAR(net->fc1->weight.grad().view(-1)[i].item<float>(), derivative_Loss_Weights1[i], 1e-4);
    //     }
    // }

    // bool isGradientBiasLayer1Defined = net->fc1->bias.grad().defined();
    // ASSERT_TRUE(isGradientBiasLayer1Defined);
    // if (isGradientBiasLayer1Defined) {
    //     for (int i = 0; i < layer1.outputSize; ++i) {
    //         EXPECT_NEAR(net->fc1->bias.grad().view(-1)[i].item<float>(), derivative_Loss_Biases1[i], 1e-4);
    //     }
    // }
    // Compare Updated Weights
    bool isUpdatedWeightsLayer2Defined = net->fc2->weight.defined();
    ASSERT_TRUE(isUpdatedWeightsLayer2Defined);
    if (isUpdatedWeightsLayer2Defined) {
        auto updated_weights = net->fc2->weight.detach().cpu();
        auto weights_data = updated_weights.view(-1).data_ptr<float>();
        std::vector<float> h_updated_weights(layer2.inputSize * layer2.outputSize);
        cudaMemcpy(h_updated_weights.data(), layer2.weights, h_updated_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < h_updated_weights.size(); ++i) {
            EXPECT_NEAR(weights_data[i], h_updated_weights[i], 1e-2);
        }
    }

    // Compare Updated Biases
    bool isUpdatedBiasesLayer2Defined = net->fc2->bias.defined();
    ASSERT_TRUE(isUpdatedBiasesLayer2Defined);
    if (isUpdatedBiasesLayer2Defined) {
        auto updated_biases = net->fc2->bias.detach().cpu();
        auto biases_data = updated_biases.view(-1).data_ptr<float>();
        std::vector<float> updated_biases_data(layer2.outputSize);
        cudaMemcpy(updated_biases_data.data(), layer2.biases, updated_biases_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < layer2.outputSize; ++i) {
            EXPECT_NEAR(biases_data[i], updated_biases_data[i], 1e-2);
        }
    }

    bool isUpdatedWeightsLayer1Defined = net->fc1->weight.defined();
    ASSERT_TRUE(isUpdatedWeightsLayer1Defined);
    if (isUpdatedWeightsLayer1Defined) {
        auto updated_weights = net->fc1->weight.detach().cpu();
        auto weights_data = updated_weights.view(-1).data_ptr<float>();
        std::vector<float> h_updated_weights(layer1.inputSize * layer1.outputSize);
        cudaMemcpy(h_updated_weights.data(), layer1.weights, h_updated_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < h_updated_weights.size(); ++i) {
            EXPECT_NEAR(weights_data[i], h_updated_weights[i], 1e-2);
        }
    }
    bool isUpdatedBiasesLayer1Defined = net->fc1->bias.defined();
    ASSERT_TRUE(isUpdatedBiasesLayer1Defined);
    if (isUpdatedBiasesLayer1Defined) {
        auto updated_biases = net->fc1->bias.detach().cpu();
        auto biases_data = updated_biases.view(-1).data_ptr<float>();
        std::vector<float> updated_biases_data(layer1.outputSize);
        cudaMemcpy(updated_biases_data.data(), layer1.biases, updated_biases_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < layer1.outputSize; ++i) {
            EXPECT_NEAR(biases_data[i], updated_biases_data[i], 1e-2);
        }
    }
    // Copy output data to host
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_softmax_output);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_dZ3);
    cudaFree(d_dZ2);
    cudaFree(d_dZ1);
    cudaFree(derivative_Loss_Weights3);
    cudaFree(derivative_Loss_Biases3);
    cudaFree(derivative_Loss_Weights2);
    cudaFree(derivative_Loss_Biases2);
    cudaFree(derivative_Loss_Weights1);
    cudaFree(derivative_Loss_Biases1);
}

TEST(MNIST_Test, BasicTest) {
    struct Net : torch::nn::Module {
        Net(int inputDim1, int outputDim1, int inputDim2, int outputDim2, int inputDim3, int outputDim3) {
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", torch::nn::Linear(inputDim1, outputDim1));
            fc2 = register_module("fc2", torch::nn::Linear(inputDim2, outputDim2));
            fc3 = register_module("fc3", torch::nn::Linear(inputDim3, outputDim3));
        }

        // Implement the Net's algorithm.
        torch::Tensor forward(torch::Tensor x) {
            // Use one of many tensor manipulation functions.
            x = torch::relu(fc1->forward(x));
            x = torch::relu(fc2->forward(x));
            x = fc3->forward(x);

            return x;
        }

        // Use one of many "standard library" modules.
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    };

    auto net = std::make_shared<Net>(28 * 28, 50,
                                     50, 50,
                                     50, 10);

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
    int batchSize = 32;
    ///////////// HERE CUDA IMPL
    DenseLayer layer1(784, 50);
    DenseLayer layer2(50, 50);
    DenseLayer layer3(50, 10);

    // Allocate GPU memory
    float* d_input;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, 784 * batchSize * sizeof(float))); // input size of layer1
    float* d_output1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output1, 50 * batchSize * sizeof(float))); // output size of layer1 (input size of layer2)
    float* d_output2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output2, 50 * batchSize * sizeof(float))); // output size of layer2 (input size of layer3)
    float* d_output3;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output3, 10 * batchSize * sizeof(float))); // output size of layer3
    float* d_softmax_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_softmax_output, 10 * batchSize * sizeof(float)));
    float dropout_rate = 0.5; // Adjust this value as needed
    float* d_mask1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mask1, 50 * batchSize * sizeof(float))); // mask for layer1
    float* d_mask2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mask2, 50 * batchSize * sizeof(float))); // mask for layer2

    // Allocate memory for gradients
    float* derivative_Loss_Weights1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Weights1, layer1.inputSize * layer1.outputSize * sizeof(float)));
    float* derivative_Loss_Biases1;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Biases1, layer1.outputSize * sizeof(float)));

    float* derivative_Loss_Weights2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Weights2, layer2.inputSize * layer2.outputSize * sizeof(float)));
    float* derivative_Loss_Biases2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Biases2, layer2.outputSize * sizeof(float)));

    float* derivative_Loss_Weights3;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Weights3, layer3.inputSize * layer3.outputSize * sizeof(float)));
    float* derivative_Loss_Biases3;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Biases3, layer3.outputSize * sizeof(float)));

    int* d_labels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batchSize * sizeof(int))); // Allocate device memory for labels

    float loss;
    float* d_loss;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, sizeof(float))); // Allocate device memory for loss

    // Allocate GPU memory for the backpropagation
    float *d_dZ1, *d_dZ2, *d_dZ3;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ1, 50 * batchSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ2, 50 * batchSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ3, 10 * batchSize * sizeof(float)));

    // Copy initialized weights and biases to the device
    {
        torch::NoGradGuard no_grad;
        cudaMemcpy(layer1.weights, net->fc1->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer1.inputSize * layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer1.biases, net->fc1->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer2.weights, net->fc2->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer2.inputSize * layer2.outputSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer2.biases, net->fc2->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer2.outputSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer3.weights, net->fc3->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer3.inputSize * layer3.outputSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(layer3.biases, net->fc3->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer3.outputSize * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/batchSize);

    float total_loss_gpu = 0.0f;
    float total_loss_libtorch = 0.0f;
    int total_batches = 0;

    for (size_t epoch = 1; epoch <= 50; ++epoch) {
        for (auto& batch : *data_loader) {
            batch.data = batch.data.view({batchSize, -1});
            {
                // copy data to gpu
                torch::NoGradGuard no_grad;
                CHECK_CUDA_ERROR(cudaMemcpy(d_labels, batch.target.clone().to(torch::kCPU).to(torch::kInt32).contiguous().data_ptr(), batchSize * sizeof(int), cudaMemcpyHostToDevice));
                // Copy to GPU memory
                CHECK_CUDA_ERROR(cudaMemcpy(d_input, batch.data.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), 784 * batchSize * sizeof(float), cudaMemcpyHostToDevice));
            }
            // Forward propagation
            ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
            ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
            ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);
            // log probabilities
            LogSoftmaxBatch<<<(10 + 255) / 50, 50>>>(d_output3, d_softmax_output, 10, batchSize);
            CHECK_LAST_CUDA_ERROR();
            CHECK_CUDA_ERROR(cudaMemset(d_loss, 0, sizeof(float)));
            // loss
            CrossEntropyLoss<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_loss, 10, batchSize);
            CHECK_LAST_CUDA_ERROR();
            // Compute Dz last layer
            ComputeDzLastLayer<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_dZ3, 10, batchSize);
            CHECK_LAST_CUDA_ERROR();
            CHECK_CUDA_ERROR(cudaMemset(derivative_Loss_Weights3, 0, layer3.inputSize * layer3.outputSize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(derivative_Loss_Biases3, 0, layer3.outputSize * sizeof(float)));
            // Compute Gradients
            ComputeGradients(layer3, d_dZ3, d_output2, derivative_Loss_Weights3, derivative_Loss_Biases3, d_dZ2, batchSize, d_output2);
            ComputeGradients(layer2, d_dZ2, d_output1, derivative_Loss_Weights2, derivative_Loss_Biases2, d_dZ1, batchSize, d_output1);
            ComputeGradients(layer1, d_dZ1, d_input, derivative_Loss_Weights1, derivative_Loss_Biases1, nullptr, batchSize, nullptr);
            // Then update weights and biases
            UpdateWeightsAndBiases(layer3, derivative_Loss_Weights3, derivative_Loss_Biases3, 0.01, 1.0);
            UpdateWeightsAndBiases(layer2, derivative_Loss_Weights2, derivative_Loss_Biases2, 0.01, 1.0);
            UpdateWeightsAndBiases(layer1, derivative_Loss_Weights1, derivative_Loss_Biases1, 0.01, 1.0);

            // Copy loss from device to host memory
            loss = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            loss /= batchSize;
            // Add this batch's loss to the total loss
            // libtorch
            optimizer.zero_grad();
            auto prediction = net->forward(batch.data);
            auto libtorch_loss = torch::nn::functional::cross_entropy(prediction, batch.target);
            libtorch_loss.backward();
            optimizer.step();

            total_loss_gpu += loss;
            total_loss_libtorch += libtorch_loss.item<float>();
            ++total_batches;
        }

        std::cout << "========= Epoch " << epoch << " =========" << std::endl;
        std::cout << "Avg. GPU loss:     " << total_loss_gpu / total_batches << std::endl;
        std::cout << "Avg Libtorch loss: " << total_loss_libtorch / total_batches << std::endl;
        total_loss_gpu = 0.0f;      // Reset total loss
        total_loss_libtorch = 0.0f; // Reset total loss
        total_batches = 0;          // Reset total batches
    }

    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_softmax_output);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_dZ3);
    cudaFree(d_dZ2);
    cudaFree(d_dZ1);
    cudaFree(derivative_Loss_Weights3);
    cudaFree(derivative_Loss_Biases3);
    cudaFree(derivative_Loss_Weights2);
    cudaFree(derivative_Loss_Biases2);
    cudaFree(derivative_Loss_Weights1);
    cudaFree(derivative_Loss_Biases1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
