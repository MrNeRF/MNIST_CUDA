#include "linear_layer.cuh"
#include "mlp.cuh"
#include "mlp_mnist.cuh"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <vector>

static const int num_epochs = 20;

int main(int argc, char** argv) {
    int batchSize = 32;
    ///////////// HERE CUDA IMPL
    MNIST_NN model(batchSize);
    float* d_input;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, 784 * batchSize * sizeof(float))); // Allocate device memory for input
    int* d_labels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batchSize * sizeof(int))); // Allocate device memory for labels

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/batchSize);

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        float total_loss_gpu = 0.0f;
        int total_batches = 0;

        for (auto& batch : *data_loader) {
            batch.data = batch.data.view({batchSize, -1});
            {
                // copy data to gpu
                torch::NoGradGuard no_grad;
                CHECK_CUDA_ERROR(cudaMemcpy(d_labels, batch.target.clone().to(torch::kCPU).to(torch::kInt32).contiguous().data_ptr(), batchSize * sizeof(int), cudaMemcpyHostToDevice));
                // Copy to GPU memory
                CHECK_CUDA_ERROR(cudaMemcpy(d_input, batch.data.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), 28 * 28 * batchSize * sizeof(float), cudaMemcpyHostToDevice));
            }
            total_loss_gpu += model.Forward(d_input, d_labels);
            model.Backward();
            model.Update(0.01);

            ++total_batches;
        }
        const float avg_loss_gpu = total_loss_gpu / total_batches;
        std::cout << "========= Epoch " << epoch << " =========" << std::endl;
        std::cout << "Avg. GPU loss:     " << avg_loss_gpu << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_labels);
}