#include "mlp_mnist.cuh"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <torch/torch.h>
#include <vector>

struct Simple_GPU_NN : public NeuralNetwork {
    Simple_GPU_NN(const int batch_size) : _batch_size(batch_size) {
        _fc1 = std::make_unique<LinearLayer>(_batch_size, 4, 3);
        _fc2 = std::make_unique<LinearLayer>(_batch_size, 3, 3);
        _fc3 = std::make_unique<LinearLayer>(_batch_size, 3, 2);
        _loss = std::make_unique<CrossEntropyLoss>(10, _batch_size);

        CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_predictions, _batch_size * sizeof(int))); // Allocate device memory for predictions
    }

    float Forward(const float* d_input, const int* d_labels) override {
        _d_input = d_input;
        _d_labels = d_labels;
        float* output = nullptr;
        output = _fc1->Forward(d_input, std::make_unique<ReLU>());
        output = _fc2->Forward(output, std::make_unique<ReLU>());
        output = _fc3->Forward(output, std::make_unique<LogSoftMax>());
        return (*_loss)(output, d_labels);
    }

    float* Backward() override {
        float* d_dZ = nullptr;
        d_dZ = (*_loss).Backward(_fc3->GetOutputGPU(), _d_labels);
        d_dZ = _fc3->Backward(d_dZ, _fc2->GetOutputGPU());
        d_dZ = _fc2->Backward(d_dZ, _fc1->GetOutputGPU());
        d_dZ = _fc1->Backward(d_dZ, _d_input);
        return nullptr;
    };

    // this right now has internally SGD optimizer
    // Need to refactor later on
    void Update(const float learning_rate) override {
        _fc3->Update(learning_rate);
        _fc2->Update(learning_rate);
        _fc1->Update(learning_rate);
    };

    float Predict(const float* d_input, const int* d_labels) override {
        float* output = nullptr;
        output = _fc1->Forward(d_input, std::make_unique<ReLU>());
        output = _fc2->Forward(output, std::make_unique<ReLU>());
        output = _fc3->Forward(output, std::make_unique<LogSoftMax>());
        ArgMax(output, _d_predictions, 10, _batch_size);
        // For now we are just returning the accuracy and not the predicted labels
        return ComputeAccuracy(_d_predictions, d_labels, _batch_size);
    }

    int _batch_size = 64;
    std::unique_ptr<LinearLayer> _fc1, _fc2, _fc3;
    std::unique_ptr<Loss> _loss;

private:
    const float* _d_input; // temporary variable, no ownership
    const int* _d_labels;  // temporary variable, no ownership
    int* _d_predictions;
};

// libtorch implementation
struct Libtorch_Simple_Net : torch::nn::Module {
    Libtorch_Simple_Net(int inputDim1, int outputDim1, int inputDim2, int outputDim2, int inputDim3, int outputDim3) {
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

TEST(ForwardPassLossLibtorch, BasicTest) {
    const int batchSize = 5;
    const int inputDim1 = 4;
    const int outputDim1 = 3;
    const int inputDim2 = 3;
    const int outputDim2 = 3;
    const int inputDim3 = 3;
    const int outputDim3 = 2;
    const int numberOfClasses = 2;

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

    // Prepare device memory
    float* d_input;
    int* d_labels;

    cudaMalloc(&d_input, input.size() * sizeof(float));
    cudaMalloc(&d_labels, labels.size() * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    // Copy initialized weights and biases to the device
    Simple_GPU_NN myNN(batchSize);
    myNN._fc1->SetWeightsFromCPU(h_weights1);
    myNN._fc1->SetBiasFromCPU(h_biases1);
    myNN._fc2->SetWeightsFromCPU(h_weights2);
    myNN._fc2->SetBiasFromCPU(h_biases2);
    myNN._fc3->SetWeightsFromCPU(h_weights3);
    myNN._fc3->SetBiasFromCPU(h_biases3);
    const float gpu_loss = myNN.Forward(d_input, d_labels);
    myNN.Backward();

    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto inputTensor = torch::from_blob(input.data(), {5, 4}, options).requires_grad_(true);
    auto weightTensor1 = torch::from_blob(h_weights1, {3, 4}, options).requires_grad_(true);
    auto biasTensor1 = torch::from_blob(h_biases1, {3}, options).requires_grad_(true);
    auto weightTensor2 = torch::from_blob(h_weights2, {3, 3}, options).requires_grad_(true);
    auto biasTensor2 = torch::from_blob(h_biases2, {3}, options).requires_grad_(true);
    auto weightTensor3 = torch::from_blob(h_weights3, {2, 3}, options).requires_grad_(true);
    auto biasTensor3 = torch::from_blob(h_biases3, {2}, options).requires_grad_(true);

    auto torchNet = std::make_shared<Libtorch_Simple_Net>(4, 3,
                                                          3, 3,
                                                          3, 2);

    torchNet->fc1->weight = weightTensor1;
    torchNet->fc1->bias = biasTensor1;
    torchNet->fc2->weight = weightTensor2;
    torchNet->fc2->bias = biasTensor2;
    torchNet->fc3->weight = weightTensor3;
    torchNet->fc3->bias = biasTensor3;

    torch::optim::SGD optimizer(torchNet->parameters(), /*lr=*/0.01);
    optimizer.zero_grad();
    auto prediction = torchNet->forward(inputTensor);

    // This step is necessary because torch is picky about the input type
    // and the labels are int64_t. If you insert std::vector<int>, the loss computation explodes.
    std::vector<int64_t> labels_torch_long(labels.begin(), labels.end());

    auto tensorLables = torch::from_blob(labels_torch_long.data(), {5}, torch::TensorOptions().dtype(torch::kLong));
    auto libtorch_loss = torch::nn::functional::cross_entropy(prediction, tensorLables);
    EXPECT_NEAR(gpu_loss, libtorch_loss.item<float>(), 1e-4);

    libtorch_loss.backward();
    optimizer.step();

    const bool isGradientWeightLayer3Defined = torchNet->fc3->weight.grad().defined();
    ASSERT_TRUE(isGradientWeightLayer3Defined);
    if (isGradientWeightLayer3Defined) {
        const auto gpuGradientWeightsLayer3 = myNN._fc3->GetWeightGradientsCPU();
        for (int i = 0; i < gpuGradientWeightsLayer3.size(); ++i) {
            EXPECT_NEAR(torchNet->fc3->weight.grad().view(-1)[i].item<float>(), gpuGradientWeightsLayer3[i], 1e-4);
        }
    }

    const bool isGradientBiasLayer3Defined = torchNet->fc3->bias.grad().defined();
    ASSERT_TRUE(isGradientBiasLayer3Defined);
    if (isGradientBiasLayer3Defined) {
        const auto gpuGradientBiasesLayer3 = myNN._fc3->GetBiasGradientsCPU();
        for (int i = 0; i < gpuGradientBiasesLayer3.size(); ++i) {
            EXPECT_NEAR(torchNet->fc3->bias.grad().view(-1)[i].item<float>(), gpuGradientBiasesLayer3[i], 1e-4);
        }
    }

    // Copy output data to host
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_labels);
}

// TEST(MNIST_Test, BasicTest) {
//     struct Net : torch::nn::Module {
//         Net(int inputDim1, int outputDim1, int inputDim2, int outputDim2, int inputDim3, int outputDim3) {
//             // Construct and register two Linear submodules.
//             fc1 = register_module("fc1", torch::nn::Linear(inputDim1, outputDim1));
//             fc2 = register_module("fc2", torch::nn::Linear(inputDim2, outputDim2));
//             fc3 = register_module("fc3", torch::nn::Linear(inputDim3, outputDim3));
//         }

//         // Implement the Net's algorithm.
//         torch::Tensor forward(torch::Tensor x) {
//             // Use one of many tensor manipulation functions.
//             x = torch::relu(fc1->forward(x));
//             x = torch::relu(fc2->forward(x));
//             x = fc3->forward(x);

//             return x;
//         }

//         // Use one of many "standard library" modules.
//         torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
//     };

//     auto net = std::make_shared<Net>(28 * 28, 50,
//                                      50, 50,
//                                      50, 10);

//     torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
//     int batchSize = 32;
//     ///////////// HERE CUDA IMPL
//     DenseLayer layer1(784, 50);
//     DenseLayer layer2(50, 50);
//     DenseLayer layer3(50, 10);

//     // Allocate GPU memory
//     float* d_input;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, 784 * batchSize * sizeof(float))); // input size of layer1
//     float* d_output1;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output1, 50 * batchSize * sizeof(float))); // output size of layer1 (input size of layer2)
//     float* d_output2;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output2, 50 * batchSize * sizeof(float))); // output size of layer2 (input size of layer3)
//     float* d_output3;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output3, 10 * batchSize * sizeof(float))); // output size of layer3
//     float* d_softmax_output;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_softmax_output, 10 * batchSize * sizeof(float)));
//     float dropout_rate = 0.5; // Adjust this value as needed
//     float* d_mask1;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mask1, 50 * batchSize * sizeof(float))); // mask for layer1
//     float* d_mask2;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_mask2, 50 * batchSize * sizeof(float))); // mask for layer2

//     // Allocate memory for gradients
//     float* derivative_Loss_Weights1;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Weights1, layer1.inputSize * layer1.outputSize * sizeof(float)));
//     float* derivative_Loss_Biases1;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Biases1, layer1.outputSize * sizeof(float)));

//     float* derivative_Loss_Weights2;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Weights2, layer2.inputSize * layer2.outputSize * sizeof(float)));
//     float* derivative_Loss_Biases2;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Biases2, layer2.outputSize * sizeof(float)));

//     float* derivative_Loss_Weights3;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Weights3, layer3.inputSize * layer3.outputSize * sizeof(float)));
//     float* derivative_Loss_Biases3;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&derivative_Loss_Biases3, layer3.outputSize * sizeof(float)));

//     int* d_labels;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batchSize * sizeof(int))); // Allocate device memory for labels

//     float loss;
//     float* d_loss;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, sizeof(float))); // Allocate device memory for loss

//     // Allocate GPU memory for the backpropagation
//     float *d_dZ1, *d_dZ2, *d_dZ3;
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ1, 50 * batchSize * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ2, 50 * batchSize * sizeof(float)));
//     CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ3, 10 * batchSize * sizeof(float)));

//     // Copy initialized weights and biases to the device
//     {
//         torch::NoGradGuard no_grad;
//         cudaMemcpy(layer1.weights, net->fc1->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer1.inputSize * layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(layer1.biases, net->fc1->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer1.outputSize * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(layer2.weights, net->fc2->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer2.inputSize * layer2.outputSize * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(layer2.biases, net->fc2->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer2.outputSize * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(layer3.weights, net->fc3->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer3.inputSize * layer3.outputSize * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(layer3.biases, net->fc3->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), layer3.outputSize * sizeof(float), cudaMemcpyHostToDevice);
//     }

//     // Create a multi-threaded data loader for the MNIST dataset.
//     auto data_loader = torch::data::make_data_loader(
//         torch::data::datasets::MNIST("./data").map(
//             torch::data::transforms::Stack<>()),
//         /*batch_size=*/batchSize);

//     float total_loss_gpu = 0.0f;
//     float total_loss_libtorch = 0.0f;
//     int total_batches = 0;

//     for (size_t epoch = 1; epoch <= 50; ++epoch) {
//         for (auto& batch : *data_loader) {
//             batch.data = batch.data.view({batchSize, -1});
//             {
//                 // copy data to gpu
//                 torch::NoGradGuard no_grad;
//                 CHECK_CUDA_ERROR(cudaMemcpy(d_labels, batch.target.clone().to(torch::kCPU).to(torch::kInt32).contiguous().data_ptr(), batchSize * sizeof(int), cudaMemcpyHostToDevice));
//                 // Copy to GPU memory
//                 CHECK_CUDA_ERROR(cudaMemcpy(d_input, batch.data.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), 784 * batchSize * sizeof(float), cudaMemcpyHostToDevice));
//             }
//             // Forward propagation
//             ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
//             ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
//             ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);
//             // log probabilities
//             LogSoftmaxBatch<<<(10 + 255) / 50, 50>>>(d_output3, d_softmax_output, 10, batchSize);
//             CHECK_LAST_CUDA_ERROR();
//             CHECK_CUDA_ERROR(cudaMemset(d_loss, 0, sizeof(float)));
//             // loss
//             CrossEntropyLoss<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_loss, 10, batchSize);
//             CHECK_LAST_CUDA_ERROR();
//             // Compute Dz last layer
//             ComputeDzLastLayer<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_dZ3, 10, batchSize);
//             CHECK_LAST_CUDA_ERROR();
//             CHECK_CUDA_ERROR(cudaMemset(derivative_Loss_Weights3, 0, layer3.inputSize * layer3.outputSize * sizeof(float)));
//             CHECK_CUDA_ERROR(cudaMemset(derivative_Loss_Biases3, 0, layer3.outputSize * sizeof(float)));
//             // Compute Gradients
//             ComputeGradients(layer3, d_dZ3, d_output2, derivative_Loss_Weights3, derivative_Loss_Biases3, d_dZ2, batchSize, d_output2);
//             ComputeGradients(layer2, d_dZ2, d_output1, derivative_Loss_Weights2, derivative_Loss_Biases2, d_dZ1, batchSize, d_output1);
//             ComputeGradients(layer1, d_dZ1, d_input, derivative_Loss_Weights1, derivative_Loss_Biases1, nullptr, batchSize, nullptr);
//             // Then update weights and biases
//             UpdateWeightsAndBiases(layer3, derivative_Loss_Weights3, derivative_Loss_Biases3, 0.01, 1.0);
//             UpdateWeightsAndBiases(layer2, derivative_Loss_Weights2, derivative_Loss_Biases2, 0.01, 1.0);
//             UpdateWeightsAndBiases(layer1, derivative_Loss_Weights1, derivative_Loss_Biases1, 0.01, 1.0);

//             // Copy loss from device to host memory
//             loss = 0;
//             CHECK_CUDA_ERROR(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
//             loss /= batchSize;
//             // Add this batch's loss to the total loss
//             // libtorch
//             optimizer.zero_grad();
//             auto prediction = net->forward(batch.data);
//             auto libtorch_loss = torch::nn::functional::cross_entropy(prediction, batch.target);
//             libtorch_loss.backward();
//             optimizer.step();

//             total_loss_gpu += loss;
//             total_loss_libtorch += libtorch_loss.item<float>();
//             ++total_batches;
//         }

//         std::cout << "========= Epoch " << epoch << " =========" << std::endl;
//         std::cout << "Avg. GPU loss:     " << total_loss_gpu / total_batches << std::endl;
//         std::cout << "Avg Libtorch loss: " << total_loss_libtorch / total_batches << std::endl;
//         total_loss_gpu = 0.0f;      // Reset total loss
//         total_loss_libtorch = 0.0f; // Reset total loss
//         total_batches = 0;          // Reset total batches
//     }

//     cudaFree(d_input);
//     cudaFree(d_output1);
//     cudaFree(d_output2);
//     cudaFree(d_output3);
//     cudaFree(d_softmax_output);
//     cudaFree(d_labels);
//     cudaFree(d_loss);
//     cudaFree(d_dZ3);
//     cudaFree(d_dZ2);
//     cudaFree(d_dZ1);
//     cudaFree(derivative_Loss_Weights3);
//     cudaFree(derivative_Loss_Biases3);
//     cudaFree(derivative_Loss_Weights2);
//     cudaFree(derivative_Loss_Biases2);
//     cudaFree(derivative_Loss_Weights1);
//     cudaFree(derivative_Loss_Biases1);
// }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
