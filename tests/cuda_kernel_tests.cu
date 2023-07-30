#include "linear_layer.cuh"
#include "mlp.cuh"
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

struct SimpleNN : public NeuralNetwork {
    SimpleNN(const int batch_size) : _batch_size(batch_size) {
        _fc1 = std::make_unique<LinearLayer>(_batch_size, 4, 3);
        _fc2 = std::make_unique<LinearLayer>(_batch_size, 3, 3);
        _fc3 = std::make_unique<LinearLayer>(_batch_size, 3, 2);
        _loss = std::make_unique<CrossEntropyLoss>(2, _batch_size);

        CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_predictions, _batch_size * sizeof(int))); // Allocate device memory for predictions
    }

    float Forward(const float* d_input, const int* d_labels) override {
        _d_input = d_input;
        _d_labels = d_labels;
        const float* output = nullptr;
        output = _fc1->Forward(d_input, std::make_unique<ReLU>());
        output = _fc2->Forward(output, std::make_unique<ReLU>());
        output = _fc3->Forward(output, std::make_unique<LogSoftMax>());
        return (*_loss)(output, _d_labels);
    }

    const float* Backward() override {
        const float* d_dZ = nullptr;
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
        const float* output = nullptr;
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
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        torch::Tensor x1, x2, x3;

        x1 = torch::relu(fc1->forward(x));
        x2 = torch::relu(fc2->forward(x1));
        x3 = fc3->forward(x2);

        return std::make_tuple(x1, x2, x3);
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

// This test is to check if the forward pass is working correctly
// The weights and biases are initialized to a known value
// The output of the forward pass is compared with the output of the libtorch implementation
// The loss is also compared with the libtorch implementation
// The gradients are also compared with the libtorch implementation
TEST(ForwardPassLossLibtorch, BasicTest) {
    const int batchSize = 5;

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
    SimpleNN myNN(batchSize);
    myNN._fc1->SetWeightsFromCPU(h_weights1);
    myNN._fc1->SetBiasFromCPU(h_biases1);
    myNN._fc2->SetWeightsFromCPU(h_weights2);
    myNN._fc2->SetBiasFromCPU(h_biases2);
    myNN._fc3->SetWeightsFromCPU(h_weights3);
    myNN._fc3->SetBiasFromCPU(h_biases3);
    const float gpu_loss = myNN.Forward(d_input, d_labels);

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

    auto [pred1, pred2, pred3] = torchNet->forward(inputTensor);

    // Because Simple_NN outputs logsoftmax, we need to convert it
    auto pred3_logsoftmax = torch::log_softmax(pred3, 1);
    auto torchnet_Layer1_output = pred1.data_ptr<float>();
    auto torchnet_Layer2_output = pred2.data_ptr<float>();
    auto torchnet_Layer3_output = pred3_logsoftmax.data_ptr<float>();

    auto myNN_Layer1_output = myNN._fc1->GetOutputCPU();
    auto myNN_Layer2_output = myNN._fc2->GetOutputCPU();
    auto myNN_Layer3_output = myNN._fc3->GetOutputCPU();

    for (int i = 0; i < 12; ++i) {
        EXPECT_NEAR(torchnet_Layer1_output[i], myNN_Layer1_output[i], 1e-4);
    }
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(torchnet_Layer2_output[i], myNN_Layer2_output[i], 1e-4);
    }

    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(torchnet_Layer3_output[i], myNN_Layer3_output[i], 1e-4);
    }

    // This step is necessary because torch is picky about the input type
    // and the labels are int64_t. If you insert std::vector<int>, the loss computation explodes.
    std::vector<int64_t> labels_torch_long(labels.begin(), labels.end());

    auto tensorLables = torch::from_blob(labels_torch_long.data(), {5}, torch::TensorOptions().dtype(torch::kLong));
    auto libtorch_loss = torch::nn::functional::cross_entropy(pred3, tensorLables);
    EXPECT_NEAR(gpu_loss, libtorch_loss.item<float>(), 1e-4);

    torch::optim::SGD optimizer(torchNet->parameters(), /*lr=*/0.01);
    optimizer.zero_grad();
    libtorch_loss.backward();
    optimizer.step();

    myNN.Backward();
    myNN.Update(0.01);
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

    const bool isGradientWeightLayer2Defined = torchNet->fc2->weight.grad().defined();
    ASSERT_TRUE(isGradientWeightLayer2Defined);
    if (isGradientWeightLayer2Defined) {
        const auto gpuGradientWeightsLayer2 = myNN._fc2->GetWeightGradientsCPU();
        for (int i = 0; i < gpuGradientWeightsLayer2.size(); ++i) {
            EXPECT_NEAR(torchNet->fc2->weight.grad().view(-1)[i].item<float>(), gpuGradientWeightsLayer2[i], 1e-4);
        }
    }

    const bool isGradientBiasLayer2Defined = torchNet->fc2->bias.grad().defined();
    ASSERT_TRUE(isGradientBiasLayer2Defined);
    if (isGradientBiasLayer2Defined) {
        const auto gpuGradientBiasesLayer2 = myNN._fc2->GetBiasGradientsCPU();
        for (int i = 0; i < gpuGradientBiasesLayer2.size(); ++i) {
            EXPECT_NEAR(torchNet->fc2->bias.grad().view(-1)[i].item<float>(), gpuGradientBiasesLayer2[i], 1e-4);
        }
    }

    const bool isGradientWeightLayer1Defined = torchNet->fc1->weight.grad().defined();
    ASSERT_TRUE(isGradientWeightLayer1Defined);
    if (isGradientWeightLayer1Defined) {
        const auto gpuGradientWeightsLayer1 = myNN._fc1->GetWeightGradientsCPU();
        for (int i = 0; i < gpuGradientWeightsLayer1.size(); ++i) {
            EXPECT_NEAR(torchNet->fc1->weight.grad().view(-1)[i].item<float>(), gpuGradientWeightsLayer1[i], 1e-4);
        }
    }

    const bool isGradientBiasLayer1Defined = torchNet->fc1->bias.grad().defined();
    ASSERT_TRUE(isGradientBiasLayer1Defined);
    if (isGradientBiasLayer1Defined) {
        const auto gpuGradientBiasesLayer1 = myNN._fc1->GetBiasGradientsCPU();
        for (int i = 0; i < gpuGradientBiasesLayer1.size(); ++i) {
            EXPECT_NEAR(torchNet->fc1->bias.grad().view(-1)[i].item<float>(), gpuGradientBiasesLayer1[i], 1e-4);
        }
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_labels);
}

TEST(MNIST_Test, BasicTest) {
    struct TorchNet : torch::nn::Module {
        TorchNet(int inputDim1, int outputDim1, int inputDim2, int outputDim2, int inputDim3, int outputDim3) {
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

    auto torch_net = std::make_shared<TorchNet>(28 * 28, 50,
                                                50, 50,
                                                50, 10);

    torch::optim::SGD optimizer(torch_net->parameters(), /*lr=*/0.01);
    int batchSize = 32;
    ///////////// HERE CUDA IMPL
    MNIST_NN model(batchSize);
    float* d_input;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, 784 * batchSize * sizeof(float))); // Allocate device memory for input
    int* d_labels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batchSize * sizeof(int))); // Allocate device memory for labels

    // Initialize weights and biases to values from libtorch
    {
        torch::NoGradGuard no_grad;
        model._fc1->SetWeightsFromCPU(reinterpret_cast<const float*>(torch_net->fc1->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr()));
        model._fc1->SetBiasFromCPU(reinterpret_cast<const float*>(torch_net->fc1->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr()));
        model._fc2->SetWeightsFromCPU(reinterpret_cast<const float*>(torch_net->fc2->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr()));
        model._fc2->SetBiasFromCPU(reinterpret_cast<const float*>(torch_net->fc2->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr()));
        model._fc3->SetWeightsFromCPU(reinterpret_cast<const float*>(torch_net->fc3->weight.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr()));
        model._fc3->SetBiasFromCPU(reinterpret_cast<const float*>(torch_net->fc3->bias.clone().to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr()));
    }

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/batchSize);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        float total_loss_gpu = 0.0f;
        float total_loss_libtorch = 0.0f;
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
            const float gpu_loss = model.Forward(d_input, d_labels);
            model.Backward();
            model.Update(0.01);

            // libtorch
            optimizer.zero_grad();
            auto prediction = torch_net->forward(batch.data);
            auto libtorch_loss = torch::nn::functional::cross_entropy(prediction, batch.target);
            libtorch_loss.backward();
            optimizer.step();

            const float libtorch_loss_float = libtorch_loss.item<float>();
            total_loss_libtorch += libtorch_loss_float;
            total_loss_gpu += gpu_loss;
            ++total_batches;
        }
        const float avg_loss_gpu = total_loss_gpu / total_batches;
        const float avg_loss_libtorch = total_loss_libtorch / total_batches;
        EXPECT_NEAR(avg_loss_gpu, avg_loss_libtorch, 1e-4);
        std::cout << "========= Epoch " << epoch << " =========" << std::endl;
        std::cout << "Avg. GPU loss:     " << avg_loss_gpu << std::endl;
        std::cout << "Avg Libtorch loss: " << avg_loss_libtorch << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_labels);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
