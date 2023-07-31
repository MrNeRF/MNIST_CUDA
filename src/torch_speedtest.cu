#include <iostream>
#include <torch/torch.h>

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

static const int num_epochs = 20;

int main(int argc, char** argv) {

    // Check if CUDA is available
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (device == torch::kCUDA) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "Training on CPU." << std::endl;
    }

    auto torch_net = std::make_shared<TorchNet>(28 * 28, 50, 50, 50, 50, 10);
    torch_net->to(device); // Move network to CUDA if available

    torch::optim::SGD optimizer(torch_net->parameters(), /*lr=*/0.01);
    int batchSize = 32;

    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/batchSize);

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        float total_loss_libtorch = 0.0f;
        int total_batches = 0;

        for (auto& batch : *data_loader) {
            batch.data = batch.data.view({batchSize, -1}).to(device); // Move input data to CUDA if available
            batch.target = batch.target.to(device);                   // Move target data to CUDA if available

            optimizer.zero_grad();
            auto prediction = torch_net->forward(batch.data);
            auto libtorch_loss = torch::nn::functional::cross_entropy(prediction, batch.target);
            libtorch_loss.backward();
            optimizer.step();

            total_loss_libtorch += libtorch_loss.item<float>();
            ++total_batches;
        }
        const float avg_loss_libtorch = total_loss_libtorch / total_batches;
        std::cout << "========= Epoch " << epoch << " =========" << std::endl;
        std::cout << "Avg Libtorch loss: " << avg_loss_libtorch << std::endl;
    }
}