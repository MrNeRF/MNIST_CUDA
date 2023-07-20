#include "activation.cuh"
#include "linear_layer.cuh"
#include "load_mnist.cuh"
#include "mlp.cuh"
#include "neural_network.cuh"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

void Predict(const DenseLayer& layer1, const DenseLayer& layer2, const DenseLayer& layer3, const float* d_input, int* outputLabels, const int batchSize) {
    float* d_output1;
    cudaMalloc((void**)&d_output1, 50 * batchSize * sizeof(float)); // output size of layer1
    float* d_output2;
    cudaMalloc((void**)&d_output2, 50 * batchSize * sizeof(float)); // output size of layer2
    float* d_output3;
    cudaMalloc((void**)&d_output3, 10 * batchSize * sizeof(float)); // output size of layer3
    float* d_softmax_output;
    cudaMalloc((void**)&d_softmax_output, 10 * batchSize * sizeof(float)); // output after softmax

    // Perform forward propagation
    ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
    ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
    ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);

    // Apply softmax
    LogSoftmaxBatch<<<(10 + 255) / 50, 50>>>(d_output3, d_softmax_output, 10, batchSize);

    // Compute argmax of each row
    ArgMaxKernel<<<(batchSize + 255) / 256, 256>>>(d_softmax_output, outputLabels, 10, batchSize);

    // Free the device memory
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_softmax_output);
}

struct MNIST_NN : public NeuralNetwork {
    MNIST_NN(const int batch_size) : _batch_size(batch_size) {
        _fc1 = std::make_unique<LinearLayer>(batch_size, 784, 50);
        _fc2 = std::make_unique<LinearLayer>(batch_size, 50, 50);
        _fc3 = std::make_unique<LinearLayer>(batch_size, 50, 10);
    }

    float Forward(const float* d_input, const int* d_labels) override {
        float* output = nullptr;
        output = _fc1->Forward(d_input, std::make_unique<ReLU>());
        output = _fc2->Forward(output, std::make_unique<ReLU>());
        output = _fc3->Forward(output, std::make_unique<LogSoftMax>());
        return output;
    }

    float* Backward(const float* d_output) override {
        return nullptr;
    };
    void Update(const float learning_rate) override {
        return nullptr;
    };
    float* Predict(const float* d_input) override {
        return nullptr;
    }

    int _batch_size = 64;
    std::unique_ptr<LinearLayer> _fc1, _fc2, _fc3;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./load_mnist <path_to_mnist_data>" << std::endl;
        return -1;
    }

    static const int batchSize = 64;
    auto data_path = std::filesystem::path(argv[1]);
    MNISTDataSet mnist_trainind_data;
    mnist_trainind_data.ReadMNISTData(data_path / "train-images-idx3-ubyte", data_path / "train-labels-idx1-ubyte");

    MNISTDataSet mnist_validation_data;
    mnist_validation_data.ReadMNISTData(data_path / "t10k-images-idx3-ubyte", data_path / "t10k-labels-idx1-ubyte");

    const std::vector<std::vector<float>>& validationData = mnist_validation_data.GetImages();
    const std::vector<uint8_t>& validationLabels = mnist_validation_data.GetLabels();

    int* d_predictions;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_predictions, batchSize * sizeof(int))); // Allocate device memory for predictions
    float validation_input[784 * batchSize];
    int* d_validationLabels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_validationLabels, batchSize * sizeof(int))); // Allocate device memory for labels

    // Create a fully connected layer
    DenseLayer layer1(784, 50);
    DenseLayer layer2(50, 50);
    DenseLayer layer3(50, 10);
    InitializeLayer(layer1);
    InitializeLayer(layer2);
    InitializeLayer(layer3);

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

    int labels[batchSize];
    int validation_labels[batchSize];
    int* d_labels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batchSize * sizeof(int))); // Allocate device memory for labels

    float loss;
    float* d_loss;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_loss, sizeof(float))); // Allocate device memory for loss

    const std::vector<std::vector<float>>& trainingData = mnist_trainind_data.GetImages();
    const std::vector<uint8_t>& trainingLabels = mnist_trainind_data.GetLabels();

    float input[784 * batchSize];

    // Allocate GPU memory for the backpropagation
    float *d_dZ1, *d_dZ2, *d_dZ3;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ1, 50 * batchSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ2, 50 * batchSize * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dZ3, 10 * batchSize * sizeof(float)));

    curandState* d_states;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_states, 50 * batchSize * sizeof(curandState)));
    InitCurandState<<<(50 * batchSize + 255) / 50, 50>>>(d_states, time(0));
    CHECK_LAST_CUDA_ERROR();

    static const int num_epochs = 100;

    auto rng = std::default_random_engine{};

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "Epoch: " << (epoch + 1) << std::endl;

        // Combine the training images and labels into a single structure
        std::vector<std::pair<std::vector<float>, uint8_t>> training_pairs(trainingData.size());
        for (size_t i = 0; i < trainingData.size(); ++i) {
            training_pairs[i] = std::make_pair(trainingData[i], trainingLabels[i]);
        }

        // Shuffle the training data at the beginning of each epoch
        std::shuffle(training_pairs.begin(), training_pairs.end(), rng);

        // Initialize total loss for this epoch
        float total_loss = 0.0f;
        int total_batches = 0;

        for (int i = 0; i < trainingData.size(); i += batchSize) {
            // Flatten and normalize the image
            for (int j = 0; j < batchSize; j++) {
                if (i + j < training_pairs.size()) {
                    const auto& image = training_pairs[i + j].first;
                    std::copy(image.begin(), image.end(), input + j * 784);

                    labels[j] = training_pairs[i + j].second;
                } else {
                    // padding with zeros if there are not enough images for a full batch
                    std::fill_n(input + j * 784, 784, 0.0f);
                    labels[j] = 0; // padding with zeros
                }
            }

            // Copy labels to device memory
            CHECK_CUDA_ERROR(cudaMemcpy(d_labels, labels, batchSize * sizeof(int), cudaMemcpyHostToDevice));
            // Copy to GPU memory
            CHECK_CUDA_ERROR(cudaMemcpy(d_input, input, 784 * batchSize * sizeof(float), cudaMemcpyHostToDevice));

            // Perform forward propagation
            ForwardPropagation(layer1, d_input, d_output1, true, batchSize);
            // float *layer1_output = new float[layer1.outputSize * batchSize];
            // CHECK_CUDA_ERROR(cudaMemcpy(layer1_output, d_output1, layer1.outputSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            // DropoutKernel<<<(50 * batchSize + 255) / 50, 50>>>(d_output1, d_mask1, dropout_rate, 50 * batchSize, d_states);
            ForwardPropagation(layer2, d_output1, d_output2, true, batchSize);
            // float *layer2_output = new float[layer2.outputSize * batchSize];
            // CHECK_CUDA_ERROR(cudaMemcpy(layer2_output, d_output2, layer2.outputSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost));
            // DropoutKernel<<<(50 * batchSize + 255) / 50, 50>>>(d_output2, d_mask2, dropout_rate, 50 * batchSize, d_states);
            ForwardPropagation(layer3, d_output2, d_output3, false, batchSize);
            // Compute the softmax function for the outputs of layer3
            LogSoftmaxBatch<<<(10 + 255) / 50, 50>>>(d_output3, d_softmax_output, 10, batchSize);
            CHECK_LAST_CUDA_ERROR();

            // Initialize loss to zero
            CHECK_CUDA_ERROR(cudaMemset(d_loss, 0, sizeof(float)));
            CrossEntropyLoss<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_loss, 10, batchSize);
            CHECK_LAST_CUDA_ERROR();

            // Calculate dZ for the last layer based on your loss function and softmax derivative
            ComputeDzLastLayer<<<(batchSize + 255) / 50, 50>>>(d_softmax_output, d_labels, d_dZ3, 10, batchSize);
            CHECK_LAST_CUDA_ERROR();

            CHECK_CUDA_ERROR(cudaMemset(derivative_Loss_Weights3, 0, layer3.inputSize * layer3.outputSize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemset(derivative_Loss_Biases3, 0, layer3.outputSize * sizeof(float)));
            // Perform backpropagation
            ComputeGradients(layer3, d_dZ3, d_output2, derivative_Loss_Weights3, derivative_Loss_Biases3, d_dZ2, batchSize, d_output2);
            ComputeGradients(layer2, d_dZ2, d_output1, derivative_Loss_Weights2, derivative_Loss_Biases2, d_dZ1, batchSize, d_output1);
            ComputeGradients(layer1, d_dZ1, d_input, derivative_Loss_Weights1, derivative_Loss_Biases1, nullptr, batchSize, nullptr);
            // ApplyMaskKernel<<<(50 * batchSize + 255) / 50, 50>>>(d_dA_prev2, d_mask1, 50 * batchSize);

            // Then update weights
            UpdateWeightsAndBiases(layer3, derivative_Loss_Weights3, derivative_Loss_Biases3, 0.01, 1.0);
            UpdateWeightsAndBiases(layer2, derivative_Loss_Weights2, derivative_Loss_Biases2, 0.01, 1.0);
            UpdateWeightsAndBiases(layer1, derivative_Loss_Weights1, derivative_Loss_Biases1, 0.01, 1.0);

            // Copy loss from device to host memory
            loss = 0;
            CHECK_CUDA_ERROR(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            loss /= batchSize;
            // Add this batch's loss to the total loss
            total_loss += loss;
            ++total_batches;
        }
        // Print the average loss for this epoch
        std::cout << "Average Loss: " << total_loss / total_batches << std::endl;

        float total_prediction_accuracy = 0.0f;
        total_batches = 0;
        // Validation
        for (int i = 0; i < validationData.size(); i += batchSize) {
            // Flatten and normalize the image
            for (int j = 0; j < batchSize; j++) {
                if (i + j < validationData.size()) {
                    const auto& image = validationData[i + j];
                    std::copy(image.begin(), image.end(), validation_input + j * 784);
                } else {
                    // padding with zeros if there are not enough images for a full batch
                    std::fill_n(validation_input + j * 784, 784, 0.0f);
                }

                if (i + j < validationLabels.size()) {
                    validation_labels[j] = validationLabels[i + j];
                } else {
                    validation_labels[j] = 0; // padding with zeros
                }
            }

            // Copy labels to device memory
            cudaMemcpy(d_validationLabels, validation_labels, batchSize * sizeof(int), cudaMemcpyHostToDevice);
            // Copy to GPU memory
            cudaMemcpy(d_input, validation_input, 784 * batchSize * sizeof(float), cudaMemcpyHostToDevice);

            Predict(layer1, layer2, layer3, d_input, d_predictions, batchSize);

            float accuracy = ComputeAccuracy(d_predictions, d_validationLabels, batchSize);

            total_prediction_accuracy += accuracy;
            ++total_batches;
        }

        std::cout << "Average Validation Accuracy: " << total_prediction_accuracy / total_batches << std::endl;
    }

    cudaFree(d_predictions);
    cudaFree(d_validationLabels);
    cudaFree(d_labels);
    cudaFree(d_loss);
    cudaFree(d_input);
    cudaFree(d_output1);
    cudaFree(d_output2);
    cudaFree(d_output3);
    cudaFree(d_mask1);
    cudaFree(d_mask2);
    cudaFree(d_softmax_output);
    cudaFree(d_dZ1);
    cudaFree(d_dZ2);
    cudaFree(d_dZ3);
    cudaFree(d_states);
    cudaFree(derivative_Loss_Weights1);
    cudaFree(derivative_Loss_Biases1);
    cudaFree(derivative_Loss_Weights2);
    cudaFree(derivative_Loss_Biases2);
    cudaFree(derivative_Loss_Weights3);
    cudaFree(derivative_Loss_Biases3);
    return 0;
}