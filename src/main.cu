#include "load_mnist.cuh"
#include "mlp_mnist.cuh"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>

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

    // training data
    int labels[batchSize];
    float* d_input;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, 784 * batchSize * sizeof(float))); // Allocate device memory for input
    int* d_labels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_labels, batchSize * sizeof(int))); // Allocate device memory for labels

    const std::vector<std::vector<float>>& trainingData = mnist_trainind_data.GetImages();
    const std::vector<uint8_t>& trainingLabels = mnist_trainind_data.GetLabels();

    float input[784 * batchSize];

    // validation data
    const std::vector<std::vector<float>>& validationData = mnist_validation_data.GetImages();
    const std::vector<uint8_t>& validationLabels = mnist_validation_data.GetLabels();

    float validation_input[784 * batchSize];
    int* d_validationLabels;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_validationLabels, batchSize * sizeof(int))); // Allocate device memory for labels
    int validation_labels[batchSize];

    static const int num_epochs = 100;

    auto rng = std::default_random_engine{};

    MNIST_NN model(batchSize);

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
            // Copy loss from device to host memory
            float loss = model.Forward(d_input, d_labels);
            model.Backward();
            model.Update(0.01f);

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

            float accuracy = model.Predict(d_input, d_validationLabels);

            total_prediction_accuracy += accuracy;
            ++total_batches;
        }

        std::cout << "Average Validation Accuracy: " << total_prediction_accuracy / total_batches << std::endl;
    }

    cudaFree(d_validationLabels);
    cudaFree(d_labels);
    cudaFree(d_input);
    return 0;
}