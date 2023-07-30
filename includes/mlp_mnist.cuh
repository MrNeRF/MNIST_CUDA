#pragma once

#include "activation.cuh"
#include "error_checking.cuh"
#include "linear_layer.cuh"
#include "loss.cuh"
#include "neural_network.cuh"
#include "validation.cuh"
#include <iomanip>
#include <memory>

struct MNIST_NN : public NeuralNetwork {
    MNIST_NN(const int batch_size) : _batch_size(batch_size) {
        _fc1 = std::make_unique<LinearLayer>(_batch_size, 784, 50);
        _fc2 = std::make_unique<LinearLayer>(_batch_size, 50, 50);
        _fc3 = std::make_unique<LinearLayer>(_batch_size, 50, 10);
        _loss = std::make_unique<CrossEntropyLoss>(10, _batch_size);

        CHECK_CUDA_ERROR(cudaMalloc((void**)&_d_predictions, _batch_size * sizeof(int))); // Allocate device memory for predictions
    }

    float Forward(const float* d_input, const int* d_labels) override {
        _d_input = d_input;
        _d_labels = d_labels;
        const float* output = nullptr;
        output = _fc1->Forward(d_input, std::make_unique<ReLU>());
        output = _fc2->Forward(output, std::make_unique<ReLU>());
        output = _fc3->Forward(output, std::make_unique<LogSoftMax>());
        return (*_loss)(output, d_labels);
    }

    float* Backward() override {
        const float* d_dZ = nullptr;
        d_dZ = (*_loss).Backward(_fc3->GetOutputGPU(), _d_labels);
        d_dZ = _fc3->Backward(d_dZ, _fc2->GetOutputGPU());
        d_dZ = _fc2->Backward(d_dZ, _fc1->GetOutputGPU());
        d_dZ = _fc1->Backward(d_dZ, _d_input);
        d_dZ = nullptr;
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

    const int _batch_size = 64;
    std::unique_ptr<LinearLayer> _fc1, _fc2, _fc3;
    std::unique_ptr<Loss> _loss;

private:
    const float* _d_input; // temporary variable, no ownership
    const int* _d_labels;  // temporary variable, no ownership
    int* _d_predictions;
};