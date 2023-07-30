#pragma once

#include <vector>

struct NeuralNetwork {
    virtual ~NeuralNetwork() = default;
    virtual float Forward(const float* d_input, const int* d_labels) = 0;
    virtual const float* Backward() = 0;
    virtual void Update(const float learning_rate) = 0;
    virtual float Predict(const float* d_input, const int* d_labels) = 0;
};
