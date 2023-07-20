#pragma once

struct NeuralNetwork {
    virtual ~NeuralNetwork() = default;
    virtual float Forward(const float* d_input, const int* d_labels) = 0;
    virtual float* Backward(const float* d_output) = 0;
    virtual void Update(const float learning_rate) = 0;
    virtual float* Predict(const float* d_input) = 0;
};
