#pragma once

struct NeuralNetwork {
    virtual ~NeuralNetwork() = default;
    virtual void Forward(const float* d_input) = 0;
    virtual float* Predict(const float* d_input) = 0;
};
