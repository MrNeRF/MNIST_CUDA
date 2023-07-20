#pragma once

#include "activation.cuh"
#include <memory>
#include <string>

class Layer {
public:
    virtual ~Layer() = default;
    virtual float* Forward(const float* d_input, std::unique_ptr<Activation> activation) = 0;

    virtual int GetInputSize() const = 0;
    virtual int GetOutputSize() const = 0;
    virtual float* GetWeightsGPU() = 0;
    virtual float* GetBiasGPU() = 0;
    virtual float* GetOutputGPU() = 0;
    virtual std::vector<float> GetWeightsCPU() const = 0;
    virtual std::vector<float> GetBiasCPU() const = 0;
    virtual std::vector<float> GetOutputCPU() const = 0;
};