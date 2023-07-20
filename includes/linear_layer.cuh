#pragma once

#include "layer.cuh"
#include <cstdint>
#include <vector>

class LinearLayer : public Layer {
public:
    LinearLayer(int batch_size, int input_size, int output_size);
    ~LinearLayer();
    int GetInputSize() const override { return _h_input_size; }
    int GetOutputSize() const override { return _h_output_size; }
    float* GetWeightsGPU() override { return _d_weights; }
    float* GetBiasGPU() override { return _d_bias; }
    float* GetOutputGPU() override { return _d_output; }

    std::vector<float> GetWeightsCPU() const override;
    std::vector<float> GetBiasCPU() const override;
    std::vector<float> GetOutputCPU() const override;

    float* Forward(const float* d_input, std::unique_ptr<Activation> activation) override;

private:
    int _h_batch_size;
    int _h_input_size;
    int _h_output_size;
    float* _d_weights;
    float* _d_bias;
    float* _d_output;
};