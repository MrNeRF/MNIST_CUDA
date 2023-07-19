#pragma once

#include "layer.cuh"
#include <cstdint>

class LinearLayer : public Layer {
public:
    LinearLayer(uint32_t input_size, uint32_t output_size);
    ~LinearLayer();
    uint32_t GetInputSize() const { return _h_input_size; }
    uint32_t GetOutputSize() const { return _h_output_size; }
    float* Forward(const float* d_input) override;

private:
    void initWeightsAndBias();

private:
    uint32_t _h_input_size;
    uint32_t _h_output_size;
    float* _d_weights;
    float* _d_bias;
    float* _d_output;
};