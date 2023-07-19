#pragma once

#include "activation.cuh"
#include <memory>
#include <string>

class Layer {
public:
    Layer(const std::string& name) : _name(name) {}
    const std::string& GetName() const { return _name; }
    virtual ~Layer() = default;
    virtual float* Forward(const float* d_input, std::unique_ptr<Activation> activation) = 0;

protected:
    // virtual void Backward() = 0;
    // virtual void Update() = 0;

private:
    std::string _name;
};