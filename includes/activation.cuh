#pragma once

struct Activation {
    ~Activation();
    virtual void operator()(float* d_output) = 0;
};

struct ReLU : public Activation {
    ReLU(const int batch_size,
         const int output_size);
    void operator()(float* d_output);

private:
    const int _batch_size;
    const int _output_size;
};
