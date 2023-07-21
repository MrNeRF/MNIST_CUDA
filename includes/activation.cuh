#pragma once

struct Activation {
    virtual ~Activation() = default;
    virtual void operator()(const int batch_size,
                            const int output_size,
                            float* d_value) = 0;
};

struct ReLU : public Activation {
    void operator()(const int batch_size,
                    const int output_size,
                    float* d_value);
};

struct LogSoftMax : public Activation {
    void operator()(const int batch_size,
                    const int output_size,
                    float* d_value);
};