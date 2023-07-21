#pragma once

struct Loss {
    virtual ~Loss() = default;
    virtual float operator()(const float* d_predictions, const int* d_labels) = 0;
    virtual float* Backward(float* d_predictions, const int* d_labels) = 0;
};

class CrossEntropyLoss : public Loss {
public:
    CrossEntropyLoss(const int num_classes, const int batch_size);
    ~CrossEntropyLoss();

    float operator()(const float* d_predictions, const int* d_labels) override;
    float* Backward(float* d_predictions, const int* d_labels) override;

private:
    float* _d_dZ;
    float* _d_loss;
    const int _num_classes;
    const int _batch_size;
};
