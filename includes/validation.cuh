#pragma once

#include <cuda_runtime.h>
#include <vector>

void ArgMax(const float* d_predictions, int* d_labels, int numClasses, int batch_size);
float ComputeAccuracy(const int* d_predictions, const int* d_labels, int batchSize);