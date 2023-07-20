#pragma once

float CrossEntropyLoss(const float* d_predictions,
                       const int* d_labels,
                       const int numClasses,
                       const int batchSize);