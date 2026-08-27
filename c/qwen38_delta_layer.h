#ifndef QWEN38_DELTA_LAYER_H
#define QWEN38_DELTA_LAYER_H

#include <stddef.h>

#include "qwen38_delta.h"
#include "qwen38_model.h"

typedef struct {
    Qwen38DeltaWeights view;
    float *storage[9];
} Qwen38DeltaLayer;

int qwen38_delta_layer_load(Qwen38Model *model, int layer,
                            Qwen38DeltaLayer *weights,
                            char *error, size_t error_size);
void qwen38_delta_layer_close(Qwen38DeltaLayer *weights);

#endif
