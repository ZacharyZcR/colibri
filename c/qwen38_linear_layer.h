#ifndef QWEN38_LINEAR_LAYER_H
#define QWEN38_LINEAR_LAYER_H

#include <stddef.h>

#include "qwen38_delta_layer.h"
#include "qwen38_moe_layer.h"

typedef struct {
    float *norm, *mix_down, *mix_up, *inject;
    Qwen38DeltaLayer delta;
    Qwen38DeltaState state;
    Qwen38MoeLayer moe;
} Qwen38LinearLayer;

int qwen38_linear_layer_load(Qwen38Model *model, int layer,
                             Qwen38LinearLayer *weights,
                             char *error, size_t error_size);
void qwen38_linear_layer_close(Qwen38LinearLayer *weights);
size_t qwen38_linear_layer_workspace_floats(const Qwen38Config *config);
int qwen38_linear_layer_step(Qwen38Model *model, int layer,
                             Qwen38LinearLayer *weights,
                             const float *hyper_input, float *hyper_output,
                             float *workspace, size_t workspace_floats,
                             char *error, size_t error_size);

#endif
