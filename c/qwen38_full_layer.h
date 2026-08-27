#ifndef QWEN38_FULL_LAYER_H
#define QWEN38_FULL_LAYER_H

#include <stddef.h>
#include <stdint.h>

#include "qwen38_attention.h"
#include "qwen38_model.h"
#include "qwen38_moe_layer.h"
#include "qwen38_ple_layer.h"

typedef struct {
    float *attention_storage[9];
    Qwen38AttentionWeights attention;
    Qwen38AttentionState state;
    float *norm, *mix_down, *mix_up, *inject;
    Qwen38MoeLayer moe;
    Qwen38PleLayer ple;
    int has_ple;
} Qwen38FullLayer;

int qwen38_full_layer_load(Qwen38Model *model, int layer, size_t capacity,
                           Qwen38FullLayer *weights,
                           char *error, size_t error_size);
void qwen38_full_layer_close(Qwen38FullLayer *weights);
size_t qwen38_full_layer_workspace_floats(const Qwen38Config *config);
int qwen38_full_layer_step(Qwen38Model *model, int layer,
                           Qwen38FullLayer *weights, int64_t token,
                           const float *hyper_input, float *hyper_output,
                           float *workspace, size_t workspace_floats,
                           char *error, size_t error_size);

#endif
