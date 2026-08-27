#ifndef QWEN38_PLE_LAYER_H
#define QWEN38_PLE_LAYER_H

#include <stddef.h>
#include <stdint.h>

#include "qwen38_model.h"
#include "qwen38_ple_hash.h"

typedef struct {
    int table_index;
    Qwen38PleHash hash;
    float *key_proj, *value_proj;
    float *norm_key, *norm_query, *norm_conv, *conv;
    int64_t history[QWEN38_PLE_MAX_NGRAM - 1];
    float *conv_state;
} Qwen38PleLayer;

int qwen38_ple_layer_load(Qwen38Model *model, int layer,
                          Qwen38PleLayer *weights,
                          char *error, size_t error_size);
void qwen38_ple_layer_close(Qwen38PleLayer *weights);
size_t qwen38_ple_layer_workspace_floats(const Qwen38Config *config);
int qwen38_ple_layer_step(Qwen38Model *model, Qwen38PleLayer *weights,
                          int64_t token, const float *hyper_input,
                          float *hyper_output, float *workspace,
                          size_t workspace_floats,
                          char *error, size_t error_size);

#endif
