#ifndef QWEN38_MOE_LAYER_H
#define QWEN38_MOE_LAYER_H

#include <stddef.h>

#include "qwen38_expert.h"

typedef struct {
    float *norm, *mix_down, *mix_up, *inject;
    float *router;
    float *shared_gate, *shared_up, *shared_down, *shared_selector;
} Qwen38MoeLayer;

int qwen38_moe_layer_load(Qwen38Model *model, int layer,
                          Qwen38MoeLayer *weights,
                          char *error, size_t error_size);
void qwen38_moe_layer_close(Qwen38MoeLayer *weights);
size_t qwen38_moe_layer_workspace_floats(const Qwen38Config *config);
int qwen38_router_topk(const float *input, const float *router,
                       int hidden_size, int experts, int topk,
                       int *selected, float *weights, float *logits);
int qwen38_moe_layer_forward(Qwen38Model *model, int row,
                             const Qwen38MoeLayer *weights,
                             const float *hyper_input, float *hyper_output,
                             float *workspace, size_t workspace_floats,
                             char *error, size_t error_size);

#endif
