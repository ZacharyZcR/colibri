#ifndef QWEN38_EXPERT_H
#define QWEN38_EXPERT_H

#include <stddef.h>
#include <stdint.h>

#include "qwen38_model.h"

typedef struct {
    int8_t *weights;
    float *scales;
} Qwen38Expert;

int qwen38_expert_load(Qwen38Model *model, int row, int expert,
                       Qwen38Expert *output, char *error, size_t error_size);
void qwen38_expert_close(Qwen38Expert *expert);
int qwen38_expert_forward(const Qwen38Expert *expert,
                          const Qwen38Config *config, const float *input,
                          float *output, float *workspace,
                          size_t workspace_floats);
int qwen38_moe_forward(Qwen38Model *model, int row, const float *input,
                       const int *expert_ids, const float *router_weights,
                       int selected, float *output, float *workspace,
                       size_t workspace_floats, char *error, size_t error_size);

#endif
