#ifndef QWEN38_DELTA_H
#define QWEN38_DELTA_H

#include <stddef.h>

#include "qwen38_config.h"

typedef struct {
    float *recurrent;
    float *convolution;
} Qwen38DeltaState;

typedef struct {
    const float *qkv, *z, *b, *a, *conv;
    const float *a_log, *dt_bias, *norm, *out;
} Qwen38DeltaWeights;

size_t qwen38_delta_workspace_floats(const Qwen38Config *config);
int qwen38_delta_state_init(Qwen38DeltaState *state,
                            const Qwen38Config *config);
void qwen38_delta_state_close(Qwen38DeltaState *state);
int qwen38_delta_step(const Qwen38Config *config,
                      const Qwen38DeltaWeights *weights,
                      Qwen38DeltaState *state, const float *input,
                      float *output, float *workspace,
                      size_t workspace_floats);

#endif
