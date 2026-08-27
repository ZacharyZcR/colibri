#ifndef QWEN38_ATTENTION_H
#define QWEN38_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

#include "qwen38_config.h"
#include "qwen38_qsa.h"

typedef struct {
    const float *q_proj, *k_proj, *v_proj, *o_proj;
    const float *q_norm, *k_norm;
    const float *index_qk_proj, *index_q_norm, *index_k_norm;
} Qwen38AttentionWeights;

typedef struct {
    size_t length, capacity;
    float *keys, *values, *raw_index_keys, *scores, *pooled_key;
    uint8_t *selected;
    Qwen38QsaEntry *heap;
    size_t heap_capacity;
} Qwen38AttentionState;

int qwen38_attention_state_init(Qwen38AttentionState *state,
                                const Qwen38Config *config,
                                size_t capacity);
void qwen38_attention_state_close(Qwen38AttentionState *state);
size_t qwen38_attention_workspace_floats(const Qwen38Config *config);
int qwen38_attention_step(const Qwen38Config *config,
                          const Qwen38AttentionWeights *weights,
                          Qwen38AttentionState *state, const float *input,
                          float *output, float *workspace,
                          size_t workspace_floats);

#endif
