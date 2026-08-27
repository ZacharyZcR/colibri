#ifndef QWEN38_RUNTIME_H
#define QWEN38_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "qwen38_full_layer.h"
#include "qwen38_linear_layer.h"

typedef struct {
    int full_attention;
    Qwen38LinearLayer linear;
    Qwen38FullLayer full;
} Qwen38RuntimeLayer;

typedef struct {
    Qwen38Model model;
    Qwen38RuntimeLayer *layers;
    float *embedding, *lm_head;
    float *final_norm, *final_mix_down, *final_mix_up;
    float *hyper_a, *hyper_b, *hidden, *workspace;
    size_t workspace_floats, length, capacity;
} Qwen38Runtime;

int qwen38_runtime_open(Qwen38Runtime *runtime, const char *source_dir,
                        const char *expert_dir, size_t capacity,
                        char *error, size_t error_size);
void qwen38_runtime_close(Qwen38Runtime *runtime);
int qwen38_runtime_step(Qwen38Runtime *runtime, int64_t token,
                        float *logits, size_t logits_count,
                        char *error, size_t error_size);

#endif
