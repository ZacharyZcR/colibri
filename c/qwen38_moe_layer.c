#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_moe_layer.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen38_mhc.h"

static int fail(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static float *load_tensor(Qwen38Model *model, const char *name, int64_t count,
                          char *error, size_t error_size) {
    st_tensor *tensor = st_find(&model->source, name);
    if (!tensor || tensor->dtype > 2 || tensor->numel != count) {
        fail(error, error_size, "invalid dense tensor %s", name);
        return NULL;
    }
    float *output = malloc((size_t)count * sizeof(float));
    if (!output) {
        fail(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    st_read_f32(&model->source, name, output, 0);
    return output;
}

#define LOAD(field, suffix, count) do { \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix); \
    weights->field = load_tensor(model, name, count, error, error_size); \
    if (!weights->field) goto fail; \
} while (0)

void qwen38_moe_layer_close(Qwen38MoeLayer *weights) {
    if (!weights) return;
    free(weights->norm); free(weights->mix_down); free(weights->mix_up);
    free(weights->inject); free(weights->router); free(weights->shared_gate);
    free(weights->shared_up); free(weights->shared_down);
    free(weights->shared_selector);
    memset(weights, 0, sizeof(*weights));
}

int qwen38_moe_layer_load(Qwen38Model *model, int layer,
                          Qwen38MoeLayer *weights,
                          char *error, size_t error_size) {
    if (!model || !weights || layer < 0 || layer >= model->config.num_hidden_layers)
        return fail(error, error_size, "invalid MoE layer");
    memset(weights, 0, sizeof(*weights));
    Qwen38Config *config = &model->config;
    int64_t hyper = (int64_t)config->hc_count * config->hidden_size;
    char prefix[256], name[512];
    snprintf(prefix, sizeof(prefix),
             "model.language_model.layers.%d.mlp_hyper_connection.", layer);
    LOAD(norm, "hc_norm.weight", hyper);
    LOAD(mix_down, "input_mix_weight_down.weight", (int64_t)config->hc_lowrank * hyper);
    LOAD(mix_up, "input_mix_weight_up.weight", hyper * config->hc_lowrank);
    LOAD(inject, "block_inject_weight.weight", (int64_t)config->hc_count * hyper);
    snprintf(prefix, sizeof(prefix), "model.language_model.layers.%d.mlp.", layer);
    LOAD(router, "gate.weight", (int64_t)config->num_experts * config->hidden_size);
    LOAD(shared_gate, "shared_expert.gate_proj.weight",
         (int64_t)config->shared_expert_intermediate_size * config->hidden_size);
    LOAD(shared_up, "shared_expert.up_proj.weight",
         (int64_t)config->shared_expert_intermediate_size * config->hidden_size);
    LOAD(shared_down, "shared_expert.down_proj.weight",
         (int64_t)config->hidden_size * config->shared_expert_intermediate_size);
    LOAD(shared_selector, "shared_expert_gate.weight", config->hidden_size);
    return 0;
fail:
    qwen38_moe_layer_close(weights);
    return -1;
}

#undef LOAD

static float dot(const float *weights, const float *input, int count) {
    double sum = 0.0;
    for (int index = 0; index < count; index++) sum += weights[index] * input[index];
    return (float)sum;
}

static void matvec(const float *weights, const float *input,
                   int rows, int columns, float *output) {
    for (int row = 0; row < rows; row++)
        output[row] = dot(weights + (int64_t)row * columns, input, columns);
}

int qwen38_router_topk(const float *input, const float *router,
                       int hidden_size, int experts, int topk,
                       int *selected, float *weights, float *logits) {
    if (!input || !router || !selected || !weights || !logits || hidden_size < 1 ||
        experts < 1 || topk < 1 || topk > experts) return -1;
    for (int expert = 0; expert < experts; expert++)
        logits[expert] = dot(router + (int64_t)expert * hidden_size, input, hidden_size);
    for (int rank = 0; rank < topk; rank++) {
        int best = -1;
        for (int expert = 0; expert < experts; expert++) {
            int used = 0;
            for (int previous = 0; previous < rank; previous++)
                used |= selected[previous] == expert;
            if (!used && (best < 0 || logits[expert] > logits[best])) best = expert;
        }
        if (best < 0 || !isfinite(logits[best])) return -1;
        selected[rank] = best;
    }
    float maximum = logits[selected[0]], total = 0.0f;
    for (int rank = 0; rank < topk; rank++) {
        weights[rank] = expf(logits[selected[rank]] - maximum);
        total += weights[rank];
    }
    if (!(total > 0.0f) || !isfinite(total)) return -1;
    for (int rank = 0; rank < topk; rank++) weights[rank] /= total;
    return 0;
}

size_t qwen38_moe_layer_workspace_floats(const Qwen38Config *config) {
    if (!config) return 0;
    size_t hyper = (size_t)config->hc_count * config->hidden_size;
    size_t scratch = qwen38_mhc_workspace_floats(
        config->hc_count, config->hidden_size, config->hc_lowrank);
    size_t expert_scratch = 2ULL * config->moe_intermediate_size + config->hidden_size;
    if (scratch < expert_scratch) scratch = expert_scratch;
    return scratch +
           hyper + config->hc_count + 3ULL * config->hidden_size +
           2ULL * config->shared_expert_intermediate_size +
           config->num_experts + 2ULL * config->experts_per_token;
}

int qwen38_moe_layer_forward(Qwen38Model *model, int row,
                             const Qwen38MoeLayer *weights,
                             const float *hyper_input, float *hyper_output,
                             float *workspace, size_t workspace_floats,
                             char *error, size_t error_size) {
    if (!model || !weights || !hyper_input || !hyper_output || !workspace ||
        workspace_floats < qwen38_moe_layer_workspace_floats(&model->config))
        return fail(error, error_size, "invalid MoE layer forward arguments");
    Qwen38Config *config = &model->config;
    size_t mhc_required = qwen38_mhc_workspace_floats(
        config->hc_count, config->hidden_size, config->hc_lowrank);
    size_t scratch_size = mhc_required;
    size_t expert_scratch = 2ULL * config->moe_intermediate_size + config->hidden_size;
    if (scratch_size < expert_scratch) scratch_size = expert_scratch;
    float *cursor = workspace;
    float *scratch = cursor; cursor += scratch_size;
    float *original = cursor; cursor += (size_t)config->hc_count * config->hidden_size;
    float *injection = cursor; cursor += config->hc_count;
    float *mixed = cursor; cursor += config->hidden_size;
    float *routed = cursor; cursor += config->hidden_size;
    float *shared = cursor; cursor += config->hidden_size;
    float *gate = cursor; cursor += config->shared_expert_intermediate_size;
    float *up = cursor; cursor += config->shared_expert_intermediate_size;
    float *logits = cursor; cursor += config->num_experts;
    int *selected = (int *)cursor;
    float *route_weights = cursor + config->experts_per_token;
    memcpy(original, hyper_input,
           (size_t)config->hc_count * config->hidden_size * sizeof(float));
    if (qwen38_mhc_mix(hyper_input, config->hc_count, config->hidden_size,
                       config->hc_lowrank, config->rms_norm_eps, weights->norm,
                       weights->mix_down, weights->mix_up, weights->inject,
                       mixed, injection, scratch, scratch_size) ||
        qwen38_router_topk(mixed, weights->router, config->hidden_size,
                           config->num_experts, config->experts_per_token,
                           selected, route_weights, logits) ||
        qwen38_moe_forward(model, row, mixed, selected, route_weights,
                           config->experts_per_token, routed, scratch,
                           scratch_size, error, error_size))
        return fail(error, error_size, "MoE routed path failed");
    matvec(weights->shared_gate, mixed, config->shared_expert_intermediate_size,
           config->hidden_size, gate);
    matvec(weights->shared_up, mixed, config->shared_expert_intermediate_size,
           config->hidden_size, up);
    for (int index = 0; index < config->shared_expert_intermediate_size; index++)
        gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
    matvec(weights->shared_down, gate, config->hidden_size,
           config->shared_expert_intermediate_size, shared);
    float selector = 1.0f / (1.0f + expf(-dot(weights->shared_selector, mixed,
                                               config->hidden_size)));
    for (int index = 0; index < config->hidden_size; index++)
        routed[index] += selector * shared[index];
    if (qwen38_mhc_inject(original, routed, injection, config->hc_count,
                          config->hidden_size, hyper_output))
        return fail(error, error_size, "MoE residual injection failed");
    return 0;
}
