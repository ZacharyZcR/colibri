#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_full_layer.h"

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

static float *load(Qwen38Model *model, const char *name, int rank,
                   const int64_t *shape, char *error, size_t error_size) {
    st_tensor *tensor = st_find(&model->source, name);
    int64_t count = 1;
    if (!tensor || tensor->dtype > 2 || tensor->rank != rank) goto invalid;
    for (int dimension = 0; dimension < rank; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) goto invalid;
        count *= shape[dimension];
    }
    float *result = malloc((size_t)count * sizeof(float));
    if (!result) {
        fail(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    st_read_f32(&model->source, name, result, 0);
    return result;
invalid:
    fail(error, error_size, "invalid full-attention tensor %s", name);
    return NULL;
}

void qwen38_full_layer_close(Qwen38FullLayer *weights) {
    if (!weights) return;
    for (int index = 0; index < 9; index++) free(weights->attention_storage[index]);
    qwen38_attention_state_close(&weights->state);
    free(weights->norm); free(weights->mix_down); free(weights->mix_up);
    free(weights->inject);
    qwen38_moe_layer_close(&weights->moe);
    qwen38_ple_layer_close(&weights->ple);
    memset(weights, 0, sizeof(*weights));
}

int qwen38_full_layer_load(Qwen38Model *model, int layer, size_t capacity,
                           Qwen38FullLayer *weights,
                           char *error, size_t error_size) {
    if (!model || !weights || layer < 0 || layer >= model->config.num_hidden_layers ||
        !model->config.layer_is_full[layer] || !capacity)
        return fail(error, error_size, "layer %d is not full attention", layer);
    memset(weights, 0, sizeof(*weights));
    Qwen38Config *config = &model->config;
    int64_t hidden = config->hidden_size;
    int64_t query = (int64_t)config->attention_heads * config->head_dim;
    int64_t kv = (int64_t)config->kv_heads * config->head_dim;
    int64_t index_query = (int64_t)config->indexer_heads * config->indexer_head_dim;
    int64_t index_total = index_query +
                          (int64_t)config->indexer_kv_heads * config->indexer_head_dim;
    int64_t shapes[9][2] = {
        {2 * query, hidden}, {kv, hidden}, {kv, hidden}, {hidden, query},
        {config->head_dim}, {config->head_dim}, {index_total, hidden},
        {config->indexer_head_dim}, {config->indexer_head_dim},
    };
    const int ranks[9] = {2, 2, 2, 2, 1, 1, 2, 1, 1};
    const char *suffixes[9] = {
        "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "q_norm.weight", "k_norm.weight", "indexer.index_qk_proj.weight",
        "indexer.q_layernorm.weight", "indexer.k_layernorm.weight",
    };
    char name[512];
    for (int index = 0; index < 9; index++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.%s",
                 layer, suffixes[index]);
        weights->attention_storage[index] = load(model, name, ranks[index],
                                                  shapes[index], error, error_size);
        if (!weights->attention_storage[index]) goto fail;
    }
    weights->attention = (Qwen38AttentionWeights){
        weights->attention_storage[0], weights->attention_storage[1],
        weights->attention_storage[2], weights->attention_storage[3],
        weights->attention_storage[4], weights->attention_storage[5],
        weights->attention_storage[6], weights->attention_storage[7],
        weights->attention_storage[8],
    };
    int64_t hyper = (int64_t)config->hc_count * hidden;
    const char *hyper_suffixes[] = {
        "hc_norm.weight", "input_mix_weight_down.weight",
        "input_mix_weight_up.weight", "block_inject_weight.weight",
    };
    const int64_t hyper_shapes[4][2] = {
        {hyper, 0}, {config->hc_lowrank, hyper},
        {hyper, config->hc_lowrank}, {config->hc_count, hyper},
    };
    const int hyper_ranks[4] = {1, 2, 2, 2};
    float **destinations[] = {
        &weights->norm, &weights->mix_down, &weights->mix_up, &weights->inject,
    };
    for (int index = 0; index < 4; index++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d."
                 "attn_hyper_connection.%s", layer, hyper_suffixes[index]);
        *destinations[index] = load(model, name, hyper_ranks[index],
                                    hyper_shapes[index], error, error_size);
        if (!*destinations[index]) goto fail;
    }
    if (qwen38_attention_state_init(&weights->state, config, capacity) ||
        qwen38_moe_layer_load(model, layer, &weights->moe, error, error_size)) goto fail;
    for (int index = 0; index < config->ple_layer_count; index++)
        weights->has_ple |= config->ple_layers[index] == layer + 1;
    if (weights->has_ple &&
        qwen38_ple_layer_load(model, layer, &weights->ple, error, error_size)) goto fail;
    return 0;
fail:
    qwen38_full_layer_close(weights);
    return -1;
}

size_t qwen38_full_layer_workspace_floats(const Qwen38Config *config) {
    if (!config) return 0;
    size_t scratch = qwen38_mhc_workspace_floats(config->hc_count,
                                                  config->hidden_size,
                                                  config->hc_lowrank);
    size_t attention = qwen38_attention_workspace_floats(config);
    size_t moe = qwen38_moe_layer_workspace_floats(config);
    size_t ple = qwen38_ple_layer_workspace_floats(config);
    if (scratch < attention) scratch = attention;
    if (scratch < moe) scratch = moe;
    if (scratch < ple) scratch = ple;
    return scratch + (size_t)config->hc_count * config->hidden_size +
           2ULL * config->hidden_size + config->hc_count;
}

int qwen38_full_layer_step(Qwen38Model *model, int layer,
                           Qwen38FullLayer *weights, int64_t token,
                           const float *hyper_input, float *hyper_output,
                           float *workspace, size_t workspace_floats,
                           char *error, size_t error_size) {
    if (!model || !weights || !hyper_input || !hyper_output || !workspace ||
        layer < 0 || layer >= model->config.num_hidden_layers)
        return fail(error, error_size, "invalid full layer arguments");
    size_t required = qwen38_full_layer_workspace_floats(&model->config);
    if (workspace_floats < required)
        return fail(error, error_size, "full layer workspace is too small");
    Qwen38Config *config = &model->config;
    size_t scratch_size = required - (size_t)config->hc_count * config->hidden_size -
                          2ULL * config->hidden_size - config->hc_count;
    float *scratch = workspace;
    float *attention_hyper = scratch + scratch_size;
    float *mixed = attention_hyper + (size_t)config->hc_count * config->hidden_size;
    float *block = mixed + config->hidden_size;
    float *injection = block + config->hidden_size;
    const float *layer_input = hyper_input;
    if (weights->has_ple) {
        if (qwen38_ple_layer_step(model, &weights->ple, token, hyper_input,
                                  attention_hyper, scratch, scratch_size,
                                  error, error_size))
            return fail(error, error_size, "PLE full layer %d failed", layer);
        layer_input = attention_hyper;
    }
    if (qwen38_mhc_mix(layer_input, config->hc_count, config->hidden_size,
                       config->hc_lowrank, config->rms_norm_eps, weights->norm,
                       weights->mix_down, weights->mix_up, weights->inject,
                       mixed, injection, scratch, scratch_size) ||
        qwen38_attention_step(config, &weights->attention, &weights->state,
                              mixed, block, scratch, scratch_size) ||
        qwen38_mhc_inject(layer_input, block, injection, config->hc_count,
                          config->hidden_size, attention_hyper) ||
        qwen38_moe_layer_forward(model, layer, &weights->moe, attention_hyper,
                                 hyper_output, scratch, scratch_size,
                                 error, error_size))
        return fail(error, error_size, "full layer %d forward failed", layer);
    return 0;
}
