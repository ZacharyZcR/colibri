#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_delta_layer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (!tensor || tensor->dtype > 2 || tensor->rank != rank)
        goto invalid;
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
    fail(error, error_size, "invalid DeltaNet tensor %s", name);
    return NULL;
}

void qwen38_delta_layer_close(Qwen38DeltaLayer *weights) {
    if (!weights) return;
    for (int index = 0; index < 9; index++) free(weights->storage[index]);
    memset(weights, 0, sizeof(*weights));
}

int qwen38_delta_layer_load(Qwen38Model *model, int layer,
                            Qwen38DeltaLayer *weights,
                            char *error, size_t error_size) {
    if (!model || !weights || layer < 0 || layer >= model->config.num_hidden_layers ||
        model->config.layer_is_full[layer])
        return fail(error, error_size, "layer %d is not DeltaNet", layer);
    memset(weights, 0, sizeof(*weights));
    Qwen38Config *config = &model->config;
    int64_t hidden = config->hidden_size;
    int64_t value_heads = config->linear_value_heads;
    int64_t value_dim = value_heads * config->linear_value_dim;
    int64_t conv_dim = 2LL * config->linear_key_heads * config->linear_key_dim + value_dim;
    int64_t shapes[9][3] = {
        {conv_dim, hidden}, {value_dim, hidden}, {value_heads, hidden},
        {value_heads, hidden}, {conv_dim, 1, config->linear_conv_kernel},
        {value_heads}, {value_heads}, {config->linear_value_dim},
        {hidden, value_dim},
    };
    const int ranks[9] = {2, 2, 2, 2, 3, 1, 1, 1, 2};
    const char *suffixes[9] = {
        "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
        "in_proj_a.weight", "conv1d.weight", "A_log", "dt_bias",
        "norm.weight", "out_proj.weight",
    };
    char name[512];
    for (int index = 0; index < 9; index++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.linear_attn.%s",
                 layer, suffixes[index]);
        weights->storage[index] = load(model, name, ranks[index], shapes[index],
                                       error, error_size);
        if (!weights->storage[index]) goto fail;
    }
    weights->view = (Qwen38DeltaWeights){
        weights->storage[0], weights->storage[1], weights->storage[2],
        weights->storage[3], weights->storage[4], weights->storage[5],
        weights->storage[6], weights->storage[7], weights->storage[8],
    };
    return 0;
fail:
    qwen38_delta_layer_close(weights);
    return -1;
}
