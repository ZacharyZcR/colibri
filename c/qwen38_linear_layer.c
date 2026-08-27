#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_linear_layer.h"

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

static float *load(Qwen38Model *model, const char *name, int64_t count,
                   char *error, size_t error_size) {
    st_tensor *tensor = st_find(&model->source, name);
    if (!tensor || tensor->dtype > 2 || tensor->numel != count) {
        fail(error, error_size, "invalid hyper-connection tensor %s", name);
        return NULL;
    }
    float *result = malloc((size_t)count * sizeof(float));
    if (!result) {
        fail(error, error_size, "out of memory loading %s", name);
        return NULL;
    }
    st_read_f32(&model->source, name, result, 0);
    return result;
}

void qwen38_linear_layer_close(Qwen38LinearLayer *weights) {
    if (!weights) return;
    free(weights->norm); free(weights->mix_down); free(weights->mix_up);
    free(weights->inject);
    qwen38_delta_layer_close(&weights->delta);
    qwen38_delta_state_close(&weights->state);
    qwen38_moe_layer_close(&weights->moe);
    memset(weights, 0, sizeof(*weights));
}

int qwen38_linear_layer_load(Qwen38Model *model, int layer,
                             Qwen38LinearLayer *weights,
                             char *error, size_t error_size) {
    if (!model || !weights || layer < 0 || layer >= model->config.num_hidden_layers ||
        model->config.layer_is_full[layer])
        return fail(error, error_size, "layer %d is not linear attention", layer);
    memset(weights, 0, sizeof(*weights));
    Qwen38Config *config = &model->config;
    int64_t hyper = (int64_t)config->hc_count * config->hidden_size;
    const char *suffixes[] = {
        "hc_norm.weight", "input_mix_weight_down.weight",
        "input_mix_weight_up.weight", "block_inject_weight.weight",
    };
    const int64_t counts[] = {
        hyper, (int64_t)config->hc_lowrank * hyper,
        hyper * config->hc_lowrank, (int64_t)config->hc_count * hyper,
    };
    float **destinations[] = {
        &weights->norm, &weights->mix_down, &weights->mix_up, &weights->inject,
    };
    char name[512];
    for (int index = 0; index < 4; index++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d."
                 "attn_hyper_connection.%s", layer, suffixes[index]);
        *destinations[index] = load(model, name, counts[index], error, error_size);
        if (!*destinations[index]) goto fail;
    }
    if (qwen38_delta_layer_load(model, layer, &weights->delta, error, error_size) ||
        qwen38_delta_state_init(&weights->state, config) ||
        qwen38_moe_layer_load(model, layer, &weights->moe, error, error_size)) goto fail;
    return 0;
fail:
    qwen38_linear_layer_close(weights);
    return -1;
}

size_t qwen38_linear_layer_workspace_floats(const Qwen38Config *config) {
    if (!config) return 0;
    size_t scratch = qwen38_mhc_workspace_floats(config->hc_count,
                                                  config->hidden_size,
                                                  config->hc_lowrank);
    size_t delta = qwen38_delta_workspace_floats(config);
    size_t moe = qwen38_moe_layer_workspace_floats(config);
    if (scratch < delta) scratch = delta;
    if (scratch < moe) scratch = moe;
    return scratch + (size_t)config->hc_count * config->hidden_size +
           2ULL * config->hidden_size + config->hc_count;
}

int qwen38_linear_layer_step(Qwen38Model *model, int layer,
                             Qwen38LinearLayer *weights,
                             const float *hyper_input, float *hyper_output,
                             float *workspace, size_t workspace_floats,
                             char *error, size_t error_size) {
    if (!model || !weights || !hyper_input || !hyper_output || !workspace ||
        layer < 0 || layer >= model->config.num_hidden_layers)
        return fail(error, error_size, "invalid linear layer arguments");
    size_t required = qwen38_linear_layer_workspace_floats(&model->config);
    if (workspace_floats < required)
        return fail(error, error_size, "linear layer workspace is too small");
    Qwen38Config *config = &model->config;
    size_t scratch_size = required - (size_t)config->hc_count * config->hidden_size -
                          2ULL * config->hidden_size - config->hc_count;
    float *scratch = workspace;
    float *attention_hyper = scratch + scratch_size;
    float *mixed = attention_hyper + (size_t)config->hc_count * config->hidden_size;
    float *block = mixed + config->hidden_size;
    float *injection = block + config->hidden_size;
    if (qwen38_mhc_mix(hyper_input, config->hc_count, config->hidden_size,
                       config->hc_lowrank, config->rms_norm_eps, weights->norm,
                       weights->mix_down, weights->mix_up, weights->inject,
                       mixed, injection, scratch, scratch_size) ||
        qwen38_delta_step(config, &weights->delta.view, &weights->state,
                          mixed, block, scratch, scratch_size) ||
        qwen38_mhc_inject(hyper_input, block, injection, config->hc_count,
                          config->hidden_size, attention_hyper) ||
        qwen38_moe_layer_forward(model, layer, &weights->moe, attention_hyper,
                                 hyper_output, scratch, scratch_size,
                                 error, error_size))
        return fail(error, error_size, "linear layer %d forward failed", layer);
    return 0;
}
