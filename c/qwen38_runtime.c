#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_runtime.h"

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
    fail(error, error_size, "invalid runtime tensor %s", name);
    return NULL;
}

static void matvec(const float *weights, const float *input,
                   int rows, int columns, float *output) {
    for (int row = 0; row < rows; row++) {
        double sum = 0.0;
        for (int column = 0; column < columns; column++)
            sum += weights[(int64_t)row * columns + column] * input[column];
        output[row] = (float)sum;
    }
}

void qwen38_runtime_close(Qwen38Runtime *runtime) {
    if (!runtime) return;
    if (runtime->layers) for (int layer = 0;
         layer < runtime->model.config.num_hidden_layers; layer++) {
        if (runtime->layers[layer].full_attention)
            qwen38_full_layer_close(&runtime->layers[layer].full);
        else
            qwen38_linear_layer_close(&runtime->layers[layer].linear);
    }
    free(runtime->layers); free(runtime->embedding); free(runtime->lm_head);
    free(runtime->final_norm); free(runtime->final_mix_down);
    free(runtime->final_mix_up); free(runtime->hyper_a); free(runtime->hyper_b);
    free(runtime->hidden); free(runtime->workspace);
    qwen38_model_close(&runtime->model);
    if (runtime->accelerator_initialized) qwen38_accel_shutdown();
    memset(runtime, 0, sizeof(*runtime));
}

static void reset_ple(Qwen38PleLayer *ple, const Qwen38Config *config) {
    if (!ple->conv_state) return;
    size_t hyper = (size_t)config->hc_count * config->hidden_size;
    size_t state = hyper * (size_t)(config->ple_conv_kernel - 1) *
                   config->ngram_size;
    memset(ple->conv_state, 0, state * sizeof(float));
    for (int index = 0; index < config->ngram_size - 1; index++)
        ple->history[index] = config->eos_token_id;
}

void qwen38_runtime_reset(Qwen38Runtime *runtime) {
    if (!runtime || !runtime->layers) return;
    Qwen38Config *config = &runtime->model.config;
    size_t recurrent = (size_t)config->linear_value_heads *
        config->linear_key_dim * config->linear_value_dim;
    size_t conv_dim = 2ULL * config->linear_key_heads * config->linear_key_dim +
                       (size_t)config->linear_value_heads * config->linear_value_dim;
    size_t convolution = conv_dim * (size_t)(config->linear_conv_kernel - 1);
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        Qwen38RuntimeLayer *item = &runtime->layers[layer];
        if (item->full_attention) {
            item->full.state.length = 0;
            if (item->full.has_ple) reset_ple(&item->full.ple, config);
        } else {
            memset(item->linear.state.recurrent, 0, recurrent * sizeof(float));
            memset(item->linear.state.convolution, 0,
                   convolution * sizeof(float));
            if (item->linear.has_ple) reset_ple(&item->linear.ple, config);
        }
    }
    runtime->length = 0;
}

int qwen38_runtime_open(Qwen38Runtime *runtime, const char *source_dir,
                        const char *expert_dir, size_t capacity,
                        char *error, size_t error_size) {
    if (!runtime || !capacity) return fail(error, error_size, "invalid runtime open");
    memset(runtime, 0, sizeof(*runtime));
    if (qwen38_model_open(&runtime->model, source_dir, expert_dir,
                          error, error_size)) return -1;
    if (qwen38_accel_init(error, error_size)) {
        qwen38_model_close(&runtime->model);
        return -1;
    }
    runtime->accelerator_initialized = 1;
    Qwen38Config *config = &runtime->model.config;
    int64_t embedding_shape[] = {config->vocab_size, config->hidden_size};
    runtime->embedding = load(&runtime->model,
        "model.language_model.embed_tokens.weight", 2, embedding_shape,
        error, error_size);
    runtime->lm_head = load(&runtime->model, "lm_head.weight", 2,
                            embedding_shape, error, error_size);
    int64_t hyper = (int64_t)config->hc_count * config->hidden_size;
    int64_t norm_shape[] = {hyper};
    int64_t down_shape[] = {config->hc_lowrank, hyper};
    int64_t up_shape[] = {hyper, config->hc_lowrank};
    runtime->final_norm = load(&runtime->model,
        "model.language_model.hyper_connection_mixer.hc_norm.weight",
        1, norm_shape, error, error_size);
    runtime->final_mix_down = load(&runtime->model,
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        2, down_shape, error, error_size);
    runtime->final_mix_up = load(&runtime->model,
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        2, up_shape, error, error_size);
    if (!runtime->embedding || !runtime->lm_head || !runtime->final_norm ||
        !runtime->final_mix_down || !runtime->final_mix_up) goto fail;

    runtime->layers = calloc((size_t)config->num_hidden_layers,
                             sizeof(*runtime->layers));
    if (!runtime->layers) {
        fail(error, error_size, "out of memory allocating runtime layers");
        goto fail;
    }
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        runtime->layers[layer].full_attention = config->layer_is_full[layer];
        int result = config->layer_is_full[layer] ?
            qwen38_full_layer_load(&runtime->model, layer, capacity,
                                    &runtime->layers[layer].full,
                                    error, error_size) :
            qwen38_linear_layer_load(&runtime->model, layer,
                                      &runtime->layers[layer].linear,
                                      error, error_size);
        if (result) goto fail;
    }
    size_t linear_workspace = qwen38_linear_layer_workspace_floats(config);
    size_t full_workspace = qwen38_full_layer_workspace_floats(config);
    size_t final_workspace = qwen38_mhc_workspace_floats(
        config->hc_count, config->hidden_size, config->hc_lowrank);
    runtime->workspace_floats = linear_workspace;
    if (runtime->workspace_floats < full_workspace)
        runtime->workspace_floats = full_workspace;
    if (runtime->workspace_floats < final_workspace)
        runtime->workspace_floats = final_workspace;
    runtime->hyper_a = malloc((size_t)hyper * sizeof(float));
    runtime->hyper_b = malloc((size_t)hyper * sizeof(float));
    runtime->hidden = malloc((size_t)config->hidden_size * sizeof(float));
    runtime->workspace = malloc(runtime->workspace_floats * sizeof(float));
    if (!runtime->hyper_a || !runtime->hyper_b || !runtime->hidden ||
        !runtime->workspace) {
        fail(error, error_size, "out of memory allocating runtime workspace");
        goto fail;
    }
    runtime->capacity = capacity;
    return 0;
fail:
    qwen38_runtime_close(runtime);
    return -1;
}

int qwen38_runtime_step(Qwen38Runtime *runtime, int64_t token,
                        float *logits, size_t logits_count,
                        char *error, size_t error_size) {
    if (!runtime || !runtime->layers || !logits ||
        token < 0 || token >= runtime->model.config.vocab_size ||
        runtime->length >= runtime->capacity ||
        logits_count < (size_t)runtime->model.config.vocab_size)
        return fail(error, error_size, "invalid runtime step");
    Qwen38Config *config = &runtime->model.config;
    const float *embedding = runtime->embedding + token * (int64_t)config->hidden_size;
    for (int stream = 0; stream < config->hc_count; stream++)
        memcpy(runtime->hyper_a + (int64_t)stream * config->hidden_size,
               embedding, (size_t)config->hidden_size * sizeof(float));
    float *input = runtime->hyper_a, *output = runtime->hyper_b;
    for (int layer = 0; layer < config->num_hidden_layers; layer++) {
        int result = runtime->layers[layer].full_attention ?
            qwen38_full_layer_step(&runtime->model, layer,
                &runtime->layers[layer].full, token, input, output,
                runtime->workspace, runtime->workspace_floats,
                error, error_size) :
            qwen38_linear_layer_step(&runtime->model, layer,
                &runtime->layers[layer].linear, token, input, output,
                runtime->workspace, runtime->workspace_floats,
                error, error_size);
        if (result) return -1;
        float *swap = input; input = output; output = swap;
    }
    if (qwen38_mhc_mix(input, config->hc_count, config->hidden_size,
                       config->hc_lowrank, config->rms_norm_eps,
                       runtime->final_norm, runtime->final_mix_down,
                       runtime->final_mix_up, NULL, runtime->hidden, NULL,
                       runtime->workspace, runtime->workspace_floats))
        return fail(error, error_size, "final hyper mixer failed");
    matvec(runtime->lm_head, runtime->hidden, config->vocab_size,
           config->hidden_size, logits);
    runtime->length++;
    return 0;
}
