#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_ple_layer.h"

#include <math.h>
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
    fail(error, error_size, "invalid PLE tensor %s", name);
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

static void group_norm(const float *input, const float *weight,
                       int groups, int width, float epsilon, float *output) {
    for (int group = 0; group < groups; group++) {
        const float *source = input + (int64_t)group * width;
        float *destination = output + (int64_t)group * width;
        double squares = 0.0;
        for (int index = 0; index < width; index++)
            squares += (double)source[index] * source[index];
        float scale = 1.0f / sqrtf((float)(squares / width) + epsilon);
        for (int index = 0; index < width; index++)
            destination[index] = source[index] * scale *
                                 (1.0f + weight[(int64_t)group * width + index]);
    }
}

void qwen38_ple_layer_close(Qwen38PleLayer *weights) {
    if (!weights) return;
    free(weights->key_proj); free(weights->value_proj);
    free(weights->norm_key); free(weights->norm_query); free(weights->norm_conv);
    free(weights->conv); free(weights->conv_state);
    memset(weights, 0, sizeof(*weights));
}

int qwen38_ple_layer_load(Qwen38Model *model, int layer,
                          Qwen38PleLayer *weights,
                          char *error, size_t error_size) {
    if (!model || !weights || layer < 0 || layer >= model->config.num_hidden_layers)
        return fail(error, error_size, "invalid PLE layer");
    memset(weights, 0, sizeof(*weights));
    weights->table_index = -1;
    Qwen38Config *config = &model->config;
    for (int index = 0; index < config->ple_layer_count; index++)
        if (config->ple_layers[index] == layer + 1) weights->table_index = index;
    if (weights->table_index < 0) return fail(error, error_size, "layer %d has no PLE", layer);
    if (qwen38_ple_hash_init(&weights->hash, config->vocab_size,
                             config->ngram_size, config->heads_per_ngram,
                             config->ngram_vocab_base, weights->table_index,
                             config->seed))
        return fail(error, error_size, "cannot initialize PLE hash");
    int64_t hidden = config->hidden_size;
    int64_t hyper = (int64_t)config->hc_count * hidden;
    int64_t embed = config->ple_embed_dim;
    int64_t shapes[6][3] = {
        {hyper, embed}, {hidden, embed}, {hyper}, {hyper}, {hyper},
        {hyper, 1, config->ple_conv_kernel},
    };
    const int ranks[6] = {2, 2, 1, 1, 1, 3};
    const char *suffixes[6] = {
        "key_proj.weight", "value_proj.weight", "norm_key.weight",
        "norm_query.weight", "norm_conv.weight", "conv1d.weight",
    };
    float **destinations[6] = {
        &weights->key_proj, &weights->value_proj, &weights->norm_key,
        &weights->norm_query, &weights->norm_conv, &weights->conv,
    };
    char name[512];
    for (int index = 0; index < 6; index++) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.ple.%s",
                 layer, suffixes[index]);
        *destinations[index] = load(model, name, ranks[index], shapes[index],
                                    error, error_size);
        if (!*destinations[index]) goto fail;
    }
    int state_length = (config->ple_conv_kernel - 1) * config->ngram_size;
    weights->conv_state = calloc((size_t)hyper * state_length, sizeof(float));
    if (!weights->conv_state) {
        fail(error, error_size, "out of memory allocating PLE convolution state");
        goto fail;
    }
    for (int index = 0; index < config->ngram_size - 1; index++)
        weights->history[index] = config->eos_token_id;
    return 0;
fail:
    qwen38_ple_layer_close(weights);
    return -1;
}

size_t qwen38_ple_layer_workspace_floats(const Qwen38Config *config) {
    if (!config) return 0;
    size_t hyper = (size_t)config->hc_count * config->hidden_size;
    return config->ple_embed_dim + 4ULL * hyper + config->hidden_size;
}

int qwen38_ple_layer_step(Qwen38Model *model, Qwen38PleLayer *weights,
                          int64_t token, const float *hyper_input,
                          float *hyper_output, float *workspace,
                          size_t workspace_floats,
                          char *error, size_t error_size) {
    if (!model || !weights || weights->table_index < 0 || !hyper_input ||
        !hyper_output || !workspace ||
        workspace_floats < qwen38_ple_layer_workspace_floats(&model->config))
        return fail(error, error_size, "invalid PLE step arguments");
    Qwen38Config *config = &model->config;
    int hidden = config->hidden_size, hc = config->hc_count;
    int hyper = hc * hidden;
    int context = config->ngram_size - 1;
    int ngram_heads = context * config->heads_per_ngram;
    int64_t tokens[QWEN38_PLE_MAX_NGRAM];
    for (int index = 0; index < context; index++) tokens[index] = weights->history[index];
    tokens[context] = token;
    uint64_t rows[QWEN38_PLE_MAX_HEADS];
    if (qwen38_ple_hash_token(&weights->hash, tokens, config->ngram_size,
                              context, config->eos_token_id, rows,
                              QWEN38_PLE_MAX_HEADS) != ngram_heads)
        return fail(error, error_size, "PLE hash failed");
    float *cursor = workspace;
    float *embedding = cursor; cursor += config->ple_embed_dim;
    float *key = cursor; cursor += hyper;
    float *query = cursor; cursor += hyper;
    float *gated = cursor; cursor += hyper;
    float *normalized = cursor; cursor += hyper;
    float *value = cursor;
    if (qwen38_ple_table_lookup(&model->ple[weights->table_index], rows,
                                (size_t)ngram_heads, embedding,
                                config->ple_embed_dim))
        return fail(error, error_size, "PLE embedding lookup failed");
    matvec(weights->key_proj, embedding, hyper, config->ple_embed_dim, key);
    matvec(weights->value_proj, embedding, hidden, config->ple_embed_dim, value);
    group_norm(key, weights->norm_key, hc, hidden, config->rms_norm_eps, key);
    group_norm(hyper_input, weights->norm_query, hc, hidden,
               config->rms_norm_eps, query);
    for (int stream = 0; stream < hc; stream++) {
        double product = 0.0;
        for (int index = 0; index < hidden; index++)
            product += key[(int64_t)stream * hidden + index] *
                       query[(int64_t)stream * hidden + index];
        float gate = (float)(product / sqrt((double)hidden));
        if (gate != 0.0f)
            gate = copysignf(sqrtf(fmaxf(fabsf(gate), 1e-6f)), gate);
        float scale = 1.0f / (1.0f + expf(-gate));
        for (int index = 0; index < hidden; index++)
            gated[(int64_t)stream * hidden + index] = scale * value[index];
    }
    group_norm(gated, weights->norm_conv, hc, hidden,
               config->rms_norm_eps, normalized);
    int dilation = config->ngram_size;
    int kernel = config->ple_conv_kernel;
    int state_length = (kernel - 1) * dilation;
    for (int channel = 0; channel < hyper; channel++) {
        float *state = weights->conv_state + (int64_t)channel * state_length;
        const float *conv = weights->conv + (int64_t)channel * kernel;
        float sum = conv[kernel - 1] * normalized[channel];
        for (int tap = 0; tap < kernel - 1; tap++) sum += conv[tap] * state[tap * dilation];
        float convolution = sum / (1.0f + expf(-sum));
        hyper_output[channel] = hyper_input[channel] + gated[channel] + convolution;
        memmove(state, state + 1, (size_t)(state_length - 1) * sizeof(float));
        state[state_length - 1] = normalized[channel];
    }
    memmove(weights->history, weights->history + 1,
            (size_t)(context - 1) * sizeof(int64_t));
    weights->history[context - 1] = token;
    return 0;
}
