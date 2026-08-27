#include "qwen38_attention.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int valid(const Qwen38Config *config) {
    return config && config->hidden_size > 0 && config->attention_heads > 0 &&
           config->kv_heads > 0 && config->attention_heads % config->kv_heads == 0 &&
           config->head_dim > 0 && config->indexer_heads > 0 &&
           config->indexer_kv_heads == 1 && config->indexer_head_dim > 0 &&
           config->indexer_compress_ratio > 0 &&
           config->indexer_budget >= config->indexer_compress_ratio &&
           config->partial_rotary_factor > 0.0f && config->rope_theta > 0.0f;
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

static void rms_norm(float *values, int heads, int width, const float *weight,
                     float epsilon) {
    for (int head = 0; head < heads; head++) {
        float *row = values + (int64_t)head * width;
        double squares = 0.0;
        for (int index = 0; index < width; index++) squares += (double)row[index] * row[index];
        float scale = 1.0f / sqrtf((float)(squares / width) + epsilon);
        for (int index = 0; index < width; index++)
            row[index] *= scale * (1.0f + weight[index]);
    }
}

static void rope(float *values, int heads, int width, int rotary_dim,
                 size_t position, float theta) {
    int half = rotary_dim / 2;
    for (int head = 0; head < heads; head++) {
        float *row = values + (int64_t)head * width;
        for (int index = 0; index < half; index++) {
            float angle = (float)position /
                powf(theta, (float)(2 * index) / (float)rotary_dim);
            float cosine = cosf(angle), sine = sinf(angle);
            float first = row[index], second = row[index + half];
            row[index] = first * cosine - second * sine;
            row[index + half] = second * cosine + first * sine;
        }
    }
}

int qwen38_attention_state_init(Qwen38AttentionState *state,
                                const Qwen38Config *config,
                                size_t capacity) {
    if (!state || !valid(config) || !capacity) return -1;
    memset(state, 0, sizeof(*state));
    size_t kv_width = (size_t)config->kv_heads * config->head_dim;
    state->keys = calloc(capacity * kv_width, sizeof(float));
    state->values = calloc(capacity * kv_width, sizeof(float));
    state->raw_index_keys = calloc(capacity * (size_t)config->indexer_head_dim,
                                   sizeof(float));
    state->scores = malloc(capacity * sizeof(float));
    state->selected = malloc(capacity);
    state->pooled_key = malloc((size_t)config->indexer_head_dim * sizeof(float));
    state->heap_capacity = (size_t)config->indexer_budget /
                           config->indexer_compress_ratio;
    state->heap = malloc(state->heap_capacity * sizeof(*state->heap));
    if (!state->keys || !state->values || !state->raw_index_keys ||
        !state->scores || !state->selected || !state->pooled_key || !state->heap) {
        qwen38_attention_state_close(state);
        return -1;
    }
    state->capacity = capacity;
    return 0;
}

void qwen38_attention_state_close(Qwen38AttentionState *state) {
    if (!state) return;
    free(state->keys); free(state->values); free(state->raw_index_keys);
    free(state->scores); free(state->selected); free(state->pooled_key);
    free(state->heap);
    memset(state, 0, sizeof(*state));
}

size_t qwen38_attention_workspace_floats(const Qwen38Config *config) {
    if (!valid(config)) return 0;
    size_t query = (size_t)config->attention_heads * config->head_dim;
    size_t kv = (size_t)config->kv_heads * config->head_dim;
    size_t index_query = (size_t)config->indexer_heads * config->indexer_head_dim;
    return 6ULL * query + 2ULL * kv + 2ULL * index_query +
           config->indexer_head_dim;
}

int qwen38_attention_step(const Qwen38Config *config,
                          const Qwen38AttentionWeights *weights,
                          Qwen38AttentionState *state, const float *input,
                          float *output, float *workspace,
                          size_t workspace_floats) {
    size_t required = qwen38_attention_workspace_floats(config);
    if (!required || !weights || !weights->q_proj || !weights->k_proj ||
        !weights->v_proj || !weights->o_proj || !weights->q_norm ||
        !weights->k_norm || !weights->index_qk_proj || !weights->index_q_norm ||
        !weights->index_k_norm || !state || state->length >= state->capacity ||
        !input || !output || !workspace || workspace_floats < required) return -1;
    int hidden = config->hidden_size, heads = config->attention_heads;
    int kv_heads = config->kv_heads, head_dim = config->head_dim;
    int query_width = heads * head_dim, kv_width = kv_heads * head_dim;
    int index_width = config->indexer_head_dim;
    int index_query_width = config->indexer_heads * index_width;
    int index_total = index_query_width + index_width;
    int rotary_dim = (int)(head_dim * config->partial_rotary_factor + 0.5f);
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -1;
    float *cursor = workspace;
    float *q_projection = cursor; cursor += 2 * query_width;
    float *query = cursor; cursor += query_width;
    float *gate = cursor; cursor += query_width;
    float *attention = cursor; cursor += query_width;
    float *projected = cursor; cursor += query_width;
    float *key = cursor; cursor += kv_width;
    float *value = cursor; cursor += kv_width;
    float *index_projection = cursor; cursor += index_total;
    float *index_query = cursor;

    matvec(weights->q_proj, input, 2 * query_width, hidden, q_projection);
    for (int head = 0; head < heads; head++) {
        memcpy(query + (int64_t)head * head_dim,
               q_projection + (int64_t)head * 2 * head_dim,
               (size_t)head_dim * sizeof(float));
        memcpy(gate + (int64_t)head * head_dim,
               q_projection + (int64_t)head * 2 * head_dim + head_dim,
               (size_t)head_dim * sizeof(float));
    }
    matvec(weights->k_proj, input, kv_width, hidden, key);
    matvec(weights->v_proj, input, kv_width, hidden, value);
    matvec(weights->index_qk_proj, input, index_total, hidden, index_projection);
    memcpy(index_query, index_projection, (size_t)index_query_width * sizeof(float));
    float *raw_key = state->raw_index_keys + state->length * (size_t)index_width;
    memcpy(raw_key, index_projection + index_query_width,
           (size_t)index_width * sizeof(float));

    rms_norm(query, heads, head_dim, weights->q_norm, config->rms_norm_eps);
    rms_norm(key, kv_heads, head_dim, weights->k_norm, config->rms_norm_eps);
    rms_norm(index_query, config->indexer_heads, index_width,
             weights->index_q_norm, config->rms_norm_eps);
    rope(query, heads, head_dim, rotary_dim, state->length, config->rope_theta);
    rope(key, kv_heads, head_dim, rotary_dim, state->length, config->rope_theta);
    rope(index_query, config->indexer_heads, index_width,
         rotary_dim < index_width ? rotary_dim : index_width,
         state->length, config->rope_theta);
    memcpy(state->keys + state->length * (size_t)kv_width, key,
           (size_t)kv_width * sizeof(float));
    memcpy(state->values + state->length * (size_t)kv_width, value,
           (size_t)kv_width * sizeof(float));
    size_t token_count = ++state->length;
    if (qwen38_qsa_select_text(index_query, config->indexer_heads,
                               state->raw_index_keys, token_count, index_width,
                               weights->index_k_norm, config->rms_norm_eps,
                               rotary_dim < index_width ? rotary_dim : index_width,
                               config->rope_theta, config->indexer_compress_ratio,
                               config->indexer_budget, state->selected,
                               state->pooled_key, state->heap,
                               state->heap_capacity) < 0) return -1;

    int groups = heads / kv_heads;
    float score_scale = 1.0f / sqrtf((float)head_dim);
    for (int head = 0; head < heads; head++) {
        const float *query_head = query + (int64_t)head * head_dim;
        int kv_head = head / groups;
        float maximum = -INFINITY;
        for (size_t token = 0; token < token_count; token++) {
            if (!state->selected[token]) continue;
            const float *key_head = state->keys + token * (size_t)kv_width +
                                    (int64_t)kv_head * head_dim;
            double dot = 0.0;
            for (int dim = 0; dim < head_dim; dim++) dot += query_head[dim] * key_head[dim];
            state->scores[token] = (float)dot * score_scale;
            if (state->scores[token] > maximum) maximum = state->scores[token];
        }
        float total = 0.0f;
        for (size_t token = 0; token < token_count; token++) if (state->selected[token]) {
            state->scores[token] = expf(state->scores[token] - maximum);
            total += state->scores[token];
        }
        if (!(total > 0.0f) || !isfinite(total)) return -1;
        float *attention_head = attention + (int64_t)head * head_dim;
        memset(attention_head, 0, (size_t)head_dim * sizeof(float));
        for (size_t token = 0; token < token_count; token++) if (state->selected[token]) {
            const float *value_head = state->values + token * (size_t)kv_width +
                                      (int64_t)kv_head * head_dim;
            float probability = state->scores[token] / total;
            for (int dim = 0; dim < head_dim; dim++)
                attention_head[dim] += probability * value_head[dim];
        }
    }
    for (int index = 0; index < query_width; index++)
        projected[index] = attention[index] / (1.0f + expf(-gate[index]));
    matvec(weights->o_proj, projected, hidden, query_width, output);
    return 0;
}
