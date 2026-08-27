#include "qwen38_delta.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float sigmoid_stable(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    float exponential = expf(value);
    return exponential / (1.0f + exponential);
}

static float softplus(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
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

static int geometry(const Qwen38Config *config, int *conv_dim, int *value_dim) {
    if (!config || config->linear_key_heads < 1 || config->linear_key_dim < 1 ||
        config->linear_value_heads < 1 || config->linear_value_dim < 1 ||
        config->linear_value_heads % config->linear_key_heads ||
        config->linear_conv_kernel < 2 || config->hidden_size < 1) return -1;
    *value_dim = config->linear_value_heads * config->linear_value_dim;
    *conv_dim = 2 * config->linear_key_heads * config->linear_key_dim + *value_dim;
    return 0;
}

size_t qwen38_delta_workspace_floats(const Qwen38Config *config) {
    int conv_dim, value_dim;
    if (geometry(config, &conv_dim, &value_dim)) return 0;
    int repeated_keys = config->linear_value_heads * config->linear_key_dim;
    return 2ULL * conv_dim + 4ULL * value_dim + 2ULL * repeated_keys +
           4ULL * config->linear_value_heads;
}

int qwen38_delta_state_init(Qwen38DeltaState *state,
                            const Qwen38Config *config) {
    int conv_dim, value_dim;
    if (!state || geometry(config, &conv_dim, &value_dim)) return -1;
    memset(state, 0, sizeof(*state));
    size_t recurrent = (size_t)config->linear_value_heads *
                       config->linear_key_dim * config->linear_value_dim;
    size_t convolution = (size_t)conv_dim * (config->linear_conv_kernel - 1);
    state->recurrent = calloc(recurrent, sizeof(float));
    state->convolution = calloc(convolution, sizeof(float));
    if (!state->recurrent || !state->convolution) {
        qwen38_delta_state_close(state);
        return -1;
    }
    return 0;
}

void qwen38_delta_state_close(Qwen38DeltaState *state) {
    if (!state) return;
    free(state->recurrent);
    free(state->convolution);
    memset(state, 0, sizeof(*state));
}

int qwen38_delta_step(const Qwen38Config *config,
                      const Qwen38DeltaWeights *weights,
                      Qwen38DeltaState *state, const float *input,
                      float *output, float *workspace,
                      size_t workspace_floats) {
    int conv_dim, value_dim;
    size_t required = qwen38_delta_workspace_floats(config);
    if (!required || !weights || !weights->qkv || !weights->z || !weights->b ||
        !weights->a || !weights->conv || !weights->a_log || !weights->dt_bias ||
        !weights->norm || !weights->out || !state || !state->recurrent ||
        !state->convolution || !input || !output || !workspace ||
        workspace_floats < required || geometry(config, &conv_dim, &value_dim)) return -1;
    int value_heads = config->linear_value_heads;
    int key_heads = config->linear_key_heads;
    int key_dim = config->linear_key_dim;
    int value_dim_head = config->linear_value_dim;
    int repeat = value_heads / key_heads;
    int key_total = key_heads * key_dim;
    int repeated_keys = value_heads * key_dim;
    int kernel = config->linear_conv_kernel;
    float *cursor = workspace;
    float *qkv = cursor; cursor += conv_dim;
    float *conv_output = cursor; cursor += conv_dim;
    float *z = cursor; cursor += value_dim;
    float *core = cursor; cursor += value_dim;
    float *normalized_output = cursor; cursor += value_dim;
    float *projected = cursor; cursor += value_dim;
    float *query = cursor; cursor += repeated_keys;
    float *key = cursor; cursor += repeated_keys;
    float *b = cursor; cursor += value_heads;
    float *a = cursor; cursor += value_heads;
    float *beta = cursor; cursor += value_heads;
    float *decay = cursor;

    matvec(weights->qkv, input, conv_dim, config->hidden_size, qkv);
    matvec(weights->z, input, value_dim, config->hidden_size, z);
    matvec(weights->b, input, value_heads, config->hidden_size, b);
    matvec(weights->a, input, value_heads, config->hidden_size, a);
    for (int head = 0; head < value_heads; head++) {
        beta[head] = sigmoid_stable(b[head]);
        decay[head] = -expf(weights->a_log[head]) *
                      softplus(a[head] + weights->dt_bias[head]);
    }
    for (int channel = 0; channel < conv_dim; channel++) {
        const float *conv_weight = weights->conv + (int64_t)channel * kernel;
        float *ring = state->convolution + (int64_t)channel * (kernel - 1);
        float sum = conv_weight[kernel - 1] * qkv[channel];
        for (int tap = 0; tap < kernel - 1; tap++) sum += conv_weight[tap] * ring[tap];
        conv_output[channel] = sum * sigmoid_stable(sum);
        memmove(ring, ring + 1, (size_t)(kernel - 2) * sizeof(float));
        ring[kernel - 2] = qkv[channel];
    }
    const float *query_source = conv_output;
    const float *key_source = conv_output + key_total;
    const float *value_source = conv_output + 2 * key_total;
    float query_scale = 1.0f / sqrtf((float)key_dim);
    for (int head = 0; head < value_heads; head++) {
        int source_head = head / repeat;
        float *query_head = query + (int64_t)head * key_dim;
        float *key_head = key + (int64_t)head * key_dim;
        memcpy(query_head, query_source + (int64_t)source_head * key_dim,
               (size_t)key_dim * sizeof(float));
        memcpy(key_head, key_source + (int64_t)source_head * key_dim,
               (size_t)key_dim * sizeof(float));
        double query_squares = 1e-6, key_squares = 1e-6;
        for (int dim = 0; dim < key_dim; dim++) {
            query_squares += (double)query_head[dim] * query_head[dim];
            key_squares += (double)key_head[dim] * key_head[dim];
        }
        float query_norm = query_scale / sqrtf((float)query_squares);
        float key_norm = 1.0f / sqrtf((float)key_squares);
        for (int dim = 0; dim < key_dim; dim++) {
            query_head[dim] *= query_norm;
            key_head[dim] *= key_norm;
        }
    }
    for (int head = 0; head < value_heads; head++) {
        float *recurrent = state->recurrent +
            (int64_t)head * key_dim * value_dim_head;
        const float *query_head = query + (int64_t)head * key_dim;
        const float *key_head = key + (int64_t)head * key_dim;
        const float *value_head = value_source + (int64_t)head * value_dim_head;
        float *core_head = core + (int64_t)head * value_dim_head;
        float factor = expf(decay[head]);
        for (int index = 0; index < key_dim * value_dim_head; index++)
            recurrent[index] *= factor;
        for (int value = 0; value < value_dim_head; value++) {
            float previous = 0.0f;
            for (int key_index = 0; key_index < key_dim; key_index++)
                previous += key_head[key_index] *
                    recurrent[(int64_t)key_index * value_dim_head + value];
            float delta = (value_head[value] - previous) * beta[head];
            for (int key_index = 0; key_index < key_dim; key_index++)
                recurrent[(int64_t)key_index * value_dim_head + value] +=
                    key_head[key_index] * delta;
            core_head[value] = 0.0f;
            for (int key_index = 0; key_index < key_dim; key_index++)
                core_head[value] += query_head[key_index] *
                    recurrent[(int64_t)key_index * value_dim_head + value];
        }
        double squares = 0.0;
        for (int value = 0; value < value_dim_head; value++)
            squares += (double)core_head[value] * core_head[value];
        float norm = 1.0f / sqrtf((float)(squares / value_dim_head) +
                                  config->rms_norm_eps);
        for (int value = 0; value < value_dim_head; value++) {
            int index = head * value_dim_head + value;
            normalized_output[index] = core_head[value] * norm * weights->norm[value] *
                                       sigmoid_stable(z[index]);
        }
    }
    memcpy(projected, normalized_output, (size_t)value_dim * sizeof(float));
    matvec(weights->out, projected, config->hidden_size, value_dim, output);
    return 0;
}
