#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_expert.h"

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

static int64_t scale_count(const Qwen38Config *config, int group) {
    if (!group) return 2LL * config->moe_intermediate_size + config->hidden_size;
    return 2LL * config->moe_intermediate_size *
               ((config->hidden_size + group - 1) / group) +
           (int64_t)config->hidden_size *
               ((config->moe_intermediate_size + group - 1) / group);
}

void qwen38_expert_close(Qwen38Expert *expert) {
    if (!expert) return;
    for (int index = 0; index < 3; index++)
        qwen38_accel_tensor_close(&expert->matrices[index]);
    free(expert->weights);
    free(expert->scales);
    memset(expert, 0, sizeof(*expert));
}

int qwen38_expert_load(Qwen38Model *model, int row, int expert,
                       Qwen38Expert *output, char *error, size_t error_size) {
    if (!model || !output || row < 0 ||
        row >= model->config.num_hidden_layers + model->config.mtp_layers ||
        expert < 0 || expert >= model->config.num_experts)
        return fail(error, error_size, "invalid expert coordinates");
    memset(output, 0, sizeof(*output));
    Qwen38Config *config = &model->config;
    int64_t weight_count = 3LL * config->hidden_size * config->moe_intermediate_size;
    int64_t packed_count = model->expert_bits <= 4 ? (weight_count + 1) / 2 : weight_count;
    int64_t scales = scale_count(config, model->expert_group_size);
    output->weights = malloc((size_t)weight_count);
    output->scales = malloc((size_t)scales * sizeof(float));
    unsigned char *packed = model->expert_bits <= 4 ? malloc((size_t)packed_count) : NULL;
    if (!output->weights || !output->scales ||
        (model->expert_bits <= 4 && !packed)) {
        free(packed);
        qwen38_expert_close(output);
        return fail(error, error_size, "out of memory loading expert");
    }
    char weight_name[256], scale_name[256];
    snprintf(weight_name, sizeof(weight_name),
             "model.layers.%d.mlp.experts.%d.merged_weight", row, expert);
    snprintf(scale_name, sizeof(scale_name),
             "model.layers.%d.mlp.experts.%d.qs", row, expert);
    if (model->expert_bits <= 4) {
        st_read_raw(&model->experts, weight_name, packed, 1);
        for (int64_t index = 0; index < weight_count; index++) {
            int8_t value = (packed[index >> 1] >> ((index & 1) * 4)) & 15;
            output->weights[index] = value & 8 ? value - 16 : value;
        }
        free(packed);
    } else {
        st_read_raw(&model->experts, weight_name, output->weights, 1);
    }
    st_read_f32(&model->experts, scale_name, output->scales, 0);
    return 0;
}

static void matvec_q(Qwen38AccelTensor *accel, const int8_t *weights,
                     const float *scales,
                     int rows, int columns, int group, const float *input,
                     float *output) {
    if (qwen38_accel_matvec(accel, output, weights, scales, input,
                            rows, columns, group)) return;
    int groups = group ? (columns + group - 1) / group : 1;
    int width = group ? group : columns;
    for (int row = 0; row < rows; row++) {
        float sum = 0.0f;
        for (int column = 0; column < columns; column++)
            sum += weights[(int64_t)row * columns + column] * input[column] *
                   scales[(int64_t)row * groups + column / width];
        output[row] = sum;
    }
}

int qwen38_expert_forward(Qwen38Expert *expert,
                          const Qwen38Config *config, const float *input,
                          float *output, float *workspace,
                          size_t workspace_floats) {
    if (!expert || !expert->weights || !expert->scales || !config || !input ||
        !output || !workspace ||
        workspace_floats < (size_t)config->moe_intermediate_size * 2) return -1;
    int hidden = config->hidden_size, intermediate = config->moe_intermediate_size;
    int group = config->expert_group_size;
    int hidden_groups = group ? (hidden + group - 1) / group : 1;
    int64_t matrix = (int64_t)hidden * intermediate;
    int64_t first_scales = (int64_t)intermediate * hidden_groups;
    float *gate = workspace, *up = workspace + intermediate;
    if (qwen38_accel_expert(expert->matrices, output, expert->weights,
                            expert->scales, input, hidden, intermediate,
                            group)) return 0;
    matvec_q(&expert->matrices[0], expert->weights, expert->scales,
             intermediate, hidden,
             group, input, gate);
    matvec_q(&expert->matrices[1], expert->weights + matrix,
             expert->scales + first_scales,
             intermediate, hidden, group, input, up);
    for (int index = 0; index < intermediate; index++)
        gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
    matvec_q(&expert->matrices[2], expert->weights + 2 * matrix,
             expert->scales + 2 * first_scales, hidden, intermediate,
             group, gate, output);
    return 0;
}

int qwen38_moe_forward(Qwen38Model *model, int row, const float *input,
                       const int *expert_ids, const float *router_weights,
                       int selected, float *output, float *workspace,
                       size_t workspace_floats, char *error, size_t error_size) {
    if (!model || !input || !expert_ids || !router_weights || !output ||
        !workspace || selected < 1 || selected > model->config.experts_per_token ||
        workspace_floats < (size_t)model->config.moe_intermediate_size * 2 +
                           model->config.hidden_size)
        return fail(error, error_size, "invalid MoE forward arguments");
    int hidden = model->config.hidden_size;
    memset(output, 0, (size_t)hidden * sizeof(float));
    float *expert_output = workspace + 2 * model->config.moe_intermediate_size;
    for (int index = 0; index < selected; index++) {
        if (!qwen38_accel_cached_expert(row, expert_ids[index], expert_output,
                                       input)) {
            Qwen38Expert expert;
            if (qwen38_expert_load(model, row, expert_ids[index], &expert,
                                   error, error_size)) return -1;
            int result = qwen38_expert_forward(&expert, &model->config, input,
                                               expert_output, workspace,
                                               2 * model->config.moe_intermediate_size);
            if (!result)
                qwen38_accel_cache_store(row, expert_ids[index],
                                         expert.matrices);
            qwen38_expert_close(&expert);
            if (result)
                return fail(error, error_size, "expert forward failed");
        }
        for (int column = 0; column < hidden; column++)
            output[column] += router_weights[index] * expert_output[column];
    }
    return 0;
}
