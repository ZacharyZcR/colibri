#include "qwen38_mhc.h"

#include <math.h>
#include <stdint.h>

static float sigmoidf_stable(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    float exp_value = expf(value);
    return exp_value / (1.0f + exp_value);
}

size_t qwen38_mhc_workspace_floats(int hc_count, int hidden_size, int lowrank) {
    if (hc_count < 1 || hidden_size < 1 || lowrank < 1) return 0;
    size_t width = (size_t)hc_count * (size_t)hidden_size;
    if (width > SIZE_MAX - (size_t)lowrank) return 0;
    return width + (size_t)lowrank;
}

int qwen38_mhc_mix(const float *hyper_input, int hc_count, int hidden_size,
                   int lowrank, float eps, const float *norm_weight,
                   const float *mix_down, const float *mix_up,
                   const float *inject_weight, float *mixed_input,
                   float *injection_weights, float *workspace,
                   size_t workspace_floats) {
    size_t required = qwen38_mhc_workspace_floats(hc_count, hidden_size, lowrank);
    if (!required || !hyper_input || !norm_weight || !mix_down || !mix_up ||
        !mixed_input || !workspace || workspace_floats < required || eps <= 0.0f)
        return -1;
    size_t width = (size_t)hc_count * (size_t)hidden_size;
    float *normalized = workspace;
    float *low = workspace + width;

    for (int stream = 0; stream < hc_count; stream++) {
        size_t base = (size_t)stream * (size_t)hidden_size;
        double squares = 0.0;
        for (int dim = 0; dim < hidden_size; dim++) {
            float value = hyper_input[base + (size_t)dim];
            squares += (double)value * value;
        }
        float scale = 1.0f / sqrtf((float)(squares / hidden_size) + eps);
        for (int dim = 0; dim < hidden_size; dim++) {
            size_t index = base + (size_t)dim;
            normalized[index] = hyper_input[index] * scale * (1.0f + norm_weight[index]);
        }
    }

    for (int row = 0; row < lowrank; row++) {
        const float *weight = mix_down + (size_t)row * width;
        double sum = 0.0;
        for (size_t column = 0; column < width; column++)
            sum += (double)weight[column] * normalized[column];
        float value = (float)sum / hc_count;
        low[row] = value / (1.0f + expf(-value));
    }

    for (int dim = 0; dim < hidden_size; dim++) mixed_input[dim] = 0.0f;
    for (int stream = 0; stream < hc_count; stream++) {
        size_t base = (size_t)stream * (size_t)hidden_size;
        for (int dim = 0; dim < hidden_size; dim++) {
            size_t row = base + (size_t)dim;
            const float *weight = mix_up + row * (size_t)lowrank;
            double sum = 0.0;
            for (int column = 0; column < lowrank; column++)
                sum += (double)weight[column] * low[column];
            mixed_input[dim] += sigmoidf_stable((float)sum) * normalized[row] / hc_count;
        }
    }

    if (injection_weights) {
        if (!inject_weight) return -1;
        for (int stream = 0; stream < hc_count; stream++) {
            const float *weight = inject_weight + (size_t)stream * width;
            double sum = 0.0;
            for (size_t column = 0; column < width; column++)
                sum += (double)weight[column] * normalized[column];
            injection_weights[stream] = 2.0f * sigmoidf_stable((float)sum / hc_count);
        }
    }
    return 0;
}
