#ifndef COLIBRI_QWEN38_MHC_H
#define COLIBRI_QWEN38_MHC_H

#include <stddef.h>

size_t qwen38_mhc_workspace_floats(int hc_count, int hidden_size, int lowrank);

int qwen38_mhc_mix(const float *hyper_input, int hc_count, int hidden_size,
                   int lowrank, float eps, const float *norm_weight,
                   const float *mix_down, const float *mix_up,
                   const float *inject_weight, float *mixed_input,
                   float *injection_weights, float *workspace,
                   size_t workspace_floats);

#endif
