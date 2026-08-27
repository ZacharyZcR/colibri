#ifndef QWEN38_ACCEL_H
#define QWEN38_ACCEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef COLI_CUDA
#include "backend_cuda.h"
#endif
#ifdef COLI_METAL
#include "backend_metal.h"
#endif

typedef struct {
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;
#endif
#ifdef COLI_METAL
    ColiMetalTensor *metal;
#endif
    int unused;
} Qwen38AccelTensor;

int qwen38_accel_init(char *error, size_t error_size);
void qwen38_accel_shutdown(void);
void qwen38_accel_tensor_close(Qwen38AccelTensor *tensor);

/* Returns 1 when an accelerator executed the matvec, 0 for CPU fallback. */
int qwen38_accel_matvec(Qwen38AccelTensor *tensor, float *output,
                        const int8_t *weights, const float *scales,
                        const float *input, int rows, int columns, int group);

/* CUDA/HIP fused path; Metal and unsupported layouts return 0. */
int qwen38_accel_expert(Qwen38AccelTensor tensors[3], float *output,
                        const int8_t *weights, const float *scales,
                        const float *input, int hidden, int intermediate,
                        int group);

#endif
