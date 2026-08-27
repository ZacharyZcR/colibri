#include "qwen38_accel.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

#ifdef COLI_CUDA
static int cuda_enabled, cuda_device;

static int init_cuda(char *error, size_t error_size) {
    const char *enabled = getenv("COLI_CUDA");
    if (!enabled || !atoi(enabled)) return 0;
    if (getenv("COLI_GPU") && getenv("COLI_GPUS"))
        return fail(error, error_size, "qwen38: use COLI_GPU or COLI_GPUS, not both");
    const char *selected = getenv("COLI_GPU");
    if (!selected) selected = getenv("COLI_GPUS");
    char *end = NULL;
    errno = 0;
    long parsed = selected ? strtol(selected, &end, 10) : 0;
    if (errno || parsed < 0 || parsed > INT_MAX ||
        (selected && (!end || *end)))
        return fail(error, error_size,
                    "qwen38: COLI_GPU/COLI_GPUS must select exactly one device");
    cuda_device = (int)parsed;
    if (!coli_cuda_init(&cuda_device, 1))
        return fail(error, error_size, "qwen38: CUDA/HIP initialization failed");
    cuda_enabled = 1;
    fprintf(stderr, "[GPU] Qwen3.8 routed expert matvec active on device %d\n",
            cuda_device);
    return 0;
}
#else
static int init_cuda(char *error, size_t error_size) {
    const char *enabled = getenv("COLI_CUDA");
    if (enabled && atoi(enabled))
        return fail(error, error_size,
                    "qwen38: COLI_CUDA requested but this binary has no CUDA/HIP backend");
    return 0;
}
#endif

#ifdef COLI_METAL
static int metal_enabled;

static int init_metal(char *error, size_t error_size) {
    const char *enabled = getenv("COLI_METAL");
    if (!enabled || !atoi(enabled)) return 0;
    if (!coli_metal_init())
        return fail(error, error_size, "qwen38: Metal initialization failed");
    metal_enabled = 1;
    fprintf(stderr, "[Metal] Qwen3.8 routed expert matvec active\n");
    return 0;
}
#else
static int init_metal(char *error, size_t error_size) {
    const char *enabled = getenv("COLI_METAL");
    if (enabled && atoi(enabled))
        return fail(error, error_size,
                    "qwen38: COLI_METAL requested but this binary has no Metal backend");
    return 0;
}
#endif

int qwen38_accel_init(char *error, size_t error_size) {
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA")) &&
        getenv("COLI_METAL") && atoi(getenv("COLI_METAL")))
        return fail(error, error_size, "qwen38: choose COLI_CUDA or COLI_METAL, not both");
    if (init_cuda(error, error_size)) return -1;
    if (init_metal(error, error_size)) {
#ifdef COLI_CUDA
        if (cuda_enabled) coli_cuda_shutdown();
        cuda_enabled = 0;
#endif
        return -1;
    }
    return 0;
}

void qwen38_accel_shutdown(void) {
#ifdef COLI_CUDA
    if (cuda_enabled) coli_cuda_shutdown();
    cuda_enabled = 0;
#endif
#ifdef COLI_METAL
    if (metal_enabled) coli_metal_shutdown();
    metal_enabled = 0;
#endif
}

void qwen38_accel_tensor_close(Qwen38AccelTensor *tensor) {
    if (!tensor) return;
#ifdef COLI_CUDA
    if (tensor->cuda) coli_cuda_tensor_free(tensor->cuda);
#endif
#ifdef COLI_METAL
    if (tensor->metal) coli_metal_tensor_free(tensor->metal);
#endif
    memset(tensor, 0, sizeof(*tensor));
}

int qwen38_accel_matvec(Qwen38AccelTensor *tensor, float *output,
                        const int8_t *weights, const float *scales,
                        const float *input, int rows, int columns, int group) {
    if (!tensor || !output || !weights || !scales || !input ||
        rows < 1 || columns < 1 || group) return 0;
#ifdef COLI_CUDA
    if (cuda_enabled &&
        coli_cuda_matmul(&tensor->cuda, output, input, weights, scales, 1,
                         1, columns, rows, cuda_device, 0)) return 1;
#endif
#ifdef COLI_METAL
    if (metal_enabled &&
        coli_metal_matmul(&tensor->metal, output, input, weights, scales, 1,
                          1, columns, rows, 0)) return 1;
#endif
    return 0;
}

int qwen38_accel_expert(Qwen38AccelTensor tensors[3], float *output,
                        const int8_t *weights, const float *scales,
                        const float *input, int hidden, int intermediate,
                        int group) {
    if (!tensors || !output || !weights || !scales || !input ||
        hidden < 1 || intermediate < 1 || group) return 0;
#ifdef COLI_CUDA
    if (cuda_enabled) {
        size_t matrix = (size_t)hidden * intermediate;
        if (!tensors[0].cuda &&
            !coli_cuda_tensor_upload(&tensors[0].cuda, weights, scales, 1,
                                     hidden, intermediate, cuda_device)) return 0;
        if (!tensors[1].cuda &&
            !coli_cuda_tensor_upload(&tensors[1].cuda, weights + matrix,
                                     scales + intermediate, 1, hidden,
                                     intermediate, cuda_device)) return 0;
        if (!tensors[2].cuda &&
            !coli_cuda_tensor_upload(&tensors[2].cuda, weights + 2 * matrix,
                                     scales + 2 * intermediate, 1, intermediate,
                                     hidden, cuda_device)) return 0;
        return coli_cuda_expert_mlp(tensors[0].cuda, tensors[1].cuda,
                                    tensors[2].cuda, output, input, 1);
    }
#endif
    return 0;
}
