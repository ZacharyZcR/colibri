#define _GNU_SOURCE
#define COLI_CUDA
#include "../qwen38_accel.c"

#include <math.h>
#include <stdint.h>

struct ColiCudaTensor { int marker; };
static struct ColiCudaTensor handle, uploaded[3];
static int calls, uploads, fused, shutdowns;

int coli_cuda_init(const int *devices, int count) {
    return devices && count == 1 && devices[0] == 3;
}
void coli_cuda_shutdown(void) { shutdowns++; }
void coli_cuda_tensor_free(ColiCudaTensor *tensor) { (void)tensor; }
int coli_cuda_tensor_upload(ColiCudaTensor **tensor, const void *weights,
                            const float *scales, int format, int columns,
                            int rows, int device) {
    if (!tensor || !weights || !scales || format != 1 || columns < 1 ||
        rows < 1 || device != 3 || uploads >= 3) return 0;
    *tensor = &uploaded[uploads++];
    return 1;
}
int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *output,
                         const float *input, int sequences) {
    if (gate != &uploaded[0] || up != &uploaded[1] || down != &uploaded[2] ||
        !output || !input || sequences != 1) return 0;
    output[0] = 7.0f; output[1] = 8.0f;
    fused++;
    return 1;
}
int coli_cuda_matmul(ColiCudaTensor **tensor, float *output, const float *input,
                     const void *weights, const float *scales, int format,
                     int sequences, int columns, int rows, int device,
                     int group_size) {
    if (!tensor || !output || !input || !weights || !scales || format != 1 ||
        sequences != 1 || columns != 3 || rows != 2 || device != 3 || group_size)
        return 0;
    const int8_t *matrix = weights;
    for (int row = 0; row < rows; row++) {
        output[row] = 0.0f;
        for (int column = 0; column < columns; column++)
            output[row] += matrix[row * columns + column] * input[column] * scales[row];
    }
    *tensor = &handle;
    calls++;
    return 1;
}

int main(void) {
    setenv("COLI_CUDA", "1", 1);
    setenv("COLI_GPU", "3", 1);
    char error[128] = {0};
    if (qwen38_accel_init(error, sizeof(error))) return 1;
    int8_t weights[] = {1, 2, 3, -2, 1, 4};
    float scales[] = {0.5f, 0.25f}, input[] = {2, -1, 3}, output[2];
    Qwen38AccelTensor tensor = {0};
    if (!qwen38_accel_matvec(&tensor, output, weights, scales, input,
                             2, 3, 0) || calls != 1 ||
        fabsf(output[0] - 4.5f) > 1e-6f || fabsf(output[1] - 1.75f) > 1e-6f)
        return 1;
    if (qwen38_accel_matvec(&tensor, output, weights, scales, input,
                            2, 3, 64)) return 1;
    int8_t expert_weights[12] = {0};
    float expert_scales[6] = {1, 1, 1, 1, 1, 1};
    Qwen38AccelTensor expert_tensors[3] = {{0}};
    if (!qwen38_accel_expert(expert_tensors, output, expert_weights,
                             expert_scales, input, 2, 2, 0) || uploads != 3 ||
        fused != 1 || output[0] != 7.0f || output[1] != 8.0f) return 1;
    for (int index = 0; index < 3; index++)
        qwen38_accel_tensor_close(&expert_tensors[index]);
    qwen38_accel_tensor_close(&tensor);
    qwen38_accel_shutdown();
    if (shutdowns != 1) return 1;
    puts("qwen38 CUDA/HIP expert dispatch: ok");
    return 0;
}
