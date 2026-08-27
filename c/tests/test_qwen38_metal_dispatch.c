#define _GNU_SOURCE
#define COLI_METAL
#include "../qwen38_accel.c"

#include <math.h>
#include <stdint.h>

struct ColiMetalTensor { int marker; };
static struct ColiMetalTensor handle;
static int calls, shutdowns;

int coli_metal_init(void) { return 1; }
void coli_metal_shutdown(void) { shutdowns++; }
void coli_metal_tensor_free(ColiMetalTensor *tensor) { (void)tensor; }
int coli_metal_matmul(ColiMetalTensor **tensor, float *output,
                      const float *input, const void *weights,
                      const float *scales, int format, int sequences,
                      int columns, int rows, int group_size) {
    if (!tensor || !output || !input || !weights || !scales || format != 1 ||
        sequences != 1 || columns != 3 || rows != 2 || group_size)
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
    setenv("COLI_METAL", "1", 1);
    char error[128] = {0};
    if (qwen38_accel_init(2, 4, 2, 2, error, sizeof(error))) return 1;
    int8_t weights[] = {1, 2, 3, -2, 1, 4};
    float scales[] = {0.5f, 0.25f}, input[] = {2, -1, 3}, output[2];
    Qwen38AccelTensor tensor = {0};
    if (!qwen38_accel_matvec(&tensor, output, weights, scales, input,
                             2, 3, 0) || calls != 1 ||
        fabsf(output[0] - 4.5f) > 1e-6f || fabsf(output[1] - 1.75f) > 1e-6f)
        return 1;
    qwen38_accel_tensor_close(&tensor);
    qwen38_accel_shutdown();
    if (shutdowns != 1) return 1;
    puts("qwen38 Metal expert dispatch: ok");
    return 0;
}
