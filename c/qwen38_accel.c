#include "qwen38_accel.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

#ifdef COLI_CUDA
static int cuda_enabled, cuda_device;
typedef struct {
    int row, expert, occupied;
    uint64_t tick;
    Qwen38AccelTensor tensors[3];
} CachedExpert;
static CachedExpert *cache;
static int32_t *cache_map;
static size_t cache_capacity;
static int cache_rows, cache_experts;
static uint64_t cache_tick, cache_hits, cache_misses, cache_uploads,
                cache_evictions;

static void close_cached(CachedExpert *entry) {
    if (!entry || !entry->occupied) return;
    if (cache_map && entry->row >= 0 && entry->row < cache_rows &&
        entry->expert >= 0 && entry->expert < cache_experts)
        cache_map[(size_t)entry->row * cache_experts + entry->expert] = 0;
    for (int index = 0; index < 3; index++)
        qwen38_accel_tensor_close(&entry->tensors[index]);
    memset(entry, 0, sizeof(*entry));
}

static int init_cache(int rows, int experts, int hidden, int intermediate,
                      char *error, size_t error_size) {
    if (rows < 1 || experts < 1 || hidden < 1 || intermediate < 1)
        return fail(error, error_size, "qwen38: invalid GPU cache geometry");
    size_t free_bytes = 0, total_bytes = 0;
    if (!coli_cuda_mem_info(cuda_device, &free_bytes, &total_bytes))
        return fail(error, error_size, "qwen38: cannot query GPU memory");
    (void)total_bytes;
    const char *setting = getenv("Q38_GPU_EXPERT_GB");
    double gib = -1.0;
    if (setting && strcmp(setting, "auto")) {
        char *end = NULL;
        errno = 0;
        gib = strtod(setting, &end);
        if (errno || !setting[0] || !end || *end || !isfinite(gib) ||
            gib < 0.0 || gib > (double)SIZE_MAX / 1073741824.0)
            return fail(error, error_size,
                        "qwen38: Q38_GPU_EXPERT_GB must be auto or non-negative");
    }
    size_t budget = gib >= 0.0 ? (size_t)(gib * 1073741824.0) :
        (free_bytes > (1ULL << 30) ? free_bytes - (1ULL << 30) : 0);
    if (budget > free_bytes) budget = free_bytes;
    size_t expert_bytes = 3ULL * (size_t)hidden * intermediate +
                          (size_t)(2 * intermediate + hidden) * sizeof(float) +
                          3 * 4096;
    if ((size_t)rows > SIZE_MAX / (size_t)experts)
        return fail(error, error_size, "qwen38: GPU cache geometry overflow");
    size_t total = (size_t)rows * experts;
    cache_capacity = expert_bytes ? budget / expert_bytes : 0;
    if (cache_capacity > total) cache_capacity = total;
    if (cache_capacity > INT32_MAX) cache_capacity = INT32_MAX;
    if (!cache_capacity) {
        fprintf(stderr, "[GPU] Qwen3.8 expert cache disabled (budget %.2f GB)\n",
                budget / 1073741824.0);
        return 0;
    }
    cache = calloc(cache_capacity, sizeof(*cache));
    cache_map = calloc(total, sizeof(*cache_map));
    if (!cache || !cache_map) {
        free(cache); free(cache_map); cache = NULL; cache_map = NULL;
        cache_capacity = 0;
        return fail(error, error_size,
                    "qwen38: cannot allocate GPU expert cache index");
    }
    cache_rows = rows;
    cache_experts = experts;
    fprintf(stderr,
            "[GPU] Qwen3.8 expert cache: %zu slots, %.2f GB budget, %.2f MB/expert\n",
            cache_capacity, budget / 1073741824.0,
            expert_bytes / 1048576.0);
    return 0;
}

static int init_cuda(int rows, int experts, int hidden, int intermediate,
                     char *error, size_t error_size) {
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
    if (init_cache(rows, experts, hidden, intermediate, error, error_size))
        return -1;
    fprintf(stderr, "[GPU] Qwen3.8 routed expert matvec active on device %d\n",
            cuda_device);
    return 0;
}
#else
static int init_cuda(int rows, int experts, int hidden, int intermediate,
                     char *error, size_t error_size) {
    (void)rows; (void)experts; (void)hidden; (void)intermediate;
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

int qwen38_accel_init(int rows, int experts, int hidden, int intermediate,
                      char *error, size_t error_size) {
    if (getenv("COLI_CUDA") && atoi(getenv("COLI_CUDA")) &&
        getenv("COLI_METAL") && atoi(getenv("COLI_METAL")))
        return fail(error, error_size, "qwen38: choose COLI_CUDA or COLI_METAL, not both");
    if (init_cuda(rows, experts, hidden, intermediate, error, error_size)) {
#ifdef COLI_CUDA
        if (cuda_enabled) coli_cuda_shutdown();
        cuda_enabled = 0;
#endif
        return -1;
    }
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
    if (cuda_enabled && cache_capacity)
        fprintf(stderr,
                "[GPU] Qwen3.8 expert cache: hits=%llu misses=%llu uploads=%llu evictions=%llu\n",
                (unsigned long long)cache_hits,
                (unsigned long long)cache_misses,
                (unsigned long long)cache_uploads,
                (unsigned long long)cache_evictions);
    for (size_t index = 0; index < cache_capacity; index++)
        close_cached(&cache[index]);
    free(cache); free(cache_map); cache = NULL; cache_map = NULL;
    cache_capacity = 0;
    cache_rows = cache_experts = 0;
    cache_tick = cache_hits = cache_misses = cache_uploads = cache_evictions = 0;
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

int qwen38_accel_cached_expert(int row, int expert, float *output,
                               const float *input) {
#ifdef COLI_CUDA
    if (!cuda_enabled || !cache || row < 0 || row >= cache_rows ||
        expert < 0 || expert >= cache_experts || !output || !input) return 0;
    int32_t mapped = cache_map[(size_t)row * cache_experts + expert];
    if (mapped > 0 && (size_t)mapped <= cache_capacity) {
        CachedExpert *entry = &cache[mapped - 1];
        if (entry->occupied && entry->row == row && entry->expert == expert) {
            if (coli_cuda_expert_mlp(entry->tensors[0].cuda,
                                     entry->tensors[1].cuda,
                                     entry->tensors[2].cuda, output, input, 1)) {
                entry->tick = ++cache_tick;
                cache_hits++;
                return 1;
            }
            close_cached(entry);
        } else {
            cache_map[(size_t)row * cache_experts + expert] = 0;
        }
    }
    cache_misses++;
#else
    (void)row; (void)expert; (void)output; (void)input;
#endif
    return 0;
}

void qwen38_accel_cache_store(int row, int expert,
                              Qwen38AccelTensor tensors[3]) {
#ifdef COLI_CUDA
    if (!cuda_enabled || !cache || !tensors || row < 0 || row >= cache_rows ||
        expert < 0 || expert >= cache_experts || !tensors[0].cuda ||
        !tensors[1].cuda || !tensors[2].cuda) return;
    CachedExpert *target = NULL;
    if (cache_map[(size_t)row * cache_experts + expert]) return;
    for (size_t index = 0; index < cache_capacity; index++) {
        CachedExpert *entry = &cache[index];
        if (!entry->occupied) { target = entry; break; }
        if (!target || entry->tick < target->tick) target = entry;
    }
    if (!target) return;
    if (target->occupied) { close_cached(target); cache_evictions++; }
    target->row = row; target->expert = expert; target->occupied = 1;
    target->tick = ++cache_tick;
    cache_map[(size_t)row * cache_experts + expert] =
        (int32_t)(target - cache + 1);
    for (int index = 0; index < 3; index++) {
        target->tensors[index] = tensors[index];
        memset(&tensors[index], 0, sizeof(tensors[index]));
    }
    cache_uploads++;
#else
    (void)row; (void)expert; (void)tensors;
#endif
}
