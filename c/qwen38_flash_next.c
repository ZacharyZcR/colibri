#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen38_runtime.h"

static int usage(const char *program) {
    fprintf(stderr,
        "usage: %s MODEL --experts DIR --ids 1,2,3 [--greedy N]\n",
        program);
    return 2;
}

static int parse_nonnegative(const char *text, int *output) {
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !text[0] || *end || value < 0 || value > 2147483647L)
        return -1;
    *output = (int)value;
    return 0;
}

static int parse_ids(const char *text, int64_t **output, size_t *count) {
    size_t capacity = 16, length = 0;
    int64_t *ids = malloc(capacity * sizeof(*ids));
    if (!ids) return -1;
    const char *cursor = text;
    while (*cursor) {
        char *end = NULL;
        errno = 0;
        long long value = strtoll(cursor, &end, 10);
        if (errno || end == cursor || value < 0 || (*end && *end != ',')) {
            free(ids);
            return -1;
        }
        if (length == capacity) {
            if (capacity > SIZE_MAX / 2 / sizeof(*ids)) {
                free(ids);
                return -1;
            }
            capacity *= 2;
            int64_t *grown = realloc(ids, capacity * sizeof(*ids));
            if (!grown) {
                free(ids);
                return -1;
            }
            ids = grown;
        }
        ids[length++] = (int64_t)value;
        cursor = *end ? end + 1 : end;
        if (!*end) break;
        if (!*cursor) {
            free(ids);
            return -1;
        }
    }
    if (!length) {
        free(ids);
        return -1;
    }
    *output = ids;
    *count = length;
    return 0;
}

static int argmax(const float *logits, int count) {
    int best = 0;
    for (int token = 1; token < count; token++)
        if (logits[token] > logits[best]) best = token;
    return best;
}

int main(int argc, char **argv) {
    if (argc < 6) return usage(argv[0]);
    const char *model_dir = argv[1], *expert_dir = NULL, *id_text = NULL;
    int greedy = 0;
    for (int arg = 2; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--experts") && arg + 1 < argc)
            expert_dir = argv[++arg];
        else if (!strcmp(argv[arg], "--ids") && arg + 1 < argc)
            id_text = argv[++arg];
        else if (!strcmp(argv[arg], "--greedy") && arg + 1 < argc) {
            if (parse_nonnegative(argv[++arg], &greedy)) return usage(argv[0]);
        } else return usage(argv[0]);
    }
    if (!expert_dir || !id_text) return usage(argv[0]);
    int64_t *ids = NULL;
    size_t count = 0;
    if (parse_ids(id_text, &ids, &count) ||
        count > SIZE_MAX - (size_t)greedy) {
        fprintf(stderr, "qwen38: invalid token IDs\n");
        return 2;
    }
    char error[512] = {0};
    Qwen38Runtime runtime;
    if (qwen38_runtime_open(&runtime, model_dir, expert_dir,
                            count + (size_t)greedy, error, sizeof(error))) {
        fprintf(stderr, "qwen38: %s\n", error);
        free(ids);
        return 1;
    }
    int vocab = runtime.model.config.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (!logits) {
        fprintf(stderr, "qwen38: out of memory allocating logits\n");
        qwen38_runtime_close(&runtime);
        free(ids);
        return 1;
    }
    printf("teacher");
    for (size_t index = 0; index < count; index++) {
        if (qwen38_runtime_step(&runtime, ids[index], logits, (size_t)vocab,
                                error, sizeof(error))) goto failed;
        printf(" %d", argmax(logits, vocab));
    }
    printf("\nlast_logits");
    for (int token = 0; token < vocab; token++) printf(" %.9g", logits[token]);
    printf("\n");
    for (int step = 0; step < greedy; step++) {
        int token = argmax(logits, vocab);
        printf("greedy %d\n", token);
        if (qwen38_runtime_step(&runtime, token, logits, (size_t)vocab,
                                error, sizeof(error))) goto failed;
    }
    free(logits);
    qwen38_runtime_close(&runtime);
    free(ids);
    return 0;
failed:
    fprintf(stderr, "qwen38: %s\n", error);
    free(logits);
    qwen38_runtime_close(&runtime);
    free(ids);
    return 1;
}
