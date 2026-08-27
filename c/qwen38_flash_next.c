#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "qwen38_runtime.h"
#include "tok.h"

typedef struct {
    char id[64];
    int payload_size, max_tokens;
    float temperature, top_p;
    char *payload;
} ServeRequest;

static int usage(const char *program) {
    fprintf(stderr,
        "usage: %s MODEL --experts DIR --ids 1,2,3 [--greedy N] [--summary] [--benchmark]\n",
        program);
    return 2;
}

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
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

static int read_request(ServeRequest *request) {
    char line[512], command[16], id[64];
    if (!fgets(line, sizeof(line), stdin)) return -1;
    if (sscanf(line, "%15s %63s", command, id) < 2) return 0;
    if (!strcmp(command, "CANCEL") || !strcmp(command, "STOP")) return 0;
    int slot, bytes, maximum;
    float temperature, top_p;
    if (strcmp(command, "SUBMIT") ||
        sscanf(line, "%*s %*s %d %d %d %f %f",
               &slot, &bytes, &maximum, &temperature, &top_p) != 5 ||
        bytes < 0 || bytes > (1 << 24) || maximum < 1) {
        printf("ERROR %s bad submit header\n", id);
        fflush(stdout);
        return 0;
    }
    (void)slot;
    request->payload = malloc((size_t)bytes + 1);
    if (!request->payload) return -1;
    if (fread(request->payload, 1, (size_t)bytes, stdin) != (size_t)bytes) {
        free(request->payload);
        return -1;
    }
    (void)fgetc(stdin);
    request->payload[bytes] = 0;
    snprintf(request->id, sizeof(request->id), "%s", id);
    request->payload_size = bytes;
    request->max_tokens = maximum;
    request->temperature = temperature;
    request->top_p = top_p;
    return 1;
}

typedef struct { float probability; int token; } Probability;

static int probability_descending(const void *left, const void *right) {
    float a = ((const Probability *)left)->probability;
    float b = ((const Probability *)right)->probability;
    return (b > a) - (a > b);
}

static int sample(const float *logits, int vocab, float temperature, float top_p) {
    if (temperature <= 0.0f) return argmax(logits, vocab);
    Probability *ranked = malloc((size_t)vocab * sizeof(*ranked));
    if (!ranked) return argmax(logits, vocab);
    float maximum = logits[0];
    for (int token = 1; token < vocab; token++)
        if (logits[token] > maximum) maximum = logits[token];
    double total = 0.0;
    for (int token = 0; token < vocab; token++) {
        float probability = expf((logits[token] - maximum) / temperature);
        ranked[token] = (Probability){probability, token};
        total += probability;
    }
    qsort(ranked, (size_t)vocab, sizeof(*ranked), probability_descending);
    double cutoff = top_p > 0.0f && top_p < 1.0f ? top_p * total : total;
    double kept = 0.0;
    int count = 0;
    while (count < vocab && kept < cutoff) kept += ranked[count++].probability;
    double target = (double)rand() / RAND_MAX * kept, cumulative = 0.0;
    int selected = ranked[0].token;
    for (int index = 0; index < count; index++) {
        cumulative += ranked[index].probability;
        if (cumulative >= target) { selected = ranked[index].token; break; }
    }
    free(ranked);
    return selected;
}

static void send_data(const char *id, const char *data, int size) {
    if (size < 1) return;
    printf("DATA %s %d\n", id, size);
    fwrite(data, 1, (size_t)size, stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static int context_capacity(void) {
    const char *value = getenv("Q38_MAXT");
    int capacity = 8192;
    if (value && parse_nonnegative(value, &capacity)) return -1;
    return capacity > 0 ? capacity : -1;
}

static int serve(const char *model_dir, const char *expert_dir) {
    int capacity = context_capacity();
    if (capacity < 1) {
        fprintf(stderr, "qwen38: invalid Q38_MAXT\n");
        return 2;
    }
    char error[512] = {0}, tokenizer_path[2048];
    Qwen38Runtime runtime;
    if (qwen38_runtime_open(&runtime, model_dir, expert_dir, (size_t)capacity,
                            error, sizeof(error))) {
        fprintf(stderr, "qwen38: %s\n", error);
        return 1;
    }
    if (snprintf(tokenizer_path, sizeof(tokenizer_path), "%s/tokenizer.json",
                 model_dir) >= (int)sizeof(tokenizer_path)) {
        qwen38_runtime_close(&runtime);
        return 2;
    }
    Tok tokenizer;
    tok_load(&tokenizer, tokenizer_path);
    float *logits = malloc((size_t)runtime.model.config.vocab_size * sizeof(float));
    if (!logits) {
        fprintf(stderr, "qwen38: out of memory allocating serve logits\n");
        tok_free(&tokenizer);
        qwen38_runtime_close(&runtime);
        return 1;
    }
    printf("\x01\x01READY\x01\x01\nSTAT 0 0.00 0.0 0.0\n");
    fflush(stdout);
    for (;;) {
        ServeRequest request = {0};
        int status = read_request(&request);
        if (status < 0) break;
        if (!status) continue;
        int token_capacity = request.payload_size + 64;
        int *ids = malloc((size_t)token_capacity * sizeof(*ids));
        int prompt_tokens = ids ? tok_encode(&tokenizer, request.payload,
            request.payload_size, ids, token_capacity) : -1;
        if (prompt_tokens < 1 || prompt_tokens + request.max_tokens > capacity) {
            printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",
                   request.id, prompt_tokens, request.max_tokens, capacity);
            fflush(stdout);
            free(ids); free(request.payload);
            continue;
        }
        printf("ACCEPT %s %d\n", request.id, prompt_tokens);
        fflush(stdout);
        qwen38_runtime_reset(&runtime);
        int failed = 0;
        for (int index = 0; index < prompt_tokens; index++)
            if (qwen38_runtime_step(&runtime, ids[index], logits,
                    (size_t)runtime.model.config.vocab_size,
                    error, sizeof(error))) { failed = 1; break; }
        int generated = 0, limited = 1;
        while (!failed && generated < request.max_tokens) {
            int token = sample(logits, runtime.model.config.vocab_size,
                               request.temperature, request.top_p);
            if (token == runtime.model.config.eos_token_id) {
                limited = 0;
                break;
            }
            char bytes[65536];
            int size = tok_decode(&tokenizer, &token, 1, bytes, sizeof(bytes));
            send_data(request.id, bytes, size);
            generated++;
            if (qwen38_runtime_step(&runtime, token, logits,
                    (size_t)runtime.model.config.vocab_size,
                    error, sizeof(error))) failed = 1;
        }
        if (failed) printf("ERROR %s %s\n", request.id, error);
        else printf("DONE %s STAT %d 0.00 0.0 0.0 %d %d\n",
                    request.id, generated, prompt_tokens, limited);
        fflush(stdout);
        free(ids); free(request.payload);
    }
    free(logits);
    tok_free(&tokenizer);
    qwen38_runtime_close(&runtime);
    return 0;
}

int main(int argc, char **argv) {
    const char *serve_dir = getenv("SNAP");
    if (getenv("SERVE") && serve_dir && *serve_dir) {
        const char *experts = getenv("Q38_EXPERTS");
        char default_experts[2048];
        if (!experts || !*experts) {
            if (snprintf(default_experts, sizeof(default_experts),
                         "%s/qwen38_experts", serve_dir) >=
                (int)sizeof(default_experts)) return 2;
            experts = default_experts;
        }
        return serve(serve_dir, experts);
    }
    if (argc < 6) return usage(argv[0]);
    const char *model_dir = argv[1], *expert_dir = NULL, *id_text = NULL;
    int greedy = 0, summary = 0, benchmark = 0;
    for (int arg = 2; arg < argc; arg++) {
        if (!strcmp(argv[arg], "--experts") && arg + 1 < argc)
            expert_dir = argv[++arg];
        else if (!strcmp(argv[arg], "--ids") && arg + 1 < argc)
            id_text = argv[++arg];
        else if (!strcmp(argv[arg], "--greedy") && arg + 1 < argc) {
            if (parse_nonnegative(argv[++arg], &greedy)) return usage(argv[0]);
        } else if (!strcmp(argv[arg], "--summary")) summary = 1;
        else if (!strcmp(argv[arg], "--benchmark")) benchmark = 1;
        else return usage(argv[0]);
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
    double load_start = now_seconds();
    if (qwen38_runtime_open(&runtime, model_dir, expert_dir,
                            count + (size_t)greedy, error, sizeof(error))) {
        fprintf(stderr, "qwen38: %s\n", error);
        free(ids);
        return 1;
    }
    double load_seconds = now_seconds() - load_start;
    int vocab = runtime.model.config.vocab_size;
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    int *predictions = malloc((count + (size_t)greedy) * sizeof(*predictions));
    if (!logits || !predictions) {
        fprintf(stderr, "qwen38: out of memory allocating diagnostic output\n");
        free(logits); free(predictions);
        qwen38_runtime_close(&runtime);
        free(ids);
        return 1;
    }
    double teacher_start = now_seconds();
    for (size_t index = 0; index < count; index++) {
        if (qwen38_runtime_step(&runtime, ids[index], logits, (size_t)vocab,
                                error, sizeof(error))) goto failed;
        predictions[index] = argmax(logits, vocab);
    }
    double teacher_seconds = now_seconds() - teacher_start;
    printf("teacher");
    for (size_t index = 0; index < count; index++)
        printf(" %d", predictions[index]);
    printf("\n");
    if (!summary) {
        printf("last_logits");
        for (int token = 0; token < vocab; token++) printf(" %.9g", logits[token]);
        printf("\n");
    }
    double decode_start = now_seconds();
    for (int step = 0; step < greedy; step++) {
        int token = argmax(logits, vocab);
        predictions[count + (size_t)step] = token;
        if (qwen38_runtime_step(&runtime, token, logits, (size_t)vocab,
                                error, sizeof(error))) goto failed;
    }
    double decode_seconds = now_seconds() - decode_start;
    for (int step = 0; step < greedy; step++)
        printf("greedy %d\n", predictions[count + (size_t)step]);
    if (benchmark)
        fprintf(stderr,
                "QWEN38_BENCH load_s=%.6f teacher_tokens=%zu teacher_s=%.6f teacher_tok_s=%.6f decode_tokens=%d decode_s=%.6f decode_tok_s=%.6f\n",
                load_seconds, count, teacher_seconds,
                teacher_seconds > 0.0 ? count / teacher_seconds : 0.0,
                greedy, decode_seconds,
                decode_seconds > 0.0 ? greedy / decode_seconds : 0.0);
    free(logits); free(predictions);
    qwen38_runtime_close(&runtime);
    free(ids);
    return 0;
failed:
    fprintf(stderr, "qwen38: %s\n", error);
    free(logits); free(predictions);
    qwen38_runtime_close(&runtime);
    free(ids);
    return 1;
}
