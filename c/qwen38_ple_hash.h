#ifndef COLIBRI_QWEN38_PLE_HASH_H
#define COLIBRI_QWEN38_PLE_HASH_H

#include <stddef.h>
#include <stdint.h>

#define QWEN38_PLE_MAX_NGRAM 8
#define QWEN38_PLE_MAX_HEADS 64

typedef struct {
    int ngram_size;
    int heads_per_ngram;
    int ngram_heads;
    uint64_t multipliers[QWEN38_PLE_MAX_NGRAM];
    uint64_t vocab_sizes[QWEN38_PLE_MAX_HEADS];
    uint64_t offsets[QWEN38_PLE_MAX_HEADS];
} Qwen38PleHash;

int qwen38_ple_hash_init(Qwen38PleHash *hash, uint64_t unigram_vocab,
                         int ngram_size, int heads_per_ngram,
                         uint64_t vocab_base, int ple_layer_index,
                         uint64_t seed);

int qwen38_ple_hash_token(const Qwen38PleHash *hash, const int64_t *tokens,
                          size_t token_count, size_t position,
                          int64_t eos_token, uint64_t *embedding_rows,
                          size_t row_capacity);

#endif
