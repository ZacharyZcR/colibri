#include "qwen38_ple_hash.h"

#include <limits.h>
#include <string.h>

#define SPLITMIX_GAMMA UINT64_C(0x9e3779b97f4a7c15)
#define SPLITMIX_M1 UINT64_C(0xbf58476d1ce4e5b9)
#define SPLITMIX_M2 UINT64_C(0x94d049bb133111eb)
#define PLE_LAYER_PRIME UINT64_C(10007)

static uint64_t splitmix64(uint64_t value) {
    value += SPLITMIX_GAMMA;
    value = (value ^ (value >> 30)) * SPLITMIX_M1;
    value = (value ^ (value >> 27)) * SPLITMIX_M2;
    return value ^ (value >> 31);
}

static int is_prime(uint64_t value) {
    if (value < 2) return 0;
    if (!(value & 1)) return value == 2;
    for (uint64_t divisor = 3; divisor <= value / divisor; divisor += 2)
        if (value % divisor == 0) return 0;
    return 1;
}

static uint64_t next_prime(uint64_t value) {
    do value++; while (!is_prime(value));
    return value;
}

/* torch.remainder interprets its overflowing int64 hash as signed and returns
 * a non-negative remainder. Compute the same result without signed-overflow UB. */
static uint64_t signed_remainder(uint64_t bits, uint64_t divisor) {
    uint64_t remainder = bits % divisor;
    if (bits <= INT64_MAX) return remainder;
    uint64_t two64 = (UINT64_MAX % divisor + 1) % divisor;
    return remainder >= two64 ? remainder - two64 : divisor - (two64 - remainder);
}

int qwen38_ple_hash_init(Qwen38PleHash *hash, uint64_t unigram_vocab,
                         int ngram_size, int heads_per_ngram,
                         uint64_t vocab_base, int ple_layer_index,
                         uint64_t seed) {
    if (!hash || !unigram_vocab || ngram_size < 2 ||
        ngram_size > QWEN38_PLE_MAX_NGRAM || heads_per_ngram < 1 ||
        (ngram_size - 1) * heads_per_ngram > QWEN38_PLE_MAX_HEADS ||
        vocab_base < 2 || ple_layer_index < 0) return -1;
    memset(hash, 0, sizeof(*hash));
    hash->ngram_size = ngram_size;
    hash->heads_per_ngram = heads_per_ngram;
    hash->ngram_heads = (ngram_size - 1) * heads_per_ngram;

    uint64_t bound = (uint64_t)INT64_MAX / unigram_vocab;
    uint64_t half_bound = bound / 2;
    if (!half_bound) return -1;
    uint64_t layer_seed = seed + PLE_LAYER_PRIME * (uint64_t)ple_layer_index;
    for (int i = 0; i < ngram_size; i++) {
        uint64_t value = layer_seed + SPLITMIX_GAMMA * (uint64_t)(i + 1);
        hash->multipliers[i] = 2 * (splitmix64(value) % half_bound) + 1;
    }

    uint64_t prime = vocab_base - 1, offset = 0;
    int skip = ple_layer_index * hash->ngram_heads;
    for (int i = 0; i < skip; i++) prime = next_prime(prime);
    for (int i = 0; i < hash->ngram_heads; i++) {
        prime = next_prime(prime);
        hash->vocab_sizes[i] = prime;
        hash->offsets[i] = offset;
        if (UINT64_MAX - offset < prime) return -1;
        offset += prime;
    }
    return 0;
}

int qwen38_ple_hash_token(const Qwen38PleHash *hash, const int64_t *tokens,
                          size_t token_count, size_t position,
                          int64_t eos_token, uint64_t *embedding_rows,
                          size_t row_capacity) {
    if (!hash || !tokens || position >= token_count || !embedding_rows ||
        row_capacity < (size_t)hash->ngram_heads) return -1;
    int64_t shifted[QWEN38_PLE_MAX_NGRAM];
    shifted[0] = tokens[position];
    int reset = 0;
    for (int shift = 1; shift < hash->ngram_size; shift++) {
        if (position < (size_t)shift || tokens[position - shift] == eos_token)
            reset = 1;
        shifted[shift] = reset ? eos_token : tokens[position - shift];
    }

    int output = 0;
    for (int ngram = 2; ngram <= hash->ngram_size; ngram++) {
        uint64_t mixed = 0;
        for (int i = 0; i < ngram; i++)
            mixed ^= (uint64_t)shifted[i] * hash->multipliers[i];
        int first = (ngram - 2) * hash->heads_per_ngram;
        for (int head = 0; head < hash->heads_per_ngram; head++) {
            int index = first + head;
            embedding_rows[output++] = hash->offsets[index] +
                signed_remainder(mixed, hash->vocab_sizes[index]);
        }
    }
    return output;
}
