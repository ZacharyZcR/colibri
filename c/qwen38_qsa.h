#ifndef COLIBRI_QWEN38_QSA_H
#define COLIBRI_QWEN38_QSA_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float score;
    size_t block;
} Qwen38QsaEntry;

/* Decode-step QSA selection. `query` is already normalized and RoPE-applied,
 * [query_heads, head_dim]. raw_keys is the persistent indexer cache,
 * [token_count, head_dim]. cos/sin are [token_count, rotary_dim]. */
int qwen38_qsa_select(const float *query, int query_heads,
                      const float *raw_keys, size_t token_count, int head_dim,
                      const float *key_norm_weight, float norm_eps,
                      const float *rope_cos, const float *rope_sin,
                      int rotary_dim, int compress_ratio, int token_budget,
                      uint8_t *selected, float *pooled_key,
                      Qwen38QsaEntry *heap, size_t heap_capacity);

/* Text-only decode path: positions are scalar, so RoPE is generated at each
 * compressed block start instead of retaining a context-sized cos/sin cache. */
int qwen38_qsa_select_text(const float *query, int query_heads,
                           const float *raw_keys, size_t token_count,
                           int head_dim, const float *key_norm_weight,
                           float norm_eps, int rotary_dim, float rope_theta,
                           int compress_ratio, int token_budget,
                           uint8_t *selected, float *pooled_key,
                           Qwen38QsaEntry *heap, size_t heap_capacity);

#endif
