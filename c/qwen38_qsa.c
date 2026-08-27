#include "qwen38_qsa.h"

#include <math.h>
#include <string.h>

static int entry_less(Qwen38QsaEntry left, Qwen38QsaEntry right) {
    return left.score < right.score ||
           (left.score == right.score && left.block > right.block);
}

static void heap_up(Qwen38QsaEntry *heap, size_t index) {
    while (index) {
        size_t parent = (index - 1) / 2;
        if (!entry_less(heap[index], heap[parent])) break;
        Qwen38QsaEntry swap = heap[index]; heap[index] = heap[parent]; heap[parent] = swap;
        index = parent;
    }
}

static void heap_down(Qwen38QsaEntry *heap, size_t count, size_t index) {
    for (;;) {
        size_t left = index * 2 + 1;
        if (left >= count) return;
        size_t right = left + 1;
        size_t child = right < count && entry_less(heap[right], heap[left]) ? right : left;
        if (!entry_less(heap[child], heap[index])) return;
        Qwen38QsaEntry swap = heap[index]; heap[index] = heap[child]; heap[child] = swap;
        index = child;
    }
}

static void apply_rope(float *key, const float *cosine, const float *sine,
                       int rotary_dim) {
    int half = rotary_dim / 2;
    for (int dim = 0; dim < half; dim++) {
        float first = key[dim], second = key[dim + half];
        key[dim] = first * cosine[dim] - second * sine[dim];
        key[dim + half] = second * cosine[dim + half] +
                          first * sine[dim + half];
    }
}

static void apply_text_rope(float *key, size_t position, int rotary_dim,
                            float theta) {
    int half = rotary_dim / 2;
    for (int dim = 0; dim < half; dim++) {
        float angle = (float)position /
            powf(theta, (float)(2 * dim) / (float)rotary_dim);
        float cosine = cosf(angle), sine = sinf(angle);
        float first = key[dim], second = key[dim + half];
        key[dim] = first * cosine - second * sine;
        key[dim + half] = second * cosine + first * sine;
    }
}

static int select_impl(const float *query, int query_heads,
                       const float *raw_keys, size_t token_count, int head_dim,
                       const float *key_norm_weight, float norm_eps,
                       const float *rope_cos, const float *rope_sin,
                       int rotary_dim, float rope_theta, int text_rope,
                       int compress_ratio, int token_budget, uint8_t *selected,
                       float *pooled_key, Qwen38QsaEntry *heap,
                       size_t heap_capacity) {
    if (!query || query_heads < 1 || !raw_keys || !token_count || head_dim < 1 ||
        !key_norm_weight || norm_eps <= 0.0f || !selected || !pooled_key ||
        compress_ratio < 1 || token_budget < compress_ratio ||
        rotary_dim < 0 || rotary_dim > head_dim || (rotary_dim & 1) ||
        (rotary_dim && !text_rope && (!rope_cos || !rope_sin)) ||
        (text_rope && !(rope_theta > 0.0f))) return -1;
    size_t complete_blocks = token_count / (size_t)compress_ratio;
    size_t top_blocks = (size_t)token_budget / (size_t)compress_ratio;
    if (top_blocks > complete_blocks) top_blocks = complete_blocks;
    if (top_blocks && (!heap || heap_capacity < top_blocks)) return -1;
    memset(selected, 0, token_count);

    if (complete_blocks <= top_blocks) {
        memset(selected, 1, complete_blocks * (size_t)compress_ratio);
    } else {
        size_t heap_count = 0;
        float score_scale = 1.0f / sqrtf((float)head_dim);
        for (size_t block = 0; block < complete_blocks; block++) {
            size_t first_token = block * (size_t)compress_ratio;
            for (int dim = 0; dim < head_dim; dim++) {
                double sum = 0.0;
                for (int offset = 0; offset < compress_ratio; offset++)
                    sum += raw_keys[(first_token + (size_t)offset) * (size_t)head_dim +
                                    (size_t)dim];
                pooled_key[dim] = (float)(sum / compress_ratio);
            }
            double squares = 0.0;
            for (int dim = 0; dim < head_dim; dim++)
                squares += (double)pooled_key[dim] * pooled_key[dim];
            float norm = 1.0f / sqrtf((float)(squares / head_dim) + norm_eps);
            for (int dim = 0; dim < head_dim; dim++)
                pooled_key[dim] *= norm * (1.0f + key_norm_weight[dim]);
            if (rotary_dim) {
                if (text_rope)
                    apply_text_rope(pooled_key, first_token, rotary_dim, rope_theta);
                else
                    apply_rope(pooled_key,
                               rope_cos + first_token * (size_t)rotary_dim,
                               rope_sin + first_token * (size_t)rotary_dim,
                               rotary_dim);
            }

            float score = 0.0f;
            for (int head = 0; head < query_heads; head++) {
                const float *query_head = query + (size_t)head * (size_t)head_dim;
                double dot = 0.0;
                for (int dim = 0; dim < head_dim; dim++)
                    dot += (double)query_head[dim] * pooled_key[dim];
                if (dot > 0.0) score += (float)dot;
            }
            Qwen38QsaEntry candidate = {score * score_scale, block};
            if (heap_count < top_blocks) {
                heap[heap_count] = candidate;
                heap_up(heap, heap_count++);
            } else if (entry_less(heap[0], candidate)) {
                heap[0] = candidate;
                heap_down(heap, heap_count, 0);
            }
        }
        for (size_t entry = 0; entry < top_blocks; entry++) {
            size_t first = heap[entry].block * (size_t)compress_ratio;
            memset(selected + first, 1, (size_t)compress_ratio);
        }
    }
    memset(selected + complete_blocks * (size_t)compress_ratio, 1,
           token_count - complete_blocks * (size_t)compress_ratio);
    return (int)(top_blocks * (size_t)compress_ratio +
                 token_count - complete_blocks * (size_t)compress_ratio);
}

int qwen38_qsa_select(const float *query, int query_heads,
                      const float *raw_keys, size_t token_count, int head_dim,
                      const float *key_norm_weight, float norm_eps,
                      const float *rope_cos, const float *rope_sin,
                      int rotary_dim, int compress_ratio, int token_budget,
                      uint8_t *selected, float *pooled_key,
                      Qwen38QsaEntry *heap, size_t heap_capacity) {
    return select_impl(query, query_heads, raw_keys, token_count, head_dim,
                       key_norm_weight, norm_eps, rope_cos, rope_sin,
                       rotary_dim, 0.0f, 0, compress_ratio, token_budget,
                       selected, pooled_key, heap, heap_capacity);
}

int qwen38_qsa_select_text(const float *query, int query_heads,
                           const float *raw_keys, size_t token_count,
                           int head_dim, const float *key_norm_weight,
                           float norm_eps, int rotary_dim, float rope_theta,
                           int compress_ratio, int token_budget,
                           uint8_t *selected, float *pooled_key,
                           Qwen38QsaEntry *heap, size_t heap_capacity) {
    return select_impl(query, query_heads, raw_keys, token_count, head_dim,
                       key_norm_weight, norm_eps, NULL, NULL, rotary_dim,
                       rope_theta, 1, compress_ratio, token_budget,
                       selected, pooled_key, heap, heap_capacity);
}
