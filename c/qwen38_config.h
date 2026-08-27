#ifndef QWEN38_CONFIG_H
#define QWEN38_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#define QWEN38_MAX_LAYERS 512

typedef struct {
    int hidden_size, num_hidden_layers, vocab_size;
    int num_experts, experts_per_token, moe_intermediate_size;
    int shared_expert_intermediate_size;
    int attention_heads, kv_heads, head_dim;
    int linear_key_heads, linear_key_dim;
    int linear_value_heads, linear_value_dim, linear_conv_kernel;
    int hc_count, hc_lowrank;
    int ngram_size, ngram_vocab_base, ngram_parts;
    int ple_conv_kernel, ple_embed_dim, heads_per_ngram;
    int indexer_budget, indexer_compress_ratio;
    int indexer_head_dim, indexer_kv_heads, indexer_heads;
    int mtp_layers, full_attention_layers;
    float rms_norm_eps, rope_theta, partial_rotary_factor;
    uint8_t layer_is_full[QWEN38_MAX_LAYERS];
} Qwen38Config;

int qwen38_config_parse(const char *json, Qwen38Config *config,
                        char *error, size_t error_size);
int qwen38_config_load(const char *model_dir, Qwen38Config *config,
                       char *error, size_t error_size);

#endif
