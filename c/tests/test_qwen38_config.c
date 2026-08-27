#include "../qwen38_config.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static const char *config_json =
    "{\"model_type\":\"qwen4_exp\",\"text_config\":{" 
    "\"model_type\":\"qwen4_exp_text\",\"hidden_size\":16,"
    "\"num_hidden_layers\":4,\"vocab_size\":32,\"num_experts\":8,"
    "\"num_experts_per_tok\":2,\"moe_intermediate_size\":6,"
    "\"shared_expert_intermediate_size\":6,\"num_attention_heads\":4,"
    "\"num_key_value_heads\":2,\"head_dim\":4,\"linear_num_key_heads\":2,"
    "\"linear_key_head_dim\":4,\"linear_num_value_heads\":4,"
    "\"linear_value_head_dim\":4,\"linear_conv_kernel_dim\":4,"
    "\"hc_count\":2,\"hc_lowrank\":4,\"ngram_size\":2,"
    "\"ngram_vocab_size_base\":100,\"split_ngram_parts\":2,"
    "\"ple_conv_kernel_size\":3,\"ple_embed_dim\":16,\"heads_per_ngram\":2,"
    "\"indexer_budget\":8,\"indexer_compress_ratio\":2,"
    "\"indexer_head_dim\":4,\"indexer_kv_heads\":1,\"indexer_n_heads\":2,"
    "\"mtp_num_hidden_layers\":1,\"rms_norm_eps\":0.000001,"
    "\"partial_rotary_factor\":0.5,"
    "\"ple_layer_ids\":[2],"
    "\"rope_parameters\":{\"rope_theta\":10000},"
    "\"layer_types\":[\"linear_attention\",\"linear_attention\","
    "\"linear_attention\",\"full_attention\"]}}";

int main(int argc, char **argv) {
    Qwen38Config config;
    char error[256] = {0};
    if (argc == 2) {
        assert(qwen38_config_load(argv[1], &config, error, sizeof(error)) == 0);
        printf("qwen38 config: %d layers, %d full, %d experts, %d MTP\n",
               config.num_hidden_layers, config.full_attention_layers,
               config.num_experts, config.mtp_layers);
        return 0;
    }
    assert(qwen38_config_parse(config_json, &config, error, sizeof(error)) == 0);
    assert(config.hidden_size == 16 && config.num_hidden_layers == 4);
    assert(config.full_attention_layers == 1 && config.layer_is_full[3]);
    assert(config.mtp_layers == 1 && config.ngram_parts == 2);
    assert(config.ple_layer_count == 1 && config.ple_layers[0] == 2);

    char broken[4096];
    snprintf(broken, sizeof(broken), "%s", config_json);
    char *type = strstr(broken, "full_attention");
    memcpy(type, "bad_attention!", 14);
    assert(qwen38_config_parse(broken, &config, error, sizeof(error)) == -1);
    assert(strstr(error, "unsupported layer_types"));
    puts("qwen38 config: ok");
    return 0;
}
