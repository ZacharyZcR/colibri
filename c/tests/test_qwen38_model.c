#define _GNU_SOURCE
#include "../qwen38_model.h"
#include "../qwen38_expert.h"
#include "../qwen38_moe_layer.h"
#include "../qwen38_delta_layer.h"
#include "../qwen38_linear_layer.h"

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *text;
    size_t length, capacity;
} Buffer;

static void append(Buffer *buffer, const char *format, ...) {
    for (;;) {
        if (buffer->capacity - buffer->length < 256) {
            buffer->capacity = buffer->capacity ? buffer->capacity * 2 : 4096;
            buffer->text = realloc(buffer->text, buffer->capacity);
            assert(buffer->text);
        }
        va_list args;
        va_start(args, format);
        int written = vsnprintf(buffer->text + buffer->length,
                                buffer->capacity - buffer->length, format, args);
        va_end(args);
        assert(written >= 0);
        if ((size_t)written < buffer->capacity - buffer->length) {
            buffer->length += (size_t)written;
            return;
        }
        buffer->capacity = buffer->length + (size_t)written + 1;
        buffer->text = realloc(buffer->text, buffer->capacity);
        assert(buffer->text);
    }
}

static void write_text(const char *directory, const char *name, const char *text) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", directory, name);
    FILE *file = fopen(path, "wb"); assert(file);
    assert(fwrite(text, 1, strlen(text), file) == strlen(text));
    assert(fclose(file) == 0);
}

static void write_safetensors(const char *directory, Buffer *header,
                              size_t data_size) {
    append(header, "}");
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    FILE *file = fopen(path, "wb"); assert(file);
    uint64_t header_size = header->length;
    assert(fwrite(&header_size, 8, 1, file) == 1);
    assert(fwrite(header->text, 1, header->length, file) == header->length);
    unsigned char zero[1024] = {0};
    while (data_size) {
        size_t chunk = data_size < sizeof(zero) ? data_size : sizeof(zero);
        assert(fwrite(zero, 1, chunk, file) == chunk);
        data_size -= chunk;
    }
    assert(fclose(file) == 0);
}

static void add_tensor(Buffer *header, int *first, const char *name,
                       const char *dtype, const char *shape,
                       size_t *offset, size_t bytes) {
    append(header, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":%s,"
                   "\"data_offsets\":[%zu,%zu]}",
           *first ? "" : ",", name, dtype, shape, *offset, *offset + bytes);
    *first = 0;
    *offset += bytes;
}

static void source_fixture(const char *directory) {
    const char *config =
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
        "\"ple_conv_kernel_size\":3,\"ple_embed_dim\":16,"
        "\"heads_per_ngram\":2,\"ple_layer_ids\":[2],"
        "\"indexer_budget\":8,\"indexer_compress_ratio\":2,"
        "\"indexer_head_dim\":4,\"indexer_kv_heads\":1,\"indexer_n_heads\":2,"
        "\"mtp_num_hidden_layers\":1,\"rms_norm_eps\":0.000001,"
        "\"partial_rotary_factor\":0.5,"
        "\"rope_parameters\":{\"rope_theta\":10000},"
        "\"layer_types\":[\"linear_attention\",\"linear_attention\","
        "\"linear_attention\",\"full_attention\"]}}";
    write_text(directory, "config.json", config);
    Buffer header = {0}; append(&header, "{");
    int first = 1; size_t offset = 0;
    add_tensor(&header, &first, "model.language_model.embed_tokens.weight",
               "BF16", "[32,16]", &offset, 32 * 16 * 2);
    add_tensor(&header, &first, "lm_head.weight", "BF16", "[32,16]",
               &offset, 32 * 16 * 2);
    for (int shard = 0; shard < 2; shard++) {
        char name[256];
        snprintf(name, sizeof(name), "model.language_model.layers.1.ple."
                 "ple_embedding.ngram_embedding.shard_%d.weight", shard);
        add_tensor(&header, &first, name, "BF16", "[2,16]", &offset, 2 * 16 * 2);
    }
    const struct { const char *name, *shape; size_t count; } dense[] = {
        {"model.language_model.layers.0.mlp_hyper_connection.hc_norm.weight", "[32]", 32},
        {"model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", "[4,32]", 4 * 32},
        {"model.language_model.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", "[32,4]", 32 * 4},
        {"model.language_model.layers.0.mlp_hyper_connection.block_inject_weight.weight", "[2,32]", 2 * 32},
        {"model.language_model.layers.0.mlp.gate.weight", "[8,16]", 8 * 16},
        {"model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", "[6,16]", 6 * 16},
        {"model.language_model.layers.0.mlp.shared_expert.up_proj.weight", "[6,16]", 6 * 16},
        {"model.language_model.layers.0.mlp.shared_expert.down_proj.weight", "[16,6]", 16 * 6},
        {"model.language_model.layers.0.mlp.shared_expert_gate.weight", "[1,16]", 16},
    };
    for (size_t index = 0; index < sizeof(dense) / sizeof(dense[0]); index++)
        add_tensor(&header, &first, dense[index].name, "BF16", dense[index].shape,
                   &offset, dense[index].count * 2);
    const struct { const char *name, *shape; size_t count; } delta[] = {
        {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "[32,16]", 32 * 16},
        {"model.language_model.layers.0.linear_attn.in_proj_z.weight", "[16,16]", 16 * 16},
        {"model.language_model.layers.0.linear_attn.in_proj_b.weight", "[4,16]", 4 * 16},
        {"model.language_model.layers.0.linear_attn.in_proj_a.weight", "[4,16]", 4 * 16},
        {"model.language_model.layers.0.linear_attn.conv1d.weight", "[32,1,4]", 32 * 4},
        {"model.language_model.layers.0.linear_attn.A_log", "[4]", 4},
        {"model.language_model.layers.0.linear_attn.dt_bias", "[4]", 4},
        {"model.language_model.layers.0.linear_attn.norm.weight", "[4]", 4},
        {"model.language_model.layers.0.linear_attn.out_proj.weight", "[16,16]", 16 * 16},
    };
    for (size_t index = 0; index < sizeof(delta) / sizeof(delta[0]); index++)
        add_tensor(&header, &first, delta[index].name, "BF16", delta[index].shape,
                   &offset, delta[index].count * 2);
    const struct { const char *name, *shape; size_t count; } attention_hyper[] = {
        {"model.language_model.layers.0.attn_hyper_connection.hc_norm.weight", "[32]", 32},
        {"model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight", "[4,32]", 4 * 32},
        {"model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "[32,4]", 32 * 4},
        {"model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight", "[2,32]", 2 * 32},
    };
    for (size_t index = 0; index < sizeof(attention_hyper) / sizeof(attention_hyper[0]); index++)
        add_tensor(&header, &first, attention_hyper[index].name, "BF16",
                   attention_hyper[index].shape, &offset, attention_hyper[index].count * 2);
    write_safetensors(directory, &header, offset);
    free(header.text);
}

static void overlay_fixture(const char *directory) {
    write_text(directory, "qwen38_expert_meta.json",
        "{\"family\":\"qwen38_flash_next\",\"rows\":5,"
        "\"experts_per_row\":8,\"expert_bits\":4,\"group_size\":0}");
    Buffer header = {0}; append(&header, "{");
    int first = 1; size_t offset = 0;
    for (int row = 0; row < 5; row++) for (int expert = 0; expert < 8; expert++) {
        char name[256];
        snprintf(name, sizeof(name),
                 "model.layers.%d.mlp.experts.%d.merged_weight", row, expert);
        add_tensor(&header, &first, name, "U8", "[144]", &offset, 144);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.experts.%d.qs", row, expert);
        add_tensor(&header, &first, name, "F32", "[28]", &offset, 28 * 4);
    }
    write_safetensors(directory, &header, offset);
    free(header.text);
}

static void cleanup(const char *directory, int source) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    assert(remove(path) == 0);
    snprintf(path, sizeof(path), "%s/%s", directory,
             source ? "config.json" : "qwen38_expert_meta.json");
    assert(remove(path) == 0);
#ifdef _WIN32
    assert(_rmdir(directory) == 0);
#else
    assert(rmdir(directory) == 0);
#endif
}

int main(void) {
    char source[] = "test_qwen38_source_XXXXXX";
    char overlay[] = "test_qwen38_overlay_XXXXXX";
    assert(mkdtemp(source) && mkdtemp(overlay));
    source_fixture(source);
    overlay_fixture(overlay);
    Qwen38Model model;
    char error[256] = {0};
    assert(qwen38_model_open(&model, source, overlay, error, sizeof(error)) == 0);
    assert(model.expert_bits == 4 && model.expert_group_size == 0);
    assert(model.source.n == 26 && model.experts.n == 80);
    assert(model.ple[0].row_offsets[2] == 4);
    Qwen38Expert expert;
    assert(qwen38_expert_load(&model, 4, 7, &expert, error, sizeof(error)) == 0);
    for (int index = 0; index < 3 * 16 * 6; index++) assert(expert.weights[index] == 0);
    qwen38_expert_close(&expert);
    float input[16], moe_output[16], moe_workspace[28];
    for (int index = 0; index < 16; index++) input[index] = 1.0f;
    const int expert_ids[] = {0, 7};
    const float router_weights[] = {0.25f, 0.75f};
    assert(qwen38_moe_forward(&model, 4, input, expert_ids, router_weights, 2,
                              moe_output, moe_workspace, 28,
                              error, sizeof(error)) == 0);
    for (int index = 0; index < 16; index++) assert(moe_output[index] == 0.0f);
    Qwen38MoeLayer layer;
    assert(qwen38_moe_layer_load(&model, 0, &layer, error, sizeof(error)) == 0);
    float hyper_input[32], hyper_output[32];
    for (int index = 0; index < 32; index++) hyper_input[index] = index + 1.0f;
    size_t layer_workspace_size = qwen38_moe_layer_workspace_floats(&model.config);
    float *layer_workspace = malloc(layer_workspace_size * sizeof(float));
    assert(layer_workspace);
    assert(qwen38_moe_layer_forward(&model, 0, &layer, hyper_input, hyper_output,
                                    layer_workspace, layer_workspace_size,
                                    error, sizeof(error)) == 0);
    for (int index = 0; index < 32; index++) assert(hyper_output[index] == hyper_input[index]);
    free(layer_workspace);
    qwen38_moe_layer_close(&layer);
    Qwen38DeltaLayer delta;
    Qwen38DeltaState delta_state;
    assert(qwen38_delta_layer_load(&model, 0, &delta, error, sizeof(error)) == 0);
    assert(qwen38_delta_state_init(&delta_state, &model.config) == 0);
    size_t delta_workspace_size = qwen38_delta_workspace_floats(&model.config);
    float *delta_workspace = malloc(delta_workspace_size * sizeof(float));
    float delta_output[16]; assert(delta_workspace);
    assert(qwen38_delta_step(&model.config, &delta.view, &delta_state, input,
                             delta_output, delta_workspace,
                             delta_workspace_size) == 0);
    for (int index = 0; index < 16; index++) assert(delta_output[index] == 0.0f);
    free(delta_workspace);
    qwen38_delta_state_close(&delta_state);
    qwen38_delta_layer_close(&delta);
    Qwen38LinearLayer linear;
    assert(qwen38_linear_layer_load(&model, 0, &linear, error, sizeof(error)) == 0);
    size_t linear_workspace_size = qwen38_linear_layer_workspace_floats(&model.config);
    float *linear_workspace = malloc(linear_workspace_size * sizeof(float));
    assert(linear_workspace);
    assert(qwen38_linear_layer_step(&model, 0, &linear, hyper_input, hyper_output,
                                    linear_workspace, linear_workspace_size,
                                    error, sizeof(error)) == 0);
    for (int index = 0; index < 32; index++) assert(hyper_output[index] == hyper_input[index]);
    free(linear_workspace);
    qwen38_linear_layer_close(&linear);
    uint64_t row = 3; float result[16];
    assert(qwen38_ple_table_lookup(&model.ple[0], &row, 1, result, 16) == 0);
    for (int index = 0; index < 16; index++) assert(result[index] == 0.0f);
    qwen38_model_close(&model);
    assert(model.source.n == 0 && model.experts.n == 0);
    write_text(overlay, "qwen38_expert_meta.json",
        "{\"family\":\"qwen38_flash_next\",\"rows\":4,"
        "\"experts_per_row\":8,\"expert_bits\":4,\"group_size\":0}");
    assert(qwen38_model_open(&model, source, overlay, error, sizeof(error)) == -1);
    assert(strstr(error, "metadata does not match"));
    assert(model.source.n == 0 && model.experts.n == 0);
    cleanup(source, 1);
    cleanup(overlay, 0);
    puts("qwen38 model loader: ok");
    return 0;
}
