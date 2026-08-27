#include "qwen38_config.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int fail(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static int integer(jval *object, const char *name, int *output,
                   char *error, size_t error_size) {
    jval *value = json_get(object, name);
    if (!value || value->t != J_NUM || value->num < 1 ||
        value->num > 2147483647.0 || value->num != (int)value->num)
        return fail(error, error_size, "invalid text_config.%s", name);
    *output = (int)value->num;
    return 0;
}

static int number(jval *object, const char *name, float *output,
                  char *error, size_t error_size) {
    jval *value = json_get(object, name);
    if (!value || value->t != J_NUM || !(value->num > 0.0))
        return fail(error, error_size, "invalid text_config.%s", name);
    *output = (float)value->num;
    return 0;
}

#define INT(name, field) \
    do { if (integer(text, name, &config->field, error, error_size)) goto done; } while (0)
#define NUM(name, field) \
    do { if (number(text, name, &config->field, error, error_size)) goto done; } while (0)

int qwen38_config_parse(const char *json, Qwen38Config *config,
                        char *error, size_t error_size) {
    if (!json || !config) return fail(error, error_size, "missing config input");
    memset(config, 0, sizeof(*config));
    config->seed = 1234;
    char *arena = NULL;
    jval *root = json_parse(json, &arena);
    int result = -1;
    if (!root || root->t != J_OBJ) {
        fail(error, error_size, "config.json is not an object");
        goto done;
    }
    jval *model_type = json_get(root, "model_type");
    jval *text = json_get(root, "text_config");
    if (!model_type || model_type->t != J_STR ||
        strcmp(model_type->str, "qwen4_exp") || !text || text->t != J_OBJ) {
        fail(error, error_size, "expected qwen4_exp with text_config");
        goto done;
    }
    jval *text_type = json_get(text, "model_type");
    if (!text_type || text_type->t != J_STR ||
        strcmp(text_type->str, "qwen4_exp_text")) {
        fail(error, error_size, "expected qwen4_exp_text");
        goto done;
    }

    INT("hidden_size", hidden_size);
    INT("num_hidden_layers", num_hidden_layers);
    INT("vocab_size", vocab_size);
    INT("num_experts", num_experts);
    INT("num_experts_per_tok", experts_per_token);
    INT("moe_intermediate_size", moe_intermediate_size);
    INT("shared_expert_intermediate_size", shared_expert_intermediate_size);
    INT("num_attention_heads", attention_heads);
    INT("num_key_value_heads", kv_heads);
    INT("head_dim", head_dim);
    INT("linear_num_key_heads", linear_key_heads);
    INT("linear_key_head_dim", linear_key_dim);
    INT("linear_num_value_heads", linear_value_heads);
    INT("linear_value_head_dim", linear_value_dim);
    INT("linear_conv_kernel_dim", linear_conv_kernel);
    INT("hc_count", hc_count);
    INT("hc_lowrank", hc_lowrank);
    INT("ngram_size", ngram_size);
    INT("ngram_vocab_size_base", ngram_vocab_base);
    INT("split_ngram_parts", ngram_parts);
    INT("ple_conv_kernel_size", ple_conv_kernel);
    INT("ple_embed_dim", ple_embed_dim);
    INT("heads_per_ngram", heads_per_ngram);
    INT("indexer_budget", indexer_budget);
    INT("indexer_compress_ratio", indexer_compress_ratio);
    INT("indexer_head_dim", indexer_head_dim);
    INT("indexer_kv_heads", indexer_kv_heads);
    INT("indexer_n_heads", indexer_heads);
    INT("mtp_num_hidden_layers", mtp_layers);
    INT("eos_token_id", eos_token_id);
    NUM("rms_norm_eps", rms_norm_eps);
    NUM("partial_rotary_factor", partial_rotary_factor);

    jval *rope = json_get(text, "rope_parameters");
    if (!rope || rope->t != J_OBJ ||
        number(rope, "rope_theta", &config->rope_theta, error, error_size))
        goto done;
    jval *seed = json_get(text, "seed");
    if (seed) {
        if (seed->t != J_NUM || seed->num < 0 || seed->num > 9007199254740991.0 ||
            seed->num != (uint64_t)seed->num) {
            fail(error, error_size, "invalid text_config.seed");
            goto done;
        }
        config->seed = (uint64_t)seed->num;
    }
    jval *layers = json_get(text, "layer_types");
    if (config->num_hidden_layers > QWEN38_MAX_LAYERS || !layers ||
        layers->t != J_ARR || layers->len != config->num_hidden_layers) {
        fail(error, error_size, "layer_types must match num_hidden_layers <= %d",
             QWEN38_MAX_LAYERS);
        goto done;
    }
    for (int layer = 0; layer < layers->len; layer++) {
        jval *kind = layers->kids[layer];
        if (!kind || kind->t != J_STR ||
            (strcmp(kind->str, "linear_attention") &&
             strcmp(kind->str, "full_attention"))) {
            fail(error, error_size, "unsupported layer_types[%d]", layer);
            goto done;
        }
        config->layer_is_full[layer] = !strcmp(kind->str, "full_attention");
        config->full_attention_layers += config->layer_is_full[layer];
    }
    jval *ple_layers = json_get(text, "ple_layer_ids");
    if (!ple_layers || ple_layers->t != J_ARR || ple_layers->len < 1 ||
        ple_layers->len > QWEN38_MAX_PLE_LAYERS) {
        fail(error, error_size, "invalid ple_layer_ids");
        goto done;
    }
    config->ple_layer_count = ple_layers->len;
    for (int index = 0; index < ple_layers->len; index++) {
        jval *value = ple_layers->kids[index];
        if (!value || value->t != J_NUM || value->num < 1 ||
            value->num > config->num_hidden_layers || value->num != (int)value->num) {
            fail(error, error_size, "invalid ple_layer_ids[%d]", index);
            goto done;
        }
        config->ple_layers[index] = (int)value->num;
        for (int previous = 0; previous < index; previous++)
            if (config->ple_layers[previous] == config->ple_layers[index]) {
                fail(error, error_size, "duplicate ple_layer_ids[%d]", index);
                goto done;
            }
    }
    if (config->experts_per_token > config->num_experts ||
        config->attention_heads % config->kv_heads ||
        config->linear_value_heads % config->linear_key_heads ||
        config->ple_embed_dim != config->hidden_size ||
        config->heads_per_ngram * config->ngram_size != config->attention_heads ||
        config->linear_conv_kernel < 2 || config->ple_conv_kernel < 2 ||
        !config->full_attention_layers) {
        fail(error, error_size, "inconsistent Qwen3.8 geometry");
        goto done;
    }
    result = 0;
done:
    free(arena);
    return result;
}

#undef INT
#undef NUM

int qwen38_config_load(const char *model_dir, Qwen38Config *config,
                       char *error, size_t error_size) {
    if (!model_dir) return fail(error, error_size, "missing model directory");
    char path[2048];
    int length = snprintf(path, sizeof(path), "%s/config.json", model_dir);
    if (length < 0 || (size_t)length >= sizeof(path))
        return fail(error, error_size, "model path is too long");
    FILE *file = fopen(path, "rb");
    if (!file) return fail(error, error_size, "cannot open %s", path);
    if (fseek(file, 0, SEEK_END) || ftell(file) < 0) {
        fclose(file);
        return fail(error, error_size, "cannot size %s", path);
    }
    long size = ftell(file);
    if (size > (16L << 20) || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return fail(error, error_size, "invalid config size");
    }
    char *json = malloc((size_t)size + 1);
    if (!json) {
        fclose(file);
        return fail(error, error_size, "out of memory reading config");
    }
    int ok = fread(json, 1, (size_t)size, file) == (size_t)size;
    fclose(file);
    json[size] = 0;
    int result = ok ? qwen38_config_parse(json, config, error, error_size) :
                      fail(error, error_size, "short read from %s", path);
    free(json);
    return result;
}
