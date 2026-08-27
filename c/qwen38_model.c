#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "qwen38_model.h"

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

static char *read_file(const char *directory, const char *name,
                       char *error, size_t error_size) {
    char path[2048];
    int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        fail(error, error_size, "path is too long");
        return NULL;
    }
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        if (file) fclose(file);
        fail(error, error_size, "cannot open %s", path);
        return NULL;
    }
    long size = ftell(file);
    if (size < 0 || size > (16L << 20) || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        fail(error, error_size, "invalid size for %s", path);
        return NULL;
    }
    char *content = malloc((size_t)size + 1);
    if (!content || fread(content, 1, (size_t)size, file) != (size_t)size) {
        free(content);
        fclose(file);
        fail(error, error_size, "cannot read %s", path);
        return NULL;
    }
    fclose(file);
    content[size] = 0;
    return content;
}

static int metadata(Qwen38Model *model, const char *directory,
                    char *error, size_t error_size) {
    char *json = read_file(directory, "qwen38_expert_meta.json",
                           error, error_size);
    if (!json) return -1;
    char *arena = NULL;
    jval *root = json_parse(json, &arena);
    int result = -1;
    jval *family = root ? json_get(root, "family") : NULL;
    jval *rows = root ? json_get(root, "rows") : NULL;
    jval *experts = root ? json_get(root, "experts_per_row") : NULL;
    jval *bits = root ? json_get(root, "expert_bits") : NULL;
    jval *group = root ? json_get(root, "group_size") : NULL;
    int expected_rows = model->config.num_hidden_layers + model->config.mtp_layers;
    if (!root || root->t != J_OBJ || !family || family->t != J_STR ||
        strcmp(family->str, "qwen38_flash_next") || !rows || rows->t != J_NUM ||
        rows->num != expected_rows || !experts || experts->t != J_NUM ||
        experts->num != model->config.num_experts || !bits || bits->t != J_NUM ||
        bits->num < 2 || bits->num > 8 || bits->num != (int)bits->num ||
        !group || group->t != J_NUM || group->num < 0 ||
        group->num > 2147483647.0 || group->num != (int)group->num) {
        fail(error, error_size, "expert overlay metadata does not match config");
        goto done;
    }
    model->expert_bits = (int)bits->num;
    model->expert_group_size = (int)group->num;
    model->config.expert_group_size = model->expert_group_size;
    result = 0;
done:
    free(arena);
    free(json);
    return result;
}

static int tensor_2d(shards *source, const char *name, int64_t rows,
                     int64_t columns, char *error, size_t error_size) {
    st_tensor *tensor = st_find(source, name);
    if (!tensor || tensor->rank != 2 || tensor->shape[0] != rows ||
        tensor->shape[1] != columns || tensor->dtype > 2)
        return fail(error, error_size, "invalid source tensor %s", name);
    return 0;
}

static int validate_experts(Qwen38Model *model, char *error, size_t error_size) {
    Qwen38Config *config = &model->config;
    int64_t weights = 3LL * config->hidden_size * config->moe_intermediate_size;
    int64_t weight_bytes = model->expert_bits <= 4 ? (weights + 1) / 2 : weights;
    int group = model->expert_group_size;
    int64_t scales = group ?
        2LL * config->moe_intermediate_size * ((config->hidden_size + group - 1) / group) +
        (int64_t)config->hidden_size * ((config->moe_intermediate_size + group - 1) / group) :
        2LL * config->moe_intermediate_size + config->hidden_size;
    int rows = config->num_hidden_layers + config->mtp_layers;
    char weight_name[256], scale_name[256];
    for (int row = 0; row < rows; row++) for (int expert = 0;
         expert < config->num_experts; expert++) {
        snprintf(weight_name, sizeof(weight_name),
                 "model.layers.%d.mlp.experts.%d.merged_weight", row, expert);
        snprintf(scale_name, sizeof(scale_name),
                 "model.layers.%d.mlp.experts.%d.qs", row, expert);
        st_tensor *weight = st_find(&model->experts, weight_name);
        st_tensor *scale = st_find(&model->experts, scale_name);
        if (!weight || weight->dtype != 3 || weight->rank != 1 ||
            weight->nbytes != weight_bytes)
            return fail(error, error_size, "invalid expert tensor %s", weight_name);
        if (!scale || scale->dtype != 2 || scale->rank != 1 ||
            scale->numel != scales)
            return fail(error, error_size, "invalid expert tensor %s", scale_name);
    }
    return 0;
}

void qwen38_model_close(Qwen38Model *model) {
    if (!model) return;
    for (int index = 0; index < model->config.ple_layer_count; index++)
        qwen38_ple_table_close(&model->ple[index]);
    st_destroy(&model->experts);
    st_destroy(&model->source);
    memset(model, 0, sizeof(*model));
}

int qwen38_model_open(Qwen38Model *model, const char *source_dir,
                      const char *expert_dir, char *error, size_t error_size) {
    if (!model || !source_dir || !expert_dir)
        return fail(error, error_size, "missing model paths");
    memset(model, 0, sizeof(*model));
    if (qwen38_config_load(source_dir, &model->config, error, error_size) ||
        metadata(model, expert_dir, error, error_size)) {
        qwen38_model_close(model);
        return -1;
    }
    st_init(&model->source, source_dir);
    st_init(&model->experts, expert_dir);
    Qwen38Config *config = &model->config;
    if (tensor_2d(&model->source, "model.language_model.embed_tokens.weight",
                  config->vocab_size, config->hidden_size, error, error_size) ||
        tensor_2d(&model->source, "lm_head.weight", config->vocab_size,
                  config->hidden_size, error, error_size) ||
        validate_experts(model, error, error_size)) goto fail;
    for (int index = 0; index < config->ple_layer_count; index++) {
        char prefix[256];
        snprintf(prefix, sizeof(prefix),
                 "model.language_model.layers.%d.ple.ple_embedding.ngram_embedding",
                 config->ple_layers[index] - 1);
        if (qwen38_ple_table_open(&model->ple[index], &model->source, prefix,
                                  config->ngram_parts, config->ple_embed_dim)) {
            fail(error, error_size, "cannot map PLE table for layer %d",
                 config->ple_layers[index]);
            goto fail;
        }
    }
    return 0;
fail:
    qwen38_model_close(model);
    return -1;
}
