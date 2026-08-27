#define _GNU_SOURCE
#include "../qwen38_ple_layer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void normalize_pair(const float *input, float *output) {
    float scale = 1.0f / sqrtf((input[0] * input[0] + input[1] * input[1]) / 2.0f + 1e-6f);
    output[0] = input[0] * scale;
    output[1] = input[1] * scale;
}

int main(void) {
    Qwen38Model model = {0};
    Qwen38Config *config = &model.config;
    config->hidden_size = 2; config->hc_count = 2;
    config->ngram_size = 2; config->heads_per_ngram = 2;
    config->ngram_vocab_base = 100; config->ple_embed_dim = 4;
    config->ple_conv_kernel = 2; config->rms_norm_eps = 1e-6f;
    config->vocab_size = 32; config->eos_token_id = 31; config->seed = 1234;

    float table_data[204 * 2];
    for (int row = 0; row < 204; row++) for (int column = 0; column < 2; column++)
        table_data[row * 2 + column] = row * 0.01f + column * 0.001f;
    Qwen38PleTable *table = &model.ple[0];
    table->maps[0].data = table_data; table->maps[0].nbytes = sizeof(table_data);
    table->row_offsets[1] = 204; table->shard_count = 1;
    table->row_width = 2; table->dtype = 2;

    float key_proj[16] = {0}, value_proj[8] = {0};
    for (int index = 0; index < 4; index++) key_proj[index * 4 + index] = 1.0f;
    value_proj[0] = 1.0f; value_proj[5] = 1.0f;
    float norms[4] = {0}, conv[8] = {0}, conv_state[16] = {0};
    Qwen38PleLayer layer = {0};
    layer.table_index = 0; layer.key_proj = key_proj; layer.value_proj = value_proj;
    layer.norm_key = norms; layer.norm_query = norms; layer.norm_conv = norms;
    layer.conv = conv; layer.conv_state = conv_state; layer.history[0] = 31;
    assert(qwen38_ple_hash_init(&layer.hash, 32, 2, 2, 100, 0, 1234) == 0);

    size_t workspace_size = qwen38_ple_layer_workspace_floats(config);
    float *workspace = malloc(workspace_size * sizeof(float)); assert(workspace);
    const float input[] = {1, 2, 3, 4};
    float output[4]; char error[128] = {0};
    assert(qwen38_ple_layer_step(&model, &layer, 7, input, output,
                                 workspace, workspace_size, error, sizeof(error)) == 0);

    int64_t tokens[] = {31, 7}; uint64_t rows[2];
    assert(qwen38_ple_hash_token(&layer.hash, tokens, 2, 1, 31, rows, 2) == 2);
    float embedding[] = {
        table_data[rows[0] * 2], table_data[rows[0] * 2 + 1],
        table_data[rows[1] * 2], table_data[rows[1] * 2 + 1],
    };
    float key_norm[4], query_norm[4];
    normalize_pair(embedding, key_norm); normalize_pair(embedding + 2, key_norm + 2);
    normalize_pair(input, query_norm); normalize_pair(input + 2, query_norm + 2);
    for (int stream = 0; stream < 2; stream++) {
        float gate = (key_norm[stream * 2] * query_norm[stream * 2] +
                      key_norm[stream * 2 + 1] * query_norm[stream * 2 + 1]) / sqrtf(2.0f);
        if (gate != 0.0f) gate = copysignf(sqrtf(fmaxf(fabsf(gate), 1e-6f)), gate);
        float scale = 1.0f / (1.0f + expf(-gate));
        assert(fabsf(output[stream * 2] - (input[stream * 2] + scale * embedding[0])) < 1e-5f);
        assert(fabsf(output[stream * 2 + 1] - (input[stream * 2 + 1] + scale * embedding[1])) < 1e-5f);
    }
    assert(layer.history[0] == 7);
    free(workspace);
    puts("qwen38 PLE layer oracle: ok");
    return 0;
}
