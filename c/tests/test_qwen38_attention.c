#include "../qwen38_attention.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void near(float actual, float expected) {
    assert(fabsf(actual - expected) < 2e-5f);
}

int main(void) {
    Qwen38Config config = {0};
    config.hidden_size = 2; config.attention_heads = 1;
    config.kv_heads = 1; config.head_dim = 2;
    config.indexer_heads = 1; config.indexer_kv_heads = 1;
    config.indexer_head_dim = 2; config.indexer_compress_ratio = 2;
    config.indexer_budget = 2; config.partial_rotary_factor = 1.0f;
    config.rope_theta = 10000.0f; config.rms_norm_eps = 1e-6f;
    const float q_proj[] = {
        1, 0, 0, 1,
        0, 0, 0, 0,
    };
    const float identity[] = {1, 0, 0, 1};
    const float norm[] = {0, 0};
    const float index_qk[] = {1, 0, 0, 1, 1, 0, 0, 1};
    Qwen38AttentionWeights weights = {
        q_proj, identity, identity, identity, norm, norm,
        index_qk, norm, norm,
    };
    Qwen38AttentionState state;
    assert(qwen38_attention_state_init(&state, &config, 2) == 0);
    size_t workspace_size = qwen38_attention_workspace_floats(&config);
    float *workspace = malloc(workspace_size * sizeof(float)); assert(workspace);
    float output[2];
    const float first[] = {1, 0};
    assert(qwen38_attention_step(&config, &weights, &state, first, output,
                                 workspace, workspace_size) == 0);
    near(output[0], 0.5f); near(output[1], 0.0f);

    const float second[] = {0, 1};
    assert(qwen38_attention_step(&config, &weights, &state, second, output,
                                 workspace, workspace_size) == 0);
    float old_score = -sqrtf(2.0f) * sinf(1.0f);
    float current_score = sqrtf(2.0f);
    float old_weight = expf(old_score - current_score);
    old_weight /= old_weight + 1.0f;
    near(output[0], 0.5f * old_weight);
    near(output[1], 0.5f * (1.0f - old_weight));
    assert(state.length == 2);
    assert(qwen38_attention_step(&config, &weights, &state, second, output,
                                 workspace, workspace_size) == -1);
    free(workspace);
    qwen38_attention_state_close(&state);
    puts("qwen38 attention oracle: ok");
    return 0;
}
