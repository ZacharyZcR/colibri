#include "../qwen38_moe_layer.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
    const float input[] = {1.0f, 1.0f};
    const float router[] = {
        1.0f, 0.0f,
        3.0f, 0.0f,
        2.0f, 0.0f,
    };
    int selected[2];
    float weights[2], logits[3];
    assert(qwen38_router_topk(input, router, 2, 3, 2,
                              selected, weights, logits) == 0);
    assert(selected[0] == 1 && selected[1] == 2);
    float expected = expf(1.0f);
    assert(fabsf(weights[0] - expected / (expected + 1.0f)) < 1e-6f);
    assert(fabsf(weights[1] - 1.0f / (expected + 1.0f)) < 1e-6f);

    Qwen38Config config = {0};
    config.hc_count = 2; config.hidden_size = 4; config.hc_lowrank = 3;
    config.moe_intermediate_size = 2; config.shared_expert_intermediate_size = 2;
    config.num_experts = 3; config.experts_per_token = 2;
    assert(qwen38_moe_layer_workspace_floats(&config) >= 2 * 2 + 4);
    puts("qwen38 MoE layer: ok");
    return 0;
}
