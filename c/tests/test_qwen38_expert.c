#include "../qwen38_expert.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static void close_enough(float actual, float expected) {
    assert(fabsf(actual - expected) < 1e-5f);
}

int main(void) {
    Qwen38Config config = {0};
    config.hidden_size = 2;
    config.moe_intermediate_size = 2;
    config.expert_group_size = 0;
    int8_t weights[] = {
        1, 0, 0, 1,
        1, 0, 0, 1,
        1, 0, 0, 1,
    };
    float scales[] = {1, 1, 1, 1, 1, 1};
    Qwen38Expert expert = {.weights = weights, .scales = scales};
    float input[] = {1, 2};
    float output[2], workspace[4];
    assert(qwen38_expert_forward(&expert, &config, input, output,
                                 workspace, 4) == 0);
    close_enough(output[0], 1.0f / (1.0f + expf(-1.0f)));
    close_enough(output[1], 4.0f / (1.0f + expf(-2.0f)));

    config.expert_group_size = 1;
    float grouped_scales[] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
    };
    expert.scales = grouped_scales;
    assert(qwen38_expert_forward(&expert, &config, input, output,
                                 workspace, 4) == 0);
    float gate0 = 1.0f, gate1 = 8.0f;
    close_enough(output[0], gate0 / (1.0f + expf(-gate0)) * gate0);
    close_enough(output[1], 4.0f * gate1 / (1.0f + expf(-gate1)) * gate1);
    puts("qwen38 expert forward: ok");
    return 0;
}
