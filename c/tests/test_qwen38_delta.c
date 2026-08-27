#include "../qwen38_delta.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void fill(float *values, int count, float multiplier, int modulus, int shift) {
    for (int index = 0; index < count; index++)
        values[index] = (float)((index % modulus) - shift) * multiplier;
}

static void near(float actual, float expected) {
    assert(fabsf(actual - expected) < 2e-5f);
}

int main(void) {
    /* Expected values come from transformers Qwen4Exp's torch recurrent
       fallback: l2norm(sum + 1e-6), sigmoid output gate, two carried steps. */
    Qwen38Config config = {0};
    config.hidden_size = 3;
    config.linear_key_heads = 1; config.linear_key_dim = 2;
    config.linear_value_heads = 2; config.linear_value_dim = 2;
    config.linear_conv_kernel = 2; config.rms_norm_eps = 1e-6f;
    float qkv[24], z[12], b[6], a[6], conv[16], out_weight[12];
    fill(qkv, 24, .07f, 7, 3); fill(z, 12, .05f, 5, 2);
    fill(b, 6, .09f, 5, 2); fill(a, 6, .04f, 7, 3);
    fill(conv, 16, .06f, 5, 2); fill(out_weight, 12, .03f, 7, 3);
    const float a_log[] = {-0.3f, 0.2f};
    const float dt_bias[] = {0.1f, -0.2f};
    const float norm[] = {0.8f, 1.2f};
    Qwen38DeltaWeights weights = {
        qkv, z, b, a, conv, a_log, dt_bias, norm, out_weight
    };
    Qwen38DeltaState state;
    assert(qwen38_delta_state_init(&state, &config) == 0);
    size_t count = qwen38_delta_workspace_floats(&config);
    float *workspace = malloc(count * sizeof(float)); assert(workspace);
    const float inputs[][3] = {{0.2f, -0.4f, 0.7f}, {-0.3f, 0.5f, 0.1f}};
    const float expected[][3] = {
        {-0.0328983665f, 0.0123443659f, -0.0288846307f},
        {-0.0001046467f, 0.0330128893f, -0.0047648074f},
    };
    float output[3];
    for (int token = 0; token < 2; token++) {
        assert(qwen38_delta_step(&config, &weights, &state, inputs[token],
                                 output, workspace, count) == 0);
        for (int index = 0; index < 3; index++) near(output[index], expected[token][index]);
    }
    const float expected_recurrent[] = {
        -0.0002698083f, -0.0001047767f, 0.0002676641f, 0.0000086137f,
        0.0007788866f, -0.0008324580f, -0.0001121892f, 0.0000363654f,
    };
    const float expected_ring[] = {
        -0.014f, 0.049f, -0.182f, 0.028f, 0.042f, 0.007f, 0.070f, -0.014f,
    };
    for (int index = 0; index < 8; index++) {
        near(state.recurrent[index], expected_recurrent[index]);
        near(state.convolution[index], expected_ring[index]);
    }
    free(workspace);
    qwen38_delta_state_close(&state);
    puts("qwen38 DeltaNet oracle: ok");
    return 0;
}
