#include "../qwen38_mhc.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void close_enough(float actual, float expected) {
    assert(fabsf(actual - expected) < 2e-6f);
}

int main(void) {
    enum { HC = 2, HIDDEN = 3, LOWRANK = 2, WIDTH = HC * HIDDEN };
    const float input[WIDTH] = {0.5f, -1.0f, 2.0f, -0.25f, 0.75f, 1.5f};
    const float norm[WIDTH] = {0.1f, -0.2f, 0.0f, 0.3f, -0.1f, 0.2f};
    const float down[LOWRANK * WIDTH] = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.1f, 0.2f,
        -0.3f, 0.2f, 0.1f, -0.2f, 0.5f, -0.4f,
    };
    const float up[WIDTH * LOWRANK] = {
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.2f,
        0.2f, 0.1f, -0.1f, 0.3f, 0.4f, -0.2f,
    };
    const float inject[HC * WIDTH] = {
        0.1f, 0.2f, -0.1f, 0.3f, -0.2f, 0.4f,
        -0.2f, 0.1f, 0.3f, -0.4f, 0.2f, 0.1f,
    };
    float mixed[HIDDEN], injection[HC], workspace[WIDTH + LOWRANK];
    assert(qwen38_mhc_mix(input, HC, HIDDEN, LOWRANK, 1e-6f, norm,
                          down, up, inject, mixed, injection, workspace,
                          WIDTH + LOWRANK) == 0);
    close_enough(mixed[0], 0.02158277f);
    close_enough(mixed[1], 0.01430508f);
    close_enough(mixed[2], 0.83782500f);
    close_enough(injection[0], 1.06675720f);
    close_enough(injection[1], 1.18883170f);

    assert(qwen38_mhc_mix(input, HC, HIDDEN, LOWRANK, 1e-6f, norm,
                          down, up, inject, mixed, injection, workspace,
                          WIDTH + LOWRANK - 1) == -1);
    puts("qwen38 mHC: ok");
    return 0;
}
