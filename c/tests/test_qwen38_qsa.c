#include "../qwen38_qsa.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    const float query[8] = {1, 0, 0, 0, 0, 1, 0, 0};
    const float keys[9 * 4] = {
        1, 0, 0, 0, 1, 0, 0, 0,
        0, 2, 2, 0, 0, 2, 2, 0,
        -1, -1, 0, 0, -1, -1, 0, 0,
        2, 2, 0, 0, 2, 2, 0, 0,
        7, 7, 7, 7,
    };
    const float norm[4] = {0, 0, 0, 0};
    const float cosines[9 * 4] = {
        1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
        1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
    };
    const float sines[9 * 4] = {0};
    uint8_t selected[9];
    float pooled[4];
    Qwen38QsaEntry heap[2];
    int count = qwen38_qsa_select(query, 2, keys, 9, 4, norm, 1e-6f,
                                  cosines, sines, 4, 2, 4, selected,
                                  pooled, heap, 2);
    assert(count == 5);
    const uint8_t expected[9] = {1, 1, 0, 0, 0, 0, 1, 1, 1};
    for (int i = 0; i < 9; i++) assert(selected[i] == expected[i]);

    /* Under budget, every complete block and the causal tail are visible. */
    assert(qwen38_qsa_select(query, 2, keys, 5, 4, norm, 1e-6f,
                             NULL, NULL, 0, 2, 8, selected,
                             pooled, heap, 2) == 5);
    for (int i = 0; i < 5; i++) assert(selected[i]);
    puts("qwen38 QSA: ok");
    return 0;
}
