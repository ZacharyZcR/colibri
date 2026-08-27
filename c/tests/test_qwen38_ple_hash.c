#include "../qwen38_ple_hash.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    Qwen38PleHash hash;
    assert(qwen38_ple_hash_init(&hash, 248320, 3, 8, 20000000, 0, 1234) == 0);
    const uint64_t multipliers[] = {
        UINT64_C(23703573157769), UINT64_C(20109073645365),
        UINT64_C(8052911324071),
    };
    const uint64_t vocab[] = {
        20000003, 20000023, 20000033, 20000047,
        20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147,
        20000153, 20000159, 20000161, 20000171,
    };
    for (int i = 0; i < 3; i++) assert(hash.multipliers[i] == multipliers[i]);
    for (int i = 0; i < 16; i++) assert(hash.vocab_sizes[i] == vocab[i]);

    const int64_t tokens[] = {248044, 10, 20, 30, 248044, 40};
    const uint64_t expected[] = {
        9878115, 26555603, 54895210, 62571545,
        80580723, 119917398, 128922427, 147596134,
        168936175, 195223391, 219226064, 233524685,
        246670267, 279816194, 297531600, 306108296,
    };
    uint64_t rows[16];
    assert(qwen38_ple_hash_token(&hash, tokens, 6, 3, 248044, rows, 16) == 16);
    for (int i = 0; i < 16; i++) assert(rows[i] == expected[i]);

    const uint64_t after_eos[] = {
        10580618, 37840707, 45212370, 71723338,
        84052565, 102293892, 120404177, 159281483,
        167500023, 187273292, 213206235, 226933061,
        259691440, 273675538, 298609479, 305321328,
    };
    assert(qwen38_ple_hash_token(&hash, tokens, 6, 5, 248044, rows, 16) == 16);
    for (int i = 0; i < 16; i++) assert(rows[i] == after_eos[i]);

    puts("qwen38 PLE hash: ok");
    return 0;
}
