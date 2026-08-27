#define _GNU_SOURCE
#include "../qwen38_ple_table.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_fixture(const char *directory) {
    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    const char *header =
        "{\"ple.shard_0.weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],"
        "\"data_offsets\":[0,12]},\"ple.shard_1.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[3,3],\"data_offsets\":[12,30]}}";
    uint64_t header_size = strlen(header);
    const float values[15] = {
        0, 1, 2, 10, 11, 12,
        20, 21, 22, 30, 31, 32, 40, 41, 42,
    };
    uint16_t bf16[15];
    for (int i = 0; i < 15; i++) {
        uint32_t bits;
        memcpy(&bits, &values[i], sizeof(bits));
        bf16[i] = (uint16_t)(bits >> 16);
    }
    FILE *file = fopen(path, "wb"); assert(file);
    assert(fwrite(&header_size, 8, 1, file) == 1);
    assert(fwrite(header, 1, header_size, file) == header_size);
    assert(fwrite(bf16, sizeof(uint16_t), 15, file) == 15);
    assert(fclose(file) == 0);
}

int main(void) {
    char directory[] = "test_qwen38_ple_table_XXXXXX";
    assert(mkdtemp(directory));
    write_fixture(directory);
    shards model;
    st_init(&model, directory);
    Qwen38PleTable table;
    assert(qwen38_ple_table_open(&table, &model, "ple", 2, 3) == 0);
    assert(table.row_offsets[0] == 0 && table.row_offsets[1] == 2 &&
           table.row_offsets[2] == 5);

    const uint64_t rows[] = {4, 0, 2};
    float output[9];
    assert(qwen38_ple_table_lookup(&table, rows, 3, output, 9) == 0);
    const float expected[] = {40, 41, 42, 0, 1, 2, 20, 21, 22};
    for (int i = 0; i < 9; i++) assert(output[i] == expected[i]);
    const uint64_t invalid[] = {5};
    assert(qwen38_ple_table_lookup(&table, invalid, 1, output, 9) == -1);

    qwen38_ple_table_close(&table);
    st_destroy(&model);
    char path[512]; snprintf(path, sizeof(path), "%s/model.safetensors", directory);
    assert(remove(path) == 0);
#ifdef _WIN32
    assert(_rmdir(directory) == 0);
#else
    assert(rmdir(directory) == 0);
#endif
    puts("qwen38 PLE table: ok");
    return 0;
}
