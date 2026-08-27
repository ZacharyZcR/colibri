#include "qwen38_ple_table.h"

#include <stdio.h>
#include <string.h>

static int find_shard(const Qwen38PleTable *table, uint64_t row) {
    int low = 0, high = table->shard_count;
    while (low < high) {
        int middle = low + (high - low) / 2;
        if (row < table->row_offsets[middle]) high = middle;
        else if (row >= table->row_offsets[middle + 1]) low = middle + 1;
        else return middle;
    }
    return -1;
}

void qwen38_ple_table_close(Qwen38PleTable *table) {
    if (!table) return;
    for (int shard = 0; shard < table->shard_count; shard++)
        st_unmap_raw(&table->maps[shard]);
    memset(table, 0, sizeof(*table));
}

int qwen38_ple_table_open(Qwen38PleTable *table, shards *model,
                          const char *tensor_prefix, int shard_count,
                          int row_width) {
    if (!table || !model || !tensor_prefix || shard_count < 1 ||
        shard_count > QWEN38_PLE_MAX_TABLE_SHARDS || row_width < 1) return -1;
    memset(table, 0, sizeof(*table));
    table->row_width = row_width;
    char name[2048];
    for (int shard = 0; shard < shard_count; shard++) {
        int length = snprintf(name, sizeof(name), "%s.shard_%d.weight",
                              tensor_prefix, shard);
        if (length < 0 || (size_t)length >= sizeof(name)) goto fail;
        st_tensor *tensor = st_find(model, name);
        if (!tensor || tensor->rank != 2 || tensor->shape[0] < 1 ||
            tensor->shape[1] != row_width || tensor->dtype > 2 ||
            (shard && tensor->dtype != table->dtype)) goto fail;
        if (!shard) table->dtype = tensor->dtype;
        if ((uint64_t)tensor->shape[0] > UINT64_MAX - table->row_offsets[shard])
            goto fail;
        if (st_map_raw(model, name, &table->maps[shard]) != 0) goto fail;
        table->row_offsets[shard + 1] = table->row_offsets[shard] +
                                        (uint64_t)tensor->shape[0];
        table->shard_count++;
    }
    return 0;
fail:
    qwen38_ple_table_close(table);
    return -1;
}

int qwen38_ple_table_lookup(const Qwen38PleTable *table, const uint64_t *rows,
                            size_t row_count, float *output,
                            size_t output_floats) {
    if (!table || table->shard_count < 1 || !rows || !output ||
        row_count > SIZE_MAX / (size_t)table->row_width ||
        output_floats < row_count * (size_t)table->row_width) return -1;
    int element_size = st_dtype_esz(table->dtype);
    for (size_t item = 0; item < row_count; item++) {
        int shard = find_shard(table, rows[item]);
        if (shard < 0) return -1;
        uint64_t local_row = rows[item] - table->row_offsets[shard];
        const unsigned char *source = table->maps[shard].data;
        source += local_row * (uint64_t)table->row_width * (uint64_t)element_size;
        float *destination = output + item * (size_t)table->row_width;
        if (table->dtype == 2) {
            memcpy(destination, source, (size_t)table->row_width * sizeof(float));
        } else {
            for (int column = 0; column < table->row_width; column++) {
                uint16_t half;
                memcpy(&half, source + (size_t)column * sizeof(half), sizeof(half));
                destination[column] = table->dtype == 0 ?
                    bf16_to_f32(half) : f16_to_f32(half);
            }
        }
    }
    return 0;
}
