#ifndef COLIBRI_QWEN38_PLE_TABLE_H
#define COLIBRI_QWEN38_PLE_TABLE_H

#include "st.h"

#include <stddef.h>
#include <stdint.h>

#define QWEN38_PLE_MAX_TABLE_SHARDS 512

typedef struct {
    st_mapped_raw maps[QWEN38_PLE_MAX_TABLE_SHARDS];
    uint64_t row_offsets[QWEN38_PLE_MAX_TABLE_SHARDS + 1];
    int shard_count;
    int row_width;
    int dtype;
} Qwen38PleTable;

int qwen38_ple_table_open(Qwen38PleTable *table, shards *model,
                          const char *tensor_prefix, int shard_count,
                          int row_width);
void qwen38_ple_table_close(Qwen38PleTable *table);
int qwen38_ple_table_lookup(const Qwen38PleTable *table, const uint64_t *rows,
                            size_t row_count, float *output,
                            size_t output_floats);

#endif
