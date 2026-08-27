#ifndef QWEN38_MODEL_H
#define QWEN38_MODEL_H

#include <stddef.h>

#include "qwen38_config.h"
#include "qwen38_ple_table.h"
#include "st.h"

typedef struct {
    int expert_bits, expert_group_size;
    Qwen38Config config;
    shards source, experts;
    Qwen38PleTable ple[QWEN38_MAX_PLE_LAYERS];
} Qwen38Model;

int qwen38_model_open(Qwen38Model *model, const char *source_dir,
                      const char *expert_dir, char *error, size_t error_size);
void qwen38_model_close(Qwen38Model *model);

#endif
