#include "segment_adapters.h"
#include "segment_adapter_internal.h"

#include <pthread.h>

#include "qwen38_runtime.h"

typedef struct {
    Qwen38Model model;
    Qwen38RuntimeLayer *layers;
    uint32_t layer_begin, layer_end, context_tokens;
    pthread_mutex_t run_lock;
    int lock_ready;
} Qwen38SegmentEngine;

typedef struct {
    Qwen38SegmentEngine *engine;
    Qwen38RuntimeLayer *layers;
    float *first, *second, *workspace;
    size_t workspace_floats, position;
    uint32_t context_tokens;
} Qwen38SegmentSession;

static void close_engine_layers(Qwen38SegmentEngine *engine) {
    if (!engine || !engine->layers) return;
    for (uint32_t index = 0; index < engine->layer_end - engine->layer_begin;
         index++) {
        if (engine->layers[index].full_attention)
            qwen38_full_layer_close(&engine->layers[index].full);
        else qwen38_linear_layer_close(&engine->layers[index].linear);
    }
    free(engine->layers);
    engine->layers = NULL;
}

static void destroy_engine(void *engine_impl) {
    Qwen38SegmentEngine *engine = engine_impl;
    if (!engine) return;
    close_engine_layers(engine);
    qwen38_model_close(&engine->model);
    if (engine->lock_ready) pthread_mutex_destroy(&engine->run_lock);
    free(engine);
}

static int engine_open(void **engine_impl, ColiSegmentCapabilities *capabilities,
                       const ColiSegmentEngineOptions *options,
                       char *error, size_t error_size) {
    if (!engine_impl || !capabilities || !options || !options->model_dir ||
        options->layer_begin >= options->layer_end || !options->context_tokens ||
        options->context_tokens > 262144)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Qwen3.8 Segment open");
    *engine_impl = NULL;
    if (options->backend_mask &&
        (options->backend_mask & ~COLI_SEGMENT_CAP_CPU))
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.8 Segment currently supports CPU");
    const char *experts = getenv("Q38_EXPERTS");
    char default_experts[2048];
    if (!experts || !*experts) {
        int length = snprintf(default_experts, sizeof(default_experts),
                              "%s/qwen38_experts", options->model_dir);
        if (length < 0 || (size_t)length >= sizeof(default_experts))
            return coli_segment_adapter_error(error, error_size,
                                               "Qwen3.8 expert path is too long");
        experts = default_experts;
    }
    Qwen38SegmentEngine *engine = calloc(1, sizeof(*engine));
    if (!engine)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory opening Qwen3.8 Segment");
    if (pthread_mutex_init(&engine->run_lock, NULL)) {
        free(engine);
        return coli_segment_adapter_error(error, error_size,
                                           "cannot initialize Qwen3.8 Segment lock");
    }
    engine->lock_ready = 1;
    if (qwen38_model_open(&engine->model, options->model_dir, experts,
                          error, error_size)) goto fail;
    Qwen38Config *config = &engine->model.config;
    if (options->layer_end > (uint32_t)config->num_hidden_layers) {
        coli_segment_adapter_error(error, error_size,
                                    "Qwen3.8 Segment range exceeds model");
        goto fail;
    }
    engine->layer_begin = options->layer_begin;
    engine->layer_end = options->layer_end;
    engine->context_tokens = options->context_tokens;
    size_t count = options->layer_end - options->layer_begin;
    engine->layers = calloc(count, sizeof(*engine->layers));
    if (!engine->layers) {
        coli_segment_adapter_error(error, error_size,
                                    "out of memory loading Qwen3.8 layers");
        goto fail;
    }
    for (uint32_t index = 0; index < count; index++) {
        int layer = (int)(options->layer_begin + index);
        Qwen38RuntimeLayer *item = &engine->layers[index];
        item->full_attention = config->layer_is_full[layer];
        int result = item->full_attention ?
            qwen38_full_layer_load(&engine->model, layer,
                options->context_tokens, &item->full, error, error_size) :
            qwen38_linear_layer_load(&engine->model, layer,
                &item->linear, error, error_size);
        if (result) goto fail;
    }
    memset(capabilities, 0, sizeof(*capabilities));
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->abi_version = COLI_SEGMENT_ABI_VERSION;
    capabilities->flags = COLI_SEGMENT_CAP_TOKEN_IDS |
        COLI_SEGMENT_CAP_SNAPSHOT | COLI_SEGMENT_CAP_RANGE_NATIVE |
        COLI_SEGMENT_CAP_MULTI_SESSION | COLI_SEGMENT_CAP_CPU;
    coli_segment_capability_string(capabilities->engine_id,
        sizeof(capabilities->engine_id), "qwen38_flash_next");
    coli_segment_capability_string(capabilities->state_schema,
        sizeof(capabilities->state_schema),
        "qwen38/qsa-deltanet-mhc-ple-f32-v1");
    coli_segment_capability_string(capabilities->numeric_class,
        sizeof(capabilities->numeric_class), "qwen38/f32-int4-8/cpu-v1");
    capabilities->state_dtype = COLI_SEGMENT_DTYPE_F32;
    capabilities->state_width = (uint32_t)(config->hc_count * config->hidden_size);
    capabilities->max_batch_rows = 128;
    capabilities->max_context_tokens = 262144;
    capabilities->num_layers = (uint32_t)config->num_hidden_layers;
    *engine_impl = engine;
    return 0;
fail:
    destroy_engine(engine);
    return -1;
}

static int clone_ple(Qwen38PleLayer *destination, const Qwen38PleLayer *source,
                     const Qwen38Config *config) {
    *destination = *source;
    size_t count = (size_t)config->hc_count * config->hidden_size *
        (config->ple_conv_kernel - 1) * config->ngram_size;
    destination->conv_state = calloc(count, sizeof(float));
    if (!destination->conv_state) return -1;
    for (int index = 0; index < config->ngram_size - 1; index++)
        destination->history[index] = config->eos_token_id;
    return 0;
}

static void destroy_session(void *session_impl) {
    Qwen38SegmentSession *session = session_impl;
    if (!session) return;
    if (session->layers) for (uint32_t index = 0;
         index < session->engine->layer_end - session->engine->layer_begin;
         index++) {
        Qwen38RuntimeLayer *item = &session->layers[index];
        if (item->full_attention) {
            qwen38_attention_state_close(&item->full.state);
            if (item->full.has_ple) free(item->full.ple.conv_state);
        } else {
            qwen38_delta_state_close(&item->linear.state);
            if (item->linear.has_ple) free(item->linear.ple.conv_state);
        }
    }
    free(session->layers); free(session->first); free(session->second);
    free(session->workspace); free(session);
}

static int session_create(void *engine_impl, void **session_impl,
                          const ColiSegmentSessionOptions *options,
                          char *error, size_t error_size) {
    Qwen38SegmentEngine *engine = engine_impl;
    if (!engine || !session_impl || !options || !options->context_tokens ||
        options->context_tokens > engine->context_tokens)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Qwen3.8 Segment session");
    *session_impl = NULL;
    Qwen38SegmentSession *session = calloc(1, sizeof(*session));
    if (!session)
        return coli_segment_adapter_error(error, error_size,
                                           "out of memory creating Qwen3.8 session");
    session->engine = engine;
    session->context_tokens = options->context_tokens;
    size_t count = engine->layer_end - engine->layer_begin;
    session->layers = calloc(count, sizeof(*session->layers));
    Qwen38Config *config = &engine->model.config;
    size_t hyper = (size_t)config->hc_count * config->hidden_size;
    session->first = malloc(hyper * sizeof(float));
    session->second = malloc(hyper * sizeof(float));
    size_t linear = qwen38_linear_layer_workspace_floats(config);
    size_t full = qwen38_full_layer_workspace_floats(config);
    session->workspace_floats = linear > full ? linear : full;
    session->workspace = malloc(session->workspace_floats * sizeof(float));
    if (!session->layers || !session->first || !session->second ||
        !session->workspace) goto oom;
    for (uint32_t index = 0; index < count; index++) {
        Qwen38RuntimeLayer *destination = &session->layers[index];
        Qwen38RuntimeLayer *source = &engine->layers[index];
        *destination = *source;
        if (destination->full_attention) {
            memset(&destination->full.state, 0, sizeof(destination->full.state));
            if (destination->full.has_ple)
                destination->full.ple.conv_state = NULL;
            if (qwen38_attention_state_init(&destination->full.state, config,
                                            options->context_tokens)) goto oom;
            if (destination->full.has_ple &&
                clone_ple(&destination->full.ple, &source->full.ple, config))
                goto oom;
        } else {
            memset(&destination->linear.state, 0, sizeof(destination->linear.state));
            if (destination->linear.has_ple)
                destination->linear.ple.conv_state = NULL;
            if (qwen38_delta_state_init(&destination->linear.state, config)) goto oom;
            if (destination->linear.has_ple &&
                clone_ple(&destination->linear.ple, &source->linear.ple, config))
                goto oom;
        }
    }
    *session_impl = session;
    return 0;
oom:
    destroy_session(session);
    return coli_segment_adapter_error(error, error_size,
                                       "out of memory allocating Qwen3.8 state");
}

static int session_run(void *session_impl, const ColiSegmentRunRequest *request,
                       char *error, size_t error_size) {
    Qwen38SegmentSession *session = session_impl;
    if (!session || !request || request->position != session->position)
        return coli_segment_adapter_error(
            error, error_size, "Qwen3.8 Segment requires contiguous positions");
    Qwen38SegmentEngine *engine = session->engine;
    Qwen38Config *config = &engine->model.config;
    size_t width = (size_t)config->hc_count * config->hidden_size;
    if (!request->token_ids || request->token_count != request->rows ||
        request->position + request->rows > session->context_tokens)
        return coli_segment_adapter_error(error, error_size,
                                           "invalid Qwen3.8 Segment token rows");
    if (request->should_cancel &&
        request->should_cancel(request->cancel_user_data))
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.8 Segment run cancelled");
    pthread_mutex_lock(&engine->run_lock);
    for (uint32_t row = 0; row < request->rows; row++) {
        int32_t token = request->token_ids[row];
        if (token < 0 || token >= config->vocab_size) {
            pthread_mutex_unlock(&engine->run_lock);
            return coli_segment_adapter_error(error, error_size,
                                               "Qwen3.8 Segment token is out of range");
        }
        const float *source = (const float *)request->input + (size_t)row * width;
        memcpy(session->first, source, width * sizeof(float));
        float *input = session->first, *output = session->second;
        size_t count = engine->layer_end - engine->layer_begin;
        for (size_t index = 0; index < count; index++) {
            int layer = (int)(engine->layer_begin + index);
            Qwen38RuntimeLayer *item = &session->layers[index];
            int result = item->full_attention ?
                qwen38_full_layer_step(&engine->model, layer, &item->full,
                    token, input, output, session->workspace,
                    session->workspace_floats, error, error_size) :
                qwen38_linear_layer_step(&engine->model, layer, &item->linear,
                    token, input, output, session->workspace,
                    session->workspace_floats, error, error_size);
            if (result) {
                pthread_mutex_unlock(&engine->run_lock);
                return -1;
            }
            float *swap = input; input = output; output = swap;
        }
        memcpy((float *)request->output + (size_t)row * width, input,
               width * sizeof(float));
    }
    pthread_mutex_unlock(&engine->run_lock);
    session->position += request->rows;
    return 0;
}

static int state_spans(Qwen38SegmentSession *session, uint32_t position,
                       ColiSegmentStateSpan **output, size_t *output_count,
                       char *error, size_t error_size) {
    Qwen38SegmentEngine *engine = session->engine;
    Qwen38Config *config = &engine->model.config;
    size_t layers = engine->layer_end - engine->layer_begin;
    ColiSegmentStateSpan *spans = calloc(layers * 5, sizeof(*spans));
    if (!spans) return coli_segment_adapter_error(
        error, error_size, "out of memory describing Qwen3.8 state");
    size_t count = 0;
    size_t recurrent = (size_t)config->linear_value_heads *
        config->linear_key_dim * config->linear_value_dim;
    size_t conv_dim = 2ULL * config->linear_key_heads * config->linear_key_dim +
        (size_t)config->linear_value_heads * config->linear_value_dim;
    size_t convolution = conv_dim * (config->linear_conv_kernel - 1);
    size_t ple_conv = (size_t)config->hc_count * config->hidden_size *
        (config->ple_conv_kernel - 1) * config->ngram_size;
    for (size_t index = 0; index < layers; index++) {
        Qwen38RuntimeLayer *item = &session->layers[index];
        if (item->full_attention) {
            size_t kv = (size_t)position * config->kv_heads * config->head_dim;
            size_t raw = (size_t)position * config->indexer_head_dim;
            spans[count++] = (ColiSegmentStateSpan){item->full.state.keys,
                                                    kv * sizeof(float)};
            spans[count++] = (ColiSegmentStateSpan){item->full.state.values,
                                                    kv * sizeof(float)};
            spans[count++] = (ColiSegmentStateSpan){item->full.state.raw_index_keys,
                                                    raw * sizeof(float)};
            if (item->full.has_ple) {
                spans[count++] = (ColiSegmentStateSpan){item->full.ple.history,
                    (size_t)(config->ngram_size - 1) * sizeof(int64_t)};
                spans[count++] = (ColiSegmentStateSpan){item->full.ple.conv_state,
                                                        ple_conv * sizeof(float)};
            }
        } else {
            spans[count++] = (ColiSegmentStateSpan){item->linear.state.recurrent,
                                                    recurrent * sizeof(float)};
            spans[count++] = (ColiSegmentStateSpan){item->linear.state.convolution,
                                                    convolution * sizeof(float)};
            if (item->linear.has_ple) {
                spans[count++] = (ColiSegmentStateSpan){item->linear.ple.history,
                    (size_t)(config->ngram_size - 1) * sizeof(int64_t)};
                spans[count++] = (ColiSegmentStateSpan){item->linear.ple.conv_state,
                                                        ple_conv * sizeof(float)};
            }
        }
    }
    *output = spans; *output_count = count;
    return 0;
}

static int session_snapshot(void *session_impl, ColiSegmentWriteFn write_fn,
                            void *write_data, char *error, size_t error_size) {
    Qwen38SegmentSession *session = session_impl;
    ColiSegmentStateSpan *spans = NULL; size_t count = 0, bytes;
    if (!session || state_spans(session, (uint32_t)session->position, &spans,
                                &count, error, error_size) ||
        coli_segment_spans_size(spans, count, &bytes)) {
        free(spans);
        return coli_segment_adapter_error(error, error_size,
                                           "Qwen3.8 snapshot size overflow");
    }
    ColiSegmentSnapshotHeader header;
    coli_segment_snapshot_header_init(&header, "qwen38_flash_next",
        session->engine->layer_begin, session->engine->layer_end,
        session->context_tokens, (uint32_t)session->position, bytes,
        coli_segment_spans_hash(spans, count));
    int result = coli_segment_stream_write(write_fn, write_data, &header,
        sizeof(header), error, error_size);
    if (!result) result = coli_segment_spans_write(
        spans, count, write_fn, write_data, error, error_size);
    free(spans);
    return result;
}

static int session_restore(void *session_impl, ColiSegmentReadFn read_fn,
                           void *read_data, char *error, size_t error_size) {
    Qwen38SegmentSession *session = session_impl;
    ColiSegmentSnapshotHeader header;
    if (!session || coli_segment_stream_read(read_fn, read_data, &header,
        sizeof(header), error, error_size)) return -1;
    ColiSegmentStateSpan *spans = NULL; size_t count = 0, bytes;
    if (state_spans(session, header.position, &spans, &count,
                    error, error_size) ||
        coli_segment_spans_size(spans, count, &bytes) ||
        coli_segment_snapshot_header_valid(&header, "qwen38_flash_next",
            session->engine->layer_begin, session->engine->layer_end,
            session->context_tokens, bytes, error, error_size)) {
        free(spans); return -1;
    }
    int result = coli_segment_spans_restore(spans, count, header.payload_hash,
        read_fn, read_data, error, error_size);
    free(spans);
    if (!result) {
        session->position = header.position;
        size_t layers = session->engine->layer_end - session->engine->layer_begin;
        for (size_t index = 0; index < layers; index++)
            if (session->layers[index].full_attention)
                session->layers[index].full.state.length = header.position;
    }
    return result;
}

static const ColiSegmentAdapter adapter = {
    sizeof(ColiSegmentAdapter), COLI_SEGMENT_ABI_VERSION,
    "qwen38_flash_next", engine_open, destroy_engine, session_create,
    destroy_session, session_run, session_snapshot, session_restore, {0}
};

int coli_qwen38_segment_adapter_register(void) {
    return coli_segment_adapter_register(&adapter);
}
