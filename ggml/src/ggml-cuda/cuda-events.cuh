// Mapped-memory barrier async eval callback infrastructure (ALPHA_ASYNC_CALLBACK).
// Uses cudaHostAllocMapped so CPU can signal GPU via plain memory store.
// No DMA engine in signal path - cannot deadlock.
#pragma once

#ifdef GGML_ASYNC_EVAL_CALLBACK

#include <cuda_runtime.h>
#include <stdint.h>

#define GGML_CUDA_ASYNC_MAX_EVENTS 128

struct ggml_tensor;

static inline uint32_t ggml_async_hash(const char * s) {
    uint32_t h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h;
}

struct ggml_cuda_async_event_ctx {
    // Record events: fired by compute stream after argsort completes
    cudaEvent_t  node_events[GGML_CUDA_ASYNC_MAX_EVENTS];
    bool         event_recorded[GGML_CUDA_ASYNC_MAX_EVENTS];

    // Hash-based matching (compute loop <-> graph nodes)
    uint32_t     node_hashes[GGML_CUDA_ASYNC_MAX_EVENTS];
    uint32_t     barrier_hashes[GGML_CUDA_ASYNC_MAX_EVENTS];
    const ggml_tensor * nodes[GGML_CUDA_ASYNC_MAX_EVENTS];

    // Mapped memory barrier flags (host + GPU point to the same physical memory)
    int *        h_barrier_flags;    // host pointer (cudaHostAllocMapped)
    int *        d_barrier_flags;    // GPU pointer (cudaHostGetDevicePointer)

    // Callback stream: all callback GPU ops go here (not default stream)
    cudaStream_t callback_stream;

    int  n_events;
    bool compute_active;  // true only during async graph compute
    bool initialized;
};

#ifdef __cplusplus
extern "C" {
#endif

void ggml_cuda_async_event_ctx_init(ggml_cuda_async_event_ctx * ctx);
void ggml_cuda_async_event_ctx_free(ggml_cuda_async_event_ctx * ctx);
void ggml_cuda_async_event_ctx_reset(ggml_cuda_async_event_ctx * ctx);

int ggml_cuda_async_event_mark_node(
    ggml_cuda_async_event_ctx * ctx,
    const ggml_tensor * node,
    const ggml_tensor * barrier_node
);

bool ggml_cuda_async_event_query(ggml_cuda_async_event_ctx * ctx, int event_idx);
void ggml_cuda_async_event_synchronize(ggml_cuda_async_event_ctx * ctx, int event_idx);

// Signal done: plain CPU store to h_barrier_flags[event_idx] - no CUDA API
void ggml_cuda_async_event_signal_done(ggml_cuda_async_event_ctx * ctx, int event_idx);

// Launch mapped-barrier kernel on compute_stream at barrier point
void ggml_cuda_async_event_launch_barrier(ggml_cuda_async_event_ctx * ctx, int event_idx, cudaStream_t compute_stream);

// Get callback_stream for use in poll thread
cudaStream_t ggml_cuda_async_event_get_callback_stream(ggml_cuda_async_event_ctx * ctx);

// Ensure callback_stream sees data from compute stream for event i
void ggml_cuda_async_event_sync_for_read(ggml_cuda_async_event_ctx * ctx, int event_idx);

#ifdef __cplusplus
}
#endif

#endif // GGML_ASYNC_EVAL_CALLBACK
