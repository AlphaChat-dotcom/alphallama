// Async eval callback built on mapped-memory barriers.
// The signal is a plain CPU store to mapped host memory that the GPU polls over PCIe;
// the signal path uses no DMA engine, stream or CUDA API call.
#ifdef GGML_ASYNC_EVAL_CALLBACK

#include "cuda-events.cuh"
#include "ggml.h"
#include <cstring>
#include <cstdio>
#ifdef _MSC_VER
#include <intrin.h>
#include <windows.h>
#endif

// Barrier kernel: one thread spins until flag >= target.
// Each read is a PCIe load of mapped host memory (~100 ns); a volatile read is sufficient, no atomics needed.
__global__ void ggml_mapped_barrier_kernel(volatile int * flag, int target) {
    while (*flag < target) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(50);  // yield between PCIe reads
#endif
    }
}

void ggml_cuda_async_event_ctx_init(ggml_cuda_async_event_ctx * ctx) {
    memset(ctx, 0, sizeof(*ctx));
    for (int i = 0; i < GGML_CUDA_ASYNC_MAX_EVENTS; i++) {
        cudaEventCreateWithFlags(&ctx->node_events[i], cudaEventDisableTiming);
    }

    // Mapped memory: host-allocated, GPU-visible via PCIe
    cudaError_t err = cudaHostAlloc(
        &ctx->h_barrier_flags,
        GGML_CUDA_ASYNC_MAX_EVENTS * sizeof(int),
        cudaHostAllocMapped
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "[ASYNC] FATAL: cudaHostAlloc mapped failed: %s\n",
            cudaGetErrorString(err));
        return;
    }
    memset(ctx->h_barrier_flags, 0, GGML_CUDA_ASYNC_MAX_EVENTS * sizeof(int));

    // GPU-visible pointer to the same physical memory
    err = cudaHostGetDevicePointer(&ctx->d_barrier_flags, ctx->h_barrier_flags, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ASYNC] FATAL: cudaHostGetDevicePointer failed: %s\n",
            cudaGetErrorString(err));
        return;
    }
    fprintf(stderr, "[ASYNC] Mapped memory barrier: host=%p gpu=%p\n",
        (void *)ctx->h_barrier_flags, (void *)ctx->d_barrier_flags);

    // Non-blocking callback stream, used for expert DMA, not for the signal
    cudaStreamCreateWithFlags(&ctx->callback_stream, cudaStreamNonBlocking);
    ctx->n_events = 0;
    ctx->compute_active = false;
    ctx->initialized = true;
}

void ggml_cuda_async_event_ctx_free(ggml_cuda_async_event_ctx * ctx) {
    if (!ctx || !ctx->initialized) return;
    for (int i = 0; i < GGML_CUDA_ASYNC_MAX_EVENTS; i++) {
        cudaEventDestroy(ctx->node_events[i]);
    }
    // Both h_barrier_flags and d_barrier_flags point to same allocation
    if (ctx->h_barrier_flags) cudaFreeHost(ctx->h_barrier_flags);
    if (ctx->callback_stream) cudaStreamDestroy(ctx->callback_stream);
    ctx->initialized = false;
}

void ggml_cuda_async_event_ctx_reset(ggml_cuda_async_event_ctx * ctx) {
    if (!ctx) return;
    ctx->n_events = 0;
    ctx->compute_active = false;
    memset(ctx->event_recorded, 0, sizeof(ctx->event_recorded));
    // Reset barrier flags: plain host memset (no CUDA API needed)
    if (ctx->h_barrier_flags) {
        memset(ctx->h_barrier_flags, 0, GGML_CUDA_ASYNC_MAX_EVENTS * sizeof(int));
#ifdef _MSC_VER
        _ReadWriteBarrier(); MemoryBarrier();
#else
        __sync_synchronize();
#endif
        // full memory barrier: the GPU sees zeros before the graph starts
    }
}

// Translate "ffn_moe_topk-N" -> "ffn_moe_argsort-N" for CUDA compute loop matching.
static uint32_t ggml_async_translate_hash(const char * name, const char * from, const char * to) {
    if (strncmp(name, from, strlen(from)) == 0) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s%s", to, name + strlen(from));
        return ggml_async_hash(buf);
    }
    return ggml_async_hash(name);
}

int ggml_cuda_async_event_mark_node(
    ggml_cuda_async_event_ctx * ctx,
    const ggml_tensor * node,
    const ggml_tensor * barrier_node
) {
    if (!ctx || ctx->n_events >= GGML_CUDA_ASYNC_MAX_EVENTS) return -1;
    int idx = ctx->n_events++;
    ctx->node_hashes[idx] = ggml_async_translate_hash(node->name, "ffn_moe_topk", "ffn_moe_argsort");
    ctx->barrier_hashes[idx] = barrier_node
        ? ggml_async_translate_hash(barrier_node->name, "ffn_moe_weights", "ffn_moe_gate")
        : 0;
    ctx->nodes[idx] = node;
    ctx->event_recorded[idx] = false;
#ifdef GGML_ASYNC_DEBUG
    fprintf(stderr, "[ASYNC] mark %d: %s -> record_hash=%u, barrier_hash=%u\n",
        idx, node->name, ctx->node_hashes[idx], ctx->barrier_hashes[idx]);
#endif
    return idx;
}

bool ggml_cuda_async_event_query(ggml_cuda_async_event_ctx * ctx, int event_idx) {
    if (!ctx || event_idx < 0 || event_idx >= ctx->n_events) return false;
    if (!ctx->event_recorded[event_idx]) return false;
    return cudaEventQuery(ctx->node_events[event_idx]) == cudaSuccess;
}

void ggml_cuda_async_event_synchronize(ggml_cuda_async_event_ctx * ctx, int event_idx) {
    if (!ctx || event_idx < 0 || event_idx >= ctx->n_events) return;
    cudaEventSynchronize(ctx->node_events[event_idx]);
}

void ggml_cuda_async_event_signal_done(
    ggml_cuda_async_event_ctx * ctx,
    int event_idx
) {
    if (!ctx || event_idx < 0 || event_idx >= ctx->n_events) return;
    // Plain CPU store to mapped memory. GPU sees this on next volatile read.
    // No DMA engine, no stream, no CUDA API.
    ctx->h_barrier_flags[event_idx] = 1;
#ifdef _MSC_VER
    _ReadWriteBarrier(); MemoryBarrier();
#else
    __sync_synchronize();
#endif
    // ensure store is visible to GPU via PCIe
#ifdef GGML_ASYNC_DEBUG
    static int sig_log = 0;
    if (sig_log < 5) { fprintf(stderr, "[ASYNC] SIGNAL event %d (mapped store)\n", event_idx); sig_log++; }
#endif
}

void ggml_cuda_async_event_launch_barrier(
    ggml_cuda_async_event_ctx * ctx,
    int event_idx,
    cudaStream_t compute_stream
) {
    if (!ctx || event_idx < 0 || event_idx >= ctx->n_events) return;
    // Launch 1-thread barrier kernel on compute stream.
    // Reads mapped host memory via PCIe until flag >= 1.
    ggml_mapped_barrier_kernel<<<1, 1, 0, compute_stream>>>(
        &ctx->d_barrier_flags[event_idx], 1
    );
}

cudaStream_t ggml_cuda_async_event_get_callback_stream(ggml_cuda_async_event_ctx * ctx) {
    if (!ctx) return nullptr;
    return ctx->callback_stream;
}

void ggml_cuda_async_event_sync_for_read(ggml_cuda_async_event_ctx * ctx, int event_idx) {
    if (!ctx || event_idx < 0 || event_idx >= ctx->n_events) return;
    if (!ctx->event_recorded[event_idx]) return;
    // Make callback_stream wait for compute stream data to be visible
    cudaStreamWaitEvent(ctx->callback_stream, ctx->node_events[event_idx], 0);
}

#endif // GGML_ASYNC_EVAL_CALLBACK
