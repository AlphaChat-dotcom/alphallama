// SPDX-License-Identifier: Apache-2.0
#pragma once

// FlashInfer GPU Engine
//
// Drop-in replacement for gpu_engine that uses ForwardEngine + paged KV
// instead of llama_decode. Same SPSC queue interface - scheduler doesn't
// need to know which engine is running.
//
// Processes BatchRequest -> forward_decode -> greedy sample -> BatchResult.
// Pre-allocates CUDA buffers per max_batch to avoid per-request cudaMalloc.

#ifdef ALPHA_HAS_FLASHINFER

#include "serve-types.h"
#include "serve-spsc-queue.h"
#include "serve-forward.h"
#include "serve-paged-kv.h"
#include "serve-kv-cache.h"
#include "llama.h"

#include <atomic>
#include <vector>
#include <cstdint>

namespace alphaserve {

class fi_gpu_engine {
public:
    fi_gpu_engine() = default;
    ~fi_gpu_engine();

    // Initialize with forward engine, KV pool, and queue references
    bool init(ForwardEngine * engine,
              PagedKVPool * kv_pool,
              const llama_vocab * vocab,
              spsc_queue<BatchRequest, 64> & in_queue,
              spsc_queue<BatchResult, 64> & out_queue,
              int max_batch);

    // Process one batch synchronously (called from main thread)
    bool process_one();

    // Stats
    int64_t total_decodes() const { return n_decodes_; }
    int64_t total_tokens() const { return n_tokens_; }

private:
    ForwardEngine * engine_ = nullptr;
    PagedKVPool * kv_pool_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    spsc_queue<BatchRequest, 64> * in_ = nullptr;
    spsc_queue<BatchResult, 64> * out_ = nullptr;

    // Pre-allocated CUDA buffers (fixes per-request cudaMalloc race)
    int32_t * d_tokens_ = nullptr;      // [max_batch]
    int32_t * d_positions_ = nullptr;   // [max_batch]
    float * d_logits_ = nullptr;        // [max_batch, vocab_size]
    int32_t * d_sampled_ids_ = nullptr; // [max_batch] GPU argmax output
    int max_batch_ = 0;

    // Pinned host buffers for fast H2D transfers
    int32_t * h_tokens_pinned_ = nullptr;    // [max_batch] pinned
    int32_t * h_positions_pinned_ = nullptr; // [max_batch] pinned
    int32_t * h_sampled_pinned_ = nullptr;   // [max_batch] pinned (GPU writes directly)

    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_event_ = nullptr;

    // Redis KV cache (optional, set by serve-main if --redis-kv)
    KVCacheManager * kv_cache_ = nullptr;

public:
    void set_kv_cache(KVCacheManager * kvc) { kv_cache_ = kvc; }

private:
    // Stats
    int64_t n_decodes_ = 0;
    int64_t n_tokens_ = 0;
};

} // namespace alphaserve

#endif // ALPHA_HAS_FLASHINFER
