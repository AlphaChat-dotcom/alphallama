#pragma once

// AlphaLlama: Smart KV Paging
//
// Extends effective context beyond VRAM by paging cold KV entries to CPU RAM.
// Uses a compressed attention index for fast approximate scoring to identify
// which old tokens are relevant without loading full KV from CPU.
//
// Architecture:
//   GPU VRAM: Hot KV (last N tokens, full precision) + Compressed Index
//   CPU RAM:  Cold KV (older tokens, INT4 quantized)
//
// Per new token:
//   1. Full attention on hot KV (GPU)           → ~0.5ms
//   2. Approximate scoring on compressed index   → ~0.1ms
//   3. Identify top-K important cold tokens      → ~0.05ms
//   4. Fetch K cold KV entries from CPU          → ~0.8ms
//   5. Full attention on hot + retrieved          → ~0.5ms
//   Total: ~2ms (vs 3,000ms for naive full-context CPU offload)

#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <vector>

struct alpha_kv_pager;

struct alpha_kv_pager_params {
    int      gpu_budget;       // max tokens to keep on GPU (hot window)
    int      cpu_budget;       // max tokens to keep in CPU RAM (cold store)
    int      retrieval_k;      // number of cold tokens to retrieve per step
    float    evict_threshold;  // importance threshold for eviction (0.0 = evict all old)
};

// Statistics for monitoring
struct alpha_kv_pager_stats {
    int      hot_tokens;       // tokens currently on GPU
    int      cold_tokens;      // tokens currently on CPU
    int      total_tokens;     // total tracked
    int      evictions;        // total eviction events
    int      retrievals;       // total retrieval events
    float    hit_rate;         // fraction of attention served from GPU
};

// Create a KV pager
// Call after llama_context is created
LLAMA_API alpha_kv_pager * alpha_kv_pager_init(
        struct llama_context * ctx,
        alpha_kv_pager_params   params);

// Called after each llama_decode() to manage KV cache
// Evicts cold entries if GPU budget exceeded, builds/updates index
LLAMA_API void alpha_kv_pager_update(
        alpha_kv_pager * pager,
        int              n_past);    // number of tokens processed so far

// Get paging statistics
LLAMA_API alpha_kv_pager_stats alpha_kv_pager_get_stats(
        const alpha_kv_pager * pager);

// Print paging status
LLAMA_API void alpha_kv_pager_print_stats(
        const alpha_kv_pager * pager);

// Free pager
LLAMA_API void alpha_kv_pager_free(alpha_kv_pager * pager);
