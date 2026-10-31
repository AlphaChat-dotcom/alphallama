// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaLlama: Expert LRU Cache for MoE Models
//
// MoE models (Qwen3.5-122B) have 64 experts per layer but only activate 4.
// Loading all experts from CPU per token wastes bandwidth. This cache keeps
// recently-used expert weight slices on GPU and evicts cold ones via LRU.
//
// Architecture:
//   - Full expert tensors live on CPU (too large for GPU)
//   - GPU cache holds N expert slots per MoE layer
//   - Per token: check which experts are needed, load misses from CPU
//   - 75-80% hit rate expected (expert routing is sticky)
//
// The cache operates at the WEIGHT SLICE level: each slot holds one expert's
// worth of weights for one FFN component (gate_up or down) in one layer.

#include "llama.h"
#include "ggml.h"

#include <cstdint>

#define MOLE_EXPERT_CACHE_MAX_LAYERS 128
#define MOLE_EXPERT_CACHE_MAX_SLOTS  4096

struct alpha_expert_cache;

struct alpha_expert_cache_config {
    size_t vram_budget_mb;        // GPU memory budget for expert cache
    int    n_moe_layers;          // number of layers with MoE
    int    n_experts_per_layer;   // total experts per layer (e.g., 64)
    int    n_experts_used;        // experts used per token (e.g., 4)
    size_t expert_size_bytes;     // size of one expert's weights
};

struct alpha_expert_cache_stats {
    int64_t hits;
    int64_t misses;
    int64_t evictions;
    int     slots_used;
    int     slots_total;
    float   hit_rate;
};

// Create expert cache
LLAMA_API alpha_expert_cache * alpha_expert_cache_init(
        alpha_expert_cache_config config);

// Check if an expert is cached, load if not (returns cache slot index)
// layer_id: which MoE layer
// expert_id: which expert within the layer
// cpu_tensor: the full 3D expert tensor on CPU [dim, dim, n_expert]
// gpu_tensor: the GPU-side 3D tensor to populate (same shape, GPU buffer)
LLAMA_API bool alpha_expert_cache_ensure(
        alpha_expert_cache * cache,
        int layer_id,
        int expert_id,
        const struct ggml_tensor * cpu_tensor,
        struct ggml_tensor * gpu_tensor);

// Prefetch: hint that these experts will be needed soon
LLAMA_API void alpha_expert_cache_prefetch(
        alpha_expert_cache * cache,
        int layer_id,
        const int * expert_ids,
        int n_experts);

// Get cache statistics
LLAMA_API alpha_expert_cache_stats alpha_expert_cache_get_stats(
        const alpha_expert_cache * cache);

// Print cache stats
LLAMA_API void alpha_expert_cache_print_stats(
        const alpha_expert_cache * cache);

// Free cache
LLAMA_API void alpha_expert_cache_free(alpha_expert_cache * cache);
