// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaLlama: Internal MoE optimizer structures
//
// Shared between alpha_mole_moe_optimizer.cpp and alpha_mole_speculative.cpp.
// Not part of the public API - these are implementation details.

#include "alpha_mole_moe_optimizer.h"
#include "alpha_mole_paged_experts.h"
#include "llama-hparams.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"

#include <vector>
#include <cstdint>

// Expert tensor types
enum expert_op_type { OP_GATE = 0, OP_UP = 1, OP_DOWN = 2, OP_GATE_UP = 3, OP_COUNT = 4 };

struct moe_layer_info {
    ggml_tensor * orig_exps[OP_COUNT];  // original CPU expert tensors
    bool has_experts;
    bool on_gpu;
    size_t expert_slice_bytes[OP_COUNT]; // bytes per expert per op
};

// Contiguous GPU expert cache: one tensor [dim1, dim2, K] per op per layer
struct expert_cache_slot {
    int expert_id;    // which expert is in this slot (-1 = empty)
    int64_t last_used; // LRU timestamp
};

struct layer_cache {
    ggml_tensor * gpu_tensor[OP_COUNT];  // contiguous GPU tensors [dim1, dim2, K]
    std::vector<expert_cache_slot> slots[OP_COUNT]; // K slots per op
    int slot_map[OP_COUNT][LLAMA_MAX_EXPERTS]; // expert_id -> slot (-1 = not cached)
};

struct alpha_moe_optimizer {
    llama_model   * model;
    llama_context * ctx;
    alpha_moe_optimizer_config config;

    bool   is_moe;
    int    n_layer;
    int    n_expert;
    int    n_expert_used;
    int    n_embd;
    int    K;  // cache slots per op (tiered: per-layer for down, global for gate/up)

    std::vector<moe_layer_info> layer_info;
    std::vector<layer_cache> caches;

    int total_tokens;
    int64_t cache_clock;
    int64_t cache_hits;
    int64_t cache_misses;

    ggml_context * cache_ctx;
    ggml_context * remap_ctx;  // keeps pre-computed ID tensors alive
    std::vector<ggml_backend_buffer_ptr> cache_bufs;
    bool cache_enabled;
    bool eval_callback_active;  // true if eval callback was set on context (false for cpu-moe)

    // GPU device/buft saved from init for deferred cache allocation
    ggml_backend_dev_t gpu_dev;
    ggml_backend_buffer_type_t gpu_buft;

    // Tiered cache: global gate/up cache (shared across layers) + per-layer down cache
    bool tiered_cache;                        // true when using tiered mode (disabled: produces garbled output)
    bool full_dim_cache;                      // true: cache ne[2]=n_expert (for qwen3next kernel compat)
    layer_cache global_gate_up_cache;         // single global cache for OP_GATE, OP_UP, OP_GATE_UP
    std::vector<ggml_tensor *> cache_expert_ids_down;  // [n_layer] per-layer down IDs

    // Scratch cache: down per-layer (K slots LRU) + gate/up scratch (K slots, refreshed per callback)
    bool use_scratch;              // true = down-only cache + gate/up scratch
    ggml_tensor * gate_scratch;    // [dim1, dim2, K] shared across layers, only slots 0-5 written
    ggml_tensor * up_scratch;      // [dim1, dim2, K] shared across layers, only slots 0-5 written

    // Expert override: bypass router selection for specific layers
    // When override_experts[layer] is non-empty, use those IDs instead of argsort
    std::vector<std::vector<int>> override_experts;  // [n_layer][n_expert_used] or empty
    std::vector<std::vector<float>> override_weights; // [n_layer][n_expert_used] or empty

    // Reactive cache: counter reset before each decode, incremented per MoE layer
    int reactive_layer_counter;
    bool speculative_pass;  // true = skip reactive callback (prediction-only, fast)
    bool draft_mode;        // true = self-speculative draft (Mode A or C)
    int  draft_expert_count; // Mode A: reduced expert count (e.g. 3), 0 = Mode C (backbone only)
    bool has_shared_experts; // auto-detected: model has ffn_*_shexp tensors
    int split_group;        // sync every N MoE layers (1 = every layer)
    bool prompt_mode;       // true during batched prefill (forces SG=1)

    // Pre-computed cache expert IDs per layer (written reactively, used by graph)
    std::vector<ggml_tensor *> cache_expert_ids;  // [n_layer] pointers to I32 tensors

    // Previous token's routing and difficulty for prediction
    std::vector<std::vector<int>> prev_routing;  // [n_layer][n_expert_used]
    std::vector<float> prev_entropy;             // [n_layer] routing entropy (0=easy, 1=hard)

    // Thermal-aware mobile expert loading
    alpha_moe_thermal_policy thermal_policy;
    float current_temp;

    // Expert tier classification (computed from activation profiling)
    std::vector<int> hot_experts;                   // global hot expert IDs
    std::vector<std::vector<int64_t>> expert_activation_count; // [n_layer][n_expert]
    std::vector<std::vector<int>> hot_per_layer;    // [n_layer][n_hot] -- top-N per layer
    int hot_update_interval;                        // update hot list every N tokens

    // Cross-layer prediction: layer N's actual routing predicts layer N+1
    std::vector<std::vector<int>> cross_layer_pred; // [n_layer][n_expert_used] -- from prev layer

    // Temporal locality: track last few tokens' routing for eviction protection
    static constexpr int TEMPORAL_DEPTH = 3;
    std::vector<std::vector<std::vector<int>>> recent_routing; // [n_layer][TEMPORAL_DEPTH][n_expert_used]
    int recent_idx;  // circular buffer index
    int64_t thermal_skipped;                        // tokens where cold experts were skipped
    int64_t thermal_substituted;                    // times a hot expert was substituted for cold

    // Inter-layer prefetch
    bool prefetch_enabled;
    int  prefetch_depth;
    int64_t prefetch_hits;     // prefetched expert was actually needed
    int64_t prefetch_total;    // total prefetch attempts

    // MoE layer index mapping (moe_idx -> actual layer index)
    std::vector<int> moe_layer_indices;
    int n_moe_layers;

    // Sticky routing momentum: EMA of recent expert activations per layer
    std::vector<std::vector<float>> expert_momentum;  // [n_layer][n_expert]

    // Expert grouping: cluster experts for contiguous DMA
    struct expert_group {
        int group_id;
        std::vector<int> expert_ids;
        int64_t last_used;          // LRU timestamp for group-level eviction
    };
    struct layer_groups {
        std::vector<expert_group> groups;       // all groups for this layer
        std::vector<int> expert_to_group;       // expert_id -> group_id mapping [n_expert]
    };
    std::vector<layer_groups> expert_grouping;  // [n_layer] - empty if no grouping
    bool groups_enabled;
    int64_t group_cache_hits;
    int64_t group_cache_misses;

    // Prediction reliability tracking
    // With many experts (e.g. 256), prev_routing rarely matches the full selected set.
    // When prediction is unreliable, prepare() skips DMA to avoid evicting hot experts.
    bool prediction_reliable;   // true = prev_routing works, prepare() does DMA
    int64_t pred_set_matches;   // total layers where predicted set matched actual
    int64_t pred_set_total;     // total layers checked

    // Cached-expert self-speculative decoding
    bool spec_draft_active;             // true during draft phase (no callback)
    bool profiles_loaded;               // true after first load_style_profiles attempt (skip per-request I/O)
    // GPU LUT: per-layer [1, n_expert] I32 tensor on GPU
    // lut[expert_id] = cache_slot if cached, 0 (zero expert) if not
    // Used by ggml_get_rows in build_moe_ffn to remap argsort -> cache slots on GPU
    std::vector<ggml_tensor *> spec_lut;  // [n_layer] I32 tensors
    ggml_context * spec_lut_ctx;
    bool spec_lut_ready;
    // Stats
    int64_t spec_draft_tokens;
    int64_t spec_verify_tokens;
    int64_t spec_accepted_tokens;

    // Mixed expert clustering
    // Cluster 256 experts -> N mixed experts per layer. All mixed experts fit in GPU.
    // Uses existing cache tensors (K = n_mixed+1) with cluster-representative data.
    // LUT maps all 256 expert IDs -> cluster slot (1-based, 0=zero expert).
    bool mixed_experts_enabled;
    int  mixed_n_clusters;                                    // e.g. 24
    std::vector<std::vector<int32_t>> mixed_expert_to_cluster; // [n_layer][256] -> cluster_id (0-based)
    std::vector<std::vector<int>>     mixed_cluster_rep;       // [n_layer][n_clusters] representative expert_id
    std::vector<std::vector<std::vector<int>>> mixed_cluster_members; // [n_layer][n_clusters][variable] member expert IDs
    std::vector<ggml_tensor *> mixed_lut;                      // [n_layer] GPU LUT [1, 256] I32

    // Saved original expert tensor pointers (restored when exiting mixed draft)
    struct saved_expert_ptrs {
        ggml_tensor * gate;
        ggml_tensor * up;
        ggml_tensor * down;
        ggml_tensor * gate_up;
        ggml_tensor * cache_ids;
        ggml_tensor * cache_ids_down;
    };
    std::vector<saved_expert_ptrs> mixed_saved_ptrs;           // [n_layer]
    bool callback_was_enabled;  // true if eval callback was set before entering mixed draft

    // Per-layer preload expert IDs (from frequency profiling)
    // If non-empty, alloc_cache preloads these instead of sequential 0..K-2
    std::vector<std::vector<int>> preload_expert_ids; // [n_layer][up to K]

    // CPU-MoE verify: when true, cached_spec_generate uses cpu-moe for batch verify
    // instead of reactive cache. Needed when K < K_min (cache has insufficient experts).
    bool cpu_moe_verify;

    // Pinned staging buffers for truly async DMA
    // cudaMemcpyAsync from mmap'd memory is secretly synchronous.
    // Stage through pinned buffers: memcpy mmap->pinned (fast CPU), then
    // cudaMemcpyAsync pinned->GPU (truly non-blocking).
    void ** staging_bufs;           // array of cudaMallocHost'd pinned buffers
    int    staging_buf_count;       // number of buffers allocated
    size_t staging_buf_size;        // size of each buffer (= max expert slice, 4KB aligned)
    int    staging_write_idx;       // next buffer to use (reset per callback)
    bool   staging_enabled;         // false if alloc fails or non-CUDA

    // Per-token cache miss tracking (for miss-rate-based quality resync)
    int last_token_miss_layers;     // MoE layers with at least one cache miss this token
    int last_token_total_layers;    // total MoE layers processed this token

    // LUT prefill mode: callback tracks misses but skips DMA
    bool lut_prefill_active;

    // Paged expert loading
    // Splits expert weights into 64KB pages with LRU eviction.
    // On cache miss: only load missing pages instead of full expert.
    // page_tables[op] = page table for gate/up/down operations.
    bool paged_experts_enabled = false;
    expert_page_table page_tables[OP_COUNT];
    ggml_backend_buffer_ptr paged_vram_buf;  // VRAM allocation for page pools

    // Hybrid LUT+CPU expert execution
    // On cache miss: remap to slot 0 (zeros on GPU), queue expert for CPU compute.
    // After GPU forward pass completes, add CPU-computed results to output.
    // Eliminates DMA sync overhead for uncached experts.
    bool hybrid_cpu_enabled = false;
    int64_t hybrid_cpu_computes = 0;   // number of experts computed on CPU
    int64_t hybrid_gpu_computes = 0;   // number of experts computed on GPU (cache hits)
};
