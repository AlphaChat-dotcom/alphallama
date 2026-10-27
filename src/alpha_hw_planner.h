#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <cstddef>

// Device Info

enum alpha_mem_type {
    ALPHA_MEM_DISCRETE = 0,  // dedicated VRAM (RTX, Radeon, MX550)
    ALPHA_MEM_UNIFIED  = 1,  // shared with system (Apple Silicon, Intel iGPU)
    ALPHA_MEM_SYSTEM   = 2,  // CPU RAM
};

struct alpha_device_info {
    ggml_backend_dev_t dev;
    int index;                    // position in ggml device list
    const char * name;            // short name
    const char * description;     // human-readable
    const char * backend_name;    // "CUDA", "Vulkan", "Metal", "CPU"
    enum alpha_mem_type mem_type;
    size_t mem_free_mb;
    size_t mem_total_mb;
    size_t bw_local_mbs;          // estimated device-local bandwidth (MB/s)
    size_t bw_xfer_mbs;           // estimated cross-device transfer bandwidth (MB/s)
    bool supports_async;
};

// Unified Strategy
// Single-pass decision from HW planner. No runtime fallbacks.

enum alpha_hw_strategy {
    ALPHA_HW_DIRECT = 0,           // model fits entirely in GPU
    ALPHA_HW_DIRECT_Q4KV,          // model fits with Q4 KV cache
    ALPHA_HW_LUT_DIRECT,           // MoE: K >= K_min, LUT-direct
    ALPHA_HW_SELF_SPEC,            // MoE: K in draft range, self-spec (LUT draft + cpu-moe verify)
    ALPHA_HW_CPU_MOE,              // MoE: K < K_draft, pure cpu-moe
    ALPHA_HW_PARTIAL_OFFLOAD,      // dense: some layers GPU, some CPU
    ALPHA_HW_SELF_SPEC_DENSE,      // dense: layer-skip draft on GPU + CPU verify (Alpha Optimized)
    ALPHA_HW_PIPELINE,             // backbone > GPU: multi-pass pipeline
    ALPHA_HW_CPU_ONLY,             // no GPU
};

// Hardware Plan

#define ALPHA_HW_MAX_DEVICES 8

struct alpha_hw_plan {
    // Device list (NULL-terminated, sorted best-first)
    ggml_backend_dev_t devices[ALPHA_HW_MAX_DEVICES + 1];
    int n_devices;

    // Main model placement
    int main_device_idx;          // index into devices[]
    int n_gpu_layers;             // layers to offload (-1 = all)
    enum llama_split_mode split_mode;
    float tensor_split[ALPHA_HW_MAX_DEVICES];  // per-device proportions

    // Expert cache (MoE models)
    int expert_cache_mb;          // 0 = not needed
    int expert_resident_device;   // where experts live when cached (-1 = CPU)

    // KV cache
    enum ggml_type kv_type_k;
    enum ggml_type kv_type_v;
    int n_ctx;                    // recommended context window (0 = no override)

    // Training mode (for train commands)
    enum alpha_train_hw_mode {
        ALPHA_TRAIN_HW_DIRECT = 0,    // model fits in VRAM: standard decode
        ALPHA_TRAIN_HW_PIPELINE = 1,  // model too large: stream layers via tensor_buft_overrides
    } train_mode;

    // Capacity check
    bool can_run;                 // false if total system capacity < model size
    size_t total_capacity_mb;     // aggregate available (GPU + RAM + disk)

    // Expert cache control
    bool no_gpu_expert_cache;     // true = experts on CPU only (no GPU cache)

    // Unified strategy (single-pass decision)
    alpha_hw_strategy strategy;

    // MoE K estimation
    int K_estimate;               // estimated cache K from VRAM budget
    int K_min;                    // minimum K for reliable quality
    int K_draft;                  // minimum K for useful draft tokens

    // Dense self-spec (Alpha Optimized), active when strategy == ALPHA_HW_SELF_SPEC_DENSE
    int self_spec_skip;            // layers to skip for draft (0 = disabled)
    int self_spec_m;               // draft tokens per cycle (default 2)
    int self_spec_draft_ngl;       // GPU layers for draft model

    // Strategy metadata (internal, not shown to user)
    const char * strategy_name;
    float est_tokens_per_sec;

    // Model info (captured during planning)
    size_t model_size_mb;
    int n_layer;
    int n_embd;
    int n_ctx_train;              // model's training context length (0 = unknown)
    int sliding_window;           // >0 => iswa (Gemma): only global layers' KV scales with n_ctx
    size_t kv_f16_bpt_layer;      // KV bytes/token/layer at F16 (accurate from attn geometry)
    int n_expert;
    int n_expert_used;
    bool is_moe;
    char arch[64];
    size_t gpu_vram_mb;           // primary GPU free VRAM (0 = no GPU)
    size_t gpu_vram_total_mb;     // primary GPU total VRAM
    size_t system_ram_mb;         // available system RAM (MemAvailable on Linux)
};

// API

// Create a hardware execution plan for the given model.
// model_path: path to GGUF file (read metadata only, no full load)
// user_ngl: -1 for auto, else user-specified GPU layer count
struct alpha_hw_plan alpha_hw_plan_create(
    const char * model_path,
    int user_ngl);

// Print plan decisions to stderr (human-readable summary)
void alpha_hw_plan_print(const struct alpha_hw_plan * plan);

// Compute optimal KV-chunk prefill parallelism from available VRAM.
// Returns max number of chunks that can be prefilled simultaneously.
// chunk_size: tokens per chunk (default 4096)
// kv_per_token_kb: KV cache bytes per token in KB (model-specific)
inline int alpha_hw_compute_prefill_parallel(
    const struct alpha_hw_plan * plan,
    int chunk_size = 4096,
    float kv_per_token_kb = 20.5f) {
    if (!plan || plan->gpu_vram_total_mb == 0) return 1;
    float overhead_gb = 0.57f;  // recurrent state + CUDA overhead
    float model_gb = (float)plan->model_size_mb / 1024.0f;
    float total_gb = (float)plan->gpu_vram_total_mb / 1024.0f;
    float free_gb = total_gb - model_gb - overhead_gb;
    if (free_gb <= 0) return 1;
    float kv_per_chunk_gb = (chunk_size * kv_per_token_kb) / (1024.0f * 1024.0f);
    if (kv_per_chunk_gb <= 0) return 1;
    int max_parallel = (int)(free_gb / kv_per_chunk_gb * 0.85f);
    return max_parallel < 1 ? 1 : max_parallel;
}
