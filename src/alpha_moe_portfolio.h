#pragma once

// Model portfolio: selects the MoE optimization strategy from model characteristics
// (architecture, expert count, VRAM fit, hybrid memory). Each model family has its own profile, so
// tuning one family does not change another; adding a model series means adding a profile entry.
// CLI flags override portfolio defaults.

#include "llama.h"
#include "ggml.h"
#include <cstddef>

// Execution Plan Enums
// These describe what alphachat should do, not how to decide.
// Portfolio fills them; alphachat reads them mechanically.

// Buffer type override: how expert tensors are placed
enum alpha_buft_mode {
    ALPHA_BUFT_NONE         = 0,  // No overrides - all tensors on GPU (CUDA graph eligible)
    ALPHA_BUFT_EXPERT_CACHE = 1,  // Pin "exps"+"nextn" to CPU, GPU expert cache does staging
    ALPHA_BUFT_CPU_MOE      = 2,  // Pin expert FFN to CPU via llm_ffn_exps_cpu_override()
};

// Eval callback: whether/how the MoE callback is installed
enum alpha_callback_mode {
    ALPHA_CB_NONE           = 0,  // No eval callback -> CUDA graph eligible
    ALPHA_CB_REACTIVE       = 1,  // Reactive callback, active from context creation
    ALPHA_CB_LUT_DISABLED   = 2,  // Install callback but start disabled (LUT-direct/cpu-moe)
};

// Warmup strategy after prompt processing
enum alpha_warmup_mode {
    ALPHA_WARMUP_NONE       = 0,  // No warmup (ALL_GPU, dense)
    ALPHA_WARMUP_SMART      = 1,  // 1-token smart fill (expert cache models)
    ALPHA_WARMUP_CLASSIFY   = 2,  // Classify prompt type + fill from profile (cpu-moe)
};

// Generation loop to use
enum alpha_gen_mode {
    ALPHA_GEN_STANDARD      = 0,  // Standard token-by-token (callback or no-cache)
    ALPHA_GEN_LUT_DIRECT    = 1,  // LUT-direct permanent draft (cached_spec_n=1)
    ALPHA_GEN_CACHED_SPEC   = 2,  // Cached-spec M>1 with verification
    ALPHA_GEN_TWOTIER       = 3,  // Two-tier speculative
};

// Optimization modes (ordered by typical speed, fastest first)
enum alpha_moe_mode {
    ALPHA_MOE_MODE_DIRECT     = 0,  // All experts on GPU, no cache needed
    ALPHA_MOE_MODE_LUT_DIRECT = 1,  // GPU LUT + adaptive renorm, zero callback
    ALPHA_MOE_MODE_CACHED_SPEC= 2,  // Cached-expert self-spec with state save/restore
    ALPHA_MOE_MODE_CPU_MOE    = 3,  // GPU attention + CPU MoE FFN, no callback
    ALPHA_MOE_MODE_REACTIVE   = 4,  // Reactive callback + DMA per miss (baseline)
    ALPHA_MOE_MODE_MIXED_SPEC = 5,  // Co-activation cluster draft + cpu-moe verify
};

// Model family classification (auto-detected from GGUF metadata)
enum alpha_model_family {
    ALPHA_FAMILY_UNKNOWN    = 0,
    ALPHA_FAMILY_QWEN_MOE   = 1,  // Qwen 3.5 MoE (57B/122B) - hybrid memory, 64 experts
    ALPHA_FAMILY_DEEPSEEK4  = 2,  // DeepSeek V4 Flash - 256 experts, sigmoid gating
    ALPHA_FAMILY_MIXTRAL    = 3,  // Mixtral 8x7B/8x22B - 8 experts, no shared expert
    ALPHA_FAMILY_DBRX       = 4,  // DBRX - 16 experts, top-4
    ALPHA_FAMILY_DENSE      = 5,  // Non-MoE model (no expert optimization)
    ALPHA_FAMILY_LLAMA      = 6,  // Llama 3/4 dense - standard transformer
    ALPHA_FAMILY_QWEN_DENSE = 7,  // Qwen 3/3.5 dense - GQA, large vocab (151K)
    ALPHA_FAMILY_GEMMA      = 8,  // Gemma 2/3 - sliding window attention
    ALPHA_FAMILY_PHI        = 9,  // Phi-3/4 - partial NoPE, small footprint
    ALPHA_FAMILY_COMMAND_R  = 10, // Command R/R+ - retrieval-optimized dense
    ALPHA_FAMILY_GEMMA_MOE  = 11, // Gemma 4 MoE (26B-A4B) - sliding window + MoE
    ALPHA_FAMILY_DEVSTRAL   = 12, // Devstral/Mistral code models
};

// Fit tier - describes how the model fits relative to GPU VRAM
enum alpha_fit_tier {
    ALPHA_FIT_NO_GPU       = 0,  // no GPU available
    ALPHA_FIT_TINY         = 1,  // model < 25% VRAM (backend sampler viable)
    ALPHA_FIT_COMFORTABLE  = 2,  // model 25-60% VRAM (standard, plenty of room)
    ALPHA_FIT_SNUG         = 3,  // model 60-85% VRAM (Q8 KV helps)
    ALPHA_FIT_TIGHT        = 4,  // model 85-97% VRAM (Q4 KV + reduced ctx)
    ALPHA_FIT_FORCE        = 5,  // model 97-112% VRAM (force all layers GPU, ctx=2048)
    ALPHA_FIT_FORCE_TIGHT  = 6,  // model 112-120% VRAM (force, ctx=1024)
    ALPHA_FIT_OFFLOAD      = 7,  // model > 120% VRAM (partial offload)
};

// MoE cache tier - describes expert cache coverage
enum alpha_moe_tier {
    ALPHA_MOE_TIER_ALL_GPU    = 0,  // K >= n_expert (all experts on GPU)
    ALPHA_MOE_TIER_LUT_DIRECT = 1,  // K >= K_min (quality LUT-direct)
    ALPHA_MOE_TIER_SPEC_DRAFT = 2,  // K >= K_draft (draft quality)
    ALPHA_MOE_TIER_CPU_MOE    = 3,  // K < K_draft (cpu-moe)
    ALPHA_MOE_TIER_CPU_PURE   = 4,  // K = 0 (no cache at all)
};

// Context for portfolio selection - filled from HW plan
struct alpha_portfolio_context {
    alpha_model_family family;
    bool   is_hybrid;
    int    n_expert;
    int    n_expert_used;
    // Fit metrics (from HW plan)
    float  fit_ratio;       // model_size_mb / gpu_vram_mb (0 = no GPU)
    int    K_estimate;      // experts that fit in remaining VRAM (MoE only)
    int    K_min;           // minimum K for quality
    int    K_draft;         // minimum K for draft
    size_t gpu_vram_mb;     // for backend sampler decision
    size_t model_size_mb;
};

// Portfolio profile - optimal settings for a model family + hardware combo
struct alpha_moe_profile {
    alpha_model_family family;
    const char *       family_name;       // human-readable (e.g. "Qwen 3.5 MoE 122B")

    // Tier info (for diagnostics/benchmarking)
    alpha_fit_tier     fit_tier;          // dense fit tier
    alpha_moe_tier     moe_tier;          // MoE cache tier

    // Core strategy
    alpha_moe_mode     mode;              // primary optimization mode
    int                topk_experts;      // top-k expert reduction (0 = model default)
    bool               flat_cache;        // force flat cache (no scratch)
    bool               cpu_moe;           // use CPU for MoE FFN
    int                cached_spec_n;     // M for cached-spec (0 = disabled)
    int                expert_cache_mb;   // expert cache budget (0 = auto from HW plan)

    // Speculative settings
    int                tier1_cycles;      // N for two-tier (0 = disabled)

    // Tuning
    int                split_group;       // sync every N MoE layers (0 = auto)
    bool               no_barrier;        // blocking sync instead of async barrier
    bool               output_cpu;        // pin output tensor to CPU (V4 precision fix)

    // Expert cache quality gate
    int                k_min;             // minimum cache K for quality (0 = auto: max(n_active*8, n_expert/4))

    // Dense model settings
    bool               flash_attn;        // recommend flash attention
    int                n_ctx_default;     // recommended default context (0 = model default)
    int                n_batch;           // recommended batch size (0 = default 512)
    int                n_threads;         // recommended thread count (0 = auto)

    // Fit-ratio driven settings
    bool               backend_sampler;   // recommend GPU-side sampling (tiny models)
    bool               force_fit;         // force all layers to GPU despite tight VRAM
    bool               partial_offload;   // some layers on CPU
    enum ggml_type     kv_quant;          // recommended KV quantization (GGML_TYPE_F16 = default)

    // Execution Plan (filled by portfolio, read mechanically by alphachat)
    // These fields fully determine runtime behavior. No inline conditionals needed.
    alpha_buft_mode     buft;               // buffer type override mode
    alpha_callback_mode callback;           // eval callback mode
    alpha_warmup_mode   warmup;             // warmup strategy after prompt
    alpha_gen_mode      gen;                // generation loop to use
    bool                need_moe_opt;       // create moe_optimizer (false for ALL_GPU/dense)
    int                 expert_cache_max;   // expert cache MB (0=none, 99999=max available)
    bool                disable_cache_for_prompt; // disable cache during prompt (cpu-moe path)
    bool                lut_init;           // initialize GPU LUT before generation
    int                 readback_interval;  // argsort readback: 0=every token, N=every N
    bool                gpu_layers_all;     // force n_gpu_layers=999

    // Quality Gate Parameters (per-tier, orthogonal)
    // Each tier gets its own thresholds, so tuning one model size does not affect another.
    float               conf_margin_threshold; // confidence gate: margin below this -> count++ (0=disabled)
    int                 conf_trigger_count;    // N consecutive low-conf tokens before resync (0=disabled)
    bool                eos_verify;            // verify EOS with cpu-moe before stopping
    int                 eos_max_retries;       // max EOS verify retries (0 if eos_verify=false)
    int                 quality_grace_tokens;  // skip quality checks for first N tokens after warmup

    // Dense Self-Speculative Settings
    int                 self_spec_skip;        // layers to skip for draft (0=disabled)
    int                 self_spec_m;           // draft tokens per cycle (default 4)

    // Performance reference (from benchmarks, informational only)
    float              ref_tok_per_sec;   // speed on reference hardware
    const char *       ref_hardware;      // e.g. "RTX 3090 24GB PCIe Gen3"
};

// Auto-detect model family from HW plan metadata
LLAMA_API alpha_model_family alpha_moe_detect_family(
    const char * arch,          // GGUF general.architecture
    int n_expert,
    int n_expert_used,
    size_t model_size_mb);

// Get the portfolio profile using full context (preferred API)
// Selects optimal config based on fit_ratio, K coverage, family
LLAMA_API alpha_moe_profile alpha_moe_portfolio_select_v2(
    const alpha_portfolio_context * ctx);

// Legacy API - wraps v2 with fit_ratio = fits_in_gpu ? 0.5 : 2.0
LLAMA_API alpha_moe_profile alpha_moe_portfolio_select(
    alpha_model_family family,
    bool fits_in_gpu,
    bool is_hybrid,
    int n_expert = 0);

// Compute fit tier from ratio
LLAMA_API alpha_fit_tier alpha_moe_compute_fit_tier(float fit_ratio);

// Compute MoE tier from K metrics
LLAMA_API alpha_moe_tier alpha_moe_compute_moe_tier(int K_estimate, int K_min, int K_draft, int n_expert);

// Get tier name strings (for logging)
LLAMA_API const char * alpha_fit_tier_name(alpha_fit_tier tier);
LLAMA_API const char * alpha_moe_tier_name(alpha_moe_tier tier);

// Get a human-readable summary of the selected profile
LLAMA_API void alpha_moe_portfolio_print(const alpha_moe_profile * profile);
