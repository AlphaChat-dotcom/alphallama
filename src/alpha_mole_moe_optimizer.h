#pragma once

// Alpha-MoLE: MoE Optimizer
//
// Integrates expert LRU cache with llama.cpp's MoE execution.
// Uses the eval callback to observe expert routing decisions and
// populate the cache for subsequent tokens.
//
// Architecture:
//   Token N: graph computes, callback observes which experts were selected
//   Token N+1: before decode, pre-load predicted experts (from token N)
//              -> most experts are already on GPU (cache hit)
//
// The optimizer wraps llama_decode() and manages the expert cache
// lifecycle automatically.

#include "llama.h"
#include "ggml-backend.h"
#include "alpha_mole_expert_cache.h"

struct alpha_moe_optimizer;

// Debug logging: completely compiled out in release builds (ALPHA_DEBUG_BUILD not defined).
// In dev builds, gated behind --debug flag at runtime.
#ifdef ALPHA_DEBUG_BUILD
  LLAMA_API extern bool g_alpha_debug;
  #define ALPHA_LOG_DBG(fmt, ...) do { if (g_alpha_debug) fprintf(stderr, fmt, ##__VA_ARGS__); } while(0)
#else
  static constexpr bool g_alpha_debug = false;
  #define ALPHA_LOG_DBG(fmt, ...) ((void)0)
#endif

// Thermal-aware expert loading policy for mobile devices
// Experts are never bulk-loaded. Hot experts are pinned at startup.
// Warm cache is filled lazily via single-expert predicted prefetch.
// Cold experts stay on storage and are only fetched one-at-a-time when predicted.
enum alpha_moe_thermal_policy {
    ALPHA_MOE_THERMAL_NORMAL   = 0, // Allow predicted single-expert prefetch from storage (max 1/token)
    ALPHA_MOE_THERMAL_THROTTLE = 1, // No storage reads. Cache-only. Substitute on miss.
    ALPHA_MOE_THERMAL_EMERGENCY= 2, // Hot experts + shared only. Evict warm cache to free memory.
};

struct alpha_moe_optimizer_config {
    size_t vram_budget_mb;         // GPU memory for expert cache (0 = auto)
    int    n_predict_ahead;        // tokens to predict ahead (default: 1)
    bool   enable_profiling;       // collect expert activation stats
    bool   prefetch_next_layer;    // prefetch next-layer experts during current compute
    int    prefetch_depth;         // how many layers ahead to prefetch (default: 1)

    // Mobile/thermal-aware settings
    bool   mobile_mode;            // enable thermal-aware expert loading
    int    n_hot_experts;          // experts always in memory (0 = auto: n_expert/8)
    int    n_warm_slots;           // LRU cache slots for warm experts (0 = auto)
    float  thermal_throttle_temp;  // temperature (C) to switch to WARM policy (default: 38)
    float  thermal_emergency_temp; // temperature (C) to switch to HOT policy (default: 43)

    // Phase A - routing locality
    float  expert_sticky;          // sticky routing momentum (0 = disabled, 0.3 = recommended)
    float  expert_layer_bias;      // cross-layer consistency bias (0 = disabled, 0.2 = recommended)

    // Phase B - expert grouping
    const char * expert_groups_path; // path to expert_groups.json (nullptr = disabled)

    // Callback grouping
    int    split_group;             // sync every N MoE layers (1 = every layer, 0 = auto)
    bool   no_barrier;              // true = use cudaStreamSynchronize (blocking), false = async barrier
    bool   flat_cache;              // true = force flat cache (no scratch), minimal callback overhead
    bool   force_cache;             // true = force expert cache even when model fits in GPU (for override/routing experiments)
    bool   debug;                   // true = enable internal debug logging (--debug flag)
};

// Create MoE optimizer for a model
// Analyzes model architecture to determine if MoE and configure cache
LLAMA_API alpha_moe_optimizer * alpha_moe_optimizer_init(
        struct llama_model   * model,
        struct llama_context * ctx,
        alpha_moe_optimizer_config config);

// Allocate expert cache on GPU (call after context creation)
// Context creation allocates compute buffer first, so remaining VRAM is used for cache.
LLAMA_API void alpha_moe_optimizer_alloc_cache(alpha_moe_optimizer * opt);

// Load mixed expert cluster assignments and preload representative experts into GPU cache.
// cluster_json: path to JSON with per-layer cluster->expert assignments.
// Requires cache already allocated (call after alloc_cache).
// Uses existing flat cache (K = n_clusters + 1) filled with cluster representatives.
LLAMA_API bool alpha_moe_optimizer_load_mixed_experts(
        alpha_moe_optimizer * opt, const char * cluster_json);

// Enter/exit mixed expert draft mode (swaps expert tensors to cluster cache, disables callback)
LLAMA_API void alpha_moe_optimizer_enter_mixed_draft_mode(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_exit_mixed_draft_mode(alpha_moe_optimizer * opt);
LLAMA_API bool alpha_moe_optimizer_has_mixed_experts(const alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_set_callback_active(alpha_moe_optimizer * opt, bool active);

// Pre-load experts before decode (call before llama_decode)
// Uses previous token's routing to predict current token's experts
// For CPU-resident expert layers: creates GPU staging tensors and
// populates needed expert slices via DMA
LLAMA_API void alpha_moe_optimizer_prepare(
        alpha_moe_optimizer * opt);

// Post-decode: record routing decisions for next prediction
// Restores any swapped tensor pointers
LLAMA_API void alpha_moe_optimizer_record(
        alpha_moe_optimizer * opt);

// Print optimizer stats
LLAMA_API void alpha_moe_optimizer_print_stats(
        const alpha_moe_optimizer * opt);

// Check if model is MoE (has experts)
LLAMA_API bool alpha_moe_optimizer_is_moe(
        const alpha_moe_optimizer * opt);

// Set context after creation (enables GPU staging setup)
// Call this after llama_init_from_model() to create GPU staging tensors
LLAMA_API void alpha_moe_optimizer_set_context(
        alpha_moe_optimizer * opt,
        struct llama_context * ctx);

// Get the eval callback for per-layer expert staging
// Set this on llama_context_params.cb_eval before creating context
LLAMA_API ggml_backend_sched_eval_callback alpha_moe_optimizer_get_eval_callback(void);
LLAMA_API void * alpha_moe_optimizer_get_eval_data(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_reinstall_callback(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_clear_expert_remap(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_restore_expert_remap(alpha_moe_optimizer * opt);

// Disable/enable cache (for prompt processing vs generation)
LLAMA_API void alpha_moe_optimizer_disable_cache(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_clear_cache(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_enable_cache(alpha_moe_optimizer * opt);

// Domain-Expert Pinning: detect output domain from prompt keywords, pin rare experts
LLAMA_API const char * alpha_moe_detect_output_domain(const char * prompt, int prompt_len);
LLAMA_API void alpha_moe_set_detected_domain(const char * domain);
LLAMA_API const char * alpha_moe_get_detected_domain();

// Multi-domain detection: returns top-2 domains for prompts spanning multiple areas
struct alpha_moe_domain_result {
    const char * domain1;  // best match (or nullptr)
    const char * domain2;  // second best (or nullptr)
    int score1, score2;
};
LLAMA_API alpha_moe_domain_result alpha_moe_detect_output_domain_top2(const char * prompt, int prompt_len);
LLAMA_API void alpha_moe_set_detected_domain2(const char * domain);
LLAMA_API const char * alpha_moe_get_detected_domain2();

// Pre-prompt domain expert pinning: load grammar + domain1 + domain2 into cache + update LUT.
// Call before prompt decode for profile-first LUT prefill (no cpu-moe needed).
LLAMA_API void alpha_moe_optimizer_pin_domain_experts(alpha_moe_optimizer * opt, const char * model_dir);

// Ranked domain detection: returns ALL matching domains sorted by keyword score.
// Returns count of domains written to out_domains (up to max_results).
LLAMA_API int alpha_moe_detect_output_domain_ranked(const char * prompt, int prompt_len,
                                                     const char ** out_domains, int * out_scores, int max_results);

// Serial expert tiling: load expert partition by frequency rank.
// pass_idx=0: top-K experts. pass_idx=1: next K. pass_idx=2: next K.
// Over N passes covers N*K out of n_expert (e.g., 3x55=165/256 = 64%).
LLAMA_API void alpha_moe_optimizer_load_expert_partition(alpha_moe_optimizer * opt, int pass_idx);

// Pipeline prefetch: madvise all expert mmap pages before prompt decode
LLAMA_API void alpha_moe_optimizer_prefetch_prompt_experts(alpha_moe_optimizer * opt);

// Evict stale expert mmap pages to prevent OOM on large models (madvise DONTNEED)
LLAMA_API void alpha_moe_optimizer_evict_expert_pages(alpha_moe_optimizer * opt);
// Clear all cache slots - fresh start for next expert loading phase
LLAMA_API void alpha_moe_optimizer_clear_cache(alpha_moe_optimizer * opt);
// Check if a named domain profile is loaded (e.g., "grammar")
LLAMA_API bool alpha_moe_optimizer_has_domain_profile(const char * name);
// Domain profile enumeration (for dynamic classify domain list)
LLAMA_API int alpha_moe_optimizer_get_domain_profile_count();
LLAMA_API const char * alpha_moe_optimizer_get_domain_profile_name(int idx);
// Get current K (expert cache slots per layer)
LLAMA_API int alpha_moe_optimizer_get_K(const alpha_moe_optimizer * opt);
// Get total number of experts in the model
LLAMA_API int alpha_moe_optimizer_get_n_expert(const alpha_moe_optimizer * opt);

// Prompt-adaptive cache: read all tokens' expert activations from argsort after prefill,
// then fill GPU cache per-layer with the most-used experts for this specific prompt.
LLAMA_API void alpha_moe_optimizer_record_prompt(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_record_prompt_activations(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_fill_cache_from_prefill(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_warmup_cache_from_decode(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_warmup_accumulate(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_warmup_fill_cache_topk(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_warmup_1token_smart(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_load_style_profiles(alpha_moe_optimizer * opt, const char * dir_path);
LLAMA_API void alpha_moe_optimizer_warmup_classify_and_fill(alpha_moe_optimizer * opt);
LLAMA_API bool alpha_moe_optimizer_has_profiles();
LLAMA_API float alpha_moe_optimizer_get_profile_quality();  // 0.0-1.0, from last classify
LLAMA_API const char * alpha_moe_optimizer_get_profile_name();  // matched profile name from last classify
LLAMA_API void alpha_moe_optimizer_set_profiling(bool enable);

// Per-token cache miss rate from last decode (for quality gate in LUT-direct mode).
// Returns fraction of MoE layers that had at least one cache miss (0.0 = all cached, 1.0 = all missed).
LLAMA_API float alpha_moe_optimizer_get_last_miss_rate(const alpha_moe_optimizer * opt);

// Sequential prompt processing with cache enabled throughout.
// Processes tokens one at a time: prepare() -> decode(1) -> record().
// Ensures GPU-consistent execution for HC-sensitive models (DeepSeek V4).
// Returns 0 on success, non-zero on decode failure.
LLAMA_API int alpha_moe_optimizer_process_prompt_sequential(
    alpha_moe_optimizer * opt, struct llama_context * ctx,
    const llama_token * tokens, int n_tokens);

// Batched prompt processing: chunks of up to 32 tokens per decode call.
// Much faster than sequential (reduces NxL callbacks to ceil(N/32)xL).
// Each chunk: prepare() -> decode(chunk) -> record().
LLAMA_API int alpha_moe_optimizer_process_prompt_batched(
    alpha_moe_optimizer * opt, struct llama_context * ctx,
    const llama_token * tokens, int n_tokens);

// Enable/disable inter-layer expert prefetch
// When enabled, record() will start async DMA for next-layer experts
// based on current-layer routing (same-expert heuristic)
LLAMA_API void alpha_moe_optimizer_set_prefetch(alpha_moe_optimizer * opt, bool enabled, int depth);

// Thermal-aware expert loading (mobile)
// Set current thermal policy (call before prepare, or let auto-govern from temperature)
LLAMA_API void alpha_moe_optimizer_set_thermal_policy(alpha_moe_optimizer * opt, alpha_moe_thermal_policy policy);

// Auto-govern: set device temperature, policy adjusts automatically
LLAMA_API void alpha_moe_optimizer_set_temperature(alpha_moe_optimizer * opt, float temp_celsius);

// Get current policy and expert tier stats
LLAMA_API alpha_moe_thermal_policy alpha_moe_optimizer_get_thermal_policy(const alpha_moe_optimizer * opt);

// Get expert activation frequency data (for quantization importance weighting)
// Returns array of n_expert floats, each = fraction of tokens activating that expert
// Caller must free with free(). Returns NULL if profiling not enabled.
LLAMA_API float * alpha_moe_optimizer_get_expert_frequencies(const alpha_moe_optimizer * opt, int layer, int * n_expert);

// Per-layer routing data (for confidence logging)
LLAMA_API const int * alpha_moe_optimizer_get_prev_routing(const alpha_moe_optimizer * opt, int layer);
LLAMA_API int alpha_moe_optimizer_get_n_expert_used(const alpha_moe_optimizer * opt);
LLAMA_API int alpha_moe_optimizer_get_n_layer(const alpha_moe_optimizer * opt);
LLAMA_API int alpha_moe_optimizer_get_n_moe_layers(const alpha_moe_optimizer * opt);  // CPU expert layers (0 = all on GPU)

// Free
LLAMA_API void alpha_moe_optimizer_free(alpha_moe_optimizer * opt);

// Internal: argsort + weight output tracking (used by graph builder + optimizer)
void alpha_mole_clear_argsort_outputs();
int alpha_mole_get_argsort_count();
void alpha_mole_set_argsort_persist(bool v);
bool alpha_mole_get_argsort_persist();
struct ggml_tensor * alpha_mole_get_argsort_output(int layer);
struct ggml_tensor * alpha_mole_get_weight_output(int layer);
struct ggml_tensor * alpha_mole_get_ffn_input(int layer);   // hidden state for CPU fallback
struct ggml_tensor * alpha_mole_get_ffn_output(int layer);   // MoE output tensor for CPU fallback overwrite
struct ggml_tensor * alpha_mole_get_gating_probs(int layer); // gating probs for CPU weight computation

// Per-token miss tracking for progressive cache warming during LUT prefill
LLAMA_API int alpha_moe_optimizer_get_miss_layers(const alpha_moe_optimizer * opt);
LLAMA_API int alpha_moe_optimizer_get_total_layers(const alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_set_lut_prefill(alpha_moe_optimizer * opt, bool active);
LLAMA_API bool alpha_moe_optimizer_is_lut_prefill(const alpha_moe_optimizer * opt);

// LUT prefill quality check: returns the position of the first "bad" token
// (where >half of selected experts were cache misses = slot 0).
// Returns -1 if all tokens had acceptable coverage.
// Used for two-pass prefill: LUT pass 1, then cpu-moe repair from bad position.
LLAMA_API int alpha_moe_optimizer_check_prefill_quality(alpha_moe_optimizer * opt, int n_tokens);

// Speculative execution for generation tokens:
// set_speculative(true) -> prediction-only mode (no reactive callback, fast)
// set_speculative(false) -> reactive mode (callback fixes misses, correct)
// verify_speculative -> check if all predictions matched actual gating
LLAMA_API void alpha_moe_optimizer_set_speculative(alpha_moe_optimizer * opt, bool enable);

// Self-speculative draft mode: skip routed experts, use shared experts only
// When enabled, the graph builder uses only shared FFN (no MoE routing/callbacks)
LLAMA_API void alpha_moe_optimizer_set_draft_mode(alpha_moe_optimizer * opt, bool enable);
LLAMA_API void alpha_moe_optimizer_set_draft_expert_count(alpha_moe_optimizer * opt, int count);

// Top-k expert reduction: override n_expert_used in graph builder
// 0 = use model default, >0 = reduce to this many experts per token
// Weights are renormalized over selected experts.
LLAMA_API void alpha_moe_optimizer_set_topk(alpha_moe_optimizer * opt, int topk);
LLAMA_API bool alpha_moe_optimizer_has_shared_experts(const alpha_moe_optimizer * opt);
LLAMA_API float alpha_moe_optimizer_get_hit_rate(const alpha_moe_optimizer * opt);
LLAMA_API bool alpha_moe_optimizer_verify_speculative(alpha_moe_optimizer * opt);

// Cached-expert self-speculative decoding (GPU LUT)
// GPU-side lookup table: per-layer [1, n_expert] I32 on GPU.
// lut[expert_id] = cache_slot if cached, 0 (zero expert) if not.
// In draft mode: ggml_get_rows(lut, argsort) remaps on GPU with no callback overhead.
LLAMA_API void alpha_moe_optimizer_init_spec_lut(alpha_moe_optimizer * opt);
LLAMA_API void alpha_moe_optimizer_update_lut(alpha_moe_optimizer * opt);

// Reload cache from last forward pass argsort outputs.
// Reads argsort, DMAs missing experts into cache, updates LUT.
// Call after exit_draft_mode, before the next enter_draft_mode.
LLAMA_API int alpha_moe_optimizer_reload_from_argsort(alpha_moe_optimizer * opt);  // returns experts loaded

// Apply portfolio-driven warmup sequence after prompt processing.
// Handles all tiers (ALL_GPU, LUT_DIRECT, CPU_MOE).
// Sequences: callback lifecycle -> warmup (CLASSIFY/SMART) -> cache enable -> draft mode.
// Both CLI and serve mode should call this for identical behavior.
struct alpha_moe_profile;  // forward decl (alpha_moe_portfolio.h)
LLAMA_API void alpha_moe_optimizer_apply_warmup(
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    const alpha_moe_profile * plan,
    const char * model_dir);

// Enter draft mode: wire LUT to model, disable callback (pure GPU path).
LLAMA_API void alpha_moe_optimizer_enter_draft_mode(alpha_moe_optimizer * opt);

// Exit draft mode: restore normal expert cache + callback.
LLAMA_API void alpha_moe_optimizer_exit_draft_mode(alpha_moe_optimizer * opt);

// Check if currently in draft mode
LLAMA_API bool alpha_moe_optimizer_is_draft_active(const alpha_moe_optimizer * opt);

// Get per-layer difficulty score (0.0=easy, 1.0=hard) from last token's routing entropy
LLAMA_API float alpha_moe_optimizer_get_difficulty(const alpha_moe_optimizer * opt, int layer);

// Speculative two-pass decode: run graph without callback, verify, re-run on miss
// Returns 0 on success. Guaranteed correct output (re-runs if any expert was missing).
LLAMA_API int alpha_moe_optimizer_speculative_decode(
    alpha_moe_optimizer * opt, struct llama_context * ctx, llama_token token);

// Expert override API
// Override router selection for a specific layer. When set, the callback uses
// these expert IDs instead of the argsort output for that layer.
// expert_ids: array of expert indices (0 to n_expert-1), length n_experts
// weights: optional softmax weights (can be NULL for equal weighting)
// Returns 0 on success.
LLAMA_API int alpha_mole_set_layer_experts(
    alpha_moe_optimizer * opt,
    int layer_idx,
    const int * expert_ids,
    const float * weights,
    int n_experts);

// Clear override for a layer (revert to router selection)
LLAMA_API void alpha_mole_clear_layer_override(
    alpha_moe_optimizer * opt,
    int layer_idx);

// Clear all layer overrides
LLAMA_API void alpha_mole_clear_all_overrides(
    alpha_moe_optimizer * opt);

// Set cpu-moe verify mode for cached-spec self-speculative decoding.
// When enabled, the verify pass in cached_spec_generate uses cpu-moe (full quality)
// instead of the reactive expert cache. Needed when K < K_min.
LLAMA_API void alpha_moe_optimizer_set_cpu_moe_verify(alpha_moe_optimizer * opt, bool enable);

// Expert frequency profiling
// Set pre-load expert IDs for a layer (from frequency profiling).
// When set, alloc_cache() loads these experts instead of sequential 0..K-2.
// expert_ids: array of expert indices sorted by frequency (most frequent first).
// count: number of expert IDs (will be capped to K-1 during alloc).
// Call before alpha_moe_optimizer_alloc_cache().
LLAMA_API void alpha_moe_optimizer_set_preload_experts(
    alpha_moe_optimizer * opt, int layer, const int * expert_ids, int count);

// Async DMA for expert cache
// These are implemented in ggml-cuda.cu. No-op on non-CUDA builds.
#ifdef __cplusplus
extern "C" {
#endif
void _acdi(void);
void _acda(struct ggml_tensor * dst, const void * src, size_t offset, size_t size);
void _acds(void);
// Non-blocking barrier: records event on DMA stream, makes compute stream wait for it.
// Host returns immediately - GPU compute stream blocks until DMA completes.
void _acdb(void);
// Pinned staging for truly async DMA (pageable mmap->pinned->GPU)
void * _acpa(size_t size);
void   _acpf(void * ptr);
void   _acdas(struct ggml_tensor * dst, const void * src,
                                    size_t offset, size_t size, void * pinned_buf);
#ifdef __cplusplus
}
#endif
