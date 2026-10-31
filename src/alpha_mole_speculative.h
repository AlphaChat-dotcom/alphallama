#pragma once

// AlphaLlama: Speculative Execution with Verified Rollback
//
// Implements speculate -> verify -> rollback for MoE expert cache.
//
// Algorithm per token:
//   1. prepare: pre-load predicted experts
//   2. speculate: run full graph with no reactive callback (fast)
//   3. verify: read argsort outputs, compare actual vs cached expert IDs,
//              find first_mismatch layer M (or -1 if all correct)
//   4. decide:
//      M == -1 -> output is valid, done (fast path)
//      M >= 0  -> discard output, re-run from M with reactive callback
//   5. record: update predictions for next token
//
// Output is only ever taken from a pass in which all selected experts were
// cached, so it is never partial or approximate.

#include "alpha_mole_moe_optimizer.h"
#include <vector>
#include <cstdint>

struct alpha_moe_optimizer;  // forward declaration (internal struct)

// Speculative execution state - allocated once, reused per token
struct alpha_speculative_state {
    // Per-layer saved hidden states (read from GPU after speculative pass)
    // layer_hidden[i] = hidden state at entry to MoE layer i
    // Only meaningful for layers 0..n_moe_layers-1
    std::vector<std::vector<float>> layer_hidden;  // [n_moe_layers][hidden_dim]

    // Per-layer actual expert IDs from argsort (read after speculative pass)
    std::vector<std::vector<int32_t>> layer_actual_ids; // [n_moe_layers][n_expert_used]

    // Model dimensions (auto-detected, not hardcoded)
    int n_moe_layers;     // number of MoE layers (auto from model)
    int hidden_dim;       // hidden dimension (auto from model n_embd * n_hc)
    int n_expert_used;    // experts per token (auto from model)

    // Verification result
    int first_mismatch;   // -1 = all correct, >= 0 = first wrong MoE layer index

    // Statistics
    int64_t total_tokens;
    int64_t speculative_hits;     // tokens where all predictions correct
    int64_t speculative_misses;   // tokens requiring re-run
    int64_t total_rerun_layers;   // cumulative layers re-run across all misses
    double  total_spec_time_ms;   // time in speculative pass
    double  total_verify_time_ms; // time in verification
    double  total_rerun_time_ms;  // time in re-run

    bool initialized;
};

// Initialize speculative state from optimizer (auto-detects all dimensions)
void alpha_speculative_init(alpha_speculative_state * state, alpha_moe_optimizer * opt);

// Free speculative state
void alpha_speculative_cleanup(alpha_speculative_state * state);

// After speculative graph completes: read argsort outputs + hidden states,
// find first mismatch layer. Sets state->first_mismatch.
// Returns true if all predictions correct (output is valid).
// Returns false if mismatch found (need re-run).
bool alpha_speculative_verify(alpha_speculative_state * state, alpha_moe_optimizer * opt);

// Re-run from first_mismatch layer with reactive callback.
// This is a full re-run (a partial one would require graph surgery).
// The callback skips layers before first_mismatch (they're already correct in cache).
void alpha_speculative_rerun(alpha_speculative_state * state, alpha_moe_optimizer * opt,
                             struct llama_context * ctx);

// Print speculative execution statistics
void alpha_speculative_print_stats(const alpha_speculative_state * state);

// Full speculative execution loop for one token:
// prepare -> speculate -> verify -> (rerun if needed) -> record
// Returns 0 on success, non-zero on decode failure.
int alpha_speculative_decode_token(
    alpha_speculative_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    llama_token token);

// Self-speculative decoding
//
// Uses the same model in two modes:
//   draft:  shared experts only (no MoE routing, no callbacks) -> fast
//   verify: full model with batch of N+1 tokens -> callbacks amortized
//
// Requires model with shared experts (DeepSeek V4, Qwen3 MoE).
// Not applicable to Mixtral-style models without shared experts.

struct alpha_self_spec_state {
    int n_draft;                    // max draft tokens per cycle

    std::vector<llama_token> draft_tokens;

    // Statistics
    int64_t n_accepted_total;
    int64_t n_draft_total;
    int64_t n_cycles;
    double  draft_ms_total;
    double  verify_ms_total;

    bool initialized;
};

// Initialize self-speculative state
void alpha_self_spec_init(alpha_self_spec_state * state, alpha_moe_optimizer * opt, int n_draft);

// Run one self-speculative cycle: draft N tokens -> verify batch -> accept prefix
// Returns number of accepted tokens (written to `accepted`).
// The caller must advance n_past by the return value.
int alpha_self_spec_generate(
    alpha_self_spec_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    struct llama_sampler * smpl,
    const struct llama_vocab * vocab,
    llama_token last_token,
    llama_pos n_past,
    std::vector<llama_token> & accepted);

// Print self-speculative statistics
void alpha_self_spec_print_stats(const alpha_self_spec_state * state);

// Cached-expert self-speculative decoding
//
// Draft: n_expert-slot GPU cache holding the hot experts, no callback.
// Verify: normal expert cache with full DMA, batched.
//
// The draft uses argsort output directly as indices into the slot cache.
// Uncached experts hit zero-filled slots (contribute nothing); most selected
// experts are cached, so acceptance stays high.

struct alpha_cached_spec_state {
    int n_draft;                    // max draft tokens per cycle (M)

    std::vector<llama_token> draft_tokens;
    std::vector<float *>     draft_logits;  // saved logits per draft token

    // Statistics
    int64_t n_accepted_total;
    int64_t n_draft_total;
    int64_t n_cycles;
    double  draft_ms_total;
    double  verify_ms_total;

    bool initialized;
    bool repetition_detected;   // set when verify detects high-confidence repetition
};

void alpha_cached_spec_init(alpha_cached_spec_state * state, int n_draft);

// Run one cached-spec cycle: draft M tokens -> verify M+1 -> accept prefix
// Returns number of accepted tokens (written to `accepted`).
int alpha_cached_spec_generate(
    alpha_cached_spec_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    struct llama_sampler * smpl,
    const struct llama_vocab * vocab,
    llama_token last_token,
    llama_pos n_past,
    std::vector<llama_token> & accepted);

void alpha_cached_spec_print_stats(const alpha_cached_spec_state * state);

// Confidence margin from last LUT-direct decode (top1 - top2 logit)
float alpha_lut_direct_get_margin();

// Two-tier speculative decoding, for models with co-activation clusters:
//   draft:  NxM tokens using merged cluster experts (GPU LUT)
//   tier 1: verify with cached real experts (GPU LUT draft mode)
//   tier 2: verify tier1-accepted tokens with cpu-moe (batched, ground truth)
//
// Tier 2 runs once per round as a batch, amortizing cpu-moe overhead.

struct alpha_twotier_state {
    int n_draft;          // M (draft tokens per cycle)
    int tier1_cycles;     // N (draft cycles before tier 2)

    std::vector<llama_token> draft_tokens;

    // Statistics
    int64_t n_draft_total;
    int64_t n_tier1_accepted_total;
    int64_t n_tier2_accepted_total;  // final accepted (after both tiers)
    int64_t n_rounds;
    int64_t n_tier1_rejects;
    int64_t n_tier2_rejects;
    double  draft_ms_total;
    double  tier1_ms_total;
    double  tier2_ms_total;

    bool initialized;
};

void alpha_twotier_init(alpha_twotier_state * state, int n_draft, int tier1_cycles);

// Run one two-tier round: draft N*M tokens -> tier 1 verify -> tier 2 verify.
// Returns number of accepted tokens (written to `accepted`).
int alpha_twotier_generate(
    alpha_twotier_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    struct llama_sampler * smpl,
    const struct llama_vocab * vocab,
    llama_token last_token,
    llama_pos n_past,
    std::vector<llama_token> & accepted);

void alpha_twotier_print_stats(const alpha_twotier_state * state);

// LUT-direct mode
// GPU LUT + adaptive renorm, no callback overhead.
// Decodes one token using cached experts, then refreshes cache from argsort.
// Use when acceptance rate is ~100% - skips verification for maximum speed.
// Returns sampled token, or LLAMA_TOKEN_NULL on failure.
// If smpl is provided, uses it for temperature/sampling; otherwise argmax (fast).
LLAMA_API llama_token alpha_lut_direct_decode(
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    const struct llama_vocab * vocab,
    llama_token input_token,
    struct llama_sampler * smpl = nullptr);

// Recurrent-state window bracket (window patch-correction decode)
// Snapshot/restore of hybrid-model recurrent R/S tensors so a serve-side
// window loop can rewind DeltaNet state around a draft window without seq_cp
// (shallow-copy hazard). No-ops (return true) on non-hybrid models.
struct alpha_recurrent_snapshot_t {
    // Device-side mirror. The recurrent state can be ~150 MB (130 tensors on a 27B model), so
    // staging it through host memory over PCIe costs milliseconds per window, while a
    // device-to-device copy is far faster. State that may be discarded never crosses the bus.
    struct ggml_context *          mctx = nullptr;   // holds the mirror tensor descriptors
    struct ggml_backend_buffer *   mbuf = nullptr;   // device buffer backing them
    std::vector<struct ggml_tensor *> mr, ms;
    bool valid = false;
    bool alloc_failed = false;   // the mirror did not fit once: do not retry a ~128 MB cudaMalloc every window
};
LLAMA_API bool alpha_recurrent_snapshot(struct llama_context * ctx, alpha_recurrent_snapshot_t & snap);
// Restores tensors and resets recurrent cell positions to reset_pos-1 so a
// subsequent llama_memory_seq_rm(mem, 0, reset_pos, -1) succeeds.
LLAMA_API bool alpha_recurrent_restore(struct llama_context * ctx, const alpha_recurrent_snapshot_t & snap, llama_pos reset_pos);

// ReplaySSM: rewind recurrent cell bookkeeping only (state tensors already folded).
LLAMA_API bool alpha_recurrent_mark_pos(struct llama_context * ctx, llama_seq_id seq_id, llama_pos reset_pos);

// ALPHA_HYBRID_PREFIX: save / restore the MTP head's carried boundary row for a recurrent prefix
// checkpoint (see llama_context::mtp_seq_export/import). Both return false when MTP is inactive or
// the sequence's carried row is not the row for `pos`; the caller then simply skips the MTP part.
LLAMA_API bool alpha_mtp_seq_save(struct llama_context * ctx, llama_seq_id seq, llama_pos pos, std::vector<float> & h);
LLAMA_API bool alpha_mtp_seq_restore(struct llama_context * ctx, llama_seq_id seq, llama_pos ckpt, const std::vector<float> & h);
LLAMA_API uint64_t alpha_mtp_seq_kv_hash(struct llama_context * ctx, llama_seq_id seq, int32_t & lo, int32_t & hi);
