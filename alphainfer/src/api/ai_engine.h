// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaInfer C API: the contract AlphaServe's engine adapter links against.
//
// Declared to match tools/serve/serve-ai-engine.h in the AlphaServe tree, function for function and
// field for field. C rather than C++ so the ABI does not depend on either side's flags.
//
// Three things the caller must know, because this API's shape cannot express them:
//
// 1. A round is batch-rigid. Internally T = L * n_streams and every slot is staged every round, so
//    one decode call advances all open streams together. seq_ids must name every stream that is not
//    finished; naming a subset returns EINVAL rather than silently re-feeding a stale token.
//
// 2. ai_engine_decode's out_sampled has no capacity argument. Writes are bounded at n_streams (the
//    maximum emissions per plain round) and a would-be overflow returns EFAIL instead of writing past
//    the end. Size out_sampled to at least n_streams.
//
// 3. ai_engine_prefill must run first for each stream, and again after every reset. Prefill is a
//    different code path (it chunks internally); a decode call on a slot with no prefill is refused
//    rather than run on an empty draft vector.
//
// MTP (mtp_k > 0) goes through ai_engine_decode_mtp, which takes an output capacity and returns a stream id per
// token: one MTP round emits 1..mtp_k+1 tokens per stream. ai_engine_decode returns ENOTSUP on an MTP engine. The
// verify configuration is the CLI's (engine/mtp_policy.cpp): exact verify by default on MoE models (every emitted
// token bit-identical to plain decode on sm_86; not yet on sm_120), fast on dense models, plus the confidence-gated
// verify width. AI_MTP_MODE=fast|exact overrides.
//
// DFlash2: AI_DFLASH2=<drafter.gguf> at create drafts with the DFlash2 drafter instead of the MTP head (the CLI's
// --dflash2). An env var, not a params field, so this struct stays field-for-field with AlphaServe's copy. Requires
// n_streams == 1, resident experts and no host_embd (create fails otherwise). The verify is always fast (not
// token-identical); mtp_k is set from the drafter's block size (7), so size buffers with
// ai_engine_max_tokens_per_round and read the mode with ai_engine_mtp_info. ai_engine_prefill_chunk buffers the
// pieces and prefills the whole prompt at is_last (one stream: nothing to interleave), identical to ai_engine_prefill.
//
// Refused explicitly rather than by omission:
//   logits_mask != NULL  -> ENOTSUP. Ignoring a sampling constraint would return plausible wrong text.
//   ai_engine_decode on an MTP engine -> ENOTSUP (use ai_engine_decode_mtp).
//   ai_engine_warm_profiles on an MTP engine -> ENOTSUP (MTP profiles are recorded on first use).

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine handle.
typedef struct ai_engine_s ai_engine;

// Return codes for the int-returning entries. 0 is success; everything else is an error the caller
// must turn into decode_error / a 503, never into a token.
enum {
    AI_ENGINE_OK      =  0,
    AI_ENGINE_EINVAL  = -1,   // bad arguments
    AI_ENGINE_ENOTSUP = -2,   // the engine cannot serve this request shape (see the header notes)
    AI_ENGINE_EFAIL   = -3,   // the engine tried and failed
};

// Capture state. A capture failure degrades to direct launches and the process keeps returning
// success: src/ops/adapter.cpp prints "[capture] end failed; falling back to direct launches for this
// engine", sets capture_ok_ = false and re-runs the span with run_range. Capture is worth ~15%, so a
// machine with thin VRAM headroom can serve slower while reporting success.
// The state is read from that assignment, not inferred from counters, and it is sticky for the
// process: once a round has degraded, every later query reports DEGRADED.
typedef enum {
    AI_CAPTURE_OWN      = 0,  // own round capture: one graph, one sync per round
    AI_CAPTURE_DEGRADED = 1,  // capture failed at least once; running direct launches, ~15% lost
    AI_CAPTURE_NONE     = 2,  // capture deliberately disabled (AI_CAPTURE=none / --no-graphs)
} ai_capture_state;

typedef struct {
    const char * model_path;
    int64_t      ctx_cells_per_stream;  // --ctx; 0 = 8192 (unchanged), -1 = auto: the largest that fits
    int          n_streams;             // --parallel (1 or 2)
    int          expert_slots;          // 0 = no expert pool (ALL_GPU); -1 = auto from the VRAM left (doorbell)
    int          ws_mib;                // workspace cap; 0 = auto: the measured peak of this configuration
                                        // x1.25 + 32 MiB, calibrated once (~1 s, first create of a
                                        // configuration) and cached in ~/.cache/alphainfer (AI_WS_CACHE).
                                        // AI_WS_AUTO=0 uses the derived estimate instead (3.4-6x the peak).
    int          host_embd;             // host-gather token_embd: frees the table's VRAM, output bit-identical.
                                        // Refused together with mtp_k > 0, see below.
    int          kv_type_is_q4_0;       // KV cell type: 0 = f16, 1 = q4_0, 2 = q8_0, -1 = auto (fit + speed
                                        // rules). The name is kept so existing callers stay source- and
                                        // ABI-compatible.
    int          mtp_k;                 // 0 = plain greedy; > 0 = MTP drafts per round (exact mode caps it at 3 on the
                                        // default build). Decode with ai_engine_decode_mtp. The model must have an
                                        // MTP head (ai_engine_has_mtp); host_embd is refused with it.
    int          verbose;               // engine's own [ai] lines to stderr
} ai_engine_params;

// Create / destroy. Returns NULL on failure; *err receives a static message (never freed).
ai_engine * ai_engine_create(const ai_engine_params * p, const char ** err);
void        ai_engine_destroy(ai_engine * e);

// Current capture state. Sticky once degraded -- see the enum comment.
ai_capture_state ai_engine_capture_state(const ai_engine * e);

// Free device memory in MiB. Graph instantiation starts failing around 0.05 GiB free, so a serving
// floor of >= 0.25 GiB leaves a wide margin.
int64_t ai_engine_free_vram_mib(const ai_engine * e);

// The per-stream context this engine actually has, in cells. It can be less than ctx_cells_per_stream asked for:
// the first start of an uncalibrated configuration (calibration state 2) may clamp it for that start only.
int64_t ai_engine_ctx_cells(const ai_engine * e);

// How this start sized its workspace:
//   0 = from a measurement that covers this context: settled, serve from it.
//   1 = it recalibrated during create (sized from a smaller-context measurement, x1.25): it serves, but the next
//       start is sized from the new measurement and has more VRAM headroom.
//   2 = bootstrap: no measurement yet and the context may be clamped. Do not serve from it; restart.
// A launcher restarts until this returns 0. -1 on a NULL engine.
int ai_engine_calibration_state(const ai_engine * e);

// Whether this model has a next-n block wired. Gates the host_embd refusal.
int ai_engine_has_mtp(const ai_engine * e);

// Prompt phase for one stream. Runs the chunked prefill and returns the stream's first token in
// *out_first, which the caller feeds back as tokens[] on its first ai_engine_decode call.
// Required before any decode on that stream, and again after ai_engine_reset_stream.
int ai_engine_prefill(ai_engine * e, int stream_id,
                      const int32_t * tokens, int n_tokens, int32_t * out_first);

// The same prompt phase, fed in pieces. A whole-prompt ai_engine_prefill blocks every other stream's decode for its
// full duration (seconds for a long prompt). Feed the prompt in chunks and run ai_engine_decode_mtp for the other
// live streams between calls; the stall is then about one engine chunk.
//   - The first call on a stream that is not mid-prefill starts a new request (resets the slot, as ai_engine_prefill).
//   - Any n_tokens >= 0 per call; the engine runs its chunks at ai_engine_prefill's boundaries as soon as it holds a
//     token past each one, so the KV, the state and *out_first equal a one-shot prefill of the concatenation.
//   - is_last = 0: *out_first = -1 and the stream stays idle: do not name it in a decode round.
//     is_last = 1: runs the rest and returns the first token in *out_first, exactly as ai_engine_prefill.
//   - Only with at most one other live stream (n_streams <= 2, the compacted round): a wider round would compute
//     the mid-prefill stream and corrupt it, so ai_engine_decode_mtp returns AI_ENGINE_EFAIL instead.
//   - ai_engine_reset_stream abandons a partial prompt. Not available with DFlash2 (EFAIL).
int ai_engine_prefill_chunk(ai_engine * e, int stream_id, const int32_t * tokens, int n_tokens, int is_last,
                            int32_t * out_first);

// The prompt phase with image input. Like ai_engine_prefill, plus:
//   - pos4: 4 x n_tokens RoPE positions, section-major (t[0..n), y[0..n), x[0..n), z[0..n)) -- what llama.cpp's
//     mtmd helper builds for M-RoPE models: text rows (p, p, p, 0), image rows (t, t + row, t + col, 0).
//   - embd / embd_row / n_rows: n_rows f32 rows of ai_engine_n_embd(e) floats each (the projector's output) that
//     replace the token embedding of prompt rows embd_row[0..n_rows), strictly increasing. tokens[] at those rows
//     may be any valid id (a placeholder); they are not embedded.
// KV cells stay one per prompt row (n_tokens), but positions continue from max(t, y, x) + 1: decode calls pass that
// RoPE position (ai_engine_decode / _mtp positions[]), which is what llama.cpp-style servers already track.
// Returns AI_ENGINE_ENOTSUP with DFlash2.
int ai_engine_prefill_mm(ai_engine * e, int stream_id, const int32_t * tokens, int n_tokens, const int32_t * pos4,
                         const float * embd, const int32_t * embd_row, int n_rows, int32_t * out_first);
// The model's embedding width (the row length ai_engine_prefill_mm's embd takes).
int ai_engine_n_embd(const ai_engine * e);

// -- Sampling ----------------------------------------------------------------------------------------------------
// 1 when this engine can sample its target tokens (ai_engine_set_sampling), 0 when it is greedy only (Flash-Next for now).
int ai_engine_supports_sampling(const ai_engine * e);
// Sampling for one stream slot; temperature <= 0 is greedy (every slot's default). top_k 0 = off (it cuts up to 64),
// top_p 1 = off, min_p 0 = off. Tokens are drawn on the device from Philox(seed, position, slot), so the same seed and
// prompt give the same text, and MTP / lookup / DFlash2 keep their speed: a draft is accepted when it equals the drawn
// token, so the output follows the sampling distribution exactly. The setting survives ai_engine_reset_stream and
// prefill: set it before the request's prefill (its first token is drawn there), and back to greedy for a greedy one.
// Returns ENOTSUP on an engine without sampling, EINVAL for a bad slot.
int ai_engine_set_sampling(ai_engine * e, int stream_id, float temperature, int top_k, float top_p, float min_p,
                           uint64_t seed);

// One decode round. See notes 1-3 at the top of this header: in particular, this advances every
// open stream and out_sampled must have room for n_streams entries.
int ai_engine_decode(ai_engine * e,
                     const int32_t * tokens, const int32_t * positions,
                     const int32_t * seq_ids, int n_tokens,
                     const uint8_t * logits_mask,
                     int32_t * out_sampled, int * out_n_sampled);

// Record every decode profile up front, so no CUDA graph is instantiated after startup. Call once
// after create() and before the first ai_engine_prefill; *out_recorded receives the count (may be
// NULL). Resets every slot before returning, so a prefill is required afterwards either way.
//
// A lazy instantiate happens mid-serving, and a capture failure degrades to direct launches while
// still reporting success (~15% slower, no error). Doing every instantiate during init puts them all
// where the VRAM floor check runs.
//
// The set is the F ladder at the configured n_streams: ~10 profiles at a typical context, ~8 MiB each.
// It is not (B x F): B is fixed for an engine's lifetime.
// Returns ENOTSUP when mtp_k > 0: an MTP profile key depends on the adaptive k schedule, which is
// not knowable at init, so enumerating it would guess the schedule rather than record it.
int ai_engine_warm_profiles(ai_engine * e, int * out_recorded);

// -- MTP (mtp_k > 0) ---------------------------------------------------------------------------------------------
// The most tokens one ai_engine_decode_mtp round can emit: n_streams x (mtp_k + 1). Size out arrays with it.
int ai_engine_max_tokens_per_round(const ai_engine * e);
// The engine's effective MTP configuration after the policy: *out_mtp_k (may be lower than requested: exact verify
// caps K at 3) and *out_exact (1 = every emitted token is bit-identical to plain decode on sm_86). Either may be NULL.
int ai_engine_mtp_info(const ai_engine * e, int * out_mtp_k, int * out_exact);
// One MTP round. Inputs are exactly ai_engine_decode's: per open stream, the last token it emitted (the pending
// token; after prefill, *out_first) and that stream's position (prompt length after prefill, then + the number of
// tokens the stream emitted). Every open stream must be named (a round is batch-rigid).
// Outputs, in emission order: out_tokens[i] emitted by stream out_seq_ids[i], *out_n of them. A stream emits
// 1..mtp_k+1 tokens per round; the last one is its new pending token. out_capacity must be at least
// ai_engine_max_tokens_per_round(e) or the call returns EINVAL before running the round.
// Stopping is the caller's: if an end-of-generation token arrives mid-round, keep the tokens up to it and
// ai_engine_reset_stream the slot, since the engine has already run past it.
int ai_engine_decode_mtp(ai_engine * e,
                         const int32_t * tokens, const int32_t * positions,
                         const int32_t * seq_ids, int n_tokens,
                         int32_t * out_tokens, int32_t * out_seq_ids, int out_capacity, int * out_n);

// Reset one slot so the next request can reuse it: frees that stream's KV region and zeroes its
// recurrent state. Other slots are untouched -- KV regions are a static partition, so a slot's cells
// are disjoint by arithmetic. Must be followed by ai_engine_prefill before decoding that stream.
int ai_engine_reset_stream(ai_engine * e, int stream_id);

// Last error message for this handle, or "" if none. Static storage, never freed.
const char * ai_engine_last_error(const ai_engine * e);

#ifdef __cplusplus
}  // extern "C"
#endif
