#pragma once

// AlphaInfer GPU engine: AlphaInfer behind AlphaServe's serving layer
//
// AlphaServe already has the serving layer (HTTP, admission, the scheduler, slot management,
// streaming, the /health contract) and admits foreign engines at the SPSC queue seam;
// `fi_gpu_engine` (serve-fi-engine.h) is the in-tree precedent. So AlphaInfer needs an engine
// adapter, not a server of its own.
//
// This does not reimplement AlphaInfer and owns none of the decode path. It translates
// BatchRequest -> AlphaInfer round -> BatchResult and enforces the preconditions that a server
// must enforce and an engine cannot.
//
// Scope:
// Supported: greedy decode (temp<=0), np <= 2, MTP rounds (ai_engine_decode_mtp) or plain rounds.
// How the scheduler's batches map onto AlphaInfer:
//   - prompt chunks are buffered per slot; the chunk carrying the logits token triggers one
//     ai_engine_prefill(slot, whole prompt), whose first token is that slot's result;
//   - generating slots' tokens become one decode round over the open streams (ascending seq ids);
//     an MTP round emits 1..k+1 tokens per stream: the first goes in sampled[], the rest in slot_extra[];
//   - an open stream that is missing from a batch's generating tokens has finished (EOG / max_tokens /
//     cancel) and is reset; clear_seqs (slot reassigned) resets too.
// Not supported, and each one fails loudly rather than returning a plausible token:
//   - temperature / top_k / top_p sampling   (BatchRequest carries per-token params)
//   - the MTP seed schedule (samp_rng_base / samp_n_gen)
//   - hybrid prefix checkpoints (restore/snapshot) and prompts not starting at position 0
// Returning decode_error is the correct behaviour for all of these: a serving layer that silently
// ignores a sampling parameter produces confidently wrong text, which is worse than a 503.

#ifdef ALPHA_HAS_ALPHAINFER

#include "serve-types.h"
#include "serve-spsc-queue.h"
#include "llama.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

// The C API is AlphaInfer's own header (include/api/ai_engine.h of a deploy bundle or checkout), not a copy:
// a redeclared struct can drift (e.g. new fields in ai_engine_params) into an ABI mismatch that compiles cleanly.
#include "api/ai_engine.h"

namespace alphaserve {

//
// The adapter. Drop-in for gpu_engine / fi_gpu_engine at the SPSC queue seam.
//
class ai_gpu_engine {
public:
    ai_gpu_engine() = default;
    ~ai_gpu_engine();

    // init() enforces the preconditions a server must own. It returns false (refusing to serve)
    // rather than starting degraded, because every failure mode here is one that otherwise
    // reports success:
    //   * capture state != OWN                  -> decode silently runs slower
    //   * free VRAM < min_free_vram_mib          -> too close to the out-of-memory cliff
    //   * host_embd requested while MTP is live  -> an untested combination
    bool init(const ai_engine_params & params,
              const llama_vocab * vocab,
              spsc_queue<BatchRequest, 64> & in_queue,
              spsc_queue<BatchResult, 64> & out_queue,
              int max_batch,
              int64_t min_free_vram_mib = 256);   // 0.25 GiB, a safety margin above the cliff

    // Process one batch. Returns false only if the queue was empty.
    bool process_one();

    // /health contract
    // Capture state is added to the existing {"status","backend","intelligence","slots_active",
    // "slots_total"} payload as a new field rather than changing an existing one, so a degraded
    // engine is visible without reading logs and without a speed test.
    bool        is_degraded() const;         // capture_state() != OWN
    const char * capture_state_str() const;  // "own" | "degraded" | "none"
    // what the engine actually runs (after its policy), so a speed card cannot report an exact run as best-speed
    int          mtp_k() const { return mtp_k_; }
    const char * mtp_mode_str() const { return mtp_k_ <= 0 ? "plain" : (mtp_exact_ ? "exact" : "fast"); }
    int64_t     free_vram_mib() const;
    int         calibration_state() const;   // -2: library predates the getter
    int64_t     ctx_cells() const;           // -1: library predates the getter

    int64_t total_decodes() const { return n_decodes_; }
    int     n_streams() const { return n_streams_; }
    int64_t total_tokens()  const { return n_tokens_;  }

    // Why init() refused, for the operator. Empty when healthy.
    const std::string & last_error() const { return last_error_; }

private:
    // Returns false and fills last_error_ if the batch uses anything this increment does
    // not support. Silence here would mean wrong text, not slow text.
    bool check_supported(const BatchRequest & req);

    // fail the whole batch: every slot in it is reported so the scheduler force-finishes it, and its stream reset
    void fail_batch(const BatchRequest & req, BatchResult & res);

    // image input: encode the request's mtmd chunks (projector) and prefill stream s through ai_engine_prefill_mm
    // with the embeddings and their M-RoPE positions; false + last_error_ on failure
    bool prefill_mm(const BatchRequest & req, int s, int32_t & first);

    ai_engine * eng_   = nullptr;
    int n_streams_ = 0;
    int mtp_k_     = 0;                       // effective (after the engine's policy), 0 = plain rounds
    int mtp_exact_ = 0;                       // effective verify mode: 1 exact, 0 fast
    int max_round_tokens_ = 0;
    std::vector<std::vector<int32_t>> prompt_; // per stream: prompt chunks received so far
    std::vector<bool> open_;                   // per stream: prefilled and not reset
    std::vector<bool> midpf_;                  // per stream: chunked prefill started, last chunk not yet sent
    void * prefill_chunk_fn_ = nullptr;        // ai_engine_prefill_chunk (newer libraries only), resolved with dlsym
    void * prefill_mm_fn_    = nullptr;        // ai_engine_prefill_mm (image input, newer libraries only)
    const llama_vocab * vocab_ = nullptr;
    spsc_queue<BatchRequest, 64> * in_  = nullptr;
    spsc_queue<BatchResult,  64> * out_ = nullptr;
    int max_batch_ = 0;
    int64_t min_free_vram_mib_ = 256;
    std::string last_error_;
    int64_t n_decodes_ = 0;
    int64_t n_tokens_  = 0;
};

// The live engine, or nullptr when the AlphaInfer backend is not running.
// This exists so the /health handler can report capture state without serve-main.cpp
// having to thread the engine through every layer.
ai_gpu_engine * ai_engine_live();

} // namespace alphaserve

#endif // ALPHA_HAS_ALPHAINFER
