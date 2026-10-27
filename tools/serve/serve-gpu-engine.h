#pragma once

// AlphaServe GPU Engine Thread
//
// The simplest thread in the pipeline:
//   1. Wait for BatchRequest from scheduler (via SPSC queue)
//   2. Build llama_batch from request
//   3. llama_decode(ctx, batch) - GPU forward pass + backend sampling
//   4. Read sampled tokens via llama_get_sampled_token_ith()
//   5. Push BatchResult to scheduler (via SPSC queue)
//
// Owns: llama_context, KV cache, backend sampler chains
// Does not own: slot state, request queue, HTTP connections

#include <functional>
#include "serve-types.h"
#include "alpha_mole_speculative.h"
#include "serve-spsc-queue.h"
#include "llama.h"

#include <deque>
#include <thread>
#include <atomic>

namespace alphaserve {

class gpu_engine {
public:
    gpu_engine() = default;
    ~gpu_engine() { stop(); }

    // Initialize with context and queue references
    void init(llama_context * ctx,
              spsc_queue<BatchRequest, 64> & in_queue,
              spsc_queue<BatchResult, 64> & out_queue);

    // Start GPU thread
    void start();

    // Stop GPU thread
    void stop();

    bool is_running() const { return running_.load(); }

    // Process one batch synchronously (for single-thread mode)
    // Returns true if a batch was processed
    bool process_one();
    // ALPHA_SERVE_PRE_BATCH_FLUSH: called after a batch is taken and before it runs, so the server routes the token
    // events already queued. Otherwise the prompt's first token (queued by the scheduler just before the next batch)
    // would wait for that batch's whole MTP window loop.
    std::function<void()> pre_batch_hook;

    // Stats
    int64_t total_decodes() const { return n_decodes_; }
    int64_t total_tokens() const { return n_tokens_; }

private:
    void loop();
    bool decode_batch(const BatchRequest & req, BatchResult & result);
    bool mm_prefill(const BatchRequest & req, BatchResult & result);   // multimodal prompt (serve-mm.h)
    void mtp_window_extend(const BatchRequest & req, BatchResult & result); // ALPHA_MTP
    alpha_recurrent_snapshot_t snap_;   // pre-window recurrent state (can be ~150 MiB per cell)

    // ALPHA_DRAFT_MODEL: external draft model for MTP.
    // The MTP head drafts every depth from the same carried hidden h_t, so its acceptance decays
    // with depth by construction. A small model chained on its own outputs decays much less, and a
    // draft step is tiny next to a verify, so depth is nearly free.
    // Unset ALPHA_DRAFT_MODEL => dctx_ stays null and the head path runs unchanged.
    // The vocab must match exactly (same size and tokenizer as the target), so tokens transfer
    // with no remap.
    bool draft_init();                  // from init(); false => silently keep the head path
    void draft_free();
    // Bring dctx_ up to the target's committed history, then chain m drafts from `last`.
    // Empty return means "no drafter this window" and the caller falls back to the head.
    std::vector<llama_token> draft_chain(llama_token last, llama_pos pos0, int m);
    // Re-sync dctx_ after a partial accept: the drafter advanced m but only some were committed.
    void draft_commit(const std::vector<llama_token> & committed, llama_pos pos0);

    llama_model   * dmodel_ = nullptr;
    llama_context * dctx_   = nullptr;
    llama_batch     dbatch_ = {};
    alpha_recurrent_snapshot_t dsnap_;  // the drafter may be hybrid too: seq_rm alone will not rewind it
    llama_pos       dpos_   = 0;        // positions dctx_ has actually consumed
    int32_t         dseq_   = -1;       // which target sequence dhist_ mirrors (-1 = none yet)
    std::vector<llama_token> dhist_;    // target's token history, index == position

    // ALPHA_MTP_NGRAM: prompt-lookup drafter
    // Per-slot committed-token ring. dhist_ above looks similar but is a single vector tied to the
    // external drafter (dctx_), so reusing it would corrupt across slots at -np > 1.
    std::vector<std::vector<llama_token>> ngram_hist_;   // [slot] -> recent committed tokens
    // ALPHA_MTP_NGRAM_ADAPT: adaptive gate. Estimates yield-per-millisecond per candidate depth from
    // the free lookup on every window, including windows it declines to route, and routes only when
    // the best depth beats the head. Learning from declined windows matters: a controller that only
    // learns from the action it stopped taking cannot recover.
    struct ngram_adapt {
        struct pending {                      // a lookup chain being scored against the committed stream
            std::vector<llama_token> chain;
            int    pos0      = 0;             // absolute token position this chain was drawn at
            int    prefix_ok = 0;             // longest correct prefix seen so far
            size_t n_scored  = 0;
            bool   broken    = false;
        };
        std::deque<pending> pend;
        double yld_sum[33] = {0};             // tokens-per-ms at each candidate depth
        // the effective sample count must be a double: as a long, the forgetting update
        // cnt = (long)(cnt * 0.95) takes 1 -> 0.95 -> 0, then +1 = 1, pinning the count at 1
        // forever so it never reaches min-obs and the gate never routes.
        double yld_cnt[33] = {0};
        double head_sum = 0;  double head_cnt = 0;
        // running least-squares fit of verify_us on verify width: verify ~= a + b*w (self-calibrating,
        // so no machine constant is baked in; break-even is derived on the machine it runs on)
        double vtab_sum[33] = {0}; long vtab_cnt[33] = {0};   // observed verify_us by width
        double snap_us = 0;   long snap_n = 0;
        double draft_us = 0;  long draft_steps = 0;
        long   routed = 0, declined = 0;
        double prefix_sum = 0; long prefix_n = 0;   // diagnostic: is the chain scoring aligned?
        // position bins: routing changes which positions get scored (a routed window advances
        // further), so a decay plotted against window index can be a position artefact. Binned by
        // absolute token position, the routed and inert arms must lie on one curve if there is no
        // feedback.
        double posb_sum[16] = {0}; long posb_cnt[16] = {0};   // 64-token bins
        long   prefix_zero = 0;
    };
    std::vector<ngram_adapt> ngram_adapt_;
    void ngram_adapt_score(int slot_id, llama_token tok);
    int  ngram_adapt_pick(int slot_id, int max_d) const;
    static double ngram_verify_us(const ngram_adapt & A, double w);

    // Most recent earlier occurrence of the last `want` tokens; returns the match length achieved
    // (3 -> 2 -> 1, else 0) and fills `out` with what followed. The return value is the routing
    // signal: a 3-token match means this context literally occurred before; a shorter one means the
    // lookup is guessing. Cheap: backwards CPU scan over a bounded window, no GPU.
    int  ngram_lookup(int slot_id, llama_token last, int m, std::vector<llama_token> & out) const;
    void ngram_commit(int slot_id, const llama_token * toks, int n);
    bool            draft_bad_ = false; // hard failure: stop trying for the rest of the run

    llama_context * ctx_ = nullptr;
    spsc_queue<BatchRequest, 64> * in_ = nullptr;
    spsc_queue<BatchResult, 64> * out_ = nullptr;

    std::thread thread_;
    std::atomic<bool> running_{false};

    // Owned batch (reused across iterations)
    // ALPHA_MTP: backend (GPU) greedy sampling, one chain per sequence. Without a registered
    // sampler llama_context::needs_raw_logits() is always true, so every decode would ship
    // n_out * n_vocab floats to the host and argmax them there.
    std::vector<llama_sampler *> samp_chain_;    // greedy chain per seq id (lazily built)
    std::vector<int8_t>          samp_backend_;  // -1 unknown, 0 = host logits, 1 = backend greedy
    void backend_sampler_sync(const BatchRequest & req);

    llama_batch batch_ = {};
    int max_batch_size_ = 2048;

    // Stats
    int64_t n_decodes_ = 0;
    int64_t n_tokens_ = 0;
};

// ALPHA_MTP: turn MTP windows off for the life of the process (serve-main calls this when f16 KV +
// the MTP reserve cannot fit; the engine also latches it if the rollback mirror fails to allocate).
void mtp_disable(const char * why);

} // namespace alphaserve
