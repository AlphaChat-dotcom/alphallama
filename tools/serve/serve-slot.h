// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaServe slot state, owned by the scheduler thread.
//
// Unlike llama.cpp's server_slot, which is shared between all phases,
// this state is read and written only by the scheduler thread, which
// builds batches, checks stop conditions and manages the slot lifecycle.

#include "llama.h"
#include "common.h"
#include "serve-mm.h"
#include <memory>
#include <string>
#include <deque>
#include <vector>
#include <cstdint>
#include <random>

namespace alphaserve {

enum class SlotState {
    IDLE,           // no active request
    PREFILLING,     // processing prompt tokens (chunked)
    GENERATING,     // producing output tokens
    STOPPING,       // stop detected, finishing up
};

// Prefill chunk size, set by serve-main from --ubatch (must match the context's
// n_ubatch so each prefill chunk fills exactly one ubatch). Decode batches are
// 1-8 tokens and fit any ubatch - only prefill benefits from a larger value.
extern int g_serve_prompt_chunk;

// --ubatch-prefill N: prompts longer than one base chunk get N-sized chunks; the GPU
// engine grows n_ubatch on demand (rolls back if VRAM is tight). 0 = disabled.
// serve-main forces 0 under fused-infer, where the grow would displace the expert pool.
extern int g_serve_ubatch_prefill;

// Per-slot context budget (-c value = llama n_ctx_seq). Generation must stop before
// the KV runs out, since llama_decode fails past it.
// 0 = no cap (backends that manage their own KV limits).
extern int g_serve_ctx_per_slot;

// Context n_batch - the hard cap on tokens per llama_decode (asserts n_tokens_all <= n_batch).
// The scheduler budgets each packed BatchRequest against this so concurrent long-prompt prefills
// (or prefill co-packed with generating slots) can never overflow and abort the server.
extern int g_serve_n_batch;

struct serve_slot {
    int id = -1;
    int request_id = -1;
    SlotState state = SlotState::IDLE;

    // Prompt (scheduler-owned)
    std::vector<llama_token> prompt_tokens;
    std::shared_ptr<mm_prompt> mm;   // multimodal prompt (prompt_tokens are placeholders); null = text only
    int prompt_offset = 0;      // how many prompt tokens sent to GPU so far
    int prompt_chunk_size = g_serve_prompt_chunk; // tokens per prefill chunk (--ubatch)

    // Generation state (scheduler-owned)
    llama_token last_sampled = LLAMA_TOKEN_NULL;
    llama_pos   next_pos = 0;   // next token position in KV cache
    int n_generated = 0;
    int max_tokens = 200;

    // Stop detection (scheduler-owned)
    std::string generated_text;
    std::vector<std::string> stop_words;
    bool has_next_token = true;
    bool stop_length = false;   // check_stop ended the reply on a token / context limit (finish_reason "length")

    // Reasoning budget (scheduler-owned; see scheduler::think_step)
    int  think_budget = -1;                  // -1 = unlimited
    bool in_think = false;                   // inside <think>...</think>
    int  think_n = 0;                        // tokens generated inside the thinking block
    bool think_forced = false;               // the close was queued (once per request)
    std::vector<llama_token> think_close;    // forced at the budget
    std::deque<llama_token> force_q;         // forced tokens still to feed, one per step

    // Sampling params (per-request)
    float temperature = 0.7f;
    int   top_k = 40;
    float top_p = 0.95f;
    int   seed = -1;
    uint64_t rng_base = 0;   // per-request sampling RNG base (=seed, or random when seed<0)

    // true when this slot should be sampled (not greedy). temp<=0 or top_k==1 => greedy argmax.
    bool needs_sampling() const {
        return temperature > 0.0f && top_k != 1;
    }

    // Client tracking
    bool stream = false;
    void * http_connection = nullptr;

    // Prefix caching
    std::vector<llama_token> cached_prompt_tokens;  // prompt tokens in KV cache from previous request
    int prefix_match_len = 0;  // how many tokens matched the cached prefix
    bool disable_prefix_match = false;  // M-RoPE models: prefix match causes position crash

    // Recurrent-state checkpoint reuse for hybrid/recurrent models (ALPHA_HYBRID_PREFIX).
    // Attention KV can be cut back to any prefix; recurrent (DeltaNet/SSM) state cannot, and
    // llama_memory_hybrid::seq_rm tries the recurrent half first and, when it refuses, returns false
    // without truncating attention either - the old KV positions survive and the next prefill lands
    // below them. So on these models a slot may resume only at a position whose recurrent state was
    // snapshotted, and the engine restores that snapshot before truncating (restore puts the recurrent
    // cell at snap-1, after which seq_rm(snap, -1) succeeds on both halves).
    bool hybrid_snapshot = false;  // set by the scheduler when ALPHA_HYBRID_PREFIX is on for this model
    int  snap_target = 0;          // where to checkpoint this request (a chunk boundary is forced here)
    int  snap_pos = 0;             // checkpoint actually scheduled for this request (0 = none)
    int  snap_len_prev = 0;        // checkpoint that belongs to cached_prompt_tokens (prev request)
    bool snap_scheduled = false;   // checkpoint directive already attached to a batch
    bool pending_restore = false;  // next batch must carry a restore directive for this slot
    uint64_t last_assign_seq = 0;  // scheduler admission counter at last assign (LRU tie-break); not reset

    // Timing
    int64_t t_start = 0;
    int64_t t_first_token = 0;

    // Methods

    bool is_idle() const { return state == SlotState::IDLE; }
    bool is_generating() const { return state == SlotState::GENERATING; }

    void reset() {
        // A multimodal prompt leaves nothing reusable behind: its placeholders would match a later prompt's
        // placeholders whatever the images were, and its KV/recurrent state is not keyed by those tokens.
        if (mm) {
            prompt_tokens.clear();
            cached_prompt_tokens.clear();
            snap_pos = 0;
            snap_len_prev = 0;
            mm.reset();
        }
        // Save prompt tokens for prefix caching before clearing
        if (!prompt_tokens.empty()) {
            cached_prompt_tokens = std::move(prompt_tokens);
            // the checkpoint taken during that request now describes the cached prompt.
            // Transferred under the same condition as the tokens, so a second reset() (assign
            // calls reset() again on an already-reset slot) cannot overwrite it with 0.
            snap_len_prev = snap_pos;
        }
        snap_target = 0;
        snap_pos = 0;
        snap_scheduled = false;
        pending_restore = false;
        request_id = -1;
        state = SlotState::IDLE;
        prompt_tokens.clear();
        prompt_offset = 0;
        prefix_match_len = 0;
        last_sampled = LLAMA_TOKEN_NULL;
        next_pos = 0;
        n_generated = 0;
        max_tokens = 200;
        generated_text.clear();
        stop_words.clear();
        has_next_token = true;
        stop_length = false;
        think_budget = -1;
        in_think = false;
        think_n = 0;
        think_forced = false;
        think_close.clear();
        force_q.clear();
        temperature = 0.7f;
        top_k = 40;
        top_p = 0.95f;
        seed = -1;
        stream = false;
        http_connection = nullptr;
        t_start = 0;
        t_first_token = 0;
    }

    // Assign a new request to this slot.
    // Returns prefix_match_len: number of cached prompt tokens that match.
    // If > 0, caller should not clear KV cache for this slot.
    int assign(int req_id, const std::vector<llama_token> & tokens, int max_tok,
                float temp, int tk, float tp, int s,
                const std::vector<std::string> & stops, bool str) {
        // Check prefix match before reset (which saves cached_prompt_tokens)
        int match = 0;
        if (!cached_prompt_tokens.empty()) {
            int max_match = std::min((int)cached_prompt_tokens.size(), (int)tokens.size());
            while (match < max_match && cached_prompt_tokens[match] == tokens[match]) {
                match++;
            }
        }
        // captured before reset(); reset() only moves it when the slot still
        // holds an un-reset prompt, so this is the checkpoint that matches cached_prompt_tokens.
        const int prev_ckpt = snap_len_prev;
        const int prev_cached_len = (int)cached_prompt_tokens.size();

        reset();
        request_id = req_id;
        state = SlotState::PREFILLING;
        // Any prompt longer than one base chunk gets the larger prefill chunk (the engine grows n_ubatch on demand),
        // so it runs as one expert-streaming pass with the expert-pool borrow rather than several small chunks
        prompt_chunk_size = (g_serve_ubatch_prefill > g_serve_prompt_chunk &&
                             (int)tokens.size() > g_serve_prompt_chunk)
                          ? g_serve_ubatch_prefill : g_serve_prompt_chunk;
        prompt_tokens = tokens;
        max_tokens = max_tok;
        temperature = temp;
        top_k = tk;
        top_p = tp;
        seed = s;
        // Deterministic when seed>=0; otherwise a per-request random base so a seedless
        // request still samples (and is internally consistent) but varies run to run.
        if (s >= 0) rng_base = (uint64_t) s;
        else { std::random_device rd; rng_base = ((uint64_t) rd() << 32) ^ (uint64_t) rd() ^ (uint64_t) req_id; }
        stop_words = stops;
        stream = str;
        t_start = ggml_time_us();

        // hybrid/recurrent models resume only at the recurrent checkpoint: the
        // new prompt must contain the whole checkpointed prefix, and must extend past it (at least
        // one token has to be prefilled to produce logits). Anything else is a full reprocess,
        // because the recurrent state at any other position is unknown.
        if (hybrid_snapshot) {
            if (prev_ckpt >= 4 && prev_ckpt <= prev_cached_len &&
                match >= prev_ckpt && prev_ckpt < (int)tokens.size()) {
                prefix_match_len = prev_ckpt;
                prompt_offset    = prev_ckpt;
                next_pos         = prev_ckpt;
                pending_restore  = true;
            }
            return prefix_match_len;
        }

        // Apply prefix match: skip already-cached tokens.
        // Disabled for M-RoPE models (position tracking doesn't support partial reuse).
        if (match >= 4 && !disable_prefix_match) {  // minimum 4 tokens to bother caching
            int usable = std::min(match, (int)tokens.size() - 1);
            if (usable >= 4) {
                prefix_match_len = usable;
                prompt_offset = usable;
                next_pos = usable;
            }
        }

        return prefix_match_len;
    }

    // Check if a sampled token triggers stop
    bool check_stop(llama_token token, const llama_vocab * vocab) {
        // EOS check (fast)
        if (llama_vocab_is_eog(vocab, token)) {
            has_next_token = false;
            return true;
        }

        // Budget check (fast)
        if (n_generated >= max_tokens) {
            has_next_token = false;
            stop_length = true;
            return true;
        }

        // Context budget: finish (reason=length) before the per-slot KV runs out.
        // next_pos is the position the next token would occupy; llama rejects
        // positions >= n_ctx_seq.
        if (g_serve_ctx_per_slot > 0 && next_pos + 1 >= g_serve_ctx_per_slot) {
            has_next_token = false;
            stop_length = true;
            return true;
        }

        // Stop word check (string-based, slower but scheduler-owned, so no race)
        if (!stop_words.empty()) {
            std::string piece = common_token_to_piece(vocab, token, false);
            generated_text += piece;

            for (const auto & word : stop_words) {
                if (generated_text.size() >= word.size()) {
                    size_t pos = generated_text.rfind(word);
                    if (pos != std::string::npos && pos + word.size() == generated_text.size()) {
                        has_next_token = false;
                        return true;
                    }
                }
            }
        }

        return false;
    }

    // Get remaining prompt tokens for next prefill chunk
    int remaining_prompt() const {
        return (int)prompt_tokens.size() - prompt_offset;
    }

    // Get next chunk of prompt tokens
    // Does not transition state - the scheduler handles that in process_gpu_results().
    // max_tokens caps this chunk so the scheduler can keep the packed batch within the context's
    // n_batch budget (default = no extra cap beyond prompt_chunk_size). Advances prompt_offset by
    // exactly the returned size, so a capped chunk leaves the remainder for the next batch.
    std::vector<llama_token> next_prompt_chunk(int max_tokens = 0x7fffffff) {
        int start = prompt_offset;
        int size  = prompt_chunk_size;
        // ALPHA_SERVE_EVEN_CHUNKS (default 1): a prompt that needs n chunks gets n equal chunks instead of n-1 full
        // ones and a short tail, since under DRE expert streaming every chunk pays the same expert copies.
        static const bool even = [] { const char * e = getenv("ALPHA_SERVE_EVEN_CHUNKS"); return !e || atoi(e) != 0; }();
        const int left = (int) prompt_tokens.size() - prompt_offset;
        if (even && size > 0 && left > size) {
            const int n = (left + size - 1) / size;
            size = (left + n - 1) / n;
        }
        size = std::min(size, max_tokens);
        if (size < 1) size = 1;  // always make progress
        int end = std::min(prompt_offset + size, (int)prompt_tokens.size());
        prompt_offset = end;
        return std::vector<llama_token>(prompt_tokens.begin() + start, prompt_tokens.begin() + end);
    }
};

} // namespace alphaserve
