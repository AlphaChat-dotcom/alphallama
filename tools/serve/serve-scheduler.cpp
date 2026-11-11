// SPDX-License-Identifier: Apache-2.0
#include "serve-scheduler.h"
#include "common.h"
#include "build-info.h"   // llama_commit(): stamp every measurement with its source id

#include <algorithm>
#include <chrono>
#include <cstring>

namespace alphaserve {

void scheduler::init(int n_slots,
                     const llama_vocab * vocab,
                     spsc_queue<InferenceRequest, 256> & request_queue,
                     spsc_queue<BatchRequest, 64> & to_gpu,
                     spsc_queue<BatchResult, 64> & from_gpu,
                     spsc_queue<TokenEvent, 4096> & to_output) {
    slots_.resize(n_slots);
    for (int i = 0; i < n_slots; i++) {
        slots_[i].id = i;
    }
    vocab_ = vocab;
    {
        const auto o = common_tokenize(vocab, "<think>", false, true), c = common_tokenize(vocab, "</think>", false, true);
        if (o.size() == 1 && c.size() == 1) { think_open_ = o[0]; think_close_ = c[0]; }
    }
    req_in_ = &request_queue;
    to_gpu_ = &to_gpu;
    from_gpu_ = &from_gpu;
    to_output_ = &to_output;
}

void scheduler::start() {
    if (running_.load()) return;
    running_.store(true);
    thread_ = std::thread([this]() { loop(); });
}

void scheduler::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
}

int scheduler::active_slots() const {
    int n = 0;
    for (const auto & s : slots_) {
        if (!s.is_idle()) n++;
    }
    return n;
}

void scheduler::loop() {
    fprintf(stderr, "[alphaserve] scheduler thread started (%d slots)\n", (int)slots_.size());

    while (running_.load()) {
        tick();
    }

    fprintf(stderr, "[alphaserve] scheduler thread stopped (%lld requests, %lld tokens)\n",
            (long long)n_requests_, (long long)n_tokens_generated_);
}

// One iteration of the pipeline.
// This runs in a tight loop. Each tick:
//   1. Process GPU results (if available)
//   2. Admit new requests (if idle slots)
//   3. Build batch and send to GPU (if not already busy)

void scheduler::tick() {
    const uint64_t seen = g_serve_bell().seen();   // before reading any queue (see serve_bell)

    // Phase 0: cancel requests whose client is gone (disconnect / handler timeout); this
    // frees the slot + its KV pages instead of decoding to completion for nobody
    process_cancellations();

    // Phase 1: Process completed GPU results
    process_gpu_results();

    // Phase 2: Admit new requests into idle slots
    admit_new_requests();

    // Phase 3: Build and send batch to GPU
    if (!gpu_busy_) {
        build_and_send_batch();
    } else {
        // a batch is in flight: wait for its result (or a new request) instead of spinning a core
        g_serve_bell().wait(seen, std::chrono::microseconds(100));
    }
}

// Phase 0: cancellations.
// httplib worker threads push ids via cancel(); this thread (exclusive slot owner)
// force-finishes matching slots (same shape as the decode-error force-finish path)
// and drops matching queued requests. Unknown ids (request already finished) are ignored.

void scheduler::cancel(int request_id) {
    std::lock_guard<std::mutex> lock(cancel_mtx_);
    cancel_pending_.push_back(request_id);
}

void scheduler::process_cancellations() {
    std::vector<int> ids;
    {
        std::lock_guard<std::mutex> lock(cancel_mtx_);
        if (cancel_pending_.empty()) return;
        ids.swap(cancel_pending_);
    }
    for (int id : ids) {
        bool found = false;
        for (auto & slot : slots_) {
            if (slot.state == SlotState::IDLE || slot.request_id != id) continue;
            fprintf(stderr, "[alphaserve] slot %d: client gone — cancelling request %d after %d tokens\n",
                    slot.id, slot.request_id, slot.n_generated);
            TokenEvent event;
            event.slot_id = slot.id;
            event.request_id = slot.request_id;
            event.text = "";
            event.is_stop = true;
            event.is_eos = false;
            event.n_generated = slot.n_generated;
            to_output_->try_push(std::move(event));
            slot.reset();
            found = true;
            break;
        }
        if (!found) {
            for (auto it = waiting_queue_.begin(); it != waiting_queue_.end(); ++it) {
                if (it->request_id == id) {
                    fprintf(stderr, "[alphaserve] cancelling queued request %d (client gone)\n", id);
                    waiting_queue_.erase(it);
                    break;
                }
            }
        }
    }
}

// Reasoning budget: count the tokens generated inside <think>...</think>. Once the budget is spent, the pending token
// (sampled but not yet decoded -- the next batch feeds it) is replaced by the next token of the forced close, so the
// emitted text and the KV / recurrent state stay the same sequence. Tokens already decoded (an MTP window's accepted
// drafts) are never replaced: build_and_send_batch caps the window at the remaining budget, so none run past it.
llama_token scheduler::think_step(serve_slot & s, llama_token tok, bool pending) {
    if (s.think_budget < 0 || think_close_ == LLAMA_TOKEN_NULL) return tok;
    if (!s.think_forced && s.in_think && s.think_n >= s.think_budget) {
        s.think_forced = true;
        s.force_q.assign(s.think_close.begin(), s.think_close.end());
        fprintf(stderr, "[alphaserve] slot %d: reasoning budget %d reached -- closing the thinking block\n", s.id, s.think_budget);
    }
    if (pending && !s.force_q.empty()) {
        tok = s.force_q.front();
        s.force_q.pop_front();
    } else if (tok == think_close_) {
        s.force_q.clear();   // the model closed it itself
    }
    if (tok == think_open_)        s.in_think = true;
    else if (tok == think_close_)  s.in_think = false;
    else if (s.in_think)           s.think_n++;
    return tok;
}

// A prompt token. The row that takes logits (the prompt's last) picks the first generated token, so it carries the
// request's sampling settings as a generating row does; without them every sampled request began with the greedy token.
static void add_prompt_token(BatchRequest & batch, const serve_slot & slot, llama_token tok, int pos, bool need_logits) {
    const bool samp = need_logits && slot.needs_sampling();
    const uint64_t rng = samp ? (slot.rng_base ^ (0x9E3779B97F4A7C15ull * (uint64_t)(slot.n_generated + 1))) : 0;
    batch.add_token(tok, pos, slot.id, slot.id, need_logits,
                    samp ? slot.temperature : 0.0f, samp ? slot.top_k : 1, samp ? slot.top_p : 1.0f, rng,
                    samp ? slot.rng_base : 0ull, (int32_t) slot.n_generated, 0x7fffffff, samp ? slot.min_p : 0.0f);
}

// ALPHA_TOKEN_TRACE=<file>: the committed token stream, for offline drafter studies (lookup simulators read the same
// line shapes, without MTP's drafts):  "Q <req> <n> t1..tn" prompt ids | "F <req> <pos> <tok>" first token |
// "R <req> <pos0> <m> g1..gm" the tokens one engine step committed (an MTP window's accepted run + bonus)
static void token_trace(const char * kind, int req, int pos, const llama_token * t, int n) {
    static FILE * f = [] { const char * p = getenv("ALPHA_TOKEN_TRACE"); return p && *p ? fopen(p, "a") : (FILE *) nullptr; }();
    if (!f) return;
    fprintf(f, "%s %d %d %d", kind, req, pos, n);
    for (int i = 0; i < n; i++) fprintf(f, " %d", (int) t[i]);
    fputc('\n', f);
    fflush(f);
}

// Phase 1: process GPU results.
// Pop BatchResult from GPU thread, update slot state, check stops.

void scheduler::process_gpu_results() {
    BatchResult result;
    if (!from_gpu_->try_pop(result)) return;

    gpu_busy_ = false;

    // Decode failed (KV/context exhausted, CUDA error): force-finish every slot that
        // was in the batch so the HTTP layer gets a completion instead of hanging, and
        // the scheduler doesn't resend the same doomed batch forever.
    if (result.decode_error) {
        for (int i = 0; i < result.n_slots; i++) {
            int slot_id = result.slot_ids[i];
            if (slot_id < 0 || slot_id >= (int)slots_.size()) continue;
            auto & slot = slots_[slot_id];
            if (slot.state == SlotState::IDLE) continue;
            fprintf(stderr, "[alphaserve] slot %d: decode error at pos %d after %d tokens — finishing request %d\n",
                    slot.id, slot.next_pos, slot.n_generated, slot.request_id);
            TokenEvent event;
            event.slot_id = slot.id;
            event.request_id = slot.request_id;
            event.token = LLAMA_TOKEN_NULL;
            event.text = "";
            event.is_stop = true;
            event.is_eos = false;
            event.is_error = true;   // a truncated reply must not reach the client as a normal "stop"
            event.n_generated = slot.n_generated;
            to_output_->try_push(std::move(event));
            slot.reset();
        }
        return;
    }

    for (int i = 0; i < result.n_slots; i++) {
        int slot_id = result.slot_ids[i];
        if (slot_id < 0 || slot_id >= (int)slots_.size()) continue;

        auto & slot = slots_[slot_id];
        llama_token tok = result.sampled[i];

        if (slot.state == SlotState::PREFILLING) {
            if (slot.remaining_prompt() > 0) {
                // More chunks to send: intermediate chunk, no token sampled
                continue;
            }

            // All prompt tokens processed: transition to GENERATING
            slot.state = SlotState::GENERATING;
            tok = think_step(slot, tok, true);   // the first token is pending (budget 0 closes at once)
            token_trace("F", slot.request_id, slot.mm ? slot.mm->n_pos : (int) slot.prompt_tokens.size(), &tok, 1);
            slot.last_sampled = tok;
            // next_pos = prompt length. The next decode will use this position.
            // Don't increment here; the increment happens after subsequent decodes.
            slot.next_pos = slot.mm ? slot.mm->n_pos : (int)slot.prompt_tokens.size();   // images: n_pos < n_tokens
            slot.t_first_token = ggml_time_us();
            slot.n_generated++;
            n_tokens_generated_++;

            // Emit the first generated token, then continue (skip the increment below)
            std::string piece = common_token_to_piece(vocab_, tok, false);
            bool stopped = slot.check_stop(tok, vocab_);
            TokenEvent event;
            event.slot_id = slot.id;
            event.request_id = slot.request_id;
            event.token = tok;
            event.text = piece;
            event.is_stop = stopped;
            event.is_eos = llama_vocab_is_eog(vocab_, tok);
            event.is_length = stopped && slot.stop_length && !event.is_eos;
            event.n_generated = slot.n_generated;
            // cached prompt tokens: the engine's own count when it reports one (AlphaInfer prefix reuse), else this
            // slot's KV prefix match (0 when prefix caching is off or the prompt is multimodal)
            event.n_cached = (i < (int) result.slot_cached.size() && result.slot_cached[i] >= 0)
                             ? result.slot_cached[i] : std::max(0, slot.prefix_match_len);
            to_output_->try_push(std::move(event));
            if (stopped) {
                double total_ms = (ggml_time_us() - slot.t_start) / 1000.0;
                double gen_ms = (ggml_time_us() - slot.t_first_token) / 1000.0;
                double tok_s = slot.n_generated / (gen_ms / 1000.0);
                fprintf(stderr, "[alphaserve] slot %d: %d tokens in %.0fms (%.1f tok/s, prefill %.0fms) [build %s]\n",
                        slot.id, slot.n_generated, gen_ms, tok_s, total_ms - gen_ms, llama_commit());
                slot.reset();
            }
            continue;
        }

        if (slot.state != SlotState::GENERATING) continue;

        // Window patch-correction decode: the window's extra (exact) tokens follow `tok`. An engine that emits several
        // tokens for several slots in one batch (AlphaInfer MTP rounds) fills slot_extra[i] instead, parallel to
        // sampled[]; extra_tokens stays the single-slot channel.
        const std::vector<llama_token> & extras = (i < (int) result.slot_extra.size() && !result.slot_extra[i].empty())
                                                  ? result.slot_extra[i] : result.extra_tokens;
        tok = think_step(slot, tok, extras.empty());   // the last token of the step is the pending one
        llama_token step_toks[64]; int n_step = 0; const int step_pos0 = slot.next_pos;
        step_toks[n_step++] = tok;

        // Update slot with sampled token (for 2nd+ generated tokens)
        slot.last_sampled = tok;
        slot.n_generated++;
        slot.next_pos++;
        n_tokens_generated_++;

        // Detokenize (scheduler owns generated_text, no race)
        std::string piece = common_token_to_piece(vocab_, tok, false);

        // Check stop conditions (scheduler owns all state, no race). Over-subscription: also stop
        // if this slot exhausted the shared KV page pool this step: emit its (KV-valid) token, then
        // force-finish so the slot is released and its pages return to the pool for waiting requests.
        bool stopped = slot.check_stop(tok, vocab_);
        for (int es : result.exhausted_slots) { if (es == slot_id) { stopped = true; slot.stop_length = true; break; } }

        // Push token event to output thread
        TokenEvent event;
        event.slot_id = slot.id;
        event.request_id = slot.request_id;
        event.token = tok;
        event.text = piece;
        event.is_stop = stopped;
        event.is_eos = llama_vocab_is_eog(vocab_, tok);
        event.is_length = stopped && slot.stop_length && !event.is_eos;
        event.n_generated = slot.n_generated;
        to_output_->try_push(std::move(event));

        // emit the window's extra tokens
        for (size_t xi = 0; xi < extras.size(); xi++) {
            if (stopped) break;
            const llama_token xt = think_step(slot, extras[xi], xi + 1 == extras.size());
            if (n_step < 64) step_toks[n_step++] = xt;
            slot.last_sampled = xt;
            slot.n_generated++;
            slot.next_pos++;
            n_tokens_generated_++;
            std::string xp = common_token_to_piece(vocab_, xt, false);
            stopped = slot.check_stop(xt, vocab_);
            TokenEvent xe;
            xe.slot_id = slot.id;
            xe.request_id = slot.request_id;
            xe.token = xt;
            xe.text = xp;
            xe.is_stop = stopped;
            xe.is_eos = llama_vocab_is_eog(vocab_, xt);
            xe.is_length = stopped && slot.stop_length && !xe.is_eos;
            xe.n_generated = slot.n_generated;
            to_output_->try_push(std::move(xe));
        }

        token_trace("R", slot.request_id, step_pos0, step_toks, n_step);
        // Release slot if stopped
        if (stopped) {
            double total_ms = (ggml_time_us() - slot.t_start) / 1000.0;
            double gen_ms = (ggml_time_us() - slot.t_first_token) / 1000.0;
            double tok_s = slot.n_generated / (gen_ms / 1000.0);
            fprintf(stderr, "[alphaserve] slot %d: %d tokens in %.0fms (%.1f tok/s, prefill %.0fms) [build %s]\n",
                    slot.id, slot.n_generated, gen_ms, tok_s, total_ms - gen_ms, llama_commit());
            slot.reset();
        }
    }
}

// Phase 2: admit new requests.
// Pop InferenceRequest from HTTP thread, assign to idle slots.

void scheduler::admit_new_requests() {
    // Drain new requests from HTTP queue into waiting queue
    InferenceRequest req;
    while (req_in_->try_pop(req)) {
        waiting_queue_.push_back(std::move(req));
    }

    // Try to assign waiting requests to idle slots
    auto it = waiting_queue_.begin();
    while (it != waiting_queue_.end()) {
        // Find idle slot
        serve_slot * target = nullptr;
        for (auto & s : slots_) {
            if (s.is_idle()) {
                target = &s;
                break;
            }
        }
        // ALPHA_HYBRID_PREFIX: prefix-affinity slot choice. First-idle is fine with one slot, but at -np>1
        // two interleaved conversations keep landing on the slot the other one just used, so every turn
        // finds a foreign prompt cached and reuses nothing. Among idle slots, prefer the one offering the
        // longest usable reuse: for hybrid models that is the recurrent checkpoint (only if the new prompt
        // contains the whole checkpointed prefix), otherwise the raw prefix match. With no usable reuse
        // anywhere, take the least-recently-assigned idle slot (an empty slot first); first-idle would
        // always overwrite slot 0, so two alternating conversations would evict each other's cache every
        // turn. Changes only which idle slot is used, and only when prefix caching is on.
        if (target && prefix_caching_enabled && slots_.size() > 1) {
            int best = 0;
            for (auto & s : slots_)
                if (s.is_idle() && s.last_assign_seq < target->last_assign_seq) target = &s;
            const auto & pt = it->prompt_tokens;
            for (auto & s : slots_) {
                if (!s.is_idle() || s.cached_prompt_tokens.empty()) continue;
                const int lim = (int)std::min(s.cached_prompt_tokens.size(), pt.size());
                int m = 0;
                while (m < lim && s.cached_prompt_tokens[m] == pt[m]) m++;
                int usable = 0;
                if (hybrid_snapshot_) {
                    const int ck = s.snap_len_prev;
                    if (ck >= 4 && m >= ck && ck < (int)pt.size()) usable = ck;
                } else {
                    usable = std::min(m, (int)pt.size() - 1);
                }
                if (usable > best) { best = usable; target = &s; }
            }
        }

        if (!target) {
            break;  // No more idle slots; remaining requests stay in waiting queue
        }

        // Assign request to slot (detects prefix match)
        int prefix_len = target->assign(it->request_id, it->prompt_tokens, it->max_tokens,
                       it->temperature, it->top_k, it->top_p, it->seed,
                       it->stop_words, it->stream);
        target->http_connection = it->http_connection;
        target->min_p = it->min_p;
        target->last_assign_seq = ++assign_counter_;
        token_trace("Q", it->request_id, 0, it->prompt_tokens.data(), (int) it->prompt_tokens.size());
        if (it->reasoning_budget >= 0 && think_close_ != LLAMA_TOKEN_NULL && !it->think_close.empty()) {
            target->think_budget = it->reasoning_budget;
            target->in_think = it->think_open;
            target->think_close = it->think_close;
        }
        // multimodal prompt: no prefix reuse and no recurrent checkpoint (placeholders are not the prompt), fresh KV
        target->mm = it->mm;
        if (target->mm) {
            prefix_len = 0;
            target->prefix_match_len = 0;
            target->prompt_offset = 0;
            target->next_pos = 0;
            target->pending_restore = false;
        }
        n_requests_++;

        // ALPHA_HYBRID_PREFIX: choose this request's recurrent checkpoint just after the last
        // end-of-turn token of the prompt. Every chat template closes a completed turn with an EOG
        // token (<|im_end|>, <|eot_id|>, <end_of_turn>, ...), and everything through it (system and
        // every finished turn) re-renders identically on the next request. What follows it is the
        // generation prompt (assistant header, an empty <think></think> under think:false), which is
        // exactly what a re-rendered history changes. Not "last control token": if the think tags
        // are control tokens that lands inside the think block and reuse would never trigger.
        // With no EOG near the end (raw /v1/completions), back off a few tokens so a BPE merge across
        // the prompt/response boundary on the next request cannot push the match below it.
        if (hybrid_snapshot_ && !target->mm) {
            const auto & pt = it->prompt_tokens;
            const int n = (int)pt.size();
            int s = std::max(4, n - 4);
            for (int i = n - 1; i >= 0 && n - i <= 512; --i) {
                if (llama_vocab_is_eog(vocab_, pt[i])) { s = i + 1; break; }
            }
            // never leave a one-token final chunk: a 1-token batch with logits is what a decode step looks like, and the
            // engine's MTP / window paths would take it for one (very short raw prompts then fail with "inconsistent
            // sequence positions" after an earlier request)
            s = std::min(s, n - 2);
            if (s < 1) s = 0;   // a 1-2 token prompt: no checkpoint
            // Round down to a multiple of the prefill chunk size, so the forced boundary coincides with
            // one a full reprocess has anyway and a reusing turn resumes on the same chunk grid. Prefill
            // numerics depend on where chunks start, so an unaligned boundary can change output even on
            // the first turn. Costs < 1 chunk of re-prefill per turn. ALPHA_HYBRID_PREFIX_ALIGN=0 keeps
            // the longest reuse instead.
            static const bool align = [] {
                const char * e = getenv("ALPHA_HYBRID_PREFIX_ALIGN");
                return !(e && strcmp(e, "0") == 0);
            }();
            const int C = target->prompt_chunk_size;
            // A prompt longer than one chunk is cut on the grid next_prompt_chunk() really uses: ALPHA_SERVE_EVEN_CHUNKS
            // makes n equal chunks of E = ceil(left / n) from the resume point, not multiples of C. Aligning to C would add
            // a boundary off that grid: one extra batch, and under DRE expert streaming every batch pays a near-full
            // expert pass.
            static const bool even = [] { const char * e = getenv("ALPHA_SERVE_EVEN_CHUNKS"); return !e || atoi(e) != 0; }();
            const int off = target->prompt_offset, left = n - off;
            if (align && even && C > 1 && left > C) {
                // replay next_prompt_chunk's sizes (it re-splits what is left each call, so the chunks can differ by one
                // token; a once-computed grid could put the checkpoint one token past a boundary -> a 1-token batch)
                int b = off, last = -1;
                while (b < n) {
                    int sz = C, l = n - b;
                    if (l > sz) { const int nch = (l + sz - 1) / sz; sz = (l + nch - 1) / nch; }
                    b = std::min(n, b + std::max(1, sz));
                    if (b <= s) last = b;
                }
                if (last >= off + 4) s = last;
            } else if (align && C > 1 && s - s % C >= 4) {
                s -= s % C;
            }
            target->snap_target = s;
        }

        if (prefix_len > 0 && prefix_caching_enabled) {
            // Prefix match: keep KV cache, only prefill suffix tokens
            fprintf(stderr, "[alphaserve] slot %d: assigned request %d (%d tokens, prefix cached %d)\n",
                    target->id, it->request_id, (int)it->prompt_tokens.size(), prefix_len);
        } else {
            // Reset prefix match if caching disabled (ggml can't handle partial prefill)
            if (prefix_len > 0) {
                target->prefix_match_len = 0;
                target->prompt_offset = 0;
                target->next_pos = 0;
            }
            // No match: clear KV cache for this slot's sequence
            pending_kv_clears_.push_back(target->id);
            fprintf(stderr, "[alphaserve] slot %d: assigned request %d (%d prompt tokens)\n",
                    target->id, it->request_id, (int)it->prompt_tokens.size());
        }

        it = waiting_queue_.erase(it);
    }
}

// Phase 3: build batch and send to GPU.
// Collect tokens from all active slots, build BatchRequest, push to GPU.

void scheduler::build_and_send_batch() {
    BatchRequest batch;
    batch.clear();

    int n_prefill_in_batch = 0;
    // Prefill slots per batch. Default 1 = interleave (decode-priority). Env-tunable, clamped
    // [1,8]. Coalescing all pending prefills does not widen decode batches: mid-run, slots finish
    // staggered, so only one prefill is ever pending.
    static const int max_prefill_per_batch = [] {
        const char * e = getenv("ALPHA_PREFILL_PER_BATCH");
        int v = e ? atoi(e) : 1;
        if (v < 1) v = 1;
        if (v > 8) v = 8;
        return v;
    }();

    // Hard token budget: the packed BatchRequest must never exceed the context's n_batch, or
    // llama_decode aborts the whole server (GGML_ASSERT(n_tokens_all <= n_batch)). A single prefill
    // chunk already equals n_batch (chunk == n_ubatch == n_batch), so co-packing any other token
    // (a generating slot's token, or a 2nd prefill) would overflow it. So two passes are budgeted
    // against n_batch: generating first (cheap, keeps active generation flowing), then prefill capped
    // to the remaining room. Leftover prefill tokens roll to the next batch.
    const int n_batch_budget = alphaserve::g_serve_n_batch > 0 ? alphaserve::g_serve_n_batch : 512;

    // ALPHA_PREFILL_ALONE (default on under ALPHA_DRE): a prompt chunk is not co-packed with other slots' decode
    // tokens; batches alternate decode-only / prefill-only while both are pending. Co-packed, the batch holds 2+
    // sequences and the hybrid recurrent model splits it into equal-sequence micro-batches whose compute buffer
    // grows to gigabytes for a long prompt next to one decode token, which can run out of memory with the expert
    // pool lent for the prefill.
    static const bool prefill_alone = [] {
        if (const char * e = getenv("ALPHA_PREFILL_ALONE")) return atoi(e) != 0;
        const char * d = getenv("ALPHA_DRE");
        return d && atoi(d) != 0;
    }();
    // ALPHA_DECODE_SHARE (default 0.5): while one slot generates and another's prompt is pending, the share of wall
    // time the decode-only batches get: after each prompt piece, decode batches run until their time reaches
    // share/(1-share) x that piece's time (at least one). 0.5 = as long decoding as prefilling, so a reply in progress
    // does not freeze for a whole long prompt (a 1:1 batch alternation is not enough when one prompt piece costs
    // as much as dozens of decode rounds). Batch time = the gap between successive builds.
    // ALPHA_PREFILL_PIECE_DECODING (default 1024): prompt piece size while another slot is generating.
    static const double decode_share = [] {
        const char * e = getenv("ALPHA_DECODE_SHARE");
        const double v = e ? atof(e) : 0.5;
        return v < 0.0 ? 0.0 : (v > 0.95 ? 0.95 : v);
    }();
    static const int piece_decoding = [] { const char * e = getenv("ALPHA_PREFILL_PIECE_DECODING"); return e ? atoi(e) : 1024; }();
    // ALPHA_PREFILL_SHORT (default 1024 tokens, 0 = off): a pending prompt this short prefills at once instead of
    // waiting for the decode share. The share is owed against the previous prompt piece, so a short prompt arriving
    // while another slot decodes would wait several times that piece with its stream idle, while the reply in progress
    // decodes alone at batch 1. A one-piece prompt stalls the other stream for one short prefill; long prompts keep
    // the alternation between their pieces.
    static const int prefill_short = [] { const char * e = getenv("ALPHA_PREFILL_SHORT"); return e ? atoi(e) : 1024; }();
    enum batch_kind { BK_NONE, BK_DEC, BK_PF };
    static batch_kind last_kind = BK_NONE;
    static auto last_build = std::chrono::steady_clock::now();
    static double pf_ms_last = 0.0, dec_ms_since_pf = 0.0;
    const auto t_build = std::chrono::steady_clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(t_build - last_build).count();
    last_build = t_build;
    if (last_kind == BK_PF) pf_ms_last = dt_ms; else if (last_kind == BK_DEC) dec_ms_since_pf += dt_ms;
    bool skip_gen = false, skip_pf = false, has_gen = false;
    batch_kind kind = BK_NONE;
    if (prefill_alone) {
        bool has_pf = false, pf_short = false;
        for (auto & slot : slots_) {
            if (slot.is_idle()) continue;
            if (slot.state == SlotState::GENERATING) has_gen = true;
            if (slot.state == SlotState::PREFILLING && slot.remaining_prompt() > 0) {
                has_pf = true;
                // a whole prompt that fits in one piece (not the tail of a long prompt already being pieced)
                if (prefill_short > 0 && slot.prompt_offset == 0 && slot.remaining_prompt() <= prefill_short) pf_short = true;
            }
        }
        if (has_gen && has_pf) {
            const double owed = decode_share / (1.0 - decode_share) * pf_ms_last;
            if (pf_short && last_kind != BK_PF)                    { skip_gen = true; kind = BK_PF; dec_ms_since_pf = 0.0; }
            else if (last_kind == BK_DEC && dec_ms_since_pf >= owed) { skip_gen = true; kind = BK_PF; dec_ms_since_pf = 0.0; }
            else                                                  { skip_pf = true; kind = BK_DEC; }
        }
    }
    // a multimodal prompt prefills alone (the engine evaluates it outside the token batch)
    for (auto & slot : slots_) {
        if (slot.mm && slot.state == SlotState::PREFILLING && slot.remaining_prompt() > 0) {
            skip_gen = true; skip_pf = false; kind = BK_PF; dec_ms_since_pf = 0.0;
            break;
        }
    }
    last_kind = kind;

    // Pass 1: generating slots (1 token each). Must not be starved by a prefill chunk that fills n_batch.
    for (auto & slot : slots_) {
        if (skip_gen) break;
        if (slot.is_idle() || slot.state != SlotState::GENERATING) continue;
        if (batch.n_tokens >= n_batch_budget) break;  // only if n_parallel > n_batch (pathological)
        // Sampling params for this generating token. Greedy (temp<=0) unless the request asked
        // for sampling. RNG seed = rng_base ^ (GOLDEN * (n_generated + 1)) -- reproducible for a
        // fixed seed, distinct per generated token. 0x9E3779B97F4A7C15 = SplitMix64 golden-ratio
        // mix constant.
        // Keyed on the generation counter, not the position. rng_base and n_generated are passed
        // through as well, because MTP emits several tokens from one batch entry and must reproduce
        // the seed each of them would have had, which samp_rng alone cannot give it.
        const bool samp = slot.needs_sampling();
        const uint64_t rng = samp ? (slot.rng_base ^ (0x9E3779B97F4A7C15ull * (uint64_t)(slot.n_generated + 1))) : 0;
        // Reasoning budget: an MTP window may not run past the budget (its tokens are decoded, so they cannot be
        // replaced), and a forced close goes one token per step
        int n_left = std::max(1, slot.max_tokens - slot.n_generated);
        // ...nor past the slot's context: check_stop guards one token (next_pos + 1 < n_ctx), but an MTP verify batch
        // writes n_left cells from next_pos. At the context edge a multi-token verify would fail to find a memory slot,
        // and the following full scheduler re-reserve could run out of memory.
        if (g_serve_ctx_per_slot > 0) n_left = std::min(n_left, std::max(1, g_serve_ctx_per_slot - 1 - (int) slot.next_pos));
        if (slot.think_budget >= 0 && think_close_ != LLAMA_TOKEN_NULL) {
            if (!slot.force_q.empty())                    n_left = 1;
            else if (slot.in_think && !slot.think_forced) n_left = std::min(n_left, std::max(1, slot.think_budget - slot.think_n));
        }
        batch.add_token(slot.last_sampled,
                       slot.next_pos,
                       slot.id,
                       slot.id,
                       true,  // always need logits for generating slots
                       samp ? slot.temperature : 0.0f,
                       samp ? slot.top_k : 1,
                       samp ? slot.top_p : 1.0f,
                       rng,
                       samp ? slot.rng_base : 0ull,
                       (int32_t) slot.n_generated,
                       (int32_t) n_left,
                       samp ? slot.min_p : 0.0f);
        batch.n_slots++;
    }

    // Pass 2: prefill chunks with the remaining budget, each capped so the total stays <= n_batch.
    for (auto & slot : slots_) {
        if (skip_pf) break;
        if (slot.is_idle() || slot.state != SlotState::PREFILLING) continue;
        if (slot.remaining_prompt() <= 0) continue;  // all chunks sent, awaiting GPU result

        // multimodal prompt: the whole prompt as one batch of its own; the tokens are placeholders the engine skips
        if (slot.mm) {
            if (batch.n_tokens > 0) continue;   // next batch
            const int n = (int) slot.prompt_tokens.size();
            for (int i = 0; i < n; i++) {
                add_prompt_token(batch, slot, slot.prompt_tokens[i], i, i == n - 1);
            }
            slot.prompt_offset = n;
            batch.prompt_end = n;
            batch.mm = slot.mm;
            batch.mm_slot = slot.id;
            batch.n_slots++;
            n_prefill_in_batch++;
            break;
        }

        if (n_prefill_in_batch >= max_prefill_per_batch) continue;
        const int budget = n_batch_budget - batch.n_tokens;
        if (budget <= 0) break;  // no room left in this batch; this slot prefills next iteration
        n_prefill_in_batch++;

        // ALPHA_HYBRID_PREFIX: a reused slot must restore its recurrent checkpoint before its first
        // suffix token is decoded; the directive rides with that very batch.
        if (slot.pending_restore) {
            batch.restore_seqs.push_back(slot.id);
            slot.pending_restore = false;
        }
        // ALPHA_HYBRID_PREFIX: force a chunk boundary at the checkpoint so the recurrent state at
        // exactly snap_target exists after some batch and can be snapshotted.
        int cap = budget;
        if (has_gen && piece_decoding > 0) cap = std::min(cap, piece_decoding);   // ALPHA_PREFILL_PIECE_DECODING
        const bool want_snap = slot.hybrid_snapshot && !slot.snap_scheduled &&
                               slot.snap_target > slot.prompt_offset;
        if (want_snap) cap = std::min(cap, slot.snap_target - slot.prompt_offset);

        // Add chunk of prompt tokens, capped to the remaining n_batch budget
        auto chunk = slot.next_prompt_chunk(cap);
        bool is_last_chunk = (slot.remaining_prompt() <= 0);
        batch.prompt_end = std::max(batch.prompt_end, (int) slot.prompt_tokens.size());
        // first chunk of a prompt (see BatchRequest); a single-chunk prompt too (ALPHA_PLE_HEAD: its rows are
        // gathered while the engine borrows and reserves)
        if ((!is_last_chunk || chunk.size() >= 64) && slot.prompt_offset == (int) chunk.size() && batch.prompt_ahead.empty()) {
            batch.prompt_ahead = slot.prompt_tokens;
        }

        if (want_snap && slot.prompt_offset == slot.snap_target) {
            batch.snapshot_seqs.push_back(slot.id);
            slot.snap_scheduled = true;
            slot.snap_pos = slot.snap_target;
        } else if (slot.hybrid_snapshot && !slot.snap_scheduled && !want_snap &&
                   slot.prefix_match_len > 0 && slot.snap_target <= slot.prefix_match_len) {
            // The new checkpoint target lies inside the reused prefix (state already past it): the
        // engine still holds the restored snapshot at prefix_match_len, so keep that as this
            // request's checkpoint instead of losing reuse for the next turn.
            slot.snap_scheduled = true;
            slot.snap_pos = slot.prefix_match_len;
        }

        for (int i = 0; i < (int)chunk.size(); i++) {
            bool need_logits = is_last_chunk && (i == (int)chunk.size() - 1);
            add_prompt_token(batch, slot, chunk[i], slot.prompt_offset - (int)chunk.size() + i, need_logits);
        }
        batch.n_slots++;
    }

    if (batch.n_tokens == 0) {
        // No work: wait for the next request or result instead of spinning
        g_serve_bell().wait(g_serve_bell().seen(), std::chrono::microseconds(100));
        return;
    }

    // Attach pending KV cache clears
    if (!pending_kv_clears_.empty()) {
        batch.clear_seqs = std::move(pending_kv_clears_);
        pending_kv_clears_.clear();
    }

    // Debug: log first batch only (avoid per-token fprintf overhead)
    {
        static int batch_log_count = 0;
        if (batch_log_count < 3 && batch.n_tokens > 0) {
            fprintf(stderr, "[sched] batch: %d tok, positions=[", batch.n_tokens);
            for (int i = 0; i < std::min(batch.n_tokens, 8); i++)
                fprintf(stderr, "%s%d@s%d", i ? "," : "", batch.positions[i], batch.seq_ids[i]);
            if (batch.n_tokens > 8) fprintf(stderr, "...");
            fprintf(stderr, "], clears=%d\n", (int)batch.clear_seqs.size());
            batch_log_count++;
        }
    }

    // Push to GPU
    if (to_gpu_->try_push(std::move(batch))) {
        gpu_busy_ = true;
    }
}

} // namespace alphaserve
