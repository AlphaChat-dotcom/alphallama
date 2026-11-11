// SPDX-License-Identifier: Apache-2.0
// AlphaInfer engine adapter. See serve-ai-engine.h for why this shape and what it
// deliberately does not support.

#ifdef ALPHA_HAS_ALPHAINFER

#include "serve-ai-engine.h"
#include "serve-mm.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#if defined(__linux__)
#include <dlfcn.h>
#endif
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <string>
#include <map>

// Sampling entry points, weak: an in-tree AlphaInfer (statically linked, symbols not exported) resolves them at link
// time, where dlsym(RTLD_DEFAULT) would not find them; an older external library without them leaves them null.
#if defined(_MSC_VER)
#define ALPHA_AI_WEAK
#else
#define ALPHA_AI_WEAK __attribute__((weak))
#endif
extern "C" {
int ai_engine_supports_sampling(const ai_engine * e) ALPHA_AI_WEAK;
int ai_engine_set_sampling(ai_engine * e, int stream_id, float temperature, int top_k, float top_p, float min_p,
                           uint64_t seed) ALPHA_AI_WEAK;
int64_t ai_engine_prompt_cached(const ai_engine * e, int stream_id) ALPHA_AI_WEAK;
}

// usage.prompt_tokens_details.cached_tokens: the prompt tokens stream s's last prefill took from the engine's kept state
// (AlphaInfer prefix reuse), recorded parallel to res.sampled for the scheduler. 0 with a library that does not say.
static void ai_note_cached(alphaserve::BatchResult & res, const ai_engine * eng, int s) {
    res.slot_cached.resize(res.sampled.size() - 1, -1);
    res.slot_cached.push_back(ai_engine_prompt_cached ? (int) ai_engine_prompt_cached(eng, s) : 0);
}

namespace alphaserve {

// See ai_engine_live() in the header: /health needs to see capture state from outside the
// process, and threading the engine through every layer to get there is not worth it.
static ai_gpu_engine * g_ai_live = nullptr;
ai_gpu_engine * ai_engine_live() { return g_ai_live; }

ai_gpu_engine::~ai_gpu_engine() {
    if (g_ai_live == this) g_ai_live = nullptr;
    if (eng_) { ai_engine_destroy(eng_); eng_ = nullptr; }
}

bool ai_gpu_engine::init(const ai_engine_params & params,
                         const llama_vocab * vocab,
                         spsc_queue<BatchRequest, 64> & in_queue,
                         spsc_queue<BatchResult, 64> & out_queue,
                         int max_batch,
                         int64_t min_free_vram_mib) {
    vocab_ = vocab;
    in_    = &in_queue;
    out_   = &out_queue;
    max_batch_ = max_batch;
    min_free_vram_mib_ = min_free_vram_mib;
    last_error_.clear();

    // Single instance only, asserted rather than assumed. AlphaInfer's kern::q8_stage_clear()
    // is a per-round global: correct while exactly one engine exists, silently wrong once
    // AlphaServe holds two (one engine's round would clear the other's staged state, and the
    // symptom is wrong numbers, not a crash). Refuse the second construction.
    if (g_ai_live != nullptr) {
        last_error_ = "a second AlphaInfer engine was constructed while one is live. "
                      "AlphaInfer's q8 staging is a per-round GLOBAL, so two engines would "
                      "silently corrupt each other's staged state. Refusing.";
        fprintf(stderr, "[ai-engine] REFUSING TO SERVE: %s\n", last_error_.c_str());
        return false;
    }

    const char * err = nullptr;
    eng_ = ai_engine_create(&params, &err);
    if (!eng_) {
        last_error_ = std::string("ai_engine_create failed: ") + (err ? err : "(no message)");
        fprintf(stderr, "[ai-engine] REFUSING TO SERVE: %s\n", last_error_.c_str());
        return false;
    }

    // AI_HOST_EMBD frees VRAM and is bit-identical, but is incompatible with MTP. The engine has
    // its own guard, but it is shadowed by an earlier has_mtp() check on current models and so
    // untested; check it here too. The failure it prevents is a mid-round sync that breaks graph
    // capture, i.e. the silent-degradation case checked below.
    if (params.host_embd && ai_engine_has_mtp(eng_)) {
        last_error_ = "AI_HOST_EMBD is incompatible with MTP (the draft head embeds "
                      "device-produced ids; a host gather needs a mid-round sync that breaks "
                      "capture). Disable one of them.";
        fprintf(stderr, "[ai-engine] REFUSING TO SERVE: %s\n", last_error_.c_str());
        return false;
    }

    // cudaGraphInstantiate runs out of memory at ~50 MiB free. Assert a floor with margin before
    // serving rather than discovering it on a later round, when the symptom is a silent slowdown
    // instead of a boot failure.
    const int64_t free_mib = ai_engine_free_vram_mib(eng_);
    if (free_mib < min_free_vram_mib_) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "free VRAM %lld MiB is below the %lld MiB floor (measured instantiate-OOM "
                 "cliff: 51 MiB). Free headroom or lower --ctx / raise --ws-mib.",
                 (long long) free_mib, (long long) min_free_vram_mib_);
        last_error_ = buf;
        fprintf(stderr, "[ai-engine] REFUSING TO SERVE: %s\n", last_error_.c_str());
        return false;
    }

    // Graph capture is worth ~15% and its failure returns rc == 0. Refuse at init if already
    // degraded; process_one() re-checks every round because capture can fail later.
    const ai_capture_state cs = ai_engine_capture_state(eng_);
    if (cs == AI_CAPTURE_DEGRADED) {
        last_error_ = "AlphaInfer is running DEGRADED (capture failed, direct launches). "
                      "That costs ~15% and does not set a non-zero exit code, so it will not "
                      "show up as a failure anywhere else. Refusing to serve.";
        fprintf(stderr, "[ai-engine] REFUSING TO SERVE: %s\n", last_error_.c_str());
        return false;
    }

    n_streams_ = params.n_streams > 0 ? params.n_streams : 1;
    prompt_.assign(n_streams_, {});
    open_.assign(n_streams_, false);
    midpf_.assign(n_streams_, false);
    // Chunked prefill: the engine takes a prompt a chunk at a time, so the other slot's decode rounds run between chunks
    // instead of freezing for the whole prefill. The engine only allows it with at most one other live stream, so it
    // is used for n_streams <= 2. The in-tree engine is linked statically and its symbols are not exported, so these are
    // taken directly: a dlsym(RTLD_DEFAULT) lookup returned null and left both features off.
    if (n_streams_ <= 2 && !getenv("ALPHA_AI_NO_CHUNKED_PREFILL")) prefill_chunk_fn_ = (void *) &ai_engine_prefill_chunk;
    prefill_mm_fn_ = (void *) &ai_engine_prefill_mm;
    // sampling: newer libraries draw sampled target tokens on the device; Flash-Next and older libraries are greedy only
    if (ai_engine_supports_sampling && ai_engine_set_sampling && ai_engine_supports_sampling(eng_) == 1) {
        set_sampling_fn_ = (void *) &ai_engine_set_sampling;
    }
    fprintf(stderr, "[ai-engine] sampling: %s\n", set_sampling_fn_ ? "ON (temperature / top_k / top_p / seed on the device)"
                                                                    : "greedy only (temperature > 0 is served greedy)");
    if (g_serve_mtmd) {
        fprintf(stderr, "[ai-engine] image input: ON (ai_engine_prefill_mm)\n");
    }
    fprintf(stderr, "[ai-engine] chunked prefill: %s\n", prefill_chunk_fn_ ? "ON (ai_engine_prefill_chunk)" :
            n_streams_ > 2 ? "off (more than 2 streams)" : "off (ALPHA_AI_NO_CHUNKED_PREFILL)");
    mtp_k_ = 0;
    int exact = 0;
    if (params.mtp_k > 0) {
        // the engine's policy may lower k (exact verify caps it at 3); report what actually runs
        if (ai_engine_mtp_info(eng_, &mtp_k_, &exact) != AI_ENGINE_OK) mtp_k_ = 0;
        mtp_exact_ = exact;
    }
    max_round_tokens_ = mtp_k_ > 0 ? ai_engine_max_tokens_per_round(eng_) : n_streams_;
    if (max_round_tokens_ < n_streams_) max_round_tokens_ = n_streams_;

    fprintf(stderr, "[ai-engine] ready: capture=%s free_vram=%lld MiB streams=%d ctx=%lld/stream "
                    "ws=%d MiB host_embd=%d kv=%d mtp_k=%d (%s) round_tokens<=%d\n",
            capture_state_str(), (long long) free_mib, n_streams_,
            (long long) params.ctx_cells_per_stream, params.ws_mib, params.host_embd,
            params.kv_type_is_q4_0, mtp_k_, mtp_k_ > 0 ? (exact ? "exact" : "fast") : "plain",
            max_round_tokens_);
    if (calibration_state() >= 0) {
        fprintf(stderr, "[ai-engine] calibration state %d (0 settled, 1 recalibrated, 2 bootstrap), actual ctx %lld cells/stream\n",
                calibration_state(), (long long) ctx_cells());
    }
    // Publish only after every precondition passed, so /health can never advertise an
    // engine that init() refused.
    g_ai_live = this;
    return true;
}

bool ai_gpu_engine::is_degraded() const {
    if (!eng_) return true;
    return ai_engine_capture_state(eng_) != AI_CAPTURE_OWN;
}

const char * ai_gpu_engine::capture_state_str() const {
    if (!eng_) return "none";
    switch (ai_engine_capture_state(eng_)) {
        case AI_CAPTURE_OWN:      return "own";
        case AI_CAPTURE_DEGRADED: return "degraded";
        default:                  return "none";
    }
}

int64_t ai_gpu_engine::free_vram_mib() const {
    return eng_ ? ai_engine_free_vram_mib(eng_) : 0;
}

// Getters added in a later libalphainfer (additive ABI). Looked up at run time so this binary still runs with an older
// libalphainfer.so; -2 / -1 = the loaded library does not have them.
//   ai_engine_calibration_state: 0 settled, 1 recalibrated this start, 2 bootstrap (clamped context), -1 NULL engine
//   ai_engine_ctx_cells:         the per-stream context the engine actually holds (a bootstrap start clamps it)
int ai_gpu_engine::calibration_state() const {
    return eng_ ? ai_engine_calibration_state(eng_) : -2;
}

int64_t ai_gpu_engine::ctx_cells() const {
    return eng_ ? (int64_t) ai_engine_ctx_cells(eng_) : -1;
}

// Reject anything this increment cannot honour. Each of these, if ignored, produces
// output that looks fine: a sampling parameter silently dropped reads as the model
// being boring, not as a bug. decode_error is the honest answer.
bool ai_gpu_engine::check_supported(const BatchRequest & req) {
    for (size_t i = 0; i < req.logits_mask.size(); ++i) {
        if (!req.logits_mask[i]) continue;   // params are only meaningful where logits are taken
        if (set_sampling_fn_) continue;      // the engine samples: set_sampling() hands it every row's settings

        if (i < req.samp_temp.size() && req.samp_temp[i] > 0.0f) {
            last_error_ = "temperature sampling is not supported by the AlphaInfer backend yet "
                          "(greedy only). Refusing rather than silently decoding greedily.";
            return false;
        }
        if (i < req.samp_top_k.size() && req.samp_top_k[i] > 1) {
            last_error_ = "top_k is not supported by the AlphaInfer backend yet (greedy only).";
            return false;
        }
        if (i < req.samp_top_p.size() && req.samp_top_p[i] < 1.0f) {
            last_error_ = "top_p is not supported by the AlphaInfer backend yet (greedy only).";
            return false;
        }
        // The MTP seed schedule, not just this token's seed. It is keyed on the slot's
        // generation counter, not the position. Reproducing it requires MTP support not
        // available here, and getting it wrong yields text that diverges from plain greedy
        // while looking perfectly fluent.
        if (i < req.samp_rng_base.size() && req.samp_rng_base[i] != 0) {
            last_error_ = "MTP seed schedule (samp_rng_base) is not supported by the AlphaInfer "
                          "backend yet.";
            return false;
        }
    }
    // ALPHA_HYBRID_PREFIX: the scheduler believes these ran; ignoring them would leave a slot resuming
    // from a recurrent state that was never restored. Refuse loudly instead.
    if (!req.restore_seqs.empty() || !req.snapshot_seqs.empty()) {
        last_error_ = "hybrid prefix checkpoints (restore/snapshot) are not supported by the AlphaInfer backend.";
        return false;
    }
    // an image prompt's tokens are placeholders: prefilled as text they would give fluent nonsense about no image
    if (req.mm && (!prefill_mm_fn_ || !g_serve_mtmd)) {
        last_error_ = "image input needs a loaded --mmproj projector.";
        return false;
    }
    return true;
}

bool ai_gpu_engine::prefill_mm(const BatchRequest & req, int s, int32_t & first) {
    using prefill_mm_t = int (*)(ai_engine *, int, const int32_t *, int, const int32_t *, const float *, const int32_t *,
                                 int, int32_t *);
    using n_embd_t = int (*)(const ai_engine *);
    mtmd_context * mc = g_serve_mtmd;
    const mtmd_input_chunks * chunks = req.mm->chunks;
    static void * n_embd_fn = (void *) &ai_engine_n_embd;
    const int E = ((n_embd_t) n_embd_fn)(eng_);   // == llama_model_n_embd_inp, which mtmd checked the projector against
    std::vector<int32_t> tok, t, y, x, z, rows;
    std::vector<float>   embd;
    int32_t n_past = 0;
    for (size_t ci = 0; ci < mtmd_input_chunks_size(chunks); ++ci) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, ci);
        const auto type = mtmd_input_chunk_get_type(c);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t nt = 0;
            const llama_token * tk = mtmd_input_chunk_get_tokens_text(c, &nt);
            for (size_t j = 0; j < nt; ++j) {
                tok.push_back(tk[j]);
                t.push_back(n_past); y.push_back(n_past); x.push_back(n_past); z.push_back(0);
                ++n_past;
            }
        } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(c);
            const size_t nt = mtmd_input_chunk_get_n_tokens(c);
            if (mtmd_encode_chunk(mc, c) != 0) {
                last_error_ = "the projector failed to encode an image";
                return false;
            }
            const float * e = mtmd_get_output_embd(mc);
            std::vector<mtmd_decoder_pos> pos(nt);
            if (mtmd_decode_use_mrope(mc)) {
                mtmd_helper_image_get_decoder_pos(it, n_past, pos.data());
            } else {   // no M-RoPE: image rows take consecutive positions like text
                for (size_t j = 0; j < nt; ++j) pos[j] = { (uint32_t) (n_past + j), (uint32_t) (n_past + j), (uint32_t) (n_past + j), 0 };
            }
            for (size_t j = 0; j < nt; ++j) {
                rows.push_back((int32_t) tok.size());
                tok.push_back(MM_PLACEHOLDER_TOKEN);
                t.push_back((int32_t) pos[j].t); y.push_back((int32_t) pos[j].y); x.push_back((int32_t) pos[j].x);
                z.push_back((int32_t) pos[j].z);
            }
            embd.insert(embd.end(), e, e + nt * (size_t) E);
            n_past += mtmd_input_chunk_get_n_pos(c);
        } else {
            last_error_ = "audio input is not supported by the AlphaInfer backend";
            return false;
        }
    }
    if ((int) tok.size() != (int) prompt_[s].size() || n_past != req.mm->n_pos) {
        char buf[200];
        snprintf(buf, sizeof buf, "image prompt does not match its chunks (%zu rows vs %zu, %d positions vs %d)",
                 tok.size(), prompt_[s].size(), n_past, req.mm->n_pos);
        last_error_ = buf;
        return false;
    }
    std::vector<int32_t> pos4;
    pos4.reserve(tok.size() * 4);
    pos4.insert(pos4.end(), t.begin(), t.end());
    pos4.insert(pos4.end(), y.begin(), y.end());
    pos4.insert(pos4.end(), x.begin(), x.end());
    pos4.insert(pos4.end(), z.begin(), z.end());
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = ((prefill_mm_t) prefill_mm_fn_)(eng_, s, tok.data(), (int) tok.size(), pos4.data(), embd.data(),
                                                   rows.data(), (int) rows.size(), &first);
    if (rc != AI_ENGINE_OK) {
        const char * le = ai_engine_last_error(eng_);
        last_error_ = std::string("ai_engine_prefill_mm failed: ") + (le ? le : "");
        return false;
    }
    fprintf(stderr, "[ai-engine] image prompt: %zu rows (%zu image rows, %d images), %d positions, prefill %.0f ms\n",
            tok.size(), rows.size(), req.mm->n_images, n_past,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return true;
}

void ai_gpu_engine::fail_batch(const BatchRequest & req, BatchResult & res) {
    res = BatchResult();
    res.decode_error = true;
    // the scheduler force-finishes exactly the slots listed here; an empty list would leave them GENERATING forever
    for (size_t i = 0; i < req.slot_ids.size(); ++i) {
        const int s = req.slot_ids[i];
        if (std::find(res.slot_ids.begin(), res.slot_ids.end(), s) == res.slot_ids.end()) res.slot_ids.push_back(s);
    }
    for (int s : res.slot_ids) {
        if (s >= 0 && s < n_streams_) {
            if (open_[s] || midpf_[s]) ai_engine_reset_stream(eng_, s);
            open_[s] = false;
            midpf_[s] = false;
            prompt_[s].clear();
        }
    }
    res.n_slots = (int) res.slot_ids.size();
    res.sampled.assign(res.n_slots, LLAMA_TOKEN_NULL);
}

bool ai_gpu_engine::process_one() {
    BatchRequest req;
    if (!in_->try_pop(req)) return false;

    BatchResult res;

    // Capture can fail on any round, not just at init, and the failure path returns rc == 0. Checking only at
    // startup would let a server that booted clean silently lose ~15% later and keep answering 200 OK.
    if (is_degraded()) {
        last_error_ = "AlphaInfer degraded to direct launches mid-run (capture lost). "
                      "~15% slower while still returning success; failing this batch so the "
                      "condition is visible instead of invisible.";
        fprintf(stderr, "[ai-engine] DEGRADED MID-RUN: %s\n", last_error_.c_str());
        fail_batch(req, res);
        out_->try_push(std::move(res));
        return true;
    }
    if (!check_supported(req)) {
        fprintf(stderr, "[ai-engine] unsupported batch: %s\n", last_error_.c_str());
        fail_batch(req, res);
        out_->try_push(std::move(res));
        return true;
    }

    // Slot reassigned: drop whatever the slot held (normally already reset when its last request ended).
    for (int32_t s : req.clear_seqs) {
        if (s < 0 || s >= n_streams_) continue;
        if (open_[s] || midpf_[s]) ai_engine_reset_stream(eng_, s);
        open_[s] = false;
        midpf_[s] = false;
        prompt_[s].clear();
    }

    // Sampling: hand each stream's settings to the engine before this batch's prefill / rounds. The engine keeps them per
    // slot (an unchanged setting costs nothing), so a greedy request after a sampled one resets its slot to greedy.
    if (set_sampling_fn_) {
        using set_t = int (*)(ai_engine *, int, float, int, float, float, uint64_t);
        for (size_t i = 0; i < req.logits_mask.size(); ++i) {
            if (!req.logits_mask[i]) continue;
            const int s = i < req.seq_ids.size() ? req.seq_ids[i] : 0;
            if (s < 0 || s >= n_streams_) continue;
            const float    t    = i < req.samp_temp.size()  ? req.samp_temp[i]  : 0.0f;
            const int      k    = i < req.samp_top_k.size() ? req.samp_top_k[i] : 0;
            const float    p    = i < req.samp_top_p.size() ? req.samp_top_p[i] : 1.0f;
            const float    mp   = i < req.samp_min_p.size() ? req.samp_min_p[i] : 0.0f;
            // the request's seed (rng_base: the client's seed, or a random one), constant over the request -- seed 0 too.
            // samp_rng is per token, and a per-token key would break exact MTP == plain decode.
            const uint64_t seed = i < req.samp_rng_base.size() ? req.samp_rng_base[i] : 0;
            ((set_t) set_sampling_fn_)(eng_, s, t, k, p, mp, seed);
        }
    }

    // Split the batch: a token of an open stream is its pending decode token; anything else is prompt.
    std::map<int, std::pair<int32_t, int32_t>> dec;   // stream -> (token, position); ordered = ascending ids
    std::vector<int> prefill_done;                    // streams whose final prompt chunk is in this batch
    std::map<int, size_t> chunk_from;                 // chunked prefill: stream -> first prompt_ index of this batch's chunk
    const int n = (int) req.tokens.size();
    for (int i = 0; i < n; ++i) {
        const int s = i < (int) req.seq_ids.size() ? req.seq_ids[i] : 0;
        if (s < 0 || s >= n_streams_) {
            last_error_ = "batch names a sequence id outside the engine's streams";
            fprintf(stderr, "[ai-engine] %s (seq %d, streams %d)\n", last_error_.c_str(), s, n_streams_);
            fail_batch(req, res); out_->try_push(std::move(res)); return true;
        }
        const bool want = i < (int) req.logits_mask.size() && req.logits_mask[i];
        if (open_[s]) {
            if (!want || dec.count(s)) {
                last_error_ = "an open stream received more than its one pending token";
                fprintf(stderr, "[ai-engine] %s (seq %d)\n", last_error_.c_str(), s);
                fail_batch(req, res); out_->try_push(std::move(res)); return true;
            }
            dec[s] = { (int32_t) req.tokens[i], (int32_t) req.positions[i] };
            continue;
        }
        if ((int) req.positions[i] != (int) prompt_[s].size()) {
            // prompts must arrive whole and in order from position 0 (prefix reuse is disabled for this backend)
            last_error_ = "prompt chunk out of order or not starting at position 0";
            fprintf(stderr, "[ai-engine] %s (seq %d pos %d, have %zu)\n", last_error_.c_str(), s,
                    (int) req.positions[i], prompt_[s].size());
            fail_batch(req, res); out_->try_push(std::move(res)); return true;
        }
        if (!chunk_from.count(s)) chunk_from[s] = prompt_[s].size();
        prompt_[s].push_back((int32_t) req.tokens[i]);
        if (want) prefill_done.push_back(s);
    }

    // An open stream with no pending token in a decode batch has finished (stop token, max_tokens or cancel): the
    // engine may have run past its end inside the last round, and a round must name every open stream, so reset it.
    // A prompt-only batch says nothing about the open streams: with ALPHA_PREFILL_ALONE (the AlphaInfer default) the
    // scheduler sends a new request's prompt chunks in batches of their own while the other slot keeps generating,
    // and resetting there would kill the generating stream. A stream that really finished is still reset by the next
    // decode round without it, or by clear_seqs.
    for (int s = 0; s < n_streams_ && !dec.empty(); ++s) {
        if (open_[s] && !dec.count(s)) {
            ai_engine_reset_stream(eng_, s);
            open_[s] = false;
        }
    }

    // Image input: an image prompt arrives whole and alone (the scheduler prefills a multimodal prompt by itself, in one
    // batch, logits on its last row): one ai_engine_prefill_mm with the projector's embeddings, nothing else this batch
    if (req.mm && req.mm_slot >= 0 && req.mm_slot < n_streams_ && !prompt_[req.mm_slot].empty()) {
        const int s = req.mm_slot;
        if (!dec.empty() || std::find(prefill_done.begin(), prefill_done.end(), s) == prefill_done.end()) {
            last_error_ = "an image prompt must arrive whole and alone in its batch";
            fprintf(stderr, "[ai-engine] %s (seq %d)\n", last_error_.c_str(), s);
            prompt_[s].clear();
            fail_batch(req, res); out_->try_push(std::move(res)); return true;
        }
        int32_t first = 0;
        const bool ok = prefill_mm(req, s, first);
        prompt_[s].clear();
        if (!ok) {
            fprintf(stderr, "[ai-engine] %s\n", last_error_.c_str());
            ai_engine_reset_stream(eng_, s);
            fail_batch(req, res); out_->try_push(std::move(res)); return true;
        }
        open_[s]  = true;
        midpf_[s] = false;
        res.slot_ids.push_back(s);
        res.sampled.push_back((llama_token) first);
        res.slot_extra.emplace_back();
        res.slot_cached.assign(res.sampled.size(), -1); res.slot_cached.back() = 0;   // image prompt: prefilled whole
        n_tokens_ += 1;
        res.n_slots = (int) res.sampled.size();
        out_->try_push(std::move(res));
        return true;
    }

    // ALPHA_AI_CHUNK_LOG=1: one line per round showing which streams decode and which prompt pieces arrive (ms since start)
    static const bool chunk_log = getenv("ALPHA_AI_CHUNK_LOG") != nullptr;
    if (chunk_log) {
        static const auto t_start = std::chrono::steady_clock::now();
        std::string ln;
        for (auto & kv : dec) ln += " D" + std::to_string(kv.first);
        for (auto & kv : chunk_from) ln += " P" + std::to_string(kv.first) + ":" + std::to_string(prompt_[kv.first].size() - kv.second) +
                                       (std::find(prefill_done.begin(), prefill_done.end(), kv.first) != prefill_done.end() ? "(last)" : "");
        fprintf(stderr, "[ai-chunk] %8.1f ms |%s\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count(), ln.c_str());
    }

    // 1) one decode round over the open streams (before any prefill, which would open another stream)
    if (!dec.empty()) {
        std::vector<int32_t> toks, pos, seqs;
        for (auto & kv : dec) { seqs.push_back(kv.first); toks.push_back(kv.second.first); pos.push_back(kv.second.second); }
        std::vector<int32_t> out_tok(max_round_tokens_), out_seq(max_round_tokens_);
        int out_n = 0;
        int rc;
        if (mtp_k_ > 0) {
            rc = ai_engine_decode_mtp(eng_, toks.data(), pos.data(), seqs.data(), (int) toks.size(),
                                      out_tok.data(), out_seq.data(), max_round_tokens_, &out_n);
        } else {
            rc = ai_engine_decode(eng_, toks.data(), pos.data(), seqs.data(), (int) toks.size(),
                                  nullptr, out_tok.data(), &out_n);
            for (int k = 0; k < out_n && k < (int) seqs.size(); ++k) out_seq[k] = seqs[k];   // plain: one per stream, in order
        }
        if (rc != AI_ENGINE_OK) {
            const char * le = ai_engine_last_error(eng_);
            char buf[256];
            snprintf(buf, sizeof(buf), "ai_engine_decode%s failed rc=%d: %s", mtp_k_ > 0 ? "_mtp" : "", rc, le ? le : "");
            last_error_ = buf;
            fprintf(stderr, "[ai-engine] %s\n", last_error_.c_str());
            fail_batch(req, res); out_->try_push(std::move(res)); return true;
        }
        // emission order -> per stream: the first token is the result, the rest ride in slot_extra
        std::map<int, std::vector<llama_token>> per;
        for (int k = 0; k < out_n; ++k) per[out_seq[k]].push_back((llama_token) out_tok[k]);
        for (auto & kv : per) {
            if (kv.second.empty()) continue;
            res.slot_ids.push_back(kv.first);
            res.sampled.push_back(kv.second[0]);
            res.slot_extra.emplace_back(kv.second.begin() + 1, kv.second.end());
            n_tokens_ += (int64_t) kv.second.size();
        }
        n_decodes_ += 1;
    }

    // 2a) chunked prefill: every stream with prompt tokens in this batch hands its chunk to the engine now; only the
    //     last chunk returns a token. The decode round above already ran for the other live stream.
    if (prefill_chunk_fn_) {
        using chunk_fn_t = int (*)(ai_engine *, int, const int32_t *, int, int, int32_t *);
        const chunk_fn_t chunk = (chunk_fn_t) prefill_chunk_fn_;
        for (auto & kv : chunk_from) {
            const int s = kv.first;
            const bool last = std::find(prefill_done.begin(), prefill_done.end(), s) != prefill_done.end();
            int32_t first = -1;
            const int n_new = (int) (prompt_[s].size() - kv.second);
            const int rc = chunk(eng_, s, prompt_[s].data() + kv.second, n_new, last ? 1 : 0, &first);
            if (rc != AI_ENGINE_OK) {
                const char * le = ai_engine_last_error(eng_);
                char buf[256];
                snprintf(buf, sizeof(buf), "ai_engine_prefill_chunk failed rc=%d: %s", rc, le ? le : "");
                last_error_ = buf;
                fprintf(stderr, "[ai-engine] %s\n", last_error_.c_str());
                ai_engine_reset_stream(eng_, s);
                midpf_[s] = false; prompt_[s].clear();
                fail_batch(req, res); out_->try_push(std::move(res)); return true;
            }
            if (!last) { midpf_[s] = true; continue; }
            midpf_[s] = false;
            prompt_[s].clear();
            open_[s] = true;
            res.slot_ids.push_back(s);
            res.sampled.push_back((llama_token) first);
            res.slot_extra.emplace_back();
            ai_note_cached(res, eng_, s);
            n_tokens_ += 1;
        }
        prefill_done.clear();
    }

    // 2) prefills whose last chunk arrived: the whole buffered prompt in one call, first token back
    for (int s : prefill_done) {
        int32_t first = 0;
        const int rc = ai_engine_prefill(eng_, s, prompt_[s].data(), (int) prompt_[s].size(), &first);
        prompt_[s].clear();
        if (rc != AI_ENGINE_OK) {
            const char * le = ai_engine_last_error(eng_);
            char buf[256];
            snprintf(buf, sizeof(buf), "ai_engine_prefill failed rc=%d: %s", rc, le ? le : "");
            last_error_ = buf;
            fprintf(stderr, "[ai-engine] %s\n", last_error_.c_str());
            ai_engine_reset_stream(eng_, s);
            fail_batch(req, res); out_->try_push(std::move(res)); return true;
        }
        open_[s] = true;
        res.slot_ids.push_back(s);
        res.sampled.push_back((llama_token) first);
        res.slot_extra.emplace_back();
        ai_note_cached(res, eng_, s);
        n_tokens_ += 1;
    }

    res.n_slots = (int) res.sampled.size();
    out_->try_push(std::move(res));
    return true;
}

} // namespace alphaserve

#endif // ALPHA_HAS_ALPHAINFER
