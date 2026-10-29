// SPDX-License-Identifier: Apache-2.0
#include "serve-gpu-engine.h"
#include "serve-mm.h"
#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <unordered_set>
#include <unordered_map>

GGML_BACKEND_API_C void ggml_cuda_set_expert_user(int user_id);
#include "alpha_mole_speculative.h"   // alpha_recurrent_snapshot/restore
#include "ggml-backend.h"

// Multi-token-prediction self-speculative decode (ALPHA_MTP). Each single-token decode drafts the
// next token from the model's MTP (nextn) head, verifies it with a 2-token batched decode, and emits
// 1-2 extra tokens per step. Exact-match accept, so output is identical to plain greedy. Owns the
// window path (mtp_window_extend).
static const bool g_mtp = []() {
    const char * e = getenv("ALPHA_MTP");
    return e && atoi(e) != 0;
}();

// ReplaySSM (ALPHA_REPLAYSSM): replace the reject relay with a record-and-fold of the recurrent
// state. The relay is a full decode that produces no tokens. With this on, the verify records the
// gated-delta op's inputs, the state publish is suppressed, and after acceptance the same op is
// re-invoked over just the accepted prefix: no rewind, no re-decode, and the attention KV tail is
// dropped with a plain seq_rm.
//
// An armed verify leaves the committed recurrent state at its pre-window value, so every armed
// window must end in either a successful fold or the relay. A failed fold forces the relay even on
// a full accept.
static const bool g_replayssm = []() {
    const char * e = getenv("ALPHA_REPLAYSSM");
    return e && atoi(e) != 0;
}();

// Draft depth (m). m=1 drafts one token and verifies 2; m=N drafts N by chaining the head and
// verifies N+1 in one batched decode, accepting the longest matching prefix. Deeper drafting
// amortises the fixed per-window cost (snapshot + verify decode) over more tokens, but acceptance
// decays geometrically (expected tokens/window = sum_{i<=N} p^i). Bounded [1,8].

// ALPHA_MTP_MARGIN: adaptive-depth gate. tau is a logit gap, not a probability; calibrate per
// model. 0 disables and restores fixed m.
// The n-gram flags are also accepted under the ALPHA_NGRAM_* names.
static const char * ngram_env(const char * ours, const char * theirs) {
    const char * e = getenv(ours);
    return e ? e : getenv(theirs);
}
// Pad a short lookup chain out to draft_n with the last token. Default off: with a margin-gated
// head drafting ~2.5 tokens, padding makes the routed verify wider than the head's.
static const int g_mtp_ngram_pad = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_PAD", "ALPHA_NGRAM_PAD");
    return e ? atoi(e) : 0;
}();
// Minimum chain length (not match length) to route. The most recent match often continues only
// 1-2 tokens, and verifying a 1-token chain costs a full window. 0 = off (route on match length).
static const int g_mtp_ngram_minlen = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_MINLEN", "ALPHA_NGRAM_MIN_CHAIN");
    return e ? atoi(e) : 0;
}();
// Cap the routed chain width. The lookup almost always returns a full draft_n chain, while far
// fewer tokens are accepted, so the extra verify columns are wasted. 0 = no cap.
static const int g_mtp_ngram_maxlen = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_MAXLEN", "ALPHA_NGRAM_MAX_CHAIN");
    // Width 16-24 helps a lot on record-structured content (e.g. JSON) and is neutral elsewhere;
    // the useful width tracks the repeating unit of the content.
    return e ? atoi(e) : 24;
}();
// ALPHA_MTP_NGRAM_ADAPT: no fixed routing method. The gate estimates yield per ms for each depth from
// the free lookup on every window, including declined ones, and routes only when the best depth
// beats the head.
static const int g_mtp_ngram_adapt = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_ADAPT", "ALPHA_NGRAM_ADAPT");
    return e ? atoi(e) : 0;                 // 0 = off (default)
}();
// Minimum observations at a depth before it may be chosen.
static const int g_mtp_ngram_minobs = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_MINOBS", "ALPHA_NGRAM_MIN_OBS");
    const int v = e ? atoi(e) : 5;
    return v < 1 ? 1 : v;
}();
// Forgetting factor for the per-depth yield estimate. The estimated quantity is non-stationary
// (scored prefix falls within one generation even with no routing), and a cumulative mean lags that
// and over-routes late. 1.0 = no forgetting (cumulative mean); lower weights recent windows more.
static const float g_mtp_ngram_decay = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_DECAY", "ALPHA_NGRAM_DECAY");
    const float v = e ? (float) atof(e) : 1.0f;
    return v <= 0.0f ? 1.0f : (v > 1.0f ? 1.0f : v);
}();
// Fixed-stride scoring. Window boundaries are anti-correlated with the lookup's yield: a routed window
// advances by accepted+1, landing one token past the end of the match it consumed, where the
// lookup's continuation has run out. Sampling there biases the measured yield down. With a stride,
// the estimator's chains are drawn at fixed token positions independent of routing. The lookup is
// free. 0 = off (sample at window boundaries).
static const int g_mtp_ngram_stride = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_STRIDE", "ALPHA_NGRAM_STRIDE");
    const int v = e ? atoi(e) : 0;
    return v < 0 ? 0 : v;
}();
// Experimental and known broken, do not enable: it desyncs the n-gram ring on an anchor-only (k=0)
// window (the guard reports "RING DESYNC"), so routing decisions are made against the wrong stream.
// Output stays correct because the exact verify rejects bad drafts. It is also not expected to pay:
// the window still pays the ReplaySSM snapshot, so an anchor-only window is slower than a plain decode.
//
// Pure n-gram: on a lookup miss, do not draft at all. The window collapses to an anchor-only verify
// (k=0, M=1), i.e. a plain decode. Intended for offload, where a head draft step is a full forward
// pass through CPU-resident weights.
static const int g_mtp_ngram_only = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_ONLY", "ALPHA_NGRAM_ONLY");
    return e ? atoi(e) : 0;
}();
static const int g_mtp_ngram = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM", "ALPHA_NGRAM_DRAFT");
    return e ? atoi(e) : 0;                 // 0 = off (default)
}();
static const int g_mtp_ngram_min = []() {
    const char * e = ngram_env("ALPHA_MTP_NGRAM_MIN", "ALPHA_NGRAM_MIN_MATCH");
    const int v = e ? atoi(e) : 3;
    // Do not clamp the upper bound. The lookup only attempts n<=3, so MIN>3 never fires, which makes
    // MIN a kill switch for separating the router's presence from its routing.
    return v < 1 ? 1 : v;
}();
static const int g_mtp_ngram_win = []() {
    const char * e = getenv("ALPHA_MTP_NGRAM_WIN");
    const int v = e ? atoi(e) : 2048;
    return v < 64 ? 64 : v;
}();

// ALPHA_MTP_TRIGGERLESS: a call that ended cleanly leaves its last emitted token undecoded with the
// head state at the position before it, which is exactly the state a window starts from. The next
// step for that slot can run as a window directly instead of a plain 1-token trigger decode followed
// by a verify-graph rebuild.
struct mtp_tl_state { bool ok = false; int slot = -1; int32_t seq = -1; llama_token tok = LLAMA_TOKEN_NULL; llama_pos pos = -1; };
static mtp_tl_state g_mtp_tl;
static bool         g_mtp_tl_active = false;
static thread_local std::vector<float> g_conf_win;   // ALPHA_MTP_CONF_LOG: this window's draft probabilities
static const float g_mtp_conf_tau = []() {
    const char * e = getenv("ALPHA_MTP_CONF_TAU");
    return e ? (float) atof(e) : 0.0f;
}();
static const float g_mtp_margin_tau = []() {
    const char * e = getenv("ALPHA_MTP_MARGIN_TAU");
    return e ? (float) atof(e) : 0.0f;
}();
// Every ALPHA_* boolean in this file uses one convention: present and non-zero = on; absent, empty
// or "0" = off.
static inline bool alpha_env_on(const char * name) {
    const char * e = getenv(name);
    return e && e[0] && atoi(e) != 0;
}

static const bool g_mtp_margin_dbg = alpha_env_on("ALPHA_MTP_MARGIN_DBG");

static const int g_mtp_draft_n = []() {
    const char * e = getenv("ALPHA_MTP_DRAFT_N");
    int v = e ? atoi(e) : 1;
    // Clamp is 32 so the cheap region of the verify-width curve (past ~width 10) can be measured.
    // Default stays 1. Shorter windows (m~4) tend to win because each verify pass also yields
    // full-quality hidden states for the next draft.
    return v < 1 ? 1 : (v > 32 ? 32 : v);
}();

// ALPHA_LOGIT_DUMP=<file>: plain decode appends, per generating token, {i32 token | fp16 logits[n_vocab]},
// with a {-1, n_vocab} marker at each request's prefill.
static const char * g_logit_dump_path = getenv("ALPHA_LOGIT_DUMP");

// Recurrent-state checkpoints for prefix reuse on hybrid models (ALPHA_HYBRID_PREFIX).
// Keyed by seq_id (== slot id). One checkpoint per slot: the scheduler only restores the one it
// asked for most recently, and a new snapshot directive for a slot is issued only after any restore
// for it has been consumed (the restore rides with the slot's first suffix chunk, the snapshot with
// a later chunk boundary), so overwriting in place is safe. Enabled by serve-main; the engine just
// executes the scheduler's directives.
struct hyb_ckpt {
    std::vector<uint8_t> buf;
    llama_pos pos = -1;          // last position folded into the state
    std::vector<float> mtp_h;    // MTP head's carried row h_{pos}, to rebuild head-KV column pos+1
    bool has_mtp = false;
};
static std::unordered_map<int32_t, hyb_ckpt> g_hyb_snap;
static int64_t g_hyb_restores = 0;
static int64_t g_hyb_snaps    = 0;
static int64_t g_hyb_mtp_rows = 0;   // restores that also re-staged the MTP head's boundary row

namespace alphaserve {

extern int g_serve_prompt_chunk;     // serve-slot.h: prefill chunk (--ubatch)
extern int g_serve_ubatch_prefill;   // serve-slot.h: runtime-grow target (0 = off)
GGML_BACKEND_API_C size_t ggml_cuda_dre_borrow(size_t want);   // ggml-cuda/dre.cu
GGML_BACKEND_API_C void   ggml_cuda_dre_return(void);
GGML_BACKEND_API_C void   ggml_cuda_dre_return_begin(size_t keep_extra);
GGML_BACKEND_API_C void   ggml_cuda_dre_return_finish(void);
GGML_BACKEND_API_C void   ggml_cuda_dre_set_prefill_tokens(int n);
GGML_BACKEND_API_C void   ggml_cuda_dre_pipe_prestart(void);

// External draft model for MTP (ALPHA_DRAFT_MODEL).
// The MTP head predicts every depth from the same carried hidden h_t (it is trained for one step
// from (h_t, x_{t+1})), so its acceptance decays with depth by construction. A small model chained
// on its own outputs does not have that ceiling, and its draft step is cheap next to a verify.
static const char * g_draft_model_path = getenv("ALPHA_DRAFT_MODEL");
static const int    g_draft_ngl        = []() { const char * e = getenv("ALPHA_DRAFT_NGL"); return e ? atoi(e) : 99; }();
static const int    g_draft_ctx        = []() { const char * e = getenv("ALPHA_DRAFT_CTX"); return e ? atoi(e) : 2048; }();

bool gpu_engine::draft_init() {
    if (!g_draft_model_path || !*g_draft_model_path) return false;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_draft_ngl;
    dmodel_ = llama_model_load_from_file(g_draft_model_path, mp);
    if (!dmodel_) {
        fprintf(stderr, "[draft] FAILED to load '%s' — falling back to the MTP head\n", g_draft_model_path);
        return false;
    }

    // The vocab must match the target exactly or drafted ids mean different tokens.
    const int nv_t = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_)));
    const int nv_d = llama_vocab_n_tokens(llama_model_get_vocab(dmodel_));
    if (nv_t != nv_d) {
        fprintf(stderr, "[draft] VOCAB MISMATCH target=%d draft=%d — refusing to draft with it\n", nv_t, nv_d);
        llama_model_free(dmodel_); dmodel_ = nullptr;
        return false;
    }

    llama_context_params cp = llama_context_default_params();
    // n_ctx must be set explicitly: it otherwise defaults to the model's training length and
    // allocates ~3 GiB for a model whose weights are 0.5 GiB.
    cp.n_ctx     = g_draft_ctx;
    cp.n_batch   = 64;
    cp.n_ubatch  = 64;
    cp.n_seq_max = 1;
    dctx_ = llama_init_from_model(dmodel_, cp);
    if (!dctx_) {
        fprintf(stderr, "[draft] context init FAILED (n_ctx=%d) — falling back to the MTP head\n", g_draft_ctx);
        llama_model_free(dmodel_); dmodel_ = nullptr;
        return false;
    }

    // Backend greedy sampler on the draft context: without it every draft step copies a full vocab
    // row to the host and scans it there, which would dominate the cost of a cheap draft step.
    {
        auto p = llama_sampler_chain_default_params();
        llama_sampler * ch = llama_sampler_chain_init(p);
        if (ch) {
            llama_sampler_chain_add(ch, llama_sampler_init_greedy());
            if (!llama_set_sampler(dctx_, 0, ch)) {
                fprintf(stderr, "[draft] llama_set_sampler failed; draft steps will use the host scan\n");
            }
        }
    }

    dbatch_ = llama_batch_init(64, 0, 1);
    dhist_.clear();
    dpos_ = 0;
    dseq_ = -1;
    fprintf(stderr, "[draft] external drafter ready: %s (ngl=%d n_ctx=%d vocab=%d)\n",
            g_draft_model_path, g_draft_ngl, g_draft_ctx, nv_d);
    return true;
}

void gpu_engine::draft_free() {
    if (dbatch_.token) { llama_batch_free(dbatch_); dbatch_ = {}; }
    if (dctx_)   { llama_free(dctx_);        dctx_ = nullptr; }
    if (dmodel_) { llama_model_free(dmodel_); dmodel_ = nullptr; }
}

std::vector<llama_token> gpu_engine::draft_chain(llama_token last, llama_pos pos0, int m) {
    std::vector<llama_token> d;
    if (!dctx_ || draft_bad_ || m <= 0) return d;

    const llama_vocab * dvocab = llama_model_get_vocab(dmodel_);

    // dhist_ mirrors the target's committed stream, indexed by absolute position. `last` sits at
    // pos0 and is the token the target just sampled, so it is not in the mirror yet.
    if ((llama_pos) dhist_.size() < pos0) return d;   // gap in the mirror -> head path this window
    dhist_.resize(pos0);
    dhist_.push_back(last);

    // Catch the drafter up to and including pos0, batched. On the first window of a request this
    // replays the prompt; after that it is usually zero or a couple of tokens.
    while (dpos_ <= pos0) {
        common_batch_clear(dbatch_);
        int n = 0;
        while (dpos_ + n <= pos0 && n < 64) {
            const bool want_logits = (dpos_ + n == pos0);
            common_batch_add(dbatch_, dhist_[dpos_ + n], dpos_ + n, { 0 }, want_logits);
            n++;
        }
        if (n == 0) break;
        if (llama_decode(dctx_, dbatch_) != 0) {
            fprintf(stderr, "[draft] catch-up decode failed at pos %d — disabling drafter\n", (int) dpos_);
            draft_bad_ = true;
            return {};
        }
        dpos_ += n;
    }

    // State through pos0. The drafter may be hybrid too, and seq_rm alone does not rewind recurrent
    // state, so snapshot here and restore in draft_commit.
    if (!alpha_recurrent_snapshot(dctx_, dsnap_)) { draft_bad_ = true; return {}; }

    llama_token cur = llama_get_sampled_token_ith(dctx_, dbatch_.n_tokens - 1);

    // ALPHA_DRAFT_DBG: dump the catch-up bookkeeping and the tail of the mirror at the start of a
    // window, to check the context the drafter is handed.
    if (getenv("ALPHA_DRAFT_DBG")) {
        static int n_dbg = 0;
        if (n_dbg < 12) {
            n_dbg++;
            fprintf(stderr, "[draft-dbg] pos0=%d dpos_=%d hist=%zu caught_up=%d | tail:",
                    (int) pos0, (int) dpos_, dhist_.size(), (int) (dpos_ == pos0 + 1));
            for (size_t q = (dhist_.size() > 4 ? dhist_.size() - 4 : 0); q < dhist_.size(); q++)
                fprintf(stderr, " [%zu]=%d", q, (int) dhist_[q]);
            fprintf(stderr, " | last=%d d0=%d\n", (int) last, (int) cur);
        }
    }

    for (int j = 0; j < m; j++) {
        if (cur == LLAMA_TOKEN_NULL || cur < 0 || llama_vocab_is_eog(dvocab, cur)) break;
        d.push_back(cur);
        if ((int) d.size() >= m) break;
        common_batch_clear(dbatch_);
        common_batch_add(dbatch_, cur, pos0 + 1 + j, { 0 }, true);
        if (llama_decode(dctx_, dbatch_) != 0) break;   // short draft is not an error
        cur = llama_get_sampled_token_ith(dctx_, 0);
    }
    return d;
}

void gpu_engine::draft_commit(const std::vector<llama_token> & committed, llama_pos pos0) {
    if (!dctx_ || draft_bad_ || committed.empty()) return;
    // The drafter ran ahead by m; only `committed` was taken. Rewind to the snapshot (state through
    // pos0) and replay in one batch rather than one decode per token.
    alpha_recurrent_restore(dctx_, dsnap_, pos0 + 1);
    llama_memory_seq_rm(llama_get_memory(dctx_), 0, pos0 + 1, -1);
    dpos_ = pos0 + 1;

    // Invariant: the drafter must end a window having consumed through pos0 + n_acc, one token
    // short of the next window's anchor. `committed` is [accepted drafts..., t_next], and t_next
    // becomes the next window's `last` at pos0_next = pos0 + n_acc + 1. Replaying t_next here would
    // leave dpos_ = pos0_next + 1, so the next draft_chain's catch-up loop would never fire and d[0]
    // would be read from a stale batch instead of a fresh decode of `last`. So decode only the
    // accepted prefix and leave `last` for the next window's catch-up.
    const size_t n_replay = committed.size() - 1;          // drop t_next
    if (n_replay > 0) {
        common_batch_clear(dbatch_);
        for (size_t j = 0; j < n_replay && j < 64; j++) {
            common_batch_add(dbatch_, committed[j], pos0 + 1 + (llama_pos) j, { 0 },
                             j + 1 == n_replay);
        }
        if (llama_decode(dctx_, dbatch_) != 0) {
            fprintf(stderr, "[draft] commit replay failed — disabling drafter\n");
            draft_bad_ = true;
            return;
        }
        dpos_ += (llama_pos) dbatch_.n_tokens;
    }
    // t_next still goes into the mirror; the next catch-up decodes it from there.
    dhist_.resize(pos0 + 1);
    for (llama_token t : committed) dhist_.push_back(t);
}

void gpu_engine::init(llama_context * ctx,
                      spsc_queue<BatchRequest, 64> & in_queue,
                      spsc_queue<BatchResult, 64> & out_queue) {
    ctx_ = ctx;
    in_ = &in_queue;
    out_ = &out_queue;

    // Allocate reusable batch
    int n_seq_max = llama_n_seq_max(ctx);
    // the batch must hold the largest prefill chunk the scheduler may send (a grown micro-batch can exceed 2048)
    max_batch_size_ = std::max<int>(max_batch_size_, (int) llama_n_batch(ctx));
    batch_ = llama_batch_init(max_batch_size_, 0, n_seq_max);

    // Optional external drafter (ALPHA_DRAFT_MODEL). Never aborts serve on failure: a bad path or a
    // vocab mismatch logs once and leaves the MTP head path unchanged.
    if (!draft_init()) {
        dctx_ = nullptr;
    }
}

void gpu_engine::start() {
    if (running_.load()) return;
    running_.store(true);
    thread_ = std::thread([this]() { loop(); });
}

void gpu_engine::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
    if (batch_.token) {
        llama_batch_free(batch_);
        batch_ = {};
    }
    draft_free();
}

// Host-side token sampler (temperature / top-k / top-p / seeded categorical).
// Applied per slot in decode_batch when the request asked for sampling; greedy slots keep the
// argmax path. Order: temperature -> top-k -> top-p (nucleus) -> softmax over the survivors ->
// categorical draw. Reproducible for a fixed RNG seed.
static llama_token alpha_sample_token(const float * logits, int n_vocab,
                                      float temperature, int top_k, float top_p, uint64_t rng_seed) {
    if (n_vocab <= 0) return 0;
    if (temperature <= 0.0f || top_k == 1) {           // degenerate -> argmax
        int best = 0; float bl = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > bl) { bl = logits[v]; best = v; }
        return best;
    }
    // candidate list (id, logit); temperature applied to logits
    static thread_local std::vector<std::pair<float,int>> cand;
    cand.clear(); cand.reserve(n_vocab);
    const float inv_t = 1.0f / temperature;
    for (int v = 0; v < n_vocab; v++) cand.emplace_back(logits[v] * inv_t, v);
    // top-k: partial-select the k largest by scaled logit
    int k = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    if (k < (int)cand.size()) {
        std::nth_element(cand.begin(), cand.begin() + k, cand.end(),
                         [](const auto & a, const auto & b){ return a.first > b.first; });
        cand.resize(k);
    }
    // sort desc (needed for nucleus + stable draw)
    std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b){ return a.first > b.first; });
    // softmax over survivors (numerically stable)
    const float maxl = cand.front().first;
    double sum = 0.0;
    static thread_local std::vector<double> probs;
    probs.clear(); probs.reserve(cand.size());
    for (auto & c : cand) { double p = std::exp((double)(c.first - maxl)); probs.push_back(p); sum += p; }
    // top-p (nucleus): keep the smallest prefix whose cumulative prob >= top_p
    if (top_p < 1.0f && top_p > 0.0f) {
        double cum = 0.0; size_t keep = probs.size();
        for (size_t i = 0; i < probs.size(); i++) { cum += probs[i] / sum; if (cum >= (double)top_p) { keep = i + 1; break; } }
        if (keep < probs.size()) { probs.resize(keep); cand.resize(keep); sum = 0.0; for (double p : probs) sum += p; }
    }
    // categorical draw with a seeded RNG
    std::mt19937_64 gen(rng_seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(gen) * sum, acc = 0.0;
    for (size_t i = 0; i < probs.size(); i++) { acc += probs[i]; if (r <= acc) return cand[i].second; }
    return cand.back().second;
}

// Core decode logic, used by both threaded and synchronous paths.
// Install/remove a backend greedy sampler per sequence so a greedy slot's token comes back as
// 4 bytes instead of a full vocab row. llama_set_sampler is only called when the state changes,
// since it sets sched_need_reserve and re-installing every decode would force a reserve per token.
// A slot that asked for temperature keeps raw logits, because alpha_sample_token needs the
// distribution.
void gpu_engine::backend_sampler_sync(const BatchRequest & req) {
    // Default on when MTP is off: the device argmax returns 4 bytes instead of a full logits row
    // D2H plus a host scan per token, with identical output.
    //
    // Not default-on under ALPHA_MTP: the backend sampler yields one token per sequence while a
    // verify has k+1 rows on one sequence (see argmax_ith below), so the accept loop would stop at
    // n_acc = 0 and the window would degrade to plain greedy. ALPHA_BACKEND_ARGMAX=1 still forces it.
    //
    // Not default-on when the prefill micro-batch can exceed 512. Installing a sampler forces a
    // re-reserve, and the prompt-processing reserve sizes outputs = n_tokens, so the argmax graph
    // needs a second n_ubatch x n_vocab f32 tensor next to the logits (1.45 GB at ubatch 1536),
    // which the grow ladder's budget does not include and which can OOM the first request. The
    // sampler can be installed at ubatch 512 before a later long prompt grows the batch, so the gate
    // uses the largest possible micro-batch, not the current one.
    static const bool on = []() {
        const char * e = getenv("ALPHA_BACKEND_ARGMAX");
        if (e) return atoi(e) != 0;
        const char * m = getenv("ALPHA_MTP");
        if (m && atoi(m) != 0) return false;
        const int max_ub = std::max(g_serve_prompt_chunk, g_serve_ubatch_prefill);
        if (max_ub > 512) {
            fprintf(stderr, "[serve] backend argmax OFF by default: prefill micro-batch can reach %d, and its "
                            "reserve would add a %d x n_vocab f32 tensor (ALPHA_BACKEND_ARGMAX=1 forces it)\n",
                    max_ub, max_ub);
            return false;
        }
        return true;
    }();
    if (!on || !ctx_) return;
    for (int i = 0; i < req.n_tokens; i++) {
        if (!req.logits_mask[i]) continue;
        const int32_t seq = req.seq_ids[i];
        if (seq < 0) continue;
        if ((size_t) seq >= samp_backend_.size()) {
            samp_backend_.resize(seq + 1, -1);
            samp_chain_.resize(seq + 1, nullptr);
        }
        const bool sampling = (i < (int) req.samp_temp.size()) && req.samp_temp[i] > 0.0f;
        const int8_t want = sampling ? 0 : 1;
        if (samp_backend_[seq] == want) continue;
        if (want) {
            if (!samp_chain_[seq]) {
                auto p = llama_sampler_chain_default_params();
                p.no_perf = true;
                samp_chain_[seq] = llama_sampler_chain_init(p);
                if (samp_chain_[seq]) llama_sampler_chain_add(samp_chain_[seq], llama_sampler_init_greedy());
            }
            if (!samp_chain_[seq]) continue;
        }
        if (llama_set_sampler(ctx_, seq, want ? samp_chain_[seq] : nullptr)) {
            samp_backend_[seq] = want;
        } else {
            samp_backend_[seq] = 0;   // could not offload -- stay on the host path
        }
    }
}

// Counts full model forwards per MTP window; the phase timers only bracket
// draft/snapshot/verify/relay.
static int64_t g_alpha_decodes = 0;
static inline int alpha_counted_decode(llama_context * c, llama_batch & b) {
    g_alpha_decodes++;
    return llama_decode(c, b);
}

// ALPHA_DRE: a prompt borrowed part of the expert pool; dre_return_pool() gives it back (scheduler back at
// the base micro-batch, pool chunks re-allocated, borrowed experts restored with ALPHA_DRE_RESTORE).
static bool g_dre_lent = false;
static void dre_return_pool(llama_context * ctx, const char * when) {
    if (!g_dre_lent) return;
    const int64_t t0 = ggml_time_us();
    alpha_set_kv_reserve_cap(0);   // decode reserves against the whole cache again
    // the scheduler reset right below reserves for the base micro-batch, so no reserve here
    llama_set_n_ubatch_lazy(ctx, (uint32_t) g_serve_prompt_chunk);
    const int64_t t1 = ggml_time_us();
    // ALPHA_DRE_RETURN_OVERLAP (default 1): the restore copies run on the DMA engine while the host
    // rebuilds the scheduler: free the prompt's compute buffers, start the return (leaving room for
    // the base micro-batch's buffer), reserve, then wait. 0 = reserve, then return, sequentially.
    static const bool overlap = [] { const char * e = getenv("ALPHA_DRE_RETURN_OVERLAP"); return !e || atoi(e) != 0; }();
    const size_t base_bytes = llama_context_reserved_bytes(ctx, (uint32_t) g_serve_prompt_chunk);
    int64_t t2;
    if (overlap && base_bytes > 0) {
        llama_context_release_sched(ctx);
        const int64_t tr = ggml_time_us();
        ggml_cuda_dre_return_begin(base_bytes);
        const int64_t tb = ggml_time_us();
        llama_context_reset_sched(ctx);
        if (getenv("ALPHA_DRE_RETURN_TIMERS")) fprintf(stderr, "[dre-rt] release sched %.1f ms, return begin %.1f ms, reserve %.1f ms\n",
                                                       (tr - t1) / 1e3, (tb - tr) / 1e3, (ggml_time_us() - tb) / 1e3);
        t2 = ggml_time_us();
        ggml_cuda_dre_return_finish();
    } else {
        llama_context_reset_sched(ctx);
        t2 = ggml_time_us();
        ggml_cuda_dre_return();
    }
    g_dre_lent = false;
    fprintf(stderr, "[dre] return path (%s): set_n_ubatch %.1f ms, reset_sched %.1f ms, pool return %.1f ms\n", when,
            (t1 - t0) / 1e3, (t2 - t1) / 1e3, (ggml_time_us() - t2) / 1e3);
}

// (inside namespace alphaserve)
mtmd_context * g_serve_mtmd = nullptr;
mm_prompt::~mm_prompt() { if (chunks) mtmd_input_chunks_free(chunks); }

// Multimodal prompt prefill: one slot's whole prompt as mtmd chunks (text runs decoded as tokens, image runs encoded
// by the projector and decoded as embeddings, with the model's own image positions), then the first token like any
// prompt. The batch's tokens are placeholders and are not decoded.
bool gpu_engine::mm_prefill(const BatchRequest & req, BatchResult & result) {
    if (!g_serve_mtmd || !req.mm || !req.mm->chunks || req.mm_slot < 0) return false;
    const int32_t seq = req.mm_slot;
    llama_memory_t mem = llama_get_memory(ctx_);
    for (int32_t s : req.clear_seqs) llama_memory_seq_rm(mem, s, -1, -1);
    llama_memory_seq_rm(mem, seq, -1, -1);   // a multimodal prompt always starts the sequence over
    g_mtp_tl.ok = false;

    // ALPHA_DRE: prompt-sized decodes stream experts through staging buffers, so borrow for them as a short prompt does
    static const bool dre = getenv("ALPHA_DRE") && atoi(getenv("ALPHA_DRE")) != 0;
    const int n_ub = (int) llama_n_ubatch(ctx_);
    bool lent = false;
    if (dre) {
        if (g_dre_lent) dre_return_pool(ctx_, "multimodal prompt");
        ggml_cuda_dre_set_prefill_tokens(n_ub);
        lent = ggml_cuda_dre_borrow((size_t) 384 << 20) > 1;
    }
    ggml_cuda_set_expert_user(req.mm_slot);

    const int64_t t0 = ggml_time_us();
    llama_pos n_past = 0;
    const int32_t rc = mtmd_helper_eval_chunks(g_serve_mtmd, ctx_, req.mm->chunks, 0, seq, n_ub, /*logits_last=*/true, &n_past);
    const int64_t t1 = ggml_time_us();
    if (lent) { ggml_cuda_dre_return_begin(0); ggml_cuda_dre_return_finish(); }

    result.sampled.clear();
    result.slot_ids.clear();
    result.n_slots = 0;
    if (rc != 0) {
        fprintf(stderr, "[mm] slot %d: multimodal prefill FAILED (rc=%d)\n", req.mm_slot, (int) rc);
        result.decode_error = true;
        return false;
    }
    if (n_past != req.mm->n_pos) {
        fprintf(stderr, "[mm] slot %d: prefill ended at position %d, expected %d\n", req.mm_slot, (int) n_past, req.mm->n_pos);
    }

    // first token: the batch's last entry carries this slot's sampling params
    const int last = req.n_tokens - 1;
    llama_token tok = LLAMA_TOKEN_NULL;
    const float * logits = llama_get_logits_ith(ctx_, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_)));
    if (logits) {
        tok = alpha_sample_token(logits, n_vocab, req.samp_temp[last], req.samp_top_k[last], req.samp_top_p[last], req.samp_rng[last]);
    } else {
        tok = llama_get_sampled_token_ith(ctx_, -1);
    }
    if (tok == LLAMA_TOKEN_NULL) {
        result.decode_error = true;
        return false;
    }
    result.sampled.push_back(tok);
    result.slot_ids.push_back(req.mm_slot);
    result.n_slots = 1;
    fprintf(stderr, "[mm] slot %d: %d images, %d tokens / %d positions prefilled in %.0f ms\n", req.mm_slot,
            req.mm->n_images, req.mm->n_tokens, (int) n_past, (t1 - t0) / 1e3);
    return true;
}

bool gpu_engine::decode_batch(const BatchRequest & req, BatchResult & result) {
    if (req.mm) return mm_prefill(req, result);
    // ALPHA_DB_TIMERS: host phases of a prompt-sized decode_batch (entry -> llama_decode -> after -> return)
    static const bool db_t = getenv("ALPHA_DB_TIMERS") != nullptr;
    static int64_t db_prev_end = 0;
    const int64_t db_t0 = ggml_time_us();
    int64_t db_t1 = 0, db_t2 = 0;
    struct db_guard { const bool on; const int n; const int64_t & t0; const int64_t & t1; const int64_t & t2; ~db_guard() {
        if (!on || n < 256) { db_prev_end = ggml_time_us(); return; }
        const int64_t t3 = ggml_time_us();
        fprintf(stderr, "[db-t] n=%d since_prev_end=%.1f pre=%.1f decode=%.1f post=%.1f ms\n", n,
                db_prev_end ? (t0 - db_prev_end) / 1e3 : -1.0, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3);
        db_prev_end = t3; } } db_g { db_t, req.n_tokens, db_t0, db_t1, db_t2 };
    g_mtp_tl.ok = false;   // any ordinary decode moves the state on
    if (req.n_tokens > 8 && getenv("ALPHA_DRE")) ggml_cuda_dre_set_prefill_tokens(req.n_tokens);   // before any borrow
    if (!req.prompt_ahead.empty()) {   // ALPHA_PLE_PREFETCH_AHEAD: the later chunks' PLE rows, read while this one computes
        llama_model_ple_prefetch(llama_get_model(ctx_), req.prompt_ahead.data(), (int32_t) req.prompt_ahead.size(), req.n_tokens);
    }
    if (req.n_tokens == 0) return false;

    // ALPHA_DRAFT_MODEL: mirror the target's token stream, indexed by absolute position, so the
    // drafter can be caught up incrementally instead of re-prefilled every round. This costs a
    // vector write per token; the catch-up decode happens lazily in draft_chain, so a request that
    // never reaches the MTP path never touches the drafter. Writes are idempotent by position, so a
    // token arriving twice (input echo vs commit replay) cannot corrupt the mirror.
    // ALPHA_MTP_NGRAM: seed the per-slot lookup ring from every token that passes through, prompt
    // included; on code workloads the prompt is where most repetition lives. Per slot, so -np > 1
    // stays correct.
    if (g_mtp_ngram > 0) {
        // ALPHA_HYBRID_PREFIX: a restored slot resumes mid-sequence and its ring still holds the old
        // request's continuation past the checkpoint. Clear it (draft quality only; verify is exact).
        for (int32_t rs : req.restore_seqs) {
            for (int i = 0; i < req.n_tokens; i++) {
                if (req.seq_ids[i] != rs) continue;
                const int sid = req.slot_ids[i];
                if (sid >= 0 && (int) ngram_hist_.size() > sid) ngram_hist_[sid].clear();
                break;
            }
        }
        for (int i = 0; i < req.n_tokens; i++) {
            const int sid = req.slot_ids[i];
            if (sid < 0) continue;
            if (req.positions[i] == 0) {                       // new sequence on this slot
                if ((int) ngram_hist_.size() > sid) ngram_hist_[sid].clear();
            }
            ngram_commit(sid, &req.tokens[i], 1);
        }
    }
    if (dctx_ && !draft_bad_) {
        for (int i = 0; i < req.n_tokens; i++) {
            const int32_t s = req.seq_ids[i];
            if (dseq_ < 0) dseq_ = s;
            if (s != dseq_) continue;           // one drafter, one sequence (n_seq_max = 1)
            const llama_pos p = req.positions[i];
            if (p == 0) {                        // new sequence on this slot: start the mirror over
                dhist_.clear();
                dpos_ = 0;
                if (dctx_) llama_memory_clear(llama_get_memory(dctx_), true);
            }
            if ((llama_pos) dhist_.size() < p) continue;   // gap: leave it, draft_chain will bail
            if ((llama_pos) dhist_.size() == p) dhist_.push_back(req.tokens[i]);
            else                                 dhist_[p] = req.tokens[i];
        }
    }

    // Clear KV cache for reused sequences
    if (!req.clear_seqs.empty()) {
        llama_memory_t mem = llama_get_memory(ctx_);
        for (int32_t seq_id : req.clear_seqs) {
            llama_memory_seq_rm(mem, seq_id, -1, -1);
        }
    }

    // ALPHA_HYBRID_PREFIX: restore recurrent checkpoints before the truncation below. The order
    // matters: llama_memory_hybrid::seq_rm tries the recurrent half first and, if it refuses (it does
    // whenever p0 <= the recurrent cell's pos), returns false without touching attention. After the
    // restore the recurrent cell sits at checkpoint-1, so seq_rm(checkpoint, -1) succeeds on both
    // halves and the attention KV is cut back to the checkpoint.
    for (int32_t seq_id : req.restore_seqs) {
        auto it = g_hyb_snap.find(seq_id);
        if (it == g_hyb_snap.end() || it->second.buf.empty()) {
            GGML_ABORT("[hybrid-prefix] restore requested for seq %d but no checkpoint is held -- "
                       "scheduler/engine checkpoint bookkeeping disagree", seq_id);
        }
        // Stale-checkpoint guard. The resumed suffix starts at this batch's lowest position for the
        // seq; the checkpoint must encode exactly the token before it. A request cancelled after its
        // checkpoint directive was queued but before it ran could otherwise leave an older snapshot
        // here while the slot believes a newer one exists, and restoring it would silently corrupt.
        llama_pos first = -1;
        for (int i = 0; i < req.n_tokens; i++) {
            if (req.seq_ids[i] == seq_id && (first < 0 || req.positions[i] < first)) first = req.positions[i];
        }
        if (first < 1 || it->second.pos != first - 1) {
            GGML_ABORT("[hybrid-prefix] STALE checkpoint for seq %d: snapshot encodes pos %d but the suffix "
                       "starts at %d", seq_id, (int) it->second.pos, (int) first);
        }
        const size_t n = llama_state_seq_set_data_ext(ctx_, it->second.buf.data(), it->second.buf.size(),
                                                      seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n == 0) {
            GGML_ABORT("[hybrid-prefix] recurrent checkpoint restore FAILED for seq %d", seq_id);
        }
        // The head-KV column for position `first` holds the old request's token there; re-stage
        // h_{first-1} so the suffix fill rebuilds it with the new token.
        if (it->second.has_mtp &&
            alpha_mtp_seq_restore(ctx_, seq_id, first, it->second.mtp_h)) {
            g_hyb_mtp_rows++;
        }
        g_hyb_restores++;
    }

    // Prefix cache: truncate KV for sequences with cached prefix.
    // Detect prefix-cached slots: their first position > 0 AND they are NOT
    // in clear_seqs (cleared slots start fresh at pos 0).
    // Must handle both multi-token prefill AND single-token suffix.
    {
        // Build set of cleared sequences (these start fresh, no truncation needed)
        std::unordered_set<int32_t> cleared_seqs(req.clear_seqs.begin(), req.clear_seqs.end());

        // Find minimum position per non-cleared sequence
        std::unordered_map<int32_t, int32_t> seq_min_pos;
        for (int i = 0; i < req.n_tokens; i++) {
            int32_t seq = req.seq_ids[i];
            if (cleared_seqs.count(seq)) continue;  // freshly cleared, skip
            int32_t pos = req.positions[i];
            auto it = seq_min_pos.find(seq);
            if (it == seq_min_pos.end() || pos < it->second) {
                seq_min_pos[seq] = pos;
            }
        }

        llama_memory_t mem = llama_get_memory(ctx_);
        for (auto & [seq_id, min_pos] : seq_min_pos) {
            if (min_pos > 0) {
                // On hybrid memory a refused truncation is a silent no-op on both halves (recurrent
                // is tried first and blocks attention), which would leave old KV positions above the
                // new prefill. Warn, and abort where the invariant is ours to keep.
                if (!llama_memory_seq_rm(mem, seq_id, min_pos, -1)) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        fprintf(stderr, "[alphaserve] WARNING: seq_rm(seq=%d, p0=%d) was REFUSED -- this model's "
                                        "state cannot be truncated; KV for this sequence is now inconsistent "
                                        "(set ALPHA_HYBRID_PREFIX=1 for checkpoint-based reuse)\n",
                                seq_id, min_pos);
                    }
                    // Fatal only where the invariant is ours: a seq restored in this batch must truncate
                    // cleanly (its recurrent cell now sits at checkpoint-1). Elsewhere, warn only.
                    if (std::find(req.restore_seqs.begin(), req.restore_seqs.end(), seq_id) !=
                        req.restore_seqs.end()) {
                        GGML_ABORT("[hybrid-prefix] seq_rm refused AFTER checkpoint restore (seq %d, p0 %d)",
                                   seq_id, min_pos);
                    }
                }
            }
        }
    }

    // Build llama_batch from BatchRequest
    common_batch_clear(batch_);
    for (int i = 0; i < req.n_tokens; i++) {
        common_batch_add(batch_,
            req.tokens[i],
            req.positions[i],
            { req.seq_ids[i] },
            req.logits_mask[i]);
    }

    // Set active user for per-user expert cache partitioning
    if (req.n_tokens > 0) {
        ggml_cuda_set_expert_user(req.slot_ids[0]);
    }

    // UBBoost (grow-only, sticky): a prefill chunk larger than the current n_ubatch means the
    // scheduler was configured for big-chunk prefill (--ubatch-prefill). Grow the compute buffers
    // once so the chunk runs as a single ubatch; on VRAM failure llama_set_n_ubatch rolls back and
    // the chunk just splits internally.
    // ALPHA_DRE borrow-and-refill: a grown prefill buffer is carved out of the expert pool for the
    // prompt and handed back (scheduler recreated at the base micro-batch, pool chunks re-allocated)
    // before the first decode.
    static const bool dre_borrow = getenv("ALPHA_DRE") && atoi(getenv("ALPHA_DRE")) != 0;
    if (dre_borrow && g_dre_lent && req.n_tokens <= g_serve_prompt_chunk) dre_return_pool(ctx_, "first decode");
    // ALPHA_DRE: a prompt that fits the base micro-batch still streams its experts through the prompt
    // staging buffers once it has more than 8 tokens, and without a borrow those come out of the
    // small decode headroom and can OOM. Borrow for it too (only what is not already free) and hand it
    // back at the next decode without the scheduler rebuild the grown-micro-batch return needs.
    {
        static bool lent_short = false;
        if (dre_borrow && lent_short && req.n_tokens <= 8) {
            ggml_cuda_dre_return_begin(0);
            ggml_cuda_dre_return_finish();
            lent_short = false;
        }
        if (dre_borrow && !g_dre_lent && !lent_short && req.n_tokens > 8 && req.n_tokens <= (int) llama_n_ubatch(ctx_)) {
            lent_short = ggml_cuda_dre_borrow((size_t) 384 << 20) > 1;
        }
    }
    if (req.n_tokens > (int)llama_n_ubatch(ctx_)) {
        if (dre_borrow) {
            static const size_t per_tok = []() {
                const char * e = getenv("ALPHA_DRE_PREFILL_MB_PER_TOK");
                return (size_t) ((e ? atof(e) : 0.40) * 1048576.0);
            }();
            const size_t need = (size_t) (req.n_tokens - (int) llama_n_ubatch(ctx_)) * per_tok + ((size_t) 384 << 20);
            const int64_t tb0 = ggml_time_us();
            if (ggml_cuda_dre_borrow(need) > 0) g_dre_lent = true;
            if (getenv("ALPHA_DRE_RETURN_TIMERS")) fprintf(stderr, "[dre-rt] borrow %.1f ms\n", (ggml_time_us() - tb0) / 1e3);
        }
        // ALPHA_KV_RESERVE_CAP (default on with the DRE borrow): reserve the grown micro-batch for the
        // KV this prompt can reach, not the whole context; the n_kv x n_tokens masks dominate the buffer
        static const bool kv_cap = [] { const char * e = getenv("ALPHA_KV_RESERVE_CAP"); return !e || atoi(e) != 0; }();
        if (dre_borrow && kv_cap) {
            // reserve for the whole prompt now; capping at this chunk would make every later chunk
            // re-reserve the scheduler
            int32_t pmax = std::max(0, req.prompt_end - 1);
            for (int i = 0; i < req.n_tokens; i++) pmax = std::max<int32_t>(pmax, req.positions[i]);
            alpha_set_kv_reserve_cap((uint32_t) pmax + 1);
        }
        const int64_t tu0 = ggml_time_us();
        llama_set_n_ubatch(ctx_, (uint32_t)req.n_tokens);
        const int64_t tu1 = ggml_time_us();
        if (dre_borrow && g_dre_lent) ggml_cuda_dre_pipe_prestart();   // the first layers' expert copies start now
        if (getenv("ALPHA_DRE_RETURN_TIMERS")) fprintf(stderr, "[dre-rt] set_n_ubatch %.1f ms, prestart %.1f ms\n",
                                                       (tu1 - tu0) / 1e3, (ggml_time_us() - tu1) / 1e3);
    } else if (dre_borrow && g_dre_lent && alpha_get_kv_reserve_cap() > 0) {
        if (req.n_tokens > 8) ggml_cuda_dre_pipe_prestart();   // a later prompt chunk: its first layers' copies start now
        // a later chunk of the same prompt reaches past the capped reservation: raise it and re-reserve
        int32_t pmax = 0;
        for (int i = 0; i < req.n_tokens; i++) pmax = std::max<int32_t>(pmax, req.positions[i]);
        if ((uint32_t) pmax + 1 > alpha_get_kv_reserve_cap()) {
            alpha_set_kv_reserve_cap((uint32_t) pmax + 1);
            llama_context_reset_sched(ctx_);
        }
    }

    backend_sampler_sync(req);

    // GPU forward pass
    db_t1 = ggml_time_us();
    int ret = alpha_counted_decode(ctx_, batch_);
    db_t2 = ggml_time_us();
    if (ret != 0) {
        fprintf(stderr, "[alphaserve] llama_decode failed, ret = %d\n", ret);
        return false;
    }

    // ALPHA_HYBRID_PREFIX: checkpoint the recurrent state of seqs whose prefill reached their
    // checkpoint in this batch. PARTIAL_ONLY on llama_memory_hybrid writes only the recurrent half
    // (fixed size, independent of prefix length), which is the half that cannot be rebuilt by
    // truncation. llama_state_seq_get_data synchronises, so this reads the post-decode state.
    for (int32_t seq_id : req.snapshot_seqs) {
        const int64_t t0 = ggml_time_us();
        const size_t sz = llama_state_seq_get_size_ext(ctx_, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        auto & ck = g_hyb_snap[seq_id];
        ck.buf.resize(sz);
        const size_t n = llama_state_seq_get_data_ext(ctx_, ck.buf.data(), sz, seq_id,
                                                      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != sz) {
            GGML_ABORT("[hybrid-prefix] checkpoint write size mismatch for seq %d (%zu vs %zu)", seq_id, n, sz);
        }
        // The batch's chunk for this seq ended at the checkpoint, so this is checkpoint-1.
        ck.pos = llama_memory_seq_pos_max(llama_get_memory(ctx_), seq_id);
        ck.has_mtp = alpha_mtp_seq_save(ctx_, seq_id, ck.pos, ck.mtp_h);
        if (!ck.has_mtp) ck.mtp_h.clear();
        g_hyb_snaps++;
        static bool logged = false;
        if (!logged) {
            logged = true;
            fprintf(stderr, "[hybrid-prefix] first recurrent checkpoint: seq %d, %.1f MB, %.2f ms, mtp_row=%d\n",
                    seq_id, sz / 1048576.0, (ggml_time_us() - t0) / 1000.0, (int) ck.has_mtp);
        }
    }
    // ALPHA_MTP_KV_HASH: after the final prefill chunk of a sequence (several rows, logits taken),
    // print a hash of its MTP head-KV columns. A reused prefix must reproduce a full reprocess's head
    // state exactly, so this must match the flag-off run request for request.
    static const bool g_mtp_kv_hash = [] { const char * e = getenv("ALPHA_MTP_KV_HASH"); return e && strcmp(e, "1") == 0; }();
    if (g_mtp_kv_hash) {
        std::unordered_map<int32_t, std::pair<int, bool>> seen;   // seq -> (rows, logits taken)
        for (int i = 0; i < req.n_tokens; i++) {
            auto & e = seen[req.seq_ids[i]];
            e.first++;
            if (i < (int) req.logits_mask.size() && req.logits_mask[i]) e.second = true;
        }
        for (const auto & kv : seen) {
            if (kv.second.first < 2 || !kv.second.second) continue;
            int32_t lo = -1, hi = -1;
            const uint64_t h = alpha_mtp_seq_kv_hash(ctx_, kv.first, lo, hi);
            fprintf(stderr, "[mtp-kvhash] seq %d slots [%d,%d] hash %016llx\n",
                    kv.first, lo, hi, (unsigned long long) h);
        }
    }
    if (!req.restore_seqs.empty() || !req.snapshot_seqs.empty()) {
        static int64_t n_logged = 0;
        if (n_logged++ < 64) {
            fprintf(stderr, "[hybrid-prefix] batch: restores=%zu snapshots=%zu (totals: %lld restores, %lld snapshots, %lld mtp rows)\n",
                    req.restore_seqs.size(), req.snapshot_seqs.size(),
                    (long long) g_hyb_restores, (long long) g_hyb_snaps, (long long) g_hyb_mtp_rows);
        }
    }

    // Read sampled tokens (backend sampling or fallback)
    result.n_slots = 0;

    // Collect unique slots and their sampled tokens
    std::vector<int> seen_slots;
    for (int i = req.n_tokens - 1; i >= 0; i--) {
        if (!req.logits_mask[i]) continue;

        int slot_id = req.slot_ids[i];
        bool already = false;
        for (int s : seen_slots) {
            if (s == slot_id) { already = true; break; }
        }
        if (already) continue;

        // Sampling: a slot that requested temperature/top_k/top_p is drawn host-side from its
        // logits (the greedy GPU-argmax token is ignored for it). Greedy slots keep the 4-byte
        // backend path. samp_temp>0 flags a sampling token (the scheduler sets it per slot).
        const bool do_sample = (i < (int)req.samp_temp.size()) && req.samp_temp[i] > 0.0f;
        llama_token tok = LLAMA_TOKEN_NULL;
        if (do_sample) {
            const float * logits = llama_get_logits_ith(ctx_, i);
            if (logits) {
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_)));
                tok = alpha_sample_token(logits, n_vocab, req.samp_temp[i],
                                         req.samp_top_k[i], req.samp_top_p[i], req.samp_rng[i]);
                // ALPHA_SEED_TRACE: the seed schedule cannot be validated by comparing generated
                // text, since MTP and plain decode see slightly different logits (different batch
                // widths, non-batch-invariant kernels). Trace (position, seed) in both paths instead.
                if (getenv("ALPHA_SEED_TRACE"))
                    fprintf(stderr, "[seedtrace] plain pos=%d seed=%llu\n",
                            (int) req.positions[i], (unsigned long long) req.samp_rng[i]);
            }
        }
        if (tok == LLAMA_TOKEN_NULL) {
            // Greedy: try backend GPU-argmax first (4 bytes, no logits transfer)
            tok = llama_get_sampled_token_ith(ctx_, i);
        }
        if (tok == LLAMA_TOKEN_NULL) {
            // Fallback: greedy (argmax) on CPU
            const float * logits = llama_get_logits_ith(ctx_, i);
            if (logits) {
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_)));
                tok = 0;
                float max_logit = logits[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (logits[v] > max_logit) {
                        max_logit = logits[v];
                        tok = v;
                    }
                }
            }
        }

        // ALPHA_LOG_MARGIN=1: log the per-token top1-top2 logit margin (confidence signal). Reads
        // CPU logits; if the GPU-sampling path did not copy them to host, logs NULL.
        static bool log_margin = alpha_env_on("ALPHA_LOG_MARGIN");
        if (log_margin) {
            const float * lg = llama_get_logits_ith(ctx_, i);
            if (lg) {
                const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_)));
                float b1 = -1e30f, b2 = -1e30f;
                for (int v = 0; v < nv; v++) { float x = lg[v]; if (x > b1) { b2 = b1; b1 = x; } else if (x > b2) { b2 = x; } }
                fprintf(stderr, "[margin] tok=%d m=%.4f\n", tok, (double)(b1 - b2));
            } else {
                fprintf(stderr, "[margin] tok=%d m=NULL\n", tok);
            }
        }

        result.sampled.push_back(tok);
        seen_slots.push_back(slot_id);
        result.n_slots++;
    }

    // Reverse to match slot order (we iterated backwards)
    std::reverse(result.sampled.begin(), result.sampled.end());
    std::reverse(seen_slots.begin(), seen_slots.end());
    result.slot_ids = std::move(seen_slots);

    // ALPHA_LOGIT_DUMP: record the logits per generated token, with a {-1, n_vocab} marker at each
    // request's prefill.
    if (g_logit_dump_path && result.n_slots == 1 && !result.sampled.empty()) {
        static FILE * df = fopen(g_logit_dump_path, "wb");
        static std::vector<uint16_t> h16;
        if (df) {
            const llama_vocab * dv = llama_model_get_vocab(llama_get_model(ctx_));
            const int nv = llama_vocab_n_tokens(dv);
            int li = -1;
            if (req.n_tokens > 1 && req.logits_mask[req.n_tokens - 1]) li = req.n_tokens - 1;
            else if (req.n_tokens == 1 && req.logits_mask[0]) li = 0;
            if (li >= 0) {
                const float * lg = llama_get_logits_ith(ctx_, li);
                if (lg) {
                    if (req.n_tokens > 1) {                       // request boundary
                        int32_t m[2] = { -1, nv };
                        fwrite(m, 4, 2, df);
                    }
                    int32_t tok = (int32_t)result.sampled[0];
                    fwrite(&tok, 4, 1, df);
                    h16.resize((size_t)nv);
                    ggml_fp32_to_fp16_row(lg, (ggml_fp16_t *)h16.data(), nv);
                    fwrite(h16.data(), 2, (size_t)nv, df);
                    fflush(df);
                }
            }
        }
    }

    // ALPHA_DRE_RETURN_AT_PREFILL (default 1): the prompt's last chunk (several rows, logits taken)
    // gives the borrowed pool back here, before its first token is handed out, so the return counts
    // as prompt time rather than first-decode time. 0 = return at the first decode batch.
    static const bool ret_at_prefill = [] { const char * e = getenv("ALPHA_DRE_RETURN_AT_PREFILL"); return !e || atoi(e) != 0; }();
    if (ret_at_prefill && g_dre_lent && req.n_tokens > 1) {
        bool last = false;
        for (int i = 0; i < req.n_tokens && !last; i++) last = req.logits_mask[i] != 0;
        if (last) dre_return_pool(ctx_, "end of prompt");
    }

    n_decodes_++;
    n_tokens_ += req.n_tokens;
    return true;
}

// Synchronous: pop one batch from the queue, decode, push the result. Called from the scheduler
// thread (single-thread mode).
// On a decode failure (KV exhausted, CUDA error, ...) the error result lists which slots were in
// the batch so the scheduler can force-finish them; otherwise it would resend the same failing
// batch forever.
static BatchResult make_error_result(const BatchRequest & req) {
    BatchResult r;
    r.decode_error = true;
    for (int i = 0; i < req.n_tokens; i++) {
        int sid = req.slot_ids[i];
        bool seen = false;
        for (int j = 0; j < r.n_slots; j++) {
            if (r.slot_ids[j] == sid) { seen = true; break; }
        }
        if (!seen) {
            r.slot_ids.push_back(sid);
            r.sampled.push_back(LLAMA_TOKEN_NULL);
            r.n_slots++;
        }
    }
    return r;
}

bool gpu_engine::process_one() {
    BatchRequest req;
    if (!in_->try_pop(req)) return false;
    if (pre_batch_hook) pre_batch_hook();

    BatchResult result;
    static const bool tl_on = [] { const char * e = getenv("ALPHA_MTP_TRIGGERLESS"); return !e || atoi(e) != 0; }();   // default on
    if (tl_on && g_mtp && g_mtp_tl.ok && req.n_tokens == 1 && req.logits_mask[0] && req.prompt_end == 0 && req.clear_seqs.empty() &&
        req.restore_seqs.empty() && req.snapshot_seqs.empty() && req.slot_ids[0] == g_mtp_tl.slot &&
        req.seq_ids[0] == g_mtp_tl.seq && req.tokens[0] == g_mtp_tl.tok && req.positions[0] == g_mtp_tl.pos &&
        (req.samp_temp.empty() || req.samp_temp[0] <= 0.0f)) {
        g_mtp_tl.ok = false;
        result.slot_ids = { req.slot_ids[0] };
        result.sampled  = { req.tokens[0] };   // placeholder: already emitted by the previous call
        result.n_slots  = 1;
        g_mtp_tl_active = true;
        mtp_window_extend(req, result);
        g_mtp_tl_active = false;
        if (!result.extra_tokens.empty()) {
            result.sampled[0] = result.extra_tokens.front();
            result.extra_tokens.erase(result.extra_tokens.begin());
            out_->try_push(std::move(result));
            return true;
        }
        result = BatchResult();   // no window ran (state untouched): decode it the ordinary way
    }
    if (decode_batch(req, result)) {
        // prompt_end == 0: a decode step, never a prompt's own 1-token chunk (a window run off a
        // 1-token prompt's prefill fails with inconsistent sequence positions)
        if (g_mtp && !result.decode_error && result.n_slots == 1 &&
            req.n_tokens == 1 && req.logits_mask[0] && req.prompt_end == 0 && result.sampled.size() == 1) {
            mtp_window_extend(req, result);
        }
        out_->try_push(std::move(result));
    } else {
        out_->try_push(make_error_result(req));
    }
    return true;
}

// MTP self-speculative decode step: draft the next token from the nextn head, verify with a
// batched decode, emit the accepted tokens. Exact-match accept keeps the output identical to plain
// greedy. Hybrid-safe: alpha_recurrent_snapshot/restore + seq_rm (never seq_cp). State contract:
// on return, KV and recurrent state are exact through pos0 + extra_tokens.size() - 1; the scheduler
// feeds extra_tokens.back() next.

// Prompt-lookup drafter (ALPHA_MTP_NGRAM).
// A backwards CPU scan over the slot's recent tokens (prompt and generated) finds the most recent
// earlier occurrence of the last 1-3 tokens and proposes what followed it. Used unconditionally it
// hurts on prose and some code, so it is routed: the match length is known before any GPU work, and
// a 3-token match means this context really occurred before. The fallback arm is the
// margin-gated MTP head. A bounded scan window keeps the cost flat at any context length.
// The flags are declared earlier (first use is in decode_batch).

// Advance every pending lookup chain against one newly committed token. This lets the gate learn
// from windows it declined: the lookup is free, so what it would have drafted can always be scored
// against what the model produced, keeping the estimate live without paying for a bad route.
//
// Verify cost as a function of verify width. This is a table rather than a line: the curve is steep
// up to K~8 and flat after, so a linear fit overestimates the marginal cost past K=8 and
// under-selects depth. The bootstrap points are model-, quant- and GPU-specific; only the shape
// transfers (re-measure with scripts/wcurve.sh). Runtime observations override them per width.
double gpu_engine::ngram_verify_us(const gpu_engine::ngram_adapt & A, double w) {
    static const int    K[8] = { 1,     4,     8,     12,    16,    20,    24,    32    };
    static const double C[8] = { 28640, 30570, 42660, 43710, 45410, 46480, 50020, 53420 };
    const int wi = (int) (w + 0.5);
    // prefer what this stream actually measured at this width
    if (wi >= 0 && wi < 33 && A.vtab_cnt[wi] >= 3) return A.vtab_sum[wi] / (double) A.vtab_cnt[wi];
    if (w <= K[0]) return C[0];
    for (int i = 1; i < 8; ++i) {
        if (w <= K[i]) return C[i-1] + (C[i] - C[i-1]) * (w - K[i-1]) / (double) (K[i] - K[i-1]);
    }
    return C[7];
}

void gpu_engine::ngram_adapt_score(int slot_id, llama_token tok) {
    if ((int) ngram_adapt_.size() <= slot_id) return;
    auto & A = ngram_adapt_[slot_id];
    const double snap = A.snap_n ? A.snap_us / (double) A.snap_n : 0.0;
    for (auto it = A.pend.begin(); it != A.pend.end(); ) {
        auto & q = *it;
        if (q.n_scored < q.chain.size()) {
            if (!q.broken && q.chain[q.n_scored] == tok) q.prefix_ok++;
            else                                         q.broken = true;
            q.n_scored++;
        }
        if (q.n_scored >= q.chain.size()) {
            for (int d = 1; d <= (int) q.chain.size() && d < 33; ++d) {
                const double w    = (double) (d + 1);          // verify width = chain + anchor
                const double cost = snap + ngram_verify_us(A, w);   // the draft is free on the routed arm
                if (cost > 1.0) {
                    const int acc = q.prefix_ok < d ? q.prefix_ok : d;
                    if (g_mtp_ngram_decay < 1.0f) {          // exponential forgetting
                        A.yld_sum[d] *= g_mtp_ngram_decay;
                        A.yld_cnt[d] *= g_mtp_ngram_decay;   // double: converges to 1/(1-decay)
                    }
                    A.yld_sum[d] += ((double) acc + 1.0) / cost;
                    A.yld_cnt[d] += 1;
                }
            }
            A.prefix_sum += q.prefix_ok; A.prefix_n += 1;
            { const int b = q.pos0 / 64; if (b >= 0 && b < 16) { A.posb_sum[b] += q.prefix_ok; A.posb_cnt[b] += 1; } }
            A.prefix_zero += (q.prefix_ok == 0) ? 1 : 0;
            it = A.pend.erase(it);
        } else {
            ++it;
        }
    }
}

int gpu_engine::ngram_adapt_pick(int slot_id, int max_d) const {
    if ((int) ngram_adapt_.size() <= slot_id) return 0;
    const auto & A = ngram_adapt_[slot_id];
    if (!A.head_cnt) return 0;                      // no head baseline yet -> do not route
    const double head = A.head_sum / (double) A.head_cnt;
    int    best_d = 0;
    double best_y = head;                           // must strictly beat the head to route
    for (int d = 1; d <= max_d && d < 33; ++d) {
        if (A.yld_cnt[d] < (double) g_mtp_ngram_minobs) continue;
        const double y = A.yld_sum[d] / (double) A.yld_cnt[d];
        if (y > best_y) { best_y = y; best_d = d; }
    }
    return best_d;
}

void gpu_engine::ngram_commit(int slot_id, const llama_token * toks, int n) {
    if (slot_id < 0 || n <= 0) return;
    if ((int) ngram_hist_.size() <= slot_id) ngram_hist_.resize(slot_id + 1);
    auto & h = ngram_hist_[slot_id];
    for (int i = 0; i < n; ++i) {
        if (g_mtp_ngram_adapt) ngram_adapt_score(slot_id, toks[i]);
        h.push_back(toks[i]);
        // Draw the estimator's sample here, on a schedule that no routing decision can influence.
        if (g_mtp_ngram_adapt && g_mtp_ngram_stride > 0 && !h.empty() &&
            (h.size() % (size_t) g_mtp_ngram_stride) == 0) {
            std::vector<llama_token> ng;
            const int want = g_mtp_ngram_maxlen > 0 ? g_mtp_ngram_maxlen : g_mtp_draft_n;
            const int m = ngram_lookup(slot_id, h.back(), want, ng);
            if (m >= g_mtp_ngram_min && !ng.empty()) {
                if ((int) ngram_adapt_.size() <= slot_id) ngram_adapt_.resize(slot_id + 1);
                ngram_adapt::pending q; q.chain = ng; q.pos0 = (int) h.size();
                ngram_adapt_[slot_id].pend.push_back(std::move(q));
                if (ngram_adapt_[slot_id].pend.size() > 64) ngram_adapt_[slot_id].pend.pop_front();
            }
        }
    }
    // bounded ring: keep the scan cost flat at any context length
    const size_t cap = (size_t) g_mtp_ngram_win * 2;
    if (h.size() > cap) h.erase(h.begin(), h.begin() + (h.size() - g_mtp_ngram_win));
}

int gpu_engine::ngram_lookup(int slot_id, llama_token last, int m,
                             std::vector<llama_token> & out) const {
    out.clear();
    if (slot_id < 0 || (int) ngram_hist_.size() <= slot_id) return 0;
    const auto & h = ngram_hist_[slot_id];
    if (h.empty()) return 0;

    // pattern = the last `want` committed tokens, with `last` as the final element
    // The ring holds every emitted token including each window's t_next, so `h` already ends with
    // the current anchor and the pattern is simply h's tail.
    if (!h.empty() && h.back() != last) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[ngram] RING DESYNC: h.back()=%d but anchor=%d -- lookups are matching a "
                            "stream that never existed\n", (int) h.back(), (int) last);
        }
    }
    for (int want = 3; want >= 1; --want) {
        if ((int) h.size() < want + 1) continue;
        const int pstart = (int) h.size() - want;      // pattern = the last `want` tokens of h
        const int lo = (int) h.size() > g_mtp_ngram_win ? (int) h.size() - g_mtp_ngram_win : 0;
        for (int s = pstart - 1; s >= lo; --s) {       // most recent earlier occurrence
            bool eq = true;
            for (int j = 0; j < want; ++j) {
                if (h[s + j] != h[pstart + j]) { eq = false; break; }
            }
            if (!eq) continue;
            for (int j = 0; j < m && s + want + j < pstart; ++j) {
                out.push_back(h[s + want + j]);
            }
            if (!out.empty()) return want;
            break;   // matched but nothing followed -> try a shorter pattern
        }
    }
    return 0;
}

// Latched when the verify-rollback mirror cannot be allocated. Otherwise the window would bail after
// drafting and retry a large cudaMalloc on every token. Fail once, loudly, and fall back to plain
// decode for the life of the process.
static bool g_mtp_snap_dead = false;

void mtp_disable(const char * why) {
    if (!g_mtp_snap_dead) fprintf(stderr, "[mtp] windows disabled: %s\n", why ? why : "(no reason)");
    g_mtp_snap_dead = true;
}

void gpu_engine::mtp_window_extend(const BatchRequest & req, BatchResult & result) {
    if (g_mtp_snap_dead) return;   // emit the plain sampled token
    llama_memory_t      mem   = llama_get_memory(ctx_);
    const llama_model * model = llama_get_model(ctx_);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t     seq     = req.seq_ids[0];
    const int         slot_id = req.slot_ids[0];
    // Chain windows inside one call (ALPHA_MTP_LOOP).
    //
    // A separate single-token trigger forward per window, only to obtain the hidden H and the next
    // token, is redundant: the verify's last row sits at pos0+k and already carries both:
    //   H(pos0+k)          -> the hidden the next window drafts from (every row is exported and the
    //                         capture keeps the last one)
    //   argmax of that row -> t_next, the token at pos0+k+1, i.e. the next window's anchor
    // and t_next is then consumed as column 0 of the next verify batch, exactly as `last` is here.
    // The reject path lands in the same shape: after the relay the last row sits at pos0+n_acc.
    // So a window hands straight to the next when generation continues.
    //
    // The final anchor is still emitted and the caller still decodes it, so one trigger forward is
    // amortised over ALPHA_MTP_LOOP windows instead of being paid by every window.
    static const int g_mtp_loop = []() {
        const char * e = getenv("ALPHA_MTP_LOOP");
        // Default 3: the smallest value that did not regress any workload in testing; larger values
        // can be faster on average but regress some prompts.
        const int v = e ? atoi(e) : 3;
        return v < 1 ? 1 : (v > 64 ? 64 : v);
    }();
    // ALPHA_MTP_LOOP_FULLONLY: chain only out of a window that verified cleanly (every draft
    // accepted). A fully accepted window needs no repair (no fold, no KV trim). Breaking chains
    // after partial accepts forfeits most of the trigger amortisation that LOOP exists for, so this
    // is off by default.
    static const bool g_mtp_loop_fullonly = alpha_env_on("ALPHA_MTP_LOOP_FULLONLY");
    // function scope: the per-chain summary below the loop reads these
    static int64_t mtp_steps = 0, mtp_accept = 0, mtp_drafted = 0, mtp_accept_full = 0;
    // This window is single-sequence by construction; check it rather than assume it.
    //
    // Everything below reads row 0 of the request: `result.sampled[0]`, `req.positions[0]`, and
    // `req.seq_ids[0]` at the rewind sites. With -np > 1 a decode request can be a mixed batch, so
    // row 0 belongs to whichever sequence the scheduler placed first, and since row order is not
    // stable across decodes the window could compute pos0 from one sequence while trimming another
    // (llama_decode then fails with an M-RoPE position error). Until the window is per-sequence
    // (partition by seq_id and run one window per sequence, as mtp_kv_fill_all does), decline on a
    // mixed batch and emit the plain sampled tokens. Single-sequence requests are unaffected.
    if (!req.seq_ids.empty()) {
        const int32_t s0 = req.seq_ids[0];
        for (size_t i = 1; i < req.seq_ids.size(); ++i) {
            if (req.seq_ids[i] != s0) {
                static int64_t n_mixed = 0;
                if (++n_mixed <= 3 || (n_mixed % 512) == 0) {
                    fprintf(stderr, "[mtp] declining a MIXED-sequence batch (%zu rows, seq %d..%d) -- "
                                    "the draft window is single-sequence and would operate on row 0's "
                                    "sequence only (occurrence %lld)\n",
                            req.seq_ids.size(), s0, (int) req.seq_ids[i], (long long) n_mixed);
                }
                return;                      // emit the plain sampled tokens; no speculation
            }
        }
    }
    llama_token last = result.sampled[0];
    llama_pos   pos0 = req.positions[0] + 1;            // position `last` occupies
    if (g_mtp_tl_active) pos0 = req.positions[0];       // triggerless: `last` is the request's own undecoded token
    if (last == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, last)) return; // emit just `last`
    result.extra_tokens.clear();
    bool stop_chain = false;
    bool mtp_failed = false;

    // ALPHA_MTP_WIN_STATS: wall time of this call's window loop, windows per call, and the gap since the previous
    // call returned (the scheduler round + the trigger decode that starts every call)
    static const int ws_every = []() { const char * e = getenv("ALPHA_MTP_WIN_STATS"); return e ? atoi(e) : 0; }();
    static std::chrono::steady_clock::time_point ws_prev_exit{};
    static double ws_gap_ms = 0, ws_loop_ms = 0; static long long ws_calls = 0, ws_wins = 0, ws_toks = 0;
    // window phase marks: 0 loop top, 1 draft start, 2 drafts done, 3 verify done, 4 fold start, 5 fold done, 6 body end
    static double ws_ph[7] = {0}; std::chrono::steady_clock::time_point ws_m[7];
    static double ws_iter_ms = 0, ws_pre_loop_ms = 0; static long long ws_iters = 0;
    static double ws_slow_ms = 0; static long long ws_slow_n = 0;
    std::chrono::steady_clock::time_point ws_iter_prev{};
    const auto ws_t0 = std::chrono::steady_clock::now();
    if (ws_every > 0 && ws_prev_exit.time_since_epoch().count() != 0) {
        const double g = std::chrono::duration<double, std::milli>(ws_t0 - ws_prev_exit).count();
        if (g < 100.0) ws_gap_ms += g;   // longer = a new request (prefill), not a decode-path gap
    }
    int ws_nw = 0;
    struct ws_guard_t {
        std::function<void()> f; ~ws_guard_t() { f(); }
    } ws_guard{[&]() {
        if (ws_every <= 0) return;
        const auto t1 = std::chrono::steady_clock::now();
        ws_loop_ms += std::chrono::duration<double, std::milli>(t1 - ws_t0).count();
        ws_calls++; ws_wins += ws_nw; ws_toks += (long long) result.extra_tokens.size() + 1;
        ws_prev_exit = t1;
        if (ws_calls % 16 == 0) {
            fprintf(stderr, "[mtp-call] last 16 calls: %.1f windows/call, %.1f tokens/call, loop %.2f ms/window, "
                            "gap between calls %.2f ms (trigger decode + scheduler), %.2f ms/token overall\n",
                    (double) ws_wins / 16, (double) ws_toks / 16, ws_loop_ms / std::max(1LL, ws_wins), ws_gap_ms / 16,
                    (ws_loop_ms + ws_gap_ms) / std::max(1LL, ws_toks));
            ws_gap_ms = ws_loop_ms = 0; ws_wins = ws_toks = 0;
        }
    }};
  // MTP budget: the scheduler drops every token past max_tokens, so windows beyond the request's remaining
  // budget are wasted work that would count in its decode time. Stop the chain at the budget.
  const int64_t mtp_n_left = req.samp_n_left.empty() ? ((int64_t) 1 << 40) : (int64_t) req.samp_n_left[0];
  for (int mtp_w = 0; mtp_w < g_mtp_loop && !stop_chain; mtp_w++) {
    // a window emits at most (drafts + 1) tokens: it needs at least one draft slot left to be worth running
    const int64_t win_room = mtp_n_left - 1 - (int64_t) result.extra_tokens.size();
    if (win_room < 2) break;
    ws_nw++;
    ws_m[0] = std::chrono::steady_clock::now();
    for (int i = 1; i < 7; i++) ws_m[i] = ws_m[0];
    if (mtp_w == 0) ws_pre_loop_ms += std::chrono::duration<double, std::milli>(ws_m[0] - ws_t0).count();
    else {
        const double it_ms = std::chrono::duration<double, std::milli>(ws_m[0] - ws_iter_prev).count();
        ws_iter_ms += it_ms; ws_iters++;
        if (it_ms > 35.0) { ws_slow_n++; ws_slow_ms += it_ms; }   // WIN_STATS outliers: graph rebuilds/recaptures
    }
    ws_iter_prev = ws_m[0];

    // 1. Draft the token for pos0+1 from the MTP head (hidden H captured by the trigger decode +
    //    embed(last)). Pure compute; no KV/recurrent mutation.
    // Chain the head m times. Each step feeds the previous draft as the embedding input; the carried
    // hidden H stays the one captured by the trigger decode (the head is trained to predict one step
    // from (h_t, x_{t+1}), so deeper steps reuse h_t, an approximation that shows up as acceptance
    // decaying with depth).
    const auto mtp_t0 = std::chrono::steady_clock::now();
    ws_m[1] = ws_m[2] = ws_m[3] = ws_m[4] = ws_m[5] = mtp_t0;
    if (getenv("ALPHA_MTP_DBG")) {
        // whole-window wall time between consecutive chained windows (everything: draft, verify, fold, accept, emit)
        static std::chrono::steady_clock::time_point prev_t0;
        static double us_win = 0; static int64_t n_win = 0;
        if (mtp_w > 0) {
            us_win += std::chrono::duration<double, std::micro>(mtp_t0 - prev_t0).count();
            if ((++n_win & 63) == 0) fprintf(stderr, "[mtp-win] %.2f ms/window wall (chained windows)\n", us_win / n_win / 1e3);
        }
        prev_t0 = mtp_t0;
    }
    // This window's sequence owns the head state the draft chain is about to use. Without this the
    // chain reads whichever sequence last touched the context, which with -np > 1 is not this one.
    llama_mtp_set_seq(ctx_, seq);
    std::vector<llama_token> d;
    d.reserve(g_mtp_draft_n);
    // ALPHA_DRAFT_MODEL: prefer an external drafter when one is loaded. Everything downstream (verify,
    // accept prefix, ReplaySSM fold, KV trim) is identical either way; only the source of `d` changes.
    // An empty return (mirror gap, catch-up failure, EOG on the first step) falls through to the
    // head, so a drafter that cannot help this window costs one check and nothing else.
    bool used_ext_draft = false;
    // ALPHA_MTP_NGRAM: free CPU lookup first; the head only runs when the lookup is guessing. The
    // match length is the routing signal and costs nothing to obtain.
    int ngram_matched = 0;
    if (g_mtp_ngram > 0 && !used_ext_draft) {
        // Print the resolved config once, so a clamped, misspelled or aliased knob is visible.
        static bool cfg_once = false;
        if (!cfg_once) {
            cfg_once = true;
            fprintf(stderr, "[ngram] resolved: min_match=%d win=%d pad=%d min_chain=%d max_chain=%d draft_n=%d tau=%.2f\n",
                    g_mtp_ngram_min, g_mtp_ngram_win, g_mtp_ngram_pad, g_mtp_ngram_minlen, g_mtp_ngram_maxlen,
                    g_mtp_draft_n, (double) g_mtp_margin_tau);
        }
        std::vector<llama_token> ng;
        // The routed chain is not bound by the head's draft_n. Drafting it is free, and past width ~10
        // extra verify columns are nearly free too, while routed acceptance grows with width.
        const int ng_want = g_mtp_ngram_maxlen > 0 ? g_mtp_ngram_maxlen : g_mtp_draft_n;
        ngram_matched = ngram_lookup(slot_id, last, ng_want, ng);
        // Pad a short chain out to draft_n with the last token. Off by default; see the
        // g_mtp_ngram_pad comment.
        if (g_mtp_ngram_pad && !ng.empty()) {
            while ((int) ng.size() < g_mtp_draft_n) ng.push_back(ng.back());
        }
        // a 1-token chain spends a whole verify window to win at most 2 tokens; optionally decline it
        if (g_mtp_ngram_minlen > 0 && (int) ng.size() < g_mtp_ngram_minlen) ngram_matched = 0;
        // truncate an over-wide chain: every column past the acceptance tail is paid for in verify
        if (g_mtp_ngram_maxlen > 0 && (int) ng.size() > g_mtp_ngram_maxlen) ng.resize(g_mtp_ngram_maxlen);
        if (g_mtp_ngram_adapt) {
            // Always record the full chain for scoring, even when declining to route: the estimate
            // stays live on declined windows because the lookup is free.
            if ((int) ngram_adapt_.size() <= slot_id) ngram_adapt_.resize(slot_id + 1);
            if (g_mtp_ngram_stride == 0 && ngram_matched >= g_mtp_ngram_min && !ng.empty()) {
                ngram_adapt::pending q; q.chain = ng; q.pos0 = pos0;
                ngram_adapt_[slot_id].pend.push_back(std::move(q));
                if (ngram_adapt_[slot_id].pend.size() > 64) ngram_adapt_[slot_id].pend.pop_front();
            }
            const int best_d = ngram_adapt_pick(slot_id, (int) ng.size());
            if (best_d > 0) {
                ng.resize(best_d);          // route at the depth that maximises yield-per-ms
                ngram_adapt_[slot_id].routed++;
            } else {
                ngram_matched = 0;          // decline; the head runs and the estimate still learns
                ngram_adapt_[slot_id].declined++;
            }
        }
        if (ngram_matched >= g_mtp_ngram_min && !ng.empty()) {
            d = ng;
            used_ext_draft = true;          // skip the head entirely: draft cost 0 ms
        }
    }
    // Pure n-gram: suppress the head on a miss without touching used_ext_draft. That flag also gates
    // the external-drafter bookkeeping further down (draft_commit / rollback), and setting it with no
    // drafter attached desyncs the n-gram ring. A dedicated flag only skips the head loop.
    const bool suppress_head = (g_mtp_ngram_only && !used_ext_draft);
    if (!used_ext_draft && dctx_ && !draft_bad_) {
        d = draft_chain(last, pos0, g_mtp_draft_n);
        used_ext_draft = !d.empty();
    }
    if (!used_ext_draft && !suppress_head) {
        const int head_cap = (int) std::min<int64_t>(g_mtp_draft_n, win_room - 1);   // MTP budget
        // The draft head runs ggml_mul_mat_id on its MoE experts (up/gate/down). Under fused-infer
        // those resolve through the expert pool, which is keyed on active_user_id, so it must be set
        // here as on every other decode path. With resident weights it does not matter; offloaded,
        // the head would read another slot's partition and produce garbage drafts that still look
        // accepted.
        ggml_cuda_set_expert_user(slot_id);
        for (int i = 0; i < head_cap; i++) {
            const llama_token e = d.empty() ? last : d.back();
            const llama_token di = llama_mtp_draft(ctx_, e);
            if (di == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, di)) break;
            d.push_back(di);
            // ALPHA_MTP_CONF_TAU: cumulative-confidence verify width. Verify draft j only while
            // prod(p_1..p_j) >= tau, p = the head's softmax probability of its draft (roughly
            // P(accept | p) ~= p). A column costs about tau * (time per token) to verify, so a draft
            // whose chance of surviving is below that is cheaper to drop than to check.
            if (g_mtp_conf_tau > 0.0f) {
                static thread_local float cum = 1.0f;
                if (i == 0) { cum = 1.0f; g_conf_win.clear(); }
                const float pc = llama_mtp_draft_conf(ctx_);
                g_conf_win.push_back(pc);
                cum *= pc;
                // ALPHA_MTP_CONF_MIN (default 1): the gate never cuts below this many drafts. A window
                // left with no draft ends the MTP call (a plain decode plus the gap between calls),
                // which costs more than verifying one low-confidence draft.
                static const int conf_min = [] { const char * e = getenv("ALPHA_MTP_CONF_MIN"); return e ? std::max(0, atoi(e)) : 1; }();
                if (cum < g_mtp_conf_tau && (int) d.size() > conf_min) { d.pop_back(); break; }
            }
            // ALPHA_MTP_MARGIN: margin-gated adaptive depth. Stop the chain early rather than pick m
            // up front, since predictability drifts within one generation and a fixed m fits a
            // moving target. Gate the draft, not the width: an extra verify column is close to free,
            // while each draft step has a real cost.
            //
            // Checked after pushing step i's token, so a confident step is never wasted, and it costs
            // nothing when it does not fire. tau <= 0 disables (fixed m).
            if (g_mtp_margin_tau > 0.0f && i + 1 < head_cap) {
                const float margin = llama_mtp_draft_margin(ctx_);
                if (margin < g_mtp_margin_tau) {
                    if (g_mtp_margin_dbg) {
                        static int64_t n_cut = 0, sum_len = 0, n_win = 0;
                        n_cut++; sum_len += (int64_t) d.size(); n_win++;
                        if ((n_win & 63) == 0)
                            fprintf(stderr, "[mtp-margin] cuts=%lld mean_chain=%.2f (m_max=%d tau=%.3f)\n",
                                    (long long) n_cut, (double) sum_len / (double) n_win,
                                    g_mtp_draft_n, (double) g_mtp_margin_tau);
                    }
                    break;
                }
            }
        }
    }
    if (d.empty()) break;    // no drafter and head not resident -> emit what we have
    const auto mtp_t1 = std::chrono::steady_clock::now();   // drafts done
    ws_m[2] = ws_m[3] = ws_m[4] = ws_m[5] = mtp_t1;

    // 2. Snapshot the pre-window recurrent state (through pos0-1); hybrid-safe (no-op on non-hybrid).
    // ALPHA_MTP_NO_SNAP=1 (needs ALPHA_REPLAYSSM): the armed verify never publishes recurrent state, so
    // the committed state is the pre-window state and a rewind only has to reset the cells' positions.
    static const bool no_snap = g_replayssm && alpha_env_on("ALPHA_MTP_NO_SNAP");
    if (!no_snap && !alpha_recurrent_snapshot(ctx_, snap_)) {
        g_mtp_snap_dead = true;
        fprintf(stderr, "[mtp] DISABLED: could not allocate the recurrent-state mirror (%.1f MB) that verify "
                        "rollback needs -- no VRAM left after weights + KV + compute. Decoding plain from here. "
                        "Free VRAM (smaller -c, fewer -ngl, Q4_0 KV) to run MTP.\n",
                llama_model_rs_size_bytes(llama_get_model(ctx_), llama_n_seq_max(ctx_)) / (1024.0 * 1024.0));
        break;
    }
    const auto mtp_ts = std::chrono::steady_clock::now();   // snapshot done
    auto rewind = [&]() {
        if (no_snap) {
            alpha_recurrent_mark_pos(ctx_, seq, pos0);   // state untouched (publish suppressed): positions only
        } else {
            alpha_recurrent_restore(ctx_, snap_, pos0);
        }
        llama_memory_seq_rm(mem, seq, pos0, -1);
    };
    // Prefer the backend's GPU argmax (4 bytes) over pulling a full vocab row per verify column.
    // Falls back to the host scan whenever no backend sampler is installed, so behaviour is
    // unchanged with ALPHA_BACKEND_ARGMAX unset.
    auto argmax_ith = [&](int i) -> llama_token {
        // First choice: the in-graph argmax. One int per verify row instead of a full logits D2H and
        // host scan per row, and it also removes a ctx->synchronize() from the accept path.
        const llama_token am = llama_mtp_argmax_ith(ctx_, i);
        // ALPHA_MTP_ARGMAX_CHECK: compare the in-graph argmax with the host scan token for token.
        static const bool am_check = alpha_env_on("ALPHA_MTP_ARGMAX_CHECK");
        if (am_check && am != LLAMA_TOKEN_NULL) {
            const float * lgc = llama_get_logits_ith(ctx_, i);
            if (lgc) {
                int hb = 0; float hv = lgc[0];
                for (int v = 1; v < n_vocab; v++) if (lgc[v] > hv) { hv = lgc[v]; hb = v; }
                static int64_t n_cmp = 0, n_bad = 0;
                n_cmp++;
                if ((llama_token) hb != am) {
                    n_bad++;
                    if (n_bad <= 5)
                        fprintf(stderr, "[argmax-check] MISMATCH row=%d graph=%d host=%d\n",
                                i, (int) am, hb);
                }
                if ((n_cmp % 200) == 0)
                    fprintf(stderr, "[argmax-check] %lld compared, %lld mismatches\n",
                            (long long) n_cmp, (long long) n_bad);
            }
        }
        if (am != LLAMA_TOKEN_NULL) return am;
        // Second choice: the backend sampler. It only yields one token per sequence
        // (copy_tensor_async_ints maps seq_id to a single output row) while the verify has k+1 rows
        // on one sequence, so rows >= 1 return NULL and the window degrades to plain greedy. Kept
        // below the in-graph path because it is still correct for the 1-row case.
        const llama_token bt = llama_get_sampled_token_ith(ctx_, i);
        if (bt != LLAMA_TOKEN_NULL) return bt;
        const float * lg = llama_get_logits_ith(ctx_, i);
        if (!lg) return LLAMA_TOKEN_NULL;
        int best = 0; float bv = lg[0];
        for (int v = 1; v < n_vocab; v++) if (lg[v] > bv) { bv = lg[v]; best = v; }
        return (llama_token) best;
    };

    // Sampling (temperature > 0): accept by sampled match rather than rejection sampling. Reproduce
    // the seed a non-speculative decode would have used at each row and accept on
    // "draft == the sampled token". Acceptance probability is p(d) either way, and the output is the
    // exact token sequence a plain sampled decode with the same seed produces, which is possible
    // because the seed schedule is deterministic.
    //
    // The schedule is keyed on the slot's generation counter, not the position. Every emitted token
    // advances it by exactly one, including each of this window's extras, so with
    // N = n_generated at batch-build time the trigger is index N and verify row i is index N+1+i:
    //     seed(row i) = rng_base ^ GOLDEN * (N + 2 + i)
    const bool mtp_do_sample = !req.samp_temp.empty() && req.samp_temp[0] > 0.0f
                            && !req.samp_rng_base.empty() && !req.samp_n_gen.empty();
    const uint64_t mtp_rng_base = mtp_do_sample ? req.samp_rng_base[0] : 0ull;
    const int32_t  mtp_n_gen0   = mtp_do_sample ? req.samp_n_gen[0]    : 0;
    auto pick_ith = [&](int i) -> llama_token {
        if (!mtp_do_sample) return argmax_ith(i);
        const float * lg = llama_get_logits_ith(ctx_, i);
        if (!lg) return LLAMA_TOKEN_NULL;
        const uint64_t seed = mtp_rng_base
                            ^ (0x9E3779B97F4A7C15ull * (uint64_t) (mtp_n_gen0 + 2 + i));
        if (getenv("ALPHA_SEED_TRACE"))
            fprintf(stderr, "[seedtrace] mtp   pos=%d seed=%llu\n",
                    (int) (pos0 + 1 + i), (unsigned long long) seed);
        return alpha_sample_token(lg, n_vocab, req.samp_temp[0],
                                  req.samp_top_k[0], req.samp_top_p[0], seed);
    };

    // 3. Verify [last@pos0, d[0]@pos0+1, ... d[k-1]@pos0+k] in one batched decode.
    const int k = (int) d.size();
    common_batch_clear(batch_);
    common_batch_add(batch_, last, pos0, { seq }, true);
    for (int i = 0; i < k; i++) common_batch_add(batch_, d[i], pos0 + 1 + i, { seq }, true);
    ggml_cuda_set_expert_user(slot_id);
    // ReplaySSM: arm the recording for this decode only. From here the recurrent state publish is
    // suppressed, so the window must end in a fold or a relay.
    if (g_replayssm) llama_gdn_record_arm(ctx_, k + 1);
    // A greedy verify reads no logits (every row's token comes from the in-graph argmax), so skip
    // the full-vocab D2H. Scoped to this decode only: the trigger decode still needs logits, and a
    // sampled request needs them for the sampled-match accept.
    static const bool am_check_env = alpha_env_on("ALPHA_MTP_ARGMAX_CHECK");
    const bool skip_logits_here = !mtp_do_sample && !am_check_env;   // the check reads logits
    if (skip_logits_here) llama_set_skip_logits(ctx_, true);
    const auto vdec_t0 = std::chrono::steady_clock::now();
    const int vrc_dec = alpha_counted_decode(ctx_, batch_);
    if (getenv("ALPHA_MTP_DBG")) {
        static double us = 0, us_pre = 0; static int64_t n = 0;
        us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - vdec_t0).count();
        us_pre += std::chrono::duration<double, std::micro>(vdec_t0 - mtp_ts).count();
        if ((++n & 63) == 0) fprintf(stderr, "[mtp-vdec] llama_decode %.2f ms, pre-decode %.2f ms (verify)\n", us / n / 1e3, us_pre / n / 1e3);
    }
    if (skip_logits_here) llama_set_skip_logits(ctx_, false);
    if (vrc_dec != 0) { rewind(); mtp_failed = true; break; }
    const auto mtp_t2 = std::chrono::steady_clock::now();   // verify done
    ws_m[3] = ws_m[4] = ws_m[5] = mtp_t2;

    // 4. Accept the longest prefix whose draft matches the model's greedy token. argmax_ith(i) is
    //    the true token at pos0+1+i given all inputs up to i are real, so the comparison is only
    //    valid while every earlier draft matched; hence a prefix, not a per-slot filter.
    int n_acc = 0;
    while (n_acc < k) {
        const llama_token ti = pick_ith(n_acc);
        if (ti == LLAMA_TOKEN_NULL || ti != d[n_acc]) break;
        n_acc++;
    }
    // Truth at the first mismatch (or the bonus).
    const llama_token t_next = pick_ith(n_acc);

    // ALPHA_MTP_ACC_DBG: print the accept decision for the first windows (draft, per-column verify
    // argmax, n_acc). Column 0 is a real token whose argmax cannot depend on any draft, so a
    // disagreement there points at the verify decode itself.
    if (getenv("ALPHA_MTP_ACC_DBG")) {
        static int n_ad = 0;
        if (n_ad < 6) {
            n_ad++;
            fprintf(stderr, "[mtp-acc] w%d pos0=%d k=%d last=%d | draft=", n_ad, (int) pos0, k, (int) last);
            for (int i = 0; i < k; i++) fprintf(stderr, "%d%s", (int) d[i], i + 1 < k ? "," : "");
            fprintf(stderr, " | verify_argmax=");
            for (int i = 0; i <= k; i++) fprintf(stderr, "%d%s", (int) pick_ith(i), i < k ? "," : "");
            fprintf(stderr, " | n_acc=%d t_next=%d\n", n_acc, (int) t_next);
        }
    }

    // No explicit head commit: the head's KV is indexed by absolute position and refilled from every
    // decode's own rows (llama_context::mtp_kv_fill), so the verify decode above already laid down
    // real columns for the accepted prefix, and the rewind path below re-lays the rest. Columns past
    // the accepted prefix are never read; the next window's draft sits at the first of them and only
    // attends to what precedes it.
    if (t_next == LLAMA_TOKEN_NULL) { rewind(); return; }

    mtp_steps++; mtp_accept += n_acc; mtp_drafted += k; mtp_accept_full += (n_acc == k) ? 1 : 0;
    if (g_mtp_ngram_adapt && slot_id >= 0) {
        if ((int) ngram_adapt_.size() <= slot_id) ngram_adapt_.resize(slot_id + 1);
        auto & A = ngram_adapt_[slot_id];
        const double us_v = std::chrono::duration<double,std::micro>(mtp_t2 - mtp_ts).count();
        const double us_s = std::chrono::duration<double,std::micro>(mtp_ts - mtp_t1).count();
        const double us_d = std::chrono::duration<double,std::micro>(mtp_t1 - mtp_t0).count();
        const double w    = (double) (k + 1);
        const int wi = (int) (w + 0.5);                       // observed verify cost at this width
        if (wi >= 0 && wi < 33) { A.vtab_sum[wi] += us_v; A.vtab_cnt[wi] += 1; }
        A.snap_us += us_s; A.snap_n += 1;
        if (!used_ext_draft && k > 0) {                 // a head window: its cost includes drafting
            A.draft_us += us_d; A.draft_steps += k;
            const double cost = us_d + us_s + us_v;
            if (cost > 1.0) {
                if (g_mtp_ngram_decay < 1.0f) {              // the head baseline must forget too,
                    A.head_sum *= g_mtp_ngram_decay;         // or the comparison drifts one-sided
                    A.head_cnt *= g_mtp_ngram_decay;
                }
                A.head_sum += ((double) n_acc + 1.0) / cost; A.head_cnt += 1;
            }
        }
        if (getenv("ALPHA_MTP_DBG") && ((A.routed + A.declined) % 64) == 0 && (A.routed + A.declined) > 0) {
            const int bd = ngram_adapt_pick(slot_id, 32);
            // fit coefficients and the scored prefix decide best_d; if the prefix is ~0 the depths
            // all tie at 1/cost(d+1), which decreases in d, so best_d=1.

            fprintf(stderr, "[ngram-adapt] routed=%ld declined=%ld (%.0f%%) best_d=%d head_yld=%.3f tok/ms\n"
                            "              prefix mean=%.2f zero=%.0f%% (n=%ld) | V(2)=%.0f V(9)=%.0f V(17)=%.0fus | yld1=%.3f yld8=%.3f yld16=%.3f\n",
                    A.routed, A.declined,
                    100.0 * (double) A.routed / (double) (A.routed + A.declined ? A.routed + A.declined : 1),
                    bd, A.head_cnt ? 1000.0 * A.head_sum / (double) A.head_cnt : 0.0,   // us -> ms
                    A.prefix_n ? A.prefix_sum / (double) A.prefix_n : 0.0,
                    A.prefix_n ? 100.0 * (double) A.prefix_zero / (double) A.prefix_n : 0.0,
                    A.prefix_n, ngram_verify_us(A, 2.0), ngram_verify_us(A, 9.0),
                    ngram_verify_us(A, 17.0),
                    A.yld_cnt[1]  ? 1000.0 * A.yld_sum[1]  / (double) A.yld_cnt[1]  : 0.0,
                    A.yld_cnt[8]  ? 1000.0 * A.yld_sum[8]  / (double) A.yld_cnt[8]  : 0.0,
                    A.yld_cnt[16] ? 1000.0 * A.yld_sum[16] / (double) A.yld_cnt[16] : 0.0);
            char pb[512]; int po = 0;
            for (int b = 0; b < 10; ++b)
                po += snprintf(pb + po, sizeof(pb) - po, " p%d=%.2f(%ld)", b * 64,
                               A.posb_cnt[b] ? A.posb_sum[b] / (double) A.posb_cnt[b] : 0.0,
                               A.posb_cnt[b]);
            fprintf(stderr, "[ngram-pos] prefix by ABSOLUTE token position:%s\n", pb);
        }
    }
    if (g_mtp_ngram > 0) {
        static int64_t ng_win = 0, ng_hit = 0, ng_tok = 0, hd_win = 0, hd_tok = 0;
        ng_win++;
        static int64_t ng_len = 0, ng_short = 0;
        if (ngram_matched >= g_mtp_ngram_min) {
            ng_hit++; ng_tok += n_acc;
            // chain width drives verify cost; if routed chains are mostly short, padding them out
            // to draft_n only adds reject columns.
            ng_len += k; ng_short += (k < g_mtp_draft_n) ? 1 : 0;
        } else { hd_win++; hd_tok += n_acc; }
        if (getenv("ALPHA_MTP_DBG") && (ng_win % 32) == 0) {
            fprintf(stderr, "[ngram] windows=%lld match%d=%.3f | ngram-route %lld win %.2f acc/win chain %.2f wide %.0f%% short | head-route %lld win %.2f acc/win\n",
                    (long long) ng_win, g_mtp_ngram_min, (double) ng_hit / (double) ng_win,
                    (long long) ng_hit, ng_hit ? (double) ng_tok / (double) ng_hit : 0.0,
                    ng_hit ? (double) ng_len / (double) ng_hit : 0.0,
                    ng_hit ? 100.0 * (double) ng_short / (double) ng_hit : 0.0,
                    (long long) hd_win, hd_win ? (double) hd_tok / (double) hd_win : 0.0);
        }
    }
    // Per-position conditional acceptance: position i only counts as attempted while every earlier
    // draft matched, which is the quantity that decays with depth. The aggregate rate above blends
    // a healthy position 0 with a collapsing tail.
    static int64_t mtp_try[32] = {0}, mtp_ok[32] = {0};
    for (int i = 0; i < k && i < 32; i++) { mtp_try[i]++; if (i < n_acc) mtp_ok[i]++; else break; }
    // d1 bucketed by position in the chain. Window 0 of a chain starts from a fresh trigger decode's
    // hidden (batch of 1); windows >= 1 start from the previous window's verify row (batch of k+1).
    // If the hand-off costs d1, bucket 0 stays high and later buckets sit lower; if the loss
    // compounds along a chain, the buckets decline monotonically.
    {
        static int64_t chain_try[8] = {0}, chain_ok[8] = {0};
        const int wslot = mtp_w < 8 ? mtp_w : 7;
        chain_try[wslot]++;
        if (n_acc > 0) chain_ok[wslot]++;
        if (getenv("ALPHA_MTP_DBG") && (mtp_steps % 128) == 0) {
            char cb2[256]; int co = 0;
            for (int z = 0; z < 8 && chain_try[z] > 0; z++)
                co += snprintf(cb2 + co, sizeof(cb2) - co, " w%d=%.1f%%(%lld)", z,
                               100.0 * (double) chain_ok[z] / (double) chain_try[z],
                               (long long) chain_try[z]);
            fprintf(stderr, "[mtp-chain] d1 by chain position:%s\n", cb2);
        }
    }
    // The accumulation is unconditional (the means are over all windows); only the print is gated.
    // These timers cover mtp_t0..mtp_t2 only (draft -> snapshot -> verify). The post-verify tail
    // (the ReplaySSM fold, or the rewind + relay every non-full-accept window pays when
    // ALPHA_REPLAYSSM is off, plus the external drafter's resync) runs after mtp_t2 and is reported
    // by the [replayssm] and [mtp-rej] lines under ALPHA_MTP_DBG.
    {
        static double us_draft = 0, us_verify = 0;
        static double us_snap = 0;
        us_draft  += std::chrono::duration<double,std::micro>(mtp_t1 - mtp_t0).count();
        us_snap   += std::chrono::duration<double,std::micro>(mtp_ts - mtp_t1).count();
        us_verify += std::chrono::duration<double,std::micro>(mtp_t2 - mtp_ts).count();
        if (getenv("ALPHA_MTP_DBG") && (mtp_steps & 31) == 0)
            fprintf(stderr, "[mtp-time] draft %.2f  SNAPSHOT %.2f  verify %.2f  ms/window"
                            " (plain decode ~24 ms; EXCLUDES the post-verify tail - see the"
                            " mtp-rej and replayssm lines)\n",
                    us_draft / 1000.0 / (double) mtp_steps, us_snap / 1000.0 / (double) mtp_steps,
                    us_verify / 1000.0 / (double) mtp_steps);
    }
    if (getenv("ALPHA_MTP_DBG") && (mtp_steps & 31) == 0) {
        fprintf(stderr, "[mtp-decodes] %lld windows, %lld llama_decode calls = %.2f forwards/window"
                        " (verify=1, relay=%.2f, remainder is the TRIGGER decode)%c",
                (long long) mtp_steps, (long long) g_alpha_decodes,
                (double) g_alpha_decodes / (double) mtp_steps,
                (double) (mtp_steps - mtp_accept_full) / (double) mtp_steps, 10);
        // full% = share of windows where every draft was accepted. It predicts whether
        // ALPHA_MTP_LOOP pays: only a fully accepted window hands over a carried hidden rather than
        // a repaired one, so low full% is where chaining can hurt.
        fprintf(stderr, "[mtp-dbg] steps=%lld drafted=%lld accepted=%lld (%.1f%%) tokens/window=%.2f full=%.1f%%\n",
                (long long) mtp_steps, (long long) mtp_drafted, (long long) mtp_accept,
                100.0 * (double) mtp_accept / (double) (mtp_drafted ? mtp_drafted : 1),
                1.0 + (double) mtp_accept / (double) mtp_steps,
                100.0 * (double) mtp_accept_full / (double) (mtp_steps ? mtp_steps : 1));
        char pb[640]; int off = 0;
        for (int i = 0; i < 32 && mtp_try[i] > 0; i++)
            off += snprintf(pb + off, sizeof(pb) - off, " d%d=%.1f%%(%lld)", i + 1,
                            100.0 * (double) mtp_ok[i] / (double) mtp_try[i], (long long) mtp_try[i]);
        fprintf(stderr, "[mtp-pos]%s\n" , pb);
    }

    // ReplaySSM: decide how the window's state is made exact through pos0+n_acc.
    //   - not armed        : relay only on a partial accept
    //   - armed + folded   : recurrent state advanced by re-invoking the gated-delta op over just
    //                        the accepted rows; the rejected attention KV columns are dropped with
    //                        a seq_rm. No rewind, no re-decode.
    //   - armed + refused  : the suppressed publish left the state a whole window stale, so the
    //                        relay is mandatory, including on a full accept.
    // ALPHA_REPLAYSSM_FULLONLY=1 folds only on a full accept, where the fold must reproduce exactly
    // what the suppressed publish would have written (isolates the m < n slicing from the
    // re-invocation itself). Partial accepts fall through to the relay.
    static const bool fullonly = alpha_env_on("ALPHA_REPLAYSSM_FULLONLY");
    bool need_relay = (n_acc != k);
    if (g_replayssm && !(fullonly && n_acc != k)) {
        const auto fold_t0 = std::chrono::steady_clock::now();
        ws_m[4] = fold_t0;
        const bool folded  = llama_gdn_fold(ctx_, n_acc + 1);
        ws_m[5] = std::chrono::steady_clock::now();
        if (folded) {
            need_relay = false;
            if (n_acc != k) {
                // Drop the rejected tail's attention KV. Order matters: the recurrent half refuses a
                // partial tail removal and the hybrid wrapper returns on that refusal without
                // touching the attention cache, so a bare seq_rm would be a silent no-op. The fold
                // has already advanced the recurrent state to pos0+n_acc, so move the recurrent
                // cells' bookkeeping to match, which makes the removal legal.
                alpha_recurrent_mark_pos(ctx_, seq, pos0 + n_acc + 1);
                if (!llama_memory_seq_rm(mem, seq, pos0 + n_acc + 1, -1)) {
                    // Should not happen after the rewind; fall back rather than run on stale KV.
                    fprintf(stderr, "[replayssm] seq_rm REFUSED after fold -- falling back to relay\n");
                    need_relay = true;
                }
            }
        } else {
            need_relay = true;
        }
        static double us_fold = 0; static int64_t n_fold = 0, n_refuse = 0;
        n_fold++; if (!folded) n_refuse++;
        us_fold += std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now() - fold_t0).count();
        if (getenv("ALPHA_MTP_DBG") && (n_fold & 31) == 0)
            fprintf(stderr, "[replayssm] %.3f ms/fold over %lld windows, %lld refused (-> relay)\n",
                    us_fold / 1000.0 / (double) n_fold, (long long) n_fold, (long long) n_refuse);
    }

    if (!need_relay) {
        // Either every draft matched (KV + recurrent already exact through pos0+k), or the fold
        // made the recurrent half exact and the seq_rm above dropped the rejected KV tail.
    } else {
        // Partial/no accept: positions after n_acc were fed wrong tokens, so rewind and re-lay the
        // real prefix to make the state exact through pos0+n_acc, then emit prefix + truth token.
        const auto mtp_tr0 = std::chrono::steady_clock::now();
        rewind();
        common_batch_clear(batch_);
        common_batch_add(batch_, last, pos0, { seq }, false);
        for (int i = 0; i < n_acc; i++) common_batch_add(batch_, d[i], pos0 + 1 + i, { seq }, false);
        ggml_cuda_set_expert_user(slot_id);
        if (alpha_counted_decode(ctx_, batch_) != 0) { rewind(); stop_chain = true; break; }
        { static double us_rej = 0; static int64_t n_rej = 0; n_rej++;
          us_rej += std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now() - mtp_tr0).count();
          if (getenv("ALPHA_MTP_DBG") && (n_rej & 3) == 0)
              fprintf(stderr, "[mtp-rej] rewind+relay %.2f ms/reject  (%lld rejects)\n",
                      us_rej / 1000.0 / (double) n_rej, (long long) n_rej); }
    }

    // Emit this window in full: the accepted drafts and t_next. t_next is also the next window's
    // anchor, so it must be emitted here rather than at the end of the chain. It is not double
    // counted: window w+1 consumes t_next as its anchor and emits only its own drafts and t_next.
    for (int i = 0; i < n_acc; i++) {
        result.extra_tokens.push_back(d[i]);
        if (g_mtp_ngram > 0) ngram_commit(slot_id, &d[i], 1);
        if (llama_vocab_is_eog(vocab, d[i])) stop_chain = true;
    }
    result.extra_tokens.push_back(t_next);
    // t_next goes into the n-gram ring too (the drafter's `committed` vector below includes it as
    // well); otherwise the lookup history would be missing one token per window and the adaptive
    // scorer would misalign.
    if (g_mtp_ngram > 0) ngram_commit(slot_id, &t_next, 1);

    // ALPHA_DRAFT_MODEL: re-sync the drafter to what was actually committed. The drafter advanced
    // through pos0+m while only n_acc+1 tokens were taken, so without the rollback its KV and
    // recurrent state hold rejected drafts, and the next window's catch-up loop would find dpos_
    // already past the target and draft from a diverged context. seq_rm alone does not undo
    // recurrent state, hence the snapshot/restore pair.
    if (used_ext_draft) {
        std::vector<llama_token> committed;
        committed.reserve(n_acc + 1);
        for (int i = 0; i < n_acc; i++) committed.push_back(d[i]);
        committed.push_back(t_next);
        draft_commit(committed, pos0);
    }

    // ALPHA_MTP_CONF_LOG=1: one line per window for an offline gate replay: "C p_1 .. p_m | M n_acc k"
    {
        static const bool conf_log = alpha_env_on("ALPHA_MTP_CONF_LOG");
        if (conf_log && g_mtp_conf_tau > 0.0f) {
            std::string ln = "[mtp-conf] C";
            char b[96];
            for (float pc : g_conf_win) { snprintf(b, sizeof(b), " %.4f", pc); ln += b; }
            snprintf(b, sizeof(b), " | M %d %d | V %.2f D %.2f\n", n_acc, k,
                     std::chrono::duration<double, std::milli>(ws_m[3] - ws_m[2]).count(),
                     std::chrono::duration<double, std::milli>(ws_m[2] - ws_m[1]).count());
            ln += b;
            fputs(ln.c_str(), stderr);
        }
    }
    // ALPHA_MTP_WIN_STATS=N: per-window acceptance every N windows (the [mtp] line below counts calls, and a
    // call runs up to ALPHA_MTP_LOOP windows): drafted, accepted and tokens per window over the last N windows.
    {
        static const int every = []() { const char * e = getenv("ALPHA_MTP_WIN_STATS"); return e ? atoi(e) : 0; }();
        if (every > 0) {
            ws_m[6] = std::chrono::steady_clock::now();
            for (int i = 0; i < 6; i++) ws_ph[i] += std::chrono::duration<double, std::milli>(ws_m[i + 1] - ws_m[i]).count();
            static long long w_n = 0, w_drafted = 0, w_acc = 0, w_hist[9] = {0};
            w_n++; w_drafted += k; w_acc += n_acc; w_hist[std::min(8, std::max(0, n_acc))]++;
            if (w_n % every == 0) {
                fprintf(stderr, "[mtp-win] last %d windows: %.2f drafted, %.2f accepted (%.0f%%), %.2f tokens/window | n_acc hist:",
                        every, (double) w_drafted / every, (double) w_acc / every,
                        w_drafted ? 100.0 * (double) w_acc / (double) w_drafted : 0.0, 1.0 + (double) w_acc / every);
                for (int i = 0; i <= 8 && i <= k + 1; i++) fprintf(stderr, " %d:%lld", i, w_hist[i]);
                fprintf(stderr, "\n");
                fprintf(stderr, "[mtp-win] phases ms/window: pre %.2f | draft %.2f | snap+verify %.2f | accept %.2f | fold %.2f | post %.2f\n",
                        ws_ph[0] / every, ws_ph[1] / every, ws_ph[2] / every, ws_ph[3] / every, ws_ph[4] / every, ws_ph[5] / every);
                fprintf(stderr, "[mtp-win] top-to-top %.2f ms/window (%lld chained), pre-loop %.2f ms/call, slow(>35ms) %lld windows %.1f ms\n",
                        ws_iter_ms / std::max(1LL, ws_iters), ws_iters, ws_pre_loop_ms / std::max(1LL, ws_calls), ws_slow_n, ws_slow_ms);
                ws_slow_n = 0; ws_slow_ms = 0;
                for (auto & v : ws_ph) v = 0;
                ws_iter_ms = 0; ws_iters = 0;   // same window of windows as the phases
                w_drafted = w_acc = 0; for (auto & h : w_hist) h = 0;
            }
        }
    }
    // Conservative chaining: only continue out of a fully-verified window.
    if (g_mtp_loop_fullonly && n_acc != k) stop_chain = true;
    if (stop_chain) break;

    // Point the carried hidden at the last accepted row. The decode path keeps the last drafted row,
    // which on a partial accept is conditioned on rejected tokens. After a relay, row n_acc is its
    // last row, so this is correct in both paths.
    llama_mtp_select_row(ctx_, n_acc);

    // Hand over: the decode just run (verify on full accept, relay otherwise) ended at pos0+n_acc,
    // so mtp_hidden_host now holds H(pos0+n_acc) and mtp_pos_base == pos0+n_acc, which is the state
    // the next window's draft expects.
    last = t_next;
    pos0 = pos0 + n_acc + 1;
    if (llama_vocab_is_eog(vocab, last)) break;
  }
    // No trailing push: the last t_next was already emitted by its own window, and it is the
    // token the caller will decode next.
    // ALPHA_MTP_TRIGGERLESS: or the next call runs as a window, if nothing breaks the chain in between
    g_mtp_tl.ok = !mtp_failed && !stop_chain && last != LLAMA_TOKEN_NULL && !llama_vocab_is_eog(vocab, last);
    g_mtp_tl.slot = slot_id; g_mtp_tl.seq = seq; g_mtp_tl.tok = last; g_mtp_tl.pos = pos0;

    // mtp_steps > 0 guard: on a model with no MTP head no window ever runs, and the summary would
    // otherwise print a NaN rate every token.
    if (mtp_steps > 0 && (mtp_steps & 15) == 0) {
        // mtp_steps counts windows and mtp_accept counts accepted draft tokens, so a window emits
        // 1 base token + n_acc drafted ones: tokens/window = 1 + accept/steps. The acceptance
        // fraction divides by what was drafted. Both denominators are printed.
        const double toks   = 1.0 + (double) mtp_accept / (double) mtp_steps;
        const double frac   = mtp_drafted > 0 ? (double) mtp_accept / (double) mtp_drafted : 0.0;
        const double per_win = (double) mtp_accept / (double) mtp_steps;
        fprintf(stderr, "[mtp] %lld windows, accepted %lld/%lld drafted (%.2f accepted/window, %.0f%% of drafts), ~%.2f tokens/window\n",
                (long long) mtp_steps, (long long) mtp_accept, (long long) mtp_drafted,
                per_win, 100.0*frac, toks);
    }
}

void gpu_engine::loop() {
    fprintf(stderr, "[alphaserve] GPU engine thread started\n");

    while (running_.load()) {
        BatchRequest req;
        if (!in_->try_pop(req)) {
            std::this_thread::yield();
            continue;
        }

        BatchResult result;
        if (decode_batch(req, result)) {
            if (g_mtp && !result.decode_error && result.n_slots == 1 &&
                req.n_tokens == 1 && req.logits_mask[0] && req.prompt_end == 0 && result.sampled.size() == 1) {
                mtp_window_extend(req, result);
            }
            out_->try_push(std::move(result));
        } else {
            out_->try_push(make_error_result(req));
        }
    }

    fprintf(stderr, "[alphaserve] GPU engine thread stopped (%lld decodes, %lld tokens)\n",
            (long long)n_decodes_, (long long)n_tokens_);
}

} // namespace alphaserve
