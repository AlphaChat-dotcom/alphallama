// SPDX-License-Identifier: Apache-2.0
// MTP self-speculative decoding.
//
// One decode step = one round (B streams, L = k_in + 1 rows each, stream-major):
//   verify   [t, d1..d_kin] at positions p..p+k_in; GDN layers record instead of publishing,
//            KV cells are written, the full head gives targets g on the device;
//   accept   m[b] = longest prefix with d == g, and a column mask (device kernel);
//   fold     all L columns folded with beta/decay neutralised past m[b] (state unchanged there),
//            conv history gathered at m[b]+1 (device kernel);
//   head     fill the head's KV at p+1..p+L from (g, h); row m[b] gives draft 1 and its hidden;
//   chain    k_out-1 head steps at p+m[b]+2.. with positions/cells/masks computed on the device;
//   egress   g, m, drafts -> one sync.
// Head rows past m[b] write KV cells that the chain (and the next step's fill) overwrite before
// any row can see them; verify rows past m[b] are overwritten by the next verify the same way.
// Profiles are keyed {F, B, k_in, k_out}: acceptance is data.
//
// Adaptive k (--mtp-adapt): per-depth conditional acceptance a_i (EMA over streams) and measured
// step time per (k_in, k_out); each step picks k_out in {k-1, k, k+1} maximising
// E[tokens/step] / time, with E = 1 + sum_j prod_{i<=j} a_i.
#include <nvtx3/nvToolsExt.h>   // AI_MTP_PROF phases as NVTX ranges (header-only)
#include "engine/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ai {

static Tensor last_row(const Tensor & h) {    // f32 [E, T] -> [E, 1]
    const int64_t E = h.ne[0], T = h.ne[1];
    return h.view(E, 1, 1, 1, h.nb[1], h.nb[1], h.nb[1], h.nb[1] * (size_t) (T - 1));
}

// AI_MTP_CONF_LOG=<file>: confidence-gate study (uncaptured AI_MTP_PROF steps only). For each step, one line
// "C c1 .. ck" = max-softmax of each draft over the draft head's rows, then the engine's round loop writes the
// next step's "M m" (accepted count), so offline the drafts of line C_i pair with M_{i+1}.
FILE * mtp_conf_file() {
    static FILE * f = []() -> FILE * {
        const char * p = getenv("AI_MTP_CONF_LOG");
        return p ? fopen(p, "w") : nullptr;
    }();
    return f;
}

// AI_DRAFT_TRACE=<file>: input for an offline drafter simulator. Exact MTP's output does not depend on the drafts
// and its draft at position p depends only on the committed prefix, so one trace with the verify width forced to 0
// (AI_MTP_GATE > 1) gives MTP's drafts at every position, against which any policy
// (prompt lookup, hybrid, any verify width) is replayed exactly. Lines:
//   "Q <stream> <n> t1 .. tn"                     the prompt token ids
//   "F <stream> <pos> <tok> | d1 .. dk"            the first generated token (from the prefill) and its drafts
//   "R <stream> <pos0> <m> g0 .. gm | d1 .. dk"    one round: the committed tokens from pos0 on, then the next drafts
FILE * draft_trace_file() {
    static FILE * f = []() -> FILE * {
        const char * p = getenv("AI_DRAFT_TRACE");
        return p ? fopen(p, "w") : nullptr;
    }();
    return f;
}

void Engine::mtp_conf_log(const std::vector<Tensor> & logits) {
    FILE * f = mtp_conf_file();
    if (f == nullptr) {
        return;
    }
    dev_.synchronize();
    fprintf(f, "C");
    std::vector<float> v;
    for (const Tensor & t : logits) {
        const int64_t V = t.ne[0];
        v.resize((size_t) V);
        dev_.download(t, v.data(), 0, (size_t) V * sizeof(float));   // row 0 (batch 1)
        float mx = v[0];
        for (float x : v) mx = std::max(mx, x);
        double z = 0.0;
        for (float x : v) z += std::exp((double) x - mx);
        fprintf(f, " %.5f", 1.0 / z);
    }
    fprintf(f, "\n");
    // the device kernel's values for the same drafts (AI_MTP_GATE on): a check of kernels/draft_conf.cu
    // against this double-precision host reference. Line "D d1 .. dk"; offline readers that only know C/M skip it.
    if (!chain_confs_.empty()) {
        fprintf(f, "D");
        for (const Tensor & t : chain_confs_) {
            float c = 0.0f;
            dev_.download(t, &c, 0, sizeof(float));
            fprintf(f, " %.5f", c);
        }
        fprintf(f, "\n");
    }
}

void Engine::chain(std::vector<Tensor> & drafts, Qwen35Hybrid::MtpOut first, const Tensor & base, const Tensor & m,
                   int offset0, int k_out, int64_t B, int64_t n_kv, int first_region, const Tensor & delta) {
    static const bool conf_log = getenv("AI_MTP_CONF_LOG") != nullptr;
    std::vector<Tensor> & lgs = conf_logits_;   // read by profile_mtp after its sync (never during capture)
    lgs.clear();
    if (conf_log) lgs.push_back(first.logits);
    chain_confs_.clear();
    if (first.conf.valid()) chain_confs_.push_back(first.conf);
    const int64_t R  = prog_.kv().region_cells();
    const int64_t fc = prog_.kv().first_cell(first_region);
    drafts.push_back(first.token);
    Tensor tok = first.token;
    Tensor H   = first.hidden;
    for (int s = 1; s < k_out; ++s) {
        Qwen35Hybrid::Inputs in;
        in.T            = B;
        in.B            = B;
        in.n_kv         = n_kv;
        in.first_region = first_region;
        in.ids          = tok;
        in.pos4         = dev_.ws_alloc(I32, 4 * B);
        in.cells        = dev_.ws_alloc(I64, B);
        in.mask         = dev_.ws_alloc(F16, n_kv, 1, 1, B);
        op::mtp_step_inputs(dev_, base, m, B, offset0 + s, R, fc, n_kv, in.pos4, in.cells, in.mask, delta);
        auto o = prog_.mtp(dev_, tok, H, in, Qwen35Hybrid::HeadRows::All);
        tok = o.token;
        H   = o.hidden;
        drafts.push_back(tok);
        if (conf_log) lgs.push_back(o.logits);
        if (o.conf.valid()) chain_confs_.push_back(o.conf);
    }
    (void) B;
}

void Engine::prefill_mtp(int stream, const std::vector<int32_t> & prompt, Stream & st) {
    int64_t off = 0;
    prefill_mtp_range(stream, prompt, off, true, st);
}

// Resumable form of the MTP prefill for AI_PREFILL_CHUNK (see Engine::prefill_range). A non-last chunk's head columns
// read prompt[off + 1 .. off + T], one token into the next chunk, so when !final a chunk runs only once the buffer
// holds at least one token past it -- the condition below -- and the chunk boundaries stay those of a one-shot run.
void Engine::prefill_mtp_range(int stream, const std::vector<int32_t> & prompt, int64_t & off, bool final, Stream & st) {
    const int     k = cfg_.mtp_k;
    const int64_t n = (int64_t) prompt.size();
    for (; off < n; off += cfg_.chunk) {
        const int64_t T    = std::min(cfg_.chunk, n - off);
        if (!final && off + T >= n) {
            break;
        }
        const bool    last = off + T == n;
        dev_.ws_reset(0);
        auto in = stage_chunk(stream, prompt.data() + off, T, off, prog_.visible_cells(off, T));
        mm_attach(in, off, T);
        static char tag1[96];
        snprintf(tag1, sizeof(tag1), "mtp-prefill forward+head T=%lld off=%lld%s", (long long) T, (long long) off,
                 last ? " last" : "");
        dev_.ws_tag(tag1);
        std::unique_ptr<op::Round> r;
        if (!chunk_eager()) {
            r = std::make_unique<op::Round>(dev_);
            r->begin();
        }
        Tensor hidden = prog_.forward(dev_, in, true);
        Tensor am;
        if (!last) {
            // head columns at positions off+1..off+T from (h[off+i], x[off+i+1])
            const int64_t nkv = prog_.visible_cells(off + 1, T);
            stage_rows(head_in_, stream, 0, T, prompt.data() + off + 1, T, off + 1, nkv, true);
            head_in_.upload(dev_, T, 1, nkv);
            auto hin = head_in_.view(T, 1, nkv, stream);
            if (!head_in_.mask_fits(nkv, T)) {
                // AI_HEAD_MASK_DEV: after the forward in round order: the main mask is dead, its buffer takes the head's
                hin.mask = main_in_.mask_view(nkv, T, 1);
                op::head_causal_mask(dev_, hin.mask, T, nkv, off + 1);
            }
            prog_.mtp(dev_, hin.ids, hidden, hin, Qwen35Hybrid::HeadRows::None);
        } else {
            const Tensor lg = prog_.logits(dev_, last_row(hidden));
            am = op::argmax(dev_, lg);
            op::sample_into(dev_, lg, am, 1, stream, nullptr, n, samp_ptr());   // the first token is at position n
        }
        if (r) {
            r->end();
            r->launch();
        }
        if (!last) {
            continue;
        }
        dev_.download_async(am, h_out_, 0, 4);
        dev_.synchronize();
        st.pend = h_out_[0];

        // the last chunk's head columns (now that the pending token is known) and the first drafts
        std::vector<int32_t> nxt(prompt.begin() + off + 1, prompt.end());
        nxt.push_back(st.pend);
        const int64_t F = frontier(n + 2 * k + 2);
        stage_rows(head_in_, stream, 0, T, nxt.data(), T, off + 1, F, true);
        head_in_.h_base[0] = (int32_t) n;   // chain step s sits at n + s
        head_in_.upload(dev_, T, 1, F);
        auto hin = head_in_.view(T, 1, F, stream);
        dev_.ws_tag("mtp-prefill last head+chain");
        std::unique_ptr<op::Round> r2;
        if (!chunk_eager()) {
            r2 = std::make_unique<op::Round>(dev_);
            r2->begin();
        }
        if (!head_in_.mask_fits(F, T)) {
            hin.mask = main_in_.mask_view(F, T, 1);
            op::head_causal_mask(dev_, hin.mask, T, F, off + 1);
        }
        auto o = prog_.mtp(dev_, hin.ids, hidden, hin, Qwen35Hybrid::HeadRows::Last);
        std::vector<Tensor> d;
        chain(d, o, head_in_.base_view(1), head_in_.zeros_view(1), 0, k, 1, F, stream, head_in_.delta_view(1));
        if (r2) {
            r2->end();
            r2->launch();
        }
        for (int s = 0; s < k; ++s) {
            dev_.download_async(d[(size_t) s], h_drafts_ + s, 0, 4);
        }
        dev_.synchronize();
        st.drafts.assign(h_drafts_, h_drafts_ + k);
    }
}

Engine::StepOut Engine::build_mtp(const Qwen35Hybrid::Inputs & vin, const Qwen35Hybrid::Inputs & hin, int k_out) {
    dev_.ws_tag("mtp verify+draft round");
    const int64_t B = vin.B;
    const int64_t L = vin.T / B;
    StepOut out;
    Qwen35Hybrid::GdnRecords rec;
    Tensor hidden = prog_.forward(dev_, vin, true, &rec);
    out.logits = prog_.logits(dev_, hidden);
    out.g      = op::argmax(dev_, out.logits);
    {
        const Tensor base = main_in_.base_view(B);
        op::sample_into(dev_, out.logits, out.g, L, vin.first_region, &base, -1, samp_ptr());
    }
    op::Accept acc = op::mtp_accept(dev_, out.g, vin.ids, L, B);
    out.m = acc.m;
    prog_.fold_masked(dev_, vin, rec, acc.colmask, acc.m);
    auto first = prog_.mtp(dev_, out.g, hidden, hin, Qwen35Hybrid::HeadRows::Accepted, acc.m);
    // chain step s at p + m + 1 + s (p = the verify's first position = main ingress base)
    chain(out.drafts, first, main_in_.base_view(B), acc.m, 1, k_out, B, hin.n_kv, vin.first_region,
          main_in_.delta_view(B));
    out.confs = chain_confs_;
    return out;
}

// One MTP step launched phase by phase, with a device sync after each, so the round's time can be
// attributed. Uncaptured by construction: the numbers are a split, not a round time.
Engine::StepOut Engine::profile_mtp(const Qwen35Hybrid::Inputs & vin, const Qwen35Hybrid::Inputs & hin, int k_out,
                                    Timing & tm) {
    using clk = std::chrono::steady_clock;
    const int64_t B = vin.B;
    const int64_t L = vin.T / B;
    const auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    StepOut out;
    Qwen35Hybrid::GdnRecords rec;

    const auto t0 = clk::now();
    nvtxRangePushA("mtp_verify");
    Tensor hidden = prog_.forward(dev_, vin, true, &rec);
    out.logits = prog_.logits(dev_, hidden);
    out.g      = op::argmax(dev_, out.logits);
    {
        const Tensor base = main_in_.base_view(B);
        op::sample_into(dev_, out.logits, out.g, L, vin.first_region, &base, -1, samp_ptr());
    }
    dev_.synchronize();
    nvtxRangePop();
    const auto t1 = clk::now();

    op::Accept acc = op::mtp_accept(dev_, out.g, vin.ids, L, B);
    out.m = acc.m;
    prog_.fold_masked(dev_, vin, rec, acc.colmask, acc.m);
    dev_.synchronize();
    const auto t2 = clk::now();

    auto first = prog_.mtp(dev_, out.g, hidden, hin, Qwen35Hybrid::HeadRows::Accepted, acc.m);
    dev_.synchronize();
    const auto t3 = clk::now();

    nvtxRangePushA("mtp_chain");
    chain(out.drafts, first, main_in_.base_view(B), acc.m, 1, k_out, B, hin.n_kv, vin.first_region,
          main_in_.delta_view(B));
    out.confs = chain_confs_;
    dev_.synchronize();
    nvtxRangePop();
    if (B == 1 && !conf_logits_.empty()) {
        mtp_conf_log(conf_logits_);   // uncaptured here by construction
    }
    const auto t4 = clk::now();

    tm.mtp_phase_ms[0] += ms(t0, t1);   // verify: forward over L rows + full-head logits + argmax
    tm.mtp_phase_ms[1] += ms(t1, t2);   // accept + masked fold
    tm.mtp_phase_ms[2] += ms(t2, t3);   // head fill (L rows through the MTP block) + draft 1
    tm.mtp_phase_ms[3] += ms(t3, t4);   // the k-1 chained draft steps
    tm.mtp_prof_steps++;
    return out;
}

// ------------------------------------------------------------------ adaptive k

static double expected_tokens(const std::array<double, 9> & a, int k) {
    double e = 1.0, p = 1.0;
    for (int i = 1; i <= k; ++i) {
        p *= a[(size_t) i];
        e += p;
    }
    return e;
}

int Engine::choose_k(int k_in) const {
    const int kmax = cfg_.mtp_k;
    // Step time at a steady state k: a line t = a + b * (k_in + k_out) fitted by least squares
    // over every measured profile (warm launches only). With one measured point, probe k-1 once;
    // until a second point is warm, the slope is a prior of 7% of a step per unit of x.
    double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (const auto & kv : step_ms_) {
        const double x = kv.first.first + kv.first.second, y = kv.second;
        n += 1; sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    double a = 1.0, b = 0.07;
    if (n >= 1) {
        const double var = n * sxx - sx * sx;
        const double my = sy / n, mx = sx / n;
        if (var <= 1e-9) {
            // one measured cost point: probe a neighbouring k once so the slope gets measured
            // (a draft costs ~13% of a step; the prior below is that, per x unit)
            if (step_ms_.size() == 1 && step_ms_.begin()->first == std::make_pair(k_in, k_in)) {
                return k_in > 1 ? k_in - 1 : std::min(kmax, k_in + 1);
            }
            b = 0.07 * my;
        } else {
            b = (n * sxy - sx * sy) / var;
        }
        b = std::max(b, 0.0);
        a = my - b * mx;
    }
    auto est = [&](int k) { return std::max(1e-3, a + b * 2.0 * k); };
    int    pick = std::max(1, std::min(k_in, kmax));
    double pv   = -1;
    for (int k = std::max(1, k_in - 1); k <= std::min(kmax, k_in + 1); ++k) {
        const double v = expected_tokens(acc_, k) / est(k);
        if (v > pv * 1.0001) {
            pv   = v;
            pick = k;
        }
    }
    static const bool dbg = getenv("AI_ADAPT_DEBUG") != nullptr;
    if (dbg) {
        fprintf(stderr, "[adapt] k_in %d -> %d | a=%.2f %.2f %.2f %.2f | t=%.2f+%.3f*x | E/t:", k_in, pick, acc_[1],
                acc_[2], acc_[3], acc_[4], a, b);
        for (int k = 1; k <= kmax; ++k) {
            fprintf(stderr, " k%d %.3f", k, expected_tokens(acc_, k) / est(k));
        }
        fputc('\n', stderr);
    }
    return pick;
}

void Engine::observe(const std::vector<int> & accepted, int k_in, int k_out, double step_ms) {
    static constexpr double eta = 0.05;
    for (int m : accepted) {
        for (int i = 1; i <= std::min(m, k_in); ++i) {
            acc_[(size_t) i] += eta * (1.0 - acc_[(size_t) i]);
        }
        if (m < k_in) {
            acc_[(size_t) m + 1] += eta * (0.0 - acc_[(size_t) m + 1]);
        }
    }
    // depths this step did not draft relax toward the deepest observed one, so a k that was
    // dropped gets re-evaluated when acceptance recovers
    if (!accepted.empty()) {
        for (int i = k_in + 1; i < (int) acc_.size(); ++i) {
            acc_[(size_t) i] += 0.02 * (acc_[(size_t) k_in] - acc_[(size_t) i]);
        }
    }
    if (step_ms > 0) {
        auto it = step_ms_.find({ k_in, k_out });
        if (it == step_ms_.end()) {
            step_ms_[{ k_in, k_out }] = step_ms;
        } else {
            it->second += 0.2 * (step_ms - it->second);
        }
    }
}

// ------------------------------------------------------------------ DFlash2
// A DFlash2 step is one round: the verify half of an MTP step (forward + targets + accept + fold), then the
// drafter on the device -- all L verify rows' taps join its context (rows past m sit at positions the mask
// hides and the next round overwrites), and it drafts from g[m] at base+m+1 -- then one sync for (g, m, drafts).
// AI_DF2_HOST=1: the host-driven form instead (drafter after the sync), kept as a control.
// The drafter's LM head: the MTP draft head (q4_0, first 131072 rows, script-adaptive via AI_DRAFT_VOCAB) when the
// model has one. Proposals only: the verify keeps the full
// head, so text cannot change, only acceptance. AI_DF2_HEAD=full uses the full output head.
static const Tensor & df2_head(const Qwen35Hybrid & prog, const Tensor *& map) {
    static const bool full = getenv("AI_DF2_HEAD") && std::string(getenv("AI_DF2_HEAD")) == "full";
    map = nullptr;
    if (!full && prog.draft_head().valid()) {
        map = prog.draft_map().valid() ? &prog.draft_map() : nullptr;
        return prog.draft_head();
    }
    return prog.output();
}

static bool df2_host_mode() {
    static const bool v = getenv("AI_DF2_HOST") && atoi(getenv("AI_DF2_HOST")) != 0;
    return v;
}

Engine::StepOut Engine::build_df2(const Qwen35Hybrid::Inputs & vin) {
    const int64_t B = vin.B;
    const int64_t L = vin.T / B;
    StepOut out;
    Qwen35Hybrid::GdnRecords rec;
    Tensor hidden = prog_.forward(dev_, vin, true, &rec);   // also writes the taps of all L rows
    out.logits = prog_.logits(dev_, hidden);
    out.g      = op::argmax(dev_, out.logits);
    {
        const Tensor base = main_in_.base_view(B);
        op::sample_into(dev_, out.logits, out.g, L, vin.first_region, &base, -1, samp_ptr());
    }
    op::Accept acc = op::mtp_accept(dev_, out.g, vin.ids, L, B);
    out.m = acc.m;
    prog_.fold_masked(dev_, vin, rec, acc.colmask, acc.m);
    if (df2_host_mode()) {
        return out;
    }
    if (B != 1 || L != df2_->block_size()) {
        throw std::runtime_error("df2: the device round needs batch 1 and L == block size");
    }
    const Tensor & tp = prog_.taps();
    const Tensor * hm = nullptr;
    const Tensor & hw = df2_head(prog_, hm);   // before the call: argument evaluation order is unspecified
    const Tensor   dr = df2_->round(dev_, tp.view(tp.ne[0], L, 1, 1, tp.nb[1], tp.nb[1] * L, tp.nb[1] * L, 0),
                                    main_in_.base_view(B), acc.m, out.g, prog_.tok_embd(), hw, hm, vin.n_kv);
    for (int64_t j = 0; j < L - 1; ++j) {
        out.drafts.push_back(dr.view(1, 1, 1, 1, 4, 4, 4, (size_t) j * 4));
    }
    return out;
}

void Engine::df2_advance(Stream & st, int64_t rows, int64_t pos0, int32_t anchor, int64_t anchor_pos) {
    dev_.ws_tag("dflash2 advance");
    const Tensor & tp = prog_.taps();
    // taps_rows() is set when a round is built; a replayed graph of another width (AI_LOOKUP buckets) writes its own L
    // rows without updating it, so bound by the tap buffer's capacity (the caller's rows <= that round's L)
    if (rows > tp.ne[1]) {
        throw std::runtime_error("df2: more committed rows than the tap buffer holds");
    }
    const size_t   mark = dev_.ws_mark();
    const Tensor * hm   = nullptr;
    const Tensor & hw   = df2_head(prog_, hm);
    df2_->ctx_append(dev_, tp.view(tp.ne[0], rows, 1, 1, tp.nb[1], tp.nb[1] * rows, tp.nb[1] * rows, 0), pos0);
    auto d = df2_->draft(dev_, anchor, anchor_pos, prog_.tok_embd(), hw, false, hm);
    dev_.ws_reset(mark);
    st.drafts = d.tokens;
}

void Engine::prefill_df2(int stream, const std::vector<int32_t> & prompt, Stream & st) {
    dev_.ws_tag("dflash2 prefill");
    const int64_t n = (int64_t) prompt.size();
    df2_->reset();
    int32_t tok = -1;
    for (int64_t off = 0; off < n; off += cfg_.chunk) {
        const int64_t T    = std::min(cfg_.chunk, n - off);
        const bool    last = off + T == n;
        dev_.ws_reset(0);
        auto in = stage_chunk(stream, prompt.data() + off, T, off, prog_.visible_cells(off, T));
        std::unique_ptr<op::Round> round;
        if (!chunk_eager()) {
            round = std::make_unique<op::Round>(dev_);
            round->begin();
        }
        // last chunk (exact replay): all rows through the final layer + last_row's logits, as plain and MTP prefill
        Tensor hidden = prog_.forward(dev_, in, last);
        Tensor am;
        if (last) {
            const Tensor lg = prog_.logits(dev_, last_row(hidden));
            am = op::argmax(dev_, lg);
            op::sample_into(dev_, lg, am, 1, stream, nullptr, n, samp_ptr());   // the first token is at position n
        }
        if (round) {
            round->end();
            round->launch();
        }
        // this chunk's taps -> the drafter's context (same stream, after the chunk's round)
        const Tensor & tp = prog_.taps();
        df2_->ctx_append(dev_, tp.view(tp.ne[0], T, 1, 1, tp.nb[1], tp.nb[1] * T, tp.nb[1] * T, 0), off);
        if (last) {
            dev_.download_async(am, h_out_, 0, 4);
            dev_.synchronize();
            tok = h_out_[0];
        }
    }
    st.pend = tok;
    const Tensor * hm = nullptr;
    const Tensor & hw = df2_head(prog_, hm);
    auto d = df2_->draft(dev_, tok, n, prog_.tok_embd(), hw, false, hm);
    st.drafts = d.tokens;
}

}  // namespace ai
