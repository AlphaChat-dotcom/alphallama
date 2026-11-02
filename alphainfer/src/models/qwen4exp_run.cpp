// SPDX-License-Identifier: Apache-2.0
#include "models/qwen4exp_run.h"
#include "core/host_mem.h"
#include "offload/cpu_topo.h"

#include "ggml.h"
#include "kernels/kernels.h"
#include "kernels/q4x_fold.h"
#include "ops/adapter.h"

#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <pthread.h>
#include <sched.h>
#include <fstream>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ai {

// AI_Q4X_ROUND_TIMING=1: host time per captured round by phase (ms, averaged at exit), i.e. the host part of a
// verify's wall time before its GPU work (the caller's logits read is the rest)
namespace {
struct RoundTiming {
    bool   on = getenv("AI_Q4X_ROUND_TIMING") && atoi(getenv("AI_Q4X_ROUND_TIMING")) != 0;
    double admit = 0, flush = 0, sync = 0, ple = 0, upload = 0, launch = 0, kick = 0;
    long   n = 0;
    // GPU side, from events (q4x_rt_mark): per cycle c = (draft c, verify c), read at round c + 1's sync. Marks 0/1 bracket
    // the MTP chain, 2/3 the verify launch; parity c & 1 holds cycle c's events.
    cudaEvent_t ev[2][4] = {};
    bool        drafted[2] = {}, v_ok[2] = {};
    long        cyc = 0, nv = 0, nd = 0, ni = 0;
    double      vg = 0, dg = 0, d2v = 0, idle = 0;
    // the commit (state folds) after verify c, bracketed by its own pair; read at the same sync (it completed before it)
    cudaEvent_t cev[2] = {};
    bool        c_ok = false;
    long        nc = 0;
    double      cg = 0, ch = 0;
    void cmark(int which, void * stream) {
        if (!cev[which]) cudaEventCreate(&cev[which]);
        cudaEventRecord(cev[which], (cudaStream_t) stream);
        if (which == 1) c_ok = true;
    }
    void mark(int which, void * stream) {
        const int p = (int) (cyc & 1);
        if (!ev[p][which]) cudaEventCreate(&ev[p][which]);
        cudaEventRecord(ev[p][which], (cudaStream_t) stream);
        if (which == 0) drafted[p] = true;
        if (which == 3) { v_ok[p] = true; ++cyc; drafted[cyc & 1] = false; }
    }
    void gpu_eval() {   // all queued work is complete (called right after a device synchronize)
        if (cyc < 1) return;
        const int p = (int) ((cyc - 1) & 1), q = (int) (cyc & 1);
        auto el = [](cudaEvent_t a, cudaEvent_t b) { float ms = 0; return cudaEventElapsedTime(&ms, a, b) == cudaSuccess ? (double) ms : -1.0; };
        const double v = el(ev[p][2], ev[p][3]);
        if (v < 0) { (void) cudaGetLastError(); return; }
        vg += v; ++nv;
        double d = 0;
        if (drafted[p]) { d = el(ev[p][0], ev[p][1]); dg += d; d2v += el(ev[p][1], ev[p][2]); ++nd; }
        if (cyc >= 2 && v_ok[q]) { const double span = el(ev[q][3], ev[p][3]); if (span > 0) { idle += span - v - d; ++ni; } }
        if (c_ok) { const double c = el(cev[0], cev[1]); if (c >= 0) { cg += c; ++nc; } c_ok = false; }
        (void) cudaGetLastError();
    }
    ~RoundTiming() {
        if (on && n > 0)
            fprintf(stderr, "[q4x-round-timing] %ld rounds, host ms/round: admit %.3f | mtp fill %.3f | sync %.3f | ingress+mask %.3f | "
                            "ple rows %.3f | launch %.3f | admit kick %.3f%c", n, admit / n, flush / n, sync / n, upload / n, ple / n,
                    launch / n, kick / n, 10);
        if (on && nv > 0)
            fprintf(stderr, "[q4x-round-timing] GPU ms per cycle (events): verify %.3f (%ld) | draft %.3f (%ld) | draft end -> verify "
                            "start %.3f | cycle idle (span - verify - draft) %.3f (%ld)%c", vg / nv, nv, nd ? dg / nd : 0.0, nd,
                    nd ? d2v / nd : 0.0, ni ? idle / ni : 0.0, ni, 10);
        if (on && nc > 0)
            fprintf(stderr, "[q4x-round-timing] commit (state folds) per cycle: host %.3f ms, GPU span %.3f ms (%ld)%c", ch / nc, cg / nc, nc, 10);
    }
};
RoundTiming g_rt;
inline double rt_ms_since(std::chrono::steady_clock::time_point & t) {
    const auto   now = std::chrono::steady_clock::now();
    const double ms  = std::chrono::duration<double, std::milli>(now - t).count();
    t = now;
    return ms;
}

}  // namespace

// AI_Q4X_ROUND_TIMING: the MTP chain's GPU bracket (0 = before, 1 = after), recorded on its stream
void q4x_rt_mark(int which, void * stream) {
    if (g_rt.on) g_rt.mark(which, stream);
}

// GPU miss share: a GPU-taken missed expert e arrives on the bell as -(e + 2) - kGpuTagH (= kernels/offload.cu kGpuTag)
static constexpr int32_t kGpuTagH = 1 << 20;
// [AI_Q4X_HIT_STATS] hits / entries, all ranks and ranks 0..6; printed once at exit
static unsigned long long g_hits[4], g_admits;
static struct HitStatsPrint {
    ~HitStatsPrint() {
        if (g_hits[1] == 0) return;
        fprintf(stderr, "[q4x] decode hit rate: %.1f%% of %llu routed entries; ranks 0-6: %.1f%% of %llu; admissions %llu%c",
                100.0 * (double) g_hits[0] / (double) g_hits[1], g_hits[1], 100.0 * (double) g_hits[2] / (double) std::max(1ULL, g_hits[3]),
                g_hits[3], g_admits, 10);
    }
} g_hit_stats_print;

static constexpr int kRopeImrope = 40;   // interleaved M-RoPE, NEOX ordering (as qwen35_hybrid.cpp)

static Tensor carve(int buf, char * base, size_t & off, int type, int64_t n0, int64_t n1 = 1) {
    Tensor t;
    t.buf   = buf;
    t.type  = type;
    t.ne[0] = n0;
    t.ne[1] = n1;
    set_contiguous_strides(t);
    off     = (off + 255) / 256 * 256;
    t.data  = base + off;
    off    += t.nbytes();
    return t;
}

Qwen4ExpRun::Qwen4ExpRun(Qwen4Exp & prog, Device & dev, int64_t max_ctx, int expert_slots)
    : prog_(prog), dev_(dev), max_ctx_(max_ctx) {
    const Qwen4ExpSpec & s = prog.spec();
    if (expert_slots < s.n_expert_used) {
        throw std::runtime_error("Qwen4ExpRun: expert_slots must be >= n_expert_used");
    }
    ap_.n_head     = s.n_head;
    ap_.n_head_kv  = s.n_head_kv;
    ap_.head_dim   = s.head_dim;
    ap_.n_rot      = s.n_rot;
    for (int i = 0; i < 4; ++i) ap_.sections[i] = s.sections[i];
    ap_.rope_mode  = kRopeImrope;
    ap_.n_ctx_orig = s.n_ctx_train;
    ap_.rope_base  = s.rope_base;
    ap_.rope_scale = 1.0f;
    ap_.eps        = s.rms_eps;
    ap_.kq_scale   = 1.0f / std::sqrt((float) s.head_dim);
    gp_.conv_k   = s.conv_k;
    gp_.channels = (int) s.conv_channels();
    gp_.k_heads  = s.ssm_groups;
    gp_.v_heads  = s.ssm_dt_rank;
    gp_.head_k   = s.ssm_state;
    gp_.head_v   = s.ssm_inner / s.ssm_dt_rank;
    gp_.eps      = s.rms_eps;
    pp_.n_heads  = s.ple_n_heads();
    pp_.head_dim = s.ple_head_dim;
    pp_.ngram    = s.ple_ngram;
    pp_.conv_k   = s.ple_conv_k;
    pp_.eps      = s.rms_eps;

    // ggml-cuda picks its dense small-M GEMM tier (MMQ at 4..8 columns for >= 16 MiB weights) once per process, from
    // the first graph that holds a MUL_MAT_ID (g_alpha_graph_has_mmid). With the routed experts as native kernels a
    // prefill graph has none, so the tier would switch on for the dense projections and change their numerics. This
    // is an MoE: pin the MoE default before the first graph. ALPHA_SMALLM_MMQ=-1 restores the probe.
    setenv("ALPHA_SMALLM_MMQ", "0", 0);
    // ggml's chunked GDN prompt scan (prefill only; decode takes the sequential kernel) in plain TF32 instead of
    // 3xTF32: slightly faster, and paired NLL is unchanged within noise. ALPHA_GDN_TF32_1X=0 restores 3xTF32.
    setenv("ALPHA_GDN_TF32_1X", "1", 0);

    // one device buffer for every layer's KV, indexer caches and recurrent state
    const int64_t D = s.hc_dim(), kvw = (int64_t) s.head_dim * s.n_head_kv;
    const int64_t Sv = gp_.head_v;
    const auto & layers = prog.layers();
    // AI_Q4X_QSA_KEYS_F32=1: keep the raw indexer keys in f32 instead of f16 (diagnostic for precision drift in the
    // indexer at long context; default off)
    static const bool keys_f32 = getenv("AI_Q4X_QSA_KEYS_F32") && atoi(getenv("AI_Q4X_QSA_KEYS_F32")) != 0;
    const int         kt  = keys_f32 ? F32 : F16;
    const size_t      kes = keys_f32 ? 4 : 2;
    size_t bytes = 0;
    for (const auto & L : layers) {
        if (L.attn) {
            bytes += 2 * (size_t) (kvw * max_ctx * 2) + 1024;
            bytes += (size_t) (s.idx_head_dim * max_ctx * kes) + 512;
            if (L.qsa_p.ratio > 0) bytes += (size_t) (s.idx_head_dim * (max_ctx / L.qsa_p.ratio + 2) * 4) + 512;
        } else {
            bytes += (size_t) ((s.conv_k - 1) * gp_.channels * 4) + 512;
            bytes += (size_t) (Sv * Sv * gp_.v_heads * 4) + 512;
        }
        if (L.ple) bytes += (size_t) ((s.ple_conv_k - 1) * s.ple_ngram * D * 4) + 512;
    }
    state_buf_ = dev.alloc_buffer(bytes, "q4x.state");
    char * base = (char *) dev.buffer_base(state_buf_);
    cudaMemset(base, 0, bytes);   // padded decode envelopes read cells past the position: zeros, not NaN
    size_t off  = 0;
    kv_.assign(layers.size(), {});
    qsa_.assign(layers.size(), {});
    gdn_.assign(layers.size(), {});
    ple_.assign(layers.size(), {});
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & L = layers[il];
        if (L.attn) {
            kv_[il].k = carve(state_buf_, base, off, F16, kvw, max_ctx);
            kv_[il].v = carve(state_buf_, base, off, F16, kvw, max_ctx);
            qsa_[il].keys = carve(state_buf_, base, off, kt, s.idx_head_dim, max_ctx);
            if (L.qsa_p.ratio > 0) {
                // f32: the pooled block keys as the graph computes them (attn_qsa_block's cache: complete blocks are
                // pooled once; the last row is the dump for the fixed-size dirty set's unused entries)
                qsa_[il].pooled = carve(state_buf_, base, off, F32, s.idx_head_dim, max_ctx / L.qsa_p.ratio + 2);
            }
        } else {
            gdn_[il].conv = carve(state_buf_, base, off, F32, (int64_t) (s.conv_k - 1) * gp_.channels, 1);
            gdn_[il].ssm  = carve(state_buf_, base, off, F32, Sv * Sv * gp_.v_heads, 1);
        }
        if (L.ple) ple_[il].hist = carve(state_buf_, base, off, F32, (int64_t) (s.ple_conv_k - 1) * s.ple_ngram * D, 1);
    }

    // eager mode: the expert pool over the host-mapped expert banks; doorbell mode: static slot banks + CPU miss pool
    std::vector<std::vector<Tensor>> banks;
    for (const auto & L : layers) banks.push_back({ L.moe.up_exps, L.moe.gate_exps, L.moe.down_exps });
    doorbell_ = getenv("AI_Q4X_DOORBELL") && atoi(getenv("AI_Q4X_DOORBELL")) != 0;
    if (doorbell_) {
        init_doorbell(expert_slots);
    } else {
        pool_.alloc(dev, banks, s.n_expert, expert_slots);
    }
    reset();
}

void Qwen4ExpRun::reset() {
    fold_join(true);
    ple_gate_idle();
    hist_.clear();
    ck_len_ = -1;
    if (mtp_on_) {   // no pending fill, no carried H (the head's K/V is masked by position)
        mtp_fill_m_ = 0;
        cudaMemset(mtp_H_.data, 0, (size_t) mtp_H_.nbytes());
    }
    // recurrent state and PLE history start at zero (the KV is masked by position, so it needs no clear)
    std::vector<char> z;
    for (size_t il = 0; il < gdn_.size(); ++il) {
        for (const Tensor * t : { &gdn_[il].conv, &gdn_[il].ssm, &ple_[il].hist }) {
            if (!t->valid()) continue;
            z.assign(t->nbytes(), 0);
            dev_.upload(*t, z.data(), 0, z.size());
        }
    }
}

// Prefix reuse. The sequence's state at a position is the KV / indexer cells below it (positional: later rounds only
// write cells above, and the mask hides them), the recurrent tensors (GDN conv + ssm, PLE history) and the MTP head's
// carried streams + pending fill. checkpoint() copies the recurrent part to the host; restore_checkpoint() writes it
// back and truncates the history, so the next prefill continues from that position.
std::vector<Tensor *> Qwen4ExpRun::recurrent_tensors() {
    std::vector<Tensor *> v;
    for (size_t il = 0; il < gdn_.size(); ++il)
        for (Tensor * t : { &gdn_[il].conv, &gdn_[il].ssm, &ple_[il].hist })
            if (t->valid()) v.push_back(t);
    if (mtp_on_ && mtp_H_.valid()) v.push_back(&mtp_H_);
    return v;
}

void Qwen4ExpRun::checkpoint() {
    fold_join(true);
    dev_.synchronize();
    const std::vector<Tensor *> ts = recurrent_tensors();
    ck_buf_.resize(ts.size());
    for (size_t i = 0; i < ts.size(); ++i) {
        ck_buf_[i].resize(ts[i]->nbytes());
        dev_.download(*ts[i], ck_buf_[i].data(), 0, ck_buf_[i].size());
    }
    ck_fill_p0_ = mtp_fill_p0_;
    ck_fill_m_  = mtp_fill_m_;
    ck_len_     = (int64_t) hist_.size();
}

bool Qwen4ExpRun::restore_checkpoint() {
    if (ck_len_ < 0 || ck_len_ > (int64_t) hist_.size()) return false;
    fold_join(true);
    ple_gate_idle();
    dev_.synchronize();
    const std::vector<Tensor *> ts = recurrent_tensors();
    if (ts.size() != ck_buf_.size()) return false;
    for (size_t i = 0; i < ts.size(); ++i) dev_.upload(*ts[i], ck_buf_[i].data(), 0, ck_buf_[i].size());
    dev_.synchronize();   // Device::upload is unordered against the stream: land it before the next launch
    mtp_fill_p0_ = ck_fill_p0_;
    mtp_fill_m_  = ck_fill_m_;
    hist_.resize((size_t) ck_len_);
    bells_live_ = false;   // the bells hold rounds past the checkpoint
    return true;
}

// [AI_Q4X_PLE_IO] the PLE rows' host reads. Rows are hashed n-grams spread over the 29 GB table and the page cache holds
// only what was read since the tier fill dropped it. One blocking pread per row, serially on the decode thread, cost
// 8.7-10.3 ms of host time per round on the 3090 starts whose disk was slow (tier read 36-40 s; 63-78 vs 91-105 tok/s,
// 2026-10-30) and 7.2 ms under an O_DIRECT read ballast. Per call:
//   1. the row ids are de-duplicated (55.7% of a call's rows repeat another's);
//   2. every unique row is tried with preadv2 RWF_NOWAIT -- a page-cache hit costs what the serial pread did;
//   3. the rows that would wait for the disk are all queued with WILLNEED readahead at once, then read: inline below
//      AI_Q4X_PLE_IO_MIN rows (a decode round), else by blocking reader threads too (AI_Q4X_PLE_IO, default 16; on the
//      host cpus, never the launch core or the spinning pool).
// A row cache across calls was tried and dropped: 0.3% hits (the next round's rows are never this round's), and waking
// the readers for page-cache hits cost +0.4 ms/round quiet. Same bytes, same to_float: the output does not change.
struct Qwen4ExpRun::PleIo {
    const WeightSet & ws;
    size_t            rb;
    struct Job { const char * src; char * dst; };
    std::vector<Job>         jobs;
    std::atomic<size_t>      next{0};
    std::atomic<size_t>      left{0};
    std::mutex               mu, call_mu;
    std::condition_variable  cv, done_cv;
    uint64_t                 gen = 0, open_gen = 0;   // open_gen: the threaded call readers may still join (0 = none)
    int                      draining = 0;
    bool                     stop = false;
    // [AI_Q4X_PLE_IO_MIN, default 64] fewer disk rows than this are read inline (a decode round's ~20: waking the
    // readers cost more than it saved); a prefill chunk's thousands go to the readers
    const int                min_par = getenv("AI_Q4X_PLE_IO_MIN") ? std::max(2, atoi(getenv("AI_Q4X_PLE_IO_MIN"))) : 64;
    // [AI_Q4X_PLE_SLOW_MS, default 2] the inline path's slow mode: on a start whose disk is slow and page cache under
    // reclaim, WILLNEED did not land and the inline reads waited one NVMe latency each -- PLE 10.2 ms/round, 74 vs 98.6
    // tok/s (3090 sd1 b3, MTP 3: ~27 disk rows per round, under min_par). A call whose reads took longer than this sends
    // the next calls' disk rows to the readers too (any count >= 2); every 256 calls the inline path is tried again.
    const double             slow_ms = getenv("AI_Q4X_PLE_SLOW_MS") ? atof(getenv("AI_Q4X_PLE_SLOW_MS")) : 2.0;
    int                      slow_left = 0;   // calls left in the slow mode
    uint64_t                 n_slow = 0;      // calls that switched it on
    bool                     direct = false;  // this call's jobs read through O_DIRECT (see ple_rows)
    uint64_t                 n_direct = 0;    // rows read that way
    std::vector<std::thread> thr;
    std::vector<char>        buf;    // one raw row per unique id of the call
    std::vector<uint64_t>    key;    // open addressing over the call's ids: id + 1 (0 = empty)
    std::vector<uint32_t>    slot;   //   -> its row in buf
    // stats (AI_Q4X_HIT_STATS)
    uint64_t n_rows = 0, n_unique = 0, n_cached = 0, n_disk = 0, n_woke = 0;

    PleIo(const WeightSet & w, size_t row_bytes, int n_thr, const std::vector<int> & cpus) : ws(w), rb(row_bytes) {
        for (int i = 0; i < n_thr; ++i) thr.emplace_back([this, cpus]() {
            if (!cpus.empty()) offload::pin_this_thread(cpus);
            uint64_t seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&]() { return stop || gen != seen; });
                    if (stop) return;
                    seen = gen;
                    // [stray-token race, 2026-11-01] a reader that wakes AFTER its call finished must not drain: run()
                    // returned on left == 0 && draining == 0 while this thread was not yet counted, and the next call
                    // (often an inline one, which never waits for readers) was refilling jobs / next while this thread
                    // pread() into its rows -- a torn PLE row, one wrong token, run-to-run different (5090 scan:
                    // 4-14% of outputs in every build since the reader; Mole 4080 production repro)
                    if (open_gen != gen) continue;
                    ++draining;
                }
                drain();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    --draining;
                }
                done_cv.notify_all();
            }
        });
    }
    ~PleIo() {
        if (getenv("AI_Q4X_HIT_STATS") && n_rows)
            fprintf(stderr, "[q4x] PLE io: %llu rows, %.1f%% unique; of those %.1f%% in the page cache, %llu read from disk (%llu calls woke the readers, slow mode entered %llu times, %llu rows O_DIRECT)\n",
                    (unsigned long long) n_rows, 100.0 * n_unique / n_rows, n_unique ? 100.0 * n_cached / n_unique : 0.0,
                    (unsigned long long) n_disk, (unsigned long long) n_woke, (unsigned long long) n_slow,
                    (unsigned long long) n_direct);
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : thr) t.join();
    }
    void drain() {
        for (size_t i; (i = next.fetch_add(1)) < jobs.size();) {
            const Job & j = jobs[i];
            if (direct && ws.host_pread_direct(j.src, j.dst, rb)) { left.fetch_sub(1); continue; }
            if (!ws.host_pread(j.src, j.dst, rb)) memcpy(j.dst, j.src, rb);   // not a shard's range: the mapping
            left.fetch_sub(1);
        }
    }
    // read every job; returns once all are in place and no reader is inside drain()
    void run() {
        if (jobs.empty()) return;
        const bool slow = slow_left > 0 || direct;   // direct reads always go to the readers (one NVMe latency each)
        if (slow_left > 0) --slow_left;
        if (thr.empty() || jobs.size() < (size_t) (slow ? 2 : min_par)) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!thr.empty()) {   // no reader may still be inside an earlier call's drain() (open_gen closes it)
                std::unique_lock<std::mutex> lk(mu);
                done_cv.wait(lk, [&]() { return draining == 0; });
            }
            next = 0; left = jobs.size();
            drain();
            if (!thr.empty() && slow_ms > 0 &&
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > slow_ms) {
                slow_left = 256;
                ++n_slow;
            }
            return;
        }
        ++n_woke;
        {
            std::unique_lock<std::mutex> lk(mu);
            done_cv.wait(lk, [&]() { return draining == 0; });
            next = 0; left = jobs.size();
            ++gen;
            open_gen = gen;
        }
        cv.notify_all();
        drain();
        std::unique_lock<std::mutex> lk(mu);
        done_cv.wait(lk, [&]() { return left.load() == 0 && draining == 0; });
        open_gen = 0;   // closed under the lock: a reader that has not counted itself in yet will skip this call
    }
};

// AI_Q4X_PLE_WARM (default auto): read the whole PLE table back into the page cache on a background thread after the
// load dropped the model files' cache (AI_Q4X_DROP_CACHE). Decode reads ~T x 30 scattered PLE rows per round; cold,
// two thirds of them come from disk (0.5 ms of host time on the critical path between the draft and the verify on the
// 5090, 2026-10-31); warm, 0.05 ms. auto = only when MemAvailable keeps 16 GiB beyond the table after the RAM tier, so
// the cached table cannot push the box into the reclaim/compaction the cache drop guards against; 1 = always, 0 = off.
void Qwen4ExpRun::ple_warm() {
    const Tensor & tbl = prog_.ple_table();
    if (!tbl.valid() || !tbl.data) return;
    static const char * ev = getenv("AI_Q4X_PLE_WARM");
    const int mode = ev ? atoi(ev) : -1;
    if (mode == 0) return;
    const size_t bytes = (size_t) tbl.nb[1] * (size_t) tbl.ne[1];
    long long avail_kb = 0;
    {
        std::ifstream mi("/proc/meminfo");
        std::string key;
        while (mi >> key) { if (key == "MemAvailable:") { mi >> avail_kb; break; } mi.ignore(1 << 10, '\n'); }
    }
    const long long margin = 16ll << 30;
    if (mode < 0 && avail_kb * 1024 < (long long) bytes + margin) {
        fprintf(stderr, "[q4x] PLE warm: off (MemAvailable %.1f GiB < table %.1f GiB + 16 GiB; AI_Q4X_PLE_WARM=1 forces)\n",
                avail_kb / 1048576.0, bytes / 1073741824.0);
        return;
    }
    const char *      src  = (const char *) tbl.data;
    const WeightSet * ws   = &prog_.weights();
    std::vector<int>  cpus = host_cpus_;
    // AI_Q4X_PLE_WARM_THREADS (default 4) readers over interleaved 8 MiB chunks, after one WILLNEED over the whole
    // table queues the kernel's readahead: ~1 GB/s from one blocking reader, several GB/s this way
    static const int n_rd = getenv("AI_Q4X_PLE_WARM_THREADS") ? std::max(1, atoi(getenv("AI_Q4X_PLE_WARM_THREADS"))) : 4;
    std::thread warm([src, bytes, ws, cpus]() {
        if (!cpus.empty()) {   // the host cpus, never the pool's
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int c : cpus) CPU_SET(c, &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);
        }
        const auto   t0    = std::chrono::steady_clock::now();
        const size_t chunk = 8u << 20;
        ws->host_prefetch(src, bytes);
        std::atomic<size_t> done{ 0 };
        std::vector<std::thread> rd;
        for (int t = 0; t < n_rd; ++t) {
            rd.emplace_back([&, t]() {
                if (!cpus.empty()) {
                    cpu_set_t set;
                    CPU_ZERO(&set);
                    for (int c : cpus) CPU_SET(c, &set);
                    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
                }
                std::vector<char> buf(chunk);
                for (size_t off = (size_t) t * chunk; off < bytes; off += (size_t) n_rd * chunk) {
                    const size_t n = std::min(chunk, bytes - off);
                    if (!ws->host_pread(src + off, buf.data(), n)) break;
                    done += n;
                }
            });
        }
        for (std::thread & r : rd) r.join();
        fprintf(stderr, "[q4x] PLE warm: %.2f GiB of the PLE table read into the page cache in %.1f s (%d readers)\n",
                done.load() / 1073741824.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), n_rd);
    });
    // (=0 detaches the readers, which hold the weights' mapping: a shutdown inside the warm is not safe -- debug only)
    // AI_Q4X_PLE_WARM_SYNC (default 1): the load waits for the warm (~12 s on the 5090's NVMe), so the first requests
    // do not decode against a half-cold table while the readers compete with the host threads; 0 = in the background
    static const bool sync = !getenv("AI_Q4X_PLE_WARM_SYNC") || atoi(getenv("AI_Q4X_PLE_WARM_SYNC")) != 0;
    fprintf(stderr, "[q4x] PLE warm: reading the %.2f GiB PLE table into the page cache%s (MemAvailable %.1f GiB)\n",
            bytes / 1073741824.0, sync ? "" : " in the background", avail_kb / 1048576.0);
    if (sync) warm.join();
    else warm.detach();
}


// PLE n-gram hash rows (alphallama llm_graph_input_ple::set_input, qwen4exp.cpp:860-930): for each position the
// window [t, t-1, .., t-ngram+1] with predecessors from this sequence's history; a missing one, or anything at or
// before an EOS in the window, reads as EOS (the token's own EOS does not cut). Rows dequantised with ggml's to_float.
void Qwen4ExpRun::ple_rows(int64_t pos0, int64_t T, float * out) const {
    const Qwen4ExpSpec & s = prog_.spec();
    const Tensor & tbl = prog_.ple_table();
    const int64_t n_gram = s.ple_ngram, n_heads = s.ple_n_heads(), per_gram = s.ple_heads_per_ngram;
    const int64_t hd = s.ple_head_dim, eos = s.ple_eos;
    const auto * tt = ggml_get_type_traits((ggml_type) tbl.type);
    std::vector<int64_t> ctx((size_t) n_gram);
    struct Row { const char * src; float * dst; uint64_t r; const char * raw; };
    std::vector<Row> rows;
    rows.reserve((size_t) (T * n_heads));
    for (int64_t i = 0; i < T; ++i) {
        const int64_t p = pos0 + i;
        ctx[0] = hist_[(size_t) p];
        bool cut = false;
        for (int64_t sft = 1; sft < n_gram; ++sft) {
            const int64_t q = p - sft;
            const int64_t t = cut || q < 0 ? -1 : hist_[(size_t) q];
            cut = cut || t < 0 || t == eos;
            ctx[(size_t) sft] = cut ? eos : t;
        }
        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * (uint64_t) s.ple_multipliers[0];
            for (int64_t j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[(size_t) j] * (uint64_t) s.ple_multipliers[(size_t) j];
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram && base + g < n_heads; ++g) {
                const int64_t h   = base + g;
                const uint64_t r  = mixed % (uint64_t) s.ple_head_vocab[(size_t) h] + (uint64_t) s.ple_head_offsets[(size_t) h];
                rows.push_back({ (const char *) tbl.data + (size_t) r * tbl.nb[1], out + (i * n_heads + h) * hd, r, nullptr });
            }
        }
    }
    const size_t rb = (size_t) tbl.nb[1];
    auto to_f32 = [&](const char * src, float * dst) {
        if (tbl.type == F32) memcpy(dst, src, (size_t) hd * 4);
        else tt->to_float(src, dst, hd);
    };
    // [AI_Q4X_PLE_PREAD, default 1] the row through pread: the 29 GB table's pages are then never mapped into this
    // process, so compaction migrating them cannot IPI the pool (WeightSet::host_pread)
    static const bool ple_pread = !getenv("AI_Q4X_PLE_PREAD") || atoi(getenv("AI_Q4X_PLE_PREAD")) != 0;
    static const int  io_thr    = getenv("AI_Q4X_PLE_IO") ? std::max(0, atoi(getenv("AI_Q4X_PLE_IO"))) : 16;
    if (ple_pread && io_thr > 0) {
        if (!ple_io_) {
            ple_io_.reset(new PleIo(prog_.weights(), rb, io_thr, host_cpus_));
            fprintf(stderr, "[q4x] PLE io: page-cache-first reads, %d reader threads for the rest%s\n", io_thr,
                    host_cpus_.empty() ? "" : (" on " + offload::cpu_list_str(host_cpus_)).c_str());
        }
        PleIo & io = *ple_io_;
        std::lock_guard<std::mutex> call_lk(io.call_mu);   // the prefill's PLE thread and the decode never overlap
        size_t cap = 16;
        while (cap < 2 * rows.size()) cap <<= 1;
        io.key.assign(cap, 0);
        io.slot.resize(cap);
        if (io.buf.size() < rows.size() * rb) io.buf.resize(rows.size() * rb);
        io.jobs.clear();
        // [AI_Q4X_PLE_DIRECT, default 0 = off; N = calls of <= N rows] DEFAULT OFF since 2026-10-31: on healthy 3090 starts
        // the buffered reads hit the page cache for 31-35% of rows and cost 0.80-0.85 ms/round vs 1.19-1.22 direct (fd2,
        // 5 + 5 fresh starts, set A), and direct did NOT prevent the stall: fd1's 2 stalled starts of 6 were direct, at
        // 17-18 ms/round (A 69-90 tok/s). The stall follows the load's disk / reclaim backlog (fd2 sampler: 3-20 ms per IO
        // in the first seconds, 1.4-2.1M pages scanned at the tier pin), not the read mode. Opt-in: 512.
        // Original rationale: a call this small (a decode / verify round) reads every unique
        // row through O_DIRECT on the readers, past the page cache. On 2 in 10 fresh 3090 starts the buffered reads ran
        // ~10x slower for the whole run (PLE 8.6-10.5 ms/round vs 0.8; 74-78 vs ~100 tok/s), and 16 parallel readers did
        // not help (sd2: the same 10 ms) -- the page-cache path under reclaim (kswapd stole 1.4-1.8M pages a run on a
        // 62 GB box with a 40 GB pinned tier), not disk latency. Strata reads these rows with --ple-io direct. Larger
        // calls (prefill chunks) keep the page cache: neighbouring rows share pages there.
        static const int direct_max = getenv("AI_Q4X_PLE_DIRECT") ? std::max(0, atoi(getenv("AI_Q4X_PLE_DIRECT"))) : 0;
        io.direct = direct_max > 0 && rows.size() <= (size_t) direct_max;
        std::vector<uint32_t> at(rows.size());
        uint32_t n_u = 0;
        for (size_t k = 0; k < rows.size(); ++k) {
            const uint64_t id = rows[k].r + 1;
            size_t h = (size_t) ((id * 0x9E3779B97F4A7C15ull) >> 20) & (cap - 1);
            while (io.key[h] != 0 && io.key[h] != id) h = (h + 1) & (cap - 1);
            if (io.key[h] == 0) {
                io.key[h] = id;
                io.slot[h] = n_u;
                char * dst = io.buf.data() + (size_t) n_u * rb;
                if (io.direct) io.jobs.push_back({ rows[k].src, dst });
                else {
                    const int r = prog_.weights().host_pread_nowait(rows[k].src, dst, rb);
                    if (r == 1) ++io.n_cached;
                    else io.jobs.push_back({ rows[k].src, dst });   // on disk (0) or not readable that way (-1): blocking pread
                }
                ++n_u;
            }
            at[k] = io.slot[h];
        }
        io.n_rows += rows.size();
        io.n_unique += n_u;
        io.n_disk += io.jobs.size();
        if (io.direct) io.n_direct += io.jobs.size();
        // every row that would wait is queued with the kernel at once (WILLNEED readahead: the serial path's deep queue,
        // which beat 8 blocking readers on a cold table -- pl4: decode 0.66 vs 1.05 ms/round, prefill 8.6 vs 9.9 s, and
        // 9.0 vs 15.2 s under a read ballast); the reads then mostly find the pages arriving
        if (!io.direct) for (const PleIo::Job & j : io.jobs) prog_.weights().host_prefetch(j.src, rb);
        io.run();
        for (size_t k = 0; k < rows.size(); ++k) to_f32(io.buf.data() + (size_t) at[k] * rb, rows[k].dst);
        return;
    }
    // the serial path (AI_Q4X_PLE_IO=0, or AI_Q4X_PLE_PREAD=0)
    // [AI_Q4X_PLE_PREFETCH, default 1] ask the kernel for EVERY row of the round before reading any (3.7 ms of host time
    // per V_v_bugs_dv round on the 3090 2026-10-25 before it). WILLNEED queues them all at once.
    static const bool ple_prefetch = !getenv("AI_Q4X_PLE_PREFETCH") || atoi(getenv("AI_Q4X_PLE_PREFETCH")) != 0;
    if (ple_prefetch) {
        for (const Row & w : rows) prog_.weights().host_prefetch(w.src, rb);
    }
    for (const Row & w : rows) {
        const char * src = w.src;
        char         row[8192];
        if (ple_pread && rb <= sizeof row && prog_.weights().host_pread(src, row, rb)) src = row;
        to_f32(src, w.dst);
    }
}

void Qwen4ExpRun::dump(const char * name, int il, const Tensor & t) {
    if (dump_dir_.empty() || t.type != F32) return;
    dev_.synchronize();
    std::vector<float> buf((size_t) t.numel());
    dev_.download(t, buf.data(), 0, buf.size() * 4);
    const std::string path = dump_dir_ + "/" + name + (il >= 0 ? "-" + std::to_string(il) : std::string()) + ".bin";
    // the first chunk of a sequence starts the file; later chunks append their rows and update the row count, so a
    // chunked run dumps the same [rows, ne0] matrix as the oracle's one-chunk prefill
    int64_t hdr[2] = { t.ne[0], t.numel() / t.ne[0] };
    FILE * f = nullptr;
    if (dump_append_) {
        f = fopen(path.c_str(), "r+b");
        int64_t old[2] = { 0, 0 };
        if (f && fread(old, 8, 2, f) == 2 && old[0] == hdr[0]) {
            hdr[1] += old[1];
            fseek(f, 0, SEEK_SET);
            fwrite(hdr, 8, 2, f);
            fseek(f, 0, SEEK_END);
        } else if (f) {
            fclose(f);
            f = nullptr;
        }
    }
    if (!f) {
        f = fopen(path.c_str(), "wb");
        if (!f) return;
        fwrite(hdr, 8, 2, f);
    }
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
}

// Routed experts through the expert pool (the same assembly as Qwen35Hybrid::moe_streamed): route on the device, read
// the selection back, page the distinct experts in groups of at most n_slots, and run each group with every row's
// non-member experts on distinct occupied filler slots at weight 0. Then the sigmoid-gated shared expert.
Tensor Qwen4ExpRun::moe_streamed(int il, const Tensor & x) {
    const Qwen4ExpSpec &   s = prog_.spec();
    const Qwen4ExpLayer &  L = prog_.layers()[(size_t) il];
    const int              k = s.n_expert_used;
    const int64_t          T = x.ne[1];
    op::MoeParams mp{ s.n_expert, k };
    op::MoeRoute  route = op::moe_route(dev_, x, L.moe.router, mp);
    std::vector<int32_t> sel((size_t) (k * T));
    dev_.download(route.sel, sel.data(), 0, sel.size() * 4);

    const ExpertPool::Layer & PL = pool_.layer(il);
    std::vector<char>    want((size_t) s.n_expert, 0);
    std::vector<int32_t> order;
    for (int32_t e : sel) if (!want[(size_t) e]) { want[(size_t) e] = 1; order.push_back(e); }
    const int    cap    = pool_.n_slots();
    const size_t groups = (order.size() + (size_t) cap - 1) / (size_t) cap;
    std::vector<int32_t> slots((size_t) (k * T));
    std::vector<float>   mask((size_t) (k * T));
    Tensor moe;
    for (size_t gi = 0; gi < groups; ++gi) {
        const size_t b = gi * (size_t) cap, e = std::min(order.size(), b + (size_t) cap);
        std::vector<int32_t> group(order.begin() + (long) b, order.begin() + (long) e);
        pool_.ensure(dev_, il, group);
        std::vector<char> in_group((size_t) s.n_expert, 0);
        for (int32_t ex : group) in_group[(size_t) ex] = 1;
        const int         bank = pool_.bank_slots(il);
        std::vector<char> used((size_t) bank, 0);
        for (int64_t row = 0; row < T; ++row) {
            const size_t r0 = (size_t) (row * k);
            std::fill(used.begin(), used.end(), 0);
            for (int j = 0; j < k; ++j) {
                const int32_t ex = sel[r0 + (size_t) j];
                if (in_group[(size_t) ex]) used[(size_t) PL.slot_of[(size_t) ex]] = 1;
            }
            int next = 0;
            for (int j = 0; j < k; ++j) {
                const int32_t ex  = sel[r0 + (size_t) j];
                const bool    hit = in_group[(size_t) ex] != 0;
                if (hit) {
                    slots[r0 + (size_t) j] = PL.slot_of[(size_t) ex];
                } else {
                    while (next < bank && (used[(size_t) next] || !pool_.occupied(il, next))) ++next;
                    if (next == bank) throw std::runtime_error("Qwen4ExpRun: no distinct filler slot");
                    used[(size_t) next]    = 1;
                    slots[r0 + (size_t) j] = next;
                }
                mask[r0 + (size_t) j] = hit ? 1.0f : 0.0f;
            }
        }
        Tensor st = dev_.ws_alloc(I32, k, T);
        dev_.upload(st, slots.data(), 0, slots.size() * 4);
        Tensor w = route.weights;
        if (groups > 1) {
            Tensor mk = dev_.ws_alloc(F32, 1, k, T);
            dev_.upload(mk, mask.data(), 0, mask.size() * 4);
            w = op::mul(dev_, route.weights, mk);
        }
        Tensor part = op::moe_experts(dev_, x, PL.up.pool, PL.gate.pool, PL.down.pool, st, w, k);
        moe = moe.valid() ? op::add(dev_, moe, part) : part;
    }
    return op::add(dev_, moe, op::moe_shared(dev_, x, L.moe));
}

// kDoorbellT (rows per doorbell window) lives in qwen4exp_run.h, shared with the prefill

// Static residency: per MoE layer, `slots` experts copied into device banks + n_used filler slots (copies of experts
// 0..k-1: valid weights, never a routed expert's slot). The resident set is the top of a DREP routing profile
// (AI_Q4X_PROFILE, alphallama's format) when given, else experts 0..slots-1. Misses go to the CPU pool.
void Qwen4ExpRun::init_doorbell(int slots) {
    const Qwen4ExpSpec & s = prog_.spec();
    const auto & layers = prog_.layers();
    const int k = s.n_expert_used, E = s.n_expert;
    std::vector<std::vector<uint32_t>> prof(layers.size());
    if (const char * pp = getenv("AI_Q4X_PROFILE")) {   // "DREP" u32 n_layers u64 n_tokens, per layer i32 blk i32 n u32[n]
        std::ifstream f(pp, std::ios::binary);
        uint32_t magic = 0, nl = 0;
        uint64_t nt = 0;
        f.read((char *) &magic, 4);
        f.read((char *) &nl, 4);
        f.read((char *) &nt, 8);
        for (uint32_t i = 0; f && magic == 0x50455244u && i < nl; ++i) {
            int32_t blk = 0, n = 0;
            f.read((char *) &blk, 4);
            f.read((char *) &n, 4);
            std::vector<uint32_t> u((size_t) std::max(n, 0));
            f.read((char *) u.data(), (std::streamsize) u.size() * 4);
            if (blk >= 0 && (size_t) blk < prof.size() && n == E) prof[(size_t) blk] = u;
        }
        fprintf(stderr, "[q4x] doorbell residency from profile %s\n", pp);
    }
    banks_.assign(layers.size(), {});
    // AI_Q4X_FILLERS (default 0): k filler rows per layer bank. Without them a missed column still gets a slot index
    // >= n_slots, but nothing reads it: the grouped experts kernel skips missed columns, so every doorbell width
    // (T = 1 included) must take it. Saves k expert copies per layer, i.e. roughly 10 more resident slots in the same
    // VRAM. T = 1 then takes the grouped kernel too: same products, but not the chain's ncols_dst == 1 reduction order.
    fillers_ = getenv("AI_Q4X_FILLERS") && atoi(getenv("AI_Q4X_FILLERS")) != 0;
    const int nb = slots + (fillers_ ? k : 0);
    const int L  = (int) layers.size();
    auto bank_bytes = [&](int il) {
        const auto & m = layers[(size_t) il].moe;
        return (m.gate_exps.nb[2] + m.up_exps.nb[2] + m.down_exps.nb[2]) * (size_t) nb + 3 * 256;
    };
    // [AI_Q4X_LEND=1] the last layers' banks go into their own buffer, big enough for a prefill's state and expert
    // scratch at the largest super-chunk (prefill() takes one chunk up to S + S/8): the prefill borrows it (lend_begin /
    // lend_end) instead of the auto slot count keeping that memory free through decode. The lent layers' experts are
    // non-resident during a prefill and copied back after it.
    static const bool lend_on = getenv("AI_Q4X_LEND") && atoi(getenv("AI_Q4X_LEND")) != 0;
    size_t lend_need = 0;
    int    nl        = 0;
    if (lend_on) {
        const int64_t sub = std::max<int64_t>(8, getenv("AI_Q4X_PREFILL_SUB") ? atoll(getenv("AI_Q4X_PREFILL_SUB")) : 256);
        int64_t       S   = std::max<int64_t>(8, getenv("AI_Q4X_PREFILL_SUPER") ? atoll(getenv("AI_Q4X_PREFILL_SUPER")) : 16384);
        S = (S + S / 8 + sub - 1) / sub * sub;
        size_t per_max = 0;
        for (const auto & l : layers) per_max = std::max(per_max, (size_t) (l.moe.gate_exps.nb[2] + l.moe.up_exps.nb[2] + l.moe.down_exps.nb[2]));
        lend_need = (pf_state_bytes(S, sub) + 255) / 256 * 256 + (size_t) E * per_max + 3 * 256;
        // [AI_Q4X_LEND2, default 1] room for a SECOND expert scratch too, so the prefill's layer stream is double-
        // buffered (layer l+1's experts copy under layer l's MoE as well as its mixer). Without it the 5090 at 311
        // slots had no VRAM for the second scratch: one buffer, the link idle 310 ms of a 2.6K-token prefill while the
        // GPU computed, and the GPU idle ~2.3 ms per layer waiting for the copies. Lends ~2 more layers' banks (no
        // extra VRAM; those layers' experts stream during a prefill and are copied back after it).
        static const bool lend2 = !getenv("AI_Q4X_LEND2") || atoi(getenv("AI_Q4X_LEND2")) != 0;
        if (lend2) lend_need += ((size_t) E * per_max + 3 * 256 + 255) / 256 * 256;
        // [AI_Q4X_PF_NBUF, default 2 (5090: 3 = V TTFT 1.21-1.25x vs 1.08x, R 1.14x vs 1.04x); 3 = a third scratch as well: the copy stream runs two layers ahead (5090 2.6K prompt:
        // with two, the link still sat idle 163 ms while the GPU computed). ~2 more lent layers, no extra VRAM.
        static const int nbuf = getenv("AI_Q4X_PF_NBUF") ? atoi(getenv("AI_Q4X_PF_NBUF")) : 2;
        if (lend2 && nbuf >= 3) lend_need += ((size_t) E * per_max + 3 * 256 + 255) / 256 * 256;
        size_t acc = 0;
        while (nl < L - 1 && acc < lend_need) acc += bank_bytes(L - 1 - nl++);
        if (acc < lend_need) nl = 0;
    }
    lend_from_ = nl > 0 ? L - nl : (1 << 30);
    size_t total = 0, lend_total = 0;
    for (int il = 0; il < L; ++il) {
        total += (size_t) E * 4 + 4096;
        (il >= lend_from_ ? lend_total : total) += bank_bytes(il);
    }
    const int buf  = dev_.alloc_buffer(total, "q4x.slots");
    char *    base = (char *) dev_.buffer_base(buf);
    size_t    off  = 0, loff = 0;
    char *    lbase = nullptr;
    if (nl > 0) {
        lend_bytes_ = std::max(lend_total, lend_need);
        lend_buf_   = dev_.alloc_buffer(lend_bytes_, "q4x.lend");
        lbase       = (char *) dev_.buffer_base(lend_buf_);
        fprintf(stderr, "[q4x] lend: layers %d..%d keep their banks in a %.0f MiB buffer a prefill borrows (needs %.0f MiB)%c",
                lend_from_, L - 1, lend_bytes_ / 1048576.0, lend_need / 1048576.0, 10);
    }
    auto bank = [&](const Tensor & h, bool lent) {
        Tensor t = h;
        t.buf    = lent ? lend_buf_ : buf;
        t.ne[2]  = nb;
        set_contiguous_strides(t);
        size_t & o = lent ? loff : off;
        o      = (o + 255) / 256 * 256;
        t.data = (lent ? lbase : base) + o;
        o     += t.nbytes();
        return t;
    };
    std::vector<offload::ExpertBanks> host;
    std::vector<double> layer_cold;   // RAM tier: per layer: profile mass of its initially non-resident experts
    double resident = 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & m = layers[il].moe;
        SlotBank &   B = banks_[il];
        B.gate    = bank(m.gate_exps, (int) il >= lend_from_);
        B.up      = bank(m.up_exps, (int) il >= lend_from_);
        B.down    = bank(m.down_exps, (int) il >= lend_from_);
        B.n_slots = slots;
        off       = (off + 255) / 256 * 256;
        B.table.buf  = buf;
        B.table.type = I32;
        B.table.ne[0] = E;
        set_contiguous_strides(B.table);
        B.table.data = base + off;
        off += (size_t) E * 4;
        std::vector<int32_t> order((size_t) E);
        for (int e = 0; e < E; ++e) order[(size_t) e] = e;
        if (!prof[il].empty()) {
            const auto & pr = prof[il];
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return pr[(size_t) a] > pr[(size_t) b]; });
            uint64_t all = 0, top = 0;
            for (int e = 0; e < E; ++e) all += pr[(size_t) e];
            for (int i = 0; i < slots; ++i) top += pr[(size_t) order[(size_t) i]];
            resident += all ? (double) top / (double) all : 0.0;
        }
        std::vector<int32_t> table((size_t) E, -1);
        B.owner.assign((size_t) nb, -1);
        B.score.assign((size_t) E, 0.0f);
        for (int sl = 0; sl < nb; ++sl) {
            const int e = sl < slots ? order[(size_t) sl] : sl - slots;   // fillers: copies of experts 0..k-1
            if (sl < slots) { table[(size_t) e] = sl; B.owner[(size_t) sl] = e; }
            const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
            const Tensor * dst[3] = { &B.gate, &B.up, &B.down };
            for (int r = 0; r < 3; ++r) {
                cudaMemcpy((char *) dst[r]->data + (size_t) sl * dst[r]->nb[2],
                           (const char *) src[r]->data + (size_t) e * src[r]->nb[2], src[r]->nb[2], cudaMemcpyHostToDevice);
            }
        }
        dev_.upload(B.table, table.data(), 0, table.size() * 4);
        B.table_h = table;
        init_owner_.push_back(std::vector<int32_t>(B.owner.begin(), B.owner.begin() + slots));
        host.push_back({ m.gate_exps, m.up_exps, m.down_exps });
        double cold = 0;
        for (int e = 0; e < E; ++e) if (table[(size_t) e] < 0) cold += prof[il].empty() ? 1.0 : (double) prof[il][(size_t) e];
        layer_cold.push_back(cold);
    }
    // RAM tier. AI_Q4X_RAM_TIER_GB=N (or "auto" = RAM - 20 GB; unset/0 = off): whole layers' expert tensors copied out
    // of the file mapping into pinned + mapped memory (core/host_mem.h), layers with the most non-resident profile mass
    // first, and the tensors re-pointed there. An admission copy from the tier is one DMA (no staging), and the CPU
    // misses read RAM the page cache cannot evict, which avoids the slow start from a cold page cache. It does not
    // prevent the compaction storms described in core/host_mem.h: pinned memory feeds them, so a start that lands in a
    // compaction-active period can still run slow.
    if (const char * tg = getenv("AI_Q4X_RAM_TIER_GB")) {
        size_t budget = 0;
        if (strcmp(tg, "auto") == 0) {
            long long mem_kb = 0;
            std::ifstream mi("/proc/meminfo");
            std::string key;
            while (mi >> key) { if (key == "MemTotal:") { mi >> mem_kb; break; } mi.ignore(1 << 10, '\n'); }
            const long long b = mem_kb * 1024 - (20ll << 30);
            budget = b > 0 ? (size_t) b : 0;
        } else {
            budget = (size_t) (atof(tg) * 1073741824.0);
        }
        std::vector<int> order_l(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) order_l[il] = (int) il;
        std::stable_sort(order_l.begin(), order_l.end(), [&](int a, int b) { return layer_cold[(size_t) a] > layer_cold[(size_t) b]; });
        tier_layer_.assign(layers.size(), 0);
        // AI_Q4X_RAM_TIER_PIN=0: the tier stays ordinary (movable) memory -- compaction can then actually move it, so it
        // never loops on it -- and admissions stage through the small pinned buffers instead of one DMA
        static const bool tier_pin = !getenv("AI_Q4X_RAM_TIER_PIN") || atoi(getenv("AI_Q4X_RAM_TIER_PIN")) != 0;
        auto & lm = prog_.layers_mut();
        // 1) the layers that fit, hottest first; 2) one block, each tensor placed at its file offset modulo 4096 so every
        // chunk is a single O_DIRECT read into place (no bounce copy, no page-cache copy); 3) register once at the end
        // (pins pages the reads already faulted in). Much faster than a per-tensor alloc + fault-in + register + bounce
        // copy.
        std::vector<int>  chosen;
        size_t            used = 0, block = 0;
        std::vector<size_t> place;   // per chosen tensor (3 per layer): its offset in the block
        for (int il : order_l) {
            const op::MoeWeights & m = lm[(size_t) il].moe;
            const Tensor * ts[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
            size_t need = 0;
            for (const Tensor * t : ts) need += t->nbytes();
            if (used + need > budget) break;
            for (const Tensor * t : ts) {
                const size_t fpo = prog_.weights().file_page_offset(t->data);
                if (fpo == SIZE_MAX) { need = SIZE_MAX; break; }
                place.push_back(block + fpo);
                block += (fpo + t->nbytes() + 4095) / 4096 * 4096;
            }
            if (need == SIZE_MAX) { place.resize(chosen.size() * 3); break; }
            chosen.push_back(il);
            used += need;
        }
        const auto t0 = std::chrono::steady_clock::now();
        bool direct = false, ok = !chosen.empty();
        double t_read = 0, t_reg = 0;
        if (ok) {
            tier_mem_ = pinned_host_reserve(block);
            ok = tier_mem_.ptr != nullptr;
        }
        if (ok) {
            std::vector<WeightSet::ReadJob> jobs;
            for (size_t ci = 0; ci < chosen.size(); ++ci) {
                const op::MoeWeights & m = lm[(size_t) chosen[ci]].moe;
                const Tensor * ts[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
                for (int r = 0; r < 3; ++r) jobs.push_back({ ts[r]->data, (char *) tier_mem_.ptr + place[ci * 3 + r], ts[r]->nbytes() });
            }
            static const int fill_threads = getenv("AI_Q4X_RAM_TIER_THREADS") ? atoi(getenv("AI_Q4X_RAM_TIER_THREADS")) : 16;
            ok = prog_.weights().read_into(jobs, fill_threads, &direct);
            t_read = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (ok && tier_pin) ok = pinned_host_register(tier_mem_);
            t_reg = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - t_read;
        }
        if (!ok) {
            pinned_host_free(tier_mem_);
            fprintf(stderr, "[q4x] RAM tier: reserve, fill or register failed -- no tier\n");
            chosen.clear();
            used = 0;
        }
        for (size_t ci = 0; ci < chosen.size(); ++ci) {
            const int il = chosen[ci];
            op::MoeWeights & m = lm[(size_t) il].moe;
            Tensor * ts[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
            for (int r = 0; r < 3; ++r) {
                tier_.push_back({ ts[r], ts[r]->data });
                ts[r]->data = (char *) tier_mem_.ptr + place[ci * 3 + r];
            }
            host[(size_t) il] = { m.gate_exps, m.up_exps, m.down_exps };
            tier_layer_[(size_t) il] = tier_pin ? 1 : 2;   // 1: pinned (admission DMAs from it), 2: unpinned (staged)
            if (tier_pin) {   // GPU miss share: the kernels read the mapped tier through these
                if (tier_dev_.size() < layers.size()) tier_dev_.assign(layers.size(), { nullptr, nullptr, nullptr });
                for (int r = 0; r < 3; ++r) {
                    void * dp = nullptr;
                    if (cudaHostGetDevicePointer(&dp, ts[r]->data, 0) != cudaSuccess) { (void) cudaGetLastError(); dp = nullptr; }
                    tier_dev_[(size_t) il][(size_t) r] = dp;
                }
            }
        }
        const int n_layers = (int) chosen.size();
        fprintf(stderr, "[q4x] RAM tier: %d of %zu layers' experts, %.2f GiB %s (budget %.2f GiB, %.1f%% huge pages), "
                        "read %.1f s%s + register %.1f s\n", n_layers, layers.size(), used / 1073741824.0,
                tier_pin ? "pinned + mapped" : "unpinned", budget / 1073741824.0, 100.0 * tier_mem_.thp, t_read,
                direct ? " (O_DIRECT, in place)" : "", t_reg);
        // the rest of the shard mappings (dense tensors read at load, the PLE table, untiered layers): drop this
        // process's page-table entries so compaction moving those page-cache pages cannot IPI the pool cpus
        static const bool drop = !getenv("AI_Q4X_DROP_PTES") || atoi(getenv("AI_Q4X_DROP_PTES")) != 0;
        if (drop && n_layers > 0) {
            const size_t d = prog_.weights().drop_mapping_ptes();
            fprintf(stderr, "[q4x] dropped this process's page-table entries over the %.2f GiB model mapping\n", d / 1073741824.0);
        }
        ple_warm();   // [AI_Q4X_PLE_WARM] after the cache drop: re-read the PLE table when RAM allows
    }
    const int wb = dev_.alloc_buffer(256, "q4x.winbase");
    win_base_.buf   = wb;
    win_base_.type  = I32;
    win_base_.ne[0] = 1;
    set_contiguous_strides(win_base_);
    win_base_.data = dev_.buffer_base(wb);
    const uint32_t zero = 0;
    dev_.upload(win_base_, &zero, 0, 4);
    cudaStreamCreateWithFlags((cudaStream_t *) &copy_stream_, cudaStreamNonBlocking);
    const size_t unit_max = (layers[0].moe.gate_exps.nb[2] + layers[0].moe.up_exps.nb[2] + layers[0].moe.down_exps.nb[2]) * 2;
    static const int n_stage = getenv("AI_Q4X_ADMIT_STAGE") ? atoi(getenv("AI_Q4X_ADMIT_STAGE")) : 80;
    // staging in huge-page pinned memory (core/host_mem.h: many small 4 KiB pinned pages are the main target of
    // compaction storms); layers the tier holds need no staging
    const size_t unit_al = (unit_max + 4095) / 4096 * 4096;
    staging_mem_ = pinned_host_alloc(unit_al * (size_t) n_stage, /*register_cuda=*/ true);
    if (!staging_mem_.ptr) throw std::runtime_error("q4x: pinned staging allocation failed");
    for (int i = 0; i < n_stage; ++i) {
        char * p = (char *) staging_mem_.ptr + unit_al * (size_t) i;
        staging_.push_back(p);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        staging_ev_.push_back((void *) ev);
    }
    recorded_.reset(new std::atomic<int64_t>[(size_t) n_stage]);
    for (int i = 0; i < n_stage; ++i) recorded_[(size_t) i].store(-1);
    cudaHostAlloc((void **) &table_pin_, layers.size() * (size_t) E * 4, cudaHostAllocDefault);
    for (size_t il = 0; il < layers.size(); ++il) memcpy(table_pin_ + il * (size_t) E, banks_[il].table_h.data(), (size_t) E * 4);
    admit_thr_ = std::thread([this] { admit_worker(); });
    std::vector<int> cpus;
    if (const char * cl = getenv("AI_Q4X_CPU_LIST")) {   // "0,2,4-7"
        for (const char * p = cl; *p;) {
            char *     e = nullptr;
            const long a = strtol(p, &e, 10);
            if (e == p) break;
            long b = a;
            if (*e == '-') { p = e + 1; b = strtol(p, &e, 10); }
            for (long c2 = a; c2 <= b; ++c2) cpus.push_back((int) c2);
            p = *e == ',' ? e + 1 : e;
        }
    }
    const int threads = getenv("AI_Q4X_CPU_THREADS") ? atoi(getenv("AI_Q4X_CPU_THREADS")) : 8;
    if (cpus.empty()) {   // default placement from the topology (offload/cpu_topo.h); AI_Q4X_CPU_LIST overrides it
        const offload::CpuPlan plan = offload::plan_cpus(getenv("AI_Q4X_CPU_THREADS") ? threads : 0);
        if (!plan.pool.empty()) {
            // the launch thread owns its physical core. [AI_Q4X_RESERVE_CORES] it is pinned at its first decode or
            // prefill call (pin_launch_thread), not here: in a server the thread that builds the engine is not the one
            // that decodes, and pinning the builder while the decoder was confined with ~38 other server threads onto
            // the launch core lost ~1/3 of decode (RTX 4080, AlphaServe: set A 71.9 vs ai-api-gen 110.3 tok/s)
            launch_cpus_ = plan.launch;
            host_cpus_   = plan.host;
            if (plan.host.empty()) offload::pin_this_thread(plan.launch);   // the old rule
            cpus = plan.pool;
        }
        fprintf(stderr, "[q4x] cpu placement: %s -> launch %s, pool %s\n", plan.why.c_str(),
                offload::cpu_list_str(plan.launch).c_str(), cpus.empty() ? "(unpinned)" : offload::cpu_list_str(cpus).c_str());
    }
    // every other thread (CUDA runtime, admission worker) off the spinning pool cpus: one landing there straggles every
    // pool phase for the whole run (offload/cpu_topo.h). AI_Q4X_CONFINE=0 turns it off.
    if (!cpus.empty() && (!getenv("AI_Q4X_CONFINE") || atoi(getenv("AI_Q4X_CONFINE")) != 0)) {
        const int n = host_cpus_.empty() ? offload::confine_threads_off(cpus) : offload::confine_threads_onto(host_cpus_);
        fprintf(stderr, "[q4x] confined %d other threads %s\n", n,
                host_cpus_.empty() ? "off the pool cpus" : ("onto the host cpus " + offload::cpu_list_str(host_cpus_)).c_str());
    }
    kern::offload_wait_stats_init();
    miss_ = std::make_unique<offload::MissPool>(host, s.n_embd, k, kDoorbellT, threads, cpus);
    fprintf(stderr, "[q4x] doorbell offload: %d resident + %d filler slots per layer (%.2f GiB), CPU pool %d threads%s; "
                    "profile mass resident %.1f%%\n", slots, fillers_ ? k : 0, total / 1073741824.0,
            cpus.empty() ? threads : (int) cpus.size(), cpus.empty() ? "" : " (pinned)",
            100.0 * resident / (double) layers.size());
}

Tensor Qwen4ExpRun::moe_doorbell(int il, const Tensor & x) {
    const Qwen4ExpSpec &  s  = prog_.spec();
    const Qwen4ExpLayer & L  = prog_.layers()[(size_t) il];
    const SlotBank &      B  = banks_[(size_t) il];
    const offload::Bell   bl = miss_->bell(il);
    const int     k = s.n_expert_used;
    const int64_t T = x.ne[1], N = x.ne[0];
    op::MoeParams mp{ s.n_expert, k };
    // AI_Q4X_ROUTE_NATIVE (default 1): router GEMV + softmax + top-k + renorm in 2 native launches
    // (src/kernels/q4x_route.cu) instead of the ggml chain's BF16 GEMM and converts; greedy tokens match the chain.
    // 0 = the ggml chain.
    static const bool route_native = !getenv("AI_Q4X_ROUTE_NATIVE") || atoi(getenv("AI_Q4X_ROUTE_NATIVE")) != 0;
    // [AI_Q4X_ROUTE_RING, default 1 since 2026-11-02: 3090 + 5090 wins vs Strata, 4080 with the other four +0.4/+1.3% vs the production config, TTFT -0.14 s; =0 off] the top-k kernel also does resolve_ring's work (kern::q4x_route_ring):
    // its T blocks write their rows' slots / miss ids / weights and x columns to the bells in parallel, the last block
    // rings. 3090 decode nsys (set A): softmax_topk 7.6 + resolve_ring 10.2 us per layer and round, vs Strata's one
    // ~4.4 us route_multi. Same top-k arithmetic: identical bits. Not with the GPU miss share or the wait-stats timing.
    static const bool ring_env = (!getenv("AI_Q4X_ROUTE_RING") || atoi(getenv("AI_Q4X_ROUTE_RING")) != 0) &&
                                 !getenv("AI_Q4X_WAIT_STATS") && !getenv("AI_Q4X_HANDOFF_STATS") && !getenv("AI_Q4X_HDR_TOUCH");
    static const int  ring_gm   = getenv("AI_Q4X_GPU_MISS") ? atoi(getenv("AI_Q4X_GPU_MISS")) : 0;
    static const bool ring_part = !getenv("AI_Q4X_PARTIAL") || atoi(getenv("AI_Q4X_PARTIAL")) != 0;
    Tensor        slots     = dev_.ws_alloc(I32, k, T);
    bool          ring_done = false;
    // [AI_Q4X_RING_PLAN=1, default 0: MEASURED NULL on the 3090 2026-11-01, rr7 verify +-0.1 ms; bit-exact, AI_Q4X_RING_CHECK
    // 0 mismatches] the experts call's Plan built by the ring kernel's last block (from
    // the slots it just wrote), so q4x_experts skips plan_k (3090: 5.6 us per layer and round). The scratch is
    // allocated here, before the route, and the experts call reuses it. Same owners, plan_k's routine: identical bits.
    // [AI_Q4X_RING_QX, default 0 until gated] also x's q8_1 copy for the experts (scratch + 0): one launch fewer per layer
    Tensor ring_scr;
    bool   ring_plan_ok = false;
    int    ring_ready   = 0;
    op::MoeRoute  route;
    const Tensor & rw = L.moe.router;
    const bool rnat = route_native && T <= kDoorbellT && (rw.type == BF16 || rw.type == F16) && rw.contiguous() &&
                      x.contiguous() && rw.ne[0] == N && rw.ne[1] == s.n_expert && s.n_expert <= 1024;
    // [AI_Q4X_EXP_FORK=2] fork BEFORE the routing: router GEMV, top-k, ring and the routed experts (all native) on the
    // backend's second stream, the shared expert (ggml) on the main stream, joined before the wait. EXP_FORK=1 forked
    // only the experts, after the ring (3090 CLI +2.7%); Strata overlaps its shared expert with the whole routed chain.
    static const int exp_fork_mode = getenv("AI_Q4X_EXP_FORK") ? atoi(getenv("AI_Q4X_EXP_FORK")) : 2;   // default 2 since 2026-11-02
    const bool fork2 = exp_fork_mode == 2 && ring_gm == 0 && rnat;
    if (fork2) {
        static std::vector<void *> evf2;
        while (evf2.size() < prog_.layers().size()) evf2.push_back(kern::event_create());
        void *   ef = evf2[(size_t) il];
        Device * dv = &dev_;
        adapter::native(dev_, [=]() {
            kern::event_record(ef, adapter::stream(*dv));
            kern::stream_wait_event(kern::backend_stream_no(dv->backend(), 1), ef);
        });
    }
    if (rnat) {
        Tensor        lgt = dev_.ws_alloc(F32, s.n_expert, T);
        route.sel         = dev_.ws_alloc(I32, k, T);
        route.weights     = dev_.ws_alloc(F32, 1, k, T);
        const void *  wp0 = rw.data;
        const bool    bf  = rw.type == BF16;
        const float * xp0 = (const float *) x.data;
        float *       lp  = (float *) lgt.data;
        int32_t *     sp0 = (int32_t *) route.sel.data;
        float *       wo  = (float *) route.weights.data;
        const int     E   = s.n_expert;
        Device *      dv  = &dev_;
        if (ring_env && ring_gm == 0) {
            const int32_t *  tb0 = (const int32_t *) B.table.data;
            int32_t *        sl0 = (int32_t *) slots.data;
            const unsigned * wb0 = (const unsigned *) win_base_.data;
            const int32_t    f0  = B.n_slots;
            float *          wbl = ring_part ? bl.d_w : nullptr;
            const int        lay = il;
            static const bool ring_plan_env = getenv("AI_Q4X_RING_PLAN") && atoi(getenv("AI_Q4X_RING_PLAN")) != 0;
            static const bool ring_qx_env = getenv("AI_Q4X_RING_QX") && atoi(getenv("AI_Q4X_RING_QX")) != 0;
            void *           rpl = nullptr, *rqx = nullptr;
            if (ring_plan_env && T * k <= 160 && (E + 31) / 32 * 32 >= T * k) {
                const int64_t Fp = B.up.ne[1];
                ring_scr     = dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(N, Fp, k, T) / 4) + 64, 1);
                rpl          = (char *) ring_scr.data + kern::q4x_experts_plan_offset(N, Fp, k, T);
                ring_plan_ok = true;
                ring_ready   = 1;
                if (ring_qx_env && N % 32 == 0 && x.nb[1] == (size_t) N * sizeof(float)) {
                    rqx = ring_scr.data;
                    ring_ready |= 2;
                }
            }
            const bool f2 = fork2;
            adapter::native(dev_, [=]() {
                if (!kern::q4x_route_ring(f2 ? kern::backend_stream_no(dv->backend(), 1) : adapter::stream(*dv), wp0, bf, xp0, (int) N, E, (int) T, k, lp, sp0, wo, tb0, f0, sl0,
                                          bl.d_miss, bl.d_x, bl.d_n_tok, bl.d_seq, wb0, lay, wbl, rpl, rqx)) {
                    fprintf(stderr, "[q4x] route+ring declined a shape the host check accepted\n");
                    abort();
                }
            });
            ring_done = true;
        } else {
            const bool f2 = fork2;
            adapter::native(dev_, [=]() { kern::q4x_route(f2 ? kern::backend_stream_no(dv->backend(), 1) : adapter::stream(*dv), wp0, bf, xp0, (int) N, E, (int) T, k, lp, sp0, wo); });
        }
    } else {
        route = op::moe_route(dev_, x, L.moe.router, mp);
    }
    const int32_t * sel   = (const int32_t *) route.sel.data;
    const float *   xp    = (const float *) x.data;
    const int32_t * tab   = (const int32_t *) B.table.data;
    int32_t *       sp    = (int32_t *) slots.data;
    const unsigned * wb   = (const unsigned *) win_base_.data;
    const int32_t filler0 = B.n_slots;
    const float *   wp    = (const float *) route.weights.data;
    static const bool partial = !getenv("AI_Q4X_PARTIAL") || atoi(getenv("AI_Q4X_PARTIAL")) != 0;   // default on
    // GPU miss share. AI_Q4X_GPU_MISS=m (default 0, opt-in): per layer, the GPU computes up to m distinct missed experts
    // itself, reading them from the mapped RAM tier inside the smem-staged grouped kernel while the CPU computes the
    // rest. Bulk zero-copy reads run at full PCIe link speed while the CPU's IQ gate/up is compute-bound, so host DRAM
    // has headroom for both. Fast mode like any residency change: a GPU-taken expert is computed by the GPU kernel
    // instead of the CPU one. Needs the grouped path, partial sums and no fillers. It is off by default because the
    // pull shares the link with the admission DMAs of the same round, which cancels the CPU time it saves.
    static const int gpu_miss = getenv("AI_Q4X_GPU_MISS") ? atoi(getenv("AI_Q4X_GPU_MISS")) : 0;
    static const int tail_rank_env = getenv("AI_Q4X_TAIL_RANK") ? atoi(getenv("AI_Q4X_TAIL_RANK")) : 8;
    const int64_t F_pre = B.up.ne[1];
    const bool gm_ok = gpu_miss > 0 && partial && !fillers_ && (size_t) il < tier_dev_.size() &&
                       tier_dev_[(size_t) il][0] && tier_dev_[(size_t) il][1] && tier_dev_[(size_t) il][2] &&
                       x.contiguous() && B.up.type == B.gate.type &&
                       kern::q4x_experts_ok((int) B.up.type, (int) B.down.type, N, F_pre, N, k, T);
    const int     gm   = gm_ok ? gpu_miss : 0;
    const int32_t gpu0 = gm > 0 ? filler0 + k : INT32_MAX;
    const int     trk  = tail_rank_env;
    Device * dev = &dev_;
    if (!ring_done) adapter::native(dev_, [=]() {
        kern::offload_resolve_ring(fork2 ? kern::backend_stream_no(dev->backend(), 1) : adapter::stream(*dev), tab, sel, xp, N, k, T, filler0, sp, bl.d_miss, bl.d_x,
                                   bl.d_n_tok, bl.d_seq, wb, il, wp, partial ? bl.d_w : nullptr, gm, gpu0, trk);
    });
    // GPU miss share: the host-tier pass forks onto the backend's second stream right after the ring, so its PCIe pull
    // overlaps the shared expert and the resident experts on the main stream (in series it costs more than the CPU
    // time it saves). It has its own scratch, writes only its own columns of the exps buffer the resident pass fills,
    // and is joined before the wait.
    Tensor gm_exps, gm_scr;
    void * gm_join = nullptr;
    if (gm > 0) {
        if (getenv("AI_Q4X_GROUPED") && atoi(getenv("AI_Q4X_GROUPED")) == 2) throw std::runtime_error("q4x doorbell: AI_Q4X_GPU_MISS is not supported with AI_Q4X_GROUPED=2");
        static std::vector<void *> evs;
        if (evs.size() < 2 * (size_t) prog_.layers().size()) {
            while (evs.size() < 2 * (size_t) prog_.layers().size()) evs.push_back(kern::event_create());
        }
        void * ev_fork = evs[(size_t) (2 * il)];
        gm_join        = evs[(size_t) (2 * il + 1)];
        gm_exps        = dev_.ws_alloc(F32, N, k, T);
        gm_scr         = dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(N, F_pre, k, T) / 4) + 64, 1);
        const Qwen4ExpLayer & LL = prog_.layers()[(size_t) il];
        const void *  hu = tier_dev_[(size_t) il][1], *hg = tier_dev_[(size_t) il][0], *hd = tier_dev_[(size_t) il][2];
        const int64_t hrg = (int64_t) LL.moe.up_exps.nb[1], heg = (int64_t) LL.moe.up_exps.nb[2];
        const int64_t hrd = (int64_t) LL.moe.down_exps.nb[1], hed = (int64_t) LL.moe.down_exps.nb[2];
        const int32_t lo = gpu0, hi = gpu0 + (int32_t) s.n_expert;
        const int     tg2 = (int) B.up.type, td2 = (int) B.down.type;
        const int64_t xs2 = (int64_t) x.nb[1] / (int64_t) sizeof(float), F2 = F_pre;
        void *        ps2 = gm_scr.data;
        float *       pm2 = (float *) gm_exps.data;
        void *        ej  = gm_join;
        static bool said = false;
        if (!said) { said = true; fprintf(stderr, "[q4x] GPU miss share: up to %d missed experts per layer from the mapped RAM tier, on a forked stream (AI_Q4X_GPU_MISS)\n", gm); }
        adapter::native(dev_, [=]() {
            void * st_main = adapter::stream(*dev);
            void * st_side = kern::backend_stream_no(dev->backend(), 1);
            kern::event_record(ev_fork, st_main);
            kern::stream_wait_event(st_side, ev_fork);
            if (!kern::q4x_experts_range(st_side, tg2, hu, hg, hrg, heg, td2, hd, hrd, hed, xp, xs2, sp, k, lo, hi, N, F2,
                                         N, k, T, ps2, pm2)) {
                fprintf(stderr, "[q4x] GPU miss share: the host-bank kernel declined the layer's shape\n");
                abort();
            }
            kern::event_record(ej, st_side);
        });
    }
    // [AI_Q4X_EXP_FORK, default 2 since 2026-11-02 (same windows as ROUTE_RING); =1 experts only, =0 off] the resident routed experts (the grouped kernel) on the backend's
    // second stream, forked right after the ring, while the shared expert runs on the main stream; joined before the
    // wait. Strata's decode runs its routed experts on their own stream concurrent with the trunk (5090 nsys:
    // ~2.2 ms/window of overlap where we have 0.1); our round is GPU-bound on the 3090 (spin 1.6 of ~24 ms) with
    // latency-bound expert / shared kernels. Same kernels on the same data: identical bits. Own scratch, as the gm fork.
    // MEASURED ~NULL (4080 2026-10-31, card 4K, MTP 2, 3 reps): md5 identical 6/6; 96.1 vs 94.4 tok/s; GPU-only bound
    // (NO_WAIT + SKIP_MISS) verify 14.83 -> 14.68 ms. The shared expert is too small to hide much; Strata's overlap
    // must reach the trunk, not just the shared expert. Opt-in.
    static const bool exp_fork_env = !getenv("AI_Q4X_EXP_FORK") || atoi(getenv("AI_Q4X_EXP_FORK")) != 0;
    const bool exp_fork = exp_fork_env && gm == 0;
    if (fork2 && gm > 0) throw std::runtime_error("q4x doorbell: AI_Q4X_EXP_FORK=2 with the GPU miss share");
    void *     ex_join  = nullptr;
    Tensor     shared;
    if (!exp_fork) shared = op::moe_shared(dev_, x, L.moe);   // runs while the CPU computes the misses
    // [P4] AI_Q4X_GROUPED: routed experts grouped by slot (src/kernels/q4x_experts.cu) -- each resident slot's weights
    // unpacked once for every window row routed to it, gate+up+SwiGLU in one launch, missed columns skipped.
    // 1 = native (default); 0 = the ggml chain; 2 = compare: run both, generate from the chain, count bitwise diffs.
    // For T >= 2 it is bit-identical to the chain and skips the chain's wasted filler columns.
    // With fillers, T = 1 stays on the chain unless AI_Q4X_GROUPED_T1=1: ggml-cuda's ncols_dst == 1 MMVQ reduces in a
    // different order, so a small fraction of values differ at T = 1.
    static const int grouped = getenv("AI_Q4X_GROUPED") ? atoi(getenv("AI_Q4X_GROUPED")) : 1;
    static const bool grouped_t1 = getenv("AI_Q4X_GROUPED_T1") && atoi(getenv("AI_Q4X_GROUPED_T1")) != 0;
    const int64_t    F       = B.up.ne[1];
    const bool gok = grouped != 0 && (T >= 2 || grouped_t1 || !fillers_) && x.contiguous() && B.up.type == B.gate.type &&
                     B.up.nb[1] == B.gate.nb[1] && B.up.nb[2] == B.gate.nb[2] && B.down.ne[0] == F && B.down.ne[1] == N &&
                     kern::q4x_experts_ok((int) B.up.type, (int) B.down.type, N, F, N, k, T);
    Tensor exps;
    if (gok) {
        Tensor        mine = gm > 0 ? gm_exps : dev_.ws_alloc(F32, N, k, T);
        Tensor        scr  = ring_plan_ok ? ring_scr : dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(N, F, k, T) / 4) + 64, 1);
        const int     plan_rdy = ring_ready;
        const int     tg = (int) B.up.type, td = (int) B.down.type;
        const void *  pu = B.up.data, *pg = B.gate.data, *pd = B.down.data;
        const int64_t rbg = (int64_t) B.up.nb[1], ebg = (int64_t) B.up.nb[2];
        const int64_t rbd = (int64_t) B.down.nb[1], ebd = (int64_t) B.down.nb[2];
        const int64_t xs  = (int64_t) x.nb[1] / (int64_t) sizeof(float);
        const int32_t * spc = sp;
        void *        ps  = scr.data;
        float *       pm  = (float *) mine.data;
        void *        ev_f = nullptr;
        if (exp_fork && grouped != 2) {
            static std::vector<void *> evx;
            while (evx.size() < 2 * prog_.layers().size()) evx.push_back(kern::event_create());
            ev_f    = evx[(size_t) (2 * il)];
            ex_join = evx[(size_t) (2 * il + 1)];
        }
        void * ej_x = ex_join;
        adapter::native(dev_, [=]() {
            void * st = adapter::stream(*dev);
            if (ej_x) {
                void * st_side = kern::backend_stream_no(dev->backend(), 1);
                kern::event_record(ev_f, st);
                kern::stream_wait_event(st_side, ev_f);
                st = st_side;
            }
            if (!kern::q4x_experts(st, tg, pu, pg, rbg, ebg, td, pd, rbd, ebd, xp, xs, spc, k, filler0,
                                   N, F, N, k, T, ps, pm, plan_rdy)) {
                fprintf(stderr, "[q4x-experts] declined a shape the host check accepted\n");
                abort();
            }
            if (ej_x) kern::event_record(ej_x, st);
        });

        if (grouped == 2) {   // the chain computes missed columns on the filler rows: compare needs AI_Q4X_FILLERS=1
            if (!fillers_) throw std::runtime_error("q4x doorbell: AI_Q4X_GROUPED=2 (compare) needs AI_Q4X_FILLERS=1");
            exps             = op::q4x::moe_experts_raw(dev_, x, B.up, B.gate, B.down, slots);
            const float * pr = (const float *) exps.data;
            adapter::native(dev_, [=]() { kern::q4x_experts_compare(adapter::stream(*dev), pm, pr, spc, k, k, filler0, N, T); });
        } else {
            exps = mine;
        }
    } else {
        // the chain computes missed columns on the filler rows: without them it would read past the bank
        if (!fillers_) throw std::runtime_error("q4x doorbell: AI_Q4X_FILLERS=0 needs the grouped experts kernel");
        if (fork2) {   // [AI_Q4X_EXP_FORK=2] the slots came from the side stream: join before the (main-stream) chain
            static std::vector<void *> evc;
            while (evc.size() < prog_.layers().size()) evc.push_back(kern::event_create());
            void * ec = evc[(size_t) il];
            adapter::native(dev_, [=]() {
                kern::event_record(ec, kern::backend_stream_no(dev->backend(), 1));
                kern::stream_wait_event(adapter::stream(*dev), ec);
            });
        }
        exps = op::q4x::moe_experts_raw(dev_, x, B.up, B.gate, B.down, slots);
    }
    if (exp_fork) {
        if (!shared.valid()) shared = op::moe_shared(dev_, x, L.moe);   // overlaps the routed experts on the side stream
        if (ex_join) {
            void * ej = ex_join;
            adapter::native(dev_, [=]() { kern::stream_wait_event(adapter::stream(*dev), ej); });
        }
    }
    float * ep  = (float *) exps.data;
    float * shp = (float *) shared.data;
    // Debug: AI_Q4X_NO_WAIT=1 skips the wait kernel (output is wrong) to time the GPU-only round
    static const bool no_wait = getenv("AI_Q4X_NO_WAIT") && atoi(getenv("AI_Q4X_NO_WAIT")) != 0;
    if (gm_join) {
        void * ej = gm_join;
        adapter::native(dev_, [=]() { kern::stream_wait_event(adapter::stream(*dev), ej); });
    }
    if (!no_wait) adapter::native(dev_, [=]() {
        if (partial) {
            kern::offload_wait_partial(adapter::stream(*dev), bl.d_done, wb, il, sp, filler0, gpu0, bl.d_yp, ep, shp, N, k, T);
        } else {
            kern::offload_wait_fill(adapter::stream(*dev), bl.d_done, wb, il, bl.d_miss, bl.d_y, ep, N, k, T);
        }
    });
    static const bool wsum_native = !getenv("AI_Q4X_WSUM_NATIVE") || atoi(getenv("AI_Q4X_WSUM_NATIVE")) != 0;
    if (wsum_native && exps.contiguous() && route.weights.contiguous() && shared.contiguous()) {
        Tensor        out = dev_.ws_alloc(F32, N, T);
        const float * we  = (const float *) route.weights.data;
        const float * sh  = (const float *) shared.data;
        float *       op  = (float *) out.data;
        adapter::native(dev_, [=]() { kern::moe_wsum(adapter::stream(*dev), ep, we, sh, op, N, k, T); });
        return out;
    }
    return op::q4x::moe_weighted_sum(dev_, exps, route.weights, shared, k);
}

// Persistent record areas, sized once for kDoorbellT rows: per GDN layer conv_input [K-1+T, C], conv_out [C, T],
// beta|alpha [2*Hv, T]; per PLE layer the padded history [hist + T, D].
void Qwen4ExpRun::keep_record(int il, op::GdnRecord * gr, const Tensor * ple_padded) {
    const Qwen4ExpSpec & s = prog_.spec();
    const auto & layers = prog_.layers();
    if (rec_buf_ < 0) {
        const int64_t Tm = kDoorbellT, K1 = gp_.conv_k - 1, C = gp_.channels, Hv = gp_.v_heads;
        const int64_t hist = (int64_t) (pp_.conv_k - 1) * pp_.ngram, D = s.hc_dim();
        rec_off_.assign(layers.size(), 0);
        size_t off = 0;
        for (size_t l = 0; l < layers.size(); ++l) {
            rec_off_[l] = off;
            if (!layers[l].attn) off += (((size_t) ((K1 + Tm) * C + C * Tm + 2 * Hv * Tm) * 4) + 4 * 256) / 256 * 256;
            if (layers[l].ple)   off += (((size_t) ((hist + Tm) * D) * 4) + 256) / 256 * 256;
        }
        rec_buf_ = dev_.alloc_buffer(off + 256, "q4x.verify_records");
        for (int t = 0; t <= (int) Tm; ++t) {
            grec_[t].assign(layers.size(), {});
            prec_[t].assign(layers.size(), {});
        }
        fprintf(stderr, "[q4x] verify records: %.1f MiB\n", off / 1048576.0);
    }
    const int64_t Tm = kDoorbellT, K1 = gp_.conv_k - 1, C = gp_.channels, Hv = gp_.v_heads;
    auto r256 = [](size_t v) { return (v + 255) / 256 * 256; };
    // fixed field offsets inside the layer's area (independent of T)
    const size_t f_ci = 0, f_co = r256((size_t) ((K1 + Tm) * C) * 4), f_b = f_co + r256((size_t) (C * Tm) * 4),
                 f_a = f_b + r256((size_t) (Hv * Tm) * 4), f_ple = f_b + r256((size_t) (2 * Hv * Tm) * 4) + 256;
    char * base = (char *) dev_.buffer_base(rec_buf_) + rec_off_[(size_t) il];
    // [AI_Q4X_REC_SPANS, default 1; =0 the per-tensor ggml copies] the layer's record copies as ONE copy_spans launch
    // when every source is contiguous f32 (5090 nsys 11-02: the 4 ggml copy nodes left a ~6 us hole on the main stream
    // after every GDN block). Plain copies: identical bits.
    static const bool spans_env = !getenv("AI_Q4X_REC_SPANS") || atoi(getenv("AI_Q4X_REC_SPANS")) != 0;
    kern::CopySpans   cs{};
    bool              spans_ok = spans_env;
    std::vector<std::pair<Tensor, Tensor>> pend;
    auto keep = [&](const Tensor & src, size_t field, const char * who) {
        Tensor t = src;
        t.buf    = rec_buf_;
        set_contiguous_strides(t);
        t.data = base + field;
        (void) who;
        pend.push_back({ src, t });
        if (!(src.type == F32 && src.contiguous()) || pend.size() > 8) spans_ok = false;
        return t;
    };
    auto flush = [&]() {
        if (spans_ok) {
            for (const auto & pr : pend) {
                cs.src[cs.count] = (const float *) pr.first.data;
                cs.dst[cs.count] = (float *) pr.second.data;
                cs.n[cs.count]   = pr.first.ne[0] * pr.first.ne[1] * pr.first.ne[2] * pr.first.ne[3];
                cs.count++;
            }
            const kern::CopySpans c = cs;
            Device *              dv = &dev_;
            adapter::native(dev_, [=]() { kern::copy_spans(adapter::stream(*dv), c); });
        } else {
            for (const auto & pr : pend) op::copy_into(dev_, pr.first, pr.second, "q4x rec");
        }
    };
    const int64_t T = rec_T_;
    if (gr) {
        op::GdnRecord & d = grec_[T][(size_t) il];
        d = *gr;
        d.conv_input = keep(gr->conv_input, f_ci, "q4x rec conv_input");
        d.conv_out   = keep(gr->conv_out, f_co, "q4x rec conv_out");
        if (gr->ba.valid()) {
            d.ba = keep(gr->ba, f_b, "q4x rec ba");
        } else {
            d.beta  = keep(gr->beta, f_b, "q4x rec beta");
            d.alpha = keep(gr->alpha, f_a, "q4x rec alpha");
        }
    }
    if (ple_padded) prec_[T][(size_t) il] = keep(*ple_padded, gr ? f_ple : 0, "q4x rec ple");
    flush();
}

Tensor Qwen4ExpRun::verify(const int32_t * tokens, int64_t T) {
    if (rec_T_ > 0) throw std::runtime_error("Qwen4ExpRun::verify: previous verify not committed");
    if (T < 1 || T > kDoorbellT) throw std::runtime_error("Qwen4ExpRun::verify: 1 <= T <= kDoorbellT");
    auto tv = std::chrono::steady_clock::now();
    admit_after_round();   // as decode(): the previous round has completed (its caller read the logits)
    if (g_rt.on) g_rt.admit += rt_ms_since(tv);
    rec_pos0_ = (int64_t) hist_.size();
    rec_T_    = T;
    rec_      = true;
    static const bool cap = getenv("AI_Q4X_CAPTURE") && atoi(getenv("AI_Q4X_CAPTURE"));
    // Debug: AI_Q4X_STATE_CK=1 checksums the first GDN layer's ssm + conv state around the verify (must not change)
    static const bool ck = getenv("AI_Q4X_STATE_CK") && atoi(getenv("AI_Q4X_STATE_CK"));
    auto cksum = [&](const char * when) {
        for (size_t il = 0; il < gdn_.size(); ++il) {
            if (!gdn_[il].ssm.valid()) continue;
            dev_.synchronize();
            std::vector<float> a((size_t) gdn_[il].ssm.numel()), c((size_t) gdn_[il].conv.numel());
            dev_.download(gdn_[il].ssm, a.data(), 0, a.size() * 4);
            dev_.download(gdn_[il].conv, c.data(), 0, c.size() * 4);
            double sa = 0, sc = 0;
            for (float v : a) sa += v;
            for (float v : c) sc += v;
            fprintf(stderr, "[q4x-ck] %s layer %zu: ssm sum %.9g conv sum %.9g\n", when, il, sa, sc);
            break;
        }
    };
    if (ck) cksum("before verify");
    Tensor lg;
    try {
        lg = cap ? round(tokens, T) : chunk(tokens, T);
    } catch (...) {
        rec_ = false;
        throw;
    }
    rec_ = false;
    if (ck) cksum("after verify ");
    return lg;
}

// [AI_Q4X_VERIFY_EARLY, default 0: neutral on the 5090 (sets A + V), the real gap is ~0.25 ms] draft + verify without the GPU idling between them: the chain (fill + n steps) is
// launched, then the verify of pending + n drafts right behind it on the stream -- its ids are copied from the chain's
// output on the device, and the PLE gate worker reads the drafts back after the chain's event (into hist_, before
// hashing its rows), so the host neither syncs on the chain nor holds the launch for it. 5090 nsys: the host's read-back
// + the 2846-node graph launch left the GPU idle ~0.4 ms a round (1.36 ms under nsys) between draft_next_k and the
// verify. Width is always n + 1: only with draft min-p off (no host cut). Only for a CAPTURED verify profile: an eager
// (recording or first) launch could run host steps that wait on the GPU, which sits at the PLE gate until the drafts
// arrive.
bool Qwen4ExpRun::draft_verify_early(int32_t pending, int n, std::vector<int32_t> & drafts, Tensor & lg) {
    static const bool on        = getenv("AI_Q4X_VERIFY_EARLY") && atoi(getenv("AI_Q4X_VERIFY_EARLY")) != 0;
    static const bool capture   = !getenv("AI_Q4X_MTP_CAPTURE") || atoi(getenv("AI_Q4X_MTP_CAPTURE")) != 0;
    static const bool trunk_cap = getenv("AI_Q4X_CAPTURE") && atoi(getenv("AI_Q4X_CAPTURE")) != 0;
    static const bool gate_env  = !getenv("AI_Q4X_PLE_GATE") || atoi(getenv("AI_Q4X_PLE_GATE")) != 0;
    const Qwen4ExpSpec & s = prog_.spec();
    static long long n_ok = 0, n_why[6] = { 0, 0, 0, 0, 0, 0 };
    auto no = [&](int why) {   // AI_Q4X_ROUND_TIMING: why the early path was declined (logged every 512 declines)
        if (g_rt.on && (++n_why[why] & 511) == 1)
            fprintf(stderr, "[q4x] verify early: declined (%d) x%lld; taken %lld\n", why, n_why[why], n_ok);
        return false;
    };
    if (!on || !mtp_on_ || !capture || !trunk_cap || !doorbell_ || !gate_env || s.ple_layers.empty()) return no(0);
    if (n < 1 || n > kMtpChain || n + 1 > kDoorbellT || mtp_minp() > 0 || rec_T_ > 0) return no(1);
    const int64_t pos0 = (int64_t) hist_.size(), T = n + 1;
    if (pos0 + T > max_ctx_ - 1) return no(2);
    int64_t F = 256;
    while (F < pos0 + T) F *= 2;
    F = std::min(F, max_ctx_);
    const auto * e = rounds_.find((F << 6) | (T << 1) | 1);
    if (!e || e->launches < 2) return no(3);
    if (g_rt.on && (++n_ok & 1023) == 1) fprintf(stderr, "[q4x] verify early: taken x%lld\n", n_ok);
    if (!ple_thr_.joinable()) ple_gate_init();
    if (!ple_ev_chain_) {
        cudaEventCreateWithFlags((cudaEvent_t *) &ple_ev_chain_, cudaEventDisableTiming);
        cudaStreamCreateWithFlags((cudaStream_t *) &ple_side_, cudaStreamNonBlocking);
        cudaHostAlloc((void **) &draft_pin_, (size_t) kMtpChain * 4, cudaHostAllocDefault);
    }
    int32_t ct[kMtpChain];
    float   cp[kMtpChain];
    mtp_chain(pending, n, ct, cp, false);
    cudaEventRecord((cudaEvent_t) ple_ev_chain_, (cudaStream_t) adapter::stream(dev_));
    admit_after_round();   // as verify(): the previous round has completed (its caller read the logits)
    rec_pos0_ = pos0;
    rec_T_    = T;
    rec_      = true;
    std::vector<int32_t> toks((size_t) T, 0);
    toks[0] = pending;
    try {
        lg = round(toks.data(), T, true);
    } catch (...) {
        rec_ = false;
        throw;
    }
    rec_ = false;
    {
        std::unique_lock<std::mutex> lk(ple_mu_);
        ple_cv_.wait(lk, [&]() { return ple_early_done_; });
        ple_early_done_ = false;
    }
    drafts.assign(hist_.begin() + pos0 + 1, hist_.begin() + pos0 + T);
    return true;
}

void Qwen4ExpRun::commit(int64_t m1) {
    if (rec_T_ <= 0 || m1 < 1 || m1 > rec_T_) throw std::runtime_error("Qwen4ExpRun::commit: no verify, or m1 out of range");
    ple_gate_idle();   // the worker read hist_ for this round's rows: done before the rejected rows are dropped
    auto tc = std::chrono::steady_clock::now();
    if (g_rt.on) g_rt.cmark(0, adapter::stream(dev_));
    struct CommitDone {   // the closing mark + host time on every exit of the folds below
        std::chrono::steady_clock::time_point & t; Device & d;
        ~CommitDone() { if (g_rt.on) { g_rt.cmark(1, adapter::stream(d)); g_rt.ch += rt_ms_since(t); } }
    } commit_done{ tc, dev_ };
    const auto & layers = prog_.layers();
    static const int  rec_mask_all = getenv("AI_Q4X_REC_MASK") ? atoi(getenv("AI_Q4X_REC_MASK")) : 3;
    // [AI_Q4X_FOLD_BATCH, default 1] every layer's fold in 3 launches (kernels/q4x_fold.cu). 3090 2026-10-28, 4K card:
    // commit GPU 0.70 -> 0.34 ms per cycle, fixed-T=3 decode 103.8 -> 105.8 tok/s, output md5 identical (2 reps x 2 configs)
    static const bool fold_batch   = !getenv("AI_Q4X_FOLD_BATCH") || atoi(getenv("AI_Q4X_FOLD_BATCH")) != 0;
    const bool batched = fold_batch && rec_mask_all == 3 && fold_batch_commit(m1);
    for (size_t il = 0; il < layers.size() && !batched; ++il) {
        Device::Scope sc(dev_);
        static const int rec_mask = getenv("AI_Q4X_REC_MASK") ? atoi(getenv("AI_Q4X_REC_MASK")) : 3;
        if (!layers[il].attn && (rec_mask & 1)) op::gdn_fold(dev_, layers[il].gdn, gp_, gdn_[il], grec_[rec_T_][il], m1);
        if (layers[il].ple && (rec_mask & 2)) op::q4x::ple_fold(dev_, pp_, ple_[il], prec_[rec_T_][il], m1);
    }
    hist_.resize((size_t) (rec_pos0_ + m1));
    rec_T_ = 0;
    if (mtp_on_) { mtp_fill_p0_ = rec_pos0_; mtp_fill_m_ = m1; }
    static const bool ck = getenv("AI_Q4X_STATE_CK") && atoi(getenv("AI_Q4X_STATE_CK"));
    if (ck) {
        for (size_t il = 0; il < gdn_.size(); ++il) {
            if (!gdn_[il].ssm.valid()) continue;
            dev_.synchronize();
            std::vector<float> a((size_t) gdn_[il].ssm.numel()), c((size_t) gdn_[il].conv.numel());
            dev_.download(gdn_[il].ssm, a.data(), 0, a.size() * 4);
            dev_.download(gdn_[il].conv, c.data(), 0, c.size() * 4);
            double sa = 0, sc = 0;
            for (float v : a) sa += v;
            for (float v : c) sc += v;
            fprintf(stderr, "[q4x-ck] after commit  layer %zu: ssm sum %.9g conv sum %.9g\n", il, sa, sc);
            break;
        }
    }
}

bool Qwen4ExpRun::fold_batch_commit(int64_t m1) {
    const int64_t T = rec_T_;
    if (T < 1 || T > kDoorbellT || fold_state_[T] < 0) return false;
    const auto & layers = prog_.layers();
    const int    nl = (int) layers.size();
    const int    Sk = gp_.head_k, Hk = gp_.k_heads, Sv = gp_.head_v, Hv = gp_.v_heads, C = gp_.channels, K1 = gp_.conv_k - 1;
    const size_t tsz = (size_t) nl * (sizeof(kern::FoldGdn) + sizeof(kern::FoldCopy)) + 256;
    if (fold_buf_ < 0) {
        int ng = 0;
        for (const auto & L : layers) ng += L.attn ? 0 : 1;
        const size_t scr = kern::q4x_fold_scratch_floats(ng, Sk, Hk, Hv, (int) kDoorbellT) * 4;
        fold_buf_  = dev_.alloc_buffer(tsz * (kDoorbellT + 1) + scr + 256, "q4x.fold_tables");
        fold_base_ = (char *) dev_.buffer_base(fold_buf_);
        fold_scr_  = (float *) (fold_base_ + (tsz * (kDoorbellT + 1) + 255) / 256 * 256);
    }
    if (fold_state_[T] == 0) {
        std::vector<kern::FoldGdn>  g;
        std::vector<kern::FoldCopy> cp;
        bool ok = Sv == 128;
        for (int il = 0; il < nl && ok; ++il) {
            const auto & L = layers[(size_t) il];
            if (!L.attn) {
                const op::GdnRecord & r = grec_[T][(size_t) il];
                const op::GdnState &  s = gdn_[(size_t) il];
                ok = r.conv_out.valid() && r.conv_out.type == F32 && r.conv_input.valid() && r.conv_input.type == F32 && r.B == 1 &&
                     s.ssm.valid() && s.conv.valid() && L.gdn.dt_bias.type == F32 && L.gdn.a.type == F32 &&
                     (size_t) r.conv_out.nb[1] == (size_t) C * 4;
                if (!ok) break;
                kern::FoldGdn e{};
                e.co = (const float *) r.conv_out.data;
                if (r.ba.valid()) {
                    ok = r.ba.type == F32;
                    e.beta = (const float *) r.ba.data;
                    e.alpha = (const float *) r.ba.data + Hv;
                    e.sb1 = 1;
                    e.sb2 = (int64_t) (r.ba.nb[1] / 4);
                } else {
                    ok = r.beta.valid() && r.alpha.valid() && r.beta.type == F32 && r.alpha.type == F32 && r.beta.nb[1] == r.alpha.nb[1] &&
                         r.beta.nb[2] == r.alpha.nb[2];
                    e.beta = (const float *) r.beta.data;
                    e.alpha = (const float *) r.alpha.data;
                    e.sb1 = (int64_t) (r.beta.nb[1] / 4);
                    e.sb2 = (int64_t) (r.beta.nb[2] / 4);
                }
                e.ssm = (float *) s.ssm.data;
                e.dt  = (const float *) L.gdn.dt_bias.data;
                e.a   = (const float *) L.gdn.a.data;
                g.push_back(e);
                cp.push_back({ (const float *) r.conv_input.data, (int64_t) (r.conv_input.nb[1] / 4), (float *) s.conv.data, K1, C });
            }
            if (L.ple) {
                const Tensor & pd = prec_[T][(size_t) il];
                const int hist = (pp_.conv_k - 1) * pp_.ngram;
                ok = ok && pd.valid() && pd.type == F32 && ple_[(size_t) il].hist.valid();
                if (ok) cp.push_back({ (const float *) pd.data, (int64_t) (pd.nb[1] / 4), (float *) ple_[(size_t) il].hist.data, hist, (int) pd.ne[1] });
            }
        }
        if (!ok) {
            fold_state_[T] = -1;
            fprintf(stderr, "[q4x] AI_Q4X_FOLD_BATCH: width %lld not batchable, per-layer folds\n", (long long) T);
            return false;
        }
        fold_off_[T] = tsz * (size_t) T;
        dev_.synchronize();   // a plain copy into memory the stream does not use yet
        cudaMemcpy(fold_base_ + fold_off_[T], g.data(), g.size() * sizeof(kern::FoldGdn), cudaMemcpyHostToDevice);
        cudaMemcpy(fold_base_ + fold_off_[T] + (size_t) nl * sizeof(kern::FoldGdn), cp.data(), cp.size() * sizeof(kern::FoldCopy),
                   cudaMemcpyHostToDevice);
        fold_ng_[T] = (int) g.size();
        fold_nc_[T] = (int) cp.size();
        fold_state_[T] = 1;
    }
    const kern::FoldGdn *  dg = (const kern::FoldGdn *) (fold_base_ + fold_off_[T]);
    const kern::FoldCopy * dc = (const kern::FoldCopy *) (fold_base_ + fold_off_[T] + (size_t) nl * sizeof(kern::FoldGdn));
    const int ng = fold_ng_[T], nc = fold_nc_[T], mi = (int) m1;
    float *   scr = fold_scr_;
    const float eps = gp_.eps;
    Device *  dev = &dev_;
    // [AI_Q4X_FOLD_SIDE, default 1] on a side stream forked after the verify, joined by the next state user (fold_join)
    static const bool side = !getenv("AI_Q4X_FOLD_SIDE") || atoi(getenv("AI_Q4X_FOLD_SIDE")) != 0;
    if (side && !fold_stream_) {
        cudaStream_t fs0 = nullptr;
        cudaStreamCreateWithFlags(&fs0, cudaStreamNonBlocking);
        fold_stream_  = fs0;
        fold_ev_fork_ = kern::event_create();
        fold_ev_join_ = kern::event_create();
    }
    fold_join();   // a still-pending fold (normally none: the round in between joined it) before this one forks
    void * fs = side ? fold_stream_ : nullptr, * ef = fold_ev_fork_, * ej = fold_ev_join_;
    adapter::native(dev_, [=]() {
        void * mainst = adapter::stream(*dev);
        void * st     = mainst;
        if (fs) { kern::event_record(ef, mainst); kern::stream_wait_event(fs, ef); st = fs; }
        if (!kern::q4x_fold_batch(st, dg, ng, dc, nc, scr, mi, (int) kDoorbellT, C, Sk, Hk, Sv, Hv, eps)) {
            fprintf(stderr, "[q4x] AI_Q4X_FOLD_BATCH: launch failed\n");
            abort();
        }
        if (fs) kern::event_record(ej, fs);
    });
    fold_pending_ = fs != nullptr;
    return true;
}

void Qwen4ExpRun::fold_join(bool host) {
    if (!fold_pending_) return;
    fold_pending_ = false;
    kern::stream_wait_event(adapter::stream(dev_), fold_ev_join_);
    if (host) cudaEventSynchronize((cudaEvent_t) fold_ev_join_);
}

void Qwen4ExpRun::pin_launch_thread() {
    if (launch_cpus_.empty()) return;
    const std::thread::id me = std::this_thread::get_id();
    if (me == pinned_thread_) return;
    offload::pin_this_thread(launch_cpus_);
    pinned_thread_ = me;
}

Tensor Qwen4ExpRun::chunk(const int32_t * tokens, int64_t T) {
    pin_launch_thread();
    fold_join();
    ple_gate_idle();
    const Qwen4ExpSpec & s = prog_.spec();
    mtp_flush();
    const int64_t pos0 = (int64_t) hist_.size();
    if (pos0 + T > max_ctx_) throw std::runtime_error("Qwen4ExpRun: context full");
    dump_append_ = pos0 > 0;
    if (doorbell_ && T > kDoorbellT) {
        throw std::runtime_error("Qwen4ExpRun: doorbell mode runs chunks of <= " + std::to_string(kDoorbellT) +
                                 " rows (use --chunk " + std::to_string(kDoorbellT) + ")");
    }
    hist_.insert(hist_.end(), tokens, tokens + T);
    // AI_Q4X_CHUNK_PAD (default 256; 1 = the exact count): the visible cells rounded up to a multiple (masked past the
    // position), as the captured rounds' power-of-two envelope and llama.cpp's 256-padded KV views are. ggml's flash
    // attention loses accuracy on an unpadded cell count past 2048 cells, so eager chunks must pad like the rounds do
    // to score the same.
    static const int64_t chunk_pad = std::max<int64_t>(1, getenv("AI_Q4X_CHUNK_PAD") ? atoll(getenv("AI_Q4X_CHUNK_PAD")) : 256);
    const int64_t n_kv = std::min<int64_t>(max_ctx_, (pos0 + T + chunk_pad - 1) / chunk_pad * chunk_pad);

    // host inputs
    std::vector<int32_t>  pos4((size_t) (4 * T));
    std::vector<int64_t>  cells((size_t) T);
    std::vector<uint16_t> mask((size_t) (n_kv * T));
    for (int64_t i = 0; i < T; ++i) {
        const int32_t p = (int32_t) (pos0 + i);
        pos4[(size_t) i] = p; pos4[(size_t) (T + i)] = p; pos4[(size_t) (2 * T + i)] = p; pos4[(size_t) (3 * T + i)] = 0;
        cells[(size_t) i] = pos0 + i;
        for (int64_t j = 0; j < n_kv; ++j) mask[(size_t) (i * n_kv + j)] = j <= pos0 + i ? 0x0000 : 0xFC00;   // f16 0 / -inf
    }
    // Device::upload is not ordered against the compute stream: staging into the reset workspace while the previous
    // chunk's kernels still read it corrupts the prefill whenever the GPU lags the host (e.g. a slow CPU miss answer),
    // making outputs differ run to run. Drain the stream first.
    dev_.synchronize();
    dev_.ws_reset(0);
    Tensor ids_t   = dev_.ws_alloc(I32, T);
    Tensor pos4_t  = dev_.ws_alloc(I32, 4 * T);
    Tensor cells_t = dev_.ws_alloc(I64, T);
    Tensor mask_t  = dev_.ws_alloc(F16, n_kv, T);
    dev_.upload(ids_t, tokens, 0, (size_t) T * 4);
    dev_.upload(pos4_t, pos4.data(), 0, pos4.size() * 4);
    dev_.upload(cells_t, cells.data(), 0, cells.size() * 8);
    dev_.upload(mask_t, mask.data(), 0, mask.size() * 2);
    Tensor ple_t;
    if (!s.ple_layers.empty()) {
        std::vector<float> rows((size_t) (s.n_embd * T));
        ple_rows(pos0, T, rows.data());
        ple_t = dev_.ws_alloc(F32, s.n_embd, T);
        dev_.upload(ple_t, rows.data(), 0, rows.size() * 4);
    }
    Tensor lg = forward(ids_t, pos4_t, cells_t, mask_t, ple_t, T, n_kv);
    if (mtp_on_ && !rec_) { mtp_fill_p0_ = pos0; mtp_fill_m_ = T; }
    admit_kick();   // the chunk is launched: the admission copies may start now
    return lg;
}

Tensor Qwen4ExpRun::forward(const Tensor & ids_t, const Tensor & pos4_t, const Tensor & cells_t, const Tensor & mask_t,
                            const Tensor & ple_t, int64_t T, int64_t n_kv) {
    const Qwen4ExpSpec & s = prog_.spec();
    op::AttnInputs ain;
    ain.write_cells  = cells_t;
    ain.pos4         = pos4_t;
    ain.mask         = mask_t.reshape(n_kv, T, 1, 1);
    ain.n_kv         = n_kv;
    ain.B            = 1;
    ain.region_cells = max_ctx_;
    ain.first_cell   = 0;

    // two residual buffers outside the per-layer scopes: each layer's intermediates (~1 GB at T = 309) are rolled back
    // when the layer ends, and its output R is copied into the other buffer (as Qwen35Hybrid's forward does)
    Tensor Rbuf[2] = { dev_.ws_alloc(F32, s.n_embd, s.hc, T), dev_.ws_alloc(F32, s.n_embd, s.hc, T) };
    int    cur = 0;
    {
        Device::Scope sc(dev_);
        Tensor R0 = op::q4x::embed_hc(dev_, prog_.tok_embd(), ids_t, s.hc);
        dump("hc_init", -1, R0);
        op::copy_into(dev_, R0, Rbuf[cur], "q4x hc_init");
    }
    const auto & layers = prog_.layers();
    for (int il = 0; il < (int) layers.size(); ++il) {
        const Qwen4ExpLayer & L = layers[(size_t) il];
        Device::Scope sc(dev_);
        Tensor R = Rbuf[cur];
        Tensor        ple_pad;
        op::GdnRecord grec;
        // Debug: AI_Q4X_REC_MASK bit 0 = record (fold later) the GDN blocks, bit 1 = the PLE blocks; default 3
        static const int rec_mask = getenv("AI_Q4X_REC_MASK") ? atoi(getenv("AI_Q4X_REC_MASK")) : 3;
        const bool rg = rec_ && (rec_mask & 1), rp = rec_ && (rec_mask & 2);
        // Debug: AI_Q4X_DEL bitmask for timing only (output is wrong): 1 = no attention/GDN block, 2 = no MoE at all,
        // 4 = no routed experts (shared expert only), 8 = no PLE
        static const int del = getenv("AI_Q4X_DEL") ? atoi(getenv("AI_Q4X_DEL")) : 0;
        if (L.ple && !(del & 8) && ple_gate_now_) {   // [AI_Q4X_PLE_GATE] the rows arrive while the layers before run
            const unsigned * fl = ple_flag_d_, * wt = (const unsigned *) in_want_.data;
            const float *    src = ple_map_d_;
            float *          dst = (float *) ple_t.data;
            const int64_t    n   = s.n_embd * T;
            Device *         dv  = &dev_;
            adapter::native(dev_, [=]() { kern::q4x_ple_gate(adapter::stream(*dv), fl, wt, src, dst, n); });
        }
        if (L.ple && !(del & 8)) R = op::q4x::ple_block(dev_, R, ple_t, L.plew, pp_, ple_[(size_t) il], rp, rp ? &ple_pad : nullptr);
        op::q4x::HcRead r = op::q4x::hc_read(dev_, R, L.hc_attn, s.rms_eps);
        Tensor out = (del & 1) ? r.mixed : L.attn
            ? op::q4x::attn_qsa_block(dev_, r.mixed, L.at, ap_, kv_[(size_t) il], ain, L.qsa, L.qsa_p, qsa_[(size_t) il])
            : op::q4x::gdn_block_sigz(dev_, r.mixed, L.gdn, gp_, gdn_[(size_t) il], rg ? &grec : nullptr);
        if (rg || rp) keep_record(il, (L.attn || !rg) ? nullptr : &grec, (L.ple && rp) ? &ple_pad : nullptr);
        // [AI_Q4X_HC_INPLACE, default 1] both scatters write R in place (q4x_hc_combine_into: the fork's hc_combine_f32
        // float ops in the same order, bit-identical; prefill 7880328) and a non-PLE layer's R stays in Rbuf[cur]: no fresh
        // R per scatter and no per-layer copy (48 copy launches per round). A PLE layer's R is the block's own output, so
        // it is copied into Rbuf[cur] first.
        static const bool inplace = !getenv("AI_Q4X_HC_INPLACE") || atoi(getenv("AI_Q4X_HC_INPLACE")) != 0;
        const bool ip = inplace && R.contiguous() && out.contiguous();
        if (ip) {
            if (R.data != Rbuf[cur].data) { op::copy_into(dev_, R, Rbuf[cur], "q4x ple out"); R = Rbuf[cur]; }
            op::q4x::hc_scatter_into(dev_, R, out, r.inject);
        } else {
            R = op::q4x::hc_scatter(dev_, R, out, r.inject);
        }
        op::q4x::HcRead f = op::q4x::hc_read(dev_, R, L.hc_ffn, s.rms_eps);
        Tensor y = (del & 2) ? f.mixed
                 : (del & 4) ? op::moe_shared(dev_, f.mixed, L.moe)
                 : doorbell_ && T <= kDoorbellT ? moe_doorbell(il, f.mixed) : moe_streamed(il, f.mixed);
        dump("ffn_out", il, y);
        if (ip && y.contiguous()) {
            op::q4x::hc_scatter_into(dev_, R, y, f.inject);
            dump("l_last", il, R);
        } else {
            R = op::q4x::hc_scatter(dev_, R, y, f.inject);
            dump("l_last", il, R);
            op::copy_into(dev_, R, Rbuf[cur ^ 1], "q4x layer out");
            cur ^= 1;
        }
    }
    const Tensor & R = Rbuf[cur];
    if (mtp_on_) {   // the trunk streams (before the final mixer) of every row -> MTP H slots 1..T
        const int64_t HW = s.hc_dim();
        const Tensor  dst = mtp_H_.view(HW, T, 1, 1, (size_t) HW * 4, (size_t) (HW * T) * 4, (size_t) (HW * T) * 4,
                                        (size_t) HW * 4);
        op::copy_into(dev_, R.reshape(HW, T, 1, 1), dst, "q4x mtp H");
    }
    if (doorbell_ && T <= kDoorbellT) {   // fresh sequence numbers for the next window
        unsigned * wb = (unsigned *) win_base_.data;
        const int  nl = (int) layers.size();
        adapter::native(dev_, [=]() { kern::offload_window_end(adapter::stream(dev_), wb, nl); });
    }
    Tensor h = op::q4x::hc_head(dev_, R, prog_.hc_head(), s.rms_eps);
    dump("result_norm", -1, h);
    return op::q4x::linear(dev_, prog_.output(), h);   // fp16 tensor-core dense head
}

// Stops the admission worker, then releases the RAM tier.
Qwen4ExpRun::~Qwen4ExpRun() {
    fold_join(true);
    if (lend_ev_) cudaEventSynchronize((cudaEvent_t) lend_ev_);
    if (ple_thr_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(ple_mu_);
            ple_stop_ = true;
        }
        ple_cv_.notify_all();
        ple_thr_.join();
    }
    if (ple_map_) cudaFreeHost(ple_map_);
    if (ple_flag_) cudaFreeHost(ple_flag_);
    if (admit_thr_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(admit_mu_);
            admit_stop_ = true;
        }
        admit_cv_.notify_all();
        admit_thr_.join();
    }
    // RAM tier: stop the pool before its memory goes, then point the model's tensors back at the file mapping
    if (!tier_.empty()) {
        miss_.reset();
        if (copy_stream_) cudaStreamSynchronize((cudaStream_t) copy_stream_);
        for (auto & t : tier_) t.tensor->data = t.orig;
        pinned_host_free(tier_mem_);
    }
}

// The admission worker: stage the expert's three tensors from the host mapping into job j's pinned buffer, queue the
// H2D copies into the victim slot on the copy stream, record the event, then mark the job recorded. Buffer j % n is
// free: job j - n was published (which needs its event done) before job j could be enqueued.
void Qwen4ExpRun::admit_worker() {
    for (;;) {
        Pending job;
        {
            std::unique_lock<std::mutex> lk(admit_mu_);
            admit_cv_.wait(lk, [&] { return admit_stop_ || !admit_q_.empty(); });
            if (admit_stop_) return;
            job = admit_q_.front();
            admit_q_.pop_front();
        }
        const size_t   si = (size_t) (job.job % (int64_t) staging_.size());
        const auto &   m  = prog_.layers()[(size_t) job.layer].moe;
        SlotBank &     B  = banks_[(size_t) job.layer];
        char *         st = staging_[si];
        size_t         o  = 0;
        const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
        const Tensor * dst[3] = { &B.gate, &B.up, &B.down };
        const bool tiered = !tier_layer_.empty() && tier_layer_[(size_t) job.layer] == 1;   // pinned tier: one DMA
        for (int r = 0; r < 3; ++r) {
            const char * sp = (const char *) src[r]->data + (size_t) job.expert * src[r]->nb[2];
            const char * cp = sp;
            // a tiered (pinned) tensor is copied by one DMA; otherwise stage through pinned memory (a pageable
            // copy would be synchronous)
            if (!tiered) {
                memcpy(st + o, sp, src[r]->nb[2]);
                cp = st + o;
                n_staged_++;
            } else {
                n_direct_++;
            }
            cudaMemcpyAsync((char *) dst[r]->data + (size_t) job.slot * dst[r]->nb[2], cp, src[r]->nb[2],
                            cudaMemcpyHostToDevice, (cudaStream_t) copy_stream_);
            o += src[r]->nb[2];
        }
        cudaEventRecord((cudaEvent_t) staging_ev_[si], (cudaStream_t) copy_stream_);
        recorded_[si].store(job.job, std::memory_order_release);
    }
}

// AI_Q4X_ADMIT_DEFER (default 1): the admission worker gets this boundary's jobs only after the next round is
// launched. Issued at the boundary, the expert DMAs would sit ahead of the round's own ingress uploads and table
// copies on the H2D path and delay the round by about their total. Deferred, the copies overlap the round and the
// next draft, and the next boundary publishes them as before.
static bool admit_defer_on() {
    static const bool v = !getenv("AI_Q4X_ADMIT_DEFER") || atoi(getenv("AI_Q4X_ADMIT_DEFER")) != 0;
    return v;
}

void Qwen4ExpRun::admit_kick() {
    if (admit_defer_.empty()) return;
    {
        std::lock_guard<std::mutex> lk(admit_mu_);
        for (const Pending & j : admit_defer_) admit_q_.push_back(j);
    }
    admit_defer_.clear();
    admit_cv_.notify_one();
}

// Admission. AI_Q4X_ADMIT = admissions per round (default 32; 0 = static residency). Scores decay by 0.97 per round;
// a missed expert is admitted when its score >= 2 and exceeds 1.5x the victim's (the coldest slot of its layer).
// Table changes go to the device as one async copy per changed layer on the compute stream, ordered before the next
// round; a victim is unmapped in that copy, and its new expert is mapped only after its H2D copy has landed.
void Qwen4ExpRun::admit_after_round() {
    static const int max_admit = getenv("AI_Q4X_ADMIT") ? atoi(getenv("AI_Q4X_ADMIT")) : 32;
    // [AI_Q4X_ADMIT_THR, default 2] a missed expert's decayed use count needed to be admitted; [AI_Q4X_ADMIT_RATIO,
    // default 1.5] and at least this times the coldest resident's (hysteresis)
    static const float admit_thr   = getenv("AI_Q4X_ADMIT_THR") ? (float) atof(getenv("AI_Q4X_ADMIT_THR")) : 2.0f;
    static const float admit_ratio = getenv("AI_Q4X_ADMIT_RATIO") ? (float) atof(getenv("AI_Q4X_ADMIT_RATIO")) : 1.5f;
    if (!doorbell_ || banks_.empty()) return;
    const Qwen4ExpSpec & s = prog_.spec();
    const int k = s.n_expert_used, E = s.n_expert;
    std::vector<char> dirty(banks_.size(), 0);
    admit_kick();   // jobs never handed over (no launch since the last boundary) would block the publish below
    lend_publish(false);   // [AI_Q4X_LEND_ASYNC] the lent layers mapped again once their give-back landed
    // 1) publish copies that landed. AI_Q4X_ADMIT_SYNC=1 (default): every job enqueued at the previous round boundary
    //    is published now, waiting for its copy if needed (the copies are short next to a round, so the wait is
    //    rarely real). Residency is then a function of the token history alone, so fast mode is run-to-run
    //    deterministic; publishing whenever a copy happens to land would make the text depend on timing.
    static const bool sync_pub = !getenv("AI_Q4X_ADMIT_SYNC") || atoi(getenv("AI_Q4X_ADMIT_SYNC")) != 0;
    for (size_t i = 0; i < pending_.size();) {
        Pending &    p  = pending_[i];
        const size_t si = (size_t) (p.job % (int64_t) staging_.size());
        if (sync_pub) {
            while (recorded_[si].load(std::memory_order_acquire) != p.job) std::this_thread::yield();
            cudaEventSynchronize((cudaEvent_t) staging_ev_[si]);
        } else if (recorded_[si].load(std::memory_order_acquire) != p.job ||
                   cudaEventQuery((cudaEvent_t) staging_ev_[si]) != cudaSuccess) { ++i; continue; }
        SlotBank & B = banks_[(size_t) p.layer];
        B.table_h[(size_t) p.expert] = p.slot;
        B.owner[(size_t) p.slot]     = p.expert;
        dirty[(size_t) p.layer]      = 1;
        n_publish_++;
        pending_.erase(pending_.begin() + (long) i);
    }
    // 2) count this round's routing (hits and misses) per layer
    struct Cand { float score; int layer, expert; };
    std::vector<Cand> cand;
    for (size_t il = 0; bells_live_ && il < banks_.size(); ++il) {   // no round of this sequence yet: nothing to count
        SlotBank &            B  = banks_[il];
        const offload::Bell & bl = miss_->bell((int) il);
        for (auto & v : B.score) v *= 0.97f;
        const int T = *bl.n_tok;
        // AI_Q4X_ROUTE_LOG=<file>: per round and layer, i32 T then the k*T routed experts (for the offline
        // residency simulation)
        static FILE * rlog = getenv("AI_Q4X_ROUTE_LOG") ? fopen(getenv("AI_Q4X_ROUTE_LOG"), "wb") : nullptr;
        if (rlog) {
            std::vector<int32_t> ex((size_t) (k * T) + 1);
            ex[0] = T;
            for (int c = 0; c < k * T; ++c) {
                const int32_t m = bl.miss[c];
                ex[(size_t) c + 1] = m >= 0 ? m : m <= -kGpuTagH ? -(m + kGpuTagH + 2) : -(m + 2);
            }
            fwrite(ex.data(), 4, ex.size(), rlog);
            if (il + 1 == banks_.size()) fflush(rlog);
        }
        for (int c = 0; c < k * T; ++c) {
            const int32_t m   = bl.miss[c];
            const bool    gpu = m <= -kGpuTagH;   // GPU miss share: computed by the GPU from the tier, still a miss
            const int32_t e   = m >= 0 ? m : gpu ? -(m + kGpuTagH + 2) : -(m + 2);
            if (e < 0 || e >= s.n_expert) continue;
            B.score[(size_t) e] += 1.0f;
            if ((m >= 0 || gpu) && B.table_h[(size_t) e] < 0) cand.push_back({ B.score[(size_t) e], (int) il, e });
            // [AI_Q4X_HIT_STATS=1] decode hit counters: all routed entries, and ranks 0..6 (Strata's lookups leave out
            // the tail-skipped ranks)
            static const bool hit_stats = getenv("AI_Q4X_HIT_STATS") && atoi(getenv("AI_Q4X_HIT_STATS")) != 0;
            if (hit_stats) {
                const bool hit = m < 0 && !gpu;
                g_hits[0] += hit; g_hits[1] += 1;
                if (c % k < 7) { g_hits[2] += hit; g_hits[3] += 1; }
            }
        }
    }
    int issued = 0;
    if (max_admit > 0 && !cand.empty()) {
        std::sort(cand.begin(), cand.end(), [](const Cand & a, const Cand & b) { return a.score > b.score; });
        for (const Cand & cd : cand) {
            if (issued >= max_admit || pending_.size() >= staging_.size()) break;
            if (cd.score < admit_thr) break;
            SlotBank & B = banks_[(size_t) cd.layer];
            if (B.table_h[(size_t) cd.expert] >= 0) continue;
            bool in_flight = false;
            for (const Pending & p : pending_) in_flight = in_flight || (p.layer == cd.layer && p.expert == cd.expert);
            if (in_flight) continue;
            int   victim = -1;
            float vs     = 1e30f;
            for (int sl = 0; sl < B.n_slots; ++sl) {
                const int32_t e = B.owner[(size_t) sl];
                if (e < 0) continue;   // pending
                if (B.score[(size_t) e] < vs) { vs = B.score[(size_t) e]; victim = sl; }
            }
            if (victim < 0 || cd.score < admit_ratio * vs + 1e-6f) continue;
            const int32_t old = B.owner[(size_t) victim];
            B.table_h[(size_t) old]  = -1;   // unmapped by this round's table copy, before the round launches
            B.owner[(size_t) victim] = -1;
            dirty[(size_t) cd.layer] = 1;
            Pending job{ cd.layer, victim, cd.expert, (int64_t) n_admit_ };
            pending_.push_back(job);
            if (admit_defer_on()) {
                admit_defer_.push_back(job);   // queued for the worker only after the next round is launched
            } else {
                std::lock_guard<std::mutex> lk(admit_mu_);
                admit_q_.push_back(job);
            }
            n_admit_++;
            g_admits++;
            issued++;
        }
        if (issued && !admit_defer_on()) admit_cv_.notify_one();
    }
    // 3) changed tables -> device, ordered on the compute stream before the next round
    void * st = adapter::stream(dev_);
    for (size_t il = 0; il < banks_.size(); ++il) {
        if (!dirty[il]) continue;
        int32_t * pin = table_pin_ + il * (size_t) E;
        memcpy(pin, banks_[il].table_h.data(), (size_t) E * 4);
        cudaMemcpyAsync(banks_[il].table.data, pin, (size_t) E * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
}

int Qwen4ExpRun::reset_residency() {
    if (!doorbell_ || banks_.empty() || init_owner_.size() != banks_.size()) return 0;
    lend_publish(true);   // [AI_Q4X_LEND_ASYNC] owners final for the lent layers too
    admit_drain();   // every in-flight admission landed and published: owners are final
    bells_live_ = false;
    const auto & layers = prog_.layers();
    const int    E      = prog_.spec().n_expert;
    void *       st     = adapter::stream(dev_);
    int          n      = 0;
    for (size_t il = 0; il < banks_.size(); ++il) {
        SlotBank &                   B    = banks_[il];
        const std::vector<int32_t> & want = init_owner_[il];
        const auto &                 m    = layers[il].moe;
        const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
        const Tensor * dst[3] = { &B.gate, &B.up, &B.down };
        bool dirty = false;
        for (size_t sl = 0; sl < want.size(); ++sl) {
            const int32_t e = want[sl], cur = B.owner[sl];
            if (cur == e) continue;
            if (e >= 0)
                for (int r = 0; r < 3; ++r)   // ordered on the compute stream, before the next request's first kernel
                    cudaMemcpyAsync((char *) dst[r]->data + sl * dst[r]->nb[2], (const char *) src[r]->data + (size_t) e * src[r]->nb[2],
                                    src[r]->nb[2], cudaMemcpyHostToDevice, (cudaStream_t) st);
            B.owner[sl] = e;
            dirty       = true;
            ++n;
        }
        std::fill(B.score.begin(), B.score.end(), 0.0f);
        if (!dirty) continue;
        // the table from the owners, not per slot: an expert evicted and re-admitted into ANOTHER slot moves back to its
        // first slot, and clearing it when its later slot is visited unmapped it while resident (a CPU miss, so the
        // rounding still followed the request order)
        std::fill(B.table_h.begin(), B.table_h.end(), -1);
        for (int sl = 0; sl < B.n_slots; ++sl)
            if (B.owner[(size_t) sl] >= 0) B.table_h[(size_t) B.owner[(size_t) sl]] = sl;
        int32_t * pin = table_pin_ + il * (size_t) E;
        memcpy(pin, B.table_h.data(), (size_t) E * 4);
        cudaMemcpyAsync(B.table.data, pin, (size_t) E * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
    return n;
}

// Publish every admission still in flight (waiting for its copy), with no new ones: the last decode round of a request
// leaves its admissions queued and their victims unmapped, so a following prefill would see more non-resident experts
// than its scratch holds (E - n_slots) and fail with "scratch bank full".
void Qwen4ExpRun::admit_drain() {
    if (!doorbell_ || banks_.empty() || pending_.empty()) return;
    const int         E = prog_.spec().n_expert;
    std::vector<char> dirty(banks_.size(), 0);
    admit_kick();
    for (const Pending & p : pending_) {
        const size_t si = (size_t) (p.job % (int64_t) staging_.size());
        while (recorded_[si].load(std::memory_order_acquire) != p.job) std::this_thread::yield();
        cudaEventSynchronize((cudaEvent_t) staging_ev_[si]);
        SlotBank & B = banks_[(size_t) p.layer];
        B.table_h[(size_t) p.expert] = p.slot;
        B.owner[(size_t) p.slot]     = p.expert;
        dirty[(size_t) p.layer]      = 1;
        n_publish_++;
    }
    pending_.clear();
    void * st = adapter::stream(dev_);
    for (size_t il = 0; il < banks_.size(); ++il) {
        if (!dirty[il]) continue;
        int32_t * pin = table_pin_ + il * (size_t) E;
        memcpy(pin, banks_[il].table_h.data(), (size_t) E * 4);
        cudaMemcpyAsync(banks_[il].table.data, pin, (size_t) E * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
}

// One decode row as a captured round (doorbell mode): inputs go to fixed ingress buffers outside the workspace,
// the visible cells are padded to a power-of-two envelope F (the key), and the round is recorded once per F, captured
// on its second launch and replayed after (engine/round_cache.h). Cells beyond the position are masked; the KV and
// indexer caches are zeroed at start so masked cells hold no NaN.
Tensor Qwen4ExpRun::decode(int32_t token) {
    admit_after_round();   // the previous round has completed: its caller read the logits
    return round(&token, 1);
}

void Qwen4ExpRun::ple_gate_init() {
    const Qwen4ExpSpec & s = prog_.spec();
    if (cudaHostAlloc((void **) &ple_map_, (size_t) (s.n_embd * kDoorbellT) * 4, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostAlloc((void **) &ple_flag_, 256, cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void **) &ple_map_d_, ple_map_, 0) != cudaSuccess ||
        cudaHostGetDevicePointer((void **) &ple_flag_d_, ple_flag_, 0) != cudaSuccess) {
        throw std::runtime_error("q4x: AI_Q4X_PLE_GATE mapped allocation failed");
    }
    memset(ple_flag_, 0, 256);
    const int b = dev_.alloc_buffer(256, "q4x.ple_want");
    in_want_.buf = b; in_want_.type = I32; in_want_.ne[0] = 1; set_contiguous_strides(in_want_);
    in_want_.data = dev_.buffer_base(b);
    ple_thr_ = std::thread([this]() {
        if (!host_cpus_.empty()) offload::pin_this_thread(host_cpus_);
        for (;;) {
            int64_t p0, T;
            unsigned seq;
            bool early;
            {
                std::unique_lock<std::mutex> lk(ple_mu_);
                ple_cv_.wait(lk, [&]() { return ple_stop_ || ple_job_; });
                if (ple_stop_) return;
                p0 = ple_p0_; T = ple_T_; seq = ple_job_seq_; early = ple_early_job_;
            }
            if (early) {   // [AI_Q4X_VERIFY_EARLY] the chain's drafts into hist_ (the rows hash them), then wake the caller
                cudaEventSynchronize((cudaEvent_t) ple_ev_chain_);
                cudaMemcpyAsync(draft_pin_, mc_tok_.data, (size_t) (T - 1) * 4, cudaMemcpyDeviceToHost, (cudaStream_t) ple_side_);
                cudaStreamSynchronize((cudaStream_t) ple_side_);
                for (int64_t i = 0; i + 1 < T; ++i) hist_[(size_t) (p0 + 1 + i)] = draft_pin_[i];
                {
                    std::lock_guard<std::mutex> lk(ple_mu_);
                    ple_early_done_ = true;
                }
                ple_cv_.notify_all();
            }
            ple_rows(p0, T, ple_map_);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned *) ple_flag_ = seq;   // the gate kernel may copy the rows now
            {
                std::lock_guard<std::mutex> lk(ple_mu_);
                ple_job_ = false;
            }
            ple_cv_.notify_all();
        }
    });
    fprintf(stderr, "[q4x] PLE gate: decode rows read by a worker while the round launches; the round waits before its PLE layer (AI_Q4X_PLE_GATE)\n");
}

void Qwen4ExpRun::ple_gate_idle() {
    if (!ple_thr_.joinable()) return;
    std::unique_lock<std::mutex> lk(ple_mu_);
    ple_cv_.wait(lk, [&]() { return !ple_job_; });
}

// [P3/P5] T <= 8 rows (decode: T = 1; verify: the pending token + drafts, record mode) as a CAPTURED round. Inputs go
// to fixed ingress buffers sized for 8 rows; the key is (envelope F, T, record mode), so each verify width and the plain
// decode get their own recorded round.
Tensor Qwen4ExpRun::round(const int32_t * tokens, int64_t T, bool early) {
    if (!doorbell_) throw std::runtime_error("Qwen4ExpRun::round: needs AI_Q4X_DOORBELL=1 (the v1 pool reads routing on the host)");
    if (T < 1 || T > kDoorbellT) throw std::runtime_error("Qwen4ExpRun::round: 1 <= T <= kDoorbellT");
    bells_live_ = true;   // this round rings the bells the next admit_after_round counts
    ple_gate_idle();
    auto tr = std::chrono::steady_clock::now();
    mtp_flush();
    if (g_rt.on) g_rt.flush += rt_ms_since(tr);
    const Qwen4ExpSpec & s = prog_.spec();
    const int64_t pos0 = (int64_t) hist_.size();
    if (pos0 + T > max_ctx_) throw std::runtime_error("Qwen4ExpRun: context full");
    hist_.insert(hist_.end(), tokens, tokens + T);
    int64_t F = 256;
    while (F < pos0 + T) F *= 2;
    F = std::min(F, max_ctx_);
    constexpr int64_t Tm = kDoorbellT;
    if (!in_ids_.valid()) {
        const size_t o_pos = 256, o_cells = 512, o_ple = 1024, o_mask = o_ple + (size_t) (s.n_embd * Tm * 4 + 255) / 256 * 256;
        const int    b     = dev_.alloc_buffer(o_mask + (size_t) (max_ctx_ * Tm * 2) + 256, "q4x.ingress");
        char *       p     = (char *) dev_.buffer_base(b);
        auto mk = [&](int type, int64_t n0, size_t off) {
            Tensor t; t.buf = b; t.type = type; t.ne[0] = n0; set_contiguous_strides(t); t.data = p + off; return t;
        };
        in_ids_   = mk(I32, Tm, 0);
        in_pos4_  = mk(I32, 4 * Tm, o_pos);
        in_cells_ = mk(I64, Tm, o_cells);
        in_ple_   = mk(F32, s.n_embd * Tm, o_ple);
        in_mask_  = mk(F16, max_ctx_ * Tm, o_mask);
    }
    if (g_rt.on) rt_ms_since(tr);
    // the ingress uploads below are unordered against the stream (see chunk()). Early: only the draft chain is in
    // flight (the previous verify was read back), and it reads none of the ingress buffers
    if (!early) dev_.synchronize();
    if (g_rt.on) { g_rt.sync += rt_ms_since(tr); g_rt.gpu_eval(); rt_ms_since(tr); }
    std::vector<int32_t>  pos4((size_t) (4 * T));
    std::vector<int64_t>  cells((size_t) T);
    std::vector<uint16_t> mask((size_t) (F * T));
    for (int64_t i = 0; i < T; ++i) {
        const int32_t q = (int32_t) (pos0 + i);
        pos4[(size_t) i] = q; pos4[(size_t) (T + i)] = q; pos4[(size_t) (2 * T + i)] = q; pos4[(size_t) (3 * T + i)] = 0;
        cells[(size_t) i] = pos0 + i;
        for (int64_t jj = 0; jj < F; ++jj) mask[(size_t) (i * F + jj)] = jj <= pos0 + i ? 0x0000 : 0xFC00;
    }
    if (early) {   // [AI_Q4X_VERIFY_EARLY] the pending token from the host, the drafts from the chain (stream-ordered)
        dev_.upload(in_ids_, tokens, 0, 4);
        kern::q4x_copy_i32(adapter::stream(dev_), (const int32_t *) mc_tok_.data, (int32_t *) in_ids_.data + 1, T - 1);
    } else {
        dev_.upload(in_ids_, tokens, 0, (size_t) T * 4);
    }
    dev_.upload(in_pos4_, pos4.data(), 0, pos4.size() * 4);
    dev_.upload(in_cells_, cells.data(), 0, cells.size() * 8);
    dev_.upload(in_mask_, mask.data(), 0, mask.size() * 2);
    if (g_rt.on) g_rt.upload += rt_ms_since(tr);
    // [AI_Q4X_PLE_GATE, default 1 since 2026-11-02: 3090 + 5090 wins vs Strata, 4080 with the other four +0.4/+1.3% vs the production config, TTFT -0.14 s; =0 off] the rows are read by the worker, posted just before the launch
    static const bool gate_env = !getenv("AI_Q4X_PLE_GATE") || atoi(getenv("AI_Q4X_PLE_GATE")) != 0;
    bool gated = false;
    if (!s.ple_layers.empty()) {
        if (gate_env) {
            if (!ple_thr_.joinable()) ple_gate_init();
            ++ple_seq_;
            dev_.upload(in_want_, &ple_seq_, 0, 4);
            gated = true;
        } else {
            std::vector<float> rows((size_t) (s.n_embd * T));
            ple_rows(pos0, T, rows.data());
            if (g_rt.on) g_rt.ple += rt_ms_since(tr);
            dev_.upload(in_ple_, rows.data(), 0, rows.size() * 4);
        }
    }
    auto v1 = [](const Tensor & t, int64_t n0, size_t es) { return t.view(n0, 1, 1, 1, n0 * es, n0 * es, n0 * es, 0); };
    const Tensor ids_v   = v1(in_ids_, T, 4);
    const Tensor pos4_v  = v1(in_pos4_, 4 * T, 4);
    const Tensor cells_v = v1(in_cells_, T, 8);
    const Tensor ple_v   = in_ple_.view(s.n_embd, T, 1, 1, (size_t) s.n_embd * 4, (size_t) (s.n_embd * T) * 4,
                                        (size_t) (s.n_embd * T) * 4, 0);
    const Tensor mask_v  = in_mask_.view(F, T, 1, 1, (size_t) F * 2, (size_t) (F * T) * 2, (size_t) (F * T) * 2, 0);
    bool warm = false;
    const std::string dd = dump_dir_;
    dump_dir_.clear();   // no host reads inside a captured round
    const int64_t key = (F << 6) | (T << 1) | (rec_ ? 1 : 0);
    pin_launch_thread();
    fold_join();   // [AI_Q4X_FOLD_SIDE] outside any capture: the round reads the folded state
    if (gated) {
        {
            std::lock_guard<std::mutex> lk(ple_mu_);
            ple_p0_ = pos0; ple_T_ = T; ple_job_seq_ = ple_seq_; ple_job_ = true; ple_early_job_ = early;
        }
        ple_cv_.notify_all();
    }
    ple_gate_now_ = gated;
    if (g_rt.on) g_rt.mark(2, adapter::stream(dev_));
    const Tensor & lg = rounds_.launch(dev_, key, [&]() {
        return forward(ids_v, pos4_v, cells_v, mask_v, s.ple_layers.empty() ? Tensor{} : ple_v, T, F);
    }, &warm, true, [&](const int64_t & k2, const auto & e) {
        fprintf(stderr, "[q4x] round recorded: F=%lld T=%lld%s, %d nodes\n", (long long) (k2 >> 6), (long long) ((k2 >> 1) & 31),
                (k2 & 1) ? " (verify)" : "", e.round->n_nodes());
    });
    ple_gate_now_ = false;
    dump_dir_ = dd;
    if (g_rt.on) { g_rt.mark(3, adapter::stream(dev_)); g_rt.launch += rt_ms_since(tr); }
    if (mtp_on_ && !rec_) { mtp_fill_p0_ = pos0; mtp_fill_m_ = T; }
    admit_kick();   // the round is launched: the admission copies may start now
    if (g_rt.on) { g_rt.kick += rt_ms_since(tr); g_rt.n++; }
    return lg;
}

}  // namespace ai
