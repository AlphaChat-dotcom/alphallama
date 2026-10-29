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
    dev_.synchronize();
    const std::vector<Tensor *> ts = recurrent_tensors();
    if (ts.size() != ck_buf_.size()) return false;
    for (size_t i = 0; i < ts.size(); ++i) dev_.upload(*ts[i], ck_buf_[i].data(), 0, ck_buf_[i].size());
    dev_.synchronize();   // Device::upload is unordered against the stream: land it before the next launch
    mtp_fill_p0_ = ck_fill_p0_;
    mtp_fill_m_  = ck_fill_m_;
    hist_.resize((size_t) ck_len_);
    return true;
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
    struct Row { const char * src; float * dst; };
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
                rows.push_back({ (const char *) tbl.data + (size_t) r * tbl.nb[1], out + (i * n_heads + h) * hd });
            }
        }
    }
    const size_t rb = (size_t) tbl.nb[1];
    // AI_Q4X_PLE_PREFETCH (default 1): ask the kernel for every row of the round before reading any. The rows are
    // hashed n-grams spread over a table of tens of GB that the page cache holds only partly (the model files are
    // evicted after the tier fill), so one pread per row would wait for one NVMe read each before the GPU could
    // start. WILLNEED queues them all at once.
    static const bool ple_prefetch = !getenv("AI_Q4X_PLE_PREFETCH") || atoi(getenv("AI_Q4X_PLE_PREFETCH")) != 0;
    if (ple_prefetch) {
        for (const Row & w : rows) prog_.weights().host_prefetch(w.src, rb);
    }
    // AI_Q4X_PLE_PREAD (default 1): read each row through pread, so the table's pages are never mapped into this
    // process and compaction migrating them cannot IPI the pool (WeightSet::host_pread)
    static const bool ple_pread = !getenv("AI_Q4X_PLE_PREAD") || atoi(getenv("AI_Q4X_PLE_PREAD")) != 0;
    for (const Row & w : rows) {
        const char * src = w.src;
        char         row[8192];
        if (ple_pread && rb <= sizeof row && prog_.weights().host_pread(src, row, rb)) src = row;
        if (tbl.type == F32) memcpy(w.dst, src, (size_t) hd * 4);
        else tt->to_float(src, w.dst, hd);
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
    size_t total = 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & m = layers[il].moe;
        total += (m.gate_exps.nb[2] + m.up_exps.nb[2] + m.down_exps.nb[2]) * (size_t) nb + (size_t) E * 4 + 4096;
    }
    const int buf  = dev_.alloc_buffer(total, "q4x.slots");
    char *    base = (char *) dev_.buffer_base(buf);
    size_t    off  = 0;
    auto bank = [&](const Tensor & h) {
        Tensor t = h;
        t.buf    = buf;
        t.ne[2]  = nb;
        set_contiguous_strides(t);
        off    = (off + 255) / 256 * 256;
        t.data = base + off;
        off   += t.nbytes();
        return t;
    };
    std::vector<offload::ExpertBanks> host;
    std::vector<double> layer_cold;   // RAM tier: per layer: profile mass of its initially non-resident experts
    double resident = 0;
    for (size_t il = 0; il < layers.size(); ++il) {
        const auto & m = layers[il].moe;
        SlotBank &   B = banks_[il];
        B.gate    = bank(m.gate_exps);
        B.up      = bank(m.up_exps);
        B.down    = bank(m.down_exps);
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
            // the launch core lost ~1/3 of decode (4080 AlphaServe 2026-10-28, MoLE: set A 71.9 vs ai-api-gen 110.3)
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
    op::MoeRoute  route;
    const Tensor & rw = L.moe.router;
    if (route_native && T <= kDoorbellT && (rw.type == BF16 || rw.type == F16) && rw.contiguous() && x.contiguous() &&
        rw.ne[0] == N && rw.ne[1] == s.n_expert && s.n_expert <= 1024) {
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
        adapter::native(dev_, [=]() { kern::q4x_route(adapter::stream(*dv), wp0, bf, xp0, (int) N, E, (int) T, k, lp, sp0, wo); });
    } else {
        route = op::moe_route(dev_, x, L.moe.router, mp);
    }
    Tensor          slots = dev_.ws_alloc(I32, k, T);
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
    adapter::native(dev_, [=]() {
        kern::offload_resolve_ring(adapter::stream(*dev), tab, sel, xp, N, k, T, filler0, sp, bl.d_miss, bl.d_x,
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
    Tensor shared = op::moe_shared(dev_, x, L.moe);   // runs while the CPU computes the misses
    // AI_Q4X_GROUPED: routed experts grouped by slot (src/kernels/q4x_experts.cu) -- each resident slot's weights
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
        Tensor        scr  = dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(N, F, k, T) / 4) + 64, 1);
        const int     tg = (int) B.up.type, td = (int) B.down.type;
        const void *  pu = B.up.data, *pg = B.gate.data, *pd = B.down.data;
        const int64_t rbg = (int64_t) B.up.nb[1], ebg = (int64_t) B.up.nb[2];
        const int64_t rbd = (int64_t) B.down.nb[1], ebd = (int64_t) B.down.nb[2];
        const int64_t xs  = (int64_t) x.nb[1] / (int64_t) sizeof(float);
        const int32_t * spc = sp;
        void *        ps  = scr.data;
        float *       pm  = (float *) mine.data;
        adapter::native(dev_, [=]() {
            if (!kern::q4x_experts(adapter::stream(*dev), tg, pu, pg, rbg, ebg, td, pd, rbd, ebd, xp, xs, spc, k, filler0,
                                   N, F, N, k, T, ps, pm)) {
                fprintf(stderr, "[q4x-experts] declined a shape the host check accepted\n");
                abort();
            }
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
        exps = op::q4x::moe_experts_raw(dev_, x, B.up, B.gate, B.down, slots);
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
    auto keep = [&](const Tensor & src, size_t field, const char * who) {
        Tensor t = src;
        t.buf    = rec_buf_;
        set_contiguous_strides(t);
        t.data = base + field;
        op::copy_into(dev_, src, t, who);
        return t;
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

void Qwen4ExpRun::commit(int64_t m1) {
    if (rec_T_ <= 0 || m1 < 1 || m1 > rec_T_) throw std::runtime_error("Qwen4ExpRun::commit: no verify, or m1 out of range");
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
    if (!doorbell_ || banks_.empty()) return;
    const Qwen4ExpSpec & s = prog_.spec();
    const int k = s.n_expert_used, E = s.n_expert;
    std::vector<char> dirty(banks_.size(), 0);
    admit_kick();   // jobs never handed over (no launch since the last boundary) would block the publish below
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
    for (size_t il = 0; il < banks_.size(); ++il) {
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
        }
    }
    int issued = 0;
    if (max_admit > 0 && !cand.empty()) {
        std::sort(cand.begin(), cand.end(), [](const Cand & a, const Cand & b) { return a.score > b.score; });
        for (const Cand & cd : cand) {
            if (issued >= max_admit || pending_.size() >= staging_.size()) break;
            if (cd.score < 2.0f) break;
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
            if (victim < 0 || cd.score < 1.5f * vs + 1e-6f) continue;
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

// T <= kDoorbellT rows (decode: T = 1; verify: the pending token + drafts, record mode) as a captured round. Inputs go
// to fixed ingress buffers sized for kDoorbellT rows; the key is (envelope F, T, record mode), so each verify width and the plain
// decode get their own recorded round.
Tensor Qwen4ExpRun::round(const int32_t * tokens, int64_t T) {
    if (!doorbell_) throw std::runtime_error("Qwen4ExpRun::round: needs AI_Q4X_DOORBELL=1 (the expert pool path reads routing on the host)");
    if (T < 1 || T > kDoorbellT) throw std::runtime_error("Qwen4ExpRun::round: 1 <= T <= kDoorbellT");
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
    dev_.synchronize();   // the ingress uploads below are unordered against the stream (see chunk())
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
    dev_.upload(in_ids_, tokens, 0, (size_t) T * 4);
    dev_.upload(in_pos4_, pos4.data(), 0, pos4.size() * 4);
    dev_.upload(in_cells_, cells.data(), 0, cells.size() * 8);
    dev_.upload(in_mask_, mask.data(), 0, mask.size() * 2);
    if (g_rt.on) g_rt.upload += rt_ms_since(tr);
    if (!s.ple_layers.empty()) {
        std::vector<float> rows((size_t) (s.n_embd * T));
        ple_rows(pos0, T, rows.data());
        if (g_rt.on) g_rt.ple += rt_ms_since(tr);
        dev_.upload(in_ple_, rows.data(), 0, rows.size() * 4);
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
    fold_join();   // AI_Q4X_FOLD_SIDE: outside any capture: the round reads the folded state
    if (g_rt.on) g_rt.mark(2, adapter::stream(dev_));
    const Tensor & lg = rounds_.launch(dev_, key, [&]() {
        return forward(ids_v, pos4_v, cells_v, mask_v, s.ple_layers.empty() ? Tensor{} : ple_v, T, F);
    }, &warm, true, [&](const int64_t & k2, const auto & e) {
        fprintf(stderr, "[q4x] round recorded: F=%lld T=%lld%s, %d nodes\n", (long long) (k2 >> 6), (long long) ((k2 >> 1) & 31),
                (k2 & 1) ? " (verify)" : "", e.round->n_nodes());
    });
    dump_dir_ = dd;
    if (g_rt.on) { g_rt.mark(3, adapter::stream(dev_)); g_rt.launch += rt_ms_since(tr); }
    if (mtp_on_ && !rec_) { mtp_fill_p0_ = pos0; mtp_fill_m_ = T; }
    admit_kick();   // the round is launched: the admission copies may start now
    if (g_rt.on) { g_rt.kick += rt_ms_since(tr); g_rt.n++; }
    return lg;
}

}  // namespace ai
