// SPDX-License-Identifier: Apache-2.0
// The qwen4exp (Flash-Next) MTP draft block -- sidecar load, head K/V fill and the draft chain.
//
// Reference: src/llama-context-mtp4.cpp. One trunk-shaped layer with
// hyper-connections:
//   R0    = h_proj(rms(H) * hnorm) per stream + eh_proj(rms(embed(tok)) * enorm)      H = trunk streams [hc*N]
//   R1    = hc_scatter(R0, attn(hc_read(R0)))       dense gated attention over the head's own K/V cache
//   R2    = hc_scatter(R1, moe(hc_read(R1)))        512 experts, all on the device, + gated shared expert
//   draft = argmax(output(hc_head(R2)))             R2 is also the H the next chained draft starts from
// H is the trunk's residual streams after the last layer, before its final mixer. The head's K/V column c depends only
// on (H_{c-1}, token_c), so committed columns are (re)filled from the trunk rows of the round that committed them.
#include "models/qwen4exp_run.h"
#include "ops/ops.h"
#include "ops/q4x_ops.h"
#include "ops/adapter.h"
#include "kernels/kernels.h"

#include <cuda_runtime.h>

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ai {

static std::string mtp_blk(int il, const char * name) {
    return "blk." + std::to_string(il) + "." + name;
}

bool Qwen4Exp::load_mtp(Device & dev, const std::string & path, bool verbose) {
    if (!mtp_w_.open(path)) return false;
    WeightSet::ConvertFilter conv = [](const std::string & n, const Tensor & t) {
        const bool hc_lr = (n.find("hc_") != std::string::npos) &&
                           (n.size() > 12 && (n.compare(n.size() - 12, 12, "_down.weight") == 0 ||
                                              n.compare(n.size() - 10, 10, "_up.weight") == 0));
        if (hc_lr && t.type == BF16 && t.ne[0] % 32 == 0) return (int) Q8_0;
        static const bool bf16_f16 = getenv("AI_Q4X_BF16_F16") && atoi(getenv("AI_Q4X_BF16_F16")) != 0;
        if (bf16_f16 && t.type == BF16 && t.ne[1] > 1) return (int) F16;
        return -1;
    };
    if (!mtp_w_.load(dev, 2ULL << 30, verbose, nullptr, {}, conv, false)) return false;
    const Qwen4ExpSpec & s = spec_;
    auto need = [&](const std::string & name, std::initializer_list<int64_t> ne) -> Tensor {
        const Tensor * t = mtp_w_.find(name);
        if (!t) throw std::runtime_error("qwen4exp mtp: missing tensor " + name);
        size_t d = 0;
        for (int64_t e : ne) {
            if (e >= 0 && t->ne[d] != e) {
                throw std::runtime_error("qwen4exp mtp: " + name + " has shape " + t->shape_str() + ", expected dim " +
                                         std::to_string(d) + " = " + std::to_string(e));
            }
            d++;
        }
        return *t;
    };
    const int     il = s.n_layer;
    const int64_t N = s.n_embd, D = s.hc_dim(), lr = s.hc_lr, hc = s.hc;
    const int64_t Dh = s.head_dim, Hq = s.n_head, Hkv = s.n_head_kv;
    const int64_t E = s.n_expert, F = s.n_ff_exp, Fs = s.n_ff_shexp;
    Qwen4ExpMtp & M = mtp_;
    M.eh_proj    = need(mtp_blk(il, "nextn.eh_proj.weight"), { N, N });
    M.h_proj     = need(mtp_blk(il, "nextn.h_proj.weight"), { N, N });
    M.enorm      = need(mtp_blk(il, "nextn.enorm.weight"), { N });
    M.hnorm      = need(mtp_blk(il, "nextn.hnorm.weight"), { D });
    M.head.norm  = need(mtp_blk(il, "nextn.hc_head_norm.weight"), { D });
    M.head.down  = need(mtp_blk(il, "nextn.hc_head_down.weight"), { D, lr });
    M.head.up    = need(mtp_blk(il, "nextn.hc_head_up.weight"), { lr, D });
    Qwen4ExpLayer & L = M.L;
    L.attn = true;
    L.ple  = false;
    for (int part = 0; part < 2; ++part) {
        op::q4x::HcWeights & h = part ? L.hc_ffn : L.hc_attn;
        const std::string pre  = part ? "hc_ffn_" : "hc_attn_";
        h.norm   = need(mtp_blk(il, (pre + "norm.weight").c_str()), { D });
        h.down   = need(mtp_blk(il, (pre + "down.weight").c_str()), { D, lr });
        h.up     = need(mtp_blk(il, (pre + "up.weight").c_str()), { lr, D });
        h.inject = need(mtp_blk(il, (pre + "inject.weight").c_str()), { D, hc });
    }
    L.at.q      = need(mtp_blk(il, "attn_q.weight"), { N, Dh * Hq * 2 });
    L.at.k      = need(mtp_blk(il, "attn_k.weight"), { N, Dh * Hkv });
    L.at.v      = need(mtp_blk(il, "attn_v.weight"), { N, Dh * Hkv });
    L.at.o      = need(mtp_blk(il, "attn_output.weight"), { Dh * Hq, N });
    L.at.q_norm = need(mtp_blk(il, "attn_q_norm.weight"), { Dh });
    L.at.k_norm = need(mtp_blk(il, "attn_k_norm.weight"), { Dh });
    L.qsa_p.ratio = 0;   // dense attention, no indexer
    L.moe.router    = need(mtp_blk(il, "ffn_gate_inp.weight"), { N, E });
    L.moe.gate_exps = need(mtp_blk(il, "ffn_gate_exps.weight"), { N, F, E });
    L.moe.up_exps   = need(mtp_blk(il, "ffn_up_exps.weight"), { N, F, E });
    L.moe.down_exps = need(mtp_blk(il, "ffn_down_exps.weight"), { F, N, E });
    L.moe.sh_gate   = need(mtp_blk(il, "ffn_gate_shexp.weight"), { N, Fs });
    L.moe.sh_up     = need(mtp_blk(il, "ffn_up_shexp.weight"), { N, Fs });
    L.moe.sh_down   = need(mtp_blk(il, "ffn_down_shexp.weight"), { Fs, N });
    L.moe.sh_score  = need(mtp_blk(il, "ffn_gate_inp_shexp.weight"), { N });
    M.valid = true;
    if (verbose) {
        fprintf(stderr, "[ai] qwen4exp MTP block bound from %s: %.2f GiB on the device\n", path.c_str(),
                mtp_w_.bytes() / 1073741824.0);
    }
    return true;
}

void Qwen4ExpRun::mtp_enable() {
    if (!prog_.mtp().valid) throw std::runtime_error("Qwen4ExpRun::mtp_enable: no MTP block loaded");
    if (mtp_on_) return;
    const Qwen4ExpSpec & s = prog_.spec();
    const int64_t kvw = (int64_t) s.head_dim * s.n_head_kv, HW = s.hc_dim();
    const size_t  kvb = (size_t) (kvw * max_ctx_) * 2;
    // mtp_H_ holds the pending row + one window (1 + kDoorbellT rows), then R2 (one row)
    constexpr int64_t HR = 1 + kDoorbellT;
    mtp_buf_ = dev_.alloc_buffer(2 * kvb + (size_t) (HW * (HR + 1)) * 4 + 4096, "q4x.mtp");
    char * base = (char *) dev_.buffer_base(mtp_buf_);
    cudaMemset(base, 0, 2 * kvb);
    auto mk = [&](int type, int64_t n0, int64_t n1, size_t off) {
        Tensor t; t.buf = mtp_buf_; t.type = type; t.ne[0] = n0; t.ne[1] = n1; set_contiguous_strides(t); t.data = base + off; return t;
    };
    mtp_kv_.k = mk(F16, kvw, max_ctx_, 0);
    mtp_kv_.v = mk(F16, kvw, max_ctx_, kvb);
    mtp_H_    = mk(F32, HW, HR, 2 * kvb);
    mtp_R2_   = mk(F32, HW, 1, 2 * kvb + (size_t) (HW * HR) * 4);
    cudaMemset(mtp_H_.data, 0, (size_t) (HW * (HR + 1)) * 4);
    mtp_on_ = true;
    fprintf(stderr, "[q4x] MTP drafting on: head K/V %.1f MiB\n", 2 * kvb / 1048576.0);
}

// One MTP layer pass over T rows: H f32 [N*hc, T], tokens toks[0..T), positions p0..p0+T-1 (= the head's K/V cells).
// full = false: only the attention half (writes the K/V columns; the fill). full = true (T = 1): the whole layer and
// the head; returns the argmax and leaves R2 in mtp_R2_.
int32_t Qwen4ExpRun::mtp_layer(const Tensor & H, const int32_t * toks, int64_t p0, int64_t T, bool full, float * p_top,
                               bool dev_mask) {
    const Qwen4ExpSpec & s = prog_.spec();
    // the visible cells padded to whole FATTN_KQ_STRIDE (256) tiles, as the trunk's envelope F and llama.cpp's KV views
    // are: with an unpadded n_kv, ggml's FA reads past the mask row and drafts differ run to run
    const int64_t n_kv = std::min<int64_t>(max_ctx_, (p0 + T + 255) / 256 * 256), V = s.n_vocab;
    // Device::upload does not order against the compute stream: the workspace below may still be in use by the
    // previous round's kernels, so drain the stream before staging
    dev_.synchronize();
    Device::Scope sc(dev_);
    Tensor ids_t = dev_.ws_alloc(I32, T);
    dev_.upload(ids_t, toks, 0, (size_t) T * 4);

    std::vector<int32_t>  pos4((size_t) (4 * T));
    std::vector<int64_t>  cells((size_t) T);
    // dev_mask: the n_kv x T mask is written on the device (for the prompt fill's wide chunks, where a host build costs n_kv * T per call)
    std::vector<uint16_t> mask(dev_mask ? 0 : (size_t) (n_kv * T));
    mtp_ingress(p0, T, dev_mask ? 0 : n_kv, pos4.data(), cells.data(), mask.data());
    Tensor pos4_t  = dev_.ws_alloc(I32, 4 * T);
    Tensor cells_t = dev_.ws_alloc(I64, T);
    Tensor mask_t  = dev_.ws_alloc(F16, n_kv, T);
    dev_.upload(pos4_t, pos4.data(), 0, pos4.size() * 4);
    dev_.upload(cells_t, cells.data(), 0, cells.size() * 8);
    if (dev_mask) {
        uint16_t * mp = (uint16_t *) mask_t.data;
        void *     st = adapter::stream(dev_);
        kern::q4x_causal_mask(st, mp, n_kv, T, p0);
        kern::q4x_mtp_mask_col0(st, mp, n_kv, T, p0);
    } else {
        dev_.upload(mask_t, mask.data(), 0, mask.size() * 2);
    }
    // every upload above is staged before the first kernel of this pass is queued (uploads are unordered against the
    // stream: staging after queued Calls could overwrite workspace a still-queued kernel reads)
    const Tensor lg = mtp_build(H, ids_t, pos4_t, cells_t, mask_t, T, n_kv, full);
    if (!full) {
        dev_.synchronize();   // the fill's K/V writes land before the scope's workspace is reused
        return -1;
    }
    const bool mapped = mtp_mapped();
    // argmax + its softmax probability on the device: 8 bytes back instead of the V-float row and a host softmax
    static const bool dev_argmax = !getenv("AI_Q4X_MTP_HOST_ARGMAX") || atoi(getenv("AI_Q4X_MTP_HOST_ARGMAX")) == 0;
    if (dev_argmax) {
        const op::q4x::ArgmaxP am = op::q4x::argmax_p(dev_, lg);
        int32_t idx = 0;
        float   pp  = 0;
        dev_.download(am.idx, &idx, 0, 4);
        dev_.download(am.p, &pp, 0, 4);
        if (p_top) *p_top = pp;
        return mapped ? draft_map_[(size_t) idx] : idx;
    }
    std::vector<float> row((size_t) V);
    dev_.synchronize();
    dev_.download(lg, row.data(), 0, row.size() * 4);
    const auto  it  = std::max_element(row.begin(), row.end());
    if (p_top) {   // softmax probability of the argmax: 1 / sum_j exp(l_j - l_max)
        const float mx = *it;
        double      z  = 0;
        for (float v : row) z += std::exp((double) (v - mx));
        *p_top = (float) (1.0 / z);
    }
    const int32_t i = (int32_t) (it - row.begin());
    return mapped ? draft_map_[(size_t) i] : i;
}

// host ingress of rows p0 .. p0+T-1 over n_kv visible cells: pos4 [4 T], cells [T], mask [T][n_kv] (f16 0 / -inf)
void Qwen4ExpRun::mtp_ingress(int64_t p0, int64_t T, int64_t n_kv, int32_t * pos4, int64_t * cells, uint16_t * mask) const {
    for (int64_t i = 0; i < T; ++i) {
        const int64_t p = p0 + i;
        pos4[i] = (int32_t) p; pos4[T + i] = (int32_t) p; pos4[2 * T + i] = (int32_t) p;
        pos4[3 * T + i] = 0;
        cells[i] = p;
        // column 0 has no H_{-1}: masked, except for the row at position 0 itself (its output is never used)
        for (int64_t j = 0; j < n_kv; ++j) {
            const bool vis = j <= p && (j >= 1 || p == 0);
            mask[i * n_kv + j] = vis ? 0x0000 : 0xFC00;
        }
    }
}

bool Qwen4ExpRun::mtp_mapped() const {
    return draft_head_.valid();
}

// The MTP layer's ops over device inputs (recordable: no host reads or uploads). full = false: the attention half only
// (the K/V fill), returns an invalid tensor; full = true: the whole layer, R2 copied to mtp_R2_, returns the draft
// head's logits.
Tensor Qwen4ExpRun::mtp_build(const Tensor & H, const Tensor & ids_t, const Tensor & pos4_t, const Tensor & cells_t,
                              const Tensor & mask_t, int64_t T, int64_t n_kv, bool full) {
    const Qwen4ExpSpec & s = prog_.spec();
    const Qwen4ExpMtp &  M = prog_.mtp();
    const Qwen4ExpLayer & L = M.L;
    const int64_t HW = s.hc_dim();
    Tensor E  = op::q4x::embed_hc(dev_, prog_.tok_embd(), ids_t, 1);
    Tensor E2 = E.reshape(s.n_embd, T, 1, 1);
    Tensor R0 = op::q4x::mtp_input(dev_, H, E2, M.eh_proj, M.h_proj, M.enorm, M.hnorm, s.hc, s.rms_eps);
    op::q4x::HcRead r = op::q4x::hc_read(dev_, R0, L.hc_attn, s.rms_eps);
    op::AttnInputs ain;
    ain.write_cells  = cells_t;
    ain.pos4         = pos4_t;
    ain.mask         = mask_t.reshape(n_kv, T, 1, 1);
    ain.n_kv         = n_kv;
    ain.B            = 1;
    ain.region_cells = max_ctx_;
    ain.first_cell   = 0;
    op::q4x::QsaCache noqsa;
    Tensor att = op::q4x::attn_qsa_block(dev_, r.mixed, L.at, ap_, mtp_kv_, ain, L.qsa, L.qsa_p, noqsa);
    if (!full) return Tensor{};
    Tensor R1 = op::q4x::hc_scatter(dev_, R0, att, r.inject);
    op::q4x::HcRead f = op::q4x::hc_read(dev_, R1, L.hc_ffn, s.rms_eps);
    op::MoeParams mp{ s.n_expert, s.n_expert_used };
    // [AI_Q4X_MTP_NATIVE_MOE, default 1] the head's routing and weighted sum through the trunk's native kernels
    // (moe_doorbell): router GEMV + softmax + top-k + renorm in 2 launches instead of ggml's chain with its ONE-block
    // argsort over 512 experts, and the fused weighted sum (ggml's exact op order) instead of a mul + k-1 adds. 5090
    // 2026-11-01: argsort 10 us + ~13 element-wise launches per draft step against Strata's draft step.
    static const bool nat = !getenv("AI_Q4X_MTP_NATIVE_MOE") || atoi(getenv("AI_Q4X_MTP_NATIVE_MOE")) != 0;
    const Tensor & mx = f.mixed;
    const Tensor & rw = L.moe.router;
    const int      mk = s.n_expert_used;
    const int64_t  mN = mx.ne[0], mT = mx.ne[1];
    op::MoeRoute   route;
    Device *       mdv = &dev_;
    if (nat && mT <= 16 && (rw.type == BF16 || rw.type == F16) && rw.contiguous() && mx.contiguous() && rw.ne[0] == mN &&
        rw.ne[1] == s.n_expert && s.n_expert <= 1024) {
        Tensor        lgt = dev_.ws_alloc(F32, s.n_expert, mT);
        route.sel         = dev_.ws_alloc(I32, mk, mT);
        route.weights     = dev_.ws_alloc(F32, 1, mk, mT);
        const void *  wp0 = rw.data;
        const bool    bf  = rw.type == BF16;
        const float * xp0 = (const float *) mx.data;
        float *       lp  = (float *) lgt.data;
        int32_t *     sp0 = (int32_t *) route.sel.data;
        float *       wo  = (float *) route.weights.data;
        const int     E   = s.n_expert;
        adapter::native(dev_, [=]() { kern::q4x_route(adapter::stream(*mdv), wp0, bf, xp0, (int) mN, E, (int) mT, mk, lp, sp0, wo); });
    } else {
        route = op::moe_route(dev_, mx, rw, mp);
    }
    Tensor        shared = op::moe_shared(dev_, f.mixed, L.moe);
    Tensor        exps   = op::q4x::moe_experts_raw(dev_, f.mixed, L.moe.up_exps, L.moe.gate_exps, L.moe.down_exps, route.sel);
    Tensor        y;
    if (nat && exps.contiguous() && route.weights.contiguous() && shared.contiguous()) {
        y                 = dev_.ws_alloc(F32, mN, mT);
        const float * ep  = (const float *) exps.data;
        const float * we  = (const float *) route.weights.data;
        const float * sh  = (const float *) shared.data;
        float *       yo  = (float *) y.data;
        adapter::native(dev_, [=]() { kern::moe_wsum(adapter::stream(*mdv), ep, we, sh, yo, mN, mk, mT); });
    } else {
        y = op::q4x::moe_weighted_sum(dev_, exps, route.weights, shared, s.n_expert_used);
    }
    Tensor        R2     = op::q4x::hc_scatter(dev_, R1, y, f.inject);
    op::copy_into(dev_, R2.reshape(HW, T, 1, 1), mtp_R2_, "q4x mtp R2");
    Tensor h  = op::q4x::hc_head(dev_, R2, M.head, s.rms_eps);
    // AI_DRAFT_VOCAB, set_draft_rows: a compact copy of script-chosen rows (draft_head_ + draft_map_) or a prefix view
    const Tensor & outw = prog_.output();
    const int64_t  pre  = draft_prefix_;
    const Tensor head = mtp_mapped() ? draft_head_
        : pre > 0 && pre < outw.ne[1]
        ? outw.view(outw.ne[0], pre, 1, 1, outw.nb[1], outw.nb[1] * (size_t) pre, outw.nb[1] * (size_t) pre, 0)
        : outw;
    return op::q4x::linear(dev_, head, h);   // fp16 tensor-core dense head
}

// AI_Q4X_MTP_CAPTURE: the pending fill + an n-draft chain as one captured round: per-step ingress (ids, pos4, cells,
// mask over a power-of-two envelope F) in fixed buffers, each step's argmax mapped to its token on the device and fed
// to the next step's embedding (draft_next), so the host waits once per chain instead of twice per step. Keyed by
// (F, fill rows m, n). Returns the n drafts and their probabilities (the caller applies tau).
// [AI_Q4X_MTP_SPLIT] off > 0 continues a chain whose first off steps already ran: no fill, step 0 reads the carried
// H from mtp_R2_ (where the last step left it) and drafts at positions pos + off + i; `pending` is the last draft.
void Qwen4ExpRun::mtp_chain(int32_t pending, int n, int32_t * toks, float * probs, bool fetch, int off) {
    const Qwen4ExpSpec & s = prog_.spec();
    const int64_t HW = s.hc_dim(), pos = (int64_t) hist_.size() + off;
    const int64_t m = off > 0 ? 0 : mtp_fill_m_, p0 = mtp_fill_p0_;
    int64_t F = 256;
    while (F < pos + n) F *= 2;
    F = std::min(F, max_ctx_);
    constexpr int64_t Tm = kDoorbellT;
    if (!mc_ids_.valid()) {
        // [fill: ids Tm | pos4 4 Tm | cells Tm | mask Tm x ctx] + kMtpChain x [ids 1 | pos4 4 | cells 1 | mask ctx] + out
        const size_t ctx = (size_t) max_ctx_;
        const size_t fill_b = 256 + 256 + 256 + (Tm * ctx * 2 + 255) / 256 * 256;
        const size_t step_b = 256 + (ctx * 2 + 255) / 256 * 256;
        const int    b      = dev_.alloc_buffer(fill_b + kMtpChain * step_b + 512, "q4x.mtp_chain");
        char *       base   = (char *) dev_.buffer_base(b);
        auto mk = [&](int type, int64_t n0, char * p) {
            Tensor t; t.buf = b; t.type = type; t.ne[0] = n0; set_contiguous_strides(t); t.data = p; return t;
        };
        mc_ids_   = mk(I32, Tm, base);
        mc_pos4_  = mk(I32, 4 * Tm, base + 256);
        mc_cells_ = mk(I64, Tm, base + 512);
        mc_mask_  = mk(F16, Tm * max_ctx_, base + 768);
        for (int i = 0; i < kMtpChain; ++i) {
            char * q = base + fill_b + (size_t) i * step_b;   // [ids 4 B | pos4 16 B | cells 8 B] in the first 256
            mc_sids_[i]  = mk(I32, 1, q);
            mc_spos4_[i] = mk(I32, 4, q + 64);
            mc_scell_[i] = mk(I64, 1, q + 128);
            mc_smask_[i] = mk(F16, max_ctx_, q + 256);
        }
        mc_tok_ = mk(I32, kMtpChain, base + fill_b + kMtpChain * step_b);
        mc_p_   = mk(F32, kMtpChain, base + fill_b + kMtpChain * step_b + 256);
    }
    // [AI_Q4X_MTP_ASYNC_UP, default 0 until measured] the ingress through a pinned mirror of the device buffer, as
    // stream-ordered async copies: no host sync before the chain (Device::upload is unordered, so the sync path drains
    // the stream first and then issues ~3 + 3n blocking pageable copies while the GPU idles). Same bytes, same order.
    // The mirror is free to rewrite here: every earlier chain's copies completed before its read-back.
    static const bool async_up = getenv("AI_Q4X_MTP_ASYNC_UP") && atoi(getenv("AI_Q4X_MTP_ASYNC_UP")) != 0;
    if (async_up && !mc_pin_) {
        const size_t ctx = (size_t) max_ctx_;
        const size_t tot = 256 + 256 + 256 + (Tm * ctx * 2 + 255) / 256 * 256 + kMtpChain * (256 + (ctx * 2 + 255) / 256 * 256) + 512;
        if (cudaHostAlloc((void **) &mc_pin_, tot, cudaHostAllocDefault) != cudaSuccess) { (void) cudaGetLastError(); mc_pin_ = nullptr; }
    }
    if (async_up && mc_pin_) {
        char *       db = (char *) mc_ids_.data;
        cudaStream_t st = (cudaStream_t) adapter::stream(dev_);
        auto put = [&](const Tensor & t, const void * src, size_t nb) {
            const size_t o = (size_t) ((char *) t.data - db);
            memcpy(mc_pin_ + o, src, nb);
            cudaMemcpyAsync(t.data, mc_pin_ + o, nb, cudaMemcpyHostToDevice, st);
        };
        if (m > 0) {
            const size_t o4 = (size_t) ((char *) mc_pos4_.data - db), oc = (size_t) ((char *) mc_cells_.data - db);
            const size_t om = (size_t) ((char *) mc_mask_.data - db);
            mtp_ingress(p0, m, F, (int32_t *) (mc_pin_ + o4), (int64_t *) (mc_pin_ + oc), (uint16_t *) (mc_pin_ + om));
            put(mc_ids_, hist_.data() + p0, (size_t) m * 4);
            cudaMemcpyAsync(mc_pos4_.data, mc_pin_ + o4, (size_t) (4 * m) * 4, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(mc_cells_.data, mc_pin_ + oc, (size_t) m * 8, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(mc_mask_.data, mc_pin_ + om, (size_t) (F * m) * 2, cudaMemcpyHostToDevice, st);
        }
        put(mc_sids_[0], &pending, 4);
        for (int i = 0; i < n; ++i) {
            const size_t o4 = (size_t) ((char *) mc_spos4_[i].data - db), oc = (size_t) ((char *) mc_scell_[i].data - db);
            const size_t om = (size_t) ((char *) mc_smask_[i].data - db);
            mtp_ingress(pos + i, 1, F, (int32_t *) (mc_pin_ + o4), (int64_t *) (mc_pin_ + oc), (uint16_t *) (mc_pin_ + om));
            // pos4 + cell are 64 bytes apart in one 256-byte header: one copy for both
            cudaMemcpyAsync(mc_spos4_[i].data, mc_pin_ + o4, (size_t) (oc - o4) + 8, cudaMemcpyHostToDevice, st);
            cudaMemcpyAsync(mc_smask_[i].data, mc_pin_ + om, (size_t) F * 2, cudaMemcpyHostToDevice, st);
        }
    } else {
    dev_.synchronize();   // the uploads below are unordered against the stream
    if (m > 0) {
        std::vector<int32_t>  p4((size_t) (4 * m));
        std::vector<int64_t>  ce((size_t) m);
        std::vector<uint16_t> mk((size_t) (F * m));
        mtp_ingress(p0, m, F, p4.data(), ce.data(), mk.data());
        dev_.upload(mc_ids_, hist_.data() + p0, 0, (size_t) m * 4);
        dev_.upload(mc_pos4_, p4.data(), 0, p4.size() * 4);
        dev_.upload(mc_cells_, ce.data(), 0, ce.size() * 8);
        dev_.upload(mc_mask_, mk.data(), 0, mk.size() * 2);
    }
    dev_.upload(mc_sids_[0], &pending, 0, 4);
    for (int i = 0; i < n; ++i) {
        int32_t p4[4]; int64_t ce; std::vector<uint16_t> mk((size_t) F);
        mtp_ingress(pos + i, 1, F, p4, &ce, mk.data());
        dev_.upload(mc_spos4_[i], p4, 0, 16);
        dev_.upload(mc_scell_[i], &ce, 0, 8);
        dev_.upload(mc_smask_[i], mk.data(), 0, mk.size() * 2);
    }
    }
    const bool    mapped = mtp_mapped();
    // [AI_Q4X_DRAFT_ROWS_EN] the compact head's row COUNT can change per request (English prompts get fewer rows): the
    // captured chain bakes it into the head GEMV and argmax, so it is part of the key (rows / 4096, bits 52-58)
    const int64_t drows  = mapped ? draft_head_.ne[1] / 4096 : 0;
    const int64_t key    = (F << 12) | (m << 6) | ((int64_t) n << 1) | (mapped ? 1 : 0) | (off > 0 ? (int64_t) 1 << 50 : 0) |
                           (drows << 52);
    bool warm = false;
    mtp_rounds_.launch(dev_, key, [&]() {
        auto v1 = [](const Tensor & t, int64_t n0, size_t es) { return t.view(n0, 1, 1, 1, n0 * es, n0 * es, n0 * es, 0); };
        if (m > 0) {
            const Tensor Hm = mtp_H_.view(HW, m, 1, 1, (size_t) HW * 4, (size_t) (HW * m) * 4, (size_t) (HW * m) * 4, 0);
            mtp_build(Hm, v1(mc_ids_, m, 4), v1(mc_pos4_, 4 * m, 4), v1(mc_cells_, m, 8),
                      mc_mask_.view(F, m, 1, 1, (size_t) F * 2, (size_t) (F * m) * 2, (size_t) (F * m) * 2, 0), m, F, false);
            const Tensor last  = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, (size_t) (HW * m) * 4);
            const Tensor slot0 = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, 0);
            op::copy_into(dev_, last, slot0, "q4x mtp Hprev");
        }
        Tensor H = off > 0 ? mtp_R2_ : mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, 0);
        for (int i = 0; i < n; ++i) {
            const Tensor lg = mtp_build(H, mc_sids_[i], mc_spos4_[i], mc_scell_[i],
                                        mc_smask_[i].view(F, 1, 1, 1, (size_t) F * 2, (size_t) F * 2, (size_t) F * 2, 0), 1, F, true);
            const int32_t * mp  = mapped ? (const int32_t *) draft_map_dev_.data : nullptr;
            int32_t *       nx  = i + 1 < n ? (int32_t *) mc_sids_[i + 1].data : nullptr;
            int32_t *       ot  = (int32_t *) mc_tok_.data + i;
            float *         op_ = (float *) mc_p_.data + i;
            Device *        dv  = &dev_;
            // [AI_Q4X_DRAFT_ARGMAX_NATIVE, default 1] top-1 + probability + draft_next in two native launches (ggml's
            // argmax runs ONE block over the draft rows: argmax + soft_max + get_rows + draft_next were 81 us a chain
            // of 3 on the 5090, 2026-10-31)
            static const bool am_native = !getenv("AI_Q4X_DRAFT_ARGMAX_NATIVE") || atoi(getenv("AI_Q4X_DRAFT_ARGMAX_NATIVE")) != 0;
            if (am_native && lg.type == F32 && lg.contiguous() && lg.numel() == lg.ne[0]) {
                const int64_t V    = lg.ne[0];
                Tensor        part = dev_.ws_alloc(F32, 3 * (int64_t) kern::q4x_draft_argmax_blocks(V));
                const float * lp   = (const float *) lg.data;
                float *       pp2  = (float *) part.data;
                adapter::native(dev_, [=]() { kern::q4x_draft_argmax_next(adapter::stream(*dv), lp, V, pp2, mp, nx, ot, op_); });
            } else {
                const op::q4x::ArgmaxP am = op::q4x::argmax_p(dev_, lg);
                const int32_t * ip = (const int32_t *) am.idx.data;
                const float *   pp = (const float *) am.p.data;
                adapter::native(dev_, [=]() { kern::q4x_draft_next(adapter::stream(*dv), ip, pp, mp, nx, ot, op_); });
            }
            H = mtp_R2_;
        }
        return mc_tok_;
    }, &warm, true, [&](const int64_t & k2, const auto & e) {
        fprintf(stderr, "[q4x] MTP chain recorded: F=%lld fill=%lld n=%lld%s, %d nodes\n", (long long) ((k2 >> 12) & ((1LL << 38) - 1)),
                (long long) ((k2 >> 6) & 63), (long long) ((k2 >> 1) & 31), (k2 >> 50) & 1 ? " (continuation)" : "",
                e.round->n_nodes());
    });
    if (off == 0) mtp_fill_m_ = 0;
    if (!fetch) return;   // [AI_Q4X_VERIFY_EARLY] read from mc_tok_ after an event, not a stream sync
    dev_.download(mc_tok_, toks, 0, (size_t) n * 4);
    dev_.download(mc_p_, probs, 0, (size_t) n * 4);
}

bool Qwen4ExpRun::set_draft_rows(const std::vector<int32_t> & ids, int64_t prefix) {
    const Tensor & outw = prog_.output();
    draft_prefix_ = ids.empty() ? prefix : 0;
    if (ids.empty()) {
        draft_head_ = Tensor{};
        draft_map_.clear();
        return true;
    }
    const int64_t rows = (int64_t) ids.size();
    for (int32_t id : ids) {
        if (id < 0 || id >= outw.ne[1]) { fprintf(stderr, "[draft-vocab] id %d out of range: refused\n", id); return false; }
    }
    // the same rows already in the buffer (serve calls this per prompt): nothing to copy
    if (draft_head_.valid() && ids == draft_map_) return true;
    // [AI_Q4X_DRAFT_HEAD_TYPE=q4_k|iq4_xs|iq3_s, default: the output head's own type] the DRAFT head's rows in a
    // smaller type (the verify head stays full): fewer bytes per draft step (Q5_K 131072 x 2560 = 231 MB; Q4_K 189,
    // IQ4_XS 178, IQ3_S 144). Output text cannot change (the verify decides every token); acceptance can, so a type
    // ships only through the multilingual acceptance gate. The head is requantised once on the host (Q5_K ->
    // f32 -> type, ggml's CPU quantiser, all threads) and the chosen rows uploaded from there.
    static const int dh_type = []() {
        const char * v = getenv("AI_Q4X_DRAFT_HEAD_TYPE");
        if (!v || !*v) return -1;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t)
            if (ggml_get_type_traits((ggml_type) t)->type_name && strcmp(ggml_type_name((ggml_type) t), v) == 0) return t;
        fprintf(stderr, "[draft-vocab] AI_Q4X_DRAFT_HEAD_TYPE=%s unknown: head type kept\n", v);
        return -1;
    }();
    const bool    rq   = dh_type >= 0 && dh_type != outw.type && outw.contiguous() &&
                         kern::q4x_dense_ok(dh_type, rows, outw.ne[0], 1) && outw.ne[0] % 256 == 0;
    const size_t  rb_src = outw.nb[1];
    const size_t  rb   = rq ? ggml_row_size((ggml_type) dh_type, outw.ne[0]) : rb_src;
    if (rq && draft_host_type_ != dh_type) {
        const auto    t0  = std::chrono::steady_clock::now();
        const int64_t V = outw.ne[1], K = outw.ne[0];
        std::vector<uint8_t> src((size_t) V * rb_src);
        dev_.synchronize();
        if (cudaMemcpy(src.data(), outw.data, src.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            fprintf(stderr, "[draft-vocab] head download failed: head type kept\n");
            return false;
        }
        draft_host_.assign((size_t) V * rb, 0);
        const auto *  tt = ggml_get_type_traits((ggml_type) outw.type);
        const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(32, (int64_t) std::thread::hardware_concurrency()));
        const int64_t per = (V + nt - 1) / nt;
        std::vector<std::thread> th;
        for (int64_t w = 0; w < nt; ++w) {
            const int64_t r0 = w * per, r1 = std::min(V, r0 + per);
            if (r0 >= r1) break;
            th.emplace_back([&, r0, r1]() {   // 256 rows at a time: 2.6 MB of f32 per thread, not the thread's whole range
                constexpr int64_t kChunk = 256;
                std::vector<float> f((size_t) (kChunk * K));
                for (int64_t c0 = r0; c0 < r1; c0 += kChunk) {
                    const int64_t c1 = std::min(r1, c0 + kChunk);
                    for (int64_t r = c0; r < c1; ++r) tt->to_float(src.data() + (size_t) r * rb_src, f.data() + (r - c0) * K, K);
                    ggml_quantize_chunk((ggml_type) dh_type, f.data(), draft_host_.data() + (size_t) c0 * rb, 0, c1 - c0, K, nullptr);
                }
            });
        }
        for (auto & t : th) t.join();
        draft_host_type_ = dh_type;
        fprintf(stderr, "[draft-vocab] draft head requantised %s -> %s (%.0f -> %.0f MB for the whole vocab) in %.2f s\n",
                ggml_type_name((ggml_type) outw.type), ggml_type_name((ggml_type) dh_type), V * rb_src / 1e6, V * rb / 1e6,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    // one buffer for the life of the run, sized at the first call (later calls must not ask for more rows)
    if (draft_buf_ < 0) {
        try {
            // sized for the largest request: AI_Q4X_DRAFT_ROWS (an English prompt first must not cap the rest)
            const char *  ev  = getenv("AI_Q4X_DRAFT_ROWS");
            const int64_t cap = std::max<int64_t>(rows, std::min<int64_t>(outw.ne[1], ev ? atoll(ev) : 131072));
            draft_buf_ = dev_.alloc_buffer((size_t) cap * rb + 256 + (size_t) cap * 4, "q4x.draft_head");   // rows + the device map
            draft_head_cap_ = cap;
        } catch (const std::exception & e) {
            fprintf(stderr, "[draft-vocab] %s: full draft head kept\n", e.what());
            return false;
        }
    }
    if (rows > draft_head_cap_) {
        fprintf(stderr, "[draft-vocab] %lld rows > %lld allocated: refused\n", (long long) rows, (long long) draft_head_cap_);
        return false;
    }
    char * base = (char *) dev_.buffer_base(draft_buf_);
    dev_.synchronize();
    if (rq) {   // gather the rows on the host, one upload
        std::vector<uint8_t> g((size_t) rows * rb);
        for (int64_t i = 0; i < rows; ++i)
            memcpy(g.data() + (size_t) i * rb, draft_host_.data() + (size_t) ids[(size_t) i] * rb, rb);
        cudaMemcpy(base, g.data(), g.size(), cudaMemcpyHostToDevice);
    } else {
        // copy runs of consecutive ids (the output rows are contiguous, nb[1] apart)
        for (int64_t i = 0; i < rows;) {
            int64_t j = i + 1;
            while (j < rows && ids[(size_t) j] == ids[(size_t) (j - 1)] + 1) ++j;
            cudaMemcpy(base + (size_t) i * rb, (const char *) outw.data + (size_t) ids[(size_t) i] * rb, (size_t) (j - i) * rb,
                       cudaMemcpyDefault);
            i = j;
        }
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[draft-vocab] copy failed: %s\n", cudaGetErrorString(e)); return false; }
    Tensor t = outw;
    t.buf    = draft_buf_;
    t.data   = base;
    t.ne[1]  = rows;
    if (rq) t.type = dh_type;
    set_contiguous_strides(t);
    draft_head_ = t;
    draft_map_  = ids;
    Tensor mt;   // the map on the device, after the rows (256-aligned)
    mt.buf  = draft_buf_;
    mt.type = I32;
    mt.ne[0] = rows;
    set_contiguous_strides(mt);
    mt.data = base + ((size_t) draft_head_cap_ * rb + 255) / 256 * 256;
    dev_.upload(mt, ids.data(), 0, (size_t) rows * 4);
    dev_.synchronize();
    draft_map_dev_ = mt;
    return true;
}

// Pending fill: columns p0..p0+m-1 = (H rows slot 0..m-1 of mtp_H_, hist_ tokens p0..p0+m-1); then slot 0 <- slot m.
void Qwen4ExpRun::mtp_flush() {
    if (!mtp_on_ || mtp_fill_m_ <= 0) return;
    const int64_t p0 = mtp_fill_p0_, m = mtp_fill_m_, HW = prog_.spec().hc_dim();
    mtp_fill_m_ = 0;
    const Tensor Hm = mtp_H_.view(HW, m, 1, 1, (size_t) HW * 4, (size_t) (HW * m) * 4, (size_t) (HW * m) * 4, 0);
    mtp_layer(Hm, hist_.data() + p0, p0, m, false);
    const Tensor last = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, (size_t) (HW * m) * 4);
    const Tensor slot0 = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, 0);
    op::copy_into(dev_, last, slot0, "q4x mtp Hprev");
}

// AI_Q4X_MTP_MINP=p (default 0 = off): extend the window by another draft only while that draft's own top-1
// probability is >= p (per draft, not the cumulative product that tau cuts on)
float Qwen4ExpRun::mtp_minp() {
    static const float v = getenv("AI_Q4X_MTP_MINP") ? (float) atof(getenv("AI_Q4X_MTP_MINP")) : 0.0f;
    return v;
}

std::vector<int32_t> Qwen4ExpRun::mtp_draft(int32_t pending, int n, float tau, std::vector<float> * probs) {
    if (!mtp_on_) throw std::runtime_error("Qwen4ExpRun::mtp_draft: mtp_enable first");
    // AI_Q4X_MTP_CAPTURE (default 1; 0 = eager steps): the fill + the whole chain as one captured round (mtp_chain);
    // the tau cut is applied to its results (a cut chain computes the dropped steps anyway). Output is identical to the
    // eager steps; what is left of the draft is device time.
    static const bool capture = !getenv("AI_Q4X_MTP_CAPTURE") || atoi(getenv("AI_Q4X_MTP_CAPTURE")) != 0;
    const int  n_eff = (int) std::min<int64_t>(n, max_ctx_ - 1 - (int64_t) hist_.size());
    // the chain is a RoundCache round (it records from workspace origin 0), so it is used only beside the captured
    // trunk rounds: with AI_Q4X_CAPTURE=0 the eager trunk path faults (illegal address after the first chain).
    static const bool trunk_cap = getenv("AI_Q4X_CAPTURE") && atoi(getenv("AI_Q4X_CAPTURE")) != 0;
    const bool chain = capture && trunk_cap && doorbell_ && n_eff >= 1 && n_eff <= kMtpChain;
    if (!chain) mtp_flush();   // the captured chain runs the pending fill itself
    const int64_t HW = prog_.spec().hc_dim();
    static const bool trace_ck = getenv("AI_Q4X_MTP_CK") != nullptr;
    if (trace_ck) {   // debug: checksums of the carried H and the head K/V up to the pending position
        dev_.synchronize();
        const int64_t pos = (int64_t) hist_.size(), kvw = mtp_kv_.k.ne[0];
        std::vector<float> h((size_t) HW);
        std::vector<uint16_t> kv((size_t) (kvw * (pos + 1)));
        dev_.download(mtp_H_, h.data(), 0, h.size() * 4);
        dev_.download(mtp_kv_.k, kv.data(), 0, kv.size() * 2);
        double sh = 0; uint64_t sk = 0;
        for (float v : h) sh += v;
        for (size_t i = 0; i < kv.size(); ++i) sk = sk * 1099511628211ull + kv[i];
        fprintf(stderr, "[mtp-ck] pos %lld H %.9g K-hash %016llx\n", (long long) pos, sh, (unsigned long long) sk);
    }
    std::vector<int32_t> out;
    int32_t tok = pending;
    int64_t pos = (int64_t) hist_.size();   // the pending token's position
    if (chain) {
        int32_t ct[kMtpChain];
        float   cp[kMtpChain];
        // [AI_Q4X_MTP_SPLIT=s, default 0 = one chain] with draft min-p on, the first s steps as one chain, then the
        // rest only if none of those s drafts failed the cut: a cut at step s no longer pays for steps s+1..n (Strata
        // drafts until the first failing step). Costs one read-back + launch when the window survives. Same drafts.
        static const int split = getenv("AI_Q4X_MTP_SPLIT") ? atoi(getenv("AI_Q4X_MTP_SPLIT")) : 0;
        q4x_rt_mark(0, adapter::stream(dev_));
        if (split > 0 && split < n_eff && (mtp_minp() > 0 || tau > 0)) {
            mtp_chain(pending, split, ct, cp);
            double c0   = 1.0;
            bool   live = true;
            for (int i = 0; i < split && live; ++i) {
                if (tau > 0 && i >= mtp_keep_ && c0 * cp[i] < tau) live = false;
                if (mtp_minp() > 0 && i >= mtp_keep_ && cp[i] < mtp_minp()) live = false;
                c0 *= cp[i];
            }
            if (live) mtp_chain(ct[split - 1], n_eff - split, ct + split, cp + split, true, split);
            else for (int i = split; i < n_eff; ++i) cp[i] = 0.0f;   // never read: the loop below stops first
        } else {
            mtp_chain(pending, n_eff, ct, cp);
        }
        q4x_rt_mark(1, adapter::stream(dev_));
        double cum = 1.0;
        for (int i = 0; i < n_eff; ++i) {
            if (tau > 0 && i >= mtp_keep_ && cum * cp[i] < tau) break;
            if (mtp_minp() > 0 && i >= mtp_keep_ && cp[i] < mtp_minp()) break;
            cum *= cp[i];
            out.push_back(ct[i]);
            if (probs) probs->push_back(cp[i]);
        }
        // [AI_Q4X_ROUND_TIMING] where the cut lands: kept drafts per chain of n_eff computed steps; a chain that kept k < n
        // computed n - k - 1 steps nobody reads (the step after the failing one)
        static const bool cut_stats = getenv("AI_Q4X_ROUND_TIMING") != nullptr;
        if (cut_stats) {
            struct CutHist {
                long long h[kMtpChain + 1][kMtpChain + 1] = {};
                ~CutHist() {
                    long long chains = 0, wasted = 0;
                    for (int a = 1; a <= kMtpChain; ++a)
                        for (int b = 0; b <= a; ++b) { chains += h[a][b]; if (b < a) wasted += h[a][b] * (a - b - 1); }
                    if (!chains) return;
                    fprintf(stderr, "[mtp-cut] %lld chains, %.3f computed steps unread per chain; kept/computed:", chains,
                            (double) wasted / (double) chains);
                    for (int a = 1; a <= kMtpChain; ++a)
                        for (int b = 0; b <= a; ++b)
                            if (h[a][b]) fprintf(stderr, " %d/%d %.1f%%", b, a, 100.0 * (double) h[a][b] / (double) chains);
                    fprintf(stderr, "\n");
                }
            };
            static CutHist ch;
            ++ch.h[n_eff][out.size()];
        }
        return out;
    }
    Tensor  H   = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, 0);
    double cum = 1.0;
    for (int i = 0; i < n && pos + 1 < max_ctx_; ++i, ++pos) {
        float p = 1.0f;
        static const bool trace = getenv("AI_Q4X_MTP_TRACE") != nullptr;
        const int32_t in_tok = tok;
        tok = mtp_layer(H, &tok, pos, 1, true, tau > 0 || mtp_minp() > 0 || probs || trace ? &p : nullptr);
        if (trace) fprintf(stderr, "[mtp-trace] pos %lld step %d in %d -> %d p %.6f\n", (long long) pos, i, in_tok, tok, p);
        if (tau > 0 && i >= mtp_keep_ && cum * p < tau) break;   // always keep the first mtp_keep_ drafts
        if (mtp_minp() > 0 && i >= mtp_keep_ && p < mtp_minp()) break;
        cum *= p;
        out.push_back(tok);
        if (probs) probs->push_back(p);
        H = mtp_R2_;
    }
    return out;
}

}  // namespace ai
