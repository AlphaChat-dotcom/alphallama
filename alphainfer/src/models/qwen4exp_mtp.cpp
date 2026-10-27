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

#include <algorithm>
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
int32_t Qwen4ExpRun::mtp_layer(const Tensor & H, const int32_t * toks, int64_t p0, int64_t T, bool full, float * p_top) {
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
    std::vector<uint16_t> mask((size_t) (n_kv * T));
    mtp_ingress(p0, T, n_kv, pos4.data(), cells.data(), mask.data());
    Tensor pos4_t  = dev_.ws_alloc(I32, 4 * T);
    Tensor cells_t = dev_.ws_alloc(I64, T);
    Tensor mask_t  = dev_.ws_alloc(F16, n_kv, T);
    dev_.upload(pos4_t, pos4.data(), 0, pos4.size() * 4);
    dev_.upload(cells_t, cells.data(), 0, cells.size() * 8);
    dev_.upload(mask_t, mask.data(), 0, mask.size() * 2);
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
    op::MoeRoute  route  = op::moe_route(dev_, f.mixed, L.moe.router, mp);
    Tensor        shared = op::moe_shared(dev_, f.mixed, L.moe);
    Tensor        exps   = op::q4x::moe_experts_raw(dev_, f.mixed, L.moe.up_exps, L.moe.gate_exps, L.moe.down_exps, route.sel);
    Tensor        y      = op::q4x::moe_weighted_sum(dev_, exps, route.weights, shared, s.n_expert_used);
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
void Qwen4ExpRun::mtp_chain(int32_t pending, int n, int32_t * toks, float * probs) {
    const Qwen4ExpSpec & s = prog_.spec();
    const int64_t HW = s.hc_dim(), pos = (int64_t) hist_.size();
    const int64_t m = mtp_fill_m_, p0 = mtp_fill_p0_;
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
    const bool    mapped = mtp_mapped();
    const int64_t key    = (F << 12) | (m << 6) | ((int64_t) n << 1) | (mapped ? 1 : 0);
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
        Tensor H = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, 0);
        for (int i = 0; i < n; ++i) {
            const Tensor lg = mtp_build(H, mc_sids_[i], mc_spos4_[i], mc_scell_[i],
                                        mc_smask_[i].view(F, 1, 1, 1, (size_t) F * 2, (size_t) F * 2, (size_t) F * 2, 0), 1, F, true);
            const op::q4x::ArgmaxP am = op::q4x::argmax_p(dev_, lg);
            const int32_t * ip  = (const int32_t *) am.idx.data;
            const float *   pp  = (const float *) am.p.data;
            const int32_t * mp  = mapped ? (const int32_t *) draft_map_dev_.data : nullptr;
            int32_t *       nx  = i + 1 < n ? (int32_t *) mc_sids_[i + 1].data : nullptr;
            int32_t *       ot  = (int32_t *) mc_tok_.data + i;
            float *         op_ = (float *) mc_p_.data + i;
            Device *        dv  = &dev_;
            adapter::native(dev_, [=]() { kern::q4x_draft_next(adapter::stream(*dv), ip, pp, mp, nx, ot, op_); });
            H = mtp_R2_;
        }
        return mc_tok_;
    }, &warm, true, [&](const int64_t & k2, const auto & e) {
        fprintf(stderr, "[q4x] MTP chain recorded: F=%lld fill=%lld n=%lld, %d nodes\n", (long long) (k2 >> 12),
                (long long) ((k2 >> 6) & 63), (long long) ((k2 >> 1) & 31), e.round->n_nodes());
    });
    mtp_fill_m_ = 0;
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
    const size_t  rb   = outw.nb[1];
    for (int32_t id : ids) {
        if (id < 0 || id >= outw.ne[1]) { fprintf(stderr, "[draft-vocab] id %d out of range: refused\n", id); return false; }
    }
    // one buffer for the life of the run, sized at the first call (later calls must not ask for more rows)
    if (draft_buf_ < 0) {
        try {
            draft_buf_ = dev_.alloc_buffer((size_t) rows * rb + 256 + (size_t) rows * 4, "q4x.draft_head");   // rows + the device map
        } catch (const std::exception & e) {
            fprintf(stderr, "[draft-vocab] %s: full draft head kept\n", e.what());
            return false;
        }
        draft_head_cap_ = rows;
    }
    if (rows > draft_head_cap_) {
        fprintf(stderr, "[draft-vocab] %lld rows > %lld allocated: refused\n", (long long) rows, (long long) draft_head_cap_);
        return false;
    }
    char * base = (char *) dev_.buffer_base(draft_buf_);
    dev_.synchronize();
    // copy runs of consecutive ids (the output rows are contiguous, nb[1] apart)
    for (int64_t i = 0; i < rows;) {
        int64_t j = i + 1;
        while (j < rows && ids[(size_t) j] == ids[(size_t) (j - 1)] + 1) ++j;
        cudaMemcpy(base + (size_t) i * rb, (const char *) outw.data + (size_t) ids[(size_t) i] * rb, (size_t) (j - i) * rb,
                   cudaMemcpyDefault);
        i = j;
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[draft-vocab] copy failed: %s\n", cudaGetErrorString(e)); return false; }
    Tensor t = outw;
    t.buf    = draft_buf_;
    t.data   = base;
    t.ne[1]  = rows;
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
        mtp_chain(pending, n_eff, ct, cp);
        double cum = 1.0;
        for (int i = 0; i < n_eff; ++i) {
            if (tau > 0 && i >= mtp_keep_ && cum * cp[i] < tau) break;
            if (mtp_minp() > 0 && i >= mtp_keep_ && cp[i] < mtp_minp()) break;
            cum *= cp[i];
            out.push_back(ct[i]);
            if (probs) probs->push_back(cp[i]);
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
