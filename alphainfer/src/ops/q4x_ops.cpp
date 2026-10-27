// qwen4exp (Qwen3.8-Flash-Next) Op bodies over ggml. Contracts and the alphallama code each Op mirrors: q4x_ops.h.
// Every body here follows the alphallama graph (src/models/qwen4exp.cpp) primitive for primitive; none of op::'s
// qwen35 fast paths are reused where they would change the math (the GDN output gate is sigmoid here, silu there).
#include "ops/q4x_ops.h"

#include "ops/adapter.h"
#include "kernels/q4x_gdn_conv.h"
#include "kernels/q4x_hc.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cmath>
#include <stdexcept>

namespace ai::op::q4x {

using adapter::Call;

// dense projections on the fp16 tensor-core kernel

static bool dense_on() {
    static const bool v = !getenv("AI_Q4X_DENSE") || atoi(getenv("AI_Q4X_DENSE")) != 0;
    return v;
}

bool dense_ok(const Tensor & w, const Tensor & x) {
    if (!dense_on() || !w.valid() || !x.valid() || x.type != F32 || w.ne[0] != x.ne[0] || w.ne[2] != 1 || w.ne[3] != 1) return false;
    if (x.nb[0] != 4 || x.ne[3] != 1 || (x.ne[2] != 1 && x.nb[2] != x.nb[1] * (size_t) x.ne[1])) return false;
    if ((size_t) w.nb[1] != ggml_row_size((ggml_type) w.type, w.ne[0])) return false;
    return kern::q4x_dense_ok((int) w.type, w.ne[1], w.ne[0], x.ne[1] * x.ne[2]);
}

// BF16/F16 weights of a small-T projection on the router's GEMV kernel (one launch, fp32 math)
static bool gemv16_ok(const Tensor & w, const Tensor & x) {
    return dense_on() && w.valid() && x.valid() && (w.type == BF16 || w.type == F16) && w.contiguous() && x.contiguous() &&
           x.type == F32 && w.ne[0] == x.ne[0] && w.ne[2] == 1 && x.ne[1] * x.ne[2] <= 16 && w.ne[0] % 512 == 0 &&
           w.ne[0] <= 4096 && ((uintptr_t) w.data & 15) == 0 && ((uintptr_t) x.data & 15) == 0;
}

Tensor linear(Device & d, const Tensor & w, const Tensor & x) {
    if (gemv16_ok(w, x)) {
        const int64_t N = w.ne[1], K = w.ne[0], T = x.ne[1] * x.ne[2];
        Tensor        out = d.ws_alloc(F32, N, T);
        const void *  wp = w.data;
        const bool    bf = w.type == BF16;
        const float * xp = (const float *) x.data;
        float *       op_ = (float *) out.data;
        Device *      dev = &d;
        adapter::native(d, [=]() {
            if (!kern::q4x_gemv16(adapter::stream(*dev), wp, bf, xp, (int) K, (int) N, (int) T, op_)) abort();
        });
        return out;
    }
    if (!dense_ok(w, x)) return op::linear(d, w, x);
    static bool said = false;
    if (!said) { said = true; fprintf(stderr, "[q4x-dense] dense projections on fp16 tensor cores (AI_Q4X_DENSE=1)\n"); }
    const int64_t N = w.ne[1], K = w.ne[0], T = x.ne[1] * x.ne[2];
    Tensor        out = d.ws_alloc(F32, N, T);
    Tensor        scr = d.ws_alloc(F32, (int64_t) (kern::q4x_dense_scratch_bytes(K, T) / 4) + 64, 1);
    const int     ty = (int) w.type;
    const void *  wp = w.data;
    const int64_t rb = (int64_t) w.nb[1], xs = (int64_t) (x.nb[1] / 4);
    const float * xp = (const float *) x.data;
    void *        sp = scr.data;
    float *       op_ = (float *) out.data;
    Device *      dev = &d;
    adapter::native(d, [=]() {
        if (!kern::q4x_dense(adapter::stream(*dev), ty, wp, rb, N, K, xp, xs, T, sp, op_, N)) {
            fprintf(stderr, "[q4x-dense] declined a shape the host check accepted\n");
            abort();
        }
    });
    return out;
}

// hyper-connections

// build_hc_mix (qwen4exp.cpp) at its default ALPHA_HC_FUSE=7: <= 8 rows -> ggml_hc_mix_fused (one op: norm, low-rank
// gate, gated mean, inject); larger -> ggml_hc_norm_inject (ALPHA_HC_PF_NI) + the two low-rank GEMMs + ggml_hc_mix_tail.
// The fused op's CUDA predicate (ggml_cuda_hc_mix_fused_supported) is mirrored here, so a shape it refuses takes the
// prefill chain instead of aborting in the backend.
static bool fused_read_ok(const Tensor & R, const HcWeights & w, int64_t t_max = 8) {
    const int64_t N = R.ne[0], hc = R.ne[1], T = R.ne[2];
    if (hc != 4 || T > t_max || R.type != F32) return false;
    if (w.down.type != w.up.type || (w.down.type != BF16 && w.down.type != Q8_0)) return false;
    const int64_t D = N * hc, LR = w.down.ne[1];
    if (w.down.type == Q8_0 && (D % 32 != 0 || LR % 32 != 0)) return false;
    if (w.inject.valid() && w.inject.type != BF16 && w.inject.type != F32) return false;
    return D % 8 == 0 && LR % 8 == 0;
}

static HcRead hc_read_impl(Device & d, const Tensor & R, const HcWeights & w, float eps, bool want_inject) {
    const int64_t N = R.ne[0], hc = R.ne[1], T = R.ne[2], D = N * hc;
    if (R.ne[3] != 1 || !R.contiguous()) {
        throw std::runtime_error("q4x::hc_read: R must be a contiguous f32 [N, hc, T]");
    }
    const bool inj = want_inject && w.inject.valid();

    // AI_Q4X_HC_NATIVE (default 1): the same three kernels as ggml_hc_mix_fused's V4 path, launched natively
    // (q4x_hc.cu) with the up-mix rows staged by batched loads. Bit-identical, fewer DRAM latencies per call.
    static const bool hc_native = !getenv("AI_Q4X_HC_NATIVE") || atoi(getenv("AI_Q4X_HC_NATIVE")) != 0;
    // AI_Q4X_HC_TC: the read on fp16 tensor cores (kernels/q4x_hc_tc.cu), cost roughly flat in T; the scalar kernels
    // stop at T = 8. 0 = never (default), 1 = always, 2 = above 8 rows only. Opt-in because above 8 rows the ggml
    // chain's MMQ is about as fast in the full model.
    static const int hc_tc = getenv("AI_Q4X_HC_TC") ? atoi(getenv("AI_Q4X_HC_TC")) : 0;
    const bool tc_rows = (hc_tc == 1 || (hc_tc == 2 && T > 8)) && kern::q4x_hc_tc_ok((int) N, (int) w.down.ne[1], (int) T);
    if (hc_native && fused_read_ok(R, w, tc_rows ? 32 : 8) && w.down.type == Q8_0 && kern::q4x_hc_chunk((int) N) > 0 &&
        w.down.ne[1] % 8 == 0 && w.down.ne[1] <= 512 && w.norm.type == F32 && w.norm.contiguous()) {
        const int64_t LR = w.down.ne[1];
        int64_t part_n = 0, pss_n = 0;
        kern::q4x_hc_ws((int) N, (int) LR, (int) T, inj, &part_n, &pss_n);
        Tensor out  = d.ws_alloc(F32, (N + hc) * T);
        Tensor part = d.ws_alloc(F32, part_n), pss = d.ws_alloc(F32, pss_n);
        Tensor lo   = d.ws_alloc(F32, LR * T), scl = d.ws_alloc(F32, hc * T);
        const char * wd = (const char *) w.down.data, * wu = (const char *) w.up.data;
        const char * wi = inj ? (const char *) w.inject.data : nullptr;
        const int64_t dnb = (int64_t) w.down.nb[1], unb = (int64_t) w.up.nb[1], inb = inj ? (int64_t) w.inject.nb[1] : 0;
        const int     wbf = inj && w.inject.type == BF16;
        const float * xp = (const float *) R.data, * wnp = (const float *) w.norm.data;
        float * pp = (float *) part.data, * sp = (float *) pss.data, * lp = (float *) lo.data, * cp = (float *) scl.data;
        float * op_ = (float *) out.data;
        const int Ni = (int) N, LRi = (int) LR, Ti = (int) T;
        Device * dev = &d;
        if (tc_rows) {
            Tensor tsc = d.ws_alloc(F32, (int64_t) (kern::q4x_hc_tc_scratch_bytes(Ni, Ti) / 4) + 64, 1);
            void * tsp = tsc.data;
            adapter::native(d, [=]() {
                kern::q4x_hc_read_tc(adapter::stream(*dev), wd, dnb, wu, unb, wi, inb, wbf, xp, wnp, pp, sp, lp, cp, op_, Ni, LRi, Ti, eps, tsp);
            });
        } else {
            adapter::native(d, [=]() {
                kern::q4x_hc_read(adapter::stream(*dev), wd, dnb, wu, unb, wi, inb, wbf, xp, wnp, pp, sp, lp, cp, op_, Ni, LRi, Ti, eps);
            });
        }
        HcRead r;
        r.mixed = out.view(N, T, 1, 1, (size_t) N * 4, (size_t) N * T * 4, (size_t) N * T * 4, 0);
        if (inj) {
            r.inject = out.view(hc, T, 1, 1, (size_t) hc * 4, (size_t) hc * T * 4, (size_t) hc * T * 4, (size_t) N * T * 4);
        }
        return r;
    }

    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor *  x = c.in(R, "hc_x");
    HcRead r;

    if (fused_read_ok(R, w)) {
        ggml_tensor * fz = ggml_hc_mix_fused(g, x, c.in(w.norm, "hc_norm"), c.in(w.down, "hc_down"), c.in(w.up, "hc_up"),
                                             inj ? c.in(w.inject, "hc_inject") : nullptr, eps);
        const Tensor f = c.run(fz);   // 1-D [(N + hc) * T]: mixed [N, T] then inject [hc, T]
        r.mixed = f.view(N, T, 1, 1, (size_t) N * 4, (size_t) N * T * 4, (size_t) N * T * 4, 0);
        if (inj) {
            r.inject = f.view(hc, T, 1, 1, (size_t) hc * 4, (size_t) hc * T * 4, (size_t) hc * T * 4, (size_t) N * T * 4);
        }
        return r;
    }

    ggml_tensor * xn     = nullptr;
    ggml_tensor * inject = nullptr;
    const bool ni_ok = hc == 4 && w.norm.type == F32 && (!inj || w.inject.type == BF16 || w.inject.type == F32);
    if (ni_ok) {
        ggml_tensor * ni = ggml_hc_norm_inject(g, x, c.in(w.norm, "hc_norm"), inj ? c.in(w.inject, "hc_inject") : nullptr, eps);
        xn = ggml_view_2d(g, ni, D, T, (size_t) D * sizeof(float), 0);
        if (inj) {
            inject = ggml_view_2d(g, ni, hc, T, (size_t) hc * sizeof(float), (size_t) D * T * sizeof(float));
        }
    } else {
        // grouped RMSNorm: reduce over one stream, then the [D] gamma (the converter folded it to 1 + w)
        xn = ggml_mul(g, ggml_reshape_2d(g, ggml_rms_norm(g, x, eps), D, T), c.in(w.norm, "hc_norm"));
        if (inj) {
            inject = ggml_mul_mat(g, c.in(w.inject, "hc_inject"), xn);
        }
    }
    ggml_tensor * lo       = ggml_silu(g, ggml_scale(g, ggml_mul_mat(g, c.in(w.down, "hc_down"), xn), 1.0f / (float) hc));
    ggml_tensor * gate_pre = ggml_mul_mat(g, c.in(w.up, "hc_up"), lo);
    ggml_tensor * mixed    = ggml_hc_mix_tail(g, xn, gate_pre, (int) hc, 1.0f / (float) hc);
    if (inject) {
        c.late(inject);
    }
    r.mixed = c.run(mixed);
    if (inject) {
        r.inject = c.to_public(inject);
    }
    return r;
}

HcRead hc_read(Device & d, const Tensor & R, const HcWeights & w, float eps) {
    return hc_read_impl(d, R, w, eps, true);
}

Tensor hc_head(Device & d, const Tensor & R, const HcWeights & w, float eps) {
    return hc_read_impl(d, R, w, eps, false).mixed;
}

// build_hc_combine (qwen4exp.cpp), ALPHA_HC_FUSE bit 1: R + repeat(out) * (2 * sigmoid(inject / hc)) as one op.
Tensor hc_scatter(Device & d, const Tensor & R, const Tensor & out, const Tensor & inject) {
    const int64_t hc = R.ne[1];
    Call c(d);
    ggml_context * g = c.ctx();
    return c.run(ggml_hc_combine(g, c.in(R, "hc_res"), c.in(out, "hc_block_out"), c.in(inject, "hc_inject_w"),
                                 1.0f / (float) hc, 2.0f));
}

// build_inp_embd + repeat_4d (qwen4exp.cpp): the hc streams start as copies of the token embedding.
// token_embd may be IQ3_S; ggml's get_rows gathers it.
Tensor embed_hc(Device & d, const Tensor & table, const Tensor & ids, int hc) {
    const int64_t N = table.ne[0], T = ids.numel();
    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor * e = ggml_get_rows(g, c.in(table, "token_embd"), c.in(ids.reshape(T), "tok_ids"));
    return c.run(ggml_repeat_4d(g, ggml_reshape_3d(g, e, N, 1, T), N, hc, T, 1));
}

// gated DeltaNet with the sigmoid output gate

// build_layer_attn_linear (qwen4exp.cpp): the unfused chain. qkv / z / beta / alpha projections, conv history
// ++ new columns -> ssm_conv -> silu, gated_delta_net_ext with L2-norm(q,k), sigmoid(beta) and softplus(alpha + dt) * A
// fused, then RMSNorm(ssm_norm) * sigmoid(z) (build_norm_gated) and ssm_out. The new recurrent state is written in
// place into the committed slot (GGML_GDN_STATE_INPLACE) and the last K-1 conv columns are published, except in record
// mode, which keeps what op::gdn_fold needs (the fold does not touch the output gate, so it applies unchanged).
// AI_Q4X_GDN_CONV_NATIVE (default 1): the same block with the short conv run natively (q4x_gdn_conv.cu) between two
// adapter calls: projections | conv + SiLU over [history | qkv] read in place (+ the record's conv input) | the delta
// net, gated norm and output projection. Removes the concat (a strided transpose gather) and the state copy; every
// value is computed by the same operations as the single-call graph.
static Tensor gdn_block_native(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                               GdnRecord * rec) {
    const int64_t T  = x.ne[1];
    const int64_t B  = s.conv.ne[1];
    const int64_t L  = T / B;
    const int64_t K1 = p.conv_k - 1;
    const int64_t C  = p.channels;
    const int64_t Sk = p.head_k, Hk = p.k_heads;
    const int64_t Sv = p.head_v, Hv = p.v_heads;
    // 1) projections
    Tensor fo_t, qkv_t, z_t, ba_t, beta_t, alpha_t;
    int64_t qkv_ld = C, z_off = 0, z_ld = Sv * Hv;
    {
        Call c(d);
        ggml_context * g  = c.ctx();
        ggml_tensor *  gx = c.in(x, "gdn_in");
        ggml_tensor *  ba = nullptr, * be = nullptr, * al = nullptr;
        // q6_K/q5_K/q4_K/iq4_xs/iq3_s projections on the fp16 tensor-core dense kernel, after the small bf16 ones
        const bool dn = w.qkvz.valid() ? dense_ok(w.qkvz, x) : (dense_ok(w.qkv, x) && dense_ok(w.z, x));
        if (w.ba.valid()) {
            ba = ggml_mul_mat(g, c.in(w.ba, "wba"), gx);
            if (!dn) c.late(ba);
        } else {
            be = ggml_mul_mat(g, c.in(w.beta, "ssm_beta"), gx);
            al = ggml_mul_mat(g, c.in(w.alpha, "ssm_alpha"), gx);
            if (!dn) c.late(be);
            c.late(al);
        }
        if (dn && (w.ba.valid() ? gemv16_ok(w.ba, x) : gemv16_ok(w.beta, x) && gemv16_ok(w.alpha, x))) {
            // the small BF16 gate projections natively too: this call launches nothing
            if (w.ba.valid()) ba_t = linear(d, w.ba, x);
            else { beta_t = linear(d, w.beta, x); alpha_t = linear(d, w.alpha, x); }
            ba = be = al = nullptr;
            if (w.qkvz.valid()) {
                fo_t   = linear(d, w.qkvz, x);
                qkv_ld = fo_t.ne[0];
                z_off  = C;
                z_ld   = fo_t.ne[0];
            } else {
                z_t   = linear(d, w.z, x);
                qkv_t = linear(d, w.qkv, x);
            }
        } else if (dn) {
            if (ba) ba_t = c.run(ba);
            else { beta_t = c.run(be); alpha_t = c.to_public(al); }
            ba = be = al = nullptr;
            if (w.qkvz.valid()) {
                fo_t   = linear(d, w.qkvz, x);
                qkv_ld = fo_t.ne[0];
                z_off  = C;
                z_ld   = fo_t.ne[0];
            } else {
                z_t   = linear(d, w.z, x);
                qkv_t = linear(d, w.qkv, x);
            }
        } else if (w.qkvz.valid()) {
            fo_t   = c.run(ggml_mul_mat(g, c.in(w.qkvz, "wqkvz_fused"), gx));
            qkv_ld = fo_t.ne[0];
            z_off  = C;
            z_ld   = fo_t.ne[0];
        } else {
            ggml_tensor * zz = ggml_mul_mat(g, c.in(w.z, "wqkv_gate"), gx);
            c.late(zz);
            qkv_t = c.run(ggml_mul_mat(g, c.in(w.qkv, "wqkv"), gx));
            z_t   = c.to_public(zz);
        }
        if (ba) ba_t = c.to_public(ba);
        else if (be) { beta_t = c.to_public(be); alpha_t = c.to_public(al); }
    }
    // 2) conv + SiLU (+ record input), state updated in place outside record mode
    Tensor conv_out = d.ws_alloc(F32, C, L, B);
    Tensor conv_in;
    if (rec) conv_in = d.ws_alloc(F32, K1 + L, C, B);
    {
        float *       st  = (float *) s.conv.data;
        const float * qp  = (const float *) (fo_t.valid() ? fo_t.data : qkv_t.data);
        const float * wp  = (const float *) w.conv.data;
        const int64_t wld = w.conv.ne[0];
        float *       op_ = (float *) conv_out.data;
        float *       ri  = rec ? (float *) conv_in.data : nullptr;
        const int     K = p.conv_k, Ci = (int) C, Li = (int) L, Bi = (int) B;
        const bool    ws = rec == nullptr;
        const int64_t ld = qkv_ld;
        Device *      dev = &d;
        adapter::native(d, [=]() { kern::q4x_gdn_conv(adapter::stream(*dev), st, qp, ld, wp, wld, K, op_, ri, Ci, Li, Bi, ws); });
    }
    // 3) delta net, gated norm, output projection
    Call c(d);
    ggml_context * g  = c.ctx();
    ggml_tensor *  co = c.in(conv_out, "gdn_conv_out");
    const size_t es  = ggml_element_size(co);
    const size_t nb1 = (size_t) C * es;
    ggml_tensor * q = ggml_view_4d(g, co, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, 0);
    ggml_tensor * k = ggml_view_4d(g, co, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, es * Sk * Hk);
    ggml_tensor * v = ggml_view_4d(g, co, Sv, Hv, L, B, es * Sv, nb1, nb1 * L, es * 2 * Sk * Hk);
    ggml_tensor * beta = nullptr, * alpha = nullptr;
    if (ba_t.valid()) {
        ggml_tensor * ba_fo = c.in(ba_t, "wba_out");
        const size_t  e     = ggml_element_size(ba_fo);
        beta  = ggml_view_4d(g, ba_fo, 1, Hv, L, B, e, ba_fo->nb[1], ba_fo->nb[1] * L, 0);
        alpha = ggml_view_4d(g, ba_fo, 1, Hv, L, B, e, ba_fo->nb[1], ba_fo->nb[1] * L, (size_t) Hv * e);
    } else {
        beta  = ggml_reshape_4d(g, c.in(beta_t, "ssm_beta_out"), 1, Hv, L, B);
        alpha = ggml_reshape_4d(g, c.in(alpha_t, "ssm_alpha_out"), 1, Hv, L, B);
    }
    ggml_tensor * z = nullptr;
    if (fo_t.valid()) {
        ggml_tensor * fo  = c.in(fo_t, "wqkvz_out");
        const size_t  fnb = fo->nb[1];
        z = ggml_view_3d(g, fo, Sv * Hv, L, B, fnb, fnb * L, (size_t) z_off * ggml_element_size(fo));
    } else {
        z = ggml_reshape_3d(g, c.in(z_t, "wqkv_gate_out"), Sv * Hv, L, B);
    }
    (void) z_ld;
    ggml_tensor * ssm_slot = c.in(s.ssm, "ssm_slot");
    ggml_tensor * state    = ggml_reshape_4d(g, ssm_slot, Sv, Sv, Hv, B);
    const int32_t flags = GGML_GDN_FUSE_L2NORM_QK | GGML_GDN_FUSE_SIGMOID_BETA | GGML_GDN_FUSE_GATE |
                          (rec ? 0 : GGML_GDN_STATE_INPLACE);
    ggml_tensor * res = ggml_gated_delta_net_ext(g, q, k, v, alpha, beta, state, c.in(w.dt_bias, "ssm_dt"),
                                                 c.in(w.a, "ssm_a"), p.eps, flags);
    const auto rs = [&](int64_t n) { return ggml_row_size(res->type, n); };
    ggml_tensor * out = ggml_view_4d(g, res, Sv, Hv, L, B, rs(Sv), rs(Sv * Hv), rs(Sv * Hv * L), 0);
    ggml_tensor * normed = ggml_mul(g, ggml_rms_norm(g, out, p.eps), c.in(w.norm, "ssm_norm"));
    ggml_tensor * gated  = ggml_mul(g, normed, ggml_sigmoid(g, ggml_reshape_4d(g, z, Sv, Hv, L, B)));
    ggml_tensor * y      = ggml_mul_mat(g, c.in(w.out, "ssm_out"), ggml_reshape_3d(g, gated, Sv * Hv, L, B));
    Tensor y_t = c.run(ggml_reshape_2d(g, y, y->ne[0], T));
    if (rec) {
        rec->T          = T;
        rec->L          = L;
        rec->B          = B;
        rec->conv_input = conv_in;
        rec->conv_out   = conv_out;
        if (ba_t.valid()) {
            rec->ba = ba_t;
        } else {
            // the record holds them as the single-call graph does, [1, Hv, L, B]: gdn_fold reads the per-head stride
            // as nb[1] and the per-column stride as nb[2], so the 2-D [Hv, T] projection outputs would give it wrong gates
            rec->beta  = beta_t.reshape(1, Hv, L, B);
            rec->alpha = alpha_t.reshape(1, Hv, L, B);
        }
    }
    return y_t;
}

static Tensor gdn_block_orig(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                             GdnRecord * rec);

// AI_Q4X_GDN_REC_CHECK=1 (debug, eager only): in record rounds run both the single-call block and the native one (record
// mode has no side effects) and compare y and the record bitwise; the native result is used.
static void gdn_rec_compare(Device & d, const Tensor & y0, const GdnRecord & r0, const Tensor & y1, const GdnRecord & r1) {
    d.synchronize();
    auto cmp = [&](const char * name, const Tensor & a, const Tensor & b) {
        if (!a.valid() || !b.valid()) { fprintf(stderr, "[gdn-rec-check] %s: valid %d/%d\n", name, a.valid(), b.valid()); return; }
        const int64_t n = a.numel();
        if (n != b.numel()) { fprintf(stderr, "[gdn-rec-check] %s: numel %lld vs %lld\n", name, (long long) n, (long long) b.numel()); return; }
        std::vector<float> ha((size_t) n), hb((size_t) n);
        auto get = [&](const Tensor & t, std::vector<float> & h) {
            if (t.contiguous()) { d.download(t, h.data(), 0, (size_t) n * 4); return; }
            // strided [ne0, ne1, ne2] f32 view
            std::vector<float> row((size_t) t.ne[0]);
            int64_t o = 0;
            for (int64_t i2 = 0; i2 < t.ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < t.ne[1]; ++i1) {
                    d.download(t, row.data(), (size_t) (i1 * t.nb[1] + i2 * t.nb[2]), (size_t) t.ne[0] * 4);
                    for (int64_t i0 = 0; i0 < t.ne[0]; ++i0) h[(size_t) o++] = row[(size_t) i0];
                }
        };
        get(a, ha); get(b, hb);
        int64_t bad = 0, first = -1;
        for (int64_t i = 0; i < n; ++i) if (memcmp(&ha[(size_t) i], &hb[(size_t) i], 4) != 0) { if (first < 0) first = i; ++bad; }
        fprintf(stderr, "[gdn-rec-check] %-10s ne [%lld,%lld,%lld] vs [%lld,%lld,%lld]: %lld of %lld differ%s", name,
                (long long) a.ne[0], (long long) a.ne[1], (long long) a.ne[2], (long long) b.ne[0], (long long) b.ne[1],
                (long long) b.ne[2], (long long) bad, (long long) n, first >= 0 ? "" : "\n");
        if (first >= 0) fprintf(stderr, " (first at %lld: %.9g vs %.9g)\n", (long long) first, ha[(size_t) first], hb[(size_t) first]);
    };
    cmp("y", y0, y1);
    cmp("conv_input", r0.conv_input, r1.conv_input);
    cmp("conv_out", r0.conv_out, r1.conv_out);
    cmp("ba", r0.ba, r1.ba);
    cmp("beta", r0.beta, r1.beta);
    cmp("alpha", r0.alpha, r1.alpha);
    if (r0.beta.valid() && r1.beta.valid() && (r0.beta.nb[1] != r1.beta.nb[1] || r0.beta.nb[2] != r1.beta.nb[2])) {
        fprintf(stderr, "[gdn-rec-check] beta strides differ: nb1 %zu/%zu nb2 %zu/%zu\n", r0.beta.nb[1], r1.beta.nb[1],
                r0.beta.nb[2], r1.beta.nb[2]);
    }
}

Tensor gdn_block_sigz(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                      GdnRecord * rec) {
    // 1 = native in every round; 2 = native outside record (verify) rounds only; 0 = off
    static const int conv_native = getenv("AI_Q4X_GDN_CONV_NATIVE") ? atoi(getenv("AI_Q4X_GDN_CONV_NATIVE")) : 1;
    const int64_t T0 = x.ne[1], B0 = s.conv.ne[1];
    if (conv_native != 0 && !(conv_native == 2 && rec) && p.conv_k == 4 && B0 > 0 && T0 % B0 == 0 && T0 / B0 <= 64 && w.conv.type == F32 &&
        w.conv.contiguous() && w.conv.ne[0] == p.conv_k && w.conv.ne[1] == p.channels && s.conv.type == F32 &&
        s.conv.contiguous() && x.type == F32 && p.channels == 2 * p.head_k * p.k_heads + p.head_v * p.v_heads) {
        static const bool check = getenv("AI_Q4X_GDN_REC_CHECK") && atoi(getenv("AI_Q4X_GDN_REC_CHECK")) != 0;
        if (check && rec && !op::recording()) {
            GdnRecord r0;
            const Tensor y0 = gdn_block_orig(d, x, w, p, s, &r0);
            const Tensor y1 = gdn_block_native(d, x, w, p, s, rec);
            gdn_rec_compare(d, y0, r0, y1, *rec);
            return y1;
        }
        return gdn_block_native(d, x, w, p, s, rec);
    }
    return gdn_block_orig(d, x, w, p, s, rec);
}

static Tensor gdn_block_orig(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                             GdnRecord * rec) {
    Call c(d);
    ggml_context * g = c.ctx();

    const int64_t T  = x.ne[1];
    const int64_t B  = s.conv.ne[1];
    const int64_t L  = B > 0 ? T / B : 0;
    const int64_t K1 = p.conv_k - 1;
    const int64_t C  = p.channels;
    const int64_t Sk = p.head_k, Hk = p.k_heads;
    const int64_t Sv = p.head_v, Hv = p.v_heads;
    if (B <= 0 || L * B != T) {
        throw std::runtime_error("q4x::gdn_block_sigz: rows are not a multiple of the stream count");
    }
    if (C != 2 * Sk * Hk + Sv * Hv) {
        throw std::runtime_error("q4x::gdn_block_sigz: channels != 2*Sk*Hk + Sv*Hv");
    }

    ggml_tensor * gx = c.in(x, "gdn_in");

    ggml_tensor * qkv = nullptr;
    ggml_tensor * z   = nullptr;   // [Sv*Hv, L, B]
    if (w.qkvz.valid()) {
        ggml_tensor * fo  = ggml_mul_mat(g, c.in(w.qkvz, "wqkvz_fused"), gx);
        const size_t  nb1 = fo->nb[1];
        qkv = ggml_view_3d(g, fo, C, L, B, nb1, nb1 * L, 0);
        z   = ggml_view_3d(g, fo, Sv * Hv, L, B, nb1, nb1 * L, (size_t) C * ggml_element_size(fo));
    } else {
        qkv = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.qkv, "wqkv"), gx), C, L, B);
        z   = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.z, "wqkv_gate"), gx), Sv * Hv, L, B);
    }
    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    ggml_tensor * ba_fo = nullptr;
    if (w.ba.valid()) {
        ba_fo = ggml_mul_mat(g, c.in(w.ba, "wba"), gx);
        const size_t e = ggml_element_size(ba_fo);
        beta  = ggml_view_4d(g, ba_fo, 1, Hv, L, B, e, ba_fo->nb[1], ba_fo->nb[1] * L, 0);
        alpha = ggml_view_4d(g, ba_fo, 1, Hv, L, B, e, ba_fo->nb[1], ba_fo->nb[1] * L, (size_t) Hv * e);
    } else {
        beta  = ggml_reshape_4d(g, ggml_mul_mat(g, c.in(w.beta, "ssm_beta"), gx), 1, Hv, L, B);
        alpha = ggml_reshape_4d(g, ggml_mul_mat(g, c.in(w.alpha, "ssm_alpha"), gx), 1, Hv, L, B);
    }

    // conv history [K-1, C] ++ new columns [T, C] -> [K-1+T, C]; the last K-1 columns become the new history
    ggml_tensor * conv_slot   = c.in(s.conv, "conv_slot");
    ggml_tensor * conv_states = ggml_reshape_3d(g, conv_slot, K1, C, B);
    ggml_tensor * conv_input  = ggml_concat(g, conv_states, ggml_transpose(g, qkv), 0);
    ggml_tensor * last_cols   = ggml_view_3d(g, conv_input, K1, C, B, conv_input->nb[1], conv_input->nb[2],
                                             (conv_input->ne[0] - K1) * ggml_element_size(conv_input));
    if (!rec) {
        c.effect(ggml_cpy(g, last_cols, ggml_view_1d(g, conv_slot, K1 * C * B, 0)));
    }

    ggml_tensor * conv_out = ggml_silu(g, ggml_ssm_conv(g, conv_input, c.in(w.conv, "ssm_conv1d")));

    const size_t es  = ggml_element_size(conv_out);
    const size_t nb1 = (size_t) C * es;
    ggml_tensor * q = ggml_view_4d(g, conv_out, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, 0);
    ggml_tensor * k = ggml_view_4d(g, conv_out, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, es * Sk * Hk);
    ggml_tensor * v = ggml_view_4d(g, conv_out, Sv, Hv, L, B, es * Sv, nb1, nb1 * L, es * 2 * Sk * Hk);

    ggml_tensor * ssm_slot = c.in(s.ssm, "ssm_slot");
    ggml_tensor * state    = ggml_reshape_4d(g, ssm_slot, Sv, Sv, Hv, B);

    const int32_t flags = GGML_GDN_FUSE_L2NORM_QK | GGML_GDN_FUSE_SIGMOID_BETA | GGML_GDN_FUSE_GATE |
                          (rec ? 0 : GGML_GDN_STATE_INPLACE);
    ggml_tensor * res = ggml_gated_delta_net_ext(g, q, k, v, alpha, beta, state, c.in(w.dt_bias, "ssm_dt"),
                                                 c.in(w.a, "ssm_a"), p.eps, flags);
    const auto rs = [&](int64_t n) { return ggml_row_size(res->type, n); };
    ggml_tensor * out = ggml_view_4d(g, res, Sv, Hv, L, B, rs(Sv), rs(Sv * Hv), rs(Sv * Hv * L), 0);

    // build_norm_gated: RMSNorm(ssm_norm) * sigmoid(z)
    ggml_tensor * normed = ggml_mul(g, ggml_rms_norm(g, out, p.eps), c.in(w.norm, "ssm_norm"));
    ggml_tensor * gated  = ggml_mul(g, normed, ggml_sigmoid(g, ggml_reshape_4d(g, z, Sv, Hv, L, B)));
    ggml_tensor * y      = ggml_mul_mat(g, c.in(w.out, "ssm_out"), ggml_reshape_3d(g, gated, Sv * Hv, L, B));

    if (rec) {
        c.late(conv_input);
        c.late(conv_out);
        if (ba_fo) {
            c.late(ba_fo);
        } else {
            c.late(beta);
            c.late(alpha);
        }
    }
    Tensor y_t = c.run(ggml_reshape_2d(g, y, y->ne[0], T));
    if (rec) {
        rec->T          = T;
        rec->L          = L;
        rec->B          = B;
        rec->conv_input = c.to_public(conv_input);
        rec->conv_out   = c.to_public(conv_out);
        if (ba_fo) {
            rec->ba = c.to_public(ba_fo);
        } else {
            rec->beta  = c.to_public(beta);
            rec->alpha = c.to_public(alpha);
        }
    }
    return y_t;
}

// MoE: softmax top-k renormalised x expert_weights_scale, sigmoid-gated shared expert

// build_layer_ffn (qwen4exp.cpp) -> build_moe_ffn (llama-graph.cpp) with softmax gating, norm_w = true and
// w_scale = expert_weights_scale; the shared expert is build_ffn(SILU, PAR) times sigmoid(ffn_gate_inp_shexp . x).
// Generic ggml top-k (no 256/8 router kernels).
Tensor moe_block_x(Device & d, const Tensor & x, const MoeWeights & w, const MoeParamsX & p) {
    Call c(d);
    ggml_context * g = c.ctx();

    const int64_t T = x.ne[1];
    const int64_t E = x.ne[0];
    const int     k = p.n_used;

    ggml_tensor * gx = c.in(x, "moe_in");

    ggml_tensor * logits  = ggml_mul_mat(g, c.in(w.router, "ffn_gate_inp"), gx);
    ggml_tensor * probs   = ggml_soft_max(g, logits);                                   // [n_expert, T]
    ggml_tensor * sel     = ggml_argsort_top_k(g, probs, k);                            // [k, T]
    ggml_tensor * weights = ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, p.n_expert, T), sel);
    weights = ggml_reshape_2d(g, weights, k, T);
    ggml_tensor * wsum = ggml_clamp(g, ggml_sum_rows(g, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(g, ggml_div(g, weights, wsum), 1, k, T);
    if (p.weights_scale != 0.0f && p.weights_scale != 1.0f) {
        weights = ggml_scale(g, weights, p.weights_scale);
    }
    c.effect(weights);   // routing chain first, as build_moe_ffn expands it early

    ggml_tensor * cur  = ggml_reshape_3d(g, gx, E, 1, T);
    ggml_tensor * up   = ggml_mul_mat_id(g, c.in(w.up_exps, "ffn_up_exps"), cur, sel);
    ggml_tensor * gate = ggml_mul_mat_id(g, c.in(w.gate_exps, "ffn_gate_exps"), cur, sel);
    ggml_tensor * act  = ggml_swiglu_split(g, gate, up);
    ggml_tensor * exps = ggml_mul(g, ggml_mul_mat_id(g, c.in(w.down_exps, "ffn_down_exps"), act, sel), weights);

    ggml_tensor * moe = ggml_view_2d(g, exps, E, T, exps->nb[2], 0);
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(g, moe, ggml_view_2d(g, exps, E, T, exps->nb[2], (size_t) i * exps->nb[1]));
    }

    if (!w.sh_up.valid()) {
        return c.run(moe);
    }
    ggml_tensor * su  = ggml_mul_mat(g, c.in(w.sh_up, "ffn_up_shexp"), gx);
    ggml_tensor * sg  = ggml_mul_mat(g, c.in(w.sh_gate, "ffn_gate_shexp"), gx);
    ggml_tensor * sd  = ggml_mul_mat(g, c.in(w.sh_down, "ffn_down_shexp"), ggml_swiglu_split(g, sg, su));
    ggml_tensor * sgt = ggml_sigmoid(g, ggml_mul_mat(g, c.in(w.sh_score, "ffn_gate_inp_shexp"), gx));   // [1, T]
    return c.run(ggml_add(g, moe, ggml_mul(g, sd, sgt)));
}

// PLE

// build_ple (qwen4exp.cpp), one stream (B = 1, as Qwen4ExpRun runs it):
//   key = ple_key . rows, value = ple_value . rows; grouped RMS norms (per hc stream, [D] gamma) on key and on R;
//   s = sum(key * query) / sqrt(N) per stream -> gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)));
//   gated = repeat(value) * gate; normalized = grouped_norm(gated, norm_conv);
//   [hist ++ normalized] -> depthwise causal conv with dilation ngram (K shifted mul/adds) -> silu;
//   R' = R + (gated + conv). The last `hist` columns become the new history (not in record mode).
Tensor ple_block(Device & d, const Tensor & R, const Tensor & rows, const PleWeights & w, const PleParams & p,
                 const PleState & s, bool rec, Tensor * rec_padded) {
    const int64_t N = R.ne[0], hc = R.ne[1], T = R.ne[2], D = N * hc;
    const int64_t K = p.conv_k, dil = p.ngram, hist = (K - 1) * dil;
    if (s.hist.ne[1] != 1 || s.hist.numel() != hist * D) {
        throw std::runtime_error("q4x::ple_block: history must be f32 [(conv_k - 1) * ngram * D, 1] (one stream)");
    }
    if (w.conv1d.ne[0] != K || w.conv1d.ne[1] != D) {
        throw std::runtime_error("q4x::ple_block: ple_conv1d must be [conv_k, D]");
    }

    Call c(d);
    ggml_context * g = c.ctx();
    const float eps = p.eps;

    ggml_tensor * hidden = c.in(R, "ple_hidden");
    ggml_tensor * emb    = c.in(rows, "ple_embd");

    ggml_tensor * key   = ggml_mul_mat(g, c.in(w.key, "ple_key"), emb);       // [D, T]
    ggml_tensor * value = ggml_mul_mat(g, c.in(w.value, "ple_value"), emb);   // [N, T]

    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * wt) {
        t = ggml_rms_norm(g, ggml_reshape_3d(g, t, N, hc, T), eps);
        t = ggml_mul(g, ggml_reshape_2d(g, t, D, T), wt);
        return ggml_reshape_3d(g, t, N, hc, T);
    };
    key = grouped_norm(key, c.in(w.norm_key, "ple_norm_key"));
    ggml_tensor * query = grouped_norm(hidden, c.in(w.norm_query, "ple_norm_query"));

    ggml_tensor * sc = ggml_scale(g, ggml_sum_rows(g, ggml_mul(g, key, query)), 1.0f / sqrtf((float) N));
    ggml_tensor * mag  = ggml_sqrt(g, ggml_clamp(g, ggml_abs(g, sc), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(g, ggml_mul(g, ggml_sgn(g, sc), mag));   // [1, hc, T]

    ggml_tensor * v3    = ggml_repeat_4d(g, ggml_reshape_3d(g, value, N, 1, T), N, hc, T, 1);
    ggml_tensor * gated = ggml_mul(g, v3, gate);                                // [N, hc, T]

    ggml_tensor * normalized = grouped_norm(ggml_reshape_2d(g, gated, D, T), c.in(w.norm_conv, "ple_norm_conv"));
    normalized = ggml_reshape_2d(g, normalized, D, T);

    ggml_tensor * hist_slot = c.in(s.hist, "ple_conv_state");
    ggml_tensor * state     = ggml_reshape_3d(g, hist_slot, hist, D, 1);
    ggml_tensor * padded    = ggml_concat(g, state, ggml_transpose(g, ggml_reshape_3d(g, normalized, D, T, 1)), 0);
    if (!rec) {
        ggml_tensor * tail = ggml_view_3d(g, padded, hist, D, 1, padded->nb[1], padded->nb[2],
                                          ggml_row_size(padded->type, padded->ne[0] - hist));
        c.effect(ggml_cpy(g, ggml_cont(g, tail), ggml_view_1d(g, hist_slot, hist * D, 0)));
    }

    ggml_tensor * conv1d   = c.in(w.conv1d, "ple_conv1d");
    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t start = hist - (K - 1 - k) * dil;   // tap k reads (K-1-k)*dil positions back
        ggml_tensor * shifted = ggml_cont(g, ggml_transpose(g, ggml_view_3d(g, padded, T, D, 1, padded->nb[1], padded->nb[2],
                                                                            ggml_row_size(padded->type, start))));
        ggml_tensor * wk = ggml_reshape_1d(g, ggml_cont(g, ggml_view_2d(g, conv1d, 1, D, conv1d->nb[1], k * conv1d->nb[0])), D);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(g, wk, GGML_TYPE_F32);
        }
        ggml_tensor * term = ggml_mul(g, shifted, wk);
        conv_out = conv_out ? ggml_add(g, conv_out, term) : term;
    }
    conv_out = ggml_reshape_3d(g, ggml_cont(g, ggml_silu(g, conv_out)), N, hc, T);

    if (rec && rec_padded) c.late(padded);
    Tensor out = c.run(ggml_add(g, hidden, ggml_add(g, gated, conv_out)));
    if (rec && rec_padded) *rec_padded = c.to_public(padded);
    return out;
}

void ple_fold(Device & d, const PleParams & p, const PleState & s, const Tensor & padded, int64_t m1) {
    const int64_t hist = (int64_t) (p.conv_k - 1) * p.ngram, D = padded.ne[1];
    if (m1 < 1 || m1 > padded.ne[0] - hist || s.hist.numel() != hist * D) {
        throw std::runtime_error("q4x::ple_fold: accepted columns out of range");
    }
    Call c(d);
    ggml_context * g    = c.ctx();
    ggml_tensor *  pd   = c.in(padded, "ple_rec_padded");
    ggml_tensor *  slot = c.in(s.hist, "ple_conv_state");
    ggml_tensor *  tail = ggml_view_3d(g, pd, hist, D, 1, pd->nb[1], pd->nb[2], (size_t) m1 * ggml_element_size(pd));
    c.effect(ggml_cpy(g, tail, ggml_view_1d(g, slot, hist * D, 0)));
    c.run_effects_only();
}

// gated full attention + QSA

// build_layer_attn + build_qsa_top_k + build_attn_qsa (qwen4exp.cpp), one stream:
//   indexer: raw key -> cache cell; pool r cells per block (mean of r slices), RMS(k_norm), imrope at the block's first
//   position; q -> RMS(q_norm) -> imrope; score = sum_heads relu(pooled . q) + block bias; select the top
//   min(n_kv, top_k + r - 1) cells by block score + KQ mask (ggml_top_k_expand when n_kv > 1024, as alphallama's
//   ALPHA_QSA_TOPK_EXPAND; the materialised expand + ggml_top_k otherwise).
//   attention: q (+ interleaved output gate) / k / v projections, RMS q/k norms, imrope, K/V -> cache; <= 8 rows: flash
//   attention over the selected cells only (ALPHA_QSA_FA_SPARSE, fattn-sparse set_cells); otherwise dense flash
//   attention under the -inf mask with the selected cells opened (the reference's masked path). Then * sigmoid(gate)
//   and the o-projection. ratio 0 = plain dense attention (no indexer). Pooling is recomputed every call (alphallama's
//   pooled-key cache mode 0); qc.pooled is not used yet.
Tensor attn_qsa_block(Device & d, const Tensor & x, const AttnWeights & w, const AttnParams & p, const KvLayer & kv,
                      const AttnInputs & in, const QsaWeights & qw, const QsaParams & qp, const QsaCache & qc) {
    const int64_t T    = x.ne[1];
    const int64_t D    = p.head_dim, Hq = p.n_head, Hkv = p.n_head_kv;
    const int64_t n_kv = in.n_kv;
    // AI_Q4X_QSA_DENSE=1 (diagnostic): no indexer, dense attention over every cell (not the model's attention)
    static const bool qsa_dense = getenv("AI_Q4X_QSA_DENSE") && atoi(getenv("AI_Q4X_QSA_DENSE")) != 0;
    const int64_t r    = qsa_dense ? 0 : qp.ratio;
    if (in.B != 1 || in.first_cell != 0) {
        throw std::runtime_error("q4x::attn_qsa_block: one stream starting at cell 0 only");
    }
    if (kv.k.type != F16 || kv.v.type != F16) {
        throw std::runtime_error("q4x::attn_qsa_block: f16 KV only");
    }

    // block inputs (native, recorded before the ggml segment)
    const int64_t n_blocks = r > 0 ? (n_kv + r - 1) / r : 0;
    Tensor cell_blk = qc.cell_blk, blk_cells = qc.blk_cells, blk_pos = qc.blk_pos, bias = qc.bias;
    if (r > 0 && !(cell_blk.valid() && blk_cells.valid() && blk_pos.valid() && bias.valid())) {
        cell_blk  = d.ws_alloc(I32, n_kv);
        blk_cells = d.ws_alloc(I32, r * n_blocks);
        blk_pos   = d.ws_alloc(I32, 4 * n_blocks);
        bias      = d.ws_alloc(F32, n_blocks, T);
        void *          st  = adapter::stream(d);
        const int32_t * pp  = (const int32_t *) in.pos4.data;   // section 0 = the positions
        int32_t *       pcb = (int32_t *) cell_blk.data;
        int32_t *       pbc = (int32_t *) blk_cells.data;
        int32_t *       pbp = (int32_t *) blk_pos.data;
        float *         pbi = (float *) bias.data;
        adapter::native(d, [=]() { kern::q4x_qsa_inputs(st, pp, T, n_kv, r, n_blocks, pcb, pbc, pbp, pbi); });
    }
    // AI_Q4X_QSA_SPARSE_FA (default 1): <= 8 rows run flash attention over the selected cells only; 0 = the masked
    // dense path that wider windows take
    static const bool sparse_fa  = !getenv("AI_Q4X_QSA_SPARSE_FA") || atoi(getenv("AI_Q4X_QSA_SPARSE_FA")) != 0;
    const bool        use_sparse = r > 0 && T <= 8 && sparse_fa;
    const Tensor zeros_src = (r > 0 && !use_sparse) ? d.ws_alloc(F32, 1, std::min<int64_t>(n_kv, (int64_t) qp.top_k + r - 1), T)
                                                    : Tensor{};

    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor *  gx = c.in(x, "attn_in");

    int sections[4] = {p.sections[0], p.sections[1], p.sections[2], p.sections[3]};
    ggml_tensor * pos = c.in(in.pos4, "pos4");
    auto rope = [&](ggml_tensor * t, ggml_tensor * pp) {
        return ggml_rope_multi(g, t, pp, nullptr, p.n_rot, sections, p.rope_mode, p.n_ctx_orig,
                               p.rope_base, p.rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    };

    // indexer: top-k cells
    ggml_tensor * top_k = nullptr;
    int64_t       width = 0;
    if (r > 0) {
        const int64_t idim = qp.idx_dim, nih = qp.idx_heads;
        ggml_tensor * keys  = c.in(qc.keys, "indexer_k_cache");
        ggml_tensor * k_raw = ggml_mul_mat(g, c.in(qw.k_proj, "indexer_k_proj"), gx);   // [idim, T]
        c.effect(ggml_set_rows(g, keys, k_raw, c.in(in.write_cells, "k_idxs")));

        ggml_tensor * k_all   = ggml_view_2d(g, keys, idim, n_kv, keys->nb[1], 0);
        ggml_tensor * members = ggml_get_rows(g, k_all, c.in(blk_cells, "qsa_blk_cells"));   // f32 [idim, r*n_blocks]
        members = ggml_reshape_4d(g, members, idim, r, n_blocks, 1);
        ggml_tensor * pl = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(g, ggml_view_3d(g, members, idim, n_blocks, 1, members->nb[2], members->nb[3],
                                                            i * members->nb[1]));
            pl = pl ? ggml_add(g, pl, slice) : slice;
        }
        pl = ggml_scale(g, pl, 1.0f / (float) r);
        pl = ggml_mul(g, ggml_rms_norm(g, ggml_reshape_3d(g, pl, idim, n_blocks, 1), p.eps), c.in(qw.k_norm, "indexer_k_norm"));
        pl = rope(ggml_reshape_3d(g, pl, idim, 1, n_blocks), c.in(blk_pos, "qsa_blk_pos"));
        ggml_tensor * pooled = ggml_reshape_2d(g, pl, idim, n_blocks);

        ggml_tensor * q = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(qw.q_proj, "indexer_q_proj"), gx), idim, nih, T);
        q = rope(ggml_mul(g, ggml_rms_norm(g, q, p.eps), c.in(qw.q_norm, "indexer_q_norm")), pos);

        ggml_tensor * score = ggml_mul_mat(g, pooled, ggml_reshape_2d(g, q, idim, nih * T));
        score = ggml_relu(g, ggml_reshape_3d(g, score, n_blocks, nih, T));
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < nih; ++h) {
            ggml_tensor * slice = ggml_view_2d(g, score, n_blocks, T, score->nb[2], h * score->nb[1]);
            summed = summed ? ggml_add(g, summed, slice) : ggml_cont(g, slice);
        }
        summed = ggml_add(g, summed, c.in(bias, "qsa_bias"));   // [n_blocks, T]

        width = std::min<int64_t>(n_kv, (int64_t) qp.top_k + r - 1);
        ggml_tensor * mask  = c.in(in.mask, "kq_mask");
        ggml_tensor * mask2 = ggml_view_2d(g, mask, n_kv, T, mask->nb[1], 0);
        ggml_tensor * cblk  = c.in(cell_blk, "qsa_cell_blk");
        // AI_Q4X_QSA_EXPAND (default 1): 0 = always the materialised expand + ggml_top_k
        static const bool expand_on = !getenv("AI_Q4X_QSA_EXPAND") || atoi(getenv("AI_Q4X_QSA_EXPAND")) != 0;
        static bool       said      = false;
        if (!said) {
            said = true;
            fprintf(stderr, "[q4x-qsa] ratio %lld, top_k %d: sparse past %lld cells; expand %s\n", (long long) r, qp.top_k,
                    (long long) (qp.top_k + r - 1), expand_on ? "on" : "off");
        }
        if (n_kv > 1024 && expand_on) {
            top_k = ggml_top_k_expand(g, summed, cblk, mask, (int) width);
        } else {
            ggml_tensor * ex = ggml_get_rows(g, ggml_cont(g, ggml_transpose(g, summed)), cblk);   // [T, n_kv]
            ex = ggml_cont(g, ggml_transpose(g, ex));                                                // [n_kv, T]
            ex = ggml_add(g, ex, mask2->type == GGML_TYPE_F32 ? mask2 : ggml_cast(g, mask2, GGML_TYPE_F32));
            top_k = ggml_cont(g, ggml_top_k(g, ex, (int) width));                                    // [width, T]
        }
    }

    // attention
    ggml_tensor * qf   = ggml_mul_mat(g, c.in(w.q, "wq"), gx);
    const size_t  es   = ggml_element_size(qf);
    ggml_tensor * q    = ggml_view_3d(g, qf, D, Hq, T, es * D * 2, es * D * 2 * Hq, 0);
    ggml_tensor * gate = ggml_cont_2d(g, ggml_view_3d(g, qf, D, Hq, T, es * D * 2, es * D * 2 * Hq, es * D), D * Hq, T);
    ggml_tensor * k    = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.k, "wk"), gx), D, Hkv, T);
    ggml_tensor * v    = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.v, "wv"), gx), D, Hkv, T);
    q = rope(ggml_mul(g, ggml_rms_norm(g, q, p.eps), c.in(w.q_norm, "attn_q_norm")), pos);
    k = rope(ggml_mul(g, ggml_rms_norm(g, k, p.eps), c.in(w.k_norm, "attn_k_norm")), pos);

    ggml_tensor * kc  = c.in(kv.k, "k_cache");
    ggml_tensor * vc  = c.in(kv.v, "v_cache");
    ggml_tensor * idx = c.in(in.write_cells, "kv_idxs");
    c.effect(ggml_set_rows(g, kc, ggml_reshape_2d(g, k, D * Hkv, T), idx));
    c.effect(ggml_set_rows(g, vc, ggml_reshape_2d(g, v, D * Hkv, T), idx));

    const size_t  rD    = ggml_row_size(kc->type, D);
    const size_t  rDH   = ggml_row_size(kc->type, D * Hkv);
    ggml_tensor * kview = ggml_view_3d(g, kc, D, Hkv, n_kv, rD, rDH, 0);
    ggml_tensor * vview = ggml_view_3d(g, vc, D, Hkv, n_kv, rD, rDH, 0);
    ggml_tensor * qperm = ggml_permute(g, q, 0, 2, 1, 3);
    ggml_tensor * kp    = ggml_permute(g, kview, 0, 2, 1, 3);
    ggml_tensor * vp    = ggml_permute(g, vview, 0, 2, 1, 3);
    ggml_tensor * mask  = c.in(in.mask, "kq_mask");

    ggml_tensor * fa = nullptr;
    if (use_sparse && D == 256 && Hq == 12 * Hkv) {
        fa = ggml_flash_attn_ext(g, qperm, kp, vp, mask, p.kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        ggml_flash_attn_ext_set_cells(fa, ggml_reshape_2d(g, top_k, width, T));
    } else {
        ggml_tensor * m = mask;
        if (r > 0) {
            // all -inf, open the selected cells, then add the KQ mask (causality, empty cells)
            ggml_tensor * mk   = ggml_view_2d(g, mask, n_kv, T, mask->nb[1], 0);
            ggml_tensor * mf   = mk->type == GGML_TYPE_F32 ? ggml_cont(g, mk) : ggml_cast(g, mk, GGML_TYPE_F32);
            ggml_tensor * all  = ggml_fill(g, mf, -INFINITY);
            ggml_tensor * all4 = ggml_view_4d(g, all, 1, n_kv, T, 1, all->nb[0], all->nb[1], all->nb[1] * T, 0);
            ggml_tensor * zr   = ggml_fill(g, c.in(zeros_src, "qsa_zero_src"), 0.0f);   // [1, width, T]
            ggml_tensor * tk   = ggml_reshape_3d(g, top_k, width, T, 1);
            ggml_tensor * sel  = ggml_set_rows(g, all4, zr, tk);
            sel = ggml_view_2d(g, sel, n_kv, T, sel->nb[2], 0);
            sel = ggml_add(g, sel, mf);
            m   = ggml_cast(g, ggml_reshape_4d(g, sel, n_kv, T, 1, 1), mask->type);
        }
        // AI_Q4X_ATTN_NOFA=1 (diagnostic): softmax(K^T Q) V with plain matmuls instead of flash attention (isolates
        // ggml-cuda's FA kernels for G = 12 heads per KV head)
        static const bool no_fa = getenv("AI_Q4X_ATTN_NOFA") && atoi(getenv("AI_Q4X_ATTN_NOFA")) != 0;
        if (no_fa) {
            ggml_tensor * kpc = ggml_cast(g, ggml_cont(g, kp), GGML_TYPE_F32);              // [D, n_kv, Hkv]
            ggml_tensor * qpc = ggml_cont(g, qperm);                                        // [D, T, Hq]
            ggml_tensor * kq  = ggml_mul_mat(g, kpc, qpc);                                  // [n_kv, T, Hq]
            ggml_tensor * mm  = ggml_cast(g, ggml_view_2d(g, m, n_kv, T, m->nb[1], 0), GGML_TYPE_F32);
            kq = ggml_soft_max_ext(g, kq, mm, p.kq_scale, 0.0f);
            ggml_tensor * vt  = ggml_cast(g, ggml_cont(g, ggml_permute(g, vview, 1, 2, 0, 3)), GGML_TYPE_F32);   // [n_kv, D, Hkv]
            ggml_tensor * kqv = ggml_mul_mat(g, vt, kq);                                    // [D, T, Hq]
            fa = ggml_cont(g, ggml_permute(g, kqv, 0, 2, 1, 3));                            // [D, Hq, T]
        } else {
            fa = ggml_flash_attn_ext(g, qperm, kp, vp, m, p.kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        }
    }
    ggml_tensor * cur = ggml_reshape_2d(g, fa, D * Hq, T);
    cur = ggml_mul(g, cur, ggml_sigmoid(g, gate));
    return c.run(ggml_mul_mat(g, c.in(w.o, "attn_output"), cur));
}

}  // namespace ai::op::q4x

namespace ai::op::q4x {

// Offload mode: the routed experts over device slot banks, unweighted: f32 [N, k, T]. Missed columns are computed on
// a filler slot (valid weights, result discarded) and overwritten by the CPU's rows before the weighted sum.
Tensor moe_experts_raw(Device & d, const Tensor & x, const Tensor & up, const Tensor & gate, const Tensor & down,
                       const Tensor & slots) {
    Call c(d);
    ggml_context * g = c.ctx();
    const int64_t T = x.ne[1], E = x.ne[0];
    ggml_tensor * ids = c.in(slots, "slots");
    ggml_tensor * cur = ggml_reshape_3d(g, c.in(x, "moe_in"), E, 1, T);
    ggml_tensor * u   = ggml_mul_mat_id(g, c.in(up, "up_slots"), cur, ids);
    ggml_tensor * gt  = ggml_mul_mat_id(g, c.in(gate, "gate_slots"), cur, ids);
    return c.run(ggml_cont(g, ggml_mul_mat_id(g, c.in(down, "down_slots"), ggml_swiglu_split(g, gt, u), ids)));
}

// sum_j exps[:, j, t] * weights[j, t] + shared, summed in slot order j = 0..k-1 (as moe_experts).
Tensor moe_weighted_sum(Device & d, const Tensor & exps, const Tensor & weights, const Tensor & shared, int k) {
    Call c(d);
    ggml_context * g = c.ctx();
    const int64_t E = exps.ne[0], T = exps.ne[2];
    ggml_tensor * ex = ggml_mul(g, c.in(exps, "exps_raw"), c.in(weights, "moe_w"));
    ggml_tensor * moe = ggml_view_2d(g, ex, E, T, ex->nb[2], 0);
    for (int i = 1; i < k; ++i) moe = ggml_add(g, moe, ggml_view_2d(g, ex, E, T, ex->nb[2], (size_t) i * ex->nb[1]));
    return c.run(ggml_add(g, moe, c.in(shared, "shexp")));
}

Tensor mtp_input(Device & d, const Tensor & H, const Tensor & E, const Tensor & eh_proj, const Tensor & h_proj,
                 const Tensor & enorm, const Tensor & hnorm, int hc, float eps) {
    const int64_t N = E.ne[0], T = E.ne[1];
    if (H.ne[0] != N * hc || H.ne[1] != T) throw std::runtime_error("q4x::mtp_input: H must be f32 [N*hc, T]");
    Call c(d);
    ggml_context * g  = c.ctx();
    ggml_tensor *  en = ggml_mul(g, ggml_rms_norm(g, c.in(E, "mtp_embd"), eps), c.in(enorm, "mtp_enorm"));
    ggml_tensor *  e2 = ggml_mul_mat(g, c.in(eh_proj, "mtp_eh_proj"), en);                                   // [N, T]
    ggml_tensor *  hn = ggml_mul(g, ggml_rms_norm(g, c.in(H, "mtp_hidden"), eps), c.in(hnorm, "mtp_hnorm"));  // one norm over hc*N
    ggml_tensor *  h2 = ggml_mul_mat(g, c.in(h_proj, "mtp_h_proj"), ggml_reshape_2d(g, hn, N, hc * T));      // per stream
    h2 = ggml_reshape_3d(g, h2, N, hc, T);
    return c.run(ggml_add(g, h2, ggml_reshape_3d(g, e2, N, 1, T)));
}

ArgmaxP argmax_p(Device & d, const Tensor & logits) {
    const int64_t V = logits.ne[0];
    Call c(d);
    ggml_context * g  = c.ctx();
    ggml_tensor *  lg = c.in(logits, "draft_logits");
    ggml_tensor *  am = ggml_argmax(g, lg);                                                   // i32 [1]
    ggml_tensor *  pt = ggml_get_rows(g, ggml_reshape_2d(g, ggml_soft_max(g, lg), 1, V), am);   // f32 [1, 1]
    c.late(am);
    ArgmaxP r;
    r.p   = c.run(pt);
    r.idx = c.to_public(am);
    return r;
}

}  // namespace ai::op::q4x
