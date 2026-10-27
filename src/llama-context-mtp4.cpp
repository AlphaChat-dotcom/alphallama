// qwen4exp (Qwen3.8-Flash-Next) MTP draft block: the hand-built draft step and head-KV fill. Built with ALPHA_MTP.
//
// The block is one trunk-shaped layer with hyper-connections (hc parallel residual streams, mixed in and
// scattered back around each sub-block) plus its own input projections and final mixer. The structure follows
// the reference MTP implementation in Strata (mtp.cpp, MIT License):
//
//   R0   = h_proj(rms(H) * hnorm) per stream  +  eh_proj(rms(embed(tok)) * enorm)      H = trunk streams [hc*N]
//   R1   = combine(R0, attn(mix_attn(R0)))          dense gated attention over the head's own K/V cache
//   R2   = combine(R1, moe(mix_ffn(R1)))            routed experts (DRE / CPU path apply) + gated shared expert
//   draft = argmax(output(mix_head(R2)))            R2 is also the hidden the next chained draft starts from
//
// The trunk exports H = its residual streams after the last layer, before the final mixer (qwen4exp.cpp), so the
// carried hidden is hc*n_embd wide (llama_context::mtp_hw). The K/V column for position p depends only on
// (H_{p-1}, token_p) through R0 and the attention mixer, so a whole decode's columns are filled in one pass.
// Everything around these graphs (cache ranges, sequences, positions, the chain) is llama-context.cpp's.

#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>
#include <cstring>
#include <vector>

int64_t llama_context::mtp_hw() const {
    return model.has_mtp4 ? (int64_t) model.hparams.hc_count * model.hparams.n_embd : model.hparams.n_embd;
}

namespace {

struct mtp4_builder {
    ggml_context *        ctx;
    const llama_model &   model;
    const llama_layer &   L;
    int64_t               N, HC;
    float                 eps;

    // the trunk's build_hc_mix: grouped RMSNorm (per stream) + gamma, low-rank gate, gated stream mean
    ggml_tensor * hc_mix(ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down, ggml_tensor * w_up,
                         ggml_tensor * w_inject, ggml_tensor ** inject) {
        const int64_t T = x->ne[2];
        static const bool fuse_read = [] { const char * e = getenv("ALPHA_HC_FUSE"); return ((e ? atoi(e) : 7) & 4) != 0; }();
        if (fuse_read && T <= 8 && HC == 4 && w_down->type == w_up->type &&
            (w_down->type == GGML_TYPE_BF16 || w_down->type == GGML_TYPE_Q8_0)) {
            ggml_tensor * fz = ggml_hc_mix_fused(ctx, x, w_norm, w_down, w_up, inject ? w_inject : nullptr, eps);
            if (inject) *inject = ggml_view_2d(ctx, fz, HC, T, HC*sizeof(float), (size_t) N*T*sizeof(float));
            return ggml_view_2d(ctx, fz, N, T, N*sizeof(float), 0);
        }
        ggml_tensor * xn = ggml_rms_norm(ctx, x, eps);
        xn = ggml_reshape_2d(ctx, xn, N * HC, T);
        xn = ggml_mul(ctx, xn, w_norm);
        ggml_tensor * lo = ggml_mul_mat(ctx, w_down, xn);
        lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float) HC));
        ggml_tensor * gpre = ggml_mul_mat(ctx, w_up, lo);
        ggml_tensor * mixed = ggml_hc_mix_tail(ctx, xn, gpre, (int) HC, 1.0f / (float) HC);
        if (inject) {
            *inject = ggml_mul_mat(ctx, w_inject, xn);
        }
        return mixed;
    }

    ggml_tensor * hc_combine(ggml_tensor * residual, ggml_tensor * block_out, ggml_tensor * inject) {
        return ggml_hc_combine(ctx, residual, block_out, inject, 1.0f / (float) HC, 2.0f);
    }

    // R0 [N, HC, T] from the carried streams Hs [HC*N, T] and the embedded tokens
    // E_in: the embedded rows when the caller already has them (the draft dequantises its one row on the host,
    // so a host-resident tok_embd does not put a CPU split -- and the CPU threadpool -- into every draft)
    ggml_tensor * input(ggml_tensor * Hs, ggml_tensor * Tk, ggml_tensor * E_in = nullptr) {
        const int64_t T = Hs->ne[1];
        ggml_tensor * E  = E_in ? E_in : ggml_get_rows(ctx, model.tok_embd, Tk);                // [N, T]
        ggml_tensor * en = ggml_mul(ctx, ggml_rms_norm(ctx, E, eps), L.nextn.enorm);
        ggml_tensor * e2 = ggml_mul_mat(ctx, L.nextn.eh_proj, en);                              // [N, T]
        ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, Hs, eps), L.nextn.hnorm);           // one norm over hc*N
        ggml_tensor * h2 = ggml_mul_mat(ctx, L.nextn.h_proj, ggml_reshape_2d(ctx, hn, N, HC * T)); // per stream
        h2 = ggml_reshape_3d(ctx, h2, N, HC, T);
        return ggml_add(ctx, h2, ggml_reshape_3d(ctx, e2, N, 1, T));
    }

    // ALPHA_MTP_HEAD_FA: flash attention straight over the F16 head cache: q [n_hd, n_h] -> [n_hd*n_h, 1]. The
    // unfused path copies the whole window (K, V and V^T conts) every draft.
    ggml_tensor * attn_fa(ggml_tensor * q, ggml_tensor * kc, ggml_tensor * vc, int64_t n_hd, int64_t n_hkv, int64_t n_h,
                          int64_t T_use, size_t woff, ggml_tensor * mask, float scale) {
        ggml_tensor * K = ggml_view_3d(ctx, kc, n_hd, T_use, n_hkv, kc->nb[1], ggml_element_size(kc) * n_hd, woff * kc->nb[1]);
        ggml_tensor * V = ggml_view_3d(ctx, vc, n_hd, T_use, n_hkv, vc->nb[1], ggml_element_size(vc) * n_hd, woff * vc->nb[1]);
        ggml_tensor * o = ggml_flash_attn_ext(ctx, ggml_reshape_3d(ctx, q, n_hd, 1, n_h), K, V, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        return ggml_reshape_2d(ctx, o, n_hd * n_h, 1);
    }

    ggml_tensor * rope(ggml_tensor * t, ggml_tensor * pos, const llama_cparams & cp) {
        const auto & hp = model.hparams;
        const int il    = (int) hp.n_layer - 1;
        const int n_rot = (int) hp.n_rot(il);
        const int n_co  = (int) cp.n_ctx_orig_yarn;
        if (hp.rope_type == LLAMA_ROPE_TYPE_MROPE || hp.rope_type == LLAMA_ROPE_TYPE_IMROPE) {
            int sec[GGML_MROPE_SECTIONS];
            std::copy(std::begin(hp.rope_sections), std::begin(hp.rope_sections) + GGML_MROPE_SECTIONS, sec);
            return ggml_rope_multi(ctx, t, pos, nullptr, n_rot, sec, hp.rope_type, n_co, cp.rope_freq_base,
                                   cp.rope_freq_scale, cp.yarn_ext_factor, cp.yarn_attn_factor, cp.yarn_beta_fast,
                                   cp.yarn_beta_slow);
        }
        return ggml_rope_ext(ctx, t, pos, nullptr, n_rot, hp.rope_type, n_co, cp.rope_freq_base, cp.rope_freq_scale,
                             cp.yarn_ext_factor, cp.yarn_attn_factor, cp.yarn_beta_fast, cp.yarn_beta_slow);
    }
};

} // namespace

bool llama_context::mtp_kv_fill_qwen4exp(const std::vector<const float *> & hsrc, const std::vector<llama_token> & ctok,
                                         const std::vector<llama_pos> & cpos, int32_t slot0, int64_t m) {
    const auto & hp = model.hparams;
    const auto & L  = model.mtp4;
    const int64_t N = hp.n_embd, HC = hp.hc_count, HW = N * HC;
    const int64_t n_hd = hp.n_embd_head_k(), n_hkv = hp.n_head_kv(hp.n_layer - 1);
    const int64_t kv_row = n_hd * n_hkv;
    const bool do_rope = hp.rope_type != LLAMA_ROPE_TYPE_NONE;
    const int  n_pos_pe = (int) hp.n_pos_per_embd();
    const bool use_alt = (bool) sched_alt;
    // ALPHA_MTP_FILL_CH (default 1024): rows per fill graph. Small values cost a graph build, scheduler reset and
    // synchronous compute per few rows; the graph is small next to a prompt's buffer
    static const int64_t CH = [] { const char * e = getenv("ALPHA_MTP_FILL_CH"); return e ? std::max<int64_t>(1, atoll(e)) : (int64_t) 1024; }();
    for (int64_t base = 0; base < m; base += CH) {
        const int64_t mb = std::min<int64_t>(CH, m - base);
        ggml_init_params ip = { ggml_graph_overhead() + ggml_tensor_overhead() * 96, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        mtp4_builder B { ctx, model, L, N, HC, hp.f_norm_rms_eps };
        ggml_tensor * Hs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HW, mb); ggml_set_input(Hs);
        // a host-resident token_embd would put a CPU split (and a host sync) into every fill: dequantise the rows
        // here instead, and write the K/V columns through an index tensor so the graph does not depend on position
        // (the CUDA backend can then replay it)
        const bool embd_host = model.tok_embd->buffer && ggml_backend_buffer_is_host(model.tok_embd->buffer) &&
                               ggml_get_type_traits(model.tok_embd->type)->to_float != nullptr;
        ggml_tensor * Tk = nullptr, * Er = nullptr;
        if (embd_host) { Er = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, mb); ggml_set_input(Er); }
        else           { Tk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb);    ggml_set_input(Tk); }
        ggml_tensor * Ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb); ggml_set_input(Ix);
        ggml_tensor * Ps = nullptr;
        if (do_rope) { Ps = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb * n_pos_pe); ggml_set_input(Ps); }
        ggml_tensor * R0 = B.input(Hs, Tk, Er);
        ggml_tensor * mixed = B.hc_mix(R0, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, nullptr, nullptr);
        ggml_tensor * kn = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, mixed), n_hd, n_hkv, mb);
        kn = ggml_mul(ctx, ggml_rms_norm(ctx, kn, B.eps), L.attn_k_norm);
        if (do_rope) kn = B.rope(kn, Ps, cparams);
        ggml_tensor * vn = ggml_mul_mat(ctx, L.wv, mixed);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_k_cache, ggml_reshape_2d(ctx, kn, kv_row, mb), Ix));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_v_cache, ggml_reshape_2d(ctx, vn, kv_row, mb), Ix));
        static const bool ftm = getenv("ALPHA_MTP_FILL_TIMERS") != nullptr;
        const int64_t ft0 = ftm ? ggml_time_us() : 0;
        int64_t ft1 = 0, ft2 = 0;
        if (use_alt) set_draft_active(true);
        ggml_backend_sched_reset(sched.get());
        if (gf_res_prev) gf_res_prev->reset();
        bool ok = false;
        const bool alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
        if (ftm) ft1 = ggml_time_us();
        if (alloc_ok) {
            // contiguous source rows (all but a carried-over boundary row) go up in one set, without a host copy
            const size_t row_b = (size_t) HW * sizeof(float);
            for (int64_t i = 0; i < mb; ) {
                int64_t j = i + 1;
                while (j < mb && hsrc[base + j] == hsrc[base + j - 1] + HW) ++j;
                ggml_backend_tensor_set(Hs, hsrc[base + i], (size_t) i * row_b, (size_t) (j - i) * row_b);
                i = j;
            }
            if (Tk) {
                std::vector<int32_t> tk((size_t) mb);
                for (int64_t i = 0; i < mb; ++i) tk[i] = (int32_t) ctok[base + i];
                ggml_backend_tensor_set(Tk, tk.data(), 0, (size_t) mb * sizeof(int32_t));
            } else {
                std::vector<float> er((size_t) N * mb);
                const ggml_tensor * te = model.tok_embd;
                for (int64_t i = 0; i < mb; ++i)
                    ggml_get_type_traits(te->type)->to_float((const char *) te->data + (size_t) ctok[base + i] * te->nb[1], er.data() + (size_t) i * N, N);
                ggml_backend_tensor_set(Er, er.data(), 0, er.size() * sizeof(float));
            }
            {
                std::vector<int32_t> ix((size_t) mb);
                for (int64_t i = 0; i < mb; ++i) ix[i] = (int32_t) (mtp_kv_base() + slot0 + base + i);
                ggml_backend_tensor_set(Ix, ix.data(), 0, (size_t) mb * sizeof(int32_t));
            }
            if (Ps) {
                std::vector<int32_t> ps((size_t) mb * n_pos_pe);
                for (int j = 0; j < n_pos_pe; ++j)
                    for (int64_t i = 0; i < mb; ++i) ps[(size_t) j * mb + i] = (int32_t) cpos[base + i];
                ggml_backend_tensor_set(Ps, ps.data(), 0, ps.size() * sizeof(int32_t));
            }
            if (ftm) ft2 = ggml_time_us();
            ok = ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS;
        }
        if (ftm) {
            const int64_t ft3 = ggml_time_us();
            fprintf(stderr, "[mtp-fill-t] rows %lld: reset+alloc %.1f inputs %.1f compute %.1f ms (alt=%d)\n", (long long) mb,
                    (ft1 - ft0) / 1000.0, (ft2 - ft1) / 1000.0, (ft3 - ft2) / 1000.0, (int) use_alt);
        }
        if (use_alt) set_draft_active(false);
        ggml_free(ctx);
        if (!ok) return false;
        mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, (int32_t) (slot0 + base + mb - 1));
    }
    return true;
}

llama_token llama_context::mtp_draft_qwen4exp(llama_token e_tok) {
    static const bool dbg = getenv("ALPHA_MTP_DBG") != nullptr;
    const auto & hp = model.hparams;
    const auto & L  = model.mtp4;
    const int64_t N = hp.n_embd, HC = hp.hc_count, HW = N * HC;
    if (!model.has_mtp4 || (int64_t) mtp_hidden_host.size() < HW) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[mtp] qwen4exp: no MTP block or no exported hidden -- not drafting\n"); }
        return LLAMA_TOKEN_NULL;
    }
    const int64_t n_expert = hp.n_expert, n_used = hp.n_expert_used;
    const float   wscale   = hp.expert_weights_scale;
    const int64_t n_hd = hp.n_embd_head_k(), n_h = hp.n_head(hp.n_layer - 1), n_hkv = hp.n_head_kv(hp.n_layer - 1);
    const int64_t n_rep = n_h / n_hkv, kv_row = n_hd * n_hkv;
    if (!mtp_kv_ensure(kv_row, ggml_backend_buffer_get_type(L.wq->buffer))) return LLAMA_TOKEN_NULL;

    static const bool tdbg = getenv("ALPHA_MTP4_TIMERS") != nullptr;
    const int64_t t_b0 = tdbg ? ggml_time_us() : 0;

    // ---- per-call scalars; only (T_use, woff, cache identity, rope) shape the graph
    const int32_t col_pos = mtp_pos_base >= 0 ? mtp_pos_base + 1 + mtp_chain_step : -1;
    const int n_pos_pe = (int) hp.n_pos_per_embd();
    const bool use_rope = col_pos >= 0 && hp.rope_type != LLAMA_ROPE_TYPE_NONE;
    const int32_t lo0 = (mtp_kv_lo >= 0) ? mtp_kv_lo : (col_pos - 1);
    const int32_t pos = (mtp_k_cache && col_pos >= 1 && col_pos - 1 < mtp_kv_cols &&
                         col_pos - 1 <= mtp_kv_hi + 1 && col_pos - 1 >= lo0) ? col_pos - 1 : -1;
    int32_t kv_idx_v = -1;
    int64_t kq_T = 0, kq_T_pad = -1;
    size_t  woff = 0;
    if (pos >= 0) {
        kv_idx_v = mtp_kv_base() + pos;
        kq_T = (int64_t) (pos - lo0 + 1);
        woff = (size_t) (mtp_kv_base() + lo0);
        const int64_t avail = (int64_t) mtp_kv_cols - (int64_t) lo0;
        const int64_t padq = (mtp_k_cache->type == GGML_TYPE_F16) ? 256 : 64;   // FA: whole FATTN_KQ_STRIDE tiles
        kq_T_pad = std::min<int64_t>(((kq_T + padq - 1) / padq) * padq, avail);
        if (kq_T_pad < kq_T) kq_T_pad = kq_T;
    }
    const bool wrote_col = pos >= 0;
    const bool embd_host = model.tok_embd->buffer && ggml_backend_buffer_is_host(model.tok_embd->buffer) &&
                           ggml_get_type_traits(model.tok_embd->type)->to_float != nullptr;

    // ALPHA_MTP_D4_CACHE (default on): keep the draft graph built and allocated on its own scheduler: a draft
    // whose shape key matches only uploads its inputs and recomputes, so the CUDA backend can replay it.
    // sched_alt is not used because the head-KV fill and the ReplaySSM fold share it and would force rebuilds.
    static const bool use_cache = [] { const char * e = getenv("ALPHA_MTP_D4_CACHE"); return !e || atoi(e) != 0; }();
    if (use_cache && !sched_d4) {
        sched_d4.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                              graph_max_nodes(1), false, cparams.op_offload));
    }
    ggml_backend_sched_t sd = use_cache ? sched_d4.get() : nullptr;
    const bool hit = sd && d4c.valid && d4c.T_use == kq_T_pad && d4c.woff == woff &&
                     d4c.kc == (const void *) mtp_k_cache && (d4c.rope != nullptr) == use_rope;
    if (tdbg && sd && !hit) {
        static int nm = 0;
        if (nm++ < 12) fprintf(stderr, "[mtp4-t] draft cache miss: valid=%d T_use %lld->%lld woff %zu->%zu kc %d rope %d\n",
                               (int) d4c.valid, (long long) d4c.T_use, (long long) kq_T_pad, d4c.woff, woff,
                               (int) (d4c.kc == (const void *) mtp_k_cache), (int) ((d4c.rope != nullptr) == use_rope));
    }
    if (!hit) {
    d4c.valid = false;
    ggml_init_params ip = { ggml_graph_overhead() + ggml_tensor_overhead() * 320, nullptr, true };
    if (!mtp_draft_ctx) mtp_draft_ctx = ggml_init(ip); else ggml_reset(mtp_draft_ctx);
    ggml_context * ctx = mtp_draft_ctx;
    if (!ctx) return LLAMA_TOKEN_NULL;
    mtp4_builder B { ctx, model, L, N, HC, hp.f_norm_rms_eps };
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * H    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HW, 1); ggml_set_input(H);
    ggml_tensor * etok = nullptr, * erow = nullptr;
    if (embd_host) { erow = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, 1); ggml_set_input(erow); }
    else           { etok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);    ggml_set_input(etok); }
    ggml_tensor * R0   = B.input(H, etok, erow);                                            // [N, HC, 1]

    // ---- attention: mixer, q|gate, own K/V append, dense attention over [lo, pos]
    ggml_tensor * inj_a = nullptr;
    ggml_tensor * xa = B.hc_mix(R0, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, &inj_a);  // [N, 1]
    ggml_tensor * qg = ggml_mul_mat(ctx, L.wq, xa);                                         // [n_hd*2*n_h, 1]
    ggml_tensor * q  = ggml_cont(ctx, ggml_view_2d(ctx, qg, n_hd, n_h, ggml_element_size(qg) * n_hd * 2, 0));
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, B.eps), L.attn_q_norm);
    ggml_tensor * gate = ggml_cont_2d(ctx, ggml_view_2d(ctx, qg, n_hd, n_h, ggml_element_size(qg) * n_hd * 2,
                                                        ggml_element_size(qg) * n_hd), n_hd * n_h, 1);
    ggml_tensor * kn = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, L.wk, xa), n_hd, n_hkv);
    kn = ggml_mul(ctx, ggml_rms_norm(ctx, kn, B.eps), L.attn_k_norm);
    ggml_tensor * vn = ggml_mul_mat(ctx, L.wv, xa);
    ggml_tensor * rope_pos_t = nullptr;
    if (use_rope) {
        rope_pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos_pe);
        ggml_set_input(rope_pos_t);
        q  = ggml_reshape_2d(ctx, B.rope(ggml_reshape_3d(ctx, q,  n_hd, n_h,   1), rope_pos_t, cparams), n_hd, n_h);
        kn = ggml_reshape_2d(ctx, B.rope(ggml_reshape_3d(ctx, kn, n_hd, n_hkv, 1), rope_pos_t, cparams), n_hd, n_hkv);
    }
    ggml_tensor * kv_idx_t = nullptr, * kq_mask_t = nullptr;
    ggml_tensor * att_out = nullptr;
    if (pos >= 0) {
        kv_idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(kv_idx_t);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_k_cache, ggml_reshape_2d(ctx, kn, kv_row, 1), kv_idx_t));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_v_cache, ggml_reshape_2d(ctx, vn, kv_row, 1), kv_idx_t));
        const int64_t T_use = kq_T_pad;
        const float kq_scale = hp.f_attention_scale == 0.0f ? 1.0f / sqrtf((float) n_hd) : hp.f_attention_scale;
        if (mtp_k_cache->type == GGML_TYPE_F16) {
            kq_mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T_use, 1);
            ggml_set_input(kq_mask_t);
            att_out = B.attn_fa(q, mtp_k_cache, mtp_v_cache, n_hd, n_hkv, n_h, T_use, woff, kq_mask_t, kq_scale);
        } else {
        kq_mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_use, n_rep);
        ggml_set_input(kq_mask_t);
        ggml_tensor * Kc = ggml_view_3d(ctx, mtp_k_cache, n_hd, n_hkv, T_use, ggml_element_size(mtp_k_cache) * n_hd,
                                        mtp_k_cache->nb[1], woff * mtp_k_cache->nb[1]);
        ggml_tensor * Vc = ggml_view_3d(ctx, mtp_v_cache, n_hd, n_hkv, T_use, ggml_element_size(mtp_v_cache) * n_hd,
                                        mtp_v_cache->nb[1], woff * mtp_v_cache->nb[1]);
        Kc = ggml_cont(ctx, ggml_permute(ctx, Kc, 0, 2, 1, 3));                                  // [n_hd, T, n_hkv]
        Vc = ggml_cont(ctx, ggml_permute(ctx, Vc, 0, 2, 1, 3));
        ggml_tensor * qh = ggml_reshape_3d(ctx, q, n_hd, n_rep, n_hkv);
        ggml_tensor * sc = ggml_mul_mat(ctx, Kc, qh);                                           // [T, n_rep, n_hkv]
        sc = ggml_soft_max_ext(ctx, sc, kq_mask_t,
                               hp.f_attention_scale == 0.0f ? 1.0f / sqrtf((float) n_hd) : hp.f_attention_scale, 0.0f);
        ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, Vc, 1, 0, 2, 3));
        att_out = ggml_cont_2d(ctx, ggml_mul_mat(ctx, Vt, sc), n_hd * n_h, 1);
        }
    } else {
        ggml_tensor * vg = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, vn, n_hd, 1, n_hkv), n_hd, n_rep, n_hkv, 1);
        att_out = ggml_cont_2d(ctx, vg, n_hd * n_h, 1);
    }
    ggml_tensor * att = ggml_mul_mat(ctx, L.wo, ggml_mul(ctx, att_out, ggml_sigmoid(ctx, gate)));  // [N, 1]
    ggml_tensor * R1  = B.hc_combine(R0, att, inj_a);

    // ---- MoE: mixer, softmax router, top-k, normalised weights, routed experts + gated shared expert
    ggml_tensor * inj_f = nullptr;
    ggml_tensor * xf = B.hc_mix(R1, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, &inj_f);  // [N, 1]
    ggml_tensor * probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp, xf));         // [n_expert, 1]
    ggml_tensor * sel   = ggml_argsort_top_k(ctx, probs, n_used);
    ggml_tensor * w2    = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, 1), sel), n_used, 1);
    w2 = ggml_div(ctx, w2, ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, 1e30f));
    if (wscale != 0.0f && wscale != 1.0f) w2 = ggml_scale(ctx, w2, wscale);
    ggml_tensor * cur3 = ggml_reshape_3d(ctx, xf, N, 1, 1);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur3, sel);
    ggml_tensor * gt   = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur3, sel);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gt, up);
    ggml_tensor * experts = ggml_mul(ctx, ggml_mul_mat_id(ctx, L.ffn_down_exps, act, sel), ggml_reshape_3d(ctx, w2, 1, n_used, 1));
    ggml_tensor * moe = ggml_view_2d(ctx, experts, N, 1, experts->nb[2], 0);
    for (int64_t i = 1; i < n_used; ++i) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, experts, N, 1, experts->nb[2], (size_t) i * experts->nb[1]));
    }
    {
        ggml_tensor * su = ggml_mul_mat(ctx, L.ffn_up_shexp, xf);
        ggml_tensor * sg = ggml_mul_mat(ctx, L.ffn_gate_shexp, xf);
        ggml_tensor * sh = ggml_mul_mat(ctx, L.ffn_down_shexp, ggml_swiglu_split(ctx, sg, su));
        sh = ggml_mul(ctx, sh, ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, xf)));
        moe = ggml_add(ctx, moe, sh);
    }
    ggml_tensor * R2 = B.hc_combine(R1, moe, inj_f);                                       // [N, HC, 1]
    ggml_tensor * R2f = ggml_reshape_2d(ctx, R2, HW, 1);
    ggml_set_output(R2f);

    // ---- the block's own final mixer, then the main model's head (shortlisted rows, as on the qwen35 path)
    ggml_tensor * sample = B.hc_mix(R2, L.nextn.hc_head_norm, L.nextn.hc_head_down, L.nextn.hc_head_up, nullptr, nullptr);
    ggml_tensor * head = model.output;
    {
        // ALPHA_MTP_DRAFT_HEAD_ROWS (default 65536): draft from the first n token ids only. BPE ids are
        // merge-ordered, so the low ids carry almost every accepted draft while the head GEMV gets cheaper
        static const int64_t hrows = getenv("ALPHA_MTP_DRAFT_HEAD_ROWS")
                                   ? (int64_t) atoll(getenv("ALPHA_MTP_DRAFT_HEAD_ROWS")) : 65536;
        if (hrows > 0 && hrows < head->ne[1]) head = ggml_view_2d(ctx, head, head->ne[0], hrows, head->nb[1], 0);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, head, sample);
    ggml_tensor * amax = ggml_argsort_top_k(ctx, logits, 2);
    ggml_set_output(amax);
    ggml_tensor * top2v = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, logits->ne[0]), amax);
    ggml_set_output(top2v);
    // ALPHA_MTP_CONF: the draft's softmax probability (top-1, top-2) over the head rows: a calibrated confidence
    // for gating the verify width on the cumulative product (serve-gpu-engine.cpp)
    ggml_tensor * conf2 = nullptr;
    {
        static const bool conf_on = [] { const char * e = getenv("ALPHA_MTP_CONF_TAU"); return e && atof(e) > 0.0; }();
        if (conf_on) {
            ggml_tensor * probs = ggml_soft_max(ctx, logits);
            conf2 = ggml_get_rows(ctx, ggml_reshape_2d(ctx, probs, 1, probs->ne[0]), amax);
            ggml_set_output(conf2);
        }
    }

    ggml_build_forward_expand(gf, amax);
    ggml_build_forward_expand(gf, top2v);
    if (conf2) ggml_build_forward_expand(gf, conf2);
    ggml_build_forward_expand(gf, R2f);
    d4c.conf = conf2;
    d4c.gf = gf; d4c.H = H; d4c.etok = etok ? etok : erow; d4c.rope = rope_pos_t; d4c.kv_idx = kv_idx_t; d4c.mask = kq_mask_t;
    d4c.amax = amax; d4c.top2v = top2v; d4c.R2f = R2f;
    d4c.T_use = kq_T_pad; d4c.woff = woff; d4c.kc = (const void *) mtp_k_cache;
    }   // !hit
    ggml_cgraph * gf = d4c.gf;
    ggml_tensor * H = d4c.H, * etok = d4c.etok, * rope_pos_t = d4c.rope, * kv_idx_t = d4c.kv_idx, * kq_mask_t = d4c.mask;
    ggml_tensor * amax = d4c.amax, * top2v = d4c.top2v, * R2f = d4c.R2f;
    const bool use_alt = !sd && (bool) sched_alt;
    if (use_alt) set_draft_active(true);
    const int64_t t_b1 = tdbg ? ggml_time_us() : 0;
    bool alloc_ok = true;
    if (sd) {
        if (!hit) {
            ggml_backend_sched_reset(sd);
            alloc_ok = ggml_backend_sched_alloc_graph(sd, gf);
            d4c.valid = alloc_ok;
            static int n_dump = 0;
            if (tdbg && n_dump++ < 2) {
                fprintf(stderr, "[mtp4-t] draft graph: %d nodes, %d splits, %d copies\n", ggml_graph_n_nodes(gf),
                        ggml_backend_sched_get_n_splits(sd), ggml_backend_sched_get_n_copies(sd));
                for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                    ggml_tensor * t = ggml_graph_node(gf, i);
                    ggml_backend_t bk = ggml_backend_sched_get_tensor_backend(sd, t);
                    if (bk && strncmp(ggml_backend_name(bk), "CUDA", 4) != 0)
                        fprintf(stderr, "[mtp4-t]   node %d %s %s on %s\n", i, ggml_op_desc(t), t->name, ggml_backend_name(bk));
                    for (int j = 0; j < GGML_MAX_SRC && t->src[j]; j++) {
                        ggml_tensor * s = t->src[j];
                        if (s->buffer && ggml_backend_buffer_is_host(s->buffer) && !(s->flags & GGML_TENSOR_FLAG_INPUT))
                            fprintf(stderr, "[mtp4-t]   host src %s (%s, %.2f MB) of %s %s\n", s->name,
                                    ggml_backend_buffer_name(s->buffer), ggml_nbytes(s) / 1048576.0, ggml_op_desc(t), t->name);
                    }
                }
            }
        }
    } else {
        ggml_backend_sched_reset(sched.get());
        if (gf_res_prev) gf_res_prev->reset();
        alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
    }
    llama_token best = LLAMA_TOKEN_NULL;
    if (alloc_ok) {
        const int64_t t_b2 = tdbg ? ggml_time_us() : 0;
        if (etok->type == GGML_TYPE_I32) {
            const int32_t ti = (int32_t) e_tok;
            ggml_backend_tensor_set(etok, &ti, 0, sizeof(int32_t));
        } else {
            static std::vector<float> row;
            row.resize((size_t) N);
            const ggml_tensor * te = model.tok_embd;
            ggml_get_type_traits(te->type)->to_float((const char *) te->data + (size_t) e_tok * te->nb[1], row.data(), N);
            ggml_backend_tensor_set(etok, row.data(), 0, (size_t) N * sizeof(float));
        }
        ggml_backend_tensor_set(H, mtp_hidden_host.data(), 0, (size_t) HW * sizeof(float));
        if (rope_pos_t && rope_pos_t->buffer) {
            std::vector<int32_t> rp((size_t) n_pos_pe, col_pos);
            ggml_backend_tensor_set(rope_pos_t, rp.data(), 0, ggml_nbytes(rope_pos_t));
        }
        if (kv_idx_t && kv_idx_t->buffer) ggml_backend_tensor_set(kv_idx_t, &kv_idx_v, 0, sizeof(int32_t));
        if (kq_mask_t && kq_mask_t->buffer && kq_mask_t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> mk((size_t) kq_T_pad);
            const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int64_t j = 0; j < kq_T_pad; ++j) mk[(size_t) j] = j < kq_T ? z : ninf;
            ggml_backend_tensor_set(kq_mask_t, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
        } else if (kq_mask_t && kq_mask_t->buffer) {
            std::vector<float> mk((size_t) kq_T_pad * n_rep, 0.0f);
            for (int64_t r = 0; r < n_rep; ++r)
                for (int64_t j = kq_T; j < kq_T_pad; ++j) mk[(size_t) r * kq_T_pad + j] = -INFINITY;
            ggml_backend_tensor_set(kq_mask_t, mk.data(), 0, mk.size() * sizeof(float));
        }
        const int64_t t_b3 = tdbg ? ggml_time_us() : 0;
        if (ggml_backend_sched_graph_compute(sd ? sd : sched.get(), gf) == GGML_STATUS_SUCCESS) {
            int32_t bi2[2] = { -1, -1 };
            ggml_backend_tensor_get(amax, bi2, 0, sizeof(bi2));
            if (tdbg) {
                static double a[5] = {0}; static int64_t n = 0;
                const int64_t t_b4 = ggml_time_us();
                a[0] += t_b1 - t_b0; a[1] += t_b2 - t_b1; a[2] += t_b3 - t_b2; a[3] += t_b4 - t_b3; n++;
                if ((n & 63) == 0) fprintf(stderr, "[mtp4-t] build %.2f alloc %.2f inputs %.2f compute+read %.2f ms/draft (n=%lld)\n",
                                           a[0] / n / 1e3, a[1] / n / 1e3, a[2] / n / 1e3, a[3] / n / 1e3, (long long) n);
            }
            ggml_backend_tensor_get(top2v, mtp_top2_v, 0, 2 * sizeof(float));
            if (d4c.conf) ggml_backend_tensor_get(d4c.conf, mtp_conf2, 0, 2 * sizeof(float));
            ggml_backend_tensor_get(R2f, mtp_hidden_host.data(), 0, (size_t) HW * sizeof(float));   // chain
            best = (llama_token) bi2[0];
            mtp_chain_step++;
            if (wrote_col) {
                if (mtp_kv_hi < 0) mtp_kv_lo = mtp_pos_base + mtp_chain_step - 1;
                mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, mtp_pos_base + mtp_chain_step - 1);
            }
            if (dbg) { static int n = 0; if (n < 8) { n++; fprintf(stderr, "[mtp4] draft e=%d -> %d (2nd %d) col_pos=%d attn_T=%lld\n",
                                                              (int) e_tok, (int) best, (int) bi2[1], col_pos, (long long) kq_T); } }
        } else {
            d4c.valid = false;
            if (dbg) fprintf(stderr, "[mtp4] draft compute failed\n");
        }
    } else {
        d4c.valid = false;
        if (dbg) fprintf(stderr, "[mtp4] draft alloc failed\n");
    }
    if (use_alt) set_draft_active(false);
    return best;
}

// ALPHA_MTP_FILL_INGRAPH (default 1): a prompt chunk fills the head's K/V inside its own graph: the columns need only
// the trunk's exported streams (row t-1, already on the device) and token t's embedding (the graph's own input rows),
// so only the last row's hidden comes back to the host instead of every row's hc*n_embd hidden.
// Same columns, same range bookkeeping (mtp_ingraph_commit mirrors mtp_kv_fill); anything unusual -- several
// sequences, a non-current sequence, gaps in the positions -- falls back to mtp_kv_fill.
bool llama_context::mtp_ingraph_append(const llama_ubatch & ub, llm_graph_result * res, ggml_cgraph * gf) {
    mtp_ig.active = false;
    mtp_ig.built  = false;
    static const bool on = [] { const char * e = getenv("ALPHA_MTP_FILL_INGRAPH"); return !e || atoi(e) != 0; }();
    static const int64_t min_tok = [] { const char * e = getenv("ALPHA_MTP_FILL_INGRAPH_MIN"); return e ? atoll(e) : (int64_t) 64; }();
    if (!on || model.arch != LLM_ARCH_QWEN4EXP || !model.has_mtp4 || !res || !res->t_mtp_hidden || !ub.token) return false;
    const int64_t n = ub.n_tokens;
    if (n < min_tok) return false;
    for (int64_t i = 0; i < n; ++i) {
        if (!ub.n_seq_id || ub.n_seq_id[i] != 1 || !ub.seq_id || ub.seq_id[i][0] != mtp_cur_seq) return false;
        if (i > 0 && ub.pos[i] != ub.pos[i - 1] + 1) return false;
    }
    const auto & hp = model.hparams;
    const auto & L  = model.mtp4;
    const int64_t N = hp.n_embd, HC = hp.hc_count, HW = N * HC;
    ggml_tensor * H = res->t_mtp_hidden;
    if (H->ne[0] != HW || H->ne[1] != n) return false;
    ggml_tensor * E = ggml_graph_get_tensor(gf, "model.input_embed");
    if (!E || E->type != GGML_TYPE_F32 || E->ne[0] != N || ggml_nelements(E) != N * n) return false;
    E = ggml_reshape_2d(res->get_ctx(), E, N, n);

    // the columns: (H_{t-1}, token_t) at position pos_t, written to slot pos_t - 1 (mtp_kv_fill's pairing)
    const llama_pos pos0 = ub.pos[0];
    const bool rewind   = pos0 <= mtp_pos_base;                       // mtp_seq_bookkeep drops the carried row
    const bool use_prev = !rewind && mtp_prev_pos >= 0 && (int64_t) mtp_prev_h.size() >= HW && mtp_prev_pos + 1 == pos0;
    const int32_t slot0 = use_prev ? pos0 - 1 : pos0;
    int64_t m = use_prev ? n : n - 1;
    if (slot0 < 0 || m < 1) return false;
    const int64_t n_hd = hp.n_embd_head_k(), n_hkv = hp.n_head_kv(hp.n_layer - 1);
    const int64_t kv_row = n_hd * n_hkv;
    if (!mtp_kv_ensure(kv_row, ggml_backend_buffer_get_type(L.wq->buffer))) return false;
    if (slot0 + m > mtp_kv_cols) m = mtp_kv_cols - slot0;
    if (m < 1) return false;
    if (!mtp_ingraph_build(res, gf, E, n, use_prev, slot0, m, mtp_ig)) return false;
    mtp_ig.active = true;
    mtp_ig.built  = true;
    return true;
}

// the fill nodes for n rows into the given plan (append: the real one; reserve: a throw-away one of the same shapes)
bool llama_context::mtp_ingraph_build(llm_graph_result * res, ggml_cgraph * gf, ggml_tensor * E, int64_t n, bool use_prev,
                                      int32_t slot0, int64_t m, mtp_ingraph_t & ig) {
    const auto & hp = model.hparams;
    const auto & L  = model.mtp4;
    const int64_t N = hp.n_embd, HC = hp.hc_count, HW = N * HC;
    ggml_tensor * H = res->t_mtp_hidden;
    const int64_t n_hd = hp.n_embd_head_k(), n_hkv = hp.n_head_kv(hp.n_layer - 1);
    const int64_t kv_row = n_hd * n_hkv;
    const int64_t cap_nodes = ggml_graph_size(gf) - ggml_graph_n_nodes(gf);
    const int64_t CH = 1024, nblk = (m + CH - 1) / CH;
    if (cap_nodes < nblk * 64) return false;
    (void) n;

    ggml_context * ctx = res->get_ctx();
    mtp4_builder B { ctx, model, L, N, HC, hp.f_norm_rms_eps };
    const bool do_rope = hp.rope_type != LLAMA_ROPE_TYPE_NONE;
    const int  n_pos_pe = (int) hp.n_pos_per_embd();
    ig.ix.clear(); ig.ps.clear(); ig.ix_h.clear(); ig.ps_h.clear();
    ig.hprev = nullptr;
    if (use_prev) {
        ig.hprev = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HW, 1);
        ggml_set_input(ig.hprev);
        ggml_set_name(ig.hprev, "mtp_ig_hprev");
    }
    // the token half of R0 (eh_proj of the normed embedding) once for the whole chunk: E lives in host memory, and a
    // per-block view of it would be copied to the GPU, with a full sync, by every block's split
    ggml_tensor * e2_all = ggml_mul_mat(ctx, L.nextn.eh_proj, ggml_mul(ctx, ggml_rms_norm(ctx, E, B.eps), L.nextn.enorm));   // [N, n]
    for (int64_t b0 = 0; b0 < m; b0 += CH) {
        const int64_t mb = std::min<int64_t>(CH, m - b0);
        const int64_t t0 = use_prev ? b0 : b0 + 1;   // token row of this block's first column; its hidden row is t0-1
        ggml_tensor * Hs;
        if (t0 == 0) {
            Hs = ig.hprev;
            if (mb > 1) Hs = ggml_concat(ctx, Hs, ggml_view_2d(ctx, H, HW, mb - 1, H->nb[1], 0), 1);
        } else {
            Hs = ggml_view_2d(ctx, H, HW, mb, H->nb[1], (size_t) (t0 - 1) * H->nb[1]);
        }
        ggml_tensor * e2 = ggml_view_2d(ctx, e2_all, N, mb, e2_all->nb[1], (size_t) t0 * e2_all->nb[1]);
        ggml_tensor * Ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb); ggml_set_input(Ix);
        ggml_tensor * Ps = nullptr;
        if (do_rope) { Ps = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb * n_pos_pe); ggml_set_input(Ps); }
        // R0 = h_proj(rms(H) * hnorm) per stream + e2 (mtp4_builder::input with the token half precomputed)
        ggml_tensor * hn = ggml_mul(ctx, ggml_rms_norm(ctx, Hs, B.eps), L.nextn.hnorm);
        ggml_tensor * h2 = ggml_mul_mat(ctx, L.nextn.h_proj, ggml_reshape_2d(ctx, hn, N, HC * mb));
        ggml_tensor * R0 = ggml_add(ctx, ggml_reshape_3d(ctx, h2, N, HC, mb), ggml_reshape_3d(ctx, ggml_cont(ctx, e2), N, 1, mb));
        ggml_tensor * mixed = B.hc_mix(R0, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, nullptr, nullptr);
        ggml_tensor * kn = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, mixed), n_hd, n_hkv, mb);
        kn = ggml_mul(ctx, ggml_rms_norm(ctx, kn, B.eps), L.attn_k_norm);
        if (do_rope) kn = B.rope(kn, Ps, cparams);
        ggml_tensor * vn = ggml_mul_mat(ctx, L.wv, mixed);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_k_cache, ggml_reshape_2d(ctx, kn, kv_row, mb), Ix));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_v_cache, ggml_reshape_2d(ctx, vn, kv_row, mb), Ix));
        std::vector<int32_t> ix((size_t) mb), ps;
        for (int64_t i = 0; i < mb; ++i) ix[i] = (int32_t) (mtp_kv_base() + slot0 + b0 + i);
        if (do_rope) {
            ps.resize((size_t) mb * n_pos_pe);
            // column c sits at position slot0 + 1 + c (pos_t of its token)
            for (int j = 0; j < n_pos_pe; ++j)
                for (int64_t i = 0; i < mb; ++i) ps[(size_t) j * mb + i] = (int32_t) (slot0 + 1 + b0 + i);
        }
        ig.ix.push_back(Ix); ig.ps.push_back(Ps);
        ig.ix_h.push_back(std::move(ix)); ig.ps_h.push_back(std::move(ps));
    }
    ig.slot0 = slot0;
    ig.m     = m;
    return true;
}

// ALPHA_MTP_FILL_INGRAPH: the reserve of a prompt-sized graph carries the same fill nodes (worst case: every row), so
// the first real prompt graph fits the reserved buffer instead of re-allocating
void llama_context::mtp_ingraph_reserve(llm_graph_result * res, ggml_cgraph * gf, int64_t n) {
    static const bool on = [] { const char * e = getenv("ALPHA_MTP_FILL_INGRAPH"); return !e || atoi(e) != 0; }();
    static const int64_t min_tok = [] { const char * e = getenv("ALPHA_MTP_FILL_INGRAPH_MIN"); return e ? atoll(e) : (int64_t) 64; }();
    static const bool dbg = getenv("ALPHA_MTP_IG_DBG") != nullptr;
    if (dbg) fprintf(stderr, "[mtp-ig-res] n=%lld hidden=%p ne1=%lld\n", (long long) n, (void *) (res ? res->t_mtp_hidden : nullptr),
                     (long long) (res && res->t_mtp_hidden ? res->t_mtp_hidden->ne[1] : -1));
    if (!on || model.arch != LLM_ARCH_QWEN4EXP || !model.has_mtp4 || !res || !res->t_mtp_hidden || n < min_tok) return;
    const auto & hp = model.hparams;
    const int64_t N = hp.n_embd;
    if (res->t_mtp_hidden->ne[1] != n) return;
    ggml_tensor * E = ggml_graph_get_tensor(gf, "model.input_embed");
    if (dbg) fprintf(stderr, "[mtp-ig-res] E=%p type=%d ne0=%lld nel=%lld\n", (void *) E, E ? (int) E->type : -1,
                     (long long) (E ? E->ne[0] : -1), (long long) (E ? ggml_nelements(E) : -1));
    if (!E || E->type != GGML_TYPE_F32 || E->ne[0] != N || ggml_nelements(E) != N * n) return;
    E = ggml_reshape_2d(res->get_ctx(), E, N, n);
    const int64_t kv_row = hp.n_embd_head_k() * hp.n_head_kv(hp.n_layer - 1);
    if (!mtp_kv_ensure(kv_row, ggml_backend_buffer_get_type(model.mtp4.wq->buffer))) return;
    const int64_t m = std::min<int64_t>(n, mtp_kv_cols);
    mtp_ingraph_t tmp;
    (void) mtp_ingraph_build(res, gf, E, n, true, 0, m, tmp);
}

void llama_context::mtp_ingraph_set_inputs() {
    if (!mtp_ig.active) return;
    for (size_t b = 0; b < mtp_ig.ix.size(); ++b) {
        ggml_backend_tensor_set(mtp_ig.ix[b], mtp_ig.ix_h[b].data(), 0, mtp_ig.ix_h[b].size() * sizeof(int32_t));
        if (mtp_ig.ps[b]) ggml_backend_tensor_set(mtp_ig.ps[b], mtp_ig.ps_h[b].data(), 0, mtp_ig.ps_h[b].size() * sizeof(int32_t));
    }
    if (mtp_ig.hprev) ggml_backend_tensor_set(mtp_ig.hprev, mtp_prev_h.data(), 0, (size_t) mtp_hw() * sizeof(float));
}
