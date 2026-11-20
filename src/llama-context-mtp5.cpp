// glm5-next (GLM-5.3-Flash) MTP draft block: the hand-built draft step and head-KV fill. Built with ALPHA_MTP.
//
// The NextN block is the model's last layer (index n_layer - 1): one plain pre-norm block -- no hyper-connection
// streams -- with the trunk's nope MLA attention and DeepSeek-style MoE, plus the nextn input projection and
// head norm. The maths follows upstream llama.cpp's llama_model_glm5_next::graph_mtp (edec462cc):
//
//   x     = eh_proj(concat(rms(embed(tok)) * enorm, rms(H) * hnorm))        H = trunk hidden after output_norm
//   x1    = x  + wo(wv_b(softmax(q_absorbed . latent) latent))              latent = rms(wkv_a(rms(x)*attn_norm))*kv_a_norm
//   x2    = x1 + moe(rms(x1)*ffn_norm) + shexp(rms(x1)*ffn_norm)            sigmoid router, clamped SwiGLU
//   h     = rms(x2) * shared_head_norm                                      also the hidden a chained draft starts from
//   draft = argmax(output(h))
//
// The head keeps its own latent cache (llama_context::mtp_k_cache, one n_lora_kv column per position; the V
// cache is a one-row dummy because MLA reads keys and values from the same latent). The column for position p
// depends only on (H_{p-1}, token_p), so a whole decode's columns are filled in one pass (mtp_kv_fill_glm5next).
// The keys carry no position (nope MLA), so nothing here depends on the absolute position of a column.
//
// Difference from the reference: the block's own k-pool indexer is not evaluated. The head attends densely over
// every column it holds, which is the reference selection while the context fits the top-k budget
// (attention.indexer.top_k tokens, 2048 for GLM-5.3-Flash) and a superset of it beyond that. Drafts are verified
// by the trunk, so this can only cost acceptance on long contexts, never output.

#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct mtp5_builder {
    ggml_context *      ctx;
    const llama_model & model;
    const llama_layer & L;
    float               eps;

    ggml_tensor * norm(ggml_tensor * x, ggml_tensor * w) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
    }

    // x [N, T] from the carried hidden Hs [N, T] and the embedded tokens (E_in when the caller dequantised the
    // rows on the host, otherwise the token ids Tk)
    ggml_tensor * input(ggml_tensor * Hs, ggml_tensor * Tk, ggml_tensor * E_in) const {
        ggml_tensor * emb_w = L.nextn.embed_tokens ? L.nextn.embed_tokens : model.tok_embd;
        ggml_tensor * E  = E_in ? E_in : ggml_get_rows(ctx, emb_w, Tk);
        ggml_tensor * en = norm(E,  L.nextn.enorm);
        ggml_tensor * hn = norm(Hs, L.nextn.hnorm);
        return ggml_mul_mat(ctx, L.nextn.eh_proj, ggml_concat(ctx, en, hn, 0));
    }

    // the cached latent of each row: [n_lora_kv, T]
    ggml_tensor * latent(ggml_tensor * attn_in) const {
        return norm(ggml_mul_mat(ctx, L.wkv_a_mqa, attn_in), L.attn_kv_a_norm);
    }
};

bool mtp5_embd_host(const llama_model & model, const llama_layer & L) {
    const ggml_tensor * te = L.nextn.embed_tokens ? L.nextn.embed_tokens : model.tok_embd;
    return te->buffer && ggml_backend_buffer_is_host(te->buffer) && ggml_get_type_traits(te->type)->to_float != nullptr;
}

void mtp5_embd_row(const llama_model & model, const llama_layer & L, llama_token tok, float * dst, int64_t N) {
    const ggml_tensor * te = L.nextn.embed_tokens ? L.nextn.embed_tokens : model.tok_embd;
    ggml_get_type_traits(te->type)->to_float((const char *) te->data + (size_t) tok * te->nb[1], dst, N);
}

bool mtp5_block_loaded(const llama_layer & L) {
    return L.nextn.eh_proj && L.nextn.enorm && L.nextn.hnorm && L.attn_norm && L.ffn_norm &&
           L.wq_a && L.wq_b && L.wkv_a_mqa && L.wk_b && L.wv_b && L.wo && L.attn_q_a_norm && L.attn_kv_a_norm &&
           L.ffn_gate_inp && L.ffn_gate_exps && L.ffn_up_exps && L.ffn_down_exps &&
           L.ffn_gate_shexp && L.ffn_up_shexp && L.ffn_down_shexp;
}

} // namespace

bool llama_context::mtp_kv_fill_glm5next(const std::vector<const float *> & hsrc, const std::vector<llama_token> & ctok,
                                         int32_t slot0, int64_t m) {
    const auto & hp = model.hparams;
    const auto & L  = model.layers[hp.n_layer - 1];
    const int64_t N = hp.n_embd, R = hp.n_lora_kv;
    const bool use_alt = (bool) sched_alt;
    static const int64_t CH = [] { const char * e = getenv("ALPHA_MTP_FILL_CH"); return e ? std::max<int64_t>(1, atoll(e)) : (int64_t) 1024; }();
    const bool embd_host = mtp5_embd_host(model, L);
    for (int64_t base = 0; base < m; base += CH) {
        const int64_t mb = std::min<int64_t>(CH, m - base);
        ggml_init_params ip = { ggml_graph_overhead() + ggml_tensor_overhead() * 64, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return false;
        mtp5_builder B { ctx, model, L, hp.f_norm_rms_eps };
        ggml_tensor * Hs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, mb); ggml_set_input(Hs);
        ggml_tensor * Tk = nullptr, * Er = nullptr;
        if (embd_host) { Er = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, mb); ggml_set_input(Er); }
        else           { Tk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb);    ggml_set_input(Tk); }
        ggml_tensor * Ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb); ggml_set_input(Ix);
        ggml_tensor * x  = B.input(Hs, Tk, Er);
        ggml_tensor * kv = B.latent(B.norm(x, L.attn_norm));                                    // [R, mb]
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, mtp_k_cache, ggml_reshape_2d(ctx, kv, R, mb), Ix));
        if (use_alt) set_draft_active(true);
        ggml_backend_sched_reset(sched.get());
        if (gf_res_prev) gf_res_prev->reset();
        bool ok = false;
        if (ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            const size_t row_b = (size_t) N * sizeof(float);
            for (int64_t i = 0; i < mb; ) {
                int64_t j = i + 1;
                while (j < mb && hsrc[base + j] == hsrc[base + j - 1] + N) ++j;
                ggml_backend_tensor_set(Hs, hsrc[base + i], (size_t) i * row_b, (size_t) (j - i) * row_b);
                i = j;
            }
            if (Tk) {
                std::vector<int32_t> tk((size_t) mb);
                for (int64_t i = 0; i < mb; ++i) tk[i] = (int32_t) ctok[base + i];
                ggml_backend_tensor_set(Tk, tk.data(), 0, (size_t) mb * sizeof(int32_t));
            } else {
                std::vector<float> er((size_t) N * mb);
                for (int64_t i = 0; i < mb; ++i) mtp5_embd_row(model, L, ctok[base + i], er.data() + (size_t) i * N, N);
                ggml_backend_tensor_set(Er, er.data(), 0, er.size() * sizeof(float));
            }
            std::vector<int32_t> ix((size_t) mb);
            for (int64_t i = 0; i < mb; ++i) ix[i] = (int32_t) (mtp_kv_base() + slot0 + base + i);
            ggml_backend_tensor_set(Ix, ix.data(), 0, (size_t) mb * sizeof(int32_t));
            ok = ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS;
        }
        if (use_alt) set_draft_active(false);
        ggml_free(ctx);
        if (!ok) return false;
        mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, (int32_t) (slot0 + base + mb - 1));
    }
    return true;
}

llama_token llama_context::mtp_draft_glm5next(llama_token e_tok) {
    static const bool dbg = getenv("ALPHA_MTP_DBG") != nullptr;
    const auto & hp = model.hparams;
    const int il = (int) hp.n_layer - 1;
    const auto & L = model.layers[il];
    const int64_t N = hp.n_embd;
    if (hp.nextn_predict_layers != 1 || !mtp5_block_loaded(L) || (int64_t) mtp_hidden_host.size() < N) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[mtp] glm5-next: %s -- not drafting\n",
                    hp.nextn_predict_layers != 1 ? "the model does not carry exactly one NextN block" :
                    !mtp5_block_loaded(L) ? "the NextN block is not loaded (start with ALPHA_MTP=1)" : "no exported hidden yet");
        }
        return LLAMA_TOKEN_NULL;
    }
    const int64_t n_expert = hp.n_expert, n_used = hp.n_expert_used;
    const int64_t n_h = hp.n_head(il), dk = hp.n_embd_head_k_mla(), dv = hp.n_embd_head_v_mla(), R = hp.n_lora_kv;
    if (!mtp_kv_ensure(R, ggml_backend_buffer_get_type(L.wkv_a_mqa->buffer))) return LLAMA_TOKEN_NULL;

    // ---- per-call scalars; only (T_use, woff, cache identity) shape the graph
    const int32_t col_pos = mtp_pos_base >= 0 ? mtp_pos_base + 1 + mtp_chain_step : -1;
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
        kq_T_pad = std::min<int64_t>(((kq_T + 63) / 64) * 64, avail);
        if (kq_T_pad < kq_T) kq_T_pad = kq_T;
        static bool noted = false;
        if (!noted && kq_T > (int64_t) hp.indexer_top_k) {
            noted = true;
            fprintf(stderr, "[mtp] glm5-next: the head now attends over %lld positions, past the indexer's top-k budget (%u): "
                            "it stays dense where the reference block selects pools (see llama-context-mtp5.cpp)\n",
                    (long long) kq_T, hp.indexer_top_k);
        }
    }
    const bool wrote_col = pos >= 0;
    const bool embd_host = mtp5_embd_host(model, L);

    // the draft graph stays built and allocated on its own scheduler; a draft whose shape key matches only uploads
    // its inputs (ALPHA_MTP_D4_CACHE=0 rebuilds it every draft)
    static const bool use_cache = [] { const char * e = getenv("ALPHA_MTP_D4_CACHE"); return !e || atoi(e) != 0; }();
    if (!sched_d4) {
        sched_d4.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                                              graph_max_nodes(1), false, cparams.op_offload));
    }
    ggml_backend_sched_t sd = sched_d4.get();
    const bool hit = use_cache && d4c.valid && d4c.T_use == kq_T_pad && d4c.woff == woff && d4c.kc == (const void *) mtp_k_cache;
    if (!hit) {
        d4c.valid = false;
        ggml_init_params ip = { ggml_graph_overhead() + ggml_tensor_overhead() * 320, nullptr, true };
        if (!mtp_draft_ctx) mtp_draft_ctx = ggml_init(ip); else ggml_reset(mtp_draft_ctx);
        ggml_context * ctx = mtp_draft_ctx;
        if (!ctx) return LLAMA_TOKEN_NULL;
        mtp5_builder B { ctx, model, L, hp.f_norm_rms_eps };
        ggml_cgraph * gf = ggml_new_graph(ctx);

        ggml_tensor * H    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, 1); ggml_set_input(H);
        ggml_tensor * etok = nullptr, * erow = nullptr;
        if (embd_host) { erow = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, 1); ggml_set_input(erow); }
        else           { etok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);    ggml_set_input(etok); }
        ggml_tensor * x = B.input(H, etok, erow);                                               // [N, 1]

        // ---- nope MLA: low-rank query with wk_b absorbed, latent appended to the head cache, dense attention
        ggml_tensor * xa = B.norm(x, L.attn_norm);
        ggml_tensor * qr = B.norm(ggml_mul_mat(ctx, L.wq_a, xa), L.attn_q_a_norm);              // [n_lora_q, 1]
        ggml_tensor * q  = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wq_b, qr), dk, 1, n_h);     // head h at d + dk*h
        ggml_tensor * qa = ggml_mul_mat(ctx, L.wk_b, q);                                        // [R, 1, n_h]
        ggml_tensor * kv = B.latent(xa);                                                        // [R, 1]
        ggml_tensor * kv_idx_t = nullptr, * kq_mask_t = nullptr;
        ggml_tensor * lat = nullptr;                                                            // [R, 1, n_h]
        if (pos >= 0) {
            kv_idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            ggml_set_input(kv_idx_t);
            // the window is read through the write, so the column is in place before the attention reads it
            ggml_tensor * kc = ggml_set_rows(ctx, mtp_k_cache, ggml_reshape_2d(ctx, kv, R, 1), kv_idx_t);
            ggml_build_forward_expand(gf, kc);
            const int64_t T_use = kq_T_pad;
            ggml_tensor * Kc = ggml_view_2d(ctx, kc, R, T_use, mtp_k_cache->nb[1], woff * mtp_k_cache->nb[1]);
            if (Kc->type != GGML_TYPE_F32) Kc = ggml_cast(ctx, Kc, GGML_TYPE_F32);
            kq_mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_use, 1);
            ggml_set_input(kq_mask_t);
            ggml_tensor * sc = ggml_mul_mat(ctx, Kc, qa);                                       // [T, 1, n_h]
            sc = ggml_soft_max_ext(ctx, sc, kq_mask_t, 1.0f / sqrtf((float) dk), 0.0f);
            ggml_tensor * Kt = ggml_cont(ctx, ggml_transpose(ctx, Kc));                         // [T, R]
            lat = ggml_mul_mat(ctx, Kt, sc);                                                    // [R, 1, n_h]
        } else {
            // no usable column: the token sees only itself, every head's attention output is its own latent
            lat = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, kv, R, 1, 1), R, 1, n_h, 1);
        }
        ggml_tensor * vo  = ggml_mul_mat(ctx, L.wv_b, lat);                                     // [dv, 1, n_h]
        ggml_tensor * att = ggml_mul_mat(ctx, L.wo, ggml_reshape_2d(ctx, ggml_cont(ctx, vo), dv * n_h, 1));
        ggml_tensor * x1  = ggml_add(ctx, x, att);

        // ---- MoE: sigmoid router (biased selection, unbiased weights), clamped SwiGLU experts + shared expert
        ggml_tensor * xf     = B.norm(x1, L.ffn_norm);
        ggml_tensor * probs  = ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp, xf));        // [n_expert, 1]
        ggml_tensor * selp   = L.ffn_exp_probs_b ? ggml_add(ctx, probs, L.ffn_exp_probs_b) : probs;
        ggml_tensor * sel    = ggml_argsort_top_k(ctx, selp, n_used);
        ggml_tensor * w2     = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, 1), sel), n_used, 1);
        if (hp.expert_weights_norm) w2 = ggml_div(ctx, w2, ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, INFINITY));
        if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f) w2 = ggml_scale(ctx, w2, hp.expert_weights_scale);
        ggml_tensor * cur3 = ggml_reshape_3d(ctx, xf, N, 1, 1);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur3, sel);
        ggml_tensor * gt   = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur3, sel);
        if (const float lim = hp.swiglu_clamp_exp[il]; lim > 1e-6f) {
            gt = ggml_clamp(ctx, gt, -INFINITY, lim);
            up = ggml_clamp(ctx, up, -lim, lim);
        }
        ggml_tensor * experts = ggml_mul(ctx, ggml_mul_mat_id(ctx, L.ffn_down_exps, ggml_swiglu_split(ctx, gt, up), sel),
                                         ggml_reshape_3d(ctx, w2, 1, n_used, 1));
        ggml_tensor * moe = ggml_view_2d(ctx, experts, N, 1, experts->nb[2], 0);
        for (int64_t i = 1; i < n_used; ++i) {
            moe = ggml_add(ctx, moe, ggml_view_2d(ctx, experts, N, 1, experts->nb[2], (size_t) i * experts->nb[1]));
        }
        {
            ggml_tensor * su = ggml_mul_mat(ctx, L.ffn_up_shexp, xf);
            ggml_tensor * sg = ggml_mul_mat(ctx, L.ffn_gate_shexp, xf);
            if (const float lim = hp.swiglu_clamp_shexp[il]; lim > 1e-6f) {
                sg = ggml_clamp(ctx, sg, -INFINITY, lim);
                su = ggml_clamp(ctx, su, -lim, lim);
            }
            moe = ggml_add(ctx, moe, ggml_mul_mat(ctx, L.ffn_down_shexp, ggml_swiglu_split(ctx, sg, su)));
        }
        ggml_tensor * x2 = ggml_add(ctx, x1, moe);

        // ---- head norm (the chained hidden), then the model's output matrix
        ggml_tensor * hn = B.norm(x2, L.nextn.shared_head_norm ? L.nextn.shared_head_norm : model.output_norm);   // [N, 1]
        ggml_set_output(hn);
        ggml_tensor * head = L.nextn.shared_head_head ? L.nextn.shared_head_head : model.output;
        {
            // ALPHA_MTP_DRAFT_HEAD_ROWS (default 0 = the whole vocabulary on this architecture): draft from the
            // first n token ids only
            static const int64_t hrows = getenv("ALPHA_MTP_DRAFT_HEAD_ROWS") ? (int64_t) atoll(getenv("ALPHA_MTP_DRAFT_HEAD_ROWS")) : 0;
            if (hrows > 0 && hrows < head->ne[1]) head = ggml_view_2d(ctx, head, head->ne[0], hrows, head->nb[1], 0);
        }
        ggml_tensor * logits = ggml_mul_mat(ctx, head, hn);
        ggml_tensor * amax = ggml_argsort_top_k(ctx, logits, 2);
        ggml_set_output(amax);
        ggml_tensor * top2v = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, logits->ne[0]), amax);
        ggml_set_output(top2v);
        ggml_tensor * conf2 = nullptr;
        {
            static const bool conf_on = [] { const char * e = getenv("ALPHA_MTP_CONF_TAU"); return e && atof(e) > 0.0; }();
            if (conf_on) {
                ggml_tensor * pr = ggml_soft_max(ctx, logits);
                conf2 = ggml_get_rows(ctx, ggml_reshape_2d(ctx, pr, 1, pr->ne[0]), amax);
                ggml_set_output(conf2);
            }
        }
        ggml_build_forward_expand(gf, amax);
        ggml_build_forward_expand(gf, top2v);
        if (conf2) ggml_build_forward_expand(gf, conf2);
        ggml_build_forward_expand(gf, hn);
        d4c.conf = conf2;
        d4c.gf = gf; d4c.H = H; d4c.etok = etok ? etok : erow; d4c.rope = nullptr; d4c.kv_idx = kv_idx_t; d4c.mask = kq_mask_t;
        d4c.amax = amax; d4c.top2v = top2v; d4c.R2f = hn;
        d4c.T_use = kq_T_pad; d4c.woff = woff; d4c.kc = (const void *) mtp_k_cache;

        ggml_backend_sched_reset(sd);
        d4c.valid = ggml_backend_sched_alloc_graph(sd, gf);
    }
    llama_token best = LLAMA_TOKEN_NULL;
    if (!d4c.valid) {
        if (dbg) fprintf(stderr, "[mtp5] draft alloc failed\n");
        return best;
    }
    ggml_cgraph * gf = d4c.gf;
    ggml_tensor * etok = d4c.etok;
    if (etok->type == GGML_TYPE_I32) {
        const int32_t ti = (int32_t) e_tok;
        ggml_backend_tensor_set(etok, &ti, 0, sizeof(int32_t));
    } else {
        std::vector<float> row((size_t) N);
        mtp5_embd_row(model, L, e_tok, row.data(), N);
        ggml_backend_tensor_set(etok, row.data(), 0, (size_t) N * sizeof(float));
    }
    ggml_backend_tensor_set(d4c.H, mtp_hidden_host.data(), 0, (size_t) N * sizeof(float));
    if (d4c.kv_idx && d4c.kv_idx->buffer) ggml_backend_tensor_set(d4c.kv_idx, &kv_idx_v, 0, sizeof(int32_t));
    if (d4c.mask && d4c.mask->buffer) {
        std::vector<float> mk((size_t) kq_T_pad, 0.0f);
        for (int64_t j = kq_T; j < kq_T_pad; ++j) mk[(size_t) j] = -INFINITY;
        ggml_backend_tensor_set(d4c.mask, mk.data(), 0, mk.size() * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(sd, gf) == GGML_STATUS_SUCCESS) {
        int32_t bi2[2] = { -1, -1 };
        ggml_backend_tensor_get(d4c.amax, bi2, 0, sizeof(bi2));
        ggml_backend_tensor_get(d4c.top2v, mtp_top2_v, 0, 2 * sizeof(float));
        if (d4c.conf) ggml_backend_tensor_get(d4c.conf, mtp_conf2, 0, 2 * sizeof(float));
        ggml_backend_tensor_get(d4c.R2f, mtp_hidden_host.data(), 0, (size_t) N * sizeof(float));   // chain
        best = (llama_token) bi2[0];
        mtp_chain_step++;
        if (wrote_col) {
            if (mtp_kv_hi < 0) mtp_kv_lo = mtp_pos_base + mtp_chain_step - 1;
            mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, mtp_pos_base + mtp_chain_step - 1);
        }
        if (dbg) {
            static int n = 0;
            if (n < 64) {
                n++;
                fprintf(stderr, "[mtp5] draft e=%d -> %d (2nd %d, logits %.3f %.3f) col_pos=%d attn_T=%lld step=%d\n", (int) e_tok,
                        (int) best, (int) bi2[1], (double) mtp_top2_v[0], (double) mtp_top2_v[1], col_pos, (long long) kq_T, mtp_chain_step);
            }
        }
    } else {
        d4c.valid = false;
        if (dbg) fprintf(stderr, "[mtp5] draft compute failed\n");
    }
    return best;
}
