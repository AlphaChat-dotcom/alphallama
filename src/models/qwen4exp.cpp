#include "models.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>

#include <vector>

#include "llama-impl.h"

#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-kv-cache.h"
#include "alpha_gdn_record.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// Qwen3.8-Flash-Next (arch qwen4exp). Ported from upstream llama.cpp 6c84c7d5d (PR #27742),
// re-expressed against this fork's llm_build_* graph API -- upstream moved to a per-arch
// llama_model_* class with build_arch_graph(), which this tree does not have.
//
// Shape of the model, and how it differs from qwen35moe (which shares the Gated DeltaNet half):
//
//   * HyperConnections. The residual stream is hc_count(4) parallel copies of n_embd, so the
//     wide state is [n_embd, hc, n_tokens]. There is no attn_norm, ffn_norm or output_norm
//     anywhere in the file: hc_*_norm replaces all three. Each block reads the streams through
//     a mixer (build_hc_mix) and writes back through a scatter (build_hc_combine).
//   * 12 full-attention layers (every 4th) and 36 Gated DeltaNet layers.
//   * MoE with a gated shared expert -- ffn_gate_inp_shexp is a per-token scalar router, not
//     the usual always-on shared expert.
//   * A sigmoid output gate on the GDN norm, where Qwen3.5 uses silu. This is the one
//     numerical difference in the linear path and it is why the Alpha fused projection path
//     (which folds silu into ggml_swiglu_split) is deliberately not used here.
//
// QSA sparse attention is applied, as upstream does, as a mask over dense attention rather than a sparse
// kernel, so when n_kv <= indexer_top_k (2048) every key is selected and the result equals dense attention.

llm_build_qwen4exp::llm_build_qwen4exp(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.hc_count;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());
    GGML_ASSERT(hc > 1 && "qwen4exp: hyper_connection.count must be > 1");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx (see llama-model.cpp), so this downcast is
    // safe; get_idx() is still null when the GGUF carries no indexer tensors.
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        // the attention cells see every ubatch regardless of the layer types, so the token
        // history for the n-gram hash comes from there
        ple_emb = build_inp_ple(inp->get_attn()->mctx);
        // keep ple_emb and build_inp_embd in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // with an MTP block loaded (ALPHA_EXTRA_GGUF sidecar) and ALPHA_MTP on, the last layer runs on
    // every row so the residual streams can be exported for the head's K/V fill
    static const bool mtp_env = [] { const char * e = getenv("ALPHA_MTP"); return e && atoi(e) != 0; }();
    const bool mtp_defer = mtp_env && model.has_mtp4;

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && !mtp_defer) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // MTP: export the residual streams of every row (the MTP block's hidden input and the source of
    // its K/V fill), then narrow to the output rows for the head
    if (mtp_defer) {
        ggml_tensor * mtp_h = ggml_reshape_2d(ctx0, res_hc, n_embd * hc, res_hc->ne[2]);
        cb(mtp_h, "mtp_hidden", -1);
        ggml_set_output(mtp_h);
        res->t_mtp_hidden = mtp_h;
        ggml_build_forward_expand(gf, mtp_h);
        if (inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // MTP: the verify reads one token per row from an in-graph argmax instead of pulling and scanning
    // every row's full-vocab logits on the host
    if (mtp_defer) {
        ggml_tensor * am = ggml_argmax(ctx0, cur);
        cb(am, "mtp_argmax", -1);
        ggml_set_output(am);
        res->t_mtp_argmax = am;
        ggml_build_forward_expand(gf, am);
    }
}

// ALPHA_HC_FUSE (default 7): bit 1 = scatter as one op (ggml_hc_combine), bit 2 = mixer tail as one op
// (ggml_hc_mix_tail) -- both bit-identical to the op-by-op chain; bit 4 = the whole read as ggml_hc_mix_fused for
// <= 8 tokens (same math, matches the chain to ~1e-7 at T=1). 0 restores the op-by-op graph.
static int hc_fuse_mask() {
    static const int v = [] { const char * e = getenv("ALPHA_HC_FUSE"); return e ? atoi(e) : 7; }();
    return v;
}
static bool hc_fuse_combine() { return (hc_fuse_mask() & 1) != 0; }
static bool hc_fuse_mix()     { return (hc_fuse_mask() & 2) != 0; }
// bit 4: the whole read (norm, low-rank gate, gated mean, inject) as one op for <= 8 tokens (decode / MTP verify)
static bool hc_fuse_read()    { return (hc_fuse_mask() & 4) != 0; }

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` receives the [hc, T] scatter weights.

ggml_tensor * llm_build_qwen4exp::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.hc_count;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // BF16 as in the file, or Q8_0 when the loader converted them (ALPHA_HC_Q8)
    const bool hc_w_ok = w_down->type == w_up->type && (w_down->type == GGML_TYPE_BF16 || w_down->type == GGML_TYPE_Q8_0);
    if (hc_fuse_read() && nt <= 8 && hc == 4 && hc_w_ok) {
        ggml_tensor * fz = ggml_hc_mix_fused(ctx0, x, w_norm, w_down, w_up, inject ? w_inject : nullptr, hparams.f_norm_rms_eps);
        ggml_tensor * mixed = ggml_view_2d(ctx0, fz, n_embd, nt, n_embd*sizeof(float), 0);
        cb(mixed, "hc_mixed", il);
        if (inject) {
            *inject = ggml_view_2d(ctx0, fz, hc, nt, hc*sizeof(float), (size_t) n_embd*nt*sizeof(float));
            cb(*inject, "hc_inject", il);
        }
        return mixed;
    }

    // ALPHA_HC_PF_NI (default 1), prompt-sized batches: the per-stream norm, the gamma and the inject projection as one
    // pass (ggml_hc_norm_inject) instead of rms_norm + mul + an f32->bf16 copy of xn + a cuBLAS BF16 GEMM for inject
    static const bool pf_ni = [] { const char * e = getenv("ALPHA_HC_PF_NI"); return !e || atoi(e) != 0; }();
    ggml_tensor * ni_inject = nullptr;
    ggml_tensor * xn = nullptr;
    if (pf_ni && hc == 4 && w_norm->type == GGML_TYPE_F32 &&
        (!w_inject || w_inject->type == GGML_TYPE_BF16 || w_inject->type == GGML_TYPE_F32)) {
        ggml_tensor * ni = ggml_hc_norm_inject(ctx0, x, w_norm, inject ? w_inject : nullptr, hparams.f_norm_rms_eps);
        xn = ggml_view_2d(ctx0, ni, hc_dim, nt, hc_dim*sizeof(float), 0);
        ni_inject = ggml_view_2d(ctx0, ni, hc, nt, hc*sizeof(float), (size_t) hc_dim*nt*sizeof(float));
    } else {
        // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma.
        // the converter folded each gamma to (1 + w)
        xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
        xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
        xn = ggml_mul(ctx0, xn, w_norm);
    }
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate_pre = build_lora_mm(w_up, lo);

    ggml_tensor * mixed = nullptr;
    if (hc_fuse_mix()) {
        // one kernel for sigmoid, gate, and the in-order stream mean; bit-identical to the chain below
        mixed = ggml_hc_mix_tail(ctx0, xn, gate_pre, (int) hc, 1.0f / (float) hc);
        cb(mixed, "hc_mixed", il);
        if (inject) {
            *inject = ni_inject ? ni_inject : build_lora_mm(w_inject, xn);
            cb(*inject, "hc_inject", il);
        }
        return mixed;
    }

    ggml_tensor * gate = ggml_sigmoid(ctx0, gate_pre);
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean
    mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    mixed = ggml_cont(ctx0, mixed);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llm_build_qwen4exp::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.hc_count;
    const int64_t nt = residual->ne[2];

    if (hc_fuse_combine()) {
        ggml_tensor * cur = ggml_hc_combine(ctx0, residual, block_out, inject, 1.0f / (float) hc, 2.0f);
        cb(cur, "hc_combine", il);
        return cur;
    }

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen4exp::build_qkvz(
        ggml_tensor * input,
        int           il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen4exp::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

ggml_tensor * llm_build_qwen4exp::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t il_n_head    = hparams.n_head(il);
    const int64_t il_n_head_kv = hparams.n_head_kv(il);

    // The indexer reads the same block input as q/k/v. No cache or no ratio on this layer means
    // plain dense attention.
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.compress_ratios[il] > 0;

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;

    // a single Q projection that outputs query + gate, interleaved per head
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, il_n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * il_n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, il_n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, il_n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * il_n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * il_n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, il_n_head_kv, n_tokens);

    // IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen4exp::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    // The Alpha fused-projection path is deliberately not used here. It folds the output
    // gate into ggml_swiglu_split, which is silu(z)*norm -- qwen4exp needs sigmoid(z)*norm.
    // Re-enabling it needs a sigmoid variant of the split op.
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * gate = alpha;   // raw alpha; softplus(alpha + dt) * a runs inside the fused op (build_delta_net_in)

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    const int64_t conv_k_pre    = model.layers[il].ssm_conv1d->ne[0];
    const int64_t conv_ch_pre   = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t n_row_r       = hparams.n_embd_r();

    // The row also carries the PLE conv history in its tail (see llama_hparams::n_embd_r),
    // so take only the GDN prefix here. build_rs_row caches the gather because the PLE path on
    // this same layer wants the identical row.
    ggml_tensor * conv_row = build_rs_row(inp, conv_states_all, il);

    ggml_tensor * conv_states = ggml_cont(ctx0,
            ggml_view_2d(ctx0, conv_row, (conv_k_pre - 1) * conv_ch_pre, n_seqs,
                    ggml_row_size(conv_row->type, n_row_r), 0));
    cb(conv_states, "conv_states", il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    // the channels must match how the tensor loader sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels, n_seqs, conv_states_all->nb[1],
                     kv_head * ggml_row_size(conv_states_all->type, n_row_r));
    cb(state_update_target, "state_update_target", il);

    // MTP/ReplaySSM: the fold owns this destination under record mode (see qwen35moe.cpp)
    if (alpha_gdn_record_conv(il, conv_input, last_conv_states, state_update_target)) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    }

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_qkv_mix = ggml_silu(ctx0, conv_output_proper);
    cb(conv_qkv_mix, "conv_output_silu", il);

    const int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    const int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim), nb1_qkv, nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;


    // if head keys and value keys differ, repeat to force matching shapes
    // (only needed when the fused GDN kernels are not doing it for us)
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // raw q/k/alpha/beta: the L2 norm, softplus gate and sigmoid run inside the fused op (or in its fallback)
    ggml_tensor * ssm_state_dst =
        ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
            kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all));
    auto attn_out = build_delta_net_in(q_conv, k_conv, v_conv, gate, beta, state,
                                       model.layers[il].ssm_dt, model.layers[il].ssm_a, eps_norm, il, ssm_state_dst);

    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    if (new_state) cb(new_state, "new_state", il);

    if (new_state && alpha_gdn_record_state(il, ssm_state_dst, new_state)) {   // nullptr when published in place
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ssm_state_dst));
    }

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen4exp::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
    cb(moe_out, "ffn_moe_out", il);

    // ALPHA_SHEXP_EARLY (default on): put the shared expert between the routed experts' gate/up and their down
    // projection in graph order. Under ALPHA_DRE_CPU the doorbell rings before gate/up and the GPU waits for the
    // CPU's missed experts after down, so the shared expert's GEMVs run inside that wait instead of after it.
    // Same ops, same math; only the order changes.
    static const bool shexp_early = [] { const char * e = getenv("ALPHA_SHEXP_EARLY"); return !e || atoi(e) != 0; }();
    if (shexp_early && model.layers[il].ffn_up_shexp != nullptr) {
        // find the routed experts' activation (the GLU feeding the down MUL_MAT_ID) among moe_out's ancestors
        ggml_tensor * glu = nullptr;
        std::vector<ggml_tensor *> stack = { moe_out };
        for (int guard = 0; !stack.empty() && guard < 64 && !glu; ++guard) {
            ggml_tensor * t = stack.back(); stack.pop_back();
            if (t->op == GGML_OP_MUL_MAT_ID && t->src[0] == model.layers[il].ffn_down_exps) { glu = t->src[1]; break; }
            for (int j = 0; j < GGML_MAX_SRC; ++j) if (t->src[j]) stack.push_back(t->src[j]);
        }
        if (glu) ggml_build_forward_expand(gf, glu);
    }

    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp,   NULL, NULL,
                model.layers[il].ffn_gate_shexp, NULL, NULL,
                model.layers[il].ffn_down_shexp, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // the shared expert has its own sigmoided gate: one scalar per token
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        if (shexp_early) ggml_build_forward_expand(gf, ffn_shexp);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.
class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_hparams & hparams, const llama_kv_cache_context * mctx) :
        hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    // Without this the base class answers "no" and the whole graph is rebuilt and re-allocated every decode
    // step. The topology depends only on the row count; the history pointer must follow the new context.
    bool can_reuse(const llm_graph_params & params) override {
        const auto * h = static_cast<const llama_memory_hybrid_context *>(params.mctx);
        mctx = h->get_attn();
        if (emb_in) return emb_in->ne[1] == (int64_t) params.ubatch.n_tokens;
        return rows && rows->ne[0] == (int64_t) hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]
    // ALPHA_PLE_INPUT, prompt-sized ubatches: the gathered rows themselves, F32 [ple_head_dim * n_heads, n_tokens],
    // filled host-side (from the precomputed cache when it covers the ubatch) instead of a CPU get_rows split
    ggml_tensor * emb_in = nullptr;
    const ggml_tensor * table = nullptr;   // the gathered table, for the host-side page prefetch

    const llama_hparams & hparams;

    // the predecessor tokens live in the attention KV cells
    const llama_kv_cache_context * mctx;

    std::vector<llama_token> prev;  // scratch, reused across calls
};

// ALPHA_PLE_PREFETCH_AHEAD: the rows llm_graph_input_ple::set_input will gather for a sequence that starts at
// position 0 (predecessors from the sequence itself; before its start reads as EOS), hinted on a detached thread
// so the CPU get_rows of a later prompt chunk does not fault its rows in from disk while the GPU waits.
// PLE row ids of one position: ctx[0] is the token, ctx[1..] its predecessors (EOS-cut)
static void ple_rows_of(const llama_hparams & hp, const int64_t * ctx, int32_t * out) {
    const int64_t n_gram = hp.ple_ngram_size, n_heads = hp.ple_n_heads, per_gram = hp.ple_heads_per_ngram;
    for (int64_t n = 2; n <= n_gram; ++n) {
        uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
        for (int64_t j = 1; j < n; ++j) mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
        const int64_t base = (n - 2) * per_gram;
        for (int64_t g = 0; g < per_gram; ++g) {
            const int64_t h_i = base + g;
            if (h_i >= n_heads) break;
            out[h_i] = (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
        }
    }
}
// dequantise rows idx[0..n) of the table into dst ([head_dim] floats each), on nth threads
static void ple_gather(const ggml_tensor * table, const int32_t * idx, int64_t n, float * dst, int nth) {
    const int64_t hd = table->ne[0];
    const auto * tt = ggml_get_type_traits(table->type);
    auto work = [&](int64_t k0, int64_t k1) {
        for (int64_t k = k0; k < k1; ++k) {
            const char * src = (const char *) table->data + (size_t) idx[k] * table->nb[1];
            if (table->type == GGML_TYPE_F32) memcpy(dst + k * hd, src, (size_t) hd * sizeof(float));
            else tt->to_float(src, dst + k * hd, hd);
        }
    };
    if (nth <= 1 || n < 4096) { work(0, n); return; }
    std::vector<std::thread> th;
    for (int t = 0; t < nth; ++t) th.emplace_back(work, n * t / nth, n * (t + 1) / nth);
    for (auto & x : th) x.join();
}
// ALPHA_PLE_INPUT: rows of a prompt gathered ahead (positions [pos0, pos0 + n)), by the serve's prefetch thread
struct ple_ahead_cache {
    std::mutex               mu;
    const ggml_tensor *      table = nullptr;
    int64_t                  pos0 = 0, n = 0;
    std::vector<llama_token> toks;     // tokens of the covered positions
    std::vector<float>       data;     // [n][n_heads * head_dim]
    bool                     ready = false;
    int                      gen = 0;
    bool                     pending = false;   // a gather for [pos0, pos0 + n) is running (head only)
    std::condition_variable  cv;
};
static ple_ahead_cache g_ple_ahead;
// ALPHA_PLE_HEAD (default 1): the first chunk's rows (positions [0, ahead_from)), gathered first by the same thread
// while the serve borrows and reserves; that chunk's set_input waits for them instead of gathering them itself
static ple_ahead_cache g_ple_head;
static bool ple_head_on() { static const bool on = [] { const char * e = getenv("ALPHA_PLE_HEAD"); return !e || atoi(e) != 0; }(); return on; }
// the cache's rows for this ubatch (same positions and tokens), or null
static const float * ple_cache_rows(const ple_ahead_cache & c, const ggml_tensor * table, const llama_ubatch * ub, int64_t row_floats) {
    const int64_t p0 = ub->pos[0], n = ub->n_tokens;
    if (!c.ready || c.table != table || !ub->token || p0 < c.pos0 || p0 + n > c.pos0 + c.n) return nullptr;
    for (int64_t i = 0; i < n; ++i) {
        if (ub->pos[i] != p0 + i || c.toks[(size_t) (p0 + i - c.pos0)] != ub->token[i]) return nullptr;
    }
    return c.data.data() + (size_t) (p0 - c.pos0) * row_floats;
}
static std::atomic<int64_t> g_ple_ahead_from { 0 };
void qwen4exp_ple_set_ahead_from(int64_t from) {
    std::lock_guard<std::mutex> lk(g_ple_ahead.mu);
    g_ple_ahead.ready = false;
    g_ple_ahead_from.store(from);
}

void qwen4exp_ple_prefetch(const llama_model & model, const llama_token * tokens, int32_t n) {
#if defined(__linux__)
    const ggml_tensor * table = model.per_layer_tok_embd;
    if (!table || !table->data || !table->buffer || !ggml_backend_buffer_is_host(table->buffer) || n <= 0) return;
    const auto & hp = model.hparams;
    if (hp.ple_n_heads <= 0 || hp.ple_ngram_size < 2) return;
    std::vector<llama_token> toks(tokens, tokens + n);
    {
        const int64_t from = g_ple_ahead_from.load();
        static const bool on_in = [] { const char * e = getenv("ALPHA_PLE_INPUT"); return !e || atoi(e) != 0; }();
        std::lock_guard<std::mutex> lk(g_ple_head.mu);
        g_ple_head.ready = false;
        g_ple_head.pending = on_in && ple_head_on() && from >= 64 && from <= n;
        g_ple_head.table = table; g_ple_head.pos0 = 0; g_ple_head.n = g_ple_head.pending ? from : 0;
    }
    std::thread([&model, toks = std::move(toks)]() {
        const auto & hp = model.hparams;
        const ggml_tensor * table = model.per_layer_tok_embd;
        const int64_t n_gram = hp.ple_ngram_size, n_heads = hp.ple_n_heads, per_gram = hp.ple_heads_per_ngram;
        const int64_t eos = hp.ple_eos_token_id;
        const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
        const size_t row_bytes = table->nb[1];
        const int64_t N = (int64_t) toks.size();
        auto work = [&](int64_t i0, int64_t i1) {
            std::vector<int64_t> ctx(n_gram);
            for (int64_t i = i0; i < i1; ++i) {
                ctx[0] = toks[i];
                bool cut = false;
                for (int64_t s = 1; s < n_gram; ++s) {
                    const llama_token t = (cut || i - s < 0) ? LLAMA_TOKEN_NULL : toks[i - s];
                    cut = cut || t < 0 || t == eos;
                    ctx[s] = cut ? eos : t;
                }
                for (int64_t g2 = 2; g2 <= n_gram; ++g2) {
                    uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
                    for (int64_t j = 1; j < g2; ++j) mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
                    const int64_t base = (g2 - 2) * per_gram;
                    for (int64_t g = 0; g < per_gram; ++g) {
                        const int64_t h_i = base + g;
                        if (h_i >= n_heads) break;
                        const int64_t row = (int64_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
                        const uintptr_t a  = (uintptr_t) table->data + (uintptr_t) row * row_bytes;
                        const uintptr_t lo = a & ~(page - 1);
                        const uintptr_t hi = (a + row_bytes + page - 1) & ~(page - 1);
                        madvise((void *) lo, hi - lo, MADV_WILLNEED);
                    }
                }
            }
        };
        auto hint_range = [&](int64_t a0, int64_t a1) {
            const int64_t len = a1 - a0;
            if (len <= 0) return;
            const int nth = len >= 4096 ? 8 : 1;
            std::vector<std::thread> th;
            for (int t = 0; t < nth; ++t) th.emplace_back(work, a0 + len * t / nth, a0 + len * (t + 1) / nth);
            for (auto & x : th) x.join();
        };
        // rows of positions [a0, a1) into a cache
        auto gather_range = [&](int64_t a0, int64_t a1, ple_ahead_cache & dst) {
            const int64_t m = a1 - a0, nh = n_heads, hd = table->ne[0];
            std::vector<int32_t> idx((size_t) (m * nh));
            std::vector<int64_t> c(n_gram);
            for (int64_t i = a0; i < a1; ++i) {
                c[0] = toks[i];
                bool cut = false;
                for (int64_t s = 1; s < n_gram; ++s) {
                    const llama_token t = (cut || i - s < 0) ? LLAMA_TOKEN_NULL : toks[i - s];
                    cut = cut || t < 0 || t == eos;
                    c[s] = cut ? eos : t;
                }
                ple_rows_of(hp, c.data(), idx.data() + (i - a0) * nh);
            }
            std::vector<float> data((size_t) (m * nh * hd));
            ple_gather(table, idx.data(), m * nh, data.data(), 8);
            {
                std::lock_guard<std::mutex> lk(dst.mu);
                dst.table = table; dst.pos0 = a0; dst.n = m;
                dst.toks.assign(toks.begin() + a0, toks.begin() + a1);
                dst.data = std::move(data);
                dst.ready = true;
                dst.pending = false;
            }
            dst.cv.notify_all();
        };

        static const bool on = [] { const char * e = getenv("ALPHA_PLE_INPUT"); return !e || atoi(e) != 0; }();
        const int64_t from = g_ple_ahead_from.load();
        bool head = false;
        { std::lock_guard<std::mutex> lk(g_ple_head.mu); head = g_ple_head.pending; }
        if (head) {
            hint_range(0, from);
            gather_range(0, from, g_ple_head);   // publishes and wakes the first chunk's set_input
            hint_range(from, N);
        } else {
            hint_range(0, N);
        }

        // ALPHA_PLE_INPUT: the rows of every position from ALPHA_PLE_AHEAD_FROM (the prompt's later chunks) on
        if (!on || from <= 0 || from >= N) return;
        gather_range(from, N, g_ple_ahead);
    }).detach();
#else
    (void) model; (void) tokens; (void) n;
#endif
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still
    // needs a row for ggml_get_rows. Stand in the image token id the reference hashes, or EOS
    // when the file has no such key.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 &&
                    "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells; apply_ubatch() already stored this ubatch, so its
    // own tokens count too
    mctx->get_prev_tokens(*ubatch, (uint32_t) n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it; a missing predecessor
        // (before the sequence start, or no cached cell) reads as EOS. The token's own EOS
        // does not cut its context, as in the reference.
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // prev[] is oldest-first; missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    if (!emb_in) ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
    // ALPHA_PLE_HEAD: the prefetch thread is gathering this (first) chunk's rows: wait for them
    if (emb_in && ubatch->token) {
        std::unique_lock<std::mutex> lk(g_ple_head.mu);
        const int64_t p0 = ubatch->pos[0];
        if (g_ple_head.pending && g_ple_head.table == table && p0 >= g_ple_head.pos0 && p0 + n_tokens <= g_ple_head.pos0 + g_ple_head.n) {
            g_ple_head.cv.wait_for(lk, std::chrono::seconds(10), [] { return !g_ple_head.pending; });
        }
    }
    // a ubatch the ahead (or head) cache covers needs no page hints
    bool ple_cached = false;
    if (emb_in && ubatch->token) {
        const int64_t p0 = ubatch->pos[0];
        for (ple_ahead_cache * c : { &g_ple_ahead, &g_ple_head }) {
            std::lock_guard<std::mutex> lk(c->mu);
            ple_cached = ple_cached || (c->ready && c->table == table && p0 >= c->pos0 && p0 + n_tokens <= c->pos0 + c->n);
        }
    }

#if defined(__linux__)
    // The table is tens of GB of mmap'd file and the rows are hash-random, so with the experts holding most of
    // RAM each row is often a page fault, and the CPU get_rows takes them one at a time. Hint every row first
    // so the reads are in flight together. ALPHA_PLE_PREFETCH=0 disables.
    static const bool prefetch = [] { const char * e = getenv("ALPHA_PLE_PREFETCH"); return !e || atoi(e) != 0; }();
    // one madvise per row is ~130k syscalls at 8192 tokens, so issue them from several threads for
    // prefill-sized batches
    if (prefetch && !ple_cached && table && table->data && table->buffer && ggml_backend_buffer_is_host(table->buffer)) {
        static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
        const size_t row_bytes = table->nb[1];
        auto hint = [&](size_t k0, size_t k1) {
            for (size_t k = k0; k < k1; ++k) {
                const uintptr_t a  = (uintptr_t) table->data + (uintptr_t) idx[k] * row_bytes;
                const uintptr_t lo = a & ~(page - 1);
                const uintptr_t hi = (a + row_bytes + page - 1) & ~(page - 1);
                madvise((void *) lo, hi - lo, MADV_WILLNEED);
            }
        };
        const size_t n = idx.size();
        const int nth = n >= 4096 ? 8 : 1;
        if (nth == 1) {
            hint(0, n);
        } else {
            std::vector<std::thread> th;
            for (int t = 0; t < nth; ++t) th.emplace_back(hint, n*t/nth, n*(t + 1)/nth);
            for (auto & x : th) x.join();
        }
    }
#endif
    if (emb_in) {
        // ALPHA_PLE_INPUT: the rows themselves, from the ahead cache when it covers this ubatch (same positions and
        // tokens), else gathered here on 8 threads
        const int64_t hd = table->ne[0];
        const size_t bytes = (size_t) n_tokens * n_heads * hd * sizeof(float);
        bool done = false;
        for (ple_ahead_cache * c : { &g_ple_ahead, &g_ple_head }) {
            if (done) break;
            std::lock_guard<std::mutex> lk(c->mu);
            if (const float * rows = ple_cache_rows(*c, table, ubatch, n_heads * hd)) {
                ggml_backend_tensor_set(emb_in, rows, 0, bytes);
                done = true;
                // ALPHA_PLE_CHECK: gather the same rows directly and compare (bitwise) with the cached ones
                static const bool chk = getenv("ALPHA_PLE_CHECK") != nullptr;
                if (chk) {
                    std::vector<float> ref((size_t) n_tokens * n_heads * hd);
                    ple_gather(table, idx.data(), n_tokens * n_heads, ref.data(), 8);
                    fprintf(stderr, "[ple-check] p0=%lld n=%lld from the %s cache: %s\n", (long long) ubatch->pos[0], (long long) n_tokens,
                            c == &g_ple_head ? "head" : "ahead", memcmp(ref.data(), rows, bytes) == 0 ? "identical" : "DIFFERENT");
                }
            }
        }
        if (!done) {
            std::vector<float> buf((size_t) n_tokens * n_heads * hd);
            ple_gather(table, idx.data(), n_tokens * n_heads, buf.data(), 8);
            ggml_backend_tensor_set(emb_in, buf.data(), 0, bytes);
        }
    }
}

ggml_tensor * llm_build_qwen4exp::build_inp_ple(const llama_kv_cache_context * mctx_attn) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(hparams, mctx_attn);

    static const bool use_in = [] { const char * e = getenv("ALPHA_PLE_INPUT"); return !e || atoi(e) != 0; }();
    if (use_in && n_tokens >= 64 && model.per_layer_tok_embd && model.per_layer_tok_embd->buffer &&
        ggml_backend_buffer_is_host(model.per_layer_tok_embd->buffer) &&
        (model.per_layer_tok_embd->type == GGML_TYPE_F32 || ggml_get_type_traits(model.per_layer_tok_embd->type)->to_float)) {
        ple_inp->emb_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb_in);
        ple_inp->table = model.per_layer_tok_embd;
        ggml_tensor * emb = ple_inp->emb_in;
        res->add_input(std::move(ple_inp));
        cb(emb, "ple_embd", -1);
        return emb;
    }

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ple_inp->table = model.per_layer_tok_embd;
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llm_build_qwen4exp::build_rs_row(
        llm_graph_input_rs * inp,
        ggml_tensor *        states_all,
        int                  il) {
    GGML_UNUSED(il);

    auto it = rs_rows.find(states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(states_all,
                build_rs(inp, states_all, hparams.n_embd_r(), ubatch.n_seqs)).first;
    }
    return it->second;
}

ggml_tensor * llm_build_qwen4exp::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc     = hparams.hc_count;
    const int64_t hc_dim = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of earlier ubatches is prepended so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // The PLE conv history occupies the tail of this layer's recurrent row, after the GDN
    // conv state (see llama_hparams::n_embd_r). Slice it out, prepend it, write the new tail back.
    ggml_tensor * conv_states_all = inp->mctx->get_r_l(il);
    const auto    kv_head         = inp->mctx->get_head();

    const int64_t n_row  = hparams.n_embd_r();
    const int64_t n_gdn  = n_row - hist * hc_dim;
    GGML_ASSERT(n_gdn >= 0 && "qwen4exp: recurrent row too small for the PLE conv history");

    ggml_tensor * row = build_rs_row(inp, conv_states_all, il);

    ggml_tensor * state = ggml_view_2d(ctx0, row, hist * hc_dim, n_seqs,
            ggml_row_size(row->type, n_row),
            ggml_row_size(row->type, n_gdn));
    state = ggml_cont(ctx0, state);
    state = ggml_reshape_3d(ctx0, state, hist, hc_dim, n_seqs);
    cb(state, "ple_conv_state", il);

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * x3 = ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs);
    ggml_tensor * padded = ggml_concat(ctx0, state, ggml_transpose(ctx0, x3), 0);
    cb(padded, "ple_conv_input", il);

    {
        // keep the last `hist` columns as the new history
        ggml_tensor * tail = ggml_view_3d(ctx0, padded, hist, hc_dim, n_seqs,
                padded->nb[1], padded->nb[2],
                ggml_row_size(padded->type, padded->ne[0] - hist));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all, hist * hc_dim, n_seqs,
                conv_states_all->nb[1],
                kv_head * ggml_row_size(conv_states_all->type, n_row) +
                          ggml_row_size(conv_states_all->type, n_gdn));

        // MTP/ReplaySSM: under record mode the fold writes the history after the accepted prefix
        if (alpha_gdn_record_conv2(il, padded, tail, dst)) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
        }
    }

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        // this kernel keeps the file type, so cast before it multiplies an f32 activation
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}

// QSA (Qwen Sparse Attention). The indexer keeps one key per token; those keys are pooled into
// blocks of `compress_ratio` cells, scored against a per-token indexer query, and the top-k
// blocks become the only cells the real attention may see.
//
// Like the reference, this is applied as a mask over dense attention rather than a sparse
// kernel, so it costs correctness-shaped work, not speed. While n_kv <= indexer_top_k every block
// is selected and the result is identical to plain dense attention.
// ALPHA_QSA_PKC (default on): pooled-key cache mode for a ubatch, from the planner's stale-block count:
//   0 = off (cache unusable: several sequences/streams, mrope ranks, or ALPHA_QSA_PKC=0)
//   1 = pool every block and write them all through to the cache (prefill-sized ubatches)
//   2 = pool only the stale blocks, gather the rest from the cache; *nu = blocks recomputed (power of two >= 4)
static int qsa_pkc_mode(int32_t plan, int64_t n_tokens, int64_t * nu) {
    static const bool on = [] { const char * e = getenv("ALPHA_QSA_PKC"); return !e || atoi(e) != 0; }();
    if (!on || plan < 0) {
        return 0;
    }
    if (n_tokens <= 16 && plan <= 32) {
        int64_t n = 4;
        while (n < plan) n *= 2;
        *nu = n;
        return 2;
    }
    return 1;
}

class llm_build_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
        mctx->set_input_qsa_pkc(pkc_mode, pk_map, pk_upd, pk_upd_cells, pk_upd_pos, ubatch, ratio, cell_blk->ne[0]);
        if (pk_valid) {
            const int32_t * map = (const int32_t *) pk_map->data;
            float * v = (float *) pk_valid->data;
            const int64_t nb = pk_valid->ne[0];
            for (int64_t b = 0; b < nb; ++b) v[b] = map[b] < nb ? 1.0f : 0.0f;   // the scratch row is nb
        }
    }

    // The shapes follow n_tokens, the indexer cache's n_kv and the stream count (n_blocks and the bias
    // width derive from those); the attention input already vets the KQ mask that decides blk_bias.
    bool can_reuse(const llm_graph_params & params) override {
        const auto * h = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);
        mctx = h;
        if (!(k_idxs->ne[0] == params.ubatch.n_tokens &&
              cell_blk->ne[0] == (int64_t) h->get_idx()->get_n_kv() &&
              cell_blk->ne[1] == (int64_t) h->get_n_stream())) {
            return false;
        }
        // the pooled-key cache mode is part of the topology, and mode 2 recomputes at most pk_upd->ne[0] blocks
        int64_t nu = 0;
        const int32_t plan = h->qsa_pkc_plan(&params.ubatch, ratio, cell_blk->ne[0]);
        const int     mode = qsa_pkc_mode(plan, params.ubatch.n_tokens, &nu);
        return mode == pkc_mode && (mode != 2 || plan <= pk_upd->ne[0]);
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    // pooled-key cache, see qsa_pkc_mode / llama_memory_hybrid_idx_context::set_input_qsa_pkc
    int           pkc_mode     = 0;
    ggml_tensor * pk_map       = nullptr;   // I32 [n_blocks]   block id -> cache row (modes 1, 2)
    ggml_tensor * pk_upd       = nullptr;   // I32 [NU]         rows to recompute (mode 2)
    ggml_tensor * pk_upd_cells = nullptr;   // I32 [ratio*NU]   their member cells
    ggml_tensor * pk_upd_pos   = nullptr;   // I32 [4*NU]       their rope positions
    ggml_tensor * pk_valid     = nullptr;   // F32 [n_blocks]   1 for real blocks (ALPHA_QSA_PKC_VERIFY only)

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;
};

ggml_tensor * llm_build_qwen4exp::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block; the rest is the
    // visible/not test the attention mask already carries, so upload the per-block half only.
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out.
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        ggml_set_input(qsa->bias);

        int64_t nu = 0;
        const int32_t pkc_plan = mctx_hyb->qsa_pkc_plan(&ubatch, (uint32_t) r, n_kv);
        qsa->pkc_mode = qsa_pkc_mode(pkc_plan, n_tokens, &nu);
        static int pkc_logs = 0;
        if (getenv("ALPHA_QSA_PKC_LOG") && pkc_logs < 400) {
            pkc_logs++;
            LLAMA_LOG_INFO("[qsa-pkc] r=%d n_tokens=%d n_kv=%d plan=%d mode=%d nu=%d\n", (int) r, (int) n_tokens, (int) n_kv, pkc_plan, qsa->pkc_mode, (int) nu);
        }
        if (qsa->pkc_mode != 0) {
            qsa->pk_map = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_blocks);
            ggml_set_input(qsa->pk_map);
        }
        if (qsa->pkc_mode == 2) {
            qsa->pk_upd       = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, nu);
            qsa->pk_upd_cells = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, r*nu);
            qsa->pk_upd_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*nu);
            ggml_set_input(qsa->pk_upd);
            ggml_set_input(qsa->pk_upd_cells);
            ggml_set_input(qsa->pk_upd_pos);
            if (getenv("ALPHA_QSA_PKC_VERIFY")) {
                qsa->pk_valid = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_blocks);
                ggml_set_input(qsa->pk_valid);
            }
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].indexer_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // pool, norm and rotate n_pool blocks whose members (r cells each) are listed in `cells_idx`, at `pos`
    auto pool_blocks = [&](ggml_tensor * k_src, ggml_tensor * cells_idx, ggml_tensor * pos, int64_t n_pool, int64_t ns) {
        // gathers per stream: blk_cells row s indexes stream s's own cells
        ggml_tensor * members = ggml_get_rows(ctx0, k_src, cells_idx);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_pool, ns);

        // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
        ggml_tensor * pl = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, members, idx_dim, n_pool, ns,
                            members->nb[2], members->nb[3], i*members->nb[1]));
            pl = pl ? ggml_add(ctx0, pl, slice) : slice;
        }
        pl = ggml_scale(ctx0, pl, 1.0f/(float) r);
        cb(pl, "indexer_k_pooled", il);

        // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535
        pl = ggml_reshape_3d(ctx0, pl, idx_dim, n_pool*ns, 1);
        pl = build_norm(pl, model.layers[il].indexer_k_norm, nullptr, LLM_NORM_RMS, il);

        // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after
        pl = ggml_reshape_3d(ctx0, pl, idx_dim, 1, n_pool*ns);
        pl = ggml_rope_multi(ctx0, pl, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        return ggml_reshape_3d(ctx0, pl, idx_dim, n_pool, ns);
    };

    ggml_tensor * pooled = nullptr;
    if (inp->pkc_mode == 0) {
        pooled = pool_blocks(k_all, inp->blk_cells, inp->blk_pos, n_blocks, n_stream);
    } else {
        // pooled-key cache: row pb of this layer's indexer V (f32, idx_dim wide) = position bucket pb's key
        ggml_tensor * vc = mctx_idx->get_v(ctx0, il);
        GGML_ASSERT(vc->type == GGML_TYPE_F32 && vc->ne[0] == idx_dim && n_stream == 1);
        vc = ggml_view_2d(ctx0, vc, idx_dim, n_kv, vc->nb[2], 0);
        if (inp->pkc_mode == 1) {
            pooled = pool_blocks(k_all, inp->blk_cells, inp->blk_pos, n_blocks, 1);
            ggml_build_forward_expand(gf, ggml_set_rows(ctx0, vc, ggml_reshape_2d(ctx0, pooled, idx_dim, n_blocks), inp->pk_map));
        } else {
            const int64_t nu = inp->pk_upd->ne[0];
            ggml_tensor * fresh = pool_blocks(k_all, inp->pk_upd_cells, inp->pk_upd_pos, nu, 1);
            ggml_tensor * vw = ggml_set_rows(ctx0, vc, ggml_reshape_2d(ctx0, fresh, idx_dim, nu), inp->pk_upd);
            pooled = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, vw, inp->pk_map), idx_dim, n_blocks, 1);
            if (inp->pk_valid) {
                // recompute every block uncached and sum |cached - recomputed| over the real blocks
                ggml_tensor * full = pool_blocks(k_all, inp->blk_cells, inp->blk_pos, n_blocks, 1);
                ggml_tensor * d = ggml_abs(ctx0, ggml_sub(ctx0, ggml_reshape_2d(ctx0, pooled, idx_dim, n_blocks),
                                                                ggml_reshape_2d(ctx0, full, idx_dim, n_blocks)));
                d = ggml_mul(ctx0, d, ggml_reshape_2d(ctx0, inp->pk_valid, 1, n_blocks));
                d = ggml_sum(ctx0, d);
                qsa_pkc_dbg = qsa_pkc_dbg ? ggml_add(ctx0, qsa_pkc_dbg, d) : d;
                ggml_set_name(qsa_pkc_dbg, "qsa_pkc_diff");
                ggml_set_output(qsa_pkc_dbg);
                ggml_build_forward_expand(gf, qsa_pkc_dbg);
                res->t_dbg = qsa_pkc_dbg;
            }
        }
    }
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].indexer_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].indexer_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width_c = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // ALPHA_QSA_CHUNK: score and select in token sub-chunks. Unchunked, every layer materialises several
    // [n_kv x n_tokens] f32 tensors, and the reserve sizes n_kv for the whole context (~0.5 GB each at 32K
    // context and a 4096-token micro-batch). Same ops on column slices, so the selection is identical.
    // chunk = an element budget (ALPHA_QSA_CHUNK_ELEMS, default 64M) over n_kv, at least 512; ALPHA_QSA_CHUNK fixes it
    static const int64_t qsa_fixed = [] { const char * e = getenv("ALPHA_QSA_CHUNK"); return e ? atoll(e) : -1LL; }();
    static const int64_t qsa_elems = [] { const char * e = getenv("ALPHA_QSA_CHUNK_ELEMS"); return e ? atoll(e) : (64LL << 20); }();
    const int64_t qsa_chunk = qsa_fixed >= 0 ? qsa_fixed : std::max<int64_t>(512, qsa_elems / std::max<int64_t>(1, n_kv));
    // slice device copies of the host inputs, not the inputs themselves (ALPHA_QSA_DEV, see models.h)
    ggml_tensor * bias_d = n_tps > 1024 ? qsa_dev(inp->bias) : inp->bias;
    ggml_tensor * kqm_d  = (kq_mask && n_tps > 1024) ? qsa_dev(kq_mask) : kq_mask;
    if (qsa_chunk > 0 && n_stream == 1 && n_tps > qsa_chunk) {
        ggml_tensor * mask_f32 = nullptr;
        if (blk_bias) mask_f32 = kqm_d->type == GGML_TYPE_F32 ? kqm_d : nullptr;
        ggml_tensor * result = nullptr;
        for (int64_t t0 = 0; t0 < n_tps; t0 += qsa_chunk) {
            const int64_t C = std::min<int64_t>(qsa_chunk, n_tps - t0);
            ggml_tensor * qc = ggml_view_3d(ctx0, q, idx_dim, n_idx_h, C, q->nb[1], q->nb[2], t0*q->nb[2]);
            qc = ggml_cont(ctx0, qc);
            ggml_tensor * sc = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, pooled, idx_dim, n_blocks),
                                            ggml_reshape_2d(ctx0, qc, idx_dim, n_idx_h*C));
            sc = ggml_reshape_3d(ctx0, sc, n_blocks, n_idx_h, C);
            sc = ggml_relu(ctx0, sc);
            ggml_tensor * sm = nullptr;
            for (int64_t h = 0; h < n_idx_h; ++h) {
                ggml_tensor * slice = ggml_view_2d(ctx0, sc, n_blocks, C, sc->nb[2], h*sc->nb[1]);
                sm = sm ? ggml_add(ctx0, sm, slice) : ggml_cont(ctx0, slice);
            }
            if (blk_bias) {
                sm = ggml_add(ctx0, sm, ggml_view_2d(ctx0, bias_d, n_blocks, C, bias_d->nb[1], t0*bias_d->nb[1]));
            }
            ggml_tensor * ex = ggml_get_rows(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, sm)), ggml_view_1d(ctx0, inp->cell_blk, n_kv, 0));
            ex = ggml_cont(ctx0, ggml_transpose(ctx0, ex));   // [n_kv, C]
            if (blk_bias) {
                ggml_tensor * mk = ggml_view_2d(ctx0, kqm_d, n_kv, C, kqm_d->nb[1], t0*kqm_d->nb[1]);
                mk = mask_f32 ? mk : ggml_cast(ctx0, mk, GGML_TYPE_F32);
                ex = ggml_add(ctx0, ex, mk);
            } else {
                ex = ggml_add(ctx0, ex, ggml_view_2d(ctx0, bias_d, n_kv, C, bias_d->nb[1], t0*bias_d->nb[1]));
            }
            ggml_tensor * tk = ggml_cont(ctx0, ggml_top_k(ctx0, ex, width_c));   // [width, C]
            result = result ? ggml_concat(ctx0, result, tk, 1) : tk;
        }
        ggml_tensor * top_k = ggml_reshape_4d(ctx0, result, width_c, n_tps, 1, n_stream);
        cb(top_k, "indexer_top_k", il);
        return top_k;
    }

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer.
    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream));
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);

    // the heads sit side by side on ne[1] and there are only a few of them
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                score->nb[2], score->nb[3], h*score->nb[1]);
        summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
    }

    score = summed;
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, bias_d);
    }

    // ALPHA_QSA_TOPK_EXPAND (default on): the same selection without the [n_kv x n_tps] expanded score tensor: the
    // top-k kernel computes each cell's score (its block's score + its mask value) as it reads it. Same keys, same
    // index-order tie rule as the radix select on the materialised scores.
    {
        static const bool fused_expand = [] { const char * e = getenv("ALPHA_QSA_TOPK_EXPAND"); return !e || atoi(e) != 0; }();
        const int64_t width_e = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);
        if (fused_expand && blk_bias && n_stream == 1 && n_kv > 1024 && kqm_d->ne[0] == n_kv && kqm_d->ne[2] == 1 &&
            kqm_d->ne[3] == 1 && ggml_is_contiguous(score)) {
            ggml_tensor * sb = ggml_reshape_2d(ctx0, score, n_blocks, n_tps);
            ggml_tensor * cb_ = ggml_view_1d(ctx0, inp->cell_blk, n_kv, 0);
            ggml_tensor * top_k = ggml_top_k_expand(ctx0, sb, cb_, kqm_d, (int) width_e);
            top_k = ggml_reshape_4d(ctx0, top_k, width_e, n_tps, 1, n_stream);
            cb(top_k, "indexer_top_k", il);
            return top_k;
        }
    }

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on
    // a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kqm_d->type == GGML_TYPE_F32 ? kqm_d : ggml_cast(ctx0, kqm_d, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, bias_d);
    }
    cb(expanded, "indexer_score_tokens", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
ggml_tensor * llm_build_qwen4exp::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // The dense path rotates q/k/v with a Hadamard before they reach a rotated quantized cache.
    // That helper (ggml_mul_mat_aux) is static inside llama-graph.cpp and is not reachable here,
    // and qwen4exp does not configure rotated KV. Assert rather than silently skip it, which would
    // produce plausible but wrong attention if it were ever enabled.
    GGML_ASSERT(!inp->self_k_rot && !inp->self_v_rot &&
                "qwen4exp QSA does not implement rotated KV");

    // added together so the scheduler does not reorder them, which keeps the split count down
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    // Assemble in F32: under flash attention get_kq_mask() hands back an F16 copy, and both
    // ggml_fill and the additive combine below require F32. Cast back at the end to whatever
    // build_attn_mha expects.
    ggml_tensor * kq_mask     = inp->self_kq_mask;    // F32, or F16 under ALPHA_KQ_MASK_F16 (cast per slice below)
    if (q_cur->ne[2] > 1024) kq_mask = qsa_dev(kq_mask);   // slices of a device copy (large micro-batches only; a captured 512-token prefill graph faults)
    ggml_tensor * kq_mask_use = inp->get_kq_mask();   // F16 when flash attention is on
    auto as_f32 = [&](ggml_tensor * t) { return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx0, t, GGML_TYPE_F32); };

    // ALPHA_QSA_CHUNK: attend in query-token chunks. The sparse mask below is [n_kv x n_tokens] F32 per layer and
    // the reserve sizes n_kv for the whole context; per chunk it is [n_kv x chunk]. Each query row attends as it
    // would unchunked (the KV and its own mask row are unchanged).
    {
        static const int64_t qsa_fixed = [] { const char * e = getenv("ALPHA_QSA_CHUNK"); return e ? atoll(e) : -1LL; }();
        static const int64_t qsa_elems = [] { const char * e = getenv("ALPHA_QSA_CHUNK_ELEMS"); return e ? atoll(e) : (64LL << 20); }();
        const int64_t qsa_chunk = qsa_fixed >= 0 ? qsa_fixed : std::max<int64_t>(512, qsa_elems / std::max<int64_t>(1, kq_mask->ne[0]));
        const int64_t n_q = q_cur->ne[2];

    // ALPHA_QSA_FA_SPARSE (default on), decode-sized batches: flash attention over the top_k cells only, read in
    // place from the cache (ggml-cuda/fattn-sparse.cu), instead of the [n_kv x n_q] mask over dense attention below,
    // which reads every cell
    {
        static const bool fa_sparse = [] { const char * e = getenv("ALPHA_QSA_FA_SPARSE"); return !e || atoi(e) != 0; }();   // default ON
        const int64_t n_q = q_cur->ne[2];
        if (fa_sparse && cparams.flash_attn && kq_mask_use && n_q <= 8 && top_k->ne[3] == 1 && top_k->ne[2] == 1 &&
            top_k->ne[1] == n_q && ggml_is_contiguous(top_k) && kq_mask_use->ne[3] == 1) {
            ggml_tensor * k = mctx_cur->get_k(ctx0, il);   // [D, H_kv, n_kv, 1]
            ggml_tensor * v = mctx_cur->get_v(ctx0, il);
            if (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && k->ne[3] == 1 && v->nb[1] <= v->nb[2] &&
                k->ne[0] == 256 && v->ne[0] == 256 && q_cur->ne[1] == 12 * k->ne[1]) {
                ggml_tensor * qp = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
                ggml_tensor * kp = ggml_permute(ctx0, k, 0, 2, 1, 3);
                ggml_tensor * vp = ggml_permute(ctx0, v, 0, 2, 1, 3);
                ggml_tensor * cur = ggml_flash_attn_ext(ctx0, qp, kp, vp, kq_mask_use, kq_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
                ggml_flash_attn_ext_set_cells(cur, ggml_reshape_2d(ctx0, top_k, top_k->ne[0], n_q));
                cb(cur, LLAMA_TENSOR_NAME_FATTN, il);   // the auto-FA check finds FA nodes by this per-layer name
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
                cb(cur, "kqv_out", il);
                return cur;
            }
        }
    }

        if (qsa_chunk > 0 && n_q > qsa_chunk && kq_mask->ne[3] == 1 && kq_mask->ne[1] == n_q && top_k->ne[3] == 1) {
            ggml_tensor * k = mctx_cur->get_k(ctx0, il);
            ggml_tensor * v = mctx_cur->get_v(ctx0, il);
            ggml_tensor * out = nullptr;
            const int64_t n_kv_m = kq_mask->ne[0];
            // ALPHA_QSA_MASK_F16 (default 1): the mask is F16 and flash attention reads F16, so build each slice in F16
            // in place (fill a fresh tensor, set the selected cells, add the causal mask into it) rather than through
            // F32 copies. Values are 0 / -inf only, so the result is bit-identical.
            static const bool m16_env = [] { const char * e = getenv("ALPHA_QSA_MASK_F16"); return !e || atoi(e) != 0; }();
            const bool m16 = m16_env && kq_mask->type == GGML_TYPE_F16 && kq_mask_use && kq_mask_use->type == GGML_TYPE_F16;
            for (int64_t t0 = 0; t0 < n_q; t0 += qsa_chunk) {
                const int64_t C = std::min<int64_t>(qsa_chunk, n_q - t0);
                ggml_tensor * mkv = ggml_view_2d(ctx0, kq_mask, n_kv_m, C, kq_mask->nb[1], t0*kq_mask->nb[1]);
                ggml_tensor * mk  = m16 ? mkv : as_f32(mkv);
                // m16 uses a FILL node shaped like the slice: a node is freed after its last use, while a fresh leaf
                // filled in place would stay allocated for the whole graph
                ggml_tensor * all = m16 ? ggml_fill(ctx0, mkv, -INFINITY)
                                        : ggml_fill(ctx0, ggml_cont(ctx0, mk), -INFINITY);                      // [n_kv, C]
                ggml_tensor * all4 = ggml_view_4d(ctx0, all, 1, n_kv_m, C, 1, all->nb[0], all->nb[1], all->nb[2], 0);
                ggml_tensor * tk  = ggml_view_3d(ctx0, top_k, top_k->ne[0], C, 1, top_k->nb[1], top_k->nb[1]*C, t0*top_k->nb[1]);
                ggml_tensor * zr  = qsa_zeros(tk->ne[0], C);
                ggml_tensor * mt  = ggml_set_rows(ctx0, all4, zr, ggml_cont(ctx0, tk));
                mt = ggml_view_2d(ctx0, mt, n_kv_m, C, mt->nb[2], 0);
                mt = m16 ? ggml_add_inplace(ctx0, mt, mk) : ggml_add(ctx0, mt, mk);
                mt = ggml_reshape_4d(ctx0, mt, n_kv_m, C, 1, 1);
                if (!m16 && kq_mask_use && kq_mask_use->type != GGML_TYPE_F32) mt = ggml_cast(ctx0, mt, kq_mask_use->type);
                ggml_tensor * qc = ggml_cont(ctx0, ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], C,
                                                                q_cur->nb[1], q_cur->nb[2], t0*q_cur->nb[2]));
                ggml_tensor * oc = build_attn_mha(qc, k, v, nullptr, mt, nullptr, nullptr, kq_scale, il);
                out = out ? ggml_concat(ctx0, out, oc, 1) : oc;
            }
            cb(out, "kqv_out", il);
            return out;
        }
    }

    // start from an all -INFINITY mask and punch holes where top_k points
    kq_mask = as_f32(kq_mask);
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // zero-filled source rows: unmasking a top_k cell means writing 0 over its -INFINITY
    // ALPHA_QSA_ZERO_SHARE: one shared zero source per shape (a leaf per layer would stay allocated for the
    // whole graph)
    ggml_tensor * zeros = top_k_3d->ne[2] == 1 ? qsa_zeros(top_k_3d->ne[0], top_k_3d->ne[1])
                        : ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]), 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // back to the KQ mask's own shape
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original mask, which still drops empty, foreign and future cells
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    if (kq_mask_use && kq_mask_use->type != GGML_TYPE_F32) {
        kq_mask_top_k = ggml_cast(ctx0, kq_mask_top_k, kq_mask_use->type);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // Dense attention under a sparse mask. The reference does the same (its sparse kernel is
    // still a TODO), so this matches it exactly rather than approximating it.
    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il);
    cb(cur, "kqv_out", il);

    return cur;
}
