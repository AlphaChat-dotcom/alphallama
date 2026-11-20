#include "models.h"

#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <cstdlib>

// GLM5-Next (GLM-5.3-Flash): hybrid KDA (linear) + nope MLA with a k-pool DSA indexer,
// mHC residual streams, DeepSeek-style MoE.
//
// Ported from upstream llama.cpp 649dcb103 (PR #27773), adapted to this fork's graph helpers:
//   - the KDA layers follow this fork's kimi-linear state handling (no recurrent rollback rows);
//   - the mHC mixers are built from plain ggml ops (upstream's unfused path); ALPHA_GLM5_HC_FUSED=1 uses
//     this fork's DeepSeek-V4 ops instead (ggml_dsv4_hc_split_sinkhorn / _weighted_sum / _expand);
//   - the k-pool indexer re-pools every complete pool on every ubatch (no pooled-key cache, no gather
//     path), and is skipped altogether while every pool fits in the top-k budget, where the selection
//     is the whole causal context;
//   - the NextN (MTP) block has no graph here: with ALPHA_MTP it is loaded and llama-context-mtp5.cpp drafts with
//     it from the hidden this graph exports (t_mtp_hidden); otherwise its tensors are skipped.
//
// ALPHA_GLM5_TOPK=<n> overrides attention.indexer.top_k (testing the sparse path on a short context).

static uint32_t glm5_indexer_top_k(const llama_hparams & hparams) {
    static const int override_top_k = [] {
        const char * e = getenv("ALPHA_GLM5_TOPK");
        return e ? atoi(e) : 0;
    }();
    if (override_top_k > 0) {
        return (uint32_t) GGML_PAD(override_top_k, (int) hparams.indexer_kpool);
    }
    return hparams.indexer_top_k;
}

static bool glm5_hc_fused() {
    static const bool on = [] {
        const char * e = getenv("ALPHA_GLM5_HC_FUSED");
        return e ? atoi(e) != 0 : false;
    }();
    return on;
}

// x / sqrt(sum(x^2) + eps), the FLA l2 norm (upstream build_gdn_l2_norm)
static ggml_tensor * glm5_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

// Causal conv1d over one of Q/K/V (qkv = 0, 1, 2); same state layout as kimi-linear
static ggml_tensor * glm5_conv1d(ggml_cgraph * gf, ggml_context * ctx0,
                                 ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
                                 int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
                                 int64_t d_conv, int64_t head_dim, int64_t n_head,
                                 int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size;

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1)   * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);
    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // save the last (d_conv - 1) columns back to the conv state
    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all, d_conv - 1, d_inner, n_seqs,
                (d_conv - 1)   * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);
    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

//
// k-pool indexer inputs
//

class llm_build_glm5_next::llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(const llama_memory_hybrid_idx_context * mctx, uint32_t top_k_pools) :
        mctx(mctx), top_k_pools(top_k_pools) {}
    virtual ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        if (sparse) {
            mctx->set_input_kpool(pool_gcells, pool_idxs, pool_mask, tail_idxs, ubatch);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        bool res = true;

        res &= k_idxs->ne[0] == params.ubatch.n_tokens;
        res &= n_pool        == mctx->get_n_kpool();
        res &= n_kv          == idx->get_n_kv();

        return res;
    }

    ggml_tensor * k_idxs      = nullptr; // I64 [n_tokens]             indexer cache rows written by this ubatch
    // only when sparse:
    ggml_tensor * pool_gcells = nullptr; // I32 [kpool, n_pool]        member cells as rows of the cache storage
    ggml_tensor * pool_idxs   = nullptr; // I32 [kpool, n_pool]        member cells as mask rows, n_kv for the padded pools
    ggml_tensor * pool_mask   = nullptr; // F32 [n_pool, n_tokens]
    ggml_tensor * tail_idxs   = nullptr; // I32 [kpool - 1, n_tokens]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t top_k_pools;

    uint32_t n_pool = 0;
    uint32_t n_kv   = 0;
    bool     sparse = false; // more pools than the top-k budget: the indexer selects
};

llm_build_glm5_next::llm_graph_input_kpool * llm_build_glm5_next::build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb) {
    const auto * mctx_idx = mctx_hyb->get_idx();
    GGML_ASSERT(mctx_idx != nullptr);

    const uint32_t kpool       = hparams.indexer_kpool;
    const uint32_t top_k_pools = glm5_indexer_top_k(hparams) / kpool;

    auto inp = std::make_unique<llm_graph_input_kpool>(mctx_hyb, top_k_pools);

    inp->n_pool = mctx_hyb->get_n_kpool();
    inp->n_kv   = mctx_idx->get_n_kv();

    // While every pool fits in the budget, top-k keeps all of them and the selection (complete pools
    // plus the token's own incomplete pool, under the causal mask) is the whole causal context.
    inp->sparse = inp->n_pool > top_k_pools;

    inp->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);

    if (inp->sparse) {
        inp->pool_gcells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_pool);
        inp->pool_idxs   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool, inp->n_pool);
        inp->pool_mask   = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, inp->n_pool, n_tokens);
        inp->tail_idxs   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, kpool - 1, n_tokens);
        ggml_set_input(inp->pool_gcells);
        ggml_set_input(inp->pool_idxs);
        ggml_set_input(inp->pool_mask);
        ggml_set_input(inp->tail_idxs);

        // keep them allocated even when no op reads them: set_input_kpool always fills them
        ggml_build_forward_expand(gf, inp->pool_gcells);
        ggml_build_forward_expand(gf, inp->pool_idxs);
        ggml_build_forward_expand(gf, inp->pool_mask);
        ggml_build_forward_expand(gf, inp->tail_idxs);
    }

    return (llm_graph_input_kpool *) res->add_input(std::move(inp));
}

//
// mHC
//

static ggml_tensor * glm5_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

static ggml_tensor * glm5_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * glm5_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llm_build_glm5_next::build_hc_mean(ggml_tensor * x) const {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx0, acc, 1.0f/hc);
}

ggml_tensor * llm_build_glm5_next::build_hc_sinkhorn(ggml_tensor * comb) const {
    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llm_build_glm5_next::build_hc_pre(
        ggml_tensor *  x,
        ggml_tensor *  hc_fn,
        ggml_tensor *  hc_scale,
        ggml_tensor *  hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.hc_count;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(x->ne[0] == n_embd && x->ne[1] == hc);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, hparams.f_norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    if (glm5_hc_fused()) {
        // this fork's DeepSeek-V4 mixers: pre | post | comb rows of one tensor
        ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx0, mixes, hc_scale, hc_base, (int) hc,
                (int) hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * pre = ggml_view_2d(ctx0, split, hc, nt, split->nb[1], 0);
        *post = ggml_view_2d(ctx0, split, hc,    nt, split->nb[1],   hc * split->nb[0]);
        *comb = ggml_view_2d(ctx0, split, hc*hc, nt, split->nb[1], 2*hc * split->nb[0]);
        if (nt != 1) {
            pre   = ggml_cont(ctx0, pre);
            *post = ggml_cont(ctx0, *post);
            *comb = ggml_cont(ctx0, *comb);
        }
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        return ggml_dsv4_hc_weighted_sum(ctx0, x, pre);
    }

    ggml_tensor * scale_pre  = glm5_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = glm5_view_1d(ctx0, hc_scale, 1, 1);
    ggml_tensor * scale_comb = glm5_view_1d(ctx0, hc_scale, 1, 2);

    ggml_tensor * base_pre  = glm5_view_1d(ctx0, hc_base, hc,    0);
    ggml_tensor * base_post = glm5_view_1d(ctx0, hc_base, hc,    hc);
    ggml_tensor * base_comb = glm5_view_1d(ctx0, hc_base, hc*hc, 2*hc);

    ggml_tensor * pre = glm5_view_2d(ctx0, mixes, hc, nt, 0);
    pre = glm5_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.hc_eps);
    cb(pre, "hc_pre", il);

    *post = glm5_view_2d(ctx0, mixes, hc, nt, hc);
    *post = glm5_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    *comb = glm5_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
    *comb = glm5_hc_affine(ctx0, *comb, scale_comb, base_comb);
    *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
    *comb = build_hc_sinkhorn(*comb);
    cb(*comb, "hc_comb", il);

    // collapse the streams with the per-stream weights
    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh  = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh  = ggml_view_2d(ctx0, pre, 1, nt, pre->nb[1], ih*pre->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llm_build_glm5_next::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_UNUSED(il);
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.hc_count);

    if (glm5_hc_fused()) {
        return ggml_dsv4_hc_expand(ctx0, x, residual, post, comb);
    }

    const int64_t hc = hparams.hc_count;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

//
// graph
//

llm_build_glm5_next::llm_build_glm5_next(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    cb(inp, "inp_embd", -1);

    // recurrent state + K-only MLA cache through the generic hybrid input, plus the indexer cache
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

    auto * inp_hyb   = build_inp_mem_hybrid_k();
    auto * inp_rs    = inp_hyb->get_recr();
    auto * inp_attn  = inp_hyb->get_attn();
    auto * inp_kpool = build_inp_kpool(mctx_hyb);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // the trailing NextN (MTP) block is not part of the trunk
    const int n_trunk = (int) n_layer - (int) hparams.nextn_predict_layers;

    const int64_t hc = hparams.hc_count;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_trunk; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            cur = build_kda_layer(cur, layer, inp_rs, n_seq_tokens, n_seqs, il);
        } else {
            cur = build_dsa_layer(cur, layer, mctx_hyb, inp_attn, inp_kpool, il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up,   nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // the shared expert is built first and placed between the router and the routed experts in the graph
            // (ALPHA_GLM5_SHEXP_FIRST=0: after them, as upstream orders it); same nodes, same sum
            static const bool shexp_first = [] { const char * e = getenv("ALPHA_GLM5_SHEXP_FIRST"); return !e || atoi(e) != 0; }();
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);
            if (shexp_first) moe_expand_first = ffn_shexp;

            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    hparams.n_expert, hparams.n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);
            moe_expand_first = nullptr;

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // ALPHA_MTP with the NextN block loaded: the block drafts from the trunk's hidden after output_norm (upstream's
    // h_nextn), and the head's latent cache needs that hidden for every row of the ubatch, so the narrowing to the
    // output rows moves behind the norm (llama_context reads res->t_mtp_hidden; llama-context-mtp5.cpp)
    static const bool mtp_env = [] { const char * e = getenv("ALPHA_MTP"); return e && atoi(e) != 0; }();
    const bool mtp_defer = mtp_env && hparams.nextn_predict_layers == 1 && model.layers[n_trunk].nextn.eh_proj != nullptr;

    // narrow to the output tokens, then collapse the streams
    if (inp_out_ids && !mtp_defer) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, ggml_cont(ctx0, inpL), n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, flat->ne[1]);
    }

    cur = build_hc_mean(inpL);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    if (mtp_defer) {
        ggml_tensor * mtp_h = ggml_cont(ctx0, cur);   // [n_embd, n_tokens]
        cb(mtp_h, "mtp_hidden", -1);
        ggml_set_output(mtp_h);
        res->t_mtp_hidden = mtp_h;
        ggml_build_forward_expand(gf, mtp_h);

        if (inp_out_ids) {
            cur = ggml_get_rows(ctx0, mtp_h, inp_out_ids);
        }
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // MTP: the verify reads one token per row from an in-graph argmax instead of scanning full-vocab logits
    if (mtp_defer) {
        ggml_tensor * am = ggml_argmax(ctx0, cur);
        cb(am, "mtp_argmax", -1);
        ggml_set_output(am);
        res->t_mtp_argmax = am;
        ggml_build_forward_expand(gf, am);
    }
}

// KDA layer, g_a/g_b output gate

ggml_tensor * llm_build_glm5_next::build_kda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs,
        int64_t n_seq_tokens, int64_t n_seqs, int il) {
    const int64_t n_head_kda = hparams.n_head();
    const int64_t head_dim   = hparams.n_embd_head_kda;
    const int64_t d_conv     = hparams.ssm_d_conv;
    const int64_t d_inner    = n_head_kda * head_dim;

    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

    ggml_tensor * Qcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Kcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    ggml_tensor * Vcur = glm5_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head_kda, n_seq_tokens, n_seqs, n_tokens, kv_head);
    cb(Qcur, "kda_q_conv", il);
    cb(Kcur, "kda_k_conv", il);
    cb(Vcur, "kda_v_conv", il);

    // Decay gate, ssm_a holds -exp(A_log)
    ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
    ggml_tensor * g1  = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
    g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);

    ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head_kda, 1);

    if (hparams.kda_gate_lower_bound > -INFINITY) {
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
        g1 = ggml_sigmoid(ctx0, ggml_scale(ctx0, g1, -1.0f));
        g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
    } else {
        g1 = ggml_softplus(ctx0, g1);
        g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head_kda, n_tokens);
        g1 = ggml_mul(ctx0, g1, A);
    }
    cb(g1, "kda_g1", il);

    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head_kda, n_seq_tokens, n_seqs);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head_kda, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head_kda, n_seqs);

    // Match FLA l2 norm
    constexpr float l2_eps = 1e-6f;
    Qcur = glm5_l2_norm(ctx0, Qcur, l2_eps);
    Kcur = glm5_l2_norm(ctx0, Kcur, l2_eps);

    auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

    ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
    ggml_tensor * new_state = attn_out.second;
    cb(output, "kda_scan_out", il);
    cb(new_state, "new_state", il);

    // update the recurrent state
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // output gate, then RMSNorm(o) * Sigmoid(g2)
    ggml_tensor * g_a = ggml_mul_mat(ctx0, layer.ssm_g_a, cur);
    ggml_tensor * g2  = ggml_mul_mat(ctx0, layer.ssm_g_b, g_a);
    g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head_kda, n_tokens);

    ggml_tensor * o      = ggml_reshape_3d(ctx0, output, head_dim, n_head_kda, n_tokens);
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    cb(g2, "kda_g2", il);
    cb(normed, "kda_normed", il);
    ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

    gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
    cur   = ggml_mul_mat(ctx0, layer.wo, gated);
    cb(cur, "kda_out", il);

    return cur;
}

// Scores pools of kpool consecutive tokens, expands the selected pools and the incomplete tail into an
// additive mask over the cache cells, with the causal mask folded in.

ggml_tensor * llm_build_glm5_next::build_kpool_select(
        ggml_tensor * cur, ggml_tensor * qr, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kpool, int il) {
    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;
    const int64_t n_pool         = inp_kpool->n_pool;
    const int64_t n_kv           = inp_kpool->n_kv;

    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_3d(ctx0, iq, n_embd_indexer, n_indexer_head, n_tokens);
    cb(iq, "indexer_q", il);

    // Pool every complete pool from the cached key | gate rows (this ubatch's rows are already written)
    ggml_tensor * kstore = mctx_hyb->get_kpool_storage(il);
    GGML_ASSERT(kstore->ne[0] == 2*n_embd_indexer);

    ggml_tensor * key_gate = ggml_view_2d(ctx0, kstore, 2*n_embd_indexer, kstore->ne[1]*kstore->ne[2], kstore->nb[1], 0);

    ggml_tensor * rows = ggml_get_rows(ctx0, key_gate, ggml_reshape_1d(ctx0, inp_kpool->pool_gcells, kpool*n_pool));
    rows = ggml_reshape_3d(ctx0, rows, 2*n_embd_indexer, kpool, n_pool);

    ggml_tensor * pk = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_pool, rows->nb[1], rows->nb[2], 0);
    ggml_tensor * pg = ggml_view_3d(ctx0, rows, n_embd_indexer, kpool, n_pool, rows->nb[1], rows->nb[2], ggml_row_size(rows->type, n_embd_indexer));

    ggml_tensor * logits = ggml_add(ctx0, pg, layer.indexer_kpool_ape);
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 1, 0, 2, 3)); // [kpool, head_dim, n_pool]
    // fold the pool axis into rows, soft_max maps ne[2] to CUDA gridDim.y which caps at 65535
    ggml_tensor * probs = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, logits, kpool, n_embd_indexer*n_pool));
    probs = ggml_reshape_3d(ctx0, probs, kpool, n_embd_indexer, n_pool);

    pk = ggml_cont(ctx0, ggml_permute(ctx0, pk, 1, 0, 2, 3));
    ggml_tensor * pooled = ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, pk)); // [1, head_dim, n_pool]
    pooled = ggml_reshape_3d(ctx0, pooled, n_embd_indexer, 1, n_pool);
    cb(pooled, "indexer_pool_k", il);

    ggml_tensor * weights = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_indexer * n_indexer_head)));
    cb(weights, "indexer_weights", il);

    ggml_tensor * score = nullptr;
    {
        ggml_tensor * q_p = ggml_permute(ctx0, iq, 0, 2, 1, 3);     // [head_dim, n_tokens, n_head]
        ggml_tensor * k_p = ggml_permute(ctx0, pooled, 0, 2, 1, 3); // [head_dim, n_pool, 1]

        ggml_tensor * kq = ggml_mul_mat(ctx0, k_p, q_p);            // [n_pool, n_tokens, n_head]
        kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));   // [n_head, n_tokens, n_pool]
        score = ggml_relu(ctx0, kq);
        score = ggml_mul(ctx0, score, weights);
        score = ggml_sum_rows(ctx0, score);                          // [1, n_tokens, n_pool]
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)); // [n_pool, n_tokens, 1]
        score = ggml_add(ctx0, score, inp_kpool->pool_mask);
    }
    cb(score, "indexer_score", il);

    const int64_t n_top_pool = std::min<int64_t>(n_pool, inp_kpool->top_k_pools);
    ggml_tensor * top_k = ggml_top_k(ctx0, score, n_top_pool); // [n_top_pool, n_tokens], UNORDERED
    cb(top_k, "indexer_top_k", il);

    ggml_tensor * sel_idx = ggml_get_rows(ctx0, inp_kpool->pool_idxs,
            ggml_reshape_1d(ctx0, top_k, n_top_pool*n_tokens));  // [kpool, n_top_pool*n_tokens]
    sel_idx = ggml_reshape_2d(ctx0, sel_idx, kpool*n_top_pool, n_tokens);

    if (hparams.indexer_kpool_select_tail) {
        // Append the incomplete tail with n_kv for missing cells.
        sel_idx = ggml_concat(ctx0, sel_idx, inp_kpool->tail_idxs, 0);
    }
    const int64_t n_sel = sel_idx->ne[0];

    // Scatter zeros over a -inf mask; row n_kv takes the sentinel indices.
    // Tie the scatter storage lifetime to this layer's selected indices.
    ggml_tensor * seed = ggml_cast(ctx0, ggml_view_1d(ctx0, sel_idx, 1, 0), GGML_TYPE_F32);

    ggml_tensor * mask_seed = ggml_fill(ctx0, seed, -INFINITY);
    ggml_tensor * mask_all  = ggml_repeat_4d(ctx0, mask_seed, 1, n_kv + 1, n_tokens, 1);
    mask_all = ggml_reshape_3d(ctx0, mask_all, 1, n_kv + 1, n_tokens);

    ggml_tensor * zero_seed = ggml_fill(ctx0, seed, 0.0f);
    ggml_tensor * zeros = ggml_repeat_4d(ctx0, zero_seed, 1, n_sel, n_tokens, 1);
    zeros = ggml_reshape_3d(ctx0, zeros, 1, n_sel, n_tokens);

    ggml_tensor * sel = ggml_set_rows(ctx0, mask_all, zeros, ggml_reshape_3d(ctx0, sel_idx, n_sel, n_tokens, 1));
    sel = ggml_view_2d(ctx0, sel, n_kv, n_tokens, sel->nb[2], 0);

    // Fold in the causal visibility (the F32 mask, before any conversion for flash attention)
    ggml_tensor * kq_mask = inp_attn->self_kq_mask;
    if (kq_mask->type != GGML_TYPE_F32) {
        // this fork's serve builds the mask input as F16 (ALPHA_KQ_MASK_F16); the scatter mask is F32
        kq_mask = ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
    }
    GGML_ASSERT(kq_mask->ne[0] == n_kv && kq_mask->ne[1]*kq_mask->ne[2]*kq_mask->ne[3] == n_tokens);
    sel = ggml_add(ctx0, sel, ggml_reshape_2d(ctx0, kq_mask, n_kv, n_tokens));
    sel = ggml_reshape_4d(ctx0, sel, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3]);

    ggml_tensor * kq_mask_cnv = inp_attn->get_kq_mask();
    if (kq_mask_cnv->type != sel->type) {
        sel = ggml_cast(ctx0, sel, kq_mask_cnv->type);
    }
    cb(sel, "indexer_sel", il);

    return sel;
}

// Nope MLA layer with sparse attention over the indexer selection

ggml_tensor * llm_build_glm5_next::build_dsa_layer(
        ggml_tensor * cur, const llama_layer & layer,
        const llama_memory_hybrid_idx_context * mctx_hyb, llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kpool, int il) {
    const auto * mctx_mla = mctx_hyb->get_attn();
    const auto * mctx_lid = mctx_hyb->get_idx();

    const int64_t n_head_mla          = hparams.n_head(il);
    const int64_t n_embd_head_k_mla   = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank        = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - hparams.n_rot();
    const int64_t n_embd_indexer      = hparams.indexer_head_size;
    const float   kq_scale = 1.0f / sqrtf((float) n_embd_head_k_mla);

    GGML_ASSERT(hparams.n_rot() == 0 && "GLM5-Next MLA is nope-only");

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "q_resid", il);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head_qk_nope, n_head_mla, n_tokens);

    ggml_tensor * kv_cmpr = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
    cb(kv_cmpr, "kv_cmpr", il);

    // absorb wk_b so the cache holds only the latent
    ggml_tensor * q_absorbed = ggml_permute(ctx0, q, 0, 2, 1, 3);
    q_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_absorbed);
    q_absorbed = ggml_permute(ctx0, q_absorbed, 0, 2, 1, 3);
    // contiguous, so that build_attn_mha's per-stream view of q is valid for any stream count
    q_absorbed = ggml_cont(ctx0, q_absorbed);
    cb(q_absorbed, "q_absorbed", il);

    // Per-token indexer key and pool gate scores, cached together as key | gate.
    // Written on every ubatch, also while the selection itself is skipped.
    {
        ggml_tensor * ik = ggml_mul_mat(ctx0, layer.indexer_attn_k, cur);
        ik = build_norm(ik, layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
        cb(ik, "indexer_k", il);

        ggml_tensor * ig = ggml_mul_mat(ctx0, layer.indexer_kpool_gate, cur);
        cb(ig, "indexer_gate", il);

        ggml_tensor * packed = ggml_concat(ctx0, ik, ig, 0);
        packed = ggml_reshape_3d(ctx0, packed, 2*n_embd_indexer, 1, n_tokens);
        ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, packed, inp_kpool->k_idxs, il));
    }

    ggml_tensor * mask = nullptr;
    if (inp_kpool->sparse) {
        mask = build_kpool_select(cur, qr, layer, mctx_hyb, inp_attn, inp_kpool, il);
        cb(mask, "kq_mask_dsa", il);
    } else {
        mask = inp_attn->get_kq_mask();
    }

    ggml_build_forward_expand(gf, q_absorbed);
    ggml_build_forward_expand(gf, kv_cmpr);
    ggml_build_forward_expand(gf, mctx_mla->cpy_k(ctx0, kv_cmpr, inp_attn->get_k_idxs(), il));

    ggml_tensor * k = mctx_mla->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, kv_lora_rank, k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * out = build_attn_mha(q_absorbed, k, v, nullptr, mask, nullptr, layer.wv_b, kq_scale, il);
    cb(out, "kqv_out", il);

    out = ggml_mul_mat(ctx0, layer.wo, out);
    cb(out, "attn_out", il);

    return out;
}
