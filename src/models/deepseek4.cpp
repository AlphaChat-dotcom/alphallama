#include "models.h"
#include <cmath>
#include <cstring>

// Self-speculative draft mode
extern thread_local bool g_alpha_draft_mode;
extern thread_local int  g_alpha_draft_expert_count;  // 0 = Mode C (backbone), >0 = Mode A (reduced)

// Top-k expert reduction: override n_expert_used in graph builder
// 0 = use model default, >0 = use this value (must be <= hparams.n_expert_used)
extern thread_local int g_alpha_topk_override;

// DeepSeek V4 graph builder (adapted from antirez/llama.cpp-deepseek-v4-flash)
// Uses custom ggml ops: dsv4_rope_tail, dsv4_hc_split_sinkhorn,
// dsv4_hc_weighted_sum, dsv4_hc_expand.
// Minimal version: HC + attention + grouped output. No compression/indexer.

namespace {

// Graph input for constant group IDs [0,1,...,n_groups-1] repeated per token.
// Keeps the tensor on GPU to avoid CPU<->GPU split at every layer.
class llm_graph_input_group_ids : public llm_graph_input_i {
public:
    llm_graph_input_group_ids(int64_t n_groups) : n_groups(n_groups) {}
    void set_input(const llama_ubatch * ubatch) override {
        if (!ids) return;
        const int64_t n_tokens = ids->ne[1];
        std::vector<int32_t> data(n_groups * n_tokens);
        for (int64_t t = 0; t < n_tokens; t++) {
            for (int64_t g = 0; g < n_groups; g++) {
                data[t * n_groups + g] = (int32_t)g;
            }
        }
        ggml_backend_tensor_set(ids, data.data(), 0, data.size() * sizeof(int32_t));
    }
    ggml_tensor * ids = nullptr;
    const int64_t n_groups;
};

struct dsv4_hc_mix {
    ggml_tensor * x;
    ggml_tensor * mixes;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

static ggml_tensor * dsv4_view_scale(ggml_context * ctx, ggml_tensor * scale, int64_t idx) {
    ggml_tensor * v = ggml_view_1d(ctx, scale, 1, idx * ggml_element_size(scale));
    return ggml_reshape_2d(ctx, v, 1, 1);
}

static ggml_tensor * dsv4_add_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_view_base(ggml_context * ctx, ggml_tensor * base, int64_t n, int64_t off) {
    // Use view_1d + reshape instead of view_2d to avoid stride issues
    ggml_tensor * v = ggml_view_1d(ctx, base, n, off * ggml_element_size(base));
    return ggml_reshape_2d(ctx, v, n, 1);
}

static ggml_tensor * dsv4_apply_rope_tail(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * inp_pos,
        int64_t n_embd_head, int64_t n_head, int64_t n_tokens, int64_t n_rot,
        int rope_type, int32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        bool inverse) {
    GGML_ASSERT(x->ne[0] == n_embd_head);
    if (n_rot == n_embd_head) {
        return inverse
            ? ggml_rope_ext_back(ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext     (ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }
    // Split nope prefix + rope tail, apply standard rope_ext to tail, concat.
    // Uses ops with proven CUDA support instead of custom dsv4_rope_tail kernel.
    const int64_t n_nope = n_embd_head - n_rot;
    ggml_tensor * x_nope = ggml_view_3d(ctx, x, n_nope, n_head, n_tokens,
                                        x->nb[1], x->nb[2], 0);
    ggml_tensor * x_tail = ggml_view_3d(ctx, x, n_rot, n_head, n_tokens,
                                        x->nb[1], x->nb[2], ggml_row_size(x->type, n_nope));
    x_tail = ggml_cont(ctx, x_tail);
    x_tail = inverse
        ? ggml_rope_ext_back(ctx, x_tail, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
        : ggml_rope_ext     (ctx, x_tail, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return ggml_concat(ctx, x_nope, x_tail, 0);
}

static dsv4_hc_mix dsv4_hc_pre(
        ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens, float norm_eps,
        int sinkhorn_iters, float hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat);
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, (int)n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre  = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) { pre = ggml_cont(ctx, pre); post = ggml_cont(ctx, post); comb = ggml_cont(ctx, comb); }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens);
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return { y, mixes, pre, post, comb };
}

static ggml_tensor * dsv4_hc_post(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * residual,
        ggml_tensor * post, ggml_tensor * comb,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens) {
    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

static ggml_tensor * dsv4_hc_head(
        ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * hc_fn, ggml_tensor * hc_scale, ggml_tensor * hc_base,
        int64_t n_embd, int64_t n_hc, int64_t n_tokens, float norm_eps, float hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat);
    pre = ggml_mul(ctx, pre, dsv4_view_scale(ctx, hc_scale, 0));
    pre = ggml_add(ctx, pre, dsv4_view_base(ctx, hc_base, n_hc, 0));
    pre = dsv4_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);
    return ggml_dsv4_hc_weighted_sum(ctx, x, pre);
}

static ggml_tensor * dsv4_grouped_out(
        ggml_context * ctx, ggml_tensor * o,
        ggml_tensor * wo_a, ggml_tensor * wo_b,
        ggml_tensor * group_ids,
        int64_t n_embd_head, int64_t n_head, int64_t n_groups,
        int64_t o_lora_rank, int64_t n_tokens) {
    const int64_t group_heads = n_head / n_groups;
    const int64_t group_dim = n_embd_head * group_heads;
    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);
    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);
    // group_ids is pre-computed as a graph input on GPU - avoids CPU REPEAT split
    ggml_tensor * ids = ggml_reshape_2d(ctx, group_ids, n_groups, n_tokens);
    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids);
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);
    return ggml_mul_mat(ctx, wo_b, low);
}

} // namespace


llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = n_embd_head_k;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const uint32_t kv_lora_rank = hparams.n_lora_kv;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head_k));
    const float norm_eps = hparams.f_norm_rms_eps;

    // DeepSeek V4 uses plain rope (no YaRN) for the dense in-window attention path.
    // The reference (dsv4_make_rope_cfg, compress_ratio==0) forces base=rope_freq_base_train,
    // scale=1, ext=0, attn=1. Using the global cparams YaRN params (freq_scale~1/factor, ext!=0)
    // corrupts the Q/K rotation angles on every layer and output degenerates into repetition.
    const int32_t rope_n_ctx_orig_ds = 0;
    const float   rope_freq_base_ds  = hparams.rope_freq_base_train;
    const float   rope_freq_scale_ds = 1.0f;
    const float   rope_ext_factor_ds = 0.0f;
    const float   rope_attn_factor_ds = 1.0f;

    // HC parameters
    const int64_t n_hc = (model.layers[0].hc_attn_fn)
        ? model.layers[0].hc_attn_fn->ne[0] / hparams.n_embd : 1;
    const int sinkhorn_iters = 20;
    const float hc_eps = 1e-6f;

    (void)n_embd_head_v; // used implicitly via n_embd_head_k

    // Top-k expert reduction: per-layer adaptive K when override is active.
    // Critical layers (first/last few, see n_critical) keep the full expert count; middle layers use reduced k.
    // This preserves input processing and output quality while saving compute in the middle.
    const int64_t full_expert_used = this->n_expert_used;
    const int64_t base_topk = (g_alpha_topk_override > 0 && g_alpha_topk_override < (int)hparams.n_expert_used)
        ? g_alpha_topk_override : full_expert_used;
    // n_expert_used is now computed per-layer inside the loop (see get_layer_k lambda below)

    // Grouped output
    const int64_t n_out_groups = (model.layers[0].wo_a)
        ? n_head / (model.layers[0].wo_a->ne[0] / n_embd_head_k) : 1;
    const int64_t o_lora_rank = (model.layers[0].wo_a && n_out_groups > 0)
        ? model.layers[0].wo_a->ne[1] / n_out_groups : 0;

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_embd_saved = inpL;  // ALPHA_MTP: save for MTP module
    ggml_tensor * inp_tokens = res->get_inp_tokens();  // I32 token ids, for hash routing (tid2eid)
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn_kv = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Pre-compute grouped output IDs as a graph input (stays on GPU, avoids CPU split)
    ggml_tensor * group_ids = nullptr;
    if (n_out_groups > 1 && o_lora_rank > 0) {
        auto inp = std::make_unique<llm_graph_input_group_ids>(n_out_groups);
        group_ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_out_groups, n_tokens);
        ggml_set_input(group_ids);
        ggml_set_name(group_ids, "dsv4_group_ids");
        inp->ids = group_ids;
        res->add_input(std::move(inp));
    }

    // Expand to n_hc streams
    if (n_hc > 1) {
        inpL = ggml_reshape_3d(ctx0, inpL, hparams.n_embd, 1, n_tokens);
        inpL = ggml_repeat_4d(ctx0, inpL, hparams.n_embd, n_hc, n_tokens, 1);
    }

    int effective_n_layers = hparams.n_layer - hparams.nextn_predict_layers;
#ifdef DSV4_DEBUG_LAYERS
    effective_n_layers = std::min(effective_n_layers, DSV4_DEBUG_LAYERS);
    fprintf(stderr, "V4_DBG: limiting to %d layers (debug mode)\n", effective_n_layers);
#endif
    for (int il = 0; il < effective_n_layers; ++il) {
        ggml_tensor * residual = inpL;

        // Per-layer RoPE (reference dsv4_make_rope_cfg): uncompressed layers (compress_ratio==0)
        // use plain rope base=rope_freq_base_train; compressed layers use compress_rope_freq_base
        // (160000) with the model's YaRN (freq_scale/ext_factor). compress_ratios=[0,0,4,128,...]
        // -> il<2 uncompressed, il>=2 compressed.
        // TODO(generic): load hparams.attn_compress_ratio[] + compress_rope_freq_base from the GGUF
        // instead of hardcoding the pattern/base.
        const bool il_compressed = (il >= 2);
        int32_t rl_n_ctx_orig; float rl_freq_base, rl_freq_scale, rl_ext_factor, rl_attn_factor;
        if (!il_compressed) {
            rl_n_ctx_orig = rope_n_ctx_orig_ds; rl_freq_base = rope_freq_base_ds;
            rl_freq_scale = rope_freq_scale_ds; rl_ext_factor = rope_ext_factor_ds;
            rl_attn_factor = rope_attn_factor_ds;
        } else {
            rl_n_ctx_orig = n_ctx_orig; rl_freq_base = 160000.0f;
            rl_freq_scale = freq_scale; rl_ext_factor = ext_factor;
            rl_attn_factor = 1.0f;
            if (ext_factor != 0.0f && freq_scale > 0.0f) {
                rl_attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
            }
        }

        // HC pre: attention
        dsv4_hc_mix mix_attn = dsv4_hc_pre(ctx0, inpL,
            model.layers[il].hc_attn_fn, model.layers[il].hc_attn_scale, model.layers[il].hc_attn_base,
            hparams.n_embd, n_hc, n_tokens, norm_eps, sinkhorn_iters, hc_eps);
        cur = mix_attn.x;

        // attention
        {
            cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
            q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
            q = ggml_rms_norm(ctx0, q, norm_eps);

            ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            if (model.layers[il].attn_kv_a_norm) {
                kv = build_norm(kv, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            }
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);

            q = dsv4_apply_rope_tail(ctx0, q, inp_pos, n_embd_head_k, n_head, n_tokens,
                n_rot, rope_type, rl_n_ctx_orig, rl_freq_base, rl_freq_scale,
                rl_ext_factor, rl_attn_factor, beta_fast, beta_slow, false);

            kv = dsv4_apply_rope_tail(ctx0, kv, inp_pos, n_embd_head_k, 1, n_tokens,
                n_rot, rope_type, rl_n_ctx_orig, rl_freq_base, rl_freq_scale,
                rl_ext_factor, rl_attn_factor, beta_fast, beta_slow, false);

            // The reference applies act_quant (FP8 QAT sim) to the kv nope dims here. It is skipped:
            // ggml_dsv4_fp8_kv_quantize has no CUDA kernel (supports_op=false), so on a GPU backbone it
            // would force a GPU->CPU->GPU round-trip every layer, and full-precision KV is a fine
            // inference approximation.
            ggml_tensor * Vcur = ggml_cont(ctx0, kv);

            cur = build_attn(inp_attn_kv, nullptr, NULL, nullptr,
                q, kv, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, kq_scale, il);

            cur = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head, n_tokens);
            cur = dsv4_apply_rope_tail(ctx0, cur, inp_pos, n_embd_head_v, n_head, n_tokens,
                n_rot, rope_type, rl_n_ctx_orig, rl_freq_base, rl_freq_scale,
                rl_ext_factor, rl_attn_factor, beta_fast, beta_slow, true);

            cur = dsv4_grouped_out(ctx0, cur, model.layers[il].wo_a, model.layers[il].wo_b,
                group_ids, n_embd_head_k, n_head, n_out_groups, o_lora_rank, n_tokens);
        }

        // HC post: attention
        inpL = dsv4_hc_post(ctx0, cur, residual, mix_attn.post, mix_attn.comb,
            hparams.n_embd, n_hc, n_tokens);
        residual = inpL;

        // HC pre: FFN
        dsv4_hc_mix mix_ffn = dsv4_hc_pre(ctx0, inpL,
            model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_scale, model.layers[il].hc_ffn_base,
            hparams.n_embd, n_hc, n_tokens, norm_eps, sinkhorn_iters, hc_eps);
        cur = mix_ffn.x;

        // FFN
        {
            cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

            if ((uint32_t) il < hparams.n_layer_dense_lead) {
                cur = build_ffn(cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            } else if (g_alpha_draft_mode && g_alpha_draft_expert_count == 0
                       && model.layers[il].ffn_up_shexp) {
                // self-speculative mode C: shared expert FFN only (backbone)
                cur = build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            } else {
                // Full model - per-layer adaptive expert count
                // Critical layers (first/last n_critical) use full k, middle layers use reduced k
                int64_t layer_k = base_topk;
                if (g_alpha_topk_override > 0 && g_alpha_topk_override < (int)hparams.n_expert_used) {
                    const int n_critical = 5;  // first/last 5 layers keep full experts
                    bool is_critical = (il < n_critical) || (il >= n_layer - n_critical);
                    layer_k = is_critical ? full_expert_used : base_topk;
                }

                // Force weight renorm when this layer uses fewer experts (sigmoid weights don't sum to 1)
                bool norm_w = hparams.expert_weights_norm ||
                    (layer_k < (int64_t)hparams.n_expert_used);
                // DeepSeek V4 hash routing: first n_hash_layers pick experts by token id
                // (ffn_gate_tid2eid[tok] -> n_expert_used ids), not learned top-k.
                ggml_tensor * hash_sel = nullptr;
                if ((uint32_t) il < hparams.n_hash_layers && model.layers[il].ffn_gate_tid2eid) {
                    hash_sel = ggml_get_rows(ctx0, model.layers[il].ffn_gate_tid2eid, inp_tokens);
                    hash_sel = ggml_cast(ctx0, hash_sel, GGML_TYPE_I32); // build_moe_ffn expects I32 ids
                }
                ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, layer_k,
                    LLM_FFN_SILU, norm_w,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il, nullptr, model.layers[il].ffn_gate_up_exps,
                    nullptr, nullptr, nullptr,
                    model.layers[il].ffn_cache_expert_ids,
                    model.layers[il].ffn_cache_expert_ids_down,
                    hash_sel);

                ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);

                // Shared expert compensation: boost shared expert when this layer drops 3+ experts
                // At dropped=2 (k=4), compensation can destabilize - only apply for larger drops
                if (layer_k <= (int64_t)hparams.n_expert_used - 3) {
                    int dropped = (int)hparams.n_expert_used - (int)layer_k;
                    float shared_scale = 1.0f + (float)dropped * 0.15f;
                    ffn_shexp = ggml_scale(ctx0, ffn_shexp, shared_scale);
                }

                // Mixed draft mode: boost shared expert (exact) to compensate for
                // approximated routed experts (averaged within-group clusters)
                extern thread_local float g_alpha_mixed_shexp_scale;
                if (g_alpha_mixed_shexp_scale != 1.0f) {
                    ffn_shexp = ggml_scale(ctx0, ffn_shexp, g_alpha_mixed_shexp_scale);
                }

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
            }
        }

        // HC post: FFN
        inpL = dsv4_hc_post(ctx0, cur, residual, mix_ffn.post, mix_ffn.comb,
            hparams.n_embd, n_hc, n_tokens);
    }

    ggml_tensor * mtp_hidden = inpL;  // ALPHA_MTP: save hidden state for MTP

    // HC head
    cur = dsv4_hc_head(ctx0, inpL,
        model.hc_head_fn, model.hc_head_scale, model.hc_head_base,
        hparams.n_embd, n_hc, n_tokens, norm_eps, hc_eps);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // ALPHA_MTP: MTP forward pass, producing draft logits for the next-next token.
    // Only during autoregressive generation (n_tokens == 1) and if MTP tensors exist.
    if (hparams.nextn_predict_layers > 0 && n_tokens == 1) {
        const int mtp_il = effective_n_layers;  // MTP layer index (e.g., 43)
        const auto & mtp_layer = model.layers[mtp_il];

        if (mtp_layer.nextn.eh_proj && mtp_layer.nextn.enorm && mtp_layer.nextn.hnorm) {
            ggml_tensor * mtp_h = mtp_hidden;
            ggml_tensor * mtp_e = inp_embd_saved;

            // If HC expanded, collapse HC streams for MTP input
            if (n_hc > 1) {
                mtp_h = dsv4_hc_head(ctx0, mtp_hidden,
                    model.hc_head_fn, model.hc_head_scale, model.hc_head_base,
                    hparams.n_embd, n_hc, n_tokens, norm_eps, hc_eps);
            }

            // RMSNorm + project embedding and hidden state
            ggml_tensor * e_norm = ggml_rms_norm(ctx0, mtp_e, norm_eps);
            e_norm = ggml_mul(ctx0, e_norm, mtp_layer.nextn.enorm);
            ggml_tensor * h_norm = ggml_rms_norm(ctx0, mtp_h, norm_eps);
            h_norm = ggml_mul(ctx0, h_norm, mtp_layer.nextn.hnorm);

            ggml_tensor * e_proj = ggml_mul_mat(ctx0, mtp_layer.nextn.eh_proj, e_norm);
            ggml_tensor * h_proj = mtp_layer.nextn.h_proj
                ? ggml_mul_mat(ctx0, mtp_layer.nextn.h_proj, h_norm)
                : ggml_mul_mat(ctx0, mtp_layer.nextn.eh_proj, h_norm);
            ggml_tensor * mtp_cur = ggml_add(ctx0, e_proj, h_proj);

            // Run through MTP transformer layer (blk.mtp_il)
            if (mtp_layer.hc_attn_fn) {
                int64_t mtp_n_hc = mtp_layer.hc_attn_fn->ne[0] / hparams.n_embd;
                if (mtp_n_hc > 1) {
                    mtp_cur = ggml_reshape_3d(ctx0, mtp_cur, hparams.n_embd, 1, n_tokens);
                    mtp_cur = ggml_repeat_4d(ctx0, mtp_cur, hparams.n_embd, mtp_n_hc, n_tokens, 1);
                }
                dsv4_hc_mix mix = dsv4_hc_pre(ctx0, mtp_cur,
                    mtp_layer.hc_attn_fn, mtp_layer.hc_attn_scale, mtp_layer.hc_attn_base,
                    hparams.n_embd, mtp_n_hc, n_tokens, norm_eps, sinkhorn_iters, hc_eps);
                ggml_tensor * mtp_attn_in = mix.x;

                // Attention: passthrough for now (needs its own KV cache)
                mtp_attn_in = build_norm(mtp_attn_in, mtp_layer.attn_norm, NULL, LLM_NORM_RMS, mtp_il);
                ggml_tensor * mtp_attn_out = mtp_attn_in;

                mtp_cur = dsv4_hc_post(ctx0, mtp_attn_out, mtp_cur, mix.post, mix.comb,
                    hparams.n_embd, mtp_n_hc, n_tokens);

                // FFN (MoE)
                dsv4_hc_mix mix_ffn = dsv4_hc_pre(ctx0, mtp_cur,
                    mtp_layer.hc_ffn_fn, mtp_layer.hc_ffn_scale, mtp_layer.hc_ffn_base,
                    hparams.n_embd, mtp_n_hc, n_tokens, norm_eps, sinkhorn_iters, hc_eps);
                ggml_tensor * mtp_ffn_in = mix_ffn.x;
                mtp_ffn_in = build_norm(mtp_ffn_in, mtp_layer.ffn_norm, NULL, LLM_NORM_RMS, mtp_il);

                ggml_tensor * mtp_moe = build_moe_ffn(mtp_ffn_in,
                    mtp_layer.ffn_gate_inp, mtp_layer.ffn_up_exps,
                    mtp_layer.ffn_gate_exps, mtp_layer.ffn_down_exps,
                    mtp_layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    mtp_il, nullptr, mtp_layer.ffn_gate_up_exps,
                    nullptr, nullptr, nullptr,
                    mtp_layer.ffn_cache_expert_ids,
                    mtp_layer.ffn_cache_expert_ids_down);
                ggml_tensor * mtp_shexp = build_ffn(mtp_ffn_in,
                    mtp_layer.ffn_up_shexp, NULL, NULL,
                    mtp_layer.ffn_gate_shexp, NULL, NULL,
                    mtp_layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, mtp_il);
                ggml_tensor * mtp_ffn_out = ggml_add(ctx0, mtp_moe, mtp_shexp);

                mtp_cur = dsv4_hc_post(ctx0, mtp_ffn_out, mtp_cur, mix_ffn.post, mix_ffn.comb,
                    hparams.n_embd, mtp_n_hc, n_tokens);

                if (mtp_n_hc > 1) {
                    mtp_cur = ggml_reshape_3d(ctx0, mtp_cur, hparams.n_embd, mtp_n_hc, n_tokens);
                    mtp_cur = ggml_view_3d(ctx0, mtp_cur, hparams.n_embd, 1, n_tokens,
                        mtp_cur->nb[1], mtp_cur->nb[2], 0);
                    mtp_cur = ggml_reshape_2d(ctx0, mtp_cur, hparams.n_embd, n_tokens);
                }
            } else {
                // No HC - simple path
                mtp_cur = build_norm(mtp_cur, mtp_layer.attn_norm, NULL, LLM_NORM_RMS, mtp_il);
                mtp_cur = build_norm(mtp_cur, mtp_layer.ffn_norm, NULL, LLM_NORM_RMS, mtp_il);
                ggml_tensor * mtp_moe = build_moe_ffn(mtp_cur,
                    mtp_layer.ffn_gate_inp, mtp_layer.ffn_up_exps,
                    mtp_layer.ffn_gate_exps, mtp_layer.ffn_down_exps,
                    mtp_layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    mtp_il, nullptr, mtp_layer.ffn_gate_up_exps,
                    nullptr, nullptr, nullptr,
                    mtp_layer.ffn_cache_expert_ids,
                    mtp_layer.ffn_cache_expert_ids_down);
                ggml_tensor * mtp_shexp = build_ffn(mtp_cur,
                    mtp_layer.ffn_up_shexp, NULL, NULL,
                    mtp_layer.ffn_gate_shexp, NULL, NULL,
                    mtp_layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, mtp_il);
                mtp_cur = ggml_add(ctx0, mtp_moe, mtp_shexp);
            }

            // Output norm + lm_head -> draft logits
            ggml_tensor * mtp_norm;
            if (mtp_layer.nextn.shared_head_norm) {
                mtp_norm = ggml_rms_norm(ctx0, mtp_cur, norm_eps);
                mtp_norm = ggml_mul(ctx0, mtp_norm, mtp_layer.nextn.shared_head_norm);
            } else {
                mtp_norm = build_norm(mtp_cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
            }

            ggml_tensor * mtp_logits = ggml_mul_mat(ctx0, model.output, mtp_norm);
            ggml_set_name(mtp_logits, "mtp_logits");
            ggml_set_output(mtp_logits);
            res->t_mtp_logits = mtp_logits;
            ggml_build_forward_expand(gf, mtp_logits);
        }
    }
}
