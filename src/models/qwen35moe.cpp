#include "models.h"

#include <mutex>
#include <set>
#include <string>

#include "llama-memory-recurrent.h"
#include "alpha_gdn_record.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

// ALPHA_MTP is on only for a non-zero value, matching the serve engine (serve-gpu-engine.cpp g_mtp), so ALPHA_MTP=0 does not load the head or build the MTP graph.
static inline bool alpha_mtp_env_on() { const char * e = getenv("ALPHA_MTP"); return e && atoi(e) != 0; }

// Top-k expert reduction (shared with deepseek4.cpp, alpha_moe_optimizer.cpp)
extern thread_local int g_alpha_topk_override;

// Multi-pass pipeline layer range (for self-spec layer-skip)
extern thread_local int g_alpha_pipeline_layer_start;
extern thread_local int g_alpha_pipeline_layer_end;
extern thread_local int g_alpha_pipeline_skip_start;
extern thread_local int g_alpha_pipeline_skip_end;

// Fused projection GEMV (ALPHA_FUSE_PROJ).
// The DeltaNet layers issue 4 separate GEMVs (wqkv, wqkv_gate, ssm_beta, ssm_alpha) on the same
// input, and the GQA layers 3 (wq, wk, wv). Each GEMV re-quantizes the activation (quantize_q8_1)
// and pays its own launch + tail. These weights are same-typed and, because llama-model.cpp
// creates them back-to-back, allocated adjacently in the backend buffer. They are aliased as one
// wide 2D weight (zero copy, no extra VRAM: the fused tensor points at the first weight's storage)
// and run as one GEMV, with the output split by strided views. Falls back to split GEMVs whenever
// any precondition fails (adapter loras, per-tensor scales, type/shape mismatch, non-adjacent
// allocation).
// ALPHA_FUSE_PROJ: default on (every fusion), 0 disables. A value other than 0 or 1 is a mask of 2 (attention
// q|k|v), 4 (DeltaNet qkv|z[|beta|alpha]) and 8 (router|shared-expert score), for bisecting.
// At one token every slice is shaped exactly like the split GEMV output it replaces, so the tokens are
// bit-identical to the split path.
static int alpha_fuse_proj_mask() {
    static const int m = []() {
        const char * e = getenv("ALPHA_FUSE_PROJ");
        const int v = e ? atoi(e) : 1;
        return v == 1 ? 0xff : v;
    }();
    return m;
}

// ssm_beta | ssm_alpha are two f32 [n_embd, H] weights created back-to-back by the loader: one GEMV
// and two strided [1, H, T, B] views instead of two launches per DeltaNet layer (ALPHA_FUSE_BA).
// Needs the fused-input GDN op (it reads beta/alpha through strides). Default on; =0 disables.
static bool alpha_fuse_ba_enabled() {
    static const bool on = []() {
        const char * e = getenv("ALPHA_FUSE_BA");
        return e ? atoi(e) != 0 : true;
    }();
    return on;
}

static void alpha_fuse_log_kind(const char * kind) {
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(kind).second) {
        fprintf(stderr, "[alpha] fused projection GEMV: %s (ALPHA_FUSE_PROJ)\n", kind);
    }
}

// Alias N same-type weights that sit back-to-back in one backend buffer as a single
// 2D tensor [ne0, sum(ne1)]. Returns nullptr if any precondition fails.
static ggml_tensor * alpha_fused_weight(ggml_context * ctx, std::initializer_list<ggml_tensor *> ws) {
    ggml_tensor * w0 = *ws.begin();
    if (!w0 || !w0->buffer || !w0->data) {
        return nullptr;
    }
    int64_t ne1_sum = 0;
    const char * expected = (const char *) w0->data;
    for (ggml_tensor * w : ws) {
        if (!w || !w->buffer || !w->data || w->view_src) {
            return nullptr;
        }
        if (w->type != w0->type || w->ne[0] != w0->ne[0] || w->ne[2] != 1 || w->ne[3] != 1) {
            return nullptr;
        }
        if (w->buffer != w0->buffer || !ggml_is_contiguous(w)) {
            return nullptr;
        }
        // The CPU repack buffer keeps each weight's layout traits in `extra`; the alias must carry the same
        // traits, or the plain kernel reads the interleaved bytes (garbage decode on ARM, where Q5_K repacks).
        if (w->extra != w0->extra) {
            return nullptr;
        }
        if ((const char *) w->data != expected) {
            return nullptr;   // not allocated adjacently
        }
        expected += ggml_nbytes(w);
        ne1_sum  += w->ne[1];
    }
    ggml_tensor * f = ggml_new_tensor_2d(ctx, w0->type, w0->ne[0], ne1_sum);
    f->buffer = w0->buffer;
    f->data   = w0->data;
    f->extra  = w0->extra;   // repacked rows are 8-row groups in order, so adjacent weights concatenate exactly
    ggml_format_name(f, "%s (fused)", w0->name);
    return f;
}

// Largest ubatch that takes the fused projections (decode and MTP verify). Prefill keeps the split GEMVs:
// it is compute-bound there and its graph stays unchanged. ALPHA_FUSE_PROJ_MAXT overrides.
static int64_t alpha_fuse_proj_maxt() {
    static const int64_t v = []() {
        const char * e = getenv("ALPHA_FUSE_PROJ_MAXT");
        return (int64_t) (e ? atoi(e) : 16);
    }();
    return v;
}

// Rows [off, off+dim) of a fused projection output fo [rows, T]. At T == 1 the slice is a contiguous [dim, 1] tensor
// with the strides of the split GEMV output it replaces, so every later op and fusion matcher sees the same input
// (the fused path then gives bit-identical tokens). At T > 1 it is a strided [dim, T] row view.
static ggml_tensor * alpha_proj_slice(ggml_context * ctx, ggml_tensor * fo, int64_t off, int64_t dim) {
    const size_t es = ggml_element_size(fo);
    if (fo->ne[1] == 1) {
        return ggml_reshape_2d(ctx, ggml_view_1d(ctx, fo, dim, off * es), dim, 1);
    }
    return ggml_view_2d(ctx, fo, dim, fo->ne[1], fo->nb[1], off * es);
}


llm_build_qwen35moe::llm_build_qwen35moe(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    // save the input-token embedding for the MTP draft head (built after the logits)
    ggml_tensor * inp_embd_saved = inpL;

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Skip NextN prediction layers (Qwen3.6 has 1 MTP layer at the end)
    const int n_layer_real = n_layer - hparams.nextn_predict_layers;

    // With the MTP draft head active the final hidden is needed at every position, not just the
    // output ones, so the head's own KV cache can be filled across the prompt. Defer the narrowing
    // to just before the final norm, i.e. narrow at the LM head instead of at the last layer.
    // Costs one extra transformer layer over the prompt and nothing at decode, where every row is
    // already an output. With ALPHA_MTP unset the graph is unchanged.
    static const bool g_mtp_graph = alpha_mtp_env_on();
    const bool mtp_defer_narrow = g_mtp_graph && hparams.nextn_predict_layers > 0;
    // ALPHA_MTP_EXACT_PREFILL: see the last-layer branch below. Default on; =0 disables.
    static const bool g_mtp_exact_prefill = []() {
        const char * e = getenv("ALPHA_MTP_EXACT_PREFILL");
        return !e || atoi(e) != 0;
    }();
    // Not when the experts are host-resident (fused-infer). The exact branch runs the last layer's
    // MoE a second time in the same graph on the same expert weights, and the expert cache keys its
    // per-pass staging/offset state by weight tensor, so the two MUL_MAT_IDs would clobber each other
    // and the all-rows branch would read wrong slots (NaN hidden/KV for the MTP head).
    const bool mtp_experts_on_host = [&]() {
        const auto & Ll = model.layers[n_layer_real - 1];
        const ggml_tensor * w = Ll.ffn_up_exps ? Ll.ffn_up_exps : Ll.ffn_gate_up_exps;
        return w && w->buffer && ggml_backend_buffer_is_host(w->buffer);
    }();
    const bool mtp_exact_branch = mtp_defer_narrow && g_mtp_exact_prefill && n_outputs < n_tokens
                               && !mtp_experts_on_host;
    ggml_tensor * mtp_exact_out = nullptr;   // plain-shaped last layer over the output rows only

    for (int il = 0; il < n_layer_real; ++il) {
        // Multi-pass pipeline: skip layers outside current pass range
        if (g_alpha_pipeline_layer_start >= 0 &&
            (il < g_alpha_pipeline_layer_start || il >= g_alpha_pipeline_layer_end)) {
            continue;
        }
        // Bookend skip: skip layers in dead zone [skip_start, skip_end)
        if (g_alpha_pipeline_skip_start >= 0 &&
            il >= g_alpha_pipeline_skip_start && il < g_alpha_pipeline_skip_end) {
            continue;
        }

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        // n_layer_real, not n_layer: the loop above stops at n_layer_real, so when
        // nextn_predict_layers > 0 the max il is n_layer-2 and `il == n_layer - 1` would never fire,
        // leaving the LM head running over every position. Identical when nextn == 0.
        if (il == n_layer_real - 1 && inp_out_ids && !mtp_defer_narrow) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        // Deferred narrowing runs this layer's FFN over every prompt row, so the output row's logits
        // come from wide kernels and differ slightly from plain although the prompt KV is identical.
        // Build the plain-shaped branch (narrow first, as the non-MTP graph does) for the logits; the
        // all-rows branch below still produces the hidden the MTP head needs. Prefill only
        // (n_outputs < n_tokens). Enabled by ALPHA_MTP_EXACT_PREFILL.
        if (il == n_layer_real - 1 && inp_out_ids && mtp_exact_branch) {
            ggml_tensor * ex    = ggml_get_rows(ctx0, cur,   inp_out_ids);
            ggml_tensor * ex_sa = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            ex = ggml_add(ctx0, ex, ex_sa);
            ggml_tensor * ex_res = ex;
            ex = build_norm(ex, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
            ex = build_layer_ffn(ex, il);
            ex = ggml_add(ctx0, ex, ex_res);
            ex = build_cvec(ex, il);
            mtp_exact_out = ex;
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }

    // Multi-pass pipeline: intermediate pass outputs hidden state, not logits
    if (g_alpha_pipeline_layer_end >= 0 && g_alpha_pipeline_layer_end < n_layer) {
        cb(inpL, "pipeline_hidden_state", -1);
        ggml_set_output(inpL);
        res->t_embd = inpL;
        ggml_build_forward_expand(gf, inpL);
        return;
    }

    cur = inpL;

    // pre-output_norm hidden H, consumed by the MTP draft head below
    ggml_tensor * mtp_H = cur;
    if (mtp_defer_narrow && inp_out_ids) {
        cur = mtp_exact_out ? mtp_exact_out                            // plain-identical logits row
                            : ggml_get_rows(ctx0, cur, inp_out_ids);   // narrow here instead, for the LM head
    }

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // Export the pre-output_norm hidden H so the MTP draft head (run after decode with the embedding
    // of the just-sampled token) can consume it. The plain graph is unchanged when ALPHA_MTP is off.
    (void) inp_embd_saved;
    if (mtp_defer_narrow) {   // [n_embd, n_tokens] -- all positions, for the head's KV fill
        cb(mtp_H, "mtp_hidden", -1);
        ggml_set_output(mtp_H);
        res->t_mtp_hidden = mtp_H;
        ggml_build_forward_expand(gf, mtp_H);

        // In-graph argmax of the verify's logits: one i32 per output row instead of a full logits
        // D2H plus a host vocab scan per row. This lets llama_context skip the logits D2H
        // (`skip_logits_d2h_ && res->t_mtp_argmax != nullptr`), a fixed per-window cost.
        //
        // See qwen35.cpp for why this is not the backend-sampler path: that maps one sampled token
        // per sequence, while the verify has k+1 rows on a single sequence.
        ggml_tensor * mtp_am = ggml_argmax(ctx0, cur);   // cur = result_output [n_vocab, n_outputs]
        cb(mtp_am, "mtp_argmax", -1);
        ggml_set_output(mtp_am);
        res->t_mtp_argmax = mtp_am;
        ggml_build_forward_expand(gf, mtp_am);
    }
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35moe::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen35moe::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Use per-layer n_head/n_head_kv (hybrid models have different values per layer)
    const int64_t il_n_head    = hparams.n_head(il);
    const int64_t il_n_head_kv = hparams.n_head_kv(il);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Fused path: wq|wk|wv are adjacent same-type weights -> one GEMV, split by views.
    ggml_tensor * Qcur = nullptr;
    ggml_tensor * Kcur = nullptr;
    ggml_tensor * Vcur = nullptr;
    ggml_tensor * gate = nullptr;

    if ((alpha_fuse_proj_mask() & 2) && n_tokens <= alpha_fuse_proj_maxt() && (!loras || loras->empty()) &&
        !model.layers[il].wq_s && !model.layers[il].wk_s && !model.layers[il].wv_s &&
        !model.layers[il].wq_b && !model.layers[il].wk_b && !model.layers[il].wv_b) {
        ggml_tensor * fw = alpha_fused_weight(ctx0,
            { model.layers[il].wq, model.layers[il].wk, model.layers[il].wv });
        if (fw) {
            const int64_t q_dim = model.layers[il].wq->ne[1]; // (n_embd_head * 2) * n_head (q + gate)
            const int64_t k_dim = model.layers[il].wk->ne[1];
            const int64_t v_dim = model.layers[il].wv->ne[1];

            alpha_fuse_log_kind("attention q|k|v");
            ggml_tensor * fo = ggml_mul_mat(ctx0, fw, cur);   // [q_dim+k_dim+v_dim, n_tokens]
            cb(fo, "qkv_proj_fused", il);

            // Same views and reshapes as the split path below, taken from row slices of the fused output.
            ggml_tensor * Qcur_full = alpha_proj_slice(ctx0, fo, 0, q_dim);
            const size_t es  = ggml_element_size(fo);
            const size_t qnb = n_tokens == 1 ? es * q_dim : fo->nb[1];
            Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, il_n_head, n_tokens, es * n_embd_head * 2, qnb, 0);
            gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, il_n_head, n_tokens, es * n_embd_head * 2, qnb, es * n_embd_head);
            if (n_tokens == 1) {
                Kcur = ggml_reshape_3d(ctx0, alpha_proj_slice(ctx0, fo, q_dim, k_dim), n_embd_head, il_n_head_kv, n_tokens);
                Vcur = ggml_reshape_3d(ctx0, alpha_proj_slice(ctx0, fo, q_dim + k_dim, v_dim), n_embd_head, il_n_head_kv, n_tokens);
            } else {
                Kcur = ggml_view_3d(ctx0, fo, n_embd_head, il_n_head_kv, n_tokens, es * n_embd_head, fo->nb[1], es * q_dim);
                Vcur = ggml_view_3d(ctx0, fo, n_embd_head, il_n_head_kv, n_tokens, es * n_embd_head, fo->nb[1], es * (q_dim + k_dim));
            }
        }
    }

    if (!Qcur) {
        // Qwen3Next uses a single Q projection that outputs query + gate
        ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
        cb(Qcur_full, "Qcur_full", il);

        Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, il_n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * il_n_head, 0);

        Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
        cb(Kcur, "Kcur", il);

        Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
        cb(Vcur, "Vcur", il);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, il_n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, il_n_head_kv, n_tokens);

        gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, il_n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * il_n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    }

    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    gate = ggml_cont_2d(ctx0, gate, n_embd_head * il_n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    // Apply IMRoPE
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

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections. Fused path: wqkv|wqkv_gate|ssm_beta|ssm_alpha are adjacent
    // same-type weights -> one GEMV + strided-view splits (see alpha_fused_weight).
    ggml_tensor * qkv_mixed = nullptr;
    ggml_tensor * z         = nullptr;  // raw gate, pre-silu
    ggml_tensor * beta      = nullptr;  // raw, pre-sigmoid
    ggml_tensor * alpha     = nullptr;  // raw, [num_v_heads, n_seq_tokens, n_seqs]
    bool fused_proj = false;

    if ((alpha_fuse_proj_mask() & 4) && n_tokens <= alpha_fuse_proj_maxt() && (!loras || loras->empty()) &&
        !model.layers[il].wqkv_s && !model.layers[il].wqkv_gate_s &&
        !model.layers[il].ssm_beta_s && !model.layers[il].ssm_alpha_s) {
        ggml_tensor * fw = alpha_fused_weight(ctx0,
            { model.layers[il].wqkv, model.layers[il].wqkv_gate,
              model.layers[il].ssm_beta, model.layers[il].ssm_alpha });
        if (fw) {
            const int64_t qkv_dim = model.layers[il].wqkv->ne[1];
            const int64_t z_dim   = model.layers[il].wqkv_gate->ne[1];
            const int64_t nvh     = model.layers[il].ssm_beta->ne[1];

            alpha_fuse_log_kind("linear-attn qkv|z|beta|alpha");
            ggml_tensor * fo = ggml_mul_mat(ctx0, fw, cur);   // [qkv_dim+z_dim+2*nvh, n_tokens]
            cb(fo, "dn_proj_fused", il);

            const size_t es   = ggml_element_size(fo);
            const size_t fnb1 = fo->nb[1];
            qkv_mixed = ggml_view_3d(ctx0, fo, qkv_dim, n_seq_tokens, n_seqs,
                                     fnb1, fnb1 * n_seq_tokens, 0);
            cb(qkv_mixed, "linear_attn_qkv_mixed", il);
            z         = ggml_view_3d(ctx0, fo, z_dim, n_seq_tokens, n_seqs,
                                     fnb1, fnb1 * n_seq_tokens, qkv_dim * es);
            cb(z, "z", il);
            // beta feeds ggml_sigmoid (requires a contiguous tensor) - tiny cont
            beta      = ggml_cont(ctx0, ggml_view_3d(ctx0, fo, nvh, n_seq_tokens, n_seqs,
                                     fnb1, fnb1 * n_seq_tokens, (qkv_dim + z_dim) * es));
            // alpha only feeds ggml_add (stride-tolerant) - keep the view
            alpha     = ggml_view_3d(ctx0, fo, nvh, n_seq_tokens, n_seqs,
                                     fnb1, fnb1 * n_seq_tokens, (qkv_dim + z_dim + nvh) * es);
            fused_proj = true;
        }
    }

    bool ba_views = false;   // beta/alpha already [1, H, T, B] strided views of one fused GEMV
    const bool z_view = fused_proj;   // z is a strided row slice of the four-way fused projection (not reshapeable)
    if (!fused_proj) {
        // qkv|z alone: on GGUFs whose beta/alpha are F32 (e.g. Qwen3.6-35B-A3B) the four-way alias above fails
        // the type check, which would leave both big quantised projections split. They are same-typed and
        // created back-to-back, so one GEMV serves both.
        ggml_tensor * fqz = ((alpha_fuse_proj_mask() & 4) && n_tokens <= alpha_fuse_proj_maxt() && (!loras || loras->empty()) &&
                             !model.layers[il].wqkv_s && !model.layers[il].wqkv_gate_s)
            ? alpha_fused_weight(ctx0, { model.layers[il].wqkv, model.layers[il].wqkv_gate }) : nullptr;
        if (fqz) {
            const int64_t qkv_dim = model.layers[il].wqkv->ne[1];
            const int64_t z_dim   = model.layers[il].wqkv_gate->ne[1];
            ggml_tensor * fo = ggml_mul_mat(ctx0, fqz, cur);   // [qkv_dim+z_dim, n_tokens]
            cb(fo, "qkvz_proj_fused", il);
            // qkv feeds the stride-aware conv-state concat; z goes through the split path's reshape, so above one
            // token it is copied out (a small cont instead of a second GEMV).
            qkv_mixed = alpha_proj_slice(ctx0, fo, 0, qkv_dim);
            qkv_mixed = n_seq_tokens * n_seqs == 1
                ? ggml_reshape_3d(ctx0, qkv_mixed, qkv_dim, n_seq_tokens, n_seqs)
                : ggml_view_3d(ctx0, fo, qkv_dim, n_seq_tokens, n_seqs, fo->nb[1], fo->nb[1] * n_seq_tokens, 0);
            cb(qkv_mixed, "linear_attn_qkv_mixed", il);
            z = alpha_proj_slice(ctx0, fo, qkv_dim, z_dim);
            if (!ggml_is_contiguous(z)) {
                z = ggml_cont(ctx0, z);
            }
            cb(z, "z", il);
            alpha_fuse_log_kind("linear-attn qkv|z");
        } else {
            auto qkvz = build_qkvz(cur, il);
            qkv_mixed = qkvz.first;
            z         = qkvz.second;
        }

        ggml_tensor * fba = (alpha_fuse_ba_enabled() && (!loras || loras->empty()) &&
                             !model.layers[il].ssm_beta_s && !model.layers[il].ssm_alpha_s &&
                             cparams.fused_gdn_in && cparams.fused_gdn_ar && cparams.fused_gdn_ch)
            ? alpha_fused_weight(ctx0, { model.layers[il].ssm_beta, model.layers[il].ssm_alpha }) : nullptr;
        if (fba) {
            // Log the fusion kind like the other fusion sites. This is the common path: on quantised
            // models wqkv/wqkv_gate are q5_K or q8_0 while ssm_beta/ssm_alpha are f32, so the four-way
            // qkv|z|beta|alpha fusion declines on its dtype check and this fallback runs.
            alpha_fuse_log_kind("linear-attn beta|alpha");
            const int64_t nvh = model.layers[il].ssm_beta->ne[1];
            ggml_tensor * fo = ggml_mul_mat(ctx0, fba, cur);   // [2*nvh, n_tokens]
            cb(fo, "ba_proj_fused", il);
            const size_t es = ggml_element_size(fo);
            beta  = ggml_view_4d(ctx0, fo, 1, nvh, n_seq_tokens, n_seqs, es, fo->nb[1], fo->nb[1] * n_seq_tokens, 0);
            alpha = ggml_view_4d(ctx0, fo, 1, nvh, n_seq_tokens, n_seqs, es, fo->nb[1], fo->nb[1] * n_seq_tokens, nvh * es);
            ba_views = true;
        } else {
            beta  = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
            alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
            alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        }
    }

    if (!ba_views) {
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);

    cb(alpha, "alpha", il);

    // raw alpha; softplus(alpha + dt) * a runs inside the fused op (build_delta_net_in)
    ggml_tensor * gate = ba_views ? alpha : ggml_reshape_4d(ctx0, alpha, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels, n_seqs, conv_states_all->nb[1],
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    // Under MTP/ReplaySSM record mode the fold owns this destination: "conv state after m accepted
    // tokens" is tail_{K-1}(H_0 || p_1..p_m), i.e. this same conv_input at column offset m, so the
    // fold writes it as a copy at a runtime offset with no recurrence. Publishing here would commit
    // the full-window value that is about to be discarded.
    if (alpha_gdn_record_conv(il, conv_input, last_conv_states, state_update_target)) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    }

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;


    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN
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

    // Update the recurrent states (skipped when the op published in place: new_state == nullptr)
    // Under MTP/ReplaySSM record mode, suppressing this publish keeps S_0 free: ssm_states_all is the
    // only thing that writes the committed state, so leaving it alone means the fold's starting
    // state is still in place, needing no snapshot. `state` (read above via build_rs) is the view
    // the fold re-reads.
    if (new_state && alpha_gdn_record_state(il, ssm_state_dst, new_state)) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ssm_state_dst));
    }

    ggml_tensor * final_output;
    if (z_view) {
        // z is a strided row-slice view of the fused projection output (not reshapeable).
        // Same math as build_norm_gated, final = silu(z) * rmsnorm(output), via the
        // GLU-split op, which is built for strided views and fuses silu+mul in one kernel.
        ggml_tensor * normalized = build_norm(output, model.layers[il].ssm_norm, nullptr, LLM_NORM_RMS, il);
        normalized = ggml_reshape_3d(ctx0, normalized, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
        final_output = ggml_swiglu_split(ctx0, z, normalized);
        cb(final_output, "final_output", il);
    } else {
        // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
        ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

        // Apply gated normalization: self.norm(core_attn_out, z)
        ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

        // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
        final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
        cb(final_output, "final_output", il);
    }

    // Output projection. The reshape to [n_embd, n_tokens] is applied to the input (a free view of a
    // contiguous tensor) instead of the result, so the projection's own output is already 2-D and sits
    // directly beside the caller's residual ADD. ggml_can_fuse compares adjacent nodes and does not skip
    // views, so a reshape in between would block the {MUL_MAT, ADD} bias fusion.
    final_output = ggml_reshape_2d(ctx0, final_output, final_output->ne[0], n_seq_tokens * n_seqs);
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llm_build_qwen35moe ::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // Top-k expert reduction. ALPHA_GEN_TOPK=K (graph-level, so it reaches every serving
    // frontend including AlphaServe, unlike the thread_local g_alpha_topk_override, which
    // the server-context reset path may miss) forces top-K for the expert-skip
    // acceptance probe. 0/unset falls back to g_alpha_topk_override, then the model default.
    static int gen_topk_env = []() {
        const char * e = getenv("ALPHA_GEN_TOPK");
        return e ? atoi(e) : 0;
    }();
    const int64_t eff_n_expert_used =
        (gen_topk_env > 0 && gen_topk_env < (int)n_expert_used) ? gen_topk_env
        : (g_alpha_topk_override > 0 && g_alpha_topk_override < (int)n_expert_used) ? (int64_t)g_alpha_topk_override
        : n_expert_used;

    // Router | shared-expert score (ALPHA_FUSE_PROJ & 8): two F32 GEMVs on the same input, created back-to-back
    // by the loader, scored by one [n_expert + 1, 1] GEMV. Single-token ubatches only, where both slices are
    // contiguous and shaped exactly like the split outputs (the top-k fusion needs contiguous logits).
    ggml_tensor * router_logits = nullptr;
    ggml_tensor * shexp_score   = nullptr;
    if ((alpha_fuse_proj_mask() & 8) && cur->ne[1] == 1 && (!loras || loras->empty()) &&
        model.layers[il].ffn_gate_inp_shexp && model.layers[il].ffn_up_shexp && !model.layers[il].ffn_gate_inp_b) {
        ggml_tensor * fw = alpha_fused_weight(ctx0, { model.layers[il].ffn_gate_inp, model.layers[il].ffn_gate_inp_shexp });
        if (fw) {
            const int64_t n_rows = model.layers[il].ffn_gate_inp->ne[1];
            ggml_tensor * fo = ggml_mul_mat(ctx0, fw, cur);   // [n_expert + 1, 1]
            cb(fo, "router_shexp_score_fused", il);
            router_logits = alpha_proj_slice(ctx0, fo, 0, n_rows);
            cb(router_logits, "ffn_moe_logits", il);
            shexp_score = alpha_proj_slice(ctx0, fo, n_rows, 1);
            alpha_fuse_log_kind("router|shared-expert score");
        }
    }

    // Shared expert (gated), as in the Qwen3Next reference implementation. ffn_gate_inp_shexp is the shared
    // expert's own scalar gate (one value per token), sigmoided.
    auto build_shexp_gated = [&]() -> ggml_tensor * {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);


        ggml_tensor * shared_gate = shexp_score ? shexp_score : build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);
        return ffn_shexp;
    };

    // Build the shared expert first (ALPHA_MOE_MERGE), so the routed-expert merge (MUL by the routing weights,
    // the k slot views, the k-1 adds), the shared-expert add and the caller's residual add are consecutive
    // graph nodes that the CUDA backend fuses into one kernel. Same ops, same values, only the node order
    // moves. ALPHA_MOE_MERGE=0 restores the previous order.
    static const bool moe_merge_order = []() {
        const char * e = getenv("ALPHA_MOE_MERGE");
        return e ? atoi(e) != 0 : true;
    }();
    const bool has_shexp = model.layers[il].ffn_up_shexp != nullptr;
    ggml_tensor * shexp_first = nullptr;
    if (has_shexp && moe_merge_order) {
        shexp_first = build_shexp_gated();
        ggml_build_forward_expand(gf, shexp_first);
    }

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, eff_n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            router_logits, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s,
            model.layers[il].ffn_cache_expert_ids,
            model.layers[il].ffn_cache_expert_ids_down);
    cb(moe_out, "ffn_moe_out", il);

    if (has_shexp) {
        ggml_tensor * ffn_shexp = shexp_first ? shexp_first : build_shexp_gated();
        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}
