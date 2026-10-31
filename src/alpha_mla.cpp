// AlphaLlama: MLA (Multi-head Latent Attention) graph builder
// This file contains the MLA attention path for DeepSeek V2/V3/V4 models.
// To be integrated into the deepseek2 model builder (src/models/deepseek2.cpp)
// or equivalent graph building code.
//
// MLA replaces standard MHA/GQA with compressed latent attention:
//   Q: cur -> wq_a -> norm -> wq_b -> split(nope|rope) -> rope_ext
//   KV: cur -> wkv_a_mqa -> split(kv_cmpr|k_pe) -> rope_ext
//   Absorption: q_nope -> permute -> wk_b -> permute (absorb K into Q)
//   Attention: concat(q_nope_absorbed, q_pe) x concat(kv_cmpr, k_pe)
//   Output: -> wv_b (value decompression)
//
// Detection: if (hparams.is_mla()), which checks n_embd_head_k_mla_impl != 0
// Generic: works for DeepSeek V2, V3, V4 Flash/Pro, GLM 5.1, Kimi K2.6

// NOTE: this is a reference implementation. The integration point depends on
// the codebase version; normally it goes into src/models/deepseek2.cpp's
// build_graph method.

/*
Integration pseudocode for deepseek2.cpp build_graph:

    // After: cur = build_norm(cur, layer.attn_norm, ...)

    const bool is_mla = hparams.is_mla();
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    ggml_tensor * q;

    // Q projection (LoRA decomposed for non-lite models)
    if (model.layers[il].wq_a) {
        q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
        cb(q, "attn_q_a", il);
        q = build_norm(q, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(q, "attn_q_a_normed", il);
        q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
        cb(q, "attn_q_b", il);
    } else {
        q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
        cb(q, "attn_q", il);
    }
    q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);

    // KV compressed projection
    ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
    cb(kv_cmpr_pe, "attn_kv_cmpr_pe", il);

    // Split KV compressed into: kv_cmpr [kv_lora_rank] + k_pe [rope_dim]
    ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe,
        kv_lora_rank, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);

    ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe,
        n_embd_head_qk_rope, 1, n_tokens,
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
        ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

    // Normalize KV compressed
    kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv_cmpr, "attn_kv_cmpr_normed", il);

    if (is_mla && n_embd_head_qk_nope > 0) {
        // Standard MLA: split Q into nope and rope parts
        ggml_tensor * q_nope = ggml_view_3d(ctx0, q,
            n_embd_head_qk_nope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head, 0);

        ggml_tensor * q_pe = ggml_view_3d(ctx0, q,
            n_embd_head_qk_rope, n_head, n_tokens,
            ggml_row_size(q->type, n_embd_head_k),
            ggml_row_size(q->type, n_embd_head_k) * n_head,
            ggml_row_size(q->type, n_embd_head_qk_nope));

        // RoPE on position-encoded parts
        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q_pe, "attn_q_pe_roped", il);

        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_pe, "attn_k_pe_roped", il);

        // MLA absorption: absorb K projection into Q via wk_b
        // q_nope [nope, n_head, n_tokens] -> permute -> wk_b -> permute
        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
        cb(q_nope_absorbed, "attn_q_nope_absorbed", il);

        // Concatenate: Q = [absorbed_nope, rope_pe], K = [kv_cmpr, k_pe]
        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        ggml_tensor * Vcur = kv_cmpr;

        // Attention (MQA with 1 KV head) + V decompression via wv_b
        float kq_scale = 1.0f / sqrtf((float)(kv_lora_rank + n_embd_head_qk_rope));
        cur = build_attn(inp_attn_k, model.layers[il].wo, NULL, model.layers[il].wo_s,
                         Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);

    } else if (is_mla && n_embd_head_qk_nope == 0) {
        // V4 Flash: all Q dims are RoPE, no nope component
        // Skip absorption, apply RoPE to full Q
        q = ggml_rope_ext(ctx0, q, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        cb(q, "attn_q_roped", il);

        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
        cb(k_pe, "attn_k_pe_roped", il);

        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        ggml_tensor * Vcur = kv_cmpr;

        float kq_scale = 1.0f / sqrtf((float)(kv_lora_rank + n_embd_head_qk_rope));
        cur = build_attn(inp_attn_k, model.layers[il].wo, NULL, model.layers[il].wo_s,
                         q, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, kq_scale, il);
    }
    // else: standard GQA path (existing code, no changes needed)

    cb(cur, "attn_out", il);
*/
