#pragma once

#include "llama-model.h"
#include "llama-graph.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

struct alpha_mole_graph_router; // forward declaration for MoLE

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
    llm_build_mamba_base(const llm_graph_params & params);

    virtual ~llm_build_mamba_base() = default;

    ggml_tensor * build_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
    ggml_tensor * build_mamba2_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il) const;

};

// qwen4exp QSA reads its indexer cache through this; declared in llama-memory-hybrid-idx.h
class llama_memory_hybrid_idx_context;

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // choose one of two implementations above based on the number of tokens
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // fused-input form: q/k before the L2 norm, alpha before softplus ([1, H, T, B]), beta before the
    // sigmoid, plus the per-head dt bias / decay weights; the producers run inside the op (one launch
    // instead of six per layer). Falls back to the in-graph chain + build_delta_net when the fused
    // op is off (cparams.fused_gdn_in / fused_gdn_ar / fused_gdn_ch).
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_in(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * alpha,
                ggml_tensor * beta,
                ggml_tensor * s,
                ggml_tensor * dt_bias,
                ggml_tensor * a,
                      float   eps,
                        int   il,
                ggml_tensor * state_dst = nullptr);   // committed cell: when `s` is this cell (state view) and no
                                                     // ReplaySSM recording, the op publishes in place and
                                                     // new_state comes back nullptr (nothing to copy)
};

struct llm_build_rwkv6_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv6_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv6_base() = default;

    ggml_tensor * build_rwkv6_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;

    ggml_tensor * build_rwkv6_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

// Base class for RWKV7-related models
struct llm_build_rwkv7_base : public llm_graph_context {
    const llama_model & model;

    llm_build_rwkv7_base(const llama_model & model, const llm_graph_params & params);

    virtual ~llm_build_rwkv7_base() = default;

    // RWKV7-specific graph building methods
    ggml_tensor * build_rwkv7_channel_mix(const llama_layer * layer,
                                          ggml_tensor *       cur,
                                          ggml_tensor *       x_prev,
                                          llm_arch            arch) const;
    ggml_tensor * build_rwkv7_time_mix(llm_graph_input_rs * inp,
                                       ggml_tensor *        cur,
                                       ggml_tensor *        x_prev,
                                       ggml_tensor *&       first_layer_value,
                                       const llama_ubatch & ubatch,
                                       int                  il) const;
};

//
// models
//

struct llm_build_afmoe : public llm_graph_context {
    llm_build_afmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_apertus : public llm_graph_context {
    llm_build_apertus(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_arcee : public llm_graph_context {
    llm_build_arcee(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_arctic : public llm_graph_context {
    llm_build_arctic(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_arwkv7 : public llm_build_rwkv7_base {
    llm_build_arwkv7(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_baichuan : public llm_graph_context {
    llm_build_baichuan(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bailingmoe2 : public llm_graph_context {
    llm_build_bailingmoe2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bailingmoe : public llm_graph_context {
    llm_build_bailingmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bert : public llm_graph_context {
    llm_build_bert(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bitnet : public llm_graph_context {
    llm_build_bitnet(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_bloom : public llm_graph_context {
    llm_build_bloom(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_chameleon : public llm_graph_context {
    llm_build_chameleon(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_chatglm : public llm_graph_context {
    llm_build_chatglm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_codeshell : public llm_graph_context {
    llm_build_codeshell(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_cogvlm : public llm_graph_context {
    llm_build_cogvlm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_cohere2_iswa : public llm_graph_context {
    llm_build_cohere2_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_command_r : public llm_graph_context {
    llm_build_command_r(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dbrx : public llm_graph_context {
    llm_build_dbrx(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deci : public llm_graph_context {
    llm_build_deci(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deepseek2 : public llm_graph_context {
    llm_build_deepseek2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deepseek4 : public llm_graph_context {
    llm_build_deepseek4(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_deepseek : public llm_graph_context {
    llm_build_deepseek(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dots1 : public llm_graph_context {
    llm_build_dots1(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dream : public llm_graph_context {
    llm_build_dream(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_ernie4_5 : public llm_graph_context {
    llm_build_ernie4_5(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_ernie4_5_moe : public llm_graph_context {
    llm_build_ernie4_5_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_paddleocr : public llm_graph_context {
    llm_build_paddleocr(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_exaone4 : public llm_graph_context {
    llm_build_exaone4(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_exaone : public llm_graph_context {
    llm_build_exaone(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_exaone_moe : public llm_graph_context {
    llm_build_exaone_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_falcon : public llm_graph_context {
    llm_build_falcon(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_falcon_h1 : public llm_build_mamba_base {
    llm_build_falcon_h1(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma2_iswa : public llm_graph_context {
    llm_build_gemma2_iswa(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_gemma3 : public llm_graph_context {
    llm_build_gemma3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma3n_iswa : public llm_graph_context {
    const llama_model & model;

    const int64_t n_embd_head;
    const int64_t n_embd_altup;
    const int64_t n_altup;
    const int     i_altup_act;
    const int     n_layer_sparsity = 10; // number of layers using activation sparsity
    const float   f_sparsity_std_mul = 1.6448533535003662f; // std_multiplier = normal_dist.icdf(0.95)

    llm_build_gemma3n_iswa(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * calc_magnitude(ggml_tensor * x);

    // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
    ggml_tensor * build_inp_per_layer();
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);

    ggml_tensor * gaussian_topk(ggml_tensor * x);
    ggml_tensor * altup_compute_router_modalities(ggml_tensor * x, int il);
    ggml_tensor * altup_predict(ggml_tensor * cur, int il);
    ggml_tensor * laurel(ggml_tensor * cur, int il);
    ggml_tensor * altup_correct(ggml_tensor * predictions, ggml_tensor * activated, int il);
};

struct llm_build_gemma4_iswa : public llm_graph_context {
    const llama_model & model;

    const int64_t n_embd_per_layer;

    llm_build_gemma4_iswa(const llama_model & model, const llm_graph_params & params);

    // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
    ggml_tensor * build_inp_per_layer();
    ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
};

struct llm_build_gemma_embedding : public llm_graph_context {
    llm_build_gemma_embedding(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gemma : public llm_graph_context {
    llm_build_gemma(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_glm4 : public llm_graph_context {
    llm_build_glm4(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_glm4_moe : public llm_graph_context {
    llm_build_glm4_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gpt2 : public llm_graph_context {
    llm_build_gpt2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_gptneox : public llm_graph_context {
    llm_build_gptneox(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_granite : public llm_graph_context {
    llm_build_granite(const llama_model & model, const llm_graph_params & params);

private:
    ggml_tensor * build_attention_layer(
              ggml_tensor             * cur,
              ggml_tensor             * inp_pos,
              llm_graph_input_attn_kv * inp_attn,
        const llama_model             & model,
        const int64_t                 n_embd_head,
        const int                     il);

    ggml_tensor * build_layer_ffn(
              ggml_tensor       * cur,
              ggml_tensor       * inpSA,
        const llama_model       & model,
        const int                 il);
};

struct llm_build_granite_hybrid : public llm_build_mamba_base {
    llm_build_granite_hybrid(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_layer_ffn(ggml_tensor * cur, ggml_tensor * inpSA, const llama_model & model, const int il);
    ggml_tensor * build_attention_layer(ggml_tensor * cur, ggml_tensor * inp_pos, llm_graph_input_attn_kv * inp_attn,
        const llama_model & model,const int64_t n_embd_head, const int il);
};

struct llm_build_grok : public llm_graph_context {
    llm_build_grok(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_grovemoe : public llm_graph_context {
    llm_build_grovemoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_hunyuan_dense : public llm_graph_context {
    llm_build_hunyuan_dense(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_hunyuan_moe : public llm_graph_context {
    llm_build_hunyuan_moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_internlm2 : public llm_graph_context {
    llm_build_internlm2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_jais : public llm_graph_context {
    llm_build_jais(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_jais2 : public llm_graph_context {
    llm_build_jais2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_jamba : public llm_build_mamba_base {
    llm_build_jamba(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_kimi_linear : public llm_build_delta_net_base {
    llm_build_kimi_linear(const llama_model & model, const llm_graph_params & params);

    std::pair<ggml_tensor *, ggml_tensor *> build_kda_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * gk,
                ggml_tensor * beta,
                ggml_tensor * state,
                        int   il);

    std::pair<ggml_tensor *, ggml_tensor *> build_kda_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * gk,
                ggml_tensor * beta,
                ggml_tensor * state,
                ggml_tensor * causal_mask,
                ggml_tensor * identity,
                ggml_tensor * diag_mask,
                        int   il);

    const llama_model & model;
};

template <bool iswa>
struct llm_build_lfm2 : public llm_graph_context {
    llm_build_lfm2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_llada : public llm_graph_context {
    llm_build_llada(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_llada_moe : public llm_graph_context {
    llm_build_llada_moe(const llama_model & model, const llm_graph_params & params);
};

template <bool embed>
struct llm_build_llama : public llm_graph_context {
    llm_build_llama(const llama_model & model, const llm_graph_params & params);
};

// Alpha-MoLE: llama with per-token LoRA routing
struct llm_build_mole_llama : public llm_graph_context {
    llm_build_mole_llama(const llama_model & model, const llm_graph_params & params);
    static const struct alpha_mole_graph_router * mole_router;
};

template <bool iswa>
struct llm_build_llama4 : public llm_graph_context {
    llm_build_llama4(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_maincoder : public llm_graph_context {
    llm_build_maincoder(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mamba : public llm_build_mamba_base {
    llm_build_mamba(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mimo2_iswa : public llm_graph_context {
    llm_build_mimo2_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_minicpm3 : public llm_graph_context {
    llm_build_minicpm3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_minimax_m2 : public llm_graph_context {
    llm_build_minimax_m2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mistral3 : public llm_graph_context {
    llm_build_mistral3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_modern_bert : public llm_graph_context {
    llm_build_modern_bert(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_mpt : public llm_graph_context {
    llm_build_mpt(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_nemotron : public llm_graph_context {
    llm_build_nemotron(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_nemotron_h : public llm_build_mamba_base {
    llm_build_nemotron_h(const llama_model & model, const llm_graph_params & params);
    ggml_tensor * build_ffn_layer(ggml_tensor * cur, const llama_model & model, int il);
    ggml_tensor * build_attention_layer(ggml_tensor * cur, llm_graph_input_attn_kv * inp_attn,
        const llama_model & model, int64_t n_embd_head, int il);
};

struct llm_build_neo_bert : public llm_graph_context {
    llm_build_neo_bert(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_eurobert : public llm_graph_context {
    llm_build_eurobert(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_olmo2 : public llm_graph_context {
    llm_build_olmo2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_olmoe : public llm_graph_context {
    llm_build_olmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_olmo : public llm_graph_context {
    llm_build_olmo(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_openai_moe_iswa : public llm_graph_context {
    llm_build_openai_moe_iswa(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_openelm : public llm_graph_context {
    llm_build_openelm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_orion : public llm_graph_context {
    llm_build_orion(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_pangu_embedded : public llm_graph_context {
    llm_build_pangu_embedded(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_phi2 : public llm_graph_context {
    llm_build_phi2(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_phi3 : public llm_graph_context {
    llm_build_phi3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_plamo2 : public llm_build_mamba_base {
    llm_build_plamo2(const llama_model & model, const llm_graph_params & params);
    private:
        ggml_tensor * build_plamo2_mamba_layer(llm_graph_input_rs * inp, ggml_tensor * cur, const llama_model & model, const llama_ubatch & ubatch, int il);
        ggml_tensor * build_plamo2_attn_layer(llm_graph_input_attn_kv * inp, ggml_tensor * inp_pos, ggml_tensor * cur,
                                                const llama_model & model, int il);
};

struct llm_build_plamo : public llm_graph_context {
    llm_build_plamo(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_plamo3 : public llm_graph_context {
    llm_build_plamo3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_plm : public llm_graph_context {
    llm_build_plm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen2 : public llm_graph_context {
    llm_build_qwen2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen2moe : public llm_graph_context {
    llm_build_qwen2moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen2vl : public llm_graph_context {
    llm_build_qwen2vl(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3 : public llm_graph_context {
    llm_build_qwen3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3moe : public llm_graph_context {
    llm_build_qwen3moe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3vl : public llm_graph_context {
    llm_build_qwen3vl(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3vlmoe : public llm_graph_context {
    llm_build_qwen3vlmoe(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_qwen3next : public llm_build_delta_net_base {
    llm_build_qwen3next(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};

struct llm_build_qwen35 : public llm_build_delta_net_base {
    llm_build_qwen35(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};

// TODO: derive llm_build_delta_net_base instead
struct llm_build_qwen4exp : public llm_build_delta_net_base {
    llm_build_qwen4exp(const llama_model & model, const llm_graph_params & params);
private:
    // HyperConnections: mix the hc parallel residual streams down to one, and scatter a
    // block's output back across them. These replace attn_norm/ffn_norm/output_norm.
    ggml_tensor * build_hc_mix(
                ggml_tensor *  x,
                ggml_tensor *  w_norm,
                ggml_tensor *  w_down,
                ggml_tensor *  w_up,
                ggml_tensor *  w_inject,
                ggml_tensor ** inject,
                        int    il);

    ggml_tensor * build_hc_combine(
                ggml_tensor * residual,
                ggml_tensor * block_out,
                ggml_tensor * inject,
                        int   il);

    class llm_graph_input_qsa;

    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp,
    const llama_memory_hybrid_idx_context * mctx_hyb,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il);

    // QSA: pool the indexer keys into blocks, score them, keep the top-k
    ggml_tensor * build_qsa_top_k(
    const llama_memory_hybrid_idx_context * mctx_hyb,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                ggml_tensor * kq_mask,
                        int * sections,
                        int   il);

    ggml_tensor * qsa_pkc_dbg   = nullptr;   // ALPHA_QSA_PKC_VERIFY: running sum of |cached - recomputed|

    ggml_tensor * build_attn_qsa(
    llm_graph_input_attn_kv * inp,
                ggml_tensor * q_cur,
                ggml_tensor * k_cur,
                ggml_tensor * v_cur,
                ggml_tensor * top_k,
                      float   kq_scale,
                        int   il);

    // layers sharing a compress ratio share one input set
    std::map<uint32_t, llm_graph_input_qsa *> qsa_inps;
    // ALPHA_QSA_DEV: one device copy per host-resident graph input (KQ mask, QSA bias) for prefill-sized batches;
    // otherwise every chunk view of a host input is its own scheduler split input, uploaded again per QSA layer
    std::map<const ggml_tensor *, ggml_tensor *> qsa_dev_cache;
    // ALPHA_QSA_ZERO_SHARE: one zero-filled [1, W, C] source per shape for the sparse-mask set_rows; a fresh leaf
    // per layer and query chunk would stay allocated for the whole graph
    std::map<std::pair<int64_t, int64_t>, ggml_tensor *> qsa_zero_cache;
    ggml_tensor * qsa_zeros(int64_t W, int64_t C) {
        auto key = std::make_pair(W, C);
        auto it = qsa_zero_cache.find(key);
        if (it != qsa_zero_cache.end()) return it->second;
        ggml_tensor * z = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, W, C, 1), 0.0f);
        qsa_zero_cache.emplace(key, z);
        return z;
    }
    ggml_tensor * qsa_dev(ggml_tensor * t) {
        auto it = qsa_dev_cache.find(t);
        if (it != qsa_dev_cache.end()) return it->second;
        ggml_tensor * d = ggml_cont(ctx0, t);
        qsa_dev_cache.emplace(t, d);
        return d;
    }

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    ggml_tensor * build_inp_ple(const llama_kv_cache_context * mctx_attn);

    ggml_tensor * build_ple(
         llm_graph_input_rs * inp,
                ggml_tensor * emb,
                ggml_tensor * hidden,
                        int   il);

    // Gather this layer's recurrent row once and cache it: the GDN conv history and the PLE
    // conv history live in the same row (see llama_hparams::n_embd_r), so both want it.
    ggml_tensor * build_rs_row(
         llm_graph_input_rs * inp,
                ggml_tensor * states_all,
                        int   il);

    std::map<ggml_tensor *, ggml_tensor *> rs_rows;

    const llama_model & model;
};

struct llm_build_qwen35moe : public llm_build_delta_net_base {
    llm_build_qwen35moe(const llama_model & model, const llm_graph_params & params);
private:
    ggml_tensor * build_layer_attn(
    llm_graph_input_attn_kv * inp_attn,
                ggml_tensor * cur,
                ggml_tensor * inp_pos,
                        int * sections,
                        int   il);

    ggml_tensor * build_layer_attn_linear(
         llm_graph_input_rs * inp,
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_layer_ffn(
                ggml_tensor * cur,
                        int   il);

    ggml_tensor * build_norm_gated(
                ggml_tensor * input,
                ggml_tensor * weights,
                ggml_tensor * gate,
                        int   layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                ggml_tensor * input,
                        int   il);

    const llama_model & model;
};

struct llm_build_qwen : public llm_graph_context {
    llm_build_qwen(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_refact : public llm_graph_context {
    llm_build_refact(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rnd1 : public llm_graph_context {
    llm_build_rnd1(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rwkv6 : public llm_build_rwkv6_base {
    llm_build_rwkv6(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rwkv6qwen2 : public llm_build_rwkv6_base {
    llm_build_rwkv6qwen2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_rwkv7 : public llm_build_rwkv7_base {
    llm_build_rwkv7(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_seed_oss : public llm_graph_context {
    llm_build_seed_oss(const llama_model & model, const llm_graph_params & params);
};

template <bool iswa>
struct llm_build_smallthinker : public llm_graph_context {
    llm_build_smallthinker(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_smollm3 : public llm_graph_context {
    llm_build_smollm3(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_stablelm : public llm_graph_context {
    llm_build_stablelm(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_starcoder2 : public llm_graph_context {
    llm_build_starcoder2(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_starcoder : public llm_graph_context {
    llm_build_starcoder(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_step35_iswa : public llm_graph_context {
    llm_build_step35_iswa(const llama_model & model, const llm_graph_params & params);
};

template <bool is_enc>
struct llm_build_t5 : public llm_graph_context {
    llm_build_t5(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_t5encoder : public llm_build_t5<true> {
    llm_build_t5encoder(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_wavtokenizer_dec : public llm_graph_context {
    llm_build_wavtokenizer_dec(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_xverse : public llm_graph_context {
    llm_build_xverse(const llama_model & model, const llm_graph_params & params);
};
