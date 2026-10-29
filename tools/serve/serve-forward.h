// SPDX-License-Identifier: Apache-2.0
#pragma once

// FlashInfer Forward Pass Engine
//
// Reads weights from llama_model (loaded via GGUF).
// Uses cuBLAS for matmuls + FlashInfer for paged attention.
// Supports LLaMA / Mistral / Qwen2 (standard GQA + SwiGLU FFN).
//
// This is a standalone forward pass - no ggml compute graph involved.
// Designed for high-concurrency batched decode on data center GPUs.

#include "serve-paged-kv.h"
#include "serve-cuda-graph.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>

// Forward declarations (avoid including heavy llama headers)
struct llama_model;
struct llama_hparams;
struct ggml_tensor;

namespace alphaserve {

struct ForwardConfig {
    int n_layers = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int vocab_size = 0;
    float rms_norm_eps = 1e-5f;
    float rope_freq_base = 10000.0f;
    float rope_freq_scale = 1.0f;
    // SSM / DeltaNet (Qwen3.5 hybrid linear-attention layers)
    int ssm_d_conv = 0;   // causal conv kernel width (K)
    int ssm_d_inner = 0;  // value_dim = head_v_dim * num_v_heads
    int ssm_d_state = 0;   // S = per-head dim (head_k_dim == head_v_dim)
    int ssm_n_group = 0;   // num_k_heads (H_k)
    int ssm_dt_rank = 0;   // num_v_heads (H_v)
    bool has_recurrent = false;  // any DeltaNet layer present
    int n_rot = 0;         // rotary dims (partial RoPE); 0 -> rotate full head_dim
    bool has_gated_attn = false; // any Qwen3.5 gated-attention layer present
    bool has_swa = false;        // any sliding-window (Gemma/Mistral) layer present
    // MoE FFN (Mixtral/Qwen-MoE/OLMoE-style routed experts)
    int n_expert = 0;        // total experts (0 -> dense model)
    int n_expert_used = 0;   // top-K experts per token
    int n_ff_exp = 0;        // per-expert FFN dim (from ffn_gate_exps->ne[1])
    int n_ff_shexp = 0;      // shared-expert FFN dim (qwen2moe; 0 -> no shared expert)
    bool has_moe = false;    // any MoE FFN layer present
    bool moe_all_quant = false; // every MoE layer is quant-resident -> decode MoE is fully
                                // device-routed (no host sync) -> CUDA graphs stay enabled
};

class ForwardEngine {
public:
    ForwardEngine() = default;
    ~ForwardEngine();

    // Initialize from loaded model (extracts config + weight pointers)
    bool init(const llama_model * model, PagedKVPool * kv_pool);

    // Initialize from Marlin INT4 weight directory (produced by convert_awq_to_marlin.py)
    bool init_marlin(const char * weight_dir, PagedKVPool * kv_pool);

    // Set KV pool (for deferred init after init_marlin)
    void set_kv_pool(PagedKVPool * pool) { kv_pool_ = pool; }

    // Run one forward pass for batched decode
    // token_ids:   [batch_size] input token IDs (GPU)
    // seq_ids:     [batch_size] sequence IDs (CPU, for KV pool lookup)
    // positions:   [batch_size] token positions (GPU)
    // logits_out:  [batch_size, vocab_size] output logits (GPU, FP32)
    // sampled_out: if non-null, [batch_size] greedy argmax token IDs (GPU, int32)
    //              runs batch_argmax inside the CUDA graph; no extra sync needed
    bool forward_decode(
        const int32_t * token_ids,
        const int32_t * seq_ids,
        const int32_t * positions,
        float * logits_out,
        int32_t * sampled_out,       // optional: GPU argmax output (included in graph)
        int batch_size,
        cudaStream_t stream);

    // Batched prefill: process N prompt tokens for one sequence in a single forward pass.
    // Uses causal self-attention (no paged KV lookup), writes K/V to cache.
    // Returns logits only for the last token. Much faster than N x forward_decode.
    bool forward_prefill(
        const int32_t * token_ids,    // [N] prompt tokens (GPU)
        int seq_id,                    // sequence ID for KV cache
        const int32_t * positions,     // [N] positions 0..N-1 (GPU)
        float * logits_out,           // [vocab_size] logits for last token (GPU, FP32)
        int32_t * sampled_out,        // [1] argmax of last token (GPU, int32)
        int N,                         // number of prompt tokens
        cudaStream_t stream);

    // Cached prefill: process N suffix tokens with M prefix tokens already in KV cache.
    // Uses paged prefill attention (suffix queries attend to prefix+suffix KV cache).
    // Single batched forward pass, much faster than N x forward_decode.
    bool forward_prefill_cached(
        const int32_t * token_ids,    // [N] suffix tokens (GPU)
        int seq_id,                    // sequence ID (already has M tokens in KV)
        const int32_t * positions,     // [N] positions M..M+N-1 (GPU)
        float * logits_out,           // [vocab_size] logits for last token (GPU, FP32)
        int N,                         // number of suffix tokens
        int prefix_len,                // M: number of cached prefix tokens in KV
        cudaStream_t stream);

    // Get config
    const ForwardConfig & config() const { return cfg_; }

    // Recurrent (DeltaNet/SSM) models must not use the cached-prefill path (it skips recurrent
    // layers, leaving stale state across prefix-cache hits). Callers gate prefix caching on this.
    bool has_recurrent() const { return cfg_.has_recurrent; }

    // Check if model architecture is supported
    static bool is_supported(const llama_model * model);

private:
    ForwardConfig cfg_;
    const llama_model * model_ = nullptr;
    PagedKVPool * kv_pool_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cublasLtHandle_t cublaslt_ = nullptr;

    // cublasLt cached descriptors + auto-tuned algorithm per (M,N,K) shape
    struct MatmulKey {
        int m, n, k;
        bool operator==(const MatmulKey & o) const { return m == o.m && n == o.n && k == o.k; }
    };
    struct MatmulKeyHash {
        size_t operator()(const MatmulKey & k) const {
            return std::hash<int>()(k.m) ^ (std::hash<int>()(k.n) << 16) ^ (std::hash<int>()(k.k) << 32);
        }
    };
    struct CachedMatmul {
        cublasLtMatmulDesc_t op_desc = nullptr;
        cublasLtMatrixLayout_t a_desc = nullptr;
        cublasLtMatrixLayout_t b_desc = nullptr;
        cublasLtMatrixLayout_t c_desc = nullptr;
        cublasLtMatmulAlgo_t algo;
        bool tuned = false;
    };
    std::unordered_map<MatmulKey, CachedMatmul, MatmulKeyHash> lt_cache_;
    void * lt_workspace_ = nullptr;
    size_t lt_workspace_size_ = 32 * 1024 * 1024;  // 32 MiB workspace
    cudaStream_t stream_ = nullptr;
    bool use_lt_matmul_ = true;  // use cublasLt with auto-tuning

    // Get or create cached cublasLt descriptors + auto-tune algorithm
    CachedMatmul & get_cached_matmul(int m, int n, int k, cudaStream_t stream,
                                      cudaDataType_t c_type = CUDA_R_16F);

public:
    static constexpr int MARLIN_TYPE = 100;  // custom type for Marlin INT4

    // Marlin weight: packed INT4 + FP16 scales + optional zero-points
    struct MarlinWeight {
        void * d_packed = nullptr;   // Marlin-repacked INT4 on GPU
        void * d_scales = nullptr;   // FP16 permuted scales [n_groups, N] on GPU
        void * d_zp = nullptr;       // INT4 packed zero-points [n_groups, N/8] on GPU (AWQ)
        int N = 0, K = 0;
        int group_size = 128;
    };

private:
    // Weight pointers per layer (extracted from model, GPU memory)
    struct LayerWeights {
        // Attention
        const void * wq = nullptr;      // [n_embd, n_head * head_dim] (or 2x for gated attn: query+gate interleaved)
        bool is_gated_attn = false;      // Qwen3.5/Qwen3Next: wq outputs [query|gate] per head, output sigmoid-gated
        const void * wk = nullptr;      // [n_embd, n_head_kv * head_dim]
        const void * wv = nullptr;      // [n_embd, n_head_kv * head_dim]
        const void * wo = nullptr;      // [n_head * head_dim, n_embd]

        // Fused weights (created at init for fewer cuBLAS calls)
        half * wqkv_fused = nullptr;     // [n_embd, (n_head + 2*n_kv) * head_dim] - Q+K+V concat
        half * ffn_gate_up_fused = nullptr; // [n_embd, 2 * n_ff] - gate+up concat
        int qkv_out_dim = 0;            // (n_head + 2*n_kv) * head_dim
        int gate_up_out_dim = 0;         // 2 * n_ff

        // FFN (SwiGLU)
        const void * ffn_gate = nullptr; // [n_embd, n_ff]
        const void * ffn_up = nullptr;   // [n_embd, n_ff]
        const void * ffn_down = nullptr; // [n_ff, n_embd]

        // MoE FFN (routed experts). Router is GPU FP16. Expert stacks are either
        // quant-resident (raw GGUF blocks on GPU, exps_quant_resident=true) or
        // dequanted GPU FP16 (fallback types).
        bool is_moe = false;
        const void * ffn_gate_inp = nullptr;   // router [n_embd, n_expert]
        const void * ffn_gate_exps = nullptr;  // stacked [n_embd, n_ff_exp, n_expert]
        const void * ffn_up_exps = nullptr;    // stacked [n_embd, n_ff_exp, n_expert]
        const void * ffn_down_exps = nullptr;  // stacked [n_ff_exp, n_embd, n_expert]
        bool exps_quant_resident = false;      // true -> exps pointers are raw GGUF quant blocks
        int  exps_gate_qtype = 1;              // ggml type per stack (1=F16, 8=q8_0, 12=q4_K, 14=q6_K)
        int  exps_up_qtype = 1;
        int  exps_down_qtype = 1;
        int64_t exps_gate_stride = 0;          // bytes per expert in each stack
        int64_t exps_up_stride = 0;
        int64_t exps_down_stride = 0;
        // Quant-resident dense decode weights (MoE models): raw GGUF blocks used by the
        // decode GEMVs (batch <= thresh); the F16 copies above still serve prefill/cuBLAS.
        // Per-tensor types (Q4_K_M mixes q4_K q/k with q6_K v on MoE models).
        const void * wq_q = nullptr;
        const void * wk_q = nullptr;
        const void * wv_q = nullptr;
        const void * wo_q = nullptr;
        int wq_qt = 0, wk_qt = 0, wv_qt = 0, wo_qt = 0;  // 0 -> not available
        // Shared expert (qwen2moe-style; null when absent)
        const void * ffn_gate_inp_shexp = nullptr; // [n_embd] sigmoid-gate vector
        const void * ffn_gate_shexp = nullptr;     // [n_embd, n_ff_shexp]
        const void * ffn_up_shexp = nullptr;       // [n_embd, n_ff_shexp]
        const void * ffn_down_shexp = nullptr;     // [n_ff_shexp, n_embd]

        // QK-norm width: false -> per-head RMSNorm over head_dim (Qwen3/3.5);
        // true -> full-width RMSNorm over the whole Q/K projection (OLMoE/OLMo2)
        bool qk_norm_full = false;

        // Attention biases (Qwen models; required, max magnitude ~130)
        const void * bq = nullptr;  // [n_head * head_dim] FP16
        const void * bk = nullptr;  // [n_head_kv * head_dim] FP16
        const void * bv = nullptr;  // [n_head_kv * head_dim] FP16

        // Norms
        const void * attn_norm = nullptr; // [n_embd]
        const void * ffn_norm = nullptr;  // [n_embd]
        const void * q_norm = nullptr;    // [head_dim] Qwen3/3.5 QK-norm (per-head RMSNorm)
        const void * k_norm = nullptr;    // [head_dim]

        // Sliding-window attention: >0 -> this layer attends only to the last `swa_window` KV
        // positions (Gemma/Mistral); 0 -> full causal. Set from model metadata per-layer.
        int swa_window = 0;

        // DeltaNet / linear-attention (Qwen3.5 recurrent layers)
        bool is_recurrent = false;
        int  recur_ord = -1;              // ordinal among recurrent layers (state-pool index)
        const void * wqkv = nullptr;       // [n_embd, key_dim*2+value_dim]  (attn_qkv)
        const void * wqkv_gate = nullptr;  // [n_embd, value_dim]            (attn_gate)
        const void * ssm_conv1d = nullptr; // [K, conv_dim]
        const void * ssm_dt = nullptr;     // [H_v] bias
        const void * ssm_a = nullptr;      // [H_v]  (== -exp(A_log))
        const void * ssm_beta = nullptr;   // [n_embd, H_v]
        const void * ssm_alpha = nullptr;  // [n_embd, H_v]
        const void * ssm_norm = nullptr;   // [head_v_dim] gated RMSNorm
        const void * ssm_out = nullptr;    // [value_dim, n_embd]
        int dn_wtype = 1;                  // ggml_type of DeltaNet matmul weights (1=F16, 30=BF16)
        half * dn_proj_fused = nullptr;    // fused [wqkv|wqkv_gate|ssm_beta|ssm_alpha] projection (one GEMM)
        int    dn_proj_n = 0;              // fused output dim = conv_dim + value_dim + 2*Hv

        // Data types (ggml_type enum values; MARLIN_TYPE=100 for Marlin INT4)
        int wq_type = 0;
        int ffn_type = 0;
        int attn_norm_type = 0;
        int ffn_norm_type = 0;

        // Marlin INT4 weights (only set when wq_type == MARLIN_TYPE)
        MarlinWeight mq, mk, mv, mo;           // attention projections
        MarlinWeight mqkv_fused;                // fused Q+K+V (concat along N)
        MarlinWeight m_gate, m_up, m_down;      // FFN projections
        MarlinWeight m_gate_up_fused;           // fused gate+up (concat along N)
    };
    std::vector<LayerWeights> layer_weights_;

    // Global weights
    const void * tok_embd_ = nullptr;   // [vocab, n_embd]
    int tok_embd_qtype_ = 0;            // if !=0: tok_embd_ is raw quant blocks (dequant-gather at embed)
    const void * output_norm_ = nullptr; // [n_embd]
    const void * output_ = nullptr;      // [n_embd, vocab]

    // Workspace buffers (GPU, pre-allocated)
    half * hidden_buf_ = nullptr;    // [max_batch, n_embd]
    half * residual_buf_ = nullptr;  // [max_batch, n_embd]
    half * q_buf_ = nullptr;         // [max_batch, n_head * head_dim]
    half * k_buf_ = nullptr;         // [max_batch, n_head_kv * head_dim]
    half * v_buf_ = nullptr;         // [max_batch, n_head_kv * head_dim]
    half * attn_out_buf_ = nullptr;  // [max_batch, n_head * head_dim]
    half * qkv_buf_ = nullptr;       // [max_batch, qkv_out_dim] fused QKV output
    half * gate_up_buf_ = nullptr;   // [max_batch, 2*n_ff] fused gate+up output
    half * qg_buf_ = nullptr;        // [max_batch, 2*n_head*head_dim] gated-attn query+gate matmul output
    half * gate_buf_ = nullptr;      // [max_batch, n_head*head_dim] deinterleaved attention gate
    half * q4k_scratch_ = nullptr;   // [max_weight_elements] scratch for Q4_K dequant

    // Device pointer arrays for cublasHgemmBatched (per-layer, pre-allocated at init)
    struct BatchedPtrs {
        const half ** d_kv_B = nullptr;   // [2] -> {Wk, Wv}
        const half ** d_kv_A = nullptr;   // [2] -> {hidden, hidden}
        half ** d_kv_C = nullptr;          // [2] -> {k_buf, v_buf}
        const half ** d_ffn_B = nullptr;  // [2] -> {Wgate, Wup}
        const half ** d_ffn_A = nullptr;  // [2] -> {hidden, hidden}
        half ** d_ffn_C = nullptr;         // [2] -> {ffn1, ffn2}
    };
    std::vector<BatchedPtrs> batched_ptrs_; // one per layer
    half * ffn_buf1_ = nullptr;      // [max_batch, n_ff]
    half * ffn_buf2_ = nullptr;      // [max_batch, n_ff]
    float * logits_buf_ = nullptr;   // [max_batch, vocab]
    float * norm_buf_ = nullptr;     // [max_batch, n_embd] (for RMSNorm output in fp32)

    int max_batch_ = 0;
    int padded_batch_ = 0;   // max(max_batch, 16) for Tensor Core M-padding
    std::mutex forward_mtx_;
    bool use_cuda_graph_ = true;
    bool use_fused_kernels_ = true;   // set false for A/B testing
    bool use_flashinfer_attn_ = true; // use FlashInfer optimized attention kernel
    CUDAGraphPool cuda_graphs_;
    std::unordered_map<int, int> graph_warmup_;  // bucket -> warmup count

    // F32->FP16 converted buffers (freed in destructor)
    std::vector<half *> f32_converted_bufs_;

    // DeltaNet / linear-attention (Qwen3.5 recurrent layers)
    // Per recurrent-layer state slabs, indexed by seq_id.
    int recur_conv_size_ = 0;  // (K-1)*conv_channels per seq (f16 elems)
    int recur_ssm_size_ = 0;   // S*S*H_v per seq (f32 elems)
    int recur_n_layers_ = 0;   // number of recurrent layers
    int recur_max_seqs_ = 0;
    int recur_scratch_slot_ = 0;          // state slot for CUDA-graph padding rows (= recur_max_seqs_)
    std::vector<int32_t> dn_sids_padded_; // host scratch: per-step padded seq-id array (real + scratch)
    std::vector<half *>  recur_conv_;  // [recur_ord] -> [max_seqs*conv_size] f16
    std::vector<half *> recur_ssm_;    // [recur_ord] -> [max_seqs*ssm_size] f16 (state; halves DRAM traffic in delta_rule)
    // DeltaNet per-token workspace (allocated in alloc_workspace when has_recurrent)
    half * dn_fused_ = nullptr; // [max_batch, dn_proj_n] fused projection GEMM output (split into the 4 below)
    half * dn_qkv_ = nullptr;   // [max_batch, conv_dim]
    half * dn_z_ = nullptr;     // [max_batch, value_dim]
    half * dn_beta_ = nullptr;  // [max_batch, H_v]
    half * dn_g_ = nullptr;     // [max_batch, H_v]
    half * dn_conv_ = nullptr;  // [max_batch, conv_dim]
    half * dn_o_ = nullptr;     // [max_batch, value_dim]
    half * dn_y_ = nullptr;     // [max_batch, value_dim]
    int32_t * dn_seq_ids_ = nullptr; // [max_batch] GPU copy of per-token seq id
    void alloc_recurrent_state(int max_seqs);
    // Run one DeltaNet layer (reads/writes recurrent state pool per seq).
    // seq_ids_cpu[i] = sequence-state slot for token i. token_pos used for prefill order.
    void forward_linear_attn_layer(int il, int batch, const int32_t * seq_ids_cpu,
                                bool is_prefill, cudaStream_t stream);

    // MoE FFN (routed experts)
    // Workspace (allocated in alloc_workspace when has_moe). Per-pair buffers (ff1/ff2/dn)
    // hold max_batch x n_expert_used rows for the device-routed id-dispatch path.
    half  * moe_logits_ = nullptr;   // [max_batch, n_expert] router logits
    half  * moe_x_ = nullptr;        // [max_batch, n_embd] gathered per-expert input rows (host-grouped path)
    half  * moe_ff1_ = nullptr;      // [max_batch*K, n_ff_exp] gate -> silu*up
    half  * moe_ff2_ = nullptr;      // [max_batch*K, n_ff_exp] up
    half  * moe_dn_ = nullptr;       // [max_batch*K, n_embd] down output (also shexp down)
    float * moe_acc_ = nullptr;      // [max_batch, n_embd] FP32 weighted-sum accumulator
    half  * moe_sh1_ = nullptr;      // [max_batch, n_ff_shexp] shared-expert gate
    half  * moe_sh2_ = nullptr;      // [max_batch, n_ff_shexp] shared-expert up
    half  * moe_shg_ = nullptr;      // [max_batch] shared-expert sigmoid-gate logits
    int32_t * d_moe_idx_ = nullptr;  // [max_batch * n_expert_used] device path: expert id per pair;
                                     //   host-grouped path: token index per (expert,token) pair
    float   * d_moe_w_ = nullptr;    // [max_batch * n_expert_used] router weight per pair
    int32_t * d_moe_zero_ids_ = nullptr; // [max_batch] all-zero ids -> dense GEMVs via gemv_id
    const void * output_q_ = nullptr;    // raw quant output (lm_head) for decode logits GEMV
    int output_qtype_ = 0;
    void * moe_act_ = nullptr;           // [max rows, maxK/32] Q81Act int8 activations (dp4a path)
    half    * h_moe_logits_ = nullptr; // pinned host mirror of moe_logits_ (host-grouped path)
    int32_t * h_moe_idx_ = nullptr;    // pinned host staging for d_moe_idx_
    float   * h_moe_w_ = nullptr;      // pinned host staging for d_moe_w_
    // Eager MoE FFN for one layer: reads hidden_buf_ (post ffn_norm), writes hidden_buf_.
    // Quant-resident experts + small batch -> fully device-routed (GPU top-K + gemv-id,
    // no host sync, CUDA-graph capturable). Larger batches / F16 fallback -> host-grouped
    // dispatch (router readback + per-expert GEMMs; syncs the stream).
    void forward_moe_ffn(int il, int batch, cudaStream_t stream);

    // Qwen3.5/Qwen3Next gated-attention QKV producer: wq->[query|gate] interleaved,
    // deinterleave, QK-norm, K/V projections, partial RoPE. Fills q_buf_/k_buf_/v_buf_
    // and gate_buf_ (applied as sigmoid gate to the attention output before wo).
    // Shared by decode/prefill/cached; the attention + wo stay in each forward path.
    void forward_gated_attn_qkv(int il, int effective_batch,
                                const int32_t * positions, cudaStream_t stream);

    // Allocate workspace buffers
    bool alloc_workspace(int max_batch);

    // Per-layer operations
    void rms_norm(half * out, const half * inp, const void * weight,
                  int n, int batch, cudaStream_t stream);
    void matmul(half * out, const half * a, const void * b,
                int m, int n, int k, int b_type, cudaStream_t stream);
    void marlin_matmul(half * out, const half * a, const MarlinWeight & w,
                       int m, cudaStream_t stream);
    void silu_mul(half * out, const half * gate, const half * up,
                  int n, int batch, cudaStream_t stream);
    void rope(half * q, half * k, const int32_t * positions,
              int head_dim, int n_head_q, int n_head_k, int batch,
              float freq_base, float freq_scale, cudaStream_t stream);
    void add_residual(half * out, const half * residual,
                      int n, int batch, cudaStream_t stream);
    void embed_tokens(half * out, const int32_t * token_ids,
                      int batch, cudaStream_t stream);
};

} // namespace alphaserve
