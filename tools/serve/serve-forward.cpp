// FlashInfer forward pass engine
//
// Reads weights from llama_model, runs inference via cuBLAS + FlashInfer.
// This file handles initialization and the main forward_decode() loop.
// CUDA kernels (RMSNorm, RoPE, SiLU, embed) are in serve-kernels.cu.

#include "serve-forward.h"
namespace alphaserve { extern int g_serve_prompt_chunk; }  // prefill chunk size (defined in alphaserve.cpp)
#include "llama-model.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <algorithm>  // partial_sort (MoE top-K router)

// Marlin kernel launchers (implemented in serve-marlin.cu)
extern "C" int launch_marlin_matmul(const void *, const void *, void *, const void *,
                                     int, int, int, int, void *);
extern "C" int launch_marlin_matmul_awq(const void *, const void *, void *, const void *,
                                         const void *, int, int, int, int, void *);
extern "C" void marlin_kernel_init_standalone();
extern "C" void launch_gptq_marlin_repack_standalone(
    const uint32_t *, uint32_t *, int, int, cudaStream_t);

// Bias add
extern "C" void launch_add_bias(half * out, const half * bias, int dim, int batch, cudaStream_t stream);

// CUDA kernel launchers (implemented in serve-kernels.cu)

extern "C" void launch_paged_decode_attention(
    half * output, const half * q,
    const half * k_pool, const half * v_pool,
    const int32_t * indptr, const int32_t * indices, const int32_t * last_page_len,
    int page_size, int n_heads, int n_kv_heads, int head_dim,
    int stride_page, int stride_n, int stride_h,
    int batch_size, int window, cudaStream_t stream);
extern "C" void launch_rms_norm(half * out, const half * inp, const half * weight,
                                 int n, int batch, float eps, cudaStream_t stream);
extern "C" void launch_rope(half * q, half * k, const int32_t * positions,
                              int head_dim, int n_head_q, int n_head_k, int batch,
                              float freq_base, float freq_scale, cudaStream_t stream);
extern "C" void launch_silu_mul(half * out, const half * gate, const half * up,
                                 int n, int batch, cudaStream_t stream);
extern "C" void launch_embed_tokens(half * out, const half * table, const int32_t * token_ids,
                                     int n_embd, int batch, cudaStream_t stream);
extern "C" void launch_embed_tokens_quant(half * out, const void * table, const int32_t * token_ids,
                                           int n_embd, int batch, int qtype, cudaStream_t stream);
extern "C" void launch_add_residual(half * out, const half * residual,
                                     int n, int batch, cudaStream_t stream);
extern "C" void launch_f32_to_f16(half * out, const float * inp, int64_t n, cudaStream_t stream);
extern "C" void launch_dequant_q4k(half * out, const void * src, int n_elements, cudaStream_t stream);
extern "C" void launch_scatter_kv(half * pool, const half * src, const int64_t * offsets,
                                   int kv_dim, int batch_size, cudaStream_t stream);
extern "C" void launch_scatter_kv_pair(half * k_pool, half * v_pool,
                                        const half * k_src, const half * v_src,
                                        const int64_t * offsets,
                                        int kv_dim, int batch_size, cudaStream_t stream);
extern "C" void marlin_reset_locks(void * stream_ptr);
extern "C" void launch_fused_rmsnorm_residual(half * out, half * residual, const half * inp,
                                               const half * weight, int n, int batch, float eps,
                                               cudaStream_t stream);
extern "C" void launch_fused_add_residual_rmsnorm(half * out, half * residual, const half * inp,
                                                    const half * weight, int n, int batch, float eps,
                                                    cudaStream_t stream);
extern "C" void launch_batch_argmax(int32_t * out_ids, const float * logits,
                                     int batch_size, int vocab_size, cudaStream_t stream);
extern "C" void launch_deinterleave_qkv(half * q, half * k, half * v, const half * qkv,
                                         int q_dim, int kv_dim, int batch_size, cudaStream_t stream);
extern "C" int flashinfer_attn_supported(int n_qo_heads, int n_kv_heads, int head_dim);
extern "C" cudaError_t launch_flashinfer_batch_decode(
    half * output, const half * q,
    half * k_pool, half * v_pool,
    const int32_t * indptr, const int32_t * indices, const int32_t * last_page_len,
    int page_size, int n_qo_heads, int n_kv_heads, int head_dim,
    int batch_size, int padded_batch_size,
    int stride_page, int stride_n, int stride_h,
    cudaStream_t stream);
extern "C" void launch_fused_deinterleave_rope(
    half * q, half * k, half * v, const half * qkv,
    const int32_t * positions,
    const half * bq, const half * bk, const half * bv,
    int q_dim, int kv_dim, int head_dim, int batch_size,
    float freq_base, float freq_scale, cudaStream_t stream);
extern "C" void launch_silu_mul_interleaved(
    half * out, const half * gate_up,
    int n_ff, int batch, cudaStream_t stream);
extern "C" void launch_qk_norm(
    half * data, const half * weight, int n_heads, int head_dim,
    int token_stride, int base_off, int batch, float eps, cudaStream_t stream);
extern "C" void launch_la_gates(half * beta, half * g, const half * dt, const half * a, int Hv, int batch, cudaStream_t stream);
extern "C" void launch_delta_rule_fused(half * o, const half * conv, const half * g_raw, const half * beta_raw,
    const half * dt, const half * a, half * ssm, const int32_t * seq_ids, int S, int Hk, int Hv,
    int key_dim, int value_dim, int ssm_size_per_seq, int conv_dim, float eps, int batch,
    cudaStream_t stream);
extern "C" void launch_dn_proj_split(const half * fused, half * qkv, half * z, half * beta, half * g,
    int conv_dim, int value_dim, int Hv, int batch, cudaStream_t stream);
extern "C" void launch_la_conv1d(half * out, const half * in, half * state, const half * w,
    int K, int conv_dim, int conv_size_per_seq, const int32_t * seq_ids, int batch, cudaStream_t stream);
extern "C" void launch_l2_norm(half * data, int head_dim, int n_heads, int token_stride, int base_off, int batch, float eps, cudaStream_t stream);
extern "C" void launch_delta_rule(half * o, const half * conv, const half * g, const half * beta,
    half * ssm, const int32_t * seq_ids, int S, int Hk, int Hv, int key_dim, int value_dim,
    int ssm_size_per_seq, int conv_dim, int batch, cudaStream_t stream);
extern "C" void launch_gated_rmsnorm(half * y, const half * o, const half * z, const half * w, int S, int Hv, int value_dim, int batch, float eps, cudaStream_t stream);
// Qwen3.5 gated attention
extern "C" void launch_deinterleave_qgate(half * q_out, half * gate_out, const half * qg,
                                          int head_dim, int n_head, int batch, cudaStream_t stream);
extern "C" void launch_partial_rope(half * q, half * k, const int32_t * positions,
    int head_dim, int n_rot, int n_head_q, int n_head_k, int batch,
    float freq_base, float freq_scale, cudaStream_t stream);
extern "C" void launch_gate_sigmoid_mul(half * attn_out, const half * gate, int n, int batch, cudaStream_t stream);
// MoE FFN (routed experts)
extern "C" void launch_moe_scatter_add(float * acc, const half * src, const int32_t * idx,
                                        const float * w, int n_embd, int count, cudaStream_t stream);
extern "C" void launch_moe_finalize(half * out, const float * acc, const half * shexp,
                                     const half * shexp_gate, int n_embd, int batch, cudaStream_t stream);
// Quant-resident MoE (serve-q4k-gemv.cu): fused-dequant GEMV on raw GGUF blocks
extern "C" void launch_q4k_matmul(half * out, const half * input, const void * weight,
                                   int M, int N, int K, int weight_type, cudaStream_t stream);
extern "C" void launch_moe_gemv_id(half * out, const half * input, const void * wbase,
                                    const int32_t * ids, int n_pairs, int k_per_tok,
                                    int64_t expert_stride_bytes, int N, int K, int qtype,
                                    cudaStream_t stream);
extern "C" void launch_moe_gemv_id_f32(float * out, const half * input, const void * wbase,
                                        const int32_t * ids, int n_pairs, int k_per_tok,
                                        int64_t expert_stride_bytes, int N, int K, int qtype,
                                        cudaStream_t stream);
extern "C" void launch_moe_gemv_id_strided(half * out, int out_stride, const half * input,
                                            const void * wbase, const int32_t * ids,
                                            int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
                                            int N, int K, int qtype, cudaStream_t stream);
extern "C" void launch_moe_topk(int32_t * ids, float * wout, const half * logits,
                                 int n_expert, int k_used, int batch, cudaStream_t stream);
extern "C" void launch_moe_combine(float * acc, const half * dn, const float * w,
                                    int k_used, int n_embd, int batch, cudaStream_t stream);
// MMVQ-style dp4a path: int8-quantized activations + integer-dot GEMV (q4_K/q6_K)
extern "C" void launch_moe_q8_act(void * out, const half * in, int rows, int K, cudaStream_t stream);
extern "C" void launch_moe_gemv_id_q8(half * out, int out_stride, const void * act,
                                       const void * wbase, const int32_t * ids,
                                       int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
                                       int N, int K, int qtype, cudaStream_t stream);
extern "C" void launch_moe_gemv_id_q8_f32(float * out, const void * act,
                                           const void * wbase, const int32_t * ids,
                                           int n_pairs, int k_per_tok, int64_t expert_stride_bytes,
                                           int N, int K, int qtype, cudaStream_t stream);

namespace alphaserve {

// dp4a int-dot GEMV exists for these weight types (q4_K, q6_K)
static inline bool moe_dp4a_type(int t) { return t == 12 || t == 14; }

ForwardEngine::~ForwardEngine() {
    for (auto & [key, cm] : lt_cache_) {
        if (cm.op_desc) cublasLtMatmulDescDestroy(cm.op_desc);
        if (cm.a_desc) cublasLtMatrixLayoutDestroy(cm.a_desc);
        if (cm.b_desc) cublasLtMatrixLayoutDestroy(cm.b_desc);
        if (cm.c_desc) cublasLtMatrixLayoutDestroy(cm.c_desc);
    }
    for (auto * buf : f32_converted_bufs_) cudaFree(buf);
    for (auto & lw : layer_weights_) {
        if (lw.wqkv_fused) cudaFree(lw.wqkv_fused);
        if (lw.ffn_gate_up_fused) cudaFree(lw.ffn_gate_up_fused);
    }
    if (qkv_buf_) cudaFree(qkv_buf_);
    if (gate_up_buf_) cudaFree(gate_up_buf_);
    if (moe_logits_) cudaFree(moe_logits_);
    if (moe_x_) cudaFree(moe_x_);
    if (moe_ff1_) cudaFree(moe_ff1_);
    if (moe_ff2_) cudaFree(moe_ff2_);
    if (moe_dn_) cudaFree(moe_dn_);
    if (moe_acc_) cudaFree(moe_acc_);
    if (moe_sh1_) cudaFree(moe_sh1_);
    if (moe_sh2_) cudaFree(moe_sh2_);
    if (moe_shg_) cudaFree(moe_shg_);
    if (d_moe_idx_) cudaFree(d_moe_idx_);
    if (d_moe_w_) cudaFree(d_moe_w_);
    if (d_moe_zero_ids_) cudaFree(d_moe_zero_ids_);
    if (moe_act_) cudaFree(moe_act_);
    if (h_moe_logits_) cudaFreeHost(h_moe_logits_);
    if (h_moe_idx_) cudaFreeHost(h_moe_idx_);
    if (h_moe_w_) cudaFreeHost(h_moe_w_);
    if (cublaslt_) cublasLtDestroy(cublaslt_);
    if (lt_workspace_) cudaFree(lt_workspace_);
    if (cublas_) cublasDestroy(cublas_);
    if (hidden_buf_) cudaFree(hidden_buf_);
    if (residual_buf_) cudaFree(residual_buf_);
    if (q_buf_) cudaFree(q_buf_);
    if (k_buf_) cudaFree(k_buf_);
    if (v_buf_) cudaFree(v_buf_);
    if (attn_out_buf_) cudaFree(attn_out_buf_);
    if (ffn_buf1_) cudaFree(ffn_buf1_);
    if (ffn_buf2_) cudaFree(ffn_buf2_);
    if (logits_buf_) cudaFree(logits_buf_);
    if (norm_buf_) cudaFree(norm_buf_);
}

bool ForwardEngine::is_supported(const llama_model * model) {
    // Check that at least one layer has separate Q/K/V + SwiGLU FFN
    // Hybrid models (Qwen3.5) mix attention and recurrent layers; supported if any
    // layer has the standard attention pattern
    if (model->layers.empty()) return false;
    for (auto & l : model->layers) {
        if (l.wq && l.wk && l.wv && l.wo &&
            l.ffn_gate && l.ffn_up && l.ffn_down) {
            return true;  // found at least one supported layer
        }
    }
    return false;
}

bool ForwardEngine::init(const llama_model * model, PagedKVPool * kv_pool) {
    model_ = model;
    kv_pool_ = kv_pool;

    const auto & hp = model->hparams;

    // Extract config
    cfg_.n_layers = hp.n_layer;
    cfg_.n_embd = hp.n_embd;
    cfg_.n_ff = hp.n_ff();
    cfg_.n_head = hp.n_head();
    cfg_.n_head_kv = hp.n_head_kv();
    cfg_.head_dim = hp.n_embd_head_k();
    // Vocab size from the vocab, not hparams
    const llama_vocab * vocab = llama_model_get_vocab(model);
    cfg_.vocab_size = vocab ? llama_vocab_n_tokens(vocab) : 0;
    cfg_.rms_norm_eps = hp.f_norm_rms_eps;
    cfg_.rope_freq_base = hp.rope_freq_base_train;
    cfg_.rope_freq_scale = hp.rope_freq_scale_train;
    cfg_.n_rot = hp.n_rot() > 0 ? (int)hp.n_rot() : cfg_.head_dim;  // partial-rope dims (Qwen3.5: 64 of 256)
    // SSM / DeltaNet hparams (0 on non-hybrid models)
    cfg_.ssm_d_conv  = hp.ssm_d_conv;
    cfg_.ssm_d_inner = hp.ssm_d_inner;
    cfg_.ssm_d_state = hp.ssm_d_state;
    cfg_.ssm_n_group = hp.ssm_n_group;
    cfg_.ssm_dt_rank = hp.ssm_dt_rank;

    fprintf(stderr, "[forward] config: %d layers, %d embd, %d ff, %d heads (%d kv), %d head_dim, %d vocab\n",
            cfg_.n_layers, cfg_.n_embd, cfg_.n_ff, cfg_.n_head, cfg_.n_head_kv,
            cfg_.head_dim, cfg_.vocab_size);

    // Extract weight pointers per layer
    layer_weights_.resize(cfg_.n_layers);
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        auto & ml = model->layers[il];

        // Norm weights: raw pointers, converted to GPU FP16 later in the norm-conversion block.
        // Qwen3.5 stores the pre-FFN norm as attn_post_norm (post-attention layernorm), not ffn_norm.
        ggml_tensor * ffn_norm_src = ml.ffn_norm ? ml.ffn_norm : ml.attn_post_norm;
        lw.attn_norm = ml.attn_norm ? ggml_get_data(ml.attn_norm) : nullptr;
        lw.ffn_norm  = ffn_norm_src ? ggml_get_data(ffn_norm_src) : nullptr;
        lw.attn_norm_type = ml.attn_norm ? ml.attn_norm->type : 0;
        lw.ffn_norm_type  = ffn_norm_src ? ffn_norm_src->type  : 0;
        // Qwen3/3.5 QK-norm (per-head RMSNorm on Q/K before RoPE); null on models without it
        lw.q_norm = ml.attn_q_norm ? ggml_get_data(ml.attn_q_norm) : nullptr;
        lw.k_norm = ml.attn_k_norm ? ggml_get_data(ml.attn_k_norm) : nullptr;

        // DeltaNet / linear-attention weights (Qwen3.5 recurrent layers)
        lw.is_recurrent = hp.is_recurrent(il);
        lw.wqkv       = ml.wqkv       ? ggml_get_data(ml.wqkv)       : nullptr;
        lw.wqkv_gate  = ml.wqkv_gate  ? ggml_get_data(ml.wqkv_gate)  : nullptr;
        lw.ssm_conv1d = ml.ssm_conv1d ? ggml_get_data(ml.ssm_conv1d) : nullptr;
        lw.ssm_dt     = ml.ssm_dt     ? ggml_get_data(ml.ssm_dt)     : nullptr;
        lw.ssm_a      = ml.ssm_a      ? ggml_get_data(ml.ssm_a)      : nullptr;
        lw.ssm_beta   = ml.ssm_beta   ? ggml_get_data(ml.ssm_beta)   : nullptr;
        lw.ssm_alpha  = ml.ssm_alpha  ? ggml_get_data(ml.ssm_alpha)  : nullptr;
        lw.ssm_norm   = ml.ssm_norm   ? ggml_get_data(ml.ssm_norm)   : nullptr;
        lw.ssm_out    = ml.ssm_out    ? ggml_get_data(ml.ssm_out)    : nullptr;
        if (lw.is_recurrent) cfg_.has_recurrent = true;

        // Matrix weights: dequant/copy is deferred until after workspace alloc below.
        // Store raw pointers for now; convert_weight_if_quantized runs in the next pass.
        lw.wq       = ml.wq       ? ggml_get_data(ml.wq)       : nullptr;
        lw.wk       = ml.wk       ? ggml_get_data(ml.wk)       : nullptr;
        lw.wv       = ml.wv       ? ggml_get_data(ml.wv)       : nullptr;
        lw.wo       = ml.wo       ? ggml_get_data(ml.wo)       : nullptr;
        lw.ffn_gate = ml.ffn_gate ? ggml_get_data(ml.ffn_gate) : nullptr;
        lw.ffn_up   = ml.ffn_up   ? ggml_get_data(ml.ffn_up)   : nullptr;
        lw.ffn_down = ml.ffn_down ? ggml_get_data(ml.ffn_down) : nullptr;
        lw.wq_type  = ml.wq       ? ml.wq->type       : 0;
        lw.ffn_type = ml.ffn_gate ? ml.ffn_gate->type : 0;

        // MoE FFN layer detection (router + stacked expert tensors). Dims read from the
        // tensor shapes (generic across archs; hparams.n_ff_exp is arch-specific/optional):
        //   ffn_gate_exps [n_embd, n_ff_exp, n_expert], ffn_down_exps [n_ff_exp, n_embd, n_expert]
        // Weight pointers are grabbed in pass 2 (dequant to GPU FP16) below.
        if (ml.ffn_gate_inp && ml.ffn_gate_exps && ml.ffn_up_exps && ml.ffn_down_exps) {
            lw.is_moe = true;
            cfg_.has_moe = true;
            cfg_.n_expert      = hp.n_expert ? (int)hp.n_expert : (int)ml.ffn_gate_exps->ne[2];
            cfg_.n_expert_used = (int)hp.n_expert_used;
            cfg_.n_ff_exp      = (int)ml.ffn_gate_exps->ne[1];
            if (ml.ffn_gate_shexp && ml.ffn_up_shexp && ml.ffn_down_shexp) {
                cfg_.n_ff_shexp = (int)ml.ffn_gate_shexp->ne[1];
            }
            if (il == 0 || !cfg_.n_expert_used) {
                fprintf(stderr, "[forward] MoE layer %d: n_expert=%d top-%d n_ff_exp=%d n_ff_shexp=%d\n",
                        il, cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp, cfg_.n_ff_shexp);
            }
        }

        // Qwen3.5/Qwen3Next gated attention: wq outputs [query|gate] interleaved per head,
        // so its output dim is 2*(n_head*head_dim). Detect and flag; the output is
        // sigmoid(gate)-scaled and uses partial RoPE. (attn_output/wo input is n_head*head_dim.)
        if (ml.wq && (int64_t)ml.wq->ne[1] == 2LL * cfg_.n_head * cfg_.head_dim) {
            lw.is_gated_attn = true;
            cfg_.has_gated_attn = true;
        }

        // Sliding-window attention (Gemma2/3, Mistral): standard SWA layers attend only to the last
        // n_swa KV positions. is_swa(il) gives the per-layer pattern (interleaved for Gemma, all for
        // Mistral). swa_type==NONE (Qwen etc.) leaves swa_window at 0 (full causal). CHUNKED/SYMMETRIC
        // need a different mask than the "last-N" kernel and are not enabled here.
        if (hp.swa_type == LLAMA_SWA_TYPE_STANDARD && hp.is_swa(il) && hp.n_swa > 0) {
            lw.swa_window = (int)hp.n_swa;
            if (il == 0 || !cfg_.has_swa)
                fprintf(stderr, "[forward] SWA: standard sliding window n_swa=%u (layer %d)\n", hp.n_swa, il);
            cfg_.has_swa = true;
        }

        // Attention biases (Qwen models; magnitudes up to ~130, required for correctness)
        lw.bq = ml.wq_b ? ggml_get_data(ml.wq_b) : nullptr;
        lw.bk = ml.wk_b ? ggml_get_data(ml.wk_b) : nullptr;
        lw.bv = ml.wv_b ? ggml_get_data(ml.wv_b) : nullptr;
        if (il == 0) fprintf(stderr, "[forward] layer 0 biases: bq=%p bk=%p bv=%p\n", lw.bq, lw.bk, lw.bv);

        if (!lw.wq || !lw.wk || !lw.wv || !lw.wo) {
            fprintf(stderr, "[forward] layer %d missing attention weights (hybrid model?)\n", il);
            // For hybrid models, only attention layers have these weights; recurrent layers
            // take the DeltaNet path
        }
    }

    // Global weights
    tok_embd_ = model->tok_embd ? ggml_get_data(model->tok_embd) : nullptr;
    output_norm_ = model->output_norm ? ggml_get_data(model->output_norm) : nullptr;
    output_ = model->output ? ggml_get_data(model->output) : nullptr;

    if (!tok_embd_) {
        fprintf(stderr, "[forward] missing token embedding tensor\n");
        return false;
    }

    // Initialize cuBLAS
    cublasStatus_t stat = cublasCreate(&cublas_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[forward] cuBLAS init failed: %d\n", stat);
        return false;
    }
    cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);

    // cuBLAS Lt for autotuned matmuls
    cublasLtCreate(&cublaslt_);
    cudaMalloc(&lt_workspace_, lt_workspace_size_);

    // Allocate workspace for max batch: large enough for both batched decode (n_slots) and
    // batched prefill (prompt_chunk_size)
    int min_batch = std::max(512, alphaserve::g_serve_prompt_chunk);  // prefill chunk may exceed 512
    // ALPHA_FI_MIN_BATCH caps the activation-workspace batch (mainly logits_buf = batch*vocab*4,
    // the largest single buffer). Lowering it trades max prefill-chunk size for VRAM, which helps
    // fit a large quant-resident MoE entirely on the GPU. Prefill chunks must then stay <= this value.
    if (const char * e = getenv("ALPHA_FI_MIN_BATCH")) min_batch = std::max(16, atoi(e));
    int max_batch = kv_pool ? std::max(kv_pool->max_seqs, min_batch) : min_batch;
    if (!alloc_workspace(max_batch)) {
        return false;
    }

    // Helper: dequantize or copy a matrix weight tensor to GPU FP16.
    // Handles F32, F16, BF16, and Q4_K (type 12). Other quantized types zero with a warning.
    // Returns a new GPU FP16 buffer tracked in f32_converted_bufs_, or nullptr on failure.
    // Updates *type_out to 1 (GGML_TYPE_F16) so matmul() uses the cuBLAS F16 path.
    auto convert_weight_if_quantized = [&](ggml_tensor * tensor, size_t n_elements, int * type_out) -> const void * {
        if (!tensor) return nullptr;

        int src_type = tensor->type;
        const void * src = ggml_get_data(tensor);
        bool is_host = ggml_backend_buffer_is_host(tensor->buffer);

        if (src_type == 1) {  // GGML_TYPE_F16: may still be on CPU, copy to GPU
            half * d_buf = nullptr;
            cudaMalloc(&d_buf, n_elements * sizeof(half));
            cudaMemcpy(d_buf, src, n_elements * sizeof(half),
                       is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
            f32_converted_bufs_.push_back(d_buf);
            *type_out = 1;
            return d_buf;
        }

        // GGML_TYPE_BF16 (30) is intentionally not special-cased: it falls through to the
        // generic dequant path below (to_float -> F32 -> F16, type_out=1), because several
        // matmuls hardcode the F16 type (fused QKV, fused FFN, DeltaNet projections).

        if (src_type == 0) {  // GGML_TYPE_F32: convert to FP16
            half * d_buf = nullptr;
            cudaMalloc(&d_buf, n_elements * sizeof(half));
            if (is_host) {
                float * d_tmp = nullptr;
                cudaMalloc(&d_tmp, n_elements * sizeof(float));
                cudaMemcpy(d_tmp, src, n_elements * sizeof(float), cudaMemcpyHostToDevice);
                launch_f32_to_f16(d_buf, d_tmp, (int64_t)n_elements, 0);
                cudaDeviceSynchronize();
                cudaFree(d_tmp);
            } else {
                launch_f32_to_f16(d_buf, (const float *)src, (int64_t)n_elements, 0);
                cudaDeviceSynchronize();
            }
            f32_converted_bufs_.push_back(d_buf);
            *type_out = 1;
            return d_buf;
        }

        if (src_type == 12 || src_type >= 2) {  // Any quantized type: CPU dequant via ggml type traits
            auto traits = ggml_get_type_traits(tensor->type);
            if (traits->to_float) {
                size_t type_sz = ggml_type_size(tensor->type);
                size_t blk_sz = ggml_blck_size(tensor->type);
                size_t raw_bytes = (n_elements / blk_sz) * type_sz;
                const void * host_src = src;
                std::vector<uint8_t> h_packed;
                if (!is_host) {
                    h_packed.resize(raw_bytes);
                    cudaMemcpy(h_packed.data(), src, raw_bytes, cudaMemcpyDeviceToHost);
                    host_src = h_packed.data();
                }
                std::vector<float> h_f32(n_elements);
                traits->to_float(host_src, h_f32.data(), (int64_t)n_elements);
                // Convert f32->f16 on the CPU and upload only the F16 buffer. A full n*4 F32
                // GPU temp would OOM on a big embedding once quant-resident experts fill VRAM.
                std::vector<uint16_t> h_f16(n_elements);
                ggml_fp32_to_fp16_row(h_f32.data(), (ggml_fp16_t *)h_f16.data(), (int64_t)n_elements);
                half * d_buf = nullptr;
                cudaError_t e1 = cudaMalloc(&d_buf, n_elements * sizeof(half));
                if (e1 != cudaSuccess) {
                    fprintf(stderr, "[forward] GPU alloc failed for type %d: %s (n=%zu, %.1fMiB)\n",
                            src_type, cudaGetErrorString(e1),
                            n_elements, (double)(n_elements * sizeof(half)) / (1024*1024));
                    *type_out = 1;
                    return nullptr;
                }
                cudaMemcpy(d_buf, h_f16.data(), n_elements * sizeof(half), cudaMemcpyHostToDevice);
                f32_converted_bufs_.push_back(d_buf);
                *type_out = 1;
                static int dq_count = 0;
                if (dq_count < 50) {
                    fprintf(stderr, "[dequant] type=%d n=%zu → FP16 OK\n", src_type, n_elements);
                    dq_count++;
                }
                return d_buf;
            }
        }

        // Unknown type: zero
        fprintf(stderr, "[forward] WARNING: unsupported weight type %d, zeroing %zu elements\n",
                src_type, n_elements);
        half * d_buf = nullptr;
        cudaMalloc(&d_buf, n_elements * sizeof(half));
        cudaMemset(d_buf, 0, n_elements * sizeof(half));
        f32_converted_bufs_.push_back(d_buf);
        *type_out = 1;
        return d_buf;
    };

    // Helper: copy a weight tensor to GPU FP16, handling CPU/GPU source and F32/F16 types
    auto convert_weight_to_gpu_f16 = [&](const void * src, ggml_tensor * tensor, int n) -> half * {
        if (!src || !tensor) return nullptr;

        half * h_buf = nullptr;
        cudaMalloc(&h_buf, n * sizeof(half));

        bool is_host = ggml_backend_buffer_is_host(tensor->buffer);
        int src_type = tensor->type;

        if (src_type == 0) {  // F32
            if (is_host) {
                float * d_tmp = nullptr;
                cudaMalloc(&d_tmp, n * sizeof(float));
                cudaMemcpy(d_tmp, src, n * sizeof(float), cudaMemcpyHostToDevice);
                launch_f32_to_f16(h_buf, d_tmp, n, 0);
                cudaDeviceSynchronize();
                cudaFree(d_tmp);
            } else {
                launch_f32_to_f16(h_buf, (const float *)src, n, 0);
                cudaDeviceSynchronize();
            }
        } else if (src_type == 1) {  // F16
            cudaMemcpy(h_buf, src, n * sizeof(half),
                       is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
        } else {
            cudaMemset(h_buf, 0, n * sizeof(half));
        }
        f32_converted_bufs_.push_back(h_buf);
        return h_buf;
    };

    // Pass 2: dequantize / GPU-copy all matrix weights per layer.
    // For Q4_K (type 12) this allocates a GPU FP16 buffer and dequantizes.
    // For F16/BF16/F32 this copies to GPU if needed and normalises the type field.
    // After this loop every lw.wq / wk / wv / wo / ffn_* pointer is a GPU FP16 buffer
    // and lw.wq_type / lw.ffn_type == 1, so matmul() always takes the cuBLAS FP16 path.
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        auto & ml = model->layers[il];

        if (ml.wq) {
            // Gated attention: wq output is 2*(n_head*head_dim) (query+gate); load full.
            size_t n = (size_t)cfg_.n_embd * cfg_.n_head * cfg_.head_dim * (lw.is_gated_attn ? 2 : 1);
            lw.wq = convert_weight_if_quantized(ml.wq, n, &lw.wq_type);
        }
        if (ml.wk) {
            size_t n = (size_t)cfg_.n_embd * cfg_.n_head_kv * cfg_.head_dim;
            lw.wk = convert_weight_if_quantized(ml.wk, n, &lw.wq_type);
        }
        if (ml.wv) {
            size_t n = (size_t)cfg_.n_embd * cfg_.n_head_kv * cfg_.head_dim;
            lw.wv = convert_weight_if_quantized(ml.wv, n, &lw.wq_type);
        }
        if (ml.wo) {
            size_t n = (size_t)cfg_.n_head * cfg_.head_dim * cfg_.n_embd;
            lw.wo = convert_weight_if_quantized(ml.wo, n, &lw.wq_type);
        }
        // MoE models: also keep the attention weights quant-resident for the decode
        // GEMVs (separate q/k/v/o buffers, since Q4_K_M mixes types per tensor). The F16
        // copies above still serve prefill/cuBLAS.
        if (cfg_.has_moe && !lw.is_gated_attn && ml.wq && ml.wk && ml.wv && ml.wo) {
            auto qres_ok = [&](ggml_tensor * t) -> int {  // returns type if usable, else 0
                int ty = t->type;
                int be = (ty == 12 || ty == 14) ? 256 : (ty == 8) ? 32 : 0;
                return (be > 0 && (int)t->ne[0] % be == 0) ? ty : 0;
            };
            auto upload = [&](ggml_tensor * t) -> const void * {
                size_t nb = ggml_nbytes(t);
                void * d = nullptr;
                if (cudaMalloc(&d, nb) != cudaSuccess) return nullptr;
                bool ih = ggml_backend_buffer_is_host(t->buffer);
                cudaMemcpy(d, ggml_get_data(t), nb,
                           ih ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
                f32_converted_bufs_.push_back((half *)d);
                return d;
            };
            int tq = qres_ok(ml.wq), tk = qres_ok(ml.wk), tv = qres_ok(ml.wv), to = qres_ok(ml.wo);
            if (tq && tk && tv && to) {
                lw.wq_q = upload(ml.wq); lw.wq_qt = tq;
                lw.wk_q = upload(ml.wk); lw.wk_qt = tk;
                lw.wv_q = upload(ml.wv); lw.wv_qt = tv;
                lw.wo_q = upload(ml.wo); lw.wo_qt = to;
                if (!lw.wq_q || !lw.wk_q || !lw.wv_q || !lw.wo_q) {  // OOM: fall back
                    lw.wq_q = lw.wk_q = lw.wv_q = lw.wo_q = nullptr;
                    lw.wq_qt = lw.wk_qt = lw.wv_qt = lw.wo_qt = 0;
                } else if (il == 0) {
                    fprintf(stderr, "[forward] attn weights quant-RESIDENT for decode (types q/k/v/o %d/%d/%d/%d)\n", tq, tk, tv, to);
                }
            }
        }
        if (ml.ffn_gate) {
            size_t n = (size_t)cfg_.n_embd * cfg_.n_ff;
            lw.ffn_gate = convert_weight_if_quantized(ml.ffn_gate, n, &lw.ffn_type);
        }
        if (ml.ffn_up) {
            size_t n = (size_t)cfg_.n_embd * cfg_.n_ff;
            lw.ffn_up = convert_weight_if_quantized(ml.ffn_up, n, &lw.ffn_type);
        }
        if (ml.ffn_down) {
            size_t n = (size_t)cfg_.n_ff * cfg_.n_embd;
            lw.ffn_down = convert_weight_if_quantized(ml.ffn_down, n, &lw.ffn_type);
        }
        // MoE FFN weights (router + stacked experts + optional shared expert).
        // Expert stacks are quant-resident when the GGUF type has a fused-dequant GEMV
        // (q4_K/q6_K/q8_0/q5_0 in serve-q4k-gemv.cu, F16 native): raw blocks are uploaded
        // as-is. Other types dequant to GPU F16.
        // K-dims must be block-aligned: gate/up reduce over n_embd, down over n_ff_exp.
        if (lw.is_moe) {
            int dummy = 0;
            size_t ne_exps = (size_t)cfg_.n_embd * cfg_.n_ff_exp * cfg_.n_expert;
            lw.ffn_gate_inp  = convert_weight_if_quantized(ml.ffn_gate_inp,  (size_t)cfg_.n_embd * cfg_.n_expert, &dummy);

            // Upload a stacked expert tensor's raw quant blocks to GPU; returns GPU ptr.
            auto upload_raw = [&](ggml_tensor * t) -> const void * {
                size_t nbytes = ggml_nbytes(t);
                void * d = nullptr;
                if (cudaMalloc(&d, nbytes) != cudaSuccess) return nullptr;
                bool ih = ggml_backend_buffer_is_host(t->buffer);
                cudaMemcpy(d, ggml_get_data(t), nbytes,
                           ih ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
                f32_converted_bufs_.push_back((half *)d);
                return d;
            };
            auto blk_elems = [](int t) -> int {  // types with a moe_gemv_id kernel
                return (t == 12 || t == 13 || t == 14) ? 256 : (t == 8) ? 32 : (t == 1) ? 1 : 0;
            };
            int tg = ml.ffn_gate_exps->type, tu = ml.ffn_up_exps->type, td = ml.ffn_down_exps->type;
            bool ok_gu = blk_elems(tg) > 0 && blk_elems(tu) > 0 &&
                         cfg_.n_embd % std::max(blk_elems(tg), 1) == 0 &&
                         cfg_.n_embd % std::max(blk_elems(tu), 1) == 0;
            bool ok_d  = blk_elems(td) > 0 && cfg_.n_ff_exp % std::max(blk_elems(td), 1) == 0;

            if (ok_gu && ok_d) {
                lw.ffn_gate_exps = upload_raw(ml.ffn_gate_exps);
                lw.ffn_up_exps   = upload_raw(ml.ffn_up_exps);
                lw.ffn_down_exps = upload_raw(ml.ffn_down_exps);
                lw.exps_gate_qtype = tg;
                lw.exps_up_qtype   = tu;
                lw.exps_down_qtype = td;
                lw.exps_gate_stride = (int64_t)(ggml_nbytes(ml.ffn_gate_exps) / cfg_.n_expert);
                lw.exps_up_stride   = (int64_t)(ggml_nbytes(ml.ffn_up_exps)   / cfg_.n_expert);
                lw.exps_down_stride = (int64_t)(ggml_nbytes(ml.ffn_down_exps) / cfg_.n_expert);
                lw.exps_quant_resident = true;
                if (il == 0) fprintf(stderr, "[forward] MoE experts quant-RESIDENT: gate/up/down types %d/%d/%d (%.1f MiB/layer)\n",
                                     tg, tu, td, (double)(ggml_nbytes(ml.ffn_gate_exps) + ggml_nbytes(ml.ffn_up_exps) + ggml_nbytes(ml.ffn_down_exps)) / (1024 * 1024));
            } else {
                lw.ffn_gate_exps = convert_weight_if_quantized(ml.ffn_gate_exps, ne_exps, &dummy);
                lw.ffn_up_exps   = convert_weight_if_quantized(ml.ffn_up_exps,   ne_exps, &dummy);
                lw.ffn_down_exps = convert_weight_if_quantized(ml.ffn_down_exps, ne_exps, &dummy);
                if (il == 0) fprintf(stderr, "[forward] MoE experts dequanted to F16 (types %d/%d/%d, no resident kernel)\n",
                                     tg, tu, td);
            }
            if (cfg_.n_ff_shexp > 0 && ml.ffn_gate_shexp && ml.ffn_up_shexp && ml.ffn_down_shexp) {
                size_t ne_sh = (size_t)cfg_.n_embd * cfg_.n_ff_shexp;
                lw.ffn_gate_shexp = convert_weight_if_quantized(ml.ffn_gate_shexp, ne_sh, &dummy);
                lw.ffn_up_shexp   = convert_weight_if_quantized(ml.ffn_up_shexp,   ne_sh, &dummy);
                lw.ffn_down_shexp = convert_weight_if_quantized(ml.ffn_down_shexp, ne_sh, &dummy);
                if (ml.ffn_gate_inp_shexp) {
                    lw.ffn_gate_inp_shexp = convert_weight_if_quantized(ml.ffn_gate_inp_shexp, (size_t)cfg_.n_embd, &dummy);
                }
            }
            if (!lw.ffn_gate_inp || !lw.ffn_gate_exps || !lw.ffn_up_exps || !lw.ffn_down_exps) {
                fprintf(stderr, "[forward] ERROR: MoE weight load failed at layer %d\n", il);
            }
        }
        // DeltaNet / linear-attention weights (recurrent layers) -> GPU FP16
        if (lw.is_recurrent) {
            int S = cfg_.ssm_d_state, Hk = cfg_.ssm_n_group, Hv = cfg_.ssm_dt_rank;
            int key_dim = S * Hk;
            int value_dim = cfg_.ssm_d_inner;        // = S * Hv
            int conv_dim = key_dim * 2 + value_dim;  // conv_channels
            int head_v = cfg_.ssm_d_inner / Hv;      // = S
            int dummy = 0;
            if (ml.wqkv)      lw.wqkv      = convert_weight_if_quantized(ml.wqkv,      (size_t)cfg_.n_embd * conv_dim,  &dummy);
            if (ml.wqkv_gate) lw.wqkv_gate = convert_weight_if_quantized(ml.wqkv_gate, (size_t)cfg_.n_embd * value_dim, &dummy);
            if (ml.ssm_out)   lw.ssm_out   = convert_weight_if_quantized(ml.ssm_out,   (size_t)value_dim * cfg_.n_embd, &dummy);
            if (ml.ssm_beta)  lw.ssm_beta  = convert_weight_if_quantized(ml.ssm_beta,  (size_t)cfg_.n_embd * Hv,        &dummy);
            if (ml.ssm_alpha) lw.ssm_alpha = convert_weight_if_quantized(ml.ssm_alpha, (size_t)cfg_.n_embd * Hv,        &dummy);
            lw.dn_wtype = dummy;  // ggml_type of the DeltaNet matmul weights (BF16=30 stays BF16)
            if (ml.ssm_conv1d) lw.ssm_conv1d = convert_weight_to_gpu_f16(lw.ssm_conv1d, ml.ssm_conv1d, cfg_.ssm_d_conv * conv_dim);
            if (ml.ssm_dt)     lw.ssm_dt     = convert_weight_to_gpu_f16(lw.ssm_dt,   ml.ssm_dt,   Hv);
            if (ml.ssm_a)      lw.ssm_a      = convert_weight_to_gpu_f16(lw.ssm_a,    ml.ssm_a,    Hv);
            if (ml.ssm_norm)   lw.ssm_norm   = convert_weight_to_gpu_f16(lw.ssm_norm, ml.ssm_norm, head_v);
        }
        // Check for CUDA errors after each layer
        {
            cudaDeviceSynchronize();
            cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) {
                fprintf(stderr, "[forward] CUDA warning at layer %d: %s (wq_type=%d ffn_type=%d) — continuing\n",
                        il, cudaGetErrorString(ce), lw.wq_type, lw.ffn_type);
            }
            if (il == 0) fprintf(stderr, "[forward] layer 0 OK: wq_type=%d ffn_type=%d\n", lw.wq_type, lw.ffn_type);
        }
    }

    // Pre-convert biases to GPU FP16
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        auto & ml = model->layers[il];
        int dummy_type = 0;
        if (lw.bq && ml.wq_b) { lw.bq = convert_weight_if_quantized(ml.wq_b, cfg_.n_head * cfg_.head_dim, &dummy_type);
            if (il == 0) fprintf(stderr, "[forward] bias converted: bq=%p type_out=%d\n", lw.bq, dummy_type); }
        if (lw.bk && ml.wk_b) lw.bk = convert_weight_if_quantized(ml.wk_b, cfg_.n_head_kv * cfg_.head_dim, &dummy_type);
        if (lw.bv && ml.wv_b) lw.bv = convert_weight_if_quantized(ml.wv_b, cfg_.n_head_kv * cfg_.head_dim, &dummy_type);
    }

    // Pre-convert norm weights to GPU FP16
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        auto & ml = model->layers[il];
        if (lw.attn_norm && ml.attn_norm) {
            lw.attn_norm = convert_weight_to_gpu_f16(lw.attn_norm, ml.attn_norm, cfg_.n_embd);
        }
        ggml_tensor * ffn_norm_src2 = ml.ffn_norm ? ml.ffn_norm : ml.attn_post_norm;
        if (lw.ffn_norm && ffn_norm_src2) {
            lw.ffn_norm = convert_weight_to_gpu_f16(lw.ffn_norm, ffn_norm_src2, cfg_.n_embd);
        }
        if (lw.q_norm && ml.attn_q_norm) {
            // Per-head QK-norm (Qwen3/3.5): weight is [head_dim]. Full-width QK-norm
            // (OLMoE/OLMo2): weight spans the whole Q projection [n_head*head_dim];
            // detected by tensor width and normalized as a single group.
            int qn = (int)ml.attn_q_norm->ne[0];
            lw.qk_norm_full = (qn != cfg_.head_dim);
            lw.q_norm = convert_weight_to_gpu_f16(lw.q_norm, ml.attn_q_norm, qn);
            if (lw.qk_norm_full && il == 0) {
                fprintf(stderr, "[forward] QK-norm: full-width (%d) RMSNorm (OLMoE-style)\n", qn);
            }
        }
        if (lw.k_norm && ml.attn_k_norm) {
            lw.k_norm = convert_weight_to_gpu_f16(lw.k_norm, ml.attn_k_norm, (int)ml.attn_k_norm->ne[0]);
        }
    }
    if (output_norm_ && model->output_norm) {
        output_norm_ = convert_weight_to_gpu_f16(output_norm_, model->output_norm, cfg_.n_embd);
    }
    // Convert output weight if on CPU (may be quantized in Q4_K_M models)
    if (output_ && model->output) {
        size_t n = (size_t)cfg_.vocab_size * cfg_.n_embd;
        int src_type = model->output->type;
        bool is_host = ggml_backend_buffer_is_host(model->output->buffer);
        if (src_type == 0 || src_type == 1) {
            // F32 or F16: copy to GPU
            size_t elem_size = (src_type == 0) ? sizeof(float) : sizeof(half);
            if (is_host) {
                void * d_buf = nullptr;
                cudaMalloc(&d_buf, n * elem_size);
                cudaMemcpy(d_buf, output_, n * elem_size, cudaMemcpyHostToDevice);
                output_ = d_buf;
                fprintf(stderr, "[forward] output weight: type=%d, host→GPU (%.1f MiB)\n",
                        src_type, (double)(n * elem_size) / (1024*1024));
            }
        } else {
            // Quantized output: dequant to FP16
            int dummy_type = 0;
            output_ = convert_weight_if_quantized(model->output, n, &dummy_type);
            fprintf(stderr, "[forward] output weight: quant type %d → FP16\n", src_type);
            // MoE models: also keep the raw quant lm_head; the decode logits GEMV reads
            // it at ~half the bytes of the F16 copy.
            int be = (src_type == 12 || src_type == 14) ? 256 : (src_type == 8) ? 32 : 0;
            if (cfg_.has_moe && be > 0 && cfg_.n_embd % be == 0) {
                size_t nb = ggml_nbytes(model->output);
                void * d = nullptr;
                if (cudaMalloc(&d, nb) == cudaSuccess) {
                    cudaMemcpy(d, ggml_get_data(model->output), nb,
                               is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
                    f32_converted_bufs_.push_back((half *)d);
                    output_q_ = d;
                    output_qtype_ = src_type;
                    fprintf(stderr, "[forward] lm_head quant-RESIDENT for decode logits (type %d, %.0f MiB)\n",
                            src_type, (double)nb / (1024 * 1024));
                }
            }
        }
    }
    // Convert tok_embd to FP16 on GPU
    // The embedding tensor may be on CPU (ggml puts it on CPU_Mapped buffer)
    if (tok_embd_ && model->tok_embd) {
        size_t n = (size_t)cfg_.vocab_size * cfg_.n_embd;
        bool is_host = ggml_backend_buffer_is_host(model->tok_embd->buffer);
        int src_type = model->tok_embd->type;

        // Quant-resident embedding: for K-quant tables with a dequant-gather kernel
        // (Q4_K/Q5_K/Q6_K, n_embd block-aligned) keep the raw blocks in VRAM (~half the F16
        // bytes) and dequant per row at embed time. Opt out with ALPHA_FI_EMBED_F16; otherwise
        // falls through to the F16 path. Requires a separate lm_head (model->output): with tied
        // embeddings tok_embd_ also serves as the lm_head F16 table, so it must stay F16.
        if ((src_type == 8 || src_type == 12 || src_type == 13 || src_type == 14) && cfg_.n_embd % 256 == 0 &&
            model->output != nullptr && !getenv("ALPHA_FI_EMBED_F16")) {
            size_t type_size = ggml_type_size(model->tok_embd->type);
            size_t block_size = ggml_blck_size(model->tok_embd->type);
            size_t raw_bytes = (n / block_size) * type_size;
            void * d_raw = nullptr;
            if (cudaMalloc(&d_raw, raw_bytes) == cudaSuccess) {
                cudaMemcpy(d_raw, tok_embd_, raw_bytes,
                           is_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice);
                tok_embd_ = d_raw;
                tok_embd_qtype_ = src_type;
                f32_converted_bufs_.push_back((half *)d_raw);
                fprintf(stderr, "[forward] tok_embd quant-RESIDENT (type %d, %.1f MiB, dequant-gather)\n",
                        src_type, (double)raw_bytes / (1024 * 1024));
                goto tok_embd_done;
            }
            // OOM on the raw upload: fall through to the F16 path below.
        }
      {
        half * h_buf = nullptr;
        cudaMalloc(&h_buf, n * sizeof(half));

        if (src_type == 0) {  // F32
            if (is_host) {
                // CPU->GPU: copy F32 to temp GPU buffer, then convert
                float * d_tmp = nullptr;
                cudaMalloc(&d_tmp, n * sizeof(float));
                cudaMemcpy(d_tmp, tok_embd_, n * sizeof(float), cudaMemcpyHostToDevice);
                launch_f32_to_f16(h_buf, d_tmp, (int64_t)n, 0);
                cudaDeviceSynchronize();
                cudaFree(d_tmp);
            } else {
                launch_f32_to_f16(h_buf, (const float *)tok_embd_, (int64_t)n, 0);
                cudaDeviceSynchronize();
            }
        } else if (src_type == 1) {  // F16
            if (is_host) {
                cudaMemcpy(h_buf, tok_embd_, n * sizeof(half), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(h_buf, tok_embd_, n * sizeof(half), cudaMemcpyDeviceToDevice);
            }
        } else {
            // Quantized embedding (Q4_K, Q6_K, Q5_0, etc.): CPU dequant -> CPU f32->f16 -> GPU FP16.
            // This runs after the experts already fill VRAM, so a full n*4 F32 GPU temp could
            // OOM. Convert on the CPU and upload only the F16 buffer.
            auto traits = ggml_get_type_traits(model->tok_embd->type);
            if (traits->to_float) {
                size_t type_size = ggml_type_size(model->tok_embd->type);
                size_t block_size = ggml_blck_size(model->tok_embd->type);
                size_t raw_bytes = (n / block_size) * type_size;
                const void * host_src = tok_embd_;
                std::vector<uint8_t> h_packed;
                if (!is_host) {
                    h_packed.resize(raw_bytes);
                    cudaMemcpy(h_packed.data(), tok_embd_, raw_bytes, cudaMemcpyDeviceToHost);
                    host_src = h_packed.data();
                }
                std::vector<float> h_f32(n);
                traits->to_float(host_src, h_f32.data(), (int64_t)n);
                std::vector<uint16_t> h_f16v(n);
                ggml_fp32_to_fp16_row(h_f32.data(), (ggml_fp16_t *)h_f16v.data(), (int64_t)n);
                cudaMemcpy(h_buf, h_f16v.data(), n * sizeof(half), cudaMemcpyHostToDevice);
                fprintf(stderr, "[forward] tok_embd: quant type %d → CPU dequant → FP16 (no GPU f32 temp)\n", src_type);
            } else {
                fprintf(stderr, "[forward] WARNING: tok_embd type %d not handled, zeroing\n", src_type);
                cudaMemset(h_buf, 0, n * sizeof(half));
            }
        }

        tok_embd_ = h_buf;
        f32_converted_bufs_.push_back(h_buf);
        fprintf(stderr, "[forward] tok_embd: type=%d, host=%d → GPU FP16 (%.1f MiB)\n",
                src_type, is_host, (double)(n * sizeof(half)) / (1024 * 1024));
      }
      tok_embd_done: ;
    }

    // Pre-allocate batched gemm pointer arrays (K+V and gate+up)
    batched_ptrs_.resize(cfg_.n_layers);
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        auto & bp = batched_ptrs_[il];
        if (!lw.wk || !lw.wv || lw.wq_type != 1) continue;

        // K+V batched pointers
        const half * h_kv_B[2] = { (const half *)lw.wk, (const half *)lw.wv };
        const half * h_kv_A[2] = { hidden_buf_, hidden_buf_ };
        half * h_kv_C[2] = { k_buf_, v_buf_ };
        cudaMalloc(&bp.d_kv_B, 2 * sizeof(const half *));
        cudaMalloc(&bp.d_kv_A, 2 * sizeof(const half *));
        cudaMalloc(&bp.d_kv_C, 2 * sizeof(half *));
        cudaMemcpy(bp.d_kv_B, h_kv_B, 2 * sizeof(const half *), cudaMemcpyHostToDevice);
        cudaMemcpy(bp.d_kv_A, h_kv_A, 2 * sizeof(const half *), cudaMemcpyHostToDevice);
        cudaMemcpy(bp.d_kv_C, h_kv_C, 2 * sizeof(half *), cudaMemcpyHostToDevice);

        // gate+up batched pointers
        if (lw.ffn_gate && lw.ffn_up && lw.ffn_type == 1) {
            const half * h_ffn_B[2] = { (const half *)lw.ffn_gate, (const half *)lw.ffn_up };
            const half * h_ffn_A[2] = { hidden_buf_, hidden_buf_ };
            half * h_ffn_C[2] = { ffn_buf1_, ffn_buf2_ };
            cudaMalloc(&bp.d_ffn_B, 2 * sizeof(const half *));
            cudaMalloc(&bp.d_ffn_A, 2 * sizeof(const half *));
            cudaMalloc(&bp.d_ffn_C, 2 * sizeof(half *));
            cudaMemcpy(bp.d_ffn_B, h_ffn_B, 2 * sizeof(const half *), cudaMemcpyHostToDevice);
            cudaMemcpy(bp.d_ffn_A, h_ffn_A, 2 * sizeof(const half *), cudaMemcpyHostToDevice);
            cudaMemcpy(bp.d_ffn_C, h_ffn_C, 2 * sizeof(half *), cudaMemcpyHostToDevice);
        }
    }
    fprintf(stderr, "[forward] batched gemm pointers: %d layers\n", cfg_.n_layers);

    // Fuse QKV and gate+up weights to reduce cuBLAS calls
    // QKV: concat Wq[n_embd, q_dim] + Wk[n_embd, kv_dim] + Wv[n_embd, kv_dim]
    //   -> Wqkv[n_embd, q_dim + 2*kv_dim]
    // gate+up: concat Wgate[n_embd, n_ff] + Wup[n_embd, n_ff]
    //   -> Wgate_up[n_embd, 2*n_ff]
    int q_dim = cfg_.n_head * cfg_.head_dim;
    int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
    int qkv_dim = q_dim + 2 * kv_dim;
    int gate_up_dim = 2 * cfg_.n_ff;
    int n_fused = 0;

    // Honor ALPHA_NO_FUSED before allocation: the fused buffers (wqkv_fused / ffn_gate_up_fused /
    // dn_proj_fused) duplicate the per-tensor weights (~0.5x model size). Fusion is roughly
    // throughput-neutral, so skipping it frees VRAM for more KV slots on big models. The unfused
    // paths read the per-tensor weights when the fused buffers are absent.
    if (getenv("ALPHA_NO_FUSED")) use_fused_kernels_ = false;
    if (use_fused_kernels_) {
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        if (!lw.wq || !lw.wk || !lw.wv) continue;
        if (lw.is_gated_attn) continue;  // gated attn has 2x-wide wq; uses its own path, not fused QKV
        // Fuse QKV: copy Q, K, V weight rows interleaved into one matrix
        // ggml stores weights as [out_dim, in_dim] row-major
        // Fused: rows 0..q_dim-1 = Wq, rows q_dim..q_dim+kv_dim-1 = Wk, etc.
        lw.qkv_out_dim = qkv_dim;
        cudaMalloc(&lw.wqkv_fused, (size_t)cfg_.n_embd * qkv_dim * sizeof(half));
        // Copy Wq rows
        cudaMemcpy(lw.wqkv_fused,
                   lw.wq, (size_t)cfg_.n_embd * q_dim * sizeof(half),
                   cudaMemcpyDeviceToDevice);
        // Copy Wk rows after Wq
        cudaMemcpy(lw.wqkv_fused + (size_t)cfg_.n_embd * q_dim,
                   lw.wk, (size_t)cfg_.n_embd * kv_dim * sizeof(half),
                   cudaMemcpyDeviceToDevice);
        // Copy Wv rows after Wk
        cudaMemcpy(lw.wqkv_fused + (size_t)cfg_.n_embd * (q_dim + kv_dim),
                   lw.wv, (size_t)cfg_.n_embd * kv_dim * sizeof(half),
                   cudaMemcpyDeviceToDevice);

        // Fuse gate+up
        if (lw.ffn_gate && lw.ffn_up) {
            lw.gate_up_out_dim = gate_up_dim;
            cudaMalloc(&lw.ffn_gate_up_fused, (size_t)cfg_.n_embd * gate_up_dim * sizeof(half));
            cudaMemcpy(lw.ffn_gate_up_fused,
                       lw.ffn_gate, (size_t)cfg_.n_embd * cfg_.n_ff * sizeof(half),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(lw.ffn_gate_up_fused + (size_t)cfg_.n_embd * cfg_.n_ff,
                       lw.ffn_up, (size_t)cfg_.n_embd * cfg_.n_ff * sizeof(half),
                       cudaMemcpyDeviceToDevice);
        }
        n_fused++;
    }

    // FFN gate+up fusion for recurrent (DeltaNet) layers: they have FFN but no wq,
    // so the loop above skipped them. Give them the fused FFN path so the shared FFN
    // tail doesn't fall through to the (unset) batched-ptr path.
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        if (!lw.is_recurrent || lw.ffn_gate_up_fused) continue;
        if (lw.ffn_gate && lw.ffn_up) {
            lw.gate_up_out_dim = gate_up_dim;
            cudaMalloc(&lw.ffn_gate_up_fused, (size_t)cfg_.n_embd * gate_up_dim * sizeof(half));
            cudaMemcpy(lw.ffn_gate_up_fused, lw.ffn_gate,
                       (size_t)cfg_.n_embd * cfg_.n_ff * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(lw.ffn_gate_up_fused + (size_t)cfg_.n_embd * cfg_.n_ff, lw.ffn_up,
                       (size_t)cfg_.n_embd * cfg_.n_ff * sizeof(half), cudaMemcpyDeviceToDevice);
        }
        // Fuse the 4 DeltaNet input projections (wqkv|wqkv_gate|ssm_beta|ssm_alpha), which all read
        // the same hidden, into one GEMM (concat along output dim), then split with a cheap kernel.
        // Removes the two tiny N=Hv GEMM launches per layer.
        if (lw.wqkv && lw.wqkv_gate && lw.ssm_beta && lw.ssm_alpha && lw.dn_wtype == 1) {
            int Hv = cfg_.ssm_dt_rank, value_dim = cfg_.ssm_d_inner;
            int conv_dim = cfg_.ssm_d_state * cfg_.ssm_n_group * 2 + value_dim;
            lw.dn_proj_n = conv_dim + value_dim + Hv + Hv;
            size_t ne = (size_t)cfg_.n_embd;
            cudaMalloc(&lw.dn_proj_fused, (size_t)lw.dn_proj_n * ne * sizeof(half));
            half * d = lw.dn_proj_fused;
            cudaMemcpy(d,                          lw.wqkv,      (size_t)conv_dim  * ne * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d + (size_t)conv_dim * ne,  lw.wqkv_gate, (size_t)value_dim * ne * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d + (size_t)(conv_dim + value_dim) * ne,      lw.ssm_beta,  (size_t)Hv * ne * sizeof(half), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d + (size_t)(conv_dim + value_dim + Hv) * ne, lw.ssm_alpha, (size_t)Hv * ne * sizeof(half), cudaMemcpyDeviceToDevice);
        }
    }
    } // end if (use_fused_kernels_): fused-buffer allocation skipped entirely when unfused

    // Debug / comparison switches
    if (getenv("ALPHA_NO_FUSED")) {
        use_fused_kernels_ = false;
        fprintf(stderr, "[forward] ALPHA_NO_FUSED set — using unfused kernels\n");
    }
    if (getenv("ALPHA_NO_LT")) {
        use_lt_matmul_ = false;
        fprintf(stderr, "[forward] ALPHA_NO_LT set — using cublasGemmEx (no auto-tuning)\n");
    }
    if (getenv("ALPHA_NO_FI_ATTN")) {
        use_flashinfer_attn_ = false;
        fprintf(stderr, "[forward] ALPHA_NO_FI_ATTN set — using simple attention kernel\n");
    } else if (!flashinfer_attn_supported(cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim)) {
        use_flashinfer_attn_ = false;
        fprintf(stderr, "[forward] FlashInfer attn unsupported for GQA %d/%d head_dim=%d — using simple kernel\n",
                cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim);
    }

    fprintf(stderr, "[forward] fused %d layers: QKV [%d→%d] + gate_up [%d→%d] (saves %d cuBLAS calls/step)\n",
            n_fused, q_dim, qkv_dim, cfg_.n_ff, gate_up_dim, n_fused * 3);
    // Check for any CUDA errors from weight conversions
    cudaDeviceSynchronize();
    cudaError_t init_err = cudaGetLastError();
    if (init_err != cudaSuccess) {
        fprintf(stderr, "[forward] CUDA warning after weight init: %s — resetting\n", cudaGetErrorString(init_err));
        cudaGetLastError();  // clear the error
    }

    // DeltaNet recurrent-state pool: assign per-layer ordinals + allocate slabs.
    if (cfg_.has_recurrent) {
        // CUDA graphs are used for recurrent decode. Two non-capturable pieces are handled
        // outside the captured region in forward_decode: (1) the DeltaNet seq_ids host->device
        // copy, and (2) graph-batch padding: padding rows [batch,bucket) route to a scratch state
        // slot (recur_scratch_slot_) and a scratch KV seq/page (kv_pool_->scratch_seq_id), so their
        // discarded reads/writes cannot corrupt real sequences. Both scratch indices equal
        // max_seqs, so one padded seq-id array serves both. Disable with ALPHA_DN_NO_GRAPH.
        recur_n_layers_ = 0;
        for (int il = 0; il < cfg_.n_layers; il++)
            if (layer_weights_[il].is_recurrent) layer_weights_[il].recur_ord = recur_n_layers_++;
        alloc_recurrent_state(kv_pool_->max_seqs);
        // On by default. graphs_active in forward_decode decides per step whether batched decode
        // (bucket>1) also uses graphs (ALPHA_NO_GRAPH_BATCH). Opt out with
        // ALPHA_DN_NO_GRAPH (ALPHA_DN_GRAPH is accepted as a no-op).
        if (getenv("ALPHA_DN_NO_GRAPH")) {
            use_cuda_graph_ = false;
            fprintf(stderr, "[forward] ALPHA_DN_NO_GRAPH: recurrent CUDA graphs disabled\n");
        } else {
            fprintf(stderr, "[forward] recurrent CUDA graphs ON (single-stream/bucket==1 only; concurrent=eager)\n");
        }
    }

    // MoE graph policy: when every MoE layer is quant-resident, the decode MoE step is
    // fully device-routed (GPU top-K + gemv-id, no host sync, fixed shapes per batch), so
    // CUDA graphs stay enabled. Any F16-fallback layer needs host routing (per-layer
    // stream sync, dynamic grouping), which disables graphs for the whole model.
    if (cfg_.has_moe) {
        cfg_.moe_all_quant = true;
        for (int il = 0; il < cfg_.n_layers; il++) {
            if (layer_weights_[il].is_moe && !layer_weights_[il].exps_quant_resident) {
                cfg_.moe_all_quant = false;
                break;
            }
        }
        // Host-grouped dispatch syncs the stream and is not capturable. Graphs replay only at
        // bucket==1, which always takes the device route when the experts are quant-resident.
        if (cfg_.moe_all_quant) {
            fprintf(stderr, "[forward] MoE model (%d experts, top-%d, n_ff_exp=%d, shexp=%d): quant-resident, device-routed decode — CUDA graphs enabled\n",
                    cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp, cfg_.n_ff_shexp);
        } else {
            use_cuda_graph_ = false;
            fprintf(stderr, "[forward] MoE model (%d experts, top-%d, n_ff_exp=%d, shexp=%d): F16 host-routed dispatch — CUDA graphs disabled\n",
                    cfg_.n_expert, cfg_.n_expert_used, cfg_.n_ff_exp, cfg_.n_ff_shexp);
        }
    }

    if (getenv("ALPHA_NO_GRAPH")) { use_cuda_graph_ = false; fprintf(stderr, "[forward] ALPHA_NO_GRAPH: CUDA graphs disabled (all models)\n"); }

    fprintf(stderr, "[forward] initialized: %d layers, max_batch=%d, weights from GGUF (%d conversions), fused=%s\n",
            cfg_.n_layers, max_batch, (int)f32_converted_bufs_.size(),
            use_fused_kernels_ ? "yes" : "no");
    return true;
}

void ForwardEngine::alloc_recurrent_state(int max_seqs) {
    int S = cfg_.ssm_d_state, Hk = cfg_.ssm_n_group, Hv = cfg_.ssm_dt_rank;
    int value_dim = cfg_.ssm_d_inner;
    int conv_dim  = S * Hk * 2 + value_dim;   // conv_channels
    recur_conv_size_ = (cfg_.ssm_d_conv - 1) * conv_dim;  // f16 elems per seq
    recur_ssm_size_  = S * S * Hv;                          // f16 elems per seq
    recur_max_seqs_  = max_seqs;
    recur_scratch_slot_ = max_seqs;          // one extra slot: CUDA-graph padding rows write here
    int n_slots = max_seqs + 1;
    recur_conv_.assign(recur_n_layers_, nullptr);
    recur_ssm_.assign(recur_n_layers_, nullptr);
    for (int o = 0; o < recur_n_layers_; o++) {
        cudaMalloc(&recur_conv_[o], (size_t)n_slots * recur_conv_size_ * sizeof(half));
        cudaMalloc(&recur_ssm_[o],  (size_t)n_slots * recur_ssm_size_  * sizeof(half));
        cudaMemset(recur_conv_[o], 0, (size_t)n_slots * recur_conv_size_ * sizeof(half));
        cudaMemset(recur_ssm_[o],  0, (size_t)n_slots * recur_ssm_size_  * sizeof(half));
    }
    fprintf(stderr, "[forward] linear-attn state: %d layers, conv=%d f16 + ssm=%d f16 per seq, max_seqs=%d (%.0f MiB)\n",
            recur_n_layers_, recur_conv_size_, recur_ssm_size_, max_seqs,
            (double)recur_n_layers_ * max_seqs * (recur_conv_size_ * 2.0 + recur_ssm_size_ * 2.0) / (1024 * 1024));
}

// One DeltaNet/linear-attention layer. Reads hidden_buf_ (post attn_norm), writes hidden_buf_
// (the layer's attn-equivalent output). Reads/updates the per-seq recurrent state pool.
// is_prefill: batch = T tokens of one fresh sequence, processed sequentially (state reset first).
// decode: batch = #sequences, 1 token each.
void ForwardEngine::forward_linear_attn_layer(int il, int batch, const int32_t * seq_ids_cpu,
                                           bool is_prefill, cudaStream_t stream) {
    auto & lw = layer_weights_[il];
    int S = cfg_.ssm_d_state, Hk = cfg_.ssm_n_group, Hv = cfg_.ssm_dt_rank;
    int key_dim = S * Hk, value_dim = cfg_.ssm_d_inner, conv_dim = key_dim * 2 + value_dim;
    int ord = lw.recur_ord;
    half  * conv_state = recur_conv_[ord];
    half * ssm_state  = recur_ssm_[ord];

    // Decode: dn_seq_ids_ is pre-filled once per step by forward_decode (outside the CUDA
    // graph, host-source copy). Prefill (not graphed) copies its own seq_ids here.
    if (is_prefill)
        cudaMemcpyAsync(dn_seq_ids_, seq_ids_cpu, batch * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    if (is_prefill) {  // fresh sequence: zero its state for this layer (single-chunk prefill)
        int sid0 = seq_ids_cpu[0];
        cudaMemsetAsync(conv_state + (int64_t)sid0 * recur_conv_size_, 0, recur_conv_size_ * sizeof(half),  stream);
        cudaMemsetAsync(ssm_state  + (int64_t)sid0 * recur_ssm_size_,  0, recur_ssm_size_  * sizeof(half), stream);
    }

    static int dn_chk = getenv("ALPHA_DN_CHK") ? atoi(getenv("ALPHA_DN_CHK")) : 0;
    auto CHK = [&](const char * tag) {
        if (!dn_chk) return;
        cudaError_t e = cudaStreamSynchronize(stream);
        cudaError_t e2 = cudaGetLastError();
        if (e || e2) fprintf(stderr, "[dn L%d CHK] %s: sync=%s last=%s (S=%d Hk=%d Hv=%d key=%d val=%d conv=%d ssm_sz=%d conv_sz=%d batch=%d)\n",
            il, tag, cudaGetErrorString(e), cudaGetErrorString(e2), S, Hk, Hv, key_dim, value_dim, conv_dim, recur_ssm_size_, recur_conv_size_, batch);
        else fprintf(stderr, "[dn L%d CHK] %s: OK\n", il, tag);
    };

    // Batched input projections. When the 4 weights are pre-concatenated (dn_proj_fused),
    // do one GEMM into dn_fused_ [batch, dn_proj_n] and then split.
    if (lw.dn_proj_fused) {
        matmul((half *)dn_fused_, hidden_buf_, lw.dn_proj_fused, batch, lw.dn_proj_n, cfg_.n_embd, lw.dn_wtype, stream);
        launch_dn_proj_split((const half *)dn_fused_, dn_qkv_, dn_z_, dn_beta_, dn_g_,
                             conv_dim, value_dim, Hv, batch, stream);
    } else {
        matmul((half *)dn_qkv_,  hidden_buf_, lw.wqkv,      batch, conv_dim,  cfg_.n_embd, lw.dn_wtype, stream);
        matmul((half *)dn_z_,    hidden_buf_, lw.wqkv_gate, batch, value_dim, cfg_.n_embd, lw.dn_wtype, stream);
        matmul((half *)dn_beta_, hidden_buf_, lw.ssm_beta,  batch, Hv,        cfg_.n_embd, lw.dn_wtype, stream);
        matmul((half *)dn_g_,    hidden_buf_, lw.ssm_alpha, batch, Hv,        cfg_.n_embd, lw.dn_wtype, stream);
    }
    CHK("projections");
    // Fused path (default): fold the gate computation and q/k L2-norm into delta_rule (one kernel
    // instead of gates + 2x l2norm + delta_rule), like vLLM's single-kernel recurrence; avoids the
    // q/k/g/beta DRAM round-trips. ALPHA_DN_FUSED=0 restores the split path.
    static const int dn_fused = []{ const char * e = getenv("ALPHA_DN_FUSED"); return (e && !atoi(e)) ? 0 : 1; }();
    if (!dn_fused) {
        launch_la_gates(dn_beta_, dn_g_, (const half *)lw.ssm_dt, (const half *)lw.ssm_a, Hv, batch, stream);
        CHK("gates");
    }

    // Recurrent part (conv + delta rule), sequential across a sequence's tokens.
    int steps = is_prefill ? batch : 1;
    int per   = is_prefill ? 1 : batch;
    for (int t = 0; t < steps; t++) {
        int64_t off = is_prefill ? t : 0;
        const int32_t * sids = dn_seq_ids_ + off;
        launch_la_conv1d(dn_conv_ + off * conv_dim, dn_qkv_ + off * conv_dim, conv_state,
                               (const half *)lw.ssm_conv1d, cfg_.ssm_d_conv, conv_dim, recur_conv_size_, sids, per, stream);
        CHK("conv1d");
        if (dn_fused) {
            // conv output is raw silu'd q|k|v; delta_rule_fused does L2-norm(q,k)+gates internally.
            launch_delta_rule_fused(dn_o_ + off * value_dim,
                                    dn_conv_ + off * conv_dim,
                                    dn_g_ + off * Hv, dn_beta_ + off * Hv,
                                    (const half *)lw.ssm_dt, (const half *)lw.ssm_a,
                                    ssm_state, sids, S, Hk, Hv, key_dim, value_dim, recur_ssm_size_, conv_dim,
                                    cfg_.rms_norm_eps, per, stream);
            CHK("delta_rule_fused");
        } else {
            launch_l2_norm(dn_conv_ + off * conv_dim, S, Hk, conv_dim, 0,       per, cfg_.rms_norm_eps, stream);
            launch_l2_norm(dn_conv_ + off * conv_dim, S, Hk, conv_dim, key_dim, per, cfg_.rms_norm_eps, stream);
            CHK("l2norm");
            launch_delta_rule(dn_o_ + off * value_dim, dn_conv_ + off * conv_dim, dn_g_ + off * Hv, dn_beta_ + off * Hv,
                              ssm_state, sids, S, Hk, Hv, key_dim, value_dim, recur_ssm_size_, conv_dim, per, stream);
            CHK("delta_rule");
        }
    }

    launch_gated_rmsnorm(dn_y_, dn_o_, dn_z_, (const half *)lw.ssm_norm, S, Hv, value_dim, batch, cfg_.rms_norm_eps, stream);
    CHK("gated_rmsnorm");
    matmul(hidden_buf_, dn_y_, lw.ssm_out, batch, cfg_.n_embd, value_dim, lw.dn_wtype, stream);
    CHK("ssm_out");
}

// Qwen3.5/Qwen3Next gated-attention QKV: produces q_buf_/k_buf_/v_buf_ (RoPE'd) + gate_buf_.
// Input: hidden_buf_ = post attn_norm hidden [effective_batch, n_embd].
void ForwardEngine::forward_gated_attn_qkv(int il, int effective_batch,
                                           const int32_t * positions, cudaStream_t stream) {
    auto & lw = layer_weights_[il];
    const int QD = cfg_.n_head * cfg_.head_dim;         // query dim (= attention out dim)
    const int KV = cfg_.n_head_kv * cfg_.head_dim;
    // Q+gate projection (wq output = 2*QD, [query|gate] interleaved per head)
    matmul((half *)qg_buf_, hidden_buf_, lw.wq, effective_batch, 2 * QD, cfg_.n_embd, lw.wq_type, stream);
    launch_deinterleave_qgate(q_buf_, gate_buf_, qg_buf_, cfg_.head_dim, cfg_.n_head, effective_batch, stream);
    // K, V projections
    matmul(k_buf_, hidden_buf_, lw.wk, effective_batch, KV, cfg_.n_embd, lw.wq_type, stream);
    matmul(v_buf_, hidden_buf_, lw.wv, effective_batch, KV, cfg_.n_embd, lw.wq_type, stream);
    // Optional attention biases (Qwen3.5 dense 0.8B has none)
    if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, QD, effective_batch, stream);
    if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, KV, effective_batch, stream);
    if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, KV, effective_batch, stream);
    // QK-norm (per-head RMSNorm over head_dim), before RoPE
    if (lw.q_norm) {
        launch_qk_norm(q_buf_, (const half *)lw.q_norm, cfg_.n_head,    cfg_.head_dim, QD, 0, effective_batch, cfg_.rms_norm_eps, stream);
        launch_qk_norm(k_buf_, (const half *)lw.k_norm, cfg_.n_head_kv, cfg_.head_dim, KV, 0, effective_batch, cfg_.rms_norm_eps, stream);
    }
    // Partial RoPE (n_rot dims only), NEOX split-half.
    launch_partial_rope(q_buf_, k_buf_, positions, cfg_.head_dim, cfg_.n_rot,
                        cfg_.n_head, cfg_.n_head_kv, effective_batch,
                        cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
}

// MoE FFN (routed experts)
// Reads hidden_buf_ = post-ffn_norm hidden [batch, n_embd]; writes hidden_buf_ = MoE output.
// Math (matches ggml build_moe_ffn for softmax gating, norm_w=false, as in qwen2moe/olmoe):
//   logits = x * gate_inp;  probs = softmax(logits) over all n_expert
//   top-K experts by prob; weights = probs of the selected experts (not renormalized)
//   y = sum_k w_k * down_k( silu(gate_k(x)) * up_k(x) )   [FP32 accumulation]
//   + shared expert (qwen2moe): y += sigmoid(x*gate_inp_shexp) * shexp_ffn(x)
//
// Two dispatch paths:
//  A. Device-routed (quant-resident experts, small batch, i.e. every decode step): GPU
//     top-K -> per-(token,k) gemv-id on raw quant blocks -> combine. No host syncs and
//     fixed shapes per batch, so it is CUDA-graph capturable. No weight reuse across
//     tokens, which is fine at decode where tokens rarely share experts.
//  B. Host-grouped (F16 fallback, or large prefill batches): router readback + top-K
//     on host, tokens grouped per unique expert so each expert's weights are read
//     once per group (matters at prefill where tokens >> experts). Syncs the stream.
void ForwardEngine::forward_moe_ffn(int il, int batch, cudaStream_t stream) {
    auto & lw = layer_weights_[il];
    const int n_expert = cfg_.n_expert;
    const int K = cfg_.n_expert_used > 0 ? std::min(cfg_.n_expert_used, n_expert) : 1;
    const int n_embd = cfg_.n_embd;
    const int n_ff_exp = cfg_.n_ff_exp;

    // Device route while batch <= this (per-pair weight reads beat grouping until tokens
    // start sharing experts); larger batches (prefill chunks) take the grouped path.
    static const int group_thresh = 32;

    // 1. Router logits: [batch, n_expert]. hidden_buf_ stays intact until finalize;
    //    experts and the shared expert read it directly (no input copy needed).
    matmul(moe_logits_, hidden_buf_, lw.ffn_gate_inp, batch, n_expert, n_embd, 1, stream);

    const int n_pairs = batch * K;

    if (lw.exps_quant_resident && batch <= group_thresh) {
        // Path A: device-routed (no host sync)
        launch_moe_topk(d_moe_idx_, d_moe_w_, moe_logits_, n_expert, K, batch, stream);
        if (moe_dp4a_type(lw.exps_gate_qtype) && moe_dp4a_type(lw.exps_up_qtype) &&
            moe_dp4a_type(lw.exps_down_qtype)) {
            // MMVQ-style: int8-quantize hidden once -> gate/up int-dot; requantize the
            // silu output (per-pair rows) -> down int-dot. Reusing moe_act_ is safe:
            // gate/up consumed it before the second quantize (stream-ordered).
            launch_moe_q8_act(moe_act_, hidden_buf_, batch, n_embd, stream);
            launch_moe_gemv_id_q8(moe_ff1_, n_ff_exp, moe_act_, lw.ffn_gate_exps, d_moe_idx_,
                                  n_pairs, K, lw.exps_gate_stride, n_ff_exp, n_embd,
                                  lw.exps_gate_qtype, stream);
            launch_moe_gemv_id_q8(moe_ff2_, n_ff_exp, moe_act_, lw.ffn_up_exps, d_moe_idx_,
                                  n_pairs, K, lw.exps_up_stride, n_ff_exp, n_embd,
                                  lw.exps_up_qtype, stream);
            silu_mul(moe_ff1_, moe_ff1_, moe_ff2_, n_ff_exp, n_pairs, stream);
            launch_moe_q8_act(moe_act_, moe_ff1_, n_pairs, n_ff_exp, stream);
            launch_moe_gemv_id_q8(moe_dn_, n_embd, moe_act_, lw.ffn_down_exps, d_moe_idx_,
                                  n_pairs, 1, lw.exps_down_stride, n_embd, n_ff_exp,
                                  lw.exps_down_qtype, stream);
        } else {
        launch_moe_gemv_id(moe_ff1_, hidden_buf_, lw.ffn_gate_exps, d_moe_idx_, n_pairs, K,
                           lw.exps_gate_stride, n_ff_exp, n_embd, lw.exps_gate_qtype, stream);
        launch_moe_gemv_id(moe_ff2_, hidden_buf_, lw.ffn_up_exps, d_moe_idx_, n_pairs, K,
                           lw.exps_up_stride, n_ff_exp, n_embd, lw.exps_up_qtype, stream);
        silu_mul(moe_ff1_, moe_ff1_, moe_ff2_, n_ff_exp, n_pairs, stream);
        // down: input rows are per pair (moe_ff1_[p]), not per token, so k_per_tok=1
        launch_moe_gemv_id(moe_dn_, moe_ff1_, lw.ffn_down_exps, d_moe_idx_, n_pairs, 1,
                           lw.exps_down_stride, n_embd, n_ff_exp, lw.exps_down_qtype, stream);
        }
        // acc[b,:] = sum_k w*dn (writes, no pre-zero). Must run before the shared expert
        // reuses moe_dn_ below.
        launch_moe_combine(moe_acc_, moe_dn_, d_moe_w_, K, n_embd, batch, stream);
    } else {
        // Path B: host-grouped
        // Readback (small). This sync also guarantees the previous layer's H2D staging
        // copies finished, so the pinned h_moe_idx_/h_moe_w_ are safe to overwrite.
        cudaMemcpyAsync(h_moe_logits_, moe_logits_, (size_t)batch * n_expert * sizeof(half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Softmax + top-K per token, then group (expert -> token list); groups laid out
        // contiguously in h_moe_idx_/h_moe_w_ (one slot per (expert,token) pair).
        std::vector<std::vector<std::pair<int32_t, float>>> groups(n_expert);
        std::vector<float> p(n_expert);
        std::vector<int> ord(n_expert);
        for (int b = 0; b < batch; b++) {
            const half * lrow = h_moe_logits_ + (size_t)b * n_expert;
            float mx = -1e30f;
            for (int e = 0; e < n_expert; e++) { float v = __half2float(lrow[e]); p[e] = v; if (v > mx) mx = v; }
            float sum = 0.0f;
            for (int e = 0; e < n_expert; e++) { p[e] = expf(p[e] - mx); sum += p[e]; }
            float inv = 1.0f / sum;
            for (int e = 0; e < n_expert; e++) p[e] *= inv;
            for (int e = 0; e < n_expert; e++) ord[e] = e;
            std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                              [&](int a2, int b2) { return p[a2] > p[b2]; });
            for (int k = 0; k < K; k++) {
                groups[ord[k]].push_back({(int32_t)b, p[ord[k]]});
            }
        }
        int m = 0;
        struct Grp { int e, off, cnt; };
        std::vector<Grp> glist;
        for (int e = 0; e < n_expert; e++) {
            if (groups[e].empty()) continue;
            glist.push_back({e, m, (int)groups[e].size()});
            for (auto & tw : groups[e]) { h_moe_idx_[m] = tw.first; h_moe_w_[m] = tw.second; m++; }
        }

        // Upload grouping (single H2D per array)
        cudaMemcpyAsync(d_moe_idx_, h_moe_idx_, (size_t)m * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_moe_w_,   h_moe_w_,   (size_t)m * sizeof(float),   cudaMemcpyHostToDevice, stream);

        // Zero the FP32 accumulator (scatter-add accumulates)
        cudaMemsetAsync(moe_acc_, 0, (size_t)batch * n_embd * sizeof(float), stream);

        // Per unique expert: gather rows -> gate/up -> silu*mul -> down -> weighted scatter-add
        for (auto & g : glist) {
            const uint8_t * bg = (const uint8_t *)lw.ffn_gate_exps;
            const uint8_t * bu = (const uint8_t *)lw.ffn_up_exps;
            const uint8_t * bd = (const uint8_t *)lw.ffn_down_exps;
            // Per-expert byte strides; F16 fallback stacks use element stride * 2 bytes.
            int64_t sg = lw.exps_quant_resident ? lw.exps_gate_stride : (int64_t)n_embd * n_ff_exp * 2;
            int64_t su = lw.exps_quant_resident ? lw.exps_up_stride   : (int64_t)n_embd * n_ff_exp * 2;
            int64_t sd = lw.exps_quant_resident ? lw.exps_down_stride : (int64_t)n_embd * n_ff_exp * 2;
            const void * w_gate = bg + (int64_t)g.e * sg;
            const void * w_up   = bu + (int64_t)g.e * su;
            const void * w_down = bd + (int64_t)g.e * sd;
            int tqg = lw.exps_quant_resident ? lw.exps_gate_qtype : 1;
            int tqu = lw.exps_quant_resident ? lw.exps_up_qtype   : 1;
            int tqd = lw.exps_quant_resident ? lw.exps_down_qtype : 1;
            // gather: moe_x_[i,:] = hidden_buf_[idx[i],:]  (embed_tokens is exactly a row gather)
            launch_embed_tokens(moe_x_, hidden_buf_, d_moe_idx_ + g.off, n_embd, g.cnt, stream);
            if (tqg == 1) matmul(moe_ff1_, moe_x_, w_gate, g.cnt, n_ff_exp, n_embd, 1, stream);
            else          launch_q4k_matmul(moe_ff1_, moe_x_, w_gate, g.cnt, n_ff_exp, n_embd, tqg, stream);
            if (tqu == 1) matmul(moe_ff2_, moe_x_, w_up, g.cnt, n_ff_exp, n_embd, 1, stream);
            else          launch_q4k_matmul(moe_ff2_, moe_x_, w_up, g.cnt, n_ff_exp, n_embd, tqu, stream);
            silu_mul(moe_ff1_, moe_ff1_, moe_ff2_, n_ff_exp, g.cnt, stream);
            if (tqd == 1) matmul(moe_dn_, moe_ff1_, w_down, g.cnt, n_embd, n_ff_exp, 1, stream);
            else          launch_q4k_matmul(moe_dn_, moe_ff1_, w_down, g.cnt, n_embd, n_ff_exp, tqd, stream);
            launch_moe_scatter_add(moe_acc_, moe_dn_, d_moe_idx_ + g.off, d_moe_w_ + g.off,
                                   n_embd, g.cnt, stream);
        }
    }

    // Shared expert (qwen2moe): dense SwiGLU FFN on all tokens, sigmoid-gated (F16 weights).
    // Runs after combine/scatter-add so reusing moe_dn_ is safe.
    const half * sh_out = nullptr;
    const half * sh_gate = nullptr;
    if (lw.ffn_gate_shexp && lw.ffn_up_shexp && lw.ffn_down_shexp) {
        matmul(moe_sh1_, hidden_buf_, lw.ffn_gate_shexp, batch, cfg_.n_ff_shexp, n_embd, 1, stream);
        matmul(moe_sh2_, hidden_buf_, lw.ffn_up_shexp,   batch, cfg_.n_ff_shexp, n_embd, 1, stream);
        silu_mul(moe_sh1_, moe_sh1_, moe_sh2_, cfg_.n_ff_shexp, batch, stream);
        matmul(moe_dn_, moe_sh1_, lw.ffn_down_shexp, batch, n_embd, cfg_.n_ff_shexp, 1, stream);
        sh_out = moe_dn_;
        if (lw.ffn_gate_inp_shexp) {
            matmul(moe_shg_, hidden_buf_, lw.ffn_gate_inp_shexp, batch, 1, n_embd, 1, stream);
            sh_gate = moe_shg_;
        }
    }

    // hidden_buf_ = (half)(acc + sigmoid(gate)*shexp)
    launch_moe_finalize(hidden_buf_, moe_acc_, sh_out, sh_gate, n_embd, batch, stream);
}

bool ForwardEngine::alloc_workspace(int max_batch) {
    max_batch_ = max_batch;
    padded_batch_ = std::max(max_batch, 16);  // pad to 16 for Tensor Core MMA

    // Allocate workspace for padded_batch_ so M-padding doesn't go OOB
    int alloc_batch = padded_batch_;
    size_t embd_bytes = alloc_batch * cfg_.n_embd * sizeof(half);
    size_t qo_bytes = alloc_batch * cfg_.n_head * cfg_.head_dim * sizeof(half);
    size_t kv_bytes = alloc_batch * cfg_.n_head_kv * cfg_.head_dim * sizeof(half);
    size_t qkv_bytes = alloc_batch * (cfg_.n_head + 2 * cfg_.n_head_kv) * cfg_.head_dim * sizeof(half);
    size_t gate_up_bytes = alloc_batch * 2 * cfg_.n_ff * sizeof(half);
    size_t ff_bytes = alloc_batch * cfg_.n_ff * sizeof(half);
    size_t logits_bytes = alloc_batch * cfg_.vocab_size * sizeof(float);

    cudaError_t err;
    err = cudaMalloc(&hidden_buf_, embd_bytes);    if (err) goto fail;
    err = cudaMalloc(&residual_buf_, embd_bytes);  if (err) goto fail;
    err = cudaMalloc(&q_buf_, qo_bytes);           if (err) goto fail;
    err = cudaMalloc(&k_buf_, kv_bytes);            if (err) goto fail;
    err = cudaMalloc(&v_buf_, kv_bytes);            if (err) goto fail;
    err = cudaMalloc(&attn_out_buf_, qo_bytes);    if (err) goto fail;
    err = cudaMalloc(&qkv_buf_, qkv_bytes);         if (err) goto fail;
    err = cudaMalloc(&gate_up_buf_, gate_up_bytes);  if (err) goto fail;
    err = cudaMalloc(&ffn_buf1_, ff_bytes);         if (err) goto fail;
    err = cudaMalloc(&ffn_buf2_, ff_bytes);         if (err) goto fail;
    err = cudaMalloc(&logits_buf_, logits_bytes);   if (err) goto fail;
    err = cudaMalloc(&norm_buf_, embd_bytes);       if (err) goto fail;

    // Qwen3.5 gated-attention workspace (query+gate matmul + deinterleaved gate)
    if (cfg_.has_gated_attn) {
        err = cudaMalloc(&qg_buf_,   (size_t)alloc_batch * 2 * cfg_.n_head * cfg_.head_dim * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&gate_buf_, (size_t)alloc_batch * cfg_.n_head * cfg_.head_dim * sizeof(half));     if (err) goto fail;
    }

    // DeltaNet per-token workspace (recurrent-layer intermediates)
    if (cfg_.has_recurrent) {
        int Hv = cfg_.ssm_dt_rank;
        int value_dim = cfg_.ssm_d_inner;
        int conv_dim  = cfg_.ssm_d_state * cfg_.ssm_n_group * 2 + value_dim;
        int dn_proj_n = conv_dim + value_dim + Hv + Hv;
        err = cudaMalloc(&dn_fused_, (size_t)alloc_batch * dn_proj_n * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_qkv_,  (size_t)alloc_batch * conv_dim  * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_z_,    (size_t)alloc_batch * value_dim * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_beta_, (size_t)alloc_batch * Hv        * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_g_,    (size_t)alloc_batch * Hv        * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_conv_, (size_t)alloc_batch * conv_dim  * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_o_,    (size_t)alloc_batch * value_dim * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_y_,    (size_t)alloc_batch * value_dim * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&dn_seq_ids_, (size_t)alloc_batch * sizeof(int32_t));       if (err) goto fail;
    }

    // MoE FFN workspace (router logits, gathered rows, per-expert FFN, FP32 accumulator).
    // ff1/ff2/dn hold one row per (token, k-slot) pair for the device-routed id-dispatch
    // path (max_batch * K rows); the host-grouped path (<= max_batch rows) fits inside.
    if (cfg_.has_moe) {
        int K = cfg_.n_expert_used > 0 ? cfg_.n_expert_used : 1;
        err = cudaMalloc(&moe_logits_, (size_t)alloc_batch * cfg_.n_expert * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&moe_x_,   (size_t)alloc_batch * cfg_.n_embd   * sizeof(half));    if (err) goto fail;
        err = cudaMalloc(&moe_ff1_, (size_t)alloc_batch * K * cfg_.n_ff_exp * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&moe_ff2_, (size_t)alloc_batch * K * cfg_.n_ff_exp * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&moe_dn_,  (size_t)alloc_batch * K * cfg_.n_embd   * sizeof(half)); if (err) goto fail;
        err = cudaMalloc(&moe_acc_, (size_t)alloc_batch * cfg_.n_embd   * sizeof(float));   if (err) goto fail;
        err = cudaMalloc(&d_moe_idx_, (size_t)alloc_batch * K * sizeof(int32_t));           if (err) goto fail;
        err = cudaMalloc(&d_moe_w_,   (size_t)alloc_batch * K * sizeof(float));             if (err) goto fail;
        err = cudaMalloc(&d_moe_zero_ids_, (size_t)alloc_batch * sizeof(int32_t));          if (err) goto fail;
        cudaMemset(d_moe_zero_ids_, 0, (size_t)alloc_batch * sizeof(int32_t));
        // dp4a path: Q81Act blocks (40B per 32 elems); worst case = per-pair rows * largest K
        {
            size_t maxK = (size_t)std::max(std::max(cfg_.n_embd, cfg_.n_ff_exp),
                                           cfg_.n_head * cfg_.head_dim);
            err = cudaMalloc(&moe_act_, (size_t)alloc_batch * K * (maxK / 32) * 40); if (err) goto fail;
        }
        err = cudaHostAlloc(&h_moe_logits_, (size_t)alloc_batch * cfg_.n_expert * sizeof(half), cudaHostAllocDefault); if (err) goto fail;
        err = cudaHostAlloc(&h_moe_idx_, (size_t)alloc_batch * K * sizeof(int32_t), cudaHostAllocDefault); if (err) goto fail;
        err = cudaHostAlloc(&h_moe_w_,   (size_t)alloc_batch * K * sizeof(float),   cudaHostAllocDefault); if (err) goto fail;
        if (cfg_.n_ff_shexp > 0) {
            err = cudaMalloc(&moe_sh1_, (size_t)alloc_batch * cfg_.n_ff_shexp * sizeof(half)); if (err) goto fail;
            err = cudaMalloc(&moe_sh2_, (size_t)alloc_batch * cfg_.n_ff_shexp * sizeof(half)); if (err) goto fail;
            err = cudaMalloc(&moe_shg_, (size_t)alloc_batch * sizeof(half));                    if (err) goto fail;
        }
    }

    // Q4_K scratch buffer: largest weight per layer is ffn_gate/up [n_embd, n_ff]
    {
        size_t max_weight = std::max((size_t)cfg_.n_embd * cfg_.n_ff,
                                      (size_t)cfg_.n_embd * cfg_.n_head * cfg_.head_dim);
        err = cudaMalloc(&q4k_scratch_, max_weight * sizeof(half));
        if (err) goto fail;
    }

    // Zero all workspace buffers (padding rows must be zero for M-padding)
    cudaMemset(hidden_buf_, 0, embd_bytes);
    cudaMemset(residual_buf_, 0, embd_bytes);
    cudaMemset(q_buf_, 0, qo_bytes);
    cudaMemset(k_buf_, 0, kv_bytes);
    cudaMemset(v_buf_, 0, kv_bytes);
    cudaMemset(attn_out_buf_, 0, qo_bytes);
    cudaMemset(ffn_buf1_, 0, ff_bytes);
    cudaMemset(ffn_buf2_, 0, ff_bytes);

    fprintf(stderr, "[forward] workspace: %.1f MiB GPU (padded to M=%d for Tensor Cores)\n",
            (double)(embd_bytes * 3 + qo_bytes * 2 + kv_bytes * 2 + ff_bytes * 2 + logits_bytes) / (1024 * 1024),
            padded_batch_);
    return true;

fail:
    fprintf(stderr, "[forward] workspace allocation failed: %s\n", cudaGetErrorString(err));
    return false;
}

// Forward decode
// For each layer:
//   1. RMSNorm(hidden) -> q_buf
//   2. Q = q_buf * Wq, K = q_buf * Wk, V = q_buf * Wv
//   3. RoPE(Q, K, positions)
//   4. Append K, V to paged KV cache
//   5. Attention: Q * KV_cache -> attn_out (via FlashInfer)
//   6. attn_out * Wo -> residual
//   7. hidden += residual
//   8. RMSNorm(hidden) -> ffn input
//   9. gate = input * W_gate, up = input * W_up
//  10. ffn_out = SiLU(gate) * up
//  11. ffn_out * W_down -> residual
//  12. hidden += residual

bool ForwardEngine::forward_decode(
    const int32_t * token_ids,
    const int32_t * seq_ids_cpu,
    const int32_t * positions,
    float * logits_out,
    int32_t * sampled_out,
    int batch_size,
    cudaStream_t stream) {

    if (batch_size <= 0 || batch_size > max_batch_) return false;

    // Serialize forward passes (cuBLAS handle is not thread-safe)
    std::lock_guard<std::mutex> lock(forward_mtx_);

    // Set cuBLAS stream
    cublasSetStream(cublas_, stream);

    // CPU preparation (not captured in graph)
    int bucket = CUDAGraphPool::next_pow2(batch_size);

    // Under CUDA graphs the captured graph runs at `bucket` rows on every replay, so all captured
    // compute (attention, logits, argmax) and the KV scatter process `bucket` rows. The KV
    // batch-args/write-offsets and the DeltaNet state map are padded to bucket, routing padding
    // rows [batch,bucket) to the scratch KV seq and scratch state slot (both == max_seqs) so their
    // discarded reads/writes cannot corrupt real sequences.
    // Batched CUDA graphs (bucket>1) are on by default; opt out with ALPHA_NO_GRAPH_BATCH=1.
    static int graph_batch_env = getenv("ALPHA_NO_GRAPH_BATCH") ? 0 : 1;
    // Only pad to `bucket` when a CUDA graph is actually captured/replayed. Eager decode processes
    // exactly batch_size rows; padding to next_pow2 would waste up to 2x compute near bucket
    // boundaries.
    bool graphs_active = use_cuda_graph_ && (bucket == 1 || graph_batch_env);
    bool graph_pad = graphs_active && sampled_out;
    int  pcount = graph_pad ? bucket : batch_size;
    const int32_t * psids = seq_ids_cpu;
    int fill = std::max(bucket, batch_size);
    if (graph_pad || cfg_.has_recurrent) {
        dn_sids_padded_.resize(fill);
        for (int b = 0; b < batch_size; b++) dn_sids_padded_[b] = seq_ids_cpu[b];
        for (int b = batch_size; b < fill; b++) dn_sids_padded_[b] = kv_pool_->scratch_seq_id;
        if (graph_pad) psids = dn_sids_padded_.data();
    }
    if (cfg_.has_recurrent) {
        // Publish the seq -> state-slot map for the DeltaNet kernels (scratch state slot == scratch_seq_id).
        cudaMemcpyAsync(dn_seq_ids_, dn_sids_padded_.data(), fill * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    }

    // Advance token count for each real sequence (allocates pages if needed)
    for (int b = 0; b < batch_size; b++) {
        kv_pool_->begin_token(seq_ids_cpu[b]);
    }

    // Build paged KV batch args + write offsets (uploads to GPU), padded to pcount.
    // Fast path skips indptr/indices rebuild when batch composition unchanged.
    if (!kv_pool_->build_batch_args_fast(psids, pcount, stream)) {
        kv_pool_->build_batch_args(psids, pcount, stream);
    }
    kv_pool_->build_kv_write_offsets(psids, pcount, stream);

    // Include current token in attention (self-attention): temporarily +1 real seqs'
    // n_tokens (scratch seq stays fixed), rebuild batch args, then restore.
    for (int b = 0; b < batch_size; b++) {
        kv_pool_->sequences[seq_ids_cpu[b]].n_tokens++;
    }
    kv_pool_->build_batch_args(psids, pcount, stream);
    for (int b = 0; b < batch_size; b++) {
        kv_pool_->sequences[seq_ids_cpu[b]].n_tokens--;
    }

    // GPU forward pass (CUDA graph capture/replay)

    bool graphs_ok = graphs_active;  // bucket==1, or bucket>1 when batched graphs are enabled
    if (graphs_ok && cuda_graphs_.has_graph(bucket) && logits_out) {
        // Replay captured graph (no per-kernel launch overhead)
        // Skip replay when logits_out=nullptr (prefill skip-logits mode) since the
        // captured graph includes the output matmul which we want to skip.
        cuda_graphs_.replay(bucket, stream);
        // Commit tokens (CPU-side, after graph replay)
        for (int b = 0; b < batch_size; b++) {
            kv_pool_->commit_token(seq_ids_cpu[b]);
        }
        return true;
    }

    // First call at this bucket size: run eagerly (and optionally capture)
    // Capture with bucket size (not batch_size) so replay works for all batch_size <= bucket
    // Only capture when sampled_out is non-null (decode), so the graph includes batch_argmax.
    // Prefill calls (sampled_out=nullptr) always run eagerly.
    bool capturing = graphs_ok && !cuda_graphs_.has_graph(bucket) && sampled_out;
    if (capturing) {
        // Warmup: need one eager run first so cuBLAS plans are cached
        if (graph_warmup_[bucket] < 1) {
            graph_warmup_[bucket]++;
            capturing = false;  // run eagerly this time, capture next time
        } else {
            cudaStreamSynchronize(stream);  // drain before capture
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
        }
    }

    // The whole graph regime (warmup + capture) runs at `bucket` so cuBLAS plans are created during
    // warmup rather than during capture, and the captured graph always processes a fixed `bucket`
    // rows on replay. Extra rows are padded (KV -> scratch seq) and their logits/sampling are
    // discarded (serve reads only [0,batch_size)).
    int effective_batch = graph_pad ? bucket : batch_size;

    // Step 0: Token embedding lookup
    embed_tokens(hidden_buf_, token_ids, effective_batch, stream);

    // Pre-zero Marlin lock pool (1 memset for entire pass instead of per-call)
    marlin_reset_locks(stream);

    // Per-layer forward (cross-layer fusion: FFN residual + next layer's attn norm)
    // Layer 0: just save residual + norm (no preceding FFN)
    {
        auto & lw0 = layer_weights_[0];
        launch_fused_rmsnorm_residual(hidden_buf_, residual_buf_, hidden_buf_,
                                       (const half *)lw0.attn_norm,
                                       cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);
    }

    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        auto & bp = batched_ptrs_[il];
        if (lw.is_recurrent) {
            forward_linear_attn_layer(il, effective_batch, seq_ids_cpu, false, stream);
        } else {
        if (!lw.wq && lw.wq_type != MARLIN_TYPE) continue;

        // 1. Attn RMSNorm already applied (by previous layer's end or pre-loop for layer 0)

        // 2+3. QKV projection + RoPE
        if (lw.is_gated_attn) {
            forward_gated_attn_qkv(il, effective_batch, positions, stream);
        } else if (lw.wqkv_fused && use_fused_kernels_) {
            // Fused path: 1 matmul + 1 kernel (deinterleave + bias + RoPE)
            if (lw.wq_q && effective_batch <= 32) {
                // Quant-resident QKV GEMVs (zero-ids -> dense; strided out fills the
                // fused-QKV buffer slices): ~half the weight bytes of the F16 GEMM.
                // dp4a path: quantize the hidden row(s) once, 3 int-dot GEMVs.
                int qd = cfg_.n_head * cfg_.head_dim;
                int kd = cfg_.n_head_kv * cfg_.head_dim;
                if (moe_dp4a_type(lw.wq_qt) && moe_dp4a_type(lw.wk_qt) &&
                    moe_dp4a_type(lw.wv_qt)) {
                    launch_moe_q8_act(moe_act_, hidden_buf_, effective_batch, cfg_.n_embd, stream);
                    launch_moe_gemv_id_q8((half *)qkv_buf_, lw.qkv_out_dim, moe_act_,
                                          lw.wq_q, d_moe_zero_ids_, effective_batch, 1, 0,
                                          qd, cfg_.n_embd, lw.wq_qt, stream);
                    launch_moe_gemv_id_q8((half *)qkv_buf_ + qd, lw.qkv_out_dim, moe_act_,
                                          lw.wk_q, d_moe_zero_ids_, effective_batch, 1, 0,
                                          kd, cfg_.n_embd, lw.wk_qt, stream);
                    launch_moe_gemv_id_q8((half *)qkv_buf_ + qd + kd, lw.qkv_out_dim, moe_act_,
                                          lw.wv_q, d_moe_zero_ids_, effective_batch, 1, 0,
                                          kd, cfg_.n_embd, lw.wv_qt, stream);
                } else {
                launch_moe_gemv_id_strided((half *)qkv_buf_, lw.qkv_out_dim, hidden_buf_,
                                           lw.wq_q, d_moe_zero_ids_, effective_batch, 1, 0,
                                           qd, cfg_.n_embd, lw.wq_qt, stream);
                launch_moe_gemv_id_strided((half *)qkv_buf_ + qd, lw.qkv_out_dim, hidden_buf_,
                                           lw.wk_q, d_moe_zero_ids_, effective_batch, 1, 0,
                                           kd, cfg_.n_embd, lw.wk_qt, stream);
                launch_moe_gemv_id_strided((half *)qkv_buf_ + qd + kd, lw.qkv_out_dim, hidden_buf_,
                                           lw.wv_q, d_moe_zero_ids_, effective_batch, 1, 0,
                                           kd, cfg_.n_embd, lw.wv_qt, stream);
                }
            } else {
            matmul((half *)qkv_buf_, hidden_buf_, lw.wqkv_fused,
                   effective_batch, lw.qkv_out_dim, cfg_.n_embd, 1 /*FP16*/, stream);
            }
            int q_dim = cfg_.n_head * cfg_.head_dim;
            int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
            // Qwen3/3.5 QK-norm: per-head RMSNorm on Q and K in the fused QKV buffer,
            // before RoPE. No-op when q_norm is null (models without QK-norm).
            if (lw.q_norm && lw.qk_norm_full) {
                // Full-width QK-norm (OLMoE/OLMo2): one RMSNorm group spanning the whole
                // Q (resp. K) projection: n_heads=1, head_dim=full width.
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.q_norm, 1, q_dim,
                               lw.qkv_out_dim, 0, effective_batch, cfg_.rms_norm_eps, stream);
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.k_norm, 1, kv_dim,
                               lw.qkv_out_dim, q_dim, effective_batch, cfg_.rms_norm_eps, stream);
            } else if (lw.q_norm) {
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.q_norm, cfg_.n_head, cfg_.head_dim,
                               lw.qkv_out_dim, 0, effective_batch, cfg_.rms_norm_eps, stream);
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.k_norm, cfg_.n_head_kv, cfg_.head_dim,
                               lw.qkv_out_dim, q_dim, effective_batch, cfg_.rms_norm_eps, stream);
            }
            launch_fused_deinterleave_rope(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                            positions,
                                            (const half *)lw.bq, (const half *)lw.bk, (const half *)lw.bv,
                                            q_dim, kv_dim, cfg_.head_dim,
                                            effective_batch,
                                            cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
        } else if (lw.wq_type == MARLIN_TYPE) {
            // Marlin INT4: fused QKV via single Marlin matmul -> deinterleave+bias+RoPE
            if (lw.mqkv_fused.d_packed && use_fused_kernels_) {
                marlin_matmul((half *)qkv_buf_, hidden_buf_, lw.mqkv_fused, effective_batch, stream);
                int q_dim = cfg_.n_head * cfg_.head_dim;
                int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
                launch_fused_deinterleave_rope(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                                positions,
                                                (const half *)lw.bq, (const half *)lw.bk, (const half *)lw.bv,
                                                q_dim, kv_dim, cfg_.head_dim,
                                                effective_batch,
                                                cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
            } else {
                // Fallback: separate Q, K, V Marlin matmuls
                marlin_matmul(q_buf_, hidden_buf_, lw.mq, effective_batch, stream);
                marlin_matmul(k_buf_, hidden_buf_, lw.mk, effective_batch, stream);
                marlin_matmul(v_buf_, hidden_buf_, lw.mv, effective_batch, stream);
                if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
                if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
                if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
                rope(q_buf_, k_buf_, positions,
                     cfg_.head_dim, cfg_.n_head, cfg_.n_head_kv, effective_batch,
                     cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
            }
        } else {
            // Fallback: separate Q, K, V matmuls + rope (cuBLAS FP16)
            matmul(q_buf_, hidden_buf_, lw.wq,
                   effective_batch, cfg_.n_head * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);

            if (bp.d_kv_B) {
                int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
                const half alpha_h = __float2half(1.0f);
                const half beta_h = __float2half(0.0f);
                cublasHgemmBatched(cublas_,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    kv_dim, effective_batch, cfg_.n_embd,
                    &alpha_h,
                    bp.d_kv_B, cfg_.n_embd,
                    bp.d_kv_A, cfg_.n_embd,
                    &beta_h,
                    bp.d_kv_C, kv_dim,
                    2);
            } else {
                matmul(k_buf_, hidden_buf_, lw.wk,
                       effective_batch, cfg_.n_head_kv * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
                matmul(v_buf_, hidden_buf_, lw.wv,
                       effective_batch, cfg_.n_head_kv * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            }
            // Add Q/K/V biases (required for Qwen; magnitudes up to ~130)
            if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
            if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
            if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
            rope(q_buf_, k_buf_, positions,
                 cfg_.head_dim, cfg_.n_head, cfg_.n_head_kv, effective_batch,
                 cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
        }

        // 4. Write K, V to paged cache (fused: 1 kernel instead of 2)
        {
            auto & lp = kv_pool_->layers[il];
            int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
            launch_scatter_kv_pair(lp.k_data, lp.v_data, k_buf_, v_buf_,
                                    kv_pool_->d_kv_write_offsets,
                                    kv_dim, effective_batch, stream);
        }

        // 5. Paged decode attention
        // Uses paged KV cache with CSR indirection (indptr/indices/last_page_len)
        {
            auto & lp = kv_pool_->layers[il];
            // Sliding-window: per-layer window (0 = full causal).
            int window = lw.swa_window;
            // The flashinfer decode path has no window arg, so SWA layers use the (windowed)
            // simple kernel; non-SWA layers keep the flashinfer path.
            if (use_flashinfer_attn_ && window <= 0) {
                launch_flashinfer_batch_decode(
                    attn_out_buf_, q_buf_,
                    lp.k_data, lp.v_data,
                    kv_pool_->d_indptr, kv_pool_->d_indices, kv_pool_->d_last_page_len,
                    PagedKVPool::PAGE_SIZE, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                    effective_batch, effective_batch,  // `bucket` rows: batch_size is baked into the graph and stale on replay
                    kv_pool_->stride_page, kv_pool_->stride_n, kv_pool_->stride_h,
                    stream);
            } else {
                launch_paged_decode_attention(
                    attn_out_buf_, q_buf_,
                    lp.k_data, lp.v_data,
                    kv_pool_->d_indptr, kv_pool_->d_indices, kv_pool_->d_last_page_len,
                    PagedKVPool::PAGE_SIZE, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                    kv_pool_->stride_page, kv_pool_->stride_n, kv_pool_->stride_h,
                    effective_batch, window, stream);
            }
        }

        // 5b. Gated attention: scale attention output by sigmoid(gate) before Wo
        if (lw.is_gated_attn) {
            launch_gate_sigmoid_mul(attn_out_buf_, gate_buf_, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
        }

        // 6. Output projection: attn_out * Wo
        if (lw.wq_type == MARLIN_TYPE) {
            marlin_matmul(hidden_buf_, attn_out_buf_, lw.mo, effective_batch, stream);
        } else if (lw.wo_q && effective_batch <= 32) {
            // Quant-resident Wo GEMV (zero-ids -> dense)
            if (moe_dp4a_type(lw.wo_qt)) {
                launch_moe_q8_act(moe_act_, attn_out_buf_, effective_batch,
                                  cfg_.n_head * cfg_.head_dim, stream);
                launch_moe_gemv_id_q8(hidden_buf_, cfg_.n_embd, moe_act_, lw.wo_q,
                                      d_moe_zero_ids_, effective_batch, 1, 0, cfg_.n_embd,
                                      cfg_.n_head * cfg_.head_dim, lw.wo_qt, stream);
            } else {
                launch_moe_gemv_id(hidden_buf_, attn_out_buf_, lw.wo_q, d_moe_zero_ids_,
                                   effective_batch, 1, 0, cfg_.n_embd,
                                   cfg_.n_head * cfg_.head_dim, lw.wo_qt, stream);
            }
        } else {
            matmul(hidden_buf_, attn_out_buf_, lw.wo,
                   effective_batch, cfg_.n_embd, cfg_.n_head * cfg_.head_dim,
                   lw.wq_type, stream);
        }

        }  // end non-recurrent (attention) path; FFN tail below is shared

        // 7+8. Fused: add residual + save new residual + FFN RMSNorm (1 kernel instead of 3)
        launch_fused_add_residual_rmsnorm(hidden_buf_, residual_buf_, hidden_buf_,
                                           (const half *)lw.ffn_norm,
                                           cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);

        // 9+10+11. FFN: MoE layers route through the expert dispatcher
        if (lw.is_moe) {
            forward_moe_ffn(il, effective_batch, stream);
        } else {
        // 9+10. Fused FFN gate+up + SiLU*mul (1 matmul + 1 kernel)
        if (lw.ffn_gate_up_fused && use_fused_kernels_) {
            matmul((half *)gate_up_buf_, hidden_buf_, lw.ffn_gate_up_fused,
                   effective_batch, lw.gate_up_out_dim, cfg_.n_embd, 1 /*FP16*/, stream);
            launch_silu_mul_interleaved(ffn_buf1_, gate_up_buf_,
                                         cfg_.n_ff, effective_batch, stream);
        } else if (lw.ffn_type == MARLIN_TYPE) {
            // Marlin INT4: fused gate+up via single Marlin matmul -> SiLU*mul
            if (lw.m_gate_up_fused.d_packed && use_fused_kernels_) {
                marlin_matmul((half *)gate_up_buf_, hidden_buf_, lw.m_gate_up_fused, effective_batch, stream);
                launch_silu_mul_interleaved(ffn_buf1_, gate_up_buf_,
                                             cfg_.n_ff, effective_batch, stream);
            } else {
                marlin_matmul(ffn_buf1_, hidden_buf_, lw.m_gate, effective_batch, stream);
                marlin_matmul(ffn_buf2_, hidden_buf_, lw.m_up, effective_batch, stream);
                silu_mul(ffn_buf1_, ffn_buf1_, ffn_buf2_, cfg_.n_ff, effective_batch, stream);
            }
        } else {
            // Fallback: separate gate/up matmuls + silu_mul (cuBLAS)
            if (bp.d_ffn_B) {
                const half alpha_h = __float2half(1.0f);
                const half beta_h = __float2half(0.0f);
                cublasHgemmBatched(cublas_,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    cfg_.n_ff, effective_batch, cfg_.n_embd,
                    &alpha_h,
                    bp.d_ffn_B, cfg_.n_embd,
                    bp.d_ffn_A, cfg_.n_embd,
                    &beta_h,
                    bp.d_ffn_C, cfg_.n_ff,
                    2);
            } else {
                matmul(ffn_buf1_, hidden_buf_, lw.ffn_gate,
                       effective_batch, cfg_.n_ff, cfg_.n_embd, lw.ffn_type, stream);
                matmul(ffn_buf2_, hidden_buf_, lw.ffn_up,
                       effective_batch, cfg_.n_ff, cfg_.n_embd, lw.ffn_type, stream);
            }
            silu_mul(ffn_buf1_, ffn_buf1_, ffn_buf2_, cfg_.n_ff, effective_batch, stream);
        }

        // 11. FFN down
        if (lw.ffn_type == MARLIN_TYPE) {
            marlin_matmul(hidden_buf_, ffn_buf1_, lw.m_down, effective_batch, stream);
        } else {
            matmul(hidden_buf_, ffn_buf1_, lw.ffn_down,
                   effective_batch, cfg_.n_embd, cfg_.n_ff, lw.ffn_type, stream);
        }
        }  // end dense FFN (MoE branch above)

        // 12. FFN residual + next layer's attn norm (fused cross-layer)
        if (il < cfg_.n_layers - 1) {
            auto & next_lw = layer_weights_[il + 1];
            launch_fused_add_residual_rmsnorm(hidden_buf_, residual_buf_, hidden_buf_,
                                               (const half *)next_lw.attn_norm,
                                               cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);
        } else {
            add_residual(hidden_buf_, residual_buf_, cfg_.n_embd, effective_batch, stream);
        }
    }

    // Final norm + output projection
    // Skipped when logits_out is nullptr (prefill tokens that don't need logits)
    if (logits_out) {
        rms_norm(hidden_buf_, hidden_buf_, output_norm_,
                 cfg_.n_embd, effective_batch, stream);

        // Output logits: hidden[batch, n_embd] * W_output[vocab, n_embd]^T -> logits[batch, vocab] FP32
        const void * w = output_ ? output_ : tok_embd_;
        // Use effective_batch (bucket under graphs), not batch_size: the latter is baked
        // into the captured graph and is wrong on replay when the real batch differs. Padding
        // rows' logits are computed then discarded (serve reads only [0,batch_size)).
        if (output_q_ && effective_batch <= 32) {
            // Quant-resident lm_head GEMV (F32 out, feeds batch_argmax)
            if (moe_dp4a_type(output_qtype_)) {
                launch_moe_q8_act(moe_act_, hidden_buf_, effective_batch, cfg_.n_embd, stream);
                launch_moe_gemv_id_q8_f32(logits_out, moe_act_, output_q_, d_moe_zero_ids_,
                                          effective_batch, 1, 0, cfg_.vocab_size, cfg_.n_embd,
                                          output_qtype_, stream);
            } else {
                launch_moe_gemv_id_f32(logits_out, hidden_buf_, output_q_, d_moe_zero_ids_,
                                       effective_batch, 1, 0, cfg_.vocab_size, cfg_.n_embd,
                                       output_qtype_, stream);
            }
        } else if (use_lt_matmul_) {
            auto & cm = get_cached_matmul(effective_batch, cfg_.vocab_size, cfg_.n_embd, stream, CUDA_R_32F);
            if (cm.tuned) {
                const float alpha = 1.0f, beta = 0.0f;
                cublasLtMatmul(cublaslt_, cm.op_desc,
                               &alpha, w, cm.b_desc, hidden_buf_, cm.a_desc,
                               &beta, logits_out, cm.c_desc, logits_out, cm.c_desc,
                               &cm.algo, lt_workspace_, lt_workspace_size_, stream);
            } else {
                const float alpha = 1.0f, beta = 0.0f;
                cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                             cfg_.vocab_size, effective_batch, cfg_.n_embd,
                             &alpha, w, CUDA_R_16F, cfg_.n_embd,
                             hidden_buf_, CUDA_R_16F, cfg_.n_embd,
                             &beta, logits_out, CUDA_R_32F, cfg_.vocab_size,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
        } else {
            const float alpha = 1.0f, beta = 0.0f;
            cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                         cfg_.vocab_size, effective_batch, cfg_.n_embd,
                         &alpha, w, CUDA_R_16F, cfg_.n_embd,
                         hidden_buf_, CUDA_R_16F, cfg_.n_embd,
                         &beta, logits_out, CUDA_R_32F, cfg_.vocab_size,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        // GPU sampling (inside the graph when capturing, so no extra sync)
        if (sampled_out) {
            launch_batch_argmax(sampled_out, logits_out, effective_batch, cfg_.vocab_size, stream);
        }
    }

    // End CUDA graph capture
    if (capturing) {
        bool ok = cuda_graphs_.capture(bucket, stream);
        if (ok) {
            fprintf(stderr, "[forward] CUDA graph captured for batch=%d\n", bucket);
            // Stream capture records work without executing it. Without this replay, this step's
            // KV scatter / SSM-state write / argmax would never run while commit_token below still
            // advances the sequence, leaving an uninitialized KV slot committed as valid.
            cuda_graphs_.replay(bucket, stream);
        }
    }

    // Commit tokens (CPU-side, after GPU work)
    for (int b = 0; b < batch_size; b++) {
        kv_pool_->commit_token(seq_ids_cpu[b]);
    }

    return true;
}

// Batched prefill
// Process N prompt tokens for one sequence in a single forward pass.
// Differences from forward_decode:
//   - Causal self-attention (not paged decode): all Q/K/V come from the current prompt
//   - KV cache: allocate N pages, write all K/V after each layer
//   - Output matmul only for the last token
//   - No CUDA graph capture

extern "C" void launch_causal_attn_prefill(
    half * output, const half * Q, const half * K, const half * V,
    int N, int n_head, int n_kv_head, int head_dim, float sm_scale,
    int window, cudaStream_t stream);

extern "C" void launch_extract_last_token(
    half * out, const half * in, int last_idx, int dim, cudaStream_t stream);

bool ForwardEngine::forward_prefill(
    const int32_t * token_ids,
    int seq_id,
    const int32_t * positions,
    float * logits_out,
    int32_t * sampled_out,
    int N,
    cudaStream_t stream) {

    if (N <= 0 || N > max_batch_) {
        fprintf(stderr, "[prefill] REJECTED: N=%d max_batch=%d\n", N, max_batch_);
        return false;
    }

    std::lock_guard<std::mutex> lock(forward_mtx_);
    cublasSetStream(cublas_, stream);
    fprintf(stderr, "[prefill] seq=%d N=%d\n", seq_id, N);

    // Allocate pages for N tokens. Must simulate proper token-by-token allocation
    // since begin_token checks n_tokens%PAGE_SIZE to decide on new pages.
    {
        auto & seq = kv_pool_->sequences[seq_id];
        int start_n = seq.n_tokens;
        for (int i = 0; i < N; i++) {
            kv_pool_->begin_token(seq_id);
            seq.n_tokens++;  // temporarily increment for correct page allocation
        }
        seq.n_tokens = start_n;  // reset; commit_token finalizes later
    }

    // Build per-token KV write offsets for N positions
    std::vector<int64_t> h_write_offsets(N);
    {
        auto & seq = kv_pool_->sequences[seq_id];
        for (int i = 0; i < N; i++) {
            int pos = seq.n_tokens + i;
            int page_idx = pos / PagedKVPool::PAGE_SIZE;
            int pos_in_page = pos % PagedKVPool::PAGE_SIZE;
            int32_t page_id = seq.page_ids[page_idx];
            h_write_offsets[i] = (int64_t)page_id * kv_pool_->stride_page
                               + (int64_t)pos_in_page * kv_pool_->stride_n;
        }
    }
    cudaMemcpyAsync(kv_pool_->d_kv_write_offsets, h_write_offsets.data(),
                    N * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // M-pad for Tensor Cores
    int effective_batch = std::max(N, 1);

    // Step 0: Token embedding lookup
    embed_tokens(hidden_buf_, token_ids, effective_batch, stream);

    marlin_reset_locks(stream);

    // Per-layer forward with M=N (cross-layer fusion)
    {
        auto & lw0 = layer_weights_[0];
        launch_fused_rmsnorm_residual(hidden_buf_, residual_buf_, hidden_buf_,
                                       (const half *)lw0.attn_norm,
                                       cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);
    }
    std::vector<int32_t> dn_sids;
    if (cfg_.has_recurrent) dn_sids.assign(N, seq_id);
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        if (lw.is_recurrent) {
            forward_linear_attn_layer(il, N, dn_sids.data(), true, stream);
        } else {
        if (!lw.wq && lw.wq_type != MARLIN_TYPE) continue;

        // 2+3. QKV projection + RoPE (same as decode)
        if (lw.is_gated_attn) {
            forward_gated_attn_qkv(il, effective_batch, positions, stream);
        } else if (lw.wqkv_fused && use_fused_kernels_) {
            matmul((half *)qkv_buf_, hidden_buf_, lw.wqkv_fused,
                   effective_batch, lw.qkv_out_dim, cfg_.n_embd, 1, stream);
            int q_dim = cfg_.n_head * cfg_.head_dim;
            int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
            // Qwen3/3.5 QK-norm: per-head RMSNorm on Q and K in the fused QKV buffer,
            // before RoPE. No-op when q_norm is null (models without QK-norm).
            if (lw.q_norm && lw.qk_norm_full) {
                // Full-width QK-norm (OLMoE/OLMo2): one RMSNorm group spanning the whole
                // Q (resp. K) projection: n_heads=1, head_dim=full width.
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.q_norm, 1, q_dim,
                               lw.qkv_out_dim, 0, effective_batch, cfg_.rms_norm_eps, stream);
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.k_norm, 1, kv_dim,
                               lw.qkv_out_dim, q_dim, effective_batch, cfg_.rms_norm_eps, stream);
            } else if (lw.q_norm) {
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.q_norm, cfg_.n_head, cfg_.head_dim,
                               lw.qkv_out_dim, 0, effective_batch, cfg_.rms_norm_eps, stream);
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.k_norm, cfg_.n_head_kv, cfg_.head_dim,
                               lw.qkv_out_dim, q_dim, effective_batch, cfg_.rms_norm_eps, stream);
            }
            launch_fused_deinterleave_rope(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                            positions,
                                            (const half *)lw.bq, (const half *)lw.bk, (const half *)lw.bv,
                                            q_dim, kv_dim, cfg_.head_dim,
                                            effective_batch,
                                            cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
        } else if (lw.wq_type == MARLIN_TYPE) {
            if (lw.mqkv_fused.d_packed && use_fused_kernels_) {
                marlin_matmul((half *)qkv_buf_, hidden_buf_, lw.mqkv_fused, effective_batch, stream);
                int q_dim = cfg_.n_head * cfg_.head_dim;
                int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
                launch_fused_deinterleave_rope(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                                positions,
                                                (const half *)lw.bq, (const half *)lw.bk, (const half *)lw.bv,
                                                q_dim, kv_dim, cfg_.head_dim,
                                                effective_batch,
                                                cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
            } else {
                marlin_matmul(q_buf_, hidden_buf_, lw.mq, effective_batch, stream);
                marlin_matmul(k_buf_, hidden_buf_, lw.mk, effective_batch, stream);
                marlin_matmul(v_buf_, hidden_buf_, lw.mv, effective_batch, stream);
                if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
                if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
                if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
                rope(q_buf_, k_buf_, positions, cfg_.head_dim, cfg_.n_head, cfg_.n_head_kv, effective_batch,
                     cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
            }
        } else {
            matmul(q_buf_, hidden_buf_, lw.wq,
                   effective_batch, cfg_.n_head * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            matmul(k_buf_, hidden_buf_, lw.wk,
                   effective_batch, cfg_.n_head_kv * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            matmul(v_buf_, hidden_buf_, lw.wv,
                   effective_batch, cfg_.n_head_kv * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
            if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
            if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
            rope(q_buf_, k_buf_, positions, cfg_.head_dim, cfg_.n_head, cfg_.n_head_kv, effective_batch,
                 cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
        }

        // 4. Write K, V to paged cache (fused: 1 kernel instead of 2)
        {
            auto & lp = kv_pool_->layers[il];
            int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
            launch_scatter_kv_pair(lp.k_data, lp.v_data, k_buf_, v_buf_,
                                    kv_pool_->d_kv_write_offsets,
                                    kv_dim, effective_batch, stream);
        }

        // 5. Causal self-attention on the Q/K/V buffers directly (not the paged KV cache).
        {
            float sm_scale = 1.0f / sqrtf((float)cfg_.head_dim);
            int window = lw.swa_window;
            launch_causal_attn_prefill(attn_out_buf_, q_buf_, k_buf_, v_buf_,
                                        N, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                                        sm_scale, window, stream);
        }

        // 5b. Gated attention: scale attention output by sigmoid(gate) before Wo
        if (lw.is_gated_attn) {
            launch_gate_sigmoid_mul(attn_out_buf_, gate_buf_, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
        }

        // 6. Output projection
        if (lw.wq_type == MARLIN_TYPE) {
            marlin_matmul(hidden_buf_, attn_out_buf_, lw.mo, effective_batch, stream);
        } else {
            matmul(hidden_buf_, attn_out_buf_, lw.wo,
                   effective_batch, cfg_.n_embd, cfg_.n_head * cfg_.head_dim,
                   lw.wq_type, stream);
        }

        }  // end non-recurrent (attention) path; FFN tail below is shared

        // 7+8. Fused: add residual + save + FFN RMSNorm
        launch_fused_add_residual_rmsnorm(hidden_buf_, residual_buf_, hidden_buf_,
                                           (const half *)lw.ffn_norm,
                                           cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);

        // 9+10+11. FFN: MoE layers route through the expert dispatcher
        if (lw.is_moe) {
            forward_moe_ffn(il, effective_batch, stream);
        } else {
        // 9+10. FFN (same paths as decode)
        if (lw.ffn_gate_up_fused && use_fused_kernels_) {
            matmul((half *)gate_up_buf_, hidden_buf_, lw.ffn_gate_up_fused,
                   effective_batch, lw.gate_up_out_dim, cfg_.n_embd, 1, stream);
            launch_silu_mul_interleaved(ffn_buf1_, gate_up_buf_,
                                         cfg_.n_ff, effective_batch, stream);
        } else if (lw.ffn_type == MARLIN_TYPE) {
            if (lw.m_gate_up_fused.d_packed && use_fused_kernels_) {
                marlin_matmul((half *)gate_up_buf_, hidden_buf_, lw.m_gate_up_fused, effective_batch, stream);
                launch_silu_mul_interleaved(ffn_buf1_, gate_up_buf_,
                                             cfg_.n_ff, effective_batch, stream);
            } else {
                marlin_matmul(ffn_buf1_, hidden_buf_, lw.m_gate, effective_batch, stream);
                marlin_matmul(ffn_buf2_, hidden_buf_, lw.m_up, effective_batch, stream);
                silu_mul(ffn_buf1_, ffn_buf1_, ffn_buf2_, cfg_.n_ff, effective_batch, stream);
            }
        } else {
            matmul(ffn_buf1_, hidden_buf_, lw.ffn_gate,
                   effective_batch, cfg_.n_ff, cfg_.n_embd, lw.ffn_type, stream);
            matmul(ffn_buf2_, hidden_buf_, lw.ffn_up,
                   effective_batch, cfg_.n_ff, cfg_.n_embd, lw.ffn_type, stream);
            silu_mul(ffn_buf1_, ffn_buf1_, ffn_buf2_, cfg_.n_ff, effective_batch, stream);
        }

        // 11. FFN down
        if (lw.ffn_type == MARLIN_TYPE) {
            marlin_matmul(hidden_buf_, ffn_buf1_, lw.m_down, effective_batch, stream);
        } else {
            matmul(hidden_buf_, ffn_buf1_, lw.ffn_down,
                   effective_batch, cfg_.n_embd, cfg_.n_ff, lw.ffn_type, stream);
        }
        }  // end dense FFN (MoE branch above)

        // 12. FFN residual + next layer's attn norm (cross-layer fusion)
        if (il < cfg_.n_layers - 1) {
            auto & next_lw = layer_weights_[il + 1];
            launch_fused_add_residual_rmsnorm(hidden_buf_, residual_buf_, hidden_buf_,
                                               (const half *)next_lw.attn_norm,
                                               cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);
        } else {
            add_residual(hidden_buf_, residual_buf_, cfg_.n_embd, effective_batch, stream);
        }
    }

    // Output logits: only compute for last token, skip if logits_out is nullptr
    if (logits_out) {
        launch_extract_last_token(residual_buf_, hidden_buf_, N - 1, cfg_.n_embd, stream);
        rms_norm(residual_buf_, residual_buf_, output_norm_, cfg_.n_embd, 1, stream);

        const void * w = output_ ? output_ : tok_embd_;
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                     cfg_.vocab_size, 1, cfg_.n_embd,
                     &alpha, w, CUDA_R_16F, cfg_.n_embd,
                     residual_buf_, CUDA_R_16F, cfg_.n_embd,
                     &beta, logits_out, CUDA_R_32F, cfg_.vocab_size,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        if (sampled_out) {
            launch_batch_argmax(sampled_out, logits_out, 1, cfg_.vocab_size, stream);
        }
    }

    // Commit all N tokens to KV cache
    kv_pool_->sequences[seq_id].n_tokens += N;

    return true;
}

// Cached prefill (paged attention)
// Process N suffix tokens with M prefix tokens already in the paged KV cache.
// Per-layer: QKV + RoPE + scatter to cache + paged prefill attention + output proj + FFN.
// Only computes for N suffix tokens (prefix matmuls fully skipped).

extern "C" void launch_paged_prefill_attention(
    half * output, const half * q,
    const half * k_pool, const half * v_pool,
    const int32_t * indptr, const int32_t * indices, const int32_t * last_page_len,
    int page_size, int n_heads, int n_kv_heads, int head_dim,
    int stride_page, int stride_n, int stride_h,
    int n_queries, int prefix_len, cudaStream_t stream);

bool ForwardEngine::forward_prefill_cached(
    const int32_t * token_ids,
    int seq_id,
    const int32_t * positions,
    float * logits_out,
    int N,
    int prefix_len,
    cudaStream_t stream) {

    if (N <= 0 || N > max_batch_) return false;

    std::lock_guard<std::mutex> lock(forward_mtx_);
    cublasSetStream(cublas_, stream);
    fprintf(stderr, "[prefill-cached] seq=%d N=%d prefix=%d\n", seq_id, N, prefix_len);

    // Allocate pages for N suffix tokens
    {
        auto & seq = kv_pool_->sequences[seq_id];
        int start_n = seq.n_tokens;  // should be prefix_len
        for (int i = 0; i < N; i++) {
            kv_pool_->begin_token(seq_id);
            seq.n_tokens++;
        }
        seq.n_tokens = start_n;  // reset; commit at end
    }

    // Build KV write offsets for suffix positions (prefix_len..prefix_len+N-1)
    {
        auto & seq = kv_pool_->sequences[seq_id];
        std::vector<int64_t> h_write_offsets(N);
        for (int i = 0; i < N; i++) {
            int pos = seq.n_tokens + i;  // = prefix_len + i
            int page_idx = pos / PagedKVPool::PAGE_SIZE;
            int pos_in_page = pos % PagedKVPool::PAGE_SIZE;
            int32_t page_id = seq.page_ids[page_idx];
            h_write_offsets[i] = (int64_t)page_id * kv_pool_->stride_page
                               + (int64_t)pos_in_page * kv_pool_->stride_n;
        }
        cudaMemcpyAsync(kv_pool_->d_kv_write_offsets, h_write_offsets.data(),
                        N * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    }

    // Build paged KV batch args for full sequence (prefix + suffix) for attention
    // Temporarily set n_tokens to prefix_len + N so batch_args covers all pages
    {
        auto & seq = kv_pool_->sequences[seq_id];
        int saved = seq.n_tokens;
        seq.n_tokens = prefix_len + N;
        int32_t sid = seq_id;
        kv_pool_->build_batch_args(&sid, 1, stream);
        seq.n_tokens = saved;
    }

    int effective_batch = N;

    // Step 0: Token embedding lookup
    embed_tokens(hidden_buf_, token_ids, effective_batch, stream);

    marlin_reset_locks(stream);

    // Per-layer forward (cross-layer fusion, paged attention)
    {
        auto & lw0 = layer_weights_[0];
        launch_fused_rmsnorm_residual(hidden_buf_, residual_buf_, hidden_buf_,
                                       (const half *)lw0.attn_norm,
                                       cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);
    }
    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        if (!lw.wq && lw.wq_type != MARLIN_TYPE) continue;

        // QKV projection + RoPE (same paths as forward_prefill)
        if (lw.is_gated_attn) {
            forward_gated_attn_qkv(il, effective_batch, positions, stream);
        } else if (lw.wqkv_fused && use_fused_kernels_) {
            matmul((half *)qkv_buf_, hidden_buf_, lw.wqkv_fused,
                   effective_batch, lw.qkv_out_dim, cfg_.n_embd, 1, stream);
            int q_dim = cfg_.n_head * cfg_.head_dim;
            int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
            // Qwen3/3.5 QK-norm: per-head RMSNorm on Q and K in the fused QKV buffer,
            // before RoPE. No-op when q_norm is null (models without QK-norm).
            if (lw.q_norm && lw.qk_norm_full) {
                // Full-width QK-norm (OLMoE/OLMo2): one RMSNorm group spanning the whole
                // Q (resp. K) projection: n_heads=1, head_dim=full width.
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.q_norm, 1, q_dim,
                               lw.qkv_out_dim, 0, effective_batch, cfg_.rms_norm_eps, stream);
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.k_norm, 1, kv_dim,
                               lw.qkv_out_dim, q_dim, effective_batch, cfg_.rms_norm_eps, stream);
            } else if (lw.q_norm) {
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.q_norm, cfg_.n_head, cfg_.head_dim,
                               lw.qkv_out_dim, 0, effective_batch, cfg_.rms_norm_eps, stream);
                launch_qk_norm((half *)qkv_buf_, (const half *)lw.k_norm, cfg_.n_head_kv, cfg_.head_dim,
                               lw.qkv_out_dim, q_dim, effective_batch, cfg_.rms_norm_eps, stream);
            }
            launch_fused_deinterleave_rope(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                            positions,
                                            (const half *)lw.bq, (const half *)lw.bk, (const half *)lw.bv,
                                            q_dim, kv_dim, cfg_.head_dim,
                                            effective_batch,
                                            cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
        } else if (lw.wq_type == MARLIN_TYPE) {
            if (lw.mqkv_fused.d_packed && use_fused_kernels_) {
                marlin_matmul((half *)qkv_buf_, hidden_buf_, lw.mqkv_fused, effective_batch, stream);
                int q_dim = cfg_.n_head * cfg_.head_dim;
                int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
                launch_fused_deinterleave_rope(q_buf_, k_buf_, v_buf_, qkv_buf_,
                                                positions,
                                                (const half *)lw.bq, (const half *)lw.bk, (const half *)lw.bv,
                                                q_dim, kv_dim, cfg_.head_dim,
                                                effective_batch,
                                                cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
            } else {
                marlin_matmul(q_buf_, hidden_buf_, lw.mq, effective_batch, stream);
                marlin_matmul(k_buf_, hidden_buf_, lw.mk, effective_batch, stream);
                marlin_matmul(v_buf_, hidden_buf_, lw.mv, effective_batch, stream);
                if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
                if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
                if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
                rope(q_buf_, k_buf_, positions, cfg_.head_dim, cfg_.n_head, cfg_.n_head_kv, effective_batch,
                     cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
            }
        } else {
            matmul(q_buf_, hidden_buf_, lw.wq,
                   effective_batch, cfg_.n_head * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            matmul(k_buf_, hidden_buf_, lw.wk,
                   effective_batch, cfg_.n_head_kv * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            matmul(v_buf_, hidden_buf_, lw.wv,
                   effective_batch, cfg_.n_head_kv * cfg_.head_dim, cfg_.n_embd, lw.wq_type, stream);
            if (lw.bq) launch_add_bias(q_buf_, (const half *)lw.bq, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
            if (lw.bk) launch_add_bias(k_buf_, (const half *)lw.bk, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
            if (lw.bv) launch_add_bias(v_buf_, (const half *)lw.bv, cfg_.n_head_kv * cfg_.head_dim, effective_batch, stream);
            rope(q_buf_, k_buf_, positions, cfg_.head_dim, cfg_.n_head, cfg_.n_head_kv, effective_batch,
                 cfg_.rope_freq_base, cfg_.rope_freq_scale, stream);
        }

        // Scatter suffix K/V to paged cache (fused)
        {
            auto & lp = kv_pool_->layers[il];
            int kv_dim = cfg_.n_head_kv * cfg_.head_dim;
            launch_scatter_kv_pair(lp.k_data, lp.v_data, k_buf_, v_buf_,
                                    kv_pool_->d_kv_write_offsets,
                                    kv_dim, effective_batch, stream);
        }

        // Paged prefill attention: N suffix queries attend to prefix+suffix KV cache
        {
            auto & lp = kv_pool_->layers[il];
            launch_paged_prefill_attention(
                attn_out_buf_, q_buf_,
                lp.k_data, lp.v_data,
                kv_pool_->d_indptr, kv_pool_->d_indices, kv_pool_->d_last_page_len,
                PagedKVPool::PAGE_SIZE, cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim,
                kv_pool_->stride_page, kv_pool_->stride_n, kv_pool_->stride_h,
                N, prefix_len, stream);
        }

        // Gated attention: scale attention output by sigmoid(gate) before Wo
        if (lw.is_gated_attn) {
            launch_gate_sigmoid_mul(attn_out_buf_, gate_buf_, cfg_.n_head * cfg_.head_dim, effective_batch, stream);
        }

        // Output projection
        if (lw.wq_type == MARLIN_TYPE) {
            marlin_matmul(hidden_buf_, attn_out_buf_, lw.mo, effective_batch, stream);
        } else {
            matmul(hidden_buf_, attn_out_buf_, lw.wo,
                   effective_batch, cfg_.n_embd, cfg_.n_head * cfg_.head_dim,
                   lw.wq_type, stream);
        }

        // Fused: add residual + save + FFN RMSNorm
        launch_fused_add_residual_rmsnorm(hidden_buf_, residual_buf_, hidden_buf_,
                                           (const half *)lw.ffn_norm,
                                           cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);

        // FFN (same paths as prefill); MoE layers route through the expert dispatcher
        if (lw.is_moe) {
            forward_moe_ffn(il, effective_batch, stream);
        } else {
        if (lw.ffn_gate_up_fused && use_fused_kernels_) {
            matmul((half *)gate_up_buf_, hidden_buf_, lw.ffn_gate_up_fused,
                   effective_batch, lw.gate_up_out_dim, cfg_.n_embd, 1, stream);
            launch_silu_mul_interleaved(ffn_buf1_, gate_up_buf_,
                                         cfg_.n_ff, effective_batch, stream);
        } else if (lw.ffn_type == MARLIN_TYPE) {
            if (lw.m_gate_up_fused.d_packed && use_fused_kernels_) {
                marlin_matmul((half *)gate_up_buf_, hidden_buf_, lw.m_gate_up_fused, effective_batch, stream);
                launch_silu_mul_interleaved(ffn_buf1_, gate_up_buf_,
                                             cfg_.n_ff, effective_batch, stream);
            } else {
                marlin_matmul(ffn_buf1_, hidden_buf_, lw.m_gate, effective_batch, stream);
                marlin_matmul(ffn_buf2_, hidden_buf_, lw.m_up, effective_batch, stream);
                silu_mul(ffn_buf1_, ffn_buf1_, ffn_buf2_, cfg_.n_ff, effective_batch, stream);
            }
        } else {
            matmul(ffn_buf1_, hidden_buf_, lw.ffn_gate,
                   effective_batch, cfg_.n_ff, cfg_.n_embd, lw.ffn_type, stream);
            matmul(ffn_buf2_, hidden_buf_, lw.ffn_up,
                   effective_batch, cfg_.n_ff, cfg_.n_embd, lw.ffn_type, stream);
            silu_mul(ffn_buf1_, ffn_buf1_, ffn_buf2_, cfg_.n_ff, effective_batch, stream);
        }

        if (lw.ffn_type == MARLIN_TYPE) {
            marlin_matmul(hidden_buf_, ffn_buf1_, lw.m_down, effective_batch, stream);
        } else {
            matmul(hidden_buf_, ffn_buf1_, lw.ffn_down,
                   effective_batch, cfg_.n_embd, cfg_.n_ff, lw.ffn_type, stream);
        }
        }  // end dense FFN (MoE branch above)

        if (il < cfg_.n_layers - 1) {
            auto & next_lw = layer_weights_[il + 1];
            launch_fused_add_residual_rmsnorm(hidden_buf_, residual_buf_, hidden_buf_,
                                               (const half *)next_lw.attn_norm,
                                               cfg_.n_embd, effective_batch, cfg_.rms_norm_eps, stream);
        } else {
            add_residual(hidden_buf_, residual_buf_, cfg_.n_embd, effective_batch, stream);
        }
    }

    // Output logits for last suffix token only
    if (logits_out) {
        launch_extract_last_token(residual_buf_, hidden_buf_, N - 1, cfg_.n_embd, stream);
        rms_norm(residual_buf_, residual_buf_, output_norm_, cfg_.n_embd, 1, stream);

        const void * w = output_ ? output_ : tok_embd_;
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                     cfg_.vocab_size, 1, cfg_.n_embd,
                     &alpha, w, CUDA_R_16F, cfg_.n_embd,
                     residual_buf_, CUDA_R_16F, cfg_.n_embd,
                     &beta, logits_out, CUDA_R_32F, cfg_.vocab_size,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    // Commit all N suffix tokens
    kv_pool_->sequences[seq_id].n_tokens += N;

    return true;
}

void ForwardEngine::rms_norm(half * out, const half * inp, const void * weight,
                              int n, int batch, cudaStream_t stream) {
    // Norm weights are converted to GPU FP16 at init.
    launch_rms_norm(out, inp, (const half *)weight, n, batch, cfg_.rms_norm_eps, stream);
}

ForwardEngine::CachedMatmul & ForwardEngine::get_cached_matmul(int m, int n, int k, cudaStream_t stream,
                                                                cudaDataType_t c_type) {
    // Include output type in cache key to distinguish FP16 vs FP32 output
    MatmulKey key{m, n, k + (c_type == CUDA_R_32F ? 1000000 : 0)};
    auto it = lt_cache_.find(key);
    if (it != lt_cache_.end()) return it->second;

    auto & cm = lt_cache_[key];

    cublasLtMatmulDescCreate(&cm.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(cm.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
    cublasLtMatmulDescSetAttribute(cm.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // B = weight [k, n], A = input [k, m], C = output [n, m]
    cublasLtMatrixLayoutCreate(&cm.b_desc, CUDA_R_16F, k, n, k);
    cublasLtMatrixLayoutCreate(&cm.a_desc, CUDA_R_16F, k, m, k);
    cublasLtMatrixLayoutCreate(&cm.c_desc, c_type, n, m, n);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t pref_ws = lt_workspace_size_;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &pref_ws, sizeof(pref_ws));

    cublasLtMatmulHeuristicResult_t results[16];
    int n_results = 0;
    cublasLtMatmulAlgoGetHeuristic(cublaslt_, cm.op_desc, cm.b_desc, cm.a_desc, cm.c_desc, cm.c_desc,
                                    pref, 16, results, &n_results);

    // Auto-tune (opt-in via ALPHA_LT_TUNE). By default the top heuristic result is used, which is
    // stable and close to optimal. Candidates the heuristic flags as non-SUCCESS or that need more
    // workspace than allocated are skipped (running those can fault with an illegal access and
    // corrupt the context), and each candidate's matmul/sync status is checked.
    static const bool lt_tune = getenv("ALPHA_LT_TUNE") != nullptr;
    if (lt_tune && n_results > 1) {
        size_t c_elem = (c_type == CUDA_R_32F) ? sizeof(float) : sizeof(half);
        half *tmp_a = nullptr, *tmp_b = nullptr;
        void *tmp_c = nullptr;
        cudaMalloc(&tmp_a, (size_t)m * k * sizeof(half));
        cudaMalloc(&tmp_b, (size_t)n * k * sizeof(half));
        cudaMalloc(&tmp_c, (size_t)m * n * c_elem);
        cudaMemset(tmp_a, 0, (size_t)m * k * sizeof(half));
        cudaMemset(tmp_b, 0, (size_t)n * k * sizeof(half));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float best_time = 1e30f;
        int best_idx = 0;
        const float alpha = 1.0f, beta = 0.0f;
        const int n_warmup = 3;
        const int n_timed = 10;

        for (int i = 0; i < n_results && i < 12; i++) {
            // Skip candidates that can fault before running them.
            if (results[i].state != CUBLAS_STATUS_SUCCESS ||
                results[i].workspaceSize > lt_workspace_size_) continue;
            cublasStatus_t cs = CUBLAS_STATUS_SUCCESS;
            for (int w = 0; w < n_warmup && cs == CUBLAS_STATUS_SUCCESS; w++) {
                cs = cublasLtMatmul(cublaslt_, cm.op_desc,
                               &alpha, tmp_b, cm.b_desc, tmp_a, cm.a_desc,
                               &beta, tmp_c, cm.c_desc, tmp_c, cm.c_desc,
                               &results[i].algo, lt_workspace_, lt_workspace_size_, stream);
            }
            if (cs != CUBLAS_STATUS_SUCCESS)                 { cudaGetLastError(); continue; }
            if (cudaStreamSynchronize(stream) != cudaSuccess) { cudaGetLastError(); continue; }
            cudaEventRecord(start, stream);
            for (int r = 0; r < n_timed; r++) {
                cublasLtMatmul(cublaslt_, cm.op_desc,
                               &alpha, tmp_b, cm.b_desc, tmp_a, cm.a_desc,
                               &beta, tmp_c, cm.c_desc, tmp_c, cm.c_desc,
                               &results[i].algo, lt_workspace_, lt_workspace_size_, stream);
            }
            cudaEventRecord(stop, stream);
            if (cudaEventSynchronize(stop) != cudaSuccess)   { cudaGetLastError(); continue; }
            float ms = 0;
            if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) { cudaGetLastError(); continue; }
            if (ms < best_time) { best_time = ms; best_idx = i; }
        }

        cm.algo = results[best_idx].algo;
        cm.tuned = true;
        if (getenv("ALPHA_LT_TUNE_VERBOSE"))
            fprintf(stderr, "[forward] auto-tuned matmul M=%d N=%d K=%d: algo %d/%d (%.3fms/%d runs)%s\n",
                m, n, k, best_idx, n_results, best_time, n_timed,
                c_type == CUDA_R_32F ? " [FP32 out]" : "");

        cudaFree(tmp_a);
        cudaFree(tmp_b);
        cudaFree(tmp_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGetLastError();
    } else if (n_results >= 1) {
        cm.algo = results[0].algo;
        cm.tuned = true;
    }

    cublasLtMatmulPreferenceDestroy(pref);
    return lt_cache_[key];
}

void ForwardEngine::matmul(half * out, const half * a, const void * b,
                            int m, int n, int k, int b_type, cudaStream_t stream) {
    // C[m,n] = A[m,k] * B[n,k]^T  (ggml stores weights as [out_features, in_features])
    // A is always FP16. B depends on weight type.

    if (b_type == 1 && use_lt_matmul_) {  // FP16 with cublasLt auto-tuning
        auto & cm = get_cached_matmul(m, n, k, stream);
        if (cm.tuned) {
            const float alpha = 1.0f, beta = 0.0f;
            cublasLtMatmul(cublaslt_, cm.op_desc,
                           &alpha, b, cm.b_desc, a, cm.a_desc,
                           &beta, out, cm.c_desc, out, cm.c_desc,
                           &cm.algo, lt_workspace_, lt_workspace_size_, stream);
            return;
        }
    }

    if (b_type == 1) {  // GGML_TYPE_F16 fallback
        const float alpha_f = 1.0f;
        const float beta_f = 0.0f;
        cublasGemmEx(cublas_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     n, m, k,
                     &alpha_f,
                     b, CUDA_R_16F, k,
                     a, CUDA_R_16F, k,
                     &beta_f,
                     out, CUDA_R_16F, n,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else if (b_type == 30) {  // GGML_TYPE_BF16
        // Mixed-precision: A=FP16, B=BF16, C=FP16, compute in FP32
        const float alpha_f = 1.0f;
        const float beta_f = 0.0f;
        cublasGemmEx(cublas_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     n, m, k,
                     &alpha_f,
                     b, CUDA_R_16BF, k,       // B is BF16
                     a, CUDA_R_16F, k,        // A is FP16
                     &beta_f,
                     out, CUDA_R_16F, n,      // C is FP16
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else if (b_type == 0) {  // GGML_TYPE_F32
        // F32 weights (norms, etc.); rare
        const float alpha_f = 1.0f;
        const float beta_f = 0.0f;
        cublasGemmEx(cublas_,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     n, m, k,
                     &alpha_f,
                     b, CUDA_R_32F, k,
                     a, CUDA_R_16F, k,
                     &beta_f,
                     out, CUDA_R_16F, n,
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else if (b_type == 12) {  // GGML_TYPE_Q4_K: dequant on the fly + cuBLAS
        // Dequant Q4_K packed data -> FP16 scratch buffer, then cuBLAS matmul
        int n_elements = n * k;
        launch_dequant_q4k(q4k_scratch_, b, n_elements, stream);
        // cuBLAS matmul with dequanted FP16 weight (same as type 1 path)
        if (use_lt_matmul_) {
            auto & cm = get_cached_matmul(m, n, k, stream);
            if (cm.tuned) {
                const float alpha = 1.0f, beta = 0.0f;
                cublasLtMatmul(cublaslt_, cm.op_desc,
                               &alpha, q4k_scratch_, cm.b_desc, a, cm.a_desc,
                               &beta, out, cm.c_desc, out, cm.c_desc,
                               &cm.algo, lt_workspace_, lt_workspace_size_, stream);
                // done
            } else {
                const float alpha_f = 1.0f, beta_f = 0.0f;
                cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                             &alpha_f, q4k_scratch_, CUDA_R_16F, k, a, CUDA_R_16F, k,
                             &beta_f, out, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
        } else {
            const float alpha_f = 1.0f, beta_f = 0.0f;
            cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                         &alpha_f, q4k_scratch_, CUDA_R_16F, k, a, CUDA_R_16F, k,
                         &beta_f, out, CUDA_R_16F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    } else if (b_type == MARLIN_TYPE) {
        // Marlin INT4: handled by marlin_matmul() with MarlinWeight struct
        // This path should not be reached; use marlin_matmul() directly
        fprintf(stderr, "[forward] ERROR: Marlin matmul called via generic dispatch\n");
        cudaMemsetAsync(out, 0, m * n * sizeof(half), stream);
    } else {
        fprintf(stderr, "[forward] WARNING: unsupported weight type %d\n", b_type);
        cudaMemsetAsync(out, 0, m * n * sizeof(half), stream);
    }
}

void ForwardEngine::marlin_matmul(half * out, const half * a, const MarlinWeight & w,
                                   int m, cudaStream_t stream) {
    static int dbg = 0;
    if (dbg < 5) {
        fprintf(stderr, "[marlin-mm] M=%d N=%d K=%d gs=%d packed=%p scales=%p\n",
                m, w.N, w.K, w.group_size, w.d_packed, w.d_scales);
        dbg++;
    }
    if (m > 16) {
        // Marlin handles M <= 16 only.
        // TODO: fall back to cuBLAS with FP16 dequant for M > 16; the output is currently zeroed.
        fprintf(stderr, "[marlin-mm] M=%d > 16, falling back\n", m);
        cudaMemsetAsync(out, 0, m * w.N * sizeof(half), stream);
        return;
    }
    int err = w.d_zp
        ? ::launch_marlin_matmul_awq(a, w.d_packed, out, w.d_scales, w.d_zp,
                                      m, w.N, w.K, w.group_size, (void *)stream)
        : ::launch_marlin_matmul(a, w.d_packed, out, w.d_scales,
                                  m, w.N, w.K, w.group_size, (void *)stream);
    if (err != 0) {
        fprintf(stderr, "[marlin-mm] FAILED: M=%d N=%d K=%d err=%d\n", m, w.N, w.K, err);
        // Check CUDA error
        cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) fprintf(stderr, "[marlin-mm] CUDA: %s\n", cudaGetErrorString(ce));
        cudaMemsetAsync(out, 0, m * w.N * sizeof(half), stream);
    }
}

void ForwardEngine::silu_mul(half * out, const half * gate, const half * up,
                              int n, int batch, cudaStream_t stream) {
    launch_silu_mul(out, gate, up, n, batch, stream);
}

void ForwardEngine::rope(half * q, half * k, const int32_t * positions,
                          int head_dim, int n_head_q, int n_head_k, int batch,
                          float freq_base, float freq_scale, cudaStream_t stream) {
    launch_rope(q, k, positions, head_dim, n_head_q, n_head_k, batch,
                freq_base, freq_scale, stream);
}

void ForwardEngine::add_residual(half * out, const half * residual,
                                  int n, int batch, cudaStream_t stream) {
    launch_add_residual(out, residual, n, batch, stream);
}

void ForwardEngine::embed_tokens(half * out, const int32_t * token_ids,
                                  int batch, cudaStream_t stream) {
    if (!tok_embd_) {
        cudaMemsetAsync(out, 0, batch * cfg_.n_embd * sizeof(half), stream);
        return;
    }
    if (tok_embd_qtype_) {
        launch_embed_tokens_quant(out, tok_embd_, token_ids, cfg_.n_embd, batch, tok_embd_qtype_, stream);
    } else {
        launch_embed_tokens(out, (const half *)tok_embd_, token_ids, cfg_.n_embd, batch, stream);
    }
}

// Marlin INT4 weight loading
// Loads GPTQ INT4 binary files produced by convert_awq_to_marlin.py
// Uploads to GPU and runs Marlin repack kernel.

extern "C" void launch_awq_marlin_repack(
    const uint32_t * b_q_weight, uint32_t * out,
    int size_k, int size_n, cudaStream_t stream);

extern "C" int launch_marlin_matmul_awq(
    const void * A, const void * B, void * C, const void * scales, const void * zp,
    int M, int N, int K, int group_size, void * stream_ptr);

static bool load_marlin_weight(ForwardEngine::MarlinWeight & mw,
                               const std::string & dir, const std::string & name,
                               int K, int N, int group_size) {
    // Load pre-processed files (generated by SGLang's awq_marlin_repack + marlin_permute_scales + awq_to_marlin_zero_points)
    std::string w_path = dir + "/" + name + "_marlin.bin";     // Marlin-repacked weights
    std::string s_path = dir + "/" + name + "_scales_perm.bin"; // permuted scales
    std::string z_path = dir + "/" + name + "_zp_marlin.bin";   // Marlin zero-points

    int n_groups = K / group_size;
    int tile_size = 16;
    int awq_cols = N / 8;

    // Load Marlin-repacked weights
    size_t marlin_size = (size_t)(K / tile_size) * (N * tile_size / 8);
    size_t w_bytes = marlin_size * sizeof(uint32_t);
    std::vector<uint32_t> h_w(marlin_size);
    FILE * f = fopen(w_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[marlin] cannot open %s\n", w_path.c_str()); return false; }
    fread(h_w.data(), 1, w_bytes, f); fclose(f);
    cudaMalloc(&mw.d_packed, w_bytes);
    cudaMemcpy(mw.d_packed, h_w.data(), w_bytes, cudaMemcpyHostToDevice);

    // Load permuted scales
    size_t s_bytes = (size_t)n_groups * N * sizeof(half);
    std::vector<half> h_s(n_groups * N);
    f = fopen(s_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[marlin] cannot open %s\n", s_path.c_str()); return false; }
    fread(h_s.data(), 1, s_bytes, f); fclose(f);
    cudaMalloc(&mw.d_scales, s_bytes);
    cudaMemcpy(mw.d_scales, h_s.data(), s_bytes, cudaMemcpyHostToDevice);

    // Load Marlin zero-points
    size_t z_bytes = (size_t)n_groups * awq_cols * sizeof(uint32_t);
    std::vector<uint32_t> h_z(n_groups * awq_cols);
    f = fopen(z_path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[marlin] cannot open %s\n", z_path.c_str()); return false; }
    fread(h_z.data(), 1, z_bytes, f); fclose(f);
    cudaMalloc(&mw.d_zp, z_bytes);
    cudaMemcpy(mw.d_zp, h_z.data(), z_bytes, cudaMemcpyHostToDevice);

    mw.N = N;
    mw.K = K;
    mw.group_size = group_size;
    return true;
}

static bool load_fp16_bin(void ** d_ptr, const std::string & path, size_t n_elements) {
    size_t bytes = n_elements * sizeof(half);
    std::vector<half> h_data(n_elements);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    if (fread(h_data.data(), 1, bytes, f) != bytes) { fclose(f); return false; }
    fclose(f);
    cudaMalloc(d_ptr, bytes);
    cudaMemcpy(*d_ptr, h_data.data(), bytes, cudaMemcpyHostToDevice);
    return true;
}

bool ForwardEngine::init_marlin(const char * weight_dir, PagedKVPool * kv_pool) {
    kv_pool_ = kv_pool;

    // CUDA graphs are safe here: the Marlin kernel's cudaMemsetAsync(d_locks) is capturable, and
    // device queries are cached at init by marlin_kernel_init_standalone().

    // Load config.json
    std::string dir(weight_dir);
    std::string config_path = dir + "/config.json";
    FILE * f = fopen(config_path.c_str(), "r");
    if (!f) { fprintf(stderr, "[forward] cannot open %s\n", config_path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::string json_str(sz, '\0');
    fread(&json_str[0], 1, sz, f); fclose(f);

    // Minimal JSON parsing (extract numeric fields)
    auto get_int = [&](const std::string & key) -> int {
        auto pos = json_str.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = json_str.find(':', pos);
        return atoi(json_str.c_str() + pos + 1);
    };
    auto get_float = [&](const std::string & key) -> float {
        auto pos = json_str.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0f;
        pos = json_str.find(':', pos);
        return atof(json_str.c_str() + pos + 1);
    };

    cfg_.n_layers = get_int("n_layers");
    cfg_.n_embd = get_int("n_embd");
    cfg_.n_ff = get_int("n_ff");
    cfg_.n_head = get_int("n_head");
    cfg_.n_head_kv = get_int("n_head_kv");
    cfg_.head_dim = get_int("head_dim");
    cfg_.vocab_size = get_int("vocab_size");
    cfg_.rms_norm_eps = get_float("rms_norm_eps");
    cfg_.rope_freq_base = get_float("rope_freq_base");
    cfg_.rope_freq_scale = 1.0f;
    int group_size = get_int("group_size");
    if (group_size == 0) group_size = 128;

    fprintf(stderr, "[forward-marlin] config: %d layers, %d embd, %d ff, %d heads (%d kv), vocab=%d, group=%d\n",
            cfg_.n_layers, cfg_.n_embd, cfg_.n_ff, cfg_.n_head, cfg_.n_head_kv, cfg_.vocab_size, group_size);

    // Initialize cuBLAS (for output projection which may stay FP16)
    cublasCreate(&cublas_);
    cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);
    cublasLtCreate(&cublaslt_);
    cudaMalloc(&lt_workspace_, lt_workspace_size_);

    // Initialize Marlin kernel (set max shared memory)
    ::marlin_kernel_init_standalone();

    // Allocate workspace
    int max_batch = kv_pool ? kv_pool->max_seqs : 8;
    if (!alloc_workspace(max_batch)) return false;

    // Load global FP16 weights
    void * d_embd = nullptr, * d_onorm = nullptr, * d_output = nullptr;
    load_fp16_bin(&d_embd, dir + "/token_embd.bin", (size_t)cfg_.vocab_size * cfg_.n_embd);
    load_fp16_bin(&d_onorm, dir + "/output_norm.bin", cfg_.n_embd);
    load_fp16_bin(&d_output, dir + "/output.bin", (size_t)cfg_.vocab_size * cfg_.n_embd);
    tok_embd_ = d_embd;
    output_norm_ = d_onorm;
    output_ = d_output;

    // Load per-layer Marlin weights
    layer_weights_.resize(cfg_.n_layers);
    int q_dim = cfg_.n_head * cfg_.head_dim;
    int kv_dim = cfg_.n_head_kv * cfg_.head_dim;

    for (int il = 0; il < cfg_.n_layers; il++) {
        auto & lw = layer_weights_[il];
        std::string pre = "l" + std::to_string(il);

        // Norms (FP16)
        void * d_an = nullptr, * d_fn = nullptr;
        load_fp16_bin(&d_an, dir + "/" + pre + "_attn_norm.bin", cfg_.n_embd);
        load_fp16_bin(&d_fn, dir + "/" + pre + "_ffn_norm.bin", cfg_.n_embd);
        lw.attn_norm = d_an; lw.attn_norm_type = 1;
        lw.ffn_norm = d_fn; lw.ffn_norm_type = 1;

        // Biases (FP16; required for Qwen, magnitudes up to ~130)
        void * d_qb = nullptr, * d_kb = nullptr, * d_vb = nullptr;
        load_fp16_bin(&d_qb, dir + "/" + pre + "_q_bias.bin", q_dim);
        load_fp16_bin(&d_kb, dir + "/" + pre + "_k_bias.bin", kv_dim);
        load_fp16_bin(&d_vb, dir + "/" + pre + "_v_bias.bin", kv_dim);
        lw.bq = d_qb; lw.bk = d_kb; lw.bv = d_vb;

        // Marlin INT4 weights (AWQ format, repacked via awq_marlin_repack)
        lw.wq_type = MARLIN_TYPE;
        lw.ffn_type = MARLIN_TYPE;

        bool ok = true;
        ok &= load_marlin_weight(lw.mq, dir, pre + "_q", cfg_.n_embd, q_dim, group_size);
        ok &= load_marlin_weight(lw.mk, dir, pre + "_k", cfg_.n_embd, kv_dim, group_size);
        ok &= load_marlin_weight(lw.mv, dir, pre + "_v", cfg_.n_embd, kv_dim, group_size);
        ok &= load_marlin_weight(lw.mo, dir, pre + "_o", q_dim, cfg_.n_embd, group_size);
        ok &= load_marlin_weight(lw.m_gate, dir, pre + "_gate", cfg_.n_embd, cfg_.n_ff, group_size);
        ok &= load_marlin_weight(lw.m_up, dir, pre + "_up", cfg_.n_embd, cfg_.n_ff, group_size);
        ok &= load_marlin_weight(lw.m_down, dir, pre + "_down", cfg_.n_ff, cfg_.n_embd, group_size);

        if (!ok) {
            fprintf(stderr, "[forward-marlin] failed to load layer %d weights\n", il);
            return false;
        }

        if (il == 0) fprintf(stderr, "[forward-marlin] layer 0 loaded (7 Marlin INT4 weights via AWQ repack)\n");
    }

    // Fuse QKV and gate+up Marlin weights (concatenate along N dimension)
    {
        int q_dim_f = cfg_.n_head * cfg_.head_dim;
        int kv_dim_f = cfg_.n_head_kv * cfg_.head_dim;
        int qkv_dim = q_dim_f + 2 * kv_dim_f;
        int gate_up_dim = 2 * cfg_.n_ff;
        int tile_sz = 16, pack_factor = 8;

        for (int il = 0; il < cfg_.n_layers; il++) {
            auto & lw = layer_weights_[il];

            // Fused QKV: concat packed weights, permuted scales, and zero-points along N
            {
                auto & mq = lw.mq, & mk = lw.mk, & mv = lw.mv;
                auto & mf = lw.mqkv_fused;
                mf.K = cfg_.n_embd; mf.N = qkv_dim; mf.group_size = group_size;

                // Packed weights: Marlin tiles are [k_tile * n_tiles + n_tile] order
                // Must interleave: for each k_tile, copy Q tiles then K then V tiles
                int k_tiles = mq.K / tile_sz;
                int q_ntiles = mq.N * tile_sz / pack_factor;  // tile_size_val per k_tile for Q
                int k_ntiles = mk.N * tile_sz / pack_factor;
                int v_ntiles = mv.N * tile_sz / pack_factor;
                int f_ntiles = q_ntiles + k_ntiles + v_ntiles;
                size_t tile_bytes = sizeof(uint32_t);  // each element is uint32
                size_t fused_packed = (size_t)k_tiles * f_ntiles * tile_bytes;
                cudaMalloc(&mf.d_packed, fused_packed);
                for (int kt = 0; kt < k_tiles; kt++) {
                    size_t dst_off = (size_t)kt * f_ntiles * tile_bytes;
                    cudaMemcpy((char*)mf.d_packed + dst_off,
                               (char*)mq.d_packed + (size_t)kt * q_ntiles * tile_bytes,
                               q_ntiles * tile_bytes, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_packed + dst_off + q_ntiles * tile_bytes,
                               (char*)mk.d_packed + (size_t)kt * k_ntiles * tile_bytes,
                               k_ntiles * tile_bytes, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_packed + dst_off + (q_ntiles + k_ntiles) * tile_bytes,
                               (char*)mv.d_packed + (size_t)kt * v_ntiles * tile_bytes,
                               v_ntiles * tile_bytes, cudaMemcpyDeviceToDevice);
                }

                // Permuted scales: [n_groups, N] FP16 -> concat along N
                int ng = cfg_.n_embd / group_size;
                size_t q_sc = (size_t)ng * mq.N * 2, k_sc = (size_t)ng * mk.N * 2, v_sc = (size_t)ng * mv.N * 2;
                cudaMalloc(&mf.d_scales, q_sc + k_sc + v_sc);
                // Row-by-row concat: for each group, copy q scales then k then v
                for (int g = 0; g < ng; g++) {
                    cudaMemcpy((char*)mf.d_scales + g * qkv_dim * 2,
                               (char*)mq.d_scales + g * mq.N * 2, mq.N * 2, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_scales + g * qkv_dim * 2 + mq.N * 2,
                               (char*)mk.d_scales + g * mk.N * 2, mk.N * 2, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_scales + g * qkv_dim * 2 + (mq.N + mk.N) * 2,
                               (char*)mv.d_scales + g * mv.N * 2, mv.N * 2, cudaMemcpyDeviceToDevice);
                }

                // Zero-points: [n_groups, N/8] int32 -> concat along N/8
                int q_zpc = mq.N/8, k_zpc = mk.N/8, v_zpc = mv.N/8, f_zpc = qkv_dim/8;
                cudaMalloc(&mf.d_zp, (size_t)ng * f_zpc * 4);
                for (int g = 0; g < ng; g++) {
                    cudaMemcpy((char*)mf.d_zp + g * f_zpc * 4,
                               (char*)mq.d_zp + g * q_zpc * 4, q_zpc * 4, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_zp + g * f_zpc * 4 + q_zpc * 4,
                               (char*)mk.d_zp + g * k_zpc * 4, k_zpc * 4, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_zp + g * f_zpc * 4 + (q_zpc + k_zpc) * 4,
                               (char*)mv.d_zp + g * v_zpc * 4, v_zpc * 4, cudaMemcpyDeviceToDevice);
                }
                lw.qkv_out_dim = qkv_dim;
            }

            // Fused gate+up
            {
                auto & mg = lw.m_gate, & mu = lw.m_up;
                auto & mf = lw.m_gate_up_fused;
                mf.K = cfg_.n_embd; mf.N = gate_up_dim; mf.group_size = group_size;

                int k_tiles2 = mg.K / tile_sz;
                int g_ntiles = mg.N * tile_sz / pack_factor;
                int u_ntiles = mu.N * tile_sz / pack_factor;
                int f_ntiles2 = g_ntiles + u_ntiles;
                size_t fused_packed2 = (size_t)k_tiles2 * f_ntiles2 * sizeof(uint32_t);
                cudaMalloc(&mf.d_packed, fused_packed2);
                for (int kt = 0; kt < k_tiles2; kt++) {
                    size_t dst = (size_t)kt * f_ntiles2 * sizeof(uint32_t);
                    cudaMemcpy((char*)mf.d_packed + dst,
                               (char*)mg.d_packed + (size_t)kt * g_ntiles * sizeof(uint32_t),
                               g_ntiles * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_packed + dst + g_ntiles * sizeof(uint32_t),
                               (char*)mu.d_packed + (size_t)kt * u_ntiles * sizeof(uint32_t),
                               u_ntiles * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
                }

                int ng = cfg_.n_embd / group_size;
                cudaMalloc(&mf.d_scales, (size_t)ng * gate_up_dim * 2);
                for (int g2 = 0; g2 < ng; g2++) {
                    cudaMemcpy((char*)mf.d_scales + g2 * gate_up_dim * 2,
                               (char*)mg.d_scales + g2 * mg.N * 2, mg.N * 2, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_scales + g2 * gate_up_dim * 2 + mg.N * 2,
                               (char*)mu.d_scales + g2 * mu.N * 2, mu.N * 2, cudaMemcpyDeviceToDevice);
                }
                int g_zpc = mg.N/8, u_zpc = mu.N/8, f_zpc2 = gate_up_dim/8;
                cudaMalloc(&mf.d_zp, (size_t)ng * f_zpc2 * 4);
                for (int g2 = 0; g2 < ng; g2++) {
                    cudaMemcpy((char*)mf.d_zp + g2 * f_zpc2 * 4,
                               (char*)mg.d_zp + g2 * g_zpc * 4, g_zpc * 4, cudaMemcpyDeviceToDevice);
                    cudaMemcpy((char*)mf.d_zp + g2 * f_zpc2 * 4 + g_zpc * 4,
                               (char*)mu.d_zp + g2 * u_zpc * 4, u_zpc * 4, cudaMemcpyDeviceToDevice);
                }
                lw.gate_up_out_dim = gate_up_dim;
            }
        }
        fprintf(stderr, "[forward-marlin] fused QKV (%d→%d) + gate_up (%d→%d) Marlin weights\n",
                q_dim_f, qkv_dim, cfg_.n_ff, gate_up_dim);
    }

    // Initialize batched pointer arrays (needed for non-fused cuBLAS path)
    batched_ptrs_.resize(cfg_.n_layers);

    // Debug / comparison switches (same as init())
    if (getenv("ALPHA_NO_FUSED")) { use_fused_kernels_ = false; fprintf(stderr, "[forward] ALPHA_NO_FUSED set\n"); }
    if (getenv("ALPHA_NO_LT"))    { use_lt_matmul_ = false;     fprintf(stderr, "[forward] ALPHA_NO_LT set\n"); }
    if (getenv("ALPHA_NO_FI_ATTN")) { use_flashinfer_attn_ = false; fprintf(stderr, "[forward] ALPHA_NO_FI_ATTN set\n"); }

    fprintf(stderr, "[forward-marlin] all %d layers loaded, INT4→FP16 dequant + cuBLAS ready\n", cfg_.n_layers);
    return true;
}

} // namespace alphaserve
