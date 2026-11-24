// SPDX-License-Identifier: Apache-2.0
// Program<Glm5Next>: GGUF arch "glm5-next" (GLM-5.3-Flash).
//
// The model, as the llama path builds it (upstream src/models/glm5-next.cpp):
//  * hc hyper-connection streams of n_embd, mixed per sub-layer by an mHC mixer (hc_*_fn / _base / _scale: pre and post
//    weights of hc each and an hc x hc combination made doubly stochastic by Sinkhorn iterations); unlike qwen4exp the
//    sub-layers keep their RMS norms (attn_norm, ffn_norm) and the head keeps output_norm;
//  * layers whose attention.head_count_kv entry is non-zero are MLA sparse attention (low-rank q through q_a/q_b, a
//    shared kv latent of kv_lora_rank expanded per head by k_b/v_b, no rotary part) with an indexer that picks top_k
//    past positions, its keys pooled kpool to one by a learned compressor; the rest are KDA linear attention (a gated
//    delta rule with per-channel q/k/v depthwise convolutions and low-rank f/g gates);
//  * the first leading_dense layers have a dense SwiGLU FFN, the rest a MoE (sigmoid scores + exp_probs_b bias for the
//    top-k choice, weights renormalised and scaled) plus one always-on shared expert; SwiGLU is clamped;
//  * nextn MTP block(s) after the trunk: an MLA + MoE layer without hyper-connections, fed by eh_proj([embd, hidden]).
//
// This header holds the spec and the shape check of every tensor against it (metadata only, no device). The weight
// binding and the forward follow in glm5next_run.
#pragma once

#include "core/weights.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

struct Glm5NextSpec {
    std::string arch;
    int     n_layer = 0;          // trunk layers (block_count - nextn)
    int     nextn   = 0;          // MTP blocks after the trunk
    int64_t n_embd = 0, n_vocab = 0, n_ff = 0;
    int     n_head = 0;
    float   rms_eps = 0;
    int     n_ctx_train = 0;
    // MoE
    int     n_expert = 0, n_expert_used = 0, n_shexp = 0, n_dense_lead = 0;
    int64_t n_ff_exp = 0, n_ff_shexp = 0;
    float   exp_scale = 0;
    bool    exp_norm = false;
    // MLA
    int     q_lora = 0, kv_lora = 0, mla_k = 0, mla_v = 0;       // q rank, kv latent rank, per-head key / value length
    // KDA
    int     conv_k = 0, kda_head_dim = 0;
    float   kda_gate_lb = 0;
    // indexer
    int     idx_n_head = 0, idx_head_dim = 0, idx_top_k = 0, idx_kpool = 0;
    bool    idx_share_mtp = false;
    // mHC
    int     hc = 0, hc_sinkhorn = 0;
    std::vector<bool> is_mla;     // per block, the MTP block(s) included

    int     n_block() const { return n_layer + nextn; }
    int64_t hc_dim() const { return (int64_t) hc * n_embd; }
    int64_t hc_fn_out() const { return (int64_t) hc * hc + 2 * hc; }         // comb (hc x hc) + pre (hc) + post (hc)
    int64_t kda_inner() const { return (int64_t) n_head * kda_head_dim; }
    bool    is_moe(int il) const { return il >= n_dense_lead; }
    bool    has_hc(int il) const { return il < n_layer; }

    static Glm5NextSpec load(const WeightSet & w);
    static bool supports(const std::string & arch) { return arch == "glm5-next"; }
};

// Bytes by tensor class, filled by glm5next_check().
struct Glm5NextBudget {
    uint64_t experts = 0, shared = 0, router = 0, dense_ffn = 0, mla = 0, kda = 0, indexer = 0, hc = 0, norms = 0,
             embd = 0, head = 0, mtp = 0, other = 0;
    uint64_t exp_unit[3] = {0, 0, 0};   // one expert's [gate, up, down] bytes, median layer
    uint64_t total() const {
        return experts + shared + router + dense_ffn + mla + kda + indexer + hc + norms + embd + head + mtp + other;
    }
    uint64_t resident_min() const { return total() - experts; }   // everything but the routed experts
};

// Every tensor the forward needs is looked up by name and its shape compared with the spec; a missing, renamed or
// reshaped tensor throws with its name. Tensors in the file that no binding names are returned in `unbound`.
Glm5NextBudget glm5next_check(const WeightSet & w, Glm5NextSpec & s, std::vector<std::string> * unbound = nullptr);

} // namespace ai
