// Program<Qwen4Exp>: GGUF arch "qwen4exp".
//
// The model, as the llama path builds it (src/models/qwen4exp.cpp; llama-model.cpp LLM_ARCH_QWEN4EXP):
//  * hc hyper-connection streams of n_embd (hc_dim = hc * n_embd); no attn/ffn/output norms -- the HC mixers replace them;
//  * layers with (il+1) % full_attention_interval == 0 are gated full attention (QSA sparse attention through an
//    indexer where compress_ratios[il] > 0), the rest gated DeltaNet (output gate sigmoid(z), not silu);
//  * every layer: HC read -> token mixer -> HC scatter -> HC read -> MoE (softmax top-k, renormalised) + a shared expert
//    gated by sigmoid(ffn_gate_inp_shexp . x) -> HC scatter; the head is a final HC mixer -> output;
//  * PLE on the layers in ple.layers: rows of per_layer_token_embd (host-only, ~29 GB) picked by host-side n-gram hashing.
//
// This header holds the spec, the metadata loader and the weight binding; the forward is Qwen4ExpRun.
#pragma once

#include "core/device.h"
#include "core/weights.h"
#include "ops/q4x_ops.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

struct Qwen4ExpSpec {
    std::string arch;
    int     n_layer = 0;          // trunk layers (block_count - nextn)
    int     nextn   = 0;          // MTP blocks after the trunk (0 in the trunk GGUF; the sidecar adds layer n_layer)
    int64_t n_embd = 0, n_vocab = 0;
    int     n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    int     sections[4] = {0, 0, 0, 0};
    float   rope_base = 0, rms_eps = 0;
    int     n_ctx_train = 0;
    int     n_expert = 0, n_expert_used = 0;
    int64_t n_ff_exp = 0, n_ff_shexp = 0;
    int     conv_k = 0, ssm_state = 0, ssm_groups = 0, ssm_dt_rank = 0, ssm_inner = 0;
    int     full_attn_interval = 4;
    int     hc = 0, hc_lr = 0;                                   // hyper_connection.count / .low_rank
    int     idx_n_head = 0, idx_head_dim = 0, idx_top_k = 0;     // attention.indexer.*
    std::vector<int64_t> compress_ratios;                        // per layer; > 0 = QSA on that attention layer
    // PLE (absent on files without ple.layers)
    std::vector<int64_t> ple_layers;
    int     ple_ngram = 0, ple_heads_per_ngram = 0, ple_conv_k = 0, ple_head_dim = 0;
    int64_t ple_eos = -1, ple_image = -1;
    std::vector<int64_t> ple_multipliers, ple_head_offsets, ple_head_vocab;   // u64 in the file, bit-cast to i64
    std::vector<bool>    is_attn;   // per trunk layer

    int64_t hc_dim() const { return (int64_t) hc * n_embd; }
    int     ple_n_heads() const { return (ple_ngram - 1) * ple_heads_per_ngram; }
    int64_t conv_channels() const { return (int64_t) ssm_inner + 2LL * ssm_groups * ssm_state; }

    // Reads every key the llama path reads for this arch and checks the invariants it asserts. Throws with the key named.
    static Qwen4ExpSpec load(const WeightSet & w);
    static bool supports(const std::string & arch) { return arch == "qwen4exp"; }
};

// Per-layer weights, bound to the Op structs (tensor names as listed by ai-q4x-inspect --names).
struct Qwen4ExpLayer {
    bool                 attn = false;   // gated full attention (+ QSA when qsa_p.ratio > 0); else gated DeltaNet
    bool                 ple  = false;
    op::q4x::HcWeights   hc_attn, hc_ffn;
    op::AttnWeights      at;
    op::q4x::QsaWeights  qsa;
    op::q4x::QsaParams   qsa_p;
    op::GdnWeights       gdn;
    op::MoeWeights       moe;
    op::q4x::PleWeights  plew;
};

// The MTP draft block (qwen4exp-mtp-sidecar GGUF, blk.<n_layer>.*): one trunk-shaped attention layer (dense,
// QSA ratio 0) whose MoE keeps all 512 experts on the device, plus its own input projections and head mixer.
struct Qwen4ExpMtp {
    bool               valid = false;
    Qwen4ExpLayer      L;
    Tensor             eh_proj, h_proj, enorm, hnorm;
    op::q4x::HcWeights head;
};

class Qwen4Exp {
public:
    static bool supports(const std::string & arch) { return Qwen4ExpSpec::supports(arch); }

    // offload_experts: routed experts and the PLE table stay in host memory, mapped from the file rather than copied
    // (the host side can be ~72 GB). hc down/up projections are converted BF16 -> Q8_0 at load (as ALPHA_HC_Q8 does on
    // the llama path).
    // Every bound tensor's shape is checked against the spec; a missing or mis-shaped tensor throws with its name.
    bool load(Device & dev, const std::string & path, bool verbose, bool offload_experts = true);

    const Qwen4ExpSpec &               spec() const { return spec_; }
    const WeightSet &                  weights() const { return w_; }
    WeightSet &                        weights() { return w_; }
    std::vector<Qwen4ExpLayer> &       layers_mut() { return layers_; }   // RAM tier: expert pointers into the tier
    const std::vector<Qwen4ExpLayer> & layers() const { return layers_; }
    const Tensor & tok_embd() const { return tok_embd_; }
    const Tensor & output() const { return output_; }
    const Tensor & ple_table() const { return ple_table_; }
    const op::q4x::HcWeights & hc_head() const { return hc_head_; }
    // Load the MTP sidecar (every tensor on the device; hc down/up BF16 -> Q8_0 as for the trunk)
    bool load_mtp(Device & dev, const std::string & path, bool verbose);
    const Qwen4ExpMtp & mtp() const { return mtp_; }

private:
    Qwen4ExpSpec               spec_;
    WeightSet                  w_;
    std::vector<Qwen4ExpLayer> layers_;
    Tensor                     tok_embd_, output_, ple_table_;
    op::q4x::HcWeights         hc_head_;
    WeightSet                  mtp_w_;
    Qwen4ExpMtp                mtp_;
};

}  // namespace ai
