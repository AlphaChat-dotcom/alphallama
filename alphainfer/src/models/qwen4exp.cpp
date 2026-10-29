// SPDX-License-Identifier: Apache-2.0
#include "models/qwen4exp.h"

#include <cstdlib>
#include <stdexcept>

namespace ai {

Qwen4ExpSpec Qwen4ExpSpec::load(const WeightSet & w) {
    Qwen4ExpSpec s;
    s.arch = w.arch();
    if (!supports(s.arch)) {
        throw std::runtime_error("Qwen4ExpSpec: arch '" + s.arch + "' is not qwen4exp");
    }
    const std::string a = s.arch + ".";
    const int n_layer_all = (int) w.get_int(a + "block_count");
    s.nextn       = (int) w.get_int(a + "nextn_predict_layers", 0);
    s.n_layer     = n_layer_all - s.nextn;
    s.n_embd      = w.get_int(a + "embedding_length");
    s.n_head      = (int) w.get_int(a + "attention.head_count");
    s.n_head_kv   = (int) w.get_int(a + "attention.head_count_kv");
    s.head_dim    = (int) w.get_int(a + "attention.key_length");
    s.n_rot       = (int) w.get_int(a + "rope.dimension_count");
    s.rope_base   = (float) w.get_float(a + "rope.freq_base");
    s.rms_eps     = (float) w.get_float(a + "attention.layer_norm_rms_epsilon");
    s.n_ctx_train = (int) w.get_int(a + "context_length");
    s.n_expert      = (int) w.get_int(a + "expert_count");
    s.n_expert_used = (int) w.get_int(a + "expert_used_count");
    s.n_ff_exp      = w.get_int(a + "expert_feed_forward_length", 0);
    s.n_ff_shexp    = w.get_int(a + "expert_shared_feed_forward_length", 0);
    s.conv_k      = (int) w.get_int(a + "ssm.conv_kernel");
    s.ssm_state   = (int) w.get_int(a + "ssm.state_size");
    s.ssm_groups  = (int) w.get_int(a + "ssm.group_count");
    s.ssm_dt_rank = (int) w.get_int(a + "ssm.time_step_rank");
    s.ssm_inner   = (int) w.get_int(a + "ssm.inner_size");
    if (!s.conv_k || !s.ssm_state || !s.ssm_groups || !s.ssm_dt_rank || !s.ssm_inner) {
        throw std::runtime_error("qwen4exp: zero ssm.* metadata");
    }
    s.full_attn_interval = (int) w.get_int(a + "full_attention_interval", 4);
    s.hc    = (int) w.get_int(a + "hyper_connection.count");
    s.hc_lr = (int) w.get_int(a + "hyper_connection.low_rank");
    if (s.hc <= 1 || s.hc_lr <= 0) {
        throw std::runtime_error("qwen4exp: hyper_connection.count must be > 1 and .low_rank > 0");
    }
    s.idx_n_head   = (int) w.get_int(a + "attention.indexer.head_count");
    s.idx_head_dim = (int) w.get_int(a + "attention.indexer.key_length");
    s.idx_top_k    = (int) w.get_int(a + "attention.indexer.top_k");
    if (!s.idx_n_head || !s.idx_head_dim || !s.idx_top_k) {
        throw std::runtime_error("qwen4exp: zero attention.indexer.* metadata");
    }
    s.compress_ratios = w.has_key(a + "attention.compress_ratios") ? w.get_int_array(a + "attention.compress_ratios")
                                                                    : std::vector<int64_t>((size_t) s.n_layer, 0);
    if (s.compress_ratios.size() == 1) s.compress_ratios.assign((size_t) n_layer_all, s.compress_ratios[0]);
    const auto sec = w.get_int_array(a + "rope.dimension_sections");
    for (size_t i = 0; i < 4 && i < sec.size(); ++i) s.sections[i] = (int) sec[i];
    if (w.has_key(a + "ple.layers")) {
        s.ple_layers          = w.get_int_array(a + "ple.layers");
        s.ple_ngram           = (int) w.get_int(a + "ple.ngram_size");
        s.ple_heads_per_ngram = (int) w.get_int(a + "ple.heads_per_ngram");
        s.ple_conv_k          = (int) w.get_int(a + "ple.conv_kernel");
        s.ple_eos             = w.get_int(a + "ple.eos_token_id");
        s.ple_image           = w.get_int(a + "ple.image_token_id", s.ple_eos);
        s.ple_head_dim        = (int) w.get_int(a + "embedding_length_per_layer_input");
        if (!s.ple_conv_k || !s.ple_head_dim) {
            throw std::runtime_error("qwen4exp: zero ple.conv_kernel / embedding_length_per_layer_input");
        }
        if (s.ple_ngram < 2 || s.ple_n_heads() <= 0) {
            throw std::runtime_error("qwen4exp: PLE n-gram size / head count out of range");
        }
        s.ple_multipliers  = w.get_int_array(a + "ple.layer_multipliers");
        s.ple_head_offsets = w.get_int_array(a + "ple.head_offsets");
        s.ple_head_vocab   = w.get_int_array(a + "ple.head_vocab_sizes");
        for (int64_t il : s.ple_layers) {
            if (il < 0 || il >= s.n_layer) throw std::runtime_error("qwen4exp: ple.layers entry out of range");
        }
    }
    s.is_attn.resize((size_t) s.n_layer);
    for (int il = 0; il < s.n_layer; ++il) s.is_attn[(size_t) il] = ((il + 1) % s.full_attn_interval) == 0;
    for (const auto & m : w.meta()) {
        if (m.name == "token_embd.weight") s.n_vocab = m.t.ne[1];
    }
    return s;
}

static std::string blk(int il, const char * name) { return "blk." + std::to_string(il) + "." + name; }

bool Qwen4Exp::load(Device & dev, const std::string & path, bool verbose, bool offload_experts) {
    if (!w_.open(path)) {
        return false;
    }
    spec_ = Qwen4ExpSpec::load(w_);
    const Qwen4ExpSpec & s = spec_;
    WeightSet::HostFilter host = [offload_experts](const std::string & n) {
        if (n.rfind("per_layer_token_embd", 0) == 0) return true;   // ~29 GB, gathered by row on the host
        return offload_experts && n.find("_exps.weight") != std::string::npos;
    };
    // AI_Q4X_HC_BF16=1 keeps the hc projections BF16 (diagnostic; the llama path's equivalent is ALPHA_HC_Q8=0)
    static const bool hc_bf16 = getenv("AI_Q4X_HC_BF16") && atoi(getenv("AI_Q4X_HC_BF16")) != 0;
    WeightSet::ConvertFilter conv = [](const std::string & n, const Tensor & t) {
        if (hc_bf16) return -1;
        const bool hc_lr = (n.find("hc_") != std::string::npos) &&
                           (n.size() > 12 && (n.compare(n.size() - 12, 12, "_down.weight") == 0 ||
                                              n.compare(n.size() - 10, 10, "_up.weight") == 0));
        if (hc_lr && t.type == BF16 && t.ne[0] % 32 == 0) return (int) Q8_0;
        // AI_Q4X_BF16_F16=1: every other BF16 matrix (router, indexer q/k, ssm_alpha/beta, hc inject, PLE) -> F16,
        // lossless, so the T <= 8 matmuls take the F16 matrix-vector path instead of cutlass BF16 + 2 converts per call
        static const bool bf16_f16 = getenv("AI_Q4X_BF16_F16") && atoi(getenv("AI_Q4X_BF16_F16")) != 0;
        if (bf16_f16 && t.type == BF16 && t.ne[1] > 1) return (int) F16;
        return -1;
    };
    if (!w_.load(dev, 4ULL << 30, verbose, host, {}, conv, /*host_mmap=*/ true)) {
        return false;
    }
    // shape check: every binding is named and compared against the spec, so a renamed or reshaped tensor fails here
    auto need = [&](const std::string & name, std::initializer_list<int64_t> ne) -> Tensor {
        const Tensor * t = w_.find(name);
        if (!t) throw std::runtime_error("qwen4exp: missing tensor " + name);
        size_t d = 0;
        for (int64_t e : ne) {
            if (e >= 0 && t->ne[d] != e) {
                throw std::runtime_error("qwen4exp: " + name + " has shape " + t->shape_str() + ", expected dim " +
                                         std::to_string(d) + " = " + std::to_string(e));
            }
            d++;
        }
        return *t;
    };
    const int64_t N = s.n_embd, D = s.hc_dim(), lr = s.hc_lr, hc = s.hc;
    const int64_t Dh = s.head_dim, Hq = s.n_head, Hkv = s.n_head_kv;
    const int64_t conv_c = s.conv_channels(), Hv = s.ssm_dt_rank, Sv = s.ssm_inner / s.ssm_dt_rank;
    const int64_t E = s.n_expert, F = s.n_ff_exp, Fs = s.n_ff_shexp;
    tok_embd_  = need("token_embd.weight", { N, s.n_vocab });
    output_    = need("output.weight", { N, s.n_vocab });
    ple_table_ = s.ple_layers.empty() ? Tensor{} : need("per_layer_token_embd.weight", { s.ple_head_dim, -1 });
    hc_head_.norm = need("output_hc_norm.weight", { D });
    hc_head_.down = need("output_hc_down.weight", { D, lr });
    hc_head_.up   = need("output_hc_up.weight", { lr, D });
    layers_.assign((size_t) s.n_layer, {});
    for (int il = 0; il < s.n_layer; ++il) {
        Qwen4ExpLayer & L = layers_[(size_t) il];
        L.attn = s.is_attn[(size_t) il];
        for (int64_t p : s.ple_layers) L.ple = L.ple || p == il;
        for (int part = 0; part < 2; ++part) {
            op::q4x::HcWeights & h = part ? L.hc_ffn : L.hc_attn;
            const std::string pre  = part ? "hc_ffn_" : "hc_attn_";
            h.norm   = need(blk(il, (pre + "norm.weight").c_str()), { D });
            h.down   = need(blk(il, (pre + "down.weight").c_str()), { D, lr });
            h.up     = need(blk(il, (pre + "up.weight").c_str()), { lr, D });
            h.inject = need(blk(il, (pre + "inject.weight").c_str()), { D, hc });
        }
        if (L.attn) {
            L.at.q      = need(blk(il, "attn_q.weight"), { N, Dh * Hq * 2 });
            L.at.k      = need(blk(il, "attn_k.weight"), { N, Dh * Hkv });
            L.at.v      = need(blk(il, "attn_v.weight"), { N, Dh * Hkv });
            L.at.o      = need(blk(il, "attn_output.weight"), { Dh * Hq, N });
            L.at.q_norm = need(blk(il, "attn_q_norm.weight"), { Dh });
            L.at.k_norm = need(blk(il, "attn_k_norm.weight"), { Dh });
            L.qsa_p.idx_heads = s.idx_n_head;
            L.qsa_p.idx_dim   = s.idx_head_dim;
            L.qsa_p.top_k     = s.idx_top_k;
            L.qsa_p.ratio     = (size_t) il < s.compress_ratios.size() ? (int) s.compress_ratios[(size_t) il] : 0;
            L.qsa.q_proj = need(blk(il, "indexer.q_proj.weight"), { N, (int64_t) s.idx_n_head * s.idx_head_dim });
            L.qsa.k_proj = need(blk(il, "indexer.k_proj.weight"), { N, s.idx_head_dim });
            L.qsa.q_norm = need(blk(il, "indexer.q_norm.weight"), { s.idx_head_dim });
            L.qsa.k_norm = need(blk(il, "indexer.k_norm.weight"), { s.idx_head_dim });
        } else {
            L.gdn.qkv     = need(blk(il, "attn_qkv.weight"), { N, conv_c });
            L.gdn.z       = need(blk(il, "attn_gate.weight"), { N, s.ssm_inner });
            L.gdn.beta    = need(blk(il, "ssm_beta.weight"), { N, Hv });
            L.gdn.alpha   = need(blk(il, "ssm_alpha.weight"), { N, Hv });
            L.gdn.conv    = need(blk(il, "ssm_conv1d.weight"), { s.conv_k, conv_c });
            L.gdn.dt_bias = need(blk(il, "ssm_dt.bias"), { Hv });
            L.gdn.a       = need(blk(il, "ssm_a"), { Hv });
            L.gdn.norm    = need(blk(il, "ssm_norm.weight"), { Sv });
            L.gdn.out     = need(blk(il, "ssm_out.weight"), { s.ssm_inner, N });
        }
        if (L.ple) {
            L.plew.key        = need(blk(il, "ple_key.weight"), { N, D });
            L.plew.value      = need(blk(il, "ple_value.weight"), { N, N });
            L.plew.norm_key   = need(blk(il, "ple_norm_key.weight"), { D });
            L.plew.norm_query = need(blk(il, "ple_norm_query.weight"), { D });
            L.plew.norm_conv  = need(blk(il, "ple_norm_conv.weight"), { D });
            L.plew.conv1d     = need(blk(il, "ple_conv1d.weight"), { s.ple_conv_k, D });
        }
        L.moe.router    = need(blk(il, "ffn_gate_inp.weight"), { N, E });
        L.moe.gate_exps = need(blk(il, "ffn_gate_exps.weight"), { N, F, E });
        L.moe.up_exps   = need(blk(il, "ffn_up_exps.weight"), { N, F, E });
        L.moe.down_exps = need(blk(il, "ffn_down_exps.weight"), { F, N, E });
        L.moe.sh_gate   = need(blk(il, "ffn_gate_shexp.weight"), { N, Fs });
        L.moe.sh_up     = need(blk(il, "ffn_up_shexp.weight"), { N, Fs });
        L.moe.sh_down   = need(blk(il, "ffn_down_shexp.weight"), { Fs, N });
        L.moe.sh_score  = need(blk(il, "ffn_gate_inp_shexp.weight"), { N });
    }
    if (verbose) {
        fprintf(stderr, "[ai] Program<Qwen4Exp>: %d layers bound (shapes checked), device weights %.2f GiB, host-mapped "
                        "%.2f GiB\n", s.n_layer, w_.bytes() / 1073741824.0, w_.host_bytes() / 1073741824.0);
    }
    return true;
}

}  // namespace ai
