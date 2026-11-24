// SPDX-License-Identifier: Apache-2.0
#include "models/glm5next.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace ai {

Glm5NextSpec Glm5NextSpec::load(const WeightSet & w) {
    Glm5NextSpec s;
    s.arch = w.arch();
    if (!supports(s.arch)) {
        throw std::runtime_error("Glm5NextSpec: arch '" + s.arch + "' is not glm5-next");
    }
    const std::string a = s.arch + ".";
    const int n_block = (int) w.get_int(a + "block_count");
    s.nextn       = (int) w.get_int(a + "nextn_predict_layers", 0);
    s.n_layer     = n_block - s.nextn;
    s.n_embd      = w.get_int(a + "embedding_length");
    s.n_ff        = w.get_int(a + "feed_forward_length");
    s.n_head      = (int) w.get_int(a + "attention.head_count");
    s.rms_eps     = (float) w.get_float(a + "attention.layer_norm_rms_epsilon");
    s.n_ctx_train = (int) w.get_int(a + "context_length");
    s.n_expert      = (int) w.get_int(a + "expert_count");
    s.n_expert_used = (int) w.get_int(a + "expert_used_count");
    s.n_shexp       = (int) w.get_int(a + "expert_shared_count", 0);
    s.n_ff_exp      = w.get_int(a + "expert_feed_forward_length");
    s.n_ff_shexp    = w.get_int(a + "expert_shared_feed_forward_length", 0);
    s.n_dense_lead  = (int) w.get_int(a + "leading_dense_block_count", 0);
    s.exp_scale     = (float) w.get_float(a + "expert_weights_scale");
    s.exp_norm      = w.get_int(a + "expert_weights_norm", 0) != 0;
    if (w.get_int(a + "expert_gating_func") != 2) {
        throw std::runtime_error("glm5-next: expert_gating_func is not sigmoid (2)");
    }
    if (s.n_shexp != 1) {
        throw std::runtime_error("glm5-next: expected exactly one shared expert");
    }
    s.q_lora  = (int) w.get_int(a + "attention.q_lora_rank");
    s.kv_lora = (int) w.get_int(a + "attention.kv_lora_rank");
    s.mla_k   = (int) w.get_int(a + "attention.key_length_mla");
    s.mla_v   = (int) w.get_int(a + "attention.value_length_mla");
    if (w.get_int(a + "rope.dimension_count", 0) != 0) {
        throw std::runtime_error("glm5-next: a rotary part in the MLA keys is not implemented (rope.dimension_count != 0)");
    }
    s.conv_k       = (int) w.get_int(a + "ssm.conv_kernel");
    s.kda_head_dim = (int) w.get_int(a + "kda.head_dim");
    s.kda_gate_lb  = (float) w.get_float(a + "kda.gate_lower_bound");
    s.idx_n_head    = (int) w.get_int(a + "attention.indexer.head_count");
    s.idx_head_dim  = (int) w.get_int(a + "attention.indexer.key_length");
    s.idx_top_k     = (int) w.get_int(a + "attention.indexer.top_k");
    s.idx_kpool     = (int) w.get_int(a + "attention.indexer.kpool", 1);
    s.idx_share_mtp = w.get_int(a + "attention.indexer.index_share_mtp", 0) != 0;
    s.hc          = (int) w.get_int(a + "hyper_connection.count");
    s.hc_sinkhorn = (int) w.get_int(a + "hyper_connection.sinkhorn_iterations");
    if (s.hc <= 1) {
        throw std::runtime_error("glm5-next: hyper_connection.count must be > 1");
    }
    const auto kv = w.get_int_array(a + "attention.head_count_kv");
    if ((int) kv.size() != n_block) {
        throw std::runtime_error("glm5-next: attention.head_count_kv has " + std::to_string(kv.size()) + " entries for " +
                                 std::to_string(n_block) + " blocks");
    }
    s.is_mla.resize((size_t) n_block);
    for (int il = 0; il < n_block; ++il) s.is_mla[(size_t) il] = kv[(size_t) il] != 0;
    for (int il = s.n_layer; il < n_block; ++il) {
        if (!s.is_mla[(size_t) il]) throw std::runtime_error("glm5-next: an MTP block that is not MLA is not implemented");
    }
    for (const auto & m : w.meta()) {
        if (m.name == "token_embd.weight") s.n_vocab = m.t.ne[1];
    }
    return s;
}

static std::string blk(int il, const std::string & name) { return "blk." + std::to_string(il) + "." + name; }

Glm5NextBudget glm5next_check(const WeightSet & w, Glm5NextSpec & s, std::vector<std::string> * unbound) {
    std::map<std::string, WeightSet::TensorMeta> by_name;
    for (const auto & m : w.meta()) by_name[m.name] = m;
    if (!s.n_vocab) {
        auto it = by_name.find("token_embd.weight");
        if (it != by_name.end()) s.n_vocab = it->second.t.ne[1];
    }
    std::set<std::string> used;
    Glm5NextBudget b;
    // need(name, shape, &class bytes): -1 in the shape = not checked
    auto need = [&](const std::string & name, std::initializer_list<int64_t> ne, uint64_t & cls) -> uint64_t {
        auto it = by_name.find(name);
        if (it == by_name.end()) throw std::runtime_error("glm5-next: missing tensor " + name);
        const Tensor & t = it->second.t;
        size_t d = 0;
        for (int64_t e : ne) {
            if (e >= 0 && t.ne[d] != e) {
                throw std::runtime_error("glm5-next: " + name + " has shape " + t.shape_str() + ", expected dim " +
                                         std::to_string(d) + " = " + std::to_string(e));
            }
            d++;
        }
        used.insert(name);
        cls += it->second.nbytes;
        return it->second.nbytes;
    };
    const int64_t N = s.n_embd, H = s.n_head, D = s.hc_dim(), HF = s.hc_fn_out();
    const int64_t KI = s.kda_inner(), E = s.n_expert, F = s.n_ff_exp, Fs = s.n_ff_shexp;
    const int64_t IH = (int64_t) s.idx_n_head * s.idx_head_dim;
    need("token_embd.weight", { N, s.n_vocab }, b.embd);
    need("output.weight", { N, s.n_vocab }, b.head);
    need("output_norm.weight", { N }, b.norms);
    std::vector<uint64_t> ug, uu, ud;
    for (int il = 0; il < s.n_block(); ++il) {
        const bool mtp = il >= s.n_layer;
        uint64_t scratch = 0;
        uint64_t & c_norm = mtp ? b.mtp : b.norms;
        need(blk(il, "attn_norm.weight"), { N }, c_norm);
        need(blk(il, "ffn_norm.weight"), { N }, c_norm);
        if (s.has_hc(il)) {
            for (const char * part : { "hc_attn_", "hc_ffn_" }) {
                need(blk(il, std::string(part) + "fn.weight"), { D, HF }, b.hc);
                need(blk(il, std::string(part) + "base.weight"), { HF }, b.hc);
                need(blk(il, std::string(part) + "scale.weight"), { 3 }, b.hc);
            }
        }
        if (s.is_mla[(size_t) il]) {
            uint64_t & c = mtp ? b.mtp : b.mla;
            uint64_t & ci = mtp ? b.mtp : b.indexer;
            need(blk(il, "attn_q_a.weight"), { N, s.q_lora }, c);
            need(blk(il, "attn_q_a_norm.weight"), { s.q_lora }, c);
            need(blk(il, "attn_q_b.weight"), { s.q_lora, H * s.mla_k }, c);
            need(blk(il, "attn_kv_a_mqa.weight"), { N, s.kv_lora }, c);
            need(blk(il, "attn_kv_a_norm.weight"), { s.kv_lora }, c);
            need(blk(il, "attn_k_b.weight"), { s.mla_k, s.kv_lora, H }, c);
            need(blk(il, "attn_v_b.weight"), { s.kv_lora, s.mla_v, H }, c);
            need(blk(il, "attn_output.weight"), { H * s.mla_v, N }, c);
            need(blk(il, "indexer.attn_q_b.weight"), { s.q_lora, IH }, ci);
            need(blk(il, "indexer.attn_k.weight"), { N, s.idx_head_dim }, ci);
            need(blk(il, "indexer.k_norm.weight"), { s.idx_head_dim }, ci);
            need(blk(il, "indexer.k_norm.bias"), { s.idx_head_dim }, ci);
            need(blk(il, "indexer.proj.weight"), { N, s.idx_n_head }, ci);
            need(blk(il, "indexer_compressor_ape.weight"), { s.idx_head_dim, s.idx_kpool }, ci);
            need(blk(il, "indexer_compressor_gate.weight"), { N, s.idx_head_dim }, ci);
        } else {
            need(blk(il, "attn_q.weight"), { N, KI }, b.kda);
            need(blk(il, "attn_k.weight"), { N, KI }, b.kda);
            need(blk(il, "attn_v.weight"), { N, KI }, b.kda);
            need(blk(il, "attn_output.weight"), { KI, N }, b.kda);
            for (const char * c : { "q", "k", "v" }) {
                need(blk(il, std::string("ssm_conv1d_") + c + ".weight"), { s.conv_k, 1, KI }, b.kda);
            }
            need(blk(il, "ssm_a"), { H }, b.kda);
            need(blk(il, "ssm_dt.bias"), { KI }, b.kda);
            need(blk(il, "ssm_beta.weight"), { N, H }, b.kda);
            need(blk(il, "ssm_f_a.weight"), { N, s.kda_head_dim }, b.kda);
            need(blk(il, "ssm_f_b.weight"), { s.kda_head_dim, KI }, b.kda);
            need(blk(il, "ssm_g_a.weight"), { N, s.kda_head_dim }, b.kda);
            need(blk(il, "ssm_g_b.weight"), { s.kda_head_dim, KI }, b.kda);
            need(blk(il, "ssm_norm.weight"), { s.kda_head_dim }, b.kda);
        }
        if (s.is_moe(il)) {
            uint64_t & c_exp = mtp ? b.mtp : b.experts;
            uint64_t & c_sh  = mtp ? b.mtp : b.shared;
            uint64_t & c_rt  = mtp ? b.mtp : b.router;
            need(blk(il, "ffn_gate_inp.weight"), { N, E }, c_rt);
            need(blk(il, "exp_probs_b.bias"), { E }, c_rt);
            const uint64_t g = need(blk(il, "ffn_gate_exps.weight"), { N, F, E }, c_exp);
            const uint64_t u = need(blk(il, "ffn_up_exps.weight"), { N, F, E }, c_exp);
            const uint64_t d = need(blk(il, "ffn_down_exps.weight"), { F, N, E }, c_exp);
            need(blk(il, "ffn_gate_shexp.weight"), { N, Fs }, c_sh);
            need(blk(il, "ffn_up_shexp.weight"), { N, Fs }, c_sh);
            need(blk(il, "ffn_down_shexp.weight"), { Fs, N }, c_sh);
            if (!mtp) {
                ug.push_back(g / (uint64_t) E);
                uu.push_back(u / (uint64_t) E);
                ud.push_back(d / (uint64_t) E);
            }
        } else {
            need(blk(il, "ffn_gate.weight"), { N, s.n_ff }, b.dense_ffn);
            need(blk(il, "ffn_up.weight"), { N, s.n_ff }, b.dense_ffn);
            need(blk(il, "ffn_down.weight"), { s.n_ff, N }, b.dense_ffn);
        }
        if (mtp) {
            need(blk(il, "nextn.eh_proj.weight"), { 2 * N, N }, b.mtp);
            need(blk(il, "nextn.enorm.weight"), { N }, b.mtp);
            need(blk(il, "nextn.hnorm.weight"), { N }, b.mtp);
            need(blk(il, "nextn.shared_head_norm.weight"), { N }, b.mtp);
        }
        (void) scratch;
    }
    auto med = [](std::vector<uint64_t> v) -> uint64_t {
        if (v.empty()) return 0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    b.exp_unit[0] = med(ug);
    b.exp_unit[1] = med(uu);
    b.exp_unit[2] = med(ud);
    for (const auto & kvp : by_name) {
        if (!used.count(kvp.first)) {
            b.other += kvp.second.nbytes;
            if (unbound) unbound->push_back(kvp.first);
        }
    }
    return b;
}

} // namespace ai
