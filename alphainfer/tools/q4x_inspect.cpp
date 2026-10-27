// ai-q4x-inspect: a qwen4exp (Qwen3.8-Flash-Next) GGUF's spec and byte budget, from metadata only.
//
//   ai-q4x-inspect <model.gguf | model-00001-of-0000N.gguf>   (a split GGUF's first shard opens every shard)
//
// Prints the spec Qwen4ExpSpec::load() reads, then bytes by tensor class (experts, dense per layer kind, HC, PLE table,
// embedding/head), the type histogram per class, the per-expert unit size [gate|up|down] per layer, and the per-token
// decode byte estimate (dense + the top-k share of the experts; the PLE table and token_embd are gathers). No GPU.
#include "core/weights.h"
#include "models/qwen4exp.h"

#include "ggml.h"

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace ai;

static std::string cls_of(const std::string & n) {
    if (n == "token_embd.weight") return "token_embd";
    if (n == "output.weight") return "output";
    if (n.rfind("per_layer_token_embd", 0) == 0) return "ple_table";
    if (n.rfind("hc_head", 0) == 0 || n.rfind("output_hc", 0) == 0) return "hc_head";
    if (n.rfind("blk.", 0) != 0) return "other_global";
    const size_t d = n.find('.', 4);
    const std::string t = d == std::string::npos ? n : n.substr(d + 1);
    if (t.find("_exps") != std::string::npos) return "experts";
    if (t.find("shexp") != std::string::npos) return "shared_expert";
    if (t.rfind("ffn_gate_inp", 0) == 0) return "router";
    if (t.rfind("hc_", 0) == 0) return "hc";
    if (t.rfind("ple_", 0) == 0) return "ple_block";
    if (t.rfind("indexer", 0) == 0) return "indexer";
    if (t.rfind("ssm_", 0) == 0 || t.rfind("attn_qkv", 0) == 0 || t.rfind("attn_gate", 0) == 0) return "gdn";
    if (t.rfind("attn_", 0) == 0) return "attention";
    return "other_layer";
}

int main(int argc, char ** argv) {
    const bool names = argc == 3 && std::string(argv[2]) == "--names";
    if (argc != 2 && !names) {
        fprintf(stderr, "usage: %s <model.gguf | model-00001-of-0000N.gguf> [--names]   (opens every shard)\n", argv[0]);
        return 2;
    }
    std::vector<std::unique_ptr<WeightSet>> shards;
    shards.push_back(std::make_unique<WeightSet>());
    if (!shards.back()->open(argv[1])) return 1;
    Qwen4ExpSpec s;
    try {
        s = Qwen4ExpSpec::load(*shards[0]);
    } catch (const std::exception & e) {
        fprintf(stderr, "spec: %s\n", e.what());
        return 1;
    }
    for (const auto & w : shards) {   // n_vocab from whichever shard holds token_embd
        for (const auto & m : w->meta()) if (m.name == "token_embd.weight") s.n_vocab = m.t.ne[1];
    }
    int n_attn = 0, n_qsa = 0;
    for (int il = 0; il < s.n_layer; ++il) {
        if (s.is_attn[(size_t) il]) { n_attn++; if ((size_t) il < s.compress_ratios.size() && s.compress_ratios[(size_t) il] > 0) n_qsa++; }
    }
    printf("spec: arch %s, %d layers (%d attention, %d with QSA; %d GDN), nextn %d\n", s.arch.c_str(), s.n_layer, n_attn,
           n_qsa, s.n_layer - n_attn, s.nextn);
    printf("      n_embd %lld, vocab %lld, heads %d / kv %d x %d (rot %d, sections %d %d %d %d, base %.0f), ctx_train %d\n",
           (long long) s.n_embd, (long long) s.n_vocab, s.n_head, s.n_head_kv, s.head_dim, s.n_rot, s.sections[0],
           s.sections[1], s.sections[2], s.sections[3], s.rope_base, s.n_ctx_train);
    printf("      experts %d top-%d, ff_exp %lld, ff_shexp %lld | ssm conv %d state %d groups %d dt_rank %d inner %d\n",
           s.n_expert, s.n_expert_used, (long long) s.n_ff_exp, (long long) s.n_ff_shexp, s.conv_k, s.ssm_state,
           s.ssm_groups, s.ssm_dt_rank, s.ssm_inner);
    printf("      hc %d (low rank %d, hc_dim %lld) | indexer %d heads x %d, top-%d\n", s.hc, s.hc_lr,
           (long long) s.hc_dim(), s.idx_n_head, s.idx_head_dim, s.idx_top_k);
    if (!s.ple_layers.empty()) {
        printf("      PLE: layers");
        for (int64_t l : s.ple_layers) printf(" %lld", (long long) l);
        printf(" | ngram %d, heads/ngram %d (%d heads x %d), conv %d, eos %lld\n", s.ple_ngram, s.ple_heads_per_ngram,
               s.ple_n_heads(), s.ple_head_dim, s.ple_conv_k, (long long) s.ple_eos);
    }

    if (names) {   // the global tensors, then every tensor of the first GDN, PLE and attention layers (name type shape)
        std::vector<int> show = { 0, 3 };
        for (int64_t l : s.ple_layers) show.push_back((int) l);
        for (const auto & m : shards[0]->meta()) {
            bool pr = m.name.rfind("blk.", 0) != 0;
            for (int il : show) pr = pr || m.name.rfind("blk." + std::to_string(il) + ".", 0) == 0;
            if (pr) printf("  %-40s %-8s [%lld, %lld, %lld]\n", m.name.c_str(), ggml_type_name((ggml_type) m.t.type),
                           (long long) m.t.ne[0], (long long) m.t.ne[1], (long long) m.t.ne[2]);
        }
    }
    std::map<std::string, size_t> bytes;
    std::map<std::string, std::map<std::string, size_t>> types;
    std::map<int, size_t> unit;   // layer -> bytes of one expert's gate+up+down
    size_t total = 0;
    for (const auto & w : shards) {
        for (const auto & m : w->meta()) {
            const std::string c = cls_of(m.name);
            bytes[c] += m.nbytes;
            types[c][ggml_type_name((ggml_type) m.t.type)] += m.nbytes;
            total += m.nbytes;
            if (c == "experts" && m.t.ne[2] > 0) {
                const int il = atoi(m.name.c_str() + 4);
                unit[il] += m.nbytes / (size_t) m.t.ne[2];
            }
        }
    }
    // oracle: AlphaInfer's dtype table (core/tensor.cpp) against ggml's own size for every tensor in the file
    size_t n_bad = 0, n_all = 0;
    for (const auto & w : shards) {
        for (const auto & m : w->meta()) {
            n_all++;
            size_t mine = 0;
            try { mine = m.t.nbytes(); } catch (const std::exception &) { mine = 0; }
            if (mine != m.nbytes && n_bad++ < 5) {
                fprintf(stderr, "dtype table MISMATCH %s: type %d, ai %zu B vs ggml %zu B\n", m.name.c_str(), m.t.type,
                        mine, m.nbytes);
            }
        }
    }
    printf("dtype table vs ggml_nbytes: %zu of %zu tensors mismatch%s\n", n_bad, n_all,
           n_bad ? "  <-- FIX core/tensor.cpp" : "");
    printf("\nbytes by class (total %.2f GB):\n", total / 1e9);
    for (const auto & kv : bytes) {
        printf("  %-14s %9.3f GB  ", kv.first.c_str(), kv.second / 1e9);
        for (const auto & t : types[kv.first]) printf(" %s %.2f", t.first.c_str(), t.second / 1e9);
        printf("\n");
    }
    size_t umin = SIZE_MAX, umax = 0;
    for (const auto & u : unit) { umin = std::min(umin, u.second); umax = std::max(umax, u.second); }
    if (!unit.empty()) printf("expert unit [gate|up|down]: %.3f .. %.3f MB over %zu layers\n", umin / 1e6, umax / 1e6, unit.size());
    const double dense = (double) (total - bytes["experts"] - bytes["ple_table"] - bytes["token_embd"]);
    const double act   = s.n_expert ? (double) bytes["experts"] * s.n_expert_used / s.n_expert : 0.0;
    printf("per-token decode bytes: dense %.3f GB (excl. experts, PLE table, token_embd) + active experts %.3f GB = %.3f GB\n",
           dense / 1e9, act / 1e9, (dense + act) / 1e9);
    return 0;
}
