// SPDX-License-Identifier: Apache-2.0
// ai-glm5-inspect: a glm5-next (GLM-5.3-Flash) GGUF's spec, shape check and byte budget, from metadata only.
//
//   ai-glm5-inspect <model.gguf | model-00001-of-0000N.gguf> [vram_gib=32] [ram_gib=121]
//
// Prints the spec Glm5NextSpec::load() reads, checks every tensor the forward binds against it, then the bytes by
// tensor class, one expert's size, the bytes a decoded token touches, and how the routed experts split across the
// device and host memory given. No GPU.
#include "core/weights.h"
#include "models/glm5next.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ai;

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s <model.gguf | model-00001-of-0000N.gguf> [vram_gib=32] [ram_gib=121]\n", argv[0]);
        return 2;
    }
    const double vram = argc > 2 ? atof(argv[2]) : 32.0, ram = argc > 3 ? atof(argv[3]) : 121.0;
    WeightSet w;
    if (!w.open(argv[1])) return 1;
    Glm5NextSpec s;
    Glm5NextBudget b;
    std::vector<std::string> unbound;
    try {
        s = Glm5NextSpec::load(w);
        b = glm5next_check(w, s, &unbound);
    } catch (const std::exception & e) {
        fprintf(stderr, "glm5-next: %s\n", e.what());
        return 1;
    }
    int n_mla = 0, n_moe = 0;
    for (int il = 0; il < s.n_layer; ++il) {
        n_mla += s.is_mla[(size_t) il];
        n_moe += s.is_moe(il);
    }
    const double G = 1073741824.0, M = 1048576.0;
    printf("spec: arch %s, %d trunk layers (%d MLA sparse attention, %d KDA; %d MoE, %d dense FFN), nextn %d\n",
           s.arch.c_str(), s.n_layer, n_mla, s.n_layer - n_mla, n_moe, s.n_layer - n_moe, s.nextn);
    printf("      n_embd %lld, vocab %lld, heads %d, n_ff %lld, ctx_train %d, rms_eps %g\n", (long long) s.n_embd,
           (long long) s.n_vocab, s.n_head, (long long) s.n_ff, s.n_ctx_train, s.rms_eps);
    printf("      experts %d top-%d + %d shared, ff_exp %lld, scale %g, norm %d\n", s.n_expert, s.n_expert_used, s.n_shexp,
           (long long) s.n_ff_exp, s.exp_scale, (int) s.exp_norm);
    printf("      MLA q rank %d, kv rank %d, key %d, value %d | KDA head_dim %d (inner %lld), conv %d, gate lb %g\n",
           s.q_lora, s.kv_lora, s.mla_k, s.mla_v, s.kda_head_dim, (long long) s.kda_inner(), s.conv_k, s.kda_gate_lb);
    printf("      indexer %d heads x %d, top-%d, kpool %d, shared with MTP %d | mHC %d streams, sinkhorn %d\n",
           s.idx_n_head, s.idx_head_dim, s.idx_top_k, s.idx_kpool, (int) s.idx_share_mtp, s.hc, s.hc_sinkhorn);
    printf("shapes: every bound tensor matches the spec; %zu tensors in the file are not bound\n", unbound.size());
    for (size_t i = 0; i < unbound.size() && i < 12; ++i) printf("        unbound: %s\n", unbound[i].c_str());
    printf("bytes (GiB): experts %.2f | shared %.2f | router %.2f | dense FFN %.2f | MLA %.2f | KDA %.2f | indexer %.2f |"
           " mHC %.2f | norms %.3f | embd %.2f | head %.2f | MTP block %.2f | unbound %.2f | total %.2f\n",
           b.experts / G, b.shared / G, b.router / G, b.dense_ffn / G, b.mla / G, b.kda / G, b.indexer / G, b.hc / G,
           b.norms / G, b.embd / G, b.head / G, b.mtp / G, b.other / G, b.total() / G);
    const uint64_t unit = b.exp_unit[0] + b.exp_unit[1] + b.exp_unit[2];
    const double tok = (double) unit * s.n_expert_used * n_moe;
    printf("expert: gate %.2f + up %.2f + down %.2f = %.2f MiB (median layer); %d experts in all\n", b.exp_unit[0] / M,
           b.exp_unit[1] / M, b.exp_unit[2] / M, unit / M, s.n_expert * n_moe);
    printf("decode: one token routes %d experts = %.2f GiB of expert bytes (+ %.2f GiB of always-read dense weights,"
           " embd/head rows aside)\n", s.n_expert_used * n_moe, tok / G,
           (b.resident_min() - b.embd - b.head - b.mtp) / G);
    // placement: everything but the routed experts is resident; 2.5 GiB of device memory kept for KV, state and workspace
    const double res = b.resident_min() / G, dev_exp = vram - res - 2.5, all = b.experts / G;
    const double dev_frac = dev_exp > 0 ? (dev_exp < all ? dev_exp / all : 1.0) : 0.0;
    const double host = all * (1.0 - dev_frac);
    printf("placement at %.0f GiB device / %.0f GiB host: resident non-expert %.2f GiB; experts on the device %.2f GiB"
           " (%.1f%% of bytes, %d of %d); on the host %.2f GiB (%s)\n", vram, ram, res, dev_exp > 0 ? all * dev_frac : 0.0,
           100.0 * dev_frac, (int) (dev_frac * s.n_expert * n_moe), s.n_expert * n_moe, host,
           host < ram - 8.0 ? "fits in RAM" : "does NOT fit in RAM");
    return 0;
}
