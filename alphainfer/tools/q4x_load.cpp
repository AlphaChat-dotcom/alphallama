// ai-q4x-load: load a qwen4exp (Flash-Next) model through Program<Qwen4Exp> -- device dense weights (hc projections
// converted BF16 -> Q8_0), host-mapped experts and PLE table -- bind and shape-check every layer, report memory.
//
//   ai-q4x-load <model.gguf | model-00001-of-0000N.gguf> [device]
#include "core/device.h"
#include "models/qwen4exp.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace ai;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf | model-00001-of-0000N.gguf> [device]\n", argv[0]);
        return 2;
    }
    Device dev;
    if (!dev.init(argc > 2 ? atoi(argv[2]) : 0)) return 1;
    size_t f0 = 0, tot = 0;
    dev.memory_info(&f0, &tot);
    const auto t0 = std::chrono::steady_clock::now();
    Qwen4Exp prog;
    try {
        if (!prog.load(dev, argv[1], true)) return 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "load: %s\n", e.what());
        return 1;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t f1 = 0;
    dev.memory_info(&f1, &tot);
    int n_attn = 0, n_ple = 0, n_qsa = 0;
    for (const auto & L : prog.layers()) { n_attn += L.attn; n_ple += L.ple; n_qsa += L.attn && L.qsa_p.ratio > 0; }
    printf("loaded in %.1f s: %zu layers (%d attention / %d QSA / %d PLE), device used %.2f GiB of %.2f (free %.2f -> "
           "%.2f GiB)\n", s, prog.layers().size(), n_attn, n_qsa, n_ple, (f0 - f1) / 1073741824.0, tot / 1073741824.0,
           f0 / 1073741824.0, f1 / 1073741824.0);
    const auto & L0 = prog.layers()[0];
    printf("layer 0 hc_attn.down: type %d %s (Q8_0 = 8 after the load-time conversion)\n", L0.hc_attn.down.type,
           L0.hc_attn.down.shape_str().c_str());
    printf("layer 0 gate_exps: %s, host %s\n", L0.moe.gate_exps.shape_str().c_str(),
           L0.moe.gate_exps.buf < 0 ? "yes (mapped)" : "NO");
    return 0;
}
