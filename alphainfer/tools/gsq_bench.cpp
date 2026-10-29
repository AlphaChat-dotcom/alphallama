// SPDX-License-Identifier: Apache-2.0
// ai-gsq-bench: decode-once multi-token GSQ CPU kernels (src/offload/gsq_cpu.cpp) vs ggml-cpu's own vec_dot, on the
// real expert tensors of a qwen4exp GGUF. Single thread, page-warm, T tokens sharing one expert.
// Prints per (layer, tensor, T): us per expert matrix for ggml (T separate row passes) and ours, GB/s of weights,
// speed-up, max relative error and whether T = 1 is bit-identical.
//
//   ai-gsq-bench <model-00001-of-0000N.gguf> [experts=32] [layers=0,1,...]
// Build: g++ -O3 -march=native -std=c++17 tools/gsq_bench.cpp src/offload/gsq_cpu.cpp -I src -I <repo>/ggml/include
//        -I <repo>/ggml/src -L <repo build>/bin -lggml-cpu -lggml-base -lggml -pthread
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "offload/gsq_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct Mapped { char * base = nullptr; size_t data0 = 0; gguf_context * g = nullptr; ggml_context * m = nullptr; };
static Mapped map_shard(const std::string & path) {
    Mapped s;
    gguf_init_params p = { true, &s.m };
    s.g = gguf_init_from_file(path.c_str(), p);
    if (!s.g) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    s.data0 = gguf_get_data_offset(s.g);
    const int fd = open(path.c_str(), O_RDONLY);
    struct stat st; fstat(fd, &st);
    s.base = (char *) mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    return s;
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model-00001-of-0000N.gguf> [experts=32] [layers=0,1]\n", argv[0]); return 2; }
    const int nex = argc > 2 ? atoi(argv[2]) : 32;
    std::vector<int> layers = { 0, 1 };
    if (argc > 3) { layers.clear(); for (char * p = strtok(argv[3], ","); p; p = strtok(nullptr, ",")) layers.push_back(atoi(p)); }
    ggml_cpu_init();
    std::string p1 = argv[1];
    std::vector<Mapped> shards = { map_shard(p1) };
    const size_t tl = strlen("-00001-of-00002.gguf");
    if (p1.size() > tl && p1.find("-00001-of-") != std::string::npos) {
        const std::string base = p1.substr(0, p1.size() - tl), n2 = p1.substr(p1.size() - 10, 5);
        for (int k = 2; k <= atoi(n2.c_str()); ++k) {
            char sfx[64]; snprintf(sfx, sizeof sfx, "-%05d-of-%s.gguf", k, n2.c_str());
            shards.push_back(map_shard(base + sfx));
        }
    }
    auto find = [&](const std::string & name, ggml_tensor ** t, const char ** data) {
        for (auto & s : shards) {
            const int64_t id = gguf_find_tensor(s.g, name.c_str());
            if (id < 0) continue;
            *t = ggml_get_tensor(s.m, name.c_str());
            *data = s.base + s.data0 + gguf_get_tensor_offset(s.g, id);
            return true;
        }
        return false;
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    printf("%-6s %-10s %-8s %2s %10s %10s %8s %8s %8s %11s %s\n", "layer", "tensor", "type", "T", "ggml_us", "ours_us",
           "GB/s_g", "GB/s_o", "speedup", "nmse", "T1_bitexact");
    for (int il : layers) {
        for (const char * nm : { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" }) {
            ggml_tensor * tw; const char * dw;
            if (!find("blk." + std::to_string(il) + "." + nm + ".weight", &tw, &dw)) continue;
            if (!ai::offload::gsq_dot_supported(tw->type)) {
                printf("%-6d %-10s %-8s  - (no decode-once kernel yet)\n", il, nm + 4, ggml_type_name(tw->type));
                continue;
            }
            const int64_t K = tw->ne[0], R = tw->ne[1], E = tw->ne[2];
            const auto *  ct = ggml_get_type_traits_cpu(tw->type);
            const auto *  ty = ggml_get_type_traits_cpu(ct->vec_dot_type);
            const size_t  rb = ggml_row_size(tw->type, K), yb = ggml_row_size(ct->vec_dot_type, K);
            std::vector<std::vector<char>> yq(4, std::vector<char>(yb));
            std::vector<float> x((size_t) K);
            for (auto & q : yq) { for (auto & v : x) v = nd(rng); ty->from_float(x.data(), q.data(), K); }
            const void * yp[4] = { yq[0].data(), yq[1].data(), yq[2].data(), yq[3].data() };
            // Q2_0: our kernel reads activations prepared once per token (ggml_cpu_q2_0_prepare); ggml's reads plain q8_0
            std::vector<std::vector<char>> prep;
            const void * yo[4] = { yp[0], yp[1], yp[2], yp[3] };
            if (tw->type == GGML_TYPE_Q2_0) {
                for (int t = 0; t < 4; ++t) {
                    prep.emplace_back(ggml_cpu_q2_0_prep_size((int) K));
                    ggml_cpu_q2_0_prepare(yq[(size_t) t].data(), (int) K, prep.back().data());
                }
                for (int t = 0; t < 4; ++t) yo[t] = prep[(size_t) t].data();
            }
            std::vector<float> og((size_t) (4 * R)), oo((size_t) (4 * R));
            volatile char sink = 0;
            for (int e = 0; e < nex; ++e) {
                const char * W = dw + (size_t) ((e * 37) % E) * R * rb;
                for (size_t o = 0; o < (size_t) R * rb; o += 4096) sink ^= W[o];
            }
            (void) sink;
            for (int T = 1; T <= 4; ++T) {
                double tg = 1e30, to = 1e30, err = 0.0;
                bool   exact = true;
                for (int rep = 0; rep < 3; ++rep) {
                    double t0 = now_us();
                    for (int e = 0; e < nex; ++e) {
                        const char * W = dw + (size_t) ((e * 37) % E) * R * rb;
                        for (int t = 0; t < T; ++t)
                            for (int64_t r = 0; r < R; ++r) ct->vec_dot((int) K, &og[(size_t) (t * R + r)], 0, W + (size_t) r * rb, 0, yp[t], 0, 1);
                    }
                    tg = std::min(tg, (now_us() - t0) / nex);
                    t0 = now_us();
                    for (int e = 0; e < nex; ++e) {
                        const char * W = dw + (size_t) ((e * 37) % E) * R * rb;
                        ai::offload::gsq_dot(tw->type, (int) K, W, rb, 0, R, yo, T, oo.data(), R);
                    }
                    to = std::min(to, (now_us() - t0) / nex);
                }
                double num = 0.0, den = 0.0;
                for (int64_t i = 0; i < T * R; ++i) {
                    const double d = (double) og[(size_t) i] - oo[(size_t) i];
                    num += d * d; den += (double) og[(size_t) i] * og[(size_t) i];
                    if (T == 1 && og[(size_t) i] != oo[(size_t) i]) exact = false;
                }
                err = num / std::max(den, 1e-30);
                const double mb = (double) R * rb / 1e6;
                printf("%-6d %-10s %-8s %2d %10.1f %10.1f %8.2f %8.2f %7.2fx %11.2e %s\n", il, nm + 4, ggml_type_name(tw->type), T,
                       tg, to, mb / tg * 1e3, mb / to * 1e3, tg / to, err, T == 1 ? (exact ? "yes" : "NO") : "-");
            }
        }
    }
    return 0;
}
