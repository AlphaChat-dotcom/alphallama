// SPDX-License-Identifier: Apache-2.0
// ai-gsq-gemm-bench: many-token VNNI kernels (src/offload/gsq_gemm.cpp) vs gsq_dot in groups of 4, on the real expert
// tensors of a qwen4exp GGUF. Single thread, page-warm. Prints per (layer, tensor, T): us per expert matrix for both,
// the speed-up and whether every output is bit-identical (it must be).
//
//   ai-gsq-gemm-bench <model-00001-of-0000N.gguf> [experts=16] [layers=0,1,...] [T list=4,8,13,16,32,64]
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "offload/gsq_cpu.h"
#include "offload/gsq_gemm.h"

#include <algorithm>
#include <chrono>
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
static std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::string c = s;
    for (char * p = strtok(c.data(), ","); p; p = strtok(nullptr, ",")) v.push_back(atoi(p));
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model-00001-of-0000N.gguf> [experts=16] [layers=0,1] [T=4,8,13,16,32,64]\n", argv[0]); return 2; }
    const int        nex    = argc > 2 ? atoi(argv[2]) : 16;
    const int        reps   = getenv("REPS") ? atoi(getenv("REPS")) : 15;
    std::vector<int> layers = argc > 3 ? parse_list(argv[3]) : std::vector<int>{ 0, 1 };
    std::vector<int> Ts     = argc > 4 ? parse_list(argv[4]) : std::vector<int>{ 4, 8, 13, 16, 32, 64 };
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
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int Tmax = *std::max_element(Ts.begin(), Ts.end());
    printf("%-6s %-10s %-8s %3s %10s %10s %8s %s\n", "layer", "tensor", "type", "T", "dot4_us", "gemm_us", "speedup", "bitexact");
    int bad = 0;
    for (int il : layers) {
        for (const char * nm : { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" }) {
            ggml_tensor * tw; const char * dw;
            if (!find("blk." + std::to_string(il) + "." + nm + ".weight", &tw, &dw)) continue;
            if (!ai::offload::gsq_gemm_supported(tw->type)) {
                printf("%-6d %-10s %-8s  - (no gemm kernel / no AVX-VNNI)\n", il, nm + 4, ggml_type_name(tw->type));
                continue;
            }
            const int64_t K = tw->ne[0], R = tw->ne[1], E = tw->ne[2];
            const auto *  ct = ggml_get_type_traits_cpu(tw->type);
            const auto *  ty = ggml_get_type_traits_cpu(ct->vec_dot_type);
            const size_t  rb = ggml_row_size(tw->type, K), yb = ggml_row_size(ct->vec_dot_type, K);
            std::vector<std::vector<char>> yq((size_t) Tmax, std::vector<char>(yb + 64));
            std::vector<float> x((size_t) K);
            for (auto & q : yq) { for (auto & v : x) v = nd(rng); ty->from_float(x.data(), q.data(), K); }
            std::vector<std::vector<char>> prep;
            std::vector<const void *>      yo((size_t) Tmax);
            for (int t = 0; t < Tmax; ++t) yo[(size_t) t] = yq[(size_t) t].data();
            if (tw->type == GGML_TYPE_Q2_0) {
                for (int t = 0; t < Tmax; ++t) {
                    prep.emplace_back(ggml_cpu_q2_0_prep_size((int) K) + 64);
                    ggml_cpu_q2_0_prepare(yq[(size_t) t].data(), (int) K, prep.back().data());
                    yo[(size_t) t] = prep.back().data();
                }
            }
            std::vector<float> od((size_t) Tmax * R), og((size_t) Tmax * R);
            std::vector<float *> outs((size_t) Tmax);
            for (int t = 0; t < Tmax; ++t) outs[(size_t) t] = og.data() + (size_t) t * R;
            volatile char sink = 0;
            for (int e = 0; e < nex; ++e) {
                const char * W = dw + (size_t) ((e * 37) % E) * R * rb;
                for (size_t o = 0; o < (size_t) R * rb; o += 4096) sink ^= W[o];
            }
            (void) sink;
            ai::offload::GsqActs acts;
            for (int T : Ts) {
                // per expert call, the minimum over reps (the pool threads of a running server time-slice this core)
                double td = 0.0, tg = 0.0;
                for (int e = 0; e < nex; ++e) {
                    const char * W = dw + (size_t) ((e * 37) % E) * R * rb;
                    double bd = 1e30, bg = 1e30;
                    for (int rep = 0; rep < reps; ++rep) {
                        double t0 = now_us();
                        for (int s0 = 0; s0 < T; s0 += 4) {
                            const int sn = std::min(4, T - s0);
                            ai::offload::gsq_dot(tw->type, (int) K, W, rb, 0, R, yo.data() + s0, sn, od.data() + (size_t) s0 * R, R);
                        }
                        bd = std::min(bd, now_us() - t0);
                        t0 = now_us();
                        ai::offload::gsq_gemm_prep(tw->type, (int) K, yo.data(), T, acts);
                        ai::offload::gsq_gemm(acts, W, rb, 0, R, outs.data());
                        bg = std::min(bg, now_us() - t0);
                    }
                    td += bd / nex; tg += bg / nex;
                }
                const bool exact = memcmp(od.data(), og.data(), (size_t) T * R * sizeof(float)) == 0;
                if (!exact) {
                    ++bad;
                    int shown = 0;
                    for (int64_t i = 0; i < T * R && shown < 3; ++i) {
                        if (od[(size_t) i] != og[(size_t) i]) {
                            printf("    t=%lld r=%lld dot4=%.9g gemm=%.9g\n", (long long) (i / R), (long long) (i % R), od[(size_t) i], og[(size_t) i]);
                            ++shown;
                        }
                    }
                }
                printf("%-6d %-10s %-8s %3d %10.1f %10.1f %7.2fx %s%s\n", il, nm + 4, ggml_type_name(tw->type), T, td, tg, td / tg,
                       exact ? "yes" : "NO", acts.fallback ? " (fallback)" : "");
            }
        }
    }
    printf(bad ? "FAIL: %d mismatching cases\n" : "ALL BIT-IDENTICAL\n", bad);
    return bad ? 1 : 0;
}
