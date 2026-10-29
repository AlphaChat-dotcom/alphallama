// SPDX-License-Identifier: Apache-2.0
// ai-q4x-expbench: is the grouped expert kernel (kernels/q4x_experts.cu) byte-bound or issue-bound at verify width T,
// and how does the wide (host-plan) kernel scale to 16..512 columns?
//
// Synthetic bank of S resident slots (random bytes: timing does not depend on the values), Flash-Next shapes by
// default (K = N = 2560, F = 640, top-10). Routing per token: 10 distinct slots drawn from a Zipf(1) popularity, each
// copied from the previous token's set with probability --share (verify windows over copied text share experts).
// Reports per T: distinct experts, us per call, and the implied DRAM rate over the distinct expert bytes -- if that
// sits near the card's bandwidth the kernel is byte-bound and a tensor-core rewrite cannot pay at that T.
//
//   ai-q4x-expbench [--gu TYPE] [--down TYPE] [--slots 230] [--share 0.5] [--reps 200] [--wide] [--T 1,2,4,8]
//                   [--path grouped|wide|tc]
#include <cstddef>
#include <cstdint>

#include "kernels/kernels.h"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

using namespace ai;

static int type_of(const char * s) {
    const std::string n = s;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn && n == tn) return t;
    }
    fprintf(stderr, "unknown type %s\n", s);
    exit(2);
}

int main(int argc, char ** argv) {
    int         tg = GGML_TYPE_IQ3_XXS, td = GGML_TYPE_IQ4_NL, S = 230, reps = 200, k = 10;
    double      share = 0.5;
    std::string path  = "grouped";
    bool        check = false, tc = false, uniform = false;
    int         fam = 0;   // kernel family under test: 0 dp4a, 1 tensor cores, 2 dp4a smem-staged; --check compares against 0 (or 1 vs 0)
    int64_t     K = 2560, F = 640, N = 2560;
    std::vector<int> Ts = { 1, 2, 3, 4, 6, 8 };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return argv[++i]; };
        if (a == "--gu") tg = type_of(nx());
        else if (a == "--down") td = type_of(nx());
        else if (a == "--slots") S = atoi(nx());
        else if (a == "--share") share = atof(nx());
        else if (a == "--reps") reps = atoi(nx());
        else if (a == "--path") path = nx();
        else if (a == "--tc") { tc = true; fam = 1; }
        else if (a == "--smem") fam = 2;
        else if (a == "--check") check = true;
        else if (a == "--uniform") uniform = true;   // equal popularity over the slots (prefill-like routing)
        else if (a == "--T") { Ts.clear(); for (char * p = strtok(nx(), ","); p; p = strtok(nullptr, ",")) Ts.push_back(atoi(p)); }
        else { fprintf(stderr, "bad arg %s\n", a.c_str()); return 2; }
    }
    const size_t rbg = ggml_row_size((ggml_type) tg, K), rbd = ggml_row_size((ggml_type) td, F);
    const size_t ebg = rbg * F, ebd = rbd * N, per_exp = 2 * ebg + ebd;
    int dev = 0;
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));
    int mclk = 0, mbus = 0;   // kHz, bits
    CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, dev));
    CK(cudaDeviceGetAttribute(&mbus, cudaDevAttrGlobalMemoryBusWidth, dev));
    const double peak = 2.0 * mclk * 1e3 * (mbus / 8) / 1e9;
    printf("%s | gu %s down %s | K %lld F %lld N %lld k %d | %d slots | %.3f MB per expert (gate+up %.3f, down %.3f) | path %s share %.2f\n",
           prop.name, ggml_type_name((ggml_type) tg), ggml_type_name((ggml_type) td), (long long) K, (long long) F,
           (long long) N, k, S, per_exp / 1e6, 2 * ebg / 1e6, ebd / 1e6, path.c_str(), share);
    void * up, * gate, * down;
    CK(cudaMalloc(&up, ebg * S));
    CK(cudaMalloc(&gate, ebg * S));
    CK(cudaMalloc(&down, ebd * S));
    {   // random bytes, every 16-bit word's exponent tamed so fp16 scales stay finite (timing only)
        std::mt19937          rng(1);
        std::vector<uint16_t> h(std::max(ebg, ebd) * (size_t) S / 2 + 1);
        for (auto & v : h) v = (uint16_t) (rng() & 0x3BFF);
        // every block starts with its fp16 d (all types here): set it to ~2^-8 so outputs keep a sane magnitude
        const size_t bg = ggml_type_size((ggml_type) tg), bd = ggml_type_size((ggml_type) td);
        std::vector<uint8_t> hu(ebg * S), hd(ebd * S);
        memcpy(hu.data(), h.data(), hu.size());
        memcpy(hd.data(), h.data(), hd.size());
        for (size_t o = 0; o + bg <= hu.size(); o += bg) { const uint16_t v = (uint16_t) (0x1C00 | (rng() & 0x3FF)); memcpy(&hu[o], &v, 2); }
        for (size_t o = 0; o + bd <= hd.size(); o += bd) { const uint16_t v = (uint16_t) (0x1C00 | (rng() & 0x3FF)); memcpy(&hd[o], &v, 2); }
        CK(cudaMemcpy(up, hu.data(), ebg * S, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(gate, hu.data(), ebg * S, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(down, hd.data(), ebd * S, cudaMemcpyHostToDevice));
    }
    const int Tmax = *std::max_element(Ts.begin(), Ts.end());
    float *   x, * out;
    CK(cudaMalloc(&x, sizeof(float) * K * Tmax));
    CK(cudaMalloc(&out, sizeof(float) * N * k * Tmax));
    {
        std::mt19937                    rng(2);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float>              h((size_t) (K * Tmax));
        for (auto & v : h) v = nd(rng);
        CK(cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    }
    void * scr;
    CK(cudaMalloc(&scr, kern::q4x_experts_scratch_bytes(K, F, k, Tmax) + (1 << 20)));
    int32_t * ids_d, * plan_d;
    constexpr int NDRAW = 8;   // routing draws per T, uploaded once, timed round-robin
    const int rec = kern::q4x_experts_wide_rec();
    const size_t plan_stride = (size_t) rec * std::min(S, k * Tmax);
    CK(cudaMalloc(&ids_d, sizeof(int32_t) * k * Tmax * NDRAW));
    CK(cudaMalloc(&plan_d, sizeof(int32_t) * plan_stride * NDRAW));
    // Zipf(1) popularity over slots
    std::vector<double> w((size_t) S);
    for (int s = 0; s < S; ++s) w[(size_t) s] = uniform ? 1.0 : 1.0 / (s + 1);
    std::discrete_distribution<int> pop(w.begin(), w.end());
    std::mt19937                    rng(3);
    std::uniform_real_distribution<double> u01(0, 1);
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    kern::q4x_experts_set_tc(fam);
    float * ref;
    CK(cudaMalloc(&ref, sizeof(float) * N * k * Tmax));
    for (int T : Ts) {
        const bool use_gu   = path == "gu";   // prefill's gate/up only (q4x_experts_wide_gu; AI_Q4X_PF_GU picks the kernel)
        const bool use_dn   = path == "dn";   // prefill's down only (q4x_experts_wide_dn over one slice of T rows; AI_Q4X_PF_DN)
        const bool use_wide = path == "wide" || T > 8;
        if (!use_wide && !kern::q4x_experts_ok(tg, td, K, F, N, k, T)) { printf("T %d: grouped kernel declines\n", T); continue; }
        std::vector<std::vector<int32_t>> ids(NDRAW), plans(NDRAW);
        std::vector<int> nown(NDRAW);
        double dist_sum = 0;
        for (int d = 0; d < NDRAW; ++d) {
            ids[d].assign((size_t) (k * T), 0);
            std::vector<int> prev;
            for (int t = 0; t < T; ++t) {
                std::set<int> cur;
                for (int j = 0; j < k && t > 0; ++j) if (u01(rng) < share) cur.insert(prev[(size_t) j]);
                while ((int) cur.size() < k) cur.insert(pop(rng));
                std::vector<int> v(cur.begin(), cur.end());
                std::shuffle(v.begin(), v.end(), rng);
                for (int j = 0; j < k; ++j) ids[d][(size_t) (t * k + j)] = v[(size_t) j];
                prev = v;
            }
            std::vector<int> own((size_t) S, -1);
            int n = 0;
            for (int c = 0; c < k * T; ++c) {
                const int s = ids[d][(size_t) c];
                if (own[(size_t) s] < 0) { own[(size_t) s] = n++; plans[d].resize((size_t) n * rec, 0); plans[d][(size_t) own[(size_t) s] * rec] = s; }
                int32_t * r = plans[d].data() + (size_t) own[(size_t) s] * rec;
                r[2 + r[1]++] = c;
            }
            nown[d] = n;
            dist_sum += n;
            CK(cudaMemcpy(ids_d + (size_t) d * k * T, ids[d].data(), ids[d].size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(plan_d + (size_t) d * plan_stride, plans[d].data(), plans[d].size() * 4, cudaMemcpyHostToDevice));
        }
        auto run = [&](int d) {
            if (use_dn) {
                if (!kern::q4x_experts_wide_dn(st, tg, up, gate, (int64_t) rbg, (int64_t) ebg, td, down, (int64_t) rbd, (int64_t) ebd,
                                               plan_d + (size_t) d * plan_stride, nown[d], K, F, N, k, T, scr, out, 0)) {
                    fprintf(stderr, "wide down declined\n"); exit(1);
                }
                return;
            }
            if (use_gu) {
                if (!kern::q4x_experts_wide_gu(st, tg, up, gate, (int64_t) rbg, (int64_t) ebg, td, down, (int64_t) rbd,
                                               (int64_t) ebd, x, K, true, plan_d + (size_t) d * plan_stride, nown[d], K, F, N,
                                               k, T, scr)) {
                    fprintf(stderr, "wide gate/up declined\n"); exit(1);
                }
            } else if (use_wide) {
                if (!kern::q4x_experts_wide(st, tg, up, gate, (int64_t) rbg, (int64_t) ebg, td, down, (int64_t) rbd,
                                            (int64_t) ebd, x, K, plan_d + (size_t) d * plan_stride, nown[d], K, F, N, k, T, scr, out)) {
                    fprintf(stderr, "wide declined\n"); exit(1);
                }
            } else {
                if (!kern::q4x_experts(st, tg, up, gate, (int64_t) rbg, (int64_t) ebg, td, down, (int64_t) rbd,
                                       (int64_t) ebd, x, K, ids_d + (size_t) d * k * T, k, S, K, F, N, k, T, scr, out)) {
                    fprintf(stderr, "grouped declined\n"); exit(1);
                }
            }
        };
        if (check) {   // the other kernel family on draw 0 -> ref, then this one; compare every (routed) column
            kern::q4x_experts_set_tc(0);
            CK(cudaMemset(out, 0, sizeof(float) * N * k * T));
            run(0);
            CK(cudaMemcpyAsync(ref, out, sizeof(float) * N * k * T, cudaMemcpyDeviceToDevice, st));
            kern::q4x_experts_set_tc(fam);
            CK(cudaMemsetAsync(out, 0, sizeof(float) * N * k * T, st));
            run(0);
            CK(cudaStreamSynchronize(st));
            std::vector<float> a((size_t) (N * k * T)), b(a.size());
            CK(cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), ref, b.size() * 4, cudaMemcpyDeviceToHost));
            double e2 = 0, r2 = 0, mx = 0; long long nan = 0, same = 0, nan_a = 0, nan_b = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                nan_a += !std::isfinite(a[i]); nan_b += !std::isfinite(b[i]);
                if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nan; continue; }
                const double d = (double) a[i] - b[i];
                e2 += d * d; r2 += (double) b[i] * b[i]; mx = std::max(mx, std::fabs(d));
                same += a[i] == b[i];
            }
            double rms = std::sqrt(r2 / a.size());
            printf("T %3d | CHECK %s vs %s | rel-L2 %.3g | max |d| %.3g (%.3g of rms %.3g) | non-finite %lld (this %lld, ref %lld) | identical %.1f%%\n",
                   T, fam == 1 ? "tc" : fam == 2 ? "smem" : "dp4a", "dp4a", r2 > 0 ? std::sqrt(e2 / r2) : 0.0, mx, rms > 0 ? mx / rms : 0.0, rms,
                   nan, nan_a, nan_b, 100.0 * same / a.size());
        }
        if (use_dn) {   // the down kernel reads the quantised SwiGLU activations: gate/up + quantise once to fill the scratch
            kern::q4x_experts_wide_gu(st, tg, up, gate, (int64_t) rbg, (int64_t) ebg, td, down, (int64_t) rbd, (int64_t) ebd, x, K,
                                      true, plan_d, nown[0], K, F, N, k, T, scr);
            kern::q4x_experts_wide_qact(st, K, F, k, T, scr);
        }
        for (int i = 0; i < 20; ++i) run(i % NDRAW);
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(e0, st));
        for (int i = 0; i < reps; ++i) run(i % NDRAW);
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaGetLastError());
        const double us = 1e3 * ms / reps, dist = dist_sum / NDRAW;
        const double gbs = dist * per_exp / (us * 1e-6) / 1e9;
        if (use_dn) {   // int8 tensor-core ops of down over every routed column
            const double tops = 2.0 * (double) F * (double) N * (double) (k * T) / (us * 1e-6) / 1e12;
            printf("T %5d | down | %6.1f experts | %9.1f us/call | %6.1f TOPS (int8 MACs x 2)\n", T, dist, us, tops);
            continue;
        }
        if (use_gu) {   // int8 tensor-core ops of gate + up over every routed column
            const double tops = 2.0 * 2.0 * (double) K * (double) F * (double) (k * T) / (us * 1e-6) / 1e12;
            printf("T %5d | gate/up | %6.1f experts | %9.1f us/call | %6.1f TOPS (int8 MACs x 2)\n", T, dist, us, tops);
            continue;
        }
        printf("T %3d | %-7s | distinct %6.1f of %4d cols | %8.1f us/call | %6.1f us/expert | %6.0f GB/s over distinct bytes (%.0f%% of %.0f peak)\n",
               T, fam == 1 ? (use_wide ? "tc-wide" : "tc") : use_wide ? "wide" : fam == 2 ? "smem" : "grouped", dist, k * T, us, us / dist, gbs,
               100.0 * gbs / peak, peak);
    }
    return 0;
}
