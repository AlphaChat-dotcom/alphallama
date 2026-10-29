// SPDX-License-Identifier: Apache-2.0
// ai-q4x-hcbench: the native hyper-connection read (kernels/q4x_hc.cu) at qwen4exp shapes on synthetic
// weights: N = 2560 streams of H = 4, low rank LR = 320, Q8_0 down [HD -> LR] / up [LR -> HD], BF16 inject rows.
// Times q4x_hc_read per call for each T and up-mix variant, and compares the variants' outputs (rel-L2, max |d|).
//   ai-q4x-hcbench [--T 1,2,3,4,8] [--reps 200] [--up 9,10]   (--up 99 = the tensor-core read)
#include <cstddef>
#include <cstdint>

#include "kernels/q4x_hc.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::vector<char> q8_rows(int rows, int k, std::mt19937 & rng, float scale) {   // Q8_0: {half d; int8 q[32]}
    const int nb = k / 32;
    std::vector<char> out((size_t) rows * nb * 34);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            char * blk = out.data() + ((size_t) r * nb + b) * 34;
            const __half d = __float2half(scale * (0.5f + 0.5f * std::fabs(nd(rng))) / 127.f);
            memcpy(blk, &d, 2);
            for (int i = 0; i < 32; ++i) blk[2 + i] = (char) std::max(-127, std::min(127, (int) std::lround(nd(rng) * 40.f)));
        }
    return out;
}

int main(int argc, char ** argv) {
    std::vector<int> Ts = { 1, 2, 3, 4, 8 }, ups = { 9, 10 };
    int reps = 200;
    const int N = 2560, H = 4, LR = 320, HD = N * H;
    auto list = [](char * s) { std::vector<int> v; for (char * p = strtok(s, ","); p; p = strtok(nullptr, ",")) v.push_back(atoi(p)); return v; };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--T") Ts = list(argv[++i]);
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--up") ups = list(argv[++i]);
    }
    std::mt19937 rng(11);
    const auto hwd = q8_rows(LR, HD, rng, 0.05f);   // down rows (K = HD)
    const auto hwu = q8_rows(HD, LR, rng, 0.5f);    // up rows (K = LR)
    std::vector<uint16_t> hwi((size_t) H * HD);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : hwi) { const float f = nd(rng) * 0.01f; uint32_t u; memcpy(&u, &f, 4); v = (uint16_t) (u >> 16); }
    const int Tmax = *std::max_element(Ts.begin(), Ts.end());
    std::vector<float> hx((size_t) Tmax * HD), hwn((size_t) HD);
    for (auto & v : hx) v = nd(rng);
    for (auto & v : hwn) v = 1.0f + 0.1f * nd(rng);
    const int64_t wd_nb1 = (HD / 32) * 34, wu_nb1 = (LR / 32) * 34, wi_nb1 = (int64_t) HD * 2;
    char * wd, * wu, * wi; float * x, * wn, * part, * pss, * lo, * scl, * dst, * ref;
    CK(cudaMalloc(&wd, hwd.size())); CK(cudaMemcpy(wd, hwd.data(), hwd.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&wu, hwu.size())); CK(cudaMemcpy(wu, hwu.data(), hwu.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&wi, hwi.size() * 2)); CK(cudaMemcpy(wi, hwi.data(), hwi.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&x, hx.size() * 4)); CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&wn, hwn.size() * 4)); CK(cudaMemcpy(wn, hwn.data(), hwn.size() * 4, cudaMemcpyHostToDevice));
    int64_t part_n = 0, pss_n = 0;
    ai::kern::q4x_hc_ws(N, LR, Tmax, true, &part_n, &pss_n);
    CK(cudaMalloc(&part, part_n * 4)); CK(cudaMalloc(&pss, pss_n * 4));
    CK(cudaMalloc(&lo, (size_t) Tmax * LR * 4)); CK(cudaMalloc(&scl, (size_t) Tmax * H * 4));
    const size_t dn = (size_t) (N + H) * Tmax;
    CK(cudaMalloc(&dst, dn * 4)); CK(cudaMalloc(&ref, dn * 4));
    void * tsc; CK(cudaMalloc(&tsc, ai::kern::q4x_hc_tc_scratch_bytes(N, Tmax)));
    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const double bytes = (double) hwd.size() + hwu.size() + hwi.size() * 2;
    printf("hc read: N %d H %d LR %d | weights %.2f MB per call\n", N, H, LR, bytes / 1e6);
    for (int T : Ts) {
        std::vector<float> outs[2];
        for (size_t u = 0; u < ups.size(); ++u) {
            if (ups[u] != 99) ai::kern::q4x_hc_set_up(ups[u]);
            auto call = [&]() {
                if (ups[u] == 99) ai::kern::q4x_hc_read_tc(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, 1, x, wn, part, pss, lo, scl, dst, N, LR, T, 1e-6f, tsc);
                else ai::kern::q4x_hc_read(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, 1, x, wn, part, pss, lo, scl, dst, N, LR, T, 1e-6f);
            };
            for (int i = 0; i < 10; ++i) call();
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            CK(cudaEventRecord(e0, st));
            for (int i = 0; i < reps; ++i) call();
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
            const double us = 1e3 * ms / reps;
            std::vector<float> o((size_t) (N + H) * T);
            CK(cudaMemcpy(o.data(), dst, o.size() * 4, cudaMemcpyDeviceToHost));
            if (u < 2) outs[u] = o;
            printf("T %d up %2d | %6.2f us per read | %5.0f GB/s over weights\n", T, ups[u], us, bytes / (us * 1e-6) / 1e9);
        }
        if (ups.size() >= 2) {
            double e2 = 0, r2 = 0, mx = 0; long long same = 0;
            for (size_t i = 0; i < outs[0].size(); ++i) {
                const double d = (double) outs[1][i] - outs[0][i];
                e2 += d * d; r2 += (double) outs[0][i] * outs[0][i]; mx = std::max(mx, std::fabs(d)); same += outs[1][i] == outs[0][i];
            }
            printf("T %d CHECK up %d vs %d: rel-L2 %.3g max|d| %.3g identical %.1f%%\n", T, ups[1], ups[0],
                   r2 > 0 ? std::sqrt(e2 / r2) : 0.0, mx, 100.0 * same / outs[0].size());
        }
    }
    return 0;
}
