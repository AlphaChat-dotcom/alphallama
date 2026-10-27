// ai-q4x-densebench: the fp16 tensor-core dense GEMV (kernels/q4x_dense.cu) on synthetic rows.
//   check: vs an fp64 reference built from ggml's own CPU dequantisation (type traits to_float) and the fp32 x;
//   speed: us per call and the DRAM rate over the weight bytes, per T.
//   ai-q4x-densebench [--type q6_K] [--N 8192] [--K 2560] [--T 1,2,3,4,8,16,32] [--reps 100] [--checkN 256]
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
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

using namespace ai;

static int type_of(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * tn = ggml_type_name((ggml_type) t);
        if (tn && !strcmp(tn, s)) return t;
    }
    fprintf(stderr, "unknown type %s\n", s);
    exit(2);
}

int main(int argc, char ** argv) {
    int ty = GGML_TYPE_Q6_K, reps = 100, checkN = 256;
    int64_t N = 8192, K = 2560;
    std::vector<int> Ts = { 1, 2, 3, 4, 8, 16, 32 };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto nx = [&]() { return argv[++i]; };
        if (a == "--type") ty = type_of(nx());
        else if (a == "--N") N = atoll(nx());
        else if (a == "--K") K = atoll(nx());
        else if (a == "--reps") reps = atoi(nx());
        else if (a == "--checkN") checkN = atoi(nx());
        else if (a == "--T") { Ts.clear(); for (char * p = strtok(nx(), ","); p; p = strtok(nullptr, ",")) Ts.push_back(atoi(p)); }
        else { fprintf(stderr, "bad arg %s\n", a.c_str()); return 2; }
    }
    const size_t rb = ggml_row_size((ggml_type) ty, K), bs = ggml_type_size((ggml_type) ty);
    int mclk = 0, mbus = 0;
    CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0));
    CK(cudaDeviceGetAttribute(&mbus, cudaDevAttrGlobalMemoryBusWidth, 0));
    const double peak = 2.0 * mclk * 1e3 * (mbus / 8) / 1e9;
    printf("%s N %lld K %lld | %.1f MB | row %zu B\n", ggml_type_name((ggml_type) ty), (long long) N, (long long) K, rb * N / 1e6, rb);
    // weights: random bytes; every block's fp16 scale words set small (q6_K keeps d last, the others first)
    std::mt19937 rng(7);
    std::vector<uint8_t> hw(rb * (size_t) N);
    for (auto & b : hw) b = (uint8_t) rng();
    const size_t doff = ty == GGML_TYPE_Q6_K ? 208 : 0;
    for (size_t o = 0; o + bs <= hw.size(); o += bs) {
        uint16_t v = (uint16_t) (0x1C00 | (rng() & 0x3FF));
        memcpy(&hw[o + doff], &v, 2);
        if (ty == GGML_TYPE_Q4_K || ty == GGML_TYPE_Q5_K) { v = (uint16_t) (0x1800 | (rng() & 0x3FF)); memcpy(&hw[o + 2], &v, 2); }
    }
    const int Tmax = *std::max_element(Ts.begin(), Ts.end());
    std::vector<float> hx((size_t) (K * Tmax));
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : hx) v = nd(rng) * 3.0f;
    void * dw; float * dx, * dout; void * scr;
    CK(cudaMalloc(&dw, hw.size()));
    CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, hx.size() * 4));
    CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, sizeof(float) * N * Tmax));
    CK(cudaMalloc(&scr, kern::q4x_dense_scratch_bytes(K, Tmax)));
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    // fp64 reference for the first checkN rows (ggml's CPU dequantisation)
    const auto * tr = ggml_get_type_traits((ggml_type) ty);
    std::vector<float> wrow((size_t) K);
    std::vector<double> ref((size_t) checkN * Tmax);
    for (int n = 0; n < checkN && n < N; ++n) {
        tr->to_float(hw.data() + rb * n, wrow.data(), K);
        for (int t = 0; t < Tmax; ++t) {
            double s = 0;
            for (int64_t k = 0; k < K; ++k) s += (double) wrow[(size_t) k] * hx[(size_t) (t * K + k)];
            ref[(size_t) t * checkN + n] = s;
        }
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int T : Ts) {
        if (!kern::q4x_dense_ok(ty, N, K, T)) { printf("T %d declined\n", T); continue; }
        CK(cudaMemset(dout, 0, sizeof(float) * N * T));
        if (!kern::q4x_dense(st, ty, dw, (int64_t) rb, N, K, dx, K, T, scr, dout, N)) { printf("declined\n"); return 1; }
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        std::vector<float> o((size_t) N * T);
        CK(cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost));
        double e2 = 0, r2 = 0, mx = 0;
        long long bad = 0;
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < checkN && n < N; ++n) {
                const double r = ref[(size_t) t * checkN + n], d = (double) o[(size_t) t * N + n] - r;
                if (!std::isfinite(o[(size_t) t * N + n])) ++bad;
                e2 += d * d; r2 += r * r; mx = std::max(mx, std::fabs(d));
            }
        // column invariance: column 0 of this T vs column 0 of T = 1 (bitwise)
        std::vector<float> o1((size_t) N);
        if (!kern::q4x_dense(st, ty, dw, (int64_t) rb, N, K, dx, K, 1, scr, dout, N)) return 1;
        CK(cudaMemcpy(o1.data(), dout, o1.size() * 4, cudaMemcpyDeviceToHost));
        long long diff1 = 0;
        for (int64_t n = 0; n < N; ++n) diff1 += memcmp(&o1[(size_t) n], &o[(size_t) n], 4) != 0;
        for (int i = 0; i < 5; ++i) kern::q4x_dense(st, ty, dw, (int64_t) rb, N, K, dx, K, T, scr, dout, N);
        CK(cudaEventRecord(e0, st));
        for (int i = 0; i < reps; ++i) kern::q4x_dense(st, ty, dw, (int64_t) rb, N, K, dx, K, T, scr, dout, N);
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        const double us = 1e3 * ms / reps, gbs = rb * (double) N / (us * 1e-6) / 1e9;
        printf("T %2d | %8.1f us | %6.0f GB/s (%.0f%% of %.0f) | rel-L2 vs fp64 %.2e max|d| %.3g | non-finite %lld | col0 vs T=1 differing %lld\n",
               T, us, gbs, 100 * gbs / peak, peak, r2 > 0 ? std::sqrt(e2 / r2) : 0.0, mx, bad, diff1);
    }
    return 0;
}
