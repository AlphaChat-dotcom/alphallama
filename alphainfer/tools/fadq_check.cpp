// Kernel-level check for kernels/fa_decode_q.cu (AI_FA_Q) on the 35B attention shape (D 256, Hq 16, Hkv 2):
//   1. accuracy vs an f64 reference on the dequantised cache with f32 Q (the kernel quantises Q per 32);
//   2. MTP == plain: an L=4 verify row is bitwise equal to the same row run alone at L=1;
//   3. time per call over 10 layer copies (so the cache is not L2-resident), live == envelope and live == 55%,
//      against the native q4_0 kernel (fa_decode_q40), reported as GB/s of live K+V bytes.
//   ai-fadq-check [reps]
#include <cstddef>
#include "kernels/kernels.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int D = 256, HQ = 16, HKV = 2, G = HQ / HKV, NL = 10;

static uint16_t h16(float x) { __half h = __float2half(x); uint16_t u; memcpy(&u, &h, 2); return u; }
static float    f16(uint16_t u) { __half h; memcpy(&h, &u, 2); return __half2float(h); }

// one 256-row in ggml q8_0 / q4_0 layout; returns the dequantised values in deq
static void quant_row(int type, const float * x, uint8_t * out, float * deq) {
    for (int b = 0; b < 8; ++b) {
        const float * xb = x + 32 * b;
        if (type == 0) {
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(xb[i]));
            const float    d  = amax / 127.0f;
            const uint16_t dh = h16(d);
            uint8_t *      o  = out + 34 * b;
            memcpy(o, &dh, 2);
            for (int i = 0; i < 32; ++i) {
                const int q = d > 0 ? (int) lrintf(xb[i] / d) : 0;
                o[2 + i]    = (uint8_t) (int8_t) q;
                deq[32 * b + i] = (float) q * f16(dh);
            }
        } else {
            float amax = 0, mx = 0;
            for (int i = 0; i < 32; ++i) if (fabsf(xb[i]) > amax) { amax = fabsf(xb[i]); mx = xb[i]; }
            const float    d  = mx / -8.0f;
            const float    id = d != 0 ? 1.0f / d : 0.0f;
            const uint16_t dh = h16(d);
            uint8_t *      o  = out + 18 * b;
            memcpy(o, &dh, 2);
            int q[32];
            for (int i = 0; i < 32; ++i) {
                int v = (int) (xb[i] * id + 8.5f);
                q[i]  = v < 0 ? 0 : v > 15 ? 15 : v;
                deq[32 * b + i] = (float) (q[i] - 8) * f16(dh);
            }
            for (int j = 0; j < 16; ++j) o[2 + j] = (uint8_t) (q[j] | (q[j + 16] << 4));
        }
    }
}

struct Cache {
    int type, rowb;
    int64_t N;
    std::vector<uint8_t> hk, hv;     // layer 0 host copy [N][HKV][rowb]
    std::vector<float>   dk, dv;     // layer 0 dequantised [N][HKV][D]
    uint8_t * k[NL];
    uint8_t * v[NL];
};

static void build(Cache & c, int type, int64_t N, std::mt19937 & rng) {
    c.type = type; c.rowb = type == 0 ? 272 : 144; c.N = N;
    const size_t cs = (size_t) c.rowb * HKV;
    c.hk.resize(N * cs); c.hv.resize(N * cs); c.dk.resize(N * HKV * D); c.dv.resize(N * HKV * D);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(D);
    for (int64_t i = 0; i < N * HKV; ++i) {
        for (auto & e : x) e = nd(rng);
        quant_row(type, x.data(), c.hk.data() + i * c.rowb, c.dk.data() + i * D);
        for (auto & e : x) e = nd(rng);
        quant_row(type, x.data(), c.hv.data() + i * c.rowb, c.dv.data() + i * D);
    }
    for (int l = 0; l < NL; ++l) {
        CK(cudaMalloc(&c.k[l], N * cs)); CK(cudaMalloc(&c.v[l], N * cs));
        CK(cudaMemcpy(c.k[l], c.hk.data(), N * cs, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c.v[l], c.hv.data(), N * cs, cudaMemcpyHostToDevice));
    }
}
static void drop(Cache & c) { for (int l = 0; l < NL; ++l) { cudaFree(c.k[l]); cudaFree(c.v[l]); } }

struct Run {
    int64_t n_kv, live, L;
    std::vector<uint16_t> mask;   // [L][n_kv]
    std::vector<int64_t>  cells;  // [L]
};
static Run make_run(int64_t n_kv, int64_t live, int64_t L) {
    Run r{ n_kv, live, L, std::vector<uint16_t>(L * n_kv), std::vector<int64_t>(L) };
    for (int64_t l = 0; l < L; ++l) {
        const int64_t pos = live - L + l;
        r.cells[l]        = pos;
        for (int64_t c = 0; c < n_kv; ++c) r.mask[l * n_kv + c] = c <= pos ? 0 : 0xFC00;
    }
    return r;
}

struct Dev {
    float * q = nullptr; float * out = nullptr; float * ws = nullptr; uint16_t * mask = nullptr; int64_t * cells = nullptr;
};

static bool call_q(const Cache & c, int layer, const Run & r, const Dev & d, int split) {
    const int64_t cs = (int64_t) c.rowb * HKV;
    return ai::kern::fa_decode_q(nullptr, c.type, d.q, c.k[layer], c.v[layer], d.mask, d.cells, d.out, d.ws, D, HQ,
                                 HKV, r.L, 1, r.n_kv, split, cs, c.rowb, cs * c.N, r.n_kv, r.n_kv * r.L, 0, c.N,
                                 1.0f / 16.0f);
}
static bool call_old(const Cache & c, int layer, const Run & r, const Dev & d) {
    const int64_t cs = (int64_t) c.rowb * HKV;
    return ai::kern::fa_decode_q40(nullptr, d.q, c.k[layer], c.v[layer], d.mask, d.out, d.ws, D, HQ, HKV, r.L, 1,
                                   r.n_kv, 128, cs, c.rowb, cs * c.N, r.n_kv, r.n_kv * r.L, 1.0f / 16.0f);
}

static void upload_run(const Run & r, const Dev & d) {
    CK(cudaMemcpy(d.mask, r.mask.data(), r.mask.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.cells, r.cells.data(), r.cells.size() * 8, cudaMemcpyHostToDevice));
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 100;
    std::mt19937 rng(1234);
    const int64_t NMAX = 65536;
    Dev d;
    CK(cudaMalloc(&d.q, sizeof(float) * D * HQ * 8));
    CK(cudaMalloc(&d.out, sizeof(float) * D * HQ * 8));
    CK(cudaMalloc(&d.mask, 2 * NMAX * 8));
    CK(cudaMalloc(&d.cells, 8 * 8));
    const size_t wsb = sizeof(float) * (size_t) ai::kern::fa_decode_q40_workspace(NMAX, HQ * 8, 32);
    CK(cudaMalloc(&d.ws, wsb));
    std::vector<float> hq(D * HQ * 8);
    { std::normal_distribution<float> nd(0.f, 1.f); for (auto & e : hq) e = nd(rng); }
    CK(cudaMemcpy(d.q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    int bad = 0;

    for (int type = 0; type < 2; ++type) {
        const char * tn = type == 0 ? "q8_0" : "q4_0";
        Cache c;
        build(c, type, NMAX, rng);

        // ---- 1+2: accuracy and identity at n_kv 4096, live 3001, L 4
        {
            const int64_t n_kv = 4096, live = 3001, L = 4;
            const Run     r    = make_run(n_kv, live, L);
            upload_run(r, d);
            const int split = ai::kern::fa_decode_q_split(n_kv, HKV, 1);
            if (!call_q(c, 0, r, d, split)) { printf("%s: declined\n", tn); return 1; }
            std::vector<float> got(D * HQ * L);
            CK(cudaMemcpy(got.data(), d.out, got.size() * 4, cudaMemcpyDeviceToHost));
            double maxerr = 0, maxref = 0, se = 0, sr = 0;
            for (int64_t l = 0; l < L; ++l) {
                for (int h = 0; h < HQ; ++h) {
                    const int     hk = h / G;
                    const float * qr = hq.data() + (l * HQ + h) * D;
                    std::vector<double> sc(n_kv);
                    double m = -INFINITY;
                    for (int64_t cc = 0; cc <= r.cells[l]; ++cc) {
                        const float * kr = c.dk.data() + (cc * HKV + hk) * D;
                        double s = 0;
                        for (int i = 0; i < D; ++i) s += (double) qr[i] * kr[i];
                        sc[cc] = s / 16.0;
                        m      = std::max(m, sc[cc]);
                    }
                    std::vector<double> o(D, 0.0);
                    double den = 0;
                    for (int64_t cc = 0; cc <= r.cells[l]; ++cc) {
                        const double p = exp(sc[cc] - m);
                        den += p;
                        const float * vr = c.dv.data() + (cc * HKV + hk) * D;
                        for (int i = 0; i < D; ++i) o[i] += p * vr[i];
                    }
                    for (int i = 0; i < D; ++i) {
                        const double ref = o[i] / den, g = got[(l * HQ + h) * D + i];
                        maxerr = std::max(maxerr, fabs(g - ref));
                        maxref = std::max(maxref, fabs(ref));
                        se += (g - ref) * (g - ref);
                        sr += ref * ref;
                    }
                }
            }
            printf("%s accuracy (n_kv %lld, live %lld, L %lld): max|err| %.3e (max|ref| %.3e), rel RMS %.3e\n", tn,
                   (long long) n_kv, (long long) live, (long long) L, maxerr, maxref, sqrt(se / sr));
            if (!(sqrt(se / sr) < 2e-2)) { printf("  FAIL accuracy\n"); ++bad; }

            // identity: each row of the L=4 call vs the same row alone at L=1 (q row, mask row, its own cell)
            int differ = 0;
            for (int64_t l = 0; l < L; ++l) {
                Run r1 = make_run(n_kv, r.cells[l] + 1, 1);
                upload_run(r1, d);
                CK(cudaMemcpy(d.q, hq.data() + l * HQ * D, sizeof(float) * HQ * D, cudaMemcpyHostToDevice));
                call_q(c, 0, r1, d, split);
                std::vector<float> one(D * HQ);
                CK(cudaMemcpy(one.data(), d.out, one.size() * 4, cudaMemcpyDeviceToHost));
                differ += memcmp(one.data(), got.data() + l * HQ * D, one.size() * 4) != 0;
            }
            CK(cudaMemcpy(d.q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice));
            printf("%s identity: L=4 rows vs L=1 alone: %d of 4 rows differ%s\n", tn, differ, differ ? "  FAIL" : "");
            bad += differ != 0;
        }

        // ---- 3: timing
        for (int64_t n_kv : { 4096LL, 32768LL, 65536LL }) {
            for (int frac : { 100, 55 }) {
                for (int64_t L : { 1LL, 4LL }) {
                    const int64_t live = n_kv * frac / 100;
                    const Run     r    = make_run(n_kv, live, L);
                    upload_run(r, d);
                    const int split = ai::kern::fa_decode_q_split(n_kv, HKV, 1);
                    const double gb = (double) live * 2 * HKV * c.rowb / 1e9;
                    auto timeit = [&](bool old) {
                        for (int w = 0; w < 3; ++w) old ? call_old(c, w % NL, r, d) : call_q(c, w % NL, r, d, split);
                        CK(cudaEventRecord(e0));
                        for (int i = 0; i < reps; ++i) old ? call_old(c, i % NL, r, d) : call_q(c, i % NL, r, d, split);
                        CK(cudaEventRecord(e1));
                        CK(cudaEventSynchronize(e1));
                        float ms = 0;
                        CK(cudaEventElapsedTime(&ms, e0, e1));
                        return ms * 1000.0 / reps;
                    };
                    const double us = timeit(false);
                    printf("%s n_kv %6lld live %3d%% L %lld: fa_q %8.1f us = %6.1f GB/s live (split %d)", tn,
                           (long long) n_kv, frac, (long long) L, us, gb / (us * 1e-6), split);
                    if (type == 1) {
                        const double uo = timeit(true);
                        printf(" | old native %8.1f us = %6.1f GB/s", uo, gb / (uo * 1e-6));
                    }
                    printf("\n");
                }
            }
        }
        CK(cudaGetLastError());
        drop(c);
    }
    printf(bad ? "FAILED (%d)\n" : "ALL PASS\n", bad);
    return bad ? 1 : 0;
}
