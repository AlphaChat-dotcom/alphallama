// SPDX-License-Identifier: Apache-2.0
// Fused DeltaNet decode prologue: conv window + depthwise conv + SiLU + q/k L2-norm + gate/beta.
//
// Replaces CONCAT (history ++ new columns), SSM_CONV(+SiLU) and ggml's gdn_prep_kernel with one
// launch per layer. None of those kernels is bytes-bound -- the working set is a few KB already in
// L2 -- so their cost is mostly launch overhead.
//
// The grid shape matters most: one block per token would be a single block at decode. Each block
// instead owns one q head, one k head and a slice of v, so a decode token runs H_k blocks and the
// reduction each block needs is local to it.
//
// Not bit-identical to the chain it replaces: ggml's ssm_conv accumulates with FMA and its prep
// reduces q/k over a different lane assignment. Enabled by AI_FUSE_GDN.
#include "kernels/kernels.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define GDN_THREADS 256

static __device__ __forceinline__ float gdn_silu(float x) {
    return x / (1.0f + expf(-x));
}

// SiLU of the K-tap window sum for channel c of column t: history for the first K1 rows of the
// window, new columns after it.
static __device__ __forceinline__ float gdn_conv_at(const float * __restrict__ hist,
                                                    const float * __restrict__ qkv_b, long long qkv_s1, int c,
                                                    long long t, int K, int K1,
                                                    const float * __restrict__ conv_w) {
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        const long long r = t + j;
        const float     x = r < K1 ? hist[r] : qkv_b[c + (r - K1) * qkv_s1];
        acc += x * conv_w[j + (long long) K * c];
    }
    return gdn_silu(acc);
}

// The last K1 columns of the window become the new history. Race-free only at L == 1 (the host
// wrapper's contract): with more columns this block would overwrite history another block is reading.
static __device__ __forceinline__ void gdn_roll(float * __restrict__ hist, const float * __restrict__ qkv_b,
                                                long long qkv_s1, int c, long long L, int K1) {
    float nh[8];   // K1 <= 8, checked by the caller in ops.cpp
    for (int r = 0; r < K1; ++r) {
        const long long src = L + r;
        nh[r] = src < K1 ? hist[src] : qkv_b[c + (src - K1) * qkv_s1];
    }
    for (int r = 0; r < K1; ++r) {
        hist[r] = nh[r];
    }
}

// grid = (H_k, L*B). Block ih owns q head ih, k head ih and the ih-th slice of v.
__global__ static void gdn_conv_prep_k(const float * __restrict__ qkv, long long qkv_s1, long long qkv_s2,
                                       const float * __restrict__ ba, long long ba_s1, long long ba_s2,
                                       const float * __restrict__ conv_w, const float * __restrict__ dt_bias,
                                       const float * __restrict__ ssm_a, float * __restrict__ conv_state,
                                       float * __restrict__ conv_out, float * __restrict__ qh,
                                       float * __restrict__ kh, float * __restrict__ gh, float * __restrict__ bh,
                                       int C, int K, int Sk, int Hk, int Hv, long long L, float eps,
                                       int update_state, float * __restrict__ conv_in) {
    extern __shared__ float sm[];   // 2*Sk floats: this block's q head, then its k head

    const int       ih  = blockIdx.x;
    const long long t   = blockIdx.y % L;
    const long long b   = blockIdx.y / L;
    const int       K1  = K - 1;
    const long long row = b * L + t;

    const float * qkv_b   = qkv + b * qkv_s2;
    float *       state_b = conv_state + b * (long long) K1 * C;
    float *       out_b   = conv_out + row * C;

    for (int i = threadIdx.x; i < 2 * Sk; i += GDN_THREADS) {
        const int     c    = i < Sk ? ih * Sk + i : Sk * Hk + ih * Sk + (i - Sk);
        float * const hist = state_b + (long long) c * K1;
        const float   v    = gdn_conv_at(hist, qkv_b, qkv_s1, c, t, K, K1, conv_w);
        out_b[c] = v;
        sm[i]    = v;
        if (conv_in != nullptr) {   // record mode: history ++ new columns, for the fold's conv gather
            float * ci = conv_in + (b * C + c) * (long long) (K1 + L);
            ci[K1 + t] = qkv_b[c + t * qkv_s1];
            if (t == 0) {
                for (int r = 0; r < K1; ++r) {
                    ci[r] = hist[r];
                }
            }
        }
        if (update_state) {
            gdn_roll(hist, qkv_b, qkv_s1, c, L, K1);
        }
    }

    // v sits above q and k and needs no norm; split it evenly across the H_k blocks.
    const int vbase  = 2 * Sk * Hk;
    const int vchunk = (C - vbase + Hk - 1) / Hk;
    for (int i = threadIdx.x; i < vchunk; i += GDN_THREADS) {
        const int c = vbase + ih * vchunk + i;
        if (c >= C) {
            break;
        }
        float * const hist = state_b + (long long) c * K1;
        out_b[c] = gdn_conv_at(hist, qkv_b, qkv_s1, c, t, K, K1, conv_w);
        if (conv_in != nullptr) {
            float * ci = conv_in + (b * C + c) * (long long) (K1 + L);
            ci[K1 + t] = qkv_b[c + t * qkv_s1];
            if (t == 0) {
                for (int r = 0; r < K1; ++r) {
                    ci[r] = hist[r];
                }
            }
        }
        if (update_state) {
            gdn_roll(hist, qkv_b, qkv_s1, c, L, K1);
        }
    }
    __syncthreads();

    __shared__ float rq[GDN_THREADS];
    __shared__ float rk[GDN_THREADS];
    float            sq = 0.0f, sk = 0.0f;
    for (int i = threadIdx.x; i < Sk; i += GDN_THREADS) {
        sq += sm[i] * sm[i];
        sk += sm[Sk + i] * sm[Sk + i];
    }
    rq[threadIdx.x] = sq;
    rk[threadIdx.x] = sk;
    __syncthreads();
    for (int st = GDN_THREADS / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st) {
            rq[threadIdx.x] += rq[threadIdx.x + st];
            rk[threadIdx.x] += rk[threadIdx.x + st];
        }
        __syncthreads();
    }
    const float scq = 1.0f / fmaxf(sqrtf(rq[0]), eps);
    const float sck = 1.0f / fmaxf(sqrtf(rk[0]), eps);
    float *     yq  = qh + (row * Hk + ih) * (long long) Sk;
    float *     yk  = kh + (row * Hk + ih) * (long long) Sk;
    for (int i = threadIdx.x; i < Sk; i += GDN_THREADS) {
        yq[i] = sm[i] * scq;
        yk[i] = sm[Sk + i] * sck;
    }

    // gate = softplus(alpha + dt_bias) * ssm_a, beta = sigmoid(beta). One block covers all H_v.
    if (ih == 0) {
        const float * ba_row = ba + b * ba_s2 + t * ba_s1;
        for (int h = threadIdx.x; h < Hv; h += GDN_THREADS) {
            const float x = ba_row[Hv + h] + dt_bias[h];
            gh[row * Hv + h] = ((x > 20.0f) ? x : log1pf(expf(x))) * ssm_a[h];
            bh[row * Hv + h] = 1.0f / (1.0f + expf(-ba_row[h]));
        }
    }
}

// The same work on a wider, shallower grid (AI_GDN_CONV2). gdn_conv_prep_k runs few blocks, each serialising a
// q head, a k head, a v slice, two 8-level __syncthreads trees and (block 0) the gates. Here every q head, every
// k head and every 128-channel v slice is its own 128-thread block, and the gates get a block of their own:
//   grid.x = [0, Hk) q heads | [Hk, 2Hk) k heads | 2Hk + v slices | one gate block.
// Bit-exact with gdn_conv_prep_k by construction: the conv, SiLU, roll and record code are the same functions in
// the same file (same flags); the head norm reproduces the tree's order exactly -- with x_i = v_i*v_i for
// i < 128 and zeros above, the tree's partial at i < 32 after strides 128, 64, 32 is
// (x_i + x_{i+64}) + (x_{i+32} + x_{i+96}), and strides 16..1 are the xor butterfly's lane-0 order.
#define GDN2_THREADS 128
__global__ static void gdn_conv_prep2_k(const float * __restrict__ qkv, long long qkv_s1, long long qkv_s2,
                                        const float * __restrict__ ba, long long ba_s1, long long ba_s2,
                                        const float * __restrict__ conv_w, const float * __restrict__ dt_bias,
                                        const float * __restrict__ ssm_a, float * __restrict__ conv_state,
                                        float * __restrict__ conv_out, float * __restrict__ qh,
                                        float * __restrict__ kh, float * __restrict__ gh, float * __restrict__ bh,
                                        int C, int K, int Sk, int Hk, int Hv, long long L, float eps,
                                        int update_state, float * __restrict__ conv_in, int nvb) {
    const int       bx  = blockIdx.x;
    const long long t   = blockIdx.y % L;
    const long long b   = blockIdx.y / L;
    const int       K1  = K - 1;
    const long long row = b * L + t;
    const int       tid = threadIdx.x;

    const float * qkv_b   = qkv + b * qkv_s2;
    float *       state_b = conv_state + b * (long long) K1 * C;
    float *       out_b   = conv_out + row * C;

    if (bx == 2 * Hk + nvb) {   // gates: softplus(alpha + dt_bias) * ssm_a, sigmoid(beta)
        const float * ba_row = ba + b * ba_s2 + t * ba_s1;
        for (int h = tid; h < Hv; h += GDN2_THREADS) {
            const float x = ba_row[Hv + h] + dt_bias[h];
            gh[row * Hv + h] = ((x > 20.0f) ? x : log1pf(expf(x))) * ssm_a[h];
            bh[row * Hv + h] = 1.0f / (1.0f + expf(-ba_row[h]));
        }
        return;
    }
    // channel owned by this thread
    int c;
    if (bx < 2 * Hk) {
        c = bx < Hk ? bx * Sk + tid : Sk * Hk + (bx - Hk) * Sk + tid;   // q head bx, or k head bx - Hk
    } else {
        c = 2 * Sk * Hk + (bx - 2 * Hk) * GDN2_THREADS + tid;          // v slice
    }
    float v = 0.0f;
    if (c < C) {
        float * const hist = state_b + (long long) c * K1;
        v        = gdn_conv_at(hist, qkv_b, qkv_s1, c, t, K, K1, conv_w);
        out_b[c] = v;
        if (conv_in != nullptr) {
            float * ci = conv_in + (b * C + c) * (long long) (K1 + L);
            ci[K1 + t] = qkv_b[c + t * qkv_s1];
            if (t == 0) {
                for (int r = 0; r < K1; ++r) {
                    ci[r] = hist[r];
                }
            }
        }
        if (update_state) {
            gdn_roll(hist, qkv_b, qkv_s1, c, L, K1);
        }
    }
    if (bx >= 2 * Hk) {
        return;
    }
    // head L2 norm, in gdn_conv_prep_k's exact summation order
    __shared__ float x2[GDN2_THREADS];
    float            sq = 0.0f;
    sq += v * v;
    x2[tid] = sq;
    __syncthreads();
    if (tid < 32) {
        float a = (x2[tid] + x2[tid + 64]) + (x2[tid + 32] + x2[tid + 96]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a = a + __shfl_xor_sync(0xffffffffu, a, o, 32);
        }
        if (tid == 0) {
            x2[0] = a;   // lane 0 holds the tree's order; other lanes hold other orders
        }
    }
    __syncthreads();
    const float sc = 1.0f / fmaxf(sqrtf(x2[0]), eps);
    float *     y  = (bx < Hk ? qh + (row * Hk + bx) * (long long) Sk : kh + (row * Hk + (bx - Hk)) * (long long) Sk);
    y[tid] = v * sc;
}

// AI_GDN_CONV2=2 instrument: bitwise differences between the two kernels' outputs
static __device__ unsigned long long g_gc2[3];   // calls, words compared, words differing
__global__ static void gdn_conv_cmp_k(const float * a, const float * b, long long n) {
    unsigned long long d = 0;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        d += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    }
    if (d) {
        atomicAdd(&g_gc2[2], d);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&g_gc2[0], 1ull);
        atomicAdd(&g_gc2[1], (unsigned long long) n);
    }
}

namespace ai::kern {

void gdn_conv_prep(void * stream, const float * qkv, int64_t qkv_s1, int64_t qkv_s2, const float * ba,
                   int64_t ba_s1, int64_t ba_s2, const float * conv_w, const float * dt_bias, const float * ssm_a,
                   float * conv_state, float * conv_out, float * qh, float * kh, float * gh, float * bh, int C,
                   int K, int Sk, int Hk, int Hv, int64_t L, int64_t B, float eps, bool update_state,
                   float * conv_in) {
    // AI_GDN_CONV2: 0 the original kernel; 1 the wide grid (gdn_conv_prep2_k, default); 2 compare: the original
    // writes to shadow buffers without rolling the state, then the wide grid runs for real, and every output word
    // is compared bitwise (printed at exit).
    static const int mode = []() {
        const char * e = getenv("AI_GDN_CONV2");
        return e ? atoi(e) : 1;
    }();
    cudaStream_t st   = (cudaStream_t) stream;
    const bool   wide = mode != 0 && Sk == GDN2_THREADS && K >= 1 && K <= 9;
    if (!wide) {
        const dim3   grid((unsigned) Hk, (unsigned) (L * B));
        const size_t shared = (size_t) 2 * Sk * sizeof(float);
        gdn_conv_prep_k<<<grid, GDN_THREADS, shared, st>>>(
            qkv, (long long) qkv_s1, (long long) qkv_s2, ba, (long long) ba_s1, (long long) ba_s2, conv_w, dt_bias,
            ssm_a, conv_state, conv_out, qh, kh, gh, bh, C, K, Sk, Hk, Hv, (long long) L, eps, update_state ? 1 : 0,
            conv_in);
        return;
    }
    const int nvb = (C - 2 * Sk * Hk + GDN2_THREADS - 1) / GDN2_THREADS;
    // compare mode: the original kernel into shadow buffers, before the wide grid rolls the history
    static float * sh = nullptr;
    constexpr long long SH_WORDS = 24ll << 20;   // 96 MiB, allocated once; calls that do not fit are skipped
    const long long     n_out = (long long) C * L * B, n_qk = (long long) Hk * Sk * L * B, n_g = (long long) Hv * L * B;
    const long long     n_ci  = conv_in ? (long long) B * C * (K - 1 + L) : 0;
    const bool          cmp   = mode == 2 && n_out + 2 * n_qk + 2 * n_g + n_ci <= SH_WORDS;
    if (cmp) {
        if (sh == nullptr) {
            cudaStreamCaptureMode m = cudaStreamCaptureModeRelaxed;
            cudaThreadExchangeStreamCaptureMode(&m);
            cudaMalloc(&sh, (size_t) SH_WORDS * sizeof(float));
            cudaThreadExchangeStreamCaptureMode(&m);
            atexit([]() {
                unsigned long long h[3] = { 0, 0, 0 };
                cudaMemcpyFromSymbol(h, g_gc2, sizeof(h));
                fprintf(stderr, "[gdn-conv2-compare] arrays %llu | words %llu, differing %llu\n", h[0], h[1], h[2]);
            });
        }
        float * so = sh, *sq = so + n_out, *sk = sq + n_qk, *sg = sk + n_qk, *sb = sg + n_g, *sci = conv_in ? sb + n_g : nullptr;
        const dim3 grid((unsigned) Hk, (unsigned) (L * B));
        gdn_conv_prep_k<<<grid, GDN_THREADS, (size_t) 2 * Sk * sizeof(float), st>>>(
            qkv, (long long) qkv_s1, (long long) qkv_s2, ba, (long long) ba_s1, (long long) ba_s2, conv_w, dt_bias,
            ssm_a, conv_state, so, sq, sk, sg, sb, C, K, Sk, Hk, Hv, (long long) L, eps, 0, sci);
    }
    const dim3 grid2((unsigned) (2 * Hk + nvb + 1), (unsigned) (L * B));
    gdn_conv_prep2_k<<<grid2, GDN2_THREADS, 0, st>>>(
        qkv, (long long) qkv_s1, (long long) qkv_s2, ba, (long long) ba_s1, (long long) ba_s2, conv_w, dt_bias,
        ssm_a, conv_state, conv_out, qh, kh, gh, bh, C, K, Sk, Hk, Hv, (long long) L, eps, update_state ? 1 : 0,
        conv_in, nvb);
    if (cmp) {
        float * so = sh, *sq = so + n_out, *sk = sq + n_qk, *sg = sk + n_qk, *sb = sg + n_g, *sci = sb + n_g;
        gdn_conv_cmp_k<<<64, 256, 0, st>>>(so, conv_out, n_out);
        gdn_conv_cmp_k<<<16, 256, 0, st>>>(sq, qh, n_qk);
        gdn_conv_cmp_k<<<16, 256, 0, st>>>(sk, kh, n_qk);
        gdn_conv_cmp_k<<<1, 256, 0, st>>>(sg, gh, n_g);
        gdn_conv_cmp_k<<<1, 256, 0, st>>>(sb, bh, n_g);
        if (conv_in) {
            gdn_conv_cmp_k<<<64, 256, 0, st>>>(sci, conv_in, n_ci);
        }
    }
}

}   // namespace ai::kern
