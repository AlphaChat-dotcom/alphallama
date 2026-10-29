// SPDX-License-Identifier: Apache-2.0
// Small F16-weight GEMV for the router|score and GDN beta|alpha spans (AI_SMALL_GEMV).
//
// ggml's mul_mat_vec_f<half, half, T, 256> runs these small (~1 MB) matrices latency-bound, and it
// accumulates in half over 2048 products.
//
// One warp per output row, all T tokens per warp: lane i reads the row's 16-byte chunks i, i+32, ...
// (8 halves each, coalesced), converts to f32 and FMAs against the T activation columns, then one butterfly
// per token. f32 accumulation throughout. Not bit-identical to ggml (different reduction order and
// precision, more accurate), so it needs the paired quality check before becoming a default.
#include "kernels/kernels.h"

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

template <int T>
__global__ static void gemv_f16w_k(const __half * __restrict__ W, const float * __restrict__ x, float * __restrict__ y,
                                   int K, int N, long long x_tok, long long y_tok) {
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= N) {
        return;
    }
    const uint4 * wr = (const uint4 *) (W + (long long) row * K);
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
    for (int c = lane; c < K / 8; c += 32) {
        const uint4     u  = wr[c];
        const __half2 * h2 = (const __half2 *) &u;
        float           w[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(h2[i]);
            w[2 * i]       = f.x;
            w[2 * i + 1]   = f.y;
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 * xv = (const float4 *) (x + t * x_tok + (long long) c * 8);
            const float4   a = xv[0], b = xv[1];
            acc[t] = fmaf(w[0], a.x, acc[t]); acc[t] = fmaf(w[1], a.y, acc[t]);
            acc[t] = fmaf(w[2], a.z, acc[t]); acc[t] = fmaf(w[3], a.w, acc[t]);
            acc[t] = fmaf(w[4], b.x, acc[t]); acc[t] = fmaf(w[5], b.y, acc[t]);
            acc[t] = fmaf(w[6], b.z, acc[t]); acc[t] = fmaf(w[7], b.w, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float v = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, o, 32);
        }
        if (lane == 0) {
            y[t * y_tok + row] = v;
        }
    }
}

namespace ai::kern {

bool gemv_f16w(void * stream, const void * W, const float * x, float * y, int64_t K, int64_t N, int64_t T,
               int64_t x_tok, int64_t y_tok) {
    if (K % 8 != 0 || T < 1 || T > 8 || x_tok % 4 != 0 || ((uintptr_t) x % 16) != 0 || ((uintptr_t) W % 16) != 0) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[small-gemv] LAUNCHED (F16 weights, f32 accumulate, warp per row)\n");
    }
    const int    wpb = 8;
    const dim3   grid((unsigned) ((N + wpb - 1) / wpb));
    cudaStream_t st  = (cudaStream_t) stream;
    const __half * w = (const __half *) W;
#define GF_LAUNCH(TT) gemv_f16w_k<TT><<<grid, wpb * 32, 0, st>>>(w, x, y, (int) K, (int) N, (long long) x_tok, (long long) y_tok)
    switch (T) {
        case 1: GF_LAUNCH(1); break;
        case 2: GF_LAUNCH(2); break;
        case 3: GF_LAUNCH(3); break;
        case 4: GF_LAUNCH(4); break;
        case 5: GF_LAUNCH(5); break;
        case 6: GF_LAUNCH(6); break;
        case 7: GF_LAUNCH(7); break;
        default: GF_LAUNCH(8); break;
    }
#undef GF_LAUNCH
    return true;
}

}   // namespace ai::kern
