// SPDX-License-Identifier: Apache-2.0
// MoE merge + residual + the next layer's input RMS norm, with the merge's full grid kept.
//
// Opt-in (AI_FUSE_MERGENORM=3) and slower than the two separate kernels: the lone last block runs the
// norm serially after the merge, on the critical path, with a quarter of the reference's threads.
// The norm is a latency-bound kernel on the critical path; folding it into the same dependency chain
// keeps its execution and can only save the inter-kernel gap. To remove its cost the norm must leave
// the path: partial sums of squares in the merge blocks, scale applied in the next consumer's load
// (the activation cast/quantise that reads h anyway).
//
// The other two fused variants:
//   - moe_merge_norm.cu runs one block per token, so at decode (T = 1..2) the merge that would
//     spread over 8-16 blocks runs on 1-2 SMs.
//   - moe_merge_norm_coop.cu keeps the grid but pays two grid.sync()s, each about a launch.
//
// This one: every block computes its slice of the merge exactly as moe_merge_gated_k does (same grid,
// same per-element operations), then the last block to finish -- found with one atomic, no barrier --
// norms every row. The norm reads 8 KB per token that is still in L2. Nothing waits except the last
// block.
//
// Bit-exactness, by construction from exact.cuh:
//   - merge half: ex::mul / ex::add (non-contracting) and ex::sigmoid, as moe_merge.cu. Using
//     `1.0f / (1.0f + exp(-s))` instead would be one ULP off the fast-math reference for some scores.
//   - norm half: the reference is rms_norm_f32<1024> (E = 2048 >= 1024). Its reduction is 1024 lanes
//     strided by 1024, warp butterfly, then 32 warp partials re-reduced. This block has 256 threads,
//     so each thread plays four virtual lanes v = tid + 256*m. The lanes of real warp w at fixed m are
//     exactly virtual warp w + 8*m, so four butterflies per warp give the same 32 partials in the same
//     order, and one more butterfly over s_sum[0..31] gives the same sum.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#define MMNL_THREADS 256
#define MMNL_REF     1024   // the reference norm's block size; the virtual-lane count

static __device__ unsigned int g_mmnl_done = 0;   // blocks finished this launch; the last one resets it

__global__ static void moe_merge_gated_norm_last_k(const float * __restrict__ e, const float * __restrict__ w,
                                                   const float * __restrict__ sd, const float * __restrict__ sc,
                                                   const float * __restrict__ r, float * y,
                                                   float * __restrict__ out, const float * __restrict__ nw,
                                                   long long E, int k, long long T, long long sc_stride,
                                                   long long sc_off, float eps) {
    __shared__ float s_sum[32];
    __shared__ bool  s_last;
    const int tid = threadIdx.x;

    // phase 1: the merge, identical per element to moe_merge_gated_k
    const long long idx = (long long) blockIdx.x * blockDim.x + tid;
    if (idx < E * T) {
        const long long t    = idx / E;
        const long long i    = idx - t * E;
        const long long base = (long long) k * t;
        float acc = ai::ex::mul(e[i + E * base], w[base]);
        for (int j = 1; j < k; ++j) {
            acc = ai::ex::add(acc, ai::ex::mul(e[i + E * (base + j)], w[base + j]));
        }
        const float sig = ai::ex::sigmoid(sc[t * sc_stride + sc_off]);
        acc    = ai::ex::add(acc, ai::ex::mul(sd[idx], sig));
        y[idx] = ai::ex::add(acc, r[idx]);
    }

    // is this the last block? (threadFenceReduction pattern)
    __syncthreads();
    if (tid == 0) {
        __threadfence();   // make this block's y visible device-wide before counting it
        const unsigned int prev = atomicAdd(&g_mmnl_done, 1u);
        s_last = prev == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // phase 2: the last block norms every row, in rms_norm_f32<1024>'s order
    for (long long t = 0; t < T; ++t) {
        const float * yr = y + t * E;
        float part[MMNL_REF / MMNL_THREADS];
#pragma unroll
        for (int m = 0; m < MMNL_REF / MMNL_THREADS; ++m) {
            float tmp = 0.0f;
            for (long long col = tid + MMNL_THREADS * m; col < E; col += MMNL_REF) {
                const float x = __ldcg(yr + col);   // L2, not a stale L1 line: other blocks wrote it
                tmp = __fmaf_rn(x, x, tmp);          // norm.cu's `tmp += x*x` contracts under fast-math
            }
            part[m] = ai::ex::warp_sum(tmp);
        }
        if (tid % 32 == 0) {
#pragma unroll
            for (int m = 0; m < MMNL_REF / MMNL_THREADS; ++m) {
                s_sum[tid / 32 + (MMNL_THREADS / 32) * m] = part[m];
            }
        }
        __syncthreads();
        const float sum   = ai::ex::warp_sum(s_sum[tid % 32]);
        const float scale = ai::ex::rms_scale(sum, (int) E, eps);
        for (long long col = tid; col < E; col += MMNL_THREADS) {
            out[t * E + col] = ai::ex::rms_apply(scale, __ldcg(yr + col), nw[col]);
        }
        __syncthreads();   // s_sum is reused by the next row
    }
    if (tid == 0) {
        g_mmnl_done = 0;   // ready for the next launch (stream order serialises launches)
    }
}

namespace ai::kern {

void moe_merge_residual_gated_norm_last(void * stream, const float * exps, const float * w,
                                        const float * shared_down, const float * score, int64_t score_stride,
                                        int64_t score_off, const float * res, float * y, float * out,
                                        const float * norm_w, int64_t E, int64_t k, int64_t T, float eps) {
    const long long n      = (long long) E * T;
    const long long blocks = (n + MMNL_THREADS - 1) / MMNL_THREADS;
    static bool     said   = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[merge-last] LAUNCHED (%lld blocks x %d threads, E=%lld T=%lld)\n", blocks, MMNL_THREADS,
                (long long) E, (long long) T);
    }
    moe_merge_gated_norm_last_k<<<(unsigned) blocks, MMNL_THREADS, 0, (cudaStream_t) stream>>>(
        exps, w, shared_down, score, res, y, out, norm_w, (long long) E, (int) k, (long long) T,
        (long long) score_stride, (long long) score_off, eps);
}

}   // namespace ai::kern
