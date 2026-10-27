// MoE merge + residual + the next layer's input RMS norm, in one kernel.
//
// The merge already ends a layer with y = sum_j e_j*w_j + shared*sigmoid(score) + residual. The very
// next thing the graph does is rms_norm(y) * w_next to open the following layer. That norm is pure
// launch latency: it re-reads 8 KB that this kernel just wrote and still has in L2. Folding it in
// removes one launch per layer.
//
// Two different rounding contracts in one kernel, which is why this is its own translation unit:
//   - the merge half must not contract, because the elementwise ops it replaces do not. moe_merge.cu
//     gets that from -fmad=false; here it is explicit (__fmul_rn / __fadd_rn), so the guarantee does
//     not depend on the file's compile flags.
//   - the norm half must match ggml's norm.cu, which is compiled with -use_fast_math: `tmp += v*v`
//     contracts to an FMA there, and the mean division is the approximate one. So that half uses
//     ordinary ops (this file has default contraction) and __fdividef. Getting this backwards makes
//     the q8 staging copy differ by one int8 code on rare rows.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cuda_runtime.h>

#define MMN_THREADS 1024

static __device__ __forceinline__ float mmn_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__global__ static void moe_merge_gated_norm_k(const float * __restrict__ e, const float * __restrict__ w,
                                              const float * __restrict__ sd, const float * __restrict__ sc,
                                              const float * __restrict__ r, float * __restrict__ y,
                                              float * __restrict__ out, const float * __restrict__ nw,
                                              long long E, int k, long long sc_stride, long long sc_off,
                                              float eps) {
    __shared__ float s_sum[32];

    const long long t    = blockIdx.x;
    const int       tid  = threadIdx.x;
    const long long base = (long long) k * t;
    const float     s    = sc[t * sc_stride + sc_off];
    const float     sig  = ai::ex::sigmoid(s);   // not 1/(1+exp(-s)): one ULP off the fast-math reference (see moe_merge.cu)

    float tmp = 0.0f;
    for (long long i = tid; i < E; i += MMN_THREADS) {
        const long long idx = t * E + i;
        float           acc = __fmul_rn(e[i + E * base], w[base]);
        for (int j = 1; j < k; ++j) {
            acc = __fadd_rn(acc, __fmul_rn(e[i + E * (base + j)], w[base + j]));
        }
        acc      = __fadd_rn(acc, __fmul_rn(sd[idx], sig));
        const float v = __fadd_rn(acc, r[idx]);
        y[idx]   = v;
        tmp += v * v;   // contracts to an FMA, as it does in norm.cu
    }

    // ggml's block_reduce<SUM, 1024>, reproduced exactly
    tmp = mmn_warp_sum(tmp);
    {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = 0.0f;
        if (lane_id < MMN_THREADS / 32) {
            tmp = s_sum[lane_id];
        }
        tmp = mmn_warp_sum(tmp);
    }

    const float mean  = __fdividef(tmp, (float) E);
    const float scale = rsqrtf(mean + eps);
    for (long long i = tid; i < E; i += MMN_THREADS) {
        const long long idx = t * E + i;
        out[idx] = scale * y[idx] * nw[i];
    }
}

namespace ai::kern {

void moe_merge_residual_gated_norm(void * stream, const float * exps, const float * w, const float * shared_down,
                                   const float * score, int64_t score_stride, int64_t score_off, const float * res,
                                   float * y, float * out, const float * norm_w, int64_t E, int64_t k, int64_t T,
                                   float eps) {
    moe_merge_gated_norm_k<<<(unsigned) T, MMN_THREADS, 0, (cudaStream_t) stream>>>(
        exps, w, shared_down, score, res, y, out, norm_w, (long long) E, (int) k, (long long) score_stride,
        (long long) score_off, eps);
}

}   // namespace ai::kern
