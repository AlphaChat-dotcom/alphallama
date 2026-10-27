// MoE merge + residual + the next layer's input RMS norm, without losing the merge's parallelism.
//
// The straightforward fusion (moe_merge_norm.cu) is bit-identical but gains nothing: a row-wide
// reduction forces one block per row, where the plain merge runs as several blocks at decode, and
// serialising the merge costs what the removed norm launches save.
//
// This version keeps the merge spread across the grid and pays for a reduction with two grid-wide
// barriers instead:
//
//   phase 1  every block computes its slice of y = sum_j e_j*w_j + shared*sigmoid(score) + residual
//   grid.sync()
//   phase 2  block 0 alone computes the row scale, with 1024 threads in ggml's exact strided order
//   grid.sync()
//   phase 3  every block applies out = scale * y * w_next over its slice
//
// Phase 2 is deliberately single-block: rms_norm_f32 reduces a 2048-wide row with block_size 1024,
// per-thread stride 1024, then a warp butterfly and a 32-partial second pass. Splitting that across
// blocks would change the summation order and cost bit-exactness, and the read is 8 KB already in L2.
//
// Rounding contracts, as in moe_merge_norm.cu: the merge half is explicitly non-contracted
// (__fmul_rn/__fadd_rn) because the elementwise ops it replaces are; the norm half uses ordinary ops
// and __fdividef to match norm.cu, which is built with -use_fast_math.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define MMC_THREADS 1024

static __device__ __forceinline__ float mmc_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__global__ static void moe_merge_gated_norm_coop_k(const float * __restrict__ e, const float * __restrict__ w,
                                                   const float * __restrict__ sd, const float * __restrict__ sc,
                                                   const float * __restrict__ r, float * __restrict__ y,
                                                   float * __restrict__ out, const float * __restrict__ nw,
                                                   float * __restrict__ scale_out, long long E, int k, long long T,
                                                   long long sc_stride, long long sc_off, float eps) {
    cg::grid_group  grid  = cg::this_grid();
    const long long n     = E * T;
    const long long per   = (n + gridDim.x - 1) / gridDim.x;   // contiguous chunk per block
    const long long start = (long long) blockIdx.x * per;
    const long long stop  = min(start + per, n);

    for (long long idx = start + threadIdx.x; idx < stop; idx += MMC_THREADS) {
        const long long t    = idx / E;
        const long long i    = idx - t * E;
        const long long base = (long long) k * t;
        const float     s    = sc[t * sc_stride + sc_off];
        const float     sig  = ai::ex::sigmoid(s);   // not 1/(1+exp(-s)): one ULP off the fast-math reference (see moe_merge.cu)
        float           acc  = __fmul_rn(e[i + E * base], w[base]);
        for (int j = 1; j < k; ++j) {
            acc = __fadd_rn(acc, __fmul_rn(e[i + E * (base + j)], w[base + j]));
        }
        acc    = __fadd_rn(acc, __fmul_rn(sd[idx], sig));
        y[idx] = __fadd_rn(acc, r[idx]);
    }

    grid.sync();

    if (blockIdx.x == 0) {
        __shared__ float s_sum[32];
        const int        tid = threadIdx.x;
        for (long long t = 0; t < T; ++t) {
            float tmp = 0.0f;
            for (long long col = tid; col < E; col += MMC_THREADS) {
                const float xi = y[t * E + col];
                tmp += xi * xi;
            }
            tmp                 = mmc_warp_sum(tmp);
            const int warp_id   = tid / 32;
            const int lane_id   = tid % 32;
            if (lane_id == 0) {
                s_sum[warp_id] = tmp;
            }
            __syncthreads();
            tmp = 0.0f;
            if (lane_id < MMC_THREADS / 32) {
                tmp = s_sum[lane_id];
            }
            tmp = mmc_warp_sum(tmp);
            if (tid == 0) {
                scale_out[t] = rsqrtf(__fdividef(tmp, (float) E) + eps);
            }
            __syncthreads();
        }
    }

    grid.sync();

    for (long long idx = start + threadIdx.x; idx < stop; idx += MMC_THREADS) {
        const long long t   = idx / E;
        const long long col = idx - t * E;
        out[idx]            = scale_out[t] * y[idx] * nw[col];
    }
}

namespace ai::kern {

bool moe_merge_residual_gated_norm_coop(void * stream, const float * exps, const float * w,
                                        const float * shared_down, const float * score, int64_t score_stride,
                                        int64_t score_off, const float * res, float * y, float * out,
                                        const float * norm_w, float * scale_scratch, int64_t E, int64_t k,
                                        int64_t T, float eps) {
    // One block per 1024 elements, capped at what can be co-resident -- a cooperative launch requires
    // every block to be scheduled at once, so asking for more than the device can hold fails.
    // Match the plain merge's grid width -- it ran ~256 elements per block, and that grid shape is
    // what the fusion has to preserve to be worth anything (AI_MERGE_COOP_CHUNK overrides).
    static const int chunk = []() {
        const char * e = getenv("AI_MERGE_COOP_CHUNK");
        const int    v = e ? atoi(e) : 256;
        return v < 1 ? 256 : v;
    }();
    int blocks = (int) ((E * T + chunk - 1) / chunk);
    if (blocks < 1) {
        blocks = 1;
    }
    int max_blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void *) moe_merge_gated_norm_coop_k,
                                                      MMC_THREADS, 0) != cudaSuccess) {
        return false;
    }
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int cap = max_blocks * sms;
    if (cap > 0 && blocks > cap) {
        blocks = cap;
    }

    const long long  Ell = E, Tll = T, ss = score_stride, so = score_off;
    const int        kk  = (int) k;
    void *           args[]  = { (void *) &exps, (void *) &w,   (void *) &shared_down, (void *) &score,
                                 (void *) &res,  (void *) &y,   (void *) &out,         (void *) &norm_w,
                                 (void *) &scale_scratch, (void *) &Ell, (void *) &kk, (void *) &Tll,
                                 (void *) &ss,   (void *) &so,  (void *) &eps };
    const dim3 grid((unsigned) blocks), block(MMC_THREADS);
    const cudaError_t rc = cudaLaunchCooperativeKernel((void *) moe_merge_gated_norm_coop_k, grid, block,
                                                      args, 0, (cudaStream_t) stream);
    // Liveness: a refused cooperative launch falls back to the single-block fusion, which is also
    // bit-identical -- so "correct and no speedup" would be indistinguishable from "never fired".
    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[merge-coop] %s (%d blocks x %d threads)%s%s\n",
                rc == cudaSuccess ? "LAUNCHED" : "REFUSED -> falling back", blocks, MMC_THREADS,
                rc == cudaSuccess ? "" : ": ", rc == cudaSuccess ? "" : cudaGetErrorString(rc));
    }
    return rc == cudaSuccess;
}

}   // namespace ai::kern
