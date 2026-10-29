// SPDX-License-Identifier: Apache-2.0
// Prefill-width y[N, T] = W[N, K] (F16, row-major) x[K, T] (f32) on fp16 tensor cores with fp32 accumulation.
// Enabled by AI_GEMM_F16_PF (default off).
//
// The router|score (N = 257) and GDN beta|alpha (N = 2 Hv) projections keep F16 weights. At T <= 8 they run the native
// GEMV (gemv_f16w); at prefill width they otherwise go through ggml, which converts x to f16 (convert_unary), runs a
// cuBLAS half-accumulate GEMM, a split-K reduce in half, converts back to f32 and copies -- slow for ~1 GFLOP per
// call. This reads x in f32, rounds it to f16 in shared memory (as the conversion does), and accumulates in fp32
// (more accurate than the half accumulation it replaces, so not bit-identical: routing can change).
// Tile: 64 tokens x 64 outputs per block, 4 warps of 16 tokens x 64 outputs, k-steps of 32.
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace ai {
namespace kern {
namespace {

constexpr int GP_BT = 64, GP_BN = 64, GP_BK = 32, GP_LD = GP_BK + 8;   // 80 B rows: ldmatrix rows hit 8 bank groups

__device__ __forceinline__ void gp_ldsm_x4(unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void gp_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                       unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128)
gemm_f16w_pf_k(const __half * __restrict__ W, const float * __restrict__ x, float * __restrict__ y, int K, int N, int T,
               int64_t x_tok, int64_t y_tok) {
    __shared__ __align__(16) __half xs[GP_BT][GP_LD];
    __shared__ __align__(16) __half ws[GP_BN][GP_LD];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int t0 = blockIdx.x * GP_BT, n0 = blockIdx.y * GP_BN;
    float acc[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][q] = 0.f;
    // loader mapping: x tile 64 x 32 f32 -> 128 threads x 16 values; W tile 64 x 32 f16 -> 128 threads x 16 halves
    const int lr = tid >> 1, lc = (tid & 1) * 16;
    const bool xok = t0 + lr < T, wok = n0 + lr < N;
    const float *  xrow = x + (int64_t) (t0 + (xok ? lr : 0)) * x_tok + lc;
    const __half * wrow = W + (int64_t) (n0 + (wok ? lr : 0)) * K + lc;
    for (int k0 = 0; k0 < K; k0 += GP_BK) {
        {
            __half hv[16];
#pragma unroll
            for (int i = 0; i < 16; i += 4) {
                const float4 f = xok ? *reinterpret_cast<const float4 *>(xrow + k0 + i) : make_float4(0.f, 0.f, 0.f, 0.f);
                hv[i] = __float2half_rn(f.x); hv[i + 1] = __float2half_rn(f.y);
                hv[i + 2] = __float2half_rn(f.z); hv[i + 3] = __float2half_rn(f.w);
            }
            *reinterpret_cast<uint4 *>(&xs[lr][lc])     = *reinterpret_cast<const uint4 *>(&hv[0]);
            *reinterpret_cast<uint4 *>(&xs[lr][lc + 8]) = *reinterpret_cast<const uint4 *>(&hv[8]);
            uint4 w0 = make_uint4(0, 0, 0, 0), w1 = w0;
            if (wok) {
                w0 = *reinterpret_cast<const uint4 *>(wrow + k0);
                w1 = *reinterpret_cast<const uint4 *>(wrow + k0 + 8);
            }
            *reinterpret_cast<uint4 *>(&ws[lr][lc])     = w0;
            *reinterpret_cast<uint4 *>(&ws[lr][lc + 8]) = w1;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GP_BK; kk += 16) {
            unsigned a0, a1, a2, a3;
            gp_ldsm_x4(a0, a1, a2, a3, &xs[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
            for (int np = 0; np < 4; ++np) {   // two n-tiles per ldmatrix.x4: (n lo, k lo), (n lo, k hi), (n hi, ...)
                unsigned b0, b1, b2, b3;
                gp_ldsm_x4(b0, b1, b2, b3, &ws[np * 16 + (lane >> 4) * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
                gp_mma(acc[2 * np], a0, a1, a2, a3, b0, b1);
                gp_mma(acc[2 * np + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        __syncthreads();
    }
    const int ta = t0 + warp * 16 + g, tb = ta + 8;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const int n = n0 + nt * 8 + 2 * t;
        if (ta < T) {
            if (n < N)     y[(int64_t) ta * y_tok + n]     = acc[nt][0];
            if (n + 1 < N) y[(int64_t) ta * y_tok + n + 1] = acc[nt][1];
        }
        if (tb < T) {
            if (n < N)     y[(int64_t) tb * y_tok + n]     = acc[nt][2];
            if (n + 1 < N) y[(int64_t) tb * y_tok + n + 1] = acc[nt][3];
        }
    }
}

}  // namespace

bool gemm_f16w_pf_enabled() {
    static const bool v = getenv("AI_GEMM_F16_PF") ? atoi(getenv("AI_GEMM_F16_PF")) != 0 : false;
    return v;
}

bool gemm_f16w_pf(void * stream, const void * W, const float * x, float * y, int64_t K, int64_t N, int64_t T,
                  int64_t x_tok, int64_t y_tok) {
    if (K % GP_BK != 0 || x_tok % 4 != 0 || ((uintptr_t) x % 16) != 0 || ((uintptr_t) W % 16) != 0 || T < 1 || N < 1) {
        return false;
    }
    const dim3 grid((unsigned) ((T + GP_BT - 1) / GP_BT), (unsigned) ((N + GP_BN - 1) / GP_BN));
    gemm_f16w_pf_k<<<grid, 128, 0, (cudaStream_t) stream>>>((const __half *) W, x, y, (int) K, (int) N, (int) T, x_tok,
                                                             y_tok);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace kern
}  // namespace ai
