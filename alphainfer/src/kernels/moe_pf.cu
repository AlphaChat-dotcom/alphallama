// SPDX-License-Identifier: Apache-2.0
// Prefill-sized routed experts as grouped int8 tensor-core GEMMs. Enabled by AI_MOE_PF.
//
// ggml's MMQ MUL_MAT_ID streams expert weights at a fraction of DRAM bandwidth at prefill shapes (stream-k, one
// persistent 8-warp block per SM). This file replaces gate + up + SwiGLU + down for T >= AI_MOE_PF_MIN rows:
//   - routing tables on device (count, scan, scatter, chunk tables for 32- and 64-row slices), so captured rounds replay;
//   - activations quantised to int8 per 32 values (d, d*sum), warp per block; SwiGLU fused with the down quantiser;
//   - int8 mma.m16n8k32 per 32-value sub-block, q4_K / q5_K scale and min applied in fp32, cp.async double buffers,
//     activation rows padded 16 B in smem (bank groups). Three tiles:
//       S: 64 weight rows x 32 routed rows, 8 warps of 16x16      -- gate/up while experts see <= ~64 rows
//       L: 128 x 64, 8 warps of 32x32 (2 m x 4 n mma tiles)        -- gate/up above that (weights reread 2x less)
//       D: 128 x 64 with the slice's whole K resident and 4 weight row tiles streamed per CTA -- down (K <= 512)
//     Per-expert traffic ~ W * (T_e / BN) + A * (N / BM); MMQ's tile is 128 x 128.
// Not bit-identical to MMQ (own activation quantiser and reduction order). Plain and MTP prefill both take it, so
// MTP and plain decoding stay identical.
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ai {
namespace kern {
namespace {

constexpr int PF_SB = 256;
constexpr int GGML_Q4_K = 12, GGML_Q5_K = 13;
constexpr int PF_KD_MAX = 512;   // tile D keeps K <= this resident
constexpr int PF_KD_RPC = 4;     // tile D: 128-row weight tiles per CTA

template <int QT> struct pf_qt;
template <> struct pf_qt<0> { static constexpr int QB = 144, QS = 16, QH = -1; };   // q4_K
template <> struct pf_qt<1> { static constexpr int QB = 176, QS = 48, QH = 16; };   // q5_K

__device__ __forceinline__ void pf_cp16(void * s, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void pf_cp4(void * s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void pf_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void pf_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }
__device__ __forceinline__ void pf_mma(int (&c)[4], const int (&a)[4], const int (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void pf_scale_min(int j, const unsigned char * q, int & sc, int & m) {   // q4_K and q5_K
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template <int QT> __device__ __forceinline__ int pf_wq4(const unsigned char * blk, int j, int kk) {
    const int lo = (*reinterpret_cast<const int *>(blk + pf_qt<QT>::QS + 32 * (j >> 1) + kk) >> ((j & 1) * 4)) & 0x0F0F0F0F;
    if constexpr (QT == 1) {
        const int hb = (*reinterpret_cast<const int *>(blk + pf_qt<QT>::QH + kk) >> j) & 0x01010101;
        return lo | (hb << 4);
    } else {
        return lo;
    }
}

// ---- activation quantiser: rows of K floats (column stride xcs floats) -> int8 + d + d*sum per 32
__global__ void pf_quant(const float * __restrict__ x, int64_t xcs, int64_t K, int64_t rows, int8_t * __restrict__ xq,
                         float * __restrict__ xd, float * __restrict__ xs) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t nb = K / 32;
    if (w >= rows * nb) return;
    const int64_t r = w / nb, b = w % nb;
    const float v = x[r * xcs + b * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    const int q = __float2int_rn(v * id);
    xq[r * K + b * 32 + lane] = (int8_t) q;
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) { xd[w] = d; xs[w] = d * (float) sum; }
}

// ---- SwiGLU (silu(g) * u, matching ggml's op_silu built with -use_fast_math) fused with the down input's quantiser
__global__ void pf_swiglu_quant(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ act,
                                int64_t F, int64_t rows, int8_t * __restrict__ aq, float * __restrict__ ad, float * __restrict__ as) {
    const int64_t w = ((int64_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w >= rows * (F / 32)) return;
    const int64_t i = w * 32 + lane;
    const float gv = g[i];
    const float v = gv / (1.0f + __expf(-gv)) * u[i];
    act[i] = v;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    const int q = __float2int_rn(v * id);
    aq[i] = (int8_t) q;
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) { ad[w] = d; as[w] = d * (float) sum; }
}

// ---- routing tables. Routed row r = t*k + slot; ids[t*ids_stride + slot].
__global__ void pf_count(const int32_t * __restrict__ ids, int64_t ids_stride, int k, int64_t T, int * __restrict__ cnt) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * k) return;
    atomicAdd(&cnt[ids[(i / k) * ids_stride + i % k]], 1);
}
// one block of n_expert threads: row offsets, cursor copy, and chunk tables (expert, first row) for 32- and 64-row
// slices with their sizes. sh = 3 * n_expert ints.
__global__ void pf_table(const int * __restrict__ cnt, int n_expert, int * __restrict__ off, int * __restrict__ cur,
                         int * __restrict__ ce32, int * __restrict__ cc32, int * __restrict__ ce64, int * __restrict__ cc64,
                         int * __restrict__ n_chunks /* [2]: 32, 64 */) {
    extern __shared__ int sh[];
    const int e = threadIdx.x;
    const int c = e < n_expert ? cnt[e] : 0, n32 = (c + 31) / 32, n64 = (c + 63) / 64;
    sh[e] = c;
    sh[n_expert + e] = n32;
    sh[2 * n_expert + e] = n64;
    __syncthreads();
    for (int s = 1; s < n_expert; s <<= 1) {   // inclusive scans (Hillis-Steele)
        int a = 0, b = 0, d = 0;
        if (e >= s) { a = sh[e - s]; b = sh[n_expert + e - s]; d = sh[2 * n_expert + e - s]; }
        __syncthreads();
        if (e >= s) { sh[e] += a; sh[n_expert + e] += b; sh[2 * n_expert + e] += d; }
        __syncthreads();
    }
    if (e < n_expert) {
        const int o = sh[e] - c, b32 = sh[n_expert + e] - n32, b64 = sh[2 * n_expert + e] - n64;
        off[e] = o;
        cur[e] = o;
        for (int i = 0; i < n32; ++i) { ce32[b32 + i] = e; cc32[b32 + i] = i * 32; }
        for (int i = 0; i < n64; ++i) { ce64[b64 + i] = e; cc64[b64 + i] = i * 64; }
        if (e == n_expert - 1) { n_chunks[0] = sh[n_expert + e]; n_chunks[1] = sh[2 * n_expert + e]; }
    }
}
__global__ void pf_scatter(const int32_t * __restrict__ ids, int64_t ids_stride, int k, int64_t T, int * __restrict__ cur,
                           int * __restrict__ rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * k) return;
    rows[atomicAdd(&cur[ids[(i / k) * ids_stride + i % k]], 1)] = (int) i;
}

// ---- tiles S and L: BM weight rows x BN routed rows, WM x WN warps, warp tile (BM/WM) x (BN/WN)
template <int QT, int BM, int BN> struct PfSm {
    static constexpr int XLD = PF_SB + 16;
    unsigned char w[2][BM][pf_qt<QT>::QB];
    int8_t        xq[2][BN][XLD];
    float         xd[2][8][BN], xs[2][8][BN];
    float         wsc[BM][8], wmn[BM][8];
    int           act[BN], row[BN];
};

template <int QT, int BM, int BN, int WM, int WN, int MINB>
__global__ void __launch_bounds__(WM * WN * 32, MINB)
pf_tile(const unsigned char * __restrict__ W, int64_t row_bytes, int64_t expert_bytes, int K, int N, int k_div,
        const int8_t * __restrict__ xq, const float * __restrict__ xd, const float * __restrict__ xs,
        const int * __restrict__ rows, const int * __restrict__ off, const int * __restrict__ cnt,
        const int * __restrict__ chunk_e, const int * __restrict__ chunk_c0, const int * __restrict__ n_chunks,
        float * __restrict__ out) {
    constexpr int QB = pf_qt<QT>::QB, NT = WM * WN * 32;
    constexpr int MT = BM / WM / 16, NTW = BN / WN / 8;
    using S_t = PfSm<QT, BM, BN>;
    constexpr int XLD = S_t::XLD;
    const int ch = blockIdx.y;
    if (ch >= *n_chunks) return;
    extern __shared__ __align__(16) unsigned char pf_smraw[];
    S_t & S = *reinterpret_cast<S_t *>(pf_smraw);
    const int NSB = K / PF_SB;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int rt = blockIdx.x;
    const int e = chunk_e[ch], c0 = chunk_c0[ch];
    const int ncol = min(BN, cnt[e] - c0);
    for (int i = tid; i < BN; i += NT) {
        const int r = i < ncol ? rows[off[e] + c0 + i] : -1;
        S.row[i] = r;
        S.act[i] = r < 0 ? -1 : r / k_div;
    }
    __syncthreads();
    const unsigned char * Wbase = W + (size_t) e * expert_bytes + (size_t) rt * BM * row_bytes;
    auto load = [&](int sb, int buf) {
        for (int i = tid; i < BM * (QB / 16); i += NT) {
            const int r = i / (QB / 16), c = i % (QB / 16);
            pf_cp16(&S.w[buf][r][c * 16], Wbase + (size_t) r * row_bytes + (size_t) sb * QB + c * 16);
        }
        for (int i = tid; i < BN * 16; i += NT) {
            const int n = i / 16, c = i % 16, a = S.act[n];
            if (a >= 0) pf_cp16(&S.xq[buf][n][c * 16], xq + (size_t) a * K + sb * PF_SB + c * 16);
        }
        for (int i = tid; i < BN * 8; i += NT) {
            const int n = i / 8, j = i % 8, a = S.act[n];
            if (a >= 0) {
                pf_cp4(&S.xd[buf][j][n], xd + (size_t) a * (K / 32) + sb * 8 + j);
                pf_cp4(&S.xs[buf][j][n], xs + (size_t) a * (K / 32) + sb * 8 + j);
            }
        }
        pf_commit();
    };
    for (int i = tid; i < 2 * BN * (XLD / 4); i += NT) {   // pad columns read zeros
        const int buf = i / (BN * XLD / 4), rem = i % (BN * XLD / 4), n = rem / (XLD / 4);
        if (S.act[n] < 0) reinterpret_cast<int *>(&S.xq[buf][n][0])[rem % (XLD / 4)] = 0;
    }
    for (int i = tid; i < 2 * BN * 8; i += NT) {
        const int buf = i / (BN * 8), rem = i % (BN * 8), n = rem % BN, j = rem / BN;
        if (S.act[n] < 0) { S.xd[buf][j][n] = 0.f; S.xs[buf][j][n] = 0.f; }
    }
    const int g = lane >> 2, tg = lane & 3;
    const int wm = warp % WM, wn = warp / WM;
    const int rbase = wm * (BM / WM), nbase = wn * (BN / WN);
    float acc[MT][NTW][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NTW; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][n][q] = 0.f;
    load(0, 0);
    for (int sb = 0; sb < NSB; ++sb) {
        const int buf = sb & 1;
        if (sb + 1 < NSB) { load(sb + 1, buf ^ 1); pf_wait<1>(); } else { pf_wait<0>(); }
        __syncthreads();
        for (int i = tid; i < BM * 8; i += NT) {
            const int r = i / 8, j = i % 8;
            const unsigned char * b = &S.w[buf][r][0];
            int sc, m;
            pf_scale_min(j, b + 4, sc, m);
            S.wsc[r][j] = __half2float(*reinterpret_cast<const __half *>(b + 0)) * sc;
            S.wmn[r][j] = __half2float(*reinterpret_cast<const __half *>(b + 2)) * m;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int a[MT][4];
            float sw[MT][2], mn[MT][2];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int r0 = rbase + m * 16;
                a[m][0] = pf_wq4<QT>(&S.w[buf][r0 + g][0], j, 4 * tg);
                a[m][1] = pf_wq4<QT>(&S.w[buf][r0 + g + 8][0], j, 4 * tg);
                a[m][2] = pf_wq4<QT>(&S.w[buf][r0 + g][0], j, 16 + 4 * tg);
                a[m][3] = pf_wq4<QT>(&S.w[buf][r0 + g + 8][0], j, 16 + 4 * tg);
                sw[m][0] = S.wsc[r0 + g][j]; sw[m][1] = S.wsc[r0 + g + 8][j];
                mn[m][0] = S.wmn[r0 + g][j]; mn[m][1] = S.wmn[r0 + g + 8][j];
            }
#pragma unroll
            for (int n = 0; n < NTW; ++n) {
                const int col = nbase + n * 8;
                int b[2];
                b[0] = *reinterpret_cast<const int *>(&S.xq[buf][col + g][j * 32 + 4 * tg]);
                b[1] = *reinterpret_cast<const int *>(&S.xq[buf][col + g][j * 32 + 16 + 4 * tg]);
                const float2 dd = *reinterpret_cast<const float2 *>(&S.xd[buf][j][col + 2 * tg]);
                const float2 ss = *reinterpret_cast<const float2 *>(&S.xs[buf][j][col + 2 * tg]);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    int c[4] = {0, 0, 0, 0};
                    pf_mma(c, a[m], b);
                    acc[m][n][0] += sw[m][0] * dd.x * (float) c[0] - mn[m][0] * ss.x;
                    acc[m][n][1] += sw[m][0] * dd.y * (float) c[1] - mn[m][0] * ss.y;
                    acc[m][n][2] += sw[m][1] * dd.x * (float) c[2] - mn[m][1] * ss.x;
                    acc[m][n][3] += sw[m][1] * dd.y * (float) c[3] - mn[m][1] * ss.y;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = rt * BM + rbase + m * 16;
#pragma unroll
        for (int n = 0; n < NTW; ++n) {
            const int na = nbase + n * 8 + 2 * tg, nb = na + 1;
            if (S.row[na] >= 0) { out[(size_t) S.row[na] * N + r0 + g] = acc[m][n][0]; out[(size_t) S.row[na] * N + r0 + g + 8] = acc[m][n][2]; }
            if (S.row[nb] >= 0) { out[(size_t) S.row[nb] * N + r0 + g] = acc[m][n][1]; out[(size_t) S.row[nb] * N + r0 + g + 8] = acc[m][n][3]; }
        }
    }
}

// ---- tile H (AI_MOE_PF_H): fp16 tensor cores, q4_K / q5_K decoded in registers.
// The int8 tiles are issue-bound in their per-32 epilogue (I2F + FMUL + 2 FFMA per output per 32 k). Here activations
// are fp16, permuted within each 32-block so one 32-bit word of codes feeds a thread's A fragment (Marlin-style lop3 to
// 1024 + q); every weight is dequantised as q * s - m in fp16 (scale and min inside the MMA: no epilogue); the MMA
// accumulates in fp16 (full rate on GeForce, where fp32 accumulation is half rate) and is flushed into fp32 every 128 k.
// One warp = 16 weight rows x 32 tokens, 8 warps = 128 rows; n-tiles holding only padding tokens issue no MMA.
// Error vs fp64 is ~1e-4 to 2e-4 of sum|terms|, below the int8 tiles' ~6e-4.
constexpr int PF_HB = 128;   // k per pipeline stage (half a superblock)
template <int QT> struct pf_hq;
template <> struct pf_hq<0> { static constexpr int R = 80, CH = 5, CS = 16; };    // header + 64 B codes
template <> struct pf_hq<1> { static constexpr int R = 112, CH = 7, CS = 48; };   // header + 32 B qh + 64 B codes

// logical position L of a 32-block holds physical element pf_perm32(L): thread (g, tg) of an m16n8k16 A fragment then
// reads codes 8 tg .. 8 tg + 7 of the block as two words (bytes 0, 2 -> k 2tg, 2tg+1; bytes 1, 3 -> k 2tg+8, 2tg+9)
__device__ __forceinline__ int pf_perm32(int L) {
    const int s = L >> 4, kk = L & 15;
    return kk < 8 ? 8 * (kk >> 1) + 4 * s + 2 * (kk & 1) : 8 * ((kk - 8) >> 1) + 4 * s + 1 + 2 * (kk & 1);
}
__global__ void pf_to_h(const float * __restrict__ x, int64_t xcs, int64_t K, int64_t rows, __half * __restrict__ xh) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * K) return;
    const int64_t r = i / K, c = i % K;
    xh[i] = __float2half_rn(fminf(fmaxf(x[r * xcs + (c & ~31ll) + pf_perm32((int) (c & 31))], -65504.f), 65504.f));
}
// SwiGLU (the same arithmetic as pf_swiglu_quant) writing act and the down input as permuted fp16; i is the logical slot
__global__ void pf_swiglu_h(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ act,
                            int64_t n, __half * __restrict__ ah) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t p = (i & ~31ll) + pf_perm32((int) (i & 31));
    const float gv = g[p];
    const float v = gv / (1.0f + __expf(-gv)) * u[p];
    act[p] = v;
    ah[i] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));   // saturate: fp16 range guard
}
__device__ __forceinline__ unsigned pf_lop3_ao(unsigned a, unsigned m, unsigned o) {   // (a & m) | o
    unsigned r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(m), "r"(o));
    return r;
}
__device__ __forceinline__ unsigned pf_dq_pair(unsigned v, __half2 s, __half2 nm) {   // bytes 0, 2 -> q * s - m
    const unsigned x = pf_lop3_ao(v, 0x001F001Fu, 0x64006400u);                       // half2(1024 + q0, 1024 + q2)
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&x), __half2half2(__ushort_as_half(0x6400)));
    const __half2 w = __hfma2(q, s, nm);
    return *reinterpret_cast<const unsigned *>(&w);
}
__device__ __forceinline__ void pf_mma_hh(unsigned (&c)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void pf_ldsm_x4(unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"((unsigned) __cvta_generic_to_shared(p)));
}
template <int QT> struct PfSmH {
    static constexpr int BM = 128, BN = 32, XLD = PF_HB + 8;   // 272 B token rows: ldmatrix conflict-free
    unsigned char w[2][BM][pf_hq<QT>::R];
    __half        x[2][BN][XLD];
    int           act[BN], row[BN];
};
template <int QT>
__global__ void __launch_bounds__(256, 2)
pf_tile_h(const unsigned char * __restrict__ W, int64_t row_bytes, int64_t expert_bytes, int K, int N, int k_div,
          const __half * __restrict__ xh, const int * __restrict__ rows, const int * __restrict__ off,
          const int * __restrict__ cnt, const int * __restrict__ chunk_e, const int * __restrict__ chunk_c0,
          const int * __restrict__ n_chunks, float * __restrict__ out) {
    using S_t = PfSmH<QT>;
    constexpr int BM = S_t::BM, BN = S_t::BN, NT = 256, QB = pf_qt<QT>::QB;
    constexpr int CH = pf_hq<QT>::CH, CS = pf_hq<QT>::CS, NTW = BN / 8, XLD = S_t::XLD, FL = 8;
    const int ch = blockIdx.y;
    if (ch >= *n_chunks) return;
    extern __shared__ __align__(16) unsigned char pf_smraw[];
    S_t & S = *reinterpret_cast<S_t *>(pf_smraw);
    const int NIT = 2 * (K / PF_SB);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int rt = blockIdx.x;
    const int e = chunk_e[ch], c0 = chunk_c0[ch];
    const int ncol = min(BN, cnt[e] - c0);
    for (int i = tid; i < BN; i += NT) {
        const int r = i < ncol ? rows[off[e] + c0 + i] : -1;
        S.row[i] = r;
        S.act[i] = r < 0 ? -1 : r / k_div;
    }
    __syncthreads();
    for (int i = tid; i < 2 * BN * (XLD / 2); i += NT) {   // padding tokens read zeros in both slots
        const int sl = i / (BN * XLD / 2), rem = i % (BN * XLD / 2), n = rem / (XLD / 2);
        if (S.act[n] < 0) reinterpret_cast<unsigned *>(&S.x[sl][n][0])[rem % (XLD / 2)] = 0u;
    }
    const unsigned char * Wbase = W + (size_t) e * expert_bytes + (size_t) rt * BM * row_bytes;
    auto load = [&](int it, int sl) {
        const int sb = it >> 1, h = it & 1;
        for (int i = tid; i < BM * CH; i += NT) {
            const int r = i / CH, c = i % CH;
            int so;
            if constexpr (QT == 1) so = c == 0 ? 0 : c < 3 ? 16 * c : 48 + 64 * h + 16 * (c - 3);
            else                   so = c == 0 ? 0 : 16 + 64 * h + 16 * (c - 1);
            pf_cp16(&S.w[sl][r][c * 16], Wbase + (size_t) r * row_bytes + (size_t) sb * QB + so);
        }
        for (int i = tid; i < BN * 16; i += NT) {
            const int n = i >> 4, c = i & 15, a = S.act[n];
            if (a >= 0) pf_cp16(&S.x[sl][n][c * 8], xh + (size_t) a * K + it * PF_HB + c * 8);
        }
        pf_commit();
    };
    const int g = lane >> 2, tg = lane & 3;
    const int r0 = warp * 16 + g;   // this thread's weight rows: r0, r0 + 8
    float acc[NTW][4];
    unsigned hacc[NTW][2];
#pragma unroll
    for (int n = 0; n < NTW; ++n) {
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[n][q] = 0.f;
        hacc[n][0] = hacc[n][1] = 0u;
    }
    const int bl_n = (lane >> 4) * 8 + (lane & 7), bl_k = ((lane >> 3) & 1) * 8;   // ldmatrix.x4 lane address for B
    load(0, 0);
    for (int it = 0; it < NIT; ++it) {
        const int sl = it & 1, h = it & 1;
        pf_wait<0>();
        __syncthreads();
        if (it + 1 < NIT) load(it + 1, sl ^ 1);
        unsigned sm_own[2];   // thread tg: sub-block 4h + tg of rows r0, r0 + 8 as half2(s, -m); the quad shuffles
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const unsigned char * hd = &S.w[sl][r0 + 8 * rr][0];
            int sc, m;
            pf_scale_min(4 * h + tg, hd + 4, sc, m);
            const float d = __half2float(*reinterpret_cast<const __half *>(hd));
            const float dm = __half2float(*reinterpret_cast<const __half *>(hd + 2));
            const __half2 p = __floats2half2_rn(d * (float) sc, -dm * (float) m);
            sm_own[rr] = *reinterpret_cast<const unsigned *>(&p);
        }
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            unsigned wv[2][2], hv[2][2];
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const unsigned char * rp = &S.w[sl][r0 + 8 * rr][0];
                const uint2 c2 = *reinterpret_cast<const uint2 *>(rp + CS + 32 * jp + 8 * tg);
                wv[rr][0] = c2.x; wv[rr][1] = c2.y;
                if constexpr (QT == 1) {
                    const uint2 h2 = *reinterpret_cast<const uint2 *>(rp + 16 + 8 * tg);
                    hv[rr][0] = h2.x; hv[rr][1] = h2.y;
                } else {
                    hv[rr][0] = hv[rr][1] = 0u;
                }
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int jl = 2 * jp + jj, j = 4 * h + jl;   // sub-block within the stage / within the superblock
                __half2 s2[2], m2[2];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                    const unsigned v = __shfl_sync(0xffffffffu, sm_own[rr], (lane & ~3) | jl);
                    const __half2 p = *reinterpret_cast<const __half2 *>(&v);
                    s2[rr] = __low2half2(p);
                    m2[rr] = __high2half2(p);
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    unsigned a[4];
#pragma unroll
                    for (int rr = 0; rr < 2; ++rr) {
                        unsigned b = jj ? (wv[rr][s] >> 4) & 0x0F0F0F0Fu : wv[rr][s] & 0x0F0F0F0Fu;
                        if constexpr (QT == 1) b |= ((hv[rr][s] >> j) & 0x01010101u) << 4;
                        a[rr]     = pf_dq_pair(b, s2[rr], m2[rr]);        // k 2tg, 2tg+1   <- bytes 0, 2
                        a[rr + 2] = pf_dq_pair(b >> 8, s2[rr], m2[rr]);   // k 2tg+8, 2tg+9 <- bytes 1, 3
                    }
                    const int kb = jl * 32 + s * 16;
#pragma unroll
                    for (int np = 0; np < NTW / 2; ++np) {
                        if (np * 16 >= ncol) break;   // warp-uniform
                        unsigned b0, b1, b2, b3;
                        pf_ldsm_x4(b0, b1, b2, b3, &S.x[sl][np * 16 + bl_n][kb + bl_k]);
                        pf_mma_hh(hacc[2 * np], a, b0, b1);
                        if (np * 16 + 8 < ncol) pf_mma_hh(hacc[2 * np + 1], a, b2, b3);
                    }
                    if (((jl * 2 + s + 1) % FL) == 0) {   // flush fp16 partials into fp32 every FL k16 steps
#pragma unroll
                        for (int n = 0; n < NTW; ++n) {
                            const float2 f0 = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[n][0]));
                            const float2 f1 = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[n][1]));
                            acc[n][0] += f0.x; acc[n][1] += f0.y; acc[n][2] += f1.x; acc[n][3] += f1.y;
                            hacc[n][0] = hacc[n][1] = 0u;
                        }
                    }
                }
            }
        }
    }
    const int ro = rt * BM + r0;
#pragma unroll
    for (int n = 0; n < NTW; ++n) {
        const int na = n * 8 + 2 * tg, nb = na + 1;
        if (S.row[na] >= 0) { out[(size_t) S.row[na] * N + ro] = acc[n][0]; out[(size_t) S.row[na] * N + ro + 8] = acc[n][2]; }
        if (S.row[nb] >= 0) { out[(size_t) S.row[nb] * N + ro] = acc[n][1]; out[(size_t) S.row[nb] * N + ro + 8] = acc[n][3]; }
    }
}

// ---- tile D: the slice's activations for all of K (<= PF_KD_MAX) resident; RPC weight row tiles streamed per CTA
template <int QT, int BM, int BN> struct PfSmD {
    static constexpr int XLD = PF_KD_MAX + 16;
    unsigned char w[2][BM][pf_qt<QT>::QB];
    int8_t        xq[BN][XLD];
    float         xd[PF_KD_MAX / 32][BN], xs[PF_KD_MAX / 32][BN];
    float         wsc[BM][8], wmn[BM][8];
    int           act[BN], row[BN];
};

template <int QT, int BM, int BN, int WM, int WN, int RPC>
__global__ void __launch_bounds__(WM * WN * 32, 1)
pf_tile_d(const unsigned char * __restrict__ W, int64_t row_bytes, int64_t expert_bytes, int K, int N, int k_div,
          const int8_t * __restrict__ xq, const float * __restrict__ xd, const float * __restrict__ xs,
          const int * __restrict__ rows, const int * __restrict__ off, const int * __restrict__ cnt,
          const int * __restrict__ chunk_e, const int * __restrict__ chunk_c0, const int * __restrict__ n_chunks,
          float * __restrict__ out) {
    constexpr int QB = pf_qt<QT>::QB, NT = WM * WN * 32;
    constexpr int MT = BM / WM / 16, NTW = BN / WN / 8;
    using S_t = PfSmD<QT, BM, BN>;
    const int ch = blockIdx.y;
    if (ch >= *n_chunks) return;
    extern __shared__ __align__(16) unsigned char pf_smraw[];
    S_t & S = *reinterpret_cast<S_t *>(pf_smraw);
    const int NSB = K / PF_SB, NB32 = K / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int rg = blockIdx.x;
    const int e = chunk_e[ch], c0 = chunk_c0[ch];
    const int ncol = min(BN, cnt[e] - c0);
    for (int i = tid; i < BN; i += NT) {
        const int r = i < ncol ? rows[off[e] + c0 + i] : -1;
        S.row[i] = r;
        S.act[i] = r < 0 ? -1 : r / k_div;
    }
    __syncthreads();
    for (int i = tid; i < BN * (K / 16); i += NT) {
        const int n = i / (K / 16), c = i % (K / 16), a = S.act[n];
        if (a >= 0) pf_cp16(&S.xq[n][c * 16], xq + (size_t) a * K + c * 16);
        else        *reinterpret_cast<int4 *>(&S.xq[n][c * 16]) = make_int4(0, 0, 0, 0);
    }
    for (int i = tid; i < BN * NB32; i += NT) {
        const int n = i / NB32, j = i % NB32, a = S.act[n];
        if (a >= 0) { pf_cp4(&S.xd[j][n], xd + (size_t) a * NB32 + j); pf_cp4(&S.xs[j][n], xs + (size_t) a * NB32 + j); }
        else        { S.xd[j][n] = 0.f; S.xs[j][n] = 0.f; }
    }
    pf_commit();
    const unsigned char * Wexp = W + (size_t) e * expert_bytes;
    const int nstage = RPC * NSB;
    auto load_w = [&](int s, int buf) {
        const int rt = rg * RPC + s / NSB, sb = s % NSB;
        const unsigned char * base = Wexp + (size_t) rt * BM * row_bytes;
        for (int i = tid; i < BM * (QB / 16); i += NT) {
            const int r = i / (QB / 16), c = i % (QB / 16);
            pf_cp16(&S.w[buf][r][c * 16], base + (size_t) r * row_bytes + (size_t) sb * QB + c * 16);
        }
        pf_commit();
    };
    const int g = lane >> 2, tg = lane & 3;
    const int wm = warp % WM, wn = warp / WM;
    const int rbase = wm * (BM / WM), nbase = wn * (BN / WN);
    float acc[MT][NTW][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NTW; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[m][n][q] = 0.f;
    load_w(0, 0);
    for (int s = 0; s < nstage; ++s) {
        const int buf = s & 1, sb = s % NSB;
        if (s + 1 < nstage) { load_w(s + 1, buf ^ 1); pf_wait<1>(); } else { pf_wait<0>(); }
        __syncthreads();
        for (int i = tid; i < BM * 8; i += NT) {
            const int r = i / 8, j = i % 8;
            const unsigned char * b = &S.w[buf][r][0];
            int sc, m;
            pf_scale_min(j, b + 4, sc, m);
            S.wsc[r][j] = __half2float(*reinterpret_cast<const __half *>(b + 0)) * sc;
            S.wmn[r][j] = __half2float(*reinterpret_cast<const __half *>(b + 2)) * m;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kb = sb * 8 + j;
            int a[MT][4];
            float sw[MT][2], mn[MT][2];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int r0 = rbase + m * 16;
                a[m][0] = pf_wq4<QT>(&S.w[buf][r0 + g][0], j, 4 * tg);
                a[m][1] = pf_wq4<QT>(&S.w[buf][r0 + g + 8][0], j, 4 * tg);
                a[m][2] = pf_wq4<QT>(&S.w[buf][r0 + g][0], j, 16 + 4 * tg);
                a[m][3] = pf_wq4<QT>(&S.w[buf][r0 + g + 8][0], j, 16 + 4 * tg);
                sw[m][0] = S.wsc[r0 + g][j]; sw[m][1] = S.wsc[r0 + g + 8][j];
                mn[m][0] = S.wmn[r0 + g][j]; mn[m][1] = S.wmn[r0 + g + 8][j];
            }
#pragma unroll
            for (int n = 0; n < NTW; ++n) {
                const int col = nbase + n * 8;
                int b[2];
                b[0] = *reinterpret_cast<const int *>(&S.xq[col + g][kb * 32 + 4 * tg]);
                b[1] = *reinterpret_cast<const int *>(&S.xq[col + g][kb * 32 + 16 + 4 * tg]);
                const float2 dd = *reinterpret_cast<const float2 *>(&S.xd[kb][col + 2 * tg]);
                const float2 ss = *reinterpret_cast<const float2 *>(&S.xs[kb][col + 2 * tg]);
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    int c[4] = {0, 0, 0, 0};
                    pf_mma(c, a[m], b);
                    acc[m][n][0] += sw[m][0] * dd.x * (float) c[0] - mn[m][0] * ss.x;
                    acc[m][n][1] += sw[m][0] * dd.y * (float) c[1] - mn[m][0] * ss.y;
                    acc[m][n][2] += sw[m][1] * dd.x * (float) c[2] - mn[m][1] * ss.x;
                    acc[m][n][3] += sw[m][1] * dd.y * (float) c[3] - mn[m][1] * ss.y;
                }
            }
        }
        if (sb == NSB - 1) {   // this row tile is complete: write it, reset the accumulators
            const int rt = rg * RPC + s / NSB;
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int r0 = rt * BM + rbase + m * 16;
#pragma unroll
                for (int n = 0; n < NTW; ++n) {
                    const int na = nbase + n * 8 + 2 * tg, nb = na + 1;
                    if (S.row[na] >= 0) { out[(size_t) S.row[na] * N + r0 + g] = acc[m][n][0]; out[(size_t) S.row[na] * N + r0 + g + 8] = acc[m][n][2]; }
                    if (S.row[nb] >= 0) { out[(size_t) S.row[nb] * N + r0 + g] = acc[m][n][1]; out[(size_t) S.row[nb] * N + r0 + g + 8] = acc[m][n][3]; }
#pragma unroll
                    for (int q = 0; q < 4; ++q) acc[m][n][q] = 0.f;
                }
            }
        }
        __syncthreads();
    }
}

struct PfArgs {
    const void * W; int64_t row_bytes, expert_bytes; int K, N, k_div;
    const int8_t * xq; const float * xd, * xs;
    const int * rows, * off, * cnt, * ce, * cc, * nch;
    int max_chunks; float * out;
};

template <int QT, int BM, int BN, int WM, int WN, int MINB>
void pf_launch_tile(cudaStream_t st, const PfArgs & a) {
    static bool init = false;
    const int smem = (int) sizeof(PfSm<QT, BM, BN>);
    if (!init) {
        cudaFuncSetAttribute(pf_tile<QT, BM, BN, WM, WN, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    pf_tile<QT, BM, BN, WM, WN, MINB><<<dim3(a.N / BM, a.max_chunks), WM * WN * 32, smem, st>>>(
        (const unsigned char *) a.W, a.row_bytes, a.expert_bytes, a.K, a.N, a.k_div, a.xq, a.xd, a.xs, a.rows, a.off,
        a.cnt, a.ce, a.cc, a.nch, a.out);
}
template <int QT>
void pf_launch_d(cudaStream_t st, const PfArgs & a) {
    static bool init = false;
    const int smem = (int) sizeof(PfSmD<QT, 128, 64>);
    if (!init) {
        cudaFuncSetAttribute(pf_tile_d<QT, 128, 64, 4, 2, PF_KD_RPC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    pf_tile_d<QT, 128, 64, 4, 2, PF_KD_RPC><<<dim3(a.N / 128 / PF_KD_RPC, a.max_chunks), 256, smem, st>>>(
        (const unsigned char *) a.W, a.row_bytes, a.expert_bytes, a.K, a.N, a.k_div, a.xq, a.xd, a.xs, a.rows, a.off,
        a.cnt, a.ce, a.cc, a.nch, a.out);
}

template <int QT>
void pf_launch_h(cudaStream_t st, const PfArgs & a, const __half * xh) {
    static bool init = false;
    const int smem = (int) sizeof(PfSmH<QT>);
    if (!init) {
        cudaFuncSetAttribute(pf_tile_h<QT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    pf_tile_h<QT><<<dim3(a.N / 128, a.max_chunks), 256, smem, st>>>(
        (const unsigned char *) a.W, a.row_bytes, a.expert_bytes, a.K, a.N, a.k_div, xh, a.rows, a.off, a.cnt, a.ce,
        a.cc, a.nch, a.out);
}

int qt_of(int ggml_type) { return ggml_type == GGML_Q4_K ? 0 : ggml_type == GGML_Q5_K ? 1 : -1; }

// tile choice. AI_MOE_PF_TILE: 0 auto (default), 1 force S, 2 force L (gate/up); down takes D when it fits, else L/S.
int pf_tile_env() {
    static const int v = getenv("AI_MOE_PF_TILE") ? atoi(getenv("AI_MOE_PF_TILE")) : 0;
    return v;
}
bool pf_h_env();
bool gu_large(int64_t T, int64_t k, int64_t n_expert) {
    const int f = pf_tile_env();
    if (f == 1) return false;
    if (f == 2) return true;
    // AI_MOE_PF_LMIN: mean routed rows per expert above which gate/up take tile L. Default 32 where tile H is off
    // (L then replaces S; S and L give bit-identical results), 64 where H is on, since H beats L for the smaller
    // per-expert row counts.
    static const int64_t lmin = getenv("AI_MOE_PF_LMIN") ? atoll(getenv("AI_MOE_PF_LMIN")) : pf_h_env() ? 64 : 32;
    return T * k > lmin * n_expert;   // > ~lmin routed rows per expert on average
}
// AI_MOE_PF_H: tile H (fp16, in-register decode) for the small-rows regime where tiles S / D run. 1 on, 0 off.
// Default: on for compute capability 8.x only, where it was validated (faster prefill, PPL neutral). Other
// architectures stay opt-in until measured there.
bool pf_h_env() {
    static const bool v = [] {
        if (const char * s = getenv("AI_MOE_PF_H")) return atoi(s) != 0;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
        return major == 8;
    }();
    return v;
}
bool use_h(int64_t T, int64_t k, int64_t n_expert, int64_t F, int64_t N_down) {
    return pf_h_env() && pf_tile_env() == 0 && !gu_large(T, k, n_expert) && F % 128 == 0 && N_down % 128 == 0;
}
bool down_d(int64_t K_down, int64_t N_down) {
    return pf_tile_env() == 0 && K_down <= PF_KD_MAX && N_down % (128 * PF_KD_RPC) == 0;
}

template <int QT>
void pf_gemm(cudaStream_t st, char tile, const PfArgs & a) {
    if (tile == 'D')      pf_launch_d<QT>(st, a);
    else if (tile == 'L') pf_launch_tile<QT, 128, 64, 4, 2, 1>(st, a);
    else                  pf_launch_tile<QT, 64, 32, 4, 2, 2>(st, a);
}
void pf_gemm_any(cudaStream_t st, int qt, char tile, const PfArgs & a) {
    if (qt == 0) pf_gemm<0>(st, tile, a); else pf_gemm<1>(st, tile, a);
}

// scratch layout (bytes, 256-aligned pieces)
struct PfLayout {
    size_t xq, xd, xs, xh, ah, cnt, off, cur, rows, ce32, cc32, ce64, cc64, nch, gbuf, ubuf, aq, ad, as, total;
    int64_t max32, max64;
};
size_t al256(size_t v) { return (v + 255) / 256 * 256; }
PfLayout pf_layout(int64_t K, int64_t F, int64_t k, int64_t n_expert, int64_t T) {
    PfLayout L{};
    size_t o = 0;
    const int64_t R = T * k;
    L.max32 = (R + 31) / 32 + n_expert;
    L.max64 = (R + 63) / 64 + n_expert;
    L.xq = o;   o += al256((size_t) T * K);
    L.xd = o;   o += al256((size_t) T * (K / 32) * 4);
    L.xs = o;   o += al256((size_t) T * (K / 32) * 4);
    L.xh = o;   o += al256((size_t) T * K * 2);
    L.ah = o;   o += al256((size_t) R * F * 2);
    L.cnt = o;  o += al256((size_t) n_expert * 4);
    L.off = o;  o += al256((size_t) n_expert * 4);
    L.cur = o;  o += al256((size_t) n_expert * 4);
    L.rows = o; o += al256((size_t) R * 4);
    L.ce32 = o; o += al256((size_t) L.max32 * 4);
    L.cc32 = o; o += al256((size_t) L.max32 * 4);
    L.ce64 = o; o += al256((size_t) L.max64 * 4);
    L.cc64 = o; o += al256((size_t) L.max64 * 4);
    L.nch = o;  o += 256;
    L.gbuf = o; o += al256((size_t) R * F * 4);
    L.ubuf = o; o += al256((size_t) R * F * 4);
    L.aq = o;   o += al256((size_t) R * F);
    L.ad = o;   o += al256((size_t) R * (F / 32) * 4);
    L.as = o;   o += al256((size_t) R * (F / 32) * 4);
    L.total = o;
    return L;
}

}  // namespace

bool moe_pf_ok(int gu_type, int64_t K, int64_t F, int64_t n_used, int64_t n_expert, int64_t T) {
    return qt_of(gu_type) >= 0 && K % PF_SB == 0 && F % 128 == 0 && F % PF_SB == 0 && n_used >= 1 &&
           n_expert >= 1 && n_expert <= 1024 && T >= 1;
}
bool moe_pf_down_ok(int down_type, int64_t K_down, int64_t N_down) {
    return qt_of(down_type) >= 0 && K_down % PF_SB == 0 && N_down % 128 == 0;
}
size_t moe_pf_scratch_bytes(int64_t K, int64_t F, int64_t n_used, int64_t n_expert, int64_t T) {
    return pf_layout(K, F, n_used, n_expert, T).total + 256;
}

bool moe_pf_gu(void * stream, int type, const void * up, const void * gate, int64_t row_bytes, int64_t expert_bytes,
               const float * x, int64_t x_col_stride, const int32_t * ids, int64_t ids_stride, float * act,
               int64_t K, int64_t F, int64_t n_used, int64_t n_expert, int64_t T, void * scratch) {
    if (!moe_pf_ok(type, K, F, n_used, n_expert, T)) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const PfLayout L = pf_layout(K, F, n_used, n_expert, T);
    char * base = (char *) (((uintptr_t) scratch + 255) / 256 * 256);
    int8_t * xq = (int8_t *) (base + L.xq);
    float * xd = (float *) (base + L.xd), * xs = (float *) (base + L.xs);
    int * cnt = (int *) (base + L.cnt), * off = (int *) (base + L.off), * cur = (int *) (base + L.cur);
    int * rows = (int *) (base + L.rows), * nch = (int *) (base + L.nch);
    int * ce32 = (int *) (base + L.ce32), * cc32 = (int *) (base + L.cc32), * ce64 = (int *) (base + L.ce64), * cc64 = (int *) (base + L.cc64);
    float * gbuf = (float *) (base + L.gbuf), * ubuf = (float *) (base + L.ubuf);
    int8_t * aq = (int8_t *) (base + L.aq);
    float * ad = (float *) (base + L.ad), * as = (float *) (base + L.as);
    const int64_t R = T * n_used;

    cudaMemsetAsync(cnt, 0, (size_t) n_expert * 4, st);
    pf_count<<<(unsigned) ((R + 255) / 256), 256, 0, st>>>(ids, ids_stride, (int) n_used, T, cnt);
    pf_table<<<1, (unsigned) n_expert, 3 * n_expert * sizeof(int), st>>>(cnt, (int) n_expert, off, cur, ce32, cc32, ce64, cc64, nch);
    pf_scatter<<<(unsigned) ((R + 255) / 256), 256, 0, st>>>(ids, ids_stride, (int) n_used, T, cur, rows);
    const bool large = gu_large(T, n_used, n_expert);
    if (use_h(T, n_used, n_expert, F, K)) {   // down is F -> K
        __half * xh = (__half *) (base + L.xh), * ah = (__half *) (base + L.ah);
        pf_to_h<<<(unsigned) ((T * K + 255) / 256), 256, 0, st>>>(x, x_col_stride, K, T, xh);
        PfArgs a{gate, row_bytes, expert_bytes, (int) K, (int) F, (int) n_used, nullptr, nullptr, nullptr, rows, off,
                 cnt, ce32, cc32, nch, (int) L.max32, gbuf};
        const int q = qt_of(type);
        if (q == 0) pf_launch_h<0>(st, a, xh); else pf_launch_h<1>(st, a, xh);
        a.W   = up;
        a.out = ubuf;
        if (q == 0) pf_launch_h<0>(st, a, xh); else pf_launch_h<1>(st, a, xh);
        pf_swiglu_h<<<(unsigned) ((R * F + 255) / 256), 256, 0, st>>>(gbuf, ubuf, act, R * F, ah);
        return cudaGetLastError() == cudaSuccess;
    }
    pf_quant<<<(unsigned) ((T * (K / 32) * 32 + 255) / 256), 256, 0, st>>>(x, x_col_stride, K, T, xq, xd, xs);
    PfArgs a{gate, row_bytes, expert_bytes, (int) K, (int) F, (int) n_used, xq, xd, xs, rows, off, cnt,
             large ? ce64 : ce32, large ? cc64 : cc32, large ? nch + 1 : nch, (int) (large ? L.max64 : L.max32), gbuf};
    pf_gemm_any(st, qt_of(type), large ? 'L' : 'S', a);
    a.W   = up;
    a.out = ubuf;
    pf_gemm_any(st, qt_of(type), large ? 'L' : 'S', a);
    pf_swiglu_quant<<<(unsigned) ((R * (F / 32) * 32 + 255) / 256), 256, 0, st>>>(gbuf, ubuf, act, F, R, aq, ad, as);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_pf_down(void * stream, int type, const void * down, int64_t row_bytes, int64_t expert_bytes, float * out,
                 int64_t K, int64_t F, int64_t N_down, int64_t n_used, int64_t n_expert, int64_t T, void * scratch) {
    if (!moe_pf_down_ok(type, F, N_down)) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const PfLayout L = pf_layout(K, F, n_used, n_expert, T);   // the same scratch moe_pf_gu filled
    char * base = (char *) (((uintptr_t) scratch + 255) / 256 * 256);
    int * cnt = (int *) (base + L.cnt), * off = (int *) (base + L.off), * rows = (int *) (base + L.rows), * nch = (int *) (base + L.nch);
    int * ce32 = (int *) (base + L.ce32), * cc32 = (int *) (base + L.cc32), * ce64 = (int *) (base + L.ce64), * cc64 = (int *) (base + L.cc64);
    int8_t * aq = (int8_t *) (base + L.aq);
    float * ad = (float *) (base + L.ad), * as = (float *) (base + L.as);
    if (use_h(T, n_used, n_expert, F, N_down)) {   // moe_pf_gu took tile H and left the permuted fp16 input in ah
        PfArgs a{down, row_bytes, expert_bytes, (int) F, (int) N_down, 1, nullptr, nullptr, nullptr, rows, off, cnt,
                 ce32, cc32, nch, (int) L.max32, out};
        const __half * ah = (const __half *) (base + L.ah);
        if (qt_of(type) == 0) pf_launch_h<0>(st, a, ah); else pf_launch_h<1>(st, a, ah);
        return cudaGetLastError() == cudaSuccess;
    }
    const bool d = down_d(F, N_down);
    const bool large = d || gu_large(T, n_used, n_expert);
    PfArgs a{down, row_bytes, expert_bytes, (int) F, (int) N_down, 1, aq, ad, as, rows, off, cnt,
             large ? ce64 : ce32, large ? cc64 : cc32, large ? nch + 1 : nch, (int) (large ? L.max64 : L.max32), out};
    pf_gemm_any(st, qt_of(type), d ? 'D' : large ? 'L' : 'S', a);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace kern
}  // namespace ai
