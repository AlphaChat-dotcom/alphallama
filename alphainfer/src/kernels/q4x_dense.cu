// SPDX-License-Identifier: Apache-2.0
// Dense small-M GEMV on fp16 tensor cores with fp16 activations: out[t][n] = sum_k W[n][k] x[t][k], T <= 32.
//
// ggml's MMVQ quantises the activations to q8_1 and runs dp4a per column: on the qwen4exp dense projections
// (q6_K / q5_K / q4_K / iq4_xs / iq3_s) it reaches well under half of DRAM bandwidth at T = 3 and goes ALU-bound
// above ~4 columns. Here each warp owns 16 weight rows over all of K; every (row, superblock) is decoded once,
// straight from 32-bit words into fp16 A fragments of mma.m16n8k16 (f32 accumulate), and up to 8 columns ride one
// mma (NT n-tiles for T <= 8 * NT): the cost is flat in T until the n-tiles add up.
//
// Layout: a thread (g, tg) of an m16n8k16 A fragment holds k = 2tg, 2tg+1, 2tg+8, 2tg+9 of each 16-wide chunk. Per
// type, the 256 elements of a superblock are assigned to (chunk, position) so that a thread's elements come from a
// few whole words of the block (DQ<type>::pos). The activations are permuted the same way once per call
// (dense_perm_k), so the mma's K order is a permutation of the true K -- the sum is the same dot product.
//
// Numerics: weights: q exact in fp16 (magic 1024 + q), times the block's fp16 scale (one rounding), minus the fp16 min
// (K-quants); activations: fp16 of x / colmax(|x|) per column (no overflow; the column max is re-applied at the end);
// fp32 accumulation in the mma. Quality-neutral by construction vs q8_1 activations; gated by paired NLL. Every column
// runs the same instructions wherever it sits, the same kernel serves every T: MTP == plain of the same build.
#include "kernels/kernels.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>

namespace {

__device__ __forceinline__ void hmma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    NO_DEVICE_CODE;
#endif
}

__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned *>(&h); }
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2 *>(&u); }
// bytes k of lo and hi (each <= 0xFF) -> half2(1024 + lo_k, 1024 + hi_k) bit patterns
__device__ __forceinline__ unsigned pair_bits(unsigned lo, unsigned hi, int k) {
    return (__byte_perm(lo, hi, (unsigned) k | ((unsigned) (4 + k) << 8)) & 0x00FF00FFu) | 0x64006400u;   // result bytes 0, 2
}
__device__ __forceinline__ int ld4(const uint8_t * p, int byte_off, bool al4) {   // 32-bit word at a 2- or 4-aligned offset
    return al4 ? *reinterpret_cast<const int *>(p + byte_off)
               : (int) ((unsigned) *reinterpret_cast<const uint16_t *>(p + byte_off) |
                        ((unsigned) *reinterpret_cast<const uint16_t *>(p + byte_off + 2) << 16));
}

// DQ<type>: SB bytes per 256-element superblock; pos(c, p) = the element of the superblock that sits at chunk c,
// position p (the activation permutation); decode(blk, tg, A) fills A[c][0..1] = the thread's two fp16 pairs of
// chunk c (logical 2tg,2tg+1 and 2tg+8,2tg+9) for one row.
template <int TY> struct DQ;

template <> struct DQ<GGML_TYPE_Q6_K> {   // ql[128] qh[64] scales[16] (int8) d
    static constexpr int SB = 210;
    static __host__ __device__ int pos(int c, int p) {
        const int tg = (p & 7) >> 1, j = p & 1, hi = p >> 3, n = c >> 3, l = 8 * tg + (c & 7);
        return 128 * n + l + 32 * hi + 64 * j;
    }
    static __device__ __forceinline__ void decode(const uint8_t * b, int tg, unsigned (&A)[16][2]) {
        const float    d  = __half2float(*reinterpret_cast<const __half *>(b + 208));
        const int8_t * sc = reinterpret_cast<const int8_t *>(b + 192);
        const __half2  off = __float2half2_rn(1056.0f);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int     s0 = 8 * n + (tg >> 1);
            const __half2 sA = __floats2half2_rn(d * sc[s0], d * sc[s0 + 4]);       // P0: elements +0, +64
            const __half2 sB = __floats2half2_rn(d * sc[s0 + 2], d * sc[s0 + 6]);   // P1: elements +32, +96
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const int      l0 = 8 * tg + 4 * w;
                const unsigned q1 = (unsigned) ld4(b, 64 * n + l0, false), q2 = (unsigned) ld4(b, 64 * n + 32 + l0, false);
                const unsigned h  = (unsigned) ld4(b, 128 + 32 * n + l0, false);
                const unsigned lo  = (q1 & 0x0F0F0F0Fu) | ((h & 0x03030303u) << 4);
                const unsigned hi  = ((q1 >> 4) & 0x0F0F0F0Fu) | (h & 0x30303030u);
                const unsigned lo2 = (q2 & 0x0F0F0F0Fu) | ((h & 0x0C0C0C0Cu) << 2);
                const unsigned hi2 = ((q2 >> 4) & 0x0F0F0F0Fu) | ((h & 0xC0C0C0C0u) >> 2);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int c = 8 * n + 4 * w + k;
                    A[c][0] = h2u(__hmul2(__hsub2(u2h(pair_bits(lo, hi, k)), off), sA));
                    A[c][1] = h2u(__hmul2(__hsub2(u2h(pair_bits(lo2, hi2, k)), off), sB));
                }
            }
        }
    }
};

// Q4_K / Q5_K: d dmin scales[12] [qh[32]] qs[128]; sub-block s of 32 has scale d*sc_s, min dmin*m_s
__device__ __forceinline__ void kq_scale_min(int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else       { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template <bool Q5> struct DQK {
    static constexpr int SB = Q5 ? 176 : 144, QS = Q5 ? 48 : 16;
    static __host__ __device__ int pos(int c, int p) {
        const int tg = (p & 7) >> 1, j = p & 1, hi = p >> 3;
        return 64 * (c >> 2) + 8 * tg + 2 * (c & 3) + hi + 32 * j;
    }
    static __device__ __forceinline__ void decode(const uint8_t * b, int tg, unsigned (&A)[16][2]) {
        const float   d = __half2float(*reinterpret_cast<const __half *>(b)), dm = __half2float(*reinterpret_cast<const __half *>(b + 2));
        const __half2 off = __float2half2_rn(1024.0f);
        unsigned      qh[2] = { 0, 0 };
        if (Q5) { qh[0] = (unsigned) ld4(b, 16 + 8 * tg, true); qh[1] = (unsigned) ld4(b, 16 + 8 * tg + 4, true); }
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            int sc0, m0, sc1, m1;
            kq_scale_min(2 * jj, b + 4, sc0, m0);
            kq_scale_min(2 * jj + 1, b + 4, sc1, m1);
            const __half2 s = __floats2half2_rn(d * sc0, d * sc1), nm = __floats2half2_rn(-dm * m0, -dm * m1);
#pragma unroll
            for (int w = 0; w < 2; ++w) {
                const unsigned q  = (unsigned) ld4(b, QS + 32 * jj + 8 * tg + 4 * w, true);
                unsigned       lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
                if (Q5) {
                    lo |= ((qh[w] >> (2 * jj)) & 0x01010101u) << 4;
                    hi |= ((qh[w] >> (2 * jj + 1)) & 0x01010101u) << 4;
                }
#pragma unroll
                for (int k2 = 0; k2 < 2; ++k2) {   // chunk (c & 3) = 2w + k2 takes bytes 2 k2, 2 k2 + 1 of this word
                    const int c = 4 * jj + 2 * w + k2;
                    A[c][0] = h2u(__hfma2(__hsub2(u2h(pair_bits(lo, hi, 2 * k2)), off), s, nm));
                    A[c][1] = h2u(__hfma2(__hsub2(u2h(pair_bits(lo, hi, 2 * k2 + 1)), off), s, nm));
                }
            }
        }
    }
};
template <> struct DQ<GGML_TYPE_Q4_K> : DQK<false> {};
template <> struct DQ<GGML_TYPE_Q5_K> : DQK<true> {};

template <> struct DQ<GGML_TYPE_IQ4_XS> {   // d scales_h scales_l[4] qs[128]
    static constexpr int SB = 136;
    static __host__ __device__ int pos(int c, int p) {
        const int tg = (p & 7) >> 1, j = p & 1, hi = p >> 3;
        return 32 * (c >> 1) + 4 * tg + 2 * (c & 1) + hi + 16 * j;
    }
    static __device__ __forceinline__ void decode(const uint8_t * b, int tg, unsigned (&A)[16][2]) {
        const float    d  = __half2float(*reinterpret_cast<const __half *>(b));
        const unsigned sh = *reinterpret_cast<const uint16_t *>(b + 2);
        const __half2  off = __float2half2_rn(1152.0f);
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int     ls = ((b[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const __half2 s  = __float2half2_rn(d * (float) (ls - 32));
            const int2    v  = get_int_from_table_16(ld4(b, 8 + 16 * ib + 4 * tg, true), kvalues_iq4nl);
            const unsigned lo = (unsigned) v.x ^ 0x80808080u, hi = (unsigned) v.y ^ 0x80808080u;   // int8 -> u8 + 128
#pragma unroll
            for (int k2 = 0; k2 < 2; ++k2) {
                const int c = 2 * ib + k2;
                A[c][0] = h2u(__hmul2(__hsub2(u2h(pair_bits(lo, hi, 2 * k2)), off), s));
                A[c][1] = h2u(__hmul2(__hsub2(u2h(pair_bits(lo, hi, 2 * k2 + 1)), off), s));
            }
        }
    }
};

template <> struct DQ<GGML_TYPE_IQ3_S> {   // d qs[64] qh[8] signs[32] scales[4]
    static constexpr int SB = 110;
    static __host__ __device__ int pos(int c, int p) {
        const int tg = (p & 7) >> 1, j = p & 1, hi = p >> 3;
        return 32 * (c >> 1) + 4 * (4 * (c & 1) + tg) + 2 * hi + j;
    }
    static __device__ __forceinline__ void decode(const uint8_t * b, int tg, unsigned (&A)[16][2]) {
        const float   d   = __half2float(*reinterpret_cast<const __half *>(b));
        const __half2 off = __float2half2_rn(1152.0f);
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const __half2 s  = __float2half2_rn(d * (float) (1 + 2 * ((b[106 + ib / 2] >> (4 * (ib & 1))) & 0xF)));
            const int     qh = b[66 + ib];
#pragma unroll
            for (int k2 = 0; k2 < 2; ++k2) {
                const int i   = 4 * k2 + tg;
                const int idx = b[2 + 8 * ib + i] | ((qh << (8 - i)) & 0x100);
                const int s8  = b[74 + 4 * ib + i / 2];
                const int sg  = (i & 1) ? __vcmpne4(((s8 & 0x30) << 3) | ((s8 & 0xC0) << 17), 0)
                                        : __vcmpne4(((s8 & 0x03) << 7) | ((s8 & 0x0C) << 21), 0);
                const unsigned v = (unsigned) __vsub4((int) iq3s_grid[idx] ^ sg, sg) ^ 0x80808080u;
                const int      c = 2 * ib + k2;
                A[c][0] = h2u(__hmul2(__hsub2(u2h(__byte_perm(v, 0, 0x4140) | 0x64006400u), off), s));
                A[c][1] = h2u(__hmul2(__hsub2(u2h(__byte_perm(v, 0, 0x4342) | 0x64006400u), off), s));
            }
        }
    }
};

// x f32 [T][K] (column t at x + t * xs) -> xp fp16 [T][K] in DQ<TY>::pos order per superblock, scaled by 1 / colmax;
// cm[t] = colmax. grid T, block 256.
template <int TY>
__global__ void dense_perm_k(const float * __restrict__ x, long long xs, int K, __half * __restrict__ xp, float * __restrict__ cm) {
    const int t = blockIdx.x;
    const float * xr = x + (long long) t * xs;
    __shared__ float red[8];
    float mx = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(xr[k]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    __syncthreads();
    mx = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) mx = fmaxf(mx, red[i]);
    const float sc = mx > 0.0f ? 1.0f / mx : 0.0f;
    if (threadIdx.x == 0) cm[t] = mx;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const int sb = k >> 8, r = k & 255, c = r >> 4, p = r & 15;
        xp[(long long) t * K + k] = __float2half_rn(xr[(sb << 8) + DQ<TY>::pos(c, p)] * sc);
    }
}

__device__ __forceinline__ void dn_cp4(void * s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}

// grid N / 16, block 32 * W: the W warps of a block share 16 weight rows and split K into W contiguous ranges of
// superblock pairs. Per pair, a warp stages its 16 rows x 2 superblocks (one 4-byte-aligned contiguous run per row)
// into shared memory with coalesced cp.async and decodes from there: decoding straight from global memory issues
// 2-byte loads that touch 8 rows each (8 L1 wavefronts per load), well under half of DRAM bandwidth.
// Partial sums meet in shared memory and warp 0 adds them in warp order (fixed: every T, every column position, the
// same bits). NT n-tiles of 8 columns.
template <int TY, int W, int NT>
__global__ void __launch_bounds__(32 * W) dense_k(const uint8_t * __restrict__ Wq, long long row_bytes, int N, int K,
                                                  const __half * __restrict__ xp, const float * __restrict__ cm, int T,
                                                  float * __restrict__ out, long long ldo) {
    using Q = DQ<TY>;
    constexpr int PB = 2 * Q::SB, PW = PB / 4;   // bytes / words per row per superblock pair
    __shared__ __align__(16) uint8_t stg[W][16 * PB];
    __shared__ float red[W][NT][4][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.x * 16;
    const uint8_t * rows = Wq + (size_t) r0 * row_bytes;
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    const __half * xc[NT];
    bool           cv[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) { const int col = 8 * n + g; cv[n] = col < T; xc[n] = xp + (size_t) (cv[n] ? col : 0) * K; }
    const int np = K >> 9, per = (np + W - 1) / W, p0 = warp * per, p1 = min(np, p0 + per);
    uint8_t * sm = stg[warp];
#pragma unroll 1
    for (int pr = p0; pr < p1; ++pr) {
        for (int i = lane; i < 16 * PW; i += 32) {
            const int r = i / PW, w = i - r * PW;
            dn_cp4(sm + r * PB + 4 * w, rows + (size_t) r * row_bytes + (size_t) pr * PB + 4 * w);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncwarp();
#pragma unroll 1
        for (int h = 0; h < 2; ++h) {
            const int sb = 2 * pr + h;
            unsigned  Aa[16][2], Ab[16][2];
            Q::decode(sm + g * PB + h * Q::SB, tg, Aa);
            Q::decode(sm + (g + 8) * PB + h * Q::SB, tg, Ab);
#pragma unroll
            for (int c = 0; c < 16; ++c) {
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const __half * xb = xc[n] + (sb << 8) + 16 * c;
                    const unsigned b0 = cv[n] ? *reinterpret_cast<const unsigned *>(xb + 2 * tg) : 0u;
                    const unsigned b1 = cv[n] ? *reinterpret_cast<const unsigned *>(xb + 2 * tg + 8) : 0u;
                    hmma(acc[n], Aa[c][0], Ab[c][0], Aa[c][1], Ab[c][1], b0, b1);
                }
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 4; ++q) red[warp][n][q][lane] = acc[n][q];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = red[0][n][q][lane];
#pragma unroll
            for (int w = 1; w < W; ++w) v += red[w][n][q][lane];
            acc[n][q] = v;
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int col = 8 * n + 2 * tg + q;
            if (col < T) {
                const float m = cm[col];
                out[(long long) col * ldo + r0 + g]     = acc[n][q] * m;
                out[(long long) col * ldo + r0 + g + 8] = acc[n][2 + q] * m;
            }
        }
    }
}


// AI_Q4X_DENSE_PIPE=1: the same split-K tile with a two-stage pipeline per warp: one superblock per stage
// (its 4-byte-aligned superset: block types of 2 mod 4 bytes start 2 bytes in), the next superblock's cp.async in
// flight while this one decodes. Same per-column arithmetic and reduction order as dense_k: identical bits.
template <int TY, int W, int NT>
__global__ void __launch_bounds__(32 * W) dense_p_k(const uint8_t * __restrict__ Wq, long long row_bytes, int N, int K,
                                                    const __half * __restrict__ xp, const float * __restrict__ cm, int T,
                                                    float * __restrict__ out, long long ldo) {
    using Q = DQ<TY>;
    constexpr int RS = (Q::SB + 2 + 3) / 4 * 4, RW = RS / 4;   // smem bytes / words per row per stage
    __shared__ __align__(16) uint8_t stg[W][2][16 * RS];
    __shared__ float red[W][NT][4][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const int r0 = blockIdx.x * 16;
    const uint8_t * rows = Wq + (size_t) r0 * row_bytes;
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    const __half * xc[NT];
    bool           cv[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) { const int col = 8 * n + g; cv[n] = col < T; xc[n] = xp + (size_t) (cv[n] ? col : 0) * K; }
    // the same superblock partition as dense_k's pairs: warp w owns pairs [w * per, ...) = superblocks 2 x that
    const int np = K >> 9, per = (np + W - 1) / W, s0 = 2 * (warp * per), s1 = 2 * min(np, warp * per + per);
    auto issue = [&](int sb, int buf) {
        const int off = sb * Q::SB, al = off & ~3, nw = (off - al + Q::SB + 3) / 4;
        uint8_t * dst = stg[warp][buf];
        for (int i = lane; i < 16 * nw; i += 32) {
            const int r = i / nw, w = i - r * nw;
            dn_cp4(dst + r * RS + 4 * w, rows + (size_t) r * row_bytes + al + 4 * w);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    if (s0 < s1) issue(s0, 0);
#pragma unroll 1
    for (int sb = s0; sb < s1; ++sb) {
        if (sb + 1 < s1) { issue(sb + 1, (sb + 1) & 1); asm volatile("cp.async.wait_group 1;\n" ::); }
        else             { asm volatile("cp.async.wait_group 0;\n" ::); }
        __syncwarp();
        const uint8_t * buf = stg[warp][sb & 1];
        const int       sh  = (sb * Q::SB) & 3;
        unsigned        Aa[16][2], Ab[16][2];
        Q::decode(buf + g * RS + sh, tg, Aa);
        Q::decode(buf + (g + 8) * RS + sh, tg, Ab);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const __half * xb = xc[n] + (sb << 8) + 16 * c;
                const unsigned b0 = cv[n] ? *reinterpret_cast<const unsigned *>(xb + 2 * tg) : 0u;
                const unsigned b1 = cv[n] ? *reinterpret_cast<const unsigned *>(xb + 2 * tg + 8) : 0u;
                hmma(acc[n], Aa[c][0], Ab[c][0], Aa[c][1], Ab[c][1], b0, b1);
            }
        }
        __syncwarp();
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 4; ++q) red[warp][n][q][lane] = acc[n][q];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = red[0][n][q][lane];
#pragma unroll
            for (int w = 1; w < W; ++w) v += red[w][n][q][lane];
            acc[n][q] = v;
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int col = 8 * n + 2 * tg + q;
            if (col < T) {
                const float m = cm[col];
                out[(long long) col * ldo + r0 + g]     = acc[n][q] * m;
                out[(long long) col * ldo + r0 + g + 8] = acc[n][2 + q] * m;
            }
        }
    }
}

template <int TY>
void dense_launch(cudaStream_t st, const void * w, long long rb, int N, int K, const float * x, long long xs, int T,
                  __half * xp, float * cm, float * out, long long ldo) {
    constexpr int W = 4;
    dense_perm_k<TY><<<T, 256, 0, st>>>(x, xs, K, xp, cm);
    const dim3 grid((unsigned) (N / 16));
    // AI_Q4X_DENSE_PIPE=1: the two-stage pipeline. Off by default: no gain on mid-size matrices and slightly slower
    // on the output head.
    static const bool pipe = getenv("AI_Q4X_DENSE_PIPE") && atoi(getenv("AI_Q4X_DENSE_PIPE")) != 0;
    if (pipe) {
        if (T <= 8)       dense_p_k<TY, W, 1><<<grid, 32 * W, 0, st>>>((const uint8_t *) w, rb, N, K, xp, cm, T, out, ldo);
        else if (T <= 16) dense_p_k<TY, W, 2><<<grid, 32 * W, 0, st>>>((const uint8_t *) w, rb, N, K, xp, cm, T, out, ldo);
        else              dense_p_k<TY, W, 4><<<grid, 32 * W, 0, st>>>((const uint8_t *) w, rb, N, K, xp, cm, T, out, ldo);
        return;
    }
    if (T <= 8)       dense_k<TY, W, 1><<<grid, 32 * W, 0, st>>>((const uint8_t *) w, rb, N, K, xp, cm, T, out, ldo);
    else if (T <= 16) dense_k<TY, W, 2><<<grid, 32 * W, 0, st>>>((const uint8_t *) w, rb, N, K, xp, cm, T, out, ldo);
    else              dense_k<TY, W, 4><<<grid, 32 * W, 0, st>>>((const uint8_t *) w, rb, N, K, xp, cm, T, out, ldo);
}

}  // namespace

namespace ai::kern {

bool q4x_dense_ok(int type, int64_t N, int64_t K, int64_t T) {
    const bool ty = type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q4_K ||
                    type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ3_S;
    return ty && N % 16 == 0 && K % 512 == 0 && T >= 1 && T <= 32;   // superblock pairs; 4-byte-aligned rows
}

size_t q4x_dense_scratch_bytes(int64_t K, int64_t T) { return ((size_t) K * T * 2 + 255) / 256 * 256 + 256 + (size_t) T * 4; }

bool q4x_dense(void * stream, int type, const void * w, int64_t row_bytes, int64_t N, int64_t K, const float * x,
               int64_t x_stride, int64_t T, void * scratch, float * out, int64_t ldo) {
    if (!q4x_dense_ok(type, N, K, T) || row_bytes % 4 || (uintptr_t) w % 4) return false;
    cudaStream_t st = (cudaStream_t) stream;
    __half *     xp = (__half *) scratch;
    float *      cm = (float *) ((char *) scratch + ((size_t) K * T * 2 + 255) / 256 * 256);
    const int    Ni = (int) N, Ki = (int) K, Ti = (int) T;
    switch (type) {
        case GGML_TYPE_Q6_K:   dense_launch<GGML_TYPE_Q6_K>(st, w, row_bytes, Ni, Ki, x, x_stride, Ti, xp, cm, out, ldo); break;
        case GGML_TYPE_Q5_K:   dense_launch<GGML_TYPE_Q5_K>(st, w, row_bytes, Ni, Ki, x, x_stride, Ti, xp, cm, out, ldo); break;
        case GGML_TYPE_Q4_K:   dense_launch<GGML_TYPE_Q4_K>(st, w, row_bytes, Ni, Ki, x, x_stride, Ti, xp, cm, out, ldo); break;
        case GGML_TYPE_IQ4_XS: dense_launch<GGML_TYPE_IQ4_XS>(st, w, row_bytes, Ni, Ki, x, x_stride, Ti, xp, cm, out, ldo); break;
        default:               dense_launch<GGML_TYPE_IQ3_S>(st, w, row_bytes, Ni, Ki, x, x_stride, Ti, xp, cm, out, ldo); break;
    }
    return true;
}

}  // namespace ai::kern
