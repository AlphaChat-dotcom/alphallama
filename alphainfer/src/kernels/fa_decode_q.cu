// Tiled split-K decode attention over a q8_0 or q4_0 KV cache (AI_FA_Q).
//
// ggml's flash_attn_ext_vec reads a quantised cache far below its f16 bandwidth, so quantised KV reads fewer bytes
// but takes more time; for long contexts where an f16 cache does not fit, that makes the quantised path the slow
// one. The earlier native kernel (fa_decode.cu) was slow for structural reasons: every warp of the GQA group re-read
// the same K/V row with 2-byte loads, and each cell was a serial load -> dot -> 5-shuffle chain.
//
// Two kernels: v4 (fa_dec_q4_k, below, the default: AI_FA_Q_TC=1) does both products on tensor cores and walks its
// splits persistently; v3 (fa_dec_q_k, AI_FA_Q_TC=0) is the dp4a/FMA reference it is checked against.
//
// v3 shape. Grid (KV split, KV head, stream); tiles of T cells, 8*T threads ("cell groups" of T threads).
//   - The block loads a tile's raw K and V rows with 16-byte coalesced loads, one tile ahead into registers
//     (the next tile's loads are in flight while this tile computes), then repacks them in shared memory into
//     int8 rows + one f32 scale per 32 (q4_0 nibbles become (n - 8) bytes).
//   - Q is quantised per 32 to int8 (as ggml's own quantised path does), so Q.K is dp4a with one float
//     epilogue per 32: thread (cell = t % T, row group = t / T).
//   - Online softmax: one cell group per row. P.V: thread (4 dims, row group), V read once per tile.
//   - T = 16 (128 threads) is the default: with T = 32 the memory side is fine, but its four barrier-separated
//     phases do not overlap the loads at 2 blocks/SM (104-128 registers); T = 16 halves the barrier scope and
//     doubles the resident blocks.
//   - Live length: a round's n_kv is a power-of-two envelope (engine.cpp ladder), the mask hides the rest;
//     the kernel reads this round's write cells and drops cells past the stream's last written cell, so
//     whole splits past it exit at once. Cells there are masked for every row, so this is exact -- and it
//     also keeps never-written cache bytes (possible NaN scales) out of the arithmetic.
//
// MTP == plain by construction. A row's arithmetic -- Q quantisation, cell order, tile and split boundaries
// (functions of n_kv only), the dp4a order, every epilogue (explicit __fmaf_rn / __fmul_rn so nvcc cannot
// contract differently per template instance), the butterfly sums, the combine -- does not depend on L or on
// which other rows share the block. Cells skipped by the live cut are masked for that row anyway: p = 0.
//
// Not bit-identical to ggml's kernel (different reduction order); needs the paired-NLL quality gate.
#include "kernels/kernels.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define FQ_D    256
#define FQ_KROW 260   // repacked int8 row stride in bytes (65 words: conflict-free across cells)

// AI_FA_Q_X (deletion bisection, wrong results): 1 no global K/V loads, 2 no repack, 4 no scores, 8 no softmax,
// 16 no P.V. Read once on the host; 0 in normal use.
__constant__ int fq_x;

template <int TYPE> struct fq_fmt;
template <> struct fq_fmt<0> { static constexpr int ROWB = 272, U = 17, BLK = 34; };   // q8_0
template <> struct fq_fmt<1> { static constexpr int ROWB = 144, U = 9,  BLK = 18; };   // q4_0

template <int W> static __device__ __forceinline__ float fq_gmax(float x) {
#pragma unroll
    for (int o = W / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, W));
    return x;
}
template <int W> static __device__ __forceinline__ float fq_gsum(float x) {
#pragma unroll
    for (int o = W / 2; o > 0; o >>= 1) x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, o, W));
    return x;
}

// Repack one (row, 32-block) from the raw smem row into 8 int8 words + its scale.
template <int TYPE>
static __device__ __forceinline__ void fq_repack(const uint8_t * raw, int row, int blk, bool valid, uint32_t * dst,
                                                 float * scl) {
    using F = fq_fmt<TYPE>;
    if (!valid) {
#pragma unroll
        for (int w = 0; w < 8; ++w) dst[w] = 0;
        *scl = 0.0f;
        return;
    }
    const uint16_t * b = (const uint16_t *) (raw + row * F::ROWB + blk * F::BLK);   // 2-byte aligned
    *scl = __half2float(__ushort_as_half(b[0]));
    if (TYPE == 0) {
#pragma unroll
        for (int w = 0; w < 8; ++w) dst[w] = (uint32_t) b[1 + 2 * w] | ((uint32_t) b[2 + 2 * w] << 16);
    } else {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t x = (uint32_t) b[1 + 2 * w] | ((uint32_t) b[2 + 2 * w] << 16);
            dst[w]     = __vsub4(x & 0x0F0F0F0Fu, 0x08080808u);
            dst[4 + w] = __vsub4((x >> 4) & 0x0F0F0F0Fu, 0x08080808u);
        }
    }
}

static __device__ __forceinline__ float fq_b2f(uint32_t w, int i) {
    return (float) (int8_t) (uint8_t) (w >> (8 * i));
}

template <int TYPE, int T>
static constexpr size_t fq_smem_c(int nr8) {
    return (size_t) 2 * T * fq_fmt<TYPE>::ROWB + 2 * T * FQ_KROW + 2 * T * 8 * 4 + (size_t) nr8 * 8 * FQ_D +
           nr8 * 8 * 8 * 4 + nr8 * 8 * (T + 1) * 4 + nr8 * 8 * 4;
}

// NR8 = ceil(R / 8), R = G * L rows (query heads of the group x tokens of the stream).
template <int TYPE, int NR8, int T>
__global__ static void __launch_bounds__(8 * T)
fa_dec_q_k(const float * __restrict__ Q, const char * __restrict__ Kc, const char * __restrict__ Vc,
           const __half * __restrict__ mask, const int64_t * __restrict__ cells, float * __restrict__ part_o,
           float2 * __restrict__ part_ml, int G, int L, int Hq, int n_kv, int split, long long cell_stride,
           long long head_stride, long long stream_stride, long long mask_row, long long mask_stream,
           long long cell_base, long long region, float scale, int rows_total) {
    using F = fq_fmt<TYPE>;
    constexpr int NT  = 8 * T;                          // threads
    constexpr int NW  = NT / 32;                        // warps
    constexpr int R8  = NR8 * 8;
    constexpr int NLD = (T * F::U + NT - 1) / NT;       // uint4 loads per thread per tensor per tile
    constexpr int RG2 = NT / 64;                        // P.V row groups (64 dim-quads each)
    constexpr int NJ  = R8 / RG2;                       // P.V rows per thread
    const int     s = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
    const int     t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int     cg = t % T, rg = t / T;               // score / softmax: cell within tile, row group
    const int     R = G * L;

    extern __shared__ __align__(16) uint8_t fq_sm[];
    uint8_t * kraw = fq_sm;                             // [T][ROWB]
    uint8_t * vraw = kraw + T * F::ROWB;
    uint8_t * ki   = vraw + T * F::ROWB;                // [T][KROW]
    uint8_t * vi   = ki + T * FQ_KROW;
    float *   ks   = (float *) (vi + T * FQ_KROW);      // [T][8]
    float *   vs   = ks + T * 8;
    uint8_t * qi   = (uint8_t *) (vs + T * 8);          // [R8][256]
    float *   qs   = (float *) (qi + R8 * FQ_D);        // [R8][8]
    float *   sp   = qs + R8 * 8;                       // [R8][T+1] scores, then probabilities
    float *   corr = sp + R8 * (T + 1);                 // [R8]

    // live cells of this stream: past the last cell this round writes, every row is masked
    int live = n_kv;
    if (cells != nullptr) {
        long long mx = -1;
        for (int l = 0; l < L; ++l) mx = max(mx, (long long) cells[b * L + l]);
        const long long lv = mx - (cell_base + (long long) b * region) + 1;
        if (lv >= 1 && lv < live) live = (int) lv;
    }
    const int c0 = s * split;
    const int c1 = min(min(c0 + split, n_kv), live);

    float m_r[NR8], l_r[NR8];   // softmax state of rows rg + 8i (identical in every thread of the group)
#pragma unroll
    for (int i = 0; i < NR8; ++i) { m_r[i] = -CUDART_INF_F; l_r[i] = 0.0f; }
    float o[NJ][4];             // P.V: dims 4dk..4dk+3 of rows rg2 + RG2*j
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.0f;
    const int dk = t & 63, rg2 = t >> 6;

    if (c0 < c1) {
        // ---- Q: quantise each row per 32 (warp per row; lane owns 8 dims, 4 lanes per block)
        for (int r = warp; r < R; r += NW) {
            const int     l = r / G, g = r % G;
            const float * qr = Q + ((size_t) ((b * L + l) * Hq + hk * G + g)) * FQ_D + lane * 8;
            const float4  a = ((const float4 *) qr)[0], c = ((const float4 *) qr)[1];
            float         x[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
            float         amax = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(x[i]));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 32));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 32));
            const float d = amax / 127.0f;
            uint32_t    w0 = 0, w1 = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int      qv = d > 0.0f ? __float2int_rn(x[i] / d) : 0;
                const uint32_t by = (uint32_t) (uint8_t) (int8_t) qv;
                if (i < 4) w0 |= by << (8 * i); else w1 |= by << (8 * (i - 4));
            }
            ((uint2 *) (qi + r * FQ_D))[lane] = make_uint2(w0, w1);
            if ((lane & 3) == 0) qs[r * 8 + (lane >> 2)] = d;
        }

        const long long base = (long long) b * stream_stride + (long long) hk * head_stride;
        const char *    kb   = Kc + base;
        const char *    vb   = Vc + base;
        const __half *  mb   = mask + (long long) b * mask_stream;

        uint4 rk[NLD], rv[NLD];
        auto load_tile = [&](int ct) {
#pragma unroll
            for (int n = 0; n < NLD; ++n) {
                const int idx = t + n * NT;
                const int row = idx / F::U, u = idx % F::U;
                if (!(fq_x & 1) && idx < T * F::U && ct + row < c1) {
                    rk[n] = __ldg((const uint4 *) (kb + (long long) (ct + row) * cell_stride) + u);
                    rv[n] = __ldg((const uint4 *) (vb + (long long) (ct + row) * cell_stride) + u);
                }
            }
        };
        load_tile(c0);

        for (int ct = c0; ct < c1; ct += T) {
            __syncthreads();   // previous tile's readers are done with every buffer
#pragma unroll
            for (int n = 0; n < NLD; ++n) {
                const int idx = t + n * NT;
                const int row = idx / F::U;
                if (idx < T * F::U && ct + row < c1) {
                    ((uint4 *) kraw)[idx] = rk[n];
                    ((uint4 *) vraw)[idx] = rv[n];
                }
            }
            __syncthreads();
            if (ct + T < c1) load_tile(ct + T);   // in flight during this tile's compute
            if (!(fq_x & 2)) {
                const int  row = t >> 3, blk = t & 7;   // T rows x 8 blocks = NT items
                const bool valid = ct + row < c1;
                fq_repack<TYPE>(kraw, row, blk, valid, (uint32_t *) (ki + row * FQ_KROW) + blk * 8, ks + row * 8 + blk);
                fq_repack<TYPE>(vraw, row, blk, valid, (uint32_t *) (vi + row * FQ_KROW) + blk * 8, vs + row * 8 + blk);
            }
            __syncthreads();

            // ---- scores: thread (cell cg, rows rg + 8i)
            if (!(fq_x & 4)) {
                const int        cell = ct + cg;
                const uint32_t * krow = (const uint32_t *) (ki + cg * FQ_KROW);
                float            sf[NR8];
#pragma unroll
                for (int i = 0; i < NR8; ++i) sf[i] = 0.0f;
#pragma unroll
                for (int blk = 0; blk < 8; ++blk) {
                    int ai[NR8];
#pragma unroll
                    for (int i = 0; i < NR8; ++i) ai[i] = 0;
#pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        const int kw = (int) krow[blk * 8 + w];
#pragma unroll
                        for (int i = 0; i < NR8; ++i) {
                            const int r = rg + 8 * i;
                            if (r < R) ai[i] = __dp4a(kw, ((const int *) (qi + r * FQ_D))[blk * 8 + w], ai[i]);
                        }
                    }
                    const float kd = ks[cg * 8 + blk];
#pragma unroll
                    for (int i = 0; i < NR8; ++i) {
                        const int r = rg + 8 * i;
                        if (r < R) sf[i] = __fmaf_rn((float) ai[i], __fmul_rn(qs[r * 8 + blk], kd), sf[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < NR8; ++i) {
                    const int r = rg + 8 * i;
                    if (r < R) {
                        const int l = r / G;
                        sp[r * (T + 1) + cg] = cell < c1
                            ? __fmaf_rn(sf[i], scale, __half2float(mb[(long long) l * mask_row + cell]))
                            : -CUDART_INF_F;
                    }
                }
            }
            __syncthreads();

            // ---- online softmax: cell group per row (T lanes)
#pragma unroll
            for (int i = 0; i < NR8; ++i) {
                const int r = rg + 8 * i;
                // every lane takes part in the shuffles (two row groups share a warp at T = 16, and only one
                // of them may own a row when R is not a multiple of 8); only the writes are guarded
                if (!(fq_x & 8)) {
                    const bool  act = r < R;
                    const float v   = act ? sp[r * (T + 1) + cg] : -CUDART_INF_F;
                    const float mt = fq_gmax<T>(v);
                    const float mn = fmaxf(m_r[i], mt);
                    float       p = 0.0f, cr = 1.0f;
                    if (mn != -CUDART_INF_F) {
                        cr = expf(m_r[i] - mn);   // exp(-inf) = 0 on the first visible tile
                        p  = expf(v - mn);
                    }
                    const float ps = fq_gsum<T>(p);
                    l_r[i]         = __fmaf_rn(l_r[i], cr, ps);
                    m_r[i]         = mn;
                    if (act) {
                        sp[r * (T + 1) + cg] = p;
                        if (cg == 0) corr[r] = cr;
                    }
                }
            }
            __syncthreads();

            // ---- P.V: thread (dims 4dk..4dk+3, rows rg2 + RG2*j)
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                const int r = rg2 + RG2 * j;
                if (r < R) {
                    const float cr = corr[r];
#pragma unroll
                    for (int e = 0; e < 4; ++e) o[j][e] = __fmul_rn(o[j][e], cr);
                }
            }
#pragma unroll 4
            for (int c = 0; c < T && !(fq_x & 16); ++c) {
                const uint32_t vw = ((const uint32_t *) (vi + c * FQ_KROW))[dk];
                const float    vd = vs[c * 8 + (dk >> 3)];
                const float    v0 = fq_b2f(vw, 0), v1 = fq_b2f(vw, 1), v2 = fq_b2f(vw, 2), v3 = fq_b2f(vw, 3);
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
                    const int r = rg2 + RG2 * j;
                    if (r < R) {
                        const float pv = __fmul_rn(sp[r * (T + 1) + c], vd);
                        o[j][0] = __fmaf_rn(pv, v0, o[j][0]);
                        o[j][1] = __fmaf_rn(pv, v1, o[j][1]);
                        o[j][2] = __fmaf_rn(pv, v2, o[j][2]);
                        o[j][3] = __fmaf_rn(pv, v3, o[j][3]);
                    }
                }
            }
        }
    }

    // ---- partials (a split with no live cell writes m = -inf, which the combine skips)
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        const int r = rg2 + RG2 * j;
        if (r < R) {
            const int l = r / G, g = r % G;
            const int row = (b * L + l) * Hq + hk * G + g;
            ((float4 *) (part_o + ((size_t) s * rows_total + row) * FQ_D))[dk] =
                make_float4(o[j][0], o[j][1], o[j][2], o[j][3]);
        }
    }
    if (cg == 0) {
#pragma unroll
        for (int i = 0; i < NR8; ++i) {
            const int r = rg + 8 * i;
            if (r < R) {
                const int l = r / G, g = r % G;
                const int row = (b * L + l) * Hq + hk * G + g;
                part_ml[(size_t) s * rows_total + row] = make_float2(m_r[i], l_r[i]);
            }
        }
    }
}

// v4 (AI_FA_Q_TC, default on): tensor cores for both products, T = 32, 256 threads.
//   - Q.K: mma.m16n8k32 s8 per 32-block, int32 exact -- the same integer dot the dp4a path sums, and the same
//     per-block float epilogue in the same order, so the scores are bit-identical to v3's.
//   - P.V: mma.m16n8k16 f16 with f32 accumulate. V's int8 values are exact in f16. Warp w owns dim block w;
//     its A operand is p * vs / vsr with vsr = the largest V scale in that block among the cells this row
//     attends to in this tile (p > 0) -- so the f16 operand is in [0, 1] and never subnormal for any p that
//     matters -- and the tile's product is folded in as o = fma(tile, vsr, o * corr). vsr is a function of the
//     row alone: a tile-wide max would let the live cut of a verify batch change a row's f16 rounding.
// v3 is compute-bound and scales poorly with L; the FMA work per (row, cell) is what moves to the tensor cores here.
#define FQ4_ROW 272   // int8 row stride (68 words) for K, V and Q: conflict-free for the fragment pattern

static __device__ __forceinline__ void fq_mma_s8(int (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
static __device__ __forceinline__ void fq_mma_f16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
static __device__ __forceinline__ uint32_t fq_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

// Shared memory: the raw K/V tile and the score/probability block (sp, corr, vsr) have disjoint lifetimes inside a
// tile (raw: store -> repack; sp: scores -> P.V; the next raw store is behind the loop-top barrier), so they share
// region A. That keeps the RT = 2 (MTP verify) variant at 2 blocks per SM.
template <int TYPE>
static constexpr __host__ __device__ size_t fq4_region_a(int rt) {
    return (size_t) 2 * 32 * fq_fmt<TYPE>::ROWB > (size_t) rt * 16 * (33 + 1 + 8) * 4
               ? (size_t) 2 * 32 * fq_fmt<TYPE>::ROWB
               : (size_t) rt * 16 * (33 + 1 + 8) * 4;
}
template <int TYPE>
static constexpr size_t fq4_smem_c(int rt) {
    return fq4_region_a<TYPE>(rt) + 2 * 32 * FQ4_ROW + 2 * 32 * 8 * 4 + (size_t) rt * 16 * FQ4_ROW + rt * 16 * 8 * 4;
}

// RT = ceil(R / 16) row tiles. Persistent over splits: block x walks splits x, x + gridDim.x, ... of its KV head
// and carries its online-softmax state across them, so it writes one partial (index x). gridDim.x is a function of
// the device and Hkv * B only and the split of n_kv only, so a row merges the same cells in the same order alone
// and in a verify batch. Splits past the live length end the walk (they are contiguous at the end).
template <int TYPE, int RT>
__global__ static void __launch_bounds__(256, 2)
fa_dec_q4_k(const float * __restrict__ Q, const char * __restrict__ Kc, const char * __restrict__ Vc,
            const __half * __restrict__ mask, const int64_t * __restrict__ cells, float * __restrict__ part_o,
            float2 * __restrict__ part_ml, int G, int L, int Hq, int n_kv, int split, long long cell_stride,
            long long head_stride, long long stream_stride, long long mask_row, long long mask_stream,
            long long cell_base, long long region, float scale, int rows_total) {
    using F = fq_fmt<TYPE>;
    constexpr int T   = 32, NT = 256;
    constexpr int R16 = RT * 16;
    constexpr int NLD = (T * F::U + NT - 1) / NT;
    const int     s = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;   // s: this block's partial index
    const int     t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int     g = lane >> 2, tg = lane & 3;   // mma fragment coordinates
    const int     R = G * L;

    extern __shared__ __align__(16) uint8_t fq_sm[];
    uint8_t * kraw = fq_sm;                             // region A: [T][ROWB] raw K, raw V ...
    uint8_t * vraw = kraw + T * F::ROWB;
    float *   sp   = (float *) fq_sm;                   // ... aliased by [R16][33] scores / probabilities
    float *   corr = sp + R16 * 33;                     // [R16]
    float *   vsr  = corr + R16;                        // [R16][8] per-row V-scale normaliser
    uint8_t * ki   = fq_sm + fq4_region_a<TYPE>(RT);    // [T][FQ4_ROW] int8
    uint8_t * vi   = ki + T * FQ4_ROW;
    float *   ks   = (float *) (vi + T * FQ4_ROW);      // [T][8]
    float *   vs   = ks + T * 8;
    uint8_t * qi   = (uint8_t *) (vs + T * 8);          // [R16][FQ4_ROW] int8 (rows >= R are zero)
    float *   qs   = (float *) (qi + R16 * FQ4_ROW);    // [R16][8]

    int live = n_kv;
    if (cells != nullptr) {
        long long mx = -1;
        for (int l = 0; l < L; ++l) mx = max(mx, (long long) cells[b * L + l]);
        const long long lv = mx - (cell_base + (long long) b * region) + 1;
        if (lv >= 1 && lv < live) live = (int) lv;
    }
    const int lim = min(n_kv, live);
    int       c0  = s * split;
    int       c1  = min(c0 + split, lim);

    float m_r[2 * RT], l_r[2 * RT];   // softmax state of rows warp + 8i (warp-uniform)
#pragma unroll
    for (int i = 0; i < 2 * RT; ++i) { m_r[i] = -CUDART_INF_F; l_r[i] = 0.0f; }
    float o[RT][4][4];                // rows rt*16 + g (+8), dims warp*32 + nt*8 + 2tg (+1)
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[rt][nt][e] = 0.0f;

    if (c0 < c1) {
        for (int r = warp; r < R16; r += 8) {
            uint32_t w0 = 0, w1 = 0;
            float    d  = 0.0f;
            if (r < R) {
                const int     l = r / G, gg = r % G;
                const float * qr = Q + ((size_t) ((b * L + l) * Hq + hk * G + gg)) * FQ_D + lane * 8;
                const float4  a = ((const float4 *) qr)[0], c = ((const float4 *) qr)[1];
                float         x[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
                float         amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(x[i]));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 32));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 32));
                d    = amax / 127.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int      qv = d > 0.0f ? __float2int_rn(x[i] / d) : 0;
                    const uint32_t by = (uint32_t) (uint8_t) (int8_t) qv;
                    if (i < 4) w0 |= by << (8 * i); else w1 |= by << (8 * (i - 4));
                }
            }
            ((uint2 *) (qi + r * FQ4_ROW))[lane] = make_uint2(w0, w1);
            if ((lane & 3) == 0) qs[r * 8 + (lane >> 2)] = d;
        }

        const long long base = (long long) b * stream_stride + (long long) hk * head_stride;
        const char *    kb   = Kc + base;
        const char *    vb   = Vc + base;
        const __half *  mb   = mask + (long long) b * mask_stream;

        uint4 rk[NLD], rv[NLD];
        auto load_tile = [&](int ct) {
#pragma unroll
            for (int n = 0; n < NLD; ++n) {
                const int idx = t + n * NT;
                const int row = idx / F::U, u = idx % F::U;
                if (!(fq_x & 1) && idx < T * F::U && ct + row < c1) {
                    rk[n] = __ldg((const uint4 *) (kb + (long long) (ct + row) * cell_stride) + u);
                    rv[n] = __ldg((const uint4 *) (vb + (long long) (ct + row) * cell_stride) + u);
                }
            }
        };
      for (; c0 < lim; c0 += gridDim.x * split) {
        c1 = min(c0 + split, lim);
        load_tile(c0);

        for (int ct = c0; ct < c1; ct += T) {
            __syncthreads();
#pragma unroll
            for (int n = 0; n < NLD; ++n) {
                const int idx = t + n * NT;
                const int row = idx / F::U;
                if (idx < T * F::U && ct + row < c1) {
                    ((uint4 *) kraw)[idx] = rk[n];
                    ((uint4 *) vraw)[idx] = rv[n];
                }
            }
            __syncthreads();
            if (ct + T < c1) load_tile(ct + T);
            if (!(fq_x & 2)) {
                const int  row = t >> 3, blk = t & 7;
                const bool valid = ct + row < c1;
                fq_repack<TYPE>(kraw, row, blk, valid, (uint32_t *) (ki + row * FQ4_ROW) + blk * 8, ks + row * 8 + blk);
                fq_repack<TYPE>(vraw, row, blk, valid, (uint32_t *) (vi + row * FQ4_ROW) + blk * 8, vs + row * 8 + blk);
            }
            __syncthreads();

            // ---- scores: warp tile (16 rows x 8 cells), 8 int8 mma per tile
            if (!(fq_x & 4)) {
                for (int wt = warp; wt < RT * 4; wt += 8) {
                    const int rA = (wt >> 2) * 16 + g, rB = rA + 8;
                    const int cn = (wt & 3) * 8;                 // first cell of the tile
                    const int cA = cn + 2 * tg, cB = cA + 1;     // this thread's output cells
                    const uint32_t * qa = (const uint32_t *) (qi + rA * FQ4_ROW);
                    const uint32_t * qb = (const uint32_t *) (qi + rB * FQ4_ROW);
                    const uint32_t * kr = (const uint32_t *) (ki + (cn + g) * FQ4_ROW);
                    float sf[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
                    for (int blk = 0; blk < 8; ++blk) {
                        const uint32_t a[4]  = { qa[blk * 8 + tg], qb[blk * 8 + tg], qa[blk * 8 + 4 + tg], qb[blk * 8 + 4 + tg] };
                        const uint32_t bb[2] = { kr[blk * 8 + tg], kr[blk * 8 + 4 + tg] };
                        int c[4] = { 0, 0, 0, 0 };
                        fq_mma_s8(c, a, bb);
                        const float qA = qs[rA * 8 + blk], qB = qs[rB * 8 + blk];
                        const float kA = ks[cA * 8 + blk], kB = ks[cB * 8 + blk];
                        sf[0] = __fmaf_rn((float) c[0], __fmul_rn(qA, kA), sf[0]);
                        sf[1] = __fmaf_rn((float) c[1], __fmul_rn(qA, kB), sf[1]);
                        sf[2] = __fmaf_rn((float) c[2], __fmul_rn(qB, kA), sf[2]);
                        sf[3] = __fmaf_rn((float) c[3], __fmul_rn(qB, kB), sf[3]);
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int r = e < 2 ? rA : rB, c = (e & 1) ? cB : cA, cell = ct + c;
                        if (r < R) {
                            sp[r * 33 + c] = cell < c1
                                ? __fmaf_rn(sf[e], scale, __half2float(mb[(long long) (r / G) * mask_row + cell]))
                                : -CUDART_INF_F;
                        }
                    }
                }
            }
            __syncthreads();

            // ---- online softmax: warp per row (rows >= R get p = 0 so the f16 A operand is zero there)
            if (!(fq_x & 8)) {
#pragma unroll
                for (int i = 0; i < 2 * RT; ++i) {
                    const int r = warp + 8 * i;
                    if (r >= R) {
                        sp[r * 33 + lane] = 0.0f;
                        if (lane == 0) corr[r] = 1.0f;
                        if (lane < 8) vsr[r * 8 + lane] = 1.0f;
                        continue;
                    }
                    const float v  = sp[r * 33 + lane];
                    const float mt = fq_gmax<32>(v);
                    const float mn = fmaxf(m_r[i], mt);
                    float       p = 0.0f, cr = 1.0f;
                    if (mn != -CUDART_INF_F) {
                        cr = expf(m_r[i] - mn);
                        p  = expf(v - mn);
                    }
                    const float ps = fq_gsum<32>(p);
                    l_r[i]         = __fmaf_rn(l_r[i], cr, ps);
                    m_r[i]         = mn;
                    sp[r * 33 + lane] = p;
                    if (lane == 0) corr[r] = cr;
#pragma unroll
                    for (int blk = 0; blk < 8; ++blk) {
                        // |scale| (q4_0 scales can be negative); non-negative float bits order like uints
                        const unsigned u = p > 0.0f ? __float_as_uint(fabsf(vs[lane * 8 + blk])) : 0u;
                        const float    m = __uint_as_float(__reduce_max_sync(0xffffffffu, u));
                        if (lane == blk) vsr[r * 8 + blk] = m > 0.0f ? m : 1.0f;
                    }
                }
            }
            __syncthreads();

            // ---- P.V: warp = dim block; f16 mma over 2 k-steps of 16 cells, 4 n-tiles of 8 dims
            if (!(fq_x & 16)) {
                const int blk = warp;
                float     vc[2][4];   // V scales of cells k2*16 + {2tg, 2tg+1, 2tg+8, 2tg+9}
#pragma unroll
                for (int k2 = 0; k2 < 2; ++k2) {
                    const int cc = k2 * 16 + 2 * tg;
                    vc[k2][0] = vs[cc * 8 + blk];
                    vc[k2][1] = vs[(cc + 1) * 8 + blk];
                    vc[k2][2] = vs[(cc + 8) * 8 + blk];
                    vc[k2][3] = vs[(cc + 9) * 8 + blk];
                }
                uint32_t bf[2][4][2];    // V fragments: k = cells, n = dims blk*32 + nt*8 + g
#pragma unroll
                for (int k2 = 0; k2 < 2; ++k2) {
                    const int cc = k2 * 16 + 2 * tg;
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const int n = blk * 32 + nt * 8 + g;
                        bf[k2][nt][0] = fq_h2((float) (int8_t) vi[cc * FQ4_ROW + n], (float) (int8_t) vi[(cc + 1) * FQ4_ROW + n]);
                        bf[k2][nt][1] = fq_h2((float) (int8_t) vi[(cc + 8) * FQ4_ROW + n], (float) (int8_t) vi[(cc + 9) * FQ4_ROW + n]);
                    }
                }
#pragma unroll
                for (int rt = 0; rt < RT; ++rt) {
                    const int   rA = rt * 16 + g, rB = rA + 8;
                    const float crA = corr[rA], crB = corr[rB];
                    const float nA = vsr[rA * 8 + blk], nB = vsr[rB * 8 + blk];
                    const float iA = __frcp_rn(nA), iB = __frcp_rn(nB);
                    float       ctl[4][4];
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                        for (int e = 0; e < 4; ++e) ctl[nt][e] = 0.0f;
#pragma unroll
                    for (int k2 = 0; k2 < 2; ++k2) {
                        const int     cc = k2 * 16 + 2 * tg;
                        const float * pa = sp + rA * 33;
                        const float * pb = sp + rB * 33;
#define FQ4_A(P, I, J) __fmul_rn(__fmul_rn(P, vc[k2][J]), I)
                        const uint32_t a[4] = {
                            fq_h2(FQ4_A(pa[cc], iA, 0), FQ4_A(pa[cc + 1], iA, 1)),
                            fq_h2(FQ4_A(pb[cc], iB, 0), FQ4_A(pb[cc + 1], iB, 1)),
                            fq_h2(FQ4_A(pa[cc + 8], iA, 2), FQ4_A(pa[cc + 9], iA, 3)),
                            fq_h2(FQ4_A(pb[cc + 8], iB, 2), FQ4_A(pb[cc + 9], iB, 3)) };
#undef FQ4_A
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) fq_mma_f16(ctl[nt], a, bf[k2][nt]);
                    }
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        o[rt][nt][0] = __fmaf_rn(ctl[nt][0], nA, __fmul_rn(o[rt][nt][0], crA));
                        o[rt][nt][1] = __fmaf_rn(ctl[nt][1], nA, __fmul_rn(o[rt][nt][1], crA));
                        o[rt][nt][2] = __fmaf_rn(ctl[nt][2], nB, __fmul_rn(o[rt][nt][2], crB));
                        o[rt][nt][3] = __fmaf_rn(ctl[nt][3], nB, __fmul_rn(o[rt][nt][3], crB));
                    }
                }
            }
        }
      }
    }

#pragma unroll
    for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = rt * 16 + g + 8 * h;
            if (r < R) {
                const int l = r / G, gg = r % G;
                const int row = (b * L + l) * Hq + hk * G + gg;
                float *   po  = part_o + ((size_t) s * rows_total + row) * FQ_D + warp * 32 + 2 * tg;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    *(float2 *) (po + nt * 8) = make_float2(o[rt][nt][2 * h], o[rt][nt][2 * h + 1]);
                }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < 2 * RT; ++i) {
            const int r = warp + 8 * i;
            if (r < R) {
                const int l = r / G, gg = r % G;
                const int row = (b * L + l) * Hq + hk * G + gg;
                part_ml[(size_t) s * rows_total + row] = make_float2(m_r[i], l_r[i]);
            }
        }
    }
}

// Combine: grid (row, 4 dim quarters), 64 threads. Every block redoes the (cheap) split weights for its row in
// shared memory; each thread then sums its dim over the splits in a fixed order. Row-local, so L-invariant.
#define FQC_MAXS 4096
__global__ static void __launch_bounds__(64)
fa_q_combine_k(const float * __restrict__ part_o, const float2 * __restrict__ part_ml, float * __restrict__ out,
               int n_split, int rows_total) {
    __shared__ float w[FQC_MAXS];
    __shared__ float red[64];
    const int row = blockIdx.x, d = blockIdx.y * 64 + threadIdx.x, t = threadIdx.x;
    float     M   = -CUDART_INF_F;
    for (int s = t; s < n_split; s += 64) M = fmaxf(M, part_ml[(size_t) s * rows_total + row].x);
    red[t] = M;
    __syncthreads();
    for (int o = 32; o > 0; o >>= 1) {
        if (t < o) red[t] = fmaxf(red[t], red[t + o]);
        __syncthreads();
    }
    M = red[0];
    __syncthreads();
    float den = 0.0f;
    for (int s = t; s < n_split; s += 64) {
        const float2 ml = part_ml[(size_t) s * rows_total + row];
        const float  ws = ml.x == -CUDART_INF_F ? 0.0f : expf(ml.x - M);
        w[s]            = ws;
        den             = __fmaf_rn(ml.y, ws, den);
    }
    red[t] = den;
    __syncthreads();
    for (int o = 32; o > 0; o >>= 1) {
        if (t < o) red[t] = __fadd_rn(red[t], red[t + o]);
        __syncthreads();
    }
    den = red[0];
    float num = 0.0f;
#pragma unroll 8
    for (int s = 0; s < n_split; ++s) {
        if (w[s] != 0.0f) num = __fmaf_rn(part_o[((size_t) s * rows_total + row) * FQ_D + d], w[s], num);
    }
    out[(size_t) row * FQ_D + d] = den > 0.0f ? num / den : 0.0f;
}

// AI_FA_Q_CHECK: in-situ row-invariance check (see fa_decode_q_selfcheck). Counters live in mapped pinned
// host memory so the check also works inside captured graphs: [0] rows compared, [1] rows that differed,
// [2] differing floats.
static unsigned long long * fq_chk = nullptr;
__global__ static void fq_cmp_k(const float * __restrict__ a, const float * __restrict__ b, int n,
                                unsigned long long * cnt) {
    __shared__ int bad;
    if (threadIdx.x == 0) bad = 0;
    __syncthreads();
    int nb = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) nb += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    if (nb) atomicAdd(&bad, nb);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&cnt[0], 1ULL);
        if (bad) { atomicAdd(&cnt[1], 1ULL); atomicAdd(&cnt[2], (unsigned long long) bad); }
    }
}
struct fq_chk_report_t {
    ~fq_chk_report_t() {
        if (fq_chk) {
            fprintf(stderr, "[fa-q-check] verify rows re-run alone: %llu compared, %llu DIFFERED (%llu floats)%s\n",
                    fq_chk[0], fq_chk[1], fq_chk[2], fq_chk[1] ? "  <-- NOT row-invariant" : "  (bitwise identical)");
        }
    }
};
static fq_chk_report_t fq_chk_report_inst;

static bool fq_tc() {
    static const bool tc = getenv("AI_FA_Q_TC") ? atoi(getenv("AI_FA_Q_TC")) != 0 : true;
    return tc;
}
static int fq_tile() {
    static const int t = getenv("AI_FA_Q_T") ? atoi(getenv("AI_FA_Q_T")) : 16;
    return fq_tc() || t == 32 ? 32 : 16;
}

namespace ai::kern {

static int fq_nsm() {
    static int nsm = 0;
    if (nsm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (nsm <= 0) nsm = 80;
    }
    return nsm;
}

// v4 persistent grid: blocks per (KV head, stream) -- one wave at 2 blocks per SM. Depends on the device, Hkv and B
// only (never L), which the MTP == plain identity needs.
static int64_t fq4_blocks(int64_t Hkv, int64_t B) {
    const int64_t gb = 2LL * fq_nsm() / (Hkv * B);
    return gb < 1 ? 1 : gb;
}

int fa_decode_q_split(int64_t n_kv, int64_t Hkv, int64_t B) {
    const int nsm = fq_nsm();
    if (fq_tc()) {
        // AI_FA_Q_SPB: splits per persistent block over the full envelope (default 16): small splits keep the walk
        // balanced when the live length is anywhere in (envelope/2, envelope].
        static const int spb = getenv("AI_FA_Q_SPB") ? atoi(getenv("AI_FA_Q_SPB")) : 16;
        const int64_t    tot = fq4_blocks(Hkv, B) * (spb > 0 ? spb : 8);
        int64_t          sp  = (n_kv + tot - 1) / tot;
        sp                   = (sp + 31) / 32 * 32;
        if (sp < 32) sp = 32;
        while ((n_kv + sp - 1) / sp > FQC_MAXS) sp += 32;
        return (int) sp;
    }
    // ~8 blocks per SM over the full envelope (a function of n_kv only, so a token gets the same split
    // boundaries alone and in a verify batch). Capped so the combine's weight table fits.
    // AI_FA_Q_WAVES: blocks per SM targeted (default 8). Fewer, larger splits shrink the partials the combine
    // reads back (the fixed per-call cost), more splits shrink the tail.
    static const int waves = getenv("AI_FA_Q_WAVES") ? atoi(getenv("AI_FA_Q_WAVES")) : 8;
    const int     T      = fq_tile();
    const int64_t target = (int64_t) (waves > 0 ? waves : 8) * nsm;
    int64_t       split  = (n_kv * Hkv * B + target - 1) / target;
    split                = (split + T - 1) / T * T;
    if (split < T) split = T;
    while ((n_kv + split - 1) / split > FQC_MAXS) split += T;
    return (int) split;
}

// AI_FA_Q default: on for compute capability 8.x (sm_86 / sm_89), where it has been validated, off elsewhere.
// AI_FA_Q=0/1 overrides. It only ever runs for a q8_0/q4_0 cache; the default KV type is f16.
bool fa_decode_q_enabled() {
    static const bool on = getenv("AI_FA_Q") ? atoi(getenv("AI_FA_Q")) != 0 : device_cc_major() == 8;
    return on;
}

// floats of partials one call needs: one per split (v3) or one per persistent block (v4)
int64_t fa_decode_q_workspace(int64_t n_kv, int64_t rows_total, int split, int64_t Hkv, int64_t B) {
    int64_t n_split = (n_kv + split - 1) / split;
    if (fq_tc() && fq4_blocks(Hkv, B) < n_split) n_split = fq4_blocks(Hkv, B);
    return n_split * rows_total * (FQ_D + 2);
}

// Bytes the engine's workspace must hold for one attention call at <= 8 tokens per stream with B streams
// (v4: partials of every persistent block + the [D, Hq, B*8] output). The engine adds this to its derived size,
// which does not cover it (e.g. at --chunk 8).
int64_t fa_decode_q_ws_bound(int64_t Hq, int64_t Hkv, int64_t B) {
    const int64_t rows = B * 8 * Hq;
    const int64_t gb   = fq_tc() ? fq4_blocks(Hkv, B) : (int64_t) FQC_MAXS;
    return 4 * (gb * rows * (FQ_D + 2) + rows * FQ_D) + (4LL << 20);   // + alignment slack
}

bool fa_decode_q(void * stream, int kv_type, const float * q, const void * k, const void * v, const void * mask,
                 const int64_t * cells, float * out, float * ws, int64_t D, int64_t Hq, int64_t Hkv, int64_t L,
                 int64_t B, int64_t n_kv, int split, int64_t cell_stride, int64_t head_stride, int64_t stream_stride,
                 int64_t mask_row, int64_t mask_stream, int64_t cell_base, int64_t region, float scale) {
    const int G  = (int) (Hq / Hkv);
    const int TT = fq_tile();
    if (D != FQ_D || Hq % Hkv != 0 || G > 8 || L < 1 || L > 8 || split % TT != 0 || (kv_type != 0 && kv_type != 1) ||
        ((uintptr_t) k & 15) || ((uintptr_t) v & 15) || (cell_stride & 15) || (head_stride & 15) ||
        (stream_stride & 15) || (n_kv + split - 1) / split > FQC_MAXS) {
        return false;
    }
    const int nr8        = (int) ((G * L + 7) / 8);
    int       n_split    = (int) ((n_kv + split - 1) / split);
    if (fq_tc()) {
        const int64_t gb = fq4_blocks(Hkv, B);
        if (gb < n_split) n_split = (int) gb;   // persistent: one partial per block
    }
    const int rows_total = (int) (B * L * Hq);
    float *   part_o     = ws;
    float2 *  part_ml    = (float2 *) (ws + (size_t) n_split * rows_total * FQ_D);
    static bool said = false;
    if (!said) {
        said = true;
        const int x = getenv("AI_FA_Q_X") ? atoi(getenv("AI_FA_Q_X")) : 0;
        if (x) fprintf(stderr, "[fa-q] DELETION BISECTION AI_FA_Q_X=%d (results are wrong)\n", x);
        cudaMemcpyToSymbol(fq_x, &x, sizeof(int));
        fprintf(stderr, "[fa-q] LAUNCHED (%s, %s, T %d, splits %d x %d cells, kv heads %lld, streams %lld, G %d, L %lld, n_kv %lld, live cut %s)\n",
                kv_type == 0 ? "q8_0" : "q4_0", fq_tc() ? "tensor-core v4 persistent" : "dp4a v3", TT, n_split, split, (long long) Hkv, (long long) B, G, (long long) L,
                (long long) n_kv, cells ? "on" : "off");
    }
    const dim3   grid((unsigned) n_split, (unsigned) Hkv, (unsigned) B);
    cudaStream_t st = (cudaStream_t) stream;
#define FQ_CASE(TY, N, TL)                                                                                        \
    case N: {                                                                                                     \
        constexpr size_t sm = fq_smem_c<TY, TL>(N);                                                               \
        static bool      attr = false;                                                                            \
        if (!attr) {                                                                                              \
            attr = true;                                                                                          \
            cudaFuncSetAttribute(fa_dec_q_k<TY, N, TL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);   \
        }                                                                                                         \
        fa_dec_q_k<TY, N, TL><<<grid, 8 * TL, sm, st>>>(q, (const char *) k, (const char *) v,                     \
                                                        (const __half *) mask, cells, part_o, part_ml, G, (int) L,  \
                                                        (int) Hq, (int) n_kv, split, cell_stride, head_stride,      \
                                                        stream_stride, mask_row, mask_stream, cell_base, region,    \
                                                        scale, rows_total);                                         \
        break;                                                                                                    \
    }
#define FQ_SWITCH(TY, TL)                                                                                         \
    switch (nr8) { FQ_CASE(TY, 1, TL) FQ_CASE(TY, 2, TL) FQ_CASE(TY, 3, TL) FQ_CASE(TY, 4, TL) FQ_CASE(TY, 5, TL)  \
                   FQ_CASE(TY, 6, TL) FQ_CASE(TY, 7, TL) FQ_CASE(TY, 8, TL) default: return false; }
#define FQ4_CASE(TY, N)                                                                                          \
    case N: {                                                                                                     \
        constexpr size_t sm = fq4_smem_c<TY>(N);                                                                  \
        static bool      attr = false;                                                                            \
        if (!attr) {                                                                                              \
            attr = true;                                                                                          \
            cudaFuncSetAttribute(fa_dec_q4_k<TY, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);      \
        }                                                                                                         \
        fa_dec_q4_k<TY, N><<<grid, 256, sm, st>>>(q, (const char *) k, (const char *) v, (const __half *) mask,    \
                                                  cells, part_o, part_ml, G, (int) L, (int) Hq, (int) n_kv, split, \
                                                  cell_stride, head_stride, stream_stride, mask_row, mask_stream,  \
                                                  cell_base, region, scale, rows_total);                           \
        break;                                                                                                    \
    }
    const int rt = (G * (int) L + 15) / 16;
    if (fq_tc()) {
        if (kv_type == 0) {
            switch (rt) { FQ4_CASE(0, 1) FQ4_CASE(0, 2) FQ4_CASE(0, 3) FQ4_CASE(0, 4) default: return false; }
        } else {
            switch (rt) { FQ4_CASE(1, 1) FQ4_CASE(1, 2) FQ4_CASE(1, 3) FQ4_CASE(1, 4) default: return false; }
        }
    } else if (kv_type == 0) {
        if (TT == 32) { FQ_SWITCH(0, 32) } else { FQ_SWITCH(0, 16) }
    } else {
        if (TT == 32) { FQ_SWITCH(1, 32) } else { FQ_SWITCH(1, 16) }
    }
#undef FQ4_CASE
#undef FQ_SWITCH
#undef FQ_CASE
    fa_q_combine_k<<<dim3((unsigned) rows_total, FQ_D / 64), 64, 0, st>>>(part_o, part_ml, out, n_split, rows_total);
    return true;
}

bool fa_decode_q_selfcheck(void * stream, int kv_type, const float * q, const void * k, const void * v,
                           const void * mask, const int64_t * cells, const float * out, float * scratch, float * ws,
                           int64_t D, int64_t Hq, int64_t Hkv, int64_t L, int64_t B, int64_t n_kv, int split,
                           int64_t cell_stride, int64_t head_stride, int64_t stream_stride, int64_t mask_row,
                           int64_t mask_stream, int64_t cell_base, int64_t region, float scale) {
    if (B != 1 || L < 2) return true;
    if (fq_chk == nullptr) {
        if (cudaHostAlloc((void **) &fq_chk, 3 * sizeof(unsigned long long), cudaHostAllocMapped) != cudaSuccess) {
            return false;
        }
        fq_chk[0] = fq_chk[1] = fq_chk[2] = 0;
    }
    unsigned long long * dcnt = nullptr;
    cudaHostGetDevicePointer((void **) &dcnt, fq_chk, 0);
    cudaStream_t st = (cudaStream_t) stream;
    for (int64_t l = 0; l < L; ++l) {
        // the same row alone: its q row, its mask row, its own write cell -- what a plain decode step would pass
        if (!fa_decode_q(stream, kv_type, q + l * Hq * D, k, v, (const __half *) mask + l * mask_row,
                         cells ? cells + l : nullptr, scratch, ws, D, Hq, Hkv, 1, 1, n_kv, split, cell_stride,
                         head_stride, stream_stride, mask_row, mask_stream, cell_base, region, scale)) {
            return false;
        }
        fq_cmp_k<<<1, 256, 0, st>>>(out + l * Hq * D, scratch, (int) (Hq * D), dcnt);
    }
    return true;
}

}   // namespace ai::kern
