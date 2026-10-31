// [ALPHA_SMALLM_TC] Register-dequant tensor-core matmul for SMALL M (2..8 activation columns): the MTP / DFlash2
// verify width on a dense model. AlphaInfer-owned, copied into the vendored fork by third_party/patch_smallm_tc.py.
//
// WHY. At 8 columns the fork's MMQ costs 1.3-1.9x the single-column MMVQ time for the same weight bytes
// (3090, 27B, nsys 2026-10-03: q4_K 1.9x, q5_K 1.4x, iq4_xs 1.3x). Measured on this kernel's first version (same
// 1.0x of MMQ): it is not memory-bound but EPILOGUE-bound -- with activations scaled per 32 values, every mma's
// 16x8 int32 tile needs a float convert + row-scale x column-scale per 32-value sub-block, ~1 ALU op per weight.
//
// DESIGN.
//  * mma.sync m16n8k32 s8: A = 16 weight rows x 32 k (one sub-block), B = 32 k x 8 columns. N = 8 = verify width.
//  * K permutation: thread t of a quad owns PHYSICAL elements 8t..8t+7 of each 32-value sub-block (fragment slots
//    4t..4t+3, 16+4t..16+4t+3). For q4_K those 8 elements are 8 consecutive qs bytes whose low nibbles are
//    sub-block 2j and high nibbles 2j+1, so the A register is just w & 0x0F0F0F0F.
//  * ACTIVATIONS QUANTISED PER 256 (one scale per superblock, as ggml's CPU q8_K), with per-32 integer sums. Then
//    the weight's integer sub-block scale folds into an integer accumulator (acc_i += sc * I) and the min term into
//    another (acc_m += m * S), and the float epilogue runs ONCE per superblock instead of 8 times.
//  * Split-K with a fixed-order reduce: deterministic run to run (no float atomics).
#include "common.cuh"
#include "smallm-tc.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace {

constexpr int TC_NCOLS = 8;    // N of the mma; narrower M is zero-padded columns
constexpr int TC_WARPS = 4;    // warps per block

// activations, per column and superblock: one scale, 8 per-32 sums of the int8 values, 256 int8 values
struct __align__(16) block_a256 {
    float   d;
    int16_t s[8];
    int8_t  pad[12];
    int8_t  qs[256];
};
static_assert(sizeof(block_a256) == 288, "block_a256 layout");

__device__ __forceinline__ void mma_s8_16832(int (&c)[4], const int (&a)[4], const int (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    NO_DEVICE_CODE;
#endif
}

// acc + dx * (f.x * ai - f.y * am), as one fixed instruction sequence (no compiler contraction choices)
__device__ __forceinline__ float col_epilogue(float acc, float dx, float2 f, int ai, int am) {
    const float t = __fmaf_rn(f.x, (float) ai, -__fmul_rn(f.y, (float) am));
    return __fmaf_rn(dx, t, acc);
}

__device__ __forceinline__ uint2 ld8(const void * p) {   // 8-byte ALIGNED load
    return *(const uint2 *) p;
}

// 4 nibbles (one per byte, 0..15) -> 4 int8 table values, table held in 4 registers (no constant-memory lookup,
// which serialises across a warp's distinct addresses). __byte_perm selector nibbles keep bit 3 CLEAR (bit 3 is
// prmt's sign-replicate flag), so the 16-entry choice is two 8-byte selects and a per-byte merge on bit 3.
__device__ __forceinline__ int lut16(uint32_t q, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t x   = q & 0x07070707u;
    const uint32_t sel = (x & 0x7u) | ((x >> 4) & 0x70u) | ((x >> 8) & 0x700u) | ((x >> 12) & 0x7000u);
    const uint32_t lo  = __byte_perm(t0, t1, sel);
    const uint32_t hi  = __byte_perm(t2, t3, sel);
    const uint32_t m   = ((q & 0x08080808u) >> 3) * 0xFFu;
    return (int) ((lo & ~m) | (hi & m));
}

__device__ __forceinline__ int byte_of(const uint4 & w, int i) {   // i is a compile-time constant after unrolling
    const uint32_t v = i < 4 ? w.x : i < 8 ? w.y : i < 12 ? w.z : w.w;
    return (int) ((v >> (8 * (i & 3))) & 0xFF);
}
// ggml's get_scale_min_k4 over the 12 scale bytes held in header bytes 4..15
__device__ __forceinline__ void scale_min_k4_r(int j, const uint4 & h, int & sc, int & m) {
    if (j < 4) {
        sc = byte_of(h, 4 + j) & 63;
        m  = byte_of(h, 4 + j + 4) & 63;
    } else {
        sc = (byte_of(h, 4 + j + 4) & 0xF) | ((byte_of(h, 4 + j - 4) >> 6) << 4);
        m  = (byte_of(h, 4 + j + 4) >> 4)  | ((byte_of(h, 4 + j) >> 6) << 4);
    }
}

// ---- per-type decoders:
//   load(b, t, R)        all of the thread's loads for one row's superblock (header + weight bytes)
//   dm(R)                the row's superblock float factors (d, dmin); dmin = 0 for symmetric types
//   pair(R, j, t, ...)   int8 A-registers {elements 8t..8t+3, 8t+4..8t+7} of sub-blocks 2j and 2j+1, with their
//                        INTEGER scale and min (value = d*sc*q - dmin*m)
struct dec_q4_K {
    using block = block_q4_K;                       // 144 B: dm(4) scales(12) qs(128); 16-byte aligned
    struct R { uint4 h; uint2 q[4]; };
    __device__ static void load(const block * b, int t, R & r) {
        r.h = *(const uint4 *) b;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.q[j] = ld8(b->qs + 32 * j + 8 * t);
    }
    __device__ static float2 dm(const R & r) { return __half22float2(*(const half2 *) &r.h.x); }
    __device__ static void pair(const R & r, int j, int t, int (&lo)[2], int (&hi)[2], int & scl, int & ml,
                                int & sch, int & mh) {
        const uint2 w = r.q[j];
        lo[0] = (int) (w.x & 0x0F0F0F0Fu);         lo[1] = (int) (w.y & 0x0F0F0F0Fu);
        hi[0] = (int) ((w.x >> 4) & 0x0F0F0F0Fu);  hi[1] = (int) ((w.y >> 4) & 0x0F0F0F0Fu);
        scale_min_k4_r(2 * j, r.h, scl, ml);
        scale_min_k4_r(2 * j + 1, r.h, sch, mh);
    }
};

struct dec_q5_K {
    using block = block_q5_K;                       // 176 B: dm(4) scales(12) qh(32) qs(128); 16-byte aligned
    struct R { uint4 h; uint2 qh; uint2 q[4]; };
    __device__ static void load(const block * b, int t, R & r) {
        r.h  = *(const uint4 *) b;
        r.qh = ld8(b->qh + 8 * t);
#pragma unroll
        for (int j = 0; j < 4; ++j) r.q[j] = ld8(b->qs + 32 * j + 8 * t);
    }
    __device__ static float2 dm(const R & r) { return __half22float2(*(const half2 *) &r.h.x); }
    __device__ static void pair(const R & r, int j, int t, int (&lo)[2], int (&hi)[2], int & scl, int & ml,
                                int & sch, int & mh) {
        // dequantize_row_q5_K: element l of chunk j is ql[32j+l] nibble + 16*(qh[l] bit 2j / 2j+1)
        const uint2 w = r.q[j], h = r.qh;
        lo[0] = (int) ((w.x & 0x0F0F0F0Fu)        | (((h.x >> (2 * j))     & 0x01010101u) << 4));
        lo[1] = (int) ((w.y & 0x0F0F0F0Fu)        | (((h.y >> (2 * j))     & 0x01010101u) << 4));
        hi[0] = (int) (((w.x >> 4) & 0x0F0F0F0Fu) | (((h.x >> (2 * j + 1)) & 0x01010101u) << 4));
        hi[1] = (int) (((w.y >> 4) & 0x0F0F0F0Fu) | (((h.y >> (2 * j + 1)) & 0x01010101u) << 4));
        scale_min_k4_r(2 * j, r.h, scl, ml);
        scale_min_k4_r(2 * j + 1, r.h, sch, mh);
    }
};

struct dec_iq4_xs {
    using block = block_iq4_xs;                     // 136 B: d(2) scales_h(2) scales_l(4) qs(128); 8-byte aligned
    struct R { uint2 h; uint2 q[8]; };
    __device__ static void load(const block * b, int t, R & r) {
        r.h = *(const uint2 *) b;
        // sub-block ib: bytes qs[16ib..16ib+15]; element e < 16 = low nibble of byte e, e >= 16 = high of byte e-16
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) r.q[ib] = ld8(b->qs + 16 * ib + 8 * (t & 1));
    }
    __device__ static float2 dm(const R & r) {
        return make_float2(__half2float(__ushort_as_half((unsigned short) (r.h.x & 0xFFFF))), 0.0f);
    }
    __device__ static void sub(const R & r, int ib, int sh, int (&v)[2], int & sc) {
        // kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113} as little-endian int8 words
        constexpr uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
        v[0] = lut16((r.q[ib].x >> sh) & 0x0F0F0F0Fu, T0, T1, T2, T3);
        v[1] = lut16((r.q[ib].y >> sh) & 0x0F0F0F0Fu, T0, T1, T2, T3);
        const uint32_t scales_h = r.h.x >> 16, scales_l = r.h.y;
        sc = (int) (((scales_l >> (8 * (ib / 2) + 4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)) - 32;
    }
    __device__ static void pair(const R & r, int j, int t, int (&lo)[2], int (&hi)[2], int & scl, int & ml,
                                int & sch, int & mh) {
        const int sh = 4 * (t >> 1);
        sub(r, 2 * j, sh, lo, scl);
        sub(r, 2 * j + 1, sh, hi, sch);
        ml = 0;
        mh = 0;
    }
};

// rows x K weights (row stride row_bytes), activations y = ncols x nsb block_a256 (column-major),
// out[col*rows + row] (or a partial slice when split). One warp = TC_MT*16 rows over superblocks [sb0, sb1).
// NSB superblocks per row are loaded per step: a CONTIGUOUS run of NSB*sizeof(block) bytes per row, because with
// one superblock (144 B) per row per step the kernel was load-bound at ~480 GB/s (loads-only time = full time).
template <typename D, int TC_MT, int NSB>
__global__ void __launch_bounds__(TC_WARPS * 32)
smallm_tc_k(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const block_a256 * __restrict__ y,
            int ncols, int nsb, int sb_per_split, float * __restrict__ out, int xflags) {
    using block = typename D::block;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t row0 = ((int64_t) blockIdx.x * TC_WARPS + warp) * (TC_MT * 16);
    const int split = blockIdx.y;
    const int sb0 = split * sb_per_split, sb1 = min(nsb, sb0 + sb_per_split);
    if (row0 >= rows) {
        return;
    }
    float acc[TC_MT][4];
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            acc[mt][r] = 0.0f;
        }
    }
    // B: column g (zero if g >= ncols); this thread's output columns are 2t and 2t+1
    const bool colg = g < ncols;
    const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
    const block_a256 * yg  = y + (int64_t) (colg ? g : 0) * nsb;
    const block_a256 * yc0 = y + (int64_t) (c0ok ? 2 * t : 0) * nsb;
    const block_a256 * yc1 = y + (int64_t) (c1ok ? 2 * t + 1 : 0) * nsb;
    const char * wr[TC_MT][2];
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int64_t r = row0 + mt * 16 + g + 8 * h;
            r = r < rows ? r : rows - 1;   // clamp: duplicate rows are computed and dropped at the store
            wr[mt][h] = W + r * row_bytes;
        }
    }

    for (int sbs = sb0; sbs < sb1; sbs += NSB) {
        // phase 1: the weight loads of NSB superblocks for every row of the warp
        typename D::R RR[NSB][TC_MT][2];
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (!(xflags & 2) && sbs + u < sb1) D::load((const block *) wr[mt][h] + sbs + u, t, RR[u][mt][h]);
                    else { memset(&RR[u][mt][h], 0x11 + mt + h, sizeof(RR[u][mt][h])); }   // bit 1: DELETE weight loads
                }
            }
        }
#pragma unroll
    for (int u = 0; u < NSB; ++u) {
        const int sb = sbs + u;
        if (sb >= sb1) {
            break;
        }
        auto & R = RR[u];
        uint2 bq[8];
        float dx0 = 0.0f, dx1 = 0.0f;
        uint4 s0 = make_uint4(0, 0, 0, 0), s1 = make_uint4(0, 0, 0, 0);   // int16 sums[8] for columns 2t, 2t+1
        uint32_t E0 = 0x0000FFFFu, E1 = 0x0000FFFFu;   // [ALPHA_SMALLM_TC_E] per-32 exponents; all 3 = no scaling
        if (!(xflags & 1)) {   // bit 0: DELETE activation loads
            const block_a256 & bg = yg[sb];
#pragma unroll
            for (int i = 0; i < 8; ++i) bq[i] = colg ? ld8(bg.qs + 32 * i + 8 * t) : make_uint2(0, 0);
            if (c0ok) { dx0 = yc0[sb].d; s0 = *(const uint4 *) yc0[sb].s; if (xflags & 16) E0 = *(const uint16_t *) yc0[sb].pad; }
            if (c1ok) { dx1 = yc1[sb].d; s1 = *(const uint4 *) yc1[sb].s; if (xflags & 16) E1 = *(const uint16_t *) yc1[sb].pad; }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) bq[i] = make_uint2(0x01020304u + i, 0x05060708u + sb);
            dx0 = dx1 = 0.5f;
        }
        if (xflags & 8) {   // bit 3: DELETE the math (loads only); fold the loaded words so the loads stay live
            uint32_t xo = bq[0].x ^ bq[7].y ^ s0.x ^ s1.w;
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t * pw = (const uint32_t *) &R[mt][h];
#pragma unroll
                    for (int k = 0; k < (int) (sizeof(R[mt][h]) / 4); ++k) xo ^= pw[k];
                }
            }
            acc[0][0] += (float) (xo & 0xFF);
            continue;
        }
        // phase 2: integer accumulation over the 8 sub-blocks, one float epilogue per superblock
        int ai[TC_MT][4], am[TC_MT][4];
#pragma unroll
        for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
            for (int o = 0; o < 4; ++o) {
                ai[mt][o] = 0;
                am[mt][o] = 0;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {   // 64-value chunk j = sub-blocks 2j and 2j+1
            const int bl[2] = { (int) bq[2 * j].x, (int) bq[2 * j].y };
            const int bh[2] = { (int) bq[2 * j + 1].x, (int) bq[2 * j + 1].y };
            // per-32 int sums of the activations, columns 2t (S0) and 2t+1 (S1), sub-blocks 2j (l) and 2j+1 (h)
            const uint32_t w0 = j == 0 ? s0.x : j == 1 ? s0.y : j == 2 ? s0.z : s0.w;
            const uint32_t w1 = j == 0 ? s1.x : j == 1 ? s1.y : j == 2 ? s1.z : s1.w;
            const int S0l = (int) (int16_t) (w0 & 0xFFFF), S0h = (int) (int16_t) (w0 >> 16);
            const int S1l = (int) (int16_t) (w1 & 0xFFFF), S1h = (int) (int16_t) (w1 >> 16);
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
                int lo0[2], hi0[2], lo1[2], hi1[2];
                int scl[2], ml[2], sch[2], mh[2];
                D::pair(R[mt][0], j, t, lo0, hi0, scl[0], ml[0], sch[0], mh[0]);   // row g
                D::pair(R[mt][1], j, t, lo1, hi1, scl[1], ml[1], sch[1], mh[1]);   // row g+8
                const int al[4] = { lo0[0], lo1[0], lo0[1], lo1[1] };
                const int ah[4] = { hi0[0], hi1[0], hi0[1], hi1[1] };
                int cl[4] = {0, 0, 0, 0}, ch[4] = {0, 0, 0, 0};
                mma_s8_16832(cl, al, bl);
                mma_s8_16832(ch, ah, bh);
#pragma unroll
                // [ALPHA_SMALLM_TC_E] column c's sub-block term times 2^(3 - e): a multiply, exact in int32. Without
                // the flag E0 = E1 = all-3 and every factor is 1 (the old arithmetic exactly)
                const int f0l = 1 << (3 - (int) ((E0 >> (4 * j)) & 3)), f0h = 1 << (3 - (int) ((E0 >> (4 * j + 2)) & 3));
                const int f1l = 1 << (3 - (int) ((E1 >> (4 * j)) & 3)), f1h = 1 << (3 - (int) ((E1 >> (4 * j + 2)) & 3));
#pragma unroll
                for (int h = 0; h < 2; ++h) {   // h=0: c[0..1] row g; h=1: c[2..3] row g+8
                    ai[mt][2*h + 0] += scl[h] * cl[2*h + 0] * f0l + sch[h] * ch[2*h + 0] * f0h;
                    ai[mt][2*h + 1] += scl[h] * cl[2*h + 1] * f1l + sch[h] * ch[2*h + 1] * f1h;
                    am[mt][2*h + 0] += ml[h] * S0l * f0l + mh[h] * S0h * f0h;
                    am[mt][2*h + 1] += ml[h] * S1l * f1l + mh[h] * S1h * f1h;
                }
            }
        }
#pragma unroll
        for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float2 f = D::dm(R[mt][h]);
                // COLUMN-POSITION INVARIANCE: every output column must get bit-identical arithmetic, because a verify
                // computes the same token in different column slots in different rounds. Written as two source
                // statements, nvcc contracted the even and odd column differently (measured: rotating the input
                // columns changed every output column, and greedy text then depended on the drafts). Explicit
                // __fmul_rn/__fmaf_rn are never re-contracted, so both columns run the same instruction sequence.
                acc[mt][2*h + 0] = col_epilogue(acc[mt][2*h + 0], dx0, f, ai[mt][2*h + 0], am[mt][2*h + 0]);
                acc[mt][2*h + 1] = col_epilogue(acc[mt][2*h + 1], dx1, f, ai[mt][2*h + 1], am[mt][2*h + 1]);
            }
        }
    }   // u
    }   // sbs
    float * o = out + (int64_t) split * ncols * rows;
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t r = row0 + mt * 16 + g + 8 * h;
            if (r < rows) {
                if (c0ok) o[(int64_t) (2 * t) * rows + r]     = acc[mt][2*h + 0];
                if (c1ok) o[(int64_t) (2 * t + 1) * rows + r] = acc[mt][2*h + 1];
            }
        }
    }
}

// ---- q6_K: 210-byte blocks (ql[128] qh[64] scales int8[16] d half), 16-value scale groups, 2-BYTE aligned.
// dequantize_row_q6_K: group gi (values 16gi..16gi+15) = half n = gi/8, quarter s = (gi%8)/2, l = 16*(gi%2) + [0,16):
// q = (ql[64n + l + 32*(s&1)] nibble (low if s < 2) | ((qh[32n + l] >> 2s) & 3) << 4) - 32, scale sc[gi].
// One mma.m16n8k16 per group: thread t owns elements 4t..4t+3 of each group (A: 1 reg per row, B: 4 bytes).
__device__ __forceinline__ void mma_s8_16816(int (&c)[4], const int (&a)[2], int b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    NO_DEVICE_CODE;
#endif
}

__device__ __forceinline__ uint32_t ld4_2(const uint8_t * p) {   // 4 bytes at 2-byte alignment
    const uint16_t * q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}

struct q6k_row {   // one row's superblock, this thread's slice: for half n and l-half h, ql at l, ql at l+32, qh at l
    uint32_t qa[2][2], qb[2][2], qh[2][2];
    int8_t   sc[16];
    float    d;
};

__device__ __forceinline__ void q6k_load(const block_q6_K * b, int t, q6k_row & r) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int l = 16 * h + 4 * t;
            r.qa[n][h] = ld4_2(b->ql + 64 * n + l);
            r.qb[n][h] = ld4_2(b->ql + 64 * n + 32 + l);
            r.qh[n][h] = ld4_2(b->qh + 32 * n + l);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint16_t v = ((const uint16_t *) b->scales)[i];
        r.sc[2 * i]     = (int8_t) (v & 0xFF);
        r.sc[2 * i + 1] = (int8_t) (v >> 8);
    }
    r.d = __half2float(b->d);
}

// int8 A register for group gi of this thread's slice: q - 32 per byte
__device__ __forceinline__ int q6k_group(const q6k_row & r, int gi) {
    const int n = gi / 8, s = (gi % 8) / 2, h = gi % 2;
    const uint32_t ql = (s & 1) ? r.qb[n][h] : r.qa[n][h];
    const uint32_t lo = (s < 2 ? ql : (ql >> 4)) & 0x0F0F0F0Fu;
    const uint32_t hi = ((r.qh[n][h] >> (2 * s)) & 0x03030303u) << 4;
    return (int) __vsubss4(lo | hi, 0x20202020u);
}

template <int TC_MT, int NSB>
__global__ void __launch_bounds__(TC_WARPS * 32)
smallm_tc_q6k(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const block_a256 * __restrict__ y,
              int ncols, int nsb, int sb_per_split, float * __restrict__ out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t row0 = ((int64_t) blockIdx.x * TC_WARPS + warp) * (TC_MT * 16);
    const int split = blockIdx.y;
    const int sb0 = split * sb_per_split, sb1 = min(nsb, sb0 + sb_per_split);
    if (row0 >= rows) {
        return;
    }
    float acc[TC_MT][4];
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mt][r] = 0.0f;
    }
    const bool colg = g < ncols;
    const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
    const block_a256 * yg  = y + (int64_t) (colg ? g : 0) * nsb;
    const block_a256 * yc0 = y + (int64_t) (c0ok ? 2 * t : 0) * nsb;
    const block_a256 * yc1 = y + (int64_t) (c1ok ? 2 * t + 1 : 0) * nsb;
    const char * wr[TC_MT][2];
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int64_t r = row0 + mt * 16 + g + 8 * h;
            r = r < rows ? r : rows - 1;
            wr[mt][h] = W + r * row_bytes;
        }
    }
    for (int sbs = sb0; sbs < sb1; sbs += NSB) {
        q6k_row RR[NSB][TC_MT][2];
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (sbs + u < sb1) q6k_load((const block_q6_K *) wr[mt][h] + sbs + u, t, RR[u][mt][h]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
            const int sb = sbs + u;
            if (sb >= sb1) {
                break;
            }
            float dx0 = 0.0f, dx1 = 0.0f;
            if (c0ok) dx0 = yc0[sb].d;
            if (c1ok) dx1 = yc1[sb].d;
            const int8_t * bqs = yg[sb].qs;
            int ai[TC_MT][4];
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
                for (int o = 0; o < 4; ++o) ai[mt][o] = 0;
            }
#pragma unroll
            for (int gi = 0; gi < 16; ++gi) {
                const int b = colg ? *(const int *) (bqs + 16 * gi + 4 * t) : 0;
#pragma unroll
                for (int mt = 0; mt < TC_MT; ++mt) {
                    const int a[2] = { q6k_group(RR[u][mt][0], gi), q6k_group(RR[u][mt][1], gi) };
                    int c[4] = {0, 0, 0, 0};
                    mma_s8_16816(c, a, b);
                    const int s0 = RR[u][mt][0].sc[gi], s1 = RR[u][mt][1].sc[gi];
                    ai[mt][0] += s0 * c[0];
                    ai[mt][1] += s0 * c[1];
                    ai[mt][2] += s1 * c[2];
                    ai[mt][3] += s1 * c[3];
                }
            }
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float dr = RR[u][mt][h].d;
                    acc[mt][2*h + 0] = col_epilogue(acc[mt][2*h + 0], dx0, make_float2(dr, 0.0f), ai[mt][2*h + 0], 0);
                    acc[mt][2*h + 1] = col_epilogue(acc[mt][2*h + 1], dx1, make_float2(dr, 0.0f), ai[mt][2*h + 1], 0);
                }
            }
        }
    }
    float * o = out + (int64_t) split * ncols * rows;
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t r = row0 + mt * 16 + g + 8 * h;
            if (r < rows) {
                if (c0ok) o[(int64_t) (2 * t) * rows + r]     = acc[mt][2*h + 0];
                if (c1ok) o[(int64_t) (2 * t + 1) * rows + r] = acc[mt][2*h + 1];
            }
        }
    }
}

// f32 activations -> block_a256: one warp per (superblock, column); lane l owns values 8l..8l+7 (so sub-block
// s = l/4, physical elements 8t..8t+7 with t = l%4 -- exactly the B-fragment slice thread t reads)
// [ALPHA_SMALLM_TC & 32] q8_0 weights (34 B blocks: fp16 d + 32 int8, so qs sit at a 2-byte offset).
//  * ACTIVATIONS PER 32 (planar: int8 qs [col][K], float d [col][K/32]), aligned to the weight blocks. q8_0 needs a
//    float epilogue per 32-value block anyway (its scales are fp16 and cannot fold into an integer accumulator), so
//    per-32 activation scales cost one multiply and give MMVQ's q8_1 accuracy; per-256 (block_a256) measured rel-L2
//    0.0068-0.0070 vs MMVQ 0.0049-0.0054 on 35B shapes (3090, 2026-10-18).
//  * TC_KW warps share one 16*MT-row tile and interleave its superblocks (warp w takes sb0+w, sb0+w+KW, ...), then
//    reduce through shared memory in FIXED warp order. v1 gave each warp its own rows: ~6 warps/SM at M=1 on a
//    2048x8192 tensor, 49.6 us vs MMVQ 28.5. The split depends only on nsb, never on ncols, so every column's sum is
//    the same at M=1 and M=8 -- plain decode and the verify can share the kernel (ALPHA_SMALLM_TC_MINCOLS=1).
// Bytes 8t..8t+7 of block j's qs. Row bases are 16-byte aligned (row_bytes = 272*nsb), so the qs address 34*j+2+8t is
// 2 mod 4 for even j and 0 mod 4 for odd j -- a compile-time parity, since j = 8*sb + i.
template <int ODD>
__device__ __forceinline__ uint2 ld8_q8_0(const char * blk, int t) {
    if (ODD) {
        const uint32_t * q = (const uint32_t *) (blk + 2 + 8 * t);
        return make_uint2(__ldg(q), __ldg(q + 1));
    }
    const uint32_t * q = (const uint32_t *) (blk + 8 * t);   // 4-aligned, 2 bytes before the span
    const uint32_t w0 = __ldg(q), w1 = __ldg(q + 1), w2 = __ldg(q + 2);
    return make_uint2(__funnelshift_r(w0, w1, 16), __funnelshift_r(w1, w2, 16));
}

constexpr int TC_KW = 8;   // warps per q8_0 tile

template <int TC_MT>
__global__ void __launch_bounds__(TC_KW * 32)
smallm_tc_q8_0(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const int8_t * __restrict__ xq,
               const float * __restrict__ xd, int64_t K, int ncols, int nsb, int sb_per_split, float * __restrict__ out) {
    __shared__ float red[TC_KW][TC_MT][4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t row0 = (int64_t) blockIdx.x * (TC_MT * 16);
    const int split = blockIdx.y;
    const int sb0 = split * sb_per_split, sb1 = min(nsb, sb0 + sb_per_split);
    float acc[TC_MT][4];
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mt][r] = 0.0f;
    }
    const bool colg = g < ncols;
    const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
    const int8_t * xg = xq + (int64_t) (colg ? g : 0) * K;
    const float *  d0 = xd + (int64_t) (c0ok ? 2 * t : 0) * (K / 32);
    const float *  d1 = xd + (int64_t) (c1ok ? 2 * t + 1 : 0) * (K / 32);
    const char * wr[TC_MT][2];
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int64_t r = row0 + mt * 16 + g + 8 * h;
            r = r < rows ? r : rows - 1;
            wr[mt][h] = W + r * row_bytes;
        }
    }
    for (int sb = sb0 + warp; sb < sb1; sb += TC_KW) {
        uint2 wq[TC_MT][2][8];
        float wd[TC_MT][2][8];
#pragma unroll
        for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const char * b = wr[mt][h] + (int64_t) (8 * sb + i) * 34;
                    wq[mt][h][i] = (i & 1) ? ld8_q8_0<1>(b, t) : ld8_q8_0<0>(b, t);
                    wd[mt][h][i] = __half2float(__ushort_as_half(__ldg((const unsigned short *) b)));
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int j = 8 * sb + i;
            const uint2 bq = colg ? *(const uint2 *) (xg + 32 * j + 8 * t) : make_uint2(0, 0);
            const int b[2] = { (int) bq.x, (int) bq.y };
            const float dx0 = c0ok ? __ldg(d0 + j) : 0.0f, dx1 = c1ok ? __ldg(d1 + j) : 0.0f;
#pragma unroll
            for (int mt = 0; mt < TC_MT; ++mt) {
                const int a[4] = { (int) wq[mt][0][i].x, (int) wq[mt][1][i].x, (int) wq[mt][0][i].y, (int) wq[mt][1][i].y };
                int c[4] = {0, 0, 0, 0};
                mma_s8_16832(c, a, b);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float dw = wd[mt][h][i];
                    acc[mt][2*h + 0] = col_epilogue(acc[mt][2*h + 0], dx0, make_float2(dw, 0.0f), c[2*h + 0], 0);
                    acc[mt][2*h + 1] = col_epilogue(acc[mt][2*h + 1], dx1, make_float2(dw, 0.0f), c[2*h + 1], 0);
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) red[warp][mt][r][lane] = acc[mt][r];
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            float s = red[0][mt][r][lane];
#pragma unroll
            for (int w = 1; w < TC_KW; ++w) s = __fadd_rn(s, red[w][mt][r][lane]);   // fixed order
            acc[mt][r] = s;
        }
    }
    float * o = out + (int64_t) split * ncols * rows;
#pragma unroll
    for (int mt = 0; mt < TC_MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t r = row0 + mt * 16 + g + 8 * h;
            if (r < rows) {
                if (c0ok) o[(int64_t) (2 * t) * rows + r]     = acc[mt][2*h + 0];
                if (c1ok) o[(int64_t) (2 * t + 1) * rows + r] = acc[mt][2*h + 1];
            }
        }
    }
}

// per-32 planar activation quantisation for the q8_0 path: one warp per 256 values of one column, 4 lanes per block
__global__ void smallm_quant_a32(const float * __restrict__ x, int64_t col_stride, int64_t K, int8_t * __restrict__ xq,
                                 float * __restrict__ xd) {
    const int sb = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    const float * xs = x + (int64_t) col * col_stride + (int64_t) sb * 256 + 8 * lane;
    float v[8];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = xs[k];
        amax = fmaxf(amax, fabsf(v[k]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) q[k] = (char) __float2int_rn(v[k] * id);
    *(uint2 *) (xq + (int64_t) col * K + (int64_t) sb * 256 + 8 * lane) = *(const uint2 *) q;
    if ((lane & 3) == 0) {
        xd[(int64_t) col * (K / 32) + sb * 8 + (lane >> 2)] = d;
    }
}

__global__ void smallm_quant_a256(const float * __restrict__ x, int64_t col_stride, int nsb, block_a256 * __restrict__ y) {
    const int sb = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    const float * xs = x + (int64_t) col * col_stride + (int64_t) sb * 256 + 8 * lane;
    float v[8];
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = xs[k];
        amax = fmaxf(amax, fabsf(v[k]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
    char q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int qi = __float2int_rn(v[k] * id);
        q[k] = (char) qi;
        sum += qi;
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    block_a256 & b = y[(int64_t) col * nsb + sb];
    *(uint2 *) (b.qs + 8 * lane) = *(const uint2 *) q;
    if ((lane & 3) == 0) {
        b.s[lane >> 2] = (int16_t) sum;
    }
    if (lane == 0) {
        b.d = d;
    }
}

// [ALPHA_SMALLM_TC_E=1] block_a256 with per-32 POWER-OF-TWO sub-scales. Sub-block j (lanes 4j..4j+3) is quantised at
// d / 2^e_j, e_j in 0..3 the largest with amax_j * 2^e_j <= amax, so |q| <= 127 still holds; the e_j live in pad[0..1]
// (2 bits each) and d is stored as d/8. The int8 kernel then accumulates sum_j (scl*c_j) * 2^(3-e_j) EXACTLY in int32
// (worst case iq4_xs ~1.06e9) and keeps ONE float epilogue per superblock. Per-256 activations (no sub-scales) cost
// the 27B exact configuration +0.17% Latin / +0.12% CJK perplexity (gx33, 2026-10-20): a quiet sub-block got the
// resolution of the loudest one in its 256.
__global__ void smallm_quant_a256e(const float * __restrict__ x, int64_t col_stride, int nsb, block_a256 * __restrict__ y) {
    const int sb = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    const float * xs = x + (int64_t) col * col_stride + (int64_t) sb * 256 + 8 * lane;
    float v[8];
    float amj = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = xs[k];
        amj = fmaxf(amj, fabsf(v[k]));
    }
    amj = fmaxf(amj, __shfl_xor_sync(0xffffffffu, amj, 1));
    amj = fmaxf(amj, __shfl_xor_sync(0xffffffffu, amj, 2));   // sub-block amax (lanes 4j..4j+3)
    float amax = amj;
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    int e = 0;
    while (e < 3 && __fmul_rn(amj, (float) (2 << e)) <= amax) ++e;
    const float d  = amax / 127.0f;
    const float id = d > 0.0f ? __fmul_rn(1.0f / d, (float) (1 << e)) : 0.0f;
    int sum = 0;
    char q[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int qi = max(-127, min(127, __float2int_rn(v[k] * id)));
        q[k] = (char) qi;
        sum += qi;
    }
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    uint32_t eb = (uint32_t) e << (2 * (lane >> 2));
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
        eb |= __shfl_xor_sync(0xffffffffu, eb, o);
    }
    block_a256 & b = y[(int64_t) col * nsb + sb];
    *(uint2 *) (b.qs + 8 * lane) = *(const uint2 *) q;
    if ((lane & 3) == 0) {
        b.s[lane >> 2] = (int16_t) sum;
    }
    if (lane == 0) {
        b.d = __fmul_rn(d, 0.125f);
        *(uint16_t *) b.pad = (uint16_t) eb;
    }
}

__global__ void smallm_tc_reduce(const float * __restrict__ part, int nsplit, int64_t n, float * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 0.0f;
        for (int k = 0; k < nsplit; ++k) {   // fixed order: deterministic
            s += part[(int64_t) k * n + i];
        }
        dst[i] = s;
    }
}

// ALPHA_SMALLM_TC: 0 off; 1 = all; or a bitmask with 2 = q4_K, 4 = q5_K, 8 = iq4_xs, 16 = q6_K, 32 = q8_0 to A/B per type
// [ALPHA_SMALLM_A16=1, with ALPHA_SMALLM_TC & 32] q8_0 x f16-activation tensor-core GEMV for 1..8 columns ("K2", plan
// 2026-10-18). On an RTX 5090: our verify round at full width (4 columns) is 4764 us vs NInfer's 4234; the dense
// Q8_0 MMVQ grows 1239 -> 1644 us/round from 1 to 4 columns, while NInfer's q8_a16 MMA is flat in M.
//  * ONE launch. Activations are read as the f32 src1 and converted to f16 in registers: no quantize_q8_1 launch
//    (125 us/round on the 5090) and no int8 activation error (f16 keeps 11 mantissa bits vs q8_1's 8).
//  * Weights: each warp stages its 16-row x 272-byte superblock slice through shared memory with 16-byte cp.async
//    (double-buffered), then extracts fragments with 4-byte LDS (the block parity trick of ld8_q8_0).
//  * int8 -> f16 is exact (0x6400|byte magic). Per 32-value block: two mma.m16n8k16 f16 -> f32 (products exact in
//    f32), then acc = fma(d_w, block, acc) in fixed order; the K split is across the CTA's warps by superblock and
//    reduced in fixed warp order in shared memory -- a function of K only, so a column's bits never depend on ncols.
constexpr int A16_NW = 4;                  // warps per CTA = K split
constexpr int A16_ROWB = 272;              // bytes of one q8_0 superblock row slice (8 blocks x 34 B)
constexpr int A16_STAGE = 16 * A16_ROWB;   // one staged tile: 16 rows x one superblock
constexpr int A16_P     = 4;               // cp.async pipeline depth (stages in flight)

__device__ __forceinline__ void a16_cp16(void * smem, const void * gmem) {
#if __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
#endif
}
__device__ __forceinline__ void a16_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
template <int N>
__device__ __forceinline__ void a16_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif
}

__device__ __forceinline__ void mma_f16_16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// 4 signed int8 (one 32-bit word) -> two exact half2: (b0,b1) and (b2,b3)
__device__ __forceinline__ void a16_i8x4_to_h2(uint32_t v, uint32_t & lo, uint32_t & hi) {
    const uint32_t u = v ^ 0x80808080u;                    // signed -> biased unsigned
    const uint32_t h01 = __byte_perm(u, 0x64646464u, 0x4140);
    const uint32_t h23 = __byte_perm(u, 0x64646464u, 0x4342);
    const half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152 = 1024 + 128
    const half2 r01 = __hsub2(*(const half2 *) &h01, bias);
    const half2 r23 = __hsub2(*(const half2 *) &h23, bias);
    lo = *(const uint32_t *) &r01;
    hi = *(const uint32_t *) &r23;
}

// the 8 qs bytes 8t..8t+7 of block i, from a 16-byte aligned staged row (block i at 34*i, qs at +2)
__device__ __forceinline__ uint2 a16_lds_q8(const char * row, int i, int t) {
    const char * b = row + 34 * i;
    if (i & 1) {   // 34*i + 2 + 8t is 0 mod 4
        const uint32_t * q = (const uint32_t *) (b + 2 + 8 * t);
        return make_uint2(q[0], q[1]);
    }
    const uint32_t * q = (const uint32_t *) (b + 8 * t);   // 2 bytes before the span, 4-aligned
    return make_uint2(__funnelshift_r(q[0], q[1], 16), __funnelshift_r(q[1], q[2], 16));
}

// parity-specialised forms (block index i is not a compile-time constant in the v2 kernel, its parity is)
__device__ __forceinline__ uint2 a16_lds_q8_odd(const char * row, int i, int t) {    // 34*i + 2 + 8t is 0 mod 4
    const uint32_t * q = (const uint32_t *) (row + 34 * i + 2 + 8 * t);
    return make_uint2(q[0], q[1]);
}
__device__ __forceinline__ uint2 a16_lds_q8_even(const char * row, int i, int t) {   // 2 bytes before the span
    const uint32_t * q = (const uint32_t *) (row + 34 * i + 8 * t);
    return make_uint2(__funnelshift_r(q[0], q[1], 16), __funnelshift_r(q[1], q[2], 16));
}

// v4 (2026-10-18): activations staged ONCE per CTA as f16 in shared memory, and a CTA walks T row tiles. v3 had every
// 16-row CTA re-read the whole activation vector from L2 per lane: the LM head's 15520 CTAs x 8 KB x M exceeded its
// 417 MB of weights at M=4 (3090 dry run: K3 913 us/launch vs MMVQ ~560-750, and growing with M). T is chosen from the
// SM count (enough CTAs to fill the GPU twice, then reuse). Arithmetic and order are v3's: bit-identical results.
// Kernel SELECTION depends on K only (never on M: exactness); the shared-memory SIZE follows ncols, which has no
// numeric effect (at M=1 a K=2048 CTA needs 23 KB, not the 51 KB of a TC_NCOLS-sized x region).
constexpr int A16_RED_BYTES = A16_NW * 4 * 32 * 4;

// column stride K + 32 halves (+64 B): with a K*2-byte stride (a multiple of 128 B) every column's loads hit the same
// banks -- an ncols-way conflict, v4 on the 3090: q8_0 2048x12288 44.9 us at M=1 -> 87.1 at M=8 (v3 42.6 -> 45.4)
constexpr int A16_XPAD = 32;
__host__ __device__ constexpr int64_t a16_x_stride(int64_t K) { return K + A16_XPAD; }
__host__ __device__ constexpr size_t a16_x_bytes(int ncols, int64_t K) { return ((size_t) ncols * (size_t) a16_x_stride(K) * 2 + 15) & ~(size_t) 15; }

// f32 activations [ncols][x_stride] -> f16 [TC_NCOLS][K] in shared memory (columns >= ncols stay unused)
__device__ __forceinline__ void a16_stage_x(half * xs, const float * __restrict__ x, int64_t x_stride, int ncols, int64_t K) {
    const int64_t n4 = (int64_t) ncols * (K / 4);
    for (int64_t idx = threadIdx.x; idx < n4; idx += blockDim.x) {
        const int64_t c = idx / (K / 4), e = (idx % (K / 4)) * 4;
        const float4 v = *(const float4 *) (x + c * x_stride + e);
        half2 * d = (half2 *) (xs + c * a16_x_stride(K) + e);
        d[0] = __floats2half2_rn(v.x, v.y);
        d[1] = __floats2half2_rn(v.z, v.w);
    }
}

__global__ void __launch_bounds__(A16_NW * 32)
smallm_a16_q8_0(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const float * __restrict__ x,
                int64_t x_stride, int ncols, int nsb, int T, float * __restrict__ out) {
    extern __shared__ __align__(16) char a16_sm[];
    const int64_t K = (int64_t) nsb * 256;
    // v5: no shared x region (v4's grew with ncols and cut occupancy: 2048x12288 42.7 us at M=1 -> 87.7 at M=8);
    // activations are read from global per lane as in v3, and the CTA's T tiles re-read them from L1, not L2
    float * red = (float *) a16_sm;
    char *  stg = (char *) red + A16_RED_BYTES;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t ntiles_all = (rows + 15) / 16;
    const int64_t tile0 = (int64_t) blockIdx.x * T;
    const int ntile = (int) (tile0 + T <= ntiles_all ? T : ntiles_all - tile0);
    const int total = ntile * nsb;
    auto stage = [&](int buf, int s) {
        const int64_t row0 = (tile0 + s / nsb) * 16;
        const int sb = s % nsb;
        char * dst = stg + buf * A16_STAGE;
        for (int idx = threadIdx.x; idx < 16 * (A16_ROWB / 16); idx += A16_NW * 32) {
            const int r = idx / (A16_ROWB / 16), c = idx % (A16_ROWB / 16);
            int64_t row = row0 + r;
            row = row < rows ? row : rows - 1;
            a16_cp16(dst + r * A16_ROWB + c * 16, W + row * row_bytes + (int64_t) sb * A16_ROWB + c * 16);
        }
        a16_commit();
    };
#pragma unroll
    for (int s0 = 0; s0 < A16_P - 1; ++s0) {
        if (s0 < total) stage(s0, s0); else a16_commit();
    }
    const bool colg = g < ncols;
    const float * xg = x + (int64_t) (colg ? g : 0) * x_stride;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < total; ++s) {
        const int buf = s % A16_P;
        const int sb = s % nsb;
        float4 xr[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = 2 * warp + j;
            if (colg) {
                xr[j][0] = *(const float4 *) (xg + (int64_t) sb * 256 + 32 * i + 8 * t);
                xr[j][1] = *(const float4 *) (xg + (int64_t) sb * 256 + 32 * i + 8 * t + 4);
            } else {
                xr[j][0] = xr[j][1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        if (s + A16_P - 1 < total) stage((s + A16_P - 1) % A16_P, s + A16_P - 1); else a16_commit();
        a16_wait<A16_P - 1>();
        __syncthreads();   // (first pass: also publishes the staged activations)
        const char * r_lo = stg + buf * A16_STAGE + g * A16_ROWB;   // row g
        const char * r_hi = r_lo + 8 * A16_ROWB;                    // row g + 8
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int i = 2 * warp + j;   // block parity == j
            const uint2 wl = j ? a16_lds_q8_odd(r_lo, i, t) : a16_lds_q8_even(r_lo, i, t);
            const uint2 wh = j ? a16_lds_q8_odd(r_hi, i, t) : a16_lds_q8_even(r_hi, i, t);
            const float dl = __half2float(*(const __half *) (r_lo + 34 * i));
            const float dh = __half2float(*(const __half *) (r_hi + 34 * i));
            uint32_t a0[4], a1[4];
            a16_i8x4_to_h2(wl.x, a0[0], a0[2]);
            a16_i8x4_to_h2(wh.x, a0[1], a0[3]);
            a16_i8x4_to_h2(wl.y, a1[0], a1[2]);
            a16_i8x4_to_h2(wh.y, a1[1], a1[3]);
            uint32_t b0[2], b1[2];
            {   // v3's conversion: x[8t..8t+7] of block i (zeros for g >= ncols)
                const float4 xa = xr[j][0], xb = xr[j][1];
                const half2 p0 = __floats2half2_rn(xa.x, xa.y), p1 = __floats2half2_rn(xa.z, xa.w);
                const half2 p2 = __floats2half2_rn(xb.x, xb.y), p3 = __floats2half2_rn(xb.z, xb.w);
                b0[0] = *(const uint32_t *) &p0; b0[1] = *(const uint32_t *) &p1;
                b1[0] = *(const uint32_t *) &p2; b1[1] = *(const uint32_t *) &p3;
            }
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            mma_f16_16816(c, a0, b0);
            mma_f16_16816(c, a1, b1);
            acc[0] = __fmaf_rn(dl, c[0], acc[0]);
            acc[1] = __fmaf_rn(dl, c[1], acc[1]);
            acc[2] = __fmaf_rn(dh, c[2], acc[2]);
            acc[3] = __fmaf_rn(dh, c[3], acc[3]);
        }
        if (sb == nsb - 1) {   // tile done: fixed-order reduce over the warps, store, reset
#pragma unroll
            for (int r = 0; r < 4; ++r) red[(warp * 4 + r) * 32 + lane] = acc[r];
            __syncthreads();
            if (warp == 0) {
                float o[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float v = red[r * 32 + lane];
#pragma unroll
                    for (int w = 1; w < A16_NW; ++w) v = __fadd_rn(v, red[(w * 4 + r) * 32 + lane]);
                    o[r] = v;
                }
                const int64_t row0 = (tile0 + s / nsb) * 16;
                const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int64_t r = row0 + g + 8 * h;
                    if (r < rows) {
                        if (c0ok) out[(int64_t) (2 * t) * rows + r]     = o[2 * h + 0];
                        if (c1ok) out[(int64_t) (2 * t + 1) * rows + r] = o[2 * h + 1];
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[r] = 0.0f;
        }
        __syncthreads();   // the stage read here is restaged A16_P - 1 iterations on; red is reused next tile
    }
}

// [ALPHA_SMALLM_A16=1, with ALPHA_SMALLM_TC & 16] q6_K x f16-activation tensor-core GEMV for 1..8 columns ("K3",
// plan 2026-10-18): the 35B LM head (248320 x 2048 Q6_K). On an RTX 5090: the verify head costs 269 us/round at 1
// column and 462 at 4, NInfer's q6_a16 MMA 252. Same discipline as smallm_a16_q8_0:
//  * one mma.m16n8k16 per 16-value scale group, so the group's int8 scale applies once, in the f32 epilogue:
//    acc = fma(d * sc, group_dot, acc) in fixed group order (q x sc reaches 4096, beyond f16's exact range);
//  * thread t owns the group's PHYSICAL values 4t..4t+3, mapped to logical k {2t,2t+1,2t+8,2t+9}; its activations
//    are then one float4 (x at the same physical positions);
//  * 6-bit values from ql/qh with 32-bit ops, (q+32) -> f16 by the exact 0x6400 magic, minus 1056;
//  * the 16-row tile is staged WHOLE (row_bytes must be a multiple of 16: 8 superblocks = 1680 B, K % 2048 == 0);
//    the CTA's 4 warps split the superblocks (warp w takes nsb/4 consecutive ones), fixed-order smem reduce.
constexpr int A16Q6_NW = 4;

__device__ __forceinline__ uint32_t a16_ld4_par(const uint8_t * p, int odd) {   // 4 bytes, p 4-aligned or 2 mod 4
    if (!odd) return *(const uint32_t *) p;
    const uint32_t * q = (const uint32_t *) (p - 2);
    return __funnelshift_r(q[0], q[1], 16);
}

// 4 unsigned bytes u (each 0..63 = q + 32) -> two exact half2 of q
__device__ __forceinline__ void a16_u6x4_to_h2(uint32_t u, uint32_t & lo, uint32_t & hi) {
    const uint32_t h01 = __byte_perm(u, 0x64646464u, 0x4140);
    const uint32_t h23 = __byte_perm(u, 0x64646464u, 0x4342);
    const half2 bias = __halves2half2(__ushort_as_half(0x6420), __ushort_as_half(0x6420));   // 1056 = 1024 + 32
    const half2 r01 = __hsub2(*(const half2 *) &h01, bias);
    const half2 r23 = __hsub2(*(const half2 *) &h23, bias);
    lo = *(const uint32_t *) &r01;
    hi = *(const uint32_t *) &r23;
}

// NBUF: tile buffers. 2 overlaps the next tile's load with compute but costs 54 KB per CTA: on the 3090 (100 KB of
// shared memory per SM) that is ONE 4-warp CTA per SM (v5: 2048x32768 157 us vs MMVQ 82-101); 1 buffer = 3 CTAs/SM.
template <int NBUF>
__global__ void __launch_bounds__(A16Q6_NW * 32)
smallm_a16_q6_K(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const float * __restrict__ x,
                int64_t x_stride, int ncols, int nsb, int T, float * __restrict__ out) {
    // v4: f16 activations staged once per CTA (shared by its T tiles), whole 16-row tiles double-buffered.
    extern __shared__ __align__(16) char a16q6_sm[];
    const int64_t K = (int64_t) nsb * 256;
    float * red = (float *) a16q6_sm;   // v5: no shared x region (see the q8_0 kernel)
    char *  stg = (char *) red + A16_RED_BYTES;
    const int64_t tile_bytes = 16 * row_bytes;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t ntiles_all = (rows + 15) / 16;
    const int64_t tile0 = (int64_t) blockIdx.x * T;
    const int ntile = (int) (tile0 + T <= ntiles_all ? T : ntiles_all - tile0);
    const int per_row = (int) (row_bytes / 16);
    auto stage = [&](int buf, int tt) {
        const int64_t row0 = (tile0 + tt) * 16;
        char * dst = stg + buf * tile_bytes;
        for (int idx = threadIdx.x; idx < 16 * per_row; idx += A16Q6_NW * 32) {
            const int r = idx / per_row, c = idx % per_row;
            int64_t row = row0 + r;
            row = row < rows ? row : rows - 1;
            a16_cp16(dst + (int64_t) r * row_bytes + c * 16, W + row * row_bytes + c * 16);
        }
        a16_commit();
    };
    stage(0, 0);
    const bool colg = g < ncols;
    const float * xg = x + (int64_t) (colg ? g : 0) * x_stride;
    const int per_warp = nsb / A16Q6_NW;
    for (int tt = 0; tt < ntile; ++tt) {
        const int buf = NBUF == 2 ? (tt & 1) : 0;
        if (NBUF == 2) {
            if (tt + 1 < ntile) { stage(buf ^ 1, tt + 1); a16_wait<1>(); } else { a16_wait<0>(); }
        } else {
            if (tt > 0) stage(0, tt);   // tile 0 was staged before the loop
            a16_wait<0>();
        }
        __syncthreads();
        const uint8_t * r_lo = (const uint8_t *) (stg + buf * tile_bytes) + (int64_t) g * row_bytes;
        const uint8_t * r_hi = r_lo + 8 * row_bytes;
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int bb = 0; bb < per_warp; ++bb) {
            const int b = warp * per_warp + bb;
            const int odd = b & 1;
            const uint8_t * bl = r_lo + 210 * b;
            const uint8_t * bh = r_hi + 210 * b;
            const float dl = __half2float(*(const __half *) (bl + 208));
            const float dh = __half2float(*(const __half *) (bh + 208));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int is = 0; is < 2; ++is) {
                    const int l = 16 * is + 4 * t;
                    const uint32_t qhl = a16_ld4_par(bl + 128 + 32 * n + l, odd), qhh = a16_ld4_par(bh + 128 + 32 * n + l, odd);
                    const uint32_t qa_l = a16_ld4_par(bl + 64 * n + l, odd),      qa_h = a16_ld4_par(bh + 64 * n + l, odd);
                    const uint32_t qb_l = a16_ld4_par(bl + 64 * n + l + 32, odd), qb_h = a16_ld4_par(bh + 64 * n + l + 32, odd);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t ql_l = (k & 1) ? qb_l : qa_l, ql_h = (k & 1) ? qb_h : qa_h;
                        const uint32_t nl = (k < 2 ? ql_l : (ql_l >> 4)) & 0x0F0F0F0Fu;
                        const uint32_t nh = (k < 2 ? ql_h : (ql_h >> 4)) & 0x0F0F0F0Fu;
                        const uint32_t ul = nl | (((qhl >> (2 * k)) & 0x03030303u) << 4);
                        const uint32_t uh = nh | (((qhh >> (2 * k)) & 0x03030303u) << 4);
                        uint32_t a[4];
                        a16_u6x4_to_h2(ul, a[0], a[2]);
                        a16_u6x4_to_h2(uh, a[1], a[3]);
                        uint32_t bv[2] = { 0u, 0u };
                        if (colg) {
                            const float4 xv = *(const float4 *) (xg + (int64_t) b * 256 + 128 * n + 32 * k + 16 * is + 4 * t);
                            const half2 p0 = __floats2half2_rn(xv.x, xv.y), p1 = __floats2half2_rn(xv.z, xv.w);
                            bv[0] = *(const uint32_t *) &p0; bv[1] = *(const uint32_t *) &p1;
                        }
                        float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        mma_f16_16816(c, a, bv);
                        const int sci = 8 * n + is + 2 * k;
                        const float sl = dl * (float) (int8_t) bl[192 + sci], sh = dh * (float) (int8_t) bh[192 + sci];
                        acc[0] = __fmaf_rn(sl, c[0], acc[0]);
                        acc[1] = __fmaf_rn(sl, c[1], acc[1]);
                        acc[2] = __fmaf_rn(sh, c[2], acc[2]);
                        acc[3] = __fmaf_rn(sh, c[3], acc[3]);
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) red[(warp * 4 + r) * 32 + lane] = acc[r];
        __syncthreads();
        if (warp == 0) {
            float o[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                float v = red[r * 32 + lane];
#pragma unroll
                for (int w = 1; w < A16Q6_NW; ++w) v = __fadd_rn(v, red[(w * 4 + r) * 32 + lane]);
                o[r] = v;
            }
            const int64_t row0 = (tile0 + tt) * 16;
            const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int64_t r = row0 + g + 8 * h;
                if (r < rows) {
                    if (c0ok) out[(int64_t) (2 * t) * rows + r]     = o[2 * h + 0];
                    if (c1ok) out[(int64_t) (2 * t + 1) * rows + r] = o[2 * h + 1];
                }
            }
        }
        __syncthreads();   // stage buffer and red are reused by the next tile
    }
}

// [ALPHA_SMALLM_A16_KQ=1, with ALPHA_SMALLM_A16=1] K-quant x f16-activation tensor-core GEMV for 1..8 columns:
// q4_K, q5_K, iq4_xs and q6_K at ANY K with nsb % 4 == 0 (the K == 2048 q6_K head kernel above is untouched).
// WHY. The int8 K-quant path quantises activations per 256 values (block_a256): rel-L2 vs f64 0.0068-0.0071 against
// MMVQ's 0.0050, and on the 27B exact configuration (plain + verify on smallm_tc, 2026-10-20 gx33) +0.17% Latin /
// +0.12% CJK perplexity. f16 activations (11 mantissa bits) remove that error, as they did for q8_0 (rel-L2 0.0002).
// DESIGN (the q8_0 / q6_K A16 discipline):
//  * every weight value is an exact small integer in f16 (q4 0..15, q5 0..31, q6 -32..31, iq4 table -127..113);
//    one mma.m16n8k16 pair per 32-value sub-block (q6_K: one per 16-value group), f32 accumulate;
//  * the sub-block scale (and min: d*sc*q - dmin*m) applies in a FIXED f32 epilogue; the min term needs the
//    column sum of x over the sub-block, taken by the same mma with an all-ones A (per column, so width-invariant);
//  * a pipeline stage is 4 CONSECUTIVE superblocks of a 16-row tile (cp.async), warp w takes superblock w of each
//    stage: the K split is a function of K only; the 4 warps reduce in fixed order in shared memory.
constexpr int A16K_NW = 4;
constexpr int A16K_P  = 3;   // pipeline depth

__device__ __forceinline__ void a16_cp8(void * smem, const void * gmem) {
#if __CUDA_ARCH__ >= 800
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(s), "l"(gmem));
#endif
}

// 4 unsigned bytes (each 0..255) -> two exact half2: (b0,b1) and (b2,b3)
__device__ __forceinline__ void a16_u8x4_to_h2(uint32_t u, uint32_t & lo, uint32_t & hi) {
    const uint32_t h01 = __byte_perm(u, 0x64646464u, 0x4140);
    const uint32_t h23 = __byte_perm(u, 0x64646464u, 0x4342);
    const half2 bias = __halves2half2(__ushort_as_half(0x6400), __ushort_as_half(0x6400));   // 1024
    const half2 r01 = __hsub2(*(const half2 *) &h01, bias);
    const half2 r23 = __hsub2(*(const half2 *) &h23, bias);
    lo = *(const uint32_t *) &r01;
    hi = *(const uint32_t *) &r23;
}

// x[e..e+7] of column g as the two B fragments (zeros for g >= ncols)
__device__ __forceinline__ void a16k_b(const float * xg, bool colg, int64_t e, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    if (!colg) { b0[0] = b0[1] = b1[0] = b1[1] = 0u; return; }
    const float4 xa = *(const float4 *) (xg + e), xb = *(const float4 *) (xg + e + 4);
    const half2 p0 = __floats2half2_rn(xa.x, xa.y), p1 = __floats2half2_rn(xa.z, xa.w);
    const half2 p2 = __floats2half2_rn(xb.x, xb.y), p3 = __floats2half2_rn(xb.z, xb.w);
    b0[0] = *(const uint32_t *) &p0; b0[1] = *(const uint32_t *) &p1;
    b1[0] = *(const uint32_t *) &p2; b1[1] = *(const uint32_t *) &p3;
}

// one 32-value sub-block: wl/wh = the 8 physical values 8t..8t+7 of rows g / g+8 as unsigned or signed bytes;
// sl/sh the rows' scales, ml/mh their min factors (MIN types only)
template <bool SIGNED, bool MIN>
__device__ __forceinline__ void a16k_sub(float (&acc)[4], uint2 wl, uint2 wh, const uint32_t (&b0)[2],
                                         const uint32_t (&b1)[2], float sl, float sh, float ml, float mh) {
    uint32_t a0[4], a1[4];
    if (SIGNED) {
        a16_i8x4_to_h2(wl.x, a0[0], a0[2]); a16_i8x4_to_h2(wh.x, a0[1], a0[3]);
        a16_i8x4_to_h2(wl.y, a1[0], a1[2]); a16_i8x4_to_h2(wh.y, a1[1], a1[3]);
    } else {
        a16_u8x4_to_h2(wl.x, a0[0], a0[2]); a16_u8x4_to_h2(wh.x, a0[1], a0[3]);
        a16_u8x4_to_h2(wl.y, a1[0], a1[2]); a16_u8x4_to_h2(wh.y, a1[1], a1[3]);
    }
    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    mma_f16_16816(c, a0, b0);
    mma_f16_16816(c, a1, b1);
    acc[0] = __fmaf_rn(sl, c[0], acc[0]);
    acc[1] = __fmaf_rn(sl, c[1], acc[1]);
    acc[2] = __fmaf_rn(sh, c[2], acc[2]);
    acc[3] = __fmaf_rn(sh, c[3], acc[3]);
    if (MIN) {
        const uint32_t one = 0x3C003C00u;
        const uint32_t ao[4] = { one, one, one, one };
        float sx[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        mma_f16_16816(sx, ao, b0);
        mma_f16_16816(sx, ao, b1);
        acc[0] = __fmaf_rn(-ml, sx[0], acc[0]);
        acc[1] = __fmaf_rn(-ml, sx[1], acc[1]);
        acc[2] = __fmaf_rn(-mh, sx[2], acc[2]);
        acc[3] = __fmaf_rn(-mh, sx[3], acc[3]);
    }
}

// q4_K / q5_K scale and min of sub-block j from the 12 scale bytes (ggml get_scale_min_k4)
__device__ __forceinline__ void a16k_scmin(const uint8_t * q, int j, int & sc, int & m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else       { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// TYPE: 0 q4_K (144 B), 1 q5_K (176 B), 2 iq4_xs (136 B), 3 q6_K (210 B). CP: cp.async chunk bytes (16, or 8 for q6_K).
template <int TYPE, int BB, int CP>
__global__ void __launch_bounds__(A16K_NW * 32)
smallm_a16_kq(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const float * __restrict__ x,
              int64_t x_stride, int ncols, int nsb, int T, float * __restrict__ out) {
    extern __shared__ __align__(16) char a16k_sm[];
    constexpr int SROW = 4 * BB;              // staged bytes per row and stage
    constexpr int STAGE = 16 * SROW;
    float * red = (float *) a16k_sm;
    char *  stg = (char *) red + A16_RED_BYTES;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int nst = nsb / 4;                  // stages per tile
    const int64_t ntiles_all = (rows + 15) / 16;
    const int64_t tile0 = (int64_t) blockIdx.x * T;
    const int ntile = (int) (tile0 + T <= ntiles_all ? T : ntiles_all - tile0);
    const int total = ntile * nst;
    auto stage = [&](int buf, int s) {
        const int64_t row0 = (tile0 + s / nst) * 16;
        const int st = s % nst;
        char * dst = stg + buf * STAGE;
        for (int idx = threadIdx.x; idx < 16 * (SROW / CP); idx += A16K_NW * 32) {
            const int r = idx / (SROW / CP), c = idx % (SROW / CP);
            int64_t row = row0 + r;
            row = row < rows ? row : rows - 1;
            const char * src = W + row * row_bytes + (int64_t) st * SROW + c * CP;
            if (CP == 16) a16_cp16(dst + r * SROW + c * CP, src); else a16_cp8(dst + r * SROW + c * CP, src);
        }
        a16_commit();
    };
#pragma unroll
    for (int s0 = 0; s0 < A16K_P - 1; ++s0) {
        if (s0 < total) stage(s0, s0); else a16_commit();
    }
    const bool colg = g < ncols;
    const float * xg = x + (int64_t) (colg ? g : 0) * x_stride;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int s = 0; s < total; ++s) {
        const int buf = s % A16K_P;
        const int st = s % nst;
        if (s + A16K_P - 1 < total) stage((s + A16K_P - 1) % A16K_P, s + A16K_P - 1); else a16_commit();
        a16_wait<A16K_P - 1>();
        __syncthreads();
        const uint8_t * bl = (const uint8_t *) (stg + buf * STAGE + g * SROW + warp * BB);   // row g, superblock warp
        const uint8_t * bh = bl + 8 * SROW;                                                 // row g + 8
        const int64_t xb = (int64_t) (st * 4 + warp) * 256;                                  // its first x index
        if (TYPE == 0 || TYPE == 1) {   // q4_K / q5_K: d, dmin | scales[12] | (q5: qh[32]) | qs[128]
            const float2 fl = __half22float2(*(const half2 *) bl), fh = __half22float2(*(const half2 *) bh);
            constexpr int QS = TYPE == 0 ? 16 : 48;
            uint2 hl = make_uint2(0u, 0u), hh = make_uint2(0u, 0u);
            if (TYPE == 1) { hl = *(const uint2 *) (bl + 16 + 8 * t); hh = *(const uint2 *) (bh + 16 + 8 * t); }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint2 wl = *(const uint2 *) (bl + QS + 32 * m + 8 * t);
                const uint2 wh = *(const uint2 *) (bh + QS + 32 * m + 8 * t);
#pragma unroll
                for (int hb = 0; hb < 2; ++hb) {   // sub-block 2m (low nibbles), 2m+1 (high nibbles)
                    const int j = 2 * m + hb;
                    uint2 ql = make_uint2((wl.x >> (4 * hb)) & 0x0F0F0F0Fu, (wl.y >> (4 * hb)) & 0x0F0F0F0Fu);
                    uint2 qh = make_uint2((wh.x >> (4 * hb)) & 0x0F0F0F0Fu, (wh.y >> (4 * hb)) & 0x0F0F0F0Fu);
                    if (TYPE == 1) {   // 5th bit: qh[l] bit j
                        ql.x |= ((hl.x >> j) & 0x01010101u) << 4; ql.y |= ((hl.y >> j) & 0x01010101u) << 4;
                        qh.x |= ((hh.x >> j) & 0x01010101u) << 4; qh.y |= ((hh.y >> j) & 0x01010101u) << 4;
                    }
                    int scl, mnl, sch, mnh;
                    a16k_scmin(bl + 4, j, scl, mnl);
                    a16k_scmin(bh + 4, j, sch, mnh);
                    uint32_t b0[2], b1[2];
                    a16k_b(xg, colg, xb + 32 * j + 8 * t, b0, b1);
                    a16k_sub<false, true>(acc, ql, qh, b0, b1, __fmul_rn(fl.x, (float) scl), __fmul_rn(fh.x, (float) sch),
                                          __fmul_rn(fl.y, (float) mnl), __fmul_rn(fh.y, (float) mnh));
                }
            }
        } else if (TYPE == 2) {   // iq4_xs: d | scales_h | scales_l[4] | qs[128]
            constexpr uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
            const float dl = __half2float(*(const __half *) bl), dh = __half2float(*(const __half *) bh);
            const uint32_t shl = *(const uint16_t *) (bl + 2), shh = *(const uint16_t *) (bh + 2);
            const uint32_t sll = *(const uint32_t *) (bl + 4), slh = *(const uint32_t *) (bh + 4);
            const int sh = 4 * (t >> 1);
#pragma unroll
            for (int ib = 0; ib < 8; ++ib) {
                const uint2 rl = *(const uint2 *) (bl + 8 + 16 * ib + 8 * (t & 1));
                const uint2 rh = *(const uint2 *) (bh + 8 + 16 * ib + 8 * (t & 1));
                const uint2 vl = make_uint2((uint32_t) lut16((rl.x >> sh) & 0x0F0F0F0Fu, T0, T1, T2, T3),
                                            (uint32_t) lut16((rl.y >> sh) & 0x0F0F0F0Fu, T0, T1, T2, T3));
                const uint2 vh = make_uint2((uint32_t) lut16((rh.x >> sh) & 0x0F0F0F0Fu, T0, T1, T2, T3),
                                            (uint32_t) lut16((rh.y >> sh) & 0x0F0F0F0Fu, T0, T1, T2, T3));
                const int scl = (int) (((sll >> (8 * (ib / 2) + 4 * (ib % 2))) & 0xF) | (((shl >> (2 * ib)) & 3) << 4)) - 32;
                const int sch = (int) (((slh >> (8 * (ib / 2) + 4 * (ib % 2))) & 0xF) | (((shh >> (2 * ib)) & 3) << 4)) - 32;
                uint32_t b0[2], b1[2];
                a16k_b(xg, colg, xb + 32 * ib + 8 * t, b0, b1);
                a16k_sub<true, false>(acc, vl, vh, b0, b1, __fmul_rn(dl, (float) scl), __fmul_rn(dh, (float) sch), 0.0f, 0.0f);
            }
        } else {   // q6_K: ql[128] qh[64] scales int8[16] d; the K == 2048 head kernel's group loop, per superblock
            const int odd = warp & 1;   // 210*warp is 2 mod 4 for odd warps (stage rows are 8-aligned)
            const float dl = __half2float(*(const __half *) (bl + 208));
            const float dh = __half2float(*(const __half *) (bh + 208));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int is = 0; is < 2; ++is) {
                    const int l = 16 * is + 4 * t;
                    const uint32_t qhl = a16_ld4_par(bl + 128 + 32 * n + l, odd), qhh = a16_ld4_par(bh + 128 + 32 * n + l, odd);
                    const uint32_t qa_l = a16_ld4_par(bl + 64 * n + l, odd),      qa_h = a16_ld4_par(bh + 64 * n + l, odd);
                    const uint32_t qb_l = a16_ld4_par(bl + 64 * n + l + 32, odd), qb_h = a16_ld4_par(bh + 64 * n + l + 32, odd);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t ql_l = (k & 1) ? qb_l : qa_l, ql_h = (k & 1) ? qb_h : qa_h;
                        const uint32_t nl = (k < 2 ? ql_l : (ql_l >> 4)) & 0x0F0F0F0Fu;
                        const uint32_t nh = (k < 2 ? ql_h : (ql_h >> 4)) & 0x0F0F0F0Fu;
                        const uint32_t ul = nl | (((qhl >> (2 * k)) & 0x03030303u) << 4);
                        const uint32_t uh = nh | (((qhh >> (2 * k)) & 0x03030303u) << 4);
                        uint32_t a[4];
                        a16_u6x4_to_h2(ul, a[0], a[2]);
                        a16_u6x4_to_h2(uh, a[1], a[3]);
                        uint32_t bv[2] = { 0u, 0u };
                        if (colg) {
                            const float4 xv = *(const float4 *) (xg + xb + 128 * n + 32 * k + 16 * is + 4 * t);
                            const half2 p0 = __floats2half2_rn(xv.x, xv.y), p1 = __floats2half2_rn(xv.z, xv.w);
                            bv[0] = *(const uint32_t *) &p0; bv[1] = *(const uint32_t *) &p1;
                        }
                        float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        mma_f16_16816(c, a, bv);
                        const int sci = 8 * n + is + 2 * k;
                        const float sl = __fmul_rn(dl, (float) (int8_t) bl[192 + sci]), shh = __fmul_rn(dh, (float) (int8_t) bh[192 + sci]);
                        acc[0] = __fmaf_rn(sl, c[0], acc[0]);
                        acc[1] = __fmaf_rn(sl, c[1], acc[1]);
                        acc[2] = __fmaf_rn(shh, c[2], acc[2]);
                        acc[3] = __fmaf_rn(shh, c[3], acc[3]);
                    }
                }
            }
        }
        if (st == nst - 1) {   // tile done: fixed-order reduce over the warps, store, reset
#pragma unroll
            for (int r = 0; r < 4; ++r) red[(warp * 4 + r) * 32 + lane] = acc[r];
            __syncthreads();
            if (warp == 0) {
                float o[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    float v = red[r * 32 + lane];
#pragma unroll
                    for (int w = 1; w < A16K_NW; ++w) v = __fadd_rn(v, red[(w * 4 + r) * 32 + lane]);
                    o[r] = v;
                }
                const int64_t row0 = (tile0 + s / nst) * 16;
                const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int64_t r = row0 + g + 8 * h;
                    if (r < rows) {
                        if (c0ok) out[(int64_t) (2 * t) * rows + r]     = o[2 * h + 0];
                        if (c1ok) out[(int64_t) (2 * t + 1) * rows + r] = o[2 * h + 1];
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[r] = 0.0f;
        }
        __syncthreads();   // this stage buffer is restaged A16K_P - 1 iterations on; red is reused next tile
    }
}

// [ALPHA_SMALLM_A16_KQ=2] v2 of the K-quant f16-activation GEMV, on the int8 path's skeleton (smallm_tc_k): no shared
// memory staging, each warp owns 16 rows and loads NSB superblocks per row per step straight from global (the
// contiguity lesson of 2026-10-03), K split across blocks by the int8 path's rule (rows, nsb, SM count -- never
// ncols) with the same fixed-order reduce. v1 (smem-staged, 4 warps/CTA, 29-42 KB) ran ~9 warps/SM: 1.5-2x MMVQ.
//  * weights: the int8 path's decoders (D::load / D::pair) give each lane the bytes of physical elements 8t..8t+7
//    of sub-blocks 2j / 2j+1 for rows g and g+8, plus their INTEGER scale and min; bytes -> f16 exactly;
//  * activations: f32 src1 -> f16 in registers (column g), as the q8_0 A16 kernel;
//  * one mma.m16n8k16 pair per 32-value sub-block, then acc = fma(d*sc, c, acc) [+ fma(-dmin*m, sum_x, acc)] in a
//    fixed order; sum_x is the column's f32 sum over the sub-block from a FIXED shuffle tree (v1 spent a second
//    mma pair per sub-block on it). Every step depends on the column alone, so a column's bits never depend on ncols.
template <typename D, bool SIGNED, bool MIN, int NSB>
__global__ void __launch_bounds__(TC_WARPS * 32)
smallm_a16k2(const char * __restrict__ W, int64_t row_bytes, int64_t rows, const float * __restrict__ x,
             int64_t x_stride, int ncols, int nsb, int sb_per_split, float * __restrict__ out) {
    using block = typename D::block;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int64_t row0 = ((int64_t) blockIdx.x * TC_WARPS + warp) * 16;
    const int split = blockIdx.y;
    const int sb0 = split * sb_per_split, sb1 = min(nsb, sb0 + sb_per_split);
    if (row0 >= rows) {
        return;
    }
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const bool colg = g < ncols;
    const bool c0ok = 2 * t < ncols, c1ok = 2 * t + 1 < ncols;
    const float * xg = x + (int64_t) (colg ? g : 0) * x_stride;
    const char * wr[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        int64_t r = row0 + g + 8 * h;
        r = r < rows ? r : rows - 1;   // clamp: duplicate rows are computed and dropped at the store
        wr[h] = W + r * row_bytes;
    }
    for (int sbs = sb0; sbs < sb1; sbs += NSB) {
        typename D::R RR[NSB][2];
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (sbs + u < sb1) D::load((const block *) wr[h] + sbs + u, t, RR[u][h]);
                else memset(&RR[u][h], 0, sizeof(RR[u][h]));
            }
        }
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
            const int sb = sbs + u;
            if (sb >= sb1) {
                break;
            }
            const float2 fl = D::dm(RR[u][0]), fh = D::dm(RR[u][1]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {   // 64-value chunk j = sub-blocks 2j and 2j+1
                int lo0[2], hi0[2], lo1[2], hi1[2];
                int scl0, ml0, sch0, mh0, scl1, ml1, sch1, mh1;
                D::pair(RR[u][0], j, t, lo0, hi0, scl0, ml0, sch0, mh0);   // row g
                D::pair(RR[u][1], j, t, lo1, hi1, scl1, ml1, sch1, mh1);   // row g+8
#pragma unroll
                for (int hb = 0; hb < 2; ++hb) {   // sub-block 2j (lo) / 2j+1 (hi)
                    const int sub = 2 * j + hb;
                    const int * wg = hb ? hi0 : lo0;
                    const int * wh = hb ? hi1 : lo1;
                    float4 xa = make_float4(0.0f, 0.0f, 0.0f, 0.0f), xb = xa;
                    if (colg) {
                        xa = *(const float4 *) (xg + (int64_t) sb * 256 + 32 * sub + 8 * t);
                        xb = *(const float4 *) (xg + (int64_t) sb * 256 + 32 * sub + 8 * t + 4);
                    }
                    uint32_t b0[2], b1[2];
                    {
                        const half2 p0 = __floats2half2_rn(xa.x, xa.y), p1 = __floats2half2_rn(xa.z, xa.w);
                        const half2 p2 = __floats2half2_rn(xb.x, xb.y), p3 = __floats2half2_rn(xb.z, xb.w);
                        b0[0] = *(const uint32_t *) &p0; b0[1] = *(const uint32_t *) &p1;
                        b1[0] = *(const uint32_t *) &p2; b1[1] = *(const uint32_t *) &p3;
                    }
                    uint32_t a0[4], a1[4];
                    if (SIGNED) {
                        a16_i8x4_to_h2((uint32_t) wg[0], a0[0], a0[2]); a16_i8x4_to_h2((uint32_t) wh[0], a0[1], a0[3]);
                        a16_i8x4_to_h2((uint32_t) wg[1], a1[0], a1[2]); a16_i8x4_to_h2((uint32_t) wh[1], a1[1], a1[3]);
                    } else {
                        a16_u8x4_to_h2((uint32_t) wg[0], a0[0], a0[2]); a16_u8x4_to_h2((uint32_t) wh[0], a0[1], a0[3]);
                        a16_u8x4_to_h2((uint32_t) wg[1], a1[0], a1[2]); a16_u8x4_to_h2((uint32_t) wh[1], a1[1], a1[3]);
                    }
                    float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                    mma_f16_16816(c, a0, b0);
                    mma_f16_16816(c, a1, b1);
                    const float sg = __fmul_rn(fl.x, (float) (hb ? sch0 : scl0));
                    const float sh = __fmul_rn(fh.x, (float) (hb ? sch1 : scl1));
                    acc[0] = __fmaf_rn(sg, c[0], acc[0]);
                    acc[1] = __fmaf_rn(sg, c[1], acc[1]);
                    acc[2] = __fmaf_rn(sh, c[2], acc[2]);
                    acc[3] = __fmaf_rn(sh, c[3], acc[3]);
                    if (MIN) {
                        // column g's sum over the sub-block: this lane's 8 values, then the quad (t) in a fixed tree
                        float ps = __fadd_rn(__fadd_rn(__fadd_rn(xa.x, xa.y), __fadd_rn(xa.z, xa.w)),
                                             __fadd_rn(__fadd_rn(xb.x, xb.y), __fadd_rn(xb.z, xb.w)));
                        ps = __fadd_rn(ps, __shfl_xor_sync(0xffffffffu, ps, 1));
                        ps = __fadd_rn(ps, __shfl_xor_sync(0xffffffffu, ps, 2));
                        // columns 2t and 2t+1 live in quads 2t and 2t+1 (lanes 8t and 8t+4)
                        const float s0 = __shfl_sync(0xffffffffu, ps, 8 * t);
                        const float s1 = __shfl_sync(0xffffffffu, ps, 8 * t + 4);
                        const float mg = __fmul_rn(fl.y, (float) (hb ? mh0 : ml0));
                        const float mhh = __fmul_rn(fh.y, (float) (hb ? mh1 : ml1));
                        acc[0] = __fmaf_rn(-mg, s0, acc[0]);
                        acc[1] = __fmaf_rn(-mg, s1, acc[1]);
                        acc[2] = __fmaf_rn(-mhh, s0, acc[2]);
                        acc[3] = __fmaf_rn(-mhh, s1, acc[3]);
                    }
                }
            }
        }
    }
    float * o = out + (int64_t) split * ncols * rows;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int64_t r = row0 + g + 8 * h;
        if (r < rows) {
            if (c0ok) o[(int64_t) (2 * t) * rows + r]     = acc[2 * h + 0];
            if (c1ok) o[(int64_t) (2 * t + 1) * rows + r] = acc[2 * h + 1];
        }
    }
}

// [ALPHA_SMALLM_STUB=1] DELETION BOUND (rental A, plan 2026-10-18): reads every weight byte with 16-byte loads and
// writes a finite garbage result, in ONE launch and with no activation quantisation -- the in-engine floor any
// dense kernel of this shape could reach. Output is WRONG by design: time it, never use its text.
__global__ void __launch_bounds__(256)
smallm_stub_stream(const char * __restrict__ W, int64_t row_bytes, int64_t rows, int ncols, float * __restrict__ out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t row = (int64_t) blockIdx.x * 8 + warp;
    if (row >= rows) return;
    const char * p = W + row * row_bytes;
    uint32_t h = 0;
    if ((row_bytes & 15) == 0) {
        const uint4 * q = (const uint4 *) p;
        for (int64_t k = lane; k < row_bytes / 16; k += 32) {
            const uint4 v = __ldg(q + k);
            h ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    } else {
        const uint32_t * q = (const uint32_t *) p;   // row_bytes is a multiple of 2 at least: tail bytes ignored
        for (int64_t k = lane; k < row_bytes / 4; k += 32) h ^= __ldg(q + k);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) h ^= __shfl_xor_sync(0xffffffffu, h, o);
    if (lane < ncols) out[(int64_t) lane * rows + row] = (float) (h & 0xff) * 1e-6f;
}

int smallm_tc_mode() {
    static const int m = []() {
        const char * e = getenv("ALPHA_SMALLM_TC");
        const int v = e ? atoi(e) : 0;
        // 16 = q6_K is NOT in the default: measured 2026-10-03 (3090, M=8, 5120x32768) 293 us vs MMQ 233 and less
        // accurate (rel-L2 0.0069 vs 0.0054): 2-byte-aligned loads, 16 k16 mmas per superblock, 168-255 registers
        return v == 1 ? (2 | 4 | 8) : v;
    }();
    return m;
}

// ALPHA_SMALLM_TC_WARPS_PER_SM: split K until about this many warps per SM are launched. Default 4: the wide
// projections then run unsplit (no reduce launch at all) and only tall-K ones split 2-way. 3090 nsys 2026-10-03 at
// the old default 32: 328 reduce launches per DFlash2 round = 1.16 ms (3.5 us each), for kernels no faster.
int smallm_tc_warps_per_sm() {
    static const int m = []() {
        const char * e = getenv("ALPHA_SMALLM_TC_WARPS_PER_SM");
        return e ? atoi(e) : 4;
    }();
    return m;
}

// ALPHA_SMALLM_TC_MINCOLS: the narrowest product this kernel takes (default 2). 1 also routes plain decode (M = 1)
// here, so a verify and a plain step compute every column with the SAME kernel: the exactness route for exact MTP.
int smallm_tc_mincols() {
    static const int m = []() {
        const char * e = getenv("ALPHA_SMALLM_TC_MINCOLS");
        const int v = e ? atoi(e) : 2;
        return v < 1 ? 1 : v;
    }();
    return m;
}

// ALPHA_SMALLM_TC_CENSUS=1: every call on a COVERED type counted by (type, src1 ne[1] x ne[2], verdict), printed at
// exit. Under graph replay the hook runs only at capture, so this counts captured nodes, not replays. Exactness needs
// plain (M = 1) and the verify (M = k+1) to take the same kernel for every covered tensor: a decline on one side
// only shows here as an "accept" row without its "declined" twin.
struct Census {
    std::map<std::string, long> n;
    ~Census() {
        if (n.empty()) return;
        fprintf(stderr, "[smallm-tc] CENSUS (type ne1xne2 verdict: calls)\n");
        for (const auto & kv : n) fprintf(stderr, "[smallm-tc]   %-40s %ld\n", kv.first.c_str(), kv.second);
    }
};
Census * smallm_tc_census() {
    static Census * c = getenv("ALPHA_SMALLM_TC_CENSUS") && atoi(getenv("ALPHA_SMALLM_TC_CENSUS")) ? new Census : nullptr;
    static bool reg = c && atexit([]() { delete smallm_tc_census(); }) == 0;
    (void) reg;
    return c;
}
void smallm_tc_count(const ggml_tensor * src0, const ggml_tensor * src1, const char * verdict) {
    Census * c = smallm_tc_census();
    if (!c) return;
    char k[128];
    snprintf(k, sizeof k, "%s %lldx%lld [%lld x %lld] %s", ggml_type_name(src0->type), (long long) src1->ne[1],
             (long long) src1->ne[2], (long long) src0->ne[0], (long long) src0->ne[1], verdict);
    ++c->n[k];
}

}  // namespace

// [ALPHA_SMALLM_A16_MINROWS] (default 4096) with the A16 kernels on, products with fewer output rows stay on MMVQ
// (and keep the fork's fused M=1 gate+up+GLU). 3090 engine census (v5, 35B exact): the shared expert's 512-row
// products took 9.5 us on A16 vs 5.0 on MMVQ (32 row tiles cannot fill the GPU), +440 us/round. MMVQ is already
// width-invariant over the 1..4 columns exact MTP uses, so a SHAPE-based split keeps exact == plain.
static bool smallm_a16_on() {
    static const bool on = getenv("ALPHA_SMALLM_A16") && atoi(getenv("ALPHA_SMALLM_A16")) != 0;
    return on;
}
static int64_t smallm_a16_minrows() {
    static const int64_t m = getenv("ALPHA_SMALLM_A16_MINROWS") ? atoll(getenv("ALPHA_SMALLM_A16_MINROWS")) : 4096;
    return m;
}

// nullptr = this kernel takes the product; else the decline reason. "uncovered" = type not in ALPHA_SMALLM_TC.
static const char * smallm_tc_decline(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int device) {
    const int mode = smallm_tc_mode();
    const ggml_type tt = src0->type;
    const bool covered = mode != 0 && ((tt == GGML_TYPE_Q6_K && (mode & 16)) ||
                         (tt == GGML_TYPE_Q4_K && (mode & 2)) || (tt == GGML_TYPE_Q5_K && (mode & 4)) ||
                         (tt == GGML_TYPE_IQ4_XS && (mode & 8)) || (tt == GGML_TYPE_Q8_0 && (mode & 32)));
    const int64_t K = src0->ne[0], ncols = src1->ne[1];
    return !covered ? "uncovered"
         : src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ? "declined:type"
         : ncols < smallm_tc_mincols() ? "declined:mincols"
         : ncols > TC_NCOLS ? "declined:wide"
         : K % QK_K != 0 ? "declined:K"
         : src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 ? "declined:batched"
         : !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) ? "declined:noncontig"
         : ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_AMPERE ? "declined:cc"
         : ((uintptr_t) src0->data & 15) || ((uintptr_t) src1->data & 15) || (src1->nb[1] & 15) ? "declined:align"
         : smallm_a16_on() && src0->ne[1] < smallm_a16_minrows() ? "declined:a16small"
         : nullptr;
}

// The fork's graph loop fuses gate+up+GLU (and mul_mat+bias) into ONE MMVQ launch at ncols == 1, deciding BEFORE
// ggml_cuda_mul_mat, so that hook never sees those products. With ALPHA_SMALLM_TC_MINCOLS=1 that left plain decode
// (M = 1) on fused MMVQ for the 35B shared expert's gate/up while the verify (M = 4, never fused) took this kernel:
// exact != plain from the first decoded position (3090 xb1, 2026-10-18; control without the kernel 119/119). The
// fusion check now asks this predicate and declines what this kernel would take, so both widths run it unfused.
bool ggml_cuda_smallm_tc_covers(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return smallm_tc_decline(src0, src1, dst, ggml_cuda_get_device()) == nullptr;
}

bool ggml_cuda_mul_mat_smallm_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                 ggml_tensor * dst) {
    if (smallm_tc_mode() == 0) {
        return false;
    }
    const ggml_type tt = src0->type;
    const int64_t K = src0->ne[0], rows = src0->ne[1], ncols = src1->ne[1];
    const char * why = smallm_tc_decline(src0, src1, dst, ctx.device);
    if (why && why[0] == 'u') {
        return false;
    }
    smallm_tc_count(src0, src1, why ? why : "ACCEPT");
    if (why) {
        return false;
    }
    cudaStream_t stream = ctx.stream();
    const int nsb = (int) (K / QK_K);
    static const bool stub = getenv("ALPHA_SMALLM_STUB") && atoi(getenv("ALPHA_SMALLM_STUB")) != 0;
    static const bool a16  = getenv("ALPHA_SMALLM_A16") && atoi(getenv("ALPHA_SMALLM_A16")) != 0;
    if (stub) {
        smallm_stub_stream<<<(unsigned) ((rows + 7) / 8), 256, 0, stream>>>((const char *) src0->data,
            (int64_t) src0->nb[1], rows, (int) ncols, (float *) dst->data);
        static int shown_stub = 0;
        if (shown_stub < 3) {
            ++shown_stub;
            fprintf(stderr, "[smallm-tc] STUB (deletion bound, WRONG OUTPUT) %s [%lld x %lld] M=%lld\n", ggml_type_name(tt),
                    (long long) K, (long long) rows, (long long) ncols);
        }
        return true;
    }
    // tiles per CTA: enough CTAs to fill the GPU twice, then reuse the staged activations (v4)
    const int64_t a16_tiles = (rows + 15) / 16;
    const int a16_nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const int a16_T = (int) std::max<int64_t>(1, std::min<int64_t>(16, a16_tiles / (2 * (int64_t) a16_nsm)));
    const unsigned a16_grid = (unsigned) ((a16_tiles + a16_T - 1) / a16_T);
    if (tt == GGML_TYPE_Q8_0 && a16) {
        const size_t smem = (size_t) A16_RED_BYTES + (size_t) A16_P * A16_STAGE;
        static bool attr = false;
        if (!attr) { attr = true; cudaFuncSetAttribute(smallm_a16_q8_0, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024); }
        smallm_a16_q8_0<<<a16_grid, A16_NW * 32, smem, stream>>>((const char *) src0->data,
            (int64_t) src0->nb[1], rows, (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)),
            (int) ncols, nsb, a16_T, (float *) dst->data);
        static int shown_a16 = 0;
        if (shown_a16 < 6) {
            ++shown_a16;
            fprintf(stderr, "[smallm-tc] ACTIVE q8_0 A16 [%lld x %lld] M=%lld ctas=%u T=%d\n", (long long) K,
                    (long long) rows, (long long) ncols, a16_grid, a16_T);
        }
        return true;
    }
    if (tt == GGML_TYPE_Q6_K && a16 && K == 2048 && (src0->nb[1] % 16) == 0) {   // the 35B LM head shape (smem bound)
        // double-buffer only where shared memory per SM allows several such CTAs (sm_12x: 228 KB); 1 buffer elsewhere
        const bool dbl = ggml_cuda_info().devices[ctx.device].cc >= 1200;   // ggml cc encoding: sm_120 = 1200, sm_86 = 860
        const size_t smem = (size_t) A16_RED_BYTES + (size_t) (dbl ? 2 : 1) * 16 * src0->nb[1];
        static bool attr = false;
        if (!attr) {
            attr = true;
            cudaFuncSetAttribute(smallm_a16_q6_K<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            cudaFuncSetAttribute(smallm_a16_q6_K<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        }
        if (dbl) {
            smallm_a16_q6_K<2><<<a16_grid, A16Q6_NW * 32, smem, stream>>>((const char *) src0->data, (int64_t) src0->nb[1],
                rows, (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), (int) ncols, nsb, a16_T, (float *) dst->data);
        } else {
            smallm_a16_q6_K<1><<<a16_grid, A16Q6_NW * 32, smem, stream>>>((const char *) src0->data, (int64_t) src0->nb[1],
                rows, (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), (int) ncols, nsb, a16_T, (float *) dst->data);
        }
        static int shown_q6 = 0;
        if (shown_q6 < 3) {
            ++shown_q6;
            fprintf(stderr, "[smallm-tc] ACTIVE q6_K A16 [%lld x %lld] M=%lld ctas=%u T=%d\n", (long long) K, (long long) rows,
                    (long long) ncols, a16_grid, a16_T);
        }
        return true;
    }
    // [ALPHA_SMALLM_A16_KQ=1] the K-quant f16-activation kernel (any K with nsb % 4 == 0); chosen by type and K only
    static const int a16kq = getenv("ALPHA_SMALLM_A16_KQ") ? atoi(getenv("ALPHA_SMALLM_A16_KQ")) : 0;   // 1 = v1, 2 = v2
    // v2 (ALPHA_SMALLM_A16_KQ=2): q4_K / q5_K / iq4_xs on the direct-load kernel, the int8 path's grid and K split
    if (a16 && a16kq == 2 && (tt == GGML_TYPE_Q4_K || tt == GGML_TYPE_Q5_K || tt == GGML_TYPE_IQ4_XS)) {
        constexpr int NSB2 = 2;
        const int64_t nblocks2 = (rows + TC_WARPS * 16 - 1) / (TC_WARPS * 16);
        const int nsm2 = ggml_cuda_info().devices[ctx.device].nsm;
        int nsplit2 = 1;
        while (nblocks2 * TC_WARPS * nsplit2 < (int64_t) nsm2 * smallm_tc_warps_per_sm() && nsplit2 * 2 <= nsb && nsplit2 < 32) {
            nsplit2 *= 2;
        }
        int sbps2 = (nsb + nsplit2 - 1) / nsplit2;
        sbps2 = (sbps2 + NSB2 - 1) / NSB2 * NSB2;
        nsplit2 = (nsb + sbps2 - 1) / sbps2;
        float * out2 = (float *) dst->data;
        ggml_cuda_pool_alloc<float> part2(ctx.pool(), nsplit2 > 1 ? (size_t) nsplit2 * ncols * rows : 1);
        float * o2 = nsplit2 > 1 ? part2.get() : out2;
        const dim3 grid2((unsigned) nblocks2, (unsigned) nsplit2);
        const char * W2 = (const char *) src0->data;
        const int64_t rb2 = (int64_t) src0->nb[1];
        const float * x2 = (const float *) src1->data;
        const int64_t xs2 = (int64_t) (src1->nb[1] / sizeof(float));
        switch (tt) {
            case GGML_TYPE_Q4_K: smallm_a16k2<dec_q4_K, false, true, NSB2><<<grid2, TC_WARPS * 32, 0, stream>>>(W2, rb2, rows, x2, xs2, (int) ncols, nsb, sbps2, o2); break;
            case GGML_TYPE_Q5_K: smallm_a16k2<dec_q5_K, false, true, NSB2><<<grid2, TC_WARPS * 32, 0, stream>>>(W2, rb2, rows, x2, xs2, (int) ncols, nsb, sbps2, o2); break;
            default:             smallm_a16k2<dec_iq4_xs, true, false, NSB2><<<grid2, TC_WARPS * 32, 0, stream>>>(W2, rb2, rows, x2, xs2, (int) ncols, nsb, sbps2, o2); break;
        }
        if (nsplit2 > 1) {
            const int64_t n = ncols * rows;
            smallm_tc_reduce<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(part2.get(), nsplit2, n, out2);
        }
        static int shown_k2 = 0;
        if (shown_k2 < 8) {
            ++shown_k2;
            fprintf(stderr, "[smallm-tc] ACTIVE %s A16-KQ2 [%lld x %lld] M=%lld split=%d blocks=%lld\n", ggml_type_name(tt),
                    (long long) K, (long long) rows, (long long) ncols, nsplit2, (long long) nblocks2);
        }
        return true;
    }
    if (a16 && a16kq && nsb % 4 == 0 && (tt == GGML_TYPE_Q4_K || tt == GGML_TYPE_Q5_K || tt == GGML_TYPE_IQ4_XS ||
                                         tt == GGML_TYPE_Q6_K) && (src0->nb[1] % (tt == GGML_TYPE_Q6_K ? 8 : 16)) == 0 &&
        ((uintptr_t) src0->data % 16) == 0) {
        const int bb = tt == GGML_TYPE_Q4_K ? 144 : tt == GGML_TYPE_Q5_K ? 176 : tt == GGML_TYPE_IQ4_XS ? 136 : 210;
        const size_t smem = (size_t) A16_RED_BYTES + (size_t) A16K_P * 16 * 4 * bb;
        static bool attr = false;
        if (!attr) {
            attr = true;
            cudaFuncSetAttribute(smallm_a16_kq<0, 144, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            cudaFuncSetAttribute(smallm_a16_kq<1, 176, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            cudaFuncSetAttribute(smallm_a16_kq<2, 136, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
            cudaFuncSetAttribute(smallm_a16_kq<3, 210, 8>,  cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024);
        }
        const char * Wp = (const char *) src0->data;
        const int64_t rbp = (int64_t) src0->nb[1];
        const float * xp = (const float *) src1->data;
        const int64_t xs = (int64_t) (src1->nb[1] / sizeof(float));
        float * op = (float *) dst->data;
        switch (tt) {
            case GGML_TYPE_Q4_K:   smallm_a16_kq<0, 144, 16><<<a16_grid, A16K_NW * 32, smem, stream>>>(Wp, rbp, rows, xp, xs, (int) ncols, nsb, a16_T, op); break;
            case GGML_TYPE_Q5_K:   smallm_a16_kq<1, 176, 16><<<a16_grid, A16K_NW * 32, smem, stream>>>(Wp, rbp, rows, xp, xs, (int) ncols, nsb, a16_T, op); break;
            case GGML_TYPE_IQ4_XS: smallm_a16_kq<2, 136, 16><<<a16_grid, A16K_NW * 32, smem, stream>>>(Wp, rbp, rows, xp, xs, (int) ncols, nsb, a16_T, op); break;
            default:               smallm_a16_kq<3, 210, 8><<<a16_grid, A16K_NW * 32, smem, stream>>>(Wp, rbp, rows, xp, xs, (int) ncols, nsb, a16_T, op); break;
        }
        static int shown_kq = 0;
        if (shown_kq < 8) {
            ++shown_kq;
            fprintf(stderr, "[smallm-tc] ACTIVE %s A16-KQ [%lld x %lld] M=%lld ctas=%u T=%d\n", ggml_type_name(tt), (long long) K,
                    (long long) rows, (long long) ncols, a16_grid, a16_T);
        }
        return true;
    }
    if (tt == GGML_TYPE_Q8_0) {
        ggml_cuda_pool_alloc<int8_t> xq(ctx.pool(), (size_t) (ncols * K));
        ggml_cuda_pool_alloc<float>  xd(ctx.pool(), (size_t) (ncols * K / 32));
        smallm_quant_a32<<<dim3((unsigned) nsb, (unsigned) ncols), 32, 0, stream>>>(
            (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), K, xq.get(), xd.get());
        // ALPHA_SMALLM_TC_Q8_MT: rows per tile / 16 (1 default, or 2). K is split across blocks only when the tiles
        // (x TC_KW warps) leave the GPU short of warps; the fixed-order reduce then costs one more launch.
        static const int MTq = getenv("ALPHA_SMALLM_TC_Q8_MT") && atoi(getenv("ALPHA_SMALLM_TC_Q8_MT")) == 2 ? 2 : 1;
        const int64_t ntile = (rows + MTq * 16 - 1) / (MTq * 16);
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        int nsplit = 1;
        while (ntile * TC_KW * nsplit < (int64_t) nsm * smallm_tc_warps_per_sm() && nsplit * 2 * TC_KW <= nsb && nsplit < 32) {
            nsplit *= 2;
        }
        const int sb_per_split = (nsb + nsplit - 1) / nsplit;
        nsplit = (nsb + sb_per_split - 1) / sb_per_split;
        float * out = (float *) dst->data;
        ggml_cuda_pool_alloc<float> part(ctx.pool(), nsplit > 1 ? (size_t) nsplit * ncols * rows : 1);
        float * o = nsplit > 1 ? part.get() : out;
        const dim3 grid((unsigned) ntile, (unsigned) nsplit);
        const char * W = (const char *) src0->data;
        const int64_t rb = (int64_t) src0->nb[1];
        if (MTq == 2) {
            smallm_tc_q8_0<2><<<grid, TC_KW * 32, 0, stream>>>(W, rb, rows, xq.get(), xd.get(), K, (int) ncols, nsb, sb_per_split, o);
        } else {
            smallm_tc_q8_0<1><<<grid, TC_KW * 32, 0, stream>>>(W, rb, rows, xq.get(), xd.get(), K, (int) ncols, nsb, sb_per_split, o);
        }
        if (nsplit > 1) {
            const int64_t n = ncols * rows;
            smallm_tc_reduce<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(part.get(), nsplit, n, out);
        }
        static int shown_q8 = 0;
        if (shown_q8 < 6) {
            ++shown_q8;
            fprintf(stderr, "[smallm-tc] ACTIVE q8_0 [%lld x %lld] M=%lld split=%d tiles=%lld kw=%d\n", (long long) K,
                    (long long) rows, (long long) ncols, nsplit, (long long) ntile, TC_KW);
        }
        return true;
    }
    ggml_cuda_pool_alloc<block_a256> ya(ctx.pool(), (size_t) ncols * nsb);
    // [ALPHA_SMALLM_TC_E=1] per-32 power-of-two sub-scales for the q4_K / q5_K / iq4_xs kernel (q6_K's has no shifts)
    static const bool tc_e = getenv("ALPHA_SMALLM_TC_E") && atoi(getenv("ALPHA_SMALLM_TC_E")) != 0;
    const bool use_e = tc_e && tt != GGML_TYPE_Q6_K;
    if (use_e) {
        smallm_quant_a256e<<<dim3((unsigned) nsb, (unsigned) ncols), 32, 0, stream>>>(
            (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), nsb, ya.get());
    } else {
        smallm_quant_a256<<<dim3((unsigned) nsb, (unsigned) ncols), 32, 0, stream>>>(
            (const float *) src1->data, (int64_t) (src1->nb[1] / sizeof(float)), nsb, ya.get());
    }

    // ALPHA_SMALLM_TC_CFG = MT*10 + NSB (rows per warp / 16, superblocks per step): 12 (default), 21, 14, 22, 11.
    // 3090 M=8 sweep 2026-10-03, 5120x17408: q4_K 21: 125 us, 11: 104, 12: 79-84, 14: 83-92, 22: 85-95 (MMQ 102-110).
    // The step must read a long CONTIGUOUS run per row: with one superblock (144 B) per row per step the kernel was
    // load-bound at ~480 GB/s, and loads-only time equalled the full time.
    static const int cfg = getenv("ALPHA_SMALLM_TC_CFG") ? atoi(getenv("ALPHA_SMALLM_TC_CFG")) : 12;
    const int MTc = cfg / 10, NSBc = cfg % 10;
    const int64_t rows_per_block = (int64_t) TC_WARPS * MTc * 16;
    const int64_t nblocks = (rows + rows_per_block - 1) / rows_per_block;
    // split K until there are enough warps to keep the memory system busy
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    int nsplit = 1;
    while (nblocks * TC_WARPS * nsplit < (int64_t) nsm * smallm_tc_warps_per_sm() && nsplit * 2 <= nsb && nsplit < 32) {
        nsplit *= 2;
    }
    int sb_per_split = (nsb + nsplit - 1) / nsplit;
    sb_per_split = (sb_per_split + NSBc - 1) / NSBc * NSBc;   // whole steps per split
    nsplit = (nsb + sb_per_split - 1) / sb_per_split;
    float * out = (float *) dst->data;
    ggml_cuda_pool_alloc<float> part(ctx.pool(), nsplit > 1 ? (size_t) nsplit * ncols * rows : 1);
    float * o = nsplit > 1 ? part.get() : out;
    const dim3 grid((unsigned) nblocks, (unsigned) nsplit);
    const char * W = (const char *) src0->data;
    const int64_t rb = (int64_t) src0->nb[1];
    const int xf = (getenv("ALPHA_SMALLM_TC_X") ? atoi(getenv("ALPHA_SMALLM_TC_X")) : 0) | (use_e ? 16 : 0);   // bisection bits | E
    auto launch = [&](auto dec) {
        using D = decltype(dec);
        const block_a256 * ya_ = ya.get();
        switch (cfg) {
            case 14: smallm_tc_k<D, 1, 4><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya_, (int) ncols, nsb, sb_per_split, o, xf); break;
            case 12: smallm_tc_k<D, 1, 2><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya_, (int) ncols, nsb, sb_per_split, o, xf); break;
            case 22: smallm_tc_k<D, 2, 2><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya_, (int) ncols, nsb, sb_per_split, o, xf); break;
            case 11: smallm_tc_k<D, 1, 1><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya_, (int) ncols, nsb, sb_per_split, o, xf); break;
            default: smallm_tc_k<D, 2, 1><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya_, (int) ncols, nsb, sb_per_split, o, xf); break;
        }
    };
    switch (tt) {
        case GGML_TYPE_Q6_K:
            if (MTc == 2) {   // rows per warp must match the grid sizing above
                smallm_tc_q6k<2, 2><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya.get(), (int) ncols, nsb, sb_per_split, o);
            } else {
                smallm_tc_q6k<1, 2><<<grid, TC_WARPS * 32, 0, stream>>>(W, rb, rows, ya.get(), (int) ncols, nsb, sb_per_split, o);
            }
            break;
        case GGML_TYPE_Q4_K: launch(dec_q4_K{}); break;
        case GGML_TYPE_Q5_K: launch(dec_q5_K{}); break;
        default:             launch(dec_iq4_xs{}); break;
    }
    if (nsplit > 1) {
        const int64_t n = ncols * rows;
        smallm_tc_reduce<<<(unsigned) ((n + 255) / 256), 256, 0, stream>>>(part.get(), nsplit, n, out);
    }
    static int shown = 0;
    if (shown < 6) {
        ++shown;
        fprintf(stderr, "[smallm-tc] ACTIVE %s [%lld x %lld] M=%lld split=%d blocks=%lld" "\n", ggml_type_name(tt),
                (long long) K, (long long) rows, (long long) ncols, nsplit, (long long) nblocks);
    }
    return true;
}
