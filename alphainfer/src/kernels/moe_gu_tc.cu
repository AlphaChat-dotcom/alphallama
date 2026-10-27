// MTP/DFlash-verify MoE gate/up + SwiGLU on tensor cores for T = 2..8 tokens (AI_MOE_TC=1; fast verify only).
//
// The expert FFN is the largest term of a width-8 verify. moe_gu_verify (dp4a, owner dedup) is bit-exact but does
// not scale to 8 tokens, since one owner warp does up to 8x the dp4a work.
//
// As in moe_gu_verify, the lowest token routed to an expert owns it (two warp ballots cover GU_NS * T <= 64
// (slot, token) entries); the owner's columns are every token sharing the expert, at most 8 = the N of
// mma.sync m16n8k32 s8. So each expert's gate and up weights are read and unpacked once and all of its tokens ride
// one mma. The weight decoding, the K permutation and the per-256 activation format (one scale per superblock, per-32
// integer sums for the min term, a single float epilogue per superblock) follow ggml's dense small-M tensor-core
// kernel (ggml-cuda/smallm-tc.cu, ALPHA_SMALLM_TC).
//
// Not bit-identical to plain decode (MMVQ dp4a over q8_1 per 32): fast verify only, never in --mtp-exact.
// Column-position invariant: every output column runs the same instruction sequence (col_epilogue), so a token's
// result does not depend on which column it lands in.
#include "kernels/kernels.h"

#include "ggml-cuda/common.cuh"

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int TC_NS    = 8;   // routed slots (n_expert_used)
constexpr int TC_MAXT  = 8;   // tokens per verify round (= mma N)
constexpr int TC_WARPS = 4;   // warps per block, 16 rows each

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

__device__ __forceinline__ float col_epilogue(float acc, float dx, float2 f, int ai, int am) {
    const float t = __fmaf_rn(f.x, (float) ai, -__fmul_rn(f.y, (float) am));
    return __fmaf_rn(dx, t, acc);
}

__device__ __forceinline__ uint2 ld8(const void * p) { return *(const uint2 *) p; }

__device__ __forceinline__ int byte_of(const uint4 & w, int i) {
    const uint32_t v = i < 4 ? w.x : i < 8 ? w.y : i < 12 ? w.z : w.w;
    return (int) ((v >> (8 * (i & 3))) & 0xFF);
}
__device__ __forceinline__ void scale_min_k4_r(int j, const uint4 & h, int & sc, int & m) {
    if (j < 4) {
        sc = byte_of(h, 4 + j) & 63;
        m  = byte_of(h, 4 + j + 4) & 63;
    } else {
        sc = (byte_of(h, 4 + j + 4) & 0xF) | ((byte_of(h, 4 + j - 4) >> 6) << 4);
        m  = (byte_of(h, 4 + j + 4) >> 4) | ((byte_of(h, 4 + j) >> 6) << 4);
    }
}

struct dec_q4_K {
    using block = block_q4_K;
    struct R { uint4 h; uint2 q[4]; };
    __device__ static void load(const block * b, int t, R & r) {
        r.h = *(const uint4 *) b;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.q[j] = ld8(b->qs + 32 * j + 8 * t);
    }
    __device__ static float2 dm(const R & r) { return __half22float2(*(const half2 *) &r.h.x); }
    __device__ static void pair(const R & r, int j, int (&lo)[2], int (&hi)[2], int & scl, int & ml, int & sch,
                                int & mh) {
        const uint2 w = r.q[j];
        lo[0] = (int) (w.x & 0x0F0F0F0Fu);        lo[1] = (int) (w.y & 0x0F0F0F0Fu);
        hi[0] = (int) ((w.x >> 4) & 0x0F0F0F0Fu); hi[1] = (int) ((w.y >> 4) & 0x0F0F0F0Fu);
        scale_min_k4_r(2 * j, r.h, scl, ml);
        scale_min_k4_r(2 * j + 1, r.h, sch, mh);
    }
};

struct dec_q5_K {
    using block = block_q5_K;
    struct R { uint4 h; uint2 qh; uint2 q[4]; };
    __device__ static void load(const block * b, int t, R & r) {
        r.h  = *(const uint4 *) b;
        r.qh = ld8(b->qh + 8 * t);
#pragma unroll
        for (int j = 0; j < 4; ++j) r.q[j] = ld8(b->qs + 32 * j + 8 * t);
    }
    __device__ static float2 dm(const R & r) { return __half22float2(*(const half2 *) &r.h.x); }
    __device__ static void pair(const R & r, int j, int (&lo)[2], int (&hi)[2], int & scl, int & ml, int & sch,
                                int & mh) {
        const uint2 w = r.q[j], h = r.qh;
        lo[0] = (int) ((w.x & 0x0F0F0F0Fu) | (((h.x >> (2 * j)) & 0x01010101u) << 4));
        lo[1] = (int) ((w.y & 0x0F0F0F0Fu) | (((h.y >> (2 * j)) & 0x01010101u) << 4));
        hi[0] = (int) (((w.x >> 4) & 0x0F0F0F0Fu) | (((h.x >> (2 * j + 1)) & 0x01010101u) << 4));
        hi[1] = (int) (((w.y >> 4) & 0x0F0F0F0Fu) | (((h.y >> (2 * j + 1)) & 0x01010101u) << 4));
        scale_min_k4_r(2 * j, r.h, scl, ml);
        scale_min_k4_r(2 * j + 1, r.h, sch, mh);
    }
};

// ggml's op_silu, as built (-use_fast_math)
__device__ __forceinline__ float tc_silu(float x) { return x / (1.0f + expf(-x)); }

// the c-th set bit of m (c-th entry of the owner's (slot, token) list), or -1
__device__ __forceinline__ int nth_bit(unsigned long long m, int c) {
#pragma unroll
    for (int i = 0; i < TC_MAXT; ++i) {
        if (m == 0ull) return -1;
        const int b = __ffsll((long long) m) - 1;
        if (i == c) return b;
        m &= m - 1ull;
    }
    return -1;
}

// x f32 [E, T] (column stride xs floats) -> block_a256 [T][nsb]; grid (nsb, T), 32 threads
__global__ void moe_tc_quant_a256(const float * __restrict__ x, int64_t xs, int nsb, block_a256 * __restrict__ y) {
    const int sb = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    const float * p = x + (int64_t) col * xs + (int64_t) sb * 256 + 8 * lane;
    float v[8], amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = p[k];
        amax = fmaxf(amax, fabsf(v[k]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    int  sum = 0;
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
    if ((lane & 3) == 0) b.s[lane >> 2] = (int16_t) sum;
    if (lane == 0) b.d = d;
}

// grid (ceil(F / 64), TC_NS * T), block 128: warp w of block x computes rows 64x + 16w .. +15 of both gate and up for
// the expert of entry (slot, tok) = blockIdx.y, against every token sharing that expert; exits unless tok owns it.
template <typename D, int NSB>
__global__ void __launch_bounds__(TC_WARPS * 32)
moe_gu_tc_k(const char * __restrict__ up, const char * __restrict__ gate, int64_t row_bytes, int64_t expert_bytes,
            const block_a256 * __restrict__ y, const int32_t * __restrict__ ids, int ids_stride, float * __restrict__ act,
            int nrows, int nsb, int T) {
    using block = typename D::block;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int slot = blockIdx.y % TC_NS, tok = blockIdx.y / TC_NS;
    const int ch   = ids[slot + tok * ids_stride];
    const int n    = TC_NS * T, l1 = lane + 32;
    const int id0  = lane < n ? ids[(lane % TC_NS) + (lane / TC_NS) * ids_stride] : -1;
    const int id1  = l1 < n ? ids[(l1 % TC_NS) + (l1 / TC_NS) * ids_stride] : -1;
    const unsigned long long match = (unsigned long long) __ballot_sync(0xffffffffu, id0 == ch) |
                                     ((unsigned long long) __ballot_sync(0xffffffffu, id1 == ch) << 32);
    if (match & ((1ull << (tok * TC_NS)) - 1ull)) {
        return;   // a lower token owns this expert
    }
    const int row0 = (blockIdx.x * TC_WARPS + warp) * 16;
    if (row0 >= nrows) {
        return;
    }
    // columns: B column g, output columns 2t and 2t+1 (entries of the owner's list; each token appears once)
    const int eg = nth_bit(match, g), e0 = nth_bit(match, 2 * t), e1 = nth_bit(match, 2 * t + 1);
    const bool colg = eg >= 0, c0ok = e0 >= 0, c1ok = e1 >= 0;
    const block_a256 * yg  = y + (int64_t) (colg ? eg / TC_NS : 0) * nsb;
    const block_a256 * yc0 = y + (int64_t) (c0ok ? e0 / TC_NS : 0) * nsb;
    const block_a256 * yc1 = y + (int64_t) (c1ok ? e1 / TC_NS : 0) * nsb;
    const char * wu[2], * wg[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        int r = row0 + g + 8 * h;
        r     = r < nrows ? r : nrows - 1;
        wu[h] = up + (int64_t) ch * expert_bytes + (int64_t) r * row_bytes;
        wg[h] = gate + (int64_t) ch * expert_bytes + (int64_t) r * row_bytes;
    }
    float accu[4] = {0.0f, 0.0f, 0.0f, 0.0f}, accg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int sbs = 0; sbs < nsb; sbs += NSB) {
        typename D::R RU[NSB][2], RG[NSB][2];
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (sbs + u < nsb) {
                    D::load((const block *) wu[h] + sbs + u, t, RU[u][h]);
                    D::load((const block *) wg[h] + sbs + u, t, RG[u][h]);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < NSB; ++u) {
            const int sb = sbs + u;
            if (sb >= nsb) break;
            uint2 bq[8];
            float dx0 = 0.0f, dx1 = 0.0f;
            uint4 s0 = make_uint4(0, 0, 0, 0), s1 = make_uint4(0, 0, 0, 0);
            const block_a256 & bg = yg[sb];
#pragma unroll
            for (int i = 0; i < 8; ++i) bq[i] = colg ? ld8(bg.qs + 32 * i + 8 * t) : make_uint2(0, 0);
            if (c0ok) { dx0 = yc0[sb].d; s0 = *(const uint4 *) yc0[sb].s; }
            if (c1ok) { dx1 = yc1[sb].d; s1 = *(const uint4 *) yc1[sb].s; }
            int aiu[4] = {0, 0, 0, 0}, amu[4] = {0, 0, 0, 0}, aig[4] = {0, 0, 0, 0}, amg[4] = {0, 0, 0, 0};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int bl[2] = { (int) bq[2 * j].x, (int) bq[2 * j].y };
                const int bh[2] = { (int) bq[2 * j + 1].x, (int) bq[2 * j + 1].y };
                const uint32_t w0 = j == 0 ? s0.x : j == 1 ? s0.y : j == 2 ? s0.z : s0.w;
                const uint32_t w1 = j == 0 ? s1.x : j == 1 ? s1.y : j == 2 ? s1.z : s1.w;
                const int S0l = (int) (int16_t) (w0 & 0xFFFF), S0h = (int) (int16_t) (w0 >> 16);
                const int S1l = (int) (int16_t) (w1 & 0xFFFF), S1h = (int) (int16_t) (w1 >> 16);
#pragma unroll
                for (int m = 0; m < 2; ++m) {   // m = 0: up, 1: gate
                    const typename D::R * RR = m == 0 ? RU[u] : RG[u];
                    int lo0[2], hi0[2], lo1[2], hi1[2], scl[2], ml[2], sch[2], mh[2];
                    D::pair(RR[0], j, lo0, hi0, scl[0], ml[0], sch[0], mh[0]);
                    D::pair(RR[1], j, lo1, hi1, scl[1], ml[1], sch[1], mh[1]);
                    const int al[4] = { lo0[0], lo1[0], lo0[1], lo1[1] };
                    const int ah[4] = { hi0[0], hi1[0], hi0[1], hi1[1] };
                    int cl[4] = {0, 0, 0, 0}, chh[4] = {0, 0, 0, 0};
                    mma_s8_16832(cl, al, bl);
                    mma_s8_16832(chh, ah, bh);
                    int * ai = m == 0 ? aiu : aig;
                    int * am = m == 0 ? amu : amg;
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        ai[2 * h + 0] += scl[h] * cl[2 * h + 0] + sch[h] * chh[2 * h + 0];
                        ai[2 * h + 1] += scl[h] * cl[2 * h + 1] + sch[h] * chh[2 * h + 1];
                        am[2 * h + 0] += ml[h] * S0l + mh[h] * S0h;
                        am[2 * h + 1] += ml[h] * S1l + mh[h] * S1h;
                    }
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float2 fu = D::dm(RU[u][h]), fg = D::dm(RG[u][h]);
                accu[2 * h + 0] = col_epilogue(accu[2 * h + 0], dx0, fu, aiu[2 * h + 0], amu[2 * h + 0]);
                accu[2 * h + 1] = col_epilogue(accu[2 * h + 1], dx1, fu, aiu[2 * h + 1], amu[2 * h + 1]);
                accg[2 * h + 0] = col_epilogue(accg[2 * h + 0], dx0, fg, aig[2 * h + 0], amg[2 * h + 0]);
                accg[2 * h + 1] = col_epilogue(accg[2 * h + 1], dx1, fg, aig[2 * h + 1], amg[2 * h + 1]);
            }
        }
    }
    // act[(tok * NS + slot) * F + row] = silu(gate) * up, for each owner column's own (slot, token) entry
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = row0 + g + 8 * h;
        if (r < nrows) {
            if (c0ok) act[(int64_t) e0 * nrows + r] = tc_silu(accg[2 * h + 0]) * accu[2 * h + 0];
            if (c1ok) act[(int64_t) e1 * nrows + r] = tc_silu(accg[2 * h + 1]) * accu[2 * h + 1];
        }
    }
}

}  // namespace

namespace ai::kern {

bool moe_gu_tc_ok(int ggml_type_id, int64_t K, int64_t F, int64_t n_used, int64_t T) {
    const bool type_ok = ggml_type_id == GGML_TYPE_Q4_K || ggml_type_id == GGML_TYPE_Q5_K;
    return type_ok && n_used == TC_NS && T >= 2 && T <= TC_MAXT && K % QK_K == 0 && F % 16 == 0;
}

size_t moe_gu_tc_scratch_bytes(int64_t K, int64_t T) { return (size_t) T * (size_t) (K / QK_K) * sizeof(block_a256); }

bool moe_gu_tc(void * stream, int ggml_type_id, const void * up, const void * gate, int64_t row_bytes,
               int64_t expert_bytes, const float * x, int64_t x_col_stride, void * scratch, const int32_t * ids,
               int64_t ids_stride, float * act, int64_t K, int64_t F, int64_t T) {
    if (!moe_gu_tc_ok(ggml_type_id, K, F, TC_NS, T)) {
        return false;
    }
    const size_t bs = ggml_type_id == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);
    if (row_bytes % 16 != 0 || expert_bytes % 16 != 0 || row_bytes % bs != 0) {
        return false;   // 16-byte block loads
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[moe-tc] LAUNCHED: verify gate/up + SwiGLU on tensor cores (T 2..8, owner dedup), FAST only\n");
    }
    cudaStream_t st  = (cudaStream_t) stream;
    const int    nsb = (int) (K / QK_K);
    block_a256 * ya  = (block_a256 *) scratch;
    moe_tc_quant_a256<<<dim3((unsigned) nsb, (unsigned) T), 32, 0, st>>>(x, x_col_stride, nsb, ya);
    static const int nsb_step = getenv("AI_MOE_TC_NSB") ? atoi(getenv("AI_MOE_TC_NSB")) : 2;
    const dim3 grid((unsigned) ((F + TC_WARPS * 16 - 1) / (TC_WARPS * 16)), (unsigned) (TC_NS * T));
    const dim3 block(TC_WARPS * 32);
    const char * pu = (const char *) up, * pg = (const char *) gate;
    const int ist = (int) ids_stride, nr = (int) F, tt = (int) T;
    if (ggml_type_id == GGML_TYPE_Q4_K) {
        if (nsb_step == 1) moe_gu_tc_k<dec_q4_K, 1><<<grid, block, 0, st>>>(pu, pg, row_bytes, expert_bytes, ya, ids, ist, act, nr, nsb, tt);
        else               moe_gu_tc_k<dec_q4_K, 2><<<grid, block, 0, st>>>(pu, pg, row_bytes, expert_bytes, ya, ids, ist, act, nr, nsb, tt);
    } else {
        if (nsb_step == 1) moe_gu_tc_k<dec_q5_K, 1><<<grid, block, 0, st>>>(pu, pg, row_bytes, expert_bytes, ya, ids, ist, act, nr, nsb, tt);
        else               moe_gu_tc_k<dec_q5_K, 2><<<grid, block, 0, st>>>(pu, pg, row_bytes, expert_bytes, ya, ids, ist, act, nr, nsb, tt);
    }
    return true;
}

}  // namespace ai::kern
