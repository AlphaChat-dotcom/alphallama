// qwen4exp routed experts at verify width, grouped by slot: gate + up + SwiGLU in one launch, then the down
// projection, over the doorbell slot banks.
//
// This replaces moe_experts_raw's ggml chain: per matrix a q8_1 quantise and a mul_mat_vec_q_moe (one warp per
// (slot, token), 2 rows each), a separate swiglu and a cont. In that chain an expert routed to several rows of the
// verify window is re-read and its IQ grid unpack (table lookups + sign expansion) re-run for every row, and a missed
// column (resolve gave it a filler slot) is computed on valid weights and thrown away by the wait kernel.
//
// Here the lowest column routed to a slot owns it: its warps unpack each weight block once and dot it against the
// q8_1 rows of every column that shares the slot. Missed columns are skipped (offload_wait_partial zeroes them). x
// is quantised once for gate and up.
//
// Bit-exact by construction against that chain, per (row, column): lane l covers kbx = l/(qi/vdr) + i*bpi with
// kqs = vdr*(l%(qi/vdr)) and accumulates in kbx order from 0.0f, then warp_reduce_sum (mul_mat_vec_q_moe); each
// unpack/dot pair below is ggml-cuda's vec_dot_*_q8_1 split in two, same integer order and the same float expression;
// the GLU is op_silu(gate) * up; q8_1 copies come from ggml-cuda's quantize_row_q8_1_cuda. Built -use_fast_math as
// ggml-cuda is. AI_Q4X_GROUPED=2 runs both and counts differing values over the computed columns.
#include "kernels/kernels.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kMaxT = 16, kMaxK = 12, kMaxCols = 160;   // k * T <= 160

template <ggml_type type> struct U;

template <> struct U<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K, qi = QI2_XXS, vdr = VDR_IQ2_XXS_Q8_1_MMVQ;
    int   g[8], ls, sub;
    float d;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_xxs * bq2  = (const block_iq2_xxs *) vbq + kbx;
        const int             q2   = get_int_b2(bq2->qs, iqs);
        const uint8_t *       aux8 = (const uint8_t *) &q2;
        const uint32_t        aux32 = get_int_b2(bq2->qs, iqs + 1);
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2    grid_pos = ((const uint2 *) iq2xxs_grid)[aux8[k0 / 2]];
            const uint32_t signs    = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int      signs0   = __vcmpne4(signs & 0x08040201, 0);
            g[k0 + 0]               = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1        = __vcmpne4(signs & 0x80402010, 0);
            g[k0 + 1]               = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        ls  = aux32 >> 27 | 1;
        d   = __half2float(bq2->d);
        sub = iqs / 2;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        int sumi = 0;
#pragma unroll
        for (int k0 = 0; k0 < 8; ++k0) sumi = ggml_cuda_dp4a(g[k0], get_int_b4(bq8_1[sub].qs, k0), sumi);
        sumi          = sumi * ls / 8;
        const float e = d * __low2float(bq8_1[sub].ds);
        return e * sumi;
    }
};

// IQ2_XS / IQ2_S: two half-sums with their own 4-bit scales
struct U2 {
    int   g[8], ls0, ls1, sub;
    float d;
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int l = 0; l < 4; ++l) sumi0 = ggml_cuda_dp4a(g[l], get_int_b4(bq8_1[sub].qs, l), sumi0);
#pragma unroll
        for (int l = 4; l < 8; ++l) sumi1 = ggml_cuda_dp4a(g[l], get_int_b4(bq8_1[sub].qs, l), sumi1);
        const int   sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
        const float e    = d * __low2float(bq8_1[sub].ds);
        return e * sumi;
    }
};

template <> struct U<GGML_TYPE_IQ2_XS> : U2 {
    static constexpr int qk = QK_K, qi = QI2_XS, vdr = VDR_IQ2_XS_Q8_1_MMVQ;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_xs * bq2       = (const block_iq2_xs *) vbq + kbx;
        const int2           q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t *     q2        = (const uint16_t *) &q2_packed;
        ls0                            = bq2->scales[iqs / 2] & 0x0F;
        ls1                            = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2    grid_pos = ((const uint2 *) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
            const uint32_t signs    = unpack_ksigns(q2[l0 / 2] >> 9);
            const int      signs0   = __vcmpne4(signs & 0x08040201, 0);
            g[l0 + 0]               = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1        = __vcmpne4(signs & 0x80402010, 0);
            g[l0 + 1]               = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        d   = __half2float(bq2->d);
        sub = iqs / 2;
    }
};

template <> struct U<GGML_TYPE_IQ2_S> : U2 {
    static constexpr int qk = QK_K, qi = QI2_S, vdr = VDR_IQ2_S_Q8_1_MMVQ;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_s * bq2             = (const block_iq2_s *) vbq + kbx;
        const int           qs_packed       = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t *     qs              = (const uint8_t *) &qs_packed;
        const int           qh              = bq2->qh[iqs / 2];
        const int           signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t *     signs_packed_8  = (const uint8_t *) &signs_packed_32;
        ls0                                 = bq2->scales[iqs / 2] & 0x0F;
        ls1                                 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int * grid_pos = (const int *) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int   signs0   = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int   signs1   = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            g[l0 + 0]            = __vsub4(grid_pos[0] ^ signs0, signs0);
            g[l0 + 1]            = __vsub4(grid_pos[1] ^ signs1, signs1);
        }
        d   = __half2float(bq2->d);
        sub = iqs / 2;
    }
};

// IQ3_XXS / IQ3_S: one sum, a type-specific integer scale step
template <> struct U<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K, qi = QI3_XXS, vdr = VDR_IQ3_XXS_Q8_1_MMVQ;
    int   g[8], ls, sub;
    float d;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_xxs * bq3       = (const block_iq3_xxs *) vbq + kbx;
        const int2            q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t *       q3        = (const uint8_t *) &q3_packed;
        const uint32_t        aux32     = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2     grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs    = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int      signs0   = __vcmpne4(signs & 0x08040201, 0);
            g[l0 + 0]               = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1        = __vcmpne4(signs & 0x80402010, 0);
            g[l0 + 1]               = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        ls  = aux32 >> 28;
        d   = __half2float(bq3->d);
        sub = iqs / 2;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) sumi = ggml_cuda_dp4a(g[l], get_int_b4(bq8_1[sub].qs, l), sumi);
        sumi          = (ls * sumi + sumi / 2) / 2;
        const float e = d * __low2float(bq8_1[sub].ds);
        return e * sumi;
    }
};

template <> struct U<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K, qi = QI3_S, vdr = VDR_IQ3_S_Q8_1_MMVQ;
    int   g[8], sc, sub;
    float d;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s * bq3             = (const block_iq3_s *) vbq + kbx;
        const int2          qs_packed       = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t *     qs              = (const uint8_t *) &qs_packed;
        const int           qh              = bq3->qh[iqs / 2];
        const int           signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t *     signs_packed_8  = (const uint8_t *) &signs_packed_32;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int  signs0   = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int  signs1   = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            g[l0 + 0]           = __vsub4(grid_pos.x ^ signs0, signs0);
            g[l0 + 1]           = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        sc  = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        d   = __half2float(bq3->d);
        sub = iqs / 2;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 8; ++l) sumi = ggml_cuda_dp4a(g[l], get_int_b4(bq8_1[sub].qs, l), sumi);
        sumi *= sc;
        const float e = d * __low2float(bq8_1[sub].ds);
        return e * sumi;
    }
};

template <> struct U<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL, qi = QI4_NL, vdr = VDR_IQ4_NL_Q8_1_MMVQ;
    int   g[4], iq;
    float d;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;
#pragma unroll
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
            const int2 v = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
            g[2 * l + 0] = v.x;
            g[2 * l + 1] = v.y;
        }
        d  = __half2float(bq4->d);
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        const int * q8   = (const int *) bq8_1->qs + iq;
        int         sumi = 0;
#pragma unroll
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
            sumi = ggml_cuda_dp4a(g[2 * l + 0], q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(g[2 * l + 1], q8[l + 4], sumi);
        }
        const float e = d * __low2float(bq8_1->ds);
        return e * sumi;
    }
};

template <> struct U<GGML_TYPE_Q2_0> {
    static constexpr int qk = QK2_0, qi = QI2_0, vdr = VDR_Q2_0_Q8_1_MMVQ;
    int   p[8], iq;
    float d2;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_q2_0 * bq2 = (const block_q2_0 *) vbq + kbx;
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            const int v = get_int_b2(bq2->qs, 2 * iqs + w);
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int x  = (v >> (8 * m)) & 0xFF;
                p[4 * w + m] = (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303;
            }
        }
        d2 = bq2->d;
        iq = iqs;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        const block_q8_1 * q8   = bq8_1 + iq;
        int                sumi = 0, sumq = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int u = get_int_b4(q8->qs, j);
            sumi        = ggml_cuda_dp4a(p[j], u, sumi);
            sumq        = ggml_cuda_dp4a(0x01010101, u, sumq);
        }
        const float d8 = __low2float(q8->ds);
        return d2 * d8 * (float) (sumi - sumq);
    }
};

// ggml-cuda's op_silu, as built (-use_fast_math)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// The window's ownership plan, computed once per call (one block): the lowest column routed to each resident slot
// owns it; plan->own[i] = { slot, ncols, the columns in ascending order } for i < plan->count. Missed columns (slot >=
// filler0) own nothing. The owner order is whatever the atomic gives -- each column's result does not depend on it.
constexpr int kPlanCols = 16;  // columns one owner can serve (= the most window rows sharing a slot; T <= 16)
struct Own { int ch, ne, cs[kPlanCols]; };
struct Plan { int count, pad[3]; Own own[kMaxCols]; };
static_assert(sizeof(Own) == (2 + kPlanCols) * sizeof(int) && offsetof(Plan, own) == 4 * sizeof(int), "TC path reads Plan as records");

// owners are ids in [lo, hi), slot = id - lo (decode: lo = 0, hi = filler0; the GPU miss share: lo = gpu0, the host bank)
__global__ void plan_k(const int32_t * __restrict__ ids, int ids_stride, int k, int n, int lo, int hi, Plan * __restrict__ plan) {
    __shared__ int sid[kMaxCols];
    __shared__ int cnt;
    const int c = threadIdx.x;
    if (c == 0) cnt = 0;
    if (c < n) sid[c] = ids[(c % k) + (c / k) * ids_stride];
    __syncthreads();
    if (c < n) {
        const int ch = sid[c];
        bool owner = ch >= lo && ch < hi;
        for (int d = 0; d < c && owner; ++d) owner = sid[d] != ch;
        if (owner) {
            const int i = atomicAdd(&cnt, 1);
            Own &     o = plan->own[i];
            int       ne = 0;
            for (int d = c; d < n && ne < kPlanCols; ++d) if (sid[d] == ch) o.cs[ne++] = d;
            o.ch = ch - lo;
            o.ne = ne;
        }
    }
    __syncthreads();
    if (c == 0) plan->count = cnt;
}

// grid (ceil(F / (W*R)), k * T), block (32, W): block y = owner y of the plan (exits past plan->count); warp w does rows
// rb + r*W + w (r < R) of the owner's slot, each exactly as one warp of mul_mat_vec_q_moe would (lane decomposition and
// accumulation order unchanged); act[c * F + row] = silu(gate) * up for every column c the owner serves.
template <ggml_type type, int W, int R, int MT>
__global__ void __launch_bounds__(32 * W) gu_k(const void * __restrict__ up, const void * __restrict__ gate,
                                               const block_q8_1 * __restrict__ y, const Plan * __restrict__ plan,
                                               float * __restrict__ act, int ncols, int nrows, int stride_row,
                                               int stride_ch, int k) {
    using X = U<type>;
    constexpr int bpi = X::vdr * 32 / X::qi;
    if ((int) blockIdx.y >= plan->count) return;
    const Own & o  = plan->own[blockIdx.y];
    const int   ch = o.ch, ne = o.ne;
    int         cs[MT];
#pragma unroll
    for (int e = 0; e < MT; ++e) cs[e] = e < ne ? o.cs[e] : 0;
    const int lane = threadIdx.x, bpr = ncols / X::qk, ystr = ncols / QK8_1;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        const int row = blockIdx.x * (W * R) + r * W + threadIdx.y;
        if (row >= nrows) break;
        float t[MT], tg[MT];
#pragma unroll
        for (int e = 0; e < MT; ++e) t[e] = tg[e] = 0.0f;
        const int off0 = ch * stride_ch + row * stride_row;
        for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
            const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
            X         xu, xg;
            xu.load(up, off0 + kbx, kqs);
            xg.load(gate, off0 + kbx, kqs);
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < ne) {
                    const block_q8_1 * yy = y + (cs[e] / k) * ystr + kby;
                    t[e] += xu.dot(yy);
                    tg[e] += xg.dot(yy);
                }
            }
        }
#pragma unroll
        for (int e = 0; e < MT; ++e) {
            if (e < ne) {
                const float a = warp_reduce_sum<32>(t[e]);
                const float b = warp_reduce_sum<32>(tg[e]);
                if (lane == 0) act[(long long) cs[e] * nrows + row] = silu(b) * a;
            }
        }
    }
}

// grid (ceil(N / (W*R)), k * T), block (32, W): out[c * N + row] = down[ch] row . act[c] (each column has its own input row)
template <ggml_type type, int W, int R, int MT>
__global__ void __launch_bounds__(32 * W) dn_k(const void * __restrict__ down, const block_q8_1 * __restrict__ y,
                                               const Plan * __restrict__ plan, float * __restrict__ out, int ncols,
                                               int nrows, int stride_row, int stride_ch) {
    using X = U<type>;
    constexpr int bpi = X::vdr * 32 / X::qi;
    if ((int) blockIdx.y >= plan->count) return;
    const Own & o  = plan->own[blockIdx.y];
    const int   ch = o.ch, ne = o.ne;
    int         cs[MT];
#pragma unroll
    for (int e = 0; e < MT; ++e) cs[e] = e < ne ? o.cs[e] : 0;
    const int lane = threadIdx.x, bpr = ncols / X::qk, ystr = ncols / QK8_1;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        const int row = blockIdx.x * (W * R) + r * W + threadIdx.y;
        if (row >= nrows) break;
        float t[MT];
#pragma unroll
        for (int e = 0; e < MT; ++e) t[e] = 0.0f;
        const int off0 = ch * stride_ch + row * stride_row;
        for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
            const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
            X         xd;
            xd.load(down, off0 + kbx, kqs);
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < ne) t[e] += xd.dot(y + (long long) cs[e] * ystr + kby);
            }
        }
#pragma unroll
        for (int e = 0; e < MT; ++e) {
            if (e < ne) {
                const float a = warp_reduce_sum<32>(t[e]);
                if (lane == 0) out[(long long) cs[e] * nrows + row] = a;
            }
        }
    }
}


// Shared-memory-staged variants (AI_Q4X_EXP_SMEM, default 1): the block first copies its W*R consecutive rows of the
// owner's slot (gate and up, or down: one contiguous 16-byte-aligned chunk per matrix) into shared memory with
// cp.async, then every warp runs exactly gu_k's / dn_k's per-lane arithmetic on the staged rows. In the global-memory
// kernels each lane's dependent 2-byte loads keep too few bytes in flight to approach DRAM bandwidth. Columns are
// processed in chunks of 4 (instantiating for 8 columns at once costs too many registers); a column's accumulation
// does not depend on its chunk, so the results are bit-identical to gu_k / dn_k.
__device__ __forceinline__ void gs_cp16(void * s, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}

template <ggml_type type, int W, int R>
__global__ void __launch_bounds__(32 * W) gus_k(const void * __restrict__ up, const void * __restrict__ gate,
                                                const block_q8_1 * __restrict__ y, const Plan * __restrict__ plan,
                                                float * __restrict__ act, int ncols, int nrows, int stride_row,
                                                int stride_ch, int k, int bsz) {
    using X = U<type>;
    constexpr int bpi = X::vdr * 32 / X::qi, MT = 4;
    extern __shared__ __align__(16) uint8_t gs_sm[];
    if ((int) blockIdx.y >= plan->count) return;
    const Own & o  = plan->own[blockIdx.y];
    const int   ch = o.ch, ne = o.ne;
    const int   rb = blockIdx.x * (W * R), tid = threadIdx.y * 32 + threadIdx.x;
    const size_t rbytes = (size_t) stride_row * bsz, chunk = (size_t) (W * R) * rbytes;
    const uint8_t * su = (const uint8_t *) up + ((size_t) ch * stride_ch + (size_t) rb * stride_row) * bsz;
    const uint8_t * sg = (const uint8_t *) gate + ((size_t) ch * stride_ch + (size_t) rb * stride_row) * bsz;
    for (size_t i = tid; i < chunk / 16; i += 32 * W) {
        gs_cp16(gs_sm + 16 * i, su + 16 * i);
        gs_cp16(gs_sm + chunk + 16 * i, sg + 16 * i);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const void * upb = gs_sm, * gab = gs_sm + chunk;
    const int lane = threadIdx.x, bpr = ncols / X::qk, ystr = ncols / QK8_1;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        const int lr = r * W + threadIdx.y, row = rb + lr;
        if (row >= nrows) break;
        const int off0 = lr * stride_row;
#pragma unroll 1
        for (int e0 = 0; e0 < ne; e0 += MT) {
            const int nc = min(MT, ne - e0);
            int       cs[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) cs[e] = e < nc ? o.cs[e0 + e] : 0;
            float t[MT], tg[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) t[e] = tg[e] = 0.0f;
            for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
                const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
                X         xu, xg;
                xu.load(upb, off0 + kbx, kqs);
                xg.load(gab, off0 + kbx, kqs);
#pragma unroll
                for (int e = 0; e < MT; ++e) {
                    if (e < nc) {
                        const block_q8_1 * yy = y + (cs[e] / k) * ystr + kby;
                        t[e] += xu.dot(yy);
                        tg[e] += xg.dot(yy);
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < nc) {
                    const float a = warp_reduce_sum<32>(t[e]);
                    const float b = warp_reduce_sum<32>(tg[e]);
                    if (lane == 0) act[(long long) cs[e] * nrows + row] = silu(b) * a;
                }
            }
        }
    }
}

template <ggml_type type, int W, int R>
__global__ void __launch_bounds__(32 * W) dns_k(const void * __restrict__ down, const block_q8_1 * __restrict__ y,
                                                const Plan * __restrict__ plan, float * __restrict__ out, int ncols,
                                                int nrows, int stride_row, int stride_ch, int bsz) {
    using X = U<type>;
    constexpr int bpi = X::vdr * 32 / X::qi, MT = 4;
    extern __shared__ __align__(16) uint8_t gs_sm[];
    if ((int) blockIdx.y >= plan->count) return;
    const Own & o  = plan->own[blockIdx.y];
    const int   ch = o.ch, ne = o.ne;
    const int   rb = blockIdx.x * (W * R), tid = threadIdx.y * 32 + threadIdx.x;
    const size_t chunk = (size_t) (W * R) * stride_row * bsz;
    const uint8_t * sd = (const uint8_t *) down + ((size_t) ch * stride_ch + (size_t) rb * stride_row) * bsz;
    for (size_t i = tid; i < chunk / 16; i += 32 * W) gs_cp16(gs_sm + 16 * i, sd + 16 * i);
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const void * db = gs_sm;
    const int lane = threadIdx.x, bpr = ncols / X::qk, ystr = ncols / QK8_1;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        const int lr = r * W + threadIdx.y, row = rb + lr;
        if (row >= nrows) break;
        const int off0 = lr * stride_row;
#pragma unroll 1
        for (int e0 = 0; e0 < ne; e0 += MT) {
            const int nc = min(MT, ne - e0);
            int       cs[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) cs[e] = e < nc ? o.cs[e0 + e] : 0;
            float t[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) t[e] = 0.0f;
            for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
                const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
                X         xd;
                xd.load(db, off0 + kbx, kqs);
#pragma unroll
                for (int e = 0; e < MT; ++e) {
                    if (e < nc) t[e] += xd.dot(y + (long long) cs[e] * ystr + kby);
                }
            }
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < nc) {
                    const float a = warp_reduce_sum<32>(t[e]);
                    if (lane == 0) out[(long long) cs[e] * nrows + row] = a;
                }
            }
        }
    }
}

__device__ unsigned long long g_cmp[4];   // calls, values compared, values differing, values off by > 1e-3 relative
__device__ unsigned int       g_cmp_maxabs;   // float bits of the largest |a - b| (non-negative floats order as uints)
__device__ double             g_cmp_l2[2];    // sum (a - b)^2, sum b^2

__global__ void compare_k(const float * a, const float * b, const int32_t * ids, int ids_stride, int k, int filler0,
                          long long N) {
    const int c  = blockIdx.x;
    const int ch = ids[(c % k) + (c / k) * ids_stride];
    if (ch >= filler0) return;
    unsigned long long d = 0, big = 0;
    float              mx = 0.0f;
    double             e2 = 0.0, r2 = 0.0;
    for (long long i = threadIdx.x; i < N; i += blockDim.x) {
        const float x = a[c * N + i], y = b[c * N + i], df = fabsf(x - y);
        d += __float_as_uint(x) != __float_as_uint(y);
        big += df > 1e-3f * fmaxf(fabsf(y), 1e-6f);
        mx = fmaxf(mx, df);
        e2 += (double) df * df;
        r2 += (double) y * y;
    }
    atomicAdd(&g_cmp[2], d);
    atomicAdd(&g_cmp[3], big);
    atomicMax(&g_cmp_maxabs, __float_as_uint(mx));
    atomicAdd(&g_cmp_l2[0], e2);
    atomicAdd(&g_cmp_l2[1], r2);
    if (threadIdx.x == 0) {
        atomicAdd(&g_cmp[1], (unsigned long long) N);
        if (c == 0) atomicAdd(&g_cmp[0], 1ull);
    }
}

size_t bsize(int ty) {
    switch (ty) {
        case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS:  return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
        case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ3_S:   return sizeof(block_iq3_s);
        case GGML_TYPE_IQ4_NL:  return sizeof(block_iq4_nl);
        case GGML_TYPE_Q2_0:    return sizeof(block_q2_0);
        default:                return 0;
    }
}

bool gu_type(int ty) {
    return ty == GGML_TYPE_IQ2_XXS || ty == GGML_TYPE_IQ2_XS || ty == GGML_TYPE_IQ2_S || ty == GGML_TYPE_IQ3_XXS ||
           ty == GGML_TYPE_IQ3_S;
}

// Block layout: warps x rows per warp. AI_Q4X_EXP_LAYOUT 0 = 4x1, 1 = 8x2 (default: 16 rows per
// block, 4x fewer blocks), 2 = 8x1, 3 = 4x2 -- a runtime switch so the bitwise compare can tell which layouts keep the
// reference's results (the per-row arithmetic is identical; only codegen differs).
static int exp_layout() {
    static const int v = getenv("AI_Q4X_EXP_LAYOUT") ? atoi(getenv("AI_Q4X_EXP_LAYOUT")) : 1;
    return v;
}

template <ggml_type type, int MT, int W, int R>
void launch_gu_l(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const Plan * plan,
                 float * act, int K, int F, int sr, int sc, int k, int n) {
    const dim3 grid((unsigned) ((F + W * R - 1) / (W * R)), (unsigned) n), block(32, W);
    gu_k<type, W, R, MT><<<grid, block, 0, st>>>(up, gate, y, plan, act, K, F, sr, sc, k);
}

template <ggml_type type, int MT>
void launch_gu(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const Plan * plan,
               float * act, int K, int F, int sr, int sc, int k, int n) {
    switch (exp_layout()) {
        case 0:  launch_gu_l<type, MT, 4, 1>(st, up, gate, y, plan, act, K, F, sr, sc, k, n); break;
        case 2:  launch_gu_l<type, MT, 8, 1>(st, up, gate, y, plan, act, K, F, sr, sc, k, n); break;
        case 3:  launch_gu_l<type, MT, 4, 2>(st, up, gate, y, plan, act, K, F, sr, sc, k, n); break;
        default: launch_gu_l<type, MT, 8, 2>(st, up, gate, y, plan, act, K, F, sr, sc, k, n); break;
    }
}

template <ggml_type type, int MT, int W, int R>
void launch_dn_l(cudaStream_t st, const void * down, const block_q8_1 * y, const Plan * plan, float * out, int F,
                 int N, int sr, int sc, int n) {
    const dim3 grid((unsigned) ((N + W * R - 1) / (W * R)), (unsigned) n), block(32, W);
    dn_k<type, W, R, MT><<<grid, block, 0, st>>>(down, y, plan, out, F, N, sr, sc);
}

template <ggml_type type, int MT>
void launch_dn(cudaStream_t st, const void * down, const block_q8_1 * y, const Plan * plan, float * out, int F,
               int N, int sr, int sc, int n) {
    switch (exp_layout()) {
        case 0:  launch_dn_l<type, MT, 4, 1>(st, down, y, plan, out, F, N, sr, sc, n); break;
        case 2:  launch_dn_l<type, MT, 8, 1>(st, down, y, plan, out, F, N, sr, sc, n); break;
        case 3:  launch_dn_l<type, MT, 4, 2>(st, down, y, plan, out, F, N, sr, sc, n); break;
        default: launch_dn_l<type, MT, 8, 2>(st, down, y, plan, out, F, N, sr, sc, n); break;
    }
}


// Smem-staged dispatch: 8 warps x 2 rows = 16 rows per block
constexpr int kSmW = 8, kSmR = 2;
template <ggml_type type>
void launch_gus(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const Plan * plan,
                float * act, int K, int F, int sr, int sc, int k, int n) {
    const int    bsz = (int) bsize(type);
    const size_t sm  = 2 * (size_t) (kSmW * kSmR) * sr * bsz;
    const dim3   grid((unsigned) (F / (kSmW * kSmR)), (unsigned) n), block(32, kSmW);
    gus_k<type, kSmW, kSmR><<<grid, block, sm, st>>>(up, gate, y, plan, act, K, F, sr, sc, k, bsz);
}
template <ggml_type type>
void launch_dns(cudaStream_t st, const void * down, const block_q8_1 * y, const Plan * plan, float * out, int F, int N,
                int sr, int sc, int n) {
    const int    bsz = (int) bsize(type);
    const size_t sm  = (size_t) (kSmW * kSmR) * sr * bsz;
    const dim3   grid((unsigned) (N / (kSmW * kSmR)), (unsigned) n), block(32, kSmW);
    dns_k<type, kSmW, kSmR><<<grid, block, sm, st>>>(down, y, plan, out, F, N, sr, sc, bsz);
}
void dispatch_smem(cudaStream_t st, int tyg, const void * up, const void * gate, int srg, int scg, int tyd,
                   const void * down, int srd, int scd, const block_q8_1 * xq, const Plan * plan, float * act,
                   const block_q8_1 * aq, float * out, int K, int F, int N, int k, int n, int phase) {
    if (phase == 0) {
        switch (tyg) {
            case GGML_TYPE_IQ2_XXS: launch_gus<GGML_TYPE_IQ2_XXS>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            case GGML_TYPE_IQ2_XS:  launch_gus<GGML_TYPE_IQ2_XS>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            case GGML_TYPE_IQ2_S:   launch_gus<GGML_TYPE_IQ2_S>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            case GGML_TYPE_IQ3_XXS: launch_gus<GGML_TYPE_IQ3_XXS>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            default:                launch_gus<GGML_TYPE_IQ3_S>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
        }
    } else if (tyd == GGML_TYPE_IQ4_NL) {
        launch_dns<GGML_TYPE_IQ4_NL>(st, down, aq, plan, out, F, N, srd, scd, n);
    } else {
        launch_dns<GGML_TYPE_Q2_0>(st, down, aq, plan, out, F, N, srd, scd, n);
    }
}
// staged rows must be whole 16-byte chunks and fit the default 48 KB dynamic shared memory
bool smem_ok(int tyg, int tyd, const void * up, const void * gate, int64_t rbg, int64_t ebg, const void * down,
             int64_t rbd, int64_t ebd, int64_t F, int64_t N) {
    const int64_t rows = kSmW * kSmR;
    return F % rows == 0 && N % rows == 0 && (rows * rbg) % 16 == 0 && ebg % 16 == 0 && (rows * rbd) % 16 == 0 &&
           ebd % 16 == 0 && (uintptr_t) up % 16 == 0 && (uintptr_t) gate % 16 == 0 && (uintptr_t) down % 16 == 0 &&
           2 * rows * rbg <= 48 * 1024 && rows * rbd <= 48 * 1024 && bsize(tyg) > 0 && bsize(tyd) > 0;
}

template <int MT>
void dispatch(cudaStream_t st, int tyg, const void * up, const void * gate, int srg, int scg, int tyd,
              const void * down, int srd, int scd, const block_q8_1 * xq, const Plan * plan, float * act,
              const block_q8_1 * aq, float * out, int K, int F, int N, int k, int n, int phase) {
    if (phase == 0) {
        switch (tyg) {
            case GGML_TYPE_IQ2_XXS: launch_gu<GGML_TYPE_IQ2_XXS, MT>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            case GGML_TYPE_IQ2_XS:  launch_gu<GGML_TYPE_IQ2_XS, MT>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            case GGML_TYPE_IQ2_S:   launch_gu<GGML_TYPE_IQ2_S, MT>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            case GGML_TYPE_IQ3_XXS: launch_gu<GGML_TYPE_IQ3_XXS, MT>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
            default:                launch_gu<GGML_TYPE_IQ3_S, MT>(st, up, gate, xq, plan, act, K, F, srg, scg, k, n); break;
        }
    } else if (tyd == GGML_TYPE_IQ4_NL) {
        launch_dn<GGML_TYPE_IQ4_NL, MT>(st, down, aq, plan, out, F, N, srd, scd, n);
    } else {
        launch_dn<GGML_TYPE_Q2_0, MT>(st, down, aq, plan, out, F, N, srd, scd, n);
    }
}


// Wide (prefill) windows: the owner plan comes from the host (prefill has the routing there), one owner per slot with
// all its columns (up to kWideCols); each block loops over them in chunks of 8, so a tile of weight rows is read from
// DRAM once per window and re-decoded from L1/L2 per chunk. Per (row, column) the arithmetic is gu_k's / dn_k's.
// Plan record: [slot, ncols, col_0 .. col_{ncols-1}], stride kWideRec ints.
constexpr int kWideCols = 512, kWideRec = 2 + kWideCols;

template <ggml_type type, int W, int R>
__global__ void __launch_bounds__(32 * W) gu_w_k(const void * __restrict__ up, const void * __restrict__ gate,
                                                 const block_q8_1 * __restrict__ y, const int32_t * __restrict__ plan,
                                                 float * __restrict__ act, int ncols, int nrows, int stride_row,
                                                 int stride_ch, int k) {
    using X = U<type>;
    constexpr int bpi = X::vdr * 32 / X::qi, MT = 8;
    const int32_t * o  = plan + (size_t) blockIdx.y * kWideRec;
    const int       ch = o[0], ne_all = o[1];
    const int lane = threadIdx.x, bpr = ncols / X::qk, ystr = ncols / QK8_1;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        const int row = blockIdx.x * (W * R) + r * W + threadIdx.y;
        if (row >= nrows) break;
        const int off0 = ch * stride_ch + row * stride_row;
#pragma unroll 1
        for (int c0 = 0; c0 < ne_all; c0 += MT) {
            const int ne = min(MT, ne_all - c0);
            int       cs[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) cs[e] = e < ne ? o[2 + c0 + e] : 0;
            float t[MT], tg[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) t[e] = tg[e] = 0.0f;
            for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
                const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
                X         xu, xg;
                xu.load(up, off0 + kbx, kqs);
                xg.load(gate, off0 + kbx, kqs);
#pragma unroll
                for (int e = 0; e < MT; ++e) {
                    if (e < ne) {
                        const block_q8_1 * yy = y + (cs[e] / k) * ystr + kby;
                        t[e] += xu.dot(yy);
                        tg[e] += xg.dot(yy);
                    }
                }
            }
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < ne) {
                    const float a = warp_reduce_sum<32>(t[e]);
                    const float b = warp_reduce_sum<32>(tg[e]);
                    if (lane == 0) act[(long long) cs[e] * nrows + row] = silu(b) * a;
                }
            }
        }
    }
}

template <ggml_type type, int W, int R>
__global__ void __launch_bounds__(32 * W) dn_w_k(const void * __restrict__ down, const block_q8_1 * __restrict__ y,
                                                 const int32_t * __restrict__ plan, float * __restrict__ out, int ncols,
                                                 int nrows, int stride_row, int stride_ch) {
    using X = U<type>;
    constexpr int bpi = X::vdr * 32 / X::qi, MT = 8;
    const int32_t * o  = plan + (size_t) blockIdx.y * kWideRec;
    const int       ch = o[0], ne_all = o[1];
    const int lane = threadIdx.x, bpr = ncols / X::qk, ystr = ncols / QK8_1;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        const int row = blockIdx.x * (W * R) + r * W + threadIdx.y;
        if (row >= nrows) break;
        const int off0 = ch * stride_ch + row * stride_row;
#pragma unroll 1
        for (int c0 = 0; c0 < ne_all; c0 += MT) {
            const int ne = min(MT, ne_all - c0);
            int       cs[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) cs[e] = e < ne ? o[2 + c0 + e] : 0;
            float t[MT];
#pragma unroll
            for (int e = 0; e < MT; ++e) t[e] = 0.0f;
            for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
                const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
                X         xd;
                xd.load(down, off0 + kbx, kqs);
#pragma unroll
                for (int e = 0; e < MT; ++e) {
                    if (e < ne) t[e] += xd.dot(y + (long long) cs[e] * ystr + kby);
                }
            }
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < ne) {
                    const float a = warp_reduce_sum<32>(t[e]);
                    if (lane == 0) out[(long long) cs[e] * nrows + row] = a;
                }
            }
        }
    }
}

template <ggml_type type>
void launch_gu_w(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const int32_t * plan,
                 int n_own, float * act, int K, int F, int sr, int sc, int k) {
    const dim3 grid((unsigned) ((F + 15) / 16), (unsigned) n_own), block(32, 8);
    gu_w_k<type, 8, 2><<<grid, block, 0, st>>>(up, gate, y, plan, act, K, F, sr, sc, k);
}

template <ggml_type type>
void launch_dn_w(cudaStream_t st, const void * down, const block_q8_1 * y, const int32_t * plan, int n_own, float * out,
                 int F, int N, int sr, int sc) {
    const dim3 grid((unsigned) ((N + 15) / 16), (unsigned) n_own), block(32, 8);
    dn_w_k<type, 8, 2><<<grid, block, 0, st>>>(down, y, plan, out, F, N, sr, sc);
}


// Tensor-core grouped experts (AI_Q4X_TC=1). The dp4a kernels above use only a fraction of DRAM bandwidth, worst at
// prefill widths: a warp owns 2 rows of a 2560-wide row and issues dependent byte loads, so the bytes in flight are
// too few. Here each warp stages a weight tile into shared memory with cp.async (gate/up: 8 gate + 8 up rows, two
// superblocks per stage, three stages; down: the whole 16-row tile), unpacks every (row, 32-block) once into int8 A
// fragments and runs mma.m16n8k16 s8 -- one k16 half of a 32-block per mma, so the integer half-sums are exact (they
// are the same integers dp4a forms). Up to 8 columns of an owner ride one mma; wide owners loop over column groups.
//
// Numerics: per (row, column, 32-block), term = (sw * d8) * (float) (s0 * m0 + s1 * m1), accumulated in K order
// from 0 with fmaf -- where s0/s1 are the k16 half-sums, m0/m1 the block's integer scale multipliers and sw its
// float scale. That is the exact dequantised dot of the block: ggml-cuda's MMVQ forms ((ls * s + s / 2) / 2 etc.)
// truncate in integers, this does not. Every column runs the same instruction sequence whatever its position in the
// mma, and the same kernel serves T = 1, verify and prefill, so MTP stays bit-identical to plain within one build.
// It is not bit-identical to the dp4a kernels: quality is gated by paired NLL.
__device__ __forceinline__ void tc_mma(int (&c)[4], int a0, int a1, int b0) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    NO_DEVICE_CODE;
#endif
}
__device__ __forceinline__ void tc_cp4(void * s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void tc_cp16(void * s, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void tc_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void tc_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ int tc_sign(uint32_t signs, int odd, int grid) {   // ggml-cuda's sign expansion
    const int sg = __vcmpne4(signs & (odd ? 0x80402010u : 0x08040201u), 0);
    return __vsub4(grid ^ sg, sg);
}
__device__ __forceinline__ int tc_sign2(int s8, int odd, int grid) {          // IQ2_S / IQ3_S sign bytes
    const int sg = odd ? __vcmpne4(((s8 & 0x30) << 3) | ((s8 & 0xC0) << 17), 0)
                       : __vcmpne4(((s8 & 0x03) << 7) | ((s8 & 0x0C) << 21), 0);
    return __vsub4(grid ^ sg, sg);
}

// TQ<type>::unit(block, sub, tg, ...): ints tg and 4 + tg (elements 4 tg.., 16 + 4 tg..) of 32-block `sub` of the
// block, its integer multipliers for the two k16 halves and its float scale. SB = bytes per block, UPB = 32-blocks
// per block.
template <ggml_type type> struct TQ;
template <> struct TQ<GGML_TYPE_IQ2_XXS> {
    static constexpr int SB = sizeof(block_iq2_xxs), UPB = 8;
    static __device__ __forceinline__ void unit(const uint8_t * p, int sub, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_iq2_xxs * b = (const block_iq2_xxs *) p;
        const int      q2    = get_int_b2(b->qs, 2 * sub);
        const uint32_t aux32 = get_int_b2(b->qs, 2 * sub + 1);
        const uint8_t * a8   = (const uint8_t *) &q2;
        const uint2 g0 = ((const uint2 *) iq2xxs_grid)[a8[tg >> 1]], g1 = ((const uint2 *) iq2xxs_grid)[a8[2 + (tg >> 1)]];
        v0 = tc_sign(unpack_ksigns(aux32 >> (7 * (tg >> 1))), tg & 1, (tg & 1) ? g0.y : g0.x);
        v1 = tc_sign(unpack_ksigns(aux32 >> (7 * (2 + (tg >> 1)))), tg & 1, (tg & 1) ? g1.y : g1.x);
        m0 = m1 = (int) (aux32 >> 27) | 1;
        sw = __half2float(b->d) * 0.125f;
    }
};
template <> struct TQ<GGML_TYPE_IQ2_XS> {
    static constexpr int SB = sizeof(block_iq2_xs), UPB = 8;
    static __device__ __forceinline__ void unit(const uint8_t * p, int sub, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_iq2_xs * b = (const block_iq2_xs *) p;
        const int2 qp = make_int2(get_int_b2(b->qs, 2 * sub), get_int_b2(b->qs, 2 * sub + 1));
        const uint16_t * q2 = (const uint16_t *) &qp;
        const int e0 = q2[tg >> 1], e1 = q2[2 + (tg >> 1)];
        const uint2 g0 = ((const uint2 *) iq2xs_grid)[e0 & 0x1FF], g1 = ((const uint2 *) iq2xs_grid)[e1 & 0x1FF];
        v0 = tc_sign(unpack_ksigns(e0 >> 9), tg & 1, (tg & 1) ? g0.y : g0.x);
        v1 = tc_sign(unpack_ksigns(e1 >> 9), tg & 1, (tg & 1) ? g1.y : g1.x);
        m0 = 2 * (b->scales[sub] & 0x0F) + 1;
        m1 = 2 * (b->scales[sub] >> 4) + 1;
        sw = __half2float(b->d) * 0.125f;
    }
};
template <> struct TQ<GGML_TYPE_IQ2_S> {
    static constexpr int SB = sizeof(block_iq2_s), UPB = 8;
    static __device__ __forceinline__ void unit(const uint8_t * p, int sub, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_iq2_s * b = (const block_iq2_s *) p;
        const int qsp = get_int_b2(b->qs, sub), sp = get_int_b2(b->qs, QK_K / 32 + sub);
        const uint8_t * qs = (const uint8_t *) &qsp, * s8 = (const uint8_t *) &sp;
        const int qh = b->qh[sub], i0 = tg >> 1, i1 = 2 + (tg >> 1);
        const int * g0 = (const int *) (iq2s_grid + (qs[i0] | ((qh << (8 - 2 * i0)) & 0x300)));
        const int * g1 = (const int *) (iq2s_grid + (qs[i1] | ((qh << (8 - 2 * i1)) & 0x300)));
        v0 = tc_sign2(s8[i0], tg & 1, g0[tg & 1]);
        v1 = tc_sign2(s8[i1], tg & 1, g1[tg & 1]);
        m0 = 2 * (b->scales[sub] & 0x0F) + 1;
        m1 = 2 * (b->scales[sub] >> 4) + 1;
        sw = __half2float(b->d) * 0.125f;
    }
};
template <> struct TQ<GGML_TYPE_IQ3_XXS> {
    static constexpr int SB = sizeof(block_iq3_xxs), UPB = 8;
    static __device__ __forceinline__ void unit(const uint8_t * p, int sub, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_iq3_xxs * b = (const block_iq3_xxs *) p;
        const int2 qp = make_int2(get_int_b2(b->qs, 2 * sub), get_int_b2(b->qs, 2 * sub + 1));
        const uint8_t * q3 = (const uint8_t *) &qp;
        const uint32_t aux32 = get_int_b2(b->qs, QK_K / 16 + sub);
        v0 = tc_sign(unpack_ksigns(aux32 >> (7 * (tg >> 1))), tg & 1, (int) iq3xxs_grid[q3[tg]]);
        v1 = tc_sign(unpack_ksigns(aux32 >> (7 * (2 + (tg >> 1)))), tg & 1, (int) iq3xxs_grid[q3[4 + tg]]);
        m0 = m1 = 2 * (int) (aux32 >> 28) + 1;
        sw = __half2float(b->d) * 0.25f;
    }
};
template <> struct TQ<GGML_TYPE_IQ3_S> {
    static constexpr int SB = sizeof(block_iq3_s), UPB = 8;
    static __device__ __forceinline__ void unit(const uint8_t * p, int sub, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_iq3_s * b = (const block_iq3_s *) p;
        const int2 qp = make_int2(get_int_b2(b->qs, 2 * sub), get_int_b2(b->qs, 2 * sub + 1));
        const uint8_t * q = (const uint8_t *) &qp;
        const int sp = get_int_b2(b->signs, sub);
        const uint8_t * s8 = (const uint8_t *) &sp;
        const int qh = b->qh[sub], j0 = tg, j1 = 4 + tg;
        v0 = tc_sign2(s8[j0 >> 1], j0 & 1, (int) iq3s_grid[q[j0] | ((qh << (8 - j0)) & 0x100)]);
        v1 = tc_sign2(s8[j1 >> 1], j1 & 1, (int) iq3s_grid[q[j1] | ((qh << (8 - j1)) & 0x100)]);
        m0 = m1 = 1 + 2 * ((b->scales[sub >> 1] >> (4 * (sub & 1))) & 0x0F);
        sw = __half2float(b->d);
    }
};
template <> struct TQ<GGML_TYPE_IQ4_NL> {
    static constexpr int SB = sizeof(block_iq4_nl), UPB = 1;
    static __device__ __forceinline__ void unit(const uint8_t * p, int, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_iq4_nl * b = (const block_iq4_nl *) p;
        const int2 v = get_int_from_table_16(get_int_b2(b->qs, tg), kvalues_iq4nl);
        v0 = v.x;   // elements 4 tg .. 4 tg + 3 (low nibbles)
        v1 = v.y;   // elements 16 + 4 tg .. (high nibbles)
        m0 = m1 = 1;
        sw = __half2float(b->d);
    }
};
template <> struct TQ<GGML_TYPE_Q2_0> {
    static constexpr int SB = sizeof(block_q2_0), UPB = 2;
    static __device__ __forceinline__ int ex(int x) { return __vsub4((x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303, 0x01010101); }
    static __device__ __forceinline__ void unit(const uint8_t * p, int sub, int tg, int & v0, int & v1, int & m0, int & m1, float & sw) {
        const block_q2_0 * b = (const block_q2_0 *) p;
        v0 = ex(b->qs[8 * sub + tg]);       // codes {0..3} -> {-1, 0, +1, +2}: sum (c - 1) u = ggml-cuda's sumi - sumq
        v1 = ex(b->qs[8 * sub + 4 + tg]);
        m0 = m1 = 1;
        sw = __half2float(b->d);
    }
};

constexpr int kTcKC = 2, kTcStages = 3, kTcWarps = 4;   // gate/up: superblocks per stage, stages per warp
constexpr int kTcGuRows = 8, kTcDnRows = 16;          // output rows per warp tile

// Owner records: [slot, ncols, col_0 ..] at rec_stride ints (the decode Plan's Own and the wide plan both fit);
// owner count from count_dev when non-null (decode: plan_k), else count. grid (ceil(F / (8 * kTcWarps)), owners).
template <ggml_type type, int NG>
__global__ void __launch_bounds__(32 * kTcWarps) gu_tc_k(const uint8_t * __restrict__ up, const uint8_t * __restrict__ gate,
                                                         const block_q8_1 * __restrict__ y, const int32_t * __restrict__ recs,
                                                         int rec_stride, const int * __restrict__ count_dev, int count,
                                                         float * __restrict__ act, int K, int F, long long row_bytes,
                                                         long long exp_bytes, int k) {
    using Q = TQ<type>;
    constexpr int CH = kTcKC * Q::SB, W4 = CH / 4, STB = 16 * CH;
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    if ((int) blockIdx.y >= (count_dev ? *count_dev : count)) return;
    const int r0 = (blockIdx.x * kTcWarps + warp) * kTcGuRows;
    if (r0 >= F) return;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], ystr = K / QK8_1, nst = K / (QK_K * kTcKC);
    uint8_t *       sm = tc_sm + warp * (kTcStages * STB);
    const uint8_t * gb = gate + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    const uint8_t * ub = up + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    auto load = [&](int st, int buf) {
        uint8_t * dst = sm + buf * STB;
        for (int i = lane; i < 16 * W4; i += 32) {
            const int r = i / W4, w = i - r * W4;
            tc_cp4(dst + r * CH + 4 * w, (r < 8 ? gb + r * row_bytes : ub + (r - 8) * row_bytes) + (size_t) st * CH + 4 * w);
        }
        tc_commit();
    };
    for (int c0 = 0; c0 < ncols; c0 += 8 * NG) {
        const block_q8_1 * yb[NG];
        const block_q8_1 * yo[NG][2];
        bool               bv[NG], ov[NG][2];
        int                oc[NG][2];
        float              ag[NG][2], au[NG][2];
#pragma unroll
        for (int G = 0; G < NG; ++G) {
            const int cb = c0 + 8 * G + g;
            bv[G] = cb < ncols;
            yb[G] = y + (size_t) (bv[G] ? o[2 + cb] / k : 0) * ystr;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                ov[G][q] = co < ncols;
                oc[G][q] = ov[G][q] ? o[2 + co] : 0;
                yo[G][q] = y + (size_t) (oc[G][q] / k) * ystr;
                ag[G][q] = au[G][q] = 0.0f;
            }
        }
        load(0, 0);
        if (nst > 1) load(1, 1);
        for (int st = 0; st < nst; ++st) {
            if (st + 2 < nst) { load(st + 2, (st + 2) % kTcStages); tc_wait<2>(); }
            else if (st + 1 < nst) { tc_wait<1>(); }
            else { tc_wait<0>(); }
            __syncwarp();
            const uint8_t * buf = sm + (st % kTcStages) * STB;
#pragma unroll
            for (int sb = 0; sb < kTcKC; ++sb) {
#pragma unroll 2
                for (int sub = 0; sub < 8; ++sub) {
                    const int u = (st * kTcKC + sb) * 8 + sub;
                    int   gv0, gv1, gm0, gm1, uv0, uv1, um0, um1;
                    float gsw, usw;
                    Q::unit(buf + g * CH + sb * Q::SB, sub, tg, gv0, gv1, gm0, gm1, gsw);
                    Q::unit(buf + (8 + g) * CH + sb * Q::SB, sub, tg, uv0, uv1, um0, um1, usw);
#pragma unroll
                    for (int G = 0; G < NG; ++G) {
                        const int b0 = bv[G] ? get_int_b4(yb[G][u].qs, tg) : 0;
                        const int b1 = bv[G] ? get_int_b4(yb[G][u].qs, 4 + tg) : 0;
                        int       c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                        tc_mma(c, gv0, uv0, b0);
                        tc_mma(e, gv1, uv1, b1);
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const float d8 = ov[G][q] ? __low2float(yo[G][q][u].ds) : 0.0f;
                            ag[G][q] = fmaf(gsw * d8, (float) (c[q] * gm0 + e[q] * gm1), ag[G][q]);
                            au[G][q] = fmaf(usw * d8, (float) (c[2 + q] * um0 + e[2 + q] * um1), au[G][q]);
                        }
                    }
                }
            }
            __syncwarp();
        }
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                if (ov[G][q]) act[(long long) oc[G][q] * F + r0 + g] = silu(ag[G][q]) * au[G][q];
    }
}

// down: the whole 16-row tile (16 * row_bytes, contiguous, 16-byte aligned) staged once; column groups reuse it.
// Column c's input is act row c. grid (ceil(N / (16 * kTcWarps)), owners).
template <ggml_type type, int NG>
__global__ void __launch_bounds__(32 * kTcWarps) dn_tc_k(const uint8_t * __restrict__ down, const block_q8_1 * __restrict__ y,
                                                         const int32_t * __restrict__ recs, int rec_stride,
                                                         const int * __restrict__ count_dev, int count, float * __restrict__ out,
                                                         int F, int N, long long row_bytes, long long exp_bytes) {
    using Q = TQ<type>;
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    if ((int) blockIdx.y >= (count_dev ? *count_dev : count)) return;
    const int r0 = (blockIdx.x * kTcWarps + warp) * kTcDnRows;
    if (r0 >= N) return;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], ystr = F / QK8_1, nu = F / QK8_1;
    const int       tb = (int) (kTcDnRows * row_bytes);
    uint8_t *       sm = tc_sm + warp * tb;
    const uint8_t * src = down + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    for (int i = lane; i < tb / 16; i += 32) tc_cp16(sm + 16 * i, src + 16 * i);
    tc_commit();
    tc_wait<0>();
    __syncwarp();
    const uint8_t * ra = sm + g * row_bytes, * rb = sm + (g + 8) * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += 8 * NG) {
        const block_q8_1 * yb[NG];
        const block_q8_1 * yo[NG][2];
        bool               bv[NG], ov[NG][2];
        int                oc[NG][2];
        float              acc[NG][4];
#pragma unroll
        for (int G = 0; G < NG; ++G) {
            const int cb = c0 + 8 * G + g;
            bv[G] = cb < ncols;
            yb[G] = y + (size_t) (bv[G] ? o[2 + cb] : 0) * ystr;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                ov[G][q] = co < ncols;
                oc[G][q] = ov[G][q] ? o[2 + co] : 0;
                yo[G][q] = y + (size_t) oc[G][q] * ystr;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[G][q] = 0.0f;
        }
#pragma unroll 2
        for (int u = 0; u < nu; ++u) {
            const int blk = u / Q::UPB, sub = u - blk * Q::UPB;
            int   av0, av1, am0, am1, bv0, bv1, bm0, bm1;
            float asw, bsw;
            Q::unit(ra + blk * Q::SB, sub, tg, av0, av1, am0, am1, asw);
            Q::unit(rb + blk * Q::SB, sub, tg, bv0, bv1, bm0, bm1, bsw);
#pragma unroll
            for (int G = 0; G < NG; ++G) {
                const int b0 = bv[G] ? get_int_b4(yb[G][u].qs, tg) : 0;
                const int b1 = bv[G] ? get_int_b4(yb[G][u].qs, 4 + tg) : 0;
                int       c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                tc_mma(c, av0, bv0, b0);
                tc_mma(e, av1, bv1, b1);
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const float d8 = ov[G][q] ? __low2float(yo[G][q][u].ds) : 0.0f;
                    acc[G][q]     = fmaf(asw * d8, (float) (c[q] * am0 + e[q] * am1), acc[G][q]);
                    acc[G][2 + q] = fmaf(bsw * d8, (float) (c[2 + q] * bm0 + e[2 + q] * bm1), acc[G][2 + q]);
                }
            }
        }
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                if (ov[G][q]) {
                    out[(long long) oc[G][q] * N + r0 + g]     = acc[G][q];
                    out[(long long) oc[G][q] * N + r0 + g + 8] = acc[G][2 + q];
                }
    }
}

// Prefill down projection (AI_Q4X_PF_DN=1): dn_tc_k over W-warp blocks (16 rows per warp) with the columns' q8_1
// activations staged once per block per column group, instead of every block of an expert reloading them from
// global memory. Rows padded to 784 bytes (196 words = 4 mod 32: the 8 columns a warp reads at once fall in distinct banks).
// Same per-(row, column, 32-block) arithmetic as dn_tc_k: bit-identical outputs.
template <ggml_type type, int NG, int W>
__global__ void __launch_bounds__(32 * W) dn_s_k(const uint8_t * __restrict__ down, const block_q8_1 * __restrict__ y,
                                                 const int32_t * __restrict__ recs, int rec_stride, int count,
                                                 float * __restrict__ out, int F, int N, long long row_bytes, long long exp_bytes,
                                                 int astride, int del) {
    using Q = TQ<type>;
    constexpr int NC = 8 * NG;
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3, tid = threadIdx.x;
    if ((int) blockIdx.y >= count) return;
    const int       r0 = (blockIdx.x * W + warp) * kTcDnRows;
    const bool      active = r0 < N;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], nu = F / QK8_1;
    const int       tb = (int) (kTcDnRows * row_bytes), rowb = nu * (int) sizeof(block_q8_1);
    uint8_t *       wsm = tc_sm + warp * tb;
    uint8_t *       asm_ = tc_sm + W * tb;
    if (active) {
        const uint8_t * src = down + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
        for (int i = lane; i < tb / 16; i += 32) tc_cp16(wsm + 16 * i, src + 16 * i);
    }
    const uint8_t * ra = wsm + g * row_bytes, * rb = wsm + (g + 8) * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        const int cpc = rowb / 16;   // 16-byte chunks per column
        for (int i = tid; i < NC * cpc; i += 32 * W) {
            const int c = i / cpc, part = i - c * cpc;
            if (c0 + c >= ncols || (del & 2)) continue;
            tc_cp16(asm_ + c * astride + 16 * part, (const uint8_t *) (y + (size_t) o[2 + c0 + c] * nu) + 16 * part);
        }
        tc_commit();
        tc_wait<0>();
        __syncthreads();
        bool  ov[NG][2];
        int   oc[NG][2];
        float acc[NG][4];
#pragma unroll
        for (int G = 0; G < NG; ++G) {
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                ov[G][q] = co < ncols;
                oc[G][q] = ov[G][q] ? o[2 + co] : 0;
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[G][q] = 0.0f;
        }
        if (active) {
#pragma unroll 2
            for (int u = 0; u < nu; ++u) {
                const int blk = u / Q::UPB, sub = u - blk * Q::UPB;
                int   av0, av1, am0, am1, bv0, bv1, bm0, bm1;
                float asw, bsw;
                if (del & 1) { av0 = lane; av1 = u; bv0 = tg; bv1 = lane ^ u; am0 = bm0 = 1; am1 = bm1 = 2; asw = bsw = 1.0f; }
                else {
                    Q::unit(ra + blk * Q::SB, sub, tg, av0, av1, am0, am1, asw);
                    Q::unit(rb + blk * Q::SB, sub, tg, bv0, bv1, bm0, bm1, bsw);
                }
#pragma unroll
                for (int G = 0; G < NG; ++G) {
                    const uint8_t * yb = asm_ + (8 * G + g) * astride + u * (int) sizeof(block_q8_1) + 4;   // qs after ds
                    const int       b0 = *(const int *) (yb + 4 * tg), b1 = *(const int *) (yb + 16 + 4 * tg);
                    int             c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                    if (del & 4) { c[0] = b0 ^ av0; c[1] = bv0; c[2] = b1; c[3] = av1 ^ bv1; e[0] = b1 ^ bv1; }
                    else { tc_mma(c, av0, bv0, b0); tc_mma(e, av1, bv1, b1); }
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const block_q8_1 * yq = (const block_q8_1 *) (asm_ + (8 * G + 2 * tg + q) * astride) + u;
                        const float        d8 = ov[G][q] ? __low2float(yq->ds) : 0.0f;
                        acc[G][q]     = fmaf(asw * d8, (float) (c[q] * am0 + e[q] * am1), acc[G][q]);
                        acc[G][2 + q] = fmaf(bsw * d8, (float) (c[2 + q] * bm0 + e[2 + q] * bm1), acc[G][2 + q]);
                    }
                }
            }
#pragma unroll
            for (int G = 0; G < NG; ++G)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    if (ov[G][q] && !(del & 8)) {
                        out[(long long) oc[G][q] * N + r0 + g]     = acc[G][q];
                        out[(long long) oc[G][q] * N + r0 + g + 8] = acc[G][2 + q];
                    }
        }
        __syncthreads();   // every warp is done with the activation tile before the next group overwrites it
    }
}

template <ggml_type type, int NG, int W>
static void launch_dn_s(cudaStream_t st, const void * down, const block_q8_1 * y, const int32_t * recs, int rec_stride,
                        int count, float * out, int F, int N, long long rb, long long eb) {
    const int    rowb    = F / QK8_1 * (int) sizeof(block_q8_1);
    int          astride = (rowb + 15) / 16 * 16;
    while ((astride / 4) % 32 != 4) astride += 16;   // 8 columns -> 8 distinct bank groups
    const size_t sm = (size_t) W * kTcDnRows * rb + (size_t) (8 * NG) * astride;
    static bool attr = false;
    if (!attr) {
        attr = true;
        cudaFuncSetAttribute(dn_s_k<type, NG, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    }
    const dim3 grid((unsigned) ((N + kTcDnRows * W - 1) / (kTcDnRows * W)), (unsigned) count);
    // AI_Q4X_DN_DEL (default 0, benchmarking only, wrong output): bitmask of stages to skip to find what bounds the
    // kernel: 1 weight decode, 2 activation staging, 4 mma, 8 stores
    static const int del = getenv("AI_Q4X_DN_DEL") ? atoi(getenv("AI_Q4X_DN_DEL")) : 0;
    dn_s_k<type, NG, W><<<grid, 32 * W, sm, st>>>((const uint8_t *) down, y, recs, rec_stride, count, out, F, N, rb, eb, astride, del);
}

// Prefill gate/up for big batches (AI_Q4X_PF_GU=1): gu_tc_k with NG column groups per weight pass and register-lean
// per-group state: an int activation row per column (-1 = none) instead of pointers + flags, so NG = 16 (128 columns
// per decode of the weight tiles) fits in registers where gu_tc_k at NG = 16 spills. The per-(row, column,
// 32-block) term and its K order are gu_tc_k's -- the same fmaf sequence -- so the outputs are bit-identical.
template <ggml_type type, int NG>
__global__ void __launch_bounds__(32 * kTcWarps) gu_pf_k(const uint8_t * __restrict__ up, const uint8_t * __restrict__ gate,
                                                         const block_q8_1 * __restrict__ y, const int32_t * __restrict__ recs,
                                                         int rec_stride, int count, float * __restrict__ act, int K, int F,
                                                         long long row_bytes, long long exp_bytes, int k) {
    using Q = TQ<type>;
    constexpr int CH = kTcKC * Q::SB, W4 = CH / 4, STB = 16 * CH;
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    if ((int) blockIdx.y >= count) return;
    const int r0 = (blockIdx.x * kTcWarps + warp) * kTcGuRows;
    if (r0 >= F) return;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], ystr = K / QK8_1, nst = K / (QK_K * kTcKC);
    uint8_t *       sm = tc_sm + warp * (kTcStages * STB);
    const uint8_t * gb = gate + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    const uint8_t * ub = up + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    auto load = [&](int st, int buf) {
        uint8_t * dst = sm + buf * STB;
        for (int i = lane; i < 16 * W4; i += 32) {
            const int r = i / W4, w = i - r * W4;
            tc_cp4(dst + r * CH + 4 * w, (r < 8 ? gb + r * row_bytes : ub + (r - 8) * row_bytes) + (size_t) st * CH + 4 * w);
        }
        tc_commit();
    };
    for (int c0 = 0; c0 < ncols; c0 += 8 * NG) {
        int   bt[NG], dt[NG][2];   // activation rows (column / k), -1 past ncols
        float ag[NG][2], au[NG][2];
#pragma unroll
        for (int G = 0; G < NG; ++G) {
            const int cb = c0 + 8 * G + g;
            bt[G] = cb < ncols ? o[2 + cb] / k : -1;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                dt[G][q] = co < ncols ? o[2 + co] / k : -1;
                ag[G][q] = au[G][q] = 0.0f;
            }
        }
        load(0, 0);
        if (nst > 1) load(1, 1);
        for (int st = 0; st < nst; ++st) {
            if (st + 2 < nst) { load(st + 2, (st + 2) % kTcStages); tc_wait<2>(); }
            else if (st + 1 < nst) { tc_wait<1>(); }
            else { tc_wait<0>(); }
            __syncwarp();
            const uint8_t * buf = sm + (st % kTcStages) * STB;
#pragma unroll
            for (int sb = 0; sb < kTcKC; ++sb) {
#pragma unroll 2
                for (int sub = 0; sub < 8; ++sub) {
                    const int u = (st * kTcKC + sb) * 8 + sub;
                    int   gv0, gv1, gm0, gm1, uv0, uv1, um0, um1;
                    float gsw, usw;
                    Q::unit(buf + g * CH + sb * Q::SB, sub, tg, gv0, gv1, gm0, gm1, gsw);
                    Q::unit(buf + (8 + g) * CH + sb * Q::SB, sub, tg, uv0, uv1, um0, um1, usw);
#pragma unroll
                    for (int G = 0; G < NG; ++G) {
                        const block_q8_1 * yb = y + (size_t) (bt[G] < 0 ? 0 : bt[G]) * ystr + u;
                        const int b0 = bt[G] >= 0 ? get_int_b4(yb->qs, tg) : 0;
                        const int b1 = bt[G] >= 0 ? get_int_b4(yb->qs, 4 + tg) : 0;
                        int       c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                        tc_mma(c, gv0, uv0, b0);
                        tc_mma(e, gv1, uv1, b1);
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            const float d8 = dt[G][q] >= 0 ? __low2float(y[(size_t) dt[G][q] * ystr + u].ds) : 0.0f;
                            ag[G][q] = fmaf(gsw * d8, (float) (c[q] * gm0 + e[q] * gm1), ag[G][q]);
                            au[G][q] = fmaf(usw * d8, (float) (c[2 + q] * um0 + e[2 + q] * um1), au[G][q]);
                        }
                    }
                }
            }
            __syncwarp();
        }
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                if (dt[G][q] >= 0) act[(long long) o[2 + c0 + 8 * G + 2 * tg + q] * F + r0 + g] = silu(ag[G][q]) * au[G][q];
    }
}

template <ggml_type type, int NG>
void launch_gu_pf(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const int32_t * recs,
                  int rec_stride, int count, float * act, int K, int F, long long rb, long long eb, int k) {
    const size_t sm = (size_t) kTcWarps * kTcStages * 16 * kTcKC * TQ<type>::SB;
    const dim3   grid((unsigned) ((F + kTcGuRows * kTcWarps - 1) / (kTcGuRows * kTcWarps)), (unsigned) count);
    gu_pf_k<type, NG><<<grid, 32 * kTcWarps, sm, st>>>((const uint8_t *) up, (const uint8_t *) gate, y, recs, rec_stride,
                                                       count, act, K, F, rb, eb, k);
}

// AI_Q4X_PF_GU=2: gu_pf_k with MT row tiles per warp (8 gate + 8 up rows each): every activation fragment (b0, b1)
// and scale loaded for a column group feeds 2 * MT mma instead of 2 -- the loads per mma fall MT-fold while the weight
// decode per element is unchanged. Same per-(row, column, 32-block) fmaf sequence: bit-identical. kPfWarps warps per
// block (the MT-tile stages take 3 * 16 * MT * CH bytes per warp).
constexpr int kPfWarps = 2;
template <ggml_type type, int NG, int MT>
__global__ void __launch_bounds__(32 * kPfWarps) gu_pfm_k(const uint8_t * __restrict__ up, const uint8_t * __restrict__ gate,
                                                          const block_q8_1 * __restrict__ y, const int32_t * __restrict__ recs,
                                                          int rec_stride, int count, float * __restrict__ act, int K, int F,
                                                          long long row_bytes, long long exp_bytes, int k) {
    using Q = TQ<type>;
    constexpr int CH = kTcKC * Q::SB, W4 = CH / 4, STB = 16 * MT * CH;
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    if ((int) blockIdx.y >= count) return;
    const int r0 = (blockIdx.x * kPfWarps + warp) * 8 * MT;
    if (r0 >= F) return;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], ystr = K / QK8_1, nst = K / (QK_K * kTcKC);
    uint8_t *       sm = tc_sm + warp * (kTcStages * STB);
    const uint8_t * gb = gate + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    const uint8_t * ub = up + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    // stage layout: tile t occupies rows 16 t .. 16 t + 15 (8 gate rows then 8 up rows)
    auto load = [&](int st, int buf) {
        uint8_t * dst = sm + buf * STB;
        for (int i = lane; i < 16 * MT * W4; i += 32) {
            const int r = i / W4, w = i - r * W4, t = r >> 4, rr = r & 15;
            const uint8_t * src = rr < 8 ? gb + (8 * t + rr) * row_bytes : ub + (8 * t + rr - 8) * row_bytes;
            if (r0 + 8 * t + (rr & 7) < F) tc_cp4(dst + r * CH + 4 * w, src + (size_t) st * CH + 4 * w);
        }
        tc_commit();
    };
    for (int c0 = 0; c0 < ncols; c0 += 8 * NG) {
        int   bt[NG], dt[NG][2];
        float ag[MT][NG][2], au[MT][NG][2];
#pragma unroll
        for (int G = 0; G < NG; ++G) {
            const int cb = c0 + 8 * G + g;
            bt[G] = cb < ncols ? o[2 + cb] / k : -1;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                dt[G][q] = co < ncols ? o[2 + co] / k : -1;
#pragma unroll
                for (int t = 0; t < MT; ++t) ag[t][G][q] = au[t][G][q] = 0.0f;
            }
        }
        load(0, 0);
        if (nst > 1) load(1, 1);
        for (int st = 0; st < nst; ++st) {
            if (st + 2 < nst) { load(st + 2, (st + 2) % kTcStages); tc_wait<2>(); }
            else if (st + 1 < nst) { tc_wait<1>(); }
            else { tc_wait<0>(); }
            __syncwarp();
            const uint8_t * buf = sm + (st % kTcStages) * STB;
#pragma unroll
            for (int sb = 0; sb < kTcKC; ++sb) {
#pragma unroll 1
                for (int sub = 0; sub < 8; ++sub) {
                    const int u = (st * kTcKC + sb) * 8 + sub;
                    int   gv0[MT], gv1[MT], gm0[MT], gm1[MT], uv0[MT], uv1[MT], um0[MT], um1[MT];
                    float gsw[MT], usw[MT];
#pragma unroll
                    for (int t = 0; t < MT; ++t) {
                        Q::unit(buf + (16 * t + g) * CH + sb * Q::SB, sub, tg, gv0[t], gv1[t], gm0[t], gm1[t], gsw[t]);
                        Q::unit(buf + (16 * t + 8 + g) * CH + sb * Q::SB, sub, tg, uv0[t], uv1[t], um0[t], um1[t], usw[t]);
                    }
#pragma unroll
                    for (int G = 0; G < NG; ++G) {
                        const block_q8_1 * yb = y + (size_t) (bt[G] < 0 ? 0 : bt[G]) * ystr + u;
                        const int   b0  = bt[G] >= 0 ? get_int_b4(yb->qs, tg) : 0;
                        const int   b1  = bt[G] >= 0 ? get_int_b4(yb->qs, 4 + tg) : 0;
                        const float d80 = dt[G][0] >= 0 ? __low2float(y[(size_t) dt[G][0] * ystr + u].ds) : 0.0f;
                        const float d81 = dt[G][1] >= 0 ? __low2float(y[(size_t) dt[G][1] * ystr + u].ds) : 0.0f;
#pragma unroll
                        for (int t = 0; t < MT; ++t) {
                            int c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                            tc_mma(c, gv0[t], uv0[t], b0);
                            tc_mma(e, gv1[t], uv1[t], b1);
                            ag[t][G][0] = fmaf(gsw[t] * d80, (float) (c[0] * gm0[t] + e[0] * gm1[t]), ag[t][G][0]);
                            au[t][G][0] = fmaf(usw[t] * d80, (float) (c[2] * um0[t] + e[2] * um1[t]), au[t][G][0]);
                            ag[t][G][1] = fmaf(gsw[t] * d81, (float) (c[1] * gm0[t] + e[1] * gm1[t]), ag[t][G][1]);
                            au[t][G][1] = fmaf(usw[t] * d81, (float) (c[3] * um0[t] + e[3] * um1[t]), au[t][G][1]);
                        }
                    }
                }
            }
            __syncwarp();
        }
#pragma unroll
        for (int t = 0; t < MT; ++t)
#pragma unroll
            for (int G = 0; G < NG; ++G)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    if (dt[G][q] >= 0 && r0 + 8 * t + g < F)
                        act[(long long) o[2 + c0 + 8 * G + 2 * tg + q] * F + r0 + 8 * t + g] = silu(ag[t][G][q]) * au[t][G][q];
    }
}

template <ggml_type type, int NG, int MT>
void launch_gu_pfm(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const int32_t * recs,
                   int rec_stride, int count, float * act, int K, int F, long long rb, long long eb, int k) {
    const size_t sm = (size_t) kPfWarps * kTcStages * 16 * MT * kTcKC * TQ<type>::SB;
    static bool attr = false;
    if (!attr) {
        attr = true;
        cudaFuncSetAttribute(gu_pfm_k<type, NG, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    }
    const dim3 grid((unsigned) ((F + 8 * MT * kPfWarps - 1) / (8 * MT * kPfWarps)), (unsigned) count);
    gu_pfm_k<type, NG, MT><<<grid, 32 * kPfWarps, sm, st>>>((const uint8_t *) up, (const uint8_t *) gate, y, recs,
                                                            rec_stride, count, act, K, F, rb, eb, k);
}

template <ggml_type type, int NG>
void launch_gu_tc(cudaStream_t st, const void * up, const void * gate, const block_q8_1 * y, const int32_t * recs,
                  int rec_stride, const int * count_dev, int count, int grid_y, float * act, int K, int F,
                  long long rb, long long eb, int k) {
    const size_t sm = (size_t) kTcWarps * kTcStages * 16 * kTcKC * TQ<type>::SB;
    const dim3   grid((unsigned) ((F + kTcGuRows * kTcWarps - 1) / (kTcGuRows * kTcWarps)), (unsigned) grid_y);
    gu_tc_k<type, NG><<<grid, 32 * kTcWarps, sm, st>>>((const uint8_t *) up, (const uint8_t *) gate, y, recs, rec_stride,
                                                       count_dev, count, act, K, F, rb, eb, k);
}

template <ggml_type type, int NG>
void launch_dn_tc(cudaStream_t st, const void * down, const block_q8_1 * y, const int32_t * recs, int rec_stride,
                  const int * count_dev, int count, int grid_y, float * out, int F, int N, long long rb, long long eb) {
    const size_t sm = (size_t) kTcWarps * kTcDnRows * rb;
    const dim3   grid((unsigned) ((N + kTcDnRows * kTcWarps - 1) / (kTcDnRows * kTcWarps)), (unsigned) grid_y);
    dn_tc_k<type, NG><<<grid, 32 * kTcWarps, sm, st>>>((const uint8_t *) down, y, recs, rec_stride, count_dev, count,
                                                       out, F, N, rb, eb);
}

// NG = column groups of 8 per pass: 1 for decode (an owner serves <= 8 columns), 4 for wide windows
template <int NG>
void tc_dispatch(cudaStream_t st, int phase, int tyg, const void * up, const void * gate, long long rbg, long long ebg,
                 int tyd, const void * down, long long rbd, long long ebd, const block_q8_1 * xq, const block_q8_1 * aq,
                 const int32_t * recs, int rec_stride, const int * count_dev, int count, int grid_y, float * act,
                 float * out, int K, int F, int N, int k) {
    if (phase == 0) {
        switch (tyg) {
            case GGML_TYPE_IQ2_XXS: launch_gu_tc<GGML_TYPE_IQ2_XXS, NG>(st, up, gate, xq, recs, rec_stride, count_dev, count, grid_y, act, K, F, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_XS:  launch_gu_tc<GGML_TYPE_IQ2_XS, NG>(st, up, gate, xq, recs, rec_stride, count_dev, count, grid_y, act, K, F, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_S:   launch_gu_tc<GGML_TYPE_IQ2_S, NG>(st, up, gate, xq, recs, rec_stride, count_dev, count, grid_y, act, K, F, rbg, ebg, k); break;
            case GGML_TYPE_IQ3_XXS: launch_gu_tc<GGML_TYPE_IQ3_XXS, NG>(st, up, gate, xq, recs, rec_stride, count_dev, count, grid_y, act, K, F, rbg, ebg, k); break;
            default:                launch_gu_tc<GGML_TYPE_IQ3_S, NG>(st, up, gate, xq, recs, rec_stride, count_dev, count, grid_y, act, K, F, rbg, ebg, k); break;
        }
    } else if (tyd == GGML_TYPE_IQ4_NL) {
        launch_dn_tc<GGML_TYPE_IQ4_NL, NG>(st, down, aq, recs, rec_stride, count_dev, count, grid_y, out, F, N, rbd, ebd);
    } else {
        launch_dn_tc<GGML_TYPE_Q2_0, NG>(st, down, aq, recs, rec_stride, count_dev, count, grid_y, out, F, N, rbd, ebd);
    }
}

// AI_Q4X_TC: 1 = the tensor-core kernels for every width (decode, verify, prefill). q4x_experts_set_tc overrides.
// Kernel family per call. Decode / verify (q4x_experts): dp4a with global loads at T <= 4 (faster for few columns),
// the smem-staged dp4a at T > 4. The two are bit-identical to each other, so a window's columns match plain's
// whatever T is. AI_Q4X_TC=1 puts the tensor-core kernels on decode too (not bit-identical to dp4a; experimental).
// Wide windows / prefill (q4x_experts_wide): tensor cores by default (about 2x faster than dp4a at T = 512, rel-L2
// ~1e-4 against it), AI_Q4X_WIDE_TC=0 for the dp4a wide kernel. q4x_experts_set_tc(f) forces family f everywhere (tools, comparisons):
// 0 = dp4a global, 1 = tensor cores, 2 = dp4a smem.
int g_tc_override = -1;
int tc_mode(int64_t T) {
    if (g_tc_override >= 0) return g_tc_override;
    static const bool tc = getenv("AI_Q4X_TC") && atoi(getenv("AI_Q4X_TC")) == 1;
    static const bool sm = !getenv("AI_Q4X_EXP_SMEM") || atoi(getenv("AI_Q4X_EXP_SMEM")) != 0;
    return tc ? 1 : (sm && T > 4) ? 2 : 0;
}
int wide_mode() {
    if (g_tc_override >= 0) return g_tc_override == 1 ? 1 : 0;
    static const bool tc = !getenv("AI_Q4X_WIDE_TC") || atoi(getenv("AI_Q4X_WIDE_TC")) != 0;
    return tc ? 1 : 0;
}
// the TC path's own shape/alignment demands: cp.async 4 B (gate/up) / 16 B (down), 8- and 16-row tiles
bool tc_ok(const void * up, const void * gate, int64_t rbg, int64_t ebg, const void * down, int64_t rbd, int64_t ebd,
           int64_t K, int64_t F, int64_t N) {
    return K % (QK_K * kTcKC) == 0 && F % (kTcGuRows) == 0 && N % kTcDnRows == 0 && F % QK8_1 == 0 && rbg % 4 == 0 &&
           ebg % 4 == 0 && (uintptr_t) up % 4 == 0 && (uintptr_t) gate % 4 == 0 && (kTcDnRows * rbd) % 16 == 0 &&
           ebd % 16 == 0 && (uintptr_t) down % 16 == 0 && kTcWarps * kTcDnRows * rbd <= 48 * 1024;
}

}  // namespace

namespace ai::kern {

bool q4x_experts_ok(int ty_gu, int ty_down, int64_t K, int64_t F, int64_t N, int k, int64_t T) {
    return gu_type(ty_gu) && (ty_down == GGML_TYPE_IQ4_NL || ty_down == GGML_TYPE_Q2_0) && T >= 1 && T <= kMaxT &&
           k >= 1 && k <= kMaxK && k * T <= kMaxCols && K % QK_K == 0 && F % QK2_0 == 0 && F % QK4_NL == 0 &&
           N % 4 == 0 && F % 4 == 0;
}

size_t q4x_experts_scratch_bytes(int64_t K, int64_t F, int k, int64_t T) {
    const size_t xq = (size_t) (K / QK8_1) * sizeof(block_q8_1) * T, act = (size_t) F * k * T * sizeof(float);
    const size_t aq = (size_t) (F / QK8_1) * sizeof(block_q8_1) * k * T;
    return ((xq + 255) / 256 + (act + 255) / 256 + (aq + 255) / 256 + (sizeof(Plan) + 255) / 256) * 256;
}

static bool experts_impl(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                         int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                         int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * ids,
                         int64_t ids_stride, int32_t id_lo, int32_t id_hi, bool force_smem, int64_t K, int64_t F,
                         int64_t N, int k, int64_t T, void * scratch, float * out) {
    const int filler0 = id_hi;
    if (!q4x_experts_ok(ty_gu, ty_down, K, F, N, k, T)) return false;
    const size_t bg = bsize(ty_gu), bd = bsize(ty_down);
    if (row_bytes_gu % bg || expert_bytes_gu % bg || row_bytes_down % bd || expert_bytes_down % bd) return false;
    if (expert_bytes_gu / bg > INT32_MAX / (filler0 + 64)) return false;
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[q4x-experts] launched: grouped gate/up+SwiGLU and down, slot dedup, missed columns skipped\n");
    }
    cudaStream_t st = (cudaStream_t) stream;
    char *       s  = (char *) scratch;
    const size_t xq_b = (size_t) (K / QK8_1) * sizeof(block_q8_1) * T, act_b = (size_t) F * k * T * sizeof(float);
    auto *       xq   = (block_q8_1 *) s;
    auto *       act  = (float *) (s + (xq_b + 255) / 256 * 256);
    auto *       aq   = (block_q8_1 *) ((char *) act + (act_b + 255) / 256 * 256);
    const size_t aq_b = (size_t) (F / QK8_1) * sizeof(block_q8_1) * k * T;
    auto *       plan = (Plan *) ((char *) aq + (aq_b + 255) / 256 * 256);
    const int    n    = (int) (k * T);
    plan_k<<<1, (kMaxCols + 31) / 32 * 32, 0, st>>>(ids, (int) ids_stride, k, n, id_lo, id_hi, plan);
    quantize_row_q8_1_cuda(x, nullptr, xq, GGML_TYPE_Q8_0, K, x_stride, x_stride * T, x_stride * T, K, T, 1, 1, st);
    const int srg = (int) (row_bytes_gu / bg), scg = (int) (expert_bytes_gu / bg);
    const int srd = (int) (row_bytes_down / bd), scd = (int) (expert_bytes_down / bd), ist = (int) ids_stride;
    const int t = (int) T, Ki = (int) K, Fi = (int) F, Ni = (int) N;
    (void) ist;
    // above 8 rows an owner can serve > 8 columns: only the chunked smem kernels (and the TC ones) take that
    const int fam = force_smem ? 2 : T > 8 && tc_mode(T) == 0 ? 2 : tc_mode(T);
    if (fam == 1 && tc_ok(up, gate, row_bytes_gu, expert_bytes_gu, down, row_bytes_down, expert_bytes_down, K, F, N)) {
        static bool said_tc = false;
        if (!said_tc) { said_tc = true; fprintf(stderr, "[q4x-experts] tensor-core grouped experts (AI_Q4X_TC=1)\n"); }
        const int32_t * recs = (const int32_t *) plan + 4;   // Plan::own[i] = { ch, ne, cs[8] }: 10 ints
        const int *     cnt  = &plan->count;
        tc_dispatch<1>(st, 0, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down,
                       expert_bytes_down, xq, aq, recs, (int) (sizeof(Own) / 4), cnt, 0, n, act, out, Ki, Fi, Ni, k);
        quantize_row_q8_1_cuda(act, nullptr, aq, GGML_TYPE_Q8_0, F, F, F * n, F * n, F, n, 1, 1, st);
        tc_dispatch<1>(st, 1, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down,
                       expert_bytes_down, xq, aq, recs, (int) (sizeof(Own) / 4), cnt, 0, n, act, out, Ki, Fi, Ni, k);
        return true;
    }
    if (fam == 2 && smem_ok(ty_gu, ty_down, up, gate, row_bytes_gu, expert_bytes_gu, down, row_bytes_down, expert_bytes_down, F, N)) {
        static bool said_sm = false;
        if (!said_sm) { said_sm = true; fprintf(stderr, "[q4x-experts] smem-staged grouped experts at T > 4 (AI_Q4X_EXP_SMEM=1)\n"); }
        dispatch_smem(st, ty_gu, up, gate, srg, scg, ty_down, down, srd, scd, xq, plan, act, aq, out, Ki, Fi, Ni, k, n, 0);
        quantize_row_q8_1_cuda(act, nullptr, aq, GGML_TYPE_Q8_0, F, F, F * n, F * n, F, n, 1, 1, st);
        dispatch_smem(st, ty_gu, up, gate, srg, scg, ty_down, down, srd, scd, xq, plan, act, aq, out, Ki, Fi, Ni, k, n, 1);
        return true;
    }
    for (int phase = 0; phase < 2; ++phase) {
        if (phase == 1) {
            // missed columns of act are unwritten: their q8_1 rows are never read (dn_k skips the same columns)
            quantize_row_q8_1_cuda(act, nullptr, aq, GGML_TYPE_Q8_0, F, F, F * n, F * n, F, n, 1, 1, st);
        }
        if (t <= 4) dispatch<4>(st, ty_gu, up, gate, srg, scg, ty_down, down, srd, scd, xq, plan, act, aq, out, Ki, Fi, Ni, k, n, phase);
        else        dispatch<8>(st, ty_gu, up, gate, srg, scg, ty_down, down, srd, scd, xq, plan, act, aq, out, Ki, Fi, Ni, k, n, phase);
    }
    return true;
}


bool q4x_experts(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                 int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                 int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * ids, int64_t ids_stride,
                 int filler0, int64_t K, int64_t F, int64_t N, int k, int64_t T, void * scratch, float * out) {
    return experts_impl(stream, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down,
                        expert_bytes_down, x, x_stride, ids, ids_stride, 0, filler0, false, K, F, N, k, T, scratch, out);
}

bool q4x_experts_range(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                       int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                       int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * ids,
                       int64_t ids_stride, int32_t id_lo, int32_t id_hi, int64_t K, int64_t F, int64_t N, int k,
                       int64_t T, void * scratch, float * out) {
    if (!smem_ok(ty_gu, ty_down, up, gate, row_bytes_gu, expert_bytes_gu, down, row_bytes_down, expert_bytes_down, F, N)) return false;
    return experts_impl(stream, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down,
                        expert_bytes_down, x, x_stride, ids, ids_stride, id_lo, id_hi, true, K, F, N, k, T, scratch, out);
}

int q4x_experts_wide_rec() { return kWideRec; }
void q4x_experts_set_tc(int mode) { g_tc_override = mode; }
int q4x_experts_wide_cols() { return kWideCols; }

bool q4x_experts_wide(void * stream, int ty_gu, const void * up, const void * gate, int64_t row_bytes_gu,
                      int64_t expert_bytes_gu, int ty_down, const void * down, int64_t row_bytes_down,
                      int64_t expert_bytes_down, const float * x, int64_t x_stride, const int32_t * plan, int n_own,
                      int64_t K, int64_t F, int64_t N, int k, int64_t T, void * scratch, float * out) {
    if (!gu_type(ty_gu) || (ty_down != GGML_TYPE_IQ4_NL && ty_down != GGML_TYPE_Q2_0) || K % QK_K || F % QK2_0 ||
        F % QK4_NL || N % 4 || F % 4 || T < 1 || T > 65536) return false;   // records hold <= kWideCols columns each
    const size_t bg = bsize(ty_gu), bd = bsize(ty_down);
    if (row_bytes_gu % bg || expert_bytes_gu % bg || row_bytes_down % bd || expert_bytes_down % bd) return false;
    if (n_own <= 0) return true;   // nothing routed to this bank
    cudaStream_t st = (cudaStream_t) stream;
    char *       s  = (char *) scratch;
    const size_t xq_b = (size_t) (K / QK8_1) * sizeof(block_q8_1) * T, act_b = (size_t) F * k * T * sizeof(float);
    auto *       xq   = (block_q8_1 *) s;
    auto *       act  = (float *) (s + (xq_b + 255) / 256 * 256);
    auto *       aq   = (block_q8_1 *) ((char *) act + (act_b + 255) / 256 * 256);
    const int    n    = (int) (k * T);
    quantize_row_q8_1_cuda(x, nullptr, xq, GGML_TYPE_Q8_0, K, x_stride, x_stride * T, x_stride * T, K, T, 1, 1, st);
    const int srg = (int) (row_bytes_gu / bg), scg = (int) (expert_bytes_gu / bg);
    const int srd = (int) (row_bytes_down / bd), scd = (int) (expert_bytes_down / bd);
    const int Ki = (int) K, Fi = (int) F, Ni = (int) N;
    if (wide_mode() == 1 && tc_ok(up, gate, row_bytes_gu, expert_bytes_gu, down, row_bytes_down, expert_bytes_down, K, F, N)) {
        static bool said_wtc = false;
        if (!said_wtc) { said_wtc = true; fprintf(stderr, "[q4x-experts] wide windows on tensor cores (AI_Q4X_WIDE_TC=1)\n"); }
        // AI_Q4X_WIDE_NG = 4 (default) | 8 | 16: column groups of 8 per weight pass above T = 8. An owner with more
        // columns than 8 * NG re-loads and re-decodes its weight tiles once per extra pass.
        static const int wide_ng = getenv("AI_Q4X_WIDE_NG") ? atoi(getenv("AI_Q4X_WIDE_NG")) : 4;
        auto go = [&](int phase) {
            if (T <= 8) tc_dispatch<1>(st, phase, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down, expert_bytes_down, xq, aq, plan, kWideRec, nullptr, n_own, n_own, act, out, Ki, Fi, Ni, k);
            else if (wide_ng >= 16) tc_dispatch<16>(st, phase, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down, expert_bytes_down, xq, aq, plan, kWideRec, nullptr, n_own, n_own, act, out, Ki, Fi, Ni, k);
            else if (wide_ng >= 8) tc_dispatch<8>(st, phase, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down, expert_bytes_down, xq, aq, plan, kWideRec, nullptr, n_own, n_own, act, out, Ki, Fi, Ni, k);
            else        tc_dispatch<4>(st, phase, ty_gu, up, gate, row_bytes_gu, expert_bytes_gu, ty_down, down, row_bytes_down, expert_bytes_down, xq, aq, plan, kWideRec, nullptr, n_own, n_own, act, out, Ki, Fi, Ni, k);
        };
        go(0);
        quantize_row_q8_1_cuda(act, nullptr, aq, GGML_TYPE_Q8_0, F, F, F * n, F * n, F, n, 1, 1, st);
        go(1);
        return true;
    }
    switch (ty_gu) {
        case GGML_TYPE_IQ2_XXS: launch_gu_w<GGML_TYPE_IQ2_XXS>(st, up, gate, xq, plan, n_own, act, Ki, Fi, srg, scg, k); break;
        case GGML_TYPE_IQ2_XS:  launch_gu_w<GGML_TYPE_IQ2_XS>(st, up, gate, xq, plan, n_own, act, Ki, Fi, srg, scg, k); break;
        case GGML_TYPE_IQ2_S:   launch_gu_w<GGML_TYPE_IQ2_S>(st, up, gate, xq, plan, n_own, act, Ki, Fi, srg, scg, k); break;
        case GGML_TYPE_IQ3_XXS: launch_gu_w<GGML_TYPE_IQ3_XXS>(st, up, gate, xq, plan, n_own, act, Ki, Fi, srg, scg, k); break;
        default:                launch_gu_w<GGML_TYPE_IQ3_S>(st, up, gate, xq, plan, n_own, act, Ki, Fi, srg, scg, k); break;
    }
    // columns this bank does not own are unwritten in act: their q8_1 rows are never read (dn_w_k reads only owners')
    quantize_row_q8_1_cuda(act, nullptr, aq, GGML_TYPE_Q8_0, F, F, F * n, F * n, F, n, 1, 1, st);
    if (ty_down == GGML_TYPE_IQ4_NL) launch_dn_w<GGML_TYPE_IQ4_NL>(st, down, aq, plan, n_own, out, Fi, Ni, srd, scd);
    else                             launch_dn_w<GGML_TYPE_Q2_0>(st, down, aq, plan, n_own, out, Fi, Ni, srd, scd);
    return true;
}

// Prefill gate/up with activations quantised per 256 (AI_Q4X_PF_GU=3; q8_K-style: one f32 scale per superblock). The
// gate/up types' 32-block multipliers are integers (m0 / m1) and their float scale is per superblock, so a whole
// superblock's 8 sub-blocks sum in int32 -- c * m0 + e * m1 per sub-block, one fmaf per superblock instead of 8.
// gu_tc_k (q8_1 activations, one fmaf per 32) is bound by its float epilogue at prefill widths. Numerics: the activation rounding changes (scale per 256, not per 32): NLL-gated.
// int32 range: |c|, |e| <= 16 * 127 * 63 per k16 half, multipliers <= 31, 8 sub-blocks: < 6.4e7.
struct block_q8_256 {
    float  d;
    int8_t qs[QK_K];
};
static_assert(sizeof(block_q8_256) == 4 + QK_K, "block_q8_256 layout");

// one warp per (superblock, row): 8 values per lane, amax by shuffle, d = amax / 127
__global__ void quantize_q8_256_k(const float * __restrict__ x, int64_t x_stride, block_q8_256 * __restrict__ y, int nsb) {
    const int sbk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (sbk >= nsb) return;
    const float * xr = x + (size_t) row * x_stride + (size_t) sbk * QK_K + 8 * lane;
    float v[8], am = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = xr[i]; am = fmaxf(am, fabsf(v[i])); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = am / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    block_q8_256 * b = y + (size_t) row * nsb + sbk;
    int8_t q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = (int8_t) __float2int_rn(v[i] * id);
    // qs starts 4 bytes into a 260-byte block: only 4-byte aligned (an int2 store faulted: misaligned address)
    int * dq = (int *) (b->qs + 8 * lane);
    dq[0]    = *(const int *) q;
    dq[1]    = *(const int *) (q + 4);
    if (lane == 0) b->d = d;
}

template <ggml_type type, int NG>
__global__ void __launch_bounds__(32 * kTcWarps) gu_q8k_k(const uint8_t * __restrict__ up, const uint8_t * __restrict__ gate,
                                                          const block_q8_256 * __restrict__ y, const int32_t * __restrict__ recs,
                                                          int rec_stride, int count, float * __restrict__ act, int K, int F,
                                                          long long row_bytes, long long exp_bytes, int k, int del) {
    using Q = TQ<type>;
    constexpr int CH = kTcKC * Q::SB, W4 = CH / 4, STB = 16 * CH;
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    if ((int) blockIdx.y >= count) return;
    const int r0 = (blockIdx.x * kTcWarps + warp) * kTcGuRows;
    if (r0 >= F) return;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], nsb = K / QK_K, nst = K / (QK_K * kTcKC);
    uint8_t *       sm = tc_sm + warp * (kTcStages * STB);
    const uint8_t * gb = gate + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    const uint8_t * ub = up + (size_t) ch * exp_bytes + (size_t) r0 * row_bytes;
    auto load = [&](int st, int buf) {
        uint8_t * dst = sm + buf * STB;
        for (int i = lane; i < 16 * W4; i += 32) {
            const int r = i / W4, w = i - r * W4;
            tc_cp4(dst + r * CH + 4 * w, (r < 8 ? gb + r * row_bytes : ub + (r - 8) * row_bytes) + (size_t) st * CH + 4 * w);
        }
        tc_commit();
    };
    for (int c0 = 0; c0 < ncols; c0 += 8 * NG) {
        const block_q8_256 * yb[NG];
        const block_q8_256 * yo[NG][2];
        bool                 bv[NG], ov[NG][2];
        int                  oc[NG][2];
        float                ag[NG][2], au[NG][2];
#pragma unroll
        for (int G = 0; G < NG; ++G) {
            const int cb = c0 + 8 * G + g;
            bv[G] = cb < ncols;
            yb[G] = y + (size_t) (bv[G] ? o[2 + cb] / k : 0) * nsb;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                ov[G][q] = co < ncols;
                oc[G][q] = ov[G][q] ? o[2 + co] : 0;
                yo[G][q] = y + (size_t) (oc[G][q] / k) * nsb;
                ag[G][q] = au[G][q] = 0.0f;
            }
        }
        load(0, 0);
        if (nst > 1) load(1, 1);
        for (int st = 0; st < nst; ++st) {
            if (st + 2 < nst) { load(st + 2, (st + 2) % kTcStages); tc_wait<2>(); }
            else if (st + 1 < nst) { tc_wait<1>(); }
            else { tc_wait<0>(); }
            __syncwarp();
            const uint8_t * buf = sm + (st % kTcStages) * STB;
#pragma unroll
            for (int sb = 0; sb < kTcKC; ++sb) {
                const int u = st * kTcKC + sb;   // the superblock
                int       ig[NG][2], iu[NG][2];
#pragma unroll
                for (int G = 0; G < NG; ++G) ig[G][0] = ig[G][1] = iu[G][0] = iu[G][1] = 0;
                float gsw = 0.0f, usw = 0.0f;
#pragma unroll 2
                for (int sub = 0; sub < 8; ++sub) {
                    int gv0, gv1, gm0, gm1, uv0, uv1, um0, um1;
                    // AI_Q4X_GU_DEL (profiling only, output is wrong): bit 0 skips the weight decode, 1 the activation
                    // loads, 2 the mma, 3 the float epilogue, to find the bound by deletion (ai-q4x-expbench --path gu)
                    if (del & 1) {
                        gv0 = uv0 = 0x01010101 ^ lane; gv1 = uv1 = 0x02020202 ^ lane; gm0 = gm1 = um0 = um1 = 1;
                        gsw = usw = 1.0f;
                    } else {
                        Q::unit(buf + g * CH + sb * Q::SB, sub, tg, gv0, gv1, gm0, gm1, gsw);
                        Q::unit(buf + (8 + g) * CH + sb * Q::SB, sub, tg, uv0, uv1, um0, um1, usw);
                    }
#pragma unroll
                    for (int G = 0; G < NG; ++G) {
                        const int b0 = (del & 2) ? 0x01010101 + tg + G : bv[G] ? get_int_b4(yb[G][u].qs, 8 * sub + tg) : 0;
                        const int b1 = (del & 2) ? 0x01010101 + sub + G : bv[G] ? get_int_b4(yb[G][u].qs, 8 * sub + 4 + tg) : 0;
                        int       c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                        if (del & 4) {
                            c[0] = gv0 ^ b0; c[1] = gv0 + b0; c[2] = uv0 ^ b0; c[3] = uv0 + b0;
                            e[0] = gv1 ^ b1; e[1] = gv1 + b1; e[2] = uv1 ^ b1; e[3] = uv1 + b1;
                        } else {
                            tc_mma(c, gv0, uv0, b0);
                            tc_mma(e, gv1, uv1, b1);
                        }
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            ig[G][q] += c[q] * gm0 + e[q] * gm1;
                            iu[G][q] += c[2 + q] * um0 + e[2 + q] * um1;
                        }
                    }
                }
#pragma unroll
                for (int G = 0; G < NG; ++G)
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        if (del & 8) { ag[G][q] += (float) ig[G][q]; au[G][q] += (float) iu[G][q]; continue; }
                        const float d8 = ov[G][q] ? yo[G][q][u].d : 0.0f;
                        ag[G][q] = fmaf(gsw * d8, (float) ig[G][q], ag[G][q]);
                        au[G][q] = fmaf(usw * d8, (float) iu[G][q], au[G][q]);
                    }
            }
            __syncwarp();
        }
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
            for (int q = 0; q < 2; ++q)
                if (ov[G][q]) act[(long long) oc[G][q] * F + r0 + g] = silu(ag[G][q]) * au[G][q];
    }
}

template <ggml_type type, int NG>
static void launch_gu_q8k(cudaStream_t st, const void * up, const void * gate, const block_q8_256 * y, const int32_t * recs,
                          int rec_stride, int count, float * act, int K, int F, long long rb, long long eb, int k) {
    const size_t sm = (size_t) kTcWarps * kTcStages * 16 * kTcKC * TQ<type>::SB;
    const dim3   grid((unsigned) ((F + kTcGuRows * kTcWarps - 1) / (kTcGuRows * kTcWarps)), (unsigned) count);
    static const int del = getenv("AI_Q4X_GU_DEL") ? atoi(getenv("AI_Q4X_GU_DEL")) : 0;
    gu_q8k_k<type, NG><<<grid, 32 * kTcWarps, sm, st>>>((const uint8_t *) up, (const uint8_t *) gate, y, recs, rec_stride,
                                                        count, act, K, F, rb, eb, k, del);
}

// AI_Q4X_PF_GU=4: split per-256 activations: int8 qs [T][K] then f32 d [T][K / 256] (16-byte aligned rows, so the
// block can stage them with 16-byte async copies)
__global__ void quantize_q8_256s_k(const float * __restrict__ x, int64_t x_stride, int8_t * __restrict__ qs,
                                   float * __restrict__ dq, int K, int nsb) {
    const int sbk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (sbk >= nsb) return;
    const float * xr = x + (size_t) row * x_stride + (size_t) sbk * QK_K + 8 * lane;
    float v[8], am = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[i] = xr[i]; am = fmaxf(am, fabsf(v[i])); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float d = am / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
    int8_t q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = (int8_t) __float2int_rn(v[i] * id);
    *(int2 *) (qs + (size_t) row * K + (size_t) sbk * QK_K + 8 * lane) = *(const int2 *) q;
    if (lane == 0) dq[(size_t) row * nsb + sbk] = d;
}

// gu_q8k_k with the activation tile staged once per block in shared memory. The warps compute different rows of the
// same columns, so in gu_q8k_k each loads the same activations itself from 8 scattered token rows per column group.
// Two stages (weights per warp + the block's activations), 16-byte async copies for the activations, rows padded to
// 528 bytes so the 8 columns a warp reads at once fall in distinct banks. Same per-(row, column) arithmetic as gu_q8k_k.
constexpr int kQsStages = 2, kQsAStride = kTcKC * QK_K + 16;
template <ggml_type type, int NG, int W>
__global__ void __launch_bounds__(32 * W) gu_q8ks_k(const uint8_t * __restrict__ up, const uint8_t * __restrict__ gate,
                                                           const int8_t * __restrict__ xqs, const float * __restrict__ xd,
                                                           const int32_t * __restrict__ recs, int rec_stride, int count,
                                                           float * __restrict__ act, int K, int F, long long row_bytes,
                                                           long long exp_bytes, int k, int del) {
    using Q = TQ<type>;
    constexpr int CH = kTcKC * Q::SB, W4 = CH / 4, STB = 16 * CH, NC = 8 * NG;
    constexpr int ATB = NC * kQsAStride, SCB = NC * kTcKC;   // activation bytes / scale floats per stage
    extern __shared__ __align__(16) uint8_t tc_sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3, tid = threadIdx.x;
    if ((int) blockIdx.y >= count) return;
    const int  r0     = (blockIdx.x * W + warp) * kTcGuRows;
    const bool active = r0 < F;
    const int32_t * o  = recs + (size_t) blockIdx.y * rec_stride;
    const int       ch = o[0], ncols = o[1], nsb = K / QK_K, nst = K / (QK_K * kTcKC);
    uint8_t *       wsm = tc_sm + warp * (kQsStages * STB);
    uint8_t *       asm_ = tc_sm + W * kQsStages * STB;
    float *         ssm = (float *) (asm_ + kQsStages * ATB);
    const uint8_t * gb = gate + (size_t) ch * exp_bytes + (size_t) (active ? r0 : 0) * row_bytes;
    const uint8_t * ub = up + (size_t) ch * exp_bytes + (size_t) (active ? r0 : 0) * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        auto load = [&](int st, int buf) {
            if (active && !(del & 16)) {
                uint8_t * dst = wsm + buf * STB;
                for (int i = lane; i < 16 * W4; i += 32) {
                    const int r = i / W4, w = i - r * W4;
                    tc_cp4(dst + r * CH + 4 * w, (r < 8 ? gb + r * row_bytes : ub + (r - 8) * row_bytes) + (size_t) st * CH + 4 * w);
                }
            }
            constexpr int CPC = kTcKC * QK_K / 16;   // 16-byte chunks per column per stage
            for (int i = tid; i < NC * CPC && !(del & 2); i += 32 * W) {
                const int c = i / CPC, part = i - c * CPC;
                if (c0 + c >= ncols) continue;
                const int row = o[2 + c0 + c] / k;
                tc_cp16(asm_ + buf * ATB + c * kQsAStride + 16 * part, xqs + (size_t) row * K + (size_t) st * (kTcKC * QK_K) + 16 * part);
            }
            for (int i = tid; i < SCB; i += 32 * W) {
                const int c = i / kTcKC, s = i - c * kTcKC;
                if (c0 + c >= ncols) continue;
                tc_cp4(ssm + buf * SCB + i, xd + (size_t) (o[2 + c0 + c] / k) * nsb + st * kTcKC + s);
            }
            tc_commit();
        };
        bool  ov[NG][2];
        int   oc[NG][2];
        float ag[NG][2], au[NG][2];
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = c0 + 8 * G + 2 * tg + q;
                ov[G][q] = co < ncols;
                oc[G][q] = ov[G][q] ? o[2 + co] : 0;
                ag[G][q] = au[G][q] = 0.0f;
            }
        load(0, 0);
        for (int st = 0; st < nst; ++st) {
            if (st + 1 < nst) { load(st + 1, (st + 1) & 1); tc_wait<1>(); }
            else { tc_wait<0>(); }
            __syncthreads();
            const uint8_t * wb = wsm + (st & 1) * STB;
            const uint8_t * ab = asm_ + (st & 1) * ATB;
            const float *   sb_ = ssm + (st & 1) * SCB;
#pragma unroll
            for (int sb = 0; sb < kTcKC; ++sb) {
                int ig[NG][2], iu[NG][2];
#pragma unroll
                for (int G = 0; G < NG; ++G) ig[G][0] = ig[G][1] = iu[G][0] = iu[G][1] = 0;
                float gsw = 0.0f, usw = 0.0f;
#pragma unroll 2
                for (int sub = 0; sub < 8; ++sub) {
                    int gv0 = 0, gv1 = 0, gm0 = 0, gm1 = 0, uv0 = 0, uv1 = 0, um0 = 0, um1 = 0;
                    if (del & 1) { gv0 = lane; gv1 = sub; uv0 = tg; uv1 = lane ^ sub; gm0 = um0 = 1; gm1 = um1 = 2; gsw = usw = 1.0f; }
                    else if (active) {
                        Q::unit(wb + g * CH + sb * Q::SB, sub, tg, gv0, gv1, gm0, gm1, gsw);
                        Q::unit(wb + (8 + g) * CH + sb * Q::SB, sub, tg, uv0, uv1, um0, um1, usw);
                    }
#pragma unroll
                    for (int G = 0; G < NG; ++G) {
                        const uint8_t * ar = ab + (8 * G + g) * kQsAStride + sb * QK_K + 32 * sub + 4 * tg;
                        const int       b0 = *(const int *) ar, b1 = *(const int *) (ar + 16);
                        int             c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                        if (del & 4) { c[0] = b0 ^ gv0; c[1] = uv0; c[2] = b1; c[3] = gv1 ^ uv1; e[0] = b1 ^ uv1; }
                        else { tc_mma(c, gv0, uv0, b0); tc_mma(e, gv1, uv1, b1); }
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            ig[G][q] += c[q] * gm0 + e[q] * gm1;
                            iu[G][q] += c[2 + q] * um0 + e[2 + q] * um1;
                        }
                    }
                }
#pragma unroll
                for (int G = 0; G < NG; ++G)
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        if (del & 8) { ag[G][q] += (float) ig[G][q]; au[G][q] += (float) iu[G][q]; continue; }
                        const float d8 = ov[G][q] ? sb_[(8 * G + 2 * tg + q) * kTcKC + sb] : 0.0f;
                        ag[G][q] = fmaf(gsw * d8, (float) ig[G][q], ag[G][q]);
                        au[G][q] = fmaf(usw * d8, (float) iu[G][q], au[G][q]);
                    }
            }
            __syncthreads();   // every warp is done with this buffer before the next load overwrites it
        }
        if (active) {
#pragma unroll
            for (int G = 0; G < NG; ++G)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    if (ov[G][q]) act[(long long) oc[G][q] * F + r0 + g] = silu(ag[G][q]) * au[G][q];
        }
    }
}

template <ggml_type type, int NG, int W>
static void launch_gu_q8ks(cudaStream_t st, const void * up, const void * gate, const int8_t * xqs, const float * xd,
                           const int32_t * recs, int rec_stride, int count, float * act, int K, int F, long long rb,
                           long long eb, int k) {
    constexpr int CH = kTcKC * TQ<type>::SB;
    const size_t  sm = (size_t) W * kQsStages * 16 * CH + (size_t) kQsStages * (8 * NG) * kQsAStride +
                      (size_t) kQsStages * (8 * NG) * kTcKC * sizeof(float);
    static bool attr = false;
    if (!attr) {
        attr = true;
        cudaFuncSetAttribute(gu_q8ks_k<type, NG, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    }
    const dim3 grid((unsigned) ((F + kTcGuRows * W - 1) / (kTcGuRows * W)), (unsigned) count);
    // AI_Q4X_GU_DEL (benchmarking only; the output is wrong): bound by deletion, a bit mask of the phases to skip --
    // 1 weight decode, 2 activation staging, 4 mma, 8 epilogue, 16 weight staging. The weight decode is the bound.
    // Not faster, so not used: the codebook in shared memory, 48 / 64 columns per pass with 6 / 4 warps, weights
    // decoded straight from global memory
    static const int del = getenv("AI_Q4X_GU_DEL") ? atoi(getenv("AI_Q4X_GU_DEL")) : 0;
    gu_q8ks_k<type, NG, W><<<grid, 32 * W, sm, st>>>((const uint8_t *) up, (const uint8_t *) gate, xqs, xd, recs,
                                                         rec_stride, count, act, K, F, rb, eb, k, del);
}

// Prefill gate/up with the structure of llama.cpp's MMQ. gu_q8ks_k is bound by its weight decode
// and decodes each warp's own rows again for every 32 columns. Here a block owns 64 gate rows + the same 64 up rows of
// one expert and up to 128 of its columns; per superblock of K all 256 threads decode the 128 x 256 weight tile once
// into shared memory (int8 values, the per-16 integer multipliers, the row scale) from raw rows prefetched under the
// previous superblock's mma, while the activations for the 128 columns land by 16-byte async copies; then warp w
// multiplies its 16 rows (8 gate + 8 up, rows 8w..8w+7) against all 128 columns. Per (row, column) the arithmetic is
// gu_q8ks_k's in the same order, so the results are bit-identical.
constexpr int kMqStride = QK_K + 16;   // smem bytes per decoded weight row / activation column
template <ggml_type type, int kMqRows, int kMqCols>
__global__ void __launch_bounds__(4 * kMqRows) gu_mmq_k(const uint8_t * __restrict__ up, const uint8_t * __restrict__ gate,
                                                const int8_t * __restrict__ xqs, const float * __restrict__ xd,
                                                const int32_t * __restrict__ recs, int rec_stride, int count,
                                                float * __restrict__ act, int K, int F, long long row_bytes,
                                                long long exp_bytes, int k, int del) {
    using Q = TQ<type>;
    constexpr int NA = 2 * kMqRows, NG = kMqCols / 8, NT = 4 * kMqRows;   // A rows (tile w: rows 16w..16w+7 gate, 16w+8.. up), column groups
    extern __shared__ __align__(16) uint8_t mq_sm[];
    int8_t * wq = (int8_t *) mq_sm;                                   // [NA][kMqStride]
    int8_t * xs = wq + NA * kMqStride;                                // [kMqCols][kMqStride]
    int *    wm = (int *) (xs + kMqCols * kMqStride);                 // [NA][8]: m0 | m1 << 16 per 32-block
    float *  wd = (float *) (wm + NA * 8);                            // [NA]
    float *  sd = wd + NA;                                            // [kMqCols]
    // raw weight rows of the next superblock, prefetched under this one's mma: 4-byte copies from the 4-aligned byte
    // below the superblock's start (blocks are 66..110 bytes, so a row's superblocks are only 2-aligned)
    constexpr int RAWS = (Q::SB + 2 + 3) / 4 * 4;
    uint8_t * raw = (uint8_t *) (sd + kMqCols);                       // [NA][RAWS]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3, tid = threadIdx.x;
    if ((int) blockIdx.y >= count) return;
    const int32_t * o = recs + (size_t) blockIdx.y * rec_stride;
    const int ch = o[0], ncols = o[1], nsb = K / QK_K, rb = blockIdx.x * kMqRows;
    const uint8_t * gbase = gate + (size_t) ch * exp_bytes, * ubase = up + (size_t) ch * exp_bytes;
    const int  myr = rb + 8 * warp + g;   // this thread's output row (gate row myr pairs with up row myr)
    const bool rok = myr < F;
    for (int c0 = 0; c0 < ncols; c0 += kMqCols) {
        const int nc = min(kMqCols, ncols - c0);
        int   oc[NG][2];
        float ag[NG][2], au[NG][2];
#pragma unroll
        for (int G = 0; G < NG; ++G)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int co = 8 * G + 2 * tg + q;
                oc[G][q] = co < nc ? o[2 + c0 + co] : -1;
                ag[G][q] = au[G][q] = 0.0f;
            }
        auto raw_issue = [&](int sb) {   // group R(sb): the raw rows of superblock sb
            const int off = (sb * Q::SB) & 3, nw = (off + Q::SB + 3) / 4;
            const size_t base = (size_t) sb * Q::SB - off;
            for (int i = tid; i < NA * (RAWS / 4); i += NT) {
                const int ar = i / (RAWS / 4), w = i - ar * (RAWS / 4);
                const int tile = ar >> 4, w16 = ar & 15, r = rb + 8 * tile + (w16 & 7);
                if (r < F && w < nw && !(del & 8)) tc_cp4(raw + ar * RAWS + 4 * w, (w16 < 8 ? gbase : ubase) + (size_t) r * row_bytes + base + 4 * w);
            }
            tc_commit();
        };
        auto act_issue = [&](int sb) {   // group A(sb): the pass's activation columns of superblock sb, + their scales
            for (int i = tid; i < kMqCols * (QK_K / 16); i += NT) {
                const int c = i / (QK_K / 16), part = i - c * (QK_K / 16);
                if (c < nc && !(del & 2)) tc_cp16(xs + c * kMqStride + 16 * part, xqs + (size_t) (o[2 + c0 + c] / k) * K + (size_t) sb * QK_K + 16 * part);
            }
            tc_commit();
            for (int i = tid; i < kMqCols; i += NT) sd[i] = i < nc ? xd[(size_t) (o[2 + c0 + i] / k) * nsb + sb] : 0.0f;
        };
        raw_issue(0);
        act_issue(0);
        for (int sb = 0; sb < nsb; ++sb) {
            tc_wait<1>();   // R(sb) landed (A(sb), the newest group, may still be in flight)
            __syncthreads();
            const int off = (sb * Q::SB) & 3;
            // a thread decodes a whole 32-block, its four quarters in a row: the block's header loads (scale, high bits,
            // signs, d) are shared instead of repeated by four threads, and the 32 values go out as two 16-byte stores
            for (int u = tid; u < NA * 8; u += NT) {   // (A row, 32-block)
                const int ar = u >> 3, sub = u & 7;
                const int tile = ar >> 4, w16 = ar & 15, r = rb + 8 * tile + (w16 & 7);
                int   v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 }, m0 = 0, m1 = 0;
                float sw = 0.0f;
                if (del & 1) {
#pragma unroll
                    for (int q = 0; q < 8; ++q) v[q] = lane ^ u ^ q;
                    m0 = m1 = 1; sw = 1.0f;
                } else if (r < F) {
                    const uint8_t * rp = raw + ar * RAWS + off;
#pragma unroll
                    for (int t4 = 0; t4 < 4; ++t4) Q::unit(rp, sub, t4, v[t4], v[4 + t4], m0, m1, sw);
                }
                int4 * dst = (int4 *) (wq + ar * kMqStride + 32 * sub);   // 16-byte aligned: kMqStride % 16 == 0
                dst[0] = make_int4(v[0], v[1], v[2], v[3]);
                dst[1] = make_int4(v[4], v[5], v[6], v[7]);
                wm[ar * 8 + sub] = (m0 & 0xFFFF) | (m1 << 16);
                if (sub == 0) wd[ar] = sw;
            }
            __syncthreads();   // the raw buffer is free
            if (sb + 1 < nsb) raw_issue(sb + 1); else tc_commit();
            tc_wait<1>();   // A(sb) landed (R(sb + 1) may still be in flight)
            __syncthreads();
            const int8_t * wa = wq + (16 * warp + g) * kMqStride, * wb = wq + (16 * warp + 8 + g) * kMqStride;
            int ig[NG][2], iu[NG][2];
#pragma unroll
            for (int G = 0; G < NG; ++G) ig[G][0] = ig[G][1] = iu[G][0] = iu[G][1] = 0;
#pragma unroll 2
            for (int sub = 0; sub < 8; ++sub) {
                const int gv0 = *(const int *) (wa + 32 * sub + 4 * tg), gv1 = *(const int *) (wa + 32 * sub + 16 + 4 * tg);
                const int uv0 = *(const int *) (wb + 32 * sub + 4 * tg), uv1 = *(const int *) (wb + 32 * sub + 16 + 4 * tg);
                const int gmm = wm[(16 * warp + g) * 8 + sub], umm = wm[(16 * warp + 8 + g) * 8 + sub];
                const int gm0 = (int) (short) (gmm & 0xFFFF), gm1 = gmm >> 16, um0 = (int) (short) (umm & 0xFFFF), um1 = umm >> 16;
#pragma unroll
                for (int G = 0; G < NG; ++G) {
                    const int8_t * ap = xs + (8 * G + g) * kMqStride + 32 * sub + 4 * tg;
                    const int      b0 = *(const int *) ap, b1 = *(const int *) (ap + 16);
                    int            c[4] = { 0, 0, 0, 0 }, e[4] = { 0, 0, 0, 0 };
                    if (del & 4) { c[0] = b0 ^ gv0; c[1] = uv0; c[2] = b1; c[3] = gv1 ^ uv1; e[0] = b1 ^ uv1; }
                    else { tc_mma(c, gv0, uv0, b0); tc_mma(e, gv1, uv1, b1); }
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        ig[G][q] += c[q] * gm0 + e[q] * gm1;
                        iu[G][q] += c[2 + q] * um0 + e[2 + q] * um1;
                    }
                }
            }
            const float gsw = wd[16 * warp + g], usw = wd[16 * warp + 8 + g];
#pragma unroll
            for (int G = 0; G < NG; ++G)
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    const float d8 = oc[G][q] >= 0 ? sd[8 * G + 2 * tg + q] : 0.0f;
                    ag[G][q] = fmaf(gsw * d8, (float) ig[G][q], ag[G][q]);
                    au[G][q] = fmaf(usw * d8, (float) iu[G][q], au[G][q]);
                }
            __syncthreads();   // every warp is done with the tile and the activations
            if (sb + 1 < nsb) act_issue(sb + 1); else tc_commit();
        }
        if (rok) {
#pragma unroll
            for (int G = 0; G < NG; ++G)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                    if (oc[G][q] >= 0) act[(long long) oc[G][q] * F + myr] = silu(ag[G][q]) * au[G][q];
        }
    }
}

template <ggml_type type, int kMqRows, int kMqCols>
static void launch_gu_mmq(cudaStream_t st, const void * up, const void * gate, const int8_t * xqs, const float * xd,
                          const int32_t * recs, int rec_stride, int count, float * act, int K, int F, long long rb,
                          long long eb, int k) {
    constexpr size_t sm = (size_t) (2 * kMqRows + kMqCols) * kMqStride + (size_t) 2 * kMqRows * 8 * sizeof(int) +
                          (size_t) (2 * kMqRows + kMqCols) * sizeof(float) +
                          (size_t) 2 * kMqRows * ((TQ<type>::SB + 2 + 3) / 4 * 4);
    static bool attr = false;
    if (!attr) {
        attr = true;
        if (cudaFuncSetAttribute(gu_mmq_k<type, kMqRows, kMqCols>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm) != cudaSuccess)
            fprintf(stderr, "[q4x] gu_mmq_k<%d,%d,%d>: %zu bytes of shared memory refused%c", (int) type, kMqRows, kMqCols, sm, 10);
    }
    const dim3 grid((unsigned) ((F + kMqRows - 1) / kMqRows), (unsigned) count);
    // AI_Q4X_GU_DEL (default 0, benchmarking only, wrong output): bitmask of stages to skip to find what bounds the
    // kernel: 1 weight decode, 2 activation staging, 4 mma, 8 raw weight prefetch
    static const int del = getenv("AI_Q4X_GU_DEL") ? atoi(getenv("AI_Q4X_GU_DEL")) : 0;
    gu_mmq_k<type, kMqRows, kMqCols><<<grid, 4 * kMqRows, sm, st>>>((const uint8_t *) up, (const uint8_t *) gate, xqs, xd, recs, rec_stride, count,
                                          act, K, F, rb, eb, k, del);
}

// Prefill with big MoE batches: the tensor-core wide path in three steps, so the caller can run gate/up over a whole
// super-chunk (each owner's tiles decoded once for all its columns) and the down projection in row slices (its N x k x T
// output would not fit for the whole chunk). Scratch layout as q4x_experts_wide: xq | act | aq for T rows. Columns are
// global (t * k + j); a bank's gate/up writes only its owners' act columns, so both banks share one act region.
static bool wide_tc_usable(int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg, int ty_down, const void * down,
                           int64_t rbd, int64_t ebd, int64_t K, int64_t F, int64_t N, int64_t T) {
    if (!gu_type(ty_gu) || (ty_down != GGML_TYPE_IQ4_NL && ty_down != GGML_TYPE_Q2_0) || K % QK_K || F % QK2_0 ||
        F % QK4_NL || N % 4 || F % 4 || T < 9 || T > 65536) return false;
    const size_t bg = bsize(ty_gu), bd = bsize(ty_down);
    if (rbg % bg || ebg % bg || rbd % bd || ebd % bd) return false;
    return wide_mode() == 1 && tc_ok(up, gate, rbg, ebg, down, rbd, ebd, K, F, N);
}

bool q4x_experts_wide_split_ok(int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg, int ty_down,
                               const void * down, int64_t rbd, int64_t ebd, int64_t K, int64_t F, int64_t N, int64_t T) {
    return wide_tc_usable(ty_gu, up, gate, rbg, ebg, ty_down, down, rbd, ebd, K, F, N, T);
}

bool q4x_experts_wide_gu(void * stream, int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg,
                         int ty_down, const void * down, int64_t rbd, int64_t ebd, const float * x, int64_t x_stride,
                         bool quantize_x, const int32_t * plan, int n_own, int64_t K, int64_t F, int64_t N, int k, int64_t T,
                         void * scratch) {
    if (!wide_tc_usable(ty_gu, up, gate, rbg, ebg, ty_down, down, rbd, ebd, K, F, N, T)) return false;
    cudaStream_t st   = (cudaStream_t) stream;
    char *       s    = (char *) scratch;
    const size_t xq_b = (size_t) (K / QK8_1) * sizeof(block_q8_1) * T, act_b = (size_t) F * k * T * sizeof(float);
    auto *       xq   = (block_q8_1 *) s;
    auto *       act  = (float *) (s + (xq_b + 255) / 256 * 256);
    auto *       aq   = (block_q8_1 *) ((char *) act + (act_b + 255) / 256 * 256);
    // AI_Q4X_PF_GU (default 4): 4 = per-256 activations staged per 64-row block (gu_q8ks_k, 8 warps); 3 = per-256
    // activations loaded per warp; 0 = the q8_1 kernel (gu_tc_k). 3 and 4 give identical tokens. Per-256 activations
    // change the numerics against 0; paired NLL shows no significant difference.
    static const int pf_gu = getenv("AI_Q4X_PF_GU") ? atoi(getenv("AI_Q4X_PF_GU")) : 4;
    if (pf_gu == 4) {   // split per-256 activations staged per block (the q8_1 region holds T*K + 4*T*K/256 bytes)
        auto *    xqs = (int8_t *) xq;
        const int nsb = (int) (K / QK_K);
        auto *    xd  = (float *) (xqs + (((size_t) T * K + 255) / 256 * 256));
        if (quantize_x)
            quantize_q8_256s_k<<<dim3((unsigned) ((nsb + 3) / 4), (unsigned) T), 128, 0, st>>>(x, x_stride, xqs, xd, (int) K, nsb);
        if (n_own <= 0) return true;
        const int Ki = (int) K, Fi = (int) F;
        // AI_Q4X_PF_GU_WARPS = 8 (default) | 4: rows per block = 8 x warps. Every block of an expert reloads its
        // columns' activations, and those loads are a large share of gate/up, so taller blocks cut that traffic.
        static const int w8 = getenv("AI_Q4X_PF_GU_WARPS") ? atoi(getenv("AI_Q4X_PF_GU_WARPS")) : 8;
        // AI_Q4X_PF_GU_MMQ (default 1): gu_mmq_k, which decodes the weight tile once per 128 columns from raw rows
        // prefetched under the previous superblock's mma; 0 = gu_q8ks_k. Both give the same bits. The 64 x 128 tile and
        // the raw-row prefetch were chosen over smaller tiles and decoding straight from global memory.
        static const int mmq = getenv("AI_Q4X_PF_GU_MMQ") ? atoi(getenv("AI_Q4X_PF_GU_MMQ")) : 1;
        if (mmq != 0) {
            switch (ty_gu) {
                case GGML_TYPE_IQ2_XXS: launch_gu_mmq<GGML_TYPE_IQ2_XXS, 64, 128>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                case GGML_TYPE_IQ2_XS:  launch_gu_mmq<GGML_TYPE_IQ2_XS, 64, 128>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                case GGML_TYPE_IQ2_S:   launch_gu_mmq<GGML_TYPE_IQ2_S, 64, 128>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                case GGML_TYPE_IQ3_XXS: launch_gu_mmq<GGML_TYPE_IQ3_XXS, 64, 128>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                default:                launch_gu_mmq<GGML_TYPE_IQ3_S, 64, 128>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            }
            return true;
        }
        if (w8 == 8) {
            switch (ty_gu) {
                case GGML_TYPE_IQ2_XXS: launch_gu_q8ks<GGML_TYPE_IQ2_XXS, 4, 8>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                case GGML_TYPE_IQ2_XS:  launch_gu_q8ks<GGML_TYPE_IQ2_XS, 4, 8>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                case GGML_TYPE_IQ2_S:   launch_gu_q8ks<GGML_TYPE_IQ2_S, 4, 8>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                case GGML_TYPE_IQ3_XXS: launch_gu_q8ks<GGML_TYPE_IQ3_XXS, 4, 8>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
                default:                launch_gu_q8ks<GGML_TYPE_IQ3_S, 4, 8>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            }
            return true;
        }
        switch (ty_gu) {
            case GGML_TYPE_IQ2_XXS: launch_gu_q8ks<GGML_TYPE_IQ2_XXS, 4, 4>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_XS:  launch_gu_q8ks<GGML_TYPE_IQ2_XS, 4, 4>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_S:   launch_gu_q8ks<GGML_TYPE_IQ2_S, 4, 4>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ3_XXS: launch_gu_q8ks<GGML_TYPE_IQ3_XXS, 4, 4>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            default:                launch_gu_q8ks<GGML_TYPE_IQ3_S, 4, 4>(st, up, gate, xqs, xd, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
        }
        return true;
    }
    if (pf_gu == 3) {   // activations per 256 (the q8_1 region holds them: 260 <= 288 bytes per superblock)
        auto * y8 = (block_q8_256 *) xq;
        const int nsb = (int) (K / QK_K);
        if (quantize_x) quantize_q8_256_k<<<dim3((unsigned) ((nsb + 3) / 4), (unsigned) T), 128, 0, st>>>(x, x_stride, y8, nsb);
        if (n_own <= 0) return true;
        const int Ki = (int) K, Fi = (int) F;
        switch (ty_gu) {
            case GGML_TYPE_IQ2_XXS: launch_gu_q8k<GGML_TYPE_IQ2_XXS, 4>(st, up, gate, y8, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_XS:  launch_gu_q8k<GGML_TYPE_IQ2_XS, 4>(st, up, gate, y8, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_S:   launch_gu_q8k<GGML_TYPE_IQ2_S, 4>(st, up, gate, y8, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ3_XXS: launch_gu_q8k<GGML_TYPE_IQ3_XXS, 4>(st, up, gate, y8, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            default:                launch_gu_q8k<GGML_TYPE_IQ3_S, 4>(st, up, gate, y8, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
        }
        return true;
    }
    if (quantize_x) quantize_row_q8_1_cuda(x, nullptr, xq, GGML_TYPE_Q8_0, K, x_stride, x_stride * T, x_stride * T, K, T, 1, 1, st);
    if (n_own <= 0) return true;
    // AI_Q4X_PF_GU = 0 | 1 | 2 with q8_1 activations: 1 = gu_pf_k (16 column groups per weight pass), 2 = gu_pfm_k (4
    // row tiles per warp). Both are bit-identical to 0 and not faster: neither weight decode nor activation loads bound
    // gate/up, the per-32-block float epilogue does, which is what the per-256 kernels (3, 4) address. ggml MMQ on these
    // types produced NaN tokens and was not faster either.
    if (pf_gu == 2) {   // MT = 4 row tiles per warp, NG = 4 column groups
        const int Ki = (int) K, Fi = (int) F;
        switch (ty_gu) {
            case GGML_TYPE_IQ2_XXS: launch_gu_pfm<GGML_TYPE_IQ2_XXS, 4, 4>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_XS:  launch_gu_pfm<GGML_TYPE_IQ2_XS, 4, 4>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_S:   launch_gu_pfm<GGML_TYPE_IQ2_S, 4, 4>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ3_XXS: launch_gu_pfm<GGML_TYPE_IQ3_XXS, 4, 4>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            default:                launch_gu_pfm<GGML_TYPE_IQ3_S, 4, 4>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
        }
        return true;
    }
    if (pf_gu == 1) {
        const int Ki = (int) K, Fi = (int) F;
        switch (ty_gu) {
            case GGML_TYPE_IQ2_XXS: launch_gu_pf<GGML_TYPE_IQ2_XXS, 16>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_XS:  launch_gu_pf<GGML_TYPE_IQ2_XS, 16>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ2_S:   launch_gu_pf<GGML_TYPE_IQ2_S, 16>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            case GGML_TYPE_IQ3_XXS: launch_gu_pf<GGML_TYPE_IQ3_XXS, 16>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
            default:                launch_gu_pf<GGML_TYPE_IQ3_S, 16>(st, up, gate, xq, plan, kWideRec, n_own, act, Ki, Fi, rbg, ebg, k); break;
        }
        return true;
    }
    tc_dispatch<4>(st, 0, ty_gu, up, gate, rbg, ebg, ty_down, down, rbd, ebd, xq, aq, plan, kWideRec, nullptr, n_own, n_own,
                   act, nullptr, (int) K, (int) F, (int) N, k);
    return true;
}

void q4x_experts_wide_qact(void * stream, int64_t K, int64_t F, int k, int64_t T, void * scratch) {
    char *       s    = (char *) scratch;
    const size_t xq_b = (size_t) (K / QK8_1) * sizeof(block_q8_1) * T, act_b = (size_t) F * k * T * sizeof(float);
    auto *       act  = (float *) (s + (xq_b + 255) / 256 * 256);
    auto *       aq   = (block_q8_1 *) ((char *) act + (act_b + 255) / 256 * 256);
    const int64_t n   = (int64_t) k * T;
    quantize_row_q8_1_cuda(act, nullptr, aq, GGML_TYPE_Q8_0, F, F, F * n, F * n, F, n, 1, 1, (cudaStream_t) stream);
}

bool q4x_experts_wide_dn(void * stream, int ty_gu, const void * up, const void * gate, int64_t rbg, int64_t ebg,
                         int ty_down, const void * down, int64_t rbd, int64_t ebd, const int32_t * plan, int n_own,
                         int64_t K, int64_t F, int64_t N, int k, int64_t T, void * scratch, float * out, int64_t col_lo) {
    if (!wide_tc_usable(ty_gu, up, gate, rbg, ebg, ty_down, down, rbd, ebd, K, F, N, T)) return false;
    if (n_own <= 0) return true;
    char *       s    = (char *) scratch;
    const size_t xq_b = (size_t) (K / QK8_1) * sizeof(block_q8_1) * T, act_b = (size_t) F * k * T * sizeof(float);
    auto *       xq   = (block_q8_1 *) s;
    auto *       act  = (float *) (s + (xq_b + 255) / 256 * 256);
    auto *       aq   = (block_q8_1 *) ((char *) act + (act_b + 255) / 256 * 256);
    // out holds columns [col_lo, col_lo + ...): the kernel writes out[col * N], so it gets the base shifted by col_lo
    float *      ob   = out - col_lo * N;
    // AI_Q4X_PF_DN = 1 (default) | 0: dn_s_k, block-staged activations over 128-row blocks (bit-identical to 0)
    static const int pf_dn = getenv("AI_Q4X_PF_DN") ? atoi(getenv("AI_Q4X_PF_DN")) : 1;
    if (pf_dn == 1 && (rbd * kTcDnRows) % 16 == 0) {
        cudaStream_t st = (cudaStream_t) stream;
        if (ty_down == GGML_TYPE_IQ4_NL) launch_dn_s<GGML_TYPE_IQ4_NL, 4, 8>(st, down, aq, plan, kWideRec, n_own, ob, (int) F, (int) N, rbd, ebd);
        else                             launch_dn_s<GGML_TYPE_Q2_0, 4, 8>(st, down, aq, plan, kWideRec, n_own, ob, (int) F, (int) N, rbd, ebd);
        return true;
    }
    tc_dispatch<4>((cudaStream_t) stream, 1, ty_gu, up, gate, rbg, ebg, ty_down, down, rbd, ebd, xq, aq, plan, kWideRec,
                   nullptr, n_own, n_own, act, ob, (int) K, (int) F, (int) N, k);
    return true;
}

void q4x_experts_compare(void * stream, const float * mine, const float * ref, const int32_t * ids, int64_t ids_stride,
                         int k, int filler0, int64_t N, int64_t T) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[4] = { 0, 0, 0, 0 };
            unsigned int       mb = 0;
            double             l2[2] = { 0, 0 };
            cudaMemcpyFromSymbol(h, g_cmp, sizeof(h));
            cudaMemcpyFromSymbol(&mb, g_cmp_maxabs, sizeof(mb));
            cudaMemcpyFromSymbol(l2, g_cmp_l2, sizeof(l2));
            float mx;
            memcpy(&mx, &mb, 4);
            fprintf(stderr, "[q4x-experts-compare] calls %llu | values %llu, differing %llu (> 1e-3 rel: %llu) | max |d| %.3g | rel-L2 %.3g\n",
                    h[0], h[1], h[2], h[3], (double) mx, l2[1] > 0 ? sqrt(l2[0] / l2[1]) : 0.0);
        });
    }
    compare_k<<<(unsigned) (k * T), 256, 0, (cudaStream_t) stream>>>(mine, ref, ids, (int) ids_stride, k, filler0, (long long) N);
}

}  // namespace ai::kern
