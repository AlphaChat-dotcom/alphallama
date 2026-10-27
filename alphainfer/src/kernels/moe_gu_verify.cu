// MTP-verify MoE gate/up + SwiGLU in one launch, with expert dedup and activation reuse (AI_MOE_GU).
//
// At verify width T (2..8) ggml runs the routed experts' gate and up as two mul_mat_vec_q_moe launches (one
// warp per (slot, token), the whole K in that warp), then a separate unary_gated_op_kernel<op_silu>.
//
// The T tokens of a verify round typically route to far fewer distinct experts than T * n_expert_used slots.
// Reading a duplicate expert twice costs little (the second read is an L2 hit), but every (slot, token) warp
// also re-runs the q4_K unpack (scales/mins reconstruction, nibble extraction) for the same weights. Here the
// lowest token routed to an expert owns it: its warp unpacks each weight block once and dots it against every
// token that shares the expert.
//
// Bit-exact by construction against that chain:
//   * per (row, token) the dot is the reference's: lane l covers kbx = l/(qi/vdr) + k*bpi with
//     kqs = vdr*(l%(qi/vdr)), accumulates in kbx order starting from 0.0f, then warp_reduce_sum;
//   * x is unpacked by the same code as vec_dot_q4_K_q8_1 (ggml's ALPHA_MMVQ_XREUSE split, which is
//     bit-identical), and the y-side and vec_dot_*_impl_vmmq are ggml's;
//   * the GLU is op_silu(gate) * up with op_silu = x / (1 + expf(-x));
//   * this file is compiled -use_fast_math, as ggml-cuda is (CMakeLists.txt).
// AI_MOE_GU=2 runs this kernel and the ggml chain, generates from the chain, and counts bitwise differences
// per call.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int GU_NS   = 8;   // routed slots (n_expert_used)
constexpr int GU_MAXT = 8;   // GU_NS * T <= 64: two warp ballots (lanes l and l + 32), so an 8-row verify
                             // (MTP7 / DFlash block 8) also avoids the 3-launch chain

template <ggml_type type> struct XU;

template <> struct XU<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K, qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
    int      v[2];
    uint16_t aux[2];
    half2    dm;
    int      bq8_offset, iq;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_q4_K * b = (const block_q4_K *) vbq + kbx;
        bq8_offset           = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        iq                   = (iqs / 2) % 4;
        const int * q4       = (const int *) (b->qs + 16 * bq8_offset + 4 * iq);
        v[0]                 = q4[0];
        v[1]                 = q4[4];
        const uint16_t * scales = (const uint16_t *) b->scales;
        const int        j      = bq8_offset / 2;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        dm = b->dm;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        int   u[2 * QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
            d8[i]                   = __low2float(bq8i->ds);
            const int * q8          = (const int *) bq8i->qs + iq;
            u[2 * i + 0]            = q8[0];
            u[2 * i + 1]            = q8[4];
        }
        const uint8_t * sc = (const uint8_t *) aux;
        return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, sc + 2, dm, d8);
    }
};

template <> struct XU<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K, qi = QI5_K, vdr = VDR_Q5_K_Q8_1_MMVQ;
    int      vl[2], vh[2];
    uint16_t aux[2];
    half2    dm;
    int      bq8_offset, iq;
    __device__ __forceinline__ void load(const void * __restrict__ vbq, int kbx, int iqs) {
        const block_q5_K * b = (const block_q5_K *) vbq + kbx;
        bq8_offset           = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        iq                   = (iqs / 2) % 4;
        const int * ql       = (const int *) (b->qs + 16 * bq8_offset + 4 * iq);
        const int * qh       = (const int *) (b->qh + 4 * iq);
        vl[0]                = ql[0];
        vl[1]                = ql[4];
        vh[0]                = qh[0] >> bq8_offset;
        vh[1]                = qh[4] >> bq8_offset;
        const uint16_t * scales = (const uint16_t *) b->scales;
        const int        j      = bq8_offset / 2;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        dm = b->dm;
    }
    __device__ __forceinline__ float dot(const block_q8_1 * __restrict__ bq8_1) const {
        int   u[2 * QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
            d8[i]                   = __low2float(bq8i->ds);
            const int * q8          = (const int *) bq8i->qs + iq;
            u[2 * i + 0]            = q8[0];
            u[2 * i + 1]            = q8[4];
        }
        const uint8_t * sc = (const uint8_t *) aux;
        return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, sc + 2, dm, d8);
    }
};

// ggml's op_silu, as built (-use_fast_math)
__device__ __forceinline__ float gu_silu(float x) {
    return x / (1.0f + expf(-x));
}

// grid (F / W, GU_NS * T), block (32, W): warp = one row of one (slot, token). The block exits unless its
// token is the lowest routed to its expert; the owner computes every token that shares the expert.
template <ggml_type type, int W, int MT>
__global__ void __launch_bounds__(32 * W) moe_gu_verify_k(const void * __restrict__ up, const void * __restrict__ gate,
                                                          const block_q8_1 * __restrict__ y, const int32_t * __restrict__ ids,
                                                          float * __restrict__ act, int ncols, int nrows, int stride_row,
                                                          int stride_ch, int ids_stride, int T) {
    using X = XU<type>;
    constexpr int bpi = X::vdr * 32 / X::qi;
    const int     lane = threadIdx.x;
    const int     slot = blockIdx.y % GU_NS, tok = blockIdx.y / GU_NS;
    const int     ch   = ids[slot + tok * ids_stride];
    const int     n    = GU_NS * T;
    const int     l1   = lane + 32;
    const int     id0  = lane < n ? ids[(lane % GU_NS) + (lane / GU_NS) * ids_stride] : -1;
    const int     id1  = l1 < n ? ids[(l1 % GU_NS) + (l1 / GU_NS) * ids_stride] : -1;
    const unsigned long long match = (unsigned long long) __ballot_sync(0xffffffffu, id0 == ch) |
                                     ((unsigned long long) __ballot_sync(0xffffffffu, id1 == ch) << 32);
    const unsigned long long lower = (1ull << (tok * GU_NS)) - 1ull;   // tok * GU_NS <= 56
    if (match & lower) {
        return;   // a lower token owns this expert
    }
    int      ms[MT], mt[MT], ne = 0;
    unsigned long long rest = match;
#pragma unroll
    for (int e = 0; e < MT; ++e) {
        if (rest) {
            const int b = __ffsll((long long) rest) - 1;
            rest &= rest - 1;
            ms[e] = b % GU_NS;
            mt[e] = b / GU_NS;
            ne    = e + 1;
        } else {
            ms[e] = slot;
            mt[e] = tok;
        }
    }
    const int row = blockIdx.x * W + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int bpr  = ncols / X::qk;
    const int ystr = ncols / QK8_1;   // q8_1 blocks per activation row
    float     t[MT], tg[MT];
#pragma unroll
    for (int e = 0; e < MT; ++e) {
        t[e]  = 0.0f;
        tg[e] = 0.0f;
    }
    const int off0 = ch * stride_ch + row * stride_row;
    for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
        const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
        X         xu, xg;
        xu.load(up, off0 + kbx, kqs);
        xg.load(gate, off0 + kbx, kqs);
#pragma unroll
        for (int e = 0; e < MT; ++e) {
            if (e < ne) {
                const block_q8_1 * yy = y + mt[e] * ystr + kby;
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
            if (lane == 0) {
                act[((long long) mt[e] * GU_NS + ms[e]) * nrows + row] = gu_silu(b) * a;
            }
        }
    }
}

// AI_GU_Q8: the same computation, plus the q8_1 copy of the activations that the down projection would
// otherwise quantise in a separate launch (one quantise per MoE layer per verify step).
// A q8_1 block is 32 consecutive rows of one (slot, token), so a block here covers 32 rows: warp w computes
// rows rb + w + 8 i (i = 0..3) exactly as moe_gu_verify_k computes its one row (same lane decomposition,
// same accumulation order -> bit-identical activations), stages them in shared memory, and then warp e
// emits the owner column e's block through ex::emit_q8_1 -- the producer epilogue that matches the
// ggml's quantize_q8_1 bit for bit (half-even rounding, clamp, (d, sum) in half2).
// grid (F / 32, GU_NS * T), block (32, 8). Requires F % 32 == 0. Ownership is per block (slot and token come
// from blockIdx.y), so a non-owner block exits entirely before the barrier.
template <ggml_type type, int MT>
__global__ void __launch_bounds__(256) moe_gu_verify_q8_k(const void * __restrict__ up, const void * __restrict__ gate,
                                                          const block_q8_1 * __restrict__ y, const int32_t * __restrict__ ids,
                                                          float * __restrict__ act, int ncols, int nrows, int stride_row,
                                                          int stride_ch, int ids_stride, int T, ai::ex::block_q8_1 * __restrict__ q8) {
    using X = XU<type>;
    constexpr int bpi = X::vdr * 32 / X::qi;
    const int     lane = threadIdx.x, warp = threadIdx.y;
    const int     slot = blockIdx.y % GU_NS, tok = blockIdx.y / GU_NS;
    const int     ch   = ids[slot + tok * ids_stride];
    const int     n    = GU_NS * T;
    const int     l1   = lane + 32;
    const int     id0  = lane < n ? ids[(lane % GU_NS) + (lane / GU_NS) * ids_stride] : -1;
    const int     id1  = l1 < n ? ids[(l1 % GU_NS) + (l1 / GU_NS) * ids_stride] : -1;
    const unsigned long long match = (unsigned long long) __ballot_sync(0xffffffffu, id0 == ch) |
                                     ((unsigned long long) __ballot_sync(0xffffffffu, id1 == ch) << 32);
    const unsigned long long lower = (1ull << (tok * GU_NS)) - 1ull;
    if (match & lower) {
        return;   // a lower token owns this expert (uniform over the block)
    }
    int      ms[MT], mt[MT], ne = 0;
    unsigned long long rest = match;
#pragma unroll
    for (int e = 0; e < MT; ++e) {
        if (rest) {
            const int b = __ffsll((long long) rest) - 1;
            rest &= rest - 1;
            ms[e] = b % GU_NS;
            mt[e] = b / GU_NS;
            ne    = e + 1;
        } else {
            ms[e] = slot;
            mt[e] = tok;
        }
    }
    __shared__ float sv[MT][32];
    const int rb   = blockIdx.x * 32;
    const int bpr  = ncols / X::qk;
    const int ystr = ncols / QK8_1;
#pragma unroll 1
    for (int i = 0; i < 4; ++i) {
        const int row = rb + warp + 8 * i;
        float     t[MT], tg[MT];
#pragma unroll
        for (int e = 0; e < MT; ++e) {
            t[e]  = 0.0f;
            tg[e] = 0.0f;
        }
        const int off0 = ch * stride_ch + row * stride_row;
        for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
            const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
            X         xu, xg;
            xu.load(up, off0 + kbx, kqs);
            xg.load(gate, off0 + kbx, kqs);
#pragma unroll
            for (int e = 0; e < MT; ++e) {
                if (e < ne) {
                    const block_q8_1 * yy = y + mt[e] * ystr + kby;
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
                if (lane == 0) {
                    const float v = gu_silu(b) * a;
                    act[((long long) mt[e] * GU_NS + ms[e]) * nrows + row] = v;
                    sv[e][row - rb] = v;
                }
            }
        }
    }
    __syncthreads();
    if (warp < ne) {   // one warp per owner column: 32 lanes = the 32 rows of one q8_1 block
#pragma unroll
        for (int e = 0; e < MT; ++e) {
            if (e == warp) {
                ai::ex::emit_q8_1(q8, ((long long) mt[e] * GU_NS + ms[e]) * nrows + rb + lane, sv[e][lane]);
            }
        }
    }
}

// AI_MOE_DN: the routed down projection at verify width, with the same expert dedup + weight-unpack reuse.
// Reference: ggml's ids MMVQ at ncols_dst == 1 (ALPHA_MMID_BI: tokens on the sample axis, small_k, 4 rows
// per block) or mul_mat_vec_q_moe -- for K = F = 512 both give each lane one vec_dot (kbx = l/(qi/vdr), kqs =
// vdr*(l%(qi/vdr))) followed by warp_reduce_sum, which is what this computes per (row, slot, token). Unlike
// gate/up, every (slot, token) has its own activation row, so the reuse is the weight unpack only.
// grid (N / W, GU_NS * T), block (32, W): warp = one output row of one (slot, token); owner = lowest token.
template <ggml_type type, int W, int MT>
__global__ void __launch_bounds__(32 * W) moe_down_verify_k(const void * __restrict__ down, const block_q8_1 * __restrict__ y,
                                                            const int32_t * __restrict__ ids, float * __restrict__ out,
                                                            int ncols, int nrows, int stride_row, int stride_ch,
                                                            int ids_stride, int T) {
    using X = XU<type>;
    constexpr int bpi = X::vdr * 32 / X::qi;
    const int     lane = threadIdx.x;
    const int     slot = blockIdx.y % GU_NS, tok = blockIdx.y / GU_NS;
    const int     ch   = ids[slot + tok * ids_stride];
    const int     n    = GU_NS * T;
    const int     l1   = lane + 32;
    const int     id0  = lane < n ? ids[(lane % GU_NS) + (lane / GU_NS) * ids_stride] : -1;
    const int     id1  = l1 < n ? ids[(l1 % GU_NS) + (l1 / GU_NS) * ids_stride] : -1;
    const unsigned long long match = (unsigned long long) __ballot_sync(0xffffffffu, id0 == ch) |
                                     ((unsigned long long) __ballot_sync(0xffffffffu, id1 == ch) << 32);
    const unsigned long long lower = (1ull << (tok * GU_NS)) - 1ull;
    if (match & lower) {
        return;   // a lower token owns this expert
    }
    int      cs[MT], ne = 0;   // owner entries by channel c = token * GU_NS + slot (= the ballot bit index)
    unsigned long long rest = match;
#pragma unroll
    for (int e = 0; e < MT; ++e) {
        if (rest) {
            cs[e] = __ffsll((long long) rest) - 1;
            rest &= rest - 1;
            ne    = e + 1;
        } else {
            cs[e] = tok * GU_NS + slot;
        }
    }
    const int row = blockIdx.x * W + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int bpr  = ncols / X::qk;
    const int ystr = ncols / QK8_1;
    float     t[MT];
#pragma unroll
    for (int e = 0; e < MT; ++e) {
        t[e] = 0.0f;
    }
    const int off0 = ch * stride_ch + row * stride_row;
    for (int kbx = lane / (X::qi / X::vdr); kbx < bpr; kbx += bpi) {
        const int kby = kbx * (X::qk / QK8_1), kqs = X::vdr * (lane % (X::qi / X::vdr));
        X         xd;
        xd.load(down, off0 + kbx, kqs);
#pragma unroll
        for (int e = 0; e < MT; ++e) {
            if (e < ne) {
                t[e] += xd.dot(y + (long long) cs[e] * ystr + kby);
            }
        }
    }
#pragma unroll
    for (int e = 0; e < MT; ++e) {
        if (e < ne) {
            const float a = warp_reduce_sum<32>(t[e]);
            if (lane == 0) {
                out[(long long) cs[e] * nrows + row] = a;
            }
        }
    }
}

__device__ unsigned long long g_dn_cmp[3];   // AI_MOE_DN=2: calls, values compared, values differing

__global__ void moe_down_compare_k(const float * a, const float * b, long long n) {
    unsigned long long d = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        d += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    }
    atomicAdd(&g_dn_cmp[2], d);
    if (threadIdx.x == 0) {
        atomicAdd(&g_dn_cmp[0], 1ull);
        atomicAdd(&g_dn_cmp[1], (unsigned long long) n);
    }
}

__device__ unsigned long long g_gu_cmp[4];   // calls, values compared, values differing, max T
__device__ double             g_gu_err[2];   // sum (a - b)^2, sum b^2 (relative L2, for the non-exact AI_MOE_TC path)

__global__ void moe_gu_compare_k(const float * a, const float * b, long long n, int T) {
    unsigned long long d  = 0;
    double             e2 = 0.0, r2 = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        d += __float_as_uint(a[i]) != __float_as_uint(b[i]);
        const double df = (double) a[i] - (double) b[i];
        e2 += df * df;
        r2 += (double) b[i] * (double) b[i];
    }
    atomicAdd(&g_gu_cmp[2], d);
    atomicAdd(&g_gu_err[0], e2);
    atomicAdd(&g_gu_err[1], r2);
    if (threadIdx.x == 0) {
        atomicAdd(&g_gu_cmp[0], 1ull);
        atomicAdd(&g_gu_cmp[1], (unsigned long long) n);
        atomicMax(&g_gu_cmp[3], (unsigned long long) T);
    }
}

}   // namespace

namespace ai::kern {

bool moe_gu_verify_ok(int ggml_type_id, int64_t K, int64_t F, int64_t n_used, int64_t T) {
    const bool type_ok = ggml_type_id == GGML_TYPE_Q4_K || ggml_type_id == GGML_TYPE_Q5_K;
    // T 5..8 is bit-exact but not faster: one owner warp serves every token sharing its expert, and at 8 tokens
    // that imbalance outweighs the unpack reuse. Off unless AI_MOE_GU8=1.
    static const bool t8 = getenv("AI_MOE_GU8") && atoi(getenv("AI_MOE_GU8")) != 0;
    return type_ok && n_used == GU_NS && T >= 2 && T <= (t8 ? GU_MAXT : 4) && K % QK_K == 0 && F % 4 == 0;
}

bool moe_gu_verify(void * stream, int ggml_type_id, const void * up, const void * gate, int64_t row_bytes,
                   int64_t expert_bytes, const void * x_q8, const int32_t * ids, int64_t ids_stride, float * act,
                   int64_t K, int64_t F, int64_t T, void * act_q8) {
    if (!moe_gu_verify_ok(ggml_type_id, K, F, GU_NS, T)) {
        return false;
    }
    const size_t bs = ggml_type_id == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);
    if (row_bytes % bs != 0 || expert_bytes % bs != 0) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[moe-gu] LAUNCHED: verify gate/up + SwiGLU in one kernel, expert dedup + x reuse\n");
    }
    constexpr int W = 4;
    const dim3    grid((unsigned) ((F + W - 1) / W), (unsigned) (GU_NS * T));
    const dim3    block(32, W);
    cudaStream_t  st = (cudaStream_t) stream;
    const int     sr = (int) (row_bytes / bs), sc = (int) (expert_bytes / bs);
    // MT = the most tokens one owner can serve; T <= 4 keeps the 4-wide accumulators it was measured with
    const block_q8_1 * y = (const block_q8_1 *) x_q8;
    const int          ist = (int) ids_stride, nc = (int) K, nr = (int) F, t = (int) T;
    if (act_q8 != nullptr) {   // AI_GU_Q8: also emit the down projection's q8_1 input
        if (F % 32 != 0) {
            return false;
        }
        static bool said_q8 = false;
        if (!said_q8) {
            said_q8 = true;
            fprintf(stderr, "[moe-gu] q8_1 copy of the activations EMITTED (down-input quantise folded in)\n");
        }
        auto * q = (ai::ex::block_q8_1 *) act_q8;
        const dim3 g8((unsigned) (F / 32), (unsigned) (GU_NS * T)), b8(32, 8);
        if (ggml_type_id == GGML_TYPE_Q4_K) {
            if (t <= 4) moe_gu_verify_q8_k<GGML_TYPE_Q4_K, 4><<<g8, b8, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t, q);
            else        moe_gu_verify_q8_k<GGML_TYPE_Q4_K, 8><<<g8, b8, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t, q);
        } else {
            if (t <= 4) moe_gu_verify_q8_k<GGML_TYPE_Q5_K, 4><<<g8, b8, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t, q);
            else        moe_gu_verify_q8_k<GGML_TYPE_Q5_K, 8><<<g8, b8, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t, q);
        }
        return true;
    }
    if (ggml_type_id == GGML_TYPE_Q4_K) {
        if (t <= 4) {
            moe_gu_verify_k<GGML_TYPE_Q4_K, W, 4><<<grid, block, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t);
        } else {
            moe_gu_verify_k<GGML_TYPE_Q4_K, W, 8><<<grid, block, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t);
        }
    } else {
        if (t <= 4) {
            moe_gu_verify_k<GGML_TYPE_Q5_K, W, 4><<<grid, block, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t);
        } else {
            moe_gu_verify_k<GGML_TYPE_Q5_K, W, 8><<<grid, block, 0, st>>>(up, gate, y, ids, act, nc, nr, sr, sc, ist, t);
        }
    }
    return true;
}

void moe_gu_compare(void * stream, const float * mine, const float * ref, int64_t n, int64_t T) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[4] = { 0, 0, 0, 0 };
            double             e[2] = { 0.0, 0.0 };
            cudaMemcpyFromSymbol(h, g_gu_cmp, sizeof(h));
            cudaMemcpyFromSymbol(e, g_gu_err, sizeof(e));
            fprintf(stderr, "[moe-gu-compare] calls %llu | values %llu, differing %llu | max T %llu | rel-L2 %.3e\n", h[0],
                    h[1], h[2], h[3], e[1] > 0.0 ? sqrt(e[0] / e[1]) : 0.0);
        });
    }
    moe_gu_compare_k<<<1, 256, 0, (cudaStream_t) stream>>>(mine, ref, (long long) n, (int) T);
}

bool moe_down_verify_ok(int ggml_type_id, int64_t F, int64_t N, int64_t n_used, int64_t T) {
    const bool type_ok = ggml_type_id == GGML_TYPE_Q4_K || ggml_type_id == GGML_TYPE_Q5_K;
    return type_ok && n_used == GU_NS && T >= 2 && T <= GU_MAXT && F % QK_K == 0 && N % 4 == 0;
}

bool moe_down_verify(void * stream, int ggml_type_id, const void * down, int64_t row_bytes, int64_t expert_bytes,
                     const void * act_q8, const int32_t * ids, int64_t ids_stride, float * out, int64_t F, int64_t N,
                     int64_t T) {
    if (!moe_down_verify_ok(ggml_type_id, F, N, GU_NS, T)) {
        return false;
    }
    const size_t bs = ggml_type_id == GGML_TYPE_Q4_K ? sizeof(block_q4_K) : sizeof(block_q5_K);
    if (row_bytes % bs != 0 || expert_bytes % bs != 0) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[moe-dn] LAUNCHED: verify down projection, expert dedup + weight-unpack reuse\n");
    }
    constexpr int W = 4;
    const dim3    grid((unsigned) ((N + W - 1) / W), (unsigned) (GU_NS * T));
    const dim3    block(32, W);
    cudaStream_t  st = (cudaStream_t) stream;
    const int     sr = (int) (row_bytes / bs), sc = (int) (expert_bytes / bs);
    const block_q8_1 * y = (const block_q8_1 *) act_q8;
    const int          ist = (int) ids_stride, nc = (int) F, nr = (int) N, t = (int) T;
    if (ggml_type_id == GGML_TYPE_Q4_K) {
        if (t <= 4) moe_down_verify_k<GGML_TYPE_Q4_K, W, 4><<<grid, block, 0, st>>>(down, y, ids, out, nc, nr, sr, sc, ist, t);
        else        moe_down_verify_k<GGML_TYPE_Q4_K, W, 8><<<grid, block, 0, st>>>(down, y, ids, out, nc, nr, sr, sc, ist, t);
    } else {
        if (t <= 4) moe_down_verify_k<GGML_TYPE_Q5_K, W, 4><<<grid, block, 0, st>>>(down, y, ids, out, nc, nr, sr, sc, ist, t);
        else        moe_down_verify_k<GGML_TYPE_Q5_K, W, 8><<<grid, block, 0, st>>>(down, y, ids, out, nc, nr, sr, sc, ist, t);
    }
    return true;
}

void moe_down_compare(void * stream, const float * mine, const float * ref, int64_t n) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[3] = { 0, 0, 0 };
            cudaMemcpyFromSymbol(h, g_dn_cmp, sizeof(h));
            fprintf(stderr, "[moe-dn-compare] calls %llu | values %llu, differing %llu\n", h[0], h[1], h[2]);
        });
    }
    moe_down_compare_k<<<1, 256, 0, (cudaStream_t) stream>>>(mine, ref, (long long) n);
}

}   // namespace ai::kern
