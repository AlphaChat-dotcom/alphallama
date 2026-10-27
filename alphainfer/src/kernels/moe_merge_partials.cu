// Take the next layer's input norm off the critical path (AI_FUSE_MERGENORM=4).
//
// Under graph replay, removing launches is worth little; what the norm costs is its own serial execution
// -- a read of the row, a block reduction, a write -- between the merge and the quantise. Folding it into
// the merge kernel keeps that work in the chain, so instead the work is removed:
//
//   before:  merge -> rms_norm -> quantize_q8_1 (per consumer) -> MMVQ
//   after:   merge + partials -> norm-apply + q8_1 (one) -> MMVQ, which finds the staged copy
//
//   - the sum of squares is computed by the merge from values it already holds in registers, as
//     per-warp partials. No re-read, no block reduction.
//   - the scale is applied by a kernel with the quantise's shape (parallel, one warp per q8_1 block),
//     which reads the row anyway, and it emits the q8_1 copy into the staging registry so the next
//     layer's projections skip their own quantise.
//
// Bit-exact by construction (exact.cuh). The reference is rms_norm_f32<1024> for E >= 1024: 1024
// lanes, lane v accumulates cols v, v+1024, ... with a contracting `tmp += x*x`, then a warp butterfly,
// then s_sum[warp] re-reduced by one more butterfly. Here merge thread v is lane v (grid 1024/256 blocks
// per row), so each warp's butterfly result is exactly s_sum[v/32]; the apply kernel's warps re-reduce
// the same 32 values in the same lane order, as every warp of the reference does. The merge half uses
// the non-contracting ops and ex::sigmoid, as moe_merge.cu. The q8_1 copy is ex::emit_q8_1, which is
// byte-identical to ggml's quantize_q8_1.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define MMP_LANES   1024   // the reference norm's block size for E >= 1024
#define MMP_THREADS 256

__global__ static void moe_merge_gated_partials_k(const float * __restrict__ e, const float * __restrict__ w,
                                                  const float * __restrict__ sd, const float * __restrict__ sc,
                                                  const float * __restrict__ r, float * __restrict__ y,
                                                  float * __restrict__ part, long long E, int k,
                                                  long long sc_stride, long long sc_off) {
    const long long t    = blockIdx.y;
    const int       v    = blockIdx.x * MMP_THREADS + threadIdx.x;   // the reference lane
    const long long base = (long long) k * t;
    const float     sig  = ai::ex::sigmoid(sc[t * sc_stride + sc_off]);

    float tmp = 0.0f;
    for (long long i = v; i < E; i += MMP_LANES) {
        const long long idx = t * E + i;
        float acc = ai::ex::mul(e[i + E * base], w[base]);
        for (int j = 1; j < k; ++j) {
            acc = ai::ex::add(acc, ai::ex::mul(e[i + E * (base + j)], w[base + j]));
        }
        acc = ai::ex::add(acc, ai::ex::mul(sd[idx], sig));
        const float val = ai::ex::add(acc, r[idx]);
        y[idx]          = val;
        tmp             = __fmaf_rn(val, val, tmp);   // norm.cu's `tmp += x*x`, contracted under fast-math
    }
    tmp = ai::ex::warp_sum(tmp);
    if (threadIdx.x % 32 == 0) {
        part[t * 32 + v / 32] = tmp;   // == the reference's s_sum[warp_id]
    }
}

// The same, with the NC columns of a lane interleaved inside the expert loop. The loop form above is
// ~1.6x slower than moe_merge_gated_k, because a lane walks its two columns one after the other on half
// the threads -- half the loads in flight. Here both columns' loads for expert j issue together. Per column the operation order is unchanged (bit-exact); the sum of squares still
// takes column c before c+1, as the reference lane does.
template <int NC>
__global__ static void moe_merge_gated_partials_nc_k(const float * __restrict__ e, const float * __restrict__ w,
                                                     const float * __restrict__ sd, const float * __restrict__ sc,
                                                     const float * __restrict__ r, float * __restrict__ y,
                                                     float * __restrict__ part, int k, long long sc_stride,
                                                     long long sc_off) {
    constexpr long long E    = (long long) NC * MMP_LANES;
    const long long     t    = blockIdx.y;
    const int           v    = blockIdx.x * MMP_THREADS + threadIdx.x;
    const long long     base = (long long) k * t;
    const float         sig  = ai::ex::sigmoid(sc[t * sc_stride + sc_off]);

    float acc[NC];
    const float w0 = w[base];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        acc[c] = ai::ex::mul(e[v + c * MMP_LANES + E * base], w0);
    }
    for (int j = 1; j < k; ++j) {
        const float wj = w[base + j];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            acc[c] = ai::ex::add(acc[c], ai::ex::mul(e[v + c * MMP_LANES + E * (base + j)], wj));
        }
    }
    float tmp = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const long long idx = t * E + v + c * MMP_LANES;
        const float     val = ai::ex::add(ai::ex::add(acc[c], ai::ex::mul(sd[idx], sig)), r[idx]);
        y[idx]              = val;
        tmp                 = __fmaf_rn(val, val, tmp);
    }
    tmp = ai::ex::warp_sum(tmp);
    if (threadIdx.x % 32 == 0) {
        part[t * 32 + v / 32] = tmp;
    }
}

// The post-attention residual add, same shape (AI_ADDNORM_SPLIT). add_rmsnorm runs one block per row, so
// at decode (T = 1..8) two blocks carry the whole add + reduction + apply. Here the add runs on the
// full lane grid and writes the reference's
// per-warp partials; rms_apply_q8_k finishes the norm. Reference: add_rmsnorm_k<1024>, whose `v = m + s`
// is a plain add and whose `tmp += v*v` contracts (default contraction there, as norm.cu).
template <int NC>
__global__ static void add_partials_nc_k(const float * __restrict__ m, const float * __restrict__ s,
                                         float * __restrict__ r, float * __restrict__ part) {
    constexpr long long E = (long long) NC * MMP_LANES;
    const long long     t = blockIdx.y;
    const int           v = blockIdx.x * MMP_THREADS + threadIdx.x;
    float               x[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const long long idx = t * E + v + c * MMP_LANES;
        x[c]                = __fadd_rn(m[idx], s[idx]);   // a plain add is one rounding either way
        r[idx]              = x[c];
    }
    float tmp = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        tmp = __fmaf_rn(x[c], x[c], tmp);
    }
    tmp = ai::ex::warp_sum(tmp);
    if (threadIdx.x % 32 == 0) {
        part[t * 32 + v / 32] = tmp;
    }
}

// AI_ADDNORM_SPLIT=2: one kernel, full grid, and every block redundantly reduces the whole row. The split
// (=1) pays a second kernel's latency floor on the add + partials, which eats most of its saving. Here
// each block reads the row's 2 x 8 KB from L2, sums squares in rms_norm_f32<1024>'s exact
// order (256 threads emulate the 1024 lanes: 4 virtual lanes each, lanes of real warp w at virtual index m
// form virtual warp w + 8m), then adds, norms and emits only its own 256 columns. Every block computes the
// identical sum, so the result is bit-exact and deterministic with no cross-block traffic.
template <int NC>
__global__ static void add_rmsnorm_redundant_nc_k(const float * __restrict__ m, const float * __restrict__ s,
                                                  const float * __restrict__ nw, float * __restrict__ r,
                                                  float * __restrict__ p, ai::ex::block_q8_1 * __restrict__ q8,
                                                  float eps) {
    constexpr long long E  = (long long) NC * MMP_LANES;
    constexpr int       VL = MMP_LANES / MMP_THREADS;   // virtual lanes per thread
    __shared__ float    s_sum[32];
    const long long     t   = blockIdx.y;
    const int           tid = threadIdx.x;
    const float *       mr  = m + t * E;
    const float *       sr  = s + t * E;

    float x[VL][NC];
#pragma unroll
    for (int u = 0; u < VL; ++u) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int col = tid + MMP_THREADS * u + MMP_LANES * c;
            x[u][c]       = __fadd_rn(mr[col], sr[col]);
        }
    }
#pragma unroll
    for (int u = 0; u < VL; ++u) {
        float tmp = 0.0f;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            tmp = __fmaf_rn(x[u][c], x[u][c], tmp);
        }
        tmp = ai::ex::warp_sum(tmp);
        if (tid % 32 == 0) {
            s_sum[tid / 32 + (MMP_THREADS / 32) * u] = tmp;
        }
    }
    __syncthreads();
    const float sum   = ai::ex::warp_sum(s_sum[tid % 32]);
    const float scale = ai::ex::rms_scale(sum, (int) E, eps);

    const long long col = (long long) blockIdx.x * MMP_THREADS + tid;   // this block's own columns
    const float     v   = __fadd_rn(mr[col], sr[col]);
    r[t * E + col]      = v;
    const float     o   = ai::ex::rms_apply(scale, v, nw[col]);
    p[t * E + col]      = o;
    if (q8 != nullptr) {
        ai::ex::emit_q8_1(q8, t * E + col, o);
    }
}

__global__ static void rms_apply_q8_k(const float * __restrict__ y, const float * __restrict__ part,
                                      const float * __restrict__ nw, float * __restrict__ out,
                                      ai::ex::block_q8_1 * __restrict__ q8, long long E, float eps) {
    const long long t   = blockIdx.y;
    const long long col = (long long) blockIdx.x * MMP_THREADS + threadIdx.x;
    const float     sum   = ai::ex::warp_sum(part[t * 32 + threadIdx.x % 32]);   // every warp, as the reference
    const float     scale = ai::ex::rms_scale(sum, (int) E, eps);
    const float     v     = ai::ex::rms_apply(scale, y[t * E + col], nw[col]);
    out[t * E + col]      = v;
    if (q8 != nullptr) {
        ai::ex::emit_q8_1(q8, t * E + col, v);   // lanes 0..31 = one q8_1 block: E % 256 == 0 guarantees it
    }
}

// AI_ADDNORM_ROUTER: the post-attention add + rms_norm and the MoE router GEMV in one launch.
// The pair runs back to back in every layer and both are latency-bound. A second kernel cannot start its
// loads until the first retires; here the router blocks issue their weight loads first, while they
// redundantly recompute the row's norm scale from L2, so the weight latency hides behind the norm.
// Grid: blockIdx.x < E/256 is exactly add_rmsnorm_redundant_nc_k (copied, not refactored, so that kernel's
// code is untouched); blockIdx.x >= E/256 handles AR_RR router rows each.
// Bit-exact by construction: the norm half is the same code; each router block reproduces the scale with the
// same reduction, applies rms_apply to the 8 columns each thread needs, and runs ggml's
// mul_mat_vec_f<half, half, 1, 256> per row: half2 `sum += w * make_half2(y.x, y.y)` over col2 = tid + 256i,
// low + high, butterfly, buf_iw[warp] (32 zeroed slots), second butterfly (as kernels/router_topk.cu).
#define AR_RR 8
template <int NC>
__global__ static void __launch_bounds__(MMP_THREADS) addnorm_router_k(
        const float * __restrict__ m, const float * __restrict__ s, const float * __restrict__ nw, float * __restrict__ r,
        float * __restrict__ p, ai::ex::block_q8_1 * __restrict__ q8, float eps, const __half * __restrict__ W, int nrows,
        float * __restrict__ logits) {
    constexpr long long E  = (long long) NC * MMP_LANES;
    constexpr int       VL = MMP_LANES / MMP_THREADS;
    constexpr int       NB = (int) (E / MMP_THREADS);   // norm blocks
    __shared__ float    s_sum[32];
    const long long     t   = blockIdx.y;
    const int           tid = threadIdx.x;
    const float *       mr  = m + t * E;
    const float *       sr  = s + t * E;
    const bool          is_router = (int) blockIdx.x >= NB;

    // router blocks: weight loads first (independent of the norm)
    constexpr int NC2 = (int) (E / 2) / MMP_THREADS;   // col2 iterations per thread (4)
    half2          wx[AR_RR][NC2];
    const int      rb = ((int) blockIdx.x - NB) * AR_RR;
    if (is_router) {
        const half2 * w2 = (const half2 *) W;
#pragma unroll
        for (int rr = 0; rr < AR_RR; ++rr) {
#pragma unroll
            for (int i = 0; i < NC2; ++i) {
                wx[rr][i] = rb + rr < nrows ? w2[(long long) (rb + rr) * (E / 2) + tid + MMP_THREADS * i]
                                            : make_half2(0.0f, 0.0f);
            }
        }
    }

    // the row's sum of squares in rms_norm_f32<1024>'s order (as add_rmsnorm_redundant_nc_k)
    float x[VL][NC];
#pragma unroll
    for (int u = 0; u < VL; ++u) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int col = tid + MMP_THREADS * u + MMP_LANES * c;
            x[u][c]       = __fadd_rn(mr[col], sr[col]);
        }
    }
#pragma unroll
    for (int u = 0; u < VL; ++u) {
        float tmp = 0.0f;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            tmp = __fmaf_rn(x[u][c], x[u][c], tmp);
        }
        tmp = ai::ex::warp_sum(tmp);
        if (tid % 32 == 0) {
            s_sum[tid / 32 + (MMP_THREADS / 32) * u] = tmp;
        }
    }
    __syncthreads();
    const float sum   = ai::ex::warp_sum(s_sum[tid % 32]);
    const float scale = ai::ex::rms_scale(sum, (int) E, eps);

    if (!is_router) {
        const long long col = (long long) blockIdx.x * MMP_THREADS + tid;
        const float     v   = __fadd_rn(mr[col], sr[col]);
        r[t * E + col]      = v;
        const float     o   = ai::ex::rms_apply(scale, v, nw[col]);
        p[t * E + col]      = o;
        if (q8 != nullptr) {
            ai::ex::emit_q8_1(q8, t * E + col, o);
        }
        return;
    }

    // router: this thread's normed inputs, then ggml's per-row dot and reduction
    half2 yh[NC2];
#pragma unroll
    for (int i = 0; i < NC2; ++i) {
        const int   c0 = 2 * (tid + MMP_THREADS * i);
        const float o0 = ai::ex::rms_apply(scale, __fadd_rn(mr[c0], sr[c0]), nw[c0]);
        const float o1 = ai::ex::rms_apply(scale, __fadd_rn(mr[c0 + 1], sr[c0 + 1]), nw[c0 + 1]);
        yh[i]          = make_half2(o0, o1);
    }
    __shared__ float buf_iw[AR_RR][32];
    if (tid < 32) {
#pragma unroll
        for (int rr = 0; rr < AR_RR; ++rr) {
            buf_iw[rr][tid] = 0.0f;
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < AR_RR; ++rr) {
        half2 sumh2 = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < NC2; ++i) {
            sumh2 += wx[rr][i] * yh[i];
        }
        float sumf = __low2float(sumh2) + __high2float(sumh2);
        sumf       = ai::ex::warp_sum(sumf);
        buf_iw[rr][tid / 32] = sumf;
    }
    __syncthreads();
    if (tid < 32) {
#pragma unroll
        for (int rr = 0; rr < AR_RR; ++rr) {
            const float v = ai::ex::warp_sum(buf_iw[rr][tid]);
            if (tid == 0 && rb + rr < nrows) {
                logits[t * nrows + rb + rr] = v;
            }
        }
    }
}


// AI_MERGE_APPLY: the MoE merge and the next layer's norm-apply (+ q8_1 staging) in one launch. Like
// add_rmsnorm_redundant_nc_k, each of the E/256 blocks redundantly recomputes the merged row (8 experts +
// shared + residual: ~80 KB from L2) to obtain the row's per-warp partial sums in rms_norm_f32<1024>'s order
// (256 threads emulate the 1024 reference lanes, virtual warp w + 8u), then writes y, the normed output and
// its q8_1 copy for its own 256 columns only.
// Bit-exact by construction: the per-column merge is moe_merge_gated_partials_nc_k's op sequence (ex::mul /
// ex::add, ex::sigmoid, experts in slot order), the partials are its `tmp = fma(val, val, tmp)` over the two
// columns of a lane then the warp butterfly, and the finish is rms_apply_q8_k's (every warp re-reduces).
template <int NC>
__device__ __forceinline__ float ma_merge_col(const float * __restrict__ e, const float * __restrict__ w,
                                              const float * __restrict__ sd, const float * __restrict__ r, int k,
                                              long long base, long long t, float sig, int col) {
    constexpr long long E = (long long) NC * MMP_LANES;
    float acc = ai::ex::mul(e[col + E * base], w[base]);
    for (int j = 1; j < k; ++j) {
        acc = ai::ex::add(acc, ai::ex::mul(e[col + E * (base + j)], w[base + j]));
    }
    const long long idx = t * E + col;
    return ai::ex::add(ai::ex::add(acc, ai::ex::mul(sd[idx], sig)), r[idx]);
}

template <int NC>
__global__ static void __launch_bounds__(MMP_THREADS) merge_apply_redundant_k(
        const float * __restrict__ e, const float * __restrict__ w, const float * __restrict__ sd,
        const float * __restrict__ sc, const float * __restrict__ r, float * __restrict__ y,
        const float * __restrict__ nw, float * __restrict__ out, ai::ex::block_q8_1 * __restrict__ q8, int k,
        long long sc_stride, long long sc_off, float eps) {
    constexpr long long E  = (long long) NC * MMP_LANES;
    constexpr int       VL = MMP_LANES / MMP_THREADS;
    __shared__ float    s_sum[32];
    const long long     t    = blockIdx.y;
    const int           tid  = threadIdx.x;
    const long long     base = (long long) k * t;
    const float         sig  = ai::ex::sigmoid(sc[t * sc_stride + sc_off]);
#pragma unroll
    for (int u = 0; u < VL; ++u) {
        const int v   = tid + MMP_THREADS * u;
        float     tmp = 0.0f;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float val = ma_merge_col<NC>(e, w, sd, r, k, base, t, sig, v + c * MMP_LANES);
            tmp             = __fmaf_rn(val, val, tmp);
        }
        tmp = ai::ex::warp_sum(tmp);
        if (tid % 32 == 0) {
            s_sum[tid / 32 + (MMP_THREADS / 32) * u] = tmp;
        }
    }
    __syncthreads();
    const float     sum   = ai::ex::warp_sum(s_sum[tid % 32]);
    const float     scale = ai::ex::rms_scale(sum, (int) E, eps);
    const int       col   = blockIdx.x * MMP_THREADS + tid;
    const float     val   = ma_merge_col<NC>(e, w, sd, r, k, base, t, sig, col);
    y[t * E + col]        = val;
    const float     o     = ai::ex::rms_apply(scale, val, nw[col]);
    out[t * E + col]      = o;
    if (q8 != nullptr) {
        ai::ex::emit_q8_1(q8, t * E + col, o);
    }
}

// AI_ADDNORM_ROUTER=2 instrument: word compare
__device__ unsigned long long g_ar_cmp[3];
__global__ static void ar_cmp_k(const float * a, const float * b, long long n) {
    unsigned long long d = 0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        d += __float_as_uint(a[i]) != __float_as_uint(b[i]);
    }
    atomicAdd(&g_ar_cmp[2], d);
    if (threadIdx.x == 0) {
        atomicAdd(&g_ar_cmp[0], 1ull);
        atomicAdd(&g_ar_cmp[1], (unsigned long long) n);
    }
}

namespace ai::kern {

bool addnorm_router(void * stream, const float * mix, const float * res, const float * w, float * r, float * p,
                    void * q8, const void * router_w, int64_t nrows, float * logits, int64_t E, int64_t T, float eps) {
    if (E != 2 * MMP_LANES || T < 1 || T > 8) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[addnorm-router] LAUNCHED (norm %lld + router %lld blocks x %lld tokens, q8 %s)\n",
                (long long) (E / MMP_THREADS), (long long) ((nrows + AR_RR - 1) / AR_RR), (long long) T,
                q8 ? "STAGED" : "off");
    }
    const dim3 g((unsigned) (E / MMP_THREADS + (nrows + AR_RR - 1) / AR_RR), (unsigned) T);
    addnorm_router_k<2><<<g, MMP_THREADS, 0, (cudaStream_t) stream>>>(mix, res, w, r, p, (ai::ex::block_q8_1 *) q8, eps,
                                                                    (const __half *) router_w, (int) nrows, logits);
    return true;
}

bool moe_merge_apply_redundant(void * stream, const float * exps, const float * w, const float * shared_down,
                               const float * score, int64_t score_stride, int64_t score_off, const float * res,
                               float * y, const float * norm_w, float * out, void * q8, int64_t E, int64_t k, int64_t T,
                               float eps) {
    if (E != 2 * MMP_LANES) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[merge-apply] LAUNCHED (%lld x %lld blocks, merge recomputed per block, q8 %s)\n",
                (long long) (E / MMP_THREADS), (long long) T, q8 ? "STAGED" : "off");
    }
    const dim3 g((unsigned) (E / MMP_THREADS), (unsigned) T);
    merge_apply_redundant_k<2><<<g, MMP_THREADS, 0, (cudaStream_t) stream>>>(
        exps, w, shared_down, score, res, y, norm_w, out, (ai::ex::block_q8_1 *) q8, (int) k, (long long) score_stride,
        (long long) score_off, eps);
    return true;
}

void addnorm_router_compare(void * stream, const float * a, const float * b, int64_t n) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[3] = { 0, 0, 0 };
            cudaMemcpyFromSymbol(h, g_ar_cmp, sizeof(h));
            fprintf(stderr, "[addnorm-router-compare] arrays %llu | words %llu, differing %llu\n", h[0], h[1], h[2]);
        });
    }
    ar_cmp_k<<<1, 256, 0, (cudaStream_t) stream>>>(a, b, (long long) n);
}

bool moe_merge_partials_ok(int64_t E) { return E >= MMP_LANES && E % MMP_THREADS == 0; }

bool add_rmsnorm_redundant(void * stream, const float * mix, const float * res, const float * w, float * r,
                           float * p, void * q8, int64_t E, int64_t T, float eps) {
    if (E != 2 * MMP_LANES && E != 5 * MMP_LANES) {   // 35B (2048) and 27B (5120): NC columns per reference lane
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[addnorm-redundant] LAUNCHED (%lld x %lld blocks, q8 %s)\n", (long long) (E / MMP_THREADS),
                (long long) T, q8 ? "STAGED" : "off");
    }
    const dim3 g((unsigned) (E / MMP_THREADS), (unsigned) T);
    if (E == 5 * MMP_LANES) {
        add_rmsnorm_redundant_nc_k<5><<<g, MMP_THREADS, 0, (cudaStream_t) stream>>>(mix, res, w, r, p,
                                                                             (ai::ex::block_q8_1 *) q8, eps);
        return true;
    }
    add_rmsnorm_redundant_nc_k<2><<<g, MMP_THREADS, 0, (cudaStream_t) stream>>>(mix, res, w, r, p,
                                                                             (ai::ex::block_q8_1 *) q8, eps);
    return true;
}

bool add_rmsnorm_split(void * stream, const float * mix, const float * res, const float * w, float * r, float * p,
                       float * partials, void * q8, int64_t E, int64_t T, float eps) {
    if (E != 2 * MMP_LANES) {
        return false;   // only the NC=2 shape is built; the caller keeps add_rmsnorm
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[addnorm-split] LAUNCHED (add %d x %lld blocks, apply %lld x %lld, q8 %s)\n",
                MMP_LANES / MMP_THREADS, (long long) T, (long long) (E / MMP_THREADS), (long long) T,
                q8 ? "STAGED" : "off");
    }
    const dim3 ga(MMP_LANES / MMP_THREADS, (unsigned) T);
    add_partials_nc_k<2><<<ga, MMP_THREADS, 0, (cudaStream_t) stream>>>(mix, res, r, partials);
    const dim3 gb((unsigned) (E / MMP_THREADS), (unsigned) T);
    rms_apply_q8_k<<<gb, MMP_THREADS, 0, (cudaStream_t) stream>>>(r, partials, w, p, (ai::ex::block_q8_1 *) q8,
                                                                 (long long) E, eps);
    return true;
}

void moe_merge_residual_gated_partials(void * stream, const float * exps, const float * w,
                                       const float * shared_down, const float * score, int64_t score_stride,
                                       int64_t score_off, const float * res, float * y, float * partials,
                                       int64_t E, int64_t k, int64_t T, bool interleave) {
    const dim3 grid(MMP_LANES / MMP_THREADS, (unsigned) T);
    if (interleave && E == 2 * MMP_LANES) {
        moe_merge_gated_partials_nc_k<2><<<grid, MMP_THREADS, 0, (cudaStream_t) stream>>>(
            exps, w, shared_down, score, res, y, partials, (int) k, (long long) score_stride,
            (long long) score_off);
        return;
    }
    moe_merge_gated_partials_k<<<grid, MMP_THREADS, 0, (cudaStream_t) stream>>>(
        exps, w, shared_down, score, res, y, partials, (long long) E, (int) k, (long long) score_stride,
        (long long) score_off);
}

void rms_apply_partials_q8(void * stream, const float * y, const float * partials, const float * norm_w,
                           float * out, void * q8, int64_t E, int64_t T, float eps) {
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[merge-partials] LAUNCHED (merge %d x %lld blocks, apply %lld x %lld blocks, q8 %s)\n",
                MMP_LANES / MMP_THREADS, (long long) T, (long long) (E / MMP_THREADS), (long long) T,
                q8 ? "STAGED" : "off");
    }
    const dim3 grid((unsigned) (E / MMP_THREADS), (unsigned) T);
    rms_apply_q8_k<<<grid, MMP_THREADS, 0, (cudaStream_t) stream>>>(y, partials, norm_w, out,
                                                                   (ai::ex::block_q8_1 *) q8, (long long) E, eps);
}

}   // namespace ai::kern
