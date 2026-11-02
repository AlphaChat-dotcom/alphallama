// SPDX-License-Identifier: Apache-2.0
// Fused qwen4exp MoE routing for T <= 16 rows -- the router GEMV straight from BF16/F16 weights, then
// softmax over the experts, top-k and the renormalised weights, in two launches (the ggml chain takes ~10: BF16 GEMM via
// cutlass + 2 activation converts + softmax + 512-wide argsort + get_rows + sum_rows + clamp + div + cont).
//
// Not bit-identical to that chain: ggml rounds the activations to BF16 before its GEMM and its argsort breaks ties
// differently, so this is a fast-mode path; validate any change to it by paired NLL like any routing change.
#include "kernels/kernels.h"
#include "kernels/q4x_plan.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kMaxT = 16;   // windows up to 16 rows

template <typename W> __device__ __forceinline__ float wf(W v);
template <> __device__ __forceinline__ float wf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float wf<__half>(__half v) { return __half2float(v); }

// One 64-thread block per expert row e: logits[t * E + e] = dot(W[e, :K], x[:, t]) for t < T. Each thread owns
// K / 8 / 64 16-byte weight chunks, all loaded up front for memory-level parallelism (one warp per row with a looped
// load is latency-bound), then a 2-warp reduction.
constexpr int kRouteThreads = 64, kRouteMaxChunks = 8;   // K <= 64 * 8 * 8 = 4096
template <typename W>
__global__ void __launch_bounds__(kRouteThreads) router_gemv_k(const W * __restrict__ w, const float * __restrict__ x,
                                                               float * __restrict__ logits, int K, int E, int T) {
    const int e = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    const uint4 * row = (const uint4 *) (w + (size_t) e * K);
    const int nchunk = K / 8 / kRouteThreads;
    uint4 q[kRouteMaxChunks];
#pragma unroll
    for (int u = 0; u < kRouteMaxChunks; ++u) if (u < nchunk) q[u] = __ldg(row + tid + u * kRouteThreads);
    float acc[kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) acc[t] = 0.0f;
#pragma unroll
    for (int u = 0; u < kRouteMaxChunks; ++u) {
        if (u >= nchunk) break;
        const W * wv = (const W *) &q[u];
        const int c  = tid + u * kRouteThreads;
#pragma unroll
        for (int t = 0; t < kMaxT; ++t) {
            if (t >= T) break;
            const float4 * xp = (const float4 *) (x + (size_t) t * K + (size_t) c * 8);
            const float4   a = __ldg(xp), b = __ldg(xp + 1);
            acc[t] += wf<W>(wv[0]) * a.x + wf<W>(wv[1]) * a.y + wf<W>(wv[2]) * a.z + wf<W>(wv[3]) * a.w +
                      wf<W>(wv[4]) * b.x + wf<W>(wv[5]) * b.y + wf<W>(wv[6]) * b.z + wf<W>(wv[7]) * b.w;
        }
    }
    __shared__ float part[2][kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
        if (t >= T) break;
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) part[wid][t] = v;
    }
    __syncthreads();
    if (tid < T) logits[(size_t) tid * E + e] = part[0][tid] + part[1][tid];
}

// [AI_Q4X_GEMV_XS, default 1] router_gemv_k with R rows per block and x [T, K] staged ONCE per block in shared memory.
// router_gemv_k re-reads all of x from L2 in every one-row block: for the 512-expert router at T = 4 that is ~21 MB of
// L2 loads against 2.6 MB of weights (7.1 us a call on the 5090, 51 calls a verify round; the 48-row GDN projections
// 7.8 us, 72 calls). Each row keeps router_gemv_k's 64 virtual threads, chunk order, per-t accumulation order and
// two-warp reduction, so the logits are bit-identical; only where x is read from changes.
template <typename W, int R, bool PRE>
__global__ void __launch_bounds__(kRouteThreads * R) router_gemv_xs_k(const W * __restrict__ w, const float * __restrict__ x,
                                                                      float * __restrict__ logits, int K, int E, int T) {
    extern __shared__ float4 xs4[];   // x [T, K]
    // [AI_Q4X_GEMV_PRELOAD=1, default 0: MEASURED NULL on the 3090 2026-11-01, rr9] the row's weight chunks are requested BEFORE x is staged, so the weight reads
    // overlap the staging instead of waiting for its barrier (same loads, same arithmetic: identical bits)
    const int r = threadIdx.x / kRouteThreads, tid = threadIdx.x % kRouteThreads, lane = tid & 31, wid = tid >> 5;
    const int e = blockIdx.x * R + r;
    const bool live = e < E;
    const uint4 * row = (const uint4 *) (w + (size_t) (live ? e : 0) * K);
    const int nchunk = K / 8 / kRouteThreads;
    uint4 q[kRouteMaxChunks];
    if (PRE) {
#pragma unroll
        for (int u = 0; u < kRouteMaxChunks; ++u) if (u < nchunk && live) q[u] = __ldg(row + tid + u * kRouteThreads);
    }
    const int n4 = T * K / 4;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) xs4[i] = __ldg((const float4 *) x + i);
    __syncthreads();
    if (!PRE) {
#pragma unroll
        for (int u = 0; u < kRouteMaxChunks; ++u) if (u < nchunk && live) q[u] = __ldg(row + tid + u * kRouteThreads);
    }
    const float * xs = (const float *) xs4;
    float acc[kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) acc[t] = 0.0f;
    if (live) {
#pragma unroll
        for (int u = 0; u < kRouteMaxChunks; ++u) {
            if (u >= nchunk) break;
            const W * wv = (const W *) &q[u];
            const int c  = tid + u * kRouteThreads;
#pragma unroll
            for (int t = 0; t < kMaxT; ++t) {
                if (t >= T) break;
                const float4 * xp = (const float4 *) (xs + (size_t) t * K + (size_t) c * 8);
                const float4   a = xp[0], b = xp[1];
                acc[t] += wf<W>(wv[0]) * a.x + wf<W>(wv[1]) * a.y + wf<W>(wv[2]) * a.z + wf<W>(wv[3]) * a.w +
                          wf<W>(wv[4]) * b.x + wf<W>(wv[5]) * b.y + wf<W>(wv[6]) * b.z + wf<W>(wv[7]) * b.w;
            }
        }
    }
    __shared__ float part[R][2][kMaxT];
#pragma unroll
    for (int t = 0; t < kMaxT; ++t) {
        if (t >= T) break;
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) part[r][wid][t] = v;
    }
    __syncthreads();
    if (live && tid < T) logits[(size_t) tid * E + e] = part[r][0][tid] + part[r][1][tid];
}

// the router-style GEMV launch: router_gemv_xs_k when x fits the default shared memory, else router_gemv_k
template <typename W>
static void router_gemv_launch(cudaStream_t st, const W * w, const float * x, float * out, int K, int E, int T) {
    static const bool xs_on = !getenv("AI_Q4X_GEMV_XS") || atoi(getenv("AI_Q4X_GEMV_XS")) != 0;
    constexpr int R = 4;
    const size_t sm = (size_t) T * K * sizeof(float);
    if (xs_on && sm <= 48 * 1024) {
        static const bool pre = getenv("AI_Q4X_GEMV_PRELOAD") && atoi(getenv("AI_Q4X_GEMV_PRELOAD")) != 0;
        if (pre) router_gemv_xs_k<W, R, true><<<(E + R - 1) / R, kRouteThreads * R, sm, st>>>(w, x, out, K, E, T);
        else     router_gemv_xs_k<W, R, false><<<(E + R - 1) / R, kRouteThreads * R, sm, st>>>(w, x, out, K, E, T);
    } else {
        router_gemv_k<W><<<E, kRouteThreads, 0, st>>>(w, x, out, K, E, T);
    }
}

// One block (E threads, E <= 1024) per row t: softmax over E, the top-k (descending; ties -> lower index), weights =
// p / max(sum of the top-k p, 6.1e-5) (qwen4exp's renormalisation, as ops.cpp's moe_route).

// [AI_Q4X_TOPK2, default 0 until gated] the same top-k in two levels: every warp takes its own k best (shuffles only,
// no block barrier), then warp 0 merges the nw * k candidates. The global top-k is inside the union of the warps' top-k
// and both levels order by (p descending, index ascending), so sel / the weights / tsum (summed in selection order) are
// those of the k block-wide rounds: 2 barriers instead of ~2k (E = 512, k = 10: 16 warps). Gated by AI_Q4X_TOPK_CHECK.
__device__ __forceinline__ bool tk_better(float ov, int oi, float v, int vi) { return ov > v || (ov == v && oi < vi); }
__device__ __forceinline__ void topk2_dev(float p, int32_t * __restrict__ sel, float * __restrict__ wout, int E, int k) {
    __shared__ float cv[32 * 32];
    __shared__ int   ci[32 * 32];
    __shared__ float tsum_s;
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, wid = e >> 5, nw = (blockDim.x + 31) >> 5;
    float pv = p;
    for (int j = 0; j < k; ++j) {
        float bv = pv;
        int   bi = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (tk_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { cv[wid * k + j] = bv; ci[wid * k + j] = bi; }
        if (e == bi) pv = -1.0f;   // taken (an exhausted warp repeats -1 candidates: below every real p >= 0)
    }
    __syncthreads();
    if (wid == 0) {
        const int nc = nw * k;
        float tsum = 0.0f;
        for (int j = 0; j < k; ++j) {
            float bv = -2.0f;
            int   bi = 0x7fffffff, bs = -1;   // this lane's best live candidate and its slot
            for (int c = lane; c < nc; c += 32)
                if (cv[c] > -1.5f && tk_better(cv[c], ci[c], bv, bi)) { bv = cv[c]; bi = ci[c]; bs = c; }
            float v = bv;
            int   vi = bi, vl = lane;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int   oi = __shfl_xor_sync(0xffffffffu, vi, o);
                const int   ol = __shfl_xor_sync(0xffffffffu, vl, o);
                if (tk_better(ov, oi, v, vi)) { v = ov; vi = oi; vl = ol; }
            }
            if (lane == vl) cv[bs] = -2.0f;   // remove exactly the winning slot (a dummy may share its index)
            __syncwarp();
            if (lane == 0) { sel[(size_t) t * k + j] = vi; wout[(size_t) t * k + j] = v; }
            tsum += v;
        }
        if (lane == 0) tsum_s = tsum;
    }
    __syncthreads();
    if (e < k) {
        const float d = fmaxf(tsum_s, 6.103515625e-5f);
        wout[(size_t) t * k + e] = wout[(size_t) t * k + e] / d;
    }
}
template <bool TK2 = false>
__device__ __forceinline__ void softmax_topk_dev(const float * __restrict__ logits, int32_t * __restrict__ sel,
                                                 float * __restrict__ wout, int E, int k) {
    extern __shared__ float sm[];   // [E] probabilities, then scratch
    float *     red = sm + E;      // [32]
    int *       redi = (int *) (red + 32);
    const int   t = blockIdx.x, e = threadIdx.x, lane = e & 31, wid = e >> 5, nw = (blockDim.x + 31) >> 5;
    const float l = e < E ? logits[(size_t) t * E + e] : -INFINITY;
    // max
    float m = l;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) red[wid] = m;
    __syncthreads();
    if (wid == 0) {
        float v = lane < nw ? red[lane] : -INFINITY;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float mx = red[0];
    __syncthreads();
    const float ex = e < E ? __expf(l - mx) : 0.0f;
    float s = ex;
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) red[wid] = s;
    __syncthreads();
    if (wid == 0) {
        float v = lane < nw ? red[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    __syncthreads();
    float p = e < E ? ex * inv : -1.0f;
    if (e < E) sm[e] = p;
    if (TK2) { topk2_dev(p, sel, wout, E, k); return; }
    // top-k by k rounds of a block arg-max (ties -> the lower index)
    float tsum = 0.0f;
    for (int j = 0; j < k; ++j) {
        float bv = p;
        int   bi = e;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { red[wid] = bv; redi[wid] = bi; }
        __syncthreads();
        if (wid == 0) {
            float v  = lane < nw ? red[lane] : -2.0f;
            int   vi = lane < nw ? redi[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int   oi = __shfl_xor_sync(0xffffffffu, vi, o);
                if (ov > v || (ov == v && oi < vi)) { v = ov; vi = oi; }
            }
            if (lane == 0) { red[0] = v; redi[0] = vi; }
        }
        __syncthreads();
        const int   win = redi[0];
        const float wp  = red[0];
        if (e == 0) { sel[(size_t) t * k + j] = win; wout[(size_t) t * k + j] = wp; }
        tsum += wp;
        if (e == win) p = -1.0f;   // taken
        __syncthreads();
    }
    if (e < k) {
        const float d = fmaxf(tsum, 6.103515625e-5f);
        wout[(size_t) t * k + e] = wout[(size_t) t * k + e] / d;
    }
}

__global__ void softmax_topk_k(const float * __restrict__ logits, int32_t * __restrict__ sel, float * __restrict__ wout,
                               int E, int k) {
    softmax_topk_dev(logits, sel, wout, E, k);
}

// [AI_Q4X_ROUTE_RING] softmax_topk_k + offload.cu's resolve_ring_k in ONE launch: after its row's top-k, block t writes
// its k slots (resident slot, else filler0 + j) and miss ids (-(e+2) = hit), its routing weights to the bell, and its
// token's x column to the mapped bell (T blocks copy in parallel; resolve_ring_k did all of it in one block). Every
// block fences at system scope, then counts itself in; the LAST block writes the token count and the window sequence,
// as resolve_ring_k's thread 0 did. The softmax / top-k arithmetic is softmax_topk_dev unchanged: identical bits.
// (No GPU miss share and no wait-stats timing: the host keeps the two-kernel path for those.)
__device__ unsigned int g_ring_done = 0;
// [AI_Q4X_TOPK_CHECK=1, debug] the ring's sel / weights against softmax_topk_k's (the k block-wide rounds), bytewise
__device__ unsigned g_tk_n = 0, g_tk_bad = 0;
__device__ int32_t  g_tk_sel[kMaxT * 32];
__device__ float    g_tk_w[kMaxT * 32];
__global__ void topk_check_k(const int32_t * a_sel, const float * a_w, const int32_t * b_sel, const float * b_w, int n) {
    __shared__ int bad;
    if (threadIdx.x == 0) bad = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (a_sel[i] != b_sel[i] || __float_as_uint(a_w[i]) != __float_as_uint(b_w[i])) atomicOr(&bad, 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned c = atomicAdd(&g_tk_n, 1u) + 1;
        unsigned       nb = g_tk_bad;
        if (bad) { nb = atomicAdd(&g_tk_bad, 1u) + 1; if (nb <= 8) printf("[topk-check] MISMATCH #%u at check %u\n", nb, c); }
        if (c % 2048 == 0) printf("[topk-check] %u checks, %u mismatched\n", c, nb);
    }
}
// [AI_Q4X_RING_QX] the experts' q8_1 copy of x (ggml quantize_q8_1, Q8_0 path), written by the ring kernel's blocks.
// The fork's kernel is built -use_fast_math and this file is not, so its exact instructions are spelled out (PTX of
// the fast-math reference on sm_86): abs/max/add.ftz in the same 32-lane xor butterfly, div.approx.ftz for amax / 127
// and x / d, roundf as add.rz.ftz of copysign(0.5) then truncation. Gated by AI_Q4X_RING_CHECK (bytewise compare).
__device__ __forceinline__ float qx_abs(float a) { float r; asm("abs.ftz.f32 %0, %1;" : "=f"(r) : "f"(a)); return r; }
__device__ __forceinline__ float qx_max(float a, float b) { float r; asm("max.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float qx_add(float a, float b) { float r; asm("add.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float qx_add_rz(float a, float b) { float r; asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
__device__ __forceinline__ float qx_div(float a, float b) { float r; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r; }
// one row of N floats -> N / 32 blocks { half2 (d, sum), int8 qs[32] } (36 bytes); the block's warps share the blocks
__device__ __forceinline__ void qx_row(const float * __restrict__ x, char * __restrict__ y, int64_t N) {
    const int lane = threadIdx.x & 31, nb = (int) (N / 32);
    for (int ib = (int) (threadIdx.x >> 5); ib < nb; ib += (int) (blockDim.x >> 5)) {
        const float xi = x[(size_t) ib * 32 + lane];
        float       amax = qx_abs(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = qx_max(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
            sum  = qx_add(sum, __shfl_xor_sync(0xffffffffu, sum, o, 32));
        }
        const float d = qx_div(amax, 127.0f);
        int         q = 0;
        if (amax != 0.0f) {   // amax is already flushed: the ftz compare and this one agree
            const float r = qx_div(xi, d);
            q = (int) truncf(qx_add_rz(r, copysignf(0.5f, r)));
        }
        char * b = y + (size_t) ib * 36;
        b[4 + lane] = (char) (signed char) q;
        if (lane == 0) *(__half2 *) b = __halves2half2(__float2half_rn(d), __float2half_rn(sum));
    }
}
template <bool TK2>
__global__ void softmax_topk_ring_k(const float * __restrict__ logits, int32_t * __restrict__ sel, float * __restrict__ wout,
                                    int E, int k, const int32_t * __restrict__ table, int32_t filler0,
                                    int32_t * __restrict__ slots, int32_t * miss_bell, const float * __restrict__ x,
                                    int64_t N, float * x_bell, int32_t * n_tok_bell, volatile unsigned * seq_bell,
                                    const unsigned * __restrict__ base, int layer, float * w_bell,
                                    q4xplan::Plan * plan, char * xq) {
    softmax_topk_dev<TK2>(logits, sel, wout, E, k);
    __syncthreads();   // this row's sel (thread 0) and renormalised weights (threads < k) are in global memory
    const int t = blockIdx.x, e = threadIdx.x;
    if (e < k) {
        const size_t  i  = (size_t) t * k + e;
        const int32_t ex = sel[i];
        const int32_t s  = table[ex];
        slots[i]     = s >= 0 ? s : filler0 + e;
        miss_bell[i] = s >= 0 ? -(ex + 2) : ex;
        if (w_bell) w_bell[i] = wout[i];
    }
    for (int64_t n = e; n < N; n += blockDim.x) x_bell[(size_t) t * N + n] = x[(size_t) t * N + n];
    if (xq) qx_row(x + (size_t) t * N, xq + (size_t) t * (N / 32) * 36, N);   // [AI_Q4X_RING_QX]
    __threadfence_system();   // every writing thread (see resolve_ring_k: one thread's fence does not cover the others)
    __syncthreads();
    __shared__ int is_last;
    if (e == 0) is_last = atomicAdd(&g_ring_done, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!is_last) return;
    // the last row: every block's slots and bell writes are visible device- and system-wide
    if (plan) {   // [plan in the ring] the experts call's Plan over the resident slots [0, filler0), plan_k's routine
        __shared__ int sid[q4xplan::kMaxCols];
        __shared__ int cnt;
        __threadfence();
        q4xplan::plan_build(slots, k, k, (int) gridDim.x * k, 0, filler0, plan, sid, cnt);
    }
    if (e == 0) {
        g_ring_done = 0;
        __threadfence();
        *n_tok_bell = (int32_t) gridDim.x;
        __threadfence_system();
        *seq_bell = *base + (unsigned) layer + 1;
        __threadfence_system();
    }
}

}  // namespace

namespace ai::kern {

bool q4x_route(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int E, int T, int k,
               float * logits, int32_t * sel, float * weights) {
    if (T < 1 || T > kMaxT || E > 1024 || k > E || K % 256 != 0 || ((uintptr_t) w & 15) || ((uintptr_t) x & 15)) return false;
    if (K % (8 * kRouteThreads) != 0 || K / 8 / kRouteThreads > kRouteMaxChunks) return false;
    if (w_is_bf16) {
        router_gemv_launch((cudaStream_t) stream, (const __nv_bfloat16 *) w, x, logits, K, E, T);
    } else {
        router_gemv_launch((cudaStream_t) stream, (const __half *) w, x, logits, K, E, T);
    }
    const int bt = (E + 31) / 32 * 32;
    softmax_topk_k<<<T, bt, (size_t) (E + 32) * sizeof(float) + 32 * sizeof(int), (cudaStream_t) stream>>>(
        logits, sel, weights, E, k);
    return true;
}

bool q4x_route_ring(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int E, int T, int k,
                    float * logits, int32_t * sel, float * weights, const int32_t * table, int32_t filler0,
                    int32_t * slots, int32_t * miss_bell, float * x_bell, int32_t * n_tok_bell, unsigned * seq_bell,
                    const unsigned * base, int layer, float * w_bell, void * plan, void * xq) {
    if (T < 1 || T > kMaxT || E > 1024 || k > E || k > 32 || K % 256 != 0 || ((uintptr_t) w & 15) || ((uintptr_t) x & 15)) return false;
    if (plan && (T * k > q4xplan::kMaxCols || (E + 31) / 32 * 32 < T * k)) return false;   // one thread per column
    if (xq && (((uintptr_t) xq & 3) || K % 32 != 0)) return false;
    if (K % (8 * kRouteThreads) != 0 || K / 8 / kRouteThreads > kRouteMaxChunks) return false;
    if (w_is_bf16) {
        router_gemv_launch((cudaStream_t) stream, (const __nv_bfloat16 *) w, x, logits, K, E, T);
    } else {
        router_gemv_launch((cudaStream_t) stream, (const __half *) w, x, logits, K, E, T);
    }
    const int bt = (E + 31) / 32 * 32;
    static const bool tk2 = getenv("AI_Q4X_TOPK2") && atoi(getenv("AI_Q4X_TOPK2")) != 0;
    const size_t smb = (size_t) (E + 32) * sizeof(float) + 32 * sizeof(int);
    if (tk2) {
        softmax_topk_ring_k<true><<<T, bt, smb, (cudaStream_t) stream>>>(
            logits, sel, weights, E, k, table, filler0, slots, miss_bell, x, (int64_t) K, x_bell, n_tok_bell,
            (volatile unsigned *) seq_bell, base, layer, w_bell, (q4xplan::Plan *) plan, (char *) xq);
    } else {
        softmax_topk_ring_k<false><<<T, bt, smb, (cudaStream_t) stream>>>(
            logits, sel, weights, E, k, table, filler0, slots, miss_bell, x, (int64_t) K, x_bell, n_tok_bell,
            (volatile unsigned *) seq_bell, base, layer, w_bell, (q4xplan::Plan *) plan, (char *) xq);
    }
    // [AI_Q4X_TOPK_CHECK=1] the reference top-k into device shadows, compared (debug: extra launches)
    static const bool tk_chk = getenv("AI_Q4X_TOPK_CHECK") && atoi(getenv("AI_Q4X_TOPK_CHECK")) != 0;
    if (tk_chk) {
        static int32_t * rs = nullptr;
        static float *   rw = nullptr;
        if (!rs && (cudaGetSymbolAddress((void **) &rs, g_tk_sel) != cudaSuccess ||   // device globals: no allocation
                    cudaGetSymbolAddress((void **) &rw, g_tk_w) != cudaSuccess)) {    // inside a capture
            (void) cudaGetLastError();
            rs = nullptr;
            rw = nullptr;
        }
        if (rs && rw) {
            softmax_topk_k<<<T, bt, smb, (cudaStream_t) stream>>>(logits, rs, rw, E, k);
            topk_check_k<<<1, 64, 0, (cudaStream_t) stream>>>(sel, weights, rs, rw, T * k);
        }
    }
    return true;
}

// [item 3] the router GEMV alone, for any small BF16/F16 projection of T <= 8 contiguous rows (fp32 math, no
// activation rounding, one launch instead of cutlass + 2 converts): out[t * N + n] = W[n, :K] . x[t, :K]
bool q4x_gemv16(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int N, int T, float * out) {
    if (T < 1 || T > kMaxT || K % (8 * kRouteThreads) != 0 || K / 8 / kRouteThreads > kRouteMaxChunks ||
        ((uintptr_t) w & 15) || ((uintptr_t) x & 15)) return false;
    if (w_is_bf16) {
        router_gemv_launch((cudaStream_t) stream, (const __nv_bfloat16 *) w, x, out, K, N, T);
    } else {
        router_gemv_launch((cudaStream_t) stream, (const __half *) w, x, out, K, N, T);
    }
    return true;
}

// Draft capture: chain one MTP draft step to the next on the device: tok = map ? map[idx] : idx; the next step's input
// id and the step's (token, probability) result slot. One thread: 3 scalar stores.
__global__ static void draft_next_k(const int32_t * idx, const float * p, const int32_t * map, int32_t * ids_next,
                                    int32_t * out_tok, float * out_p) {
    const int32_t i = *idx, t = map ? map[i] : i;
    if (ids_next) *ids_next = t;
    *out_tok = t;
    *out_p   = *p;
}
void q4x_draft_next(void * stream, const int32_t * idx, const float * p, const int32_t * map, int32_t * ids_next,
                    int32_t * out_tok, float * out_p) {
    draft_next_k<<<1, 1, 0, (cudaStream_t) stream>>>(idx, p, map, ids_next, out_tok, out_p);
}

// [AI_Q4X_DRAFT_ARGMAX_NATIVE] a draft step's top-1 and its softmax probability p = 1 / sum_j exp(l_j - l_max), then
// draft_next's three stores, in two launches: blocks keep an online (max, sum exp, first argmax) over a strided slice,
// one block merges them. Replaces ggml's argmax (ONE block over the 131072 draft rows: 20 us a step on the 5090) +
// soft_max + get_rows + draft_next. The token is the lowest index of the maximum (ggml's tie order is unspecified);
// p differs from the ggml chain only by summation order.
constexpr int kAmThreads = 256;
__device__ __forceinline__ void am_merge(float & m, float & s, int & i, float m2, float s2, int i2) {
    if (s2 == 0.0f) return;
    if (s == 0.0f) { m = m2; s = s2; i = i2; return; }
    if (m2 > m) { s = s * expf(m - m2) + s2; m = m2; i = i2; }
    else { s += s2 * expf(m2 - m); if (m2 == m && i2 < i) i = i2; }
}
__device__ __forceinline__ void am_block(float & m, float & s, int & i) {
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
        const int   i2 = __shfl_xor_sync(0xffffffffu, i, o);
        am_merge(m, s, i, m2, s2, i2);
    }
    __shared__ float sm[kAmThreads / 32], ss[kAmThreads / 32];
    __shared__ int   si[kAmThreads / 32];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) { sm[w] = m; ss[w] = s; si[w] = i; }
    __syncthreads();
    if (w == 0) {
        m = lane < kAmThreads / 32 ? sm[lane] : -INFINITY;
        s = lane < kAmThreads / 32 ? ss[lane] : 0.0f;
        i = lane < kAmThreads / 32 ? si[lane] : 0x7fffffff;
        for (int o = 16; o > 0; o >>= 1) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m, o), s2 = __shfl_xor_sync(0xffffffffu, s, o);
            const int   i2 = __shfl_xor_sync(0xffffffffu, i, o);
            am_merge(m, s, i, m2, s2, i2);
        }
    }
}
__global__ static void __launch_bounds__(kAmThreads) am_part_k(const float * __restrict__ x, int64_t V, float * __restrict__ part) {
    float m = -INFINITY, s = 0.0f;
    int   i = 0x7fffffff;
    for (int64_t c = (int64_t) blockIdx.x * kAmThreads + threadIdx.x; c < V; c += (int64_t) gridDim.x * kAmThreads) {
        const float v = x[c];
        if (v > m) { s = (s == 0.0f ? 0.0f : s * expf(m - v)) + 1.0f; m = v; i = (int) c; }
        else       { s += expf(v - m); }   // equal: the earlier (lower) index stays
    }
    am_block(m, s, i);
    if (threadIdx.x == 0) {
        part[blockIdx.x] = m;
        part[gridDim.x + blockIdx.x] = s;
        ((int *) part)[2 * gridDim.x + blockIdx.x] = i;
    }
}
__global__ static void __launch_bounds__(kAmThreads) am_final_k(const float * __restrict__ part, int nb, const int32_t * map,
                                                               int32_t * ids_next, int32_t * out_tok, float * out_p) {
    float m = -INFINITY, s = 0.0f;
    int   i = 0x7fffffff;
    for (int b = threadIdx.x; b < nb; b += kAmThreads) am_merge(m, s, i, part[b], part[nb + b], ((const int *) part)[2 * nb + b]);
    am_block(m, s, i);
    if (threadIdx.x == 0) {
        const int32_t t = map ? map[i] : i;
        if (ids_next) *ids_next = t;
        *out_tok = t;
        *out_p   = 1.0f / s;
    }
}
int q4x_draft_argmax_blocks(int64_t V) {
    const int64_t nb = (V + 4 * kAmThreads - 1) / (4 * kAmThreads);
    return (int) (nb < 1 ? 1 : nb > 1024 ? 1024 : nb);
}
void q4x_draft_argmax_next(void * stream, const float * logits, int64_t V, float * part, const int32_t * map,
                           int32_t * ids_next, int32_t * out_tok, float * out_p) {
    const int nb = q4x_draft_argmax_blocks(V);
    am_part_k<<<nb, kAmThreads, 0, (cudaStream_t) stream>>>(logits, V, part);
    am_final_k<<<1, kAmThreads, 0, (cudaStream_t) stream>>>(part, nb, map, ids_next, out_tok, out_p);
}

// [prefill] causal mask of one sub-chunk on the device: m[i * n_kv + j] = j <= p0 + i ? 0 : -inf (f16)
__global__ static void causal_mask_k(uint16_t * m, int64_t n_kv, int64_t T, int64_t p0) {
    const int64_t n = n_kv * T;
    for (int64_t x = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; x < n; x += (int64_t) gridDim.x * blockDim.x) {
        const int64_t i = x / n_kv, j = x - i * n_kv;
        m[x] = j <= p0 + i ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;
    }
}
// Prefill plans: a copy by a kernel from mapped pinned host memory. A cudaMemcpyAsync H2D would queue on the copy engine
// behind the expert stream's queued blobs (a layer's ~1200 copies), so the expert kernels would wait for their own plan.
__global__ static void copy_i32_k(const int32_t * __restrict__ src, int32_t * __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
void q4x_copy_i32(void * stream, const int32_t * src, int32_t * dst, int64_t n) {
    if (n <= 0) return;
    const int blocks = (int) (n < 256 * 256 ? (n + 255) / 256 : 256);
    copy_i32_k<<<blocks, 256, 0, (cudaStream_t) stream>>>(src, dst, n);
}

// [AI_Q4X_PLE_GATE] each block's thread 0 spins until the host's mapped flag reaches this round's sequence (*want, a
// device word uploaded before the launch; wrap-safe compare), then the block copies its slice of the rows the host
// wrote into mapped memory (volatile loads: never a stale cached line of an earlier round)
__global__ static void ple_gate_k(const volatile unsigned * flag, const unsigned * want, const float * src, float * dst,
                                  int64_t n) {
    if (threadIdx.x == 0) {
        const unsigned t = *want;
        while ((int) (*flag - t) < 0) __nanosleep(256);
        __threadfence();
    }
    __syncthreads();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = __ldcv(src + i);
}
void q4x_ple_gate(void * stream, const unsigned * flag, const unsigned * want, const float * src, float * dst, int64_t n) {
    if (n <= 0) return;
    const int blocks = (int) (n < 32 * 256 ? (n + 255) / 256 : 32);
    ple_gate_k<<<blocks, 256, 0, (cudaStream_t) stream>>>(flag, want, src, dst, n);
}

// [MTP fill] the draft layer's mask is the causal one except column 0 (no H_{-1}): masked for every row past position 0
__global__ static void mtp_mask_col0_k(uint16_t * m, int64_t n_kv, int64_t T, int64_t p0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T && p0 + i >= 1) m[i * n_kv] = 0xFC00;
}
void q4x_mtp_mask_col0(void * stream, uint16_t * m, int64_t n_kv, int64_t T, int64_t p0) {
    if (T <= 0) return;
    mtp_mask_col0_k<<<(unsigned) ((T + 255) / 256), 256, 0, (cudaStream_t) stream>>>(m, n_kv, T, p0);
}

void q4x_causal_mask(void * stream, uint16_t * m, int64_t n_kv, int64_t T, int64_t p0) {
    const int64_t n      = n_kv * T;
    const int     blocks = (int) (n < 4096 * 256 ? (n + 255) / 256 : 4096);
    causal_mask_k<<<blocks, 256, 0, (cudaStream_t) stream>>>(m, n_kv, T, p0);
}

}  // namespace ai::kern
