// Fused qwen4exp MoE routing for T <= 16 rows -- the router GEMV straight from BF16/F16 weights, then
// softmax over the experts, top-k and the renormalised weights, in two launches (the ggml chain takes ~10: BF16 GEMM via
// cutlass + 2 activation converts + softmax + 512-wide argsort + get_rows + sum_rows + clamp + div + cont).
//
// Not bit-identical to that chain: ggml rounds the activations to BF16 before its GEMM and its argsort breaks ties
// differently, so this is a fast-mode path; validate any change to it by paired NLL like any routing change.
#include "kernels/kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

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

// One block (E threads, E <= 1024) per row t: softmax over E, the top-k (descending; ties -> lower index), weights =
// p / max(sum of the top-k p, 6.1e-5) (qwen4exp's renormalisation, as ops.cpp's moe_route).
__global__ void softmax_topk_k(const float * __restrict__ logits, int32_t * __restrict__ sel, float * __restrict__ wout,
                               int E, int k) {
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

}  // namespace

namespace ai::kern {

bool q4x_route(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int E, int T, int k,
               float * logits, int32_t * sel, float * weights) {
    if (T < 1 || T > kMaxT || E > 1024 || k > E || K % 256 != 0 || ((uintptr_t) w & 15) || ((uintptr_t) x & 15)) return false;
    if (K % (8 * kRouteThreads) != 0 || K / 8 / kRouteThreads > kRouteMaxChunks) return false;
    if (w_is_bf16) {
        router_gemv_k<__nv_bfloat16><<<E, kRouteThreads, 0, (cudaStream_t) stream>>>(
            (const __nv_bfloat16 *) w, x, logits, K, E, T);
    } else {
        router_gemv_k<__half><<<E, kRouteThreads, 0, (cudaStream_t) stream>>>((const __half *) w, x, logits, K, E, T);
    }
    const int bt = (E + 31) / 32 * 32;
    softmax_topk_k<<<T, bt, (size_t) (E + 32) * sizeof(float) + 32 * sizeof(int), (cudaStream_t) stream>>>(
        logits, sel, weights, E, k);
    return true;
}

// The router GEMV alone, for any small BF16/F16 projection of T <= 16 contiguous rows (fp32 math, no
// activation rounding, one launch instead of cutlass + 2 converts): out[t * N + n] = W[n, :K] . x[t, :K]
bool q4x_gemv16(void * stream, const void * w, bool w_is_bf16, const float * x, int K, int N, int T, float * out) {
    if (T < 1 || T > kMaxT || K % (8 * kRouteThreads) != 0 || K / 8 / kRouteThreads > kRouteMaxChunks ||
        ((uintptr_t) w & 15) || ((uintptr_t) x & 15)) return false;
    if (w_is_bf16) {
        router_gemv_k<__nv_bfloat16><<<N, kRouteThreads, 0, (cudaStream_t) stream>>>((const __nv_bfloat16 *) w, x, out, K, N, T);
    } else {
        router_gemv_k<__half><<<N, kRouteThreads, 0, (cudaStream_t) stream>>>((const __half *) w, x, out, K, N, T);
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

// Prefill causal mask of one sub-chunk on the device: m[i * n_kv + j] = j <= p0 + i ? 0 : -inf (f16)
__global__ static void causal_mask_k(uint16_t * m, int64_t n_kv, int64_t T, int64_t p0) {
    const int64_t n = n_kv * T;
    for (int64_t x = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; x < n; x += (int64_t) gridDim.x * blockDim.x) {
        const int64_t i = x / n_kv, j = x - i * n_kv;
        m[x] = j <= p0 + i ? (uint16_t) 0x0000 : (uint16_t) 0xFC00;
    }
}
void q4x_causal_mask(void * stream, uint16_t * m, int64_t n_kv, int64_t T, int64_t p0) {
    const int64_t n      = n_kv * T;
    const int     blocks = (int) (n < 4096 * 256 ? (n + 255) / 256 : 4096);
    causal_mask_k<<<blocks, 256, 0, (cudaStream_t) stream>>>(m, n_kv, T, p0);
}

}  // namespace ai::kern
