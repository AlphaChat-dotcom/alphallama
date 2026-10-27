// Router GEMV + top-k in one launch (AI_ROUTER_TOPK). Bit-exact against the two ggml kernels it replaces:
//
//   phase 1  every block = one router row: ggml's mul_mat_vec_f<half, half, T, 256> verbatim -- half2
//            accumulation `sumh2 += x2 * make_half2(y)` over col2 = tid, tid+256, ...; low + high; warp
//            butterfly; buf_iw[warp] (zero-initialised 32 slots); a second butterfly; thread t writes token t.
//   phase 2  the last block to finish (one atomic, no grid barrier) runs topk_moe_cuda per token, one warp
//            per token, verbatim: softmax over the 256 experts (fast-math: __expf, __fdividef, as ggml is
//            built), NaN -> -FLT_MAX, k rounds of warp argmax (ties -> lowest expert), weights
//            normalised by 1 / max(sum, clamp) and scaled by 1.0.
//
// Both kernels are latency-bound and run once per MoE layer. The top-k's work is tiny; its cost is the launch and
// its place on the critical path. Folding it into the GEMV's last block removes one hop per layer.
#include "kernels/kernels.h"

#include <cfloat>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define RT_BS 256
#define RT_WS 32

static __device__ unsigned int g_rt_done = 0;

static __device__ __forceinline__ float rt_warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    }
    return x;
}
static __device__ __forceinline__ float rt_warp_max(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    }
    return x;
}

template <int T, int NE, int K>
__global__ static void __launch_bounds__(RT_BS) router_topk_k(const __half * __restrict__ W, const float * __restrict__ y,
                                                             float * logits, int32_t * __restrict__ ids,
                                                             float * __restrict__ weights, int ncols2, int nrows,
                                                             int stride_col_y2, float clamp_val) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float buf_iw[RT_WS];
    __shared__ bool  s_last;

    // phase 1: mul_mat_vec_f<half, half, T, 256>, one row
    if (tid < RT_WS) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();
    const half2 *  x2 = (const half2 *) (W + (long long) row * ncols2 * 2);
    const float2 * y2 = (const float2 *) y;
    float sumf[T];
    {
        half2 sumh2[T];
#pragma unroll
        for (int j = 0; j < T; ++j) {
            sumh2[j] = make_half2(0.0f, 0.0f);
        }
        for (int col2 = tid; col2 < ncols2; col2 += RT_BS) {
            const half2 tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 tmpy = y2[j * stride_col_y2 + col2];
                sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);
            }
        }
#pragma unroll
        for (int j = 0; j < T; ++j) {
            sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
        sumf[j] = rt_warp_sum(sumf[j]);
        buf_iw[tid / RT_WS] = sumf[j];
        __syncthreads();
        if (tid < RT_WS) {
            sumf[j] = buf_iw[tid];
            sumf[j] = rt_warp_sum(sumf[j]);
        }
        __syncthreads();
    }
    if (tid < T) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < T; ++j) {
            if (j == tid) {
                v = sumf[j];
            }
        }
        logits[(long long) tid * nrows + row] = v;
    }

    // is this the last block?
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = atomicAdd(&g_rt_done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // phase 2: topk_moe_cuda, one warp per token
    const int warp = tid / RT_WS, lane = tid % RT_WS;
    if (warp < T) {
        constexpr int EPT = NE / RT_WS;   // experts per thread
        const float * lg = logits + (long long) warp * nrows;
        float wt[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            wt[i] = __ldcg(lg + i * RT_WS + lane);
        }
        // softmax_warp_inplace<EPT, false>
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            mx = fmaxf(mx, wt[i]);
        }
        mx = rt_warp_max(mx);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            const float v = __expf(wt[i] - mx);
            wt[i]         = v;
            sum += v;
        }
        sum = rt_warp_sum(sum);
        const float inv_sum = __fdividef(1.0f, sum);
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            wt[i] *= inv_sum;
        }
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            if (__isnanf(wt[i])) {
                wt[i] = -FLT_MAX;
            }
        }
        float wt_sum = 0.f;
        float output_weights[EPT];
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            output_weights[i] = 0.f;
        }
        for (int k = 0; k < K; ++k) {
            float max_val    = wt[0];
            int   max_expert = lane;
#pragma unroll
            for (int i = 1; i < EPT; ++i) {
                const int expert = lane + i * RT_WS;
                if (wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }
#pragma unroll
            for (int mask = RT_WS / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, RT_WS);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, RT_WS);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }
            if ((max_expert & (RT_WS - 1)) == lane) {
                wt[max_expert / RT_WS] = -INFINITY;
            }
            if ((k & (RT_WS - 1)) == lane) {
                output_weights[k / RT_WS] = max_val;
            }
            if ((max_expert & (RT_WS - 1)) == lane) {
                ids[warp * K + k] = max_expert;
                wt_sum += max_val;
            }
        }
        wt_sum              = rt_warp_sum(wt_sum);
        wt_sum              = fmaxf(wt_sum, clamp_val);
        const float inv_ws  = __fdividef(1.0f, wt_sum);
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            output_weights[i] *= inv_ws;
        }
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
            const int idx = i * RT_WS + lane;
            if (idx < K) {
                weights[warp * K + idx] = output_weights[i] * 1.0f;
            }
        }
    }
    if (tid == 0) {
        g_rt_done = 0;
    }
}

// AI_ROUTER_TOPK=2 instrument: bitwise comparison of the fused outputs against the graph's, per call.
static __device__ unsigned long long g_rtc[6];   // calls, logit diffs, id diffs, weight diffs, first-bad T, max T seen
__global__ static void rt_compare_k(const float * a_l, const float * b_l, long long b_l_tok, int nl, const int32_t * a_i,
                                    const int32_t * b_i, long long b_i_tok, const float * a_w, const float * b_w, int k, int T) {
    unsigned long long dl = 0, di = 0, dw = 0;
    for (int t = 0; t < T; ++t) {
        for (int i = threadIdx.x; i < nl; i += blockDim.x) {
            dl += __float_as_uint(a_l[(long long) t * nl + i]) != __float_as_uint(b_l[t * b_l_tok + i]);
        }
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            di += a_i[t * k + i] != b_i[t * b_i_tok + i];
            dw += __float_as_uint(a_w[t * k + i]) != __float_as_uint(b_w[t * k + i]);
        }
    }
    atomicAdd(&g_rtc[1], dl); atomicAdd(&g_rtc[2], di); atomicAdd(&g_rtc[3], dw);
    if (threadIdx.x == 0) {
        atomicAdd(&g_rtc[0], 1ull);
        atomicMax(&g_rtc[5], (unsigned long long) T);
    }
    __syncthreads();
    if (threadIdx.x == 0 && (dl | di | dw) != 0) {
        atomicMax(&g_rtc[4], (unsigned long long) T);
    }
}

namespace ai::kern {

void router_topk_compare(void * stream, const float * a_l, const float * b_l, int64_t b_l_tok, int64_t nl,
                         const int32_t * a_i, const int32_t * b_i, int64_t b_i_tok, const float * a_w, const float * b_w,
                         int64_t k, int64_t T) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[6] = { 0, 0, 0, 0, 0, 0 };
            cudaMemcpyFromSymbol(h, g_rtc, sizeof(h));
            fprintf(stderr, "[router-topk-compare] calls %llu | differing logits %llu, ids %llu, weights %llu | max T %llu, a T with diffs %llu\n",
                    h[0], h[1], h[2], h[3], h[5], h[4]);
        });
    }
    rt_compare_k<<<1, 256, 0, (cudaStream_t) stream>>>(a_l, b_l, (long long) b_l_tok, (int) nl, a_i, b_i,
                                                       (long long) b_i_tok, a_w, b_w, (int) k, (int) T);
}

bool router_topk(void * stream, const void * W, const float * y, float * logits, int32_t * ids, float * weights,
                 int64_t K_in, int64_t nrows, int64_t n_expert, int64_t k_used, int64_t T, int64_t y_tok, float clamp_val) {
    if (n_expert != 256 || k_used != 8 || K_in % 2 != 0 || y_tok % 2 != 0 || T < 1 || T > 8 || nrows < n_expert) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[router-topk] LAUNCHED (router GEMV + top-k in one kernel, %lld rows)\n", (long long) nrows);
    }
    const dim3   grid((unsigned) nrows);
    cudaStream_t st = (cudaStream_t) stream;
    const __half * w = (const __half *) W;
    const int    c2 = (int) (K_in / 2), r = (int) nrows, sy = (int) (y_tok / 2);
#define RT_LAUNCH(TT) router_topk_k<TT, 256, 8><<<grid, RT_BS, 0, st>>>(w, y, logits, ids, weights, c2, r, sy, clamp_val)
    switch (T) {
        case 1: RT_LAUNCH(1); break;
        case 2: RT_LAUNCH(2); break;
        case 3: RT_LAUNCH(3); break;
        case 4: RT_LAUNCH(4); break;
        case 5: RT_LAUNCH(5); break;
        case 6: RT_LAUNCH(6); break;
        case 7: RT_LAUNCH(7); break;
        default: RT_LAUNCH(8); break;
    }
#undef RT_LAUNCH
    return true;
}

}   // namespace ai::kern
