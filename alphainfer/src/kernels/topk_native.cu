// SPDX-License-Identifier: Apache-2.0
// MoE top-8 routing as a native kernel (AI_TOPK_NATIVE), reading the router logits the add+norm launch made.
//
// Replaces ggml's topk_moe_cuda: softmax over 256 experts, then eight sequential warp-argmax rounds
// (5 dependent shuffle levels each), then the clamp-normalise.
//
// Same softmax as router_topk.cu (bitwise equal to ggml's). Selection: each lane sorts its own 8 values,
// then 5 xor-merge steps keep the top 8 of the union (bitonic merge of two sorted 8-lists). The order
// "higher value first, then lower expert" is a strict total order, so the top-8 list is unique and equals
// the argmax rounds' list element for element, ties and NaN (-> -FLT_MAX) included. The normalisation
// reproduces the reference's summation: each lane sums, in selection order, the selected values whose
// expert it owns (expert & 31), then the warp butterfly.
//
// Built -use_fast_math (as ggml's CUDA backend); __expf / __fdividef are also written explicitly.
#include "kernels/kernels.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

constexpr int TK_WS = 32, TK_NE = 256, TK_K = 8, TK_EPT = TK_NE / TK_WS;

__device__ __forceinline__ float tk_wsum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    }
    return x;
}
__device__ __forceinline__ float tk_wmax(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o, 32));
    }
    return x;
}
__device__ __forceinline__ bool tk_before(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// grid (ceil(T/4)), block (32, 4): one warp per token
__global__ void topk_native_k(const float * __restrict__ logits, int nrows, int T, int32_t * __restrict__ ids,
                              float * __restrict__ w, float clamp_val) {
    const int lane = threadIdx.x, t = blockIdx.x * blockDim.y + threadIdx.y;
    if (t >= T) {
        return;
    }
    const float * lg = logits + (long long) t * nrows;
    float         v[TK_EPT];
    int           x[TK_EPT];
#pragma unroll
    for (int i = 0; i < TK_EPT; ++i) {
        v[i] = __ldcg(lg + i * TK_WS + lane);
        x[i] = lane + i * TK_WS;
    }
    // softmax_warp_inplace, as in ggml's kernel
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < TK_EPT; ++i) {
        mx = fmaxf(mx, v[i]);
    }
    mx        = tk_wmax(mx);
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TK_EPT; ++i) {
        const float e = __expf(v[i] - mx);
        v[i]          = e;
        sum += e;
    }
    sum             = tk_wsum(sum);
    const float inv = __fdividef(1.0f, sum);
#pragma unroll
    for (int i = 0; i < TK_EPT; ++i) {
        v[i] *= inv;
    }
#pragma unroll
    for (int i = 0; i < TK_EPT; ++i) {
        if (__isnanf(v[i])) {
            v[i] = -FLT_MAX;
        }
    }
    // per-lane sort, descending (odd-even transposition)
#pragma unroll
    for (int p = 0; p < TK_EPT; ++p) {
#pragma unroll
        for (int i = p & 1; i + 1 < TK_EPT; i += 2) {
            if (tk_before(v[i + 1], x[i + 1], v[i], x[i])) {
                const float tv = v[i];
                v[i]           = v[i + 1];
                v[i + 1]       = tv;
                const int tx   = x[i];
                x[i]           = x[i + 1];
                x[i + 1]       = tx;
            }
        }
    }
    // 5 xor merge steps: top-8 of my list and my partner's
#pragma unroll
    for (int mask = 1; mask < TK_WS; mask <<= 1) {
        float pv[TK_EPT];
        int   px[TK_EPT];
#pragma unroll
        for (int i = 0; i < TK_EPT; ++i) {
            pv[i] = __shfl_xor_sync(0xffffffffu, v[i], mask, 32);
            px[i] = __shfl_xor_sync(0xffffffffu, x[i], mask, 32);
        }
#pragma unroll
        for (int i = 0; i < TK_EPT; ++i) {
            const float bv = pv[TK_EPT - 1 - i];
            const int   bx = px[TK_EPT - 1 - i];
            if (tk_before(bv, bx, v[i], x[i])) {
                v[i] = bv;
                x[i] = bx;
            }
        }
#pragma unroll
        for (int d = TK_EPT / 2; d > 0; d >>= 1) {
#pragma unroll
            for (int i = 0; i < TK_EPT; ++i) {
                if ((i & d) == 0) {
                    const int j = i + d;
                    if (tk_before(v[j], x[j], v[i], x[i])) {
                        const float tv = v[i];
                        v[i]           = v[j];
                        v[j]           = tv;
                        const int tx   = x[i];
                        x[i]           = x[j];
                        x[j]           = tx;
                    }
                }
            }
        }
    }
    // the reference's normalisation order
    float wt_sum = 0.f;
#pragma unroll
    for (int k = 0; k < TK_K; ++k) {
        if ((x[k] & (TK_WS - 1)) == lane) {
            wt_sum += v[k];
        }
    }
    wt_sum             = tk_wsum(wt_sum);
    wt_sum             = fmaxf(wt_sum, clamp_val);
    const float inv_ws = __fdividef(1.0f, wt_sum);
    if (lane < TK_K) {
        float mv = 0.f;
        int   mi = 0;
#pragma unroll
        for (int k = 0; k < TK_K; ++k) {
            if (k == lane) {
                mv = v[k];
                mi = x[k];
            }
        }
        ids[t * TK_K + lane] = mi;
        w[t * TK_K + lane]   = (mv * inv_ws) * 1.0f;
    }
}

__device__ unsigned long long g_tk_cmp[4];   // calls, ids differing, weights differing, values
__global__ void topk_cmp_k(const int32_t * ia, const float * wa, const int32_t * ib, long long ib_tok, const float * wb,
                           long long wb_tok, int T) {
    unsigned long long di = 0, dw = 0;
    for (int i = threadIdx.x; i < T * TK_K; i += blockDim.x) {
        const int t = i / TK_K, k = i % TK_K;
        di += ia[i] != ib[t * ib_tok + k];
        dw += __float_as_uint(wa[i]) != __float_as_uint(wb[t * wb_tok + k]);
    }
    atomicAdd(&g_tk_cmp[1], di);
    atomicAdd(&g_tk_cmp[2], dw);
    if (threadIdx.x == 0) {
        atomicAdd(&g_tk_cmp[0], 1ull);
        atomicAdd(&g_tk_cmp[3], (unsigned long long) (T * TK_K));
    }
}

}   // namespace

namespace ai::kern {

bool topk_native_ok(int64_t n_expert, int64_t k_used, int64_t T) {
    return n_expert == TK_NE && k_used == TK_K && T >= 1 && T <= 64;
}

bool topk_native(void * stream, const float * logits, int64_t nrows, int64_t T, int32_t * ids, float * w,
                 float clamp_val) {
    if (!topk_native_ok(TK_NE, TK_K, T) || nrows < TK_NE) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[topk-native] LAUNCHED: top-8 by per-lane sort + 5 xor merges\n");
    }
    topk_native_k<<<(unsigned) ((T + 3) / 4), dim3(32, 4), 0, (cudaStream_t) stream>>>(logits, (int) nrows, (int) T,
                                                                                     ids, w, clamp_val);
    return true;
}

void topk_native_compare(void * stream, const int32_t * ia, const float * wa, const int32_t * ib, int64_t ib_tok,
                         const float * wb, int64_t wb_tok, int64_t T) {
    static bool armed = false;
    if (!armed) {
        armed = true;
        atexit([]() {
            unsigned long long h[4] = { 0, 0, 0, 0 };
            cudaMemcpyFromSymbol(h, g_tk_cmp, sizeof(h));
            fprintf(stderr, "[topk-native-compare] calls %llu | values %llu, ids differing %llu, weights differing %llu\n",
                    h[0], h[3], h[1], h[2]);
        });
    }
    topk_cmp_k<<<1, 256, 0, (cudaStream_t) stream>>>(ia, wa, ib, (long long) ib_tok, wb, (long long) wb_tok, (int) T);
}

int device_cc_major() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        return 0;
    }
    return major;
}

int device_cc() {
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        return 0;
    }
    return major * 10 + minor;
}

}   // namespace ai::kern
