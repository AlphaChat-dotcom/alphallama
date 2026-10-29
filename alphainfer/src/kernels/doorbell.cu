// SPDX-License-Identifier: Apache-2.0
// Generic doorbell offload kernels; see kernels/doorbell.h and offload/doorbell_moe.h.
#include "kernels/doorbell.h"

#include <cuda_runtime.h>
#include <cstdint>

__global__ static void door_resolve_ring_k(const int32_t * table, const int32_t * sel, const float * w, const float * x,
                                           int64_t N, int k, int64_t T, int32_t filler0, int32_t * slots,
                                           float * w_masked, int32_t * miss_bell, float * w_bell, float * x_bell,
                                           int32_t * n_tok_bell, volatile unsigned * seq_bell, unsigned * counter) {
    const int64_t kt = (int64_t) k * T;
    for (int64_t i = threadIdx.x; i < kt; i += blockDim.x) {
        const int32_t e = sel[i];
        const int32_t s = table[e];
        slots[i]     = s >= 0 ? s : filler0 + (int32_t) (i % k);
        w_masked[i]  = s >= 0 ? w[i] : 0.0f;
        miss_bell[i] = s >= 0 ? -(e + 2) : e;
        w_bell[i]    = w[i];
    }
    for (int64_t i = threadIdx.x; i < N * T; i += blockDim.x) {
        x_bell[i] = x[i];
    }
    // every writing thread fences at system scope before the barrier, so all of the bell is visible before seq
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned c = *counter + 1;
        *counter    = c;
        *n_tok_bell = (int32_t) T;
        __threadfence_system();
        *seq_bell = c;
        __threadfence_system();
    }
}

__global__ static void door_wait_add_k(volatile const unsigned * done_bell, const unsigned * counter,
                                       const int32_t * slots, int32_t filler0, const float * yp_bell, float * out,
                                       int64_t N, int k) {
    const int64_t t = blockIdx.x;
    __shared__ int any;
    if (threadIdx.x == 0) {
        any = 0;
        for (int j = 0; j < k; ++j) any |= slots[t * k + j] >= filler0;
        if (any) {
            const unsigned want = *counter;
            while (*done_bell < want) {
                __nanosleep(64);
            }
        }
    }
    __syncthreads();
    if (!any) return;
    __threadfence_system();
    const float * src = yp_bell + t * N;
    float *       dst = out + t * N;
    // all of a thread's host loads issued before any is used: one PCIe round trip per batch, not one per element
    constexpr int kV = 4;
    if ((N % 4) == 0 && ((uintptr_t) src & 15) == 0 && ((uintptr_t) dst & 15) == 0) {
        const int64_t n4   = N / 4;
        const float4 * s4  = (const float4 *) src;
        float4 *       d4  = (float4 *) dst;
        for (int64_t b = 0; b < n4; b += (int64_t) kV * blockDim.x) {
            float4 v[kV];
#pragma unroll
            for (int i = 0; i < kV; ++i) {
                const int64_t idx = b + (int64_t) i * blockDim.x + threadIdx.x;
                if (idx < n4) v[i] = __ldcv(s4 + idx);
            }
#pragma unroll
            for (int i = 0; i < kV; ++i) {
                const int64_t idx = b + (int64_t) i * blockDim.x + threadIdx.x;
                if (idx < n4) {
                    float4 o = d4[idx];
                    o.x += v[i].x; o.y += v[i].y; o.z += v[i].z; o.w += v[i].w;
                    d4[idx] = o;
                }
            }
        }
    } else {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) dst[n] += ((const volatile float *) src)[n];
    }
}

namespace ai::kern {

void door_resolve_ring(void * stream, const int32_t * table, const int32_t * sel, const float * w, const float * x,
                       int64_t N, int k, int64_t T, int32_t filler0, int32_t * slots, float * w_masked,
                       int32_t * miss_bell, float * w_bell, float * x_bell, int32_t * n_tok_bell, unsigned * seq_bell,
                       unsigned * counter) {
    door_resolve_ring_k<<<1, 512, 0, (cudaStream_t) stream>>>(table, sel, w, x, N, k, T, filler0, slots, w_masked,
                                                               miss_bell, w_bell, x_bell, n_tok_bell, seq_bell, counter);
}

void door_wait_add(void * stream, const unsigned * done_bell, const unsigned * counter, const int32_t * slots,
                   int32_t filler0, const float * yp_bell, float * out, int64_t N, int k, int64_t T) {
    door_wait_add_k<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(done_bell, counter, slots, filler0, yp_bell, out,
                                                                     N, k);
}

}  // namespace ai::kern
