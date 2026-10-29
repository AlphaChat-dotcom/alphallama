// SPDX-License-Identifier: Apache-2.0
// The device half of the offloaded MoE: resolve routed experts against a device residency table, ring the layer's
// doorbell for the CPU with the misses, and later wait for the CPU's rows and splice them in. No host
// synchronisation: the CPU worker polls mapped memory (src/offload/miss_pool.cpp) and these kernels spin on it, so the
// whole window can be one captured graph (a handshake costs a few us).
//
// Window sequence numbers: seq(layer) = *base + layer + 1, and *base advances by the layer count at the end of each
// window (offload_window_end), so a replayed graph produces fresh numbers without host involvement.
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Debug: AI_Q4X_WAIT_STATS=1 records, per layer, the ns from the doorbell ring (resolve) to the end of the wait, and the ns
// the wait actually spun; block 0 of each wait only. Read at exit.
__device__ int                g_ws_on;
// GPU miss share: a GPU-taken missed expert e is rung as -(e + 2) - kGpuTag (miss_pool.h decodes it)
constexpr int32_t kGpuTag = 1 << 20;
// Debug: AI_Q4X_HDR_TOUCH enables the mapped bell-header stamp writes without the stats (bit 0: the ring stamp, bit 1:
// the wait start/end stamps), to isolate whether those writes themselves change the handoff speed.
__device__ int                g_touch;
__device__ unsigned long long g_ws_ring[128];
__device__ unsigned long long g_ws[4];   // waits, waits that spun, spin ns, ring -> wait-end ns
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// One block. slots[t*k+j] = table[sel] when resident, else filler slot j (k reserved, always-valid slots, distinct
// within a row and never holding a routed expert). miss[t*k+j] = expert id (missed) or -(id+2) (resident): the host
// admission policy counts both. x copied to the mapped bell.
__global__ static void resolve_ring_k(const int32_t * table, const int32_t * sel, const float * x, int64_t N, int k,
                                      int64_t T, int32_t filler0, int32_t * slots, int32_t * miss_bell, float * x_bell,
                                      int32_t * n_tok_bell, volatile unsigned * seq_bell, const unsigned * base, int layer,
                                      const float * w, float * w_bell, int gpu_m, int32_t gpu0, int tail_rank) {
    const int64_t kt = (int64_t) k * T;
    for (int64_t i = threadIdx.x; i < kt; i += blockDim.x) {
        const int32_t e = sel[i];
        const int32_t s = table[e];
        slots[i]     = s >= 0 ? s : filler0 + (int32_t) (i % k);
        miss_bell[i] = s >= 0 ? -(e + 2) : e;   // hit: -(e+2) (<= -2, so "< 0 = resident" holds); host decodes both
        if (w_bell) w_bell[i] = w[i];
    }
    // GPU miss share: the first gpu_m distinct missed experts (column order, routing rank < tail_rank) are computed
    // by the GPU straight from the mapped RAM tier (q4x_experts with the host bank): their columns get slot gpu0 + e
    // (>= filler0, so the resident pass skips them; the wait neither zeroes nor waits for them) and the CPU ring sees
    // kGpuTag-encoded ids (< 0: skipped by the pool; the host admission still counts them as misses).
    if (gpu_m > 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t taken[16];
            int     nt = 0;
            for (int64_t i = 0; i < kt; ++i) {
                const int32_t e = miss_bell[i];
                if (e < 0) continue;
                bool have = false;
                for (int q = 0; q < nt; ++q) have = have || taken[q] == e;
                if (!have) {
                    if (nt >= gpu_m || nt >= 16 || (int) (i % k) >= tail_rank) continue;
                    taken[nt++] = e;
                }
                slots[i]     = gpu0 + e;
                miss_bell[i] = -(e + 2) - kGpuTag;
            }
        }
        __syncthreads();
    }
    for (int64_t i = threadIdx.x; i < N * T; i += blockDim.x) {
        x_bell[i] = x[i];
    }
    // Every writing thread fences at system scope before the barrier: thread 0's fence after a __syncthreads does
    // not make the other threads' mapped-memory (PCIe) writes visible to the CPU before seq. Delaying the CPU's
    // answers (AI_Q4X_POOL_DELAY_US) exposes the race as run-to-run different outputs.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *n_tok_bell = (int32_t) T;
        __threadfence_system();
        if (g_ws_on || (g_touch & 1)) {
            const unsigned long long tr = gtimer();
            if (g_ws_on) g_ws_ring[layer & 127] = tr;
            // AI_Q4X_HANDOFF_STATS: the ring time into the bell header (byte 16) for the host's notice-latency log
            *(volatile unsigned long long *) ((char *) seq_bell + 16) = tr;
            __threadfence_system();
        }
        *seq_bell = *base + (unsigned) layer + 1;
        __threadfence_system();
    }
}

// One block per (j, t) column group: wait until the CPU answered this window's ring, then overwrite the missed
// columns of exps [N, k, T] with the CPU's unweighted expert outputs y [N, k*T] (column t*k + j).
__global__ static void wait_fill_k(volatile const unsigned * done_bell, const unsigned * base, int layer,
                                   const int32_t * miss_bell, const float * y_bell, float * exps, int64_t N, int k,
                                   int64_t T) {
    const unsigned want = *base + (unsigned) layer + 1;
    if (threadIdx.x == 0) {
        while (*done_bell < want) {
            __nanosleep(64);
        }
    }
    __syncthreads();
    __threadfence_system();
    const int64_t c = blockIdx.x;   // column t*k + j
    if (c >= (int64_t) k * T || ((volatile const int32_t *) miss_bell)[c] < 0) return;
    const volatile float * src = y_bell + c * N;
    float *                dst = exps + c * N;
    for (int64_t n = threadIdx.x; n < N; n += blockDim.x) dst[n] = src[n];
}

// AI_Q4X_PARTIAL: one block per token. Wait for the CPU, zero the token's missed expert columns (their filler-slot
// outputs) and add the CPU's weighted sum over them into the shared-expert output, so moe_weighted_sum yields
// shared + cpu + sum_resident. The device reads N floats per token from host memory instead of N per missed column
// (SM reads of mapped memory are slow, a few GB/s, and get slower with more concurrent readers).
// Fast mode: the CPU's sum is added before the resident ones, a different float order from moe_weighted_sum's.
__global__ static void wait_partial_k(volatile const unsigned * done_bell, const unsigned * base, int layer,
                                      const int32_t * slots, int32_t filler0, int32_t gpu0, const float * yp_bell,
                                      float * exps, float * shared, int64_t N, int k) {
    // missed <=> the resolve kernel gave the column a filler slot (device memory: no host round trip per check; ten
    // dependent mapped-memory reads by one thread would cost tens of us)
    const int64_t t = blockIdx.x;
    __shared__ int any;
    if (threadIdx.x == 0) {
        any = 0;
        for (int j = 0; j < k; ++j) any |= slots[t * k + j] >= filler0 && slots[t * k + j] < gpu0;   // CPU columns only
        if (any) {
            const unsigned want = *base + (unsigned) layer + 1;
            const bool     st   = (g_ws_on || (g_touch & 2)) && t == 0;
            const unsigned long long t0 = st ? gtimer() : 0;
            bool spun = false;
            while (*done_bell < want) {
                spun = true;
                __nanosleep(64);
            }
            if (st) {
                const unsigned long long t1 = gtimer();
                // AI_Q4X_HANDOFF_STATS: the wait's start / end into the bell header (bytes 32 / 24): the host pairs
                // them with its done write
                *(volatile unsigned long long *) ((char *) done_bell + 28) = t0;
                __threadfence_system();
                *(volatile unsigned long long *) ((char *) done_bell + 20) = t1;
                if (g_ws_on) {
                    atomicAdd(&g_ws[0], 1ull);
                    if (spun) atomicAdd(&g_ws[1], 1ull);
                    atomicAdd(&g_ws[2], t1 - t0);
                    atomicAdd(&g_ws[3], t1 - g_ws_ring[layer & 127]);
                }
            }
        }
    }
    __syncthreads();
    if (!any) return;
    __threadfence_system();
    for (int j = 0; j < k; ++j) {
        if (slots[t * k + j] < filler0 || slots[t * k + j] >= gpu0) continue;   // resident or GPU-taken: computed
        float * dst = exps + (t * k + j) * N;
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) dst[n] = 0.0f;
    }
    const float * src = yp_bell + t * N;
    float *       sh  = shared + t * N;
    // Every host load of a thread is issued before any is used: one PCIe round trip per thread, not one per element
    // (each dependent volatile load costs on the order of 2 us).
    constexpr int kV = 4;   // float4 loads in flight per thread: 256 threads x 4 x 16 B = 16 KiB = N 4096
    // Any N % 4 == 0 takes this path, with a bounds check per float4. Same single add per element as the scalar
    // fallback below, so bit-identical.
    if ((N % 4) == 0 && ((uintptr_t) src & 15) == 0 && ((uintptr_t) sh & 15) == 0) {
        const int64_t n4 = N / 4;
        for (int64_t base = 0; base < n4; base += kV * (int64_t) blockDim.x) {
            float4 v[kV];
#pragma unroll
            for (int i = 0; i < kV; ++i) {
                const int64_t q = base + (int64_t) i * blockDim.x + threadIdx.x;
                if (q < n4) v[i] = __ldcv((const float4 *) src + q);
            }
#pragma unroll
            for (int i = 0; i < kV; ++i) {
                const int64_t q = base + (int64_t) i * blockDim.x + threadIdx.x;
                if (q >= n4) continue;
                float4 * d = (float4 *) sh + q;
                float4   o = *d;
                o.x += v[i].x; o.y += v[i].y; o.z += v[i].z; o.w += v[i].w;
                *d = o;
            }
        }
    } else {
        for (int64_t n = threadIdx.x; n < N; n += blockDim.x) sh[n] += ((const volatile float *) src)[n];
    }
}

__global__ static void window_end_k(unsigned * base, int layers) { *base += (unsigned) layers; }

// out[:, t] = ((w0 e0 + w1 e1) + ... + w_{k-1} e_{k-1}) + shared: the same IEEE mul/add sequence as
// q4x::moe_weighted_sum's ggml chain (mul, k-1 sequential adds, + shared), in one launch instead of k + 1
__global__ static void moe_wsum_k(const float * exps, const float * w, const float * shared, float * out, int64_t N,
                                  int k, int64_t T) {
    const int64_t t = blockIdx.y;
    for (int64_t n = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; n < N; n += (int64_t) gridDim.x * blockDim.x) {
        float acc = __fmul_rn(exps[(t * k) * N + n], w[t * k]);
        for (int j = 1; j < k; ++j) acc = __fadd_rn(acc, __fmul_rn(exps[(t * k + j) * N + n], w[t * k + j]));
        out[t * N + n] = __fadd_rn(acc, shared[t * N + n]);
    }
}

namespace ai::kern {

void offload_resolve_ring(void * stream, const int32_t * table, const int32_t * sel, const float * x, int64_t N, int k,
                          int64_t T, int32_t filler0, int32_t * slots, int32_t * miss_bell, float * x_bell,
                          int32_t * n_tok_bell, unsigned * seq_bell, const unsigned * base, int layer,
                          const float * w, float * w_bell, int gpu_m, int32_t gpu0, int tail_rank) {
    resolve_ring_k<<<1, 512, 0, (cudaStream_t) stream>>>(table, sel, x, N, k, T, filler0, slots, miss_bell, x_bell,
                                                          n_tok_bell, seq_bell, base, layer, w, w_bell, gpu_m, gpu0,
                                                          tail_rank);
}

void offload_wait_stats_init() {
    static bool done = false;
    if (!done && getenv("AI_Q4X_HDR_TOUCH")) {
        const int touch = atoi(getenv("AI_Q4X_HDR_TOUCH"));
        cudaMemcpyToSymbol(g_touch, &touch, sizeof(touch));
    }
    if (done || !getenv("AI_Q4X_WAIT_STATS") || atoi(getenv("AI_Q4X_WAIT_STATS")) == 0) return;
    done = true;
    const int one = 1;
    cudaMemcpyToSymbol(g_ws_on, &one, sizeof(one));
    atexit([]() {
        unsigned long long h[4] = { 0, 0, 0, 0 };
        cudaMemcpyFromSymbol(h, g_ws, sizeof(h));
        fprintf(stderr, "[q4x-wait] waits %llu (spun %llu) | spin %.3f ms total | ring->wait-end %.3f ms total\n", h[0], h[1],
                h[2] / 1e6, h[3] / 1e6);
    });
}

void offload_wait_partial(void * stream, const unsigned * done_bell, const unsigned * base, int layer,
                          const int32_t * slots, int32_t filler0, int32_t gpu0, const float * yp_bell, float * exps,
                          float * shared, int64_t N, int k, int64_t T) {
    wait_partial_k<<<(unsigned) T, 256, 0, (cudaStream_t) stream>>>(done_bell, base, layer, slots, filler0, gpu0, yp_bell,
                                                                    exps, shared, N, k);
}

void offload_wait_fill(void * stream, const unsigned * done_bell, const unsigned * base, int layer,
                       const int32_t * miss_bell, const float * y_bell, float * exps, int64_t N, int k, int64_t T) {
    wait_fill_k<<<(unsigned) (k * T), 256, 0, (cudaStream_t) stream>>>(done_bell, base, layer, miss_bell, y_bell, exps,
                                                                       N, k, T);
}

// A pure select, no multiply: a filler column's value on the unselected side can be anything, inf included, without
// reaching the result (masking by multiplication would turn 0 * inf into NaN).
__global__ static void select_cols_k(const float * a, const float * b, const int32_t * use_a, float * out, int64_t N) {
    const int64_t c   = blockIdx.y;
    const float * src = use_a[c] ? a : b;
    for (int64_t n = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; n < N; n += (int64_t) gridDim.x * blockDim.x)
        out[c * N + n] = src[c * N + n];
}

void select_cols(void * stream, const float * a, const float * b, const int32_t * use_a, float * out, int64_t N,
                 int64_t cols) {
    const unsigned bx = (unsigned) std::min<int64_t>((N + 255) / 256, 16);
    select_cols_k<<<dim3(bx, (unsigned) cols), 256, 0, (cudaStream_t) stream>>>(a, b, use_a, out, N);
}

void moe_wsum(void * stream, const float * exps, const float * w, const float * shared, float * out, int64_t N, int k,
              int64_t T) {
    const unsigned bx = (unsigned) std::min<int64_t>((N + 255) / 256, 16);
    moe_wsum_k<<<dim3(bx, (unsigned) T), 256, 0, (cudaStream_t) stream>>>(exps, w, shared, out, N, k, T);
}

void offload_window_end(void * stream, unsigned * base, int layers) {
    window_end_k<<<1, 1, 0, (cudaStream_t) stream>>>(base, layers);
}

}  // namespace ai::kern
