// Draft confidence for AI_MTP_GATE: the max-softmax of a draft head's logits row, 1 / sum_j exp(x_j - max).
// The confidence-gated verify width reads it (engine.cpp). It only decides how many drafts the next round verifies;
// it never changes a draft or a verify row, so it has no bearing on exactness. It is well calibrated:
// P(accepted | reached) tracks this value along the diagonal.
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>

namespace {

constexpr int CONF_THREADS = 1024;

__global__ void __launch_bounds__(CONF_THREADS) max_softmax_rows_k(const float * __restrict__ x, long long ncols,
                                                                    float * __restrict__ out) {
    const float * row = x + (long long) blockIdx.x * ncols;
    __shared__ float red[CONF_THREADS / 32];
    float            m = -FLT_MAX;
    for (long long i = threadIdx.x; i < ncols; i += CONF_THREADS) {
        m = fmaxf(m, row[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < CONF_THREADS / 32 ? red[threadIdx.x] : -FLT_MAX;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float mx = red[0];
    __syncthreads();
    float s = 0.0f;
    for (long long i = threadIdx.x; i < ncols; i += CONF_THREADS) {
        s += __expf(row[i] - mx);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < CONF_THREADS / 32 ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) out[blockIdx.x] = v > 0.0f ? 1.0f / v : 0.0f;
    }
}

// Split version (the default). The single-block kernel above puts a whole vocabulary row on one SM and reads it
// twice; at batch 1 that is ~15-20 us per draft, on the draft chain's critical path.
// Here: SPLIT_BLOCKS blocks per row, one online pass (running max + rescaled sum), and the last block to finish
// (atomic ticket) combines the partials. One kernel, so no second launch. The ticket self-resets, so graph replay
// needs no memset. Calls on one stream serialise; two engines on two streams at once would share the tickets (not a
// supported configuration). Both kernels match a double-precision host softmax to 1e-5.
constexpr int SPLIT_BLOCKS  = 64;
constexpr int SPLIT_THREADS = 256;
constexpr int SPLIT_MAXROWS = 64;
__device__ unsigned int g_conf_ticket[SPLIT_MAXROWS];

__device__ __forceinline__ void ms_merge(float & m, float & s, float m2, float s2) {
    const float M = fmaxf(m, m2);
    s             = s * __expf(m - M) + s2 * __expf(m2 - M);
    m             = M;
}

__global__ void __launch_bounds__(SPLIT_THREADS) max_softmax_split_k(const float * __restrict__ x, long long ncols,
                                                                      float2 * __restrict__ part,
                                                                      float * __restrict__ out) {
    const int     r   = blockIdx.y;
    const float * row = x + (long long) r * ncols;
    float         m   = -FLT_MAX, s = 0.0f;
    for (long long i = (long long) blockIdx.x * SPLIT_THREADS + threadIdx.x; i < ncols;
         i += (long long) SPLIT_BLOCKS * SPLIT_THREADS) {
        const float v = row[i];
        if (v > m) {
            s = s * __expf(m - v) + 1.0f;
            m = v;
        } else {
            s += __expf(v - m);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ms_merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    }
    __shared__ float2 red[SPLIT_THREADS / 32];
    __shared__ bool   last;
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = make_float2(m, s);
    __syncthreads();
    if (threadIdx.x == 0) {
        float2 a = red[0];
        for (int w = 1; w < SPLIT_THREADS / 32; ++w) ms_merge(a.x, a.y, red[w].x, red[w].y);
        part[r * SPLIT_BLOCKS + blockIdx.x] = a;
        __threadfence();
        last = atomicAdd(&g_conf_ticket[r], 1u) == SPLIT_BLOCKS - 1;
    }
    __syncthreads();
    if (!last) return;
    if (threadIdx.x < 32) {
        float mm = -FLT_MAX, ss = 0.0f;
        for (int b = threadIdx.x; b < SPLIT_BLOCKS; b += 32) {
            const float2 p = __ldcg(&part[r * SPLIT_BLOCKS + b]);
            ms_merge(mm, ss, p.x, p.y);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            ms_merge(mm, ss, __shfl_xor_sync(0xffffffffu, mm, o), __shfl_xor_sync(0xffffffffu, ss, o));
        }
        if (threadIdx.x == 0) {
            out[r]             = ss > 0.0f ? 1.0f / ss : 0.0f;
            g_conf_ticket[r]   = 0u;
        }
    }
}

}  // namespace

namespace ai::kern {

int64_t max_softmax_scratch_floats(int64_t rows) { return rows <= SPLIT_MAXROWS ? rows * SPLIT_BLOCKS * 2 : 0; }

void max_softmax_rows(void * stream, const float * x, int64_t ncols, int64_t rows, float * out, float * scratch) {
    static const bool single = getenv("AI_CONF_SINGLE") != nullptr;   // one-block-per-row kernel
    if (single || !scratch || rows > SPLIT_MAXROWS) {
        max_softmax_rows_k<<<(unsigned) rows, CONF_THREADS, 0, (cudaStream_t) stream>>>(x, (long long) ncols, out);
        return;
    }
    max_softmax_split_k<<<dim3(SPLIT_BLOCKS, (unsigned) rows), SPLIT_THREADS, 0, (cudaStream_t) stream>>>(
        x, (long long) ncols, (float2 *) scratch, out);
}

}  // namespace ai::kern
