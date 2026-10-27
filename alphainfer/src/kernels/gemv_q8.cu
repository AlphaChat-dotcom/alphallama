// Prototype: Q8_0 weights x int8 activations, one token (GEMV).
//
//   quantize_act_q8: x (f32, n = 32*nb) -> q (int8, per-32 symmetric) and d (f32 per block)
//   gemv_q8_0:       y[r] = sum_b (d_w[r,b] * d_x[b]) * sum_j q_w[r,b,j] * q_x[b,j]
//
// Layout: one region per weight row; its 32 warps each stream every 32nd block of the row
// (block t, t+32, ...) and the partial sums are reduced across the warps.
#include "kernels/kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>

static constexpr int kWarp = 32;

static inline __host__ __device__ float fp16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t) (h >> 15) & 1u;
    int32_t  e = (int32_t) ((h >> 10) & 0x1f);
    uint32_t m = (uint32_t) (h & 0x3ff);
    uint32_t f;
    if (e == 0) {
        if (m == 0) {
            f = s << 31;
        } else {                                  // subnormal: normalise the mantissa
            e = 1;
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            f = (s << 31) | ((uint32_t) (e + 112) << 23) | (m << 13);
        }
    } else if (e == 31) {
        f = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        f = (s << 31) | ((uint32_t) (e + 112) << 23) | (m << 13);
    }
    float v;
    memcpy(&v, &f, 4);
    return v;
}

static inline __device__ float reduce_sum(float x) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, kWarp);
    }
    return x;
}

__global__ static void quantize_act_q8_k(const float * __restrict__ x, int8_t * __restrict__ q,
                                         float * __restrict__ d, long long nb) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) {
        return;
    }
    const float * xb = x + b * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        amax = std::fmax(amax, std::fabs(xb[j]));
    }
    const float db  = amax / 127.0f;
    const float inv = db > 0.0f ? 1.0f / db : 0.0f;
    for (int j = 0; j < 32; ++j) {
        q[b * 32 + j] = (int8_t) std::lrint(xb[j] * inv);
    }
    d[b] = db;
}

__global__ static void gemv_q8_0_k(const uint8_t * __restrict__ w, const int8_t * __restrict__ xq,
                                   const float * __restrict__ dx, float * __restrict__ y, int nb) {
    const long long r = blockIdx.x;
    const int       t = threadIdx.x;
    const uint8_t * row = w + (size_t) r * (size_t) nb * 34;
    float acc = 0.0f;
    for (int b = t; b < nb; b += kWarp) {
        const uint8_t * blk = row + (size_t) b * 34;
        uint16_t h;
        memcpy(&h, blk, 2);
        const int8_t * qw = (const int8_t *) (blk + 2);
        const int8_t * qx = xq + b * 32;
        int s = 0;
        for (int j = 0; j < 32; ++j) {
            s += (int) qw[j] * (int) qx[j];
        }
        acc += (float) s * (fp16_to_f32(h) * dx[b]);
    }
    acc = reduce_sum(acc);
    if (t == 0) {
        y[r] = acc;
    }
}

namespace ai::kern {

void quantize_act_q8(void * stream, const float * x, int8_t * q, float * d, int64_t nb) {
    const int threads = 64;
    const long long blocks = (nb + threads - 1) / threads;
    quantize_act_q8_k<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(x, q, d, (long long) nb);
}

void gemv_q8_0(void * stream, const uint8_t * w, int64_t rows, int64_t cols, const int8_t * xq, const float * dx,
               float * y, int rows_per_cta) {
    (void) rows_per_cta;
    gemv_q8_0_k<<<(unsigned) rows, kWarp, 0, (cudaStream_t) stream>>>(w, xq, dx, y, (int) (cols / 32));
}

}  // namespace ai::kern
