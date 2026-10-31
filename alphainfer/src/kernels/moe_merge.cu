// SPDX-License-Identifier: Apache-2.0
// Native MoE merge: weighted expert sum + shared expert + residual in one kernel.
// Compiled with FMA disabled so every product and sum is the same IEEE operation, in the same
// order, as the separate elementwise ops it replaces.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cuda_runtime.h>

__global__ static void moe_merge_residual_k(const float * __restrict__ e, const float * __restrict__ w,
                                            const float * __restrict__ sh, const float * __restrict__ r,
                                            float * __restrict__ y, long long E, int k, long long T) {
    const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * T) {
        return;
    }
    const long long t    = idx / E;
    const long long i    = idx - t * E;
    const long long base = (long long) k * t;          // weights row offset for token t
    float acc = e[i + E * base] * w[base];
    for (int j = 1; j < k; ++j) {
        acc = acc + e[i + E * (base + j)] * w[base + j];
    }
    acc = acc + sh[idx];
    y[idx] = acc + r[idx];
}

// Gated variant: the shared expert's output `sd` is scaled by sigmoid(score[t]) here, so the
// separate SIGMOID and MUL launches disappear. score[t] = sc[t * sc_stride + sc_off].
// Same float operations as the ggml path: sigmoid = 1 / (1 + exp(-x)), then sd * sigmoid.
__global__ static void moe_merge_gated_k(const float * __restrict__ e, const float * __restrict__ w,
                                         const float * __restrict__ sd, const float * __restrict__ sc,
                                         const float * __restrict__ r, float * __restrict__ y, long long E, int k,
                                         long long T, long long sc_stride, long long sc_off) {
    const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * T) {
        return;
    }
    const long long t    = idx / E;
    const long long i    = idx - t * E;
    const long long base = (long long) k * t;
    float acc = e[i + E * base] * w[base];
    for (int j = 1; j < k; ++j) {
        acc = acc + e[i + E * (base + j)] * w[base + j];
    }
    const float s   = sc[t * sc_stride + sc_off];
    // ex::sigmoid, not `1.0f / (1.0f + exp(-s))`: the ggml SIGMOID this replaces is compiled with
    // -use_fast_math, so its division is __fdividef and its exp is __expf. Exact division here puts
    // `sig` one ULP away, and that one ULP survives the multiply by sd[idx] for a few per cent of
    // elements.
    const float sig = ai::ex::sigmoid(s);
    const float sh  = sd[idx] * sig;
    acc = acc + sh;
    y[idx] = acc + r[idx];
}

namespace ai::kern {

void moe_merge_residual_gated(void * stream, const float * exps, const float * w, const float * shared_down,
                              const float * score, int64_t score_stride, int64_t score_off, const float * res,
                              float * y, int64_t E, int64_t k, int64_t T) {
    const long long n       = (long long) E * T;
    const int       threads = 256;
    const long long blocks  = (n + threads - 1) / threads;
    moe_merge_gated_k<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(
        exps, w, shared_down, score, res, y, (long long) E, (int) k, (long long) T, (long long) score_stride,
        (long long) score_off);
}

void moe_merge_residual(void * stream, const float * exps, const float * w, const float * shared,
                        const float * res, float * y, int64_t E, int64_t k, int64_t T) {
    const long long n       = (long long) E * T;
    const int       threads = 256;
    const long long blocks  = (n + threads - 1) / threads;
    moe_merge_residual_k<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(
        exps, w, shared, res, y, (long long) E, (int) k, (long long) T);
}

}  // namespace ai::kern

// y = (sum_j exps[.., j, ..] * w[j]) [+ add]: the MUL + k-1 ADDs (+ ADD) of the offloaded MoE paths in one launch.
// add may alias y (an accumulator: each element is read then written by the same thread; IEEE addition commutes).
__global__ static void moe_weighted_sum_k(const float * __restrict__ e, const float * __restrict__ w, const float * add,
                                          float * y, long long E, int k, long long T) {
    const long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= E * T) {
        return;
    }
    const long long t    = idx / E;
    const long long i    = idx - t * E;
    const long long base = (long long) k * t;
    float acc = e[i + E * base] * w[base];
    for (int j = 1; j < k; ++j) {
        acc = acc + e[i + E * (base + j)] * w[base + j];
    }
    y[idx] = add ? acc + add[idx] : acc;
}

namespace ai::kern {

void moe_weighted_sum(void * stream, const float * exps, const float * w, const float * add, float * y, int64_t E,
                      int64_t k, int64_t T) {
    const long long n       = (long long) E * T;
    const int       threads = 256;
    const long long blocks  = (n + threads - 1) / threads;
    moe_weighted_sum_k<<<(unsigned) blocks, threads, 0, (cudaStream_t) stream>>>(exps, w, add, y, (long long) E,
                                                                                (int) k, (long long) T);
}

}  // namespace ai::kern
