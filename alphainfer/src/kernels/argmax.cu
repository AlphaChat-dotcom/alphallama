// SPDX-License-Identifier: Apache-2.0
// Parallel argmax over the vocabulary.
//
// ggml's argmax kernel uses one block per row, so a decode step reduces a ~248k-entry vocabulary row on
// a single block, which is far slower than the 1 MB read warrants. This splits each row across
// AMAX_BLOCKS blocks and reduces the partials in a second small launch; the extra launch costs much
// less than the serialisation it removes.
//
// Bit-exact with ggml's kernel, tie-break included: comparisons are strictly `>` and partials
// are combined in ascending index order, so the lowest index wins among equal values.
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <cfloat>

#define AMAX_THREADS 256
#define AMAX_BLOCKS  256

__global__ static void argmax_partial_k(const float * __restrict__ x, long long ncols, float * __restrict__ pmax,
                                        int32_t * __restrict__ pidx) {
    __shared__ float   smax[AMAX_THREADS];
    __shared__ int32_t sidx[AMAX_THREADS];

    const long long row  = blockIdx.y;
    const float *   rowx = x + row * ncols;

    float   best = -FLT_MAX;
    int32_t bidx = -1;
    for (long long col = (long long) blockIdx.x * AMAX_THREADS + threadIdx.x; col < ncols;
         col += (long long) AMAX_BLOCKS * AMAX_THREADS) {
        const float v = rowx[col];
        if (v > best) {
            best = v;
            bidx = (int32_t) col;
        }
    }
    smax[threadIdx.x] = best;
    sidx[threadIdx.x] = bidx;
    __syncthreads();

    // halving preserves ascending index order, so ties keep the lower index
    for (int st = AMAX_THREADS / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st) {
            if (smax[threadIdx.x + st] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + st];
                sidx[threadIdx.x] = sidx[threadIdx.x + st];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const long long slot = row * AMAX_BLOCKS + blockIdx.x;
        pmax[slot] = smax[0];
        pidx[slot] = sidx[0];
    }
}

__global__ static void argmax_final_k(const float * __restrict__ pmax, const int32_t * __restrict__ pidx,
                                      int32_t * __restrict__ dst, const int32_t * __restrict__ map) {
    __shared__ float   smax[AMAX_BLOCKS];
    __shared__ int32_t sidx[AMAX_BLOCKS];

    const long long row = blockIdx.x;
    smax[threadIdx.x]   = pmax[row * AMAX_BLOCKS + threadIdx.x];
    sidx[threadIdx.x]   = pidx[row * AMAX_BLOCKS + threadIdx.x];
    __syncthreads();
    for (int st = AMAX_BLOCKS / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st) {
            if (smax[threadIdx.x + st] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + st];
                sidx[threadIdx.x] = sidx[threadIdx.x + st];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dst[row] = map != nullptr ? map[sidx[0]] : sidx[0];
    }
}

namespace ai::kern {

int argmax_partials_per_row() {
    return AMAX_BLOCKS;
}

void argmax_rows(void * stream, const float * x, int64_t ncols, int64_t rows, float * pmax, int32_t * pidx,
                 int32_t * dst, const int32_t * map) {
    const dim3 grid(AMAX_BLOCKS, (unsigned) rows);
    argmax_partial_k<<<grid, AMAX_THREADS, 0, (cudaStream_t) stream>>>(x, (long long) ncols, pmax, pidx);
    argmax_final_k<<<(unsigned) rows, AMAX_BLOCKS, 0, (cudaStream_t) stream>>>(pmax, pidx, dst, map);
}

}   // namespace ai::kern
