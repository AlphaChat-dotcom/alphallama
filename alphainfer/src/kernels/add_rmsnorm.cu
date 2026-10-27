// Residual add fused into the RMS norm that always follows it.
//
// Every layer does `r = mix + res` then `p = rms_norm(r) * w`, which is two launches: the ADD
// (k_bin_bcast) and the fused RMS_NORM+MUL. The add's 8 KB is already in L2, so its cost is almost
// all launch. Both outputs are still produced: r feeds the next residual, p feeds the block.
//
// ggml has an ADD fusion in rms_norm_f32 already, but for the other expression --
// `rms_norm(x)*w + y`, matched as RMS_NORM -> MUL -> ADD -- whereas this one is `rms_norm(x + y)*w`.
//
// Bit-exact with the two ops it replaces, by construction: same block size (1024 for ncols >= 1024,
// else 256), same strided accumulation order, same two-level block reduce (warp butterfly, lane 0 to
// shared, every warp re-reduces the 32 partials), same `scale * x * w` order, and this file compiles
// with the default FMA contraction, as norm.cu does. These guarantees come from exact.cuh; the q8_1
// copy can be checked against the reference quantiser with AI_Q8_STAGE=3.
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace ex = ai::ex;

template <int block_size>
__global__ static void add_rmsnorm_k(const float * __restrict__ mix, const float * __restrict__ res,
                                     const float * __restrict__ w, float * __restrict__ r,
                                     float * __restrict__ p, void * __restrict__ q8, int ncols, float eps) {
    __shared__ float s_sum[32];

    const long long row = blockIdx.x;
    const int        tid = threadIdx.x;

    const float * m = mix + row * ncols;
    const float * s = res + row * ncols;
    float *       ro = r + row * ncols;
    float *       po = p + row * ncols;

    // Ordinary ops, deliberately: norm.cu is built with -use_fast_math, where `tmp += v*v` contracts
    // to an FMA. ex::mul/ex::add here would be wrong; see the rounding contracts in exact.cuh.
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float v = m[col] + s[col];
        ro[col] = v;
        tmp += v * v;
    }

    tmp = ex::block_sum<block_size>(tmp, s_sum, tid);

    const float scale = ex::rms_scale(tmp, ncols, eps);
    for (int col = tid; col < ncols; col += block_size) {
        const float v = ex::rms_apply(scale, ro[col], w[col]);
        po[col] = v;

        // Optional q8_1 copy for consumers that would otherwise each launch their own quantise.
        // The column stride is the block size, so lanes 0..31 of a warp hold 32 consecutive columns
        // -- exactly one q8_1 block -- which is the precondition ex::emit_q8_1 documents. The host
        // wrapper below only passes q8 when the geometry guarantees it.
        if (q8 != nullptr) {
            ex::emit_q8_1((ex::block_q8_1 *) q8, row * ncols + col, v);
        }
    }
}

namespace ai::kern {

void add_rmsnorm(void * stream, const float * mix, const float * res, const float * w, float * r, float * p,
                 void * q8, int64_t ncols, int64_t rows, float eps) {
    const dim3 grid((unsigned) rows);
    const int  bs = ex::norm_block_size((int) ncols);
    // Fail closed rather than silently: see ex::q8_geometry_ok for why each condition is there.
    if (q8 != nullptr && !ex::q8_geometry_ok(ncols, bs)) {
        q8 = nullptr;
    }
    if (bs == 256) {
        add_rmsnorm_k<256><<<grid, 256, 0, (cudaStream_t) stream>>>(mix, res, w, r, p, q8, (int) ncols, eps);
    } else {
        add_rmsnorm_k<1024><<<grid, 1024, 0, (cudaStream_t) stream>>>(mix, res, w, r, p, q8, (int) ncols, eps);
    }
}

}   // namespace ai::kern
