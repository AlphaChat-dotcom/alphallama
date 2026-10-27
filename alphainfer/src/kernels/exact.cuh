// exact.cuh -- device primitives that reproduce ggml's CUDA arithmetic by construction.
//
// A fused kernel replaces a chain of ggml ops and must produce the same bits. Three things can make
// it differ:
//
//   1. Reduction order -- tree shape, warp width, how partials are combined.
//   2. FMA contraction -- whether `a*b + c` fuses into one rounding or two.
//   3. Approximate transcendentals -- ggml's CUDA backend builds with -use_fast_math
//      (ggml/src/ggml-cuda/CMakeLists.txt), so its `/` is __fdividef and its `expf` is __expf.
//      An exact division can put roundf() on a different integer, flipping a single int8 code,
//      which is enough to flip a near-tie token.
//
// Kernels built only from these primitives inherit bit-exactness instead of re-proving it.
//
// A fused kernel can span ops with different reference contracts, so no single compile flag is right
// for the whole file (see moe_merge_norm.cu: the merge half must not contract, the norm half must
// match norm.cu under -use_fast_math). The contract is therefore chosen per expression:
//
//   replacing a plain elementwise ggml op   -> ex::mul / ex::add      (explicitly non-contracting)
//   replacing a -use_fast_math reference    -> ordinary * and +       (contracts, as it does there)
//                                              plus ex::div / ex::exp
//
// Naming them makes the choice visible at the call site and independent of the file's compile flags.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace ai::ex {

// Explicit rounding contracts

// Non-contracting. Use when replacing a plain elementwise ggml op (ADD, MUL, the merge half of a
// fusion): those are separate kernels in the reference, so each product and sum is its own IEEE
// rounding. moe_merge.cu gets this from -fmad=false on the file; spelling it out here keeps the
// guarantee independent of CMakeLists.txt.
__device__ __forceinline__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add(float a, float b) { return __fadd_rn(a, b); }

// Approximate, matching -use_fast_math. Use for every division and transcendental that corresponds
// to one in ggml. Do not write bare `/` in a kernel that must match a ggml reference.
__device__ __forceinline__ float div(float a, float b) { return __fdividef(a, b); }
__device__ __forceinline__ float exp(float x)          { return __expf(x); }
__device__ __forceinline__ float rsqrt(float x)        { return rsqrtf(x); }

// sigmoid as ggml computes it: 1/(1+expf(-x)) under -use_fast_math.
__device__ __forceinline__ float sigmoid(float x) { return div(1.0f, 1.0f + exp(-x)); }

// SiLU / swish, in ggml's order.
__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }

// Reductions, in ggml's tree shape
//
// Butterfly over the full mask, offset 16 -> 1. This is common.cuh's warp_reduce_sum<32> /
// warp_reduce_max; the three hand-copies in add_rmsnorm.cu, moe_merge_norm.cu and
// moe_merge_norm_coop.cu are all this function and should be replaced by it.

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, 32));
    }
    return x;
}

// ggml's block_reduce<SUM, block_size>. The structure is load-bearing: warp butterfly -> lane 0
// writes s_sum[warp_id] -> __syncthreads() -> reset to zero -> lanes < block_size/32 reload -> every
// warp re-reduces the partials. Changing any step changes the summation order and therefore the bits.
//
// s_sum must be a __shared__ float[32] supplied by the caller.
template <int block_size>
__device__ __forceinline__ float block_sum(float tmp, float * s_sum, int tid) {
    tmp = warp_sum(tmp);
    if (block_size > 32) {
        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = 0.0f;
        if (lane_id < block_size / 32) {
            tmp = s_sum[lane_id];
        }
        tmp = warp_sum(tmp);
    }
    return tmp;
}

// ggml picks 1024 for ncols >= 1024 and 256 below it (norm.cu, rms_norm_f32_cuda). A fused
// kernel that norms a row must make the same choice or the reduction width differs.
__host__ __device__ __forceinline__ int norm_block_size(int ncols) { return ncols < 1024 ? 256 : 1024; }

// RMS norm scale, in norm.cu's order
//
// Two halves with different contracts:
//   - the accumulation `tmp += v*v` contracts to an FMA in norm.cu (built -use_fast_math), so the
//     caller must write it with ordinary ops, not ex::mul/ex::add;
//   - the mean division is the approximate one.
// The epilogue must then be written `scale * x * w`, in that association -- see rms_apply below.
__device__ __forceinline__ float rms_scale(float sum_sq, int ncols, float eps) {
    const float mean = div(sum_sq, (float) ncols);
    return rsqrt(mean + eps);
}

// norm.cu's epilogue is `dst[col] = scale * x[col] * mul[mul_col];` -- left-to-right, default
// contraction. Do not reassociate.
__device__ __forceinline__ float rms_apply(float scale, float x, float w) { return scale * x * w; }

// block_q8_1 producer epilogue
// ggml's block_q8_1: a 4-byte half2 (scale, sum) followed by 32 int8 codes = 36 bytes.
struct block_q8_1 {
    __half2 ds;
    int8_t  qs[32];
};

// Emit one value into its q8_1 block, following quantize_q8_1: amax over the block, d = amax/127,
// q = round(v/d), ds = (d, sum of the unquantised values).
//
// Precondition: the caller's column stride must be the block size, so that lanes 0..31 of a warp hold
// 32 consecutive columns -- exactly one q8_1 block. That makes this a warp reduction with no shared
// memory, and it requires every lane of the warp to be active (ncols % block_size == 0). Check
// q8_geometry_ok() on the host before passing a non-null buffer.
//
// Rounding is half to even (__float2int_rn) with a clamp to +-127; a quotient landing exactly on x.5
// differs by one code from roundf (half away from zero). Both producers (add_rmsnorm,
// moe_merge_partials) use this function. AI_Q8_STAGE=3 compares the staged codes against the reference.
// TODO: ggml's quantize_q8_1 in this repository uses roundf; confirm which rounding the consumer expects.
__device__ __forceinline__ void emit_q8_1(block_q8_1 * y, long long i_cont, float v) {
    const float amax = warp_max(fabsf(v));
    const float sum  = warp_sum(v);
    const float d    = div(amax, 127.0f);
    int         qi   = d > 0.0f ? __float2int_rn(div(v, d)) : 0;
    qi               = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
    const int8_t q   = (int8_t) qi;
    y[i_cont / 32].qs[i_cont % 32] = q;
    if ((i_cont % 32) == 0) {
        y[i_cont / 32].ds = __floats2half2_rn(d, sum);
    }
}

// Fails closed. Three conditions, each for its own reason:
//   ncols % 32   -- whole q8_1 blocks;
//   ncols % bs   -- the in-loop warp reductions need every lane of the warp active;
//   ncols % 512  -- MATRIX_ROW_PADDING. The reference quantises a padded row and zero-fills the
//                   tail; a producer epilogue writes only real columns, so with padding present the
//                   tail blocks would be uninitialised and the consumer would read them.
// A producer that cannot satisfy these must emit nothing and let the consumer launch its own
// quantise -- slower, but correct. A partial row is never staged.
__host__ __device__ __forceinline__ bool q8_geometry_ok(long long ncols, int block_size) {
    return ncols % 32 == 0 && ncols % block_size == 0 && ncols % 512 == 0;
}

}   // namespace ai::ex
