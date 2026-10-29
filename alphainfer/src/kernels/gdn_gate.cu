// SPDX-License-Identifier: Apache-2.0
// GDN output gate in one kernel: fin = silu(z) * (rms_norm(out) * w), plus its q8_1 copy (AI_GDN_GATE).
//
// Every GDN layer ends with rms_norm_f32<256>, the SwiGLU-split gate and a quantize_q8_1 for the
// out-projection: three small kernels of fixed cost. This does all three in one and hands the q8_1 copy to
// the ssm_out GEMV via the staging registry, so its quantise disappears too.
//
// Bit-exact against that chain, by construction:
//   - the norm is the reference's own geometry: one 256-thread block per (head, token) row of Sv = 128
//     columns (rms_norm_f32<256>, whose upper 128 lanes contribute zeros), reduced by ex::block_sum<256>,
//     scale by ex::rms_scale, epilogue ex::rms_apply (scale * x * w);
//   - the gate is ggml's unary_gated_op_kernel<op_silu>: op(z) * normed, with op_silu =
//     x / (1 + expf(-x)) under -use_fast_math, i.e. ex::div(z, 1 + ex::exp(-z)). Note that ex::silu in
//     exact.cuh is x * sigmoid(x), a different float, and is not used here;
//   - the q8_1 copy is ex::emit_q8_1 on 32 consecutive columns per warp (warps 0-3 cover the 128 columns).
#include "kernels/kernels.h"
#include "kernels/exact.cuh"

#include <cstdio>
#include <cuda_runtime.h>

__global__ static void gdn_gated_norm_q8_k(const float * __restrict__ out, long long out_tok, const float * __restrict__ z,
                                           long long z_tok, const float * __restrict__ nw, float * __restrict__ fin,
                                           ai::ex::block_q8_1 * __restrict__ q8, int Sv, int Hv, float eps) {
    __shared__ float s_sum[32];
    const int  h   = blockIdx.x;
    const long t   = blockIdx.y;
    const int  tid = threadIdx.x;
    const long row = t * (long) Sv * Hv + (long) h * Sv;   // fin / q8 element index of this row's column 0

    const float x   = tid < Sv ? out[t * out_tok + (long) h * Sv + tid] : 0.0f;
    float       tmp = x * x;   // norm.cu's first `tmp += x*x` from 0: one rounding either way
    tmp             = ai::ex::block_sum<256>(tmp, s_sum, tid);
    const float scale = ai::ex::rms_scale(tmp, Sv, eps);
    if (tid < Sv) {
        const float nrm = ai::ex::rms_apply(scale, x, nw[tid]);
        const float zz  = z[t * z_tok + (long) h * Sv + tid];
        const float g   = ai::ex::div(zz, 1.0f + ai::ex::exp(-zz)) * nrm;   // op_silu(z) * normed
        fin[row + tid]  = g;
        if (q8 != nullptr) {
            ai::ex::emit_q8_1(q8, row + tid, g);   // whole warps only: Sv % 32 == 0
        }
    }
}

namespace ai::kern {

bool gdn_gated_norm_q8(void * stream, const float * out, int64_t out_tok, const float * z, int64_t z_tok,
                       const float * norm_w, float * fin, void * q8, int64_t Sv, int64_t Hv, int64_t T, float eps) {
    if (Sv > 256 || Sv % 32 != 0 || T < 1) {
        return false;
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[gdn-gate] LAUNCHED (%lld x %lld blocks, q8 %s)\n", (long long) Hv, (long long) T,
                q8 ? "STAGED" : "off");
    }
    gdn_gated_norm_q8_k<<<dim3((unsigned) Hv, (unsigned) T), 256, 0, (cudaStream_t) stream>>>(
        out, (long long) out_tok, z, (long long) z_tok, norm_w, fin, (ai::ex::block_q8_1 *) q8, (int) Sv, (int) Hv,
        eps);
    return true;
}

}   // namespace ai::kern
