// SPDX-License-Identifier: Apache-2.0
// Native GDN short conv for <= 8 rows per stream (q4x::gdn_block_sigz): the history columns and the new qkv columns are
// read in place -- no materialised concat (the transpose makes concat_f32_non_cont a slow strided gather), no separate
// state copy -- and the conv + SiLU is ssm_conv_f32<true>'s arithmetic exactly (the same
// ring, the same `sumf += x*w` order, x / (1 + expf(-x))), so conv_out is bit-identical. In record mode the conv input
// [K-1+L, C, B] is written out for the ReplaySSM fold, as the concat would have left it.
#include "kernels/q4x_gdn_conv.h"

#include <cuda_runtime.h>

namespace {

template <int K>
__global__ void gdn_conv_k(float * __restrict__ state, const float * __restrict__ qkv, const int64_t ld,
                           const float * __restrict__ w, const int64_t w_ld, float * __restrict__ out,
                           float * __restrict__ rec_in, const int C, const int L, const int write_state) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (c >= C) return;
    float * st = state + ((int64_t) b*C + c)*(K - 1);
    const float * q = qkv + (int64_t) b*L*ld + c;
    float hist[K - 1];
#pragma unroll
    for (int j = 0; j < K - 1; ++j) hist[j] = st[j];
    auto col = [&](int k) -> float { return k < K - 1 ? hist[k] : q[(int64_t) (k - (K - 1))*ld]; };
    float wv[K];
#pragma unroll
    for (int j = 0; j < K; ++j) wv[j] = w[(int64_t) c*w_ld + j];
    float x[K];
    for (int i = 0; i < L; ++i) {
        float sumf = 0.0f;
        if (i == 0) {
#pragma unroll
            for (int j = 0; j < K; ++j) x[j] = col(j);
        } else {
            x[(i - 1) % K] = col(i + K - 1);
        }
#pragma unroll
        for (int j = 0; j < K; ++j) sumf += x[(i + j) % K]*wv[j];
        out[((int64_t) b*L + i)*C + c] = sumf / (1.0f + expf(-sumf));
    }
    if (rec_in) {
        float * r = rec_in + ((int64_t) b*C + c)*(K - 1 + L);
        for (int k = 0; k < K - 1 + L; ++k) r[k] = col(k);
    }
    if (write_state) {
#pragma unroll
        for (int j = 0; j < K - 1; ++j) st[j] = col(L + j);
    }
}

}  // namespace

namespace ai::kern {

bool q4x_gdn_conv(void * stream, float * state, const float * qkv, int64_t ld, const float * w, int64_t w_ld, int K,
                  float * out, float * rec_in, int C, int L, int B, bool write_state) {
    if (K != 4) return false;
    const dim3 grid((C + 127)/128, B);
    gdn_conv_k<4><<<grid, 128, 0, (cudaStream_t) stream>>>(state, qkv, ld, w, w_ld, out, rec_in, C, L, write_state ? 1 : 0);
    return true;
}

}  // namespace ai::kern
