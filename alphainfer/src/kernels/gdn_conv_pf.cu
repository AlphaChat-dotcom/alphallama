// GDN prefill short convolution without the history concat (AI_GDN_CONV_PF, default on).
//
// The unfused prefill path builds conv_input = concat(conv history [K-1, C], qkv^T [L, C]) and runs ggml's SSM_CONV
// (+ fused SiLU) on it, then copies the last K-1 columns back as the new history. The concat copies the whole qkv
// block to prepend three columns. This reads the history and the projection rows in place:
//   gdn_conv_pf_k     out[t][c] = silu( sum_{j<K} x_{t-K+1+j}[c] * w[c][j] ),  x_{<0} from the history
//   gdn_conv_pf_state history[c][j] = x_{L-K+1+j}[c] (or the old history when L < K-1), after the conv has read it
// Same arithmetic as ggml's ssm_conv_long_token_f32<apply_silu> (sumf += x * w in j order from 0, then
// x / (1 + expf(-x))), and this file is built -use_fast_math as ggml-cuda is, so it is meant to be
// bit-identical to the chain it replaces (verify with AI_TF_DUMP: off vs on).
#include "kernels/kernels.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace ai {
namespace kern {
namespace {

constexpr int CPF_THREADS = 128;
constexpr int CPF_TB      = 64;   // tokens per thread

template <int K>
__global__ void __launch_bounds__(CPF_THREADS)
gdn_conv_pf_k(const float * __restrict__ x, int64_t x_tok, int64_t x_seq, const float * __restrict__ hist,
              const float * __restrict__ w, float * __restrict__ out, int C, int L) {
    constexpr int K1 = K - 1;
    const int c = blockIdx.x * CPF_THREADS + threadIdx.x;
    if (c >= C) return;
    const int b = blockIdx.z, t0 = blockIdx.y * CPF_TB, t1 = min(L, t0 + CPF_TB);
    const float * xb = x + (int64_t) b * x_seq + c;
    const float * hb = hist + ((int64_t) b * C + c) * K1;
    float wr[K];
#pragma unroll
    for (int j = 0; j < K; ++j) wr[j] = w[(int64_t) c * K + j];
    float win[K1 > 0 ? K1 : 1];
#pragma unroll
    for (int j = 0; j < K1; ++j) {
        const int i = t0 - K1 + j;
        win[j] = i >= 0 ? xb[(int64_t) i * x_tok] : hb[K1 + i];
    }
    float * ob = out + ((int64_t) b * L) * C + c;
    for (int t = t0; t < t1; ++t) {
        const float xt = xb[(int64_t) t * x_tok];
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < K1; ++j) sumf += win[j] * wr[j];
        sumf += xt * wr[K1];
        ob[(int64_t) t * C] = sumf / (1.0f + expf(-sumf));
#pragma unroll
        for (int j = 0; j + 1 < K1; ++j) win[j] = win[j + 1];
        if (K1 > 0) win[K1 - 1] = xt;
    }
}

__global__ void gdn_conv_pf_state(const float * __restrict__ x, int64_t x_tok, int64_t x_seq, float * __restrict__ hist,
                                  int C, int K1, int L, int B) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) C * B) return;
    const int c = (int) (i % C), b = (int) (i / C);
    float * hb = hist + ((int64_t) b * C + c) * K1;
    float nv[8];
    for (int j = 0; j < K1; ++j) {   // conv_input index K1 + L - K1 + j = L + j; < K1 means the old history
        const int k = L + j;
        nv[j] = k >= K1 ? x[(int64_t) b * x_seq + (int64_t) (k - K1) * x_tok + c] : hb[k];
    }
    for (int j = 0; j < K1; ++j) hb[j] = nv[j];
}

}  // namespace

bool gdn_conv_pf_enabled() {
    // default on: bit-identical to the concat + SSM_CONV chain, and a few percent faster on long prefills
    static const bool v = getenv("AI_GDN_CONV_PF") ? atoi(getenv("AI_GDN_CONV_PF")) != 0 : true;
    return v;
}

bool gdn_conv_pf(void * stream, const float * x, int64_t x_tok, int64_t x_seq, float * hist, const float * w,
                 float * out, int64_t C, int64_t K, int64_t L, int64_t B) {
    if (K < 2 || K > 9 || L < 1) return false;
    cudaStream_t st = (cudaStream_t) stream;
    const dim3 grid((unsigned) ((C + CPF_THREADS - 1) / CPF_THREADS), (unsigned) ((L + CPF_TB - 1) / CPF_TB), (unsigned) B);
    switch (K) {
        case 3: gdn_conv_pf_k<3><<<grid, CPF_THREADS, 0, st>>>(x, x_tok, x_seq, hist, w, out, (int) C, (int) L); break;
        case 4: gdn_conv_pf_k<4><<<grid, CPF_THREADS, 0, st>>>(x, x_tok, x_seq, hist, w, out, (int) C, (int) L); break;
        case 5: gdn_conv_pf_k<5><<<grid, CPF_THREADS, 0, st>>>(x, x_tok, x_seq, hist, w, out, (int) C, (int) L); break;
        default: return false;
    }
    gdn_conv_pf_state<<<(unsigned) ((C * B + 255) / 256), 256, 0, st>>>(x, x_tok, x_seq, hist, (int) C, (int) (K - 1),
                                                                         (int) L, (int) B);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace kern
}  // namespace ai
