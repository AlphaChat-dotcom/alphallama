// [AI_Q4X_FOLD_BATCH] the speculative commit's state folds for EVERY layer in three launches (Qwen4ExpRun::commit):
//   1. prep: the vendored gdn_prep_kernel's k_hat (L2-normalised k rows), g = softplus(alpha + dt) * a and
//      beta = sigmoid(beta), per (layer, accepted column);
//   2. the vendored gated_delta_net_cuda<S_v = 128, scalar gate> state recurrence over the accepted columns, per (head,
//      layer, 4 state columns), state in place -- the attention output the fold discards is not computed (it never
//      feeds the state);
//   3. the conv history rolls (GDN conv_input tail, PLE padded tail) as one strided copy.
// The per-element float operations are the vendored kernels' in the same order, and this file is built
// -use_fast_math as ggml-cuda is (CMakeLists.txt), so the folded state is the same bits. The ggml path issued
// prep + recurrence + copy per GDN layer and a copy per PLE layer (~110 launches, 0.71 ms GPU per cycle on the 3090).
#include "kernels/q4x_fold.h"

#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, kWarp);
    return x;
}

// gdn_prep_kernel's k / g / beta branch (its q branch only feeds the attention output). One block per (column,
// layer), 128 threads; a warp per k head.
__global__ void fold_prep_k(const ai::kern::FoldGdn * __restrict__ L, float * __restrict__ scratch, int Sk, int Hk, int H,
                            int C, int tmax, float eps) {
    const int it = blockIdx.x, il = blockIdx.y;
    const ai::kern::FoldGdn & g = L[il];
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp, nwarp = blockDim.x / kWarp;
    const int64_t per_kh = (int64_t) tmax * Hk * Sk, per_h = (int64_t) tmax * H;
    float * kh = scratch + (int64_t) il * (per_kh + 2 * per_h);
    float * gh = kh + per_kh;
    float * bh = gh + per_h;
    for (int ih = warp; ih < Hk; ih += nwarp) {
        const float * xk = g.co + (int64_t) it * C + (int64_t) Hk * Sk + (int64_t) ih * Sk;
        float *       yk = kh + ((int64_t) it * Hk + ih) * Sk;
        float sk = 0.0f;
        for (int i = lane; i < Sk; i += kWarp) sk += xk[i] * xk[i];
        sk = warp_sum(sk);
        const float sck = 1.0f / fmaxf(sqrtf(sk), eps);
        for (int i = lane; i < Sk; i += kWarp) yk[i] = xk[i] * sck;
    }
    for (int ih = threadIdx.x; ih < H; ih += blockDim.x) {
        float gv = g.alpha[ih * g.sb1 + it * g.sb2];
        float bv = g.beta[ih * g.sb1 + it * g.sb2];
        const float x = gv + g.dt[ih];
        gv = ((x > 20.0f) ? x : log1pf(expf(x))) * g.a[ih];
        bv = 1.0f / (1.0f + expf(-bv));
        gh[(int64_t) it * H + ih] = gv;
        bh[(int64_t) it * H + ih] = bv;
    }
}

// gated_delta_net_cuda<128, false>'s state update for one (head, layer, column): rows in registers, 4 per lane.
__global__ void __launch_bounds__(kWarp * 4, 2) fold_rec_k(const ai::kern::FoldGdn * __restrict__ L,
                                                          const float * __restrict__ scratch, int Sk, int Hk, int H, int C,
                                                          int tmax, int m1) {
    constexpr int S_v = 128, rows = S_v / kWarp;
    const int h = blockIdx.x, il = blockIdx.y, lane = threadIdx.x, col = blockIdx.z * blockDim.y + threadIdx.y;
    const ai::kern::FoldGdn & g = L[il];
    const int64_t per_kh = (int64_t) tmax * Hk * Sk, per_h = (int64_t) tmax * H;
    const float * kh = scratch + (int64_t) il * (per_kh + 2 * per_h);
    const float * gh = kh + per_kh;
    const float * bh = gh + per_h;
    const int     iq1 = h % Hk;
    float * state = g.ssm + (int64_t) h * S_v * S_v;
    float   s_shard[rows];
#pragma unroll
    for (int r = 0; r < rows; r++) s_shard[r] = state[col * S_v + r * kWarp + lane];
    for (int t = 0; t < m1; t++) {
        const float * k_t = kh + ((int64_t) t * Hk + iq1) * Sk;
        const float   v_c = g.co[(int64_t) t * C + 2 * (int64_t) Hk * Sk + (int64_t) h * S_v + col];
        const float   beta_val = bh[(int64_t) t * H + h];
        float k_reg[rows];
#pragma unroll
        for (int r = 0; r < rows; r++) k_reg[r] = k_t[r * kWarp + lane];
        const float g_val = expf(gh[(int64_t) t * H + h]);
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < rows; r++) kv_shard += s_shard[r] * k_reg[r];
        const float kv_col    = warp_sum(kv_shard);
        const float delta_col = (v_c - g_val * kv_col) * beta_val;
#pragma unroll
        for (int r = 0; r < rows; r++) s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
    }
#pragma unroll
    for (int r = 0; r < rows; r++) state[col * S_v + r * kWarp + lane] = s_shard[r];
}

// dst[c * rows + j] = src[c * ld + m1 + j] for every copy entry (the ggml_cpy of a [rows, cols] view starting at column m1)
__global__ void fold_copy_k(const ai::kern::FoldCopy * __restrict__ cp, int m1) {
    const ai::kern::FoldCopy & e = cp[blockIdx.y];
    const int64_t n = (int64_t) e.rows * e.cols;
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t c = i / e.rows, j = i - c * e.rows;
        e.dst[i] = e.src[c * e.ld + m1 + j];
    }
}

}  // namespace

namespace ai::kern {

size_t q4x_fold_scratch_floats(int n_gdn, int Sk, int Hk, int H, int tmax) {
    return (size_t) n_gdn * ((size_t) tmax * Hk * Sk + 2 * (size_t) tmax * H);
}

bool q4x_fold_batch(void * stream, const FoldGdn * d_gdn, int n_gdn, const FoldCopy * d_cp, int n_cp, float * scratch, int m1,
                    int tmax, int C, int Sk, int Hk, int Sv, int Hv, float eps) {
    if (m1 < 1 || m1 > tmax || (n_gdn > 0 && Sv != 128)) return false;
    cudaStream_t st = (cudaStream_t) stream;
    if (n_gdn > 0) {
        fold_prep_k<<<dim3((unsigned) m1, (unsigned) n_gdn), 128, 0, st>>>(d_gdn, scratch, Sk, Hk, Hv, C, tmax, eps);
        fold_rec_k<<<dim3((unsigned) Hv, (unsigned) n_gdn, (unsigned) (Sv / 4)), dim3(kWarp, 4, 1), 0, st>>>(d_gdn, scratch, Sk, Hk,
                                                                                                         Hv, C, tmax, m1);
    }
    if (n_cp > 0) fold_copy_k<<<dim3(32, (unsigned) n_cp), 256, 0, st>>>(d_cp, m1);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace ai::kern
