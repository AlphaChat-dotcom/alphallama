// GDN prompt scan for prefill (q4x::gdn_block_sigz, rows per call > 64): the gated delta rule token by token with the
// state slice in registers. One block per (key head, 32-column group): 16 x 4 = 64 blocks
// of 32 x 4 threads; a thread holds column `col` of the three value heads that share its key head (head kh + j * Hk, the
// ggml's head % Hk mapping), rows rg * 32 .. rg * 32 + 31, so every q / k value it loads feeds three heads. Inputs come
// in blocks of 8 tokens, copied to shared memory by cp.async while the block before computes. FP32 FMA throughout: not
// ggml's chunked TF32 bits (another order of the sums), gated by the paired NLL.
//
// The prep (gdn_pf_prep_k) is ggml's gdn_prep_kernel arithmetic: L2-normalised q / k rows (written in place into
// conv_out), beta = sigmoid(beta), gate = softplus(alpha + dt) * a. The parallel conv (gdn_pf_conv_k) is
// q4x_gdn_conv.cu's arithmetic per output (the same `sumf += x*w` order over the window), one thread per (row, channel)
// instead of a loop over rows.
#include "kernels/q4x_gdn_scan.h"

#include <cuda_runtime.h>

namespace {

constexpr int S = 128, CB = 32, RG = 4, RPG = S / RG, NCB = S / CB, TB = 8, VPK = 3;

__device__ __forceinline__ void cp16(float * smem, const float * gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp4(float * smem, const float * gmem) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_prev() { asm volatile("cp.async.wait_group 1;\n" ::); }

// h: conv_out rows [T][C] (q | k | v), q / k already L2-normalised; gate / beta [T][Hv]; state [Hv][col][row];
// y [T][Hv][S]
__global__ void __launch_bounds__(CB * RG) gdn_pf_scan_k(float * __restrict__ state, const float * __restrict__ h,
                                                         const int64_t C, const float * __restrict__ gate,
                                                         const float * __restrict__ beta, float * __restrict__ y,
                                                         const int Hk, const int64_t T, const float scale) {
    constexpr int NT = CB * RG, QKP = S / 4, VP = CB / 4;
    __shared__ __align__(16) float sq[2][TB][S];
    __shared__ __align__(16) float sk[2][TB][S];
    __shared__ __align__(16) float sv[2][TB][VPK][CB];
    __shared__ float sg[2][TB][VPK], sb[2][TB][VPK], rkv[VPK][RG][CB], ro[VPK][RG][CB];
    const int Hv = Hk * VPK;
    const int kh = blockIdx.x / NCB, cb = blockIdx.x % NCB;
    const int c = threadIdx.x, rg = threadIdx.y, tid = rg * CB + c, col = cb * CB + c;
    float s[VPK][RPG];
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        const float4 * base = reinterpret_cast<const float4 *>(state + ((size_t) (kh + j * Hk) * S + col) * S + rg * RPG);
#pragma unroll
        for (int r = 0; r < RPG / 4; ++r) {
            const float4 v4 = base[r];
            s[j][4 * r] = v4.x; s[j][4 * r + 1] = v4.y; s[j][4 * r + 2] = v4.z; s[j][4 * r + 3] = v4.w;
        }
    }
    const int64_t nblk = (T + TB - 1) / TB;
    auto stage = [&](int64_t k) {
        const int     bb = (int) (k & 1);
        const int64_t t0 = k * TB;
        for (int p = tid; p < TB * 2 * QKP; p += NT) {
            const int i = p / (2 * QKP), w = p % (2 * QKP), isk = w / QKP, jj = (w % QKP) * 4;
            if (t0 + i < T)
                cp16(isk ? &sk[bb][i][jj] : &sq[bb][i][jj], h + (t0 + i) * C + (isk ? (int64_t) Hk * S : 0) + kh * S + jj);
        }
        for (int p = tid; p < TB * VPK * VP; p += NT) {
            const int i = p / (VPK * VP), w = p % (VPK * VP), j = w / VP, jj = (w % VP) * 4;
            if (t0 + i < T)
                cp16(&sv[bb][i][j][jj], h + (t0 + i) * C + 2 * (int64_t) Hk * S + (kh + j * Hk) * S + cb * CB + jj);
        }
        for (int p = tid; p < 2 * TB * VPK; p += NT) {
            const int isb = p / (TB * VPK), w = p % (TB * VPK), i = w / VPK, j = w % VPK;
            if (t0 + i < T) cp4(isb ? &sb[bb][i][j] : &sg[bb][i][j], (isb ? beta : gate) + (t0 + i) * Hv + kh + j * Hk);
        }
    };
    if (nblk > 0) stage(0);
    cp_commit();
    for (int64_t k = 0; k < nblk; ++k) {
        if (k + 1 < nblk) stage(k + 1);
        cp_commit();
        cp_wait_prev();
        __syncthreads();
        const int bb = (int) (k & 1);
        const int n  = (int) ((T - k * TB) < TB ? (T - k * TB) : TB);
        for (int i = 0; i < n; ++i) {
            const int64_t t = k * TB + i;
            float kc[RPG];
#pragma unroll
            for (int r = 0; r < RPG; ++r) kc[r] = sk[bb][i][rg * RPG + r];
            float g[VPK], kv[VPK], delta[VPK], o[VPK];
#pragma unroll
            for (int j = 0; j < VPK; ++j) { g[j] = __expf(sg[bb][i][j]); kv[j] = 0.0f; o[j] = 0.0f; }
#pragma unroll
            for (int r = 0; r < RPG; ++r)
#pragma unroll
                for (int j = 0; j < VPK; ++j) kv[j] = fmaf(s[j][r], kc[r], kv[j]);
#pragma unroll
            for (int j = 0; j < VPK; ++j) rkv[j][rg][c] = kv[j];
            __syncthreads();
#pragma unroll
            for (int j = 0; j < VPK; ++j) {
                const float kv_col = rkv[j][0][c] + rkv[j][1][c] + rkv[j][2][c] + rkv[j][3][c];
                delta[j] = (sv[bb][i][j][c] - g[j] * kv_col) * sb[bb][i][j];
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                const float qr = sq[bb][i][rg * RPG + r];
#pragma unroll
                for (int j = 0; j < VPK; ++j) {
                    s[j][r] = fmaf(g[j], s[j][r], kc[r] * delta[j]);
                    o[j]    = fmaf(s[j][r], qr, o[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < VPK; ++j) ro[j][rg][c] = o[j];
            __syncthreads();
            if (rg < VPK)   // row group j writes head j's output
                y[(t * Hv + kh + rg * Hk) * S + col] = (ro[rg][0][c] + ro[rg][1][c] + ro[rg][2][c] + ro[rg][3][c]) * scale;
        }
    }
#pragma unroll
    for (int j = 0; j < VPK; ++j) {
        float4 * base = reinterpret_cast<float4 *>(state + ((size_t) (kh + j * Hk) * S + col) * S + rg * RPG);
#pragma unroll
        for (int r = 0; r < RPG / 4; ++r) base[r] = make_float4(s[j][4 * r], s[j][4 * r + 1], s[j][4 * r + 2], s[j][4 * r + 3]);
    }
}

// one block per token: a warp per q / k head row (L2 norm in place), then the gates
__global__ void gdn_pf_prep_k(float * __restrict__ h, const int64_t C, const float * __restrict__ bsrc,
                              const float * __restrict__ asrc, const int64_t ba_ld, const float * __restrict__ dt,
                              const float * __restrict__ a, float * __restrict__ gate, float * __restrict__ beta,
                              const int Hk, const int Hv, const float eps) {
    const int64_t t    = blockIdx.x;
    const int     lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarp = blockDim.x / 32;
    float *       row  = h + t * C;
    for (int ih = warp; ih < Hk; ih += nwarp) {
        float * xq = row + (int64_t) ih * S;
        float * xk = row + (int64_t) (Hk + ih) * S;
        float   aq = 0.0f, ak = 0.0f;
        for (int i = lane; i < S; i += 32) { aq += xq[i] * xq[i]; ak += xk[i] * xk[i]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            aq += __shfl_xor_sync(0xffffffffu, aq, o);
            ak += __shfl_xor_sync(0xffffffffu, ak, o);
        }
        const float scq = 1.0f / fmaxf(sqrtf(aq), eps), sck = 1.0f / fmaxf(sqrtf(ak), eps);
        for (int i = lane; i < S; i += 32) { xq[i] *= scq; xk[i] *= sck; }
    }
    for (int ih = threadIdx.x; ih < Hv; ih += blockDim.x) {
        const float x  = asrc[t * ba_ld + ih] + dt[ih];
        const float bv = bsrc[t * ba_ld + ih];
        gate[t * Hv + ih] = ((x > 20.0f) ? x : log1pf(expf(x))) * a[ih];
        beta[t * Hv + ih] = 1.0f / (1.0f + expf(-bv));
    }
}

// out[i][c] = silu(sum_j col(i + j) * w[c][j]), col(k) = history[k] for k < K-1 else qkv row k - (K-1)
template <int K>
__global__ void gdn_pf_conv_k(const float * __restrict__ state, const float * __restrict__ qkv, const int64_t ld,
                              const float * __restrict__ w, const int64_t w_ld, float * __restrict__ out, const int C,
                              const int L) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (c >= C) return;
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const int   k = i + j;
        const float x = k < K - 1 ? state[(int64_t) c * (K - 1) + k] : qkv[(int64_t) (k - (K - 1)) * ld + c];
        sumf += x * w[(int64_t) c * w_ld + j];
    }
    out[(int64_t) i * C + c] = sumf / (1.0f + expf(-sumf));
}

template <int K>
__global__ void gdn_pf_conv_state_k(float * __restrict__ state, const float * __restrict__ qkv, const int64_t ld,
                                    const int C, const int L) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
#pragma unroll
    for (int j = 0; j < K - 1; ++j) state[(int64_t) c * (K - 1) + j] = qkv[(int64_t) (L - (K - 1) + j) * ld + c];
}

}  // namespace

namespace ai::kern {

bool q4x_gdn_pf_conv(void * stream, float * state, const float * qkv, int64_t ld, const float * w, int64_t w_ld, int K,
                     float * out, int C, int L) {
    if (K != 4 || L < K - 1) return false;
    const cudaStream_t st = (cudaStream_t) stream;
    gdn_pf_conv_k<4><<<dim3((C + 127) / 128, L), 128, 0, st>>>(state, qkv, ld, w, w_ld, out, C, L);
    gdn_pf_conv_state_k<4><<<(C + 127) / 128, 128, 0, st>>>(state, qkv, ld, C, L);
    return true;
}

bool q4x_gdn_pf_scan(void * stream, float * state, float * h, int64_t C, const float * bsrc, const float * asrc,
                     int64_t ba_ld, const float * dt, const float * a, float * gate, float * beta,
                     float * y, int Sk, int Hk, int Sv, int Hv, int64_t T, float eps) {
    if (Sk != S || Sv != S || Hv != VPK * Hk || T <= 0) return false;
    const cudaStream_t st = (cudaStream_t) stream;
    gdn_pf_prep_k<<<(unsigned) T, 128, 0, st>>>(h, C, bsrc, asrc, ba_ld, dt, a, gate, beta, Hk, Hv, eps);
    gdn_pf_scan_k<<<Hk * NCB, dim3(CB, RG), 0, st>>>(state, h, C, gate, beta, y, Hk, T, 1.0f / sqrtf((float) S));
    return true;
}

}  // namespace ai::kern
