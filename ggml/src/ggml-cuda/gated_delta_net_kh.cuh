// Token-serial prefill kernel for GGML_OP_GATED_DELTA_NET (scalar gate, S_k == S_v == 128), enabled by ALPHA_GDN_KH.
// Organised around the key head: the R = H / H_k value heads that share a key head (h % H_k) are computed by the same
// threads, so every q / k value staged into shared memory feeds R heads, and a block owns a 32-column slice of all R
// states in registers (4 row groups x 32 rows per thread). The chunked tensor-core form (gated_delta_net_chunked.cuh)
// spends most of its time in the per-chunk scan and the intra-chunk solves; at prefill length the plain recurrence with
// the state in registers and the inputs streamed in token blocks by cp.async is roughly 2x faster per token-layer.
// Output matches the sequential kernel to ~1e-8.
//
// Per token and value head (FP32, the same algebra as gated_delta_net_cuda):
//   kv[c]   = sum_i S[i][c] k[i]                       (row-group partials summed rg 0..3 in order)
//   d[c]    = (v[c] - exp(g) kv[c]) beta
//   S[i][c] = exp(g) S[i][c] + k[i] d[c]
//   o[c]    = scale * sum_i S[i][c] q[i]
// The state is stored transposed (M[c][i] = S[i][c]), as the other kernels store it.
#pragma once

#include "common.cuh"

namespace gdn_kh {

constexpr int S  = 128;   // S_k == S_v
constexpr int CB = 32;    // value columns per block
constexpr int RG = 4;     // row groups
constexpr int RPG = S / RG;
constexpr int TB = 16;    // tokens per staged block

static __device__ __forceinline__ void cp4(float * smem, const float * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *smem = *gmem;
#endif
}
static __device__ __forceinline__ void cp16(float * smem, const float * gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *(float4 *) smem = *(const float4 *) gmem;
#endif
}
static __device__ __forceinline__ void cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
static __device__ __forceinline__ void cp_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

// per staged token: q[S], k[S], v[R][CB], g[R], b[R]
template <int R> struct layout {
    static constexpr int Q = 0, K = S, V = 2 * S, G = 2 * S + R * CB, B = G + R;
    static constexpr int PER_TOK = B + R;
    static constexpr int PER_TOK_PAD = (PER_TOK + 3) & ~3;   // keep q / k 16-byte aligned for cp16
};

template <int R>
__global__ void __launch_bounds__(CB * RG, 1)
gdn_kh_kernel(const float * __restrict__ q, const float * __restrict__ k, const float * __restrict__ v,
              const float * __restrict__ g, const float * __restrict__ b,
              const float * __restrict__ curr_state, float * __restrict__ dst, float * __restrict__ state_out,
              const int H, const int H_k, const int n_tokens,
              const int64_t sq1, const int64_t sq2, const int64_t sq3,
              const int64_t sv1, const int64_t sv2, const int64_t sv3,
              const int64_t sb1, const int64_t sb2, const int64_t sb3,
              const int rq3, const float scale) {
    using LY = layout<R>;
    extern __shared__ float smem[];   // [2][TB][PER_TOK_PAD] staging + reduction arrays
    float * stage   = smem;
    float * red_kv  = smem + 2 * TB * LY::PER_TOK_PAD;   // [R][RG][CB]
    float * red_o   = red_kv + R * RG * CB;               // [R][RG][CB]

    constexpr int NCB = S / CB;
    const int kh   = blockIdx.x / NCB;
    const int cb   = blockIdx.x % NCB;
    const int seq  = blockIdx.y;
    const int c    = threadIdx.x;
    const int rg   = threadIdx.y;
    const int tid  = rg * CB + c;
    const int col  = cb * CB + c;
    const int iq3  = seq / rq3;

    const int64_t head_state = (int64_t) S * S;
    float s[R][RPG];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const int h = kh + r * H_k;
        const float * src = curr_state + ((int64_t) seq * H + h) * head_state + (int64_t) col * S + rg * RPG;
#pragma unroll
        for (int j = 0; j < RPG; j += 4) {
            const float4 x = *(const float4 *) (src + j);
            s[r][j] = x.x; s[r][j + 1] = x.y; s[r][j + 2] = x.z; s[r][j + 3] = x.w;
        }
    }

    // stage tokens [t0, t0 + TB) into buffer buf
    auto stage_block = [&](int t0, int buf) {
        float * st = stage + buf * TB * LY::PER_TOK_PAD;
        const int nt = min(TB, n_tokens - t0);
        // q and k rows: 2 x S floats per token = 2 x 32 16-byte chunks
        for (int i = tid; i < nt * 2 * (S / 4); i += CB * RG) {
            const int tt = i / (2 * (S / 4));
            const int w  = i % (2 * (S / 4));
            const int which = w / (S / 4), off = (w % (S / 4)) * 4;
            const float * gsrc = (which ? k : q) + iq3 * sq3 + (int64_t) (t0 + tt) * sq2 + (int64_t) kh * sq1 + off;
            cp16(st + tt * LY::PER_TOK_PAD + (which ? LY::K : LY::Q) + off, gsrc);
        }
        // v columns of the R heads, gates, betas
        for (int i = tid; i < nt * R * CB; i += CB * RG) {
            const int tt = i / (R * CB), w = i % (R * CB), r = w / CB, cc = w % CB;
            const int h = kh + r * H_k;
            cp4(st + tt * LY::PER_TOK_PAD + LY::V + r * CB + cc,
                v + (int64_t) seq * sv3 + (int64_t) (t0 + tt) * sv2 + (int64_t) h * sv1 + cb * CB + cc);
        }
        for (int i = tid; i < nt * 2 * R; i += CB * RG) {
            const int tt = i / (2 * R), w = i % (2 * R), which = w / R, r = w % R;
            const int h = kh + r * H_k;
            const int64_t o = (int64_t) seq * sb3 + (int64_t) (t0 + tt) * sb2 + (int64_t) h * sb1;
            cp4(st + tt * LY::PER_TOK_PAD + (which ? LY::B : LY::G) + r, (which ? b : g) + o);
        }
        cp_commit();
    };

    const int64_t out_tok = (int64_t) S * H;
    float * out_base = dst + (int64_t) seq * n_tokens * out_tok + col;

    if (n_tokens > 0) stage_block(0, 0);
    for (int t0 = 0, blk = 0; t0 < n_tokens; t0 += TB, blk++) {
        cp_wait_all();
        __syncthreads();                                   // block blk landed; everyone is done with buffer blk-1
        if (t0 + TB < n_tokens) stage_block(t0 + TB, (blk + 1) & 1);
        const float * st = stage + (blk & 1) * TB * LY::PER_TOK_PAD;
        const int nt = min(TB, n_tokens - t0);
        for (int tt = 0; tt < nt; tt++) {
            const float * tk = st + tt * LY::PER_TOK_PAD;
            const float * sk = tk + LY::K + rg * RPG;
            const float * sq = tk + LY::Q + rg * RPG;
            float kreg[RPG], qreg[RPG];
#pragma unroll
            for (int j = 0; j < RPG; j += 4) {
                const float4 xk = *(const float4 *) (sk + j), xq = *(const float4 *) (sq + j);
                kreg[j] = xk.x; kreg[j + 1] = xk.y; kreg[j + 2] = xk.z; kreg[j + 3] = xk.w;
                qreg[j] = xq.x; qreg[j + 1] = xq.y; qreg[j + 2] = xq.z; qreg[j + 3] = xq.w;
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;   // 4 chains: a single 32-long FMA chain is latency-bound
#pragma unroll
                for (int j = 0; j < RPG; j += 4) {
                    a0 = fmaf(s[r][j], kreg[j], a0); a1 = fmaf(s[r][j + 1], kreg[j + 1], a1);
                    a2 = fmaf(s[r][j + 2], kreg[j + 2], a2); a3 = fmaf(s[r][j + 3], kreg[j + 3], a3);
                }
                red_kv[(r * RG + rg) * CB + c] = (a0 + a1) + (a2 + a3);
            }
            __syncthreads();
#pragma unroll
            for (int r = 0; r < R; r++) {
                const float kv_col = red_kv[(r * RG + 0) * CB + c] + red_kv[(r * RG + 1) * CB + c] +
                                     red_kv[(r * RG + 2) * CB + c] + red_kv[(r * RG + 3) * CB + c];
                const float gv = expf(tk[LY::G + r]);
                const float d  = (tk[LY::V + r * CB + c] - gv * kv_col) * tk[LY::B + r];
                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
#pragma unroll
                for (int j = 0; j < RPG; j += 4) {
                    s[r][j]     = fmaf(gv, s[r][j],     kreg[j]     * d); o0 = fmaf(s[r][j],     qreg[j],     o0);
                    s[r][j + 1] = fmaf(gv, s[r][j + 1], kreg[j + 1] * d); o1 = fmaf(s[r][j + 1], qreg[j + 1], o1);
                    s[r][j + 2] = fmaf(gv, s[r][j + 2], kreg[j + 2] * d); o2 = fmaf(s[r][j + 2], qreg[j + 2], o2);
                    s[r][j + 3] = fmaf(gv, s[r][j + 3], kreg[j + 3] * d); o3 = fmaf(s[r][j + 3], qreg[j + 3], o3);
                }
                red_o[(r * RG + rg) * CB + c] = (o0 + o1) + (o2 + o3);
            }
            __syncthreads();
            if (rg == 0) {
                float * od = out_base + (int64_t) (t0 + tt) * out_tok;
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const int h = kh + r * H_k;
                    od[(int64_t) h * S] = (red_o[(r * RG + 0) * CB + c] + red_o[(r * RG + 1) * CB + c] +
                                           red_o[(r * RG + 2) * CB + c] + red_o[(r * RG + 3) * CB + c]) * scale;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        const int h = kh + r * H_k;
        float * d = state_out + ((int64_t) seq * H + h) * head_state + (int64_t) col * S + rg * RPG;
#pragma unroll
        for (int j = 0; j < RPG; j += 4) *(float4 *) (d + j) = make_float4(s[r][j], s[r][j + 1], s[r][j + 2], s[r][j + 3]);
    }
}

// true when launched; false = the caller takes another path
static bool launch(const float * q, const float * k, const float * v, const float * g, const float * b,
                   const float * curr_state, float * dst, float * state_out_override,
                   int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_seqs,
                   int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
                   int64_t sb1, int64_t sb2, int64_t sb3, int64_t rq3, float scale, cudaStream_t stream) {
    if (S_v != S || H_k <= 0 || H % H_k != 0) return false;
    const int R = (int) (H / H_k);
    if (R < 1 || R > 4) return false;
    if ((sq1 | sq2 | sq3) & 3) return false;                  // 16-byte q / k rows
    if (((uintptr_t) q | (uintptr_t) k) & 15) return false;
    if (((uintptr_t) curr_state & 15)) return false;
    float * state_out = state_out_override ? state_out_override : dst + S_v * H * n_tokens * n_seqs;
    if (((uintptr_t) state_out & 15)) return false;
    const dim3 grid((unsigned) (H_k * (S / CB)), (unsigned) n_seqs);
    const dim3 block(CB, RG);
    auto go = [&](auto kern, int per_tok_pad, int r) {
        const size_t smem = sizeof(float) * (2 * TB * per_tok_pad + 2 * r * RG * CB);
        static bool attr_set[5] = {};
        if (!attr_set[r]) {
            CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
            attr_set[r] = true;
        }
        kern<<<grid, block, smem, stream>>>(q, k, v, g, b, curr_state, dst, state_out, (int) H, (int) H_k, (int) n_tokens,
                                            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, (int) rq3, scale);
    };
    switch (R) {
        case 1: go(gdn_kh_kernel<1>, layout<1>::PER_TOK_PAD, 1); break;
        case 2: go(gdn_kh_kernel<2>, layout<2>::PER_TOK_PAD, 2); break;
        case 3: go(gdn_kh_kernel<3>, layout<3>::PER_TOK_PAD, 3); break;
        case 4: go(gdn_kh_kernel<4>, layout<4>::PER_TOK_PAD, 4); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

}   // namespace gdn_kh
