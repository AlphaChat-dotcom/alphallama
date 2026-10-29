// SPDX-License-Identifier: Apache-2.0
// Native chunked Gated DeltaNet prefill with 16-token chunks. Enabled by AI_GDN16.
//
// Replaces ggml's chunked scan (gated_delta_net_chunked.cuh: 64-token chunks, 32-column state slices in shared
// memory, 3xTF32/TF32 WMMA over per-chunk fp32 W / U' / P scratch) for the prefill recurrence.
// The decomposition follows NInfer's chunked GDN (src/ops/linear_attention/gated_delta_net/chunked, Apache-2.0); the
// code is independent and the arithmetic is TF32 throughout (NInfer uses bf16 for the Q.S and state-update products):
//
//   prepare     grid (n_chunks, Hv, B), fully parallel, FP32 CUDA cores. Per chunk of C = 16 tokens with log-gate
//               cumsum G and sigmoid(beta) b:  L_ij = b_i e^(G_i-G_j) k_i.k_j (j < i);  solve = (I + L)^-1 diag(b);
//               mqk_ij = e^(G_i-G_j) q_i.k_j (j <= i);  prefix_i = e^(G_i);  suffix_i = e^(G_C - G_i).
//   recurrence  grid (Hv * B, S_v / DV): each block owns a DV = 16 or 32 column stripe of one head's state S[key][value], held
//               in registers in fp32 (delta-rule columns are independent), sequential over chunks:
//                 P = K S, H = Q S                      (16 x 128 x 16, TF32 MMA)
//                 D = solve (V - prefix o P)           (16 x 16 x 16)
//                 O = scale * (prefix o H + mqk D)     (16 x 16 x 16)
//                 S = prefix_C S + K^T diag(suffix) D   (128 x 16 x 16)
//
// Semantics match ggml's GGML_OP_GATED_DELTA_NET scalar-gate form: a_t = exp(g_t), beta already sigmoid,
// q/k head = h % Hk, scale = 1/sqrt(S_v), state stored transposed as M[value][key] (key contiguous), updated in place.
// S_k == S_v == 128 only.
#include "kernels/kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace ai {
namespace kern {
namespace {

constexpr int G16_C   = 16;    // tokens per chunk
constexpr int G16_D   = 128;   // S_k == S_v
constexpr int G16_KLD = G16_D + 4;    // fp32 row stride of staged Q / K (132: 8 rows x 4 lanes hit 32 banks)

struct G16Ctrl {   // per (seq, head, chunk), 2176 B
    float solve[G16_C * G16_C];
    float mqk[G16_C * G16_C];
    float prefix[G16_C];
    float suffix[G16_C];
};

#ifndef G16_ABL
#define G16_ABL 0   // bench-only timing ablations: 1 no in-loop loads, 2 no P/H MMA, 4 no state update (output wrong)
#endif
#ifndef G16_RAW
#define G16_RAW 0   // 1: feed fp32 bits to the TF32 MMA (hardware truncation) instead of cvt.rna -- bench variant
#endif
__device__ __forceinline__ unsigned g16_tf32(float x) {
#if G16_RAW
    return __float_as_uint(x);
#endif
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ void g16_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void g16_cp16(void * s, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void g16_cp16z(void * s, const void * g, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(s)), "l"(g), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void g16_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void g16_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }

// ---- prepare: one block of 256 threads per (chunk, head, seq)
__global__ void __launch_bounds__(256)
g16_prepare(const float * __restrict__ q, const float * __restrict__ k, int64_t qk_tok, int64_t qk_seq,
            const float * __restrict__ g, const float * __restrict__ beta, int64_t gb_tok, int64_t gb_seq,
            int Hk, int Hv, int L, G16Ctrl * __restrict__ ctrl) {
    __shared__ float qs[G16_C][G16_KLD], ks[G16_C][G16_KLD];
    __shared__ float G[G16_C], bt[G16_C], Lm[G16_C][G16_C + 1], X[G16_C][G16_C + 1];
    const int ch = blockIdx.x, h = blockIdx.y, b = blockIdx.z, tid = threadIdx.x;
    const int t0 = ch * G16_C, n = min(G16_C, L - t0), hq = h % Hk;
    for (int i = tid; i < G16_C * (G16_D / 4); i += 256) {
        const int r = i / (G16_D / 4), c = (i % (G16_D / 4)) * 4;
        float4 qv = make_float4(0.f, 0.f, 0.f, 0.f), kv = qv;
        if (r < n) {
            const int64_t o = (int64_t) b * qk_seq + (int64_t) (t0 + r) * qk_tok + (int64_t) hq * G16_D + c;
            qv = *reinterpret_cast<const float4 *>(q + o);
            kv = *reinterpret_cast<const float4 *>(k + o);
        }
        *reinterpret_cast<float4 *>(&qs[r][c]) = qv;
        *reinterpret_cast<float4 *>(&ks[r][c]) = kv;
    }
    if (tid < 32) {   // one warp: inclusive cumsum of the log gates over the chunk (padding: g = 0, beta = 0)
        const int64_t o = (int64_t) b * gb_seq + (int64_t) (t0 + tid) * gb_tok + h;
        float gv = tid < n ? g[o] : 0.f;
        const float bv = tid < n ? beta[o] : 0.f;
#pragma unroll
        for (int s = 1; s < G16_C; s <<= 1) {
            const float p = __shfl_up_sync(0xffffffffu, gv, s);
            if (tid >= s) gv += p;
        }
        if (tid < G16_C) { G[tid] = gv; bt[tid] = bv; }
    }
    __syncthreads();
    G16Ctrl & C = ctrl[((int64_t) b * Hv + h) * gridDim.x + ch];
    {   // thread (i, j): k_i.k_j and q_i.k_j in fp32
        const int i = tid >> 4, j = tid & 15;
        float kk = 0.f, qk = 0.f;
#pragma unroll 8
        for (int c = 0; c < G16_D; ++c) {
            kk = fmaf(ks[i][c], ks[j][c], kk);
            qk = fmaf(qs[i][c], ks[j][c], qk);
        }
        const float e = j <= i ? expf(G[i] - G[j]) : 0.f;   // never evaluate the upper triangle (g can be very negative)
        Lm[i][j] = j < i ? bt[i] * e * kk : 0.f;
        C.mqk[i * G16_C + j] = e * qk;
    }
    if (tid < G16_C) {
        C.prefix[tid] = expf(G[tid]);
        C.suffix[tid] = expf(G[G16_C - 1] - G[tid]);
    }
    __syncthreads();
    if (tid < G16_C) {   // column tid of (I + L)^-1 by forward substitution
        const int c = tid;
        for (int r = 0; r < G16_C; ++r) {
            float v = r == c ? 1.f : 0.f;
            for (int j = 0; j < r; ++j) v = fmaf(-Lm[r][j], X[j][c], v);
            X[r][c] = v;
        }
    }
    __syncthreads();
    { const int i = tid >> 4, j = tid & 15; C.solve[i * G16_C + j] = X[i][j] * bt[j]; }
}

// ---- recurrence: 2 * DV / 8 warps per (head * seq, DV-column stripe). DV = 16 (4 warps) or 32 (8 warps), chosen so the
// grid fits one wave of resident blocks (e.g. 32 heads x 8 stripes = 256 blocks need >= 128 SMs at 2 blocks per SM).
// Warps 0..NT-1 compute P = K S (and then D), warps NT..2NT-1 compute H = Q S (and then O), NT = DV / 8 n-tiles.
// K, V and the control packet are double-buffered (K is read again by the state update); Q is single-buffered and
// refilled for the next chunk as soon as Q S is done.
template <int DV> struct G16Stage {
    static constexpr int SLD = DV + 8;   // 24 / 40: B-fragment reads [kk + t][n * 8 + g] hit 32 distinct banks
    float k[G16_C][G16_KLD];
    float v[G16_C][SLD];
    G16Ctrl c;
};
template <int DV> struct G16Sm {
    static constexpr int SLD = DV + 8;
    G16Stage<DV> st[2];
    float q[G16_C][G16_KLD];
    float snap[G16_D][SLD];   // state stripe S[key][value-in-stripe]
    float r[G16_C][SLD];      // V - prefix o P, then D in place
};

template <int DV>
__global__ void __launch_bounds__(DV * 8, 2)
g16_recurrence(const float * __restrict__ q, const float * __restrict__ k, int64_t qk_tok, int64_t qk_seq,
               const float * __restrict__ v, int64_t v_tok, int64_t v_seq, const G16Ctrl * __restrict__ ctrl,
               float * __restrict__ state, float * __restrict__ out, int Hk, int Hv, int L, float scale) {
    constexpr int NT = DV / 8, NW = 2 * NT, NTH = 32 * NW;
    extern __shared__ __align__(16) unsigned char g16_raw[];
    G16Sm<DV> & S = *reinterpret_cast<G16Sm<DV> *>(g16_raw);
    const int hb = blockIdx.x, h = hb % Hv, b = hb / Hv, v0 = blockIdx.y * DV;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int hq = h % Hk, nch = (L + G16_C - 1) / G16_C;
    float * st = state + ((int64_t) b * Hv + h) * G16_D * G16_D;   // M[value][key]
    const G16Ctrl * cbase = ctrl + ((int64_t) b * Hv + h) * nch;
    // state tiles: this warp owns n-tile sn and key m-tiles 4 * sm .. 4 * sm + 3
    const int sn = warp % NT, sm = warp / NT;

    auto load_q = [&](int ch) {
        const int t0 = ch * G16_C, n = min(G16_C, L - t0);
        for (int i = tid; i < G16_C * (G16_D / 4); i += NTH) {
            const int r = i / (G16_D / 4), c = (i % (G16_D / 4)) * 4;
            const int64_t o = (int64_t) b * qk_seq + (int64_t) (t0 + min(r, n - 1)) * qk_tok + (int64_t) hq * G16_D + c;
            g16_cp16z(&S.q[r][c], q + o, r < n);
        }
        g16_commit();
    };
    auto load = [&](int ch, int sl) {
        G16Stage<DV> & s = S.st[sl];
        const int t0 = ch * G16_C, n = min(G16_C, L - t0);
        for (int i = tid; i < G16_C * (G16_D / 4); i += NTH) {
            const int r = i / (G16_D / 4), c = (i % (G16_D / 4)) * 4;
            const int64_t o = (int64_t) b * qk_seq + (int64_t) (t0 + min(r, n - 1)) * qk_tok + (int64_t) hq * G16_D + c;
            g16_cp16z(&s.k[r][c], k + o, r < n);
        }
        for (int i = tid; i < G16_C * (DV / 4); i += NTH) {
            const int r = i / (DV / 4), c = (i % (DV / 4)) * 4;
            const int64_t o = (int64_t) b * v_seq + (int64_t) (t0 + min(r, n - 1)) * v_tok + (int64_t) h * G16_D + v0 + c;
            g16_cp16z(&s.v[r][c], v + o, r < n);
        }
        const float4 * src = reinterpret_cast<const float4 *>(cbase + ch);
        float4 *       dst = reinterpret_cast<float4 *>(&s.c);
        for (int i = tid; i < (int) (sizeof(G16Ctrl) / 16); i += NTH) g16_cp16(dst + i, src + i);
        g16_commit();
    };

    float acc[4][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int key = (4 * sm + mi) * 16 + g, val = v0 + sn * 8 + 2 * t;
        acc[mi][0] = st[(int64_t) val * G16_D + key];
        acc[mi][1] = st[(int64_t) (val + 1) * G16_D + key];
        acc[mi][2] = st[(int64_t) val * G16_D + key + 8];
        acc[mi][3] = st[(int64_t) (val + 1) * G16_D + key + 8];
    }
    load(0, 0);
    load_q(0);
    if (nch > 1) load(1, 1); else g16_commit();   // in flight at the top of chunk ch: [q(ch)], stage(ch+1)
    for (int ch = 0; ch < nch; ++ch) {
        const int sl = ch & 1, t0 = ch * G16_C, n = min(G16_C, L - t0);
        G16Stage<DV> & s = S.st[sl];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const int key = (4 * sm + mi) * 16 + g, col = sn * 8 + 2 * t;
            S.snap[key][col] = acc[mi][0];     S.snap[key][col + 1] = acc[mi][1];
            S.snap[key + 8][col] = acc[mi][2]; S.snap[key + 8][col + 1] = acc[mi][3];
        }
        g16_wait<1>();
        __syncthreads();
        // P = K S (warps 0..NT-1) and H = Q S (warps NT..2NT-1), n-tile = warp % NT
        float ph[4] = {0.f, 0.f, 0.f, 0.f};
        const int nt = warp % NT;
        {
            const float (*A)[G16_KLD] = warp < NT ? s.k : S.q;
#pragma unroll
            for (int kk = 0; kk < ((G16_ABL & 2) ? 0 : G16_D); kk += 8) {
                g16_mma(ph, g16_tf32(A[g][kk + t]), g16_tf32(A[g + 8][kk + t]), g16_tf32(A[g][kk + t + 4]),
                        g16_tf32(A[g + 8][kk + t + 4]), g16_tf32(S.snap[kk + t][nt * 8 + g]),
                        g16_tf32(S.snap[kk + t + 4][nt * 8 + g]));
            }
            if (warp < NT) {   // R = V - prefix o P
                const int col = nt * 8 + 2 * t;
                S.r[g][col]         = s.v[g][col] - s.c.prefix[g] * ph[0];
                S.r[g][col + 1]     = s.v[g][col + 1] - s.c.prefix[g] * ph[1];
                S.r[g + 8][col]     = s.v[g + 8][col] - s.c.prefix[g + 8] * ph[2];
                S.r[g + 8][col + 1] = s.v[g + 8][col + 1] - s.c.prefix[g + 8] * ph[3];
            }
        }
        __syncthreads();   // Q S done: refill Q for the next chunk now (its group completes before stage(ch+2)'s)
        if (ch + 1 < nch && !(G16_ABL & 1)) load_q(ch + 1); else g16_commit();
        // D = solve R (warps 0..NT-1), written back over R: each warp reads and writes only its own 8 columns
        if (warp < NT) {
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            const float * M = s.c.solve;
#pragma unroll
            for (int kk = 0; kk < G16_C; kk += 8) {
                g16_mma(d, g16_tf32(M[g * G16_C + kk + t]), g16_tf32(M[(g + 8) * G16_C + kk + t]),
                        g16_tf32(M[g * G16_C + kk + t + 4]), g16_tf32(M[(g + 8) * G16_C + kk + t + 4]),
                        g16_tf32(S.r[kk + t][nt * 8 + g]), g16_tf32(S.r[kk + t + 4][nt * 8 + g]));
            }
            __syncwarp();
            const int col = nt * 8 + 2 * t;
            S.r[g][col] = d[0]; S.r[g][col + 1] = d[1]; S.r[g + 8][col] = d[2]; S.r[g + 8][col + 1] = d[3];
        }
        __syncthreads();
        // O = scale (prefix o H + mqk D) (warps NT..2NT-1)
        if (warp >= NT) {
            ph[0] *= s.c.prefix[g]; ph[1] *= s.c.prefix[g]; ph[2] *= s.c.prefix[g + 8]; ph[3] *= s.c.prefix[g + 8];
            const float * M = s.c.mqk;
#pragma unroll
            for (int kk = 0; kk < G16_C; kk += 8) {
                g16_mma(ph, g16_tf32(M[g * G16_C + kk + t]), g16_tf32(M[(g + 8) * G16_C + kk + t]),
                        g16_tf32(M[g * G16_C + kk + t + 4]), g16_tf32(M[(g + 8) * G16_C + kk + t + 4]),
                        g16_tf32(S.r[kk + t][nt * 8 + g]), g16_tf32(S.r[kk + t + 4][nt * 8 + g]));
            }
            const int col = v0 + nt * 8 + 2 * t;
            const int64_t ro = (int64_t) Hv * G16_D;   // out [S_v, Hv, L, B]
            float * o = out + ((int64_t) b * L + t0) * ro + (int64_t) h * G16_D + col;
            if (g < n)     { o[(int64_t) g * ro] = ph[0] * scale;       o[(int64_t) g * ro + 1] = ph[1] * scale; }
            if (g + 8 < n) { o[(int64_t) (g + 8) * ro] = ph[2] * scale; o[(int64_t) (g + 8) * ro + 1] = ph[3] * scale; }
        }
        // S = prefix_C S + K^T diag(suffix) D (all warps, their own tiles)
        if (!(G16_ABL & 4)) {
            const float decay = s.c.prefix[G16_C - 1];
            unsigned bd[2][2];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                bd[ks][0] = g16_tf32(S.r[ks * 8 + t][sn * 8 + g]);
                bd[ks][1] = g16_tf32(S.r[ks * 8 + t + 4][sn * 8 + g]);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int kb = (4 * sm + mi) * 16;
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) acc[mi][qd] *= decay;
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {   // A[key][token] = K[token][key] * suffix[token]
                    const int kk = ks * 8;
                    g16_mma(acc[mi], g16_tf32(s.k[kk + t][kb + g] * s.c.suffix[kk + t]),
                            g16_tf32(s.k[kk + t][kb + g + 8] * s.c.suffix[kk + t]),
                            g16_tf32(s.k[kk + t + 4][kb + g] * s.c.suffix[kk + t + 4]),
                            g16_tf32(s.k[kk + t + 4][kb + g + 8] * s.c.suffix[kk + t + 4]), bd[ks][0], bd[ks][1]);
                }
            }
        }
        __syncthreads();   // stage sl, snap and r are free
        if (ch + 2 < nch && !(G16_ABL & 1)) load(ch + 2, sl); else g16_commit();
    }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int key = (4 * sm + mi) * 16 + g, val = v0 + sn * 8 + 2 * t;
        st[(int64_t) val * G16_D + key]           = acc[mi][0];
        st[(int64_t) (val + 1) * G16_D + key]     = acc[mi][1];
        st[(int64_t) val * G16_D + key + 8]       = acc[mi][2];
        st[(int64_t) (val + 1) * G16_D + key + 8] = acc[mi][3];
    }
}

template <int DV> void g16_launch_rec(cudaStream_t st, const float * q, const float * k, int64_t qk_tok, int64_t qk_seq,
                                     const float * v, int64_t v_tok, int64_t v_seq, const G16Ctrl * ctrl, float * state,
                                     float * out, int Hk, int Hv, int L, int B, float scale) {
    static bool init = false;
    const int smem = (int) sizeof(G16Sm<DV>);
    if (!init) {
        cudaFuncSetAttribute(g16_recurrence<DV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    g16_recurrence<DV><<<dim3((unsigned) (Hv * B), G16_D / DV), DV * 8, smem, st>>>(q, k, qk_tok, qk_seq, v, v_tok, v_seq,
                                                                                     ctrl, state, out, Hk, Hv, L, scale);
}

// stripe width: the narrowest that keeps the grid within one wave of resident blocks (AI_GDN16_DV=16|32 overrides)
int g16_dv(int64_t Hv, int64_t B) {
    static const int forced = getenv("AI_GDN16_DV") ? atoi(getenv("AI_GDN16_DV")) : 0;
    if (forced == 16 || forced == 32) return forced;
    static int slots16 = -1;
    if (slots16 < 0) {
        int dev = 0, sms = 0, occ = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaFuncSetAttribute(g16_recurrence<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(G16Sm<16>));
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, g16_recurrence<16>, 128, sizeof(G16Sm<16>));
        slots16 = sms * occ;
    }
    return Hv * B * (G16_D / 16) <= slots16 ? 16 : 32;
}

}  // namespace

// Default: on for compute capability 8.x and 12.x, where it was validated (faster prefill, PPL neutral, MTP and plain
// decoding identical). Other architectures are opt-in (AI_GDN16=1).
bool gdn16_enabled() {
    static const bool v = [] {
        if (const char * s = getenv("AI_GDN16")) return atoi(s) != 0;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
        return major == 8 || major == 12;
    }();
    return v;
}

size_t gdn16_scratch_bytes(int64_t Hv, int64_t L, int64_t B) {
    return (size_t) B * Hv * ((L + G16_C - 1) / G16_C) * sizeof(G16Ctrl) + 256;
}

bool gdn16_ok(int64_t Sk, int64_t Sv, int64_t Hk, int64_t Hv, int64_t L) {
    return Sk == G16_D && Sv == G16_D && Hk >= 1 && Hv % Hk == 0 && L >= 1;
}

bool gdn16(void * stream, const float * q, const float * k, int64_t qk_tok, int64_t qk_seq, const float * v,
           int64_t v_tok, int64_t v_seq, const float * g, const float * beta, int64_t gb_tok, int64_t gb_seq,
           float * state, float * out, int64_t Hk, int64_t Hv, int64_t L, int64_t B, float scale, void * scratch) {
    cudaStream_t st = (cudaStream_t) stream;
    const int nch = (int) ((L + G16_C - 1) / G16_C);
    G16Ctrl * ctrl = (G16Ctrl *) (((uintptr_t) scratch + 255) / 256 * 256);
    g16_prepare<<<dim3(nch, (unsigned) Hv, (unsigned) B), 256, 0, st>>>(q, k, qk_tok, qk_seq, g, beta, gb_tok, gb_seq,
                                                                         (int) Hk, (int) Hv, (int) L, ctrl);
    if (g16_dv(Hv, B) == 16)
        g16_launch_rec<16>(st, q, k, qk_tok, qk_seq, v, v_tok, v_seq, ctrl, state, out, (int) Hk, (int) Hv, (int) L, (int) B, scale);
    else
        g16_launch_rec<32>(st, q, k, qk_tok, qk_seq, v, v_tok, v_seq, ctrl, state, out, (int) Hk, (int) Hv, (int) L, (int) B, scale);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
// Same arithmetic as ggml's gdn_prep_kernel (GGML_GDN_FUSE_L2NORM_QK | SIGMOID_BETA | GATE): per token, q / k heads
// scaled by 1 / max(||x||, eps), g = softplus(alpha + dt_bias) * a (linear above 20), beta = sigmoid(beta).
// Input is the unfused path's conv output [C, L, B] (q | k | v per token) and the ba projection [2 Hv, L, B] (beta |
// alpha). Writes q / k [Sk, Hk, L, B], g / beta [Hv, L, B].
__global__ void g16_prep(const float * __restrict__ conv, int64_t C, const float * __restrict__ ba, int64_t ba_tok,
                         const float * __restrict__ dt, const float * __restrict__ a, float * __restrict__ qh,
                         float * __restrict__ kh, float * __restrict__ gh, float * __restrict__ bh, int Hk, int Hv,
                         float eps) {
    const int64_t t = blockIdx.x;   // flat token over L * B
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarp = blockDim.x >> 5;
    const float * xq = conv + t * C;
    const float * xk = xq + (int64_t) G16_D * Hk;
    for (int ih = warp; ih < Hk; ih += nwarp) {
        const float * pq = xq + ih * G16_D;
        const float * pk = xk + ih * G16_D;
        float sq = 0.f, sk = 0.f;
        for (int i = lane; i < G16_D; i += 32) { sq += pq[i] * pq[i]; sk += pk[i] * pk[i]; }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
            sk += __shfl_xor_sync(0xffffffffu, sk, o);
        }
        const float scq = 1.0f / fmaxf(sqrtf(sq), eps), sck = 1.0f / fmaxf(sqrtf(sk), eps);
        float * yq = qh + (t * Hk + ih) * G16_D;
        float * yk = kh + (t * Hk + ih) * G16_D;
        for (int i = lane; i < G16_D; i += 32) { yq[i] = pq[i] * scq; yk[i] = pk[i] * sck; }
    }
    for (int ih = threadIdx.x; ih < Hv; ih += blockDim.x) {
        const float bv = ba[t * ba_tok + ih];
        const float x  = ba[t * ba_tok + Hv + ih] + dt[ih];
        gh[t * Hv + ih] = ((x > 20.0f) ? x : log1pf(expf(x))) * a[ih];
        bh[t * Hv + ih] = 1.0f / (1.0f + expf(-bv));
    }
}
size_t g16_al(size_t v) { return (v + 255) / 256 * 256; }

// ---- v2 (AI_GDN16_V=2): prep fused into prepare, Q / K handed to the recurrence as fp16 packets.
// q / k are L2-normalised (|x| <= 1), so fp16's 10-bit mantissa represents them as finely as the TF32 MMA consumes
// them; prepare's KK^T / QK^T use the same fp16-rounded values the recurrence loads, so the solve and the scan agree.
// One 8 KB packet per (seq, qk head, chunk): the recurrence's per-chunk Q / K load halves and the fp32 q / k round
// trip of v1's separate prep kernel (16 KB per token) disappears.
struct G16Qk {   // per (seq, qk head, chunk)
    __half q[G16_C][G16_D];
    __half k[G16_C][G16_D];
};

__global__ void __launch_bounds__(256)
g16_prepare2(const float * __restrict__ conv, int64_t C, const float * __restrict__ ba, int64_t ba_tok,
             const float * __restrict__ dt, const float * __restrict__ a, float eps, int Hk, int Hv, int L,
             G16Ctrl * __restrict__ ctrl, G16Qk * __restrict__ qkp) {
    __shared__ float qs[G16_C][G16_KLD], ks[G16_C][G16_KLD];
    __shared__ float G[G16_C], bt[G16_C], Lm[G16_C][G16_C + 1], X[G16_C][G16_C + 1];
    const int ch = blockIdx.x, h = blockIdx.y, b = blockIdx.z, tid = threadIdx.x, nch = gridDim.x;
    const int t0 = ch * G16_C, n = min(G16_C, L - t0), hq = h % Hk;
    {   // thread (token r, 8 columns c): L2 norm over the head (16 lanes), fp16-round, keep the rounded value
        const int r = tid >> 4, c = (tid & 15) * 8;
        float qv[8], kv[8];
        if (r < n) {
            const float * row = conv + ((int64_t) b * L + t0 + r) * C;
            const float4 q0 = *reinterpret_cast<const float4 *>(row + hq * G16_D + c);
            const float4 q1 = *reinterpret_cast<const float4 *>(row + hq * G16_D + c + 4);
            const float4 k0 = *reinterpret_cast<const float4 *>(row + (Hk + hq) * G16_D + c);
            const float4 k1 = *reinterpret_cast<const float4 *>(row + (Hk + hq) * G16_D + c + 4);
            qv[0] = q0.x; qv[1] = q0.y; qv[2] = q0.z; qv[3] = q0.w; qv[4] = q1.x; qv[5] = q1.y; qv[6] = q1.z; qv[7] = q1.w;
            kv[0] = k0.x; kv[1] = k0.y; kv[2] = k0.z; kv[3] = k0.w; kv[4] = k1.x; kv[5] = k1.y; kv[6] = k1.z; kv[7] = k1.w;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) { qv[i] = 0.f; kv[i] = 0.f; }
        }
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { sq = fmaf(qv[i], qv[i], sq); sk = fmaf(kv[i], kv[i], sk); }
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            sq += __shfl_xor_sync(0xffffffffu, sq, o);
            sk += __shfl_xor_sync(0xffffffffu, sk, o);
        }
        const float scq = 1.0f / fmaxf(sqrtf(sq), eps), sck = 1.0f / fmaxf(sqrtf(sk), eps);
        __half hqv[8], hkv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            hqv[i] = __float2half_rn(qv[i] * scq);
            hkv[i] = __float2half_rn(kv[i] * sck);
            qs[r][c + i] = __half2float(hqv[i]);
            ks[r][c + i] = __half2float(hkv[i]);
        }
        if (h < Hk) {   // one producer per qk head (v heads h and h + Hk, ... share it)
            G16Qk & P = qkp[((int64_t) b * Hk + hq) * nch + ch];
            *reinterpret_cast<uint4 *>(&P.q[r][c]) = *reinterpret_cast<const uint4 *>(hqv);
            *reinterpret_cast<uint4 *>(&P.k[r][c]) = *reinterpret_cast<const uint4 *>(hkv);
        }
    }
    if (tid < 32) {   // gate g = softplus(alpha + dt) * a (linear above 20), beta = sigmoid; inclusive cumsum of g
        const int64_t o = ((int64_t) b * L + t0 + tid) * ba_tok;
        float gv = 0.f, bv = 0.f;
        if (tid < n) {
            const float x = ba[o + Hv + h] + dt[h];
            gv = ((x > 20.0f) ? x : log1pf(expf(x))) * a[h];
            bv = 1.0f / (1.0f + expf(-ba[o + h]));
        }
#pragma unroll
        for (int s = 1; s < G16_C; s <<= 1) {
            const float p = __shfl_up_sync(0xffffffffu, gv, s);
            if (tid >= s) gv += p;
        }
        if (tid < G16_C) { G[tid] = gv; bt[tid] = bv; }
    }
    __syncthreads();
    G16Ctrl & Cc = ctrl[((int64_t) b * Hv + h) * nch + ch];
    {
        const int i = tid >> 4, j = tid & 15;
        float kk = 0.f, qk = 0.f;
#pragma unroll 8
        for (int c = 0; c < G16_D; ++c) {
            kk = fmaf(ks[i][c], ks[j][c], kk);
            qk = fmaf(qs[i][c], ks[j][c], qk);
        }
        const float e = j <= i ? expf(G[i] - G[j]) : 0.f;
        Lm[i][j] = j < i ? bt[i] * e * kk : 0.f;
        Cc.mqk[i * G16_C + j] = __uint_as_float(g16_tf32(e * qk));   // TF32-rounded once here, not per read
    }
    if (tid < G16_C) {
        Cc.prefix[tid] = expf(G[tid]);
        Cc.suffix[tid] = expf(G[G16_C - 1] - G[tid]);
    }
    __syncthreads();
    if (tid < G16_C) {
        const int c = tid;
        for (int r = 0; r < G16_C; ++r) {
            float v = r == c ? 1.f : 0.f;
            for (int j = 0; j < r; ++j) v = fmaf(-Lm[r][j], X[j][c], v);
            X[r][c] = v;
        }
    }
    __syncthreads();
    { const int i = tid >> 4, j = tid & 15; Cc.solve[i * G16_C + j] = __uint_as_float(g16_tf32(X[i][j] * bt[j])); }
}

constexpr int G16_HLD = G16_D + 8;   // fp16 row stride (272 B): A-fragment reads [g][kk + t] hit 8 distinct bank groups
template <int DV> struct G16Stage2 {
    static constexpr int SLD = DV + 8;
    __half k[G16_C][G16_HLD];
    float  v[G16_C][SLD];
    G16Ctrl c;
};
template <int DV> struct G16Sm2 {
    static constexpr int SLD = DV + 8;
    G16Stage2<DV> st[2];
    __half q[G16_C][G16_HLD];
    float snap[G16_D][SLD];
    float r[G16_C][SLD];
};

template <int DV>
__global__ void __launch_bounds__(DV * 8, 2)
g16_recurrence2(const G16Qk * __restrict__ qkp, const float * __restrict__ v, int64_t v_tok, int64_t v_seq,
                const G16Ctrl * __restrict__ ctrl, float * __restrict__ state, float * __restrict__ out, int Hk, int Hv,
                int L, float scale) {
    constexpr int NT = DV / 8, NW = 2 * NT, NTH = 32 * NW;
    extern __shared__ __align__(16) unsigned char g16_raw2[];
    G16Sm2<DV> & S = *reinterpret_cast<G16Sm2<DV> *>(g16_raw2);
    const int hb = blockIdx.x, h = hb % Hv, b = hb / Hv, v0 = blockIdx.y * DV;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int hq = h % Hk, nch = (L + G16_C - 1) / G16_C;
    float * st = state + ((int64_t) b * Hv + h) * G16_D * G16_D;
    const G16Ctrl * cbase = ctrl + ((int64_t) b * Hv + h) * nch;
    const G16Qk *   pbase = qkp + ((int64_t) b * Hk + hq) * nch;
    const int sn = warp % NT, sm = warp / NT;

    auto load_q = [&](int ch) {
        const G16Qk & P = pbase[ch];
        for (int i = tid; i < G16_C * (G16_D / 8); i += NTH) {
            const int r = i / (G16_D / 8), c = (i % (G16_D / 8)) * 8;
            g16_cp16(&S.q[r][c], &P.q[r][c]);
        }
        g16_commit();
    };
    auto load = [&](int ch, int sl) {
        G16Stage2<DV> & s = S.st[sl];
        const G16Qk & P = pbase[ch];
        const int t0 = ch * G16_C, n = min(G16_C, L - t0);
        for (int i = tid; i < G16_C * (G16_D / 8); i += NTH) {
            const int r = i / (G16_D / 8), c = (i % (G16_D / 8)) * 8;
            g16_cp16(&s.k[r][c], &P.k[r][c]);
        }
        for (int i = tid; i < G16_C * (DV / 4); i += NTH) {
            const int r = i / (DV / 4), c = (i % (DV / 4)) * 4;
            const int64_t o = (int64_t) b * v_seq + (int64_t) (t0 + min(r, n - 1)) * v_tok + (int64_t) h * G16_D + v0 + c;
            g16_cp16z(&s.v[r][c], v + o, r < n);
        }
        const float4 * src = reinterpret_cast<const float4 *>(cbase + ch);
        float4 *       dst = reinterpret_cast<float4 *>(&s.c);
        for (int i = tid; i < (int) (sizeof(G16Ctrl) / 16); i += NTH) g16_cp16(dst + i, src + i);
        g16_commit();
    };

    float acc[4][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int key = (4 * sm + mi) * 16 + g, val = v0 + sn * 8 + 2 * t;
        acc[mi][0] = st[(int64_t) val * G16_D + key];
        acc[mi][1] = st[(int64_t) (val + 1) * G16_D + key];
        acc[mi][2] = st[(int64_t) val * G16_D + key + 8];
        acc[mi][3] = st[(int64_t) (val + 1) * G16_D + key + 8];
    }
    load(0, 0);
    load_q(0);
    if (nch > 1) load(1, 1); else g16_commit();
    for (int ch = 0; ch < nch; ++ch) {
        const int sl = ch & 1, t0 = ch * G16_C, n = min(G16_C, L - t0);
        G16Stage2<DV> & s = S.st[sl];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const int key = (4 * sm + mi) * 16 + g, col = sn * 8 + 2 * t;
            S.snap[key][col] = acc[mi][0];     S.snap[key][col + 1] = acc[mi][1];
            S.snap[key + 8][col] = acc[mi][2]; S.snap[key + 8][col + 1] = acc[mi][3];
        }
        g16_wait<1>();
        __syncthreads();
        float ph[4] = {0.f, 0.f, 0.f, 0.f};
        const int nt = warp % NT;
        {
            const __half (*A)[G16_HLD] = warp < NT ? s.k : S.q;
#pragma unroll
            for (int kk = 0; kk < G16_D; kk += 8) {
                g16_mma(ph, g16_tf32(__half2float(A[g][kk + t])), g16_tf32(__half2float(A[g + 8][kk + t])),
                        g16_tf32(__half2float(A[g][kk + t + 4])), g16_tf32(__half2float(A[g + 8][kk + t + 4])),
                        g16_tf32(S.snap[kk + t][nt * 8 + g]), g16_tf32(S.snap[kk + t + 4][nt * 8 + g]));
            }
            if (warp < NT) {
                const int col = nt * 8 + 2 * t;
                S.r[g][col]         = s.v[g][col] - s.c.prefix[g] * ph[0];
                S.r[g][col + 1]     = s.v[g][col + 1] - s.c.prefix[g] * ph[1];
                S.r[g + 8][col]     = s.v[g + 8][col] - s.c.prefix[g + 8] * ph[2];
                S.r[g + 8][col + 1] = s.v[g + 8][col + 1] - s.c.prefix[g + 8] * ph[3];
            }
        }
        __syncthreads();
        if (ch + 1 < nch) load_q(ch + 1); else g16_commit();
        if (warp < NT) {
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            const float * M = s.c.solve;
#pragma unroll
            for (int kk = 0; kk < G16_C; kk += 8) {
                g16_mma(d, g16_tf32(M[g * G16_C + kk + t]), g16_tf32(M[(g + 8) * G16_C + kk + t]),
                        g16_tf32(M[g * G16_C + kk + t + 4]), g16_tf32(M[(g + 8) * G16_C + kk + t + 4]),
                        g16_tf32(S.r[kk + t][nt * 8 + g]), g16_tf32(S.r[kk + t + 4][nt * 8 + g]));
            }
            __syncwarp();
            const int col = nt * 8 + 2 * t;
            S.r[g][col] = d[0]; S.r[g][col + 1] = d[1]; S.r[g + 8][col] = d[2]; S.r[g + 8][col + 1] = d[3];
        }
        __syncthreads();
        if (warp >= NT) {
            ph[0] *= s.c.prefix[g]; ph[1] *= s.c.prefix[g]; ph[2] *= s.c.prefix[g + 8]; ph[3] *= s.c.prefix[g + 8];
            const float * M = s.c.mqk;
#pragma unroll
            for (int kk = 0; kk < G16_C; kk += 8) {
                g16_mma(ph, g16_tf32(M[g * G16_C + kk + t]), g16_tf32(M[(g + 8) * G16_C + kk + t]),
                        g16_tf32(M[g * G16_C + kk + t + 4]), g16_tf32(M[(g + 8) * G16_C + kk + t + 4]),
                        g16_tf32(S.r[kk + t][nt * 8 + g]), g16_tf32(S.r[kk + t + 4][nt * 8 + g]));
            }
            const int col = v0 + nt * 8 + 2 * t;
            const int64_t ro = (int64_t) Hv * G16_D;
            float * o = out + ((int64_t) b * L + t0) * ro + (int64_t) h * G16_D + col;
            if (g < n)     { o[(int64_t) g * ro] = ph[0] * scale;       o[(int64_t) g * ro + 1] = ph[1] * scale; }
            if (g + 8 < n) { o[(int64_t) (g + 8) * ro] = ph[2] * scale; o[(int64_t) (g + 8) * ro + 1] = ph[3] * scale; }
        }
        {
            const float decay = s.c.prefix[G16_C - 1];
            unsigned bd[2][2];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                bd[ks][0] = g16_tf32(S.r[ks * 8 + t][sn * 8 + g]);
                bd[ks][1] = g16_tf32(S.r[ks * 8 + t + 4][sn * 8 + g]);
            }
            const float su0 = s.c.suffix[t], su1 = s.c.suffix[t + 4], su2 = s.c.suffix[8 + t], su3 = s.c.suffix[12 + t];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int kb = (4 * sm + mi) * 16;
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) acc[mi][qd] *= decay;
                g16_mma(acc[mi], g16_tf32(__half2float(s.k[t][kb + g]) * su0), g16_tf32(__half2float(s.k[t][kb + g + 8]) * su0),
                        g16_tf32(__half2float(s.k[t + 4][kb + g]) * su1), g16_tf32(__half2float(s.k[t + 4][kb + g + 8]) * su1),
                        bd[0][0], bd[0][1]);
                g16_mma(acc[mi], g16_tf32(__half2float(s.k[8 + t][kb + g]) * su2),
                        g16_tf32(__half2float(s.k[8 + t][kb + g + 8]) * su2),
                        g16_tf32(__half2float(s.k[12 + t][kb + g]) * su3),
                        g16_tf32(__half2float(s.k[12 + t][kb + g + 8]) * su3), bd[1][0], bd[1][1]);
            }
        }
        __syncthreads();
        if (ch + 2 < nch) load(ch + 2, sl); else g16_commit();
    }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int key = (4 * sm + mi) * 16 + g, val = v0 + sn * 8 + 2 * t;
        st[(int64_t) val * G16_D + key]           = acc[mi][0];
        st[(int64_t) (val + 1) * G16_D + key]     = acc[mi][1];
        st[(int64_t) val * G16_D + key + 8]       = acc[mi][2];
        st[(int64_t) (val + 1) * G16_D + key + 8] = acc[mi][3];
    }
}

template <int DV> void g16_launch_rec2(cudaStream_t st, const G16Qk * qkp, const float * v, int64_t v_tok,
                                      int64_t v_seq, const G16Ctrl * ctrl, float * state, float * out, int Hk, int Hv,
                                      int L, int B, float scale) {
    static bool init = false;
    const int smem = (int) sizeof(G16Sm2<DV>);
    if (!init) {
        cudaFuncSetAttribute(g16_recurrence2<DV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    g16_recurrence2<DV><<<dim3((unsigned) (Hv * B), G16_D / DV), DV * 8, smem, st>>>(qkp, v, v_tok, v_seq, ctrl, state,
                                                                                      out, Hk, Hv, L, scale);
}
// ---- v3 (AI_GDN16_V=3, default): v2's data layout, fewer instructions per chunk on the serial path.
//   - K / Q A-fragments by ldmatrix (one per 8x8 tile) instead of scalar fp16 loads + 2 conversions each; an fp16
//     value is exact in TF32, so no rounding is needed. ldmatrix hands thread (g, t) the adjacent pair (2t, 2t+1), so
//     the k-slots of the m16n8k8 fragment map to columns as slot t <-> 2t, slot t + 4 <-> 2t + 1, and every B read
//     follows the same permutation (the summation is over k, so any consistent permutation is exact algebra).
//   - the state update takes K^T by ldmatrix.trans and folds diag(suffix) into the D operand once per warp
//     (v2 multiplied K by suffix and rounded per element in every m-tile).
//   - the state snapshot, solve and mqk are TF32-rounded once at write time instead of at every read.
// Numerics: TF32 throughout as in v2; the k order inside each MMA and the rounding point of suffix * (.) differ.
__device__ __forceinline__ void g16_ldsm_x2(unsigned & r0, unsigned & r1, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void g16_ldsm_x4t(unsigned & r0, unsigned & r1, unsigned & r2, unsigned & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"((unsigned) __cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void g16_h2tf(unsigned h2, unsigned & lo, unsigned & hi) {   // half2 -> two exact TF32
    const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h2));
    lo = __float_as_uint(f.x);
    hi = __float_as_uint(f.y);
}
template <int DV> struct G16Sm3 {
    static constexpr int SLD = DV + 4;   // 20 / 36: B reads [kk + 2t (+1)][n * 8 + g] -> word 8t + g: 32 distinct banks
    G16Stage2<DV> st[2];
    __half q[G16_C][G16_HLD];
    float snap[G16_D][SLD];   // TF32-rounded copy of the state stripe
    float r[G16_C][SLD];
};

template <int DV>
__global__ void __launch_bounds__(DV * 8, 2)
g16_recurrence3(const G16Qk * __restrict__ qkp, const float * __restrict__ v, int64_t v_tok, int64_t v_seq,
                const G16Ctrl * __restrict__ ctrl, float * __restrict__ state, float * __restrict__ out, int Hk, int Hv,
                int L, float scale) {
    constexpr int NT = DV / 8, NW = 2 * NT, NTH = 32 * NW;
    extern __shared__ __align__(16) unsigned char g16_raw3[];
    G16Sm3<DV> & S = *reinterpret_cast<G16Sm3<DV> *>(g16_raw3);
    const int hb = blockIdx.x, h = hb % Hv, b = hb / Hv, v0 = blockIdx.y * DV;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int hq = h % Hk, nch = (L + G16_C - 1) / G16_C;
    float * st = state + ((int64_t) b * Hv + h) * G16_D * G16_D;
    const G16Ctrl * cbase = ctrl + ((int64_t) b * Hv + h) * nch;
    const G16Qk *   pbase = qkp + ((int64_t) b * Hk + hq) * nch;
    const int sn = warp % NT, sm = warp / NT;
    // ldmatrix lane addresses: x2 (rows = 16 tokens), x4.trans (tokens x keys: matrix lane / 8 = (token half, key half))
    const int lx2_row = lane & 15;
    const int lx4_tok = ((lane >> 4) & 1) * 8 + (lane & 7), lx4_key = ((lane >> 3) & 1) * 8;

    auto load_q = [&](int ch) {
        const G16Qk & P = pbase[ch];
        for (int i = tid; i < G16_C * (G16_D / 8); i += NTH) {
            const int r = i / (G16_D / 8), c = (i % (G16_D / 8)) * 8;
            g16_cp16(&S.q[r][c], &P.q[r][c]);
        }
        g16_commit();
    };
    auto load = [&](int ch, int sl) {
        G16Stage2<DV> & s = S.st[sl];
        const G16Qk & P = pbase[ch];
        const int t0 = ch * G16_C, n = min(G16_C, L - t0);
        for (int i = tid; i < G16_C * (G16_D / 8); i += NTH) {
            const int r = i / (G16_D / 8), c = (i % (G16_D / 8)) * 8;
            g16_cp16(&s.k[r][c], &P.k[r][c]);
        }
        for (int i = tid; i < G16_C * (DV / 4); i += NTH) {
            const int r = i / (DV / 4), c = (i % (DV / 4)) * 4;
            const int64_t o = (int64_t) b * v_seq + (int64_t) (t0 + min(r, n - 1)) * v_tok + (int64_t) h * G16_D + v0 + c;
            g16_cp16z(&s.v[r][c], v + o, r < n);
        }
        const float4 * src = reinterpret_cast<const float4 *>(cbase + ch);
        float4 *       dst = reinterpret_cast<float4 *>(&s.c);
        for (int i = tid; i < (int) (sizeof(G16Ctrl) / 16); i += NTH) g16_cp16(dst + i, src + i);
        g16_commit();
    };

    float acc[4][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int key = (4 * sm + mi) * 16 + g, val = v0 + sn * 8 + 2 * t;
        acc[mi][0] = st[(int64_t) val * G16_D + key];
        acc[mi][1] = st[(int64_t) (val + 1) * G16_D + key];
        acc[mi][2] = st[(int64_t) val * G16_D + key + 8];
        acc[mi][3] = st[(int64_t) (val + 1) * G16_D + key + 8];
    }
    load(0, 0);
    load_q(0);
    if (nch > 1) load(1, 1); else g16_commit();
    for (int ch = 0; ch < nch; ++ch) {
        const int sl = ch & 1, t0 = ch * G16_C, n = min(G16_C, L - t0);
        G16Stage2<DV> & s = S.st[sl];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const int key = (4 * sm + mi) * 16 + g, col = sn * 8 + 2 * t;
            S.snap[key][col]         = __uint_as_float(g16_tf32(acc[mi][0]));
            S.snap[key][col + 1]     = __uint_as_float(g16_tf32(acc[mi][1]));
            S.snap[key + 8][col]     = __uint_as_float(g16_tf32(acc[mi][2]));
            S.snap[key + 8][col + 1] = __uint_as_float(g16_tf32(acc[mi][3]));
        }
        g16_wait<1>();
        __syncthreads();
        float ph[4] = {0.f, 0.f, 0.f, 0.f};
        const int nt = warp % NT;
        {
            const __half (*A)[G16_HLD] = warp < NT ? s.k : S.q;
#pragma unroll
            for (int kk = 0; kk < G16_D; kk += 8) {
                unsigned r0, r1, a0, a1, a2, a3;
                g16_ldsm_x2(r0, r1, &A[lx2_row][kk]);
                g16_h2tf(r0, a0, a2);   // token g:     slots t, t + 4 <- columns kk + 2t, kk + 2t + 1
                g16_h2tf(r1, a1, a3);   // token g + 8
                g16_mma(ph, a0, a1, a2, a3, __float_as_uint(S.snap[kk + 2 * t][nt * 8 + g]),
                        __float_as_uint(S.snap[kk + 2 * t + 1][nt * 8 + g]));
            }
            if (warp < NT) {
                const int col = nt * 8 + 2 * t;
                S.r[g][col]         = s.v[g][col] - s.c.prefix[g] * ph[0];
                S.r[g][col + 1]     = s.v[g][col + 1] - s.c.prefix[g] * ph[1];
                S.r[g + 8][col]     = s.v[g + 8][col] - s.c.prefix[g + 8] * ph[2];
                S.r[g + 8][col + 1] = s.v[g + 8][col + 1] - s.c.prefix[g + 8] * ph[3];
            }
        }
        __syncthreads();
        if (ch + 1 < nch) load_q(ch + 1); else g16_commit();
        if (warp < NT) {
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            const float * M = s.c.solve;   // TF32-rounded by prepare
#pragma unroll
            for (int kk = 0; kk < G16_C; kk += 8) {
                g16_mma(d, __float_as_uint(M[g * G16_C + kk + t]), __float_as_uint(M[(g + 8) * G16_C + kk + t]),
                        __float_as_uint(M[g * G16_C + kk + t + 4]), __float_as_uint(M[(g + 8) * G16_C + kk + t + 4]),
                        g16_tf32(S.r[kk + t][nt * 8 + g]), g16_tf32(S.r[kk + t + 4][nt * 8 + g]));
            }
            __syncwarp();
            const int col = nt * 8 + 2 * t;
            S.r[g][col] = d[0]; S.r[g][col + 1] = d[1]; S.r[g + 8][col] = d[2]; S.r[g + 8][col + 1] = d[3];
        }
        __syncthreads();
        if (warp >= NT) {
            ph[0] *= s.c.prefix[g]; ph[1] *= s.c.prefix[g]; ph[2] *= s.c.prefix[g + 8]; ph[3] *= s.c.prefix[g + 8];
            const float * M = s.c.mqk;   // TF32-rounded by prepare
#pragma unroll
            for (int kk = 0; kk < G16_C; kk += 8) {
                g16_mma(ph, __float_as_uint(M[g * G16_C + kk + t]), __float_as_uint(M[(g + 8) * G16_C + kk + t]),
                        __float_as_uint(M[g * G16_C + kk + t + 4]), __float_as_uint(M[(g + 8) * G16_C + kk + t + 4]),
                        g16_tf32(S.r[kk + t][nt * 8 + g]), g16_tf32(S.r[kk + t + 4][nt * 8 + g]));
            }
            const int col = v0 + nt * 8 + 2 * t;
            const int64_t ro = (int64_t) Hv * G16_D;
            float * o = out + ((int64_t) b * L + t0) * ro + (int64_t) h * G16_D + col;
            if (g < n)     { o[(int64_t) g * ro] = ph[0] * scale;       o[(int64_t) g * ro + 1] = ph[1] * scale; }
            if (g + 8 < n) { o[(int64_t) (g + 8) * ro] = ph[2] * scale; o[(int64_t) (g + 8) * ro + 1] = ph[3] * scale; }
        }
        {   // S = prefix_C S + K^T (diag(suffix) D): suffix folded into the B operand once per warp
            const float decay = s.c.prefix[G16_C - 1];
            unsigned bd[2][2];
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {   // slots t, t + 4 <- tokens 8 ks + 2t, 8 ks + 2t + 1 (ldmatrix.trans order)
                const int t0r = ks * 8 + 2 * t;
                bd[ks][0] = g16_tf32(s.c.suffix[t0r] * S.r[t0r][sn * 8 + g]);
                bd[ks][1] = g16_tf32(s.c.suffix[t0r + 1] * S.r[t0r + 1][sn * 8 + g]);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                const int kb = (4 * sm + mi) * 16;
#pragma unroll
                for (int qd = 0; qd < 4; ++qd) acc[mi][qd] *= decay;
                unsigned m0, m1, m2, m3;   // (tokens 0-7, keys lo), (0-7, hi), (8-15, lo), (8-15, hi)
                g16_ldsm_x4t(m0, m1, m2, m3, &s.k[lx4_tok][kb + lx4_key]);
                unsigned a0, a1, a2, a3;
                g16_h2tf(m0, a0, a2);
                g16_h2tf(m1, a1, a3);
                g16_mma(acc[mi], a0, a1, a2, a3, bd[0][0], bd[0][1]);
                g16_h2tf(m2, a0, a2);
                g16_h2tf(m3, a1, a3);
                g16_mma(acc[mi], a0, a1, a2, a3, bd[1][0], bd[1][1]);
            }
        }
        __syncthreads();
        if (ch + 2 < nch) load(ch + 2, sl); else g16_commit();
    }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int key = (4 * sm + mi) * 16 + g, val = v0 + sn * 8 + 2 * t;
        st[(int64_t) val * G16_D + key]           = acc[mi][0];
        st[(int64_t) (val + 1) * G16_D + key]     = acc[mi][1];
        st[(int64_t) val * G16_D + key + 8]       = acc[mi][2];
        st[(int64_t) (val + 1) * G16_D + key + 8] = acc[mi][3];
    }
}

template <int DV> void g16_launch_rec3(cudaStream_t st, const G16Qk * qkp, const float * v, int64_t v_tok,
                                      int64_t v_seq, const G16Ctrl * ctrl, float * state, float * out, int Hk, int Hv,
                                      int L, int B, float scale) {
    static bool init = false;
    const int smem = (int) sizeof(G16Sm3<DV>);
    if (!init) {
        cudaFuncSetAttribute(g16_recurrence3<DV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        init = true;
    }
    g16_recurrence3<DV><<<dim3((unsigned) (Hv * B), G16_D / DV), DV * 8, smem, st>>>(qkp, v, v_tok, v_seq, ctrl, state,
                                                                                      out, Hk, Hv, L, scale);
}

int g16_version() {
    // AI_GDN16_V selects the variant (1, 2, 3); default 3, which has the same error class as v2 with a faster kernel.
    static const int v = getenv("AI_GDN16_V") ? atoi(getenv("AI_GDN16_V")) : 3;
    return v;
}
}  // namespace

size_t gdn16_prefill_scratch_bytes(int64_t Hk, int64_t Hv, int64_t L, int64_t B) {   // the larger of v1 / v2
    const int64_t R = L * B;
    return 2 * g16_al((size_t) R * Hk * G16_D * 4) + 2 * g16_al((size_t) R * Hv * 4) + gdn16_scratch_bytes(Hv, L, B) + 256;
}

bool gdn16_prefill(void * stream, const float * conv_out, int64_t C, const float * ba, int64_t ba_tok,
                   const float * dt_bias, const float * ssm_a, float eps, float * state, float * out, int64_t Hk,
                   int64_t Hv, int64_t L, int64_t B, void * scratch) {
    cudaStream_t st = (cudaStream_t) stream;
    const int64_t R = L * B;
    char * p = (char *) (((uintptr_t) scratch + 255) / 256 * 256);
    if (g16_version() >= 2) {
        const int nch = (int) ((L + G16_C - 1) / G16_C);
        G16Ctrl * ctrl = (G16Ctrl *) p;
        G16Qk *   qkp  = (G16Qk *) (p + g16_al((size_t) B * Hv * nch * sizeof(G16Ctrl)));
        g16_prepare2<<<dim3(nch, (unsigned) Hv, (unsigned) B), 256, 0, st>>>(conv_out, C, ba, ba_tok, dt_bias, ssm_a, eps,
                                                                              (int) Hk, (int) Hv, (int) L, ctrl, qkp);
        const float * v = conv_out + 2 * Hk * G16_D;
        const float sc = 1.0f / sqrtf((float) G16_D);
        const bool d16 = g16_dv(Hv, B) == 16;
        if (g16_version() == 3) {
            if (d16) g16_launch_rec3<16>(st, qkp, v, C, C * L, ctrl, state, out, (int) Hk, (int) Hv, (int) L, (int) B, sc);
            else     g16_launch_rec3<32>(st, qkp, v, C, C * L, ctrl, state, out, (int) Hk, (int) Hv, (int) L, (int) B, sc);
        } else {
            if (d16) g16_launch_rec2<16>(st, qkp, v, C, C * L, ctrl, state, out, (int) Hk, (int) Hv, (int) L, (int) B, sc);
            else     g16_launch_rec2<32>(st, qkp, v, C, C * L, ctrl, state, out, (int) Hk, (int) Hv, (int) L, (int) B, sc);
        }
        return cudaGetLastError() == cudaSuccess;
    }
    float * qh = (float *) p; p += g16_al((size_t) R * Hk * G16_D * 4);
    float * kh = (float *) p; p += g16_al((size_t) R * Hk * G16_D * 4);
    float * gh = (float *) p; p += g16_al((size_t) R * Hv * 4);
    float * bh = (float *) p; p += g16_al((size_t) R * Hv * 4);
    g16_prep<<<(unsigned) R, 128, 0, st>>>(conv_out, C, ba, ba_tok, dt_bias, ssm_a, qh, kh, gh, bh, (int) Hk, (int) Hv, eps);
    return gdn16(stream, qh, kh, Hk * G16_D, Hk * G16_D * L, conv_out + 2 * Hk * G16_D, C, C * L, gh, bh, Hv, Hv * L,
                 state, out, Hk, Hv, L, B, 1.0f / sqrtf((float) G16_D), p);
}

}  // namespace kern
}  // namespace ai
