// Chunked (parallel-in-time) prefill path for GGML_OP_GATED_DELTA_NET, scalar-gate (non-KDA) only.
//
// The sequential kernel in gated_delta_net.cu runs the recurrence one token at a time inside the
// kernel (n_tokens is not in the grid), so prefill is a latency chain of ~n_tokens dependent
// steps per head: two warp reductions per token, single-digit % of FP32 peak. This is the
// chunkwise form of the gated delta rule (Yang et al. 2024, "Gated Delta Networks";
// flash-linear-attention's chunk_gated_delta_rule), which turns the intra-chunk work into dense
// C x C / C x S_v products and leaves only a per-chunk dependency.
//
// Per head and chunk of C tokens with entering state S0 (S_k x S_v, S_k == S_v here),
// a_t = exp(g_t), gamma_i = prod_{l<=i} a_l (within the chunk), beta_i in (0,1):
//
//   sequential:  S_t = a_t S_{t-1};  u_t = beta_t (v_t - S_t^T k_t);  S_t += k_t u_t^T;
//                o_t = scale * S_t^T q_t
//   unrolled:    (I + A) U = beta o V - diag(beta gamma) K S0,   A[i][j] = beta_i (k_i.k_j) gamma_i/gamma_j  (j < i)
//                U   = U' - W S0,      U' = T (beta o V),  W = T diag(beta gamma) K,  T = (I + A)^-1
//                O   = scale * [ diag(gamma) Q S0 + (Q K^T o M) U ],   M[i][j] = gamma_i/gamma_j  (j <= i)
//                S_C = gamma_C S0 + (diag(gamma_C/gamma) K)^T U
//
// Every ratio gamma_i/gamma_j is formed as exp(gc_i - gc_j) from the log-space cumsum gc, never as a
// quotient of two possibly-underflowed products. Two kernels:
//
//   gdn_chunk_intra   grid (n_chunks, H, n_seqs)   the S0-independent part: A = K K^T, the unit-lower
//                     triangular solve for U' and W (blocked: 16-row diagonal solves in FP32, the
//                     off-diagonal updates on tensor cores), P = Q K^T o M, and gc.  Fully parallel.
//   gdn_chunk_scan    grid (H, n_seqs, S_v/32)     sequential over chunks; each block owns a 32-column
//                     slice of the state in shared memory (the S0-dependent part is column-separable
//                     in the value dimension), and each chunk is three tensor-core products:
//                     W S0 (C x S_v x 32), (gamma Q) S0 + P U, and K_dec^T U (S_v x C x 32).
//
// Chunks are processed in batches of GDN_CHUNK_BATCH so the scratch buffer is bounded, and the intra
// kernel of batch b+1 runs on a second stream while the scan of batch b runs (double-buffered
// scratch, fork/join through events), so the two kernels' costs overlap instead of adding.
//
// Arithmetic: the products run on tensor cores as 3xTF32 (each FP32 operand split into a TF32 hi
// part and a TF32 residual; hi*hi + hi*lo + lo*hi with FP32 accumulate), which is ~FP32 accuracy at
// tensor-core speed; ALPHA_GDN_TF32_1X=1 selects plain TF32 (bf16-class product rounding, FP32
// state). Requires sm_80+; the host dispatcher falls back to the sequential kernel below that.
// The chunked form does not reduce FLOPs (the recurrence is O(d^2) per token either way); with plain
// FP32 FMAs it is LSU-bound (~1 load per FMA) and no faster than the sequential kernel. It only makes
// the work GEMM-shaped, which is what tensor cores need.
//
// Knobs (all read once): ALPHA_GDN_CHUNK_MIN (dispatch threshold, in gated_delta_net.cu),
// ALPHA_GDN_OVERLAP (default 1), ALPHA_GDN_FULLSTAGE (default 0: quarter stages, 2 blocks/SM;
// 1: stage a chunk's Q / K once per step, 1 block/SM, slower), ALPHA_GDN_TF32_1X
// (fails some op cases vs the CPU reference; diagnostic only), ALPHA_GDN_PROF (serial
// event timing of the two kernels), ALPHA_GDN_DBG (phase-ablation bits; output is wrong when set).

#pragma once

#include "common.cuh"

#include <cstdio>
#include <cstdlib>

// TF32 WMMA exists from sm_80 (the kernels only use it there); nvcuda::wmma is not even declared for the
// pre-Volta PTX targets of the default arch list (50/61-virtual), so declare it for the host pass and sm_80+ only.
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#include <mma.h>
#define GDN_HAVE_WMMA 1
namespace gdn_wmma = nvcuda::wmma;
#else
#define GDN_HAVE_WMMA 0
#endif

#define GDN_CHUNK       64
#define GDN_CHUNK_BATCH 8
#define GDN_COLS        32

// per (seq, head, chunk) scratch: U' (C*S_v) | W (C*S_v) | P (C*C) | gc (C)
template <int S_v, int C>
struct gdn_scratch_layout {
    static constexpr int up  = 0;
    static constexpr int w   = C * S_v;
    static constexpr int p   = 2 * C * S_v;
    static constexpr int gc  = 2 * C * S_v + C * C;
    static constexpr int total = 2 * C * S_v + C * C + C;
};

// intra: Kh [C][64+4] (a 64-wide half of K along the head dim) | As [C][C+4] | gc [C] | beta [C]
//   35 KB, so two blocks share an SM (a [C][S_v+4] full-K stage would be ~53 KB -> one block/SM,
//   which is latency-bound).
template <int S_v, int C>
static constexpr size_t gdn_intra_smem_bytes() {
    return (size_t) (C * (64 + 4) + C * (C + 4) + 2 * C) * sizeof(float);
}

// scan: S [S_v][CB+4] | Us [C][CB+4] | stg [C][CB+4] | stage | gam [C] | dec [C] | gamC [8]
//   stage = [C][S_v+4] (FULLSTAGE) or [C][CB+4] which also holds [16][S_v+4]
template <int S_v, int C, int CB, bool FULLSTAGE>
static constexpr size_t gdn_scan_smem_bytes() {
    return (size_t) (S_v * (CB + 4) + 2 * C * (CB + 4) + (FULLSTAGE ? C * (S_v + 4) : C * (CB + 4)) + 2 * C + 8) * sizeof(float);
}

#define GDN_DBG_TF32_1X 128   // outside the WMMA block: the host launcher reads it on every arch pass

#if GDN_HAVE_WMMA
using gdn_frag_c     = gdn_wmma::fragment<gdn_wmma::accumulator, 16, 16, 8, float>;
using gdn_frag_a_row = gdn_wmma::fragment<gdn_wmma::matrix_a, 16, 16, 8, gdn_wmma::precision::tf32, gdn_wmma::row_major>;
using gdn_frag_a_col = gdn_wmma::fragment<gdn_wmma::matrix_a, 16, 16, 8, gdn_wmma::precision::tf32, gdn_wmma::col_major>;
using gdn_frag_b_row = gdn_wmma::fragment<gdn_wmma::matrix_b, 16, 16, 8, gdn_wmma::precision::tf32, gdn_wmma::row_major>;
using gdn_frag_b_col = gdn_wmma::fragment<gdn_wmma::matrix_b, 16, 16, 8, gdn_wmma::precision::tf32, gdn_wmma::col_major>;

// acc += a * b with the 3xTF32 split (hi*hi + hi*lo + lo*hi; the residual of a TF32 rounding is
// exactly representable in FP32, so a - hi is exact). Small terms first, big term last.
template <typename FA, typename FB>
static __device__ __forceinline__ void gdn_mma_3xtf32(gdn_frag_c & acc, const FA & a, const FB & b) {
    FA ah, al;
    FB bh, bl;
#pragma unroll
    for (int e = 0; e < FA::num_elements; e++) {
        const float hi = gdn_wmma::__float_to_tf32(a.x[e]);
        ah.x[e] = hi;
        al.x[e] = gdn_wmma::__float_to_tf32(a.x[e] - hi);
    }
#pragma unroll
    for (int e = 0; e < FB::num_elements; e++) {
        const float hi = gdn_wmma::__float_to_tf32(b.x[e]);
        bh.x[e] = hi;
        bl.x[e] = gdn_wmma::__float_to_tf32(b.x[e] - hi);
    }
    gdn_wmma::mma_sync(acc, al, bh, acc);
    gdn_wmma::mma_sync(acc, ah, bl, acc);
    gdn_wmma::mma_sync(acc, ah, bh, acc);
}

// plain TF32
template <typename FA, typename FB>
static __device__ __forceinline__ void gdn_mma_1xtf32(gdn_frag_c & acc, const FA & a, const FB & b) {
    FA ah;
    FB bh;
#pragma unroll
    for (int e = 0; e < FA::num_elements; e++) {
        ah.x[e] = gdn_wmma::__float_to_tf32(a.x[e]);
    }
#pragma unroll
    for (int e = 0; e < FB::num_elements; e++) {
        bh.x[e] = gdn_wmma::__float_to_tf32(b.x[e]);
    }
    gdn_wmma::mma_sync(acc, ah, bh, acc);
}


template <typename FA, typename FB>
static __device__ __forceinline__ void gdn_mma(gdn_frag_c & acc, const FA & a, const FB & b, const int dbg) {
    if (dbg & GDN_DBG_TF32_1X) {
        gdn_mma_1xtf32(acc, a, b);
    } else {
        gdn_mma_3xtf32(acc, a, b);
    }
}
#endif // GDN_HAVE_WMMA

// ---------------------------------------------------------------------------------------------
// kernel 1: intra-chunk, S0-independent
// ---------------------------------------------------------------------------------------------
template <int S_v, int C>
__global__ void __launch_bounds__(256, 2)
gdn_chunk_intra(const float * __restrict__ q,
                const float * __restrict__ k,
                const float * __restrict__ v,
                const float * __restrict__ g,
                const float * __restrict__ beta,
                float *       __restrict__ scratch,
                const int64_t H,
                const int64_t n_tokens,
                const int64_t chunk0,        // first chunk index of this batch
                const int64_t n_chunks_batch,
                const int64_t sq1, const int64_t sq2, const int64_t sq3,
                const int64_t sv1, const int64_t sv2, const int64_t sv3,
                const int64_t sb1, const int64_t sb2, const int64_t sb3,
                const uint3   neqk1_magic,
                const uint3   rq3_magic,
                const int     dbg) {         // ALPHA_GDN_DBG ablation bits (timing only; output is wrong when set)
#if GDN_HAVE_WMMA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using L = gdn_scratch_layout<S_v, C>;
    constexpr int NT   = 256;
    constexpr int HLD  = 64 + 4;     // row stride of a 64-wide K / Q half-stage
    constexpr int ALD  = C + 4;      // As row stride (16-row tile starts stay 32-byte aligned: 16*68*4 = 4352)
    constexpr int NTX  = 2 * S_v / 16;   // n-tiles across the [U' | W] solve columns (16 / 8)
    constexpr int TPWX = (NTX + 7) / 8;  // per warp (2 / 1)
    static_assert(C == 64, "the tile bookkeeping below assumes a 64-token chunk");
    static_assert(S_v % 64 == 0, "head size must be a multiple of 64");
    static_assert(HLD == ALD, "the Q half-stage reuses the As region with the same stride");

    const int chunk_local = blockIdx.x;
    const int chunk       = (int) chunk0 + chunk_local;
    const int h           = blockIdx.y;
    const int seq         = blockIdx.z;
    const int tid         = threadIdx.x;
    const int warp        = tid / 32;

    const int64_t t0    = (int64_t) chunk * C;
    const int     C_eff = (int) min((int64_t) C, n_tokens - t0);

    const uint32_t iq1 = fastmodulo((uint32_t) h,   neqk1_magic);
    const uint32_t iq3 = fastdiv   ((uint32_t) seq, rq3_magic);

    extern __shared__ __align__(128) float smem[];
    float * Kh = smem;                       // [C][HLD]  K[:, half]
    float * As = Kh + C * HLD;               // [C][ALD]  (also the Q half-stage)
    float * gc = As + C * ALD;               // [C]  inclusive cumsum of g (log space)
    float * bs = gc + C;                     // [C]  beta

    const float * kbase = k + iq3 * sq3 + iq1 * sq1;
    const float * qbase = q + iq3 * sq3 + iq1 * sq1;

    float * base = scratch + ((int64_t) (seq * H + h) * n_chunks_batch + chunk_local) * L::total;
    float * X0   = base + L::up;             // [C][S_v]  solve columns 0 .. S_v-1     -> U'
    float * X1   = base + L::w;              // [C][S_v]  solve columns S_v .. 2S_v-1  -> W

    if (tid < C) {
        float gv = 0.0f;
        float bv = 0.0f;
        if (tid < C_eff) {
            const int64_t off = seq * sb3 + (t0 + tid) * sb2 + h * sb1;
            gv = g[off];
            bv = beta[off];
        }
        gc[tid] = gv;   // g = 0 past C_eff keeps the cumsum flat, so gc[C-1] == gc[C_eff-1]
        bs[tid] = bv;   // beta = 0 past C_eff zeroes those rows of A, U' and W
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            acc  += gc[c];
            gc[c] = acc;
        }
    }
    __syncthreads();

    // Gram: As = K K^T  (M = C rows i, N = C cols j, K = S_v), accumulated over 64-wide halves of
    // the head dim staged into Kh (zero rows past C_eff). 16 tiles, 2 per warp.
    {
        gdn_frag_c acc[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            gdn_wmma::fill_fragment(acc[r], 0.0f);
        }
        for (int st = 0; st < S_v / 64; st++) {
            for (int idx = tid; idx < C * 64; idx += NT) {
                const int c = idx / 64;
                const int d = idx % 64;
                float val = 0.0f;
                if (c < C_eff) {
                    val = kbase[(t0 + c) * sq2 + st * 64 + d];
                }
                Kh[c * HLD + d] = val;
            }
            __syncthreads();
            if (!(dbg & 1)) {
#pragma unroll
                for (int kk = 0; kk < 8; kk++) {
#pragma unroll
                    for (int r = 0; r < 2; r++) {
                        const int t  = warp + 8 * r;
                        const int mt = t / 4;
                        const int nt = t % 4;
                        gdn_frag_a_row a;
                        gdn_frag_b_col b;
                        gdn_wmma::load_matrix_sync(a, Kh + mt * 16 * HLD + kk * 8, HLD);
                        gdn_wmma::load_matrix_sync(b, Kh + nt * 16 * HLD + kk * 8, HLD);   // B(d, j) = Kh[j][d]
                        gdn_mma(acc[r], a, b, dbg);
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int t  = warp + 8 * r;
            const int mt = t / 4;
            const int nt = t % 4;
            gdn_wmma::store_matrix_sync(As + mt * 16 * ALD + nt * 16, acc[r], ALD, gdn_wmma::mem_row_major);
        }
    }
    // B = [beta o V | diag(beta gamma) K] straight into the scratch that becomes X = [U' | W]
    if (tid < 2 * S_v) {
        const bool isV = tid < S_v;
        const int  col = isV ? tid : tid - S_v;
        float *    xp  = (isV ? X0 : X1) + col;
        for (int i = 0; i < C; i++) {
            float b = 0.0f;
            if (i < C_eff) {
                if (isV) {
                    b = bs[i] * v[seq * sv3 + (t0 + i) * sv2 + h * sv1 + col];
                } else {
                    b = bs[i] * expf(gc[i]) * kbase[(t0 + i) * sq2 + col];
                }
            }
            xp[i * S_v] = b;
        }
    }
    __syncthreads();

    // As <- -A,  A[i][j] = beta_i * exp(gc_i - gc_j) * (k_i . k_j) for j < i, else 0.  Negated so
    // that both the block updates below and the diagonal solves are plain accumulates.
    for (int idx = tid; idx < C * C; idx += NT) {
        const int i = idx / C;
        const int j = idx % C;
        const float v_ = (j < i) ? -(bs[i] * expf(gc[i] - gc[j]) * As[i * ALD + j]) : 0.0f;
        As[i * ALD + j] = v_;
    }
    __syncthreads();

    // Blocked unit-lower-triangular solve (I + A) X = B in place on X = [U' | W] (global, L2-hot):
    // for each 16-row block bb:  X_bb += (-A)[bb][0..bb) X[0..bb)  on tensor cores,  then the
    // 16x16 diagonal block by forward substitution, one solve column per thread (16-deep chain
    // instead of the 2016-deep one of the unblocked form).
    for (int bb = 0; bb < C / 16; bb++) {
        if (bb > 0 && !(dbg & 2)) {
#pragma unroll
            for (int r = 0; r < TPWX; r++) {
                const int t = warp + 8 * r;
                if (t < NTX) {
                    float *   region = (t < S_v / 16) ? X0 : X1;
                    const int ncol   = (t % (S_v / 16)) * 16;
                    gdn_frag_c acc;
                    gdn_wmma::load_matrix_sync(acc, region + bb * 16 * S_v + ncol, S_v, gdn_wmma::mem_row_major);
                    for (int kk = 0; kk < 2 * bb; kk++) {
                        gdn_frag_a_row a;
                        gdn_frag_b_row b;
                        gdn_wmma::load_matrix_sync(a, As + bb * 16 * ALD + kk * 8, ALD);
                        gdn_wmma::load_matrix_sync(b, region + kk * 8 * S_v + ncol, S_v);
                        gdn_mma(acc, a, b, dbg);
                    }
                    gdn_wmma::store_matrix_sync(region + bb * 16 * S_v + ncol, acc, S_v, gdn_wmma::mem_row_major);
                }
            }
            __syncthreads();
        }
        if (!(dbg & 2) && tid < 2 * S_v) {
            const bool isV = tid < S_v;
            const int  col = isV ? tid : tid - S_v;
            float *    xp  = (isV ? X0 : X1) + (bb * 16) * S_v + col;
            const float * Ab = As + (bb * 16) * ALD + bb * 16;
            float x[16];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                float b = xp[i * S_v];
#pragma unroll
                for (int j = 0; j < i; j++) {
                    b += Ab[i * ALD + j] * x[j];
                }
                x[i] = b;
                xp[i * S_v] = b;
            }
        }
        __syncthreads();
    }

    // P = Q K^T o M  (M = C rows c, N = C cols j, K = S_v): Q and K staged in 64-wide halves along
    // the head dim (Q into the As region, K into Kh; zero rows past C_eff). 2 tiles per warp.
    if (!(dbg & 4)) {
        gdn_frag_c acc[2];
#pragma unroll
        for (int r = 0; r < 2; r++) {
            gdn_wmma::fill_fragment(acc[r], 0.0f);
        }
        for (int st = 0; st < S_v / 64; st++) {
            for (int idx = tid; idx < C * 64; idx += NT) {
                const int c  = idx / 64;
                const int ii = idx % 64;
                float qv = 0.0f;
                float kv = 0.0f;
                if (c < C_eff) {
                    qv = qbase[(t0 + c) * sq2 + st * 64 + ii];
                    kv = kbase[(t0 + c) * sq2 + st * 64 + ii];
                }
                As[c * HLD + ii] = qv;
                Kh[c * HLD + ii] = kv;
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < 8; kk++) {
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int t  = warp + 8 * r;
                    const int mt = t / 4;
                    const int nt = t % 4;
                    gdn_frag_a_row a;
                    gdn_frag_b_col b;
                    gdn_wmma::load_matrix_sync(a, As + mt * 16 * HLD + kk * 8, HLD);
                    gdn_wmma::load_matrix_sync(b, Kh + nt * 16 * HLD + kk * 8, HLD);
                    gdn_mma(acc[r], a, b, dbg);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            const int t  = warp + 8 * r;
            const int mt = t / 4;
            const int nt = t % 4;
            gdn_wmma::store_matrix_sync(As + mt * 16 * ALD + nt * 16, acc[r], ALD, gdn_wmma::mem_row_major);
        }
    }
    __syncthreads();
    {
        float * P = base + L::p;
        for (int idx = tid; idx < C * C; idx += NT) {
            const int c = idx / C;
            const int j = idx % C;
            P[c * C + j] = (c < C_eff && j <= c) ? expf(gc[c] - gc[j]) * As[c * ALD + j] : 0.0f;
        }
        if (tid < C) {
            base[L::gc + tid] = gc[tid];
        }
    }
#else
    GGML_UNUSED(q); GGML_UNUSED(k); GGML_UNUSED(v); GGML_UNUSED(g); GGML_UNUSED(beta); GGML_UNUSED(scratch);
    GGML_UNUSED(H); GGML_UNUSED(n_tokens); GGML_UNUSED(chunk0); GGML_UNUSED(n_chunks_batch);
    GGML_UNUSED(sq1); GGML_UNUSED(sq2); GGML_UNUSED(sq3); GGML_UNUSED(sv1); GGML_UNUSED(sv2); GGML_UNUSED(sv3);
    GGML_UNUSED(sb1); GGML_UNUSED(sb2); GGML_UNUSED(sb3); GGML_UNUSED(neqk1_magic); GGML_UNUSED(rq3_magic);
    GGML_UNUSED(dbg);
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------------------------
// kernel 2: sequential over the chunks of one batch; a 32-column slice of the state in shared memory
// ---------------------------------------------------------------------------------------------
template <int S_v, int C, int CB, bool FULLSTAGE>
__global__ void __launch_bounds__(256, FULLSTAGE ? 1 : 2)
gdn_chunk_scan(const float * __restrict__ q,
               const float * __restrict__ k,
               const float * __restrict__ scratch,
               const float * __restrict__ state_in,     // transposed [col][i], (seq*H + h) * S_v*S_v
               float *       __restrict__ state_out,    // same layout (may alias state_in)
               float *       __restrict__ attn,         // ((seq*n_tokens + t)*H + h)*S_v + col
               const int64_t H,
               const int64_t n_tokens,
               const int64_t chunk0,
               const int64_t n_chunks_batch,
               const int64_t sq1, const int64_t sq2, const int64_t sq3,
               const uint3   neqk1_magic,
               const uint3   rq3_magic,
               const float   scale,
               const int     dbg) {
#if GDN_HAVE_WMMA && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using L = gdn_scratch_layout<S_v, C>;
    constexpr int NT   = 256;
    constexpr int SLD  = CB + 4;         // row stride of the [.][CB] tiles (36)
    constexpr int KLD  = S_v + 4;        // row stride of the [.][S_v] stages
    constexpr int NTN  = CB / 16;        // n-tiles (2)
    constexpr int MT1  = C / 16;         // m-tiles of the C-row products (4)
    constexpr int MT3  = S_v / 16;       // m-tiles of the state update (8 / 4)
    constexpr int TPW3 = MT3 * NTN / 8;  // state-update tiles per warp (2 / 1)
    constexpr int STG  = FULLSTAGE ? C * KLD : C * SLD;
    static_assert(CB == 32 && C == 64, "tile bookkeeping assumes 32 columns and a 64-token chunk");
    static_assert(MT1 * NTN == 8 && (MT3 * NTN) % 8 == 0, "one C-row tile per warp; state tiles split evenly");
    static_assert(FULLSTAGE || 16 * KLD <= C * SLD, "K stage must fit in the stage buffer");

    const int h    = blockIdx.x;
    const int seq  = blockIdx.y;
    const int col0 = blockIdx.z * CB;
    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int mt1  = warp / NTN;         // this warp's tile in the C-row products
    const int nt1  = warp % NTN;

    const uint32_t iq1 = fastmodulo((uint32_t) h,   neqk1_magic);
    const uint32_t iq3 = fastdiv   ((uint32_t) seq, rq3_magic);

    extern __shared__ __align__(128) float smem[];
    float * S_sh  = smem;                    // [S_v][SLD]   state slice, S[i][col0 + n]
    float * Us    = S_sh + S_v * SLD;        // [C][SLD]     U
    float * stg   = Us + C * SLD;            // [C][SLD]     accumulator staging
    float * stage = stg + C * SLD;           // operand staging (see STG)
    float * gam   = stage + STG;             // [C]  gamma_c
    float * dec   = gam + C;                 // [C]  gamma_C / gamma_j
    float * gamC  = dec + C;                 // [8]

    const int64_t sbase = (int64_t) (seq * H + h) * S_v * S_v;
    const float * qbase = q + iq3 * sq3 + iq1 * sq1;
    const float * kbase = k + iq3 * sq3 + iq1 * sq1;

    // state slice in: S_sh[i][n] = state_in[(col0+n)*S_v + i]  (coalesced along i)
    for (int idx = tid; idx < CB * S_v; idx += NT) {
        const int n = idx / S_v;
        const int i = idx % S_v;
        S_sh[i * SLD + n] = state_in[sbase + (int64_t) (col0 + n) * S_v + i];
    }
    __syncthreads();

    for (int cl = 0; cl < n_chunks_batch; cl++) {
        const int64_t chunk = chunk0 + cl;
        const int64_t t0    = chunk * C;
        const int     C_eff = (int) min((int64_t) C, n_tokens - t0);

        const float * base = scratch + ((int64_t) (seq * H + h) * n_chunks_batch + cl) * L::total;
        const float * Up   = base + L::up;
        const float * W    = base + L::w;
        const float * P    = base + L::p;
        const float * gcs  = base + L::gc;

        if (tid < C) {
            const float gcC = gcs[C - 1];    // == gc[C_eff-1], see kernel 1
            gam[tid] = expf(gcs[tid]);
            dec[tid] = tid < C_eff ? expf(gcC - gcs[tid]) : 0.0f;
            if (tid == 0) {
                gamC[0] = expf(gcC);
            }
        }
        __syncthreads();

        // ---- step 1: stg = W S0   (M = C, N = CB, K = S_v); then U = U' - stg
        if (!(dbg & 8)) {
            gdn_frag_c acc;
            gdn_wmma::fill_fragment(acc, 0.0f);
#pragma unroll
            for (int kk = 0; kk < S_v / 8; kk++) {
                gdn_frag_a_row a;
                gdn_frag_b_row b;
                gdn_wmma::load_matrix_sync(a, W + mt1 * 16 * S_v + kk * 8, S_v);
                gdn_wmma::load_matrix_sync(b, S_sh + kk * 8 * SLD + nt1 * 16, SLD);
                gdn_mma(acc, a, b, dbg);
            }
            gdn_wmma::store_matrix_sync(stg + mt1 * 16 * SLD + nt1 * 16, acc, SLD, gdn_wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < C * CB; idx += NT) {
            const int c = idx / CB;
            const int n = idx % CB;
            Us[c * SLD + n] = Up[c * S_v + col0 + n] - stg[c * SLD + n];
        }
        __syncthreads();

        // ---- step 2: O = scale * ( (gamma o Q) S0 + P U )
        if (!(dbg & 16)) {
            gdn_frag_c acc;
            gdn_wmma::fill_fragment(acc, 0.0f);
            if constexpr (FULLSTAGE) {
                // stage gamma_c * Q[c][:] once, zero rows past C_eff
                for (int idx = tid; idx < C * S_v; idx += NT) {
                    const int c  = idx / S_v;
                    const int ii = idx % S_v;
                    float val = 0.0f;
                    if (c < C_eff && !(dbg & 64)) {
                        val = gam[c] * qbase[(t0 + c) * sq2 + ii];
                    }
                    stage[c * KLD + ii] = val;
                }
                __syncthreads();
#pragma unroll
                for (int kk = 0; kk < S_v / 8; kk++) {
                    gdn_frag_a_row a;
                    gdn_frag_b_row b;
                    gdn_wmma::load_matrix_sync(a, stage + mt1 * 16 * KLD + kk * 8, KLD);
                    gdn_wmma::load_matrix_sync(b, S_sh + kk * 8 * SLD + nt1 * 16, SLD);
                    gdn_mma(acc, a, b, dbg);
                }
            } else {
                for (int st = 0; st < S_v / 32; st++) {
                    for (int idx = tid; idx < C * 32; idx += NT) {
                        const int c  = idx / 32;
                        const int ii = idx % 32;
                        float val = 0.0f;
                        if (c < C_eff && !(dbg & 64)) {
                            val = gam[c] * qbase[(t0 + c) * sq2 + st * 32 + ii];
                        }
                        stage[c * SLD + ii] = val;
                    }
                    __syncthreads();
#pragma unroll
                    for (int kk = 0; kk < 4; kk++) {
                        gdn_frag_a_row a;
                        gdn_frag_b_row b;
                        gdn_wmma::load_matrix_sync(a, stage + mt1 * 16 * SLD + kk * 8, SLD);
                        gdn_wmma::load_matrix_sync(b, S_sh + (st * 32 + kk * 8) * SLD + nt1 * 16, SLD);
                        gdn_mma(acc, a, b, dbg);
                    }
                    __syncthreads();
                }
            }
#pragma unroll
            for (int kk = 0; kk < C / 8; kk++) {
                gdn_frag_a_row a;
                gdn_frag_b_row b;
                gdn_wmma::load_matrix_sync(a, P + mt1 * 16 * C + kk * 8, C);
                gdn_wmma::load_matrix_sync(b, Us + kk * 8 * SLD + nt1 * 16, SLD);
                gdn_mma(acc, a, b, dbg);
            }
#pragma unroll
            for (int e = 0; e < gdn_frag_c::num_elements; e++) {
                acc.x[e] *= scale;
            }
            if (C_eff == C) {
                float * o = attn + (((int64_t) seq * n_tokens + t0 + mt1 * 16) * H + h) * S_v + col0 + nt1 * 16;
                gdn_wmma::store_matrix_sync(o, acc, (unsigned) (H * S_v), gdn_wmma::mem_row_major);
            } else {
                gdn_wmma::store_matrix_sync(stg + mt1 * 16 * SLD + nt1 * 16, acc, SLD, gdn_wmma::mem_row_major);
                __syncthreads();
                for (int idx = tid; idx < C_eff * CB; idx += NT) {
                    const int c = idx / CB;
                    const int n = idx % CB;
                    attn[(((int64_t) seq * n_tokens + t0 + c) * H + h) * S_v + col0 + n] = stg[c * SLD + n];
                }
            }
        }
        __syncthreads();

        // ---- step 3: S = gamma_C S0 + K_dec^T U   (M = S_v rows i, N = CB, K = C tokens j)
        if (!(dbg & 32)) {
            const float gC = gamC[0];
            gdn_frag_c acc[TPW3];
#pragma unroll
            for (int r = 0; r < TPW3; r++) {
                const int t  = warp + 8 * r;
                const int mt = t / NTN;
                const int nt = t % NTN;
                gdn_wmma::load_matrix_sync(acc[r], S_sh + mt * 16 * SLD + nt * 16, SLD, gdn_wmma::mem_row_major);
#pragma unroll
                for (int e = 0; e < gdn_frag_c::num_elements; e++) {
                    acc[r].x[e] *= gC;
                }
            }
            constexpr int JST = FULLSTAGE ? C : 16;   // K rows staged per round
            for (int st = 0; st < C / JST; st++) {
                // stage K_dec rows j in [st*JST, +JST): stage[jj][i] = dec[j] * k[t0+j][i], zero past C_eff
                for (int idx = tid; idx < JST * S_v; idx += NT) {
                    const int jj = idx / S_v;
                    const int i  = idx % S_v;
                    const int j  = st * JST + jj;
                    float val = 0.0f;
                    if (j < C_eff && !(dbg & 64)) {
                        val = dec[j] * kbase[(t0 + j) * sq2 + i];
                    }
                    stage[jj * KLD + i] = val;
                }
                __syncthreads();
#pragma unroll
                for (int kk = 0; kk < JST / 8; kk++) {
#pragma unroll
                    for (int r = 0; r < TPW3; r++) {
                        const int t  = warp + 8 * r;
                        const int mt = t / NTN;
                        const int nt = t % NTN;
                        gdn_frag_a_col a;   // A(i, j) = stage[jj][i]
                        gdn_frag_b_row b;
                        gdn_wmma::load_matrix_sync(a, stage + kk * 8 * KLD + mt * 16, KLD);
                        gdn_wmma::load_matrix_sync(b, Us + (st * JST + kk * 8) * SLD + nt * 16, SLD);
                        gdn_mma(acc[r], a, b, dbg);
                    }
                }
                __syncthreads();
            }
#pragma unroll
            for (int r = 0; r < TPW3; r++) {
                const int t  = warp + 8 * r;
                const int mt = t / NTN;
                const int nt = t % NTN;
                gdn_wmma::store_matrix_sync(S_sh + mt * 16 * SLD + nt * 16, acc[r], SLD, gdn_wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // state slice out
    for (int idx = tid; idx < CB * S_v; idx += NT) {
        const int n = idx / S_v;
        const int i = idx % S_v;
        state_out[sbase + (int64_t) (col0 + n) * S_v + i] = S_sh[i * SLD + n];
    }
#else
    GGML_UNUSED(q); GGML_UNUSED(k); GGML_UNUSED(scratch); GGML_UNUSED(state_in); GGML_UNUSED(state_out);
    GGML_UNUSED(attn); GGML_UNUSED(H); GGML_UNUSED(n_tokens); GGML_UNUSED(chunk0); GGML_UNUSED(n_chunks_batch);
    GGML_UNUSED(sq1); GGML_UNUSED(sq2); GGML_UNUSED(sq3); GGML_UNUSED(neqk1_magic); GGML_UNUSED(rq3_magic);
    GGML_UNUSED(scale); GGML_UNUSED(dbg);
    NO_DEVICE_CODE;
#endif
}

static bool gdn_chunked_supported(int64_t S_v, int64_t sq2) {
#if GDN_HAVE_WMMA
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    // sm_80+ for TF32 mma; the attn store uses ldm = H*S_v and the K/Q stages read rows of stride sq2
    return cc >= GGML_CUDA_CC_AMPERE && (S_v == 64 || S_v == 128) && (sq2 % 4) == 0;
#else
    GGML_UNUSED(S_v); GGML_UNUSED(sq2);
    return false;
#endif
}

static int gdn_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

template <int S_v, bool FULLSTAGE>
static void gdn_chunked_launch_t(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_out_override,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, ggml_backend_cuda_context & ctx) {
    constexpr int C  = GDN_CHUNK;
    constexpr int CB = GDN_COLS;
    using L = gdn_scratch_layout<S_v, C>;

    const int64_t n_chunks   = (n_tokens + C - 1) / C;
    const int64_t attn_elems = (int64_t) S_v * H * n_tokens * n_seqs;
    float * attn_d  = dst_d;
    float * state_d = state_out_override ? state_out_override : dst_d + attn_elems;

    const uint3 neqk1_magic = init_fastdiv_values((uint32_t) neqk1);
    const uint3 rq3_magic   = init_fastdiv_values((uint32_t) rq3);

    constexpr size_t smem_intra = gdn_intra_smem_bytes<S_v, C>();
    constexpr size_t smem_scan  = gdn_scan_smem_bytes<S_v, C, CB, FULLSTAGE>();

    static const bool prof    = getenv("ALPHA_GDN_PROF") != nullptr;
    static const int  dbg     = gdn_env_int("ALPHA_GDN_DBG", 0) | (gdn_env_int("ALPHA_GDN_TF32_1X", 0) ? GDN_DBG_TF32_1X : 0);
    static const bool overlap = gdn_env_int("ALPHA_GDN_OVERLAP", 1) != 0 && !prof;

    // per-instantiation, per-process: the func attributes are per-device; the side stream and
    // events are the fork/join machinery for overlapping intra(b+1) with scan(b)
    static bool         init = false;
    static cudaStream_t s2   = nullptr;
    static cudaEvent_t  ev_fork, ev_intra[2], ev_scan[2];
    if (!init) {
        CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_intra<S_v, C>,                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_intra));
        CUDA_CHECK(cudaFuncSetAttribute(gdn_chunk_scan <S_v, C, CB, FULLSTAGE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_scan));
        CUDA_CHECK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork,     cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_intra[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_intra[1], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_scan[0],  cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_scan[1],  cudaEventDisableTiming));
        init = true;
    }

    const int64_t batch     = std::min<int64_t>(n_chunks, GDN_CHUNK_BATCH);
    const size_t  per_batch = (size_t) n_seqs * H * batch * L::total;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), per_batch * (overlap ? 2 : 1));

    cudaStream_t stream = ctx.stream();

    // ALPHA_GDN_PROF=1: event-time the two kernels (serial, synchronous; diagnostics only).
    static double  prof_intra_ms = 0.0, prof_scan_ms = 0.0;
    static int64_t prof_tokens = 0;
    static int     prof_calls = 0;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr;
    if (prof) {
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventCreate(&ev2));
    }

    if (overlap) {
        CUDA_CHECK(cudaEventRecord(ev_fork, stream));
        CUDA_CHECK(cudaStreamWaitEvent(s2, ev_fork, 0));
    }
    int bi = 0;
    for (int64_t chunk0 = 0; chunk0 < n_chunks; chunk0 += batch, bi++) {
        const int64_t nb  = std::min<int64_t>(batch, n_chunks - chunk0);
        float *       buf = scratch.get() + (overlap ? (size_t) (bi & 1) * per_batch : 0);
        cudaStream_t  si  = overlap ? s2 : stream;

        if (overlap && bi >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(s2, ev_scan[bi & 1], 0));   // scan(bi-2) is done reading buf
        }
        if (prof) CUDA_CHECK(cudaEventRecord(ev0, stream));
        const dim3 grid_intra((unsigned) nb, (unsigned) H, (unsigned) n_seqs);
        gdn_chunk_intra<S_v, C><<<grid_intra, 256, smem_intra, si>>>(
            q_d, k_d, v_d, g_d, b_d, buf, H, n_tokens, chunk0, nb,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, dbg);
        if (overlap) {
            CUDA_CHECK(cudaEventRecord(ev_intra[bi & 1], s2));
            CUDA_CHECK(cudaStreamWaitEvent(stream, ev_intra[bi & 1], 0));
        }
        if (prof) CUDA_CHECK(cudaEventRecord(ev1, stream));
        const dim3 grid_scan((unsigned) H, (unsigned) n_seqs, (unsigned) (S_v / CB));
        gdn_chunk_scan<S_v, C, CB, FULLSTAGE><<<grid_scan, 256, smem_scan, stream>>>(
            q_d, k_d, buf, chunk0 == 0 ? s_d : state_d, state_d, attn_d,
            H, n_tokens, chunk0, nb, sq1, sq2, sq3, neqk1_magic, rq3_magic, scale, dbg);
        if (overlap) {
            CUDA_CHECK(cudaEventRecord(ev_scan[bi & 1], stream));
        }
        if (prof) {
            CUDA_CHECK(cudaEventRecord(ev2, stream));
            CUDA_CHECK(cudaEventSynchronize(ev2));
            float ms01 = 0.0f, ms12 = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms01, ev0, ev1));
            CUDA_CHECK(cudaEventElapsedTime(&ms12, ev1, ev2));
            prof_intra_ms += ms01;
            prof_scan_ms  += ms12;
        }
    }
    // join: every s2 kernel was awaited by the main stream before the scan that consumes it, so the
    // main stream already depends on all side-stream work.

    if (prof) {
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaEventDestroy(ev2));
        prof_tokens += n_tokens * n_seqs;
        prof_calls++;
        if (prof_calls % 30 == 0) {
            const double per = 2048.0 / (double) prof_tokens;
            fprintf(stderr, "[gdn-prof] calls=%d tokens=%lld H=%lld S_v=%d fullstage=%d intra=%.1fms scan=%.1fms  per-2048-tok: intra=%.3fms scan=%.3fms\n",
                    prof_calls, (long long) prof_tokens, (long long) H, S_v, (int) FULLSTAGE, prof_intra_ms, prof_scan_ms,
                    prof_intra_ms * per, prof_scan_ms * per);
        }
    }
}

template <int S_v>
static void gdn_chunked_launch(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_out_override,
        int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, ggml_backend_cuda_context & ctx) {
    // FULLSTAGE=1 (1 block/SM) makes the scan slower than the default 2 blocks/SM
    static const bool fullstage = gdn_env_int("ALPHA_GDN_FULLSTAGE", 0) != 0;
    if (fullstage) {
        gdn_chunked_launch_t<S_v, true >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, ctx);
    } else {
        gdn_chunked_launch_t<S_v, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_out_override, H, n_tokens, n_seqs,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, ctx);
    }
}
