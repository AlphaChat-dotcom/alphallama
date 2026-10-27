// Decode flash attention restricted to listed KV cells. Enabled by ALPHA_QSA_FA_SPARSE.
//
// QSA (qwen4exp's sparse attention) selects ~2K of the n_kv cells per query. Expressing that as a mask over dense
// attention (an [n_kv, n_q] mask plus flash attention over every cell) does far more work than needed at long
// context. Here each query reads only its listed cells' K/V rows, in place, without gathered copies.
//
//   pass 1 (partial): grid (S chunks of the cell list, H_kv, n_q); the G = H/H_kv query heads sharing a KV head are
//           handled together, so each K/V row is read once for all of them. Every warp walks its own cells with an
//           online softmax per head; the warps of a block merge in shared memory and write one partial per chunk.
//   pass 2 (merge): one block per (head, query) combines the S partials (log-sum-exp) into dst.
// The chunking depends only on the list length, never on n_q, so a query's result does not depend on how many
// queries share the batch (an MTP verify reproduces plain decode).

#include "fattn-sparse.cuh"

#include <cuda_fp16.h>

static constexpr int FAS_NW = 4;   // warps per block

// lane l owns dims {2l + 64e, 2l + 64e + 1}, e < D/64: half2 loads are coalesced (128 B per warp per e) and the
// float2 reads of the shared query rows are conflict-free
// G query heads share a KV head; a block handles GB of them (blockIdx.y = g * (G/GB) + head block): fewer
// accumulators per warp, more resident warps to hide the per-cell load latency (K/V rows are re-read from L2)
template <int D, int G, int GB>
static __global__ void __launch_bounds__(FAS_NW * WARP_SIZE)
fa_sparse_partial(const char * __restrict__ q, const int64_t q_nb1, const int64_t q_nb2,
                  const char * __restrict__ k, const int64_t k_nb1, const int64_t k_nb2,
                  const char * __restrict__ v, const int64_t v_nb1, const int64_t v_nb2,
                  const char * __restrict__ mask, const int64_t m_nb1, const bool mask_f16,
                  const int32_t * __restrict__ cells, const int64_t c_stride, const int W, const int S,
                  const float scale, float * __restrict__ part) {
    constexpr int E2 = D / 64;   // half2 per lane
    constexpr int NB = G / GB;
    const int s = blockIdx.x, g = blockIdx.y / NB, hb = blockIdx.y % NB, t = blockIdx.z, Hkv = gridDim.y / NB;
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;

    __shared__ float2 qs[GB][D / 2];
    __shared__ float2 sacc[GB][D / 2];
    __shared__ float sM[GB], sL[GB];

    // scores in log2 units: exp2f below
    const float qscale = scale * 1.4426950408889634f;
    for (int i = threadIdx.x; i < GB * D / 2; i += blockDim.x) {
        const int gi = i / (D / 2), d2 = i % (D / 2);
        const float * qp = (const float *) (q + t * q_nb1 + (int64_t) (g * G + hb * GB + gi) * q_nb2) + 2 * d2;
        qs[gi][d2] = make_float2(qp[0] * qscale, qp[1] * qscale);
    }
    __syncthreads();

    float M[GB], L[GB];
    float2 acc[GB][E2];
#pragma unroll
    for (int gi = 0; gi < GB; ++gi) {
        M[gi] = -INFINITY; L[gi] = 0.0f;
#pragma unroll
        for (int e = 0; e < E2; ++e) acc[gi][e] = make_float2(0.0f, 0.0f);
    }

    const int c0 = (int) ((int64_t) s * W / S), c1 = (int) ((int64_t) (s + 1) * W / S);
    const int32_t * cl = cells + t * c_stride;
    const char * mrow = mask + t * m_nb1;
    // one cell: scores for the G heads, then the online-softmax update (log2 units)
    auto step = [&](const float2 (&kf)[E2], const float2 (&vf)[E2], const float mv) {
        float sc[GB];
#pragma unroll
        for (int gi = 0; gi < GB; ++gi) {
            float p = 0.0f;
#pragma unroll
            for (int e = 0; e < E2; ++e) {
                const float2 qq = qs[gi][lane + 32 * e];
                p += qq.x * kf[e].x + qq.y * kf[e].y;
            }
            sc[gi] = p;
        }
        // all G reductions interleaved: independent shuffles pipeline
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
#pragma unroll
            for (int gi = 0; gi < GB; ++gi) sc[gi] += __shfl_xor_sync(0xffffffff, sc[gi], off);
        }
        const float mv2 = mv * 1.4426950408889634f;
#pragma unroll
        for (int gi = 0; gi < GB; ++gi) {
            const float x = sc[gi] + mv2;
            const float mn = fmaxf(M[gi], x);
            const float corr = exp2f(M[gi] - mn);   // M = -inf on the first cell: 0
            const float pe = exp2f(x - mn);
            L[gi] = L[gi] * corr + pe;
#pragma unroll
            for (int e = 0; e < E2; ++e) {
                acc[gi][e].x = acc[gi][e].x * corr + pe * vf[e].x;
                acc[gi][e].y = acc[gi][e].y * corr + pe * vf[e].y;
            }
            M[gi] = mn;
        }
    };
    auto load = [&](const int cell, float2 (&kf)[E2], float2 (&vf)[E2]) {
        const half2 * kr = (const half2 *) (k + (int64_t) cell * k_nb1 + (int64_t) g * k_nb2) + lane;
        const half2 * vr = (const half2 *) (v + (int64_t) cell * v_nb1 + (int64_t) g * v_nb2) + lane;
#pragma unroll
        for (int e = 0; e < E2; ++e) { kf[e] = __half22float2(kr[32 * e]); vf[e] = __half22float2(vr[32 * e]); }
    };
    auto maskv = [&](const int cell) {
        return mask_f16 ? __half2float(((const half *) mrow)[cell]) : ((const float *) mrow)[cell];
    };
    // cells in a fixed order per warp (warp, warp + NW, ...). Each lane first fetches one upcoming cell's index and
    // mask value, so the K/V loads of a cell depend on a shuffle, not on two dependent global loads.
    for (int jb = c0 + warp; jb < c1; jb += WARP_SIZE * FAS_NW) {
        const int jl = jb + lane * FAS_NW;
        int   cell_l = 0;
        float mv_l   = -INFINITY;
        if (jl < c1) { cell_l = cl[jl]; mv_l = maskv(cell_l); }
        const int n_here = min(WARP_SIZE, (c1 - jb + FAS_NW - 1) / FAS_NW);
        for (int i = 0; i < n_here; ++i) {
            const float mv = __shfl_sync(0xffffffff, mv_l, i);
            if (mv == -INFINITY) continue;   // warp-uniform
            const int cell = __shfl_sync(0xffffffff, cell_l, i);
            float2 kf[E2], vf[E2];
            load(cell, kf, vf);
            step(kf, vf, mv);
        }
    }

    // merge the warps, in warp order (fixed: the result does not depend on scheduling)
    for (int w = 0; w < FAS_NW; ++w) {
        __syncthreads();
        if (warp == w) {
#pragma unroll
            for (int gi = 0; gi < GB; ++gi) {
                if (w == 0) {
#pragma unroll
                    for (int e = 0; e < E2; ++e) sacc[gi][lane + 32 * e] = acc[gi][e];
                    if (lane == 0) { sM[gi] = M[gi]; sL[gi] = L[gi]; }
                    continue;
                }
                if (M[gi] == -INFINITY) continue;
                const float om = sM[gi], ol = sL[gi];
                __syncwarp();
                const float mn = fmaxf(om, M[gi]);
                const float co = exp2f(om - mn), cn = exp2f(M[gi] - mn);
#pragma unroll
                for (int e = 0; e < E2; ++e) {
                    float2 a = sacc[gi][lane + 32 * e];
                    a.x = a.x * co + acc[gi][e].x * cn;
                    a.y = a.y * co + acc[gi][e].y * cn;
                    sacc[gi][lane + 32 * e] = a;
                }
                if (lane == 0) { sM[gi] = mn; sL[gi] = ol * co + L[gi] * cn; }
                __syncwarp();
            }
        }
    }
    __syncthreads();

    float * pb = part + ((((int64_t) t * Hkv + g) * S + s) * G + hb * GB) * (D + 2);
    for (int i = threadIdx.x; i < GB * D / 2; i += blockDim.x) {
        const int gi = i / (D / 2), d2 = i % (D / 2);
        pb[(int64_t) gi * (D + 2) + 2 * d2]     = sacc[gi][d2].x;
        pb[(int64_t) gi * (D + 2) + 2 * d2 + 1] = sacc[gi][d2].y;
    }
    if (threadIdx.x < GB) {
        pb[(int64_t) threadIdx.x * (D + 2) + D]     = sM[threadIdx.x];   // log2 units
        pb[(int64_t) threadIdx.x * (D + 2) + D + 1] = sL[threadIdx.x];
    }
}

template <int D, int G>
static __global__ void fa_sparse_merge(const float * __restrict__ part, const int S, const int Hkv,
                                       char * __restrict__ dst, const int64_t d_nb1, const int64_t d_nb2) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int g = h / G, gi = h % G;
    const float * pb = part + (((int64_t) t * Hkv + g) * S) * G * (D + 2) + (int64_t) gi * (D + 2);
    const int64_t pstride = (int64_t) G * (D + 2);
    float mmax = -INFINITY;
    for (int s = 0; s < S; ++s) mmax = fmaxf(mmax, pb[s * pstride + D]);
    float l = 0.0f, a = 0.0f;
    if (mmax != -INFINITY) {
        for (int s = 0; s < S; ++s) {
            const float ms = pb[s * pstride + D];
            if (ms == -INFINITY) continue;
            const float c = exp2f(ms - mmax);   // partial maxima are in log2 units
            l += pb[s * pstride + D + 1] * c;
            a += pb[s * pstride + d] * c;
        }
    }
    *(float *) (dst + (int64_t) d * sizeof(float) + h * d_nb1 + t * d_nb2) = l > 0.0f ? a / l : 0.0f;
}

bool ggml_cuda_flash_attn_sparse_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * C = dst->src[5];
    if (!C || !mask || dst->src[4]) return false;
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) return false;
    if (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) return false;
    if (C->type != GGML_TYPE_I32 || C->ne[1] != Q->ne[1] || C->nb[0] != sizeof(int32_t)) return false;
    if (Q->ne[3] != 1 || K->ne[3] != 1 || Q->ne[1] > 8) return false;
    const int64_t D = Q->ne[0], H = Q->ne[2], Hkv = K->ne[2];
    if (K->ne[0] != D || V->ne[0] != D || H % Hkv != 0) return false;
    if (K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half)) return false;
    float params[3]; memcpy(params, dst->op_params, sizeof(params));
    if (params[1] != 0.0f || params[2] != 0.0f) return false;   // no ALiBi, no softcap
    return D == 256 && H / Hkv == 12;
}

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0], * K = dst->src[1], * V = dst->src[2], * mask = dst->src[3], * C = dst->src[5];
    GGML_ASSERT(ggml_cuda_flash_attn_sparse_supported(dst));
    constexpr int D = 256, G = 12;
    float scale; memcpy(&scale, dst->op_params, sizeof(float));
    const int n_q = (int) Q->ne[1], Hkv = (int) K->ne[2], H = (int) Q->ne[2];
    const int W = (int) C->ne[0];
    static const int fas_chunk = [] { const char * e = getenv("ALPHA_QSA_FA_CHUNK"); return e ? std::max(8, atoi(e)) : 128; }();
    const int S = std::max(1, std::min(256, (W + fas_chunk - 1) / fas_chunk));
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) n_q * Hkv * S * G * (D + 2));
    static const int fas_gb = [] { const char * e = getenv("ALPHA_QSA_FA_GB"); return e ? atoi(e) : 6; }();
    auto kern = fas_gb == 12 ? fa_sparse_partial<D, G, 12> : fas_gb == 6 ? fa_sparse_partial<D, G, 6> :
                fas_gb == 3 ? fa_sparse_partial<D, G, 3> : fa_sparse_partial<D, G, 4>;
    const int nb = G / (fas_gb == 12 || fas_gb == 6 || fas_gb == 3 ? fas_gb : 4);
    kern<<<dim3(S, Hkv * nb, n_q), FAS_NW * WARP_SIZE, 0, stream>>>(
        (const char *) Q->data, Q->nb[1], Q->nb[2],
        (const char *) K->data, K->nb[1], K->nb[2],
        (const char *) V->data, V->nb[1], V->nb[2],
        (const char *) mask->data, mask->nb[1], mask->type == GGML_TYPE_F16,
        (const int32_t *) C->data, C->nb[1] / (int64_t) sizeof(int32_t), W, S, scale, part.get());
    fa_sparse_merge<D, G><<<dim3(H, n_q), D, 0, stream>>>(part.get(), S, Hkv, (char *) dst->data, dst->nb[1], dst->nb[2]);
}
