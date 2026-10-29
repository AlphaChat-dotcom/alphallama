// SPDX-License-Identifier: Apache-2.0
// The qwen4exp hyper-connection read on fp16 tensor cores, flat in T. Enabled by AI_Q4X_HC_TC=1.
//
// The scalar read (q4x_hc.cu: down6 + down4_fin + up_mix9) moves ~7 MB of Q8_0 weights per call, but every weight is
// dequantised and FMA'd once per column in scalar chains, so its cost grows with T and is several times the
// bandwidth bound at T = 8.
//
//  prep : per column t, xw = fp16(x * wn / cm[t]) with cm[t] = max |x * wn| (no fp16 overflow; re-applied in fp32), and
//         pss[ks][t] = sum of x^2 over K-chunk ks (down6's sums, the same chunking: chunks never straddle a stream).
//  down : mma.m16n8k16 (f16, f32 accumulate) over Q8_0 rows read straight into A fragments (Q8_0 is plain int8 with one
//         scale per 32: the pair (2tg, 2tg+1) is two consecutive bytes -- no activation permutation). Grid (row tiles,
//         KS chunks): part[ks][t][j] in down6's layout, so down4_fin (scale per stream, SiLU, inject) runs unchanged.
//         The H inject rows (BF16/F32) are dotted by one extra block per chunk in fp32.
//  up   : a warp owns 16 outputs d and all H streams (H row tiles s*N + d .. + 15 over K = LR), so the sigmoid-gated
//         mix of the H streams is done in registers: dst[t][d] = inv_h * sum_s xn(t,s,d) * sigmoid(g[s][t][d]).
// Every column runs the same instructions whatever T is (column-invariant). Not bit-identical to the scalar kernels
// (fp16 activations, a different sum tree): gated by paired NLL.
#include "kernels/q4x_hc.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace {

__device__ __forceinline__ void hmma16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned *>(&h); }

// two consecutive int8 of a Q8_0 row (block = {half d; int8 q[32]}, 34 bytes) at element k (even) -> half2(d*q0, d*q1)
__device__ __forceinline__ unsigned q8pair(const char * __restrict__ row, int k) {
    const char *   blk = row + (k >> 5) * 34;
    const __half   d   = *reinterpret_cast<const __half *>(blk);
    const unsigned u   = *reinterpret_cast<const unsigned short *>(blk + 2 + (k & 31));
    const __half2  q   = __floats2half2_rn((float) (int8_t) (u & 0xff), (float) (int8_t) (u >> 8));
    return h2u(__hmul2(q, __half2half2(d)));
}

__device__ __forceinline__ float hc_bf16(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }

__device__ __forceinline__ void hc_cp4(void * sm, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(sm)), "l"(g));
}
__device__ __forceinline__ void hc_cp16(void * sm, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(sm)), "l"(g));
}
__device__ __forceinline__ void hc_cp_wait() {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// grid (KS, T), block 256: chunk ks of column t -> xw fp16 of x * wn / cm[ks][t] with cm[ks][t] = the chunk's max |x * wn|
// (a per-chunk scale: the down block that reads this chunk re-applies it), and pss[ks][t] = sum of x^2 over the chunk
__global__ void hc_prep_k(const float * __restrict__ x, const float * __restrict__ wn, __half * __restrict__ xw,
                          float * __restrict__ cm, float * __restrict__ pss, int HD, int KC, int T) {
    const int ks = blockIdx.x, t = blockIdx.y, k0 = ks * KC;
    const float * xr = x + (size_t) t * HD + k0;
    const float * wr = wn + k0;
    __shared__ float rm[8], rq[8], s_sc;
    float mx = 0.0f, q = 0.0f;
    for (int k = threadIdx.x; k < KC; k += blockDim.x) { const float v = xr[k]; mx = fmaxf(mx, fabsf(v * wr[k])); q = fmaf(v, v, q); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); q += __shfl_xor_sync(0xffffffffu, q, o); }
    if ((threadIdx.x & 31) == 0) { rm[threadIdx.x >> 5] = mx; rq[threadIdx.x >> 5] = q; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f, qq = 0.0f;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) { m = fmaxf(m, rm[i]); qq += rq[i]; }
        cm[ks * T + t]  = m;
        pss[ks * T + t] = qq;
        s_sc            = m > 0.0f ? 1.0f / m : 0.0f;
    }
    __syncthreads();
    const float sc = s_sc;
    for (int k = threadIdx.x; k < KC; k += blockDim.x) xw[(size_t) t * HD + k0 + k] = __float2half_rn(xr[k] * wr[k] * sc);
}

// down: grid (LR/16 tiles + 1 inject block, KS), block 32*W: warp w takes sub-range w of the chunk's K; the warps' partials
// meet in smem (warp order) and the block writes part[ks][t][j] for its 16 rows (times cm[t]: back to x * wn units).
template <int NT, int W>
__global__ void __launch_bounds__(32 * W) hc_down_tc_k(const char * __restrict__ wd, int64_t wd_nb1,
                                                       const char * __restrict__ wi, int64_t wi_nb1, int wi_bf16,
                                                       const __half * __restrict__ xw, const float * __restrict__ cm,
                                                       const float * __restrict__ x, const float * __restrict__ wn,
                                                       float * __restrict__ part, int HD, int LR, int H, int KC, int T) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const int ks = blockIdx.y, k0 = ks * KC, nrows = LR + (wi ? H : 0);
    const int ntiles = LR / 16;
    if ((int) blockIdx.x == ntiles) {   // the inject rows: fp32 dots against x * wn (H rows, few columns)
        if (!wi) return;
        for (int p = warp; p < H * T; p += W) {
            const int h = p / T, t = p % T;
            const char * row = wi + (int64_t) h * wi_nb1;
            float a = 0.0f;
            for (int k = k0 + lane; k < k0 + KC; k += 32) {
                const float w = wi_bf16 ? hc_bf16(((const uint16_t *) row)[k]) : ((const float *) row)[k];
                a = fmaf(w, x[(size_t) t * HD + k] * wn[k], a);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) part[((size_t) ks * T + t) * nrows + LR + h] = a;
        }
        return;
    }
    __shared__ float red[W][NT][4][32];
    extern __shared__ __align__(16) unsigned char hd_sm[];   // [W][16][seg]: each warp's rows over its K sub-range
    const int r0 = blockIdx.x * 16;
    const int nch = KC / 16, per = (nch + W - 1) / W, c0 = warp * per, c1 = min(nch, c0 + per);
    // the sub-range [k0 + 16 c0, k0 + 16 c1) is whole Q8_0 blocks (per = KC / 16 / W chunks of 16: a multiple of 2)
    const int kb0 = (k0 + 16 * c0) >> 5, nkb = (16 * (c1 - c0)) >> 5, seg = nkb * 34, segw = (seg + 3) / 4;
    const int segs = (((16 * per) >> 5) * 34 + 3) / 4 * 4;   // bytes reserved per row (same for every warp)
    unsigned char * ws = hd_sm + (size_t) warp * 16 * segs;
    for (int i = lane; i < 16 * segw; i += 32) {
        const int r = i / segw, w = i - r * segw;
        hc_cp4(ws + r * segs + 4 * w, wd + (int64_t) (r0 + r) * wd_nb1 + (int64_t) kb0 * 34 + 4 * w);
    }
    hc_cp_wait();
    __syncwarp();
    // row pointers rebased so that element k of the row is found at (k - 32 kb0) in the staged segment
    const char * ra = (const char *) ws + g * segs - (int64_t) kb0 * 34;
    const char * rb = (const char *) ws + (g + 8) * segs - (int64_t) kb0 * 34;
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    const __half * xc[NT];
    bool cv[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) { const int col = 8 * n + g; cv[n] = col < T; xc[n] = xw + (size_t) (cv[n] ? col : 0) * HD; }
#pragma unroll 4
    for (int c = c0; c < c1; ++c) {
        const int k = k0 + 16 * c;
        const unsigned a0 = q8pair(ra, k + 2 * tg), a1 = q8pair(rb, k + 2 * tg);
        const unsigned a2 = q8pair(ra, k + 2 * tg + 8), a3 = q8pair(rb, k + 2 * tg + 8);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const unsigned b0 = cv[n] ? *reinterpret_cast<const unsigned *>(xc[n] + k + 2 * tg) : 0u;
            const unsigned b1 = cv[n] ? *reinterpret_cast<const unsigned *>(xc[n] + k + 2 * tg + 8) : 0u;
            hmma16(acc[n], a0, a1, a2, a3, b0, b1);
        }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 4; ++q) red[warp][n][q][lane] = acc[n][q];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = red[0][n][q][lane];
#pragma unroll
            for (int w = 1; w < W; ++w) v += red[w][n][q][lane];
            acc[n][q] = v;
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int t = 8 * n + 2 * tg + q;
            if (t < T) {
                const float m = cm[ks * T + t];
                part[((size_t) ks * T + t) * nrows + r0 + g]     = acc[n][q] * m;
                part[((size_t) ks * T + t) * nrows + r0 + g + 8] = acc[n][2 + q] * m;
            }
        }
    }
}

// up: grid N/16, block 32*KW: the KW warps of a block share outputs d0 .. d0+15 and all H streams and split K = LR
// into KW ranges; each warp stages its H x 16 row pieces (cp.async) and reads B (lo16 = fp16 lo / lmax[t], written by
// hc_fin_k) from L2; the partial sums meet in smem in warp order and warp 0 runs the gated H-stream mix.
template <int NT, int H, int KW>
__global__ void __launch_bounds__(32 * KW) hc_up_tc_k(const char * __restrict__ wu, int64_t wu_nb1,
                                                      const __half * __restrict__ lo16, const float * __restrict__ lmax,
                                                      const float * __restrict__ x, const float * __restrict__ wn,
                                                      const float * __restrict__ scl, float * __restrict__ dst, int N,
                                                      int LR, int T, float inv_h) {
    extern __shared__ __align__(16) unsigned char hc_sm[];   // [KW][H][16][seg] staged rows, then the partials
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tg = lane & 3;
    const int d0 = blockIdx.x * 16;
    const int nkb = LR / 32, kbw = ((nkb + KW - 1) / KW + 1) & ~1, kb0 = warp * kbw, kb1 = min(nkb, kb0 + kbw);
    const int segs = (kbw * 34 + 15) / 16 * 16;   // bytes per staged row piece (same for every warp)
    unsigned char * stg = hc_sm + (size_t) warp * H * 16 * segs;
    const int segw = ((kb1 - kb0) * 34 + 3) / 4;
    for (int s2 = 0; s2 < H; ++s2)
        for (int i = lane; i < 16 * segw; i += 32) {
            const int r = i / segw, w = i - r * segw;
            hc_cp4(stg + ((size_t) s2 * 16 + r) * segs + 4 * w, wu + (int64_t) (s2 * N + d0 + r) * wu_nb1 + (int64_t) kb0 * 34 + 4 * w);
        }
    hc_cp_wait();
    __syncwarp();
    float acc[H][NT][4];
#pragma unroll
    for (int s2 = 0; s2 < H; ++s2)
#pragma unroll
        for (int n = 0; n < NT; ++n) acc[s2][n][0] = acc[s2][n][1] = acc[s2][n][2] = acc[s2][n][3] = 0.0f;
    const __half * bc[NT];
    bool cv[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) { const int t = 8 * n + g; cv[n] = t < T; bc[n] = lo16 + (size_t) (cv[n] ? t : 0) * LR; }
    for (int kb = kb0; kb < kb1; ++kb) {
#pragma unroll
        for (int h2 = 0; h2 < 2; ++h2) {   // the two 16-element chunks of a 32-element block share its scale
            const int k = 32 * kb + 16 * h2;
            unsigned b[NT][2];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                b[n][0] = cv[n] ? *reinterpret_cast<const unsigned *>(bc[n] + k + 2 * tg) : 0u;
                b[n][1] = cv[n] ? *reinterpret_cast<const unsigned *>(bc[n] + k + 2 * tg + 8) : 0u;
            }
#pragma unroll
            for (int s2 = 0; s2 < H; ++s2) {
                const char * blk_a = (const char *) stg + ((size_t) s2 * 16 + g) * segs + (kb - kb0) * 34;
                const char * blk_b = (const char *) stg + ((size_t) s2 * 16 + g + 8) * segs + (kb - kb0) * 34;
                const __half2 da = __half2half2(*reinterpret_cast<const __half *>(blk_a));
                const __half2 db = __half2half2(*reinterpret_cast<const __half *>(blk_b));
                auto pr = [&](const char * blk, __half2 dd, int e) {
                    const unsigned u = *reinterpret_cast<const unsigned short *>(blk + 2 + e);
                    return h2u(__hmul2(__floats2half2_rn((float) (int8_t) (u & 0xff), (float) (int8_t) (u >> 8)), dd));
                };
                const int e = 16 * h2 + 2 * tg;
                const unsigned a0 = pr(blk_a, da, e), a1 = pr(blk_b, db, e), a2 = pr(blk_a, da, e + 8), a3 = pr(blk_b, db, e + 8);
#pragma unroll
                for (int n = 0; n < NT; ++n) hmma16(acc[s2][n], a0, a1, a2, a3, b[n][0], b[n][1]);
            }
        }
    }
    __syncthreads();   // the staged rows are dead: the partials reuse the smem
    float * red = reinterpret_cast<float *>(hc_sm);   // [KW][H][NT][4][32]
#pragma unroll
    for (int s2 = 0; s2 < H; ++s2)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) red[(((size_t) warp * H + s2) * NT + n) * 128 + q * 32 + lane] = acc[s2][n][q];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int s2 = 0; s2 < H; ++s2)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float v = red[((size_t) s2 * NT + n) * 128 + q * 32 + lane];
                for (int w = 1; w < KW; ++w) v += red[(((size_t) w * H + s2) * NT + n) * 128 + q * 32 + lane];
                acc[s2][n][q] = v;
            }
    const int HD = N * H;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int t = 8 * n + 2 * tg + q;
            if (t >= T) continue;
            const float m = lmax[t];
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int d = d0 + g + 8 * hr;
                float a = 0.0f;
#pragma unroll
                for (int s2 = 0; s2 < H; ++s2) {
                    const float gs = acc[s2][n][2 * hr + q] * m;
                    const float xn = x[(int64_t) t * HD + s2 * N + d] * scl[t * H + s2] * wn[s2 * N + d];
                    a += xn * (1.0f / (1.0f + expf(-gs)));
                }
                dst[(int64_t) t * N + d] = a * inv_h;
            }
        }
    }
}

// the scalar kernels' finalise (q4x_hc.cu down4_fin), reproduced here because it is file-local there: per t, the
// stream scales from pss, then lo = SiLU(inv_h * sum_s scl_s * sum_c part) and the inject rows
__global__ void hc_fin_k(const float * __restrict__ part, const float * __restrict__ pss, float * __restrict__ lo,
                         float * __restrict__ scl_out, float * __restrict__ inj_out, int N, int H, int LR, int nrows,
                         int T, int KS, float inv_h, float eps, __half * __restrict__ lo16, float * __restrict__ lmax) {
    const int t = blockIdx.x, cps = KS / H;
    __shared__ float scl[8], red[16];
    extern __shared__ float lrow[];   // [LR]
    if (threadIdx.x < H) {
        float q = 0.0f;
        for (int c = 0; c < cps; ++c) q += pss[(threadIdx.x * cps + c) * T + t];
        const float s = 1.0f / sqrtf(q / (float) N + eps);
        scl[threadIdx.x] = s;
        scl_out[t * H + threadIdx.x] = s;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < nrows; j += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < H; ++s) {
            float a = 0.0f;
            for (int c = 0; c < cps; ++c) a += part[((size_t) (s * cps + c) * T + t) * nrows + j];
            v += a * scl[s];
        }
        if (j < LR) {
            const float a = v * inv_h, l = a / (1.0f + expf(-a));
            lo[(int64_t) t * LR + j] = l;
            lrow[j] = l;
        } else {
            inj_out[(int64_t) t * H + (j - LR)] = v;
        }
    }
    __syncthreads();
    float m = 0.0f;
    for (int j = threadIdx.x; j < LR; j += blockDim.x) m = fmaxf(m, fabsf(lrow[j]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = 0.0f;
        for (int i = 0; i < (int) ((blockDim.x + 31) >> 5); ++i) mm = fmaxf(mm, red[i]);
        red[15] = mm;
        lmax[t] = mm;
    }
    __syncthreads();
    const float sc = red[15] > 0.0f ? 1.0f / red[15] : 0.0f;
    for (int j = threadIdx.x; j < LR; j += blockDim.x) lo16[(int64_t) t * LR + j] = __float2half_rn(lrow[j] * sc);
}

template <int NT>
void hc_tc_launch(cudaStream_t st, const char * wd, int64_t wd_nb1, const char * wu, int64_t wu_nb1, const char * wi,
                  int64_t wi_nb1, int wi_bf16, const float * x, const float * wn, float * part, float * pss, float * lo,
                  float * scl, float * dst, __half * xw, float * cm, __half * lo16, float * lmax, int N, int LR, int T,
                  float eps, int KC) {
    constexpr int H = 4, W = 4;
    const int HD = N * H, KS = HD / KC, nrows = LR + (wi ? H : 0);
    hc_prep_k<<<dim3((unsigned) KS, (unsigned) T), 256, 0, st>>>(x, wn, xw, cm, pss, HD, KC, T);
    const int    per  = (KC / 16 + W - 1) / W;
    const size_t dsm  = (size_t) W * 16 * ((((16 * per) >> 5) * 34 + 3) / 4 * 4);
    hc_down_tc_k<NT, W><<<dim3((unsigned) (LR / 16 + 1), (unsigned) KS), 32 * W, dsm, st>>>(wd, wd_nb1, wi, wi_nb1, wi_bf16, xw,
                                                                                      cm, x, wn, part, HD, LR, H, KC, T);
    hc_fin_k<<<T, 352, (size_t) LR * sizeof(float), st>>>(part, pss, lo, scl, dst + (size_t) N * T, N, H, LR, nrows, T, KS,
                                                          0.25f, eps, lo16, lmax);
    constexpr int KW = 2;
    const int     kbw  = ((LR / 32 + KW - 1) / KW + 1) & ~1;   // even: every warp range starts 4-byte aligned
    const size_t  segs = (size_t) (kbw * 34 + 15) / 16 * 16;
    const size_t  usm  = std::max((size_t) KW * H * 16 * segs, (size_t) KW * H * NT * 4 * 32 * sizeof(float));
    hc_up_tc_k<NT, H, KW><<<(unsigned) (N / 16), 32 * KW, usm, st>>>(wu, wu_nb1, lo16, lmax, x, wn, scl, dst, N, LR, T, 0.25f);
}

}  // namespace

namespace ai::kern {

bool q4x_hc_tc_ok(int N, int LR, int T) {
    if (LR > 1024) return false;   // lo16 scratch per column
    const int KC = q4x_hc_chunk(N);
    return KC > 0 && KC % 128 == 0 && LR % 32 == 0 && N % 16 == 0 && T >= 1 && T <= 32;
}

size_t q4x_hc_tc_scratch_bytes(int N, int T) {
    return ((size_t) 4 * N * T * 2 + 255) / 256 * 256 + (size_t) 64 * T * 4 + ((size_t) 1024 * T * 2 + 255) / 256 * 256 + (size_t) T * 4 + 512;
}

void q4x_hc_read_tc(void * stream, const char * wd, int64_t wd_nb1, const char * wu, int64_t wu_nb1, const char * wi,
                    int64_t wi_nb1, int wi_bf16, const float * x, const float * wn, float * part, float * pss, float * lo,
                    float * scl, float * dst, int N, int LR, int T, float eps, void * scratch) {
    cudaStream_t st = (cudaStream_t) stream;
    const int    KC = q4x_hc_chunk(N);
    __half *     xw   = (__half *) scratch;
    float *      cm   = (float *) ((char *) scratch + ((size_t) 4 * N * T * 2 + 255) / 256 * 256);
    __half *     lo16 = (__half *) ((char *) cm + (size_t) 64 * T * 4);
    float *      lmax = (float *) ((char *) lo16 + ((size_t) 1024 * T * 2 + 255) / 256 * 256);
    if (T <= 8)       hc_tc_launch<1>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, xw, cm, lo16, lmax, N, LR, T, eps, KC);
    else if (T <= 16) hc_tc_launch<2>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, xw, cm, lo16, lmax, N, LR, T, eps, KC);
    else              hc_tc_launch<4>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, xw, cm, lo16, lmax, N, LR, T, eps, KC);
}

}  // namespace ai::kern
