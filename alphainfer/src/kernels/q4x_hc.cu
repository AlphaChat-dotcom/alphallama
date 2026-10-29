// SPDX-License-Identifier: Apache-2.0
// Native qwen4exp hyper-connection read: see q4x_hc.h. Kernels down6 / down4_fin / up_mix5 are ggml's hc-fuse.cu
// V4 kernels (ALPHA_HC_FUSE_V4, ALPHA_HC_DOWN_RR=2) verbatim; up_mix9 is up_mix5 with the warp's four weight rows
// staged into shared memory by batched 4-byte loads (~10-15% faster with weights from DRAM, bitwise identical output).
#include "kernels/q4x_hc.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace {

__device__ __forceinline__ float hcf_bf16(uint32_t b) { return __uint_as_float((b & 0xffffu) << 16); }

// 8 consecutive weights (vector v of a Q8_0 row: 34-byte blocks {half d; int8 qs[32]}, 4 vectors per block)
__device__ __forceinline__ void hcf_load8(const char * __restrict__ row, const int v, float * wf) {
    const char * blk = row + (v >> 2)*34;
    const float d = __half2float(*(const half *) blk);
    const unsigned short * qs = (const unsigned short *) (blk + 2 + (v & 3)*8);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const unsigned short u = qs[k];
        wf[2*k + 0] = d*(float) (int8_t) (u & 0xff);
        wf[2*k + 1] = d*(float) (int8_t) (u >> 8);
    }
}

template <int NT, int RW, int RR>
__global__ void hcf_down6(const char * __restrict__ wd, const int64_t wd_nb1, const char * __restrict__ wi,
                          const int64_t wi_nb1, const int wi_bf16,
                          const float * __restrict__ x, const float * __restrict__ wn,
                          float * __restrict__ part, float * __restrict__ pss,
                          const int N, const int H, const int LR, const int T, const int KC) {
    extern __shared__ float sm[];          // xs[T][KC], wns[KC]
    float * xs  = sm;
    float * wns = sm + (size_t) T*KC;
    const int jg = blockIdx.x, ks = blockIdx.y;
    const int HD = N*H;
    const int k0 = ks*KC;
    const int nrows = LR + (wi ? H : 0);
    for (int i = threadIdx.x; i < KC/4; i += blockDim.x) {
        ((float4 *) wns)[i] = ((const float4 *) (wn + k0))[i];
        for (int t = 0; t < T; ++t) ((float4 *) (xs + (size_t) t*KC))[i] = ((const float4 *) (x + (size_t) t*HD + k0))[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (jg == 0) {
        __shared__ float red[NT][32];
        for (int t = 0; t < T; ++t) {
            float q = 0.0f;
            for (int i = threadIdx.x; i < KC; i += blockDim.x) { const float v = xs[(size_t) t*KC + i]; q = fmaf(v, v, q); }
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
            if (lane == 0) red[t][w] = q;
        }
        __syncthreads();
        if (threadIdx.x < T) {
            float q = 0.0f;
            for (int i = 0; i < (int) (blockDim.x >> 5); ++i) q += red[threadIdx.x][i];
            pss[ks*T + threadIdx.x] = q;
        }
    }
    const int j0 = (jg*RW + w)*RR;
    if (j0 >= nrows) return;
    float acc[RR][NT];
#pragma unroll
    for (int r = 0; r < RR; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
    const int nv = KC/8;
    for (int v = lane; v < nv; v += 32) {
        const int vg = k0/8 + v;
        const float4 n0 = ((const float4 *) wns)[2*v + 0];
        const float4 n1 = ((const float4 *) wns)[2*v + 1];
        float4 xa[NT], xb[NT];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                xa[t] = ((const float4 *) (xs + (size_t) t*KC))[2*v + 0];
                xb[t] = ((const float4 *) (xs + (size_t) t*KC))[2*v + 1];
            }
        }
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            const int j = j0 + r;
            if (j >= nrows) break;
            float wf[8];
            if (j < LR) {
                hcf_load8(wd + (int64_t) j*wd_nb1, vg, wf);
            } else {
                const char * row = wi + (int64_t) (j - LR)*wi_nb1;
                if (wi_bf16) {
                    const uint4 q = ((const uint4 *) row)[vg];
                    wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
                    wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
                    wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
                    wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) wf[e] = ((const float *) row)[vg*8 + e];
                }
            }
            wf[0] *= n0.x; wf[1] *= n0.y; wf[2] *= n0.z; wf[3] *= n0.w;
            wf[4] *= n1.x; wf[5] *= n1.y; wf[6] *= n1.z; wf[7] *= n1.w;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float4 a = xa[t], b = xb[t];
                    float d = acc[r][t];
                    d = fmaf(wf[0], a.x, d); d = fmaf(wf[1], a.y, d); d = fmaf(wf[2], a.z, d); d = fmaf(wf[3], a.w, d);
                    d = fmaf(wf[4], b.x, d); d = fmaf(wf[5], b.y, d); d = fmaf(wf[6], b.z, d); d = fmaf(wf[7], b.w, d);
                    acc[r][t] = d;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RR; ++r) {
        const int j = j0 + r;
        if (j >= nrows) break;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[r][t];
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
            if (lane == 0 && t < T) part[((size_t) ks*T + t)*nrows + j] = a;
        }
    }
}

// AI_Q4X_HC_FIN_FUSE (default 0): down6 + the finalize in one kernel. Every block publishes its partials, and the
// last block to finish (a device counter, reset by that block) runs down4_fin's loops for every token: the same
// operations in the same order, so lo / scl / inject are bit-identical. Off by default because the last block's
// serial finalize costs more than the separate launch.
__device__ unsigned int g_hc_done = 0;

template <int NT, int RW, int RR>
__global__ void hcf_down6f(const char * __restrict__ wd, const int64_t wd_nb1, const char * __restrict__ wi,
                           const int64_t wi_nb1, const int wi_bf16,
                           const float * __restrict__ x, const float * __restrict__ wn,
                           float * __restrict__ part, float * __restrict__ pss,
                           float * __restrict__ lo, float * __restrict__ scl_out, float * __restrict__ inj_out,
                           const int N, const int H, const int LR, const int T, const int KC, const float inv_h,
                           const float eps);

__global__ void hcf_down4_fin(const float * __restrict__ part, const float * __restrict__ pss,
                              float * __restrict__ lo, float * __restrict__ scl_out, float * __restrict__ inj_out,
                              const int N, const int H, const int LR, const int nrows, const int T, const int KS,
                              const float inv_h, const float eps) {
    const int t = blockIdx.x;
    const int cps = KS / H;
    __shared__ float scl[8];
    if (threadIdx.x < H) {
        float q = 0.0f;
        for (int c = 0; c < cps; ++c) q += pss[(threadIdx.x*cps + c)*T + t];
        const float s = 1.0f/sqrtf(q/(float) N + eps);
        scl[threadIdx.x] = s;
        scl_out[t*H + threadIdx.x] = s;
    }
    __syncthreads();
    for (int j = threadIdx.x; j < nrows; j += blockDim.x) {
        float v = 0.0f;
        for (int s = 0; s < H; ++s) {
            float a = 0.0f;
            for (int c = 0; c < cps; ++c) a += part[((size_t) (s*cps + c)*T + t)*nrows + j];
            v += a*scl[s];
        }
        if (j < LR) {
            const float a = v*inv_h;
            lo[(int64_t) t*LR + j] = a/(1.0f + expf(-a));
        } else {
            inj_out[(int64_t) t*H + (j - LR)] = v;
        }
    }
}

template <int NT, int RW, int RR>
__global__ void hcf_down6f(const char * __restrict__ wd, const int64_t wd_nb1, const char * __restrict__ wi,
                           const int64_t wi_nb1, const int wi_bf16,
                           const float * __restrict__ x, const float * __restrict__ wn,
                           float * __restrict__ part, float * __restrict__ pss,
                           float * __restrict__ lo, float * __restrict__ scl_out, float * __restrict__ inj_out,
                           const int N, const int H, const int LR, const int T, const int KC, const float inv_h,
                           const float eps) {
    extern __shared__ float sm[];
    float * xs  = sm;
    float * wns = sm + (size_t) T*KC;
    const int jg = blockIdx.x, ks = blockIdx.y;
    const int HD = N*H;
    const int k0 = ks*KC;
    const int nrows = LR + (wi ? H : 0);
    for (int i = threadIdx.x; i < KC/4; i += blockDim.x) {
        ((float4 *) wns)[i] = ((const float4 *) (wn + k0))[i];
        for (int t = 0; t < T; ++t) ((float4 *) (xs + (size_t) t*KC))[i] = ((const float4 *) (x + (size_t) t*HD + k0))[i];
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (jg == 0) {
        __shared__ float red[NT][32];
        for (int t = 0; t < T; ++t) {
            float q = 0.0f;
            for (int i = threadIdx.x; i < KC; i += blockDim.x) { const float v = xs[(size_t) t*KC + i]; q = fmaf(v, v, q); }
            for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
            if (lane == 0) red[t][w] = q;
        }
        __syncthreads();
        if (threadIdx.x < T) {
            float q = 0.0f;
            for (int i = 0; i < (int) (blockDim.x >> 5); ++i) q += red[threadIdx.x][i];
            pss[ks*T + threadIdx.x] = q;
        }
    }
    const int j0 = (jg*RW + w)*RR;
    if (j0 < nrows) {
        float acc[RR][NT];
#pragma unroll
        for (int r = 0; r < RR; ++r)
#pragma unroll
            for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;
        const int nv = KC/8;
        for (int v = lane; v < nv; v += 32) {
            const int vg = k0/8 + v;
            const float4 n0 = ((const float4 *) wns)[2*v + 0];
            const float4 n1 = ((const float4 *) wns)[2*v + 1];
            float4 xa[NT], xb[NT];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    xa[t] = ((const float4 *) (xs + (size_t) t*KC))[2*v + 0];
                    xb[t] = ((const float4 *) (xs + (size_t) t*KC))[2*v + 1];
                }
            }
#pragma unroll
            for (int r = 0; r < RR; ++r) {
                const int j = j0 + r;
                if (j >= nrows) break;
                float wf[8];
                if (j < LR) {
                    hcf_load8(wd + (int64_t) j*wd_nb1, vg, wf);
                } else {
                    const char * row = wi + (int64_t) (j - LR)*wi_nb1;
                    if (wi_bf16) {
                        const uint4 q = ((const uint4 *) row)[vg];
                        wf[0] = hcf_bf16(q.x); wf[1] = hcf_bf16(q.x >> 16);
                        wf[2] = hcf_bf16(q.y); wf[3] = hcf_bf16(q.y >> 16);
                        wf[4] = hcf_bf16(q.z); wf[5] = hcf_bf16(q.z >> 16);
                        wf[6] = hcf_bf16(q.w); wf[7] = hcf_bf16(q.w >> 16);
                    } else {
#pragma unroll
                        for (int e = 0; e < 8; ++e) wf[e] = ((const float *) row)[vg*8 + e];
                    }
                }
                wf[0] *= n0.x; wf[1] *= n0.y; wf[2] *= n0.z; wf[3] *= n0.w;
                wf[4] *= n1.x; wf[5] *= n1.y; wf[6] *= n1.z; wf[7] *= n1.w;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (t < T) {
                        const float4 a = xa[t], b = xb[t];
                        float d = acc[r][t];
                        d = fmaf(wf[0], a.x, d); d = fmaf(wf[1], a.y, d); d = fmaf(wf[2], a.z, d); d = fmaf(wf[3], a.w, d);
                        d = fmaf(wf[4], b.x, d); d = fmaf(wf[5], b.y, d); d = fmaf(wf[6], b.z, d); d = fmaf(wf[7], b.w, d);
                        acc[r][t] = d;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RR; ++r) {
            const int j = j0 + r;
            if (j >= nrows) break;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                float a = acc[r][t];
                for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffff, a, o);
                if (lane == 0 && t < T) part[((size_t) ks*T + t)*nrows + j] = a;
            }
        }
    }
    // publish, then the last block finalises
    __threadfence();
    __shared__ unsigned int s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned int nblk = gridDim.x*gridDim.y;
        s_last = atomicAdd(&g_hc_done, 1u) == nblk - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int KS = gridDim.y, cps = KS / H;
    __shared__ float scl[8];
    for (int t = 0; t < T; ++t) {
        if (threadIdx.x < H) {
            float q = 0.0f;
            for (int c = 0; c < cps; ++c) q += __ldcg(pss + (threadIdx.x*cps + c)*T + t);
            const float s = 1.0f/sqrtf(q/(float) N + eps);
            scl[threadIdx.x] = s;
            scl_out[t*H + threadIdx.x] = s;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < nrows; j += blockDim.x) {
            float v = 0.0f;
            for (int s = 0; s < H; ++s) {
                float a = 0.0f;
                for (int c = 0; c < cps; ++c) a += __ldcg(part + ((size_t) (s*cps + c)*T + t)*nrows + j);
                v += a*scl[s];
            }
            if (j < LR) {
                const float a = v*inv_h;
                lo[(int64_t) t*LR + j] = a/(1.0f + expf(-a));
            } else {
                inj_out[(int64_t) t*H + (j - LR)] = v;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) g_hc_done = 0;
}

// the up-mix accumulation and epilogue shared by up_mix5 (rows read in place) and up_mix9 (rows staged)
template <int NT, int H>
__device__ __forceinline__ void hcf_up_body(const char * const * rows, const float * __restrict__ slo,
                                            const float * __restrict__ x, const float * __restrict__ wn,
                                            const float * __restrict__ scl, float * __restrict__ dst, const int N,
                                            const int LR, const int T, const float inv_h, const int d, const int lane) {
    const int HD = N*H;
    const int n8 = LR/8;
    float g[H][NT];
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) g[s][t] = 0.0f;
    for (int v = lane; v < n8; v += 32) {
        float l[NT][8];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                const float4 p = ((const float4 *) (slo + t*LR + v*8))[0];
                const float4 q = ((const float4 *) (slo + t*LR + v*8))[1];
                l[t][0] = p.x; l[t][1] = p.y; l[t][2] = p.z; l[t][3] = p.w;
                l[t][4] = q.x; l[t][5] = q.y; l[t][6] = q.z; l[t][7] = q.w;
            }
        }
#pragma unroll
        for (int s = 0; s < H; ++s) {
            float wf[8];
            hcf_load8(rows[s], v, wf);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    float a = g[s][t];
#pragma unroll
                    for (int e = 0; e < 8; ++e) a = fmaf(wf[e], l[t][e], a);
                    g[s][t] = a;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t)
            for (int o = 16; o > 0; o >>= 1) g[s][t] += __shfl_xor_sync(0xffffffff, g[s][t], o);
    if (lane < T && lane < NT) {
        float acc = 0.0f;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t == lane) {
#pragma unroll
                for (int s = 0; s < H; ++s) {
                    const float xn = x[(int64_t) t*HD + s*N + d]*scl[t*H + s]*wn[s*N + d];
                    acc += xn*(1.0f/(1.0f + expf(-g[s][t])));
                }
            }
        }
        dst[(int64_t) lane*N + d] = acc*inv_h;
    }
}

template <int NT, int H>
__global__ void hcf_up_mix5(const char * __restrict__ wu, const int64_t wu_nb1, const float * __restrict__ lo,
                            const float * __restrict__ x, const float * __restrict__ wn, const float * __restrict__ scl,
                            float * __restrict__ dst, const int N, const int LR, const int T, const float inv_h) {
    extern __shared__ float slo[];
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int d = blockIdx.x*(blockDim.x >> 5) + wid;
    if (d >= N) return;
    const char * rows[H];
#pragma unroll
    for (int s = 0; s < H; ++s) rows[s] = wu + (int64_t) (s*N + d)*wu_nb1;
    hcf_up_body<NT, H>(rows, slo, x, wn, scl, dst, N, LR, T, inv_h, d, lane);
}

// WPL = 4-byte words per lane per row (ceil(nb1/4 / 32)); rows start 4-byte aligned (nb1 % 4 == 0)
template <int NT, int H, int WPL>
__global__ void hcf_up_mix9(const char * __restrict__ wu, const int64_t wu_nb1, const float * __restrict__ lo,
                            const float * __restrict__ x, const float * __restrict__ wn, const float * __restrict__ scl,
                            float * __restrict__ dst, const int N, const int LR, const int T, const float inv_h) {
    extern __shared__ float smem[];
    float * slo = smem;                               // [T][LR] (NT*LR reserved)
    char *  sw  = (char *) (smem + NT*LR);            // [warps][H][nb1]
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int d = blockIdx.x*(blockDim.x >> 5) + wid;
    char * my = sw + (size_t) wid*H*wu_nb1;
    const int nw = (int) (wu_nb1/4);
    if (d < N) {
        uint32_t buf[H][WPL];
#pragma unroll
        for (int s = 0; s < H; ++s) {
            const uint32_t * src = (const uint32_t *) (wu + (int64_t) (s*N + d)*wu_nb1);
#pragma unroll
            for (int k = 0; k < WPL; ++k) { const int i = lane + 32*k; if (i < nw) buf[s][k] = __ldg(src + i); }
        }
#pragma unroll
        for (int s = 0; s < H; ++s) {
            uint32_t * dw = (uint32_t *) (my + s*wu_nb1);
#pragma unroll
            for (int k = 0; k < WPL; ++k) { const int i = lane + 32*k; if (i < nw) dw[i] = buf[s][k]; }
        }
    }
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    if (d >= N) return;
    const char * rows[H];
#pragma unroll
    for (int s = 0; s < H; ++s) rows[s] = my + s*wu_nb1;
    hcf_up_body<NT, H>(rows, slo, x, wn, scl, dst, N, LR, T, inv_h, d, lane);
}


// AI_Q4X_HC_UP=10: the up-mix with four outputs per warp, 8 lanes each: an output d reads its H rows of LR
// Q8_0 weights as LR/8 eight-element vectors, 5 per lane at LR = 320 (up_mix9 gives 32 lanes 40 vectors: the second
// pass runs 8 of them), and each (stream, column) partial is reduced over 8 lanes with 3 shuffles instead of 32 lanes
// with 5 (up_mix9 at T = 3: 60 dependent shuffles per output). Loads straight from global/L2 in 8-byte vectors.
// Every column runs the same instructions whatever T is (column-invariant); not bit-identical to up_mix9 (a different
// partial-sum tree), so quality is checked by paired NLL.
template <int NT, int H>
__global__ void __launch_bounds__(128) hcf_up_mix10(const char * __restrict__ wu, const int64_t wu_nb1,
                                                    const float * __restrict__ lo, const float * __restrict__ x,
                                                    const float * __restrict__ wn, const float * __restrict__ scl,
                                                    float * __restrict__ dst, const int N, const int LR, const int T,
                                                    const float inv_h) {
    extern __shared__ float slo[];   // [T][LR]
    for (int i = threadIdx.x; i < T*LR; i += blockDim.x) slo[i] = lo[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, grp = lane >> 3, sl = lane & 7;
    const int d = (blockIdx.x*(blockDim.x >> 5) + wid)*4 + grp;
    if (d >= N) return;
    const int HD = N*H, n8 = LR/8;
    float g[H][NT];
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) g[s][t] = 0.0f;
#pragma unroll
    for (int s = 0; s < H; ++s) {
        const char * row = wu + (int64_t) (s*N + d)*wu_nb1;
        for (int v = sl; v < n8; v += 8) {
            float wf[8];
            hcf_load8(row, v, wf);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if (t < T) {
                    const float4 p = ((const float4 *) (slo + t*LR + v*8))[0];
                    const float4 q = ((const float4 *) (slo + t*LR + v*8))[1];
                    float a = g[s][t];
                    a = fmaf(wf[0], p.x, a); a = fmaf(wf[1], p.y, a); a = fmaf(wf[2], p.z, a); a = fmaf(wf[3], p.w, a);
                    a = fmaf(wf[4], q.x, a); a = fmaf(wf[5], q.y, a); a = fmaf(wf[6], q.z, a); a = fmaf(wf[7], q.w, a);
                    g[s][t] = a;
                }
            }
        }
    }
#pragma unroll
    for (int s = 0; s < H; ++s)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = g[s][t];
            a += __shfl_xor_sync(0xffffffff, a, 4);
            a += __shfl_xor_sync(0xffffffff, a, 2);
            a += __shfl_xor_sync(0xffffffff, a, 1);
            g[s][t] = a;
        }
    if (sl == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (t < T) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < H; ++s) {
                    const float xn = x[(int64_t) t*HD + s*N + d]*scl[t*H + s]*wn[s*N + d];
                    acc += xn*(1.0f/(1.0f + expf(-g[s][t])));
                }
                dst[(int64_t) t*N + d] = acc*inv_h;
            }
        }
    }
}

int g_hc_up_override = -1;
int hc_up_mode() {
    if (g_hc_up_override >= 0) return g_hc_up_override;
    static const int v = getenv("AI_Q4X_HC_UP") ? atoi(getenv("AI_Q4X_HC_UP")) : 9;
    return v;
}

template <int NT>
void launch(cudaStream_t st, const char * wd, int64_t wd_nb1, const char * wu, int64_t wu_nb1, const char * wi,
            int64_t wi_nb1, int wi_bf16, const float * x, const float * wn, float * part, float * pss, float * lo,
            float * scl, float * dst, int N, int LR, int T, float eps, int KC) {
    constexpr int H = 4, RW = 8, RR = 2;
    const int nrows = LR + (wi ? H : 0);
    const int KS = H*N / KC;
    float * inj = dst + (size_t) N*T;
    const size_t dsm = (size_t) (T + 1)*KC*sizeof(float);
    static const bool fin_fuse = getenv("AI_Q4X_HC_FIN_FUSE") && atoi(getenv("AI_Q4X_HC_FIN_FUSE")) != 0;
    if (fin_fuse) {
        hcf_down6f<NT, RW, RR><<<dim3((nrows + RW*RR - 1)/(RW*RR), KS), RW*32, dsm, st>>>(
            wd, wd_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, inj, N, H, LR, T, KC, 0.25f, eps);
    } else {
        hcf_down6<NT, RW, RR><<<dim3((nrows + RW*RR - 1)/(RW*RR), KS), RW*32, dsm, st>>>(
            wd, wd_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, N, H, LR, T, KC);
        hcf_down4_fin<<<T, 352, 0, st>>>(part, pss, lo, scl, inj, N, H, LR, nrows, T, KS, 0.25f, eps);
    }
    if (hc_up_mode() == 10 && LR % 8 == 0) {
        const int nb10 = (N + 15)/16;   // 4 warps x 4 outputs per block
        hcf_up_mix10<NT, H><<<nb10, 128, (size_t) T*LR*sizeof(float), st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f);
        return;
    }
    const int words = (int) (wu_nb1/4), wpl = (words + 31)/32;
    const size_t usm9 = (size_t) NT*LR*sizeof(float) + (size_t) 8*H*wu_nb1;
    const int    nb   = (N + 7)/8;
    if (wu_nb1 % 4 == 0 && usm9 <= 48*1024) {
        switch (wpl) {
            case 1: hcf_up_mix9<NT, H, 1><<<nb, 256, usm9, st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f); return;
            case 2: hcf_up_mix9<NT, H, 2><<<nb, 256, usm9, st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f); return;
            case 3: hcf_up_mix9<NT, H, 3><<<nb, 256, usm9, st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f); return;
            case 4: hcf_up_mix9<NT, H, 4><<<nb, 256, usm9, st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f); return;
            case 5: hcf_up_mix9<NT, H, 5><<<nb, 256, usm9, st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f); return;
            default: break;
        }
    }
    hcf_up_mix5<NT, H><<<nb, 256, (size_t) T*LR*sizeof(float), st>>>(wu, wu_nb1, lo, x, wn, scl, dst, N, LR, T, 0.25f);
}

}  // namespace

namespace ai::kern {

void q4x_hc_set_up(int v) { g_hc_up_override = v; }

int q4x_hc_chunk(int N) {
    for (int c = 1; c <= 8; ++c) {
        if (N % c == 0 && (N / c) % 32 == 0 && N / c <= 1280) return N / c;
    }
    return 0;
}

void q4x_hc_ws(int N, int LR, int T, bool inject, int64_t * part_n, int64_t * pss_n) {
    const int KC = q4x_hc_chunk(N), KS = KC ? 4*N / KC : 0;
    *part_n = (int64_t) KS*T*(LR + (inject ? 4 : 0));
    *pss_n  = (int64_t) KS*T;
}

void q4x_hc_read(void * stream, const char * wd, int64_t wd_nb1, const char * wu, int64_t wu_nb1, const char * wi,
                 int64_t wi_nb1, int wi_bf16, const float * x, const float * wn, float * part, float * pss, float * lo,
                 float * scl, float * dst, int N, int LR, int T, float eps) {
    cudaStream_t st = (cudaStream_t) stream;
    const int    KC = q4x_hc_chunk(N);
    switch (T) {
        case 1:  launch<1>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, N, LR, T, eps, KC); break;
        case 2:  launch<2>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, N, LR, T, eps, KC); break;
        case 3:  launch<3>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, N, LR, T, eps, KC); break;
        case 4:  launch<4>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, N, LR, T, eps, KC); break;
        default: launch<8>(st, wd, wd_nb1, wu, wu_nb1, wi, wi_nb1, wi_bf16, x, wn, part, pss, lo, scl, dst, N, LR, T, eps, KC); break;
    }
}

// Prefill: hc_combine in place, R[e, h, t] += out[e, t] * (s_out * sigmoid(s_in * inj[h, t])), with ggml-cuda's
// hc_combine_f32 float operations in the same order (bit-identical). R is [N, H, T] contiguous, out rows out_ld floats
// apart, inject rows inj_ld floats apart. Saves the copy back that the graph op's fresh output needs.
__global__ static void hc_combine_into_k(float * __restrict__ R, const float * __restrict__ out, const float * __restrict__ inj,
                                         float s_in, float s_out, int64_t N, int64_t H, int64_t T, int64_t out_ld, int64_t inj_ld) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * H * T) return;
    const int64_t e = i % N, r = i / N, h = r % H, t = r / H;
    const float   c = inj[t * inj_ld + h];
    const float   w = __fmaf_rn(s_out, 1.0f / (1.0f + expf(-__fmaf_rn(s_in, c, 0.0f))), 0.0f);
    R[i] = __fadd_rn(R[i], __fmul_rn(out[t * out_ld + e], w));
}

void q4x_hc_combine_into(void * stream, float * R, const float * out, const float * inj, float s_in, float s_out, int64_t N,
                         int64_t H, int64_t T, int64_t out_ld, int64_t inj_ld) {
    const int64_t n = N * H * T;
    if (n <= 0) return;
    hc_combine_into_k<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(R, out, inj, s_in, s_out, N, H, T, out_ld, inj_ld);
}

}  // namespace ai::kern
