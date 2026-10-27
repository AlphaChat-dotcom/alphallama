// Native split-K decode attention over a q4_0 KV cache (AI_FA_NATIVE).
//
// Attention is the largest part of what an MTP verify round costs over a plain one. ggml's
// flash_attn_ext_vec<256, 1> handles one query column per block (ALPHA_FA_BI_MAXQ forces the single-query
// choice so MTP stays bit-identical to plain), so a verify of L tokens re-reads the KV L times, and even
// plain decode reaches only a fraction of the card's memory bandwidth.
//
// Shape: grid (KV split, KV head, stream); one warp per query head of the GQA group, and each warp carries
// all L tokens of its stream. So each KV cell is read once per block and serves G x L query rows. Lane i
// owns dims 8i..8i+7 and dequantises them straight from the q4_0 block (4 lanes per 32-element block), so
// there is no shared memory. Online softmax runs over groups of 4 cells. A second kernel merges splits.
//
// Exactness, and what it is relative to:
//   - Not bit-identical to ggml's kernel, which quantises Q to q8_1 and dots in int8; this one keeps
//     Q in f32 and dequantises K/V exactly (a q4_0 value is (q-8)*d, exact in f32).
//   - MTP == plain by construction: a query row's arithmetic (cell order, split boundaries -- which
//     depend only on n_kv -- butterfly shape, 4-cell grouping) does not depend on L, so a token verified
//     in a batch of 4 gets the same bits it gets alone.
#include "kernels/kernels.h"

#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#define FAD_D     256
#define FAD_GROUP 4

static __device__ __forceinline__ float fad_warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, o, 32);
    }
    return x;
}

// Lane's 8 dims of one q4_0 row of 256: block lane/4 (18 bytes: half d, 16 bytes of nibbles), elements
// (lane%4)*8 .. +7. Elements 0..15 are the low nibbles of qs[0..15], 16..31 the high nibbles.
static __device__ __forceinline__ void fad_load8(const char * __restrict__ row, int lane, float v[8]) {
    const char *     blk = row + (lane >> 2) * 18;
    const float      d   = __half2float(*(const __half *) blk);
    const int        sub = lane & 3;
    const uint16_t * qs  = (const uint16_t *) (blk + 2) + (sub & 1) * 4;   // 2-byte aligned: blocks are 18 B
    const bool       hi  = sub >= 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint16_t w  = qs[i];
        const int      b0 = w & 0xFF, b1 = w >> 8;
        const int      n0 = hi ? (b0 >> 4) : (b0 & 0xF);
        const int      n1 = hi ? (b1 >> 4) : (b1 & 0xF);
        v[2 * i]          = (float) (n0 - 8) * d;
        v[2 * i + 1]      = (float) (n1 - 8) * d;
    }
}

template <int L>
__global__ static void __launch_bounds__(1024)
fa_dec_q40_k(const float * __restrict__ Q, const char * __restrict__ Kc, const char * __restrict__ Vc,
             const __half * __restrict__ mask, float * __restrict__ part_o, float2 * __restrict__ part_ml, int Hq,
             int n_kv, int split, long long cell_stride, long long head_stride, long long stream_stride,
             long long mask_row, long long mask_stream, float scale, int rows_total) {
    const int s = blockIdx.x, hk = blockIdx.y, b = blockIdx.z;
    const int g = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = blockDim.x >> 5;
    const int h = hk * G + g;

    float q[L][8];
#pragma unroll
    for (int l = 0; l < L; ++l) {
        const float4 * qr = (const float4 *) (Q + ((size_t) ((b * L + l) * Hq + h)) * FAD_D + lane * 8);
        const float4   a = qr[0], c = qr[1];
        q[l][0] = a.x; q[l][1] = a.y; q[l][2] = a.z; q[l][3] = a.w;
        q[l][4] = c.x; q[l][5] = c.y; q[l][6] = c.z; q[l][7] = c.w;
    }
    float m[L], ls[L], o[L][8];
#pragma unroll
    for (int l = 0; l < L; ++l) {
        m[l]  = -CUDART_INF_F;
        ls[l] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[l][i] = 0.0f;
        }
    }

    const long long base = (long long) b * stream_stride + (long long) hk * head_stride;
    const char *    kb   = Kc + base;
    const char *    vb   = Vc + base;
    const __half *  mb   = mask + (long long) b * mask_stream;
    const int       c0   = s * split;
    const int       c1   = min(c0 + split, n_kv);

    for (int c = c0; c < c1; c += FAD_GROUP) {
        float sc[L][FAD_GROUP];
#pragma unroll
        for (int j = 0; j < FAD_GROUP; ++j) {
            const int cell = c + j;
            if (cell < c1) {
                float kv[8];
                fad_load8(kb + (long long) cell * cell_stride, lane, kv);
#pragma unroll
                for (int l = 0; l < L; ++l) {
                    float dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        dot = fmaf(q[l][i], kv[i], dot);
                    }
                    dot         = fad_warp_sum(dot);
                    sc[l][j]    = dot * scale + __half2float(mb[(long long) l * mask_row + cell]);
                }
            } else {
#pragma unroll
                for (int l = 0; l < L; ++l) {
                    sc[l][j] = -CUDART_INF_F;
                }
            }
        }
        float p[L][FAD_GROUP];
#pragma unroll
        for (int l = 0; l < L; ++l) {
            float mx = sc[l][0];
#pragma unroll
            for (int j = 1; j < FAD_GROUP; ++j) {
                mx = fmaxf(mx, sc[l][j]);
            }
            const float mn = fmaxf(m[l], mx);
            if (mn == -CUDART_INF_F) {   // nothing visible yet for this row
#pragma unroll
                for (int j = 0; j < FAD_GROUP; ++j) {
                    p[l][j] = 0.0f;
                }
                continue;
            }
            const float corr = expf(m[l] - mn);   // exp(-inf) = 0 on the first visible group
            float       sum  = 0.0f;
#pragma unroll
            for (int j = 0; j < FAD_GROUP; ++j) {
                p[l][j] = expf(sc[l][j] - mn);
                sum += p[l][j];
            }
            ls[l] = ls[l] * corr + sum;
            m[l]  = mn;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                o[l][i] *= corr;
            }
        }
#pragma unroll
        for (int j = 0; j < FAD_GROUP; ++j) {
            const int cell = c + j;
            if (cell < c1) {
                float vv[8];
                fad_load8(vb + (long long) cell * cell_stride, lane, vv);
#pragma unroll
                for (int l = 0; l < L; ++l) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o[l][i] = fmaf(p[l][j], vv[i], o[l][i]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int l = 0; l < L; ++l) {
        const int row = (b * L + l) * Hq + h;
        float4 *  po  = (float4 *) (part_o + ((size_t) s * rows_total + row) * FAD_D + lane * 8);
        po[0]         = make_float4(o[l][0], o[l][1], o[l][2], o[l][3]);
        po[1]         = make_float4(o[l][4], o[l][5], o[l][6], o[l][7]);
        if (lane == 0) {
            part_ml[(size_t) s * rows_total + row] = make_float2(m[l], ls[l]);
        }
    }
}

__global__ static void fa_combine_k(const float * __restrict__ part_o, const float2 * __restrict__ part_ml,
                                    float * __restrict__ out, int n_split, int rows_total) {
    const int row = blockIdx.x, d = threadIdx.x;
    float     M   = -CUDART_INF_F;
    for (int s = 0; s < n_split; ++s) {
        M = fmaxf(M, part_ml[(size_t) s * rows_total + row].x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float2 ml = part_ml[(size_t) s * rows_total + row];
        if (ml.x == -CUDART_INF_F) {
            continue;
        }
        const float w = expf(ml.x - M);
        num           = fmaf(part_o[((size_t) s * rows_total + row) * FAD_D + d], w, num);
        den           = fmaf(ml.y, w, den);
    }
    out[(size_t) row * FAD_D + d] = den > 0.0f ? num / den : 0.0f;
}

namespace ai::kern {

int64_t fa_decode_q40_workspace(int64_t n_kv, int64_t rows_total, int split) {
    const int64_t n_split = (n_kv + split - 1) / split;
    return n_split * rows_total * (FAD_D + 2);   // floats: partial O plus (m, l)
}

bool fa_decode_q40(void * stream, const float * q, const void * k, const void * v, const void * mask, float * out,
                   float * ws, int64_t D, int64_t Hq, int64_t Hkv, int64_t L, int64_t B, int64_t n_kv, int split,
                   int64_t cell_stride, int64_t head_stride, int64_t stream_stride, int64_t mask_row,
                   int64_t mask_stream, float scale) {
    const int G = (int) (Hq / Hkv);
    if (D != FAD_D || Hq % Hkv != 0 || G * 32 > 1024 || L < 1 || L > 8 || split % FAD_GROUP != 0) {
        return false;
    }
    const int n_split    = (int) ((n_kv + split - 1) / split);
    const int rows_total = (int) (B * L * Hq);
    float *   part_o     = ws;
    float2 *  part_ml    = (float2 *) (ws + (size_t) n_split * rows_total * FAD_D);
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[fa-native] LAUNCHED (splits %d x kv heads %lld x streams %lld, %d warps, L=%lld, n_kv %lld)\n",
                n_split, (long long) Hkv, (long long) B, G, (long long) L, (long long) n_kv);
    }
    const dim3   grid((unsigned) n_split, (unsigned) Hkv, (unsigned) B);
    const dim3   blk((unsigned) (G * 32));
    cudaStream_t st = (cudaStream_t) stream;
    const char * kc = (const char *) k;
    const char * vc = (const char *) v;
    const __half * mk = (const __half *) mask;
#define FAD_LAUNCH(LL)                                                                                          \
    fa_dec_q40_k<LL><<<grid, blk, 0, st>>>(q, kc, vc, mk, part_o, part_ml, (int) Hq, (int) n_kv, split,         \
                                           cell_stride, head_stride, stream_stride, mask_row, mask_stream,       \
                                           scale, rows_total)
    switch (L) {
        case 1: FAD_LAUNCH(1); break;
        case 2: FAD_LAUNCH(2); break;
        case 3: FAD_LAUNCH(3); break;
        case 4: FAD_LAUNCH(4); break;
        case 5: FAD_LAUNCH(5); break;
        case 6: FAD_LAUNCH(6); break;
        case 7: FAD_LAUNCH(7); break;
        default: FAD_LAUNCH(8); break;
    }
#undef FAD_LAUNCH
    fa_combine_k<<<rows_total, FAD_D, 0, st>>>(part_o, part_ml, out, n_split, rows_total);
    return true;
}

}   // namespace ai::kern
