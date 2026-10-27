// CUDA dequantization for the ik_llama.cpp quant types.
// Correct-but-slow: one CUDA thread per output row runs the exact scalar dequant that
// ggml-quants-ik.c uses on the CPU (row-meta aware: consumes the per-row float/half
// scale prefix of iq4_ks/iq5_ks/iq3_ks/iq1_kt). No fast trellis MMVQ yet; this feeds the
// dequantize->cuBLAS backbone path and the pool's F16 expert compute.
//
// Ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp), MIT license,
// Copyright (c) Iwan Kawrakow.
#include "common.cuh"
#include "convert.cuh"

// Reinterpret the 2 raw bytes at p as an IEEE half and widen to float. Takes a pointer
// so it works whether ggml_half is `half` (CUDA) or `uint16_t` (host) - passing the value
// would value-convert a CUDA `half` field (1.0 -> int 1 -> denorm) instead of keeping bits.
static __device__ __forceinline__ float ik_h2f(const void * p) { return __half2float(*(const __half *) p); }

template <typename dst_t> static __device__ __forceinline__ void ik_store(dst_t * p, float v);
template <> __device__ __forceinline__ void ik_store<float>(float * p, float v) { *p = v; }
template <> __device__ __forceinline__ void ik_store<half>(half * p, float v) { *p = __float2half(v); }

// IQ4_K (no row prefix)
template <typename dst_t>
static __device__ void deq_row_iq4_k(const uint8_t * row, dst_t * y, int ne) {
    const block_iq4_k * x = (const block_iq4_k *) row;
    const int nb = ne / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = ik_h2f(&x[i].d);
        uint16_t extra = x[i].extra;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * v1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * v2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                ik_store(y + j + 0,  dl1 * v1[qs[j] & 0xf]);
                ik_store(y + j + 16, dl2 * v2[qs[j] >> 4]);
            }
            y  += 32;
            qs += 16;
        }
    }
}

// IQ6_K (no row prefix)
#define A_IQ6K -127.f
#define B_IQ6K 6.2568f
#define C_IQ6K 0.11218f
#define D_IQ6K 0.0011972f
#define S_IQ6K 1.f
template <typename dst_t>
static __device__ void deq_row_iq6_k(const uint8_t * row, dst_t * y, int ne) {
    const block_iq6_k * x = (const block_iq6_k *) row;
    const int nb = ne / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = ik_h2f(&x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;
        uint16_t extra = x[i].extra;
        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * sl[4*ib64 + 0], dl2 = d * sl[4*ib64 + 1];
            float dl3 = d * sl[4*ib64 + 2], dl4 = d * sl[4*ib64 + 3];
            float m1 = extra & 1 ? S_IQ6K : 0, m2 = extra & 2 ? S_IQ6K : 0;
            float m3 = extra & 4 ? S_IQ6K : 0, m4 = extra & 8 ? S_IQ6K : 0;
            for (int j = 0; j < 16; ++j) {
                float q1 = (float)((qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4));
                float q2 = (float)((qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4));
                float q3 = (float)((qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2));
                float q4 = (float)((qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2));
                ik_store(y + j + 0,  dl1 * (A_IQ6K + q1*(B_IQ6K + q1*(-C_IQ6K + q1*D_IQ6K)) + m1));
                ik_store(y + j + 16, dl2 * (A_IQ6K + q2*(B_IQ6K + q2*(-C_IQ6K + q2*D_IQ6K)) + m2));
                ik_store(y + j + 32, dl3 * (A_IQ6K + q3*(B_IQ6K + q3*(-C_IQ6K + q3*D_IQ6K)) + m3));
                ik_store(y + j + 48, dl4 * (A_IQ6K + q4*(B_IQ6K + q4*(-C_IQ6K + q4*D_IQ6K)) + m4));
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }
    }
}

// IQ4_KS (float row prefix)
template <typename dst_t>
static __device__ void deq_row_iq4_ks(const uint8_t * row, dst_t * y, int ne) {
    const int kBlockSize = 32;
    const float * dptr = (const float *) row;
    const float d = *dptr;
    const block_iq4_ks * x = (const block_iq4_ks *)(dptr + 1);
    const int nblock = ne / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * v = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                ik_store(y + j,              dl * v[qs[j] & 0xf]);
                ik_store(y + j+kBlockSize/2, dl * v[qs[j] >> 4]);
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

// IQ5_KS (float row prefix)
template <typename dst_t>
static __device__ void deq_row_iq5_ks(const uint8_t * row, dst_t * y, int ne) {
    const int kBlockSize = 32;
    const float * dptr = (const float *) row;
    const float d = *dptr;
    const block_iq5_ks * x = (const block_iq5_ks *)(dptr + 1);
    const int nblock = ne / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            const float dl1 = d * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            const float dl2 = d * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * v1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * v2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            for (int j = 0; j < kBlockSize; ++j) {
                ik_store(y + j,            dl1 * v1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)]);
                ik_store(y + j+kBlockSize, dl2 * v2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)]);
            }
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
}

// IQ3_KS (half row prefix)
template <typename dst_t>
static __device__ void deq_row_iq3_ks(const uint8_t * row, dst_t * y, int ne) {
    const int kBlockSize = 32;
    const uint16_t * dptr = (const uint16_t *) row;
    const float d = ik_h2f(dptr);
    const block_iq3_ks * x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    const int nblock = ne / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int j = 0; j < 4; ++j) {
            const int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            const int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * v = iq3nl_values + (((x[ibl].extra >> (8 + (4*i128+ib))) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    ik_store(y + j, dl[4*i128 + ib] * v[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)]);
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

// IQ1_KT: integer trellis (float row prefix)
static __device__ __forceinline__ void iq1kt_set_values(uint32_t i, float * result, float scale) {
    const uint32_t ka = 0xCBAC1FEDu;
    uint32_t x = i + 4096;
    for (int g = 0; g < 8; ++g) {
        x *= ka;
        const uint32_t s = x & 0x3f3f3f3fu;
        const int8_t * i8 = (const int8_t *) &s;
        result[g] = scale * ((float)i8[0] + (float)i8[1] + (float)i8[2] + (float)i8[3] - 126.0f);
    }
}
template <typename dst_t>
static __device__ void deq_row_iq1_kt(const uint8_t * row, dst_t * y, int ne) {
    const int kNblock = 8, kNg = 4;
    const int nb = ne / QK_K;
    const float * dptr = (const float *) row;
    const float d = *dptr;
    const block_iq1_kt * x = (const block_iq1_kt *)(dptr + 1);
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < kNblock; ++ib) {
            const float sl = d * iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < kNg; ++ig) {
                uint16_t idx = x[ibl].ql[ib*kNg + ig]
                             | ((x[ibl].qh[(ib%(kNblock/2))*kNg + ig] << (8 - 4*(ib/(kNblock/2)))) & 0xf00);
                idx |= ((x[ibl].sh[ib] << (8 - ig)) & 0x1000);
                float g8[8];
                iq1kt_set_values(idx, g8, sl);
                for (int t = 0; t < 8; ++t) ik_store(y + t, g8[t]);
                y += 8;
            }
        }
    }
}

// row-aware kernel: one thread per row
template <typename dst_t>
static __global__ void ik_deq_rows_kernel(const uint8_t * __restrict__ x, dst_t * __restrict__ y,
                                          int nrows, int ne, int64_t row_stride, int type) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    const uint8_t * row = x + (int64_t)r * row_stride;
    dst_t * yr = y + (int64_t)r * ne;
    switch (type) {
        case GGML_TYPE_IQ4_K:  deq_row_iq4_k (row, yr, ne); break;
        case GGML_TYPE_IQ6_K:  deq_row_iq6_k (row, yr, ne); break;
        case GGML_TYPE_IQ4_KS: deq_row_iq4_ks(row, yr, ne); break;
        case GGML_TYPE_IQ5_KS: deq_row_iq5_ks(row, yr, ne); break;
        case GGML_TYPE_IQ3_KS: deq_row_iq3_ks(row, yr, ne); break;
        case GGML_TYPE_IQ1_KT: deq_row_iq1_kt(row, yr, ne); break;
    }
}

extern "C" void ggml_cuda_dequantize_ik_rows_f16(const void * vx, void * y, int64_t nrows, int64_t ne,
                                                 int64_t row_stride, int type, cudaStream_t stream) {
    const int nt = 128;
    const int nb = (int)((nrows + nt - 1) / nt);
    ik_deq_rows_kernel<half><<<nb, nt, 0, stream>>>((const uint8_t *) vx, (half *) y, (int) nrows, (int) ne, row_stride, type);
}
GGML_BACKEND_API_C void ggml_cuda_dequantize_ik_rows_f32(const void * vx, void * y, int64_t nrows, int64_t ne,
                                                 int64_t row_stride, int type, cudaStream_t stream) {
    const int nt = 128;
    const int nb = (int)((nrows + nt - 1) / nt);
    ik_deq_rows_kernel<float><<<nb, nt, 0, stream>>>((const uint8_t *) vx, (float *) y, (int) nrows, (int) ne, row_stride, type);
}

// Fused trellis MMVQ for IQ1_KT: decode-in-register dot product.
// The correct-but-slow path (ggml_cuda_mul_mat_id_ik_pool) dequantizes the whole expert
// matrix to F32 in global memory (~50 MB/expert) then does an M=1 cuBLAS GEMV that touches
// each weight once, so the F32 write+read is the entire cost. This kernel instead assigns one
// thread per output row j: it reconstructs row j's weights via the QTIP trellis in registers
// and accumulates the dot with the token activation x directly, never materializing F32.
// Bit-exact to deq_row_iq1_kt (same iq1kt_set_values path + index unpacking), ~QK_K/8 bits
// of weight read per element vs 4 bytes written+read.
static __global__ void ik_mmvq_iq1kt_kernel(const uint8_t * __restrict__ wbase, int64_t nb01,
                                            const float * __restrict__ x, float * __restrict__ d_out,
                                            int ne00, int ne01) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ne01) return;
    const float * dptr = (const float *)(wbase + (int64_t)j * nb01);
    const float d = *dptr;
    const block_iq1_kt * xb = (const block_iq1_kt *)(dptr + 1);
    const int nb = ne00 / QK_K;
    float sum = 0.0f;
    int pos = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < 8; ++ib) {
            const float sl = d * iq4k_values[xb[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < 4; ++ig) {
                uint16_t idx = xb[ibl].ql[ib*4 + ig]
                             | ((xb[ibl].qh[(ib % 4)*4 + ig] << (8 - 4*(ib/4))) & 0xf00);
                idx |= ((xb[ibl].sh[ib] << (8 - ig)) & 0x1000);
                float g8[8];
                iq1kt_set_values(idx, g8, sl);
                #pragma unroll
                for (int t = 0; t < 8; ++t) sum += g8[t] * x[pos + t];
                pos += 8;
            }
        }
    }
    d_out[j] = sum;
}

// One fused M=1 GEMV: d_out[0..ne01) = W_expert (ne01 x ne00, IQ1_KT) - x[0..ne00).
// wexpert = expert-base pointer (input_cpy), nb01 = quant row stride (bytes).
extern "C" void ggml_cuda_ik_mmvq_iq1kt(const void * wexpert, int64_t nb01,
                                        const void * xrow, void * drow,
                                        int64_t ne00, int64_t ne01, cudaStream_t stream) {
    const int nt  = 128;
    const int nbk = (int)((ne01 + nt - 1) / nt);
    ik_mmvq_iq1kt_kernel<<<nbk, nt, 0, stream>>>((const uint8_t *)wexpert, nb01,
                                                 (const float *)xrow, (float *)drow,
                                                 (int)ne00, (int)ne01);
}

extern "C" bool ggml_cuda_is_ik_type(int type) {
    switch (type) {
        case GGML_TYPE_IQ4_K: case GGML_TYPE_IQ6_K: case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS: case GGML_TYPE_IQ3_KS: case GGML_TYPE_IQ1_KT: return true;
        default: return false;
    }
}
