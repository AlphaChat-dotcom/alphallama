// ik_llama.cpp quant-type support.
//
// Dequantization for the ik_llama.cpp types used by ik-produced GGUF files:
//   IQ4_K (139), IQ6_K (141), IQ4_KS (144), IQ5_KS (152), IQ3_KS (156), IQ1_KT (158)
//
// Ported from ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp), MIT license,
// Copyright (c) Iwan Kawrakow - source: ggml/src/iqk/iqk_quantize.cpp; translated
// C++ -> C where needed (IQ1_KT trellis: QuantizerIQKT<32,8,13,false,true>).
//
// Row layout note: iq4_ks / iq5_ks / iq1_kt rows begin with a float row-scale and
// iq3_ks with a ggml_half (traits row_meta_size = 4 / 2); the block structs exclude
// the prefix. Quantization (from_float) is intentionally not ported: load + compute
// only.

#include "ggml-quants-ik.h"
#include "ggml-quants.h"   // dequantize_row_q8_K (vec_dot src1 tier)

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// IQ4_K
void dequantize_row_iq4_k(const block_iq4_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = GGML_FP16_TO_FP32(x[i].d);
        uint16_t extra = x[i].extra;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = extra & 1 ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = extra & 2 ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[qs[j] & 0xf];
                y[j+16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

// IQ6_K (cubic-polynomial value map)
#define A_IQ6K -127.f
#define B_IQ6K 6.2568f
#define C_IQ6K 0.11218f
#define D_IQ6K 0.0011972f
#define S_IQ6K 1.f

void dequantize_row_iq6_k(const block_iq6_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;
        uint16_t extra = x[i].extra;
        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * sl[4*ib64 + 0];
            float dl2 = d * sl[4*ib64 + 1];
            float dl3 = d * sl[4*ib64 + 2];
            float dl4 = d * sl[4*ib64 + 3];
            float m1 = extra & 1 ? S_IQ6K : 0;
            float m2 = extra & 2 ? S_IQ6K : 0;
            float m3 = extra & 4 ? S_IQ6K : 0;
            float m4 = extra & 8 ? S_IQ6K : 0;
            for (int j = 0; j < 16; ++j) {
                float q1 = (float)((qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4));
                float q2 = (float)((qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4));
                float q3 = (float)((qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2));
                float q4 = (float)((qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2));
                y[j+ 0] = dl1 * (A_IQ6K + q1*(B_IQ6K + q1*(-C_IQ6K + q1*D_IQ6K)) + m1);
                y[j+16] = dl2 * (A_IQ6K + q2*(B_IQ6K + q2*(-C_IQ6K + q2*D_IQ6K)) + m2);
                y[j+32] = dl3 * (A_IQ6K + q3*(B_IQ6K + q3*(-C_IQ6K + q3*D_IQ6K)) + m3);
                y[j+48] = dl4 * (A_IQ6K + q4*(B_IQ6K + q4*(-C_IQ6K + q4*D_IQ6K)) + m4);
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }
    }
}

// IQ4_KS (float row-scale prefix)
void dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k) {
    const int kBlockSize = 32;
    GGML_ASSERT(k % QK_K == 0);
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq4_ks *)(dptr + 1);
    const int nblock = k / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j             ] = dl * values[qs[j] & 0xf];
                y[j+kBlockSize/2] = dl * values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

// IQ5_KS (float row-scale prefix)
void dequantize_row_iq5_ks(const block_iq5_ks * x, float * y, int64_t k) {
    const int kBlockSize = 32;
    GGML_ASSERT(k % QK_K == 0);
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq5_ks *)(dptr + 1);
    const int nblock = k / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            const float dl1 = d * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            const float dl2 = d * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            for (int j = 0; j < kBlockSize; ++j) {
                y[j           ] = dl1 * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                y[j+kBlockSize] = dl2 * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
}

// IQ3_KS (ggml_half row-scale prefix)
void dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k) {
    const int kBlockSize = 32;
    GGML_ASSERT(k % QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    const int nblock = k / QK_K;
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
                const int8_t * values = iq3nl_values + (((x[ibl].extra >> (8 + (4*i128+ib))) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

// IQ1_KT: integer trellis (QuantizerIQKT<32, 8, 13, is_abs=false, is_int=true>)
// Each group of 8 values is generated from a 13-bit index (+4096 offset) via the
// multiplicative generator x = ka*x; value = sum of 4 masked int8 lanes - 126.
// Row layout: float row-scale, then blocks of {sh[8], ql[32], qh[16]} per 256.
static inline void iq1kt_set_values(uint32_t i, float * result, float scale) {
    const uint32_t ka = 0xCBAC1FEDu;
    uint32_t x = i + 4096;
    for (int g = 0; g < 8; ++g) {
        x *= ka;
        const uint32_t s = x & 0x3f3f3f3fu;
        const int8_t * i8 = (const int8_t *)&s;
        result[g] = scale * ((float)i8[0] + (float)i8[1] + (float)i8[2] + (float)i8[3] - 126.0f);
    }
}

void dequantize_row_iq1_kt(const block_iq1_kt * x, float * y, int64_t k) {
    GGML_ASSERT(k % QK_K == 0);
    // kSuperBlockSize=256, kBlockSize=32, kGroupSize=8 -> kNblock=8, kNg=4; kScale=1
    const int kNblock = 8, kNg = 4;
    const int nb = (int)(k / QK_K);
    const float * dptr = (const float *)x;
    const float d = *dptr;   // kScale = 1 for the int trellis
    x = (const block_iq1_kt *)(dptr + 1);
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < kNblock; ++ib) {
            const float sl = d * iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < kNg; ++ig) {
                uint16_t idx = x[ibl].ql[ib*kNg + ig]
                             | ((x[ibl].qh[(ib%(kNblock/2))*kNg + ig] << (8 - 4*(ib/(kNblock/2)))) & 0xf00);
                idx |= ((x[ibl].sh[ib] << (8 - ig)) & 0x1000);
                iq1kt_set_values(idx, y, sl);
                y += 8;
            }
        }
    }
}

// CPU vec_dot (correctness-first: dequantize-then-dot).
// ggml CPU mul_mat / mul_mat_id quantize src1 (activations) to the vec_dot_type
// (Q8_K) then call vec_dot(n, s, x_row, y_row). There is no SIMD kernel for the ik
// types yet, so both the ik weight row (x, via its to_float, which consumes the
// per-row scale prefix) and the Q8_K activation row (y) are dequantized to float and dotted.
// Slow but exact. n is a multiple of QK_K (== a whole matmul row). nrc==1 (single row).
#define IK_VD_MAX 40960   // >= max n we see (n_embd 6144, expert ff 2048, output cols)
// `__thread` is a GCC/Clang extension; MSVC spells thread-local storage __declspec(thread).
// MSVC would otherwise parse __thread as an identifier of type int.
#if defined(_MSC_VER)
#  define IK_THREAD_LOCAL __declspec(thread)
#else
#  define IK_THREAD_LOCAL __thread
#endif
static void ik_vec_dot_deq(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx,
                           const void * GGML_RESTRICT vy,
                           void (*deqx)(const void * GGML_RESTRICT, float * GGML_RESTRICT, int64_t)) {
    static IK_THREAD_LOCAL float xb[IK_VD_MAX];
    static IK_THREAD_LOCAL float yb[IK_VD_MAX];
    GGML_ASSERT(n <= IK_VD_MAX);
    deqx(vx, xb, n);
    dequantize_row_q8_K((const block_q8_K *)vy, yb, n);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double)xb[i] * (double)yb[i];
    *s = (float)acc;
}

#define IK_VEC_DOT(NAME, DEQ, BLK)                                                         \
void NAME(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,        \
          size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {                  \
    (void) bs; (void) bx; (void) by; (void) nrc;                                           \
    ik_vec_dot_deq(n, s, vx, vy, (void (*)(const void * GGML_RESTRICT, float * GGML_RESTRICT, int64_t)) DEQ); \
}
IK_VEC_DOT(ggml_vec_dot_iq4_k_q8_K,  dequantize_row_iq4_k,  block_iq4_k)
IK_VEC_DOT(ggml_vec_dot_iq6_k_q8_K,  dequantize_row_iq6_k,  block_iq6_k)
IK_VEC_DOT(ggml_vec_dot_iq4_ks_q8_K, dequantize_row_iq4_ks, block_iq4_ks)
IK_VEC_DOT(ggml_vec_dot_iq5_ks_q8_K, dequantize_row_iq5_ks, block_iq5_ks)
IK_VEC_DOT(ggml_vec_dot_iq3_ks_q8_K, dequantize_row_iq3_ks, block_iq3_ks)
IK_VEC_DOT(ggml_vec_dot_iq1_kt_q8_K, dequantize_row_iq1_kt, block_iq1_kt)
