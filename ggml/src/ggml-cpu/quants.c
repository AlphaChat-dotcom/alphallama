#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"   // GGML_BACKEND_API declarations, so the Q2_0 prepared-dot entry points are exported on Windows
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "arch-fallback.h"

#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>   // for fabsf
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, y, k);
}

void quantize_row_q2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q2_0_ref(x, y, k);
}

// Q2_0 x Q8_0. One Q2_0 block (64) spans two Q8_0 blocks (32).
#if defined(__AVX2__)
#include <immintrin.h>
// AVX2: unpack the 64 2-bit codes with four shift+mask passes and two interleave levels (byte b carries
// weights 4b..4b+3), then sum(code*y) - sum(y) per Q8_0 block via maddubs (codes are unsigned 0..3).
// The scalar fallback below decodes one code per multiply, which is too slow for CPU-side expert compute.
void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK2_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    const int nb = n / QK2_0;
    const __m128i m3     = _mm_set1_epi8(3);
    const __m256i ones8  = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        const __m128i q  = _mm_loadu_si128((const __m128i *) x[i].qs);
        const __m128i c0 = _mm_and_si128(q, m3);
        const __m128i c1 = _mm_and_si128(_mm_srli_epi16(q, 2), m3);
        const __m128i c2 = _mm_and_si128(_mm_srli_epi16(q, 4), m3);
        const __m128i c3 = _mm_and_si128(_mm_srli_epi16(q, 6), m3);
        const __m128i a01l = _mm_unpacklo_epi8(c0, c1), a01h = _mm_unpackhi_epi8(c0, c1);
        const __m128i a23l = _mm_unpacklo_epi8(c2, c3), a23h = _mm_unpackhi_epi8(c2, c3);
        const __m256i cw[2] = {
            _mm256_set_m128i(_mm_unpackhi_epi16(a01l, a23l), _mm_unpacklo_epi16(a01l, a23l)),   // weights  0..31
            _mm256_set_m128i(_mm_unpackhi_epi16(a01h, a23h), _mm_unpacklo_epi16(a01h, a23h)),   // weights 32..63
        };
        const float dx = GGML_CPU_FP16_TO_FP32(x[i].d);
        for (int h = 0; h < 2; h++) {
            const block_q8_0 * yb = &y[i*2 + h];
            const __m256i yv = _mm256_loadu_si256((const __m256i *) yb->qs);
            const __m256i p  = _mm256_madd_epi16(_mm256_maddubs_epi16(cw[h], yv), ones16);
            const __m256i sy = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, yv), ones16);
            const __m256  f  = _mm256_cvtepi32_ps(_mm256_sub_epi32(p, sy));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dx * GGML_CPU_FP16_TO_FP32(yb->d)), f, acc);
        }
    }
    const __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    *s = _mm_cvtss_f32(r1);
}

// CPU expert miss path (ALPHA_DRE): Q2_0 rows against a prepared q8_0 activation. The activation is re-laid
// out once into the Q2_0 bit-plane order (plane j of block byte b = weight 4b+j), with its per-lane sums and
// scales hoisted, so a row costs only: load codes, 4 shift/mask planes, 4 maddubs against pre-permuted
// activations, one scaled FMA per pair of blocks -- no per-row code interleave, no per-row activation sums.
// About 2x the per-core throughput of ggml_vec_dot_q2_0_q8_0. Results agree with it to float rounding
// (different summation order), and are identical for a row whatever else is batched.
// Layout per pair of blocks: yp[32*4] int8 (plane j: bytes 0..15 block i, 16..31 block i+1), sy[8] int32,
// dyl[8] float.
size_t ggml_cpu_q2_0_prep_size(int k) {
    const int np = (k / QK2_0 + 1) / 2;
    return (size_t) np * (128 + 8 * sizeof(int32_t) + 8 * sizeof(float));
}
bool ggml_cpu_q2_0_prepared_available(void) { return true; }
void ggml_cpu_q2_0_prepare(const void * GGML_RESTRICT vy, int k, void * GGML_RESTRICT prep) {
    const block_q8_0 * GGML_RESTRICT y8 = vy;
    const int nb = k / QK2_0, np = (nb + 1) / 2;
    int8_t  * yp  = (int8_t *) prep;
    int32_t * sy  = (int32_t *) (yp + (size_t) np * 128);
    float   * dyl = (float *) (sy + (size_t) np * 8);
    memset(prep, 0, ggml_cpu_q2_0_prep_size(k));
    for (int i = 0; i < nb; i++) {
        const int p = i / 2, h = i % 2;
        for (int b = 0; b < 16; b++) {
            for (int j = 0; j < 4; j++) {
                const int w = 4 * b + j;
                const int8_t v = y8[2 * i + w / 32].qs[w % 32];
                yp[p * 128 + j * 32 + h * 16 + b] = v;
                sy[p * 8 + h * 4 + b / 4] += v;
            }
        }
        for (int l = 0; l < 4; l++) dyl[p * 8 + h * 4 + l] = GGML_CPU_FP16_TO_FP32(y8[2 * i + (l >= 2)].d);
    }
}
void ggml_cpu_q2_0_dot_prepared(int k, int nrows, const void * GGML_RESTRICT vx, size_t row_stride,
                                const void * GGML_RESTRICT prep, float * GGML_RESTRICT out) {
    const int nb = k / QK2_0, np = (nb + 1) / 2;
    const int8_t  * yp  = (const int8_t *) prep;
    const int32_t * sy  = (const int32_t *) (yp + (size_t) np * 128);
    const float   * dyl = (const float *) (sy + (size_t) np * 8);
    const __m256i m3 = _mm256_set1_epi8(3), ones16 = _mm256_set1_epi16(1);
    for (int r = 0; r < nrows; r++) {
        const block_q2_0 * GGML_RESTRICT x = (const block_q2_0 *) ((const char *) vx + (size_t) r * row_stride);
        __m256 acc = _mm256_setzero_ps();
        int i = 0, p = 0;
        for (; i + 1 < nb; i += 2, p++) {
            const __m256i q = _mm256_set_m128i(_mm_loadu_si128((const __m128i *) x[i + 1].qs), _mm_loadu_si128((const __m128i *) x[i].qs));
            const int8_t * Y = yp + p * 128;
            const __m256i c0 = _mm256_and_si256(q, m3);
            const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(q, 2), m3);
            const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m3);
            const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(q, 6), m3);
            // each int16 is at most 2*3*128 = 768 in magnitude, four of them sum to < 32767: no overflow
            const __m256i s01 = _mm256_add_epi16(_mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) (Y +  0))),
                                                 _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) (Y + 32))));
            const __m256i s23 = _mm256_add_epi16(_mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *) (Y + 64))),
                                                 _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *) (Y + 96))));
            __m256i s = _mm256_madd_epi16(_mm256_add_epi16(s01, s23), ones16);
            s = _mm256_sub_epi32(s, _mm256_loadu_si256((const __m256i *) (sy + p * 8)));
            const __m256 dx = _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x[i + 1].d)), _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d)));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(dx, _mm256_loadu_ps(dyl + p * 8)), acc);
        }
        if (i < nb) {   // odd tail: block i in the low half; the pad half has zero codes, sums and scale
            const __m256i q = _mm256_set_m128i(_mm_setzero_si128(), _mm_loadu_si128((const __m128i *) x[i].qs));
            const int8_t * Y = yp + p * 128;
            __m256i s16 = _mm256_setzero_si256();
            for (int j = 0; j < 4; j++) {
                const __m256i c = _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3);
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(c, _mm256_loadu_si256((const __m256i *) (Y + 32 * j))));
            }
            __m256i s = _mm256_madd_epi16(s16, ones16);
            s = _mm256_sub_epi32(s, _mm256_loadu_si256((const __m256i *) (sy + p * 8)));
            const __m256 dx = _mm256_set_m128(_mm_setzero_ps(), _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d)));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(dx, _mm256_loadu_ps(dyl + p * 8)), acc);
        }
        const __m128 r4 = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
        const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
        out[r] = _mm_cvtss_f32(_mm_add_ss(r2, _mm_movehdup_ps(r2)));
    }
}
#else
size_t ggml_cpu_q2_0_prep_size(int k) { (void) k; return 0; }
bool ggml_cpu_q2_0_prepared_available(void) { return false; }
void ggml_cpu_q2_0_prepare(const void * GGML_RESTRICT vy, int k, void * GGML_RESTRICT prep) { (void) vy; (void) k; (void) prep; }
void ggml_cpu_q2_0_dot_prepared(int k, int nrows, const void * GGML_RESTRICT vx, size_t row_stride,
                                const void * GGML_RESTRICT prep, float * GGML_RESTRICT out) {
    (void) k; (void) nrows; (void) vx; (void) row_stride; (void) prep; (void) out;
}
void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK2_0 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    const int nb = n / QK2_0;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_FP16_TO_FP32(x[i].d);
        float sumi = 0.0f;
        for (int h = 0; h < 2; h++) {
            const block_q8_0 * yb = &y[i*2 + h];
            int acc = 0;
            for (int j = 0; j < QK8_0; j++) {
                const int e = h * QK8_0 + j;
                const int code = (x[i].qs[e / 4] >> (2 * (e % 4))) & 3;
                acc += (code - 1) * yb->qs[j];
            }
            sumi += GGML_FP16_TO_FP32(yb->d) * (float) acc;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}
#endif

void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_0_ref(x, y, k);
}

void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_1_ref(x, y, k);
}

void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_mxfp4_ref(x, y, k);
}

void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, y, k);
}

//
// 2-6 bit quantization in super-blocks
//

//========================- 2-bit (de)-quantization

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q4_K * GGML_RESTRICT y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q5_K * GGML_RESTRICT y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq1_0 * GGML_RESTRICT y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

//===================================== Q8_K ==============================================

void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_FP16_TO_FP32(x[i].d);

        float sumi = 0.0f;

        for (int k = 0; k < 4; k++) {
            const float d1 = GGML_FP16_TO_FP32(y[i*4 + k].d);

            int sumi_block = 0;

            for (int j = 0; j < QK8_0; j++) {
                const int bit_index = k * QK8_0 + j;
                const int byte_index = bit_index / 8;
                const int bit_offset = bit_index % 8;

                const int xi = ((x[i].qs[byte_index] >> bit_offset) & 1) ? 1 : -1;
                sumi_block += xi * y[i*4 + k].qs[j];
            }

            sumi += d1 * sumi_block;
        }

        sumf += d0 * sumi;
    }

    *s = sumf;
}


// DSQ1 quantize wrapper (void* interface for CPU backend)
void quantize_row_dsq1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_dsq1_ref(x, (block_dsq1 *)y, k);
}

// DSQ1 x Q8_K vec_dot: 4-point grid {-d_hi, -d_lo, +d_lo, +d_hi}
void ggml_vec_dot_dsq1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int nb = n / QK_DSQ1;
    assert(n % QK_DSQ1 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_dsq1 * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float dl = GGML_FP16_TO_FP32(x[i].d_lo);
        const float dh = GGML_FP16_TO_FP32(x[i].d_hi);
        const float grid[4] = { -dh, -dl, dl, dh };
        const float d8 = y[i].d;
        float sumi = 0.0f;
        for (int j = 0; j < QK_DSQ1; j++) {
            const uint8_t idx = (x[i].qs[j/4] >> ((j%4)*2)) & 0x3;
            sumi += grid[idx] * y[i].qs[j];
        }
        sumf += d8 * sumi;
    }
    *s = sumf;
}

// DSQ3 quantize wrapper
void quantize_row_dsq3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_dsq3_ref(x, (block_dsq3 *)y, k);
}

// DSQ3 x Q8_K vec_dot: 8-point grid {-3dh, -2dh, -dh, -dl, +dl, +dh, +2dh, +3dh}
void ggml_vec_dot_dsq3_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int nb = n / QK_DSQ3;
    assert(n % QK_DSQ3 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_dsq3 * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float dl = GGML_FP16_TO_FP32(x[i].d_lo);
        const float dh = GGML_FP16_TO_FP32(x[i].d_hi);
        const float grid[8] = { -3*dh, -2*dh, -dh, -dl, dl, dh, 2*dh, 3*dh };
        const float d8 = y[i].d;
        float sumi = 0.0f;

        for (int j = 0; j < QK_DSQ3; j++) {
            int bit_pos = j * 3;
            int byte_pos = bit_pos / 8;
            int bit_off = bit_pos % 8;
            uint8_t idx = (x[i].qs[byte_pos] >> bit_off) & 0x7;
            if (bit_off > 5) {
                idx |= (x[i].qs[byte_pos + 1] << (8 - bit_off)) & 0x7;
            }
            sumi += grid[idx] * y[i].qs[j];
        }
        sumf += d8 * sumi;
    }
    *s = sumf;
}

// DSQ4 quantize wrapper
void quantize_row_dsq4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_dsq4_ref(x, (block_dsq4 *)y, k);
}

// DSQ4 x Q8_K vec_dot: 16-point dual-scale grid
void ggml_vec_dot_dsq4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int nb = n / QK_DSQ4;
    assert(n % QK_DSQ4 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_dsq4 * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float dl = GGML_FP16_TO_FP32(x[i].d_lo);
        const float dh = GGML_FP16_TO_FP32(x[i].d_hi);
        float grid[16];
        grid[ 0] = -4*dh; grid[ 1] = -3*dh; grid[ 2] = -2*dh; grid[ 3] = -dh;
        grid[ 4] = -2*dl; grid[ 5] = -dl;   grid[ 6] = -dl/2; grid[ 7] = 0;
        grid[ 8] = dl/2;  grid[ 9] = dl;    grid[10] = 2*dl;  grid[11] = dh;
        grid[12] = 2*dh;  grid[13] = 3*dh;  grid[14] = 4*dh;  grid[15] = 5*dh;

        const float d8 = y[i].d;
        float sumi = 0.0f;

        for (int j = 0; j < QK_DSQ4; j++) {
            uint8_t idx = (j % 2 == 0) ? (x[i].qs[j/2] & 0xF) : (x[i].qs[j/2] >> 4);
            sumi += grid[idx] * y[i].qs[j];
        }
        sumf += d8 * sumi;
    }
    *s = sumf;
}

void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

// TODO: add WASM SIMD
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_E8M0_TO_FP32_HALF(x[ib].e);

        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_mxfp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_mxfp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

// NVFP4: super-block of 64 elements = 4 sub-blocks of 16 = 2 q8_0 blocks
void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;

    float sumf = 0;

    for (int ib = 0; ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = GGML_CPU_UE4M3_TO_FP32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_mxfp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_mxfp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d))*sumi + GGML_CPU_FP16_TO_FP32(x[ib].m)*GGML_CPU_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs)*5 + l*sizeof(x->qh) + j];
            }
        }

        sumf += (float) sum * (GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
}

void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
}

// AlphaLlama: access to Q2_K multi-r values (defined in ggml-quants.c)
extern void ggml_q2k_get_r_values(float r[4]);

// Select r bin from per-sub-group scale and min
static inline int q2k_r_bin_cpu(float dl, float ml) {
    if (dl < 0.001f) return 0;
    float ratio = ml / dl;
    if (ratio < 0.15f) return 0;
    if (ratio < 0.35f) return 1;
    if (ratio < 0.60f) return 2;
    return 3;
}

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // Check if multi-r is enabled
    float r_vals[4];
    ggml_q2k_get_r_values(r_vals);
    const bool use_r = (r_vals[0] != 1.0f || r_vals[1] != 1.0f || r_vals[2] != 1.0f || r_vals[3] != 1.0f);

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        if (use_r) {
            // Non-uniform reconstruction: use float accumulation with r-based levels
            float fsum = 0;
            int is = 0;
            for (int k = 0; k < QK_K/128; ++k) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    uint8_t sca = sc[is++];
                    float dl = GGML_CPU_FP16_TO_FP32(x[i].d) * (sca & 0xF);
                    float ml = GGML_CPU_FP16_TO_FP32(x[i].dmin) * (sca >> 4);
                    float r = r_vals[q2k_r_bin_cpu(dl, ml)];
                    float lv[4] = {0.0f, r, 2.0f - r, 2.0f};
                    for (int l = 0; l < 16; ++l) {
                        fsum += y[i].d * dl * lv[(q2[l] >> shift) & 3] * q8[l]
                              - y[i].d * ml * q8[l];
                    }

                    sca = sc[is++];
                    dl = GGML_CPU_FP16_TO_FP32(x[i].d) * (sca & 0xF);
                    ml = GGML_CPU_FP16_TO_FP32(x[i].dmin) * (sca >> 4);
                    r = r_vals[q2k_r_bin_cpu(dl, ml)];
                    lv[1] = r; lv[2] = 2.0f - r;
                    for (int l = 16; l < 32; ++l) {
                        fsum += y[i].d * dl * lv[(q2[l] >> shift) & 3] * q8[l]
                              - y[i].d * ml * q8[l];
                    }
                    shift += 2;
                    q8 += 32;
                }
                q2 += 32;
            }
            sumf += fsum;
        } else {
            // Standard uniform path (original, fast integer accumulation)
            int summs = 0;
            for (int j = 0; j < 16; ++j) {
                summs += y[i].bsums[j] * (sc[j] >> 4);
            }

            int isum = 0;
            int is = 0;
            int d;
            for (int k = 0; k < QK_K/128; ++k) {
                int shift = 0;
                for (int j = 0; j < 4; ++j) {
                    d = sc[is++] & 0xF;
                    int isuml = 0;
                    for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                    isum += d * isuml;
                    d = sc[is++] & 0xF;
                    isuml = 0;
                    for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                    isum += d * isuml;
                    shift += 2;
                    q8 += 32;
                }
                q2 += 32;
            }
            sumf += dall * isum - dmin * summs;
        }
    }
    *s = sumf;
}

// AlphaLlama: access to Q3_K/Q4_K multi-r values (defined in ggml-quants.c)
extern void ggml_q3k_get_r_values(float r[4]);
extern void ggml_q4k_get_r_values(float r[4]);

// Q3_K r bin: based on absolute scale magnitude (no dmin in Q3_K)
static inline int q3k_r_bin_cpu(float scale_val) {
    float a = fabsf(scale_val);
    if (a < 8.0f) return 0;
    if (a < 16.0f) return 1;
    if (a < 24.0f) return 2;
    return 3;
}

// Q3_K: adjust integer quant value by r for inner levels
static inline float q3k_adjust_cpu(int q3_val, float r) {
    if (q3_val >= -1 && q3_val <= 1) return (float)q3_val * r;
    return (float)q3_val;
}

void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // Check if multi-r is enabled
    float r_vals[4];
    ggml_q3k_get_r_values(r_vals);
    const bool use_r = (r_vals[0] != 1.0f || r_vals[1] != 1.0f || r_vals[2] != 1.0f || r_vals[3] != 1.0f);

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].hmask;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        if (use_r) {
            // Non-uniform reconstruction: float path with r-adjusted levels
            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
            for (int j = 0; j < QK_K/16; ++j) {
                int sc = scales[j] - 32;
                float r = r_vals[q3k_r_bin_cpu((float)sc)];
                float fsum = 0;
                for (int l = 0; l < 8; ++l) fsum += q3k_adjust_cpu(a[l], r) * (float)q8[l];
                q8 += 8; a += 8;
                for (int l = 0; l < 8; ++l) fsum += q3k_adjust_cpu(a[l], r) * (float)q8[l];
                q8 += 8; a += 8;
                sumf += d * (float)sc * fsum;
            }
        } else {
            for (int j = 0; j < QK_K/16; ++j) {
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
                q8 += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
                q8 += 8; a += 8;
            }
            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

// Q4_K r bin: uses dmin/d ratio like Q2_K
static inline int q4k_r_bin_cpu(float dl, float ml) {
    if (dl < 0.001f) return 0;
    float ratio = ml / dl;
    if (ratio < 0.15f) return 0;
    if (ratio < 0.35f) return 1;
    if (ratio < 0.60f) return 2;
    return 3;
}

// Build Q4_K non-uniform levels (same as in ggml-quants.c)
static inline void build_q4k_levels_cpu(float r, float lv[16]) {
    for (int i = 0; i < 16; i++) {
        if (i <= 7) {
            lv[i] = (float)i * r;
        } else {
            float base = 7.0f * r;
            lv[i] = base + (float)(i - 7) * (15.0f - base) / 8.0f;
        }
    }
    if (lv[15] > 0) {
        float scale = 15.0f / lv[15];
        for (int i = 0; i < 16; i++) lv[i] *= scale;
    }
}

void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    // Check if multi-r is enabled
    float r_vals[4];
    ggml_q4k_get_r_values(r_vals);
    const bool use_r = (r_vals[0] != 1.0f || r_vals[1] != 1.0f || r_vals[2] != 1.0f || r_vals[3] != 1.0f);

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        if (use_r) {
            // Non-uniform reconstruction: float path with level-mapped quant values
            // Q4_K: value = d * scale_j * lv[q4] - dmin * min_j
            const float d_all = GGML_CPU_FP16_TO_FP32(x[i].d);
            const float dmin_all = GGML_CPU_FP16_TO_FP32(x[i].dmin);
            a = aux8;
            int is = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                float dl = d_all * scales[is];
                float ml = dmin_all * mins[is];
                float r = r_vals[q4k_r_bin_cpu(dl, ml)];
                float lv[16];
                build_q4k_levels_cpu(r, lv);
                is++;
                for (int l = 0; l < 32; ++l) {
                    sumf += y[i].d * (dl * lv[(uint8_t)a[l]] - ml) * (float)q8[l];
                }
                q8 += 32; a += 32;
            }
        } else {
            // Standard uniform path (original, fast integer accumulation)
            int sumi = 0;
            for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
            a = aux8;
            int is = 0;
            for (int j = 0; j < QK_K/32; ++j) {
                int32_t scale = scales[is++];
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
                for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
                for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
                q8 += 8; a += 8;
            }
            const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
            for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
            const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
            sumf -= dmin * sumi;
        }
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const uint8_t * GGML_RESTRICT hm = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_CPU_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
}

void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const uint8_t  * GGML_RESTRICT sc = x[i].scales;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;
}

void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
}

void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint8_t * GGML_RESTRICT signs = x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
}

void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;
}

void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_CPU_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;
}

void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_CPU_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
}

// ============================ 4-bit non-linear quants

void quantize_row_iq4_nl(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_iq4_xs(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}
