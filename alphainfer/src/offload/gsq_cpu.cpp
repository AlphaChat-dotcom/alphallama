// Decode-once multi-token CPU kernels for the GSQ-RCO expert types. See gsq_cpu.h.
// Tables and block layouts come from ggml-common.h (the same ones ggml-cpu uses).
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "offload/gsq_cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ai::offload {

#if defined(__AVX2__)
// ggml-cpu's keven_signs_q2xs (arch/x86/quants.c, not exported): entry i = 8 bytes of +1 / -1 from ksigns_iq2xs[i]
struct EvenSigns {
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t e = 0;
            for (int j = 0; j < 8; ++j) e |= (uint64_t) (((ksigns_iq2xs[i] >> j) & 1) ? 0xFFu : 0x01u) << (8 * j);
            v[i] = e;
        }
    }
};
static const EvenSigns k_even_signs;

// fp16 -> fp32 inline (F16C), as ggml-cpu's GGML_CPU_FP16_TO_FP32; ggml_fp16_to_fp32 is an out-of-line call
static inline float gsq_h2f(ggml_half h) { return _cvtsh_ss(h); }

static inline float hsum8(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// IQ2_XXS (ggml_vec_dot_iq2_xxs_q8_K's AVX2 arithmetic): per pair of 32-blocks, 8 grid rows + 8 sign masks + 2 scales
// are decoded once, then each token's q8 is sign-applied and multiplied.
template <int T, bool G>
static void dot_iq2_xxs(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const block_q8_K * const * y,
                        float * out, int64_t os) {
    const int        nb      = K / QK_K;
    const uint64_t * signs64 = k_even_signs.v;
    for (int64_t r = r0; r < r1; ++r) {
        const block_iq2_xxs * x = (const block_iq2_xxs *) (W + (size_t) r * row_bytes);
        __m256 acc[T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint16_t * q2 = x[i].qs;
            __m256i sumi[T];
            for (int t = 0; t < T; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                uint32_t aux32[4];
                memcpy(aux32, q2, 4 * sizeof(uint32_t));
                q2 += 8;
                const uint8_t * aux8 = (const uint8_t *) aux32;
                __m256i q2_1, q2_2;
                if constexpr (G) {
                    q2_1 = _mm256_i32gather_epi64((const long long *) iq2xxs_grid, _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) aux32[0])), 8);
                    q2_2 = _mm256_i32gather_epi64((const long long *) iq2xxs_grid, _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) aux32[2])), 8);
                } else {
                    q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
                    q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
                }
                const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                       signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
                const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                       signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
                const __m256i sc1 = _mm256_set1_epi16((int16_t) (2 * (aux32[1] >> 28) + 1));
                const __m256i sc2 = _mm256_set1_epi16((int16_t) (2 * (aux32[3] >> 28) + 1));
                for (int t = 0; t < T; ++t) {
                    const int8_t * q8 = y[t][i].qs + ib32 * 32;
                    const __m256i  a1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), s2_1);
                    const __m256i  a2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32)), s2_2);
                    const __m256i  p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1);
                    const __m256i  p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p1, p2));
                }
            }
            const float dx = gsq_h2f(x[i].d);
            for (int t = 0; t < T; ++t) {
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        for (int t = 0; t < T; ++t) out[t * os + r] = 0.125f * hsum8(acc[t]);
    }
}
#define GSQ_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline __m128i scale_shuffle(int i) {   // ggml-cpu x86 get_scale_shuffle
    static const uint8_t k[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,  2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,  6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13, 14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15 };
    return _mm_loadu_si128((const __m128i *) k + i);
}
// ggml-cpu x86 get_scale_shuffle_k4: row r = 32 bytes of (2r, 2r+1) pairs
struct ScaleK4 {
    alignas(32) uint8_t k[256];
    ScaleK4() { for (int r = 0; r < 8; ++r) for (int j = 0; j < 32; ++j) k[r * 32 + j] = (uint8_t) (2 * r + (j & 1)); }
};
static const ScaleK4 k_scale_k4;
static inline __m256i scale_shuffle_k4(int i) { return _mm256_load_si256((const __m256i *) k_scale_k4.k + i); }

// IQ2_XS (ggml_vec_dot_iq2_xs_q8_K AVX2): per 4 x 32 block, 16 grid rows, the sign masks (parity-completed 7-bit
// signs) and 4 scale vectors are decoded once.
template <int T, bool G>
static void dot_iq2_xs(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const block_q8_K * const * y,
                       float * out, int64_t os) {
    const int nb = K / QK_K;
    const __m256i mone = _mm256_set1_epi8(1);
    static const char bss1[32] = { 0,0,0,0,0,0,0,0, 2,2,2,2,2,2,2,2, 4,4,4,4,4,4,4,4, 6,6,6,6,6,6,6,6 };
    static const char bss2[32] = { 8,8,8,8,8,8,8,8, 10,10,10,10,10,10,10,10, 12,12,12,12,12,12,12,12, 14,14,14,14,14,14,14,14 };
    static const uint8_t bsel[32] = { 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
    static const uint8_t khelp[32] = { 0x00,0x80,0x80,0x00,0x80,0x00,0x00,0x80,0x80,0x00,0x00,0x80,0x00,0x80,0x80,0x00,
                                       0x00,0x80,0x80,0x00,0x80,0x00,0x00,0x80,0x80,0x00,0x00,0x80,0x00,0x80,0x80,0x00 };
    const __m256i bit_selector = _mm256_loadu_si256((const __m256i *) bsel);
    const __m256i shuf1 = _mm256_loadu_si256((const __m256i *) bss1);
    const __m256i shuf2 = _mm256_loadu_si256((const __m256i *) bss2);
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i *) khelp);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    for (int64_t r = r0; r < r1; ++r) {
        const block_iq2_xs * x = (const block_iq2_xs *) (W + (size_t) r * row_bytes);
        __m256 acc[T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint16_t * q2 = x[i].qs;
            uint64_t aux64;
            memcpy(&aux64, x[i].scales, 8);
            __m128i stmp = _mm_set1_epi64x((long long) aux64);
            stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
            const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
            __m256i sumi[T];
            for (int t = 0; t < T; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
                const __m256i q2_data = _mm256_loadu_si256((const __m256i *) q2);
                q2 += 16;
                alignas(32) uint16_t gindex[16];
                const __m256i gidx = _mm256_and_si256(q2_data, m511);
                if constexpr (!G) _mm256_store_si256((__m256i *) gindex, gidx);
                const __m256i psb  = _mm256_srli_epi16(q2_data, 9);
                const __m256i psbu = _mm256_srli_epi16(q2_data, 13);
                const __m256i full = _mm256_or_si256(psb, _mm256_shuffle_epi8(bit_helper, _mm256_xor_si256(psb, psbu)));
                __m256i q2v[4];
                if constexpr (G) {
                    const __m256i i32l = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(gidx)), i32h = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(gidx, 1));
                    const long long * g64 = (const long long *) iq2xs_grid;
                    q2v[0] = _mm256_i32gather_epi64(g64, _mm256_castsi256_si128(i32l), 8);
                    q2v[1] = _mm256_i32gather_epi64(g64, _mm256_extracti128_si256(i32l, 1), 8);
                    q2v[2] = _mm256_i32gather_epi64(g64, _mm256_castsi256_si128(i32h), 8);
                    q2v[3] = _mm256_i32gather_epi64(g64, _mm256_extracti128_si256(i32h, 1), 8);
                } else {
                    q2v[0] = _mm256_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]], iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
                    q2v[1] = _mm256_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]], iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
                    q2v[2] = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]], iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
                    q2v[3] = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]], iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
                }
                const __m128i fl = _mm256_castsi256_si128(full), fh = _mm256_extractf128_si256(full, 1);
                const __m256i f1 = GSQ_SET_M128I(fl, fl), f2 = GSQ_SET_M128I(fh, fh);
                auto smask = [&](__m256i f, __m256i sh) {
                    const __m256i s = _mm256_shuffle_epi8(f, sh);
                    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s, bit_selector), bit_selector), mone);
                };
                const __m256i sg[4] = { smask(f1, shuf1), smask(f1, shuf2), smask(f2, shuf1), smask(f2, shuf2) };
                const __m256i sc[4] = { _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle(ib32 + 0))),
                                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle(ib32 + 1))),
                                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle(ib32 + 2))),
                                        _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, scale_shuffle(ib32 + 3))) };
                for (int t = 0; t < T; ++t) {
                    const int8_t * q8 = y[t][i].qs + ib32 * 32;
                    __m256i s = sumi[t];
                    for (int j = 0; j < 4; ++j) {
                        const __m256i a = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32 * j)), sg[j]);
                        s = _mm256_add_epi32(s, _mm256_madd_epi16(_mm256_maddubs_epi16(q2v[j], a), sc[j]));
                    }
                    sumi[t] = s;
                }
            }
            const float dx = gsq_h2f(x[i].d);
            for (int t = 0; t < T; ++t) {
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        for (int t = 0; t < T; ++t) out[t * os + r] = 0.125f * hsum8(acc[t]);
    }
}

// IQ2_S (ggml_vec_dot_iq2_s_q8_K AVX2): 10-bit grid indices (qs + qh), explicit sign bytes, nibble scales.
template <int T, bool G>
static void dot_iq2_s(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const block_q8_K * const * y,
                      float * out, int64_t os) {
    const int nb = K / QK_K;
    static const uint8_t km1[32] = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    static const uint8_t km2[32] = { 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) km1), mask2 = _mm256_loadu_si256((const __m256i *) km2);
    const __m128i m4 = _mm_set1_epi8(0xf), m1 = _mm_set1_epi8(1);
    for (int64_t r = r0; r < r1; ++r) {
        const block_iq2_s * x = (const block_iq2_s *) (W + (size_t) r * row_bytes);
        __m256 acc[T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t *  qs    = x[i].qs;
            const uint8_t *  qh    = x[i].qh;
            const uint16_t * signs = (const uint16_t *) (x[i].qs + QK_K / 8);
            uint64_t aux64;
            memcpy(&aux64, x[i].scales, 8);
            const __m128i scales8  = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x((long long) (aux64 >> 4), (long long) aux64), m4), 1), m1);
            const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
            __m256i sumi[T];
            for (int t = 0; t < T; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                __m256i q2_1, q2_2;
                if constexpr (G) {
                    // entry j (0..7) = qs[j] | ((qh[ib32 + j / 4] >> 2 (j % 4)) & 3) << 8: one 8 x 32-bit index vector, two 4-wide gathers
                    const __m256i hb = _mm256_set_epi32(qh[ib32 + 1], qh[ib32 + 1], qh[ib32 + 1], qh[ib32 + 1], qh[ib32 + 0], qh[ib32 + 0], qh[ib32 + 0], qh[ib32 + 0]);
                    const __m256i hi = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(hb, _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0)), _mm256_set1_epi32(3)), 8);
                    const __m256i ix = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) qs)), hi);
                    q2_1 = _mm256_i32gather_epi64((const long long *) iq2s_grid, _mm256_castsi256_si128(ix), 8);
                    q2_2 = _mm256_i32gather_epi64((const long long *) iq2s_grid, _mm256_extracti128_si256(ix, 1), 8);
                } else {
                    q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32 + 0] << 2) & 0x300)], iq2s_grid[qs[2] | ((qh[ib32 + 0] << 4) & 0x300)],
                                             iq2s_grid[qs[1] | ((qh[ib32 + 0] << 6) & 0x300)], iq2s_grid[qs[0] | ((qh[ib32 + 0] << 8) & 0x300)]);
                    q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32 + 1] << 2) & 0x300)], iq2s_grid[qs[6] | ((qh[ib32 + 1] << 4) & 0x300)],
                                             iq2s_grid[qs[5] | ((qh[ib32 + 1] << 6) & 0x300)], iq2s_grid[qs[4] | ((qh[ib32 + 1] << 8) & 0x300)]);
                }
                qs += 8;
                __m256i a = _mm256_set1_epi32((int) (signs[0] | ((uint32_t) signs[1] << 16)));
                const __m256i s2_1 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(a, mask1), mask2), mask2);
                a = _mm256_set1_epi32((int) (signs[2] | ((uint32_t) signs[3] << 16)));
                const __m256i s2_2 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(a, mask1), mask2), mask2);
                signs += 4;
                const __m256i sc1 = _mm256_shuffle_epi8(scales16, scale_shuffle_k4(ib32 + 0));
                const __m256i sc2 = _mm256_shuffle_epi8(scales16, scale_shuffle_k4(ib32 + 1));
                for (int t = 0; t < T; ++t) {
                    const int8_t * q8 = y[t][i].qs + ib32 * 32;
                    const __m256i  y1 = _mm256_loadu_si256((const __m256i *) q8), y2 = _mm256_loadu_si256((const __m256i *) (q8 + 32));
                    const __m256i  a1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, y1), s2_1);
                    const __m256i  a2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, y2), s2_2);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1),
                                                                         _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2)));
                }
            }
            const float dx = gsq_h2f(x[i].d);
            for (int t = 0; t < T; ++t) {
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        for (int t = 0; t < T; ++t) out[t * os + r] = 0.125f * hsum8(acc[t]);
    }
}

// IQ3_XXS (ggml_vec_dot_iq3_xxs_q8_K AVX2)
template <int T, bool G>
static void dot_iq3_xxs(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const block_q8_K * const * y,
                        float * out, int64_t os) {
    const int        nb      = K / QK_K;
    const uint64_t * signs64 = k_even_signs.v;
    for (int64_t r = r0; r < r1; ++r) {
        const block_iq3_xxs * x = (const block_iq3_xxs *) (W + (size_t) r * row_bytes);
        __m256 acc[T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t * q3  = x[i].qs;
            const uint8_t * gas = x[i].qs + QK_K / 4;
            __m256i sumi[T];
            for (int t = 0; t < T; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                __m256i q2_1, q2_2;
                if constexpr (G) {
                    q2_1 = _mm256_i32gather_epi32((const int *) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) q3)), 4);
                    q2_2 = _mm256_i32gather_epi32((const int *) iq3xxs_grid, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) (q3 + 8))), 4);
                } else {
                    q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                            iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
                    q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[15]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[12]],
                                            iq3xxs_grid[q3[11]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[9]], iq3xxs_grid[q3[8]]);
                }
                q3 += 16;
                uint32_t aux32[2];
                memcpy(aux32, gas, 8);
                gas += 8;
                const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                       signs64[(aux32[0] >> 7) & 127], signs64[(aux32[0] >> 0) & 127]);
                const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                       signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
                const __m256i sc1 = _mm256_set1_epi16((int16_t) (2 * (aux32[0] >> 28) + 1));
                const __m256i sc2 = _mm256_set1_epi16((int16_t) (2 * (aux32[1] >> 28) + 1));
                for (int t = 0; t < T; ++t) {
                    const int8_t * q8 = y[t][i].qs + ib32 * 32;
                    const __m256i  a1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) q8), s2_1);
                    const __m256i  a2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) (q8 + 32)), s2_2);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1),
                                                                         _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2)));
                }
            }
            const float dx = gsq_h2f(x[i].d);
            for (int t = 0; t < T; ++t) {
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        for (int t = 0; t < T; ++t) out[t * os + r] = 0.25f * hsum8(acc[t]);
    }
}

// IQ3_S (ggml_vec_dot_iq3_s_q8_K AVX2): 9-bit grid indices built with sllv, explicit sign bytes, nibble scales.
template <int T, bool G>
static void dot_iq3_s(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const block_q8_K * const * y,
                      float * out, int64_t os) {
    const int nb = K / QK_K;
    static const uint8_t km1[32] = { 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3 };
    static const uint8_t km2[32] = { 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) km1), mask2 = _mm256_loadu_si256((const __m256i *) km2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);
    for (int64_t r = r0; r < r1; ++r) {
        const block_iq3_s * x = (const block_iq3_s *) (W + (size_t) r * row_bytes);
        __m256 acc[T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t *  qs    = x[i].qs;
            const uint8_t *  qh    = x[i].qh;
            const uint16_t * signs = (const uint16_t *) x[i].signs;
            __m256i sumi[T];
            for (int t = 0; t < T; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) qs));
                qs += 16;
                alignas(32) uint32_t idx[16];
                __m256i v0 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 0]), idx_shift), idx_mask);
                __m256i v1 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32 + 1]), idx_shift), idx_mask);
                v0 = _mm256_or_si256(v0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
                v1 = _mm256_or_si256(v1, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
                __m256i q2_1, q2_2;
                if constexpr (G) {
                    q2_1 = _mm256_i32gather_epi32((const int *) iq3s_grid, v0, 4);
                    q2_2 = _mm256_i32gather_epi32((const int *) iq3s_grid, v1, 4);
                } else {
                    _mm256_store_si256((__m256i *) idx, v0);
                    _mm256_store_si256((__m256i *) (idx + 8), v1);
                    q2_1 = _mm256_set_epi32(iq3s_grid[idx[7]], iq3s_grid[idx[6]], iq3s_grid[idx[5]], iq3s_grid[idx[4]],
                                            iq3s_grid[idx[3]], iq3s_grid[idx[2]], iq3s_grid[idx[1]], iq3s_grid[idx[0]]);
                    q2_2 = _mm256_set_epi32(iq3s_grid[idx[15]], iq3s_grid[idx[14]], iq3s_grid[idx[13]], iq3s_grid[idx[12]],
                                            iq3s_grid[idx[11]], iq3s_grid[idx[10]], iq3s_grid[idx[9]], iq3s_grid[idx[8]]);
                }
                __m256i a = _mm256_set1_epi32((int) (signs[0] | ((uint32_t) signs[1] << 16)));
                const __m256i s2_1 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(a, mask1), mask2), mask2);
                a = _mm256_set1_epi32((int) (signs[2] | ((uint32_t) signs[3] << 16)));
                const __m256i s2_2 = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(a, mask1), mask2), mask2);
                signs += 4;
                const __m256i sc1 = _mm256_set1_epi16((int16_t) (2 * (x[i].scales[ib32 / 2] & 0xf) + 1));
                const __m256i sc2 = _mm256_set1_epi16((int16_t) (2 * (x[i].scales[ib32 / 2] >> 4) + 1));
                for (int t = 0; t < T; ++t) {
                    const int8_t * q8 = y[t][i].qs + ib32 * 32;
                    const __m256i  y1 = _mm256_loadu_si256((const __m256i *) q8), y2 = _mm256_loadu_si256((const __m256i *) (q8 + 32));
                    const __m256i  a1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, y1), s2_1);
                    const __m256i  a2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, y2), s2_2);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, a1), sc1),
                                                                         _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, a2), sc2)));
                }
            }
            const float dx = gsq_h2f(x[i].d);
            for (int t = 0; t < T; ++t) {
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
            }
        }
        for (int t = 0; t < T; ++t) out[t * os + r] = hsum8(acc[t]);
    }
}
// IQ4_NL x q8_0 (ggml_vec_dot_iq4_nl_q8_0 AVX2): the 4-bit lookup is decoded once; mul_add_epi8 is split into the
// per-row |w| (once) and the per-token sign(y, w).
template <int T>
static void dot_iq4_nl(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const block_q8_0 * const * y,
                       float * out, int64_t os) {
    const int     nb        = K / QK4_NL;
    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b       = _mm_set1_epi8(0x0f);
    const __m256i mone      = _mm256_set1_epi16(1);
    for (int64_t r = r0; r < r1; ++r) {
        const block_iq4_nl * x = (const block_iq4_nl *) (W + (size_t) r * row_bytes);
        __m256 acc1[T], acc2[T];
        for (int t = 0; t < T; ++t) { acc1[t] = _mm256_setzero_ps(); acc2[t] = _mm256_setzero_ps(); }
        int ib = 0;
        for (; ib + 1 < nb; ib += 2) {
            const __m128i b1 = _mm_loadu_si128((const __m128i *) x[ib + 0].qs);
            const __m128i b2 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs);
            const __m256i w1 = GSQ_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b1, 4), m4b)),
                                             _mm_shuffle_epi8(values128, _mm_and_si128(b1, m4b)));
            const __m256i w2 = GSQ_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(b2, 4), m4b)),
                                             _mm_shuffle_epi8(values128, _mm_and_si128(b2, m4b)));
            const __m256i a1 = _mm256_sign_epi8(w1, w1), a2 = _mm256_sign_epi8(w2, w2);
            const float   d1 = gsq_h2f(x[ib + 0].d), d2 = gsq_h2f(x[ib + 1].d);
            for (int t = 0; t < T; ++t) {
                const __m256i y1 = _mm256_loadu_si256((const __m256i *) y[t][ib + 0].qs);
                const __m256i y2 = _mm256_loadu_si256((const __m256i *) y[t][ib + 1].qs);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(a1, _mm256_sign_epi8(y1, w1)), mone);
                const __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(a2, _mm256_sign_epi8(y2, w2)), mone);
                acc1[t] = _mm256_fmadd_ps(_mm256_set1_ps(gsq_h2f(y[t][ib + 0].d) * d1), _mm256_cvtepi32_ps(p1), acc1[t]);
                acc2[t] = _mm256_fmadd_ps(_mm256_set1_ps(gsq_h2f(y[t][ib + 1].d) * d2), _mm256_cvtepi32_ps(p2), acc2[t]);
            }
        }
        for (int t = 0; t < T; ++t) {
            float s = hsum8(_mm256_add_ps(acc1[t], acc2[t]));
            for (int j = ib; j < nb; ++j) {   // odd tail, scalar as ggml
                int sumi = 0;
                for (int k = 0; k < QK4_NL / 2; ++k) {
                    sumi += y[t][j].qs[k] * kvalues_iq4nl[x[j].qs[k] & 0xf] + y[t][j].qs[k + QK4_NL / 2] * kvalues_iq4nl[x[j].qs[k] >> 4];
                }
                s += gsq_h2f(y[t][j].d) * gsq_h2f(x[j].d) * sumi;
            }
            out[t * os + r] = s;
        }
    }
}

// Q2_0 x prepared q8_0 (ggml-cpu's ggml_cpu_q2_0_dot_prepared, which halves the cost of the plain vec_dot): y[t] is a
// buffer filled by ggml_cpu_q2_0_prepare. The four code planes of a block pair are extracted once per row and reused.
template <int T>
static void dot_q2_0_prep(int K, const char * W, size_t row_bytes, int64_t r0, int64_t r1, const void * const * prep,
                          float * out, int64_t os) {
    const int nb = K / QK2_0, np = (nb + 1) / 2;
    const int8_t * yp[T]; const int32_t * sy[T]; const float * dyl[T];
    for (int t = 0; t < T; ++t) {
        yp[t]  = (const int8_t *) prep[t];
        sy[t]  = (const int32_t *) (yp[t] + (size_t) np * 128);
        dyl[t] = (const float *) (sy[t] + (size_t) np * 8);
    }
    const __m256i m3 = _mm256_set1_epi8(3), ones16 = _mm256_set1_epi16(1);
    for (int64_t r = r0; r < r1; ++r) {
        const block_q2_0 * x = (const block_q2_0 *) (W + (size_t) r * row_bytes);
        __m256 acc[T];
        for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
        int i = 0, p = 0;
        for (; i < nb; i += 2, p++) {
            const bool    pair = i + 1 < nb;
            const __m256i q = _mm256_set_m128i(pair ? _mm_loadu_si128((const __m128i *) x[i + 1].qs) : _mm_setzero_si128(),
                                               _mm_loadu_si128((const __m128i *) x[i].qs));
            const __m256i c0 = _mm256_and_si256(q, m3);
            const __m256i c1 = _mm256_and_si256(_mm256_srli_epi16(q, 2), m3);
            const __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m3);
            const __m256i c3 = _mm256_and_si256(_mm256_srli_epi16(q, 6), m3);
            const __m256  dx = _mm256_set_m128(_mm_set1_ps(pair ? gsq_h2f(x[i + 1].d) : 0.0f), _mm_set1_ps(gsq_h2f(x[i].d)));
            for (int t = 0; t < T; ++t) {
                const int8_t * Y = yp[t] + p * 128;
                const __m256i s01 = _mm256_add_epi16(_mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *) (Y + 0))),
                                                     _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *) (Y + 32))));
                const __m256i s23 = _mm256_add_epi16(_mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *) (Y + 64))),
                                                     _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *) (Y + 96))));
                __m256i s = _mm256_madd_epi16(_mm256_add_epi16(s01, s23), ones16);
                s = _mm256_sub_epi32(s, _mm256_loadu_si256((const __m256i *) (sy[t] + p * 8)));
                acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(dx, _mm256_loadu_ps(dyl[t] + p * 8)), acc[t]);
            }
        }
        for (int t = 0; t < T; ++t) out[t * os + r] = hsum8(acc[t]);
    }
}
#endif

#define GSQ_DISPATCH_G(fn, G)                                                                 \
    switch (T) {                                                                                \
        case 1: fn<1, G>(K, w, row_bytes, r0, r1, yy, out, os); return;                        \
        case 2: fn<2, G>(K, w, row_bytes, r0, r1, yy, out, os); return;                        \
        case 3: fn<3, G>(K, w, row_bytes, r0, r1, yy, out, os); return;                        \
        case 4: fn<4, G>(K, w, row_bytes, r0, r1, yy, out, os); return;                        \
        default: break;                                                                         \
    }
#define GSQ_DISPATCH(fn) \
    if (gather) { GSQ_DISPATCH_G(fn, true) } else { GSQ_DISPATCH_G(fn, false) }

#if defined(__AVX2__)
// Grid decode by gathers vs scalar loads + inserts: the same integer values, so the choice never changes a result.
// Gathers are faster on Intel hybrid P cores for most types, but much slower on E cores (Gracemont).
// AI_GSQ_GATHER: unset = auto (gather on Intel hybrid P cores -- /sys/devices/cpu_core/cpus -- for the types it wins
// on), 0 = never, 1 = every core and type (test mode). A thread's core is read at its first call: callers must be
// pinned (the miss pool is).
static int gsq_gather_mode() {
    static const int m = getenv("AI_GSQ_GATHER") ? atoi(getenv("AI_GSQ_GATHER")) : 2;
    return m;
}
static bool cpu_is_hybrid_big(int cpu) {
    static const std::vector<bool> big = [] {
        std::vector<bool> v;
        if (FILE * f = fopen("/sys/devices/cpu_core/cpus", "r")) {   // Intel hybrid only; absent on Zen / non-hybrid
            char buf[256] = {};
            if (fgets(buf, sizeof buf, f)) {
                for (char * p = strtok(buf, ",\n"); p; p = strtok(nullptr, ",\n")) {
                    int a = 0, b = 0;
                    const int n = sscanf(p, "%d-%d", &a, &b);
                    if (n < 1) continue;
                    if (n == 1) b = a;
                    if ((int) v.size() <= b) v.resize((size_t) b + 1, false);
                    for (int c = a; c <= b; ++c) v[(size_t) c] = true;
                }
            }
            fclose(f);
        }
        return v;
    }();
    return cpu >= 0 && cpu < (int) big.size() && big[(size_t) cpu];
}
static bool gsq_use_gather(int type, int T) {
    const int mode = gsq_gather_mode();
    if (mode == 0) return false;
    if (mode == 1) return true;
#if defined(__linux__)
    static thread_local int tl_big = -1;
    if (tl_big < 0) tl_big = cpu_is_hybrid_big(sched_getcpu()) ? 1 : 0;
    if (!tl_big) return false;
#else
    return false;
#endif
    // types where gathers win on a P core; iq2_xxs loses from T = 2, iq2_s is a wash
    return type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ2_XS || (type == GGML_TYPE_IQ2_XXS && T == 1);
}
#endif

bool gsq_dot_supported(int type) {
#if defined(__AVX2__)
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q2_0;
#else
    (void) type;
    return false;
#endif
}

bool gsq_dot_preferred(int type, int T) {
    if (T < 1 || T > 4 || !gsq_dot_supported(type)) return false;
    return type == GGML_TYPE_IQ4_NL ? T >= 3 : true;
}

void gsq_dot(int type, int K, const void * W, size_t row_bytes, int64_t r0, int64_t r1, const void * const * y, int T,
             float * out, int64_t os) {
#if defined(__AVX2__)
    const bool                 gather = gsq_use_gather(type, T);
    const char *               w  = (const char *) W;
    const block_q8_K * const * yy = (const block_q8_K * const *) y;
    switch (type) {
        case GGML_TYPE_IQ2_XXS: GSQ_DISPATCH(dot_iq2_xxs); break;
        case GGML_TYPE_IQ2_XS:  GSQ_DISPATCH(dot_iq2_xs);  break;
        case GGML_TYPE_IQ2_S:   GSQ_DISPATCH(dot_iq2_s);   break;
        case GGML_TYPE_IQ3_XXS: GSQ_DISPATCH(dot_iq3_xxs); break;
        case GGML_TYPE_IQ3_S:   GSQ_DISPATCH(dot_iq3_s);   break;
        case GGML_TYPE_IQ4_NL: {
            const block_q8_0 * const * y8 = (const block_q8_0 * const *) y;
            switch (T) {
                case 1: dot_iq4_nl<1>(K, w, row_bytes, r0, r1, y8, out, os); return;
                case 2: dot_iq4_nl<2>(K, w, row_bytes, r0, r1, y8, out, os); return;
                case 3: dot_iq4_nl<3>(K, w, row_bytes, r0, r1, y8, out, os); return;
                case 4: dot_iq4_nl<4>(K, w, row_bytes, r0, r1, y8, out, os); return;
                default: break;
            }
            break;
        }
        case GGML_TYPE_Q2_0:
            switch (T) {
                case 1: dot_q2_0_prep<1>(K, w, row_bytes, r0, r1, y, out, os); return;
                case 2: dot_q2_0_prep<2>(K, w, row_bytes, r0, r1, y, out, os); return;
                case 3: dot_q2_0_prep<3>(K, w, row_bytes, r0, r1, y, out, os); return;
                case 4: dot_q2_0_prep<4>(K, w, row_bytes, r0, r1, y, out, os); return;
                default: break;
            }
            break;
        default: break;
    }
#endif
    (void) type; (void) K; (void) W; (void) row_bytes; (void) r0; (void) r1; (void) y; (void) T; (void) out; (void) os;
    GGML_ABORT("gsq_dot: unsupported type / T");
}

}  // namespace ai::offload
