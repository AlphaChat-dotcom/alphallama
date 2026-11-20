// SPDX-License-Identifier: Apache-2.0
// Many-token AVX-VNNI kernels for the GSQ-RCO expert types. See gsq_gemm.h. Built with -mavxvnni; every entry point
// checks the CPU first (gsq_gemm_supported).
//
// Lane order. gsq_dot's IQ2/IQ3 kernels sum maddubs(grid, sign * q8) then madd by the scale, so 32-bit lane j of a
// 32-byte block holds bytes 4j..4j+3. Here the weights are decoded once per row to signed 16-bit (grid * sign *
// scale) and the activations once per call to 16-bit, both in the order A = bytes {4j, 4j+1}, B = {4j+2, 4j+3}
// (j = 0..7): vpdpwssd(A) + vpdpwssd(B) puts the same four products in lane j. IQ4_NL and Q2_0 use vpdpbusd, whose
// lane j is bytes 4j..4j+3 in natural order: IQ4_NL as (y + 128) x w with -128 * sum(w) preloaded per lane, Q2_0 as
// codes x prepared q8 (the same products as maddubs there). Integer sums are exact either way, and the float steps
// (per-block scale, fmadd order, final hsum) are gsq_dot's, so the results are bit-identical.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"

#include "offload/gsq_cpu.h"
#include "offload/gsq_gemm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if defined(__AVX2__) && defined(__AVXVNNI__)
#include <cpuid.h>
#include <immintrin.h>
#define GSQ_GEMM 1
#endif

namespace ai::offload {

#if defined(GSQ_GEMM)
namespace {

struct EvenSigns {   // as gsq_cpu.cpp: entry i = 8 bytes of +1 / -1 from ksigns_iq2xs[i]
    uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            uint64_t e = 0;
            for (int j = 0; j < 8; ++j) e |= (uint64_t) (((ksigns_iq2xs[i] >> j) & 1) ? 0xFFu : 0x01u) << (8 * j);
            v[i] = e;
        }
    }
};
const EvenSigns k_signs;

inline float h2f(ggml_half h) { return _cvtsh_ss(h); }

inline float hsum8(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// 32 signed bytes -> 32 int16 in A/B lane order (dst[0..15] = A, dst[16..31] = B); with scales, bytes 0..15 are
// multiplied by lo and bytes 16..31 by hi (every GSQ type's scale is constant over 16 weights)
inline __m256i ab_order(__m256i w8) {
    const __m256i ctl = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                         0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    return _mm256_permute4x64_epi64(_mm256_shuffle_epi8(w8, ctl), 0xD8);
}
inline void emit32(__m256i w8, int lo, int hi, int16_t * dst) {
    const __m256i p = ab_order(w8);
    const __m256i s = _mm256_setr_m128i(_mm_set1_epi16((int16_t) lo), _mm_set1_epi16((int16_t) hi));
    _mm256_storeu_si256((__m256i *) dst, _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(p)), s));
    _mm256_storeu_si256((__m256i *) (dst + 16), _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1)), s));
}
inline void emit32(__m256i w8, int16_t * dst) {
    const __m256i p = ab_order(w8);
    _mm256_storeu_si256((__m256i *) dst, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p)));
    _mm256_storeu_si256((__m256i *) (dst + 16), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1)));
}
inline __m256i signs4(uint32_t a) {   // the 4 x 7-bit even-parity sign groups of an IQ2_XXS / IQ3_XXS word
    const uint64_t * s = k_signs.v;
    return _mm256_set_epi64x((long long) s[(a >> 21) & 127], (long long) s[(a >> 14) & 127], (long long) s[(a >> 7) & 127],
                             (long long) s[a & 127]);
}
inline __m256i mask_signs(const uint16_t * signs) {   // IQ2_S / IQ3_S explicit sign bytes -> 0xFF / 0x00 per weight
    const __m256i mask1 = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i mask2 = _mm256_set1_epi64x((long long) 0x8040201008040201ULL);
    const __m256i a = _mm256_set1_epi32((int) (signs[0] | ((uint32_t) signs[1] << 16)));
    return _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(a, mask1), mask2), mask2);
}
inline __m256i apply_mask(__m256i q, __m256i m) { return _mm256_sub_epi8(_mm256_xor_si256(m, q), m); }

// ---- row decoders: one row of nb superblocks -> w16[nb * 256] (A/B order, scaled), dx[nb]
void dec_iq2_xxs(const char * row, int nb, int16_t * w16, float * dx) {
    const block_iq2_xxs * x = (const block_iq2_xxs *) row;
    for (int i = 0; i < nb; ++i, w16 += QK_K) {
        dx[i] = h2f(x[i].d);
        const uint16_t * q2 = x[i].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 2, q2 += 8) {
            uint32_t a[4];
            memcpy(a, q2, sizeof a);
            const uint8_t * a8 = (const uint8_t *) a;
            const __m256i g1 = _mm256_set_epi64x(iq2xxs_grid[a8[3]], iq2xxs_grid[a8[2]], iq2xxs_grid[a8[1]], iq2xxs_grid[a8[0]]);
            const __m256i g2 = _mm256_set_epi64x(iq2xxs_grid[a8[11]], iq2xxs_grid[a8[10]], iq2xxs_grid[a8[9]], iq2xxs_grid[a8[8]]);
            const int s1 = 2 * (int) (a[1] >> 28) + 1, s2 = 2 * (int) (a[3] >> 28) + 1;
            emit32(_mm256_sign_epi8(g1, signs4(a[1])), s1, s1, w16 + ib * 32);
            emit32(_mm256_sign_epi8(g2, signs4(a[3])), s2, s2, w16 + ib * 32 + 32);
        }
    }
}
void dec_iq2_xs(const char * row, int nb, int16_t * w16, float * dx) {
    static const char bss1[32] = { 0,0,0,0,0,0,0,0, 2,2,2,2,2,2,2,2, 4,4,4,4,4,4,4,4, 6,6,6,6,6,6,6,6 };
    static const char bss2[32] = { 8,8,8,8,8,8,8,8, 10,10,10,10,10,10,10,10, 12,12,12,12,12,12,12,12, 14,14,14,14,14,14,14,14 };
    static const uint8_t bsel[32] = { 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
    static const uint8_t khelp[32] = { 0x00,0x80,0x80,0x00,0x80,0x00,0x00,0x80,0x80,0x00,0x00,0x80,0x00,0x80,0x80,0x00,
                                       0x00,0x80,0x80,0x00,0x80,0x00,0x00,0x80,0x80,0x00,0x00,0x80,0x00,0x80,0x80,0x00 };
    const __m256i mone = _mm256_set1_epi8(1), m511 = _mm256_set1_epi16(511);
    const __m256i bit_selector = _mm256_loadu_si256((const __m256i *) bsel), bit_helper = _mm256_loadu_si256((const __m256i *) khelp);
    const __m256i shuf1 = _mm256_loadu_si256((const __m256i *) bss1), shuf2 = _mm256_loadu_si256((const __m256i *) bss2);
    const block_iq2_xs * x = (const block_iq2_xs *) row;
    for (int i = 0; i < nb; ++i, w16 += QK_K) {
        dx[i] = h2f(x[i].d);
        const uint16_t * q2 = x[i].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 4, q2 += 16) {
            const __m256i q2_data = _mm256_loadu_si256((const __m256i *) q2);
            alignas(32) uint16_t gi[16];
            _mm256_store_si256((__m256i *) gi, _mm256_and_si256(q2_data, m511));
            const __m256i psb  = _mm256_srli_epi16(q2_data, 9);
            const __m256i psbu = _mm256_srli_epi16(q2_data, 13);
            const __m256i full = _mm256_or_si256(psb, _mm256_shuffle_epi8(bit_helper, _mm256_xor_si256(psb, psbu)));
            const __m128i fl = _mm256_castsi256_si128(full), fh = _mm256_extractf128_si256(full, 1);
            const __m256i f1 = _mm256_setr_m128i(fl, fl), f2 = _mm256_setr_m128i(fh, fh);
            auto smask = [&](__m256i f, __m256i sh) {
                const __m256i s = _mm256_shuffle_epi8(f, sh);
                return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(s, bit_selector), bit_selector), mone);
            };
            const __m256i sg[4] = { smask(f1, shuf1), smask(f1, shuf2), smask(f2, shuf1), smask(f2, shuf2) };
            for (int j = 0; j < 4; ++j) {
                const __m256i g = _mm256_set_epi64x(iq2xs_grid[gi[4 * j + 3]], iq2xs_grid[gi[4 * j + 2]], iq2xs_grid[gi[4 * j + 1]],
                                                    iq2xs_grid[gi[4 * j]]);
                const uint8_t sc = x[i].scales[ib + j];
                emit32(_mm256_sign_epi8(g, sg[j]), 2 * (sc & 0xf) + 1, 2 * (sc >> 4) + 1, w16 + (ib + j) * 32);
            }
        }
    }
}
void dec_iq2_s(const char * row, int nb, int16_t * w16, float * dx) {
    const block_iq2_s * x = (const block_iq2_s *) row;
    for (int i = 0; i < nb; ++i, w16 += QK_K) {
        dx[i] = h2f(x[i].d);
        const uint8_t *  qs    = x[i].qs;
        const uint8_t *  qh    = x[i].qh;
        const uint16_t * signs = (const uint16_t *) (x[i].qs + QK_K / 8);
        for (int ib = 0; ib < QK_K / 32; ib += 2, qs += 8, signs += 4) {
            const __m256i g1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib + 0] << 2) & 0x300)], iq2s_grid[qs[2] | ((qh[ib + 0] << 4) & 0x300)],
                                                 iq2s_grid[qs[1] | ((qh[ib + 0] << 6) & 0x300)], iq2s_grid[qs[0] | ((qh[ib + 0] << 8) & 0x300)]);
            const __m256i g2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib + 1] << 2) & 0x300)], iq2s_grid[qs[6] | ((qh[ib + 1] << 4) & 0x300)],
                                                 iq2s_grid[qs[5] | ((qh[ib + 1] << 6) & 0x300)], iq2s_grid[qs[4] | ((qh[ib + 1] << 8) & 0x300)]);
            const uint8_t s1 = x[i].scales[ib], s2 = x[i].scales[ib + 1];
            emit32(apply_mask(g1, mask_signs(signs)), 2 * (s1 & 0xf) + 1, 2 * (s1 >> 4) + 1, w16 + ib * 32);
            emit32(apply_mask(g2, mask_signs(signs + 2)), 2 * (s2 & 0xf) + 1, 2 * (s2 >> 4) + 1, w16 + ib * 32 + 32);
        }
    }
}
void dec_iq3_xxs(const char * row, int nb, int16_t * w16, float * dx) {
    const block_iq3_xxs * x = (const block_iq3_xxs *) row;
    for (int i = 0; i < nb; ++i, w16 += QK_K) {
        dx[i] = h2f(x[i].d);
        const uint8_t * q3  = x[i].qs;
        const uint8_t * gas = x[i].qs + QK_K / 4;
        for (int ib = 0; ib < QK_K / 32; ib += 2, q3 += 16, gas += 8) {
            const __m256i g1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m256i g2 = _mm256_set_epi32(iq3xxs_grid[q3[15]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[12]],
                                                iq3xxs_grid[q3[11]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[9]], iq3xxs_grid[q3[8]]);
            uint32_t a[2];
            memcpy(a, gas, sizeof a);
            const int s1 = 2 * (int) (a[0] >> 28) + 1, s2 = 2 * (int) (a[1] >> 28) + 1;
            emit32(_mm256_sign_epi8(g1, signs4(a[0])), s1, s1, w16 + ib * 32);
            emit32(_mm256_sign_epi8(g2, signs4(a[1])), s2, s2, w16 + ib * 32 + 32);
        }
    }
}
void dec_iq3_s(const char * row, int nb, int16_t * w16, float * dx) {
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8), idx_mask = _mm256_set1_epi32(256);
    const block_iq3_s * x = (const block_iq3_s *) row;
    for (int i = 0; i < nb; ++i, w16 += QK_K) {
        dx[i] = h2f(x[i].d);
        const uint8_t *  qs    = x[i].qs;
        const uint8_t *  qh    = x[i].qh;
        const uint16_t * signs = (const uint16_t *) x[i].signs;
        for (int ib = 0; ib < QK_K / 32; ib += 2, qs += 16, signs += 4) {
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *) qs));
            __m256i v0 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib + 0]), idx_shift), idx_mask);
            __m256i v1 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib + 1]), idx_shift), idx_mask);
            v0 = _mm256_or_si256(v0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            v1 = _mm256_or_si256(v1, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            alignas(32) uint32_t idx[16];
            _mm256_store_si256((__m256i *) idx, v0);
            _mm256_store_si256((__m256i *) (idx + 8), v1);
            const __m256i g1 = _mm256_set_epi32(iq3s_grid[idx[7]], iq3s_grid[idx[6]], iq3s_grid[idx[5]], iq3s_grid[idx[4]],
                                                iq3s_grid[idx[3]], iq3s_grid[idx[2]], iq3s_grid[idx[1]], iq3s_grid[idx[0]]);
            const __m256i g2 = _mm256_set_epi32(iq3s_grid[idx[15]], iq3s_grid[idx[14]], iq3s_grid[idx[13]], iq3s_grid[idx[12]],
                                                iq3s_grid[idx[11]], iq3s_grid[idx[10]], iq3s_grid[idx[9]], iq3s_grid[idx[8]]);
            const int s1 = 2 * (x[i].scales[ib / 2] & 0xf) + 1, s2 = 2 * (x[i].scales[ib / 2] >> 4) + 1;
            emit32(apply_mask(g1, mask_signs(signs)), s1, s1, w16 + ib * 32);
            emit32(apply_mask(g2, mask_signs(signs + 2)), s2, s2, w16 + ib * 32 + 32);
        }
    }
}

// ---- IQ2/IQ3 x q8_K: R rows x TN tokens
template <int R, int TN>
void tile_q8k(int nb, const int16_t * const * w, const float * const * dx, const GsqActs & A, int t0, float factor,
              float * const * out, int64_t r) {
    const int K = A.K;
    __m256 acc[R][TN];
    for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) acc[i][t] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        __m256i s[R][TN];
        for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) s[i][t] = _mm256_setzero_si256();
        const int16_t * wp[R];
        const int16_t * ap[TN];
        for (int i = 0; i < R; ++i) wp[i] = w[i] + (size_t) b * QK_K;
        for (int t = 0; t < TN; ++t) ap[t] = A.a16.data() + (size_t) (t0 + t) * K + (size_t) b * QK_K;
#pragma GCC unroll 4
        for (int k = 0; k < QK_K; k += 16) {
            __m256i wv[R];
            for (int i = 0; i < R; ++i) wv[i] = _mm256_loadu_si256((const __m256i *) (wp[i] + k));
            for (int t = 0; t < TN; ++t) {
                const __m256i av = _mm256_loadu_si256((const __m256i *) (ap[t] + k));
                for (int i = 0; i < R; ++i) s[i][t] = _mm256_dpwssd_avx_epi32(s[i][t], wv[i], av);
            }
        }
        for (int i = 0; i < R; ++i) {
            for (int t = 0; t < TN; ++t) {
                acc[i][t] = _mm256_fmadd_ps(_mm256_set1_ps(dx[i][b] * A.dy[(size_t) (t0 + t) * nb + b]), _mm256_cvtepi32_ps(s[i][t]), acc[i][t]);
            }
        }
    }
    for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) out[t0 + t][r + i] = factor * hsum8(acc[i][t]);
}

// ---- IQ4_NL x q8_0: w8 natural order, negc = -128 * (per-lane sum of w) per block
template <int R, int TN>
void tile_iq4nl(int nb, const int8_t * const * w, const __m256i * const * negc, const float * const * dx, const GsqActs & A,
                int t0, float * const * out, int64_t r) {
    const int K = A.K;
    __m256 acc1[R][TN], acc2[R][TN];
    for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) { acc1[i][t] = _mm256_setzero_ps(); acc2[i][t] = _mm256_setzero_ps(); }
    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        for (int h = 0; h < 2; ++h) {
            __m256i wv[R];
            for (int i = 0; i < R; ++i) wv[i] = _mm256_loadu_si256((const __m256i *) (w[i] + (size_t) (ib + h) * 32));
            for (int t = 0; t < TN; ++t) {
                const __m256i av = _mm256_loadu_si256((const __m256i *) (A.au8.data() + (size_t) (t0 + t) * K + (size_t) (ib + h) * 32));
                const float   d  = A.dy[(size_t) (t0 + t) * nb + ib + h];
                for (int i = 0; i < R; ++i) {
                    const __m256i p = _mm256_dpbusd_avx_epi32(negc[i][ib + h], av, wv[i]);
                    __m256 & a = h ? acc2[i][t] : acc1[i][t];
                    a = _mm256_fmadd_ps(_mm256_set1_ps(d * dx[i][ib + h]), _mm256_cvtepi32_ps(p), a);
                }
            }
        }
    }
    for (int i = 0; i < R; ++i) {
        for (int t = 0; t < TN; ++t) {
            float s = hsum8(_mm256_add_ps(acc1[i][t], acc2[i][t]));
            const block_q8_0 * y = (const block_q8_0 *) A.y[t0 + t];
            for (int j = ib; j < nb; ++j) {   // odd tail, scalar as gsq_dot
                int sumi = 0;
                for (int k = 0; k < QK4_NL; ++k) sumi += y[j].qs[k] * w[i][(size_t) j * 32 + k];
                s += h2f(y[j].d) * dx[i][j] * sumi;
            }
            out[t0 + t][r + i] = s;
        }
    }
}

// ---- Q2_0 x prepared q8_0: per block pair, the 4 code planes (32 bytes each) and the pair's two scales
template <int R, int TN>
void tile_q2_0(int np, const uint8_t * const * c, const __m256 * const * dxv, const GsqActs & A, int t0, float * const * out,
               int64_t r) {
    __m256 acc[R][TN];
    for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) acc[i][t] = _mm256_setzero_ps();
    const int8_t * yp[TN]; const int32_t * sy[TN]; const float * dyl[TN];
    for (int t = 0; t < TN; ++t) {
        yp[t]  = (const int8_t *) A.y[t0 + t];
        sy[t]  = (const int32_t *) (yp[t] + (size_t) np * 128);
        dyl[t] = (const float *) (sy[t] + (size_t) np * 8);
    }
    for (int p = 0; p < np; ++p) {
        __m256i s[R][TN];
        for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) s[i][t] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            __m256i cv[R];
            for (int i = 0; i < R; ++i) cv[i] = _mm256_loadu_si256((const __m256i *) (c[i] + (size_t) p * 128 + j * 32));
            for (int t = 0; t < TN; ++t) {
                const __m256i yv = _mm256_loadu_si256((const __m256i *) (yp[t] + (size_t) p * 128 + j * 32));
                for (int i = 0; i < R; ++i) s[i][t] = _mm256_dpbusd_avx_epi32(s[i][t], cv[i], yv);
            }
        }
        for (int t = 0; t < TN; ++t) {
            const __m256i syv = _mm256_loadu_si256((const __m256i *) (sy[t] + p * 8));
            const __m256  dl  = _mm256_loadu_ps(dyl[t] + p * 8);
            for (int i = 0; i < R; ++i) {
                acc[i][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(s[i][t], syv)), _mm256_mul_ps(dxv[i][p], dl), acc[i][t]);
            }
        }
    }
    for (int i = 0; i < R; ++i) for (int t = 0; t < TN; ++t) out[t0 + t][r + i] = hsum8(acc[i][t]);
}

#define GSQ_TILES(fn, ...)                                                          \
    for (int t0 = 0; t0 < T; t0 += 4) {                                             \
        const int tn = std::min(4, T - t0);                                         \
        if (R == 2) {                                                               \
            switch (tn) {                                                           \
                case 4: fn<2, 4>(__VA_ARGS__); break; case 3: fn<2, 3>(__VA_ARGS__); break; \
                case 2: fn<2, 2>(__VA_ARGS__); break; default: fn<2, 1>(__VA_ARGS__); break; \
            }                                                                       \
        } else {                                                                    \
            switch (tn) {                                                           \
                case 4: fn<1, 4>(__VA_ARGS__); break; case 3: fn<1, 3>(__VA_ARGS__); break; \
                case 2: fn<1, 2>(__VA_ARGS__); break; default: fn<1, 1>(__VA_ARGS__); break; \
            }                                                                       \
        }                                                                           \
    }

bool is_q8k_type(int type) {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ3_S;
}

bool cpu_has_avxvnni() {
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(7, 1, &a, &b, &c, &d)) return false;
    return ((a >> 4) & 1) != 0 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

}  // namespace
#endif

bool gsq_gemm_supported(int type) {
#if defined(GSQ_GEMM)
    static const bool on = cpu_has_avxvnni() && (!getenv("AI_GSQ_GEMM") || atoi(getenv("AI_GSQ_GEMM")) != 0);
    return on && (is_q8k_type(type) || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q2_0);
#else
    (void) type;
    return false;
#endif
}

void gsq_gemm_prep(int type, int K, const void * const * y, int T, GsqActs & A) {
    A.type = type; A.K = K; A.T = T; A.y = y; A.fallback = false;
#if defined(GSQ_GEMM)
    if (is_q8k_type(type)) {
        const int nb = K / QK_K;
        A.a16.resize((size_t) T * K);
        A.dy.resize((size_t) T * nb);
        const __m256i m128 = _mm256_set1_epi8((char) -128);
        for (int t = 0; t < T; ++t) {
            const block_q8_K * yb = (const block_q8_K *) y[t];
            int16_t *          dst = A.a16.data() + (size_t) t * K;
            for (int b = 0; b < nb; ++b) {
                A.dy[(size_t) t * nb + b] = yb[b].d;
                for (int k = 0; k < QK_K; k += 32) {
                    const __m256i q = _mm256_loadu_si256((const __m256i *) (yb[b].qs + k));
                    // gsq_dot negates q8 by sign_epi8 / xor-sub, which leaves -128 as -128: not the product here
                    if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(q, m128))) A.fallback = true;
                    emit32(q, dst + (size_t) b * QK_K + k);
                }
            }
        }
    } else if (type == GGML_TYPE_IQ4_NL) {
        const int nb = K / QK4_NL;
        A.au8.resize((size_t) T * K);
        A.dy.resize((size_t) T * nb);
        for (int t = 0; t < T; ++t) {
            const block_q8_0 * yb = (const block_q8_0 *) y[t];
            for (int b = 0; b < nb; ++b) {
                A.dy[(size_t) t * nb + b] = h2f(yb[b].d);
                const __m256i q = _mm256_loadu_si256((const __m256i *) yb[b].qs);
                _mm256_storeu_si256((__m256i *) (A.au8.data() + (size_t) t * K + (size_t) b * 32), _mm256_xor_si256(q, _mm256_set1_epi8((char) 0x80)));
            }
        }
    }
#endif
}

void gsq_gemm(const GsqActs & A, const void * W, size_t row_bytes, int64_t r0, int64_t r1, float * const * out) {
    const int T = A.T, K = A.K, type = A.type;
    if (A.fallback || !gsq_gemm_supported(type)) {
        for (int t = 0; t < T; ++t) gsq_dot(type, K, W, row_bytes, r0, r1, &A.y[t], 1, out[t], 0);
        return;
    }
#if defined(GSQ_GEMM)
    const char * w = (const char *) W;
    if (is_q8k_type(type)) {
        const int nb = K / QK_K;
        void (*dec)(const char *, int, int16_t *, float *) =
            type == GGML_TYPE_IQ2_XXS ? dec_iq2_xxs : type == GGML_TYPE_IQ2_XS ? dec_iq2_xs : type == GGML_TYPE_IQ2_S ? dec_iq2_s
            : type == GGML_TYPE_IQ3_XXS ? dec_iq3_xxs : dec_iq3_s;
        const float factor = type == GGML_TYPE_IQ3_XXS ? 0.25f : type == GGML_TYPE_IQ3_S ? 1.0f : 0.125f;
        thread_local std::vector<int16_t> wb;
        thread_local std::vector<float>   db;
        if (wb.size() < (size_t) 2 * K) wb.resize((size_t) 2 * K);
        if (db.size() < (size_t) 2 * nb) db.resize((size_t) 2 * nb);
        for (int64_t r = r0; r < r1; r += 2) {
            const int R = (int) std::min<int64_t>(2, r1 - r);
            for (int i = 0; i < R; ++i) dec(w + (size_t) (r + i) * row_bytes, nb, wb.data() + (size_t) i * K, db.data() + (size_t) i * nb);
            const int16_t * wp[2] = { wb.data(), wb.data() + K };
            const float *   dp[2] = { db.data(), db.data() + nb };
            GSQ_TILES(tile_q8k, nb, wp, dp, A, t0, factor, out, r)
        }
    } else if (type == GGML_TYPE_IQ4_NL) {
        const int nb = K / QK4_NL;
        thread_local std::vector<int8_t>  wb;
        thread_local std::vector<__m256i> nc;
        thread_local std::vector<float>   db;
        if (wb.size() < (size_t) 2 * K) wb.resize((size_t) 2 * K);
        if (nc.size() < (size_t) 2 * nb) nc.resize((size_t) 2 * nb);
        if (db.size() < (size_t) 2 * nb) db.resize((size_t) 2 * nb);
        const __m128i values = _mm_loadu_si128((const __m128i *) kvalues_iq4nl), m4b = _mm_set1_epi8(0x0f);
        const __m256i ones = _mm256_set1_epi8(1);
        for (int64_t r = r0; r < r1; r += 2) {
            const int R = (int) std::min<int64_t>(2, r1 - r);
            for (int i = 0; i < R; ++i) {
                const block_iq4_nl * x = (const block_iq4_nl *) (w + (size_t) (r + i) * row_bytes);
                for (int b = 0; b < nb; ++b) {
                    const __m128i q  = _mm_loadu_si128((const __m128i *) x[b].qs);
                    const __m256i wv = _mm256_setr_m128i(_mm_shuffle_epi8(values, _mm_and_si128(q, m4b)),
                                                         _mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(q, 4), m4b)));
                    _mm256_storeu_si256((__m256i *) (wb.data() + (size_t) i * K + (size_t) b * 32), wv);
                    const __m256i sw = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones, wv);
                    nc[(size_t) i * nb + b] = _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_slli_epi32(sw, 7));
                    db[(size_t) i * nb + b] = h2f(x[b].d);
                }
            }
            const int8_t *  wp[2] = { wb.data(), wb.data() + K };
            const __m256i * np[2] = { nc.data(), nc.data() + nb };
            const float *   dp[2] = { db.data(), db.data() + nb };
            GSQ_TILES(tile_iq4nl, nb, wp, np, dp, A, t0, out, r)
        }
    } else if (type == GGML_TYPE_Q2_0) {
        const int nb = K / QK2_0, np = (nb + 1) / 2;
        thread_local std::vector<uint8_t> cb;
        thread_local std::vector<__m256>  dv;
        if (cb.size() < (size_t) 2 * np * 128) cb.resize((size_t) 2 * np * 128);
        if (dv.size() < (size_t) 2 * np) dv.resize((size_t) 2 * np);
        const __m256i m3 = _mm256_set1_epi8(3);
        for (int64_t r = r0; r < r1; r += 2) {
            const int R = (int) std::min<int64_t>(2, r1 - r);
            for (int i = 0; i < R; ++i) {
                const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) (r + i) * row_bytes);
                for (int p = 0, b = 0; p < np; ++p, b += 2) {
                    const bool    pair = b + 1 < nb;
                    const __m256i q = _mm256_set_m128i(pair ? _mm_loadu_si128((const __m128i *) x[b + 1].qs) : _mm_setzero_si128(),
                                                       _mm_loadu_si128((const __m128i *) x[b].qs));
                    uint8_t * dst = cb.data() + ((size_t) i * np + p) * 128;
                    for (int j = 0; j < 4; ++j) _mm256_storeu_si256((__m256i *) (dst + 32 * j), _mm256_and_si256(_mm256_srli_epi16(q, 2 * j), m3));
                    dv[(size_t) i * np + p] = _mm256_set_m128(_mm_set1_ps(pair ? h2f(x[b + 1].d) : 0.0f), _mm_set1_ps(h2f(x[b].d)));
                }
            }
            const uint8_t * cp[2] = { cb.data(), cb.data() + (size_t) np * 128 };
            const __m256 *  dp[2] = { dv.data(), dv.data() + np };
            GSQ_TILES(tile_q2_0, np, cp, dp, A, t0, out, r)
        }
    }
#else
    (void) row_bytes; (void) r0; (void) r1; (void) out; (void) W;
#endif
}

}  // namespace ai::offload
