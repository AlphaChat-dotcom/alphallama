// ik_llama.cpp quant-type dequantization, ported from ik_llama.cpp
// (https://github.com/ikawrakow/ik_llama.cpp), MIT license, Copyright (c) Iwan Kawrakow.
// Used for GLM 5.2 support. See ggml-quants-ik.c.
#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void dequantize_row_iq4_k (const block_iq4_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq6_k (const block_iq6_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq4_ks(const block_iq4_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq5_ks(const block_iq5_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq3_ks(const block_iq3_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void dequantize_row_iq1_kt(const block_iq1_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// CPU vec_dot (dequantize-then-dot, Q8_K activation tier), used by ggml-cpu type traits.
// GGML_API is required here: these are defined in ggml-base but called from ggml-cpu, a separate
// shared library, and a Windows DLL exports nothing unless marked. The dequantize_row_* above are
// called only from within ggml-base, so they do not need it.
GGML_API void ggml_vec_dot_iq4_k_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq6_k_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq4_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq5_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq3_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq1_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
