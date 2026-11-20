// SPDX-License-Identifier: Apache-2.0
// Many-token CPU kernels for the GSQ-RCO expert types (AVX-VNNI), for groups larger than gsq_dot's 4 tokens: the
// prefill CPU share runs every token routed to an expert (often dozens) through it. Each weight row is decoded once
// per call into a small buffer (signed 16-bit with the block scales folded in for the IQ2/IQ3 types, signed / 2-bit
// bytes for IQ4_NL / Q2_0) and every token is then a run of vpdpwssd / vpdpbusd over that buffer, 2 rows x 4 tokens
// at a time.
//
// Results are bit-identical to gsq_dot (and so to ggml's vec_dot at T = 1): each 32-bit lane sums the same integer
// products as the maddubs / madd lanes there, and the float scaling and accumulation order are the same.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ai::offload {

// true when gsq_gemm has a kernel for this ggml type and the CPU has AVX-VNNI (AI_GSQ_GEMM=0 turns it off)
bool gsq_gemm_supported(int ggml_type);

// The tokens' activations, converted once for the kernels (one GsqActs per group, reused across row chunks and
// across tensors of the same type and K: gate and up). y[t] is in the format gsq_dot takes for this type.
struct GsqActs {
    int                  type = -1, K = 0, T = 0;
    bool                 fallback = false;   // a q8_K row holds -128 (never from ggml's quantizer): use gsq_dot
    const void * const * y = nullptr;        // the caller's rows, read directly by Q2_0 and the fallback
    std::vector<int16_t> a16;                // IQ2/IQ3: T x K, lane-ordered (see gsq_gemm.cpp)
    std::vector<uint8_t> au8;                // IQ4_NL: T x K, y + 128
    std::vector<float>   dy;                 // per token and block: q8_K superblock d / q8_0 block d
};
void gsq_gemm_prep(int ggml_type, int K, const void * const * y, int T, GsqActs & acts);

// out[t][r] = dot(row r of W, y[t]) for r in [r0, r1), t in [0, acts.T). W rows are `row_bytes` apart.
void gsq_gemm(const GsqActs & acts, const void * W, size_t row_bytes, int64_t r0, int64_t r1, float * const * out);

}  // namespace ai::offload
