// CPU kernels for the GSQ-RCO expert types (Flash-Next misses), decoding each weight block once for
// up to 4 tokens. ggml-cpu's vec_dot decodes a row per (row, token) call; with T tokens of a round sharing a missed
// expert, the decode (grid lookups, sign tables, scales) is the dominant cost and is amortised T ways here.
//
// Activations are block_q8_K (the vec_dot_type ggml uses for the IQ2/IQ3 types), one row per token. Results are
// identical in math to ggml's vec_dot (same integer products, same per-superblock float scaling); the float
// accumulation order across superblocks is the same, so a T = 1 call is bit-identical to ggml's.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ai::offload {

// true when gsq_dot has a decode-once kernel for this ggml type on this CPU
bool gsq_dot_supported(int ggml_type);

// true when gsq_dot beats ggml-cpu's per-token vec_dot for a group of T tokens of this type (the IQ2/IQ3 types
// tie at T = 1 and win from T = 2; Q2_0 wins at every T; IQ4_NL only from T = 3)
bool gsq_dot_preferred(int ggml_type, int T);

// Activation formats by weight type: the IQ2/IQ3 types take block_q8_K rows; IQ4_NL takes block_q8_0 rows; Q2_0 takes
// buffers prepared by ggml_cpu_q2_0_prepare (ggml-cpu.h) from block_q8_0 rows (ggml_cpu_q2_0_prep_size(K) bytes each).
// out[t * out_stride + r] = dot(row r of W, y[t]) for r in [r0, r1), t in [0, T), T <= 4.
// W rows are `row_bytes` apart; y[t] is a block_q8_K row of K elements (K % 256 == 0).
void gsq_dot(int ggml_type, int K, const void * W, size_t row_bytes, int64_t r0, int64_t r1,
             const void * const * y, int T, float * out, int64_t out_stride);

}  // namespace ai::offload
