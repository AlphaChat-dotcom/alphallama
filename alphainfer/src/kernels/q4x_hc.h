// SPDX-License-Identifier: Apache-2.0
// Native qwen4exp hyper-connection read (q4x::hc_read for <= 8 rows, Q8_0 low-rank down/up): ggml's
// ggml_hc_mix_fused V4 kernels (hc-fuse.cu: down6 + down4_fin + up_mix5) moved into AlphaInfer, with the up-mix's
// weight rows staged through shared memory by batched 4-byte loads (one DRAM latency per warp instead of a chain of
// 2-byte loads). Same lanes, vectors, operations and reduction order: bit-identical to the ggml op.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ai::kern {

// Up-mix kernel: 9 = up_mix9 (bit-identical to ggml), 10 = up_mix10 (4 outputs per warp); overrides
// AI_Q4X_HC_UP (for tools and comparisons)
void q4x_hc_set_up(int v);
// The read on fp16 tensor cores (kernels/q4x_hc_tc.cu): same inputs / outputs / workspace as q4x_hc_read,
// plus scratch of q4x_hc_tc_scratch_bytes(N, T); flat in T
bool   q4x_hc_tc_ok(int N, int LR, int T);
size_t q4x_hc_tc_scratch_bytes(int N, int T);
void   q4x_hc_read_tc(void * stream, const char * wd, int64_t wd_nb1, const char * wu, int64_t wu_nb1, const char * wi,
                      int64_t wi_nb1, int wi_bf16, const float * x, const float * wn, float * part, float * pss,
                      float * lo, float * scl, float * dst, int N, int LR, int T, float eps, void * scratch);
// K-split chunk for n_embd N (0 = this shape is not supported natively)
int  q4x_hc_chunk(int N);
// workspace floats: part [KS][T][nrows], pss [KS][T], lo [T][LR], scl [T][H]
void q4x_hc_ws(int N, int LR, int T, bool inject, int64_t * part_n, int64_t * pss_n);

// dst f32: mixed [N, T] then (when wi) inject [H, T]. wd: Q8_0 [HD -> LR] rows (nb1 bytes), wu: Q8_0 [LR -> HD] rows,
// wi: inject rows [H][HD] BF16 or F32 (nullptr = no inject), wn: f32 norm gamma [HD], x: f32 streams [N, H, T].
// Prefill: hc_combine in place over R [N, H, T] (bit-identical to ggml-cuda's hc_combine_f32)
void q4x_hc_combine_into(void * stream, float * R, const float * out, const float * inj, float s_in, float s_out, int64_t N,
                         int64_t H, int64_t T, int64_t out_ld, int64_t inj_ld);
void q4x_hc_read(void * stream, const char * wd, int64_t wd_nb1, const char * wu, int64_t wu_nb1, const char * wi,
                 int64_t wi_nb1, int wi_bf16, const float * x, const float * wn, float * part, float * pss, float * lo,
                 float * scl, float * dst, int N, int LR, int T, float eps);

}  // namespace ai::kern
