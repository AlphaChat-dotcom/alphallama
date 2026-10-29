// SPDX-License-Identifier: Apache-2.0
// GDN prompt scan for prefill (q4x_gdn_scan.cu): a parallel short conv and the gated delta rule token by token with the
// state in registers.
#pragma once

#include <cstdint>

namespace ai::kern {

// q4x_gdn_conv's arithmetic for L >= K-1 rows of one stream, one thread per (row, channel); state f32 [C][K-1] updated.
bool q4x_gdn_pf_conv(void * stream, float * state, const float * qkv, int64_t ld, const float * w, int64_t w_ld, int K,
                     float * out, int C, int L);
// h: conv_out f32 [T][C] (q | k | v; q and k are L2-normalised IN PLACE), bsrc / asrc: beta / alpha pre-activations with
// row stride ba_ld floats, dt / a f32 [Hv], gate / beta scratch f32 [T][Hv], state f32 [Hv][S][S] (the ggml layout,
// updated), y f32 [T][Hv][S]. false = shape unsupported (S = 128, Hv = 3 Hk only).
bool q4x_gdn_pf_scan(void * stream, float * state, float * h, int64_t C, const float * bsrc, const float * asrc,
                     int64_t ba_ld, const float * dt, const float * a, float * gate, float * beta, float * y, int Sk,
                     int Hk, int Sv, int Hv, int64_t T, float eps);

}  // namespace ai::kern
