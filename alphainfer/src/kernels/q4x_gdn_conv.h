// Native GDN short conv (q4x_gdn_conv.cu): conv + SiLU over [history | new qkv columns] read in place.
#pragma once

#include <cstdint>

namespace ai::kern {

// state f32 [B][C][K-1] (updated in place when write_state), qkv f32 rows (b*L + t) with row stride ld floats (channel c
// at +c), w f32 [C][K] with row stride w_ld, out f32 [B][L][C], rec_in (nullable) f32 [B][C][K-1+L]. false = K unsupported.
bool q4x_gdn_conv(void * stream, float * state, const float * qkv, int64_t ld, const float * w, int64_t w_ld, int K,
                  float * out, float * rec_in, int C, int L, int B, bool write_state);

}  // namespace ai::kern
