#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

    // a relative position bias folded into an attention mask, for flash attention:
    //   dst[k, t, h, s] = f16((W[:, b] . R[:, h, row] - sinks[h, row]) + mask[k, t, 0, s]),  row = t + s*nr, b = rel_pos[k, row]
    // where bucket b == n_bucket (one past the table) has a zero bias
    // W [n_rel, n_bucket] F32 contiguous, R [n_rel, n_head, nr*n_stream] F32, sinks [n_head, nr*n_stream] F32,
    // rel_pos [n_kv, nr*n_stream] I32 in [0, n_bucket], mask [n_kv, >= nr, 1, n_stream] F16 or F32 (rows 0..nr - 1 used)
    // result [n_kv, nr, n_head, n_stream] F16
    // a GGML_OP_CUSTOM node run by ggml_rel_bias_mask_cpu; a backend with a kernel recognizes it by that function
    GGML_API struct ggml_tensor * ggml_rel_bias_mask(
            struct ggml_context * ctx,
            struct ggml_tensor  * W,
            struct ggml_tensor  * R,
            struct ggml_tensor  * sinks,
            struct ggml_tensor  * rel_pos,
            struct ggml_tensor  * mask);

    GGML_API void ggml_rel_bias_mask_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata);

#ifdef __cplusplus
}
#endif
