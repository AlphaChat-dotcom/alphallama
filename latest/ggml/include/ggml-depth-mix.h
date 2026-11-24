#pragma once

#include "ggml.h"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

    // attention over the depth: the rows [blocks[n_blocks - 1], ..., blocks[0], cur], each RMS-normalized with eps, are
    // weighted by softmax(row . query / sqrt(n_embd)) over the rows, for each query, and summed
    // query [n_embd, n_query] or [n_embd, n_query, n_tokens]; cur and the blocks [n_embd, n_tokens] with contiguous rows
    // result [n_embd, n_query, n_tokens]; at most GGML_MAX_SRC - 3 blocks
    // a GGML_OP_CUSTOM node run by ggml_depth_mix_cpu; a backend with a kernel recognizes it by that function
    GGML_API struct ggml_tensor * ggml_depth_mix(
            struct ggml_context * ctx,
            struct ggml_tensor  * query,
            struct ggml_tensor  * cur,
            struct ggml_tensor ** blocks,
            int                   n_blocks,
            float                 eps);

    GGML_API void ggml_depth_mix_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata);

    // the eps of a depth mix node, kept in the custom op userdata
    static inline float ggml_depth_mix_eps(const void * userdata) {
        const uint32_t u = (uint32_t) (uintptr_t) userdata;
        float eps;
        memcpy(&eps, &u, sizeof(eps));
        return eps;
    }

#ifdef __cplusplus
}
#endif
