#pragma once

#include "common.cuh"

// Fused short convolution with a residual, the chain
//   y = scale*(cur + ssm_conv(x, w)) + bias       (SSM_CONV, ADD, SCALE)
// in one kernel, rounded as the separate ops for the token counts where ssm_conv runs its short kernel (<= 32).
// x is [d_conv - 1 + n_t, nr, n_s] with contiguous time, w [d_conv, nr], cur and dst [nr, n_t*n_s] with contiguous rows.
// copy_x: dst overlaps x (the allocator reuses x once the convolution has read it), so the kernel reads a copy of x.
#define CONV_RES_MAX_TOKENS 32

void ggml_cuda_op_conv_res_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * conv, const ggml_tensor * cur,
        float scale, float bias, ggml_tensor * dst, bool copy_x);
