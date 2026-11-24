#pragma once

#include "common.cuh"

// x = concat(x->src[0], x->src[1]) along dim 0 (F32, 3D, x contiguous), and the copy of its columns [off, off + n_cols)
// into st (F32, any strides, same element count as the copied view, elements in flattened order) in one kernel
void ggml_cuda_op_concat_store_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, int64_t off, int64_t n_cols, ggml_tensor * st);
