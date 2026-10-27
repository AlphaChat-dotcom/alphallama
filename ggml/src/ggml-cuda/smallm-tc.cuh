#pragma once
// [ALPHA_SMALLM_TC] register-dequant tensor-core matmul for 2..8 activation columns (see smallm-tc.cu).
// Returns false (and does nothing) when ALPHA_SMALLM_TC is unset/0 or the shape/type is not covered.
#include "common.cuh"

// true when ggml_cuda_mul_mat_smallm_tc would take this product (the fork's MMVQ fusion check declines those)
bool ggml_cuda_smallm_tc_covers(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

bool ggml_cuda_mul_mat_smallm_tc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                 ggml_tensor * dst);
