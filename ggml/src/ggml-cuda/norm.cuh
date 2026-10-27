#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// ALPHA_Q8_STAGE: q8_dst, when non-null, additionally receives a q8_1 copy of the normed output so the
// consuming GEMV can skip its own quantise launch. Bit-identical to quantize_q8_1's output by construction:
// the same warp-wide reduction over the same 32 contiguous columns. The caller must ensure ne00 % QK8_1 == 0
// and that the buffer covers the padded row length.
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 block_q8_1 * q8_dst = nullptr);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor,
                                     block_q8_1 *                q8_dst = nullptr);  // ALPHA_Q8_STAGE

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
