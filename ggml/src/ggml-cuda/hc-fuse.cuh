#pragma once
#include "common.cuh"

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_mix_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_mix_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_hc_mix_fused_supported(const ggml_tensor * op);
