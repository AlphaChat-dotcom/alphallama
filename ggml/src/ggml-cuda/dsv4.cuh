#pragma once
#include "ggml.h"
void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
int ggml_cuda_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_moe_router_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_idx_score_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
bool ggml_cuda_glm_conv_concat(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_glm_conv_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int ggml_cuda_glm_gate_fused(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
