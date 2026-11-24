#pragma once

#include "common.cuh"

// the relative bias + mask custom op (ggml-rel-bias.h), one kernel per call; true only for such a node this kernel runs
bool ggml_cuda_rel_bias_mask_supported(const ggml_tensor * op);

void ggml_cuda_op_rel_bias_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
