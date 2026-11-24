#pragma once

#include "common.cuh"
#include "ggml-depth-mix.h"

// the depth mix custom op (ggml-depth-mix.h), one kernel per call; true only for a depth mix node this kernel runs
bool ggml_cuda_depth_mix_supported(const ggml_tensor * op);

void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
