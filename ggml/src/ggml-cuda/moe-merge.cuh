#pragma once

#include "common.cuh"

// ALPHA_MOE_MERGE: one kernel for the MoE output merge:
//   dst = ((((e_0*w_0 + e_1*w_1) + ...) + e_{k-1}*w_{k-1}) + a_0) + a_1 ...
// replacing MUL(experts, weights) -> k VIEWs -> (k-1) ADDs -> n_add trailing ADDs. The sum runs in exactly
// the left-to-right order of the op chain and this file is compiled without FP contraction, so the result
// is bitwise equal to the unfused chain.
//   experts : [n_embd, k, T]   (MUL_MAT_ID output)
//   weights : [1, k, T]        (routing weights)
//   addends : n_add tensors of shape [n_embd, T]
//   dst     : [n_embd, T], contiguous
void ggml_cuda_op_moe_merge(ggml_backend_cuda_context & ctx,
                            const ggml_tensor * experts, const ggml_tensor * weights,
                            const ggml_tensor * const * addends, int n_add,
                            ggml_tensor * dst);
