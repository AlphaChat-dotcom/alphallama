#pragma once

#include "common.cuh"

// Fused loop control of a looped layer input, the chain
//   gate   = D x [cur_norm; msg_norm]            (CONCAT along dim 0, MUL_MAT)
//   forget = tanh(F x gate)                      (MUL_MAT, TANH)
//   gain   = tanh(G x gate)                      (MUL_MAT, TANH)
//   y      = (cur - forget*cur) + gain*msg       (MUL, SUB, MUL, ADD)
// in two kernels instead of ten, with the same rounding as the separate ops where the unfused path would use
// mul_mat_vec_f: each dot product follows its reduction for the same row length, the products are not contracted.
// D is [2*n, rank], F and G are [rank, n], all contiguous F32; cur, msg, cur_norm, msg_norm and dst are [n, n_tokens]
// F32 with contiguous rows (any row stride), n_tokens <= LOOP_CTRL_MAX_TOKENS.
#define LOOP_CTRL_MAX_TOKENS 8

void ggml_cuda_op_loop_ctrl_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur, const ggml_tensor * cur_norm, const ggml_tensor * msg, const ggml_tensor * msg_norm,
        const ggml_tensor * D, const ggml_tensor * F, const ggml_tensor * G, ggml_tensor * dst);
