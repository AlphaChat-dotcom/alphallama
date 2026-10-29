// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common.cuh"

// ALPHA_MOE_DOWN9: one kernel for the tail of a MoE decode block: the routed experts' down
// projection over k slots, the shared expert's down projection under its own quant type, the
// routing-weighted merge, the shared-expert sigmoid gate and the trailing addends (residual).
// The [n_embd, k, T] per-slot intermediate never exists.
//
// Shape follows NInfer's nine-way down kernel: one block per output row, one warp per path, the
// codec selected by warp index, and warp 0 merging in fixed rank order so the result is reproducible
// run to run. It is not bitwise equal to the split path, which reduces each row's K across four
// warps, so ALPHA_MOE_DOWN9 is opt-in.
//
// down_shexp / act_shexp / score may all be null, in which case the shared expert is expected to
// arrive as one of the addends (computed by its own kernels).
//
// `score` is [1, T] and its row pitch is honoured (nb[1]), not assumed to be 1: a fused router|score
// projection hands over a view whose pitch is the full router width, so flat indexing would read
// another expert's logit for every T > 1 (the MTP verify rows).
void ggml_cuda_op_moe_down9(ggml_backend_cuda_context & ctx,
                            const ggml_tensor * down_exps,   // [n_ff, n_embd, n_expert]
                            const ggml_tensor * act_routed,  // [n_ff, k, T] f32
                            const ggml_tensor * ids,         // [k, T] i32
                            const ggml_tensor * weights,     // [1, k, T] f32
                            const ggml_tensor * down_shexp,  // [n_ff, n_embd] or null
                            const ggml_tensor * act_shexp,   // [n_ff, T] f32 or null
                            const ggml_tensor * score,       // [1, T] f32 raw gate, or null
                            const ggml_tensor * const * addends, int n_add,
                            ggml_tensor *       dst);        // [n_embd, T] f32

// Width ceiling for the fused path. Above the MMVQ batch bound of 8 the reference dispatch
// switches to MMQ, which no warp count reproduces, and prefill chunks can exceed that width. So
// under exact mode the fused path is not taken above 8.
// Opt out with ALPHA_MOE_DOWN9_EXACT=0 when bit-exactness is not required.
bool ggml_cuda_moe_down9_width_ok(int64_t T);

bool ggml_cuda_moe_down9_supported(const ggml_tensor * down_exps, const ggml_tensor * act_routed,
                                   const ggml_tensor * ids, const ggml_tensor * weights,
                                   const ggml_tensor * down_shexp, const ggml_tensor * act_shexp,
                                   const ggml_tensor * score,
                                   const ggml_tensor * const * addends, int n_add,
                                   const ggml_tensor * dst);
