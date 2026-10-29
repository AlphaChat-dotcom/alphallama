// SPDX-License-Identifier: Apache-2.0
// The MoE block's tail in one kernel (ALPHA_MOE_DOWN9): the routed experts' down projection over k
// slots, the shared expert's down under its own codec, the routing-weighted merge, the sigmoid gate
// and the residual. One block per output row, one warp per path, codec chosen by warp index.
//
// Separate from kernels.h on purpose: that header is a plain C++ interface with no ggml or Tensor
// types in it, and this one needs both.
#pragma once

#include "core/tensor.h"

struct ggml_backend;

namespace ai::kern {

// Returns false if the shapes or quant types are unsupported; the caller must then fall back to the
// split path. Both activations are f32 and are quantised inside.
// Ask before shaping the graph. Under AI_MOE_DOWN9 the caller omits the down projections, so a
// late decline cannot be recovered from.
bool moe_down9_width_ok(int64_t T);

bool moe_down9(ggml_backend * backend, const Tensor & down_exps, const Tensor & act_routed,
               const Tensor & ids, const Tensor & weights, const Tensor & down_shexp,
               const Tensor & act_shexp, const Tensor & score, const Tensor & residual, const Tensor & dst);

}   // namespace ai::kern
