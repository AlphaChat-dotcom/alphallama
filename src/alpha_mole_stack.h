#pragma once

// Alpha-MoLE: LoRA weight stacking for ggml_mul_mat_id dispatch
//
// Pre-stacks all LoRA A and B matrices for each target weight tensor
// into contiguous 3D tensors. Required by ggml_mul_mat_id for per-token
// expert dispatch.
//
// Example: for base tensor "blk.0.attn_q.weight" with 3 LoRAs of rank 8:
//   stacked_a: [in_features, rank, 3]   — all 3 LoRA A matrices stacked
//   stacked_b: [rank, out_features, 3]  — all 3 LoRA B matrices stacked

#include "llama.h"
#include "ggml.h"
#include "alpha_mole_graph.h"

#include <string>
#include <unordered_map>
#include <vector>

struct llama_adapter_lora;

// Collection of stacked weights for all target tensors
struct alpha_mole_stacked_weights {
    // Map from base tensor name to stacked LoRA weights
    std::unordered_map<std::string, alpha_mole_stacked_lora> stacks;

    // ggml context that owns the stacked tensors
    ggml_context * ctx;

    int n_loras;
    int rank;       // uniform rank (all LoRAs must match)
    float scale;    // combined alpha/rank scale
};

// Stack all LoRA weights from multiple adapters into 3D tensors
// All LoRAs must have the same rank — returns nullptr if ranks differ
// Caller must call alpha_mole_stacked_weights_free when done
LLAMA_API alpha_mole_stacked_weights * alpha_mole_stack_loras(
        llama_adapter_lora ** loras,
        int                   n_loras,
        float               * scales);  // per-adapter scales (can be nullptr for equal)

// Free stacked weights
LLAMA_API void alpha_mole_stacked_weights_free(alpha_mole_stacked_weights * sw);
