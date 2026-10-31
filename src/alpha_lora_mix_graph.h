#pragma once

// AlphaLlama: Graph-based router and per-token LoRA dispatch
// Uses ggml compute graph ops for GPU-accelerated routing

#include "llama.h"
#include "ggml.h"

#include <vector>

struct llama_adapter_lora;

// Stacked LoRA weights for a single base tensor (e.g., all lora_a for wq)
// These are pre-stacked into contiguous 3D tensors for use with ggml_mul_mat_id
struct alpha_lora_mix_stacked_lora {
    ggml_tensor * stacked_a;  // [in_features, rank, n_loras]
    ggml_tensor * stacked_b;  // [rank, out_features, n_loras]
    float         scale;      // combined alpha/rank scale
    int           n_loras;
    int           rank;
};

// Router weights stored as ggml tensors (for in-graph computation)
struct alpha_lora_mix_graph_router {
    ggml_tensor * w_down;     // [hidden_dim, router_hidden]
    ggml_tensor * w_up;       // [router_hidden, n_loras]
    ggml_tensor * b_down;     // [router_hidden] or nullptr
    ggml_tensor * b_up;       // [n_loras] or nullptr
    int           n_loras;
    int           top_k;
};

// Build the router MLP as a ggml compute graph
// Input:  hidden [n_embd, n_tokens]
// Output: selected_experts [top_k, n_tokens] (indices)
//         expert_weights   [top_k, n_tokens] (normalized weights)
struct alpha_lora_mix_route_tensors {
    ggml_tensor * indices;    // [top_k, n_tokens] int32
    ggml_tensor * weights;    // [top_k, n_tokens] float
};

LLAMA_API alpha_lora_mix_route_tensors alpha_lora_mix_build_router_graph(
        ggml_context              * ctx,
        const alpha_lora_mix_graph_router * router,
        ggml_tensor               * hidden);

// Build per-token multi-LoRA application using ggml_mul_mat_id
// Replaces build_lora_mm for LoRA-mixture mode
// base_result: output of base matmul (w @ cur)
// cur:         input tensor
// stacked:     pre-stacked LoRA A/B matrices
// route:       routing decisions from alpha_lora_mix_build_router_graph
LLAMA_API ggml_tensor * alpha_lora_mix_build_lora_dispatch(
        ggml_context                    * ctx,
        ggml_tensor                     * base_result,
        ggml_tensor                     * cur,
        const alpha_lora_mix_stacked_lora   * stacked,
        const alpha_lora_mix_route_tensors  * route);
