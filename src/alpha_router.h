#pragma once

// Alpha-MoLE: Router MLP implementation

#include "llama.h"

#include <stdint.h>
#include <vector>

struct alpha_router {
    int hidden_dim;       // base model hidden size (e.g., 2048 for 4B)
    int router_hidden;    // intermediate size (default: 64)
    int n_loras;          // number of LoRA experts
    int top_k;            // how many to select (default: 2)
    bool shared;          // same router for all layers (true) or per-layer (false)

    // Weights for shared router (FP32)
    std::vector<float> w_down;   // [hidden_dim * router_hidden]
    std::vector<float> b_down;   // [router_hidden] (optional, may be empty)
    std::vector<float> w_up;     // [router_hidden * n_loras]
    std::vector<float> b_up;     // [n_loras] (optional, may be empty)
};

struct alpha_route_result {
    int   indices[4];    // top-K LoRA indices
    float weights[4];    // top-K LoRA weights (sum to 1.0)
    int   n_selected;    // actual K used
};

// Load router weights from GGUF file
// Returns nullptr on failure
LLAMA_API alpha_router * alpha_router_load(const char * path);

// Run router forward pass on a single hidden state vector
// hidden: [hidden_dim] float array
// top_k: number of LoRAs to select (clamped to [1, n_loras])
LLAMA_API alpha_route_result alpha_router_forward(
        const alpha_router * router,
        const float        * hidden,
        int                  top_k);

// Free router
LLAMA_API void alpha_router_free(alpha_router * router);
