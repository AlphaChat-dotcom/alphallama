#pragma once

// Alpha-MoLE: Novel Layer Training Support
//
// Enables training of custom/novel layer types using only a forward function.
// No backward derivation needed - the system selects the strategy automatically:
//
//   Tier 2 (< 50K params): per-parameter finite differences on the layer forward
//     Exact gradient, cost = N+1 layer-forwards (microseconds to milliseconds)
//
//   Tier 3 (>= 50K params): layer-local MeZO (perturb only this layer)
//     Approximate gradient, variance reduced by (total_params / layer_params)x
//
// Usage:
//   // 1. Define your forward function
//   void my_gate_forward(float * output, const float * input,
//                        const float * params, int n_tokens, int hidden_dim);
//
//   // 2. Register
//   alpha_novel_layer layer = {};
//   layer.name = "delta_gate";
//   layer.forward_fn = my_gate_forward;
//   layer.n_params = hidden_dim;
//   layer.hidden_dim = hidden_dim;
//   alpha_novel_init_layer(layer);
//
//   // 3. Push into trainer->novel_layers
//   // 4. Training works automatically - no backward needed

#include <vector>
#include <string>
#include <random>
#include <cstring>
#include <cmath>
#include <cstdio>

// Forward function signature for novel layers.
typedef void (*alpha_novel_forward_fn)(
    float       * output,
    const float * input,
    const float * params,
    int           n_tokens,
    int           hidden_dim);

// Per-instance novel layer
struct alpha_novel_layer {
    std::string name;           // e.g., "blk.5.delta_gate"
    int         layer_id;       // transformer layer index (-1 = global)

    alpha_novel_forward_fn forward_fn;

    int n_params;
    int hidden_dim;

    // Learnable parameters (F32)
    std::vector<float> params;

    // Gradient accumulator
    std::vector<float> grad;

    // Adam optimizer state
    std::vector<float> m, v;

    // Anchor for elastic anchoring
    std::vector<float> params_anchor;

    // Saved input from forward (for backward)
    std::vector<float> input_saved;
    int saved_n_tokens;

    // Training tier (auto-selected based on n_params)
    enum { TIER_2_FINITE_DIFF = 2, TIER_3_LOCAL_MEZO = 3 } tier;

    // Tier 3: MeZO state
    std::vector<float> z;
    int mezo_k;                 // perturbations per step (default: 4)
};

// Initialize a novel layer: allocate optimizer state, auto-select tier.
// Call this before pushing into the trainer's novel_layers vector.
void alpha_novel_init_layer(alpha_novel_layer & layer);

// Run forward for all novel layers at a given layer position.
void alpha_novel_forward_all(
    std::vector<alpha_novel_layer> & layers,
    int layer_id,
    float * hidden,
    int n_tokens);

// Compute gradients via finite diff (Tier 2) or accumulate MeZO (Tier 3).
void alpha_novel_backward_all(
    std::vector<alpha_novel_layer> & layers,
    int layer_id,
    const float * d_output,
    float * d_input,
    int n_tokens);

// MeZO integration for Tier 3 layers
void alpha_novel_mezo_sample_z(std::vector<alpha_novel_layer> & layers, std::mt19937 & rng);
void alpha_novel_mezo_perturb(std::vector<alpha_novel_layer> & layers, float sign, float eps);
void alpha_novel_mezo_accumulate(std::vector<alpha_novel_layer> & layers, float g_proj);

// Apply Adam update to all novel layers.
void alpha_novel_update_all(
    std::vector<alpha_novel_layer> & layers,
    float lr, float beta1, float beta2,
    float adam_eps, float elastic_lambda,
    int step);

// Print summary of registered novel layers.
void alpha_novel_print_summary(const std::vector<alpha_novel_layer> & layers);
