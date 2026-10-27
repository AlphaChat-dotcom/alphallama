#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>
#include <cstdint>

// Quantization-Aware Training (QAT)
//
// Trains directly on quantized weights without FP32 round-trip.
// Uses Straight-Through Estimator (STE) for gradient flow through quantization:
//
//   Forward:  y = dequant(quant(x))   (quantized matmul)
//   Backward: dx = dy                  (gradient passes through unchanged)
//
// This works because:
// - The quantization function is piecewise constant (zero gradient almost everywhere)
// - STE approximates: "pretend quantization is identity for gradient purposes"
// - Empirically this converges well for 4-bit and higher
//
// For K-quant types (Q2_K, Q3_K, Q4_K), the forward uses the quantized tensor directly
// (as ggml already does), and the backward treats the weight as if it were the dequantized
// FP32 value. The optimizer updates are applied to a "shadow" FP32 copy of the weights,
// which is re-quantized after each step.
//
// Memory: shadow FP32 copy = 4x quantized size (e.g. Q4_K at 4.5 bpw -> 32 bpw shadow)
// For a 35B model at Q4_K: quantized = 20 GB, shadow = ~140 GB

struct alpha_qat_config {
    float lr;
    float beta1, beta2, eps;
    float weight_decay;
    int warmup_steps;

    // QAT-specific
    bool use_shadow_weights;    // maintain FP32 shadow for optimizer (default: true)
    int requantize_every;       // re-quantize shadow -> model every N steps (default: 1)
    float grad_clip;            // gradient clipping norm (0 = disabled)

    // Which quant types to train through (others remain frozen)
    bool train_q2k;
    bool train_q3k;
    bool train_q4k;
    bool train_q5k;
    bool train_q6k;
    bool train_q8;
};

static inline alpha_qat_config alpha_qat_default_config() {
    alpha_qat_config cfg = {};
    cfg.lr = 5e-5f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.eps = 1e-8f;
    cfg.weight_decay = 0.0f;
    cfg.warmup_steps = 50;
    cfg.use_shadow_weights = true;
    cfg.requantize_every = 1;
    cfg.grad_clip = 1.0f;
    cfg.train_q2k = false;
    cfg.train_q3k = true;
    cfg.train_q4k = true;
    cfg.train_q5k = true;
    cfg.train_q6k = true;
    cfg.train_q8 = true;
    return cfg;
}

// Shadow weight: FP32 copy of a quantized tensor for optimizer updates
struct alpha_qat_shadow {
    ggml_tensor * original;      // the quantized tensor in the model
    std::vector<float> fp32;     // FP32 shadow copy [nelements]
    std::vector<float> grad;     // accumulated gradient [nelements]
    int64_t nelements;
};

// QAT training state
struct alpha_qat_state {
    std::vector<alpha_qat_shadow> shadows;
    size_t total_shadow_bytes;
    int n_trainable;
};

// Initialize QAT: create FP32 shadow copies of selected quantized tensors
alpha_qat_state * alpha_qat_init(
    struct llama_model * model,
    alpha_qat_config config
);

// Free QAT state
void alpha_qat_free(alpha_qat_state * state);

// After computing gradients (via STE: d_quant_weight ~ d_dequant_weight),
// apply optimizer step to shadow weights and re-quantize back to model.
void alpha_qat_step(
    alpha_qat_state * state,
    alpha_qat_config config,
    int step
);

// Re-quantize all shadow weights back into the model tensors
void alpha_qat_requantize(alpha_qat_state * state);
