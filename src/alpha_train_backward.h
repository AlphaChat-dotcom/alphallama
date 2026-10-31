#pragma once

// Exact-gradient backward pass through a frozen quantized transformer.
//
// ggml_mul_mat(W, dY) computes W^T @ dY using the existing quantized matmul kernels, so no new
// GPU code is needed for backward through frozen weights.
//
// For each LoRA injection point, we compute:
//   dB = scale * dY @ z^T       (z = A @ x, saved during forward)
//   dA = scale * B^T @ dY @ x^T
//
// The backward walks the transformer graph in reverse:
//   output_proj -> attention -> RoPE -> Q/K/V proj -> norm -> FFN down -> SwiGLU -> FFN up/gate -> norm

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <cstdint>

// Forward declarations
struct alpha_train_engine;

// Activation checkpoint saved during forward
struct alpha_train_activation {
    std::vector<float> data;  // flattened tensor data (CPU, F32)
    int64_t ne[4];            // shape
    int      n_dims;
};

// Per-layer saved state for backward
struct alpha_train_layer_state {
    // Activations saved at key points
    alpha_train_activation input;          // layer input (after residual)
    alpha_train_activation attn_norm_out;  // after attention norm
    alpha_train_activation attn_out;       // after attention (before FFN)
    alpha_train_activation ffn_norm_out;   // after FFN norm
    alpha_train_activation ffn_gate_pre;   // gate projection output (pre-SiLU)
    alpha_train_activation ffn_up_out;     // up projection output
    alpha_train_activation ffn_down_in;    // input to down projection (SiLU(gate)*up)
};

// LoRA gradient accumulator for one injection point
struct alpha_lora_grad {
    std::vector<float> dA;  // [rank, in_dim]
    std::vector<float> dB;  // [out_dim, rank]
};

// Per-layer LoRA gradients
struct alpha_train_layer_grads {
    alpha_lora_grad q;
    alpha_lora_grad k;
    alpha_lora_grad v;
    alpha_lora_grad o;
    // FFN (optional, if LoRA on FFN)
    alpha_lora_grad ffn_gate;
    alpha_lora_grad ffn_up;
    alpha_lora_grad ffn_down;
};

// Compute cross-entropy loss gradient (dLogits) from logits and target tokens
// Returns loss value, fills dLogits [n_vocab, n_tokens]
float alpha_train_loss_gradient(
    const float * logits,    // [n_vocab, n_tokens]
    const int   * targets,   // [n_tokens] target token IDs
    float       * dLogits,   // [n_vocab, n_tokens] output gradient
    int n_vocab,
    int n_tokens);

// RMSNorm backward
// Given d_out and the original input x, compute d_input
// x: [n, m], d_out: [n, m], d_input: [n, m], gamma: [n]
void alpha_train_rmsnorm_backward(
    const float * x,         // original input
    const float * d_out,     // gradient from above
    const float * gamma,     // norm weight
    float       * d_input,   // output gradient
    int n,                   // feature dimension
    int m,                   // batch dimension
    float eps);

// SwiGLU backward
// Forward: out = SiLU(gate) * up
// Backward: d_gate = d_out * up * SiLU'(gate), d_up = d_out * SiLU(gate)
void alpha_train_swiglu_backward(
    const float * gate_pre,  // gate values before SiLU
    const float * up_val,    // up projection values
    const float * d_out,     // gradient from above
    float       * d_gate,    // output: gradient for gate
    float       * d_up,      // output: gradient for up
    int n);                  // total elements

// LoRA gradient computation at one injection point
// Forward: y = (W + scale * B @ A) @ x
// Given: dY (gradient of loss w.r.t. y), x (saved input), A, B (LoRA matrices)
// Computes: dA, dB
void alpha_train_lora_gradient(
    const float * dY,        // [out_dim, n_tokens]
    const float * x,         // [in_dim, n_tokens] (saved during forward)
    const float * A,         // [rank, in_dim]
    const float * B,         // [out_dim, rank]
    float       * dA,        // [rank, in_dim] output
    float       * dB,        // [out_dim, rank] output
    float scale,
    int in_dim,
    int out_dim,
    int rank,
    int n_tokens);

// Backward through one frozen linear layer using ggml quantized matmul
// Computes: dX = W^T @ dY using ggml_mul_mat on the quantized weight tensor
// This uses the existing GPU kernels; no new kernel code is needed.
void alpha_train_linear_backward_gpu(
    struct ggml_context * ctx,
    struct ggml_backend * backend,
    struct ggml_tensor  * W,        // frozen weight [out_dim, in_dim] (quantized)
    const float         * dY,       // [out_dim, n_tokens]
    float               * dX,       // [in_dim, n_tokens] output
    int in_dim,
    int out_dim,
    int n_tokens);

// Full one-layer backward pass
// Given dOutput (gradient flowing from next layer), compute dInput and LoRA gradients
void alpha_train_layer_backward(
    struct alpha_train_engine * engine,
    int layer_id,
    const float * dOutput,           // [n_embd, n_tokens] gradient from next layer
    float       * dInput,            // [n_embd, n_tokens] gradient to previous layer
    alpha_train_layer_grads * grads, // accumulated LoRA gradients
    const alpha_train_layer_state * saved); // saved activations from forward
