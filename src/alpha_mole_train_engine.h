#pragma once

// Alpha-MoLE: Exact-Gradient LoRA Training Engine
//
// Trains LoRA on models much larger than VRAM.
// - Layer streaming: only 2 layers in GPU at once (double-buffered)
// - Exact backward: ggml_mul_mat through quantized weights (existing kernels)
// - Gradient checkpointing: sqrt(N) activation saves
// - CPU Adam: optimizer states on pinned memory
//
// Exact-gradient alternative to MeZO, which converges faster.

#include "llama.h"
#include "alpha_mole_train_backward.h"

#include <vector>

struct alpha_train_config {
    int    rank;              // LoRA rank (default: 32)
    float  lr;                // learning rate (default: 2e-5)
    float  beta1;             // Adam beta1 (default: 0.9)
    float  beta2;             // Adam beta2 (default: 0.999)
    float  eps;               // Adam epsilon (default: 1e-8)
    float  weight_decay;      // weight decay (default: 0.0)
    int    warmup_steps;      // LR warmup (default: 10)
    int    grad_accum;        // gradient accumulation steps (default: 1)
    size_t vram_budget_mb;    // GPU VRAM budget (0 = auto)
    size_t cpu_budget_mb;     // CPU RAM budget (0 = auto)
    int    checkpoint_segments; // gradient checkpoint segments (0 = auto sqrt(N))
    bool   use_mezo;          // fallback to MeZO (default: false)
};

struct alpha_train_engine;

// Create training engine
alpha_train_engine * alpha_train_engine_init(
    struct llama_model   * model,
    struct llama_context * ctx,
    alpha_train_config     config);

// Run one training step: forward + backward + optimizer update
// Returns loss
float alpha_train_engine_step(
    alpha_train_engine * engine,
    const llama_token  * tokens,
    int                  n_tokens);

// Save LoRA adapter to GGUF
void alpha_train_engine_save(
    const alpha_train_engine * engine,
    const char               * path);

// Print training statistics
void alpha_train_engine_print_stats(const alpha_train_engine * engine);

// Get eval callback for activation capture during training
// Set on llama_context_params.cb_eval before creating context
ggml_backend_sched_eval_callback alpha_train_engine_get_callback(void);
void * alpha_train_engine_get_callback_data(alpha_train_engine * engine);

// Free
void alpha_train_engine_free(alpha_train_engine * engine);

// Full training loop (convenience wrapper)
int alpha_train_engine_train(
    struct llama_model * model,
    struct llama_context * ctx,
    const char * data_file,
    const char * output_file,
    alpha_train_config config,
    int epochs,
    int seq_len);
