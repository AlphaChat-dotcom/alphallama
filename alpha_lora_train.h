#pragma once

// AlphaLlama: Native LoRA Training
//
// Bypasses ggml's broken build_backward_expand (upstream issue #18805).
// Uses MeZO (Memory-efficient Zeroth-Order) optimization:
//   - Only needs forward passes, no backward graph
//   - Perturb parameters, measure loss difference, estimate gradients
//   - Works with any model architecture, any backend (CPU/CUDA/Vulkan/Metal)
//
// LoRA: trains small A/B adapter matrices injected at attention projections.
// No view ops. No SET_ROWS. No build_backward_expand.

#include "llama.h"
#include "ggml.h"

struct alpha_lora_trainer;

struct alpha_lora_train_config {
    int    rank;          // LoRA rank (default: 32)
    float  lr;            // learning rate (default: 1e-4)
    float  beta1;         // Adam beta1 (default: 0.9)
    float  beta2;         // Adam beta2 (default: 0.999)
    float  eps;           // Adam epsilon (default: 1e-8)
    float  weight_decay;  // weight decay (default: 0.0)
    int    warmup_steps;  // LR warmup (default: 10)
    float  mezo_eps;      // MeZO perturbation magnitude (default: 1e-3)
};

// Initialize LoRA trainer for a model
// Creates A/B matrices for each attention projection (Q, V)
alpha_lora_trainer * alpha_lora_trainer_init(
        struct llama_model * model,
        struct llama_context * ctx,
        alpha_lora_train_config config);

// Run one MeZO training step: perturb → forward → estimate gradient → update
// Returns loss
float alpha_lora_train_step(
        alpha_lora_trainer * trainer,
        const llama_token * tokens,  // input sequence
        int n_tokens);

// Save trained LoRA adapter to GGUF format
void alpha_lora_save_gguf(const alpha_lora_trainer * trainer, const char * path);

// Get current training stats
void alpha_lora_print_stats(const alpha_lora_trainer * trainer);

// Free trainer
void alpha_lora_trainer_free(alpha_lora_trainer * trainer);

// Full training loop: load data file, train for N epochs, save result
int alpha_lora_train(
        struct llama_model * model,
        struct llama_context * ctx,
        const char * data_file,
        const char * output_file,
        alpha_lora_train_config config,
        int epochs,
        int seq_len);
