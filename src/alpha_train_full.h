#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>

// Full model training (from scratch or continued pre-training).
// Uses ggml autodiff for GPU-accelerated backward pass.

struct alpha_train_full_config {
    // What to train
    enum alpha_train_mode_t {
        TRAIN_FULL = 0,          // all parameters
        TRAIN_LORA = 1,          // LoRA adapters only
        TRAIN_FREEZE_ATTN = 2,   // FFN only (attention frozen)
    } mode;

    int lora_rank;               // LoRA rank (for TRAIN_LORA mode)

    // Optimization
    float lr;                    // learning rate
    float beta1, beta2, eps;     // Adam hyperparams
    float weight_decay;
    int warmup_steps;
    int grad_accumulation;       // accumulate gradients over N batches before step

    // Memory
    bool gradient_checkpointing; // recompute activations to save memory
    int checkpoint_interval;     // layers between checkpoints (0 = auto sqrt(n_layer))

    // Data
    int batch_size;              // tokens per batch
    int seq_len;                 // sequence length

    // Logging
    int log_every;               // print loss every N steps
    int save_every;              // save checkpoint every N steps
    const char * save_path;      // checkpoint save path (NULL = no saving)
};

// Gradient checkpointing: controls memory/compute tradeoff.
// Without checkpointing: all activations saved (~2x model size in memory)
// With checkpointing: only checkpoint layers saved, rest recomputed in backward
//   Memory: O(sqrt(n_layer)) instead of O(n_layer)
//   Compute: ~33% overhead (one extra forward pass per segment)
struct alpha_grad_checkpoint_config {
    bool enabled;
    int interval;    // layers between checkpoints (0 = auto: sqrt(n_layer))
    // Computed at runtime:
    int n_checkpoints;
    std::vector<int> checkpoint_layers;  // which layers to save
};

// Default training config
static inline alpha_train_full_config alpha_train_full_default_config() {
    alpha_train_full_config cfg = {};
    cfg.mode = alpha_train_full_config::TRAIN_FULL;
    cfg.lora_rank = 0;
    cfg.lr = 1e-4f;
    cfg.beta1 = 0.9f;
    cfg.beta2 = 0.999f;
    cfg.eps = 1e-8f;
    cfg.weight_decay = 0.01f;
    cfg.warmup_steps = 100;
    cfg.grad_accumulation = 1;
    cfg.gradient_checkpointing = true;
    cfg.checkpoint_interval = 0;
    cfg.batch_size = 1;
    cfg.seq_len = 512;
    cfg.log_every = 10;
    cfg.save_every = 500;
    cfg.save_path = nullptr;
    return cfg;
}

// Run full training loop.
// Returns 0 on success, non-zero on error.
LLAMA_API int alpha_train_full_run(
    struct llama_model * model,
    struct llama_context * ctx,
    const char * data_path,       // training data (text file, one doc per line)
    const char * output_path,     // output model/adapter path
    alpha_train_full_config config,
    int n_epochs
);
