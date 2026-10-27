#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "alpha_hw_planner.h"

#include <vector>

// Multi-GPU pipeline parallel training.
// Splits model layers across available GPUs proportional to VRAM.
// Each GPU computes forward+backward for its assigned layers.
// Hidden states and gradients are transferred between pipeline stages via DMA.

struct alpha_pipeline_stage {
    int device_idx;             // index into hw plan devices
    int layer_start;            // first layer (inclusive)
    int layer_end;              // last layer (inclusive)
    size_t vram_budget_mb;      // available VRAM for this stage
    ggml_backend_t backend;     // compute backend for this stage
};

struct alpha_train_pipeline_config {
    int n_stages;               // number of pipeline stages (0 = auto from hw scan)
    int micro_batch_size;       // tokens per micro-batch
    int n_micro_batches;        // micro-batches in flight for pipeline fill (0 = auto: n_stages * 2)
    bool async_transfers;       // overlap DMA with compute
};

struct alpha_train_pipeline {
    std::vector<alpha_pipeline_stage> stages;
    int n_layers_total;
    int n_embd;

    // Communication buffers (hidden states between stages)
    // Forward: stage[i] -> stage[i+1] passes hidden state [micro_batch, n_embd]
    // Backward: stage[i+1] -> stage[i] passes gradient [micro_batch, n_embd]
    std::vector<std::vector<float>> fwd_buffers;   // [n_stages-1][micro_batch * n_embd]
    std::vector<std::vector<float>> bwd_buffers;   // [n_stages-1][micro_batch * n_embd]

    // Pipeline schedule: 1F1B (one forward, one backward) for efficient utilization
    // Warmup: stages fill with forward passes
    // Steady state: each stage alternates forward and backward
    // Cooldown: stages drain backward passes
    int n_warmup;
    int n_steady;
    int n_cooldown;
};

// Create pipeline from hardware plan
alpha_train_pipeline * alpha_train_pipeline_create(
    struct llama_model * model,
    const struct alpha_hw_plan * hw_plan,
    alpha_train_pipeline_config config
);

// Free pipeline resources
void alpha_train_pipeline_free(alpha_train_pipeline * pipe);

// Print pipeline layout
void alpha_train_pipeline_print(const alpha_train_pipeline * pipe);

// Compute one training step with pipeline parallelism.
// Handles micro-batching, 1F1B schedule, and gradient accumulation.
// Returns average loss across all micro-batches.
float alpha_train_pipeline_step(
    alpha_train_pipeline * pipe,
    struct llama_model * model,
    struct llama_context * ctx,
    const llama_token * tokens,     // [total_batch_size]
    const llama_token * targets,    // [total_batch_size]
    int total_tokens
);
