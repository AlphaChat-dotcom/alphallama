#pragma once

// CPU-side Adam optimizer with pinned memory.
//
// Adam states (m, v for each parameter) live on CPU pinned memory.
// Per-layer update: load grads from GPU -> update on CPU -> DMA updated params back.
// For 200GB+ models: optimizer states can spill to NVMe.

#include <cstddef>
#include <vector>

struct alpha_train_optim;

struct alpha_optim_config {
    float  lr;
    float  beta1;
    float  beta2;
    float  eps;
    float  weight_decay;
    int    warmup_steps;
    const char * nvme_spill_path; // NULL = no NVMe spill
};

// Per-parameter-group state
struct alpha_optim_param_group {
    float * params;     // pointer to parameter array
    float * grads;      // pointer to gradient array
    int     n_params;   // number of parameters
    float * m;          // first moment (CPU pinned)
    float * v;          // second moment (CPU pinned)
};

// Create optimizer
alpha_train_optim * alpha_optim_init(alpha_optim_config config);

// Register a parameter group (e.g., one LoRA layer's A or B)
void alpha_optim_add_param_group(
    alpha_train_optim * optim,
    float * params,
    float * grads,
    int n_params);

// Run one Adam step on all parameter groups
// step = current training step (for bias correction + warmup)
void alpha_optim_step(alpha_train_optim * optim, int step);

// Zero all gradients
void alpha_optim_zero_grad(alpha_train_optim * optim);

// Get effective learning rate (with warmup)
float alpha_optim_get_lr(const alpha_train_optim * optim, int step);

// Free
void alpha_optim_free(alpha_train_optim * optim);
