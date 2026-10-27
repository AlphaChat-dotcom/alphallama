#pragma once

// Alpha-MoLE: 3-Tier Memory Pipeline for Training
//
// Tier 0: GPU VRAM - 2 layer compute buffers (double-buffered) + LoRA + gradients
// Tier 1: CPU RAM - full model (mmap), activation checkpoints, Adam states
// Tier 2: NVMe SSD - model overflow (200GB+), optimizer state spill
//
// The pipeline ensures only 2 layers worth of weights are in VRAM at any time.
// Forward and backward stream weights layer-by-layer with async prefetch.

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <cstddef>

struct alpha_train_pipeline;

struct alpha_pipeline_config {
    size_t vram_budget_mb;    // GPU VRAM for training (0 = auto)
    size_t cpu_budget_mb;     // CPU RAM for activations (0 = auto)
    int    checkpoint_segments; // gradient checkpoint segments (0 = sqrt(N))
    const char * tier2_path;  // NVMe spill directory (NULL = no tier 2)
};

// Create pipeline
alpha_train_pipeline * alpha_pipeline_init(
    struct llama_model * model,
    alpha_pipeline_config config);

// Get the number of gradient checkpoint segments
int alpha_pipeline_n_checkpoints(const alpha_train_pipeline * pipe);

// Check if a layer's activations are checkpointed (should be saved during forward)
bool alpha_pipeline_is_checkpoint(const alpha_train_pipeline * pipe, int layer);

// Save activation checkpoint for a layer
void alpha_pipeline_save_checkpoint(
    alpha_train_pipeline * pipe,
    int layer,
    const float * data,
    int n_elements);

// Load activation checkpoint for backward
const float * alpha_pipeline_load_checkpoint(
    alpha_train_pipeline * pipe,
    int layer);

// NVMe tier: spill/load optimizer states for 200GB+ models
void alpha_pipeline_spill_optim(alpha_train_pipeline * pipe, int layer,
    const float * m_data, const float * v_data, int n_params);
bool alpha_pipeline_load_optim(alpha_train_pipeline * pipe, int layer,
    float * m_data, float * v_data, int n_params);

// Get VRAM budget analysis
void alpha_pipeline_print_budget(const alpha_train_pipeline * pipe);

// Free
void alpha_pipeline_free(alpha_train_pipeline * pipe);
