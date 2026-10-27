#pragma once

// Chunked Progressive Training Engine
//
// Unified engine for all training scenarios:
//   Mode A: Streamed LoRA (tiny adapters, stream base model)
//   Mode B: Chunked full backprop (fine-tuning pretrained models)
//   Mode D: 3-stage from scratch (random weights -> production)
//
// Auto-selects performance level (0-4) based on hardware:
//   Level 0: Full GPU (model+grads+adam all on GPU)
//   Level 1: Model GPU-resident, optimizer in RAM
//   Level 2: Partial GPU residency + RAM streaming
//   Level 3: Full RAM streaming
//   Level 4: Disk streaming
//
// Divides model into K chunks. Each chunk gets exact gradients.
// Frozen downstream layers propagate input gradients only (no weight grad).

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>

// Training mode

enum alpha_chunked_mode {
    ALPHA_CHUNKED_AUTO       = 0, // auto-select based on model + hardware
    ALPHA_CHUNKED_LORA       = 1, // Mode A: streamed LoRA
    ALPHA_CHUNKED_FULL_BP    = 2, // Mode B: chunked full backprop (fine-tune)
    ALPHA_CHUNKED_FROM_SCRATCH = 3, // Mode D: 3-stage from scratch
};

// Performance level (auto-detected)

enum alpha_chunked_level {
    ALPHA_LEVEL_AUTO        = -1,
    ALPHA_LEVEL_FULL_GPU    = 0,
    ALPHA_LEVEL_GPU_RESIDENT = 1,
    ALPHA_LEVEL_PARTIAL_GPU = 2,
    ALPHA_LEVEL_RAM_STREAM  = 3,
    ALPHA_LEVEL_DISK_STREAM = 4,
};

// Configuration

struct alpha_chunked_config {
    alpha_chunked_mode  mode;
    alpha_chunked_level level;

    int  K;                    // number of chunks (0 = auto)

    // LoRA (Mode A)
    int  lora_rank;            // default 64
    bool lora_merge;           // merge into base model after training

    // Optimization
    float lr;                  // learning rate (0 = auto per mode)
    float beta1, beta2, eps;
    float weight_decay;
    int   warmup_steps;
    int   grad_accum;

    // Training schedule
    int   steps_per_chunk;     // steps before rotating (0 = auto)
    int   max_cycles;          // backward sweep cycles (0 = auto)
    int   max_steps;           // hard stop (0 = unlimited)
    float convergence_threshold; // stop if eval improvement < this

    // Data
    int   seq_len;
    int   batch_size;

    // Logging
    int   log_every;
    int   eval_every;
    int   save_every;
    bool  json_progress;

    // Memory budgets (0 = auto-detect)
    size_t gpu_budget_mb;
    size_t ram_budget_mb;
    const char * work_dir;     // optimizer state spill + checkpoints
};

// Auto-detected hardware + plan

struct alpha_chunked_plan {
    alpha_chunked_mode  mode;
    alpha_chunked_level level;
    int K;

    int gpu_resident_layers;
    int ram_streamed_layers;
    int disk_streamed_layers;

    float ms_per_step;
    float total_minutes;

    int   n_layer;
    int   n_embd;
    int   n_vocab;
    int64_t n_params;
    size_t model_bytes;
    size_t layer_bytes;

    size_t gpu_vram_mb;
    size_t ram_mb;
};

struct alpha_chunked_engine;

// Default config

static inline alpha_chunked_config alpha_chunked_default_config() {
    alpha_chunked_config cfg = {};
    cfg.mode  = ALPHA_CHUNKED_AUTO;
    cfg.level = ALPHA_LEVEL_AUTO;
    cfg.K     = 0;

    cfg.lora_rank  = 64;
    cfg.lora_merge = true;

    cfg.lr           = 0;
    cfg.beta1        = 0.9f;
    cfg.beta2        = 0.999f;
    cfg.eps          = 1e-8f;
    cfg.weight_decay = 0.01f;
    cfg.warmup_steps = 0;
    cfg.grad_accum   = 1;

    cfg.steps_per_chunk = 0;
    cfg.max_cycles      = 0;
    cfg.max_steps       = 0;
    cfg.convergence_threshold = 0.01f;

    cfg.seq_len    = 512;
    cfg.batch_size = 1;

    cfg.log_every  = 10;
    cfg.eval_every = 100;
    cfg.save_every = 500;
    cfg.json_progress = false;

    cfg.gpu_budget_mb = 0;
    cfg.ram_budget_mb = 0;
    cfg.work_dir      = nullptr;
    return cfg;
}

// Public API

alpha_chunked_plan alpha_chunked_plan_training(
    struct llama_model * model,
    alpha_chunked_config config);

void alpha_chunked_print_plan(const alpha_chunked_plan & plan);

alpha_chunked_engine * alpha_chunked_init(
    struct llama_model   * model,
    struct llama_context * ctx,
    alpha_chunked_config   config,
    alpha_chunked_plan     plan);

float alpha_chunked_step(
    alpha_chunked_engine * engine,
    const llama_token    * tokens,
    int                    n_tokens);

bool alpha_chunked_next_chunk(alpha_chunked_engine * engine);
bool alpha_chunked_should_stop(const alpha_chunked_engine * engine);

int   alpha_chunked_current_chunk(const alpha_chunked_engine * engine);
int   alpha_chunked_current_cycle(const alpha_chunked_engine * engine);
int   alpha_chunked_current_step(const alpha_chunked_engine * engine);
float alpha_chunked_current_loss(const alpha_chunked_engine * engine);
float alpha_chunked_eval_loss(const alpha_chunked_engine * engine);

void alpha_chunked_save(const alpha_chunked_engine * engine, const char * path);
void alpha_chunked_save_full_model(const alpha_chunked_engine * engine, const char * path);
void alpha_chunked_print_stats(const alpha_chunked_engine * engine);
void alpha_chunked_free(alpha_chunked_engine * engine);

// Full training loop (handles chunking, sweeps, 3-stage pipeline)
LLAMA_API int alpha_chunked_train(
    struct llama_model   * model,
    struct llama_context * ctx,
    const char           * data_file,
    const char           * output_file,
    alpha_chunked_config   config);

LLAMA_API ggml_backend_sched_eval_callback alpha_chunked_get_callback(void);
void * alpha_chunked_get_callback_data(alpha_chunked_engine * engine);
