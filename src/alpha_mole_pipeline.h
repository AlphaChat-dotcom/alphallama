#pragma once

// Alpha-MoLE: Layer Pipeline Manager
//
// Enhances llama.cpp's partial offload for optimal large model inference.
//
// V1 approach: Smart configuration of existing -ngl + pinned memory + pipeline parallel
// - Calculates optimal n_gpu_layers based on model size and VRAM
// - Uses pinned (page-locked) CPU memory for faster CPU↔GPU DMA
// - Enables pipeline parallelism for overlapped transfers
// - Keeps first/last layers on GPU (hot layers) for best throughput
//
// V2 (future): Double-buffered layer streaming with per-layer graph execution

#include "llama.h"
#include "ggml.h"

#include <cstdint>

struct alpha_pipeline_config {
    // Model info
    uint64_t model_size_mb;
    int      n_layer;

    // Hardware
    size_t   vram_free_mb;
    size_t   ram_free_mb;

    // Computed config
    int      n_gpu_layers;        // layers on GPU
    int      n_hot_first;         // hot layers at start (always GPU)
    int      n_hot_last;          // hot layers at end (always GPU)
    bool     use_pinned_memory;   // use page-locked CPU memory
    bool     pipeline_parallel;   // overlap compute + transfer
    size_t   est_layer_size_mb;   // per-layer weight size

    // Estimated speed
    float    est_prefill_tok_s;   // tokens/sec for prompt processing
    float    est_gen_tok_s;       // tokens/sec for generation
};

// Calculate optimal pipeline configuration for a model
// If actual_ngl >= 0, uses that instead of auto-calculating from VRAM
LLAMA_API void alpha_pipeline_configure(
        alpha_pipeline_config * cfg,
        const struct llama_model * model,
        size_t vram_free_mb,
        size_t ram_free_mb,
        int n_loras,
        int lora_rank,
        int actual_ngl = -1);

// Apply pipeline config to model/context params
// Call BEFORE llama_model_load_from_file and llama_init_from_model
LLAMA_API void alpha_pipeline_apply_model_params(
        const alpha_pipeline_config * cfg,
        llama_model_params * model_params);

LLAMA_API void alpha_pipeline_apply_context_params(
        const alpha_pipeline_config * cfg,
        llama_context_params * ctx_params);

// Print pipeline configuration
LLAMA_API void alpha_pipeline_print(const alpha_pipeline_config * cfg);
