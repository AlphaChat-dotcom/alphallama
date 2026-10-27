#pragma once

// Alpha-MoLE: Hardware auto-detection and configuration
//
// Detects GPU type, VRAM, system RAM, and automatically configures:
// - n_gpu_layers: how many layers to offload
// - n_ctx: context size based on available KV cache memory
// - cache_type_k/v: KV quantization level
// - Recommendations for pipeline mode

#include "llama.h"
#include "ggml.h"

#include <cstdint>

struct alpha_autoconfig {
    // Hardware detected
    const char * gpu_name;        // e.g. "NVIDIA GeForce RTX 3090"
    int          gpu_type;        // 0=none, 1=GPU, 2=iGPU
    size_t       vram_total_mb;   // total VRAM in MB
    size_t       vram_free_mb;    // free VRAM in MB
    size_t       ram_total_mb;    // total system RAM in MB
    size_t       ram_free_mb;     // free system RAM in MB

    // Model info (filled after model load)
    uint64_t     model_size_mb;   // model file size in MB
    int          n_layer;         // number of layers
    int          n_embd;          // embedding dimension

    // Recommended config
    int          n_gpu_layers;    // layers to offload to GPU
    int          n_ctx;           // context size
    enum ggml_type cache_type_k;  // KV cache key type
    enum ggml_type cache_type_v;  // KV cache value type
    bool         use_pipeline;    // recommend layer pipelining
    bool         use_kv_paging;   // recommend KV paging

    // Estimated performance
    float        est_tok_per_sec; // rough estimate
};

// Detect hardware capabilities (call before model load)
LLAMA_API void alpha_autoconfig_detect_hardware(alpha_autoconfig * cfg);

// Configure based on model + hardware (call after model load)
LLAMA_API void alpha_autoconfig_for_model(
        alpha_autoconfig * cfg,
        const struct llama_model * model,
        int n_loras,         // number of LoRA adapters
        int lora_rank);      // LoRA rank (for memory estimation)

// Pre-read GGUF metadata to calculate optimal ngl BEFORE model load
// This avoids OOM by knowing the model size before allocating GPU memory
LLAMA_API void alpha_autoconfig_from_gguf(
        const char       * model_path,
        alpha_autoconfig * cfg,
        int                n_loras,
        int                lora_rank);

// Print configuration summary
LLAMA_API void alpha_autoconfig_print(const alpha_autoconfig * cfg);
