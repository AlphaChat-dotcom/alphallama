#pragma once

// Alpha-MoLE: Layer Streaming
//
// When a model barely exceeds VRAM, streaming trades context length for
// more GPU layers. Instead of running layers on CPU (slow), we maximize
// GPU layers and use Q4_0 KV cache with reduced context to fit.
//
// For truly oversized models (>>VRAM), double-buffered weight streaming
// prefetches layer weights from CPU pinned memory to GPU staging buffers.
//
// Strategy:
//   1. Calculate: max ngl that fits with Q4_0 KV + minimal context
//   2. If all layers fit (model ≈ VRAM): use all GPU, reduce context
//   3. If model >> VRAM: stream weights layer-by-layer (future)

#include "llama.h"
#include "ggml.h"

struct alpha_stream_config {
    int      n_gpu_layers;         // optimized: maximize GPU layers
    int      n_ctx;                // reduced context to fit model
    enum ggml_type cache_type_k;   // always Q4_0 for streaming
    enum ggml_type cache_type_v;
    bool     fits_fully;           // true if all layers on GPU
    size_t   model_size_mb;
    int      n_layer;
    size_t   vram_free_mb;
};

// Calculate streaming configuration: maximize GPU layers, minimize KV overhead
LLAMA_API void alpha_stream_configure(
        alpha_stream_config * cfg,
        size_t model_size_mb,
        int    n_layer,
        int    n_embd,
        size_t vram_free_mb,
        int    n_loras,
        int    lora_rank);

// Print streaming configuration
LLAMA_API void alpha_stream_print(const alpha_stream_config * cfg);
