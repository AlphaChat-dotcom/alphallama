#pragma once

// Multi-pass dense pipeline
//
// Runs dense models that exceed GPU VRAM by splitting layers into passes.
// Each pass loads a chunk of layer weights to GPU via async DMA,
// computes those layers contiguously on GPU, then swaps in the next chunk.
//
// Double-buffering: while GPU computes pass N, async DMA loads pass N+1.
// The hidden state carried between passes is small (n_embd x sizeof(float)).
//
// Uses the same DMA infrastructure as expert cache (alpha_cuda_dma_*).

#include "llama.h"

struct alpha_dense_pipeline;

// Thread-local globals for graph builder layer range
// When >= 0, the graph builder only builds layers in [start, end)
// When < 0, all layers are built (normal mode)
extern thread_local int g_alpha_pipeline_layer_start;
extern thread_local int g_alpha_pipeline_layer_end;

// Bookend skip: skip layers in [skip_start, skip_end) while running all others
// When skip_start >= 0, layers in [skip_start, skip_end) are skipped
// When skip_start < 0, no exclusion (normal mode)
extern thread_local int g_alpha_pipeline_skip_start;
extern thread_local int g_alpha_pipeline_skip_end;

// DLL-safe setters/getters (thread_local can't be dllexport on MSVC)
LLAMA_API void alpha_pipeline_set_layer_range(int start, int end);
LLAMA_API void alpha_pipeline_clear_layer_range(void);
LLAMA_API void alpha_pipeline_set_skip_range(int skip_start, int skip_end);
LLAMA_API void alpha_pipeline_clear_skip_range(void);

// Initialize pipeline for a model that doesn't fit in GPU
// Returns nullptr if model fits entirely (no pipeline needed)
// vram_budget_mb: available GPU VRAM after model backbone + KV cache
LLAMA_API alpha_dense_pipeline * alpha_dense_pipeline_init(
    struct llama_model * model,
    size_t vram_budget_mb);

// Number of passes per token
LLAMA_API int alpha_dense_pipeline_n_passes(const alpha_dense_pipeline * pipe);

// Prepare pass: swap layer tensor pointers to GPU buffer, set layer range globals
// Must call before llama_decode for each pass
LLAMA_API void alpha_dense_pipeline_prepare_pass(
    alpha_dense_pipeline * pipe,
    struct llama_model * model,
    int pass_idx);

// After pass completes: restore tensor pointers, start async DMA for next pass
LLAMA_API void alpha_dense_pipeline_finish_pass(
    alpha_dense_pipeline * pipe,
    struct llama_model * model,
    int pass_idx);

// Wait for DMA to complete (call before prepare_pass if not first pass)
LLAMA_API void alpha_dense_pipeline_sync(alpha_dense_pipeline * pipe);

// Print pipeline stats
LLAMA_API void alpha_dense_pipeline_print_stats(const alpha_dense_pipeline * pipe);

// Free
LLAMA_API void alpha_dense_pipeline_free(alpha_dense_pipeline * pipe);
