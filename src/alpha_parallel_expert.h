#pragma once

// Parallel CPU expert compute with Q8 VNNI acceleration.
//
// The GPU handles the backbone + shared expert and the CPU computes the 2 routed experts
// (top-K=2) in parallel, so no expert DMA is needed. Three parts:
//   1. Parallel 2-expert compute: half the cores per expert instead of all cores sequentially
//   2. Q8_0 weights for VNNI: no Q4_K dequant in the inner loop
//   3. Reusable ggml contexts: no per-token context malloc/free

#include "ggml.h"
#include <cstdint>
#include <cstddef>

struct alpha_parallel_expert_ctx {
    // Pre-allocated ggml contexts (one per parallel expert)
    struct ggml_context * ctx[2];
    size_t ctx_size;

    // Pre-allocated output buffers
    float * output[2];
    int64_t n_embd;

    // Thread configuration
    int threads_per_expert;   // cores allocated to each expert (total/2)
    int total_threads;

    // VNNI acceleration
    bool use_vnni;            // true if AVX-VNNI detected at init

    // Stats
    int64_t total_calls;
    double  total_ms;
};

// Initialize parallel expert context.
// n_embd: embedding dimension (for output buffer allocation)
// n_ff: max FFN intermediate dimension (for context sizing)
void alpha_parallel_expert_init(
    struct alpha_parallel_expert_ctx * pctx,
    int64_t n_embd, int64_t n_ff);

// Free parallel expert context.
void alpha_parallel_expert_free(
    struct alpha_parallel_expert_ctx * pctx);

// Compute 2 experts in parallel on CPU.
// expert_weights_0/1: pointers to expert weight tensors (Q4_K_M from mmap)
// hidden_state: input vector [n_embd] (FP32)
// output_0/output_1: output vectors [n_embd] (FP32, pre-allocated in pctx)
//
// Gate+Up path: gate_up_w [n_embd, n_ff*2], down_w [n_ff, n_embd]
// Separate path: gate_w [n_embd, n_ff], up_w [n_embd, n_ff], down_w [n_ff, n_embd]
void alpha_parallel_expert_compute(
    struct alpha_parallel_expert_ctx * pctx,
    // Expert 0
    struct ggml_tensor * gate_up_0,   // [n_embd, n_ff*2] or nullptr
    struct ggml_tensor * gate_0,      // [n_embd, n_ff] or nullptr (if no gate_up)
    struct ggml_tensor * up_0,        // [n_embd, n_ff] or nullptr
    struct ggml_tensor * down_0,      // [n_ff, n_embd]
    int expert_id_0,
    size_t slice_bytes_gu_0,          // bytes per expert slice (gate_up)
    size_t slice_bytes_d_0,           // bytes per expert slice (down)
    // Expert 1
    struct ggml_tensor * gate_up_1,
    struct ggml_tensor * gate_1,
    struct ggml_tensor * up_1,
    struct ggml_tensor * down_1,
    int expert_id_1,
    size_t slice_bytes_gu_1,
    size_t slice_bytes_d_1,
    // Input
    const float * hidden_state,
    int64_t n_embd, int64_t n_ff);

// Print performance stats.
void alpha_parallel_expert_print_stats(
    const struct alpha_parallel_expert_ctx * pctx);
