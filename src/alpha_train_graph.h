#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

// Training graph: wraps a forward pass with ggml autodiff backward pass.
// Marks trainable parameters, computes loss, and builds gradient graph.

struct alpha_train_graph {
    // Contexts
    ggml_context * ctx;              // memory for graph tensors
    ggml_backend_sched_t sched;      // backend scheduler

    // Graph
    ggml_cgraph * graph;             // forward + backward combined graph

    // Trainable parameters (marked with GGML_TENSOR_FLAG_PARAM)
    std::vector<ggml_tensor *> params;

    // Gradient accumulators (one per param, persist across forward/backward)
    std::vector<ggml_tensor *> grad_accs;

    // Loss tensor (marked with GGML_TENSOR_FLAG_LOSS)
    ggml_tensor * loss;

    // Model info
    int n_layer;
    int n_vocab;
};

enum alpha_train_mode {
    ALPHA_TRAIN_FULL,   // train all model parameters
    ALPHA_TRAIN_LORA,   // train only LoRA adapters (if present)
    ALPHA_TRAIN_FREEZE_ATTN, // freeze attention, train FFN only
};

// Build a training graph for one batch.
// This:
//   1. Runs the forward pass (via llama_decode internally)
//   2. Computes cross-entropy loss against target tokens
//   3. Marks parameters with GGML_TENSOR_FLAG_PARAM
//   4. Calls ggml_build_backward_expand() to create backward graph
//   5. Returns the combined forward+backward graph ready for compute
//
// After ggml_backend_graph_compute(graph), read gradients from grad_accs[i].
alpha_train_graph * alpha_train_graph_create(
    llama_model * model,
    llama_context * ctx,
    enum alpha_train_mode mode
);

// Free training graph resources
void alpha_train_graph_free(alpha_train_graph * tg);

// Compute forward + backward for one batch.
// tokens: input token IDs [seq_len]
// targets: ground truth next-token IDs [seq_len] (shifted by 1 from tokens)
// Returns loss value.
float alpha_train_graph_compute(
    alpha_train_graph * tg,
    llama_context * ctx,
    const llama_token * tokens,
    const llama_token * targets,
    int seq_len
);

// Get gradient for a parameter (after compute).
// Returns pointer to gradient tensor (same shape as param).
ggml_tensor * alpha_train_graph_get_grad(alpha_train_graph * tg, int param_idx);

// GPU-accelerated training decode
// Performs forward + backward through model in a single graph execution.
// Uses ggml autodiff: builds forward graph, appends CE loss, builds backward, computes all on GPU.
// Returns loss value. Gradients accessible via alpha_train_graph_get_grad().
float alpha_train_decode_backward(
    alpha_train_graph * tg,
    struct llama_model * model,
    struct llama_context * ctx,
    const llama_token * tokens,    // input [seq_len]
    const llama_token * targets,   // ground truth [seq_len]
    int seq_len
);
