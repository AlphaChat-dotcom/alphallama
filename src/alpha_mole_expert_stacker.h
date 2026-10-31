// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaLlama: Per-Expert Tensor Stacker
//
// Some GGUF converters (antirez V4, some HuggingFace tools) store MoE experts
// as individual tensors (ffn.experts.0.w1, ffn.experts.1.w1, ...) instead of
// stacked tensors (ffn_gate_exps [dim1, dim2, n_expert]).
//
// This utility detects per-expert layout and stacks them into the format
// that ggml_mul_mat_id expects: [dim1, dim2, n_expert].
//
// Usage: call alpha_mole_stack_experts() after model loading.
// It checks weights_map for per-expert patterns, and if found,
// reads individual expert data and writes into the stacked tensor.
//
// Generic: works for any model with per-expert layout, not just DeepSeek V4.

#include "llama.h"
#include "ggml.h"

struct llama_model;

// Check if the loaded model has per-expert tensors that need stacking.
// Returns true if per-expert layout detected and stacking is needed.
bool alpha_mole_has_per_expert_tensors(const struct llama_model * model);

// Stack per-expert tensors into the model's expert tensor slots.
// Must be called after model loading but before inference.
// Reads from GGUF file via mmap/read and writes into model tensors.
bool alpha_mole_stack_experts(struct llama_model * model, const char * model_path);
