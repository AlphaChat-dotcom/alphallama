#pragma once

// AlphaLlama: Pinned Memory Registration
//
// Registers CPU-resident model weight buffers as pinned (page-locked) memory.
// This enables faster DMA transfers between CPU and GPU (~24% speedup).
//
// llama.cpp tries to use CUDA_Host buffer type but falls back to regular CPU
// for quantized tensors. We fix this by registering the CPU buffers AFTER
// model load using cudaHostRegister().
//
// Call after llama_model_load_from_file() and before inference.

#include "llama.h"

// Register all CPU-resident model buffers as pinned memory
// Returns number of buffers registered
LLAMA_API int alpha_pinned_register_model(struct llama_model * model);

// Unregister (called automatically on model free, but can be explicit)
LLAMA_API void alpha_pinned_unregister_model(struct llama_model * model);
