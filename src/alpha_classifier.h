#pragma once
// AlphaLlama: Hybrid prompt type classifier
// 1. Fast keyword scan (0ms) - catches obvious domain matches
// 2. 0.8B model fallback (~130ms) - semantic understanding for ambiguous prompts
//
// Usage:
//   alpha_classifier * cls = alpha_classifier_init("/path/to/0.8B.gguf");
//   const char * domain = alpha_classifier_detect(cls, prompt, prompt_len);
//   // domain = "chemical", "math", "code", "general", etc.
//   alpha_classifier_free(cls);

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct alpha_classifier;

// Initialize classifier with a small model (0.6B/0.8B GGUF)
// model_path can be NULL to use keyword-only mode (no model fallback)
LLAMA_API struct alpha_classifier * alpha_classifier_init(const char * model_path);

// Classify prompt into output domain
// Returns domain name (static string, do not free): "chemical", "math", "code", etc.
// Falls back to "general" if no domain detected
LLAMA_API const char * alpha_classifier_detect(
    struct alpha_classifier * cls,
    const char * prompt,
    int prompt_len);

// Free classifier resources
LLAMA_API void alpha_classifier_free(struct alpha_classifier * cls);

#ifdef __cplusplus
}
#endif
