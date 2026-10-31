#pragma once

// AlphaLlama: Mixture of LoRA Experts
// Per-token LoRA routing for llama.cpp

#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ALPHA_LORA_MIX_MAX_TOP_K 4
#define ALPHA_LORA_MIX_MAX_LORAS 64

// Strategy auto-selected based on what's loaded
enum alpha_lora_mix_strategy {
    LORA_MIX_STRATEGY_RAW,            // no LoRAs — maximum speed, pure base model
    LORA_MIX_STRATEGY_SINGLE_LORA,    // 1 LoRA — static application, no routing
    LORA_MIX_STRATEGY_STATIC_BLEND,   // N LoRAs, no router — fixed weights
    LORA_MIX_STRATEGY_DYNAMIC,        // N LoRAs + router — per-token routing
};

struct alpha_lora_mix_context;

struct alpha_lora_mix_params {
    int                          n_loras;                           // number of LoRA adapters
    const char                ** lora_paths;                        // paths to LoRA GGUF files
    const char                 * router_path;                       // path to router.gguf (NULL = no routing)
    int                          top_k;                             // how many LoRAs per token (default: 2)
    bool                         enable_routing;                    // false = static merge mode
    float                      * static_weights;                    // if enable_routing=false, fixed weights per LoRA (NULL = equal)
};

// Per-token routing decision (for inspection/debugging)
struct alpha_lora_mix_token_route {
    int   lora_indices[ALPHA_LORA_MIX_MAX_TOP_K];
    float lora_weights[ALPHA_LORA_MIX_MAX_TOP_K];
    int   n_selected;                                               // actual number of LoRAs selected
};

struct alpha_lora_mix_routing_info {
    int                          n_tokens;
    struct alpha_lora_mix_token_route * routes;                         // array of n_tokens entries
};

// Initialize LoRA-mixture context with base model context + LoRA adapters + optional router
// Returns NULL on failure
LLAMA_API struct alpha_lora_mix_context * alpha_lora_mix_init(
        struct llama_model   * model,
        struct llama_context * ctx,
        struct alpha_lora_mix_params params);

// Decode a batch with LoRA-mixture routing
// In dynamic routing mode: routes each token to top-K LoRAs
// In static mode: applies fixed-weight LoRA blend
// Returns 0 on success, negative on error
LLAMA_API int alpha_lora_mix_decode(
        struct alpha_lora_mix_context * mctx,
        struct llama_batch          batch);

// Get routing decisions from the last alpha_lora_mix_decode() call
// Caller must free info->routes when done
// Returns 0 on success, negative on error
LLAMA_API int alpha_lora_mix_get_routing(
        struct alpha_lora_mix_context        * mctx,
        struct alpha_lora_mix_routing_info   * info);

// Free routing info routes array
LLAMA_API void alpha_lora_mix_routing_info_free(struct alpha_lora_mix_routing_info * info);

// Get the active strategy (auto-detected based on loaded LoRAs/router)
LLAMA_API enum alpha_lora_mix_strategy alpha_lora_mix_get_strategy(
        const struct alpha_lora_mix_context * mctx);

// Get strategy name as string
LLAMA_API const char * alpha_lora_mix_strategy_name(enum alpha_lora_mix_strategy s);

// Free LoRA-mixture context (does not free the underlying llama_context or model)
LLAMA_API void alpha_lora_mix_free(struct alpha_lora_mix_context * mctx);

#ifdef __cplusplus
}
#endif
