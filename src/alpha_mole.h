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

#define ALPHA_MOLE_MAX_TOP_K 4
#define ALPHA_MOLE_MAX_LORAS 64

// Strategy auto-selected based on what's loaded
enum alpha_mole_strategy {
    MOLE_STRATEGY_RAW,            // no LoRAs — maximum speed, pure base model
    MOLE_STRATEGY_SINGLE_LORA,    // 1 LoRA — static application, no routing
    MOLE_STRATEGY_STATIC_BLEND,   // N LoRAs, no router — fixed weights
    MOLE_STRATEGY_DYNAMIC,        // N LoRAs + router — per-token routing
};

struct alpha_mole_context;

struct alpha_mole_params {
    int                          n_loras;                           // number of LoRA adapters
    const char                ** lora_paths;                        // paths to LoRA GGUF files
    const char                 * router_path;                       // path to router.gguf (NULL = no routing)
    int                          top_k;                             // how many LoRAs per token (default: 2)
    bool                         enable_routing;                    // false = static merge mode
    float                      * static_weights;                    // if enable_routing=false, fixed weights per LoRA (NULL = equal)
};

// Per-token routing decision (for inspection/debugging)
struct alpha_mole_token_route {
    int   lora_indices[ALPHA_MOLE_MAX_TOP_K];
    float lora_weights[ALPHA_MOLE_MAX_TOP_K];
    int   n_selected;                                               // actual number of LoRAs selected
};

struct alpha_mole_routing_info {
    int                          n_tokens;
    struct alpha_mole_token_route * routes;                         // array of n_tokens entries
};

// Initialize LoRA-mixture context with base model context + LoRA adapters + optional router
// Returns NULL on failure
LLAMA_API struct alpha_mole_context * alpha_mole_init(
        struct llama_model   * model,
        struct llama_context * ctx,
        struct alpha_mole_params params);

// Decode a batch with LoRA-mixture routing
// In dynamic routing mode: routes each token to top-K LoRAs
// In static mode: applies fixed-weight LoRA blend
// Returns 0 on success, negative on error
LLAMA_API int alpha_mole_decode(
        struct alpha_mole_context * mctx,
        struct llama_batch          batch);

// Get routing decisions from the last alpha_mole_decode() call
// Caller must free info->routes when done
// Returns 0 on success, negative on error
LLAMA_API int alpha_mole_get_routing(
        struct alpha_mole_context        * mctx,
        struct alpha_mole_routing_info   * info);

// Free routing info routes array
LLAMA_API void alpha_mole_routing_info_free(struct alpha_mole_routing_info * info);

// Get the active strategy (auto-detected based on loaded LoRAs/router)
LLAMA_API enum alpha_mole_strategy alpha_mole_get_strategy(
        const struct alpha_mole_context * mctx);

// Get strategy name as string
LLAMA_API const char * alpha_mole_strategy_name(enum alpha_mole_strategy s);

// Free LoRA-mixture context (does not free the underlying llama_context or model)
LLAMA_API void alpha_mole_free(struct alpha_mole_context * mctx);

#ifdef __cplusplus
}
#endif
