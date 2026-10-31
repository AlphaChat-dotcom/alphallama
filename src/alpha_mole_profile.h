#pragma once

// AlphaLlama: Expert Profile System
//
// Manages expert activation profiles for MoE models. Profiles contain
// per-layer hot expert lists, enabling:
//   1. Warm cache startup (pre-load hot experts before first token)
//   2. Expert prediction (which experts to cache for next token)
//   3. Cache size optimization (how many slots needed for target hit rate)
//
// Profiles are generated from calibration data (offline) and stored
// as local profile files.
//
// Generic: works for any MoE model, any expert count, any top-K.

#include "llama.h"

#include <cstdint>

struct alpha_expert_profile;

struct alpha_expert_profile_params {
    int    n_layers;
    int    n_experts;
    int    n_expert_used;     // top-K
    int    n_hot;             // how many experts to mark as "hot" per layer
};

// Load profile from file. Returns NULL if file doesn't exist.
alpha_expert_profile * alpha_expert_profile_load(const char * path);

// Generate profile by running calibration tokens through the model.
// Records which experts activate per layer across n_tokens.
alpha_expert_profile * alpha_expert_profile_generate(
        struct llama_model   * model,
        struct llama_context * ctx,
        const char * calibration_text,
        int n_tokens);

// Save profile to file.
bool alpha_expert_profile_save(const alpha_expert_profile * profile, const char * path);

// Get hot experts for a layer (sorted by frequency, most active first).
// Returns n_hot expert IDs.
const int * alpha_expert_profile_get_hot(const alpha_expert_profile * profile, int layer);

// Get estimated cache hit rate for a given cache size K.
float alpha_expert_profile_hit_rate(const alpha_expert_profile * profile, int K);

// Get model hash (for verifying profile matches model).
const char * alpha_expert_profile_model_hash(const alpha_expert_profile * profile);

// Free profile.
void alpha_expert_profile_free(alpha_expert_profile * profile);
