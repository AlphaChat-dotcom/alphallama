#pragma once

// AlphaServe multimodal prompts (image input through an mtmd projector, --mmproj).
//
// A multimodal prompt travels with its request as the mtmd chunks the HTTP thread tokenized (text runs + encoded-on-
// demand image runs). The scheduler still sees a token vector of the same length -- placeholders for the image runs --
// so its length, budget and slot bookkeeping are unchanged; the engine ignores those tokens and evaluates the chunks.
// Images take n_pos positions, which on M-RoPE / IMRoPE models is fewer than their n_tokens, so the slot's next
// position after the prompt is n_pos, not the token count.

#include <memory>

struct mtmd_context;
struct mtmd_input_chunks;

namespace alphaserve {

struct mm_prompt {
    mtmd_input_chunks * chunks = nullptr;   // owned
    int n_tokens = 0;                       // KV cells the prompt fills (text tokens + image tokens)
    int n_pos    = 0;                       // positions it consumes (images: n_pos <= n_tokens)
    int n_images = 0;
    ~mm_prompt();
};

// the projector, set by serve-main when --mmproj loaded (null = no image input); used by the engine thread only after
extern mtmd_context * g_serve_mtmd;

// placeholder id for image positions in the scheduler's token vector (never decoded)
constexpr int MM_PLACEHOLDER_TOKEN = 0;

} // namespace alphaserve
