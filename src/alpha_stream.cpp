#include "alpha_stream.h"

#include <cstdio>
#include <algorithm>

void alpha_stream_configure(
        alpha_stream_config * cfg,
        size_t model_size_mb,
        int    n_layer,
        int    n_embd,
        size_t vram_free_mb,
        int    n_loras,
        int    lora_rank) {
    if (!cfg) return;

    cfg->model_size_mb = model_size_mb;
    cfg->n_layer       = n_layer;
    cfg->vram_free_mb  = vram_free_mb;
    cfg->cache_type_k  = GGML_TYPE_Q4_0;
    cfg->cache_type_v  = GGML_TYPE_Q4_0;

    size_t layer_size_mb = model_size_mb / std::max(n_layer, 1);

    // LoRA memory
    size_t lora_mb = 0;
    if (n_loras > 0 && lora_rank > 0 && n_embd > 0) {
        lora_mb = (size_t)n_loras * n_layer * 2 * 2 * n_embd * lora_rank * 4 / (1024 * 1024);
    }

    // Fixed overhead: CUDA workspace + graph buffers
    size_t fixed_overhead_mb = 768 + lora_mb;

    // Available for model + KV
    size_t available_mb = vram_free_mb > fixed_overhead_mb
        ? vram_free_mb - fixed_overhead_mb
        : 0;

    // Strategy: maximize ngl, use remaining VRAM for KV
    // With Q4_0 KV, each token uses: n_layer * n_embd * 2 (K+V) / 4 (Q4 compression) bytes
    // Minimum usable context: 256 tokens
    size_t min_kv_mb = (size_t)n_layer * n_embd * 2 * 256 / 4 / (1024 * 1024);
    min_kv_mb = std::max(min_kv_mb, (size_t)128);

    // Max model budget = available - min KV
    size_t max_model_mb = available_mb > min_kv_mb ? available_mb - min_kv_mb : 0;

    // How many layers fit?
    int max_layers = layer_size_mb > 0 ? (int)(max_model_mb / layer_size_mb) : 0;
    cfg->n_gpu_layers = std::min(max_layers, n_layer + 1);

    // Check if model fits fully
    cfg->fits_fully = (cfg->n_gpu_layers >= n_layer);

    // Calculate context from remaining VRAM
    size_t model_on_gpu_mb = (size_t)std::min(cfg->n_gpu_layers, n_layer) * layer_size_mb;
    size_t vram_for_kv = available_mb > model_on_gpu_mb ? available_mb - model_on_gpu_mb : min_kv_mb;

    // KV per token with Q4_0: n_layer * n_embd * 2 / 4 bytes
    size_t kv_per_token = (size_t)n_layer * n_embd * 2 / 4;
    if (kv_per_token > 0) {
        cfg->n_ctx = (int)std::min(vram_for_kv * 1024 * 1024 / kv_per_token, (size_t)131072);
    } else {
        cfg->n_ctx = 4096;
    }
    cfg->n_ctx = std::max(cfg->n_ctx, 256);
}

void alpha_stream_print(const alpha_stream_config * cfg) {
    if (!cfg) return;

    fprintf(stderr, "\n=== Stream Configuration ===\n");
    fprintf(stderr, "  Model:  %zu MB, %d layers\n", cfg->model_size_mb, cfg->n_layer);
    fprintf(stderr, "  VRAM:   %zu MB free\n", cfg->vram_free_mb);

    if (cfg->fits_fully) {
        fprintf(stderr, "  Mode:   ALL GPU (%d layers, Q4_0 KV, ctx=%d)\n",
                cfg->n_gpu_layers, cfg->n_ctx);
        fprintf(stderr, "  Gain:   Maximum speed — all layers on GPU\n");
    } else {
        int n_cpu = cfg->n_layer - std::min(cfg->n_gpu_layers, cfg->n_layer);
        fprintf(stderr, "  Mode:   PARTIAL (%d GPU + %d CPU, Q4_0 KV, ctx=%d)\n",
                std::min(cfg->n_gpu_layers, cfg->n_layer), n_cpu, cfg->n_ctx);
        fprintf(stderr, "  Note:   Model too large for full GPU — %d layers on CPU\n", n_cpu);
    }
    fprintf(stderr, "\n");
}
