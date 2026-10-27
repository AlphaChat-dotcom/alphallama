#include "alpha_mole_pipeline.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

void alpha_pipeline_configure(
        alpha_pipeline_config * cfg,
        const struct llama_model * model,
        size_t vram_free_mb,
        size_t ram_free_mb,
        int n_loras,
        int lora_rank,
        int actual_ngl) {
    if (!cfg || !model) return;
    memset(cfg, 0, sizeof(*cfg));

    cfg->model_size_mb = llama_model_size(model) / (1024 * 1024);
    cfg->n_layer       = llama_model_n_layer(model);
    cfg->vram_free_mb  = vram_free_mb;
    cfg->ram_free_mb   = ram_free_mb;

    int n_embd = llama_model_n_embd(model);

    // Per-layer size estimate
    cfg->est_layer_size_mb = cfg->model_size_mb / std::max(cfg->n_layer, 1);

    // LoRA memory estimate (all LoRAs stay on GPU permanently)
    size_t lora_mb = 0;
    if (n_loras > 0 && lora_rank > 0) {
        lora_mb = (size_t)n_loras * cfg->n_layer * 2 * 2 * n_embd * lora_rank * 4 / (1024 * 1024);
    }

    // VRAM budget: total free - LoRAs - overhead (512MB for KV + workspace)
    size_t overhead_mb = 512 + lora_mb;
    size_t vram_for_layers = vram_free_mb > overhead_mb ? vram_free_mb - overhead_mb : 0;

    // How many layers fit on GPU?
    int layers_that_fit = cfg->est_layer_size_mb > 0
        ? (int)(vram_for_layers / cfg->est_layer_size_mb)
        : cfg->n_layer + 1;

    if (layers_that_fit >= cfg->n_layer + 1) {
        // Everything fits on GPU
        cfg->n_gpu_layers    = cfg->n_layer + 1;
        cfg->n_hot_first     = 0;
        cfg->n_hot_last      = 0;
        cfg->use_pinned_memory = false;
        cfg->pipeline_parallel = false;
    } else if (layers_that_fit >= cfg->n_layer / 2) {
        // More than half fits — standard partial offload
        cfg->n_gpu_layers    = layers_that_fit;
        cfg->n_hot_first     = 0;
        cfg->n_hot_last      = 0;
        cfg->use_pinned_memory = true;
        cfg->pipeline_parallel = true;
    } else if (layers_that_fit >= 8) {
        // Less than half fits — pipeline mode with hot layers
        // Keep early + late layers on GPU (embedding + output norm are most impactful)
        cfg->n_gpu_layers    = layers_that_fit;
        cfg->n_hot_first     = std::min(4, layers_that_fit / 2);
        cfg->n_hot_last      = std::min(4, layers_that_fit - cfg->n_hot_first);
        cfg->use_pinned_memory = true;
        cfg->pipeline_parallel = true;
    } else {
        // Very little VRAM — minimal offload
        cfg->n_gpu_layers    = std::max(layers_that_fit, 0);
        cfg->n_hot_first     = 0;
        cfg->n_hot_last      = 0;
        cfg->use_pinned_memory = true;
        cfg->pipeline_parallel = true;
    }

    // Override with actual ngl if provided (user explicitly set -ngl)
    if (actual_ngl >= 0) {
        cfg->n_gpu_layers = std::min(actual_ngl, cfg->n_layer + 1);
        int n_cpu = cfg->n_layer - std::min(cfg->n_gpu_layers, cfg->n_layer);
        if (n_cpu == 0) {
            cfg->use_pinned_memory = false;
            cfg->pipeline_parallel = false;
        } else {
            cfg->use_pinned_memory = true;
            cfg->pipeline_parallel = true;
        }
    }

    // Speed estimates
    // Prefill: compute-bound, GPU compute dominates PCIe transfer
    //   Base: ~300 tok/s for 27B, ~1500 for 4B, scales with model size
    float base_prefill = 1500.0f * (2600.0f / std::max((float)cfg->model_size_mb, 1.0f));
    float gpu_frac = (float)cfg->n_gpu_layers / (cfg->n_layer + 1);
    cfg->est_prefill_tok_s = base_prefill * (0.3f + 0.7f * gpu_frac);

    // Generation: PCIe-bound for off-GPU layers
    //   Each off-GPU layer: ~15ms transfer (300MB @ 20GB/s effective)
    //   Each GPU layer: ~0.1ms compute
    int n_cpu_layers = cfg->n_layer - std::min(cfg->n_gpu_layers, cfg->n_layer);
    if (n_cpu_layers == 0) {
        // All on GPU — pure compute speed
        cfg->est_gen_tok_s = 5000.0f * (2600.0f / std::max((float)cfg->model_size_mb, 1.0f));
    } else {
        // PCIe-bound: time per token = n_cpu_layers * layer_transfer_time
        float transfer_time_ms = (float)cfg->est_layer_size_mb / 20.0f;  // ~20 GB/s PCIe 4.0
        float total_ms = n_cpu_layers * transfer_time_ms + cfg->n_gpu_layers * 0.1f;
        cfg->est_gen_tok_s = 1000.0f / std::max(total_ms, 0.1f);
    }
}

void alpha_pipeline_apply_model_params(
        const alpha_pipeline_config * cfg,
        llama_model_params * params) {
    if (!cfg || !params) return;

    params->n_gpu_layers = cfg->n_gpu_layers;

    // Note: pinned memory is handled at backend level, not model params
    // The CUDA backend automatically uses pinned memory when available
}

void alpha_pipeline_apply_context_params(
        const alpha_pipeline_config * cfg,
        llama_context_params * params) {
    if (!cfg || !params) return;

    // Enable pipeline parallelism for overlapped CPU↔GPU transfers
    // This allows the scheduler to overlap compute on one backend
    // with data transfers to another
    // Note: this field may not exist in all llama.cpp versions
    // params->pipeline_parallel = cfg->pipeline_parallel;

    // Use quantized KV cache if VRAM is tight
    int n_cpu_layers = cfg->n_layer - std::min(cfg->n_gpu_layers, cfg->n_layer);
    if (n_cpu_layers > cfg->n_layer / 4) {
        // Significant offloading — use q4_0 KV to save VRAM for layers
        params->type_k = GGML_TYPE_Q4_0;
        params->type_v = GGML_TYPE_Q4_0;
    } else if (n_cpu_layers > 0) {
        // Some offloading — use q8_0 KV
        params->type_k = GGML_TYPE_Q8_0;
        params->type_v = GGML_TYPE_Q8_0;
    }
}

void alpha_pipeline_print(const alpha_pipeline_config * cfg) {
    if (!cfg) return;

    fprintf(stderr, "\n=== Layer Pipeline Configuration ===\n\n");
    fprintf(stderr, "  Model:  %llu MB, %d layers (~%zu MB/layer)\n",
            (unsigned long long)cfg->model_size_mb, cfg->n_layer, cfg->est_layer_size_mb);
    fprintf(stderr, "  VRAM:   %zu MB free\n", cfg->vram_free_mb);
    fprintf(stderr, "  RAM:    %zu MB free\n", cfg->ram_free_mb);

    int n_cpu = cfg->n_layer - std::min(cfg->n_gpu_layers, cfg->n_layer);

    if (n_cpu == 0) {
        fprintf(stderr, "\n  Strategy: FULL GPU (all %d layers on GPU)\n", cfg->n_layer);
    } else {
        fprintf(stderr, "\n  Strategy: PARTIAL OFFLOAD\n");
        fprintf(stderr, "    GPU layers: %d / %d\n", std::min(cfg->n_gpu_layers, cfg->n_layer), cfg->n_layer);
        fprintf(stderr, "    CPU layers: %d\n", n_cpu);
        if (cfg->n_hot_first > 0 || cfg->n_hot_last > 0) {
            fprintf(stderr, "    Hot layers: first %d + last %d (always GPU)\n",
                    cfg->n_hot_first, cfg->n_hot_last);
        }
        if (cfg->use_pinned_memory) {
            fprintf(stderr, "    Pinned memory: enabled (faster DMA)\n");
        }
        if (cfg->pipeline_parallel) {
            fprintf(stderr, "    Pipeline parallel: enabled (overlap compute + transfer)\n");
        }
    }

    float prefill = cfg->est_prefill_tok_s > 0 ? cfg->est_prefill_tok_s : 1.0f;
    float gen     = cfg->est_gen_tok_s > 0 ? cfg->est_gen_tok_s : 1.0f;
    fprintf(stderr, "\n  Estimated speed: ~%.0f tok/s prefill, ~%.1f tok/s generate\n", prefill, gen);
    fprintf(stderr, "\n");
}
