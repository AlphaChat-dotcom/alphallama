#include "alpha_mole_autoconfig.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif

// Get system RAM (cross-platform)
static void get_system_ram(size_t * total_mb, size_t * free_mb) {
    *total_mb = 0;
    *free_mb  = 0;

#if defined(_WIN32)
    // Windows
    MEMORYSTATUSEX mem = {};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        *total_mb = (size_t)(mem.ullTotalPhys / (1024 * 1024));
        *free_mb  = (size_t)(mem.ullAvailPhys / (1024 * 1024));
    }
#elif defined(__linux__)
    // Linux: read /proc/meminfo
    FILE * f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            unsigned long val = 0;
            if (sscanf(line, "MemTotal: %lu kB", &val) == 1) {
                *total_mb = val / 1024;
            } else if (sscanf(line, "MemAvailable: %lu kB", &val) == 1) {
                *free_mb = val / 1024;
            }
        }
        fclose(f);
    }
#elif defined(__APPLE__)
    // macOS
    int64_t memsize = 0;
    size_t len = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &len, NULL, 0);
    *total_mb = (size_t)(memsize / (1024 * 1024));

    vm_size_t page_size;
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    host_page_size(mach_host_self(), &page_size);
    host_statistics64(mach_host_self(), HOST_VM_INFO, (host_info64_t)&vm_stats, &count);
    *free_mb = (size_t)((vm_stats.free_count + vm_stats.inactive_count) * page_size / (1024 * 1024));
#endif
}

void alpha_autoconfig_detect_hardware(alpha_autoconfig * cfg) {
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));

    // Detect GPU via ggml backends
    size_t n_dev = ggml_backend_dev_count();

    // Pick GPU with the most free VRAM (not just the first one)
    size_t best_free = 0;
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);

        if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);

            if (free_bytes > best_free) {
                best_free = free_bytes;
                cfg->gpu_name     = ggml_backend_dev_description(dev);
                cfg->gpu_type     = (dtype == GGML_BACKEND_DEVICE_TYPE_GPU) ? 1 : 2;
                cfg->vram_total_mb = total_bytes / (1024 * 1024);
                cfg->vram_free_mb  = free_bytes  / (1024 * 1024);
            }
        }
    }

    // Detect system RAM
    get_system_ram(&cfg->ram_total_mb, &cfg->ram_free_mb);

    // Defaults
    cfg->cache_type_k = GGML_TYPE_F16;
    cfg->cache_type_v = GGML_TYPE_F16;
}

void alpha_autoconfig_for_model(
        alpha_autoconfig * cfg,
        const struct llama_model * model,
        int n_loras,
        int lora_rank) {
    if (!cfg || !model) return;

    cfg->model_size_mb = llama_model_size(model) / (1024 * 1024);
    cfg->n_layer       = llama_model_n_layer(model);
    cfg->n_embd        = llama_model_n_embd(model);

    // Estimate LoRA memory: n_loras * n_layers * 2 targets * 2 matrices * (n_embd * rank) * 4 bytes
    size_t lora_mb = 0;
    if (n_loras > 0 && lora_rank > 0) {
        lora_mb = (size_t)n_loras * cfg->n_layer * 2 * 2 * cfg->n_embd * lora_rank * 4 / (1024 * 1024);
    }

    // Estimate per-layer size
    size_t layer_size_mb = cfg->model_size_mb / std::max(cfg->n_layer, 1);

    // Available VRAM for model (leave 512MB for overhead)
    size_t vram_budget_mb = 0;
    if (cfg->vram_free_mb > 512) {
        vram_budget_mb = cfg->vram_free_mb - 512 - lora_mb;
    }

    if (cfg->gpu_type == 0) {
        // No GPU — CPU only
        cfg->n_gpu_layers = 0;
        cfg->use_pipeline = false;

        // Use all available RAM for context
        size_t ram_for_kv = cfg->ram_free_mb > cfg->model_size_mb + 2048
            ? cfg->ram_free_mb - cfg->model_size_mb - 2048
            : 1024;

        // KV per token: 2 * n_layer * n_embd * 2 bytes (FP16 K+V)
        size_t kv_per_token = 2 * cfg->n_layer * cfg->n_embd * 2;
        cfg->n_ctx = (int)std::min((size_t)(ram_for_kv * 1024 * 1024 / kv_per_token), (size_t)32768);
        cfg->est_tok_per_sec = 2.0f;  // rough CPU estimate
    } else if (cfg->model_size_mb <= vram_budget_mb) {
        // Model fits entirely on GPU
        cfg->n_gpu_layers = cfg->n_layer + 1;  // all layers + embeddings
        cfg->use_pipeline = false;

        size_t vram_for_kv = vram_budget_mb - cfg->model_size_mb;

        // KV per token (FP16): 2 * n_layer * n_embd * 2
        size_t kv_per_token_fp16 = 2 * cfg->n_layer * cfg->n_embd * 2;
        size_t ctx_fp16 = vram_for_kv * 1024 * 1024 / kv_per_token_fp16;

        if (ctx_fp16 < 8192) {
            // Not enough for decent context with FP16 — use Q4_0 KV (4x more)
            cfg->cache_type_k = GGML_TYPE_Q4_0;
            cfg->cache_type_v = GGML_TYPE_Q4_0;
            size_t kv_per_token_q4 = 2 * cfg->n_layer * cfg->n_embd / 2;  // ~4x smaller
            cfg->n_ctx = (int)std::min(vram_for_kv * 1024 * 1024 / kv_per_token_q4, (size_t)524288);
        } else if (ctx_fp16 < 32768) {
            // Use Q8_0 for moderate improvement
            cfg->cache_type_k = GGML_TYPE_Q8_0;
            cfg->cache_type_v = GGML_TYPE_Q8_0;
            cfg->n_ctx = (int)std::min(ctx_fp16 * 2, (size_t)262144);
        } else {
            cfg->n_ctx = (int)std::min(ctx_fp16, (size_t)131072);
        }

        cfg->est_tok_per_sec = 50.0f + (float)cfg->vram_total_mb / 200.0f;
    } else {
        // Model doesn't fit — partial offload or pipeline
        size_t layers_that_fit = vram_budget_mb / std::max(layer_size_mb, (size_t)1);
        cfg->n_gpu_layers = (int)std::min(layers_that_fit, (size_t)(cfg->n_layer + 1));

        if (cfg->n_gpu_layers < cfg->n_layer / 2) {
            // Less than half fits — recommend pipeline mode
            cfg->use_pipeline = true;
            cfg->n_gpu_layers = 8;  // hot layers only, rest streamed
        }

        // KV in remaining VRAM + CPU RAM
        size_t vram_for_kv = vram_budget_mb > (size_t)cfg->n_gpu_layers * layer_size_mb
            ? vram_budget_mb - (size_t)cfg->n_gpu_layers * layer_size_mb
            : 0;

        cfg->cache_type_k = GGML_TYPE_Q4_0;
        cfg->cache_type_v = GGML_TYPE_Q4_0;
        size_t kv_per_token_q4 = 2 * cfg->n_layer * cfg->n_embd / 2;
        size_t ctx_gpu = vram_for_kv * 1024 * 1024 / std::max(kv_per_token_q4, (size_t)1);

        // If system RAM allows, recommend KV paging for more context
        if (cfg->ram_free_mb > cfg->model_size_mb + 8192) {
            cfg->use_kv_paging = true;
            size_t ram_for_kv = cfg->ram_free_mb - cfg->model_size_mb - 4096;
            size_t ctx_total = ctx_gpu + ram_for_kv * 1024 * 1024 / std::max(kv_per_token_q4, (size_t)1);
            cfg->n_ctx = (int)std::min(ctx_total, (size_t)1048576);
        } else {
            cfg->n_ctx = (int)std::min(ctx_gpu, (size_t)131072);
        }

        // Speed depends on how many layers are offloaded
        float gpu_fraction = (float)cfg->n_gpu_layers / cfg->n_layer;
        cfg->est_tok_per_sec = cfg->use_pipeline
            ? 13.0f * (16384.0f / cfg->model_size_mb)  // pipeline-bound
            : 5.0f + gpu_fraction * 50.0f;               // partial offload
    }

    // Clamp context
    if (cfg->n_ctx < 512)    cfg->n_ctx = 512;
    if (cfg->n_ctx > 1048576) cfg->n_ctx = 1048576;
}

static size_t get_file_size(const char * path) {
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA fad;
    if (GetFileAttributesExA(path, GetFileExInfoStandard, &fad)) {
        return ((size_t)fad.nFileSizeHigh << 32) | fad.nFileSizeLow;
    }
    return 0;
#else
    struct stat st;
    if (stat(path, &st) == 0) {
        return (size_t)st.st_size;
    }
    return 0;
#endif
}

void alpha_autoconfig_from_gguf(
        const char       * model_path,
        alpha_autoconfig * cfg,
        int                n_loras,
        int                lora_rank) {
    if (!model_path || !cfg) return;

    // Step 1: Detect hardware (if not already done)
    if (cfg->vram_total_mb == 0) {
        alpha_autoconfig_detect_hardware(cfg);
    }

    // Step 2: Read GGUF metadata without allocating tensors
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "alpha_autoconfig: cannot read GGUF metadata from '%s'\n", model_path);
        return;
    }

    // Read architecture
    std::string arch = "llama";
    int arch_idx = gguf_find_key(gguf_ctx, "general.architecture");
    if (arch_idx >= 0) {
        arch = gguf_get_val_str(gguf_ctx, arch_idx);
    }

    // Read block count (n_layers)
    std::string block_key = arch + ".block_count";
    int block_idx = gguf_find_key(gguf_ctx, block_key.c_str());
    if (block_idx >= 0) {
        cfg->n_layer = (int)gguf_get_val_u32(gguf_ctx, block_idx);
    }

    // Read embedding length
    std::string embd_key = arch + ".embedding_length";
    int embd_idx = gguf_find_key(gguf_ctx, embd_key.c_str());
    if (embd_idx >= 0) {
        cfg->n_embd = (int)gguf_get_val_u32(gguf_ctx, embd_idx);
    }

    gguf_free(gguf_ctx);

    // Step 3: Get model file size
    cfg->model_size_mb = get_file_size(model_path) / (1024 * 1024);

    if (cfg->n_layer <= 0 || cfg->model_size_mb == 0) {
        fprintf(stderr, "alpha_autoconfig: could not determine model size from '%s'\n", model_path);
        return;
    }

    // Step 4: Calculate optimal ngl
    size_t layer_size_mb = cfg->model_size_mb / cfg->n_layer;

    // LoRA memory estimate
    size_t lora_mb = 0;
    if (n_loras > 0 && lora_rank > 0 && cfg->n_embd > 0) {
        lora_mb = (size_t)n_loras * cfg->n_layer * 2 * 2 * cfg->n_embd * lora_rank * 4 / (1024 * 1024);
    }

    // VRAM budget: free - LoRAs - KV reserve - workspace overhead
    // Reserve generously for KV cache + CUDA workspace + graph buffers
    // For 27B with n_embd=5120, 64 layers: KV needs ~2-3GB at reasonable context
    size_t overhead_mb = 1024 + lora_mb;  // CUDA workspace + graph buffers
    size_t kv_reserve_mb = std::max((size_t)2048, (size_t)(cfg->n_layer * cfg->n_embd / 512));  // scale with model
    size_t vram_for_model = 0;
    if (cfg->vram_free_mb > overhead_mb + kv_reserve_mb) {
        vram_for_model = cfg->vram_free_mb - overhead_mb - kv_reserve_mb;
    }

    // How many layers fit?
    int layers_fit = layer_size_mb > 0 ? (int)(vram_for_model / layer_size_mb) : 0;
    cfg->n_gpu_layers = std::min(layers_fit, cfg->n_layer + 1);

    // Determine KV cache type based on remaining VRAM
    size_t vram_after_model = 0;
    if (vram_for_model > (size_t)cfg->n_gpu_layers * layer_size_mb) {
        vram_after_model = vram_for_model - (size_t)cfg->n_gpu_layers * layer_size_mb;
    }
    if (vram_after_model < 512) {
        cfg->cache_type_k = GGML_TYPE_Q4_0;
        cfg->cache_type_v = GGML_TYPE_Q4_0;
    } else if (vram_after_model < 2048) {
        cfg->cache_type_k = GGML_TYPE_Q8_0;
        cfg->cache_type_v = GGML_TYPE_Q8_0;
    }

    // Determine if pipeline/streaming is needed
    cfg->use_pipeline = cfg->n_gpu_layers < cfg->n_layer;

    fprintf(stderr, "alpha_autoconfig: %s — %llu MB, %d layers (%zu MB/layer)\n",
            model_path, (unsigned long long)cfg->model_size_mb, cfg->n_layer, layer_size_mb);
    fprintf(stderr, "alpha_autoconfig: VRAM %zu MB free, budget %zu MB for model → ngl=%d/%d\n",
            cfg->vram_free_mb, vram_for_model, cfg->n_gpu_layers, cfg->n_layer);
}

void alpha_autoconfig_print(const alpha_autoconfig * cfg) {
    if (!cfg) return;

    fprintf(stderr, "\n=== Alpha-MoLE Auto Configuration ===\n\n");

    // Hardware
    if (cfg->gpu_type == 1) {
        fprintf(stderr, "  GPU:     %s (%zu MB VRAM, %zu MB free)\n",
                cfg->gpu_name ? cfg->gpu_name : "Unknown",
                cfg->vram_total_mb, cfg->vram_free_mb);
    } else if (cfg->gpu_type == 2) {
        fprintf(stderr, "  iGPU:    %s (shared memory)\n",
                cfg->gpu_name ? cfg->gpu_name : "Unknown");
    } else {
        fprintf(stderr, "  GPU:     None detected (CPU-only mode)\n");
    }
    fprintf(stderr, "  RAM:     %zu MB total, %zu MB free\n", cfg->ram_total_mb, cfg->ram_free_mb);

    // Model
    if (cfg->model_size_mb > 0) {
        fprintf(stderr, "  Model:   %llu MB, %d layers, %d embd\n",
                (unsigned long long)cfg->model_size_mb, cfg->n_layer, cfg->n_embd);
    }

    // Config
    fprintf(stderr, "\n  Recommended:\n");
    fprintf(stderr, "    -ngl %d", cfg->n_gpu_layers);
    if (cfg->n_gpu_layers >= cfg->n_layer) {
        fprintf(stderr, "  (all layers on GPU)\n");
    } else {
        fprintf(stderr, "  (%d of %d layers on GPU)\n", cfg->n_gpu_layers, cfg->n_layer);
    }
    fprintf(stderr, "    -c %d\n", cfg->n_ctx);

    const char * kname = cfg->cache_type_k == GGML_TYPE_Q4_0 ? "q4_0" :
                         cfg->cache_type_k == GGML_TYPE_Q8_0 ? "q8_0" : "f16";
    const char * vname = cfg->cache_type_v == GGML_TYPE_Q4_0 ? "q4_0" :
                         cfg->cache_type_v == GGML_TYPE_Q8_0 ? "q8_0" : "f16";
    fprintf(stderr, "    --cache-type-k %s --cache-type-v %s\n", kname, vname);

    if (cfg->use_pipeline) {
        fprintf(stderr, "    --pipeline  (model too large for VRAM, stream from CPU)\n");
    }
    if (cfg->use_kv_paging) {
        fprintf(stderr, "    --kv-paging  (extend context with CPU RAM)\n");
    }

    fprintf(stderr, "\n  Est. speed: ~%.0f tok/s\n", cfg->est_tok_per_sec);
    fprintf(stderr, "\n");
}
