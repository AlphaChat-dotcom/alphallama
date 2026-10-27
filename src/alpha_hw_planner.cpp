#include "alpha_hw_planner.h"
#include "alpha_mole_moe_optimizer.h"  // for ALPHA_LOG_DBG
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <sys/statvfs.h>
#endif

// Get free disk space on the partition containing the given path (MB)
static size_t get_disk_free_mb(const char * path) {
#ifdef _WIN32
    ULARGE_INTEGER free_bytes = {};
    char dir[MAX_PATH];
    strncpy(dir, path, MAX_PATH - 1);
    // Strip filename to get directory
    char * last_sep = strrchr(dir, '\\');
    if (!last_sep) last_sep = strrchr(dir, '/');
    if (last_sep) *last_sep = '\0';
    if (GetDiskFreeSpaceExA(dir, &free_bytes, nullptr, nullptr)) {
        return (size_t)(free_bytes.QuadPart / (1024 * 1024));
    }
    return 0;
#else
    struct statvfs st;
    if (statvfs(path, &st) == 0) {
        return (size_t)((uint64_t)st.f_bavail * st.f_frsize / (1024 * 1024));
    }
    return 0;
#endif
}

// Bandwidth Estimation

static size_t estimate_local_bw(const alpha_device_info & dev) {
    if (dev.mem_type == ALPHA_MEM_SYSTEM) return 40000;  // DDR5 ~40 GB/s

    if (dev.mem_type == ALPHA_MEM_UNIFIED) {
        // Apple Silicon: M1=68, M2=100, M3=150, M3Max=400 GB/s
        // Intel iGPU: ~50 GB/s (shared DDR)
        if (dev.backend_name && strstr(dev.backend_name, "Metal")) return 150000;
        return 50000;
    }

    // Discrete GPU
    if (dev.backend_name && strstr(dev.backend_name, "CUDA")) return 500000;
    return 200000;  // Vulkan discrete (conservative)
}

static size_t estimate_xfer_bw(const alpha_device_info & dev) {
    if (dev.mem_type == ALPHA_MEM_UNIFIED) return estimate_local_bw(dev);  // no transfer cost
    if (dev.mem_type == ALPHA_MEM_SYSTEM) return 40000;

    // Discrete GPU: PCIe bandwidth depends on slot width
    // Heuristic: >8GB VRAM = likely desktop (x16), <=8GB = likely laptop (x4)
    if (dev.mem_total_mb > 8000) return 25000;  // PCIe x16 ~25 GB/s
    return 6000;  // PCIe x4 ~6 GB/s (laptop)
}

// Device Scan

static std::vector<alpha_device_info> scan_devices() {
    std::vector<alpha_device_info> devices;
    size_t n = ggml_backend_dev_count();

    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);

        alpha_device_info info = {};
        info.dev = dev;
        info.index = (int)i;
        info.name = ggml_backend_dev_name(dev);
        info.description = ggml_backend_dev_description(dev);

        // Get backend registry name
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        info.backend_name = reg ? ggml_backend_reg_name(reg) : "unknown";

        // Memory type from device type
        switch (dtype) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:
                info.mem_type = ALPHA_MEM_DISCRETE;
                break;
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                info.mem_type = ALPHA_MEM_SYSTEM;
                break;
            default:  // IGPU, ACCEL
                info.mem_type = ALPHA_MEM_UNIFIED;
                break;
        }

        // Memory
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        info.mem_free_mb = free_bytes / (1024 * 1024);
        info.mem_total_mb = total_bytes / (1024 * 1024);

        // ALPHA_FORCE_VRAM_MB overrides the discrete-GPU free VRAM the planner sees, so one machine can
        // dry-run the auto-fit for any card size. No effect on iGPU/UMA/CPU devices. Testing-only knob.
        if (info.mem_type != ALPHA_MEM_UNIFIED) {
            const char * fv = getenv("ALPHA_FORCE_VRAM_MB");
            if (fv) { long v = atol(fv); if (v > 0) info.mem_free_mb = (size_t) v; }
        }

        // Bandwidth estimates
        info.bw_local_mbs = estimate_local_bw(info);
        info.bw_xfer_mbs = estimate_xfer_bw(info);

        // Async support (from capabilities if available)
        info.supports_async = (dtype != GGML_BACKEND_DEVICE_TYPE_CPU);

        devices.push_back(info);
    }

    return devices;
}

// Model Metadata (lightweight read)

struct model_meta {
    size_t file_size_mb;
    int n_layer;
    int n_embd;
    int n_vocab;      // vocab drives the embd+output tensor size (a fixed GPU cost)
    size_t embd_output_mb;   // GPU-fixed cost (output.weight, or token_embd if tied)
    size_t non_repeating_mb; // output+token_embd bytes, subtracted from the per-layer divisor
    int n_ctx_train;
    int n_expert;
    int n_expert_used;
    int sliding_window;   // >0 => iswa (Gemma sliding-window attn): only global layers' KV scales with ctx
    size_t kv_f16_bpt_layer;  // KV bytes/token/layer at F16 = n_head_kv*(key_len+val_len)*2 (proxy 4*n_embd)
    bool is_moe;
    char arch[64];
};

static model_meta read_model_meta(const char * path) {
    model_meta meta = {};

    // Get file size (use _fseeki64/_ftelli64 on Windows for >2GB files)
#ifdef _WIN32
    FILE * f = fopen(path, "rb");
    if (f) {
        _fseeki64(f, 0, SEEK_END);
        meta.file_size_mb = (size_t)(_ftelli64(f) / (1024 * 1024));
        fclose(f);
    }
#else
    FILE * f = fopen(path, "rb");
    if (f) {
        fseeko(f, 0, SEEK_END);
        meta.file_size_mb = (size_t)(ftello(f) / (1024 * 1024));
        fclose(f);
    }
#endif

    // Check for split GGUF: filename-00001-of-NNNNN.gguf pattern
    // Sum all shard file sizes for correct model size
    {
        std::string p(path);
        // Look for pattern like "-00001-of-" in the path
        size_t pos = p.find("-00001-of-");
        if (pos != std::string::npos) {
            size_t of_pos = pos + 10; // after "-00001-of-"
            size_t ext_pos = p.find(".gguf", of_pos);
            if (ext_pos != std::string::npos) {
                int n_shards = std::atoi(p.c_str() + of_pos);
                if (n_shards > 1) {
                    size_t total_mb = 0;
                    for (int i = 1; i <= n_shards; i++) {
                        char shard_path[4096];
                        std::string prefix = p.substr(0, pos);
                        snprintf(shard_path, sizeof(shard_path), "%s-%05d-of-%05d.gguf",
                                 prefix.c_str(), i, n_shards);
                        FILE * sf = fopen(shard_path, "rb");
                        if (sf) {
#ifdef _WIN32
                            _fseeki64(sf, 0, SEEK_END);
                            total_mb += (size_t)(_ftelli64(sf) / (1024 * 1024));
#else
                            fseeko(sf, 0, SEEK_END);
                            total_mb += (size_t)(ftello(sf) / (1024 * 1024));
#endif
                            fclose(sf);
                        }
                    }
                    if (total_mb > meta.file_size_mb) {
                        meta.file_size_mb = total_mb;
                    }
                }
            }
        }
    }

    // Read GGUF metadata (no_alloc=true: only the header is needed, not tensor data)
    struct gguf_context * ctx = gguf_init_from_file(path, {true, nullptr});
    if (!ctx) {
        // Split GGUF metadata-only shards may fail to parse.
        // Return what we have (file_size_mb from shard sum) and let llama.cpp handle actual loading.
        ALPHA_LOG_DBG("[HW] Warning: could not read GGUF metadata from %s (split GGUF?)\n", path);
        return meta;
    }

    // Find architecture prefix
    char arch[64] = {};
    int arch_idx = gguf_find_key(ctx, "general.architecture");
    if (arch_idx >= 0) {
        const char * a = gguf_get_val_str(ctx, arch_idx);
        if (a) snprintf(arch, sizeof(arch), "%s", a);
    }
    snprintf(meta.arch, sizeof(meta.arch), "%s", arch);

    auto get_u32 = [&](const char * suffix) -> uint32_t {
        char key[128];
        snprintf(key, sizeof(key), "%s.%s", arch, suffix);
        int idx = gguf_find_key(ctx, key);
        if (idx < 0) return 0;
        // Some keys (e.g. Gemma's per-layer attention.head_count_kv / sliding_window) are stored as
        // arrays, not scalars. gguf_get_val_u32 asserts ne==1 and would abort on those, so read element 0
        // as a representative instead.
        if (gguf_get_kv_type(ctx, idx) == GGUF_TYPE_ARRAY) {
            if (gguf_get_arr_type(ctx, idx) == GGUF_TYPE_UINT32 && gguf_get_arr_n(ctx, idx) > 0) {
                return ((const uint32_t *) gguf_get_arr_data(ctx, idx))[0];
            }
            return 0;
        }
        return gguf_get_val_u32(ctx, idx);
    };

    meta.n_layer = (int)get_u32("block_count");
    meta.n_embd = (int)get_u32("embedding_length");
    meta.n_ctx_train = (int)get_u32("context_length");
    meta.n_expert = (int)get_u32("expert_count");
    meta.n_expert_used = (int)get_u32("expert_used_count");
    meta.sliding_window = (int)get_u32("attention.sliding_window");   // 0 if not an SWA/iswa model
    {   // accurate KV bytes/token/layer (F16) from attention geometry; proxy 4*n_embd if unknown
        int nhkv = (int)get_u32("attention.head_count_kv");
        int klen = (int)get_u32("attention.key_length");
        int vlen = (int)get_u32("attention.value_length");
        meta.kv_f16_bpt_layer = (nhkv > 0 && klen > 0)
            ? (size_t)nhkv * (size_t)(klen + (vlen > 0 ? vlen : klen)) * 2
            : (size_t)2 * (size_t)(meta.n_embd > 0 ? meta.n_embd : 1) * 2;
    }
    meta.is_moe = (meta.n_expert > 1 && meta.n_expert_used > 0);

    // vocab size = length of the tokenizer tokens array (fallback {arch}.vocab_size).
    // The token_embd + output(lm_head) tensors are ~vocab*n_embd each and are GPU-resident even under
    // partial offload; big-vocab models (e.g. Gemma 256K) hide ~1-3GB of fixed cost the flat reserve misses.
    {
        int tk = gguf_find_key(ctx, "tokenizer.ggml.tokens");
        if (tk >= 0) {
            meta.n_vocab = (int) gguf_get_arr_n(ctx, tk);
        } else {
            meta.n_vocab = (int) get_u32("vocab_size");
        }
    }

    // GPU-fixed cost from the actual tensor bytes. The output (lm_head) layer is always offloaded to
    // GPU under partial offload; token_embd is typically on the CPU, except when embeddings are tied
    // (no separate output.weight), in which case token_embd serves as the lm_head and is GPU-resident.
    // Using the real quantized size is exact for any vocab.
    {
        int64_t out_id = gguf_find_tensor(ctx, "output.weight");
        int64_t emb_id = gguf_find_tensor(ctx, "token_embd.weight");
        size_t out_b = out_id >= 0 ? gguf_get_tensor_size(ctx, out_id) : 0;
        size_t emb_b = emb_id >= 0 ? gguf_get_tensor_size(ctx, emb_id) : 0;
        // GPU-fixed cost = the output/lm_head layer (always GPU under partial offload); when embeddings
        // are tied (no output.weight) the token_embd tensor is the lm_head -> GPU-resident.
        size_t gpu_fixed_b = out_b > 0 ? out_b : emb_b;
        // Both output+token_embd are non-repeating: subtract from the per-layer divisor so the estimated
        // per-repeating-layer cost is accurate (else the CPU-resident token_embd inflates it -> under-pack).
        size_t non_rep_b = out_b + emb_b;
        meta.embd_output_mb   = gpu_fixed_b / (1024 * 1024);
        meta.non_repeating_mb = non_rep_b   / (1024 * 1024);
    }

    gguf_free(ctx);
    return meta;
}

// Planning Algorithm

static int estimate_layers_that_fit(size_t vram_mb, size_t model_mb, int n_layer,
                                    size_t gpu_fixed_mb, size_t non_repeating_mb) {
    if (n_layer <= 0) return 0;
    // Compute/misc reserve (activation + KV metadata + margin), env-tunable for packing calibration.
    size_t compute_reserve = model_mb < 2000 ? 300 : model_mb < 4000 ? 500 : 700;
    { const char * e = getenv("ALPHA_FIT_RESERVE_MB"); if (e) { long v = atol(e); if (v >= 0) compute_reserve = (size_t) v; } }
    // gpu_fixed_mb = the output/lm_head layer, GPU-resident regardless of ngl. The per-repeating-layer
    // divisor excludes all non-repeating bytes (output + token_embd, the latter on the CPU under partial
    // offload) so it reflects the true per-GPU-layer weight: big-vocab models do not OOM, and
    // lighter-per-layer models are not under-packed by a CPU-resident token_embd bleeding in.
    size_t fixed_mb = compute_reserve + gpu_fixed_mb;
    if (vram_mb <= fixed_mb) return 0;
    size_t available    = vram_mb - fixed_mb;
    size_t repeating_mb = model_mb > non_repeating_mb ? model_mb - non_repeating_mb : model_mb;
    size_t per_layer    = repeating_mb / n_layer;
    if (per_layer == 0) return n_layer;
    int layers = (int)(available / per_layer);
    return std::min(layers, n_layer);
}

// Estimate the GPU-fixed vocab tensor cost (token_embd + output ~ vocab*n_embd each).
// Bytes-per-param factor is env-tunable (ALPHA_VOCAB_FIT_BYTES) for calibration; default 2.0 ~ two
// tensors at ~1 byte/param (Q8-ish embd/output, the common high-precision choice in UD/XL quants).
static size_t estimate_embd_output_mb(int n_vocab, int n_embd) {
    if (n_vocab <= 0 || n_embd <= 0) return 0;
    double vocab_bytes = 2.0;
    const char * e = getenv("ALPHA_VOCAB_FIT_BYTES");
    if (e) { double v = atof(e); if (v > 0) vocab_bytes = v; }
    return (size_t)((double)n_vocab * (double)n_embd * vocab_bytes / (1024.0 * 1024.0));
}

struct alpha_hw_plan alpha_hw_plan_create(
    const char * model_path,
    int user_ngl)
{
    alpha_hw_plan plan = {};
    memset(plan.devices, 0, sizeof(plan.devices));
    memset(plan.tensor_split, 0, sizeof(plan.tensor_split));
    plan.main_device_idx = -1;
    plan.expert_resident_device = -1;
    plan.kv_type_k = GGML_TYPE_F16;
    plan.kv_type_v = GGML_TYPE_F16;
    plan.split_mode = LLAMA_SPLIT_MODE_NONE;

    // Scan hardware
    auto devices = scan_devices();

    // Separate GPUs from CPU
    std::vector<alpha_device_info> gpus;
    alpha_device_info cpu_dev = {};
    bool has_cpu = false;

    for (auto & d : devices) {
        if (d.mem_type == ALPHA_MEM_SYSTEM) {
            cpu_dev = d;
            has_cpu = true;
        } else if (d.mem_type == ALPHA_MEM_DISCRETE) {
            // Only include discrete GPUs - UMA iGPUs (Intel UHD, Iris Xe)
            // report system RAM as "free" memory, which confuses the planner.
            // iGPUs also can't run CUDA kernels on NVIDIA-compiled binaries.
            gpus.push_back(d);
        } else {
            // UMA/iGPU - skip. Treat as CPU-equivalent.
            ALPHA_LOG_DBG("[HW] skipping UMA device: %s (%zuMB reported)\n",
                    d.name, d.mem_free_mb);
        }
    }

    // Sort GPUs for general use (memory first, bandwidth tiebreaker).
    // MoE models get a special re-sort below after reading model metadata.
    std::sort(gpus.begin(), gpus.end(), [](const alpha_device_info & a, const alpha_device_info & b) {
        if (a.mem_free_mb != b.mem_free_mb) return a.mem_free_mb > b.mem_free_mb;
        return a.bw_local_mbs > b.bw_local_mbs;
    });

    // Build device list (GPUs sorted best-first, CPU last)
    plan.n_devices = 0;
    for (auto & g : gpus) {
        if (plan.n_devices < ALPHA_HW_MAX_DEVICES) {
            plan.devices[plan.n_devices++] = g.dev;
        }
    }
    if (has_cpu && plan.n_devices < ALPHA_HW_MAX_DEVICES) {
        plan.devices[plan.n_devices++] = cpu_dev.dev;
    }
    plan.devices[plan.n_devices] = nullptr;

    // Read model metadata
    model_meta meta = read_model_meta(model_path);
    plan.model_size_mb = meta.file_size_mb;
    plan.n_layer = meta.n_layer;
    plan.n_embd = meta.n_embd;
    plan.n_ctx_train = meta.n_ctx_train;
    plan.sliding_window = meta.sliding_window;
    plan.kv_f16_bpt_layer = meta.kv_f16_bpt_layer;
    plan.n_expert = meta.n_expert;
    plan.n_expert_used = meta.n_expert_used;
    plan.is_moe = meta.is_moe;
    snprintf(plan.arch, sizeof(plan.arch), "%s", meta.arch);
    plan.gpu_vram_mb = gpus.empty() ? 0 : gpus[0].mem_free_mb;
    plan.gpu_vram_total_mb = gpus.empty() ? 0 : gpus[0].mem_total_mb;
    plan.system_ram_mb = has_cpu ? cpu_dev.mem_free_mb : 0;
#ifndef _WIN32
    // Use MemAvailable for more accurate RAM estimate (includes reclaimable page cache)
    {
        FILE * mi = fopen("/proc/meminfo", "r");
        if (mi) {
            char line[256];
            while (fgets(line, sizeof(line), mi)) {
                size_t val = 0;
                if (sscanf(line, "MemAvailable: %zu kB", &val) == 1) {
                    size_t ma = val / 1024;
                    if (ma > plan.system_ram_mb) plan.system_ram_mb = ma;
                    break;
                }
            }
            fclose(mi);
        }
    }
#endif

    // User override.
    // Set the layer count but don't skip MoE planning: the user overrides GPU layers, not the expert
    // cache/strategy decision. The MoE path uses user_ngl for backbone placement, then computes K
    // from the remaining VRAM.
    bool user_ngl_override = (user_ngl >= 0);
    if (user_ngl_override && !meta.is_moe) {
        if (!gpus.empty() && user_ngl > meta.n_layer) {
            // user said -ngl 999 (meaning "as many as possible")
            // Don't skip: fall through to step 1, which calculates the optimal layer count
            ALPHA_LOG_DBG("[HW] Dense model: user ngl=%d > n_layer=%d, treating as auto\n",
                    user_ngl, meta.n_layer);
        } else {
            // user gave explicit layer count (e.g. -ngl 55): respect it
            plan.n_gpu_layers = user_ngl;
            plan.main_device_idx = gpus.empty() ? -1 : 0;
            plan.strategy_name = "UO";
            goto plan_kv;
        }
    }
    // MoE with user_ngl falls through; ngl is applied in the MoE path below

    // DeepSeek V4 is a regular MoE model with more experts: it falls through to the dynamic MoE
    // detection below, which computes the cache size from available VRAM.

    // MoE-specific: prefer a discrete GPU for the backbone (dedicated bandwidth).
    // For MoE models, backbone is small (~5-15% of model) and fits on a small dGPU.
    // A 2GB dGPU with dedicated GDDR6 beats an 8GB UMA iGPU because:
    // - Backbone compute uses dedicated bandwidth (not shared with CPU)
    // - Remaining dGPU VRAM acts as hot expert cache
    // - Cold experts stay in system RAM (Vulkan_Host)
    // This planner is the alphachat CLI path (DIRECT / LUT_DIRECT / SELF_SPEC / CPU_MOE, K from VRAM).
    // The serve path uses fused-infer and picks its serve mode by model-vs-VRAM in
    // tools/serve/serve-main.cpp: whole model fits -> ALL_GPU; else fused-infer, then pool-direct
    // (hit-rich) vs CPU-inline split (miss-heavy, pool < ~20% of expert bytes).
    if (meta.is_moe && gpus.size() >= 1) {
        // Backbone fraction depends on expert count: more experts = smaller backbone fraction.
        // 8 experts (Mixtral): ~30%. 64 experts: ~10%. 256 experts: ~3-6%.
        int backbone_pct = meta.n_expert <= 8 ? 30 :
                           meta.n_expert <= 64 ? 12 :
                           meta.n_expert <= 128 ? 8 : 4;
        size_t backbone_mb = meta.file_size_mb * backbone_pct / 100;
        // Overhead: KV cache + compute buffers + CUDA context
        // ~1000-1500 MB total (KV cache at 4K ctx + compute scratch)
        size_t backbone_overhead = 1200;

        // Find best discrete GPU that can fit backbone
        int best_discrete = -1;
        for (int i = 0; i < (int)gpus.size(); i++) {
            if (gpus[i].mem_type == ALPHA_MEM_DISCRETE &&
                gpus[i].mem_free_mb >= backbone_mb + backbone_overhead) {
                if (best_discrete < 0 ||
                    gpus[i].bw_local_mbs > gpus[best_discrete].bw_local_mbs) {
                    best_discrete = i;
                }
            }
        }

        if (best_discrete >= 0) {
            // Re-sort: put the chosen discrete GPU first
            if (best_discrete != 0) {
                std::swap(gpus[0], gpus[best_discrete]);
            }
            // Rebuild device list: discrete dGPU + CPU only
            // Exclude UMA iGPU: its Vulkan backend may not support all ops (e.g. SSM/DeltaNet)
            // and shared memory bandwidth provides no benefit when dGPU handles backbone
            plan.n_devices = 0;
            plan.devices[plan.n_devices++] = gpus[0].dev;  // discrete dGPU
            if (has_cpu && plan.n_devices < ALPHA_HW_MAX_DEVICES) {
                plan.devices[plan.n_devices++] = cpu_dev.dev;
            }
            plan.devices[plan.n_devices] = nullptr;

            auto & dgpu = gpus[0];
            plan.main_device_idx = 0;
            plan.split_mode = LLAMA_SPLIT_MODE_NONE;

            // Check if the entire model fits in GPU (no expert cache needed)
            // Skip DIRECT if user explicitly set low ngl (they want expert cache)
            if (!user_ngl_override && dgpu.mem_free_mb >= meta.file_size_mb + backbone_overhead) {
                plan.n_gpu_layers = 999;
                plan.expert_cache_mb = 0;
                plan.expert_resident_device = 0;
                plan.K_estimate = meta.n_expert;  // all experts fit -> K = n_expert
                plan.K_min = meta.n_expert_used;
                plan.K_draft = meta.n_expert_used;
                plan.strategy = ALPHA_HW_DIRECT;
                plan.strategy_name = "AG";
                ALPHA_LOG_DBG("[HW] MoE model fits in GPU (%zuMB free >= %zuMB model) — DIRECT\n",
                    dgpu.mem_free_mb, meta.file_size_mb + backbone_overhead);
                goto plan_kv;
            }

            // Model doesn't fit entirely: experts go to CPU, backbone stays on GPU
            if (user_ngl_override) {
                // User specified --gpu-layers: use their value for backbone
                plan.n_gpu_layers = user_ngl;
                ALPHA_LOG_DBG("[HW] user ngl=%d override for MoE backbone\n", user_ngl);
            } else {
                // Auto-compute: how many backbone layers fit in VRAM
                size_t per_layer_bb = meta.n_layer > 0 ? backbone_mb / meta.n_layer : backbone_mb;
                if (per_layer_bb > 0 && backbone_mb + backbone_overhead > dgpu.mem_free_mb) {
                    plan.n_gpu_layers = (int)((dgpu.mem_free_mb - backbone_overhead) / per_layer_bb);
                    if (plan.n_gpu_layers < 0) plan.n_gpu_layers = 0;
                    ALPHA_LOG_DBG("[HW] backbone %zuMB > VRAM %zuMB — fitting %d/%d layers on GPU\n",
                            backbone_mb, dgpu.mem_free_mb, plan.n_gpu_layers, meta.n_layer);
                } else {
                    plan.n_gpu_layers = 999;  // backbone fits entirely
                }
            }

            // Tiered K strategy: single-pass decision
            // K_min: minimum for reliable quality (25% of experts)
            // K_draft: minimum for useful draft tokens (12% of experts)
            // Below K_draft: cache is noise, use cpu-moe
            size_t vram_after_backbone = (dgpu.mem_free_mb > backbone_mb + backbone_overhead)
                ? dgpu.mem_free_mb - backbone_mb - backbone_overhead : 0;
            size_t remaining = vram_after_backbone;
            plan.K_min   = std::max(meta.n_expert_used * 8, meta.n_expert / 4);
            plan.K_draft = std::max(meta.n_expert_used * 4, meta.n_expert / 8);

            if (meta.n_expert > 0 && meta.n_layer > 0) {
                size_t expert_total_mb = meta.file_size_mb * (100 - backbone_pct) / 100;
                size_t per_K_mb = expert_total_mb / meta.n_expert;
                plan.K_estimate = per_K_mb > 0 ? (int)(remaining / per_K_mb) : 0;
            }

            if (plan.K_estimate >= plan.K_min) {
                // K is sufficient for reliable quality -> LUT-direct
                plan.expert_cache_mb = (int)std::max(remaining, (size_t)256);
                plan.expert_resident_device = -1;
                plan.strategy = ALPHA_HW_LUT_DIRECT;
                plan.strategy_name = "LD";
                ALPHA_LOG_DBG("[HW] K≈%d >= %d → LUT-direct (reliable quality)\n",
                        plan.K_estimate, plan.K_min);
            } else if (plan.K_estimate >= plan.K_draft) {
                // K is marginal -> self-spec (LUT draft + cpu-moe verify)
                plan.expert_cache_mb = (int)std::max(remaining, (size_t)256);
                plan.expert_resident_device = -1;
                plan.strategy = ALPHA_HW_SELF_SPEC;
                plan.strategy_name = "SS";
                ALPHA_LOG_DBG("[HW] K≈%d in draft range [%d,%d) → self-spec (LUT draft + cpu-moe verify)\n",
                        plan.K_estimate, plan.K_draft, plan.K_min);
            } else {
                // K too low for even draft -> pure cpu-moe
                plan.expert_cache_mb = 0;
                plan.expert_resident_device = -1;
                plan.strategy = ALPHA_HW_CPU_MOE;
                plan.strategy_name = "CO";
                ALPHA_LOG_DBG("[HW] K≈%d < %d → cpu-moe (cache too noisy for draft)\n",
                        plan.K_estimate, plan.K_draft);
            }
            goto plan_kv;
        }
    }

    // Step 1: main model placement (non-MoE or no suitable discrete GPU)
    if (gpus.empty()) {
        plan.n_gpu_layers = 0;
        plan.main_device_idx = -1;
        plan.strategy = ALPHA_HW_CPU_ONLY;
        plan.strategy_name = "CO";
    } else {
        auto & best = gpus[0];
        size_t model_mb = meta.file_size_mb;
        // Overhead: KV cache (regular + SWA) + compute buffers + CUDA context.
        // Models with sliding window attention (Gemma 4) need ~1.3GB for SWA KV alone.
        // Conservative estimates to avoid OOM on tight VRAM.
        size_t overhead_mb = model_mb < 2000 ? 400 : model_mb < 4000 ? 800 : 1500;

        if (model_mb + overhead_mb <= best.mem_free_mb) {
            plan.main_device_idx = 0;
            plan.n_gpu_layers = 999;
            plan.split_mode = LLAMA_SPLIT_MODE_NONE;
            // Chunked KV: context costs no upfront VRAM, use large default
            size_t headroom = best.mem_free_mb - model_mb;
            if (headroom < 1200) {
                plan.kv_type_k = GGML_TYPE_Q4_0;
                plan.kv_type_v = GGML_TYPE_Q4_0;
            }
            plan.strategy = ALPHA_HW_DIRECT;
            plan.strategy_name = "D";
        } else if (gpus.size() >= 2) {
            auto & second = gpus[1];
            size_t total_vram = best.mem_free_mb + second.mem_free_mb;
            float bw_ratio = (float)best.bw_local_mbs / std::max((size_t)1, second.bw_local_mbs);

            if (model_mb + overhead_mb <= total_vram && bw_ratio < 3.0f) {
                plan.main_device_idx = 0;
                plan.n_gpu_layers = 999;
                float total = (float)(best.mem_free_mb + second.mem_free_mb);
                plan.tensor_split[0] = (float)best.mem_free_mb / total;
                plan.tensor_split[1] = (float)second.mem_free_mb / total;

                // Tensor parallelism: use when GPUs are balanced (similar VRAM)
                // and connected via high-bandwidth link (NVLink or same-socket PCIe).
                // bw_ratio < 1.5 = balanced GPUs, good for TP.
                // bw_ratio 1.5-3.0 = imbalanced, use layer split.
                if (bw_ratio < 1.5f && !meta.is_moe) {
                    plan.split_mode = LLAMA_SPLIT_MODE_TENSOR;
                    plan.strategy_name = "TP";
                    ALPHA_LOG_DBG("[HW] Balanced GPUs (ratio=%.1f) — tensor parallelism\n", bw_ratio);
                } else {
                    plan.split_mode = LLAMA_SPLIT_MODE_LAYER;
                    plan.strategy_name = "MS";
                    ALPHA_LOG_DBG("[HW] Imbalanced GPUs (ratio=%.1f) — layer split\n", bw_ratio);
                }
            } else {
                plan.main_device_idx = 0;
                plan.n_gpu_layers = estimate_layers_that_fit(best.mem_free_mb, model_mb, meta.n_layer, meta.embd_output_mb > 0 ? meta.embd_output_mb : estimate_embd_output_mb(meta.n_vocab, meta.n_embd), meta.non_repeating_mb > 0 ? meta.non_repeating_mb : estimate_embd_output_mb(meta.n_vocab, meta.n_embd));
                plan.split_mode = LLAMA_SPLIT_MODE_NONE;
                plan.strategy_name = plan.n_gpu_layers >= meta.n_layer ? "D" : "PO";
            }
        } else {
            plan.main_device_idx = 0;
            if (model_mb <= best.mem_free_mb * 80 / 100) {
                // Model uses <=80% of VRAM - safe for all layers + Q4 KV
                plan.n_gpu_layers = 999;
                plan.kv_type_k = GGML_TYPE_Q4_0;
                plan.kv_type_v = GGML_TYPE_Q4_0;
                plan.n_ctx = 4096;
                plan.strategy = ALPHA_HW_DIRECT_Q4KV;
                plan.strategy_name = "DQ";
            } else {
                // Model exceeds VRAM: use partial offload. Tight-fit strategies (ngl=999 + Q4 KV +
                // reduced ctx) run out of memory on hybrid/recurrent models due to recurrent state;
                // partial offload is always safe and beats a naive whole-layer split.
                int est_layers = estimate_layers_that_fit(best.mem_free_mb, model_mb, meta.n_layer, meta.embd_output_mb > 0 ? meta.embd_output_mb : estimate_embd_output_mb(meta.n_vocab, meta.n_embd), meta.non_repeating_mb > 0 ? meta.non_repeating_mb : estimate_embd_output_mb(meta.n_vocab, meta.n_embd));
                float fit_ratio = (float)est_layers / (float)meta.n_layer;

                if (fit_ratio >= 0.35f && !meta.is_moe) {
                    // Alpha Optimized: self-spec layer-skip draft on GPU + CPU verify
                    // Faster than partial offload and pure CPU on small GPUs
                    int draft_layers = est_layers;
                    int skip = meta.n_layer - draft_layers;
                    plan.n_gpu_layers = draft_layers;
                    plan.kv_type_k = GGML_TYPE_Q4_0;
                    plan.kv_type_v = GGML_TYPE_Q4_0;
                    plan.n_ctx = 2048;
                    plan.self_spec_skip = skip;
                    plan.self_spec_m = 2;  // M=2 is optimal on small GPUs
                    plan.self_spec_draft_ngl = draft_layers;
                    plan.strategy = ALPHA_HW_SELF_SPEC_DENSE;
                    plan.strategy_name = "AO";  // Alpha Optimized
                } else if (fit_ratio >= 0.50f) {
                    // Standard partial offload (only when >50% fits; below that, CPU is faster)
                    plan.n_gpu_layers = est_layers;
                    plan.kv_type_k = GGML_TYPE_Q4_0;
                    plan.kv_type_v = GGML_TYPE_Q4_0;
                    plan.n_ctx = 2048;
                    plan.strategy = ALPHA_HW_PARTIAL_OFFLOAD;
                    plan.strategy_name = "PO";
                } else {
                    // Too little GPU benefit: pure CPU is faster
                    plan.n_gpu_layers = 0;
                    plan.n_ctx = 4096;
                    plan.strategy = ALPHA_HW_CPU_ONLY;
                    plan.strategy_name = "CO";
                }
            }
        }
    }

    // Step 3: expert cache (MoE, non-discrete path).
    // With KV chunks, extended context is handled by CPU save/load,
    // so we only need minimal GPU context (512 tokens). This frees
    // VRAM for more expert cache slots (higher K = better quality).
    if (meta.is_moe && plan.n_ctx > 512) {
        plan.n_ctx = 512;
        plan.kv_type_k = GGML_TYPE_Q4_0;
        plan.kv_type_v = GGML_TYPE_Q4_0;
    }

    if (meta.is_moe && plan.expert_cache_mb == 0 && plan.main_device_idx >= 0 && !gpus.empty()) {
        size_t main_vram = gpus[0].mem_free_mb;
        size_t backbone_mb = meta.file_size_mb * 15 / 100;
        size_t expert_total_mb = meta.file_size_mb * 85 / 100;

        if (backbone_mb + expert_total_mb + 700 <= main_vram) {
            plan.expert_cache_mb = 0;
        } else {
            // Reduced overhead: KV chunks need only a 512-token context (~10MB Q4)
            size_t overhead = 400;
            if (main_vram > backbone_mb + overhead + 512) {
                int cache = (int)(main_vram - backbone_mb - overhead);
                cache = std::min(cache, (int)(main_vram / 2));
                cache = std::max(cache, 512);
                plan.expert_cache_mb = cache;
            } else {
                plan.n_gpu_layers = 0;
                plan.expert_cache_mb = 0;
                plan.strategy_name = "CO";
            }
            plan.expert_resident_device = -1;
        }
    }

plan_kv:

    // Step 4: KV cache quantization (if not already set).
    // MoE models already have n_ctx=512 from step 3 (KV chunks handle extended context)
    if (plan.n_ctx == 0 && plan.main_device_idx >= 0 && !gpus.empty()) {
        size_t model_on_device = (plan.n_gpu_layers >= meta.n_layer)
            ? meta.file_size_mb
            : (size_t)plan.n_gpu_layers * meta.file_size_mb / std::max(1, meta.n_layer);
        size_t remaining = gpus[0].mem_free_mb > model_on_device
            ? gpus[0].mem_free_mb - model_on_device : 0;

        // KV per token (FP16): 2 * n_layer * n_embd * 2 bytes
        size_t kv_per_tok_bytes = 2ULL * meta.n_layer * meta.n_embd * 2;
        size_t kv_per_tok_mb_x1000 = kv_per_tok_bytes * 1000 / (1024 * 1024);  // MB * 1000

        if (kv_per_tok_mb_x1000 > 0) {
            size_t remaining_x1000 = remaining * 1000;
            size_t max_ctx_fp16 = remaining_x1000 / kv_per_tok_mb_x1000;

            // Use as much context as VRAM allows (no artificial 8192 cap)
            // Leave 20% headroom for compute buffers + SWA state
            int max_ctx_safe = (int)(max_ctx_fp16 * 80 / 100);
            // Round down to 4096 alignment
            max_ctx_safe = (max_ctx_safe / 4096) * 4096;
            // Cap at model's training context (if known)
            int model_max = meta.n_ctx_train > 0 ? meta.n_ctx_train : 131072;

            if (max_ctx_safe >= 4096) {
                plan.n_ctx = std::min(max_ctx_safe, model_max);
            } else {
                // Try Q4 KV (4x smaller)
                size_t max_ctx_q4 = max_ctx_fp16 * 4 * 80 / 100;
                plan.kv_type_k = GGML_TYPE_Q4_0;
                plan.kv_type_v = GGML_TYPE_Q4_0;
                int q4_ctx = ((int)max_ctx_q4 / 4096) * 4096;
                plan.n_ctx = q4_ctx >= 4096 ? std::min(q4_ctx, model_max) : 2048;
            }
            ALPHA_LOG_DBG("[HW] KV budget: %zuMB remaining, FP16 max=%zu tokens, plan ctx=%d\n",
                    remaining, max_ctx_fp16, plan.n_ctx);
        }
    }

    // Determine training mode: if model doesn't fully fit on GPU, use pipeline
    if (!gpus.empty() && meta.file_size_mb > 0) {
        size_t best_vram = gpus[0].mem_free_mb;
        // For training, model + activations + overhead must fit
        // Conservative: model must fit within 80% of VRAM (leave room for activations/KV)
        if (meta.file_size_mb > best_vram * 80 / 100) {
            plan.train_mode = alpha_hw_plan::ALPHA_TRAIN_HW_PIPELINE;
        } else {
            plan.train_mode = alpha_hw_plan::ALPHA_TRAIN_HW_DIRECT;
        }
    } else {
        plan.train_mode = alpha_hw_plan::ALPHA_TRAIN_HW_PIPELINE;  // CPU-only -> always pipeline concept
    }

    // Capacity check: GPU + RAM + swap + disk must fit the model.
    // mmap uses virtual memory backed by page cache (MemAvailable, not MemFree).
    // ggml_backend_dev_memory returns MemFree which excludes reclaimable cache.
    // On a system with most memory in buff/cache, MemFree can be tiny while MemAvailable is large.
    {
        size_t gpu_mb = 0;
        for (auto & g : gpus) gpu_mb += g.mem_free_mb;
        size_t ram_mb = has_cpu ? cpu_dev.mem_free_mb : 0;
        size_t disk_mb = get_disk_free_mb(model_path);

#ifndef _WIN32
        // Read /proc/meminfo for MemAvailable + SwapFree (Linux)
        // MemAvailable includes reclaimable page cache, the real capacity for mmap
        FILE * meminfo = fopen("/proc/meminfo", "r");
        if (meminfo) {
            char line[256];
            size_t mem_available_mb = 0;
            size_t swap_free_mb = 0;
            while (fgets(line, sizeof(line), meminfo)) {
                size_t val = 0;
                if (sscanf(line, "MemAvailable: %zu kB", &val) == 1) {
                    mem_available_mb = val / 1024;
                } else if (sscanf(line, "SwapFree: %zu kB", &val) == 1) {
                    swap_free_mb = val / 1024;
                }
            }
            fclose(meminfo);
            // Use MemAvailable if larger than ggml's MemFree report
            if (mem_available_mb > ram_mb) {
                ram_mb = mem_available_mb;
            }
            ram_mb += swap_free_mb;
        }
#endif

        plan.total_capacity_mb = gpu_mb + ram_mb + disk_mb;
        // mmap loads model pages on demand from disk, so the entire model need not be in RAM.
        // Only require that we can fit the backbone (non-expert layers) + working memory.
        // Expert layers are paged from disk as needed (cpu-moe mode).
        // Use 50% of model size as minimum (backbone + active working set).
        plan.can_run = (plan.total_capacity_mb >= meta.file_size_mb / 2);
    }

    return plan;
}

// Pretty Print

void alpha_hw_plan_print(const struct alpha_hw_plan * plan) {
    // Minimal output: no internal details (memory sizes, strategy names, expert counts)
    (void)plan;
}
