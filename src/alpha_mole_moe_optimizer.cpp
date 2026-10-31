// SPDX-License-Identifier: Apache-2.0
#include "alpha_mole_moe_optimizer_internal.h"
#include "alpha_mole_moe_optimizer.h"  // for _acda/sync declarations
#include "alpha_moe_portfolio.h"       // for alpha_moe_profile (apply_warmup)
#include "alpha_moe_pack.h"            // for .apack encrypted profile loading
#include <atomic>
#include <chrono>
#ifdef _WIN32
#define NOMINMAX  // prevent windows.h min/max macros conflicting with std::min/max
#include <windows.h>
#else
#include <dirent.h>
#endif
#include "alpha_mole_expert_cache.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
// Note: direct CUDA API calls (cudaMemcpyAsync) are guarded with GGML_USE_CUDA
// but cuda_runtime.h may not be in the include path for the llama target.
// Use ggml_backend_tensor_set() instead for portability.
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <numeric>
#include <thread>

// _acda/sync live in ggml-cuda.cu. On Windows shared builds ggml-cuda is a separate DLL, so these
// stubs resolve the symbols at runtime and llama.dll links without them. Static and Linux builds
// link the CUDA symbols directly.
#if defined(_WIN32) && defined(LLAMA_SHARED)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static HMODULE g_cuda_dll = nullptr;
static bool g_cuda_dll_loaded = false;

typedef void (*fn_dma_init)(void);
typedef void (*fn_dma_async)(struct ggml_tensor *, const void *, size_t, size_t);
typedef void (*fn_dma_sync)(void);
typedef void (*fn_dma_barrier)(void);
typedef void* (*fn_pinned_alloc)(size_t);
typedef void (*fn_pinned_free)(void *);
typedef void (*fn_dma_async_staged)(struct ggml_tensor *, const void *, size_t, size_t, void *);

static fn_dma_init         p_dma_init = nullptr;
static fn_dma_async        p_dma_async = nullptr;
static fn_dma_sync         p_dma_sync = nullptr;
static fn_dma_barrier      p_dma_barrier = nullptr;
static fn_pinned_alloc     p_pinned_alloc = nullptr;
static fn_pinned_free      p_pinned_free = nullptr;
static fn_dma_async_staged p_dma_async_staged = nullptr;

static void load_cuda_dll() {
    if (g_cuda_dll_loaded) return;
    g_cuda_dll_loaded = true;
    g_cuda_dll = LoadLibraryA("ggml-cuda.dll");
    if (!g_cuda_dll) { ALPHA_LOG_DBG("[alpha] ggml-cuda.dll not found, DMA disabled\n"); return; }
    p_dma_init         = (fn_dma_init)GetProcAddress(g_cuda_dll, "_acdi");
    p_dma_async        = (fn_dma_async)GetProcAddress(g_cuda_dll, "_acda");
    p_dma_sync         = (fn_dma_sync)GetProcAddress(g_cuda_dll, "_acds");
    p_dma_barrier      = (fn_dma_barrier)GetProcAddress(g_cuda_dll, "_acdb");
    p_pinned_alloc     = (fn_pinned_alloc)GetProcAddress(g_cuda_dll, "_acpa");
    p_pinned_free      = (fn_pinned_free)GetProcAddress(g_cuda_dll, "_acpf");
    p_dma_async_staged = (fn_dma_async_staged)GetProcAddress(g_cuda_dll, "_acdas");
}

extern "C" {
void _acdi(void) { load_cuda_dll(); if (p_dma_init) p_dma_init(); }
void _acda(struct ggml_tensor *d, const void *s, size_t o, size_t sz) { if (p_dma_async) p_dma_async(d,s,o,sz); }
void _acds(void) { if (p_dma_sync) p_dma_sync(); }
void _acdb(void) { if (p_dma_barrier) p_dma_barrier(); }
void * _acpa(size_t sz) { load_cuda_dll(); return p_pinned_alloc ? p_pinned_alloc(sz) : nullptr; }
void   _acpf(void *p) { if (p_pinned_free) p_pinned_free(p); }
void   _acdas(struct ggml_tensor *d, const void *s, size_t o, size_t sz, void *b) { if (p_dma_async_staged) p_dma_async_staged(d,s,o,sz,b); }
}
#endif

// Self-speculative draft mode: graph builder reads these
thread_local bool g_alpha_draft_mode = false;
thread_local int  g_alpha_draft_expert_count = 0;  // 0 = Mode C (backbone), >0 = Mode A (reduced)

// Top-k expert reduction: override n_expert_used in graph builder
thread_local int g_alpha_topk_override = 0;  // 0 = use model default

// Frequency profiling: when true, argsort tensors are marked as graph outputs so their data
// persists after llama_decode() returns (needed by the cpu-moe record()). Only set during
// --profile-frequency runs; normal inference is unaffected.
std::atomic<bool> g_alpha_freq_profiling{false};

// Debug logging: when false, internal debug logs are suppressed.
// Set by --debug CLI flag via alpha_moe_optimizer_config.debug.
#ifdef ALPHA_DEBUG_BUILD
bool g_alpha_debug = false;
#endif

// Mixed draft shared expert scaling: boost exact shared expert to compensate for
// approximated routed experts. Set by enter/exit_mixed_draft_mode().
thread_local float g_alpha_mixed_shexp_scale = 1.0f;

// Struct definitions are in alpha_mole_moe_optimizer_internal.h

alpha_moe_optimizer * alpha_moe_optimizer_init(
        struct llama_model   * model,
        struct llama_context * ctx,
        alpha_moe_optimizer_config config) {
    if (!model) return nullptr;

    auto * opt = new alpha_moe_optimizer();
    opt->model  = model;
    opt->ctx    = ctx;
    opt->config = config;
    opt->total_tokens = 0;

    // Set debug logging from config
#ifdef ALPHA_DEBUG_BUILD
    if (config.debug) g_alpha_debug = true;
#endif

    // Enable profiling flag so build_moe_ffn marks argsort tensors as outputs.
    // This ensures ggml_backend_tensor_get() works after llama_decode() in cpu-moe mode.
    if (config.enable_profiling) {
        g_alpha_freq_profiling.store(true, std::memory_order_relaxed);
    }
    opt->cache_clock = 0;
    opt->cache_hits  = 0;
    opt->cache_misses = 0;
    opt->cache_enabled = false;
    opt->tiered_cache = false;
    opt->gpu_dev = nullptr;
    opt->gpu_buft = nullptr;
    opt->mixed_experts_enabled = false;
    opt->mixed_n_clusters = 0;
    opt->eval_callback_active = false;  // set to true when alphachat installs callback
    opt->callback_was_enabled = false;
    opt->reactive_layer_counter = 0;
    opt->speculative_pass = false;
    opt->prompt_mode = false;
    opt->last_token_miss_layers = 0;
    opt->last_token_total_layers = 0;
    if (config.split_group > 0) {
        opt->split_group = config.split_group;
    } else {
        opt->split_group = 1;
    }
    opt->cache_ctx = nullptr;
    opt->remap_ctx = nullptr;
    opt->thermal_policy = ALPHA_MOE_THERMAL_NORMAL;
    opt->current_temp = 0.0f;
    opt->thermal_skipped = 0;
    opt->thermal_substituted = 0;
    opt->prediction_reliable = true;  // assume reliable until proven otherwise
    opt->spec_draft_active = false;
    opt->lut_prefill_active = false;
    opt->profiles_loaded = false;
    opt->spec_lut_ctx = nullptr;
    opt->spec_lut_ready = false;
    opt->spec_draft_tokens = 0;
    opt->spec_verify_tokens = 0;
    opt->spec_accepted_tokens = 0;
    opt->pred_set_matches = 0;
    opt->pred_set_total = 0;
    opt->prefetch_enabled = config.prefetch_next_layer;
    opt->prefetch_depth = config.prefetch_depth > 0 ? config.prefetch_depth : 1;
    opt->prefetch_hits = 0;
    opt->prefetch_total = 0;
    opt->n_layer = llama_model_n_layer(model);
    opt->n_embd  = llama_model_n_embd(model);
    opt->K = 0;

    // Read expert count from model metadata
    opt->n_expert = 0;
    opt->n_expert_used = 0;
    char buf[64];
    for (int i = 0; i < llama_model_meta_count(model); i++) {
        char key[128] = {};
        llama_model_meta_key_by_index(model, i, key, sizeof(key));
        if (strstr(key, "expert_count")) {
            llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
            opt->n_expert = atoi(buf);
        } else if (strstr(key, "expert_used_count")) {
            llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
            opt->n_expert_used = atoi(buf);
        }
    }

    opt->is_moe = (opt->n_expert > 1 && opt->n_expert_used > 0);
    if (!opt->is_moe) {
        return opt;
    }

    // Log architecture and detect hybrid SSM models
    {
        char arch_buf[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));
        (void)arch_buf;

        // qwen3next (hybrid SSM+attention MoE, tiny 512-wide expert FFN): the expert cache tensor
        // swap crashes in quantize_mmq_q8_1_cuda, because ggml_cuda_launch_mm_ids_helper builds
        // invalid ids_src1 when expert IDs are sparse cache slots instead of contiguous expert
        // indices. The model runs well without the optimizer, so it is disabled for this arch.
        if (strcmp(arch_buf, "qwen3next") == 0) {
            opt->is_moe = false;
            return opt;
        }
    }

    // split_group=2 for high-expert models would halve callback stalls, but the zero-expert path
    // for skipped layers produces garbled output.
    // TODO: fix the zero-expert write path, then re-enable.
    // if (opt->split_group == 1 && config.split_group <= 0 && opt->n_expert >= 128) {
    //     opt->split_group = 2;
    // }

    // Discover layer expert tensors
    opt->layer_info.resize(opt->n_layer);
    int moe_layers = 0;
    size_t total_cpu = 0;

    // MTP layers are not regular MoE layers; the expert cache skips them
    int n_mtp = (int)model->hparams.nextn_predict_layers;
    int n_regular_layers = opt->n_layer - n_mtp;

    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        info.orig_exps[OP_GATE]    = model->get_moe_expert_tensor(il, 0);
        info.orig_exps[OP_UP]      = model->get_moe_expert_tensor(il, 1);
        info.orig_exps[OP_DOWN]    = model->get_moe_expert_tensor(il, 2);
        info.orig_exps[OP_GATE_UP] = model->get_moe_expert_tensor(il, 3);
        info.has_experts = (info.orig_exps[OP_GATE] || info.orig_exps[OP_GATE_UP]) && info.orig_exps[OP_DOWN];

        // Skip MTP layers: they have different tensor types (F16) and run independently
        if (n_mtp > 0 && il >= n_regular_layers) {
            info.has_experts = false;
            if (info.orig_exps[OP_GATE] || info.orig_exps[OP_DOWN]) {
                ALPHA_LOG_DBG( "  [MTP] skipping layer %d (MTP layer, not cached)\n", il);
            }
            continue;
        }

        if (!info.has_experts) continue;
        moe_layers++;

        for (int op = 0; op < OP_COUNT; op++) {
            ggml_tensor * t = info.orig_exps[op];
            if (t && ggml_n_dims(t) >= 3) {
                info.expert_slice_bytes[op] = ggml_nbytes(t) / t->ne[2];
            } else if (t) {
                info.expert_slice_bytes[op] = ggml_nbytes(t);
            }
        }

        // GPU residency uses both the buffer name and the host check. If the optimizer is active
        // the model did not fit in VRAM (direct mode skips the optimizer), so expert data is
        // host-resident even when -fit overflow marks some layers as "CUDA".
        info.on_gpu = false;
        ggml_tensor * t = info.orig_exps[OP_DOWN];
        if (t && t->buffer) {
            const char * bn = ggml_backend_buffer_name(t->buffer);
            bool buf_is_gpu = (strstr(bn, "CUDA") || strstr(bn, "Metal") || strstr(bn, "Vulkan"));
            bool buf_is_host = ggml_backend_buffer_is_host(t->buffer);
            // Trust is_host if true (definitive). For GPU buffers, check if this is a
            // -fit overflow case: if model > VRAM, GPU-named buffers may have host-spilled data.
            info.on_gpu = buf_is_gpu && !buf_is_host;
            if (!info.on_gpu) total_cpu += ggml_nbytes(t);
        }
    }

    // If any expert layer is on CPU, treat all as CPU (consistent pipeline).
    // -fit overflow may mark the first partial layer as GPU even though its data
    // is host-resident. Since the pipeline uses the same callback for all layers,
    // mixed GPU/CPU is worse than all-CPU (cache allocation mismatch).
    if (total_cpu > 0) {
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            if (info.has_experts && info.on_gpu) {
                info.on_gpu = false;
                total_cpu += ggml_nbytes(info.orig_exps[OP_DOWN]);
                ALPHA_LOG_DBG("  Layer %d: forced CPU (mixed→uniform pipeline)\n", il);
            }
        }
    }

    // Detect shared experts (enables self-speculative draft mode)
    opt->has_shared_experts = false;
    opt->draft_mode = false;
    opt->draft_expert_count = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & layer = model->layers[il];
        if (layer.ffn_up_shexp && layer.ffn_gate_shexp && layer.ffn_down_shexp) {
            opt->has_shared_experts = true;
            break;
        }
    }

    ALPHA_LOG_DBG( "alpha_moe_optimizer: MoE — %d layers, %d experts/layer, %d active/token%s\n",
            moe_layers, opt->n_expert, opt->n_expert_used,
            opt->has_shared_experts ? ", shared experts available (self-spec OK)" : "");

#if !defined(_WIN32)
    // Expert access is router-driven (random), so MADV_RANDOM on the mmap'd expert tensors keeps
    // readahead from evicting hot experts from the page cache.
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        for (int op = 0; op < OP_COUNT; op++) {
            ggml_tensor * t = info.orig_exps[op];
            if (t && t->data && ggml_backend_buffer_is_host(t->buffer)) {
                madvise(t->data, ggml_nbytes(t), MADV_RANDOM);
            }
        }
    }
#endif

    // Initialize expert activation profiling + prediction helpers
    opt->expert_activation_count.resize(opt->n_layer);
    opt->hot_per_layer.resize(opt->n_layer);
    opt->cross_layer_pred.resize(opt->n_layer);
    opt->recent_routing.resize(opt->n_layer);
    opt->recent_idx = 0;
    opt->hot_update_interval = 50;  // update hot experts every 50 tokens
    for (int il = 0; il < opt->n_layer; il++) {
        opt->expert_activation_count[il].resize(opt->n_expert, 0);
        opt->cross_layer_pred[il].resize(opt->n_expert_used, 0);
        opt->recent_routing[il].resize(alpha_moe_optimizer::TEMPORAL_DEPTH);
        for (int d = 0; d < alpha_moe_optimizer::TEMPORAL_DEPTH; d++) {
            opt->recent_routing[il][d].resize(opt->n_expert_used, -1);
        }
    }

    // prev_routing and prev_entropy are needed for all MoE layers (per-layer logging), even when
    // the cache is disabled
    opt->prev_routing.resize(opt->n_layer);
    opt->prev_entropy.resize(opt->n_layer, 0.5f);
    for (int il = 0; il < opt->n_layer; il++) {
        if (opt->layer_info[il].has_experts) {
            opt->prev_routing[il].resize(opt->n_expert_used, 0);
        }
    }

    // Initialize sticky routing momentum
    opt->expert_momentum.resize(opt->n_layer);
    for (int il = 0; il < opt->n_layer; il++) {
        opt->expert_momentum[il].resize(opt->n_expert, 0.0f);
    }

    // Determine hot experts (always resident)
    int n_hot = config.n_hot_experts > 0 ? config.n_hot_experts : std::max(1, opt->n_expert / 8);
    // Initially, hot experts = first N (will be updated from profiling data)
    for (int e = 0; e < n_hot && e < opt->n_expert; e++) {
        opt->hot_experts.push_back(e);
    }

    if (config.mobile_mode) {
        float t_throttle = config.thermal_throttle_temp > 0 ? config.thermal_throttle_temp : 38.0f;
        float t_emergency = config.thermal_emergency_temp > 0 ? config.thermal_emergency_temp : 43.0f;
        // fprintf(stderr, "  mobile mode: %d hot experts, thermal throttle at %.0f degC, emergency at %.0f degC\n",
        //         n_hot, t_throttle, t_emergency);
    }

    // MoE layer index mapping (moe_idx -> layer index)
    for (int il = 0; il < opt->n_layer; il++) {
        if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) {
            opt->moe_layer_indices.push_back(il);
        }
    }
    opt->n_moe_layers = (int)opt->moe_layer_indices.size();

    // Load expert grouping from JSON if provided
    opt->groups_enabled = false;
    opt->group_cache_hits = 0;
    opt->group_cache_misses = 0;
    if (config.expert_groups_path) {
        FILE * gf = fopen(config.expert_groups_path, "r");
        if (gf) {
            fseek(gf, 0, SEEK_END);
            long gsize = ftell(gf);
            fseek(gf, 0, SEEK_SET);
            std::string gjson(gsize, '\0');
            fread(&gjson[0], 1, gsize, gf);
            fclose(gf);

            // Minimal JSON parser: extract group assignments per layer
            // Format: {"layers": {"0": {"groups": [{"id":0,"experts":[1,2,3]}, ...]}, ...}}
            opt->expert_grouping.resize(opt->n_layer);
            bool parse_ok = false;

            // Find "layers" object
            size_t layers_pos = gjson.find("\"ly\"");
            if (layers_pos != std::string::npos) {
                parse_ok = true;
                for (int il = 0; il < opt->n_layer; il++) {
                    if (!opt->layer_info[il].has_experts) continue;

                    auto & lg = opt->expert_grouping[il];
                    lg.expert_to_group.resize(opt->n_expert, -1);

                    // Find this layer's entry
                    std::string layer_key = "\"" + std::to_string(il) + "\"";
                    size_t lpos = gjson.find(layer_key, layers_pos);
                    if (lpos == std::string::npos) continue;

                    // Find "groups" array for this layer
                    size_t gpos = gjson.find("\"gp\"", lpos);
                    if (gpos == std::string::npos || gpos > lpos + 500) continue;

                    // Parse each group: find "experts":[...] blocks
                    size_t search_start = gpos;
                    int group_id = 0;
                    while (true) {
                        size_t epos = gjson.find("\"ex\"", search_start);
                        if (epos == std::string::npos) break;

                        // Check we haven't crossed into the next layer
                        size_t next_layer = gjson.find("\"" + std::to_string(il + 1) + "\"", lpos);
                        if (next_layer != std::string::npos && epos > next_layer) break;

                        // Find the array brackets
                        size_t arr_start = gjson.find('[', epos);
                        size_t arr_end = gjson.find(']', arr_start);
                        if (arr_start == std::string::npos || arr_end == std::string::npos) break;

                        // Parse comma-separated integers
                        alpha_moe_optimizer::expert_group grp;
                        grp.group_id = group_id;
                        grp.last_used = 0;

                        std::string arr = gjson.substr(arr_start + 1, arr_end - arr_start - 1);
                        size_t pos = 0;
                        while (pos < arr.size()) {
                            while (pos < arr.size() && !isdigit(arr[pos])) pos++;
                            if (pos >= arr.size()) break;
                            int eid = atoi(&arr[pos]);
                            if (eid >= 0 && eid < opt->n_expert) {
                                grp.expert_ids.push_back(eid);
                                lg.expert_to_group[eid] = group_id;
                            }
                            while (pos < arr.size() && (isdigit(arr[pos]) || arr[pos] == '-')) pos++;
                        }

                        if (!grp.expert_ids.empty()) {
                            lg.groups.push_back(std::move(grp));
                            group_id++;
                        }
                        search_start = arr_end + 1;
                    }
                }
            }

            if (parse_ok) {
                int total_groups = 0;
                for (auto & lg : opt->expert_grouping) total_groups += (int)lg.groups.size();
                if (total_groups > 0) {
                    opt->groups_enabled = true;
                    ALPHA_LOG_DBG( "  expert grouping: loaded %d groups from %s\n",
                            total_groups, config.expert_groups_path);
                }
            }
            if (!opt->groups_enabled) {
                ALPHA_LOG_DBG( "  expert grouping: failed to parse %s\n", config.expert_groups_path);
            }
        } else {
            ALPHA_LOG_DBG( "  expert grouping: cannot open %s\n", config.expert_groups_path);
        }
    }

    if (total_cpu == 0) {
        // fprintf(stderr, "  ALL experts on GPU - maximum speed, no cache needed\n");
        return opt;
    }
    // fprintf(stderr, "  expert tensors: %.1f MB on CPU\n", (float)total_cpu/(1024*1024));

    // Setup contiguous GPU cache if budget is set
    if (config.vram_budget_mb > 0) {
        alpha_moe_optimizer_set_context(opt, ctx);
    }

    return opt;
}

void alpha_moe_optimizer_set_context(alpha_moe_optimizer * opt, struct llama_context * ctx) {
    if (!opt || !opt->is_moe) return;
    if (opt->cache_enabled) {
        // Already set up: update the context and re-reserve the scheduler
        opt->ctx = ctx;
        if (ctx) {
            llama_context_reset_sched(ctx);
        }
        return;
    }
    opt->ctx = ctx;

    // Check if experts are on CPU and cache should be set up
    bool has_cpu_experts = false;
    for (int il = 0; il < opt->n_layer; il++) {
        if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) {
            has_cpu_experts = true;
            break;
        }
    }
    if (!has_cpu_experts || opt->config.vram_budget_mb == 0) return;

    // Pick the GPU device (discrete or integrated) with the most free VRAM
    ggml_backend_dev_t gpu_dev = nullptr;
    size_t best_vram = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            if (free > best_vram) {
                best_vram = free;
                gpu_dev = dev;
            }
        }
    }
    if (!gpu_dev) return;

    // Initialize async DMA stream for expert cache loading
    _acdi();

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(gpu_dev);

    // Cache allocation deferred to alpha_moe_optimizer_alloc_cache()
    // (called after context creation so compute buffer is already allocated)
    opt->gpu_dev = gpu_dev;
    opt->gpu_buft = gpu_buft;
}

void alpha_moe_optimizer_alloc_cache(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || !opt->gpu_dev) return;
    if (opt->cache_enabled) return;  // already allocated

    ggml_backend_dev_t gpu_dev = opt->gpu_dev;
    ggml_backend_buffer_type_t gpu_buft = opt->gpu_buft;

    // Query VRAM after context creation, so the compute buffer is already accounted for
    size_t vram_free = 0, vram_total = 0;
    ggml_backend_dev_memory(gpu_dev, &vram_free, &vram_total);
    ALPHA_LOG_DBG( "  VRAM: %.0f MB free / %.0f MB total (after model load)\n",
        vram_free / 1048576.0, vram_total / 1048576.0);

    // Calculate bytes per expert per op type
    size_t down_bytes = 0, gate_bytes = 0, up_bytes = 0, gate_up_bytes = 0;
    bool has_fused = false;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        down_bytes = info.expert_slice_bytes[OP_DOWN];
        gate_bytes = info.expert_slice_bytes[OP_GATE];
        up_bytes = info.expert_slice_bytes[OP_UP];
        gate_up_bytes = info.expert_slice_bytes[OP_GATE_UP];
        break;
    }
    if (down_bytes == 0) return;

    int moe_layers = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) moe_layers++;
    }

    // The compute buffer is already allocated, but graph_reserve during draft/verify or a two-pass
    // serve may re-allocate a larger one (different graph shape), so keep headroom. Small GPUs need
    // proportionally more: without it cuBLAS runs out of memory after a few two-pass cycles.
    size_t budget_bytes = opt->config.vram_budget_mb * 1024ULL * 1024ULL;
    size_t vram_used_by_model = vram_total - vram_free;  // backbone + KV + compute buffer
    size_t reserve;
    if (vram_total < 12ULL * 1024 * 1024 * 1024) {
        // Small GPU (8-10GB): reserve 40% for compute buffers + graph realloc
        reserve = vram_free * 4 / 10;
        if (reserve < 2048ULL * 1024 * 1024) reserve = 2048ULL * 1024 * 1024;  // min 2 GB
    } else {
        // Large GPU (16-24GB+): 20% reserve, capped at 3GB
        reserve = std::min(vram_free / 5, (size_t)(3072ULL * 1024 * 1024));
        if (reserve < 512ULL * 1024 * 1024) reserve = 512ULL * 1024 * 1024;
    }
    size_t vram_max = (vram_free > reserve) ? vram_free - reserve : 0;
    size_t available = (budget_bytes > 0 && budget_bytes < vram_max) ? budget_bytes : vram_max;

    // Tiered cache: down per-layer (K x moe_layers) + gate/up global (K x 1 each)
    // K = available / (down_bytes x moe_layers + gate_bytes + up_bytes)
    // For fused gate_up: K = available / (down_bytes x moe_layers + gate_up_bytes)
    size_t per_layer_bytes = down_bytes;  // per-layer: only down
    size_t global_bytes = has_fused ? gate_up_bytes : (gate_bytes + up_bytes);  // global: gate+up
    size_t bytes_per_K = per_layer_bytes * moe_layers + global_bytes;

    // Compare tiered vs flat K
    size_t all_ops = has_fused ? (down_bytes + gate_up_bytes) : (down_bytes + gate_bytes + up_bytes);
    int K_flat = (int)(available / (all_ops * moe_layers));
    int K_tiered = (int)(available / bytes_per_K);

    opt->tiered_cache = false;
    // Scratch mode: cache only down_exps per layer, gate+up into shared scratch (ne02=K)
    // Scratch is refreshed inside each callback (no cross-layer corruption)
    // ne02=K satisfies MUL_MAT_ID kernel bounds check; only slots 0-(n_expert_used-1) written
    bool use_scratch = !has_fused && !opt->config.flat_cache;  // scratch disabled by --flat-cache
    if (use_scratch) {
        // K x down_bytes x moe_layers + K x (gate_bytes + up_bytes) = available
        int K_scratch = (int)(available / (down_bytes * moe_layers + gate_bytes + up_bytes));
        if (K_scratch > K_flat * 1.3) {
            opt->K = K_scratch;
            opt->use_scratch = true;
            ALPHA_LOG_DBG( "  SCRATCH cache: K=%d (vs flat K=%d, %.1fx), down per-layer + gate/up scratch\n",
                K_scratch, K_flat, (float)K_scratch / K_flat);
        } else {
            opt->K = K_flat;
            opt->use_scratch = false;
            ALPHA_LOG_DBG( "  FLAT cache: K=%d, per_expert=%.0f KB, layers=%d\n",
                K_flat, all_ops / 1024.0, moe_layers);
        }
    } else {
        opt->K = K_flat;
        opt->use_scratch = false;
        ALPHA_LOG_DBG( "  FLAT cache: K=%d, per_expert=%.0f KB, layers=%d\n",
            K_flat, all_ops / 1024.0, moe_layers);
    }
    opt->K = std::min(opt->K, opt->n_expert);
    if (opt->K >= opt->n_expert) {
        // All experts fit in cache. Check if experts are actually on GPU already:
        // if so, skip the cache. If experts are on CPU, keep the cache
        // and preload everything for a 100% hit rate with GPU-speed MUL_MAT_ID.
        bool experts_on_gpu = true;
        for (int il = 0; il < opt->n_layer; il++) {
            if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) {
                experts_on_gpu = false;
                break;
            }
        }
        if (experts_on_gpu && !opt->config.force_cache) {
            ALPHA_LOG_DBG( "  K=%d >= n_expert=%d, experts on GPU — skipping cache (direct GPU mode)\n",
                opt->K, opt->n_expert);
            opt->cache_enabled = false;
            return;
        }
        // Experts on CPU but all fit in cache: preload everything
        opt->K = opt->n_expert;
        ALPHA_LOG_DBG( "  K=%d = n_expert=%d, experts on CPU — cache ALL for 100%% hit rate\n",
            opt->K, opt->n_expert);
    }
    if (opt->K < opt->n_expert_used) {
        ALPHA_LOG_DBG( "  expert cache: not enough VRAM for minimum K=%d (computed K=%d), skipping cache\n",
            opt->n_expert_used, opt->K);
        return;
    }

    // Quality gate: warn if K is below the threshold for reliable output
    {
        int K_min = std::max(opt->n_expert_used * 8, opt->n_expert / 4);
        if (opt->K < K_min) {
            ALPHA_LOG_DBG( "\n  *** WARNING: Expert cache K=%d < K_min=%d ***\n", opt->K, K_min);
            ALPHA_LOG_DBG( "  Output quality will be degraded (wrong answers, repetition loops).\n");
            ALPHA_LOG_DBG( "  For reliable output: use --cpu-moe, or increase VRAM, or use a smaller model.\n\n");
        }
    }

    // Cap hot experts to leave room for active experts + zero-expert in cache
    // Without this, K <= n_hot+1 means all slots are protected, nothing is evicted, and outputs are zero
    int max_hot = opt->K - opt->n_expert_used - 1;  // reserve n_expert_used + 1 (zero) slots
    if (max_hot < 0) max_hot = 0;
    if ((int)opt->hot_experts.size() > max_hot) {
        ALPHA_LOG_DBG( "  capping hot experts: %d -> %d (K=%d, need %d free slots)\n",
            (int)opt->hot_experts.size(), max_hot, opt->K, opt->n_expert_used + 1);
        opt->hot_experts.resize(max_hot);
    }

    float total_gb;
    if (opt->use_scratch) {
        total_gb = (float)(down_bytes * moe_layers * opt->K +
                   (gate_bytes + up_bytes) * opt->K) / (1024.0f*1024*1024);
    } else {
        size_t all_ops_sz = has_fused ? (down_bytes + gate_up_bytes) : (down_bytes + gate_bytes + up_bytes);
        total_gb = (float)(all_ops_sz * moe_layers * opt->K) / (1024.0f*1024*1024);
    }
    ALPHA_LOG_DBG( "  cache: K=%d/%d, hot=%d, total %.1f GB\n", opt->K, opt->n_expert,
            (int)opt->hot_experts.size(), total_gb);

    // Allocate cache tensors + ID tensors in same context
    int n_ops = has_fused ? 2 : 3;
    int n_cache_tensors;
    if (opt->use_scratch) {
        // Per-layer: 1 down cache + 2 ID tensors (gate/up IDs + down IDs)
        // Global: 2 scratch tensors (gate + up)
        n_cache_tensors = moe_layers * 1 + moe_layers * 2 + 2;
    } else {
        n_cache_tensors = moe_layers * n_ops + moe_layers;  // cache tensors + ID tensors
    }
    // Padding for per-tensor overhead and alignment
    size_t ctx_size = (n_cache_tensors + 16) * ggml_tensor_overhead() + 16384;
    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
    opt->cache_ctx = ggml_init(ctx_params);
    if (!opt->cache_ctx) return;

    opt->caches.resize(opt->n_layer);
    opt->gate_scratch = nullptr;
    opt->up_scratch = nullptr;

    // Allocate scratch tensors for gate/up (shared across all layers)
    if (opt->use_scratch) {
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            if (!info.has_experts || info.on_gpu) continue;
            if (info.orig_exps[OP_GATE]) {
                opt->gate_scratch = ggml_new_tensor_3d(opt->cache_ctx,
                    info.orig_exps[OP_GATE]->type,
                    info.orig_exps[OP_GATE]->ne[0], info.orig_exps[OP_GATE]->ne[1],
                    opt->K);  // ne02=K: kernel needs ne02>=K, only slots 0-5 written
            }
            if (info.orig_exps[OP_UP]) {
                opt->up_scratch = ggml_new_tensor_3d(opt->cache_ctx,
                    info.orig_exps[OP_UP]->type,
                    info.orig_exps[OP_UP]->ne[0], info.orig_exps[OP_UP]->ne[1],
                    opt->K);  // ne02=K: kernel needs ne02>=K, only slots 0-5 written
            }
            ALPHA_LOG_DBG( "  scratch: gate %.1f MB + up %.1f MB (ne02=%d, slots 0-%d used)\n",
                gate_bytes * opt->K / 1048576.0,
                up_bytes * opt->K / 1048576.0, opt->K, opt->n_expert_used - 1);
            break;
        }
    }

    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];

        for (int op = 0; op < OP_COUNT; op++) {
            cache.gpu_tensor[op] = nullptr;
            memset(cache.slot_map[op], -1, sizeof(cache.slot_map[op]));
        }

        if (!info.has_experts || info.on_gpu) continue;

        if (opt->use_scratch) {
            // Scratch mode: only allocate the per-layer down cache
            ggml_tensor * ref = info.orig_exps[OP_DOWN];
            if (ref) {
                ggml_tensor * ct = ggml_new_tensor_3d(opt->cache_ctx, ref->type,
                    ref->ne[0], ref->ne[1], opt->K);
                ggml_set_name(ct, ref->name);  // preserve name for LoRA matching
                cache.gpu_tensor[OP_DOWN] = ct;
                cache.slots[OP_DOWN].resize(opt->K, { -1, 0 });
            }
            // Point gate/up to scratch tensors (refreshed per callback)
            cache.gpu_tensor[OP_GATE] = opt->gate_scratch;
            cache.gpu_tensor[OP_UP] = opt->up_scratch;
        } else {
            // Flat: allocate all ops per layer
            bool layer_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
            for (int op = 0; op < OP_COUNT; op++) {
                if (layer_fused && (op == OP_GATE || op == OP_UP)) continue;
                ggml_tensor * ref = info.orig_exps[op];
                if (!ref) continue;
                ggml_tensor * ct = ggml_new_tensor_3d(opt->cache_ctx, ref->type,
                    ref->ne[0], ref->ne[1], opt->K);
                // Preserve original tensor name for LoRA adapter matching.
                // build_lora_mm_id() looks up LoRA weights by tensor name.
                // Without this, cache tensors get generic names and LoRA
                // adapters for expert FFN (ffn_*_exps) won't be applied.
                ggml_set_name(ct, ref->name);
                cache.gpu_tensor[op] = ct;
                cache.slots[op].resize(opt->K, { -1, 0 });
            }
        }
    }

    // Create pre-computed ID tensors in same context (before GPU allocation)
    // Must fit the largest ubatch that will be processed (n_ubatch, typically 512)
    int max_tokens = 512;
    opt->cache_expert_ids.resize(opt->n_layer, nullptr);
    opt->cache_expert_ids_down.resize(opt->n_layer, nullptr);

    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;

        // Gate/up IDs (or single IDs for flat mode)
        ggml_tensor * ids = ggml_new_tensor_2d(opt->cache_ctx, GGML_TYPE_I32,
            opt->n_expert_used, max_tokens);
        ggml_set_name(ids, "cei");
        ggml_set_input(ids);
        opt->cache_expert_ids[il] = ids;

        // Down IDs (scratch/tiered mode: separate from gate/up)
        if (opt->use_scratch || opt->tiered_cache) {
            ggml_tensor * ids_down = ggml_new_tensor_2d(opt->cache_ctx, GGML_TYPE_I32,
                opt->n_expert_used, max_tokens);
            ggml_set_name(ids_down, "ced");
            ggml_set_input(ids_down);
            opt->cache_expert_ids_down[il] = ids_down;
        }

        opt->prev_routing[il].resize(opt->n_expert_used);
        for (int e = 0; e < opt->n_expert_used; e++) {
            opt->prev_routing[il][e] = e;
        }
    }

    // Allocate all tensors (cache + IDs) on GPU in one buffer
    ggml_backend_buffer_t cache_buf = ggml_backend_alloc_ctx_tensors_from_buft(opt->cache_ctx, gpu_buft);
    if (!cache_buf) {
        ALPHA_LOG_DBG( "  cache allocation failed (OOM)\n");
        ggml_free(opt->cache_ctx);
        opt->cache_ctx = nullptr;
        return;
    }
        opt->cache_bufs.emplace_back(cache_buf);
    opt->cache_enabled = true;

    // Zero-initialize all rows of the cache_expert_ids tensors. prepare() only writes row 0, but
    // multi-token prompt batches read every row via mul_mat_id, and garbage IDs >= K would make
    // the kernel read out of bounds.
    {
        std::vector<int32_t> zeros(opt->n_expert_used * 512, 0);  // max_tokens=512
        for (int il = 0; il < opt->n_layer; il++) {
            if (opt->cache_expert_ids[il]) {
                size_t bytes = ggml_nbytes(opt->cache_expert_ids[il]);
                ggml_backend_tensor_set(opt->cache_expert_ids[il], zeros.data(), 0,
                    std::min(bytes, zeros.size() * sizeof(int32_t)));
            }
            if (il < (int)opt->cache_expert_ids_down.size() && opt->cache_expert_ids_down[il]) {
                size_t bytes = ggml_nbytes(opt->cache_expert_ids_down[il]);
                ggml_backend_tensor_set(opt->cache_expert_ids_down[il], zeros.data(), 0,
                    std::min(bytes, zeros.size() * sizeof(int32_t)));
            }
        }
    }

    size_t cache_mb = ggml_backend_buffer_get_size(cache_buf) / (1024*1024);
    // fprintf(stderr, "  GPU cache allocated: %zu MB\n", cache_mb);

    // Pinned staging buffers: cudaMemcpyAsync from mmap'd (pageable) memory is synchronous, so
    // copy mmap -> pinned on the CPU, then cudaMemcpyAsync pinned -> GPU.
    {
        size_t max_slice = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            if (!info.has_experts || info.on_gpu) continue;
            for (int op = 0; op < OP_COUNT; op++) {
                if (info.expert_slice_bytes[op] > max_slice)
                    max_slice = info.expert_slice_bytes[op];
            }
        }
        if (max_slice > 0) {
            max_slice = (max_slice + 4095) & ~(size_t)4095;  // 4KB align
            // Worst case DMAs per callback: n_expert_used x 3 ops (flat mode)
            int max_dma = opt->n_expert_used * 3;
            size_t total_pinned = (size_t)max_dma * max_slice;
            if (total_pinned > 128ULL * 1024 * 1024) {
                max_dma = (int)(128ULL * 1024 * 1024 / max_slice);
            }
            if (max_dma < 3) max_dma = 3;

            opt->staging_bufs = (void **)calloc(max_dma, sizeof(void *));
            opt->staging_buf_count = max_dma;
            opt->staging_buf_size = max_slice;
            opt->staging_enabled = true;
            for (int i = 0; i < max_dma; i++) {
                opt->staging_bufs[i] = _acpa(max_slice);
                if (!opt->staging_bufs[i]) {
                    ALPHA_LOG_DBG("[STAGING] cudaMallocHost failed for buf %d/%d (%zu bytes)\n",
                            i, max_dma, max_slice);
                    for (int j = 0; j < i; j++) {
                        _acpf(opt->staging_bufs[j]);
                        opt->staging_bufs[j] = nullptr;
                    }
                    free(opt->staging_bufs);
                    opt->staging_bufs = nullptr;
                    opt->staging_enabled = false;
                    break;
                }
            }
            if (opt->staging_enabled) {
                ALPHA_LOG_DBG( "  pinned staging: %d buffers × %.1f MB = %.0f MB (truly async DMA)\n",
                        max_dma, (float)max_slice / (1024*1024),
                        (float)max_dma * max_slice / (1024*1024));
            }
        }
    }

    // Pre-load first K experts + zero slot 0
    fflush(stderr);

    auto preload_cache = [&](layer_cache & c, int op, ggml_tensor * ref, size_t sb, int il = -1) {
        if (!ref || !ref->data || !c.gpu_tensor[op]) return;
        if (c.slots[op].empty()) return;
        // Check for profiled expert IDs for this layer
        const int * prof_ids = nullptr;
        int prof_count = 0;
        if (il >= 0 && il < (int)opt->preload_expert_ids.size() &&
            !opt->preload_expert_ids[il].empty()) {
            prof_ids = opt->preload_expert_ids[il].data();
            prof_count = (int)opt->preload_expert_ids[il].size();
        }
        // Load experts into slots 1..K-1 (slot 0 = zero expert)
        for (int k = 1; k < opt->K && k < opt->n_expert; k++) {
            int eid;
            if (prof_ids && (k - 1) < prof_count) {
                eid = prof_ids[k - 1];  // profiled: most frequent first
            } else {
                eid = k - 1;  // fallback: sequential
            }
            if (eid < 0 || eid >= opt->n_expert) eid = k - 1;
            ggml_backend_tensor_set(c.gpu_tensor[op],
                (const char *)ref->data + eid * sb, k * sb, sb);
            c.slots[op][k] = { eid, ++opt->cache_clock };
            c.slot_map[op][eid] = k;
        }
        // Zero slot 0
        std::vector<char> zeros(sb, 0);
        ggml_backend_tensor_set(c.gpu_tensor[op], zeros.data(), 0, sb);
        c.slots[op][0] = { -2, INT64_MAX };
    };

    bool has_profiled_preload = !opt->preload_expert_ids.empty();
    if (has_profiled_preload) {
        ALPHA_LOG_DBG( "  using profiled expert pre-load (%d layers configured)\n",
                (int)opt->preload_expert_ids.size());
    }

    if (opt->use_scratch) {
        // Scratch mode: only pre-load per-layer down (gate/up refreshed per callback)
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            preload_cache(cache, OP_DOWN, info.orig_exps[OP_DOWN], info.expert_slice_bytes[OP_DOWN], il);
        }
    } else if (opt->tiered_cache) {
        // Pre-load global gate/up (once, from first MoE layer)
        auto & gcache = opt->global_gate_up_cache;
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            if (!info.has_experts || info.on_gpu) continue;
            preload_cache(gcache, OP_GATE, info.orig_exps[OP_GATE], info.expert_slice_bytes[OP_GATE], il);
            preload_cache(gcache, OP_UP, info.orig_exps[OP_UP], info.expert_slice_bytes[OP_UP], il);
            break;
        }
        // Pre-load per-layer down
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            preload_cache(cache, OP_DOWN, info.orig_exps[OP_DOWN], info.expert_slice_bytes[OP_DOWN], il);
        }
    } else {
        // Flat: pre-load all ops per layer
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            bool fused_pl = (info.orig_exps[OP_GATE_UP] != nullptr);
            for (int op = 0; op < OP_COUNT; op++) {
                if (fused_pl && (op == OP_GATE || op == OP_UP)) continue;
                preload_cache(cache, op, info.orig_exps[op], info.expert_slice_bytes[op], il);
            }
        }
    }
    fflush(stderr);

    // Initialize pre-computed ID tensors (already allocated on GPU in cache_ctx)
    {
        std::vector<int32_t> init_ids(opt->n_expert_used * max_tokens);
        for (int t = 0; t < max_tokens; t++) {
            for (int e = 0; e < opt->n_expert_used; e++) {
                init_ids[t * opt->n_expert_used + e] = e % opt->K;
            }
        }
        for (int il = 0; il < opt->n_layer; il++) {
            ggml_tensor * ids = opt->cache_expert_ids[il];
            if (!ids) continue;
            ggml_backend_tensor_set(ids, init_ids.data(), 0,
                init_ids.size() * sizeof(int32_t));
            opt->model->layers[il].ffn_cache_expert_ids = ids;

            if ((opt->use_scratch || opt->tiered_cache) && il < (int)opt->cache_expert_ids_down.size()) {
                ggml_tensor * ids_down = opt->cache_expert_ids_down[il];
                if (ids_down) {
                    ggml_backend_tensor_set(ids_down, init_ids.data(), 0,
                        init_ids.size() * sizeof(int32_t));
                    opt->model->layers[il].ffn_cache_expert_ids_down = ids_down;
                }
            }
        }
    }
    fflush(stderr);

    // Pre-load first K experts
    if (opt->tiered_cache) {
        auto & gc = opt->global_gate_up_cache;
        // Pre-load global gate/up (once, from first MoE layer)
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            if (!info.has_experts || info.on_gpu) continue;
            for (int op : { OP_GATE, OP_UP }) {
                ggml_tensor * ref = info.orig_exps[op];
                if (!ref || !ref->data || !gc.gpu_tensor[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                for (int k = 1; k < opt->K && k < opt->n_expert; k++) {
                    int eid = k - 1;
                    ggml_backend_tensor_set(gc.gpu_tensor[op],
                        (const char *)ref->data + eid * sb, k * sb, sb);
                    gc.slots[op][k] = { eid, ++opt->cache_clock };
                    gc.slot_map[op][eid] = k;
                }
                // Zero slot 0
                std::vector<char> zeros(sb, 0);
                ggml_backend_tensor_set(gc.gpu_tensor[op], zeros.data(), 0, sb);
                gc.slots[op][0] = { -2, INT64_MAX };
            }
            break;
        }
        // Pre-load per-layer down
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            ggml_tensor * ref = info.orig_exps[OP_DOWN];
            if (!ref || !ref->data || !cache.gpu_tensor[OP_DOWN]) continue;
            size_t sb = info.expert_slice_bytes[OP_DOWN];
            for (int k = 1; k < opt->K && k < opt->n_expert; k++) {
                int eid = k - 1;
                ggml_backend_tensor_set(cache.gpu_tensor[OP_DOWN],
                    (const char *)ref->data + eid * sb, k * sb, sb);
                cache.slots[OP_DOWN][k] = { eid, ++opt->cache_clock };
                cache.slot_map[OP_DOWN][eid] = k;
            }
            // Zero slot 0
            std::vector<char> zeros(sb, 0);
            ggml_backend_tensor_set(cache.gpu_tensor[OP_DOWN], zeros.data(), 0, sb);
            cache.slots[OP_DOWN][0] = { -2, INT64_MAX };
        }
    } else if (opt->use_scratch) {
        // Scratch mode: only pre-load per-layer down cache (gate/up scratch refreshed per callback)
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            ggml_tensor * ref = info.orig_exps[OP_DOWN];
            if (!ref || !ref->data || !cache.gpu_tensor[OP_DOWN]) continue;
            size_t sb = info.expert_slice_bytes[OP_DOWN];
            for (int k = 1; k < opt->K && k < opt->n_expert; k++) {
                int eid = k - 1;
                ggml_backend_tensor_set(cache.gpu_tensor[OP_DOWN],
                    (const char *)ref->data + eid * sb, k * sb, sb);
                cache.slots[OP_DOWN][k] = { eid, ++opt->cache_clock };
                cache.slot_map[OP_DOWN][eid] = k;
            }
            // Zero slot 0
            std::vector<char> zeros(sb, 0);
            ggml_backend_tensor_set(cache.gpu_tensor[OP_DOWN], zeros.data(), 0, sb);
            cache.slots[OP_DOWN][0] = { -2, INT64_MAX };
        }
    } else {
        // Flat pre-load
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            bool fused_pl = (info.orig_exps[OP_GATE_UP] != nullptr);
            for (int op = 0; op < OP_COUNT; op++) {
                if (fused_pl && (op == OP_GATE || op == OP_UP)) continue;
                ggml_tensor * ref = info.orig_exps[op];
                if (!ref || !ref->data || !cache.gpu_tensor[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                // Load experts 0..K-2 into slots 1..K-1 (slot 0 = zero expert)
                for (int k = 1; k < opt->K && k < opt->n_expert; k++) {
                    int eid = k - 1;
                    ggml_backend_tensor_set(cache.gpu_tensor[op],
                        (const char *)ref->data + eid * sb, k * sb, sb);
                    cache.slots[op][k] = { eid, ++opt->cache_clock };
                    cache.slot_map[op][eid] = k;
                }
                // Zero slot 0
                std::vector<char> zeros(sb, 0);
                ggml_backend_tensor_set(cache.gpu_tensor[op], zeros.data(), 0, sb);
                cache.slots[op][0] = { -2, INT64_MAX };
            }
        }
    }
    ALPHA_LOG_DBG( "  reserved slot 0 as zero-expert for FFN skip\n");
    fflush(stderr);

    // Prefault the most frequently used experts (per the frequency profile) into the OS page
    // cache with madvise(WILLNEED), so later reads run at RAM speed instead of disk speed.
#ifndef _WIN32
    {
        int n_prefault = 0;
        size_t prefault_bytes = 0;
        // How many experts can fit in ~60% of system RAM (leave room for OS + buffers)
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        size_t ram_bytes = (size_t)pages * page_size;
        size_t budget = (size_t)(ram_bytes * 0.6);  // 60% of RAM for page cache

        for (int il = 0; il < opt->n_layer && prefault_bytes < budget; il++) {
            auto & info = opt->layer_info[il];
            if (!info.has_experts || info.on_gpu) continue;

            // Determine expert order: frequency-sorted if available, else sequential
            const int * freq_order = nullptr;
            int freq_count = 0;
            if (il < (int)opt->preload_expert_ids.size() && !opt->preload_expert_ids[il].empty()) {
                freq_order = opt->preload_expert_ids[il].data();
                freq_count = (int)opt->preload_expert_ids[il].size();
            }

            // Prefault experts for each op tensor
            for (int op = 0; op < OP_COUNT; op++) {
                ggml_tensor * t = info.orig_exps[op];
                if (!t || !t->data) continue;
                size_t sb = info.expert_slice_bytes[op];
                if (sb == 0) continue;

                int n_to_prefault = freq_count > 0 ? freq_count : std::min(opt->n_expert, 200);
                for (int k = 0; k < n_to_prefault && prefault_bytes < budget; k++) {
                    int eid = freq_order ? freq_order[k] : k;
                    if (eid < 0 || eid >= opt->n_expert) continue;
                    void * addr = (char *)t->data + (size_t)eid * sb;
                    // Align to page boundary
                    uintptr_t page_addr = (uintptr_t)addr & ~(uintptr_t)(page_size - 1);
                    size_t len = sb + ((uintptr_t)addr - page_addr);
                    madvise((void *)page_addr, len, MADV_WILLNEED);
                    prefault_bytes += sb;
                    n_prefault++;
                }
            }
        }
        if (n_prefault > 0) {
            ALPHA_LOG_DBG( "  page cache prefault: %d expert slices (%.1f GB) via madvise(WILLNEED)\n",
                    n_prefault, prefault_bytes / (1024.0 * 1024 * 1024));
        }
    }
#endif

    // Swap model expert tensor pointers to GPU cache tensors
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;

        auto & layer = opt->model->layers[il];
        if (opt->use_scratch) {
            if (cache.gpu_tensor[OP_DOWN])  layer.ffn_down_exps = cache.gpu_tensor[OP_DOWN];
            if (opt->gate_scratch)          layer.ffn_gate_exps = opt->gate_scratch;
            if (opt->up_scratch)            layer.ffn_up_exps   = opt->up_scratch;
        } else if (opt->tiered_cache) {
            if (cache.gpu_tensor[OP_DOWN]) layer.ffn_down_exps = cache.gpu_tensor[OP_DOWN];
            auto & gc = opt->global_gate_up_cache;
            if (gc.gpu_tensor[OP_GATE]) layer.ffn_gate_exps = gc.gpu_tensor[OP_GATE];
            if (gc.gpu_tensor[OP_UP])   layer.ffn_up_exps   = gc.gpu_tensor[OP_UP];
        } else {
            if (cache.gpu_tensor[OP_GATE])    layer.ffn_gate_exps    = cache.gpu_tensor[OP_GATE];
            if (cache.gpu_tensor[OP_UP])      layer.ffn_up_exps      = cache.gpu_tensor[OP_UP];
            if (cache.gpu_tensor[OP_DOWN])    layer.ffn_down_exps    = cache.gpu_tensor[OP_DOWN];
            if (cache.gpu_tensor[OP_GATE_UP]) layer.ffn_gate_up_exps = cache.gpu_tensor[OP_GATE_UP];
        }
        if (opt->cache_expert_ids[il]) {
            layer.ffn_cache_expert_ids = opt->cache_expert_ids[il];
        }
        if ((opt->use_scratch || opt->tiered_cache) &&
            il < (int)opt->cache_expert_ids_down.size() && opt->cache_expert_ids_down[il]) {
            layer.ffn_cache_expert_ids_down = opt->cache_expert_ids_down[il];
        }
    }

    // Re-reserve scheduler to see GPU expert tensors
    if (opt->ctx) {
        llama_context_reset_sched(opt->ctx);
    }
}

// Mixed Expert Clustering

#include <fstream>

bool alpha_moe_optimizer_load_mixed_experts(alpha_moe_optimizer * opt, const char * cluster_json) {
    if (!opt || !opt->is_moe || !opt->cache_enabled) return false;

    // Parse cluster JSON - minimal parser for our specific format
    std::ifstream ifs(cluster_json);
    if (!ifs.is_open()) {
        ALPHA_LOG_DBG("[MIXED] Cannot open cluster file: %s\n", cluster_json);
        return false;
    }
    std::string json_str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    // Extract n_clusters
    auto find_int = [&](const std::string & s, const std::string & key) -> int {
        size_t pos = s.find("\"" + key + "\"");
        if (pos == std::string::npos) return -1;
        pos = s.find(':', pos);
        if (pos == std::string::npos) return -1;
        return atoi(s.c_str() + pos + 1);
    };

    int n_clusters = find_int(json_str, "nc");
    if (n_clusters <= 0 || n_clusters > opt->K - 1) {
        ALPHA_LOG_DBG("[MIXED] n_clusters=%d invalid (need 1..%d, K=%d)\n", n_clusters, opt->K - 1, opt->K);
        return false;
    }

    opt->mixed_n_clusters = n_clusters;
    opt->mixed_expert_to_cluster.resize(opt->n_layer);
    opt->mixed_cluster_rep.resize(opt->n_layer);
    opt->mixed_cluster_members.resize(opt->n_layer);

    // Parse per-layer expert_to_cluster, representatives, and cluster_to_experts arrays
    for (int il = 0; il < opt->n_layer; il++) {
        opt->mixed_expert_to_cluster[il].resize(opt->n_expert, 0);
        opt->mixed_cluster_rep[il].resize(n_clusters, 0);
        opt->mixed_cluster_members[il].resize(n_clusters);

        // Find layer section: "il": { ... }
        char layer_key[32];
        snprintf(layer_key, sizeof(layer_key), "\"%d\":", il);
        size_t layer_pos = json_str.find(layer_key);
        if (layer_pos == std::string::npos) continue;

        // Parse expert_to_cluster array
        size_t etc_pos = json_str.find("\"e2c\"", layer_pos);
        if (etc_pos != std::string::npos) {
            size_t arr_start = json_str.find('[', etc_pos);
            size_t arr_end = json_str.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = json_str.substr(arr_start + 1, arr_end - arr_start - 1);
                int idx = 0;
                const char * p = arr.c_str();
                while (*p && idx < opt->n_expert) {
                    while (*p && (*p == ' ' || *p == ',')) p++;
                    if (!*p) break;
                    opt->mixed_expert_to_cluster[il][idx++] = atoi(p);
                    while (*p && *p != ',' && *p != ']') p++;
                }
            }
        }

        // Parse representatives array
        size_t rep_pos = json_str.find("\"rp\"", layer_pos);
        if (rep_pos != std::string::npos) {
            size_t arr_start = json_str.find('[', rep_pos);
            size_t arr_end = json_str.find(']', arr_start);
            if (arr_start != std::string::npos && arr_end != std::string::npos) {
                std::string arr = json_str.substr(arr_start + 1, arr_end - arr_start - 1);
                int idx = 0;
                const char * p = arr.c_str();
                while (*p && idx < n_clusters) {
                    while (*p && (*p == ' ' || *p == ',')) p++;
                    if (!*p) break;
                    opt->mixed_cluster_rep[il][idx++] = atoi(p);
                    while (*p && *p != ',' && *p != ']') p++;
                }
            }
        }

        // Parse cluster_to_experts: [[e1,e2,...], [e3,e4,...], ...]
        size_t cte_pos = json_str.find("\"c2e\"", layer_pos);
        if (cte_pos != std::string::npos) {
            size_t outer_start = json_str.find('[', cte_pos);
            if (outer_start != std::string::npos) {
                int cid = 0;
                size_t pos = outer_start + 1;
                while (cid < n_clusters && pos < json_str.size()) {
                    size_t inner_start = json_str.find('[', pos);
                    if (inner_start == std::string::npos) break;
                    size_t inner_end = json_str.find(']', inner_start);
                    if (inner_end == std::string::npos) break;
                    std::string inner = json_str.substr(inner_start + 1, inner_end - inner_start - 1);
                    const char * p = inner.c_str();
                    while (*p) {
                        while (*p && (*p == ' ' || *p == ',')) p++;
                        if (!*p) break;
                        int eid = atoi(p);
                        if (eid >= 0 && eid < opt->n_expert) {
                            opt->mixed_cluster_members[il][cid].push_back(eid);
                        }
                        while (*p && *p != ',' && *p != ']') p++;
                    }
                    cid++;
                    pos = inner_end + 1;
                }
            }
        }
    }

    // Merge cluster member experts into cache slots 1..n_clusters
    // For each cluster: dequantize all member experts to F32, average, requantize, write to GPU
    // Slot 0 = zero expert (already set by alloc_cache)
    // Slot c+1 = cluster c's merged expert
    //
    // The merged data is cached in a .bin file next to the JSON and loaded directly on later runs.
    std::string cache_path = std::string(cluster_json);
    // Replace .json with .bin for cache file
    if (cache_path.size() > 5 && cache_path.substr(cache_path.size()-5) == ".json") {
        cache_path = cache_path.substr(0, cache_path.size()-5) + ".bin";
    } else {
        cache_path += ".bin";
    }

    // Try loading from cache file
    FILE * cache_file = fopen(cache_path.c_str(), "rb");
    if (cache_file) {
        ALPHA_LOG_DBG("[MIXED] Loading pre-merged data from %s\n", cache_path.c_str());

        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;

            bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);

            for (int c = 0; c < n_clusters; c++) {
                int slot = c + 1;
                if (slot >= opt->K) break;

                for (int op = 0; op < OP_COUNT; op++) {
                    if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                    if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;

                    size_t sb = info.expert_slice_bytes[op];
                    std::vector<uint8_t> buf(sb);
                    if (fread(buf.data(), 1, sb, cache_file) != sb) {
                        ALPHA_LOG_DBG("[MIXED] Cache file truncated at layer %d cluster %d op %d\n", il, c, op);
                        fclose(cache_file);
                        goto do_merge;  // fall through to full merge
                    }
                    ggml_backend_tensor_set(cache.gpu_tensor[op], buf.data(), (size_t)slot * sb, sb);

                    int rep_eid = opt->mixed_cluster_rep[il][c];
                    cache.slots[op][slot] = { rep_eid, ++opt->cache_clock };
                    cache.slot_map[op][rep_eid] = slot;
                }
            }
        }
        fclose(cache_file);
        ALPHA_LOG_DBG("[MIXED] Loaded pre-merged data (skipped merge)\n");
        goto after_merge;
    }

do_merge:
    {
    ALPHA_LOG_DBG("[MIXED] Merging %d clusters into cache (K=%d)...\n",
            n_clusters, opt->K);

    // Open cache file for writing merged data
    FILE * save_file = fopen(cache_path.c_str(), "wb");

    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;

        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);

        for (int c = 0; c < n_clusters; c++) {
            int slot = c + 1;
            if (slot >= opt->K) break;

            auto & members = opt->mixed_cluster_members[il][c];
            if (members.empty()) continue;

            // Single member - just copy directly (no merge needed)
            if (members.size() == 1) {
                int eid = members[0];
                for (int op = 0; op < OP_COUNT; op++) {
                    if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                    if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                    size_t sb = info.expert_slice_bytes[op];
                    const void * src = (const char *)info.orig_exps[op]->data + (size_t)eid * sb;
                    ggml_backend_tensor_set(cache.gpu_tensor[op], src, (size_t)slot * sb, sb);
                    if (save_file) fwrite(src, 1, sb, save_file);
                    cache.slots[op][slot] = { eid, ++opt->cache_clock };
                    cache.slot_map[op][eid] = slot;
                }
                continue;
            }

            // Multiple members - merge via dequant -> average -> requant
            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;

                size_t sb = info.expert_slice_bytes[op];
                ggml_type qtype = info.orig_exps[op]->type;
                auto * traits = ggml_get_type_traits(qtype);

                ggml_tensor * t = info.orig_exps[op];
                int64_t n_elements = (int64_t)(t->ne[0]) * t->ne[1];

                std::vector<float> merged(n_elements, 0.0f);
                std::vector<float> temp(n_elements);

                int n_members = (int)members.size();
                for (int m = 0; m < n_members; m++) {
                    int eid = members[m];
                    const void * src = (const char *)t->data + (size_t)eid * sb;
                    if (traits->to_float) {
                        traits->to_float(src, temp.data(), n_elements);
                    } else {
                        memcpy(temp.data(), src, std::min<size_t>(sb, n_elements * sizeof(float)));
                    }
                    float w = 1.0f / n_members;
                    for (int64_t i = 0; i < n_elements; i++) {
                        merged[i] += temp[i] * w;
                    }
                }

                std::vector<uint8_t> quant_buf(sb + 1024);
                ggml_quantize_chunk(qtype, merged.data(), quant_buf.data(), 0, t->ne[1], t->ne[0], nullptr);

                ggml_backend_tensor_set(cache.gpu_tensor[op], quant_buf.data(), (size_t)slot * sb, sb);
                if (save_file) fwrite(quant_buf.data(), 1, sb, save_file);

                int rep_eid = opt->mixed_cluster_rep[il][c];
                cache.slots[op][slot] = { rep_eid, ++opt->cache_clock };
                cache.slot_map[op][rep_eid] = slot;
            }
        }

        if (il % 10 == 0) {
            ALPHA_LOG_DBG( "  [MIXED] Layer %d/%d merged\n", il, opt->n_layer);
        }
    }

    if (save_file) {
        fclose(save_file);
        ALPHA_LOG_DBG("[MIXED] Saved merged data to %s\n", cache_path.c_str());
    }
    } // end do_merge block

after_merge:

    // Create mixed LUT: expert_id -> cluster_slot (1-based)
    // Reuses spec_lut infrastructure
    if (!opt->spec_lut_ready) {
        alpha_moe_optimizer_init_spec_lut(opt);
    }

    std::vector<int32_t> lut_data(opt->n_expert);
    for (int il = 0; il < opt->n_layer; il++) {
        if (!opt->spec_lut[il]) continue;
        for (int e = 0; e < opt->n_expert; e++) {
            int cluster_id = opt->mixed_expert_to_cluster[il][e];
            lut_data[e] = cluster_id + 1;  // 1-based (0 = zero expert)
        }
        ggml_backend_tensor_set(opt->spec_lut[il], lut_data.data(), 0,
            opt->n_expert * sizeof(int32_t));
    }

    opt->mixed_experts_enabled = true;
    ALPHA_LOG_DBG("[MIXED] Loaded: %d clusters, %d layers, LUT ready\n",
            n_clusters, opt->n_layer);
    return true;
}

void alpha_moe_optimizer_enter_mixed_draft_mode(alpha_moe_optimizer * opt) {
    if (!opt || !opt->mixed_experts_enabled || !opt->cache_enabled) return;

    // Save original expert tensor pointers
    opt->mixed_saved_ptrs.resize(opt->n_layer);
    for (int il = 0; il < opt->n_layer; il++) {
        auto & layer = opt->model->layers[il];
        opt->mixed_saved_ptrs[il] = {
            layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps,
            layer.ffn_gate_up_exps, layer.ffn_cache_expert_ids,
            layer.ffn_cache_expert_ids_down
        };

        auto & info = opt->layer_info[il];
        if (!info.has_experts) continue;

        // Swap expert tensors to cache tensors (containing cluster representatives)
        auto & cache = opt->caches[il];
        if (cache.gpu_tensor[OP_GATE])    layer.ffn_gate_exps    = cache.gpu_tensor[OP_GATE];
        if (cache.gpu_tensor[OP_UP])      layer.ffn_up_exps      = cache.gpu_tensor[OP_UP];
        if (cache.gpu_tensor[OP_DOWN])    layer.ffn_down_exps    = cache.gpu_tensor[OP_DOWN];
        if (cache.gpu_tensor[OP_GATE_UP]) layer.ffn_gate_up_exps = cache.gpu_tensor[OP_GATE_UP];

        // Wire mixed LUT for 256->cluster remapping
        if (il < (int)opt->spec_lut.size() && opt->spec_lut[il]) {
            layer.ffn_cache_expert_ids = opt->spec_lut[il];
            layer.ffn_cache_expert_ids_down = nullptr;
        }
    }

    // Save whether callback was active, then disable (all experts on GPU, zero DMA)
    // With cpu-moe, callback was never set - don't re-enable on exit
    opt->callback_was_enabled = opt->eval_callback_active;
    if (opt->ctx) {
        llama_set_eval_callback(opt->ctx, nullptr, nullptr);
    }
    opt->spec_draft_active = true;

    // Mark argsort as output so reload_from_argsort can read valid data after decode.
    // Without this, ggml allocator reuses argsort buffers -> garbage -> verify never fires.
    extern std::atomic<bool> g_alpha_argsort_persist;
    g_alpha_argsort_persist.store(true, std::memory_order_relaxed);

    // Boost shared expert to compensate for approximated routed experts
    extern thread_local float g_alpha_mixed_shexp_scale;
    g_alpha_mixed_shexp_scale = 1.3f;

    // Force graph rebuild (new tensor shapes)
    if (opt->ctx) {
        llama_context_reset_sched(opt->ctx);
    }

    ALPHA_LOG_DBG("[MIXED] Entered mixed draft mode (%d clusters, shexp_scale=%.1f)\n",
            opt->mixed_n_clusters, g_alpha_mixed_shexp_scale);
}

void alpha_moe_optimizer_exit_mixed_draft_mode(alpha_moe_optimizer * opt) {
    if (!opt || opt->mixed_saved_ptrs.empty()) return;

    // Restore original expert tensor pointers
    for (int il = 0; il < opt->n_layer && il < (int)opt->mixed_saved_ptrs.size(); il++) {
        auto & layer = opt->model->layers[il];
        auto & saved = opt->mixed_saved_ptrs[il];
        layer.ffn_gate_exps         = saved.gate;
        layer.ffn_up_exps           = saved.up;
        layer.ffn_down_exps         = saved.down;
        layer.ffn_gate_up_exps      = saved.gate_up;
        layer.ffn_cache_expert_ids  = saved.cache_ids;
        layer.ffn_cache_expert_ids_down = saved.cache_ids_down;
    }

    // Re-enable callback only if it was active before entering mixed draft
    if (opt->callback_was_enabled) {
        auto saved_cb = alpha_moe_optimizer_get_eval_callback();
        auto saved_cb_data = alpha_moe_optimizer_get_eval_data(opt);
        if (opt->ctx && saved_cb) {
            llama_set_eval_callback(opt->ctx, saved_cb, saved_cb_data);
        }
    }
    opt->spec_draft_active = false;

    // Restore shared expert scale
    extern thread_local float g_alpha_mixed_shexp_scale;
    g_alpha_mixed_shexp_scale = 1.0f;

    // Force graph rebuild
    if (opt->ctx) {
        llama_context_reset_sched(opt->ctx);
    }
}

bool alpha_moe_optimizer_has_mixed_experts(const alpha_moe_optimizer * opt) {
    return opt && opt->mixed_experts_enabled;
}

void alpha_moe_optimizer_set_callback_active(alpha_moe_optimizer * opt, bool active) {
    if (opt) opt->eval_callback_active = active;
}

// Phase 1, prepare: reset the reactive counter (the reactive callback does the actual loading)
void alpha_moe_optimizer_prepare(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe) return;

    // Reset reactive layer counter (callback increments per MoE layer)
    // Must reset even when cache disabled (override in direct mode uses it)
    opt->reactive_layer_counter = 0;
    // Compute per-token miss rate from prev_routing (for miss-rate quality gate).
    // Uses previous token's routing as predictor for current token's cache coverage.
    // Lightweight: just checks slot_map, no GPU reads.
    // During LUT prefill, the callback tracks actual misses (slot_map lookup).
    // Don't overwrite with prediction - just reset for callback to fill.
    if (opt->lut_prefill_active) {
        opt->last_token_miss_layers = 0;
        opt->last_token_total_layers = 0;
    } else if (opt->cache_enabled && opt->spec_draft_active) {
        int miss_layers = 0, total_layers = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                         (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                         (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));
            if (cache.slots[ref_op].empty()) continue;
            total_layers++;
            bool any_miss = false;
            for (int e = 0; e < opt->n_expert_used && e < (int)opt->prev_routing[il].size(); e++) {
                int eid = opt->prev_routing[il][e];
                if (eid >= 0 && eid < opt->n_expert && cache.slot_map[ref_op][eid] < 0) {
                    any_miss = true;
                    break;
                }
            }
            if (any_miss) miss_layers++;
        }
        opt->last_token_miss_layers = miss_layers;
        opt->last_token_total_layers = total_layers;
    } else {
        opt->last_token_miss_layers = 0;
        opt->last_token_total_layers = 0;
    }

    if (!opt->cache_enabled) return;

    // Do not clear argsort here: with graph caching build_moe_ffn() does not rewrite the pointers
    // on every decode. The clear is in the llm_graph_context constructor, so it only runs when the
    // graph is rebuilt.

    // prepare() DMA pre-loads predicted experts for temporal locality.
    // The reactive callback corrects any mispredictions during the forward pass.

    // Hybrid predict+react: prediction warms cache (speed), callback guarantees correctness.
    // Even when the predicted set rarely matches (256-expert models), loading prev_routing gives
    // temporal locality, since adjacent tokens share some experts, and keeps non-hot slots fresh.

    // Tiered cache: prepare() only handles down (per-layer), gate/up managed by callback
    // Skip prepare() DMA entirely in tiered mode - the reactive callback handles everything
    if (opt->tiered_cache) return;

    // Track what we predict per layer (for layer consistency bias on next layer)
    std::vector<int> prev_layer_predicted;  // experts predicted for the previous layer

    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        ggml_tensor * ids_tensor = opt->cache_expert_ids[il];
        if (!info.has_experts || info.on_gpu || !ids_tensor) continue;

        // Build prediction: prev_routing + sticky momentum + layer consistency
        std::vector<int> predicted(opt->n_expert_used);
        bool use_scoring = (opt->config.expert_sticky > 0.0f || opt->config.expert_layer_bias > 0.0f);

        if (use_scoring) {
            // Score all experts and pick top-K
            std::vector<std::pair<float, int>> scores(opt->n_expert);
            for (int e = 0; e < opt->n_expert; e++) {
                float score = 0.0f;

                // Base: was this expert in prev_routing?
                for (int p = 0; p < opt->n_expert_used; p++) {
                    if (opt->prev_routing[il][p] == e) {
                        score += 1.0f;
                        break;
                    }
                }

                // Sticky momentum: boost recently active experts
                if (opt->config.expert_sticky > 0.0f &&
                    il < (int)opt->expert_momentum.size()) {
                    score += opt->config.expert_sticky * opt->expert_momentum[il][e];
                }

                // Layer consistency: boost experts predicted for previous layer
                if (opt->config.expert_layer_bias > 0.0f && !prev_layer_predicted.empty()) {
                    for (int plp : prev_layer_predicted) {
                        if (plp == e) {
                            score += opt->config.expert_layer_bias;
                            break;
                        }
                    }
                }

                // Group affinity: small boost for experts in the same group as
                // prev_routing experts (they'll be loaded together anyway)
                if (opt->groups_enabled && il < (int)opt->expert_grouping.size()
                    && !opt->expert_grouping[il].groups.empty()) {
                    auto & lg = opt->expert_grouping[il];
                    int my_group = (e < (int)lg.expert_to_group.size()) ? lg.expert_to_group[e] : -1;
                    if (my_group >= 0) {
                        for (int p = 0; p < opt->n_expert_used; p++) {
                            int pe = opt->prev_routing[il][p];
                            if (pe >= 0 && pe < (int)lg.expert_to_group.size()
                                && lg.expert_to_group[pe] == my_group) {
                                score += 0.1f;  // small group affinity boost
                                break;
                            }
                        }
                    }
                }

                scores[e] = {score, e};
            }

            // Partial sort: top n_expert_used by score descending
            std::partial_sort(scores.begin(),
                scores.begin() + opt->n_expert_used,
                scores.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });

            for (int e = 0; e < opt->n_expert_used; e++) {
                predicted[e] = scores[e].second;
            }
        } else if (!opt->prediction_reliable &&
                   il < (int)opt->expert_activation_count.size() &&
                   !opt->expert_activation_count[il].empty()) {
            // Frequency-based prediction for unreliable prev_routing (256-expert models).
            // Use per-layer activation counts to pick the most frequently used experts.
            // These are more likely to be needed than prev_routing (which has 0% set_match).
            std::vector<std::pair<int64_t, int>> freq(opt->n_expert);
            for (int e = 0; e < opt->n_expert; e++) {
                freq[e] = {opt->expert_activation_count[il][e], e};
            }
            std::partial_sort(freq.begin(),
                freq.begin() + opt->n_expert_used,
                freq.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int e = 0; e < opt->n_expert_used; e++) {
                predicted[e] = freq[e].second;
            }
        } else {
            // Default: use prev_routing directly (zero overhead when features disabled)
            predicted = opt->prev_routing[il];
        }

        // Save this layer's prediction for next layer's consistency bias
        prev_layer_predicted = predicted;

#if !defined(_WIN32)
        // Prefetch predicted expert pages from mmap (async kernel readahead).
        // This overlaps disk I/O with the GPU compute for the previous layer,
        // turning synchronous page faults into pre-warmed cache hits.
        for (int op = 0; op < OP_COUNT; op++) {
            ggml_tensor * t = info.orig_exps[op];
            if (!t || !t->data || !ggml_backend_buffer_is_host(t->buffer)) continue;
            size_t sb = info.expert_slice_bytes[op];
            if (sb == 0) continue;
            for (int e = 0; e < opt->n_expert_used; e++) {
                int eid = predicted[e];
                if (eid < 0 || eid >= opt->n_expert) continue;
                void * addr = (char *)t->data + (size_t)eid * sb;
                uintptr_t pa = (uintptr_t)addr & ~(uintptr_t)4095;
                madvise((void *)pa, sb + ((uintptr_t)addr - pa), MADV_WILLNEED);
            }
        }
#endif

        int layer_loads = 0;  // count storage reads this token (mobile: max 1)

        // Check if expert should be protected from eviction:
        // hot globally, hot per-layer, or used in recent tokens
        auto is_protected = [&](int eid) -> bool {
            // Global hot experts
            for (int h : opt->hot_experts) {
                if (h == eid) return true;
            }
            // Per-layer hot experts (from activation profiling)
            if (il < (int)opt->hot_per_layer.size()) {
                for (int h : opt->hot_per_layer[il]) {
                    if (h == eid) return true;
                }
            }
            // Recently used (temporal locality)
            if (il < (int)opt->recent_routing.size()) {
                for (int d = 0; d < alpha_moe_optimizer::TEMPORAL_DEPTH; d++) {
                    for (int re : opt->recent_routing[il][d]) {
                        if (re == eid) return true;
                    }
                }
            }
            return false;
        };

        // For each predicted expert: ensure it's in cache (respecting thermal policy)
        int32_t remapped[16] = {};  // max n_expert_used
        for (int e = 0; e < opt->n_expert_used && e < 16; e++) {
            int eid = predicted[e];
            if (eid < 0 || eid >= opt->n_expert) eid = 0;

            // Check if cached (scratch/tiered = check down only, flat = check gate/gate_up)
            int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                         (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                         (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));
            int slot = cache.slot_map[ref_op][eid];

            if (slot >= 0) {
                // Cache hit - use existing slot
                remapped[e] = slot;
                opt->cache_hits++;
                if (opt->groups_enabled) opt->group_cache_hits++;
            } else if (opt->groups_enabled && il < (int)opt->expert_grouping.size()
                       && !opt->expert_grouping[il].groups.empty()) {
                // Group-aware cache miss: load the ENTIRE group containing this expert
                // This is more DMA up front, but contiguous and amortizes future hits
                auto & lg = opt->expert_grouping[il];
                int gid = (eid < (int)lg.expert_to_group.size()) ? lg.expert_to_group[eid] : -1;

                if (gid >= 0 && gid < (int)lg.groups.size()) {
                    auto & grp = lg.groups[gid];
                    opt->group_cache_misses++;

                    // Load each expert in the group that isn't already cached
                    for (int ge : grp.expert_ids) {
                        if (cache.slot_map[ref_op][ge] >= 0) continue;  // already cached

                        // Find LRU slot for eviction
                        int lru_slot = 0;
                        int64_t lru_time = cache.slots[ref_op][0].last_used;
                        for (int k = 1; k < opt->K; k++) {
                            if (cache.slots[ref_op][k].expert_id >= 0 &&
                                is_protected(cache.slots[ref_op][k].expert_id)) continue;
                            if (cache.slots[ref_op][k].last_used < lru_time) {
                                lru_time = cache.slots[ref_op][k].last_used;
                                lru_slot = k;
                            }
                        }

                        // DMA all ops for this expert
                        for (int op = 0; op < OP_COUNT; op++) {
                            if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                            bool has_fused_g = (info.orig_exps[OP_GATE_UP] != nullptr);
                            if (has_fused_g && (op == OP_GATE || op == OP_UP)) continue;

                            int old_eid = cache.slots[op][lru_slot].expert_id;
                            if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;

                            size_t sb = info.expert_slice_bytes[op];
                            ggml_backend_tensor_set(cache.gpu_tensor[op],
                                (const char *)info.orig_exps[op]->data + ge * sb,
                                lru_slot * sb, sb);

                            cache.slot_map[op][ge] = lru_slot;
                            cache.slots[op][lru_slot] = { ge, ++opt->cache_clock };
                        }
                    }
                    grp.last_used = opt->cache_clock;

                    // Now the target expert should be cached
                    slot = cache.slot_map[ref_op][eid];
                    remapped[e] = (slot >= 0) ? slot : 0;
                } else {
                    // Expert not in any group - fall through to individual loading
                    remapped[e] = 0;
                    opt->cache_misses++;
                }
            } else {
                // Cache miss - decide whether to load based on thermal policy
                bool should_load = true;

                if (opt->config.mobile_mode) {
                    switch (opt->thermal_policy) {
                        case ALPHA_MOE_THERMAL_NORMAL:
                            // Load everything, including cold experts from storage
                            should_load = true;
                            break;
                        case ALPHA_MOE_THERMAL_THROTTLE:
                            // Only load hot experts; skip storage I/O for cold
                            should_load = is_protected(eid);
                            break;
                        case ALPHA_MOE_THERMAL_EMERGENCY:
                            // Emergency: don't load anything new, use what's cached
                            should_load = false;
                            break;
                    }
                }

                if (should_load) {
                    // Mobile: limit to max 1 storage read per layer per token
                    // This prevents NAND heat from stacking multiple reads
                    if (opt->config.mobile_mode && layer_loads >= 1) {
                        // Already did one load this token for this layer - substitute instead
                        int best_hot_slot = -1;
                        for (int h : opt->hot_experts) {
                            int hs = cache.slot_map[ref_op][h];
                            if (hs >= 0) { best_hot_slot = hs; break; }
                        }
                        remapped[e] = (best_hot_slot >= 0) ? best_hot_slot : 0;
                        opt->thermal_substituted++;
                    } else {
                    // Load single expert: evict LRU, DMA from CPU/storage
                    int lru_slot = 0;
                    int64_t lru_time = cache.slots[ref_op][0].last_used;
                    for (int k = 1; k < opt->K; k++) {
                        // Never evict hot expert slots
                        // Always protect hot/temporal experts from eviction (not just mobile)
                    if (cache.slots[ref_op][k].expert_id >= 0 &&
                            is_protected(cache.slots[ref_op][k].expert_id)) continue;
                        if (cache.slots[ref_op][k].last_used < lru_time) {
                            lru_time = cache.slots[ref_op][k].last_used;
                            lru_slot = k;
                        }
                    }

                    // Evict + DMA for all ops (direct copy, same type)
                    for (int op = 0; op < OP_COUNT; op++) {
                        if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                        // Tiered: skip gate/up (managed globally by callback)
                        if (opt->tiered_cache && (op == OP_GATE || op == OP_UP)) continue;
                        if (cache.slots[op].empty()) continue;

                        int old_eid = cache.slots[op][lru_slot].expert_id;
                        if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;

                        size_t sb = info.expert_slice_bytes[op];
                        ggml_backend_tensor_set(cache.gpu_tensor[op],
                            (const char *)info.orig_exps[op]->data + eid * sb,
                            lru_slot * sb, sb);

                        cache.slot_map[op][eid] = lru_slot;
                        cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
                    }
                    remapped[e] = lru_slot;
                    opt->cache_misses++;
                    layer_loads++;
                    }
                } else {
                    // Thermal skip: substitute with the best available hot expert
                    // Find the hot expert with the lowest slot index (most recently used)
                    int best_hot_slot = -1;
                    for (int h : opt->hot_experts) {
                        int hs = cache.slot_map[ref_op][h];
                        if (hs >= 0) { best_hot_slot = hs; break; }
                    }
                    remapped[e] = (best_hot_slot >= 0) ? best_hot_slot : 0;
                    opt->thermal_substituted++;
                }
            }

            // Update LRU timestamp (but not for hot experts in emergency mode - keep them pinned)
            for (int op = 0; op < OP_COUNT; op++) {
                int s = cache.slot_map[op][eid];
                if (s >= 0) cache.slots[op][s].last_used = ++opt->cache_clock;
            }
        }

        // Validate and write remapped IDs to GPU tensor
        for (int e = 0; e < opt->n_expert_used && e < 16; e++) {
            if (remapped[e] < 0 || remapped[e] >= opt->K) {
                ALPHA_LOG_DBG("[BUG5-PREPARE] INVALID slot=%d (K=%d, layer=%d, expert=%d)\n",
                    remapped[e], opt->K, il, predicted[e]);
                remapped[e] = 0;
            }
        }
        // Dedup: assign unique slot IDs. Duplicates get unused slots.
        // Skip dedup when all entries are slot 0 (zero-expert for split_group skipped layers).
        {
            bool all_zero = true;
            for (int e = 0; e < opt->n_expert_used && e < 16; e++) {
                if (remapped[e] != 0) { all_zero = false; break; }
            }
          if (!all_zero) {
            bool used[256] = {};  // which slots are taken
            for (int e = 0; e < opt->n_expert_used && e < 16; e++) {
                if (remapped[e] >= 0 && remapped[e] < opt->K && !used[remapped[e]]) {
                    used[remapped[e]] = true;
                } else {
                    // Find first unused slot (prefer 0 = zero expert for safety)
                    int alt = 0;
                    for (int s = 0; s < opt->K; s++) {
                        if (!used[s]) { alt = s; break; }
                    }
                    remapped[e] = alt;
                    used[alt] = true;
                }
            }
          } // !all_zero
        }
        if (opt->use_scratch) {
            // Scratch mode: gate/up use sequential IDs (0,1,2,...), down uses cache slots
            int32_t scratch_ids[16];
            for (int e = 0; e < opt->n_expert_used; e++) scratch_ids[e] = e;
            ggml_backend_tensor_set(ids_tensor, scratch_ids, 0,
                opt->n_expert_used * sizeof(int32_t));
            if (il < (int)opt->cache_expert_ids_down.size() && opt->cache_expert_ids_down[il]) {
                ggml_backend_tensor_set(opt->cache_expert_ids_down[il], remapped, 0,
                    opt->n_expert_used * sizeof(int32_t));
            }
        } else {
            ggml_backend_tensor_set(ids_tensor, remapped, 0,
                opt->n_expert_used * sizeof(int32_t));

            // Tiered cache: also write down IDs (same remapped for now - callback corrects later)
            if (opt->tiered_cache && il < (int)opt->cache_expert_ids_down.size() &&
                opt->cache_expert_ids_down[il]) {
                ggml_backend_tensor_set(opt->cache_expert_ids_down[il], remapped, 0,
                    opt->n_expert_used * sizeof(int32_t));
            }
        }

        // Pre-load cross-layer predicted experts and hot experts into cache
        // These are not used for this token's MUL_MAT_ID (remapped[] already written),
        // but they'll be ready for the reactive callback or next token's prediction.
        auto preload_expert = [&](int eid) {
            if (eid < 0 || eid >= opt->n_expert) return;
            int ref_op2 = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                         (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                         (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));
            if (cache.slot_map[ref_op2][eid] >= 0) return;  // already cached

            // Find LRU slot (with protection)
            int pl_slot = 0;
            int64_t pl_time = INT64_MAX;
            for (int k = 0; k < opt->K; k++) {
                if (cache.slots[ref_op2][k].expert_id >= 0 &&
                    is_protected(cache.slots[ref_op2][k].expert_id)) continue;
                if (cache.slots[ref_op2][k].last_used < pl_time) {
                    pl_time = cache.slots[ref_op2][k].last_used;
                    pl_slot = k;
                }
            }

            // DMA all ops for this expert
            for (int op = 0; op < OP_COUNT; op++) {
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                int old_eid = cache.slots[op][pl_slot].expert_id;
                if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                size_t sb = info.expert_slice_bytes[op];
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + eid * sb,
                    pl_slot * sb, sb);
                cache.slot_map[op][eid] = pl_slot;
                cache.slots[op][pl_slot] = { eid, ++opt->cache_clock };
            }
        };

        // Cross-layer prediction: load predicted experts for this layer
        if (il < (int)opt->cross_layer_pred.size()) {
            for (int e : opt->cross_layer_pred[il]) {
                preload_expert(e);
            }
        }
        // Hot per-layer experts: ensure they're cached
        if (il < (int)opt->hot_per_layer.size()) {
            for (int e : opt->hot_per_layer[il]) {
                preload_expert(e);
            }
        }

    }
}

// Phase 3, record: read the actual routing and update predictions for the next token
void alpha_moe_optimizer_record(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe) return;
    opt->total_tokens++;

    // Read actual expert IDs from argsort outputs (set during graph building)
    // Process all MoE layers (including on_gpu) so per-layer logging has data
    int n_argsort = alpha_mole_get_argsort_count();

    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx);
        moe_idx++;

        if (!argsort_out || !argsort_out->buffer) continue;

        // Read expert IDs from GPU (small: n_expert_used ints for last token)
        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || n_used <= 0) continue;

        // Read only the last token's expert IDs (most recent routing decision)
        int32_t actual_ids[16] = {};
        size_t offset = (n_tokens - 1) * argsort_out->nb[1];
        ggml_backend_tensor_get(argsort_out, actual_ids, offset, n_used * sizeof(int32_t));

        // Update prediction for next token: use this token's actual routing
        // Validate: expert IDs must be in [0, n_expert). Garbage = recycled GPU memory.
        bool valid_ids = true;
        for (int e = 0; e < n_used; e++) {
            if (actual_ids[e] < 0 || actual_ids[e] >= opt->n_expert) {
                valid_ids = false;
                break;
            }
        }
        if (valid_ids) {
            for (int e = 0; e < n_used && e < (int)opt->prev_routing[il].size(); e++) {
                opt->prev_routing[il][e] = actual_ids[e];
                opt->expert_activation_count[il][actual_ids[e]]++;
            }
        } else if (opt->config.enable_profiling) {
            // In profiling mode, retry individual IDs (some may be valid even if others aren't)
            for (int e = 0; e < n_used; e++) {
                if (actual_ids[e] >= 0 && actual_ids[e] < opt->n_expert) {
                    opt->expert_activation_count[il][actual_ids[e]]++;
                }
            }
        }

        // Update sticky routing momentum from actual routing
        if (opt->config.expert_sticky > 0.0f && il < (int)opt->expert_momentum.size()) {
            // Decay all experts
            for (int e = 0; e < opt->n_expert; e++) {
                opt->expert_momentum[il][e] *= 0.7f;
            }
            // Boost actually-used experts
            for (int e = 0; e < n_used; e++) {
                int eid = actual_ids[e];
                if (eid >= 0 && eid < opt->n_expert) {
                    opt->expert_momentum[il][eid] += 0.3f;
                }
            }
        }

        // Cross-layer prediction: this layer's routing predicts next layer's
        if (il + 1 < opt->n_layer && il + 1 < (int)opt->cross_layer_pred.size()) {
            for (int e = 0; e < n_used && e < (int)opt->cross_layer_pred[il + 1].size(); e++) {
                opt->cross_layer_pred[il + 1][e] = actual_ids[e];
            }
        }

        // Temporal locality: store in circular buffer
        int tidx = opt->recent_idx % alpha_moe_optimizer::TEMPORAL_DEPTH;
        for (int e = 0; e < n_used && e < (int)opt->recent_routing[il][tidx].size(); e++) {
            opt->recent_routing[il][tidx][e] = actual_ids[e];
        }

        // Compute Shannon entropy from gating softmax distribution
        ggml_tensor * gating = alpha_mole_get_gating_probs(moe_idx - 1);
        if (gating && gating->buffer && il < (int)opt->prev_entropy.size()) {
            int n_exp = (int)gating->ne[0];
            std::vector<float> probs(n_exp);
            ggml_backend_tensor_get(gating, probs.data(), 0, n_exp * sizeof(float));

            float entropy = 0.0f;
            for (int e = 0; e < n_exp; e++) {
                if (probs[e] > 1e-8f) {
                    entropy -= probs[e] * log2f(probs[e]);
                }
            }
            float max_entropy = log2f((float)n_exp);
            opt->prev_entropy[il] = (max_entropy > 0) ? entropy / max_entropy : 0.5f;
        }
    }

    // Advance temporal circular buffer
    opt->recent_idx++;

    // Periodically update hot expert list from activation data
    if (opt->total_tokens > 0 && (opt->total_tokens % opt->hot_update_interval) == 0) {
        int n_hot = (int)opt->hot_experts.size();
        if (n_hot > 0 && n_hot < opt->n_expert) {
            // Aggregate activation counts across all MoE layers
            std::vector<int64_t> total_acts(opt->n_expert, 0);
            for (int il = 0; il < opt->n_layer; il++) {
                for (int e = 0; e < opt->n_expert; e++) {
                    total_acts[e] += opt->expert_activation_count[il][e];
                }
            }

            // Find top-N experts by activation count
            std::vector<int> sorted_experts(opt->n_expert);
            std::iota(sorted_experts.begin(), sorted_experts.end(), 0);
            std::sort(sorted_experts.begin(), sorted_experts.end(),
                [&](int a, int b) { return total_acts[a] > total_acts[b]; });

            opt->hot_experts.clear();
            for (int i = 0; i < n_hot; i++) {
                opt->hot_experts.push_back(sorted_experts[i]);
            }
        }

        // Per-layer hot experts: pin top-8 per layer
        int n_pin = std::min(8, opt->K / 2);  // pin at most half the cache
        for (int il = 0; il < opt->n_layer; il++) {
            if (!opt->layer_info[il].has_experts) continue;
            auto & counts = opt->expert_activation_count[il];
            std::vector<int> sorted(opt->n_expert);
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(),
                [&](int a, int b) { return counts[a] > counts[b]; });
            opt->hot_per_layer[il].assign(sorted.begin(), sorted.begin() + n_pin);
        }
    }

    // Inter-layer prefetch: pre-warm cache for next-layer experts
    // Heuristic: same experts are used in adjacent layers ~75% of the time
    if (opt->prefetch_enabled && !opt->caches.empty()) {
        for (int il = 0; il < opt->n_layer; il++) {
            if (!opt->layer_info[il].has_experts || opt->layer_info[il].on_gpu) continue;
            auto & routing = opt->prev_routing[il];

            // Prefetch these experts into next `prefetch_depth` layers
            for (int d = 1; d <= opt->prefetch_depth; d++) {
                int target_layer = il + d;
                if (target_layer >= opt->n_layer) break;
                if (!opt->layer_info[target_layer].has_experts) continue;
                if (opt->layer_info[target_layer].on_gpu) continue;
                if (target_layer >= (int)opt->caches.size()) continue;

                auto & target_cache = opt->caches[target_layer];
                for (int e = 0; e < (int)routing.size(); e++) {
                    int eid = routing[e];
                    if (eid < 0 || eid >= opt->n_expert) continue;

                    // Check if expert is already in target cache via slot_map
                    bool already_cached = (target_cache.slot_map[0][eid] >= 0);

                    if (!already_cached) {
                        // Seed the target layer's routing prediction so prepare()
                        // will pre-load this expert on the next decode call
                        if (target_layer < (int)opt->prev_routing.size()) {
                            auto & target_routing = opt->prev_routing[target_layer];
                            bool found = false;
                            for (int tr = 0; tr < (int)target_routing.size(); tr++) {
                                if (target_routing[tr] == eid) { found = true; break; }
                            }
                            if (!found && (int)target_routing.size() > 0) {
                                // Add as extra prediction - prepare() handles overflow
                                target_routing.push_back(eid);
                            }
                        }
                    }
                    opt->prefetch_total++;
                    if (already_cached) opt->prefetch_hits++;
                }
            }
        }
    }
}

// Read all tokens' expert IDs from argsort (for prompt profiling).
// Unlike record() which reads only the last token, this reads all positions.
void alpha_moe_optimizer_set_profiling(bool enable) {
    g_alpha_freq_profiling.store(enable, std::memory_order_relaxed);
}

// Read argsort from last decode and DMA cache misses into GPU cache.
// Call after a single-token cpu-moe decode to warm cache with prompt-specific experts.
void alpha_moe_optimizer_warmup_cache_from_decode(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || opt->caches.empty()) return;

    int n_argsort = alpha_mole_get_argsort_count();
    if (n_argsort <= 0) return;

    int dma_count = 0;
    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx++);
        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || n_used <= 0) continue;

        // Read last token's expert IDs
        int32_t ids[16] = {};
        size_t offset = (size_t)(n_tokens - 1) * argsort_out->nb[1];
        if (offset + n_used * sizeof(int32_t) > ggml_nbytes(argsort_out)) continue;
        ggml_backend_tensor_get(argsort_out, ids, offset, n_used * sizeof(int32_t));

        // DMA each miss into cache
        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        for (int e = 0; e < n_used; e++) {
            int eid = ids[e];
            if (eid < 0 || eid >= opt->n_expert) continue;

            // Check if already cached (any op)
            int first_op = has_fused ? OP_GATE_UP : OP_GATE;
            if (cache.slot_map[first_op][eid] >= 0) {
                // Already cached - update LRU timestamp (protects frequently-used experts)
                int slot = cache.slot_map[first_op][eid];
                for (int op = 0; op < OP_COUNT; op++) {
                    if (cache.slot_map[op][eid] >= 0)
                        cache.slots[op][cache.slot_map[op][eid]].last_used = ++opt->cache_clock;
                }
                (void)slot;
                continue;
            }

            // Find LRU slot to evict
            int lru_slot = -1;
            int64_t oldest = INT64_MAX;
            for (int k = 1; k < opt->K; k++) {  // skip slot 0 (zero expert)
                if (cache.slots[first_op][k].last_used < oldest) {
                    oldest = cache.slots[first_op][k].last_used;
                    lru_slot = k;
                }
            }
            if (lru_slot < 0) continue;

            // DMA all ops for this expert into the evicted slot
            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];

                // Evict old expert from slot_map
                int old_eid = cache.slots[op][lru_slot].expert_id;
                if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;

                // DMA new expert
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)lru_slot * sb, sb);

                cache.slot_map[op][eid] = lru_slot;
                cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
            }
            dma_count++;
        }
    }

    // Update LUT to reflect new cache state
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }

    ALPHA_LOG_DBG("[WARMUP] DMA'd %d expert misses into GPU cache from 1-token decode\n", dma_count);
}

// Smart 1-token warmup: read argsort from 1 decode, DMA only misses,
// fill remaining empty slots with cross-layer predicted experts.
void alpha_moe_optimizer_warmup_1token_smart(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || opt->caches.empty()) return;

    int n_argsort = alpha_mole_get_argsort_count();
    if (n_argsort <= 0) return;

    // Step 1: Read argsort for all MoE layers, build cross-layer frequency
    std::vector<int> cross_freq(opt->n_expert, 0);  // how many layers each expert appeared in
    struct LayerData { int ids[16]; int n_used; };
    std::vector<LayerData> layer_data;
    layer_data.reserve(n_argsort);

    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx++);
        LayerData ld = {};
        if (!argsort_out || !argsort_out->buffer) { layer_data.push_back(ld); continue; }

        ld.n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || ld.n_used <= 0) { layer_data.push_back(ld); continue; }

        size_t offset = (size_t)(n_tokens - 1) * argsort_out->nb[1];
        if (offset + ld.n_used * sizeof(int32_t) > ggml_nbytes(argsort_out)) {
            layer_data.push_back(ld); continue;
        }
        ggml_backend_tensor_get(argsort_out, ld.ids, offset, ld.n_used * sizeof(int32_t));

        // Update cross-layer frequency
        for (int e = 0; e < ld.n_used; e++) {
            if (ld.ids[e] >= 0 && ld.ids[e] < opt->n_expert)
                cross_freq[ld.ids[e]]++;
        }
        layer_data.push_back(ld);
    }

    // Step 2: Sort cross_freq to get globally hot experts for this prompt
    std::vector<std::pair<int, int>> cross_sorted(opt->n_expert);
    for (int e = 0; e < opt->n_expert; e++) {
        cross_sorted[e] = {cross_freq[e], e};
    }
    std::sort(cross_sorted.begin(), cross_sorted.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });

    // Step 3: Per layer - DMA misses from fired experts, fill empty with cross-layer
    int dma_misses = 0, dma_predictions = 0;
    moe_idx = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (moe_idx >= (int)layer_data.size()) break;

        auto & ld = layer_data[moe_idx++];
        if (ld.n_used <= 0) continue;
        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        int ref_op = has_fused ? OP_GATE_UP : (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);
        if (!cache.gpu_tensor[ref_op]) continue;
        if (cache.slots[ref_op].empty() || (int)cache.slots[ref_op].size() < opt->K) continue;

        // 3a: DMA only misses from the 10 fired experts
        for (int e = 0; e < ld.n_used; e++) {
            int eid = ld.ids[e];
            if (eid < 0 || eid >= opt->n_expert) continue;
            if (cache.slot_map[ref_op][eid] >= 0) {
                // Hit - just update LRU timestamp
                for (int op = 0; op < OP_COUNT; op++) {
                    if (cache.slot_map[op][eid] >= 0)
                        cache.slots[op][cache.slot_map[op][eid]].last_used = ++opt->cache_clock;
                }
                continue;
            }
            // Miss - evict LRU, DMA this expert
            int lru_slot = -1;
            int64_t oldest = INT64_MAX;
            for (int k = 1; k < opt->K; k++) {
                if (cache.slots[ref_op][k].last_used < oldest) {
                    oldest = cache.slots[ref_op][k].last_used;
                    lru_slot = k;
                }
            }
            if (lru_slot < 0) continue;
            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                int old_eid = cache.slots[op][lru_slot].expert_id;
                if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)lru_slot * sb, sb);
                cache.slot_map[op][eid] = lru_slot;
                cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
            }
            dma_misses++;
        }

        // 3b: Fill remaining empty slots with cross-layer predicted experts
        for (int ci = 0; ci < (int)cross_sorted.size(); ci++) {
            int eid = cross_sorted[ci].second;
            int freq = cross_sorted[ci].first;
            if (freq <= 0) break;
            if (eid < 0 || eid >= opt->n_expert) continue;
            if (cache.slot_map[ref_op][eid] >= 0) continue;

            // Find an empty slot
            int empty_slot = -1;
            for (int k = 1; k < opt->K; k++) {
                if (cache.slots[ref_op][k].expert_id < 0) {
                    empty_slot = k;
                    break;
                }
            }
            if (empty_slot < 0) break;  // no empty slots left

            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)empty_slot * sb, sb);
                cache.slot_map[op][eid] = empty_slot;
                cache.slots[op][empty_slot] = { eid, ++opt->cache_clock };
            }
            dma_predictions++;
        }
    }

    // Update LUT
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }

    ALPHA_LOG_DBG("[WARMUP] 1-token smart: %d misses DMA'd + %d cross-layer predictions\n",
            dma_misses, dma_predictions);
}

// Prompt-Type Profile Classification
// Pre-computed profiles: top-33 experts per layer for each prompt style.
// At runtime, 1 token's routing is matched against profiles to find best fit.

#include "alpha_mole_embedded_profiles.h"

#define MAX_PROFILES 10
#define MAX_PROFILE_LAYERS 64
#define EXPERTS_PER_PROFILE 33
#define MAX_DOMAIN_PROFILES 128  // 57 MMLU subjects + 24 domains + core/grammar/classify
#define EXPERTS_PER_DOMAIN 20

struct prompt_profile {
    char name[32];
    int n_layers;
    int32_t experts[MAX_PROFILE_LAYERS][EXPERTS_PER_PROFILE];
};

static prompt_profile g_profiles[MAX_PROFILES];
static int g_n_profiles = 0;
static float g_profile_match_quality = 0.0f;  // 0.0-1.0, set by classify
static const char * g_matched_profile_name = "none";  // set by classify

// Domain-Expert Pinning (Tier 3 output domains)
// Separate from style profiles - domain profiles pin rare experts for
// specialized output tokens (chemical notation, math formulas, etc.)
struct domain_profile {
    char name[32];
    int n_layers;
    int32_t experts[MAX_PROFILE_LAYERS][EXPERTS_PER_DOMAIN];
};

static domain_profile g_domain_profiles[MAX_DOMAIN_PROFILES];
static int g_n_domain_profiles = 0;
static const char * g_detected_domain = nullptr;   // primary domain (keyword/LUT classify)
static const char * g_detected_domain2 = nullptr;  // secondary domain (multi-domain prompts)

// Keyword lists for output domain detection
// Keyword sets for 100+ domains

// --- Programming languages ---
static const char * python_keywords[] = {
    "python", "def ", "import ", "pip", "numpy", "pandas",
    "django", "flask", "pytest", "lambda", "__init__", nullptr
};
static const char * javascript_keywords[] = {
    "javascript", "js ", "node", "react", "vue", "angular",
    "async", "await", "promise", "npm", "webpack", "DOM", nullptr
};
static const char * typescript_keywords[] = {
    "typescript", "ts ", "interface ", "type ", "generic",
    "union type", "enum ", nullptr
};
static const char * java_keywords[] = {
    "java ", "JVM", "spring", "maven", "gradle",
    "public static void", "ArrayList", "HashMap", nullptr
};
static const char * cpp_keywords[] = {
    "c++", "cpp", "RAII", "template", "smart pointer",
    "std::", "namespace", "destructor", "virtual", nullptr
};
static const char * rust_keywords[] = {
    "rust ", "cargo", "ownership", "borrow checker",
    "lifetime", "trait ", "impl ", "Result<", "Option<", nullptr
};
static const char * golang_keywords[] = {
    "golang", "go ", "goroutine", "channel", "defer",
    "go func", "interface{}", nullptr
};
static const char * shell_bash_keywords[] = {
    "bash", "shell", "#!/bin", "grep", "awk", "sed",
    "chmod", "pipe", "$HOME", ".bashrc", nullptr
};

// --- CS topics ---
static const char * algorithms_keywords[] = {
    "dijkstra", "shortest path", "topological sort",
    "red-black tree", "dynamic programming", "BFS", "DFS",
    "merge sort", "heap", "graph algorithm", nullptr
};
static const char * data_structures_keywords[] = {
    "trie", "segment tree", "union-find", "disjoint set",
    "linked list", "stack", "queue", "binary tree",
    "balanced tree", "priority queue", nullptr
};
static const char * machine_learning_keywords[] = {
    "machine learning", "gradient descent", "neural network",
    "bias-variance", "overfitting", "CNN", "RNN", "transformer",
    "backpropagation", "training data", "deep learning", nullptr
};
static const char * computer_security_keywords[] = {
    "buffer overflow", "XSS", "SQL injection", "encryption",
    "firewall", "vulnerability", "HTTPS", "TLS", "SSL",
    "authentication", "authorization", "OWASP", nullptr
};
static const char * database_keywords[] = {
    "migration", "schema", "PostgreSQL", "MySQL", "MongoDB",
    "index", "foreign key", "normalization", "ORM", nullptr
};
static const char * devops_keywords[] = {
    "docker", "kubernetes", "CI/CD", "pipeline", "deploy",
    "container", "helm", "terraform", "ansible", "k8s", nullptr
};
static const char * git_workflow_keywords[] = {
    "git ", "rebase", "merge", "branch", "commit",
    "pull request", ".gitignore", "bisect", "cherry-pick", nullptr
};
static const char * web_frontend_keywords[] = {
    "React", "component", "useState", "CSS", "HTML",
    "responsive", "flexbox", "grid layout", "webpack", nullptr
};
static const char * testing_keywords[] = {
    "pytest", "jest", "unittest", "test case", "mock",
    "assertion", "coverage", "TDD", "integration test", nullptr
};
static const char * debugging_keywords[] = {
    "debug", "bug", "traceback", "stack trace", "breakpoint",
    "race condition", "segfault", "core dump", nullptr
};
static const char * api_design_keywords[] = {
    "REST", "GraphQL", "API", "endpoint", "swagger",
    "OpenAPI", "webhook", "rate limit", nullptr
};
static const char * performance_keywords[] = {
    "optimize", "profil", "benchmark", "latency",
    "throughput", "cache miss", "bottleneck", "memoiz", nullptr
};
static const char * refactoring_keywords[] = {
    "refactor", "clean code", "single responsibility",
    "design pattern", "SOLID", "code smell", nullptr
};
static const char * documentation_keywords[] = {
    "docstring", "README", "JSDoc", "API documentation",
    "changelog", "comment", nullptr
};
static const char * code_review_keywords[] = {
    "code review", "pull request", "code smell",
    "security vulnerabilit", "improvement", nullptr
};
static const char * security_keywords[] = {
    "sanitize", "JWT", "token", "CSRF", "content security policy",
    "OAuth", "CORS", "hash password", nullptr
};

// --- MMLU Science ---
static const char * college_biology_keywords[] = {
    "mitosis", "meiosis", "ribosome", "Calvin cycle",
    "photosynthesis", "ATP", "Krebs cycle", nullptr
};
static const char * college_chemistry_keywords[] = {
    "hybridization", "Le Chatelier", "molarity", "titration",
    "electronegativity", "orbital", "sp3", nullptr
};
static const char * college_physics_keywords[] = {
    "Maxwell", "Heisenberg", "harmonic motion", "Lagrangian",
    "Schrodinger", "wave function", "angular momentum", nullptr
};
static const char * high_school_physics_keywords[] = {
    "Newton", "kinetic energy", "potential energy", "momentum",
    "acceleration", "force", "velocity", "F=ma", nullptr
};
static const char * high_school_chemistry_keywords[] = {
    "ionic bond", "covalent", "Avogadro", "mole ",
    "electron configuration", "valence", nullptr
};
static const char * high_school_biology_keywords[] = {
    "DNA", "allele", "dominant", "recessive", "gene",
    "cell membrane", "mitochondria", "replication", nullptr
};
static const char * conceptual_physics_keywords[] = {
    "why does ice float", "speed and velocity", "lever",
    "buoyancy", "friction", "inertia", "gravity", nullptr
};
static const char * astronomy_keywords[] = {
    "star", "planet", "galaxy", "light-year", "solar",
    "nebula", "black hole", "supernova", "orbit", nullptr
};
static const char * anatomy_keywords[] = {
    "organ", "muscle", "bone", "heart", "lung", "brain",
    "tendon", "ligament", "artery", "vein", nullptr
};
static const char * virology_keywords[] = {
    "virus", "vaccine", "herd immunity", "RNA virus",
    "antibody", "pandemic", "pathogen", nullptr
};
static const char * nutrition_keywords[] = {
    "macronutrient", "calorie", "vitamin", "protein intake",
    "glycemic index", "dietary", "carbohydrate", nullptr
};

// --- MMLU Math/Stats ---
static const char * high_school_mathematics_keywords[] = {
    "quadratic", "derivative", "integral", "polynomial",
    "parabola", "logarithm", "exponent", "sine", "cosine", nullptr
};
static const char * college_mathematics_keywords[] = {
    "Cauchy", "irrational", "fundamental theorem",
    "topology", "group theory", "eigenvalue", nullptr
};
static const char * elementary_mathematics_keywords[] = {
    "times", "multiply", "divide", "fraction", "decimal",
    "percentage", "area of", "perimeter", nullptr
};
static const char * high_school_statistics_keywords[] = {
    "standard deviation", "p-value", "central limit",
    "hypothesis test", "confidence interval", "mean", "median", nullptr
};
static const char * abstract_algebra_keywords[] = {
    "group theory", "ring", "field", "isomorphism",
    "Lagrange", "abelian", "homomorphism", nullptr
};
static const char * formal_logic_keywords[] = {
    "modus ponens", "tautology", "syllogism", "valid argument",
    "logical fallacy", "deduction", nullptr
};
static const char * econometrics_keywords[] = {
    "heteroscedasticity", "Gauss-Markov", "instrumental variable",
    "regression", "endogeneity", "OLS", nullptr
};

// --- MMLU Social Science / History ---
static const char * high_school_geography_keywords[] = {
    "continent", "tectonic", "river", "ocean", "mountain",
    "climate zone", "latitude", "longitude", nullptr
};
static const char * high_school_macroeconomics_keywords[] = {
    "GDP", "inflation", "fiscal policy", "monetary policy",
    "unemployment", "interest rate", "central bank", nullptr
};
static const char * high_school_microeconomics_keywords[] = {
    "supply and demand", "elasticity", "monopoly", "oligopoly",
    "marginal cost", "consumer surplus", nullptr
};
static const char * high_school_psychology_keywords[] = {
    "classical conditioning", "Maslow", "cognitive dissonance",
    "Pavlov", "Skinner", "Freud", "behaviorism", nullptr
};
static const char * high_school_us_history_keywords[] = {
    "Civil Rights", "Monroe Doctrine", "Great Depression",
    "Constitution", "Amendment", "Civil War", nullptr
};
static const char * high_school_world_history_keywords[] = {
    "Silk Road", "Roman Empire", "Industrial Revolution",
    "Renaissance", "colonialism", "Ottoman", nullptr
};
static const char * high_school_european_history_keywords[] = {
    "World War I", "French Revolution", "Renaissance",
    "Napoleon", "Cold War", "Berlin Wall", nullptr
};
static const char * high_school_government_keywords[] = {
    "branches of government", "filibuster", "federalism",
    "legislature", "executive", "judicial", "Congress", nullptr
};
static const char * philosophy_keywords[] = {
    "Plato", "Kant", "existentialism", "ethics", "metaphysics",
    "epistemology", "Socrates", "Aristotle", "Nietzsche", nullptr
};
static const char * sociology_keywords[] = {
    "stratification", "Durkheim", "anomie", "social construct",
    "inequality", "institution", "Weber", nullptr
};
static const char * finance_keywords[] = {
    "P/E ratio", "stock", "bond", "compound interest",
    "portfolio", "dividend", "asset", "equity", nullptr
};
static const char * management_keywords[] = {
    "SWOT", "leadership", "team development", "KPI",
    "stakeholder", "agile", "scrum", nullptr
};
static const char * marketing_keywords[] = {
    "4 Ps", "market segment", "brand", "target audience",
    "advertising", "SEO", "conversion", nullptr
};
static const char * professional_law_keywords[] = {
    "precedent", "habeas corpus", "civil law", "criminal law",
    "tort", "appeal", "jury", "verdict", nullptr
};

// --- Languages ---
static const char * french_keywords[] = {
    "french", "francais", "bonjour", "conjugat", "Paris",
    "aller", "avoir", "etre", nullptr
};
static const char * spanish_keywords[] = {
    "spanish", "espanol", "hola", "ser ", "estar ",
    "tener", "conjugat", nullptr
};
static const char * german_keywords[] = {
    "german", "deutsch", "der ", "die ", "das ",
    "guten", "sprechen", nullptr
};
static const char * japanese_keywords[] = {
    "japanese", "hiragana", "katakana", "kanji",
    "desu", "nihongo", nullptr
};
static const char * chinese_keywords[] = {
    "chinese", "mandarin", "pinyin", "tone",
    "hanzi", "simplified", nullptr
};
static const char * korean_keywords[] = {
    "korean", "hangul", "honorific", "annyeong",
    "polite form", nullptr
};

// --- Professional ---
static const char * clinical_knowledge_keywords[] = {
    "blood pressure", "anemia", "Type 1 diabetes",
    "clinical sign", "symptom", "prognosis", nullptr
};
static const char * professional_medicine_keywords[] = {
    "Glasgow Coma", "ABCDE", "brain death", "triage",
    "emergency", "ICU", "ventilator", nullptr
};
static const char * medical_genetics_keywords[] = {
    "point mutation", "autosomal", "CRISPR", "Cas9",
    "genetic disorder", "karyotype", "allele frequency", nullptr
};
static const char * electrical_engineering_keywords[] = {
    "Ohm's law", "MOSFET", "transistor", "capacitor",
    "resistor", "inductor", "AC ", "DC ", "circuit", nullptr
};
static const char * global_facts_keywords[] = {
    "world population", "largest country", "deepest ocean",
    "highest mountain", "longest river", nullptr
};
static const char * miscellaneous_keywords[] = {
    "tallest building", "national animal", "light bulb",
    "invented", "world record", nullptr
};

static const char * chemical_keywords[] = {
    "formula", "chemical", "compound", "molecule", "element",
    "reaction", "H2O", "NaCl", "periodic table", "molar",
    "bond", "ionic", "acid", "base", "oxide", "hydroxide", nullptr
};
static const char * math_keywords[] = {
    "equation", "solve", "integral", "derivative", "calculate",
    "quadratic", "polynomial", "matrix", "vector", "proof",
    "theorem", "algebra", "calculus", "trigonometric", nullptr
};
static const char * latex_keywords[] = {
    "LaTeX", "latex", "typeset", "\\frac", "\\begin",
    "mathematical notation", "render equation", nullptr
};
static const char * medical_keywords[] = {
    "dosage", "prescription", "diagnosis", "mg/dL",
    "pharmaceutical", "clinical", "pathology", nullptr
};
static const char * biology_keywords[] = {
    "photosynthesis", "biology", "organism", "evolution",
    "ecosystem", "cell", "DNA", "protein", "enzyme",
    "mitochondria", "chloroplast", "metabolism", "gene",
    "species", "bacteria", "virus", "chromosome", nullptr
};
static const char * scientific_keywords[] = {
    "scientific notation", "avogadro", "planck", "boltzmann",
    "constant", "SI units", "electron charge", "physics",
    "thermodynamic", "quantum", "relativity", nullptr
};
static const char * legal_keywords[] = {
    "contract", "clause", "liability", "jurisdiction",
    "habeas corpus", "plaintiff", "defendant", "statute", nullptr
};
static const char * multilingual_keywords[] = {
    "translate", "translation", "french", "spanish", "german",
    "japanese", "chinese", "korean", "arabic", "hindi", "russian", nullptr
};
static const char * regex_keywords[] = {
    "regex", "regular expression", "pattern matching",
    "regexp", "capture group", nullptr
};
static const char * binary_hex_keywords[] = {
    "binary", "hexadecimal", "0x", "0b", "two's complement",
    "bitwise", "byte", nullptr
};
static const char * phonetic_keywords[] = {
    "IPA", "phonetic", "transcription", "pronunciation",
    "phoneme", "vowel sound", nullptr
};
static const char * musical_keywords[] = {
    "chord", "scale", "music", "melody", "tempo",
    "time signature", "key of", nullptr
};
static const char * assembly_keywords[] = {
    "assembly", "x86", "ARM", "MIPS", "register",
    "mov eax", "opcode", "instruction set", nullptr
};
static const char * symbolic_logic_keywords[] = {
    "predicate logic", "truth table", "De Morgan",
    "logical formula", "propositional", "first-order logic", nullptr
};
// Prose domains
static const char * factual_keywords[] = {
    "capital of", "who invented", "what year", "how many",
    "largest", "tallest", "longest", "deepest", "oldest",
    "population", "founded", "discovered", "born", nullptr
};
static const char * analytical_keywords[] = {
    "compare", "advantage", "disadvantage", "pros and cons",
    "analyze", "evaluate", "impact", "trade-off", "versus", nullptr
};
static const char * narrative_keywords[] = {
    "write a story", "short story", "describe a", "opening paragraph",
    "vivid detail", "creative writing", "fiction", "poem", nullptr
};
static const char * conversational_keywords[] = {
    "how are you", "help me", "tell me a joke",
    "what do you think", "can you help", "opinion", nullptr
};
static const char * instructional_keywords[] = {
    "step by step", "how do I", "how to", "explain how",
    "steps to", "tutorial", "guide", "instructions", nullptr
};
static const char * technical_keywords[] = {
    "how does", "architecture", "protocol", "TCP",
    "garbage collection", "compiler", "three-way handshake",
    "microservice", "distributed", "concurrency", nullptr
};
static const char * code_keywords[] = {
    "function", "implement", "algorithm", "class",
    "write a program", "code", "def ", "return",
    "python", "javascript", "typescript", "rust",
    "variable", "loop", "array", "debug", "compile", nullptr
};
static const char * json_keywords[] = {
    "JSON", "json", "schema", "object", "nested",
    "key-value", nullptr
};
static const char * sql_keywords[] = {
    "SQL", "SELECT", "query", "JOIN", "GROUP BY",
    "database", "table", nullptr
};
static const char * markdown_keywords[] = {
    "markdown", "README", "bullet point",
    "changelog", "heading", nullptr
};

struct domain_keyword_set {
    const char * domain;
    const char ** keywords;
};

// Ordered by specificity: rare domains first (more specific keywords)
static const domain_keyword_set g_domain_keyword_sets[] = {
    // Level 4 - most specialized (rare keywords, high precision)
    { "regex",              regex_keywords },
    { "binary_hex",         binary_hex_keywords },
    { "phonetic",           phonetic_keywords },
    { "musical",            musical_keywords },
    { "assembly",           assembly_keywords },
    { "symbolic_logic",     symbolic_logic_keywords },
    { "abstract_algebra",   abstract_algebra_keywords },
    { "econometrics",       econometrics_keywords },
    { "medical_genetics",   medical_genetics_keywords },
    { "virology",           virology_keywords },
    { "electrical_engineering", electrical_engineering_keywords },

    // Level 3 - specialized academic / professional
    { "college_physics",    college_physics_keywords },
    { "college_chemistry",  college_chemistry_keywords },
    { "college_biology",    college_biology_keywords },
    { "college_mathematics", college_mathematics_keywords },
    { "clinical_knowledge", clinical_knowledge_keywords },
    { "professional_medicine", professional_medicine_keywords },
    { "professional_law",   professional_law_keywords },
    { "formal_logic",       formal_logic_keywords },
    { "astronomy",          astronomy_keywords },
    { "anatomy",            anatomy_keywords },
    { "sociology",          sociology_keywords },
    { "philosophy",         philosophy_keywords },

    // Level 2 - domain-specific
    { "high_school_physics",    high_school_physics_keywords },
    { "high_school_chemistry",  high_school_chemistry_keywords },
    { "high_school_biology",    high_school_biology_keywords },
    { "high_school_mathematics", high_school_mathematics_keywords },
    { "high_school_statistics", high_school_statistics_keywords },
    { "high_school_macroeconomics", high_school_macroeconomics_keywords },
    { "high_school_microeconomics", high_school_microeconomics_keywords },
    { "high_school_psychology", high_school_psychology_keywords },
    { "high_school_geography",  high_school_geography_keywords },
    { "high_school_us_history", high_school_us_history_keywords },
    { "high_school_world_history", high_school_world_history_keywords },
    { "high_school_european_history", high_school_european_history_keywords },
    { "high_school_government_and_politics", high_school_government_keywords },
    { "conceptual_physics",     conceptual_physics_keywords },
    { "elementary_mathematics",  elementary_mathematics_keywords },
    { "global_facts",           global_facts_keywords },
    { "nutrition",              nutrition_keywords },
    { "finance",                finance_keywords },
    { "management",             management_keywords },
    { "marketing",              marketing_keywords },

    { "chemical",       chemical_keywords },
    { "math",           math_keywords },
    { "latex",          latex_keywords },
    { "scientific",     scientific_keywords },
    { "medical",        medical_keywords },
    { "legal",          legal_keywords },
    { "biology",        biology_keywords },

    // Level 1 - programming & tools
    { "python",         python_keywords },
    { "javascript",     javascript_keywords },
    { "typescript",     typescript_keywords },
    { "java",           java_keywords },
    { "cpp",            cpp_keywords },
    { "rust",           rust_keywords },
    { "golang",         golang_keywords },
    { "shell_bash",     shell_bash_keywords },
    { "sql",            sql_keywords },
    { "algorithms",     algorithms_keywords },
    { "data_structures", data_structures_keywords },
    { "machine_learning", machine_learning_keywords },
    { "computer_security", computer_security_keywords },
    { "database",       database_keywords },
    { "devops",         devops_keywords },
    { "git_workflow",   git_workflow_keywords },
    { "web_frontend",   web_frontend_keywords },
    { "testing",        testing_keywords },
    { "debugging",      debugging_keywords },
    { "api_design",     api_design_keywords },
    { "performance",    performance_keywords },
    { "refactoring",    refactoring_keywords },
    { "security",       security_keywords },
    { "code_review",    code_review_keywords },
    { "documentation",  documentation_keywords },
    { "json_output",    json_keywords },
    { "markdown",       markdown_keywords },
    { "code",           code_keywords },

    // Languages
    { "french",         french_keywords },
    { "spanish",        spanish_keywords },
    { "german",         german_keywords },
    { "japanese",       japanese_keywords },
    { "chinese",        chinese_keywords },
    { "korean",         korean_keywords },
    { "multilingual",   multilingual_keywords },

    // Level 0 - prose / general (broad keywords, lowest priority)
    { "factual",        factual_keywords },
    { "analytical",     analytical_keywords },
    { "narrative",      narrative_keywords },
    { "conversational", conversational_keywords },
    { "instructional",  instructional_keywords },
    { "technical",      technical_keywords },
    { "miscellaneous",  miscellaneous_keywords },
    { nullptr,          nullptr }
};

// Case-insensitive substring search
static bool str_contains_ci(const char * haystack, int haystack_len, const char * needle) {
    int needle_len = (int)strlen(needle);
    if (needle_len > haystack_len) return false;
    for (int i = 0; i <= haystack_len - needle_len; i++) {
        bool match = true;
        for (int j = 0; j < needle_len; j++) {
            if (tolower((unsigned char)haystack[i+j]) != tolower((unsigned char)needle[j])) {
                match = false; break;
            }
        }
        if (match) return true;
    }
    return false;
}

const char * alpha_moe_detect_output_domain(const char * prompt, int prompt_len) {
    if (!prompt || prompt_len <= 0) return nullptr;
    int best_matches = 0;
    const char * best_domain = nullptr;
    for (int d = 0; g_domain_keyword_sets[d].domain; d++) {
        int matches = 0;
        for (int k = 0; g_domain_keyword_sets[d].keywords[k]; k++) {
            if (str_contains_ci(prompt, prompt_len, g_domain_keyword_sets[d].keywords[k])) {
                matches++;
            }
        }
        if (matches > best_matches) {
            best_matches = matches;
            best_domain = g_domain_keyword_sets[d].domain;
        }
    }
    // Require at least 1 keyword match
    return (best_matches >= 1) ? best_domain : nullptr;
}

alpha_moe_domain_result alpha_moe_detect_output_domain_top2(const char * prompt, int prompt_len) {
    alpha_moe_domain_result result = { nullptr, nullptr, 0, 0 };
    if (!prompt || prompt_len <= 0) return result;
    for (int d = 0; g_domain_keyword_sets[d].domain; d++) {
        int matches = 0;
        for (int k = 0; g_domain_keyword_sets[d].keywords[k]; k++) {
            if (str_contains_ci(prompt, prompt_len, g_domain_keyword_sets[d].keywords[k])) {
                matches++;
            }
        }
        if (matches > result.score1) {
            // Demote current best to second
            result.domain2 = result.domain1;
            result.score2 = result.score1;
            result.domain1 = g_domain_keyword_sets[d].domain;
            result.score1 = matches;
        } else if (matches > result.score2 && g_domain_keyword_sets[d].domain != result.domain1) {
            result.domain2 = g_domain_keyword_sets[d].domain;
            result.score2 = matches;
        }
    }
    // Require at least 1 match for each
    if (result.score1 < 1) { result.domain1 = nullptr; }
    if (result.score2 < 1) { result.domain2 = nullptr; }
    return result;
}

void alpha_moe_set_detected_domain2(const char * domain) { g_detected_domain2 = domain; }

int alpha_moe_detect_output_domain_ranked(const char * prompt, int prompt_len,
                                           const char ** out_domains, int * out_scores, int max_results) {
    if (!prompt || prompt_len <= 0 || !out_domains || max_results <= 0) return 0;
    // Score all domains
    struct scored { const char * domain; int score; };
    std::vector<scored> all;
    for (int d = 0; g_domain_keyword_sets[d].domain; d++) {
        int matches = 0;
        for (int k = 0; g_domain_keyword_sets[d].keywords[k]; k++) {
            if (str_contains_ci(prompt, prompt_len, g_domain_keyword_sets[d].keywords[k]))
                matches++;
        }
        if (matches >= 1) {
            all.push_back({ g_domain_keyword_sets[d].domain, matches });
        }
    }
    // Sort descending by score
    std::sort(all.begin(), all.end(), [](const scored & a, const scored & b) { return a.score > b.score; });
    int n = std::min((int)all.size(), max_results);
    for (int i = 0; i < n; i++) {
        out_domains[i] = all[i].domain;
        if (out_scores) out_scores[i] = all[i].score;
    }
    return n;
}

// Load profiles from a directory of JSON freq files (freq_profile.json format).
// Each file becomes one profile. Called once at init.
void alpha_moe_optimizer_load_style_profiles(alpha_moe_optimizer * opt, const char * dir_path) {
    if (!opt || !dir_path) return;
    // Try to load profiles from directory: creative_freq.json, factual_freq.json, etc.
    const char * styles[] = {"factual", "creative", "reasoning", "code", "translate",
                             "summarize", "chat", "analysis", "instruct", "knowledge"};
    g_n_profiles = 0;

    for (int s = 0; s < 10 && g_n_profiles < MAX_PROFILES; s++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/%s_fp.json", dir_path, styles[s]);
        FILE * f = fopen(path, "r");
        if (!f) continue;

        auto & prof = g_profiles[g_n_profiles];
        snprintf(prof.name, sizeof(prof.name), "%s", styles[s]);
        prof.n_layers = 0;

        // Parse freq_profile.json format:
        // {"layers": [{"layer": 0, "top_experts": [463, 151, ...], "frequencies": [...]}, ...]}
        // Only top_experts (first 33) per layer is needed.
        char line[8192];
        int current_layer = -1;

        while (fgets(line, sizeof(line), f)) {
            // Look for "layer": N
            char * lk = strstr(line, "\"l\":");
            if (lk) {
                current_layer = atoi(lk + 4);
            }
            // Look for "top_experts": [id, id, id, ...]
            char * te = strstr(line, "\"te\":");
            if (te && current_layer >= 0 && current_layer < MAX_PROFILE_LAYERS) {
                char * bracket = strchr(te, '[');
                if (bracket) {
                    int n_parsed = 0;
                    char * p = bracket + 1;
                    while (*p && *p != ']' && n_parsed < EXPERTS_PER_PROFILE) {
                        while (*p && (*p == ' ' || *p == ',')) p++;
                        if (*p >= '0' && *p <= '9') {
                            prof.experts[current_layer][n_parsed++] = atoi(p);
                            while (*p >= '0' && *p <= '9') p++;
                        } else {
                            p++;
                        }
                    }
                    if (n_parsed > 0) {
                        prof.n_layers = std::max(prof.n_layers, current_layer + 1);
                    }
                }
            }
        }
        fclose(f);

        if (prof.n_layers > 0) {
            ALPHA_LOG_DBG("[PROFILE] Loaded '%s' (%d layers)\n", prof.name, prof.n_layers);
            g_n_profiles++;
        }
    }
    if (g_n_profiles > 0) {
        ALPHA_LOG_DBG("[PROFILE] %d prompt-type profiles loaded from %s\n", g_n_profiles, dir_path);
    }

    // Try .apack encrypted pack if no JSON profiles found
    if (g_n_profiles == 0 && opt) {
        char arch_buf[64] = {};
        if (opt->model) {
            llama_model_meta_val_str(opt->model, "general.architecture", arch_buf, sizeof(arch_buf));
        }
        uint8_t fp[APACK_FP_SIZE];
        apack_compute_fingerprint(arch_buf, opt->n_expert, opt->n_expert_used,
                                  opt->n_layer, opt->n_embd, fp);
        // a cached pack, or (ALPHA_PACK_SYNC=1 only) one fetched from the pack server
        apack_sync(fp);
        char pack_path[512];
        if (apack_find_cached(fp, pack_path, sizeof(pack_path))) {
            apack_contents ac = apack_load(pack_path, fp);
            if (ac.valid) {
                for (int p = 0; p < ac.n_profiles && g_n_profiles < MAX_PROFILES; p++) {
                    auto & prof = g_profiles[g_n_profiles];
                    snprintf(prof.name, sizeof(prof.name), "%s", ac.profiles[p].name);
                    prof.n_layers = ac.profiles[p].n_layers;
                    for (int l = 0; l < prof.n_layers && l < MAX_PROFILE_LAYERS; l++) {
                        for (int e = 0; e < EXPERTS_PER_PROFILE; e++) {
                            prof.experts[l][e] = ac.profiles[p].experts[l][e];
                        }
                    }
                    g_n_profiles++;
                }
                ALPHA_LOG_DBG("[PACK] Loaded %d profiles from %s\n", ac.n_profiles, pack_path);
            }
        }
    }

    // Fallback: use embedded profiles when no files or packs found
    if (g_n_profiles == 0 && opt) {
        for (int m = 0; m < EMBEDDED_N_MODELS; m++) {
            const auto & em = g_embedded_profiles[m];
            if (em.n_expert == opt->n_expert && em.n_moe_layers == opt->n_moe_layers) {
                for (int p = 0; p < em.n_profiles && g_n_profiles < MAX_PROFILES; p++) {
                    auto & prof = g_profiles[g_n_profiles];
                    snprintf(prof.name, sizeof(prof.name), "%s", em.profiles[p].name);
                    prof.n_layers = em.profiles[p].n_layers;
                    int n_exp = std::min(em.profiles[p].n_experts_per_layer, EXPERTS_PER_PROFILE);
                    for (int l = 0; l < prof.n_layers && l < MAX_PROFILE_LAYERS; l++) {
                        for (int e = 0; e < n_exp; e++) {
                            prof.experts[l][e] = em.profiles[p].experts[l][e];
                        }
                        for (int e = n_exp; e < EXPERTS_PER_PROFILE; e++) {
                            prof.experts[l][e] = -1;
                        }
                    }
                    ALPHA_LOG_DBG("[PROFILE] Embedded '%s' (%d layers)\n", prof.name, prof.n_layers);
                    g_n_profiles++;
                }
                ALPHA_LOG_DBG("[PROFILE] %d embedded profiles for %d-expert/%d-layer model\n",
                        g_n_profiles, opt->n_expert, opt->n_moe_layers);
                break;
            }
        }
    }

    if (g_n_profiles == 0) {
        ALPHA_LOG_DBG("[PROFILE] No profiles found (file or embedded) for this model\n");
    }

    // Load all *_dp.json domain profiles from the model directory.
    // Includes MMLU subjects (57), hand-crafted domains (24), core, classify, grammar.
    // No whitelist - any *_dp.json file is loaded as a domain profile.
    //
    g_n_domain_profiles = 0;
    {
        const char * profile_dir = dir_path;

        // Scan directory for *_dp.json files
        char glob_pattern[512];
        snprintf(glob_pattern, sizeof(glob_pattern), "%s", profile_dir);
#ifdef _WIN32
        WIN32_FIND_DATAA fd;
        char search[512];
        snprintf(search, sizeof(search), "%s/*_dp.json", profile_dir);
        HANDLE h = FindFirstFileA(search, &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                if (g_n_domain_profiles >= MAX_DOMAIN_PROFILES) break;
                // Extract domain name: "chemical_dp.json" -> "chemical"
                std::string fname = fd.cFileName;
                std::string domain = fname.substr(0, fname.size() - 8); // remove "_dp.json"
                char path[512];
                snprintf(path, sizeof(path), "%s/%s", profile_dir, fd.cFileName);
#else
        DIR * dir = opendir(profile_dir);
        if (dir) {
            struct dirent * ent;
            while ((ent = readdir(dir)) != nullptr) {
                if (g_n_domain_profiles >= MAX_DOMAIN_PROFILES) break;
                std::string fname = ent->d_name;
                if (fname.size() < 9 || fname.substr(fname.size() - 8) != "_dp.json") continue;
                std::string domain = fname.substr(0, fname.size() - 8);
                char path[512];
                snprintf(path, sizeof(path), "%s/%s", profile_dir, ent->d_name);
#endif
        FILE * f = fopen(path, "r");
        if (!f) {
            fprintf(stderr, "[DOMAIN] %s not found\n", path);
            continue;
        }
        fprintf(stderr, "[DOMAIN] Loading %s...\n", path);

        auto & dp = g_domain_profiles[g_n_domain_profiles];
        memset(&dp, 0, sizeof(dp));
        snprintf(dp.name, sizeof(dp.name), "%s", domain.c_str());
        dp.n_layers = 0;

        char line[8192];
        int current_layer = -1;
        int line_num = 0;
        while (fgets(line, sizeof(line), f)) {
            line_num++;
            char * lk = strstr(line, "\"l\":");
            // Skip false match on "layers": - check char after "l" is quote not alpha
            if (lk) {
                // Verify it's "l": not "la..." or "ly..."
                char after_l = *(lk + 2);  // char after the 'l'
                if (after_l != '"') lk = nullptr;  // "layers", "ly" etc - skip
            }
            if (!lk) lk = strstr(line, "\"layer\":");
            if (lk) {
                char * num = lk;
                while (*num && (*num < '0' || *num > '9')) num++;
                current_layer = atoi(num);
            }
            char * te = strstr(line, "\"te\":");
            if (!te) te = strstr(line, "\"top_experts\":");
            if (te && current_layer >= 0 && current_layer < MAX_PROFILE_LAYERS) {
                char * bracket = strchr(te, '[');
                if (bracket) {
                    int n_parsed = 0;
                    char * p = bracket + 1;
                    while (*p && *p != ']' && n_parsed < EXPERTS_PER_DOMAIN) {
                        while (*p && (*p == ' ' || *p == ',')) p++;
                        if (*p >= '0' && *p <= '9') {
                            dp.experts[current_layer][n_parsed++] = atoi(p);
                            while (*p >= '0' && *p <= '9') p++;
                        } else p++;
                    }
                    if (n_parsed > 0) dp.n_layers = std::max(dp.n_layers, current_layer + 1);
                }
            }
        }
        fclose(f);
        fprintf(stderr, "[DOMAIN] Parsed '%s': n_layers=%d (read %d lines, last current_layer=%d)\n",
                dp.name, dp.n_layers, line_num, current_layer);
        if (dp.n_layers > 0) {
            // Validate profile matches current model:
            // 1. Layer count must match model's MoE layer count
            // 2. Expert IDs must be within model's expert range
            int model_moe_layers = 0;
            int model_n_expert = 0;
            if (opt) {
                for (int il = 0; il < opt->n_layer; il++) {
                    if (opt->layer_info[il].has_experts) model_moe_layers++;
                }
                model_n_expert = opt->n_expert;
            }
            bool valid = true;
            if (model_moe_layers > 0 && dp.n_layers != model_moe_layers) {
                fprintf(stderr, "[DOMAIN] REJECTED '%s': profile has %d layers, model has %d MoE layers — wrong model?\n",
                        dp.name, dp.n_layers, model_moe_layers);
                valid = false;
            }
            if (valid && model_n_expert > 0) {
                for (int il = 0; il < dp.n_layers && valid; il++) {
                    for (int e = 0; e < EXPERTS_PER_DOMAIN; e++) {
                        if (dp.experts[il][e] >= model_n_expert) {
                            fprintf(stderr, "[DOMAIN] REJECTED '%s': expert %d at layer %d >= n_expert %d — wrong model?\n",
                                    dp.name, dp.experts[il][e], il, model_n_expert);
                            valid = false;
                            break;
                        }
                    }
                }
            }
            if (valid) {
                fprintf(stderr, "[DOMAIN] Loaded '%s' domain profile (%d layers)\n", dp.name, dp.n_layers);
                g_n_domain_profiles++;
            }
        } else {
            fprintf(stderr, "[DOMAIN] WARNING: '%s' parsed 0 layers — JSON format issue?\n", dp.name);
        }
#ifdef _WIN32
            } while (FindNextFileA(h, &fd));
            FindClose(h);
        }
#else
            }
            closedir(dir);
        }
#endif
    } // end directory scan block
    if (g_n_domain_profiles > 0) {
        fprintf(stderr, "[DOMAIN] %d domain profiles loaded\n", g_n_domain_profiles);
    }
}

// Classify prompt type from 1 token's argsort and fill cache with matching profile.
// DMA only the delta (experts in profile but not already cached).
void alpha_moe_optimizer_warmup_classify_and_fill(alpha_moe_optimizer * opt) {
    fprintf(stderr, "[WARMUP-DBG] entry: opt=%p is_moe=%d caches=%d\n",
            (void*)opt, opt ? opt->is_moe : -1, opt ? (int)opt->caches.size() : -1);
    if (!opt || !opt->is_moe || opt->caches.empty()) return;
    if (g_n_profiles <= 0) {
        fprintf(stderr, "[WARMUP-DBG] no profiles loaded, falling back to smart fill\n");
        alpha_moe_optimizer_warmup_1token_smart(opt);
        return;
    }

    int n_argsort = alpha_mole_get_argsort_count();
    fprintf(stderr, "[WARMUP-DBG] n_argsort=%d, n_profiles=%d, cache_enabled=%d\n",
            n_argsort, g_n_profiles, opt->cache_enabled);
    int best_profile = -1;
    int dma_count = 0, dma_fired = 0;

    if (n_argsort <= 0) {
        fprintf(stderr, "[WARMUP-DBG] no argsort — using default style profile + domain pinning\n");
        if (g_n_profiles > 0) {
            best_profile = 0;
            g_matched_profile_name = g_profiles[0].name;
        }
    }

    // Steps 1-2: Style profile matching (needs argsort)
    struct LayerRouting { int ids[16]; int n_used; };
    std::vector<LayerRouting> routing;
    int moe_idx = 0;

    if (n_argsort > 0) {

    // Step 1: Read argsort for all layers
    routing.reserve(n_argsort);
    moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx++);
        LayerRouting lr = {};
        if (!argsort_out || !argsort_out->buffer) { routing.push_back(lr); continue; }

        lr.n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || lr.n_used <= 0) { routing.push_back(lr); continue; }

        size_t offset = (size_t)(n_tokens - 1) * argsort_out->nb[1];
        if (offset + lr.n_used * sizeof(int32_t) > ggml_nbytes(argsort_out)) {
            routing.push_back(lr); continue;
        }
        ggml_backend_tensor_get(argsort_out, lr.ids, offset, lr.n_used * sizeof(int32_t));
        routing.push_back(lr);
    }

    // Step 2: Score each profile against this token's routing
    best_profile = -1;
    int best_score = 0;
    int max_score = 0;
    for (auto & lr : routing) max_score += lr.n_used;

    for (int p = 0; p < g_n_profiles; p++) {
        int score = 0;
        int n_compare = std::min((int)routing.size(), g_profiles[p].n_layers);
        for (int l = 0; l < n_compare; l++) {
            for (int e = 0; e < routing[l].n_used; e++) {
                int eid = routing[l].ids[e];
                for (int pe = 0; pe < EXPERTS_PER_PROFILE; pe++) {
                    if (g_profiles[p].experts[l][pe] == eid) { score++; break; }
                }
            }
        }
        if (score > best_score) { best_score = score; best_profile = p; }
    }

    float match_pct = max_score > 0 ? (float)best_score / max_score : 0.0f;
    ALPHA_LOG_DBG("[PROFILE] Best match: '%s' (score %d/%d = %.0f%%)\n",
            best_profile >= 0 ? g_profiles[best_profile].name : "none",
            best_score, max_score, match_pct * 100.0f);

    // Store profile match quality and name for dynamic thresholds
    g_profile_match_quality = match_pct;
    g_matched_profile_name = best_profile >= 0 ? g_profiles[best_profile].name : "none";

    } // end if (n_argsort > 0) - Steps 1-2 need argsort, Steps 3+ don't

    // Step 3: DMA style profile experts (works with or without argsort)
    // If n_argsort=0 (profile-first path), best_profile defaults to 0 (factual).
    if (best_profile < 0) {
        // No profile matched: use smart fill, but do not return
        // (domain pinning below still needs to run)
        alpha_moe_optimizer_warmup_1token_smart(opt);
    }

    dma_count = 0;
    moe_idx = 0;
    if (best_profile < 0) goto skip_style_dma;  // no profile, skip style DMA
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (moe_idx >= g_profiles[best_profile].n_layers) { moe_idx++; continue; }

        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        int ref_op = has_fused ? OP_GATE_UP : (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);
        if (!cache.gpu_tensor[ref_op] || cache.slots[ref_op].empty() ||
            (int)cache.slots[ref_op].size() < opt->K) { moe_idx++; continue; }

        // Load profile's top-33 for this layer - DMA only misses
        for (int pe = 0; pe < EXPERTS_PER_PROFILE; pe++) {
            int eid = g_profiles[best_profile].experts[moe_idx][pe];
            if (eid < 0 || eid >= opt->n_expert) continue;
            if (cache.slot_map[ref_op][eid] >= 0) continue;  // already cached

            // Find LRU slot to evict
            int lru_slot = -1;
            int64_t oldest = INT64_MAX;
            for (int k = 1; k < opt->K; k++) {
                if (cache.slots[ref_op][k].last_used < oldest) {
                    oldest = cache.slots[ref_op][k].last_used;
                    lru_slot = k;
                }
            }
            if (lru_slot < 0) continue;

            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                int old_eid = cache.slots[op][lru_slot].expert_id;
                if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)lru_slot * sb, sb);
                cache.slot_map[op][eid] = lru_slot;
                cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
            }
            dma_count++;
        }
        moe_idx++;
    }

skip_style_dma:
    // Also DMA fired experts that are misses (from this token's argsort)
    dma_fired = 0;
    if (n_argsort > 0 && !routing.empty()) {
    moe_idx = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (moe_idx >= (int)routing.size()) break;

        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        int ref_op = has_fused ? OP_GATE_UP : (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);
        if (!cache.gpu_tensor[ref_op] || cache.slots[ref_op].empty()) { moe_idx++; continue; }

        for (int e = 0; e < routing[moe_idx].n_used; e++) {
            int eid = routing[moe_idx].ids[e];
            if (eid < 0 || eid >= opt->n_expert) continue;
            if (cache.slot_map[ref_op][eid] >= 0) continue;
            // Find LRU
            int lru_slot = -1;
            int64_t oldest = INT64_MAX;
            for (int k = 1; k < opt->K; k++) {
                if (cache.slots[ref_op][k].last_used < oldest) {
                    oldest = cache.slots[ref_op][k].last_used;
                    lru_slot = k;
                }
            }
            if (lru_slot < 0) continue;
            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                int old_eid = cache.slots[op][lru_slot].expert_id;
                if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)lru_slot * sb, sb);
                cache.slot_map[op][eid] = lru_slot;
                cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
            }
            dma_fired++;
        }
        moe_idx++;
    }
    } // end if (n_argsort > 0) for fired experts

    // Step 4: Pin domain-specific experts + grammar priority experts
    int dma_domain = 0;
    if (g_n_domain_profiles > 0) {
        // Always pin grammar experts (quality-critical, not domain-specific)
        // Pin domain experts when keyword-detected domain matches
        // Both use the same pinning mechanism - just different profile selection
        // 3-pass domain pinning: grammar + domain1 + domain2
        // Adaptive EXPERTS_PER_DOMAIN based on K budget:
        // K >= 60: full 20 per domain (3 x 20 = 60)
        // K >= 30: 10 per domain (3 x 10 = 30)
        // K < 30:  skip domain2, reduce to K/2 per domain
        int experts_per_domain = EXPERTS_PER_DOMAIN;  // default 20
        if (opt->K > 0 && opt->K < 60) {
            experts_per_domain = std::max(5, opt->K / 3);
        }
        for (int pass = 0; pass < 3; pass++) {
            int dp_idx = -1;
            if (pass == 0) {
                // Always pin core experts first (common across all domains).
                // Fall back to grammar if no core profile.
                for (int d = 0; d < g_n_domain_profiles; d++) {
                    if (strcmp(g_domain_profiles[d].name, "core") == 0) {
                        dp_idx = d; break;
                    }
                }
                if (dp_idx < 0) {
                    for (int d = 0; d < g_n_domain_profiles; d++) {
                        if (strcmp(g_domain_profiles[d].name, "grammar") == 0) {
                            dp_idx = d; break;
                        }
                    }
                }
            } else if (pass == 1 && g_detected_domain) {
                // Pin primary domain experts (chemical, math, etc.)
                for (int d = 0; d < g_n_domain_profiles; d++) {
                    if (strcmp(g_domain_profiles[d].name, g_detected_domain) == 0) {
                        dp_idx = d; break;
                    }
                }
            } else if (pass == 2 && g_detected_domain2) {
                // Pin secondary domain experts (multi-domain prompts)
                // Skip if K is too small for 3 profiles
                if (opt->K > 0 && opt->K < 30) continue;
                for (int d = 0; d < g_n_domain_profiles; d++) {
                    if (strcmp(g_domain_profiles[d].name, g_detected_domain2) == 0) {
                        dp_idx = d; break;
                    }
                }
            }
            if (dp_idx < 0) continue;

            auto & dp = g_domain_profiles[dp_idx];
            int pin_count = 0;
            int moe_idx_pin = 0;
            for (int il = 0; il < opt->n_layer; il++) {
                auto & info = opt->layer_info[il];
                auto & cache = opt->caches[il];
                if (!info.has_experts || info.on_gpu) continue;
                if (moe_idx_pin >= dp.n_layers) { moe_idx_pin++; continue; }

                bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
                int ref_op = has_fused ? OP_GATE_UP : (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);
                if (!cache.gpu_tensor[ref_op] || cache.slots[ref_op].empty() ||
                    (int)cache.slots[ref_op].size() < opt->K) { moe_idx_pin++; continue; }

                for (int pe = 0; pe < experts_per_domain; pe++) {
                    int eid = dp.experts[moe_idx_pin][pe];
                    if (eid < 0 || eid >= opt->n_expert) continue;
                    if (cache.slot_map[ref_op][eid] >= 0) continue;

                    int lru_slot = -1;
                    int64_t oldest = INT64_MAX;
                    for (int k = 1; k < opt->K; k++) {
                        if (cache.slots[ref_op][k].last_used < oldest) {
                            oldest = cache.slots[ref_op][k].last_used;
                            lru_slot = k;
                        }
                    }
                    if (lru_slot < 0) continue;

                    for (int op = 0; op < OP_COUNT; op++) {
                        if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                        if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                        size_t sb = info.expert_slice_bytes[op];
                        int old_eid = cache.slots[op][lru_slot].expert_id;
                        if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                        ggml_backend_tensor_set(cache.gpu_tensor[op],
                            (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                            (size_t)lru_slot * sb, sb);
                        cache.slot_map[op][eid] = lru_slot;
                        cache.slots[op][lru_slot] = { eid, opt->cache_clock + 1000000 };
                    }
                    pin_count++;
                }
                moe_idx_pin++;
            }
            dma_domain += pin_count;
            fprintf(stderr, "[DOMAIN] Pinned %d '%s' experts\n",
                    pin_count, g_domain_profiles[dp_idx].name);
        } // end for pass 0,1
    }

    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }

    if (n_argsort > 0) {
        ALPHA_LOG_DBG("[WARMUP] Profile '%s': %d profile + %d fired + %d domain = %d total DMA\n",
                g_profiles[best_profile].name, dma_count, dma_fired, dma_domain,
                dma_count + dma_fired + dma_domain);
    }
}

void alpha_moe_set_detected_domain(const char * domain) { g_detected_domain = domain; }
const char * alpha_moe_get_detected_domain() { return g_detected_domain; }
const char * alpha_moe_get_detected_domain2() { return g_detected_domain2; }

void alpha_moe_optimizer_pin_domain_experts(alpha_moe_optimizer * opt, const char * model_dir) {
    if (!opt || !opt->is_moe || !opt->cache_enabled) return;

    // Domain profiles must be loaded at startup (serve init loads *_dp.json files).
    // Do not enter draft mode here: this is called before prompt processing. Draft mode changes
    // the graph topology (ne[2]=K), so the prompt batch would need exit_draft + reset_sched, which
    // leaks VRAM on every request.
    // Draft mode is entered later by apply_warmup after prompt is processed.

    // Pin grammar + domain1 + domain2 experts using the warmup pinning logic.
    // This is the same 3-pass loop from warmup_classify_and_fill, but callable
    // independently before prompt decode (profile-first prefill).
    int experts_per_domain = EXPERTS_PER_DOMAIN;
    if (opt->K > 0 && opt->K < 60) {
        experts_per_domain = std::max(5, opt->K / 3);
    }

    // Pass 0: core experts (basic language, always loaded first)
    // Pass 1: grammar experts
    // Pass 2: domain1 experts (detected domain - classify or actual domain)
    // Pass 3: domain2 experts (secondary domain)
    for (int pass = 0; pass < 4; pass++) {
        int dp_idx = -1;
        if (pass == 0) {
            // Core experts - basic language needed for any prompt
            for (int d = 0; d < g_n_domain_profiles; d++) {
                if (strcmp(g_domain_profiles[d].name, "core") == 0) { dp_idx = d; break; }
            }
        } else if (pass == 1) {
            for (int d = 0; d < g_n_domain_profiles; d++) {
                if (strcmp(g_domain_profiles[d].name, "grammar") == 0) { dp_idx = d; break; }
            }
        } else if (pass == 2 && g_detected_domain) {
            for (int d = 0; d < g_n_domain_profiles; d++) {
                if (strcmp(g_domain_profiles[d].name, g_detected_domain) == 0) { dp_idx = d; break; }
            }
        } else if (pass == 3 && g_detected_domain2) {
            if (opt->K > 0 && opt->K < 30) continue;
            for (int d = 0; d < g_n_domain_profiles; d++) {
                if (strcmp(g_domain_profiles[d].name, g_detected_domain2) == 0) { dp_idx = d; break; }
            }
        }
        if (dp_idx < 0) continue;

        auto & dp = g_domain_profiles[dp_idx];
        int pin_count = 0;
        int moe_idx = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            auto & cache = opt->caches[il];
            if (!info.has_experts || info.on_gpu) continue;
            if (moe_idx >= dp.n_layers) { moe_idx++; continue; }

            bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
            int ref_op = has_fused ? OP_GATE_UP : (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);
            if (!cache.gpu_tensor[ref_op] || cache.slots[ref_op].empty() ||
                (int)cache.slots[ref_op].size() < opt->K) { moe_idx++; continue; }

            for (int pe = 0; pe < experts_per_domain; pe++) {
                int eid = dp.experts[moe_idx][pe];
                if (eid < 0 || eid >= opt->n_expert) continue;
                if (cache.slot_map[ref_op][eid] >= 0) continue;  // already cached

                int lru_slot = -1;
                int64_t oldest = INT64_MAX;
                for (int k = 1; k < opt->K; k++) {
                    if (cache.slots[ref_op][k].last_used < oldest) {
                        oldest = cache.slots[ref_op][k].last_used;
                        lru_slot = k;
                    }
                }
                if (lru_slot < 0) continue;

                int64_t pin_time = opt->cache_clock + 1000000;

                for (int op = 0; op < OP_COUNT; op++) {
                    if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                    if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                    size_t sb = info.expert_slice_bytes[op];
                    int old_eid = cache.slots[op][lru_slot].expert_id;
                    if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                    ggml_backend_tensor_set(cache.gpu_tensor[op],
                        (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                        (size_t)lru_slot * sb, sb);
                    cache.slot_map[op][eid] = lru_slot;
                    cache.slots[op][lru_slot] = { eid, pin_time };
                }
                pin_count++;
            }
            moe_idx++;
        }
        fprintf(stderr, "[DOMAIN] Pre-prompt pinned %d '%s' experts\n",
                pin_count, g_domain_profiles[dp_idx].name);
    }

    // Update LUT from cache state so prompt decode uses pinned experts
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }
}

// Serial expert tiling: load a specific partition of experts ranked by frequency.
// pass_idx=0: top-K most-used experts. pass_idx=1: next K. pass_idx=2: next K.
// Over N passes covers N*K out of n_expert (e.g., 3x55=165/256 = 64%).
void alpha_moe_optimizer_load_expert_partition(alpha_moe_optimizer * opt, int pass_idx) {
    if (!opt || !opt->is_moe || !opt->cache_enabled) return;

    int total_loaded = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;

        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        int ref_op = has_fused ? OP_GATE_UP : (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);
        if (!cache.gpu_tensor[ref_op] || cache.slots[ref_op].empty()) continue;

        // Rank experts by activation frequency for this layer
        std::vector<std::pair<int64_t, int>> ranked(opt->n_expert);
        for (int e = 0; e < opt->n_expert; e++) {
            int64_t freq = (il < (int)opt->expert_activation_count.size() &&
                            e < (int)opt->expert_activation_count[il].size())
                           ? opt->expert_activation_count[il][e] : 0;
            ranked[e] = { freq, e };
        }
        std::sort(ranked.begin(), ranked.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        // Partition: skip first pass_idx*K experts, load next K
        int start = pass_idx * (opt->K - 1);  // -1 for zero-expert slot 0
        int end = std::min(start + opt->K - 1, opt->n_expert);
        if (start >= opt->n_expert) continue;

        // Clear cache for this layer
        for (int op = 0; op < OP_COUNT; op++) {
            for (auto & sm : cache.slot_map[op]) sm = -1;
            for (auto & s : cache.slots[op]) { s.expert_id = -1; s.last_used = 0; }
        }

        // Load partition experts into slots 1..K-1 (slot 0 = zero expert)
        int slot = 1;
        for (int r = start; r < end && slot < opt->K; r++) {
            int eid = ranked[r].second;
            if (eid < 0 || eid >= opt->n_expert) continue;
            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)slot * sb, sb);
                cache.slot_map[op][eid] = slot;
                cache.slots[op][slot] = { eid, opt->cache_clock + 1000000 };
            }
            slot++;
            total_loaded++;
        }
    }

    // Update LUT + enter draft mode
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }
    if (!opt->spec_draft_active && opt->spec_lut_ready) {
        alpha_moe_optimizer_enter_draft_mode(opt);
    }
    (void)pass_idx; // partition loaded
}

int alpha_moe_optimizer_get_K(const alpha_moe_optimizer * opt) {
    return opt ? opt->K : 0;
}

int alpha_moe_optimizer_get_n_expert(const alpha_moe_optimizer * opt) {
    return opt ? opt->n_expert : 0;
}

bool alpha_moe_optimizer_has_domain_profile(const char * name) {
    if (!name) return false;
    for (int d = 0; d < g_n_domain_profiles; d++) {
        if (strcmp(g_domain_profiles[d].name, name) == 0) return true;
    }
    return false;
}

int alpha_moe_optimizer_get_domain_profile_count() {
    return g_n_domain_profiles;
}

const char * alpha_moe_optimizer_get_domain_profile_name(int idx) {
    if (idx < 0 || idx >= g_n_domain_profiles) return nullptr;
    return g_domain_profiles[idx].name;
}

void alpha_moe_optimizer_prefetch_prompt_experts(alpha_moe_optimizer * opt) {
#if !defined(_WIN32)
    if (!opt || !opt->is_moe) return;
    // Prefetch all expert tensor pages for all MoE layers.
    // For prompt processing with CPU experts (cache disabled), every expert
    // access is a potential mmap page fault. Calling madvise(MADV_WILLNEED)
    // tells the kernel to start loading pages from disk asynchronously.
    // Early layers' pages load while later layers' prefetch is issued.
    int prefetched = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        for (int op = 0; op < OP_COUNT; op++) {
            ggml_tensor * t = info.orig_exps[op];
            if (!t || !t->data || !ggml_backend_buffer_is_host(t->buffer)) continue;
            // Prefetch the entire expert tensor (all experts for this layer+op)
            // This is aggressive but prompt processing touches most experts
            size_t total_bytes = ggml_nbytes(t);
            if (total_bytes == 0) continue;
            uintptr_t pa = (uintptr_t)t->data & ~(uintptr_t)4095;
            size_t len = total_bytes + ((uintptr_t)t->data - pa);
            madvise((void *)pa, len, MADV_WILLNEED);
            prefetched++;
        }
    }
    if (prefetched > 0) {
        fprintf(stderr, "[PREFETCH] Issued madvise for %d expert tensors (prompt pipeline)\n", prefetched);
    }
#else
    (void)opt;
#endif
}

void alpha_moe_optimizer_evict_expert_pages(alpha_moe_optimizer * opt) {
#if !defined(_WIN32)
    if (!opt || !opt->is_moe) return;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        for (int op = 0; op < OP_COUNT; op++) {
            ggml_tensor * t = info.orig_exps[op];
            if (t && t->data) {
                madvise(t->data, ggml_nbytes(t), MADV_DONTNEED);
            }
        }
    }
#else
    (void)opt;
#endif
}

void alpha_moe_optimizer_clear_cache(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || !opt->cache_enabled) return;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & cache = opt->caches[il];
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        for (int op = 0; op < OP_COUNT; op++) {
            if (!cache.gpu_tensor[op]) continue;
            memset(cache.slot_map[op], -1, sizeof(cache.slot_map[op]));
            for (int k = 0; k < opt->K; k++) {
                cache.slots[op][k] = { -1, 0 };
            }
        }
    }
    // Reset draft mode so pin_domain_experts will re-enter + update LUT
    opt->spec_draft_active = false;
}

bool alpha_moe_optimizer_has_profiles() { return g_n_profiles > 0; }
float alpha_moe_optimizer_get_profile_quality() { return g_profile_match_quality; }
const char * alpha_moe_optimizer_get_profile_name() { return g_matched_profile_name; }

// Accumulate expert usage from the last decode without DMA.
// Call after each cpu-moe warmup token to build frequency counts.
// Then call warmup_fill_cache_topk() once after all warmup tokens to load optimal set.
static std::vector<std::vector<int>> g_warmup_freq;  // [moe_layer][expert_id] = count

void alpha_moe_optimizer_warmup_accumulate(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe) return;

    int n_argsort = alpha_mole_get_argsort_count();
    if (n_argsort <= 0) return;

    // Init frequency table on first call
    if (g_warmup_freq.empty()) {
        int n_moe = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) n_moe++;
        }
        g_warmup_freq.resize(n_moe, std::vector<int>(opt->n_expert, 0));
    }

    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (moe_idx >= (int)g_warmup_freq.size()) break;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx);
        if (!argsort_out || !argsort_out->buffer) { moe_idx++; continue; }

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || n_used <= 0) { moe_idx++; continue; }

        int32_t ids[16] = {};
        size_t offset = (size_t)(n_tokens - 1) * argsort_out->nb[1];
        if (offset + n_used * sizeof(int32_t) > ggml_nbytes(argsort_out)) { moe_idx++; continue; }
        ggml_backend_tensor_get(argsort_out, ids, offset, n_used * sizeof(int32_t));

        for (int e = 0; e < n_used; e++) {
            int eid = ids[e];
            if (eid >= 0 && eid < opt->n_expert) {
                g_warmup_freq[moe_idx][eid]++;
            }
        }
        moe_idx++;
    }
}

// Fill cache with top-K experts per layer from accumulated warmup frequency data.
// Call once after all warmup tokens are decoded.
void alpha_moe_optimizer_warmup_fill_cache_topk(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || opt->caches.empty() || g_warmup_freq.empty()) return;

    int dma_count = 0;
    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (moe_idx >= (int)g_warmup_freq.size()) break;

        // Sort experts by frequency (descending) for this layer
        std::vector<std::pair<int, int>> freq_sorted(opt->n_expert);
        for (int e = 0; e < opt->n_expert; e++) {
            freq_sorted[e] = {g_warmup_freq[moe_idx][e], e};
        }
        std::partial_sort(freq_sorted.begin(),
            freq_sorted.begin() + std::min(opt->K - 1, opt->n_expert),  // -1 for slot 0 (zero expert)
            freq_sorted.end(),
            [](const auto & a, const auto & b) { return a.first > b.first; });

        // Clear existing cache state for this layer
        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        for (int op = 0; op < OP_COUNT; op++) {
            if (!cache.gpu_tensor[op]) continue;
            for (int k = 0; k < opt->K; k++) {
                if (cache.slots[op][k].expert_id >= 0) {
                    cache.slot_map[op][cache.slots[op][k].expert_id] = -1;
                }
                cache.slots[op][k] = {-1, 0};
            }
        }

        // Load top K-1 experts into slots 1..K-1 (slot 0 = zero expert)
        int n_load = std::min(opt->K - 1, opt->n_expert);
        for (int k = 0; k < n_load; k++) {
            int eid = freq_sorted[k].second;
            int freq = freq_sorted[k].first;
            if (freq <= 0) break;  // no more active experts
            int slot = k + 1;  // skip slot 0

            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];

                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)slot * sb, sb);

                cache.slot_map[op][eid] = slot;
                cache.slots[op][slot] = { eid, ++opt->cache_clock };
            }
            dma_count++;
        }
        moe_idx++;
    }

    // Update LUT
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }

    // Log top-3 per first layer for debugging
    if (!g_warmup_freq.empty() && !g_warmup_freq[0].empty()) {
        ALPHA_LOG_DBG("[WARMUP] Optimal fill: %d experts loaded (top-%d per layer from %d warmup tokens)\n",
                dma_count, std::min(opt->K - 1, opt->n_expert),
                g_warmup_freq[0][0] > 0 ? -1 : 0);  // placeholder
    }

    // Clear frequency data
    g_warmup_freq.clear();

    ALPHA_LOG_DBG("[WARMUP] DMA'd %d experts into GPU cache (frequency-optimal)\n", dma_count);
}

// Minimal argsort reader for prompt profiling: only reads argsort (marked as output) and does not
// touch gating_probs, momentum, cache state, or any other optimizer internals.
// Safe to call during cpu-moe prompt decode with cache_enabled=false.
void alpha_moe_optimizer_record_prompt_activations(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe) return;
    int n_argsort = alpha_mole_get_argsort_count();
    if (n_argsort <= 0) return;

    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        if (!opt->layer_info[il].has_experts) continue;
        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx++);
        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || n_used <= 0) continue;

        // Read only the last token's expert IDs
        int32_t ids[16] = {};
        size_t offset = (size_t)(n_tokens - 1) * argsort_out->nb[1];
        if (offset + n_used * sizeof(int32_t) > ggml_nbytes(argsort_out)) continue;
        ggml_backend_tensor_get(argsort_out, ids, offset, n_used * sizeof(int32_t));

        for (int e = 0; e < n_used; e++) {
            if (ids[e] >= 0 && ids[e] < opt->n_expert) {
                opt->expert_activation_count[il][ids[e]]++;
            }
        }
    }
}

void alpha_moe_optimizer_record_prompt(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe) return;

    int n_argsort = alpha_mole_get_argsort_count();
    if (n_argsort <= 0) return;
    int moe_idx = 0;
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx);
        moe_idx++;
        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || n_used <= 0) continue;

        // Safety: check tensor has valid buffer and reasonable size
        size_t read_bytes = (size_t)n_used * n_tokens * sizeof(int32_t);
        if (read_bytes > ggml_nbytes(argsort_out)) {
            ALPHA_LOG_DBG("[PREFILL-CACHE] layer %d: argsort size mismatch (%zu > %zu), skip\n",
                    il, read_bytes, ggml_nbytes(argsort_out));
            continue;
        }

        // Read all tokens' expert IDs (not just the last)
        std::vector<int32_t> all_ids(n_used * n_tokens);
        ggml_backend_tensor_get(argsort_out, all_ids.data(), 0, read_bytes);

        for (int t = 0; t < n_tokens; t++) {
            for (int e = 0; e < n_used; e++) {
                int eid = all_ids[t * n_used + e];
                if (eid >= 0 && eid < opt->n_expert) {
                    opt->expert_activation_count[il][eid]++;
                }
            }
        }
    }
}

// Fill GPU cache per-layer from prefill activation counts.
// Each layer gets its top-(K-1) most-activated experts loaded.
void alpha_moe_optimizer_fill_cache_from_prefill(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || opt->caches.empty()) return;

    int total_loaded = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;

        // Sort experts by activation count for this layer (descending)
        auto & counts = opt->expert_activation_count[il];
        std::vector<std::pair<int64_t, int>> sorted;
        for (int e = 0; e < opt->n_expert; e++) {
            if (counts[e] > 0) sorted.push_back({counts[e], e});
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });

        // Clear existing slot_map
        bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
        for (int op = 0; op < OP_COUNT; op++) {
            if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
            for (int e = 0; e < opt->n_expert; e++) cache.slot_map[op][e] = -1;
            for (int k = 0; k < opt->K; k++) cache.slots[op][k] = { -1, 0 };
            // Slot 0 = zero expert
            cache.slots[op][0] = { -2, INT64_MAX };
        }

        // Load top-(K-1) into slots 1..K-1
        int n_load = std::min(opt->K - 1, (int)sorted.size());
        for (int k = 0; k < n_load; k++) {
            int eid = sorted[k].second;
            int slot = k + 1;  // slot 0 reserved for zero expert
            for (int op = 0; op < OP_COUNT; op++) {
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                size_t sb = info.expert_slice_bytes[op];
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)slot * sb, sb);
                cache.slot_map[op][eid] = slot;
                cache.slots[op][slot] = { eid, ++opt->cache_clock };
            }
            total_loaded++;
        }
    }

    // Update LUT to reflect new cache state
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
    }

    ALPHA_LOG_DBG("[PREFILL-CACHE] Loaded %d prompt-specific experts into GPU cache (K=%d)\n",
            total_loaded, opt->K);
}

bool alpha_moe_optimizer_is_moe(const alpha_moe_optimizer * opt) {
    return opt && opt->is_moe;
}

void alpha_moe_optimizer_print_stats(const alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe) return;

    ALPHA_LOG_DBG( "\n=== MoE Optimizer Stats ===\n");
    ALPHA_LOG_DBG( "  Tokens: %d\n", opt->total_tokens);
    ALPHA_LOG_DBG( "  Cache: K=%d, %s\n", opt->K, opt->cache_enabled ? "enabled" : "disabled");
    if (opt->cache_hits + opt->cache_misses > 0) {
        int64_t total = opt->cache_hits + opt->cache_misses;
        ALPHA_LOG_DBG( "  Hits: %lld, Misses: %lld (%.1f%% hit rate)\n",
            (long long)opt->cache_hits, (long long)opt->cache_misses,
            (float)opt->cache_hits / total * 100.0f);
    }
    // Show average difficulty across layers
    if (!opt->prev_entropy.empty()) {
        float avg_entropy = 0;
        int n = 0;
        for (float e : opt->prev_entropy) {
            if (e >= 0) { avg_entropy += e; n++; }
        }
        if (n > 0) avg_entropy /= n;
        const char * diff_label = avg_entropy < 0.3f ? "easy" : avg_entropy < 0.6f ? "moderate" : avg_entropy < 0.8f ? "hard" : "very hard";
        ALPHA_LOG_DBG( "  Difficulty: %.2f (%s) — last token routing entropy\n", avg_entropy, diff_label);
    }

    if (opt->groups_enabled && (opt->group_cache_hits + opt->group_cache_misses) > 0) {
        int64_t gtot = opt->group_cache_hits + opt->group_cache_misses;
        ALPHA_LOG_DBG( "  Groups: %lld hits, %lld loads (%.1f%% group hit rate)\n",
            (long long)opt->group_cache_hits, (long long)opt->group_cache_misses,
            (float)opt->group_cache_hits / gtot * 100.0f);
    }
    if (opt->config.mobile_mode) {
        const char * policy_str[] = { "NORMAL", "THROTTLED", "EMERGENCY" };
        ALPHA_LOG_DBG( "  Thermal: policy=%s, temp=%.1f°C\n",
                policy_str[opt->thermal_policy], opt->current_temp);
        // fprintf(stderr, "  Hot experts: %d pinned, %lld substitutions, %lld skipped\n",
        //         (int)opt->hot_experts.size(),
        //         (long long)opt->thermal_substituted,
        //         (long long)opt->thermal_skipped);

        // Print top-10 most active experts (aggregated across layers)
        if (opt->total_tokens > 10) {
            std::vector<int64_t> total_acts(opt->n_expert, 0);
            for (int il = 0; il < opt->n_layer; il++) {
                for (int e = 0; e < opt->n_expert; e++) {
                    total_acts[e] += opt->expert_activation_count[il][e];
                }
            }
            std::vector<int> sorted(opt->n_expert);
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(),
                [&](int a, int b) { return total_acts[a] > total_acts[b]; });

            int64_t sum = 0;
            for (auto & a : total_acts) sum += a;
            ALPHA_LOG_DBG( "  Expert activation (top-10):");
            for (int i = 0; i < std::min(10, opt->n_expert); i++) {
                ALPHA_LOG_DBG( " e%d=%.1f%%", sorted[i],
                        sum > 0 ? 100.0f * total_acts[sorted[i]] / sum : 0.0f);
            }
            ALPHA_LOG_DBG( "\n");
        }
    }
    if (opt->prefetch_enabled && opt->prefetch_total > 0) {
        ALPHA_LOG_DBG( "  Prefetch: %lld/%lld cache hits (%.1f%% pre-warmed)\n",
                (long long)opt->prefetch_hits, (long long)opt->prefetch_total,
                100.0f * opt->prefetch_hits / opt->prefetch_total);
    }
    ALPHA_LOG_DBG( "\n");
}

void alpha_moe_optimizer_disable_cache(alpha_moe_optimizer * opt) {
    if (!opt || !opt->cache_enabled) return;
    // Restore original CPU expert tensors for prompt processing
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        auto & layer = opt->model->layers[il];
        if (info.orig_exps[OP_GATE])    layer.ffn_gate_exps    = info.orig_exps[OP_GATE];
        if (info.orig_exps[OP_UP])      layer.ffn_up_exps      = info.orig_exps[OP_UP];
        if (info.orig_exps[OP_DOWN])    layer.ffn_down_exps    = info.orig_exps[OP_DOWN];
        if (info.orig_exps[OP_GATE_UP]) layer.ffn_gate_up_exps = info.orig_exps[OP_GATE_UP];
        layer.ffn_cache_expert_ids = nullptr;
        layer.ffn_cache_expert_ids_down = nullptr;
    }
    // Temporarily disable cache so callback is a no-op during cpu-moe processing
    opt->cache_enabled = false;
}

void alpha_moe_optimizer_enable_cache(alpha_moe_optimizer * opt) {
    if (!opt || opt->caches.empty()) return;  // no cache was ever allocated
    opt->cache_enabled = true;  // re-enable (may have been disabled by disable_cache)
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;
        auto & layer = opt->model->layers[il];
        if (opt->use_scratch) {
            // Scratch mode: down = per-layer cache, gate/up = shared scratch tensors
            if (cache.gpu_tensor[OP_DOWN])  layer.ffn_down_exps = cache.gpu_tensor[OP_DOWN];
            if (opt->gate_scratch)          layer.ffn_gate_exps = opt->gate_scratch;
            if (opt->up_scratch)            layer.ffn_up_exps   = opt->up_scratch;
        } else if (opt->tiered_cache) {
            auto & gc = opt->global_gate_up_cache;
            if (cache.gpu_tensor[OP_DOWN]) layer.ffn_down_exps = cache.gpu_tensor[OP_DOWN];
            if (gc.gpu_tensor[OP_GATE])    layer.ffn_gate_exps = gc.gpu_tensor[OP_GATE];
            if (gc.gpu_tensor[OP_UP])      layer.ffn_up_exps   = gc.gpu_tensor[OP_UP];
        } else {
            if (cache.gpu_tensor[OP_GATE])    layer.ffn_gate_exps    = cache.gpu_tensor[OP_GATE];
            if (cache.gpu_tensor[OP_UP])      layer.ffn_up_exps      = cache.gpu_tensor[OP_UP];
            if (cache.gpu_tensor[OP_DOWN])    layer.ffn_down_exps    = cache.gpu_tensor[OP_DOWN];
            if (cache.gpu_tensor[OP_GATE_UP]) layer.ffn_gate_up_exps = cache.gpu_tensor[OP_GATE_UP];
        }
        if (opt->cache_expert_ids[il]) {
            layer.ffn_cache_expert_ids = opt->cache_expert_ids[il];
        }
        if ((opt->use_scratch || opt->tiered_cache) &&
            il < (int)opt->cache_expert_ids_down.size() && opt->cache_expert_ids_down[il]) {
            layer.ffn_cache_expert_ids_down = opt->cache_expert_ids_down[il];
        }
    }
}

void alpha_moe_optimizer_set_prefetch(alpha_moe_optimizer * opt, bool enabled, int depth) {
    if (!opt) return;
    opt->prefetch_enabled = enabled;
    opt->prefetch_depth = depth > 0 ? depth : 1;
    if (enabled) {
        ALPHA_LOG_DBG( "%s: expert prefetch enabled (depth=%d)\n", __func__, opt->prefetch_depth);
    }
}

void alpha_moe_optimizer_set_thermal_policy(alpha_moe_optimizer * opt, alpha_moe_thermal_policy policy) {
    if (!opt) return;
    opt->thermal_policy = policy;
}

void alpha_moe_optimizer_set_temperature(alpha_moe_optimizer * opt, float temp_celsius) {
    if (!opt) return;
    opt->current_temp = temp_celsius;

    // Auto-govern policy from temperature
    float t_throttle = opt->config.thermal_throttle_temp > 0 ? opt->config.thermal_throttle_temp : 38.0f;
    float t_emergency = opt->config.thermal_emergency_temp > 0 ? opt->config.thermal_emergency_temp : 43.0f;

    alpha_moe_thermal_policy new_policy;
    if (temp_celsius >= t_emergency) {
        new_policy = ALPHA_MOE_THERMAL_EMERGENCY;
    } else if (temp_celsius >= t_throttle) {
        new_policy = ALPHA_MOE_THERMAL_THROTTLE;
    } else {
        new_policy = ALPHA_MOE_THERMAL_NORMAL;
    }

    if (new_policy != opt->thermal_policy) {
        const char * names[] = { "NORMAL", "THROTTLED", "EMERGENCY" };
        ALPHA_LOG_DBG( "alpha_moe: thermal policy %s → %s (%.1f°C)\n",
                names[opt->thermal_policy], names[new_policy], temp_celsius);
        opt->thermal_policy = new_policy;
    }
}

alpha_moe_thermal_policy alpha_moe_optimizer_get_thermal_policy(const alpha_moe_optimizer * opt) {
    return opt ? opt->thermal_policy : ALPHA_MOE_THERMAL_NORMAL;
}

float alpha_moe_optimizer_get_difficulty(const alpha_moe_optimizer * opt, int layer) {
    if (!opt || !opt->is_moe || layer < 0 || layer >= (int)opt->prev_entropy.size()) return 0.5f;
    return opt->prev_entropy[layer];
}

const int * alpha_moe_optimizer_get_prev_routing(const alpha_moe_optimizer * opt, int layer) {
    if (!opt || !opt->is_moe || layer < 0 || layer >= (int)opt->prev_routing.size()) return nullptr;
    if (opt->prev_routing[layer].empty()) return nullptr;
    return opt->prev_routing[layer].data();
}

int alpha_moe_optimizer_get_n_expert_used(const alpha_moe_optimizer * opt) {
    if (!opt) return 0;
    return opt->n_expert_used;
}

int alpha_moe_optimizer_get_n_layer(const alpha_moe_optimizer * opt) {
    if (!opt) return 0;
    return opt->n_layer;
}

int alpha_moe_optimizer_get_n_moe_layers(const alpha_moe_optimizer * opt) {
    if (!opt) return 0;
    return opt->n_moe_layers;  // layers with experts not on GPU
}

float * alpha_moe_optimizer_get_expert_frequencies(const alpha_moe_optimizer * opt, int layer, int * n_expert) {
    if (!opt || !opt->is_moe || layer < 0 || layer >= opt->n_layer) {
        if (n_expert) *n_expert = 0;
        return nullptr;
    }

    if (n_expert) *n_expert = opt->n_expert;

    int64_t total = 0;
    for (int e = 0; e < opt->n_expert; e++) {
        total += opt->expert_activation_count[layer][e];
    }

    if (total == 0) return nullptr;

    float * freqs = (float *)malloc(opt->n_expert * sizeof(float));
    for (int e = 0; e < opt->n_expert; e++) {
        freqs[e] = (float)opt->expert_activation_count[layer][e] / (float)total;
    }
    return freqs;
}

void alpha_moe_optimizer_set_preload_experts(
        alpha_moe_optimizer * opt, int layer, const int * expert_ids, int count) {
    if (!opt || !opt->is_moe || layer < 0 || layer >= opt->n_layer || !expert_ids || count <= 0) return;
    if (opt->preload_expert_ids.empty()) {
        opt->preload_expert_ids.resize(opt->n_layer);
    }
    opt->preload_expert_ids[layer].assign(expert_ids, expert_ids + count);
}

void alpha_moe_optimizer_set_cpu_moe_verify(alpha_moe_optimizer * opt, bool enable) {
    if (!opt) return;
    opt->cpu_moe_verify = enable;
    if (enable) {
        ALPHA_LOG_DBG("[MOE] cpu-moe verify enabled for cached-spec\n");
    }
}

void alpha_moe_optimizer_free(alpha_moe_optimizer * opt) {
    if (!opt) return;
    // For non-MoE models, layer_info was never populated - skip tensor restore
    if (!opt->is_moe) { delete opt; return; }
    // Restore original tensor pointers
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts) continue;
        auto & layer = opt->model->layers[il];
        if (info.orig_exps[OP_GATE])    layer.ffn_gate_exps    = info.orig_exps[OP_GATE];
        if (info.orig_exps[OP_UP])      layer.ffn_up_exps      = info.orig_exps[OP_UP];
        if (info.orig_exps[OP_DOWN])    layer.ffn_down_exps    = info.orig_exps[OP_DOWN];
        if (info.orig_exps[OP_GATE_UP]) layer.ffn_gate_up_exps = info.orig_exps[OP_GATE_UP];
        layer.ffn_cache_expert_ids = nullptr;
        layer.ffn_cache_expert_ids_down = nullptr;
    }
    // Free pinned staging buffers
    if (opt->staging_enabled && opt->staging_bufs) {
        for (int i = 0; i < opt->staging_buf_count; i++) {
            if (opt->staging_bufs[i]) _acpf(opt->staging_bufs[i]);
        }
        free(opt->staging_bufs);
        opt->staging_bufs = nullptr;
        opt->staging_enabled = false;
    }
    if (opt->cache_ctx) ggml_free(opt->cache_ctx);
    if (opt->remap_ctx) ggml_free(opt->remap_ctx);
    opt->cache_bufs.clear();
    delete opt;
}

// Sequential prompt processing: decode tokens one at a time with cache enabled.
// This ensures GPU-consistent execution for HC-sensitive models (DeepSeek V4)
// where CPU->GPU backend transitions cause quality degradation.
int alpha_moe_optimizer_process_prompt_sequential(
        alpha_moe_optimizer * opt, struct llama_context * ctx,
        const llama_token * tokens, int n_tokens) {
    if (!opt || !ctx || !tokens || n_tokens <= 0) return -1;

    // If cache isn't enabled, fall back to batch decode
    if (!opt->cache_enabled) {
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens), n_tokens);
        return llama_decode(ctx, batch);
    }

    // Sequential processing: each token decoded individually with cache.
    // Note: token 0 uses initial prediction (possibly wrong). For models
    // where this matters (V4), the quality depends on prediction accuracy.
    for (int i = 0; i < n_tokens; i++) {
        alpha_moe_optimizer_prepare(opt);

        llama_token tok = tokens[i];
        llama_batch batch = llama_batch_get_one(&tok, 1);
        int rc = llama_decode(ctx, batch);
        if (rc != 0) return rc;

        alpha_moe_optimizer_record(opt);
    }
    return 0;
}

int alpha_moe_optimizer_process_prompt_batched(
        alpha_moe_optimizer * opt, struct llama_context * ctx,
        const llama_token * tokens, int n_tokens) {
    if (!opt || !ctx || !tokens || n_tokens <= 0) return -1;

    // If cache isn't enabled, fall back to regular batch decode
    if (!opt->cache_enabled) {
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens), n_tokens);
        return llama_decode(ctx, batch);
    }

    // Process in chunks that fit the cache_expert_ids tensor (max_tokens=32).
    const int max_chunk = 32;
    opt->prompt_mode = true;
    int offset = 0;
    while (offset < n_tokens) {
        int chunk = std::min(max_chunk, n_tokens - offset);

        alpha_moe_optimizer_prepare(opt);

        llama_batch batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens + offset), chunk);
        int rc = llama_decode(ctx, batch);
        if (rc != 0) { opt->prompt_mode = false; return rc; }

        alpha_moe_optimizer_record(opt);
        offset += chunk;
    }
    opt->prompt_mode = false;
    return 0;
}

// Reactive expert cache: eval callback that fires after gating (argsort).
// Reads actual expert IDs, loads correct experts to GPU cache, writes remapped IDs.
// CPU fallback: compute one expert's FFN on CPU using ggml ops (handles Q4_K_M dequant).
// Returns expert output in `output` buffer [n_embd floats].
static void compute_one_expert_cpu(
        const moe_layer_info & info, int expert_id,
        int64_t n_embd, int64_t n_ff,
        const float * hidden_state, float * output) {
    // Allocate temp context (enough for tensors + compute workspace)
    size_t ctx_size = 512 * 1024 + n_embd * n_ff * 8;  // generous
    struct ggml_init_params params = { ctx_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return;

    // Input tensor (F32, on CPU - data is in the context's buffer)
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    memcpy(input->data, hidden_state, n_embd * sizeof(float));

    ggml_tensor * result = nullptr;

    if (info.orig_exps[OP_GATE_UP]) {
        // Merged gate+up path: [n_embd, n_ff*2, n_expert]
        ggml_tensor * gate_up_all = info.orig_exps[OP_GATE_UP];
        size_t sb = info.expert_slice_bytes[OP_GATE_UP];
        ggml_tensor * gate_up_w = ggml_view_2d(ctx, gate_up_all,
            gate_up_all->ne[0], gate_up_all->ne[1],
            gate_up_all->nb[1], expert_id * sb);

        ggml_tensor * gate_up = ggml_mul_mat(ctx, gate_up_w, input);  // [n_ff*2]
        ggml_tensor * activated = ggml_swiglu(ctx, gate_up);           // [n_ff]

        ggml_tensor * down_all = info.orig_exps[OP_DOWN];
        size_t sb_d = info.expert_slice_bytes[OP_DOWN];
        ggml_tensor * down_w = ggml_view_2d(ctx, down_all,
            down_all->ne[0], down_all->ne[1],
            down_all->nb[1], expert_id * sb_d);

        result = ggml_mul_mat(ctx, down_w, activated);  // [n_embd]
    } else if (info.orig_exps[OP_GATE] && info.orig_exps[OP_UP]) {
        // Separate gate + up path
        ggml_tensor * gate_all = info.orig_exps[OP_GATE];
        size_t sb_g = info.expert_slice_bytes[OP_GATE];
        ggml_tensor * gate_w = ggml_view_2d(ctx, gate_all,
            gate_all->ne[0], gate_all->ne[1], gate_all->nb[1], expert_id * sb_g);

        ggml_tensor * up_all = info.orig_exps[OP_UP];
        size_t sb_u = info.expert_slice_bytes[OP_UP];
        ggml_tensor * up_w = ggml_view_2d(ctx, up_all,
            up_all->ne[0], up_all->ne[1], up_all->nb[1], expert_id * sb_u);

        ggml_tensor * gate_out = ggml_mul_mat(ctx, gate_w, input);
        ggml_tensor * up_out = ggml_mul_mat(ctx, up_w, input);
        ggml_tensor * activated = ggml_swiglu_split(ctx, gate_out, up_out);

        ggml_tensor * down_all = info.orig_exps[OP_DOWN];
        size_t sb_d = info.expert_slice_bytes[OP_DOWN];
        ggml_tensor * down_w = ggml_view_2d(ctx, down_all,
            down_all->ne[0], down_all->ne[1], down_all->nb[1], expert_id * sb_d);

        result = ggml_mul_mat(ctx, down_w, activated);
    }

    if (result) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        // Use half of hardware threads for expert FFN (P-cores on hybrid CPUs)
        static int expert_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
        ggml_graph_compute_with_ctx(ctx, gf, expert_threads);
        memcpy(output, result->data, n_embd * sizeof(float));
    }

    ggml_free(ctx);
}

// Predictive callback: reads the gating logits (unfused node), does top-k on the CPU and loads
// cache misses. Compatible with async compute (the logits are visible in the CUDA compute loop).

// Helper: check if an expert should be protected from eviction
// (hot globally, hot per-layer, or used in recent tokens)
static bool expert_is_protected(const alpha_moe_optimizer * opt, int il, int eid) {
    // Protect domain experts pinned by pin_domain_experts().
    // Those experts get last_used = cache_clock + 1000000 at pin time.
    // During reactive prefill LRU must not evict them: they provide the stable domain coverage
    // that generation needs. The caller already has the slot and checks last_used for LRU, so
    // this only checks whether the expert was pinned (high timestamp).
    if (il < 0 || il >= opt->n_layer) return false;
    if (eid < 0 || eid >= opt->n_expert) return false;
    auto & cache = opt->caches[il];
    int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                 (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                 (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));
    int s = cache.slot_map[ref_op][eid];
    if (s < 0 || s >= opt->K) return false;
    // Protected if timestamp is far ahead of current clock (pinned experts)
    bool prot = cache.slots[ref_op][s].last_used > opt->cache_clock + 100000;
    static int dbg_count = 0;
    if (dbg_count < 5 && prot) {
        fprintf(stderr, "[PROTECT] layer=%d eid=%d slot=%d last_used=%lld clock=%lld → PROTECTED\n",
                il, eid, s, (long long)cache.slots[ref_op][s].last_used, (long long)opt->cache_clock);
        dbg_count++;
    }
    if (dbg_count < 5 && !prot && s > 0 && s < 5) {
        fprintf(stderr, "[PROTECT] layer=%d eid=%d slot=%d last_used=%lld clock=%lld → not protected\n",
                il, eid, s, (long long)cache.slots[ref_op][s].last_used, (long long)opt->cache_clock);
        dbg_count++;
    }
    return prot;
}

static void cpu_topk(const float * scores, int n_scores, int * out_ids, int k) {
    // Simple selection sort for small k (6-8 experts)
    std::vector<bool> used(n_scores, false);
    for (int i = 0; i < k; i++) {
        int best = -1;
        float best_score = -1e30f;
        for (int j = 0; j < n_scores; j++) {
            if (!used[j] && scores[j] > best_score) {
                best_score = scores[j];
                best = j;
            }
        }
        out_ids[i] = (best >= 0) ? best : 0;
        if (best >= 0) used[best] = true;
    }
}

// Helper: use pinned staging if available, fall back to direct DMA
static inline void staged_dma(alpha_moe_optimizer * opt,
                               ggml_tensor * dst, const void * src,
                               size_t offset, size_t size) {
    if (opt->staging_enabled && opt->staging_write_idx < opt->staging_buf_count) {
        void * pbuf = opt->staging_bufs[opt->staging_write_idx++];
        _acdas(dst, src, offset, size, pbuf);
    } else {
        _acda(dst, src, offset, size);
    }
}

static bool predictive_cache_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * opt = (alpha_moe_optimizer *)user_data;
    if (!opt || !opt->cache_enabled) {
        if (ask) return false;  // don't need any tensor data when cache disabled
        return true;            // continue execution
    }

    // Speculative mode: skip all callbacks
    if (opt->speculative_pass) {
        if (ask) return false;
        return true;
    }

    // Triggers on "ffn_moe_logits-N", the gating MUL_MAT (unfused, always visible)
    if (strncmp(t->name, "ffn_moe_logits-", 15) != 0) {
        if (ask) return false;
        return true;
    }

    if (ask) {
        return true;  // need this tensor's data (gating logits)
    }


    // ask=false: gating logits are available. Do CPU topk to find actual expert IDs.
    int layer_idx = atoi(t->name + 15);  // parse layer from "ffn_moe_logits-N"

    // Find the corresponding tracked MoE layer
    int moe_layer_idx = -1;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (il == layer_idx) {
            moe_layer_idx = il;
            break;
        }
    }
    if (moe_layer_idx < 0) return true;

    opt->staging_write_idx = 0;  // reset for this callback

    auto & info = opt->layer_info[moe_layer_idx];
    auto & cache = opt->caches[moe_layer_idx];
    ggml_tensor * ids_tensor = opt->cache_expert_ids[moe_layer_idx];
    if (!ids_tensor) return true;

    int n_experts = opt->n_expert;
    int n_used = opt->n_expert_used;
    int n_tokens = (int)t->ne[1];  // batch size from tensor shape (self-spec verify uses N+1)

    int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                 (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                 (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));

    // Read gating logits from GPU (n_experts floats per token)
    std::vector<float> logits(n_experts * n_tokens);
    ggml_backend_tensor_get(t, logits.data(), 0, n_experts * n_tokens * sizeof(float));

    // CPU topk to find actual expert IDs
    std::vector<int32_t> all_remapped(n_used * n_tokens);

    // Gate weight distribution tracking
    static thread_local double gate_weight_sum[16] = {};  // sum of softmax weights per position
    static thread_local int64_t gate_samples = 0;
    static thread_local int64_t gate_last_log = 0;

    for (int tok = 0; tok < n_tokens; tok++) {
        int actual_ids[16] = {};
        cpu_topk(logits.data() + tok * n_experts, n_experts, actual_ids, n_used);

        // Compute softmax weights for top-k experts
        {
            const float * lg = logits.data() + tok * n_experts;
            // Find max for numerical stability
            float maxl = lg[0];
            for (int j = 1; j < n_experts; j++) if (lg[j] > maxl) maxl = lg[j];
            // Compute exp and sum
            float sum_exp = 0;
            float expert_exp[16] = {};
            for (int e = 0; e < n_used; e++) {
                expert_exp[e] = expf(lg[actual_ids[e]] - maxl);
                sum_exp += expert_exp[e];
            }
            // Also add non-selected experts to sum for proper softmax
            for (int j = 0; j < n_experts; j++) sum_exp += expf(lg[j] - maxl);
            // Subtract double-counted selected experts
            for (int e = 0; e < n_used; e++) sum_exp -= expert_exp[e];
            // Now sum_exp = sum of all exp(logits)
            float total_exp = sum_exp;
            for (int e = 0; e < n_used; e++) total_exp += expert_exp[e];

            // Accumulate normalized weights per position
            for (int e = 0; e < n_used; e++) {
                gate_weight_sum[e] += expert_exp[e] / total_exp;
            }
            gate_samples++;

            // Log every 500 samples (~10 tokens x 48 layers)
            if (gate_samples % 500 == 0 && gate_samples > gate_last_log) {
                gate_last_log = gate_samples;
                ALPHA_LOG_DBG("[GATE-WEIGHTS] samples=%lld avg_weights:",
                        (long long)gate_samples);
                double cumulative = 0;
                for (int e = 0; e < n_used; e++) {
                    double avg = gate_weight_sum[e] / gate_samples;
                    cumulative += avg;
                    ALPHA_LOG_DBG( " p%d=%.3f", e, avg);
                }
                ALPHA_LOG_DBG( " sum=%.3f\n", cumulative);
            }
        }

        for (int e = 0; e < n_used; e++) {
            int eid = actual_ids[e];
            if (eid < 0 || eid >= n_experts) eid = 0;

            int slot = cache.slot_map[ref_op][eid];
            if (slot >= 0) {
                // Cache hit
                all_remapped[tok * n_used + e] = slot;
                opt->cache_hits++;
            } else {
                // Cache miss: load the expert via batched DMA (no per-copy sync)
                int lru_slot = 0;
                int64_t lru_time = INT64_MAX;
                for (int k = 0; k < opt->K; k++) {
                    // Skip protected experts (hot/temporal)
                    if (cache.slots[ref_op][k].expert_id >= 0 &&
                        expert_is_protected(opt, moe_layer_idx, cache.slots[ref_op][k].expert_id))
                        continue;
                    if (cache.slots[ref_op][k].last_used < lru_time) {
                        lru_time = cache.slots[ref_op][k].last_used;
                        lru_slot = k;
                    }
                }

                for (int op = 0; op < OP_COUNT; op++) {
                    if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                    int old_eid = cache.slots[op][lru_slot].expert_id;
                    if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;

                    size_t sb = info.expert_slice_bytes[op];
                    // Async DMA: queue on dedicated stream, no per-copy sync
                    staged_dma(opt, cache.gpu_tensor[op],
                        (const char *)info.orig_exps[op]->data + eid * sb,
                        lru_slot * sb, sb);

                    cache.slot_map[op][eid] = lru_slot;
                    cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
                }
                all_remapped[tok * n_used + e] = lru_slot;
                opt->cache_misses++;
            }

            // Update LRU
            for (int op = 0; op < OP_COUNT; op++) {
                int s = cache.slot_map[op][eid];
                if (s >= 0) cache.slots[op][s].last_used = ++opt->cache_clock;
            }
        }

        // Update prev_routing for next token prediction
        for (int e = 0; e < n_used && e < (int)opt->prev_routing[moe_layer_idx].size(); e++) {
            opt->prev_routing[moe_layer_idx][e] = actual_ids[e];
        }
    }

    // Validate remapped IDs: range check + duplicate check
    for (size_t i = 0; i < all_remapped.size(); i++) {
        if (all_remapped[i] < 0 || all_remapped[i] >= opt->K) {
            ALPHA_LOG_DBG("[BUG5] INVALID slot=%d at pos=%zu (K=%d, layer=%d)\n",
                all_remapped[i], i, opt->K, moe_layer_idx);
            all_remapped[i] = 0;
        }
    }
    // Dedup: assign unique slot IDs per token. Duplicates get unused slots.
    // Skip dedup when all entries point to slot 0 (zero-expert for split_group skipped layers).
    // The CUDA MUL_MAT_ID kernel handles duplicate IDs correctly (reads same row N times).
    // Deduping zero-expert entries wastes N-1 cache slots that could hold real experts.
    for (int tok = 0; tok < n_tokens; tok++) {
        // Check if all-zero (skipped layer) - skip dedup
        bool all_zero = true;
        for (int e = 0; e < n_used; e++) {
            if (all_remapped[tok * n_used + e] != 0) { all_zero = false; break; }
        }
        if (all_zero) continue;

        bool used[256] = {};
        for (int e = 0; e < n_used; e++) {
            int idx = tok * n_used + e;
            if (all_remapped[idx] >= 0 && all_remapped[idx] < opt->K && !used[all_remapped[idx]]) {
                used[all_remapped[idx]] = true;
            } else {
                int alt = 0;
                for (int s = 0; s < opt->K; s++) {
                    if (!used[s]) { alt = s; break; }
                }
                all_remapped[idx] = alt;
                used[alt] = true;
            }
        }
    }

    // Barrier: ensure DMA completes before compute uses the data
    if (opt->config.no_barrier) {
        _acds();  // blocking
    } else {
        _acdb();  // non-blocking: GPU waits via event
    }

    // Write remapped IDs to GPU tensor
    ggml_backend_tensor_set(ids_tensor, all_remapped.data(), 0,
        all_remapped.size() * sizeof(int32_t));

    return true;
}

// This eliminates prediction errors - the model gets exactly the experts it asks for.
static bool reactive_cache_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * opt = (alpha_moe_optimizer *)user_data;
    if (!opt) { if (ask) return false; return true; }

    // Debug: count callback invocations per request
    static int cb_fire_count = 0;
    if (!ask && strncmp(t->name, "ffn_moe_topk", 12) == 0) {
        cb_fire_count++;
    }

    // Direct mode (no cache): only intervene if expert overrides exist
    if (!opt->cache_enabled) {
        if (opt->override_experts.empty()) {
            if (ask) return false;
            return true;
        }
        if (strncmp(t->name, "ffn_moe_topk", 12) != 0) {
            if (ask) return false;
            return true;
        }

        if (ask) {
            // Identify which MoE layer this is, only sync if it has an override
            int target = opt->reactive_layer_counter;
            opt->reactive_layer_counter++;
            int layer_idx = -1, moe_count = 0;
            for (int il = 0; il < opt->n_layer; il++) {
                if (!opt->layer_info[il].has_experts) continue;
                if (moe_count == target) { layer_idx = il; break; }
                moe_count++;
            }
            if (layer_idx >= 0 && layer_idx < (int)opt->override_experts.size()
                && !opt->override_experts[layer_idx].empty()) {
                return true;  // sync this layer - has override
            }
            return false;  // skip - no override for this layer
        }

        // ask=false: data available - find layer and apply override
        int target = opt->reactive_layer_counter - 1;
        int layer_idx = -1, moe_count = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            if (!opt->layer_info[il].has_experts) continue;
            if (moe_count == target) { layer_idx = il; break; }
            moe_count++;
        }
        if (layer_idx < 0 || layer_idx >= (int)opt->override_experts.size()
            || opt->override_experts[layer_idx].empty()) {
            return true;
        }

        // Apply override: read argsort, replace expert IDs, write back
        int n_used = std::min((int)t->ne[0], opt->n_expert_used);
        int n_tokens = (int)t->ne[1];
        auto & ovr = opt->override_experts[layer_idx];

        // Read current argsort from tensor (may need full row, not just n_used)
        size_t row_bytes = t->nb[1];  // bytes per token row
        std::vector<int32_t> ids(n_used * n_tokens);
        for (int tok = 0; tok < n_tokens; tok++) {
            ggml_backend_tensor_get(t, &ids[tok * n_used],
                tok * row_bytes, n_used * sizeof(int32_t));
        }

        // Log original vs override (first token only, for debugging)
        static int override_log_count = 0;
        if (override_log_count < 5) {
            ALPHA_LOG_DBG("[OVERRIDE] layer %d: orig=[", layer_idx);
            for (int e = 0; e < n_used; e++) ALPHA_LOG_DBG( "%s%d", e?",":"", ids[e]);
            ALPHA_LOG_DBG( "] → ovr=[");
            for (int e = 0; e < (int)ovr.size() && e < n_used; e++) ALPHA_LOG_DBG( "%s%d", e?",":"", ovr[e]);
            ALPHA_LOG_DBG( "] tensor=%s ne=[%lld,%lld] nb=[%lld,%lld]\n",
                t->name, (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->nb[0], (long long)t->nb[1]);
            override_log_count++;
        }

        for (int tok = 0; tok < n_tokens; tok++) {
            for (int e = 0; e < n_used && e < (int)ovr.size(); e++) {
                ids[tok * n_used + e] = ovr[e];
            }
        }
        // Write back using the same stride-aware offsets
        for (int tok = 0; tok < n_tokens; tok++) {
            ggml_backend_tensor_set(t, &ids[tok * n_used],
                tok * row_bytes, n_used * sizeof(int32_t));
        }
        return true;
    }

    // Speculative mode: skip all callbacks (prediction-only, no syncs)
    if (opt->speculative_pass) {
        if (ask) return false;
        return true;
    }

    // We only care about the argsort output (expert selection)
    // Named "ffn_moe_topk" with layer index encoded in name
    if (strncmp(t->name, "ffn_moe_topk", 12) != 0) {
        if (ask) return false;  // don't need this tensor, skip sync
        return true;
    }

    // LUT prefill callback

    if (ask) {
        // Grouped splits: sync every split_group MoE layers
        // Force split_group=1 during batched prefill (>1 writes zero-expert IDs that crash multi-token decode)
        int sg = opt->prompt_mode ? 1 : opt->split_group;
        int layer_in_group = opt->reactive_layer_counter % sg;
        bool is_sync_point = (layer_in_group == sg - 1) ||
                             (opt->reactive_layer_counter >= opt->n_moe_layers - 1);
        opt->reactive_layer_counter++;
        if (!is_sync_point) {
            // Skipped layer - prepare()'s prediction-based slot IDs stay in place.
            // The previous sync point already corrected them for the current group.
            return false;
        }
        return true;  // sync at group boundary
    }

    // ask=false: tensor data is available, read expert IDs and load cache
    int n_used = std::min((int)t->ne[0], opt->n_expert_used);
    // Use min of tensor dim and 1 for sequential prompt (graph may reserve more)
    int n_tokens = (int)t->ne[1];  // batch size from tensor (self-spec verify uses N+1)
    if (n_tokens <= 0 || n_used <= 0) return true;

    // Find which MoE layer this corresponds to
    // reactive_layer_counter was already incremented in ask=true, so use counter-1
    int target_count = opt->reactive_layer_counter - 1;
    int moe_layer_idx = -1;
    int moe_count = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        if (moe_count == target_count) {
            moe_layer_idx = il;
            break;
        }
        moe_count++;
    }

    if (moe_layer_idx < 0) return true;

    // Reset staging buffer index for this callback (all buffers safe to reuse after previous sync)
    opt->staging_write_idx = 0;

    // Per-layer routing entropy is skipped here (a 256-float GPU read per layer is too expensive)
    // and computed in record() after the forward pass instead.

    auto & info = opt->layer_info[moe_layer_idx];
    auto & cache = opt->caches[moe_layer_idx];
    ggml_tensor * ids_tensor = opt->cache_expert_ids[moe_layer_idx];
    if (!ids_tensor) return true;

    int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                 (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                 (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));

    // Process each token's expert routing
    std::vector<int32_t> all_remapped(n_used * n_tokens);
    int32_t saved_actual_ids[16] = {};  // save last token's actual IDs for reuse below

    for (int tok = 0; tok < n_tokens; tok++) {
        int32_t actual_ids[16] = {};
        size_t offset = tok * t->nb[1];
        ggml_backend_tensor_get(t, actual_ids, offset, n_used * sizeof(int32_t));

        // Expert override: replace argsort output with caller's choice
        if (moe_layer_idx < (int)opt->override_experts.size() &&
            !opt->override_experts[moe_layer_idx].empty()) {
            auto & ovr = opt->override_experts[moe_layer_idx];
            for (int e = 0; e < n_used && e < (int)ovr.size(); e++) {
                actual_ids[e] = ovr[e];
            }
        }

        // Save for reuse in tracking code below (avoids redundant GPU reads)
        memcpy(saved_actual_ids, actual_ids, n_used * sizeof(int32_t));

        // Next-cache measurement: check the cache before any DMA
        // Only run when log_confidence is enabled (controlled by --log-confidence flag)
        if (opt->config.enable_profiling) {
            static thread_local int64_t nc_layers_total = 0;
            static thread_local int64_t nc_layers_cached = 0;
            static thread_local int64_t nc_expert_total = 0;
            static thread_local int64_t nc_expert_cached = 0;
            static thread_local int64_t nc_tokens_total = 0;
            static thread_local int64_t nc_tokens_all_cached = 0;
            static thread_local bool nc_token_ok = true;
            static thread_local int64_t nc_last_log = 0;

            // Reset at first MoE layer of each token
            if (opt->reactive_layer_counter <= 1) {
                if (nc_layers_total > 0) {
                    nc_tokens_total++;
                    if (nc_token_ok) nc_tokens_all_cached++;
                }
                nc_token_ok = true;
            }

            bool layer_all_cached = true;
            for (int e = 0; e < n_used; e++) {
                int eid = actual_ids[e];
                if (eid < 0 || eid >= opt->n_expert) continue;
                nc_expert_total++;
                if (cache.slot_map[ref_op][eid] >= 0) {
                    nc_expert_cached++;
                } else {
                    layer_all_cached = false;
                }
            }
            nc_layers_total++;
            if (layer_all_cached) nc_layers_cached++;
            if (!layer_all_cached) nc_token_ok = false;

            // Log every 10 tokens (at first MoE layer to avoid spam)
            if (nc_tokens_total > 0 && nc_tokens_total % 10 == 0 &&
                opt->reactive_layer_counter <= 1 && nc_tokens_total > nc_last_log) {
                nc_last_log = nc_tokens_total;
                ALPHA_LOG_DBG("[NEXT-CACHE] tokens=%lld expert=%.1f%% layer=%.1f%% "
                        "token_all_cached=%lld/%lld (%.1f%%)\n",
                        (long long)nc_tokens_total,
                        nc_expert_total > 0 ? 100.0 * nc_expert_cached / nc_expert_total : 0,
                        nc_layers_total > 0 ? 100.0 * nc_layers_cached / nc_layers_total : 0,
                        (long long)nc_tokens_all_cached, (long long)nc_tokens_total,
                        nc_tokens_total > 0 ? 100.0 * nc_tokens_all_cached / nc_tokens_total : 0);
            }
        }

        // Scratch mode: DMA gate+up for all active experts into scratch slots 0..n_used-1
        // Skip in draft mode (no DMA, use whatever's cached)
        if (opt->use_scratch && !opt->spec_draft_active) {
            for (int e = 0; e < n_used; e++) {
                int eid = actual_ids[e];
                if (eid < 0 || eid >= opt->n_expert) eid = 0;
                if (info.orig_exps[OP_GATE] && opt->gate_scratch) {
                    size_t sb = info.expert_slice_bytes[OP_GATE];
                    staged_dma(opt, opt->gate_scratch,
                        (const char *)info.orig_exps[OP_GATE]->data + eid * sb, e * sb, sb);
                }
                if (info.orig_exps[OP_UP] && opt->up_scratch) {
                    size_t sb = info.expert_slice_bytes[OP_UP];
                    staged_dma(opt, opt->up_scratch,
                        (const char *)info.orig_exps[OP_UP]->data + eid * sb, e * sb, sb);
                }
            }
        }

        // Per-token miss tracking: count layers with any miss (for quality gate)
        bool layer_had_miss = false;

        for (int e = 0; e < n_used; e++) {
            int eid = actual_ids[e];
            if (eid < 0 || eid >= opt->n_expert) eid = 0;

            // In scratch mode, check OP_DOWN only (gate/up always in scratch)
            int slot = opt->use_scratch
                ? cache.slot_map[OP_DOWN][eid]
                : cache.slot_map[ref_op][eid];
            if (slot >= 0) {
                // Cache hit
                all_remapped[tok * n_used + e] = slot;
                opt->cache_hits++;
            } else if (opt->spec_draft_active) {
                // Draft mode: zero-expert for all misses (no DMA)
                all_remapped[tok * n_used + e] = 0;  // zero-expert slot
                opt->cache_hits++;  // count as synthetic hit
                layer_had_miss = true;
            } else {
                // Cache miss - async DMA on dedicated stream (batched, synced after all copies)
                int lru_slot = 0;
                int64_t lru_time = INT64_MAX;
                for (int k = 0; k < opt->K; k++) {
                    if (cache.slots[ref_op][k].expert_id >= 0 &&
                        expert_is_protected(opt, moe_layer_idx, cache.slots[ref_op][k].expert_id))
                        continue;
                    if (cache.slots[ref_op][k].last_used < lru_time) {
                        lru_time = cache.slots[ref_op][k].last_used;
                        lru_slot = k;
                    }
                }

                if (opt->tiered_cache) {
                    // Tiered: DMA down into per-layer cache
                    {
                        int old_eid = cache.slots[OP_DOWN][lru_slot].expert_id;
                        if (old_eid >= 0) cache.slot_map[OP_DOWN][old_eid] = -1;
                        size_t sb = info.expert_slice_bytes[OP_DOWN];
                        staged_dma(opt, cache.gpu_tensor[OP_DOWN],
                            (const char *)info.orig_exps[OP_DOWN]->data + eid * sb,
                            lru_slot * sb, sb);
                        cache.slot_map[OP_DOWN][eid] = lru_slot;
                        cache.slots[OP_DOWN][lru_slot] = { eid, ++opt->cache_clock };
                    }
                    // Gate/up: check global cache, DMA if needed
                    auto & gc = opt->global_gate_up_cache;
                    for (int gop : { OP_GATE, OP_UP }) {
                        if (!gc.gpu_tensor[gop] || !info.orig_exps[gop]) continue;
                        if (gc.slot_map[gop][eid] >= 0) {
                            gc.slots[gop][gc.slot_map[gop][eid]].last_used = ++opt->cache_clock;
                            continue;
                        }
                        int gc_lru = 0;
                        int64_t gc_time = INT64_MAX;
                        for (int k = 0; k < opt->K; k++) {
                            if (gc.slots[gop][k].last_used < gc_time) {
                                gc_time = gc.slots[gop][k].last_used;
                                gc_lru = k;
                            }
                        }
                        int old_gc = gc.slots[gop][gc_lru].expert_id;
                        if (old_gc >= 0) gc.slot_map[gop][old_gc] = -1;
                        size_t sb = info.expert_slice_bytes[gop];
                        staged_dma(opt, gc.gpu_tensor[gop],
                            (const char *)info.orig_exps[gop]->data + eid * sb,
                            gc_lru * sb, sb);
                        gc.slot_map[gop][eid] = gc_lru;
                        gc.slots[gop][gc_lru] = { eid, ++opt->cache_clock };
                    }
                } else if (opt->use_scratch) {
                    // Scratch mode: only DMA into per-layer down cache (gate/up handled by scratch DMA above)
                    int old_eid = cache.slots[OP_DOWN][lru_slot].expert_id;
                    if (old_eid >= 0) cache.slot_map[OP_DOWN][old_eid] = -1;
                    size_t sb = info.expert_slice_bytes[OP_DOWN];
                    staged_dma(opt, cache.gpu_tensor[OP_DOWN],
                        (const char *)info.orig_exps[OP_DOWN]->data + eid * sb,
                        lru_slot * sb, sb);
                    cache.slot_map[OP_DOWN][eid] = lru_slot;
                    cache.slots[OP_DOWN][lru_slot] = { eid, ++opt->cache_clock };
                } else {
                    for (int op = 0; op < OP_COUNT; op++) {
                        if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                        int old_eid = cache.slots[op][lru_slot].expert_id;
                        if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                        size_t sb = info.expert_slice_bytes[op];
                        staged_dma(opt, cache.gpu_tensor[op],
                            (const char *)info.orig_exps[op]->data + eid * sb,
                            lru_slot * sb, sb);
                        cache.slot_map[op][eid] = lru_slot;
                        cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
                    }
                }
                all_remapped[tok * n_used + e] = lru_slot;
                opt->cache_misses++;
            }

            // Update LRU
            if (opt->use_scratch || opt->tiered_cache) {
                int s = cache.slot_map[OP_DOWN][eid];
                if (s >= 0) cache.slots[OP_DOWN][s].last_used = ++opt->cache_clock;
            } else {
                for (int op = 0; op < OP_COUNT; op++) {
                    int s = cache.slot_map[op][eid];
                    if (s >= 0 && !cache.slots[op].empty())
                        cache.slots[op][s].last_used = ++opt->cache_clock;
                }
            }
        }

        // Track per-token miss layers (for quality gate in LUT-direct mode)
        if (tok == 0) {  // only track first token (LUT-direct = 1 token)
            opt->last_token_total_layers++;
            if (layer_had_miss) opt->last_token_miss_layers++;
        }
    }

    // Dedup: assign unique slot IDs per token (for down / flat cache)
    // Skip dedup when all entries are slot 0 (zero-expert for split_group skipped layers).
    for (int tok = 0; tok < n_tokens; tok++) {
        bool all_zero = true;
        for (int e = 0; e < n_used; e++) {
            if (all_remapped[tok * n_used + e] != 0) { all_zero = false; break; }
        }
        if (all_zero) continue;

        bool used[256] = {};
        for (int e = 0; e < n_used; e++) {
            int idx = tok * n_used + e;
            if (all_remapped[idx] >= 0 && all_remapped[idx] < opt->K && !used[all_remapped[idx]]) {
                used[all_remapped[idx]] = true;
            } else {
                int alt = 0;
                for (int s = 0; s < opt->K; s++) {
                    if (!used[s]) { alt = s; break; }
                }
                all_remapped[idx] = alt;
                used[alt] = true;
            }
        }
    }

    // Barrier: ensure DMA completes before compute uses the data.
    if (opt->config.no_barrier) {
        _acds();  // blocking: host waits for DMA
    } else {
        _acdb();  // non-blocking: GPU waits via cudaStreamWaitEvent
    }

    // Write remapped IDs to GPU
    if (opt->use_scratch) {
        // Scratch mode: gate/up IDs = sequential (0,1,2,...n_used-1)
        //               down IDs = from all_remapped (cache slots)
        std::vector<int32_t> scratch_ids(n_used * n_tokens);
        for (int tok = 0; tok < n_tokens; tok++) {
            for (int e = 0; e < n_used; e++) {
                scratch_ids[tok * n_used + e] = e;  // scratch slot = expert position
            }
        }
        // gate/up use scratch IDs (sequential 0..5)
        ggml_backend_tensor_set(ids_tensor, scratch_ids.data(), 0,
            scratch_ids.size() * sizeof(int32_t));
        // down uses cache IDs (from all_remapped - LRU slots)
        ggml_tensor * ids_down = (moe_layer_idx < (int)opt->cache_expert_ids_down.size())
            ? opt->cache_expert_ids_down[moe_layer_idx] : nullptr;
        if (ids_down) {
            ggml_backend_tensor_set(ids_down, all_remapped.data(), 0,
                all_remapped.size() * sizeof(int32_t));
        }
    } else if (opt->tiered_cache) {
        // Write down IDs (per-layer slots)
        ggml_tensor * ids_down = (moe_layer_idx < (int)opt->cache_expert_ids_down.size())
            ? opt->cache_expert_ids_down[moe_layer_idx] : nullptr;
        if (ids_down) {
            ggml_backend_tensor_set(ids_down, all_remapped.data(), 0,
                all_remapped.size() * sizeof(int32_t));
        }
        // Write gate/up IDs (global slots)
        auto & gc = opt->global_gate_up_cache;
        int gc_ref = gc.gpu_tensor[OP_GATE] ? OP_GATE : OP_UP;
        std::vector<int32_t> gu_remapped(all_remapped.size());
        for (int tok = 0; tok < n_tokens; tok++) {
            for (int e = 0; e < n_used; e++) {
                int eid = saved_actual_ids[e];
                int gs = (eid >= 0 && eid < opt->n_expert) ? gc.slot_map[gc_ref][eid] : -1;
                gu_remapped[tok * n_used + e] = (gs >= 0) ? gs : 0;
            }
            // Dedup gate/up IDs (skip for all-zero / skipped layers)
            bool gu_all_zero = true;
            for (int e = 0; e < n_used; e++) {
                if (gu_remapped[tok * n_used + e] != 0) { gu_all_zero = false; break; }
            }
            if (!gu_all_zero) {
                bool gu_used[256] = {};
                for (int e = 0; e < n_used; e++) {
                    int idx = tok * n_used + e;
                    if (gu_remapped[idx] >= 0 && gu_remapped[idx] < opt->K && !gu_used[gu_remapped[idx]]) {
                        gu_used[gu_remapped[idx]] = true;
                    } else {
                        int alt = 0;
                        for (int s = 0; s < opt->K; s++) {
                            if (!gu_used[s]) { alt = s; break; }
                        }
                        gu_remapped[idx] = alt;
                        gu_used[alt] = true;
                    }
                }
            }
        }
        ggml_backend_tensor_set(ids_tensor, gu_remapped.data(), 0,
            gu_remapped.size() * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(ids_tensor, all_remapped.data(), 0,
            all_remapped.size() * sizeof(int32_t));
    }

    // Update prev_routing + prediction reliability (always needed)
    // Detailed prediction-accuracy measurement is gated behind enable_profiling
    {
        int32_t * last_ids = saved_actual_ids;

        // Lightweight set_match check for prediction reliability (always on)
        auto & predicted = opt->prev_routing[moe_layer_idx];
        bool set_match = true;
        for (int e = 0; e < n_used; e++) {
            int eid = last_ids[e];
            bool found = false;
            for (int p = 0; p < n_used && p < (int)predicted.size(); p++) {
                if (eid == predicted[p]) { found = true; break; }
            }
            if (!found) { set_match = false; break; }
        }

        opt->pred_set_total++;
        if (set_match) opt->pred_set_matches++;
        if (opt->pred_set_total == 50) {
            float pred_accuracy = (float)opt->pred_set_matches / opt->pred_set_total;
            bool was_reliable = opt->prediction_reliable;
            opt->prediction_reliable = (pred_accuracy > 0.3f);
            if (was_reliable != opt->prediction_reliable) {
                ALPHA_LOG_DBG("[PRED] prediction %s (%.0f%% set_match after %lld layers)\n",
                    opt->prediction_reliable ? "RELIABLE" : "UNRELIABLE (prev_routing ~0%% set_match)",
                    100.0f * pred_accuracy, (long long)opt->pred_set_total);
            }
        }

        // Detailed prediction-accuracy logging (expensive nested loop, only when profiling)
        if (opt->config.enable_profiling) {
            static thread_local int64_t spec_layers_total = 0;
            static thread_local int64_t spec_layers_set_match = 0;
            static thread_local int64_t spec_layers_all_cached = 0;
            static thread_local int64_t spec_tokens = 0;
            static thread_local int64_t spec_tokens_all_set = 0;
            static thread_local int64_t spec_tokens_all_cached = 0;
            static thread_local int spec_layer_count = 0;
            static thread_local bool spec_token_all_set = true;
            static thread_local bool spec_token_all_cached = true;

            if (opt->reactive_layer_counter <= 1) {
                if (spec_layer_count > 0) {
                    spec_tokens++;
                    if (spec_token_all_set) spec_tokens_all_set++;
                    if (spec_token_all_cached) spec_tokens_all_cached++;
                    if (spec_tokens % 5 == 0 && spec_tokens > 0) {
                        ALPHA_LOG_DBG("[PRED-ACC] tokens=%lld set_match=%.1f%% all_cached=%.1f%% "
                                "token_all_set=%lld/%lld token_all_cached=%lld/%lld\n",
                                (long long)spec_tokens,
                                spec_layers_total > 0 ? 100.0 * spec_layers_set_match / spec_layers_total : 0,
                                spec_layers_total > 0 ? 100.0 * spec_layers_all_cached / spec_layers_total : 0,
                                (long long)spec_tokens_all_set, (long long)spec_tokens,
                                (long long)spec_tokens_all_cached, (long long)spec_tokens);
                    }
                }
                spec_layer_count = 0;
                spec_token_all_set = true;
                spec_token_all_cached = true;
            }
            spec_layer_count++;

            auto & cache2 = opt->caches[moe_layer_idx];
            int ref_op2 = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                         (cache2.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                         (cache2.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));
            bool all_cached = true;
            for (int e = 0; e < n_used; e++) {
                int eid = last_ids[e];
                if (eid >= 0 && eid < opt->n_expert && cache2.slot_map[ref_op2][eid] < 0) {
                    all_cached = false;
                }
            }
            spec_layers_total++;
            if (set_match) spec_layers_set_match++;
            if (all_cached) spec_layers_all_cached++;
            if (!set_match) spec_token_all_set = false;
            if (!all_cached) spec_token_all_cached = false;
        }

        // Update prev_routing for this layer and the skipped layers in the group
        for (int e = 0; e < n_used && e < (int)opt->prev_routing[moe_layer_idx].size(); e++) {
            opt->prev_routing[moe_layer_idx][e] = last_ids[e];
        }
        // For sg>1, propagate routing to skipped layers in the same group
        int sg = opt->split_group;
        if (sg > 1) {
            int target_count = opt->reactive_layer_counter - 1;
            int group_start = (target_count / sg) * sg;
            int mc2 = 0;
            for (int il2 = 0; il2 < opt->n_layer; il2++) {
                if (!opt->layer_info[il2].has_experts || opt->layer_info[il2].on_gpu) continue;
                if (mc2 >= group_start && mc2 < group_start + sg && il2 != moe_layer_idx) {
                    for (int e = 0; e < n_used && e < (int)opt->prev_routing[il2].size(); e++) {
                        opt->prev_routing[il2][e] = last_ids[e];
                    }
                }
                mc2++;
            }
        }
    }

    // Expert usage: count unique experts per token + overlap between tokens
    if (opt->config.enable_profiling) {
        static thread_local std::vector<bool> expert_seen;
        static thread_local std::vector<bool> prev_experts;
        static thread_local int unique_count = 0;
        static thread_local int64_t eu_tokens = 0;
        static thread_local int64_t eu_total_unique = 0;
        static thread_local int64_t eu_total_overlap = 0;
        static thread_local int eu_layers = 0;
        static thread_local int64_t eu_last_log = 0;
        const int ne = opt->n_expert;
        if ((int)expert_seen.size() != ne) { expert_seen.assign(ne, false); prev_experts.assign(ne, false); }

        if (opt->reactive_layer_counter <= 1) {
            if (eu_layers > 0) {
                int overlap = 0;
                for (int i = 0; i < ne; i++)
                    if (expert_seen[i] && prev_experts[i]) overlap++;
                eu_tokens++;
                eu_total_unique += unique_count;
                eu_total_overlap += overlap;
                prev_experts = expert_seen;
                if (eu_tokens % 10 == 0 && eu_tokens > eu_last_log) {
                    eu_last_log = eu_tokens;
                    ALPHA_LOG_DBG("[EXPERT-USE] tokens=%lld unique=%.0f/%d overlap_prev=%.0f\n",
                            (long long)eu_tokens,
                            (double)eu_total_unique / eu_tokens, ne,
                            (double)eu_total_overlap / eu_tokens);
                }
            }
            std::fill(expert_seen.begin(), expert_seen.end(), false);
            unique_count = 0;
            eu_layers = 0;
        }
        eu_layers++;
        {
            // Reuse saved_actual_ids - no extra GPU read needed
            for (int e = 0; e < n_used; e++) {
                int eid = saved_actual_ids[e];
                if (eid >= 0 && eid < ne && !expert_seen[eid]) {
                    expert_seen[eid] = true;
                    unique_count++;
                }
            }
        }
    }

    // Grouped splits: write slot IDs for the next group's skipped layers.
    // Cache hits -> correct slot. Cache misses -> slot 0 (zero expert, residual passthrough).
    // No DMA needed - just remapping. Wrong experts contaminate, zero experts don't.
    if (opt->split_group > 1 && !opt->moe_layer_indices.empty()) {
        int sg = opt->split_group;
        int current_moe_idx = -1;
        for (int mi = 0; mi < opt->n_moe_layers; mi++) {
            if (opt->moe_layer_indices[mi] == moe_layer_idx) {
                current_moe_idx = mi;
                break;
            }
        }
        if (current_moe_idx >= 0) {
            int next_start = current_moe_idx + 1;
            int next_end = std::min(next_start + sg - 1, opt->n_moe_layers - 1);
            for (int mi = next_start; mi <= next_end; mi++) {
                int skip_il = opt->moe_layer_indices[mi];
                auto & skip_cache = opt->caches[skip_il];
                ggml_tensor * skip_ids = opt->cache_expert_ids[skip_il];
                if (!skip_ids) continue;

                int skip_ref = skip_cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                               (skip_cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN);

                int32_t corrected[16] = {};
                for (int e = 0; e < opt->n_expert_used && e < 16; e++) {
                    int eid = (skip_il < (int)opt->prev_routing.size() &&
                               e < (int)opt->prev_routing[skip_il].size())
                              ? opt->prev_routing[skip_il][e] : e;
                    if (eid < 0 || eid >= opt->n_expert) eid = 0;

                    int slot = skip_cache.slot_map[skip_ref][eid];
                    corrected[e] = (slot >= 0) ? slot : 0;  // miss -> zero expert (slot 0)
                }
                ggml_backend_tensor_set(skip_ids, corrected, 0,
                    opt->n_expert_used * sizeof(int32_t));
            }
        }
    }

    return true;  // continue execution
}

void alpha_moe_optimizer_set_speculative(alpha_moe_optimizer * opt, bool enable) {
    if (opt) opt->speculative_pass = enable;
}

extern thread_local bool g_alpha_draft_mode;
extern thread_local int  g_alpha_draft_expert_count;

void alpha_moe_optimizer_set_draft_mode(alpha_moe_optimizer * opt, bool enable) {
    if (!opt) return;
    opt->draft_mode = enable;
    g_alpha_draft_mode = enable;

    if (enable && opt->draft_expert_count > 0) {
        // Mode A: expert reduce - speculative pass (no callbacks), reduced experts
        opt->speculative_pass = true;
        g_alpha_draft_expert_count = opt->draft_expert_count;
    } else if (enable) {
        // Mode C: backbone only - speculative pass, zero routed experts
        opt->speculative_pass = true;
        g_alpha_draft_expert_count = 0;
    } else {
        // Full model
        opt->speculative_pass = false;
        g_alpha_draft_expert_count = 0;
    }
}

void alpha_moe_optimizer_set_draft_expert_count(alpha_moe_optimizer * opt, int count) {
    if (opt) opt->draft_expert_count = count;
}

// Cached-Expert Self-Speculative Decoding (GPU LUT)
//
// GPU-side lookup table eliminates callback overhead entirely.
// Per-layer LUT[n_expert] I32: maps expert_id -> cache_slot (or 0 for zero expert).
// In the graph: ggml_get_rows(lut, argsort_output) remaps on GPU, no CPU sync.

void alpha_moe_optimizer_init_spec_lut(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || !opt->cache_enabled || opt->spec_lut_ready) return;

    // Find GPU buffer type
    ggml_backend_buffer_type_t gpu_buft = nullptr;
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        for (int op = 0; op < OP_COUNT; op++) {
            if (opt->caches[il].gpu_tensor[op]) {
                gpu_buft = ggml_backend_buffer_get_type(opt->caches[il].gpu_tensor[op]->buffer);
                break;
            }
        }
        if (gpu_buft) break;
    }
    if (!gpu_buft) return;

    // Count MoE layers
    int n_moe = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) n_moe++;
    }

    // Allocate LUT tensors: [1, n_expert] I32 per MoE layer
    size_t ctx_size = (n_moe + 4) * ggml_tensor_overhead() + 4096;
    struct ggml_init_params ctx_params = { ctx_size, nullptr, true };
    opt->spec_lut_ctx = ggml_init(ctx_params);
    if (!opt->spec_lut_ctx) return;

    opt->spec_lut.resize(opt->n_layer, nullptr);
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        opt->spec_lut[il] = ggml_new_tensor_2d(opt->spec_lut_ctx, GGML_TYPE_I32,
            1, opt->n_expert);  // [1, 256] - each "row" is 1 slot ID
        ggml_set_name(opt->spec_lut[il], "spec_lut");
        ggml_set_input(opt->spec_lut[il]);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(opt->spec_lut_ctx, gpu_buft);
    if (!buf) {
        ggml_free(opt->spec_lut_ctx);
        opt->spec_lut_ctx = nullptr;
        return;
    }
    opt->cache_bufs.emplace_back(buf);

    // Fill LUT from current slot_map
    alpha_moe_optimizer_update_lut(opt);
    opt->spec_lut_ready = true;

    ALPHA_LOG_DBG("[spec-lut] GPU LUT allocated: %d MoE layers × %d experts = %zu bytes\n",
            n_moe, opt->n_expert, (size_t)n_moe * opt->n_expert * sizeof(int32_t));
}

void alpha_moe_optimizer_update_lut(alpha_moe_optimizer * opt) {
    if (!opt || !opt->spec_lut_ready && !opt->spec_lut_ctx) return;

    std::vector<int32_t> lut_data(opt->n_expert);
    for (int il = 0; il < opt->n_layer; il++) {
        if (!opt->spec_lut[il]) continue;
        auto & cache = opt->caches[il];

        // Determine which op to check for slot mapping
        int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                     (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                     (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));

        for (int e = 0; e < opt->n_expert; e++) {
            int slot = cache.slot_map[ref_op][e];
            lut_data[e] = (slot >= 0) ? slot : 0;  // 0 = zero expert
        }

        ggml_backend_tensor_set(opt->spec_lut[il], lut_data.data(), 0,
            opt->n_expert * sizeof(int32_t));
    }
}

int alpha_moe_optimizer_check_prefill_quality(alpha_moe_optimizer * opt, int n_tokens) {
    if (!opt || !opt->is_moe || n_tokens <= 0) return -1;

    // LUT-based quality check: read LUT values to compute cache coverage.
    // The LUT maps expert_id -> slot_id. Slot 0 = zero expert (cache miss).
    // Count what fraction of experts map to slot 0 across all MoE layers.
    int total_zero = 0;
    int total_experts = 0;

    for (int il = 0; il < opt->n_layer; il++) {
        if (!opt->spec_lut[il]) continue;
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;

        int32_t lut_vals[LLAMA_MAX_EXPERTS] = {};
        int lut_size = std::min((int)opt->spec_lut[il]->ne[1], (int)LLAMA_MAX_EXPERTS);
        lut_size = std::min(lut_size, opt->n_expert);
        ggml_backend_tensor_get(opt->spec_lut[il], lut_vals, 0, lut_size * sizeof(int32_t));

        for (int e = 0; e < lut_size; e++) {
            total_experts++;
            if (lut_vals[e] == 0) total_zero++;
        }
    }

    if (total_experts == 0) return -1;

    float coverage = 1.0f - (float)total_zero / total_experts;
    // LUT prefill needs roughly 25% coverage for reliable output; below the threshold the cpu-moe
    // repair path is used instead. A threshold based on n_expert_used alone is too low:
    float threshold = (float)(opt->n_expert_used / 2) / opt->n_expert;
    // For 8-active/256: threshold = 4/256 = 1.6% (too low)
    // so use a 25% minimum instead:
    threshold = std::max(threshold, 0.25f);

    if (coverage < threshold) {
        fprintf(stderr, "[PREFILL-QC] FAIL: coverage=%.1f%% (%d/%d zero) < %.0f%% threshold → REPAIR\n",
                coverage * 100.0f, total_zero, total_experts, threshold * 100.0f);
        return 0;
    }

    fprintf(stderr, "[PREFILL-QC] OK: coverage=%.1f%% (%d/%d non-zero)\n",
            coverage * 100.0f, total_experts - total_zero, total_experts);
    return -1;
}

int alpha_moe_optimizer_reload_from_argsort(alpha_moe_optimizer * opt) {
    if (!opt || !opt->cache_enabled) return 0;

    // Read argsort outputs from the last forward pass and DMA missing experts into cache.
    // Capped at MAX_DMA_PER_LAYER: loading every miss costs more than it saves.
    // Argsort is ordered by routing weight - first entries are most important.
    static constexpr int MAX_DMA_PER_LAYER = 2;  // only top-2 misses per layer
    int n_argsort = alpha_mole_get_argsort_count();
    int moe_idx = 0;
    int total_loaded = 0;

    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts) { continue; }
        if (info.on_gpu) { moe_idx++; continue; }

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx);
        moe_idx++;
        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        int n_tokens = (int)argsort_out->ne[1];
        if (n_tokens <= 0 || n_used <= 0) continue;

        // Read last token's expert IDs
        int32_t actual_ids[16] = {};
        size_t offset = (n_tokens - 1) * argsort_out->nb[1];
        ggml_backend_tensor_get(argsort_out, actual_ids, offset, n_used * sizeof(int32_t));

        auto & cache = opt->caches[il];
        int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                     (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                     (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));

        int layer_loaded = 0;
        for (int e = 0; e < n_used && layer_loaded < MAX_DMA_PER_LAYER; e++) {
            int eid = actual_ids[e];
            if (eid < 0 || eid >= opt->n_expert) continue;
            if (cache.slot_map[ref_op][eid] >= 0) continue;  // already cached

            // Find LRU slot
            int lru_slot = 0;
            int64_t lru_time = INT64_MAX;
            for (int k = 1; k < opt->K; k++) {  // skip slot 0 (zero expert)
                if (cache.slots[ref_op][k].last_used < lru_time) {
                    lru_time = cache.slots[ref_op][k].last_used;
                    lru_slot = k;
                }
            }

            // DMA all ops for this expert
            for (int op = 0; op < OP_COUNT; op++) {
                if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                bool has_fused = (info.orig_exps[OP_GATE_UP] != nullptr);
                if (has_fused && (op == OP_GATE || op == OP_UP)) continue;

                int old_eid = cache.slots[op][lru_slot].expert_id;
                if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;

                size_t sb = info.expert_slice_bytes[op];
                ggml_backend_tensor_set(cache.gpu_tensor[op],
                    (const char *)info.orig_exps[op]->data + (size_t)eid * sb,
                    (size_t)lru_slot * sb, sb);

                cache.slot_map[op][eid] = lru_slot;
                cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
            }
            layer_loaded++;
            total_loaded++;
        }

        // Update prev_routing
        for (int e = 0; e < n_used && e < (int)opt->prev_routing[il].size(); e++) {
            if (actual_ids[e] >= 0 && actual_ids[e] < opt->n_expert) {
                opt->prev_routing[il][e] = actual_ids[e];
            }
        }
    }

    // Update LUT with new cache state
    if (opt->spec_lut_ready && total_loaded > 0) {
        alpha_moe_optimizer_update_lut(opt);
    }
    return total_loaded;
}

void alpha_moe_optimizer_apply_warmup(
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    const alpha_moe_profile * plan,
    const char * model_dir)
{
    if (!opt || !plan) return;

    // Serve mode detection: if draft mode is already active from a previous request,
    // we're in serve mode (CLI only enters draft at the end of apply_warmup).
    bool is_serve = opt->spec_draft_active;

    // Evict stale expert pages (also called from serve handler before classify)
    if (is_serve) {
        alpha_moe_optimizer_evict_expert_pages(opt);
    }

    // 0. Reset callback state
    alpha_moe_optimizer_set_callback_active(opt, false);

    // 1. Load style profiles for the classify warmup (once only, skipped if already attempted)
    if (plan->warmup == ALPHA_WARMUP_CLASSIFY && model_dir && !opt->profiles_loaded) {
        alpha_moe_optimizer_load_style_profiles(opt, model_dir);
        opt->profiles_loaded = true;  // don't retry every request
    }

    // 2. Disable cache during prompt for CPU_MOE (CLI only)
    //    In serve mode the prompt is already decoded before apply_warmup is called, so swapping
    //    tensor pointers (ne[2]=K <-> ne[2]=n_expert) is pointless and dangerous: it can trigger
    //    sched_reserve with mismatched dimensions and fragments VRAM until a later request
    //    runs out of memory on tight systems.
    if (plan->disable_cache_for_prompt && !is_serve) {
        alpha_moe_optimizer_disable_cache(opt);
    }

    // 3. Activate callback for reactive expert loading during prompt (CLI only)
    if (plan->callback == ALPHA_CB_LUT_DISABLED && !plan->disable_cache_for_prompt && !is_serve) {
        alpha_moe_optimizer_set_callback_active(opt, true);
    }

    // 4. Warmup: update cache from prompt's routing data
    //    In serve mode: prompt is already decoded, argsort data is available.
    //    warmup_classify reads argsort -> scores profiles -> DMAs missing experts.
    //    warmup_smart reads argsort -> DMAs fired experts + cross-layer hot.
    //    Both are safe in serve mode (only DMA, no tensor pointer swaps).
    if (plan->warmup == ALPHA_WARMUP_CLASSIFY) {
        alpha_moe_optimizer_warmup_classify_and_fill(opt);
    } else if (plan->warmup == ALPHA_WARMUP_SMART) {
        alpha_moe_optimizer_warmup_1token_smart(opt);
    }
    // ALPHA_WARMUP_NONE: skip (all experts on GPU need no warmup)

    // 5. Re-enable cache and reinstall callback for generation
    //    In serve mode the cache is already enabled (disable_cache was skipped),
    //    but enable_cache is idempotent - it just ensures tensor pointers are correct.
    alpha_moe_optimizer_enable_cache(opt);

    // Eval callback: the speculation mechanism. Intercepts expert routing during
    // forward pass and redirects to cached experts. Essential for LUT speed.
    //
    // Disable it only when every expert is in VRAM (K == n_expert) or K is high, where the callback
    // is pure overhead from GPU syncs; otherwise keep it.
    {
        int K = alpha_moe_optimizer_get_K(opt);
        int K_min = opt->n_expert / 4;
        if (K >= opt->n_expert || K >= K_min) {
            // All experts on GPU or high K: the callback adds overhead without benefit.
            // At high K, LUT speculation garbles hybrid models.
            // cpu-moe dispatch (no callback) is both faster and correct.
            llama_set_eval_callback(opt->ctx, nullptr, nullptr);
        } else {
            // Low K (< K_min): callback needed for warmup-LUT speculation
            alpha_moe_optimizer_reinstall_callback(opt);
        }
    }

    // 6. Enter LUT-direct draft mode if plan says so
    if (plan->gen == ALPHA_GEN_LUT_DIRECT || plan->gen == ALPHA_GEN_CACHED_SPEC) {
        alpha_moe_optimizer_enter_draft_mode(opt);
    }
}

void alpha_moe_optimizer_enter_draft_mode(alpha_moe_optimizer * opt) {
    if (!opt || !opt->cache_enabled) return;

    // Update LUT from current cache state
    if (opt->spec_lut_ready) {
        alpha_moe_optimizer_update_lut(opt);
        // Wire LUT to model layers - graph will use ggml_get_rows(lut, argsort)
        int wired = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            if (!opt->spec_lut[il]) continue;
            opt->model->layers[il].ffn_cache_expert_ids = opt->spec_lut[il];
            opt->model->layers[il].ffn_cache_expert_ids_down = nullptr;
            wired++;
        }
        // Dump LUT values for first MoE layer
        for (int il = 0; il < opt->n_layer; il++) {
            auto & info = opt->layer_info[il];
            if (!info.has_experts || info.on_gpu) continue;
            if (!opt->spec_lut[il]) continue;
            int32_t lut_vals[LLAMA_MAX_EXPERTS] = {};
            int lut_size = std::min((int)opt->spec_lut[il]->ne[1], LLAMA_MAX_EXPERTS);
            ggml_backend_tensor_get(opt->spec_lut[il], lut_vals, 0, lut_size * sizeof(int32_t));
            int non_zero = 0;
            for (int e = 0; e < lut_size; e++) {
                if (lut_vals[e] != 0) non_zero++;
            }
            (void)non_zero;
            break;
        }
        // Disable callback - LUT handles remapping on GPU
        if (opt->ctx && !opt->lut_prefill_active) {
            llama_set_eval_callback(opt->ctx, nullptr, nullptr);
        }
    }

    opt->spec_draft_active = true;
}

void alpha_moe_optimizer_exit_draft_mode(alpha_moe_optimizer * opt) {
    if (!opt) return;

    // Restore original expert data pointers and callback IDs.
    for (int il = 0; il < opt->n_layer; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;
        auto & layer = opt->model->layers[il];
        if (info.orig_exps[OP_GATE])    layer.ffn_gate_exps    = info.orig_exps[OP_GATE];
        if (info.orig_exps[OP_UP])      layer.ffn_up_exps      = info.orig_exps[OP_UP];
        if (info.orig_exps[OP_DOWN])    layer.ffn_down_exps    = info.orig_exps[OP_DOWN];
        if (info.orig_exps[OP_GATE_UP]) layer.ffn_gate_up_exps = info.orig_exps[OP_GATE_UP];
        if (il < (int)opt->cache_expert_ids.size() && opt->cache_expert_ids[il]) {
            layer.ffn_cache_expert_ids = opt->cache_expert_ids[il];
        }
        if (il < (int)opt->cache_expert_ids_down.size() && opt->cache_expert_ids_down[il]) {
            layer.ffn_cache_expert_ids_down = opt->cache_expert_ids_down[il];
        }
    }
    // Re-enable callback
    if (opt->ctx) {
        llama_set_eval_callback(opt->ctx,
            alpha_moe_optimizer_get_eval_callback(),
            alpha_moe_optimizer_get_eval_data(opt));
    }

    opt->spec_draft_active = false;
}

bool alpha_moe_optimizer_is_draft_active(const alpha_moe_optimizer * opt) {
    return opt && opt->spec_draft_active;
}

extern thread_local int g_alpha_topk_override;
void alpha_moe_optimizer_set_topk(alpha_moe_optimizer * opt, int topk) {
    (void)opt;
    g_alpha_topk_override = topk;
    if (topk > 0) {
    }
}

bool alpha_moe_optimizer_has_shared_experts(const alpha_moe_optimizer * opt) {
    return opt && opt->has_shared_experts;
}

// alpha_mole_get_gating_probs defined in llama-graph.cpp (uses graph globals)

// Expert Override API

int alpha_mole_set_layer_experts(
        alpha_moe_optimizer * opt, int layer_idx,
        const int * expert_ids, const float * weights, int n_experts) {
    if (!opt || !opt->is_moe) return -1;
    if (layer_idx < 0 || layer_idx >= opt->n_layer) return -1;
    if (!expert_ids || n_experts <= 0) return -1;

    if (opt->override_experts.empty()) {
        opt->override_experts.resize(opt->n_layer);
        opt->override_weights.resize(opt->n_layer);
    }

    opt->override_experts[layer_idx].resize(n_experts);
    opt->override_weights[layer_idx].resize(n_experts);
    for (int i = 0; i < n_experts; i++) {
        opt->override_experts[layer_idx][i] = expert_ids[i];
        opt->override_weights[layer_idx][i] = weights ? weights[i] : 1.0f / n_experts;
    }
    return 0;
}

void alpha_mole_clear_layer_override(alpha_moe_optimizer * opt, int layer_idx) {
    if (!opt || layer_idx < 0 || layer_idx >= (int)opt->override_experts.size()) return;
    opt->override_experts[layer_idx].clear();
    opt->override_weights[layer_idx].clear();
}

void alpha_mole_clear_all_overrides(alpha_moe_optimizer * opt) {
    if (!opt) return;
    for (auto & v : opt->override_experts) v.clear();
    for (auto & v : opt->override_weights) v.clear();
}

float alpha_moe_optimizer_get_last_miss_rate(const alpha_moe_optimizer * opt) {
    if (!opt || opt->last_token_total_layers == 0) return 0.0f;
    return (float)opt->last_token_miss_layers / (float)opt->last_token_total_layers;
}

int alpha_moe_optimizer_get_miss_layers(const alpha_moe_optimizer * opt) {
    return opt ? opt->last_token_miss_layers : 0;
}

int alpha_moe_optimizer_get_total_layers(const alpha_moe_optimizer * opt) {
    return opt ? opt->last_token_total_layers : 0;
}

void alpha_moe_optimizer_set_lut_prefill(alpha_moe_optimizer * opt, bool active) {
    if (opt) opt->lut_prefill_active = active;
}

bool alpha_moe_optimizer_is_lut_prefill(const alpha_moe_optimizer * opt) {
    return opt && opt->lut_prefill_active;
}

float alpha_moe_optimizer_get_hit_rate(const alpha_moe_optimizer * opt) {
    if (!opt) return 0.0f;
    int64_t total = opt->cache_hits + opt->cache_misses;
    if (total == 0) return 0.0f;
    return (float)opt->cache_hits / (float)total;
}

// Check if speculative pass predictions were all correct.
// Returns true if all predictions matched (result can be used as-is).
// Returns false if any mismatch (need re-execution with reactive callback).
bool alpha_moe_optimizer_verify_speculative(alpha_moe_optimizer * opt) {
    if (!opt || !opt->is_moe || !opt->cache_enabled) return true;

    int n_argsort = alpha_mole_get_argsort_count();
    int moe_idx = 0;
    bool all_set_match = true;

    int layers_checked = 0;
    int layers_all_cached = 0;   // all 8 actual experts in GPU cache
    int layers_set_match = 0;    // all 8 actual experts in predicted set
    int total_expert_cache_misses = 0;
    int total_expert_set_misses = 0;
    char layer_map[64] = {};     // per-layer status character

    // Debug: count layer types
    {
        int n_has = 0, n_gpu = 0, n_cpu = 0;
        for (int il = 0; il < opt->n_layer; il++) {
            if (opt->layer_info[il].has_experts) {
                n_has++;
                if (opt->layer_info[il].on_gpu) n_gpu++; else n_cpu++;
            }
        }
    }

    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx);
        moe_idx++;
        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        if (n_used <= 0) continue;

        int32_t actual_ids[16] = {};
        ggml_backend_tensor_get(argsort_out, actual_ids, 0, n_used * sizeof(int32_t));


        auto & cache = opt->caches[il];
        int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                     (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                     (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));

        bool layer_cache_ok = true;
        bool layer_set_ok = true;
        auto & predicted = opt->prev_routing[il];

        for (int e = 0; e < n_used; e++) {
            int eid = actual_ids[e];
            // Cache presence check
            if (eid >= 0 && eid < opt->n_expert && cache.slot_map[ref_op][eid] < 0) {
                layer_cache_ok = false;
                total_expert_cache_misses++;
            }
            // Set match check (is eid in the predicted set?)
            bool found = false;
            for (int p = 0; p < n_used && p < (int)predicted.size(); p++) {
                if (eid == predicted[p]) { found = true; break; }
            }
            if (!found) {
                layer_set_ok = false;
                total_expert_set_misses++;
            }
        }

        if (layer_cache_ok) layers_all_cached++;
        if (layer_set_ok) layers_set_match++;
        if (!layer_set_ok) all_set_match = false;

        // '.' = all cached + set match, 'o' = cached but set mismatch, 'X' = cache miss
        layer_map[layers_checked] = layer_cache_ok ? (layer_set_ok ? '.' : 'o') : 'X';
        layers_checked++;
    }
    layer_map[layers_checked] = '\0';

    ALPHA_LOG_DBG("[SPEC-DIAG] t=%d cached=%d/%d set=%d/%d "
            "cmiss=%d smiss=%d %s %s\n",
            opt->total_tokens,
            layers_all_cached, layers_checked,
            layers_set_match, layers_checked,
            total_expert_cache_misses, total_expert_set_misses,
            (layers_all_cached == layers_checked) ? "ALL-CACHED" :
            (layers_set_match == layers_checked) ? "ALL-SET-MATCH" : "",
            layer_map);

    return all_set_match;
}

// Speculative Two-Pass Decode
// No callback, no graph splits. Run graph speculatively, verify, re-run on miss.
// Much faster than the reactive callback, which splits the graph at every MoE layer.
int alpha_moe_optimizer_speculative_decode(
        alpha_moe_optimizer * opt, struct llama_context * ctx, llama_token token) {
    if (!opt || !opt->is_moe || !opt->cache_enabled) {
        llama_batch b = llama_batch_get_one(&token, 1);
        return llama_decode(ctx, b);
    }

    // Phase 1, prepare: load predicted experts from prev_routing
    alpha_moe_optimizer_set_speculative(opt, true);  // disable callback
    alpha_moe_optimizer_prepare(opt);

    // Phase 2, speculate: run the graph at full speed (no callback, no graph splits)
    llama_batch batch = llama_batch_get_one(&token, 1);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) return rc;

    // Phase 3, verify: read argsort outputs and check cache coverage
    int n_argsort = alpha_mole_get_argsort_count();
    bool need_rerun = false;
    int moe_idx = 0;
    int total_misses = 0;

    // First pass: check all layers and DMA any missing experts
    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        auto & cache = opt->caches[il];
        if (!info.has_experts || info.on_gpu) continue;

        ggml_tensor * argsort_out = alpha_mole_get_argsort_output(moe_idx);
        moe_idx++;
        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        if (n_used <= 0) continue;

        // Read actual expert IDs
        int32_t actual_ids[16] = {};
        ggml_backend_tensor_get(argsort_out, actual_ids, 0, n_used * sizeof(int32_t));

        int ref_op = (opt->use_scratch || opt->tiered_cache) ? OP_DOWN :
                     (cache.gpu_tensor[OP_GATE_UP] ? OP_GATE_UP :
                     (cache.gpu_tensor[OP_GATE] ? OP_GATE : OP_DOWN));

        // Check each expert and DMA misses immediately
        int32_t remapped[16] = {};
        for (int e = 0; e < n_used; e++) {
            int eid = actual_ids[e];
            if (eid < 0 || eid >= opt->n_expert) eid = 0;

            int slot = cache.slot_map[ref_op][eid];
            if (slot >= 0) {
                // Cache hit
                remapped[e] = slot;
                opt->cache_hits++;
            } else {
                // Cache miss - DMA this expert into cache
                need_rerun = true;
                total_misses++;
                opt->cache_misses++;

                // LRU evict
                int lru_slot = 1;  // slot 0 is reserved (zero expert)
                int64_t lru_time = cache.slots[ref_op][1].last_used;
                for (int k = 1; k < opt->K; k++) {
                    if (cache.slots[ref_op][k].last_used < lru_time) {
                        lru_time = cache.slots[ref_op][k].last_used;
                        lru_slot = k;
                    }
                }

                // DMA all ops for this expert
                if (opt->use_scratch) {
                    // Scratch mode: only DMA into per-layer down cache
                    int old_eid = cache.slots[OP_DOWN][lru_slot].expert_id;
                    if (old_eid >= 0) cache.slot_map[OP_DOWN][old_eid] = -1;
                    size_t sb = info.expert_slice_bytes[OP_DOWN];
                    ggml_backend_tensor_set(cache.gpu_tensor[OP_DOWN],
                        (const char *)info.orig_exps[OP_DOWN]->data + eid * sb,
                        lru_slot * sb, sb);
                    cache.slot_map[OP_DOWN][eid] = lru_slot;
                    cache.slots[OP_DOWN][lru_slot] = { eid, ++opt->cache_clock };
                } else {
                    for (int op = 0; op < OP_COUNT; op++) {
                        if (!cache.gpu_tensor[op] || !info.orig_exps[op]) continue;
                        if (cache.slots[op].empty()) continue;
                        int old_eid = cache.slots[op][lru_slot].expert_id;
                        if (old_eid >= 0) cache.slot_map[op][old_eid] = -1;
                        size_t sb = info.expert_slice_bytes[op];
                        ggml_backend_tensor_set(cache.gpu_tensor[op],
                            (const char *)info.orig_exps[op]->data + eid * sb,
                            lru_slot * sb, sb);
                        cache.slot_map[op][eid] = lru_slot;
                        cache.slots[op][lru_slot] = { eid, ++opt->cache_clock };
                    }
                }
                remapped[e] = cache.slot_map[ref_op][eid];
            }

            // Update LRU timestamp
            if (opt->use_scratch || opt->tiered_cache) {
                int s = cache.slot_map[OP_DOWN][eid];
                if (s >= 0) cache.slots[OP_DOWN][s].last_used = ++opt->cache_clock;
            } else {
                for (int op = 0; op < OP_COUNT; op++) {
                    int s = cache.slot_map[op][eid];
                    if (s >= 0 && !cache.slots[op].empty())
                        cache.slots[op][s].last_used = ++opt->cache_clock;
                }
            }
        }

        // Write correct cache_expert_ids for this layer
        ggml_tensor * ids_tensor = opt->cache_expert_ids[il];
        if (ids_tensor) {
            ggml_backend_tensor_set(ids_tensor, remapped, 0,
                n_used * sizeof(int32_t));
        }

        // Update prev_routing with actual routing (for next token's prediction)
        for (int e = 0; e < n_used && e < (int)opt->prev_routing[il].size(); e++) {
            opt->prev_routing[il][e] = actual_ids[e];
        }
    }

    // Phase 4: Re-run if any misses (or if ordering was wrong)
    // Always re-run for now to guarantee correctness.
    // TODO: skip the re-run when all experts were in cache and
    //       prepare()'s cache_expert_ids matched the actual routing.
    {
        // KV rewind - remove speculative token's KV entries
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            if (pos_max >= 0) {
                llama_memory_seq_rm(mem, 0, pos_max, -1);
            }
        }

        // Re-run with correct cache_expert_ids (no callback, full speed)
        rc = llama_decode(ctx, batch);
        if (rc != 0) return rc;
    }

    // Update statistics
    opt->total_tokens++;
    static int64_t spec_tokens = 0;
    static int64_t spec_reruns = 0;
    spec_tokens++;
    if (need_rerun) spec_reruns++;

    // Log periodically
    if (spec_tokens % 10 == 0) {
        ALPHA_LOG_DBG("[SPEC-2P] tokens=%lld reruns=%lld (%.1f%%) misses=%d "
                "hit_rate=%.1f%%\n",
                (long long)spec_tokens, (long long)spec_reruns,
                100.0 * spec_reruns / spec_tokens, total_misses,
                (opt->cache_hits + opt->cache_misses > 0) ?
                100.0 * opt->cache_hits / (opt->cache_hits + opt->cache_misses) : 0);
    }

    return 0;
}

ggml_backend_sched_eval_callback alpha_moe_optimizer_get_eval_callback(void) {
    return reactive_cache_eval_callback;
}

void * alpha_moe_optimizer_get_eval_data(alpha_moe_optimizer * opt) {
    return (void *)opt;
}

void alpha_moe_optimizer_reinstall_callback(alpha_moe_optimizer * opt) {
    if (!opt || !opt->ctx) return;
    llama_set_eval_callback(opt->ctx,
        reactive_cache_eval_callback, (void *)opt);
}

void alpha_moe_optimizer_clear_expert_remap(alpha_moe_optimizer * opt) {
    if (!opt || !opt->model) return;
    for (int il = 0; il < opt->n_layer; il++) {
        opt->model->layers[il].ffn_cache_expert_ids = nullptr;
        opt->model->layers[il].ffn_cache_expert_ids_down = nullptr;
    }
    if (opt->ctx) llama_context_reset_sched(opt->ctx);
}

void alpha_moe_optimizer_restore_expert_remap(alpha_moe_optimizer * opt) {
    if (!opt || !opt->model) return;
    for (int il = 0; il < opt->n_layer; il++) {
        if (il < (int)opt->spec_lut.size() && opt->spec_lut[il]) {
            opt->model->layers[il].ffn_cache_expert_ids = opt->spec_lut[il];
        }
    }
    if (opt->ctx) llama_context_reset_sched(opt->ctx);
}

// Non-CUDA stub implementations, for every build that lacks the Windows DLL forwarders above (those exist only
// with LLAMA_SHARED): Linux without CUDA and static Windows CPU builds would otherwise leave these unresolved.
// They are no-ops - DMA functions do nothing without a GPU.
#if !defined(GGML_USE_CUDA) && !(defined(_WIN32) && defined(LLAMA_SHARED))
extern "C" {
void _acdi(void) {}
void _acda(struct ggml_tensor *, const void *, size_t, size_t) {}
void _acds(void) {}
void _acdb(void) {}
void * _acpa(size_t) { return nullptr; }
void   _acpf(void *) {}
void   _acdas(struct ggml_tensor *, const void *, size_t, size_t, void *) {}
}
#endif
