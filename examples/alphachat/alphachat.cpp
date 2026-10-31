// Platform macros must be first (before any includes)
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

// AlphaChat CLI - wraps the AlphaLlama engine
//
// Ollama-compatible commands + training, benchmarking, provider mode.
//
// Usage:
//   alphachat run deepseek-v4-flash "Hello"
//   alphachat pull qwen3.5:4b
//   alphachat list
//   alphachat train qwen3.5:4b --data train.jsonl
//   alphachat bench deepseek-v4-flash
//   alphachat info

#include "alpha_lora_mix.h"
#include "alpha_autoconfig.h"
#include "alpha_expert_cache.h"
#include "alpha_hw_planner.h"
#include "alpha_registry.h"
#include "llama-arch.h"
#include "llama-model.h"
#include "common.h"
#include "download.h"
#include "alpha_moe_optimizer.h"
#include "alpha_speculative.h"
#include "alpha_lora_train.h"
#include "alpha_train_engine.h"
#include "alpha_train_combined.h"
#include "alpha_train_full.h"
#include "alpha_train_chunked.h"
#include "alpha_moe_portfolio.h"
#include "alpha_dense_pipeline.h"
#include "speculative.h"
#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "gguf.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#if defined(__linux__)
#include <sys/prctl.h>
#endif
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>
#endif

#include <sys/stat.h>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#if !defined(_WIN32)
#include <signal.h>
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;

// JSON string escaping

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char hex[8]; snprintf(hex, sizeof(hex), "\\u%04x", (unsigned char)c);
                    out += hex;
                } else { out += c; }
        }
    }
    return out;
}

// Self-location (find alphallama in same directory)

static std::string get_self_dir() {
#if defined(_WIN32)
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf);
    auto slash = p.find_last_of("\\/");
    return (slash != std::string::npos) ? p.substr(0, slash) : ".";
#elif defined(__linux__)
    char buf[4096] = {};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        std::string p(buf, len);
        auto slash = p.rfind('/');
        return (slash != std::string::npos) ? p.substr(0, slash) : ".";
    }
    return ".";
#elif defined(__APPLE__)
    char buf[4096] = {};
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        std::string p(buf);
        auto slash = p.rfind('/');
        return (slash != std::string::npos) ? p.substr(0, slash) : ".";
    }
    return ".";
#else
    return ".";
#endif
}

// Config

static std::string get_models_dir() {
    const char * home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");
    if (!home) home = ".";
    return std::string(home) + "/.alphachat/models";
}

static std::string get_config_dir() {
    const char * home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");
    if (!home) home = ".";
    return std::string(home) + "/.alphachat";
}

// Model Resolution

// Find a model GGUF file by name. Checks:
// 1. Exact path (if file exists)
// 2. ~/.alphachat/models/<name>.gguf
// 3. ~/.alphachat/models/<name>-*.gguf (any quant)
// 4. AlphaLlama blobs (if name matches ollama tag)
static std::string resolve_model(const std::string & name) {
    // 1. Exact path
    if (fs::exists(name)) return name;

    // 2. Models dir
    std::string models_dir = get_models_dir();
    std::string path = models_dir + "/" + name + ".gguf";
    if (fs::exists(path)) return path;

    // 3. Glob with quant suffix
    if (fs::exists(models_dir)) {
        for (auto & entry : fs::directory_iterator(models_dir)) {
            std::string fn = entry.path().filename().string();
            if (fn.find(name) == 0 && fn.size() > 5 &&
                fn.substr(fn.size()-5) == ".gguf") {
                return entry.path().string();
            }
        }
    }

    // 3b. Custom model directory from ALPHACHAT_MODEL_DIR environment variable
    const char * custom_dir = getenv("ALPHACHAT_MODEL_DIR");
    if (custom_dir && fs::exists(custom_dir)) {
        std::string p = std::string(custom_dir) + "/" + name + ".gguf";
        if (fs::exists(p)) return p;
        try {
            for (auto & entry : fs::directory_iterator(custom_dir)) {
                std::string fn = entry.path().filename().string();
                if (fn.find(name) != std::string::npos && fn.size() > 5 &&
                    fn.substr(fn.size()-5) == ".gguf") {
                    return entry.path().string();
                }
            }
        } catch (...) {}
    }

    // 3c. Scan common model directories (Linux/server environments)
    const char * extra_dirs[] = {"/data", "/data/models", "/models", nullptr};
    for (int d = 0; extra_dirs[d]; d++) {
        if (!fs::exists(extra_dirs[d])) continue;
        std::string p = std::string(extra_dirs[d]) + "/" + name + ".gguf";
        if (fs::exists(p)) return p;
        try {
            for (auto & entry : fs::directory_iterator(extra_dirs[d])) {
                if (!entry.is_regular_file() && !entry.is_symlink()) continue;
                std::string fn = entry.path().filename().string();
                if (fn.find(name) != std::string::npos && fn.size() > 5 &&
                    fn.substr(fn.size()-5) == ".gguf") {
                    return entry.path().string();
                }
            }
        } catch (...) {}
    }

    // 4. Ollama blobs - parse manifest
    std::string ollama_dir;
    const char * home = getenv("HOME");
    if (home) ollama_dir = std::string(home) + "/.ollama/models";
#ifdef _WIN32
    const char * userprofile = getenv("USERPROFILE");
    if (userprofile) ollama_dir = std::string(userprofile) + "/.ollama/models";
#endif

    if (!ollama_dir.empty()) {
        // Convert name like "qwen3.5:4b" to manifest path
        std::string tag = name;
        std::string variant = "latest";
        auto colon = tag.find(':');
        if (colon != std::string::npos) {
            variant = tag.substr(colon + 1);
            tag = tag.substr(0, colon);
        }
        std::string manifest = ollama_dir + "/manifests/registry.ollama.ai/library/" + tag + "/" + variant;
        if (fs::exists(manifest)) {
            std::ifstream f(manifest);
            std::string content((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
            // Find model layer - search for the layer block containing the model media type
            // AlphaLlama manifest JSON: {"layers": [{"digest":"sha256:...","mediaType":"...model"}, ...]}
            auto model_pos = content.find("application/vnd.ollama.image.model");
            if (model_pos != std::string::npos) {
                // Find the enclosing layer object - search backwards for the digest
                auto block_start = content.rfind("{", model_pos);
                if (block_start != std::string::npos) {
                    auto dpos = content.find("sha256:", block_start);
                    if (dpos != std::string::npos) {
                        auto dend = content.find("\"", dpos);
                        if (dend != std::string::npos) {
                            std::string digest = content.substr(dpos, dend - dpos);
                            std::replace(digest.begin(), digest.end(), ':', '-');
                            std::string blob = ollama_dir + "/blobs/" + digest;
                            if (fs::exists(blob)) return blob;
                        }
                    }
                }
            }
        }
    }

    return "";
}

// Commands

static int cmd_info() {
    ggml_backend_load_all();

    alpha_autoconfig cfg = {};
    alpha_autoconfig_detect_hardware(&cfg);

    printf("AlphaChat CLI:  v1.0.0\n");
    printf("Engine:         v1.0.0\n");
    if (cfg.gpu_name) {
        printf("GPU:            %s\n", cfg.gpu_name);
    } else {
        printf("GPU:            none detected\n");
    }
    printf("Models dir:     %s\n", get_models_dir().c_str());

    // Check models dir size
    std::string mdir = get_models_dir();
    size_t total_bytes = 0;
    int count = 0;
    if (fs::exists(mdir)) {
        for (auto & entry : fs::directory_iterator(mdir)) {
            if (entry.path().extension() == ".gguf") {
                total_bytes += entry.file_size();
                count++;
            }
        }
    }
    printf("Local models:   %d (%.1f GB)\n", count, total_bytes / 1e9);

    return 0;
}

static int cmd_list() {
    std::string mdir = get_models_dir();

    printf("%-35s %10s %10s %s\n", "NAME", "SIZE", "QUANT", "MODIFIED");

    // List ~/.alphachat/models/
    if (fs::exists(mdir)) {
        for (auto & entry : fs::directory_iterator(mdir)) {
            if (entry.path().extension() != ".gguf") continue;
            std::string name = entry.path().stem().string();
            auto size = entry.file_size();
            auto mtime = fs::last_write_time(entry);

            // Read quant from GGUF metadata
            std::string quant = "?";
            auto * ctx = gguf_init_from_file(entry.path().string().c_str(), {false, nullptr});
            if (ctx) {
                int ft_idx = gguf_find_key(ctx, "general.file_type");
                if (ft_idx >= 0) {
                    int ft = gguf_get_val_u32(ctx, ft_idx);
                    switch (ft) {
                        case 0: quant = "F32"; break;
                        case 1: quant = "F16"; break;
                        case 2: quant = "Q4_0"; break;
                        case 7: quant = "Q8_0"; break;
                        case 15: quant = "Q4_K_M"; break;
                        default: quant = "Q" + std::to_string(ft); break;
                    }
                }
                gguf_free(ctx);
            }

            char size_str[32];
            if (size > 1e9) snprintf(size_str, sizeof(size_str), "%.1f GB", size / 1e9);
            else snprintf(size_str, sizeof(size_str), "%.0f MB", size / 1e6);

            printf("%-35s %10s %10s\n", name.c_str(), size_str, quant.c_str());
        }
    }

    // Also scan AlphaLlama models
    std::string ollama_manifests;
    const char * home = getenv("HOME");
    if (!home) home = getenv("USERPROFILE");
    if (home) ollama_manifests = std::string(home) + "/.ollama/models/manifests/registry.ollama.ai/library";

    if (!ollama_manifests.empty() && fs::exists(ollama_manifests)) {
        for (auto & model_dir : fs::directory_iterator(ollama_manifests)) {
            if (!fs::is_directory(model_dir)) continue;
            std::string model_name = model_dir.path().filename().string();
            for (auto & tag_file : fs::directory_iterator(model_dir)) {
                std::string tag = tag_file.path().filename().string();
                std::string full_name = model_name + ":" + tag;
                std::string resolved = resolve_model(full_name);
                if (!resolved.empty()) {
                    size_t size = 0;
                    try { size = fs::file_size(resolved); } catch (...) {}
                    char size_str[32];
                    if (size > 1e9) snprintf(size_str, sizeof(size_str), "%.1f GB", size / 1e9);
                    else if (size > 0) snprintf(size_str, sizeof(size_str), "%.0f MB", size / 1e6);
                    else snprintf(size_str, sizeof(size_str), "?");
                    printf("%-35s %10s %10s (local)\n", full_name.c_str(), size_str, "");
                }
            }
        }
    }

    return 0;
}

static int cmd_show(const std::string & model_name) {
    std::string path = resolve_model(model_name);
    if (path.empty()) {
        fprintf(stderr, "Model not found: %s\n", model_name.c_str());
        fprintf(stderr, "Try: alphachat pull %s\n", model_name.c_str());
        return 2;
    }

    auto * ctx = gguf_init_from_file(path.c_str(), {false, nullptr});
    if (!ctx) {
        fprintf(stderr, "Failed to read GGUF: %s\n", path.c_str());
        return 1;
    }

    auto get_str = [&](const char * key) -> std::string {
        int idx = gguf_find_key(ctx, key);
        if (idx < 0) return "?";
        return gguf_get_val_str(ctx, idx);
    };
    auto get_u32 = [&](const char * key) -> uint32_t {
        int idx = gguf_find_key(ctx, key);
        if (idx < 0) return 0;
        return gguf_get_val_u32(ctx, idx);
    };

    std::string arch = get_str("general.architecture");
    std::string name = get_str("general.name");
    uint32_t n_layer = get_u32((arch + ".block_count").c_str());
    uint32_t n_embd = get_u32((arch + ".embedding_length").c_str());
    uint32_t n_expert = get_u32((arch + ".expert_count").c_str());
    uint32_t n_expert_used = get_u32((arch + ".expert_used_count").c_str());
    uint32_t n_ctx = get_u32((arch + ".context_length").c_str());
    uint32_t n_head = get_u32((arch + ".attention.head_count").c_str());

    size_t file_size = 0;
    try { file_size = fs::file_size(path); } catch (...) {}

    printf("Name:          %s\n", name.c_str());
    printf("Architecture:  %s\n", arch.c_str());
    printf("Layers:        %u\n", n_layer);
    printf("Embedding:     %u\n", n_embd);
    printf("Heads:         %u\n", n_head);
    if (n_expert > 0) {
        printf("Experts:       %u routed", n_expert);
        if (n_expert_used > 0) printf(" (%u active)", n_expert_used);
        printf("\n");
    }
    printf("Context:       %u tokens\n", n_ctx);
    printf("Size:          %.1f GB\n", file_size / 1e9);
    printf("Location:      %s\n", path.c_str());

    gguf_free(ctx);
    return 0;
}

static int cmd_rm(const std::string & model_name) {
    std::string mdir = get_models_dir();
    std::string path = mdir + "/" + model_name + ".gguf";

    if (!fs::exists(path)) {
        // Try glob
        bool found = false;
        if (fs::exists(mdir)) {
            for (auto & entry : fs::directory_iterator(mdir)) {
                if (entry.path().filename().string().find(model_name) == 0) {
                    path = entry.path().string();
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            fprintf(stderr, "Model not found in %s: %s\n", mdir.c_str(), model_name.c_str());
            return 2;
        }
    }

    size_t size = 0;
    try { size = fs::file_size(path); } catch (...) {}
    printf("Removing %s (%.1f GB)... ", path.c_str(), size / 1e9);
    if (!fs::exists(path)) { printf("file already missing\n"); return 0; }
    fs::remove(path);
    printf("done\n");
    return 0;
}

static bool g_flat_cache = false;   // set by --flat-cache in main()
static bool g_force_cache = false;  // set by --force-expert-cache in main()
static int g_top_k = 20;           // set by --top-k (matches Ollama model default)
static float g_top_p = 0.95f;      // set by --top-p in main()
static float g_min_p = 0.0f;       // set by --min-p in main()
static float g_presence_penalty = 1.5f;  // set by --presence-penalty (matches Ollama default)
static float g_frequency_penalty = 0.0f; // set by --frequency-penalty
// g_alpha_debug is declared in alpha_moe_optimizer.h, set by --debug
static bool g_cpu_moe = false;      // set by --cpu-moe or --mode cpu-moe in main()
static alpha_inference_mode g_alpha_mode = ALPHA_MODE_STANDARD; // set by --mode in main()

// Multi-token warmup state (cpu-moe tokens generated before self-spec takes over)
static std::vector<llama_token> g_warmup_tokens;
static llama_token g_warmup_last_tok = 0;
static bool g_warmup_eog = false;

// Layer-skip API from alpha_dense_pipeline (for self-spec draft)
// alpha_pipeline_set_layer_range() and alpha_pipeline_clear_layer_range()
// are declared in alpha_dense_pipeline.h (already included above)

static int cmd_run(const std::string & model_name, const std::string & prompt,
                   int n_predict, bool flash_attn, int expert_cache_mb,
                   int n_gpu_layers, int n_ctx, float temperature, uint32_t seed = 42,
                   bool ngl_explicit = false, bool interactive = false,
                   bool no_think = false,
                   float expert_sticky = 0.0f, float expert_layer_bias = 0.0f,
                   const std::string & expert_groups_path = "",
                   bool log_confidence = false,
                   const std::string & router_patch_path = "",
                   bool log_gate_entropy = false,
                   int split_group = 0,
                   bool expert_cache_explicit = false,
                   bool speculative_mode = false,
                   const std::vector<std::pair<int, std::vector<int>>> & expert_overrides = {},
                   int self_spec_n = 0,
                   int self_spec_experts = 0,
                   bool no_barrier = false,
                   bool json_output = false,
                   int topk_experts = 0,
                   int cached_spec_n = 0,
                   int tier1_cycles = 0,
                   const std::string & mixed_cluster_path = "",
                   const std::string & profile_experts_path = "",
                   const std::string & profile_freq_path = "",
                   const std::string & load_freq_path = "",
                   const std::string & draft_model_path = "",
                   int draft_n = 4,
                   int draft_gpu_layers = -1,
                   int self_spec_skip = 0,
                   int self_spec_m = 4) {
    std::string path = resolve_model(model_name);
    if (path.empty()) {
        fprintf(stderr, "Model not found: %s\n", model_name.c_str());
        fprintf(stderr, "Try: alphachat pull %s\n", model_name.c_str());
        return 2;
    }

    // Auto-discover frequency profile if not explicitly specified
    std::string auto_freq_path = load_freq_path;
    if (auto_freq_path.empty()) {
        // Check for freq_profile.json in the model directory
        std::string model_dir = path;
        size_t last_sep = model_dir.find_last_of("/\\");
        if (last_sep != std::string::npos) {
            model_dir = model_dir.substr(0, last_sep);
        }
        std::string candidate = model_dir + "/freq_profile.json";
        FILE * test_fp = fopen(candidate.c_str(), "r");
        if (test_fp) {
            fclose(test_fp);
            auto_freq_path = candidate;
            ALPHA_LOG_DBG("[FREQ] Auto-discovered frequency profile: %s\n", candidate.c_str());
        }
    }

    // Suppress all llama/ggml internal log output (tensor names, KV cache, architecture)
    // Only pass through fatal errors - everything else is internal detail
    llama_log_set([](enum ggml_log_level level, const char * msg, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fputs(msg, stderr);
        }
    }, nullptr);

    ggml_backend_load_all();

    // Hardware Planner: auto-detect devices and compute optimal placement
    alpha_hw_plan hw_plan = alpha_hw_plan_create(
        path.c_str(),
        ngl_explicit ? n_gpu_layers : -1
    );
    // Check total capacity
    if (!hw_plan.can_run) {
        fprintf(stderr, "Error: not enough memory to run this model.\n");
        fprintf(stderr, "Try a smaller model or a lower quantization (e.g. Q2_K, IQ2_XS).\n");
        return 4;
    }

    // Apply plan
    n_gpu_layers = hw_plan.n_gpu_layers;
    enum ggml_type kv_type_k = hw_plan.kv_type_k;
    enum ggml_type kv_type_v = hw_plan.kv_type_v;
    int auto_ctx = hw_plan.n_ctx;

    // Alpha Optimized: auto-enable self-spec for dense models that partially fit
    if (hw_plan.strategy == ALPHA_HW_SELF_SPEC_DENSE && self_spec_skip == 0) {
        self_spec_skip = hw_plan.self_spec_skip;
        self_spec_m = hw_plan.self_spec_m > 0 ? hw_plan.self_spec_m : 2;
        n_gpu_layers = hw_plan.self_spec_draft_ngl;
        ALPHA_LOG_DBG("[alpha-optimized] Auto-enabled: skip=%d, M=%d, draft_ngl=%d\n",
                self_spec_skip, self_spec_m, n_gpu_layers);
    }

    // Portfolio: auto-select optimal strategy based on fit ratio
    bool fits_in_gpu = (hw_plan.expert_cache_mb == 0);
    alpha_model_family family = alpha_moe_detect_family(
        hw_plan.arch, hw_plan.n_expert, hw_plan.n_expert_used, hw_plan.model_size_mb);

    alpha_portfolio_context port_ctx = {};
    port_ctx.family        = family;
    port_ctx.is_hybrid     = false;  // determined after model load
    port_ctx.n_expert      = hw_plan.n_expert;
    port_ctx.n_expert_used = hw_plan.n_expert_used;
    port_ctx.gpu_vram_mb   = hw_plan.gpu_vram_mb;
    port_ctx.model_size_mb = hw_plan.model_size_mb;
    port_ctx.K_estimate    = hw_plan.K_estimate;
    port_ctx.K_min         = hw_plan.K_min;
    port_ctx.K_draft       = hw_plan.K_draft;
    port_ctx.fit_ratio     = (hw_plan.gpu_vram_mb > 0)
        ? (float)hw_plan.model_size_mb / (float)hw_plan.gpu_vram_mb : 0.0f;

    alpha_moe_profile portfolio = alpha_moe_portfolio_select_v2(&port_ctx);
    alpha_moe_portfolio_print(&portfolio);

    // Apply execution plan from portfolio (CLI flags override)
    // The plan drives all runtime decisions. No HW strategy switch needed.

    // Expert cache budget: plan -> HW plan -> CLI override
    if (!expert_cache_explicit) {
        if (portfolio.expert_cache_max > 0)
            expert_cache_mb = portfolio.expert_cache_max;
        else if (portfolio.expert_cache_mb > 0)
            expert_cache_mb = portfolio.expert_cache_mb;
        else if (hw_plan.expert_cache_mb > 0)
            expert_cache_mb = hw_plan.expert_cache_mb;
    }
    // Plan-driven flags (CLI always wins)
    if (!expert_cache_explicit && !g_cpu_moe && portfolio.cpu_moe) {
        g_cpu_moe = true;
    }
    // CLI --cpu-moe overrides portfolio plan to CPU_MOE behavior
    if (g_cpu_moe && portfolio.buft != ALPHA_BUFT_CPU_MOE) {
        ALPHA_LOG_DBG("[CLI] --cpu-moe: overriding portfolio plan to CPU_MOE\n");
        portfolio.buft = ALPHA_BUFT_CPU_MOE;
        portfolio.callback = ALPHA_CB_LUT_DISABLED;
        portfolio.warmup = ALPHA_WARMUP_CLASSIFY;
        portfolio.gen = ALPHA_GEN_LUT_DIRECT;
        portfolio.need_moe_opt = true;
        portfolio.expert_cache_max = 99999;
        portfolio.disable_cache_for_prompt = true;
        portfolio.lut_init = true;
        portfolio.gpu_layers_all = true;
        portfolio.cpu_moe = true;
        portfolio.conf_margin_threshold = 0.3f;
        portfolio.conf_trigger_count = 3;
        portfolio.eos_verify = false;
        portfolio.quality_grace_tokens = 5;
    }
    if (!g_flat_cache && portfolio.flat_cache) {
        g_flat_cache = true;
    }
    if (topk_experts == 0 && portfolio.topk_experts > 0) {
        topk_experts = portfolio.topk_experts;
    }
    if (cached_spec_n == 0 && portfolio.cached_spec_n > 0) {
        cached_spec_n = portfolio.cached_spec_n;
    }
    if (split_group == 0 && portfolio.split_group > 0) {
        split_group = portfolio.split_group;
    }
    if (portfolio.kv_quant != 0 && portfolio.kv_quant != GGML_TYPE_F16) {
        if (kv_type_k == GGML_TYPE_F16) kv_type_k = portfolio.kv_quant;
        if (kv_type_v == GGML_TYPE_F16) kv_type_v = portfolio.kv_quant;
    }
    if (n_ctx == 0 && auto_ctx == 0 && portfolio.n_ctx_default > 0) {
        auto_ctx = portfolio.n_ctx_default;
    }
    // GPU layers: plan says all, or force-fit, or explicit
    if (!ngl_explicit && (portfolio.gpu_layers_all || portfolio.force_fit)) {
        n_gpu_layers = 999;
    }

    // When ALL_GPU and user explicitly asked for expert cache - warn and disable
    if (portfolio.buft == ALPHA_BUFT_NONE && expert_cache_explicit && expert_cache_mb > 0
            && !g_force_cache && !g_cpu_moe) {
        ALPHA_LOG_DBG("[alphachat] model fits in GPU — ignoring --expert-cache %d\n", expert_cache_mb);
        expert_cache_mb = 0;
    }

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    if (hw_plan.devices[0]) {
        model_params.devices = hw_plan.devices;
    }
    model_params.split_mode = hw_plan.split_mode;
    if (hw_plan.main_device_idx >= 0) {
        model_params.main_gpu = hw_plan.main_device_idx;
    }
    if (hw_plan.split_mode == LLAMA_SPLIT_MODE_LAYER && hw_plan.tensor_split[0] > 0) {
        model_params.tensor_split = hw_plan.tensor_split;
    }

    // Expert offload if needed + output layer CPU pinning for precision
    ggml_backend_buffer_type_t cpu_buft_for_overrides = nullptr;
    llama_model_tensor_buft_override buft_overrides[8] = {};
    int n_overrides = 0;

    // Find CPU buffer type for overrides
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_buft_for_overrides = ggml_backend_dev_buffer_type(dev);
            break;
        }
    }

    if (cpu_buft_for_overrides) {
        // Buffer type overrides driven by execution plan
        switch (portfolio.buft) {
            case ALPHA_BUFT_CPU_MOE: {
                auto ovr = llm_ffn_exps_cpu_override();
                buft_overrides[n_overrides++] = { ovr.pattern, ovr.buft };
                ALPHA_LOG_DBG("[PLAN] cpu-moe: experts pinned to CPU\n");
                break;
            }
            case ALPHA_BUFT_EXPERT_CACHE:
                buft_overrides[n_overrides++] = { "exps", cpu_buft_for_overrides };
                buft_overrides[n_overrides++] = { "nextn", cpu_buft_for_overrides };
                ALPHA_LOG_DBG("[PLAN] expert-cache: exps+nextn pinned to CPU\n");
                break;
            case ALPHA_BUFT_NONE:
                break;  // All on GPU - no overrides needed
        }
        // CLI --cpu-moe override (when plan didn't set it but user forced it)
        // Replaces any portfolio-chosen overrides (e.g. EXPERT_CACHE) with cpu-moe
        if (g_cpu_moe && portfolio.buft != ALPHA_BUFT_CPU_MOE) {
            n_overrides = 0;  // clear portfolio overrides
            auto ovr = llm_ffn_exps_cpu_override();
            buft_overrides[n_overrides++] = { ovr.pattern, ovr.buft };
            ALPHA_LOG_DBG("[CLI] --cpu-moe override: experts pinned to CPU\n");
        }
        // CLI --expert-cache override (when plan didn't set it but user forced it)
        if (expert_cache_explicit && expert_cache_mb > 0 && portfolio.buft != ALPHA_BUFT_EXPERT_CACHE && n_overrides == 0) {
            buft_overrides[n_overrides++] = { "exps", cpu_buft_for_overrides };
            buft_overrides[n_overrides++] = { "nextn", cpu_buft_for_overrides };
            ALPHA_LOG_DBG("[CLI] --expert-cache override: exps+nextn pinned to CPU\n");
        }
        // Portfolio-driven output CPU pin (V4 Flash precision fix)
        if (portfolio.output_cpu) {
            buft_overrides[n_overrides++] = { "output", cpu_buft_for_overrides };
            ALPHA_LOG_DBG("[PLAN] output pinned to CPU (%s precision fix)\n", portfolio.family_name);
        }
        if (n_overrides > 0) {
            buft_overrides[n_overrides] = { nullptr, nullptr };
            model_params.tensor_buft_overrides = buft_overrides;
        }
    }

    // Detect split GGUF (e.g. model-00001-of-00006.gguf) and use explicit shard loader.
    // Needed for models where shard 1's tokenizer spans across shards (397B).
    llama_model * model = nullptr;
    {
        size_t split_pos = path.rfind("-00001-of-");
        if (split_pos != std::string::npos) {
            int n_shards = std::atoi(path.c_str() + split_pos + 10);
            if (n_shards > 1) {
                std::string prefix = path.substr(0, split_pos);
                std::vector<std::string> shard_paths;
                std::vector<const char *> shard_ptrs;
                for (int i = 1; i <= n_shards; i++) {
                    char shard[4096];
                    snprintf(shard, sizeof(shard), "%s-%05d-of-%05d.gguf", prefix.c_str(), i, n_shards);
                    shard_paths.push_back(shard);
                }
                for (auto & s : shard_paths) shard_ptrs.push_back(s.c_str());
                ALPHA_LOG_DBG("[alphachat] Split GGUF detected: %d shards\n", n_shards);
                model = llama_model_load_from_splits(shard_ptrs.data(), shard_ptrs.size(), model_params);
            }
        }
        if (!model && split_pos == std::string::npos) {
            // Only try single-file loader if not a split GGUF
            model = llama_model_load_from_file(path.c_str(), model_params);
        }
    }
    if (!model) {
        if (!fs::exists(path)) {
            fprintf(stderr, "Error: model file not found: %s\n", path.c_str());
        } else {
            size_t file_sz = 0;
            try { file_sz = fs::file_size(path); } catch (...) {}
            if (file_sz < 1024) {
                fprintf(stderr, "Error: model file too small or corrupted (%zu bytes): %s\n", file_sz, path.c_str());
            } else {
                fprintf(stderr, "Error: failed to load model: %s\n", path.c_str());
                fprintf(stderr, "  Possible causes: insufficient GPU memory, unsupported format, or corrupted file\n");
            }
        }
        return 3;
    }

    // Router patch: override gate projection weights with retrained router
    // Format: safetensors with tensors "layers.{L}.mlp.gate.weight" [n_expert, n_embd] F32
    // and optional "layers.{L}.mlp.gate.bias" [n_expert] F32
    if (!router_patch_path.empty()) {
        ALPHA_LOG_DBG("[router-patch] Loading retrained router from %s\n", router_patch_path.c_str());

        FILE * pf = fopen(router_patch_path.c_str(), "rb");
        if (!pf) {
            ALPHA_LOG_DBG("[router-patch] ERROR: cannot open %s\n", router_patch_path.c_str());
        } else {
            // Read safetensors header
            uint64_t header_size = 0;
            fread(&header_size, 8, 1, pf);
            if (header_size > 0 && header_size < 1024 * 1024) {
                std::string header_json(header_size, '\0');
                fread(&header_json[0], 1, header_size, pf);
                long data_offset = 8 + (long)header_size;

                int n_layer = llama_model_n_layer(model);
                int patched = 0;

                // Parse each tensor entry from the header
                // Look for "layers.{L}.mlp.gate.weight" entries
                for (int il = 0; il < n_layer; il++) {
                    char key_w[64], key_b[64];
                    snprintf(key_w, sizeof(key_w), "\"layers.%d.mlp.gate.weight\"", il);
                    snprintf(key_b, sizeof(key_b), "\"layers.%d.mlp.gate.bias\"", il);

                    // Find weight tensor in header
                    size_t wpos = header_json.find(key_w);
                    if (wpos == std::string::npos) continue;

                    // Extract data_offsets: [start, end]
                    size_t dpos = header_json.find("\"data_offsets\"", wpos);
                    if (dpos == std::string::npos) continue;
                    size_t arr_s = header_json.find('[', dpos);
                    size_t arr_e = header_json.find(']', arr_s);
                    if (arr_s == std::string::npos || arr_e == std::string::npos) continue;

                    long off_start = 0, off_end = 0;
                    sscanf(header_json.c_str() + arr_s + 1, "%ld, %ld", &off_start, &off_end);
                    long tensor_bytes = off_end - off_start;

                    if (tensor_bytes <= 0 || tensor_bytes > 100 * 1024 * 1024) continue;

                    // Get the model's gate tensor for this layer
                    // Access internal model structure (llama-model.h included)
                    const llama_model & mdl = *model;
                    ggml_tensor * gate = (il < (int)mdl.layers.size()) ? mdl.layers[il].ffn_gate_inp : nullptr;
                    if (!gate || !gate->buffer) {
                        ALPHA_LOG_DBG("[router-patch] layer %d: no gate tensor in model\n", il);
                        continue;
                    }

                    size_t gate_bytes = ggml_nbytes(gate);

                    // Read patch data
                    std::vector<float> patch_data(tensor_bytes / sizeof(float));
                    fseek(pf, data_offset + off_start, SEEK_SET);
                    fread(patch_data.data(), 1, tensor_bytes, pf);

                    if ((size_t)tensor_bytes == gate_bytes) {
                        // Same total size - check if transpose needed
                        // GGUF stores [n_embd, n_expert], safetensors may store [n_expert, n_embd]
                        int64_t gate_ne0 = gate->ne[0]; // n_embd in GGUF
                        int64_t gate_ne1 = gate->ne[1]; // n_expert in GGUF

                        // Extract patch shape from header
                        size_t shpos = header_json.find("\"shape\"", wpos);
                        int patch_d0 = 0, patch_d1 = 0;
                        if (shpos != std::string::npos) {
                            size_t sh_s = header_json.find('[', shpos);
                            if (sh_s != std::string::npos)
                                sscanf(header_json.c_str() + sh_s + 1, "%d, %d", &patch_d0, &patch_d1);
                        }

                        bool needs_transpose = (patch_d0 == gate_ne1 && patch_d1 == gate_ne0
                                                && patch_d0 != patch_d1);

                        if (needs_transpose) {
                            // Transpose [n_expert, n_embd] -> [n_embd, n_expert]
                            std::vector<float> transposed(patch_data.size());
                            for (int r = 0; r < patch_d0; r++)
                                for (int c = 0; c < patch_d1; c++)
                                    transposed[c * patch_d0 + r] = patch_data[r * patch_d1 + c];
                            ggml_backend_tensor_set(gate, transposed.data(), 0, gate_bytes);
                            ALPHA_LOG_DBG("[router-patch] layer %d: transposed [%d,%d] → [%d,%d]\n",
                                    il, patch_d0, patch_d1, (int)gate_ne0, (int)gate_ne1);
                        } else {
                            ggml_backend_tensor_set(gate, patch_data.data(), 0, gate_bytes);
                        }
                        patched++;
                    } else {
                        ALPHA_LOG_DBG("[router-patch] layer %d: size mismatch (patch=%ld, model=%zu)\n",
                                il, tensor_bytes, gate_bytes);
                    }
                }

                ALPHA_LOG_DBG("[router-patch] patched %d/%d layers\n", patched, n_layer);
            } else {
                ALPHA_LOG_DBG("[router-patch] ERROR: invalid safetensors header size %llu\n",
                        (unsigned long long)header_size);
            }
            fclose(pf);
        }
    }

    // Context
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_context_params ctx_params = llama_context_default_params();

    int n_prompt_tokens;
    if (interactive) {
        n_prompt_tokens = 512;  // estimate for interactive
        ctx_params.n_ctx = auto_ctx > 0 ? auto_ctx : (n_ctx > 0 ? n_ctx : 4096);
    } else {
        n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
        ctx_params.n_ctx = auto_ctx > 0 ? auto_ctx : (n_ctx > 0 ? n_ctx : n_prompt_tokens + n_predict + 16);
    }
    // n_batch: use 512 as default (handles both MoE and dense models safely)
    ctx_params.n_batch = 512;
    // Use all hardware threads.
    // For GPU-offloaded models, CPU threads affect prompt processing + sampling.
    ctx_params.n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    ctx_params.type_k = kv_type_k;
    ctx_params.type_v = kv_type_v;
    if (flash_attn) ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // Self-spec and layer-skip need n_seq_max=2 for seq branching
    // cached_spec does not need seq branching - uses state save/restore instead
    // (seq_cp breaks on hybrid models with recurrent layers like qwen35moe 397B)
    if (self_spec_n > 0 || self_spec_skip > 0) ctx_params.n_seq_max = 2;

    // Backend sampler: run sampling on the GPU as part of the compute graph.
    // Eliminates ~3.6ms/token GPU->CPU logits transfer (151Kx4B = 600KB per token).
    // llama_sampler_sample() returns the GPU-selected token instantly.
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (temperature > 0.0f) {
        // Penalty samplers (must be before filtering - they modify logits)
        // Matches Ollama defaults: presence_penalty=1.5
        if (g_presence_penalty != 0.0f || g_frequency_penalty != 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                256,                   // last_n tokens to penalize
                1.0f,                  // repeat penalty (1.0 = disabled)
                g_frequency_penalty,   // frequency penalty (0.0 = disabled)
                g_presence_penalty     // presence penalty (1.5 = Ollama default)
            ));
        }
        // Filtering: top-k -> top-p -> min-p -> temperature -> dist
        if (g_top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(g_top_k));
        }
        if (g_top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(g_top_p, 1));
        }
        if (g_min_p > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(g_min_p, 1));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }
    // Enable GPU backend sampling - sampling runs as part of the CUDA graph,
    // no logits transfer needed. Works for both greedy and distribution sampling.
    // Backend sampler graph reserves n_batch x n_vocab x ops ~ 1.2GB extra VRAM.
    // Portfolio sets backend_sampler=true for TINY tier (model < 25% VRAM).
    // Disabled for --draft mode (batched decode needs multiple output tokens per sequence).
    bool use_backend_sampler = (portfolio.backend_sampler && draft_model_path.empty() && self_spec_skip == 0);
    llama_sampler_seq_config sampler_cfg = { 0, smpl };
    if (use_backend_sampler) {
        ctx_params.samplers   = &sampler_cfg;
        ctx_params.n_samplers = 1;
    }

    // MoE optimizer: plan-driven or CLI-driven (log-confidence, overrides, profiling)
    alpha_moe_optimizer * moe_opt = nullptr;
    if (portfolio.need_moe_opt || expert_cache_mb > 0 || log_confidence || log_gate_entropy || !expert_overrides.empty() || !profile_freq_path.empty()) {
        alpha_moe_optimizer_config moe_cfg = {};
        moe_cfg.vram_budget_mb = expert_cache_mb;
        moe_cfg.prefetch_next_layer = false;
        moe_cfg.prefetch_depth = 0;
        moe_cfg.expert_sticky = expert_sticky;
        moe_cfg.expert_layer_bias = expert_layer_bias;
        moe_cfg.expert_groups_path = expert_groups_path.empty() ? nullptr : expert_groups_path.c_str();
        moe_cfg.split_group = split_group;
        moe_cfg.no_barrier = no_barrier;
        moe_cfg.flat_cache = g_flat_cache;
        moe_cfg.force_cache = g_force_cache;
        moe_cfg.debug = g_alpha_debug;
        moe_cfg.enable_profiling = !profile_freq_path.empty() || (portfolio.warmup == ALPHA_WARMUP_CLASSIFY);
        moe_opt = alpha_moe_optimizer_init(model, nullptr, moe_cfg);
        // Callback setup driven by execution plan
        if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) && portfolio.callback != ALPHA_CB_NONE) {
            ctx_params.cb_eval = alpha_moe_optimizer_get_eval_callback();
            ctx_params.cb_eval_user_data = alpha_moe_optimizer_get_eval_data(moe_opt);
            // REACTIVE: active from start. LUT_DISABLED: start disabled (activated after warmup).
            bool active = (portfolio.callback == ALPHA_CB_REACTIVE);
            alpha_moe_optimizer_set_callback_active(moe_opt, active);
        }
    }

    // Load prompt-type profiles - model directory only (no shared /tmp)
    // Each model's profiles live next to its GGUF files, preventing cross-model contamination.
    if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
        std::string mdir = path.substr(0, path.find_last_of("/\\"));
        alpha_moe_optimizer_load_style_profiles(moe_opt, mdir.c_str());
    }

    // Apply top-k expert reduction before context creation
    // (warmup graph build caches tensor shapes - must use correct k from the start)
    if (topk_experts > 0) {
        if (moe_opt) {
            alpha_moe_optimizer_set_topk(moe_opt, topk_experts);
        }
#ifndef _WIN32
        else {
            // On Linux, set global directly (same .so address space)
            extern thread_local int g_alpha_topk_override;
            g_alpha_topk_override = topk_experts;
        }
#endif
        ALPHA_LOG_DBG("[alphachat] --top-k-experts %d\n", topk_experts);
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 3;
    }

    if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
        alpha_moe_optimizer_set_context(moe_opt, ctx);

        // Load frequency profile for cache pre-population (before alloc_cache)
        if (!auto_freq_path.empty()) {
            FILE * fp = fopen(auto_freq_path.c_str(), "r");
            if (fp) {
                fseek(fp, 0, SEEK_END);
                long fsize = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                std::string json_str(fsize, '\0');
                fread(&json_str[0], 1, fsize, fp);
                fclose(fp);

                // Simple JSON parser for frequency profile
                // Format: {"layers": [{"layer": N, "top_experts": [id, ...]}, ...]}
                int n_loaded = 0;
                size_t pos = 0;
                while ((pos = json_str.find("\"te\"", pos)) != std::string::npos) {
                    // Find the layer number before this
                    size_t layer_pos = json_str.rfind("\"l\"", pos);
                    int layer_id = -1;
                    if (layer_pos != std::string::npos) {
                        size_t colon = json_str.find(':', layer_pos + 3);
                        if (colon != std::string::npos) {
                            layer_id = atoi(json_str.c_str() + colon + 1);
                        }
                    }

                    // Parse the expert ID array
                    size_t arr_start = json_str.find('[', pos);
                    size_t arr_end = json_str.find(']', arr_start);
                    if (arr_start != std::string::npos && arr_end != std::string::npos && layer_id >= 0) {
                        std::vector<int> expert_ids;
                        size_t p = arr_start + 1;
                        while (p < arr_end) {
                            while (p < arr_end && (json_str[p] == ' ' || json_str[p] == ',')) p++;
                            if (p >= arr_end) break;
                            int eid = atoi(json_str.c_str() + p);
                            expert_ids.push_back(eid);
                            while (p < arr_end && json_str[p] != ',') p++;
                        }
                        if (!expert_ids.empty()) {
                            alpha_moe_optimizer_set_preload_experts(moe_opt, layer_id,
                                expert_ids.data(), (int)expert_ids.size());
                            n_loaded++;
                        }
                    }
                    pos = (arr_end != std::string::npos) ? arr_end : pos + 1;
                }
                ALPHA_LOG_DBG("[FREQ] Loaded frequency profile: %d layers from %s\n",
                        n_loaded, auto_freq_path.c_str());
            } else {
                ALPHA_LOG_DBG("[FREQ] WARNING: cannot open frequency profile: %s\n",
                        auto_freq_path.c_str());
            }
        }

        // Allocate expert cache after context creation - compute buffer already allocated
        alpha_moe_optimizer_alloc_cache(moe_opt);

        // LUT init must happen while cache_enabled=true (before disable_cache)
        if (portfolio.lut_init) {
            alpha_moe_optimizer_init_spec_lut(moe_opt);
        }

        // Plan-driven: disable cache for prompt processing (cpu-moe path).
        // enter_draft_mode will swap back to GPU cache when draft phase begins.
        if (portfolio.disable_cache_for_prompt) {
            alpha_moe_optimizer_disable_cache(moe_opt);
            ALPHA_LOG_DBG("[PLAN] cache disabled for prompt (cpu-moe path)\n");
        }

        // Load mixed expert clusters if specified
        if (!mixed_cluster_path.empty()) {
            if (alpha_moe_optimizer_load_mixed_experts(moe_opt, mixed_cluster_path.c_str())) {
                ALPHA_LOG_DBG("[alphachat] Mixed expert clusters loaded from %s\n", mixed_cluster_path.c_str());
            } else {
                ALPHA_LOG_DBG("[alphachat] WARNING: Failed to load mixed clusters\n");
            }
        }

        // Apply expert overrides from CLI
        // Disable CUDA graphs only when overrides are active
        // (in-place argsort modification not visible to graph replay)
        if (!expert_overrides.empty()) {
            #ifdef _WIN32
            _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
            #else
            setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
            #endif
        }
        for (auto & [layer, ids] : expert_overrides) {
            if (alpha_lora_mix_set_layer_experts(moe_opt, layer, ids.data(), nullptr, (int)ids.size()) == 0) {
                ALPHA_LOG_DBG("[override] layer %d: %d experts overridden\n", layer, (int)ids.size());
            } else {
                ALPHA_LOG_DBG("[override] layer %d: FAILED (invalid layer or expert count)\n", layer);
            }
        }
    }

    // Dense pipeline: multi-pass for models that don't fit in GPU
    alpha_dense_pipeline * dense_pipe = nullptr;
    if (!hw_plan.is_moe && !fits_in_gpu && hw_plan.gpu_vram_mb > 0) {
        dense_pipe = alpha_dense_pipeline_init(model, hw_plan.gpu_vram_mb);
    }

    // Speculative execution state (if enabled)
    alpha_speculative_state spec_state = {};
    if (speculative_mode && moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
        alpha_speculative_init(&spec_state, moe_opt);
        if (spec_state.initialized) {
            ALPHA_LOG_DBG("[alphachat] speculative execution enabled\n");
        } else {
            ALPHA_LOG_DBG("[alphachat] speculative init failed — falling back to reactive\n");
            speculative_mode = false;
        }
    }

    // Self-speculative state (if enabled)
    alpha_self_spec_state self_spec_state = {};
    bool self_spec_mode = false;
    if (self_spec_n > 0 && moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
        alpha_moe_optimizer_set_draft_expert_count(moe_opt, self_spec_experts);
        alpha_self_spec_init(&self_spec_state, moe_opt, self_spec_n);
        self_spec_mode = self_spec_state.initialized;
        if (self_spec_mode) {
            ALPHA_LOG_DBG("[SELF-SPEC] Mode %s: %d drafts, %d experts (save/restore, all memory types)\n",
                    self_spec_experts > 0 ? "A (expert reduce)" : "C (backbone)",
                    self_spec_n, self_spec_experts > 0 ? self_spec_experts : 0);
        }
    }

    // Cached-expert self-spec state (if enabled)
    alpha_cached_spec_state cached_spec_state = {};
    bool cached_spec_mode = false;
    alpha_twotier_state twotier_state = {};
    bool twotier_mode = false;
    // Cached-spec init (LUT already initialized above, before disable_cache)
    if (cached_spec_n > 0 && moe_opt && alpha_moe_optimizer_is_moe(moe_opt) && expert_cache_mb > 0) {
        if (tier1_cycles > 0) {
            // Two-tier mode: draft N*M tokens, tier1 verify (LUT), tier2 verify (cpu-moe)
            alpha_twotier_init(&twotier_state, cached_spec_n, tier1_cycles);
            twotier_mode = twotier_state.initialized;
            if (twotier_mode) {
                ALPHA_LOG_DBG("[TWOTIER] M=%d, N=%d tier1 cycles (%d draft/round), GPU LUT + cpu-moe verify\n",
                        cached_spec_n, tier1_cycles, cached_spec_n * tier1_cycles);
            }
        } else {
            alpha_cached_spec_init(&cached_spec_state, cached_spec_n);
            cached_spec_mode = cached_spec_state.initialized;
            if (cached_spec_mode) {
                ALPHA_LOG_DBG("[CACHED-SPEC] M=%d draft tokens, GPU LUT (zero callback)\n",
                        cached_spec_n);
                // cpu-moe verify: when --cpu-moe is set and M > 1, verify batch uses cpu-moe.
                // But skip cpu-moe verify when profiles are loaded - LUT verify is faster
                // and profiles ensure cache has good experts (no callback needed).
                if (g_cpu_moe && cached_spec_n > 1 && moe_opt && !alpha_moe_optimizer_has_profiles()) {
                    alpha_moe_optimizer_set_cpu_moe_verify(moe_opt, true);
                }
            }
        }
    }

    // Sampler already created before context (backend sampler for GPU-side argmax)
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Expert profiling output
    FILE * profile_experts_fp = nullptr;
    if (!profile_experts_path.empty()) {
        // Append if file exists (for multi-prompt profiling), write header if new
#ifdef _WIN32
        bool file_exists = (_access(profile_experts_path.c_str(), 0) == 0);
#else
        bool file_exists = (access(profile_experts_path.c_str(), F_OK) == 0);
#endif
        profile_experts_fp = fopen(profile_experts_path.c_str(), file_exists ? "ab" : "wb");
        if (profile_experts_fp) {
            if (!file_exists) {
                // Header: n_moe_layers (int32), n_expert_used (int32)
                int32_t hdr[2] = { alpha_lora_mix_get_argsort_count() > 0 ? alpha_lora_mix_get_argsort_count() : 43, 6 };
                fwrite(hdr, sizeof(int32_t), 2, profile_experts_fp);
            }
            ALPHA_LOG_DBG("[PROFILE] %s expert routing to %s\n",
                    file_exists ? "Appending" : "Writing", profile_experts_path.c_str());
        }
    }

    // Interactive REPL or single-shot
    if (interactive) {
        fprintf(stderr, "Model loaded. Type your message (or /bye to quit).\n\n");
    }

    bool first_turn = true;
    do {
        // Get prompt for this turn
        std::string turn_prompt;
        if (interactive && !first_turn) {
            printf(">>> ");
            fflush(stdout);
            if (!std::getline(std::cin, turn_prompt) || turn_prompt == "/bye" || turn_prompt == "/exit") {
                break;
            }
            if (turn_prompt.empty()) continue;
        } else if (interactive && first_turn) {
            printf(">>> ");
            fflush(stdout);
            if (!std::getline(std::cin, turn_prompt) || turn_prompt == "/bye" || turn_prompt == "/exit") {
                break;
            }
            if (turn_prompt.empty()) continue;
        } else {
            turn_prompt = prompt;
        }

        // Clear KV cache for new turn (keeps model loaded)
        if (!first_turn) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }
        first_turn = false;

        // Apply chat template to wrap user message properly
        std::string formatted_prompt;
        {
            const char * model_tmpl = llama_model_chat_template(model, NULL);
            const char * tmpl_to_use = model_tmpl;

            // If no template in GGUF, detect from model description
            if (!tmpl_to_use) {
                char desc_buf[256] = {};
                llama_model_desc(model, desc_buf, sizeof(desc_buf));
                std::string desc(desc_buf);
                for (auto & c : desc) c = tolower(c);

                if (desc.find("gemma") != std::string::npos) {
                    tmpl_to_use = "gemma";
                } else if (desc.find("llama") != std::string::npos) {
                    tmpl_to_use = "llama3";
                } else if (desc.find("mistral") != std::string::npos ||
                           desc.find("mixtral") != std::string::npos) {
                    tmpl_to_use = "mistral-v7";
                } else if (desc.find("deepseek") != std::string::npos) {
                    tmpl_to_use = "deepseek2";
                } else {
                    tmpl_to_use = "chatml";
                }
            }

            // Build message array
            llama_chat_message msgs[1] = {
                { "user", turn_prompt.c_str() }
            };

            int32_t needed = llama_chat_apply_template(tmpl_to_use, msgs, 1, true, nullptr, 0);
            if (needed > 0) {
                formatted_prompt.resize(needed);
                llama_chat_apply_template(tmpl_to_use, msgs, 1, true,
                    &formatted_prompt[0], formatted_prompt.size());
            } else {
                formatted_prompt = turn_prompt;
            }

            // --no-think: suppress reasoning/thinking blocks
            if (no_think) {
                // Qwen3/3.5: add /no_think marker before assistant turn
                // DeepSeek: append empty think block
                char desc_buf2[256] = {};
                llama_model_desc(model, desc_buf2, sizeof(desc_buf2));
                std::string desc2(desc_buf2);
                for (auto & c : desc2) c = tolower(c);
                if (desc2.find("qwen") != std::string::npos) {
                    // Qwen3/3.5: insert /no_think system message before assistant turn
                    size_t asst_pos = formatted_prompt.rfind("<|im_start|>assistant");
                    if (asst_pos != std::string::npos) {
                        formatted_prompt.insert(asst_pos, "<|im_start|>system\n/no_think<|im_end|>\n");
                    }
                    // Always append empty think block as fallback.
                    // The /no_think system message alone doesn't reliably suppress thinking
                    // on all Qwen variants (35B MoE needs the think block too).
                    formatted_prompt += "<think>\n\n</think>\n\n";
                } else {
                    // DeepSeek and others: append empty think block
                    formatted_prompt += "<think>\n\n</think>\n\n";
                }
            }
        }

        // Tokenize the formatted prompt
        int n_toks = -llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), NULL, 0, true, true);
        std::vector<llama_token> tokens(n_toks);
        llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), tokens.data(), tokens.size(), true, true);

        // Prompt processing (plan-driven)
        bool need_warmup = (portfolio.warmup != ALPHA_WARMUP_NONE) && moe_opt;
        bool disable_cache_prompt = portfolio.disable_cache_for_prompt && moe_opt;
        if (disable_cache_prompt) {
            // cpu-moe path: disable cache so model uses CPU expert tensors directly
            alpha_moe_optimizer_disable_cache(moe_opt);
        }
        // LUT_DISABLED callback: activate during prompt for reactive expert loading (122B path).
        // After warmup, enter_draft_mode() will disable it for LUT generation.
        if (moe_opt && portfolio.callback == ALPHA_CB_LUT_DISABLED && !disable_cache_prompt) {
            alpha_moe_optimizer_set_callback_active(moe_opt, true);
        }
        if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) && !disable_cache_prompt) {
            int rc = alpha_moe_optimizer_process_prompt_batched(
                moe_opt, ctx, tokens.data(), tokens.size());
            if (rc != 0) {
                fprintf(stderr, "Error: prompt processing failed\n");
                if (!interactive) break;
                continue;
            }
        } else {
            llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "Error: prompt processing failed\n");
                if (!interactive) break;
                continue;
            }
        }
        // Warmup (plan-driven)
        if (need_warmup) {
            llama_token warmup_tok = llama_sampler_sample(smpl, ctx, -1);
            g_warmup_tokens.clear();

            if (!llama_vocab_is_eog(vocab, warmup_tok)) {
                g_warmup_tokens.push_back(warmup_tok);
                llama_batch warmup_batch = llama_batch_get_one(&warmup_tok, 1);
                llama_decode(ctx, warmup_batch);
            }

            // Apply portfolio warmup (classify/smart -> enable cache -> reinstall -> draft mode)
            std::string model_dir_str = fs::path(path).parent_path().string();
            alpha_moe_optimizer_apply_warmup(moe_opt, ctx, &portfolio, model_dir_str.c_str());

            // Set readback_interval based on matched profile type.
            // Short-output profiles: no refresh needed (cache is fine for a few tokens).
            // Long-output profiles: periodic refresh prevents garbled spelling from cache drift.
            if (portfolio.readback_interval > 0) {
                const char * pname = alpha_moe_optimizer_get_profile_name();
                if (pname) {
                    if (strcmp(pname, "factual") == 0 || strcmp(pname, "code") == 0 ||
                        strcmp(pname, "translate") == 0) {
                        portfolio.readback_interval = 0;  // short answers, no refresh
                    } else if (strcmp(pname, "chat") == 0 || strcmp(pname, "instruct") == 0) {
                        portfolio.readback_interval = 30;
                    } else if (strcmp(pname, "creative") == 0 || strcmp(pname, "knowledge") == 0) {
                        portfolio.readback_interval = 15;
                    } else {
                        // reasoning, analysis, summarize, etc. - moderate refresh
                        portfolio.readback_interval = 20;
                    }
                }
            }

            if (!llama_vocab_is_eog(vocab, warmup_tok)) {
                // Sample after warmup (logits unchanged - warmup only DMAs expert weights)
                warmup_tok = llama_sampler_sample(smpl, ctx, -1);
            }
            g_warmup_last_tok = warmup_tok;
            g_warmup_eog = llama_vocab_is_eog(vocab, warmup_tok);
        }

        // Generate response
        // Buffer stdout for non-interactive mode to avoid per-token fflush syscall overhead
        if (!interactive) setvbuf(stdout, nullptr, _IOFBF, 8192);
        const auto t_start = ggml_time_us();
        int n_decoded = 0;

        // JSON output: buffer all generated text and per-token info
        std::string json_full_text;
        struct json_token_info { int id; std::string text; float prob; };
        std::vector<json_token_info> json_tokens;

        // Draft Model Speculative Decoding
        // Isolated path: uses upstream common_speculative_* API with a separate draft model.
        // No MoE optimizer, no expert cache - pure two-model speculative decoding.
        if (!draft_model_path.empty()) {
            std::string draft_path = resolve_model(draft_model_path);
            if (draft_path.empty()) {
                ALPHA_LOG_DBG("[draft] Draft model not found: %s\n", draft_model_path.c_str());
                goto gen_done;
            }

            // Load draft model (separate from target)
            llama_model_params dparams = llama_model_default_params();
            dparams.n_gpu_layers = n_gpu_layers;
            llama_model * draft_model = llama_model_load_from_file(draft_path.c_str(), dparams);
            if (!draft_model) {
                ALPHA_LOG_DBG("[draft] Failed to load draft model\n");
                goto gen_done;
            }

            // Set up speculative params
            common_params_speculative spec_params = {};
            spec_params.type = COMMON_SPECULATIVE_TYPE_DRAFT;
            spec_params.n_max = draft_n;
            spec_params.n_min = 1;
            spec_params.p_min = 0.5f;
            spec_params.mparams_dft.path = draft_path;  // needed by common_speculative_init
            spec_params.model_dft = draft_model;
            spec_params.cparams_dft = llama_context_default_params();
            spec_params.cparams_dft.n_ctx = ctx_params.n_ctx;
            spec_params.cparams_dft.n_batch = ctx_params.n_batch;
            spec_params.cparams_dft.flash_attn_type = ctx_params.flash_attn_type;
            spec_params.cparams_dft.n_threads = ctx_params.n_threads;

            common_speculative * spec = common_speculative_init(spec_params, ctx);
            if (!spec) {
                ALPHA_LOG_DBG("[draft] Failed to init speculative context (incompatible vocab?)\n");
                llama_model_free(draft_model);
                goto gen_done;
            }

            ALPHA_LOG_DBG("[draft] Speculative decoding: draft=%s, n=%d\n",
                    draft_model_path.c_str(), draft_n);

            // Build prompt token list for speculative API
            llama_tokens prompt_toks(tokens.begin(), tokens.end());
            common_speculative_begin(spec, prompt_toks);

            // Sample first token from prompt logits
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            int n_accept_total = 0, n_draft_total = 0;
            llama_memory_t draft_mem = llama_get_memory(ctx);

            while (!llama_vocab_is_eog(vocab, tok) && n_decoded < n_predict) {
                // Output accepted token
                char buf[128];
                int nb = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
                if (nb > 0) {
                    if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({tok, std::string(buf, nb), 0.0f}); }
                    else { printf("%.*s", nb, buf); if (interactive) fflush(stdout); }
                }
                n_decoded++;
                prompt_toks.push_back(tok);

                // Draft N tokens using draft model
                llama_tokens draft = common_speculative_draft(spec, spec_params, prompt_toks, tok);

                if (draft.empty()) {
                    // No draft - single-token decode
                    llama_batch batch = llama_batch_get_one(&tok, 1);
                    if (llama_decode(ctx, batch) != 0) break;
                    tok = llama_sampler_sample(smpl, ctx, -1);
                    continue;
                }

                n_draft_total += (int)draft.size();

                // Batched decode: tok + all draft tokens in one forward pass
                // Build batch with output flags on each position for sampling
                llama_batch batch = llama_batch_init((int)draft.size() + 1, 0, 1);
                llama_pos pos_base = draft_mem ? llama_memory_seq_pos_max(draft_mem, 0) + 1 : (llama_pos)prompt_toks.size();

                // Add current token
                common_batch_add(batch, tok, pos_base, {0}, true);
                // Add draft tokens
                for (size_t i = 0; i < draft.size(); i++) {
                    common_batch_add(batch, draft[i], pos_base + 1 + (int)i, {0}, true);
                }

                if (llama_decode(ctx, batch) != 0) { llama_batch_free(batch); break; }

                // Verify: sample from each position, compare with draft
                int n_accepted = 0;
                for (size_t i = 0; i < draft.size(); i++) {
                    llama_token target_tok = llama_sampler_sample(smpl, ctx, (int)i);
                    if (target_tok == draft[i]) {
                        n_accepted++;
                        nb = llama_token_to_piece(vocab, draft[i], buf, sizeof(buf), 0, true);
                        if (nb > 0) {
                            if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({draft[i], std::string(buf, nb), 0.0f}); }
                            else { printf("%.*s", nb, buf); if (interactive) fflush(stdout); }
                        }
                        n_decoded++;
                        prompt_toks.push_back(draft[i]);
                    } else {
                        // Mismatch: remove rejected positions from KV
                        if (draft_mem) {
                            llama_pos rm_start = pos_base + 1 + (int)i;
                            llama_memory_seq_rm(draft_mem, 0, rm_start, -1);
                        }
                        tok = target_tok;
                        break;
                    }
                }

                if (n_accepted == (int)draft.size()) {
                    // All accepted - sample bonus token from last position
                    tok = llama_sampler_sample(smpl, ctx, (int)draft.size());
                }

                n_accept_total += n_accepted;
                common_speculative_accept(spec, n_accepted);
                llama_batch_free(batch);
            }

            ALPHA_LOG_DBG("\n[draft] %d drafted, %d accepted (%.1f%% acceptance)\n",
                    n_draft_total, n_accept_total,
                    n_draft_total > 0 ? 100.0 * n_accept_total / n_draft_total : 0.0);

            common_speculative_free(spec);
            llama_model_free(draft_model);
            goto gen_done;
        }

        // Self-Speculative Layer-Skip Decoding
        // Same model for draft and verify. Draft skips last N layers (faster).
        // Verify uses all layers in a single batched forward pass.
        if (self_spec_skip > 0) {
            const int n_layer = llama_model_n_layer(model);
            const int draft_layers = n_layer - self_spec_skip;
            if (draft_layers < 2) {
                ALPHA_LOG_DBG("[self-spec] Error: --self-spec-skip %d leaves only %d layers (need >=2)\n",
                        self_spec_skip, draft_layers);
                goto gen_done;
            }

            ALPHA_LOG_DBG("[self-spec] Layer-skip mode: %d/%d layers for draft, M=%d\n",
                    draft_layers, n_layer, self_spec_m);

            // Draft sampler: greedy argmax to avoid corrupting main sampler state
            llama_sampler * draft_smpl = llama_sampler_init_greedy();

            // JSON output buffers
            std::string json_full_text;
            struct json_token_info { int id; std::string text; float prob; };
            std::vector<json_token_info> json_tokens;

            llama_memory_t ss_mem = llama_get_memory(ctx);

            // Sample first token from prompt logits
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            int n_decoded = 0;
            int n_draft_total = 0;
            int n_accept_total = 0;

            const llama_vocab * vocab = llama_model_get_vocab(model);

            while (!llama_vocab_is_eog(vocab, tok) && n_decoded < n_predict) {
                // Output accepted token
                char buf[128];
                int nb = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
                if (nb > 0) {
                    if (json_output) {
                        json_full_text.append(buf, nb);
                        json_tokens.push_back({tok, std::string(buf, nb), 0.0f});
                    } else {
                        printf("%.*s", nb, buf);
                        if (interactive) fflush(stdout);
                    }
                }
                n_decoded++;

                // Decode current token into seq 0 KV first
                {
                    llama_batch tok_batch = llama_batch_init(1, 0, 1);
                    llama_pos cur_pos = ss_mem ? llama_memory_seq_pos_max(ss_mem, 0) + 1
                                               : (llama_pos)(n_decoded - 1);
                    common_batch_add(tok_batch, tok, cur_pos, {0}, false);
                    if (llama_decode(ctx, tok_batch) != 0) { llama_batch_free(tok_batch); break; }
                    llama_batch_free(tok_batch);
                }

                // Draft phase: skip last N layers
                alpha_pipeline_set_layer_range(0, draft_layers);

                std::vector<llama_token> draft_tokens;
                llama_token draft_tok = tok;
                llama_pos draft_pos = ss_mem ? llama_memory_seq_pos_max(ss_mem, 0) + 1
                                             : (llama_pos)(n_decoded);

                // Branch KV cache: copy seq 0 -> seq 1 for draft
                if (ss_mem) llama_memory_seq_cp(ss_mem, 0, 1, 0, -1);

                for (int d = 0; d < self_spec_m; d++) {
                    llama_batch dbatch = llama_batch_init(1, 0, 1);
                    common_batch_add(dbatch, draft_tok, draft_pos + d, {1}, true);
                    if (llama_decode(ctx, dbatch) != 0) { llama_batch_free(dbatch); break; }
                    draft_tok = llama_sampler_sample(draft_smpl, ctx, -1);
                    draft_tokens.push_back(draft_tok);
                    llama_batch_free(dbatch);
                }
                n_draft_total += (int)draft_tokens.size();

                // Clean up draft KV (seq 1)
                if (ss_mem) llama_memory_seq_rm(ss_mem, 1, 0, -1);

                // Clear layer skip for verify (all layers)
                alpha_pipeline_clear_layer_range();

                if (draft_tokens.empty()) {
                    // No draft tokens - single-token decode
                    llama_batch batch = llama_batch_get_one(&tok, 1);
                    if (llama_decode(ctx, batch) != 0) break;
                    tok = llama_sampler_sample(smpl, ctx, -1);
                    continue;
                }

                // Verify phase: all layers, batched
                llama_batch vbatch = llama_batch_init((int)draft_tokens.size() + 1, 0, 1);
                llama_pos vpos = draft_pos;

                // Add current token + all draft tokens
                common_batch_add(vbatch, tok, vpos, {0}, true);
                for (size_t i = 0; i < draft_tokens.size(); i++) {
                    common_batch_add(vbatch, draft_tokens[i], vpos + 1 + (int)i, {0}, true);
                }

                if (llama_decode(ctx, vbatch) != 0) { llama_batch_free(vbatch); break; }

                // Verify: sample from each position, compare with draft
                int n_accepted = 0;
                for (size_t i = 0; i < draft_tokens.size(); i++) {
                    llama_token target_tok = llama_sampler_sample(smpl, ctx, (int)i);
                    if (target_tok == draft_tokens[i]) {
                        n_accepted++;
                        nb = llama_token_to_piece(vocab, draft_tokens[i], buf, sizeof(buf), 0, true);
                        if (nb > 0) {
                            if (json_output) {
                                json_full_text.append(buf, nb);
                                json_tokens.push_back({draft_tokens[i], std::string(buf, nb), 0.0f});
                            } else {
                                printf("%.*s", nb, buf);
                                if (interactive) fflush(stdout);
                            }
                        }
                        n_decoded++;
                    } else {
                        // Mismatch: remove rejected positions from KV
                        if (ss_mem) {
                            llama_pos rm_start = vpos + 1 + (int)i;
                            llama_memory_seq_rm(ss_mem, 0, rm_start, -1);
                        }
                        tok = target_tok;
                        break;
                    }
                }

                if (n_accepted == (int)draft_tokens.size()) {
                    // All accepted - sample bonus token from last position
                    tok = llama_sampler_sample(smpl, ctx, (int)draft_tokens.size());
                }

                n_accept_total += n_accepted;
                llama_batch_free(vbatch);
            }

            ALPHA_LOG_DBG("\n[self-spec] %d drafted, %d accepted (%.1f%% acceptance)\n",
                    n_draft_total, n_accept_total,
                    n_draft_total > 0 ? 100.0 * n_accept_total / n_draft_total : 0.0);

            llama_sampler_free(draft_smpl);

            // JSON output (simple escape for self-spec)
            if (json_output) {
                std::string escaped;
                for (char c : json_full_text) {
                    if (c == '"') escaped += "\\\"";
                    else if (c == '\\') escaped += "\\\\";
                    else if (c == '\n') escaped += "\\n";
                    else if (c == '\r') escaped += "\\r";
                    else if (c == '\t') escaped += "\\t";
                    else escaped += c;
                }
                printf("{\"content\":\"%s\",\"eval_count\":%d,\"tokens_per_sec\":0.0}\n",
                       escaped.c_str(), n_decoded);
            }

            goto gen_done;
        }

        // Two-Tier Speculative Generation
        if (twotier_mode && twotier_state.initialized) {
            llama_memory_t tt_mem = llama_get_memory(ctx);
            llama_pos tt_n_past = tt_mem ? llama_memory_seq_pos_max(tt_mem, 0) + 1 : (llama_pos)tokens.size();

            // Sample first token from prompt logits
            llama_token last_tok = llama_sampler_sample(smpl, ctx, -1);
            if (!llama_vocab_is_eog(vocab, last_tok)) {
                char buf[128];
                int nb = llama_token_to_piece(vocab, last_tok, buf, sizeof(buf), 0, true);
                if (nb > 0) {
                    if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({last_tok, std::string(buf, nb), 0.0f}); }
                    else { printf("%.*s", nb, buf); fflush(stdout); }
                }
                n_decoded++;

                while (n_decoded < n_predict) {
                    std::vector<llama_token> accepted;
                    int n = alpha_twotier_generate(&twotier_state, moe_opt, ctx, smpl, vocab,
                                                    last_tok, tt_n_past, accepted);
                    if (n <= 0) break;

                    bool eog = false;
                    for (auto tok : accepted) {
                        if (llama_vocab_is_eog(vocab, tok)) { eog = true; break; }
                        char buf2[128];
                        int nb2 = llama_token_to_piece(vocab, tok, buf2, sizeof(buf2), 0, true);
                        if (nb2 > 0) {
                            if (json_output) { json_full_text.append(buf2, nb2); json_tokens.push_back({tok, std::string(buf2, nb2), 0.0f}); }
                            else { printf("%.*s", nb2, buf2); fflush(stdout); }
                        }
                        n_decoded++;
                    }
                    if (eog) break;

                    last_tok = accepted.back();
                    tt_n_past += n;
                }
            }

            alpha_twotier_print_stats(&twotier_state);
            goto gen_done;
        }

        // Cached-Expert Self-Speculative Generation
        if (cached_spec_mode && cached_spec_state.initialized) {
            // LUT-direct mode (--cached-spec 1): permanent draft with per-token cache refresh.
            // No state save/restore, no verification.
            bool lut_direct = (cached_spec_n == 1);

            if (lut_direct) {
                // Enter draft mode permanently (GPU LUT + adaptive renorm, no callback)
                // If mixed clusters loaded, use mixed draft mode (swaps expert tensors)
                if (alpha_moe_optimizer_has_mixed_experts(moe_opt)) {
                    alpha_moe_optimizer_enter_mixed_draft_mode(moe_opt);
                } else {
                    alpha_moe_optimizer_enter_draft_mode(moe_opt);
                }
                // Re-reserve graph after entering draft mode - expert tensors may have
                // changed from CPU (cpu-moe) to GPU (cache). Without this, the CUDA
                // compute buffer is too small for GPU MoE operations.
                llama_graph_reserve(ctx, 1, 1, 1);

                // Output warmup tokens first (generated by cpu-moe, correct)
                for (auto wtok : g_warmup_tokens) {
                    if (llama_vocab_is_eog(vocab, wtok)) break;
                    char buf[128];
                    int nb = llama_token_to_piece(vocab, wtok, buf, sizeof(buf), 0, true);
                    if (nb > 0) {
                        if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({wtok, std::string(buf, nb), 0.0f}); }
                        else { printf("%.*s", nb, buf); if (interactive) fflush(stdout); }
                    }
                    n_decoded++;
                }

                // Continue from last warmup token (or sample fresh)
                llama_token tok = g_warmup_tokens.empty()
                    ? llama_sampler_sample(smpl, ctx, -1) : g_warmup_last_tok;

                // Three-layer quality protection state (thresholds from portfolio - per-tier, orthogonal)
                int low_conf_count = 0;
                std::vector<llama_token> lut_history;
                llama_token prev_tok = tok;
                int eos_verify_count = 0;

                // Helper lambda: quality resync - KV rewind + re-decode with correct experts.
                // Rewinds the LUT-decoded entry (wrong experts), re-decodes prev_tok with
                // reactive callback or cpu-moe (correct experts), re-samples tok.
                auto quality_resync = [&](const char * reason) {
                    ALPHA_LOG_DBG("[%s] resync\n", reason);
                    alpha_moe_optimizer_exit_draft_mode(moe_opt);

                    // KV rewind: remove the LUT-decoded entry (had zero-expert data)
                    llama_memory_t mem = llama_get_memory(ctx);
                    if (mem) {
                        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
                        if (pos_max >= 0) {
                            llama_memory_seq_rm(mem, 0, pos_max, -1);
                        }
                    }

                    if (g_cpu_moe) {
                        // CPU-MOE: disable cache, re-decode prev_tok with all CPU experts
                        alpha_moe_optimizer_disable_cache(moe_opt);
                        llama_batch vb = llama_batch_get_one(&prev_tok, 1);
                        llama_decode(ctx, vb);
                        alpha_moe_optimizer_warmup_1token_smart(moe_opt);
                        tok = llama_sampler_sample(smpl, ctx, -1);
                        alpha_moe_optimizer_enable_cache(moe_opt);
                    } else {
                        // Expert-cache: enable reactive callback, re-decode prev_tok
                        alpha_moe_optimizer_set_callback_active(moe_opt, true);
                        llama_batch vb = llama_batch_get_one(&prev_tok, 1);
                        llama_decode(ctx, vb);
                        tok = llama_sampler_sample(smpl, ctx, -1);
                        // Refresh cache from correct routing
                        alpha_moe_optimizer_reload_from_argsort(moe_opt);
                        alpha_moe_optimizer_set_callback_active(moe_opt, false);
                    }
                    alpha_moe_optimizer_reinstall_callback(moe_opt);
                    alpha_moe_optimizer_enter_draft_mode(moe_opt);
                    llama_graph_reserve(ctx, 1, 1, 1);
                };

                while (n_decoded < n_predict) {
                    // EOS check - plan-driven: eos_verify=true -> resync with cpu-moe to confirm
                    // For CPU_MOE tier: eos_verify=false (EOS is genuine on that tier)
                    if (llama_vocab_is_eog(vocab, tok)) {
                        if (portfolio.eos_verify && eos_verify_count < portfolio.eos_max_retries) {
                            eos_verify_count++;
                            quality_resync("EOS-VERIFY");
                            if (!llama_vocab_is_eog(vocab, tok)) {
                                ALPHA_LOG_DBG("[EOS-VERIFY] False EOS corrected (retry %d/%d)\n",
                                        eos_verify_count, portfolio.eos_max_retries);
                                continue;
                            }
                        }
                        break;
                    }

                    // Output token
                    char buf[128];
                    int nb = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
                    if (nb > 0) {
                        if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({tok, std::string(buf, nb), 0.0f}); }
                        else { printf("%.*s", nb, buf); if (interactive) fflush(stdout); }
                    }
                    n_decoded++;
                    lut_history.push_back(tok);
                    prev_tok = tok;

                    // Decode next with GPU LUT
                    // Pass sampler for temperature support; nullptr = greedy argmax (fast)
                    llama_sampler * lut_smpl = (temperature > 0.0f) ? smpl : nullptr;
                    tok = alpha_lut_direct_decode(moe_opt, ctx, vocab, tok, lut_smpl);
                    if (tok == LLAMA_TOKEN_NULL) break;

                    // Layer 0: Token integrity check (round-trip + spacing)
                    // Detect corruption from cache misses: decode prev_tok+tok to text,
                    // re-encode, compare. Also check for missing spaces between word tokens.
                    // On detection: refresh cache + look-back verify with updated LUT.
                    // Cost: ~microseconds per token (tokenizer only, no LLM).
                    if (n_decoded > portfolio.quality_grace_tokens && !lut_history.empty()) {
                        char cur_text[128] = {}, prev_text[128] = {};
                        int cur_len = llama_token_to_piece(vocab, tok, cur_text, sizeof(cur_text), 0, true);
                        int prev_len = llama_token_to_piece(vocab, prev_tok, prev_text, sizeof(prev_text), 0, true);
                        bool corrupt = false;

                        // Round-trip check: decode last N tokens to text, re-encode,
                        // compare token IDs. If any ID differs -> corruption detected.
                        // This catches merged tokens, wrong BPE variants, and spacing errors.
                        {
                            static constexpr int RT_WINDOW = 6;
                            if ((int)lut_history.size() >= RT_WINDOW) {
                                // Decode last RT_WINDOW tokens + current tok to text
                                size_t hn = lut_history.size();
                                std::string chunk;
                                char tb[128];
                                for (int ri = RT_WINDOW - 1; ri >= 0; ri--) {
                                    int tnb = llama_token_to_piece(vocab, lut_history[hn-1-ri], tb, sizeof(tb), 0, true);
                                    if (tnb > 0) chunk.append(tb, tnb);
                                }
                                { int tnb = llama_token_to_piece(vocab, tok, tb, sizeof(tb), 0, true);
                                  if (tnb > 0) chunk.append(tb, tnb); }

                                // Re-tokenize the chunk
                                int max_re = RT_WINDOW + 8;
                                std::vector<llama_token> re_ids(max_re);
                                int n_re = llama_tokenize(vocab, chunk.c_str(), chunk.size(),
                                        re_ids.data(), max_re, false, true);

                                // Compare: check if any token ID differs
                                if (n_re == RT_WINDOW + 1) {
                                    for (int ri = 0; ri < RT_WINDOW; ri++) {
                                        if (re_ids[ri] != (llama_token)lut_history[hn - RT_WINDOW + ri]) {
                                            corrupt = true;
                                            break;
                                        }
                                    }
                                    if (!corrupt && re_ids[RT_WINDOW] != tok) {
                                        corrupt = true;
                                    }
                                } else if (n_re != RT_WINDOW + 1) {
                                    // Token count changed - definitely corruption
                                    corrupt = true;
                                }
                            }
                        }

                        if (corrupt) {
                            int loaded = alpha_moe_optimizer_reload_from_argsort(moe_opt);
                            fprintf(stderr, "[SPACING-FIX] t=%d prev='%.*s' cur='%.*s' loaded=%d\n",
                                    n_decoded, prev_len, prev_text, cur_len, cur_text, loaded);
                            if (loaded > 0) {
                                // Look-back verify with updated LUT
                                llama_memory_t vmem = llama_get_memory(ctx);
                                if (vmem) {
                                    llama_pos vpos = llama_memory_seq_pos_max(vmem, 0);
                                    if (vpos >= 0) llama_memory_seq_rm(vmem, 0, vpos, -1);
                                }
                                llama_sampler * vs = (temperature > 0.0f) ? smpl : nullptr;
                                llama_token vt = alpha_lut_direct_decode(moe_opt, ctx, vocab, prev_tok, vs);
                                if (vt != LLAMA_TOKEN_NULL && vt != tok) {
                                    fprintf(stderr, "[SPACING-FIX] corrected tok %d → %d\n", tok, vt);
                                    tok = vt;
                                }
                            }
                            continue;
                        }
                    }

                    // Layer 0b: Duplicate word detection
                    // "result result", "from from" - same token appears twice consecutively.
                    // At K<K_min, cache misses cause the model to repeat the previous token.
                    // Detection: tok == prev_tok (consecutive duplicate).
                    // Fix: refresh cache + look-back verify.
                    if (tok == prev_tok && n_decoded > portfolio.quality_grace_tokens) {
                        // Verify it's a word token (not punctuation/whitespace)
                        char dup_text[128] = {};
                        int dup_len = llama_token_to_piece(vocab, tok, dup_text, sizeof(dup_text), 0, true);
                        bool is_word = (dup_len >= 2);
                        if (is_word) {
                            for (int di = 0; di < dup_len; di++) {
                                unsigned char c = (unsigned char)dup_text[di];
                                if (!isalpha(c) && c != ' ') { is_word = false; break; }
                            }
                        }
                        if (is_word) {
                            int loaded = alpha_moe_optimizer_reload_from_argsort(moe_opt);
                            if (loaded > 0) {
                                llama_memory_t vmem = llama_get_memory(ctx);
                                if (vmem) {
                                    llama_pos vpos = llama_memory_seq_pos_max(vmem, 0);
                                    if (vpos >= 0) llama_memory_seq_rm(vmem, 0, vpos, -1);
                                }
                                llama_sampler * vs = (temperature > 0.0f) ? smpl : nullptr;
                                llama_token vt = alpha_lut_direct_decode(moe_opt, ctx, vocab, prev_tok, vs);
                                if (vt != LLAMA_TOKEN_NULL && vt != tok) {
                                    tok = vt;
                                }
                            }
                            continue;
                        }
                    }

                    // Layer 1: Confidence gate (dynamic threshold from profile quality)
                    if (portfolio.conf_margin_threshold > 0 && n_decoded > portfolio.quality_grace_tokens) {
                        // Dynamic threshold: stricter when profile match is low
                        float pmq = alpha_moe_optimizer_get_profile_quality();
                        float dynamic_threshold = portfolio.conf_margin_threshold;
                        if (pmq > 0.0f && pmq < 0.3f) {
                            dynamic_threshold = portfolio.conf_margin_threshold * (0.3f / std::max(pmq, 0.05f));
                        }
                        float margin = alpha_lut_direct_get_margin();
                        if (margin < dynamic_threshold) low_conf_count++;
                        else low_conf_count = std::max(0, low_conf_count - 1);

                        if (low_conf_count >= portfolio.conf_trigger_count) {
                            ALPHA_LOG_DBG("[CONF-CHECK] Low confidence (%d tokens, margin=%.2f)\n",
                                    low_conf_count, margin);
                            low_conf_count = 0;
                            llama_token old_tok = tok;
                            quality_resync("CONF-CHECK");
                            if (tok != old_tok) {
                                ALPHA_LOG_DBG("[CONF-CHECK] Corrected: LUT=%d → cpu-moe=%d\n", old_tok, tok);
                            }
                            continue;
                        }
                    }

                    // Layer 3: N-gram repetition + periodic cache refresh
                    {
                        bool rep_found = false;

                        // (a) 4-gram repetition in last 100 tokens
                        if (lut_history.size() >= 16) {
                            size_t n = lut_history.size();
                            llama_token g0 = lut_history[n-4], g1 = lut_history[n-3],
                                        g2 = lut_history[n-2], g3 = lut_history[n-1];
                            size_t start = (n > 104) ? n - 104 : 0;
                            for (size_t j = start; j + 3 < n - 4; j++) {
                                if (lut_history[j]==g0 && lut_history[j+1]==g1 &&
                                    lut_history[j+2]==g2 && lut_history[j+3]==g3) {
                                    rep_found = true;
                                    break;
                                }
                            }
                        }

                        if (rep_found) {
                            if (g_cpu_moe) {
                                quality_resync("REP-CHECK");
                            } else {
                                // Lightweight: refresh cache from actual routing, update LUT
                                alpha_moe_optimizer_reload_from_argsort(moe_opt);
                            }
                            lut_history.clear();
                            continue;
                        }

                        // (b) Adaptive periodic cache refresh + 1-token look-back verify.
                        // At K<K_min, static cache drifts as content changes topics.
                        // Every N tokens: refresh cache (DMA correct experts + update LUT),
                        // then KV rewind + re-decode prev_tok using updated LUT to verify
                        // the last token. No mode switch - same LUT graph, just updated values.
                        // Cost: 1 extra LUT decode per refresh cycle (~17ms per 20 tokens).
                        {
                            static int adaptive_interval = 0;
                            static int base_interval = 0;
                            if (base_interval == 0 && portfolio.readback_interval > 0) {
                                base_interval = portfolio.readback_interval;
                                adaptive_interval = base_interval;
                            }
                            if (adaptive_interval > 0 && n_decoded > 0 &&
                                n_decoded % adaptive_interval == 0) {
                                int loaded = alpha_moe_optimizer_reload_from_argsort(moe_opt);
                                if (loaded == 0) {
                                    adaptive_interval = std::min(adaptive_interval * 2, base_interval * 4);
                                } else {
                                    adaptive_interval = base_interval;
                                    // Look-back verify: KV rewind + re-decode with updated LUT.
                                    // The last LUT decode used stale experts. With fresh cache,
                                    // re-decoding may produce a different (correct) token.
                                    llama_memory_t vmem = llama_get_memory(ctx);
                                    if (vmem) {
                                        llama_pos vpos = llama_memory_seq_pos_max(vmem, 0);
                                        if (vpos >= 0) llama_memory_seq_rm(vmem, 0, vpos, -1);
                                    }
                                    // Re-decode prev_tok with updated LUT (same graph, new LUT values)
                                    llama_sampler * lut_smpl2 = (temperature > 0.0f) ? smpl : nullptr;
                                    llama_token verified = alpha_lut_direct_decode(
                                            moe_opt, ctx, vocab, prev_tok, lut_smpl2);
                                    if (verified != LLAMA_TOKEN_NULL && verified != tok) {
                                        tok = verified;
                                    }
                                }
                            }
                        }
                    }
                }
                if (alpha_moe_optimizer_has_mixed_experts(moe_opt)) {
                    alpha_moe_optimizer_exit_mixed_draft_mode(moe_opt);
                } else {
                    alpha_moe_optimizer_exit_draft_mode(moe_opt);
                }
                goto gen_done;
            }

            llama_memory_t cs_mem = llama_get_memory(ctx);
            llama_pos cs_n_past = cs_mem ? llama_memory_seq_pos_max(cs_mem, 0) + 1 : (llama_pos)tokens.size();

            // Output warmup tokens (already generated by cpu-moe, correct output)
            for (auto wtok : g_warmup_tokens) {
                if (llama_vocab_is_eog(vocab, wtok)) break;
                char buf[128];
                int nb = llama_token_to_piece(vocab, wtok, buf, sizeof(buf), 0, true);
                if (nb > 0) {
                    if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({wtok, std::string(buf, nb), 0.0f}); }
                    else { printf("%.*s", nb, buf); fflush(stdout); }
                }
                n_decoded++;
            }

            // Use last warmup token (or sample fresh if no warmup)
            llama_token last_tok = g_warmup_tokens.empty()
                ? llama_sampler_sample(smpl, ctx, -1) : g_warmup_last_tok;
            if (!llama_vocab_is_eog(vocab, last_tok) && !g_warmup_eog) {
                // Output the transition token (sampled from last warmup decode)
                char buf[128];
                int nb = llama_token_to_piece(vocab, last_tok, buf, sizeof(buf), 0, true);
                if (nb > 0) {
                    if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({last_tok, std::string(buf, nb), 0.0f}); }
                    else { printf("%.*s", nb, buf); fflush(stdout); }
                }
                n_decoded++;

                while (n_decoded < n_predict) {
                    // Recompute n_past from actual KV state (may have changed from prev cycle)
                    cs_n_past = cs_mem ? llama_memory_seq_pos_max(cs_mem, 0) + 1 : cs_n_past;
                    std::vector<llama_token> accepted;
                    int n = alpha_cached_spec_generate(&cached_spec_state, moe_opt, ctx, smpl, vocab,
                                                       last_tok, cs_n_past, accepted);
                    if (n <= 0) break;

                    bool eog = false;
                    for (auto tok : accepted) {
                        if (llama_vocab_is_eog(vocab, tok)) { eog = true; break; }
                        char buf2[128];
                        int nb2 = llama_token_to_piece(vocab, tok, buf2, sizeof(buf2), 0, true);
                        if (nb2 > 0) {
                            if (json_output) { json_full_text.append(buf2, nb2); json_tokens.push_back({tok, std::string(buf2, nb2), 0.0f}); }
                            else { printf("%.*s", nb2, buf2); fflush(stdout); }
                        }
                        n_decoded++;
                    }
                    if (eog) break;

                    last_tok = accepted.back();
                    cs_n_past += n;

                    // N-gram repetition detection in accepted token stream.
                    // Track recent tokens, search for last 4-gram anywhere in history.
                    static std::vector<llama_token> cs_token_history;
                    for (auto tok : accepted) cs_token_history.push_back(tok);
                    bool ngram_rep = false;
                    if (moe_opt && g_cpu_moe && cs_token_history.size() >= 20) {
                        size_t n = cs_token_history.size();
                        // Take last 4 tokens as the query 4-gram
                        llama_token g0 = cs_token_history[n-4], g1 = cs_token_history[n-3],
                                    g2 = cs_token_history[n-2], g3 = cs_token_history[n-1];
                        // Search backwards in history (skip last 4) for same 4-gram
                        size_t search_start = (n > 104) ? n - 104 : 0;
                        for (size_t j = search_start; j + 3 < n - 4; j++) {
                            if (cs_token_history[j] == g0 && cs_token_history[j+1] == g1 &&
                                cs_token_history[j+2] == g2 && cs_token_history[j+3] == g3) {
                                ngram_rep = true;
                                break;
                            }
                        }
                    }
                    if ((cached_spec_state.repetition_detected || ngram_rep) && moe_opt && g_cpu_moe) {
                        cached_spec_state.repetition_detected = false;
                        ALPHA_LOG_DBG("[RESYNC] Repetition detected — switching to cpu-moe for 5 tokens\n");
                        alpha_moe_optimizer_disable_cache(moe_opt);

                        // 1-token cpu-moe resync: decode 1 token, classify + fill
                        llama_batch wb = llama_batch_get_one(&last_tok, 1);
                        llama_decode(ctx, wb);
                        alpha_moe_optimizer_warmup_classify_and_fill(moe_opt);
                        last_tok = llama_sampler_sample(smpl, ctx, -1);
                        bool resync_eog = llama_vocab_is_eog(vocab, last_tok);
                        if (!resync_eog) {
                            char buf2[128];
                            int nb2 = llama_token_to_piece(vocab, last_tok, buf2, sizeof(buf2), 0, true);
                            if (nb2 > 0) {
                                if (json_output) { json_full_text.append(buf2, nb2); json_tokens.push_back({last_tok, std::string(buf2, nb2), 0.0f}); }
                                else { printf("%.*s", nb2, buf2); fflush(stdout); }
                            }
                            n_decoded++;
                        }
                        if (resync_eog) break;

                        alpha_moe_optimizer_enable_cache(moe_opt);
                        alpha_moe_optimizer_reinstall_callback(moe_opt);
                        cs_n_past = cs_mem ? llama_memory_seq_pos_max(cs_mem, 0) + 1 : cs_n_past;
                    }
                }
            }

            alpha_cached_spec_print_stats(&cached_spec_state);
            goto gen_done;
        }

        // Self-Speculative Generation
        if (self_spec_mode && self_spec_state.initialized) {
            // Track KV cache position
            llama_memory_t ss_mem = llama_get_memory(ctx);
            llama_pos ss_n_past = ss_mem ? llama_memory_seq_pos_max(ss_mem, 0) + 1 : (llama_pos)tokens.size();

            // First: sample from prompt logits
            llama_token last_tok = llama_sampler_sample(smpl, ctx, -1);
            if (!llama_vocab_is_eog(vocab, last_tok)) {
                char buf[128];
                int nb = llama_token_to_piece(vocab, last_tok, buf, sizeof(buf), 0, true);
                if (nb > 0) {
                    if (json_output) { json_full_text.append(buf, nb); json_tokens.push_back({last_tok, std::string(buf, nb), 0.0f}); }
                    else { printf("%.*s", nb, buf); fflush(stdout); }
                }
                n_decoded++;

                while (n_decoded < n_predict) {
                    std::vector<llama_token> accepted;
                    int n = alpha_self_spec_generate(&self_spec_state, moe_opt, ctx, smpl, vocab,
                                                     last_tok, ss_n_past, accepted);
                    if (n <= 0) break;

                    bool eog = false;
                    for (auto tok : accepted) {
                        if (llama_vocab_is_eog(vocab, tok)) { eog = true; break; }
                        char buf2[128];
                        int nb2 = llama_token_to_piece(vocab, tok, buf2, sizeof(buf2), 0, true);
                        if (nb2 > 0) {
                            if (json_output) { json_full_text.append(buf2, nb2); json_tokens.push_back({tok, std::string(buf2, nb2), 0.0f}); }
                            else { printf("%.*s", nb2, buf2); fflush(stdout); }
                        }
                        n_decoded++;
                    }
                    if (eog) break;

                    last_tok = accepted.back();
                    ss_n_past += n;
                }
            }

            goto gen_done;
        }

        for (int i = 0; i < n_predict; i++) {
#ifdef DSV4_DEBUG_LAYERS
            // Print first 10 logits for CPU vs CUDA comparison
            if (i == 0) {
                float logits[10];
                const float * lp = llama_get_logits(ctx);
                if (lp) {
                    memcpy(logits, lp, 10 * sizeof(float));
                    fprintf(stdout, "LOGITS[0..9]: ");
                    for (int li = 0; li < 10; li++) fprintf(stdout, "%.6f ", logits[li]);
                    fprintf(stdout, "\n");
                    fflush(stdout);
                }
            }
#endif
            // Sample next token (backend sampler returns GPU-selected token instantly)
            llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, new_token)) break;

            float token_prob = 0.0f;
            if (log_confidence || json_output) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                if (logits && n_vocab > 0 && new_token >= 0 && new_token < n_vocab) {
                float max_logit = logits[0];
                for (int v = 1; v < n_vocab; v++) {
                    if (logits[v] > max_logit) max_logit = logits[v];
                }
                float sum_exp = 0.0f;
                for (int v = 0; v < n_vocab; v++) {
                    sum_exp += expf(logits[v] - max_logit);
                }
                token_prob = expf(logits[new_token] - max_logit) / sum_exp;
                if (log_confidence) {
                    ALPHA_LOG_DBG("[CONFIDENCE] token=%d prob=%.6f\n", new_token, token_prob);
                }
                } // if (logits && n_vocab > 0)
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
            if (n > 0) {
                if (json_output) {
                    json_full_text.append(buf, n);
                    json_tokens.push_back({new_token, std::string(buf, n), token_prob});
                } else {
                    printf("%.*s", n, buf);
                    if (interactive) fflush(stdout);
                }
            }

            if (speculative_mode && spec_state.initialized) {
                // Speculative execution: prepare -> speculate -> verify -> (rerun if miss)
                if (alpha_speculative_decode_token(&spec_state, moe_opt, ctx, new_token) != 0) break;
            } else {
                // Standard reactive path
                if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
                    alpha_moe_optimizer_prepare(moe_opt);
                }

                if (dense_pipe) {
                    // Multi-pass pipeline: each pass computes a layer range on GPU
                    // Pass 1: token batch -> layers [0, N/2) -> hidden state
                    // Pass 2: embeddings batch (hidden state) -> layers [N/2, N) -> logits
                    int n_passes = alpha_dense_pipeline_n_passes(dense_pipe);
                    int n_embd = llama_model_n_embd(model);
                    bool decode_ok = true;
                    std::vector<float> hidden_state(n_embd);

                    // Enable embeddings output for reading hidden state between passes
                    llama_set_embeddings(ctx, true);

                    for (int pass = 0; pass < n_passes; pass++) {
                        alpha_dense_pipeline_prepare_pass(dense_pipe, model, pass);
                        llama_context_reset_sched(ctx);

                        if (pass == 0) {
                            // First pass: normal token input
                            llama_batch batch = llama_batch_get_one(&new_token, 1);
                            if (llama_decode(ctx, batch) != 0) { decode_ok = false; break; }
                        } else {
                            // Subsequent passes: feed hidden state as embeddings
                            llama_batch batch = llama_batch_init(1, n_embd, 1);
                            memcpy(batch.embd, hidden_state.data(), n_embd * sizeof(float));
                            batch.n_tokens = 1;
                            batch.pos[0] = 0;  // position doesn't matter for continuation
                            batch.n_seq_id[0] = 1;
                            llama_seq_id seq = 0;
                            batch.seq_id[0] = &seq;
                            batch.logits[0] = (pass == n_passes - 1) ? 1 : 0;
                            if (llama_decode(ctx, batch) != 0) {
                                llama_batch_free(batch);
                                decode_ok = false;
                                break;
                            }
                            llama_batch_free(batch);
                        }

                        // Read hidden state for next pass (not needed for last pass)
                        if (pass < n_passes - 1) {
                            float * embd = llama_get_embeddings(ctx);
                            if (embd) {
                                memcpy(hidden_state.data(), embd, n_embd * sizeof(float));
                            }
                        }

                        alpha_dense_pipeline_finish_pass(dense_pipe, model, pass);
                    }

                    llama_set_embeddings(ctx, false);
                    if (!decode_ok) break;
                } else {
                    llama_batch batch = llama_batch_get_one(&new_token, 1);
                    if (llama_decode(ctx, batch) != 0) break;
                }

                if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
                    alpha_moe_optimizer_record(moe_opt);
                }
            }

            // Expert routing profiling: write per-layer top-k expert IDs to binary file
            // Works in any mode (cpu-moe, expert-cache, direct) - reads graph argsort outputs
            if (profile_experts_fp) {
                int n_argsort = alpha_lora_mix_get_argsort_count();
                for (int m = 0; m < n_argsort; m++) {
                    ggml_tensor * argsort = alpha_lora_mix_get_argsort_output(m);
                    if (!argsort || !argsort->buffer) continue;
                    int n_used = std::min((int)argsort->ne[0], 6);
                    int32_t ids[6] = {};
                    ggml_backend_tensor_get(argsort, ids, 0, n_used * sizeof(int32_t));
                    fwrite(ids, sizeof(int32_t), 6, profile_experts_fp);
                }
                fflush(profile_experts_fp);
            }

            // Gate entropy: raw per-layer gate score entropy
            // Compute per-layer gate entropy (used by both --log-gate-entropy and --log-confidence [LAYER])
            // Stores in layer_entropy[] for reuse by [LAYER] difficulty field
            std::vector<float> layer_entropy;
            int n_expert_for_max = 256;
            if ((log_gate_entropy || log_confidence) && moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
                int n_layer = llama_model_n_layer(model);
                layer_entropy.resize(n_layer, -1.0f);
                float total_entropy = 0;
                int n_moe = 0;
                for (int il = 0; il < n_layer; il++) {
                    ggml_tensor * gating = alpha_lora_mix_get_gating_probs(il);
                    if (!gating || !gating->buffer) continue;
                    int n_exp = (int)gating->ne[0];
                    if (il == 0) n_expert_for_max = n_exp;
                    std::vector<float> probs(n_exp);
                    ggml_backend_tensor_get(gating, probs.data(), 0, n_exp * sizeof(float));
                    // Check if already probabilities (sum ~ 1) or raw logits
                    float sum_check = 0.0f;
                    for (int e = 0; e < n_exp; e++) sum_check += probs[e];
                    if (sum_check < 0.9f || sum_check > 1.1f) {
                        // Raw logits - apply softmax
                        float max_val = probs[0];
                        for (int e = 1; e < n_exp; e++) {
                            if (probs[e] > max_val) max_val = probs[e];
                        }
                        float sum_exp = 0.0f;
                        for (int e = 0; e < n_exp; e++) {
                            probs[e] = expf(probs[e] - max_val);
                            sum_exp += probs[e];
                        }
                        for (int e = 0; e < n_exp; e++) {
                            probs[e] /= sum_exp;
                        }
                    }
                    // Compute entropy: H = -sum(p * log(p))
                    float entropy = 0.0f;
                    for (int e = 0; e < n_exp; e++) {
                        if (probs[e] > 1e-10f) {
                            entropy -= probs[e] * logf(probs[e]);
                        }
                    }
                    layer_entropy[il] = entropy;
                    total_entropy += entropy;
                    n_moe++;
                }
                if (log_gate_entropy && n_moe > 0) {
                    float max_entropy = logf((float)n_expert_for_max);
                    ALPHA_LOG_DBG("[GATE-ENTROPY] token=%d avg=%.3f nats (%.1f%% of max=%.2f) layers=%d\n",
                            new_token, total_entropy / n_moe,
                            100.0f * (total_entropy / n_moe) / max_entropy,
                            max_entropy, n_moe);
                    // Per-layer output
                    for (int il = 0; il < n_layer; il++) {
                        if (layer_entropy[il] >= 0.0f) {
                            ALPHA_LOG_DBG("[GATE-ENTROPY] token=%d layer=%d entropy=%.3f nats (%.1f%%)\n",
                                    new_token, il, layer_entropy[il],
                                    100.0f * layer_entropy[il] / max_entropy);
                        }
                    }
                }
            }

            // Per-layer routing entropy + expert IDs
            if (log_confidence && moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
                int n_layer = llama_model_n_layer(model);
                float max_ent = logf((float)n_expert_for_max);
                int n_argsort = alpha_lora_mix_get_argsort_count();
                int moe_idx = 0;
                for (int il = 0; il < n_layer; il++) {
                    // Use real gate entropy from layer_entropy[]
                    float diff;
                    if (il < (int)layer_entropy.size() && layer_entropy[il] >= 0.0f) {
                        diff = (max_ent > 0) ? layer_entropy[il] / max_ent : 0.5f;
                    } else {
                        diff = alpha_moe_optimizer_get_difficulty(moe_opt, il);
                    }
                    // Skip non-MoE layers
                    if (il < (int)layer_entropy.size() && layer_entropy[il] < 0.0f) continue;
                    ALPHA_LOG_DBG("[LAYER] token=%d layer=%d difficulty=%.4f entropy=%.3f",
                            new_token, il, diff,
                            (il < (int)layer_entropy.size() && layer_entropy[il] >= 0.0f)
                                ? layer_entropy[il] : 0.0f);
                    // Expert IDs: prefer prev_routing (valid in cache mode, updated by callback)
                    // Fall back to argsort direct read (may have recycled data in direct mode)
                    const int * routing = alpha_moe_optimizer_get_prev_routing(moe_opt, il);
                    int n_used = alpha_moe_optimizer_get_n_expert_used(moe_opt);
                    bool have_valid_routing = false;
                    if (routing && n_used > 0) {
                        // Check if routing has valid data (init state = 0,1,2,...n_used-1)
                        bool is_init = true;
                        for (int k = 0; k < n_used && k < 8; k++) {
                            if (routing[k] != k) { is_init = false; break; }
                        }
                        // Also check for out-of-range values (garbage from recycled memory)
                        if (!is_init) {
                            bool in_range = true;
                            for (int k = 0; k < n_used && k < 8; k++) {
                                if (routing[k] < 0 || routing[k] >= 256) { in_range = false; break; }
                            }
                            if (in_range) have_valid_routing = true;
                        }
                    }
                    if (have_valid_routing) {
                        fprintf(stderr, " experts=");
                        for (int k = 0; k < n_used && k < 8; k++)
                            fprintf(stderr, "%s%d", k > 0 ? "," : "", routing[k]);
                    } else if (moe_idx < n_argsort) {
                        // Fallback: read argsort tensor directly
                        ggml_tensor * argsort = alpha_lora_mix_get_argsort_output(moe_idx);
                        if (argsort && argsort->buffer) {
                            int n_rd = std::min((int)argsort->ne[0], 8);
                            int32_t ids[16] = {};
                            ggml_backend_tensor_get(argsort, ids, 0, n_rd * sizeof(int32_t));
                            fprintf(stderr, " experts=");
                            for (int k = 0; k < n_rd; k++) {
                                int eid = ids[k];
                                if (eid < 0 || eid >= 256) eid = -1;
                                fprintf(stderr, "%s%d", k > 0 ? "," : "", eid);
                            }
                        }
                    }
                    if (moe_idx < n_argsort) moe_idx++;
                    fprintf(stderr, "\n");
                }
            }

            n_decoded++;
        }

        gen_done:
        if (profile_experts_fp) {
            fclose(profile_experts_fp);
            profile_experts_fp = nullptr;
            ALPHA_LOG_DBG("[PROFILE] Expert routing data saved\n");
        }

        // Write frequency profile JSON if requested
        if (!profile_freq_path.empty() && moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
            FILE * freq_fp = fopen(profile_freq_path.c_str(), "w");
            if (freq_fp) {
                int n_exp_used = alpha_moe_optimizer_get_n_expert_used(moe_opt);
                fprintf(freq_fp, "{\n");
                fprintf(freq_fp, "  \"model\": \"%s\",\n", path.c_str());
                fprintf(freq_fp, "  \"tp\": %d,\n", n_decoded);
                fprintf(freq_fp, "  \"ly\": [\n");

                // Get layer count from optimizer
                int n_expert_out = 0;
                bool first_layer = true;
                int n_total_layers = alpha_moe_optimizer_get_n_layer(moe_opt);
                for (int il = 0; il < n_total_layers; il++) {
                    float * freqs = alpha_moe_optimizer_get_expert_frequencies(moe_opt, il, &n_expert_out);
                    if (!freqs || n_expert_out == 0) continue;  // skip layers with no data

                    // Sort experts by frequency (descending)
                    std::vector<std::pair<float, int>> sorted_experts;
                    for (int e = 0; e < n_expert_out; e++) {
                        if (freqs[e] > 0.0f) {
                            sorted_experts.push_back({freqs[e], e});
                        }
                    }
                    std::sort(sorted_experts.begin(), sorted_experts.end(),
                              [](const auto & a, const auto & b) { return a.first > b.first; });

                    if (!first_layer) fprintf(freq_fp, ",\n");
                    first_layer = false;

                    fprintf(freq_fp, "    {\"l\": %d, \"te\": [", il);
                    for (size_t j = 0; j < sorted_experts.size(); j++) {
                        if (j > 0) fprintf(freq_fp, ", ");
                        fprintf(freq_fp, "%d", sorted_experts[j].second);
                    }
                    fprintf(freq_fp, "], \"fr\": [");
                    for (size_t j = 0; j < sorted_experts.size(); j++) {
                        if (j > 0) fprintf(freq_fp, ", ");
                        fprintf(freq_fp, "%.4f", sorted_experts[j].first);
                    }
                    fprintf(freq_fp, "]}");

                    free(freqs);
                }
                fprintf(freq_fp, "\n  ]\n}\n");
                fclose(freq_fp);
                ALPHA_LOG_DBG("[FREQ] Expert frequency profile saved to %s (%d tokens)\n",
                        profile_freq_path.c_str(), n_decoded);
            } else {
                ALPHA_LOG_DBG("[FREQ] WARNING: cannot write frequency profile to %s\n",
                        profile_freq_path.c_str());
            }
        }
        if (json_output) {
            const auto t_end = ggml_time_us();
            double elapsed = (t_end - t_start) / 1000000.0;
            // Split thinking vs content at </think>
            std::string thinking, content;
            size_t think_end = json_full_text.find("</think>");
            if (think_end != std::string::npos) {
                size_t think_start = json_full_text.find("<think>");
                size_t body_start = (think_start != std::string::npos) ? think_start + 7 : 0;
                thinking = json_full_text.substr(body_start, think_end - body_start);
                size_t content_start = think_end + 8; // len("</think>")
                while (content_start < json_full_text.size() && (json_full_text[content_start] == '\n' || json_full_text[content_start] == ' '))
                    content_start++;
                content = json_full_text.substr(content_start);
            } else {
                // No </think> tag - check if <think> is present (model hit token limit mid-thinking)
                size_t think_start = json_full_text.find("<think>");
                if (think_start != std::string::npos) {
                    // Strip <think> tag, put everything as thinking (incomplete)
                    thinking = json_full_text.substr(think_start + 7);
                    // Trim leading/trailing whitespace
                    while (!thinking.empty() && (thinking.front() == '\n' || thinking.front() == ' ')) thinking.erase(0, 1);
                    while (!thinking.empty() && (thinking.back() == '\n' || thinking.back() == ' ')) thinking.pop_back();
                    content = "";
                } else {
                    content = json_full_text;
                }
            }
            // Trim whitespace from thinking/content
            while (!thinking.empty() && (thinking.back() == '\n' || thinking.back() == ' ')) thinking.pop_back();
            while (!content.empty() && (content.front() == '\n' || content.front() == ' ')) content.erase(0, 1);
            // Escape JSON strings
            auto json_escape = [](const std::string & s) -> std::string {
                std::string out;
                out.reserve(s.size() + 16);
                for (char c : s) {
                    switch (c) {
                        case '"':  out += "\\\""; break;
                        case '\\': out += "\\\\"; break;
                        case '\n': out += "\\n"; break;
                        case '\r': out += "\\r"; break;
                        case '\t': out += "\\t"; break;
                        default:
                            if ((unsigned char)c < 0x20) {
                                char hex[8]; snprintf(hex, sizeof(hex), "\\u%04x", (unsigned char)c);
                                out += hex;
                            } else { out += c; }
                    }
                }
                return out;
            };
            printf("{\"content\":\"%s\"", json_escape(content).c_str());
            if (!thinking.empty()) {
                printf(",\"thinking\":\"%s\"", json_escape(thinking).c_str());
            }
            if (log_confidence && !json_tokens.empty()) {
                printf(",\"tokens\":[");
                for (size_t ti = 0; ti < json_tokens.size(); ti++) {
                    if (ti > 0) printf(",");
                    printf("{\"id\":%d,\"text\":\"%s\",\"prob\":%.6f}",
                           json_tokens[ti].id,
                           json_escape(json_tokens[ti].text).c_str(),
                           json_tokens[ti].prob);
                }
                printf("]");
            }
            printf(",\"eval_count\":%d", n_decoded);
            printf(",\"eval_duration_ms\":%.0f", elapsed * 1000.0);
            printf(",\"tokens_per_sec\":%.2f", n_decoded > 0 ? n_decoded / elapsed : 0.0);
            printf("}\n");
        } else {
            printf("\n");
        }

        if (!interactive && !json_output) {
            const auto t_end = ggml_time_us();
            double elapsed = (t_end - t_start) / 1000000.0;
            // Restore line buffering and flush remaining output
            setvbuf(stdout, nullptr, _IOLBF, 0);
            fflush(stdout);
            fprintf(stderr, "\n%d tokens in %.2f s (%.2f tok/s)\n", n_decoded, elapsed,
                    n_decoded > 0 ? n_decoded / elapsed : 0);
        }

        if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt)) {
            alpha_moe_optimizer_print_stats(moe_opt);
        }

        if (spec_state.initialized) {
            alpha_speculative_print_stats(&spec_state);
        }

        if (self_spec_state.initialized) {
            alpha_self_spec_print_stats(&self_spec_state);
        }

    } while (interactive);

    if (spec_state.initialized) {
        alpha_speculative_cleanup(&spec_state);
    }

    if (moe_opt) {
        alpha_moe_optimizer_free(moe_opt);
    }

    // Cleanup
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int cmd_bench(const std::string & model_name, int n_tokens,
                     int n_gpu_layers, bool ngl_explicit) {
    std::string path = resolve_model(model_name);
    if (path.empty()) {
        fprintf(stderr, "Model not found: %s\n", model_name.c_str());
        return 2;
    }

    printf("Benchmarking: %s\n", model_name.c_str());
    printf("Tokens:       %d\n\n", n_tokens);

    // Use cmd_run with a standard prompt
    return cmd_run(model_name,
        "Write a comprehensive Python function that implements a binary search tree.",
        n_tokens, true, 0, n_gpu_layers, 0, 0.0f, 42, ngl_explicit);
}

static bool is_hf_repo(const std::string & name) {
    // HF repos have format "owner/repo" or "owner/repo:tag"
    // Must contain exactly one / and not start with / or .
    if (name.empty() || name[0] == '/' || name[0] == '.') return false;
    auto slash = name.find('/');
    if (slash == std::string::npos) return false;
    // Check it's not a file path (no second slash except in tag)
    auto second_slash = name.find('/', slash + 1);
    return second_slash == std::string::npos;
}

// Model Catalog

struct model_entry {
    const char * alias;        // short name (alphachat pull qwen3.6)
    const char * hf_repo;      // HuggingFace repo
    const char * description;  // one-line description
    int size_b;                // total params in billions
    bool is_moe;
    int active_b;              // active params (MoE) or same as size_b
};

// Remote catalog (fetched from GitHub, cached locally)

static const char * CATALOG_URL = "https://alphachat.com/api/catalog.json";
static const int CATALOG_TTL_DAYS = 7;  // auto-refresh after 7 days

static std::string get_catalog_path() {
    return get_config_dir() + "/catalog.json";
}

static bool catalog_is_stale() {
    std::string path = get_catalog_path();
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return true;  // doesn't exist
    time_t now = time(nullptr);
    double age_days = difftime(now, st.st_mtime) / 86400.0;
    return age_days > CATALOG_TTL_DAYS;
}

// Dynamic catalog entries loaded from JSON
static std::vector<model_entry> g_remote_catalog;
static std::vector<std::string> g_remote_strings;  // owns string memory

static bool load_remote_catalog() {
    std::string path = get_catalog_path();
    FILE * f = fopen(path.c_str(), "r");
    if (!f) return false;

    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 10*1024*1024) { fclose(f); return false; }

    std::string json(sz, '\0');
    fread(&json[0], 1, sz, f);
    fclose(f);

    // Simple JSON array parser: [{"alias":"...","hf_repo":"...","description":"...","size_b":N,"is_moe":bool,"active_b":N}, ...]
    g_remote_catalog.clear();
    g_remote_strings.clear();

    size_t pos = json.find('[');
    if (pos == std::string::npos) return false;

    while (pos < json.size()) {
        size_t obj_start = json.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = json.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = json.substr(obj_start, obj_end - obj_start + 1);

        auto get_str = [&](const char * key) -> std::string {
            std::string k = std::string("\"") + key + "\"";
            size_t kp = obj.find(k);
            if (kp == std::string::npos) return "";
            size_t vs = obj.find('"', kp + k.size() + 1);
            if (vs == std::string::npos) return "";
            size_t ve = obj.find('"', vs + 1);
            if (ve == std::string::npos) return "";
            return obj.substr(vs + 1, ve - vs - 1);
        };
        auto get_int = [&](const char * key) -> int {
            std::string k = std::string("\"") + key + "\"";
            size_t kp = obj.find(k);
            if (kp == std::string::npos) return 0;
            size_t vs = obj.find(':', kp);
            if (vs == std::string::npos) return 0;
            return atoi(obj.c_str() + vs + 1);
        };
        auto get_bool = [&](const char * key) -> bool {
            std::string k = std::string("\"") + key + "\"";
            size_t kp = obj.find(k);
            if (kp == std::string::npos) return false;
            return obj.find("true", kp) < obj_end;
        };

        std::string alias = get_str("alias");
        std::string repo = get_str("hf_repo");
        if (!alias.empty() && !repo.empty()) {
            g_remote_strings.push_back(alias);
            g_remote_strings.push_back(repo);
            g_remote_strings.push_back(get_str("description"));
            model_entry e;
            e.alias = g_remote_strings[g_remote_strings.size()-3].c_str();
            e.hf_repo = g_remote_strings[g_remote_strings.size()-2].c_str();
            e.description = g_remote_strings[g_remote_strings.size()-1].c_str();
            e.size_b = get_int("size_b");
            e.is_moe = get_bool("is_moe");
            e.active_b = get_int("active_b");
            g_remote_catalog.push_back(e);
        }
        pos = obj_end + 1;
    }
    return !g_remote_catalog.empty();
}

static bool fetch_catalog(bool quiet = false) {
    std::string path = get_catalog_path();
    std::string dir = get_config_dir();
    fs::create_directories(dir);

    if (!quiet) printf("Updating model catalog...\n");
    common_download_opts opts;
    int status = common_download_file_single(CATALOG_URL, path, opts, true);
    if (status == 200) {
        if (!quiet) printf("Catalog updated: %s\n", path.c_str());
        load_remote_catalog();
        return true;
    }
    if (!quiet) fprintf(stderr, "Catalog update failed (HTTP %d)\n", status);
    return false;
}

static void ensure_catalog_loaded() {
    static bool loaded = false;
    if (loaded) return;
    loaded = true;

    // Try loading cached catalog
    if (load_remote_catalog()) return;

    // No cache - try fetching (quiet, best-effort)
    fetch_catalog(true);
}

static const model_entry MODEL_CATALOG[] = {
    // Qwen family
    {"qwen3.6",           "Qwen/Qwen3.6-35B-A3B-GGUF",                  "Qwen 3.6 35B MoE — best open MoE",               35,  true,  3},
    {"qwen3.5",           "Qwen/Qwen3.5-32B-GGUF",                      "Qwen 3.5 32B — strong reasoning",                 32,  false, 32},
    {"qwen3.5:4b",        "Qwen/Qwen3.5-4B-GGUF",                       "Qwen 3.5 4B — fast, good quality",                4,   false, 4},
    {"qwen3.5:0.8b",      "Qwen/Qwen3.5-0.8B-GGUF",                     "Qwen 3.5 0.8B — tiny, instant",                   1,   false, 1},
    {"qwen3:235b",        "Qwen/Qwen3-235B-A22B-GGUF",                  "Qwen 3 235B MoE — flagship",                      235, true,  22},
    {"qwen3:30b",         "Qwen/Qwen3-30B-A3B-GGUF",                    "Qwen 3 30B MoE — efficient",                      30,  true,  3},
    {"qwen3:32b",         "Qwen/Qwen3-32B-GGUF",                        "Qwen 3 32B dense — strong",                       32,  false, 32},
    {"qwen3:14b",         "Qwen/Qwen3-14B-GGUF",                        "Qwen 3 14B — balanced",                           14,  false, 14},
    {"qwen3:8b",          "Qwen/Qwen3-8B-GGUF",                         "Qwen 3 8B — versatile",                           8,   false, 8},
    {"qwen3:4b",          "Qwen/Qwen3-4B-GGUF",                         "Qwen 3 4B — compact",                             4,   false, 4},
    {"qwen3:1.7b",        "Qwen/Qwen3-1.7B-GGUF",                       "Qwen 3 1.7B — small",                             2,   false, 2},
    {"qwen3:0.6b",        "Qwen/Qwen3-0.6B-GGUF",                       "Qwen 3 0.6B — smallest",                          1,   false, 1},
    {"qwen2.5:72b",       "Qwen/Qwen2.5-72B-Instruct-GGUF",            "Qwen 2.5 72B — large, strong",                    72,  false, 72},
    {"qwen2.5:32b",       "Qwen/Qwen2.5-32B-Instruct-GGUF",            "Qwen 2.5 32B — balanced",                         32,  false, 32},
    {"qwen2.5:14b",       "Qwen/Qwen2.5-14B-Instruct-GGUF",            "Qwen 2.5 14B — efficient",                        14,  false, 14},
    {"qwen2.5:7b",        "Qwen/Qwen2.5-7B-Instruct-GGUF",             "Qwen 2.5 7B — fast",                              7,   false, 7},
    {"qwen2.5:3b",        "Qwen/Qwen2.5-3B-Instruct-GGUF",             "Qwen 2.5 3B — compact",                           3,   false, 3},
    {"qwen2.5:1.5b",      "Qwen/Qwen2.5-1.5B-Instruct-GGUF",           "Qwen 2.5 1.5B — tiny",                            2,   false, 2},
    {"qwen2.5:0.5b",      "Qwen/Qwen2.5-0.5B-Instruct-GGUF",           "Qwen 2.5 0.5B — smallest",                        1,   false, 1},
    {"qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",      "Qwen 2.5 Coder 32B — best open coder",            32,  false, 32},
    {"qwen2.5-coder:14b", "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",      "Qwen 2.5 Coder 14B — strong coder",               14,  false, 14},
    {"qwen2.5-coder:7b",  "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",       "Qwen 2.5 Coder 7B — fast coder",                  7,   false, 7},
    {"qwq:32b",           "Qwen/QwQ-32B-GGUF",                          "QwQ 32B — Qwen reasoning model",                  32,  false, 32},

    // Llama family
    {"llama4:scout",      "meta-llama/Llama-4-Scout-17B-16E-Instruct-GGUF",   "Llama 4 Scout 109B MoE — latest Meta",      109, true,  17},
    {"llama4:maverick",   "meta-llama/Llama-4-Maverick-17B-128E-Instruct-GGUF","Llama 4 Maverick 400B MoE — frontier",      400, true,  17},
    {"llama3.3:70b",      "meta-llama/Llama-3.3-70B-Instruct-GGUF",           "Llama 3.3 70B — best Llama dense",           70,  false, 70},
    {"llama3.1:70b",      "meta-llama/Llama-3.1-70B-Instruct-GGUF",           "Llama 3.1 70B — strong all-around",          70,  false, 70},
    {"llama3.1:8b",       "meta-llama/Llama-3.1-8B-Instruct-GGUF",            "Llama 3.1 8B — fast, versatile",             8,   false, 8},
    {"llama3.2:3b",       "meta-llama/Llama-3.2-3B-Instruct-GGUF",            "Llama 3.2 3B — compact",                     3,   false, 3},
    {"llama3.2:1b",       "meta-llama/Llama-3.2-1B-Instruct-GGUF",            "Llama 3.2 1B — tiny",                        1,   false, 1},
    {"codellama:34b",     "codellama/CodeLlama-34b-Instruct-GGUF",            "CodeLlama 34B — Meta code model",             34,  false, 34},
    {"codellama:13b",     "codellama/CodeLlama-13b-Instruct-GGUF",            "CodeLlama 13B — balanced coder",              13,  false, 13},
    {"codellama:7b",      "codellama/CodeLlama-7b-Instruct-GGUF",             "CodeLlama 7B — fast coder",                   7,   false, 7},

    // Gemma family
    {"gemma4:27b",        "google/gemma-4-27b-it-GGUF",                  "Gemma 4 27B — Google, strong reasoning",          27,  true,  4},
    {"gemma3:27b",        "google/gemma-3-27b-it-GGUF",                  "Gemma 3 27B — Google, balanced",                  27,  false, 27},
    {"gemma3:12b",        "google/gemma-3-12b-it-GGUF",                  "Gemma 3 12B — Google, efficient",                 12,  false, 12},
    {"gemma3:4b",         "google/gemma-3-4b-it-GGUF",                   "Gemma 3 4B — Google, fast",                       4,   false, 4},
    {"gemma3:1b",         "google/gemma-3-1b-it-GGUF",                   "Gemma 3 1B — Google, tiny",                       1,   false, 1},
    {"gemma2:27b",        "google/gemma-2-27b-it-GGUF",                  "Gemma 2 27B — Google, strong",                    27,  false, 27},
    {"gemma2:9b",         "google/gemma-2-9b-it-GGUF",                   "Gemma 2 9B — Google, balanced",                   9,   false, 9},
    {"gemma2:2b",         "google/gemma-2-2b-it-GGUF",                   "Gemma 2 2B — Google, tiny",                       2,   false, 2},

    // Mistral / Mixtral family
    {"mistral:7b",        "mistralai/Mistral-7B-Instruct-v0.3-GGUF",    "Mistral 7B — fast, efficient",                    7,   false, 7},
    {"mistral-nemo:12b",  "mistralai/Mistral-Nemo-Instruct-2407-GGUF",  "Mistral Nemo 12B — strong, compact",              12,  false, 12},
    {"mistral-small:24b", "mistralai/Mistral-Small-24B-Instruct-2501-GGUF", "Mistral Small 24B — balanced",                24,  false, 24},
    {"mixtral:8x7b",      "mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF",  "Mixtral 8x7B MoE — efficient",                   47,  true,  13},
    {"mixtral:8x22b",     "mistralai/Mixtral-8x22B-Instruct-v0.1-GGUF", "Mixtral 8x22B MoE — large",                      141, true,  39},
    {"codestral:22b",     "mistralai/Codestral-22B-v0.1-GGUF",          "Codestral 22B — Mistral code model",              22,  false, 22},

    // DeepSeek family
    {"deepseek-r1",       "unsloth/DeepSeek-R1-GGUF",                          "DeepSeek R1 671B MoE — best reasoning",    671, true,  37},
    {"deepseek-r1:70b",   "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-GGUF",   "DeepSeek R1 70B distill — strong reasoning",70, false, 70},
    {"deepseek-r1:32b",   "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B-GGUF",    "DeepSeek R1 32B distill — balanced",        32,  false, 32},
    {"deepseek-r1:14b",   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF",    "DeepSeek R1 14B distill — efficient",       14,  false, 14},
    {"deepseek-r1:8b",    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B-GGUF",    "DeepSeek R1 8B distill — fast",             8,   false, 8},
    {"deepseek-r1:7b",    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF",     "DeepSeek R1 7B distill — compact",          7,   false, 7},
    {"deepseek-r1:1.5b",  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",   "DeepSeek R1 1.5B distill — tiny",           2,   false, 2},
    {"deepseek-v3",       "unsloth/DeepSeek-V3-0324-GGUF",                     "DeepSeek V3 671B MoE — frontier quality",   671, true,  37},
    {"deepseek-v4-flash", "unsloth/DeepSeek-V4-0506-Flash-GGUF",               "DeepSeek V4 Flash 236B MoE — fast+smart",   236, true,  21},

    // Phi family (Microsoft)
    {"phi4:14b",          "microsoft/phi-4-GGUF",                        "Phi 4 14B — Microsoft, strong reasoning",         14,  false, 14},
    {"phi3.5:3.8b",       "microsoft/Phi-3.5-mini-instruct-GGUF",       "Phi 3.5 Mini 3.8B — Microsoft, compact",          4,   false, 4},
    {"phi3.5-moe:42b",    "microsoft/Phi-3.5-MoE-instruct-GGUF",        "Phi 3.5 MoE 42B — Microsoft, efficient MoE",      42,  true,  7},
    {"phi3:14b",          "microsoft/Phi-3-medium-4k-instruct-gguf",     "Phi 3 Medium 14B — Microsoft, balanced",          14,  false, 14},
    {"phi3:3.8b",         "microsoft/Phi-3-mini-4k-instruct-gguf",       "Phi 3 Mini 3.8B — Microsoft, compact",            4,   false, 4},

    // GPT-OSS (OpenAI open-weight)
    {"gpt-oss:20b",       "openai/gpt-oss-20b-GGUF",                    "GPT-OSS 20B MoE — OpenAI open-weight",            20,  true,  4},

    // Command R (Cohere)
    {"command-r-plus",    "CohereForAI/c4ai-command-r-plus-GGUF",       "Command R+ 104B — Cohere, RAG-optimized",         104, false, 104},
    {"command-r",         "CohereForAI/c4ai-command-r-v01-GGUF",        "Command R 35B — Cohere, efficient RAG",           35,  false, 35},

    // DBRX (Databricks)
    {"dbrx",              "databricks/dbrx-instruct-GGUF",              "DBRX 132B MoE — Databricks, strong",              132, true,  36},

    // Yi (01.AI)
    {"yi1.5:34b",         "01-ai/Yi-1.5-34B-Chat-GGUF",                "Yi 1.5 34B — 01.AI, strong multilingual",         34,  false, 34},
    {"yi1.5:9b",          "01-ai/Yi-1.5-9B-Chat-GGUF",                 "Yi 1.5 9B — 01.AI, efficient",                    9,   false, 9},
    {"yi1.5:6b",          "01-ai/Yi-1.5-6B-Chat-GGUF",                 "Yi 1.5 6B — 01.AI, compact",                      6,   false, 6},

    // InternLM (Shanghai AI Lab)
    {"internlm2.5:20b",  "internlm/internlm2_5-20b-chat-GGUF",        "InternLM 2.5 20B — strong multilingual",          20,  false, 20},
    {"internlm2.5:7b",   "internlm/internlm2_5-7b-chat-GGUF",         "InternLM 2.5 7B — fast multilingual",             7,   false, 7},

    // StarCoder (BigCode)
    {"starcoder2:15b",    "bigcode/starcoder2-15b-GGUF",               "StarCoder 2 15B — open code model",               15,  false, 15},
    {"starcoder2:7b",     "bigcode/starcoder2-7b-GGUF",                "StarCoder 2 7B — fast code model",                7,   false, 7},
    {"starcoder2:3b",     "bigcode/starcoder2-3b-GGUF",                "StarCoder 2 3B — compact coder",                  3,   false, 3},

    // Falcon (TII)
    {"falcon2:11b",       "tiiuae/falcon-11B-GGUF",                    "Falcon 2 11B — TII, efficient",                   11,  false, 11},

    // OLMo (Allen AI)
    {"olmo2:13b",         "allenai/OLMo-2-1124-13B-Instruct-GGUF",     "OLMo 2 13B — Allen AI, open research",            13,  false, 13},
    {"olmo2:7b",          "allenai/OLMo-2-1124-7B-Instruct-GGUF",      "OLMo 2 7B — Allen AI, open research",             7,   false, 7},

    // Nemotron (NVIDIA)
    {"nemotron:70b",      "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF", "Nemotron 70B — NVIDIA, strong",              70,  false, 70},

    // Granite (IBM)
    {"granite3.1:8b",     "ibm-granite/granite-3.1-8b-instruct-GGUF",  "Granite 3.1 8B — IBM, enterprise",                8,   false, 8},
    {"granite3.1:2b",     "ibm-granite/granite-3.1-2b-instruct-GGUF",  "Granite 3.1 2B — IBM, compact",                   2,   false, 2},

    // Aya (Cohere multilingual)
    {"aya:35b",           "CohereForAI/aya-23-35B-GGUF",               "Aya 23 35B — 23 languages",                       35,  false, 35},
    {"aya:8b",            "CohereForAI/aya-23-8B-GGUF",                "Aya 23 8B — 23 languages, fast",                  8,   false, 8},

    // SmolLM (HuggingFace)
    {"smollm3:3b",        "HuggingFaceTB/SmolLM3-3B-GGUF",             "SmolLM3 3B — HuggingFace, tiny+smart",            3,   false, 3},

    // Jamba (AI21)
    {"jamba1.5-mini",     "ai21labs/AI21-Jamba-1.5-Mini-GGUF",          "Jamba 1.5 Mini 12B — Mamba-Transformer hybrid",   12,  false, 12},

    // GLM (Zhipu AI)
    {"glm4:9b",           "THUDM/glm-4-9b-chat-GGUF",                 "GLM-4 9B — Zhipu AI, multilingual",               9,   false, 9},

    {nullptr, nullptr, nullptr, 0, false, 0}  // sentinel
};

// Find model in catalog by alias (remote catalog takes priority over compiled-in)
static const model_entry * find_catalog_model(const std::string & name) {
    // Check remote catalog first (newer entries override compiled-in)
    ensure_catalog_loaded();
    for (auto & e : g_remote_catalog) {
        if (name == e.alias) return &e;
        std::string alias(e.alias);
        auto colon = alias.find(':');
        if (colon != std::string::npos && name == alias.substr(0, colon)) {
            return &e;
        }
    }
    // Fall back to compiled-in catalog
    for (int i = 0; MODEL_CATALOG[i].alias; i++) {
        if (name == MODEL_CATALOG[i].alias) return &MODEL_CATALOG[i];
        std::string alias(MODEL_CATALOG[i].alias);
        auto colon = alias.find(':');
        if (colon != std::string::npos && name == alias.substr(0, colon)) {
            return &MODEL_CATALOG[i];
        }
    }
    return nullptr;
}

// Recommend quant based on available VRAM + RAM
static std::string recommend_quant(int model_size_b, size_t vram_mb) {
    // Size estimates per quant level (GB per billion params)
    // IQ2_XS: ~0.27, Q2_K: ~0.35, Q3_K: ~0.45, Q4_K_M: ~0.55, Q5_K_M: ~0.65, Q8_0: ~1.1
    float gb_per_b_q8  = 1.1f;
    float gb_per_b_q4  = 0.55f;
    float gb_per_b_q3  = 0.45f;
    float gb_per_b_q2  = 0.35f;
    float gb_per_b_iq2 = 0.27f;

    float q8_size  = model_size_b * gb_per_b_q8;
    float q4_size  = model_size_b * gb_per_b_q4;
    float q3_size  = model_size_b * gb_per_b_q3;
    float q2_size  = model_size_b * gb_per_b_q2;
    float iq2_size = model_size_b * gb_per_b_iq2;

    float vram_gb = vram_mb / 1024.0f;

    // Pick the best quant that fits in VRAM (80% budget for KV cache overhead)
    if (q8_size  < vram_gb * 0.8f) return "Q8_0";
    if (q4_size  < vram_gb * 0.8f) return "Q4_K_M";
    if (q3_size  < vram_gb * 0.8f) return "Q3_K_S";
    if (q2_size  < vram_gb * 0.8f) return "Q2_K";
    if (iq2_size < vram_gb * 0.8f) return "IQ2_XS";

    // Model won't fit in VRAM at any quant - use Q4_K_M with CPU offload
    // (the hw_planner will handle partial GPU offload at runtime)
    return "Q4_K_M";
}

static int cmd_models() {
    printf("\nAvailable models:\n\n");
    printf("  %-20s %-6s %-50s\n", "NAME", "SIZE", "DESCRIPTION");
    printf("  %-20s %-6s %-50s\n", "----", "----", "-----------");
    for (int i = 0; MODEL_CATALOG[i].alias; i++) {
        auto & m = MODEL_CATALOG[i];
        char size[16];
        if (m.is_moe) {
            snprintf(size, sizeof(size), "%dB/%dB", m.active_b, m.size_b);
        } else {
            snprintf(size, sizeof(size), "%dB", m.size_b);
        }
        printf("  %-20s %-6s %s\n", m.alias, size, m.description);
    }
    printf("\nPull a model:  alphachat pull <name>\n");
    printf("               alphachat pull owner/repo:Q4_K_M   (any HuggingFace GGUF)\n");
    printf("               alphachat pull <name>              (auto-searches HF + Ollama)\n");
    printf("Run a model:   alphachat run <name>\n\n");
    return 0;
}

static int cmd_pull(const std::string & model_name, const std::string & quant) {
    // Auto-refresh catalog if stale
    if (catalog_is_stale()) {
        fetch_catalog(true);  // quiet, best-effort
    }

    // Check if already exists
    std::string existing = resolve_model(model_name);
    if (!existing.empty()) {
        printf("Model already available: %s\n", existing.c_str());
        return 0;
    }

    // Check catalog for short aliases (e.g., "qwen3.6" -> HF repo)
    auto * catalog_entry = find_catalog_model(model_name);
    if (catalog_entry && !is_hf_repo(model_name)) {
        std::string repo = catalog_entry->hf_repo;
        std::string tag = quant;
        if (tag.empty()) {
            // Smart quant selection based on VRAM
            ggml_backend_load_all();
            size_t best_vram = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                    size_t free_bytes = 0, total_bytes = 0;
                    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
                    if (free_bytes > best_vram) best_vram = free_bytes;
                }
            }
            tag = recommend_quant(catalog_entry->size_b, best_vram / (1024 * 1024));
            printf("Auto-selected quant: %s (based on %.1f GB VRAM)\n",
                   tag.c_str(), best_vram / (1024.0 * 1024.0 * 1024.0));
        }
        printf("Resolving %s → %s:%s\n", model_name.c_str(), repo.c_str(), tag.c_str());
        return cmd_pull(repo + ":" + tag, "");
    }

    printf("Pulling %s...\n", model_name.c_str());

    // Route 1: HuggingFace repo (owner/repo or owner/repo:tag)
    if (is_hf_repo(model_name)) {
        printf("Detected HuggingFace repo: %s\n", model_name.c_str());

        common_params_model model_params;
        std::string repo = model_name;
        std::string tag = quant.empty() ? "Q4_K_M" : quant;

        // Parse tag from repo if present (owner/repo:tag)
        auto colon = repo.find(':');
        if (colon != std::string::npos) {
            tag = repo.substr(colon + 1);
            repo = repo.substr(0, colon);
        }

        model_params.hf_repo = repo + ":" + tag;

        printf("Downloading from HuggingFace: %s (quant: %s)\n", repo.c_str(), tag.c_str());

        common_download_opts opts;
        // Use HF token from environment if available
        const char * hf_token = getenv("HF_TOKEN");
        if (!hf_token) hf_token = getenv("HUGGING_FACE_HUB_TOKEN");
        if (hf_token) opts.bearer_token = hf_token;

        auto result = common_download_model(model_params, opts);
        if (!result.model_path.empty()) {
            printf("Downloaded: %s\n", result.model_path.c_str());

            // Symlink or copy to ~/.alphachat/models/ for easy access
            std::string models_dir = get_models_dir();
            fs::create_directories(models_dir);

            // Create a friendly name
            std::string friendly = repo;
            std::replace(friendly.begin(), friendly.end(), '/', '-');
            std::string dest = models_dir + "/" + friendly + "-" + tag + ".gguf";

            if (!fs::exists(dest)) {
                try {
                    fs::create_symlink(result.model_path, dest);
                    printf("Linked: %s\n", dest.c_str());
                } catch (...) {
                    // Symlink failed (Windows or permissions) - that's ok, HF cache works
                    printf("Available via HF cache: %s\n", result.model_path.c_str());
                }
            }
            return 0;
        } else {
            fprintf(stderr, "Failed to download from HuggingFace.\n");
            fprintf(stderr, "Check: repo exists, GGUF files available, network connected.\n");
            if (!hf_token) {
                fprintf(stderr, "For gated models, set HF_TOKEN environment variable.\n");
            }
            return 1;
        }
    }

    // Route 2: Try common HuggingFace GGUF repo patterns
    // Many models have community GGUFs at predictable locations
    {
        // Normalize name: "deepseek-v4-flash" -> "DeepSeek-V4-Flash" (title case with hyphens)
        // Also try the name as-is for exact matches
        std::string base = model_name;
        auto colon = base.find(':');
        std::string size_tag;
        if (colon != std::string::npos) {
            size_tag = base.substr(colon + 1);
            base = base.substr(0, colon);
        }

        // Build candidate HF repos to try
        std::vector<std::string> candidates;

        // Common GGUF providers
        const char * providers[] = { "unsloth", "bartowski", "lmstudio-community", nullptr };
        for (int p = 0; providers[p]; p++) {
            candidates.push_back(std::string(providers[p]) + "/" + base + "-GGUF");
        }

        // Try with title case (deepseek-v4-flash -> DeepSeek-V4-Flash)
        std::string title_case;
        bool cap_next = true;
        for (char c : base) {
            if (c == '-') { title_case += '-'; cap_next = true; }
            else if (cap_next) { title_case += toupper(c); cap_next = false; }
            else { title_case += c; }
        }
        if (title_case != base) {
            for (int p = 0; providers[p]; p++) {
                candidates.push_back(std::string(providers[p]) + "/" + title_case + "-GGUF");
            }
        }

        std::string tag = quant.empty() ? "Q4_K_M" : quant;
        for (auto & repo : candidates) {
            printf("Trying %s:%s...\n", repo.c_str(), tag.c_str());

            common_params_model model_params;
            model_params.hf_repo = repo + ":" + tag;

            common_download_opts opts;
            const char * hf_token = getenv("HF_TOKEN");
            if (!hf_token) hf_token = getenv("HUGGING_FACE_HUB_TOKEN");
            if (hf_token) opts.bearer_token = hf_token;

            auto result = common_download_model(model_params, opts);
            if (!result.model_path.empty()) {
                printf("Downloaded: %s\n", result.model_path.c_str());

                std::string models_dir = get_models_dir();
                fs::create_directories(models_dir);
                std::string friendly = base + "-" + tag + ".gguf";
                std::string dest = models_dir + "/" + friendly;
                if (!fs::exists(dest)) {
                    try {
                        fs::create_symlink(result.model_path, dest);
                        printf("Linked: %s\n", dest.c_str());
                    } catch (...) {
                        printf("Available via HF cache: %s\n", result.model_path.c_str());
                    }
                }
                return 0;
            }
        }
    }

    // Route 3: Ollama fallback
#ifdef _WIN32
    std::string cmd = "ollama pull " + model_name + " >nul 2>&1";
#else
    std::string cmd = "ollama pull " + model_name + " >/dev/null 2>&1";
#endif
    int ret = system(cmd.c_str());
    if (ret == 0) {
        std::string resolved = resolve_model(model_name);
        if (!resolved.empty()) {
            printf("Available at: %s\n", resolved.c_str());
            return 0;
        }
    }

    // Route 4: Nothing worked
    fprintf(stderr, "Could not pull '%s'.\n\n", model_name.c_str());
    fprintf(stderr, "Try one of:\n");
    fprintf(stderr, "  alphachat pull owner/repo          (HuggingFace GGUF repo)\n");
    fprintf(stderr, "  alphachat pull owner/repo:Q4_K_M   (specific quantization)\n");
    fprintf(stderr, "  alphachat pull model:tag            (Ollama model)\n");
    return 1;
}

// Server Mode

static int cmd_serve(const std::string & model_name, int port,
                     int n_gpu_layers, int expert_cache_mb,
                     bool ngl_explicit, bool quiet, int ctx_extended = 0) {
    std::string path = resolve_model(model_name);
    if (path.empty()) {
        fprintf(stderr, "Model not found: %s\n", model_name.c_str());
        return 2;
    }

    // Run hw_planner for optimal GPU config
    ggml_backend_load_all();
    alpha_hw_plan hw_plan = alpha_hw_plan_create(path.c_str(), ngl_explicit ? n_gpu_layers : -1);
    if (!quiet) alpha_hw_plan_print(&hw_plan);

    if (expert_cache_mb <= 0 && hw_plan.expert_cache_mb > 0) {
        expert_cache_mb = hw_plan.expert_cache_mb;
    }

    // Find alphallama in same directory as alphachat
    std::string self_dir = get_self_dir();
#ifdef _WIN32
    std::string server_bin = self_dir + "\\alphallama.exe";
#else
    std::string server_bin = self_dir + "/alphallama";
#endif

    if (!fs::exists(server_bin)) {
        fprintf(stderr, "Error: alphallama not found at %s\n", server_bin.c_str());
        fprintf(stderr, "Build it with: cmake --build build --config Release --target alphallama\n");
        return 3;
    }

    // Build command line
    std::vector<std::string> args;
    args.push_back(server_bin);
    args.push_back("-m");
    args.push_back(path);
    args.push_back("--port");
    args.push_back(std::to_string(port));
    args.push_back("-ngl");
    args.push_back(std::to_string(hw_plan.n_gpu_layers));
    args.push_back("--ctx-size");
    args.push_back(std::to_string(hw_plan.n_ctx > 0 ? hw_plan.n_ctx : 4096));

    // Flash attention (requires explicit value)
    args.push_back("--flash-attn");
    args.push_back("auto");

    // MoE expert offload to CPU (only when model doesn't fit entirely in GPU)
    // Skip for --mode standard (vanilla dispatch handles experts natively)
    if (expert_cache_mb > 0 && hw_plan.strategy != ALPHA_HW_DIRECT
        && g_alpha_mode != ALPHA_MODE_STANDARD) {
        args.push_back("--override-tensor");
        args.push_back("blk\\..*\\.ffn_(gate|up|down)_exps.*=CPU");
    }

    // KV quantization
    if (hw_plan.kv_type_k != GGML_TYPE_F16) {
        args.push_back("--cache-type-k");
        args.push_back(ggml_type_name(hw_plan.kv_type_k));
    }
    if (hw_plan.kv_type_v != GGML_TYPE_F16) {
        args.push_back("--cache-type-v");
        args.push_back(ggml_type_name(hw_plan.kv_type_v));
    }

    // Always suppress alphallama's verbose logs - they leak model architecture,
    // tensor types, GPU details, buffer sizes. Our summary line is enough.
    args.push_back("--log-disable");

    // Extended context via KV-chunk system
    if (ctx_extended > 0) {
        args.push_back("--ctx-extended");
        args.push_back(std::to_string(ctx_extended));
    }

    // Forward --mode to alphallama
    args.push_back("--mode");
    switch (g_alpha_mode) {
        case ALPHA_MODE_STANDARD: args.push_back("standard"); break;
        case ALPHA_MODE_CPU_MOE:  args.push_back("offload"); break;
    }

    // Bind to localhost only (security)
    args.push_back("--host");
    args.push_back("127.0.0.1");

    // Print summary (minimal - no internal details)
    fprintf(stderr, "Starting server: %s on port %d\n", model_name.c_str(), port);
    fprintf(stderr, "  API: http://127.0.0.1:%d/v1/chat/completions\n", port);
    fprintf(stderr, "Loading model...\n");
    fprintf(stderr, "\n");

    // Build command string for system()
    // On Windows, wrap each arg in quotes to handle backslashes and spaces in paths
    auto build_cmd_line = [&]() -> std::string {
        std::string cl;
#ifndef _WIN32
        // Ensure alphallama finds shared libs in its own directory.
        // LD_LIBRARY_PATH may not be inherited if alphachat was run with
        // "LD_LIBRARY_PATH=x ./alphachat" (not exported). Prepend explicitly.
        cl = "LD_LIBRARY_PATH=\"" + self_dir + ":$LD_LIBRARY_PATH\" ";
#endif
        for (size_t i = 0; i < args.size(); i++) {
            if (i > 0) cl += " ";
            cl += "\"" + args[i] + "\"";
        }
#ifdef _WIN32
        // Windows system() passes to cmd.exe - need outer quotes for the whole command
        cl = "\"" + cl + "\"";
#endif
        return cl;
    };
    std::string cmd_line = build_cmd_line();

    // Graduated fallback: GPU full -> GPU partial -> GPU minimal -> CPU
    // Use actual layer count (not ngl which can be 999)
    int actual_layers = hw_plan.n_layer > 0 ? hw_plan.n_layer : 40;
    int planned_ngl = hw_plan.n_gpu_layers > actual_layers ? actual_layers : hw_plan.n_gpu_layers;
    int ngl_attempts[] = {
        planned_ngl,                    // Level 0: hw_planner recommendation
        actual_layers / 2,              // Level 1: half layers
        actual_layers / 4,              // Level 2: quarter layers
        1,                              // Level 3: minimal (output layer only)
        0,                              // Level 4: pure CPU
    };
    int n_attempts = (planned_ngl > 0) ? 5 : 1;

    for (int attempt = 0; attempt < n_attempts; attempt++) {
        int try_ngl = ngl_attempts[attempt];

        // Update -ngl in args
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i] == "-ngl" && i + 1 < args.size()) {
                args[i + 1] = std::to_string(try_ngl);
                break;
            }
        }

        // Reduce context on later attempts to fit in memory
        if (attempt >= 2) {
            for (size_t i = 0; i < args.size(); i++) {
                if (args[i] == "--ctx-size" && i + 1 < args.size()) {
                    args[i + 1] = (attempt >= 4) ? "512" : "2048";
                    break;
                }
            }
        }

        // Rebuild command line
        cmd_line = build_cmd_line();

        if (attempt > 0) {
            fprintf(stderr, "\n[alphachat] Previous attempt failed. Trying %d GPU layers (attempt %d/%d)...\n\n",
                    try_ngl, attempt + 1, n_attempts);
        }

        int rc = system(cmd_line.c_str());

        if (rc == 0) return 0;  // Server ran and exited cleanly (user stopped it)

        // If server crashed quickly (< a few seconds), it's likely an allocation failure
        // Try next fallback level
        if (attempt == n_attempts - 1) {
            fprintf(stderr, "\n[alphachat] All GPU levels failed. Server could not start.\n");
            return rc;
        }
    }

    return 1;
}

// Daemon Serve Mode

// Disable SSL before including httplib header (not available on this build)
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.h"

// State for the daemon: tracks currently loaded model + alphallama subprocess
struct daemon_state {
    std::string current_model;       // currently loaded model name
    std::string current_model_path;  // resolved GGUF path
    int backend_port;                // alphallama internal port
    int user_ngl = -1;              // user-specified GPU layers (-1 = auto)
#ifdef _WIN32
    HANDLE backend_process;
#else
    pid_t backend_pid;
#endif
    bool backend_running;
};

static void daemon_stop_backend(daemon_state & state) {
    if (!state.backend_running) return;

#ifdef _WIN32
    if (state.backend_process != INVALID_HANDLE_VALUE) {
        TerminateProcess(state.backend_process, 0);
        WaitForSingleObject(state.backend_process, 5000);
        CloseHandle(state.backend_process);
        state.backend_process = INVALID_HANDLE_VALUE;
    }
#else
    if (state.backend_pid > 0) {
        kill(state.backend_pid, SIGTERM);
        int wstatus;
        waitpid(state.backend_pid, &wstatus, 0);
        state.backend_pid = 0;
    }
#endif
    state.backend_running = false;
    state.current_model.clear();
    state.current_model_path.clear();
}

static bool daemon_start_backend(daemon_state & state, const std::string & model_name) {
    // Resolve model
    std::string path = resolve_model(model_name);
    if (path.empty()) return false;

    // If same model already running, skip
    if (state.backend_running && state.current_model_path == path) return true;

    // Stop previous backend
    daemon_stop_backend(state);

    // Run hw_planner
    alpha_hw_plan hw_plan = alpha_hw_plan_create(path.c_str(), state.user_ngl);

    // Pick internal port
    state.backend_port = 39100 + (rand() % 100);

    // Find alphallama
    std::string self_dir = get_self_dir();
#ifdef _WIN32
    std::string server_bin = self_dir + "\\alphallama.exe";
#else
    std::string server_bin = self_dir + "/alphallama";
#endif
    if (!fs::exists(server_bin)) return false;

    // Build args
    std::string cmd_line;
    std::vector<std::string> args = {
        server_bin,
        "-m", path,
        "--port", std::to_string(state.backend_port),
        "-ngl", std::to_string(hw_plan.n_gpu_layers),
        "--ctx-size", std::to_string(hw_plan.n_ctx > 0 ? hw_plan.n_ctx : 4096),
        "--flash-attn", "auto",
        "--host", "127.0.0.1",
        "--log-disable",
        "--jinja",  // enable Jinja templates for think:false support
    };

    // MoE expert offload: pin experts to CPU for any non-DIRECT strategy
    // (includes LUT_DIRECT, CPU_MOE, SELF_SPEC - all need experts on host)
    if (hw_plan.strategy != ALPHA_HW_DIRECT && hw_plan.n_expert > 1) {
        args.push_back("--override-tensor");
        args.push_back("blk\\..*\\.ffn_(gate|up|down)_exps.*=CPU");
        // Expert cache swaps tensor dimensions (ne[2]=K, not n_expert).
        // Graph reuse with stale dimensions crashes on subsequent requests.
#ifdef _WIN32
        _putenv_s("LLAMA_GRAPH_REUSE_DISABLE", "1");
#else
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
#endif
    }

    // KV quant
    if (hw_plan.kv_type_k != GGML_TYPE_F16) {
        args.push_back("--cache-type-k");
        args.push_back(ggml_type_name(hw_plan.kv_type_k));
    }
    if (hw_plan.kv_type_v != GGML_TYPE_F16) {
        args.push_back("--cache-type-v");
        args.push_back(ggml_type_name(hw_plan.kv_type_v));
    }

    // Alpha Optimized: for daemon mode, self-spec isn't available in alphallama server.
    // Instead, use the draft ngl for offloading (still faster than pure CPU in most cases).
    // For full Alpha Optimized speed, use: alphachat run <model> --self-spec-skip N
    if (hw_plan.strategy == ALPHA_HW_SELF_SPEC_DENSE) {
        fprintf(stderr, "[alpha-optimized] Model partially fits GPU (skip=%d, draft_ngl=%d).\n",
                hw_plan.self_spec_skip, hw_plan.self_spec_draft_ngl);
        fprintf(stderr, "[alpha-optimized] For best speed, use: alphachat run %s --self-spec-skip %d --self-spec-m %d\n",
                model_name.c_str(), hw_plan.self_spec_skip, hw_plan.self_spec_m);
    }

    fprintf(stderr, "Loading model: %s\n", model_name.c_str());

#ifdef _WIN32
    // Create a Job Object so the alphallama child is killed when the parent dies
    static HANDLE s_job = nullptr;
    if (!s_job) {
        s_job = CreateJobObjectA(nullptr, nullptr);
        if (s_job) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli = {};
            jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(s_job, JobObjectExtendedLimitInformation, &jeli, sizeof(jeli));
        }
    }
#endif

    // Helper: launch the backend process with current args
    auto launch_backend = [&]() -> bool {
        // Build command line from args
        cmd_line.clear();
        for (size_t i = 0; i < args.size(); i++) {
            if (i > 0) cmd_line += " ";
            cmd_line += "\"" + args[i] + "\"";
        }
#ifdef _WIN32
        // Use lpApplicationName to avoid command-line quoting issues
        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;
        PROCESS_INFORMATION pi = {};
        std::vector<char> buf(cmd_line.begin(), cmd_line.end());
        buf.push_back('\0');
        if (!CreateProcessA(server_bin.c_str(), buf.data(), nullptr, nullptr, FALSE,
                            CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
            return false;
        }
        if (s_job) AssignProcessToJobObject(s_job, pi.hProcess);
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        state.backend_process = pi.hProcess;
#else
        pid_t pid = fork();
        if (pid == 0) {
            // Die when the parent exits so the engine process is not orphaned (Linux only; elsewhere the check
            // below catches a parent that already exited)
#if defined(__linux__)
            prctl(PR_SET_PDEATHSIG, SIGTERM);
#endif
            // Re-check: parent may have died between fork and prctl
            if (getppid() == 1) _exit(0);

            std::vector<const char *> c_args;
            c_args.push_back("alphachat");
            for (size_t i = 1; i < args.size(); i++) c_args.push_back(args[i].c_str());
            c_args.push_back(nullptr);
            execv(server_bin.c_str(), (char * const *)c_args.data());
            _exit(1);
        }
        if (pid < 0) return false;
        state.backend_pid = pid;
#endif
        state.backend_running = true;
        return true;
    };

    // Helper: update -ngl value in args
    auto set_ngl = [&](int ngl) {
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i] == "-ngl" && i + 1 < args.size()) {
                args[i + 1] = std::to_string(ngl);
                return;
            }
        }
    };

    // Helper: update --ctx-size in args
    auto set_ctx = [&](int ctx) {
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i] == "--ctx-size" && i + 1 < args.size()) {
                args[i + 1] = std::to_string(ctx);
                return;
            }
        }
    };

    // Helper: poll health endpoint, also detect child exit
    auto wait_for_health = [&](int timeout_sec) -> bool {
        httplib::Client probe("127.0.0.1", state.backend_port);
        probe.set_connection_timeout(1);
        probe.set_read_timeout(2);

        for (int i = 0; i < timeout_sec; i++) {
#ifdef _WIN32
            DWORD exit_code = 0;
            if (GetExitCodeProcess(state.backend_process, &exit_code) && exit_code != STILL_ACTIVE) {
                return false;  // child died
            }
#else
            int wstatus;
            if (waitpid(state.backend_pid, &wstatus, WNOHANG) > 0) {
                state.backend_pid = 0;
                return false;  // child died
            }
#endif
            auto res = probe.Get("/health");
            if (res && res->status == 200) return true;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return false;  // timeout
    };

    // Graduated fallback: GPU full -> half -> quarter -> minimal -> CPU-only
    int actual_layers = hw_plan.n_layer > 0 ? hw_plan.n_layer : 40;
    int planned_ngl = hw_plan.n_gpu_layers > actual_layers ? actual_layers : hw_plan.n_gpu_layers;
    int ngl_attempts[] = { planned_ngl, actual_layers / 2, actual_layers / 4, 1, 0, 0 };
    int n_attempts;
    if (planned_ngl > 0) {
        n_attempts = 6;  // GPU full -> half -> quarter -> 1 -> 0 -> 0+device=none
    } else {
        // hw_planner says CPU-only - try ngl=0 with GPU, then pure CPU
        ngl_attempts[0] = 0;
        ngl_attempts[1] = 0;
        n_attempts = 2;
    }

    state.current_model = model_name;
    state.current_model_path = path;

    // Helper: add or remove a flag with value from args
    auto set_arg = [&](const std::string & flag, const std::string & value) {
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i] == flag && i + 1 < args.size()) {
                args[i + 1] = value;
                return;
            }
        }
        args.push_back(flag);
        args.push_back(value);
    };
    auto add_flag = [&](const std::string & flag) {
        for (const auto & a : args) {
            if (a == flag) return;
        }
        args.push_back(flag);
    };

    for (int attempt = 0; attempt < n_attempts; attempt++) {
        int try_ngl = ngl_attempts[attempt];
        set_ngl(try_ngl);

        // Reduce context and resources on later attempts to fit in memory
        bool is_last = (attempt == n_attempts - 1);
        if (attempt >= 2 || is_last) {
            set_ctx(is_last ? 512 : 2048);
            set_arg("--parallel", "1");
        }
        if (is_last) {
            set_arg("--device", "none");  // Force pure CPU (no CUDA compute buffers)
            add_flag("--kv-unified");
            // Remove KV quant - some models crash with Q4 KV cache (ggml contiguous assert)
            for (size_t i = 0; i < args.size(); ) {
                if ((args[i] == "--cache-type-k" || args[i] == "--cache-type-v") && i + 1 < args.size()) {
                    args.erase(args.begin() + i, args.begin() + i + 2);
                } else { i++; }
            }
        }

        if (attempt > 0) {
            fprintf(stderr, "Retrying with %d GPU layers (attempt %d/%d)%s...\n",
                    try_ngl, attempt + 1, n_attempts,
                    is_last ? " [CPU-only]" : "");
        }

        if (!launch_backend()) continue;

        // Timeout scales with model size: large models need mmap load time
        // 76GB model = ~65s on mmap, 168GB = ~120s
        int base_timeout = is_last ? 120 : 15;
        int size_timeout = hw_plan.model_size_mb / 500;  // ~2s per GB
        int timeout = std::max(base_timeout, size_timeout);
        if (wait_for_health(timeout)) {
            fprintf(stderr, "Model ready: %s (%d GPU layers)\n",
                    state.current_model.c_str(), try_ngl);
            return true;
        }

        // Failed - kill and try next level
        daemon_stop_backend(state);
    }

    fprintf(stderr, "Model load failed after %d attempts: %s\n", n_attempts, model_name.c_str());
    return false;
}

// Extract model name from JSON request body
static std::string extract_model_from_body(const std::string & body) {
    // Simple JSON parse: find "model":"value"
    auto pos = body.find("\"model\"");
    if (pos == std::string::npos) return "";
    pos = body.find("\"", pos + 7);
    if (pos == std::string::npos) return "";
    pos++;
    auto end = body.find("\"", pos);
    if (end == std::string::npos) return "";
    return body.substr(pos, end - pos);
}

// List models in ~/.alphachat/models/ as Ollama-compatible JSON
static std::string list_models_json() {
    std::string models_dir = get_models_dir();
    std::string json = "{\"models\":[";
    bool first = true;

    if (fs::exists(models_dir)) {
        for (auto & entry : fs::directory_iterator(models_dir)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) continue;
            std::string fn = entry.path().filename().string();
            if (fn.size() < 5 || fn.substr(fn.size() - 5) != ".gguf") continue;

            if (!first) json += ",";
            first = false;

            std::string name = fn.substr(0, fn.size() - 5);
            size_t size_bytes = 0;
            try { size_bytes = fs::file_size(entry.path()); } catch (...) {}

            // Get modified time (use stat for portability - fs::last_write_time clock_cast needs C++20)
            std::string modified_at = "1970-01-01T00:00:00.000000000Z";
            {
                struct stat st;
                if (stat(entry.path().string().c_str(), &st) == 0) {
                    struct tm * t = gmtime(&st.st_mtime);
                    char tbuf[64];
                    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%S.000000000Z", t);
                    modified_at = tbuf;
                }
            }

            json += "{\"name\":\"" + name + "\"";
            json += ",\"model\":\"" + name + "\"";
            json += ",\"modified_at\":\"" + modified_at + "\"";
            json += ",\"size\":" + std::to_string(size_bytes);
            json += ",\"digest\":\"sha256:local\"";
            json += ",\"details\":{\"format\":\"gguf\",\"family\":\"unknown\",\"parameter_size\":\"unknown\",\"quantization_level\":\"unknown\"}";
            json += "}";
        }
    }

    json += "]}";
    return json;
}

static int cmd_serve_daemon(int port, int user_ngl = -1) {
    // Suppress Vulkan/CUDA device enumeration output
    llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    ggml_backend_load_all();

    daemon_state state = {};
    state.backend_running = false;
    state.backend_port = 0;
    state.user_ngl = user_ngl;
#ifdef _WIN32
    state.backend_process = INVALID_HANDLE_VALUE;
#else
    state.backend_pid = 0;
#endif

    httplib::Server svr;

    // Health endpoint
    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // AlphaLlama: list models
    svr.Get("/api/tags", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(list_models_json(), "application/json");
    });

    // OpenAI: list models
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(list_models_json(), "application/json");
    });

    // Ollama: running models
    svr.Get("/api/ps", [&](const httplib::Request &, httplib::Response & res) {
        std::string json = "{\"models\":[";
        if (state.backend_running && !state.current_model.empty()) {
            json += "{\"name\":\"" + state.current_model + "\"";
            json += ",\"model\":\"" + state.current_model + "\"";
            size_t sz = 0;
            try { if (!state.current_model_path.empty()) sz = fs::file_size(state.current_model_path); } catch (...) {}
            json += ",\"size\":" + std::to_string(sz);
            json += ",\"digest\":\"local\"";
            json += ",\"details\":{\"format\":\"gguf\",\"family\":\"unknown\",\"parameter_size\":\"unknown\",\"quantization_level\":\"unknown\"}";
            json += "}";
        }
        json += "]}";
        res.set_content(json, "application/json");
    });

    // Ollama: show model info
    svr.Post("/api/show", [&](const httplib::Request & req, httplib::Response & res) {
        std::string name;
        auto npos = req.body.find("\"name\"");
        if (npos != std::string::npos) {
            auto qs = req.body.find("\"", npos + 7);
            if (qs != std::string::npos) {
                qs++;
                auto qe = req.body.find("\"", qs);
                if (qe != std::string::npos) name = req.body.substr(qs, qe - qs);
            }
        }
        if (name.empty()) name = state.current_model;
        std::string path = resolve_model(name);
        if (path.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"model not found: " + name + "\"}", "application/json");
            return;
        }
        size_t sz = 0;
        try { sz = fs::file_size(path); } catch (...) {}
        // Use model name instead of absolute path to avoid leaking filesystem info
        std::string display_name = name.empty() ? fs::path(path).filename().string() : name;
        std::string json = "{\"modelfile\":\"FROM " + json_escape(display_name) + "\"";
        json += ",\"parameters\":\"\"";
        json += ",\"template\":\"\"";
        json += ",\"details\":{\"format\":\"gguf\",\"family\":\"unknown\"";
        json += ",\"parameter_size\":\"" + std::to_string(sz / (1024*1024)) + "MB\"";
        json += ",\"quantization_level\":\"unknown\"}";
        json += ",\"model_info\":{\"general.file_size\":" + std::to_string(sz) + "}}";
        res.set_content(json, "application/json");
    });

    // Ollama: pull model
    svr.Post("/api/pull", [&](const httplib::Request & req, httplib::Response & res) {
        std::string name;
        auto npos = req.body.find("\"name\"");
        if (npos != std::string::npos) {
            auto qs = req.body.find("\"", npos + 7);
            if (qs != std::string::npos) {
                qs++;
                auto qe = req.body.find("\"", qs);
                if (qe != std::string::npos) name = req.body.substr(qs, qe - qs);
            }
        }
        if (name.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"name required\"}", "application/json");
            return;
        }
        // Check if already available
        std::string path = resolve_model(name);
        if (!path.empty()) {
            res.set_content("{\"status\":\"success\",\"digest\":\"local\"}", "application/json");
            return;
        }
        // Model not found locally - return error (alphachat pull handles downloads via CLI)
        res.status = 404;
        res.set_content("{\"error\":\"model not available locally. Use 'alphachat pull " + name + "' from CLI.\"}", "application/json");
    });

    // Ollama: delete model
    svr.Delete("/api/delete", [&](const httplib::Request & req, httplib::Response & res) {
        std::string name;
        auto npos = req.body.find("\"name\"");
        if (npos != std::string::npos) {
            auto qs = req.body.find("\"", npos + 7);
            if (qs != std::string::npos) {
                qs++;
                auto qe = req.body.find("\"", qs);
                if (qe != std::string::npos) name = req.body.substr(qs, qe - qs);
            }
        }
        if (name.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"name required\"}", "application/json");
            return;
        }
        std::string path = resolve_model(name);
        if (path.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"model not found\"}", "application/json");
            return;
        }
        // Only delete from models dir (not HF cache or system paths)
        std::string models_dir = get_models_dir();
        if (path.find(models_dir) != 0) {
            res.status = 403;
            res.set_content("{\"error\":\"can only delete models in " + models_dir + "\"}", "application/json");
            return;
        }
        if (state.backend_running && state.current_model == name) {
            daemon_stop_backend(state);
        }
        std::error_code ec;
        fs::remove(path, ec);
        if (ec) {
            res.status = 500;
            res.set_content("{\"error\":\"failed to delete: " + ec.message() + "\"}", "application/json");
        } else {
            res.status = 200;
        }
    });

    // Ollama: copy model
    svr.Post("/api/copy", [&](const httplib::Request & req, httplib::Response & res) {
        std::string source, destination;
        auto spos = req.body.find("\"source\"");
        if (spos != std::string::npos) {
            auto qs = req.body.find("\"", spos + 9);
            if (qs != std::string::npos) { qs++; auto qe = req.body.find("\"", qs); if (qe != std::string::npos) source = req.body.substr(qs, qe - qs); }
        }
        auto dpos = req.body.find("\"destination\"");
        if (dpos != std::string::npos) {
            auto qs = req.body.find("\"", dpos + 14);
            if (qs != std::string::npos) { qs++; auto qe = req.body.find("\"", qs); if (qe != std::string::npos) destination = req.body.substr(qs, qe - qs); }
        }
        if (source.empty() || destination.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"source and destination required\"}", "application/json");
            return;
        }
        std::string src_path = resolve_model(source);
        if (src_path.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"source model not found\"}", "application/json");
            return;
        }
        std::string models_dir = get_models_dir();
        std::string dest_path = models_dir + "/" + destination + ".gguf";
        std::error_code ec;
        fs::create_symlink(src_path, dest_path, ec);
        if (ec) {
            // Fallback: try copy if symlink fails
            fs::copy_file(src_path, dest_path, fs::copy_options::skip_existing, ec);
            if (ec) {
                res.status = 500;
                res.set_content("{\"error\":\"copy failed: " + ec.message() + "\"}", "application/json");
                return;
            }
        }
        res.status = 200;
    });

    // Proxy handler: forward chat requests to alphallama
    auto proxy_chat = [&](const httplib::Request & req, httplib::Response & res) {
        // Validate JSON body
        if (req.body.empty() || (req.body[0] != '{' && req.body[0] != '[')) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON body\"}", "application/json");
            return;
        }

        // Validate numeric parameters if present
        {
            // max_tokens validation
            auto mt_pos = req.body.find("\"max_tokens\"");
            if (mt_pos != std::string::npos) {
                auto colon = req.body.find(':', mt_pos + 12);
                if (colon != std::string::npos) {
                    int val = atoi(req.body.c_str() + colon + 1);
                    if (val < 0) {
                        res.status = 400;
                        res.set_content("{\"error\":\"max_tokens must be >= 0\"}", "application/json");
                        return;
                    }
                }
            }
            // temperature validation
            auto tp_pos = req.body.find("\"temperature\"");
            if (tp_pos != std::string::npos) {
                auto colon = req.body.find(':', tp_pos + 13);
                if (colon != std::string::npos) {
                    float val = (float)atof(req.body.c_str() + colon + 1);
                    if (val < 0.0f) {
                        res.status = 400;
                        res.set_content("{\"error\":\"temperature must be >= 0\"}", "application/json");
                        return;
                    }
                }
            }
        }

        std::string model = extract_model_from_body(req.body);
        if (model.empty()) {
            // Use currently loaded model
            if (!state.backend_running) {
                res.status = 400;
                res.set_content("{\"error\":\"no model specified and no model loaded\"}", "application/json");
                return;
            }
        } else {
            // Load/swap model if needed
            if (!daemon_start_backend(state, model)) {
                std::string reason;
                std::string rpath = resolve_model(model);
                if (rpath.empty()) {
                    reason = "model not found";
                    res.status = 404;
                } else {
                    reason = "failed to load (possible causes: insufficient GPU memory, unsupported format, or corrupted file)";
                    res.status = 500;
                }
                res.set_content("{\"error\":\"" + json_escape(model) + ": " + reason + "\"}", "application/json");
                return;
            }
        }

        // Determine if this is an Ollama-format request (/api/chat or /api/generate)
        bool is_ollama = (req.path.find("/api/") == 0);
        bool is_generate = (req.path.find("/api/generate") == 0);

        // Helper: extract a JSON string value by key (simple, no nested objects)
        auto json_str = [](const std::string & s, const std::string & key) -> std::string {
            auto kp = s.find("\"" + key + "\"");
            if (kp == std::string::npos) return "";
            auto qs = s.find("\"", kp + key.size() + 2);
            if (qs == std::string::npos) return "";
            qs++;
            size_t qe = qs;
            while (qe < s.size()) {
                if (s[qe] == '\\') { qe += 2; continue; }
                if (s[qe] == '"') break;
                qe++;
            }
            return s.substr(qs, qe - qs);
        };
        // Helper: extract a JSON number value by key
        auto json_num = [](const std::string & s, const std::string & key) -> std::string {
            auto kp = s.find("\"" + key + "\"");
            if (kp == std::string::npos) return "";
            auto cp = s.find(":", kp);
            if (cp == std::string::npos) return "";
            cp++;
            while (cp < s.size() && s[cp] == ' ') cp++;
            size_t end = cp;
            while (end < s.size() && (isdigit(s[end]) || s[end] == '.' || s[end] == '-' || s[end] == 'e' || s[end] == 'E')) end++;
            return s.substr(cp, end - cp);
        };
        // Helper: extract a JSON sub-object by key
        auto json_obj = [](const std::string & s, const std::string & key) -> std::string {
            auto kp = s.find("\"" + key + "\"");
            if (kp == std::string::npos) return "";
            auto bp = s.find("{", kp);
            if (bp == std::string::npos) return "";
            int depth = 0;
            for (size_t i = bp; i < s.size(); i++) {
                if (s[i] == '{') depth++;
                else if (s[i] == '}') { depth--; if (depth == 0) return s.substr(bp, i - bp + 1); }
            }
            return "";
        };
        // Helper: extract a JSON array by key
        auto json_arr = [](const std::string & s, const std::string & key) -> std::string {
            auto kp = s.find("\"" + key + "\"");
            if (kp == std::string::npos) return "";
            auto bp = s.find("[", kp);
            if (bp == std::string::npos) return "";
            int depth = 0;
            for (size_t i = bp; i < s.size(); i++) {
                if (s[i] == '[') depth++;
                else if (s[i] == ']') { depth--; if (depth == 0) return s.substr(bp, i - bp + 1); }
            }
            return "";
        };

        // Extract Ollama options from request body
        std::string options_json = json_obj(req.body, "options");
        std::string opt_num_predict = json_num(options_json, "num_predict");
        std::string opt_temperature = json_num(options_json, "temperature");
        std::string opt_stop = json_arr(req.body, "stop");
        if (opt_stop.empty()) opt_stop = json_arr(options_json, "stop");
        std::string opt_seed = json_num(options_json, "seed");
        std::string opt_top_k = json_num(options_json, "top_k");
        std::string opt_top_p = json_num(options_json, "top_p");

        // Check think flag (Ollama-style)
        bool no_think = false;
        {
            auto tp = req.body.find("\"think\"");
            if (tp != std::string::npos) {
                auto vp = req.body.find("false", tp);
                if (vp != std::string::npos && vp - tp < 20) no_think = true;
            }
        }

        // Translate request body
        std::string proxy_body = req.body;
        if (is_ollama) {
            // Build OpenAI-compatible request from Ollama request
            std::string messages;
            if (is_generate) {
                std::string prompt = json_str(req.body, "prompt");
                messages = "[{\"role\":\"user\",\"content\":\"" + prompt + "\"}]";
            } else {
                messages = json_arr(req.body, "messages");
                if (messages.empty()) messages = "[]";
            }

            proxy_body = "{\"model\":\"" + model + "\","
                "\"messages\":" + messages + ","
                "\"stream\":false";
            if (!opt_num_predict.empty()) proxy_body += ",\"max_tokens\":" + opt_num_predict;
            if (!opt_temperature.empty()) proxy_body += ",\"temperature\":" + opt_temperature;
            if (!opt_stop.empty()) proxy_body += ",\"stop\":" + opt_stop;
            if (!opt_seed.empty()) proxy_body += ",\"seed\":" + opt_seed;
            if (!opt_top_k.empty()) proxy_body += ",\"top_k\":" + opt_top_k;
            if (!opt_top_p.empty()) proxy_body += ",\"top_p\":" + opt_top_p;
            if (no_think) proxy_body += ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
            proxy_body += "}";
        }

        // Always proxy to backend's OpenAI endpoint
        httplib::Client backend("127.0.0.1", state.backend_port);
        backend.set_read_timeout(300);  // 5 min for long generations

        auto backend_res = backend.Post("/v1/chat/completions", proxy_body, "application/json");
        if (backend_res) {
            res.status = backend_res->status;

            if (is_ollama && backend_res->status == 200) {
                // Translate OpenAI response to Ollama format
                auto & body = backend_res->body;

                // Extract content from OpenAI "choices[0].message.content"
                // If content is empty, fall back to reasoning_content (thinking models)
                std::string content;
                std::string reasoning;
                {
                    size_t search_from = 0;
                    auto choices_pos = body.find("\"choices\"");
                    if (choices_pos != std::string::npos) search_from = choices_pos;
                    content = json_str(body.substr(search_from), "content");
                    reasoning = json_str(body.substr(search_from), "reasoning_content");
                }
                // If content is empty but reasoning exists, use reasoning as content
                if (content.empty() && !reasoning.empty()) {
                    content = reasoning;
                }
                std::string role = "assistant";

                // Extract usage stats from OpenAI response
                std::string usage = json_obj(body, "usage");
                std::string prompt_tokens = json_num(usage, "prompt_tokens");
                std::string completion_tokens = json_num(usage, "completion_tokens");

                // Model name (use display name, fallback to path filename)
                std::string model_name = state.current_model;
                if (model_name.empty() && !state.current_model_path.empty()) {
                    model_name = fs::path(state.current_model_path).stem().string();
                }

                // ISO 8601 timestamp
                char timebuf[64];
                {
                    time_t now = time(nullptr);
                    struct tm * t = gmtime(&now);
                    strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%S.000000000Z", t);
                }

                // Build Ollama response with all required fields
                std::string ollama_resp = "{\"model\":\"" + json_escape(model_name) + "\","
                    "\"created_at\":\"" + std::string(timebuf) + "\",";

                if (is_generate) {
                    ollama_resp += "\"response\":\"" + content + "\",";
                } else {
                    ollama_resp += "\"message\":{\"role\":\"" + role + "\",\"content\":\"" + content + "\"},";
                }

                ollama_resp += "\"done\":true,"
                    "\"done_reason\":\"stop\"";

                // Add timing/stats fields (Ollama uses nanoseconds)
                if (!completion_tokens.empty()) {
                    ollama_resp += ",\"eval_count\":" + completion_tokens;
                }
                if (!prompt_tokens.empty()) {
                    ollama_resp += ",\"prompt_eval_count\":" + prompt_tokens;
                }
                // Approximate durations (exact timing is not available through the proxy)
                ollama_resp += ",\"total_duration\":0,\"load_duration\":0,\"prompt_eval_duration\":0,\"eval_duration\":0";

                ollama_resp += "}";
                res.set_content(ollama_resp, "application/json");
            } else {
                res.set_content(backend_res->body, "application/json");
            }
        } else {
            // Backend request failed - check if process died and auto-restart
            bool restarted = false;
            bool process_dead = false;
#ifdef _WIN32
            if (state.backend_process != INVALID_HANDLE_VALUE) {
                DWORD exit_code = 0;
                if (GetExitCodeProcess(state.backend_process, &exit_code) && exit_code != STILL_ACTIVE) {
                    process_dead = true;
                }
            }
#else
            if (state.backend_pid > 0) {
                int wstatus;
                if (waitpid(state.backend_pid, &wstatus, WNOHANG) > 0) {
                    state.backend_pid = 0;
                    process_dead = true;
                }
            }
#endif
            if (process_dead && !state.current_model_path.empty()) {
                fprintf(stderr, "Backend process died — restarting for %s\n",
                        state.current_model.c_str());
                state.backend_running = false;
                std::string saved_model = state.current_model;
                if (daemon_start_backend(state, saved_model)) {
                    // Retry the request once on the restarted backend
                    httplib::Client retry_backend("127.0.0.1", state.backend_port);
                    retry_backend.set_read_timeout(300);
                    auto retry_res = retry_backend.Post("/v1/chat/completions",
                                                        proxy_body, "application/json");
                    if (retry_res) {
                        restarted = true;
                        res.status = retry_res->status;
                        if (is_ollama && retry_res->status == 200) {
                            // Re-translate to Ollama format (simplified - just extract content)
                            std::string content = json_str(retry_res->body, "content");
                            std::string model_name = state.current_model;
                            res.set_content("{\"model\":\"" + json_escape(model_name) +
                                "\",\"response\":\"" + content +
                                "\",\"done\":true,\"done_reason\":\"stop\"}", "application/json");
                        } else {
                            res.set_content(retry_res->body, "application/json");
                        }
                    }
                }
            }
            if (!restarted) {
                res.status = 502;
                res.set_content("{\"error\":\"backend unavailable\"}", "application/json");
            }
        }
    };

    svr.Post("/v1/chat/completions", proxy_chat);
    svr.Post("/v1/completions", proxy_chat);
    svr.Post("/api/chat", proxy_chat);
    svr.Post("/api/generate", proxy_chat);
    svr.Post("/chat/completions", proxy_chat);

    fprintf(stderr, "\nAlphaChat daemon listening on http://127.0.0.1:%d\n", port);
    fprintf(stderr, "  OpenAI API:  http://127.0.0.1:%d/v1/chat/completions\n", port);
    fprintf(stderr, "  Ollama API:  http://127.0.0.1:%d/api/chat\n", port);
    fprintf(stderr, "  Models:      http://127.0.0.1:%d/api/tags\n", port);
    fprintf(stderr, "  Running:     http://127.0.0.1:%d/api/ps\n", port);
    fprintf(stderr, "  Health:      http://127.0.0.1:%d/health\n", port);
    fprintf(stderr, "  Management:  /api/show, /api/pull, /api/delete, /api/copy\n");
    fprintf(stderr, "\nSend a request with any model name — it will be loaded automatically.\n");
    fprintf(stderr, "Press Ctrl+C to stop.\n\n");

    svr.listen("127.0.0.1", port);

    // Cleanup
    daemon_stop_backend(state);
    return 0;
}

// Architecture Validation

struct model_arch_info {
    std::string arch;
    std::string name;
    uint32_t n_layer = 0, n_embd = 0, n_head = 0;
    uint32_t n_expert = 0, n_expert_used = 0, n_ctx = 0;
    int n_tensors = 0;
    std::vector<std::string> tensor_patterns;  // unique patterns from layer 0
    bool arch_supported = false;
    std::string similar_arch;
    float similarity = 0.0f;
};

static model_arch_info validate_model_arch(const std::string & path) {
    model_arch_info info;

    auto * ctx = gguf_init_from_file(path.c_str(), {false, nullptr});
    if (!ctx) return info;

    // Read architecture
    int arch_idx = gguf_find_key(ctx, "general.architecture");
    if (arch_idx >= 0) info.arch = gguf_get_val_str(ctx, arch_idx);

    int name_idx = gguf_find_key(ctx, "general.name");
    if (name_idx >= 0) info.name = gguf_get_val_str(ctx, name_idx);

    // Read hparams
    auto get_u32 = [&](const std::string & key) -> uint32_t {
        int idx = gguf_find_key(ctx, key.c_str());
        return idx >= 0 ? gguf_get_val_u32(ctx, idx) : 0;
    };

    info.n_layer = get_u32(info.arch + ".block_count");
    info.n_embd = get_u32(info.arch + ".embedding_length");
    info.n_head = get_u32(info.arch + ".attention.head_count");
    info.n_expert = get_u32(info.arch + ".expert_count");
    info.n_expert_used = get_u32(info.arch + ".expert_used_count");
    info.n_ctx = get_u32(info.arch + ".context_length");
    info.n_tensors = gguf_get_n_tensors(ctx);

    // Extract layer-0 tensor name patterns
    for (int i = 0; i < info.n_tensors; i++) {
        std::string tname = gguf_get_tensor_name(ctx, i);
        if (tname.find("blk.0.") == 0) {
            // Convert to pattern: blk.0.attn_q -> blk.%d.attn_q
            std::string pattern = "blk.%d." + tname.substr(6);
            // Remove .weight suffix
            if (pattern.size() > 7 && pattern.substr(pattern.size()-7) == ".weight")
                pattern = pattern.substr(0, pattern.size()-7);
            info.tensor_patterns.push_back(pattern);
        }
    }

    // Check if arch is supported (compiled-in)
    llm_arch resolved = llm_arch_from_string(info.arch);
    if (resolved != LLM_ARCH_UNKNOWN) {
        info.arch_supported = true;
    }

    // Check runtime registry
    if (!info.arch_supported) {
        alpha_registry reg = alpha_registry_load();
        if (alpha_registry_has_arch(reg, info.arch)) {
            std::string fallback = alpha_registry_get_fallback(reg, info.arch);
            if (!fallback.empty() && llm_arch_from_string(fallback) != LLM_ARCH_UNKNOWN) {
                info.arch_supported = true;
                info.similar_arch = fallback;
            }
        }
    }

    // If still unsupported, try pattern matching against known archs
    // (compare layer-0 tensor patterns with known architectures)
    if (!info.arch_supported && !info.tensor_patterns.empty()) {
        // Standard llama-family patterns
        static const std::vector<std::pair<std::string, std::vector<std::string>>> known_patterns = {
            {"llama", {"blk.%d.attn_q", "blk.%d.attn_k", "blk.%d.attn_v", "blk.%d.attn_output",
                       "blk.%d.ffn_gate", "blk.%d.ffn_up", "blk.%d.ffn_down", "blk.%d.attn_norm"}},
            {"qwen2", {"blk.%d.attn_q", "blk.%d.attn_k", "blk.%d.attn_v", "blk.%d.attn_output",
                       "blk.%d.ffn_gate", "blk.%d.ffn_up", "blk.%d.ffn_down"}},
        };

        for (auto & [arch_name, patterns] : known_patterns) {
            int match = 0;
            for (auto & p : patterns) {
                for (auto & tp : info.tensor_patterns) {
                    if (tp.find(p.substr(7)) != std::string::npos) { match++; break; }
                }
            }
            float sim = (float)match / patterns.size();
            if (sim > info.similarity) {
                info.similarity = sim;
                info.similar_arch = arch_name;
            }
        }
    }

    gguf_free(ctx);
    return info;
}

static int cmd_update() {
    printf("Checking for updates...\n\n");

    // Update model catalog from GitHub
    printf("Model catalog:\n");
    if (fetch_catalog(false)) {
        printf("  %zu models available\n", g_remote_catalog.size());
    } else {
        printf("  Using compiled-in catalog (%d+ models)\n", 90);
    }

    // Update architecture registry
    printf("\nArchitecture registry:\n");
    alpha_registry reg = alpha_registry_load();
    if (reg.version > 0) {
        printf("  v%d (%s), %zu architectures\n", reg.version, reg.updated_at.c_str(), reg.architectures.size());
    } else {
        printf("  Using compiled-in defaults\n");
    }
    return 0;
}

// Main

static void print_usage() {
    printf("\nAlphaChat — Run AI models locally.\n\n");
    printf("Usage:\n");
    printf("  alphachat run <model> [prompt]      Run a model (interactive or one-shot)\n");
    printf("  alphachat pull <model>               Download a model\n");
    printf("  alphachat list                       List local models\n");
    printf("  alphachat show <model>               Show model details\n");
    printf("  alphachat rm <model>                 Remove a local model\n");
    printf("  alphachat bench <model>              Benchmark a model\n");
    printf("  alphachat info                       Show system info\n");
    printf("  alphachat update                     Check for registry updates\n");
    printf("  alphachat serve [--port N]            Start API server (Ollama-compatible)\n");
    printf("\nOptions:\n");
    printf("  -n <tokens>          Max tokens to generate (default: 256)\n");
    printf("  -c <ctx>             Context size\n");
    printf("  --temp <f>           Temperature (default: 0.7)\n");
    printf("  --top-k <n>          Top-K sampling (default: 40, 0=disabled)\n");
    printf("  --top-p <f>          Top-P sampling (default: 0.95, 1.0=disabled)\n");
    printf("  --min-p <f>          Min-P sampling (default: 0, 0=disabled)\n");
    printf("  --seed <n>           Random seed (-1=random, default: 42)\n");
    printf("  --gpu-layers <n>     GPU layers (default: auto)\n");
    printf("  --no-think           Disable thinking/reasoning blocks\n");
    printf("  --json               JSON output mode\n");
    printf("  --flash-attn         Enable flash attention\n");
    printf("  --arch-override <a>  Force architecture (for unsupported models)\n");
    printf("  --quant <type>       Quantization for pull (q4_k_m, q8_0, etc.)\n");
    printf("  --debug              Enable verbose internal logging\n");
    printf("\nExamples:\n");
    printf("  alphachat run qwen3.5:4b \"What is 2+2?\"\n");
    printf("  alphachat run deepseek-v4-flash\n");
    printf("  alphachat pull mixtral:8x7b\n");
    printf("  alphachat bench qwen3.5:4b\n");
    printf("  alphachat info\n");
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string cmd = argv[1];

    // Parse global options
    int n_predict = 256;
    int n_ctx = 0;
    int n_gpu_layers = -1;  // -1 = auto, 0 = explicit CPU-only
    int draft_gpu_layers = -1;  // -1 = same as target
    int self_spec_skip = 0;     // 0 = off, N = skip last N layers for draft
    int self_spec_m = 4;        // draft tokens per self-spec step
    int expert_cache_mb = 0;
    bool expert_cache_explicit = false;
    bool ngl_explicit = false;
    float temperature = 0.7f;
    uint32_t seed = 42;              // --seed N (42 = deterministic, -1 = random)
    bool flash_attn = true;
    bool no_think = false;
    float expert_sticky = 0.0f;       // --expert-sticky (0 = disabled)
    float expert_layer_bias = 0.0f;   // --expert-layer-bias (0 = disabled)
    std::string expert_groups_path;   // --expert-groups (empty = disabled)
    int split_group = 0;              // --split-group (0 = auto/default)
    int topk_experts = 0;             // --top-k-experts N (0 = model default, >0 = reduce)
    bool no_barrier = false;          // --no-barrier (use blocking sync instead of async)
    bool flat_cache = false;          // --flat-cache (force flat cache, no scratch overhead)
    // note: flat_cache is passed via g_flat_cache below
    bool speculative_mode = false;    // --expert-cache-mode speculative
    int self_spec_n = 0;              // --self-spec N or N:K (0 = disabled)
    int self_spec_experts = 0;        // draft expert count (0 = Mode C backbone, >0 = Mode A)
    int cached_spec_n = 0;            // --cached-spec M (0 = disabled, >0 = draft M tokens)
    int tier1_cycles = 0;             // --tier1-cycles N (0 = single-cycle, >0 = two-tier with N cycles)
    std::string mixed_cluster_path;   // --mixed-clusters JSON path
    std::string profile_experts_path; // --profile-experts FILE (binary routing output)
    std::string profile_freq_path;   // --profile-frequency FILE (JSON frequency output)
    std::string load_freq_path;      // --load-frequency FILE (JSON frequency input for cache pre-load)
    std::string draft_model_path;    // --draft <model> for speculative decoding
    int draft_n = 4;                 // --draft-n N (draft tokens per cycle)
    std::vector<std::pair<int, std::vector<int>>> expert_overrides; // --override-experts L:E1,E2,...
    bool log_confidence = false;      // --log-confidence
    bool log_gate_entropy = false;    // --log-gate-entropy
    std::string router_patch_path;    // --router-patch
    std::string quant = "q4_k_m";
    std::string arch_override;
    std::string data_path;
    std::string output_path;
    int lora_rank = 32;
    float lora_lr = 2e-5f;
    int train_epochs = 1;
    int train_seq_len = 256;
    bool use_mezo = false;
    bool use_combined = false;
    bool use_chunked = false;
    int  chunked_K = 0;  // 0 = auto
    bool json_output = false;
    bool json_progress = false;
    bool quiet_mode = false;
    bool from_scratch = false;
    int scratch_warmup_steps = 5000;
    int serve_port = 11434;  // Ollama-compatible default
    int ctx_extended = 0;    // Extended context via KV-chunk (0 = disabled, max 1M)

    // Collect positional args and flags
    std::string model_name;
    std::string prompt;
    std::vector<std::string> extra_args;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            g_top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            g_top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            g_min_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--presence-penalty") == 0 && i + 1 < argc) {
            g_presence_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--frequency-penalty") == 0 && i + 1 < argc) {
            g_frequency_penalty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            int s = atoi(argv[++i]);
            seed = (s < 0) ? (uint32_t)time(NULL) : (uint32_t)s;
        } else if ((strcmp(argv[i], "--gpu-layers") == 0 || strcmp(argv[i], "-ngl") == 0) && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
            ngl_explicit = true;
        } else if ((strcmp(argv[i], "--offload-budget") == 0 || strcmp(argv[i], "--xb") == 0) && i + 1 < argc) {
            expert_cache_mb = atoi(argv[++i]);
            expert_cache_explicit = true;
        } else if (strcmp(argv[i], "--flash-attn") == 0) {
            flash_attn = true;
        } else if (strcmp(argv[i], "--no-flash-attn") == 0) {
            flash_attn = false;
        } else if (strcmp(argv[i], "--no-think") == 0) {
            no_think = true;
        } else if (strcmp(argv[i], "--opt-a") == 0 && i + 1 < argc) {
            expert_sticky = atof(argv[++i]);
        } else if (strcmp(argv[i], "--opt-b") == 0 && i + 1 < argc) {
            expert_layer_bias = atof(argv[++i]);
        } else if (strcmp(argv[i], "--opt-groups") == 0 && i + 1 < argc) {
            expert_groups_path = argv[++i];
        } else if (strcmp(argv[i], "--layer-group") == 0 && i + 1 < argc) {
            split_group = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-k-active") == 0 && i + 1 < argc) {
            topk_experts = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug") == 0) {
#ifdef ALPHA_DEBUG_BUILD
            g_alpha_debug = true;
#endif
        } else if (strcmp(argv[i], "--flat") == 0) {
            flat_cache = true;
            g_flat_cache = true;
        } else if (strcmp(argv[i], "--force-offload") == 0) {
            g_force_cache = true;
            g_flat_cache = true;  // scratch cache breaks at high K - use flat for forced mode
        } else if (strcmp(argv[i], "--cpu-offload") == 0) {
            g_cpu_moe = true;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char * mode = argv[++i];
            if (strcmp(mode, "standard") == 0) {
                g_alpha_mode = ALPHA_MODE_STANDARD;
            } else if (strcmp(mode, "cpu-moe") == 0 || strcmp(mode, "cpu_moe") == 0) {
                g_alpha_mode = ALPHA_MODE_CPU_MOE;
                g_cpu_moe = true;  // cpu-moe mode implies cpu-moe for CLI
            } else {
                fprintf(stderr, "Invalid --mode: %s (use: standard, offload)\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--no-barrier") == 0) {
            no_barrier = true;
        } else if (strcmp(argv[i], "--or") == 0 && i + 1 < argc) {
            // Format: LAYER:E1,E2,E3,... (e.g. 0:50,51,52,53,54,55,56,57)
            const char * spec = argv[++i];
            const char * colon = strchr(spec, ':');
            if (colon) {
                int layer = atoi(spec);
                std::vector<int> ids;
                const char * p = colon + 1;
                while (*p) {
                    ids.push_back(atoi(p));
                    while (*p && *p != ',') p++;
                    if (*p == ',') p++;
                }
                if (!ids.empty()) {
                    expert_overrides.push_back({layer, ids});
                }
            }
        } else if (strcmp(argv[i], "--offload-mode") == 0 && i + 1 < argc) {
            const char * mode = argv[++i];
            if (strcmp(mode, "speculative") == 0) {
                speculative_mode = true;
            } else if (strcmp(mode, "reactive") == 0) {
                speculative_mode = false;
            } else {
                fprintf(stderr, "Unknown --offload-mode: %s\n", mode);
                return 1;
            }
        } else if (strcmp(argv[i], "--draft-skip") == 0 && i + 1 < argc) {
            // Format: N or N:K (N=draft tokens, K=draft expert count, 0=backbone)
            const char * spec_arg = argv[++i];
            const char * colon = strchr(spec_arg, ':');
            if (colon) {
                self_spec_n = atoi(spec_arg);
                self_spec_experts = atoi(colon + 1);
            } else {
                self_spec_n = atoi(spec_arg);
                self_spec_experts = 0;  // Mode C by default
            }
        } else if (strcmp(argv[i], "--draft-cache") == 0 && i + 1 < argc) {
            cached_spec_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--t1c") == 0 && i + 1 < argc) {
            tier1_cycles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--opt-clusters") == 0 && i + 1 < argc) {
            mixed_cluster_path = argv[++i];
        } else if (strcmp(argv[i], "--profile-layers") == 0 && i + 1 < argc) {
            profile_experts_path = argv[++i];
        } else if (strcmp(argv[i], "--profile-freq") == 0 && i + 1 < argc) {
            profile_freq_path = argv[++i];
        } else if (strcmp(argv[i], "--load-frequency") == 0 && i + 1 < argc) {
            load_freq_path = argv[++i];
        } else if (strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            draft_model_path = argv[++i];
        } else if (strcmp(argv[i], "--draft-n") == 0 && i + 1 < argc) {
            draft_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--draft-gpu-layers") == 0 && i + 1 < argc) {
            draft_gpu_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ds") == 0 && i + 1 < argc) {
            self_spec_skip = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dm") == 0 && i + 1 < argc) {
            self_spec_m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--log-prob") == 0) {
            log_confidence = true;
        } else if (strcmp(argv[i], "--log-entropy") == 0) {
            log_gate_entropy = true;
        } else if (strcmp(argv[i], "--weight-patch") == 0 && i + 1 < argc) {
            router_patch_path = argv[++i];
        } else if (strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
            quant = argv[++i];
        } else if (strcmp(argv[i], "--arch-override") == 0 && i + 1 < argc) {
            arch_override = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
            lora_rank = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lora_lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            train_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            train_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--forward-only") == 0) {
            use_mezo = true;
        } else if (strcmp(argv[i], "--combined") == 0) {
            use_combined = true;
        } else if (strcmp(argv[i], "--chunked") == 0) {
            use_chunked = true;
        } else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) {
            chunked_K = atoi(argv[++i]);
            use_chunked = true;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_output = true;
        } else if (strcmp(argv[i], "--json-progress") == 0) {
            json_progress = true;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet_mode = true;
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            serve_port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ctx-extended") == 0 && i + 1 < argc) {
            ctx_extended = std::min(atoi(argv[++i]), 1048576);  // cap at 1M
        } else if (strcmp(argv[i], "--from-scratch") == 0) {
            from_scratch = true;
        } else if (strcmp(argv[i], "--warmup-steps") == 0 && i + 1 < argc) {
            scratch_warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else if (model_name.empty()) {
            model_name = argv[i];
        } else if (prompt.empty()) {
            // Non-flag arg after model name = start of prompt
            // But don't consume remaining args that look like flags
            prompt = argv[i];
            for (int j = i + 1; j < argc; j++) {
                if (argv[j][0] == '-') {
                    // Flag after prompt - rewind and let the parser handle it
                    i = j - 1;
                    break;
                }
                prompt += " ";
                prompt += argv[j];
                if (j == argc - 1) i = j;  // consumed all
            }
        }
    }

    // Quiet mode: suppress all llama/ggml log output (for service/API use)
    // Only JSON progress events on stderr remain visible
    if (quiet_mode) {
        llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
    }

    // Service mode: read config from stdin, no args visible in process list
    // Process shows only: "alphachat" - no flags, no paths, no config
    // All config (model path, data path, flags) comes via stdin JSON pipe
    if (cmd == "--service") {
        llama_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);

        // Ensure JSON events are flushed immediately (Windows stderr is block-buffered)
        setvbuf(stderr, nullptr, _IONBF, 0);

        // Stealth layer 1: Mask argv from ps/proc
#if defined(__linux__)
        prctl(PR_SET_NAME, "alphachat", 0, 0, 0);
#endif
        {
            const char * mask_name = "alphachat";
            size_t mask_len = strlen(mask_name);
            size_t arg0_len = strlen(argv[0]);
            memset(argv[0], 0, arg0_len);
            memcpy(argv[0], mask_name, (mask_len < arg0_len) ? mask_len : arg0_len);
            for (int i = 1; i < argc; i++) {
                memset(argv[i], 0, strlen(argv[i]));
            }
        }

        // Stealth layer 2: Clear sensitive env vars
        // Prevents /proc/pid/environ from leaking paths/keys
        {
            const char * sensitive_vars[] = {
                "ALPHA_MODEL_PATH", "ALPHA_DATA_PATH", "ALPHA_API_KEY",
                "OPENROUTER_API_KEY", "OPENAI_API_KEY", "HF_TOKEN",
                "CUDA_VISIBLE_DEVICES",  // reveals GPU targeting
                nullptr
            };
            for (int i = 0; sensitive_vars[i]; i++) {
#if defined(_WIN32)
                _putenv_s(sensitive_vars[i], "");
#else
                unsetenv(sensitive_vars[i]);
#endif
            }
        }

        // Read JSON config from stdin (one line)
        std::string config_line;
        if (!std::getline(std::cin, config_line) || config_line.empty()) {
            fprintf(stderr, "{\"event\":\"error\",\"message\":\"no config on stdin\"}\n");
            return 1;
        }

        // Minimal JSON parser for flat config: extract string/number values by key
        auto get_str = [&](const char * key) -> std::string {
            std::string needle = std::string("\"") + key + "\":\"";
            size_t pos = config_line.find(needle);
            if (pos == std::string::npos) return "";
            pos += needle.size();
            size_t end = config_line.find('"', pos);
            return (end != std::string::npos) ? config_line.substr(pos, end - pos) : "";
        };
        auto get_int = [&](const char * key, int def) -> int {
            std::string needle = std::string("\"") + key + "\":";
            size_t pos = config_line.find(needle);
            if (pos == std::string::npos) return def;
            return atoi(config_line.c_str() + pos + needle.size());
        };
        auto get_float = [&](const char * key, float def) -> float {
            std::string needle = std::string("\"") + key + "\":";
            size_t pos = config_line.find(needle);
            if (pos == std::string::npos) return def;
            return (float)atof(config_line.c_str() + pos + needle.size());
        };

        std::string svc_cmd = get_str("command");
        if (svc_cmd.empty()) svc_cmd = "train";

        if (svc_cmd == "train") {
            std::string svc_model = get_str("model");
            std::string svc_data = get_str("data");
            std::string svc_output = get_str("output");
            if (svc_output.empty()) svc_output = "lora_output.gguf";

            if (svc_model.empty() || svc_data.empty()) {
                fprintf(stderr, "{\"event\":\"error\",\"message\":\"missing model or data in config\"}\n");
                return 1;
            }

            ggml_backend_load_all();

            alpha_hw_plan svc_plan = alpha_hw_plan_create(svc_model.c_str(), -1);
            llama_model_params mp = llama_model_default_params();
            mp.n_gpu_layers = svc_plan.n_gpu_layers;
            if (svc_plan.devices[0]) mp.devices = svc_plan.devices;
            mp.split_mode = svc_plan.split_mode;
            if (svc_plan.main_device_idx >= 0) mp.main_gpu = svc_plan.main_device_idx;

            // Pipeline mode
            ggml_backend_buffer_type_t svc_cpu_buft = nullptr;
            llama_model_tensor_buft_override svc_overrides[3] = {};
            if (svc_plan.train_mode == alpha_hw_plan::ALPHA_TRAIN_HW_PIPELINE) {
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                        svc_cpu_buft = ggml_backend_dev_buffer_type(dev);
                        break;
                    }
                }
                if (svc_cpu_buft) {
                    svc_overrides[0] = { "blk.", svc_cpu_buft };
                    svc_overrides[1] = { "output", svc_cpu_buft };
                    svc_overrides[2] = { nullptr, nullptr };
                    mp.tensor_buft_overrides = svc_overrides;
                }
            }

            llama_model * svc_mdl = llama_model_load_from_file(svc_model.c_str(), mp);
            if (!svc_mdl) {
                fprintf(stderr, "{\"event\":\"error\",\"message\":\"failed to load model: %s (insufficient GPU memory, unsupported format, or corrupted file)\"}\n", svc_model.c_str());
                return 3;
            }

            int svc_seq_len = get_int("seq_len", 256);
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = svc_seq_len + 16;
            cp.n_batch = svc_seq_len;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.cb_eval = alpha_combined_get_callback();
            cp.cb_eval_user_data = nullptr;

            llama_context * svc_ctx = llama_init_from_model(svc_mdl, cp);
            if (!svc_ctx) {
                fprintf(stderr, "{\"event\":\"error\",\"message\":\"failed to create context\"}\n");
                llama_model_free(svc_mdl);
                return 3;
            }

            alpha_combined_train_config cfg = alpha_combined_default_config();
            cfg.rank = get_int("rank", 32);
            cfg.json_progress = true;
            float user_lr = get_float("lr", 0.0f);
            if (user_lr > 0) {
                cfg.lr = user_lr;
                cfg.lr_mezo = user_lr * 5;
            }
            cfg.max_steps = get_int("max_steps", cfg.max_steps);

            int svc_epochs = get_int("epochs", 1);
            int rc = alpha_combined_train(svc_mdl, svc_ctx, svc_data.c_str(),
                                           svc_output.c_str(), cfg, svc_epochs, svc_seq_len);

            llama_free(svc_ctx);
            llama_model_free(svc_mdl);
            return rc;
        }

        if (svc_cmd == "serve") {
            std::string svc_model = get_str("model");
            int svc_port = get_int("port", 11434);
            if (svc_model.empty()) {
                // Daemon mode
                return cmd_serve_daemon(svc_port);
            }
            return cmd_serve(svc_model, svc_port, -1, 0, false, true);
        }

        fprintf(stderr, "{\"event\":\"error\",\"message\":\"unknown service command: %s\"}\n", svc_cmd.c_str());
        return 1;
    }

    // Dispatch
    if (cmd == "--version" || cmd == "version") {
        printf("alphachat v1.0.0 (%s, built " __DATE__ ")\n",
#if defined(_WIN32)
            "windows-x64"
#elif defined(__APPLE__)
#  if defined(__aarch64__)
            "macos-arm64"
#  else
            "macos-x64"
#  endif
#elif defined(__aarch64__)
            "linux-arm64"
#else
            "linux-x64"
#endif
        );
        return 0;
    }
    else if (cmd == "info") {
        return cmd_info();
    }
    else if (cmd == "list" || cmd == "ls") {
        return cmd_list();
    }
    else if (cmd == "show") {
        if (model_name.empty()) { fprintf(stderr, "Usage: alphachat show <model>\n"); return 1; }
        return cmd_show(model_name);
    }
    else if (cmd == "rm" || cmd == "remove") {
        if (model_name.empty()) { fprintf(stderr, "Usage: alphachat rm <model>\n"); return 1; }
        return cmd_rm(model_name);
    }
    else if (cmd == "pull") {
        if (model_name.empty()) { fprintf(stderr, "Usage: alphachat pull <model>\n"); return 1; }
        return cmd_pull(model_name, quant);
    }
    else if (cmd == "serve") {
        if (model_name.empty()) {
            // Daemon mode: serve any model on demand (Ollama-compatible)
            return cmd_serve_daemon(serve_port, ngl_explicit ? n_gpu_layers : -1);
        }
        // Explicit model mode: serve a specific model
        return cmd_serve(model_name, serve_port, n_gpu_layers, expert_cache_mb, ngl_explicit, quiet_mode, ctx_extended);
    }
    else if (cmd == "run") {
        if (model_name.empty()) { fprintf(stderr, "Usage: alphachat run <model> [prompt]\n"); return 1; }
        // Pre-validate architecture before loading
        std::string resolved = resolve_model(model_name);
        if (!resolved.empty() && arch_override.empty()) {
            auto meta = validate_model_arch(resolved);
            if (!meta.arch_supported && !meta.arch.empty()) {
                // Skip validation for split GGUFs (arch is empty when metadata shard can't be parsed)
                fprintf(stderr, "\n");
                fprintf(stderr, "  Architecture '%s' is not yet supported by Alpha Engine.\n", meta.arch.c_str());
                fprintf(stderr, "\n");
                fprintf(stderr, "  Model:     %s\n", meta.name.empty() ? model_name.c_str() : meta.name.c_str());
                fprintf(stderr, "  Layers:    %u\n", meta.n_layer);
                fprintf(stderr, "  Embedding: %u\n", meta.n_embd);
                fprintf(stderr, "  Heads:     %u\n", meta.n_head);
                if (meta.n_expert > 0)
                    fprintf(stderr, "  Experts:   %u (%u active)\n", meta.n_expert, meta.n_expert_used);
                fprintf(stderr, "  Tensors:   %d\n", meta.n_tensors);
                fprintf(stderr, "\n");
                if (meta.similarity >= 0.7f) {
                    fprintf(stderr, "  Tensor patterns match '%s' (%.0f%% similarity).\n",
                            meta.similar_arch.c_str(), meta.similarity * 100);
                    fprintf(stderr, "  Try: alphachat run %s --arch-override %s\n\n",
                            model_name.c_str(), meta.similar_arch.c_str());
                }
                fprintf(stderr, "  Report this model to AlphaChat for support? [Y/n] ");
                fflush(stderr);
                // Non-interactive: just print the suggestion
                fprintf(stderr, "\n  To report: alphachat report %s\n", resolved.c_str());
                return 2;
            }
        }

        return cmd_run(model_name, prompt, n_predict, flash_attn, expert_cache_mb, n_gpu_layers, n_ctx, temperature, seed, ngl_explicit, prompt.empty(), no_think, expert_sticky, expert_layer_bias, expert_groups_path, log_confidence, router_patch_path, log_gate_entropy, split_group, expert_cache_explicit, speculative_mode, expert_overrides, self_spec_n, self_spec_experts, no_barrier, json_output, topk_experts, cached_spec_n, tier1_cycles, mixed_cluster_path, profile_experts_path, profile_freq_path, load_freq_path, draft_model_path, draft_n, draft_gpu_layers, self_spec_skip, self_spec_m);
    }
    else if (cmd == "bench") {
        if (model_name.empty()) { fprintf(stderr, "Usage: alphachat bench <model>\n"); return 1; }
        return cmd_bench(model_name, n_predict > 256 ? n_predict : 500, n_gpu_layers, ngl_explicit);
    }
    else if (cmd == "grow") {
        // alphachat grow <model> --add-expert --source-expert N [--output path] [--domain name]
        if (model_name.empty()) {
            fprintf(stderr, "Usage: alphachat grow <model> --add-expert --source-expert <id> [--output <path>] [--domain <name>]\n");
            return 1;
        }
        // Parse grow-specific args
        int source_expert = -1;
        bool add_expert = false;
        std::string grow_output;
        std::string grow_domain;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--add-expert") == 0) add_expert = true;
            else if (strcmp(argv[i], "--source-expert") == 0 && i + 1 < argc) source_expert = atoi(argv[++i]);
            else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) grow_output = argv[++i];
            else if (strcmp(argv[i], "--domain") == 0 && i + 1 < argc) grow_domain = argv[++i];
        }
        if (!add_expert) {
            fprintf(stderr, "Error: --add-expert is required\n");
            return 1;
        }
        if (source_expert < 0) {
            fprintf(stderr, "Error: --source-expert <id> is required\n");
            return 1;
        }
        std::string resolved = resolve_model(model_name);
        if (resolved.empty()) { fprintf(stderr, "Model not found: %s\n", model_name.c_str()); return 1; }
        if (grow_output.empty()) {
            // Default output: model_grown.gguf
            size_t dot = resolved.rfind('.');
            if (dot != std::string::npos)
                grow_output = resolved.substr(0, dot) + "_grown" + resolved.substr(dot);
            else
                grow_output = resolved + "_grown.gguf";
        }
        // Find the grow script next to the binary
        std::string script;
        {
            // Try relative paths from binary location
            const char * candidates[] = {
                "../../../scripts/alphachat_grow.py",
                "../../scripts/alphachat_grow.py",
                "../scripts/alphachat_grow.py",
                "scripts/alphachat_grow.py",
                nullptr
            };
            for (int j = 0; candidates[j]; j++) {
                FILE * test = fopen(candidates[j], "r");
                if (test) { fclose(test); script = candidates[j]; break; }
            }
            if (script.empty()) {
                fprintf(stderr, "Error: alphachat_grow.py not found. Place it in scripts/ directory.\n");
                return 1;
            }
        }
        std::string cmd_line = "python3 \"" + script + "\" \"" + resolved + "\" --add-expert"
                             + " --source-expert " + std::to_string(source_expert)
                             + " --output \"" + grow_output + "\"";
        if (!grow_domain.empty()) cmd_line += " --domain \"" + grow_domain + "\"";
        fprintf(stderr, "[alphachat] Growing model: %s\n", cmd_line.c_str());
        return system(cmd_line.c_str());
    }
    else if (cmd == "update") {
        return cmd_update();
    }
    else if (cmd == "models" || cmd == "catalog") {
        return cmd_models();
    }
    else if (cmd == "train") {
        // Subcommand: alphachat train validate <data_file>
        if (model_name == "validate" && !data_path.empty()) {
            fprintf(stderr, "Validating training data: %s\n", data_path.c_str());
            std::ifstream vf(data_path);
            if (!vf.is_open()) {
                fprintf(stderr, "Error: cannot open %s\n", data_path.c_str());
                return 1;
            }
            std::string vtext, vline;
            int line_count = 0;
            size_t total_bytes = 0;
            while (std::getline(vf, vline)) {
                line_count++;
                total_bytes += vline.size();
                vtext += vline + "\n";
            }
            vf.close();
            fprintf(stderr, "  Lines:  %d\n", line_count);
            fprintf(stderr, "  Size:   %.1f KB\n", total_bytes / 1024.0f);
            fprintf(stderr, "  Status: OK (plain text, ready for training)\n");
            return 0;
        }

        if (model_name.empty() || data_path.empty()) {
            fprintf(stderr, "Usage: alphachat train <model> --data <file> [options]\n");
            fprintf(stderr, "       alphachat train validate --data <file>\n");
            fprintf(stderr, "Options:\n");
            fprintf(stderr, "  --data <file>       Training data (text file, required)\n");
            fprintf(stderr, "  --output <file>     Output path (default: lora_output.gguf)\n");
            fprintf(stderr, "  --rank <n>          LoRA rank (default: 32)\n");
            fprintf(stderr, "  --lr <f>            Learning rate (default: auto by model size)\n");
            fprintf(stderr, "  --epochs <n>        Number of epochs (default: 1)\n");
            fprintf(stderr, "  --seq-len <n>       Sequence length (default: 256)\n");
            fprintf(stderr, "  --chunked           Chunked progressive training (auto-selects level)\n");
            fprintf(stderr, "  --chunks <n>        Number of chunks K (default: auto)\n");
            fprintf(stderr, "  --combined          Combined optimizer\n");
            fprintf(stderr, "  --forward-only      Forward-only optimizer\n");
            fprintf(stderr, "  --from-scratch      Train from random weights\n");
            fprintf(stderr, "  --warmup-steps <n>  Model warmup steps (default: auto)\n");
            fprintf(stderr, "  --json-progress     Output JSON progress lines\n");
            fprintf(stderr, "  -ngl <n>            GPU layers (default: auto)\n");
            return 1;
        }

        // Auto-select --chunked for --from-scratch (it's always better)
        if (from_scratch && !use_chunked && !use_combined) {
            use_chunked = true;
        }

        // Resolve model
        std::string path = resolve_model(model_name);
        if (path.empty()) {
            fprintf(stderr, "Model not found: %s\n", model_name.c_str());
            return 1;
        }

        fprintf(stderr, "Training LoRA adapter for: %s\n", path.c_str());
        fprintf(stderr, "  (Quantized models train via LoRA — achieves 92-99%% of full fine-tuning quality)\n");

        ggml_backend_load_all();

        // Use hw_planner to determine if model fits in VRAM or needs pipeline
        alpha_hw_plan train_plan = alpha_hw_plan_create(path.c_str(), ngl_explicit ? n_gpu_layers : -1);

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = train_plan.n_gpu_layers;
        if (train_plan.devices[0]) {
            model_params.devices = train_plan.devices;
        }
        model_params.split_mode = train_plan.split_mode;
        if (train_plan.main_device_idx >= 0) {
            model_params.main_gpu = train_plan.main_device_idx;
        }

        // Pipeline training: when model exceeds VRAM, put all weights on CPU
        // The ggml backend scheduler will stream layers through GPU automatically
        // (same mechanism as inference with the expert cache)
        ggml_backend_buffer_type_t cpu_buft = nullptr;
        llama_model_tensor_buft_override train_buft_overrides[3] = {};
        if (train_plan.train_mode == alpha_hw_plan::ALPHA_TRAIN_HW_PIPELINE) {
            // Find CPU buffer type
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    cpu_buft = ggml_backend_dev_buffer_type(dev);
                    break;
                }
            }
            if (cpu_buft) {
                // Override all weight tensors to CPU - scheduler streams them to GPU per-layer
                train_buft_overrides[0] = { "blk.", cpu_buft };   // all layer weights
                train_buft_overrides[1] = { "output", cpu_buft }; // output head
                train_buft_overrides[2] = { nullptr, nullptr };
                model_params.tensor_buft_overrides = train_buft_overrides;
                fprintf(stderr, "Training: streaming mode\n");
            }
        } else {
            fprintf(stderr, "Training: direct mode\n");
        }

        llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
        if (!model) {
            fprintf(stderr, "Error: failed to load model: %s\n", path.c_str());
            fprintf(stderr, "  Possible causes: insufficient GPU memory, unsupported format, or corrupted file\n");
            return 3;
        }

        // Create context with enough space for training sequences
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = train_seq_len + 16;
        ctx_params.n_batch = train_seq_len;
        // Disable flash attention for training (need softmax weights for attention backward)
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        // Set eval callback for activation capture
        if (use_chunked || from_scratch) {
            ctx_params.cb_eval = alpha_chunked_get_callback();
            ctx_params.cb_eval_user_data = nullptr;
        } else if (use_combined) {
            ctx_params.cb_eval = alpha_combined_get_callback();
            ctx_params.cb_eval_user_data = nullptr;
        } else if (!use_mezo) {
            ctx_params.cb_eval = alpha_train_engine_get_callback();
            ctx_params.cb_eval_user_data = nullptr;
        }

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            fprintf(stderr, "Error: failed to create context\n");
            llama_model_free(model);
            return 3;
        }

        // Set output path
        if (output_path.empty()) {
            output_path = "lora_output.gguf";
        }

        int rc;
        if (use_chunked) {
            // Chunked progressive backprop (Mode B: fine-tune, Mode D: from-scratch)
            alpha_chunked_config cfg = alpha_chunked_default_config();
            cfg.mode = from_scratch ? ALPHA_CHUNKED_FROM_SCRATCH : ALPHA_CHUNKED_FULL_BP;
            cfg.K = chunked_K;
            cfg.lora_rank = lora_rank;
            cfg.lr = lora_lr != 2e-5f ? lora_lr : 0; // 0 = auto
            cfg.seq_len = train_seq_len;
            cfg.json_progress = json_progress;
            if (json_progress) setvbuf(stderr, nullptr, _IONBF, 0);

            rc = alpha_chunked_train(model, ctx, data_path.c_str(), output_path.c_str(), cfg);
        } else if (from_scratch) {
            // ZOWarmUp: 2-phase from-scratch training
            // Phase 1: Warmup (escape random init, CPU backward)
            // Phase 2: MeZO (GPU-fast main training)
            // Pure MeZO cannot train from random init (variance proportional to d);
            // ZOWarmUp solves this with short backprop warmup first.

            fprintf(stderr, "\n=== Full Training ===\n");
            fprintf(stderr, "  Phase 1: Warmup (%d steps, ~%.0f min)\n",
                    scratch_warmup_steps, scratch_warmup_steps * 11.5 / 60.0);
            fprintf(stderr, "  Phase 2: Fast optimizer (remaining steps)\n");
            fprintf(stderr, "============================================\n\n");

            // Phase 1: Combined MeZO + backprop for warmup steps
            {
                fprintf(stderr, "── Phase 1: Warmup ──\n");
                alpha_combined_train_config cfg1 = alpha_combined_default_config();
                cfg1.rank = lora_rank;
                cfg1.max_steps = scratch_warmup_steps;
                cfg1.convergence_window = scratch_warmup_steps + 1;  // don't auto-stop Phase 1
                cfg1.json_progress = json_progress;
                if (json_progress) setvbuf(stderr, nullptr, _IONBF, 0);

                // Phase 1 saves intermediate adapter
                std::string phase1_output = output_path + ".phase1";
                rc = alpha_combined_train(model, ctx, data_path.c_str(), phase1_output.c_str(),
                                           cfg1, 1, train_seq_len);

                if (rc != 0) {
                    fprintf(stderr, "Warmup phase failed\n");
                    llama_free(ctx);
                    llama_model_free(model);
                    return rc;
                }
                fprintf(stderr, "\n── Phase 1 complete. Transitioning to fast optimizer ──\n\n");
            }

            // Phase 2: MeZO-only (GPU-fast, forward-only)
            // The LoRA adapter from Phase 1 is already loaded in the context
            {
                fprintf(stderr, "── Phase 2: Fast training ──\n");
                alpha_lora_train_config cfg2 = {};
                cfg2.rank = lora_rank;
                // From-scratch needs higher LR than fine-tuning
                cfg2.lr = (lora_lr != 2e-5f) ? lora_lr : 1e-3f;
                cfg2.beta1 = 0.9f;
                cfg2.beta2 = 0.999f;
                cfg2.eps = 1e-8f;
                cfg2.weight_decay = 0.01f;
                cfg2.warmup_steps = 100;
                cfg2.mezo_eps = 1e-3f;

                rc = alpha_lora_train(model, ctx, data_path.c_str(), output_path.c_str(),
                                       cfg2, train_epochs, train_seq_len);
            }
        } else if (use_combined || (!use_mezo)) {
            // Combined MeZO + sparse backprop (default - best quality/speed tradeoff)
            alpha_combined_train_config config = alpha_combined_default_config();
            config.rank = lora_rank;
            // Use defaults for lr/lr_mezo unless user overrides
            // Default: lr=1e-5 (backprop), lr_mezo=5e-5 (MeZO)
            if (lora_lr != 2e-5f) {
                // User specified --lr, use it for backprop, 5x for MeZO
                config.lr = lora_lr;
                config.lr_mezo = lora_lr * 5;
            }
            config.max_steps = train_epochs * 50000;
            config.json_progress = json_progress;
            if (json_progress) {
                setvbuf(stderr, nullptr, _IONBF, 0);  // flush each JSON line immediately
            }

            rc = alpha_combined_train(model, ctx, data_path.c_str(), output_path.c_str(),
                                       config, train_epochs, train_seq_len);
        } else {
            // Forward-only optimizer mode
            alpha_lora_train_config config = {};
            config.rank = lora_rank;
            config.lr = lora_lr;
            config.beta1 = 0.9f;
            config.beta2 = 0.999f;
            config.eps = 1e-8f;
            config.weight_decay = 0.0f;
            config.warmup_steps = 10;
            config.mezo_eps = 1e-3f;

            rc = alpha_lora_train(model, ctx, data_path.c_str(), output_path.c_str(),
                                   config, train_epochs, train_seq_len);
        }

        llama_free(ctx);
        llama_model_free(model);

        if (rc == 0) {
            fprintf(stderr, "\nTraining complete! LoRA saved to: %s\n", output_path.c_str());
            fprintf(stderr, "To use: alphachat run %s --lora %s \"your prompt\"\n",
                    model_name.c_str(), output_path.c_str());
        }
        return rc;
    }
    else if (cmd == "train-full") {
        if (model_name.empty() || data_path.empty()) {
            fprintf(stderr, "Usage: alphachat train-full <model> --data <file> [options]\n");
            fprintf(stderr, "Options:\n");
            fprintf(stderr, "  --data <file>       Training data (text file, required)\n");
            fprintf(stderr, "  --output <file>     Output model path (default: model_trained.gguf)\n");
            fprintf(stderr, "  --lr <f>            Learning rate (default: 1e-4)\n");
            fprintf(stderr, "  --epochs <n>        Number of epochs (default: 1)\n");
            fprintf(stderr, "  --seq-len <n>       Sequence length (default: 512)\n");
            fprintf(stderr, "  -ngl <n>            GPU layers (default: auto)\n");
            return 1;
        }

        std::string path = resolve_model(model_name);
        if (path.empty()) {
            fprintf(stderr, "Model not found: %s\n", model_name.c_str());
            return 1;
        }

        ggml_backend_load_all();

        // Use hw_planner for pipeline detection
        alpha_hw_plan tf_plan = alpha_hw_plan_create(path.c_str(), ngl_explicit ? n_gpu_layers : -1);

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = tf_plan.n_gpu_layers;
        if (tf_plan.devices[0]) model_params.devices = tf_plan.devices;
        model_params.split_mode = tf_plan.split_mode;
        if (tf_plan.main_device_idx >= 0) model_params.main_gpu = tf_plan.main_device_idx;

        // Pipeline: stream weights from CPU when model exceeds VRAM
        ggml_backend_buffer_type_t tf_cpu_buft = nullptr;
        llama_model_tensor_buft_override tf_overrides[3] = {};
        if (tf_plan.train_mode == alpha_hw_plan::ALPHA_TRAIN_HW_PIPELINE) {
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    tf_cpu_buft = ggml_backend_dev_buffer_type(dev);
                    break;
                }
            }
            if (tf_cpu_buft) {
                tf_overrides[0] = { "blk.", tf_cpu_buft };
                tf_overrides[1] = { "output", tf_cpu_buft };
                tf_overrides[2] = { nullptr, nullptr };
                model_params.tensor_buft_overrides = tf_overrides;
                fprintf(stderr, "Pipeline training: streaming layers through GPU\n");
            }
        }

        llama_model * model = llama_model_load_from_file(path.c_str(), model_params);
        if (!model) {
            fprintf(stderr, "Error: failed to load model: %s\n", path.c_str());
            fprintf(stderr, "  Possible causes: insufficient GPU memory, unsupported format, or corrupted file\n");
            return 3;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = train_seq_len + 16;
        ctx_params.n_batch = train_seq_len;
        ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;  // need full attention for backward

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) { fprintf(stderr, "Error: failed to create context\n"); llama_model_free(model); return 3; }

        if (output_path.empty()) output_path = "model_trained.gguf";

        alpha_train_full_config cfg = alpha_train_full_default_config();
        cfg.lr = lora_lr;
        cfg.seq_len = train_seq_len;
        cfg.log_every = 10;
        cfg.save_every = 500;

        int rc = alpha_train_full_run(model, ctx, data_path.c_str(), output_path.c_str(), cfg, train_epochs);

        llama_free(ctx);
        llama_model_free(model);
        return rc;
    }
    else if (cmd == "--version" || cmd == "-v" || cmd == "version") {
        printf("AlphaChat v1.0.0\n");
        return 0;
    }
    else if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        print_usage();
        return 0;
    }
    else {
        // Maybe they typed "alphachat model_name prompt" without "run"
        // Treat first arg as model name, rest as prompt
        model_name = cmd;
        if (!prompt.empty()) {
            // already parsed
        } else if (argc > 2) {
            prompt = argv[2];
            for (int j = 3; j < argc; j++) {
                prompt += " ";
                prompt += argv[j];
            }
        }

        std::string resolved = resolve_model(model_name);
        if (!resolved.empty()) {
            if (prompt.empty()) prompt = "Hello! How can I help you today?";
            return cmd_run(model_name, prompt, n_predict, flash_attn, expert_cache_mb, n_gpu_layers, n_ctx, temperature, seed, ngl_explicit, prompt.empty(), no_think, expert_sticky, expert_layer_bias, expert_groups_path, log_confidence, router_patch_path, log_gate_entropy, split_group, expert_cache_explicit, speculative_mode, expert_overrides, self_spec_n, self_spec_experts, no_barrier, json_output, topk_experts, cached_spec_n, tier1_cycles, mixed_cluster_path, profile_experts_path, profile_freq_path, load_freq_path, draft_model_path, draft_n);
        }

        fprintf(stderr, "Unknown command: %s\n", cmd.c_str());
        print_usage();
        return 1;
    }
}
