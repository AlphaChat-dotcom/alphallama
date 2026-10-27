#include "alpha_moe_portfolio.h"
#include "alpha_mole_moe_optimizer.h"  // for ALPHA_LOG_DBG
#include "ggml.h"
#include <cstdio>
#include <cstring>
#include <cctype>

// Family detection

alpha_model_family alpha_moe_detect_family(
    const char * arch,
    int n_expert,
    int n_expert_used,
    size_t model_size_mb)
{
    // Normalize arch to lowercase for matching
    char arch_lower[64] = {};
    if (arch) {
        for (int i = 0; i < 63 && arch[i]; i++) {
            arch_lower[i] = (char)tolower(arch[i]);
        }
    }

    bool is_moe = (n_expert > 1 && n_expert_used > 0);

    // MoE families
    if (is_moe) {
        if (strstr(arch_lower, "deepseek") || strstr(arch_lower, "deepseek4")) {
            return ALPHA_FAMILY_DEEPSEEK4;
        }
        if (strstr(arch_lower, "qwen")) {
            return ALPHA_FAMILY_QWEN_MOE;
        }
        if (strstr(arch_lower, "gemma")) {
            return ALPHA_FAMILY_GEMMA_MOE;
        }
        if (strstr(arch_lower, "mixtral") || (n_expert == 8 && strstr(arch_lower, "llama"))) {
            return ALPHA_FAMILY_MIXTRAL;
        }
        if (strstr(arch_lower, "dbrx")) {
            return ALPHA_FAMILY_DBRX;
        }
        if (strstr(arch_lower, "devstral") || strstr(arch_lower, "mistral")) {
            return ALPHA_FAMILY_MIXTRAL;  // Mistral MoE family uses similar topology
        }
        // Fallback by expert count; be conservative, don't assume DeepSeek
        if (n_expert >= 128) return ALPHA_FAMILY_UNKNOWN;
        if (n_expert >= 32)  return ALPHA_FAMILY_QWEN_MOE;
        if (n_expert <= 16)  return ALPHA_FAMILY_MIXTRAL;
        return ALPHA_FAMILY_UNKNOWN;
    }

    // Dense families
    if (strstr(arch_lower, "devstral") || strstr(arch_lower, "mistral")) {
        return ALPHA_FAMILY_DEVSTRAL;
    }
    if (strstr(arch_lower, "llama") || strstr(arch_lower, "llama4")) {
        return ALPHA_FAMILY_LLAMA;
    }
    if (strstr(arch_lower, "qwen")) {
        return ALPHA_FAMILY_QWEN_DENSE;
    }
    if (strstr(arch_lower, "gemma")) {
        return ALPHA_FAMILY_GEMMA;
    }
    if (strstr(arch_lower, "phi")) {
        return ALPHA_FAMILY_PHI;
    }
    if (strstr(arch_lower, "command")) {
        return ALPHA_FAMILY_COMMAND_R;
    }

    return ALPHA_FAMILY_DENSE;
}

// Tier computation

alpha_fit_tier alpha_moe_compute_fit_tier(float fit_ratio) {
    if (fit_ratio <= 0.0f)  return ALPHA_FIT_NO_GPU;
    if (fit_ratio <  0.25f) return ALPHA_FIT_TINY;
    if (fit_ratio <  0.60f) return ALPHA_FIT_COMFORTABLE;
    if (fit_ratio <  0.85f) return ALPHA_FIT_SNUG;
    if (fit_ratio <  0.97f) return ALPHA_FIT_TIGHT;
    if (fit_ratio <  1.12f) return ALPHA_FIT_FORCE;
    if (fit_ratio <  1.20f) return ALPHA_FIT_FORCE_TIGHT;
    return ALPHA_FIT_OFFLOAD;
}

alpha_moe_tier alpha_moe_compute_moe_tier(int K_estimate, int K_min, int K_draft,
                                          int n_expert, float fit_ratio) {
    if (K_estimate <= 0)         return ALPHA_MOE_TIER_CPU_PURE;
    if (K_estimate >= n_expert)  return ALPHA_MOE_TIER_ALL_GPU;
    // Model much larger than GPU -> cpu_moe
    if (fit_ratio > 5.0f)        return ALPHA_MOE_TIER_CPU_MOE;
    // K covers < 10% of expert pool -> too many cache misses
    if (n_expert > 0 && K_estimate * 10 < n_expert) return ALPHA_MOE_TIER_CPU_MOE;
    if (K_estimate >= K_min)     return ALPHA_MOE_TIER_LUT_DIRECT;
    // K between K_draft and K_min: use CPU_MOE (not SPEC_DRAFT).
    // At K < K_min, LUT/spec generation produces wrong output because too many
    // experts are missing from cache. CPU_MOE uses all experts via mmap for
    // correct output, with LUT generation after warmup fills the cache.
    return ALPHA_MOE_TIER_CPU_MOE;
}

const char * alpha_fit_tier_name(alpha_fit_tier tier) {
    switch (tier) {
        case ALPHA_FIT_NO_GPU:      return "NO_GPU";
        case ALPHA_FIT_TINY:        return "TINY";
        case ALPHA_FIT_COMFORTABLE: return "COMFORTABLE";
        case ALPHA_FIT_SNUG:        return "SNUG";
        case ALPHA_FIT_TIGHT:       return "TIGHT";
        case ALPHA_FIT_FORCE:       return "FORCE_FIT";
        case ALPHA_FIT_FORCE_TIGHT: return "FORCE_FIT_TIGHT";
        case ALPHA_FIT_OFFLOAD:     return "OFFLOAD";
    }
    return "UNKNOWN";
}

const char * alpha_moe_tier_name(alpha_moe_tier tier) {
    switch (tier) {
        case ALPHA_MOE_TIER_ALL_GPU:    return "AG";
        case ALPHA_MOE_TIER_LUT_DIRECT: return "LD";
        case ALPHA_MOE_TIER_SPEC_DRAFT: return "SD";
        case ALPHA_MOE_TIER_CPU_MOE:    return "CO";
        case ALPHA_MOE_TIER_CPU_PURE:   return "CP";
    }
    return "UNKNOWN";
}

// Is this a MoE family?

static bool is_moe_family(alpha_model_family family) {
    switch (family) {
        case ALPHA_FAMILY_QWEN_MOE:
        case ALPHA_FAMILY_DEEPSEEK4:
        case ALPHA_FAMILY_MIXTRAL:
        case ALPHA_FAMILY_DBRX:
        case ALPHA_FAMILY_GEMMA_MOE:
            return true;
        default:
            return false;
    }
}

// Set family-specific defaults (name, ref speed)

static void apply_family_defaults(alpha_moe_profile * p, alpha_model_family family) {
    switch (family) {
        case ALPHA_FAMILY_LLAMA:
            p->family_name = "Llama";
            p->ref_tok_per_sec = 45.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_QWEN_DENSE:
            p->family_name = "Qwen Dense";
            p->ref_tok_per_sec = 50.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_GEMMA:
            p->family_name = "Gemma";
            p->ref_tok_per_sec = 55.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_PHI:
            p->family_name = "Phi";
            p->ref_tok_per_sec = 70.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_COMMAND_R:
            p->family_name = "Command R";
            p->ref_tok_per_sec = 30.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_DEVSTRAL:
            p->family_name = "Devstral";
            p->ref_tok_per_sec = 52.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_QWEN_MOE:
            p->family_name = "Qwen MoE";
            p->ref_tok_per_sec = 67.8f;
            p->ref_hardware = "3090 24GB PCIe Gen3";
            break;
        case ALPHA_FAMILY_DEEPSEEK4:
            p->family_name = "DeepSeek V4 Flash";
            p->ref_tok_per_sec = 5.25f;
            p->ref_hardware = "3090 24GB PCIe Gen3";
            break;
        case ALPHA_FAMILY_GEMMA_MOE:
            p->family_name = "Gemma MoE";
            p->ref_tok_per_sec = 112.0f;
            p->ref_hardware = "3090 24GB";
            break;
        case ALPHA_FAMILY_MIXTRAL:
            p->family_name = "Mixtral";
            p->ref_tok_per_sec = 40.0f;
            p->ref_hardware = "24GB GPU";
            break;
        case ALPHA_FAMILY_DBRX:
            p->family_name = "DBRX";
            p->ref_tok_per_sec = 0;
            p->ref_hardware = "N/A";
            break;
        default:
            p->family_name = "Unknown";
            p->ref_tok_per_sec = 0;
            p->ref_hardware = "N/A";
            break;
    }
}

// Apply dense fit-tier settings

static void apply_dense_tier(alpha_moe_profile * p, alpha_fit_tier tier) {
    p->fit_tier = tier;
    p->mode = ALPHA_MOE_MODE_DIRECT;

    // Dense models: no MoE optimizer, no callbacks, no cache, standard generation
    p->buft              = ALPHA_BUFT_NONE;
    p->callback          = ALPHA_CB_NONE;
    p->warmup            = ALPHA_WARMUP_NONE;
    p->gen               = ALPHA_GEN_STANDARD;
    p->need_moe_opt      = false;
    p->expert_cache_max  = 0;
    p->disable_cache_for_prompt = false;
    p->lut_init          = false;
    p->readback_interval = 0;
    p->gpu_layers_all    = false;

    switch (tier) {
        case ALPHA_FIT_NO_GPU:
            p->n_ctx_default = 2048;
            break;
        case ALPHA_FIT_TINY:
            // Model is tiny relative to VRAM: enable backend sampler + large ctx.
            // Self-spec stays off: tiny models are already fast on the GPU.
            p->backend_sampler = true;
            p->flash_attn      = true;
            p->n_ctx_default   = 8192;
            break;
        case ALPHA_FIT_COMFORTABLE:
            // Self-spec stays off: models in this tier are already fast on the GPU
            p->flash_attn      = true;
            p->n_ctx_default   = 4096;
            break;
        case ALPHA_FIT_SNUG:
            p->flash_attn      = true;
            p->n_ctx_default   = 4096;
            p->kv_quant        = GGML_TYPE_Q8_0;
            p->self_spec_skip  = 16;  // skip ~33% of layers on a 48-layer model
            p->self_spec_m     = 3;
            break;
        case ALPHA_FIT_TIGHT:
            p->flash_attn      = true;
            p->n_ctx_default   = 2048;
            p->kv_quant        = GGML_TYPE_Q4_0;
            break;
        case ALPHA_FIT_FORCE:
            p->flash_attn      = true;
            p->n_ctx_default   = 2048;
            p->kv_quant        = GGML_TYPE_Q4_0;
            p->force_fit       = true;
            break;
        case ALPHA_FIT_FORCE_TIGHT:
            p->flash_attn      = true;
            p->n_ctx_default   = 1024;
            p->kv_quant        = GGML_TYPE_Q4_0;
            p->force_fit       = true;
            break;
        case ALPHA_FIT_OFFLOAD:
            p->flash_attn      = true;
            p->n_ctx_default   = 4096;
            p->kv_quant        = GGML_TYPE_Q8_0;
            p->partial_offload = true;
            break;
    }
}

// Apply MoE tier settings

static void apply_moe_tier(alpha_moe_profile * p, alpha_moe_tier tier, const alpha_portfolio_context * ctx) {
    p->moe_tier = tier;

    switch (tier) {
        case ALPHA_MOE_TIER_ALL_GPU:
            // All experts fit in GPU: no cache, no callback -> CUDA graphs
            p->mode       = ALPHA_MOE_MODE_DIRECT;
            p->flash_attn = true;
            // Execution plan
            p->buft              = ALPHA_BUFT_NONE;
            p->callback          = ALPHA_CB_NONE;
            p->warmup            = ALPHA_WARMUP_NONE;
            p->gen               = ALPHA_GEN_STANDARD;
            p->need_moe_opt      = false;
            p->expert_cache_max  = 0;
            p->disable_cache_for_prompt = false;
            p->lut_init          = false;
            p->readback_interval = 0;
            p->gpu_layers_all    = false;
            // Quality gates: disabled (all experts on GPU, perfect routing)
            p->conf_margin_threshold = 0;
            p->conf_trigger_count    = 0;
            p->eos_verify            = false;
            p->eos_max_retries       = 0;
            p->quality_grace_tokens  = 0;
            break;
        case ALPHA_MOE_TIER_LUT_DIRECT:
            // Expert cache + GPU LUT, zero callback during generation
            p->mode         = ALPHA_MOE_MODE_LUT_DIRECT;
            p->flat_cache   = true;
            p->cached_spec_n = 1;
            p->split_group  = 1;
            p->flash_attn   = true;
            // Execution plan, same as CPU_MOE: cpu-moe prefill loads the correct experts,
            // then LUT generation uses those experts, giving both correct prefill and fast
            // LUT generation. Using the LUT for prefill picks wrong experts at K < K_min,
            // so the confidence gates would fire repeatedly.
            p->buft              = ALPHA_BUFT_CPU_MOE;
            p->callback          = ALPHA_CB_LUT_DISABLED;
            p->warmup            = ALPHA_WARMUP_CLASSIFY;
            p->gen               = ALPHA_GEN_LUT_DIRECT;
            p->need_moe_opt      = true;
            p->expert_cache_max  = 0;  // auto from HW plan
            p->disable_cache_for_prompt = true;  // cpu-moe prefill -> correct experts in cache
            p->lut_init          = true;
            p->readback_interval = 0;  // no periodic refresh (cpu-moe prefill warms cache correctly)
            p->gpu_layers_all    = true;
            // Quality gates: lenient (cpu-moe prefill ensures good cache state)
            p->conf_margin_threshold = 0.3f;
            p->conf_trigger_count    = 5;
            p->eos_verify            = false;
            p->eos_max_retries       = 0;
            p->quality_grace_tokens  = 10;
            break;
        case ALPHA_MOE_TIER_SPEC_DRAFT:
            // Expert cache + cached-spec M>1 verification
            p->mode         = ALPHA_MOE_MODE_CACHED_SPEC;
            p->flat_cache   = true;
            p->cached_spec_n = 1;
            p->flash_attn   = true;
            // Execution plan
            p->buft              = ALPHA_BUFT_EXPERT_CACHE;
            p->callback          = ALPHA_CB_LUT_DISABLED;
            p->warmup            = ALPHA_WARMUP_SMART;
            p->gen               = ALPHA_GEN_CACHED_SPEC;
            p->need_moe_opt      = true;
            p->expert_cache_max  = 0;  // auto from HW plan
            p->disable_cache_for_prompt = false;
            p->lut_init          = true;
            p->readback_interval = 0;
            p->gpu_layers_all    = true;
            // Quality gates: same as LUT_DIRECT
            p->conf_margin_threshold = 0.5f;
            p->conf_trigger_count    = 3;
            p->eos_verify            = false;
            p->eos_max_retries       = 0;
            p->quality_grace_tokens  = 5;
            break;
        case ALPHA_MOE_TIER_CPU_MOE:
            // cpu_moe + LUT-direct: GPU LUT for fast generation,
            // cpu_moe quality gates catch LUT errors. cached_spec_n=1 enables LUT path.
            p->mode         = ALPHA_MOE_MODE_CPU_MOE;
            p->cpu_moe      = true;
            p->cached_spec_n = 1;
            p->flat_cache   = true;
            // Execution plan
            p->buft              = ALPHA_BUFT_CPU_MOE;
            p->callback          = ALPHA_CB_LUT_DISABLED;
            p->warmup            = ALPHA_WARMUP_CLASSIFY;
            p->gen               = ALPHA_GEN_LUT_DIRECT;
            p->need_moe_opt      = true;
            p->expert_cache_max  = 0;  // auto from HW plan (a fixed large value OOMs 8 GB GPUs)
            p->disable_cache_for_prompt = true;
            p->lut_init          = true;
            p->readback_interval = 0;  // tokenizer checks trigger resync on demand
            p->gpu_layers_all    = false;  // let HW planner decide (true can OOM on 8 GB GPUs)
            // Quality gates: conservative (low expert coverage).
            // EOS is treated as genuine. Low threshold + high trigger count.
            p->conf_margin_threshold = 0.3f;
            p->conf_trigger_count    = 5;
            p->eos_verify            = false;
            p->eos_max_retries       = 0;
            p->quality_grace_tokens  = 10;
            break;
        case ALPHA_MOE_TIER_CPU_PURE:
            // Pure CPU MoE, no cache, no LUT
            p->mode    = ALPHA_MOE_MODE_CPU_MOE;
            p->cpu_moe = true;
            // Execution plan
            p->buft              = ALPHA_BUFT_CPU_MOE;
            p->callback          = ALPHA_CB_NONE;
            p->warmup            = ALPHA_WARMUP_NONE;
            p->gen               = ALPHA_GEN_STANDARD;
            p->need_moe_opt      = false;
            p->expert_cache_max  = 0;
            p->disable_cache_for_prompt = false;
            p->lut_init          = false;
            p->readback_interval = 0;
            p->gpu_layers_all    = true;
            // Quality gates: disabled (no LUT, no cache; standard cpu-moe)
            p->conf_margin_threshold = 0;
            p->conf_trigger_count    = 0;
            p->eos_verify            = false;
            p->eos_max_retries       = 0;
            p->quality_grace_tokens  = 0;
            break;
    }

    // Set k_min from context
    if (ctx) {
        p->k_min = ctx->K_min;
    }
}

// Family-specific MoE overrides

static void apply_family_moe_overrides(alpha_moe_profile * p, const alpha_portfolio_context * ctx) {
    if (!ctx) return;

    switch (ctx->family) {
        case ALPHA_FAMILY_DEEPSEEK4:
            // V4 Flash: LUT-direct output degrades after a few dozen tokens, so use cpu-moe
            if (p->moe_tier != ALPHA_MOE_TIER_ALL_GPU) {
                p->mode       = ALPHA_MOE_MODE_CPU_MOE;
                p->cpu_moe    = true;
                p->flat_cache = false;
                p->cached_spec_n = 0;
                p->topk_experts  = 4;  // adaptive k=4
                p->split_group   = 1;
            }
            break;

        case ALPHA_FAMILY_QWEN_MOE:
            // 397B-class (512 experts): update ref speed and name
            if (ctx->n_expert >= 512) {
                p->family_name = "Qwen MoE (512 experts)";
                if (p->moe_tier == ALPHA_MOE_TIER_ALL_GPU) {
                    p->ref_tok_per_sec = 0;  // no ref (needs multi-GPU)
                }
            }
            break;

        case ALPHA_FAMILY_MIXTRAL:
            // 8 experts, top-2: with K >= 8 all experts are preloaded, so reactive always hits
            if (p->moe_tier != ALPHA_MOE_TIER_ALL_GPU && ctx->n_expert <= 8) {
                p->mode = ALPHA_MOE_MODE_REACTIVE;
            }
            break;

        case ALPHA_FAMILY_DBRX:
            // 16 experts, top-4: reactive with auto split group
            if (p->moe_tier != ALPHA_MOE_TIER_ALL_GPU) {
                p->mode = ALPHA_MOE_MODE_REACTIVE;
                p->split_group = 0;
            }
            break;

        case ALPHA_FAMILY_GEMMA_MOE:
            // Small MoE: reactive works well when not all on GPU
            if (p->moe_tier != ALPHA_MOE_TIER_ALL_GPU) {
                p->mode = ALPHA_MOE_MODE_REACTIVE;
            }
            break;

        default:
            break;
    }
}

// Portfolio profiles (ratio-aware)

alpha_moe_profile alpha_moe_portfolio_select_v2(const alpha_portfolio_context * ctx) {
    alpha_moe_profile p = {};
    if (!ctx) return p;

    p.family = ctx->family;
    apply_family_defaults(&p, ctx->family);

    if (is_moe_family(ctx->family)) {
        // MoE: tier by K coverage + fit ratio
        alpha_moe_tier mt = alpha_moe_compute_moe_tier(
            ctx->K_estimate, ctx->K_min, ctx->K_draft, ctx->n_expert, ctx->fit_ratio);
        apply_moe_tier(&p, mt, ctx);
        apply_family_moe_overrides(&p, ctx);

        // Also compute fit tier for logging (MoE backbone fit)
        p.fit_tier = alpha_moe_compute_fit_tier(ctx->fit_ratio);
    } else {
        // Dense: tier by fit ratio
        alpha_fit_tier ft = alpha_moe_compute_fit_tier(ctx->fit_ratio);
        apply_dense_tier(&p, ft);
    }

    return p;
}

// Legacy API wrapper

alpha_moe_profile alpha_moe_portfolio_select(
    alpha_model_family family,
    bool fits_in_gpu,
    bool is_hybrid,
    int n_expert)
{
    alpha_portfolio_context ctx = {};
    ctx.family       = family;
    ctx.is_hybrid    = is_hybrid;
    ctx.n_expert     = n_expert;
    ctx.n_expert_used = 0;
    // Approximate fit_ratio from boolean
    ctx.fit_ratio    = fits_in_gpu ? 0.5f : 2.0f;
    ctx.K_estimate   = fits_in_gpu ? n_expert : 0;
    ctx.K_min        = (n_expert > 0) ? (n_expert / 4) : 0;
    ctx.K_draft      = (n_expert > 0) ? (n_expert / 8) : 0;
    ctx.gpu_vram_mb  = 0;
    ctx.model_size_mb = 0;

    return alpha_moe_portfolio_select_v2(&ctx);
}

// Print

static const char * mode_name(alpha_moe_mode m) {
    switch (m) {
        case ALPHA_MOE_MODE_DIRECT:      return "direct (all GPU)";
        case ALPHA_MOE_MODE_LUT_DIRECT:  return "LUT-direct";
        case ALPHA_MOE_MODE_CACHED_SPEC: return "cached-spec";
        case ALPHA_MOE_MODE_CPU_MOE:     return "cpu-moe";
        case ALPHA_MOE_MODE_REACTIVE:    return "reactive cache";
        case ALPHA_MOE_MODE_MIXED_SPEC:  return "mixed-spec";
    }
    return "unknown";
}

void alpha_moe_portfolio_print(const alpha_moe_profile * p) {
    if (!p) return;
    ALPHA_LOG_DBG( "[PORTFOLIO] %s → %s", p->family_name, mode_name(p->mode));

    // Print tier info
    if (is_moe_family(p->family)) {
        ALPHA_LOG_DBG( " [%s]", alpha_moe_tier_name(p->moe_tier));
    } else {
        ALPHA_LOG_DBG( " [%s]", alpha_fit_tier_name(p->fit_tier));
    }

    if (p->topk_experts > 0)    ALPHA_LOG_DBG( ", top-k=%d", p->topk_experts);
    if (p->cpu_moe)             ALPHA_LOG_DBG( ", cpu-moe");
    if (p->flat_cache)          ALPHA_LOG_DBG( ", flat-cache");
    if (p->cached_spec_n > 0)   ALPHA_LOG_DBG( ", spec=%d", p->cached_spec_n);
    if (p->output_cpu)          ALPHA_LOG_DBG( ", output-cpu");
    if (p->flash_attn)          ALPHA_LOG_DBG( ", flash-attn");
    if (p->backend_sampler)     ALPHA_LOG_DBG( ", backend-sampler");
    if (p->force_fit)           ALPHA_LOG_DBG( ", force-fit");
    if (p->partial_offload)     ALPHA_LOG_DBG( ", partial-offload");
    if (p->kv_quant != 0 && p->kv_quant != GGML_TYPE_F16)
        ALPHA_LOG_DBG( ", kv=%s", ggml_type_name(p->kv_quant));
    if (p->n_ctx_default > 0)   ALPHA_LOG_DBG( ", ctx=%d", p->n_ctx_default);
    if (p->ref_tok_per_sec > 0)
        ALPHA_LOG_DBG( " (ref: %.1f tok/s on %s)", p->ref_tok_per_sec, p->ref_hardware);
    ALPHA_LOG_DBG( "\n");

    // Print execution plan
    static const char * buft_names[]  = {"N", "OC", "CO"};
    static const char * cb_names[]    = {"N", "R", "LD"};
    static const char * warmup_names[] = {"N", "S", "C"};
    static const char * gen_names[]   = {"S", "LD", "CS", "TT"};
    ALPHA_LOG_DBG( "[PORTFOLIO] Plan: buft=%s, cb=%s, warmup=%s, gen=%s",
            buft_names[p->buft], cb_names[p->callback],
            warmup_names[p->warmup], gen_names[p->gen]);
    if (p->need_moe_opt)              ALPHA_LOG_DBG( ", moe_opt");
    if (p->expert_cache_max > 0)      ALPHA_LOG_DBG( ", cache=%dMB", p->expert_cache_max);
    if (p->disable_cache_for_prompt)   ALPHA_LOG_DBG( ", disable_cache_prompt");
    if (p->lut_init)                  ALPHA_LOG_DBG( ", lut_init");
    if (p->gpu_layers_all)            ALPHA_LOG_DBG( ", gpu_all");
    ALPHA_LOG_DBG( "\n");
}
