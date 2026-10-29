#include "llama-context.h"
#include <mutex>

#include "ggml.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"
#include "alpha_dense_pipeline.h"
#include "alpha_gdn_record.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

// ALPHA_MTP is on only for a non-zero value, matching the serve engine (serve-gpu-engine.cpp g_mtp), so ALPHA_MTP=0 does not load the head or build the MTP graph.
static std::vector<std::pair<const char *, ggml_tensor *>> g_mtp_nan_early;   // ALPHA_MTP_NAN_DBG
static inline bool alpha_mtp_env_on() { const char * e = getenv("ALPHA_MTP"); return e && atoi(e) != 0; }

//
// llama_context
//


// ALPHA_RESERVE_PP_OUTPUTS: the pp reservation sizes the logits buffer for every token of the micro-batch
// (n_vocab * n_ubatch * 4 B, several GB for a large vocab at a big micro-batch), while a serving prefill only reads the
// last row. Cap the reserved output rows; a batch asking for more makes the allocator grow at run time. Never capped
// for embeddings / pooling (inp_mean needs n_outputs == n_tokens). Default: 64 under ALPHA_DRE, else uncapped.
static uint32_t alpha_pp_reserve_outputs(const llama_cparams & cp, uint32_t n_tokens) {
    if (cp.embeddings || cp.pooling_type != LLAMA_POOLING_TYPE_NONE) return n_tokens;
    static const int cap = [] {
        const char * e = getenv("ALPHA_RESERVE_PP_OUTPUTS");
        if (e) return atoi(e);
        const char * d = getenv("ALPHA_DRE");
        return (d && atoi(d) != 0) ? 64 : 0;
    }();
    return cap > 0 ? std::min<uint32_t>(n_tokens, (uint32_t) cap) : n_tokens;
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if ((model.arch == LLM_ARCH_DEEPSEEK2 || model.arch == LLM_ARCH_DEEPSEEK4) && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    // ALPHA_GDN_FUSE_IN: the elementwise producers of the gated-delta-net inputs run inside the op, one launch
    // per DeltaNet layer at decode instead of six (=0 restores the in-graph chain).
    cparams.fused_gdn_in = getenv("ALPHA_GDN_FUSE_IN") ? atoi(getenv("ALPHA_GDN_FUSE_IN")) != 0 : true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // Encoder-decoder models (T5, BART) require unified KV cache.
    // Non-unified mode uses multi-stream which is unsupported for
    // cross-attention and T5 relative position bias.
    if (llama_model_has_encoder(&model)) {
        cparams.kv_unified = true;
    }

    cparams.n_layer_draft = params.n_layer_draft;
    cparams.draft_skip_mask = params.draft_skip_mask;

    if (cparams.draft_skip_mask != 0) {
        int skipped = 0;
        for (uint64_t m = cparams.draft_skip_mask; m; m &= m - 1) skipped++;
        LLAMA_LOG_INFO("%s: draft context — ISWA skip mask, %d layers skipped\n",
            __func__, skipped);
    } else if (cparams.n_layer_draft > 0) {
        LLAMA_LOG_INFO("%s: draft context — using first %d of %d layers\n",
            __func__, cparams.n_layer_draft, model.hparams.n_layer);
    }

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    static const bool sr_t = getenv("ALPHA_RESERVE_TIMERS") != nullptr;
    const int64_t sr0 = ggml_time_us();
    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    // ALPHA_GRAPH_RES_REUSE (default 1): keep graph results that already hold max_nodes. A new one zero-fills hundreds
    // of MB of tensor metadata at a large micro-batch (n_tokens x 40 nodes) on every prompt reserve; a capacity that
    // is too large is harmless (the graph is only as big as what gets built into it).
    static const bool res_reuse = [] { const char * e = getenv("ALPHA_GRAPH_RES_REUSE"); return !e || atoi(e) != 0; }();
    auto res_fits = [&](const llm_graph_result_ptr & r) {
        return res_reuse && r && static_cast<const llm_graph_result *>(r.get())->get_max_nodes() >= (int64_t) max_nodes;
    };
    if (res_fits(gf_res_prev)) gf_res_prev->reset(); else gf_res_prev.reset(new llm_graph_result(max_nodes));
    if (res_fits(gf_res_reserve)) gf_res_reserve->reset(); else gf_res_reserve.reset(new llm_graph_result(max_nodes));

    // Reuse the existing scheduler if there is one: recreating it fragments CUDA memory. Only create a new
    // scheduler on the first call or when force_reserve() explicitly requests it (periodic defrag).
    if (!sched || sched_force_new) {
        sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
        sched_force_new = false;
    } else {
        // Reset the existing scheduler for new graph reservation
        // without deallocating/reallocating CUDA compute buffers.
        ggml_backend_sched_reset(sched.get());
    }

    const int64_t sr1 = ggml_time_us();
    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }
    if (sr_t) fprintf(stderr, "[reserve-t] sched_reserve: sync %.1f ms, results+sched %.1f ms, init_full %.1f ms\n",
                      (t_start_us - sr0) / 1e3, (sr1 - t_start_us) / 1e3, (ggml_time_us() - sr1) / 1e3);

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails. this matches the pp reservation below (line ~553).
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    const int64_t sr2 = ggml_time_us();
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, alpha_pp_reserve_outputs(cparams, n_tokens), mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, alpha_pp_reserve_outputs(cparams, n_tokens), mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    if (sr_t) fprintf(stderr, "[reserve-t] sched_reserve: pp reserve %.1f ms\n", (ggml_time_us() - sr2) / 1e3);
    // ALPHA_RESERVE_PP_ONLY (default 1): a reserve for a grown prompt micro-batch (the DRE borrow) reserves the prompt
    // graph once: the token graph and the second prompt pass only matter for decode, and the return to the base
    // micro-batch reserves again anyway. Each pass is a full graph build + allocation plan.
    static const bool pp_only_env = [] { const char * e = getenv("ALPHA_RESERVE_PP_ONLY"); return !e || atoi(e) != 0; }();
    const bool pp_only = pp_only_env && n_tokens > 1024;
    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (!pp_only) {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    if (!pp_only) {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, alpha_pp_reserve_outputs(cparams, n_tokens), mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    // ALPHA_MTP: reserve a second (draft) scheduler with its own compute buffer. The draft must not run on
    // the main sched, because resetting it + gf_res_prev destroys the main decode's graph-reuse cache and
    // forces a full rebuild on the next token, which would cost about as much as a full forward pass.
    // Reserve a decode (tg) graph (the draft is single-token) via swap-reserve-swap so the alt sched gets
    // its own buffer.
    if (alpha_mtp_env_on() && !sched_alt) {
        sched_alt.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
        gf_res_prev_alt.reset(new llm_graph_result(max_nodes));
        gf_res_reserve_alt.reset(new llm_graph_result(max_nodes));
        set_draft_active(true);
        auto * gfd = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        set_draft_active(false);
        LLAMA_LOG_INFO("%s: [ALPHA_MTP] 2nd (draft) scheduler reserved (%s)\n", __func__, gfd ? "ok" : "FAILED");
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    {   // device compute bytes this micro-batch reserved (llama_context_reserved_bytes)
        size_t dev = 0;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            if (!ggml_backend_buft_is_host(backend_buft[i])) dev += backend_buf_exp_size[i];
        }
        reserved_dev_bytes[cparams.n_ubatch] = dev;
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

void llama_context::set_eval_callback(ggml_backend_sched_eval_callback cb, void * user_data) {
    cparams.cb_eval = cb;
    cparams.cb_eval_user_data = user_data;
}

void llama_context::set_draft_active(bool enable) {
    if (enable == draft_active_) {
        return;
    }
    if (!sched_alt) {
        LLAMA_LOG_WARN("%s: 2nd (draft) scheduler not reserved — ignoring\n", __func__);
        return;
    }
    // O(1) swap: all decode sites use `sched`/`gf_res_prev`; swapping switches which physical scheduler
    // (and its compute buffer + graph-reuse cache) is active. The inactive sched keeps its paused state.
    std::swap(sched,           sched_alt);
    std::swap(gf_res_prev,     gf_res_prev_alt);
    std::swap(gf_res_reserve,  gf_res_reserve_alt);
    // isolate the draft's output bookkeeping so its decode doesn't overwrite the paused verify's
    // n_outputs/output_ids (else the verify's accept reads the wrong rows and asserts)
    std::swap(n_outputs,       n_outputs_alt);
    std::swap(output_ids,      output_ids_alt);
    draft_active_ = enable;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, alpha_pp_reserve_outputs(cparams, n_tokens), mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    // while the second (draft) sched is active, its logits live in draft_logits_ (written in order by a
    // single-token decode, so no reorder is needed). Verify reads logits.data as usual.
    if (draft_active_ && !draft_logits_.empty()) {
        return draft_logits_.data();
    }
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    // the draft (second sched) reads its own logits sink; skip reorder (single-token, in order)
    const bool draft = draft_active_ && !draft_logits_.empty();
    if (!draft) {
        output_reorder();
    }

    try {
        // ALPHA_MTP: logits were deliberately not fetched for this decode (greedy MTP verify).
        // Report absence rather than hand back the previous decode's buffer.
        if (!draft && !logits_valid_) {
            return nullptr;
        }
        float * base = draft ? draft_logits_.data() : logits.data;
        if (base == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return base + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

void llama_context::mtp_switch_seq(int32_t seq) {
    if (seq < 0) seq = 0;
    const int32_t need = std::max(seq, mtp_cur_seq) + 1;
    if ((int32_t) mtp_seq.size() < need) mtp_seq.resize(need);
    if (seq == mtp_cur_seq) return;
    mtp_seq_state & cur = mtp_seq[mtp_cur_seq];          // park the live scalars
    cur.kv_hi       = mtp_kv_hi;
    cur.kv_lo       = mtp_kv_lo;
    cur.chain_step  = mtp_chain_step;
    cur.pos_base    = mtp_pos_base;
    cur.prev_pos    = mtp_prev_pos;
    cur.prev_h      = mtp_prev_h;
    cur.hidden_host = mtp_hidden_host;
    mtp_seq_state & nx = mtp_seq[seq];                   // load the incoming sequence's
    mtp_kv_hi       = nx.kv_hi;
    mtp_kv_lo       = nx.kv_lo;
    mtp_chain_step  = nx.chain_step;
    mtp_pos_base    = nx.pos_base;
    mtp_prev_pos    = nx.prev_pos;
    mtp_prev_h      = nx.prev_h;
    mtp_hidden_host = nx.hidden_host;
    mtp_cur_seq     = seq;
}

bool llama_context::mtp_seq_export(int32_t seq, int32_t pos, std::vector<float> & h) {
    if (!mtp_kv_buf || seq < 0) return false;
    const int32_t save = mtp_cur_seq;
    mtp_switch_seq(seq);
    const bool ok = mtp_prev_pos == pos && (int64_t) mtp_prev_h.size() == mtp_hw();
    if (ok) h = mtp_prev_h;
    mtp_switch_seq(save);
    return ok;
}

bool llama_context::mtp_seq_import(int32_t seq, int32_t ckpt, const std::vector<float> & h) {
    if (!mtp_kv_buf || seq < 0 || ckpt < 1 ||
        (int64_t) h.size() != mtp_hw()) return false;
    const int32_t save = mtp_cur_seq;
    mtp_switch_seq(seq);
    // Keep columns for positions < ckpt (slots <= ckpt-2); column ckpt (slot ckpt-1) is rebuilt by
    // the next fill from the staged row. pos_base = ckpt-1 makes the suffix (first == ckpt) read as a
    // continuation in mtp_seq_bookkeep, so prev_pos survives to mtp_kv_fill's boundary-column test.
    mtp_kv_hi = std::min<int32_t>(mtp_kv_hi, ckpt - 2);
    if (mtp_kv_hi < mtp_kv_lo) { mtp_kv_hi = -1; mtp_kv_lo = -1; }
    mtp_prev_h      = h;
    mtp_prev_pos    = ckpt - 1;
    mtp_pos_base    = ckpt - 1;
    mtp_hidden_host = h;
    mtp_chain_step  = 0;
    mtp_switch_seq(save);
    return true;
}

uint64_t llama_context::mtp_seq_kv_hash(int32_t seq, int32_t & lo, int32_t & hi) {
    lo = hi = -1;
    if (!mtp_kv_buf || !mtp_k_cache || seq < 0) return 0;
    const int32_t save = mtp_cur_seq;
    mtp_switch_seq(seq);
    lo = mtp_kv_lo; hi = mtp_kv_hi;
    const int32_t base = mtp_kv_base();
    mtp_switch_seq(save);
    if (hi < 0 || lo < 0 || hi < lo) return 0;
    ggml_backend_sched_synchronize(sched.get());
    uint64_t h = 1469598103934665603ull;
    std::vector<uint8_t> buf;
    for (ggml_tensor * t : { mtp_k_cache, mtp_v_cache }) {
        const size_t row = t->nb[1];
        buf.resize(row * (size_t) (hi - lo + 1));
        ggml_backend_tensor_get(t, buf.data(), row * (size_t) (base + lo), buf.size());
        for (uint8_t b : buf) { h ^= b; h *= 1099511628211ull; }
    }
    return h;
}

bool llama_context::mtp_kv_prealloc() {
    const auto & hp = model.hparams;
    if (model.arch == LLM_ARCH_QWEN4EXP) {
        if (!model.has_mtp4 || !model.mtp4.wq || !model.mtp4.wq->buffer) return false;
        return mtp_kv_ensure(hp.n_embd_head_k() * hp.n_head_kv(hp.n_layer - 1), ggml_backend_buffer_get_type(model.mtp4.wq->buffer));
    }
    const int mtp_il = (int) hp.n_layer - 1;
    if (mtp_il < 0 || mtp_il >= (int) model.layers.size()) return false;
    const auto & L = model.layers[mtp_il];
    if (!L.nextn.eh_proj || !L.wq || !L.wq->buffer) return false;
    return mtp_kv_ensure(hp.n_embd_head_k() * hp.n_head_kv(mtp_il), ggml_backend_buffer_get_type(L.wq->buffer));
}

bool alpha_mtp_kv_prealloc(llama_context * ctx) {
    return ctx && ctx->mtp_kv_prealloc();
}

bool llama_context::mtp_kv_ensure(int64_t kv_row, ggml_backend_buffer_type_t buft) {
    if (mtp_kv_buf) return mtp_k_cache != nullptr;
    // Callers pass the buffer type of the head's attention weights. On a CPU that can be an "extra" type
    // (CPU_REPACK on ARM, which keeps the int8-matmul layout and runs only MUL_MAT), so the cache's SET_ROWS
    // aborted the first MTP decode. A cache on a CPU device takes that device's plain buffer type; GPU caches
    // keep the weights' type as before.
    if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft); dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        buft = ggml_backend_dev_buffer_type(dev);
    }
    // One column per sequence position (slot j == position j+1), so the head can hold the whole
    // context, not just what its own draft windows wrote (~2 KB per position per cache at 4 kv heads
    // x 128 dims). One region per sequence: column (seq * stride + slot), so two sequences at the same
    // position do not write the same column.
    const int64_t stride = (int64_t) cparams.n_ctx + 8;
    const int64_t nseq   = std::max<int64_t>(1, (int64_t) cparams.n_seq_max);
    const int64_t cols   = stride * nseq;
    ggml_init_params kp = { ggml_tensor_overhead() * 8, nullptr, /*no_alloc*/ true };
    mtp_kv_ctx = ggml_init(kp);
    if (!mtp_kv_ctx) return false;
    // ALPHA_MTP_HEAD_FA (default on, =0 restores F32): qwen4exp uses an F16 cache read by flash attention
    // (llama-context-mtp4.cpp); every other arch's draft graph reads it as F32
    static const bool head_fa = [] { const char * e = getenv("ALPHA_MTP_HEAD_FA"); return !e || atoi(e) != 0; }();
    const ggml_type kvt = (head_fa && model.arch == LLM_ARCH_QWEN4EXP) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    mtp_k_cache = ggml_new_tensor_2d(mtp_kv_ctx, kvt, kv_row, cols);
    mtp_v_cache = ggml_new_tensor_2d(mtp_kv_ctx, kvt, kv_row, cols);
    mtp_kv_buf  = ggml_backend_alloc_ctx_tensors_from_buft(mtp_kv_ctx, buft);
    if (mtp_kv_buf) {
        ggml_backend_buffer_clear(mtp_kv_buf, 0);   // the padded attention window reads past hi: must be finite
    }
    if (!mtp_kv_buf) {
        ggml_free(mtp_kv_ctx); mtp_kv_ctx = nullptr; mtp_k_cache = mtp_v_cache = nullptr;
        return false;
    }
    mtp_kv_stride = (int32_t) stride;
    mtp_kv_cols   = (int32_t) stride;   // per-sequence bound: existing checks stay correct
    return true;
}

// Per-sequence bookkeeping: `first`/`last` are this sequence's own positions, and the caller has already
// switched the live scalars to this sequence.
void llama_context::mtp_seq_bookkeep(int32_t first, int32_t last, const float * last_h,
                                     int64_t n_embd) {
    // A start position that does not continue from the last decode means the sequence was rewound
    // (the MTP reject path re-lays the accepted prefix) or restarted (new request). Columns at and
    // after the rewind point are no longer causally valid, so truncate to it rather than clearing:
    // every rejected window re-lays from pos0, which reads as a rewind, so clearing would leave the
    // draft without context. A new request restarts at position 0 and this same rule empties the cache.
    if (first <= mtp_pos_base) {
        mtp_kv_hi = std::min<int32_t>(mtp_kv_hi, first - 1);
        // Rewinding past the start of the range empties it -- otherwise lo > hi and the attention
        // window would have a negative length.
        if (mtp_kv_hi < mtp_kv_lo) { mtp_kv_hi = -1; mtp_kv_lo = -1; }
        mtp_prev_pos = -1;   // the carried row came from the discarded continuation
    }
    if (last_h) {
        mtp_hidden_host.assign(last_h, last_h + n_embd);
    }
    mtp_pos_base   = last;
    mtp_chain_step = 0;      // a new window starts its chain at step 0
}


// ALPHA_MTP_PINNED_H: see llama-context.h
namespace {
std::mutex g_pin_mu;
std::map<void *, ggml_backend_buffer_t> g_pin_bufs;
ggml_backend_buffer_type_t pin_buft() {
    static ggml_backend_buffer_type_t bt = [] () -> ggml_backend_buffer_type_t {
        const char * e = getenv("ALPHA_MTP_PINNED_H");
        if (e && atoi(e) == 0) return nullptr;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            if (ggml_backend_buffer_type_t h = ggml_backend_dev_host_buffer_type(dev)) return h;
        }
        return nullptr;
    }();
    return bt;
}
}
template <typename T> T * llama_pinned_alloc<T>::allocate(size_t n) {
    const size_t bytes = std::max<size_t>(1, n) * sizeof(T);
    if (ggml_backend_buffer_type_t bt = pin_buft()) {
        if (ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(bt, bytes)) {
            void * p = ggml_backend_buffer_get_base(b);
            std::lock_guard<std::mutex> lk(g_pin_mu);
            g_pin_bufs[p] = b;
            return (T *) p;
        }
    }
    void * p = malloc(bytes);
    if (!p) throw std::bad_alloc();
    return (T *) p;
}
template <typename T> void llama_pinned_alloc<T>::deallocate(T * p, size_t) {
    if (!p) return;
    {
        std::lock_guard<std::mutex> lk(g_pin_mu);
        auto it = g_pin_bufs.find((void *) p);
        if (it != g_pin_bufs.end()) { ggml_backend_buffer_free(it->second); g_pin_bufs.erase(it); return; }
    }
    free(p);
}
template struct llama_pinned_alloc<float>;

// ALPHA_MTP_FILL_INGRAPH: the bookkeeping of a fill the graph already did: mtp_kv_fill_all + mtp_kv_fill, without
// the data movement (same order: sequence bookkeeping, carried row, range latch/restart, then hi)
void llama_context::mtp_ingraph_commit() {
    mtp_ig.active = false;
    const size_t n = mtp_fill_pos.size();
    const int64_t hw = mtp_hw();
    ggml_backend_sched_synchronize(sched.get());   // the last row's read
    if ((int64_t) mtp_ig.last_row.size() < hw) return;
    mtp_seq_bookkeep((int32_t) mtp_fill_pos[0], (int32_t) mtp_fill_pos[n - 1], mtp_ig.last_row.data(), hw);
    mtp_prev_h.assign(mtp_ig.last_row.begin(), mtp_ig.last_row.begin() + hw);
    mtp_prev_pos = mtp_fill_pos[n - 1];
    const int32_t slot0 = mtp_ig.slot0;
    if (mtp_kv_hi < 0) {
        mtp_kv_lo = slot0;
    } else if (slot0 > mtp_kv_hi + 1 || slot0 < mtp_kv_lo) {
        mtp_kv_lo = slot0;
        mtp_kv_hi = -1;
    }
    mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, (int32_t) (slot0 + mtp_ig.m - 1));
    if (getenv("ALPHA_MTP_DBG")) {
        fprintf(stderr, "[mtp-fill] in-graph: slot0=%d m=%lld -> range [%d,%d]\n", slot0, (long long) mtp_ig.m, mtp_kv_lo, mtp_kv_hi);
    }
}

// Drive mtp_kv_fill once per sequence present in the batch. mtp_kv_fill() assumes every row it sees belongs
// to one sequence: it takes slot0 from cpos[0] and compares it against the active sequence's [lo,hi]. With
// n_seq_max > 1 the scheduler batches tokens from several sequences into one decode, so mtp_fill_pos is
// mixed and would be checked against the wrong sequence's range, refusing every fill.
void llama_context::mtp_kv_fill_all() {
    if (mtp_fill_pos.empty()) return;
    if (mtp_ig.active) { mtp_ingraph_commit(); return; }
    const size_t n = mtp_fill_pos.size();
    if (mtp_fill_seq.size() != n) { mtp_kv_fill(); return; }   // no seq info: legacy behaviour

    bool mixed = false;
    for (size_t i = 1; i < n; ++i) if (mtp_fill_seq[i] != mtp_fill_seq[0]) { mixed = true; break; }

    const int32_t save_seq = mtp_cur_seq;
    if (!mixed) {
        // Single-sequence batch (the -np 1 case): identical to calling mtp_kv_fill directly
        // (mtp_switch_seq returns immediately when the sequence is already active).
        const int32_t sq = (int32_t) mtp_fill_seq[0];
        if (sq != mtp_cur_seq) mtp_switch_seq(sq);
        const int64_t n_embd_1 = mtp_hw();
        const float * last_h_1 = ((int64_t) mtp_fill_h.size() >= (int64_t) n * n_embd_1)
                               ? mtp_fill_h.data() + (size_t) (n - 1) * n_embd_1 : nullptr;
        mtp_seq_bookkeep((int32_t) mtp_fill_pos[0], (int32_t) mtp_fill_pos[n - 1], last_h_1, n_embd_1);
        mtp_kv_fill();
        if (mtp_cur_seq != save_seq) mtp_switch_seq(save_seq);
        return;
    }

    // Mixed batch: partition by sequence and present each sequence's rows alone, so slot0 and the
    // [lo,hi] range it is checked against belong to the same sequence.
    const int64_t n_embd = mtp_hw();
    std::vector<llama_pos>     all_pos = mtp_fill_pos;
    std::vector<llama_token>   all_tok = mtp_fill_tok;
    std::vector<llama_seq_id>  all_seq = mtp_fill_seq;
    std::vector<float>         all_h(mtp_fill_h.begin(), mtp_fill_h.end());

    std::vector<llama_seq_id> seqs;
    for (size_t i = 0; i < n; ++i)
        if (std::find(seqs.begin(), seqs.end(), all_seq[i]) == seqs.end()) seqs.push_back(all_seq[i]);

    for (llama_seq_id sq : seqs) {
        std::vector<llama_pos>   fp; std::vector<llama_token> ft; std::vector<llama_seq_id> fs;
        std::vector<float>       fh;
        for (size_t i = 0; i < n; ++i) {
            if (all_seq[i] != sq) continue;
            fp.push_back(all_pos[i]); ft.push_back(all_tok[i]); fs.push_back(sq);
            if ((int64_t) all_h.size() >= (int64_t) (i + 1) * n_embd)
                fh.insert(fh.end(), all_h.begin() + (size_t) i * n_embd,
                                    all_h.begin() + (size_t) (i + 1) * n_embd);
        }
        if (fp.empty()) continue;
        mtp_fill_pos = fp; mtp_fill_tok = ft; mtp_fill_seq = fs; mtp_fill_h.assign(fh.begin(), fh.end());
        if (sq != mtp_cur_seq) mtp_switch_seq(sq);
        // this sequence's own first/last position and its own last hidden row
        const float * last_h = ((int64_t) fh.size() >= (int64_t) fp.size() * n_embd)
                             ? fh.data() + (size_t) (fp.size() - 1) * n_embd : nullptr;
        mtp_seq_bookkeep((int32_t) fp.front(), (int32_t) fp.back(), last_h, n_embd);
        mtp_kv_fill();
    }
    if (mtp_cur_seq != save_seq) mtp_switch_seq(save_seq);
    mtp_fill_pos = all_pos; mtp_fill_tok = all_tok; mtp_fill_seq = all_seq; mtp_fill_h.assign(all_h.begin(), all_h.end());
}

void llama_context::mtp_kv_fill() {
    // Fill the draft head's KV for every position covered by the decode that just ran.
    //
    // The head's K and V at a column depend only on (h_{p-1}, token_p): embed -> enorm/hnorm ->
    // eh_proj -> attn_norm -> wk/wv. None of that touches the head's own attention, so an entire
    // prompt's columns are computable in one batched pass rather than one per draft window. Filling
    // the head's KV across the prompt during prefill is what lets draft 0 see real context; it is the
    // prefill-side half of the decode-side increment in llama-context.h.
    static const bool dbg = getenv("ALPHA_MTP_DBG") != nullptr;
    const int64_t n_row  = (int64_t) mtp_fill_pos.size();
    const int64_t n_embd = mtp_hw();   // the exported hidden's width (hc*n_embd on qwen4exp)
    // ALPHA_MTP_DBG: one refused column stops the head KV from growing (see the range rules below), so
    // announce which gate stopped us.
    if (dbg) fprintf(stderr, "[mtp-fill] enter rows=%lld pos=[%d..%d] hi=%d\n",
                     (long long) n_row, n_row ? mtp_fill_pos[0] : -1,
                     n_row ? mtp_fill_pos[n_row-1] : -1, mtp_kv_hi);
    if (n_row < 1 || (int64_t) mtp_fill_tok.size() != n_row) {
        if (dbg) fprintf(stderr, "[mtp-fill] bail: rows/tok mismatch\n"); return; }
    if ((int64_t) mtp_fill_h.size() < n_embd * n_row) {
        if (dbg) fprintf(stderr, "[mtp-fill] bail: h buffer short\n"); return; }

    // The rows are written with tensor_get_async during decode; make them real before any host
    // read, including the carry-over copy below.
    {
        static const bool ftm = getenv("ALPHA_MTP_FILL_TIMERS") != nullptr;
        const int64_t t0 = ftm ? ggml_time_us() : 0;
        ggml_backend_sched_synchronize(sched.get());
        if (ftm) fprintf(stderr, "[mtp-fill-t] rows %lld: entry sync %.1f ms\n", (long long) n_row, (ggml_time_us() - t0) / 1000.0);
    }

    // Column p is built from (h_{p-1}, token_p). Within this decode that pairs hidden row i with
    // token row i+1; the first column additionally pairs the row carried over from the previous
    // decode with token row 0, which is the only way to write the boundary column at all (a
    // single-token decode contributes no internal pair).
    std::vector<const float *> hsrc;
    std::vector<llama_token>   ctok;
    std::vector<llama_pos>     cpos;
    std::vector<float>         prev_row;   // stable copy: mtp_prev_h is reassigned below
    if (mtp_prev_pos >= 0 && (int64_t) mtp_prev_h.size() >= n_embd &&
        mtp_prev_pos + 1 == mtp_fill_pos[0]) {
        prev_row = mtp_prev_h;
        hsrc.push_back(prev_row.data()); ctok.push_back(mtp_fill_tok[0]); cpos.push_back(mtp_fill_pos[0]);
    }
    for (int64_t i = 0; i + 1 < n_row; ++i) {
        if (mtp_fill_pos[i + 1] != mtp_fill_pos[i] + 1) break;   // consecutive slots only
        if (!hsrc.empty() && mtp_fill_pos[i + 1] != cpos.back() + 1) break;
        hsrc.push_back(mtp_fill_h.data() + (size_t) i * n_embd);
        ctok.push_back(mtp_fill_tok[i + 1]); cpos.push_back(mtp_fill_pos[i + 1]);
    }
    // Remember this decode's last row for the next call's boundary column.
    mtp_prev_h.assign(mtp_fill_h.data() + (size_t) (n_row - 1) * n_embd,
                      mtp_fill_h.data() + (size_t) n_row * n_embd);
    mtp_prev_pos = mtp_fill_pos[n_row - 1];

    int64_t m = (int64_t) cpos.size();
    if (m < 1) { if (dbg) fprintf(stderr, "[mtp-fill] bail: no columns built\n"); return; }
    const int32_t slot0 = (int32_t) cpos[0] - 1;
    // Range rules. An empty cache may begin its range anywhere: the head then attends over a suffix of
    // history rather than over nothing, which is a sliding window (positions are RoPE'd absolutely, so
    // relative offsets within [lo, hi] stay correct). Accepting only slot0 == 0 here would let one refused
    // first fill disable the head KV for the life of the process. A non-empty cache still refuses anything
    // that would punch a hole in [lo, hi].
    if (slot0 < 0) {
        if (dbg) fprintf(stderr, "[mtp-fill] gap: slot0=%d -- negative, skipped\n", slot0);
        return;
    }
    if (mtp_kv_hi < 0) {
        mtp_kv_lo = slot0;                                   // start the range here, not at 0
        if (dbg && slot0 > 0)
            fprintf(stderr, "[mtp-fill] cache empty -- starting range at slot %d (suffix attention)\n",
                    slot0);
    } else if (slot0 > mtp_kv_hi + 1 || slot0 < mtp_kv_lo) {
        // A hole that can never be filled, so restart the range instead of refusing forever. The hidden
        // states for [hi+1, slot0) are gone (mtp_fill_h only holds the current decode's rows), so no later
        // fill could close the gap, and holding the stale range would block every future fill.
        // Restarting drops the old prefix and attends over a suffix of history, as the empty-cache path
        // does. It cannot thrash: a healthy sequence has slot0 == hi+1 and never takes this branch.
        //
        // This is recovery, not a root fix: the head KV is filled after the decode from captured hiddens
        // and may be refused, so it can fall behind the text KV. The root fix is to advance the head-KV
        // frontier in lockstep with the text KV frontier from the same cursor.
        static int64_t n_restart = 0;
        ++n_restart;
        if (n_restart <= 3 || (n_restart % 256) == 0) {
            fprintf(stderr, "[mtp-fill] RESTART #%lld: slot0=%d is outside range [%d,%d] and the "
                            "hiddens for the hole are gone -- dropping the old prefix and restarting "
                            "the head-KV at %d (suffix attention). Repeated restarts mean the fill is "
                            "not keeping up with the decode.\n",
                    (long long) n_restart, slot0, mtp_kv_lo, mtp_kv_hi, slot0);
        }
        mtp_kv_lo = slot0;
        mtp_kv_hi = -1;          // the fill below re-establishes hi from this slot0
    }

    if (model.arch == LLM_ARCH_QWEN4EXP) {
        if (!model.has_mtp4) return;
        const auto & hq = model.hparams;
        const int64_t kv_row4 = hq.n_embd_head_k() * hq.n_head_kv(hq.n_layer - 1);
        if (!mtp_kv_ensure(kv_row4, ggml_backend_buffer_get_type(model.mtp4.wq->buffer))) return;
        if (slot0 + m > mtp_kv_cols) m = mtp_kv_cols - slot0;
        if (m < 1) return;
        if (!mtp_kv_fill_qwen4exp(hsrc, ctok, cpos, slot0, m) && dbg) fprintf(stderr, "[mtp-fill] qwen4exp chunk failed\n");
        return;
    }
    const float   eps    = model.hparams.f_norm_rms_eps;
    const int mtp_il = (int) model.hparams.n_layer - 1;
    if (mtp_il < 0 || mtp_il >= (int) model.layers.size()) return;
    const auto & L  = model.layers[mtp_il];
    const auto & hp = model.hparams;
    if (!L.nextn.eh_proj || !L.nextn.enorm || !L.nextn.hnorm || !L.wq || !L.wk || !L.wv) {
        if (dbg) fprintf(stderr, "[mtp-fill] bail: head tensors absent\n"); return; }
    const int64_t n_hd  = hp.n_embd_head_k();
    const int64_t n_hkv = hp.n_head_kv(mtp_il);
    if (n_hd <= 0 || n_hkv <= 0) { if (dbg) fprintf(stderr, "[mtp-fill] bail: bad head geometry\n"); return; }
    const int64_t kv_row = n_hd * n_hkv;
    if (!mtp_kv_ensure(kv_row, ggml_backend_buffer_get_type(L.wq->buffer))) {
        if (dbg) fprintf(stderr, "[mtp-fill] bail: kv_ensure FAILED (alloc)\n"); return; }
    if (slot0 + m > mtp_kv_cols) m = mtp_kv_cols - slot0;
    if (m < 1) { if (dbg) fprintf(stderr, "[mtp-fill] bail: clamped to 0 (slot0=%d cols=%d)\n",
                                  slot0, mtp_kv_cols); return; }


    const bool do_rope  = hp.rope_type != LLAMA_ROPE_TYPE_NONE;
    const int  n_pos_pe = (int) hp.n_pos_per_embd();
    ggml_tensor * emb_w = L.nextn.embed_tokens ? L.nextn.embed_tokens : model.tok_embd;
    if (!emb_w) return;

    const bool use_alt = (bool) sched_alt;
    // Chunked so the [2*n_embd, chunk] concat stays small and the draft scheduler's compute buffer
    // never has to grow to prompt size.
    const int64_t CH = 256;
    for (int64_t base = 0; base < m; base += CH) {
        const int64_t mb = std::min<int64_t>(CH, m - base);
        ggml_init_params ip = { ggml_graph_overhead() + ggml_tensor_overhead() * 64, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return;
        // The MTP block's enorm / hnorm / shared_head.norm are plain RMSNorm weights: convert_hf_to_gguf.py
        // remaps mtp.pre_fc_norm_* and mtp.norm straight through, with no +1 layernorm1p offset (that path is
        // Nemotron-only), and the in-tree deepseek4 NextN reference applies them as ggml_mul(rms_norm(x), w).
        auto mtp_norm = [&](ggml_tensor * t, ggml_tensor * w) {
            return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w);
        };
        ggml_tensor * Hs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, mb); ggml_set_input(Hs);
        ggml_tensor * Tk = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb);         ggml_set_input(Tk);
        ggml_tensor * Ps = nullptr;
        if (do_rope) { Ps = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, mb * n_pos_pe); ggml_set_input(Ps); }

        ggml_tensor * E  = ggml_get_rows(ctx, emb_w, Tk);
        ggml_tensor * x  = ggml_mul_mat(ctx, L.nextn.eh_proj,
                               ggml_concat(ctx, mtp_norm(E, L.nextn.enorm), mtp_norm(Hs, L.nextn.hnorm), 0));
        ggml_tensor * xn = L.attn_norm ? ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), L.attn_norm) : x;
        ggml_tensor * kn = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, L.wk, xn), n_hd, n_hkv, mb);
        if (L.attn_k_norm) kn = ggml_mul(ctx, ggml_rms_norm(ctx, kn, eps), L.attn_k_norm);
        if (do_rope) {
            const int n_rot = (int) hp.n_rot(mtp_il);
            const int n_co  = (int) cparams.n_ctx_orig_yarn;
            if (hp.rope_type == LLAMA_ROPE_TYPE_MROPE || hp.rope_type == LLAMA_ROPE_TYPE_IMROPE) {
                int sec[GGML_MROPE_SECTIONS];
                std::copy(std::begin(hp.rope_sections), std::begin(hp.rope_sections) + GGML_MROPE_SECTIONS, sec);
                kn = ggml_rope_multi(ctx, kn, Ps, nullptr, n_rot, sec, hp.rope_type, n_co,
                                     cparams.rope_freq_base, cparams.rope_freq_scale, cparams.yarn_ext_factor,
                                     cparams.yarn_attn_factor, cparams.yarn_beta_fast, cparams.yarn_beta_slow);
            } else {
                kn = ggml_rope_ext(ctx, kn, Ps, nullptr, n_rot, hp.rope_type, n_co,
                                   cparams.rope_freq_base, cparams.rope_freq_scale, cparams.yarn_ext_factor,
                                   cparams.yarn_attn_factor, cparams.yarn_beta_fast, cparams.yarn_beta_slow);
            }
        }
        ggml_tensor * vn = ggml_mul_mat(ctx, L.wv, xn);
        const size_t coff = (size_t) (mtp_kv_base() + slot0 + base) * mtp_k_cache->nb[1];
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, kn, kv_row, mb),
                                      ggml_view_2d(ctx, mtp_k_cache, kv_row, mb, mtp_k_cache->nb[1], coff)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, vn, kv_row, mb),
                                      ggml_view_2d(ctx, mtp_v_cache, kv_row, mb, mtp_v_cache->nb[1], coff)));

        if (use_alt) set_draft_active(true);
        ggml_backend_sched_reset(sched.get());
        if (gf_res_prev) gf_res_prev->reset();
        bool ok = false;
        if (ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            // The rows are not contiguous once the carried-over boundary row is prepended, so
            // stage them. One memcpy of at most n_embd*chunk floats.
            std::vector<float> hb((size_t) n_embd * mb);
            for (int64_t i = 0; i < mb; ++i)
                memcpy(hb.data() + (size_t) i * n_embd, hsrc[base + i], (size_t) n_embd * sizeof(float));
            ggml_backend_tensor_set(Hs, hb.data(), 0, hb.size() * sizeof(float));
            std::vector<int32_t> tk((size_t) mb);
            for (int64_t i = 0; i < mb; ++i) tk[i] = (int32_t) ctok[base + i];
            ggml_backend_tensor_set(Tk, tk.data(), 0, (size_t) mb * sizeof(int32_t));
            if (Ps) {
                // M-RoPE broadcasts one position across all sections for text; the layout is
                // pos[section*m + i], matching llama_batch_allocr::ubatch_add.
                std::vector<int32_t> ps((size_t) mb * n_pos_pe);
                for (int j = 0; j < n_pos_pe; ++j)
                    for (int64_t i = 0; i < mb; ++i)
                        ps[(size_t) j*mb + i] = (int32_t) cpos[base + i];
                ggml_backend_tensor_set(Ps, ps.data(), 0, ps.size() * sizeof(int32_t));
            }
            ok = ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS;
        }
        if (use_alt) set_draft_active(false);
        ggml_free(ctx);
        if (!ok) { if (dbg) fprintf(stderr, "[mtp-fill] chunk compute failed\n"); return; }
        mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, (int32_t) (slot0 + base + mb - 1));
    }
    if (dbg) {
        static int64_t nf = 0, nc = 0; nf++; nc += m;
        if ((nf & 63) == 1) fprintf(stderr, "[mtp-fill] %lld fills, %lld columns, hi=%d\n",
                                    (long long) nf, (long long) nc, mtp_kv_hi);
    }
}

// Does this architecture build its MTP draft block in the model (LLM_GRAPH_TYPE_DECODER_MTP) rather than via
// mtp_draft's hand-built graph below? Opt in one architecture at a time.
//
// Returns false for every architecture for now. The enum value and this hook exist so the model side can be
// written against a fixed interface; the graph-input plumbing that feeds a model-built MTP graph lands together
// with its first consumer, so that it is never an untested path.
//
// Contract for a model-built MTP graph:
//   INPUTS   - the trunk hidden H captured by the last single-token decode, held host-side in
//              llama_context::mtp_hidden_host and refreshed per accepted token; plus the embedding of
//              the just-sampled token e_tok; plus that token's position.
//              H is not always n_embd wide: size it by the exported `t_mtp_hidden->ne[0]`. It is n_embd
//              for qwen35moe, but n_embd*hc for an hc architecture (qwen4exp): the MTP block takes the trunk
//              residual before the final hc mixer, nextn.hnorm is [n_embd, hc] and RMS-norms each stream
//              separately, and eh_proj runs per stream. hparams.n_embd_out() is the hparams-side width (it
//              falls back to n_embd when there is no widening), but prefer the producer's actual ne[0]:
//              over-reading a host buffer here is an overrun.
//              The capture and row-select paths size H with mtp_hw(). The hand-built qwen35moe draft graph
//              (mtp_draft) uploads and writes back exactly n_embd floats, so it refuses an architecture whose
//              exported width differs; qwen4exp has its own draft path (mtp_draft_qwen4exp).
//   KV       - a separate memory object. llama_kv_cache's ctor checks hparams.has_kv(il) before any filter
//              (llama-kv-cache.cpp), and the generic loader sets n_layer_kv_from_start = n_layer -
//              nextn_predict_layers (llama-model.cpp). The trunk's indexer cache inherits that through the
//              hparams_idx copy, so the trunk's memory contains nothing for the MTP layer: a rejected draft
//              cannot touch trunk cells, and rewind is a seq_rm on the MTP memory alone with no GDN fold.
//              Build it as a second llama_memory_hybrid_idx from an hparams copy (the hparams_idx trick,
//              because has_kv runs before the filter) with n_layer_kv_from_start = -1,
//              filter_attn = filter_idx = (il >= n_layer_real), filter_recr = false always; size it
//              n_ctx_seq, indexer the same cell count. Matches upstream c061df19's LLAMA_CONTEXT_TYPE_MTP.
//              Note: build_inp_mem_hybrid() still builds a recurrent input even with zero recurrent layers
//              (upstream expands inp_hyb->get_recr()->s_copy for that reason), so check that
//              llama_memory_recurrent tolerates zero layers before assuming it is a no-op.
//              compress_ratios[il] does not drive allocation (the indexer stores one raw key per token per
//              indexer layer; pooling happens at read time). It only needs to be > 0 for the QSA path to be
//              taken; at 0 the block falls back to dense attention, still correct but not QSA.
//              The hand-built path keeps an F32 head KV filled by mtp_kv_fill()/mtp_kv_fill_all(); a
//              model-built graph may instead use the memory module its own helpers expect (qwen4exp also
//              needs a QSA indexer cache). Whatever it uses must survive rewind on reject (see ALPHA_REPLAYSSM).
//   OUTPUT   - logits for exactly one drafted token, over n_vocab, as the graph result's logits tensor.
//              mtp_draft returns the argmax; it does not sample. Greedy output must equal plain decode.
//   FORBIDDEN- no write to the trunk's KV, and no dependence on attn_post_norm or model.output_norm
//              (qwen4exp has neither; requiring them is what makes the hand-built path arch-specific).
static bool alpha_arch_has_mtp_graph(llm_arch arch) {
    switch (arch) {
        // case LLM_ARCH_QWEN4EXP: return true;   // enable with its graph case + input plumbing
        default: return false;
    }
}

// See the contract above. Not implemented yet: the graph-input plumbing lands with its first consumer. It
// aborts rather than returning LLAMA_TOKEN_NULL because NULL is the "this model has no MTP head" signal, and
// returning it would silently downgrade an opted-in architecture to plain decode.
llama_token llama_context::mtp_draft_graph(llama_token e_tok) {
    GGML_UNUSED(e_tok);
    GGML_ABORT("[ALPHA_MTP] arch opted into LLM_GRAPH_TYPE_DECODER_MTP but the model-built MTP route is "
               "not wired yet: add the architecture's LLM_GRAPH_TYPE_DECODER_MTP case in "
               "llama-model.cpp's build_graph switch, and the hidden/token/position graph inputs here. "
               "See the contract above alpha_arch_has_mtp_graph().");
}

llama_token llama_context::mtp_draft(llama_token e_tok) {
    if (model.arch == LLM_ARCH_QWEN4EXP) {
        return mtp_draft_qwen4exp(e_tok);   // llama-context-mtp4.cpp
    }
    // Architectures whose MTP block the hand-built graph below cannot express delegate to a model-built
    // LLM_GRAPH_TYPE_DECODER_MTP graph. No architecture opts in yet, so the qwen35moe path below is reached.
    if (alpha_arch_has_mtp_graph(model.arch)) {
        return mtp_draft_graph(e_tok);
    }
    // Style-B MTP draft: run the trained MTP block on the hidden H captured by the last single-token decode
    // + the embedding of e_tok (the just-sampled token). Builds a mini-graph that mirrors the qwen35moe MTP
    // head (concat[embed,hidden] after enorm/hnorm -> eh_proj -> attention + FFN + residual ->
    // shared_head_norm -> head) and returns its argmax.
    static const bool dbg = getenv("ALPHA_MTP_DBG") != nullptr;
    if (mtp_hw() != model.hparams.n_embd) {
        // the graph below reads and writes n_embd floats of H; a wider exported hidden needs its own draft path
        static bool warned = false;
        if (!warned) { warned = true;
            fprintf(stderr, "[mtp] disabled: exported hidden width %lld != n_embd %lld for this architecture\n",
                    (long long) mtp_hw(), (long long) model.hparams.n_embd);
        }
        return LLAMA_TOKEN_NULL;
    }
    if (mtp_hidden_host.empty()) {
        static bool warned = false;
        // Printed without ALPHA_MTP_DBG, so setting ALPHA_MTP on a model with no MTP head is not silent.
        // One-shot, so it cannot spam.
        if (!warned) { warned = true;
            fprintf(stderr, "[mtp] DISABLED: this model exports no MTP hidden state "
                            "(no blk.N.nextn.* tensors). Decoding normally.\n");
        }
        (void) dbg;
        return LLAMA_TOKEN_NULL;
    }
    const int64_t n_embd   = model.hparams.n_embd;
    const int64_t n_vocab  = model.vocab.n_tokens();
    const int64_t n_expert = model.hparams.n_expert;
    const int64_t n_used   = model.hparams.n_expert_used;
    const float   eps      = model.hparams.f_norm_rms_eps;
    const float   wscale   = model.hparams.expert_weights_scale;
    const int mtp_il = (int) model.hparams.n_layer - 1;
    if (mtp_il < 0 || mtp_il >= (int) model.layers.size()) return LLAMA_TOKEN_NULL;
    const auto & L = model.layers[mtp_il];
    // Accept either FFN kind: a dense MTP block (e.g. arch qwen35) has no expert tensors and takes the
    // dense SwiGLU path below. nextn.* + tok_embd are required either way.
    const bool mtp_has_moe   = L.ffn_gate_inp && L.ffn_gate_exps && L.ffn_up_exps && L.ffn_down_exps;
    const bool mtp_has_dense = L.ffn_gate && L.ffn_up && L.ffn_down;
    if (!L.nextn.eh_proj || !L.nextn.enorm || !L.nextn.hnorm || !L.attn_post_norm ||
        !model.tok_embd || (!mtp_has_moe && !mtp_has_dense)) {
        static bool w=false; if(dbg&&!w){w=true;fprintf(stderr,"[mtp-dbg] MTP block tensors not resident\n");}
        return LLAMA_TOKEN_NULL;
    }
    ggml_tensor * head = L.nextn.shared_head_head ? L.nextn.shared_head_head : model.output;
    if (!head) return LLAMA_TOKEN_NULL;

    ggml_init_params ip = { ggml_graph_overhead() + ggml_tensor_overhead()*256, nullptr, /*no_alloc*/ true };
    // ALPHA_MTP_HEAD_REPLAY (default on; =0 restores the per-call context): see llama-context.h
    static const bool head_replay = []() {
        const char * e = getenv("ALPHA_MTP_HEAD_REPLAY");
        return e ? atoi(e) != 0 : true;
    }();
    ggml_context * ctx = nullptr;
    if (head_replay) {
        if (!mtp_draft_ctx) {
            mtp_draft_ctx = ggml_init(ip);
        } else {
            ggml_reset(mtp_draft_ctx);
        }
        ctx = mtp_draft_ctx;
    } else {
        ctx = ggml_init(ip);
    }
    if (!ctx) return LLAMA_TOKEN_NULL;
    ggml_tensor * kv_idx_t  = nullptr;   // [1] i32: cache column written by set_rows (replay mode)
    ggml_tensor * kq_mask_t = nullptr;   // [T_pad, n_rep] f32: 0 inside the window, -inf on the padding
    int32_t       kv_idx_v  = -1;
    int64_t       kq_T = 0, kq_T_pad = 0, kq_n_rep = 0;

    auto mtp_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w);
    };

    ggml_tensor * H    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); ggml_set_input(H);
    ggml_tensor * etok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);      ggml_set_input(etok);
    // The MTP head ships its own embed table (nextn.embed_tokens); use it when present (it may be
    // untied from the main tok_embd), else fall back to the main embedding.
    ggml_tensor * emb_w = L.nextn.embed_tokens ? L.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * E  = ggml_get_rows(ctx, emb_w, etok);                                  // [n_embd,1]
    ggml_tensor * en = mtp_norm(E, L.nextn.enorm);
    ggml_tensor * hn = mtp_norm(H, L.nextn.hnorm);
    ggml_tensor * x  = ggml_mul_mat(ctx, L.nextn.eh_proj, ggml_concat(ctx, en, hn, 0));  // concat[embed,hidden]
    if (getenv("ALPHA_MTP_NAN_DBG")) { g_mtp_nan_early.clear(); g_mtp_nan_early.push_back({"hn", hn}); g_mtp_nan_early.push_back({"x_proj", x}); }
    // MTP transformer block. The head ships real attention weights (attn_q [n_embd, n_hd*2*n_h],
    // attn_k/v [n_embd, n_embd_gqa], attn_output [n_hd*n_h, n_embd], q/k norms), all of which are used.
    // Cache writes must join the graph, which is created here, before the attention, so the cache-write
    // nodes can be expanded the moment they are created: ggml executes nodes in construction order and
    // does not track buffer aliasing between a ggml_cpy into a slot and a ggml_view reading it, so writes
    // expanded after the reads would let the attention see a stale column.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_tensor * mtp_kw = nullptr, * mtp_vw = nullptr;
    ggml_tensor * rope_pos_t = nullptr;   // [n_pos_per_embd] i32, filled after alloc
    int32_t       rope_pos_v = 0;
    if (L.wq && L.wk && L.wv && L.wo) {
        const int64_t n_hd  = model.hparams.n_embd_head_k();
        const int64_t n_h   = model.hparams.n_head(mtp_il);
        const int64_t n_hkv = model.hparams.n_head_kv(mtp_il);
        if (n_hd > 0 && n_hkv > 0 && n_h >= n_hkv && (n_h % n_hkv) == 0) {
            const int64_t n_rep  = n_h / n_hkv;
            const int64_t kv_row = n_hd * n_hkv;
            mtp_kv_ensure(kv_row, ggml_backend_buffer_get_type(L.wq->buffer));
            ggml_tensor * xn = L.attn_norm ? ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), L.attn_norm) : x;
            ggml_tensor * qg = ggml_mul_mat(ctx, L.wq, xn);              // [n_hd*2*n_h, 1]
            // q and gate are interleaved per head: q at 0, gate at +n_hd, stride 2*n_hd.
            ggml_tensor * q = ggml_view_2d(ctx, qg, n_hd, n_h,
                                           ggml_element_size(qg) * n_hd * 2, 0);
            q = ggml_cont(ctx, q);
            if (L.attn_q_norm) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), L.attn_q_norm);
            ggml_tensor * gate = ggml_view_2d(ctx, qg, n_hd, n_h,
                                              ggml_element_size(qg) * n_hd * 2,
                                              ggml_element_size(qg) * n_hd);
            gate = ggml_cont_2d(ctx, gate, n_hd * n_h, 1);
            ggml_tensor * kn = ggml_mul_mat(ctx, L.wk, xn);              // [n_hd*n_hkv, 1]
            kn = ggml_reshape_2d(ctx, kn, n_hd, n_hkv);
            if (L.attn_k_norm) kn = ggml_mul(ctx, ggml_rms_norm(ctx, kn, eps), L.attn_k_norm);
            ggml_tensor * vn = ggml_mul_mat(ctx, L.wv, xn);              // [n_hd*n_hkv, 1]

            // Rotary position embedding for the draft column, matching the main attention layers the
            // head was trained with; without it the head would attend over its own cached history with
            // no positional information.
            // Cached K is stored post-rotation at its own position, as the main KV cache does, so a query
            // rotated at rope_pos_v sees the correct relative offset to every stored column. Chain step s
            // sits at mtp_pos_base + 1 + s: the position of the token the head embeds (draft 0 embeds
            // `last`, which occupies mtp_pos_base + 1).
            // Generic across architectures: MROPE/IMROPE take ggml_rope_multi with the model's
            // sections, everything else takes ggml_rope_ext, and rope_type NONE opts out.
            const auto & hp = model.hparams;
            // Sequence position of the column this draft writes: the position of the token it
            // embeds. Chain step s of a window sits at mtp_pos_base + 1 + s.
            const int32_t col_pos = mtp_pos_base >= 0 ? mtp_pos_base + 1 + mtp_chain_step : -1;
            if (col_pos >= 0 && hp.rope_type != LLAMA_ROPE_TYPE_NONE) {
                rope_pos_v = col_pos;
                const int n_pos_pe   = (int) hp.n_pos_per_embd();
                const int n_rot      = (int) hp.n_rot(mtp_il);
                const int n_ctx_orig = (int) cparams.n_ctx_orig_yarn;
                rope_pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_pos_pe);
                ggml_set_input(rope_pos_t);
                ggml_tensor * q3 = ggml_reshape_3d(ctx, q,  n_hd, n_h,   1);
                ggml_tensor * k3 = ggml_reshape_3d(ctx, kn, n_hd, n_hkv, 1);
                if (hp.rope_type == LLAMA_ROPE_TYPE_MROPE || hp.rope_type == LLAMA_ROPE_TYPE_IMROPE) {
                    int sec[GGML_MROPE_SECTIONS];
                    std::copy(std::begin(hp.rope_sections), std::begin(hp.rope_sections) + GGML_MROPE_SECTIONS, sec);
                    q3 = ggml_rope_multi(ctx, q3, rope_pos_t, nullptr, n_rot, sec, hp.rope_type,
                                         n_ctx_orig, cparams.rope_freq_base, cparams.rope_freq_scale,
                                         cparams.yarn_ext_factor, cparams.yarn_attn_factor,
                                         cparams.yarn_beta_fast, cparams.yarn_beta_slow);
                    k3 = ggml_rope_multi(ctx, k3, rope_pos_t, nullptr, n_rot, sec, hp.rope_type,
                                         n_ctx_orig, cparams.rope_freq_base, cparams.rope_freq_scale,
                                         cparams.yarn_ext_factor, cparams.yarn_attn_factor,
                                         cparams.yarn_beta_fast, cparams.yarn_beta_slow);
                } else {
                    q3 = ggml_rope_ext(ctx, q3, rope_pos_t, nullptr, n_rot, hp.rope_type,
                                       n_ctx_orig, cparams.rope_freq_base, cparams.rope_freq_scale,
                                       cparams.yarn_ext_factor, cparams.yarn_attn_factor,
                                       cparams.yarn_beta_fast, cparams.yarn_beta_slow);
                    k3 = ggml_rope_ext(ctx, k3, rope_pos_t, nullptr, n_rot, hp.rope_type,
                                       n_ctx_orig, cparams.rope_freq_base, cparams.rope_freq_scale,
                                       cparams.yarn_ext_factor, cparams.yarn_attn_factor,
                                       cparams.yarn_beta_fast, cparams.yarn_beta_slow);
                }
                q  = ggml_reshape_2d(ctx, q3, n_hd, n_h);
                kn = ggml_reshape_2d(ctx, k3, n_hd, n_hkv);
            }

            ggml_tensor * att_out = nullptr;
            // Slot j holds the column for position j+1, so this draft owns slot col_pos-1. Refuse
            // to write past the end of what the fill has laid down (that would leave a hole the
            // causal read would consume as garbage) and fall back to the single-position form.
            // Eligible when this column extends the valid range (or opens it). `lo` is where the
            // range actually starts -- see mtp_kv_lo: an empty cache may begin anywhere, so the
            // window is [lo, pos], not [0, pos].
            const int32_t lo0 = (mtp_kv_lo >= 0) ? mtp_kv_lo : (col_pos - 1);
            const int32_t pos = (mtp_k_cache && col_pos >= 1 && col_pos - 1 < mtp_kv_cols &&
                                 col_pos - 1 <= mtp_kv_hi + 1 && col_pos - 1 >= lo0) ? col_pos - 1 : -1;
            if (pos >= 0) {
                // Append this column, then attend over [0, pos].
                if (head_replay) {
                    // column index as a graph input: the write node is identical for every position
                    kv_idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
                    ggml_set_input(kv_idx_t);
                    kv_idx_v = mtp_kv_base() + pos;
                    mtp_kw = ggml_set_rows(ctx, mtp_k_cache, ggml_reshape_2d(ctx, kn, kv_row, 1), kv_idx_t);
                    ggml_build_forward_expand(gf, mtp_kw);
                    mtp_vw = ggml_set_rows(ctx, mtp_v_cache, ggml_reshape_2d(ctx, vn, kv_row, 1), kv_idx_t);
                    ggml_build_forward_expand(gf, mtp_vw);
                } else {
                    ggml_tensor * kslot = ggml_view_2d(ctx, mtp_k_cache, kv_row, 1, mtp_k_cache->nb[1],
                                                       (size_t) (mtp_kv_base() + pos) * mtp_k_cache->nb[1]);
                    ggml_tensor * vslot = ggml_view_2d(ctx, mtp_v_cache, kv_row, 1, mtp_v_cache->nb[1],
                                                       (size_t) (mtp_kv_base() + pos) * mtp_v_cache->nb[1]);
                    mtp_kw = ggml_cpy(ctx, ggml_reshape_2d(ctx, kn, kv_row, 1), kslot);
                    ggml_build_forward_expand(gf, mtp_kw);
                    mtp_vw = ggml_cpy(ctx, ggml_reshape_2d(ctx, vn, kv_row, 1), vslot);
                    ggml_build_forward_expand(gf, mtp_vw);
                }
                // Attend over [lo0, pos], offset by mtp_kv_base() (the per-sequence region): a cache that
                // began mid-sequence reads only the columns it actually wrote, and seq > 0 does not read
                // another sequence's columns.
                const int64_t T    = (int64_t) (pos - lo0 + 1);
                const size_t  woff = (size_t) (mtp_kv_base() + lo0);
                // replay mode: attend over a window padded to 64 columns with a mask input, so the graph
                // only changes shape every 64 tokens (columns past hi are finite: cleared at allocation)
                int64_t T_use = T;
                if (head_replay) {
                    const int64_t avail = (int64_t) mtp_kv_cols - (int64_t) lo0;
                    T_use = std::min<int64_t>(((T + 63) / 64) * 64, avail);
                    if (T_use < T) T_use = T;
                    kq_T = T; kq_T_pad = T_use; kq_n_rep = n_rep;
                    kq_mask_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_use, n_rep);
                    ggml_set_input(kq_mask_t);
                }
                ggml_tensor * Kc = ggml_view_3d(ctx, mtp_k_cache, n_hd, n_hkv, T_use,
                                                ggml_element_size(mtp_k_cache) * n_hd,
                                                mtp_k_cache->nb[1], woff * mtp_k_cache->nb[1]);
                ggml_tensor * Vc = ggml_view_3d(ctx, mtp_v_cache, n_hd, n_hkv, T_use,
                                                ggml_element_size(mtp_v_cache) * n_hd,
                                                mtp_v_cache->nb[1], woff * mtp_v_cache->nb[1]);
                // GQA: expand kv heads to query heads, then scores = K^T q per head.
                Kc = ggml_cont(ctx, ggml_permute(ctx, Kc, 0, 2, 1, 3));      // [n_hd, T, n_hkv]
                Vc = ggml_cont(ctx, ggml_permute(ctx, Vc, 0, 2, 1, 3));
                ggml_tensor * qh = ggml_reshape_3d(ctx, q, n_hd, n_rep, n_hkv);
                ggml_tensor * sc = ggml_mul_mat(ctx, Kc, qh);                // [T, n_rep, n_hkv]
                // Same expression the main attention layers use (llm_build_qwen35): an arch that
                // sets f_attention_scale does not use 1/sqrt(d).
                if (head_replay) {
                    sc = ggml_soft_max_ext(ctx, sc, kq_mask_t,
                                           hp.f_attention_scale == 0.0f ? 1.0f / sqrtf((float) n_hd) : hp.f_attention_scale, 0.0f);
                } else {
                    sc = ggml_scale(ctx, sc, hp.f_attention_scale == 0.0f
                                                 ? 1.0f / sqrtf((float) n_hd) : hp.f_attention_scale);
                    sc = ggml_soft_max(ctx, sc);
                }
                ggml_tensor * Vt = ggml_cont(ctx, ggml_permute(ctx, Vc, 1, 0, 2, 3)); // [T, n_hd, n_hkv]
                ggml_tensor * o  = ggml_mul_mat(ctx, Vt, sc);                // [n_hd, n_rep, n_hkv]
                att_out = ggml_cont_2d(ctx, o, n_hd * n_h, 1);
            } else {
                // Cache unavailable/full: fall back to the single-position form (softmax over one
                // key is 1, so the output is this head group's V).
                ggml_tensor * vg = ggml_reshape_3d(ctx, vn, n_hd, 1, n_hkv);
                vg = ggml_repeat_4d(ctx, vg, n_hd, n_rep, n_hkv, 1);
                att_out = ggml_cont_2d(ctx, vg, n_hd * n_h, 1);
            }
            ggml_tensor * att = ggml_mul_mat(ctx, L.wo, ggml_mul(ctx, att_out, ggml_sigmoid(ctx, gate)));
            if (getenv("ALPHA_MTP_NAN_DBG")) { g_mtp_nan_early.push_back({"att_out", att_out}); g_mtp_nan_early.push_back({"att", att}); }
            x = ggml_add(ctx, x, att);                                   // attention residual
        }
    }
    ggml_tensor * ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), L.attn_post_norm);  // plain norm
    // The MTP block carries whatever FFN kind its arch uses: dense (arch qwen35) or MoE (arch
    // qwen35moe). Take the dense SwiGLU path when this block has no experts, otherwise the MoE
    // tensors below are null and the graph builds garbage.
    ggml_tensor * moe = nullptr;
    if (n_expert <= 0 || !L.ffn_gate_exps) {
        if (!L.ffn_gate || !L.ffn_up || !L.ffn_down) {
            if (dbg) fprintf(stderr, "[mtp-dbg] MTP block has neither expert nor dense FFN tensors\n");
            return LLAMA_TOKEN_NULL;
        }
        ggml_tensor * dup = ggml_mul_mat(ctx, L.ffn_up,   ffn_in);
        ggml_tensor * dgt = ggml_mul_mat(ctx, L.ffn_gate, ffn_in);
        moe = ggml_mul_mat(ctx, L.ffn_down, ggml_mul(ctx, ggml_silu(ctx, dgt), dup));
    } else {
    // routed MoE (mirrors build_moe_ffn: softmax gating, norm_w, silu, weighted expert sum)
    ggml_tensor * probs = ggml_soft_max(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp, ffn_in)); // [n_expert,1]
    ggml_tensor * sel   = ggml_argsort_top_k(ctx, probs, n_used);                        // [n_used,1] i32
    ggml_tensor * w2    = ggml_reshape_2d(ctx, ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, 1), sel), n_used, 1);
    w2 = ggml_div(ctx, w2, ggml_clamp(ctx, ggml_sum_rows(ctx, w2), 6.103515625e-5f, 1e30f)); // norm_w
    if (wscale != 0.0f && wscale != 1.0f) w2 = ggml_scale(ctx, w2, wscale);
    ggml_tensor * wexp = ggml_reshape_3d(ctx, w2, 1, n_used, 1);
    ggml_tensor * cur3 = ggml_reshape_3d(ctx, ffn_in, n_embd, 1, 1);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur3, sel);               // [n_ff,n_used,1]
    ggml_tensor * gate = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur3, sel);
    ggml_tensor * act  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * experts = ggml_mul(ctx, ggml_mul_mat_id(ctx, L.ffn_down_exps, act, sel), wexp); // [n_embd,n_used,1]
    moe = ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2], 0);
    for (int64_t i = 1; i < n_used; ++i) {
        moe = ggml_add(ctx, moe, ggml_view_2d(ctx, experts, n_embd, 1, experts->nb[2], (size_t) i*experts->nb[1]));
    }
    if (L.ffn_up_shexp && L.ffn_gate_shexp && L.ffn_down_shexp) {   // gated shared expert
        ggml_tensor * su = ggml_mul_mat(ctx, L.ffn_up_shexp, ffn_in);
        ggml_tensor * sg = ggml_mul_mat(ctx, L.ffn_gate_shexp, ffn_in);
        ggml_tensor * sh = ggml_mul_mat(ctx, L.ffn_down_shexp, ggml_mul(ctx, ggml_silu(ctx, sg), su));
        if (L.ffn_gate_inp_shexp) {
            sh = ggml_mul(ctx, sh, ggml_sigmoid(ctx, ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, ffn_in)));
        }
        moe = ggml_add(ctx, moe, sh);
    }
    }   // end MoE branch
    ggml_tensor * out = ggml_add(ctx, x, moe);                                            // residual
    // ALPHA_MTP_NAN_DBG: mark the head's intermediates as outputs so the first draws can show
    // which op first goes non-finite.
    static std::vector<std::pair<const char *, ggml_tensor *>> nan_probes;
    if (getenv("ALPHA_MTP_NAN_DBG")) {
        nan_probes.clear();
        for (auto & pe : g_mtp_nan_early) if (pe.second) { ggml_set_output(pe.second); ggml_build_forward_expand(gf, pe.second); nan_probes.push_back(pe); }
        for (auto pr : { std::make_pair("E", E), std::make_pair("x_resid", x),
                         std::make_pair("ffn_in", ffn_in), std::make_pair("moe", moe), std::make_pair("out", out) }) {
            if (pr.second) { ggml_set_output(pr.second); ggml_build_forward_expand(gf, pr.second); nan_probes.push_back(pr); }
        }
    }
    // `out` is the MTP block's output hidden, i.e. the hidden state for the next position. Export it
    // so the next chained draft starts from the advanced hidden rather than reusing h_t for every
    // step, which makes acceptance decay quickly with depth.
    ggml_set_output(out);
    ggml_tensor * head_norm = L.nextn.shared_head_norm ? mtp_norm(out, L.nextn.shared_head_norm)
                                                        : ggml_mul(ctx, ggml_rms_norm(ctx, out, eps), model.output_norm);
    // Draft LM-head shortlist. The head GEMV dominates the draft step (output.weight is ~1 GiB at
    // q6_K for a 248K vocab). Qwen's vocab is merge-ordered (low ids are the frequent tokens, ids
    // above ~120k are almost entirely non-Latin scripts), so for Latin/code text a leading row range
    // is already a frequency shortlist. Rows are token ids, so this is a plain view: no gather, no
    // index table, no extra VRAM, and whole rows at offset 0 keep a quantised src0 block-aligned.
    //
    // Only the draft is shortlisted. The verify still runs the full head, so emitted tokens remain
    // the target model's and output correctness is unaffected; this can only change which tokens
    // get proposed, i.e. acceptance and speed.
    //
    // Workload-dependent by construction: a target token whose id is outside the shortlist can never
    // be proposed, so this trades acceptance on rare/non-Latin tokens for speed on common ones
    // (about +3% on code, neutral on CJK at the default). ALPHA_MTP_DRAFT_HEAD_ROWS (default 131072;
    // 0 restores the full head).
    //
    // The cut is an absolute row count, so its aggressiveness depends on n_vocab (about half the rows
    // here, a quarter on a 500K vocab, a no-op below 131072 rows), and its safety rests on the model's
    // id layout: a much smaller cut loses heavily on Chinese because CJK ids sit above it. Before
    // serving a model with a different vocab size or ordering, re-check greedy identity with a
    // non-Latin prompt, or switch this to a fraction of n_vocab.
    ggml_tensor * dhead = head;
    {
        // 0 (or a value >= n_vocab) = full head
        static const int64_t hrows = getenv("ALPHA_MTP_DRAFT_HEAD_ROWS")
                                   ? (int64_t) atoll(getenv("ALPHA_MTP_DRAFT_HEAD_ROWS")) : 131072;
        if (hrows > 0 && hrows < head->ne[1]) {
            dhead = ggml_view_2d(ctx, head, head->ne[0], hrows, head->nb[1], 0);
        }
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, dhead, head_norm);                            // [n_vocab,1]
    // GPU-side argmax: read back a single int (avoids a ~1MB logits DMA + host scan per draft).
    // Top-2 instead of top-1: the top-1/top-2 logits feed the margin gate (ALPHA_MTP_MARGIN), and
    // the top-2 costs about the same as the argmax. ALPHA_MTP_TOP2=0 restores the plain argmax.
    static const bool mtp_top2 = []() {
        const char * e = getenv("ALPHA_MTP_TOP2");
        return !e || atoi(e) != 0;   // default on; harmless when the caller ignores the 2nd slot
    }();
    ggml_tensor * amax = mtp_top2 ? ggml_argsort_top_k(ctx, logits, 2)   // [2] i32, rank-ordered
                                  : ggml_argmax(ctx, logits);            // [1] i32
    ggml_set_output(amax);
    // ALPHA_MTP_MARGIN: gather the top-2 logit values alongside their indices. logits is
    // [n_vocab]; reshape to [1, n_vocab] so ggml_get_rows can index it by the argsort output.
    // ALPHA_MTP_TOP2_VALS=0 drops only the value gather, keeping the argsort and the runner-up
    // index.
    static const bool top2_vals = []() {
        const char * e = getenv("ALPHA_MTP_TOP2_VALS");
        return !e || atoi(e) != 0;   // default ON
    }();
    ggml_tensor * top2v = nullptr;
    if (mtp_top2 && top2_vals) {
        // Size this from logits, not from n_vocab: with the draft head shortlist above active
        // (ALPHA_MTP_DRAFT_HEAD_ROWS=N < n_vocab) `dhead` is a row view, so `logits` holds N
        // elements, and ggml_reshape_2d asserts on the element count.
        const int64_t n_dlog = logits->ne[0];
        top2v = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, n_dlog), amax);  // [1,2] f32
        ggml_set_output(top2v);
    }
    (void) n_vocab;

    // Swap to the second scheduler for the draft. Everything below then resets and allocates on
    // the alt sched/graph-cache, leaving the main decode's cached graph intact.
    const bool mtp_use_alt = (bool) sched_alt;
    if (mtp_use_alt) set_draft_active(true);
    ggml_build_forward_expand(gf, amax);
    if (top2v)   ggml_build_forward_expand(gf, top2v);

    ggml_backend_sched_reset(sched.get());
    if (gf_res_prev) gf_res_prev->reset();
    llama_token best = LLAMA_TOKEN_NULL;
    if (ggml_backend_sched_alloc_graph(sched.get(), gf)) {
        int32_t ti = (int32_t) e_tok;
        ggml_backend_tensor_set(etok, &ti, 0, sizeof(int32_t));
        ggml_backend_tensor_set(H, mtp_hidden_host.data(), 0, (size_t) n_embd * sizeof(float));
        // Both rotated tensors are dead when the draft falls back to the single-position form
        // (it consumes only V), so the scheduler never allocates this input; setting it then trips
        // "tensor buffer not set".
        if (rope_pos_t && rope_pos_t->buffer) {
            // M-RoPE broadcasts the same position across all sections for text tokens
            // (llama_batch_allocr::ubatch_add does the same); non-MRoPE has a single entry.
            int32_t rp[GGML_MROPE_SECTIONS];
            for (int i = 0; i < GGML_MROPE_SECTIONS; ++i) rp[i] = rope_pos_v;
            ggml_backend_tensor_set(rope_pos_t, rp, 0, ggml_nbytes(rope_pos_t));
        }
        if (kv_idx_t && kv_idx_t->buffer) {
            ggml_backend_tensor_set(kv_idx_t, &kv_idx_v, 0, sizeof(int32_t));
        }
        if (kq_mask_t && kq_mask_t->buffer) {
            std::vector<float> mk((size_t) kq_T_pad * kq_n_rep, 0.0f);
            for (int64_t r = 0; r < kq_n_rep; ++r) {
                for (int64_t j = kq_T; j < kq_T_pad; ++j) mk[(size_t) r * kq_T_pad + j] = -INFINITY;
            }
            ggml_backend_tensor_set(kq_mask_t, mk.data(), 0, mk.size() * sizeof(float));
        }
        if (ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS) {
            // ALPHA_MTP_NAN_DBG: first draws only: is the head's input hidden already bad, or does
            // the head itself produce NaN/garbage? (Draft token 0 = "!" = the NaN-logits signature.)
            static const bool nan_dbg = getenv("ALPHA_MTP_NAN_DBG") != nullptr;
            static int nan_dbg_n = 0;
            if (nan_dbg && nan_dbg_n < 6) {
                nan_dbg_n++;
                auto stats = [](const float * v, int64_t n, double & l2, int64_t & bad) {
                    l2 = 0.0; bad = 0;
                    for (int64_t i = 0; i < n; ++i) { if (!std::isfinite(v[i])) bad++; else l2 += (double) v[i] * v[i]; }
                    l2 = std::sqrt(l2);
                };
                double l2_in, l2_out; int64_t bad_in, bad_out;
                stats(mtp_hidden_host.data(), n_embd, l2_in, bad_in);
                std::vector<float> o((size_t) n_embd);
                ggml_backend_tensor_get(out, o.data(), 0, o.size() * sizeof(float));
                stats(o.data(), n_embd, l2_out, bad_out);
                fprintf(stderr, "[mtp-nan] draw %d: hidden_in |h|=%.3f nonfinite=%lld | head_out |h|=%.3f nonfinite=%lld\n",
                        nan_dbg_n, l2_in, (long long) bad_in, l2_out, (long long) bad_out);
                for (auto & pr : nan_probes) {
                    if (!pr.second->buffer || pr.second->type != GGML_TYPE_F32) continue;
                    const int64_t ne = ggml_nelements(pr.second);
                    std::vector<float> v((size_t) ne);
                    ggml_backend_tensor_get(pr.second, v.data(), 0, v.size() * sizeof(float));
                    double l2; int64_t bad; stats(v.data(), ne, l2, bad);
                    fprintf(stderr, "[mtp-nan]   %-8s n=%lld |v|=%.3f nonfinite=%lld\n", pr.first, (long long) ne, l2, (long long) bad);
                }
            }
            int32_t bi = -1;
            if (mtp_top2) {
                int32_t bi2[2] = { -1, -1 };
                ggml_backend_tensor_get(amax, bi2, 0, 2 * sizeof(int32_t));
                bi = bi2[0];
                if (top2v) ggml_backend_tensor_get(top2v, mtp_top2_v, 0, 2 * sizeof(float));
            } else {
                ggml_backend_tensor_get(amax, &bi, 0, sizeof(int32_t));
            }
            // Advance the carried hidden for the next chained draft. Shares the sync point the
            // amax readback already forces, so it costs one small D2H and no extra stall.
            if ((int64_t) mtp_hidden_host.size() >= n_embd) {
                ggml_backend_tensor_get(out, mtp_hidden_host.data(), 0, (size_t) n_embd * sizeof(float));
            }
            best = (llama_token) bi;
            mtp_chain_step++;                       // next chained draft sits one position later
            if (mtp_kw) {
                // A draft column also opens the range if nothing has been laid down yet.
                if (mtp_kv_hi < 0) mtp_kv_lo = mtp_pos_base + mtp_chain_step - 1;
                mtp_kv_hi = std::max<int32_t>(mtp_kv_hi, mtp_pos_base + mtp_chain_step - 1);
            }
            if (dbg) { static int n=0; if(n<6){n++; fprintf(stderr,"[mtp-dbg] draft e=%d d0=%d\n",(int)e_tok,(int)best);} }
        } else if (dbg) { static bool w=false; if(!w){w=true;fprintf(stderr,"[mtp-dbg] mini-graph compute failed\n");} }
    } else if (dbg) { static bool w=false; if(!w){w=true;fprintf(stderr,"[mtp-dbg] sched_alloc failed\n");} }
    if (mtp_use_alt) set_draft_active(false);
    if (!head_replay) ggml_free(ctx);
    return best;
}

// Token for output row i, straight from the in-graph argmax.
// Indexed in submission order: the MTP verify submits one sequence with logits=true on every row and
// strictly increasing positions, so the output rows are already sorted and output_reorder() is a
// no-op for it. Guarded on size, and the caller falls back to the host scan on LLAMA_TOKEN_NULL.
llama_token llama_context::mtp_argmax_ith(int32_t i) const {
    if (i < 0 || (size_t) i >= mtp_argmax_host.size()) return LLAMA_TOKEN_NULL;
    return (llama_token) mtp_argmax_host[(size_t) i];
}

// Re-point the carried hidden at the accepted row. mtp_fill_h already holds every row of the verify,
// so this is a memcpy, not a recompute.
void llama_context::mtp_select_row(int32_t row) {
    const int64_t n_embd = mtp_hw();
    const int64_t n_row  = (int64_t) mtp_fill_pos.size();
    if (row < 0 || (int64_t) row >= n_row) return;
    if ((int64_t) mtp_fill_h.size() < (int64_t) (row + 1) * n_embd) return;
    mtp_hidden_host.assign(mtp_fill_h.begin() + (size_t) row * n_embd,
                           mtp_fill_h.begin() + (size_t) (row + 1) * n_embd);
    mtp_prev_h   = mtp_hidden_host;
    mtp_prev_pos = mtp_fill_pos[row];

    // The position must follow the row. decode() sets mtp_pos_base = p_last, the position of the
    // decode's last row, and the draft rotates at mtp_pos_base + 1 + chain_step. After a partial
    // accept the verify's last row is pos0+k while the accepted row is pos0+n_acc, so without this
    // the base is too large by (k - n_acc) and every chained draft is RoPE'd at the wrong position.
    mtp_pos_base   = mtp_fill_pos[row];
    mtp_chain_step = 0;
}

// ReplaySSM: replace the reject relay by folding the recurrent state over just the accepted prefix.
// See alpha_gdn_record.h for why this needs no kernel change and why S_0 costs nothing.
bool llama_context::gdn_fold(int32_t m) {
    if (!alpha_gdn_record_enabled()) return false;

    const auto & rec = alpha_gdn_record_get();
    if (!rec.valid)                return false;
    if (m < 0 || m > rec.n_tokens) return false;

    static const bool dbg = getenv("ALPHA_REPLAYSSM_DBG") != nullptr;

    // m == 0: nothing was accepted. The committed state is already S_0 (the publish was
    // suppressed) and tail_{K-1}(H_0) is the conv state at column offset 0, also already
    // committed, so a zero-accept fold is a no-op.
    if (m == 0) {
        if (dbg) fprintf(stderr, "[replayssm] fold m=0 -> no-op\n");
        alpha_gdn_record_invalidate();
        return true;
    }

    // Budget generously. Each layer contributes ~18 tensors even in the normal path (6 input refs
    // plus their m-slices, the op, its new_state view, two destination refs, the conv views), and
    // ALPHA_REPLAYSSM_CHECK adds 5 more per layer. Undersizing surfaces as
    // GGML_ASSERT(cgraph->n_nodes < cgraph->size), which aborts the process rather than refusing.
    // This is host-side bookkeeping, trivial next to a fold.
    const size_t n_layers = rec.layers.size();
    const size_t n_nodes  = 32 * n_layers + 256;
    ggml_init_params ip = {
        ggml_tensor_overhead() * (64 * n_layers + 256) + ggml_graph_overhead_custom(n_nodes, false),
        nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);

    // Reference an already-allocated tensor from the verify graph without copying it. The verify's
    // storage is in the main scheduler's compute buffer and the fold runs on sched_alt, which has
    // its own, so nothing being read here is overwritten by the fold itself.
    // Resolve a tensor's storage. The two destination views are the reason this is not just
    // `t->data`: their only consumer was the publish that record mode suppressed, so they are
    // unreferenced graph tensors and ggml-alloc never gives them a buffer (data and buffer are
    // both null, and the scheduler would trip GGML_ASSERT(buffer) in ggml_backend_tensor_copy).
    // Their view_src (the memory module's state tensor) is allocated, so walk up to it and apply
    // the accumulated view offsets.
    auto resolve = [](ggml_tensor * t, void ** data, ggml_backend_buffer_t * buf) -> bool {
        size_t off = 0;
        ggml_tensor * cur = t;
        while (cur && !cur->data) { off += cur->view_offs; cur = cur->view_src; }
        if (!cur || !cur->data) return false;
        *data = (char *) cur->data + off;
        *buf  = cur->buffer;
        return true;
    };
    bool resolve_ok = true;
    auto ref = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * r = ggml_new_tensor_4d(ctx, t->type, t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) r->nb[i] = t->nb[i];
        void * data = nullptr; ggml_backend_buffer_t buf = nullptr;
        if (!resolve(t, &data, &buf)) { resolve_ok = false; }
        r->data   = data;
        r->buffer = buf;
        // Deliberately not ggml_set_input(): the scheduler duplicates every input-flagged tensor
        // into each split via ggml_backend_tensor_copy(input, input_cpy), and for a tensor that is
        // already allocated galloc never creates that copy, so input_cpy arrives null and trips
        // GGML_ASSERT(buffer). These are pre-allocated leaves; treat them like weights, which carry
        // no input flag either.
        return r;
    };
    // Keep only the first `m` token columns. ne[2] is the token axis for every gdn input and the
    // strides are untouched, so this is the verify's own memory, just shorter.
    auto head_m = [&](ggml_tensor * t) -> ggml_tensor * {
        return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], m, t->ne[3], t->nb[1], t->nb[2], t->nb[3], 0);
    };

    static const bool check_resid = getenv("ALPHA_REPLAYSSM_CHECK") != nullptr;
    std::vector<std::pair<int, ggml_tensor *>> resid;
    std::vector<std::pair<int, ggml_tensor *>> probe;

    int n_folded = 0;
    for (size_t il = 0; il < n_layers; ++il) {
        const auto & L = rec.layers[il];
        if (!L.have_gdn || !L.have_conv || !L.state_dst) continue;

        // conv half: a copy at column offset m. No recurrence, no op.
        {
            ggml_tensor * ci  = ref(L.conv_input);
            const size_t  esz = ggml_element_size(ci);
            ggml_tensor * src = ggml_view_3d(ctx, ci,
                    L.conv_src->ne[0], L.conv_src->ne[1], L.conv_src->ne[2],
                    ci->nb[1], ci->nb[2], (size_t) m * esz);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, ref(L.conv_dst)));
        }
        // the layer's second conv-shaped history (qwen4exp PLE), same rule
        if (L.have_conv2) {
            ggml_tensor * ci  = ref(L.conv2_input);
            const size_t  esz = ggml_element_size(ci);
            ggml_tensor * src = ggml_view_3d(ctx, ci,
                    L.conv2_src->ne[0], L.conv2_src->ne[1], L.conv2_src->ne[2],
                    ci->nb[1], ci->nb[2], (size_t) m * esz);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, src), ref(L.conv2_dst)));
        }

        // gdn half.
        // ALPHA_REPLAYSSM_FULL_ECHO (default on): a full accept (m == the verify's token count) needs
        // exactly the state the verify already computed, so copy it rather than re-running the op over
        // the same m tokens.
        static const bool full_echo = []() { const char * e = getenv("ALPHA_REPLAYSSM_FULL_ECHO"); return !e || atoi(e) != 0; }();
        if (full_echo && m == rec.n_tokens && L.new_state) {
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ref(L.new_state), ref(L.state_dst)));
            n_folded++;
            continue;
        }
        {
            // ggml_gated_delta_net asserts
            //     ggml_nelements(state) == S_v*S_v*H*n_seqs ,  n_seqs = v->ne[3]
            // and a GGML_ASSERT aborts the process. With a 2-sequence verify batch (OR-verify, tree
            // verify) the recorded v carries ne[3]=2 while the recorded state may still be shaped for
            // one sequence. Check the op's precondition here and refuse, as the n_nodes guard below
            // does: the caller's relay fallback then repairs the state with a real decode.
            {
                const int64_t cS = L.v->ne[0], cH = L.v->ne[1], cB = L.v->ne[3];
                const int64_t want = cS * cS * cH * cB;
                if (ggml_nelements(L.s) != want) {
                    static bool warned = false;
                    if (!warned || dbg) {
                        warned = true;
                        fprintf(stderr,
                            "[replayssm] REFUSING fold: layer %zu state has %lld elems, op wants "
                            "S_v*S_v*H*n_seqs = %lld*%lld*%lld*%lld = %lld  (v.ne=[%lld,%lld,%lld,%lld], "
                            "s.ne=[%lld,%lld,%lld,%lld]).  n_seqs>1 => 2-seq verify; falling back to the relay.\n",
                            il, (long long) ggml_nelements(L.s), (long long) cS, (long long) cS,
                            (long long) cH, (long long) cB, (long long) want,
                            (long long) L.v->ne[0], (long long) L.v->ne[1], (long long) L.v->ne[2], (long long) L.v->ne[3],
                            (long long) L.s->ne[0], (long long) L.s->ne[1], (long long) L.s->ne[2], (long long) L.s->ne[3]);
                    }
                    ggml_free(ctx);
                    alpha_gdn_record_invalidate();
                    return false;
                }
            }
            ggml_tensor * res = L.flags != 0
                ? ggml_gated_delta_net_ext(ctx,
                    head_m(ref(L.q)), head_m(ref(L.k)), head_m(ref(L.v)),
                    head_m(ref(L.g)), head_m(ref(L.b)), ref(L.s), ref(L.dt), ref(L.a), L.eps, L.flags)
                : ggml_gated_delta_net(ctx,
                    head_m(ref(L.q)), head_m(ref(L.k)), head_m(ref(L.v)),
                    head_m(ref(L.g)), head_m(ref(L.b)), ref(L.s));
            // Same packing as llm_build_delta_net_base::build_delta_net_fused, with n_tokens = m.
            const int64_t S_v    = L.v->ne[0];
            const int64_t H_v    = L.v->ne[1];
            const int64_t n_seqs = L.v->ne[3];
            ggml_tensor * new_state = ggml_view_4d(ctx, res, S_v, S_v, H_v, n_seqs,
                    ggml_row_size(res->type, S_v),
                    ggml_row_size(res->type, S_v * S_v),
                    ggml_row_size(res->type, S_v * S_v * H_v),
                    ggml_row_size(res->type, S_v * H_v * m * n_seqs));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ref(L.state_dst)));

            // ALPHA_REPLAYSSM_CHECK=1: on-device residual between the re-invoked state and the
            // verify's own new_state. At m == n these must be bit-identical if the re-invocation
            // reproduces the verify. Per-layer, so a nonzero answer also says where.
            // Input-finiteness probe (layer 0). If any recorded input is non-finite at fold time
            // the records did not survive the verify (a retention bug); if they are all finite the
            // records are good and the defect is in how the op is re-invoked.
            if (check_resid && il == 0) {
                ggml_tensor * ins[6]  = { L.q, L.k, L.v, L.g, L.b, L.s };
                for (int z = 0; z < 6; ++z) {
                    ggml_tensor * t  = ggml_cont(ctx, ref(ins[z]));
                    ggml_tensor * ss = ggml_sum(ctx, ggml_sqr(ctx, t));
                    ggml_set_output(ss);
                    ggml_build_forward_expand(gf, ss);
                    probe.push_back({ z, ss });
                }
            }
            if (check_resid && L.new_state) {
                ggml_tensor * d  = ggml_sub(ctx, ggml_cont(ctx, new_state),
                                                 ggml_cont(ctx, ref(L.new_state)));
                ggml_tensor * ss = ggml_sum(ctx, ggml_sqr(ctx, d));
                ggml_set_output(ss);
                ggml_build_forward_expand(gf, ss);
                resid.push_back({ (int) il, ss });
            }
        }
        n_folded++;
    }

    // A tensor whose storage could not be resolved would reach the scheduler with a null buffer and
    // abort the process. Refuse instead; the caller's relay fallback then repairs the state.
    if (!resolve_ok) {
        if (dbg) fprintf(stderr, "[replayssm] fold ABORTED: unresolved tensor storage\n");
        ggml_free(ctx);
        alpha_gdn_record_invalidate();
        return false;
    }

    bool ok = n_folded > 0;
    if (ok) {
        // Run on the alt scheduler for the same reason the MTP draft does: resetting the main
        // sched destroys the decode's graph-reuse cache and forces a full rebuild on the next token.
        const bool use_alt = (bool) sched_alt;
        if (use_alt) set_draft_active(true);
        ggml_backend_sched_reset(sched.get());
        ok = ggml_backend_sched_graph_compute(sched.get(), gf) == GGML_STATUS_SUCCESS;
        if (use_alt) set_draft_active(false);
    }

    if (ok && check_resid && !probe.empty()) {
        static const char * nm[6] = { "q", "k", "v", "g", "b", "s" };
        char line[256]; int off = 0;
        for (auto & pr : probe) {
            float v = 0.0f;
            ggml_backend_tensor_get(pr.second, &v, 0, sizeof(float));
            off += snprintf(line + off, sizeof(line) - off, " %s=%.4g", nm[pr.first], v);
        }
        fprintf(stderr, "[replayssm-probe] layer0 input L2sq:%s\n", line);
    }

    if (ok && check_resid && !resid.empty()) {
        int    n_bad = 0, worst_il = -1;
        double worst = 0.0;
        for (auto & pr : resid) {
            float v = 0.0f;
            ggml_backend_tensor_get(pr.second, &v, 0, sizeof(float));
            if (v > 0.0f) { n_bad++; if (v > worst) { worst = v; worst_il = pr.first; } }
        }
        fprintf(stderr, "[replayssm-check] m=%d/%lld: %d/%zu layers DIFFER, worst L2sq=%.6g at layer %d\n",
                (int) m, (long long) rec.n_tokens, n_bad, resid.size(), worst, worst_il);
    }

    if (dbg) {
        fprintf(stderr, "[replayssm] fold m=%d/%lld layers=%d -> %s\n",
                (int) m, (long long) rec.n_tokens, n_folded, ok ? "ok" : "FAILED");
    }

    ggml_free(ctx);
    // One recording serves exactly one fold. Anything later must re-record or fall back.
    alpha_gdn_record_invalidate();
    return ok;
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

// UBBoost: runtime micro-batch resize. The compute-buffer worst case is derived from
// n_ubatch (sched_reserve at min(n_ctx, n_ubatch)), so changing it requires an eager
// re-reserve here with rollback: sched_reserve() throws on allocation failure and is
// otherwise called from inside decode(), where an exception would escape the C API.
// Growing reuses the existing scheduler (ggml-alloc grows in place); the old size is
// restored if the grow does not fit. Shrinking keeps the larger buffer (ggml-alloc
// does not release), which is fine for the grow-only prefill-boost use case.
int32_t llama_context::set_n_ubatch(uint32_t value) {
    value = std::max(1u, std::min(value, cparams.n_batch));

    if (cparams.n_ubatch == value) {
        return (int32_t) cparams.n_ubatch;
    }

    const uint32_t old = cparams.n_ubatch;

    cparams.n_ubatch = value;
    sched_need_reserve = true;

    try {
        sched_reserve();
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: re-reserve for n_ubatch=%u failed (%s) — keeping n_ubatch=%u\n",
                __func__, value, e.what(), old);
        cparams.n_ubatch = old;
        sched_need_reserve = true;
        try {
            sched_reserve();  // old size fit before: restore its reservation
        } catch (const std::exception & e2) {
            LLAMA_LOG_ERROR("%s: rollback re-reserve failed (%s)\n", __func__, e2.what());
        }
    }

    LLAMA_LOG_INFO("%s: n_ubatch = %u\n", __func__, cparams.n_ubatch);

    return (int32_t) cparams.n_ubatch;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

// ALPHA_UBATCH_TIMERS=N: host-side phase timers for every ubatch with >= N tokens (default off).
// Synchronizes after compute when on, so "compute" is GPU time and the other phases are host time.
static int alpha_ubatch_timers_min() {
    static const int v = []() { const char * e = getenv("ALPHA_UBATCH_TIMERS"); return e ? atoi(e) : 0; }();
    return v;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const bool ubt = alpha_ubatch_timers_min() > 0 && (int) ubatch.n_tokens >= alpha_ubatch_timers_min();
    const int64_t ubt_t0 = ubt ? ggml_time_us() : 0;
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    const int64_t ubt_t1 = ubt ? ggml_time_us() : 0;
    bool ubt_reused = false;

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    // ALPHA_MTP_FILL_INGRAPH: a graph that carries head-KV fill nodes is never reused (its fill inputs are per chunk)
    // ReplaySSM: reuse only a graph built in the same record mode as this decode
    if (!graph_reuse_disable && !mtp_ig.built && res->alpha_built_armed == alpha_gdn_record_armed() && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);
        mtp_ig.active = false;

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
        ubt_reused = true;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        mtp_ig.active = mtp_ig.built = false;
        res->alpha_built_armed = alpha_gdn_record_armed();
        if (gtype == LLM_GRAPH_TYPE_DECODER) (void) mtp_ingraph_append(ubatch, res, gf);

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    const int64_t ubt_t2 = ubt ? ggml_time_us() : 0;
    // ALPHA_SCHED_SMALL_INPUT_ASYNC: prompt-sized ubatches copy their small per-layer inputs asynchronously
    ggml_backend_sched_alpha_async_inputs(ubatch.n_tokens >= 64);
    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);
        mtp_ingraph_set_inputs();

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }
    // ALPHA_INPUT_DUMP=w: one-shot: list the graph's input leaves (name, type, shape, bytes, buffer) for a
    // ubatch of width w, i.e. what each decode uploads
    if (const char * e = getenv("ALPHA_INPUT_DUMP")) {
        static int done = 0;
        if (!done && (int) ubatch.n_tokens == atoi(e)) {
            done = 1;
            size_t tot = 0;
            std::vector<const ggml_tensor *> seen;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++)
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * t = ggml_graph_node(gf, i)->src[j];
                if (!t || !(t->flags & GGML_TENSOR_FLAG_INPUT)) continue;
                if (std::find(seen.begin(), seen.end(), t) != seen.end()) continue;
                seen.push_back(t);
                tot += ggml_nbytes(t);
                fprintf(stderr, "[inputs] %-28s %-5s [%lld,%lld,%lld,%lld] %8zu B %s\n", t->name, ggml_type_name(t->type),
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], ggml_nbytes(t),
                        t->buffer ? ggml_backend_buffer_name(t->buffer) : "-");
            }
            fprintf(stderr, "[inputs] total %.1f KB for n_tokens=%d\n", tot / 1024.0, (int) ubatch.n_tokens);
        }
    }

    // ALPHA_GRAPH_DUMP: one-shot per distinct ubatch width, list every MUL_MAT node by name, so the lists
    // for two widths can be diffed to see which matmuls a wider batch adds.
    if (getenv("ALPHA_GRAPH_DUMP")) {
        // Key on (n_tokens, n_seqs) and skip reserve passes: graph_reserve builds a worst-case graph with
        // n_seqs = n_parallel (e.g. n_tokens=1, n_seqs=4), which is not what a real 1-token decode executes.
        // Real serve decodes are n_seqs == 1.
        static std::set<std::pair<int,int>> dumped;
        const int w  = (int) ubatch.n_tokens;
        const int ns = (int) ubatch.n_seqs;
        if (ns == 1 && dumped.insert({w, ns}).second) {
            ggml_cgraph * dg = res->get_gf();
            int n_mm = 0, n_mmid = 0;
            fprintf(stderr, "[graphdump] === width %d (n_seqs=%d) : %d nodes ===\n", w, ns, ggml_graph_n_nodes(dg));
            for (int i = 0; i < ggml_graph_n_nodes(dg); ++i) {
                ggml_tensor * nd = ggml_graph_node(dg, i);
                if (nd->op == GGML_OP_MUL_MAT_ID) { n_mmid++; continue; }
                if (nd->op != GGML_OP_MUL_MAT) continue;
                n_mm++;
                fprintf(stderr, "[graphdump] w%d MUL_MAT %-28s src0=%-26s [%lld,%lld] x [%lld,%lld]\n",
                        w, nd->name,
                        nd->src[0] ? ggml_get_name(nd->src[0]) : "?",
                        (long long) (nd->src[0] ? nd->src[0]->ne[0] : 0),
                        (long long) (nd->src[0] ? nd->src[0]->ne[1] : 0),
                        (long long) (nd->src[1] ? nd->src[1]->ne[0] : 0),
                        (long long) (nd->src[1] ? nd->src[1]->ne[1] : 0));
            }
            fprintf(stderr, "[graphdump] width %d TOTAL MUL_MAT=%d MUL_MAT_ID=%d\n", w, n_mm, n_mmid);
        }
    }

    const int64_t ubt_t3 = ubt ? ggml_time_us() : 0;
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (ubt) {
        const int64_t ubt_t4 = ggml_time_us();
        synchronize();
        const int64_t ubt_t5 = ggml_time_us();
        fprintf(stderr, "[ubt] n_tokens=%u pos0=%d apply=%.2f %s=%.2f inputs=%.2f launch=%.2f gpu_wait=%.2f ms\n",
                ubatch.n_tokens, ubatch.pos ? ubatch.pos[0] : -1,
                (ubt_t1 - ubt_t0) / 1000.0, ubt_reused ? "reuse" : "build+alloc", (ubt_t2 - ubt_t1) / 1000.0,
                (ubt_t3 - ubt_t2) / 1000.0, (ubt_t4 - ubt_t3) / 1000.0, (ubt_t5 - ubt_t4) / 1000.0);
    }

    // ReplaySSM: close the recording for this graph. Unconditional and before the error return: arming
    // must never leak into a later decode, because an armed build suppresses the state publish and a stale
    // arm would suppress it with no fold to follow.
    if (alpha_gdn_record_armed()) {
        // Count only real recurrent layers. The trailing NextN/MTP block reports is_recurrent()
        // (it inherits the arch's period-4 interleave) but never builds a delta-net, so counting
        // it would make `expected` one too high and the fold would refuse every good recording.
        const uint32_t n_real = model.hparams.n_layer - model.hparams.nextn_predict_layers;
        int n_rec = 0;
        for (uint32_t il = 0; il < n_real; ++il) {
            if (model.hparams.is_recurrent(il)) n_rec++;
        }
        alpha_gdn_record_end(n_rec);
    }

    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        alpha_gdn_record_invalidate();
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // extract logits
    // A greedy MTP verify never reads logits: every row's token comes from the in-graph argmax.
    // Skip the copy when the caller has declared it does not need them and the argmax is present
    // to replace them. When skipped, logits are marked invalid rather than left stale, so
    // get_logits_ith() returns null and every caller takes its fallback.
    const bool skip_logits = skip_logits_d2h_ && res->t_mtp_argmax != nullptr;
    logits_valid_ = !skip_logits;
    if (logits.data && t_logits && !skip_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT
    // ALPHA_DECODE_TIMERS=1: host phase split of decode() for small batches (prepare / ubatches / fill)
    static const bool dct = getenv("ALPHA_DECODE_TIMERS") != nullptr;
    const int64_t dct_t0 = dct ? ggml_time_us() : 0;
    int64_t dct_t1 = 0, dct_t2 = 0;

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;
        const bool ubt_on = alpha_ubatch_timers_min() > 0 && (int) ubatch.n_tokens >= alpha_ubatch_timers_min();
        static int64_t ubt_prev_end = 0;
        const int64_t ubt_iter0 = ubt_on ? ggml_time_us() : 0;
        if (ubt_on && ubt_prev_end > 0 && ubt_iter0 - ubt_prev_end < 5000000) {
            fprintf(stderr, "[ubt] host gap since previous ubatch ended: %.2f ms\n", (ubt_iter0 - ubt_prev_end) / 1000.0);
        }
        if (dct && dct_t1 == 0) dct_t1 = ggml_time_us();
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);
        if (ubt_on) {
            const int64_t ubt_after = ggml_time_us();
            ubt_prev_end = ubt_after;
            fprintf(stderr, "[ubt] process_ubatch total %.2f ms\n", (ubt_after - ubt_iter0) / 1000.0);
        }

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits = res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        // ALPHA_MTP: a greedy verify takes every row's token from the in-graph argmax, so skip the full-vocab
        // D2H and mark the logits absent.
        const bool skip_raw = !draft_active_ && skip_logits_d2h_ && res->t_mtp_argmax != nullptr;
        if (!draft_active_) logits_valid_ = !skip_raw;
        if (logits.data && t_logits && n_outputs > 0 && !skip_raw && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            // the draft (second sched) writes its logits to a separate host buffer so a concurrent verify
            // decode's logits.data is never clobbered. get_logits_ith() reads it back while draft_active_
            // (so the draft's own sampler gets the right rows).
            float * logits_out;
            if (draft_active_) {
                draft_logits_.resize((size_t)(n_outputs_prev + n_outputs) * n_vocab);
                logits_out = draft_logits_.data() + (size_t) n_outputs_prev*n_vocab;
            } else {
                logits_out = logits.data + n_outputs_prev*n_vocab;
            }

            if (n_outputs) {
                if (!draft_active_) {
                    GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                    GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                }
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // ALPHA_MTP: extract the exported pre-norm hidden H. Both builders export every row of the
        // ubatch under `mtp_defer_narrow` (the out_ids narrowing is deferred to the LM head), and the
        // capture below is shared by the dense and MoE paths (it keys only on t_mtp_hidden). If a
        // multi-token decode yields no hidden, look at backend_h below (it goes null when the tensor
        // lands on a backend the sched cannot name, e.g. under an -ot override), not at the builders.
        // Then pull the in-graph argmax: n_outputs int32, tens of bytes, replacing a full-logits D2H +
        // host scan per verify row on the accept path.
        if (res->t_dbg) {
            ggml_backend_t backend_d = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_dbg);
            if (backend_d) {
                float v = 0.0f;
                ggml_backend_sched_synchronize(sched.get());   // the compute stream may still be running
                ggml_backend_tensor_get(res->t_dbg, &v, 0, sizeof(float));
                static int64_t n_dbg = 0, n_bad = 0;
                n_dbg++;
                n_bad += v != 0.0f;
                if (v != 0.0f || n_dbg % 64 == 1) {
                    LLAMA_LOG_INFO("[dbg] %s = %g (call %lld, %lld nonzero so far, n_tokens %u)\n",
                                   ggml_get_name(res->t_dbg), (double) v, (long long) n_dbg, (long long) n_bad, ubatch.n_tokens);
                }
            }
        }
        if (res->t_mtp_argmax) {
            ggml_backend_t backend_a = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_mtp_argmax);
            const int64_t n_am = backend_a ? (int64_t) (ggml_nbytes(res->t_mtp_argmax) / sizeof(int32_t)) : 0;
            if (n_am > 0) {
                mtp_argmax_host.resize((size_t) n_am);
                ggml_backend_tensor_get_async(backend_a, res->t_mtp_argmax, mtp_argmax_host.data(),
                                              0, (size_t) n_am * sizeof(int32_t));
            } else {
                mtp_argmax_host.clear();
            }
        } else {
            mtp_argmax_host.clear();
        }

        if (res->t_mtp_hidden) {
            ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_mtp_hidden);
            // Read only if the exported tensor really holds a full n_embd row. t_mtp_hidden is
            // `cur` after inp_out_ids row selection, so a decode pass with n_tokens==1 but zero
            // output rows yields [n_embd, 0] (0 bytes), and an unconditional n_embd*4 read
            // trips "tensor read out of bounds" in ggml_backend_tensor_get_async.
            const int64_t n_embd = mtp_hw();   // hc*n_embd on qwen4exp: the trunk's residual streams
            const size_t  need   = (size_t) n_embd * sizeof(float);
            const int64_t n_rows = backend_h ? (int64_t) (ggml_nbytes(res->t_mtp_hidden) / need) : 0;
            // The export carries every row of the ubatch (the out_ids narrowing is deferred to the
            // LM head), so row i is the final hidden at ubatch.pos[i]. Keep the whole block for
            // the batched KV fill; the last row is also the H the next draft starts from.
            // ALPHA_MTP_DBG: distinguishes the causes of "the draft returned NULL after a
            // multi-token decode":
            //   backend_h == nullptr      -> rows=0            (sched cannot name the tensor's
            //                                                   backend; suspect an -ot override)
            //   export was narrowed       -> rows=1, ntok=26   (build lacks the deferred out_ids
            //                                                   narrowing; capture skipped by the
            //                                                   n_tokens==n_rows guard below)
            //   healthy                   -> rows == ntok
            if (getenv("ALPHA_MTP_DBG")) {
                static int n_cap = 0;
                if (n_cap < 12) {
                    n_cap++;
                    fprintf(stderr, "[mtp-cap] rows=%lld ntok=%lld backend=%s -> %s\n",
                            (long long) n_rows, (long long) ubatch.n_tokens,
                            backend_h ? ggml_backend_name(backend_h) : "NULL",
                            (n_rows >= 1 && ubatch.token && (int64_t) ubatch.n_tokens == n_rows)
                                ? "CAPTURED" : "SKIPPED");
                }
            }
            if (n_rows >= 1 && ubatch.token && (int64_t) ubatch.n_tokens == n_rows) {
                if (mtp_ig.active) {   // ALPHA_MTP_FILL_INGRAPH: the graph filled the head KV, only the last row comes back
                    mtp_ig.last_row.resize((size_t) n_embd);
                    ggml_backend_tensor_get_async(backend_h, res->t_mtp_hidden, mtp_ig.last_row.data(), need * (n_rows - 1), need);
                    mtp_fill_h.clear();
                } else {
                mtp_fill_h.resize((size_t) n_embd * n_rows);
                ggml_backend_tensor_get_async(backend_h, res->t_mtp_hidden, mtp_fill_h.data(), 0, need * n_rows);
                }
                mtp_fill_tok.assign(ubatch.token, ubatch.token + n_rows);
                mtp_fill_pos.resize(n_rows);
                for (int64_t i = 0; i < n_rows; ++i) mtp_fill_pos[i] = ubatch.pos[i];
                // Which sequence each captured row belongs to. A ubatch may mix sequences, and the
                // fill must be driven once per sequence or slot0 jumps between them.
                mtp_fill_seq.resize(n_rows);
                for (int64_t i = 0; i < n_rows; ++i) {
                    mtp_fill_seq[i] = (ubatch.seq_id && ubatch.seq_id[i]) ? ubatch.seq_id[i][0] : 0;
                }
                // The carried hidden, the rewind truncation, mtp_pos_base and the chain reset are all
                // per-sequence, so they are derived in mtp_kv_fill_all(), which partitions the rows by
                // sequence, rather than here from the whole (possibly mixed) ubatch.
            } else {
                mtp_hidden_host.clear();   // no usable H this pass -> mtp_draft declines cleanly
                mtp_fill_pos.clear(); mtp_fill_tok.clear();
                mtp_pos_base = -1;
            }
        } else if (!mtp_hidden_host.empty()) {
            mtp_hidden_host.clear();
            mtp_fill_pos.clear(); mtp_fill_tok.clear();
            mtp_pos_base = -1;
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // ALPHA_MTP: lay this decode's positions into the draft head's own KV. Runs on the rows just
    // captured above, so prefill fills the whole prompt in one batched pass and each decode extends
    // the head's history by its own real tokens. No-op when the head is absent or ALPHA_MTP is off.
    if (dct) dct_t2 = ggml_time_us();
    if (!mtp_fill_pos.empty()) mtp_kv_fill_all();   // per sequence; see mtp_kv_fill_all

    if (dct && (batch_inp.n_tokens <= 8 || getenv("ALPHA_DECODE_TIMERS_ALL"))) {
        const int64_t t3 = ggml_time_us();
        fprintf(stderr, "[dct] n=%d prepare=%.2f ubatches+outputs=%.2f mtp_fill=%.2f total=%.2f ms\n", (int) batch_inp.n_tokens,
                (dct_t1 - dct_t0) / 1000.0, (dct_t2 - dct_t1) / 1000.0, (t3 - dct_t2) / 1000.0, (t3 - dct_t0) / 1000.0);
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }


    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    logits.size = has_logits ? n_vocab*n_outputs_max : 0;
    embd.size   = has_embd ? n_embd_out*n_outputs_max : 0;

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + backend_float_count) * sizeof(float) +
        (                          backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    // MoE models: during warmup, the build_moe_ffn aggregation loop creates n_expert views +
    // (n_expert-1) adds per MoE layer (256 experts x 40 layers = 20K+ extra tensors). Scale
    // max_nodes to accommodate.
    if (model.hparams.n_expert > 0) {
        const uint32_t n_expert = model.hparams.n_expert;
        const uint32_t n_layer  = model.hparams.n_layer;
        const uint32_t moe_nodes = (uint32_t)((2u * n_expert + 100u) * n_layer * 1.3f) + 4096u;
        return std::max<uint32_t>(moe_nodes, std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors()));
    }
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_QWEN35) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    // ALPHA_RESERVE_TIMERS: host phases of a prompt-sized reserve (graph build vs. split + allocation plan)
    static const bool rv_t = getenv("ALPHA_RESERVE_TIMERS") != nullptr;
    const int64_t rv0 = ggml_time_us();

    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);
    if (gf) mtp_ingraph_reserve(res, gf, (int64_t) n_tokens);   // ALPHA_MTP_FILL_INGRAPH: same nodes as the real prompt graph
    const int64_t rv1 = ggml_time_us();
    struct rv_guard { bool on; uint32_t n; int64_t a, b; ~rv_guard() {
        if (on && n >= 1024) fprintf(stderr, "[reserve-t] n_tokens=%u build %.1f ms, sched reserve %.1f ms\n", n, (b - a) / 1e3, (ggml_time_us() - b) / 1e3); } }
        rv_g { rv_t, n_tokens, rv0, rv1 };

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    // ALPHA_DUMP_BIG_NODES=N: once per reserve with >= 1024 tokens, list the N largest graph nodes (what the
    // per-token compute buffer is made of)
    if (const char * e = getenv("ALPHA_DUMP_BIG_NODES"); e && n_tokens >= 1024) {
        std::vector<std::pair<size_t, ggml_tensor *>> v;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (t->view_src) continue;
            v.push_back({ ggml_nbytes(t), t });
        }
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        std::map<std::string, std::pair<int, size_t>> by_op;
        for (auto & x : v) { auto & a = by_op[std::string(ggml_op_desc(x.second))]; a.first++; a.second = std::max(a.second, x.first); }
        LLAMA_LOG_INFO("[bignodes] n_tokens=%u nodes=%d\n", n_tokens, ggml_graph_n_nodes(gf));
        // approximate live set: a non-view node is alive from its index to its last consumer (views count against
        // their source); report the peak and what is alive there
        {
            const int nn = ggml_graph_n_nodes(gf);
            std::map<const ggml_tensor *, int> idx, last;
            auto base = [](const ggml_tensor * t) { while (t->view_src) t = t->view_src; return t; };
            for (int i = 0; i < nn; i++) idx[ggml_graph_node(gf, i)] = i;
            for (int i = 0; i < nn; i++) {
                const ggml_tensor * t = ggml_graph_node(gf, i);
                last[base(t)] = std::max(last[base(t)], i);
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (!t->src[s]) continue;
                    const ggml_tensor * b = base(t->src[s]);
                    last[b] = std::max(last[b], i);
                }
            }
            size_t live = 0, peak = 0; int peak_i = 0;
            std::vector<std::vector<const ggml_tensor *>> frees(nn);
            for (auto & [t, l] : last) if (idx.count(t)) frees[l].push_back(t);
            for (int i = 0; i < nn; i++) {
                const ggml_tensor * t = ggml_graph_node(gf, i);
                if (!t->view_src && !(t->flags & GGML_TENSOR_FLAG_INPUT)) live += ggml_nbytes(t);
                if (live > peak) { peak = live; peak_i = i; }
                for (auto * f : frees[i]) if (!f->view_src) live -= ggml_nbytes(f);
            }
            LLAMA_LOG_INFO("[bignodes] approx peak live %.1f MB at node %d (%s)\n", peak / 1048576.0, peak_i,
                           ggml_graph_node(gf, peak_i)->name);
            std::vector<std::pair<size_t, const ggml_tensor *>> alive;
            for (int i = 0; i <= peak_i; i++) {
                const ggml_tensor * t = ggml_graph_node(gf, i);
                if (t->view_src || (t->flags & GGML_TENSOR_FLAG_INPUT)) continue;
                if (last[t] >= peak_i) alive.push_back({ ggml_nbytes(t), t });
            }
            std::sort(alive.begin(), alive.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int i = 0; i < (int) alive.size() && i < 24; i++) {
                LLAMA_LOG_INFO("[bignodes]   alive %8.1f MB %-12s %-32s [%lld,%lld,%lld,%lld]\n", alive[i].first / 1048576.0,
                               ggml_op_desc(alive[i].second), alive[i].second->name, (long long) alive[i].second->ne[0],
                               (long long) alive[i].second->ne[1], (long long) alive[i].second->ne[2], (long long) alive[i].second->ne[3]);
            }
        }
        for (int i = 0; i < (int) v.size() && i < atoi(e); i++) {
            LLAMA_LOG_INFO("[bignodes] %8.1f MB %-14s %-32s [%lld,%lld,%lld,%lld]\n", v[i].first / 1048576.0, ggml_op_desc(v[i].second),
                           v[i].second->name, (long long) v[i].second->ne[0], (long long) v[i].second->ne[1],
                           (long long) v[i].second->ne[2], (long long) v[i].second->ne[3]);
        }
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.n_layer_draft               =*/ 0,
        /*.draft_skip_mask             =*/ 0,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (ggml_is_quantized(params.type_k) || ggml_is_quantized(params.type_v)) {
            LLAMA_LOG_ERROR("%s: simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

int32_t llama_set_n_ubatch(llama_context * ctx, uint32_t n_ubatch) {
    return ctx->set_n_ubatch(n_ubatch);
}

void llama_set_n_ubatch_lazy(llama_context * ctx, uint32_t n_ubatch) {
    ctx->set_n_ubatch_lazy(n_ubatch);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}


void llama_gdn_record_arm(llama_context * ctx, int32_t n_tokens) {
    (void) ctx;   // the recorder is per-process; the graph builder has no ctx handle
    alpha_gdn_record_arm(n_tokens);
}

bool llama_gdn_fold(llama_context * ctx, int32_t m) {
    if (!ctx) return false;
    return ctx->gdn_fold(m);
}

void llama_set_skip_logits(llama_context * ctx, bool value) {
    if (!ctx) return;
    ctx->set_skip_logits(value);
}

llama_token llama_mtp_argmax_ith(llama_context * ctx, int32_t i) {
    if (!ctx) return LLAMA_TOKEN_NULL;
    return ctx->mtp_argmax_ith(i);
}

void llama_mtp_select_row(llama_context * ctx, int32_t row) {
    if (!ctx) return;
    ctx->mtp_select_row(row);
}

void llama_mtp_set_seq(llama_context * ctx, int32_t seq_id) {
    if (ctx) ctx->mtp_switch_seq(seq_id);
}

llama_token llama_mtp_draft(llama_context * ctx, llama_token e_tok) {
    // The H copy was issued during decode, which already synchronized to read logits for
    // sampling before this is reached (result.sampled is populated). A second full-device
    // synchronize here just drains the pipeline. ALPHA_MTP_SYNC=1 restores it if ever needed.
    static const bool mtp_hard_sync = getenv("ALPHA_MTP_SYNC") != nullptr;
    if (mtp_hard_sync) ctx->synchronize();
    return ctx->mtp_draft(e_tok);
}

float llama_mtp_draft_conf(llama_context * ctx) {
    return ctx ? ctx->mtp_get_conf() : 1.0f;
}

float llama_mtp_draft_margin(llama_context * ctx) {
    return ctx ? ctx->mtp_get_margin() : 0.0f;
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto * memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_set_eval_callback(struct llama_context * ctx,
        ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data) {
    if (!ctx) return;
    ctx->set_eval_callback(cb_eval, cb_eval_user_data);
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_gpu_snapshot(llama_memory_t mem) {
    if (mem) mem->gpu_snapshot();
}

double llama_memory_state_checksum(llama_memory_t mem) { // ALPHA_GDN_DUMP
    return mem ? mem->state_checksum() : 0.0;
}

void llama_memory_gpu_restore(llama_memory_t mem) {
    if (mem) mem->gpu_restore();
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

// ALPHA_DRE_RETURN_OVERLAP: drop the scheduler and its compute buffers now; the next reserve (llama_context_reset_sched)
// builds a fresh one. Lets a caller hand the freed memory back to someone else before the reserve's host work.
void llama_context_release_sched(llama_context * ctx) {
    if (!ctx) return;
    ctx->release_sched();
}

size_t llama_context_reserved_bytes(const llama_context * ctx, uint32_t n_ubatch) {
    if (!ctx) return 0;
    auto it = ctx->reserved_dev_bytes.find(n_ubatch);
    return it == ctx->reserved_dev_bytes.end() ? 0 : it->second;
}

void llama_context_reset_sched(llama_context * ctx) {
    if (!ctx) return;
    // Force the scheduler to destroy old compute buffers and create fresh ones. Without this,
    // ggml_backend_sched buffers grow monotonically and never shrink, and small GPUs eventually OOM.
    ctx->force_reserve();
}

void llama_context_invalidate_graph(llama_context * ctx) {
    if (!ctx) return;
    ctx->invalidate_graph();
}

void llama_context_set_draft_layers(llama_context * ctx, int32_t n_layers) {
    if (!ctx) return;
    ctx->set_draft_layers(n_layers);
}

void llama_context_clear_draft_layers(llama_context * ctx) {
    if (!ctx) return;
    ctx->clear_draft_layers();
}

void llama_context_set_draft_skip_mask(llama_context * ctx, uint64_t mask) {
    if (!ctx) return;
    ctx->set_draft_skip_mask(mask);
}

void llama_memory_breakdown_print(const struct llama_context * ctx) {
    const auto & devices = ctx->get_model().devices;

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i].dev == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t          dev = devices[i].dev;
        llama_memory_breakdown_data mb  = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const size_t unaccounted = total - self - free;

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / MiB)});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LLAMA_LOG_INFO(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
