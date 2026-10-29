
#include "server-context.h"
#include "server-common.h"
#include "server-tools.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-help.h"
#include "server-digest.h"
#include "server-tiered-index.h"
#include "server-kv-pool.h"
#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim: return freed build-time index maps to the OS
#endif
#include "server-radix-cache.h"
#ifdef ALPHA_HAS_REDIS
#include "server-redis.h"
#endif
#include "server-dense.h"   // dense-L0 semantic backend (self-guarded on ALPHA_DENSE_L0)

#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <set>
#include <unordered_set>   // union-mode dedup (/api/corpus/query mode=union)
#include <cctype>          // span-return selector (isalnum/tolower/isspace)
#include <regex>
#include <chrono>   // auto-ingest phase profiling (ALPHA_INGEST_PROFILE)

// Alpha-MoLE: MoE optimization for server mode
#include "../../src/alpha_mole_moe_optimizer.h"
#include "../../src/alpha_mole_speculative.h"
#include "../../src/alpha_moe_portfolio.h"
#include "../../src/alpha_hw_planner.h"
#include "../../src/alpha_dense_pipeline.h"
// KV-chunk: per-layer cache access for centroid extraction
#include "../../src/llama-memory-hybrid.h"
#include "../../src/llama-kv-cache.h"
#ifdef ALPHA_DELTANET
#include "../../src/alpha_deltanet_state.h"
#endif
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <fstream>
#include <utility>
#include <random>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static server_prompt_checkpoint server_get_checkpoint(llama_context * ctx, int id, int64_t n_tokens, llama_pos pos_min = -1, llama_pos pos_max = -1, llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
    if (pos_min == -1) {
        pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), id);
    }
    if (pos_max == -1) {
        pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), id);
    }

    const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, id, flags);

    auto cur = server_prompt_checkpoint {
        /*.pos_min  = */ pos_min,
        /*.pos_max  = */ pos_max,
        /*.n_tokens = */ n_tokens,
        /*.data     = */ std::vector<uint8_t>(checkpoint_size),
    };

    const size_t n = llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, id, flags);
    if (n != checkpoint_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
    }

    return cur;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

// Slot Snapshot for Pipeline Parallelism
// Immutable copy of slot state - safe to read from any thread.
// Output processing reads from snapshot while GPU runs next iteration.
struct slot_snapshot {
    int slot_id = -1;
    llama_token sampled = LLAMA_TOKEN_NULL;
    std::string generated_text;
    size_t n_sent_text = 0;
    int n_decoded = 0;
    bool has_next_token = true;
    stop_type stop = STOP_TYPE_NONE;
    bool valid = false;

    // Timing
    int64_t t_start_generation = 0;
    int64_t t_start_process_prompt = 0;
    double t_prompt_processing = 0;
    double t_token_generation = 0;
};

struct server_slot {
    int id;

    llama_context * ctx = nullptr;

    common_context_seq_rm_type ctx_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    llama_tokens spec_draft;
    std::vector<int32_t> spec_i_batch;
    server_prompt_checkpoint spec_ckpt;
    common_speculative_ptr spec;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    // MoE quality gate state (per-slot for multi-user serve)
    std::vector<llama_token> lut_history;
    llama_token moe_prev_tok = LLAMA_TOKEN_NULL;
    int moe_gen_tokens = 0;  // generation counter for readback_interval
    bool moe_long_gen = false;  // true for non-factual profiles (need two-pass)
    bool moe_lut_prefill = false; // true if this request used LUT for prefill (needs QC)
    bool moe_early_eos_repaired = false; // true if early EOS triggered cpu-moe repair
    int moe_confidence_fixes = 0;       // count of confidence-gated expert loads this request
    int moe_cpu_remaining = 0;          // tokens remaining in cpu-moe mode (for cache warming)
    bool moe_adaptive_warmup = false;   // true = generating in cpu-moe, checking cache stability
    int moe_warmup_tokens_remaining = 0; // adaptive warmup tokens left before forced LUT switch
    bool moe_sparse_prefill = false;    // true: drop ~60% of prompt tokens for faster prefill
    // self-spec layer-skip uses single-context with n_layer_draft switching
    int moe_pass = 0;           // 0 = Pass 1 (draft), 1 = Pass 2 (refine)
    std::string moe_draft_text; // Pass 1 output, used as context for Pass 2
    bool force_cpu_moe = false;  // true = disable LUT cache, use all experts via mmap

    // Expert-usage logging (ALPHA_LOG_EXPERT_USAGE) - passive read-only tap on the router's per-token
    // top-k selection, accumulated per completion. eu_counts[l*n_expert + eid] = # decode activations of
    // expert eid in MoE layer l. Does not touch sampling/routing -> byte-identical tokens flag-on vs off.
    std::vector<uint32_t> eu_counts;
    int eu_n_layer = 0, eu_n_expert = 0, eu_n_used = 0;
    long long eu_decode_tokens = 0;

    // Branch-and-recover decode (ALPHA_BRANCH_RECOVER) - per-generation state. Reset when n_decoded==0.
    int             br_recover_left = 0;   // forced-greedy tokens remaining after a branch
    int             br_branch_count = 0;   // branches taken this generation (capped by br_max_branch)
    std::mt19937_64 br_rng;                // seeded per generation from br_seed -> reproducible draws

    std::vector<completion_token_output> generated_token_probs;

    // Batch streaming: accumulate tokens, flush every N to reduce queue overhead
    std::vector<completion_token_output> pending_stream_tokens;

    // Overlapped grammar: precomputed mask from before GPU forward pass
    common_grammar_precompute_result precomputed_grammar;

    // Pipeline snapshot: immutable copy for async output processing
    slot_snapshot snapshot;

    slot_snapshot take_snapshot() const {
        slot_snapshot s;
        s.slot_id = id;
        s.sampled = sampled;
        s.generated_text = generated_text;
        s.n_sent_text = n_sent_text;
        s.n_decoded = n_decoded;
        s.has_next_token = has_next_token;
        s.stop = stop;
        s.t_start_generation = t_start_generation;
        s.t_start_process_prompt = t_start_process_prompt;
        s.t_prompt_processing = t_prompt_processing;
        s.t_token_generation = t_token_generation;
        s.valid = true;
        return s;
    }

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size = llama_state_seq_get_size_ext(ctx, id, 0);

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx, cur->data.data(), cur_size, id, 0);
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        llama_memory_seq_rm(llama_get_memory(ctx), id, -1, -1);
        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    // Adaptive verify-skip: when draft acceptance is consistently high,
    // skip the expensive main-model verify and trust draft tokens directly.
    int32_t adaptive_verify_window = 0;     // tokens in current measurement window
    int32_t adaptive_verify_accepted = 0;   // accepted in current window
    int32_t adaptive_verify_batches = 0;    // batch counter for periodic verify
    bool    adaptive_skip_verify = false;   // true = currently skipping verify
    static constexpr int    ADAPTIVE_WINDOW_SIZE  = 24;  // tokens before first skip decision (~3 batches)
    static constexpr float  ADAPTIVE_SKIP_THRESH  = 0.95f; // 95% acceptance to enable skip
    static constexpr int    ADAPTIVE_VERIFY_EVERY = 20;  // verify 1 in N batches when skipping

    // Two-pass expert inference (K=55 LUT + confidence-guided repair)
    struct token_record {
        llama_token token_id;
        float logit_margin;      // top-1 minus top-2 logit
        int expert_miss_layers;  // number of layers with cache misses
    };
    std::vector<token_record> twopass_records;
    bool twopass_enabled = false;

    // Two-pass analysis: find suspect tokens
    struct suspect_info {
        int position;
        float margin;
        int miss_layers;
    };

    std::vector<suspect_info> find_suspects(float margin_threshold = 1.5f) const {
        std::vector<suspect_info> suspects;
        for (int i = 0; i < (int)twopass_records.size(); i++) {
            auto & rec = twopass_records[i];
            if (rec.logit_margin < margin_threshold && rec.expert_miss_layers > 0) {
                suspects.push_back({i, rec.logit_margin, rec.expert_miss_layers});
            }
        }
        return suspects;
    }

    bool suspects_are_clustered(const std::vector<suspect_info> & suspects, int max_gap = 5) const {
        if (suspects.size() <= 1) return false;
        for (size_t i = 1; i < suspects.size(); i++) {
            if (suspects[i].position - suspects[i-1].position <= max_gap)
                return true;
        }
        return false;
    }

    // Build annotated text with suspect markers for Strategy B self-repair
    std::string build_repair_prompt(const std::string & original_prompt,
                                     const std::vector<suspect_info> & suspects) const {
        if (suspects.empty() || twopass_records.empty()) return "";

        // Decode Pass 1 output with [?...?] markers on suspects
        std::string annotated;
        std::set<int> suspect_positions;
        for (auto & s : suspects) suspect_positions.insert(s.position);

        // We need the token text - stored in generated_text already
        // Mark suspect regions in the generated text
        // For now, return the repair prompt template
        std::string draft = generated_text;

        return original_prompt + "\n\n"
            "Draft answer (some words may be incorrect due to processing errors):\n"
            + draft + "\n\n"
            "Review the draft carefully. If any words seem wrong or out of place, "
            "fix them. Rewrite the complete corrected answer:";
    }

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;
        twopass_records.clear();
        force_cpu_moe = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        pending_stream_tokens.clear();
        lut_history.clear();
        moe_prev_tok = LLAMA_TOKEN_NULL;
        moe_gen_tokens = 0;
        moe_long_gen = false;
        moe_lut_prefill = false;
        moe_early_eos_repaired = false;
        moe_confidence_fixes = 0;
        moe_cpu_remaining = 0;
        moe_adaptive_warmup = false;
        moe_warmup_tokens_remaining = 0;
        moe_sparse_prefill = false;
        moe_pass = 0;
        moe_draft_text.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        // NOTE: adaptive verify-skip state persists across requests.
        // Once skip-verify is earned (95%+ acceptance), it stays until a rejection.

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_INF(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx) && llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec || (task && task->params.speculative.self_spec_skip > 0);
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        int n_draft_max = task->params.speculative.n_max;

        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        n_draft_max = std::min(n_draft_max, n_ctx - prompt.n_tokens() - 2);

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        if (n_draft_max < task->params.speculative.n_min) {
            SLT_DBG(*this, "the max possible draft is too small: %d < %d - skipping speculative decoding\n", n_draft_max, task->params.speculative.n_min);
            n_draft_max = 0;
        }

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        const int n_draft_max = get_n_draft_max();
        if (n_draft_max > 0) {
            GGML_ASSERT(can_speculate());

            // generate draft tokens in speculative decoding mode
            const llama_tokens & tokens = prompt.tokens.get_text_tokens();

            const auto & params_spec = task->params.speculative;

            if (!spec_draft.empty()) {
                // we have a previous (partial) draft to reuse
                if (ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    GGML_ASSERT(!spec_ckpt.empty());
                }
            } else if (params_spec.self_spec_skip > 0) {
                GGML_ASSERT(spec_i_batch.empty());
                // Layer-skip draft tokens were generated in the pre-batch phase
                // (see self_spec_generate_drafts). They're already in spec_draft.
            } else {
                GGML_ASSERT(spec_i_batch.empty());

                // generate a new draft (n-gram)
                spec_draft = common_speculative_draft(spec.get(), params_spec, tokens, sampled);

                if (spec_draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(*this, "draft size %d exceeds max %d, truncating\n", (int) spec_draft.size(), n_draft_max);
                    spec_draft.resize(n_draft_max);
                }

                if (spec_draft.size() < (size_t) params_spec.n_min) {
                    SLT_DBG(*this, "ignoring small draft: %d < %d\n", (int) spec_draft.size(), params_spec.n_min);
                    spec_draft.clear();
                }

                if (!spec_draft.empty() && ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    const auto n_tokens = prompt.tokens.size();

                    spec_ckpt = server_get_checkpoint(ctx, this->id, n_tokens);

                    SLT_DBG(*this, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %zu, size = %.3f MiB)\n",
                            spec_ckpt.pos_min, spec_ckpt.pos_max, n_tokens, (float) spec_ckpt.data.size() / 1024 / 1024);
                }
            }

            GGML_ASSERT(spec_draft.size() <= (size_t) n_draft_max);
        }

        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else if (adaptive_skip_verify &&
                   (adaptive_verify_batches + 1) % ADAPTIVE_VERIFY_EVERY != 0) {
            // Adaptive verify-skip: trust draft tokens, only decode sampled on main model.
            // spec_i_batch stays empty -> verify section will detect skip mode.
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "adaptive skip-verify: id=%d, #draft=%zu (trusting draft)\n",
                    sampled, spec_draft.size());

            adaptive_verify_batches++;
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    // Single-context layer-skip self-speculative decoding.
    void self_spec_generate_drafts() {
        if (!task || task->params.speculative.self_spec_skip <= 0) return;
        if (state != SLOT_STATE_GENERATING) return;
        if (!spec_draft.empty()) return;

        const auto & params_spec = task->params.speculative;
        const int n_draft_max = get_n_draft_max();
        if (n_draft_max <= 0) return;

        const int M = std::min(params_spec.self_spec_m, n_draft_max);
        if (M <= 0) return;

        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
        const int n_layer = llama_model_n_layer(llama_get_model(ctx));

        // Middle-band skip: skip [band_lo, band_hi) (the offloaded CPU layers) but keep the last
        // out_keep output layers, so the draft is a cheap all-GPU forward. Skipping the tail instead
        // gives near-zero acceptance on hybrid models.
        const int out_keep = []() { const char * e = getenv("ALPHA_SPEC_OUT_KEEP"); return e ? atoi(e) : 6; }();
        const int band_hi  = n_layer - out_keep;
        const int band_lo  = band_hi - params_spec.self_spec_skip;
        if (band_lo < 1 || band_hi > n_layer || band_hi <= band_lo) return;

        llama_memory_t mem = llama_get_memory(ctx);

        // Hybrid-safe rollback: checkpoint the recurrent (DeltaNet) state with a deep copy (seq_cp shares
        // recurrent state shallowly, which corrupts it), draft in the main seq, then restore. The
        // positional KV is rolled back separately via seq_rm, which is much cheaper than snapshotting
        // the whole KV every step. The verify's partial-acceptance path restores the same checkpoint.
        const bool dbg_timing = getenv("ALPHA_SPEC_TIMING") != nullptr;
        const int64_t t_a = dbg_timing ? ggml_time_us() : 0;

        // GPU-side recurrent (DeltaNet) snapshot: a device-to-device copy, much cheaper than the host
        // round-trip of llama_state_seq_get/set_data_ext. spec_ckpt keeps only positions and a 1-byte
        // payload so update_batch's !empty() assert passes; the verify's partial-accept rollback
        // restores via gpu_restore, not the host path.
        llama_memory_gpu_snapshot(mem);
        spec_ckpt.pos_min  = llama_memory_seq_pos_min(mem, this->id);
        spec_ckpt.pos_max  = llama_memory_seq_pos_max(mem, this->id);
        spec_ckpt.n_tokens = prompt.tokens.size();
        spec_ckpt.data.assign(1, 0);

        const int64_t t_b = dbg_timing ? ggml_time_us() : 0;

        // Middle-band skip via the graph-reuse-keyed mask (draft_skip_mask) rather than a thread-local:
        // toggling it forces a graph rebuild, so the verify (mask cleared) cannot reuse the draft's
        // skip graph and silently run a skip forward.
        uint64_t skip_mask = 0;
        for (int i = band_lo; i < band_hi; i++) skip_mask |= (1ull << i);
        llama_context_set_draft_skip_mask(ctx, skip_mask);

        // Draft M tokens with greedy sampler, decoding into the main seq (rolled back below).
        llama_sampler * draft_smpl = llama_sampler_init_greedy();
        llama_token draft_tok = sampled;
        llama_pos   draft_pos = prompt.tokens.pos_next();

        for (int d = 0; d < M; d++) {
            llama_batch dbatch = llama_batch_init(1, 0, 1);
            common_batch_add(dbatch, draft_tok, draft_pos + d, {this->id}, true);
            int rc = llama_decode(ctx, dbatch);
            llama_batch_free(dbatch);
            if (rc != 0) break;

            draft_tok = llama_sampler_sample(draft_smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, draft_tok)) break;
            spec_draft.push_back(draft_tok);
        }

        llama_sampler_free(draft_smpl);
        llama_context_set_draft_skip_mask(ctx, 0);  // verify + all subsequent decodes run the full model

        const int64_t t_c = dbg_timing ? ggml_time_us() : 0;

        // Roll the draft back: the draft advanced canonical -> gpu_restore it. The verify then
        // re-decodes [sampled, d0, d1, ...] on canonical from P.
        llama_memory_gpu_restore(mem);
        llama_memory_seq_rm(mem, this->id, spec_ckpt.pos_max + 1, -1);

        if (dbg_timing) {
            const int64_t t_d = ggml_time_us();
            static int64_t s_snap = 0, s_draft = 0, s_rest = 0; static int s_n = 0;
            s_snap += t_b - t_a; s_draft += t_c - t_b; s_rest += t_d - t_c; s_n++;
            if (s_n % 20 == 0) {
                fprintf(stderr, "[self-spec timing] avg/window over %d: gpu_snap=%.2fms draft(%dtok+graph)=%.2fms gpu_restore+rm=%.2fms\n",
                        s_n, s_snap / 1e3 / s_n, M, s_draft / 1e3 / s_n, s_rest / 1e3 / s_n);
            }
        }
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            if (callback_on_release) callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string_view text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)%s\n",
                    draft_ratio, n_draft_accepted, n_draft_total,
                    adaptive_skip_verify ? " [adaptive skip-verify ACTIVE]" : ""
            );
        }

        common_speculative_print_stats(spec.get());
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        // Guard: seq_cp corrupts shared recurrent state on hybrid models (DeltaNet+GQA)
        auto * mem = llama_get_memory(ctx);
        if (mem && dynamic_cast<llama_memory_hybrid *>(mem) != nullptr) {
            SLT_WRN(other, "%s", "copy_state_to skipped: hybrid model does not support seq_cp\n");
            return;
        }

        llama_memory_seq_rm(mem, other.id,     -1, -1);
        llama_memory_seq_cp(mem, id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        // Fast path: count busy slots without full iteration when possible
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
        }
        // Skip n_tokens_max update - only needed for metrics endpoint, not hot path
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


// ---------------------------------------------------------------------------
// Unified retrieval index (--context unified): a question router + per-class
// focused assembly.
// Built in one pass at ingest over Key:Value record corpora; queried per request to
// inject exactly one focused context (entity record / aggregation count / section
// stats) instead of a merged retrieval dump ("route and focus", never merge-all).
// Generic: keys off Key:Value / "## " record structure only. Prose corpora yield no
// route -> the caller falls back to BM25.
// ---------------------------------------------------------------------------
struct unified_index {
    // Build-time scratch (populated during ingest, consumed by finalize(), then freed).
    // These dicts are how the corpus is parsed; the resident serving state is the packed `tier`
    // below (records.bin mmap + columnar/CSR), which finalize() builds from these and then clears.
    // Aggregation: value -> total count across the corpus (answers "how many use X").
    std::unordered_map<std::string, int> val_count;
    // Entity: id -> the whole record block (a record straddles chunk boundaries; keep it intact).
    std::unordered_map<std::string, std::string> entity;
    // Grouped index + multi-hop state, generic over parsed Key:Value fields + sections.
    std::unordered_map<std::string, std::vector<std::pair<std::string,std::string>>> rec_kv;  // id -> [(key,val)] lower
    std::unordered_map<std::string, std::string> rec_type;      // id -> record type (agent/base/service...) for prose totals
    std::unordered_map<std::string, std::string> rec_section;   // id -> section name (lower)
    std::unordered_map<std::string, std::vector<std::string>> rec_deps;  // id -> [linked rec ids] (edges)
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> groups; // key -> value -> [ids]

    // Resident (packed, tiered): the query path reads this after finalize(); the dicts above are gone.
    tiered_store tier;

    // Section/division stats stay a small resident map (few sections; not text-heavy).
    struct section { std::string name, lead; int n_items = 0; std::unordered_map<std::string,int> region; };
    std::unordered_map<std::string, section> sections;
    bool built = false;
    std::string records_path;  // full path finalize() writes records.bin to (unlinked after mmap); set at ingest

    static std::string lower(std::string s){ for(char&c:s) if(c>='A'&&c<='Z') c=(char)(c-'A'+'a'); return s; }
    static std::string trim(const std::string & s){
        size_t a=s.find_first_not_of(" \t\r\n"); if(a==std::string::npos) return "";
        size_t b=s.find_last_not_of(" \t\r\n"); return s.substr(a,b-a+1);
    }
    // split query into lowercased tokens, stripping surrounding punctuation but keeping internal - and .
    static std::vector<std::string> toks(const std::string & q){
        std::vector<std::string> out; std::string cur;
        for(char c : q){
            bool k = (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='-'||c=='.'||c=='_';
            if(k) cur+=c; else { if(cur.size()) out.push_back(lower(cur)); cur.clear(); }
        }
        if(cur.size()) out.push_back(lower(cur));
        return out;
    }

    // Common English function/question words - never a valid record subject or multi-hop seed.
    static bool is_common(const std::string & t){
        static const std::unordered_set<std::string> cw = {
            "the","a","an","of","to","in","on","at","by","for","with","and","or","is","are","was","were",
            "be","been","this","that","these","those","it","its","as","from","but","not","no","what","which",
            "who","whom","whose","when","where","why","how","will","would","can","could","should","may","might",
            "must","do","does","did","has","have","had","you","your","they","their","them","he","she","his","her",
            "we","our","us","me","my","all","any","some","one","two","new","more","most","other","such","only",
            "also","than","then","there","here","about","into","over","under","out","up","down","if","so","because"};
        return cw.count(t) > 0;
    }

    // Prose record extractor (generic, no schema): a labeled prose line like
    //   "AGENT DOSSIER: Agent Halloway, field code QX-4471, assigned to Base Ironwood, clearance level 5, specialty cryptography."
    // -> subject id (first Capitalized non-type token) + attribute values (last token of each comma clause).
    // Gated on an ALLCAPS(>=4)+":" or bracketed "[...]" record marker so free-flowing filler is not extracted.
    // The crude values feed groups[value] (filtered agg / diversity) + edges (a value == another record id).
    static bool extract_prose(const std::string & ls, std::string & subj, std::string & rtype,
                              std::vector<std::pair<std::string,std::string>> & kvs){
        std::string head = ls.substr(0, std::min<size_t>(ls.size(), 90));
        bool labeled = (!ls.empty() && ls[0]=='[');
        if(!labeled){ int run=0; for(char c : head){
            if(c>='A'&&c<='Z') run++;
            else if(c==':'){ if(run>=4){ labeled=true; break; } run=0; }
            else if(c==' '||c=='['||c==']') { /* keep run across multiword labels */ }
            else run=0; } }
        if(!labeled) return false;
        size_t cs=0; { size_t p=head.rfind(": "); if(p!=std::string::npos) cs=p+2; }
        std::string content = trim(ls.substr(cs));
        if(content.size()<3 || content.size()>400) return false;
        std::vector<std::string> clauses; { size_t s=0; while(s<=content.size()){
            size_t e=content.find(',',s); if(e==std::string::npos) e=content.size();
            clauses.push_back(trim(content.substr(s,e-s))); s=e+1; } }
        if(clauses.size()<2) return false;
        // Subject/type from clause[0] without a type-word list: type = 1st capitalized token, subject = 2nd (else 1st).
        // The "type precedes name" convention ("Agent Halloway", "Base Ironwood", "Employee Smith")
        // generalizes to any entity type with no hardcoded type-word list.
        { std::vector<std::string> caps; std::string cur;
          auto flush=[&](){ if(!cur.empty() && cur[0]>='A' && cur[0]<='Z'){
              std::string cl; for(char d:cur){ char e=(d>='A'&&d<='Z')?(char)(d-'A'+'a'):d;
                  if((e>='a'&&e<='z')||(e>='0'&&e<='9')||e=='-') cl+=e; } if(cl.size()>=2) caps.push_back(cl); } cur.clear(); };
          for(char c : clauses[0]){ if(c==' ') flush(); else cur+=c; } flush();
          if(caps.size()>=2){ rtype=caps[0]; subj=caps[1]; }
          else if(caps.size()==1) subj=caps[0]; }
        if(subj.empty() || subj.size()<3 || is_common(subj)) return false;   // no filler/common-word records
        for(const auto & cl : clauses){ auto w=toks(cl); if(w.empty()) continue;   // (attr key, value=last token)
            const std::string & val = w.back(); if(val.size()<2 || val==subj) continue;
            std::string key; for(size_t k=0;k+1<w.size();k++) key += (k?" ":"") + w[k];
            kvs.push_back({key.empty()?std::string("_p"):key, val}); }
        return !kvs.empty();
    }

    // Biographical lead sentences encode birth/death years in their first parenthetical:
    // "Name (born 28 September 1934)" or "Name (1879-1955) was ...". This is the one reliably parseable
    // structured fact in encyclopedic prose, and it feeds the grouped-index routes ("how many born in Y",
    // "list people born in Y"). Returns birth (>0) and death (0 if none).
    static bool extract_bio(const std::string & text, int & birth, int & death){
        birth = death = 0;
        size_t scan = std::min<size_t>(text.size(), 300);
        size_t op = text.find('(');
        if(op==std::string::npos || op>=scan) return false;
        size_t cp = text.find(')', op);
        if(cp==std::string::npos || cp-op>82) return false;
        std::string in = text.substr(op+1, cp-op-1);
        std::vector<int> yrs;
        for(size_t i=0;i<in.size();){
            if(in[i]>='0'&&in[i]<='9'){
                size_t j=i; while(j<in.size() && in[j]>='0'&&in[j]<='9') j++;
                if(j-i==4){ int y=atoi(in.substr(i,4).c_str());
                    if((y>=1000&&y<=1999)||(y>=2000&&y<=2029)) yrs.push_back(y); }
                i=j;
            } else i++;
        }
        if(yrs.empty()) return false;
        birth=yrs[0]; death=(yrs.size()>=2)?yrs[1]:0;
        if(birth<1000||birth>2020){ birth=death=0; return false; }
        if(death && (death<birth||death>2025)) death=0;
        return true;
    }

    // Parse one file: records separated by blank lines; "## " starts an entity record; "# Division"
    // scopes a section; Key:Value lines feed aggregation counts + section lead/region stats.
    // val_count counts distinct records per value (a value repeated within one record counts once),
    // and the file's NER entities are indexed as aggregatable values (distinct per file) so
    // "how many docs mention <entity>" works on code and prose too. doc_entities/source are the
    // caller's already-extracted NER entities and file path.
    void ingest(const std::string & text,
                const std::vector<entity_t> & doc_entities = {},
                const std::string & /*source*/ = "",
                const std::string & article_title = ""){
        built = true;
        // For an article with a title (Wikipedia-style corpus), lift birth/death years
        // from the lead sentence and register the article subject as a record keyed by those years, so
        // the grouped index answers born-in-year aggregation / diversity / cross-ref on real bios.
        if(!article_title.empty()){
            int by=0, dy=0;
            if(extract_bio(text, by, dy)){
                std::string subj = lower(article_title);
                if(entity.find(subj)==entity.end()){
                    std::string lead = text.substr(0, std::min<size_t>(text.size(), 300));
                    entity[subj] = "[Article: " + article_title + "]\n" + lead;
                }
                auto & kvv = rec_kv[subj];
                kvv.push_back({"birth_year", std::to_string(by)});
                // death_year is deliberately not indexed as a grouped value: a year that is one person's
                // death year is frequently another's birth year, so a searchable death_year group corrupts
                // every "born in Y" count/list/identify (find_val can't tell which the query means). The
                // death year stays in the lead-sentence record text (entity[subj]) for cross-ref reading.
                (void)dy;
                if(rec_type.find(subj)==rec_type.end()) rec_type[subj]="person";
            }
        }
        std::string cur_section, rec_id, rec_block;
        std::unordered_set<std::string> rec_vals;   // distinct values in the current record
        std::vector<std::pair<std::string,std::string>> cur_kv;   // (key,val) of the current record
        auto flush = [&](){
            if(!rec_id.empty() && !rec_block.empty()){
                std::string id = lower(rec_id);
                entity[id] = rec_block;
                rec_kv[id] = cur_kv;                 // retain fields per record
                rec_section[id] = lower(cur_section);
            }
            for(const auto & v : rec_vals) val_count[v]++;   // one increment per distinct value/record
            rec_vals.clear(); cur_kv.clear();
            rec_id.clear(); rec_block.clear();
        };
        size_t i=0, n=text.size();
        while(i<n){
            size_t eol=text.find('\n',i); if(eol==std::string::npos) eol=n;
            std::string line=text.substr(i,eol-i); i=eol+1;
            if(!line.empty()&&line.back()=='\r') line.pop_back();
            std::string ls=trim(line);
            if(ls.empty()){ flush(); continue; }
            if(ls.rfind("# Division",0)==0){
                flush();
                size_t c=ls.find(':'); cur_section = (c!=std::string::npos)? trim(ls.substr(c+1)) : "";
                if(!cur_section.empty()) sections[lower(cur_section)].name = cur_section;
            } else if(ls.size()>3 && ls[0]=='#' && ls[1]=='#'){   // "## Service: <id>" (any "## Header: id")
                flush();
                size_t c=ls.find(':');
                rec_id = (c!=std::string::npos)? trim(ls.substr(c+1)) : trim(ls.substr(2));
                rec_block = line + "\n";
                if(!cur_section.empty()) sections[lower(cur_section)].n_items++;
            } else {
                if(!rec_id.empty()) rec_block += line + "\n";
                size_t c=ls.find(": ");
                if(c!=std::string::npos && c>0){
                    std::string key=lower(trim(ls.substr(0,c))), val=trim(ls.substr(c+2));
                    if(!rec_id.empty() && !val.empty()) cur_kv.push_back({key, lower(val)}); // incl. long link lists
                    if(!val.empty() && val.size()<60){
                        rec_vals.insert(lower(val));   // distinct per record
                        if(key=="lead" && !cur_section.empty()) sections[lower(cur_section)].lead = val;
                        if(key=="region" && !cur_section.empty()) sections[lower(cur_section)].region[val]++;
                    }
                }
                // Prose record: a standalone labeled line (not inside a ## record) -> extract subject + values
                // so the grouped index and multi-hop work on prose too, not just ## Key:Value records.
                if(rec_id.empty()){
                    std::string subj, rtype; std::vector<std::pair<std::string,std::string>> pkv;
                    if(extract_prose(ls, subj, rtype, pkv)){
                        if(entity.find(subj)==entity.end()) entity[subj] = ls;   // first mention keeps the record line
                        auto & kvv = rec_kv[subj];
                        for(const auto & p : pkv) kvv.push_back(p);
                        // NER augment: index entities the NER found anywhere in this record line as values
                        // (not just comma-tail tokens) -> lets free prose work where attributes aren't comma-delimited.
                        std::string ll = lower(ls);
                        for(const auto & e : doc_entities){ std::string et=lower(e.text);
                            if(et.size()>=3 && et!=subj && !is_common(et) && ll.find(et)!=std::string::npos)
                                kvv.push_back({"_e", et}); }
                        if(!rtype.empty() && rec_type.find(subj)==rec_type.end()) rec_type[subj]=rtype;
                        if(!cur_section.empty() && rec_section.find(subj)==rec_section.end()) rec_section[subj]=lower(cur_section);
                    }
                }
            }
        }
        flush();
        // Index this document's NER entities as aggregatable values (distinct per file).
        std::unordered_set<std::string> ent_seen;
        for(const auto & e : doc_entities){
            std::string et = lower(e.text);
            if(et.size()>1 && ent_seen.insert(et).second) val_count[et]++;
        }
    }

    // After all files are ingested, build the grouped index (groups[field][value]->[ids])
    // and resolve entity edges (a field value that is another record's id, or a comma-list link field).
    // Pure O(records) id-lists; enables filtered counts, full-group diversity, and multi-hop chaining.
    void finalize(){
        groups.clear(); rec_deps.clear();
        for(const auto & kv : rec_kv){
            const std::string & id = kv.first;
            for(const auto & p : kv.second){
                const std::string & key = p.first; const std::string & val = p.second;
                if(val.size()>=2 && val.size()<60) groups[key][val].push_back(id);   // grouped index
                // edge resolution: explicit link fields (comma lists) or an implicit ref (value == a record id)
                bool link_field = key.find("depend")!=std::string::npos || key=="links" ||
                                  key=="requires" || key=="base" || key=="parent" || key=="uses";
                if(link_field){
                    size_t s=0;
                    while(s<=val.size()){
                        size_t e=val.find(',',s); if(e==std::string::npos) e=val.size();
                        std::string m=trim(val.substr(s,e-s)); s=e+1;
                        if(!m.empty() && m!=id && entity.count(m)) rec_deps[id].push_back(m);
                    }
                } else if(val!=id && entity.count(val)){
                    rec_deps[id].push_back(val);
                }
            }
        }
        for(const auto & kv : rec_section)
            if(!kv.second.empty()) groups["_section"][kv.second].push_back(kv.first);
        for(const auto & kv : rec_type)
            if(!kv.second.empty()) groups["_type"][kv.second].push_back(kv.first);

        // Pack into the resident tier (records.bin mmap + columnar/CSR), then free the build dicts.
        // The tier gives the same query output with far less resident RAM. On build failure it keeps
        // records in memory, so the query path always goes through `tier`.
        tier.build(records_path, entity, rec_kv, rec_type, rec_section, rec_deps, groups, val_count);
        std::unordered_map<std::string,std::string>().swap(entity);
        std::unordered_map<std::string,std::vector<std::pair<std::string,std::string>>>().swap(rec_kv);
        std::unordered_map<std::string,std::string>().swap(rec_type);
        std::unordered_map<std::string,std::string>().swap(rec_section);
        std::unordered_map<std::string,std::vector<std::string>>().swap(rec_deps);
        std::unordered_map<std::string,std::unordered_map<std::string,std::vector<std::string>>>().swap(groups);
        std::unordered_map<std::string,int>().swap(val_count);
    }

    // Words that carry aggregation intent, not the subject - excluded when picking the DF subject.
    static bool agg_stopword(const std::string & t){
        static const std::unordered_set<std::string> sw = {
            "how","many","much","number","of","count","total","records","record","documents",
            "document","docs","doc","use","uses","using","used","have","has","had","the","a","an",
            "are","is","was","in","with","that","which","mention","mentions","mentioned","do","does",
            "there","and","or","for","to","by","on","at","this","these","those","them","their"};
        return sw.count(t) > 0;
    }

    // Router + per-class focused assembly. Returns the focused context, or "" (-> caller falls back).
    // `route_out` receives the chosen class name for logging. `bm25` (optional) enables the
    // document-frequency aggregation fallback for prose/unstructured queries with no exact tally.
    std::string route(const std::string & q, std::string & route_out,
                      const bm25_index * bm25 = nullptr) const {
        std::string ql = lower(q);
        auto has = [&](const char* s){ return ql.find(s)!=std::string::npos; };

        // Does the query name an indexed section? (used to disambiguate structural from aggregation)
        const section * sec = nullptr; std::string sec_key;
        for(const auto & kv : sections){
            if(!kv.first.empty() && ql.find(kv.first)!=std::string::npos){ sec=&kv.second; sec_key=kv.first; break; }
        }

        // Find the most-specific grouped field value named in the query (the "db"
        // analog) -> its record-id list (as int ids into the tier). Longest match wins so "redis-cluster"
        // beats "redis". Scans the flat inverted index (grp rows); ties -> first row (deterministic).
        auto find_val = [&](std::vector<int> & ids_out, std::string & val_out)->bool{
            int best=-1; size_t blen=0;
            for(int g=0; g<tier.gn(); g++){
                if(tier.grp_key[g]=="_section") continue;
                const std::string & v = tier.grp_val[g];
                if(v.size()<2 || v.size()<=blen) continue;
                if(ql.find(v)!=std::string::npos){ best=g; val_out=v; blen=v.size(); }
            }
            if(best>=0){
                ids_out.assign(tier.grp_ids.begin()+tier.grp_ids_ptr[best],
                               tier.grp_ids.begin()+tier.grp_ids_ptr[best+1]);
                return true;
            }
            return false;
        };

        // Multi-hop: "depend" phrasing + a seed record with links -> resolve and label the hop the
        // question selects. Co-locating the neighborhood is not enough; labeling the target keeps the
        // model from guessing among distractors.
        {
            std::string seed; int sid=-1;
            for(const auto & t : toks(q)) if(t.size()>=3 && !is_common(t)){
                int id=tier.find_id(t); if(id>=0 && tier.has_deps(id)){ seed=t; sid=id; break; } }
            bool chain = has("depend") || ql.find("'s ")!=std::string::npos || ql.find("s' ")!=std::string::npos;
            if(sid>=0 && chain){
                auto dp = tier.deps_of(sid);
                std::vector<int> deps(dp.first, dp.second);
                route_out="MULTIHOP";
                // Lead with the seed record plainly so a direct-attribute question ("X's field code") is
                // answerable without misleading "dependency" framing. Use the ordered first-listed labeling
                // only when the question asks about dependencies (the disambiguation the structured case needs).
                std::string out;
                out += "Record " + seed + ":\n" + tier.block_of(sid) + "\n---\n";
                if(has("depend") || has("first")){
                    std::string order; for(size_t k=0;k<deps.size();k++) order += (k?", ":"") + tier.name_str(deps[k]);
                    out += seed + " — dependencies in listed order: " + order + ".\n";
                    if(!deps.empty())
                        out += "Record for the FIRST-listed dependency of " + seed + " (= " + tier.name_str(deps[0]) +
                               "):\n" + tier.block_of(deps[0]) + "\n---\n";
                    for(size_t k=1;k<deps.size();k++)
                        out += "Record for another dependency " + tier.name_str(deps[k]) + ":\n" + tier.block_of(deps[k]) + "\n---\n";
                } else {
                    for(int d : deps)
                        out += "Related record " + tier.name_str(d) + " (linked from " + seed + "):\n" + tier.block_of(d) + "\n---\n";
                }
                return out;
            }
        }

        // Computed aggregation (prose): range (>=,<=,>,<), or-union, distinct, type totals over the
        // grouped index - arithmetic over values that exact-value lookup can't do.
        if(has("how many")||has("how much")||has("number of")||has("count")||has("total")){
            long thr=0; bool has_num=false;
            for(const auto & t : toks(q)){ if(t.empty())continue; bool n=true; for(char c:t) if(!(c>='0'&&c<='9')){n=false;break;}
                if(n){ thr=atol(t.c_str()); has_num=true; break; } }
            int cmp=0;
            if(has("or higher")||has("or more")||has("or greater")||has("at least")||has("or above")) cmp=1;       // >=
            else if(has("or lower")||has("or less")||has("or fewer")||has("at most")||has("or below")) cmp=2;      // <=
            else if(has("greater than")||has("more than")||has("higher than")||has("above")||has("over ")) cmp=3;  // >
            else if(has("less than")||has("fewer than")||has("lower than")||has("below")||has("under ")) cmp=4;    // <
            std::string akey; bool have_ag=false;   // chosen aggregation field key
            for(int g=0; g<tier.gn() && !have_ag; g++){ const std::string & k=tier.grp_key[g]; if(!k.empty()&&k[0]=='_') continue;
                for(const auto & w : toks(k)) if(w.size()>=4 && ql.find(w)!=std::string::npos){ akey=k; have_ag=true; break; } }
            if(have_ag && has_num && cmp){                                // range
                std::unordered_set<int> ids;
                for(int g=0; g<tier.gn(); g++){ if(tier.grp_key[g]!=akey) continue; const std::string & vs=tier.grp_val[g];
                    bool n=!vs.empty(); for(char c:vs) if(!(c>='0'&&c<='9')){n=false;break;} if(!n) continue; long v=atol(vs.c_str());
                    if((cmp==1&&v>=thr)||(cmp==2&&v<=thr)||(cmp==3&&v>thr)||(cmp==4&&v<thr))
                        for(int j=tier.grp_ids_ptr[g];j<tier.grp_ids_ptr[g+1];j++) ids.insert(tier.grp_ids[j]); }
                route_out="AGG_RANGE"; return "[Exact precomputed count]\n"+std::to_string(ids.size())+" records match the "+akey+" range.\n";
            }
            if(has(" or ")){                                             // or-union
                std::unordered_set<int> ids; int nv=0;
                for(int g=0; g<tier.gn(); g++){ const std::string & k=tier.grp_key[g]; if(!k.empty()&&k[0]=='_') continue;
                    if(tier.grp_val[g].size()>=3 && ql.find(tier.grp_val[g])!=std::string::npos){
                        for(int j=tier.grp_ids_ptr[g];j<tier.grp_ids_ptr[g+1];j++) ids.insert(tier.grp_ids[j]); nv++; } }
                if(nv>=2){ route_out="AGG_OR"; return "[Exact precomputed count]\n"+std::to_string(ids.size())+" records match any of the listed values.\n"; }
            }
            if((has("distinct")||has("different")||has("unique")) && have_ag){ // distinct
                int dv=0; for(int g=0;g<tier.gn();g++) if(tier.grp_key[g]==akey) dv++;
                route_out="AGG_DISTINCT"; return "[Exact precomputed count]\n"+std::to_string(dv)+" distinct "+akey+" values across the corpus.\n";
            }
            { std::vector<int> _i; std::string _v; bool fv=find_val(_i,_v);   // type total (no value filter)
              if(!fv && (has("total")||has("recorded")||has("described")||has("are there")||has("altogether")||has("in all"))){
                for(int g=0; g<tier.gn(); g++){ if(tier.grp_key[g]!="_type") continue; const std::string & tv=tier.grp_val[g];
                    if(tv.size()>=3 && ql.find(tv)!=std::string::npos){
                        route_out="AGG_TYPE"; return "[Exact precomputed count]\n"+std::to_string(tier.gids_n(g))+" "+tv+" records in the corpus.\n"; } }
              } }
        }

        // Filtered aggregation: "how many" + a grouped value (+ optional section) -> exact count via
        // intersection, rather than leaving the model to count records in the context.
        if(has("how many")||has("how much")||has("number of")||has("count")||has("total")){
            std::vector<int> vids; std::string val;
            if(find_val(vids, val)){
                if(sec && !sec_key.empty()){
                    std::vector<int> matched;
                    for(int id : vids){ if(tier.sect_of(id)==sec_key) matched.push_back(id); }
                    route_out="AGG_FILTERED";
                    std::string names; for(size_t k=0;k<matched.size();k++) names += (k?", ":"") + tier.name_str(matched[k]);
                    return "[Exact precomputed count]\n" + std::to_string(matched.size()) + " records in the " +
                           sec->name + " section match \"" + val + "\".\nMatching records: " + names + "\n";
                }
                route_out="AGG_EXACT";
                return "[Exact precomputed count]\n" + std::to_string(vids.size()) +
                       " records match \"" + val + "\" across the corpus.\n";
            }
        }

        // Computed diversity (prose): min/max over a numeric attribute, or mode (value with most records).
        // "which agent has the highest clearance", "which base has the most agents".
        if(has("highest")||has("lowest")||has("most")||has("least")||has("largest")||has("smallest")||
           has("maximum")||has("minimum")||has("greatest")){
            bool wantmax = has("highest")||has("most")||has("largest")||has("maximum")||has("greatest");
            std::string akey; bool have_ag=false;
            for(int g=0; g<tier.gn() && !have_ag; g++){ const std::string & k=tier.grp_key[g]; if(!k.empty()&&k[0]=='_') continue;
                for(const auto & w : toks(k)) if(w.size()>=4 && ql.find(w)!=std::string::npos){ akey=k; have_ag=true; break; } }
            if(have_ag){
                bool numeric=false; for(int g=0;g<tier.gn();g++) if(tier.grp_key[g]==akey){ const std::string & vs=tier.grp_val[g];
                    if(vs.empty())continue; bool n=true; for(char c:vs) if(!(c>='0'&&c<='9')){n=false;break;} numeric=n; break; }
                if(numeric){                                     // min/max over numeric attribute
                    long best=0; bool have=false; int bid=-1;
                    for(int g=0;g<tier.gn();g++){ if(tier.grp_key[g]!=akey)continue; const std::string & vs=tier.grp_val[g];
                        bool n=!vs.empty(); for(char c:vs) if(!(c>='0'&&c<='9')){n=false;break;} if(!n)continue; long v=atol(vs.c_str());
                        if(!have||(wantmax?v>best:v<best)){ best=v; have=true; if(tier.gids_n(g)>0) bid=tier.grp_ids[tier.grp_ids_ptr[g]]; } }
                    if(have && bid>=0){ route_out="DIV_MINMAX";
                        std::string out="The record with the "+std::string(wantmax?"highest":"lowest")+" "+akey+" is "+tier.name_str(bid)+" ("+akey+" = "+std::to_string(best)+").\n";
                        out+=tier.block_of(bid)+"\n"; return out; }
                } else {                                         // mode: attribute value with the most records
                    std::string bv; size_t bn=0; for(int g=0;g<tier.gn();g++){ if(tier.grp_key[g]!=akey)continue;
                        size_t sz=(size_t)tier.gids_n(g); if(sz>bn){ bn=sz; bv=tier.grp_val[g]; } }
                    if(!bv.empty()){ route_out="DIV_MODE";
                        std::string out="The "+akey+" with the most records is \""+bv+"\" ("+std::to_string(bn)+" records).\n";
                        int vid=tier.find_id(bv); if(vid>=0) out+=tier.block_of(vid)+"\n"; return out; }
                }
            }
        }

        // Diversity / list-all: "list|all|which|every|name the" + a grouped value/section -> dump the
        // whole matching group. Records are byte-capped; the names list is always included.
        if(has("list")||has(" all ")||has("which ")||has("every ")||has("name the")){
            std::vector<int> vids; std::string val; bool hv=find_val(vids,val);
            std::vector<int> matched;
            if(sec && !sec_key.empty()){
                if(hv){ for(int id: vids){ if(tier.sect_of(id)==sec_key) matched.push_back(id); } }
                else { for(int g=0; g<tier.gn(); g++) if(tier.grp_key[g]=="_section" && tier.grp_val[g]==sec_key){
                        for(int j=tier.grp_ids_ptr[g];j<tier.grp_ids_ptr[g+1];j++) matched.push_back(tier.grp_ids[j]); break; } }
            } else if(hv){ matched = vids; }
            if(!matched.empty()){
                route_out="DIVERSITY";
                // Lead with the exact enumerated list from the grouped index so the model only has to relay
                // it. Full records follow for "describe/compare each" style questions (byte-capped).
                std::string names; for(size_t k=0;k<matched.size();k++) names += (k?", ":"") + tier.name_str(matched[k]);
                std::string head = "Matching records (" + std::to_string(matched.size()) + "): " + names + "\n";
                const size_t budget = 8000;
                std::string recs;
                for(int id : matched){ std::string b=tier.block_of(id); if(b.empty()) continue;
                    if(recs.size()+b.size()>budget) break; recs += b + "\n---\n"; }
                return head + (recs.empty() ? "" : ("Full records:\n" + recs));
            }
        }

        // Identify by attribute value: the query names a grouped value (not a record id) and asks about
        // the matching record(s), e.g. "the person born in 1590 died in what year?". Inject those records
        // so the model reads the answer off them. Runs after multi-hop and how-many/list/superlative.
        // Matches exact query tokens (not find_val's longest substring) so a long prose value cannot mask
        // the real key; picks the smallest group and requires <= 4 records so a broad value cannot hijack
        // single-fact retrieval.
        {
            int best=-1; std::string best_val;
            for(const auto & t : toks(q)){
                if(t.size()<3) continue;
                for(int g=0; g<tier.gn(); g++){
                    const std::string & k=tier.grp_key[g];
                    if(k=="_section" || k=="_type") continue;
                    if(tier.grp_val[g]!=t) continue;
                    int sz=tier.gids_n(g);
                    if(sz>0 && sz<=4 && (best<0 || sz<tier.gids_n(best))){ best=g; best_val=t; }
                }
            }
            if(best>=0){
                route_out="ATTR_IDENTIFY";
                std::string out = "Record(s) where " + best_val + " matches:\n";
                for(int j=tier.grp_ids_ptr[best]; j<tier.grp_ids_ptr[best+1]; j++) out += tier.block_of(tier.grp_ids[j]) + "\n---\n";
                if(out.size() > 40) return out;   // had at least one real record
            }
        }

        // 1) Entity: the query contains an indexed record id (return the whole record).
        for(const auto & t : toks(q)){
            int id = tier.find_id(t);
            if(id>=0){ route_out="ENTITY"; return "[Entity Record]\n" + tier.block_of(id); }
        }

        // 2) Structural: names a section + asks a section-scoped stat (lead / count / region).
        if(sec && (has("who lead")||has("leads")||has("how many service")||has("most common")||
                   has("region")||has("services does")||has("number of service"))){
            route_out="STRUCTURAL";
            std::string out = "[Section: " + (sec->name.empty()?std::string("(section)"):sec->name) + "]\n";
            if(!sec->lead.empty()) out += "Lead: " + sec->lead + "\n";
            out += "Services: " + std::to_string(sec->n_items) + "\n";
            if(!sec->region.empty()){
                std::string top; int best=-1; for(const auto & r : sec->region) if(r.second>best){best=r.second; top=r.first;}
                std::string mc = top + " (" + std::to_string(best) + " services)";
                out += "Most common region: " + mc + "\n";
                out += "Region distribution:";
                for(const auto & r : sec->region) out += " " + r.first + "=" + std::to_string(r.second);
                out += "\n";
                // Hand the model the answer in the exact expected form so it states it verbatim.
                out += "If asked for the most common region, answer exactly: " + mc + "\n";
            }
            return out;
        }

        // 3) Aggregation: whole-corpus count of a value ("how many use X"), not section-scoped.
        if(!sec && (has("how many")||has("how much")||has("number of")||has("count")||has("total"))){
            // 3a) Exact tally (structured records + indexed NER entities): distinct-record count.
            for(const auto & t : toks(q)){
                if(t.size()>1){ int c=tier.val_count_of(t);
                    if(c>0){
                        route_out="AGGREGATION";
                        return "[Computed count across the entire corpus]\n" +
                               std::to_string(c) + " records have \"" + t + "\".\n";
                    }
                }
            }
            // 3b) No exact tally: approximate the count with the BM25 document frequency of the query's
            // rarest content token, taken as the aggregation subject.
            if(bm25 && bm25->N > 0){
                int best_df = -1; std::string best_tok;
                for(const auto & t : toks(q)){
                    if(t.size()<=2 || agg_stopword(t)) continue;
                    int d = bm25->df_of(bm25->cipher.hash_term(t));
                    if(d>0 && (best_df<0 || d<best_df)){ best_df=d; best_tok=t; }  // rarest = the subject
                }
                if(best_df>0){
                    route_out="AGGREGATION_DF";
                    return "[Approximate count across the corpus]\n~" + std::to_string(best_df) +
                           " documents mention \"" + best_tok + "\" (approximate document-frequency, "
                           "not an exact entity count).\n";
                }
            }
        }

        route_out="FALLBACK";
        return "";   // no structured path -> caller falls back to BM25
    }
};

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    // Self-spec layer-skip: uses single context with n_layer_draft switching

    server_queue    queue_tasks;
    server_response queue_results;

    // KV-chunk extended context routing (mutable - accessed from const route handlers)
    mutable bm25_index chunk_bm25;
    kv_chunk_routing routing_strategy = kv_chunk_routing::BM25_ONLY;
    int n_ctx_extended = 0;

    // Layer 2: NER + entity linking (built alongside BM25 during ingest)
    mutable code_entity_extractor entity_extractor;
    mutable entity_index chunk_entities;
    mutable entity_linker chunk_linker;

    // Auto-context: resolved mode (from AUTO classification or explicit --context)
    alpha_context_mode effective_ctx_mode = ALPHA_CTX_NATIVE;

    // Comprehend mode: offline-built summary pyramid (pyramid.json), navigated per query.
    json comprehend_pyramid;               // {chunks[], nodes[{id,level,pos,text,child_ids}], levels[], root}
    bool comprehend_loaded = false;
    // Method-2 iterative deepening: descend the pyramid scoring nodes vs the query, returning
    // awareness (corpus + division summaries) + the selected leaf chunks as a context string.
    std::string comprehend_walk(const std::string & query, int budget_chars) const;

#ifdef ALPHA_HAS_REDIS
    // Redis retrieval backend (replaces BM25 when --redis is set)
    std::unique_ptr<redis_index> redis;
    bool use_redis = false;
#endif

#ifdef ALPHA_DENSE_L0
    // Dense-L0 semantic retrieval: in-engine bge-base embedder + cosine store (paraphrase tail).
    // mutable: server_routes holds a const server_context&, and add()/search() mutate the store /
    // decode through the embedding context (same pattern as `mutable metadata_index chunk_metadata`).
    mutable dense_index dense;
    bool use_dense = false;
    // ALPHA_DENSE_L0_PERSIST named an existing file that load_vectors() refused (wrong embedder/config, V1, corrupt):
    // the end-of-ingest save then must not overwrite it -- a refusal that destroys the store it refused turns one boot
    // with a wrong env into a second full embedding pass, and erases the evidence. ALPHA_DENSE_L0_PERSIST_OVERWRITE=1
    // opts into replacing it.
    bool dense_persist_refused = false;
#endif

    // Layer 3: Metadata sub-index for structured record precision
    mutable metadata_index chunk_metadata;

    // Unified mode (--context unified): question router + per-class focused assembly.
    unified_index unified;

    // VRAM stability: reset scheduler periodically to prevent compute buffer growth
    int n_requests_since_reset = 0;
    static constexpr int RESET_SCHED_INTERVAL = 15;  // reset every 15 requests

    // cpu_moe: global flag set by handle_completions_impl, checked at warmup
    mutable bool g_cpu_moe_active = false;
    mutable bool g_classify_active = false;  // true during LUT classify request

    // Domain classifier (0.6B LoRA model for fast domain detection)
    llama_model * classify_model = nullptr;
    llama_context * classify_ctx = nullptr;
    const llama_vocab * classify_vocab = nullptr;
    std::string classify_system_prompt;  // cached system prompt with domain list

    // Classify a user prompt into a domain using the 0.6B classifier model.
    // Returns the domain name or nullptr if classifier not loaded.
    const char * classify_domain_fast(const std::string & user_msg) {
        if (!classify_model || !classify_ctx || !classify_vocab) return nullptr;

        // Build classify prompt using the model's native chat template
        // Supports both ChatML (Qwen) and Gemma formats automatically
        std::string prompt;
        const char * tmpl = llama_model_chat_template(classify_model, nullptr);
        if (tmpl && strstr(tmpl, "gemma")) {
            // Gemma format
            prompt = "<start_of_turn>user\n" + classify_system_prompt + "\n\n" + user_msg +
                     "<end_of_turn>\n<start_of_turn>model\n";
        } else {
            // ChatML format (Qwen, default)
            prompt = "<|im_start|>system\n" + classify_system_prompt +
                     "<|im_end|>\n<|im_start|>user\n" + user_msg +
                     "<|im_end|>\n<|im_start|>assistant\n";
        }

        // Tokenize
        auto tokens = common_tokenize(classify_vocab, prompt, true, true);
        if (tokens.empty() || (int)tokens.size() > 2048) return nullptr;  // increased for larger classifiers

        // Clear KV
        auto * mem = llama_get_memory(classify_ctx);
        if (mem) llama_memory_seq_rm(mem, 0, -1, -1);

        // Decode prompt
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (int i = 0; i < (int)tokens.size(); i++) {
            common_batch_add(batch, tokens[i], i, {0}, i == (int)tokens.size() - 1);
        }
        if (llama_decode(classify_ctx, batch) != 0) {
            llama_batch_free(batch);
            return nullptr;
        }
        llama_batch_free(batch);

        // Constrained decoding: score each domain name against logits.
        // Tokenize each domain name, compute log-probability from logits,
        // pick the domain with highest probability.
        float * logits = llama_get_logits(classify_ctx);
        if (!logits) return nullptr;

        // Build candidate list from loaded domain profiles (cached on first call)
        static std::vector<std::pair<std::string, std::vector<llama_token>>> domain_tokens;
        if (domain_tokens.empty()) {
            int n_dp = alpha_moe_optimizer_get_domain_profile_count();
            for (int d = 0; d < n_dp; d++) {
                const char * name = alpha_moe_optimizer_get_domain_profile_name(d);
                if (!name || strcmp(name, "classify") == 0 || strcmp(name, "grammar") == 0 ||
                    strcmp(name, "core") == 0) continue;
                auto toks = common_tokenize(classify_vocab, name, false, false);
                if (!toks.empty()) {
                    domain_tokens.push_back({name, toks});
                }
            }
        }

        // Score each domain by its first token's logit - return top-2
        // First pass: find best domain
        float best_score = -1e30f;
        static std::string best_domain_buf, second_domain_buf;
        llama_token best_first_tok = -1;
        best_domain_buf.clear();
        second_domain_buf.clear();

        for (auto & [name, toks] : domain_tokens) {
            float score = logits[toks[0]];
            if (score > best_score) {
                best_score = score;
                best_domain_buf = name;
                best_first_tok = toks[0];
            }
        }

        // Second pass: find best domain with different first token
        // (avoids duplicates like high_school_physics + high_school_chemistry)
        float second_score = -1e30f;
        for (auto & [name, toks] : domain_tokens) {
            if (toks[0] == best_first_tok) continue;  // skip same-prefix domains
            float score = logits[toks[0]];
            if (score > second_score) {
                second_score = score;
                second_domain_buf = name;
            }
        }

        // Set domain2 for dual-domain expert loading
        // Disabled: a wrong domain2 fills K slots with bad experts and lowers accuracy.
        // TODO: re-enable when the classifier can generate multi-token domain names.
        if (false && !second_domain_buf.empty()) {
            alpha_moe_set_detected_domain2(second_domain_buf.c_str());
        }

        return best_domain_buf.empty() ? nullptr : best_domain_buf.c_str();
    }

    bool has_classifier() const { return classify_model != nullptr; }

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;

    // Alpha-MoLE: MoE optimization state
    alpha_moe_optimizer * moe_opt = nullptr;
    alpha_moe_profile moe_portfolio = {};
    bool moe_lut_active = false;
    int  moe_K_estimate = 0;       // portfolio K estimate from HW planner
    size_t moe_gpu_vram_mb = 0;    // GPU VRAM in MB (for feature gating)
    // Expert-usage logging (ALPHA_LOG_EXPERT_USAGE): generic expert counts + the side-log file handle.
    int    eu_n_expert_total = 0;  // per-MoE-layer expert count (n_expert; auto-detected, never hardcoded)
    int    eu_n_expert_used  = 0;  // top-k per token
    bool   eu_log_enabled    = false;
    bool   eu_log_verbose    = false;  // ALPHA_LOG_EXPERT_USAGE=2 -> also emit the full [L][E] matrix
    FILE * eu_log_fp         = nullptr;
    // Branch-and-recover decode (ALPHA_BRANCH_RECOVER) - greedy-by-default sampler with sparse,
    // entropy-gated nucleus "branches", each followed by a forced-greedy recovery window. Default off
    // => byte-identical to the normal sampler path (br_sample never runs). Pure sampler policy - no
    // routing/model assumptions; seeded => a candidate set is reproducible. Set from env once at load.
    bool     br_enabled     = false;
    float    br_entropy_min = 1.5f;   // nats; branch only when H(next-token) >= this (a real fork)
    int      br_recover_n   = 6;      // forced-greedy tokens after a branch (re-condition the branch)
    float    br_topp        = 0.95f;  // nucleus the branch token is drawn from
    int      br_max_branch  = 8;      // cap total branches per generation (bounds divergence)
    uint64_t br_seed        = 0;      // per-generation RNG seed (reproducible branch draws)
public:
    bool is_moe_lut_active() const { return moe_lut_active; }
    int    get_moe_K_estimate() const { return moe_K_estimate; }
    size_t get_moe_gpu_vram_mb() const { return moe_gpu_vram_mb; }
    alpha_moe_optimizer * moe_opt_public() const { return moe_opt; }
    llama_context * get_ctx() const { return ctx; }
    const std::vector<server_slot> & get_slots() const { return slots; }
    kv_pool & get_kv_pool() { return kv_pool_cache; }
    const kv_pool & get_kv_pool() const { return kv_pool_cache; }
private:

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    std::vector<server_slot> slots;

    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    // RadixAttention: token-level prefix tree for KV cache sharing
    radix_cache radix_kv_cache;

    // 3-tier KV pool: L2 (CPU RAM) + L3 (Redis persistent)
    kv_pool kv_pool_cache;

    // Pipeline mode: snapshot-based output processing
    bool pipeline_snapshot_mode = false;
    std::mutex * snap_mtx_ptr = nullptr;
    std::condition_variable * snap_cv_ptr = nullptr;
    std::vector<slot_snapshot> * snap_queue_ptr = nullptr;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    void destroy() {
        // Alpha-MoLE cleanup
        if (moe_opt) {
            alpha_moe_optimizer_free(moe_opt);
            moe_opt = nullptr;
        }
        moe_lut_active = false;

        // Classifier cleanup
        if (classify_ctx) { llama_free(classify_ctx); classify_ctx = nullptr; }
        if (classify_model) { llama_model_free(classify_model); classify_model = nullptr; }
        classify_vocab = nullptr;

        llama_init.reset();

        ctx = nullptr;
        model = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        for (server_slot & slot : slots) {
            if (slot.can_speculate()) {
                slot.spec.reset();
            }
        }

        llama_batch_free(batch);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache + radix tree\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CLEAR_IDLE_SLOT__\n");

        // Save to radix tree (token-level prefix sharing)
        if (radix_kv_cache.memory_budget() > 0 && slot.prompt.tokens.size() > 0) {
            auto tokens_vec = slot.prompt.tokens.get_tokens();
            const size_t state_size = llama_state_seq_get_size_ext(ctx, slot.id, 0);
            if (state_size > 0 && state_size < radix_kv_cache.memory_budget() / 2) {
                std::vector<uint8_t> kv_data(state_size);
                llama_state_seq_get_data_ext(ctx, kv_data.data(), state_size, slot.id, 0);
                radix_kv_cache.insert(tokens_vec.data(), tokens_vec.size(),
                                       kv_data.data(), kv_data.size());
                SLT_INF(slot, "radix: saved %d tokens, %.1f MB KV state\n",
                        (int)tokens_vec.size(), state_size / (1024.0 * 1024.0));
            }
        }

        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;


        // MoE Portfolio: architecture-aware strategy selection
        // One system for all models. Strategy decided at load time, respected throughout.
        //
        // | Condition                          | Strategy          | Expert Cache |
        // |------------------------------------|-------------------|-------------|
        // | model fits on GPU                  | ALL_GPU           | none        |
        // | MoE + arch unsupported (qwen3next) | STANDARD_OFFLOAD  | none        |
        // | MoE + arch supported               | LUT_DIRECT        | K slots     |
        //
        int moe_expert_cache_mb_pre = 0;
        enum { STRAT_DEFAULT, STRAT_ALL_GPU, STRAT_STANDARD_OFFLOAD, STRAT_LUT_DIRECT } moe_strategy = STRAT_DEFAULT;
        {
            alpha_hw_plan hw_pre = alpha_hw_plan_create(params.model.path.c_str(), -1);
            moe_expert_cache_mb_pre = hw_pre.expert_cache_mb;

            if (hw_pre.n_expert > 1) {
                // Auto-detect MoE models and apply offload strategy.
                // Even in STANDARD mode, if the model doesn't fit in VRAM,
                // auto-switch to cpu-moe to avoid OOM.
                // Check if user explicitly forced experts to CPU via --override-tensor
                // Skip MoE strategy for --mode standard (vanilla llama.cpp, no ctx cap)
                bool user_forced_cpu_experts = false;
                for (auto & o : params_base.tensor_buft_overrides) {
                    if (o.pattern && strstr(o.pattern, "ffn_") && strstr(o.pattern, "CPU")) {
                        user_forced_cpu_experts = true;
                        break;
                    }
                }

                if (hw_pre.model_size_mb <= hw_pre.gpu_vram_total_mb && !user_forced_cpu_experts) {
                    // Strategy 1: ALL_GPU - model fits entirely on GPU
                    // No expert cache needed, native speed. Don't touch anything.
                    moe_strategy = STRAT_ALL_GPU;
                    SRV_INF("MoE portfolio: ALL_GPU (%zuMB model fits in %zuMB VRAM)\n",
                            hw_pre.model_size_mb, hw_pre.gpu_vram_total_mb);

                } else if (strcmp(hw_pre.arch, "qwen3next") == 0) {
                    // Strategy 2: STANDARD_OFFLOAD - let llama.cpp handle layer placement.
                    moe_strategy = STRAT_STANDARD_OFFLOAD;
                    params_base.use_mmap = true;
                    params_base.n_parallel = 1;
                    if (params_base.n_ctx > 4096 || params_base.n_ctx == 0) {
                        params_base.n_ctx = 4096;
                    }
                    SRV_INF("MoE portfolio: STANDARD_OFFLOAD for %s (%d experts, llama.cpp layer split)\n",
                            hw_pre.arch, hw_pre.n_expert);
                    fprintf(stderr, "\n"
                        "This model has %d experts and needs %zuMB, but your GPU has %zuMB VRAM.\n"
                        "Splitting layers between GPU and CPU. Quality is identical — speed is reduced.\n"
                        "For full GPU speed, use a GPU with %zuMB+ VRAM.\n\n",
                        hw_pre.n_expert, hw_pre.model_size_mb, hw_pre.gpu_vram_total_mb,
                        hw_pre.model_size_mb + 2048);

                } else {
                    // Strategy 3: LUT_DIRECT - full expert cache pipeline
                    // Expert tensors -> CPU, backbone -> GPU, LUT for generation.
                    moe_strategy = STRAT_LUT_DIRECT;
                    bool has_override = false;
                    for (auto & o : params_base.tensor_buft_overrides) {
                        if (o.pattern && strstr(o.pattern, "ffn_")) { has_override = true; break; }
                    }
                    if (!has_override) {
                        params_base.tensor_buft_overrides.clear();
                        params_base.tensor_buft_overrides.push_back({"exps", ggml_backend_cpu_buffer_type()});
                        params_base.tensor_buft_overrides.push_back({"nextn", ggml_backend_cpu_buffer_type()});
                        params_base.tensor_buft_overrides.push_back({nullptr, nullptr});
                        params_base.fit_params = false;
                        if (params_base.n_gpu_layers == 0 || params_base.n_gpu_layers >= 999) {
                            params_base.n_gpu_layers = 999;
                        } else {
                            SRV_INF("MoE: respecting explicit --gpu-layers %d\n", params_base.n_gpu_layers);
                        }
                        params_base.use_mmap = true;
                        params_base.n_parallel = 1;
                        // Default: clamp GPU ctx to 2048 (extended context comes from KV chunks;
                        // small window maximizes the expert pool). ALPHA_KEEP_CTX=1 honors an
                        // explicit --ctx-size instead - safe here because this hybrid's KV covers
                        // only the few GQA layers (~0.4GB @40K), so a large window still fits.
                        {
                            const char * keep = getenv("ALPHA_KEEP_CTX");
                            bool honor = keep && keep[0] && keep[0] != '0' && params_base.n_ctx > 0;
                            if (!honor && (params_base.n_ctx > 2048 || params_base.n_ctx == 0)) {
                                params_base.n_ctx = 2048;
                            } else if (honor) {
                                // common_context_params_to_llama reserves +1 seq for KV-chunk ops
                                // (n_seq_max = n_parallel+1) and n_ctx is partitioned evenly across all
                                // seqs, so "-c 16384" would give the single real slot only 8192. Scale
                                // n_ctx up so each real slot keeps the full requested context. KV here
                                // covers only the few GQA layers, so the extra partition is cheap.
                                const int np        = params_base.n_parallel > 0 ? params_base.n_parallel : 1;
                                const long long req = params_base.n_ctx;                 // requested per-slot ctx
                                params_base.n_ctx   = (int)((req * (np + 1) + np - 1) / np); // ceil(req*(np+1)/np)
                                fprintf(stderr,
                                    "[ctx] ALPHA_KEEP_CTX: honouring -c %lld as usable per-slot context — "
                                    "allocating n_ctx=%d across %d slot(s) + 1 reserved KV-chunk seq "
                                    "(n_seq_max=%d); each request may use up to %lld tokens.\n",
                                    req, params_base.n_ctx, np, np + 1, req);
                            }
                        }
                    }
                    SRV_INF("MoE portfolio: LUT_DIRECT (%d experts to CPU, ngl=%d, ctx=%d)\n",
                            hw_pre.n_expert, params_base.n_gpu_layers, params_base.n_ctx);
                    // Explain offload + recommend models that fit
                    {
                        size_t vram = hw_pre.gpu_vram_total_mb;
                        size_t model_mb = hw_pre.model_size_mb;
                        fprintf(stderr, "\n"
                            "This model has %d experts and needs %zuMB, but your GPU has %zuMB VRAM.\n"
                            "Expert weights stored in RAM, loaded on-demand. Quality is identical — speed is reduced.\n",
                            hw_pre.n_expert, model_mb, vram);

                        // Speed comparison
                        fprintf(stderr, "\n"
                            "  Your config:    %zuMB model on %zuMB GPU → CPU offload, slower\n"
                            "  Full GPU:       would need %zuMB+ VRAM  → much faster\n",
                            model_mb, vram, model_mb + 2048);

                        // Suggest sizing guidance with fit status for this GPU
                        fprintf(stderr, "\n  Model size guide (Q4 quantization) for your %zuMB GPU:\n", vram);
                        struct size_guide { const char * label; size_t mb; const char * note; };
                        size_guide guides[] = {
                            {"<1B params",    500,  "fastest, basic tasks"},
                            {"~2B params",   1300,  "good quality, fast"},
                            {"~4B params",   2600,  "strong quality"},
                            {"~7-9B params", 5500,  "very strong"},
                            {"~14B params",  8500,  "excellent quality"},
                            {"~35B MoE",     21000, "best quality"},
                            {nullptr, 0, nullptr},
                        };
                        for (int i = 0; guides[i].label; i++) {
                            bool fits = guides[i].mb + 1024 <= vram;
                            fprintf(stderr, "      %-14s ~%5zuMB  %-22s %s\n",
                                guides[i].label, guides[i].mb, guides[i].note,
                                fits ? "[fits in VRAM]" : "[needs offloading]");
                        }
                        fprintf(stderr, "    Tip: IQ2 quantization halves the size (e.g. 4B IQ2 = ~1.7GB)\n");
                        fprintf(stderr, "    Tip: models with 'vision' or 'VL' in the name support image input\n");
                        fprintf(stderr, "\n"
                            "  Your choice:    %zuMB model → offloaded, slower but best quality\n"
                            "  Alternative:    smaller model on full GPU → much faster, slightly less capable\n"
                            "\n  Current config works correctly. No action needed.\n\n",
                            model_mb);
                    }
                }
            }
        }

        // Dense model auto-fit: when model doesn't fit in VRAM, apply HW planner's
        // layer/KV/ctx recommendations instead of OOM crashing.
        // MoE models handle this via their own expert cache path above.
        bool moe_direct_load = (!params_base.tensor_buft_overrides.empty() &&
            params_base.tensor_buft_overrides.size() > 1 &&
            params_base.tensor_buft_overrides[0].pattern != nullptr);

        if (!moe_direct_load && params_base.n_gpu_layers > 0) {
            // Dense model auto-fit: apply HW planner's recommendations for
            // layer count, KV quantization, and context reduction.
            // Handles both "model doesn't fit" and "model fits but KV cache doesn't".
            alpha_hw_plan hw_dense = alpha_hw_plan_create(
                params_base.model.path.c_str(), -1);  // -1 = auto

            if (hw_dense.n_expert <= 1) {
                bool applied = false;

                // An explicit --ctx-size is a hard requirement, not a hint: honour it up to the model's
                // training length and free VRAM (drop GPU layers) to hold its KV instead of clamping the
                // context. For sliding-window (iswa) models only the few global attention layers' KV
                // scales with n_ctx; the sliding-window layers cost a fixed window.
                {
                    const int user_ctx = params_base.n_ctx;
                    const int native   = hw_dense.n_ctx_train > 0 ? hw_dense.n_ctx_train : user_ctx;
                    if (user_ctx > 0 && user_ctx > (int) hw_dense.n_ctx && user_ctx <= native &&
                        hw_dense.n_layer > 0 && hw_dense.kv_f16_bpt_layer > 0 &&
                        hw_dense.gpu_vram_total_mb > 0 && hw_dense.model_size_mb > 0) {
                        // common_context_params_to_llama reserves +1 seq for KV-chunk ops and partitions
                        // n_ctx evenly across n_seq_max=n_parallel+1, so to give the real slot the full
                        // requested per-slot context we must allocate that many extra tokens.
                        const int    np        = params_base.n_parallel > 0 ? params_base.n_parallel : 1;
                        const long long total_ctx = ((long long) user_ctx * (np + 1) + np - 1) / np;
                        // Only the ctx-scaling layers pay per-token KV. For iswa that's the global layers
                        // (few); the sliding-window layers cost a fixed window. Global count isn't in GGUF
                        // metadata cleanly, so estimate n_layer/6 (Gemma-ish) - override with
                        // ALPHA_KV_GLOBAL_LAYERS for a model with fewer/more (e.g. =1 lets Gemma-4 reach 256K).
                        const char * ge = getenv("ALPHA_KV_GLOBAL_LAYERS");
                        const int    ctx_layers = ge && atoi(ge) > 0 ? atoi(ge)
                                                : (hw_dense.sliding_window > 0 ? std::max(1, hw_dense.n_layer / 6)
                                                                              : hw_dense.n_layer);
                        // KV at Q4 (forced below) = F16 bytes/4. Scaling KV over total_ctx; SWA layers add a
                        // fixed window cost (all layers, bounded to the window).
                        const double q4 = 0.25;
                        const size_t kv_mb  = (size_t) ((double) hw_dense.kv_f16_bpt_layer * ctx_layers * q4
                                                        * (double) total_ctx / (1024.0 * 1024.0));
                        const size_t swa_mb = hw_dense.sliding_window > 0
                            ? (size_t) ((double) hw_dense.kv_f16_bpt_layer * hw_dense.n_layer * q4
                                        * (double) hw_dense.sliding_window / (1024.0 * 1024.0)) : 0;
                        // Compute/activation buffers scale with context (~n_embd*3 bytes/token). Reserve
                        // them so a long context fails cleanly (fewer layers) rather than OOM-crashing on
                        // the downstream graph allocation.
                        const size_t compute_mb   = (size_t) ((double) total_ctx * hw_dense.n_embd * 3.0
                                                              / (1024.0 * 1024.0));
                        const size_t overhead_mb  = 1024 + compute_mb;   // CUDA context + compute buffers
                        const size_t per_layer_mb = std::max<size_t>(1, hw_dense.model_size_mb / hw_dense.n_layer);
                        const long   avail_mb     = (long) hw_dense.gpu_vram_total_mb
                                                  - (long) kv_mb - (long) swa_mb - (long) overhead_mb;
                        if (avail_mb > (long) per_layer_mb) {
                            int ngl_fit = (int) (avail_mb / (long) per_layer_mb);
                            if (ngl_fit > hw_dense.n_layer) ngl_fit = hw_dense.n_layer;
                            SRV_INF("Dense auto-fit [context-priority]: honour -c %d (native %d; alloc n_ctx=%lld "
                                    "across %d slot+1 seq) — KV~%zuMB (Q4, %d ctx-scaling of %d layers) + "
                                    "SWA~%zuMB → ngl %d (planner had %d), freeing VRAM for the requested context\n",
                                    user_ctx, native, total_ctx, np, kv_mb, ctx_layers, hw_dense.n_layer, swa_mb,
                                    ngl_fit, hw_dense.n_gpu_layers);
                            hw_dense.n_gpu_layers = ngl_fit;            // fewer layers on GPU -> room for KV
                            hw_dense.n_ctx        = (int) total_ctx;    // >= params_base.n_ctx -> clamp below is a no-op
                            params_base.n_ctx     = (int) total_ctx;    // de-halved so per-slot == user's -c
                            hw_dense.kv_type_k    = GGML_TYPE_Q4_0;
                            hw_dense.kv_type_v    = GGML_TYPE_Q4_0;
                        } else {
                            SRV_WRN("Dense auto-fit [context-priority]: -c %d needs ~%zuMB KV+SWA+overhead, "
                                    "exceeds %zuMB VRAM even at ngl=0 — cannot honour it; planner's smaller "
                                    "context applies. Use a smaller -c, ALPHA_KV_GLOBAL_LAYERS, or a bigger GPU.\n",
                                    user_ctx, kv_mb + swa_mb + overhead_mb, hw_dense.gpu_vram_total_mb);
                        }
                    }
                }

                // If planner chose CPU-only but model almost fits, override to partial offload.
                // This handles models like Gemma 12B (6.7GB) on 8GB where the model fits
                // but KV+compute don't - partial offload at ~90% layers is much faster than CPU.
                if (hw_dense.n_gpu_layers == 0 &&
                    hw_dense.model_size_mb > 0 &&
                    hw_dense.gpu_vram_total_mb > 0 &&
                    hw_dense.model_size_mb < hw_dense.gpu_vram_total_mb) {
                    // Model fits in VRAM but planner chose CO - override to partial offload
                    int safe_layers = hw_dense.n_layer > 0
                        ? (int)(hw_dense.n_layer * hw_dense.gpu_vram_total_mb * 75 / 100 / hw_dense.model_size_mb)
                        : 0;
                    if (safe_layers > 0 && safe_layers <= hw_dense.n_layer) {
                        SRV_INF("Dense auto-fit: planner chose CO but model fits — override to PO ngl=%d/%d\n",
                            safe_layers, hw_dense.n_layer);
                        hw_dense.n_gpu_layers = safe_layers;
                        hw_dense.kv_type_k = GGML_TYPE_Q4_0;
                        hw_dense.kv_type_v = GGML_TYPE_Q4_0;
                        hw_dense.n_ctx = 2048;
                    }
                }

                // Reduce ngl if planner says model doesn't fully fit
                int auto_ngl = hw_dense.n_gpu_layers;

                // The planner sizes ngl for one sequence; its fixed VRAM reserve does not cover the
                // per-slot cost of --parallel N (recurrent state per seq on hybrid models, a compute buffer
                // that grows with n_seqs, per-slot sampling/output buffers), so at np > 1 its pick can OOM.
                // Pull ngl down by the per-slot VRAM: ALPHA_PARALLEL_SLOT_MB, default 320.
                {
                    const int np = params_base.n_parallel;
                    if (np > 1 && auto_ngl > 0 && hw_dense.n_layer > 0 && hw_dense.model_size_mb > 0) {
                        const size_t per_layer_mb = std::max<size_t>(1, (size_t) hw_dense.model_size_mb / hw_dense.n_layer);
                        const size_t per_slot_mb  = getenv("ALPHA_PARALLEL_SLOT_MB") ? (size_t) atoi(getenv("ALPHA_PARALLEL_SLOT_MB")) : 320;
                        const size_t extra_mb     = (size_t) (np - 1) * per_slot_mb;
                        const int    reduce       = (int) (extra_mb / per_layer_mb);
                        if (reduce > 0) {
                            const int adj = std::max(1, auto_ngl - reduce);
                            SRV_INF("Dense auto-fit: --parallel %d reserves ~%zuMB for %d extra slots → ngl %d → %d\n",
                                np, extra_mb, np - 1, auto_ngl, adj);
                            auto_ngl = adj;
                        }
                    }
                }

                // Opt-in VRAM packing (ALPHA_NGL_BOOST=N, default 0): the planner keeps a conservative
                // reserve and its per-layer estimate (model_mb/n_layer) is padded by the non-repeating
                // embd/output tensors, so partial offload can leave VRAM idle. N packs N more layers onto
                // the GPU.
                if (auto_ngl > 0 && getenv("ALPHA_NGL_BOOST")) {
                    const int nb  = atoi(getenv("ALPHA_NGL_BOOST"));
                    const int nlt = hw_dense.n_layer > 0 ? hw_dense.n_layer : auto_ngl + nb;
                    if (nb > 0) {
                        const int boosted = std::min(auto_ngl + nb, nlt);
                        SRV_INF("Dense auto-fit: ALPHA_NGL_BOOST=%d packs ngl %d → %d (pack idle VRAM)\n",
                            nb, auto_ngl, boosted);
                        auto_ngl = boosted;
                    }
                }
                if (auto_ngl >= 0 && auto_ngl < params_base.n_gpu_layers) {
                    SRV_INF("Dense auto-fit: ngl %d → %d (model %zuMB, VRAM %zuMB, strategy=%s)\n",
                        params_base.n_gpu_layers, auto_ngl,
                        hw_dense.model_size_mb, hw_dense.gpu_vram_total_mb,
                        hw_dense.strategy_name);
                    params_base.n_gpu_layers = auto_ngl;
                    applied = true;
                }
                // Apply KV quantization if planner recommends it (tight VRAM)
                if (hw_dense.kv_type_k != GGML_TYPE_F16) {
                    SRV_INF("Dense auto-fit: KV cache → Q4_0 (strategy=%s)\n",
                        hw_dense.strategy_name);
                    params_base.cache_type_k = hw_dense.kv_type_k;
                    params_base.cache_type_v = hw_dense.kv_type_v;
                    applied = true;
                }
                // Apply context reduction if needed
                if (hw_dense.n_ctx > 0 && hw_dense.n_ctx < params_base.n_ctx) {
                    SRV_INF("Dense auto-fit: ctx %d → %d\n",
                        params_base.n_ctx, (int)hw_dense.n_ctx);
                    params_base.n_ctx = hw_dense.n_ctx;
                    applied = true;
                }
                // Disable llama_params_fit if we applied any overrides
                if (applied) {
                    params_base.fit_params = false;
                }
            }
        }

        if (moe_direct_load) {
            // Step 1: Load model only (no context yet - preserves VRAM for expert cache)
            auto mparams = common_model_params_to_llama(params_base);
            model = llama_model_load_from_file(params_base.model.path.c_str(), mparams);
            if (!model) {
                SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
                fprintf(stderr, "\nFailed to load model. Check that:\n");
                fprintf(stderr, "  1. The file path is correct: %s\n", params_base.model.path.c_str());
                fprintf(stderr, "  2. The file is a valid GGUF (not corrupted or partially downloaded)\n");
                fprintf(stderr, "  3. You have enough RAM to load the model\n");
                fprintf(stderr, "Download models from huggingface.co in GGUF format.\n\n");
                return false;
            }
            SRV_INF("%s", "model loaded (MoE direct path)\n");

            // Step 2: Allocate expert cache now (full VRAM available, no context buffers yet)
            // This is done early - the actual optimizer setup happens below in the normal path.
            // We just need the cache allocated before context eats VRAM.
            {
                alpha_hw_plan hw_early = alpha_hw_plan_create(params_base.model.path.c_str(), -1);
                alpha_model_family fam = alpha_moe_detect_family(
                    hw_early.arch, hw_early.n_expert, hw_early.n_expert_used, hw_early.model_size_mb);
                alpha_portfolio_context pc = {};
                pc.family = fam;
                pc.n_expert = hw_early.n_expert;
                pc.n_expert_used = hw_early.n_expert_used;
                pc.K_estimate = hw_early.K_estimate;
                pc.K_min = hw_early.K_min;
                pc.K_draft = hw_early.K_draft;
                pc.fit_ratio = hw_early.model_size_mb > 0 ? (float)hw_early.model_size_mb / std::max(hw_early.gpu_vram_mb, (size_t)1) : 1.0f;
                pc.gpu_vram_mb = hw_early.gpu_vram_mb;
                pc.model_size_mb = hw_early.model_size_mb;
                alpha_moe_profile port = alpha_moe_portfolio_select_v2(&pc);

                // --mode override: inference mode controls portfolio behavior
                switch (params_base.alpha_mode) {
                    case ALPHA_MODE_AUTO:
                        // Let portfolio decide - don't override callback/gen/warmup
                        SRV_INF("mode: auto (portfolio: cb=%d gen=%d warmup=%d)\n",
                                port.callback, port.gen, port.warmup);
                        break;
                    case ALPHA_MODE_STANDARD:
                        port.callback = ALPHA_CB_NONE;
                        port.gen = ALPHA_GEN_STANDARD;
                        port.warmup = ALPHA_WARMUP_NONE;
                        port.cpu_moe = false;
                        SRV_INF("%s", "mode: standard (expert cache disabled)\n");
                        break;
                    case ALPHA_MODE_CPU_MOE:
                        port.callback = ALPHA_CB_NONE;
                        port.gen = ALPHA_GEN_STANDARD;
                        port.warmup = ALPHA_WARMUP_NONE;
                        port.cpu_moe = true;
                        SRV_INF("%s", "mode: offload (MoE experts on CPU)\n");
                        break;
                }

                // Only allocate expert cache if portfolio wants a callback (LUT/reactive).
                // For cpu-moe standard (cb=none, gen=STANDARD): experts computed on CPU
                // by ggml directly, no GPU cache needed. Skipping cache saves VRAM for backbone.
                bool needs_cache = (port.callback != ALPHA_CB_NONE);
                if (needs_cache) {
                    alpha_moe_optimizer_config cfg = {};
                    cfg.vram_budget_mb = port.expert_cache_max > 0 ? port.expert_cache_max : 99999;
                    cfg.flat_cache = port.flat_cache;
                    cfg.debug = false;
                    cfg.expert_sticky = 0.3f;      // sticky routing momentum
                    cfg.expert_layer_bias = 0.2f;   // cross-layer consistency
                    moe_opt = alpha_moe_optimizer_init(model, nullptr, cfg);
                    if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
                        alpha_moe_optimizer_get_n_moe_layers(moe_opt) > 0) {
                        alpha_moe_optimizer_alloc_cache(moe_opt);
                        // Load expert profiles (for warmup classify - picks hot experts per style)
                        std::string mdir = std::filesystem::path(params_base.model.path).parent_path().string();
                        alpha_moe_optimizer_load_style_profiles(moe_opt, mdir.c_str());
                        SRV_INF("%s", "expert cache allocated + profiles loaded before context creation\n");
                    }
                } else {
                    SRV_INF("cpu-moe: experts computed on CPU (no GPU cache), %d experts\n", hw_early.n_expert);
                }
                moe_portfolio = port;
                // Serve portfolio adjustments:
                // - Always use classify warmup (profile-based expert loading)
                // - Only force LUT_DIRECT gen when K >= K_min (good coverage).
                //   When K < K_min, respect portfolio's CPU_MOE gen - LUT with low K
                //   produces garbage (echoed input, garbled text).
                if (moe_portfolio.warmup == ALPHA_WARMUP_SMART) {
                    moe_portfolio.warmup = ALPHA_WARMUP_CLASSIFY;
                }
                if (moe_portfolio.gen != ALPHA_GEN_LUT_DIRECT &&
                    hw_early.K_estimate >= hw_early.K_min &&
                    hw_early.K_estimate < hw_early.n_expert) {
                    moe_portfolio.gen = ALPHA_GEN_LUT_DIRECT;
                }
                moe_K_estimate = hw_early.K_estimate;
                moe_gpu_vram_mb = hw_early.gpu_vram_mb;
                if (hw_early.K_estimate > 0 && hw_early.K_estimate < hw_early.K_min) {
                    fprintf(stderr, "\n");
                    fprintf(stderr, "[WARNING] Expert cache undersized: K=%d < K_min=%d\n",
                            hw_early.K_estimate, hw_early.K_min);
                    fprintf(stderr, "[WARNING] Quality may degrade. Use cpu-moe prefill for reliable output.\n");
                    fprintf(stderr, "[WARNING] For best quality, use a GPU with more VRAM (need K >= %d).\n\n",
                            hw_early.K_min);
                }
            }

            // Disable graph reuse for expert cache models.
            // Expert cache swaps tensor dimensions (ne[2]=K, not n_expert); graph reuse
            // with stale dimensions crashes. Also needed so argsort tensor buffers persist
            // for post-decode readback (reload_from_argsort / batch verification).
            if (moe_direct_load) {
#ifdef _WIN32
                _putenv_s("LLAMA_GRAPH_REUSE_DISABLE", "1");
#else
                setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
#endif
            }

            // Step 3: Create context (KV cache, compute buffers use remaining VRAM)
            auto cparams = common_context_params_to_llama(params_base);
            ctx = llama_init_from_model(model, cparams);
            if (!ctx) {
                SRV_ERR("%s", "failed to create context\n");
                llama_model_free(model);
                model = nullptr;
                return false;
            }
        } else {
            llama_init = common_init_from_params(params_base);
            model = llama_init->model();
            ctx   = llama_init->context();
        }

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            fprintf(stderr, "\nFailed to load model. Check that:\n");
            fprintf(stderr, "  1. The file path is correct: %s\n", params_base.model.path.c_str());
            fprintf(stderr, "  2. The file is a valid GGUF (not corrupted or partially downloaded)\n");
            fprintf(stderr, "  3. You have enough RAM to load the model\n");
            fprintf(stderr, "Download models from huggingface.co in GGUF format.\n\n");
            return false;
        }

        // Self-spec layer-skip: single-context approach.
        // Draft decodes switch n_layer_draft on the same context - graph rebuilds
        // automatically because allow_reuse checks cparams.n_layer_draft.
        // No separate draft context needed.
        if (params_base.speculative.self_spec_skip > 0 && model) {
            // Layer-skip self-spec (middle-band): drafts M tokens skipping the offloaded CPU band
            // (keeping the output layers), then rolls back via a deep full-seq checkpoint before the
            // verify. The deep snapshot/restore (llama_state_seq_get/set_data_ext) restores recurrent
            // (DeltaNet) state correctly - unlike seq_cp's shallow share - so this is hybrid-safe.
            const int n_layer  = llama_model_n_layer(model);
            const int ds       = params_base.speculative.self_spec_skip;
            const int out_keep = getenv("ALPHA_SPEC_OUT_KEEP") ? atoi(getenv("ALPHA_SPEC_OUT_KEEP")) : 6;
            SRV_INF("self-spec: middle-band skip %d layers [%d,%d), keep last %d of %d%s\n",
                    ds, n_layer - out_keep - ds, n_layer - out_keep, out_keep, n_layer,
                    llama_model_is_hybrid(model) ? " (hybrid: recurrent GPU-snapshot rollback)" : "");
            // Pre-allocate the GPU recurrent-snapshot scratch now, while VRAM headroom is still free.
            // If left lazy, the first draft mid-request fails to allocate (the CUDA pool has claimed
            // the headroom for compute buffers). No-op for non-recurrent memory. If the scratch does not
            // fit, disable self-spec rather than crash.
            if (llama_model_is_hybrid(model) && ctx) {
                try {
                    llama_memory_gpu_snapshot(llama_get_memory(ctx));
                    SRV_INF("%s", "self-spec: recurrent GPU-snapshot scratch pre-allocated\n");
                } catch (const std::exception & e) {
                    SRV_INF("self-spec: DISABLED — no VRAM for recurrent-snapshot scratch (%s). "
                            "Needs a box with headroom (smaller quant / bigger GPU). Running plain decode.\n", e.what());
                    params_base.speculative.self_spec_skip = 0;
                }
            }
        }

        vocab = llama_model_get_vocab(model);

        // Expert-usage logging (ALPHA_LOG_EXPERT_USAGE): passive per-generation router-selection log.
        // Auto-detect expert counts from the model file (generic - 8-expert Mixtral, 256-expert
        // DeepSeek-V4, any MoE; dense -> inert). Enable argsort persist so the selected_experts buffers
        // survive decode for a read-only tap. Byte-identical tokens flag-on vs off (nothing perturbed).
        if (getenv("ALPHA_LOG_EXPERT_USAGE")) {
            alpha_hw_plan hwp = alpha_hw_plan_create(params_base.model.path.c_str(), -1);
            eu_n_expert_total = hwp.n_expert;
            eu_n_expert_used  = hwp.n_expert_used;
            if (eu_n_expert_total > 1) {
                eu_log_enabled = true;
                const char * lv = getenv("ALPHA_LOG_EXPERT_USAGE");
                eu_log_verbose = (lv && lv[0] == '2');
                alpha_mole_set_argsort_persist(true);
                const char * p = getenv("ALPHA_LOG_EXPERT_USAGE_PATH");
                std::string path = (p && *p) ? p : "expert_usage.jsonl";
                eu_log_fp = fopen(path.c_str(), "a");
                SRV_INF("expert-usage logging ON: n_expert=%d n_expert_used=%d → %s (passive, byte-identical)\n",
                        eu_n_expert_total, eu_n_expert_used, path.c_str());
            } else {
                SRV_INF("%s", "expert-usage logging requested but model is dense (n_expert<=1) — inert\n");
            }
        }

        // Branch-and-recover decode (ALPHA_BRANCH_RECOVER): greedy-by-default sampler with sparse,
        // entropy-gated nucleus branches. Load knobs once. Default off => br_sample never runs (identical).
        if (const char * v = getenv("ALPHA_BRANCH_RECOVER"); v && v[0] && v[0] != '0') {
            br_enabled = true;
            auto envf = [](const char * k, float d){ const char * e = getenv(k); return (e && *e) ? (float)atof(e) : d; };
            auto envi = [](const char * k, int   d){ const char * e = getenv(k); return (e && *e) ? atoi(e) : d; };
            br_entropy_min = envf("ALPHA_BR_ENTROPY_MIN", 1.5f);
            br_recover_n   = envi("ALPHA_BR_RECOVER",     6);
            br_topp        = envf("ALPHA_BR_TOPP",        0.95f);
            br_max_branch  = envi("ALPHA_BR_MAX_BRANCHES", 8);
            br_seed        = (uint64_t) strtoull(getenv("ALPHA_BR_SEED") ? getenv("ALPHA_BR_SEED") : "0", nullptr, 10);
            SRV_INF("branch-and-recover decode ON: H_min=%.2f recover=%d top_p=%.2f max_branch=%d seed=%llu "
                    "(greedy default + sparse nucleus branches; seeded ⇒ reproducible)\n",
                    br_entropy_min, br_recover_n, br_topp, br_max_branch, (unsigned long long) br_seed);
        }


        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_vocab_get_add_bos(vocab);

#ifdef ALPHA_DELTANET
        // Load initial DeltaNet state before prescan (for parallel prescan verification)
        if (!params_base.deltanet_state_init_path.empty()) {
            uint64_t init_tokens = alpha_deltanet_state_load(params_base.deltanet_state_init_path, ctx);
            if (init_tokens > 0) {
                SRV_INF("DeltaNet: loaded initial state (%llu tokens) before prescan\n",
                        (unsigned long long)init_tokens);
            } else {
                SRV_ERR("DeltaNet: failed to load initial state from %s\n",
                        params_base.deltanet_state_init_path.c_str());
            }
        }

        // DeltaNet pre-scan: process chunks to build recurrent state
        // Digest mode: build NER digests (~50 tokens/chunk) instead of raw text
        if (!params_base.deltanet_prescan_dir.empty() &&
            chunk_bm25.N > 0 && params_base.corpus_dir == params_base.deltanet_prescan_dir) {
            // Digest prescan: BM25+NER already built, use digests for faster+better prescan
            SRV_INF("DeltaNet digest pre-scan: %d chunks (IDF+NER+relations)\n", chunk_bm25.N);
            auto t_prescan = std::chrono::steady_clock::now();

            // Build concatenated digest text for all chunks
            std::string all_digests;
            all_digests.reserve(chunk_bm25.N * 400);  // ~300 chars per digest

            for (int i = 0; i < chunk_bm25.N; i++) {
                std::string text = chunk_bm25.get_text(i);
                auto entities = entity_extractor.extract(text);
                auto imports = entity_extractor.extract_imports(text);

                std::string digest = build_chunk_digest(
                    chunk_bm25, i, entities, imports, text, false, "");
                all_digests += digest + "\n";
            }

            // Write digest file for streaming prescan
            std::string digest_path = params_base.deltanet_state_path + ".digest.tmp";
            {
                std::ofstream f(digest_path);
                f << all_digests;
            }
            SRV_INF("DeltaNet digest: %d chunks → %.1f MB digest file\n",
                    chunk_bm25.N, all_digests.size() / 1e6);

            // Run standard streaming prescan on digest file
            alpha_deltanet_prescan(ctx, digest_path, params_base.deltanet_state_path);

            // Clean up temp file
            std::filesystem::remove(digest_path);

            auto prescan_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_prescan).count();
            SRV_INF("DeltaNet digest pre-scan complete in %.1fs\n", prescan_ms / 1000.0f);

        } else if (!params_base.deltanet_prescan_dir.empty()) {
            // Raw prescan (original mode): reads first 511 bytes per chunk
            SRV_INF("DeltaNet raw pre-scan: processing chunks in %s\n",
                    params_base.deltanet_prescan_dir.c_str());
            alpha_deltanet_prescan(ctx, params_base.deltanet_prescan_dir,
                                   params_base.deltanet_state_path);
        }
        // Load pre-scanned DeltaNet state if specified (and not just generated)
        else if (!params_base.deltanet_state_path.empty()) {
            uint64_t tokens = alpha_deltanet_state_load(params_base.deltanet_state_path, ctx);
            if (tokens > 0) {
                SRV_INF("DeltaNet pre-scan state loaded: %llu tokens from %s\n",
                        (unsigned long long)tokens, params_base.deltanet_state_path.c_str());
            } else {
                SRV_ERR("Failed to load DeltaNet state from %s\n",
                        params_base.deltanet_state_path.c_str());
            }
        }
#endif

        // KV-chunk extended context: auto-select routing strategy from model size
        {
            uint64_t n_params = llama_model_n_params(model);
            routing_strategy = select_routing_strategy(n_params);
            n_ctx_extended = params_base.n_ctx_extended;
            if (params_base.ctx_mode == ALPHA_CTX_EXTENDED_CODE) {
                SRV_INF("%s", "context mode: extended-code (NER entity linking + multihop enabled)\n");
            }
            if (params_base.ctx_mode == ALPHA_CTX_COMPREHEND) {
                if (!params_base.pyramid_file.empty()) {
                    try {
                        std::ifstream pf(params_base.pyramid_file);
                        comprehend_pyramid = json::parse(pf);
                        comprehend_loaded = comprehend_pyramid.contains("nodes") &&
                                            comprehend_pyramid.contains("root") &&
                                            comprehend_pyramid.contains("levels");
                        if (comprehend_loaded) {
                            SRV_INF("context mode: comprehend — loaded pyramid %s (%zu nodes); "
                                    "Method-2 iterative deepening active\n",
                                    params_base.pyramid_file.c_str(),
                                    (size_t)comprehend_pyramid["nodes"].size());
                        } else {
                            SRV_INF("context mode: comprehend — %s missing nodes/root/levels; "
                                    "falling back to BM25 retrieval\n", params_base.pyramid_file.c_str());
                        }
                    } catch (const std::exception & e) {
                        SRV_INF("context mode: comprehend — failed to load pyramid %s (%s); "
                                "falling back to BM25\n", params_base.pyramid_file.c_str(), e.what());
                    }
                } else {
                    SRV_INF("%s", "context mode: comprehend — no --pyramid-file given; serving via BM25 "
                                  "(build a pyramid offline with alpha-intelligent hierarchical_summary.py)\n");
                }
            }
            if (n_ctx_extended > 0) {
                // Auto-compute prefill parallelism from VRAM
                alpha_hw_plan hw_prefill = alpha_hw_plan_create(
                    params_base.model.path.c_str(), -1);
                int prefill_parallel = alpha_hw_compute_prefill_parallel(&hw_prefill);
                SRV_INF("KV-chunk extended context: %d tokens (routing: %s, prefill: %d-way)\n",
                        n_ctx_extended,
                        routing_strategy == kv_chunk_routing::BM25_EXPAND
                            ? "bm25+expand" : "bm25_only",
                        prefill_parallel);
            }
        }

        // Alpha-MoLE: Initialize MoE optimizer for expert cache + LUT-direct
        // Skip init if already done via moe_direct_load path (expert cache allocated early)
        {
            alpha_hw_plan hw = alpha_hw_plan_create(params_base.model.path.c_str(), -1);
            if (!moe_opt && hw.n_expert > 1 && hw.n_expert_used > 0 &&
                params_base.alpha_mode != ALPHA_MODE_STANDARD) {
                alpha_model_family family = alpha_moe_detect_family(
                    hw.arch, hw.n_expert, hw.n_expert_used, hw.model_size_mb);

                alpha_portfolio_context pctx = {};
                pctx.family = family;
                pctx.n_expert = hw.n_expert;
                pctx.n_expert_used = hw.n_expert_used;
                pctx.K_estimate = hw.K_estimate;
                pctx.K_min = hw.K_min;
                pctx.K_draft = hw.K_draft;
                pctx.fit_ratio = hw.model_size_mb > 0 ? (float)hw.model_size_mb / std::max(hw.gpu_vram_mb, (size_t)1) : 1.0f;
                pctx.gpu_vram_mb = hw.gpu_vram_mb;
                pctx.model_size_mb = hw.model_size_mb;
                pctx.is_hybrid = false;
                moe_portfolio = alpha_moe_portfolio_select_v2(&pctx);

                alpha_moe_optimizer_config cfg = {};
                // Expert cache budget: use portfolio's recommendation (99999 = all available VRAM).
                // The portfolio computes the right budget even for cpu-moe strategy.
                // Fall back to pre-load planner estimate, then post-load estimate.
                if (moe_portfolio.expert_cache_max > 0)
                    cfg.vram_budget_mb = moe_portfolio.expert_cache_max;
                else if (moe_expert_cache_mb_pre > 0)
                    cfg.vram_budget_mb = moe_expert_cache_mb_pre;
                else
                    cfg.vram_budget_mb = hw.expert_cache_mb;
                cfg.flat_cache = moe_portfolio.flat_cache;
                cfg.enable_profiling = (moe_portfolio.warmup == ALPHA_WARMUP_CLASSIFY);
                cfg.debug = false;
                cfg.expert_sticky = 0.3f;      // sticky routing momentum
                cfg.expert_layer_bias = 0.2f;   // cross-layer consistency
                // Pass ctx=NULL like CLI does - allows expert cache to allocate
                // with full VRAM (context compute buffers not yet claiming VRAM).
                // Callback is set on ctx separately after allocation.
                if (moe_opt) {
                    alpha_moe_optimizer_free(moe_opt);
                    moe_opt = nullptr;
                }
                moe_opt = alpha_moe_optimizer_init(model, nullptr, cfg);

                if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
                    alpha_moe_optimizer_get_n_moe_layers(moe_opt) > 0) {
                    // Give the optimizer the context so apply_warmup/apply_plan can
                    // manage the eval callback lifecycle (install/uninstall by K).
                    alpha_moe_optimizer_set_context(moe_opt, ctx);
                    // Install the eval callback only for reactive mode (active from
                    // token 0). For ALPHA_CB_LUT_DISABLED, do not install here: an
                    // installed-but-passive callback still forces the scheduler into
                    // per-node evaluation with a per-split synchronize and blocks CUDA
                    // graph capture (ggml-backend.cpp branches on pointer presence,
                    // not the active flag). apply_plan step 5 installs it when it is
                    // actually needed (low-K warmup-LUT speculation) and nulls it for
                    // high-K/ALL_GPU where it is pure overhead.
                    if (moe_portfolio.callback == ALPHA_CB_REACTIVE) {
                        llama_set_eval_callback(ctx,
                            alpha_moe_optimizer_get_eval_callback(),
                            alpha_moe_optimizer_get_eval_data(moe_opt));
                        alpha_moe_optimizer_set_callback_active(moe_opt, true);
                    }
                    if (!moe_direct_load) {
                        // Only alloc cache here for non-direct path (direct already did it)
                        alpha_moe_optimizer_alloc_cache(moe_opt);
                    }
                    if (moe_portfolio.lut_init) {
                        alpha_moe_optimizer_init_spec_lut(moe_opt);
                    }
                    moe_lut_active = (moe_portfolio.gen == ALPHA_GEN_LUT_DIRECT);
                    SRV_INF("MoE optimizer initialized: %s mode, %d experts\n",
                            moe_lut_active ? "optimized" : "standard",
                            hw.n_expert);
                } else {
                    // All experts on GPU (direct) or not MoE - no optimizer needed.
                    // Optimizer callback adds ~4.5x decode overhead on GPU-fit models.
                    if (moe_opt) { alpha_moe_optimizer_free(moe_opt); moe_opt = nullptr; }
                }
            }
        }

        // Direct-load MoE: set context on optimizer (was created with ctx=nullptr for
        // early cache allocation). Without this, optimizer can't read argsort, reinstall
        // callback, or do reload_from_argsort - the whole quality pipeline breaks.
        if (moe_direct_load && moe_opt) {
            alpha_moe_optimizer_set_context(moe_opt, ctx);
        }

        // Direct-load MoE: set callback on ctx but keep inactive for first prompt.
        // CLI flow: disable cache for prompt -> decode prompt with full CPU experts ->
        // warmup_classify fills cache with correct hot experts -> enable LUT for generation.
        // Serve equivalent: install callback inactive, disable cache. First request's
        // apply_warmup will enable cache + LUT after routing data is available.
        if (moe_direct_load && moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
            alpha_moe_optimizer_get_n_moe_layers(moe_opt) > 0) {
            if (moe_portfolio.callback == ALPHA_CB_REACTIVE) {
                llama_set_eval_callback(ctx,
                    alpha_moe_optimizer_get_eval_callback(),
                    alpha_moe_optimizer_get_eval_data(moe_opt));
                // Start inactive - first prompt must process with full CPU expert access
                // (not stale K=34 cache). apply_warmup activates after prompt routing data.
                alpha_moe_optimizer_set_callback_active(moe_opt, false);
            }
            // ALPHA_CB_LUT_DISABLED: no eager install (see standard path above) - the
            // first request's apply_warmup step 5 installs it via reinstall_callback
            // when K is low enough to need warmup-LUT speculation. This keeps CUDA
            // graphs + full-split compute for the first prompt and for high-K configs.
            // Disable cache for first prompt (same as CLI's disable_cache_for_prompt)
            // This ensures prompt decode uses all experts from CPU, not stale cache.
            if (moe_portfolio.disable_cache_for_prompt) {
                alpha_moe_optimizer_disable_cache(moe_opt);
            }
            if (moe_portfolio.lut_init) {
                alpha_moe_optimizer_init_spec_lut(moe_opt);
            }
            moe_lut_active = (moe_portfolio.gen == ALPHA_GEN_LUT_DIRECT);
            SRV_INF("%s", "MoE direct-load: cache disabled for first prompt, LUT ready for gen\n");
        }

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to common/speculative.cpp?
            SRV_INF("loading draft model '%s'\n", params_base.speculative.mparams_dft.path.c_str());

            const auto & params_spec = params_base.speculative;

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams_dft;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model);

        int n_ctx_slot = llama_n_ctx_seq(ctx);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        const auto ctx_seq_rm_type = common_context_can_seq_rm(ctx);
        if (ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id    = i;
            slot.ctx   = ctx;
            slot.n_ctx = n_ctx_slot;

            slot.ctx_seq_rm_type = ctx_seq_rm_type;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            // Enable two-pass for expert-cached MoE models
            slot.twopass_enabled = (moe_opt != nullptr &&
                                    alpha_moe_optimizer_is_moe(moe_opt) &&
                                    moe_lut_active);

            // try speculative decoding
            if (ctx_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx));

                if (slot.spec) {
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                }
            }

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("slots debug = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        // MoE expert cache needs VRAM - disable prompt cache for MoE models
        // unless user explicitly set a limit. Each cached prompt saves ~188 MiB
        // of KV+recurrent state, which quickly exhausts VRAM alongside expert cache.
        if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
            alpha_moe_optimizer_get_K(moe_opt) > 0 &&
            params_base.cache_ram_mib == 8192) {  // only override default
            params_base.cache_ram_mib = 0;
            SRV_WRN("%s", "prompt cache disabled for MoE model (expert cache needs VRAM)\n");
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_WRN("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);

            // Initialize RadixAttention tree (uses half the prompt cache budget)
            size_t radix_budget = (size_t)params_base.cache_ram_mib * 512ULL * 1024ULL;  // half of cache_ram
            radix_kv_cache.init(radix_budget);
            SRV_INF("radix cache: %.0f MB budget for token-level prefix tree\n",
                    radix_budget / (1024.0 * 1024.0));
        } else {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_WRN("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(model != nullptr);
        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s: --cache-idle-slots requires --kv-unified, disabling\n", __func__);
                params_base.cache_idle_slots = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s: --cache-idle-slots requires --cache-ram, disabling\n", __func__);
                params_base.cache_idle_slots = false;
            } else {
                SRV_INF("%s: idle slots will be saved to prompt cache and cleared upon starting a new task\n", __func__);
                SRV_DBG("%s", "__TEST_TAG_CLEAR_IDLE_ENABLED__\n");
            }
        }

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.reasoning_budget,
                /* reasoning_budget_msg  */ params_base.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        // Load domain classifier model (0.6B LoRA) if specified
        if (!params_base.classify_model_path.empty()) {
            SRV_INF("loading domain classifier: %s\n", params_base.classify_model_path.c_str());
            llama_model_params cl_mparams = llama_model_default_params();
            cl_mparams.n_gpu_layers = 999;  // GPU: ~60ms vs ~3s CPU (336MB Q4_0)
            classify_model = llama_model_load_from_file(params_base.classify_model_path.c_str(), cl_mparams);
            if (classify_model) {
                llama_context_params cl_cparams = llama_context_default_params();
                cl_cparams.n_ctx = 2048;  // domain list + user prompt can be 500+ tokens
                cl_cparams.n_batch = 2048;
                cl_cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                classify_ctx = llama_init_from_model(classify_model, cl_cparams);
                classify_vocab = llama_model_get_vocab(classify_model);

                // Build system prompt with domain list
                classify_system_prompt = "You are a domain classifier. Given a user's prompt, "
                    "output ONLY the domain name from the following list. Output nothing else.\n\n"
                    "Domains: ";
                int n_dp = alpha_moe_optimizer_get_domain_profile_count();
                for (int d = 0; d < n_dp; d++) {
                    const char * name = alpha_moe_optimizer_get_domain_profile_name(d);
                    if (!name || strcmp(name, "classify") == 0 || strcmp(name, "grammar") == 0 ||
                        strcmp(name, "core") == 0) continue;
                    if (classify_system_prompt.back() != ' ') classify_system_prompt += ", ";
                    classify_system_prompt += name;
                }

                SRV_INF("domain classifier loaded (%d domains, ctx=%d)\n",
                        n_dp, (int)cl_cparams.n_ctx);
            } else {
                SRV_ERR("failed to load classifier: %s\n", params_base.classify_model_path.c_str());
            }
        }

        // Connect to Redis if --redis is set
#ifdef ALPHA_HAS_REDIS
        if (!params_base.redis_url.empty()) {
            redis = std::make_unique<redis_index>();
            // Parse host:port
            std::string host = "localhost";
            int port = 6379;
            auto colon = params_base.redis_url.find(':');
            if (colon != std::string::npos) {
                host = params_base.redis_url.substr(0, colon);
                port = std::stoi(params_base.redis_url.substr(colon + 1));
            } else {
                host = params_base.redis_url;
            }
            if (redis->connect(host, port)) {
                redis->create_indexes();
                use_redis = true;
                fprintf(stderr, "redis: connected to %s:%d\n", host.c_str(), port);
            } else {
                fprintf(stderr, "redis: connection failed, falling back to BM25\n");
                redis.reset();
            }
        }
#endif

#ifdef ALPHA_DENSE_L0
        // Dense-L0 semantic backend: load the bge-base GGUF embedder as a dedicated context.
        // Env-gated (no shared arg/common_params change) - set ALPHA_DENSE_L0_EMBED_MODEL=<gguf>.
        if (const char * dpath = getenv("ALPHA_DENSE_L0_EMBED_MODEL")) {
            if (dpath[0] && dense.load_embedder(dpath, 999)) {
                use_dense = true;
                dense.set_windows([] { const char * e = getenv("ALPHA_DENSE_L0_SW"); return e && atoi(e) != 0; }());   // sentence-window opt-in
                SRV_INF("dense-l0: semantic backend enabled (dim=%d, sentence-window=%s)\n",
                        dense.dim(), dense.windows_enabled() ? "on" : "off");
                // ALPHA_DENSE_L0_PERSIST=<path>: restore the vector store rather than re-embedding
                // the corpus. Deliberately after load_embedder, because load_vectors() validates the
                // file dim against the embedder's and refuses a mismatch instead of reinterpreting
                // it. A refusal is non-fatal: the corpus pass below re-embeds exactly as it does now.
                if (const char * ppath2 = getenv("ALPHA_DENSE_L0_PERSIST")) {
                    std::error_code fse; const bool exists = ppath2[0] && std::filesystem::exists(ppath2, fse);
                    if (exists && !dense.load_vectors(ppath2)) dense_persist_refused = true;
                    if (exists && !dense_persist_refused) {
                        SRV_INF("dense-l0: restored %zu vectors from %s -- skipping re-embed\n",
                                dense.size(), ppath2);
                        // IVF centroids are not persisted and load_vectors clears ann_active, so build the index here
                        // before serving; otherwise a restored store would be searched by linear scan.
                        dense.build_ann();
                    }
                }
            } else {
                SRV_ERR("dense-l0: failed to load embedder '%s' — semantic retrieval disabled\n", dpath);
            }
        }
#endif

        // Initialize 3-tier KV pool (L2: CPU RAM, L3: Redis)
        // Uses half the prompt cache budget for the KV pool tier
        if (params_base.cache_ram_mib > 0) {
            size_t kv_pool_budget_mb = std::max<size_t>(256, params_base.cache_ram_mib / 2);
            redisContext * redis_ctx = nullptr;
#ifdef ALPHA_HAS_REDIS
            if (redis && redis->is_connected()) {
                // Borrow the Redis connection for L3 tier
                // Note: redis_index owns the connection, kv_pool borrows it
                redis_ctx = nullptr;  // TODO: expose redis_index's redisContext for L3
            }
#endif
            kv_pool_cache.init(kv_pool_budget_mb, redis_ctx, 3600);
        }

        // Auto-ingest code corpus if --corpus-dir is set
        if (!params_base.corpus_dir.empty()) {
            if (params_base.ctx_mode != ALPHA_CTX_EXTENDED_CODE &&
                params_base.ctx_mode != ALPHA_CTX_AUTO &&
                params_base.ctx_mode != ALPHA_CTX_EXTENDED &&
                params_base.ctx_mode != ALPHA_CTX_UNIFIED) {
                SRV_ERR("%s", "--corpus-dir requires --context extended-code, extended, unified, or auto\n");
                return false;
            }
            corpus_auto_ingest(params_base.corpus_dir);
            if (params_base.ctx_mode == ALPHA_CTX_UNIFIED) {
                SRV_INF("unified index: %d entities, %d distinct values, %zu sections | tiered resident %zu B (%.0f B/rec), records.bin %zu B on disk\n",
                        unified.tier.n, unified.tier.vc_n, unified.sections.size(),
                        unified.tier.resident_bytes(),
                        unified.tier.n ? (double)unified.tier.resident_bytes()/unified.tier.n : 0.0,
                        unified.tier.mm_len);
            }

            // Auto-context classification: resolve AUTO to effective mode
            if (params_base.ctx_mode == ALPHA_CTX_AUTO) {
                int chunks_with_entities = 0;
                for (auto & [cid, ents] : chunk_entities.chunk_to_entities) {
                    if (!ents.empty()) chunks_with_entities++;
                }
                float entity_ratio = chunk_bm25.N > 0
                    ? (float)chunks_with_entities / chunk_bm25.N : 0.0f;

                if (entity_ratio >= 0.3f) {
                    effective_ctx_mode = ALPHA_CTX_EXTENDED_CODE;
                } else if (chunk_bm25.N > 0) {
                    effective_ctx_mode = ALPHA_CTX_EXTENDED;
                } else {
                    effective_ctx_mode = ALPHA_CTX_NATIVE;
                }
                SRV_INF("auto-context: classified as %s (entity_ratio=%.2f, %d chunks, %d with entities)\n",
                    effective_ctx_mode == ALPHA_CTX_EXTENDED_CODE ? "code" :
                    effective_ctx_mode == ALPHA_CTX_EXTENDED ? "document" : "native",
                    entity_ratio, chunk_bm25.N, chunks_with_entities);
            } else {
                effective_ctx_mode = params_base.ctx_mode;
            }
        } else {
            effective_ctx_mode = params_base.ctx_mode;
        }

        return true;
    }

    // Auto-ingest source files from a directory for extended-code mode
    void corpus_auto_ingest(const std::string & dir) {
        namespace fs = std::filesystem;
        static const std::unordered_set<std::string> source_exts = {
            ".php", ".py", ".js", ".ts", ".tsx", ".jsx",
            ".java", ".cs", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".swift", ".kt",
            ".txt", ".md", ".csv", ".json", ".log", ".xml",  // data files
        };
        static const std::unordered_set<std::string> skip_dirs = {
            ".git", "node_modules", "vendor", "build", "dist",
            "__pycache__", ".venv", "target", ".idea", ".vscode",
        };
        // Per-file size cap: larger files are skipped, not truncated. Override with ALPHA_MAX_FILE_KB.
        // Files are sub-chunked into 1KB BM25 docs downstream, so a high cap is safe.
        size_t max_file_size = 4 * 1024 * 1024;  // 4MB default
        if (const char * mfk = getenv("ALPHA_MAX_FILE_KB")) {
            long v = atol(mfk);
            if (v > 0) max_file_size = (size_t)v * 1024;
        }

        auto t0 = std::chrono::steady_clock::now();
        int chunk_id = 0;
        int n_files = 0;
        int n_experiences = 0;
        int n_articles = 0;   // files carrying a "[Article:]" boundary header

        // Back BM25 chunk text with a private mmap'd file so the ciphertext lives off-RAM instead of
        // holding the full text twice. It stays encrypted at rest.
        {
            const char * td = getenv("TMPDIR"); std::string tmpdir = (td && *td) ? td : "/tmp";
            chunk_bm25.open_text_store(tmpdir + "/alpha_bm25_" + std::to_string((long)getpid()) + ".text.bin");
            // Stream the metadata inverted-index postings to temp files too, so the resident maps are
            // not a build-time transient. ALPHA_META_SELFCHECK keeps the maps in parallel and verifies
            // streamed lookup == map for every key at finalize.
            bool self = getenv("ALPHA_META_SELFCHECK") != nullptr;
            chunk_metadata.enable_streaming(tmpdir + "/alpha_meta_" + std::to_string((long)getpid()), self);
            // External inversion: in inverted mode, spill BM25 postings to shard files during ingest so
            // the resident forward index is never accumulated. No-op unless ALPHA_BM25_INVERTED is set;
            // falls back to resident inversion if the spill file cannot be opened.
            if (bm25_index::bm25_inverted()) {
                chunk_bm25.open_postings_spill(tmpdir + "/alpha_bm25_" + std::to_string((long)getpid()) + ".spill");
            }
        }
        chunk_bm25.enable_df_streaming();   // don't build the resident df map; derive df from tf at finalize

        SRV_INF("auto-ingest: scanning %s\n", dir.c_str());

        // Parallel-ingest buffer. Sub-chunks accumulate here and are flushed in batches so the
        // per-doc work (ChaCha20 + tokenize + SipHash + sort) can use every core; the merge stays
        // ordered so the resulting index is byte-identical to the serial path.
        std::vector<std::pair<std::string, std::string>> pending_docs;
        int pending_first_id = 0;
        int ingest_batch_sz = 2048;                     // ~2048 x ~2KB = a few MB in flight
        if (const char * bs = getenv("ALPHA_INGEST_BATCH")) {
            int v = atoi(bs); if (v > 0) ingest_batch_sz = v;
        }
        pending_docs.reserve(ingest_batch_sz);
        SRV_INF("auto-ingest: parallel ingest threads=%d batch=%d (ALPHA_INGEST_THREADS=1 for serial)\n",
                bm25_index::ingest_threads(), ingest_batch_sz);

        // ALPHA_INGEST_PROFILE=1: per-phase timers for the ingest loop.
        const bool ing_prof = getenv("ALPHA_INGEST_PROFILE") != nullptr;
        double t_read = 0, t_bm25 = 0, t_meta = 0, t_unified = 0, t_subloop = 0, t_ner = 0;
        auto ing_t0 = std::chrono::steady_clock::now();
        auto ing_now = []() { return std::chrono::steady_clock::now(); };
        auto ing_sec = [](auto a, auto b) {
            return std::chrono::duration<double>(b - a).count();
        };

        // Fast-ingest split: plain `--context extended` is BM25-only at query (use_ner=false, no
        // graph), so skip the heavy per-chunk NER pass + entity linker + digest. Modes that use the
        // entity graph keep the full path: extended-code, --redis, and auto (auto needs entities to
        // classify code-vs-document). This brings plain-extended ingest close to a pure-BM25 pass.
#ifdef ALPHA_HAS_REDIS
        const bool light_ingest = (params_base.ctx_mode == ALPHA_CTX_EXTENDED) && !use_redis;
#else
        const bool light_ingest = (params_base.ctx_mode == ALPHA_CTX_EXTENDED);
#endif
        if (light_ingest) {
            SRV_INF("%s", "auto-ingest: LIGHT mode (BM25-only; skipping NER/entity graph) for plain --context extended\n");
        }

        for (auto it = fs::recursive_directory_iterator(dir,
                fs::directory_options::skip_permission_denied);
             it != fs::recursive_directory_iterator(); ++it) {
            // Skip excluded directories
            if (it->is_directory()) {
                if (skip_dirs.count(it->path().filename().string())) {
                    it.disable_recursion_pending();
                    continue;
                }
                continue;
            }
            if (!it->is_regular_file()) continue;

            // Check extension
            std::string ext = it->path().extension().string();
            for (auto & c : ext) c = (char)std::tolower((unsigned char)c);
            if (!source_exts.count(ext)) continue;

            // Check size
            if (it->file_size() > max_file_size) {
                SRV_INF("auto-ingest: skipping %s (%zu bytes > %zu-byte cap; raise ALPHA_MAX_FILE_KB)\n",
                        it->path().string().c_str(), (size_t)it->file_size(), (size_t)max_file_size);
                continue;
            }

            // Binary check: read first 512 bytes
            std::ifstream f(it->path(), std::ios::binary);
            if (!f) continue;
            char buf[512];
            f.read(buf, 512);
            auto n_read = f.gcount();
            bool is_binary = false;
            for (int i = 0; i < n_read; i++) {
                if (buf[i] == '\0') { is_binary = true; break; }
            }
            if (is_binary) continue;

            // Read full file (clear EOF after binary check read)
            f.clear();
            f.seekg(0);
            auto rd0 = ing_now();
            std::string text((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
            f.close();
            if (ing_prof) t_read += ing_sec(rd0, ing_now());

            // Relative path as source
            std::string source = fs::relative(it->path(), dir).string();
            // Normalize path separators
            for (auto & c : source) { if (c == '\\') c = '/'; }

            // Experience files: index raw text, skip NER/metadata
            bool is_experience = (source.find("experience") == 0);
            if (is_experience) {
                chunk_bm25.add_document(chunk_id, text, source);
#ifdef ALPHA_DENSE_L0
                if (use_dense) dense.add((uint32_t) chunk_id, text);
#endif
                n_experiences++;
                chunk_id++;
            } else {
                // Article-boundary metadata: if the file starts with a "[Article: title] [ID: ID]"
                // header (one file per article), lift title+id out of the body and strip the header.
                // The title is then stamped on every sub-chunk below (not just chunk 0), so a
                // needle mid-article stays BM25-discoverable by title and every chunk carries
                // its article for citation. No-op for corpora without the header.
                std::string article_title, article_id_str;
                if (text.compare(0, 9, "[Article:") == 0) {
                    size_t eol = text.find('\n');
                    std::string hdr = (eol == std::string::npos) ? text : text.substr(0, eol);
                    auto extract_bracket = [&](const std::string & key) -> std::string {
                        size_t k = hdr.find(key);
                        if (k == std::string::npos) return "";
                        size_t v0 = k + key.size();
                        size_t v1 = hdr.find(']', v0);
                        if (v1 == std::string::npos) return "";
                        std::string v = hdr.substr(v0, v1 - v0);
                        size_t a = v.find_first_not_of(" \t"), b = v.find_last_not_of(" \t");
                        return (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
                    };
                    article_title  = extract_bracket("[Article:");
                    article_id_str = extract_bracket("[ID:");
                    if (!article_title.empty()) {
                        // Drop the header line (and the blank line after it)
                        // so it never pollutes NER / records / the unified aggregation index.
                        size_t body = (eol == std::string::npos) ? text.size() : eol + 1;
                        while (body < text.size() && (text[body] == '\n' || text[body] == '\r')) body++;
                        text.erase(0, body);
                    }
                }

                // NER entity extraction on full file - only for modes that use the entity graph.
                // In light mode (plain --context extended) entities/imports stay empty so the heavy
                // per-chunk NER pass, linker, and digest are all skipped; BM25 + metadata still run.
                auto ner0 = ing_now();
                auto entities = light_ingest ? decltype(entity_extractor.extract(text)){}
                                             : entity_extractor.extract(text);
                auto imports  = light_ingest ? decltype(entity_extractor.extract_imports(text)){}
                                             : entity_extractor.extract_imports(text);
                if (ing_prof) t_ner += ing_sec(ner0, ing_now());
                std::string ner_digest;
                if (!light_ingest) {
                    chunk_entities.add(chunk_id, entities);
                    if (!imports.empty()) {
                        chunk_linker.add_imports(source, imports);
                    }
                    // NER digest (appended to the first sub-chunk for entity-aware routing)
                    for (auto & e : entities) ner_digest += " " + e.text;
                    for (auto & imp : imports) ner_digest += " " + imp;
                }

                // Sub-chunk BM25 indexing: split file into 1K sub-chunks.
                // Each sub-chunk is a separate BM25 document with concentrated IDF.
                // Source format: "file.py@0" "file.py@1024" etc.
                // The parent chunk_id (first sub-chunk) is used for NER entity mapping.
                // Target size, but records are kept intact: after reaching the target the cut snaps
                // to the next record/paragraph boundary ("\n\n", else newline) within 2x the target.
                // Fixed byte splits cut records mid-way and separate a record's header (e.g. a service
                // name) from its fields, which hurts retrieval on record-structured corpora.
                // Default 2KB; override with ALPHA_SUBCHUNK_KB.
                int sub_chunk_size = 2048;
                if (const char * scz = getenv("ALPHA_SUBCHUNK_KB")) {
                    int v = atoi(scz); if (v > 0) sub_chunk_size = v * 1024;
                }
                int parent_id = chunk_id;
                int n_sub = 0;

                auto sub0 = ing_now();
                for (size_t offset = 0; offset < text.size(); ) {
                    size_t len = std::min((size_t)sub_chunk_size, text.size() - offset);
                    if (offset + len < text.size()) {
                        // Snap the cut to a record boundary so we don't split a record mid-way.
                        size_t search_from = offset + len;
                        size_t cap  = offset + (size_t)sub_chunk_size * 2;
                        size_t para = text.find("\n\n", search_from);
                        size_t nl   = text.find('\n', search_from);
                        if (para != std::string::npos && para < cap)     len = para + 2 - offset;
                        else if (nl != std::string::npos && nl < cap)    len = nl + 1 - offset;
                    }
                    if (len < 50) break;  // skip tiny trailing fragments

                    std::string sub_text = text.substr(offset, len);
                    std::string sub_source = source + "@" + std::to_string(offset);

                    // Indexed text: sub-chunk + source tag (+ article title on every sub-chunk
                    // for title-discoverability & citation) + NER digest.
                    std::string indexed = "[Source: " + source + "] ";
                    if (!article_title.empty()) indexed += "[Article: " + article_title + "] ";
                    indexed += sub_text;
                    if (!ner_digest.empty() && n_sub == 0) {
                        // Append NER digest to first sub-chunk only (avoid bloating all)
                        indexed += " [Entities:" + ner_digest + "]";
                    }

                    // Buffer instead of indexing inline: add_documents_batch() parallelises the
                    // pure per-doc work (encrypt/tokenize/hash/sort) and merges strictly in order,
                    // so chunk_ids, text offsets and docs[] are identical to the serial path.
                    // Batching across files matters: one file yields too few sub-chunks to fill
                    // many cores. The parent bookkeeping below is pure chunk_id arithmetic and
                    // does not depend on the batch having been flushed.
                    if (pending_docs.empty()) pending_first_id = chunk_id;
                    pending_docs.emplace_back(std::move(indexed), sub_source);
                    if ((int)pending_docs.size() >= ingest_batch_sz) {
                        auto bm0 = ing_now();
                        chunk_bm25.add_documents_batch(pending_first_id, pending_docs);
                        if (ing_prof) t_bm25 += ing_sec(bm0, ing_now());
#ifdef ALPHA_DENSE_L0
                        // Dense-L0: mirror this batch into the semantic index (same chunk ids as BM25).
                        // Batched embedding (packs many sequences per decode) - far faster than per-chunk.
                        if (use_dense) {
                            std::vector<std::pair<uint32_t, std::string>> dvec; dvec.reserve(pending_docs.size());
                            for (size_t i = 0; i < pending_docs.size(); i++)
                                dvec.emplace_back((uint32_t)(pending_first_id + (int) i), pending_docs[i].first);
                            dense.add_batch(dvec);
                        }
#endif
                        pending_docs.clear();
                    }
                    chunk_id++;
                    n_sub++;
                    offset += len;
                }
                if (ing_prof) t_subloop += ing_sec(sub0, ing_now());

                // Store parent mapping: sub-chunk IDs -> parent file
                if (n_sub > 0) {
                    chunk_bm25.parent_source[parent_id] = source;
                    // Register the whole [parent_id, parent_id+n_sub) range in one shot (8 B/file)
                    // instead of a hash entry per sub-chunk, which dominates RAM at large scale.
                    chunk_bm25.register_parent_range(parent_id, n_sub);
                    // Article metadata (title, article_id, position derivable as chunk-parent_id).
                    if (!article_title.empty()) {
                        chunk_bm25.article_meta[parent_id] = { article_title, article_id_str, n_sub };
                        n_articles++;
                    }
                }

                // Metadata sub-index: parse records within full file.
                // This runs six regex extractors (persons/dates/locations/numerics/categories/ids)
                // on every paragraph and dominates ingest time. It builds a structured-record index
                // (service configs, key:value corpora); on prose it extracts almost nothing, and in
                // plain `--context extended` (light mode) the query path is BM25 anyway, so skip it
                // there like the NER gate above. ALPHA_FORCE_METADATA=1 forces it on.
                static const bool force_metadata = [] { const char * e = getenv("ALPHA_FORCE_METADATA"); return e && atoi(e) != 0; }();
                if (!light_ingest || force_metadata) {
                    auto meta0 = ing_now();
                    chunk_metadata.index_chunk(parent_id, text);
                    if (ing_prof) t_meta += ing_sec(meta0, ing_now());
                }

                // Build the unified router index (entity/aggregation/section, one pass) for every
                // mode that uses it: `unified` and `extended` (extended routes through unified,
                // falling back to BM25), plus `auto` (which can resolve to extended at query time).
                if (params_base.ctx_mode == ALPHA_CTX_UNIFIED ||
                    params_base.ctx_mode == ALPHA_CTX_EXTENDED ||
                    params_base.ctx_mode == ALPHA_CTX_AUTO) {
                    auto uni0 = ing_now();
                    unified.ingest(text, entities, source, article_title);   // NER values + bio facts by title
                    if (ing_prof) t_unified += ing_sec(uni0, ing_now());
                }

#ifdef ALPHA_HAS_REDIS
                // Redis ingest: sentences + entities + imports + code patterns
                if (use_redis && redis) {
                    std::vector<redis_entity> r_entities;
                    for (auto & e : entities) {
                        redis_entity re;
                        re.name = e.text;
                        switch (e.type) {
                            case entity_type::FILE:     re.type = "FILE"; break;
                            case entity_type::FUNCTION: re.type = "FUNC"; break;
                            case entity_type::CLASS:    re.type = "CLASS"; break;
                            case entity_type::ERROR:    re.type = "ERROR"; break;
                            case entity_type::VARIABLE: re.type = "VAR"; break;
                            default:                    re.type = "OTHER"; break;
                        }
                        re.source = source;
                        r_entities.push_back(re);
                    }
                    // Relationships: imports are relationships
                    std::vector<redis_relationship> r_rels;
                    for (auto & imp : imports) {
                        r_rels.push_back({source, "imports", imp});
                    }
                    redis->ingest_file(source, text, r_entities, r_rels, imports);
                }
#endif
            }

            n_files++;
        }

        // Flush the parallel-ingest tail. must run before finalize_text_store(), which closes and
        // mmaps the ciphertext file - anything still buffered here would be silently dropped from
        // both the index and the text store.
        if (!pending_docs.empty()) {
            auto bm0 = ing_now();
            chunk_bm25.add_documents_batch(pending_first_id, pending_docs);
            if (ing_prof) t_bm25 += ing_sec(bm0, ing_now());
#ifdef ALPHA_DENSE_L0
            if (use_dense) {
                std::vector<std::pair<uint32_t, std::string>> dvec; dvec.reserve(pending_docs.size());
                for (size_t i = 0; i < pending_docs.size(); i++)
                    dvec.emplace_back((uint32_t)(pending_first_id + (int) i), pending_docs[i].first);
                dense.add_batch(dvec);
            }
#endif
            pending_docs.clear();
        }
        if (ing_prof) {
            double tot = ing_sec(ing_t0, ing_now());
            double t_split = t_subloop - t_bm25;   // sub-chunk substr/indexed build, minus the flush
            auto pct = [&](double t) { return 100.0 * t / tot; };
            SRV_INF("auto-ingest PROFILE: total=%.1fs | read=%.1fs(%.0f%%) ner=%.1fs(%.0f%%) "
                    "split=%.1fs(%.0f%%) bm25=%.1fs(%.0f%%) meta=%.1fs(%.0f%%) unified=%.1fs(%.0f%%) "
                    "rest=%.1fs(%.0f%%)\n",
                    tot, t_read, pct(t_read), t_ner, pct(t_ner),
                    t_split, pct(t_split), t_bm25, pct(t_bm25), t_meta, pct(t_meta),
                    t_unified, pct(t_unified),
                    tot - t_read - t_ner - t_split - t_bm25 - t_meta - t_unified,
                    pct(tot - t_read - t_ner - t_split - t_bm25 - t_meta - t_unified));
        }

        chunk_bm25.finalize_text_store();   // close + mmap the BM25 ciphertext store (off-RAM)
        chunk_bm25.finalize_text_offsets(); // checkpoint + free the per-doc offset array
        chunk_bm25.finalize_df();           // pack the global df map -> sorted arrays
        // ALPHA_BM25_INVERTED: invert the resident forward tf arrays -> varint postings on a mmap'd file,
        // then free the tf arrays. Moves postings off-RAM and turns the query into posting-list
        // traversal. No-op unless the flag is set. Needs the tf arrays, which nothing above frees.
        if (bm25_index::bm25_inverted()) {
            const char * td = getenv("TMPDIR"); std::string tmpdir = (td && *td) ? td : "/tmp";
            std::string ppath = tmpdir + "/alpha_bm25_" + std::to_string((long)getpid()) + ".post.bin";
            auto iv0 = std::chrono::steady_clock::now();
            bool was_spill = chunk_bm25.spill_active;
            uint64_t spilled = chunk_bm25.spill_records;
            chunk_bm25.build_inverted(ppath);
            auto iv_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - iv0).count();
            size_t nterms = chunk_bm25.dict_mm_active ? chunk_bm25.dict_n : chunk_bm25.df_terms.size();
            SRV_INF("BM25 INVERTED (%s): %s — %zu terms, %llu postings, %.1f MB postings %s, dict %s (%.1f MB), built in %.1fs\n",
                    was_spill ? "external/spill" : "resident",
                    chunk_bm25.inverted_active ? "active" : "FAILED (stayed on forward path)",
                    nterms, (unsigned long long)spilled, chunk_bm25.post_mm_len / 1e6,
                    chunk_bm25.post_fallback.empty() ? "mmap" : "resident-fallback",
                    chunk_bm25.dict_mm_active ? "off-RAM mmap" : "resident",
                    chunk_bm25.dict_mm_len / 1e6, iv_ms / 1000.0f);
        }
        chunk_metadata.finalize();          // pack the 4 inverted indexes -> key arena + CSR

        if (unified.built) {
            // records.bin -> a private temp file (not the corpus dir: ingest lists corpus files, and
            // build() unlinks it right after mmap so nothing is left on disk).
            const char * td = getenv("TMPDIR"); std::string tmpdir = (td && *td) ? td : "/tmp";
            unified.records_path = tmpdir + "/alpha_unified_" + std::to_string((long)getpid()) + ".records.bin";
            unified.finalize();
        }

        // The finalize_*() calls freed the build-time df/metadata/unified maps, but glibc keeps those
        // pages on its heap arenas, so serving RSS would stay near the build peak. malloc_trim returns
        // them to the OS.
#if defined(__GLIBC__)
        malloc_trim(0);
#endif

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        SRV_INF("auto-ingest: indexed %d files → %d sub-chunks (%d experiences, %d articles), %d entities, %d records in %.1fs\n",
                n_files, chunk_bm25.N, n_experiences, n_articles, chunk_entities.n_unique, chunk_metadata.n_records, elapsed / 1000.0f);
#ifdef ALPHA_DENSE_L0
        // Build the IVF ANN over the dense vectors so queries stay sub-linear past ~1M vectors
        // (no-op below the threshold; brute-force cosine is fine there).
        if (use_dense && !dense.ann_current()) dense.build_ann();   // a restored store already built it (no re-embed ran)
        // Persist the vectors so the next boot does not re-embed (hours per 1B tokens on a GPU
        // embedder, far more on a CPU one).
        if (use_dense && dense.size() > 0) {
            if (const char * ppath3 = getenv("ALPHA_DENSE_L0_PERSIST")) {
                const char * ow = getenv("ALPHA_DENSE_L0_PERSIST_OVERWRITE");
                if (ppath3[0] && dense_persist_refused && !(ow && atoi(ow) != 0)) {
                    SRV_WRN("dense-l0: NOT persisting to %s -- that store was refused at load and is kept as it is; "
                            "fix the configuration, or set ALPHA_DENSE_L0_PERSIST_OVERWRITE=1 (or another path) to "
                            "replace it with this run's vectors\n", ppath3);
                } else if (ppath3[0] && !dense.save_vectors(ppath3)) {
                    SRV_ERR("dense-l0: failed to persist the vector store to %s\n", ppath3);
                }
            }
        }
#endif
        SRV_INF("off-RAM text: BM25 %.1f MB (%s) + unified records.bin %.1f MB → %.1f MB paged from disk instead of resident\n",
                chunk_bm25.text_mm_len / 1e6, chunk_bm25.text_on_disk ? "mmap" : "resident-fallback",
                unified.tier.mm_len / 1e6, (chunk_bm25.text_mm_len + unified.tier.mm_len) / 1e6);
        {
            auto br = chunk_bm25.resident();
            SRV_INF("RESIDENT PROFILE: bm25[tf=%.0f df=%.0f inverted=%.0f source=%.0f docstruct=%.0f maps=%.0f textResident=%.0f] "
                    "metadata=%.0f unified_tier=%.0f (all MB)\n",
                    br.tf/1e6, br.df/1e6, br.inverted/1e6, br.source/1e6, br.docstruct/1e6, br.maps/1e6, br.text/1e6,
                    chunk_metadata.resident_bytes()/1e6, unified.tier.resident_bytes()/1e6);
        }

#ifdef ALPHA_HAS_REDIS
        if (use_redis && redis) {
            redis->flush();
            fprintf(stderr, "redis: %d sentences, %d entities, %d relationships, %d codes\n",
                    redis->n_sentences, redis->n_entities, redis->n_relationships, redis->n_codes);
        }
#endif
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            // Try RadixAttention first: token-level prefix matching
            bool radix_hit = false;
            if (radix_kv_cache.memory_budget() > 0 && !task.tokens.empty() &&
                task.type == SERVER_TASK_TYPE_COMPLETION) {
                auto task_tokens = task.tokens.get_tokens();
                auto match = radix_kv_cache.find_prefix(task_tokens.data(), task_tokens.size());
                if (match.has_kv && match.matched_tokens > 0) {
                    // Radix hit: restore KV from tree node
                    const size_t n = llama_state_seq_set_data_ext(
                        ctx, match.node->kv_data.data(), match.node->kv_data.size(), ret->id, 0);
                    if (n == match.node->kv_data.size()) {
                        // Rebuild prompt token tracking to reflect restored state
                        ret->prompt.tokens.clear();
                        for (int i = 0; i < match.matched_tokens; i++) {
                            ret->prompt.tokens.push_back(task_tokens[i]);
                        }
                        radix_kv_cache.record_hit();
                        radix_hit = true;
                        SRV_INF("radix cache hit: %d/%d tokens matched (%.0f%% skip)\n",
                                match.matched_tokens, (int)task_tokens.size(),
                                100.0 * match.matched_tokens / task_tokens.size());
                    }
                    radix_kv_cache.release(match.node);
                }
                if (!radix_hit) {
                    radix_kv_cache.record_miss();
                }
            }

            // Fall back to existing prompt cache if radix didn't match
            update_cache = update_cache && prompt_cache;
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;
            update_cache = update_cache && !radix_hit;

            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                // Save to KV pool L2 before discarding (if enabled)
                if (kv_pool_cache.is_enabled() && slot.prompt.tokens.size() > 0) {
                    auto tokens_vec = slot.prompt.tokens.get_tokens();
                    uint64_t hash = kv_pool::hash_tokens(tokens_vec.data(), tokens_vec.size());
                    std::string key = kv_pool::make_key("slot" + std::to_string(slot.id), hash);

                    const size_t state_size = llama_state_seq_get_size_ext(ctx, slot.id, 0);
                    if (state_size > 0) {
                        std::vector<uint8_t> state_data(state_size);
                        llama_state_seq_get_data_ext(ctx, state_data.data(), state_size, slot.id, 0);
                        kv_pool_cache.save(key, state_data.data(), state_size, (int32_t)tokens_vec.size());
                    }
                }

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_INF(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_INF(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_logits = task.params.sampling.n_probs > 0;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate() && task.params.speculative.n_max > 0);

            // TODO: getting post/pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx, slot.id, nullptr);
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string_view str_test(slot.generated_text.data() + pos,
                                             slot.generated_text.size() - pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send.assign(slot.generated_text, pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                // Batch streaming: accumulate tokens, flush every 4 to reduce queue overhead
                slot.pending_stream_tokens.push_back(result);
                bool should_flush = slot.pending_stream_tokens.size() >= 4
                                 || slot.stop != STOP_TYPE_NONE
                                 || !slot.has_next_token;
                if (should_flush) {
                    send_partial_response_batch(slot, slot.pending_stream_tokens);
                    slot.pending_stream_tokens.clear();
                }
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits (no n_decoded > 0 guard - so max_tokens=0 produces empty response)
        if (slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        // Warmup-LUT Cycle
        // Two rules, continuously enforced:
        //   1. cpu-moe until cache warm (loaded==0) -> switch to LUT
        //   2. Every LUT token: check confidence. Low -> back to cpu-moe warmup
        // No caps, no special cases. The cycle self-corrects:
        //   warmup -> LUT -> confidence low -> warmup -> LUT -> confidence OK -> stay LUT

        // Speculative LUT Cycle (30-token windows)
        // Every 30 tokens: 2 cpu-moe anchors + 28 LUT tokens + confidence gate
        // Anchors keep recurrent state correct. LUT generates fast between anchors.
        // Low-confidence LUT tokens get cpu-moe fix inline.
        if (moe_opt && slot.moe_adaptive_warmup) {
            int cycle_pos = slot.n_decoded % 30;  // position within 30-token window

            if (cycle_pos < 2) {
                // Anchor: cpu-moe for first 2 tokens of each window
                // Correct experts via mmap. Establishes correct recurrent state.
                if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                    alpha_moe_optimizer_exit_draft_mode(moe_opt);
                    alpha_moe_optimizer_disable_cache(moe_opt);
                }
                // Still call reload to update cache with correct experts
                alpha_moe_optimizer_enable_cache(moe_opt);
                alpha_moe_optimizer_reload_from_argsort(moe_opt);
                alpha_moe_optimizer_disable_cache(moe_opt);

                if (slot.n_decoded <= 10 || cycle_pos == 0) {
                }
            } else if (cycle_pos == 2) {
                // Switch to LUT at position 2
                alpha_moe_optimizer_enable_cache(moe_opt);
                alpha_moe_optimizer_enter_draft_mode(moe_opt);
                if (slot.n_decoded <= 10 || (slot.n_decoded / 30) % 5 == 0) {
                }
            } else {
                // LUT generation (positions 3-29)
                // No per-token confidence gate. At low K (e.g. 54/256), LUT logit gap
                // is naturally ~0.1 on every token. Gating would fix every token = slower
                // than pure cpu-moe. Instead, trust the 2 anchor tokens at each window
                // boundary to correct any drift from speculative LUT generation.
                // The LUT runs freely here; anchors at positions 0-1 handle correction.
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            // Early EOS repair: if LUT prefill produced bad KV, the model
            // outputs EOS immediately (n_decoded <= 2). Re-do with cpu-moe.
            auto * moe = moe_opt;
            if (slot.n_decoded <= 5 && moe && alpha_moe_optimizer_is_draft_active(moe) &&
                !slot.moe_early_eos_repaired) {
                slot.moe_early_eos_repaired = true;

                // Exit draft + disable cache -> cpu-moe for both prefill and generation
                if (alpha_moe_optimizer_is_draft_active(moe)) {
                    alpha_moe_optimizer_exit_draft_mode(moe);
                }
                alpha_moe_optimizer_disable_cache(moe);

                // Clear garbage KV
                auto * mem = llama_get_memory(ctx);
                if (mem) llama_memory_seq_rm(mem, slot.id, -1, -1);

                // Reset slot to re-process with cpu-moe
                // Use g_cpu_moe_active to keep cache disabled through generation
                g_cpu_moe_active = true;
                slot.prompt_clear(true);
                slot.n_prompt_tokens_processed = 0;
                slot.n_decoded = 0;
                slot.stop = STOP_TYPE_NONE;
                slot.has_next_token = true;
                slot.state = SLOT_STATE_STARTED;

                return true;
            }

            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    // Batch streaming: send multiple accumulated tokens in one queue message
    void send_partial_response_batch(server_slot & slot, const std::vector<completion_token_output> & tokens) {
        if (tokens.empty()) return;

        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        // Concatenate text from all tokens
        std::string combined_text;
        llama_tokens combined_toks;
        for (const auto & tkn : tokens) {
            combined_text += tkn.text_to_send;
            combined_toks.push_back(tkn.tok);
        }
        res->content = std::move(combined_text);
        res->tokens  = std::move(combined_toks);

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // Use last token's probs if requested
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tokens.back();
        }

        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    // ALPHA_BRANCH_RECOVER: entropy-gated sparse-branch sampler.
    // Greedy (argmax) by default. At a genuinely uncertain step (softmax entropy >= br_entropy_min) take
    // one token sampled from the top-p nucleus - the "branch" - then force greedy for br_recover_n
    // tokens so the model commits to and stabilises the branch before diverging again. Capped at
    // br_max_branch branches per generation. Operates on the raw model logits (research decode policy,
    // independent of the request sampler chain). Seeded per generation (n_decoded==0) from br_seed =>
    // reproducible candidate sets. Only called when br_enabled, so the flag-off path is unchanged.
    llama_token br_sample(server_slot & slot, int tok_idx) {
        const float * logits = llama_get_logits_ith(slot.ctx, tok_idx);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        // argmax - the greedy choice and the fallback
        int argmax = 0; float mx = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > mx) { mx = logits[v]; argmax = v; }

        // Reset per-generation state on the first sampled token (deterministic seed => reproducible).
        if (slot.n_decoded == 0) {
            slot.br_rng.seed(br_seed);
            slot.br_recover_left = 0;
            slot.br_branch_count = 0;
        }

        // Recovery window still open, or branch budget spent => forced greedy.
        if (slot.br_recover_left > 0 || slot.br_branch_count >= br_max_branch) {
            if (slot.br_recover_left > 0) slot.br_recover_left--;
            return argmax;
        }

        // Shannon entropy (nats) of the softmax over raw logits - numerically stable via log-sum-exp.
        double Z = 0.0;
        for (int v = 0; v < n_vocab; v++) Z += std::exp((double)logits[v] - mx);
        const double logZ = std::log(Z) + mx;
        double H = 0.0;
        for (int v = 0; v < n_vocab; v++) {
            const double p = std::exp((double)logits[v] - logZ);
            if (p > 1e-12) H -= p * std::log(p);
        }
        if (H < br_entropy_min) return argmax;   // confident step => stay greedy

        // branch: draw one token from the top-p nucleus (renormalised within the nucleus mass).
        std::vector<std::pair<float,int>> pr; pr.reserve(n_vocab);
        for (int v = 0; v < n_vocab; v++) pr.emplace_back((float)std::exp((double)logits[v] - logZ), v);
        std::sort(pr.begin(), pr.end(), [](const std::pair<float,int>&a, const std::pair<float,int>&b){ return a.first > b.first; });
        double cum = 0.0; size_t cut = 0;
        for (; cut < pr.size(); ++cut) { cum += pr[cut].first; if (cum >= br_topp) { cut++; break; } }
        if (cut == 0) cut = 1;
        std::uniform_real_distribution<double> U(0.0, cum);
        const double r = U(slot.br_rng);
        int pick = pr[0].second; double acc = 0.0;
        for (size_t i = 0; i < cut; i++) { acc += pr[i].first; if (r <= acc) { pick = pr[i].second; break; } }
        slot.br_recover_left = br_recover_n;
        slot.br_branch_count++;
        return pick;
    }

    // ALPHA_LOG_EXPERT_USAGE: passive per-generation router-selection accounting
    // Read-only tap on the persisted `selected_experts` argsort tensors (already computed for real
    // routing) - reads the just-decoded token's top-k row per MoE layer and accumulates per-(layer,
    // expert) counts on the slot. Touches no sampling/routing state, so tokens are byte-identical to
    // the flag-off path. Validated for single-stream (-np1) non-spec decode:
    // each decode forward emits one token, so the argsort's last row is exactly that token's routing.
    void eu_accumulate(server_slot & slot) {
        if (!eu_log_enabled || eu_n_expert_total <= 0) return;
        int nl = alpha_mole_get_argsort_count();
        if (nl <= 0) return;
        if (slot.eu_counts.empty()) {
            slot.eu_n_layer  = nl;
            slot.eu_n_expert = eu_n_expert_total;
            slot.eu_counts.assign((size_t)nl * (size_t)eu_n_expert_total, 0u);
        }
        if (nl != slot.eu_n_layer) return;   // graph layout changed unexpectedly; skip this token
        int used = 0;
        for (int l = 0; l < nl; l++) {
            ggml_tensor * a = alpha_mole_get_argsort_output(l);
            if (!a || !a->buffer) continue;
            int nu = (int)a->ne[0];          // n_expert_used
            int nt = (int)a->ne[1];          // tokens in this forward (1 at -np1 non-spec decode)
            if (nu <= 0 || nt <= 0) continue;
            used = nu;
            int32_t ids[64];
            int rd = std::min(nu, 64);
            size_t off = (size_t)(nt - 1) * (size_t)a->ne[0] * sizeof(int32_t);  // last token's row
            ggml_backend_tensor_get(a, ids, off, (size_t)rd * sizeof(int32_t));
            uint32_t * row = &slot.eu_counts[(size_t)l * (size_t)eu_n_expert_total];
            for (int e = 0; e < rd; e++) {
                int32_t eid = ids[e];
                if (eid >= 0 && eid < eu_n_expert_total) row[eid]++;
            }
        }
        if (used) slot.eu_n_used = used;
        slot.eu_decode_tokens++;
    }

    // Emit one jsonl line per completion, keyed by request/task id, then reset the accumulator.
    void eu_emit(server_slot & slot) {
        if (!eu_log_enabled || !eu_log_fp || slot.eu_decode_tokens <= 0 || slot.eu_counts.empty()) return;
        const int L = slot.eu_n_layer, E = eu_n_expert_total;
        std::string cov_per_layer = "[";
        std::string matrix = "[";
        long long union_pairs = 0;
        double cov_sum = 0.0;
        std::vector<long long> flat((size_t)E, 0);   // activations per expert id summed across layers
        for (int l = 0; l < L; l++) {
            const uint32_t * rrow = &slot.eu_counts[(size_t)l * (size_t)E];
            int distinct = 0;
            if (eu_log_verbose) { if (l) matrix += ","; matrix += "["; }
            for (int e = 0; e < E; e++) {
                uint32_t c = rrow[e];
                if (c) { distinct++; union_pairs++; flat[(size_t)e] += c; }
                if (eu_log_verbose) { if (e) matrix += ","; matrix += std::to_string(c); }
            }
            if (eu_log_verbose) matrix += "]";
            if (l) cov_per_layer += ",";
            char b[24]; snprintf(b, sizeof(b), "%.4f", E ? (double)distinct / E : 0.0);
            cov_per_layer += b;
            cov_sum += E ? (double)distinct / E : 0.0;
        }
        cov_per_layer += "]";
        matrix += "]";
        std::string flat_s = "[";
        for (int e = 0; e < E; e++) { if (e) flat_s += ","; flat_s += std::to_string(flat[(size_t)e]); }
        flat_s += "]";
        double cov_mean   = L ? cov_sum / L : 0.0;
        double cov_global = (L && E) ? (double)union_pairs / ((double)L * E) : 0.0;
        char hdr[640];
        snprintf(hdr, sizeof(hdr),
            "{\"req_id\":%d,\"index\":%d,\"phase\":\"decode\",\"n_decode_tokens\":%lld,"
            "\"n_moe_layers\":%d,\"n_expert_total\":%d,\"n_expert_used\":%d,"
            "\"coverage_mean\":%.5f,\"coverage_global\":%.5f,",
            slot.task ? slot.task->id : -1, slot.task ? slot.task->index : -1,
            (long long)slot.eu_decode_tokens, L, E, slot.eu_n_used, cov_mean, cov_global);
        std::string line = hdr;
        line += "\"coverage_per_layer\":" + cov_per_layer + ",";
        line += "\"per_expert_count\":" + flat_s;
        if (eu_log_verbose) line += ",\"layer_expert_counts\":" + matrix;
        line += "}\n";
        fputs(line.c_str(), eu_log_fp);
        fflush(eu_log_fp);
        // reset for slot reuse
        std::vector<uint32_t>().swap(slot.eu_counts);
        slot.eu_n_layer = 0; slot.eu_decode_tokens = 0;
    }

    void send_final_response(server_slot & slot) {
        eu_emit(slot);   // expert-usage jsonl (no-op unless ALPHA_LOG_EXPERT_USAGE); before slot reset
        // Flush any pending batched stream tokens before final response
        if (!slot.pending_stream_tokens.empty()) {
            send_partial_response_batch(slot, slot.pending_stream_tokens);
            slot.pending_stream_tokens.clear();
        }

        // Radix KV save: done at DONE_PROMPT (prompt-only depth, before generation).
        // not saved here - generated tokens make the path unique and non-shareable.

        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(ctx, i);
            } else {
                embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        const auto & cur = slot.prompt.checkpoints.emplace_back(server_get_checkpoint(ctx, slot.id, slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max));

        SLT_WRN(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
            // KV-Chunk Tasks (processed on main thread for thread safety)
            case SERVER_TASK_TYPE_KV_DECODE_STEP:
                {
                    auto & kd = task.kv_data;

                    llama_token tok = kd.token_id;
                    llama_batch batch = llama_batch_get_one(&tok, 1);
                    int rc = llama_decode(ctx, batch);
                    // batch_get_one: no free needed (stack-allocated, no malloc)

                    auto res = std::make_unique<server_task_result_json>();
                    res->id = task.id;
                    if (rc != 0) {
                        res->data = {{"error", "decode failed", "rc", rc}};
                    } else {
                        const float * logits = llama_get_logits(ctx);
                        if (!logits) {
                            res->data = {{"error", "null logits"}};
                            queue_results.send(std::move(res));
                            break;
                        }
                        const llama_vocab * vocab_ptr = llama_model_get_vocab(model);
                        int n_vocab = llama_vocab_n_tokens(vocab_ptr);
                        std::vector<std::pair<float, int>> scored(n_vocab);
                        for (int i = 0; i < n_vocab; i++) scored[i] = {logits[i], i};
                        int tk = std::min(kd.top_k, n_vocab);
                        std::partial_sort(scored.begin(), scored.begin() + tk, scored.end(),
                            [](const auto & a, const auto & b) { return a.first > b.first; });
                        json logits_arr = json::array();
                        for (int i = 0; i < tk; i++)
                            logits_arr.push_back({{"token_id", scored[i].second}, {"logit", scored[i].first}});
                        float margin = tk >= 2 ? scored[0].first - scored[1].first : 0.0f;
                        res->data = {{"logits_top_k", logits_arr}, {"margin", margin}, {"position", kd.position}};
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_KV_PREFILL_CHUNK:
                {
                    auto & kd = task.kv_data;
                    // Use seq_id above n_parallel to avoid slot conflicts
                    // chunk_id=-1 -> seq_id=0 (shared), chunk_id>=0 -> seq_id=n_parallel (reused)
                    int np = (int)params_base.n_parallel;
                    llama_seq_id sid = kd.chunk_id < 0 ? 0 : np;
                    int n = (int)kd.tokens.size();
                    int n_batch = llama_n_batch(ctx);
                    if (n_batch <= 0) n_batch = 512;
                    int rc = 0;

                    // Split into sub-batches respecting n_batch limit
                    for (int offset = 0; offset < n && rc == 0; offset += n_batch) {
                        int sub_n = std::min(n_batch, n - offset);
                        llama_batch batch = llama_batch_init(sub_n, 0, 1);
                        for (int i = 0; i < sub_n; i++) {
                            common_batch_add(batch, kd.tokens[offset + i],
                                             kd.position_offset + offset + i, { sid },
                                             (offset + i) == n - 1);
                        }
                        rc = llama_decode(ctx, batch);
                        llama_batch_free(batch);
                    }

                    auto res = std::make_unique<server_task_result_json>();
                    res->id = task.id;
                    res->data = {
                        {"chunk_id", kd.chunk_id},
                        {"tokens_processed", (int)kd.tokens.size()},
                        {"position_range", {kd.position_offset, kd.position_offset + (int)kd.tokens.size()}},
                        {"sealed", kd.seal},
                        {"rc", rc}
                    };
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_KV_ACTIVATE_CHUNKS:
                {
                    auto & kd = task.kv_data;
                    llama_memory_t mem = llama_get_memory(ctx);
                    int np = (int)params_base.n_parallel;
                    int chunk_size = 4096;  // tokens per chunk

                    if (kd.active_chunk_ids.empty()) {
                        // Clear all chunk KV
                        llama_memory_seq_rm(mem, np, -1, -1);
                    } else {
                        // Selective activation: keep only selected chunks' positions.
                        // First, copy seq np to a temp seq, clear np, then copy back
                        // only the selected position ranges.
                        // Simpler approach: remove positions of NON-selected chunks.
                        // Build a set of active position ranges.
                        std::set<int> active_set(kd.active_chunk_ids.begin(), kd.active_chunk_ids.end());

                        // Find the total number of chunks in KV by checking max position
                        llama_pos max_pos = llama_memory_seq_pos_max(mem, np);
                        int n_total_chunks = (max_pos > 0) ? (int)(max_pos / chunk_size) + 1 : 0;

                        // Remove inactive chunk position ranges
                        int removed = 0;
                        for (int ci = 0; ci < n_total_chunks; ci++) {
                            if (active_set.count(ci) == 0) {
                                llama_pos p0 = (llama_pos)ci * chunk_size;
                                llama_pos p1 = p0 + chunk_size;
                                llama_memory_seq_rm(mem, np, p0, p1);
                                removed++;
                            }
                        }

                        // Also remove any "tail" positions beyond chunks (query residue)
                        if (!kd.include_tail) {
                            llama_pos tail_start = (llama_pos)n_total_chunks * chunk_size;
                            if (tail_start <= max_pos) {
                                llama_memory_seq_rm(mem, np, tail_start, -1);
                            }
                        }

                        fprintf(stderr, "[ACTIVATE] %zu chunks active, %d removed (total %d, max_pos %d)\n",
                                kd.active_chunk_ids.size(), removed, n_total_chunks, (int)max_pos);
                    }

                    auto res = std::make_unique<server_task_result_json>();
                    res->id = task.id;
                    res->data = {
                        {"active_chunks", kd.active_chunk_ids},
                        {"include_tail", kd.include_tail}
                    };
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_KV_EXTRACT_CENTROIDS:
                {
                    auto & kd = task.kv_data;
                    auto res = std::make_unique<server_task_result_json>();
                    res->id = task.id;

                    // Access KV cache (hybrid or direct)
                    llama_memory_t mem = llama_get_memory(ctx);
                    llama_kv_cache * kv = nullptr;
                    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
                    if (hybrid) {
                        kv = hybrid->get_mem_attn();
                    } else {
                        kv = dynamic_cast<llama_kv_cache *>(mem);
                    }
                    if (!kv) {
                        res->data = {{"error", "No KV cache available"}};
                        queue_results.send(std::move(res));
                        break;
                    }

                    // Find layer
                    ggml_tensor * k_t = nullptr;
                    ggml_tensor * v_t = nullptr;
                    for (const auto & kl : kv->layers) {
                        if ((int)kl.il == kd.layer_idx) {
                            k_t = kl.k; v_t = kl.v; break;
                        }
                    }
                    if (!k_t || !v_t) {
                        res->data = {{"error", "Layer not found in KV cache"}};
                        queue_results.send(std::move(res));
                        break;
                    }

                    int64_t n_embd = k_t->ne[0];
                    int64_t kv_size = k_t->ne[1];
                    int pos_end = kd.pos_end > 0 ? std::min(kd.pos_end, (int)kv_size) : (int)kv_size;

                    std::vector<float> centroid_k(n_embd, 0.0f);
                    std::vector<float> centroid_v(n_embd, 0.0f);
                    std::vector<float> row(n_embd);
                    int count = 0;

                    for (int pos = kd.pos_start; pos < pos_end; pos++) {
                        // Read K row
                        size_t off = pos * k_t->nb[1];
                        if (k_t->type == GGML_TYPE_F32) {
                            ggml_backend_tensor_get(k_t, row.data(), off, n_embd * sizeof(float));
                        } else if (k_t->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> tmp(n_embd);
                            ggml_backend_tensor_get(k_t, tmp.data(), off, n_embd * sizeof(ggml_fp16_t));
                            for (int i = 0; i < n_embd; i++) row[i] = ggml_fp16_to_fp32(tmp[i]);
                        } else if (k_t->type == GGML_TYPE_BF16) {
                            std::vector<ggml_bf16_t> tmp(n_embd);
                            ggml_backend_tensor_get(k_t, tmp.data(), off, n_embd * sizeof(ggml_bf16_t));
                            for (int i = 0; i < n_embd; i++) {
                                uint32_t u = (uint32_t)tmp[i].bits << 16;
                                float f; memcpy(&f, &u, 4); row[i] = f;
                            }
                        } else { continue; }
                        bool has = false;
                        for (int i = 0; i < std::min((int64_t)4, n_embd); i++) if (row[i] != 0) { has = true; break; }
                        if (!has) continue;
                        for (int i = 0; i < n_embd; i++) centroid_k[i] += row[i];

                        // Read V row
                        off = pos * v_t->nb[1];
                        if (v_t->type == GGML_TYPE_F32) {
                            ggml_backend_tensor_get(v_t, row.data(), off, n_embd * sizeof(float));
                        } else if (v_t->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> tmp(n_embd);
                            ggml_backend_tensor_get(v_t, tmp.data(), off, n_embd * sizeof(ggml_fp16_t));
                            for (int i = 0; i < n_embd; i++) row[i] = ggml_fp16_to_fp32(tmp[i]);
                        } else if (v_t->type == GGML_TYPE_BF16) {
                            std::vector<ggml_bf16_t> tmp(n_embd);
                            ggml_backend_tensor_get(v_t, tmp.data(), off, n_embd * sizeof(ggml_bf16_t));
                            for (int i = 0; i < n_embd; i++) {
                                uint32_t u = (uint32_t)tmp[i].bits << 16;
                                float f; memcpy(&f, &u, 4); row[i] = f;
                            }
                        }
                        for (int i = 0; i < n_embd; i++) centroid_v[i] += row[i];
                        count++;
                    }
                    if (count > 0) {
                        for (int i = 0; i < n_embd; i++) { centroid_k[i] /= count; centroid_v[i] /= count; }
                    }
                    res->data = {
                        {"layer_idx", kd.layer_idx}, {"token_count", count}, {"n_embd", n_embd},
                        {"centroid_k", centroid_k}, {"centroid_v", centroid_v}
                    };
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_KV_SAVE:
                {
                    auto & kd = task.kv_data;
                    auto res = std::make_unique<server_task_result_json>();
                    res->id = task.id;
                    // Use per-sequence save - chunk seq_id = n_parallel
                    int np = (int)params_base.n_parallel;
                    llama_seq_id save_seq = kd.chunk_id >= 0 ? np : 0;
                    size_t bytes = llama_state_seq_save_file(ctx, kd.path.c_str(), save_seq, nullptr, 0);
                    res->data = {{"saved", bytes > 0}, {"path", kd.path},
                                 {"bytes", bytes}, {"seq_id", save_seq}};
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_KV_LOAD:
                {
                    auto & kd = task.kv_data;
                    auto res = std::make_unique<server_task_result_json>();
                    res->id = task.id;
                    // Use per-sequence load - chunk seq_id = n_parallel
                    int np = (int)params_base.n_parallel;
                    llama_seq_id load_seq = kd.chunk_id >= 0 ? np : 0;
                    std::vector<llama_token> tokens_out(4096);
                    size_t n_token_count = 0;
                    auto t0 = std::chrono::high_resolution_clock::now();
                    size_t bytes = llama_state_seq_load_file(ctx, kd.path.c_str(), load_seq,
                        tokens_out.data(), tokens_out.size(), &n_token_count);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    res->data = {{"loaded", bytes > 0}, {"path", kd.path},
                                 {"load_time_ms", ms}, {"bytes", bytes},
                                 {"n_tokens", n_token_count}, {"seq_id", load_seq}};
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                // Periodic scheduler reset to prevent VRAM growth from
                // compute buffer fragmentation over many requests.
                // Without it, small GPUs eventually run out of memory.
                n_requests_since_reset++;
                if (n_requests_since_reset >= RESET_SCHED_INTERVAL) {
                    llama_context_reset_sched(ctx);
                    n_requests_since_reset = 0;
                    SRV_INF("scheduler reset (VRAM defrag) after %d requests\n",
                            RESET_SCHED_INTERVAL);
                }

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, n_keep, n_keep + n_discard)) {
                    // Hybrid model: partial removal failed (recurrent state is monolithic)
                    SLT_WRN(slot, "%s", "context shift failed (hybrid model) - clearing memory\n");
                    slot.prompt_clear(true);
                    continue;
                }
                llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            // Layer-skip self-spec: generate drafts before batch building.
            if (slot.task && slot.task->params.speculative.self_spec_skip > 0 &&
                slot.state == SLOT_STATE_GENERATING) {
                slot.self_spec_generate_drafts();
            }

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        // MoE prompt processing: profile-first or cpu-moe
                        // Only for models with expert cache (K > 0). ALL_GPU models
                        // (all experts on GPU, K=0) need no intervention - skip entirely.
                        if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
                            alpha_moe_optimizer_get_K(moe_opt) > 0) {
                            int cur_K = alpha_moe_optimizer_get_K(moe_opt);
                            int n_exp = alpha_moe_optimizer_get_n_expert(moe_opt);

                            // Force cpu-moe when:
                            // - explicit cpu_moe request (repair)
                            // - classify sub-request
                            bool force_cpu_moe = g_cpu_moe_active || g_classify_active;

                            // Skip keyword domain scan for classify - classify sets its
                            // own domain ("classify") in pin_domain_experts.
                            // For main requests: detect domain from prompt keywords.
                            if (!g_classify_active) {
                                if (slot.task && slot.task->tokens.size() > 0) {
                                    std::string prompt_text = slot.task->tokens.detokenize(ctx, false);

                                    // Try 0.6B classifier first (fast, 124 domains)
                                    const char * cl_domain = has_classifier()
                                        ? classify_domain_fast(prompt_text) : nullptr;

                                    if (cl_domain) {
                                        alpha_moe_set_detected_domain(cl_domain);
                                        // domain2 already set by classify_domain_fast()
                                        const char * d2 = alpha_moe_get_detected_domain2();
                                        (void)d2;
                                    } else {
                                        // Fallback to keyword scan
                                        auto domains = alpha_moe_detect_output_domain_top2(
                                            prompt_text.c_str(), (int)prompt_text.size());
                                        alpha_moe_set_detected_domain(domains.domain1);
                                        alpha_moe_set_detected_domain2(domains.domain2);
                                        if (domains.domain2)
                                            fprintf(stderr, " + %s", domains.domain2);
                                        fprintf(stderr, "\n");
                                    }
                                }
                            } else {
                                // Classify: set domain to "classify" for classify experts
                                alpha_moe_set_detected_domain("classify");
                            }

                            // Exit draft mode from previous request's generation.
                            // The scheduler was reserved for LUT graph (ne[2]=K).
                            // Callback graph needs ne[2]=n_expert (larger buffers).
                            // Reset sched only on first transition to establish large
                            // buffers. After that, LUT graph fits in the callback-size
                            // buffers (K < n_expert), so no reset needed.
                            if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                                alpha_moe_optimizer_exit_draft_mode(moe_opt);
                                static bool first_exit = true;
                                if (first_exit) {
                                    llama_context_reset_sched(ctx);
                                    first_exit = false;
                                }
                            }

                            // LUT prefill: use when expert cache exists and not cpu_moe/classify.
                            // Use profile-first LUT prefill only when K >= K_min (good coverage).
                            // When K < K_min, use cpu-moe for prompt (all experts via mmap)
                            // to ensure correct output. LUT with K < K_min = garbage.
                            int cur_K_min = alpha_moe_optimizer_get_n_expert(moe_opt) / 4;
                            bool use_profile_prompt = !force_cpu_moe &&
                                alpha_moe_optimizer_get_K(moe_opt) >= cur_K_min;

                            if (use_profile_prompt) {
                                // LUT prefill with domain experts.
                                // Pin domain experts into cache, but don't enter draft mode yet.
                                // Draft mode changes graph topology (LUT vs callback). If we
                                // enter here, the multi-token prompt batch later must exit draft
                                // + reset_sched, which leaks VRAM on every request.
                                // Instead: pin experts (DMA only), keep callback mode for prompt,
                                // let apply_warmup enter draft after prompt is done.
                                std::string model_dir = std::filesystem::path(params_base.model.path).parent_path().string();
                                alpha_moe_optimizer_pin_domain_experts(moe_opt, model_dir.c_str());
                                alpha_mole_set_argsort_persist(true);
                                slot.moe_lut_prefill = true;

                                int n_expert_used = alpha_moe_optimizer_get_n_expert_used(moe_opt);
                                int prefill_topk = std::max(4, n_expert_used - 2);
                                alpha_moe_optimizer_set_topk(moe_opt, prefill_topk);

                                // TODO: sparse prefill
                            } else {
                                // cpu-moe prompt: all experts via mmap
                                if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                                    alpha_moe_optimizer_exit_draft_mode(moe_opt);
                                }
                                alpha_moe_optimizer_disable_cache(moe_opt);
                                alpha_mole_set_argsort_persist(true);
                                alpha_moe_optimizer_prefetch_prompt_experts(moe_opt);
                            }
                        }

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, head_p, head_c)) {
                                                // Hybrid model: partial removal failed (recurrent state is monolithic).
                                                // Fall back to full clear - reprocess entire prompt.
                                                SLT_WRN(slot, "%s", "cache reuse failed (hybrid model) - clearing memory\n");
                                                slot.prompt_clear(true);
                                                slot.n_prompt_tokens_cache = 0;
                                                n_past = 0;
                                                break;
                                            }
                                            llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // note: when n_swa == 0, the model does not use SWA
                            const auto n_swa = std::max(0, llama_model_n_swa(model));

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa);

                            if (n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min, n_swa);

                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        const size_t checkpoint_size = it->data.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                                        } else {
                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.data.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_INF(slot, "n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, p0, -1)) {
                        SLT_WRN(slot, "failed to truncate tokens with position >= %d - clearing the memory\n", p0);

                        slot.prompt_clear(true);

                        // there is no common part left
                        slot.n_prompt_tokens_cache = 0;
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA and we are not using `swa_full`
                    do_checkpoint = do_checkpoint && (
                            (slot.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) ||
                            (llama_model_n_swa(model) > 0 && !params_base.swa_full));

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // Encoder-decoder models (T5, BART, etc.): encode prompt, then
                        // replace batch with decoder start token for the decode loop.
                        if (llama_model_has_encoder(model)) {
                            // Encode all prompt tokens
                            const int ret_enc = llama_encode(ctx, batch);
                            if (ret_enc != 0) {
                                SLT_ERR(slot, "llama_encode() failed, ret = %d\n", ret_enc);
                                send_error(slot, "Encoder failed.");
                                slot.release();
                                slot.prompt_clear(false);
                                continue; // skip to next slot
                            }

                            SLT_INF(slot, "encoder done, n_tokens = %d\n", batch.n_tokens);

                            // Clear decoder KV cache - llama_encode() stores encoder
                            // output internally for cross-attention, but also creates
                            // position entries in the memory that conflict with the
                            // decoder starting at position 0.
                            // prompt_clear(true) calls llama_memory_seq_rm + clears tokens.
                            slot.prompt_clear(true);

                            // Get decoder start token
                            llama_token dec_start = llama_model_decoder_start_token(model);
                            if (dec_start == LLAMA_TOKEN_NULL) {
                                dec_start = llama_vocab_bos(vocab);
                            }

                            // Replace batch with decoder start token at position 0
                            common_batch_clear(batch);
                            common_batch_add(batch, dec_start, 0, { slot.id }, true);
                            slot.prompt.tokens.push_back(dec_start);
                        }

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                        SLT_INF(slot, "prompt processing done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);
                    } else {
                        if (slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch) {
                            // near the end of the prompt
                            do_checkpoint = do_checkpoint && true;
                        } else {
                            // only do non-end checkpoints if the "checkpoint every n tokens" option is set
                            do_checkpoint = do_checkpoint && params_base.checkpoint_every_nt > 0;

                            if (do_checkpoint) {
                                llama_pos last_checkpoint = 0;
                                if (!slot.prompt.checkpoints.empty()) {
                                    last_checkpoint = slot.prompt.checkpoints.back().n_tokens;
                                }

                                do_checkpoint = do_checkpoint && slot.prompt.n_tokens() - batch.n_tokens - last_checkpoint >= params_base.checkpoint_every_nt;

                                if (do_checkpoint) {
                                    SLT_INF(slot, "%d tokens since last checkpoint at %d, creating new checkpoint during processing at position %d\n", params_base.checkpoint_every_nt, last_checkpoint, slot.prompt.n_tokens());
                                }
                            }
                        }

                        SLT_INF(slot, "prompt processing progress, n_tokens = %d, batch.n_tokens = %d, progress = %f\n", slot.prompt.n_tokens(), batch.n_tokens, (float) slot.prompt.n_tokens() / slot.task->n_tokens());
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                    // no need for empty or small checkpoints
                    do_checkpoint = do_checkpoint && (pos_min >= 0 && slot.prompt.n_tokens() >= 64);

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || slot.prompt.n_tokens() - n_tokens_cur > slot.prompt.checkpoints.back().n_tokens + 64);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx, slot_batched->task->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // Alpha-MoLE: if somehow still in draft mode during multi-token prompt,
        // exit it. This should rarely happen now - pin_domain_experts exits draft
        // immediately to avoid the topology change that caused ~134 MiB/req leak.
        // Safety net only.
        bool moe_was_in_draft = false;
        if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
            batch.n_tokens > 1 && alpha_moe_optimizer_is_draft_active(moe_opt) &&
            !alpha_moe_optimizer_is_lut_prefill(moe_opt)) {
            alpha_moe_optimizer_exit_draft_mode(moe_opt);
            alpha_moe_optimizer_set_callback_active(moe_opt, true);
            llama_context_reset_sched(ctx);
            moe_was_in_draft = true;
        }

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            // Encoder-decoder models: use llama_encode() for prompt token batches.
            // The final prompt batch is already encoded at DONE_PROMPT above and
            // replaced with the decoder start token, so this only fires for
            // intermediate prompt batches (prompt longer than n_batch).
            bool enc_batch = false;
            if (llama_model_has_encoder(model)) {
                for (const auto & slot : slots) {
                    if (slot.state == SLOT_STATE_PROCESSING_PROMPT) {
                        enc_batch = true;
                        break;
                    }
                }
            }

            // LUT-only fast path: disable eval callback after warmup
            // The eval callback fires for ~1000 tensors per forward pass, costing ~1ms/token
            // even when all experts are cached. After warmup (first 5 tokens), disable it
            // and rely on the GPU LUT for expert remapping. Re-enable every 20 tokens to
            // refresh the cache for any routing changes.
            if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
                moe_lut_active && !enc_batch && batch_view.n_tokens == 1) {
                // Count total generated tokens across all active slots
                int total_gen = 0;
                for (auto & s : slots) {
                    if (s.state == SLOT_STATE_GENERATING) total_gen += s.n_decoded;
                }
                bool warmup_done = total_gen >= 5;
                bool refresh_needed = (total_gen % 20) == 0;

                if (warmup_done && !refresh_needed) {
                    // LUT-only: disable callback for this decode
                    llama_set_eval_callback(ctx, nullptr, nullptr);
                } else if (warmup_done && refresh_needed) {
                    // Refresh: re-enable callback for one token to update cache
                    llama_set_eval_callback(ctx,
                        alpha_moe_optimizer_get_eval_callback(),
                        alpha_moe_optimizer_get_eval_data(moe_opt));
                    alpha_moe_optimizer_set_callback_active(moe_opt, true);
                }
                // During warmup (first 5 tokens): callback stays active
            }

            const int ret = enc_batch ? llama_encode(ctx, batch_view)
                                      : llama_decode(ctx, batch_view);

            // Async overlap: the GPU is decoding, do CPU work now
            {
                // 1. Grammar precompute for all generating slots
                for (auto & slot : slots) {
                    if (slot.state == SLOT_STATE_GENERATING &&
                        common_sampler_has_grammar(slot.smpl.get())) {
                        slot.precomputed_grammar = common_sampler_precompute_grammar(slot.smpl.get());
                    }
                }

                // 2. Radix KV snapshots for slots that just finished prompt
                for (auto & slot : slots) {
                    if (slot.state == SLOT_STATE_DONE_PROMPT &&
                        radix_kv_cache.memory_budget() > 0 &&
                        slot.prompt.tokens.size() > 50) {
                        auto tokens_vec = slot.prompt.tokens.get_tokens();
                        const size_t n_prompt = slot.task->n_tokens();
                        if (n_prompt > 0 && n_prompt <= tokens_vec.size()) {
                            const size_t state_size = llama_state_seq_get_size_ext(ctx, slot.id, 0);
                            if (state_size > 0 && state_size < radix_kv_cache.memory_budget() / 4) {
                                std::vector<uint8_t> kv_data(state_size);
                                llama_state_seq_get_data_ext(ctx, kv_data.data(), state_size, slot.id, 0);
                                radix_kv_cache.insert(tokens_vec.data(), n_prompt,
                                                       kv_data.data(), kv_data.size());
                            }
                        }
                    }
                }
            }
            // end async overlap

            // After refresh decode: update LUT from potentially-changed cache, then disable callback
            if (moe_opt && moe_lut_active && !enc_batch && batch_view.n_tokens == 1) {
                int total_gen = 0;
                for (auto & s : slots) {
                    if (s.state == SLOT_STATE_GENERATING) total_gen += s.n_decoded;
                }
                if (total_gen >= 5 && (total_gen % 20) == 0) {
                    // Just did a refresh - update LUT and disable callback for next 19 tokens
                    alpha_moe_optimizer_update_lut(moe_opt);
                    llama_set_eval_callback(ctx, nullptr, nullptr);
                }
            }

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx);

            // Alpha-MoLE: Per-token cache warming during cpu-moe prefill.
            // After each decode, load fired experts into cache via reload_from_argsort.
            // This progressively warms the cache with prompt-specific experts,
            // replacing tiling (24s) with per-token DMA (~2s total).
            if (moe_opt && !alpha_moe_optimizer_is_draft_active(moe_opt) &&
                alpha_moe_optimizer_is_moe(moe_opt) && alpha_moe_optimizer_get_K(moe_opt) > 0) {
                bool has_prompt_slot = false;
                for (auto & slot : slots) {
                    if (slot.state == SLOT_STATE_PROCESSING_PROMPT ||
                        slot.state == SLOT_STATE_DONE_PROMPT) {
                        has_prompt_slot = true;
                        break;
                    }
                }
                if (has_prompt_slot) {
                    // Enable cache temporarily to load experts, then disable for next cpu-moe token
                    alpha_moe_optimizer_enable_cache(moe_opt);
                    int loaded = alpha_moe_optimizer_reload_from_argsort(moe_opt);
                    alpha_moe_optimizer_disable_cache(moe_opt);
                    static int resync_log = 0;
                    if (resync_log < 5 || loaded > 0) {
                        fprintf(stderr, "[RESYNC] loaded %d experts into cache\n", loaded);
                        resync_log++;
                    }
                }
            }

            // Alpha-MoLE: Per-token progressive cache warming during LUT prefill.
            // After each ubatch decode, check if the token had high miss rate.
            // If so, re-decode with cpu-moe and load actual experts into cache.
            if (moe_opt) {
                for (auto & slot : slots) {
                    if (!slot.moe_lut_prefill) continue;
                    // Check all prompt-processing states
                    if (slot.state != SLOT_STATE_PROCESSING_PROMPT &&
                        slot.state != SLOT_STATE_DONE_PROMPT) continue;

                    int miss = alpha_moe_optimizer_get_miss_layers(moe_opt);
                    int total = alpha_moe_optimizer_get_total_layers(moe_opt);
                    fprintf(stderr, "[PREFILL-TOK] slot=%d miss=%d/%d\n", slot.id, miss, total);
                    if (total <= 0) continue;

                    float miss_rate = (float)miss / total;
                    if (miss_rate > 0.5f) {
                        // Bad token - switch entire prefill to cpu-moe.
                        // Hybrid memory (recurrent state) doesn't support single-position
                        // seq_rm, so we must re-do full prefill with cpu-moe.
                        int view_idx = batch_view.n_tokens - 1;
                        llama_pos fix_pos = batch_view.pos[view_idx];

                        fprintf(stderr, "[PREFILL-FIX] pos=%d miss=%d/%d (%.0f%%) — full cpu-moe re-prefill\n",
                                (int)fix_pos, miss, total, miss_rate * 100.0f);

                        // 1. Exit draft + disable cache for cpu-moe
                        alpha_moe_optimizer_exit_draft_mode(moe_opt);
                        alpha_moe_optimizer_disable_cache(moe_opt);
                        slot.moe_lut_prefill = false;
                        alpha_moe_optimizer_set_lut_prefill(moe_opt, false);
                        alpha_moe_optimizer_set_topk(moe_opt, 0);

                        // 2. Clear all KV (LUT-computed = garbage)
                        auto * mem = llama_get_memory(ctx);
                        if (mem) llama_memory_seq_rm(mem, slot.id, -1, -1);

                        // 3. Reset slot for full cpu-moe re-prefill
                        slot.prompt_clear(true);
                        slot.n_prompt_tokens_processed = 0;

                        // Break out of per-slot loop. The batch loop will continue,
                        // and the slot will re-enter prompt processing on the next
                        // iteration with cache disabled -> cpu-moe -> correct KV.
                        break;
                    }
                }
            }

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    // Alpha-MoLE: LUT prefill quality check + auto-repair
                    // After LUT prefill, check if expert coverage was sufficient.
                    // If bad tokens found, re-prefill from that position with cpu-moe.
                    // Reset prefill mode + topk when entering generation
                    if (moe_opt) {
                        if (slot.moe_lut_prefill) {
                            slot.moe_lut_prefill = false;
                            alpha_moe_optimizer_set_lut_prefill(moe_opt, false);
                        }
                        // Alpha-MoLE expert-skip probe: ALPHA_GEN_TOPK=K forces
                        // generation-time top-K (drop lowest-weight experts + renorm via
                        // the standard norm_w path). 0/unset = model default (unchanged).
                        // The model builders ignore values >= n_expert_used, so no clamp.
                        static int gen_topk_env = []() {
                            const char * e = getenv("ALPHA_GEN_TOPK");
                            int v = e ? atoi(e) : 0;
                            return v < 0 ? 0 : v;
                        }();
                        alpha_moe_optimizer_set_topk(moe_opt, gen_topk_env); // 0 = default; env overrides for expert-skip

                        // For reactive prefill: keep draft=OFF and callback active.
                        // Warmup's enter_draft -> LUT gen would lose the reactive
                        // experts loaded during prefill. Instead, keep callback for
                        // generation too - reactive DMA for both prefill and gen.
                        // This sacrifices LUT gen speed but ensures correct experts.
                    }
                    // Per-prompt QC disabled - per-token progressive fix handles quality
                    if (false && moe_opt && slot.moe_lut_prefill &&
                        slot.n_prompt_tokens_processed > 0) {
                        int first_bad = alpha_moe_optimizer_check_prefill_quality(
                            moe_opt, slot.n_prompt_tokens_processed);
                        if (first_bad >= 0) {
                            // Bad tokens detected - need cpu-moe re-prefill.
                            // Instead of calling llama_decode directly (which causes logit
                            // index mismatch), reset the slot to re-process the entire
                            // prompt via the normal path.

                            // 1. Exit draft + disable cache -> next prefill uses cpu-moe
                            if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                                alpha_moe_optimizer_exit_draft_mode(moe_opt);
                            }
                            alpha_moe_optimizer_disable_cache(moe_opt);

                            // 2. Clear all KV (garbage from LUT prefill)
                            auto * mem = llama_get_memory(ctx);
                            if (mem) {
                                llama_memory_seq_rm(mem, slot.id, -1, -1);
                            }

                            // 3. Reset slot to re-process prompt from scratch
                            slot.prompt_clear(true);
                            slot.n_prompt_tokens_processed = 0;
                            slot.state = SLOT_STATE_PROCESSING_PROMPT;

                            // Reset topk before cpu-moe re-prefill (use full expert count)
                            alpha_moe_optimizer_set_topk(moe_opt, 0);

                            // Slot will re-enter the prompt processing loop on next iteration
                            // with cache disabled -> cpu-moe prefill -> correct KV
                            continue;
                        }

                        // LUT prefill passed QC - topk already reset above
                    }

                    // Alpha-MoLE: portfolio-driven warmup after prompt processing
                    // Skip for ALL_GPU models (K=0, all experts on GPU - no cache to warm)
                    // Skip for cpu_moe requests (repair pass - keep cache disabled, use all experts)
                    if (g_cpu_moe_active && moe_opt) {
                        // Exit draft mode (unwires LUT from layers) + disable cache
                        // This forces all experts via mmap for the entire request
                        if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                            alpha_moe_optimizer_exit_draft_mode(moe_opt);
                        }
                        alpha_moe_optimizer_disable_cache(moe_opt);

                        // Clear garbage KV from previous LUT prefill.
                        // Without this, cache_prompt=true reuses the corrupted KV
                        // (n_past = full prompt) and skips re-prefill -> still empty.
                        auto * mem = llama_get_memory(ctx);
                        if (mem) {
                            llama_memory_seq_rm(mem, slot.id, -1, -1);
                        }
                        slot.prompt_clear(true);

                        // Keep g_cpu_moe_active=true through generation (don't re-enable LUT)
                        // Reset happens at slot release, not here.
                    } else if (moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
                        alpha_moe_optimizer_get_K(moe_opt) > 0) {
                        // Skip warmup for classify sub-requests - their argsort data is from
                        // the classify prompt ("Classify into one of these domains..."), not
                        // the user's actual question. Loading those experts pollutes the cache
                        // with useless classify-specific experts.
                        if (g_classify_active) {
                            g_classify_active = false;  // reset after use
                        } else {
                        // Domain-Expert Pinning: detect output domain from prompt keywords
                        // before warmup so domain experts get pinned into cache
                        {
                            const char * domain = nullptr;
                            if (slot.task && slot.task->tokens.size() > 0) {
                                std::string prompt_text = slot.task->tokens.detokenize(ctx, false);
                                domain = alpha_moe_detect_output_domain(
                                    prompt_text.c_str(), (int)prompt_text.size());
                            } else {
                            }
                            alpha_moe_set_detected_domain(domain);
                        }

                        std::string model_dir = std::filesystem::path(params_base.model.path).parent_path().string();
                        alpha_moe_optimizer_apply_warmup(moe_opt, ctx, &moe_portfolio, model_dir.c_str());
                        int extra_loaded = alpha_moe_optimizer_reload_from_argsort(moe_opt);
                        (void)extra_loaded;

                        // Adaptive cpu-moe cache warming for K < K_min:
                        // Generate initial tokens with cpu-moe (all experts correct),
                        // after each token check how many new experts were loaded into
                        // cache via reload_from_argsort(). When loaded==0, the cache
                        // has all needed experts -> switch to LUT for fast generation.
                        // Adapts automatically: factual Q -> 1-2 warmup tokens,
                        // creative Q -> 5+ warmup tokens. Warmup tokens are the response.
                        {
                            int K_cur = alpha_moe_optimizer_get_K(moe_opt);
                            int n_expert = alpha_moe_optimizer_get_n_expert(moe_opt);
                            int K_min_cur = n_expert / 4;
                            if (K_cur > 0 && K_cur < K_min_cur) {
                                // Low K: callback adds 150ms/tok overhead (GPU sync per tensor node).
                                // LUT at K<K_min generates garbage (not enough cached experts).
                                // Best strategy: disable callback + cpu-moe dispatch = standard speed.
                                // Profiles still loaded, classification still works, just no LUT/callback.
                                if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                                    alpha_moe_optimizer_exit_draft_mode(moe_opt);
                                }
                                alpha_moe_optimizer_disable_cache(moe_opt);
                                llama_set_eval_callback(ctx, nullptr, nullptr);
                                slot.force_cpu_moe = true;
                                slot.moe_adaptive_warmup = false;
                            } else if (K_cur >= K_min_cur && K_cur < n_expert) {
                                // Mid K: on hybrid models (recurrent + attention) the LUT garbles output
                                // because wrong expert mappings corrupt the recurrent state, so force
                                // cpu-moe there. Pure MoE models are fine with the LUT at mid K.
                                bool is_hybrid_model = (llama_get_memory(ctx) != nullptr &&
                                    dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx)) != nullptr);
                                // Mid-K: force cpu-moe (no callback, no LUT).
                                // At K >= K_min, callback disabled for speed (150ms overhead).
                                // Without callback, LUT garbles. cpu-moe via mmap is correct + fast.
                                if (alpha_moe_optimizer_is_draft_active(moe_opt)) {
                                    alpha_moe_optimizer_exit_draft_mode(moe_opt);
                                }
                                alpha_moe_optimizer_disable_cache(moe_opt);
                                slot.force_cpu_moe = true;
                                slot.moe_adaptive_warmup = false;
                            }
                        }

                        // Per-profile readback_interval (same logic as CLI)
                        const char * pname = alpha_moe_optimizer_get_profile_name();
                        if (pname && strcmp(pname, "none") != 0) {
                            if (strcmp(pname, "factual") == 0 || strcmp(pname, "code") == 0 ||
                                strcmp(pname, "translate") == 0) {
                                moe_portfolio.readback_interval = 0;
                            } else if (strcmp(pname, "chat") == 0 || strcmp(pname, "instruct") == 0) {
                                moe_portfolio.readback_interval = 30;
                            } else if (strcmp(pname, "creative") == 0 || strcmp(pname, "knowledge") == 0) {
                                moe_portfolio.readback_interval = 15;
                            } else {
                                moe_portfolio.readback_interval = 20;
                            }
                        }
                        // If profile=none (no profiles loaded), use default refresh
                        if (!pname || strcmp(pname, "none") == 0) {
                            moe_portfolio.readback_interval = 20;
                        }
                        // Detect long-gen: need progressive CPU sync for quality.
                        // Skip sync for: factual/code/translate profiles (short answers),
                        // or when max_tokens is small (user expects short output).
                        slot.moe_long_gen = true;
                        if (pname && (strcmp(pname, "factual") == 0 || strcmp(pname, "code") == 0 ||
                                     strcmp(pname, "translate") == 0)) {
                            slot.moe_long_gen = false;
                        }
                        // When profile=none (no profiles loaded), use max_tokens as heuristic
                        if ((!pname || strcmp(pname, "none") == 0) &&
                            slot.task->params.n_predict >= 0 && slot.task->params.n_predict <= 30) {
                            slot.moe_long_gen = false;
                        }
                        slot.moe_pass = 0;
                        } // end else (non-classify warmup)
                    }

                    // Reset cpu_moe flag after warmup decision.
                    // If repair kept it true, warmup was skipped -> generation uses cpu-moe.
                    // Re-enable cache+draft for next request (not this one).
                    if (g_cpu_moe_active && moe_opt) {
                        g_cpu_moe_active = false;
                    }

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec.get(), slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;
                slot.i_batch = -1;

                // MoE Self-Speculative Draft-Verify
                // For MoE models with expert cache: draft N tokens with LUT (fast),
                // per-token gates catch obvious errors during draft, then batch verify
                // all N tokens with correct experts. Accept up to first mismatch.
                // For non-MoE or ALL_GPU: standard single-token path (no overhead).
                // ALPHA_BRANCH_RECOVER needs the standard single-token path (where br_sample runs), so
                // force self-spec off when it is enabled; otherwise the branch policy would be a silent
                // no-op on MoE models whose LUT draft is active.
                bool moe_self_spec = !br_enabled && moe_opt && alpha_moe_optimizer_is_moe(moe_opt) &&
                                     alpha_moe_optimizer_is_draft_active(moe_opt);

                if (moe_self_spec) {
                    static constexpr int DRAFT_N = 8;

                    // Enable argsort persistence from cycle 2+ so reload_from_argsort
                    // can read valid data after this cycle's draft decodes.
                    if (slot.moe_gen_tokens >= DRAFT_N) {
                        if (!alpha_mole_get_argsort_persist()) {
                            alpha_mole_set_argsort_persist(true);
                        }
                    }

                    // Save sampler state for rollback on rejection
                    common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                    // Sample first token from current batch logits
                    llama_token first_tok = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);
                    common_sampler_accept(slot.smpl.get(), first_tok, true);

                    // Draft phase: generate N tokens with LUT
                    std::vector<llama_token> draft;
                    draft.push_back(first_tok);

                    // Track position for proper KV management
                    int n_prompt_before = slot.prompt.n_tokens();

                    // Add first token to prompt tracking
                    slot.prompt.tokens.push_back(first_tok);

                    for (int di = 1; di < DRAFT_N; di++) {
                        // Context overflow check: stop draft before crashing
                        if (slot.prompt.n_tokens() >= slot.n_ctx - 2) {
                            SLT_WRN(slot, "draft stopped: context full (%d/%d tokens)\n",
                                    slot.prompt.n_tokens(), slot.n_ctx);
                            break;
                        }
                        // Decode with explicit position using common_batch_add
                        llama_batch db = {};
                        db = llama_batch_init(1, 0, 1);
                        common_batch_add(db, draft.back(), slot.prompt.tokens.pos_next() - 1, { slot.id }, true);
                        int rc = llama_decode(ctx, db);
                        llama_batch_free(db);
                        if (rc != 0) {
                            SLT_WRN(slot, "draft decode failed: rc=%d\n", rc);
                            break;
                        }

                        llama_token dt = common_sampler_sample(slot.smpl.get(), ctx, -1);
                        common_sampler_accept(slot.smpl.get(), dt, true);

                        // Per-token gate: duplicate word -> stop draft cycle
                        // Hybrid models (Qwen3.5) don't support partial KV rewind
                        // (recurrent state can't be partially removed). Instead of
                        // CPU sync + re-decode, just end the draft cycle early.
                        // Two-pass refinement will fix duplicates in Pass 2.
                        if (dt == draft.back()) {
                            char dup_text[128] = {};
                            int dup_len = llama_token_to_piece(vocab, dt, dup_text, sizeof(dup_text), 0, true);
                            bool is_word = (dup_len >= 2);
                            for (int ci = 0; is_word && ci < dup_len; ci++) {
                                unsigned char c = (unsigned char)dup_text[ci];
                                if (!isalpha(c) && c != ' ') is_word = false;
                            }
                            if (is_word) {
                                // Don't add the duplicate - end draft cycle
                                break;
                            }
                        }

                        // Per-token gate: n-gram repetition (check history + draft combined)
                        // "the atmosphere of the atmosphere" = 4-gram repeat
                        {
                            // Build combined view: lut_history + current draft
                            // Only check last 100 tokens for efficiency
                            std::vector<llama_token> combined;
                            size_t hist_start = slot.lut_history.size() > 100 ? slot.lut_history.size() - 100 : 0;
                            for (size_t hi = hist_start; hi < slot.lut_history.size(); hi++)
                                combined.push_back(slot.lut_history[hi]);
                            for (auto & d : draft) combined.push_back(d);
                            combined.push_back(dt);

                            if (combined.size() >= 6) {
                                size_t n = combined.size();
                                bool rep = false;
                                // Check 4-gram repetition
                                if (n >= 8) {
                                    llama_token g0 = combined[n-4], g1 = combined[n-3],
                                                g2 = combined[n-2], g3 = combined[n-1];
                                    for (size_t j = 0; j + 3 < n - 4; j++) {
                                        if (combined[j]==g0 && combined[j+1]==g1 &&
                                            combined[j+2]==g2 && combined[j+3]==g3) {
                                            rep = true; break;
                                        }
                                    }
                                }
                                // Check 3-gram repetition (catches phrase-level repeats
                                // like "the scattering of" at very low K coverage)
                                if (!rep && n >= 6) {
                                    llama_token g0 = combined[n-3], g1 = combined[n-2],
                                                g2 = combined[n-1];
                                    // Only flag if at least one token is a word (>=3 chars)
                                    // to avoid false positives on short function words
                                    bool has_word = false;
                                    llama_token gram[3] = {g0, g1, g2};
                                    for (int gi = 0; gi < 3; gi++) {
                                        char tbuf[64] = {};
                                        int tlen = llama_token_to_piece(vocab, gram[gi], tbuf, sizeof(tbuf), 0, true);
                                        if (tlen >= 3) { has_word = true; break; }
                                    }
                                    if (has_word) {
                                        for (size_t j = 0; j + 2 < n - 3; j++) {
                                            if (combined[j]==g0 && combined[j+1]==g1 &&
                                                combined[j+2]==g2) {
                                                rep = true; break;
                                            }
                                        }
                                    }
                                }
                                if (rep) {
                                    // Repetition detected - stop generation.
                                    // Cache resync doesn't help (same context -> same experts -> same loop).
                                    // KV rewind this repeated token (may fail on hybrid models - OK,
                                    // EOS injection below will end generation regardless)
                                    llama_memory_seq_rm(llama_get_memory(ctx), slot.id,
                                            slot.prompt.tokens.pos_next() - 1, -1);
                                    // Inject EOS to stop generation cleanly
                                    dt = llama_vocab_eot(vocab);
                                    if (dt == LLAMA_TOKEN_NULL) dt = llama_vocab_eos(vocab);
                                    draft.push_back(dt);
                                    slot.prompt.tokens.push_back(dt);
                                    break;
                                }
                            }
                        }

                        // Per-token gate: round-trip tokenizer check (every 6 tokens)
                        // Catches BPE corruption like "Raymersion"
                        if (draft.size() >= 6 && (draft.size() % 6) == 0) {
                            // Decode last 6 draft tokens to text, re-tokenize, compare
                            std::string chunk;
                            size_t start = draft.size() - 6;
                            for (size_t ri = start; ri < draft.size(); ri++) {
                                char buf[128] = {};
                                int len = llama_token_to_piece(vocab, draft[ri], buf, sizeof(buf), 0, true);
                                if (len > 0) chunk.append(buf, len);
                            }
                            chunk.append(1, '\0'); // for re-tokenization safety
                            chunk.pop_back();
                            if (!chunk.empty()) {
                                int n_re = -llama_tokenize(vocab, chunk.c_str(), chunk.size(), nullptr, 0, false, true);
                                if (n_re > 0) {
                                    std::vector<llama_token> re_toks(n_re);
                                    llama_tokenize(vocab, chunk.c_str(), chunk.size(), re_toks.data(), re_toks.size(), false, true);
                                    // Compare: if different count or any mismatch -> BPE corruption
                                    bool mismatch = (n_re != 6);
                                    if (!mismatch) {
                                        for (int ri = 0; ri < 6; ri++) {
                                            if (re_toks[ri] != draft[start + ri]) { mismatch = true; break; }
                                        }
                                    }
                                    if (mismatch) {
                                        // BPE corruption detected - end draft cycle.
                                        // Can't KV rewind on hybrid models (recurrent state).
                                        // Two-pass refinement will fix in Pass 2.
                                        break;
                                    }
                                }
                            }
                        }

                        draft.push_back(dt);
                        slot.prompt.tokens.push_back(dt);

                        if (llama_vocab_is_eog(vocab, dt)) break;
                    }

                    int n_draft = (int)draft.size();

                    int n_accepted = n_draft;

                    // Output accepted tokens
                    const int64_t t_current = ggml_time_us();
                    bool slot_released = false;
                    for (int oi = 0; oi < n_accepted; oi++) {
                        slot.n_decoded += 1;
                        slot.moe_gen_tokens++;
                        slot.lut_history.push_back(draft[oi]);
                        slot.moe_prev_tok = draft[oi];

                        if (slot.n_decoded == 1) {
                            slot.t_start_generation = t_current;
                            slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                            metrics.on_prompt_eval(slot);
                        }
                        slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                        completion_token_output result;
                        result.tok = draft[oi];
                        result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
                        result.prob = 1.0f;

                        if (!process_token(result, slot)) {
                            slot.print_timings();
                            send_final_response(slot);
                            metrics.on_prediction(slot);
                            slot.release();
                            slot_released = true;
                            break;
                        }
                    }

                    continue;
                }

                // Standard single-token path (non-MoE / ALL_GPU)
                // Apply precomputed grammar mask (overlapped with GPU forward pass)
                if (slot.precomputed_grammar.has_mask) {
                    common_sampler_apply_precomputed_grammar(slot.smpl.get(), slot.precomputed_grammar);
                    slot.precomputed_grammar.has_mask = false;  // consumed
                }
                // ALPHA_BRANCH_RECOVER: greedy-by-default + sparse entropy-gated nucleus branches.
                // Flag-off => unchanged sampler-chain path (byte-identical). We still accept(id) below so
                // penalty/grammar/context state stays consistent for the next token.
                llama_token id = br_enabled
                    ? br_sample(slot, tok_idx)
                    : common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);

                // ALPHA_LOG_EXPERT_USAGE: tap this decode token's router selection (read-only, no-op
                // unless enabled). Placed after sampling - the argsort holds this forward's routing.
                if (eu_log_enabled) eu_accumulate(slot);

                // Two-pass: record logit margin + expert miss for this token
                if (slot.twopass_enabled && moe_opt) {
                    const float * logits = llama_get_logits_ith(ctx, tok_idx);
                    if (logits) {
                        int n_vocab = llama_vocab_n_tokens(vocab);
                        // Find top-2 logits for margin
                        float top1 = -1e9f, top2 = -1e9f;
                        for (int v = 0; v < n_vocab; v++) {
                            if (logits[v] > top1) { top2 = top1; top1 = logits[v]; }
                            else if (logits[v] > top2) { top2 = logits[v]; }
                        }
                        float margin = top1 - top2;
                        float miss_rate = alpha_moe_optimizer_get_last_miss_rate(moe_opt);
                        int miss_layers = (int)(miss_rate * 48 + 0.5f);  // approximate
                        slot.twopass_records.push_back({id, margin, miss_layers});
                    }
                }

                common_sampler_accept(slot.smpl.get(), id, true);

                // Jump-forward: if grammar has only one valid next token,
                // emit it immediately without running the model.
                // Saves ~10% on tool call generation (deterministic JSON boilerplate).
                if (common_sampler_has_grammar(slot.smpl.get())) {
                    while (true) {
                        llama_token jf = common_sampler_jump_forward(slot.smpl.get());
                        if (jf == LLAMA_TOKEN_NULL) break;

                        common_sampler_accept(slot.smpl.get(), jf, true);
                        slot.n_decoded += 1;

                        completion_token_output jf_result;
                        jf_result.tok = jf;
                        jf_result.text_to_send = common_token_to_piece(slot.ctx, jf, accept_special_token(slot, jf));
                        jf_result.prob = 1.0f;

                        if (!process_token(jf_result, slot)) {
                            slot.print_timings();
                            send_final_response(slot);
                            metrics.on_prediction(slot);
                            slot.release();
                            break;
                        }
                    }
                }

                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f;

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                // Adaptive verify-skip: trust draft tokens directly (not for self-spec - exact)
                const bool slot_is_self_spec = slot.task && slot.task->params.speculative.self_spec_skip > 0;
                if (slot.spec_i_batch.empty() && slot.adaptive_skip_verify && !slot_is_self_spec) {
                    // Skip-verify mode: all draft tokens accepted without main model check.
                    // Periodically force a verify batch to detect quality drift.
                    bool force_verify_next = (slot.adaptive_verify_batches % slot.ADAPTIVE_VERIFY_EVERY == 0);
                    if (force_verify_next) {
                        // Next batch will do a full verify (adaptive_skip_verify stays true,
                        // but update_batch won't skip because we toggle temporarily)
                        SLT_DBG(slot, "adaptive: periodic verify check (batch %d)\n", slot.adaptive_verify_batches);
                    }

                    // Accept all draft tokens + sampled
                    llama_tokens accepted;
                    accepted.push_back(slot.sampled);
                    accepted.insert(accepted.end(), slot.spec_draft.begin(), slot.spec_draft.end());

                    // Feed accepted tokens to sampler so it stays in sync
                    for (auto tok : accepted) {
                        common_sampler_accept(slot.smpl.get(), tok, true);
                    }

                    if (slot.spec) {
                        common_speculative_accept(slot.spec.get(), n_draft);
                    }

                    // Update adaptive tracking - all accepted
                    slot.adaptive_verify_window += n_draft;
                    slot.adaptive_verify_accepted += n_draft;

                    slot.spec_draft = std::move(accepted);

                    SLT_DBG(slot, "adaptive skip: accepted all %zu draft tokens (batch %d)\n",
                            n_draft, slot.adaptive_verify_batches);
                } else {
                    // Standard verify path
                    // verify and try to accept the draft
                    {
                        common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                        GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                        auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx, slot.spec_i_batch, slot.spec_draft);
                        slot.spec_i_batch.clear();

                        SLT_DBG(slot, "%s: n_draft=%zu, accepted=%zu\n", __func__, slot.spec_draft.size(), accepted.size());

                        GGML_ASSERT(accepted.size() >= 1);

                        // Update adaptive tracking
                        slot.adaptive_verify_window += n_draft;
                        slot.adaptive_verify_accepted += accepted.size() - 1;

                        // check for partial draft acceptance
                        if (accepted.size() < slot.spec_draft.size() + 1) {
                            // Rejection detected - disable skip-verify
                            if (slot.adaptive_skip_verify) {
                                SLT_INF(slot, "adaptive: rejection detected (%zu/%zu), disabling skip-verify\n",
                                        accepted.size() - 1, n_draft);
                                slot.adaptive_skip_verify = false;
                                slot.adaptive_verify_window = 0;
                                slot.adaptive_verify_accepted = 0;
                            }

                            if (slot.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                                // partial acceptance is not supported by the context -> truncate the draft and restore the state
                                slot.spec_draft = std::move(accepted);

                                const auto & ckpt = slot.spec_ckpt;

                                SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n",
                                        ckpt.pos_min, ckpt.pos_max, ckpt.size());

                                const bool ss = slot.task && slot.task->params.speculative.self_spec_skip > 0;
                                if (ss) {
                                    // Layer-skip self-spec: recurrent state held in a GPU buffer
                                    // (spec_ckpt.data is a 1-byte placeholder), restore device-to-device;
                                    // snap_* is still the pre-draft state -> gpu_restore.
                                    llama_memory_gpu_restore(llama_get_memory(slot.ctx));
                                } else {
                                    const size_t n = llama_state_seq_set_data_ext(slot.ctx, ckpt.data.data(), ckpt.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                    if (n != ckpt.size()) {
                                        GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu, get_data_ext->%zu, set_data_ext->%zu",
                                                __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size(), ckpt.size(), n);
                                    }
                                }

                                llama_memory_seq_rm(llama_get_memory(slot.ctx), slot.id, ckpt.pos_max + 1, -1);

                                slot.prompt.tokens.keep_first(ckpt.n_tokens);
                                slot.smpl = std::move(smpl_save);

                                continue;
                            }

                            LOG_DBG("%s: partial acceptance: %zu < %zu\n", __func__, accepted.size(), slot.spec_draft.size());
                        }

                        if (slot.spec) {
                            common_speculative_accept(slot.spec.get(), accepted.size() - 1);
                        }

                        slot.spec_draft = std::move(accepted);
                    }

                    // Check if we should enable skip-verify.
                    // Never for layer-skip self-spec: the draft argmax often differs from the full
                    // model's, so trusting it would break exactness. Self-spec always verifies.
                    const bool is_self_spec = slot.task && slot.task->params.speculative.self_spec_skip > 0;
                    if (!is_self_spec &&
                        !slot.adaptive_skip_verify &&
                        slot.adaptive_verify_window >= slot.ADAPTIVE_WINDOW_SIZE) {
                        float rate = (float)slot.adaptive_verify_accepted / slot.adaptive_verify_window;
                        if (rate >= slot.ADAPTIVE_SKIP_THRESH) {
                            slot.adaptive_skip_verify = true;
                            slot.adaptive_verify_batches = 0;
                            SLT_INF(slot, "adaptive: enabling skip-verify (acceptance %.1f%% over %d tokens)\n",
                                    rate * 100.0f, slot.adaptive_verify_window);
                        }
                        // Reset window for next measurement
                        slot.adaptive_verify_window = 0;
                        slot.adaptive_verify_accepted = 0;
                    }
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.n_decoded += ids.size();
                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;
                slot.n_draft_total += n_draft;

                // add accepted tokens to the prompt
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                if (!llama_memory_seq_rm(llama_get_memory(slot.ctx), slot.id, slot.prompt.n_tokens(), -1)) {
                    SLT_WRN(slot, "%s", "draft accept seq_rm failed (hybrid model)\n");
                }

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

// Pipeline Mode: Standard loop with backend GPU sampling
// Uses the proven update_slots() loop but with backend samplers attached.
// Backend sampling runs temp/top-k/top-p on GPU - returns 4-byte token IDs
// instead of 512KB logits per slot. Eliminates GPU->CPU logits transfer.

void server_context::start_pipeline_loop() {
    auto & ctx_server = *impl;
    auto * ctx = ctx_server.ctx;
    auto & slots = ctx_server.slots;
    const int n_slots = (int)slots.size();

    SRV_INF("pipeline: starting async loop with %d slots\n", n_slots);

    // Setup backend samplers per slot
    int n_backend_samplers = 0;
    for (int i = 0; i < n_slots; i++) {
        llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
        cparams.no_perf = true;
        llama_sampler * chain = llama_sampler_chain_init(cparams);
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(42 + i));

        if (llama_set_sampler(ctx, (llama_seq_id)i, chain)) {
            n_backend_samplers++;
        } else {
            llama_sampler_free(chain);
        }
    }
    SRV_INF("pipeline: %d/%d backend samplers attached\n", n_backend_samplers, n_slots);

    bool use_backend_sampling = (n_backend_samplers == n_slots);
    if (!use_backend_sampling) {
        SRV_WRN("%s", "pipeline: backend sampling unavailable, falling back to CPU sampling\n");
    }

    // Backend samplers attached. Run standard loop with output thread.
    //
    // Architecture: Main thread does batch+decode+sample (GPU-bound).
    // Output thread does string conversion + stop word search + streaming (CPU-bound).
    // These run in parallel - GPU never waits for string operations.
    //
    // The output thread processes a concurrent queue of (slot_id, token_id) pairs.
    // Main thread pushes sampled tokens to the queue after each iteration.
    // Output thread pops and calls process_token + send_partial_response.

    // Pipeline mode: backend sampling + sync removal.
    // process_token modifies live slot state (stop, has_next_token) that batch building reads,
    // so output processing cannot run on a separate thread without copying that state.
    auto & params = ctx_server.params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model),
        /* model_n_params         */ llama_model_n_params(impl->model),
        /* model_size             */ llama_model_size(impl->model),
    };
}



corpus_tool_context server_context::get_corpus_tool_context() const {
    corpus_tool_context ctx;
    ctx.bm25       = &impl->chunk_bm25;
    ctx.entities   = &impl->chunk_entities;
    ctx.linker     = &impl->chunk_linker;
    ctx.extractor  = &impl->entity_extractor;
    ctx.model_name = impl->model_name;
    return ctx;
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        // VRAM debug: track free memory per request (rate-limited)
#ifdef ALPHA_INTERNAL
        {
            static int req_count = 0;
            static size_t vram_free_first = 0;
            req_count++;
            // Only check VRAM every 1000 requests (GPU driver call is expensive)
            if (req_count <= 3 || req_count % 1000 == 0) {
                size_t vram_free = 0, vram_total = 0;
                ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
                if (dev) ggml_backend_dev_memory(dev, &vram_free, &vram_total);
                if (req_count == 1) vram_free_first = vram_free;
                int64_t drift_kb = ((int64_t)vram_free_first - (int64_t)vram_free) / 1024;
                fprintf(stderr, "[VRAM] req=%d free=%zu/%zu MiB (%.1f%%) drift=%+lldKB\n",
                        req_count,
                        vram_free/(1024*1024), vram_total/(1024*1024),
                        100.0*vram_free/std::max(vram_total,(size_t)1),
                        (long long)drift_kb);
            }
        }
#endif

        // Evict stale expert pages before classify + main task
        // Prevents OOM on large models (397B 166GB on 62GB RAM).
        if (auto * moe = ctx_server.moe_opt_public()) {
            alpha_moe_optimizer_evict_expert_pages(moe);
        }

        // LUT Domain Classification
        // Before posting the main task, use the 35B itself to classify the prompt
        // into a domain. This runs a short classify prompt through LUT with
        // classify-specific experts (from classify_dp.json), then sets the
        // detected domain so the main task's warmup loads the right experts.
        //
        // Flow: load classify experts -> LUT classify (1-3 tokens) -> load domain experts -> main task
        bool is_pass2 = json_value(data, "_is_pass2", false);
        bool is_classify = json_value(data, "_is_classify", false);
        bool is_cpu_moe = json_value(data, "_cpu_moe", false);
        std::string original_msg = json_value(data, "_original_user_msg", std::string());

        // cpu_moe: set global flag so warmup skips LUT re-enable
        ctx_server.g_cpu_moe_active = is_cpu_moe;
        ctx_server.g_classify_active = is_classify;  // classify uses cpu-moe prefill
        // LUT classify is disabled: the extra decode cycle runs low-VRAM devices out of memory and
        // adds a large per-request cost (KV state restore). Keyword detection is used instead, and
        // early-EOS repair catches prompts where it picks the wrong domain.
        if (false && ctx_server.is_moe_lut_active() && !is_pass2 && !is_classify && !is_cpu_moe &&
            alpha_moe_optimizer_has_domain_profile("classify") && !original_msg.empty() &&
            ctx_server.get_moe_gpu_vram_mb() >= 12000) {

            // Set domain to "classify" so warmup pins classify experts
            alpha_moe_set_detected_domain("classify");

            // Build classify prompt with dynamic domain list from loaded profiles
            std::string domain_list;
            int n_dp = alpha_moe_optimizer_get_domain_profile_count();
            for (int d = 0; d < n_dp; d++) {
                const char * name = alpha_moe_optimizer_get_domain_profile_name(d);
                if (!name || strcmp(name, "classify") == 0 || strcmp(name, "grammar") == 0 ||
                    strcmp(name, "core") == 0) continue;  // skip meta-profiles
                if (!domain_list.empty()) domain_list += ", ";
                domain_list += name;
            }
            if (domain_list.empty()) {
                // Fallback to hardcoded list
                domain_list = "chemical, math, scientific, medical, code, sql, "
                    "factual, conversational, instructional, narrative, "
                    "latex, legal, multilingual, technical, analytical, markdown, "
                    "json_output, regex, assembly, musical";
            }
            std::string classify_instruction =
                "Classify into one domain: " + domain_list + ". "
                "Reply ONLY the domain name. "
                "Prompt: " + original_msg;

            json cl_body;
            cl_body["messages"] = json::array({
                {{"role", "user"}, {"content", classify_instruction}}
            });
            cl_body["chat_template_kwargs"] = json::object();
            cl_body["chat_template_kwargs"]["enable_thinking"] = false;
            cl_body["max_tokens"] = 10;
            cl_body["temperature"] = 0;
            cl_body["stream"] = false;

            std::vector<raw_buffer> cl_files;
            json cl_data;
            try {
                cl_data = oaicompat_chat_params_parse(cl_body, meta->chat_params, cl_files);
            } catch (...) {
                cl_data = json();
            }
            cl_data["_is_classify"] = true;

            // Classify KV caching: the domain list prefix (~400 tokens) is identical
            // for every request. Cache it after first cpu-moe computation, then reuse.
            // Only the user message suffix (~10 tokens) needs fresh computation.
            // Periodic full refresh (every 50 requests) prevents any cache drift.
            static int classify_cache_count = 0;
            if (classify_cache_count++ % 50 == 0) {
                // Full cpu-moe classify - refreshes cached KV
                cl_data["cache_prompt"] = false;
                auto * mem = llama_get_memory(ctx_server.get_ctx());
                if (mem) {
                    llama_memory_seq_rm(mem, 0, -1, -1);
                }
            } else {
                // Fast cached classify - reuse prefix KV from prompt cache
                cl_data["cache_prompt"] = true;
            }

            // Set classify flag before posting sub-task so event loop uses cpu-moe
            ctx_server.g_classify_active = true;

            if (cl_data.contains("prompt")) {
                auto cl_inputs = tokenize_input_prompts(
                    ctx_server.vocab, ctx_server.mctx,
                    cl_data.at("prompt"), true, true);

                if (!cl_inputs.empty()) {
                    auto cl_res = create_response();
                    auto & cl_rd = cl_res->rd;

                    server_task cl_task(type);
                    cl_task.id = cl_rd.get_new_id();
                    cl_task.tokens = std::move(cl_inputs[0]);
                    cl_task.params = server_task::params_from_json_cmpl(
                        ctx_server.vocab, params, meta->slot_n_ctx,
                        meta->logit_bias_eog, cl_data);
                    cl_task.params.res_type = res_type;

                    std::vector<server_task> cl_tasks;
                    cl_tasks.push_back(std::move(cl_task));
                    cl_rd.post_tasks(std::move(cl_tasks));

                    auto cl_results = cl_rd.wait_for_all(req.should_stop);
                    if (!cl_results.is_terminated && !cl_results.error &&
                        !cl_results.results.empty()) {
                        // Parse domain name from classify output
                        auto cl_json = cl_results.results[0]->to_json();
                        std::string cl_content;
                        if (cl_json.contains("choices") && !cl_json["choices"].empty() &&
                            cl_json["choices"][0].contains("message") &&
                            cl_json["choices"][0]["message"].contains("content")) {
                            cl_content = cl_json["choices"][0]["message"]["content"].get<std::string>();
                        }
                        // Clean: lowercase, trim, take first word
                        std::string detected;
                        for (char c : cl_content) {
                            if (c == '\n' || c == ' ' || c == '.') break;
                            detected += (char)tolower((unsigned char)c);
                        }
                        // Validate against loaded domain profiles (dynamic, not hardcoded)
                        const char * matched_domain = nullptr;
                        for (int d = 0; d < alpha_moe_optimizer_get_domain_profile_count(); d++) {
                            const char * pname = alpha_moe_optimizer_get_domain_profile_name(d);
                            if (pname && detected == pname) { matched_domain = pname; break; }
                        }
                        if (matched_domain) {
                            alpha_moe_set_detected_domain(matched_domain);
                        } else {
                            // Classify failed - fall back to keyword detection
                            const char * kw_domain = alpha_moe_detect_output_domain(
                                original_msg.c_str(), (int)original_msg.size());
                            alpha_moe_set_detected_domain(kw_domain);
                        }
                    }

                    // Don't clear cache here - the main task's SLOT_STATE_STARTED
                    // will handle exit_draft + disable_cache + argsort_persist.
                    // Extra cache clears between classify/main/two-pass corrupt state.
                }
            }

            // Re-enable cache after classify's cpu-moe disabled it.
            // This lets pin_domain_experts load domain experts for LUT prefill.
            // The auto-repair system catches any LUT quality issues.
            {
                auto * moe = ctx_server.moe_opt_public();
                if (moe && alpha_moe_optimizer_is_moe(moe)) {
                    // Don't force LUT - start warmup cycle instead
                    // The warmup will switch to LUT when cache is actually warm
                    alpha_moe_optimizer_disable_cache(moe);
                    // slot.moe_adaptive_warmup = true is set at request start
                }
            }

            // Reset classify flag - main task should not use cpu-moe path
            ctx_server.g_classify_active = false;
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");

            // Clean up residual <think>...</think> tags in content.
            // Some models (122B, 397B) generate think tags even when enable_thinking=false.
            // The reasoning parser may miss malformed tags (multiple </think>, etc).
            // Strip any remaining think blocks from content, keeping only the answer.
            for (auto & r : arr) {
                if (!r.contains("choices")) continue;
                for (auto & choice : r["choices"]) {
                    if (!choice.contains("message") || !choice["message"].contains("content")) continue;
                    std::string content = choice["message"]["content"].get<std::string>();
                    if (content.find("<think>") == std::string::npos &&
                        content.find("</think>") == std::string::npos) continue;
                    // Remove all <think>...</think> blocks and orphaned tags
                    std::string cleaned;
                    size_t pos = 0;
                    bool in_think = false;
                    while (pos < content.size()) {
                        if (!in_think) {
                            size_t ts = content.find("<think>", pos);
                            size_t te = content.find("</think>", pos);
                            if (ts != std::string::npos && (te == std::string::npos || ts < te)) {
                                cleaned += content.substr(pos, ts - pos);
                                pos = ts + 7; // skip "<think>"
                                in_think = true;
                            } else if (te != std::string::npos) {
                                // orphaned </think> - skip it
                                cleaned += content.substr(pos, te - pos);
                                pos = te + 8;
                            } else {
                                cleaned += content.substr(pos);
                                break;
                            }
                        } else {
                            size_t te = content.find("</think>", pos);
                            if (te != std::string::npos) {
                                pos = te + 8;
                                in_think = false;
                            } else {
                                break; // unclosed think - discard rest
                            }
                        }
                    }
                    // Trim leading whitespace
                    size_t first = cleaned.find_first_not_of(" \n\r\t");
                    if (first != std::string::npos && first > 0) cleaned = cleaned.substr(first);
                    else if (first == std::string::npos) cleaned.clear();
                    choice["message"]["content"] = cleaned;
                }
            }

            // Two-pass LUT refinement for MoE long-gen
            // Pass 1 complete. If MoE + long-gen, create a second request
            // with refinement prompt to fix errors. Each pass goes through
            // the full serve pipeline (fresh task -> slot -> warmup -> generate).
            // Two-pass LUT refinement: enabled when MoE expert cache is active.
            // Check via moe_lut_active flag (set during model load for MoE models).
            // Two-pass LUT refinement: enabled for all VRAM sizes.
            // Expert cache reserve (30% on <12GB, 20% on >=12GB) ensures enough
            // headroom for Pass 2 compute buffers.
            if (arr.size() == 1 && ctx_server.is_moe_lut_active()) {
                // Extract Pass 1 content
                std::string p1_content;
                if (arr[0].contains("choices") && !arr[0]["choices"].empty() &&
                    arr[0]["choices"][0].contains("message") &&
                    arr[0]["choices"][0]["message"].contains("content")) {
                    p1_content = arr[0]["choices"][0]["message"]["content"].get<std::string>();
                }
                // Two-pass for any meaningful output (>5 tokens).
                // Skip for very short answers (1-5 tokens like "Paris" - correct from prompt logits).
                // Skip if this is Pass 2 (prevent recursive two-pass).
                int p1_tokens = 0;
                if (arr[0].contains("usage"))
                    p1_tokens = arr[0]["usage"].value("completion_tokens", 0);
                int req_max_tokens = json_value(data, "max_tokens", -1);
                int pass_num = json_value(data, "_pass_num", 0);  // 0 = Pass 1
                bool is_pass2 = json_value(data, "_is_pass2", false);

                // Two-pass confidence-guided repair (Strategy B)
                // Check if Pass 1 output has suspect tokens (low margin + expert miss).
                // If suspects found, generate a self-repair pass asking the model
                // to review and fix the suspicious parts.
                // Only on <12GB GPUs (which skip multi-pass tiling), and only for Pass 1.
                // Two-pass repair: only on GPUs with enough VRAM for two concurrent
                // generations. On <12GB, repair must happen client-side (separate request).
                // Disabled: server-side two-pass crashes; use client-side _cpu_moe repair.
                if (false && ctx_server.is_moe_lut_active() && !is_pass2 && pass_num == 0 &&
                    p1_tokens > 5 && !p1_content.empty() &&
                    ctx_server.get_moe_gpu_vram_mb() >= 12000 &&
                    ctx_server.get_moe_gpu_vram_mb() < 24000) {

                    // Check for duplicate words (the most common K<256 artifact)
                    bool has_dup = false;
                    {
                        std::regex dup_re("\\b(\\w{3,})\\s+\\1\\b");
                        has_dup = std::regex_search(p1_content, dup_re);
                    }

                    if (has_dup) {
                        fprintf(stderr, "[TWO-PASS] Duplicate word detected in Pass 1 output, triggering repair\n");

                        // Build repair prompt
                        std::string original_msg = json_value(data, "_original_user_msg", std::string());
                        std::string repair_instruction;
                        if (!original_msg.empty()) {
                            repair_instruction =
                                "The following answer has some duplicate or garbled words. "
                                "Rewrite it correctly with no repeated words or errors.\n\n"
                                "Original question: " + original_msg + "\n\n"
                                "Draft answer: " + p1_content + "\n\n"
                                "Corrected answer:";
                        } else {
                            repair_instruction =
                                "The following text has duplicate or garbled words. "
                                "Rewrite it with corrections:\n\n"
                                + p1_content + "\n\nCorrected:";
                        }

                        json repair_body;
                        repair_body["messages"] = json::array({
                            {{"role", "user"}, {"content", repair_instruction}}
                        });
                        repair_body["chat_template_kwargs"] = json::object();
                        repair_body["chat_template_kwargs"]["enable_thinking"] = false;
                        repair_body["max_tokens"] = std::max(req_max_tokens,
                            (int)(p1_content.size() / 3 + 20));
                        repair_body["temperature"] = 0;
                        repair_body["stream"] = false;

                        std::vector<raw_buffer> repair_files;
                        json repair_data;
                        try {
                            repair_data = oaicompat_chat_params_parse(
                                repair_body, meta->chat_params, repair_files);
                        } catch (...) {
                            fprintf(stderr, "[TWO-PASS] Repair template parse failed\n");
                            repair_data = json();
                        }

                        if (repair_data.contains("prompt")) {
                            repair_data["_is_pass2"] = true;
                            repair_data["_pass_num"] = 1;

                            auto repair_res = handle_completions_impl(
                                req, type, repair_data, repair_files, res_type);
                            if (repair_res) {
                                auto repair_rd_result = repair_res->rd.wait_for_all(req.should_stop);
                                if (!repair_rd_result.is_terminated && !repair_rd_result.error &&
                                    !repair_rd_result.results.empty()) {
                                    json repair_json = repair_rd_result.results[0]->to_json();
                                    if (repair_json.contains("choices") &&
                                        !repair_json["choices"].empty() &&
                                        repair_json["choices"][0].contains("message") &&
                                        repair_json["choices"][0]["message"].contains("content")) {
                                        std::string repaired = repair_json["choices"][0]["message"]["content"].get<std::string>();
                                        if (!repaired.empty() && repaired.size() > 5) {
                                            fprintf(stderr, "[TWO-PASS] Repair successful: %zu → %zu chars\n",
                                                    p1_content.size(), repaired.size());
                                            // Replace Pass 1 content with repaired content
                                            arr[0]["choices"][0]["message"]["content"] = repaired;
                                            p1_content = repaired;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Serial expert tiling via ranked domain profiles
                // Each pass loads core + a different domain's experts.
                // Pass 1: core + domain_rank[0] (most important domain)
                // Pass 2: core + domain_rank[1] (second domain)
                // Pass N: core + domain_rank[N-1] or grammar
                // ~6 passes covers all 256 experts (4 core + 51 domain per pass).
                //
                // disabled on <12GB GPUs: each pass allocates new compute buffers,
                // which eventually runs 8GB GPUs out of VRAM. Single-pass is stable.
                const char * ranked_domains[10] = {};
                int ranked_scores[10] = {};
                int n_ranked = 0;
                {
                    std::string orig = json_value(data, "_original_user_msg", std::string());
                    if (!orig.empty()) {
                        n_ranked = alpha_moe_detect_output_domain_ranked(
                            orig.c_str(), (int)orig.size(), ranked_domains, ranked_scores, 10);
                    }
                }
                // Build pass schedule: each ranked domain that has a profile
                std::vector<const char *> domain_passes;
                for (int i = 0; i < n_ranked; i++) {
                    if (alpha_moe_optimizer_has_domain_profile(ranked_domains[i]))
                        domain_passes.push_back(ranked_domains[i]);
                }
                // Always end with grammar (quality fix)
                if (alpha_moe_optimizer_has_domain_profile("grammar"))
                    domain_passes.push_back("grammar");
                // Cap tiling passes: 1 pass for KV warming
                if (domain_passes.size() > 1) domain_passes.resize(1);
                int max_passes = 1 + (int)domain_passes.size();

                // Skip tiling for low-K models (K/n_expert < 10%).
                // Tiling does 2x cpu-moe warmup passes, which is slower than a
                // single cpu-moe prefill. The prefill argsort
                // already captures prompt-specific experts for generation.
                // Tiling disabled: confidence-gated cache warming handles quality.
                // Tiling pass 2 crashes on hybrid memory during multi-token re-prefill.
                if (false && p1_tokens > 15 && pass_num < max_passes - 1 &&
                    ctx_server.get_moe_gpu_vram_mb() >= 12000) {
                    int next_pass = pass_num + 1;
                    const char * next_domain = domain_passes[next_pass - 1];

                    for (int i = 0; i < (int)domain_passes.size(); i++)
                        fprintf(stderr, "%s%s", i ? "," : "", domain_passes[i]);
                    fprintf(stderr, ")\n");

                    // Set domain for warmup - core always pinned, domain cycles
                    alpha_moe_set_detected_domain(next_domain);
                    alpha_moe_set_detected_domain2(nullptr);

                    // Build Pass 2: short prompt for KV warming + expert loading.
                    // Only needs the original question (not P1 answer) to warm
                    // the model's routing state for generation.
                    std::string original_msg = json_value(data, "_original_user_msg", std::string());
                    std::string p2_instruction;
                    if (!original_msg.empty()) {
                        // Minimal: just the question (no P1 answer = shorter = faster)
                        p2_instruction = original_msg;
                    } else {
                        p2_instruction = p1_content.substr(0, 100); // first 100 chars
                    }

                    json p2_body;
                    p2_body["messages"] = json::array({
                        {{"role", "user"}, {"content", p2_instruction}}
                    });
                    p2_body["chat_template_kwargs"] = json::object();
                    p2_body["chat_template_kwargs"]["enable_thinking"] = false;
                    p2_body["max_tokens"] = std::max(req_max_tokens,
                        (int)(p1_content.size() / 3 + 20));
                    p2_body["temperature"] = 0;
                    p2_body["stream"] = false;

                    // Format through Jinja template (same path as manual curl)
                    std::vector<raw_buffer> p2_files;
                    json p2_data;
                    try {
                        p2_data = oaicompat_chat_params_parse(
                            p2_body, meta->chat_params, p2_files);
                    } catch (...) {
                        p2_data = json();
                    }

                    // Track pass number for serial expert tiling (chains Pass 2 -> Pass 3)
                    p2_data["_is_pass2"] = true;
                    p2_data["_pass_num"] = next_pass;
                    p2_data["_original_user_msg"] = json_value(data, "_original_user_msg", std::string());

                    if (p2_data.contains("prompt")) {
                        auto & p2_prompt = p2_data.at("prompt");
                        auto ref_inputs = tokenize_input_prompts(
                            ctx_server.vocab, ctx_server.mctx,
                            p2_prompt, true, true);

                    if (!ref_inputs.empty()) {
                        // Don't clear cache here - Pass 2's SLOT_STATE_STARTED
                        // handles exit_draft + disable_cache + argsort_persist.
                        // Extra clears between passes corrupt optimizer state.

                        // Create fresh response dispatcher for Pass 2
                        auto res2 = create_response();
                        auto & rd2 = res2->rd;

                        server_task task2(type);
                        task2.id = rd2.get_new_id();
                        task2.tokens = std::move(ref_inputs[0]);
                        task2.params = server_task::params_from_json_cmpl(
                            ctx_server.vocab, params, meta->slot_n_ctx,
                            meta->logit_bias_eog, p2_data);
                        task2.params.res_type = res_type;
                        task2.params.oaicompat_cmpl_id = completion_id;
                        task2.params.oaicompat_model = meta->model_name;

                        std::vector<server_task> tasks2;
                        tasks2.push_back(std::move(task2));
                        rd2.post_tasks(std::move(tasks2));

                        auto p2_results = rd2.wait_for_all(req.should_stop);
                        if (!p2_results.is_terminated && !p2_results.error &&
                            !p2_results.results.empty()) {
                            // Use Pass 2 output
                            arr.clear();
                            for (auto & r2 : p2_results.results) {
                                arr.push_back(r2->to_json());
                            }
                        }
                        // If Pass 2 fails, fall through with Pass 1 output
                    }
                    } // p2_data.contains("prompt")
                }
            }

            if (arr.size() == 1 && res_type != TASK_RESPONSE_TYPE_OLLAMA_CHAT) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OLLAMA_CHAT) {
                // Convert OpenAI format to Ollama format
                json oai = arr[0];
                json ollama;
                ollama["model"] = oai.value("model", "");
                ollama["done"] = true;
                // Extract message from choices[0].message
                if (oai.contains("choices") && !oai["choices"].empty()) {
                    ollama["message"] = oai["choices"][0]["message"];
                    // Merge reasoning_content into content (Ollama compat).
                    // Ollama never separates thinking - it's all in content.
                    // Models like Falcon-H1R, Jamba, DeepSeek-R1 always generate <think>
                    // even with think:false. Prepending reasoning ensures the full
                    // output is visible, matching Ollama behavior.
                    {
                        auto & msg = ollama["message"];
                        if (msg.contains("reasoning_content") &&
                            !msg["reasoning_content"].is_null() &&
                            msg["reasoning_content"].is_string()) {
                            std::string reasoning = msg["reasoning_content"].get<std::string>();
                            if (!reasoning.empty()) {
                                std::string content = msg.value("content", "");
                                msg["content"] = reasoning + content;
                                // Also keep as "thinking" for clients that want it separate
                                msg["thinking"] = reasoning;
                            }
                            msg.erase("reasoning_content");
                        }
                    }
                    ollama["done_reason"] = oai["choices"][0].value("finish_reason", "stop");

                    // logprobs available via OpenAI-format /v1/chat/completions endpoint
                }
                // Map timing fields (OpenAI uses ms in timings, Ollama uses nanoseconds)
                if (oai.contains("timings")) {
                    auto & t = oai["timings"];
                    int64_t eval_count = t.value("predicted_n", 0);
                    double eval_ms = t.value("predicted_ms", 0.0);
                    double prompt_ms = t.value("prompt_ms", 0.0);
                    ollama["eval_count"] = eval_count;
                    ollama["eval_duration"] = (int64_t)(eval_ms * 1000000);  // ms -> ns
                    ollama["prompt_eval_duration"] = (int64_t)(prompt_ms * 1000000);
                    ollama["total_duration"] = (int64_t)((eval_ms + prompt_ms) * 1000000);
                    ollama["prompt_eval_count"] = t.value("prompt_n", 0);
                } else if (oai.contains("usage")) {
                    ollama["eval_count"] = oai["usage"].value("completion_tokens", 0);
                    ollama["prompt_eval_count"] = oai["usage"].value("prompt_tokens", 0);
                }
                res->ok(ollama);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (res_type == TASK_RESPONSE_TYPE_OLLAMA_CHAT) {
            // Ollama streaming: newline-delimited JSON (not SSE)
            // Note: partial chat results return array of deltas, not single object
            json first_obj = first_result_json.is_array() && !first_result_json.empty()
                ? first_result_json[0] : first_result_json;
            json ollama;
            ollama["model"] = first_obj.is_object() ? first_obj.value("model", "") : "";
            ollama["done"] = false;
            if (first_obj.is_object() && first_obj.contains("choices") && !first_obj["choices"].empty()) {
                auto & delta = first_obj["choices"][0]["delta"];
                std::string content;
                if (delta.contains("content") && !delta["content"].is_null()) {
                    content = delta["content"].get<std::string>();
                }
                ollama["message"] = {{"role", "assistant"}, {"content", content}};
            } else {
                ollama["message"] = {{"role", "assistant"}, {"content", ""}};
            }
            res->data = ollama.dump() + "\n";
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = (res_type == TASK_RESPONSE_TYPE_OLLAMA_CHAT) ? "application/x-ndjson" : "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type /*res_type*/, const json & res_json) {
                return format_oai_sse(json {{ "error", res_json }});
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                            output = "";
                            break;

                        case TASK_RESPONSE_TYPE_OLLAMA_CHAT:
                            // Ollama final chunk: done=true with empty content
                            output = json({{"done", true}, {"message", {{"role", "assistant"}, {"content", ""}}}}).dump() + "\n";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_OLLAMA_CHAT) {
                        // Ollama streaming: newline-delimited JSON
                        // Note: partial chat results may be array of deltas
                        json obj = res_json.is_array() && !res_json.empty()
                            ? res_json[res_json.size() - 1] : res_json;
                        json ollama;
                        ollama["done"] = false;
                        if (obj.is_object() && obj.contains("choices") && !obj["choices"].empty()) {
                            auto & delta = obj["choices"][0]["delta"];
                            std::string content;
                            if (delta.contains("content") && !delta["content"].is_null()) {
                                content = delta["content"].get<std::string>();
                            }
                            // Merge reasoning into content (Ollama compat)
                            if (delta.contains("reasoning_content") && !delta["reasoning_content"].is_null()) {
                                content = delta["reasoning_content"].get<std::string>() + content;
                            }
                            ollama["message"] = {{"role", "assistant"}, {"content", content}};
                            // alphachat extras: per-token confidence in streaming
                            if (obj["choices"][0].contains("logprobs") &&
                                !obj["choices"][0]["logprobs"].is_null() &&
                                obj["choices"][0]["logprobs"].contains("content")) {
                                for (auto & lp : obj["choices"][0]["logprobs"]["content"]) {
                                    json tok;
                                    tok["id"] = lp.value("id", 0);
                                    tok["text"] = lp.value("token", "");
                                    if (lp.contains("prob")) {
                                        tok["prob"] = lp["prob"];
                                    } else if (lp.contains("logprob")) {
                                        tok["prob"] = std::exp(lp["logprob"].get<float>());
                                    }
                                    ollama["token"] = tok;
                                }
                            }
                        } else {
                            ollama["message"] = {{"role", "assistant"}, {"content", ""}};
                        }
                        output = ollama.dump() + "\n";
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->get_api_show = [this](const server_http_req &) {
        auto res = create_response();
        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        json data = {
            {
                "model_info", {
                    { "llama.context_length", meta->slot_n_ctx },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", tmpl_default},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})}
        };

        res->ok(data);
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);

        // Auto-retrieve: inject context as a system message. Comprehend mode navigates the loaded
        // summary pyramid (Method-2) inside corpus_augment_messages; other extended modes use BM25/NER.
        if (((ctx_server.effective_ctx_mode == ALPHA_CTX_COMPREHEND && ctx_server.comprehend_loaded) ||
             (ctx_server.effective_ctx_mode != ALPHA_CTX_NATIVE &&
              ctx_server.effective_ctx_mode != ALPHA_CTX_SEQUENTIAL &&
              ctx_server.chunk_bm25.N > 0)) &&
            body.contains("messages") && body["messages"].is_array()) {
            corpus_augment_messages(body);
        }

        // Support think:false in OpenAI API (same as Ollama path)
        if (body.contains("think")) {
            if (!body.contains("chat_template_kwargs")) {
                body["chat_template_kwargs"] = json::object();
            }
            body["chat_template_kwargs"]["enable_thinking"] = body["think"];
            body.erase("think");
        }
        // cpu_moe: force all experts via mmap for this request (repair pass)
        bool req_cpu_moe = json_value(body, "cpu_moe", false);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        if (req_cpu_moe) {
            body_parsed["_cpu_moe"] = true;
        }
        // Save original user message for two-pass refinement
        if (body.contains("messages") && body["messages"].is_array()) {
            for (int mi = (int)body["messages"].size() - 1; mi >= 0; mi--) {
                if (body["messages"][mi].value("role", "") == "user" &&
                    body["messages"][mi].contains("content") &&
                    body["messages"][mi]["content"].is_string()) {
                    body_parsed["_original_user_msg"] = body["messages"][mi]["content"];
                    break;
                }
            }
        }
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    // Ollama /api/chat - native Ollama format
    this->post_ollama_chat = [this](const server_http_req & req) {
        // Minimal Ollama->OpenAI translation: just map options to top-level and pass through
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);

        // Auto-retrieve: inject context as system message (Ollama chat path)
        if (ctx_server.effective_ctx_mode != ALPHA_CTX_NATIVE &&
            ctx_server.effective_ctx_mode != ALPHA_CTX_SEQUENTIAL &&
            ctx_server.chunk_bm25.N > 0 &&
            body.contains("messages") && body["messages"].is_array()) {
            corpus_augment_messages(body);
        }

        // Map options.* -> top-level OpenAI params
        if (body.contains("options") && body["options"].is_object()) {
            json opts = body["options"];  // copy, not reference
            if (opts.contains("num_predict")) body["max_tokens"] = opts["num_predict"];
            if (opts.contains("temperature")) body["temperature"] = opts["temperature"];
            if (opts.contains("top_p"))       body["top_p"]       = opts["top_p"];
            if (opts.contains("seed"))        body["seed"]        = opts["seed"];
            if (opts.contains("stop"))        body["stop"]        = opts["stop"];
        }

        // alphachat extra: log_confidence -> enable logprobs
        if (body.contains("log_confidence") && body["log_confidence"].get<bool>()) {
            body["logprobs"] = true;
            if (!body.contains("top_logprobs")) {
                body["top_logprobs"] = 1;
            }
        }

        // Map Ollama think field -> OpenAI chat_template_kwargs.enable_thinking
        // Default think:false for non-streaming (thinking tokens invisible in non-streaming responses)
        if (body.contains("think")) {
            if (!body.contains("chat_template_kwargs")) {
                body["chat_template_kwargs"] = json::object();
            }
            body["chat_template_kwargs"]["enable_thinking"] = body["think"];
        } else if (!body.value("stream", false)) {
            if (!body.contains("chat_template_kwargs")) {
                body["chat_template_kwargs"] = json::object();
            }
            body["chat_template_kwargs"]["enable_thinking"] = false;
        }

        // Remove Ollama-only fields before passing to OpenAI parser
        body.erase("options");
        body.erase("think");
        body.erase("keep_alive");
        body.erase("log_confidence");
        // model field is fine - OpenAI parser handles it

        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        // Save original user message for two-pass refinement
        if (body.contains("messages") && body["messages"].is_array()) {
            for (int mi = (int)body["messages"].size() - 1; mi >= 0; mi--) {
                if (body["messages"][mi].value("role", "") == "user" &&
                    body["messages"][mi].contains("content") &&
                    body["messages"][mi]["content"].is_string()) {
                    body_parsed["_original_user_msg"] = body["messages"][mi]["content"];
                    break;
                }
            }
        }
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OLLAMA_CHAT);
    };

    // Ollama /api/generate - text completion
    this->post_ollama_generate = [this](const server_http_req & req) {
        json body = json::parse(req.body);

        // Map Ollama generate format to internal completion format
        if (body.contains("options") && body["options"].is_object()) {
            auto & opts = body["options"];
            if (opts.contains("num_predict")) body["n_predict"] = opts["num_predict"];
            if (opts.contains("temperature")) body["temperature"] = opts["temperature"];
            if (opts.contains("top_p"))       body["top_p"]       = opts["top_p"];
            if (opts.contains("seed"))        body["seed"]        = opts["seed"];
            if (opts.contains("stop"))        body["stop"]        = opts["stop"];
        }
        // /api/generate with bare "prompt": apply the chat template
        // so the model gets properly formatted input (not raw text echo).
        if (body.contains("prompt") && body["prompt"].is_string()) {
            json chat_body;
            chat_body["messages"] = json::array({
                {{"role", "user"}, {"content", body["prompt"].get<std::string>()}}
            });
            chat_body["chat_template_kwargs"] = json::object();
            chat_body["chat_template_kwargs"]["enable_thinking"] = false;
            // Copy generation params
            for (auto & [k, v] : body.items()) {
                if (k != "prompt" && k != "messages") chat_body[k] = v;
            }
            std::vector<raw_buffer> files;
            try {
                json formatted = oaicompat_chat_params_parse(chat_body, meta->chat_params, files);
                body["prompt"] = formatted["prompt"];
                body["_original_user_msg"] = body.count("prompt") ? "" : "";
            } catch (...) {
                // If template fails, fall through with raw prompt
            }
        }
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            {},
            TASK_RESPONSE_TYPE_OLLAMA_CHAT);
    };

    // Ollama model management endpoints

    // /api/ps - list running models (Ollama format)
    this->get_ollama_ps = [this](const server_http_req &) {
        auto res = create_response();
        json model_info;
        std::string model_name_str = meta->model_aliases.empty() ? meta->model_name : *meta->model_aliases.begin();
        model_info["name"] = model_name_str;
        model_info["model"] = model_name_str;
        model_info["size"] = 0;  // TODO: actual model file size
        model_info["digest"] = "";
        model_info["details"] = {
            {"format", "gguf"},
            {"family", ""},
            {"parameter_size", ""},
            {"quantization_level", ""}
        };

        // Get VRAM usage
        size_t vram_used = 0, vram_total = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                size_t free_bytes = 0, total_bytes = 0;
                ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
                vram_used = total_bytes - free_bytes;
                vram_total = total_bytes;
                break;
            }
        }
        model_info["size_vram"] = vram_used;

        res->ok(json{{"models", json::array({model_info})}});
        return res;
    };

    // /api/pull - stub (model already loaded in single-model server)
    this->post_ollama_pull = [this](const server_http_req & req) {
        auto res = create_response();
        json body = json::parse(req.body);
        std::string name = body.value("name", "");
        // In single-model server mode, we can't pull new models
        // Return status indicating the current model
        res->ok(json{
            {"status", "success"},
            {"digest", ""},
            {"total", 0},
            {"completed", 0}
        });
        return res;
    };

    // /api/delete - stub (can't delete loaded model)
    this->delete_ollama_model = [this](const server_http_req &) {
        auto res = create_response();
        res->status = 400;
        res->data = json{{"error", "cannot delete model while it is loaded"}}.dump();
        return res;
    };

    // /api/copy - stub (not supported in single-model mode)
    this->post_ollama_copy = [this](const server_http_req &) {
        auto res = create_response();
        res->status = 400;
        res->data = json{{"error", "copy not supported in single-model server mode"}}.dump();
        return res;
    };

    // /api/embed - stub (embedding not supported in this mode)
    this->post_ollama_embed = [this](const server_http_req &) {
        auto res = create_response();
        res->status = 501;
        res->data = json{{"error", "embedding not supported — use /v1/embeddings with an embedding model"}}.dump();
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       meta->model_name},
                    {"aliases",  meta->model_aliases},
                    {"tags",     meta->model_tags},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     {
                        {"vocab_type",  meta->model_vocab_type},
                        {"n_vocab",     meta->model_vocab_n_tokens},
                        {"n_ctx_train", meta->model_n_ctx_train},
                        {"n_embd",      meta->model_n_embd_inp},
                        {"n_params",    meta->model_n_params},
                        {"size",        meta->model_size},
                    }},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

// KV-Chunk Endpoints
// Extended-code auto-retrieve: query corpus, build NER summary, inject as system message.
// Called from chat handlers when --context extended-code is active and corpus is indexed.
// Context budget calculator: scales injection to actual context window
struct context_budget {
    int max_code_chars;
    int max_ner_chars;
    int max_experience_chars;
    int max_experience_count;
};

static context_budget compute_context_budget(int slot_n_ctx, bool has_tools, int n_history_msgs) {
    // Reserve tokens for overhead
    int tool_overhead  = has_tools ? 1500 : 0;
    int gen_reserve    = 1024;
    int sys_overhead   = 200;
    int history_tokens = n_history_msgs * 150;

    int available_tokens = slot_n_ctx - tool_overhead - gen_reserve - sys_overhead - history_tokens;
    if (available_tokens < 512) available_tokens = 512;

    // ~3.5 chars per token (conservative for code)
    int available_chars = (int)(available_tokens * 3.5f);

    context_budget b;
    b.max_experience_count = 3;
    b.max_experience_chars = std::min(5000, available_chars / 10);
    b.max_ner_chars        = available_chars * 15 / 100;
    b.max_code_chars       = available_chars - b.max_ner_chars - b.max_experience_chars;
    if (b.max_code_chars < 500) b.max_code_chars = 500;
    return b;
}

// Method-2 iterative deepening over the loaded summary pyramid (comprehend mode). Descends the
// tree top-down, scoring each level's node summaries against the query (keyword overlap) and
// keeping the top-K, until leaves; returns awareness (corpus + division summaries) + the selected
// leaf chunks as a context string. No LLM calls - the summaries come from the offline pyramid build.
std::string server_context_impl::comprehend_walk(const std::string & query, int budget_chars) const {
    if (!comprehend_loaded) return "";
    auto lc = [](char c) -> char { return (c >= 'A' && c <= 'Z') ? (char)(c + 32) : c; };
    auto is_word = [](char c) -> bool {
        return (c>='a'&&c<='z') || (c>='A'&&c<='Z') || (c>='0'&&c<='9');
    };
    try {
        const json & pyr = comprehend_pyramid;
        std::unordered_map<std::string, const json*> byid;
        for (const auto & n : pyr["nodes"]) byid[n.value("id", std::string())] = &n;

        // Query terms (lowercased words, length > 2).
        std::vector<std::string> qterms;
        std::string cur;
        for (char c : query) {
            if (is_word(c)) cur += lc(c);
            else { if (cur.size() > 2) qterms.push_back(cur); cur.clear(); }
        }
        if (cur.size() > 2) qterms.push_back(cur);

        auto score = [&](const std::string & text) -> int {
            std::string lt; lt.reserve(text.size());
            for (char c : text) lt += lc(c);
            int s = 0;
            for (const auto & t : qterms) if (lt.find(t) != std::string::npos) s++;
            return s;
        };

        const json & levels = pyr["levels"];
        int top = (int)levels.size() - 1;
        if (top < 1) return "";
        auto it_root = byid.find(pyr.value("root", std::string()));
        if (it_root == byid.end()) return "";

        // Descend: corpus root's children (divisions) -> ... -> just above leaves.
        std::vector<std::string> frontier;
        for (const auto & c : (*it_root->second)["child_ids"]) frontier.push_back(c.get<std::string>());
        const int KEEP = 6;
        for (int lvl = top - 1; lvl > 0; lvl--) {
            std::vector<std::pair<int,std::string>> scored;
            for (const auto & nid : frontier) {
                auto it = byid.find(nid);
                if (it != byid.end()) scored.push_back({ score(it->second->value("text", std::string())), nid });
            }
            std::sort(scored.begin(), scored.end(),
                      [](const std::pair<int,std::string> & a, const std::pair<int,std::string> & b){ return a.first > b.first; });
            std::vector<std::string> next;
            for (int i = 0; i < (int)scored.size() && i < KEEP; i++) {
                auto it = byid.find(scored[i].second);
                if (it == byid.end()) continue;
                for (const auto & c : (*it->second)["child_ids"]) next.push_back(c.get<std::string>());
            }
            frontier = std::move(next);
        }

        // Leaves: score, then pull chunks best-first within the char budget.
        std::vector<std::pair<int,std::string>> leaves;
        for (const auto & nid : frontier) {
            auto it = byid.find(nid);
            if (it != byid.end()) leaves.push_back({ score(it->second->value("text", std::string())), nid });
        }
        std::sort(leaves.begin(), leaves.end(),
                  [](const std::pair<int,std::string> & a, const std::pair<int,std::string> & b){ return a.first > b.first; });

        std::string out = "[Corpus Overview]\n" + levels[top][0].get<std::string>() + "\n";
        int used = (int)out.size();
        // Awareness (division summaries) must be bounded too: large corpora can have many divisions,
        // and injecting them all can overrun the context slot. Reserve at least half the budget for
        // the source chunks below; stop adding divisions once the awareness half is full.
        const int awareness_cap = budget_chars / 2;
        for (const auto & d : levels[top - 1]) {
            std::string ds = "\n[Division]\n" + d.get<std::string>() + "\n";
            if (used + (int)ds.size() > awareness_cap) break;
            out += ds;
            used += (int)ds.size();
        }
        const json chunks = pyr.contains("chunks") ? pyr["chunks"] : json::array();
        for (const auto & sc_nid : leaves) {
            auto it = byid.find(sc_nid.second);
            if (it == byid.end()) continue;
            for (const auto & c : (*it->second)["child_ids"]) {
                std::string cs = c.get<std::string>();
                if (cs.rfind("chunk:", 0) != 0) continue;
                int ci = std::atoi(cs.c_str() + 6);
                if (ci < 0 || ci >= (int)chunks.size()) continue;
                std::string ctext = chunks[ci].get<std::string>();
                if (used + (int)ctext.size() > budget_chars) return out;
                out += "\n[Source chunk " + std::to_string(ci) + "]\n" + ctext + "\n";
                used += (int)ctext.size();
            }
        }
        return out;
    } catch (const std::exception & e) {
        SRV_INF("comprehend_walk failed: %s\n", e.what());
        return "";
    }
}

void server_routes::corpus_augment_messages(json & body) {
    // 1. Extract query from last user message
    std::string query;
    auto & msgs = body["messages"];
    for (int i = (int)msgs.size() - 1; i >= 0; i--) {
        if (msgs[i].value("role", "") == "user" &&
            msgs[i].contains("content") && msgs[i]["content"].is_string()) {
            query = msgs[i]["content"].get<std::string>();
            break;
        }
    }
    if (query.empty()) return;

    // 2. Compute context-aware budget
    bool has_tools = (tools_ptr && !tools_ptr->tools.empty());
    auto budget = compute_context_budget(meta->slot_n_ctx, has_tools, (int)msgs.size());
    bool use_ner = (ctx_server.effective_ctx_mode == ALPHA_CTX_EXTENDED_CODE);

    // Comprehend mode: navigate the pre-built summary pyramid (Method-2 iterative deepening) and
    // inject awareness + selected chunks, instead of flat BM25/NER retrieval.
    if (ctx_server.effective_ctx_mode == ALPHA_CTX_COMPREHEND && ctx_server.comprehend_loaded) {
        // Comprehend fills its whole char budget with dense summary/record text (~2.4 chars/token,
        // vs the 3.5 compute_context_budget assumes), so max_code_chars would overrun the token slot.
        // Budget from the token slot instead: reserve generation + question headroom, then use a
        // conservative 2.3 chars/token so the injected prompt fits.
        int comp_safe_tokens = meta->slot_n_ctx - 1024 /*gen*/ - 512 /*question + template overhead*/;
        if (comp_safe_tokens < 512) comp_safe_tokens = 512;
        int comp_budget_chars = (int)(comp_safe_tokens * 2.3f);
        std::string aug = ctx_server.comprehend_walk(query, comp_budget_chars);
        if (!aug.empty()) {
            msgs.insert(msgs.begin(), json{{"role", "system"}, {"content", aug}});
            SRV_INF("comprehend: injected %d chars via Method-2 walk\n", (int)aug.size());
        }
        return;
    }

    // Unified router: route the question to one class and inject a single focused context
    // (entity record / aggregation count / section stats). On no structured match, fall through
    // to the BM25/redis path below (never merge-all). `extended` uses this too, as unified with
    // BM25 fallback.
    if ((ctx_server.effective_ctx_mode == ALPHA_CTX_UNIFIED ||
         ctx_server.effective_ctx_mode == ALPHA_CTX_EXTENDED) && ctx_server.unified.built) {
        std::string route_cls;
        std::string aug = ctx_server.unified.route(query, route_cls, &ctx_server.chunk_bm25);
        if (!aug.empty()) {
            // Additive, not replacing: on a real corpus an entity/record match can be spurious (a
            // query name token hitting a noise record), and injected alone it would suppress the BM25
            // retrieval that holds the answer. Keep the focused hint at the top and append a capped
            // BM25 source-passage supplement below, so the source text is always present.
            // Precomputed-count classes (AGG_* / DIV_* return an exact number) stay focused, since
            // source text only dilutes an exact count.
            bool self_contained = (route_cls.rfind("AGG", 0) == 0) || (route_cls.rfind("DIV_", 0) == 0);
            if (!self_contained && ctx_server.chunk_bm25.N > 0) {
                // Lean supplement: the answer for a retrieval/record query usually sits in BM25 rank
                // 1-2, so a few passages suffice; more only adds latency and dilution.
                int sup_cap = std::min(budget.max_code_chars, 4500);
                std::string extra;
                for (int cid : ctx_server.chunk_bm25.top_k(query, 4)) {
                    std::string t = ctx_server.chunk_bm25.get_text(cid);
                    if (t.empty()) continue;
                    if ((int)extra.size() + (int)t.size() + 6 > sup_cap) break;
                    extra += "\n---\n" + t;
                }
                if (!extra.empty()) aug += "\n\n[Source passages]\n" + extra;
            }
            msgs.insert(msgs.begin(), json{{"role", "system"}, {"content", aug}});
            fprintf(stderr, "unified route=%s path=focused %d chars (supp=%s)\n",
                    route_cls.c_str(), (int)aug.size(), self_contained ? "no" : "bm25");
            return;
        }
        fprintf(stderr, "unified route=FALLBACK path=bm25\n");
        // fall through to BM25 fallback
    }

    std::string code_context;
    std::string ner_summary;
    std::string experience_context;
    int code_chars = 0, exp_chars = 0, exp_count = 0;

    // 3. Try Redis first (entity -> graph -> aggregation -> FT.search)
#ifdef ALPHA_HAS_REDIS
    if (ctx_server.use_redis && ctx_server.redis) {
        auto redis_result = ctx_server.redis->query(query);
        if (redis_result.found) {
            std::string augmented;

            if (redis_result.method == "graph") {
                // Graph traversal: format relationship chain
                augmented += "[Knowledge Graph]\n";
                for (auto & sent : redis_result.sentences) {
                    augmented += "  " + sent + "\n";
                }
                // Fetch actual sentence text for each discovered entity
                for (auto & ent : redis_result.entities) {
                    auto ent_result = ctx_server.redis->try_entity_lookup(ent);
                    if (ent_result.found) {
                        for (auto & s : ent_result.sentences) {
                            if ((int)augmented.size() + (int)s.size() > budget.max_code_chars) break;
                            augmented += s + "\n";
                        }
                    }
                    auto imports = ctx_server.redis->get_imports(ent);
                    if (!imports.empty()) {
                        augmented += ent + " imports:";
                        for (auto & imp : imports) augmented += " " + imp;
                        augmented += "\n";
                    }
                }
                // Also FT.search for the original query terms
                auto ft = ctx_server.redis->ft_search(query, 10);
                if (!ft.sentences.empty()) {
                    augmented += "\n[Related Text]\n";
                    for (auto & s : ft.sentences) {
                        if ((int)augmented.size() + (int)s.size() > budget.max_code_chars) break;
                        augmented += s + "\n";
                    }
                }

            } else if (redis_result.method == "aggregation") {
                // Aggregation: clean list of all matching codes/records
                augmented += "[Records Found]\n";
                for (auto & ent : redis_result.entities) {
                    augmented += "  • " + ent + "\n";
                    std::string src = ctx_server.redis->get_code_source(ent);
                    if (!src.empty()) {
                        augmented += "    Source: " + src + "\n";
                    }
                }

            } else if (redis_result.method == "entity") {
                // Entity lookup: show where entity appears + fetch sentence text
                augmented += "[Entity Locations]\n";
                for (auto & loc : redis_result.sentences) {
                    augmented += "  " + loc + "\n";
                }
                // Fetch actual sentences for context
                auto ft = ctx_server.redis->ft_search(query, 20);
                if (!ft.sentences.empty()) {
                    augmented += "\n[Relevant Text]\n";
                    for (auto & s : ft.sentences) {
                        if ((int)augmented.size() + (int)s.size() > budget.max_code_chars) break;
                        augmented += s + "\n";
                    }
                }

            } else {
                // FT.search: sentences are already text
                for (auto & sent : redis_result.sentences) {
                    if ((int)augmented.size() + (int)sent.size() > budget.max_code_chars) break;
                    augmented += sent + "\n";
                }
            }

            if (!augmented.empty()) {
                json sys_msg = {{"role", "system"}, {"content", augmented}};
                msgs.insert(msgs.begin(), sys_msg);

                if (has_tools) {
                    json tools_arr = json::array();
                    for (auto & t : tools_ptr->tools) tools_arr.push_back(t->get_definition());
                    body["tools"] = tools_arr;
                    body["tool_choice"] = "auto";
                }

                fprintf(stderr, "corpus-augment: redis [%s] %d chars, %d entities, %.1fms\n",
                        redis_result.method.c_str(), (int)augmented.size(),
                        (int)redis_result.entities.size(), redis_result.latency_ms);
                return;  // Redis handled it
            }
        }
    }
#endif

    // 3b. Try metadata sub-index (precise, fast)
    auto meta_result = ctx_server.chunk_metadata.query(query);
    if (meta_result.used_metadata && !meta_result.records.empty()) {
        for (auto & rec : meta_result.records) {
            std::string text = ctx_server.chunk_bm25.get_text(rec.chunk_id);
            if (text.empty() || rec.byte_offset >= (int)text.size()) continue;
            int len = std::min(rec.byte_length, (int)text.size() - rec.byte_offset);
            std::string record = text.substr(rec.byte_offset, len);
            if (code_chars + (int)record.size() > budget.max_code_chars) break;
            std::string source = ctx_server.chunk_bm25.get_source(rec.chunk_id);
            code_context += "// Record from: " + source + "\n" + record + "\n\n";
            code_chars += (int)record.size();
        }
    } else {
        // 3b. BM25 + optional multihop
        auto bm25_ranked = ctx_server.chunk_bm25.top_k(query, 50);
#ifdef ALPHA_DENSE_L0
        // Dense-L0 union: a low-echo paraphrase ("fraud-like language") shares few words with its
        // answer, so BM25 often misses it while the semantic embedder recovers it. Interleave BM25
        // and dense so a dense-#1 hit lands near the front (survives the char budget) rather than
        // after 50 weak BM25 chunks. dense ids == chunk_bm25 ids (shared ingest).
        if (ctx_server.use_dense && ctx_server.dense.size() > 0) {
            auto dh = ctx_server.dense.search_semantic(query, 50);
            std::vector<int> merged; merged.reserve(bm25_ranked.size() + dh.size());
            std::unordered_set<int> seen;
            size_t bi = 0, di = 0;
            while (bi < bm25_ranked.size() || di < dh.size()) {
                if (bi < bm25_ranked.size()) { int c = bm25_ranked[bi++];        if (seen.insert(c).second) merged.push_back(c); }
                if (di < dh.size())          { int c = (int) dh[di++].first;     if (seen.insert(c).second) merged.push_back(c); }
            }
            fprintf(stderr, "dense-l0: BM25(%zu) union dense(%zu) -> %zu chunks\n",
                    bm25_ranked.size(), dh.size(), merged.size());
            bm25_ranked.swap(merged);
        }
#endif
        std::vector<int> ranked;
        if (use_ner && ctx_server.chunk_entities.n_unique > 0) {
            auto mh = multihop_query(query, bm25_ranked,
                                     ctx_server.chunk_entities,
                                     ctx_server.chunk_linker,
                                     ctx_server.entity_extractor, 50);
            ranked = mh.chunk_ids;
        } else {
            ranked = bm25_ranked;
        }
        if (ranked.empty()) return;

        // 4. Partition into code and experience chunks
        std::vector<int> code_ranked, exp_ranked;
        for (int cid : ranked) {
            std::string source = ctx_server.chunk_bm25.get_source(cid);
            if (source.find("experience") == 0) {
                exp_ranked.push_back(cid);
            } else {
                code_ranked.push_back(cid);
            }
        }

        // 5. NER summary (code mode only)
        if (use_ner && !code_ranked.empty()) {
            auto get_source_fn = [this](int cid) -> std::string {
                return ctx_server.chunk_bm25.get_source(cid);
            };
            auto get_text_fn = [this](int cid) -> std::string {
                return ctx_server.chunk_bm25.get_text(cid);
            };
            ner_summary = build_ner_summary_for_chunks(
                code_ranked, ctx_server.chunk_entities, ctx_server.chunk_linker,
                ctx_server.entity_extractor, get_source_fn, get_text_fn);
            if ((int)ner_summary.size() > budget.max_ner_chars) {
                ner_summary.resize(budget.max_ner_chars);
                auto last_nl = ner_summary.rfind('\n');
                if (last_nl != std::string::npos && last_nl > ner_summary.size() / 2) {
                    ner_summary.resize(last_nl + 1);
                }
            }
        }

        // 6. Fill code context (primary budget)
        // With sub-chunk BM25, deduplicate by parent file to avoid
        // injecting 16 sub-chunks from the same file.
        std::unordered_set<int> seen_parents;
        for (int cid : code_ranked) {
            int parent_id = ctx_server.chunk_bm25.get_parent_id(cid);
            if (seen_parents.count(parent_id)) continue;  // already included this file
            seen_parents.insert(parent_id);

            // Use parent source (file path without @offset)
            std::string source = ctx_server.chunk_bm25.get_parent_source(cid);
            std::string text = ctx_server.chunk_bm25.get_text(cid);
            if (text.empty()) continue;
            if (code_chars + (int)text.size() > budget.max_code_chars) break;
            code_context += "// File: " + source + "\n" + text + "\n\n";
            code_chars += (int)text.size();
        }

        // 7. Fill experience context (capped: max 3 experiences, max 5000 chars)
        for (int cid : exp_ranked) {
            if (exp_count >= budget.max_experience_count) break;
            std::string text = ctx_server.chunk_bm25.get_text(cid);
            if (text.empty()) continue;
            if (exp_chars + (int)text.size() > budget.max_experience_chars) continue;
            std::string source = ctx_server.chunk_bm25.get_source(cid);
            experience_context += "// Experience: " + source + "\n" + text + "\n\n";
            exp_chars += (int)text.size();
            exp_count++;
        }
    }

    // 8. Assemble and inject
    std::string augmented;
    if (!ner_summary.empty()) augmented += ner_summary + "\n\n";
    augmented += code_context;
    if (!experience_context.empty()) {
        augmented += "\n--- Prior Experiences ---\n" + experience_context;
    }
    if (augmented.empty()) return;

    // 9. OOM fallback: estimate total prompt tokens, truncate if needed
    int est_chars = (int)augmented.size();
    for (auto & msg : msgs) {
        if (msg.contains("content") && msg["content"].is_string())
            est_chars += (int)msg["content"].get<std::string>().size();
    }
    int est_tokens = (int)(est_chars / 3.5f) + (has_tools ? 1500 : 0) + 1024;
    if (est_tokens > meta->slot_n_ctx) {
        // Truncate: remove experiences first, then halve code
        SRV_WRN("corpus-augment: est %d tokens > ctx %d — truncating injection\n",
                est_tokens, meta->slot_n_ctx);
        augmented.clear();
        if (!ner_summary.empty()) augmented += ner_summary + "\n\n";
        // Halve code context
        if (code_context.size() > 1000) {
            code_context.resize(code_context.size() / 2);
            auto nl = code_context.rfind('\n');
            if (nl != std::string::npos) code_context.resize(nl + 1);
        }
        augmented += code_context;
        // Skip experiences entirely in truncated mode
    }

    json sys_msg = {{"role", "system"}, {"content", augmented}};
    msgs.insert(msgs.begin(), sys_msg);

    // 10. Auto-include tools if registered
    if (has_tools) {
        json tools_arr = json::array();
        for (auto & t : tools_ptr->tools) {
            tools_arr.push_back(t->get_definition());
        }
        body["tools"] = tools_arr;
        body["tool_choice"] = "auto";
    }

    if (budget.max_code_chars < 2000) {
        SRV_WRN("corpus-augment: tight budget (%d chars code) — consider larger --ctx-size\n",
                budget.max_code_chars);
    }

    SRV_INF("corpus-augment: budget=%d/%d/%d used=%d/%d/%d exp=%d ctx=%d%s\n",
            budget.max_code_chars, budget.max_ner_chars, budget.max_experience_chars,
            code_chars, (int)ner_summary.size(), exp_chars, exp_count,
            meta->slot_n_ctx,
            meta_result.used_metadata ? " [metadata]" : "");
}

// These endpoints support the training-free long-context system.
// Python handles routing/escalation intelligence. Engine provides cache control.

// Query-aware span extraction (baseline selector for the reduce-tree span-return hook).
// Compress a candidate chunk to the top-N query-overlapping sentences (verbatim, in original order),
// so a needle sentence that is not the single best-overlap sentence still survives: this stage must
// preserve recall (shorten, never drop) rather than act as a precision reranker. Returns the span
// text and sets out_offset to the byte offset of the first kept sentence in `text` (provenance).
static std::string alpha_extract_query_span(const std::string & text, const std::string & query,
                                            int max_chars, int top_sents, int & out_offset) {
    out_offset = 0;
    if (text.empty()) return "";
    // surface query terms (lowercase alphanumeric, length>2) - matched as substrings, not stemmed,
    // so they hit the raw document idiom the selector is judging.
    std::vector<std::string> qt; {
        std::string cur;
        for (char c : query) {
            if (std::isalnum((unsigned char)c)) cur += (char)std::tolower((unsigned char)c);
            else { if (cur.size() > 2) qt.push_back(cur); cur.clear(); }
        }
        if (cur.size() > 2) qt.push_back(cur);
    }
    // sentence boundaries: . ! ? followed by whitespace/end
    std::vector<std::pair<size_t,size_t>> sents;   // (start, len)
    size_t s = 0;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if ((c == '.' || c == '!' || c == '?') &&
            (i + 1 >= text.size() || std::isspace((unsigned char)text[i + 1]))) {
            sents.push_back({s, i + 1 - s});
            s = i + 1;
            while (s < text.size() && std::isspace((unsigned char)text[s])) s++;
            i = s ? s - 1 : 0;
        }
    }
    if (s < text.size()) sents.push_back({s, text.size() - s});
    if (sents.empty()) return text.substr(0, (size_t)max_chars);

    // score each sentence by query-term occurrences (lowercased substring count)
    std::vector<std::pair<int,int>> scored;            // (score, sentence index)
    scored.reserve(sents.size());
    for (size_t si = 0; si < sents.size(); si++) {
        std::string low = text.substr(sents[si].first, sents[si].second);
        for (auto & ch : low) ch = (char)std::tolower((unsigned char)ch);
        int sc = 0;
        for (auto & t : qt) {
            size_t p = 0;
            while ((p = low.find(t, p)) != std::string::npos) { sc++; p += t.size(); }
        }
        scored.push_back({sc, (int)si});
    }
    // keep the top_sents highest-scoring sentences, then restore original order
    std::sort(scored.begin(), scored.end(), [](auto & a, auto & b){ return a.first > b.first; });
    int keep = std::min(top_sents < 1 ? 1 : top_sents, (int)scored.size());
    std::vector<int> idx;
    for (int i = 0; i < keep; i++) idx.push_back(scored[i].second);
    std::sort(idx.begin(), idx.end());
    out_offset = (int)sents[idx.front()].first;
    std::string span;
    for (int i : idx) {
        if (!span.empty()) span += " ";
        span += text.substr(sents[i].first, sents[i].second);
    }
    // trim trailing whitespace and cap length
    while (!span.empty() && std::isspace((unsigned char)span.back())) span.pop_back();
    if ((int)span.size() > max_chars) span.resize((size_t)max_chars);
    return span;
}

void server_routes::init_kv_chunk_routes() {
    // All KV-chunk handlers use the task queue pattern:
    // 1. Parse request -> fill task.kv_data
    // 2. Post task to queue (processed on main thread)
    // 3. Wait for result via rd.next()
    // Direct llama_decode calls crash (concurrent with main loop).

    // /api/decode_step - single-token decode with logits for escalation
    this->post_decode_step = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = create_response();
        auto & rd = res->rd;

        server_task task(SERVER_TASK_TYPE_KV_DECODE_STEP);
        task.id = rd.get_new_id();
        task.kv_data.token_id = json_value(body, "token_id", -1);
        task.kv_data.position = json_value(body, "position", 0);
        task.kv_data.top_k = json_value(body, "top_k", 10);
        rd.post_task(std::move(task));

        auto result = rd.next(req.should_stop);
        if (!result) return res;
        res->ok(result->to_json());
        return res;
    };

    // /api/prefill_chunk - chunked prefill with position offset and seal
    this->post_prefill_chunk = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = create_response();
        auto & rd = res->rd;

        server_task task(SERVER_TASK_TYPE_KV_PREFILL_CHUNK);
        task.id = rd.get_new_id();
        task.kv_data.chunk_id = json_value(body, "chunk_id", -1);
        task.kv_data.position_offset = json_value(body, "position_offset", 0);
        task.kv_data.seal = json_value(body, "seal", true);
        task.kv_data.tokens = body.value("tokens", std::vector<llama_token>{});
        rd.post_task(std::move(task));

        auto result = rd.next(req.should_stop);
        if (!result) return res;
        res->ok(result->to_json());
        return res;
    };

    // /api/activate_chunks - select which chunks are in attention
    this->post_activate_chunks = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = create_response();
        auto & rd = res->rd;
        server_task task(SERVER_TASK_TYPE_KV_ACTIVATE_CHUNKS);
        task.id = rd.get_new_id();
        task.kv_data.active_chunk_ids = body.value("active_chunk_ids", std::vector<int>{});
        task.kv_data.include_tail = json_value(body, "include_tail", true);
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) return res;
        res->ok(result->to_json());
        return res;
    };

    // /api/kv_cache/extract_centroids - via task queue (main thread access needed)
    this->post_kv_extract_centroids = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = create_response();
        auto & rd = res->rd;
        server_task task(SERVER_TASK_TYPE_KV_EXTRACT_CENTROIDS);
        task.id = rd.get_new_id();
        task.kv_data.layer_idx = json_value(body, "layer_idx", -1);
        task.kv_data.pos_start = json_value(body, "pos_start", 0);
        task.kv_data.pos_end = json_value(body, "pos_end", -1);
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) return res;
        res->ok(result->to_json());
        return res;
    };

    // /api/kv_cache/save - via task queue
    this->post_kv_save = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = create_response();
        auto & rd = res->rd;
        server_task task(SERVER_TASK_TYPE_KV_SAVE);
        task.id = rd.get_new_id();
        task.kv_data.path = json_value(body, "path", std::string());
        task.kv_data.chunk_id = json_value(body, "chunk_id", -1);
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) return res;
        res->ok(result->to_json());
        return res;
    };

    // /api/kv_cache/load - via task queue
    this->post_kv_load = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = create_response();
        auto & rd = res->rd;
        server_task task(SERVER_TASK_TYPE_KV_LOAD);
        task.id = rd.get_new_id();
        task.kv_data.path = json_value(body, "path", std::string());
        task.kv_data.chunk_id = json_value(body, "chunk_id", -1);
        rd.post_task(std::move(task));
        auto result = rd.next(req.should_stop);
        if (!result) return res;
        res->ok(result->to_json());
        return res;
    };

    // /api/kv_chunk/index - add chunk text to BM25 index with metadata
    this->post_kv_index = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = std::make_unique<server_http_res>();
        int chunk_id = json_value(body, "chunk_id", -1);
        std::string text = json_value(body, "text", std::string());
        std::string source = json_value(body, "source", std::string());
        int position = json_value(body, "position", -1);
        int instance = json_value(body, "instance", -1);
        int total = json_value(body, "total", -1);

        // A document added after a --corpus-dir build (frozen df dictionary, and in ALPHA_BM25_INVERTED mode
        // immutable postings) goes to the BM25 post-finalize overlay, so it is searchable by all of its terms.
        if (chunk_id >= 0 && !text.empty()) {
            // Prepend metadata tags for BM25 searchability
            std::string indexed_text;
            if (total > 0) {
                indexed_text += "[Chunk " + std::to_string(position >= 0 ? position + 1 : chunk_id + 1)
                             + "/" + std::to_string(total) + "] ";
            }
            if (!source.empty()) {
                indexed_text += "[Source: " + source + "] ";
            }
            if (instance > 0) {
                indexed_text += "[Instance " + std::to_string(instance) + "] ";
            }
            indexed_text += text;

            // Layer 2: NER entity extraction (code corpus only)
            if (params.ctx_mode == ALPHA_CTX_EXTENDED_CODE) {
                auto entities = ctx_server.entity_extractor.extract(text);
                ctx_server.chunk_entities.add(chunk_id, entities);

                // Extract imports for call chain building
                auto imports = ctx_server.entity_extractor.extract_imports(text);
                if (!imports.empty()) {
                    ctx_server.chunk_linker.add_imports(source, imports);
                }

                // Append NER digest to BM25 text - makes BM25 entity-aware
                std::string ner_digest;
                for (auto & e : entities) {
                    ner_digest += " " + e.text;
                }
                for (auto & imp : imports) {
                    ner_digest += " " + imp;
                }
                if (!ner_digest.empty()) {
                    indexed_text += " [Entities:" + ner_digest + "]";
                }
            }

            ctx_server.chunk_bm25.add_document(chunk_id, indexed_text, source);
#ifdef ALPHA_DENSE_L0
            // ALPHA_DENSE_L0_DEFER_INGEST=1 builds the lexical index without embedding anything, so the
            // server can serve immediately while the dense store is still empty (the state a background
            // embedding pass starts from). Queries still work: the coverage-proportional interleave sees
            // rho=0 and keeps the lexical order, and query-time rescoring embeds only the candidates it needs.
            static const bool defer_dense = [] {
                const char * e = getenv("ALPHA_DENSE_L0_DEFER_INGEST");
                return e && atoi(e) != 0;
            }();
            if (ctx_server.use_dense && !defer_dense) ctx_server.dense.add((uint32_t) chunk_id, indexed_text);
#endif

            // Metadata sub-index
            ctx_server.chunk_metadata.index_chunk(chunk_id, text);
        }
        json result = {
            {"indexed", true}, {"chunk_id", chunk_id},
            {"n_chunks", ctx_server.chunk_bm25.N},
            {"n_words", (int)bm25_index::tokenize(text).size()},
            {"n_entities", ctx_server.chunk_entities.n_unique},
        };
        res->status = 200;
        res->data = safe_json_to_str(result);
        return res;
    };

    // POST /api/corpus/finalize {} -> pack df, and with ALPHA_BM25_INVERTED=1 build the inverted
    // postings. Mirrors the tail of the --corpus-dir auto-ingest path, which was the only place this
    // sequence ran; HTTP ingest is incremental and has no end-of-ingest signal, so the caller signals
    // it here. Idempotent: finalize_df() and build_inverted() both early-return once done. Documents
    // indexed afterwards stay searchable via the post-finalize overlay.
    this->post_corpus_finalize = [this](const server_http_req & req) {
        (void) req;
        auto res = std::make_unique<server_http_res>();
        const bool   want_inverted = bm25_index::bm25_inverted();
        const size_t n_before      = ctx_server.chunk_bm25.N;
        auto t0 = std::chrono::steady_clock::now();

        ctx_server.chunk_bm25.finalize_text_store();
        ctx_server.chunk_bm25.finalize_text_offsets();
        ctx_server.chunk_bm25.finalize_df();

        float build_ms = 0.0f;
        if (want_inverted) {
            const char * td = getenv("TMPDIR");
            std::string tmpdir = (td && *td) ? td : "/tmp";
            std::string ppath = tmpdir + "/alpha_bm25_" + std::to_string((long) getpid()) + ".post.bin";
            auto iv0 = std::chrono::steady_clock::now();
            const bool was_spill = ctx_server.chunk_bm25.spill_active;
            ctx_server.chunk_bm25.build_inverted(ppath);
            build_ms = (float) std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - iv0).count();
            SRV_INF("BM25 INVERTED (%s, http-finalize): %s - %zu postings bytes, built in %.1fs\n",
                    was_spill ? "external/spill" : "resident",
                    ctx_server.chunk_bm25.inverted_active ? "active" : "FAILED (stayed on forward path)",
                    ctx_server.chunk_bm25.post_mm_len, build_ms / 1000.0f);
        }
        ctx_server.chunk_metadata.finalize();

        json result = {
            {"finalized",          true},
            {"n_chunks",           ctx_server.chunk_bm25.N},
            {"n_chunks_before",    (int) n_before},
            {"inverted_requested", want_inverted},
            {"inverted_active",    ctx_server.chunk_bm25.inverted_active},
            {"build_ms",           build_ms},
            {"total_ms",           (float) std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::steady_clock::now() - t0).count()},
        };
        res->status = 200;
        res->data = safe_json_to_str(result);
        return res;
    };

    // /api/kv_chunk/route - BM25 query routing with context filling
    // Supports multi-hop entity expansion when entity index is populated.
    this->post_kv_route = [this](const server_http_req & req) {
        json body = json::parse(req.body);
        auto res = std::make_unique<server_http_res>();
        std::string query = json_value(body, "query", std::string());
        // Corpus-type-aware defaults
        bool is_code = (params.ctx_mode == ALPHA_CTX_EXTENDED_CODE);
        int k_default = is_code ? 50 : (ctx_server.chunk_bm25.N <= 10000000) ? 50 : 20;
        int k = json_value(body, "k", k_default);
        bool use_multihop = json_value(body, "multihop", is_code);  // auto-enable for code
        int max_evidence = json_value(body, "max_evidence", 50);
        // Per-request rare-bigram-first override: -1 (default) = follow ALPHA_BM25_BIGRAM_FIRST.
        // The env is read once at startup, so this allows comparing both settings without a re-ingest.
        int bigram_first = json_value(body, "bigram_first", -1);

        // Union mode: candidates = plain-BM25 top-k1 union rare-bigram-first top-k2. The plain arm keeps
        // the lexical leader in place, the bigram arm rescues paraphrase needles the plain score buries;
        // neither is safe alone (hence bigram-first defaults off), the union is. Plain results come first
        // (rank-preserving), bigram-only extras are appended. The rank a needle reaches in the bigram arm
        // varies widely with corpus and seed, so the default k2 is generous: 100 at ~6.7M sub-chunks
        // (~2B tokens), sqrt-scaled with corpus size, min 60. k2 is cheap on the inverted index; callers
        // with a tight injection budget should set k1/k2 explicitly.
        std::string mode = json_value(body, "mode", std::string());
        std::vector<int> bm25_ranked;
        // Query-time dense rescoring needs the BM25 score, which top_k() discards. Populated from
        // top_k_scored() below (non-union path only; union mode has no single score scale).
        std::vector<std::pair<float, int>> bm25_scored;
        bool lex_underfill = false;   // set on the scored path only: fewer than k documents matched any query term
        int u_k1 = 0, u_k2 = 0;
        double dense_rho_used = -1.0;    // reported back so a run can be checked against its arm
        int qtd_hits = 0, qtd_embedded = 0, qtd_scored = 0;   // self-describing response
        // "qtd_debug": true returns the rescoring pool in its PRE-rescore (interleave) order with each candidate's raw
        // BM25 score and raw cosine (null = absent, i.e. imputed with the pool mean), for offline fusion analysis.
        json qtd_dbg = json::array();
        if (mode == "union") {
            u_k1 = json_value(body, "k1", 30);
            u_k2 = json_value(body, "k2", 0);
            if (u_k2 <= 0) {
                double scale = (double)ctx_server.chunk_bm25.N / 6720000.0;   // 6.72M sub-chunks ~ 2B tok
                u_k2 = std::max(60, (int)std::ceil(100.0 * std::pow(std::max(0.01, scale), 0.5)));
            }
            auto plain  = ctx_server.chunk_bm25.top_k(query, u_k1, 0);
            auto bigram = ctx_server.chunk_bm25.top_k(query, u_k2, 1);
            std::unordered_set<int> seen;
            bm25_ranked.reserve(plain.size() + bigram.size());
            for (int id : plain)  if (seen.insert(id).second) bm25_ranked.push_back(id);
            for (int id : bigram) if (seen.insert(id).second) bm25_ranked.push_back(id);
        } else {
            // top_k_scored() keeps the score top_k() throws away, for the same cost (one score()
            // call). Its truncation mirrors top_k() exactly, including k<=0 => empty.
            bm25_scored = ctx_server.chunk_bm25.top_k_scored(query, k, bigram_first);
            lex_underfill = (int) bm25_scored.size() < k;   // the scorer saw every document and fewer than k matched
            bm25_ranked.reserve(bm25_scored.size());
            for (const auto & p : bm25_scored) bm25_ranked.push_back(p.second);
        }
#ifdef ALPHA_DENSE_L0
        // Dense-L0 union (same interleave as the chat RAG path): recover the low-echo paraphrase tail
        // that lexical BM25 misses. dense ids == chunk_bm25 ids (shared ingest).
        if (ctx_server.use_dense && ctx_server.dense.size() > 0) {
            auto dh = ctx_server.dense.search_semantic(query, std::max(k, 50));
            std::vector<int> merged; merged.reserve(bm25_ranked.size() + dh.size());
            std::unordered_set<int> dseen; size_t bi = 0, di = 0;

            // Coverage-proportional interleave. A 1:1 interleave assumes the dense arm has seen the
            // whole corpus. With a partial index (incremental ingest, background embedding pass) the
            // dense top-50 comes from whatever fraction is embedded yet would still claim every other
            // front slot, and at low coverage that ranks worse than no dense arm at all. So the dense
            // arm gets a share of the slots equal to its share of the corpus: rho = dense.size()/n_chunks,
            // emit bm25:dense in ratio 1:rho. rho >= 1 runs the plain 1:1 loop.
            //
            // fuse_rho: per-request override, -1 = derive from dense.size()/n_chunks, 1 = force 1:1.
            // Also settable via ALPHA_DENSE_L0_FUSE_RHO.
            double fuse_rho = json_value(body, "fuse_rho", -1.0);
            if (fuse_rho < 0.0) {
                if (const char * fr = getenv("ALPHA_DENSE_L0_FUSE_RHO")) fuse_rho = atof(fr);
            }
            if (fuse_rho < 0.0) {
                const size_t nch = (size_t) ctx_server.chunk_bm25.N;
                fuse_rho = (nch > 0) ? (double) ctx_server.dense.size() / (double) nch : 1.0;
            }
            if (fuse_rho > 1.0) fuse_rho = 1.0;

            if (fuse_rho >= 1.0) {
                // Full coverage: 1:1 interleave.
                while (bi < bm25_ranked.size() || di < dh.size()) {
                    if (bi < bm25_ranked.size()) { int c = bm25_ranked[bi++];    if (dseen.insert(c).second) merged.push_back(c); }
                    if (di < dh.size())          { int c = (int) dh[di++].first; if (dseen.insert(c).second) merged.push_back(c); }
                }
            } else if (fuse_rho <= 0.0) {
                // Nothing embedded: keep the lexical order. This also keeps the accumulator below
                // from spinning.
                merged.assign(bm25_ranked.begin(), bm25_ranked.end());
            } else {
                // 1:rho. `acc` earns a dense slot every 1/rho bm25 slots. The rho<=0 case is handled
                // above, so acc strictly increases and `di` always advances: no unbounded loop.
                double acc = 0.0;
                while (bi < bm25_ranked.size()) {
                    int c = bm25_ranked[bi++];
                    if (dseen.insert(c).second) merged.push_back(c);
                    acc += fuse_rho;
                    while (acc >= 1.0 && di < dh.size()) {
                        acc -= 1.0;
                        int d = (int) dh[di++].first;
                        if (dseen.insert(d).second) merged.push_back(d);
                    }
                }
                // bm25 exhausted. Do not drain the rest of dh: that is exactly the flooding this
                // guards against, and the caller truncates to k anyway.
            }
            bm25_ranked.swap(merged);
            dense_rho_used = fuse_rho;
        }

        // Query-time dense rescoring. The interleave above decides which candidates; this decides their
        // order, using the dense cosine as a score rather than a rank (the interleave always puts the
        // lexical pick at rank 1). Vectors come from the store, or are embedded here for just the
        // candidates, which is cheap compared with embedding the whole corpus; the full index mainly
        // adds recall, not better top-10 ordering.
        //
        // Score = w * cosine + (1-w) * BM25, both min-max normalised over the pool. Default w = 0.7.
        // Off: request "qtd_w": -1, or ALPHA_DENSE_L0_QTD_W=-1. A request value wins over the env, the
        // env over 0.7. An unparseable env value reads as 0 (atof), which turns rescoring off; the
        // response field dense.qtd_w_eff reports the weight actually used (-1 = off).
        //
        // Coverage guard: rescore only when coverage (dense.size()/n_chunks, never the per-request
        // fuse_rho) is high enough. On a partial index most pool members have no stored cosine and
        // would be ordered by the imputed pool mean. "qtd_guard": false disables it (experiments only).
        double qtd_w = 0.7;
        if (body.contains("qtd_w")) {
            qtd_w = json_value(body, "qtd_w", -1.0);
        } else if (const char * e = getenv("ALPHA_DENSE_L0_QTD_W")) {
            qtd_w = atof(e);
        }
        double qtd_w_eff = -1.0;
        if (qtd_w >= 0.0) {
            const size_t nch = (size_t) ctx_server.chunk_bm25.N;
            double cov = (nch > 0) ? (double) ctx_server.dense.size() / (double) nch : 0.0;
            if (cov > 1.0) cov = 1.0;
            // Scaling the weight by coverage is not safe: an intermediate weight shuffles the lexical order
            // without imposing a better one. So run at full w when coverage >= ALPHA_DENSE_L0_QTD_MIN_COV
            // (default 0.9), otherwise do not rescore at all.
            static const double min_cov = [] {
                const char * e = getenv("ALPHA_DENSE_L0_QTD_MIN_COV");
                return e ? atof(e) : 0.9;
            }();
            qtd_w_eff = !json_value(body, "qtd_guard", true) ? qtd_w : (cov + 1e-9 >= min_cov ? qtd_w : -1.0);
            qtd_w = qtd_w_eff;
        }
        if (qtd_w > 0.0 && ctx_server.dense.ready() && !bm25_ranked.empty()) {
            const int dim  = ctx_server.dense.dim();
            const int pool = std::min((int) bm25_ranked.size(),
                                      std::max(1, json_value(body, "qtd_pool", 50)));
            // Cap on-demand embedding so a cold store cannot blow the latency budget. 0 = use only
            // what is already stored. Misses beyond the cap stay at their lexical position.
            const int max_embed = std::max(0, json_value(body, "qtd_max_embed", 0));
            // qtd_cache=true writes the vectors embedded here back to the dense index. Safe under concurrent
            // queries (dense_index appends under its exclusive lock, searches take it shared); appended vectors
            // join the IVF's scanned tail until the next build_ann. Off by default.
            const bool cache = json_value(body, "qtd_cache", false);

            std::vector<float> qv = ctx_server.dense.embed(query, /*is_query*/ true);
            if ((int) qv.size() == dim && dim > 0) {
                auto cosine = [&](const std::vector<float> & v) {
                    double s = 0.0;
                    for (int j = 0; j < dim; j++) s += (double) v[j] * (double) qv[j];
                    return (float) s;
                };
                std::unordered_map<int, float> cmap;
                std::vector<int> miss;
                std::vector<float> v;
                for (int i = 0; i < pool; i++) {
                    const int c = bm25_ranked[i];
                    if (ctx_server.dense.get_vec((uint32_t) c, v) && (int) v.size() == dim) {
                        cmap[c] = cosine(v);
                        qtd_hits++;
                    } else if ((int) miss.size() < max_embed) {
                        miss.push_back(c);
                    }
                }
                if (!miss.empty()) {
                    std::vector<std::string> texts;
                    texts.reserve(miss.size());
                    for (int c : miss) texts.push_back(ctx_server.chunk_bm25.get_text(c));
                    auto vs = ctx_server.dense.embed_batch(texts, /*is_query*/ false);
                    for (size_t i = 0; i < miss.size() && i < vs.size(); i++) {
                        if ((int) vs[i].size() != dim) continue;
                        cmap[miss[i]] = cosine(vs[i]);
                        qtd_embedded++;
                        if (cache) ctx_server.dense.add_vec((uint32_t) miss[i], vs[i]);
                    }
                }
                qtd_scored = (int) cmap.size();

                // Reorder only the pool; the tail keeps its lexical order. Min-max both signals over
                // the pool so the weight means the same thing whatever the score scales are.
                if (qtd_scored > 1) {
                    std::unordered_map<int, float> bmap;
                    for (const auto & p : bm25_scored) bmap[p.second] = p.first;
                    float blo = 0.0f, bhi = 0.0f, clo = 0.0f, chi = 0.0f;
                    bool bfirst = true, cfirst = true;
                    for (int i = 0; i < pool; i++) {
                        auto ib = bmap.find(bm25_ranked[i]);
                        if (ib != bmap.end()) {
                            if (bfirst) { blo = bhi = ib->second; bfirst = false; }
                            else { blo = std::min(blo, ib->second); bhi = std::max(bhi, ib->second); }
                        }
                        auto ic = cmap.find(bm25_ranked[i]);
                        if (ic != cmap.end()) {
                            if (cfirst) { clo = chi = ic->second; cfirst = false; }
                            else { clo = std::min(clo, ic->second); chi = std::max(chi, ic->second); }
                        }
                    }
                    // Underfill: a known-0 bm25 is a real member of the population, so the min-max range includes it:
                    // documents with no lexical evidence then normalise to 0, strictly below every matching document.
                    if (lex_underfill) blo = 0.0f;
                    const float brng = bhi - blo, crng = chi - clo;
                    // A candidate with no cosine (not stored, and past the embed cap) gets the pool
                    // mean, so it is neither promoted nor buried by its absence. Treating it as 0
                    // would push every un-embedded chunk to the back, the same flooding the rho
                    // interleave prevents.
                    double csum = 0.0; size_t cn_n = 0;
                    for (const auto & kv : cmap) {
                        csum += (crng > 1e-12f) ? (kv.second - clo) / crng : 0.0; cn_n++;
                    }
                    const float cmean = (float) (csum / std::max<size_t>(cn_n, 1));
                    // Same for the BM25 side: the interleaved list contains dense-arm candidates that
                    // were never in the BM25 ranking and have no bm25 score. Defaulting those to 0 would
                    // sort every dense-only candidate to the back and discard the dense arm's
                    // contribution, so an absent score also gets the pool mean.
                    double bsum = 0.0; size_t bn_n = 0;
                    for (int i = 0; i < pool; i++) {
                        auto ib2 = bmap.find(bm25_ranked[i]);
                        if (ib2 == bmap.end()) continue;
                        bsum += (brng > 1e-12f) ? (ib2->second - blo) / brng : 0.0; bn_n++;
                    }
                    // Except when the lexical list underfilled: then an absent bm25 is known to be 0 (the scorer saw
                    // every document and this one matched no query term). With 0 inside the range (blo = 0 above) it
                    // normalises to 0, strictly below every matching document; the pool mean would promote such
                    // dense-only candidates above genuine matches. A zero default without underfill ranks worse,
                    // hence the condition.
                    const float bmean = lex_underfill ? 0.0f : (float) (bsum / std::max<size_t>(bn_n, 1));
                    auto fused = [&](int c) {
                        float bn = bmean, cn = cmean;
                        auto ib = bmap.find(c);
                        if (ib != bmap.end()) bn = (brng > 1e-12f) ? (ib->second - blo) / brng : 0.0f;
                        auto ic = cmap.find(c);
                        if (ic != cmap.end()) cn = (crng > 1e-12f) ? (ic->second - clo) / crng : 0.0f;
                        return (float) ((1.0 - qtd_w) * bn + qtd_w * cn);
                    };
                    if (json_value(body, "qtd_debug", false)) {
                        for (int i = 0; i < pool; i++) {
                            const int c = bm25_ranked[i];
                            auto ib = bmap.find(c);
                            auto ic = cmap.find(c);
                            qtd_dbg.push_back({{"id", c},
                                               {"bm25", ib != bmap.end() ? json(ib->second) : json(nullptr)},
                                               {"cos",  ic != cmap.end() ? json(ic->second) : json(nullptr)}});
                        }
                    }
                    std::stable_sort(bm25_ranked.begin(), bm25_ranked.begin() + pool,
                                     [&](int a, int b) { return fused(a) > fused(b); });
                }
            }
        }
#endif

        // Multi-hop entity expansion (Layer 2)
        std::vector<int> ranked;
        int hop1 = 0, hop2 = 0, hop3 = 0;
        if (use_multihop && ctx_server.chunk_entities.n_unique > 0) {
            auto mh = multihop_query(query, bm25_ranked,
                                     ctx_server.chunk_entities,
                                     ctx_server.chunk_linker,
                                     ctx_server.entity_extractor,
                                     max_evidence);
            ranked = mh.chunk_ids;
            hop1 = mh.hop1_count;
            hop2 = mh.hop2_count;
            hop3 = mh.hop3_count;
        } else {
            ranked = bm25_ranked;
            hop1 = (int)ranked.size();
        }

        std::string routing_name = ctx_server.routing_strategy == kv_chunk_routing::BM25_EXPAND
                                   ? "bm25+expand" : "bm25_only";
        if (use_multihop && ctx_server.chunk_entities.n_unique > 0) {
            routing_name += "+entity";
        }

        json result = json::object();
        result["chunk_ids"] = ranked;
        result["n_chunks_indexed"] = ctx_server.chunk_bm25.N;
        result["n_entities_indexed"] = ctx_server.chunk_entities.n_unique;
        result["routing"] = routing_name;
        result["ctx_extended"] = ctx_server.n_ctx_extended;
        result["hops"] = {{"bm25", hop1}, {"entity", hop2}, {"links", hop3}};
        if (mode == "union") result["union"] = {{"k1", u_k1}, {"k2", u_k2}};
#ifdef ALPHA_DENSE_L0
        // Report what the dense arm actually did, so a run is self-describing. `rho_used` is -1 when
        // the dense branch never ran (embedder absent, or dense.size() == 0).
        result["dense"] = {{"size", (uint64_t) ctx_server.dense.size()},
                           {"use_dense", ctx_server.use_dense},
                           {"rho_used", dense_rho_used},
                           {"qtd_cache_hits", qtd_hits},
                           {"qtd_embedded", qtd_embedded},
                           {"qtd_scored", qtd_scored},
                           {"qtd_w_eff", qtd_w_eff},
                           // false at >= ALPHA_DENSE_L0_ANN_MIN vectors = brute force (e.g. HTTP ingest without
                           // POST /api/dense/build_ann)
                           {"ann_active", ctx_server.dense.ann_ready()}};
        if (!qtd_dbg.empty()) result["dense"]["qtd_debug"] = qtd_dbg;
#endif

#ifdef ALPHA_INTERNAL
        // Text retrieval only available in internal builds (benchmark scripts).
        // Production builds return chunk IDs only - server handles retrieval
        // and inference internally via context_mode=extended.
        int max_context = json_value(body, "max_context_chars", 0);
        bool include_text = json_value(body, "include_text", false);
        if (max_context > 0 && include_text) {
            auto sorted_ids = ranked;
            std::sort(sorted_ids.begin(), sorted_ids.end());
            json chunks_arr = json::array();
            int total_chars = 0;
            for (int cid : sorted_ids) {
                std::string text = ctx_server.chunk_bm25.get_text(cid);
                if (text.empty()) continue;
                if (total_chars + (int)text.size() > max_context) break;
                chunks_arr.push_back({{"chunk_id", cid}, {"text", text}});
                total_chars += (int)text.size();
            }
            result["context_chunks"] = chunks_arr;
            result["context_chars"] = total_chars;
        }

        // Span-return mode: per candidate, the best query span + provenance (chunk_id, source, byte
        // offset, lengths), in rank order. Feeds an external reduce tree that turns each candidate into
        // a verbatim span for fan-in. `span_selector` is pluggable (default: the overlap baseline above).
        // Gated to ALPHA_INTERNAL like text retrieval, since spans are document text.
        if (json_value(body, "spans", false)) {
            int span_chars = json_value(body, "span_chars", 220);
            int span_sents = json_value(body, "span_sentences", 2);
            json spans_arr = json::array();
            int rnk = 0;
            for (int cid : ranked) {
                rnk++;
                std::string txt = ctx_server.chunk_bm25.get_text(cid);
                if (txt.empty()) continue;
                int off = 0;
                std::string span = alpha_extract_query_span(txt, query, span_chars, span_sents, off);
                spans_arr.push_back({
                    {"chunk_id", cid}, {"rank", rnk},
                    {"source", ctx_server.chunk_bm25.get_parent_source(cid)},
                    {"span", span}, {"offset", off},
                    {"span_len", (int)span.size()}, {"chunk_len", (int)txt.size()},
                });
            }
            result["spans"] = spans_arr;
        }
#endif

        res->status = 200;
        res->data = safe_json_to_str(result);
        return res;
    };

    // /api/expert_routing - return per-layer expert routing data from last generation
    // Used by serve-based profiling to build domain profiles without reloading the model.
    // Call this after a /v1/chat/completions request to get routing data for that prompt.
    // Enable argsort persist so routing data survives after decode.
    this->post_expert_routing = [this](const server_http_req & req) {
        // Ensure argsort data persists for future requests
        alpha_mole_set_argsort_persist(true);
        auto res = std::make_unique<server_http_res>();
        int n_layers = alpha_mole_get_argsort_count();
        if (n_layers <= 0) {
            res->status = 200;
            res->data = safe_json_to_str(json{{"layers", json::array()}, {"error", "no routing data available"}});
            return res;
        }

        json layers = json::array();
        for (int il = 0; il < n_layers; il++) {
            ggml_tensor * argsort_out = alpha_mole_get_argsort_output(il);
            if (!argsort_out || !argsort_out->buffer) continue;

            int n_used = (int)argsort_out->ne[0];   // n_expert_used
            int n_tokens = (int)argsort_out->ne[1];  // tokens in batch
            if (n_tokens <= 0 || n_used <= 0) continue;

            // Read all tokens' expert IDs (for frequency counting)
            // Each token has n_used expert IDs (the top-k selected experts)
            std::vector<int32_t> all_ids(n_used * n_tokens);
            ggml_backend_tensor_get(argsort_out, all_ids.data(), 0, all_ids.size() * sizeof(int32_t));

            // Count expert frequencies across all tokens
            std::map<int, int> freq;
            int valid = 0;
            for (int t = 0; t < n_tokens; t++) {
                for (int e = 0; e < n_used; e++) {
                    int32_t eid = all_ids[t * n_used + e];
                    if (eid >= 0 && eid < 1024) {  // sanity check
                        freq[eid]++;
                        valid++;
                    }
                }
            }

            // Sort by frequency (descending)
            std::vector<std::pair<int, int>> sorted_experts(freq.begin(), freq.end());
            std::sort(sorted_experts.begin(), sorted_experts.end(),
                      [](const auto & a, const auto & b) { return a.second > b.second; });

            json top_experts = json::array();
            json frequencies = json::array();
            for (const auto & [eid, count] : sorted_experts) {
                top_experts.push_back(eid);
                frequencies.push_back((float)count / std::max(valid, 1));
            }

            layers.push_back(json{
                {"layer", il},
                {"top_experts", top_experts},
                {"frequencies", frequencies},
                {"n_tokens", n_tokens},
                {"n_expert_used", n_used},
            });
        }

        json result = json{
            {"layers", layers},
            {"n_layers", n_layers},
        };
        res->status = 200;
        res->data = safe_json_to_str(result);
        return res;
    };

#ifdef ALPHA_DELTANET
    // DeltaNet checkpoint load: dynamically load a checkpoint into recurrent state
    // Used by the oracle pipeline: load nearest checkpoint before generating hint
    this->post_deltanet_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "invalid JSON"}});
            return res;
        }

        std::string path = body.value("path", "");
        if (path.empty()) {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "provide 'path' to checkpoint file"}});
            return res;
        }

        auto * llama_ctx = ctx_server.get_ctx();
        if (!llama_ctx) {
            res->status = 503;
            res->data = safe_json_to_str(json{{"error", "model not loaded"}});
            return res;
        }
        uint64_t tokens = alpha_deltanet_state_load(path, llama_ctx);
        if (tokens > 0) {
            SRV_INF("deltanet/load: loaded checkpoint %s (%llu tokens)\n",
                    path.c_str(), (unsigned long long)tokens);
            res->status = 200;
            res->data = safe_json_to_str(json{
                {"status", "loaded"}, {"path", path}, {"tokens", tokens}});
        } else {
            res->status = 500;
            res->data = safe_json_to_str(json{
                {"error", "failed to load checkpoint"}, {"path", path}});
        }
        return res;
    };
#endif

    // Structured-record analytics.
    // Retrieve-then-inject can only surface the top-k it injects; it can never scan/sum/
    // group-by a whole corpus. These two endpoints run the reduction in Redis over all
    // records and return a small table that fits the injection budget (query-and-narrate).
    //
    // POST /api/records/index      {index, tags:[...], numerics:[...], records:[{id, fields:{...}}]}
    // POST /api/records/aggregate  {index, metric, field, group_by, filters:[{field,lo,hi}]}
    auto records_field_to_str = [](const json & v) -> std::string {
        if (v.is_string())         return v.get<std::string>();
        if (v.is_number_integer()) return std::to_string(v.get<long long>());
        if (v.is_number_float())   { char b[64]; snprintf(b, sizeof(b), "%.10g", v.get<double>()); return b; }
        if (v.is_boolean())        return v.get<bool>() ? "1" : "0";
        return v.is_null() ? std::string() : v.dump();
    };

    this->post_records_index = [this, records_field_to_str](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400; res->data = safe_json_to_str(json{{"error", "invalid JSON"}}); return res;
        }
#ifdef ALPHA_HAS_REDIS
        if (!ctx_server.redis || !ctx_server.use_redis) {
            res->status = 503; res->data = safe_json_to_str(json{{"error", "redis backend not enabled (launch with --redis host:port)"}}); return res;
        }
        std::string index = body.value("index", "");
        if (index.empty()) { res->status = 400; res->data = safe_json_to_str(json{{"error", "provide 'index'"}}); return res; }

        // Define/re-create the index if a schema was supplied.
        if (body.contains("tags") || body.contains("numerics")) {
            std::vector<std::string> tags, numerics;
            for (const auto & t : body.value("tags", json::array()))     tags.push_back(t.get<std::string>());
            for (const auto & n : body.value("numerics", json::array())) numerics.push_back(n.get<std::string>());
            if (!ctx_server.redis->create_record_index(index, tags, numerics)) {
                res->status = 500; res->data = safe_json_to_str(json{{"error", "FT.CREATE failed for '" + index + "'"}}); return res;
            }
        }

        // Ingest records.
        size_t n = 0;
        for (const auto & rec : body.value("records", json::array())) {
            std::string id = rec.value("id", "");
            if (id.empty()) continue;
            std::vector<std::pair<std::string, std::string>> fields;
            fields.emplace_back("id", id);
            for (const auto & kv : rec.value("fields", json::object()).items()) {
                fields.emplace_back(kv.key(), records_field_to_str(kv.value()));
            }
            ctx_server.redis->add_record(index, id, fields);
            n++;
        }
        ctx_server.redis->flush();
        res->status = 200;
        res->data = safe_json_to_str(json{{"status", "ok"}, {"index", index}, {"ingested", n}});
#else
        res->status = 501; res->data = safe_json_to_str(json{{"error", "redis backend not compiled (build with ALPHA_HAS_REDIS)"}});
#endif
        return res;
    };

    this->post_records_aggregate = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400; res->data = safe_json_to_str(json{{"error", "invalid JSON"}}); return res;
        }
#ifdef ALPHA_HAS_REDIS
        if (!ctx_server.redis || !ctx_server.use_redis) {
            res->status = 503; res->data = safe_json_to_str(json{{"error", "redis backend not enabled (launch with --redis host:port)"}}); return res;
        }
        std::string index    = body.value("index", "");
        std::string metric   = body.value("metric", "");
        std::string field    = body.value("field", "");
        std::string group_by = body.value("group_by", "");
        if (index.empty() || metric.empty() || group_by.empty()) {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "provide 'index', 'metric', and 'group_by'"}});
            return res;
        }
        std::vector<redis_agg_filter> filters;
        for (const auto & f : body.value("filters", json::array())) {
            redis_agg_filter rf;
            rf.field = f.value("field", "");
            // lo/hi accept string or number
            auto as_str = [](const json & v) -> std::string {
                if (v.is_string()) return v.get<std::string>();
                if (v.is_number_integer()) return std::to_string(v.get<long long>());
                if (v.is_number_float()) { char b[64]; snprintf(b, sizeof(b), "%.10g", v.get<double>()); return b; }
                return "";
            };
            if (f.contains("lo")) rf.lo = as_str(f["lo"]);
            if (f.contains("hi")) rf.hi = as_str(f["hi"]);
            if (f.contains("value")) rf.lo = as_str(f["value"]);  // tag equality alias
            if (!rf.field.empty()) filters.push_back(rf);
        }

        redis_agg_result agg = ctx_server.redis->aggregate(index, metric, field, group_by, filters);
        if (!agg.ok) {
            // Validation failures (unknown/hallucinated field, bad metric) are 400, not 500 -
            // the tool-call layer surfaces the error back to the model to re-ground its query.
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", agg.error}, {"index", index}});
            return res;
        }
        json rows = json::array();
        for (const auto & r : agg.rows) rows.push_back(json{{"group", r.group}, {"value", r.value}});
        res->status = 200;
        res->data = safe_json_to_str(json{
            {"ok", true}, {"index", index}, {"metric", metric}, {"field", field},
            {"group_by", group_by}, {"rows", rows}, {"latency_ms", agg.latency_ms}});
#else
        res->status = 501; res->data = safe_json_to_str(json{{"error", "redis backend not compiled (build with ALPHA_HAS_REDIS)"}});
#endif
        return res;
    };

    // Dense-L0 semantic retrieval: the paraphrase tail BM25 can't serve
    // POST /api/dense/index   {chunks:[{id, text}]}         -> embed passages + store
    // POST /api/dense/search  {query, k, is_query=true}     -> semantic top-k (id, score)
    this->post_dense_index = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400; res->data = safe_json_to_str(json{{"error", "invalid JSON"}}); return res;
        }
#ifdef ALPHA_DENSE_L0
        if (!ctx_server.use_dense) {
            res->status = 503; res->data = safe_json_to_str(json{{"error", "dense-l0 not enabled (set ALPHA_DENSE_L0_EMBED_MODEL)"}}); return res;
        }
        size_t n = 0;
        for (const auto & c : body.value("chunks", json::array())) {
            if (!c.contains("id") || !c.contains("text")) continue;
            ctx_server.dense.add((uint32_t) c["id"].get<uint64_t>(), c["text"].get<std::string>());
            n++;
        }
        res->status = 200;
        res->data = safe_json_to_str(json{{"status", "ok"}, {"ingested", n}, {"total", ctx_server.dense.size()}});
#else
        res->status = 501; res->data = safe_json_to_str(json{{"error", "dense-l0 not compiled (build with ALPHA_DENSE_L0)"}});
#endif
        return res;
    };

    // POST /api/dense/build_ann {} -> (re)build the IVF index over the current store. HTTP ingest (/api/corpus/index,
    // /api/dense/index) never builds it: build_ann otherwise runs only at the --corpus-dir finalize and after a restore,
    // so a store filled over HTTP is brute-forced at any size. Call this once an HTTP ingest is complete. A no-op below
    // ALPHA_DENSE_L0_ANN_MIN (default 1,048,576 vectors). Blocks searches while it runs (exclusive store lock).
    this->post_dense_build_ann = [this](const server_http_req & req) {
        (void) req;
        auto res = std::make_unique<server_http_res>();
#ifdef ALPHA_DENSE_L0
        if (!ctx_server.use_dense) {
            res->status = 503; res->data = safe_json_to_str(json{{"error", "dense-l0 not enabled (set ALPHA_DENSE_L0_EMBED_MODEL)"}}); return res;
        }
        const auto t0 = std::chrono::steady_clock::now();
        ctx_server.dense.build_ann();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        res->status = 200;
        res->data = safe_json_to_str(json{{"ann_active", ctx_server.dense.ann_ready()}, {"nlist", ctx_server.dense.ann_nlists()},
                                          {"vectors", ctx_server.dense.size()}, {"ann_min", dense_index::ann_min()},
                                          {"build_s", s}});
#else
        res->status = 501; res->data = safe_json_to_str(json{{"error", "dense-l0 not compiled (build with ALPHA_DENSE_L0)"}});
#endif
        return res;
    };

    this->post_dense_search = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400; res->data = safe_json_to_str(json{{"error", "invalid JSON"}}); return res;
        }
#ifdef ALPHA_DENSE_L0
        if (!ctx_server.use_dense) {
            res->status = 503; res->data = safe_json_to_str(json{{"error", "dense-l0 not enabled (set ALPHA_DENSE_L0_EMBED_MODEL)"}}); return res;
        }
        std::string query = body.value("query", "");
        if (query.empty()) { res->status = 400; res->data = safe_json_to_str(json{{"error", "provide 'query'"}}); return res; }
        int  k        = body.value("k", 10);
        bool is_query = body.value("is_query", true);   // prepend the bge retrieval instruction
        std::vector<float> qv = ctx_server.dense.embed(query, is_query);
        auto hits = ctx_server.dense.search_vec(qv, k);
        json arr = json::array();
        for (auto & h : hits) arr.push_back(json{{"id", h.first}, {"score", h.second}});
        res->status = 200;
        res->data = safe_json_to_str(json{{"query", query}, {"k", k}, {"total_indexed", ctx_server.dense.size()}, {"results", arr}});
#else
        res->status = 501; res->data = safe_json_to_str(json{{"error", "dense-l0 not compiled (build with ALPHA_DENSE_L0)"}});
#endif
        return res;
    };

    // Oracle endpoint: 0.8B error assistant for user help
    this->post_oracle = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "invalid JSON"}});
            return res;
        }

        std::string error_msg = body.value("error", "");
        std::string context   = body.value("context", "");
        std::string question  = body.value("question", "");

        // Build the query - support both error explanation and general help
        std::string query;
        if (!error_msg.empty()) {
            query = error_msg;
        } else if (!question.empty()) {
            query = question;
        } else {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "provide 'error' or 'question' field"}});
            return res;
        }

        std::string explanation = !error_msg.empty()
            ? alpha_error_explain(query, "", context)
            : alpha_oracle_answer(query);

        res->status = 200;
        res->data = safe_json_to_str(json{
            {"explanation", explanation},
            {"oracle_used", alpha_error_assistant::available()},
        });
        return res;
    };

    // KV Pool endpoints

    // GET /api/kv_pool/stats - pool usage statistics
    this->get_kv_pool_stats = [this](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->data = safe_json_to_str(json{
            {"enabled",      ctx_server.get_kv_pool().is_enabled()},
            {"cpu_entries",  (int)ctx_server.get_kv_pool().cpu_entries()},
            {"cpu_used_mb",  ctx_server.get_kv_pool().cpu_bytes_used() / (1024.0 * 1024.0)},
            {"cpu_budget_mb", ctx_server.get_kv_pool().cpu_bytes_budget() / (1024.0 * 1024.0)},
        });
        return res;
    };

    // POST /api/kv_pool/save - explicitly save current slot state to pool
    // Body: {"slot_id": 0, "session_id": "abc123"}
    this->post_kv_pool_save = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "invalid JSON"}});
            return res;
        }

        int id_slot = json_value(body, "slot_id", 0);
        std::string session_id = json_value(body, "session_id", std::string("default"));

        if (!ctx_server.get_kv_pool().is_enabled()) {
            res->status = 503;
            res->data = safe_json_to_str(json{{"error", "kv_pool not enabled (use --cache-ram)"}});
            return res;
        }

        // Find the slot
        bool found = false;
        for (const auto & slot : ctx_server.get_slots()) {
            if (slot.id == id_slot % (int)ctx_server.get_slots().size()) {
                if (slot.prompt.n_tokens() == 0) {
                    res->status = 400;
                    res->data = safe_json_to_str(json{{"error", "slot has no cached state"}});
                    return res;
                }

                auto tokens_vec = slot.prompt.tokens.get_tokens();
                uint64_t hash = kv_pool::hash_tokens(tokens_vec.data(), tokens_vec.size());
                std::string key = kv_pool::make_key(session_id, hash);

                const size_t state_size = llama_state_seq_get_size_ext(ctx_server.get_ctx(), slot.id, 0);
                if (state_size == 0) {
                    res->status = 500;
                    res->data = safe_json_to_str(json{{"error", "failed to get state size"}});
                    return res;
                }

                std::vector<uint8_t> state_data(state_size);
                llama_state_seq_get_data_ext(ctx_server.get_ctx(), state_data.data(), state_size, slot.id, 0);

                bool ok = const_cast<server_context_impl &>(ctx_server).get_kv_pool().save(
                    key, state_data.data(), state_size, (int32_t)tokens_vec.size());

                res->status = ok ? 200 : 500;
                res->data = safe_json_to_str(json{
                    {"saved", ok},
                    {"key", key},
                    {"size_bytes", (int64_t)state_size},
                    {"n_tokens", (int)tokens_vec.size()},
                });
                found = true;
                break;
            }
        }

        if (!found) {
            res->status = 404;
            res->data = safe_json_to_str(json{{"error", "slot not found"}});
        }
        return res;
    };

    // POST /api/kv_pool/load - restore session state from pool to a slot
    // Body: {"slot_id": 0, "session_id": "abc123", "prompt_hash": "..."}
    this->post_kv_pool_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->status = 400;
            res->data = safe_json_to_str(json{{"error", "invalid JSON"}});
            return res;
        }

        std::string key = json_value(body, "key", std::string(""));
        if (key.empty()) {
            std::string session_id = json_value(body, "session_id", std::string("default"));
            std::string hash_str = json_value(body, "prompt_hash", std::string(""));
            if (hash_str.empty()) {
                res->status = 400;
                res->data = safe_json_to_str(json{{"error", "provide 'key' or 'session_id'+'prompt_hash'"}});
                return res;
            }
            key = session_id + ":" + hash_str;
        }

        if (!ctx_server.get_kv_pool().is_enabled()) {
            res->status = 503;
            res->data = safe_json_to_str(json{{"error", "kv_pool not enabled"}});
            return res;
        }

        std::vector<uint8_t> state_data;
        int32_t n_tokens = 0;
        bool found = const_cast<server_context_impl &>(ctx_server).get_kv_pool().load(key, state_data, n_tokens);

        if (!found) {
            res->status = 404;
            res->data = safe_json_to_str(json{{"error", "key not found in pool"}, {"key", key}});
            return res;
        }

        int id_slot = json_value(body, "slot_id", 0);
        id_slot = id_slot % (int)ctx_server.get_slots().size();

        // Restore state into the slot's KV cache
        const size_t n = llama_state_seq_set_data_ext(
            ctx_server.get_ctx(), state_data.data(), state_data.size(), id_slot, 0);

        res->status = (n == state_data.size()) ? 200 : 500;
        res->data = safe_json_to_str(json{
            {"restored", n == state_data.size()},
            {"key", key},
            {"size_bytes", (int64_t)state_data.size()},
            {"n_tokens", n_tokens},
        });
        return res;
    };

    // Batch Inference API

    // POST /api/batch - multi-prompt batch inference
    // Body: {"prompts": ["prompt1", "prompt2", ...], "max_tokens": 200, ...}
    // Or: {"messages_batch": [[{role,content},...], [{role,content},...]], ...}
    // Returns: {"results": [{"text": "...", "tokens": N}, ...]}
    this->post_batch_completions = [this](const server_http_req & req) {
        auto res = create_response();
        auto & rd = res->rd;

        json body;
        try { body = json::parse(req.body); } catch (...) {
            res->error(format_error_response("invalid JSON", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        // Collect prompts from either "prompts" array or "messages_batch" array
        std::vector<std::string> raw_prompts;

        if (body.contains("prompts") && body["prompts"].is_array()) {
            for (auto & p : body["prompts"]) {
                raw_prompts.push_back(p.get<std::string>());
            }
        } else if (body.contains("messages_batch") && body["messages_batch"].is_array()) {
            // Each element is a messages array - apply chat template
            for (auto & msgs : body["messages_batch"]) {
                json chat_body = body;  // inherit params
                chat_body["messages"] = msgs;
                chat_body.erase("messages_batch");
                chat_body.erase("prompts");
                std::vector<raw_buffer> files;
                try {
                    json formatted = oaicompat_chat_params_parse(chat_body, meta->chat_params, files);
                    if (formatted.contains("prompt")) {
                        raw_prompts.push_back(formatted["prompt"].get<std::string>());
                    }
                } catch (const std::exception & e) {
                    res->error(format_error_response(
                        std::string("template error: ") + e.what(), ERROR_TYPE_INVALID_REQUEST));
                    return res;
                }
            }
        } else {
            res->error(format_error_response(
                "provide 'prompts' (string array) or 'messages_batch' (array of message arrays)",
                ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        if (raw_prompts.empty()) {
            res->error(format_error_response("empty batch", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        size_t n_batch = raw_prompts.size();
        SRV_INF("batch: processing %zu prompts\n", n_batch);

        // Tokenize all prompts
        std::vector<server_tokens> tokenized;
        for (auto & p : raw_prompts) {
            auto inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, p, true, true);
            if (!inputs.empty()) {
                tokenized.push_back(std::move(inputs[0]));
            }
        }

        // Create tasks for each prompt
        std::vector<int> task_ids;
        for (size_t i = 0; i < tokenized.size(); i++) {
            server_task task(SERVER_TASK_TYPE_COMPLETION);
            task.id = rd.get_new_id();
            task.tokens = std::move(tokenized[i]);
            task.params = server_task::params_from_json_cmpl(
                ctx_server.vocab, params, meta->slot_n_ctx, meta->logit_bias_eog, body);
            task.params.res_type = TASK_RESPONSE_TYPE_NONE;
            task_ids.push_back(task.id);
            rd.post_task(std::move(task));
        }

        // Collect results (blocking, waits for all)
        json results = json::array();
        size_t completed = 0;

        while (completed < task_ids.size()) {
            auto result = rd.next(req.should_stop);
            if (!result) break;

            auto result_json = result->to_json();
            results.push_back(json{
                {"index", completed},
                {"text", result_json.value("content", "")},
                {"tokens_predicted", result_json.value("tokens_predicted", 0)},
                {"tokens_evaluated", result_json.value("tokens_evaluated", 0)},
                {"stop_type", result_json.value("stop_type", "")},
            });
            completed++;
        }

        res->ok(json{
            {"results", results},
            {"n_prompts", (int)n_batch},
            {"n_completed", (int)completed},
        });
        return res;
    };
}
