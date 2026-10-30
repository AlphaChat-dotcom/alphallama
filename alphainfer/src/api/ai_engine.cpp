// SPDX-License-Identifier: Apache-2.0
// AlphaInfer C API implementation. See api/ai_engine.h for what is and is not implemented and why.

#include "api/ai_engine.h"
#ifdef AI_Q4X_FORWARD
#include "api/q4x_serve.h"
#endif

#include "engine/draft_vocab.h"
#include "engine/engine.h"
#include "engine/mtp_policy.h"
#include "engine/tokenizer.h"
#include "ops/ops.h"
#include "core/tensor.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ai_engine_s {
    ai::Engine  engine;
    ai::Timing  timing;          // accumulates across calls, so a server can surface decode stats
    std::string last_error;
    int         n_streams    = 1;
    int         mtp_k        = 0;      // after the policy's clamp (exact verify: K <= 3 on the default build)
    bool        host_embd    = false;
    bool        mtp_exact    = false;
    std::vector<uint8_t> vocab_cls;    // MTP: per-token script classes for the draft vocabulary (empty = off)
    // One-stream DFlash2 (AI_DFLASH2): prefill_chunk buffers the prompt here and prefills it whole at is_last
    bool                 df2 = false;
    std::vector<int32_t> df2_pf;
    bool                 df2_pf_open = false;
    // Per-request drafter routing with DFlash2 loaded (AI_DF2_ROUTE=1, as the CLI): Latin + English prompts on DFlash2,
    // everything else on the MTP head. Needs the tokenizer (English function-word share) and the script classes.
    bool                           df2_route = false;
    std::unique_ptr<ai::Tokenizer> route_tok;
    std::vector<uint8_t>           route_cls;
#ifdef AI_Q4X_FORWARD
    // qwen4exp (Flash-Next) models run on their own runtime (api/q4x_serve.h); `engine` stays unused then
    std::unique_ptr<ai::Q4xServe> q4x;
#endif
};

namespace {

// Static so the pointer handed out through *err stays valid after we return.
const char * set_static(const char * msg) {
    return msg;
}

}  // namespace

extern "C" {

ai_engine * ai_engine_create(const ai_engine_params * p, const char ** err) {
    auto fail = [&](const char * m) -> ai_engine * {
        if (err) { *err = set_static(m); }
        return nullptr;
    };
    if (!p || !p->model_path) {
        return fail("ai_engine_create: params or model_path is null");
    }
    if (p->n_streams < 1 || p->n_streams > 2) {
        return fail("ai_engine_create: this increment supports n_streams 1 or 2");
    }
#ifdef AI_Q4X_FORWARD
    if (ai::gguf_arch(p->model_path) == "qwen4exp") {
        if (p->n_streams != 1) return fail("ai_engine_create: qwen4exp serves one stream (n_streams must be 1)");
        if (p->host_embd) return fail("ai_engine_create: host_embd is not available on qwen4exp");
        ai_engine * h = new (std::nothrow) ai_engine_s();
        if (!h) return fail("ai_engine_create: out of memory");
        ai::Q4xServe::Config c;
        c.model_path   = p->model_path;
        c.ctx          = p->ctx_cells_per_stream > 0 ? p->ctx_cells_per_stream : p->ctx_cells_per_stream < 0 ? 32768 : 8192;
        c.expert_slots = p->expert_slots;
        c.mtp_k        = p->mtp_k;
        c.ws_mib       = p->ws_mib;
        c.verbose      = p->verbose != 0;
        try {
            h->q4x = std::make_unique<ai::Q4xServe>(c);
        } catch (const std::exception & ex) {
            fprintf(stderr, "[ai-api] ai_engine_create (qwen4exp): %s\n", ex.what());
            delete h;
            return fail("ai_engine_create: qwen4exp engine init failed (see stderr for the cause)");
        }
        h->n_streams = 1;
        h->mtp_k     = h->q4x->mtp_k();
        h->mtp_exact = false;
        if (err) { *err = ""; }
        return h;
    }
#endif
    // host_embd gathers token_embd on the host, and the MTP draft head embeds device-produced ids, so a
    // host gather would need a mid-round sync that breaks the round's capture. The engine has its own
    // guard, but it is reachable only with a model that has a next-n block; the check is duplicated here
    // so the API refuses the combination regardless.
    if (p->host_embd && p->mtp_k > 0) {
        return fail("ai_engine_create: host_embd is incompatible with mtp_k > 0 (the draft head embeds "
                    "device-produced ids; a host gather needs a mid-round sync that breaks capture)");
    }
    // mtp_k > 0 is served by ai_engine_decode_mtp (an explicit output capacity and stream ids); ai_engine_decode
    // refuses it, because its signature cannot bound several tokens per stream.

    // AI_DFLASH2=<drafter.gguf>: DFlash2 drafts instead of the MTP head (the CLI's --dflash2). Selected by env, not a
    // params field, because the params struct must stay field-for-field with AlphaServe's copy of this header. The
    // engine holds one drafter context, so DFlash2 needs n_streams == 1; its verify is always the fast one (the policy
    // never claims exact for a block-size verify), so it is a best-speed mode, never the exact default.
    const char * df2_path = getenv("AI_DFLASH2");
    const bool   want_df2 = df2_path != nullptr && df2_path[0] != '\0';
    if (want_df2 && p->n_streams != 1) {
        return fail("ai_engine_create: AI_DFLASH2 needs n_streams == 1 (the engine holds one drafter context)");
    }
    if (want_df2 && p->expert_slots != 0) {
        return fail("ai_engine_create: AI_DFLASH2 needs resident experts (expert_slots == 0)");
    }
    if (want_df2 && p->host_embd) {
        return fail("ai_engine_create: AI_DFLASH2 is incompatible with host_embd (device-produced draft ids)");
    }

    // AI_HOST_EMBD is read by the weight loader's host filter at model load, so it must be set before
    // init(). Set, never cleared: a later engine in the same process would otherwise silently differ.
    if (p->host_embd) {
#ifdef _WIN32
        _putenv_s("AI_HOST_EMBD", "1");
#else
        setenv("AI_HOST_EMBD", "1", 1);
#endif
    }

    ai_engine * h = new (std::nothrow) ai_engine_s();
    if (!h) {
        return fail("ai_engine_create: out of memory");
    }
    h->n_streams = p->n_streams;
    h->mtp_k     = p->mtp_k;
    h->host_embd = p->host_embd != 0;

    ai::EngineConfig cfg;
    cfg.model_path   = p->model_path;
    cfg.max_ctx      = p->ctx_cells_per_stream > 0 ? p->ctx_cells_per_stream
                     : p->ctx_cells_per_stream < 0 ? 0 /* auto */ : 8192;
    cfg.max_batch    = p->n_streams;
    cfg.expert_slots      = p->expert_slots < 0 ? 1 : p->expert_slots;   // -1 = auto (doorbell offload)
    cfg.expert_slots_auto = p->expert_slots < 0;
    cfg.mtp_k        = p->mtp_k > 0 ? p->mtp_k : 0;
    if (want_df2) {
        // as the CLI: DFlash2 takes MTP's verify configuration; the engine sets k from the drafter's block size
        cfg.dflash2_path = df2_path;
        cfg.mtp_k        = 7;
        // With AI_DF2_ROUTE, hold the MTP chain at 5 so its graph profiles never share a key with DFlash2's 7-draft block
        if (getenv("AI_DF2_ROUTE") && atoi(getenv("AI_DF2_ROUTE")) != 0) setenv("AI_MTP_CHAIN", "5", 0);
    }
    cfg.kv_type      = p->kv_type_is_q4_0 < 0 ? ai::kKvAuto
                     : p->kv_type_is_q4_0 == 1 ? ai::Q4_0 : p->kv_type_is_q4_0 == 2 ? ai::Q8_0 : ai::F16;
    cfg.verbose      = p->verbose != 0;
    // Prefill chunk 1024 (the engine default is 512; AI_CHUNK overrides): about 13% less prefill time than 512 for a
    // few MiB more workspace. Chunk 2048 roughly doubles the workspace. A longer chunk holds the other slots' decode
    // for its duration.
    cfg.chunk        = 1024;
    // ws_mib 0 = auto: the engine sizes the workspace from the measured peak when ws_bytes == 0.
    cfg.ws_bytes     = p->ws_mib > 0 ? ((size_t) p->ws_mib << 20) : 0;

    // The same verify configuration as the CLI (engine/mtp_policy.cpp): exact on MoE by default, the verify pins,
    // the K clamp and the confidence-gate default. AI_MTP_MODE=fast|exact overrides, since the params struct must
    // stay field-for-field with AlphaServe's copy of this header. Called for plain engines too, as the CLI does.
    h->mtp_exact = ai::apply_mtp_policy(cfg, ai::mtp_mode_from_env());
    h->mtp_k     = cfg.mtp_k;
    bool ok = false;
    try {
        ok = h->engine.init(cfg);
    } catch (const std::exception & ex) {
        h->last_error = std::string("ai_engine_create: init threw: ") + ex.what();
        fprintf(stderr, "[ai-api] %s\n", h->last_error.c_str());
        delete h;
        return fail("ai_engine_create: engine init threw (see stderr for the cause)");
    }
    if (!ok) {
        delete h;
        return fail("ai_engine_create: engine init returned false (see stderr for the cause)");
    }
    h->mtp_k = h->engine.mtp_k();   // DFlash2: the drafter's block size - 1, not the policy's request
    h->df2   = h->engine.dflash2();
    if (h->df2) {
        fprintf(stderr, "[ai-api] DFlash2 drafter %s: %d drafts per round, 1 stream, %s verify (NOT token-identical)\n",
                df2_path, h->mtp_k, h->mtp_exact ? "EXACT" : "FAST");
    }
    // Script-adaptive draft rows (AI_DRAFT_VOCAB, default on), as the CLI picks them per prompt: classify the vocab
    // once here, choose the rows at each prefill. Raises MTP acceptance substantially on non-Latin scripts.
    if (h->mtp_k > 0 && h->engine.draft_vocab_rows() > 0 &&
        (!getenv("AI_DRAFT_VOCAB") || atoi(getenv("AI_DRAFT_VOCAB")) != 0)) {
        ai::Tokenizer tok;
        if (tok.load(p->model_path)) {
            h->vocab_cls = ai::classify_vocab(tok, tok.n_vocab());
        } else {
            fprintf(stderr, "[ai-api] draft vocabulary: tokenizer load failed; MTP keeps the default draft rows\n");
        }
    }
    if (h->df2 && getenv("AI_DF2_ROUTE") && atoi(getenv("AI_DF2_ROUTE")) != 0) {
        h->route_tok = std::make_unique<ai::Tokenizer>();
        if (h->route_tok->load(p->model_path)) {
            h->route_cls = ai::classify_vocab(*h->route_tok, h->route_tok->n_vocab());
            h->df2_route = true;
            fprintf(stderr, "[ai-api] df2 route ON: Latin + English prompts -> DFlash2, others -> MTP head\n");
        } else {
            h->route_tok.reset();
            fprintf(stderr, "[ai-api] df2 route: tokenizer load failed; every request stays on DFlash2\n");
        }
    }
    if (err) { *err = ""; }
    return h;
}

// AI_DF2_ROUTE: choose this request's drafter from its whole prompt, before its prefill (the CLI's rule)
static void df2_route(ai_engine * e, const std::vector<int32_t> & prompt) {
    if (!e->df2_route) return;
    const bool          latin = ai::prompt_is_latin(e->route_cls, prompt);
    const double        en    = ai::prompt_english_share(*e->route_tok, prompt);
    static const double thr   = getenv("AI_DF2_ROUTE_EN") ? atof(getenv("AI_DF2_ROUTE_EN")) : 0.08;
    // AI_DF2_ROUTE_EDIT=1 (default off): send code-edit requests to the MTP head. Helps some edit kinds and hurts
    // others, a net loss overall, so it is opt-in.
    static const bool   edit_on = getenv("AI_DF2_ROUTE_EDIT") && atoi(getenv("AI_DF2_ROUTE_EDIT")) != 0;
    const bool          df2     = latin && en >= thr;
    e->engine.set_df2_active(df2 && !(edit_on && ai::prompt_is_code_edit(*e->route_tok, prompt)));
}

void ai_engine_destroy(ai_engine * e) {
    delete e;
}

ai_capture_state ai_engine_capture_state(const ai_engine * e) {
    (void) e;
    // Read from the adapter's own record of what happened, not inferred from replay counters: a
    // process with no rounds yet has zero of both, and inference cannot tell that apart from a
    // failure. op::capture_degraded() is set at src/ops/adapter.cpp where capture_ok_ is cleared.
    if (ai::op::capture_degraded()) {
        return AI_CAPTURE_DEGRADED;
    }
    switch (ai::op::capture_mode()) {
        case ai::op::Capture::Own:     return AI_CAPTURE_OWN;
        case ai::op::Capture::Backend: return AI_CAPTURE_OWN;   // still captured, different owner
        case ai::op::Capture::None:    return AI_CAPTURE_NONE;
    }
    return AI_CAPTURE_NONE;
}

int64_t ai_engine_free_vram_mib(const ai_engine * e) {
    if (!e) { return -1; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) { return e->q4x->free_vram_mib(); }
#endif
    size_t free_b = 0, total_b = 0;
    const_cast<ai_engine_s *>(e)->engine.device().memory_info(&free_b, &total_b);
    return (int64_t) (free_b >> 20);
}

int64_t ai_engine_ctx_cells(const ai_engine * e) {
    if (!e) { return -1; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) { return e->q4x->ctx_cells(); }
#endif
    return e->engine.ctx_cells();
}

int ai_engine_calibration_state(const ai_engine * e) {
    if (!e) { return -1; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) { return 0; }   // a fixed workspace: nothing to calibrate
#endif
    return e->engine.calibration_state();
}

int ai_engine_has_mtp(const ai_engine * e) {
    if (!e) { return 0; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) { return e->q4x->has_mtp() ? 1 : 0; }
#endif
    return const_cast<ai_engine_s *>(e)->engine.program().has_mtp() ? 1 : 0;
}

int ai_engine_supports_sampling(const ai_engine * e) {
    if (!e) { return 0; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) { return 0; }   // Flash-Next: greedy only for now
#endif
    return e->engine.sampling_enabled() ? 1 : 0;
}

int ai_engine_set_sampling(ai_engine * e, int stream_id, float temperature, int top_k, float top_p, float min_p,
                           uint64_t seed) {
    if (!e) { return AI_ENGINE_EINVAL; }
    if (!ai_engine_supports_sampling(e)) {
        if (!(temperature > 0.0f)) { return AI_ENGINE_OK; }   // greedy is what it does anyway
        e->last_error = "ai_engine_set_sampling: this engine is greedy only";
        return AI_ENGINE_ENOTSUP;
    }
    if (stream_id < 0 || stream_id >= e->n_streams) {
        e->last_error = "ai_engine_set_sampling: stream_id out of range";
        return AI_ENGINE_EINVAL;
    }
    return e->engine.set_sampling(stream_id, temperature, top_k, top_p, min_p, seed) ? AI_ENGINE_OK : AI_ENGINE_EFAIL;
}

int ai_engine_prefill(ai_engine * e, int stream_id,
                      const int32_t * tokens, int n_tokens, int32_t * out_first) {
    if (out_first) { *out_first = 0; }
    if (!e) { return AI_ENGINE_EINVAL; }
    if (!tokens || !out_first || n_tokens < 1) {
        e->last_error = "ai_engine_prefill: tokens and out_first must be non-null and n_tokens >= 1";
        return AI_ENGINE_EINVAL;
    }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {
        if (stream_id != 0) {
            e->last_error = "ai_engine_prefill: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        try {
            *out_first = e->q4x->prefill(tokens, n_tokens);
            return AI_ENGINE_OK;
        } catch (const std::exception & ex) {
            e->last_error = std::string("ai_engine_prefill: ") + ex.what();
            return AI_ENGINE_EFAIL;
        }
    }
#endif
    try {
        // First prefill opens the batch. Doing it here rather than in create() keeps create() free of
        // device work that would have to be undone if the caller never decodes.
        if (e->engine.stream_count() != e->n_streams) {
            e->engine.open_streams(e->n_streams);
        }
        if (stream_id < 0 || stream_id >= e->engine.stream_count()) {
            e->last_error = "ai_engine_prefill: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        const std::vector<int32_t> prompt(tokens, tokens + n_tokens);
        e->df2_pf.clear();   // DFlash2: a whole prefill abandons any buffered partial prompt
        e->df2_pf_open = false;
        df2_route(e, prompt);
        if (!e->vocab_cls.empty()) {
            // The draft head is shared by every slot, so its rows follow the most recent prefill. That moves only
            // the other slots' acceptance, never what the verify accepts (the verify keeps the full head).
            std::string why;
            const std::vector<int32_t> rows = ai::draft_rows_for(e->vocab_cls, prompt, e->engine.draft_vocab_rows(), &why);
            e->engine.set_draft_vocab(rows);
        }
        *out_first = e->engine.prefill_one(stream_id, prompt, e->timing);
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_prefill: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

int ai_engine_n_embd(const ai_engine * e) {
    if (!e) { return 0; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) { return (int) e->q4x->n_embd(); }
#endif
    return (int) e->engine.program().spec().n_embd;
}

int ai_engine_prefill_mm(ai_engine * e, int stream_id, const int32_t * tokens, int n_tokens, const int32_t * pos4,
                         const float * embd, const int32_t * embd_row, int n_rows, int32_t * out_first) {
    if (out_first) { *out_first = 0; }
    if (!e) { return AI_ENGINE_EINVAL; }
    if (!tokens || !out_first || !pos4 || n_tokens < 1 || n_rows < 0 || (n_rows > 0 && (!embd || !embd_row))) {
        e->last_error = "ai_engine_prefill_mm: tokens, pos4 and out_first must be non-null, n_tokens >= 1, and embd / "
                        "embd_row non-null when n_rows > 0";
        return AI_ENGINE_EINVAL;
    }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {
        e->last_error = "ai_engine_prefill_mm: image input is not available on qwen4exp";
        return AI_ENGINE_ENOTSUP;
    }
#endif
    if (e->engine.dflash2()) {
        e->last_error = "ai_engine_prefill_mm: image input is not available with DFlash2";
        return AI_ENGINE_ENOTSUP;
    }
    try {
        if (e->engine.stream_count() != e->n_streams) {
            e->engine.open_streams(e->n_streams);
        }
        if (stream_id < 0 || stream_id >= e->engine.stream_count()) {
            e->last_error = "ai_engine_prefill_mm: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        const std::vector<int32_t> prompt(tokens, tokens + n_tokens);
        if (!e->vocab_cls.empty()) {
            std::string why;
            const std::vector<int32_t> rows = ai::draft_rows_for(e->vocab_cls, prompt, e->engine.draft_vocab_rows(), &why);
            e->engine.set_draft_vocab(rows);
        }
        *out_first = e->engine.prefill_mm(stream_id, prompt, pos4, embd, embd_row, n_rows, e->timing);
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_prefill_mm: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

int ai_engine_prefill_chunk(ai_engine * e, int stream_id, const int32_t * tokens, int n_tokens, int is_last,
                            int32_t * out_first) {
    if (out_first) { *out_first = -1; }
    if (!e) { return AI_ENGINE_EINVAL; }
    if (!out_first || n_tokens < 0 || (n_tokens > 0 && !tokens)) {
        e->last_error = "ai_engine_prefill_chunk: out_first must be non-null, n_tokens >= 0, tokens non-null when n_tokens > 0";
        return AI_ENGINE_EINVAL;
    }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {   // one stream, nothing to interleave: buffered, then prefilled whole at is_last
        if (stream_id != 0) {
            e->last_error = "ai_engine_prefill_chunk: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        try {
            *out_first = e->q4x->prefill_feed(tokens, n_tokens, is_last != 0);
            return AI_ENGINE_OK;
        } catch (const std::exception & ex) {
            e->last_error = std::string("ai_engine_prefill_chunk: ") + ex.what();
            return AI_ENGINE_EFAIL;
        }
    }
#endif
    try {
        if (e->engine.stream_count() != e->n_streams) {
            e->engine.open_streams(e->n_streams);
        }
        if (stream_id < 0 || stream_id >= e->engine.stream_count()) {
            e->last_error = "ai_engine_prefill_chunk: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        if (e->df2) {
            // The engine's chunked feed has no DFlash2 path (the drafter context fills per prefill chunk),
            // and with one stream there is no other request to interleave: buffer, then prefill whole --
            // identical to ai_engine_prefill of the concatenation, so a chunking server needs no special case
            if (!e->df2_pf_open) {
                e->df2_pf.clear();
                e->df2_pf_open = true;
            }
            e->df2_pf.insert(e->df2_pf.end(), tokens, tokens + n_tokens);
            if (!is_last) {
                return AI_ENGINE_OK;
            }
            std::vector<int32_t> prompt;
            prompt.swap(e->df2_pf);
            e->df2_pf_open = false;
            if (prompt.empty()) {
                e->last_error = "ai_engine_prefill_chunk: empty prompt";
                return AI_ENGINE_EINVAL;
            }
            if (!e->vocab_cls.empty()) {
                std::string why;
                e->engine.set_draft_vocab(ai::draft_rows_for(e->vocab_cls, prompt, e->engine.draft_vocab_rows(), &why));
            }
            df2_route(e, prompt);
            *out_first = e->engine.prefill_one(stream_id, prompt, e->timing);
            return AI_ENGINE_OK;
        }
        if (is_last && !e->vocab_cls.empty()) {
            // the draft rows from the whole prompt, as ai_engine_prefill chooses them, before the chunk that drafts
            std::vector<int32_t> prompt = e->engine.prefill_buffer(stream_id);
            prompt.insert(prompt.end(), tokens, tokens + n_tokens);
            std::string why;
            const std::vector<int32_t> rows = ai::draft_rows_for(e->vocab_cls, prompt, e->engine.draft_vocab_rows(), &why);
            e->engine.set_draft_vocab(rows);
        }
        *out_first = e->engine.prefill_feed(stream_id, tokens, n_tokens, is_last != 0, e->timing);
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_prefill_chunk: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

int ai_engine_decode(ai_engine * e,
                     const int32_t * tokens, const int32_t * positions,
                     const int32_t * seq_ids, int n_tokens,
                     const uint8_t * logits_mask,
                     int32_t * out_sampled, int * out_n_sampled) {
    if (out_n_sampled) { *out_n_sampled = 0; }
    if (!e) { return AI_ENGINE_EINVAL; }
    if (!tokens || !positions || !seq_ids || !out_sampled || !out_n_sampled || n_tokens < 1) {
        e->last_error = "ai_engine_decode: tokens/positions/seq_ids/out_sampled/out_n_sampled must be "
                        "non-null and n_tokens >= 1";
        return AI_ENGINE_EINVAL;
    }
    if (logits_mask) {
        // Refused, not ignored. A silently dropped sampling constraint returns plausible wrong text,
        // which is the exact failure class this API exists to prevent.
        e->last_error = "ai_engine_decode: logits_mask is not supported; ignoring it would return "
                        "plausible wrong text, so it is refused rather than dropped";
        return AI_ENGINE_ENOTSUP;
    }
    if (e->mtp_k > 0) {
        e->last_error = "ai_engine_decode: this engine runs MTP (mtp_k > 0) -- use ai_engine_decode_mtp, which takes "
                        "an output capacity and returns a stream id per token; this signature cannot bound several "
                        "tokens per stream per round";
        return AI_ENGINE_ENOTSUP;
    }
    // out_sampled carries no capacity. One token per stream per round is the bound while mtp_k is
    // refused, so n_streams is the contract stated in the header.
    const int cap = e->n_streams;
    if (n_tokens > cap) {
        e->last_error = "ai_engine_decode: n_tokens exceeds n_streams; a plain round carries exactly "
                        "one token per stream";
        return AI_ENGINE_EINVAL;
    }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {
        if (seq_ids[0] != 0) {
            e->last_error = "ai_engine_decode: seq_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        if (!e->q4x->prefilled()) {
            e->last_error = "ai_engine_decode: no prefill on this stream -- ai_engine_prefill must run first";
            return AI_ENGINE_EINVAL;
        }
        if ((int64_t) positions[0] != e->q4x->position()) {
            char m[192];
            snprintf(m, sizeof m, "ai_engine_decode: position mismatch on stream 0 -- engine has %lld, caller sent %lld",
                     (long long) e->q4x->position(), (long long) positions[0]);
            e->last_error = m;
            return AI_ENGINE_EINVAL;
        }
        try {
            out_sampled[0] = e->q4x->decode_plain(tokens[0]);
            *out_n_sampled = 1;
            return AI_ENGINE_OK;
        } catch (const std::exception & ex) {
            e->last_error = std::string("ai_engine_decode: ") + ex.what();
            return AI_ENGINE_EFAIL;
        }
    }
#endif
    try {
        if (e->engine.stream_count() != e->n_streams) {
            e->last_error = "ai_engine_decode: no open batch -- ai_engine_prefill must run first";
            return AI_ENGINE_EINVAL;
        }
        std::vector<ai::Engine::DecodeIn> in;
        in.reserve((size_t) n_tokens);
        int prev = -1;
        for (int i = 0; i < n_tokens; ++i) {
            const int sid = (int) seq_ids[i];
            if (sid < 0 || sid >= e->engine.stream_count()) {
                e->last_error = "ai_engine_decode: seq_id outside 0..n_streams-1";
                return AI_ENGINE_EINVAL;
            }
            // out_sampled has no companion stream-id array, so attribution rests on order. The
            // engine emits in ascending stream order, so requiring ascending seq_ids is what makes
            // out_sampled[i] unambiguously the i-th named stream. Without this a caller passing
            // {1,0} would read stream 0's token as stream 1's and never see an error.
            if (sid <= prev) {
                e->last_error = "ai_engine_decode: seq_ids must be strictly ascending -- out_sampled "
                                "carries no stream ids, so attribution depends on emission order";
                return AI_ENGINE_EINVAL;
            }
            prev = sid;
            // Checked here as well as inside decode_step, so a position mismatch surfaces as EINVAL
            // rather than EFAIL. decode_step keeps its own assert because the engine is the authority
            // on position; the duplication is deliberate.
            const int64_t have = e->engine.stream_position(sid);
            if ((int64_t) positions[i] != have) {
                char m[192];
                snprintf(m, sizeof m,
                         "ai_engine_decode: position mismatch on stream %d -- engine has %lld, "
                         "caller sent %lld", sid, (long long) have, (long long) positions[i]);
                e->last_error = m;
                return AI_ENGINE_EINVAL;
            }
            in.push_back(ai::Engine::DecodeIn{ sid, (int64_t) positions[i], tokens[i] });
        }
        std::vector<std::pair<int, int32_t>> out;
        e->engine.decode_step(in, out, e->timing);
        if ((int) out.size() > cap) {
            e->last_error = "ai_engine_decode: the round emitted more tokens than out_sampled is "
                            "contracted to hold and this signature has no capacity argument; "
                            "nothing was written";
            return AI_ENGINE_EFAIL;
        }
        for (size_t k = 0; k < out.size(); ++k) {
            out_sampled[k] = out[k].second;
        }
        *out_n_sampled = (int) out.size();
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_decode: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

int ai_engine_max_tokens_per_round(const ai_engine * e) {
    if (!e) { return 0; }
    return e->n_streams * (e->mtp_k + 1);
}

int ai_engine_mtp_info(const ai_engine * e, int * out_mtp_k, int * out_exact) {
    if (!e) { return AI_ENGINE_EINVAL; }
    if (out_mtp_k) { *out_mtp_k = e->mtp_k; }
    if (out_exact) { *out_exact = e->mtp_exact ? 1 : 0; }
    return AI_ENGINE_OK;
}

int ai_engine_decode_mtp(ai_engine * e,
                         const int32_t * tokens, const int32_t * positions,
                         const int32_t * seq_ids, int n_tokens,
                         int32_t * out_tokens, int32_t * out_seq_ids, int out_capacity, int * out_n) {
    if (out_n) { *out_n = 0; }
    if (!e) { return AI_ENGINE_EINVAL; }
    if (!tokens || !positions || !seq_ids || !out_tokens || !out_seq_ids || !out_n || n_tokens < 1) {
        e->last_error = "ai_engine_decode_mtp: tokens/positions/seq_ids/out_tokens/out_seq_ids/out_n must be "
                        "non-null and n_tokens >= 1";
        return AI_ENGINE_EINVAL;
    }
    // The bound is checked before the round runs: a round that has run has advanced the engine, so discovering
    // afterwards that its tokens do not fit would leave the caller unable to recover them.
    const int need = ai_engine_max_tokens_per_round(e);
    if (out_capacity < need) {
        char m[192];
        snprintf(m, sizeof m, "ai_engine_decode_mtp: out_capacity %d < the round's bound %d (n_streams x "
                              "(mtp_k + 1)); size it with ai_engine_max_tokens_per_round", out_capacity, need);
        e->last_error = m;
        return AI_ENGINE_EINVAL;
    }
    if (n_tokens > e->n_streams) {
        e->last_error = "ai_engine_decode_mtp: n_tokens exceeds n_streams; pass one pending token per stream";
        return AI_ENGINE_EINVAL;
    }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {
        if (seq_ids[0] != 0) {
            e->last_error = "ai_engine_decode_mtp: seq_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        if (!e->q4x->prefilled()) {
            e->last_error = "ai_engine_decode_mtp: no prefill on this stream -- ai_engine_prefill must run first";
            return AI_ENGINE_EINVAL;
        }
        if ((int64_t) positions[0] != e->q4x->position()) {
            char m[192];
            snprintf(m, sizeof m, "ai_engine_decode_mtp: position mismatch on stream 0 -- engine has %lld, caller sent %lld",
                     (long long) e->q4x->position(), (long long) positions[0]);
            e->last_error = m;
            return AI_ENGINE_EINVAL;
        }
        try {
            std::vector<int32_t> out;
            if (e->q4x->has_mtp()) e->q4x->decode_spec(tokens[0], out);
            else out.push_back(e->q4x->decode_plain(tokens[0]));
            if ((int) out.size() > out_capacity) {   // cannot happen if the bound above holds; checked, not assumed
                e->last_error = "ai_engine_decode_mtp: the round emitted more tokens than its bound; nothing written";
                return AI_ENGINE_EFAIL;
            }
            for (size_t k = 0; k < out.size(); ++k) {
                out_seq_ids[k] = 0;
                out_tokens[k]  = out[k];
            }
            *out_n = (int) out.size();
            return AI_ENGINE_OK;
        } catch (const std::exception & ex) {
            e->last_error = std::string("ai_engine_decode_mtp: ") + ex.what();
            return AI_ENGINE_EFAIL;
        }
    }
#endif
    try {
        if (e->engine.stream_count() != e->n_streams) {
            e->last_error = "ai_engine_decode_mtp: no open batch -- ai_engine_prefill must run first";
            return AI_ENGINE_EINVAL;
        }
        std::vector<ai::Engine::DecodeIn> in;
        in.reserve((size_t) n_tokens);
        for (int i = 0; i < n_tokens; ++i) {
            const int sid = (int) seq_ids[i];
            if (sid < 0 || sid >= e->engine.stream_count()) {
                e->last_error = "ai_engine_decode_mtp: seq_id outside 0..n_streams-1";
                return AI_ENGINE_EINVAL;
            }
            const int64_t have = e->engine.stream_position(sid);
            if ((int64_t) positions[i] != have) {
                char m[192];
                snprintf(m, sizeof m, "ai_engine_decode_mtp: position mismatch on stream %d -- engine has %lld, "
                                      "caller sent %lld", sid, (long long) have, (long long) positions[i]);
                e->last_error = m;
                return AI_ENGINE_EINVAL;
            }
            in.push_back(ai::Engine::DecodeIn{ sid, (int64_t) positions[i], tokens[i] });
        }
        std::vector<std::pair<int, int32_t>> out;
        e->engine.decode_step(in, out, e->timing);
        if ((int) out.size() > out_capacity) {   // cannot happen if the bound above holds; checked, not assumed
            e->last_error = "ai_engine_decode_mtp: the round emitted more tokens than its bound; nothing written";
            return AI_ENGINE_EFAIL;
        }
        for (size_t k = 0; k < out.size(); ++k) {
            out_seq_ids[k] = (int32_t) out[k].first;
            out_tokens[k]  = out[k].second;
        }
        *out_n = (int) out.size();
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_decode_mtp: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

int ai_engine_warm_profiles(ai_engine * e, int * out_recorded) {
    if (out_recorded) { *out_recorded = 0; }
    if (!e) { return AI_ENGINE_EINVAL; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {
        e->last_error = "ai_engine_warm_profiles: not supported on qwen4exp (its rounds are recorded on first use)";
        return AI_ENGINE_ENOTSUP;
    }
#endif
    if (e->mtp_k > 0) {
        e->last_error = "ai_engine_warm_profiles: not supported with mtp_k > 0 -- an MTP profile key "
                        "depends on the adaptive k schedule, which is not knowable at init";
        return AI_ENGINE_ENOTSUP;
    }
    try {
        if (e->engine.stream_count() != e->n_streams) {
            e->engine.open_streams(e->n_streams);
        }
        const int n = e->engine.warm_profiles(e->timing);
        if (out_recorded) { *out_recorded = n; }
        if (n < 1) {
            // Zero recorded means no graph was captured, which is the silent-degrade case this entry
            // exists to rule out -- so it is an error, not a quiet success.
            e->last_error = "ai_engine_warm_profiles: recorded 0 profiles; capture is not happening";
            return AI_ENGINE_EFAIL;
        }
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_warm_profiles: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

int ai_engine_reset_stream(ai_engine * e, int stream_id) {
    if (!e) { return AI_ENGINE_EINVAL; }
#ifdef AI_Q4X_FORWARD
    if (e->q4x) {
        if (stream_id != 0) {
            e->last_error = "ai_engine_reset_stream: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        try {
            e->q4x->reset();
            return AI_ENGINE_OK;
        } catch (const std::exception & ex) {
            e->last_error = std::string("ai_engine_reset_stream: ") + ex.what();
            return AI_ENGINE_EFAIL;
        }
    }
#endif
    try {
        if (e->engine.stream_count() < 1) {
            e->last_error = "ai_engine_reset_stream: no open batch -- ai_engine_prefill must run first";
            return AI_ENGINE_EINVAL;
        }
        if (stream_id < 0 || stream_id >= e->engine.stream_count()) {
            e->last_error = "ai_engine_reset_stream: stream_id outside 0..n_streams-1";
            return AI_ENGINE_EINVAL;
        }
        // Frees only this slot's KV region and zeroes only its recurrent state. Safe because KV
        // regions are a static partition -- first_cell(seq) = seq * region_ -- so the cells of one
        // slot are disjoint from another's by arithmetic, not by allocator bookkeeping.
        e->engine.reset_one(stream_id);
        e->df2_pf.clear();   // DFlash2: abandons a buffered partial prompt, as it does a chunked one
        e->df2_pf_open = false;
        return AI_ENGINE_OK;
    } catch (const std::exception & ex) {
        e->last_error = std::string("ai_engine_reset_stream: ") + ex.what();
        return AI_ENGINE_EFAIL;
    }
}

const char * ai_engine_last_error(const ai_engine * e) {
    return e ? e->last_error.c_str() : "";
}

}  // extern "C"
