#pragma once

// AlphaServe Inter-Thread Message Types
//
// These structs are passed between threads via SPSC queues.
// They are owned by the receiving thread after pop, with no sharing.
// All vectors are moved, not copied.

#include "llama.h"
#include <memory>
#include <vector>
#include <string>
#include <cstdint>

namespace alphaserve {

struct mm_prompt;   // serve-mm.h: a multimodal prompt (mtmd chunks); null for text-only requests

// Scheduler -> GPU
// Batch of tokens to decode. Built by scheduler, consumed by GPU thread.
struct BatchRequest {
    // Flattened tokens from all active slots
    std::vector<llama_token> tokens;
    std::vector<llama_pos>   positions;
    std::vector<int32_t>     seq_ids;      // which sequence (slot) each token belongs to
    std::vector<int>         slot_ids;      // slot index per token
    std::vector<bool>        logits_mask;   // which positions need logits/sampling

    // Per-token sampling params (only meaningful where logits_mask is true). Default =
    // greedy (temp<=0 -> argmax), so prefill tokens and greedy slots are unchanged. A slot
    // requesting temperature/top_k/top_p is sampled host-side in decode_batch from its logits.
    std::vector<float>    samp_temp;   // temperature (<=0 => greedy argmax)
    std::vector<int32_t>  samp_top_k;  // 0 or >=vocab => no top-k cap; 1 => argmax
    std::vector<float>    samp_top_p;  // >=1 => no nucleus cap
    std::vector<uint64_t> samp_rng;    // per-token RNG seed for this token (see samp_rng_base)

    // ALPHA_MTP: the seed schedule, not just this token's seed. MTP emits several tokens from one
    // batch entry and must reproduce the seed a non-speculative decode would have used for each of
    // them; samp_rng alone cannot be inverted for that (one equation, two unknowns). The schedule is
    //     seed(j) = rng_base ^ (0x9E3779B97F4A7C15 * (n_generated + 1 + j))
    // It is keyed on the slot's generation counter, not the position. See serve-scheduler.cpp where
    // rng is computed.
    std::vector<uint64_t> samp_rng_base;  // slot.rng_base (0 when not sampling)
    std::vector<int32_t>  samp_n_gen;     // slot.n_generated at batch-build time
    std::vector<int32_t>  samp_n_left;    // tokens the slot may still emit (max_tokens - n_generated); caps MTP windows

    // KV cache management
    std::vector<int32_t> clear_seqs;  // sequence IDs to clear before decode

    // ALPHA_HYBRID_PREFIX: recurrent-state checkpoints for prefix reuse on hybrid (DeltaNet/SSM)
    // models. Attention KV can be truncated to any prefix; recurrent state cannot, so a slot may only
    // resume from a position at which its recurrent state was snapshotted. The engine executes these
    // around llama_decode, in queue order, so a directive always sees exactly the state it names:
    std::vector<int32_t> restore_seqs;   // restore the seq's saved recurrent snapshot before decode
    std::vector<int32_t> snapshot_seqs;  // save the seq's recurrent state after this batch decodes

    // Metadata
    int n_tokens = 0;
    int n_slots = 0;               // number of unique slots in this batch
    int prompt_end = 0;            // largest full-prompt length among the prompt chunks in this batch (0 = none)
    // ALPHA_PLE_PREFETCH_AHEAD: the whole prompt, on the first chunk of a multi-chunk prompt: the engine hints the
    // rows its later chunks will gather while this chunk runs on the GPU
    std::vector<llama_token> prompt_ahead;

    // Multimodal prompt prefill: when set, this batch is that one slot's whole prompt (tokens are placeholders) and the
    // engine evaluates the chunks instead of the tokens.
    std::shared_ptr<mm_prompt> mm;
    int mm_slot = -1;

    void clear() {
        tokens.clear();
        positions.clear();
        seq_ids.clear();
        slot_ids.clear();
        logits_mask.clear();
        samp_temp.clear();
        samp_top_k.clear();
        samp_top_p.clear();
        samp_rng.clear();
        samp_rng_base.clear();
        samp_n_gen.clear();
        samp_n_left.clear();
        clear_seqs.clear();
        restore_seqs.clear();
        snapshot_seqs.clear();
        n_tokens = 0;
        prompt_end = 0;
        prompt_ahead.clear();
        n_slots = 0;
        mm.reset();
        mm_slot = -1;
    }

    void add_token(llama_token tok, llama_pos pos, int32_t seq_id, int slot_id, bool need_logits,
                   float temp = 0.0f, int32_t top_k = 1, float top_p = 1.0f, uint64_t rng = 0,
                   uint64_t rng_base = 0, int32_t n_gen = 0, int32_t n_left = 0x7fffffff) {
        tokens.push_back(tok);
        positions.push_back(pos);
        seq_ids.push_back(seq_id);
        slot_ids.push_back(slot_id);
        logits_mask.push_back(need_logits);
        samp_temp.push_back(temp);
        samp_top_k.push_back(top_k);
        samp_top_p.push_back(top_p);
        samp_rng.push_back(rng);
        samp_rng_base.push_back(rng_base);
        samp_n_gen.push_back(n_gen);
        samp_n_left.push_back(n_left);
        n_tokens++;
    }
};

// GPU -> Scheduler
// Sampled tokens from completed decode. One token per slot.
struct BatchResult {
    std::vector<llama_token> sampled;   // token per slot (4 bytes each)
    std::vector<int>         slot_ids;  // which slot each token belongs to
    int n_slots = 0;
    // MTP window decode (ALPHA_MTP): tokens emitted after sampled[0] by the window loop. Single-slot
    // only.
    std::vector<llama_token> extra_tokens;
    // Per-slot extra tokens, parallel to sampled[] (empty = none): for engines that emit several tokens for several
    // slots in one batch (AlphaInfer MTP rounds). Takes precedence over extra_tokens for that entry.
    std::vector<std::vector<llama_token>> slot_extra;
    // llama_decode failed for this batch (e.g. KV/context exhausted). The scheduler
    // must force-finish the affected slots; an empty result would leave them generating
    // forever.
    bool decode_error = false;
    // Per-slot KV-pool exhaustion (dynamic over-subscription): these slots could not allocate a
    // KV page this step because the shared page pool is full. The scheduler force-finishes them
    // (clean stop, frees their pages for waiting requests) instead of letting them generate with
    // inconsistent KV. Empty in the common case (short gens never exhaust an over-subscribed pool).
    std::vector<int> exhausted_slots;
};

// Scheduler -> Output
// Token event for streaming to a client.
struct TokenEvent {
    int slot_id = -1;
    int request_id = -1;
    llama_token token = LLAMA_TOKEN_NULL;
    std::string text;          // detokenized text
    bool is_stop = false;      // generation complete
    bool is_eos = false;       // end-of-sequence token
    bool is_error = false;     // finished by a decode error: the reply is truncated (finish_reason "error")
    bool is_length = false;    // finished by max_tokens / the context / the KV pool, not by the model (finish_reason "length")
    int n_generated = 0;       // total tokens generated so far for this slot
};

// HTTP -> Scheduler
// New inference request from HTTP handler.
struct InferenceRequest {
    int request_id = -1;
    std::vector<llama_token> prompt_tokens;
    std::string raw_prompt;     // for logging
    std::shared_ptr<mm_prompt> mm;   // image input: prompt_tokens are then placeholders of the same length

    // Sampling parameters
    float temperature = 0.7f;
    int   top_k = 40;
    float top_p = 0.95f;
    int   seed = -1;
    int   max_tokens = 200;

    // Stop conditions
    std::vector<std::string> stop_words;

    // Reasoning budget: at most this many tokens inside <think>...</think> (-1 = unlimited, 0 = close at once). At the
    // budget the scheduler feeds think_close (an optional "time to answer" line + "</think>\n\n") as the next tokens.
    int reasoning_budget = -1;
    bool think_open = false;                 // the prompt ends inside an open <think> (template-opened thinking)
    std::vector<llama_token> think_close;

    // Response tracking
    bool stream = false;
    void * http_connection = nullptr;  // opaque pointer to HTTP response writer
};

} // namespace alphaserve
