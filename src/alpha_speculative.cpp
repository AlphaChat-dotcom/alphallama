#include "alpha_speculative.h"
#include <fstream>
#include <algorithm>
#include <string>
#include "alpha_moe_optimizer_internal.h"
#include "llama-ext.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-context.h"   // get_sched() for the batched async snapshot
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"   // ggml_backend_alloc_ctx_tensors_from_buft

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

// Timer helper
static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Init / cleanup

void alpha_speculative_init(alpha_speculative_state * state, alpha_moe_optimizer * opt) {
    if (!state || !opt || !opt->is_moe || !opt->cache_enabled) return;

    // Auto-detect dimensions from optimizer (no hardcoded values)
    state->n_expert_used = opt->n_expert_used;
    state->hidden_dim = opt->n_embd;

    // Count MoE layers (layers with CPU-resident experts and active cache)
    state->n_moe_layers = 0;
    for (int il = 0; il < opt->n_layer; il++) {
        if (opt->layer_info[il].has_experts && !opt->layer_info[il].on_gpu) {
            state->n_moe_layers++;
        }
    }

    if (state->n_moe_layers == 0) {
        ALPHA_LOG_DBG( "alpha_speculative: no MoE layers with cache — speculative disabled\n");
        state->initialized = false;
        return;
    }

    // Allocate per-layer storage
    state->layer_hidden.resize(state->n_moe_layers);
    state->layer_actual_ids.resize(state->n_moe_layers);
    for (int i = 0; i < state->n_moe_layers; i++) {
        state->layer_hidden[i].resize(state->hidden_dim, 0.0f);
        state->layer_actual_ids[i].resize(state->n_expert_used, 0);
    }

    state->first_mismatch = -1;

    // Statistics
    state->total_tokens = 0;
    state->speculative_hits = 0;
    state->speculative_misses = 0;
    state->total_rerun_layers = 0;
    state->total_spec_time_ms = 0;
    state->total_verify_time_ms = 0;
    state->total_rerun_time_ms = 0;

    state->initialized = true;

    ALPHA_LOG_DBG( "alpha_speculative: initialized — %d MoE layers, %d experts/token, hidden_dim=%d\n",
            state->n_moe_layers, state->n_expert_used, state->hidden_dim);
}

void alpha_speculative_cleanup(alpha_speculative_state * state) {
    if (!state) return;
    state->layer_hidden.clear();
    state->layer_actual_ids.clear();
    state->initialized = false;
}

// Verify
//
// After a speculative (prediction-only) graph completes:
// 1. Read each MoE layer's argsort output (actual expert IDs selected by gate)
// 2. For each layer, check: were all actual experts present in the GPU cache?
// 3. Find the first layer where a cache miss occurred (first_mismatch)
//
// If the prediction matched at a layer, the cache had the correct experts, MUL_MAT_ID
// read correct weights, and the hidden state is bit-identical to sync reactive. So
// layers 0..M-1 are guaranteed correct.

bool alpha_speculative_verify(alpha_speculative_state * state, alpha_moe_optimizer * opt) {
    if (!state || !state->initialized || !opt) return true;

    double t0 = now_ms();
    state->first_mismatch = -1;

    int n_argsort = alpha_lora_mix_get_argsort_count();
    int moe_idx = 0;

    for (int il = 0; il < opt->n_layer && moe_idx < n_argsort; il++) {
        auto & info = opt->layer_info[il];
        if (!info.has_experts || info.on_gpu) continue;

        int local_moe = moe_idx;  // MoE layer index (0-based within MoE layers)
        ggml_tensor * argsort_out = alpha_lora_mix_get_argsort_output(moe_idx);
        moe_idx++;

        if (!argsort_out || !argsort_out->buffer) continue;

        int n_used = std::min((int)argsort_out->ne[0], opt->n_expert_used);
        if (n_used <= 0) continue;

        // Read actual expert IDs for token 0 (single-token generation)
        int32_t actual_ids[16] = {};
        ggml_backend_tensor_get(argsort_out, actual_ids, 0, n_used * sizeof(int32_t));

        // Save actual IDs for statistics / next-token prediction
        if (local_moe < state->n_moe_layers) {
            for (int e = 0; e < n_used && e < state->n_expert_used; e++) {
                state->layer_actual_ids[local_moe][e] = actual_ids[e];
            }
        }

        // Also read the hidden state (FFN input) for this layer
        // (the hidden state before MoE), for a possible partial re-run from a checkpoint
        ggml_tensor * ffn_in = alpha_lora_mix_get_ffn_input(local_moe);
        if (ffn_in && ffn_in->buffer && local_moe < state->n_moe_layers) {
            int dim = std::min((int)ggml_nelements(ffn_in), state->hidden_dim);
            ggml_backend_tensor_get(ffn_in, state->layer_hidden[local_moe].data(),
                                    0, dim * sizeof(float));
        }

        // If first_mismatch was already found, skip verification for later layers
        // but still read their data for statistics
        if (state->first_mismatch >= 0) continue;

        // Verify: are all actual experts present in the GPU cache?
        //
        // Check the cache slot_map: if slot_map[op][expert_id] >= 0, the expert
        // is in a cache slot (either hot or recently loaded). The speculative
        // pass used prepare() which wrote cache_expert_ids based on predicted
        // experts + hot experts. If the actual experts are ALL in cache,
        // MUL_MAT_ID read correct weights and the speculative output is valid.
        //
        // Hot experts are always in cache, so if the actual routing selects only
        // cached experts, the speculative output is correct even when the
        // prev_routing prediction was wrong.
        auto & cache = opt->caches[il];

        for (int e = 0; e < n_used; e++) {
            int eid = actual_ids[e];
            if (eid < 0 || eid >= opt->n_expert) {
                state->first_mismatch = local_moe;
                break;
            }
            // Check OP_GATE (slot_map is [OP_COUNT][256], all ops share mapping)
            if (cache.slot_map[0][eid] < 0) {
                // Expert not in cache: speculative used wrong/zero weights
                state->first_mismatch = local_moe;
                break;
            }
        }
    }

    double t1 = now_ms();
    state->total_verify_time_ms += (t1 - t0);

    return (state->first_mismatch < 0);
}

// Re-run
//
// When verification finds a mismatch, the speculative output is wrong and the
// graph must be re-run with the reactive callback enabled.
//
// This does a full re-run with the reactive callback, but the
// callback is aware of which layers had correct predictions and skips
// expensive DMA for those layers (they're already in cache from prepare()).
//
// Why not partial re-run from layer M?
// - GGML graphs are monolithic: there is no API to "start from layer M"
// - Building a partial graph requires deep model-specific changes
// - The reactive callback already efficiently handles this: layers with
//     correct predictions get instant cache hits (no DMA), so re-running
//     them is cheap (~0.5ms per correct layer vs ~3ms per incorrect layer)
//
// TODO: partial re-run once GGML supports partial graph execution.

void alpha_speculative_rerun(alpha_speculative_state * state, alpha_moe_optimizer * opt,
                             struct llama_context * ctx) {
    // Note: the full re-run logic is inside alpha_speculative_decode_token().
    // This standalone function is provided for callers that manage the decode loop
    // themselves. It sets up reactive mode; the caller must call llama_decode() after.
    if (!state || !state->initialized || !opt || !ctx) return;
    if (state->first_mismatch < 0) return;

    // Count re-run layers for statistics
    int rerun_layers = state->n_moe_layers - state->first_mismatch;
    state->total_rerun_layers += rerun_layers;

    // Rewind KV cache to remove speculative token's entries
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        if (pos_max >= 0) {
            llama_memory_seq_rm(mem, 0, pos_max, -1);
        }
    }

    // Switch to reactive mode
    alpha_moe_optimizer_set_speculative(opt, false);
    alpha_moe_optimizer_prepare(opt);
}

// Full speculative decode loop
//
// This is the main entry point for speculative token generation.
// Replaces the simple prepare -> decode -> record loop with:
//   prepare -> speculate -> verify -> (rerun if needed) -> record

int alpha_speculative_decode_token(
    alpha_speculative_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    llama_token token)
{
    if (!state || !state->initialized || !opt || !ctx) {
        // Fallback: regular decode
        llama_batch batch = llama_batch_get_one(&token, 1);
        return llama_decode(ctx, batch);
    }

    state->total_tokens++;

    // Adaptive disable: if the hit rate is too low, skip speculative.
    // After 10 tokens, if hit rate < 10%, speculative is slower than reactive.
    // Fall back to pure reactive (single graph pass with callback).
    if (state->total_tokens > 10 && state->speculative_hits == 0) {
        // Zero hits after 10+ tokens -> speculative is pure overhead
        alpha_moe_optimizer_set_speculative(opt, false);
        alpha_moe_optimizer_prepare(opt);

        llama_batch batch = llama_batch_get_one(&token, 1);
        int rc = llama_decode(ctx, batch);
        if (rc != 0) return rc;

        alpha_moe_optimizer_record(opt);
        state->speculative_misses++;
        return 0;
    }

    // Phase 1: prepare (pre-load predicted experts).
    // This is the existing prepare(): loads predicted experts to GPU cache
    // and writes cache_expert_ids with predicted slot mappings.
    alpha_moe_optimizer_set_speculative(opt, true);
    alpha_moe_optimizer_prepare(opt);

    // Phase 2: speculate (run graph with no reactive callback).
    // The callback is disabled (speculative_pass = true), so the graph
    // runs at full speed using predicted expert slots.
    double t_spec_start = now_ms();

    llama_batch batch = llama_batch_get_one(&token, 1);
    int rc = llama_decode(ctx, batch);
    if (rc != 0) return rc;

    double t_spec_end = now_ms();
    state->total_spec_time_ms += (t_spec_end - t_spec_start);

    // Phase 3: verify (check predictions against actual routing)
    bool all_correct = alpha_speculative_verify(state, opt);

    ALPHA_LOG_DBG("[SPEC] token %lld: verify=%s first_mismatch=%d\n",
            (long long)state->total_tokens,
            all_correct ? "HIT" : "MISS",
            state->first_mismatch);

    if (all_correct) {
        // Fast path: all predictions correct.
        // Output is bit-identical to sync reactive. Use it.
        state->speculative_hits++;

        // Record actual routing for next token's prediction
        alpha_moe_optimizer_set_speculative(opt, false);
        alpha_moe_optimizer_record(opt);

        return 0;
    }

    // Phase 4: rerun (mismatch found, the output is invalid)
    state->speculative_misses++;

    // Rewind the KV cache: remove the speculative token's KV entries
    // so the re-run produces clean KV cache entries.
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        if (pos_max >= 0) {
            llama_memory_seq_rm(mem, 0, pos_max, -1);
        }
    }

    // Switch to reactive mode and re-decode
    alpha_moe_optimizer_set_speculative(opt, false);
    alpha_moe_optimizer_prepare(opt);

    double t_rerun_start = now_ms();

    rc = llama_decode(ctx, batch);
    if (rc != 0) return rc;

    double t_rerun_end = now_ms();
    state->total_rerun_time_ms += (t_rerun_end - t_rerun_start);

    // Record actual routing for next token
    alpha_moe_optimizer_record(opt);

    return 0;
}

// Statistics

void alpha_speculative_print_stats(const alpha_speculative_state * state) {
    if (!state || !state->initialized || state->total_tokens == 0) return;

    double hit_rate = 100.0 * state->speculative_hits / state->total_tokens;
    double avg_rerun_layers = state->speculative_misses > 0
        ? (double)state->total_rerun_layers / state->speculative_misses : 0;

    bool adaptive_disabled = (state->total_tokens > 10 && state->speculative_hits == 0);

    ALPHA_LOG_DBG( "\n=== Speculative Execution Stats ===\n");
    ALPHA_LOG_DBG( "  Tokens: %lld\n", (long long)state->total_tokens);
    ALPHA_LOG_DBG( "  Hits:   %lld (%.1f%%) — all predictions correct, full speed\n",
            (long long)state->speculative_hits, hit_rate);
    ALPHA_LOG_DBG( "  Misses: %lld (%.1f%%) — required re-run\n",
            (long long)state->speculative_misses, 100.0 - hit_rate);
    if (adaptive_disabled) {
        ALPHA_LOG_DBG( "  NOTE: Adaptive fallback triggered after 10 tokens (0%% hits)\n");
        ALPHA_LOG_DBG( "        Switched to pure reactive mode (no double-graph penalty)\n");
    }

    if (state->speculative_misses > 0) {
        ALPHA_LOG_DBG( "  Avg re-run depth: %.1f/%d layers (first mismatch)\n",
                avg_rerun_layers, state->n_moe_layers);
    }

    double total_time = state->total_spec_time_ms + state->total_verify_time_ms + state->total_rerun_time_ms;
    if (total_time > 0) {
        double tok_per_sec = 1000.0 * state->total_tokens / total_time;
        ALPHA_LOG_DBG( "  Timing:\n");
        ALPHA_LOG_DBG( "    Speculative: %.1f ms total (%.2f ms/tok)\n",
                state->total_spec_time_ms, state->total_spec_time_ms / state->total_tokens);
        ALPHA_LOG_DBG( "    Verify:      %.1f ms total (%.2f ms/tok)\n",
                state->total_verify_time_ms, state->total_verify_time_ms / state->total_tokens);
        ALPHA_LOG_DBG( "    Re-run:      %.1f ms total (%.2f ms/tok)\n",
                state->total_rerun_time_ms, state->total_rerun_time_ms / state->total_tokens);
        ALPHA_LOG_DBG( "    Throughput:  %.1f tok/s\n", tok_per_sec);
    }

    ALPHA_LOG_DBG( "\n");
}

// Self-speculative decoding: draft with shared experts, verify with full model

void alpha_self_spec_init(alpha_self_spec_state * state, alpha_moe_optimizer * opt, int n_draft) {
    if (!state || !opt) return;

    state->n_draft = n_draft;
    state->draft_tokens.reserve(n_draft);
    state->n_accepted_total = 0;
    state->n_draft_total = 0;
    state->n_cycles = 0;
    state->draft_ms_total = 0;
    state->verify_ms_total = 0;
    // Mode A (expert reduce): works on any MoE model
    // Mode C (backbone only): requires shared experts
    bool mode_a = (opt->draft_expert_count > 0);
    state->initialized = mode_a || alpha_moe_optimizer_has_shared_experts(opt);

    if (!state->initialized) {
        ALPHA_LOG_DBG("[SELF-SPEC] No shared experts and no expert reduce — disabled\n");
    } else {
        ALPHA_LOG_DBG("[SELF-SPEC] Initialized: %d draft tokens, %s\n", n_draft,
                mode_a ? "Mode A (expert reduce)" : "Mode C (backbone)");
    }
}

int alpha_self_spec_generate(
    alpha_self_spec_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    struct llama_sampler * smpl,
    const struct llama_vocab * vocab,
    llama_token last_token,
    llama_pos n_past,
    std::vector<llama_token> & accepted)
{
    if (!state || !state->initialized || !opt || !ctx) return 0;

    // Adaptive disable: if acceptance rate < 30% after 5 cycles, fall back to reactive
    if (state->n_cycles >= 5 && state->n_draft_total > 0) {
        double accept_rate = (double)state->n_accepted_total / state->n_draft_total;
        if (accept_rate < 0.3) {
            // Self-spec not worth it: fall back to single reactive decode
            alpha_moe_optimizer_set_draft_mode(opt, false);
            alpha_moe_optimizer_prepare(opt);

            llama_batch batch = llama_batch_get_one(&last_token, 1);
            if (llama_decode(ctx, batch) != 0) return 0;

            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            alpha_moe_optimizer_record(opt);
            accepted.push_back(tok);
            return 1;
        }
    }

    // Draft phase (seq branching + recurrent snapshot).
    // Attention KV: seq_cp deep-copies correctly (seq 0 -> seq 1)
    // Recurrent state: seq_cp does shallow copy (shares cell). We snapshot
    // the recurrent r_l/s_l tensors via direct GPU copy and restore after draft.
    alpha_moe_optimizer_set_draft_mode(opt, true);

    // Disable eval callback during draft for zero overhead (backbone-only, no expert cache)
    auto saved_cb = alpha_moe_optimizer_get_eval_callback();
    auto saved_cb_data = alpha_moe_optimizer_get_eval_data(opt);
    llama_set_eval_callback(ctx, nullptr, nullptr);

    state->draft_tokens.clear();
    llama_token input = last_token;

    llama_memory_t mem = llama_get_memory(ctx);
    llama_pos draft_start = mem ? llama_memory_seq_pos_max(mem, 0) + 1 : n_past;

    // Snapshot recurrent state before seq_cp (which shares the cell)
    const struct llama_model * draft_model = llama_get_model(ctx);
    auto * hybrid_mem = draft_model && llama_model_is_hybrid(draft_model)
        ? dynamic_cast<llama_memory_hybrid *>(mem) : nullptr;
    llama_memory_recurrent * recr = hybrid_mem ? hybrid_mem->get_mem_recr() : nullptr;
    std::vector<std::vector<uint8_t>> r_snap, s_snap;
    if (recr) {
        for (auto * r : recr->r_l) {
            if (!r) { r_snap.push_back({}); continue; }
            std::vector<uint8_t> buf(ggml_nbytes(r));
            ggml_backend_tensor_get(r, buf.data(), 0, buf.size());
            r_snap.push_back(std::move(buf));
        }
        for (auto * s : recr->s_l) {
            if (!s) { s_snap.push_back({}); continue; }
            std::vector<uint8_t> buf(ggml_nbytes(s));
            ggml_backend_tensor_get(s, buf.data(), 0, buf.size());
            s_snap.push_back(std::move(buf));
        }
    }

    const llama_seq_id draft_seq = 1;
    if (mem) {
        llama_memory_seq_cp(mem, 0, draft_seq, 0, -1);
    }

    double t0 = now_ms();

    for (int i = 0; i < state->n_draft; i++) {
        if (opt->draft_expert_count > 0) {
            alpha_moe_optimizer_prepare(opt);
        }

        llama_batch dbatch = llama_batch_init(1, 0, 1);
        dbatch.n_tokens = 1;
        dbatch.token[0]    = input;
        dbatch.pos[0]      = draft_start + i;
        dbatch.n_seq_id[0] = 1;
        dbatch.seq_id[0][0]= draft_seq;
        dbatch.logits[0]   = 1;

        if (llama_decode(ctx, dbatch) != 0) {
            llama_batch_free(dbatch);
            break;
        }
        llama_batch_free(dbatch);

        // Use argmax (greedy) for draft sampling, not the sampler.
        const float * logits = llama_get_logits(ctx);
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token tok = 0;
        if (logits && n_vocab > 0) {
            float best = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best) {
                    best = logits[v];
                    tok = v;
                }
            }
        }
        state->draft_tokens.push_back(tok);

        if (llama_vocab_is_eog(vocab, tok)) break;
        input = tok;
    }

    double t1 = now_ms();
    state->draft_ms_total += (t1 - t0);

    // Delete seq 1 (full clear always succeeds on all memory types)
    if (mem) {
        llama_memory_seq_rm(mem, draft_seq, 0, -1);
    }

    // Restore recurrent state from snapshot (undoes draft's modifications)
    if (recr) {
        for (size_t i = 0; i < recr->r_l.size() && i < r_snap.size(); i++) {
            if (recr->r_l[i] && !r_snap[i].empty())
                ggml_backend_tensor_set(recr->r_l[i], r_snap[i].data(), 0, r_snap[i].size());
        }
        for (size_t i = 0; i < recr->s_l.size() && i < s_snap.size(); i++) {
            if (recr->s_l[i] && !s_snap[i].empty())
                ggml_backend_tensor_set(recr->s_l[i], s_snap[i].data(), 0, s_snap[i].size());
        }
    }

    // Re-enable eval callback for verify phase
    llama_set_eval_callback(ctx, saved_cb, saved_cb_data);

    if (state->draft_tokens.empty()) {
        alpha_moe_optimizer_set_draft_mode(opt, false);
        return 0;
    }

    // Verify phase: full model with reactive callbacks + SG=2.
    // SG=2 during verify halves the callback overhead and produces correct output.
    // Skipped layers use zero-expert fallback; shared experts bridge the gap.
    alpha_moe_optimizer_set_draft_mode(opt, false);
    int saved_sg = opt->split_group;
    if (opt->split_group < 2) {
        opt->split_group = 2;
    }
    alpha_moe_optimizer_prepare(opt);

    int n_draft_actual = (int)state->draft_tokens.size();
    int n_verify = 1 + n_draft_actual;

    // Build verification batch: [last_accepted, draft_0, ..., draft_N-1]
    // Positions start at draft_start (seq 0's next position after prompt)
    llama_batch vbatch = llama_batch_init(n_verify, 0, 1);
    vbatch.n_tokens = n_verify;

    vbatch.token[0]    = last_token;
    vbatch.pos[0]      = draft_start;
    vbatch.n_seq_id[0] = 1;
    vbatch.seq_id[0][0]= 0;
    vbatch.logits[0]   = 1;

    for (int i = 0; i < n_draft_actual; i++) {
        vbatch.token[1 + i]    = state->draft_tokens[i];
        vbatch.pos[1 + i]      = draft_start + 1 + i;
        vbatch.n_seq_id[1 + i] = 1;
        vbatch.seq_id[1 + i][0]= 0;
        vbatch.logits[1 + i]   = 1;
    }

    double t2 = now_ms();

    int rc = llama_decode(ctx, vbatch);
    if (rc != 0) {
        opt->split_group = saved_sg;
        llama_batch_free(vbatch);
        return 0;
    }

    alpha_moe_optimizer_record(opt);
    opt->split_group = saved_sg;  // restore generation SG

    double t3 = now_ms();
    state->verify_ms_total += (t3 - t2);

    // Accept prefix: compare verify logits vs draft tokens
    int n_accepted = 0;

    for (int i = 0; i < n_draft_actual; i++) {
        // Sample from verify logits at position i (predicts token at position n_past+1+i)
        llama_token vtok = llama_sampler_sample(smpl, ctx, i);

        if (vtok == state->draft_tokens[i]) {
            // Match: accept this draft token
            accepted.push_back(vtok);
            n_accepted++;
        } else {
            // Mismatch: accept the verify token (correct), reject the rest
            accepted.push_back(vtok);
            n_accepted++;
            break;
        }
    }

    // If all drafts were accepted, sample a bonus token from the last position
    if (n_accepted == n_draft_actual) {
        llama_token bonus = llama_sampler_sample(smpl, ctx, n_verify - 1);
        accepted.push_back(bonus);
        n_accepted++;
    }

    // Roll back rejected tokens from the KV cache.
    // For hybrid memory: partial removal may fail. If so, keep all verify tokens
    // (they're from the full model, so they're correct; this just wastes KV slots).
    if (mem && n_accepted < n_verify) {
        bool rm_ok = llama_memory_seq_rm(mem, 0, draft_start + n_accepted, -1);
        if (!rm_ok) {
            // Hybrid memory can't partially rollback verify tokens.
            // Accept all verify tokens instead (full model output is correct).
            // This means we use N+1 KV slots per cycle instead of n_accepted.
            n_accepted = n_verify;
            // Re-sample all tokens as accepted
            accepted.clear();
            for (int i = 0; i < n_draft_actual; i++) {
                llama_token vtok = llama_sampler_sample(smpl, ctx, i);
                accepted.push_back(vtok);
            }
            llama_token bonus = llama_sampler_sample(smpl, ctx, n_verify - 1);
            accepted.push_back(bonus);
            n_accepted = (int)accepted.size();
        }
    }

    // Stats
    state->n_accepted_total += n_accepted;
    state->n_draft_total += n_draft_actual;
    state->n_cycles++;

    ALPHA_LOG_DBG("[SELF-SPEC] cycle %lld: drafted %d, accepted %d (%.0f%%), draft %.0fms, verify %.0fms\n",
            (long long)state->n_cycles, n_draft_actual, n_accepted,
            n_draft_actual > 0 ? 100.0 * (n_accepted - 1) / n_draft_actual : 0.0,
            t1 - t0, t3 - t2);

    llama_batch_free(vbatch);
    return n_accepted;
}

void alpha_self_spec_print_stats(const alpha_self_spec_state * state) {
    if (!state || !state->initialized || state->n_cycles == 0) return;

    double accept_rate = state->n_draft_total > 0
        ? 100.0 * (state->n_accepted_total - state->n_cycles) / state->n_draft_total : 0;
    double avg_accepted = (double)state->n_accepted_total / state->n_cycles;
    double total_time = state->draft_ms_total + state->verify_ms_total;
    double tok_per_sec = total_time > 0 ? 1000.0 * state->n_accepted_total / total_time : 0;

    ALPHA_LOG_DBG( "\n=== Self-Speculative Stats ===\n");
    ALPHA_LOG_DBG( "  Cycles:     %lld\n", (long long)state->n_cycles);
    ALPHA_LOG_DBG( "  Drafted:    %lld tokens total\n", (long long)state->n_draft_total);
    ALPHA_LOG_DBG( "  Accepted:   %lld tokens total (%.1f%% draft match rate)\n",
            (long long)state->n_accepted_total, accept_rate);
    ALPHA_LOG_DBG( "  Avg accept: %.1f tokens/cycle (of %d drafted)\n",
            avg_accepted, state->n_draft);
    ALPHA_LOG_DBG( "  Timing:\n");
    ALPHA_LOG_DBG( "    Draft:    %.0f ms total (%.1f ms/cycle)\n",
            state->draft_ms_total, state->draft_ms_total / state->n_cycles);
    ALPHA_LOG_DBG( "    Verify:   %.0f ms total (%.1f ms/cycle)\n",
            state->verify_ms_total, state->verify_ms_total / state->n_cycles);
    ALPHA_LOG_DBG( "  Throughput: %.1f tok/s\n", tok_per_sec);
    ALPHA_LOG_DBG( "\n");
}

// Cached-expert self-speculative decoding

void alpha_cached_spec_init(alpha_cached_spec_state * state, int n_draft) {
    if (!state) return;
    state->n_draft = n_draft;
    state->draft_tokens.reserve(n_draft);
    state->n_accepted_total = 0;
    state->n_draft_total = 0;
    state->n_cycles = 0;
    state->draft_ms_total = 0;
    state->verify_ms_total = 0;
    state->initialized = true;
    state->repetition_detected = false;
}

int alpha_cached_spec_generate(
    alpha_cached_spec_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    struct llama_sampler * smpl,
    const struct llama_vocab * vocab,
    llama_token last_token,
    llama_pos n_past,
    std::vector<llama_token> & accepted) {

    if (!state || !state->initialized || !opt || !ctx) return 0;

    const int M = state->n_draft;
    int n_vocab = llama_vocab_n_tokens(vocab);
    accepted.clear();
    state->draft_tokens.clear();

    llama_memory_t mem = llama_get_memory(ctx);

    // Recompute n_past from actual KV state (caller's value may be stale)
    if (mem) {
        n_past = llama_memory_seq_pos_max(mem, 0) + 1;
    }

    // Step 1: snapshot recurrent state for rollback.
    // Hybrid models (e.g. qwen35moe) have recurrent layers whose state is a running
    // accumulator that seq_rm can't partially erase. Saving/restoring only the
    // recurrent R/S tensors (~180 KB) resets the tail position so seq_rm succeeds.
    auto * hybrid_mem = dynamic_cast<llama_memory_hybrid *>(mem);
    llama_memory_recurrent * recr = hybrid_mem ? hybrid_mem->get_mem_recr() : nullptr;
    std::vector<std::vector<uint8_t>> r_snap, s_snap;
    if (recr) {
        for (auto * r : recr->r_l) {
            if (!r) { r_snap.push_back({}); continue; }
            std::vector<uint8_t> buf(ggml_nbytes(r));
            ggml_backend_tensor_get(r, buf.data(), 0, buf.size());
            r_snap.push_back(std::move(buf));
        }
        for (auto * s : recr->s_l) {
            if (!s) { s_snap.push_back({}); continue; }
            std::vector<uint8_t> buf(ggml_nbytes(s));
            ggml_backend_tensor_get(s, buf.data(), 0, buf.size());
            s_snap.push_back(std::move(buf));
        }
    }

    // Step 2: draft M tokens (fast, GPU LUT, no callback)
    double t_draft_start = now_ms();

    bool cpu_verify = opt->cpu_moe_verify;
    bool use_mixed = alpha_moe_optimizer_has_mixed_experts(opt);

    alpha_moe_optimizer_enable_cache(opt);
    if (use_mixed) {
        alpha_moe_optimizer_enter_mixed_draft_mode(opt);
    } else {
        alpha_moe_optimizer_enter_draft_mode(opt);
    }
    // Re-reserve graph: expert tensors may have moved from CPU to GPU cache
    llama_graph_reserve(ctx, 1, 1, 1);

    llama_token cur_token = last_token;
    std::vector<llama_token> decode_seq;
    decode_seq.push_back(last_token);

    // Adaptive M: draft up to M tokens, stop early if confidence drops.
    // High confidence -> keep drafting (likely to be accepted by verify).
    // Low confidence -> stop and verify what we have.
    const float conf_threshold = 0.5f;  // logit margin (top1 - top2) threshold
    int actual_M = 0;

    for (int i = 0; i < M; i++) {
        llama_batch batch = llama_batch_get_one(&cur_token, 1);
        int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            if (use_mixed) alpha_moe_optimizer_exit_mixed_draft_mode(opt);
            else alpha_moe_optimizer_exit_draft_mode(opt);
            return 0;
        }

        float * logits = llama_get_logits(ctx);

        // Find top-1 and top-2 logit values for confidence
        float best1 = -1e30f, best2 = -1e30f;
        llama_token draft_tok = 0;
        for (int v = 0; v < n_vocab; v++) {
            if (logits[v] > best1) {
                best2 = best1;
                best1 = logits[v];
                draft_tok = v;
            } else if (logits[v] > best2) {
                best2 = logits[v];
            }
        }

        state->draft_tokens.push_back(draft_tok);
        decode_seq.push_back(draft_tok);
        cur_token = draft_tok;
        actual_M++;

        // Stop drafting if confidence drops (logit margin too small)
        // or if we hit EOS. Always draft at least 4 tokens before checking confidence.
        float margin = best1 - best2;
        if (margin < conf_threshold && i >= 4) {
            break;  // low confidence: verify what we have
        }
        // Don't break on EOS during the first few draft tokens: verify needs to run
        // at least once to warm the cache with prompt-specific experts via DMA.
        if (llama_vocab_is_eog(vocab, draft_tok) && i >= 4) {
            break;
        }
        // Draft repetition detection: if last 3 tokens are identical, stop drafting.
        // Repetitive draft = corrupted hidden state (zero-expert contamination).
        if (i >= 5 && state->draft_tokens.size() >= 3) {
            size_t n = state->draft_tokens.size();
            if (state->draft_tokens[n-1] == state->draft_tokens[n-2] &&
                state->draft_tokens[n-2] == state->draft_tokens[n-3]) {
                break;  // 3 identical tokens in a row: repetition loop
            }
        }
    }

    // Use pure LUT verify and skip the callback overhead entirely.
    // The LUT maps all experts to cached slots, so verify computes identically to draft.
    // Safe because the draft already proved these tokens work with the LUT and verify
    // only confirms them. Avoids a GPU pipeline drain/refill sync per layer.
    bool lut_verify = !cpu_verify && !use_mixed;

    if (!lut_verify) {
        if (use_mixed) {
            alpha_moe_optimizer_exit_mixed_draft_mode(opt);
        } else {
            alpha_moe_optimizer_exit_draft_mode(opt);
        }
    }
    // else: stay in draft/LUT mode; verify uses the same GPU path as draft
    double t_draft_end = now_ms();

    // Step 3: reload cache from draft argsort.
    // Skip reload when verify uses cpu-moe or LUT (cache already correct)
    if (!use_mixed && !cpu_verify && !lut_verify) {
        alpha_moe_optimizer_reload_from_argsort(opt);
    }

    // Step 4: restore recurrent state + rollback KV
    double t_verify_start = now_ms();

    // Restore recurrent R/S tensors to pre-draft state (if hybrid model)
    // This resets the recurrent tail position so seq_rm can succeed.
    if (recr) {
        for (size_t i = 0; i < r_snap.size() && i < recr->r_l.size(); i++) {
            if (recr->r_l[i] && !r_snap[i].empty()) {
                ggml_backend_tensor_set(recr->r_l[i], r_snap[i].data(), 0, r_snap[i].size());
            }
        }
        for (size_t i = 0; i < s_snap.size() && i < recr->s_l.size(); i++) {
            if (recr->s_l[i] && !s_snap[i].empty()) {
                ggml_backend_tensor_set(recr->s_l[i], s_snap[i].data(), 0, s_snap[i].size());
            }
        }
        // Also restore the recurrent cell metadata (pos, seq_id)
        // The cell tracks the tail position, which must reset to pre-draft
        for (uint32_t i = 0; i < recr->size; i++) {
            if (recr->cells[i].has_seq_id(0) && recr->cells[i].pos >= n_past) {
                recr->cells[i].pos = n_past - 1;  // reset to last prompt position
            }
        }
    }

    // Now seq_rm succeeds: recurrent tail is at n_past-1, we remove n_past+
    bool rollback_ok = true;
    if (mem) {
        rollback_ok = llama_memory_seq_rm(mem, 0, n_past, -1);
    }

    // Step 5: batched verify (one forward pass for M+1 tokens)
    int n_verify = (int)decode_seq.size();  // M+1 tokens
    int n_accepted = 0;
    int vrc = -1;

    // Verify: try GPU callback first, fall back to cpu-moe if it fails.
    // After exit_draft_mode, callback-mode cache_expert_ids are restored.
    // GPU callback verify: DMA misses from CPU, GPU computes all experts.
    // TODO: GPU callback verify crashes on hybrid models: optimizer internal state
    // (cache clock, slot_map) is inconsistent after draft+rollback and needs a reset
    // after rollback. Until then cpu-moe verify is used (slower but stable).
    if (!rollback_ok) {
        ALPHA_LOG_DBG("[CACHED-SPEC] WARNING: KV rollback failed despite recurrent restore\n");
    }

    if (lut_verify) {
        ALPHA_LOG_DBG("[CACHED-SPEC] LUT verify (100%% cache hit, no callback overhead)\n");
    }
    // GPU verify: either LUT (no callback) or callback (DMA on miss).
    // graph_reserve(1,1,1) is safe for both modes.
    // (Batched verify would be faster but requires multi-token graph_reserve
    // which fails on hybrid models with recurrent layers.)
    llama_graph_reserve(ctx, 1, 1, 1);

    // Per-token recurrent checkpoints for mid-verify rollback on hybrid models.
    // Each save is ~180 KB (tiny). Enables partial acceptance on any prompt type.
    std::vector<std::vector<uint8_t>> last_r_snap, last_s_snap;

    // Repetition detection: track consecutive identical verify tokens with high confidence.
    // Pattern: 3+ identical tokens with high margin = repetition loop (model stuck).
    llama_token prev_verify_tok = -1;
    int repeat_count = 0;
    const int REPEAT_LIMIT = 3;  // break after 3 identical high-confidence tokens

    for (int i = 0; i < n_verify; i++) {
        llama_token tok = decode_seq[i];
        llama_batch batch = llama_batch_get_one(&tok, 1);
        vrc = llama_decode(ctx, batch);
        if (vrc != 0) break;
        if (!lut_verify) {
            alpha_moe_optimizer_record(opt);  // skip in LUT mode: one GPU sync per layer
        }

        float * logits = llama_get_logits(ctx);
        if (!logits) break;

        llama_token verify_tok = 0;
        float best = logits[0], second = -1e30f;
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best) { second = best; best = logits[v]; verify_tok = v; }
            else if (logits[v] > second) { second = logits[v]; }
        }

        // Confidence pattern: high margin + same token = repetition loop
        float margin = best - second;
        if (verify_tok == prev_verify_tok && margin > 2.0f) {
            repeat_count++;
            if (repeat_count >= REPEAT_LIMIT && n_accepted > 0) {
                // Repetition detected: reject remaining draft tokens, break cycle
                ALPHA_LOG_DBG("[CACHED-SPEC] Repetition detected at verify pos %d "
                    "(tok=%d repeated %d times, margin=%.1f) — breaking cycle\n",
                    i, verify_tok, repeat_count + 1, margin);
                state->repetition_detected = true;
                break;
            }
        } else {
            repeat_count = 0;
        }
        prev_verify_tok = verify_tok;

        if (i < actual_M) {
            if (verify_tok == state->draft_tokens[i]) {
                accepted.push_back(state->draft_tokens[i]);
                n_accepted++;
                // Save recurrent checkpoint after each accepted token (~180 KB, ~0.01ms)
                if (recr) {
                    last_r_snap.clear(); last_s_snap.clear();
                    for (auto * r : recr->r_l) {
                        if (!r) { last_r_snap.push_back({}); continue; }
                        std::vector<uint8_t> buf(ggml_nbytes(r));
                        ggml_backend_tensor_get(r, buf.data(), 0, buf.size());
                        last_r_snap.push_back(std::move(buf));
                    }
                    for (auto * s : recr->s_l) {
                        if (!s) { last_s_snap.push_back({}); continue; }
                        std::vector<uint8_t> buf(ggml_nbytes(s));
                        ggml_backend_tensor_get(s, buf.data(), 0, buf.size());
                        last_s_snap.push_back(std::move(buf));
                    }
                }
            } else {
                // Mismatch: restore recurrent to last accepted checkpoint
                if (recr && !last_r_snap.empty()) {
                    for (size_t j = 0; j < last_r_snap.size() && j < recr->r_l.size(); j++) {
                        if (recr->r_l[j] && !last_r_snap[j].empty())
                            ggml_backend_tensor_set(recr->r_l[j], last_r_snap[j].data(), 0, last_r_snap[j].size());
                    }
                    for (size_t j = 0; j < last_s_snap.size() && j < recr->s_l.size(); j++) {
                        if (recr->s_l[j] && !last_s_snap[j].empty())
                            ggml_backend_tensor_set(recr->s_l[j], last_s_snap[j].data(), 0, last_s_snap[j].size());
                    }
                    // Reset recurrent cell pos to last accepted
                    for (uint32_t c = 0; c < recr->size; c++) {
                        if (recr->cells[c].has_seq_id(0) && recr->cells[c].pos >= (llama_pos)(n_past + n_accepted)) {
                            recr->cells[c].pos = n_past + n_accepted - 1;
                        }
                    }
                }
                // Accept the divergent verify token (correct from full model)
                accepted.push_back(verify_tok);
                n_accepted++;
                break;
            }
        } else {
            // Bonus token from last position (i == actual_M)
            accepted.push_back(verify_tok);
            n_accepted++;
        }
    }

    // Rollback rejected attention KV entries
    // After recurrent restore above, seq_rm should succeed (tail at last accepted)
    if (mem && n_accepted < n_verify) {
        llama_pos rollback_pos = n_past + n_accepted;
        llama_memory_seq_rm(mem, 0, rollback_pos, -1);
    }

    // Exit LUT/draft mode after verify (if we stayed in it for LUT verify)
    if (lut_verify) {
        alpha_moe_optimizer_exit_draft_mode(opt);
    }

    // Re-reserve graph for single-token generation
    llama_graph_reserve(ctx, 1, 1, 1);

    double t_verify_end = now_ms();

    // Update stats
    state->n_cycles++;
    state->n_draft_total += actual_M;
    state->n_accepted_total += n_accepted;
    state->draft_ms_total += (t_draft_end - t_draft_start);
    state->verify_ms_total += (t_verify_end - t_verify_start);

    return n_accepted;
}

void alpha_cached_spec_print_stats(const alpha_cached_spec_state * state) {
    if (!state || state->n_cycles == 0) return;
    double total_ms = state->draft_ms_total + state->verify_ms_total;
    double tok_per_sec = total_ms > 0 ? state->n_accepted_total * 1000.0 / total_ms : 0;
    double accept_rate = state->n_draft_total > 0 ?
        100.0 * state->n_accepted_total / state->n_draft_total : 0;
    double avg_accepted = state->n_cycles > 0 ?
        (double)state->n_accepted_total / state->n_cycles : 0;

    ALPHA_LOG_DBG( "\n=== Cached-Expert Self-Spec Stats ===\n");
    ALPHA_LOG_DBG( "  Cycles:     %lld\n", (long long)state->n_cycles);
    ALPHA_LOG_DBG( "  Drafted:    %lld tokens (M=%d)\n",
            (long long)state->n_draft_total, state->n_draft);
    ALPHA_LOG_DBG( "  Accepted:   %lld tokens (%.1f%% match rate)\n",
            (long long)state->n_accepted_total, accept_rate);
    ALPHA_LOG_DBG( "  Avg accept: %.1f tokens/cycle\n", avg_accepted);
    ALPHA_LOG_DBG( "  Draft:      %.0f ms total (%.1f ms/tok, %.1f tok/s)\n",
            state->draft_ms_total,
            state->draft_ms_total / std::max((int64_t)1, state->n_draft_total),
            state->n_draft_total * 1000.0 / std::max(1.0, state->draft_ms_total));
    ALPHA_LOG_DBG( "  Verify:     %.0f ms total (%.1f ms/tok)\n",
            state->verify_ms_total,
            state->verify_ms_total / std::max((int64_t)1, state->n_accepted_total));
    ALPHA_LOG_DBG( "  Throughput: %.1f tok/s (effective)\n", tok_per_sec);
    ALPHA_LOG_DBG( "\n");
}

// Two-tier speculative decoding
//
// Architecture: N cycles of (draft M + batched tier 1 verify), then batched tier 2.
//   Draft:  M tokens using co-activation cluster experts (GPU, no callback)
//   Tier 1: batched verify of M+1 tokens with adaptive-k cpu-moe (critical layers k=6, middle k=4)
//   Tier 2: batched verify of all accumulated tokens with full k=6 cpu-moe (ground truth)
//
// Each tier 1 cycle produces up to M+1 tokens. After N cycles, tier 2 verifies all at once.

void alpha_twotier_init(alpha_twotier_state * state, int n_draft, int tier1_cycles) {
    if (!state) return;
    state->n_draft = n_draft;
    state->tier1_cycles = tier1_cycles;
    state->draft_tokens.reserve(n_draft * tier1_cycles);
    state->n_draft_total = 0;
    state->n_tier1_accepted_total = 0;
    state->n_tier2_accepted_total = 0;
    state->n_rounds = 0;
    state->n_tier1_rejects = 0;
    state->n_tier2_rejects = 0;
    state->draft_ms_total = 0;
    state->tier1_ms_total = 0;
    state->tier2_ms_total = 0;
    state->initialized = true;
}

// Helper: save sequence state into buffer
static void save_seq_state(struct llama_context * ctx, llama_memory_t mem,
                           std::vector<uint8_t> & buf) {
    buf.clear();
    if (!mem) return;
    size_t sz = llama_state_seq_get_size(ctx, 0);
    if (sz > 0) {
        buf.resize(sz);
        llama_state_seq_get_data(ctx, buf.data(), sz, 0);
    }
}

// Helper: restore sequence state from buffer
static void restore_seq_state(struct llama_context * ctx, llama_memory_t mem,
                              const std::vector<uint8_t> & buf, llama_pos fallback_pos) {
    if (!buf.empty() && mem) {
        llama_memory_seq_rm(mem, 0, 0, -1);
        llama_state_seq_set_data(ctx, buf.data(), buf.size(), 0);
    } else if (mem) {
        llama_memory_seq_rm(mem, 0, fallback_pos, -1);
    }
}

// Helper: compute average per-layer routing confidence from gating probs
// Returns [0,1]: 1 = certain (low entropy, high top_prob), 0 = ambiguous
static float compute_draft_confidence(int n_moe_layers) {
    float total_conf = 0;
    int n = 0;

    for (int il = 0; il < n_moe_layers; il++) {
        ggml_tensor * gating = alpha_lora_mix_get_gating_probs(il);
        if (!gating || !gating->buffer) continue;

        int n_exp = (int)gating->ne[0];
        if (n_exp > 256) n_exp = 256;
        float raw[256];
        ggml_backend_tensor_get(gating, raw, 0, n_exp * sizeof(float));

        // Softmax normalization (raw values are logits, not probabilities)
        float max_val = raw[0];
        for (int e = 1; e < n_exp; e++) {
            if (raw[e] > max_val) max_val = raw[e];
        }
        float sum_exp = 0;
        for (int e = 0; e < n_exp; e++) {
            raw[e] = expf(raw[e] - max_val);
            sum_exp += raw[e];
        }
        if (sum_exp > 0) {
            for (int e = 0; e < n_exp; e++) raw[e] /= sum_exp;
        }

        // Entropy + top prob from softmax probs
        float entropy = 0;
        float top_prob = 0;
        for (int e = 0; e < n_exp; e++) {
            if (raw[e] > top_prob) top_prob = raw[e];
            if (raw[e] > 1e-8f) entropy -= raw[e] * log2f(raw[e]);
        }
        float max_ent = log2f((float)n_exp);
        float norm_ent = (max_ent > 0) ? entropy / max_ent : 0.5f;

        float layer_conf = (1.0f - norm_ent) * top_prob;
        total_conf += layer_conf;
        n++;
    }

    return n > 0 ? total_conf / n : 0.5f;
}

// Helper: argmax over logits
static llama_token argmax_logits(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_val = logits[0];
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > best_val) { best_val = logits[v]; best = v; }
    }
    return best;
}

int alpha_twotier_generate(
    alpha_twotier_state * state,
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    struct llama_sampler * smpl,
    const struct llama_vocab * vocab,
    llama_token last_token,
    llama_pos n_past,
    std::vector<llama_token> & accepted) {

    if (!state || !state->initialized || !opt || !ctx) return 0;

    int n_vocab = llama_vocab_n_tokens(vocab);
    accepted.clear();

    llama_memory_t mem = llama_get_memory(ctx);
    bool use_mixed = alpha_moe_optimizer_has_mixed_experts(opt);

    extern thread_local int g_alpha_topk_override;
    int saved_topk = g_alpha_topk_override;

    const int M_MIN = 2;
    const int M_MAX = 100;
    const float CONF_THRESHOLD = 0.8f;

    // Pre-decode last_token with k=6
    double t_predecode_start = now_ms();

    llama_set_eval_callback(ctx, nullptr, nullptr);
    g_alpha_topk_override = 0;

    llama_batch pre_batch = llama_batch_get_one(&last_token, 1);
    if (llama_decode(ctx, pre_batch) != 0) {
        alpha_moe_optimizer_reinstall_callback(opt);
        g_alpha_topk_override = saved_topk;
        return 0;
    }

    float * pre_logits = llama_get_logits(ctx);
    llama_token first_tok = argmax_logits(pre_logits, n_vocab);
    accepted.push_back(first_tok);  // 1 free verified token
    n_past++;

    alpha_moe_optimizer_reinstall_callback(opt);
    g_alpha_topk_override = saved_topk;

    double t_predecode_end = now_ms();

    // Save state (one save, one restore, no per-cycle overhead)
    std::vector<uint8_t> state_backup;
    save_seq_state(ctx, mem, state_backup);
    llama_pos saved_n_past = n_past;

    // Continuous confidence-gated draft
    double t_draft_start = now_ms();

    if (use_mixed) {
        alpha_moe_optimizer_enter_mixed_draft_mode(opt);
    } else {
        alpha_moe_optimizer_enter_draft_mode(opt);
    }

    std::vector<llama_token> draft_tokens;
    llama_token d_cur = first_tok;
    int n_drafted = 0;

    for (int i = 0; i < M_MAX; i++) {
        llama_batch dbatch = llama_batch_get_one(&d_cur, 1);
        if (llama_decode(ctx, dbatch) != 0) break;
        float * dl = llama_get_logits(ctx);
        d_cur = argmax_logits(dl, n_vocab);
        draft_tokens.push_back(d_cur);
        n_drafted++;

        // Confidence gate: check logit margin after M_MIN tokens
        if (i >= M_MIN - 1) {
            float top1 = -1e30f, top2 = -1e30f;
            for (int v = 0; v < n_vocab; v++) {
                if (dl[v] > top1) { top2 = top1; top1 = dl[v]; }
                else if (dl[v] > top2) { top2 = dl[v]; }
            }
            float margin = top1 - top2;
            if (margin < CONF_THRESHOLD) {
                ALPHA_LOG_DBG("[CONF] stop at tok=%d margin=%.2f\n", i, margin);
                break;
            }
        }
    }

    if (use_mixed) alpha_moe_optimizer_exit_mixed_draft_mode(opt);
    else alpha_moe_optimizer_exit_draft_mode(opt);

    double t_draft_end = now_ms();

    // Restore state
    restore_seq_state(ctx, mem, state_backup, saved_n_past);

    // Tier 2: chunked batch verify (k=6 cpu-moe)
    double t_verify_start = now_ms();

    if (n_drafted <= 0) {
        state->n_rounds++;
        state->n_tier2_accepted_total += 1;
        return (int)accepted.size();
    }

    llama_set_eval_callback(ctx, nullptr, nullptr);
    g_alpha_topk_override = 0;

    // Clear cache_expert_ids so verify uses raw argsort IDs (0-255) into the CPU
    // expert tensor, not the LUT-remapped IDs (0-27) from draft mode; otherwise
    // verify indexes the wrong experts and accepts garbage.
    alpha_moe_optimizer_clear_expert_remap(opt);

    llama_pos t2_start = mem ? llama_memory_seq_pos_max(mem, 0) + 1 : saved_n_past;

    // Build verify sequence: [first_tok, d0, d1, ..., d_{n-1}]
    std::vector<llama_token> verify_seq;
    verify_seq.push_back(first_tok);
    for (auto tok : draft_tokens) verify_seq.push_back(tok);

    const int T2_CHUNK = 16;
    int n_verify_total = (int)verify_seq.size();
    std::vector<llama_token> final_accepted;
    final_accepted.push_back(first_tok);  // always kept
    int n_verified = 0;
    bool done = false;
    int offset = 0;

    while (offset < n_verify_total && !done) {
        int chunk = std::min(T2_CHUNK, n_verify_total - offset);
        llama_pos chunk_start = t2_start + offset;

        llama_graph_reserve(ctx, chunk, 1, chunk);
        llama_batch vbatch = llama_batch_init(chunk, 0, 1);
        vbatch.n_tokens = chunk;
        for (int i = 0; i < chunk; i++) {
            vbatch.token[i]     = verify_seq[offset + i];
            vbatch.pos[i]       = chunk_start + i;
            vbatch.n_seq_id[i]  = 1;
            vbatch.seq_id[i][0] = 0;
            vbatch.logits[i]    = 1;
        }

        int rc = llama_decode(ctx, vbatch);
        llama_batch_free(vbatch);

        if (rc != 0) {
            ALPHA_LOG_DBG("[SPEC] verify chunk failed at offset %d (rc=%d)\n", offset, rc);
            done = true;
            break;
        }

        for (int i = 0; i < chunk; i++) {
            int gi = offset + i;
            if (gi >= n_drafted) {
                // Last position: bonus token
                float * bl = llama_get_logits_ith(ctx, i);
                if (bl) {
                    final_accepted.push_back(argmax_logits(bl, n_vocab));
                    n_verified++;
                }
                done = true;
                break;
            }

            float * vl = llama_get_logits_ith(ctx, i);
            if (!vl) { done = true; break; }
            llama_token v = argmax_logits(vl, n_vocab);

            if (v == draft_tokens[gi]) {
                final_accepted.push_back(draft_tokens[gi]);
                n_verified++;
            } else {
                final_accepted.push_back(v);  // substitute
                n_verified++;
                state->n_tier2_rejects++;
                done = true;
                break;
            }
        }

        offset += chunk;
    }

    // Rollback KV
    llama_pos final_n_past = t2_start + (int)final_accepted.size();
    if (mem) {
        llama_memory_seq_rm(mem, 0, final_n_past, -1);
    }

    // Restore cache_expert_ids (spec_lut) for next round's draft phase
    alpha_moe_optimizer_restore_expert_remap(opt);

    alpha_moe_optimizer_reinstall_callback(opt);
    g_alpha_topk_override = saved_topk;
    llama_graph_reserve(ctx, 1, 1, 1);

    double t_verify_end = now_ms();

    // Statistics
    state->n_rounds++;
    state->n_draft_total += n_drafted;
    state->n_tier1_accepted_total += n_drafted;  // no tier 1, draft = tier 1
    state->n_tier2_accepted_total += (int)final_accepted.size();
    state->draft_ms_total += (t_draft_end - t_draft_start);
    state->tier2_ms_total += (t_verify_end - t_verify_start);

    ALPHA_LOG_DBG("[SPEC] round %lld: drafted=%d verified=%d/%d accepted "
            "(predec %.0fms, draft %.0fms, verify %.0fms)\n",
            (long long)state->n_rounds,
            n_drafted, (int)final_accepted.size() - 1, n_drafted,
            t_predecode_end - t_predecode_start,
            t_draft_end - t_draft_start,
            t_verify_end - t_verify_start);

    accepted = final_accepted;
    return (int)accepted.size();
}

void alpha_twotier_print_stats(const alpha_twotier_state * state) {
    if (!state || state->n_rounds == 0) return;

    double total_ms = state->draft_ms_total + state->tier1_ms_total + state->tier2_ms_total;
    double tok_per_sec = total_ms > 0 ? state->n_tier2_accepted_total * 1000.0 / total_ms : 0;
    double draft_match = state->n_draft_total > 0 ?
        100.0 * state->n_tier1_accepted_total / state->n_draft_total : 0;
    double tier1_match = state->n_tier1_accepted_total > 0 ?
        100.0 * (state->n_tier2_accepted_total - state->n_rounds) /
        state->n_tier1_accepted_total : 0;  // subtract first_tok (1 per round)

    ALPHA_LOG_DBG( "\n=== Two-Tier Spec Stats ===\n");
    ALPHA_LOG_DBG( "  Rounds:      %lld\n", (long long)state->n_rounds);
    ALPHA_LOG_DBG( "  Drafted:     %lld tokens (M=%d, N=%d, %d/round)\n",
            (long long)state->n_draft_total, state->n_draft, state->tier1_cycles,
            state->n_draft * state->tier1_cycles);
    ALPHA_LOG_DBG( "  Tier 1:      %lld accepted (%.1f%% draft match)\n",
            (long long)state->n_tier1_accepted_total, draft_match);
    ALPHA_LOG_DBG( "  Tier 1 rej:  %lld\n", (long long)state->n_tier1_rejects);
    ALPHA_LOG_DBG( "  Tier 2:      %lld final accepted (%.1f%% tier1 match)\n",
            (long long)state->n_tier2_accepted_total, tier1_match);
    ALPHA_LOG_DBG( "  Tier 2 rej:  %lld\n", (long long)state->n_tier2_rejects);
    ALPHA_LOG_DBG( "  Draft:       %.0f ms (%.1f ms/tok, %.1f tok/s)\n",
            state->draft_ms_total,
            state->draft_ms_total / std::max((int64_t)1, state->n_draft_total),
            state->n_draft_total * 1000.0 / std::max(1.0, state->draft_ms_total));
    ALPHA_LOG_DBG( "  Tier 1:      %.0f ms (%.1f ms/cycle)\n",
            state->tier1_ms_total,
            state->tier1_ms_total / std::max((int64_t)1,
                (int64_t)(state->n_rounds * state->tier1_cycles)));
    ALPHA_LOG_DBG( "  Tier 2:      %.0f ms (%.1f ms/tok)\n",
            state->tier2_ms_total,
            state->tier2_ms_total / std::max((int64_t)1, state->n_tier2_accepted_total));
    ALPHA_LOG_DBG( "  Throughput:  %.1f tok/s (effective)\n", tok_per_sec);
    ALPHA_LOG_DBG( "\n");
}

// LUT-direct mode
// GPU LUT + adaptive renorm, no callback, no verification.
// After each decode, reload cache from argsort for next token.
// Equivalent to permanent draft mode with per-token cache refresh.

// Confidence margin from last LUT decode (top1 - top2 logit)
static float g_lut_last_margin = 0.0f;
float alpha_lut_direct_get_margin() { return g_lut_last_margin; }

llama_token alpha_lut_direct_decode(
    alpha_moe_optimizer * opt,
    struct llama_context * ctx,
    const struct llama_vocab * vocab,
    llama_token input_token,
    struct llama_sampler * smpl) {

    if (!opt || !ctx) return LLAMA_TOKEN_NULL;

    llama_batch batch = llama_batch_get_one(&input_token, 1);
    if (llama_decode(ctx, batch) != 0) return LLAMA_TOKEN_NULL;

    // When frequency profiling is active, call record() to populate expert_activation_count.
    // This adds argsort readback overhead but is only used during --profile-frequency runs.
    if (opt->config.enable_profiling) {
        alpha_moe_optimizer_record(opt);
    }

    // Compute margin from raw logits (confidence signal)
    float * logits = llama_get_logits(ctx);
    if (!logits) return LLAMA_TOKEN_NULL;

    int n_vocab = llama_vocab_n_tokens(vocab);
    llama_token best_tok = 0;
    float best1 = logits[0], best2 = -1e30f;
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > best1) { best2 = best1; best1 = logits[v]; best_tok = v; }
        else if (logits[v] > best2) { best2 = logits[v]; }
    }
    g_lut_last_margin = best1 - best2;

    // Cache refresh: without profiles, always refresh (no hot set).
    // With profiles, the cache is static; refresh is triggered externally on degradation.
    if (!alpha_moe_optimizer_has_profiles()) {
        alpha_moe_optimizer_reload_from_argsort(opt);
    }

    // Sample: use sampler if provided (temperature support), else argmax (fast)
    if (smpl) {
        return llama_sampler_sample(smpl, ctx, -1);
    }
    return best_tok;
}

// Recurrent-state window bracket (window patch-correction decode)
bool alpha_recurrent_snapshot(struct llama_context * ctx, alpha_recurrent_snapshot_t & snap) {
    snap.valid = false;
    llama_memory_t mem = llama_get_memory(ctx);
    auto * hybrid_mem = dynamic_cast<llama_memory_hybrid *>(mem);
    llama_memory_recurrent * recr = hybrid_mem ? hybrid_mem->get_mem_recr() : nullptr;
    if (!recr) { snap.valid = true; return true; }  // non-hybrid: nothing to save

    // One-time: build device-resident mirrors in the same buffer type as the live state, so
    // ggml_backend_tensor_copy takes the device-to-device path (neither side is host).
    if (!snap.mbuf) {
        if (snap.alloc_failed) return false;
        size_t n_t = 0, bytes = 0;
        ggml_backend_buffer_type_t buft = nullptr;
        for (auto * t : recr->r_l) if (t) { n_t++; bytes += ggml_nbytes(t); if (!buft) buft = ggml_backend_buffer_get_type(t->buffer); }
        for (auto * t : recr->s_l) if (t) { n_t++; bytes += ggml_nbytes(t); if (!buft) buft = ggml_backend_buffer_get_type(t->buffer); }
        if (!buft || n_t == 0) { snap.valid = true; return true; }
        ggml_init_params ip = { ggml_tensor_overhead() * (n_t + 8), nullptr, /*no_alloc*/ true };
        snap.mctx = ggml_init(ip);
        if (!snap.mctx) return false;
        snap.mr.clear(); snap.ms.clear();
        for (auto * t : recr->r_l) snap.mr.push_back(t ? ggml_dup_tensor(snap.mctx, t) : nullptr);
        for (auto * t : recr->s_l) snap.ms.push_back(t ? ggml_dup_tensor(snap.mctx, t) : nullptr);
        snap.mbuf = ggml_backend_alloc_ctx_tensors_from_buft(snap.mctx, buft);
        if (!snap.mbuf) {
            ggml_free(snap.mctx); snap.mctx = nullptr;
            snap.alloc_failed = true;
            fprintf(stderr, "[mtp-snap] recurrent-state mirror (%.1f MB) does not fit in VRAM -- snapshots disabled\n", bytes / 1048576.0);
            return false;
        }
        if (getenv("ALPHA_MTP_DBG")) {
            fprintf(stderr, "[mtp-snap] device mirror: %zu tensors, %.1f MB (D2D, was host round-trip)\n",
                    n_t, bytes / (1024.0 * 1024.0));
        }
    }
    for (size_t i = 0; i < snap.mr.size() && i < recr->r_l.size(); i++)
        if (recr->r_l[i] && snap.mr[i]) ggml_backend_tensor_copy(recr->r_l[i], snap.mr[i]);
    for (size_t i = 0; i < snap.ms.size() && i < recr->s_l.size(); i++)
        if (recr->s_l[i] && snap.ms[i]) ggml_backend_tensor_copy(recr->s_l[i], snap.ms[i]);
    snap.valid = true;
    return true;
}

// ALPHA_MTP / ReplaySSM: rewind only the recurrent cell bookkeeping, without touching the state
// tensors. ReplaySSM has already folded the state forward to `reset_pos - 1` by re-invoking the
// gated-delta op, so the tensors are correct and must not be overwritten, but the cells still
// carry the position of the last verified row, and llama_memory_recurrent::seq_rm refuses any
// partial removal that includes the final position. llama_memory_hybrid::seq_rm tries the
// recurrent half first and returns early on that refusal, so the attention tail would never be
// removed either and the rejected tokens' KV would survive. Moving the cells back first makes the
// subsequent seq_rm legal and lets it reach the attention cache.
bool alpha_recurrent_mark_pos(struct llama_context * ctx, llama_seq_id seq_id, llama_pos reset_pos) {
    llama_memory_t mem = llama_get_memory(ctx);
    auto * hybrid_mem = dynamic_cast<llama_memory_hybrid *>(mem);
    llama_memory_recurrent * recr = hybrid_mem ? hybrid_mem->get_mem_recr() : nullptr;
    if (!recr) return true;   // non-hybrid: no recurrent bookkeeping to rewind
    for (uint32_t i = 0; i < recr->size; i++) {
        if (recr->cells[i].has_seq_id(seq_id) && recr->cells[i].pos >= reset_pos) {
            recr->cells[i].pos = reset_pos - 1;
        }
    }
    return true;
}

bool alpha_mtp_seq_save(struct llama_context * ctx, llama_seq_id seq, llama_pos pos, std::vector<float> & h) {
    return ctx && ctx->mtp_seq_export((int32_t) seq, (int32_t) pos, h);
}

bool alpha_mtp_seq_restore(struct llama_context * ctx, llama_seq_id seq, llama_pos ckpt, const std::vector<float> & h) {
    return ctx && ctx->mtp_seq_import((int32_t) seq, (int32_t) ckpt, h);
}

uint64_t alpha_mtp_seq_kv_hash(struct llama_context * ctx, llama_seq_id seq, int32_t & lo, int32_t & hi) {
    lo = hi = -1;
    return ctx ? ctx->mtp_seq_kv_hash((int32_t) seq, lo, hi) : 0;
}

bool alpha_recurrent_restore(struct llama_context * ctx, const alpha_recurrent_snapshot_t & snap, llama_pos reset_pos) {
    if (!snap.valid) return false;
    llama_memory_t mem = llama_get_memory(ctx);
    auto * hybrid_mem = dynamic_cast<llama_memory_hybrid *>(mem);
    llama_memory_recurrent * recr = hybrid_mem ? hybrid_mem->get_mem_recr() : nullptr;
    if (!recr) return true;  // non-hybrid: nothing to restore
    for (size_t i = 0; i < snap.mr.size() && i < recr->r_l.size(); i++)
        if (recr->r_l[i] && snap.mr[i]) ggml_backend_tensor_copy(snap.mr[i], recr->r_l[i]);
    for (size_t i = 0; i < snap.ms.size() && i < recr->s_l.size(); i++)
        if (recr->s_l[i] && snap.ms[i]) ggml_backend_tensor_copy(snap.ms[i], recr->s_l[i]);
    // Reset recurrent cell tail so seq_rm(reset_pos..) succeeds
    for (uint32_t i = 0; i < recr->size; i++) {
        if (recr->cells[i].has_seq_id(0) && recr->cells[i].pos >= reset_pos) {
            recr->cells[i].pos = reset_pos - 1;
        }
    }
    return true;
}
