// SPDX-License-Identifier: Apache-2.0
#ifdef ALPHA_HAS_FLASHINFER

#include "serve-fi-engine.h"
#include "common.h"

#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <chrono>

extern "C" void launch_batch_argmax(int32_t * out_ids, const float * logits,
                                     int batch_size, int vocab_size, cudaStream_t stream);

namespace alphaserve {

fi_gpu_engine::~fi_gpu_engine() {
    if (d_tokens_) cudaFree(d_tokens_);
    if (d_positions_) cudaFree(d_positions_);
    if (d_logits_) cudaFree(d_logits_);
    if (d_sampled_ids_) cudaFree(d_sampled_ids_);
    if (h_tokens_pinned_) cudaFreeHost(h_tokens_pinned_);
    if (h_positions_pinned_) cudaFreeHost(h_positions_pinned_);
    if (h_sampled_pinned_) cudaFreeHost(h_sampled_pinned_);
    if (done_event_) cudaEventDestroy(done_event_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool fi_gpu_engine::init(ForwardEngine * engine,
                          PagedKVPool * kv_pool,
                          const llama_vocab * vocab,
                          spsc_queue<BatchRequest, 64> & in_queue,
                          spsc_queue<BatchResult, 64> & out_queue,
                          int max_batch) {
    engine_ = engine;
    kv_pool_ = kv_pool;
    vocab_ = vocab;
    in_ = &in_queue;
    out_ = &out_queue;
    max_batch_ = max_batch;

    int vocab_size = engine->config().vocab_size;
    {
        size_t free_b = 0, total_b = 0; cudaMemGetInfo(&free_b, &total_b);
        fprintf(stderr, "[fi-engine] init: max_batch=%d vocab=%d — free=%.0f MiB, d_logits needs %.0f MiB\n",
                max_batch, vocab_size, free_b / 1048576.0,
                (double)max_batch * vocab_size * 4 / 1048576.0);
    }
    cudaError_t err;
    err = cudaMalloc(&d_tokens_, max_batch * sizeof(int32_t));
    if (err) goto fail;
    err = cudaMalloc(&d_positions_, max_batch * sizeof(int32_t));
    if (err) goto fail;
    err = cudaMalloc(&d_logits_, (size_t)max_batch * vocab_size * sizeof(float));
    if (err) goto fail;
    err = cudaMalloc(&d_sampled_ids_, max_batch * sizeof(int32_t));
    if (err) goto fail;

    // Pinned host buffers for fast async H2D
    err = cudaHostAlloc(&h_tokens_pinned_, max_batch * sizeof(int32_t), cudaHostAllocDefault);
    if (err) goto fail;
    err = cudaHostAlloc(&h_positions_pinned_, max_batch * sizeof(int32_t), cudaHostAllocDefault);
    if (err) goto fail;
    // Pinned output buffer: GPU writes sampled IDs directly to host memory
    err = cudaHostAlloc(&h_sampled_pinned_, max_batch * sizeof(int32_t), cudaHostAllocDefault);
    if (err) goto fail;

    cudaStreamCreate(&stream_);
    cudaEventCreateWithFlags(&done_event_, cudaEventDisableTiming);

    fprintf(stderr, "[fi-engine] initialized: max_batch=%d, vocab=%d\n", max_batch, vocab_size);
    return true;

fail:
    fprintf(stderr, "[fi-engine] buffer allocation failed: %s\n", cudaGetErrorString(err));
    return false;
}

bool fi_gpu_engine::process_one() {
    BatchRequest req;
    if (!in_->try_pop(req)) return false;
    if (req.n_tokens == 0) return false;

    // Per-step host/GPU breakdown (ALPHA_FI_TIMING) + hot-path log gate (ALPHA_FI_VERBOSE)
    static const bool g_fi_timing  = getenv("ALPHA_FI_TIMING")  != nullptr;
    static const bool g_fi_verbose = getenv("ALPHA_FI_VERBOSE") != nullptr;
    std::chrono::steady_clock::time_point _t0, _t1, _t2, _tL;
    if (g_fi_timing) { _t0 = std::chrono::steady_clock::now(); _tL = _t0; }

    cudaGetLastError();  // clear stale errors

    // Hybrid-prefix checkpoint directives are executed only by the ggml GPU engine. The scheduler
    // assumes they ran, so ignoring them here would resume a slot from a recurrent state that was
    // never restored; abort instead.
    if (!req.restore_seqs.empty() || !req.snapshot_seqs.empty()) {
        GGML_ABORT("[hybrid-prefix] restore/snapshot directives reached the FlashInfer engine, which does "
                   "not implement them (ALPHA_HYBRID_PREFIX is ggml-backend only)");
    }

    // Clear KV for reused sequences
    for (int32_t seq_id : req.clear_seqs) {
        kv_pool_->clear_seq(seq_id);
    }

    // Prefix cache: truncate KV for slots with cached prefix.
    // Must happen before the prefill/decode split because a 1-token suffix
    // goes through the decode path, which expects correct n_tokens.
    for (int i = 0; i < req.n_tokens; i++) {
        int32_t seq_id = req.seq_ids[i];
        int cached_n = kv_pool_->sequences[seq_id].n_tokens;
        int first_pos = req.positions[i];
        // Detect prefix-cached slot: KV exists and first position > 0
        if (cached_n > 0 && first_pos > 0 && cached_n > first_pos) {
            // Truncate to prefix boundary (discard generated tokens from previous request)
            kv_pool_->truncate_seq(seq_id, first_pos);
            fprintf(stderr, "[fi-engine] KV truncated: seq=%d %d->%d tokens\n",
                    seq_id, cached_n, first_pos);
        }
    }

    // Separate prefill tokens from decode tokens.
    // Prefill: multiple tokens for a slot, only last has logits=true
    // Decode:  one token per slot, always logits=true
    //
    // Group tokens by slot_id to identify prefill vs decode:
    // - Slot with >1 token in batch = prefilling
    // - Slot with 1 token in batch = decoding (or last prefill chunk)

    struct SlotTokens {
        std::vector<int> batch_indices;  // indices into req.tokens
    };
    std::unordered_map<int, SlotTokens> slot_groups;

    for (int i = 0; i < req.n_tokens; i++) {
        slot_groups[req.slot_ids[i]].batch_indices.push_back(i);
    }

    // Phase 1: batched prefill.
    // Instead of processing each slot sequentially (N_slots x N_tokens serial calls),
    // batch across slots at each position step (N_tokens batched calls of size N_slots).
    // For 4 slots x 6 tokens: 6 rounds of batch_size=4 instead of 24 rounds of batch_size=1.
    struct PrefillResult {
        int slot_id;
        int batch_idx;
        std::vector<float> logits;
    };
    std::vector<PrefillResult> prefill_results;
    int vocab_size = engine_->config().vocab_size;

    // Collect prefill slots (those with >1 token in this batch)
    struct PrefillSlot {
        int slot_id;
        std::vector<int> batch_indices;
    };
    std::vector<PrefillSlot> prefill_slots;
    for (auto & [slot_id, st] : slot_groups) {
        if ((int)st.batch_indices.size() <= 1) continue;
        prefill_slots.push_back({slot_id, st.batch_indices});
    }

    if (!prefill_slots.empty()) {
        // Find max prefill length across all slots
        int max_prefill_len = 0;
        for (auto & ps : prefill_slots) {
            max_prefill_len = std::max(max_prefill_len, (int)ps.batch_indices.size());
        }

        int n_prefill_slots = (int)prefill_slots.size();

        // Batched prefill with Redis KV cache check.
        for (auto & ps : prefill_slots) {
            int n_tokens = (int)ps.batch_indices.size();
            std::vector<int32_t> h_tokens(n_tokens), h_positions(n_tokens);
            int seq_id = -1;
            for (int i = 0; i < n_tokens; i++) {
                int idx = ps.batch_indices[i];
                h_tokens[i] = req.tokens[idx];
                h_positions[i] = req.positions[idx];
                seq_id = req.seq_ids[idx];
            }

            // Check Redis KV cache for this exact prompt
            // Cache format: [logits (vocab_size * float)] + [KV data]
            // On hit: restore KV + logits directly, skip prefill entirely.
            // We store logits because forward_decode with restored KV has
            // precision drift (causal vs paged attention kernels), causing
            // output divergence. Storing logits gives bit-identical output.
            bool cache_hit = false;
            if (kv_cache_) {
                auto result = kv_cache_->load(h_tokens.data(), n_tokens);
                size_t logits_bytes = (size_t)vocab_size * sizeof(float);
                if (result.n_tokens > 0 && result.data.size() > logits_bytes) {
                    size_t kv_size = result.data.size() - logits_bytes;
                    const uint8_t * kv_data = result.data.data() + logits_bytes;

                    int restored = kv_pool_->restore_seq_kv(
                        seq_id, kv_data, kv_size,
                        result.n_tokens, stream_);
                    if (restored > 0) {
                        cache_hit = true;
                        cudaMemcpyAsync(d_logits_, result.data.data(), logits_bytes,
                                        cudaMemcpyHostToDevice, stream_);
                    }
                }
            }

            if (!cache_hit) {
                // Prefix already truncated above (before prefill/decode split).
                // Check if remaining KV matches prefix for paged prefill.
                int cached_n = kv_pool_->sequences[seq_id].n_tokens;
                // Recurrent (DeltaNet) models must take the full-prefill path: forward_prefill_cached
                // skips the recurrent layers, so on a prefix-cache hit the recurrent state would not be
                // reconstructed and decode would continue from the previous request's state.
                bool prefix_cached = (cached_n > 0 && h_positions[0] > 0 && !engine_->has_recurrent());

                if (prefix_cached && n_tokens > 0) {
                    // Prefix cache hit: process suffix via paged prefill attention
                    fprintf(stderr, "[fi-engine] prefix prefill: seq=%d cached=%d suffix=%d\n",
                            seq_id, cached_n, n_tokens);
                    cudaMemcpyAsync(d_tokens_, h_tokens.data(), n_tokens * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_positions_, h_positions.data(), n_tokens * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    engine_->forward_prefill_cached(d_tokens_, seq_id, d_positions_,
                                                     d_logits_, n_tokens, cached_n, stream_);
                } else {
                    // Full prefill via causal attention
                    cudaMemcpyAsync(d_tokens_, h_tokens.data(), n_tokens * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    cudaMemcpyAsync(d_positions_, h_positions.data(), n_tokens * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream_);
                    engine_->forward_prefill(d_tokens_, seq_id, d_positions_, d_logits_,
                                              nullptr, n_tokens, stream_);
                }

                // Save KV + logits to Redis
                if (kv_cache_ && !prefix_cached) {
                    cudaStreamSynchronize(stream_);
                    auto kv_data = kv_pool_->serialize_seq_kv(seq_id, stream_);
                    if (!kv_data.empty()) {
                        size_t logits_bytes = (size_t)vocab_size * sizeof(float);
                        std::vector<uint8_t> combined(logits_bytes + kv_data.size());
                        cudaMemcpy(combined.data(), d_logits_, logits_bytes,
                                   cudaMemcpyDeviceToHost);
                        memcpy(combined.data() + logits_bytes, kv_data.data(), kv_data.size());
                        kv_cache_->save(h_tokens.data(), n_tokens,
                                        combined.data(), combined.size());
                    }
                }
            }

            int last_idx = ps.batch_indices.back();
            if (req.logits_mask[last_idx]) {
                cudaStreamSynchronize(stream_);
                PrefillResult pr;
                pr.slot_id = ps.slot_id;
                pr.batch_idx = last_idx;
                pr.logits.resize(vocab_size);
                cudaMemcpy(pr.logits.data(), d_logits_,
                           vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
                prefill_results.push_back(std::move(pr));
            }
        }
    }

    // Phase 2: batch all decode tokens (1 per slot).
    std::vector<int> decode_indices;
    for (auto & [slot_id, st] : slot_groups) {
        if ((int)st.batch_indices.size() == 1) {
            decode_indices.push_back(st.batch_indices[0]);
        }
    }

    // Also add the last token of each prefill group (it has logits=true and
    // was already processed in phase 1; its logits are in d_logits_[0]).
    // It is not re-processed; only its logits are needed for sampling.

    // For decode tokens, run one batched forward
    if (!decode_indices.empty()) {
        int n_decode = (int)decode_indices.size();
        std::vector<int32_t> h_seq_ids(n_decode);

        for (int i = 0; i < n_decode; i++) {
            int idx = decode_indices[i];
            h_tokens_pinned_[i] = req.tokens[idx];
            h_positions_pinned_[i] = req.positions[idx];
            h_seq_ids[i] = req.seq_ids[idx];
        }

        if (g_fi_timing) _tL = std::chrono::steady_clock::now();  // split host-build vs launch
        // Pinned -> GPU async transfers (no staging copy overhead)
        cudaMemcpyAsync(d_tokens_, h_tokens_pinned_, n_decode * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_positions_, h_positions_pinned_, n_decode * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream_);

        // Pass d_sampled_ids_ so argmax runs inside the CUDA graph
        engine_->forward_decode(d_tokens_, h_seq_ids.data(), d_positions_, d_logits_,
                                d_sampled_ids_, n_decode, stream_);
    }

    // Copy sampled IDs to pinned host buffer (async, part of stream)
    if (!decode_indices.empty()) {
        int n_decode = (int)decode_indices.size();
        cudaMemcpyAsync(h_sampled_pinned_, d_sampled_ids_, n_decode * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream_);
    }

    // Wait for GPU completion
    if (g_fi_timing) _t1 = std::chrono::steady_clock::now();
    cudaStreamSynchronize(stream_);
    if (g_fi_timing) _t2 = std::chrono::steady_clock::now();

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        // Auto-tuning benchmark matmuls can leave stale errors; warn but continue
        static int cuda_warn_count = 0;
        if (cuda_warn_count < 5) {
            fprintf(stderr, "[fi-engine] CUDA warning (continuing): %s\n", cudaGetErrorString(cuda_err));
            cuda_warn_count++;
        }
    }

    // Sample from logits
    BatchResult result;
    result.n_slots = 0;

    // For decode tokens: sampled IDs already in pinned host memory (no D2H copy needed)
    if (!decode_indices.empty()) {
        int n_decode = (int)decode_indices.size();
        for (int i = 0; i < n_decode; i++) {
            int idx = decode_indices[i];
            if (!req.logits_mask[idx]) continue;

            int sid = req.slot_ids[idx];
            result.sampled.push_back(h_sampled_pinned_[i]);
            result.slot_ids.push_back(sid);
            result.n_slots++;
            // Over-subscription: begin_token (run inside the forward above) flags a sequence that
            // couldn't get a KV page. Report it so the scheduler force-finishes and frees its pages.
            if (kv_pool_ && kv_pool_->is_exhausted(sid)) result.exhausted_slots.push_back(sid);
        }
    }

    // Sample from saved prefill logits (already on CPU from Phase 1)
    // These are small (1 per slot), so CPU argmax is fine here
    for (auto & pr : prefill_results) {
        const float * logits = pr.logits.data();
        int best_tok = 0;
        float best_val = logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_tok = v;
            }
        }
        result.sampled.push_back(best_tok);
        result.slot_ids.push_back(pr.slot_id);
        result.n_slots++;
    }

    if (g_fi_verbose) {
        fprintf(stderr, "[fi-engine] batch: %d tok, %d prefill_results, %d decode, result: %d slots [",
                req.n_tokens, (int)prefill_results.size(), (int)decode_indices.size(), result.n_slots);
        for (int i = 0; i < result.n_slots; i++) {
            fprintf(stderr, "%s%d", i ? "," : "", result.slot_ids[i]);
        }
        fprintf(stderr, "]\n");
    }

    // Host vs GPU per-step breakdown for pure-decode steps
    if (g_fi_timing && !decode_indices.empty()) {
        auto _t3 = std::chrono::steady_clock::now();
        auto us = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() / 1000.0; };
        static double acc_build = 0, acc_launch = 0, acc_sync = 0, acc_post = 0; static long acc_n = 0;
        acc_build += us(_t0, _tL); acc_launch += us(_tL, _t1); acc_sync += us(_t1, _t2); acc_post += us(_t2, _t3); acc_n++;
        if ((acc_n % 512) == 0) {
            double prep = acc_build + acc_launch;
            fprintf(stderr, "[fi-timing] %ld steps avg: build=%.1fus launch=%.1fus gpu-sync=%.1fus post=%.1fus | HOST=%.1fus GPU=%.1fus (host%%=%.0f)\n",
                    acc_n, acc_build/acc_n, acc_launch/acc_n, acc_sync/acc_n, acc_post/acc_n,
                    (prep+acc_post)/acc_n, acc_sync/acc_n,
                    100.0*(prep+acc_post)/(prep+acc_sync+acc_post));
        }
    }

    out_->try_push(std::move(result));
    n_decodes_++;
    n_tokens_ += req.n_tokens;
    return true;
}

} // namespace alphaserve

#endif // ALPHA_HAS_FLASHINFER
