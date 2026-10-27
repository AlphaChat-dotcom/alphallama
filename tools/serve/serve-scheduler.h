#pragma once

// AlphaServe Scheduler Thread
//
// Owns all slot state exclusively.
// No other thread reads or writes slot data.
//
// Responsibilities:
//   1. Admit new requests -> assign to idle slots
//   2. Build batches from active slots (prefill + decode)
//   3. Process sampled tokens from GPU -> stop detection
//   4. Push token events to output thread
//   5. Release completed slots -> admit next request
//
// Communication:
//   HTTP -> Scheduler: InferenceRequest (via SPSC queue)
//   Scheduler -> GPU: BatchRequest (via SPSC queue)
//   GPU -> Scheduler: BatchResult (via SPSC queue)
//   Scheduler -> Output: TokenEvent (via SPSC queue)

#include "serve-types.h"
#include "serve-slot.h"
#include "serve-spsc-queue.h"
#include "llama.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <functional>

namespace alphaserve {

class scheduler {
public:
    scheduler() = default;
    ~scheduler() { stop(); }

    // Initialize with queues and model info
    void init(int n_slots,
              const llama_vocab * vocab,
              spsc_queue<InferenceRequest, 256> & request_queue,
              spsc_queue<BatchRequest, 64> & to_gpu,
              spsc_queue<BatchResult, 64> & from_gpu,
              spsc_queue<TokenEvent, 4096> & to_output);

    void start();
    void stop();
    bool is_running() const { return running_.load(); }

    // Cancel a request whose client is gone (disconnect / handler timeout). Thread-safe
    // (called from httplib worker threads); the scheduler thread drains the list each tick
    // and force-finishes the matching slot (or drops the queued request) so the slot and
    // its KV pages are freed instead of decoding to completion for nobody.
    void cancel(int request_id);

    // Run one tick synchronously (for single-thread mode)
    void tick_once() { tick(); }

    // Stats
    int64_t total_requests() const { return n_requests_; }
    int64_t total_tokens_generated() const { return n_tokens_generated_; }
    int active_slots() const;

private:
    void loop();

    // One iteration of the scheduler
    void tick();

    // Sub-phases of tick()
    void process_gpu_results();    // pop BatchResult, update slots, check stops
    void admit_new_requests();     // pop InferenceRequest, assign to idle slots
    void build_and_send_batch();   // build BatchRequest from active slots, push to GPU

    // Slots (owned exclusively by this thread)
    std::vector<serve_slot> slots_;
    const llama_vocab * vocab_ = nullptr;

    // Reasoning budget: the model's <think> / </think> tokens (LLAMA_TOKEN_NULL when either is not one token:
    // the budget is then ignored)
    llama_token think_open_ = LLAMA_TOKEN_NULL, think_close_ = LLAMA_TOKEN_NULL;
    llama_token think_step(serve_slot & s, llama_token tok, bool pending);

    // Queues (not owned, just references)
    spsc_queue<InferenceRequest, 256> * req_in_ = nullptr;
    spsc_queue<BatchRequest, 64> * to_gpu_ = nullptr;
    spsc_queue<BatchResult, 64> * from_gpu_ = nullptr;
    spsc_queue<TokenEvent, 4096> * to_output_ = nullptr;

    // Thread
    std::thread thread_;
    std::atomic<bool> running_{false};

    // Request ID counter
    int next_request_id_ = 0;

    // Stats
    int64_t n_requests_ = 0;
    int64_t n_tokens_generated_ = 0;

    // Batch building state
    int prefill_chunk_size_ = 512;
    bool gpu_busy_ = false;  // true while waiting for BatchResult
    std::vector<int32_t> pending_kv_clears_;  // seq IDs to clear on next batch

    // Waiting queue: requests that arrived when no idle slot was available
    std::vector<InferenceRequest> waiting_queue_;

    // Cancel requests (multi-producer: httplib workers push, scheduler thread drains)
    std::mutex cancel_mtx_;
    std::vector<int> cancel_pending_;
    void process_cancellations();

public:
    // Enable only for FlashInfer backend (ggml can't handle partial prefill)
    bool prefix_caching_enabled = false;

    // Disable slot-level prefix matching (M-RoPE models crash with partial KV reuse)
    void disable_slot_prefix_match() {
        for (auto & slot : slots_) {
            slot.disable_prefix_match = true;
        }
    }

    // ALPHA_HYBRID_PREFIX: prefix reuse for hybrid/recurrent models via recurrent-state checkpoints.
    // The partial-KV-reuse crash comes from llama_memory_hybrid::seq_rm silently doing nothing;
    // restoring a checkpoint and then truncating avoids it.
    bool hybrid_snapshot_ = false;
    uint64_t assign_counter_ = 0;   // feeds ServeSlot::last_assign_seq
    void enable_hybrid_snapshot() {
        hybrid_snapshot_ = true;
        prefix_caching_enabled = true;
        for (auto & slot : slots_) {
            slot.hybrid_snapshot = true;
            slot.disable_prefix_match = false;
        }
    }
};

} // namespace alphaserve
