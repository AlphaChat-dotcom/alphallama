// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaServe: Async Pipeline Server Library
//
// Architecture (CUDA-safe):
//   Main thread:      GPU decode loop (llama_decode must stay on context-creating thread)
//   Scheduler thread: batch building, stop detection, request admission (pure CPU)
//   HTTP thread:      accepts connections, pushes requests (future)
//
// Usage:
//   alphaserve::server srv;
//   srv.init(ctx, n_slots);
//   srv.start();               // starts scheduler thread
//   srv.submit(request);       // from any thread
//   while (running) {
//       srv.step();            // GPU decode on main thread
//       srv.poll_event(event); // collect results
//   }
//   srv.stop();

#include <functional>
#include "serve-types.h"
#include "serve-spsc-queue.h"
#include "serve-scheduler.h"
#include "serve-gpu-engine.h"
#include "llama.h"

#include <atomic>

namespace alphaserve {

class server {
public:
    server() = default;
    ~server() { stop(); }

    // Initialize with model context
    void init(llama_context * ctx, int n_slots);

    // Start scheduler thread (GPU stays on main thread)
    void start();

    // Stop scheduler thread
    void stop();

    // Submit a new inference request (thread-safe: pushes to SPSC queue)
    bool submit(InferenceRequest req);

    // Cancel an in-flight/queued request (thread-safe) - client disconnected or the
    // handler gave up; frees the slot instead of decoding for nobody
    void cancel(int request_id) { scheduler_.cancel(request_id); }

    // Poll for completed token events (called from main/output thread)
    bool poll_event(TokenEvent & event);

    // Drive GPU decode from main thread - call in a loop
    // Returns true if a batch was processed
    bool step();
    // see gpu_engine::pre_batch_hook
    void set_pre_batch_hook(std::function<void()> f) { gpu_engine_.pre_batch_hook = std::move(f); }

    // Run one tick fully synchronously (scheduler + GPU on caller thread)
    // For testing and single-thread mode
    void tick_sync();

    // Initialize scheduler only (no ggml gpu_engine) - for FlashInfer backend
    void init_scheduler_only(const llama_vocab * vocab, int n_slots);

    // Access queues for pluggable GPU engines
    spsc_queue<BatchRequest, 64> & gpu_in_queue() { return scheduler_to_gpu_; }
    spsc_queue<BatchResult, 64> & gpu_out_queue() { return gpu_to_scheduler_; }

    // Status
    bool is_running() const { return running_.load(); }
    int active_slots() const { return scheduler_.active_slots(); }
    void set_prefix_caching(bool enabled) { scheduler_.prefix_caching_enabled = enabled; }
    void disable_slot_prefix_match() { scheduler_.disable_slot_prefix_match(); }
    void set_slot_affinity(bool on) { scheduler_.slot_affinity = on; }
    // ALPHA_HYBRID_PREFIX: checkpoint-based prefix reuse for hybrid/recurrent models
    void enable_hybrid_snapshot() { scheduler_.enable_hybrid_snapshot(); }

private:
    std::atomic<bool> running_{false};

    // Context
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    int n_slots_ = 0;

    // SPSC queues (owned by server, referenced by threads)
    spsc_queue<InferenceRequest, 256> http_to_scheduler_;
    spsc_queue<BatchRequest, 64>     scheduler_to_gpu_;
    spsc_queue<BatchResult, 64>      gpu_to_scheduler_;
    spsc_queue<TokenEvent, 4096>     scheduler_to_output_;

    // Thread objects
    scheduler   scheduler_;
    gpu_engine  gpu_engine_;
};

} // namespace alphaserve
