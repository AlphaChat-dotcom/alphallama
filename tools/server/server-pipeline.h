#pragma once

// Async pipeline server for multi-user throughput
//
// Two modes:
//   standard: synchronous update_slots(), lowest single-user latency
//   pipeline: async multi-thread pipeline, higher multi-user throughput
//
// Pipeline threads:
//   scheduler: admits requests, builds batch N+1 while the GPU executes N
//   GPU:       llama_decode + backend sampling (returns token IDs, not logits)
//   output:    processes sampled tokens, streams to clients
//
// Backend sampling avoids the device-to-host logits copy (one token ID per slot instead of a
// full logits row), runs the temp/top-k/top-p chain on the GPU, and avoids llama_synchronize.
//
// API: llama_set_sampler(ctx, seq_id, sampler_chain)
//      llama_get_sampled_token_ith(ctx, i) -> pre-sampled token

#include "server-common.h"
#include "llama.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <functional>

struct server_context_impl;  // forward declaration

struct pipeline_batch_buffer {
    llama_batch batch = {};
    std::vector<int> slot_indices;       // which slots are in this batch
    std::vector<llama_token> sampled;    // GPU-sampled token IDs
    std::atomic<bool> ready{false};      // scheduler -> GPU: batch is prepared
    std::atomic<bool> decoded{false};    // GPU -> scheduler: decode complete
    int n_generating = 0;               // number of generating slots in batch
};

class pipeline_server {
public:
    pipeline_server() = default;
    ~pipeline_server() { stop(); }

    // Initialize with context and start threads
    void start(server_context_impl & ctx_server, llama_context * ctx, int n_slots);

    // Stop all threads gracefully
    void stop();

    bool is_running() const { return running.load(); }

private:
    std::atomic<bool> running{false};

    // Double-buffered batch metadata
    pipeline_batch_buffer buffers[2];
    std::atomic<int> gpu_buf{0};     // GPU is working on this
    std::atomic<int> prep_buf{1};    // Scheduler is preparing this

    // Synchronization
    std::mutex mtx;
    std::condition_variable cv_batch_ready;
    std::condition_variable cv_decode_done;

    // Threads
    std::thread scheduler_thread;
    std::thread gpu_thread;

    // Thread entry points
    void scheduler_loop(server_context_impl & ctx_server, llama_context * ctx);
    void gpu_loop(llama_context * ctx);

    // Backend sampler chains (one per slot, owned by this struct)
    std::vector<llama_sampler *> backend_samplers;
};

// end of server-pipeline.h
