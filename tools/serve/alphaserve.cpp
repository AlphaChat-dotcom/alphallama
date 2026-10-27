#include "alphaserve.h"

#include <mutex>

namespace alphaserve {

// UBBoost: prefill chunk size (tokens per prefill llama_decode call). Set by
// serve-main from --ubatch before server::init; must equal the context's n_ubatch.
int g_serve_prompt_chunk = 512;

// UBBoost runtime-grow: --ubatch-prefill (0 = off). See serve-slot.h.
int g_serve_ubatch_prefill = 0;

// Per-slot context budget for the generation length cap (see serve-slot.h).
int g_serve_ctx_per_slot = 0;

// Context n_batch: hard per-decode token cap the scheduler budgets against (see serve-slot.h).
// Set by serve-main from ctx_params.n_batch before the scheduler starts.
int g_serve_n_batch = 512;

void server::init(llama_context * ctx, int n_slots) {
    ctx_ = ctx;
    n_slots_ = n_slots;
    vocab_ = llama_model_get_vocab(llama_get_model(ctx));

    // Initialize components with queue references
    scheduler_.init(n_slots, vocab_,
                    http_to_scheduler_,
                    scheduler_to_gpu_,
                    gpu_to_scheduler_,
                    scheduler_to_output_);

    gpu_engine_.init(ctx,
                     scheduler_to_gpu_,
                     gpu_to_scheduler_);

    fprintf(stderr, "[alphaserve] server initialized: %d slots, vocab=%d\n",
            n_slots, llama_vocab_n_tokens(vocab_));
}

void server::start() {
    if (running_.load()) return;
    running_.store(true);

    // Start scheduler thread (CPU work: batch building, stop detection)
    // GPU engine stays on main thread (CUDA context affinity)
    scheduler_.start();

    fprintf(stderr, "[alphaserve] server started (scheduler thread + main-thread GPU)\n");
}

void server::stop() {
    if (!running_.load()) return;
    running_.store(false);

    // Stop scheduler thread
    scheduler_.stop();
    // GPU engine cleanup (no thread to join; it runs on main)
    gpu_engine_.stop();

    fprintf(stderr, "[alphaserve] server stopped (requests=%lld, tokens=%lld)\n",
            (long long)scheduler_.total_requests(),
            (long long)scheduler_.total_tokens_generated());
}

void server::init_scheduler_only(const llama_vocab * vocab, int n_slots) {
    vocab_ = vocab;
    n_slots_ = n_slots;

    scheduler_.init(n_slots, vocab,
                    http_to_scheduler_,
                    scheduler_to_gpu_,
                    gpu_to_scheduler_,
                    scheduler_to_output_);

    fprintf(stderr, "[alphaserve] scheduler-only init: %d slots, vocab=%d\n",
            n_slots, llama_vocab_n_tokens(vocab));
}

bool server::submit(InferenceRequest req) {
    // http_to_scheduler_ is an SPSC queue, but submit() is called from httplib's thread pool,
    // i.e. multiple producers. Concurrent submits would race the SPSC write index and drop a
    // push, leaving that request unscheduled. The producers are serialized with a mutex; the
    // consumer (scheduler) is still single, so the SPSC invariant holds. submit() is per-request
    // (not per-token), so the lock is not on any hot path.
    static std::mutex submit_mu;
    std::lock_guard<std::mutex> lk(submit_mu);
    return http_to_scheduler_.try_push(std::move(req));
}

bool server::poll_event(TokenEvent & event) {
    return scheduler_to_output_.try_pop(event);
}

bool server::step() {
    // Process one batch from the scheduler queue on the main thread
    // This is where llama_decode() runs; it must be on the CUDA context thread
    return gpu_engine_.process_one();
}

void server::tick_sync() {
    // Single-thread mode: scheduler + GPU on caller's thread
    scheduler_.tick_once();
    gpu_engine_.process_one();
}

} // namespace alphaserve
