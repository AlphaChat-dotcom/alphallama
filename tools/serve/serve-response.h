// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaServe Response Router
//
// Routes TokenEvents from the pipeline to waiting HTTP handlers.
// Each pending request has a condition variable for synchronization.
//
// Thread safety:
//   - HTTP threads register/wait on pending responses
//   - Main thread routes events via route_event()
//   - Mutex protects the map; per-response mutex protects the response data

#include "serve-types.h"

#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <memory>
#include <string>
#include <chrono>

namespace alphaserve {

struct PendingResponse {
    std::mutex mtx;
    std::condition_variable cv;

    // Accumulated output
    std::string text;
    int n_tokens = 0;
    bool done = false;
    bool error = false;   // finished by a decode error (truncated) -> finish_reason "error", not "stop"
    bool length = false;  // finished by max_tokens / the context (truncated by a limit) -> finish_reason "length"

    // Streaming support
    bool stream = false;
    std::queue<std::string> chunks;    // individual token texts for SSE
    bool stop_sent = false;

    // Wait until done (non-streaming)
    void wait() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return done; });
    }

    // Wait for next chunk (streaming); returns false when done
    bool wait_chunk(std::string & chunk) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !chunks.empty() || done; });
        if (!chunks.empty()) {
            chunk = std::move(chunks.front());
            chunks.pop();
            return true;
        }
        return false;  // done, no more chunks
    }

    // Bounded waits: time out on no progress, so a request that is never scheduled (e.g. dropped
    // intake) or whose decode stalls returns instead of blocking its httplib worker thread forever
    // (which, accumulated, exhausts the thread pool and hangs the whole server, /health included).
    // A healthy generation resets the idle timer on every token, so long-but-progressing runs never
    // trip it. The window must exceed worst-case prefill (no token is emitted during prefill).

    // Non-streaming. Returns true if done, false if no new token within the window.
    // Before the first token the slot is prefilling (no token is emitted for the whole prompt,
    // which can take minutes for a very long prompt with offload), which is legitimate long
    // no-progress, so it gets the generous prefill_timeout; after the first token, inter-token
    // gaps use the tight idle_timeout (a real stall).
    bool wait_done_for(std::chrono::milliseconds idle_timeout,
                       std::chrono::milliseconds prefill_timeout) {
        std::unique_lock<std::mutex> lock(mtx);
        int last_n = n_tokens;
        while (!done) {
            const auto budget = (n_tokens == 0) ? prefill_timeout : idle_timeout;
            if (!cv.wait_for(lock, budget,
                             [this, &last_n] { return done || n_tokens != last_n; })) {
                return false;  // no progress within the window
            }
            last_n = n_tokens;
        }
        return true;
    }

    // Like wait_done_for, but wakes every poll_ms to run alive(), the client-disconnect
    // probe (httplib req.is_connection_closed). The idle-timeout semantics are unchanged:
    // only a full idle_timeout with no progress counts as a stall; poll wakeups that see
    // progress reset the window. Returns 1 = done, 0 = idle-timeout, -1 = client gone.
    template <typename AliveCb>
    int wait_done_poll(std::chrono::milliseconds idle_timeout,
                       std::chrono::milliseconds prefill_timeout,
                       std::chrono::milliseconds poll_ms, AliveCb && alive) {
        std::unique_lock<std::mutex> lock(mtx);
        int last_n = n_tokens;
        // Prefill (before the first token) gets the generous prefill_timeout; once tokens
        // flow, inter-token gaps use the tight idle_timeout. Otherwise a long prompt whose
        // prefill exceeds idle_timeout would trip the watchdog.
        auto idle_left = (n_tokens == 0) ? prefill_timeout : idle_timeout;
        auto last_probe = std::chrono::steady_clock::now();
        while (!done) {
            const auto slice = poll_ms < idle_left ? poll_ms : idle_left;
            const bool progressed = cv.wait_for(lock, slice,
                [this, &last_n] { return done || n_tokens != last_n; });
            if (done) return 1;
            if (progressed) { last_n = n_tokens; idle_left = (n_tokens == 0) ? prefill_timeout : idle_timeout; }
            else {
                idle_left -= slice;
                if (idle_left.count() <= 0) return 0;
            }
            // probe the socket every ~poll_ms regardless of progress: a healthy generation
            // always progresses, and that is exactly the case that must notice a dead client
            // (tokens would keep flowing to nobody and leak the slot)
            const auto t = std::chrono::steady_clock::now();
            if (t - last_probe >= poll_ms) {
                last_probe = t;
                lock.unlock();
                const bool ok = alive();
                lock.lock();
                if (!ok) return -1;
            }
        }
        return 1;
    }

    // Streaming. Returns 1 = got chunk, 0 = done (no more), -1 = timed out (no progress).
    int wait_chunk_for(std::string & chunk, std::chrono::milliseconds idle_timeout,
                       std::chrono::milliseconds prefill_timeout) {
        std::unique_lock<std::mutex> lock(mtx);
        // No chunk yet + no token => still prefilling: use the generous prefill budget.
        const auto budget = (chunks.empty() && n_tokens == 0) ? prefill_timeout : idle_timeout;
        if (!cv.wait_for(lock, budget, [this] { return !chunks.empty() || done; })) {
            return -1;  // stalled
        }
        if (!chunks.empty()) {
            chunk = std::move(chunks.front());
            chunks.pop();
            return 1;
        }
        return 0;  // done, no more chunks
    }
};

class response_router {
public:
    // Register a pending response (called from HTTP handler thread)
    std::shared_ptr<PendingResponse> create(int request_id, bool streaming) {
        auto pr = std::make_shared<PendingResponse>();
        pr->stream = streaming;
        std::lock_guard<std::mutex> lock(map_mtx_);
        pending_[request_id] = pr;
        return pr;
    }

    // Route a token event to its pending response (called from main thread)
    void route_event(const TokenEvent & event) {
        std::shared_ptr<PendingResponse> pr;
        {
            std::lock_guard<std::mutex> lock(map_mtx_);
            auto it = pending_.find(event.request_id);
            if (it == pending_.end()) return;
            pr = it->second;
            if (event.is_stop) {
                pending_.erase(it);
            }
        }

        std::lock_guard<std::mutex> lock(pr->mtx);
        pr->text += event.text;
        pr->n_tokens = event.n_generated;
        if (pr->stream) {
            pr->chunks.push(event.text);
        }
        if (event.is_error) pr->error = true;
        if (event.is_length) pr->length = true;
        if (event.is_stop) {
            pr->done = true;
        }
        pr->cv.notify_all();
    }

    // Remove a pending response (cleanup on timeout/error)
    void remove(int request_id) {
        std::lock_guard<std::mutex> lock(map_mtx_);
        pending_.erase(request_id);
    }

    int pending_count() const {
        std::lock_guard<std::mutex> lock(map_mtx_);
        return (int)pending_.size();
    }

private:
    mutable std::mutex map_mtx_;
    std::map<int, std::shared_ptr<PendingResponse>> pending_;
};

} // namespace alphaserve
