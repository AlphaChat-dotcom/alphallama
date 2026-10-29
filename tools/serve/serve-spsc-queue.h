// SPDX-License-Identifier: Apache-2.0
#pragma once

// Lock-Free Single-Producer Single-Consumer Ring Buffer
//
// Zero allocation, zero contention. Cache-line aligned head/tail.
// Used for inter-thread communication in the AlphaServe pipeline:
//   Scheduler -> GPU: BatchRequest
//   GPU -> Scheduler: BatchResult
//   Scheduler -> Output: TokenEvent
//
// Producer calls try_push() from one thread only.
// Consumer calls try_pop() from one thread only.
// No mutex, no condition variable, no allocation after construction.

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <new>  // for std::hardware_destructive_interference_size

namespace alphaserve {

// Cache line size (64 bytes on x86/ARM)
#ifdef __cpp_lib_hardware_interference_size
    static constexpr size_t CACHE_LINE = std::hardware_destructive_interference_size;
#else
    static constexpr size_t CACHE_LINE = 64;
#endif

// Wake-up for the serve threads' idle waits. Every successful queue push rings it, so a thread with nothing to do
// waits until the next push (or its timeout) instead of polling with fixed sleeps. Polling cost a CPU core spinning
// while a batch was in flight, and on Windows a sleep of 50-100 us lasts a whole timer tick (1-15.6 ms), so a short
// sleep between tokens added several ms to every decoded token there. Read seen() before checking the queues and pass
// it to wait(): a push in between is never missed.
struct serve_bell {
    std::mutex              m;
    std::condition_variable cv;
    std::atomic<uint64_t>   n{0};

    uint64_t seen() const { return n.load(std::memory_order_acquire); }

    void ring() {
        {
            std::lock_guard<std::mutex> lock(m);
            n.fetch_add(1, std::memory_order_release);
        }
        cv.notify_all();
    }

    void wait(uint64_t seen_before, std::chrono::microseconds timeout) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait_for(lock, timeout, [&] { return n.load(std::memory_order_acquire) != seen_before; });
    }
};

inline serve_bell & g_serve_bell() {
    static serve_bell bell;
    return bell;
}

template<typename T, size_t N = 4096>
class spsc_queue {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");

public:
    spsc_queue() = default;

    // Producer: push item. Returns false if queue is full.
    bool try_push(const T & item) {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t next = (h + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        buffer_[h] = item;
        head_.store(next, std::memory_order_release);
        g_serve_bell().ring();
        return true;
    }

    // Producer: push item (move semantics).
    bool try_push(T && item) {
        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t next = (h + 1) & (N - 1);
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        buffer_[h] = std::move(item);
        head_.store(next, std::memory_order_release);
        g_serve_bell().ring();
        return true;
    }

    // Consumer: pop item. Returns false if queue is empty.
    bool try_pop(T & item) {
        const size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        item = std::move(buffer_[t]);
        tail_.store((t + 1) & (N - 1), std::memory_order_release);
        return true;
    }

    // Check if empty (approximate - may be stale by the time caller acts)
    bool empty() const {
        return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
    }

    // Approximate size (may be stale)
    size_t size() const {
        const size_t h = head_.load(std::memory_order_acquire);
        const size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & (N - 1);
    }

    static constexpr size_t capacity() { return N - 1; }

private:
    // Cache-line aligned to prevent false sharing between producer and consumer
    alignas(CACHE_LINE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE) std::atomic<size_t> tail_{0};
    alignas(CACHE_LINE) std::array<T, N> buffer_;
};

} // namespace alphaserve
