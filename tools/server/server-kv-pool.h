#pragma once

// 3-Tier Hierarchical KV Pool for AlphaLlama
//
// Tier 1 (L1): GPU VRAM: active slots, managed by llama.cpp KV cache
// Tier 2 (L2): CPU pinned RAM: evicted slot KV states, ~50us restore
// Tier 3 (L3): Redis persistent: session KV states, ~1-5ms restore
//
// Flow:
//   Slot active -> GPU KV cache (L1)
//   Slot evicted -> serialize to CPU RAM (L2) via llama_state_seq_get_data
//   L2 full -> spill oldest to Redis (L3) with TTL
//   Slot needed -> promote L3->L2->L1 via llama_state_seq_set_data
//
// This extends the existing server_prompt_cache (L2-only) with:
//   1. Session-aware keying (session_id + prompt_hash)
//   2. Redis L3 tier for cross-restart persistence
//   3. Proactive eviction hooks in slot management
//   4. LRU tracking per entry

#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <chrono>

#include "server-common.h"

// Forward declarations
struct redisContext;
struct redisReply;
struct llama_context;

// KV Pool Entry

struct kv_pool_entry {
    std::string key;                    // session_id:prompt_hash
    std::vector<uint8_t> kv_data;       // serialized KV state
    int32_t n_tokens = 0;               // token count when serialized
    int64_t last_access = 0;            // monotonic timestamp for LRU
    size_t size_bytes = 0;              // cached size for budget tracking

    enum tier_t { TIER_CPU, TIER_REDIS } tier = TIER_CPU;
};

// Hierarchical KV Pool

class kv_pool {
public:
    kv_pool() = default;
    ~kv_pool() = default;

    // Initialize with budgets. cpu_budget_mb=0 disables L2, redis=nullptr disables L3.
    void init(size_t cpu_budget_mb, redisContext * redis_ctx = nullptr, int redis_ttl_sec = 3600) {
        cpu_budget_bytes = cpu_budget_mb * 1024ULL * 1024ULL;
        redis = redis_ctx;
        ttl_sec = redis_ttl_sec;
        SRV_INF("kv_pool: initialized, CPU budget = %zu MB, Redis = %s, TTL = %d s\n",
                cpu_budget_mb, redis ? "enabled" : "disabled", ttl_sec);
    }

    bool is_enabled() const { return cpu_budget_bytes > 0; }

    // Save: demote from GPU to L2 (CPU RAM)
    // Called when a slot is being evicted/reassigned.
    // Returns true if successfully saved.
    bool save(const std::string & key, const uint8_t * data, size_t data_size, int32_t n_tokens) {
        if (!is_enabled() || data_size == 0) return false;

        std::lock_guard<std::mutex> lock(mtx);

        // Check if already in CPU cache; update in place
        auto it = cpu_index.find(key);
        if (it != cpu_index.end()) {
            auto & entry = *(it->second);
            cpu_used -= entry.size_bytes;
            entry.kv_data.assign(data, data + data_size);
            entry.size_bytes = data_size;
            entry.n_tokens = n_tokens;
            entry.last_access = now_ms();
            entry.tier = kv_pool_entry::TIER_CPU;
            cpu_used += data_size;
            // Move to back (most recently used)
            cpu_lru.splice(cpu_lru.end(), cpu_lru, it->second);
            SRV_INF("kv_pool: updated L2 entry '%s' (%zu bytes, %d tokens)\n",
                    key.c_str(), data_size, n_tokens);
            return true;
        }

        // Evict oldest entries if over budget
        while (cpu_used + data_size > cpu_budget_bytes && !cpu_lru.empty()) {
            evict_oldest_cpu();
        }

        // Still over budget after evicting everything? Skip.
        if (cpu_used + data_size > cpu_budget_bytes) {
            SRV_WRN("kv_pool: L2 budget exceeded, cannot save '%s' (%zu bytes)\n",
                    key.c_str(), data_size);
            return false;
        }

        // Insert new entry
        kv_pool_entry entry;
        entry.key = key;
        entry.kv_data.assign(data, data + data_size);
        entry.size_bytes = data_size;
        entry.n_tokens = n_tokens;
        entry.last_access = now_ms();
        entry.tier = kv_pool_entry::TIER_CPU;

        cpu_lru.push_back(std::move(entry));
        cpu_index[key] = std::prev(cpu_lru.end());
        cpu_used += data_size;

        SRV_INF("kv_pool: saved to L2 '%s' (%zu bytes, %d tokens, L2 used: %.1f/%.1f MB)\n",
                key.c_str(), data_size, n_tokens,
                cpu_used / (1024.0 * 1024.0), cpu_budget_bytes / (1024.0 * 1024.0));
        return true;
    }

    // Load: promote from L2/L3 to caller's buffer
    // Returns true if found. Populates out_data and out_n_tokens.
    bool load(const std::string & key, std::vector<uint8_t> & out_data, int32_t & out_n_tokens) {
        std::lock_guard<std::mutex> lock(mtx);

        // Try L2 (CPU RAM) first
        auto it = cpu_index.find(key);
        if (it != cpu_index.end()) {
            auto & entry = *(it->second);
            out_data = entry.kv_data;  // copy
            out_n_tokens = entry.n_tokens;
            entry.last_access = now_ms();
            // Move to back (most recently used)
            cpu_lru.splice(cpu_lru.end(), cpu_lru, it->second);
            SRV_INF("kv_pool: L2 hit '%s' (%zu bytes)\n", key.c_str(), out_data.size());
            return true;
        }

        // Try L3 (Redis)
        if (redis) {
            if (load_from_redis(key, out_data, out_n_tokens)) {
                SRV_INF("kv_pool: L3 hit '%s' (%zu bytes), promoting to L2\n",
                        key.c_str(), out_data.size());
                // Promote to L2
                save(key, out_data.data(), out_data.size(), out_n_tokens);
                return true;
            }
        }

        return false;
    }

    // Check if key exists (without loading)
    bool has(const std::string & key) {
        std::lock_guard<std::mutex> lock(mtx);
        return cpu_index.count(key) > 0;
    }

    // Remove a specific entry
    void remove(const std::string & key) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cpu_index.find(key);
        if (it != cpu_index.end()) {
            cpu_used -= it->second->size_bytes;
            cpu_lru.erase(it->second);
            cpu_index.erase(it);
        }
        if (redis) {
            remove_from_redis(key);
        }
    }

    // Stats
    size_t cpu_entries() const { return cpu_index.size(); }
    size_t cpu_bytes_used() const { return cpu_used; }
    size_t cpu_bytes_budget() const { return cpu_budget_bytes; }

    // Generate a key from session ID + prompt hash
    static std::string make_key(const std::string & session_id, uint64_t prompt_hash) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s:%016llx", session_id.c_str(), (unsigned long long)prompt_hash);
        return std::string(buf);
    }

    // Simple hash of token sequence
    static uint64_t hash_tokens(const int32_t * tokens, int n_tokens) {
        // FNV-1a 64-bit
        uint64_t h = 14695981039346656037ULL;
        const uint8_t * p = reinterpret_cast<const uint8_t *>(tokens);
        size_t len = n_tokens * sizeof(int32_t);
        for (size_t i = 0; i < len; i++) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

private:
    std::mutex mtx;

    // L2: CPU RAM cache (LRU ordered)
    std::list<kv_pool_entry> cpu_lru;
    std::unordered_map<std::string, std::list<kv_pool_entry>::iterator> cpu_index;
    size_t cpu_used = 0;
    size_t cpu_budget_bytes = 0;

    // L3: Redis connection (borrowed from server, not owned)
    redisContext * redis = nullptr;
    int ttl_sec = 3600;

    int64_t now_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    // Evict oldest CPU entry -> Redis (if available) or discard
    void evict_oldest_cpu() {
        if (cpu_lru.empty()) return;

        auto & oldest = cpu_lru.front();
        SRV_INF("kv_pool: evicting L2 '%s' (%zu bytes)\n",
                oldest.key.c_str(), oldest.size_bytes);

        // Spill to Redis before discarding
        if (redis && oldest.size_bytes > 0) {
            save_to_redis(oldest.key, oldest.kv_data.data(), oldest.size_bytes, oldest.n_tokens);
        }

        cpu_used -= oldest.size_bytes;
        cpu_index.erase(oldest.key);
        cpu_lru.pop_front();
    }

    // Redis L3 operations

    void save_to_redis(const std::string & key, const uint8_t * data, size_t size, int32_t n_tokens);
    bool load_from_redis(const std::string & key, std::vector<uint8_t> & out_data, int32_t & out_n_tokens);
    void remove_from_redis(const std::string & key);
};

// end of server-kv-pool.h
