#pragma once

// RadixAttention: Token-Level Prefix Tree for KV Cache
//
// Replaces hash-based prompt caching with a radix tree that matches
// at token granularity. Benefits:
//   - Any prefix length matches (not block-aligned)
//   - Multi-level sharing: system prompt + retrieval results + user query
//   - Cross-user sharing: different users with same system prompt share KV
//   - O(prompt_length) lookup regardless of cache size
//
// Architecture:
//   Root -> [system_prompt tokens] -> [tool_defs tokens] -> [redis_results] -> [user_query]
//                                                            |
//                                                    [different_redis_results] -> [user_query_2]
//
// Each node in the tree stores a reference to a KV cache block.
// When a new request arrives, walk the tree to find the longest matching prefix.
// Only the unmatched suffix needs prefill; the rest is already in KV cache.

#include "server-common.h"
#include "llama.h"

#include <unordered_map>
#include <vector>
#include <list>
#include <cstdint>
#include <mutex>

struct radix_node {
    // Children: token_id -> child node
    std::unordered_map<llama_token, radix_node *> children;

    // KV cache state at this node (serialized via llama_state_seq_get_data)
    std::vector<uint8_t> kv_data;
    int n_tokens = 0;       // depth from root = number of tokens from start

    // LRU + reference counting
    int ref_count = 0;      // active requests using this node's KV
    int64_t last_access = 0;
    int64_t created_at = 0;

    // Memory tracking
    size_t kv_size() const { return kv_data.size(); }
    bool has_kv() const { return !kv_data.empty(); }

    ~radix_node() {
        for (auto & [tok, child] : children) {
            delete child;
        }
    }
};

struct radix_match_result {
    radix_node * node = nullptr;     // deepest matching node
    int matched_tokens = 0;          // how many tokens matched from start
    bool has_kv = false;             // whether the matched node has KV data
};

class radix_cache {
public:
    radix_cache() = default;
    ~radix_cache() = default;  // root's destructor recursively frees children

    // Initialize with memory budget
    void init(size_t max_memory_bytes) {
        budget_bytes = max_memory_bytes;
    }

    // Find longest cached prefix for a token sequence
    radix_match_result find_prefix(const llama_token * tokens, int n_tokens) {
        std::lock_guard<std::mutex> lock(mtx);

        radix_match_result result;
        radix_node * current = &root;
        radix_node * last_with_kv = nullptr;
        int last_kv_depth = 0;

        for (int i = 0; i < n_tokens; i++) {
            auto it = current->children.find(tokens[i]);
            if (it == current->children.end()) break;

            current = it->second;
            current->last_access = ++clock;

            if (current->has_kv()) {
                last_with_kv = current;
                last_kv_depth = i + 1;
            }
        }

        if (last_with_kv) {
            result.node = last_with_kv;
            result.matched_tokens = last_kv_depth;
            result.has_kv = true;
            last_with_kv->ref_count++;
        }

        return result;
    }

    // Insert KV state for a token sequence into the tree
    // kv_data: serialized KV state (from llama_state_seq_get_data)
    // tokens: the full token sequence up to this point
    // n_tokens: number of tokens
    // save_interval: save KV every N tokens (not every node, which is too expensive)
    void insert(const llama_token * tokens, int n_tokens,
                const uint8_t * kv_data, size_t kv_size,
                int save_at_depth = -1) {
        std::lock_guard<std::mutex> lock(mtx);

        radix_node * current = &root;

        for (int i = 0; i < n_tokens; i++) {
            auto it = current->children.find(tokens[i]);
            if (it == current->children.end()) {
                auto * child = new radix_node();
                child->n_tokens = i + 1;
                child->created_at = ++clock;
                child->last_access = clock;
                current->children[tokens[i]] = child;
                current = child;
                nodes_total++;
            } else {
                current = it->second;
                current->last_access = ++clock;
            }
        }

        // Save KV at the leaf (or at specified depth)
        if (kv_data && kv_size > 0 && (save_at_depth < 0 || n_tokens == save_at_depth)) {
            // Evict if over budget
            while (used_bytes + kv_size > budget_bytes && nodes_with_kv > 0) {
                evict_lru();
            }

            current->kv_data.assign(kv_data, kv_data + kv_size);
            used_bytes += kv_size;
            nodes_with_kv++;
        }
    }

    // Release reference to a matched node (call when request completes)
    void release(radix_node * node) {
        if (!node) return;
        std::lock_guard<std::mutex> lock(mtx);
        if (node->ref_count > 0) node->ref_count--;
    }

    // Stats
    size_t memory_used() const { return used_bytes; }
    size_t memory_budget() const { return budget_bytes; }
    int total_nodes() const { return nodes_total; }
    int kv_nodes() const { return nodes_with_kv; }
    int64_t total_hits() const { return cache_hits; }
    int64_t total_misses() const { return cache_misses; }

    void record_hit() { cache_hits++; }
    void record_miss() { cache_misses++; }

private:
    radix_node root;
    std::mutex mtx;
    int64_t clock = 0;

    size_t budget_bytes = 0;
    size_t used_bytes = 0;
    int nodes_total = 0;
    int nodes_with_kv = 0;
    int64_t cache_hits = 0;
    int64_t cache_misses = 0;

    // Evict least-recently-used leaf node with KV data and ref_count=0
    void evict_lru() {
        radix_node * victim = nullptr;
        int64_t oldest = INT64_MAX;

        find_lru_leaf(&root, victim, oldest);

        if (victim && victim->has_kv()) {
            used_bytes -= victim->kv_size();
            victim->kv_data.clear();
            victim->kv_data.shrink_to_fit();
            nodes_with_kv--;
        }
    }

    void find_lru_leaf(radix_node * node, radix_node *& victim, int64_t & oldest) {
        if (node->children.empty() && node->has_kv() && node->ref_count == 0) {
            if (node->last_access < oldest) {
                oldest = node->last_access;
                victim = node;
            }
            return;
        }
        for (auto & [tok, child] : node->children) {
            find_lru_leaf(child, victim, oldest);
        }
    }
};

// end of server-radix-cache.h
