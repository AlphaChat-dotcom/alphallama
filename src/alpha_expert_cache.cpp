// SPDX-License-Identifier: Apache-2.0
#include "alpha_expert_cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

struct cache_slot {
    int     layer_id;
    int     expert_id;
    int64_t last_used;   // step counter for LRU
    bool    occupied;
};

struct alpha_expert_cache {
    alpha_expert_cache_config config;

    // LRU tracking
    std::vector<cache_slot> slots;
    int64_t step_counter;

    // Fast lookup: (layer_id * n_experts + expert_id) → slot index
    std::unordered_map<int, int> lookup;

    // Stats
    int64_t hits;
    int64_t misses;
    int64_t evictions;
};

static int make_key(int layer_id, int expert_id, int n_experts) {
    return layer_id * n_experts + expert_id;
}

alpha_expert_cache * alpha_expert_cache_init(
        alpha_expert_cache_config config) {

    auto * cache = new alpha_expert_cache();
    cache->config = config;
    cache->step_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    cache->evictions = 0;

    // Calculate number of slots that fit in VRAM budget
    int n_slots = 0;
    if (config.expert_size_bytes > 0) {
        n_slots = (int)(config.vram_budget_mb * 1024 * 1024 / config.expert_size_bytes);
    }
    n_slots = std::min(n_slots, ALPHA_EXPERT_CACHE_MAX_SLOTS);
    n_slots = std::max(n_slots, 0);

    cache->slots.resize(n_slots);
    for (int i = 0; i < n_slots; i++) {
        cache->slots[i].layer_id  = -1;
        cache->slots[i].expert_id = -1;
        cache->slots[i].last_used = 0;
        cache->slots[i].occupied  = false;
    }

    int total_experts = config.n_moe_layers * config.n_experts_per_layer;
    float coverage = total_experts > 0 ? (float)n_slots / total_experts * 100.0f : 0;

    fprintf(stderr, "alpha_expert_cache: %d slots (%.0f%% of %d total experts), "
            "expert_size=%zu bytes, vram_budget=%zu MB\n",
            n_slots, coverage, total_experts,
            config.expert_size_bytes, config.vram_budget_mb);

    return cache;
}

bool alpha_expert_cache_ensure(
        alpha_expert_cache * cache,
        int layer_id,
        int expert_id,
        const struct ggml_tensor * cpu_tensor,
        struct ggml_tensor * gpu_tensor) {
    if (!cache || cache->slots.empty()) return false;

    cache->step_counter++;

    int key = make_key(layer_id, expert_id, cache->config.n_experts_per_layer);
    auto it = cache->lookup.find(key);

    if (it != cache->lookup.end()) {
        // CACHE HIT
        cache->slots[it->second].last_used = cache->step_counter;
        cache->hits++;
        return true;
    }

    // CACHE MISS — find slot to evict
    cache->misses++;

    int evict_slot = -1;
    int64_t oldest = INT64_MAX;

    // Find empty or LRU slot
    for (int i = 0; i < (int)cache->slots.size(); i++) {
        if (!cache->slots[i].occupied) {
            evict_slot = i;
            break;
        }
        if (cache->slots[i].last_used < oldest) {
            oldest = cache->slots[i].last_used;
            evict_slot = i;
        }
    }

    if (evict_slot < 0) return false;

    // Evict old entry from lookup
    if (cache->slots[evict_slot].occupied) {
        int old_key = make_key(cache->slots[evict_slot].layer_id,
                               cache->slots[evict_slot].expert_id,
                               cache->config.n_experts_per_layer);
        cache->lookup.erase(old_key);
        cache->evictions++;
    }

    // Copy expert slice from CPU tensor to GPU tensor
    // Expert weights in 3D tensor: [dim0, dim1, n_expert]
    // Expert slice = tensor data at offset expert_id * dim0 * dim1 * element_size
    if (cpu_tensor && gpu_tensor && cpu_tensor->data) {
        size_t expert_offset = (size_t)expert_id * cache->config.expert_size_bytes;
        size_t total_size = ggml_nbytes(cpu_tensor);

        if (expert_offset + cache->config.expert_size_bytes <= total_size) {
            // Copy the expert slice from CPU to GPU
            // Note: this is a synchronous copy — async version would use
            // ggml_backend_tensor_copy_async with separate streams
            const char * src = (const char *)cpu_tensor->data + expert_offset;
            ggml_backend_tensor_set(gpu_tensor, src, expert_offset, cache->config.expert_size_bytes);
        }
    }

    // Update slot
    cache->slots[evict_slot].layer_id  = layer_id;
    cache->slots[evict_slot].expert_id = expert_id;
    cache->slots[evict_slot].last_used = cache->step_counter;
    cache->slots[evict_slot].occupied  = true;

    // Update lookup
    cache->lookup[key] = evict_slot;

    return true;
}

void alpha_expert_cache_prefetch(
        alpha_expert_cache * cache,
        int layer_id,
        const int * expert_ids,
        int n_experts) {
    if (!cache || !expert_ids) return;

    // Pre-warm the cache for predicted experts
    // In a full implementation, this would use async DMA
    for (int i = 0; i < n_experts; i++) {
        int key = make_key(layer_id, expert_ids[i], cache->config.n_experts_per_layer);
        auto it = cache->lookup.find(key);
        if (it != cache->lookup.end()) {
            // Already cached — just touch for LRU
            cache->slots[it->second].last_used = cache->step_counter + 1;
        }
        // Actual prefetch (async copy) would go here
    }
}

alpha_expert_cache_stats alpha_expert_cache_get_stats(
        const alpha_expert_cache * cache) {
    alpha_expert_cache_stats stats = {};
    if (!cache) return stats;

    stats.hits       = cache->hits;
    stats.misses     = cache->misses;
    stats.evictions  = cache->evictions;
    stats.slots_total = (int)cache->slots.size();

    int used = 0;
    for (const auto & s : cache->slots) {
        if (s.occupied) used++;
    }
    stats.slots_used = used;

    int64_t total = stats.hits + stats.misses;
    stats.hit_rate = total > 0 ? (float)stats.hits / total : 0.0f;

    return stats;
}

void alpha_expert_cache_print_stats(const alpha_expert_cache * cache) {
    if (!cache) return;

    auto s = alpha_expert_cache_get_stats(cache);

    fprintf(stderr, "\n=== Expert Cache Stats ===\n");
    fprintf(stderr, "  Slots: %d / %d used\n", s.slots_used, s.slots_total);
    fprintf(stderr, "  Hits:  %lld\n", (long long)s.hits);
    fprintf(stderr, "  Misses: %lld\n", (long long)s.misses);
    fprintf(stderr, "  Evictions: %lld\n", (long long)s.evictions);
    fprintf(stderr, "  Hit rate: %.1f%%\n", s.hit_rate * 100.0f);
    fprintf(stderr, "\n");
}

void alpha_expert_cache_free(alpha_expert_cache * cache) {
    if (!cache) return;
    cache->slots.clear();
    cache->lookup.clear();
    delete cache;
}
