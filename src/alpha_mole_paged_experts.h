#pragma once

// AlphaLlama: paged expert cache
//
// Splits MoE expert weight matrices into fixed-size pages (default 64KB).
// Benefits over full-expert caching:
//   1. Higher effective hit rate: hot pages from all experts are cached,
//      rather than only K full experts with the rest cold
//   2. Cheaper miss: load a 64KB page (~20us) instead of a 4MB expert (~1ms)
//   3. DMA-compute overlap: stream pages while computing on cached pages
//
// Architecture:
//   Expert weights [n_ff, n_embd, n_expert] split into pages of PAGE_ROWS rows each.
//   Page table tracks which pages are in VRAM (hot), CPU (cold), or in-flight (DMA).
//   On cache miss: only missing pages are loaded, not the entire expert.
//   GEMM accumulates results page-by-page (tiled matmul).
//
// Integration:
//   Sits between the routing decision (argsort) and the GEMM execution.
//   Replaces the full-expert DMA in alpha_moe_optimizer's callback.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <atomic>

// Page size in bytes (64KB = good balance between granularity and overhead)
// For Q4_K_M with n_embd=2048: one page ~ 32 rows of weight matrix
static constexpr size_t EXPERT_PAGE_SIZE = 64 * 1024;  // 64 KB

// Maximum pages per expert (4MB expert / 64KB page = 64 pages)
static constexpr int MAX_PAGES_PER_EXPERT = 256;

struct expert_page {
    int expert_id = -1;
    int page_idx  = -1;        // which page within the expert (0..n_pages-1)
    int8_t * vram_ptr = nullptr;  // non-null if cached in VRAM pool
    int64_t last_access = 0;   // LRU timestamp
    bool in_flight = false;    // DMA in progress
};

struct expert_page_table {
    // Configuration
    int n_experts = 0;
    int n_pages_per_expert = 0;
    size_t page_size = EXPERT_PAGE_SIZE;
    size_t expert_slice_bytes = 0;  // bytes per full expert

    // Page tracking: [expert_id][page_idx] -> page metadata
    // -1 = not cached, >=0 = pool slot index
    std::vector<std::vector<int>> page_slots;  // [n_experts][n_pages_per_expert]

    // VRAM page pool
    int8_t * vram_pool = nullptr;      // contiguous VRAM allocation
    size_t vram_pool_bytes = 0;
    int n_pool_slots = 0;
    std::vector<expert_page> pool;     // pool[slot_idx] = page metadata
    std::vector<int> free_slots;       // available pool slots

    // LRU eviction
    int64_t clock = 0;

    // Stats
    int64_t page_hits = 0;
    int64_t page_misses = 0;
    int64_t page_evictions = 0;

    // Initialize page table for one operation type (gate, up, or down)
    // vram_budget: bytes of VRAM for this page pool
    // expert_bytes: bytes per full expert weight slice
    // n_exp: total number of experts
    void init(size_t vram_budget, size_t expert_bytes, int n_exp) {
        n_experts = n_exp;
        expert_slice_bytes = expert_bytes;
        n_pages_per_expert = (int)((expert_bytes + page_size - 1) / page_size);
        n_pool_slots = (int)(vram_budget / page_size);

        page_slots.resize(n_experts);
        for (auto & v : page_slots) {
            v.assign(n_pages_per_expert, -1);
        }

        pool.resize(n_pool_slots);
        free_slots.reserve(n_pool_slots);
        for (int i = n_pool_slots - 1; i >= 0; i--) {
            free_slots.push_back(i);
        }
    }

    // Check how many pages of an expert are cached
    int cached_pages(int expert_id) const {
        int count = 0;
        for (int p = 0; p < n_pages_per_expert; p++) {
            if (page_slots[expert_id][p] >= 0) count++;
        }
        return count;
    }

    // Check if expert is fully cached
    bool fully_cached(int expert_id) const {
        return cached_pages(expert_id) == n_pages_per_expert;
    }

    // Get hit rate
    float hit_rate() const {
        int64_t total = page_hits + page_misses;
        return total > 0 ? (float)page_hits / total : 0.0f;
    }

    // Allocate a pool slot (evict LRU if full)
    int alloc_slot() {
        if (!free_slots.empty()) {
            int slot = free_slots.back();
            free_slots.pop_back();
            return slot;
        }

        // Evict LRU page
        int lru_slot = -1;
        int64_t lru_time = INT64_MAX;
        for (int i = 0; i < n_pool_slots; i++) {
            if (pool[i].expert_id >= 0 && pool[i].last_access < lru_time && !pool[i].in_flight) {
                lru_time = pool[i].last_access;
                lru_slot = i;
            }
        }

        if (lru_slot < 0) return -1;  // all slots in-flight, can't evict

        // Unmap evicted page
        auto & evicted = pool[lru_slot];
        page_slots[evicted.expert_id][evicted.page_idx] = -1;
        evicted.expert_id = -1;
        evicted.page_idx = -1;
        page_evictions++;

        return lru_slot;
    }

    // Ensure a page is in VRAM. Returns VRAM pointer to the page data.
    // If not cached, caller must DMA the data into the returned pointer.
    // cpu_data: pointer to the full expert's CPU data (expert_slice_bytes)
    // Returns: {vram_ptr, was_hit}
    struct ensure_result {
        int8_t * vram_ptr;
        bool was_hit;
    };

    ensure_result ensure_page(int expert_id, int page_idx) {
        int slot = page_slots[expert_id][page_idx];
        if (slot >= 0) {
            // Cache hit
            pool[slot].last_access = ++clock;
            page_hits++;
            return { pool[slot].vram_ptr, true };
        }

        // Cache miss: allocate slot
        slot = alloc_slot();
        if (slot < 0) {
            page_misses++;
            return { nullptr, false };  // pool exhausted
        }

        pool[slot].expert_id = expert_id;
        pool[slot].page_idx = page_idx;
        pool[slot].last_access = ++clock;
        pool[slot].vram_ptr = vram_pool + (size_t)slot * page_size;
        page_slots[expert_id][page_idx] = slot;
        page_misses++;

        return { pool[slot].vram_ptr, false };
    }

    // Get VRAM pointer for a cached page (returns nullptr if not cached)
    int8_t * get_page_ptr(int expert_id, int page_idx) const {
        int slot = page_slots[expert_id][page_idx];
        if (slot < 0) return nullptr;
        return pool[slot].vram_ptr;
    }
};

// end of alpha_mole_paged_experts.h
