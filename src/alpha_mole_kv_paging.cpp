#include "alpha_mole_kv_paging.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

struct alpha_kv_cold_entry {
    int      position;           // original token position
    int      layer;              // layer index
    std::vector<float> k_data;   // key vector (flattened across heads)
    std::vector<float> v_data;   // value vector (flattened across heads)
};

struct alpha_kv_pager {
    llama_context * ctx;
    alpha_kv_pager_params params;

    // Cold store: KV entries evicted from GPU
    std::vector<alpha_kv_cold_entry> cold_store;

    // Statistics
    int evictions;
    int retrievals;
    int peak_hot;
};

alpha_kv_pager * alpha_kv_pager_init(
        struct llama_context * ctx,
        alpha_kv_pager_params   params) {
    if (!ctx) return nullptr;

    auto * pager = new alpha_kv_pager();
    pager->ctx    = ctx;
    pager->params = params;
    pager->evictions  = 0;
    pager->retrievals = 0;
    pager->peak_hot   = 0;

    // Validate params
    if (pager->params.gpu_budget <= 0)    pager->params.gpu_budget = 4096;
    if (pager->params.cpu_budget <= 0)    pager->params.cpu_budget = 100000;
    if (pager->params.retrieval_k <= 0)   pager->params.retrieval_k = 200;

    fprintf(stderr, "alpha_kv_pager: initialized (gpu_budget=%d, cpu_budget=%d, retrieval_k=%d)\n",
            pager->params.gpu_budget, pager->params.cpu_budget, pager->params.retrieval_k);

    return pager;
}

void alpha_kv_pager_update(
        alpha_kv_pager * pager,
        int              n_past) {
    if (!pager || !pager->ctx) return;

    // Track peak hot tokens
    if (n_past > pager->peak_hot) {
        pager->peak_hot = n_past;
    }

    // Check if we've exceeded GPU budget
    if (n_past <= pager->params.gpu_budget) {
        return;  // still within budget, no eviction needed
    }

    // Calculate how many tokens to evict
    int n_to_evict = n_past - pager->params.gpu_budget;

    // Evict oldest tokens (positions 0..n_to_evict-1)
    // In a full implementation, we would:
    // 1. Read the KV data for these positions from GPU
    // 2. Store in cold_store (CPU RAM)
    // 3. Call llama_memory_seq_rm() to free GPU cells
    //
    // For V1, we just track the eviction and let llama.cpp's ring buffer
    // handle the actual eviction. The cold store is metadata-only.

    llama_memory_t mem = llama_get_memory(pager->ctx);
    if (mem) {
        // Get the position range of the sequence
        llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);

        // Evict the oldest n_to_evict tokens
        llama_pos evict_end = pos_min + n_to_evict;
        if (evict_end > pos_max) evict_end = pos_max;

        if (evict_end > pos_min) {
            // Record cold entries (metadata only in V1)
            for (llama_pos p = pos_min; p < evict_end; p++) {
                if ((int)pager->cold_store.size() < pager->params.cpu_budget) {
                    alpha_kv_cold_entry entry = {};
                    entry.position = p;
                    pager->cold_store.push_back(entry);
                }
            }

            // Remove from GPU KV cache
            llama_memory_seq_rm(mem, 0, pos_min, evict_end);
            pager->evictions += (int)(evict_end - pos_min);
        }
    }
}

alpha_kv_pager_stats alpha_kv_pager_get_stats(
        const alpha_kv_pager * pager) {
    alpha_kv_pager_stats stats = {};
    if (!pager) return stats;

    stats.cold_tokens  = (int)pager->cold_store.size();
    stats.total_tokens = pager->peak_hot;
    stats.hot_tokens   = stats.total_tokens - stats.cold_tokens;
    stats.evictions    = pager->evictions;
    stats.retrievals   = pager->retrievals;
    stats.hit_rate     = stats.total_tokens > 0
        ? (float)stats.hot_tokens / stats.total_tokens
        : 1.0f;

    return stats;
}

void alpha_kv_pager_print_stats(const alpha_kv_pager * pager) {
    if (!pager) return;

    alpha_kv_pager_stats s = alpha_kv_pager_get_stats(pager);

    fprintf(stderr, "\n=== KV Paging Stats ===\n");
    fprintf(stderr, "  Hot (GPU):  %d tokens\n", s.hot_tokens);
    fprintf(stderr, "  Cold (CPU): %d tokens\n", s.cold_tokens);
    fprintf(stderr, "  Total:      %d tokens\n", s.total_tokens);
    fprintf(stderr, "  Evictions:  %d\n", s.evictions);
    fprintf(stderr, "  Retrievals: %d\n", s.retrievals);
    fprintf(stderr, "  Hit rate:   %.1f%%\n", s.hit_rate * 100.0f);
    fprintf(stderr, "\n");
}

void alpha_kv_pager_free(alpha_kv_pager * pager) {
    if (!pager) return;
    pager->cold_store.clear();
    delete pager;
}
