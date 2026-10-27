#include "alpha_mole_paged_experts.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <algorithm>
#include <cstdio>

// Gather expert pages from page table into a contiguous GPU buffer.
// This enables using the existing ggml_mul_mat_id kernel which expects
// contiguous expert weight slices.
//
// For each page of the expert, the page table records a hit or miss (for LRU) and the page is
// copied into the expert's slot of the contiguous cache tensor:
//   [expert_0_weights | expert_1_weights | ... | expert_K_weights]
//   each expert_i = n_pages x page_size bytes
// Page-level LRU keeps hot rows from many experts tracked as cached.

void paged_expert_gather(
    expert_page_table & pt,
    int expert_id,
    const void * cpu_expert_data,      // CPU pointer to this expert's full weight slice
    ggml_tensor * gpu_cache_tensor,    // GPU cache tensor [dim, dim, K]
    int cache_slot,                     // which slot in the cache tensor
    size_t expert_bytes                 // bytes per full expert
) {
    size_t slot_offset = (size_t)cache_slot * expert_bytes;

    for (int p = 0; p < pt.n_pages_per_expert; p++) {
        size_t page_offset = (size_t)p * pt.page_size;
        size_t page_bytes = std::min(pt.page_size, expert_bytes - page_offset);
        if (page_bytes == 0) break;

        auto result = pt.ensure_page(expert_id, p);

        if (result.was_hit && result.vram_ptr) {
            // Page is cached in the VRAM pool. ggml_backend_tensor_set only takes host pointers,
            // so the page is still copied from host memory; only the LRU tracking differs.
            // TODO: copy device-to-device from the pool page instead.
            const char * src = (const char *)cpu_expert_data + page_offset;
            ggml_backend_tensor_set(gpu_cache_tensor, src,
                                    slot_offset + page_offset, page_bytes);
        } else if (result.vram_ptr) {
            // Page miss: copy from host memory into the cache slot
            const char * src = (const char *)cpu_expert_data + page_offset;

            // Copy to contiguous cache slot
            ggml_backend_tensor_set(gpu_cache_tensor, src,
                                    slot_offset + page_offset, page_bytes);

            // TODO: also copy the page into the pool (async, on a separate stream). The pool
            // currently only records the miss for LRU purposes.
        } else {
            // Pool exhausted: direct copy without caching
            const char * src = (const char *)cpu_expert_data + page_offset;
            ggml_backend_tensor_set(gpu_cache_tensor, src,
                                    slot_offset + page_offset, page_bytes);
        }
    }
}

// Check how much of an expert is already cached (0.0 = none, 1.0 = fully cached)
float paged_expert_coverage(const expert_page_table & pt, int expert_id) {
    if (expert_id < 0 || expert_id >= pt.n_experts) return 0.0f;
    return (float)pt.cached_pages(expert_id) / pt.n_pages_per_expert;
}

// Print page table statistics
void paged_expert_print_stats(const expert_page_table & pt) {
    int total_cached = 0;
    int experts_with_pages = 0;
    for (int e = 0; e < pt.n_experts; e++) {
        int c = pt.cached_pages(e);
        if (c > 0) {
            experts_with_pages++;
            total_cached += c;
        }
    }

    fprintf(stderr, "[paged-experts] %d/%d pages cached (%.1f%%), "
            "%d experts have cached pages, "
            "hits=%lld misses=%lld evictions=%lld hit_rate=%.1f%%\n",
            total_cached, pt.n_pool_slots, 100.0f * total_cached / std::max(pt.n_pool_slots, 1),
            experts_with_pages,
            (long long)pt.page_hits, (long long)pt.page_misses, (long long)pt.page_evictions,
            100.0f * pt.hit_rate());
}
