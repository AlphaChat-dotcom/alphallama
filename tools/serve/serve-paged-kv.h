// SPDX-License-Identifier: Apache-2.0
#pragma once

// Paged KV Cache for FlashInfer Backend
//
// Manages GPU memory as a pool of fixed-size pages.
// Each sequence owns a list of pages via a page table.
// FlashInfer's paged_kv_t reads from this pool via CSR indirection.
//
// Memory layout per page: [PAGE_SIZE, n_kv_heads, head_dim] in FP16 (NHD)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace alphaserve {

struct PagedKVPool {
    static constexpr int PAGE_SIZE = 16;  // tokens per page (FlashInfer default)

    // Config
    int32_t n_kv_heads = 0;
    int32_t head_dim = 0;
    int32_t n_layers = 0;
    int32_t max_seqs = 0;

    // GPU memory pool - one pool per layer
    // Layout: [n_pages, PAGE_SIZE, n_kv_heads, head_dim] FP16
    struct LayerPool {
        half * k_data = nullptr;  // GPU pointer
        half * v_data = nullptr;  // GPU pointer
    };
    std::vector<LayerPool> layers;

    // Strides (in elements, not bytes)
    int32_t stride_page = 0;  // PAGE_SIZE * n_kv_heads * head_dim
    int32_t stride_n = 0;     // n_kv_heads * head_dim
    int32_t stride_h = 0;     // head_dim

    // Per-sequence state (CPU-side)
    struct SeqState {
        std::vector<int32_t> page_ids;
        int32_t n_tokens = 0;
        bool    exhausted = false;   // over-subscription: begin_token could not get a page

        int32_t n_pages() const {
            return (int32_t)page_ids.size();
        }
        int32_t last_page_len() const {
            if (n_tokens == 0) return 0;
            int rem = n_tokens % PAGE_SIZE;
            return rem == 0 ? PAGE_SIZE : rem;
        }
    };
    std::vector<SeqState> sequences;

    // Page management (mutex-protected for concurrent requests)
    std::mutex pool_mtx;
    std::vector<int32_t> free_pages;
    int32_t n_pages_total = 0;
    int32_t n_pages_used = 0;

    // GPU-side indirection arrays (pre-allocated, updated per batch)
    int32_t * d_indptr = nullptr;        // [max_batch + 1]
    int32_t * d_indices = nullptr;       // [max_total_pages]
    int32_t * d_last_page_len = nullptr; // [max_batch]

    // GPU-side KV write offset buffer (for CUDA graph-compatible scatter writes)
    int64_t * d_kv_write_offsets = nullptr;  // [max_batch] - element offsets into pool

    // Scratch sequence/page for CUDA-graph batch padding: graph-capture computes `bucket`
    // rows, but only `batch_size` are real. Padding rows [batch,bucket) are pointed at this
    // scratch seq (1 reserved page) so their (discarded) KV reads/writes can't touch real
    // sequences' pages. seq_id == max_seqs (sequences[] is sized max_seqs+1).
    int32_t scratch_seq_id = -1;
    int32_t scratch_page_id = -1;

    // API

    // Initialize pool with total page budget
    // reserve_bytes: VRAM to keep free for weights/compute allocated after this pool (FI GPU weight
    // copy + forward buffers). When >0, the pool measures free VRAM and auto-caps max_seqs so the KV
    // allocation fits (vLLM-style). max_seqs may be reduced - the caller must read it back after init.
    bool init(int32_t n_kv_heads, int32_t head_dim, int32_t n_layers,
              int32_t max_seqs, int32_t max_ctx_per_seq, size_t reserve_bytes = 0);

    // Cleanup
    void destroy();

    // Allocate a page from free list (returns page_id, -1 if full)
    int32_t alloc_page();

    // Free a page back to pool
    void free_page(int32_t page_id);

    // Over-subscription: true if this sequence hit pool exhaustion (begin_token could not
    // allocate a page). The scheduler force-finishes such sequences to reclaim their pages.
    bool is_exhausted(int seq_id) const {
        return seq_id >= 0 && seq_id < max_seqs && sequences[seq_id].exhausted;
    }

    // Three-phase token append:
    // 1. begin_token: pre-allocate page (does not update count)
    // 2. append_kv: write K/V per layer at position n_tokens
    // 3. commit_token: increment n_tokens (makes data visible to future attention)
    void begin_token(int seq_id);
    bool append_kv(int seq_id, int layer, const half * k, const half * v, cudaStream_t stream);
    void commit_token(int seq_id);

    // Compute KV write offsets for scatter kernel (CUDA graph compatible)
    // Uploads per-sequence offsets to d_kv_write_offsets
    void build_kv_write_offsets(const int32_t * seq_ids, int count, cudaStream_t stream);

    // Clear all pages for a sequence
    void clear_seq(int seq_id);

    // Truncate a sequence to keep only the first n_keep tokens.
    // Frees pages beyond n_keep. Useful for prefix caching: after generation
    // completes (prompt + N generated tokens), truncate back to prefix_len.
    void truncate_seq(int seq_id, int n_keep);

    // Build FlashInfer CSR indirection arrays for a batch
    // seq_ids: which sequences are in this batch
    // Copies indptr/indices/last_page_len to GPU
    void build_batch_args(const int32_t * seq_ids, int count, cudaStream_t stream);

    // Fast path: only update d_last_page_len when batch composition unchanged
    // Returns true if fast path was taken, false if full rebuild needed
    bool build_batch_args_fast(const int32_t * seq_ids, int count, cudaStream_t stream);

    // Serialize KV for a sequence to CPU buffer (GPU -> CPU, all layers)
    // Format: [n_layers x n_tokens x stride_n] FP16, K then V per layer
    std::vector<uint8_t> serialize_seq_kv(int seq_id, cudaStream_t stream);

    // Restore KV from CPU buffer, allocate pages, write to GPU
    // Returns number of tokens restored (0 on failure)
    int restore_seq_kv(int seq_id, const uint8_t * data, size_t size,
                        int n_tokens, cudaStream_t stream);

    // Get total GPU memory used (bytes)
    size_t gpu_memory_used() const;

    // Stats
    void print_stats() const;

private:
    // Cached state for fast batch args path
    std::vector<int> last_batch_seq_ids_;
    std::vector<int32_t> last_batch_page_counts_;
    // Pre-allocated host buffers (avoid heap alloc per step)
    std::vector<int32_t> h_last_page_len_;
    std::vector<int32_t> h_indptr_;
    std::vector<int32_t> h_indices_;
    std::vector<int64_t> h_kv_offsets_;
};

} // namespace alphaserve
