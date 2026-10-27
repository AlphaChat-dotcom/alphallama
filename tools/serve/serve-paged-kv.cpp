#include "serve-paged-kv.h"
#include <algorithm>
#include <numeric>

namespace alphaserve {

bool PagedKVPool::init(int32_t n_kv_heads_, int32_t head_dim_, int32_t n_layers_,
                       int32_t max_seqs_, int32_t max_ctx_per_seq, size_t reserve_bytes) {
    n_kv_heads = n_kv_heads_;
    head_dim = head_dim_;
    n_layers = n_layers_;
    max_seqs = max_seqs_;

    // Compute strides (NHD layout)
    stride_h = head_dim;
    stride_n = n_kv_heads * head_dim;
    stride_page = PAGE_SIZE * stride_n;

    int pages_per_seq = (max_ctx_per_seq + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t page_bytes = (size_t)stride_page * sizeof(half);
    // Per-seq KV footprint across all layers (K + V). +1 scratch page (graph padding) is negligible.
    size_t per_seq_bytes = (size_t)pages_per_seq * page_bytes * (size_t)n_layers * 2;

    // Auto-fit with dynamic over-subscription of a shared page pool.
    // reserve_bytes accounts for the FI weight copy + forward buffers allocated after this pool.
    // The page pool is a shared budget (pool_seqs full-ctx sequences worth of pages), not one
    // full-ctx reservation per slot: short generations use few pages, so the requested slot count
    // (max_seqs) is kept and all slots draw from the shared pool. A slot that cannot get a page is
    // force-finished by the scheduler (begin_token sets `exhausted`), so memory pressure truncates
    // gracefully instead of OOMing or corrupting KV. ALPHA_FI_KV_OVERSUB=0 caps slots to the fit.
    int pool_seqs = max_seqs;  // number of full-ctx sequences the shared page pool holds
    if (reserve_bytes > 0 && per_seq_bytes > 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        size_t budget = (free_b > reserve_bytes) ? (size_t)((free_b - reserve_bytes) * 0.90) : 0;
        int fit_seqs = (int)(budget / per_seq_bytes);
        if (fit_seqs < 1) fit_seqs = 1;   // always keep at least one slot
        static const bool oversub = getenv("ALPHA_FI_KV_OVERSUB")
                                    ? atoi(getenv("ALPHA_FI_KV_OVERSUB")) != 0 : true;
        if (fit_seqs < max_seqs && oversub) {
            pool_seqs = fit_seqs;                 // pool = budget; slot count stays = requested
            fprintf(stderr,
                "[paged-kv] OVER-SUBSCRIBE: %d slots on a %d-seq shared page pool (%.1f GiB, %d ctx "
                "worst-case) — short gens share pages; slots that exhaust the pool are force-finished\n",
                max_seqs, fit_seqs, (double)fit_seqs * per_seq_bytes / 1073741824.0, max_ctx_per_seq);
        } else if (fit_seqs < max_seqs) {
            fprintf(stderr, "[paged-kv] AUTO-FIT: capping %d → %d slots (%.1f GiB KV, oversub off)\n",
                    max_seqs, fit_seqs, (double)fit_seqs * per_seq_bytes / 1073741824.0);
            max_seqs = fit_seqs;
            pool_seqs = fit_seqs;
        } else {
            pool_seqs = max_seqs;
            fprintf(stderr, "[paged-kv] fits: %d slots × %d ctx = %.1f GiB KV (%.1f GiB free)\n",
                    max_seqs, max_ctx_per_seq, (double)max_seqs * per_seq_bytes / 1073741824.0,
                    (double)free_b / 1073741824.0);
        }
    }

    // Shared page pool sized to the budget. Guard: never fewer pages than slots+scratch, so
    // every admitted slot can hold at least its first page.
    n_pages_total = pages_per_seq * pool_seqs;
    if (n_pages_total < max_seqs + 1) n_pages_total = max_seqs + 1;
    n_pages_used = 0;

    // Allocate GPU memory per layer
    size_t pool_bytes = (size_t)n_pages_total * page_bytes;

    layers.resize(n_layers);
    for (int il = 0; il < n_layers; il++) {
        cudaError_t err;
        err = cudaMalloc(&layers[il].k_data, pool_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[paged-kv] cudaMalloc failed for layer %d K: %s\n",
                    il, cudaGetErrorString(err));
            destroy();
            return false;
        }
        err = cudaMalloc(&layers[il].v_data, pool_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[paged-kv] cudaMalloc failed for layer %d V: %s\n",
                    il, cudaGetErrorString(err));
            destroy();
            return false;
        }
        // Zero init
        cudaMemset(layers[il].k_data, 0, pool_bytes);
        cudaMemset(layers[il].v_data, 0, pool_bytes);
    }

    // Initialize free list (all pages free)
    free_pages.resize(n_pages_total);
    std::iota(free_pages.begin(), free_pages.end(), 0);

    // Initialize sequences (+1 slot for the CUDA-graph padding scratch sequence)
    sequences.resize(max_seqs + 1);

    // Reserve one scratch page for the scratch sequence (graph-padding rows write here).
    // Take it off the free list so no real sequence can be assigned it.
    scratch_seq_id = max_seqs;
    scratch_page_id = free_pages.back();
    free_pages.pop_back();
    n_pages_used++;
    sequences[scratch_seq_id].page_ids = { scratch_page_id };
    sequences[scratch_seq_id].n_tokens = 1;   // 1 page, last_page_len()==1

    // Allocate GPU indirection arrays. Sized for max_seqs padded rows plus up to max_seqs
    // extra scratch page-refs (one per padding row, all pointing at scratch_page_id).
    int max_total_pages = n_pages_total + max_seqs + 1;
    cudaMalloc(&d_indptr, (max_seqs + 2) * sizeof(int32_t));
    cudaMalloc(&d_indices, max_total_pages * sizeof(int32_t));
    cudaMalloc(&d_last_page_len, (max_seqs + 1) * sizeof(int32_t));
    // Size for max_ctx_per_seq: forward_prefill writes one offset per token (N), not per sequence.
    int max_offsets = std::max(max_seqs + 1, max_ctx_per_seq);
    cudaMalloc(&d_kv_write_offsets, max_offsets * sizeof(int64_t));

    // Pre-allocate host buffers for build_batch_args
    h_indptr_.resize(max_seqs + 2);
    h_last_page_len_.resize(max_seqs + 1);
    h_indices_.reserve(max_total_pages);
    h_kv_offsets_.resize(max_offsets);

    size_t total_mb = (pool_bytes * n_layers * 2) / (1024 * 1024);
    fprintf(stderr, "[paged-kv] initialized: %d pages × %d layers, "
            "%d heads × %d dim, page_size=%d, %.0f MiB GPU\n",
            n_pages_total, n_layers, n_kv_heads, head_dim, PAGE_SIZE,
            (double)total_mb);

    return true;
}

void PagedKVPool::destroy() {
    for (auto & lp : layers) {
        if (lp.k_data) { cudaFree(lp.k_data); lp.k_data = nullptr; }
        if (lp.v_data) { cudaFree(lp.v_data); lp.v_data = nullptr; }
    }
    layers.clear();
    if (d_indptr) { cudaFree(d_indptr); d_indptr = nullptr; }
    if (d_indices) { cudaFree(d_indices); d_indices = nullptr; }
    if (d_last_page_len) { cudaFree(d_last_page_len); d_last_page_len = nullptr; }
    if (d_kv_write_offsets) { cudaFree(d_kv_write_offsets); d_kv_write_offsets = nullptr; }
    free_pages.clear();
    sequences.clear();
}

int32_t PagedKVPool::alloc_page() {
    // Caller must hold pool_mtx
    if (free_pages.empty()) return -1;
    int32_t page_id = free_pages.back();
    free_pages.pop_back();
    n_pages_used++;
    return page_id;
}

void PagedKVPool::free_page(int32_t page_id) {
    // Caller must hold pool_mtx
    free_pages.push_back(page_id);
    n_pages_used--;
}

// Pre-allocate page for the next token (call once per forward pass, before layers).
// Does not increment n_tokens; that happens in commit_token().
void PagedKVPool::begin_token(int seq_id) {
    if (seq_id < 0 || seq_id >= max_seqs) return;
    std::lock_guard<std::mutex> lock(pool_mtx);
    auto & seq = sequences[seq_id];

    int pos_in_page = seq.n_tokens % PAGE_SIZE;
    if (pos_in_page == 0) {
        int32_t page_id = alloc_page();
        if (page_id < 0) {
            // Shared pool exhausted (over-subscription backpressure). Flag the sequence so the
            // scheduler force-finishes it and reclaims its pages; append_kv's bounds check keeps
            // this token from corrupting KV in the meantime.
            seq.exhausted = true;
            static int64_t oom_log = 0;
            if ((oom_log++ & 63) == 0)
                fprintf(stderr, "[paged-kv] pool full — seq %d exhausted (%d/%d pages) → force-finish\n",
                        seq_id, n_pages_used, n_pages_total);
            return;
        }
        seq.page_ids.push_back(page_id);
    }
}

// Write K/V for a specific layer at position n_tokens (the new token)
bool PagedKVPool::append_kv(int seq_id, int layer,
                             const half * k, const half * v,
                             cudaStream_t stream) {
    if (seq_id < 0 || seq_id >= max_seqs) return false;
    auto & seq = sequences[seq_id];

    // Write at position n_tokens (the pre-allocated slot from begin_token)
    int pos = seq.n_tokens;
    int pos_in_page = pos % PAGE_SIZE;
    int page_idx = pos / PAGE_SIZE;
    if (page_idx >= (int)seq.page_ids.size()) return false;
    int32_t page_id = seq.page_ids[page_idx];

    size_t offset = (size_t)page_id * stride_page + (size_t)pos_in_page * stride_n;

    auto & lp = layers[layer];
    cudaMemcpyAsync(lp.k_data + offset, k, stride_n * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(lp.v_data + offset, v, stride_n * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);

    return true;
}

// Finalize the token after all layers have written K/V
void PagedKVPool::commit_token(int seq_id) {
    if (seq_id < 0 || seq_id >= max_seqs) return;
    sequences[seq_id].n_tokens++;
}

void PagedKVPool::clear_seq(int seq_id) {
    if (seq_id < 0 || seq_id >= max_seqs) return;
    std::lock_guard<std::mutex> lock(pool_mtx);
    auto & seq = sequences[seq_id];

    for (int32_t page_id : seq.page_ids) {
        free_page(page_id);
    }
    seq.page_ids.clear();
    seq.n_tokens = 0;
    seq.exhausted = false;
}

void PagedKVPool::truncate_seq(int seq_id, int n_keep) {
    if (seq_id < 0 || seq_id >= max_seqs) return;
    std::lock_guard<std::mutex> lock(pool_mtx);
    auto & seq = sequences[seq_id];

    if (n_keep <= 0) {
        for (int32_t page_id : seq.page_ids) free_page(page_id);
        seq.page_ids.clear();
        seq.n_tokens = 0;
        last_batch_seq_ids_.clear();  // invalidate fast path
        return;
    }
    if (n_keep >= seq.n_tokens) return;

    int pages_needed = (n_keep + PAGE_SIZE - 1) / PAGE_SIZE;
    while ((int)seq.page_ids.size() > pages_needed) {
        free_page(seq.page_ids.back());
        seq.page_ids.pop_back();
    }
    seq.n_tokens = n_keep;
    last_batch_seq_ids_.clear();  // invalidate fast path after page changes
}

void PagedKVPool::build_batch_args(const int32_t * seq_ids, int count,
                                    cudaStream_t stream) {
    // Build CPU-side arrays (using pre-allocated buffers)
    h_indptr_[0] = 0;
    h_indices_.clear();
    for (int i = 0; i < count; i++) {
        auto & seq = sequences[seq_ids[i]];
        h_indptr_[i + 1] = h_indptr_[i] + seq.n_pages();
        for (int32_t pid : seq.page_ids) {
            h_indices_.push_back(pid);
        }
        h_last_page_len_[i] = seq.last_page_len();
    }

    // Copy to GPU
    cudaMemcpyAsync(d_indptr, h_indptr_.data(),
                    (count + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    if (!h_indices_.empty()) {
        cudaMemcpyAsync(d_indices, h_indices_.data(),
                        h_indices_.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(d_last_page_len, h_last_page_len_.data(),
                    count * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);

    // Save state for fast path
    last_batch_seq_ids_.assign(seq_ids, seq_ids + count);
    last_batch_page_counts_.resize(count);
    for (int i = 0; i < count; i++) {
        last_batch_page_counts_[i] = sequences[seq_ids[i]].n_pages();
    }
}

bool PagedKVPool::build_batch_args_fast(const int32_t * seq_ids, int count,
                                         cudaStream_t stream) {
    // Check same batch composition
    if (count != (int)last_batch_seq_ids_.size()) return false;
    for (int i = 0; i < count; i++) {
        if (seq_ids[i] != last_batch_seq_ids_[i]) return false;
    }

    // Check no new pages allocated (indptr/indices unchanged)
    for (int i = 0; i < count; i++) {
        if (sequences[seq_ids[i]].n_pages() != last_batch_page_counts_[i]) return false;
    }

    // Fast path: only update d_last_page_len (1 small H2D vs 3)
    for (int i = 0; i < count; i++) {
        h_last_page_len_[i] = sequences[seq_ids[i]].last_page_len();
    }
    cudaMemcpyAsync(d_last_page_len, h_last_page_len_.data(),
                    count * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    return true;
}

void PagedKVPool::build_kv_write_offsets(const int32_t * seq_ids, int count,
                                          cudaStream_t stream) {
    int batch_size = count;

    for (int i = 0; i < batch_size; i++) {
        auto & seq = sequences[seq_ids[i]];
        int pos = seq.n_tokens;  // position to write at (pre-committed)
        int pos_in_page = pos % PAGE_SIZE;
        int page_idx = pos / PAGE_SIZE;

        int64_t offset = 0;
        if (page_idx < (int)seq.page_ids.size()) {
            int32_t page_id = seq.page_ids[page_idx];
            offset = (int64_t)page_id * stride_page + (int64_t)pos_in_page * stride_n;
        }
        h_kv_offsets_[i] = offset;
    }

    cudaMemcpyAsync(d_kv_write_offsets, h_kv_offsets_.data(),
                    batch_size * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
}

size_t PagedKVPool::gpu_memory_used() const {
    size_t page_bytes = (size_t)stride_page * sizeof(half);
    size_t pool_bytes = (size_t)n_pages_total * page_bytes;
    return pool_bytes * n_layers * 2;  // K + V
}

std::vector<uint8_t> PagedKVPool::serialize_seq_kv(int seq_id, cudaStream_t stream) {
    if (seq_id < 0 || seq_id >= max_seqs) return {};
    auto & seq = sequences[seq_id];
    if (seq.n_tokens == 0) return {};

    int kv_dim = stride_n;  // n_kv_heads * head_dim (in FP16 elements)
    size_t per_token_bytes = kv_dim * sizeof(half);
    // Format: [layer, K_or_V, token] -> total = n_layers * 2 * n_tokens * per_token_bytes
    size_t total = (size_t)n_layers * 2 * seq.n_tokens * per_token_bytes;
    std::vector<uint8_t> out(total);

    cudaStreamSynchronize(stream);

    size_t offset = 0;
    for (int il = 0; il < n_layers; il++) {
        // Copy K for all tokens (may span multiple pages)
        for (int t = 0; t < seq.n_tokens; t++) {
            int page_idx = t / PAGE_SIZE;
            int pos_in_page = t % PAGE_SIZE;
            int32_t page_id = seq.page_ids[page_idx];
            size_t gpu_off = (size_t)page_id * stride_page + (size_t)pos_in_page * stride_n;
            cudaMemcpy(out.data() + offset, layers[il].k_data + gpu_off,
                       per_token_bytes, cudaMemcpyDeviceToHost);
            offset += per_token_bytes;
        }
        // Copy V for all tokens
        for (int t = 0; t < seq.n_tokens; t++) {
            int page_idx = t / PAGE_SIZE;
            int pos_in_page = t % PAGE_SIZE;
            int32_t page_id = seq.page_ids[page_idx];
            size_t gpu_off = (size_t)page_id * stride_page + (size_t)pos_in_page * stride_n;
            cudaMemcpy(out.data() + offset, layers[il].v_data + gpu_off,
                       per_token_bytes, cudaMemcpyDeviceToHost);
            offset += per_token_bytes;
        }
    }
    return out;
}

int PagedKVPool::restore_seq_kv(int seq_id, const uint8_t * data, size_t size,
                                  int n_tokens, cudaStream_t stream) {
    if (seq_id < 0 || seq_id >= max_seqs || !data || n_tokens <= 0) return 0;

    int kv_dim = stride_n;
    size_t per_token_bytes = kv_dim * sizeof(half);
    size_t expected = (size_t)n_layers * 2 * n_tokens * per_token_bytes;
    if (size < expected) return 0;

    // Clear existing sequence
    clear_seq(seq_id);

    // Allocate pages for n_tokens
    auto & seq = sequences[seq_id];
    {
        std::lock_guard<std::mutex> lock(pool_mtx);
        int n_pages_needed = (n_tokens + PAGE_SIZE - 1) / PAGE_SIZE;
        for (int p = 0; p < n_pages_needed; p++) {
            int32_t pid = alloc_page();
            if (pid < 0) {
                fprintf(stderr, "[paged-kv] restore failed: out of pages at %d/%d\n", p, n_pages_needed);
                return 0;
            }
            seq.page_ids.push_back(pid);
        }
    }

    // Copy data from CPU to GPU pages
    size_t offset = 0;
    for (int il = 0; il < n_layers; il++) {
        for (int t = 0; t < n_tokens; t++) {
            int page_idx = t / PAGE_SIZE;
            int pos_in_page = t % PAGE_SIZE;
            int32_t page_id = seq.page_ids[page_idx];
            size_t gpu_off = (size_t)page_id * stride_page + (size_t)pos_in_page * stride_n;
            cudaMemcpyAsync(layers[il].k_data + gpu_off, data + offset,
                            per_token_bytes, cudaMemcpyHostToDevice, stream);
            offset += per_token_bytes;
        }
        for (int t = 0; t < n_tokens; t++) {
            int page_idx = t / PAGE_SIZE;
            int pos_in_page = t % PAGE_SIZE;
            int32_t page_id = seq.page_ids[page_idx];
            size_t gpu_off = (size_t)page_id * stride_page + (size_t)pos_in_page * stride_n;
            cudaMemcpyAsync(layers[il].v_data + gpu_off, data + offset,
                            per_token_bytes, cudaMemcpyHostToDevice, stream);
            offset += per_token_bytes;
        }
    }
    cudaStreamSynchronize(stream);
    seq.n_tokens = n_tokens;
    return n_tokens;
}

void PagedKVPool::print_stats() const {
    fprintf(stderr, "[paged-kv] pages: %d/%d used (%.1f%%), seqs: %d, mem: %.0f MiB\n",
            n_pages_used, n_pages_total,
            n_pages_total > 0 ? 100.0 * n_pages_used / n_pages_total : 0.0,
            max_seqs,
            (double)gpu_memory_used() / (1024 * 1024));
}

} // namespace alphaserve
