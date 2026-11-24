#pragma once

#include "llama-kv-cache.h"

#include <array>
#include <map>
#include <memory>
#include <vector>

//
// llama_kv_cache_dsv41
//
// Memory of DeepSeek-V4.1 (arch deepseek41). Hand port of what upstream llama.cpp does in
// llama-kv-cache-dsv4.{h,cpp} for this architecture, reduced to what the fork needs:
//
//   - the raw (sliding window) K of every layer lives in the base llama_kv_cache, which is created as a small SWA
//     cache (n_swa + n_ubatch cells per stream) and prunes cells that left the window on its own
//   - every KV source layer owns the compressed rows [kv latent | index key] of its sequence, one row per `ratio`
//     tokens, at row pos / ratio; the layers that share the source read the same tensor (no per-device mirrors)
//   - a source layer with ratio > 1 also keeps a ring of `ratio` compressor state rows (kv, score) so a block that
//     straddles two ubatches can be completed
//
// Differences from upstream: one sequence per ubatch (several sequences are served one after the other), no
// rollback snapshots (a partial seq_rm only succeeds where the compressor state is still exact), no session
// save / restore of the compressed rows, no quantised KV types.

struct llama_dsv41_plan {
    uint32_t ratio    = 0;
    int64_t  n_comp   = 0; // graph width of the compressed rows (padded to 256)
    int64_t  n_blocks = 0; // rows written by this ubatch; a masked scratch row when no block completes

    std::vector<int32_t> n_visible;   // [n_tokens]         completed rows each token may attend to
    std::vector<int32_t> read_idxs;   // [ratio*n_blocks]   rows of [state ring | ubatch tokens] (ratio > 1)
    std::vector<int64_t> write_idxs;  // [n_blocks]         stream-local row
    std::vector<int32_t> write_pos;   // [n_blocks]         position of the first token of the block
    std::vector<int32_t> persist_src; // ubatch token ...
    std::vector<int32_t> persist_dst; // ... kept in this ring row (ratio > 1)
};

class llama_kv_cache_dsv41 : public llama_kv_cache {
public:
    llama_kv_cache_dsv41(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch);

    ~llama_kv_cache_dsv41() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_dsv41 specific API
    //

    struct comp_layer {
        int32_t  il    = -1;
        uint32_t ratio = 0;
        ggml_tensor * rows     = nullptr; // [n_embd_head + n_idx_embd, capacity + 1, n_stream], type_k
        ggml_tensor * st_kv    = nullptr; // [n_embd_head, ratio, n_stream] F32 (ratio > 1)
        ggml_tensor * st_score = nullptr; // [n_embd_head, ratio, n_stream] F32 (ratio > 1)
    };

    // rows a stream can hold for this ratio; the row after them is the scratch row
    uint32_t comp_capacity(uint32_t ratio) const;

    const comp_layer & get_comp_layer(int32_t il_src) const;

    uint32_t  comp_stream(llama_seq_id seq_id) const;
    llama_pos get_pos_next(llama_seq_id seq_id) const;
    void      set_pos_next(llama_seq_id seq_id, llama_pos pos);

    uint32_t get_n_ctx_seq() const { return n_ctx_seq; }

    llama_dsv41_plan build_plan(const llama_ubatch & ubatch, uint32_t ratio) const;
    llama_dsv41_plan build_plan_reserve(const llama_ubatch & ubatch, uint32_t ratio) const;

private:
    const uint32_t n_ctx_seq;
    const uint32_t n_seq_max_41;

    std::vector<comp_layer> comp_layers;
    std::map<int32_t, int32_t> comp_map; // source layer -> comp_layers index

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> comp_bufs;

    // [n_seq_max] position the next token of the sequence must have
    std::vector<llama_pos> pos_next;

    void clear_comp(llama_seq_id seq_id);
};

class llama_kv_cache_dsv41_context : public llama_kv_cache_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_kv_cache_dsv41_context(llama_memory_status status);

    // full-cache context (graph reservation)
    explicit llama_kv_cache_dsv41_context(llama_kv_cache_dsv41 * kv);

    llama_kv_cache_dsv41_context(
            llama_kv_cache_dsv41 * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    ~llama_kv_cache_dsv41_context() = default;

    bool next()  override;
    bool apply() override;

    //
    // llama_kv_cache_dsv41_context specific API
    //

    bool is_full() const { return full; }

    // stream of the sequence of the current ubatch (0 for the full context)
    uint32_t get_stream() const;

    // ratio is 1 or 2; for the full context the worst case for `ubatch`
    const llama_dsv41_plan & get_plan(uint32_t ratio, const llama_ubatch & ubatch) const;

    // the stream's compressed rows of source layer il_src, scratch row included: [n_row, 1, capacity + 1, 1]
    ggml_tensor * get_comp(ggml_context * ctx, int32_t il_src) const;

    // rows [n_row, n_blocks] -> the stream's rows idxs (I64)
    ggml_tensor * cpy_comp(ggml_context * ctx, ggml_tensor * rows, ggml_tensor * idxs, int32_t il_src) const;

    // compressor state ring of the stream: [n_embd_head, ratio]
    ggml_tensor * get_state_kv   (ggml_context * ctx, int32_t il_src) const;
    ggml_tensor * get_state_score(ggml_context * ctx, int32_t il_src) const;
    ggml_tensor * cpy_state_kv   (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il_src) const;
    ggml_tensor * cpy_state_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il_src) const;

private:
    llama_kv_cache_dsv41 * kv41 = nullptr;
    bool full = false;

    size_t i_cur = 0;
    std::vector<llama_ubatch> ubatches41;

    // [n_ubatch][ratio - 1]
    std::vector<std::array<llama_dsv41_plan, 2>> plans;

    mutable std::array<llama_dsv41_plan, 2> plans_reserve;
};
