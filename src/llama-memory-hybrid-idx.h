#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache holding one indexer key per token, for the block-sparse
// attention (QSA) that qwen4exp uses above its top_k budget.
//
// The indexer is a side buffer over the attention cells: same size, padding, streams and slot
// assignment, so cell j is the same token in both. That invariant is what lets build_qsa_top_k
// score blocks of attention cells using indexer keys - it is enforced in init_batch by handing
// the indexer the attention cache's own slot layout rather than letting it pick its own.
//
// Ported from upstream llama.cpp 6c84c7d5d (PR #27742), adapted to this llama_memory_hybrid.

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx  = nullptr,
                            /* indexer key geometry */
                ggml_type   type_idx    = GGML_TYPE_F16);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    // null when the model has no indexer tensors
    llama_kv_cache * get_mem_idx() const { return mem_idx.get(); }

    // fills the per-block description QSA needs; see the implementation for the tensor shapes
    void set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const;

    // ALPHA_QSA_PKC: pooled-key cache. A block's pooled indexer key (mean of its `ratio` raw keys, normed, rotated
    // at the block's position) only changes when one of its cells is written, so it is kept per position bucket in
    // the indexer cache's V rows (otherwise unused; laid out as idx_dim f32 per cell) and a decode-sized ubatch
    // recomputes only the stale blocks. Valid for one stream holding one sequence.
    //   qsa_pkc_plan  -> -1 when the cache cannot be used, else the number of stale full blocks (listed in `dirty`,
    //                    their member cells in `members`, r per block); the table is not modified
    //   qsa_pkc_commit   records `pbs` as fresh with the given members
    int32_t qsa_pkc_plan(const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv,
                         std::vector<int32_t> * dirty, std::vector<int32_t> * members) const;
    void    qsa_pkc_commit(uint32_t ratio, const std::vector<int32_t> & pbs, const std::vector<int32_t> & members) const;
    void    qsa_pkc_invalidate(uint32_t ratio = 0) const { if (ratio) pkc_sig.erase(ratio); else pkc_sig.clear(); }
    bool    qsa_pkc_ok() const { return pkc_ok; }

    // glm5-next: pool size of the k-pool indexer (0 for every other architecture). With a k-pool the
    // indexer cache row of a token holds key | gate (2 * indexer_head_size), see llm_build_glm5_next.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

private:
    // a modified copy of the model's hparams (MQA, one key head of indexer_head_size, rope NONE).
    // must be declared before mem_idx: the cache holds a reference to it.
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // ALPHA_QSA_PKC: per ratio: the member cells each position bucket's cached key was computed from (-1 = stale)
    mutable std::map<uint32_t, std::vector<int32_t>> pkc_sig;
    mutable std::vector<int32_t> pkc_scratch;
    bool pkc_ok = false;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize);

    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      slot_info_vec_t   sinfos_attn,
                      slot_info_vec_t   sinfos_idx,
            std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    bool next()  override;
    bool apply() override;

    // null when the model has no indexer tensors - callers must check
    const llama_kv_cache_context * get_idx() const;

    // streams in this batch; build_qsa_top_k needs it to split blocks per stream
    uint32_t get_n_stream() const;

    // fills the per-block description QSA needs; see the implementation for the tensor shapes
    void set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const;

    // ALPHA_QSA_PKC: see llama_memory_hybrid_idx
    int32_t qsa_pkc_plan(const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv) const;

    //
    // glm5-next k-pool indexer (ported from upstream llama.cpp 649dcb103, simplified: the pool layout is
    // rebuilt from the cells on every ubatch and every pool is re-pooled, there is no pooled-key cache)
    //
    // A pool is `kpool` consecutive positions of one sequence, counted from the sequence's first position.
    // Only complete pools are scored; the incomplete tail of a token's own pool is always selected.

    // padded pool count of the current ubatch (the last padded pool is always unused);
    // for the full (reserve) context: the capacity
    uint32_t get_n_kpool() const;

    // the indexer cache's whole K tensor of layer il: [2*indexer_head_size, kv_size, n_stream]
    ggml_tensor * get_kpool_storage(int32_t il) const;

    //   pool_gcells [kpool, n_pool]       I32  member cells as rows of the storage (stream*kv_size + cell), for the gather
    //   pool_idxs   [kpool, n_pool]       I32  member cells as stream-local mask rows; n_kv for the padded pools
    //   pool_mask   [n_pool, n_tokens]    F32  0 where the pool belongs to the token's sequence and ends at or before it
    //   tail_idxs   [kpool - 1, n_tokens] I32  cells of the token's incomplete pool; n_kv where there is none
    void set_input_kpool(ggml_tensor * pool_gcells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask,
                         ggml_tensor * tail_idxs, const llama_ubatch * ubatch) const;
    void    set_input_qsa_pkc(int mode, ggml_tensor * pk_map, ggml_tensor * pk_upd, ggml_tensor * pk_upd_cells,
                              ggml_tensor * pk_upd_pos, const llama_ubatch * ubatch, uint32_t ratio, int64_t n_kv) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;
    bool is_update = false;
    bool is_full   = false;

    // glm5-next k-pool layout of the sequences in the current ubatch, built in apply()
    struct kpool_seq {
        llama_seq_id id      = 0;
        uint32_t     strm    = 0; // stream holding this sequence's cells
        llama_pos    pos_min = 0;
        std::vector<std::pair<llama_pos, uint32_t>> cells; // (position, stream-local cell), sorted by position
        std::vector<uint32_t> pools;                       // index into cells of each complete pool's first member
    };
    std::vector<kpool_seq> kpool_seqs;
    uint32_t kpool_n_real = 0;
    bool     kpool_valid  = false;

    void kpool_build(const llama_ubatch & ubatch);

private:
    const llama_memory_context_ptr ctx_idx;
};
