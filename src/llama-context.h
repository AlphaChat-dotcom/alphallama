#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

// Pinned host allocator for the per-decode hidden capture (mtp_fill_h), so a prompt chunk's rows
// (hc*n_embd floats each) do not round-trip through pageable memory around the head-KV fill.
// Enabled by ALPHA_MTP_PINNED_H (default 1). Falls back to plain malloc without a GPU host buffer type.
template <typename T> struct llama_pinned_alloc {
    using value_type = T;
    llama_pinned_alloc() = default;
    template <class U> llama_pinned_alloc(const llama_pinned_alloc<U> &) {}
    T *  allocate(size_t n);
    void deallocate(T * p, size_t n);
    template <class U> bool operator==(const llama_pinned_alloc<U> &) const { return true; }
    template <class U> bool operator!=(const llama_pinned_alloc<U> &) const { return false; }
};

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();
    void force_reserve() { sched_need_reserve = true; sched_force_new = true; sched_reserve(); }
    void release_sched() { synchronize(); sched.reset(); sched_need_reserve = true; sched_force_new = true; }
    std::map<uint32_t, size_t> reserved_dev_bytes;   // device compute bytes of the last reserve at each n_ubatch

    // Invalidate graph reuse cache so next llama_decode rebuilds the graph.
    // Lightweight: does not recreate the scheduler or re-reserve buffers.
    // Used by layer-skip self-spec to force graph rebuild when layer range changes.
    void invalidate_graph() { gf_res_prev->reset(); }

    // Set/clear layer skip for self-spec. Changes n_layer_draft in cparams
    // which makes allow_reuse return false, so the graph is rebuilt on next decode.
    void set_draft_layers(int n_layer_draft) { cparams.n_layer_draft = n_layer_draft; }
    void clear_draft_layers() { cparams.n_layer_draft = 0; }
    // Middle-band skip: bitmask (bit i = skip layer i). In the graph-reuse key (draft_skip_mask),
    // so toggling it rebuilds the graph: draft (skip) and verify (full) never share a cached graph.
    void set_draft_skip_mask(uint64_t mask) { cparams.draft_skip_mask = mask; }

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    void set_eval_callback(ggml_backend_sched_eval_callback cb, void * user_data);
    // Swap the active scheduler with the second (draft) one, which has its own compute buffer, so a
    // draft decode does not clobber the main decode's graph-reuse cache. O(1) unique_ptr swap; all
    // decode sites use the active `sched`.
    void set_draft_active(bool enable);

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    // Argmax of the MTP draft logits produced in-graph by the last single-token decode
    // (res->t_mtp_logits). Returns LLAMA_TOKEN_NULL if ALPHA_MTP is off or the head is not loaded.
    // e_tok is unused (the embedding was baked into the decode graph); kept for API stability.
    llama_token mtp_draft(llama_token e_tok);
    // Model-built MTP draft route (LLM_GRAPH_TYPE_DECODER_MTP), for architectures whose MTP block
    // mtp_draft's hand-built graph cannot express. Reached only when alpha_arch_has_mtp_graph() opts
    // the architecture in; with nothing opted in it aborts rather than silently disabling speculation.
    // The contract is documented at its definition in llama-context.cpp.
    llama_token mtp_draft_graph(llama_token e_tok);
    // qwen4exp (hyper-connection) MTP block, see llama-context-mtp4.cpp
    llama_token mtp_draft_qwen4exp(llama_token e_tok);
    bool        mtp_kv_fill_qwen4exp(const std::vector<const float *> & hsrc, const std::vector<llama_token> & ctok,
                                     const std::vector<llama_pos> & cpos, int32_t slot0, int64_t m);
    // Append the head-KV fill of a prompt chunk to the chunk's own graph (ALPHA_MTP_FILL_INGRAPH, qwen4exp)
    bool        mtp_ingraph_append(const llama_ubatch & ub, llm_graph_result * res, ggml_cgraph * gf);
    void        mtp_ingraph_set_inputs();
    void        mtp_ingraph_commit();
    struct mtp_ingraph_t;
    bool        mtp_ingraph_build(llm_graph_result * res, ggml_cgraph * gf, ggml_tensor * E, int64_t n, bool use_prev,
                                  int32_t slot0, int64_t m, mtp_ingraph_t & ig);
    void        mtp_ingraph_reserve(llm_graph_result * res, ggml_cgraph * gf, int64_t n);
    // width of the exported MTP hidden: n_embd, or hc*n_embd for qwen4exp (its residual streams)
    int64_t     mtp_hw() const;
    void        mtp_kv_fill();
    bool        mtp_kv_ensure(int64_t kv_row, ggml_backend_buffer_type_t buft);
    // allocate the MTP head's KV now (before a fused-infer expert pool takes the free VRAM); false if no MTP head
    bool        mtp_kv_prealloc();

    // Fold the recurrent state forward by exactly `m` accepted rows of the verify that was just
    // recorded (MTP/ReplaySSM), instead of re-laying the prefix through a full decode.
    // Returns false if there is no usable recording; the caller must then run the relay, because
    // record mode left the committed state at S_0. See alpha_gdn_record.h.
    bool        gdn_fold(int32_t m);

    // Re-point the carried hidden at the last accepted row rather than the last drafted one.
    // The decode path cannot know acceptance, so the caller supplies it once the verify has been
    // compared. Without this, the next window's draft starts from a hidden conditioned on rejected
    // tokens.
    void        mtp_select_row(int32_t row);

    // Token for output row `i` from the in-graph argmax (a few bytes read back) instead of a
    // full-logits D2H plus a host scan over the vocab. LLAMA_TOKEN_NULL if absent.
    llama_token mtp_argmax_ith(int32_t i) const;

    // Top-1/top-2 logit values from the last mtp_draft's top-2 argsort (ALPHA_MTP_MARGIN).
    // Their difference is the head's confidence in the token it is about to chain from, which the
    // adaptive-depth gate early-stops on. Costs one extra 8-byte D2H on a sync the argmax readback
    // already forces.
    float mtp_get_margin() const { return mtp_top2_v[0] - mtp_top2_v[1]; }
    float mtp_get_conf() const { return mtp_conf2[0]; }

    // Peek/poke the carried hidden the MTP head drafts from (diagnostic).
    //
    // Decoding accepted drafts gives both a refreshed h_t and real KV for the accepted prefix.
    // Restoring the trigger-time h_t before each head call, while still decoding the accepted drafts,
    // separates the two effects:
    //     (true-hidden      - stale-h+real-KV) = the hidden-state term alone
    //     (stale-h+real-KV  - carried-h      ) = the KV term alone
    const std::vector<float> & mtp_peek_hidden() const { return mtp_hidden_host; }
    void mtp_poke_hidden(const std::vector<float> & h) { mtp_hidden_host = h; }

    // Carry the head's boundary row across requests (ALPHA_HYBRID_PREFIX). A recurrent checkpoint at
    // position ckpt restores the target, but the head-KV column for ckpt is built from
    // (h_{ckpt-1}, token_ckpt) and the next request's token_ckpt may differ, so that column must be
    // rebuilt, which needs h_{ckpt-1}, a row no later decode reproduces. export: copy the sequence's
    // carried row if it is exactly the row for pos (false otherwise). import: stage it so the next
    // fill treats the suffix as a continuation from ckpt-1 (rebuilding column ckpt) instead of a
    // rewind (which keeps the stale column and drops the carried row).
    bool mtp_seq_export(int32_t seq, int32_t pos, std::vector<float> & h);
    bool mtp_seq_import(int32_t seq, int32_t ckpt, const std::vector<float> & h);
    // FNV-1a over the sequence's head-KV columns [lo, hi] (K then V), for a state-equality check
    // against a full reprocess (ALPHA_MTP_KV_HASH). Returns 0 when MTP is inactive or the range is empty.
    uint64_t mtp_seq_kv_hash(int32_t seq, int32_t & lo, int32_t & hi);

    // Declare that the next decode's logits will not be read, so the D2H can be skipped.
    // Ignored unless the in-graph argmax is present. Caller must clear it afterwards.
    void set_skip_logits(bool v) { skip_logits_d2h_ = v; }

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    // UBBoost: change the micro-batch size at runtime (re-reserves compute buffers).
    // Returns the applied value; rolls back to the old size on allocation failure.
    int32_t set_n_ubatch(uint32_t value);
    void    set_n_ubatch_lazy(uint32_t value) {
        value = std::max(1u, std::min(value, cparams.n_batch));
        if (cparams.n_ubatch != value) { cparams.n_ubatch = value; sched_need_reserve = true; }
    }

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    // Second scheduler (own compute buffer) for the MTP draft. Swapped into `sched` by
    // set_draft_active(). Reserved at context init when ALPHA_MTP is on.
    ggml_backend_sched_ptr sched_alt;
    llm_graph_result_ptr   gf_res_prev_alt;
    llm_graph_result_ptr   gf_res_reserve_alt;
    bool                   draft_active_ = false;
    // Separate host logits sink for the second (draft) scheduler: routing the draft's copy here (and
    // reading it back through get_logits_ith while draft_active_) keeps it from clobbering logits.data.
    std::vector<float>     draft_logits_;
    // The draft's output bookkeeping (n_outputs/output_ids), swapped in by set_draft_active so a
    // draft decode on the second sched can't clobber the paused verify's rows.
    uint32_t               n_outputs_alt = 0;
    std::vector<int32_t>   output_ids_alt;

    bool sched_need_reserve = true;
    bool sched_force_new = false;  // when true, force_reserve creates new scheduler (defrag)

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // host copy of the exported pre-norm hidden H (n_embd) for the single decode token
    std::vector<float> mtp_hidden_host;
    // in-graph argmax of the last decode's logits, one entry per output row
    std::vector<int32_t> mtp_argmax_host;

    float       mtp_top2_v[2] = { 0.0f, 0.0f };   // top-1/top-2 logits (ALPHA_MTP_MARGIN)
    // Caller-declared "this decode's logits will not be read" (greedy MTP verify).
    // Only honoured when t_mtp_argmax exists to supply the tokens instead.
    bool skip_logits_d2h_ = false;
    bool logits_valid_    = true;
    // The draft head's own KV cache. Without it the mini-graph holds a single position, the causal
    // softmax is over one key, and chained drafts predict from a hidden with no memory of the tokens
    // they just drafted. The KV is filled across the whole prompt during prefill; the decode-side
    // increment lets the chain within one window attend to itself.
    struct ggml_context *        mtp_kv_ctx = nullptr;
    struct ggml_backend_buffer * mtp_kv_buf = nullptr;
    // The draft mini-graph is built into this context every call (ggml_reset, never freed) so its
    // tensors keep their addresses; with the KV write as set_rows(index input) and a padded attention
    // window + mask input, consecutive drafts are property-identical and the CUDA backend replays the
    // captured graph instead of launching ~40-55 kernels per draft. Enabled by ALPHA_MTP_HEAD_REPLAY.
    struct ggml_context *        mtp_draft_ctx = nullptr;
    // qwen4exp draft graph cache: its own scheduler, so the graph stays allocated between drafts
    // (the KV fill and the ReplaySSM fold share sched_alt and would otherwise force a rebuild every draft)
    ggml_backend_sched_ptr       sched_d4;
    struct d4_cache_t {
        ggml_cgraph * gf = nullptr;
        ggml_tensor * H = nullptr, * etok = nullptr, * rope = nullptr, * kv_idx = nullptr, * mask = nullptr;
        ggml_tensor * amax = nullptr, * top2v = nullptr, * R2f = nullptr, * conf = nullptr;
        int64_t       T_use = -2;
        size_t        woff = 0;
        const void *  kc = nullptr;
        bool          valid = false;
    } d4c;
    float mtp_conf2[2] = { 1.0f, 0.0f };   // last draft's top-1/top-2 softmax probabilities (ALPHA_MTP_CONF)
    struct ggml_tensor *         mtp_k_cache = nullptr;   // [n_hd*n_hkv, n_ctx+8]
    struct ggml_tensor *         mtp_v_cache = nullptr;
    // Columns are indexed by absolute position: slot j holds the head's column for sequence
    // position j+1 (the column at position p consumes h_{p-1} and the token at p, so p=0 has no
    // predecessor and there is no slot -1). Absolute indexing lets the cache be filled over the prompt.
    int32_t                      mtp_kv_hi   = -1;  // highest slot written; -1 = empty
    // Lowest valid slot. The cache holds the contiguous range [lo, hi]; -1 = empty.
    // Lets an empty cache begin its range wherever it first gets a usable fill, so a refused first fill
    // does not keep the head KV empty for the life of the process. Positions are RoPE'd absolutely, so
    // relative offsets inside [lo, hi] are correct; the head attends over a suffix of history.
    int32_t                      mtp_kv_lo   = -1;  // lowest valid slot; -1 = empty
    // The fields above are per context, i.e. single-sequence. With n_seq_max > 1 the scheduler
    // batches tokens from several sequences into one decode, so mtp_fill_pos holds a mixed batch and
    // mtp_kv_hi would be driven backwards by whichever sequence filled last; two sequences at the
    // same position would also collide in the position-indexed cache.
    //
    // The scalars stay as the active sequence's working set; mtp_switch_seq parks and restores them,
    // and each sequence gets its own stride in the cache.
    struct mtp_seq_state {
        int32_t   kv_hi      = -1;
        int32_t   kv_lo      = -1;
        int32_t   chain_step = 0;
        int32_t   pos_base   = -1;
        llama_pos prev_pos   = -1;
        std::vector<float> prev_h;
        std::vector<float> hidden_host;
    };
    std::vector<mtp_seq_state>   mtp_seq;             // indexed by seq_id
    int32_t                      mtp_cur_seq   = 0;   // whose scalars are live right now
    int32_t                      mtp_kv_stride = 0;   // columns per sequence in the K/V cache

public:
    // Public because the llama_mtp_set_seq C wrapper calls it; everything else here stays private.
    void    mtp_switch_seq(int32_t seq);
private:
    void    mtp_kv_fill_all();          // drive mtp_kv_fill once per sequence in the batch
    // Per-sequence carried-hidden / rewind / pos_base / chain-step bookkeeping,
    // driven from that sequence's own rows by mtp_kv_fill_all.
    void    mtp_seq_bookkeep(int32_t first, int32_t last, const float * last_h, int64_t n_embd);
    int32_t mtp_kv_base() const { return mtp_cur_seq * mtp_kv_stride; }
    // plan of the fill the current graph carries (ALPHA_MTP_FILL_INGRAPH, see mtp_ingraph_append)
    // Public to match its forward declaration above (clang rejects a redeclaration that changes access).
public:
    struct mtp_ingraph_t {
        bool active = false;                    // the last built graph fills the head KV itself
        bool built  = false;                    // the graph held in gf_res_prev carries fill nodes (no reuse)
        int32_t slot0 = 0;
        int64_t m     = 0;
        std::vector<ggml_tensor *>          ix, ps;      // per block inputs
        std::vector<std::vector<int32_t>>   ix_h, ps_h;
        ggml_tensor *       hprev = nullptr;
        std::vector<float>  last_row;
    };
private:
    mtp_ingraph_t                mtp_ig;
    std::vector<llama_seq_id>    mtp_fill_seq;        // seq of each captured row

    int32_t                      mtp_chain_step = 0;  // drafts issued in the current window
    int32_t                      mtp_kv_cols = 0;   // allocated columns (>= n_ctx)
    // Absolute sequence position of the token whose hidden H is currently held in
    // mtp_hidden_host, i.e. the trigger decode's position. The draft column for chain step s sits
    // at mtp_pos_base + 1 + s, the position of the token the head embeds. Needed because the
    // head's attention is rotary: without a real position its q/k carry no positional signal.
    // -1 means "no usable H", which makes mtp_draft skip RoPE rather than rotate by garbage.
    int32_t                      mtp_pos_base     = -1;
    // Every row of the last decode, kept so the head's KV can be filled in one batched pass.
    // The head's K and V at a column depend only on (h_{p-1}, token_p), not on its own attention,
    // so the whole prompt's columns are computable in parallel.
    std::vector<float, llama_pinned_alloc<float>> mtp_fill_h;   // [n_embd * n] final hidden, one row per token (pinned)
    std::vector<llama_token>     mtp_fill_tok;
    std::vector<llama_pos>       mtp_fill_pos;
    // The column at position p needs (h_{p-1}, token_p), a pair that straddles two decodes at
    // the boundary: h_{p-1} is the last row of the decode that just ran, but token_p is only known
    // after it is sampled. Carrying the last row forward closes that one-column gap; without it
    // every window would leave slot p-1 unwritten.
    std::vector<float>           mtp_prev_h;
    llama_pos                    mtp_prev_pos = -1;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
