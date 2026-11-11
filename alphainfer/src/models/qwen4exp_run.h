// SPDX-License-Identifier: Apache-2.0
// Qwen4ExpRun: the runtime of Program<Qwen4Exp>. The eager path (one stream, no rounds) streams routed experts
// through the ExpertPool with a host routing read per MoE layer and is the reference checked against the
// alphallama oracle (tools/ref_oracle.cpp, --cpu-experts); the doorbell path runs decode as captured rounds.
//
// Per chunk of T tokens at positions [pos0, pos0 + T):
//   host:   PLE row ids from the token history (n-gram hash, EOS-cut), rows gathered + dequantised from the mapped
//           table into f32 [N, T]; positions, KV write cells, causal mask
//   device: embed_hc -> per layer { [PLE] -> HC read -> GDN(sigmoid z) | attention(+QSA) -> HC scatter ->
//           HC read -> streamed MoE + gated shared expert -> HC scatter } -> HC head -> output -> logits f32 [V, T]
#pragma once

#include "core/expert_pool.h"
#include "core/host_mem.h"
#include "models/qwen4exp.h"
#include "engine/round_cache.h"
#include "offload/miss_pool.h"
#include "ops/ops.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

#include <cstdint>
#include <array>
#include <vector>

namespace ai {

static constexpr int64_t kDoorbellT = 16;  // max rows per doorbell window (decode / verify widths)

// [AI_Q4X_ROUND_TIMING] GPU event marks of the decode cycle (qwen4exp_run.cpp): 0/1 around the MTP chain
void q4x_rt_mark(int which, void * stream);

class Qwen4ExpRun {
public:
    // max_ctx cells of attention KV (one stream); expert_slots per MoE layer in the v1 pool (>= n_expert_used).
    Qwen4ExpRun(Qwen4Exp & prog, Device & dev, int64_t max_ctx, int expert_slots);

    // One prefill/decode chunk; tokens are appended to the history. Returns logits f32 [V, T] (workspace).
    Tensor chunk(const int32_t * tokens, int64_t T);
    // One decode token as a captured round (doorbell mode only). Returns logits f32 [V, 1].
    Tensor decode(int32_t token);
    // Speculative verify: run T <= kDoorbellT rows (the pending token + drafts) without committing the recurrent state
    // (GDN ssm/conv and PLE history keep their pre-round values; the records go to persistent buffers). The KV and
    // indexer cells are written at their positions and simply rewritten by later rounds. Must be followed by
    // commit(m1), 1 <= m1 <= T: the first m1 rows are accepted (ReplaySSM fold), the rest dropped.
    Tensor verify(const int32_t * tokens, int64_t T);
    // [AI_Q4X_VERIFY_EARLY] the draft chain, then the verify of pending + its n drafts launched right behind it (ids
    // from the chain's device output); false = not eligible, nothing done (use mtp_draft + verify)
    bool   draft_verify_early(int32_t pending, int n, std::vector<int32_t> & drafts, Tensor & lg);
    // [P5] MTP drafting (needs prog.load_mtp): call before the prefill. The head's K/V column c holds
    // f(trunk streams H_{c-1}, token c); columns are filled as rows commit (lazily, at the next call). mtp_draft
    // chains n drafts after `pending` (the token at position position()).
    void                 mtp_enable();
    // tau > 0: stop drafting once the product of the drafts' top-1 probabilities would fall below tau (the drafts
    // that pass are returned; probs, when given, receives each returned draft's probability)
    std::vector<int32_t> mtp_draft(int32_t pending, int n, float tau = 0.0f, std::vector<float> * probs = nullptr);
    // Layer-major prefill of n prompt tokens in doorbell mode (qwen4exp_prefill.cpp); returns the last sub-chunk's
    // logits, and every row's when all_logits is given (scoring). Falls back to kDoorbellT-row rounds if its buffers do not fit.
    // AI_Q4X_PF_WARM: one dummy prefill of n tokens at start, then reset + residency reset: the backend's memory pool
    // grows to the prefill's high-water now, while VRAM is free, instead of inside a later request (where cuMemCreate
    // can fail and ggml aborts). Returns false when skipped (n too small for the context).
    bool                 prefill_warmup(int64_t n);
    Tensor               prefill(const int32_t * tokens, int64_t n, std::vector<float> * all_logits = nullptr);
    void                 set_mtp_keep(int n) { mtp_keep_ = n < 1 ? 1 : n; }   // drafts never cut by tau (>= 1)
    // AI_DRAFT_VOCAB: the MTP draft head's rows (the verify head stays full): ids ascending -> a compact device copy
    // of those output rows (argmax index mapped back to the id); empty ids + prefix > 0 -> a view of the first `prefix`
    // rows (no copy); both empty -> the full head. Returns false when the copy does not fit.
    bool                 set_draft_rows(const std::vector<int32_t> & ids, int64_t prefix = 0);
    void   commit(int64_t m1);
    // Clear the history, KV and recurrent state (a new sequence).
    // Exact serving: put the expert residency back to the initial profile fill (admissions undone, scores cleared;
    // only the slots that changed are recopied), so a request's output does not depend on the requests before it.
    // Returns the slots recopied.
    int    reset_residency();
    void   reset();
    // Prefix reuse: save the recurrent state at the current position (host copy), one entry per position, kept sorted;
    // rewind to the longest saved position <= max_len (false: none). The attention KV needs no copy: it is masked by
    // position, and every saved position lies inside history(), so the KV below it is the history's. The committed
    // tokens: history(). checkpoint_len(): the longest saved position (-1: none). checkpoint_mid(true): prefill() also
    // saves at every super-chunk boundary inside a prompt (AI_Q4X_PREFIX_CKPT), so an edited prompt resumes from the
    // last boundary before its first changed token instead of from token 0.
    void                         checkpoint();
    bool                         restore_checkpoint(int64_t max_len);
    bool                         restore_checkpoint() { return restore_checkpoint(checkpoint_len()); }
    int64_t                      checkpoint_len() const { return cks_.empty() ? -1 : cks_.back().len; }
    int64_t                      checkpoint_at_most(int64_t max_len) const;
    size_t                       checkpoint_count() const { return cks_.size(); }
    void                         checkpoint_mid(bool on) { ck_mid_ = on; }
    const std::vector<int32_t> & history() const { return hist_; }

    int64_t position() const { return (int64_t) hist_.size(); }
    const ExpertPool & pool() const { return pool_; }
    const offload::MissPool * miss_pool() const { return miss_.get(); }
    void reset_pool_stats() { if (miss_) miss_->reset_stats(); }

    // AI_Q4X_DUMP=<dir>: per-layer tensors under the oracle's names (hc_init, l_last-N, ffn_out-N, result_norm),
    // in ref_oracle's format (i64 ne0, i64 rows, f32), for comparison against the alphallama dump.
    void   set_dump_dir(const std::string & dir) { dump_dir_ = dir; }

private:
    void   ple_rows(int64_t pos0, int64_t T, float * out) const;   // host: [N, T] f32
    void   ple_warm();   // AI_Q4X_PLE_WARM: read the PLE table into the page cache after the load's cache drop
    // [AI_Q4X_PLE_IO] the round's PLE row reads: de-duplicated, page-cache hits read inline, the rest queued with
    // WILLNEED readahead at once, then read inline (decode) or by blocking reader threads too (prefill) -- PleIo
    struct PleIo;
    mutable std::unique_ptr<PleIo> ple_io_;
    Tensor round(const int32_t * tokens, int64_t T, bool early = false);                // captured T <= 8 row round (decode / verify)
    Tensor forward(const Tensor & ids, const Tensor & pos4, const Tensor & cells, const Tensor & mask, const Tensor & ple,
                   int64_t T, int64_t n_kv);
    Tensor moe_streamed(int il, const Tensor & x);                 // routed (pool) + gated shared expert
    // AI_Q4X_DOORBELL=1: static device residency (a slot table per layer), misses computed by the CPU
    // pool through mapped-memory doorbells -- no host synchronisation inside the forward. Rows <= kDoorbellT.
    Tensor moe_doorbell(int il, const Tensor & x);
    Tensor moe_prefill(int il, const Tensor & x);                  // resident bank + streamed scratch bank
    bool   prefill_alloc(int64_t S, int64_t sub);
    size_t pf_state_bytes(int64_t S, int64_t sub) const;   // prefill_alloc's state buffer (residuals, PLE, ids, masks)
    int    pf_cap(int il) const;   // the layer's prefill scratch capacity: its non-resident experts (all of them when lent)
    void   lend_begin(size_t used = (size_t) -1);   // [AI_Q4X_LEND] the lent layers' banks become the prefill's buffers (their experts non-resident)
    void   lend_end();             // ... and are copied back (the same experts) once the prefill's work has finished
    void   init_doorbell(int slots);
    void   dump(const char * name, int il, const Tensor & t);
    void   keep_record(int il, op::GdnRecord * gr, const Tensor * ple_padded);   // copy a layer's record out of the scope
    void    mtp_flush();                                   // run the pending head-K/V fill (after the caller read logits)
    int32_t mtp_layer(const Tensor & H, const int32_t * toks, int64_t p0, int64_t T, bool full, float * p_top = nullptr,
                              bool dev_mask = false);
    void    mtp_ingress(int64_t p0, int64_t T, int64_t n_kv, int32_t * pos4, int64_t * cells, uint16_t * mask) const;
    bool    mtp_mapped() const;
    Tensor  mtp_build(const Tensor & H, const Tensor & ids_t, const Tensor & pos4_t, const Tensor & cells_t,
                      const Tensor & mask_t, int64_t T, int64_t n_kv, bool full);
    void    mtp_chain(int32_t pending, int n, int32_t * toks, float * probs, bool fetch = true,
                      int off = 0);   // [AI_Q4X_MTP_CAPTURE]; off > 0: continue a chain after its first off steps
    static constexpr int kMtpChain = 8;
    RoundCache<int64_t, Tensor> mtp_rounds_;   // captured fill + draft chains, keyed by (F, fill rows, n, mapped)
    Tensor  mc_ids_, mc_pos4_, mc_cells_, mc_mask_, mc_tok_, mc_p_;
    Tensor  mc_sids_[kMtpChain], mc_spos4_[kMtpChain], mc_scell_[kMtpChain], mc_smask_[kMtpChain];
    bool          mtp_on_ = false;
    int64_t       mtp_fill_p0_ = -1, mtp_fill_m_ = 0;     // pending fill: columns p0..p0+m-1
    int           mtp_buf_ = -1;
    op::KvLayer   mtp_kv_;
    Tensor        mtp_H_;     // f32 [N*hc, 9]: slot 0 = H of the row before the last forward, slots 1..T its rows
    Tensor        mtp_R2_;    // f32 [N*hc, 1]: the last draft's streams (the next chained draft's H)
    int                  draft_buf_ = -1;   // AI_DRAFT_VOCAB: compact draft head rows (set_draft_rows)
    Tensor               draft_head_;       // invalid = no compact copy (draft_prefix_ or the full head)
    Tensor               draft_map_dev_;    // draft_map_ on the device (the captured chain maps on the device)
    std::vector<int32_t> draft_map_;        // compact row -> token id
    int64_t              draft_prefix_ = 0, draft_head_cap_ = 0;
    std::vector<uint8_t> draft_host_;       // [AI_Q4X_DRAFT_HEAD_TYPE] the whole output head requantised, on the host
    int                  draft_host_type_ = -1;
    char *               mc_pin_ = nullptr;   // [AI_Q4X_MTP_ASYNC_UP] pinned mirror of the chain's ingress buffer

    bool                       rec_ = false;     // forward() in record mode (verify)
    int64_t                    rec_pos0_ = -1, rec_T_ = 0;
    int                        rec_buf_ = -1;
    // per row count T (1..kDoorbellT), per layer: the persistent GDN record and PLE padded history. Addresses are fixed
    // per field (sized for kDoorbellT rows), so a replayed round of width T writes exactly where grec_[T] / prec_[T]
    // point. Sized from kDoorbellT so every verify width has its own entry.
    std::vector<op::GdnRecord> grec_[kDoorbellT + 1];
    std::vector<Tensor>        prec_[kDoorbellT + 1];
    std::vector<size_t>        rec_off_;         // per layer: offset of its record area in rec_buf_
    // [AI_Q4X_FOLD_BATCH] per verify width T: every layer's fold as device tables (kern::FoldGdn / FoldCopy), built on
    // the first commit of that width; state 1 = usable, -1 = a layer the batch does not take (per-layer path)
    bool fold_batch_commit(int64_t m1);
    int     fold_buf_ = -1;
    char *  fold_base_ = nullptr;
    float * fold_scr_ = nullptr;
    int     fold_state_[kDoorbellT + 1] = {};
    int     fold_ng_[kDoorbellT + 1] = {}, fold_nc_[kDoorbellT + 1] = {};
    size_t  fold_off_[kDoorbellT + 1] = {};
    // [AI_Q4X_FOLD_SIDE] the batched fold on its own stream, overlapping the next MTP draft chain (which reads no GDN /
    // PLE state); everything that next touches that state joins it first (fold_join: the device waits; host = true:
    // the host waits too, for host reads / uploads of the state)
    void   fold_join(bool host = false);
    // [AI_Q4X_RESERVE_CORES] the launch core goes to the thread that decodes (first round / chunk / prefill call of a
    // new thread), the other host threads to host_cpus_
    void                 pin_launch_thread();
    std::vector<int>     launch_cpus_, host_cpus_;
    std::thread::id      pinned_thread_{};
    void * fold_stream_ = nullptr;
    void * fold_ev_fork_ = nullptr;
    void * fold_ev_join_ = nullptr;
    bool   fold_pending_ = false;

    Qwen4Exp &   prog_;
    Device &     dev_;
    int64_t      max_ctx_ = 0;
    ExpertPool   pool_;
    int          state_buf_ = -1;

    std::vector<op::KvLayer>          kv_;    // per layer (valid on attention layers)
    std::vector<op::q4x::QsaCache>    qsa_;
    std::vector<op::GdnState>         gdn_;   // per layer (valid on GDN layers)
    std::vector<op::q4x::PleState>    ple_;   // per layer (valid on PLE layers)
    op::AttnParams                    ap_{};
    op::GdnParams                     gp_{};
    op::q4x::PleParams                pp_{};

    std::vector<int32_t> hist_;            // committed tokens of the sequence
    std::vector<Tensor *>          recurrent_tensors();
    struct Ckpt {
        int64_t                        len = -1, fill_p0 = -1, fill_m = 0;
        std::vector<std::vector<char>> buf;
    };
    std::vector<Ckpt>              cks_;   // ascending len, all inside hist_
    bool                           ck_mid_ = false;
    // doorbell mode
    struct SlotBank {
        Tensor gate, up, down, table;
        int    n_slots = 0;
        std::vector<int32_t> table_h;    // host mirror: expert -> slot or -1
        std::vector<int32_t> owner;      // slot -> expert (-1 free / pending)
        std::vector<float>   score;      // expert -> decayed use count
    };
    // Admission, after each decode round: count every routed expert (hits and misses) from the bells, then copy the
    // hottest missed experts into the coldest slots on a copy stream (via pinned staging) and publish them in the table
    // once the copy has landed. A victim is unmapped before the next round launches, so no round reads a slot in flight.
    void   admit_after_round();
    void   admit_drain();   // publish every in-flight admission (before a prefill)
    std::vector<std::vector<int32_t>> init_owner_;   // per layer: the initial fill's expert per slot (reset_residency)
    static float mtp_minp();   // AI_Q4X_MTP_MINP
    // job j uses staging buffer / event j % staging_.size(); recorded_[si] = j once the worker has queued its copies
    struct Pending { int layer, slot, expert; int64_t job; };
    std::vector<Pending> pending_;
    // Admission pipeline: the launch thread picks victims and enqueues; a worker stages the expert from the
    // host mapping into pinned memory and queues the H2D copies (nothing synchronous on the launch thread)
    std::thread                             admit_thr_;
    std::mutex                              admit_mu_;
    std::condition_variable                 admit_cv_;
    std::deque<Pending>                     admit_q_;
    std::vector<Pending>                    admit_defer_;   // AI_Q4X_ADMIT_DEFER: jobs held until the round is launched
    bool                                    admit_stop_ = false;
    void                                    admit_kick();   // hand the deferred jobs to the worker
    std::unique_ptr<std::atomic<int64_t>[]> recorded_;
    int32_t *                               table_pin_ = nullptr;   // pinned mirror of every layer's table [L][E]
    void   admit_worker();
public:
    ~Qwen4ExpRun();
private:
    void *               copy_stream_ = nullptr;
    std::vector<char *>  staging_;       // pinned staging buffers (one per in-flight admission), slices of staging_mem_
    PinnedHost           staging_mem_;
    std::vector<void *>  staging_ev_;
    uint64_t             n_admit_ = 0, n_publish_ = 0;
    // the bells hold a round of this sequence: prefill does not ring them, so without this the first decode after a
    // prompt would count the previous request's last round into the admission scores
    bool                 bells_live_ = false;
    std::atomic<uint64_t> n_staged_{ 0 }, n_direct_{ 0 };   // admission copies staged / straight from the RAM tier
    struct TierBuf { Tensor * tensor; void * orig; };
    std::vector<TierBuf> tier_;         // RAM tier: tensors re-pointed into tier_mem_ (and what they pointed at)
    PinnedHost           tier_mem_;     // one block for every tiered expert tensor
    std::vector<char>    tier_layer_;
    // GPU miss share: per tiered layer, device pointers of its gate / up / down expert tensors in the mapped tier
    std::vector<std::array<const void *, 3>> tier_dev_;
public:
    uint64_t admitted() const { return n_admit_; }
    uint64_t admit_direct() const { return n_direct_.load(); }
    uint64_t admit_staged() const { return n_staged_.load(); }
private:
    std::vector<SlotBank>                   banks_;
    bool                                    fillers_ = true;   // AI_Q4X_FILLERS: k filler rows per layer bank
    int                                     mtp_keep_ = 1;   // mtp_draft: the first n drafts are kept whatever tau says
    // layer-major prefill state
    int                  pf_buf_ = -1, pf_scr_ = -1, pf_layer_ = -1, pf_n_ = 0;
    int64_t              pf_S_ = 0, pf_sub_ = 0;
    // [AI_Q4X_LEND] layers lend_from_.. keep their slot banks in one buffer (lend_buf_) that a prefill borrows for its
    // state (offset pf_buf_off_) and expert scratch (pf_scr_off_): the prefill's ~2 GB need not stay free in decode.
    int                  lend_buf_ = -1, lend_from_ = 1 << 30;
    int                  lend_n_ = 0;   // [AI_Q4X_LEND_PART] layers lend_from_ .. lend_from_ + lend_n_ - 1 are lent now
    bool                 lent_layer(int il) const { return lent_ && il >= lend_from_ && il < lend_from_ + lend_n_; }
    size_t               lend_bytes_ = 0, pf_buf_off_ = 0, pf_scr_off_ = 0;
    size_t               pf_scr2_off_ = 0;   // [AI_Q4X_LEND2] the second expert scratch inside the lent buffer
    size_t               pf_scr3_off_ = 0;   // [AI_Q4X_PF_NBUF=3] the third, also inside the lent buffer
    int                  pf_nb_ = 1;         // scratch buffers in the layer-stream ring (1, 2 or 3)
    int                  pf_b(int il) const { return pf_nb_ > 1 ? il % pf_nb_ : 0; }
    bool                 lent_ = false;
    // [AI_Q4X_LEND_ASYNC] the give-back copies run on their own stream after the prefill returns (its first token is not
    // held for ~2 GB of copies); the lent layers stay non-resident (owners -1: no admission picks them) until
    // lend_publish sees the event and maps them again
    bool                 lend_back_ = false;
    void *               lend_stream_ = nullptr, * lend_ev_ = nullptr;
    void                 lend_publish(bool wait);
    std::vector<std::vector<int32_t>> lend_owner_;   // per lent layer: its slots' experts while lent
    size_t               pf_off_R_ = 0, pf_off_ple_ = 0, pf_off_ids_ = 0, pf_off_pos4_ = 0, pf_off_cell_ = 0, pf_off_mask_ = 0;
    std::vector<int32_t> pf_slot_of_;
    // AI_Q4X_PF_SHARE: the CPU share's cost model in seconds (A per CPU decode pass, G0 + C per copied expert on the
    // GPU), and timing events around the last shared batch's GPU part (read at the next batch)
    double               pf_share_A_ = 5e-5, pf_share_G0_ = -1.0, pf_share_C_ = 0.0;
    int                  pf_share_ngpu_ = 0, pf_share_il_ = 0;
    std::vector<double>  pf_share_Al_, pf_share_G0l_;   // per layer (-1 = no sample yet)
    bool                 pf_share_pending_ = false;
    void *               pf_ev_s0_ = nullptr, * pf_ev_s1_ = nullptr;
    uint64_t             pf_streamed_ = 0, pf_shared_cols_ = 0;   // expert copies streamed / columns the CPU share took
    // AI_Q4X_PF_STREAM: every non-resident expert of layer l+1 copied on a copy stream into the second scratch bank
    // while layer l computes (double-buffered; buffer il % 2 holds layer il)
    int                  pf_scr2_ = -1;
    bool                 pf_stream_on_ = false, pf_stream_now_ = false;   // available / used for this super-chunk
    void                 pf_bank(int il, int buf, Tensor & sg, Tensor & su, Tensor & sd, int n = 0, int which = 0) const;
    void                 pf_bank_b(int il, int b, Tensor & sg, Tensor & su, Tensor & sd) const;   // ring buffer b
    // [AI_Q4X_PF_SPARE] two spare buffers of pf_spare_P_ experts (when a second full scratch does not fit)
    int                  pf_spare_[2] = { -1, -1 }, pf_spare_P_ = 0, pf_spare_layer_[2] = { -1, -1 };
    void *               pf_ev_spfree_[3] = { nullptr, nullptr, nullptr };
    std::vector<int32_t> pf_spare_of_;
    void                 pf_stream_part(int il, int part);
    void *               pf_cstream_ = nullptr;
    void *               pf_ostream_ = nullptr, * pf_ev_ocopy_ = nullptr, * pf_ev_omain_ = nullptr;   // [AI_Q4X_PF_COPY_OVERLAP]
    // [AI_Q4X_PF_DEVPLAN] per-layer scratch / spare slot maps in mapped pinned memory, read by the device plan kernel;
    // the device overflow flag (a bank needed more records than its cap)
    int32_t *            pf_dmap_pin_ = nullptr, * pf_dmap_dev_ = nullptr;
    int                  pf_dmap_n_ = 0;
    int *                pf_dplan_ovf_ = nullptr;
    void *               pf_ev_ready_[3] = { nullptr, nullptr, nullptr }, * pf_ev_free_[3] = { nullptr, nullptr, nullptr };
    int                  pf_buf_layer_[3] = { -1, -1, -1 };   // the layer each scratch buffer holds (or is being filled with)
    void                 pf_stream_layer(int il);         // queue layer il's non-resident experts into its buffer
    void                 pf_layer_begin(int il);          // the copies to queue at the start of layer il
    Tensor               moe_prefill_big(int il, const Tensor & fm, int64_t slice);   // AI_Q4X_PF_MOE_BIG
    // AI_Q4X_PF_ROUTE_OVERLAP: the routing read-back on its own stream into pinned memory while the shared expert runs;
    // the owner plans go up as ordered async copies from a pinned staging buffer
    void *               pf_rstream_ = nullptr, * pf_ev_route_ = nullptr;
    int32_t *            pf_sel_pin_ = nullptr, * pf_plan_pin_ = nullptr, * pf_plan_dev_ = nullptr;   // plan staging: mapped
    size_t               pf_sel_cap_ = 0, pf_plan_cap_ = 0;
    std::vector<std::vector<int32_t>> pf_nonres_;          // per layer: the non-resident experts, in scratch order
    std::unique_ptr<offload::MissPool>      miss_;
    Tensor                                  win_base_;   // u32 [1] device: the window sequence base
    bool                                    doorbell_ = false;
    RoundCache<int64_t, Tensor>             rounds_;     // captured decode rounds, keyed by the envelope F
    Tensor in_ids_, in_pos4_, in_cells_, in_mask_, in_ple_;   // fixed decode ingress (outside the workspace)
    // [AI_Q4X_PLE_GATE] the round's PLE rows read by a worker thread WHILE the round launches and its first layers run:
    // the worker writes them into mapped pinned memory and bumps a mapped flag; a gate kernel before the PLE layer waits
    // for this round's sequence (in_want_) and copies them into in_ple_. Off the host critical path between draft and
    // verify (3090: ple rows 0.8-1.2 ms + launch 0.45 ms, serial, before the GPU started).
    bool                    ple_gate_ = false, ple_gate_now_ = false;   // enabled / this forward is a gated round
    float *                 ple_map_ = nullptr, * ple_map_d_ = nullptr;
    unsigned *              ple_flag_ = nullptr, * ple_flag_d_ = nullptr;
    unsigned                ple_seq_ = 0;
    Tensor                  in_want_;
    std::thread             ple_thr_;
    std::mutex              ple_mu_;
    std::condition_variable ple_cv_;
    bool                    ple_job_ = false, ple_stop_ = false;
    int64_t                 ple_p0_ = 0, ple_T_ = 0;
    unsigned                ple_job_seq_ = 0;
    // [AI_Q4X_VERIFY_EARLY] the worker reads the chain's drafts into hist_ before the job's rows
    bool                    ple_early_job_ = false, ple_early_done_ = false;
    void *                  ple_ev_chain_ = nullptr, * ple_side_ = nullptr;
    int32_t *               draft_pin_ = nullptr;
    void                    ple_gate_init();
    void                    ple_gate_idle();   // the worker has finished its job (before hist_ changes or the next post)
    std::string          dump_dir_;
    bool                 dump_append_ = false;   // this chunk continues the sequence: dumps append rows
};

}  // namespace ai
