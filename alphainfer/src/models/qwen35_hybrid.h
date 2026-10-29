// SPDX-License-Identifier: Apache-2.0
// Program<Qwen35Hybrid>: the hand-written forward for Qwen3.5 / 3.6 hybrid models
// (gated DeltaNet + gated full attention, sparse MoE with a shared expert).
//
// Config supplies dimensions and layer kinds; it never supplies a topology.
#pragma once

#include "core/device.h"
#include "core/expert_pool.h"
#include "core/kv.h"
#include "core/weights.h"
#include "ops/ops.h"
#if defined(AI_DOORBELL)
#include "offload/cpu_ffn.h"
#include "offload/doorbell_moe.h"
#include <memory>
#endif

#include <algorithm>
#include <cstdint>
#include <string>
#include <array>
#include <vector>

namespace ai {

struct Qwen35Spec {
    std::string arch;
    int     n_layer = 0;
    int64_t n_embd = 0, n_vocab = 0;
    int     n_head = 0, n_head_kv = 0, head_dim = 0;
    int     n_rot = 0;
    int     sections[4] = {0, 0, 0, 0};
    float   rope_base = 0, rms_eps = 0;
    int     n_ctx_train = 0;
    int     n_expert = 0, n_expert_used = 0;
    int64_t ffn_length = 0;     // dense FFN width; 0 on a MoE artifact
    bool    dense = false;      // no expert_count in the metadata => plain FFN per block
    int     conv_k = 0, ssm_state = 0, ssm_groups = 0, ssm_dt_rank = 0, ssm_inner = 0;
    int     full_attn_interval = 4;
    std::vector<bool> is_gdn;   // per layer
    int     nextn = 0;          // MTP (next-n) blocks after the main stack

    int64_t conv_channels() const { return (int64_t) ssm_inner + 2LL * ssm_groups * ssm_state; }
    int     head_v() const { return ssm_inner / ssm_dt_rank; }
};

class Qwen35Hybrid {
public:
    // Both the MoE and the dense Qwen3.5-family artifacts declare one of these. Accepting "qwen35"
    // here does not mean dense runs: load_spec() refuses a dense artifact by name, because the dense
    // FFN is not implemented. The arch gate's job is to recognise the family; the refusal belongs
    // where the reason is known.
    static bool supports(const std::string & arch) { return arch == "qwen35moe" || arch == "qwen35"; }

    // offload_experts: routed-expert banks stay in host memory (streamed MoE assembly).
    bool load(Device & dev, const std::string & path, bool verbose, bool offload_experts = false);
    // n_streams regions of max_ctx cells each, and as many state slots.
    void alloc_state(Device & dev, int64_t max_ctx, int n_streams, int kv_type = F16);
    // Rounds of up to this many rows keep every layer's activations (no per-layer rollback);
    // verify rounds must fit (B * (k+1) rows). Set before sizing the workspace.
    void set_unscoped_rows(int64_t n) { unscoped_rows_ = std::max<int64_t>(16, n); }
    // Bound on one scoped record-mode round's GDN record arena at T rows (batch 1)
    size_t rec_arena_bytes(int64_t T) const;
    // max_rows: the widest recorded decode/verify round the engine will run (sizes the doorbell's t_max)
    void alloc_expert_pool(Device & dev, int n_slots, bool with_head, int64_t max_rows = 0);
    const ExpertPool * expert_pool() const { return pool_.n_slots() ? &pool_ : nullptr; }
#if defined(AI_DOORBELL)
    // AI_OFFLOAD=doorbell: the generic doorbell offload (offload/doorbell_moe.h) instead of the v1 pool
    const offload::DoorbellMoe * doorbell() const { return door_.get(); }
#endif
    bool offloaded() const {
#if defined(AI_DOORBELL)
        if (door_) return true;
#endif
        return pool_.n_slots() != 0;
    }
    // Resident doorbell slots per bank that fit `budget` device bytes (fillers and staging included);
    // -1 when the doorbell offload is not selected (AI_OFFLOAD=doorbell in an AI_DOORBELL build)
    int doorbell_slots_for(size_t budget, bool with_head) const;
    // AI_FFN_CPU (dense FFN layers on the CPU): rows a recorded round may have (0 = not active); prefill is eager
    int64_t cpu_ffn_tmax() const {
#if defined(AI_DOORBELL)
        return ffn_cpu_list_.empty() ? 0 : ffn_cpu_tmax_;
#else
        return 0;
#endif
    }
    // between rounds, after the previous round completed (doorbell admission); no-op otherwise
    void offload_after_round() {
#if defined(AI_DOORBELL)
        if (door_) door_->after_round();
#endif
    }
    // before a new request: doorbell residency back to its initial fill (no-op otherwise)
    void offload_reset_residency() {
#if defined(AI_DOORBELL)
        if (door_) door_->reset_residency();
#endif
    }
    // Streamed-MoE host steps: time blocked on the device (compute + queued copies) and host CPU time.
    // overlap_*: routed experts of a step that the same layer also used in its previous step.
    struct OffloadProf {
        double   wait_ms = 0, cpu_ms = 0;
        uint64_t steps = 0, overlap_hit = 0, overlap_all = 0;
    };
    const OffloadProf & offload_prof() const { return prof_; }
    // Can a round of `rows` rows run the fixed-shape (recordable) streamed MoE?
    bool streamed_fixed_ok(int64_t rows) const {
#if defined(AI_DOORBELL)
        if (door_) return rows <= door_->t_max();
#endif
        return pool_.n_slots() && (int64_t) spec_.n_expert_used * rows <= pool_.n_slots() && rows <= h_route_cap_;
    }

    const Qwen35Spec & spec() const { return spec_; }
    const WeightSet &  weights() const { return w_; }
    PagedKv &          kv() { return kv_; }
    const PagedKv &    kv() const { return kv_; }
    StateSlots &       slots() { return slots_; }

    // Bytes of workspace one forward needs for a chunk of `t_max` rows and `n_logits` logit rows.
    size_t workspace_bytes(int64_t t_max, int64_t n_logits) const;
    // Sizing before alloc_state, for Engine's automatic KV-type / context choice. Each returns the
    // bytes the corresponding allocation will take (same rounding), so the plan can be checked against what
    // was actually allocated afterwards.
    size_t workspace_bytes_for(int64_t t_max, int64_t n_logits, int64_t capacity_cells, int n_regions) const;
    static bool mix_scope_on();   // AI_MIX_SCOPE: prefill layers roll the mixer block back before the FFN (default on)
    static bool chunk_scope_on();   // AI_CHUNK_SCOPE: prefill chunks are scoped at any width (default on)
    int64_t kv_cache_bytes(int64_t max_ctx, int n_streams, int kv_type, size_t alignment) const;
    int64_t state_bytes(int n_streams, int64_t q8_rows_per_stream, size_t alignment) const;

    // Per-round inputs. All of them are data: nothing here is a graph key except the shapes
    // (B streams x L rows, and the visible-cell count n_kv), which select the round profile.
    // Rows are stream-major (row = b*L + t); stream b is KV region first_region + b and state
    // slot slot0 + b.
    struct Inputs {
        Tensor  ids;       // i32 [T]
        Tensor  pos4;      // i32 [4T]
        Tensor  cells;     // i64 [T]
        Tensor  mask;      // f16 [n_kv, L, 1, B]
        Tensor  last_id;   // i32 [B] = b*L + L-1 (only read when narrowing)
        // AI_HOST_EMBD: set only when token_embd is host-kept: the round's rows already gathered
        // into a q8_0 [E,T] block, with an iota id list. embed() then runs the same get_rows over
        // this instead of over the full table, which is why the result is bit-identical.
        Tensor  embd;      // q8_0 [E, T], or invalid
        Tensor  embd_iota; // i32 [T]
        // Precomputed input rows (image embeddings) that replace the token embedding of their rows: img f32
        // [E, R], and per contiguous run {row in this round, length, first row of img}. Empty for text rounds.
        Tensor  img;
        std::vector<std::array<int64_t, 3>> img_runs;
        int64_t T    = 0;
        int64_t n_kv = 0;
        int64_t B    = 1;
        int     first_region = 0;        // AI_CHUNK_SCOPE: a prefill chunk (Engine::stage_chunk) runs scoped at any width, so a short last chunk does
        // not keep every layer's activations the way a decode / verify round of <= unscoped_rows_ rows does
        bool    chunk = false;
    };

    // Host-side values for rows [row0, row0+L) of a T-row round (sequence bookkeeping only; no
    // device work). Buffers hold T, 4T, T and n_kv*T elements. Checks L cells against the region.
    void fill_inputs(int seq, const int32_t * tokens, int64_t L, int64_t pos0, int64_t n_kv,
                     int64_t row0, int64_t T, int32_t * ids, int32_t * pos4, int64_t * cells, uint16_t * mask);

    // RoPE positions apart from KV cells. A prompt with images fills one KV cell per row, but its image rows
    // carry 2-D M-RoPE positions and the whole image takes fewer positions than cells. Per stream: an override for
    // prompt rows 0..n-1 (section-major t|y|x|z, 4n entries, valid only during that stream's prefill) and a delta
    // for every later row (RoPE position = cell + delta). Both 0 / absent for text: positions == cells.
    void    set_pos_override(int seq, const int32_t * pos4, int64_t n);
    void    set_rope_delta(int seq, int64_t delta);
    int64_t rope_delta(int seq) const { return seq >= 0 && (size_t) seq < rope_delta_.size() ? rope_delta_[(size_t) seq] : 0; }

    // Visible-cell count for a chunk: exact, or padded (AI_KV_PAD) to reproduce ggml's padding.
    int64_t visible_cells(int64_t pos0, int64_t T) const;

    // One chunk of one sequence (records or launches the Ops; never touches host data).
    // Writes KV and recurrent state; returns the final residual [E, T] or [E, 1] (last row).
    // With `rec`, GDN layers publish nothing and keep what a later fold needs (speculative verify).
    struct GdnRecords {
        std::vector<op::GdnRecord> layers;   // indexed by recurrent-state layer
    };
    // Streams use state slots [in.first_region, + in.B).
    Tensor forward(Device & dev, const Inputs & in, bool all_logits, GdnRecords * rec = nullptr);

    // Commit columns t <= m[b] of every stream in every GDN layer (masked ReplaySSM fold;
    // m is device data, so one profile serves every acceptance pattern).
    void fold_masked(Device & dev, const Inputs & in, const GdnRecords & rec, const Tensor & colmask,
                     const Tensor & m);

    // ---- MTP head (Qwen3.6 next-n block) ----
    bool has_mtp() const { return mtp_.valid; }
    // Head inputs for T rows at positions [pos0, pos0+T): the head attends over cells 1..p
    // (position 0 never has a head column).
    void fill_head_inputs(int seq, int64_t L, int64_t pos0, int64_t n_kv, int64_t row0, int64_t T,
                          int32_t * pos4, int64_t * cells, uint16_t * mask);
    enum class HeadRows {
        None,       // fill the head's KV only
        All,        // a token for every row
        Last,       // a token for the last row (single stream)
        Accepted,   // a token for row m[b] of each stream
    };
    struct MtpOut {
        Tensor hidden;   // f32 [E, rows]  the head's output for the selected rows (next chain step's H)
        Tensor token;    // i32 [rows]     argmax of the head's logits
        Tensor logits;   // f32 [V_draft, rows] the draft head's logits (AI_MTP_CONF_LOG reads them; ws-lifetime)
        Tensor conf;     // f32 [rows] max-softmax of those logits; set only when AI_MTP_GATE is on
    };
    // ids: i32 [T] tokens to embed; H: f32 [E, T] hidden of the previous position per row.
    MtpOut mtp(Device & dev, const Tensor & ids, const Tensor & H, const Inputs & head_in, HeadRows rows,
               const Tensor & m = Tensor{});

    // Final norm + LM head over `hidden` rows -> f32 [V, rows].
    Tensor logits(Device & dev, const Tensor & hidden);

private:
    void   bind_layers();
    void   fuse_beta_alpha(Device & dev);
    Tensor moe_streamed(Device & dev, int pool_layer, const op::MoeWeights & w, const Tensor & x);
    // Fixed-shape streamed MoE (every routed expert of the round fits the slots at once): the
    // routing read and the page-in are one host step, so the rest records and replays.
    Tensor moe_streamed_fixed(Device & dev, int pool_layer, const op::MoeWeights & w, const Tensor & x,
                              const Tensor & resid = Tensor{});
    Tensor ffn_residual(Device & dev, int pool_layer, const op::MoeWeights & w, const Tensor & p,
                        const Tensor & r, int dump_layer);
    op::AttnInputs attn_inputs(const Inputs & in) const;
    void   bind_mtp();

    struct MtpHead {
        bool            valid = false;
        int             kv    = -1;   // attention store index of the head's KV
        Tensor          enorm, hnorm, eh_proj, shared_norm, attn_norm, post_norm;
        op::AttnWeights attn;
        op::MoeWeights  moe;
    };
    MtpHead mtp_;

    Qwen35Spec spec_;
    WeightSet  w_;
    PagedKv    kv_;
    // see set_pos_override / set_rope_delta
    struct PosOverride { const int32_t * pos4 = nullptr; int64_t n = 0; };
    std::vector<PosOverride> pos_ov_;
    std::vector<int64_t>     rope_delta_;
    void pos_at(int seq, int64_t cell, int32_t out[4]) const;   // the 4 RoPE components of cell `cell`
    StateSlots slots_;
    ExpertPool pool_;
#if defined(AI_DOORBELL)
    std::unique_ptr<offload::DoorbellMoe> door_;
    std::unique_ptr<offload::CpuFfn>      cpu_ffn_;
    int64_t                               ffn_cpu_tmax_ = 16;
#endif
    std::vector<int>                      ffn_cpu_layer_;   // layer -> index into ffn_cpu_list_, or -1
    std::vector<int>                      ffn_cpu_list_;    // AI_FFN_CPU: dense FFN layers kept in host memory
    Tensor ffn_cpu(Device & dev, int il, const op::MoeWeights & w, const Tensor & p);

    std::vector<int>             kv_layer_;    // layer -> attention store index (-1 for GDN)
    std::vector<int>             gdn_layer_;   // layer -> state index (-1 for attention)
    std::vector<Tensor>          attn_norm_, post_norm_;
    std::vector<op::AttnWeights> attn_;
    std::vector<op::GdnWeights>  gdn_;
    std::vector<op::MoeWeights>  moe_;
    op::AttnParams               attn_p_{};
    op::GdnParams                gdn_p_{};

    // AI_Q8_STAGE: per-layer persistent buffers for the post-norm activation and its q8_1 copy.
    // They cannot come from the round's workspace arena: that arena resets at every layer boundary,
    // so layer N+1's activation lands on layer N's address, and the staging registry is keyed on
    // that address and geometry -- a stale entry would read as fresh and the consumer would get the
    // previous layer's activation. A per-layer address also keeps the pointer baked into the captured
    // graph stable across rounds, which the registry requires.
    // `pre_normed`, when the merge kernel folded the next layer's input norm in, is that norm's
    // output -- the following layer uses it instead of launching its own rms_norm.
    Tensor              ffn_residual_norm(Device & dev, int pool_layer, const op::MoeWeights & w,
                                          const Tensor & p, const Tensor & r, int dump_layer,
                                          const Tensor * next_norm, const Tensor * norm_buf, float eps,
                                          Tensor & pre_normed, const Tensor * q8_in = nullptr,
                                          const Tensor * router_pre = nullptr, bool router_cmp = false);
    void                alloc_stage(Device & dev, int n_streams);
    // AI_BOUND_QKVZ_HALF: timing-only byte-halving of the qkvz projection. Output is
    // deliberately wrong; see the definition for why substitution, not deletion, is
    // required here (deleting it collapses expert selection and changes the work).
    void                bound_halve_qkvz(op::GdnWeights & g, int il);
    const Tensor *      bound_donor(int type, int64_t ncols, int64_t rows_needed) const;
    // Timing-only in-place requantisation for roles the donor trick cannot reach (head,
    // expert_down). A smaller type always fits inside the larger one's buffer, so no
    // allocation is needed. Output is deliberately degraded.
    bool                bound_requant_inplace(Device & dev, Tensor & t, int want,
                                              const char * role, int idx, bool announce);
    void                bound_apply_roles(Device & dev);
    void                build_draft_head(Device & dev);   // AI_MTP_DRAFT_Q
    void                router_to_f16(Device & dev);      // AI_ROUTER_F16
    std::vector<Tensor> stage_out_;
    std::vector<Tensor> stage_q8_;
    std::vector<Tensor> stage_q8_in_;
    std::vector<Tensor> stage_gdn_out_, stage_gdn_q8_;   // AI_GDN_GATE per-layer output-gate buffers
    op::GdnState        gdn_state(int g, int il, int slot, int B) {
        op::GdnState st{ slots_.conv(g, slot, B), slots_.ssm(g, slot, B), Tensor{}, Tensor{} };
        if (il < (int) stage_gdn_out_.size()) {
            st.gate_out = stage_gdn_out_[(size_t) il];
            st.gate_q8  = stage_gdn_q8_[(size_t) il];
        }
        return st;
    }   // q8_1 copy of the next layer's input norm (AI_FUSE_MERGENORM=4)
    // The next layer's input norm, produced by this layer's merge kernel. It is read after the
    // arena has been reset for the next layer, so it cannot live in the arena.
    std::vector<Tensor> norm_out_;
    op::MoeParams                moe_p_{};
    Tensor                       tok_embd_, out_norm_, output_;
    Tensor                       embd_mapped_;   // AI_HOST_EMBD: the table in pinned host memory, read by the Ops over PCIe
    Tensor                       draft_head_;   // AI_MTP_DRAFT_Q: lower-bit copy of the draft rows
    // AI_TAPS=l0,l1,...: copy these layers' outputs (the post-MLP residual, `y`) into one persistent buffer,
    // F32 [n_taps*E, taps_cap_], tap i in rows [i*E, (i+1)*E), column t = row t of the round. For a block
    // drafter that conditions on target hidden states (e.g. DFlash2 taps 5,19,33,47,61). Fixed address,
    // so captured rounds stay valid; a round wider than taps_cap_ is refused, never truncated.
    std::vector<int>             tap_slot_;     // per layer: tap index or -1
    Tensor                       taps_;
    int64_t                      taps_cap_  = 0;
    int64_t                      taps_rows_ = 0;   // rows written by the last forward
    void                         init_taps(Device & dev);
    // AI_DRAFT_VOCAB: every vocab row in the draft type, on the host (no VRAM), and the device row -> id map the
    // draft argmax reads; set_draft_rows() swaps the draft head's rows per request (engine/draft_vocab.h)
    std::vector<char>            draft_full_host_;
    Tensor                       draft_map_;
    int64_t                      draft_rowb_ = 0;
    bool                         draft_identity_ = true;
    std::vector<int32_t>         draft_ids_;     // the ids now resident in draft_head_ (empty while identity)
    std::vector<char>            draft_stage_;   // reused staging for changed runs (never value-initialised per call)

  public:
    const Tensor & tok_embd() const { return tok_embd_; }
    // AI_HOST_EMBD: the host-kept table is GPU-readable (pinned, unified address space): every gather -- the round's
    // rows and the MTP head's device-produced ids -- runs as on a device table, so MTP works with it
    bool embd_mapped() const { return embd_mapped_.valid(); }
    const Tensor & output() const { return output_; }      // LM head (aliases tok_embd when tied)
    // the MTP draft head (a truncated, lower-bit copy of the first rows of output) and its row -> id map
    const Tensor & draft_head() const { return draft_head_; }
    const Tensor & draft_map() const { return draft_map_; }
    // AI_DRAFT_VOCAB: replace the draft head's rows with these ids (ascending, exactly draft_rows() of them);
    // empty = the default rows. Uploads only buffer contents, so captured graphs stay valid. Returns false
    // when the draft vocabulary is unavailable.
    bool    set_draft_rows(Device & dev, const std::vector<int32_t> & ids);
    int64_t draft_rows() const { return draft_head_.valid() ? draft_head_.ne[1] : 0; }
    // AI_TAPS: the last forward's tapped layer outputs (see tap_slot_), and how many rows it wrote
    const Tensor & taps() const { return taps_; }
    int64_t        taps_rows() const { return taps_rows_; }
    bool has_separate_output() const { return w_.has("output.weight"); }

  private:
    int64_t                      unscoped_rows_ = 16;
    int32_t *                    h_sel_   = nullptr;   // pinned routing read (streamed MoE)
    int32_t *                    h_slots_ = nullptr;   // pinned slot ids
    int64_t                      h_route_cap_ = 0;
    OffloadProf                  prof_;
    // Routing experiment (AI_ROUTE_MODE): 0 = model routing; 1 = freeze each layer's first expert
    // set; 2 = shuffle: each step takes k already-resident experts the layer did not use last step.
    // Modes 1 and 2 keep the kernel work identical and page nothing; only expert identity changes.
    std::vector<std::vector<int32_t>> route_prev_, route_frozen_;
    uint64_t                          route_rng_ = 0x9e3779b97f4a7c15ULL;
    // L2 prefetch (AI_L2_PREFETCH): 1 = during layer l, prefetch the slots layer l+1 used in its
    // previous step (a guess, with lead time); 2 = right before layer l's experts, prefetch its
    // exact slots (no lead time). 0 = off.
    std::vector<int32_t *>            h_pf_;   // pinned per pool layer: slot guesses
    // Router prediction (AI_PF_PRED): after layer l, layer l+1's router on layer l's output ranks
    // experts; 1 = copy the top non-resident ones (AI_PF_MAX per step, default 2, among the top
    // AI_PF_TOP, default 10) on the copy stream while layer l+1's mixer runs; 2 = predict and read
    // only (control).
    void    predict_next(Device & dev, int il, const Tensor & y);
    int32_t * h_pred_ = nullptr;
public:
    struct PredStats { uint64_t steps = 0, cand = 0, cold_true = 0, cold_pred = 0; };
    const PredStats & pred_stats() const { return pred_; }
private:
    PredStats pred_;
    std::vector<std::vector<int32_t>> pred_last_;   // per layer: the last predicted cold set
};

}  // namespace ai
