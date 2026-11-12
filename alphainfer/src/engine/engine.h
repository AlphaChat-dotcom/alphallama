// SPDX-License-Identifier: Apache-2.0
// Engine: request lifecycle over one Program. No CUDA calls here beyond what Device exposes.
//
// Execution model:
//  * every forward is one round: the Program's Ops are recorded into one graph and launched once;
//  * decode rounds are recorded once per profile and relaunched unchanged every step, so each is
//    captured once and replayed. Profile key = {visible-cell frontier F, streams B, drafts in,
//    drafts out}; everything else (tokens, positions, cells, masks, acceptance) is data;
//  * per-step values reach the device by non-blocking copies from pinned memory; results come
//    back through pinned egress copies; a step's only blocking call is the final synchronize.
//
// An MTP step is one round:
//   verify [t, d1..dk] -> greedy targets g -> accept m on the device -> masked fold of the
//   recurrent state -> head fill from (g, h) -> head token/hidden at row m -> k_out-1 chained
//   head steps whose inputs are computed on the device from m  -> sync (g, m, drafts).
// Streams run stream-major in one round (B > 1: KV regions + a state-slot block).
// With expert offload the same builders run eagerly (routing needs host round trips).
#pragma once

#include <cstdio>
#include "core/device.h"
#include "models/dflash2.h"
#include "models/qwen35_hybrid.h"
#include "ops/ops.h"
#include "engine/round_cache.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <string>
#include <tuple>
#include <vector>

namespace ai {

FILE * mtp_conf_file();   // AI_MTP_CONF_LOG (engine_mtp.cpp); nullptr when unset
FILE * draft_trace_file();   // AI_DRAFT_TRACE (engine_mtp.cpp): drafter-simulator trace; nullptr when unset

// kv_type value meaning "choose from VRAM, model and mode" (Engine::plan_kv); max_ctx <= 0 = the
// largest context that fits.
constexpr int kKvAuto = -1;

struct EngineConfig {
    std::string model_path;
    int         device    = 0;
    int64_t     max_ctx   = 8192;    // per stream; <= 0 = auto (largest that fits, capped at the training context)
    int64_t     chunk     = 512;     // prefill rows per round
    size_t      ws_bytes  = 0;       // 0 = derived from the Program's estimate
    bool        eager     = false;   // launch each Op separately (A/B and parity)
    int         expert_slots = 0;    // > 0: routed experts in host memory, this many device slots per layer
    bool        expert_slots_auto = false;   // expert_slots from the VRAM left after weights, KV, state,
                                             // workspace and ingress (expert_slots must be > 0 as the offload flag)
    int         mtp_k     = 0;       // > 0: MTP self-speculation with up to k drafts per step
    std::string dflash2_path;        // DFlash2 drafter GGUF: drafts come from it instead of the MTP head
                                     // (k = its block size - 1, batch 1, greedy acceptance as MTP)
    bool        mtp_adapt = false;   // choose k per step from measured acceptance and step time
    bool        exact_verify = false;   // set by apply_mtp_policy: verify rows must stay bit-identical to plain decode
    bool        stream_cap   = false;   // set by apply_mtp_policy: cap verify columns at 2+ live streams (see decode_round)
    int         max_batch = 1;       // streams per decode round
    int         kv_type   = kKvAuto;   // KV cell type: kKvAuto (default) | F16 | Q8_0 | Q4_0. See Engine::plan_kv.
    bool        verbose   = true;
};

struct Timing {
    double   load_s        = 0;
    double   prefill_s     = 0;
    double   decode_s      = 0;
    int64_t  prompt_n      = 0;      // summed over streams
    int64_t  generated_n   = 0;      // summed over streams
    int64_t  decode_tokens = 0;      // tokens emitted by decode steps (the first token comes from prefill)
    int64_t  decode_rounds = 0;      // decode steps (one round each)
    uint64_t decode_blocking_calls = 0;   // blocking host<->device calls during decode
    double   host_ms_per_round     = 0;   // staging + launch, before the sync
    int      rounds_recorded       = 0;
    int      round_nodes           = 0;
    int      batch                 = 1;
    int64_t  mtp_drafted           = 0;
    int64_t  mtp_accepted          = 0;
    std::array<int64_t, 9> k_steps{};    // decode steps per drafts-out value
    // per draft position j (0-based): stream-rounds that verified draft j (j < k_in) and that accepted it (j < m).
    // Both are kept because the gate verifies fewer drafts than are made.
    std::array<int64_t, 9> mtp_verified_by_pos{};
    std::array<int64_t, 9> mtp_acc_by_pos{};
    // MTP rounds whose profile was recorded or captured inside the round (inside the decode timer), and their ms
    int64_t                cold_rounds = 0;
    double                 cold_ms     = 0;
    // AI_MTP_PROF=1: per-phase ms of an MTP step, summed over the run (uncaptured, one sync per
    // phase, so the split is the point and not the absolute round time).
    std::array<double, 5> mtp_phase_ms{};   // verify, accept+fold, head fill, draft chain, egress
    int64_t               mtp_prof_steps = 0;
};

class Engine {
public:
    bool init(const EngineConfig & cfg);

    // Greedy generation. `on_token` is called per generated token; return false to stop.
    std::vector<int32_t> generate(const std::vector<int32_t> & prompt, int64_t n_gen,
                                  const std::function<bool(int32_t)> & on_token, Timing * t);

    // Greedy generation for up to max_batch prompts in shared decode rounds.
    // `on_token(stream, token)`; return false to stop that stream.
    std::vector<std::vector<int32_t>> generate_batch(const std::vector<std::vector<int32_t>> & prompts,
                                                     int64_t n_gen,
                                                     const std::function<bool(int, int32_t)> & on_token,
                                                     Timing * t);

    // Teacher-forced: full-vocab logits for every position of `tokens`, streamed to `sink`
    // one row at a time as (position, float* V).
    void score(const std::vector<int32_t> & tokens,
               const std::function<void(int64_t, const float *)> & sink, Timing * t);

    // Caller-driven decode, for a server that owns the loop (the AlphaServe adapter).
    // generate_batch runs to n_gen and owns stopping. These let a caller drive one round at a time
    // and decide stopping itself, reusing a slot across requests.
    //
    // A round is batch-rigid: T = L * active_batch_ and every slot is staged every round, so a
    // step advances all open streams together -- a subset cannot be decoded. open_streams(B) fixes B
    // for the batch, and decode_step must name every stream that is not finished.
    struct DecodeIn {
        int     stream;
        int64_t position;   // the caller's belief about this stream's next position
        int32_t token;      // the pending token to feed this round
    };
    // Size the batch and clear every slot (what generate_batch does on entry).
    void    open_streams(int B);
    // Per-slot sampling for the target tokens (temperature <= 0 = greedy, the default). It survives reset_stream and
    // prefill, so a caller sets it before a request's prefill (the first token is drawn there) and resets it to greedy
    // for the next greedy request. Drafts stay argmax; a draft is accepted when it equals the sampled target, so MTP,
    // prompt lookup and DFlash2 keep working and the output follows the sampling distribution exactly. Returns false
    // when sampling is off (AI_SAMPLING=0) or the slot is out of range.
    bool    set_sampling(int stream, float temperature, int top_k, float top_p, float min_p, uint64_t seed);
    bool    sampling_enabled() const { return samp_buf_ >= 0; }
    // Prompt phase for one slot; returns the first token, which the caller feeds back on step 1.
    int32_t prefill_one(int stream, const std::vector<int32_t> & prompt, Timing & tm);
    // Prompt phase with precomputed input rows (image embeddings): rows `embd_row[r]` take embd[r * E ..] instead
    // of their token's embedding, and every row takes its 4 RoPE components from pos4 (section-major t|y|x|z, 4n
    // entries). KV cells stay one per row; later rows run at RoPE position = cell + (max(t,y,x) + 1 - n). Not with DFlash2.
    int32_t prefill_mm(int stream, const std::vector<int32_t> & prompt, const int32_t * pos4, const float * embd,
                       const int32_t * embd_row, int64_t n_rows, Timing & tm);
    // AI_PREFILL_CHUNK: the same prompt phase, fed in pieces so a server can run other streams' decode rounds
    // between them (a whole-prompt prefill blocks every other slot for its full duration). The first call on a
    // stream that is not mid-prefill starts a new request (resets the slot, as prefill_one does); the stream stays
    // idle -- no round may name it -- until the call with last = true, which
    // returns the first token exactly as prefill_one would. Chunks run at prefill_one's boundaries (see
    // prefill_range), so the KV, the state and the first token are those of a one-shot prefill of the concatenation.
    // Returns -1 while the prompt is incomplete. Not available with DFlash2 (throws).
    int32_t prefill_feed(int stream, const int32_t * tokens, int64_t n, bool last, Timing & tm);
    // Tokens fed so far to a stream that is mid-prefill (empty otherwise): the C API chooses the draft rows from it.
    const std::vector<int32_t> & prefill_buffer(int stream) const;
    // One round. Emitted tokens are appended to `out` as (stream, token) in emission order -- MTP can
    // emit several per stream per round.
    // Positions are checked: a mismatch throws naming both the engine's position and the caller's.
    // This detects divergence; it does not let a caller re-position a slot, so prefix reuse is out
    // of scope until positions are authoritative.
    void    decode_step(const std::vector<DecodeIn> & in, std::vector<std::pair<int, int32_t>> & out,
                        Timing & tm);
    // [round cap] the NEXT decode round emits at most n tokens per stream (n >= 1; 0 = no cap). One round only.
    void    set_round_cap(int n) { round_cap_ = n; }
    // Reset one slot so the next request can reuse it. Must be followed by prefill_one before a step:
    // decode_step refuses a slot with fewer drafts than the round's k_in rather than staging garbage.
    void    reset_one(int stream);
    // [AI_PREFIX_REUSE, default 1] prompt tokens the stream's last prefill took from its cache (0: none)
    int64_t prompt_cached(int stream) const;
    // A new request is starting on a slot (called by the prefill paths after the slot reset). With one stream and
    // exact output (plain decode or the EXACT verify; AI_DOOR_RESET_PER_REQUEST=0|1 overrides), the doorbell offload
    // goes back to its initial residency, so the request's output does not depend on earlier requests (the CPU
    // rounds missed experts differently from the GPU).
    void    request_start();
    // Record every decode profile the F ladder can ask for, so nothing instantiates after startup:
    // a lazy instantiate mid-serving can fail, and a capture failure silently degrades to direct
    // launches. Returns the number of profiles recorded.
    //
    // The set is the F ladder alone, not (B x F): step_round() is called only from decode_round, so
    // prefill records nothing, and active_batch_ is fixed for an engine's lifetime (~8 MiB/profile).
    //
    // Warming stages real rounds, and staging reserves KV to max(n_kv, pos0+L), so every slot is
    // left with length F and garbage recurrent state. All slots are reset before returning, which
    // is required for correctness. Call it before any prefill.
    int     warm_profiles(Timing & tm);
    // AI_WARM_MTP: MTP counterpart. Record and capture (3 launches each: direct, capture, replay) every MTP profile a
    // batch-1 decode with positions in [pos_lo, pos_hi] can ask for -- each F bucket in that range x verify width
    // k_in 0..K (the gate picks any of them). Drives the real decode_round on a synthetic stream, so key, staging, the
    // device mask ops and the builder are the live ones by construction (warm_profiles builds build_plain without the
    // AI_DECODE_MASK_DEV ops, and only records). Needs mtp_k > 0, no --mtp-adapt, no DFlash2, active batch 1. Resets
    // every slot and restores k_in / acceptance / step-time state before returning. Returns profiles made warm.
    // Keeps cold-round capture time out of the first request's decode time.
    int     warm_mtp_profiles(int64_t pos_lo, int64_t pos_hi, Timing & tm);
    // AI_DRAFT_VOCAB: choose the MTP draft head's rows (engine/draft_vocab.h); call between requests only
    bool    set_draft_vocab(const std::vector<int32_t> & ids) { return prog_.set_draft_rows(dev_, ids); }
    int64_t draft_vocab_rows() const { return prog_.draft_rows(); }
    int     mtp_k() const { return cfg_.mtp_k; }   // after init: DFlash2 sets it to the drafter's block size - 1
    bool    dflash2() const { return df2_ != nullptr; }
    // AI_DF2_ROUTE, per request: with a DFlash2 drafter loaded, draft this request with DFlash2 (true, the default) or
    // with the model's own MTP head (false). Set before prefill_stream; batch 1 (every live stream shares the mode).
    void    set_df2_active(bool on) { df2_active_ = on; }
    bool    df2_on() const { return df2_ != nullptr && df2_active_; }
    int     stream_count() const { return (int) streams_.size(); }
    int64_t stream_position(int stream) const;
    bool    stream_done(int stream) const;

    const Qwen35Hybrid & program() const { return prog_; }
    Device &             device() { return dev_; }
    // What this start actually got, for a server's readiness check (a clamped bootstrap start holds fewer cells
    // than were requested).
    int64_t ctx_cells() const { return cfg_.max_ctx; }   // per stream, after kv/ws planning
    int     calibration_state() const { return ws_bootstrap_ ? 2 : ws_calibrated_now_ ? 1 : 0; }

private:
    // Resolve kKvAuto / max_ctx <= 0 after the weights are resident and before alloc_state.
    bool    plan_kv();
    int64_t kv_plan_bytes_  = -1;   // what the plan expects alloc_state..finalize to allocate (-1: no plan)
    size_t  kv_alloc_before_ = 0;
    // Measured workspace sizing (see ws_auto_lookup in engine.cpp)
    void        ws_auto_lookup();
    void        ws_auto_calibrate();
    std::string ws_key_;
    bool        ws_cal_pending_ = false;
    bool        ws_calibrating_ = false;   // AI_TF dumps skip the calibration's synthetic rounds
    bool        ws_bootstrap_   = false;   // derived workspace did not fit: running on the remainder to calibrate
    bool        ws_calibrated_now_ = false;   // this create ran the calibration (a bootstrap or a recalibration)
    int64_t     ws_cal_region_  = 0;       // envelope (cells per stream) the cached peak was measured up to
    // Round fit on a fixed workspace (--ws-mib / ALPHA_AI_WS_MIB, which skip the calibration): the widest verify
    // width k_in whose round fits at each envelope F, measured at init by recording rounds without launching them.
    // decode_round narrows a round to it. Empty = no limit (calibrated workspace, or every width fits).
    void        ws_fit_rounds();
    int         ws_fit_k(int64_t F) const;
    std::vector<std::pair<int64_t, int>> ws_fit_;   // (F, widest k_in that fits; -1 = none)

    // Fixed device ingress with pinned mirrors. A round of B streams x L rows uses a prefix.
    struct Ingress {
        Tensor     ids, pos, cells, mask, last, base, zeros, delta;
        int32_t *  h_ids   = nullptr;
        int32_t *  h_pos   = nullptr;
        int64_t *  h_cells = nullptr;
        uint16_t * h_mask  = nullptr;
        int32_t *  h_last  = nullptr;
        int32_t *  h_base  = nullptr;   // per-stream first position (device step inputs)
        int32_t *  h_zero  = nullptr;
        int32_t *  h_delta = nullptr;   // per-stream RoPE-minus-cell offset (device MTP chain steps)
        int64_t    T_cap = 0, mask_cap = 0, B_cap = 0;
        // AI_HEAD_MASK_DEV: the mask buffer is sized for decode/verify rounds only; a round whose mask does not fit
        // (a prefill chunk) has it written on device by its caller, into the main ingress's mask buffer
        bool       mask_on_device = false;
        // AI_DECODE_MASK_DEV: set by decode_round around its staging: the mask is written in the captured round
        // from the device base positions, so stage_rows skips building it and upload() skips sending it
        bool       mask_dev_now = false;
        bool       mask_fits(int64_t n_kv, int64_t T) const { return n_kv * T <= mask_cap; }
        // AI_HOST_EMBD: staging for a host-kept token_embd. The host gathers the round's rows --
        // a pure memcpy, T x tbl_rowb bytes -- and the device runs the same ggml_get_rows over the
        // small block with an iota id list, so the result is bit-identical to the resident path.
        Tensor       embd, embd_iota;
        char *       h_embd   = nullptr;
        const char * tbl_host = nullptr;   // host-kept token_embd row data
        size_t       tbl_rowb = 0;
        int64_t      tbl_rows = 0;         // vocab, for the id bounds check
        int64_t      tbl_cols = 0;         // E; embd is carved 1-D, so its ne[0] is E*T_cap, not E
        void alloc(Device & dev, int64_t T, int64_t mask_elems, int64_t B, const char * name);
        void alloc_embd(Device & dev, const Tensor & table, const char * name);
        Qwen35Hybrid::Inputs view(int64_t L, int64_t B, int64_t n_kv, int first_region) const;
        void upload(Device & dev, int64_t L, int64_t B, int64_t n_kv) const;
        Tensor base_view(int64_t B) const;    // i32 [B]
        Tensor zeros_view(int64_t B) const;   // i32 [B] of 0
        Tensor delta_view(int64_t B) const;   // i32 [B] RoPE position - cell
        Tensor mask_view(int64_t n_kv, int64_t L, int64_t B) const;   // f16 [n_kv, L, 1, B]
    };

    // Egress of one decode step (all device tensors in the workspace).
    struct StepOut {
        Tensor              token;    // plain: i32 [B]
        Tensor              logits;   // f32 [V, rows] the round's full logits (read only by AI_TF_DUMP)
        Tensor              g, m;     // MTP: i32 [B*L] targets, i32 [B] accepted drafts
        std::vector<Tensor> drafts;   // MTP: k_out x i32 [B]
        std::vector<Tensor> confs;    // MTP + AI_MTP_GATE: k_out x f32 [B], each draft's max-softmax
    };
    using ProfileKey = std::tuple<int64_t, int, int, int, int>;   // F, B, k_in, k_out, first region (compacted rounds)

    struct Stream {
        std::vector<int32_t> out;
        std::vector<int32_t> drafts;
        int64_t              pos  = 0;    // position of the pending token
        int32_t              pend = -1;   // emitted, not yet in the KV
        bool                 done = false;
        // AI_PREFILL_CHUNK mid-prefill: tokens fed so far and the next unprocessed chunk offset. While set, the
        // stream is idle (done) and decode_round refuses any round whose width would include it.
        bool                 prefilling = false;
        std::vector<int32_t> pf;
        int64_t              pf_off = 0;
        // AI_LOOKUP prompt-lookup drafting: the stream's token history (prompt + every emitted token, the last
        // one being the pending token) and, per n-gram length, n-gram hash -> the most recent end position of that
        // n-gram (indexed up to lk_done). lk_k > 0: the next round verifies lk_k lookup drafts instead of MTP's.
        std::vector<int32_t>                                hist;
        std::vector<std::unordered_map<uint64_t, int32_t>> lk;
        int64_t                                            lk_done = 0;
        int                                                lk_k    = 0;
        int                                                lk_len  = 0;   // slow-start proposal length (0 = L0)
        int                                                lk_n    = 0;   // real (unpadded) length of the lookup drafts
        // AI_LOOKUP_ARM shadow mode: a lookup continuation is tracked against the emitted tokens without being used;
        // lookup drafts are used only once a shadow run reached AI_LOOKUP_ARM tokens (disarmed after 2 bad rounds)
        std::vector<int32_t>                               lk_sh;
        int                                                lk_shi = 0, lk_run = 0, lk_miss = 0, lk_full = 0;
        bool                                               lk_armed = false;
        // [AI_PREFIX_REUSE] chunked prefill: the tokens fed so far are still the stream's cached prompt, nothing run yet
        bool                                               probe    = false;
    };
    // [AI_PREFIX_REUSE] per stream: the last prompt and host copies of the slot's recurrent state at chunk boundaries
    // inside it (the attention KV below a boundary is still the slot's, masked by position), so the next prompt that
    // shares a prefix resumes from the last boundary inside it instead of from token 0
    struct PfxSnap { int64_t at = 0; std::vector<char> buf; };
    struct PfxCache { std::vector<int32_t> toks; std::vector<PfxSnap> snaps; int64_t cached = 0; };
    std::vector<PfxCache> pfx_;
    bool    pfx_on() const;
    void    pfx_capture(int stream, const std::vector<int32_t> & prompt, int64_t at);
    int64_t pfx_resume(int stream, const int32_t * p, int64_t n, int64_t limit) const;
    void    pfx_continue(int stream, int64_t at);   // the slot holds the cached prompt through `at`; prefill from there

    void    reset_streams(int B);
    // Per-slot reset, so a caller-driven server can reuse a slot across requests. This cannot be
    // built on reset_streams(), which calls kv().free_all() -- a global wipe that would destroy every
    // other slot. It is built on free_seq(), and it is sound because KV regions are a static
    // partition: first_cell(seq) = seq * region_ (kv.h), so a slot's cells are disjoint from every
    // other slot's by arithmetic, not by an allocator whose bookkeeping could alias them.
    void    reset_stream(int stream);
    // Prompt phase for one stream: chunked prefill, sets pos/pend, adds to tm.prompt_n, returns the
    // first token. Separate from decode_round because prefill chunks internally via stage_chunk
    // while a decode round stages a single L = k_in+1 row set -- one entry covering both would hide
    // two very different cost profiles behind one call.
    int32_t prefill_stream(int stream, const std::vector<int32_t> & prompt, Timing & tm);
    void    lk_begin(Stream & s, const std::vector<int32_t> & prompt);   // AI_LOOKUP per-request state
    // One decode round over the live streams, driven from the hoisted state. `sink` receives every
    // emitted token and returns whether that stream continues: the engine does not own the stopping
    // rule, which is precisely what makes a caller-driven decode possible. Returns streams still live.
    int     decode_round(const std::function<bool(int, int32_t)> & sink, Timing & tm);
    // Records a token against a stream and applies the sink's verdict. Shared by the prefill emit and
    // decode_round so the two cannot drift -- they differ only in when they run.
    bool    emit_token(int stream, int32_t tok, const std::function<bool(int, int32_t)> & sink, Timing & tm);
    int64_t frontier(int64_t cells) const;
    // sequence `seq`: L rows at pos0 into rows [row0, row0+L) of a T-row round in g (tokens may be
    // null for head rows whose ids come from the device); records base[row0/L] = pos0.
    void    stage_rows(Ingress & g, int seq, int64_t row0, int64_t T, const int32_t * tokens, int64_t L,
                       int64_t pos0, int64_t n_kv, bool head);
    Qwen35Hybrid::Inputs stage_chunk(int stream, const int32_t * tokens, int64_t T, int64_t pos0, int64_t n_kv);

    int32_t prefill(int stream, const std::vector<int32_t> & prompt);
    void    prefill_mtp(int stream, const std::vector<int32_t> & prompt, Stream & st);
    // AI_PREFILL_CHUNK: resumable bodies of the two above: run chunks from `off` (advanced); !final skips the tail
    int32_t prefill_range(int stream, const std::vector<int32_t> & prompt, int64_t & off, bool final);
    void    prefill_mtp_range(int stream, const std::vector<int32_t> & prompt, int64_t & off, bool final, Stream & st);
    // Prefill slot lending (AI_DOOR_LEND_MIB, doorbell offload): a one-shot prefill runs the rows before its last
    // chunk in larger chunks whose workspace is borrowed from expert slot banks. Returns the rows done (0 = not lent).
    int64_t lend_prefill(int stream, const std::vector<int32_t> & prompt, bool mtp);

    StepOut build_plain(const Qwen35Hybrid::Inputs & in);
    StepOut build_mtp(const Qwen35Hybrid::Inputs & vin, const Qwen35Hybrid::Inputs & hin, int k_out);
    // DFlash2: the verify half of an MTP step only (forward + targets + accept + fold); drafts come from df2_
    StepOut build_df2(const Qwen35Hybrid::Inputs & vin);
    void    prefill_df2(int stream, const std::vector<int32_t> & prompt, Stream & st);
    // append the committed tap rows [0, rows) of the last forward at pos0, then draft from anchor at anchor_pos
    void    mtp_conf_log(const std::vector<Tensor> & logits);   // AI_MTP_CONF_LOG: per-draft max-softmax
    std::vector<Tensor> conf_logits_;                           // the last chain's draft logits (AI_MTP_CONF_LOG)
    void    df2_advance(Stream & st, int64_t rows, int64_t pos0, int32_t anchor, int64_t anchor_pos);
    // the same step, launched phase by phase with a sync between each (AI_MTP_PROF)
    StepOut profile_mtp(const Qwen35Hybrid::Inputs & vin, const Qwen35Hybrid::Inputs & hin, int k_out, Timing & tm);
    // the image context of the prefill in progress (prefill_mm), and its device / pinned staging
    struct MmCtx {
        const int32_t * pos4 = nullptr;
        const float *   embd = nullptr;
        const int32_t * row  = nullptr;   // sorted prompt row of each embedding row
        int64_t         n_rows = 0;
    } mm_;
    int     mm_buf_ = -1;
    Tensor  mm_dev_;             // f32 [E, chunk]
    float * mm_host_ = nullptr;  // pinned mirror
    void    mm_attach(Qwen35Hybrid::Inputs & in, int64_t off, int64_t T);
    void    chain(std::vector<Tensor> & drafts, Qwen35Hybrid::MtpOut first, const Tensor & base, const Tensor & m,
                  int offset0, int k_out, int64_t B, int64_t n_kv, int first_region, const Tensor & delta = Tensor{});
    // Launch the step's round (recording it on first use); `warm` = replayed at least twice before.
    const StepOut & step_round(const ProfileKey & key, const std::function<StepOut()> & build, Timing & tm,
                               bool * warm);

    // prefill/score chunks (AI_FFN_CPU: its prefill stages weights through VRAM with host copies, so eager too)
    bool chunk_eager() const { return cfg_.eager || cfg_.expert_slots > 0 || prog_.cpu_ffn_tmax() > 0; }
    static bool head_mask_dev();   // AI_HEAD_MASK_DEV: the MTP head's prefill mask is built on device (default on)
    int  choose_k(int k_in) const;
    void observe(const std::vector<int> & accepted, int k_in, int k_out, double step_ms);

    EngineConfig cfg_;
    Device       dev_;
    Qwen35Hybrid prog_;
    std::unique_ptr<DFlash2> df2_;
    bool                     df2_active_ = true;
    int          live_streams_ = 0;
    int          active_batch_ = 1;
    // Per-stream decode state, hoisted out of generate_batch so a caller can drive one round at a
    // time. k_in_ and in_decode_ are round-carried and batch-global, not per-stream, so they live
    // here rather than in Stream; moving them into Stream would silently reset the MTP draft depth
    // on every call.
    std::vector<Stream>  streams_;
    int                  k_in_      = 0;
    int                  round_cap_ = 0;   // set_round_cap: tokens per stream the next round may emit (0 = no cap)
    bool                 in_decode_ = false;
    std::vector<int32_t> vt_;        // scratch: pending token + k_in drafts
    std::vector<int>     acc_now_;   // scratch: per-round accept counts

    Ingress  main_in_, head_in_;
    // prefill slot lending: the lent chunks' ingress (no host masks: lend_prefill writes them on device into the
    // borrowed workspace), the largest lent chunk (0 = off), and whether init is done (no lending while calibrating)
    Ingress  lend_in_, lend_head_in_;
    int64_t  lend_rows_ = 0;
    bool     lend_ok_   = false;
    // per-slot kern::SampleParams on the device (read by the captured rounds) and its pinned host copy
    int       samp_buf_  = -1;
    Tensor    samp_dev_;
    void *    samp_host_ = nullptr;
    int       samp_n_    = 0;
    const void * samp_ptr() const { return samp_buf_ >= 0 ? samp_dev_.data : nullptr; }
    int32_t * h_out_    = nullptr;   // plain tokens [B]
    int32_t * h_g_      = nullptr;   // [B*(k+1)]
    int32_t * h_m_      = nullptr;   // [B]
    int32_t * h_drafts_ = nullptr;   // [k*B]
    float *   h_conf_   = nullptr;   // [k*B] AI_MTP_GATE: draft confidences
    std::vector<Tensor> chain_confs_;   // the last chain's per-draft confidences (AI_MTP_GATE)

    RoundCache<ProfileKey, StepOut> rounds_;   // recorded / captured / replayed decode rounds, graph budget

    // adaptive k: per-depth conditional acceptance and per-(k_in, k_out) step time
    std::array<double, 9> acc_{};
    std::map<std::pair<int, int>, double> step_ms_;
    StepOut eager_out_;
};

}  // namespace ai
