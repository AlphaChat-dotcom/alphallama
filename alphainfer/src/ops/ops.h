// SPDX-License-Identifier: Apache-2.0
// AlphaInfer closed Op set.
//
// Each Op is a semantic unit of the forward pass: it owns every primitive inside it, reads only
// the tensors it is handed, writes only its output (from the workspace) and the state it is
// explicitly given (KV cells, conv history, recurrent state). No Op allocates persistent memory,
// synchronises, or reads the model spec.
//
// Most Ops are adapters over ggml's CUDA kernels, with the primitive order mirroring the llama.cpp
// graph so numerics are comparable; some bodies are native kernels.
#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <functional>
#include <memory>

namespace ai::adapter { class Recorder; }

namespace ai::op {

// ---- rounds -------------------------------------------------------------------------------
// Between begin() and end(), every Op call is appended to this round instead of launching.
// launch() issues the whole round at once; relaunching an unchanged round replays it.
class Round {
public:
    explicit Round(Device & d);
    ~Round();
    void begin();
    void end();
    void launch();
    int  n_nodes() const;

private:
    std::unique_ptr<adapter::Recorder> rec_;
};
bool recording();   // true while some Round is between begin() and end()

// A host step: runs now (eager), or at this point of every launch of the round being recorded.
// It may block on the device (e.g. read routing); the device work before it is its own capture.
void host_step(Device & d, std::function<void()> fn);

// L2 prefetch of the expert row spans named by `slots` (i32 [n], device; < 0 = skip) in three
// slot banks [.., .., S]. A hint: it changes no values.
// Predictor: rmsnorm(x) * norm, then the router; returns the top-P expert ids per row (i32 [P, T],
// contiguous, best first).
Tensor router_topk(Device & d, const Tensor & x, const Tensor & norm, const Tensor & router, float eps, int P);

void l2_prefetch(Device & d, const Tensor & up, const Tensor & gate, const Tensor & down, const Tensor & slots);

// Who captures relaunched rounds: AlphaInfer on the backend stream (default; includes native
// kernels), the backend per ggml graph, or nobody. Set before the Device is initialised.
enum class Capture { Own, Backend, None };
void set_capture(Capture c);
bool    capture_degraded();   // a round has fallen back to direct launches
Capture capture_mode();

// ---- elementary --------------------------------------------------------------------------
Tensor embed(Device & d, const Tensor & table, const Tensor & ids);           // [E,V] x i32[T] -> f32[E,T]
Tensor rmsnorm(Device & d, const Tensor & x, const Tensor & w, float eps);    // rms_norm(x) * w
// One kernel for the pair every layer runs: sum = mix + res, out = rms_norm(sum) * w. Both are
// returned -- sum feeds the next residual. Bit-exact with add() followed by rmsnorm().
struct AddNorm { Tensor sum, out, q8, router; bool router_cmp = false; };   // q8: optional block_q8_1 copy of out
// router: when router_w is given (AI_ADDNORM_ROUTER), the router logits [nrows, T] computed in the same launch;
// router_cmp: they are a shadow for the =2 compare, not to be consumed
// out_buf/q8_buf, when given, are used instead of workspace allocations: the registry is keyed on the
// activation's address and the arena reuses addresses across layers, so a staged entry has to live
// where no other layer can occupy.
AddNorm add_rmsnorm(Device & d, const Tensor & mix, const Tensor & res, const Tensor & w, float eps,
                    const Tensor * out_buf = nullptr, const Tensor * q8_buf = nullptr,
                    const Tensor * router_w = nullptr);
Tensor add(Device & d, const Tensor & a, const Tensor & b);
Tensor rows(Device & d, const Tensor & x, const Tensor & ids);                // select rows (i32 ids)
Tensor argmax(Device & d, const Tensor & logits, const Tensor * map = nullptr);   // f32[V,N] -> i32[N]; map: row -> id
Tensor max_softmax(Device & d, const Tensor & logits);   // f32[V,N] -> f32[N]: a draft's confidence (AI_MTP_GATE)
void   copy_into(Device & d, const Tensor & src, const Tensor & dst, const char * who = "?");  // dst <- src

// ---- linear (bench) ---------------------------------------------------------------------
Tensor linear(Device & d, const Tensor & w, const Tensor & x);                 // adapter mul_mat
Tensor linear_q8_native(Device & d, const Tensor & w, const Tensor & x, int rows_per_cta);   // Q8_0, T=1
// Q8_0 whose bytes were repacked planar (kern::q8_repack_planar); f32 x of 1..8 columns, no activation
// quantisation. `w` keeps its Q8_0 metadata; only the byte layout differs, which the caller guarantees.
// mode: 1/2/4 = shared-x kernel (T=1 only), -1/-2/-4 = split-K over that many warps per row.
Tensor linear_q8_planar(Device & d, const Tensor & w_planar, const Tensor & x, int mode);
// fp16 x [cols, T<=8] against Q8_0 in GGUF's own layout (no repack; spans work), split 1/2/4.
Tensor linear_q8_gguf_f16x(Device & d, const Tensor & w, const Tensor & x16, int split);
Tensor cast_f16(Device & d, const Tensor & x);   // contiguous f32 -> f16, same shape
// one MUL_MAT_ID over an expert bank [K, N, n_expert] with ids i32 [k, T] (bench)
Tensor linear_id(Device & d, const Tensor & w, const Tensor & x, const Tensor & ids);

// ---- LM head -----------------------------------------------------------------------------
Tensor head(Device & d, const Tensor & x, const Tensor & norm_w, const Tensor & out_w, float eps);

// ---- gated full attention (Qwen3.5 layout) ----------------------------------------------
struct AttnWeights {
    Tensor q;        // [E, 2*D*Hq]   (query and output gate, interleaved per head)
    Tensor k;        // [E, D*Hkv]
    Tensor v;        // [E, D*Hkv]
    Tensor o;        // [D*Hq, E]
    Tensor q_norm;   // [D]
    Tensor k_norm;   // [D]
    Tensor qkv;      // optional [E, 2*D*Hq + 2*D*Hkv]: q (with gate), k, v rows in one projection
};
struct AttnParams {
    int   n_head, n_head_kv, head_dim;
    int   n_rot;
    int   sections[4];
    int   rope_mode;       // imrope
    int   n_ctx_orig;
    float rope_base, rope_scale;
    float eps, kq_scale;
};
struct KvLayer {
    Tensor k;        // f16 [D*Hkv, n_cells]
    Tensor v;        // f16 [D*Hkv, n_cells]
};
struct AttnInputs {
    Tensor write_cells;   // i64 [T]   absolute cache cells this round writes
    Tensor pos4;          // i32 [4T]  imrope positions (p, p, p, 0)
    Tensor mask;          // f16 [n_kv, L, 1, B]  per-stream causal mask over each region's first n_kv cells
    int64_t n_kv = 0;     // visible cells per stream
    int64_t B = 1;        // streams; rows are stream-major, L = T / B tokens per stream
    int64_t region_cells = 0;   // cells per stream region (stride between streams)
    int64_t first_cell   = 0;   // start of stream 0's region
};
Tensor attn_block(Device & d, const Tensor & x, const AttnWeights & w, const AttnParams & p,
                  const KvLayer & kv, const AttnInputs & in);

// ---- gated DeltaNet ---------------------------------------------------------------------
struct GdnWeights {
    Tensor qkv;      // [E, 2*Sk*Hk + Sv*Hv]
    Tensor z;        // [E, Sv*Hv]
    Tensor beta;     // [E, Hv]
    Tensor alpha;    // [E, Hv]
    Tensor conv;     // [K, C]
    Tensor dt_bias;  // [Hv]
    Tensor a;        // [Hv]
    Tensor norm;     // [Sv]
    Tensor out;      // [Sv*Hv, E]
    Tensor ba;       // optional [E, 2*Hv]: beta rows then alpha rows, one projection for both
    Tensor qkvz;     // optional [E, C + Sv*Hv]: qkv rows then z rows, one projection for both
};
struct GdnParams {
    int   conv_k;          // K
    int   channels;        // C = 2*Sk*Hk + Sv*Hv
    int   k_heads, v_heads;
    int   head_k, head_v;
    float eps;
};
struct GdnState {
    Tensor conv;     // f32 [(K-1)*C, B]  committed conv history of B consecutive slots (one per stream)
    Tensor ssm;      // f32 [Sv*Sv*Hv, B] committed recurrent state of the same slots
    // AI_GDN_GATE: this layer's own output-gate buffers (f32 [Sv*Hv, rows] and its q8_1 copy). Per layer
    // because the q8 staging registry is keyed by address, and the attention o-proj input has the same
    // [4096, T] geometry -- an arena buffer could match a stale entry. Invalid = keep the ggml tail.
    Tensor gate_out;
    Tensor gate_q8;
};
// Speculative verify (ReplaySSM): in record mode the block publishes nothing (neither
// conv history nor recurrent state) and keeps the tensors a later fold needs.
struct GdnRecord {
    Tensor  conv_input;   // f32 [K-1+T, C, 1]  history ++ new columns
    Tensor  conv_out;     // f32 [C, T, 1]      q|k|v after conv + silu
    Tensor  ba;           // f32 [2*Hv, T]      fused beta|alpha projection (when fused)
    Tensor  beta, alpha;  // f32 [1, Hv, L, B]  split projections (when not fused)
    // Fused-prologue verify (AI_FUSE_GDN_REC): the packed q/k (L2-normed), gate and sigmoid(beta) it already
    // computed. When valid, the fold reuses them instead of re-deriving them through ggml's prep.
    Tensor  qh, kh, gh, bh;
    int64_t T = 0, L = 0, B = 1;
};
Tensor gdn_block(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                 GdnRecord * rec = nullptr);
// Commit the first m1 recorded columns: advance the recurrent state in place and roll the conv
// history to end at column m1.
void   gdn_fold(Device & d, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                const GdnRecord & rec, int64_t m1);

// Commit only columns t <= m[b] of every stream, with a fixed shape: masked columns get beta = 0 and
// decay = 1 (colmask adds -1e30 to their raw beta/alpha), so they leave the state untouched; the conv
// history is gathered at the per-stream offset m[b]+1. m is device data: one profile serves every m.
void   gdn_fold_masked(Device & d, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                       const GdnRecord & rec, const Tensor & colmask, const Tensor & m);

// Device-side accept over verify targets g (i32 [B*L]) and verify ids (drafts at t >= 1).
struct Accept {
    Tensor m;         // i32 [B]
    Tensor colmask;   // f32 [B*L]
};
Accept mtp_accept(Device & d, const Tensor & g, const Tensor & ids, int64_t L, int64_t B);
// Row m[b] of each stream: i32 [B*L] -> [B], or f32 [E, B*L] -> [E, B].
Tensor select_rows(Device & d, const Tensor & src, int64_t L, int64_t B, const Tensor & m);
// Fill head-step inputs (pos4 [4B], cells [B], mask [n_kv*B]) at base[b] + m[b] + offset, on the device.
// AI_HEAD_MASK_DEV: write the MTP head's prefill mask (row i at pos0 + i, cells 1..pos0+i) into `mask` [n_kv, T]
void   head_causal_mask(Device & d, const Tensor & mask, int64_t T, int64_t n_kv, int64_t pos0);
// AI_DECODE_MASK_DEV: a decode round's mask [n_kv, L, 1, B] from the ingress base positions (device), main or head rule
void   decode_causal_mask(Device & d, const Tensor & mask, const Tensor & base, int64_t L, int64_t B, int64_t n_kv,
                          bool head);
// delta: i32 [B] RoPE-minus-cell offset per stream (an invalid Tensor = 0)
void   mtp_step_inputs(Device & d, const Tensor & base, const Tensor & m, int64_t B, int offset,
                       int64_t region_cells, int64_t first_cell, int64_t n_kv, const Tensor & pos4,
                       const Tensor & cells, const Tensor & mask, const Tensor & delta = Tensor{});

// MTP head input: eh_proj(concat(rmsnorm(embed(ids)) * enorm, rmsnorm(H) * hnorm))
Tensor mtp_input(Device & d, const Tensor & table, const Tensor & ids, const Tensor & H,
                 const Tensor & enorm, const Tensor & hnorm, const Tensor & eh_proj, float eps);

// ---- sparse MoE with one sigmoid-gated shared expert ------------------------------------
struct MoeWeights {
    Tensor router;       // f32 [E, n_expert]
    Tensor up_exps;      // [E, F, n_expert]
    Tensor gate_exps;    // [E, F, n_expert]
    Tensor down_exps;    // [F, E, n_expert]
    Tensor sh_up;        // [E, Fs]
    Tensor sh_gate;      // [E, Fs]
    Tensor sh_down;      // [Fs, E]
    Tensor sh_score;     // f32 [E]
    // zero-copy fused projections (spans over adjacently placed weights), when available:
    Tensor route_score;  // f32 [E, n_expert + 1]  router rows then the shared-gate score row
    Tensor sh_gate_up;   // [E, 2 Fs]  shared gate rows then shared up rows
};
struct MoeParams {
    int n_expert, n_used;
};
Tensor moe_block(Device & d, const Tensor & x, const MoeWeights & w, const MoeParams & p);

// y = r + moe_block(x): the routed and shared experts on the adapter, then one native kernel for
// the weighted merge, the shared-expert add and the residual (replaces 10 launches per layer).
// AI_DUMP_DIR: tells the MoE op which layer index to label its router dumps with. Debug only;
// the dumps fire only when AI_DUMP_DIR is set and the round is not being recorded.
void set_dump_layer(int il);

Tensor moe_residual(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w, const MoeParams & p);
// The same, with the next layer's input norm folded into the merge kernel. `normed` is the norm's
// output; it is only produced when the fused merge path is taken, and is invalid otherwise.
struct MoeOut { Tensor y, normed; };
// Dense FFN (the 27B artifacts): y = r + down(silu(gate(x)) * up(x)), weights in w.sh_gate / sh_up / sh_down.
// With next_norm, the residual add also produces the next layer's input norm (and its q8_1 staging) through
// add_rmsnorm, returned as `normed`.
MoeOut dense_ffn_residual_norm(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w,
                               const Tensor * next_norm, const Tensor * norm_buf, float eps,
                               const Tensor * q8_buf = nullptr);
MoeOut moe_residual_norm(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w,
                         const MoeParams & p, const Tensor & next_norm, const Tensor & norm_buf, float eps,
                         const Tensor * q8_buf = nullptr, const Tensor * router_pre = nullptr,
                         bool router_cmp = false);

// Streamed MoE (offload assembly): the same math as moe_block, split at the point
// where the host must learn which experts to page in.
struct MoeRoute {
    Tensor sel;       // i32 [k, T]  selected experts
    Tensor weights;   // f32 [1, k, T] renormalised routing weights
};
MoeRoute moe_route(Device & d, const Tensor & x, const Tensor & router, const MoeParams & p);
// Routed experts over banks whose third axis is pool slots; `slots` is `sel` remapped to slots.
Tensor   moe_experts(Device & d, const Tensor & x, const Tensor & up, const Tensor & gate, const Tensor & down,
                     const Tensor & slots, const Tensor & weights, int k);
Tensor   moe_shared(Device & d, const Tensor & x, const MoeWeights & w);
Tensor   mul(Device & d, const Tensor & a, const Tensor & b);

}  // namespace ai::op
