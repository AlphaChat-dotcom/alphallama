// SPDX-License-Identifier: Apache-2.0
// The qwen4exp Ops: contracts. Bodies are adapter Ops over ggml, in src/ops/q4x_ops.cpp.
//
// Every Op mirrors the llama graph builder function named next to it (src/models/qwen4exp.cpp), primitive for
// primitive, so per-layer outputs, teacher-forced top-1 and paired PPL can be compared against the llama path like
// with like. Same rules as ops.h: an Op reads only what it is handed, writes only its output (from the workspace) and
// the state it is given, never allocates persistent memory, never synchronises, never reads the spec.
//
// Shapes: N = n_embd (2560), hc = hyper-connection streams (4), D = N*hc (10240), lr = hc low rank (320), T = rows.
// The residual is the HC stream block R: f32 [N, hc, T] (res_hc in the llama graph).
#pragma once

#include "ops/ops.h"

namespace ai::op::q4x {

// hyper-connections
struct HcWeights {
    Tensor norm;     // [D]        hc_{attn,ffn}_norm / hc_head_norm (RMS weight over each stream)
    Tensor down;     // [D, lr]    BF16 in the file; Q8_0 after the load-time requant
    Tensor up;       // [lr, D]
    Tensor inject;   // [D, hc]    invalid for the head mixer
};
struct HcRead {
    Tensor mixed;    // f32 [N, T]   the block input: mean_s(xn_s * sigmoid(g_s))
    Tensor inject;   // f32 [hc, T]  inject = w_inject . xn (invalid for the head mixer)
};
// HC read before a block (build_hc_mix): per-stream RMS * w_norm -> lo = silu(w_down . xn / hc) ->
// g = w_up . lo -> mixed, inject. T <= 8: the fused GGML_OP_HC_MIX_FUSED; larger T: hc_norm_inject + 2 GEMMs + tail.
HcRead hc_read(Device & d, const Tensor & R, const HcWeights & w, float eps);
// HC scatter after a block (GGML_OP_HC_COMBINE): R' = R + out * 2 sigmoid(inject / hc), per stream.
Tensor hc_scatter(Device & d, const Tensor & R, const Tensor & out, const Tensor & inject);
// hc_scatter written back into R itself (contiguous f32 [N, hc, T]; out [N, T] and inject [hc, T] with row strides)
void   hc_scatter_into(Device & d, const Tensor & R, const Tensor & out, const Tensor & inject);
// Final mixer = the output norm: hc_read without inject. Returns f32 [N, T].
Tensor hc_head(Device & d, const Tensor & R, const HcWeights & w, float eps);

// w . x for a qwen4exp dense projection: q6_K / q5_K / q4_K / iq4_xs / iq3_s rows with <= 32 columns run on the
// fp16 tensor-core dense kernel (kernels/q4x_dense.cu); anything else (or AI_Q4X_DENSE=0) is op::linear (ggml MMVQ/MMQ).
bool   dense_ok(const Tensor & w, const Tensor & x);
Tensor linear(Device & d, const Tensor & w, const Tensor & x);
// Token embedding repeated across the hc streams (repeat_4d): f32 [N, hc, T].
Tensor embed_hc(Device & d, const Tensor & table, const Tensor & ids, int hc);

// gated full attention + QSA (indexer sparse attention)
// The attention half reuses op::AttnWeights / AttnParams / KvLayer / AttnInputs (q rows carry the output gate,
// interleaved per head: wq [N, D_head * n_head * 2]).
struct QsaWeights {
    Tensor q_proj;   // [N, idx_heads * idx_dim]
    Tensor k_proj;   // [N, idx_dim]
    Tensor q_norm;   // [idx_dim]
    Tensor k_norm;   // [idx_dim]
};
struct QsaParams {
    int idx_heads = 0, idx_dim = 0, top_k = 0;
    int ratio = 0;   // compress_ratios[il]; 0 = dense attention on this layer
};
struct QsaCache {
    Tensor keys;     // f16 [idx_dim, n_cells]  raw indexer key per cell (same cells as the attention KV)
    Tensor pooled;   // f16 [idx_dim, n_cells / ratio]  pooled (mean over `ratio` cells) + normed + roped block keys
    // Optional block inputs (as set_input_qsa in the llama path). Invalid = derived on the device from the round's
    // positions for one stream whose cell j holds position j (kernels/q4x_qsa.cu, the ALPHA_QSA_FAST rule). Paged KV
    // with several streams fills them explicitly. n_blocks = ceil(n_kv / ratio).
    Tensor cell_blk;    // i32 [n_kv]              cell -> block
    Tensor blk_cells;   // i32 [ratio * n_blocks]  block -> member cells
    Tensor blk_pos;     // i32 [4 * n_blocks]      block rope positions (4 sections)
    Tensor bias;        // f32 [n_blocks, T]       per-block selection bias (-inf / 0 / 1e9)
};
// QSA (build_qsa_top_k): write this round's indexer keys, refresh the stale pooled blocks, score
// blocks (sum over heads relu(q . k) + bias), select the top (top_k + ratio - 1) cells (top_k_expand), then sparse
// flash attention over exactly those cells (fattn-sparse set_cells) for T <= 8; a dense mask for prefill. At
// n_kv <= top_k the result must equal dense attention (the same invariant as the llama path).
Tensor attn_qsa_block(Device & d, const Tensor & x, const AttnWeights & w, const AttnParams & p, const KvLayer & kv,
                      const AttnInputs & in, const QsaWeights & qw, const QsaParams & qp, const QsaCache & qc);

// gated DeltaNet, qwen4exp variant
// Same weights / state / record as op::gdn_block; the output gate is RMSNorm(ssm_norm) * sigmoid(z), not silu(z).
// gated_delta_net_ext fuses the L2 norm of q/k, sigmoid(beta) and softplus(alpha + dt) * A.
Tensor gdn_block_sigz(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                      GdnRecord * rec = nullptr);

// PLE (per-layer token embedding)
struct PleWeights {
    Tensor key;         // [N, D]   ple_key
    Tensor value;       // [N, N]   ple_value
    Tensor norm_key;    // [D]      grouped RMS on the key
    Tensor norm_query;  // [D]      grouped RMS on the hidden streams
    Tensor norm_conv;   // [D]
    Tensor conv1d;      // [conv_k, D] depthwise causal conv, dilation = ngram
};
struct PleParams {
    int   n_heads = 0, head_dim = 0, ngram = 0, conv_k = 0;
    float eps = 0;
};
struct PleState {
    Tensor hist;        // f32 [(conv_k - 1) * ngram * D, B]  the conv history (the llama path keeps it in the recurrent row)
};
// build_ple: rows = this round's PLE embedding rows gathered on the host (f32 [N, T]: n_heads heads of head_dim, row
// ids from the n-gram hash) -> key/value, grouped norms, per-stream dot
// / sqrt(N) -> sigmoid(sgn * sqrt|s|) gate -> value * gate, RMS(norm_conv) -> causal conv (K shifted mul/adds) -> silu
// -> R + gated + conv. Updates s.hist (or leaves it to a fold, with `rec`, as gdn_block does).
Tensor ple_block(Device & d, const Tensor & R, const Tensor & rows, const PleWeights & w, const PleParams & p,
                 const PleState & s, bool rec = false, Tensor * rec_padded = nullptr);
// Commit the first m1 columns of a recorded PLE round: the history becomes columns [m1, m1 + hist) of the
// recorded history ++ new columns (`padded`, f32 [hist + T, D, 1], from ple_block's rec_padded).
void   ple_fold(Device & d, const PleParams & p, const PleState & s, const Tensor & padded, int64_t m1);

// MoE: softmax top-k, renormalised x scale; sigmoid-gated shared expert
// Reuses op::MoeWeights (sh_score = ffn_gate_inp_shexp [N]: one scalar gate per token). n_expert can be 512 with
// n_used 10, so the router_topk / topk_native kernels (hard-wired to 256 / 8) must not be used here.
struct MoeParamsX {
    int   n_expert = 0, n_used = 0;
    float weights_scale = 1.0f;   // expert_weights_scale
};
// Resident assembly (all experts on the device; the offload path uses moe_experts_raw below).
Tensor moe_block_x(Device & d, const Tensor & x, const MoeWeights & w, const MoeParamsX & p);

// Offload: routed experts over device slot banks (third axis = slots), unweighted f32 [N, k, T]; then the
// weighted sum plus the shared expert, after kern::offload_wait_fill spliced the CPU's missed columns in.
Tensor moe_experts_raw(Device & d, const Tensor & x, const Tensor & up, const Tensor & gate, const Tensor & down,
                       const Tensor & slots);
Tensor moe_weighted_sum(Device & d, const Tensor & exps, const Tensor & weights, const Tensor & shared, int k);

// MTP block input (as mtp4_builder::input in the llama path): R0 = h_proj(rms(H) * hnorm) per stream +
// eh_proj(rms(E) * enorm). H f32 [N*hc, T] (the trunk streams, one norm over all hc*N), E f32 [N, T] -> f32 [N, hc, T].
// Argmax of logits f32 [V, 1] and its softmax probability, on the device: idx i32 [1], p f32 [1]
struct ArgmaxP { Tensor idx, p; };
ArgmaxP argmax_p(Device & d, const Tensor & logits);

Tensor mtp_input(Device & d, const Tensor & H, const Tensor & E, const Tensor & eh_proj, const Tensor & h_proj,
                 const Tensor & enorm, const Tensor & hnorm, int hc, float eps);

}  // namespace ai::op::q4x
