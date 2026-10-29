// SPDX-License-Identifier: Apache-2.0
// DFlash2 block-diffusion drafter, as closed Ops. Reference: z-lab/dflash dflash/model.py @ 07ebd93.
//
// ggml expressions on the adapter, eager or recorded like every other Op.
#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <vector>

namespace ai::op {

struct Df2Layer {
    Tensor attn_norm, ffn_norm;                   // f32 [E]
    Tensor q, k, v, o;                            // [E, D*Hq], [E, D*Hkv] x2, [D*Hq, E]
    Tensor q_norm, k_norm;                        // f32 [D]
    Tensor gate, up, down;                        // [E, F] x2, [F, E]
    Tensor aconv_base, fconv_base;                // f32 [E, 2 taps, 2 phases]   (DFlash2 only; invalid in v1)
    Tensor aconv_proj, fconv_proj;                // [E, 2 phases * 2 taps * G]
    bool   sliding = true;                        // sliding-window layer (cyclic KV); else full attention (linear KV)
};
struct Df2Weights {
    Tensor ctx_fc;                                // [n_taps*E, E]
    Tensor ctx_norm, out_norm;                    // f32 [E]
    std::vector<Df2Layer> layers;
};
struct Df2Params {
    int   n_embd, n_head, n_head_kv, head_dim, group_size;
    float eps, rope_base;
};
// Per-layer context K/V: f16 [D*Hkv, n_cells]. Cells [0, window) are the cyclic context; the block's own K/V
// go to scratch cells after it and are never kept.
struct Df2Kv {
    std::vector<Tensor> k, v;
};

// Context append: taps f32 [n_taps*E, n] -> ctx = rmsnorm(fc(taps)) * ctx_norm; per layer K = RoPE(k_norm(k(ctx)))
// at pos (i32 [n]) and V = v(ctx), written at cells (i64 [n], sliding layers) or cells_lin (full-attention layers;
// may be invalid when there are none).
void df2_ctx_append(Device & d, const Df2Weights & w, const Df2Params & p, const Df2Kv & kv, const Tensor & taps,
                    const Tensor & pos, const Tensor & cells, const Tensor & cells_lin = Tensor{});

// Block forward over x f32 [E, B] (embeddings of [anchor, MASK x (B-1)]) at pos i32 [B]. The block's K/V are
// written at block_cells (i64 [B]); attention reads cells [0, n_kv) under mask f16 [n_kv, B] (0 / -inf), which
// the caller builds from the sliding-window rule and the block cells, non-causal. Returns the final-normed
// hidden f32 [E, B] (all B columns; the drafter uses 1..B-1).
// Full-attention layers (DFlash v1) use block_lin / mask_full [n_full, B] / n_full instead.
Tensor df2_block(Device & d, const Df2Weights & w, const Df2Params & p, const Df2Kv & kv, const Tensor & x,
                 const Tensor & pos, const Tensor & block_cells, const Tensor & mask, int64_t n_kv,
                 const Tensor & block_lin = Tensor{}, const Tensor & mask_full = Tensor{}, int64_t n_full = 0);

// Top-k per column of logits f32 [V, n]: ids i32 [k, n] (no particular order) and their logits f32 [k, n].
struct Df2TopK {
    Tensor ids, vals;
};
Df2TopK df2_topk(Device & d, const Tensor & logits, int k);

// Per-round drafter inputs, on the device (kernels/dflash2.cu df2_round_inputs). m == nullptr: no verify,
// the anchor is g[0] at position base[0]. Outputs are the caller's persistent tensors.
struct Df2RoundIn {
    Tensor ctx_pos, ctx_cells, blk_ids, blk_pos, blk_cells, mask;
    Tensor ctx_lin, blk_lin;   // full-attention layers (DFlash v1): cell = position; invalid when there are none
};
// causal: the sliding mask stops each block query at its own column (DFlash v1). mask_full (may be invalid):
// f16 [n_full, B] for full-attention layers.
void df2_round_inputs(Device & d, const Tensor & base, const Tensor * m, const Tensor & g, int32_t mask_id,
                      int block, int64_t window, int64_t C, const Df2RoundIn & out, bool causal = false,
                      const Tensor & mask_full = Tensor{});
// dst[0..n) = src[0..n) (i32), on the device
void df2_copy_ids(Device & d, const Tensor & src, const Tensor & dst, int64_t n);
// ids[i] = map[ids[i]] in place (truncated draft head: row index -> token id)
void df2_map_ids(Device & d, const Tensor & ids, const Tensor & map);
// Greedy coherent selector into drafts (i32 [nd]); the first predecessor is first_pred[0].
void df2_select(Device & d, const Df2TopK & tk, const Tensor & hh, const Tensor & pred_cb, const Tensor & succ_cb,
                const Tensor & first_pred, const Tensor & drafts);

}  // namespace ai::op
