// SPDX-License-Identifier: Apache-2.0
// DFlash2 drafter (Qwen3.8-27B): weights, cyclic context KV and one draft round. Also DFlash v1
// (z-lab/Qwen3.6-35B-A3B-DFlash, dflash2.variant = 1): no convs, no selector (drafts = argmax of the head),
// causal sliding layers and full-attention layers, which keep a linear context KV (cell = position).
//
// Everything runs on the device through recordable Ops (ops/dflash2.h, kernels/dflash2.cu): per-round inputs
// from the verify's accept count, the forward, top-k and the coherent selector. round() can therefore sit in
// the captured decode round; ctx_append()/draft() are the host-driven forms (prefill, tools/dflash2_check.cpp).
#pragma once

#include "core/device.h"
#include "core/weights.h"
#include "ops/dflash2.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ai {

class DFlash2 {
public:
    bool load(Device & dev, const std::string & path, bool verbose);
    void alloc(Device & dev, int64_t max_ctx);   // before finalize: the context KV (max_ctx sizes full layers)
    void reset();               // forget the context

    // Append n target tap rows (f32 [n_taps*E, n], device) at absolute positions pos0 .. pos0+n-1. Host-driven.
    void ctx_append(Device & dev, const Tensor & taps, int64_t pos0);
    // Device-driven round, recordable: append all block_size verify rows (taps f32 [n_taps*E, block]) at
    // base[0].., then draft from the anchor g[m] at base+m+1. Returns the drafts (persistent i32 [block-1]).
    // head_map: when head_w is a truncated draft head, its row index -> token id map (null = rows are ids)
    // n_kv: the target round's visible cells (bounds full-attention layers: they read n_kv + block cells)
    Tensor round(Device & dev, const Tensor & taps, const Tensor & base, const Tensor & m, const Tensor & g,
                 const Tensor & embed, const Tensor & head_w, const Tensor * head_map, int64_t n_kv);

    struct Draft {
        std::vector<int32_t> tokens;   // block_size-1 drafts
        std::vector<float>   h;        // [B-1][E] final-normed hidden (when asked)
        std::vector<int32_t> cand;     // [B-1][top_k] candidate ids, sorted ascending (when asked)
    };
    // One draft round for `anchor` at absolute position `pos`, using the target's embedding table and LM head.
    // Host-driven (uploads the anchor, downloads the drafts).
    Draft draft(Device & dev, int32_t anchor, int64_t pos, const Tensor & embed, const Tensor & head_w, bool debug,
                const Tensor * head_map = nullptr);

    int                      block_size() const { return block_; }
    int32_t                  mask_id() const { return mask_id_; }
    const std::vector<int> & target_layers() const { return target_layers_; }
    int64_t                  tap_width() const { return (int64_t) target_layers_.size() * p_.n_embd; }
    int64_t                  ctx_rows() const { return n_ctx_; }
    int64_t                  cycle() const { return C_; }
    int                      variant() const { return variant_; }

private:
    // the block forward + head + top-k + selector from the round inputs in in_, into drafts_; keeps debug views
    void draft_core(Device & dev, const Tensor & embed, const Tensor & head_w, const Tensor * head_map,
                    int64_t n_full);
    int64_t full_cells(int64_t visible) const;   // full-layer cells read for `visible` positions (256-padded)
    Tensor  mask_full(int64_t n_full) const;     // view of mask_full_ as f16 [n_full, block]

    WeightSet         w_;
    op::Df2Weights    W_;
    op::Df2Params     p_{};
    op::Df2Kv         kv_;
    int               block_ = 0, window_ = 0, top_k_ = 0, rank_ = 0;
    int32_t           mask_id_ = 0;
    int64_t           C_ = 0, n_cells_ = 0, n_ctx_ = 0;   // cycle (window + block), cells incl. scratch + pad
    int               variant_ = 2;
    bool              causal_ = false, has_full_ = false;
    int64_t           n_lin_ = 0;      // full-attention layers' linear cells
    Tensor            mask_full_;      // f16 [n_lin_ * block] backing store
    std::vector<int>  target_layers_;
    Tensor            sel_proj_, pred_cb_, succ_cb_;
    int64_t           vocab_ = 0;
    op::Df2RoundIn    in_;            // persistent per-round inputs (written by the inputs kernel)
    Tensor            base_, g_, drafts_;   // host-driven staging: base [1], anchor [1]; drafts [block-1]
    Tensor            dbg_h_;         // last forward's hidden columns 1.. (debug)
    op::Df2TopK       dbg_tk_;
};

}  // namespace ai
