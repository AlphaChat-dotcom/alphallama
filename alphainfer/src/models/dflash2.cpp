#include "models/dflash2.h"

#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace ai {

static std::string L(int il, const char * name) { return "blk." + std::to_string(il) + "." + name; }

bool DFlash2::load(Device & dev, const std::string & path, bool verbose) {
    if (!w_.open(path)) {
        return false;
    }
    if (w_.arch() != "dflash2") {
        fprintf(stderr, "[df2] %s: arch '%s', expected 'dflash2'\n", path.c_str(), w_.arch().c_str());
        return false;
    }
    const std::string a = "dflash2.";
    variant_ = w_.has_key(a + "variant") ? (int) w_.get_int(a + "variant") : 2;
    p_.n_embd     = (int) w_.get_int(a + "embedding_length");
    p_.n_head     = (int) w_.get_int(a + "attention.head_count");
    p_.n_head_kv  = (int) w_.get_int(a + "attention.head_count_kv");
    p_.head_dim   = (int) w_.get_int(a + "attention.key_length");
    p_.group_size = variant_ == 2 ? (int) w_.get_int(a + "conv_group_size") : 0;
    p_.eps        = (float) w_.get_float(a + "attention.layer_norm_rms_epsilon");
    p_.rope_base  = (float) w_.get_float(a + "rope.freq_base");
    block_   = (int) w_.get_int(a + "block_size");
    window_  = (int) w_.get_int(a + "attention.sliding_window");
    top_k_   = variant_ == 2 ? (int) w_.get_int(a + "selector_top_k") : 1;
    rank_    = variant_ == 2 ? (int) w_.get_int(a + "selector_rank") : 0;
    mask_id_ = (int32_t) w_.get_int(a + "mask_token_id");
    vocab_   = w_.get_int(a + "vocab_size");
    // AI_DF2_BLOCK: draft with a shorter block than trained (the block's queries only attend to the context and to
    // each other, so a prefix of the block is a valid block).
    // v1 defaults to 8 for that reason; v2 keeps its trained block.
    {
        const int trained = block_;
        const char * e = getenv("AI_DF2_BLOCK");
        int want = e ? atoi(e) : (variant_ == 1 ? 8 : trained);
        if (want < 2 || want > trained) {
            fprintf(stderr, "[df2] block %d out of range [2, %d]; using %d\n", want, trained, trained);
            want = trained;
        }
        block_ = want;
    }
    if (variant_ == 2 && w_.get_int(a + "conv_kernel_size") != 2) {
        fprintf(stderr, "[df2] conv_kernel_size != 2 is not implemented\n");
        return false;
    }
    for (int64_t t : w_.get_int_array(a + "target_layer_ids")) {
        target_layers_.push_back((int) t);
    }
    if (!w_.load(dev, 4ULL << 30, verbose)) {
        return false;
    }
    const int n_layer = (int) w_.get_int(a + "block_count");
    std::vector<int64_t> sl, ca;
    if (w_.has_key(a + "layer_sliding")) {
        sl = w_.get_int_array(a + "layer_sliding");
        ca = w_.get_int_array(a + "layer_causal");
    }
    if (sl.empty()) {
        sl.assign((size_t) n_layer, 1);   // DFlash2: every layer sliding, non-causal
        ca.assign((size_t) n_layer, 0);
    }
    if ((int) sl.size() != n_layer || (int) ca.size() != n_layer) {
        fprintf(stderr, "[df2] layer_sliding/layer_causal: %zu/%zu entries for %d layers\n", sl.size(), ca.size(),
                n_layer);
        return false;
    }
    {
        int n_sl = 0, n_ca = 0;
        for (int il = 0; il < n_layer; ++il) {
            n_sl += sl[(size_t) il] != 0;
            n_ca += sl[(size_t) il] != 0 && ca[(size_t) il] != 0;
            if (sl[(size_t) il] == 0 && ca[(size_t) il] != 0) {
                fprintf(stderr, "[df2] a causal full-attention layer is not implemented\n");
                return false;
            }
        }
        if (n_ca != 0 && n_ca != n_sl) {
            fprintf(stderr, "[df2] mixed causal/non-causal sliding layers are not implemented\n");
            return false;
        }
        causal_   = n_ca != 0;
        has_full_ = n_sl != n_layer;
    }
    W_.ctx_fc   = w_.get("ctx_fc.weight");
    W_.ctx_norm = w_.get("ctx_norm.weight");
    W_.out_norm = w_.get("output_norm.weight");
    for (int il = 0; il < n_layer; ++il) {
        op::Df2Layer l;
        l.attn_norm  = w_.get(L(il, "attn_norm.weight"));
        l.ffn_norm   = w_.get(L(il, "ffn_norm.weight"));
        l.q          = w_.get(L(il, "attn_q.weight"));
        l.k          = w_.get(L(il, "attn_k.weight"));
        l.v          = w_.get(L(il, "attn_v.weight"));
        l.o          = w_.get(L(il, "attn_output.weight"));
        l.q_norm     = w_.get(L(il, "attn_q_norm.weight"));
        l.k_norm     = w_.get(L(il, "attn_k_norm.weight"));
        l.gate       = w_.get(L(il, "ffn_gate.weight"));
        l.up         = w_.get(L(il, "ffn_up.weight"));
        l.down       = w_.get(L(il, "ffn_down.weight"));
        if (variant_ == 2) {
            l.aconv_base = w_.get(L(il, "attn_conv.base"));
            l.aconv_proj = w_.get(L(il, "attn_conv.proj.weight"));
            l.fconv_base = w_.get(L(il, "ffn_conv.base"));
            l.fconv_proj = w_.get(L(il, "ffn_conv.proj.weight"));
        }
        l.sliding = sl[(size_t) il] != 0;
        W_.layers.push_back(l);
    }
    if (variant_ == 2) {
        sel_proj_ = w_.get("sel_proj.weight");
        pred_cb_  = w_.get("sel_pred_cb");
        succ_cb_  = w_.get("sel_succ_cb");
        for (const Tensor * t : {&pred_cb_, &succ_cb_}) {
            if (t->type != BF16 || t->ne[0] != rank_ || t->ne[1] != vocab_) {
                fprintf(stderr, "[df2] codebook: expected BF16 [%d, %lld], got %s %s\n", rank_, (long long) vocab_,
                        dtype_name(t->type), t->shape_str().c_str());
                return false;
            }
        }
    }
    fprintf(stderr, "[df2] v%d: %d layers (%d full-attention), E %d, block %d, window %d%s, taps %zu, %s, ctx_fc %s, "
                    "%.2f GB\n",
            variant_, n_layer, (int) std::count(sl.begin(), sl.end(), 0), p_.n_embd, block_, window_,
            causal_ ? " causal" : "", target_layers_.size(),
            variant_ == 2 ? ("top-" + std::to_string(top_k_) + " rank " + std::to_string(rank_) + " selector").c_str()
                          : "argmax drafts",
            dtype_name(W_.ctx_fc.type), (double) w_.bytes() / 1e9);
    return true;
}

void DFlash2::alloc(Device & dev, int64_t max_ctx) {
    // cyclic context over C = window + block (so the uncommitted verify rows never reach an in-window slot),
    // then the block's scratch cells, rounded up to 256 for the attention kernels
    C_       = (int64_t) window_ + block_;
    n_cells_ = (C_ + block_ + 255) / 256 * 256;
    const int64_t row = (int64_t) p_.head_dim * p_.n_head_kv;
    auto mk = [&](int type, int64_t n0, int64_t n1, const char * tag) {
        Tensor t;
        t.type  = type;
        t.ne[0] = n0;
        t.ne[1] = n1;
        set_contiguous_strides(t);
        t.buf  = dev.alloc_buffer(t.nbytes() + dev.alignment(), tag);
        t.data = dev.buffer_base(t.buf);
        return t;
    };
    // full-attention layers: linear, cell = position; a round reads up to (target n_kv) + block cells
    n_lin_ = has_full_ ? full_cells(std::max<int64_t>(max_ctx, 1) + block_) : 0;
    for (size_t il = 0; il < W_.layers.size(); ++il) {
        const int64_t n = W_.layers[il].sliding ? n_cells_ : n_lin_;
        kv_.k.push_back(mk(F16, row, n, "df2_kv"));
        kv_.v.push_back(mk(F16, row, n, "df2_kv"));
    }
    if (has_full_) {
        in_.ctx_lin = mk(I64, block_, 1, "df2_in");
        in_.blk_lin = mk(I64, block_, 1, "df2_in");
        mask_full_  = mk(F16, n_lin_ * block_, 1, "df2_in");
    }
    in_.ctx_pos   = mk(I32, block_, 1, "df2_in");
    in_.ctx_cells = mk(I64, block_, 1, "df2_in");
    in_.blk_ids   = mk(I32, block_, 1, "df2_in");
    in_.blk_pos   = mk(I32, block_, 1, "df2_in");
    in_.blk_cells = mk(I64, block_, 1, "df2_in");
    in_.mask      = mk(F16, n_cells_, block_, "df2_in");
    base_   = mk(I32, 1, 1, "df2_in");
    g_      = mk(I32, 1, 1, "df2_in");
    drafts_ = mk(I32, block_ - 1, 1, "df2_out");
    reset();
}

void DFlash2::reset() { n_ctx_ = 0; }

int64_t DFlash2::full_cells(int64_t visible) const { return (visible + 255) / 256 * 256; }

Tensor DFlash2::mask_full(int64_t n_full) const {
    if (!has_full_) {
        return Tensor{};
    }
    if (n_full > n_lin_) {
        throw std::runtime_error("df2: the full-attention context is larger than max_ctx");
    }
    return mask_full_.view(n_full, block_, 1, 1, (size_t) n_full * 2, (size_t) n_full * 2 * block_,
                           (size_t) n_full * 2 * block_, 0);
}

void DFlash2::ctx_append(Device & dev, const Tensor & taps, int64_t pos0) {
    const int64_t n = taps.ne[1];
    if (taps.ne[0] != tap_width() || taps.type != F32) {
        throw std::runtime_error("df2 ctx_append: taps must be f32 [n_taps*E, n]");
    }
    // Device::upload does not order against the compute stream, and this workspace was just handed back by
    // the previous call's Scope while its kernels may still be reading it: drain first.
    dev.synchronize();
    Device::Scope        sc(dev);
    std::vector<int32_t> pos((size_t) n);
    std::vector<int64_t> cells((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        pos[(size_t) i]   = (int32_t) (pos0 + i);
        cells[(size_t) i] = (pos0 + i) % C_;
    }
    Tensor tp = dev.ws_alloc(I32, n);
    Tensor tc = dev.ws_alloc(I64, n);
    dev.upload(tp, pos.data(), 0, (size_t) n * 4);
    dev.upload(tc, cells.data(), 0, (size_t) n * 8);
    Tensor tl;
    if (has_full_) {
        if (pos0 + n + block_ > n_lin_) {
            throw std::runtime_error("df2 ctx_append: the context is larger than max_ctx");
        }
        std::vector<int64_t> lin((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            lin[(size_t) i] = pos0 + i;
        }
        tl = dev.ws_alloc(I64, n);
        dev.upload(tl, lin.data(), 0, (size_t) n * 8);
    }
    op::df2_ctx_append(dev, W_, p_, kv_, taps, tp, tc, tl);
    n_ctx_ = pos0 + n;
}

void DFlash2::draft_core(Device & dev, const Tensor & embed, const Tensor & head_w, const Tensor * head_map,
                         int64_t n_full) {
    const int64_t E = p_.n_embd, nd = block_ - 1;
    const Tensor  x  = op::embed(dev, embed, in_.blk_ids);
    const Tensor  h  = op::df2_block(dev, W_, p_, kv_, x, in_.blk_pos, in_.blk_cells, in_.mask, n_cells_,
                                     in_.blk_lin, mask_full(n_full), has_full_ ? n_full : 0);
    const Tensor  hd = h.view(E, nd, 1, 1, h.nb[1], h.nb[1] * nd, h.nb[1] * nd, h.nb[1]);   // columns 1..B-1
    const Tensor  lg = op::linear(dev, head_w, hd);      // [V, nd]
    if (variant_ == 1) {
        // DFlash v1: the draft is the head's argmax (no selector)
        const Tensor am = op::argmax(dev, lg, head_map);
        op::df2_copy_ids(dev, am, drafts_, nd);
        dbg_h_ = hd;
        return;
    }
    const Tensor  hh = op::linear(dev, sel_proj_, hd);   // [rank, nd]
    const auto    tk = op::df2_topk(dev, lg, top_k_);   // ids/logits [top_k, nd]
    if (head_map) {
        op::df2_map_ids(dev, tk.ids, *head_map);   // truncated draft head: row -> token id
    }
    op::df2_select(dev, tk, hh, pred_cb_, succ_cb_, in_.blk_ids, drafts_);
    dbg_h_  = hd;
    dbg_tk_ = tk;
}

Tensor DFlash2::round(Device & dev, const Tensor & taps, const Tensor & base, const Tensor & m, const Tensor & g,
                      const Tensor & embed, const Tensor & head_w, const Tensor * head_map, int64_t n_kv) {
    if (taps.ne[0] != tap_width() || taps.ne[1] != block_) {
        throw std::runtime_error("df2 round: taps must be the block_size verify rows");
    }
    // the anchor sits at base + m + 1 <= base + block, inside the target's n_kv window + 1; its block ends block
    // cells later
    const int64_t nf = has_full_ ? full_cells(n_kv + block_ + 1) : 0;
    op::df2_round_inputs(dev, base, &m, g, mask_id_, block_, window_, C_, in_, causal_, mask_full(nf));
    op::df2_ctx_append(dev, W_, p_, kv_, taps, in_.ctx_pos, in_.ctx_cells, in_.ctx_lin);
    draft_core(dev, embed, head_w, head_map, nf);
    return drafts_;
}

DFlash2::Draft DFlash2::draft(Device & dev, int32_t anchor, int64_t pos, const Tensor & embed, const Tensor & head_w,
                              bool debug, const Tensor * head_map) {
    const int64_t nd = block_ - 1, E = p_.n_embd;
    if (pos != n_ctx_) {
        throw std::runtime_error("df2 draft: the anchor must directly follow the context");
    }
    dev.synchronize();   // see ctx_append: the uploads below are not ordered against queued kernels
    Device::Scope sc(dev);
    const int32_t p32 = (int32_t) pos;
    dev.upload(base_, &p32, 0, 4);
    dev.upload(g_, &anchor, 0, 4);
    const int64_t nf = has_full_ ? full_cells(pos + block_) : 0;
    op::df2_round_inputs(dev, base_, nullptr, g_, mask_id_, block_, window_, C_, in_, causal_, mask_full(nf));
    draft_core(dev, embed, head_w, head_map, nf);
    Draft out;
    out.tokens.resize((size_t) nd);
    dev.download(drafts_, out.tokens.data(), 0, (size_t) nd * 4);
    if (debug) {
        out.h.resize((size_t) (E * nd));
        dev.download(dbg_h_, out.h.data(), 0, out.h.size() * 4);
    }
    if (debug && variant_ == 2) {
        out.cand.resize((size_t) (top_k_ * nd));
        dev.download(dbg_tk_.ids, out.cand.data(), 0, out.cand.size() * 4);
        for (int64_t t = 0; t < nd; ++t) {
            std::sort(out.cand.begin() + t * top_k_, out.cand.begin() + (t + 1) * top_k_);
        }
    }
    return out;
}

}  // namespace ai
