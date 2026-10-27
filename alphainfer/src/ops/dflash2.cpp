#include "ops/dflash2.h"

#include "ops/adapter.h"
#include "ops/ops.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ai::op {

using adapter::Call;

// Two-tap grouped dynamic conv, one phase (dflash/model.py _grouped_dynamic_convolve):
//   out_i[h] = (base[ph,0,h] + dyn_i[ph,0,g]) * y_i[h] + (base[ph,1,h] + dyn_i[ph,1,g]) * y_{i-1}[h],  g = h / gs,
// y_{-1} = 0 (the shift stays inside the block). dyn is kernel_projection(n) [2*2*G, T], index (ph*2+tap)*G + g.
static ggml_tensor * conv_phase(ggml_context * g, ggml_tensor * y, ggml_tensor * dyn, ggml_tensor * base, int ph,
                                int64_t gs) {
    const int64_t E = y->ne[0], T = y->ne[1], G = E / gs;
    ggml_tensor * out = nullptr;
    for (int tap = 0; tap < 2; ++tap) {
        ggml_tensor * yy = y;
        if (tap == 1) {   // y_{i-1}: drop the last column, pad one zero column in front
            yy = ggml_pad_ext(g, ggml_cont(g, ggml_view_2d(g, y, E, T - 1, y->nb[1], 0)), 0, 0, 1, 0, 0, 0, 0, 0);
        }
        const size_t  k  = (size_t) (ph * 2 + tap);
        ggml_tensor * bk = ggml_view_1d(g, base, E, k * (size_t) E * sizeof(float));
        ggml_tensor * dk = ggml_cont(g, ggml_view_3d(g, dyn, 1, G, T, sizeof(float), dyn->nb[1],
                                                     k * (size_t) G * sizeof(float)));
        ggml_tensor * s = ggml_add(g, ggml_mul(g, yy, bk),
                                   ggml_reshape_2d(g, ggml_mul(g, ggml_reshape_3d(g, yy, gs, G, T), dk), E, T));
        out = out ? ggml_add(g, out, s) : s;
    }
    return out;
}

static ggml_tensor * rope_neox(ggml_context * g, ggml_tensor * x, ggml_tensor * pos, const Df2Params & p) {
    return ggml_rope_ext(g, x, pos, nullptr, p.head_dim, GGML_ROPE_TYPE_NEOX, 0, p.rope_base, 1.0f, 0.0f, 1.0f,
                         0.0f, 0.0f);
}

void df2_ctx_append(Device & d, const Df2Weights & w, const Df2Params & p, const Df2Kv & kv, const Tensor & taps,
                    const Tensor & pos, const Tensor & cells, const Tensor & cells_lin) {
    const int64_t n = taps.ne[1], D = p.head_dim, Hkv = p.n_head_kv;
    Call          c(d);
    ggml_context * g   = c.ctx();
    ggml_tensor *  ctx = ggml_mul(g, ggml_rms_norm(g, ggml_mul_mat(g, c.in(w.ctx_fc, "df2_fc"), c.in(taps, "df2_taps")),
                                                   p.eps),
                                  c.in(w.ctx_norm));
    ggml_tensor *  gp  = c.in(pos, "df2_ctx_pos");
    ggml_tensor *  gc  = c.in(cells, "df2_ctx_cells");
    ggml_tensor *  gl  = cells_lin.valid() ? c.in(cells_lin, "df2_ctx_lin") : nullptr;
    for (size_t il = 0; il < w.layers.size(); ++il) {
        const Df2Layer & L = w.layers[il];
        ggml_tensor *    lc = L.sliding ? gc : gl;
        if (lc == nullptr) {
            throw std::runtime_error("df2_ctx_append: a full-attention layer needs linear cells");
        }
        ggml_tensor * k = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(L.k), ctx), D, Hkv, n);
        k = rope_neox(g, ggml_mul(g, ggml_rms_norm(g, k, p.eps), c.in(L.k_norm)), gp, p);
        ggml_tensor * v = ggml_mul_mat(g, c.in(L.v), ctx);
        c.effect(ggml_set_rows(g, c.in(kv.k[il]), ggml_reshape_2d(g, k, D * Hkv, n), lc));
        c.effect(ggml_set_rows(g, c.in(kv.v[il]), v, lc));
    }
    c.run_effects_only();
}

// One layer per Call: a Call graph holds at most 512 nodes and a layer is ~150 (the conv phases dominate).
static Tensor df2_layer(Device & d, const Df2Layer & L, const Df2Params & p, const Tensor & kcache,
                        const Tensor & vcache, const Tensor & x, const Tensor & pos, const Tensor & block_cells,
                        const Tensor & mask, int64_t n_kv, const Tensor * final_norm) {
    const int64_t T = x.ne[1], D = p.head_dim, Hq = p.n_head, Hkv = p.n_head_kv;
    Call          c(d);
    ggml_context * g  = c.ctx();
    ggml_tensor *  h  = c.in(x, "df2_x");
    ggml_tensor *  gp = c.in(pos, "df2_pos");
    // attention branch: conv phase 0 on the normed input, phase 1 on the o-projection
    // DFlash2 wraps attention and MLP in grouped dynamic convs; DFlash v1 has none (aconv_proj invalid)
    const bool    cv = L.aconv_proj.valid();
    ggml_tensor * n  = ggml_mul(g, ggml_rms_norm(g, h, p.eps), c.in(L.attn_norm));
    ggml_tensor * dA = cv ? ggml_mul_mat(g, c.in(L.aconv_proj), n) : nullptr;
    ggml_tensor * ab = cv ? c.in(L.aconv_base) : nullptr;
    ggml_tensor * a  = cv ? conv_phase(g, n, dA, ab, 0, p.group_size) : n;

    ggml_tensor * q = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(L.q), a), D, Hq, T);
    ggml_tensor * k = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(L.k), a), D, Hkv, T);
    ggml_tensor * v = ggml_mul_mat(g, c.in(L.v), a);
    q = rope_neox(g, ggml_mul(g, ggml_rms_norm(g, q, p.eps), c.in(L.q_norm)), gp, p);
    k = rope_neox(g, ggml_mul(g, ggml_rms_norm(g, k, p.eps), c.in(L.k_norm)), gp, p);

    // the block's K/V into its scratch cells; the attention reads views of the writes, so the
    // write-before-read order is a graph dependency rather than a property of the node order
    ggml_tensor * gc  = c.in(block_cells, "df2_cells");
    ggml_tensor * kc  = ggml_set_rows(g, c.in(kcache), ggml_reshape_2d(g, k, D * Hkv, T), gc);
    ggml_tensor * vc  = ggml_set_rows(g, c.in(vcache), v, gc);
    const size_t  rD  = ggml_row_size(kc->type, D);
    const size_t  rDH = ggml_row_size(kc->type, D * Hkv);
    ggml_tensor * kp  = ggml_permute(g, ggml_view_3d(g, kc, D, Hkv, n_kv, rD, rDH, 0), 0, 2, 1, 3);
    ggml_tensor * vp  = ggml_permute(g, ggml_view_3d(g, vc, D, Hkv, n_kv, rD, rDH, 0), 0, 2, 1, 3);
    ggml_tensor * qp  = ggml_permute(g, q, 0, 2, 1, 3);
    ggml_tensor * fa  = ggml_flash_attn_ext(g, qp, kp, vp, c.in(mask, "df2_mask"), 1.0f / sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * o = ggml_mul_mat(g, c.in(L.o), ggml_reshape_2d(g, fa, D * Hq, T));
    h = ggml_add(g, h, cv ? conv_phase(g, o, dA, ab, 1, p.group_size) : o);

    // MLP branch
    n = ggml_mul(g, ggml_rms_norm(g, h, p.eps), c.in(L.ffn_norm));
    ggml_tensor * dM = cv ? ggml_mul_mat(g, c.in(L.fconv_proj), n) : nullptr;
    ggml_tensor * fb = cv ? c.in(L.fconv_base) : nullptr;
    ggml_tensor * m  = cv ? conv_phase(g, n, dM, fb, 0, p.group_size) : n;
    ggml_tensor * u  = ggml_mul(g, ggml_silu(g, ggml_mul_mat(g, c.in(L.gate), m)), ggml_mul_mat(g, c.in(L.up), m));
    ggml_tensor * dn = ggml_mul_mat(g, c.in(L.down), u);
    h = ggml_add(g, h, cv ? conv_phase(g, dn, dM, fb, 1, p.group_size) : dn);
    if (final_norm) {
        h = ggml_mul(g, ggml_rms_norm(g, h, p.eps), c.in(*final_norm));
    }
    return c.run(h);
}

Tensor df2_block(Device & d, const Df2Weights & w, const Df2Params & p, const Df2Kv & kv, const Tensor & x,
                 const Tensor & pos, const Tensor & block_cells, const Tensor & mask, int64_t n_kv,
                 const Tensor & block_lin, const Tensor & mask_full, int64_t n_full) {
    if (mask.ne[0] != n_kv || mask.ne[1] < x.ne[1] || mask.type != F16) {
        throw std::runtime_error("df2_block: mask must be f16 [n_kv, >=T]");
    }
    Tensor h = x;
    for (size_t il = 0; il < w.layers.size(); ++il) {
        const bool last = il + 1 == w.layers.size();
        const bool sl = w.layers[il].sliding;
        if (!sl && (!block_lin.valid() || !mask_full.valid() || mask_full.ne[0] != n_full)) {
            throw std::runtime_error("df2_block: a full-attention layer needs linear block cells and mask_full");
        }
        h = df2_layer(d, w.layers[il], p, kv.k[il], kv.v[il], h, pos, sl ? block_cells : block_lin,
                      sl ? mask : mask_full, sl ? n_kv : n_full, last ? &w.out_norm : nullptr);
        static const bool trace = getenv("DF2_TRACE") != nullptr && !recording();
        if (trace) {
            d.synchronize();
            fprintf(stderr, "[trace]     layer %zu ok\n", il);
        }
    }
    return h;
}

Df2TopK df2_topk(Device & d, const Tensor & logits, int k) {
    const int64_t V = logits.ne[0], n = logits.ne[1];
    // AI_DF2_TOPK=ggml: use ggml_top_k (a full segmented sort, slower) instead, for comparison
    static const bool k_ggml = getenv("AI_DF2_TOPK") && std::string(getenv("AI_DF2_TOPK")) == "ggml";
    if (!k_ggml && logits.type == F32 && logits.nb[0] == 4) {
        Df2TopK out;
        out.ids  = d.ws_alloc(I32, k, n);
        out.vals = d.ws_alloc(F32, k, n);
        int          wbuf = -1;
        void *       work = d.ws_alloc_raw((size_t) kern::topk_rows_workspace(n, k), &wbuf);
        void *       st   = adapter::stream(d);
        const float * px  = (const float *) logits.data;
        const int64_t rs  = (int64_t) logits.nb[1] / 4;
        int32_t *    pi   = (int32_t *) out.ids.data;
        float *      pv   = (float *) out.vals.data;
        adapter::native(d, [=]() {
            if (!kern::topk_rows(st, px, V, rs, n, k, work, pi, pv)) {
                fprintf(stderr, "[df2] topk_rows declined V=%lld k=%d" "\n", (long long) V, k);
                abort();
            }
        });
        return out;
    }
    Call          c(d);
    ggml_context * g   = c.ctx();
    ggml_tensor *  lg  = c.in(logits, "df2_logits");
    ggml_tensor *  ids = ggml_top_k(g, lg, k);                                        // i32 [k, n]
    ggml_tensor *  val = ggml_get_rows(g, ggml_reshape_3d(g, lg, 1, V, n), ids);      // f32 [1, k, n]
    c.effect(ids);
    Df2TopK out;
    out.vals = c.run(ggml_reshape_2d(g, val, k, n));
    out.ids  = c.to_public(ids);
    return out;
}

void df2_round_inputs(Device & d, const Tensor & base, const Tensor * m, const Tensor & g, int32_t mask_id,
                      int block, int64_t window, int64_t C, const Df2RoundIn & o, bool causal, const Tensor & mask_full) {
    void *          st = adapter::stream(d);
    const int32_t * pb = (const int32_t *) base.data, *pm = m ? (const int32_t *) m->data : nullptr;
    const int32_t * pg = (const int32_t *) g.data;
    int32_t *       cp = (int32_t *) o.ctx_pos.data, *bi = (int32_t *) o.blk_ids.data, *bp = (int32_t *) o.blk_pos.data;
    int64_t *       cc = (int64_t *) o.ctx_cells.data, *bc = (int64_t *) o.blk_cells.data;
    uint16_t *      mk = (uint16_t *) o.mask.data;
    const int64_t   nc = o.mask.ne[0];
    int64_t *       cl = o.ctx_lin.valid() ? (int64_t *) o.ctx_lin.data : nullptr;
    int64_t *       bl = o.blk_lin.valid() ? (int64_t *) o.blk_lin.data : nullptr;
    uint16_t *      mf = mask_full.valid() ? (uint16_t *) mask_full.data : nullptr;
    const int64_t   nf = mask_full.valid() ? mask_full.ne[0] : 0;
    const int       ca = causal ? 1 : 0;
    adapter::native(d, [=]() {
        kern::df2_round_inputs(st, pb, pm, pg, mask_id, block, window, C, nc, ca, cp, cc, bi, bp, bc, mk, nf, cl, bl,
                               mf);
    });
}

void df2_copy_ids(Device & d, const Tensor & src, const Tensor & dst, int64_t n) {
    void *          st = adapter::stream(d);
    const int32_t * ps = (const int32_t *) src.data;
    int32_t *       pd = (int32_t *) dst.data;
    adapter::native(d, [=]() { kern::copy_i32(st, ps, pd, n); });
}

void df2_map_ids(Device & d, const Tensor & ids, const Tensor & map) {
    void *          st = adapter::stream(d);
    int32_t *       pi = (int32_t *) ids.data;
    const int32_t * pm = (const int32_t *) map.data;
    const int64_t   n  = ids.numel();
    adapter::native(d, [=]() { kern::map_ids(st, pi, n, pm); });
}

void df2_select(Device & d, const Df2TopK & tk, const Tensor & hh, const Tensor & pred_cb, const Tensor & succ_cb,
                const Tensor & first_pred, const Tensor & drafts) {
    void *          st = adapter::stream(d);
    const int32_t * ci = (const int32_t *) tk.ids.data;
    const float *   cv = (const float *) tk.vals.data, *ph = (const float *) hh.data;
    const void *    pc = pred_cb.data, *sc = succ_cb.data;
    const int32_t * fp = (const int32_t *) first_pred.data;
    int32_t *       dr = (int32_t *) drafts.data;
    const int       k = (int) tk.ids.ne[0], rank = (int) hh.ne[0], nd = (int) hh.ne[1];
    if (tk.ids.nb[1] != (size_t) k * 4 || tk.vals.nb[1] != (size_t) k * 4 || hh.nb[1] != (size_t) rank * 4) {
        throw std::runtime_error("df2_select: top-k and hidden projection must be contiguous");
    }
    adapter::native(d, [=]() {
        if (!kern::df2_select(st, ci, cv, ph, pc, sc, fp, 0, k, rank, nd, dr)) {
            fprintf(stderr, "[df2] selector declined k=%d rank=%d\n", k, rank);
            abort();
        }
    });
}

}  // namespace ai::op
