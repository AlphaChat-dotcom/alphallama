// SPDX-License-Identifier: Apache-2.0
#include "ops/ops.h"

#include "ops/adapter.h"
#include "kernels/kernels.h"
#include "kernels/moe_down9.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai::op {

// AI_DUMP_DIR router tap. The model's per-layer dump covers attn/ffn boundaries but not the MoE
// block's interior. These files separate three cases: router logits differ (router GEMV or
// upstream), logits equal but expert ids differ (top-k), or both equal but the output differs
// (below the router).
//
// Same file format as qwen35_hybrid.cpp's dump(): int64 cols, int64 rows, then the payload.
// Fires only when AI_DUMP_DIR is set and not recording, so it cannot perturb a captured round.
static int g_dump_layer = -1;
void set_dump_layer(int il) { g_dump_layer = il; }

static void moe_dump(Device & d, const char * name, const Tensor & t) {
    const char * dir = getenv("AI_DUMP_DIR");
    if (!dir || g_dump_layer < 0 || recording() || !t.valid()) {
        return;
    }
    const int64_t cols = t.ne[0], rows = t.numel() / t.ne[0];
    const size_t  esz  = 4;   // F32 and I32 alike
    std::vector<char> host((size_t) t.numel() * esz);
    d.download(t, host.data(), 0, host.size());
    std::string path = std::string(dir) + "/" + name + "-" + std::to_string(g_dump_layer) + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return;
    }
    int64_t hdr[2] = { cols, rows };
    fwrite(hdr, 8, 2, f);
    fwrite(host.data(), 1, host.size(), f);
    fclose(f);
}

using adapter::Call;

// Fusion-alignment switches. The first three are bit-neutral against the llama.cpp graph.
// AI_TOPK_ORDER lets the backend fuse top-k routing into one kernel (about +8% decode). The
// llama.cpp graph never takes that path, so it changes bits, by about as much as llama.cpp's own
// fused-vs-split beta|alpha difference. Parity runs set AI_TOPK_ORDER=0, which is bitwise identical.
static bool knob(const char * name, bool def) {
    const char * e = getenv(name);
    return e ? atoi(e) != 0 : def;
}
// AI_BOUND_DELETE: skip whole kernel classes to bound their cost (the ggml CUDA backend handles its own
// classes). A deleted class leaves its output stale, so this is a timing instrument, never a generation
// mode. Classes here: gdnconv, gdngate, merge, rmsapply, router.
static bool bound_delete(const char * cls) {
    static const std::string v = []() {
        const char * e = getenv("AI_BOUND_DELETE");
        return e ? std::string(",") + e + "," : std::string();
    }();
    return !v.empty() && v.find(std::string(",") + cls + ",") != std::string::npos;
}
static const bool k_rope_view   = knob("AI_ROPE_VIEW", true);
static const bool k_conv_view   = knob("AI_CONV_VIEW", true);
static const bool k_gdn_inplace = knob("AI_GDN_INPLACE", true);
// AI_PARITY=1 turns every bit-changing fusion off by default (bitwise equal to the llama.cpp graph).
static const bool k_parity      = knob("AI_PARITY", false);
static const bool k_topk_order  = knob("AI_TOPK_ORDER", !k_parity);
// AI_SMALL_GEMV (opt-in, slower): the F16 router|score and GDN beta|alpha GEMVs through kernels/gemv_f16w.cu
// (warp per row, f32 accumulate) instead of ggml's mul_mat_vec_f. 257 rows give only 33 blocks, too little
// parallelism for a latency-bound kernel; ggml's block-per-row has more. Changes numerics.
static const bool k_small_gemv  = knob("AI_SMALL_GEMV", false);
// Second-stream fork/join, recorded in sequence (same mechanism as AI_MOE_CONCURRENT): the switch happens inside
// native lambdas, so it applies to exactly the Calls between side_begin() and side_end().
struct SideFork {
    void * ev_join = nullptr;
    void * st_main = nullptr;
    void * st_side = nullptr;
};
static void * side_event() {
    static std::vector<void *> evs;
    static size_t              evi = 0;
    if (evs.size() < 512) {
        evs.push_back(kern::event_create());
        return evs.back();
    }
    return evs[(evi++) % evs.size()];
}
static SideFork side_begin(Device & d) {
    SideFork f;
    void *         ev_fork = side_event();
    f.ev_join              = side_event();
    f.st_main              = adapter::stream(d);
    f.st_side              = kern::backend_stream_no(d.backend(), 1);
    ggml_backend * bk      = d.backend();
    const SideFork ff      = f;
    adapter::native(d, [=]() {
        kern::event_record(ev_fork, ff.st_main);
        kern::stream_wait_event(ff.st_side, ev_fork);
        kern::backend_set_stream_no(bk, 1);
    });
    return f;
}
static void side_end(Device & d, const SideFork & f) {
    ggml_backend * bk = d.backend();
    adapter::native(d, [=]() {
        kern::backend_set_stream_no(bk, 0);
        kern::event_record(f.ev_join, f.st_side);
    });
}
static void side_join(Device & d, const SideFork & f) {
    adapter::native(d, [=]() { kern::stream_wait_event(f.st_main, f.ev_join); });
}

// Returns an invalid Tensor when not applicable; the caller keeps the ggml GEMV.
static Tensor small_gemv_f16(Device & d, const Tensor & w, const Tensor & x) {
    // AI_GEMM_F16_PF, prefill width (T > 8): native fp16 tensor-core GEMM with fp32 accumulation instead of ggml's
    // convert + cuBLAS half-accumulate + split-K + convert-back chain (kernels/gemm_f16w_pf.cu). Opt-in.
    if (x.ne[1] > 8 && kern::gemm_f16w_pf_enabled() && w.type == F16 && x.type == F32 && w.contiguous() &&
        x.contiguous() && w.ne[0] == x.ne[0] && w.ne[0] % 32 == 0 && ((uintptr_t) w.data % 16) == 0 &&
        ((uintptr_t) x.data % 16) == 0) {
        const int64_t K = w.ne[0], N = w.ne[1], T = x.ne[1];
        Tensor        y  = d.ws_alloc(F32, N, T);
        void *        st = adapter::stream(d);
        const void *  pw = w.data;
        const float * px = (const float *) x.data;
        float *       py = (float *) y.data;
        adapter::native(d, [=]() {
            if (!kern::gemm_f16w_pf(st, pw, px, py, K, N, T, K, N)) {
                fprintf(stderr, "[gemm-f16-pf] declined a shape the host check accepted\n");
                abort();
            }
        });
        return y;
    }
    if (!k_small_gemv || w.type != F16 || x.type != F32 || !w.contiguous() || !x.contiguous() || x.ne[1] > 8 ||
        w.ne[0] != x.ne[0] || w.ne[0] % 8 != 0) {
        return Tensor{};
    }
    const int64_t K = w.ne[0], N = w.ne[1], T = x.ne[1];
    Tensor        y  = d.ws_alloc(F32, N, T);
    void *        st = adapter::stream(d);
    const void *  pw = w.data;
    const float * px = (const float *) x.data;
    float *       py = (float *) y.data;
    if (((uintptr_t) pw % 16) != 0 || ((uintptr_t) px % 16) != 0) {
        return Tensor{};
    }
    adapter::native(d, [=]() { kern::gemv_f16w(st, pw, px, py, K, N, T, K, N); });
    return y;
}
// AI_BOUND_SKIP_BA: skip the GDN beta|alpha GEMV to bound its cost (timing only, output stale). File
// scope because the GEMV has two sites, gdn_block (unfused) and gdn_block_fused (AI_FUSE_GDN, default on), and both
// must be covered. Skipping it also frees its bytes for other kernels, so the measured delta is an upper
// bound on the GEMV's own cost.
static const bool k_bound_skip_ba = getenv("AI_BOUND_SKIP_BA") != nullptr;

Round::Round(Device & d) : rec_(std::make_unique<adapter::Recorder>(d)) {}
Round::~Round() = default;
void Round::begin() { rec_->begin(); }
void Round::end() { rec_->end(); }
void Round::launch() { rec_->launch(); }
int  Round::n_nodes() const { return rec_->n_nodes(); }
bool recording() { return adapter::Recorder::active() != nullptr; }
void host_step(Device & d, std::function<void()> fn) { adapter::host(d, std::move(fn)); }

Tensor router_topk(Device & d, const Tensor & x, const Tensor & norm, const Tensor & router, float eps, int P) {
    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor * h = ggml_mul(g, ggml_rms_norm(g, c.in(x, "pred_in"), eps), c.in(norm, "pred_norm"));
    ggml_tensor * logits = ggml_mul_mat(g, c.in(router, "pred_router"), h);
    return c.run(ggml_cont(g, ggml_argsort_top_k(g, logits, P)));
}

void l2_prefetch(Device & d, const Tensor & up, const Tensor & gate, const Tensor & down, const Tensor & slots) {
    void *          st = adapter::stream(d);
    const void *    pu = up.data, * pg = gate.data, * pd = down.data;
    const int64_t   su = (int64_t) up.nb[2], sg = (int64_t) gate.nb[2], sd = (int64_t) down.nb[2];
    const int32_t * ps = (const int32_t *) slots.data;
    const int64_t   n  = slots.numel();
    adapter::native(d, [=]() { kern::l2_prefetch_experts(st, pu, pg, pd, su, sg, sd, ps, n); });
}

// Read side of the capture contract, for a serving layer that must refuse or surface a degraded
// engine rather than keep answering at reduced speed.
bool capture_degraded() {
    return adapter::capture_degraded();
}

Capture capture_mode() {
    switch (adapter::capture_mode()) {
        case adapter::CaptureMode::Own:     return Capture::Own;
        case adapter::CaptureMode::Backend: return Capture::Backend;
        case adapter::CaptureMode::None:    return Capture::None;
    }
    return Capture::None;
}

void set_capture(Capture c) {
    adapter::set_capture_mode(c == Capture::Own ? adapter::CaptureMode::Own
                              : c == Capture::Backend ? adapter::CaptureMode::Backend
                                                      : adapter::CaptureMode::None);
}

// ggml's CUDA get_rows has no Q4_K path: gather a Q4_K table natively (kernels/embed_q4k.cu).
static Tensor embed_native_q4k(Device & d, const Tensor & table, const Tensor & ids) {
    const int64_t   E   = table.ne[0];
    const int64_t   T   = ids.numel();
    Tensor          out = d.ws_alloc(F32, E, T);
    void *          st  = adapter::stream(d);
    const void *    pt  = table.data;
    const int32_t * pi  = (const int32_t *) ids.data;
    float *         po  = (float *) out.data;
    const int64_t   rb  = (int64_t) table.nb[1];
    adapter::native(d, [=]() {
        if (!kern::embed_q4k(st, pt, E, rb, pi, T, po)) {
            fprintf(stderr, "[embed-q4k] declined a shape the host check accepted\n");
            abort();
        }
    });
    return out;
}

Tensor embed(Device & d, const Tensor & table, const Tensor & ids) {
    if (table.type == Q4_K && ids.type == I32 && ids.contiguous()) {
        return embed_native_q4k(d, table, ids);
    }
    Call c(d);
    return c.run(ggml_get_rows(c.ctx(), c.in(table, "tok_embd"), c.in(ids, "ids")));
}

Tensor rmsnorm(Device & d, const Tensor & x, const Tensor & w, float eps) {
    Call c(d);
    ggml_tensor * y = ggml_rms_norm(c.ctx(), c.in(x), eps);
    return c.run(ggml_mul(c.ctx(), y, c.in(w)));
}

AddNorm add_rmsnorm(Device & d, const Tensor & mix, const Tensor & res, const Tensor & w, float eps,
                    const Tensor * out_buf, const Tensor * q8_buf, const Tensor * router_w) {
    const int64_t ncols = mix.ne[0];
    const int64_t rows  = mix.numel() / ncols;
    AddNorm       o;
    o.sum = d.ws_alloc(F32, ncols, rows);
    // alloc_stage sizes this buffer for the widest round (rows = n_streams), and it is handed downstream
    // as a typed tensor, so narrow it to this round's rows (otherwise the carry copy in
    // Qwen35Hybrid::forward fails ggml_cpy's nelements assert with --parallel > 1). The view is at
    // offset 0, so .data and the q8 staging registry key are unchanged.
    if (out_buf != nullptr && out_buf->ne[0] == ncols && out_buf->numel() >= ncols * rows) {
        o.out = out_buf->view(ncols, rows, 1, 1, out_buf->nb[1],
                              out_buf->nb[1] * (size_t) rows, out_buf->nb[1] * (size_t) rows, 0);
    } else {
        o.out = d.ws_alloc(F32, ncols, rows);
    }
    void *        st = adapter::stream(d);
    const float * pm = (const float *) mix.data;
    const float * pr = (const float *) res.data;
    const float * pw = (const float *) w.data;
    float *       ps = (float *) o.sum.data;
    float *       po = (float *) o.out.data;

    // AI_Q8_STAGE=1: also emit the q8_1 copy of the normed output, so the consumers that would each
    // launch their own quantise (the router|score + shexp gate|up span, and the expert gate/up
    // MUL_MAT_ID) take it pre-quantised. One staging serves two of the per-layer quantise calls, so
    // expect consumed ~= 2x staged.
    // 1 = emit and register; 2 = emit but do not register, which isolates whether a divergence comes
    // from the emission itself or from the consumer using it.
    static const int k_stage = []() {
        // default: follow the registry, so an engine that turned staging on gets both producers and one
        // that did not (offload) gets neither and no warning
        const char * e = getenv("AI_Q8_STAGE");
        return e ? atoi(e) : (kern::q8_stage_enabled() ? 1 : 0);
    }();
    void * pq = nullptr;
    // Two switches gate this fold: AI_Q8_STAGE turns on the producer side here, ALPHA_Q8_STAGE turns on
    // the ggml registry that the consuming GEMV looks in. With only the first set the fold silently does
    // nothing, so warn once.
    if (k_stage != 0 && !kern::q8_stage_enabled()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr,
                    "[q8-stage] AI_Q8_STAGE=%d but the backend registry is OFF: set ALPHA_Q8_STAGE=1 "
                    "too, or this fold does nothing at all.\n",
                    k_stage);
        }
    }
    if (k_stage != 0 && ncols % 512 == 0 && kern::q8_stage_enabled()) {
        // Only from a caller-provided per-layer buffer: an arena allocation shares its address with
        // another layer, and the registry (keyed on that address, with identical geometry across
        // layers) would then serve the wrong layer's activation.
        const int64_t bytes = ncols / 32 * rows * 36;
        if (q8_buf != nullptr && q8_buf->numel() >= bytes) {
            o.q8 = *q8_buf;
            pq   = o.q8.data;
        }
    }
    // AI_BOUND_DELETE=norm also covers this native kernel, which the ggml-side switch cannot see.
    // Output left stale.
    static const bool k_bd_norm = []() {
        const char * e = getenv("AI_BOUND_DELETE");
        // ",addnorm," prices this kernel alone (the ggml norms stay); ",norm," deletes it with all of them
        const std::string v = e ? std::string(",") + e + "," : std::string();
        return e && (v.find(",norm,") != std::string::npos || v.find(",addnorm,") != std::string::npos);
    }();
    static long long k_bd_norm_n = 0;
    if (k_bd_norm && k_bd_norm_n++ == 0) {
        atexit([]() { fprintf(stderr, "[ai-bound-delete] native add_rmsnorm deleted at record time: %lld\n", k_bd_norm_n); });
    }
    // AI_ADDNORM_SPLIT=1: at decode widths, a full-grid add + partials then the quantise-shaped apply,
    // instead of one block per row. Bit-exact by the same lane construction as AI_FUSE_MERGENORM=5.
    // Prefill keeps add_rmsnorm, which is already parallel over its rows.
    // =2 (default): one full-grid kernel in which every block redundantly reduces the whole row, so there
    // is no second launch; about half the time of add_rmsnorm, bit-identical output. 0 = add_rmsnorm.
    static const int k_split = getenv("AI_ADDNORM_SPLIT") ? atoi(getenv("AI_ADDNORM_SPLIT")) : 2;
    float *          ppart   = nullptr;
    if (k_split == 1 && rows <= 64) {
        ppart = (float *) d.ws_alloc(F32, 32 * rows).data;
    }
    const bool redundant = k_split == 2 && rows <= 64;
    // AI_ADDNORM_ROUTER: 0 off; 1 the router GEMV computed in this launch (kernels/moe_merge_partials.cu);
    // 2 compare: the normal kernels run and are consumed, the fused one writes shadows that are compared
    // bitwise (r and p here, the logits against the ggml router in moe_residual_fused).
    // Default 1; the compare mode shows the fused results bit-identical.
    static const int k_ar = getenv("AI_ADDNORM_ROUTER") ? atoi(getenv("AI_ADDNORM_ROUTER")) : 1;
    const bool ar = k_ar != 0 && redundant && !k_bd_norm && router_w != nullptr && router_w->valid() &&
                    router_w->type == F16 && router_w->ne[0] == ncols && router_w->contiguous() && rows <= 8;
    float *       plog = nullptr;
    float *       sh_r = nullptr, *sh_p = nullptr;
    const void *  prw  = ar ? router_w->data : nullptr;
    const int64_t nrw  = ar ? router_w->ne[1] : 0;
    if (ar) {
        o.router     = d.ws_alloc(F32, nrw, rows);
        o.router_cmp = k_ar == 2;
        plog         = (float *) o.router.data;
        if (k_ar == 2) {
            sh_r = (float *) d.ws_alloc(F32, ncols, rows).data;
            sh_p = (float *) d.ws_alloc(F32, ncols, rows).data;
        }
    }
    adapter::native(d, [=]() {
        if (k_bd_norm) {
        } else if (ar && k_ar == 1 && kern::addnorm_router(st, pm, pr, pw, ps, po, pq, prw, nrw, plog, ncols, rows, eps)) {
        } else if (redundant && kern::add_rmsnorm_redundant(st, pm, pr, pw, ps, po, pq, ncols, rows, eps)) {
            if (ar && k_ar == 2 && kern::addnorm_router(st, pm, pr, pw, sh_r, sh_p, nullptr, prw, nrw, plog, ncols, rows, eps)) {
                kern::addnorm_router_compare(st, sh_r, ps, ncols * rows);
                kern::addnorm_router_compare(st, sh_p, po, ncols * rows);
            }
        } else if (ppart != nullptr && kern::add_rmsnorm_split(st, pm, pr, pw, ps, po, ppart, pq, ncols, rows, eps)) {
        } else {
            kern::add_rmsnorm(st, pm, pr, pw, ps, po, pq, ncols, rows, eps);
        }
        if (pq != nullptr && k_stage == 4) {
            kern::q8_stage_fill_reference(st, po, pq, ncols, rows);
        }
        if (pq != nullptr && k_stage == 3) {
            kern::q8_stage_verify(st, po, pq, ncols, rows);
        }
        if (pq != nullptr && (k_stage == 1 || k_stage == 3 || k_stage == 4)) {
            // ne11/ne12/ne13 = 1 at decode; s11 is the f32 row stride the copy was made with
            kern::q8_stage_put(po, pq, ncols, rows, 1, 1, ncols, ncols * rows, ncols * rows);
        }
    });
    return o;
}

Tensor add(Device & d, const Tensor & a, const Tensor & b) {
    Call c(d);
    return c.run(ggml_add(c.ctx(), c.in(a), c.in(b)));
}

Tensor rows(Device & d, const Tensor & x, const Tensor & ids) {
    Call c(d);
    return c.run(ggml_get_rows(c.ctx(), c.in(x), c.in(ids)));
}

static void ws_past(Device & d, const Tensor & x);   // below, with argmax

Tensor max_softmax(Device & d, const Tensor & logits) {
    const int64_t ncols = logits.ne[0];
    const int64_t rows  = logits.numel() / ncols;
    if (logits.type != F32 || !logits.contiguous()) {
        throw std::runtime_error("max_softmax: f32 contiguous logits only");
    }
    ws_past(d, logits);
    Tensor        dst = d.ws_alloc(F32, rows);
    const int64_t nsc = kern::max_softmax_scratch_floats(rows);
    Tensor        sc  = nsc > 0 ? d.ws_alloc(F32, nsc) : Tensor{};
    void *        st  = adapter::stream(d);
    const float * px  = (const float *) logits.data;
    float *       pd  = (float *) dst.data;
    float *       ps  = nsc > 0 ? (float *) sc.data : nullptr;
    adapter::native(d, [=]() { kern::max_softmax_rows(st, px, ncols, rows, pd, ps); });
    return dst;
}

// [stray-token bug, 2026-11-01] scratch allocated after a RoundCache replay can land INSIDE the replayed round's output
// (rounds record from workspace origin 0; a replay leaves the caller's offset where it was), and argmax_partial_k then
// overwrote logits other blocks were still reading. Before allocating scratch next to an f32 input that lives in the
// workspace, move the offset past that input's end (forward only: what the caller allocated stays valid). Q4xServe's
// argmax_cols does the same (c924e5b); this covers every caller, the CLI's greedy path included.
static void ws_past(Device & d, const Tensor & x) {
    const char * wb = (const char *) d.ws_base();
    if (!wb || !x.valid() || x.type != F32) return;
    const char * p0 = (const char *) x.data;
    const size_t n  = (size_t) x.numel() * sizeof(float);
    if (p0 < wb || p0 + n > wb + d.ws_capacity()) return;
    const size_t e = (size_t) (p0 - wb) + n;
    if (d.ws_mark() < e && e + ((size_t) 1 << 20) <= d.ws_capacity()) d.ws_reset(e);
}

Tensor argmax(Device & d, const Tensor & logits, const Tensor * map) {
    ws_past(d, logits);
    // ggml's argmax is one block per row, so a 248k-column decode row reduces on a single block.
    // The native path splits the row and is bit-exact, ties included.
    static const bool k_native_argmax = []() {
        const char * e = getenv("AI_NATIVE_ARGMAX");
        return e ? atoi(e) != 0 : true;
    }();
    const int64_t ncols = logits.ne[0];
    const int64_t rows  = logits.numel() / ncols;
    if (k_native_argmax && logits.type == F32 && logits.contiguous()) {
        const int     np   = kern::argmax_partials_per_row();
        Tensor        pmax = d.ws_alloc(F32, np * rows);
        Tensor        pidx = d.ws_alloc(I32, np * rows);
        Tensor        dst  = d.ws_alloc(I32, rows);
        void *        st   = adapter::stream(d);
        const float * px   = (const float *) logits.data;
        float *       pm   = (float *) pmax.data;
        int32_t *     pi   = (int32_t *) pidx.data;
        int32_t *     pd   = (int32_t *) dst.data;
        const int32_t * pmap = (map != nullptr && map->valid()) ? (const int32_t *) map->data : nullptr;
        adapter::native(d, [=]() { kern::argmax_rows(st, px, ncols, rows, pm, pi, pd, pmap); });
        return dst;
    }
    Call c(d);
    return c.run(ggml_argmax(c.ctx(), c.in(logits)));
}

void sample_into(Device & d, const Tensor & logits, const Tensor & token, int64_t L, int first_region,
                 const Tensor * base, int64_t pos_const, const void * params) {
    if (params == nullptr) {
        return;
    }
    if (logits.type != F32 || !logits.contiguous() || token.type != I32) {
        throw std::runtime_error("sample_into: f32 contiguous logits and i32 tokens only");
    }
    const int64_t   ncols = logits.ne[0];
    const int64_t   rows  = logits.numel() / ncols;
    ws_past(d, logits);
    Tensor          sc    = d.ws_alloc(F32, kern::sample_scratch_floats(rows));
    void *          st    = adapter::stream(d);
    const float *   px    = (const float *) logits.data;
    float *         ps    = (float *) sc.data;
    int32_t *       pd    = (int32_t *) token.data;
    const int32_t * pb    = (base != nullptr && base->valid()) ? (const int32_t *) base->data : nullptr;
    const auto *    pp    = (const kern::SampleParams *) params;
    adapter::native(d, [=]() { kern::sample_rows(st, px, ncols, rows, L, first_region, pb, pos_const, pp, ps, pd); });
}


void copy_into(Device & d, const Tensor & src, const Tensor & dst, const char * who) {
    // ggml_cpy asserts nelements(src) == nelements(dst) and aborts with only a backtrace, which
    // names the function but not the shapes. Report them first: a shape bug is diagnosable from
    // the numbers and near-undiagnosable without them.
    const int64_t ns = src.ne[0] * src.ne[1] * src.ne[2] * src.ne[3];
    const int64_t nd = dst.ne[0] * dst.ne[1] * dst.ne[2] * dst.ne[3];
    if (ns != nd) {
        fprintf(stderr,
                "[copy_into] SHAPE MISMATCH src [%lld,%lld,%lld,%lld]=%lld  dst [%lld,%lld,%lld,%lld]=%lld\n",
                (long long) src.ne[0], (long long) src.ne[1], (long long) src.ne[2], (long long) src.ne[3],
                (long long) ns,
                (long long) dst.ne[0], (long long) dst.ne[1], (long long) dst.ne[2], (long long) dst.ne[3],
                (long long) nd);
    }
    Call c(d);
    c.effect(ggml_cpy(c.ctx(), c.in(src), c.in(dst)));
    c.run_effects_only();
}

Tensor linear(Device & d, const Tensor & w, const Tensor & x) {
    Call c(d);
    return c.run(ggml_mul_mat(c.ctx(), c.in(w, "w"), c.in(x, "x")));
}

Tensor linear_id(Device & d, const Tensor & w, const Tensor & x, const Tensor & ids) {
    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor * cur = ggml_reshape_3d(g, c.in(x, "mmid_in"), x.ne[0], 1, x.ne[1]);
    return c.run(ggml_mul_mat_id(g, c.in(w, "mmid_w"), cur, c.in(ids, "mmid_ids")));
}

Tensor linear_q8_native(Device & d, const Tensor & w, const Tensor & x, int rows_per_cta) {
    if (w.type != Q8_0 || x.type != F32 || x.ne[1] != 1 || x.ne[0] != w.ne[0] || x.ne[0] % 32 != 0 ||
        !x.contiguous()) {
        throw std::runtime_error("linear_q8_native: needs Q8_0 weights and one contiguous f32 token");
    }
    const int64_t cols = w.ne[0], rows = w.ne[1], nb = cols / 32;
    Tensor q  = d.ws_alloc(I8, cols);
    Tensor dx = d.ws_alloc(F32, nb);
    Tensor y  = d.ws_alloc(F32, rows);
    void * st = adapter::stream(d);
    const float *   px  = (const float *) x.data;
    int8_t *        pq  = (int8_t *) q.data;
    float *         pd  = (float *) dx.data;
    float *         py  = (float *) y.data;
    const uint8_t * pw  = (const uint8_t *) w.data;
    adapter::native(d, [=]() {
        kern::quantize_act_q8(st, px, pq, pd, nb);
        kern::gemv_q8_0(st, pw, rows, cols, pq, pd, py, rows_per_cta);
    });
    return y;
}

Tensor linear_q8_planar(Device & d, const Tensor & w, const Tensor & x, int mode) {
    const int64_t T = x.ne[1];
    const bool    f16 = x.type == F16;   // fp16 activations, split-K only
    if (w.type != Q8_0 || (x.type != F32 && !f16) || (f16 && mode > 0) || T < 1 || T > 8 || x.ne[2] != 1 ||
        x.ne[3] != 1 || x.ne[0] != w.ne[0] || !x.contiguous() || !kern::gemv_q8_planar_ok(w.ne[0])) {
        throw std::runtime_error(
            "linear_q8_planar: needs planar Q8_0, cols % 256 == 0, 1..8 contiguous f32 (or f16 with split-K) columns");
    }
    const int64_t cols = w.ne[0], rows = w.ne[1];
    Tensor y = d.ws_alloc(F32, rows, T);
    void * st = adapter::stream(d);
    const void *    px = x.data;
    float *         py = (float *) y.data;
    const uint8_t * pw = (const uint8_t *) w.data;
    if (f16) {
        adapter::native(d, [=]() { kern::gemv_q8_planar_f16x(st, pw, rows, cols, T, px, py, -mode); });
    } else {
        adapter::native(d, [=]() { kern::gemv_q8_planar(st, pw, rows, cols, T, (const float *) px, py, mode); });
    }
    return y;
}

Tensor cast_f16(Device & d, const Tensor & x) {
    if (x.type != F32 || !x.contiguous() || x.numel() % 4 != 0) {
        throw std::runtime_error("cast_f16: needs a contiguous f32 tensor with numel % 4 == 0");
    }
    Tensor y = d.ws_alloc(F16, x.ne[0], x.ne[1], x.ne[2], x.ne[3]);
    void * st = adapter::stream(d);
    const float * px = (const float *) x.data;
    void *        py = y.data;
    const int64_t n  = x.numel();
    adapter::native(d, [=]() { kern::cast_f32_f16(st, px, py, n); });
    return y;
}

Tensor linear_q8_gguf_f16x(Device & d, const Tensor & w, const Tensor & x16, int split) {
    const int64_t T = x16.ne[1];
    if (w.type != Q8_0 || x16.type != F16 || T < 1 || T > 8 || x16.ne[2] != 1 || x16.ne[3] != 1 ||
        x16.ne[0] != w.ne[0] || !x16.contiguous() || !kern::gemv_q8_planar_split_ok(w.ne[0], split)) {
        throw std::runtime_error("linear_q8_gguf_f16x: Q8_0 (GGUF layout), 1..8 contiguous f16 columns, cols % (split*256) == 0");
    }
    const int64_t cols = w.ne[0], rows = w.ne[1];
    Tensor y = d.ws_alloc(F32, rows, T);
    void * st = adapter::stream(d);
    const void *    px = x16.data;
    float *         py = (float *) y.data;
    const uint8_t * pw = (const uint8_t *) w.data;
    adapter::native(d, [=]() { kern::gemv_q8_gguf_f16x(st, pw, rows, cols, T, px, py, split); });
    return y;
}

// fp16-activation projections at the norm-fed call sites.
// AI_V2_F16X=1 routes the three norm-fed Q8_0 spans (GDN qkv|z, attention q|k|v, shared-expert gate|up)
// through a cast to fp16 plus the no-activation-quant GEMV, instead of ggml's quantize_q8_1 + MMVQ.
// Only at T <= 8 (decode, MTP verify); prefill keeps ggml's path on the same weight bytes.
// The cast is a standalone launch, so the launch count matches the quantize it replaces.
// Routed and declined counts per site are reported at exit, so a partially live flag is visible.
namespace {
struct V2Count {
    const char * site;
    long         routed = 0, declined = 0;
};
V2Count g_v2_sites[3] = { { "gdn qkv|z" }, { "attn q|k|v" }, { "shexp gate|up" } };
bool v2_enabled() {
    static const bool on = []() {
        const char * e = getenv("AI_V2_F16X");
        const bool   v = e && atoi(e) != 0;
        if (v) {
            atexit([]() {
                fprintf(stderr, "[v2-f16x] routed/declined at record time:");
                for (const V2Count & s : g_v2_sites) {
                    fprintf(stderr, " %s %ld/%ld;", s.site, s.routed, s.declined);
                }
                fprintf(stderr, "\n");
            });
        }
        return v;
    }();
    return on;
}
// Returns the projection [rows, T] f32, or an invalid Tensor when this site must keep ggml's path.
Tensor v2_proj(Device & d, const Tensor & w, const Tensor & x, int site) {
    if (!v2_enabled() || !w.valid()) {
        return Tensor{};
    }
    const int64_t T = x.ne[1];
    const bool ok = w.type == Q8_0 && x.type == F32 && x.contiguous() && x.ne[2] == 1 && x.ne[3] == 1 &&
                    T >= 1 && T <= 8 && x.ne[0] == w.ne[0];
    const int split = (w.ne[0] % 1024 == 0) ? 4 : 1;
    if (!ok || !kern::gemv_q8_planar_split_ok(w.ne[0], split)) {
        ++g_v2_sites[site].declined;
        return Tensor{};
    }
    ++g_v2_sites[site].routed;
    return linear_q8_gguf_f16x(d, w, cast_f16(d, x), split);
}
}  // namespace

Tensor head(Device & d, const Tensor & x, const Tensor & norm_w, const Tensor & out_w, float eps) {
    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor * h = ggml_mul(g, ggml_rms_norm(g, c.in(x), eps), c.in(norm_w));
    // named "output.weight" so ggml's LM-head-only switches (ALPHA_OUTPUT_MMQ) can recognise it
    return c.run(ggml_mul_mat(g, c.in(out_w, "output.weight"), h));
}

Tensor attn_block(Device & d, const Tensor & x, const AttnWeights & w, const AttnParams & p,
                  const KvLayer & kv, const AttnInputs & in) {
    Tensor v2qkv = v2_proj(d, w.qkv, x, 1);   // before the Call, so it is recorded first
    // AI_ATTN_PRO: q/k norms + RoPE + K/V cache writes + gate copy as one native launch (kernels/attn_prologue.cu).
    // 0 off; 1 on (default, f16 KV only); 2 compare (the ggml chain runs and is consumed; the kernel writes shadows,
    // compared bitwise). The compare mode shows the native results bit-identical.
    static const int  k_ap = getenv("AI_ATTN_PRO") ? atoi(getenv("AI_ATTN_PRO")) : 1;
    static const bool k_fa_native_env = getenv("AI_FA_NATIVE") && atoi(getenv("AI_FA_NATIVE")) != 0;
    const int64_t     Ta  = x.ne[1];
    const bool ap = k_ap != 0 && !k_fa_native_env && (w.qkv.valid() || (w.q.valid() && w.k.valid() && w.v.valid())) &&
                    kv.k.type == F16 && kv.v.type == F16 &&
                    kern::attn_prologue_ok(p.head_dim, p.n_rot) && w.q_norm.valid() && w.k_norm.valid() &&
                    w.q_norm.type == F32 && w.k_norm.type == F32 && w.q_norm.numel() == p.head_dim &&
                    w.k_norm.numel() == p.head_dim && in.write_cells.type == I64 && in.pos4.type == I32 &&
                    in.pos4.numel() >= 4 * Ta && (p.rope_mode & GGML_ROPE_TYPE_MROPE) != 0 &&
                    p.rope_mode != GGML_ROPE_TYPE_VISION && x.type == F32 && x.contiguous() &&
                    (size_t) kv.k.nb[1] == (size_t) p.head_dim * p.n_head_kv * 2 &&
                    (size_t) kv.v.nb[1] == (size_t) p.head_dim * p.n_head_kv * 2;
    Tensor ap_q, ap_g, ap_k, ap_v;
    if (ap) {
        // q|gate, k, v: one fused projection (35B) or three separate ones (27B, whose q/k/v types differ)
        Tensor sep_k, sep_v;
        if (!v2qkv.valid()) {
            Call ca(d);
            ggml_tensor * gxa = ca.in(x, "attn_in");
            if (w.qkv.valid()) {
                v2qkv = ca.run(ggml_mul_mat(ca.ctx(), ca.in(w.qkv, "wqkv_fused"), gxa));
            } else {
                ggml_tensor * kf = ggml_mul_mat(ca.ctx(), ca.in(w.k, "wk"), gxa);
                ggml_tensor * vf = ggml_mul_mat(ca.ctx(), ca.in(w.v, "wv"), gxa);
                ca.effect(kf);
                ca.effect(vf);
                v2qkv = ca.run(ggml_mul_mat(ca.ctx(), ca.in(w.q, "wq"), gxa));
                sep_k = ca.to_public(kf);
                sep_v = ca.to_public(vf);
            }
        }
        const int64_t D = p.head_dim, Hq = p.n_head, Hkv = p.n_head_kv;
        ap_q = d.ws_alloc(F32, D, Hq, Ta);
        ap_g = d.ws_alloc(F32, D * Hq, Ta);
        if (k_ap == 2) {
            ap_k = d.ws_alloc(F16, D * Hkv, Ta);
            ap_v = d.ws_alloc(F16, D * Hkv, Ta);
        }
        void *          st    = adapter::stream(d);
        const float *   pfo   = (const float *) v2qkv.data;
        const int64_t   fo_rw = (int64_t) v2qkv.nb[1] / 4;
        const bool      sep   = sep_k.valid();
        const float *   pk3   = sep ? (const float *) sep_k.data : pfo + 2 * D * Hq;
        const int64_t   k3_rw = sep ? (int64_t) sep_k.nb[1] / 4 : fo_rw;
        const float *   pv3   = sep ? (const float *) sep_v.data : pfo + 2 * D * Hq + D * Hkv;
        const int64_t   v3_rw = sep ? (int64_t) sep_v.nb[1] / 4 : fo_rw;
        const float *   pqw   = (const float *) w.q_norm.data, *pkw = (const float *) w.k_norm.data;
        const int32_t * ppos  = (const int32_t *) in.pos4.data;
        const int64_t * pcell = (const int64_t *) in.write_cells.data;
        const bool      shadow = k_ap == 2;
        void *          pkc   = shadow ? ap_k.data : kv.k.data;
        void *          pvc   = shadow ? ap_v.data : kv.v.data;
        const int64_t   kcr   = shadow ? D * Hkv : (int64_t) kv.k.nb[1] / 2;
        const int64_t   vcr   = shadow ? D * Hkv : (int64_t) kv.v.nb[1] / 2;
        float *         pq    = (float *) ap_q.data, *pg = (float *) ap_g.data;
        const float     eps = p.eps, fb = p.rope_base, fs = p.rope_scale;
        const int       nrot = p.n_rot;
        const int       sec[4] = { p.sections[0], p.sections[1], p.sections[2], p.sections[3] };
        const bool      imrope = p.rope_mode == GGML_ROPE_TYPE_IMROPE;
        adapter::native(d, [=]() {
            if (!kern::attn_prologue3(st, pfo, fo_rw, pk3, k3_rw, pv3, v3_rw, Hq, Hkv, Ta, pqw, pkw, eps, ppos, nrot, sec,
                                      imrope, fb, fs, pcell, pkc, kcr, pvc, vcr, pq, pg, shadow)) {
                fprintf(stderr, "[attn-pro] declined a shape the host check accepted\n");
                abort();
            }
        });
        if (k_ap == 1) {
            const int64_t B = in.B, L = Ta / B;
            Call          c(d);
            ggml_context * g  = c.ctx();
            ggml_tensor *  kc = c.in(kv.k, "k_cache");
            ggml_tensor *  vc = c.in(kv.v, "v_cache");
            const int64_t region = in.region_cells > 0 ? in.region_cells : kv.k.ne[1];
            const size_t  rD     = ggml_row_size(kc->type, D);
            const size_t  rDH    = ggml_row_size(kc->type, D * Hkv);
            ggml_tensor * kview = ggml_view_4d(g, kc, D, Hkv, in.n_kv, B, rD, rDH, rDH * region, rDH * in.first_cell);
            ggml_tensor * vview = ggml_view_4d(g, vc, D, Hkv, in.n_kv, B, rD, rDH, rDH * region, rDH * in.first_cell);
            ggml_tensor * qp = ggml_permute(g, ggml_reshape_4d(g, c.in(ap_q, "attn_q_pro"), D, Hq, L, B), 0, 2, 1, 3);
            ggml_tensor * kp = ggml_permute(g, kview, 0, 2, 1, 3);
            ggml_tensor * vp = ggml_permute(g, vview, 0, 2, 1, 3);
            ggml_tensor * fa = ggml_flash_attn_ext(g, qp, kp, vp, c.in(in.mask, "kq_mask"), p.kq_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
            ggml_tensor * cur = ggml_reshape_2d(g, fa, fa->ne[0] * fa->ne[1], fa->ne[2] * fa->ne[3]);
            // AI_ATTN_OGATE: sigmoid gate + the o-projection's q8_1 copy as one native kernel (0 off, 1 on, 2 compare).
            // Bit-identical but no measurable gain, so off by default.
            static const int k_og = getenv("AI_ATTN_OGATE") ? atoi(getenv("AI_ATTN_OGATE")) : 0;
            const int64_t    nc   = D * Hq;
            if (k_og != 0 && kern::q8_stage_enabled() && nc % 512 == 0) {
                const Tensor  fat = c.run(cur);
                const Tensor  go  = d.ws_alloc(F32, nc, Ta);
                const Tensor  gq  = d.ws_alloc(F32, (nc / 32) * 9, Ta);   // q8_1: 36 bytes per 32 values
                void *        st2 = adapter::stream(d);
                const float * pa = (const float *) fat.data, *pgt = (const float *) ap_g.data;
                float *       pgo = (float *) go.data;
                void *        pgq = gq.data;
                const int64_t Tt  = Ta;
                if (k_og == 1) {
                    adapter::native(d, [=]() {
                        if (kern::attn_ogate_q8(st2, pa, pgt, pgo, pgq, nc, Tt)) {
                            kern::q8_stage_put(pgo, pgq, nc, Tt, 1, 1, nc, nc * Tt, nc * Tt);
                        }
                    });
                    Call c2(d);
                    return c2.run(ggml_mul_mat(c2.ctx(), c2.in(w.o, "wo"), c2.in(go, "attn_gated")));
                }
                adapter::native(d, [=]() { kern::attn_ogate_q8(st2, pa, pgt, pgo, nullptr, nc, Tt); });
                Call          c2(d);
                ggml_context * g2 = c2.ctx();
                ggml_tensor *  gd = ggml_mul(g2, c2.in(fat, "fa_out"), ggml_sigmoid(g2, c2.in(ap_g, "attn_gate_pro")));
                c2.effect(gd);
                Tensor         o2 = c2.run(ggml_mul_mat(g2, c2.in(w.o, "wo"), gd));
                const Tensor   gdt = c2.to_public(gd);
                const float *  pr  = (const float *) gdt.data;
                adapter::native(d, [=]() { kern::attn_ogate_compare(st2, pgo, pr, nc * Tt); });
                return o2;
            }
            cur = ggml_mul(g, cur, ggml_sigmoid(g, c.in(ap_g, "attn_gate_pro")));
            cur = ggml_mul_mat(g, c.in(w.o, "wo"), cur);
            return c.run(cur);
        }
    }
    Call c(d);
    ggml_context * g = c.ctx();

    const int64_t T   = x.ne[1];
    const int64_t B   = in.B;
    const int64_t L   = T / B;
    const int64_t D   = p.head_dim;
    const int64_t Hq  = p.n_head;
    const int64_t Hkv = p.n_head_kv;
    if (L * B != T) {
        throw std::runtime_error("attn_block: rows are not a multiple of the stream count");
    }

    ggml_tensor * gx = c.in(x, "attn_in");

    // joint query+gate projection, split per head: [q(D), gate(D)] x Hq
    ggml_tensor * q = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    if (w.qkv.valid()) {
        // one projection for q|gate, k and v; the splits are strided views of its rows
        ggml_tensor * fo  = v2qkv.valid() ? c.in(v2qkv, "v2_qkv")
                                          : ggml_mul_mat(g, c.in(w.qkv, "wqkv_fused"), gx);
        const size_t  es  = ggml_element_size(fo);
        const size_t  nb1 = fo->nb[1];
        const int64_t qd  = 2 * D * Hq;
        const int64_t kd  = D * Hkv;
        q    = ggml_view_3d(g, fo, D, Hq, T, es * D * 2, nb1, 0);
        gate = ggml_view_3d(g, fo, D, Hq, T, es * D * 2, nb1, es * D);
        k    = ggml_view_3d(g, fo, D, Hkv, T, es * D, nb1, es * qd);
        v    = ggml_view_3d(g, fo, D, Hkv, T, es * D, nb1, es * (qd + kd));
    } else {
        ggml_tensor * qf = ggml_mul_mat(g, c.in(w.q, "wq"), gx);
        const size_t es  = ggml_element_size(qf);
        q    = ggml_view_3d(g, qf, D, Hq, T, es * D * 2, es * D * 2 * Hq, 0);
        gate = ggml_view_3d(g, qf, D, Hq, T, es * D * 2, es * D * 2 * Hq, es * D);
        k    = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.k, "wk"), gx), D, Hkv, T);
        v    = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.v, "wv"), gx), D, Hkv, T);
    }

    q = ggml_mul(g, ggml_rms_norm(g, q, p.eps), c.in(w.q_norm));
    k = ggml_mul(g, ggml_rms_norm(g, k, p.eps), c.in(w.k_norm));

    gate = ggml_cont_2d(g, gate, D * Hq, T);

    int sections[4] = {p.sections[0], p.sections[1], p.sections[2], p.sections[3]};
    ggml_tensor * pos = c.in(in.pos4, "pos4");
    q = ggml_rope_multi(g, q, pos, nullptr, p.n_rot, sections, p.rope_mode, p.n_ctx_orig,
                        p.rope_base, p.rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_multi(g, k, pos, nullptr, p.n_rot, sections, p.rope_mode, p.n_ctx_orig,
                        p.rope_base, p.rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    if (ap && k_ap == 2) {   // publish the chain's q and gate for the compare
        c.effect(q);
        c.effect(gate);
    }

    // cache writes are launched before the attention read
    ggml_tensor * kc   = c.in(kv.k, "k_cache");
    ggml_tensor * vc   = c.in(kv.v, "v_cache");
    ggml_tensor * idx  = c.in(in.write_cells, "k_idxs");
    c.effect(ggml_set_rows(g, kc, k_rope_view ? ggml_view_2d(g, k, D * Hkv, T, k->nb[2], 0)
                                              : ggml_reshape_2d(g, k, D * Hkv, T), idx));
    c.effect(ggml_set_rows(g, vc, ggml_view_2d(g, v, D * Hkv, T, v->nb[2], 0), idx));

    // per-stream views: stream b reads region [first_cell + b*region, + n_kv)
    const int64_t region  = in.region_cells > 0 ? in.region_cells : kv.k.ne[1];
    const size_t  rD      = ggml_row_size(kc->type, D);
    const size_t  rDH     = ggml_row_size(kc->type, D * Hkv);

    // AI_FA_Q (default on for sm_8x): tiled split-K decode attention over a q8_0 / q4_0 cache (kernels/fa_decode_q.cu), same Call split
    // as AI_FA_NATIVE below. It reads this round's write cells and skips cells past each stream's last one.
    static const bool k_fa_q = kern::fa_decode_q_enabled();
    if (k_fa_q && kv.k.type == kv.v.type && (kv.k.type == Q8_0 || kv.k.type == Q4_0) && D == 256 && Hq % Hkv == 0 &&
        Hq / Hkv <= 8 && L >= 1 && L <= 8 && in.mask.type == F16 && in.n_kv > 0 && in.write_cells.type == I64 &&
        in.write_cells.numel() >= T && rDH == (size_t) kv.k.nb[1] && rDH == (size_t) kv.v.nb[1]) {
        c.late(gate);
        const Tensor  qt    = c.run(q);
        const Tensor  gt    = c.to_public(gate);
        // AI_FA_Q_REGION_SPLIT: size the split from the stream's KV region, not this round's envelope. n_kv is a
        // power-of-two envelope that an MTP verify round reaches a few positions before plain decode, so splitting by
        // n_kv reduces the same row over different split boundaries there, and exact MTP output differs from plain
        // decode at each crossing. Splits past the live length end the walk, so a constant split keeps a row's cell
        // order the same at every envelope. Set by the policy for exact and plain modes.
        static const bool fq_region_split = getenv("AI_FA_Q_REGION_SPLIT") && atoi(getenv("AI_FA_Q_REGION_SPLIT")) != 0;
        const int     split = kern::fa_decode_q_split(fq_region_split ? std::max(region, in.n_kv) : in.n_kv, Hkv, B);
        if (fq_region_split) {
            static bool told = false;
            if (!told) { told = true; fprintf(stderr, "[ai] AI_FA_Q split from the KV region (%lld cells): %d cells per split\n", (long long) region, split); }
        }
        const int64_t rows  = B * L * Hq;
        const Tensor  att   = d.ws_alloc(F32, D * Hq, T);
        const Tensor  wsb   = d.ws_alloc(F32, kern::fa_decode_q_workspace(in.n_kv, rows, split, Hkv, B));
        void *        st    = adapter::stream(d);
        const int     kt    = kv.k.type == Q8_0 ? 0 : 1;
        const float * pq    = (const float *) qt.data;
        const char *  pk    = (const char *) kv.k.data + rDH * (size_t) in.first_cell;
        const char *  pv    = (const char *) kv.v.data + rDH * (size_t) in.first_cell;
        const void *  pm    = in.mask.data;
        const int64_t * pc  = (const int64_t *) in.write_cells.data;
        float *       po    = (float *) att.data;
        float *       pw    = (float *) wsb.data;
        const int64_t n_kv = in.n_kv, cs = (int64_t) rDH, hs = (int64_t) rD, ss = (int64_t) (rDH * region);
        const int64_t mr = in.mask.nb[1] / 2, ms = in.mask.nb[3] / 2;
        const int64_t cb = in.first_cell, rg = region;
        const float   sc = p.kq_scale;
        static const bool k_chk = getenv("AI_FA_Q_CHECK") && atoi(getenv("AI_FA_Q_CHECK")) != 0;
        float *       pscr  = (k_chk && L > 1 && B == 1) ? (float *) d.ws_alloc(F32, D * Hq).data : nullptr;
        adapter::native(d, [=]() {
            if (!kern::fa_decode_q(st, kt, pq, pk, pv, pm, pc, po, pw, D, Hq, Hkv, L, B, n_kv, split, cs, hs, ss, mr,
                                   ms, cb, rg, sc)) {
                fprintf(stderr, "[fa-q] declined a shape the host check accepted\n");
                abort();
            }
            if (pscr != nullptr) {
                kern::fa_decode_q_selfcheck(st, kt, pq, pk, pv, pm, pc, po, pscr, pw, D, Hq, Hkv, L, B, n_kv, split,
                                            cs, hs, ss, mr, ms, cb, rg, sc);
            }
        });
        Call          c2(d);
        ggml_context * g2  = c2.ctx();
        ggml_tensor *  cur = ggml_mul(g2, c2.in(att, "fa_q"), ggml_sigmoid(g2, c2.in(gt, "attn_gate")));
        cur                = ggml_mul_mat(g2, c2.in(w.o, "wo"), cur);
        return c2.run(cur);
    }

    // AI_FA_NATIVE=1: native split-K decode attention (kernels/fa_decode.cu). The Call is split here: the
    // graph so far (projection, norms, rope, cache writes as effects) ends at the roped q, with the gate
    // as a late output; the attention runs natively; a second Call does the gate and the o-projection.
    static const bool k_fa_native = getenv("AI_FA_NATIVE") && atoi(getenv("AI_FA_NATIVE")) != 0;
    if (k_fa_native && kv.k.type == Q4_0 && kv.v.type == Q4_0 && D == 256 && Hq % Hkv == 0 &&
        (Hq / Hkv) * 32 <= 1024 && L >= 1 && L <= 8 && in.mask.type == F16 && in.n_kv > 0 &&
        rDH == (size_t) kv.k.nb[1] && rDH == (size_t) kv.v.nb[1]) {
        c.late(gate);
        const Tensor  qt    = c.run(q);
        const Tensor  gt    = c.to_public(gate);
        const int     split = 128;
        const int64_t rows  = B * L * Hq;
        const Tensor  att   = d.ws_alloc(F32, D * Hq, T);
        const Tensor  wsb   = d.ws_alloc(F32, kern::fa_decode_q40_workspace(in.n_kv, rows, split));
        void *        st    = adapter::stream(d);
        const float * pq    = (const float *) qt.data;
        const char *  pk    = (const char *) kv.k.data + rDH * (size_t) in.first_cell;
        const char *  pv    = (const char *) kv.v.data + rDH * (size_t) in.first_cell;
        const void *  pm    = in.mask.data;
        float *       po    = (float *) att.data;
        float *       pw    = (float *) wsb.data;
        const int64_t n_kv = in.n_kv, cs = (int64_t) rDH, hs = (int64_t) rD, ss = (int64_t) (rDH * region);
        const int64_t mr = in.mask.nb[1] / 2, ms = in.mask.nb[3] / 2;
        const float   sc = p.kq_scale;
        adapter::native(d, [=]() {
            if (!kern::fa_decode_q40(st, pq, pk, pv, pm, po, pw, D, Hq, Hkv, L, B, n_kv, split, cs, hs, ss, mr, ms,
                                     sc)) {
                fprintf(stderr, "[fa-native] declined a shape the host check accepted\n");
                abort();
            }
        });
        Call          c2(d);
        ggml_context * g2  = c2.ctx();
        ggml_tensor *  cur = ggml_mul(g2, c2.in(att, "fa_native"), ggml_sigmoid(g2, c2.in(gt, "attn_gate")));
        cur                = ggml_mul_mat(g2, c2.in(w.o, "wo"), cur);
        return c2.run(cur);
    }

    ggml_tensor * kview = ggml_view_4d(g, kc, D, Hkv, in.n_kv, B, rD, rDH, rDH * region, rDH * in.first_cell);
    ggml_tensor * vview = ggml_view_4d(g, vc, D, Hkv, in.n_kv, B, rD, rDH, rDH * region, rDH * in.first_cell);

    ggml_tensor * qp = ggml_permute(g, ggml_reshape_4d(g, q, D, Hq, L, B), 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(g, kview, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(g, vview, 0, 2, 1, 3);

    ggml_tensor * fa = ggml_flash_attn_ext(g, qp, kp, vp, c.in(in.mask, "kq_mask"), p.kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

    ggml_tensor * cur = ggml_reshape_2d(g, fa, fa->ne[0] * fa->ne[1], fa->ne[2] * fa->ne[3]);
    cur = ggml_mul(g, cur, ggml_sigmoid(g, gate));
    cur = ggml_mul_mat(g, c.in(w.o, "wo"), cur);
    Tensor out = c.run(cur);
    if (ap && k_ap == 2) {
        const Tensor    qt = c.to_public(q), gt2 = c.to_public(gate);
        void *          st = adapter::stream(d);
        const float *   a_q = (const float *) ap_q.data, *b_q = (const float *) qt.data;
        const float *   a_g = (const float *) ap_g.data, *b_g = (const float *) gt2.data;
        const void *    a_k = ap_k.data, *a_v = ap_v.data;
        const void *    kcp = kv.k.data, *vcp = kv.v.data;
        const int64_t   kcr = (int64_t) kv.k.nb[1] / 2, vcr = (int64_t) kv.v.nb[1] / 2;
        const int64_t * pcell = (const int64_t *) in.write_cells.data;
        const int64_t   nq = p.head_dim * p.n_head * T, width = p.head_dim * p.n_head_kv, TT = T;
        adapter::native(d, [=]() {
            kern::attn_prologue_compare(st, a_q, b_q, nq, a_g, b_g, nq, a_k, kcp, kcr, a_v, vcp, vcr, pcell, TT, width);
        });
    }
    return out;
}

// AI_GDN_GATE tail shared by the fused (plain decode) and unfused (verify/record) GDN paths: the native
// gated-norm kernel writes this layer's gate buffer and its q8_1 copy, then ssm_out runs as its own Call.
// Returns an invalid Tensor when the shape is not covered; the caller then keeps the ggml tail.
static bool gdn_gate_ok(const GdnWeights & w, const GdnState & s, int64_t Sv, int64_t Hv, int64_t T) {
    static const bool k_gdn_gate = getenv("AI_GDN_GATE") ? atoi(getenv("AI_GDN_GATE")) != 0 : true;
    return k_gdn_gate && w.norm.valid() && w.norm.type == F32 && w.norm.numel() == Sv && s.gate_out.valid() &&
           s.gate_out.numel() >= Sv * Hv * T && Sv % 32 == 0 && Sv <= 256;
}

static Tensor gdn_gate_tail(Device & d, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                            const Tensor & outt, const float * pz, int64_t zt, int64_t Sv, int64_t Hv, int64_t T) {
    const int64_t NH  = Sv * Hv;
    const Tensor  fin = s.gate_out.view(NH, T, 1, 1, (size_t) NH * 4, (size_t) NH * 4 * T, (size_t) NH * 4 * T, 0);
    void *        pq8 = nullptr;
    if (s.gate_q8.valid() && NH % 512 == 0 && kern::q8_stage_enabled() && s.gate_q8.numel() >= NH / 32 * 36 * T) {
        pq8 = s.gate_q8.data;
    }
    void *        st  = adapter::stream(d);
    const float * po  = (const float *) outt.data;
    const float * pnw = (const float *) w.norm.data;
    float *       pf  = (float *) fin.data;
    const float   eps = p.eps;
    adapter::native(d, [=]() {
        if (!bound_delete("gdngate")) {
            kern::gdn_gated_norm_q8(st, po, NH, pz, zt, pnw, pf, pq8, Sv, Hv, T, eps);
        }
        if (pq8 != nullptr) {
            kern::q8_stage_put(pf, pq8, NH, T, 1, 1, NH, NH * T, NH * T);
        }
    });
    Call          c2(d);
    ggml_tensor * y2 = ggml_mul_mat(c2.ctx(), c2.in(w.out, "ssm_out"), c2.in(fin, "gdn_fin"));
    return c2.run(ggml_reshape_2d(c2.ctx(), y2, y2->ne[0], T));
}

// AI_FUSE_GDN: gdn_block with the prologue in one native kernel. The ggml chain
// CONCAT -> SSM_CONV -> SILU -> gdn_prep is four small launches per layer, none of them bytes-bound;
// kern::gdn_conv_prep does all four and hands the op packed q/k/gate/beta, so the op runs with its
// fuse flags off. Without a GdnRecord the history roll is in place and only race-free at L == 1;
// in record mode the state is not rolled, so any L works.
static Tensor gdn_block_fused(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p,
                              const GdnState & s, GdnRecord * rec = nullptr) {
    const int64_t T  = x.ne[1];
    const int64_t B  = s.conv.ne[1];
    const int64_t L  = T / B;
    const int64_t C  = p.channels;
    const int64_t Sk = p.head_k, Hk = p.k_heads;
    const int64_t Sv = p.head_v, Hv = p.v_heads;
    const int64_t N  = C + Sv * Hv;   // rows of the fused qkv|z projection

    // AI_BOUND_SKIP_BA is timing only: the output is deliberately wrong. Deleting the GEMV and timing
    // the whole engine bounds any beta|alpha optimisation from above without a profiler. Token count
    // stays fixed; output hashes will not match.
    // 1. projections (one ggml segment)
    Tensor fo_t, ba_t;
    const Tensor ba_nat = k_bound_skip_ba ? Tensor{} : small_gemv_f16(d, w.ba, x);   // AI_SMALL_GEMV
    // AI_GDN_BA_CONC (default on): the small, latency-bound beta|alpha GEMV on stream 1, concurrent with the qkvz
    // projection; joined before the fused prologue reads both. Outputs are identical. =0 turns it off.
    static const bool k_ba_conc = knob("AI_GDN_BA_CONC", true);
    if (k_ba_conc && !k_bound_skip_ba && !ba_nat.valid()) {
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "[gdn-ba-conc] beta|alpha GEMV on stream 1, qkvz on stream 0\n");
        }
        const SideFork f = side_begin(d);
        {
            Call cb(d);
            ba_t = cb.run(ggml_mul_mat(cb.ctx(), cb.in(w.ba, "wba"), cb.in(x, "gdn_in_ba")));
        }
        side_end(d, f);
        {
            Call c(d);
            fo_t = c.run(ggml_mul_mat(c.ctx(), c.in(w.qkvz, "wqkvz_fused"), c.in(x, "gdn_in")));
        }
        side_join(d, f);
    } else {
        Call          c  = Call(d);
        ggml_context * g = c.ctx();
        ggml_tensor * gx = c.in(x, "gdn_in");
        ggml_tensor * fo = ggml_mul_mat(g, c.in(w.qkvz, "wqkvz_fused"), gx);
        if (ba_nat.valid()) {
            fo_t = c.run(fo);
            ba_t = ba_nat;
        } else if (k_bound_skip_ba) {
            fo_t = c.run(fo);
        } else {
            ggml_tensor * ba = ggml_mul_mat(g, c.in(w.ba, "wba"), gx);
            c.effect(ba);
            fo_t = c.run(fo);
            ba_t = c.to_public(ba);
        }
    }
    if (k_bound_skip_ba) {
        // Uninitialised workspace of the right shape: the output is garbage, but the shape and token
        // count are correct, which is all a timing bound needs.
        ba_t = d.ws_alloc(F32, w.ba.ne[1], L * B);
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "[bound] AI_BOUND_SKIP_BA: ba GEMV skipped, output is wrong by design. "
                            "Timing bound only -- do not report this arm's text or hash.\n");
        }
    }

    // 2. the fused prologue (one native segment). In record mode (MTP verify, AI_FUSE_GDN_REC) the state is not
    // rolled in place (the fold commits it later), which also makes L > 1 race-free, and the kernel writes the
    // history ++ new columns the fold's conv gather needs.
    const int64_t K1s      = p.conv_k - 1;
    Tensor        conv_in  = rec ? d.ws_alloc(F32, K1s + L, C, B) : Tensor{};
    Tensor conv_out = d.ws_alloc(F32, C, L * B);
    Tensor qh       = d.ws_alloc(F32, Sk, Hk, L, B);
    Tensor kh       = d.ws_alloc(F32, Sk, Hk, L, B);
    Tensor gh       = d.ws_alloc(F32, Hv, L * B);
    Tensor bh       = d.ws_alloc(F32, Hv, L * B);
    {
        void *          st  = adapter::stream(d);
        const float *   qkv = (const float *) fo_t.data;
        const float *   ba  = (const float *) ba_t.data;
        const float *   cw  = (const float *) w.conv.data;
        const float *   dtb = (const float *) w.dt_bias.data;
        const float *   ssa = (const float *) w.a.data;
        float *         cs  = (float *) s.conv.data;
        float *         co  = (float *) conv_out.data;
        float *         pq  = (float *) qh.data;
        float *         pk  = (float *) kh.data;
        float *         pg  = (float *) gh.data;
        float *         pb  = (float *) bh.data;
        const int64_t   K   = p.conv_k;
        const float     eps = p.eps;
        const bool    upd = rec == nullptr;
        float *       pci = rec ? (float *) conv_in.data : nullptr;
        adapter::native(d, [=]() {
            if (!bound_delete("gdnconv")) {
                kern::gdn_conv_prep(st, qkv, N, N * L, ba, 2 * Hv, 2 * Hv * L, cw, dtb, ssa, cs, co, pq, pk, pg, pb,
                                    (int) C, (int) K, (int) Sk, (int) Hk, (int) Hv, L, B, eps, upd, pci);
            }
        });
    }
    if (rec) {
        rec->T          = T;
        rec->L          = L;
        rec->B          = B;
        rec->conv_input = conv_in;
        rec->conv_out   = conv_out;
        rec->ba         = ba_t;
        rec->qh         = qh;
        rec->kh         = kh;
        rec->gh         = gh;
        rec->bh         = bh;
    }

    const size_t es = sizeof(float);

    // 3. the recurrence and the output projection (one ggml segment)
    Call          c  = Call(d);
    ggml_context * g = c.ctx();
    ggml_tensor * q  = ggml_reshape_4d(g, c.in(qh, "gdn_q"), Sk, Hk, L, B);
    ggml_tensor * k  = ggml_reshape_4d(g, c.in(kh, "gdn_k"), Sk, Hk, L, B);
    ggml_tensor * co = c.in(conv_out, "gdn_conv_out");
    ggml_tensor * v  = ggml_view_4d(g, co, Sv, Hv, L, B, es * Sv, es * C, es * C * L, es * 2 * Sk * Hk);
    ggml_tensor * gt = ggml_view_4d(g, c.in(gh, "gdn_gate"), 1, Hv, L, B, es, es * Hv, es * Hv * L, 0);
    ggml_tensor * bt = ggml_view_4d(g, c.in(bh, "gdn_beta"), 1, Hv, L, B, es, es * Hv, es * Hv * L, 0);
    ggml_tensor * ssm_slot = c.in(s.ssm, "ssm_slot");
    ggml_tensor * state    = ggml_reshape_4d(g, ssm_slot, Sv, Sv, Hv, B);

    // fuse flags off: the native kernel already applied the q/k L2 norm, the gate and sigmoid(beta)
    ggml_tensor * res = ggml_gated_delta_net_ext(g, q, k, v, gt, bt, state, c.in(w.dt_bias, "dt_bias"),
                                                 c.in(w.a, "ssm_a"), p.eps, rec ? 0 : GGML_GDN_STATE_INPLACE);

    const auto    rs  = [&](int64_t n) { return ggml_row_size(res->type, n); };
    ggml_tensor * out = ggml_view_4d(g, res, Sv, Hv, L, B, rs(Sv), rs(Sv * Hv), rs(Sv * Hv * L), 0);

    // AI_GDN_GATE (default on): norm + SiLU gate + the out-projection's quantise in one native kernel
    if (res->type == GGML_TYPE_F32 && gdn_gate_ok(w, s, Sv, Hv, T)) {
        const Tensor outt = c.run(out);
        return gdn_gate_tail(d, w, p, s, outt, (const float *) fo_t.data + C, fo_t.nb[1] / 4, Sv, Hv, T);
    }

    ggml_tensor * normed = ggml_mul(g, ggml_rms_norm(g, out, p.eps), c.in(w.norm, "ssm_norm"));
    ggml_tensor * fo2    = c.in(fo_t, "gdn_qkvz");
    ggml_tensor * z3     = ggml_view_3d(g, fo2, Sv * Hv, L, B, fo2->nb[1], fo2->nb[1] * L, (size_t) C * es);
    ggml_tensor * fin    = ggml_swiglu_split(g, z3, ggml_reshape_3d(g, normed, Sv * Hv, L, B));
    ggml_tensor * y      = ggml_mul_mat(g, c.in(w.out, "ssm_out"), fin);
    return c.run(ggml_reshape_2d(g, y, y->ne[0], T));
}

Tensor gdn_block(Device & d, const Tensor & x, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                 GdnRecord * rec) {
    // AI_V2_F16X projection: computed before the Call so it is recorded first, and skipped when the
    // fused-GDN early return below will be taken, since that path has its own projection.
    Tensor v2qkvz;
    {
        static const bool k_fuse = []() {
            const char * e = getenv("AI_FUSE_GDN");
            return e ? atoi(e) != 0 : true;   // default on
        }();
        const int64_t Bv = s.conv.ne[1];
        const bool fused_path = k_fuse && !rec && Bv > 0 && x.ne[1] / Bv == 1 && w.qkvz.valid() && w.ba.valid() &&
                                p.conv_k - 1 <= 8;
        if (!fused_path) {
            v2qkvz = v2_proj(d, w.qkvz, x, 0);
        }
    }
    // AI_GEMM_F16_PF: the beta|alpha projection at prefill width, native, issued before this function's ggml segment
    // (it reads only x). Prefill only: record mode (verify) keeps its ggml product.
    const Tensor ba_pf = (!rec && x.ne[1] > 8 && w.ba.valid() && !k_bound_skip_ba) ? small_gemv_f16(d, w.ba, x) : Tensor{};
    Call c(d);
    ggml_context * g = c.ctx();

    const int64_t T  = x.ne[1];
    const int64_t B  = s.conv.ne[1];
    const int64_t L  = T / B;
    const int64_t K1 = p.conv_k - 1;
    const int64_t C  = p.channels;
    const int64_t Sk = p.head_k, Hk = p.k_heads;
    const int64_t Sv = p.head_v, Hv = p.v_heads;
    if (L * B != T) {
        throw std::runtime_error("gdn_block: rows are not a multiple of the stream count");
    }
    static const int k_fuse_gdn = []() {
        // Default on; AI_FUSE_GDN=0 turns it off.
        const char * e = getenv("AI_FUSE_GDN");
        return e ? atoi(e) : 1;
    }();
    // AI_FUSE_GDN_REC (default on): MTP verify rounds (record mode, any L) through the same fused prologue plain
    // decode uses, replacing concat + ssm_conv + gdn_prep per layer. Verify numerics then match plain decode's
    // prologue, so the output text differs from =0. =0 turns it off.
    static const bool k_fuse_gdn_rec = getenv("AI_FUSE_GDN_REC") ? atoi(getenv("AI_FUSE_GDN_REC")) != 0 : true;
    if (k_fuse_gdn && ((!rec && L == 1) || (rec && k_fuse_gdn_rec)) && w.qkvz.valid() && w.ba.valid() &&
        p.conv_k - 1 <= 8) {
        return gdn_block_fused(d, x, w, p, s, rec);
    }

    ggml_tensor * gx = c.in(x, "gdn_in");

    ggml_tensor * qkv = nullptr;
    ggml_tensor * z3  = nullptr;   // [Sv*Hv, T, 1], possibly a strided view
    if (w.qkvz.valid()) {
        ggml_tensor * fo  = v2qkvz.valid() ? c.in(v2qkvz, "v2_qkvz")
                                           : ggml_mul_mat(g, c.in(w.qkvz, "wqkvz_fused"), gx);   // [C + Sv*Hv, T]
        const size_t  nb1 = fo->nb[1];
        qkv = ggml_view_3d(g, fo, C, L, B, nb1, nb1 * L, 0);
        z3  = ggml_view_3d(g, fo, Sv * Hv, L, B, nb1, nb1 * L, (size_t) C * ggml_element_size(fo));
    } else {
        qkv = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.qkv, "wqkv"), gx), C, L, B);
        z3  = ggml_reshape_3d(g, ggml_mul_mat(g, c.in(w.z, "wz"), gx), Sv * Hv, L, B);
    }
    ggml_tensor * beta  = nullptr;
    ggml_tensor * alpha = nullptr;
    ggml_tensor * ba_fo = nullptr;
    if (w.ba.valid()) {
        // AI_BOUND_SKIP_BA, timing only (see gdn_block_fused): this unfused path is taken whenever the
        // fused one is not. ggml_scale by 0 of a view of the input keeps the node and the shape but
        // drops the GEMV's weight bytes and dot product. Output is deliberately wrong; token count is
        // unaffected.
        if (k_bound_skip_ba) {
            // The liveness line names this site and its shape, so it shows which ba site actually ran.
            static bool warned = false;
            if (!warned) {
                warned = true;
                fprintf(stderr,
                        "[bound] AI_BOUND_SKIP_BA active at site=gdn_block (the unfused path, taken when "
                        "gdn_block_fused is not; AI_FUSE_GDN=%d) ba=[%lld,%lld] Hv=%lld L*B=%lld -- GEMV "
                        "skipped, output wrong by design, timing bound only.\n",
                        (int) k_fuse_gdn, (long long) w.ba.ne[0], (long long) w.ba.ne[1],
                        (long long) Hv, (long long) (L * B));
            }
        }
        ggml_tensor * fo = k_bound_skip_ba
                             ? ggml_scale(g, ggml_reshape_2d(g, ggml_cont(g, ggml_view_2d(
                                   g, gx, 2 * Hv, L * B, gx->nb[1], 0)), 2 * Hv, L * B), 0.0f)
                             : ba_pf.valid() ? c.in(ba_pf, "ba_pf")                       // AI_GEMM_F16_PF
                                             : ggml_mul_mat(g, c.in(w.ba, "wba"), gx);   // [2Hv, T]
        ba_fo = fo;
        const size_t  e  = ggml_element_size(fo);
        beta  = ggml_view_4d(g, fo, 1, Hv, L, B, e, fo->nb[1], fo->nb[1] * L, 0);
        alpha = ggml_view_4d(g, fo, 1, Hv, L, B, e, fo->nb[1], fo->nb[1] * L, (size_t) Hv * e);
    } else {
        beta  = ggml_reshape_4d(g, ggml_mul_mat(g, c.in(w.beta, "wbeta"), gx), 1, Hv, L, B);
        alpha = ggml_reshape_4d(g, ggml_mul_mat(g, c.in(w.alpha, "walpha"), gx), 1, Hv, L, B);
    }

    // AI_GDN16, prefill: ggml's fused prep + recurrence replaced by the native 16-token chunked kernel
    // (kernels/gdn_chunk16.cu). Prefill only (not record mode). The new state is written in place.
    // The native gate tail's staging buffer is sized for decode / verify rows, so at prefill length the ggml tail
    // (norm, SiLU gate, out projection) runs on the native output instead.
    // Decided before the conv nodes so that AI_GDN_CONV_PF (kernels/gdn_conv_pf.cu) can replace the concat + SSM_CONV
    // + SiLU chain and its conv-state publish effect; the concat is a large share of long-prompt prefill.
    const bool g16 = !rec && L >= 16 && kern::gdn16_enabled() && kern::gdn16_ok(Sk, Sv, Hk, Hv, L) && ba_fo &&
                     w.qkvz.valid() && z3->type == GGML_TYPE_F32 && ba_fo->type == GGML_TYPE_F32 &&
                     w.dt_bias.type == F32 && w.a.type == F32;
    const bool g16_conv = g16 && kern::gdn_conv_pf_enabled() && qkv->type == GGML_TYPE_F32 && w.conv.type == F32 &&
                          p.conv_k >= 3 && p.conv_k <= 5;
    auto g16_said = [&](const char * conv) {
        static bool said = false;   // printed on a prefill-sized call only (calibration's short calls also land here)
        if (!said && T >= 512) {
            said = true;
            fprintf(stderr, "[gdn16] LIVE at site=gdn_block (unfused prefill path) L=%lld B=%lld Hk=%lld Hv=%lld tail=%s conv=%s\n",
                    (long long) L, (long long) B, (long long) Hk, (long long) Hv,
                    gdn_gate_ok(w, s, Sv, Hv, T) ? "native" : "ggml", conv);
        }
    };
    auto g16_finish = [&](const Tensor & co, const Tensor & zt, const Tensor & bat) -> Tensor {
        const Tensor outt = d.ws_alloc(F32, Sv * Hv, T);
        const Tensor scr  = d.ws_alloc(I8, (int64_t) kern::gdn16_prefill_scratch_bytes(Hk, Hv, L, B));
        {
            void *        st      = adapter::stream(d);
            const float * pco     = (const float *) co.data;
            const float * pba     = (const float *) bat.data;
            const int64_t bat_tok = bat.nb[1] / 4;
            const float * pdt     = (const float *) w.dt_bias.data;
            const float * pa      = (const float *) w.a.data;
            float *       pst     = (float *) s.ssm.data;
            float *       po      = (float *) outt.data;
            void *        ps      = scr.data;
            const float   eps     = p.eps;
            adapter::native(d, [=]() {
                kern::gdn16_prefill(st, pco, C, pba, bat_tok, pdt, pa, eps, pst, po, Hk, Hv, L, B, ps);
            });
        }
        if (gdn_gate_ok(w, s, Sv, Hv, T)) {
            return gdn_gate_tail(d, w, p, s, outt, (const float *) zt.data, zt.nb[1] / 4, Sv, Hv, T);
        }
        Call           c2     = Call(d);
        ggml_context * g2     = c2.ctx();
        ggml_tensor *  o4     = ggml_reshape_4d(g2, c2.in(outt, "gdn16_out"), Sv, Hv, L, B);
        ggml_tensor *  normed = ggml_mul(g2, ggml_rms_norm(g2, o4, p.eps), c2.in(w.norm, "ssm_norm"));
        ggml_tensor *  zz     = c2.in(zt, "gdn16_z");
        ggml_tensor *  fin    = ggml_swiglu_split(g2, zz, ggml_reshape_3d(g2, normed, Sv * Hv, L, B));
        ggml_tensor *  y      = ggml_mul_mat(g2, c2.in(w.out, "ssm_out"), fin);
        return c2.run(ggml_reshape_2d(g2, y, y->ne[0], T));
    };
    if (g16_conv) {
        g16_said("native");
        c.late(z3);
        c.late(ba_fo);
        const Tensor qt  = c.run(qkv);   // the projections only: no concat, no SSM_CONV, no conv-state effect
        const Tensor zt  = c.to_public(z3);
        const Tensor bat = c.to_public(ba_fo);
        const Tensor co  = d.ws_alloc(F32, C, T);
        {
            void *        st  = adapter::stream(d);
            const float * px  = (const float *) qt.data;
            const int64_t xt  = qt.nb[1] / 4, xs = qt.nb[2] / 4;
            float *       ph  = (float *) s.conv.data;
            const float * pw  = (const float *) w.conv.data;
            float *       pco = (float *) co.data;
            const int64_t K   = p.conv_k;
            adapter::native(d, [=]() { kern::gdn_conv_pf(st, px, xt, xs, ph, pw, pco, C, K, L, B); });
        }
        return g16_finish(co, zt, bat);
    }

    // conv history [K-1, C] ++ new columns [T, C] -> [K-1+T, C]
    ggml_tensor * conv_slot   = c.in(s.conv, "conv_slot");
    ggml_tensor * conv_states = ggml_reshape_3d(g, conv_slot, K1, C, B);
    ggml_tensor * conv_input  = ggml_concat(g, conv_states, ggml_transpose(g, qkv), 0);
    ggml_tensor * last_cols   = ggml_view_3d(g, conv_input, K1, C, B, conv_input->nb[1], conv_input->nb[2],
                                             (conv_input->ne[0] - K1) * ggml_element_size(conv_input));
    if (!rec) {
        c.effect(ggml_cpy(g, last_cols, k_conv_view ? ggml_view_1d(g, conv_slot, K1 * C * B, 0) : conv_slot));
    }

    ggml_tensor * conv_out = ggml_silu(g, ggml_ssm_conv(g, conv_input, c.in(w.conv, "conv1d")));

    const size_t es  = ggml_element_size(conv_out);
    const size_t nb1 = (size_t) C * es;
    ggml_tensor * q = ggml_view_4d(g, conv_out, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, 0);
    ggml_tensor * k = ggml_view_4d(g, conv_out, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, es * Sk * Hk);
    ggml_tensor * v = ggml_view_4d(g, conv_out, Sv, Hv, L, B, es * Sv, nb1, nb1 * L, es * 2 * Sk * Hk);

    ggml_tensor * ssm_slot = c.in(s.ssm, "ssm_slot");
    ggml_tensor * state    = ggml_reshape_4d(g, ssm_slot, Sv, Sv, Hv, B);

    if (g16) {   // conv on ggml (AI_GDN_CONV_PF=0, or a conv shape the native kernel does not take)
        g16_said("ggml");
        c.late(z3);
        c.late(ba_fo);
        const Tensor co  = c.run(conv_out);   // also runs the conv-state publish effect above
        const Tensor zt  = c.to_public(z3);
        const Tensor bat = c.to_public(ba_fo);
        return g16_finish(co, zt, bat);
    }

    // the op writes the new state straight into the committed slot (no publish copy)
    const bool    inplace = k_gdn_inplace && !rec;
    const int32_t flags = GGML_GDN_FUSE_L2NORM_QK | GGML_GDN_FUSE_SIGMOID_BETA | GGML_GDN_FUSE_GATE |
                          (inplace ? GGML_GDN_STATE_INPLACE : 0);
    ggml_tensor * res = ggml_gated_delta_net_ext(g, q, k, v, alpha, beta, state,
                                                 c.in(w.dt_bias, "dt_bias"), c.in(w.a, "ssm_a"), p.eps, flags);

    const auto rs = [&](int64_t n) { return ggml_row_size(res->type, n); };
    ggml_tensor * out = ggml_view_4d(g, res, Sv, Hv, L, B, rs(Sv), rs(Sv * Hv), rs(Sv * Hv * L), 0);
    if (!inplace && !rec) {
        ggml_tensor * new_state = ggml_view_4d(g, res, Sv, Sv, Hv, B, rs(Sv), rs(Sv * Sv), rs(Sv * Sv * Hv),
                                               rs(Sv * Hv * L * B));
        c.effect(ggml_cpy(g, new_state, ssm_slot));
    }

    // AI_GDN_GATE_VERIFY (default on): the same native tail on the unfused path MTP verify rounds use, in
    // record mode. Bit-exact. AI_GDN_GATE_VERIFY=0 turns it off.
    static const bool k_gate_verify = getenv("AI_GDN_GATE_VERIFY") ? atoi(getenv("AI_GDN_GATE_VERIFY")) != 0 : true;
    if (k_gate_verify && w.qkvz.valid() && res->type == GGML_TYPE_F32 && z3->type == GGML_TYPE_F32 &&
        gdn_gate_ok(w, s, Sv, Hv, T)) {
        c.late(z3);
        if (rec) {
            c.late(conv_input);
            c.late(conv_out);
            if (ba_fo) {
                c.late(ba_fo);
            } else {
                c.late(beta);
                c.late(alpha);
            }
        }
        const Tensor outt = c.run(out);
        const Tensor zt   = c.to_public(z3);
        if (rec) {
            rec->T          = T;
            rec->L          = L;
            rec->B          = B;
            rec->conv_input = c.to_public(conv_input);
            rec->conv_out   = c.to_public(conv_out);
            if (ba_fo) {
                rec->ba = c.to_public(ba_fo);
            } else {
                rec->beta  = c.to_public(beta);
                rec->alpha = c.to_public(alpha);
            }
        }
        return gdn_gate_tail(d, w, p, s, outt, (const float *) zt.data, zt.nb[1] / 4, Sv, Hv, T);
    }

    ggml_tensor * normed = ggml_mul(g, ggml_rms_norm(g, out, p.eps), c.in(w.norm, "ssm_norm"));
    ggml_tensor * fin    = nullptr;
    if (w.qkvz.valid()) {
        // silu(z) * normed on the strided z view (the same products as the split path)
        fin = ggml_swiglu_split(g, z3, ggml_reshape_3d(g, normed, Sv * Hv, L, B));
    } else {
        ggml_tensor * z4    = ggml_reshape_4d(g, z3, Sv, Hv, L, B);
        ggml_tensor * gated = ggml_mul(g, normed, ggml_silu(g, z4));
        fin = ggml_reshape_3d(g, gated, Sv * Hv, L, B);
    }
    ggml_tensor * y      = ggml_mul_mat(g, c.in(w.out, "ssm_out"), fin);
    if (rec) {
        c.late(conv_input);
        c.late(conv_out);
        if (ba_fo) {
            c.late(ba_fo);
        } else {
            c.late(beta);
            c.late(alpha);
        }
    }
    Tensor out_t = c.run(ggml_reshape_2d(g, y, y->ne[0], T));
    if (rec) {
        rec->T          = T;
        rec->L          = L;
        rec->B          = B;
        rec->conv_input = c.to_public(conv_input);
        rec->conv_out   = c.to_public(conv_out);
        if (ba_fo) {
            rec->ba = c.to_public(ba_fo);
        } else {
            rec->beta  = c.to_public(beta);
            rec->alpha = c.to_public(alpha);
        }
    }
    return out_t;
}

void gdn_fold(Device & d, const GdnWeights & w, const GdnParams & p, const GdnState & s,
              const GdnRecord & rec, int64_t m1) {
    if (m1 < 1 || m1 > rec.T || rec.B != 1) {
        throw std::runtime_error("gdn_fold: accepted columns out of range (or batched record)");
    }
    Call c(d);
    ggml_context * g = c.ctx();
    const int64_t K1 = p.conv_k - 1;
    const int64_t C  = p.channels;
    const int64_t Sk = p.head_k, Hk = p.k_heads;
    const int64_t Sv = p.head_v, Hv = p.v_heads;

    ggml_tensor * co  = c.in(rec.conv_out, "rec_conv_out");
    const size_t  es  = ggml_element_size(co);
    const size_t  nb1 = (size_t) C * es;
    ggml_tensor * q = ggml_view_4d(g, co, Sk, Hk, m1, 1, es * Sk, nb1, nb1 * m1, 0);
    ggml_tensor * k = ggml_view_4d(g, co, Sk, Hk, m1, 1, es * Sk, nb1, nb1 * m1, es * Sk * Hk);
    ggml_tensor * v = ggml_view_4d(g, co, Sv, Hv, m1, 1, es * Sv, nb1, nb1 * m1, es * 2 * Sk * Hk);
    ggml_tensor * beta = nullptr;
    ggml_tensor * alpha = nullptr;
    if (rec.ba.valid()) {
        ggml_tensor * fo = c.in(rec.ba, "rec_ba");
        const size_t  e  = ggml_element_size(fo);
        beta  = ggml_view_4d(g, fo, 1, Hv, m1, 1, e, fo->nb[1], fo->nb[1] * m1, 0);
        alpha = ggml_view_4d(g, fo, 1, Hv, m1, 1, e, fo->nb[1], fo->nb[1] * m1, (size_t) Hv * e);
    } else {
        ggml_tensor * b = c.in(rec.beta, "rec_beta");
        ggml_tensor * a = c.in(rec.alpha, "rec_alpha");
        // [1, Hv, L, B]: nb1 = per-head stride, nb2 = per-column stride. Using nb[0] / nb[1] here would read the
        // columns with the head stride and fold with the wrong gates.
        beta  = ggml_view_4d(g, b, 1, Hv, m1, 1, b->nb[1], b->nb[2], b->nb[2] * m1, 0);
        alpha = ggml_view_4d(g, a, 1, Hv, m1, 1, a->nb[1], a->nb[2], a->nb[2] * m1, 0);
    }
    ggml_tensor * ssm_slot = c.in(s.ssm, "ssm_slot");
    ggml_tensor * state    = ggml_reshape_4d(g, ssm_slot, Sv, Sv, Hv, 1);
    const int32_t flags = GGML_GDN_FUSE_L2NORM_QK | GGML_GDN_FUSE_SIGMOID_BETA | GGML_GDN_FUSE_GATE |
                          GGML_GDN_STATE_INPLACE;
    c.effect(ggml_gated_delta_net_ext(g, q, k, v, alpha, beta, state,
                                      c.in(w.dt_bias, "dt_bias"), c.in(w.a, "ssm_a"), p.eps, flags));

    // conv history after m1 columns = columns [m1, m1+K-1) of history ++ new
    ggml_tensor * ci   = c.in(rec.conv_input, "rec_conv_input");
    ggml_tensor * last = ggml_view_3d(g, ci, K1, C, 1, ci->nb[1], ci->nb[2], (size_t) m1 * ggml_element_size(ci));
    ggml_tensor * slot = c.in(s.conv, "conv_slot");
    c.effect(ggml_cpy(g, last, ggml_view_1d(g, slot, K1 * C, 0)));
    c.run_effects_only();
}

void gdn_fold_masked(Device & d, const GdnWeights & w, const GdnParams & p, const GdnState & s,
                     const GdnRecord & rec, const Tensor & colmask, const Tensor & m) {
    const int64_t L  = rec.L, B = rec.B;
    const int64_t K1 = p.conv_k - 1;
    const int64_t C  = p.channels;
    const int64_t Sk = p.head_k, Hk = p.k_heads;
    const int64_t Sv = p.head_v, Hv = p.v_heads;
    static const bool k_fold_packed = getenv("AI_FOLD_PACKED") ? atoi(getenv("AI_FOLD_PACKED")) != 0 : true;
    const bool        packed = k_fold_packed && rec.qh.valid() && rec.kh.valid() && rec.gh.valid() && rec.bh.valid();
    // AI_FOLD_PREP (default on): the packed path's exp(colmask) + two muls + conv_gather as one native launch before
    // the Call. Bit-identical: the masked factor is exactly 1 or 0, and the gather is conv_gather_k's copy.
    static const bool k_fold_prep = getenv("AI_FOLD_PREP") ? atoi(getenv("AI_FOLD_PREP")) != 0 : true;
    const bool        prep = packed && k_fold_prep && rec.gh.contiguous() && rec.bh.contiguous() &&
                      colmask.contiguous() && rec.conv_input.contiguous() && s.conv.contiguous() && m.contiguous();
    Tensor gtm_t, btm_t;
    if (prep) {
        gtm_t = d.ws_alloc(F32, Hv, L * B);
        btm_t = d.ws_alloc(F32, Hv, L * B);
        void *          st  = adapter::stream(d);
        const float *   pgh = (const float *) rec.gh.data, *pbh = (const float *) rec.bh.data;
        const float *   pcm = (const float *) colmask.data;
        float *         pgt = (float *) gtm_t.data, *pbt = (float *) btm_t.data;
        const float *   csrc = (const float *) rec.conv_input.data;
        float *         cdst = (float *) s.conv.data;
        const int32_t * pm   = (const int32_t *) m.data;
        const int64_t   Ci = C, K1i = K1, Hvi = Hv, Li = L, Bi = B;
        adapter::native(d, [=]() { kern::gdn_fold_prep(st, pgh, pbh, pcm, pgt, pbt, Hvi, Li, Bi, csrc, K1i, Ci, pm, cdst); });
    }
    {
        Call c(d);
        ggml_context * g = c.ctx();
        ggml_tensor * co  = c.in(rec.conv_out, "rec_conv_out");
        const size_t  es  = ggml_element_size(co);
        const size_t  nb1 = (size_t) C * es;
        ggml_tensor * q = ggml_view_4d(g, co, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, 0);
        ggml_tensor * k = ggml_view_4d(g, co, Sk, Hk, L, B, es * Sk, nb1, nb1 * L, es * Sk * Hk);
        ggml_tensor * v = ggml_view_4d(g, co, Sv, Hv, L, B, es * Sv, nb1, nb1 * L, es * 2 * Sk * Hk);
        // AI_FOLD_PACKED (default on): reuse the fused verify's packed q/k/gate/beta. Masked columns need
        // gate = 0 and beta = 0 -- exactly what adding -inf before softplus/sigmoid gave -- so multiply by
        // exp(colmask) (exactly 0 or 1). Fuse flags off: nothing is re-derived, and the committed state
        // uses the same prologue numerics as plain decode.
        if (packed) {
            ggml_tensor * q2  = ggml_reshape_4d(g, c.in(rec.qh, "rec_qh"), Sk, Hk, L, B);
            ggml_tensor * k2  = ggml_reshape_4d(g, c.in(rec.kh, "rec_kh"), Sk, Hk, L, B);
            const size_t  ef  = sizeof(float);
            ggml_tensor * gt  = nullptr;
            ggml_tensor * bt  = nullptr;
            if (prep) {
                gt = ggml_view_4d(g, c.in(gtm_t, "fold_gt"), 1, Hv, L, B, ef, ef * Hv, ef * Hv * L, 0);
                bt = ggml_view_4d(g, c.in(btm_t, "fold_bt"), 1, Hv, L, B, ef, ef * Hv, ef * Hv * L, 0);
            } else {
                ggml_tensor * m01 = ggml_exp(g, ggml_reshape_4d(g, c.in(colmask, "colmask"), 1, 1, L, B));
                gt = ggml_view_4d(g, c.in(rec.gh, "rec_gh"), 1, Hv, L, B, ef, ef * Hv, ef * Hv * L, 0);
                bt = ggml_view_4d(g, c.in(rec.bh, "rec_bh"), 1, Hv, L, B, ef, ef * Hv, ef * Hv * L, 0);
                gt = ggml_mul(g, gt, m01);
                bt = ggml_mul(g, bt, m01);
            }
            ggml_tensor * state2 = ggml_reshape_4d(g, c.in(s.ssm, "ssm_block"), Sv, Sv, Hv, B);
            c.effect(ggml_gated_delta_net_ext(g, q2, k2, v, gt, bt, state2, c.in(w.dt_bias, "dt_bias"),
                                              c.in(w.a, "ssm_a"), p.eps, GGML_GDN_STATE_INPLACE));
            c.run_effects_only();
        } else {
        ggml_tensor * beta = nullptr;
        ggml_tensor * alpha = nullptr;
        if (rec.ba.valid()) {
            ggml_tensor * fo = c.in(rec.ba, "rec_ba");
            const size_t  e  = ggml_element_size(fo);
            beta  = ggml_view_4d(g, fo, 1, Hv, L, B, e, fo->nb[1], fo->nb[1] * L, 0);
            alpha = ggml_view_4d(g, fo, 1, Hv, L, B, e, fo->nb[1], fo->nb[1] * L, (size_t) Hv * e);
        } else {
            beta  = c.in(rec.beta, "rec_beta");
            alpha = c.in(rec.alpha, "rec_alpha");
        }
        ggml_tensor * cm = ggml_reshape_4d(g, c.in(colmask, "colmask"), 1, 1, L, B);
        beta  = ggml_add(g, ggml_cont(g, beta), cm);
        alpha = ggml_add(g, ggml_cont(g, alpha), cm);
        ggml_tensor * state = ggml_reshape_4d(g, c.in(s.ssm, "ssm_block"), Sv, Sv, Hv, B);
        const int32_t flags = GGML_GDN_FUSE_L2NORM_QK | GGML_GDN_FUSE_SIGMOID_BETA | GGML_GDN_FUSE_GATE |
                              GGML_GDN_STATE_INPLACE;
        c.effect(ggml_gated_delta_net_ext(g, q, k, v, alpha, beta, state,
                                          c.in(w.dt_bias, "dt_bias"), c.in(w.a, "ssm_a"), p.eps, flags));
        c.run_effects_only();
        }
    }
    if (!rec.conv_input.contiguous() || !s.conv.contiguous() || !m.contiguous()) {
        throw std::runtime_error("gdn_fold_masked: conv gather needs contiguous tensors");
    }
    if (prep) {
        return;   // the conv gather ran in the prep launch
    }
    void * st = adapter::stream(d);
    const float *   src = (const float *) rec.conv_input.data;
    float *         dst = (float *) s.conv.data;
    const int32_t * pm  = (const int32_t *) m.data;
    adapter::native(d, [=]() { kern::conv_gather(st, src, K1, L, C, B, pm, dst); });
}

Accept mtp_accept(Device & d, const Tensor & g, const Tensor & ids, int64_t L, int64_t B) {
    if (!g.contiguous() || !ids.contiguous() || g.numel() != L * B || ids.numel() < L * B) {
        throw std::runtime_error("mtp_accept: bad inputs");
    }
    Accept a;
    a.m       = d.ws_alloc(I32, B);
    a.colmask = d.ws_alloc(F32, L * B);
    void * st = adapter::stream(d);
    const int32_t * pg = (const int32_t *) g.data;
    const int32_t * pi = (const int32_t *) ids.data;
    int32_t *       pm = (int32_t *) a.m.data;
    float *         pc = (float *) a.colmask.data;
    adapter::native(d, [=]() { kern::mtp_accept(st, pg, pi, L, B, pm, pc); });
    return a;
}

Tensor select_rows(Device & d, const Tensor & src, int64_t L, int64_t B, const Tensor & m) {
    const bool    vec  = src.ne[1] > 1 || src.ne[0] != L * B;   // [E, T] vs [T]
    const int64_t rows = vec ? src.numel() / src.ne[0] : src.ne[0];
    if (!src.contiguous() || rows != L * B) {
        throw std::runtime_error("select_rows: bad source");
    }
    Tensor      dst = vec ? d.ws_alloc(src.type, src.ne[0], B) : d.ws_alloc(src.type, B);
    const int64_t row_bytes = vec ? (int64_t) src.nb[1] : (int64_t) src.nb[0];
    void * st = adapter::stream(d);
    const void *    ps = src.data;
    void *          pd = dst.data;
    const int32_t * pm = (const int32_t *) m.data;
    adapter::native(d, [=]() { kern::select_rows(st, ps, L, B, row_bytes, pm, pd); });
    return dst;
}

void head_causal_mask(Device & d, const Tensor & mask, int64_t T, int64_t n_kv, int64_t pos0) {
    void *     st = adapter::stream(d);
    uint16_t * pk = (uint16_t *) mask.data;
    adapter::native(d, [=]() { kern::head_causal_mask(st, pk, T, n_kv, pos0); });
}

void decode_causal_mask(Device & d, const Tensor & mask, const Tensor & base, int64_t L, int64_t B, int64_t n_kv,
                        bool head) {
    void *          st = adapter::stream(d);
    uint16_t *      pk = (uint16_t *) mask.data;
    const int32_t * pb = (const int32_t *) base.data;
    const int       h  = head ? 1 : 0;
    adapter::native(d, [=]() { kern::decode_causal_mask(st, pk, pb, L, B, n_kv, h); });
}

void mtp_step_inputs(Device & d, const Tensor & base, const Tensor & m, int64_t B, int offset,
                     int64_t region_cells, int64_t first_cell, int64_t n_kv, const Tensor & pos4,
                     const Tensor & cells, const Tensor & mask, const Tensor & delta) {
    void * st = adapter::stream(d);
    const int32_t * pb = (const int32_t *) base.data;
    const int32_t * pm = (const int32_t *) m.data;
    const int32_t * pdl = delta.valid() ? (const int32_t *) delta.data : nullptr;
    int32_t *       pp = (int32_t *) pos4.data;
    int64_t *       pc = (int64_t *) cells.data;
    uint16_t *      pk = (uint16_t *) mask.data;
    adapter::native(d, [=]() {
        kern::mtp_step_inputs(st, pb, pm, pdl, B, offset, region_cells, first_cell, n_kv, pp, pc, pk);
    });
}

Tensor mtp_input(Device & d, const Tensor & table, const Tensor & ids, const Tensor & H,
                 const Tensor & enorm, const Tensor & hnorm, const Tensor & eh_proj, float eps) {
    const bool    q4k = table.type == Q4_K && ids.type == I32 && ids.contiguous();
    const Tensor  et  = q4k ? embed_native_q4k(d, table, ids) : Tensor{};
    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor * e  = q4k ? c.in(et, "mtp_embd") : ggml_get_rows(g, c.in(table, "tok_embd"), c.in(ids, "mtp_ids"));
    ggml_tensor * en = ggml_mul(g, ggml_rms_norm(g, e, eps), c.in(enorm, "enorm"));
    ggml_tensor * hn = ggml_mul(g, ggml_rms_norm(g, c.in(H, "mtp_h"), eps), c.in(hnorm, "hnorm"));
    return c.run(ggml_mul_mat(g, c.in(eh_proj, "eh_proj"), ggml_concat(g, en, hn, 0)));
}

Tensor moe_block(Device & d, const Tensor & x, const MoeWeights & w, const MoeParams & p) {
    Call c(d);
    ggml_context * g = c.ctx();

    const int64_t T = x.ne[1];
    const int64_t E = x.ne[0];
    const int     k = p.n_used;

    ggml_tensor * gx = c.in(x, "moe_in");

    // router: softmax -> top-k -> renormalised weights
    ggml_tensor * logits  = ggml_mul_mat(g, c.in(w.router, "router"), gx);
    ggml_tensor * probs   = ggml_soft_max(g, logits);
    ggml_tensor * sel     = ggml_argsort_top_k(g, probs, k);
    ggml_tensor * weights = ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, p.n_expert, T), sel);
    weights = ggml_reshape_2d(g, weights, k, T);
    ggml_tensor * wsum = ggml_clamp(g, ggml_sum_rows(g, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(g, ggml_div(g, weights, wsum), 1, k, T);
    if (k_topk_order) {
        c.effect(weights);   // launch the routing chain first, contiguous, as the fused top-k kernel expects
    }

    // routed experts
    ggml_tensor * cur  = ggml_reshape_3d(g, gx, E, 1, T);
    ggml_tensor * up   = ggml_mul_mat_id(g, c.in(w.up_exps, "up_exps"), cur, sel);
    ggml_tensor * gate = ggml_mul_mat_id(g, c.in(w.gate_exps, "gate_exps"), cur, sel);
    ggml_tensor * act  = ggml_swiglu_split(g, gate, up);
    ggml_tensor * exps = ggml_mul_mat_id(g, c.in(w.down_exps, "down_exps"), act, sel);
    exps = ggml_mul(g, exps, weights);

    ggml_tensor * moe = ggml_view_2d(g, exps, E, T, exps->nb[2], 0);
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(g, moe, ggml_view_2d(g, exps, E, T, exps->nb[2], (size_t) i * exps->nb[1]));
    }

    // shared expert, sigmoid-gated
    ggml_tensor * su  = ggml_mul_mat(g, c.in(w.sh_up, "up_shexp"), gx);
    ggml_tensor * sg  = ggml_mul_mat(g, c.in(w.sh_gate, "gate_shexp"), gx);
    ggml_tensor * sd  = ggml_mul_mat(g, c.in(w.sh_down, "down_shexp"), ggml_swiglu_split(g, sg, su));
    ggml_tensor * sgt = ggml_sigmoid(g, ggml_mul_mat(g, c.in(w.sh_score, "shexp_score"), gx));
    ggml_tensor * sh  = ggml_mul(g, sd, sgt);

    return c.run(ggml_add(g, moe, sh));
}

// Fused decode MoE: router|score in one projection, shared gate|up in one projection, and the
// shared-gate sigmoid + scale folded into the native merge. 4 fewer launches per layer.
static MoeOut moe_residual_fused(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w,
                                 const MoeParams & p, bool fuse_gate_up, const Tensor * next_norm,
                                 const Tensor * norm_buf, float eps, const Tensor * q8_buf = nullptr,
                                 const Tensor * router_pre = nullptr, bool router_cmp = false) {
    const int64_t T  = x.ne[1];
    const int64_t E  = x.ne[0];
    const int     k  = p.n_used;
    const int64_t NE = p.n_expert;
    const int64_t Fs = w.sh_gate.ne[1];
    Tensor exps, weights, sdown, scores, ids_pub;
    // AI_MOE_DOWN9: replace the two down projections and the native merge with one kernel that is
    // one block per output row and one warp per expert path.
    static const bool k_down9_env = []() {
        const char * e = getenv("AI_MOE_DOWN9");
        return e ? atoi(e) != 0 : false;
    }();
    // The width test must happen here, before the graph is shaped: under down9 the graph below omits
    // both down projections, so a kernel that declined at execution time would leave nothing to fall
    // back to. The rule itself lives in ggml_cuda_moe_down9_width_ok; this only asks.
    const bool k_down9 = k_down9_env && kern::moe_down9_width_ok(T);
    const Tensor v2gu = fuse_gate_up ? v2_proj(d, w.sh_gate_up, x, 2) : Tensor{};   // AI_V2_F16X, before the Call
    // AI_MOE_CONCURRENT (default on): the shared expert runs on the backend's second stream, concurrent with the
    // router and the routed experts; they meet at the merge. Fork/join are event pairs, captured as parallel graph
    // branches. The stream switch happens inside the recorded sequence (native lambdas run in order with the
    // Calls), so it applies to exactly the Call between them. Only the default shape (separate shared
    // gate/up, no down9, no v2 projection). Outputs are identical; this relies on ggml's MMVQ global q8 cache
    // being restricted to stream 0. AI_MOE_CONCURRENT=0 turns it off.
    static const bool k_moe_conc = getenv("AI_MOE_CONCURRENT") ? atoi(getenv("AI_MOE_CONCURRENT")) != 0 : true;
    const bool conc = k_moe_conc && !k_down9 && !fuse_gate_up && !v2gu.valid();
    Tensor sdown_c;
    void * ev_join = nullptr;
    void * st_main = nullptr;
    if (conc) {
        static std::vector<void *> evs;
        static size_t              evi = 0;
        const auto next_ev = [&]() {
            if (evs.size() < 256) {
                evs.push_back(kern::event_create());
                return evs.back();
            }
            return evs[(evi++) % evs.size()];
        };
        void * ev_fork = next_ev();
        ev_join        = next_ev();
        st_main        = adapter::stream(d);
        void * st_side = kern::backend_stream_no(d.backend(), 1);
        ggml_backend * bk = d.backend();
        static bool said = false;
        if (!said) {
            said = true;
            fprintf(stderr, "[moe-concurrent] shared expert on stream 1, routed experts on stream 0\n");
        }
        adapter::native(d, [=]() {
            kern::event_record(ev_fork, st_main);
            kern::stream_wait_event(st_side, ev_fork);
            kern::backend_set_stream_no(bk, 1);
        });
        {
            Call cs(d);
            ggml_context * g  = cs.ctx();
            ggml_tensor *  gx = cs.in(x, "moe_in_shexp");
            ggml_tensor *  su = ggml_mul_mat(g, cs.in(w.sh_up, "up_shexp"), gx);
            ggml_tensor *  sg = ggml_mul_mat(g, cs.in(w.sh_gate, "gate_shexp"), gx);
            sdown_c = cs.run(ggml_mul_mat(g, cs.in(w.sh_down, "down_shexp"), ggml_swiglu_split(g, sg, su)));
        }
        void * evj = ev_join;
        adapter::native(d, [=]() {
            kern::backend_set_stream_no(bk, 0);
            kern::event_record(evj, st_side);
        });
    }
    // AI_ADDNORM_ROUTER: the logits were computed by the norm launch; =2 only compares them at the end
    const bool   rs_pre = router_pre != nullptr && router_pre->valid() && !router_cmp && router_pre->ne[1] == T;
    const Tensor rs_nat = rs_pre ? *router_pre : small_gemv_f16(d, w.route_score, x);   // AI_SMALL_GEMV, on the main stream
    // AI_TOPK_NATIVE: top-8 from the logits the add+norm launch produced (kernels/topk_native.cu), bit-exact with
    // topk_moe_cuda. 0 off; 1 on; 2 compare (ggml routes, native writes shadows, bitwise counts at exit).
    // Default: on for compute capability < 12, off for >= 12 (Blackwell), where it is slower than ggml's top-k.
    static const int k_tk = getenv("AI_TOPK_NATIVE") ? atoi(getenv("AI_TOPK_NATIVE"))
                                                     : (kern::device_cc_major() >= 12 ? 0 : 1);
    Tensor           tk_ids, tk_w;
    if (k_tk != 0 && rs_nat.valid() && rs_nat.type == F32 && rs_nat.ne[0] == NE + 1 && kern::topk_native_ok(NE, k, T) &&
        !k_down9 && getenv("AI_ROUTER_TOPK") == nullptr) {
        tk_ids               = d.ws_alloc(I32, k, T);
        tk_w                 = d.ws_alloc(F32, 1, k, T);
        void *        st_tk  = adapter::stream(d);
        const float * plg    = (const float *) rs_nat.data;
        int32_t *     pid_tk = (int32_t *) tk_ids.data;
        float *       pw_tk  = (float *) tk_w.data;
        const int64_t nr = NE + 1, Tt = T;
        adapter::native(d, [=]() {
            if (!kern::topk_native(st_tk, plg, nr, Tt, pid_tk, pw_tk, 6.103515625e-5f)) {
                fprintf(stderr, "[topk-native] declined a shape the host check accepted\n");
                abort();
            }
        });
    }
    const bool tk_use = tk_ids.valid() && k_tk == 1;
    // AI_ROUTER_TOPK (opt-in, not faster): router GEMV + top-k as one launch (kernels/router_topk.cu). Bit-exact
    // at T=1 and T=4, but the logits differ at T=2 because ggml dispatches that width to a different kernel, so do
    // not enable it with 2 parallel streams. The last block's serial top-k costs what the saved launch did.
    // 0 off; 1 on; 2 compare: compute both, generate from the graph's, count bitwise differences (instrument)
    static const int k_router_topk = getenv("AI_ROUTER_TOPK") ? atoi(getenv("AI_ROUTER_TOPK")) : 0;
    Tensor rt_logits, rt_ids, rt_w;
    if (k_router_topk && !rs_nat.valid() && !k_down9 && w.route_score.valid() && w.route_score.type == F16 &&
        w.route_score.contiguous() && w.route_score.ne[1] == NE + 1 && x.type == F32 && x.contiguous() && T <= 8 &&
        NE == 256 && k == 8) {
        rt_logits = d.ws_alloc(F32, NE + 1, T);
        rt_ids    = d.ws_alloc(I32, k, T);
        rt_w      = d.ws_alloc(F32, 1, k, T);
        void *        st  = adapter::stream(d);
        const void *  pw  = w.route_score.data;
        const float * px  = (const float *) x.data;
        float *       pl  = (float *) rt_logits.data;
        int32_t *     pid = (int32_t *) rt_ids.data;
        float *       pwt = (float *) rt_w.data;
        const int64_t Ei = E, NEi = NE, ki = k, Ti = T;
        adapter::native(d, [=]() {
            if (!kern::router_topk(st, pw, px, pl, pid, pwt, Ei, NEi + 1, NEi, ki, Ti, Ei, 6.103515625e-5f)) {
                fprintf(stderr, "[router-topk] declined a shape the host check accepted\n");
                abort();
            }
        });
    }
    // AI_MOE_GU: at MTP-verify widths (T 2..4) the routed gate + up + SwiGLU run as one native kernel with
    // expert dedup and x reuse (kernels/moe_gu_verify.cu), bit-exact with the ggml chain. The routed block is
    // split in two Calls around it: router/top-k, then the down projection over the kernel's activations.
    // 0 off; 1 on (default); 2 compare: run both, generate from the ggml chain, count bitwise differences.
    static const int k_gu = getenv("AI_MOE_GU") ? atoi(getenv("AI_MOE_GU")) : 1;
    // AI_MOE_TC=1: the same split, with gate/up + SwiGLU on tensor cores at T 2..8 (kernels/moe_gu_tc.cu). Not
    // bit-exact: the app sets it only for the fast verify.
    static const bool k_tc = getenv("AI_MOE_TC") && atoi(getenv("AI_MOE_TC")) != 0;
    const bool gu_shape = k_gu != 0 && conc && !k_down9 && k_router_topk == 0 && w.up_exps.valid() &&
                          w.gate_exps.valid() && w.up_exps.type == w.gate_exps.type &&
                          w.up_exps.nb[1] == w.gate_exps.nb[1] && w.up_exps.nb[2] == w.gate_exps.nb[2] &&
                          x.type == F32 && x.contiguous() && E % 512 == 0;
    const bool tc = gu_shape && k_tc && kern::moe_gu_tc_ok(w.up_exps.type, E, w.up_exps.ne[1], k, T);
    // AI_MOE_PF, prefill-sized rounds: routed gate/up/down as grouped int8 tensor-core GEMMs (kernels/moe_pf.cu),
    // in place of ggml's MMQ MUL_MAT_ID. Not bit-exact with MMQ (own activation quantiser), but plain and MTP prefill
    // both take it, so MTP == plain holds; perplexity is unchanged within noise. 1 on (default), 0 off.
    // AI_MOE_PF_MIN is the smallest round width that takes it (default 64).
    static const int     k_pf     = getenv("AI_MOE_PF") ? atoi(getenv("AI_MOE_PF")) : 1;
    static const int64_t k_pf_min = getenv("AI_MOE_PF_MIN") ? atoll(getenv("AI_MOE_PF_MIN")) : 64;
    const bool pf = gu_shape && !tc && k_pf != 0 && T >= k_pf_min &&
                    kern::moe_pf_ok(w.up_exps.type, E, w.up_exps.ne[1], k, NE, T);
    const bool gu = gu_shape && (pf || tc || kern::moe_gu_verify_ok(w.up_exps.type, E, w.up_exps.ne[1], k, T));
    if (gu) {
        const int64_t F = w.up_exps.ne[1];
        Tensor        sel_pub;
        if (tk_use) {
            sel_pub = tk_ids;
            weights = tk_w;
            scores  = rs_nat;
        } else {
            Call c(d);
            ggml_context * g  = c.ctx();
            ggml_tensor *  gx = c.in(x, "moe_in");
            ggml_tensor *  rs = rs_nat.valid() ? c.in(rs_nat, "router_score_nat")
                                               : ggml_mul_mat(g, c.in(w.route_score, "router_score"), gx);
            ggml_tensor * logits = ggml_cont(g, ggml_view_2d(g, rs, NE, T, rs->nb[1], 0));
            ggml_tensor * probs  = ggml_soft_max(g, logits);
            ggml_tensor * sel    = ggml_argsort_top_k(g, probs, k);
            ggml_tensor * wts    = ggml_reshape_2d(g, ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, NE, T), sel), k, T);
            ggml_tensor * wsum   = ggml_clamp(g, ggml_sum_rows(g, wts), 6.103515625e-5f, INFINITY);
            wts                  = ggml_reshape_3d(g, ggml_div(g, wts, wsum), 1, k, T);
            c.effect(wts);
            c.effect(sel);
            // ggml's topk_moe matcher (ggml_cuda_topk_moe_fusion) reads one node past the norm chain's final
            // reshape without a bounds check, looking for a scale. A segment that ends at that reshape, as this
            // split does, would read past the node array. A no-op reshape (no kernel) gives it a node to read.
            (void) c.run(ggml_reshape_2d(g, wts, k, T));
            sel_pub = c.to_public(sel);
            weights = c.to_public(wts);
            scores  = rs_nat.valid() ? rs_nat : c.to_public(rs);
        }
        if (tk_ids.valid() && k_tk == 2) {   // verify width: native top-k against the graph's
            void *          stc = adapter::stream(d);
            const int32_t * a_i = (const int32_t *) tk_ids.data;
            const float *   a_w = (const float *) tk_w.data;
            const int32_t * b_i = (const int32_t *) sel_pub.data;
            const float *   b_w = (const float *) weights.data;
            const int64_t   bit = sel_pub.nb[1] / 4, bwt = weights.nb[2] / 4, TT = T;
            adapter::native(d, [=]() { kern::topk_native_compare(stc, a_i, a_w, b_i, bit, b_w, bwt, TT); });
        }
        Tensor        act_t = d.ws_alloc(F32, F, k, T);
        Tensor        xq_t  = pf ? d.ws_alloc(F32, (int64_t) (kern::moe_pf_scratch_bytes(E, F, k, NE, T) / 4 + 1), 1)
                              : d.ws_alloc(F32, tc ? (int64_t) (kern::moe_gu_tc_scratch_bytes(E, T) / 4 + 1) / T + 1
                                                 : (E / 32) * 9, T);   // q8_1 fallback (36 B / 32), TC or PF scratch
        void *        st    = adapter::stream(d);
        const float * px    = (const float *) x.data;
        void *        pxq   = xq_t.data;
        const void *  pu = w.up_exps.data, *pg = w.gate_exps.data;
        const int64_t rb = (int64_t) w.up_exps.nb[1], eb = (int64_t) w.up_exps.nb[2];
        const int32_t * pid = (const int32_t *) sel_pub.data;
        const int64_t ids_stride = (int64_t) sel_pub.nb[1] / (int64_t) sizeof(int32_t);
        float *       pa = (float *) act_t.data;
        const int     ty = w.up_exps.type;
        const int64_t xcs = (int64_t) x.nb[1] / (int64_t) sizeof(float);
        // AI_GU_Q8_BOUND, timing instrument only (wrong output): register an unfilled buffer as the q8_1 copy of the
        // gate/up activations, so the down projection's quantise launch is deleted and nothing replaces it. The
        // round-time change is the strict upper bound of emitting that copy from this kernel.
        static const bool k_gu_q8_bound = getenv("AI_GU_Q8_BOUND") && atoi(getenv("AI_GU_Q8_BOUND")) != 0;
        void * pq8b = nullptr;
        // AI_GU_Q8: the gate/up kernel also emits the down projection's q8_1 input (bit-exact with ggml's
        // quantiser) and registers it, so the down GEMV skips its own quantise launch. Fast and exact verify alike;
        // a lookup that misses (exact mode's sample-axis geometry) just quantises as before.
        static const bool k_gu_q8 = getenv("AI_GU_Q8") && atoi(getenv("AI_GU_Q8")) != 0;
        void * pq8e = nullptr;
        if (k_gu_q8 && !k_gu_q8_bound && !tc && !pf && kern::q8_stage_enabled() && F % 512 == 0) {
            pq8e = d.ws_alloc(F32, (F * k * T / 32 * 36) / 4 + 64, 1).data;
        }
        if (k_gu_q8_bound) {
            const int64_t q8_bytes = F * k * T / 32 * 36;
            pq8b = d.ws_alloc(F32, q8_bytes / 4 + 64, 1).data;
        }
        const int64_t k_used = k, n_exp = NE;
        adapter::native(d, [=]() {
            if (pf) {
                if (!kern::moe_pf_gu(st, ty, pu, pg, rb, eb, px, xcs, pid, ids_stride, pa, E, F, k_used, n_exp, T, pxq)) {
                    fprintf(stderr, "[moe-pf] gate/up declined a shape the host check accepted\n");
                    abort();
                }
                return;
            }
            if (pq8b != nullptr) {
                static bool said = false;
                if (!said) {
                    said = true;
                    fprintf(stderr, "[gu-q8-bound] ACTIVE: down-input quantise DELETED (wrong output, timing bound only)\n");
                }
                kern::q8_stage_put(pa, pq8b, F, k, T, 1, F, F * k, F * k * T);
            }
            if (tc) {
                if (!kern::moe_gu_tc(st, ty, pu, pg, rb, eb, px, xcs, pxq, pid, ids_stride, pa, E, F, T)) {
                    fprintf(stderr, "[moe-tc] declined a shape the host check accepted\n");
                    abort();
                }
                return;
            }
            // the post-attention norm staged x's q8_1 copy earlier in this pass; look it up at record time
            static long long hit = 0, miss = 0;
            static bool      armed = false;
            if (!armed) {
                armed = true;
                atexit([]() { fprintf(stderr, "[moe-gu] x q8_1: staged %lld, quantised here %lld\n", hit, miss); });
            }
            const void * xq = kern::q8_stage_get(px, E, T, 1, 1, E, E * T, E * T);
            if (xq != nullptr) {
                hit++;
            } else {
                miss++;
                kern::q8_stage_fill_reference(st, px, pxq, E, T);
                xq = pxq;
            }
            if (!kern::moe_gu_verify(st, ty, pu, pg, rb, eb, xq, pid, ids_stride, pa, E, F, T, pq8e)) {
                fprintf(stderr, "[moe-gu] declined a shape the host check accepted\n");
                abort();
            }
            if (pq8e != nullptr) {
                static const int k_stage = getenv("AI_Q8_STAGE") ? atoi(getenv("AI_Q8_STAGE")) : 1;
                if (k_stage == 3) {   // byte-compare against the reference quantiser (rows = slots x tokens)
                    kern::q8_stage_verify(st, pa, pq8e, F, k * T);
                }
                kern::q8_stage_put(pa, pq8e, F, k, T, 1, F, F * k, F * k * T);
            }
        });
        const bool pf_down = pf && kern::moe_pf_down_ok(w.down_exps.type, F, w.down_exps.ne[1]);
        // AI_MOE_DN: the routed down projection at verify width as one native kernel with the same expert dedup +
        // weight-unpack reuse (kernels/moe_gu_verify.cu), bit-exact with ggml's ids MMVQ. 0 off (default); 1 on;
        // 2 compare: run both, generate from the ggml chain, count bitwise differences.
        static const int k_dn = getenv("AI_MOE_DN") ? atoi(getenv("AI_MOE_DN")) : 0;
        const bool dn = k_dn != 0 && !tc && !pf_down && k_gu != 2 &&
                        kern::moe_down_verify_ok(w.down_exps.type, F, w.down_exps.ne[1], k, T);
        Tensor dn_out;
        if (dn) {
            const int64_t Nd  = w.down_exps.ne[1];
            dn_out            = d.ws_alloc(F32, Nd, k, T);
            const bool fill   = pq8e == nullptr;   // AI_GU_Q8 already emitted the q8_1 copy of act
            void *     pq     = fill ? d.ws_alloc(F32, (F * k * T / 32 * 36) / 4 + 64, 1).data : pq8e;
            const void *  pd  = w.down_exps.data;
            const int     tyd = w.down_exps.type;
            const int64_t rbd = (int64_t) w.down_exps.nb[1], ebd = (int64_t) w.down_exps.nb[2];
            float *       po  = (float *) dn_out.data;
            adapter::native(d, [=]() {
                if (fill) {
                    kern::q8_stage_fill_reference(st, pa, pq, F, k_used * T);
                }
                if (!kern::moe_down_verify(st, tyd, pd, rbd, ebd, pq, pid, ids_stride, po, F, Nd, T)) {
                    fprintf(stderr, "[moe-dn] declined a shape the host check accepted\n");
                    abort();
                }
            });
        }
        if (dn && k_dn != 2) {
            exps = dn_out;
        } else if (pf_down) {   // AI_MOE_PF: the down projection over the gate/up kernel's int8 copy of the activations
            exps = d.ws_alloc(F32, w.down_exps.ne[1], k, T);
            const void *  pd  = w.down_exps.data;
            const int     tyd = w.down_exps.type;
            const int64_t rbd = (int64_t) w.down_exps.nb[1], ebd = (int64_t) w.down_exps.nb[2], Nd = w.down_exps.ne[1];
            float *       po  = (float *) exps.data;
            adapter::native(d, [=]() {
                if (!kern::moe_pf_down(st, tyd, pd, rbd, ebd, po, E, F, Nd, k_used, n_exp, T, pxq)) {
                    fprintf(stderr, "[moe-pf] down declined a shape the host check accepted\n");
                    abort();
                }
            });
        } else {
            Call c(d);
            ggml_context * g   = c.ctx();
            ggml_tensor *  s   = c.in(sel_pub, "moe_sel");
            ggml_tensor *  act = c.in(act_t, "moe_act_gu");
            ggml_tensor *  ref = nullptr;
            if (k_gu == 2) {
                ggml_tensor * cur  = ggml_reshape_3d(g, c.in(x, "moe_in"), E, 1, T);
                ggml_tensor * up   = ggml_mul_mat_id(g, c.in(w.up_exps, "up_exps"), cur, s);
                ggml_tensor * gate = ggml_mul_mat_id(g, c.in(w.gate_exps, "gate_exps"), cur, s);
                ref                = ggml_swiglu_split(g, gate, up);
                c.effect(ref);
            }
            exps = c.run(ggml_mul_mat_id(g, c.in(w.down_exps, "down_exps"), ref != nullptr ? ref : act, s));
            if (ref != nullptr) {
                const Tensor  rt = c.to_public(ref);
                const float * pr = (const float *) rt.data;
                const int64_t n  = F * k * T;
                adapter::native(d, [=]() { kern::moe_gu_compare(st, pa, pr, n, T); });
            }
            if (dn) {   // AI_MOE_DN=2
                const float * pm = (const float *) dn_out.data;
                const float * pr = (const float *) exps.data;
                const int64_t n  = w.down_exps.ne[1] * k * T;
                adapter::native(d, [=]() { kern::moe_down_compare(st, pm, pr, n); });
            }
        }
        sdown   = sdown_c;
        ids_pub = sel_pub;
    } else {
        Call c(d);
        ggml_context * g = c.ctx();
        ggml_tensor * gx = c.in(x, "moe_in");

        ggml_tensor * rs  = nullptr;
        ggml_tensor * sel = nullptr;
        ggml_tensor * wts = nullptr;
        size_t        es  = sizeof(float);
        if (rt_logits.valid() && k_router_topk == 1) {
            rs  = c.in(rt_logits, "router_score_rt");
            sel = c.in(rt_ids, "router_ids_rt");
            wts = c.in(rt_w, "router_wts_rt");
        } else if (tk_use) {
            rs  = c.in(rs_nat, "router_score_nat");
            sel = c.in(tk_ids, "router_ids_native");
            wts = c.in(tk_w, "router_wts_native");
        } else {
        if (bound_delete("router")) {   // timing instrument: stale workspace logits instead of the router GEMV
            rs = c.in(d.ws_alloc(F32, NE + 1, T), "router_deleted");
        } else {
        rs     = rs_nat.valid() ? c.in(rs_nat, "router_score_nat")
                                : ggml_mul_mat(g, c.in(w.route_score, "router_score"), gx);   // [NE+1, T]
        }
        es     = ggml_element_size(rs);
        ggml_tensor * logits = ggml_view_2d(g, rs, NE, T, rs->nb[1], 0);
        if (T > 1) {
            logits = ggml_cont(g, logits);
        }
        ggml_tensor * probs  = ggml_soft_max(g, logits);
        sel    = ggml_argsort_top_k(g, probs, k);
        wts    = ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, NE, T), sel);
        wts = ggml_reshape_2d(g, wts, k, T);
        ggml_tensor * wsum = ggml_clamp(g, ggml_sum_rows(g, wts), 6.103515625e-5f, INFINITY);
        wts = ggml_reshape_3d(g, ggml_div(g, wts, wsum), 1, k, T);
        c.effect(wts);
        }

        ggml_tensor * cur  = ggml_reshape_3d(g, gx, E, 1, T);
        ggml_tensor * up   = ggml_mul_mat_id(g, c.in(w.up_exps, "up_exps"), cur, sel);
        ggml_tensor * gate = ggml_mul_mat_id(g, c.in(w.gate_exps, "gate_exps"), cur, sel);
        ggml_tensor * act  = ggml_swiglu_split(g, gate, up);
        ggml_tensor * down = k_down9 ? nullptr
                                     : ggml_mul_mat_id(g, c.in(w.down_exps, "down_exps"), act, sel);

        ggml_tensor * sg = nullptr;
        ggml_tensor * su = nullptr;
        if (conc) {
            // shared expert already launched on stream 1 above
        } else if (fuse_gate_up) {
            ggml_tensor * gu  = v2gu.valid() ? c.in(v2gu, "v2_gate_up_shexp")
                                             : ggml_mul_mat(g, c.in(w.sh_gate_up, "gate_up_shexp"), gx);   // [2Fs, T]
            const size_t  nb1 = gu->nb[1];
            sg = ggml_view_2d(g, gu, Fs, T, nb1, 0);
            su = ggml_view_2d(g, gu, Fs, T, nb1, (size_t) Fs * ggml_element_size(gu));
        } else {
            // separate projections: the backend fuses gate + up + SwiGLU into one MMVQ call
            su = ggml_mul_mat(g, c.in(w.sh_up, "up_shexp"), gx);
            sg = ggml_mul_mat(g, c.in(w.sh_gate, "gate_shexp"), gx);
        }
        ggml_tensor * sact = conc ? nullptr : ggml_swiglu_split(g, sg, su);
        ggml_tensor * sd   = (k_down9 || conc) ? nullptr : ggml_mul_mat(g, c.in(w.sh_down, "down_shexp"), sact);
        if (sd != nullptr) {
            c.late(sd);
        }

        if (k_down9) {
            // stop at the activations: the fused kernel does both downs, the merge, the gate and
            // the residual, so the [n_embd, k, T] per-slot intermediate never exists
            c.effect(sact);
            c.effect(sel);
            exps    = c.run(act);
            sdown   = c.to_public(sact);
            ids_pub = c.to_public(sel);
        } else {
            exps  = c.run(down);
            sdown = conc ? sdown_c : c.to_public(sd);
        }
        const bool rt_use = rt_w.valid() && k_router_topk == 1;
        if (tk_use) {
            weights = tk_w;
        }
        if (tk_ids.valid() && k_tk == 2) {   // compare against the graph's sel / wts
            const Tensor    sel_t = c.to_public(sel);
            const Tensor    w_t   = c.to_public(wts);
            void *          stc   = adapter::stream(d);
            const int32_t * a_i = (const int32_t *) tk_ids.data;
            const float *   a_w = (const float *) tk_w.data;
            const int32_t * b_i = (const int32_t *) sel_t.data;
            const float *   b_w = (const float *) w_t.data;
            const int64_t   bit = sel_t.nb[1] / 4, bwt = w_t.nb[2] / 4, TT = T;
            adapter::native(d, [=]() { kern::topk_native_compare(stc, a_i, a_w, b_i, bit, b_w, bwt, TT); });
        }
        weights = rt_use ? rt_w : c.to_public(wts);
        scores  = rt_use ? rt_logits : (rs_nat.valid() ? rs_nat : c.to_public(rs));
        if (rt_w.valid() && k_router_topk == 2) {
            const Tensor  sel_t = c.to_public(sel);
            void *        st    = adapter::stream(d);
            const float * al = (const float *) rt_logits.data;  const float * bl = (const float *) scores.data;
            const int32_t * ai = (const int32_t *) rt_ids.data; const int32_t * bi = (const int32_t *) sel_t.data;
            const float * aw = (const float *) rt_w.data;       const float * bw = (const float *) weights.data;
            const int64_t blt = scores.nb[1] / 4, bit = sel_t.nb[1] / 4, nl = NE + 1, kk = k, TT = T;
            adapter::native(d, [=]() { kern::router_topk_compare(st, al, bl, blt, nl, ai, bi, bit, aw, bw, kk, TT); });
        }
        if (getenv("AI_DUMP_DIR") && !recording()) {
            ids_pub = c.to_public(sel);   // metadata only, no launch
        }
        (void) es;
    }
    if (router_cmp && router_pre != nullptr && router_pre->valid() && router_pre->ne[1] == T && scores.valid()) {
        // AI_ADDNORM_ROUTER=2: the fused launch's logits against the ggml router's, word for word
        const float * a   = (const float *) router_pre->data;
        const float * b   = (const float *) scores.data;
        const int64_t n   = router_pre->ne[0] * T;
        void *        stc = adapter::stream(d);
        adapter::native(d, [=]() { kern::addnorm_router_compare(stc, a, b, n); });
    }
    if (conc) {
        // join: every merge path below reads sdown, which stream 1 produced
        void * evj = ev_join;
        void * stm = st_main;
        adapter::native(d, [=]() { kern::stream_wait_event(stm, evj); });
    }
    if (!exps.contiguous() || !weights.contiguous() || !sdown.contiguous() || !r.contiguous() ||
        !scores.contiguous()) {
        throw std::runtime_error("moe_residual_fused: native merge needs contiguous inputs");
    }
    // Router tap (AI_DUMP_DIR): logits differ -> router GEMV or upstream; logits equal but ids differ
    // -> top-k / argsort; both equal but y differs -> below the router.
    moe_dump(d, "ffn_moe_logits", scores);
    moe_dump(d, "ffn_moe_ids", ids_pub);
    moe_dump(d, "ffn_moe_weights", weights);

    MoeOut o;
    o.y = d.ws_alloc(F32, E, T);
    void * st = adapter::stream(d);
    const float * pe = (const float *) exps.data;
    const float * pw = (const float *) weights.data;
    const float * ps = (const float *) sdown.data;
    const float * pc = (const float *) scores.data;
    const float * pr = (const float *) r.data;
    float *       py = (float *) o.y.data;
    const int64_t stride = NE + 1, off = NE;

    // AI_FUSE_MERGENORM: fold the next layer's input norm into this kernel. The norm re-reads the
    // row this kernel just wrote, so as a separate kernel it is mostly launch cost.
    // 0 off; 1 the single-block fused kernel; 2 the cooperative one, which keeps the merge's
    // parallelism and pays two grid barriers for the row reduction; 3 full grid, last block norms;
    // 4 merge writes partial sums, a quantise-shaped kernel finishes the norm (+ stages q8_1); 5 = 4 with
    // the merge's two columns per lane interleaved.
    static const int k_merge_norm = []() {
        // Default 5: bit-exact with 0. Set 0 to get the separate norm back.
        const char * e = getenv("AI_FUSE_MERGENORM");
        return e ? atoi(e) : 5;
    }();
    // !k_down9 is required: under down9 the graph above stopped at the activations, so exps is
    // [n_ff, k, T] activations, not the [E, k, T] down outputs this kernel expects, and the kernel
    // cannot detect the mismatch.
    if (k_merge_norm && !k_down9 && next_norm != nullptr && next_norm->valid() && next_norm->type == F32 &&
        next_norm->numel() == E && norm_buf != nullptr && norm_buf->numel() >= E * T) {
        // As in add_rmsnorm: norm_buf is sized for the widest round (n_streams rows); narrow it to
        // this round's T.
        o.normed          = norm_buf->ne[0] == E
                                ? norm_buf->view(E, T, 1, 1, norm_buf->nb[1],
                                                 norm_buf->nb[1] * (size_t) T, norm_buf->nb[1] * (size_t) T, 0)
                                : *norm_buf;
        float *       pn  = (float *) o.normed.data;
        const float * pnw = (const float *) next_norm->data;
        if ((k_merge_norm == 4 || k_merge_norm == 5) && kern::moe_merge_partials_ok(E)) {
            // Off the critical path: partial sums from the merge, norm finished by a quantise-shaped
            // kernel (faster than rms_norm) that can also stage the q8_1 copy (ALPHA_Q8_STAGE=1).
            // =4 walks the columns serially and loses in the merge what the apply saves.
            Tensor  part_t = d.ws_alloc(F32, 32 * T);
            float * ppart  = (float *) part_t.data;
            void *  pq8    = nullptr;
            // q8 staging: bit-identical at T = 1..4, since the emitter's rounding matches ggml's quantiser
            // (half-to-even, exact.cuh). Opt-in (ALPHA_Q8_STAGE=1) until MTP rounds (T up to 8) are checked.
            if (q8_buf != nullptr && E % 512 == 0 && kern::q8_stage_enabled() &&
                q8_buf->numel() >= (E / 32) * 36 * T) {
                pq8 = q8_buf->data;
            }
            // AI_MERGE_APPLY: 0 off (default); 1 merge + apply in one launch; 2 compare (the pair runs and is consumed;
            // the fused kernel writes shadow y / out, compared bitwise). Bit-exact but slower: each of only 8 blocks
            // re-reads the whole merged row to rebuild the partials, which does not pay at this row size.
            static const int k_ma = getenv("AI_MERGE_APPLY") ? atoi(getenv("AI_MERGE_APPLY")) : 0;
            float *          sh_y = nullptr, *sh_o = nullptr;
            if (k_ma == 2) {
                sh_y = (float *) d.ws_alloc(F32, E, T).data;
                sh_o = (float *) d.ws_alloc(F32, E, T).data;
            }
            adapter::native(d, [=]() {
                if (k_ma == 1 && !bound_delete("merge") &&
                    kern::moe_merge_apply_redundant(st, pe, pw, ps, pc, stride, off, pr, py, pnw, pn, pq8, E, k, T, eps)) {
                } else {
                if (!bound_delete("merge")) {
                    kern::moe_merge_residual_gated_partials(st, pe, pw, ps, pc, stride, off, pr, py, ppart, E, k, T,
                                                           k_merge_norm == 5);
                }
                if (!bound_delete("rmsapply")) {
                    kern::rms_apply_partials_q8(st, py, ppart, pnw, pn, pq8, E, T, eps);
                }
                if (k_ma == 2 && kern::moe_merge_apply_redundant(st, pe, pw, ps, pc, stride, off, pr, sh_y, pnw, sh_o,
                                                                 nullptr, E, k, T, eps)) {
                    kern::addnorm_router_compare(st, sh_y, py, E * T);
                    kern::addnorm_router_compare(st, sh_o, pn, E * T);
                }
                }
                if (pq8 != nullptr) {
                    kern::q8_stage_put(pn, pq8, E, T, 1, 1, E, E * T, E * T);
                }
            });
            return o;
        }
        if (k_merge_norm == 3) {
            adapter::native(d, [=]() {
                kern::moe_merge_residual_gated_norm_last(st, pe, pw, ps, pc, stride, off, pr, py, pn, pnw, E, k, T,
                                                         eps);
            });
            return o;
        }
        if (k_merge_norm == 2) {
            Tensor  sc_t = d.ws_alloc(F32, T);
            float * psc  = (float *) sc_t.data;
            adapter::native(d, [=]() {
                if (!kern::moe_merge_residual_gated_norm_coop(st, pe, pw, ps, pc, stride, off, pr, py, pn, pnw,
                                                              psc, E, k, T, eps)) {
                    // refused (capture, or not co-resident): the single-block fusion is still correct
                    kern::moe_merge_residual_gated_norm(st, pe, pw, ps, pc, stride, off, pr, py, pn, pnw, E, k,
                                                        T, eps);
                }
            });
            return o;
        }
        adapter::native(d, [=]() {
            kern::moe_merge_residual_gated_norm(st, pe, pw, ps, pc, stride, off, pr, py, pn, pnw, E, k, T, eps);
        });
        return o;
    }
    if (k_down9) {
        Tensor sc_col = scores;   // the shared expert's raw gate is column NE of router|score
        sc_col.ne[0]  = 1;
        sc_col.ne[1]  = T;
        sc_col.data   = (char *) scores.data + (size_t) off * sizeof(float);
        ggml_backend * bk = d.backend();
        const Tensor dw = w.down_exps, sdw = w.sh_down, act_t = exps, ids_t = ids_pub, wts_t = weights,
                     sact_t = sdown, res_t = r, dst_t = o.y;
        adapter::native(d, [=]() {
            if (!kern::moe_down9(bk, dw, act_t, ids_t, wts_t, sdw, sact_t, sc_col, res_t, dst_t)) {
                // Width is pre-checked at graph-build time, so reaching here means a shape or
                // quant-type decline, which is a programming error.
                throw std::runtime_error("moe_down9: declined after the graph omitted the downs "
                                         "(width is pre-checked; this is a shape/type mismatch)");
            }
        });
        return o;
    }
    adapter::native(d, [=]() {
        kern::moe_merge_residual_gated(st, pe, pw, ps, pc, stride, off, pr, py, E, k, T);
    });
    return o;
}

MoeOut dense_ffn_residual_norm(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w,
                               const Tensor * next_norm, const Tensor * norm_buf, float eps, const Tensor * q8_buf) {
    Tensor dn;
    {
        Call c(d);
        ggml_context * g  = c.ctx();
        ggml_tensor *  gx = c.in(x, "ffn_in");
        // separate gate and up: at one column the backend fuses gate + up + SwiGLU into one MMVQ call
        ggml_tensor * up   = ggml_mul_mat(g, c.in(w.sh_up, "ffn_up"), gx);
        ggml_tensor * gate = ggml_mul_mat(g, c.in(w.sh_gate, "ffn_gate"), gx);
        dn = c.run(ggml_mul_mat(g, c.in(w.sh_down, "ffn_down"), ggml_swiglu_split(g, gate, up)));
    }
    MoeOut o;
    if (next_norm != nullptr && next_norm->valid()) {
        const AddNorm an = add_rmsnorm(d, dn, r, *next_norm, eps, norm_buf, q8_buf);
        o.y      = an.sum;
        o.normed = an.out;
        return o;
    }
    o.y = add(d, dn, r);
    return o;
}

MoeOut moe_residual_norm(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w,
                         const MoeParams & p, const Tensor & next_norm, const Tensor & norm_buf, float eps,
                         const Tensor * q8_buf, const Tensor * router_pre, bool router_cmp) {
    const int k = p.n_used;
    static const int fuse_moe = getenv("AI_FUSE_MOE") ? atoi(getenv("AI_FUSE_MOE")) : (k_parity ? 0 : 1);
    if (fuse_moe > 0 && k_topk_order && w.route_score.valid() && (w.route_score.type == F32 || w.route_score.type == F16) &&
        (fuse_moe == 1 || w.sh_gate_up.valid())) {
        return moe_residual_fused(d, x, r, w, p, fuse_moe == 2, &next_norm, &norm_buf, eps, q8_buf, router_pre,
                                  router_cmp);
    }
    MoeOut o;
    o.y = moe_residual(d, x, r, w, p);
    return o;
}

Tensor moe_residual(Device & d, const Tensor & x, const Tensor & r, const MoeWeights & w, const MoeParams & p) {
    const int64_t T = x.ne[1];
    const int64_t E = x.ne[0];
    const int     k = p.n_used;
    // AI_FUSE_MOE: 0 off; 1 router|score + gated merge (default); 2 also shared gate|up as one projection
    static const int fuse_moe = getenv("AI_FUSE_MOE") ? atoi(getenv("AI_FUSE_MOE")) : (k_parity ? 0 : 1);
    if (fuse_moe > 0 && k_topk_order && w.route_score.valid() && (w.route_score.type == F32 || w.route_score.type == F16) &&
        (fuse_moe == 1 || w.sh_gate_up.valid())) {
        return moe_residual_fused(d, x, r, w, p, fuse_moe == 2, nullptr, nullptr, 0.0f).y;
    }
    Tensor exps, weights, shared;
    {
        Call c(d);
        ggml_context * g = c.ctx();
        ggml_tensor * gx = c.in(x, "moe_in");

        ggml_tensor * logits = ggml_mul_mat(g, c.in(w.router, "router"), gx);
        ggml_tensor * probs  = ggml_soft_max(g, logits);
        ggml_tensor * sel    = ggml_argsort_top_k(g, probs, k);
        ggml_tensor * wts    = ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, p.n_expert, T), sel);
        wts = ggml_reshape_2d(g, wts, k, T);
        ggml_tensor * wsum = ggml_clamp(g, ggml_sum_rows(g, wts), 6.103515625e-5f, INFINITY);
        wts = ggml_reshape_3d(g, ggml_div(g, wts, wsum), 1, k, T);
        if (k_topk_order) {
            c.effect(wts);
        } else {
            c.late(wts);
        }

        ggml_tensor * cur  = ggml_reshape_3d(g, gx, E, 1, T);
        ggml_tensor * up   = ggml_mul_mat_id(g, c.in(w.up_exps, "up_exps"), cur, sel);
        ggml_tensor * gate = ggml_mul_mat_id(g, c.in(w.gate_exps, "gate_exps"), cur, sel);
        ggml_tensor * act  = ggml_swiglu_split(g, gate, up);
        ggml_tensor * down = ggml_mul_mat_id(g, c.in(w.down_exps, "down_exps"), act, sel);

        ggml_tensor * su  = ggml_mul_mat(g, c.in(w.sh_up, "up_shexp"), gx);
        ggml_tensor * sg  = ggml_mul_mat(g, c.in(w.sh_gate, "gate_shexp"), gx);
        ggml_tensor * sd  = ggml_mul_mat(g, c.in(w.sh_down, "down_shexp"), ggml_swiglu_split(g, sg, su));
        ggml_tensor * sgt = ggml_sigmoid(g, ggml_mul_mat(g, c.in(w.sh_score, "shexp_score"), gx));
        ggml_tensor * sh  = ggml_mul(g, sd, sgt);
        c.late(sh);

        exps    = c.run(down);
        weights = c.to_public(wts);
        shared  = c.to_public(sh);
        // Same router tap as the fused path, which AI_PARITY does not take (k_topk_order is false).
        if (getenv("AI_DUMP_DIR") && !recording()) {
            moe_dump(d, "ffn_moe_logits", c.to_public(logits));
            moe_dump(d, "ffn_moe_ids", c.to_public(sel));
            moe_dump(d, "ffn_moe_weights", weights);
        }
    }
    if (!exps.contiguous() || !weights.contiguous() || !shared.contiguous() || !r.contiguous()) {
        throw std::runtime_error("moe_residual: native merge needs contiguous inputs");
    }
    Tensor y = d.ws_alloc(F32, E, T);
    void * st = adapter::stream(d);
    const float * pe = (const float *) exps.data;
    const float * pw = (const float *) weights.data;
    const float * ps = (const float *) shared.data;
    const float * pr = (const float *) r.data;
    float *       py = (float *) y.data;
    adapter::native(d, [=]() { kern::moe_merge_residual(st, pe, pw, ps, pr, py, E, k, T); });
    return y;
}

MoeRoute moe_route(Device & d, const Tensor & x, const Tensor & router, const MoeParams & p) {
    Call c(d);
    ggml_context * g = c.ctx();
    const int64_t T = x.ne[1];
    const int     k = p.n_used;
    ggml_tensor * logits  = ggml_mul_mat(g, c.in(router, "router"), c.in(x, "moe_in"));
    ggml_tensor * probs   = ggml_soft_max(g, logits);
    ggml_tensor * sel     = ggml_argsort_top_k(g, probs, k);
    ggml_tensor * weights = ggml_get_rows(g, ggml_reshape_3d(g, probs, 1, p.n_expert, T), sel);
    weights = ggml_reshape_2d(g, weights, k, T);
    ggml_tensor * wsum = ggml_clamp(g, ggml_sum_rows(g, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(g, ggml_div(g, weights, wsum), 1, k, T);
    ggml_tensor * sel_c = ggml_cont(g, sel);   // host reads it: contiguous
    c.effect(sel_c);
    MoeRoute r;
    r.weights = c.run(weights);
    r.sel     = c.to_public(sel_c);
    return r;
}

bool moe_wsum_native() {
    static const bool v = !getenv("AI_MOE_WSUM") || atoi(getenv("AI_MOE_WSUM")) != 0;
    return v;
}

Tensor moe_experts_raw(Device & d, const Tensor & x, const Tensor & up, const Tensor & gate, const Tensor & down,
                       const Tensor & slots) {
    Call c(d);
    ggml_context * g = c.ctx();
    const int64_t T = x.ne[1];
    const int64_t E = x.ne[0];
    ggml_tensor * ids  = c.in(slots, "slots");
    ggml_tensor * cur  = ggml_reshape_3d(g, c.in(x, "moe_in"), E, 1, T);
    ggml_tensor * u    = ggml_mul_mat_id(g, c.in(up, "up_pool"), cur, ids);
    ggml_tensor * gt   = ggml_mul_mat_id(g, c.in(gate, "gate_pool"), cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(g, gt, u);
    return c.run(ggml_mul_mat_id(g, c.in(down, "down_pool"), act, ids));
}

Tensor moe_wsum(Device & d, const Tensor & exps, const Tensor & weights, int k, const Tensor * add, const Tensor * into) {
    const int64_t E = exps.ne[0], T = exps.ne[2];
    if (!moe_wsum_native() || !exps.contiguous() || !weights.contiguous() || weights.type != F32 ||
        (add && (!add->contiguous() || add->type != F32))) {
        Call c(d);
        ggml_context * g = c.ctx();
        ggml_tensor * ex  = ggml_mul(g, c.in(exps, "moe_exps"), c.in(weights, "moe_w"));
        ggml_tensor * moe = ggml_view_2d(g, ex, E, T, ex->nb[2], 0);
        for (int i = 1; i < k; ++i) {
            moe = ggml_add(g, moe, ggml_view_2d(g, ex, E, T, ex->nb[2], (size_t) i * ex->nb[1]));
        }
        if (add) moe = ggml_add(g, moe, c.in(*add, "moe_add"));
        Tensor out = c.run(moe);
        if (into) {
            copy_into(d, out, *into, "moe_wsum into");
            return *into;
        }
        return out;
    }
    Tensor out = into ? *into : d.ws_alloc(F32, E, T);
    if (out.type != F32 || !out.contiguous() || out.ne[0] != E || out.ne[1] != T) {
        throw std::runtime_error("moe_wsum: into must be f32 [E, T] contiguous");
    }
    const float * ep = (const float *) exps.data;
    const float * wp = (const float *) weights.data;
    const float * ap = add ? (const float *) add->data : nullptr;
    float *       yp = (float *) out.data;
    Device *      dv = &d;
    adapter::native(d, [=]() { kern::moe_weighted_sum(adapter::stream(*dv), ep, wp, ap, yp, E, k, T); });
    return out;
}

Tensor moe_experts(Device & d, const Tensor & x, const Tensor & up, const Tensor & gate, const Tensor & down,
                   const Tensor & slots, const Tensor & weights, int k) {
    if (moe_wsum_native() && weights.contiguous()) {
        return moe_wsum(d, moe_experts_raw(d, x, up, gate, down, slots), weights, k);
    }
    Call c(d);
    ggml_context * g = c.ctx();
    const int64_t T = x.ne[1];
    const int64_t E = x.ne[0];
    ggml_tensor * ids  = c.in(slots, "slots");
    ggml_tensor * cur  = ggml_reshape_3d(g, c.in(x, "moe_in"), E, 1, T);
    ggml_tensor * u    = ggml_mul_mat_id(g, c.in(up, "up_pool"), cur, ids);
    ggml_tensor * gt   = ggml_mul_mat_id(g, c.in(gate, "gate_pool"), cur, ids);
    ggml_tensor * act  = ggml_swiglu_split(g, gt, u);
    ggml_tensor * exps = ggml_mul_mat_id(g, c.in(down, "down_pool"), act, ids);
    exps = ggml_mul(g, exps, c.in(weights, "moe_w"));
    ggml_tensor * moe = ggml_view_2d(g, exps, E, T, exps->nb[2], 0);
    for (int i = 1; i < k; ++i) {
        moe = ggml_add(g, moe, ggml_view_2d(g, exps, E, T, exps->nb[2], (size_t) i * exps->nb[1]));
    }
    return c.run(moe);
}

Tensor moe_shared(Device & d, const Tensor & x, const MoeWeights & w) {
    Call c(d);
    ggml_context * g = c.ctx();
    ggml_tensor * gx  = c.in(x, "moe_in");
    ggml_tensor * su  = ggml_mul_mat(g, c.in(w.sh_up, "up_shexp"), gx);
    ggml_tensor * sg  = ggml_mul_mat(g, c.in(w.sh_gate, "gate_shexp"), gx);
    ggml_tensor * sd  = ggml_mul_mat(g, c.in(w.sh_down, "down_shexp"), ggml_swiglu_split(g, sg, su));
    ggml_tensor * sgt = ggml_sigmoid(g, ggml_mul_mat(g, c.in(w.sh_score, "shexp_score"), gx));
    return c.run(ggml_mul(g, sd, sgt));
}

Tensor mul(Device & d, const Tensor & a, const Tensor & b) {
    Call c(d);
    return c.run(ggml_mul(c.ctx(), c.in(a), c.in(b)));
}

}  // namespace ai::op
