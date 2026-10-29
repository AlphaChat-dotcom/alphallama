// SPDX-License-Identifier: Apache-2.0
// ai-kernel-diff -- per-kernel differential test: every native fused kernel against the ggml op
// chain it replaces, byte for byte, on the same inputs.
//
// Bit-identity of the native kernels depends on per-file compile flags and intrinsic choices that a
// build-system change could break silently; this tool checks it per kernel, in a loop.
//
// Rules:
//  1. Eager, never under capture. A cudaStreamSynchronize inside a capturing stream is illegal and
//     silently invalidates the comparison. This tool never opens a Recorder.
//  2. Sweep T = 1..16. Batch variance is the main bug class: a verify row and a decode token of the
//     same input must agree bit for bit, and a kernel that is exact at T=1 and wrong at T=3 is the
//     normal failure, not the exotic one.
//  3. Count invocations, not tokens. Every case reports how many times the changed path ran.
//  4. Compare bytes, not norms. An NMSE check passes on a kernel that is wrong in the last bit, and
//     the last bit is what flips a near-tie token.
//  5. A case that did not run must not look like a case that passed. Every case reports LIVE/DEAD
//     separately from PASS/FAIL, because a silently declining kernel and a genuine pass produce
//     the same zero-difference output.
//
//   ai-kernel-diff [-n reps] [-v] [case ...]      (no case names = run all)
#include "core/device.h"
#include "core/tensor.h"
#include "kernels/kernels.h"
#include "kernels/moe_down9.h"
#include "ops/adapter.h"

#include "ggml.h"

#include <cuda_runtime.h>

#include <sys/stat.h>
#include <ctime>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace ai;

// ------------------------------------------------------------------------------------------------
// Reporting
// ------------------------------------------------------------------------------------------------

// A kernel is gated one of two ways, and the harness has to know which, because a byte compare on
// an approximate kernel is 100% noise and a tolerance on an exact one hides real regressions.
//
//   ByteIdentical  -- zero difference, every byte. Every kernel defaults here.
//   ReductionOrder -- for a named short list of kernels whose only admitted difference is
//                     summation order. Passes on nmse < tol.
//
// A ReductionOrder pass is not exactness: such a kernel's output can feed the next layer's router,
// where a tiny difference can reroute an expert. The check that settles argmax stability is
// end-to-end token identity against the flag-off arm at temp 0; an nmse bound only bounds the input.
enum class Gate { ByteIdentical, ReductionOrder };

struct Result {
    bool        live = false;    // did the native kernel actually run?
    bool        pass = true;
    Gate        gate = Gate::ByteIdentical;
    double      nmse = 0.0;
    double      tol  = 0.0;
    long long   invocations = 0;
    long long   bytes_total = 0;
    long long   bytes_diff = 0;
    long long   first_off = -1;
    double      mine = 0, ref = 0;
    std::string note;
};

static int g_fail = 0, g_run = 0;
static bool g_verbose = false;

static void report(const char * name, int T, const Result & r) {
    ++g_run;
    const bool ok = r.live && r.pass;
    if (!ok) {
        ++g_fail;
    }
    printf("  %-28s T=%-3d %-4s %-4s  %6lld inv  ", name, T, r.live ? "LIVE" : "DEAD",
           ok ? "PASS" : "FAIL", r.invocations);
    if (!r.live) {
        printf("kernel did not run: %s\n", r.note.c_str());
        return;
    }
    if (r.gate == Gate::ReductionOrder) {
        printf("%s  nmse %.3g (tol %.3g) [reduction-order only, NOT byte-identical]%s%s\n",
               r.pass ? "within tolerance" : "OVER TOLERANCE",
               r.nmse, r.tol, r.note.empty() ? "" : "  ", r.note.c_str());
        return;
    }
    if (r.pass) {
        printf("%lld/%lld bytes equal\n", r.bytes_total, r.bytes_total);
    } else {
        printf("%lld/%lld bytes DIFFER, first at +%lld: mine %.9g vs ref %.9g%s%s\n", r.bytes_diff,
               r.bytes_total, r.first_off, r.mine, r.ref, r.note.empty() ? "" : "  ", r.note.c_str());
    }
}

// Byte-compare two host buffers interpreted as floats. Bytes, not values: a NaN payload difference
// or a -0.0/+0.0 difference is a real divergence in a bit-exactness test.
static void diff_f32(Result & r, const std::vector<float> & mine, const std::vector<float> & ref) {
    r.bytes_total = (long long) (mine.size() * sizeof(float));
    const unsigned char * a = (const unsigned char *) mine.data();
    const unsigned char * b = (const unsigned char *) ref.data();
    for (long long i = 0; i < r.bytes_total; ++i) {
        if (a[i] != b[i]) {
            ++r.bytes_diff;
            if (r.first_off < 0) {
                r.first_off = i;
                r.mine = mine[i / 4];
                r.ref  = ref[i / 4];
            }
        }
    }
    r.pass = r.bytes_diff == 0;
}

static void diff_raw(Result & r, const std::vector<unsigned char> & mine,
                     const std::vector<unsigned char> & ref) {
    r.bytes_total = (long long) mine.size();
    for (long long i = 0; i < r.bytes_total; ++i) {
        if (mine[i] != ref[i]) {
            ++r.bytes_diff;
            if (r.first_off < 0) {
                r.first_off = i;
                r.mine = mine[i];
                r.ref  = ref[i];
            }
        }
    }
    r.pass = r.bytes_diff == 0;
}

// ------------------------------------------------------------------------------------------------
// Input generation
// ------------------------------------------------------------------------------------------------
// Deterministic, and deliberately not uniform-benign. Bit-exactness bugs hide in the tails: values
// that straddle a rounding boundary, exact zeros (which take the amax == 0 branch in the quantiser),
// and mixed magnitudes that make summation order matter. A clean N(0,1) fill would pass kernels
// that a real activation distribution breaks.
static std::vector<float> make_input(size_t n, uint32_t seed) {
    std::mt19937                          rng(seed);
    std::normal_distribution<float>       g(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float>                    v(n);
    for (size_t i = 0; i < n; ++i) {
        const float p = u(rng);
        if (p < 0.02f) {
            v[i] = 0.0f;                          // exercises the amax == 0 path
        } else if (p < 0.06f) {
            v[i] = g(rng) * 1e-6f;                // denormal-adjacent
        } else if (p < 0.10f) {
            v[i] = g(rng) * 1e4f;                 // wide dynamic range -> order matters
        } else {
            v[i] = g(rng);
        }
    }
    return v;
}

// A realistic activation distribution, for kernels that are not bit-exact.
//
// make_input above is adversarial (10% of its values are 1e4-scale), which suits a byte compare but
// misleads for an approximate kernel: summing many terms with 1e4 outliers cancels catastrophically
// and turns a 1e-7 reduction-order difference into a large apparent relative error.
// Hostile inputs answer "is it exact", realistic inputs answer "how wrong is it".
static std::vector<float> make_input_benign(size_t n, uint32_t seed) {
    std::mt19937                    rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float>              v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = g(rng);
    }
    return v;
}

// ------------------------------------------------------------------------------------------------
// Harness plumbing
// ------------------------------------------------------------------------------------------------

struct Ctx {
    Device & d;
    int      reps;
};

static Tensor upload_new(Device & d, const std::vector<float> & host, int64_t n0, int64_t n1 = 1,
                         int64_t n2 = 1) {
    Tensor t = d.ws_alloc(F32, n0, n1, n2);
    d.upload(t, host.data(), 0, host.size() * sizeof(float));
    return t;
}

static std::vector<float> download_all(Device & d, const Tensor & t) {
    std::vector<float> h((size_t) t.numel());
    d.download(t, h.data(), 0, h.size() * sizeof(float));
    return h;
}

// ------------------------------------------------------------------------------------------------
// CASE: add_rmsnorm  --  r = mix + res ; p = rms_norm(r) * w
// Reference chain: ADD, then the fused RMS_NORM+MUL that ggml builds from RMS_NORM -> MUL.
// ------------------------------------------------------------------------------------------------
static void case_add_rmsnorm(Ctx & c, int T) {
    const int64_t E   = 2048;          // n_embd; the real geometry, so the q8 guards are exercised
    const float   eps = 1e-6f;
    Device &      d   = c.d;

    Device::Scope scope(d);

    auto h_mix = make_input((size_t) (E * T), 1001u + T);
    auto h_res = make_input((size_t) (E * T), 2002u + T);
    auto h_w   = make_input((size_t) E, 3003u);

    Tensor mix = upload_new(d, h_mix, E, T);
    Tensor res = upload_new(d, h_res, E, T);
    Tensor w   = upload_new(d, h_w, E);

    // --- reference: two ggml ops, exactly as the engine would issue them unfused
    Tensor r_ref, p_ref;
    {
        adapter::Call call(d);
        ggml_context * g  = call.ctx();
        ggml_tensor *  gm = call.in(mix, "mix");
        ggml_tensor *  gs = call.in(res, "res");
        ggml_tensor *  gw = call.in(w, "w");
        ggml_tensor *  rr = ggml_add(g, gm, gs);
        call.effect(rr);
        ggml_tensor * pp = ggml_mul(g, ggml_rms_norm(g, rr, eps), gw);
        p_ref = call.run(pp);
        r_ref = call.to_public(rr);
    }
    auto h_r_ref = download_all(d, r_ref);
    auto h_p_ref = download_all(d, p_ref);

    // --- ours: one native launch
    Tensor r_mine = d.ws_alloc(F32, E, T);
    Tensor p_mine = d.ws_alloc(F32, E, T);
    Result res_r;
    for (int i = 0; i < c.reps; ++i) {
        adapter::native(d, [&]() {
            kern::add_rmsnorm(adapter::stream(d), (const float *) mix.data, (const float *) res.data,
                              (const float *) w.data, (float *) r_mine.data, (float *) p_mine.data, nullptr,
                              E, T, eps);
        });
        ++res_r.invocations;
    }
    d.synchronize();
    res_r.live = res_r.invocations > 0;

    auto h_r_mine = download_all(d, r_mine);
    auto h_p_mine = download_all(d, p_mine);

    Result res_p = res_r;
    diff_f32(res_r, h_r_mine, h_r_ref);
    diff_f32(res_p, h_p_mine, h_p_ref);
    report("add_rmsnorm[r=mix+res]", T, res_r);
    report("add_rmsnorm[p=norm*w]", T, res_p);
}

// ------------------------------------------------------------------------------------------------
// CASE: add_rmsnorm q8_1 epilogue  --  the staged quantised copy vs quantize_row_q8_1_cuda
// The fold depends on -use_fast_math: exact division puts roundf() on a different integer in rare
// calls, which is invisible end-to-end until a near-tie token flips.
// ------------------------------------------------------------------------------------------------
static void case_add_rmsnorm_q8(Ctx & c, int T) {
    const int64_t E   = 2048;
    const float   eps = 1e-6f;
    Device &      d   = c.d;

    Device::Scope scope(d);

    auto h_mix = make_input((size_t) (E * T), 4004u + T);
    auto h_res = make_input((size_t) (E * T), 5005u + T);
    auto h_w   = make_input((size_t) E, 6006u);

    Tensor mix = upload_new(d, h_mix, E, T);
    Tensor res = upload_new(d, h_res, E, T);
    Tensor w   = upload_new(d, h_w, E);

    Tensor r_t = d.ws_alloc(F32, E, T);
    Tensor p_t = d.ws_alloc(F32, E, T);

    // block_q8_1 is 36 bytes per 32 columns.
    const int64_t nblk  = (E / 32) * T;
    int           bufid = 0;
    void *        q_mine = d.ws_alloc_raw((size_t) nblk * 36, &bufid);
    void *        q_ref  = d.ws_alloc_raw((size_t) nblk * 36, &bufid);

    Result r;
    adapter::native(d, [&]() {
        kern::add_rmsnorm(adapter::stream(d), (const float *) mix.data, (const float *) res.data,
                          (const float *) w.data, (float *) r_t.data, (float *) p_t.data, q_mine, E, T, eps);
    });
    ++r.invocations;
    // The reference quantiser, over the same p the fused kernel just produced.
    adapter::native(d, [&]() {
        kern::q8_stage_fill_reference(adapter::stream(d), (const float *) p_t.data, q_ref, E, T);
    });
    d.synchronize();

    std::vector<unsigned char> h_mine((size_t) nblk * 36), h_ref((size_t) nblk * 36);
    cudaMemcpy(h_mine.data(), q_mine, h_mine.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), q_ref, h_ref.size(), cudaMemcpyDeviceToHost);

    // Liveness: the kernel declines to stage unless the geometry is safe (see ex::q8_geometry_ok).
    // A declined stage leaves the buffer untouched; even if that compares equal it is not a pass.
    bool all_zero = true;
    for (unsigned char b : h_mine) {
        if (b != 0) {
            all_zero = false;
            break;
        }
    }
    r.live = !all_zero;
    if (!r.live) {
        r.note = "staging declined (geometry guard) or wrote nothing";
        report("add_rmsnorm[q8_1 epilogue]", T, r);
        return;
    }
    diff_raw(r, h_mine, h_ref);
    report("add_rmsnorm[q8_1 epilogue]", T, r);
}

// ------------------------------------------------------------------------------------------------
// CASE: moe_merge_residual_gated
//   y = res + (sum_j exps[.., j, ..] * w[j]) + shared_down * sigmoid(score)
// Reference chain: MUL, k-1 ADDs, SIGMOID, MUL, ADD(shared), ADD(residual) -- the ~10 elementwise
// launches per layer this kernel replaces.
// ------------------------------------------------------------------------------------------------
static void case_moe_merge(Ctx & c, int T) {
    const int64_t E = 2048, k = 8, NE = 256;
    Device &      d = c.d;

    Device::Scope scope(d);

    auto h_exps   = make_input((size_t) (E * k * T), 7007u + T);
    auto h_w      = make_input((size_t) (k * T), 8008u + T);
    auto h_shared = make_input((size_t) (E * T), 9009u + T);
    auto h_res    = make_input((size_t) (E * T), 1010u + T);
    auto h_score  = make_input((size_t) ((NE + 1) * T), 1111u + T);

    Tensor exps   = upload_new(d, h_exps, E, k, T);
    Tensor wts    = upload_new(d, h_w, 1, k, T);
    Tensor shared = upload_new(d, h_shared, E, T);
    Tensor resid  = upload_new(d, h_res, E, T);
    Tensor score  = upload_new(d, h_score, NE + 1, T);

    // --- reference, built from the elementwise ops the kernel replaces, in the same order
    Tensor y_ref;
    {
        adapter::Call call(d);
        ggml_context * g  = call.ctx();
        ggml_tensor *  ge = call.in(exps, "exps");
        ggml_tensor *  gw = call.in(wts, "w");
        ggml_tensor *  gs = call.in(shared, "shared");
        ggml_tensor *  gr = call.in(resid, "res");
        ggml_tensor *  gc = call.in(score, "score");

        // sum over experts in rank order j = 0..k-1, which is what the kernel does
        ggml_tensor * acc = nullptr;
        for (int64_t j = 0; j < k; ++j) {
            ggml_tensor * ej = ggml_view_2d(g, ge, E, T, ge->nb[2], (size_t) j * ge->nb[1]);
            ggml_tensor * wj = ggml_view_2d(g, gw, 1, T, gw->nb[2], (size_t) j * gw->nb[1]);
            ggml_tensor * t  = ggml_mul(g, ggml_cont(g, ej), wj);
            acc              = acc == nullptr ? t : ggml_add(g, acc, t);
        }
        // shared gate: sigmoid of column NE of the router|score row
        ggml_tensor * col = ggml_view_2d(g, gc, 1, T, gc->nb[1], (size_t) NE * sizeof(float));
        ggml_tensor * sh  = ggml_mul(g, gs, ggml_sigmoid(g, ggml_cont(g, col)));
        ggml_tensor * out = ggml_add(g, ggml_add(g, acc, sh), gr);
        y_ref             = call.run(out);
    }
    auto h_y_ref = download_all(d, y_ref);

    Tensor y_mine = d.ws_alloc(F32, E, T);
    Result r;
    for (int i = 0; i < c.reps; ++i) {
        adapter::native(d, [&]() {
            kern::moe_merge_residual_gated(adapter::stream(d), (const float *) exps.data,
                                           (const float *) wts.data, (const float *) shared.data,
                                           (const float *) score.data, NE + 1, NE, (const float *) resid.data,
                                           (float *) y_mine.data, E, k, T);
        });
        ++r.invocations;
    }
    d.synchronize();
    r.live = r.invocations > 0;

    auto h_y_mine = download_all(d, y_mine);
    diff_f32(r, h_y_mine, h_y_ref);
    report("moe_merge_residual_gated", T, r);
}

// ------------------------------------------------------------------------------------------------
// CASE: argmax  --  greedy token selection, ties included
// Reference: ggml_argmax. The tie rule is part of the contract: comparisons are strictly >, and
// partials combine in ascending index order, so the lowest index wins among equal values.
// ------------------------------------------------------------------------------------------------
static void case_argmax(Ctx & c, int T) {
    const int64_t V = 248320;          // the real vocabulary; ggml's kernel uses one block per row
    Device &      d = c.d;

    Device::Scope scope(d);

    auto h_x = make_input((size_t) (V * T), 1212u + T);
    // Force ties: without them the tie-break rule is never exercised, and the tie-break is exactly
    // what a "reasonable looking" reimplementation gets wrong.
    for (int64_t t = 0; t < T; ++t) {
        const float peak = 12.5f;
        h_x[(size_t) (t * V + 100)]    = peak;
        h_x[(size_t) (t * V + 50000)]  = peak;   // same value, higher index -> must not win
        h_x[(size_t) (t * V + 200000)] = peak;
    }
    Tensor x = upload_new(d, h_x, V, T);

    Tensor ref_t;
    {
        adapter::Call call(d);
        ggml_context * g = call.ctx();
        ref_t            = call.run(ggml_argmax(g, call.in(x, "x")));
    }
    std::vector<int32_t> h_ref((size_t) T);
    d.download(ref_t, h_ref.data(), 0, h_ref.size() * sizeof(int32_t));

    const int np      = kern::argmax_partials_per_row();
    Tensor    pmax    = d.ws_alloc(F32, np, T);
    Tensor    pidx    = d.ws_alloc(I32, np, T);
    Tensor    dst     = d.ws_alloc(I32, T);
    Result    r;
    for (int i = 0; i < c.reps; ++i) {
        adapter::native(d, [&]() {
            kern::argmax_rows(adapter::stream(d), (const float *) x.data, V, T, (float *) pmax.data,
                              (int32_t *) pidx.data, (int32_t *) dst.data);
        });
        ++r.invocations;
    }
    d.synchronize();
    r.live = r.invocations > 0;

    std::vector<int32_t> h_mine((size_t) T);
    d.download(dst, h_mine.data(), 0, h_mine.size() * sizeof(int32_t));

    r.bytes_total = (long long) (h_mine.size() * sizeof(int32_t));
    for (size_t i = 0; i < h_mine.size(); ++i) {
        if (h_mine[i] != h_ref[i]) {
            r.bytes_diff += 4;
            if (r.first_off < 0) {
                r.first_off = (long long) (i * 4);
                r.mine      = h_mine[i];
                r.ref       = h_ref[i];
            }
        }
    }
    r.pass = r.bytes_diff == 0;
    if (r.pass) {
        r.note = "ties present, lowest index won";
    }
    report("argmax_rows", T, r);
}


// ------------------------------------------------------------------------------------------------
// CASE: moe_down9  --  the nine-way fused down+merge (AI_MOE_DOWN9)
//
// Replaces, per layer: quantise(act) + MUL_MAT_ID(down) + quantise(shared act) + MUL_MAT(shared
// down) + the native merge. The reference is the flag-off path: the two ggml matmuls plus
// kern::moe_merge_residual_gated, itself byte-verified against its own ggml chain for T = 1..16.
//
// Two things this checks:
//  1. Whether down9 is bit-exact or only close. With the default warp reduction it reduces each
//     row's K across one warp where MMVQ uses four; AI_D9_EXACT reproduces the reference split.
//  2. The score stride. `score` is a [1, T] tensor that need not be contiguous: the engine passes
//     column NE of the fused router|score row, ne[0] == 1 with a row pitch of NE+1. Both layouts
//     run below, and the strided one is the one the engine uses.
//
// n_expert is 64 rather than the model's 256 purely to keep the weight allocation modest: the
// expert count only selects which rows are read, while bit-exactness lives in the reduction over
// n_ff. The geometry that matters -- n_ff 512, n_embd 2048, k 8, routed Q5_K + shared Q8_0 -- is
// the model's own.
// ------------------------------------------------------------------------------------------------
static void case_down9_impl(Ctx & c, int T, bool strided_score) {
    const int64_t n_ff = 512, E = 2048, NEXP = 64, k = 8, NE = 256;
    Device &      d    = c.d;

    Device::Scope scope(d);

    // Quantised weights built by ggml's own quantiser from f32, so they are structurally valid.
    // Random bytes would not be: K-quants carry scale blocks that must decode sanely.
    auto h_dw_f32 = make_input_benign((size_t) (n_ff * E * NEXP), 2121u);
    auto h_sw_f32 = make_input_benign((size_t) (n_ff * E), 2222u);

    Tensor dw = d.ws_alloc(Q5_K, n_ff, E, NEXP);
    Tensor sw = d.ws_alloc(Q8_0, n_ff, E);
    {
        const size_t      bytes = (size_t) ggml_row_size(GGML_TYPE_Q5_K, n_ff) * E * NEXP;
        std::vector<char> q(bytes);
        ggml_quantize_chunk(GGML_TYPE_Q5_K, h_dw_f32.data(), q.data(), 0, E * NEXP, n_ff, nullptr);
        d.upload(dw, q.data(), 0, bytes);
    }
    {
        const size_t      bytes = (size_t) ggml_row_size(GGML_TYPE_Q8_0, n_ff) * E;
        std::vector<char> q(bytes);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h_sw_f32.data(), q.data(), 0, E, n_ff, nullptr);
        d.upload(sw, q.data(), 0, bytes);
    }

    auto h_act  = make_input_benign((size_t) (n_ff * k * T), 3131u + T);
    auto h_sact = make_input_benign((size_t) (n_ff * T), 3232u + T);
    auto h_wts  = make_input_benign((size_t) (k * T), 3333u + T);
    auto h_res  = make_input_benign((size_t) (E * T), 3434u + T);

    Tensor act   = upload_new(d, h_act, n_ff, k, T);
    Tensor sact  = upload_new(d, h_sact, n_ff, T);
    Tensor wts   = upload_new(d, h_wts, 1, k, T);
    Tensor resid = upload_new(d, h_res, E, T);

    // ids distinct within a row: ggml's MUL_MAT_ID MMQ path faults on repeated ids in a row, so
    // duplicates would be testing an unsupported input rather than the kernel.
    std::vector<int32_t> h_ids((size_t) (k * T));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            h_ids[(size_t) (t * k + j)] = (int32_t) ((t * 13 + j * 7) % NEXP);
        }
    }
    Tensor ids = d.ws_alloc(I32, k, T);
    d.upload(ids, h_ids.data(), 0, h_ids.size() * sizeof(int32_t));

    // score: a clean contiguous [1, T], or column NE of a [NE+1, T] row -- the engine's real layout.
    Tensor score;
    int64_t sc_stride = 1;
    if (strided_score) {
        auto   h_row = make_input_benign((size_t) ((NE + 1) * T), 3535u + T);
        Tensor row   = upload_new(d, h_row, NE + 1, T);
        score        = row;
        score.ne[0]  = 1;
        score.ne[1]  = T;
        score.data   = (char *) row.data + (size_t) NE * sizeof(float);
        sc_stride    = NE + 1;
    } else {
        auto h_sc = make_input_benign((size_t) T, 3636u + T);
        score     = upload_new(d, h_sc, 1, T);
    }

    // --- reference: the flag-off path, op for op
    Tensor y_ref = d.ws_alloc(F32, E, T);
    {
        Tensor down_pub, sd_pub;
        {
            adapter::Call  call(d);
            ggml_context * g   = call.ctx();
            ggml_tensor *  gsd = ggml_mul_mat(g, call.in(sw, "sh_down"), call.in(sact, "sact"));
            call.effect(gsd);
            ggml_tensor * gd = ggml_mul_mat_id(g, call.in(dw, "down_exps"), call.in(act, "act"),
                                               call.in(ids, "ids"));
            down_pub = call.run(gd);
            sd_pub   = call.to_public(gsd);
        }
        void *        st = adapter::stream(d);
        const float * pe = (const float *) down_pub.data;
        const float * pw = (const float *) wts.data;
        const float * ps = (const float *) sd_pub.data;
        const float * pc = (const float *) score.data;
        const float * pr = (const float *) resid.data;
        float *       py = (float *) y_ref.data;
        const int64_t ss = sc_stride;
        adapter::native(d, [=]() {
            kern::moe_merge_residual_gated(st, pe, pw, ps, pc, ss, 0, pr, py, E, k, T);
        });
        d.synchronize();
    }
    auto h_y_ref = download_all(d, y_ref);

    // --- ours: one fused kernel
    Tensor y_mine = d.ws_alloc(F32, E, T);
    Result r;
    bool   launched = false;
    adapter::native(d, [&]() {
        launched = kern::moe_down9(d.backend(), dw, act, ids, wts, sw, sact, score, resid, y_mine);
    });
    d.synchronize();
    r.invocations = 1;
    r.live        = launched;
    if (!r.live) {
        r.note = "declined -- see the [moe-down9] DECLINED geometry dump on stderr";
        report(strided_score ? "moe_down9[strided score]" : "moe_down9[contig score]", T, r);
        return;
    }

    auto h_y_mine = download_all(d, y_mine);
    diff_f32(r, h_y_mine, h_y_ref);

    // AI_D9_EXACT: d9_row_dot reproduces the reference's four-way K split inside a single warp. When
    // the exact split is active and the reference is still MMVQ, this case is a byte compare;
    // falling back to the tolerance would let it pass without testing exactness.
    const bool d9_exact_split = [] {
        const char * e = getenv("AI_D9_EXACT");
        return !(e && e[0] == '0');
    }();
    const bool byte_gated = d9_exact_split && T <= 8;
    r.gate = byte_gated ? Gate::ByteIdentical : Gate::ReductionOrder;
    {
        double sse = 0.0, ssr = 0.0;
        for (size_t i = 0; i < h_y_mine.size(); ++i) {
            const double e = (double) h_y_mine[i] - (double) h_y_ref[i];
            sse += e * e;
            ssr += (double) h_y_ref[i] * (double) h_y_ref[i];
        }
        r.nmse = ssr > 0 ? sse / ssr : 0.0;
    }

    // Above T = 8 the reference is a different path: MMVQ_MAX_BATCH_SIZE is 8, so ggml_mul_mat_id
    // switches from MMVQ to MMQ (int8 tensor cores, ~1e-4 nmse on its own). Prefill runs at the chunk
    // width, so T > 8 is a real path; under AI_D9_EXACT down9 declines above T=8 (ggml-cuda/moe-down9.cu), and
    // these rows still guard that runtime decision.
    const bool tier_change = T > 8;
    r.tol  = tier_change ? 1e-3 : 1e-12;
    if (byte_gated) {
        // r.pass already carries the byte verdict from diff_f32; do not relax it to the nmse test.
        // nmse stays populated so a near miss is reported with its magnitude rather than a bare
        // "differs" -- 3.5e-16 and 1e-3 are very different failures and the note must say which.
        if (!r.pass) {
            char b2[220];
            snprintf(b2, sizeof(b2),
                     "exact split active and still differs: %lld/%lld bytes, nmse %.3g. Either the "
                     "nwarps table is wrong for this T (see d9_nwarps_for) or the residual order is "
                     "not the reference's -- the K-split is no longer the explanation.",
                     r.bytes_diff, r.bytes_total, r.nmse);
            r.note = b2;
        }
    } else {
        r.pass = r.nmse < r.tol;
    }
    if (tier_change) {
        r.note = "T>8: reference switched MMVQ->MMQ, not apples-to-apples; outside our operating range";
    } else if (!r.pass && !byte_gated) {
        // Without the exact split down9 is not bit-identical (one warp reduces each row's K where
        // MMVQ uses four), so report the magnitude of the difference.
        // Scale the error against the tensor's magnitude, not each element's. A per-element
        // relative error divides by a denominator that can be ~0, so one near-zero output makes an
        // otherwise excellent kernel look catastrophic -- the metric then reports the input
        // distribution rather than the kernel.
        double maxabs = 0.0, refmax = 0.0, sse = 0.0, ssr = 0.0;
        for (size_t i = 0; i < h_y_mine.size(); ++i) {
            const double a = h_y_mine[i], b = h_y_ref[i], e = fabs(a - b);
            maxabs = e > maxabs ? e : maxabs;
            refmax = fabs(b) > refmax ? fabs(b) : refmax;
            sse += e * e;
            ssr += b * b;
        }
        char buf[200];
        snprintf(buf, sizeof(buf), "max|err| %.3g vs max|ref| %.3g (%.2g of scale)", maxabs, refmax,
                 refmax > 0 ? maxabs / refmax : 0.0);
        r.note = buf;
    }
    report(strided_score ? "moe_down9[strided score]" : "moe_down9[contig score]", T, r);
}

static void case_down9(Ctx & c, int T) {
    case_down9_impl(c, T, false);
    case_down9_impl(c, T, true);
}


// ------------------------------------------------------------------------------------------------
// CASE: batch invariance  --  does row 0 compute the same thing alone as it does in a T-row pass?
//
// Every case above compares fused vs unfused at the same T, which is blind to a kernel whose
// arithmetic depends on how many rows share the pass: both arms are equally affected.
//
// Here the two arms are the same kernel at different widths, comparing the one row they share.
// Anything that keys a decision off T -- a warp count, a tile width, a dispatch threshold, a
// batch-wide scale -- shows up here and nowhere else.
//
// Known instances of this class:
//   ggml_cuda_should_use_mmvf   F32 takes mmvf at ne11<=3 and cuBLAS at >=4        (closed)
//   calc_nwarps                 4 warps at ncols_dst 1..4, 2 at 5..8, 1 at 9+      (open at T>=5)
//   flash attention             tile geometry chosen from Q->ne[1]                 (pinned)
//   MUL_MAT_ID                  per-token-warp K-reduction vs the 1-token kernel   (pinned)
//
// Sweep to the operating range plus one. Decode is T=1 and an MTP verify is k+1, so k<=3 means T<=4.
// This sweeps to T=5 so the known calc_nwarps break is recorded rather than sitting just outside
// the swept range.
// ------------------------------------------------------------------------------------------------
static void case_invariance(Ctx & c, int T) {
    const int64_t E   = 2048;
    const float   eps = 1e-6f;
    Device &      d   = c.d;

    Device::Scope scope(d);

    // Row 0 is shared; rows 1..T-1 are filler and are never compared.
    auto h_mix = make_input((size_t) (E * T), 7777u);
    auto h_res = make_input((size_t) (E * T), 8888u);
    auto h_w   = make_input((size_t) E, 9999u);

    Tensor mix_T = upload_new(d, h_mix, E, T);
    Tensor res_T = upload_new(d, h_res, E, T);
    Tensor w     = upload_new(d, h_w, E);

    // The same row 0, alone.
    std::vector<float> h_mix1(h_mix.begin(), h_mix.begin() + (size_t) E);
    std::vector<float> h_res1(h_res.begin(), h_res.begin() + (size_t) E);
    Tensor mix_1 = upload_new(d, h_mix1, E, 1);
    Tensor res_1 = upload_new(d, h_res1, E, 1);

    Tensor r_T = d.ws_alloc(F32, E, T), p_T = d.ws_alloc(F32, E, T);
    Tensor r_1 = d.ws_alloc(F32, E, 1), p_1 = d.ws_alloc(F32, E, 1);

    Result r;
    adapter::native(d, [&]() {
        kern::add_rmsnorm(adapter::stream(d), (const float *) mix_T.data, (const float *) res_T.data,
                          (const float *) w.data, (float *) r_T.data, (float *) p_T.data, nullptr, E, T, eps);
        kern::add_rmsnorm(adapter::stream(d), (const float *) mix_1.data, (const float *) res_1.data,
                          (const float *) w.data, (float *) r_1.data, (float *) p_1.data, nullptr, E, 1, eps);
    });
    d.synchronize();
    r.invocations = 2;
    r.live        = true;

    auto hp_T = download_all(d, p_T);
    auto hp_1 = download_all(d, p_1);
    hp_T.resize((size_t) E);   // row 0 only: the rest is filler, not the same content
    diff_f32(r, hp_T, hp_1);
    if (r.pass) {
        r.note = "row 0 identical alone and in a T-row pass";
    } else {
        r.note = "ROW 0 DEPENDS ON BATCH WIDTH -- a decision is keyed off T";
    }
    report("add_rmsnorm[T=1 vs T=k]", T, r);

    // Positive control: the check above proves nothing unless the same machinery can also fail. Rows
    // 0 and 1 hold different content by construction and must differ; if they compare equal, the
    // comparison is broken (wrong offset, empty buffer, stale download) and every PASS above is void.
    if (T >= 2) {
        Result ctl;
        ctl.invocations = 0;
        ctl.live        = true;
        std::vector<float> row1(hp_T.size());
        auto full = download_all(d, p_T);
        for (size_t i = 0; i < row1.size(); ++i) {
            row1[i] = full[(size_t) E + i];
        }
        diff_f32(ctl, hp_T, row1);
        const bool differ = !ctl.pass;   // diff_f32 sets pass when equal; the control needs differ
        // Printed directly rather than through report(), which would render this inverted check as
        // a green "bytes equal" line.
        ++g_run;
        if (!differ) {
            ++g_fail;
        }
        printf("  %-28s T=%-3d LIVE %-4s  %6d inv  %s\n", "  [control] row0 vs row1", T,
               differ ? "PASS" : "FAIL", 0,
               differ ? "rows differ, as they must -- comparator is live"
                      : "ROWS COMPARED EQUAL: comparator is broken, ignore every PASS above");
    }
}


// down9 invariance: also confirms the fused path is taken at every width in the operating range with
// no width-gated fallback, since a fusion admitted at some widths and not others breaks batch
// invariance. If the kernel declines at any width, `live` is false and the row says DEAD rather than
// quietly comparing nothing.
static void case_down9_invariance(Ctx & c, int T) {
    const int64_t n_ff = 512, E = 2048, NEXP = 64, k = 8;
    Device &      d    = c.d;

    Device::Scope scope(d);

    auto h_dw_f32 = make_input_benign((size_t) (n_ff * E * NEXP), 4141u);
    auto h_sw_f32 = make_input_benign((size_t) (n_ff * E), 4242u);
    Tensor dw = d.ws_alloc(Q5_K, n_ff, E, NEXP);
    Tensor sw = d.ws_alloc(Q8_0, n_ff, E);
    {
        const size_t b = (size_t) ggml_row_size(GGML_TYPE_Q5_K, n_ff) * E * NEXP;
        std::vector<char> q(b);
        ggml_quantize_chunk(GGML_TYPE_Q5_K, h_dw_f32.data(), q.data(), 0, E * NEXP, n_ff, nullptr);
        d.upload(dw, q.data(), 0, b);
    }
    {
        const size_t b = (size_t) ggml_row_size(GGML_TYPE_Q8_0, n_ff) * E;
        std::vector<char> q(b);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h_sw_f32.data(), q.data(), 0, E, n_ff, nullptr);
        d.upload(sw, q.data(), 0, b);
    }

    // Token 0 occupies the front of every contiguous input, so the T=1 arm is a prefix slice of the
    // T-row arm and row 0 is the same content in both by construction.
    auto h_act  = make_input_benign((size_t) (n_ff * k * T), 4343u);
    auto h_sact = make_input_benign((size_t) (n_ff * T), 4444u);
    auto h_wts  = make_input_benign((size_t) (k * T), 4545u);
    auto h_res  = make_input_benign((size_t) (E * T), 4646u);
    auto h_sc   = make_input_benign((size_t) T, 4747u);

    std::vector<int32_t> h_ids((size_t) (k * T));
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t j = 0; j < k; ++j) {
            h_ids[(size_t) (t * k + j)] = (int32_t) ((t * 13 + j * 7) % NEXP);
        }
    }

    auto slice = [](const std::vector<float> & v, size_t n) {
        return std::vector<float>(v.begin(), v.begin() + n);
    };

    Tensor actT = upload_new(d, h_act, n_ff, k, T), act1 = upload_new(d, slice(h_act, (size_t) (n_ff * k)), n_ff, k, 1);
    Tensor sacT = upload_new(d, h_sact, n_ff, T),   sac1 = upload_new(d, slice(h_sact, (size_t) n_ff), n_ff, 1);
    Tensor wtsT = upload_new(d, h_wts, 1, k, T),    wts1 = upload_new(d, slice(h_wts, (size_t) k), 1, k, 1);
    Tensor resT = upload_new(d, h_res, E, T),       res1 = upload_new(d, slice(h_res, (size_t) E), E, 1);
    Tensor scT  = upload_new(d, h_sc, 1, T),        sc1  = upload_new(d, slice(h_sc, 1), 1, 1);

    Tensor idsT = d.ws_alloc(I32, k, T);
    d.upload(idsT, h_ids.data(), 0, h_ids.size() * sizeof(int32_t));
    Tensor ids1 = d.ws_alloc(I32, k, 1);
    d.upload(ids1, h_ids.data(), 0, (size_t) k * sizeof(int32_t));

    Tensor yT = d.ws_alloc(F32, E, T), y1 = d.ws_alloc(F32, E, 1);

    Result r;
    bool   okT = false, ok1 = false;
    adapter::native(d, [&]() {
        okT = kern::moe_down9(d.backend(), dw, actT, idsT, wtsT, sw, sacT, scT, resT, yT);
        ok1 = kern::moe_down9(d.backend(), dw, act1, ids1, wts1, sw, sac1, sc1, res1, y1);
    });
    d.synchronize();
    r.invocations = 2;
    r.live        = okT && ok1;
    if (!r.live) {
        r.note = okT ? "DECLINED at T=1" : "DECLINED at this width -- width-gated fallback: the kernel does not cover this batch width";
        report("moe_down9[T=1 vs T=k]", T, r);
        return;
    }

    auto hT = download_all(d, yT);
    auto h1 = download_all(d, y1);
    hT.resize((size_t) E);
    diff_f32(r, hT, h1);
    r.note = r.pass ? "row 0 identical alone and in a T-row pass; fused path live at both widths"
                    : "ROW 0 DEPENDS ON BATCH WIDTH";
    report("moe_down9[T=1 vs T=k]", T, r);
}


// ------------------------------------------------------------------------------------------------
// CASE: q8 staging invariance
//
// Producer-side q8_1 emission computes a staged copy once and reuses it in later consumers, which
// can hide two invariance failures that no same-T test sees:
//
//   1. A batch-keyed scale. If any amax or scale is taken over the whole staged batch rather than
//      per row, a row's quantised bytes depend on which other rows shared the pass, so T=1 decode
//      and a width-k verify row diverge at last-bit level and the expert GEMVs amplify it.
//   2. A cross-row or cross-width cache hit. The registry is keyed on an address plus geometry.
//      A hit that crosses rows, or that survives a width change, serves the wrong row's bytes while
//      every shape check passes.
//
// Invariant: scales are per row; no cross-row state; any reuse is keyed on (row identity, shape) and
// never on the batch. Row 0's staged q8_1 block bytes, emitted alone, must equal row 0's staged bytes
// emitted inside a T-row pass. Bytes, not values: a scale one ULP out changes an int8 code.
// ------------------------------------------------------------------------------------------------
static void case_q8_stage_invariance(Ctx & c, int T) {
    const int64_t E   = 2048;
    const float   eps = 1e-6f;
    Device &      d   = c.d;

    Device::Scope scope(d);

    auto h_mix = make_input((size_t) (E * T), 5151u);
    auto h_res = make_input((size_t) (E * T), 5252u);
    auto h_w   = make_input((size_t) E, 5353u);

    Tensor mixT = upload_new(d, h_mix, E, T);
    Tensor resT = upload_new(d, h_res, E, T);
    Tensor w    = upload_new(d, h_w, E);
    Tensor mix1 = upload_new(d, std::vector<float>(h_mix.begin(), h_mix.begin() + (size_t) E), E, 1);
    Tensor res1 = upload_new(d, std::vector<float>(h_res.begin(), h_res.begin() + (size_t) E), E, 1);

    Tensor rT = d.ws_alloc(F32, E, T), pT = d.ws_alloc(F32, E, T);
    Tensor r1 = d.ws_alloc(F32, E, 1), p1 = d.ws_alloc(F32, E, 1);

    const int64_t blocks_row = E / 32;
    int    buf = 0;
    void * qT  = d.ws_alloc_raw((size_t) blocks_row * T * 36, &buf);
    void * q1  = d.ws_alloc_raw((size_t) blocks_row * 36, &buf);

    Result r;
    adapter::native(d, [&]() {
        kern::add_rmsnorm(adapter::stream(d), (const float *) mixT.data, (const float *) resT.data,
                          (const float *) w.data, (float *) rT.data, (float *) pT.data, qT, E, T, eps);
        kern::add_rmsnorm(adapter::stream(d), (const float *) mix1.data, (const float *) res1.data,
                          (const float *) w.data, (float *) r1.data, (float *) p1.data, q1, E, 1, eps);
    });
    d.synchronize();
    r.invocations = 2;

    std::vector<unsigned char> hT((size_t) blocks_row * 36), h1((size_t) blocks_row * 36);
    cudaMemcpy(hT.data(), qT, hT.size(), cudaMemcpyDeviceToHost);   // row 0 is the front of the buffer
    cudaMemcpy(h1.data(), q1, h1.size(), cudaMemcpyDeviceToHost);

    // Liveness: the producer declines to stage unless the geometry makes the copy bit-identical, and
    // a declined stage leaves the buffer untouched. All-zero on both sides would compare equal and
    // read as a pass while nothing was tested.
    bool anyT = false, any1 = false;
    for (size_t i = 0; i < hT.size(); ++i) {
        anyT = anyT || hT[i] != 0;
        any1 = any1 || h1[i] != 0;
    }
    r.live = anyT && any1;
    if (!r.live) {
        r.note = "staging declined at one or both widths -- geometry guard, nothing was compared";
        report("q8_stage[T=1 vs T=k]", T, r);
        return;
    }

    diff_raw(r, hT, h1);
    r.note = r.pass ? "row 0's staged bytes identical alone and in a T-row pass"
                    : "STAGED BYTES DEPEND ON BATCH WIDTH -- batch-keyed scale or cross-row reuse";
    report("q8_stage[T=1 vs T=k]", T, r);
}

// ------------------------------------------------------------------------------------------------

struct CaseDef {
    const char * name;
    void (*fn)(Ctx &, int);
    int max_T;
};

static const CaseDef kCases[] = {
    { "add_rmsnorm",     case_add_rmsnorm,     16 },
    { "add_rmsnorm_q8",  case_add_rmsnorm_q8,  16 },
    { "moe_merge",       case_moe_merge,       16 },
    { "argmax",          case_argmax,           4 },   // V=248320 rows; 4 is plenty and keeps it quick
    { "down9",           case_down9,           16 },
    { "invariance",      case_invariance,       5 },   // operating range (T<=4) plus one
    { "down9_inv",       case_down9_invariance, 16 },  // no width-gated fallback, also past T=8.
                                                       // Reference-free (down9 vs itself), so the MMQ
                                                       // step at 9 that bounds the diff case does not
                                                       // bound this one.
    { "q8_stage_inv",    case_q8_stage_invariance, 8 },
};

// Provenance: which libggml-cuda is loaded. Half of the cases compare a kernel that lives in ggml's
// shared library, not in this binary, and a stale library makes a fix look like it does not work.
// Print the path and build time so a stale one is visible before any number is read.
static void print_ggml_provenance() {
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) {
        printf("ggml: (cannot read /proc/self/maps)\n");
        return;
    }
    char line[4096], found[4096] = {0};
    while (fgets(line, sizeof(line), f)) {
        const char * p = strstr(line, "libggml-cuda");
        if (p && strstr(line, " r-xp ") != nullptr) {
            const char * slash = strchr(line, '/');
            if (slash) {
                snprintf(found, sizeof(found), "%s", slash);
                char * nl = strchr(found, '\n');
                if (nl) {
                    *nl = 0;
                }
                break;
            }
        }
    }
    fclose(f);
    if (!found[0]) {
        printf("ggml: libggml-cuda not found in the address space\n");
        return;
    }
    struct stat st;
    if (stat(found, &st) == 0) {
        char when[64];
        strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", localtime(&st.st_mtime));
        printf("ggml: %s\n      built %s  (%.1f MB)\n", found, when, st.st_size / 1048576.0);
        printf("      NOTE: a library older than the ggml sources does not contain a ggml kernel change:\n"
               "      rebuild ggml before believing the number.\n");
    } else {
        printf("ggml: %s (stat failed)\n", found);
    }
}

int main(int argc, char ** argv) {
    int                      reps = 1;
    std::vector<std::string> want;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (a == "-v") {
            g_verbose = true;
        } else if (a == "-h" || a == "--help") {
            printf("usage: ai-kernel-diff [-n reps] [-v] [case ...]\ncases:");
            for (const auto & cd : kCases) {
                printf(" %s", cd.name);
            }
            printf("\n");
            return 0;
        } else {
            want.push_back(a);
        }
    }

    // Rule 1: eager. No Recorder is ever opened in this file, so every download below is legal and
    // every comparison is of real post-launch memory.
    adapter::set_capture_mode(adapter::CaptureMode::None);

    Device d;
    if (!d.init(0)) {
        fprintf(stderr, "kernel_diff: no CUDA device\n");
        return 2;
    }
    // Generous: argmax at V=248320 x T=4 is ~4 MB per buffer, add_rmsnorm is small, moe_merge's
    // exps at [2048, 8, 16] is 1 MB. 512 MB leaves room for the reference graphs' intermediates.
    d.init_workspace((size_t) 512 << 20);
    d.finalize();

    Ctx ctx{ d, reps };
    printf("ai-kernel-diff -- native kernels vs the ggml chains they replace, byte for byte\n");
    printf("eager mode (no capture), reps=%d\n", reps);
    // The invariance case runs the same kernel in both arms, so it only shows that kernel is
    // width-invariant. Batch-invariance bugs are usually dispatch changes with the width (mmvf ->
    // cuBLAS at ne11 4, the 1-token MoE kernel -> mul_mat_vec_q_moe at 2, VEC -> MMA in flash
    // attention, calc_nwarps 4 warps -> 2 at ncols_dst 5), so a green run here is not end-to-end safety.
    printf("  scope: the invariance case proves THIS KERNEL is width-invariant.\n"
           "         Only scripts/invariance.sh proves THE ENGINE computes row j the same at every\n"
           "         width -- that is the one that catches dispatch changes. Keep both; never quote\n"
           "         the first for the second.\n");
    print_ggml_provenance();
    printf("\n");

    for (const auto & cd : kCases) {
        if (!want.empty()) {
            bool hit = false;
            for (const auto & wname : want) {
                if (wname == cd.name) {
                    hit = true;
                }
            }
            if (!hit) {
                continue;
            }
        }
        printf("%s:\n", cd.name);
        for (int T = 1; T <= cd.max_T; ++T) {
            // 1,2,3,4 then 8,16 -- the small T are where batch-variance bugs live, and the powers
            // of two are where tile geometry changes.
            // The invariance case wants every width in 1..max, densely: the thresholds it hunts
            // sit at specific integers (mmvf at 4, calc_nwarps at 5 and 9) and a sparse sweep walks
            // straight past them. Other cases keep the sparse sweep for runtime.
            const bool dense = strcmp(cd.name, "invariance") == 0 || strcmp(cd.name, "down9_inv") == 0 ||
                              strcmp(cd.name, "q8_stage_inv") == 0;
            if (!dense && T > 4 && T != 7 && T != 8 && T != 9 && T != 12 && T != 16) {
                continue;
            }
            cd.fn(ctx, T);
        }
        printf("\n");
    }

    printf("%d checks, %d failed\n", g_run, g_fail);
    return g_fail == 0 ? 0 : 1;
}
