// SPDX-License-Identifier: Apache-2.0
#include "engine/engine.h"
#include "engine/tf_dump.h"
#include "engine/lookup.h"

#include "kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
extern char ** environ;
#include <stdexcept>

namespace ai {

using clock_t_ = std::chrono::steady_clock;
static double ms_between(clock_t_::time_point a, clock_t_::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// K (--mtp) is the widest verify a round may take; the MTP chain itself drafts at most kMaxChain (AI_MTP_CHAIN lower).
// Only lookup drafts (AI_LOOKUP) use widths above the chain.
static constexpr int kMaxK     = 128;
static constexpr int kMaxChain = 8;
static int chain_cap(int K) {
    static const int c = getenv("AI_MTP_CHAIN") ? atoi(getenv("AI_MTP_CHAIN")) : 0;
    return std::min(K, c > 0 ? std::min(c, kMaxChain) : kMaxChain);
}

// Prompt-lookup drafting (AI_LOOKUP) counters; the core (config, buckets, n-gram index, proposal) is in engine/lookup.h
static long long g_lk_rounds = 0, g_lk_drafted = 0, g_lk_acc = 0, g_lk_all_rounds = 0;
// AI_MTP_GATE report counters: process-wide, printed by an atexit handler after the engine is gone.
static long long g_cold_rounds = 0;
static double    g_cold_ms     = 0.0;

static Tensor carve(Device & dev, int buf, size_t & off, int type, int64_t n) {
    Tensor t;
    t.type  = type;
    t.ne[0] = n;
    set_contiguous_strides(t);
    t.buf  = buf;
    t.data = (char *) dev.buffer_base(buf) + off;
    off += (dev.alloc_size(t) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
    return t;
}

static Tensor as_shape(const Tensor & buf, int type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    Tensor t = buf;
    t.type  = type;
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3;
    set_contiguous_strides(t);
    return t;
}

// ------------------------------------------------------------------ ingress

void Engine::Ingress::alloc(Device & dev, int64_t T, int64_t mask_elems, int64_t B, const char * name) {
    T_cap    = T;
    mask_cap = mask_elems;
    B_cap    = B;
    const size_t bytes = (size_t) (T * 4 + 4 * T * 4 + T * 8 + mask_elems * 2 + 4 * B * 4) + 16 * dev.alignment();
    const int buf = dev.alloc_buffer(bytes, name);
    size_t off = 0;
    ids   = carve(dev, buf, off, I32, T);
    pos   = carve(dev, buf, off, I32, 4 * T);
    cells = carve(dev, buf, off, I64, T);
    mask  = carve(dev, buf, off, F16, std::max<int64_t>(1, mask_elems));
    last  = carve(dev, buf, off, I32, B);
    base  = carve(dev, buf, off, I32, B);
    zeros = carve(dev, buf, off, I32, B);
    delta = carve(dev, buf, off, I32, B);
    h_ids   = (int32_t *)  dev.alloc_pinned((size_t) T * 4);
    h_pos   = (int32_t *)  dev.alloc_pinned((size_t) T * 16);
    h_cells = (int64_t *)  dev.alloc_pinned((size_t) T * 8);
    h_mask  = mask_elems > 0 ? (uint16_t *) dev.alloc_pinned((size_t) mask_elems * 2) : nullptr;
    h_last  = (int32_t *)  dev.alloc_pinned((size_t) B * 4);
    h_base  = (int32_t *)  dev.alloc_pinned((size_t) B * 4);
    h_zero  = (int32_t *)  dev.alloc_pinned((size_t) B * 4);
    h_delta = (int32_t *)  dev.alloc_pinned((size_t) B * 4);
    memset(h_delta, 0, (size_t) B * 4);
    memset(h_ids, 0, (size_t) T * 4);
    memset(h_zero, 0, (size_t) B * 4);
}

// AI_HOST_EMBD: a device q8_0 [E, T_cap] staging block, its pinned host mirror, and a constant
// iota id list. Only allocated when token_embd is host-kept.
void Engine::Ingress::alloc_embd(Device & dev, const Tensor & table, const char * name) {
    if (table.buf >= 0) {
        throw std::runtime_error("alloc_embd: token_embd is device-resident; staging is not needed");
    }
    const int64_t E   = table.ne[0];
    const int64_t blk = block_size(table.type);
    if (blk <= 0 || E % blk != 0) {
        throw std::runtime_error("alloc_embd: token_embd row width " + std::to_string(E) +
                                 " is not a multiple of the block size " + std::to_string(blk));
    }
    tbl_host = (const char *) table.data;
    tbl_rowb = (size_t) (E / blk) * type_size(table.type);
    tbl_rows = table.ne[1];
    tbl_cols = E;
    if (!tbl_host || tbl_rowb == 0 || tbl_rows <= 0) {
        throw std::runtime_error("alloc_embd: host-kept token_embd has no host storage");
    }
    const size_t bytes = tbl_rowb * (size_t) T_cap + (size_t) T_cap * 4 + 8 * dev.alignment();
    const int    buf   = dev.alloc_buffer(bytes, name);
    size_t       off   = 0;
    embd      = carve(dev, buf, off, table.type, E * T_cap);   // carve is 1-D; view() reshapes to [E,T]
    embd_iota = carve(dev, buf, off, I32, T_cap);
    h_embd    = (char *) dev.alloc_pinned(tbl_rowb * (size_t) T_cap);
    // The iota is constant, so it is uploaded once here rather than per round.
    std::vector<int32_t> iota((size_t) T_cap);
    for (int64_t i = 0; i < T_cap; ++i) {
        iota[(size_t) i] = (int32_t) i;
    }
    dev.upload(embd_iota, iota.data(), 0, (size_t) T_cap * 4);
    fprintf(stderr, "[ai] host-embd staging: %s %lld rows x %zu B = %.2f MiB pinned + iota (vocab %lld)\n",
            name, (long long) T_cap, tbl_rowb, (double) (tbl_rowb * (size_t) T_cap) / 1048576.0,
            (long long) tbl_rows);
}

Qwen35Hybrid::Inputs Engine::Ingress::view(int64_t L, int64_t B, int64_t n_kv, int first_region) const {
    const int64_t T = L * B;
    Qwen35Hybrid::Inputs in;
    in.T            = T;
    in.B            = B;
    in.n_kv         = n_kv;
    in.first_region = first_region;
    in.ids          = as_shape(ids, I32, T);
    in.pos4         = as_shape(pos, I32, 4 * T);
    in.cells        = as_shape(cells, I64, T);
    in.mask         = as_shape(mask, F16, n_kv, L, 1, B);
    in.last_id      = as_shape(last, I32, B);
    if (h_embd) {
        in.embd      = as_shape(embd, embd.type, tbl_cols, T);
        in.embd_iota = as_shape(embd_iota, I32, T);
    }
    return in;
}

Tensor Engine::Ingress::base_view(int64_t B) const { return as_shape(base, I32, B); }
Tensor Engine::Ingress::zeros_view(int64_t B) const { return as_shape(zeros, I32, B); }
Tensor Engine::Ingress::delta_view(int64_t B) const { return as_shape(delta, I32, B); }
// AI_HEAD_MASK_DEV (default 1): the round writes the MTP head's mask on device into the main ingress's mask buffer
// after the main forward has read it (same stream, in round order), so the head needs no second chunk x region f16
// mask and pinned mirror of its own. The mask is byte-identical to the uploaded one; 0 restores the upload.
bool Engine::head_mask_dev() {
    static const bool v = getenv("AI_HEAD_MASK_DEV") ? atoi(getenv("AI_HEAD_MASK_DEV")) != 0 : true;
    return v;
}

Tensor Engine::Ingress::mask_view(int64_t n_kv, int64_t L, int64_t B) const { return as_shape(mask, F16, n_kv, L, 1, B); }

void Engine::Ingress::upload(Device & dev, int64_t L, int64_t B, int64_t n_kv) const {
    const int64_t T = L * B;
    if (T > T_cap || (!mask_on_device && !mask_fits(n_kv, T)) || B > B_cap) {
        throw std::runtime_error("ingress: round larger than its buffers");
    }
    const auto v = view(L, B, n_kv, 0);
    dev.upload_async(v.ids, h_ids, 0, (size_t) T * 4);
    dev.upload_async(v.pos4, h_pos, 0, (size_t) T * 16);
    dev.upload_async(v.cells, h_cells, 0, (size_t) T * 8);
    // otherwise (AI_HEAD_MASK_DEV / AI_DECODE_MASK_DEV) the caller writes the mask on device in the round
    if (mask_fits(n_kv, T) && !mask_dev_now) {
        dev.upload_async(v.mask, h_mask, 0, (size_t) (n_kv * T * 2));
    }
    dev.upload_async(v.last_id, h_last, 0, (size_t) B * 4);
    dev.upload_async(base_view(B), h_base, 0, (size_t) B * 4);
    dev.upload_async(zeros_view(B), h_zero, 0, (size_t) B * 4);
    dev.upload_async(delta_view(B), h_delta, 0, (size_t) B * 4);
    if (h_embd) {
        // Pure memcpy: one q8_0 row per token out of the host-kept table. The ids are already here
        // in the pinned mirror, so this needs no device->host transfer.
        for (int64_t t = 0; t < T; ++t) {
            const int32_t id = h_ids[t];
            if (id < 0 || (int64_t) id >= tbl_rows) {
                throw std::runtime_error("host-embd: token id " + std::to_string(id) +
                                         " out of range for vocab " + std::to_string(tbl_rows));
            }
            memcpy(h_embd + (size_t) t * tbl_rowb, tbl_host + (size_t) id * tbl_rowb, tbl_rowb);
        }
        dev.upload_async(v.embd, h_embd, 0, tbl_rowb * (size_t) T);
    }
}

// ------------------------------------------------------------------ init

// Choose the KV cell type (and, with max_ctx <= 0, the context) from the VRAM left after the weights, the model
// and the decode mode.
//
// 1. Fit. Every byte alloc_state..finalize will take is computed before allocating, with the same rounding: KV
//    (Qwen35Hybrid::kv_cache_bytes), GDN state + staging (state_bytes), the derived workspace at the candidate
//    capacity (workspace_bytes_for, incl. AI_FA_Q partials), and the ingress mask(s). Plus a margin
//    (AI_KV_MARGIN_MB, default 512) for what is not modelled (DFlash2 buffers, cuBLAS/ggml pools). After init
//    the plan is printed against what was really allocated, so a wrong model shows up on every run.
// 2. Speed:
//    - AI_FA_Q active (sm_8x) and draft/verify rounds (MTP, DFlash2): q8_0 first. It is as fast as f16 at short
//      context and faster at long context, fits more cells, and its NLL is within noise of f16.
//    - plain decode: f16 first where it fits (1-2% faster).
//    - no AI_FA_Q (not sm_8x, or AI_FA_Q=0): f16 first; ggml's quantised attention is ~3x slower per cell, so a
//      quantised cache is only for fit.
//    q4_0 is only the last resort (+0.4% PPL); if even q4_0 does not fit the requested context, the context is
//    clamped to what fits, with a warning.
// AI_FORCE_VRAM_MB pretends the card has this much VRAM, so the decision can be dry-run for other card sizes.
static const char * kv_name(int t) { return t == F16 ? "f16" : t == Q8_0 ? "q8_0" : t == Q4_0 ? "q4_0" : "?"; }

bool Engine::plan_kv() {
    const bool auto_type = cfg_.kv_type == kKvAuto;
    const bool auto_ctx  = cfg_.max_ctx <= 0;
    // Type and context both given: nothing to choose, unless the workspace is still the derived estimate with a
    // calibration pending, which may not fit beside the requested cache; then the bootstrap below must run.
    if (!auto_type && !auto_ctx && !(ws_cal_pending_ && cfg_.ws_bytes == 0)) {
        return true;
    }
    const int64_t B = cfg_.max_batch, K = std::max(0, cfg_.mtp_k);
    size_t free_b = 0, total_b = 0;
    dev_.memory_info(&free_b, &total_b);
    bool forced = false;
    if (const char * fv = getenv("AI_FORCE_VRAM_MB")) {
        const long v = atol(fv);
        if (v > 0) {
            const size_t used = total_b - free_b, tot = (size_t) v << 20;
            free_b  = tot > used ? tot - used : 0;
            total_b = tot;
            forced  = true;
        }
    }
    const size_t  margin = (size_t) (getenv("AI_KV_MARGIN_MB") ? atol(getenv("AI_KV_MARGIN_MB")) : 512) << 20;
    const size_t  avail  = free_b > margin ? free_b - margin : 0;
    const size_t  al     = dev_.alignment();
    const int64_t T      = std::max(cfg_.chunk, B * (K + 1));
    auto need = [&](int type, int64_t ctx, bool with_ws = true) -> int64_t {
        const int64_t region = (ctx + PagedKv::kPage - 1) / PagedKv::kPage * PagedKv::kPage;
        const int64_t kv     = prog_.kv_cache_bytes(ctx, (int) B, type, al);
        const int64_t st     = prog_.state_bytes((int) B, 16, al);
        const int64_t ws     = !with_ws ? 0 : cfg_.ws_bytes ? (int64_t) cfg_.ws_bytes
                                             : (int64_t) prog_.workspace_bytes_for(cfg_.chunk, std::max<int64_t>(16, B * (K + 1)),
                                                                                   region * B, (int) B) * (K > 0 ? 2 : 1);
        const int64_t ing1   = T * 4 + 16 * T + 8 * T + 12 * B + 16 * (int64_t) al;
        const int64_t ing    = (ing1 + T * region * 2) +
                               (K > 0 ? ing1 + (head_mask_dev() ? std::min<int64_t>(T, B * (K + 1)) : T) * region * 2 : 0);
        return kv + st + ws + ing;
    };
    // AI_FA_Q must also cover this model's attention shape (kernels/fa_decode_q.cu: D 256, G <= 8), or a
    // quantised cache would run on ggml's slow path
    const auto & sp  = prog_.spec();
    const bool faq   = kern::fa_decode_q_enabled() && sp.head_dim == 256 && sp.n_head_kv > 0 &&
                       sp.n_head % sp.n_head_kv == 0 && sp.n_head / sp.n_head_kv <= 8;
    const bool draft = K > 0;
    std::vector<int> order;
    const char *     why;
    if (!auto_type) {
        order = { cfg_.kv_type };
        why   = "type given";
    } else if (faq && (draft || auto_ctx)) {
        order = { Q8_0, Q4_0 };
        why   = draft ? "draft/verify rounds with AI_FA_Q: q8_0 >= f16 at every measured depth, quality-neutral, "
                        "half the bytes"
                      : "largest context with AI_FA_Q: q8_0 holds 2x the cells of f16 at f16-like speed";
    } else {
        order = { F16, Q8_0, Q4_0 };
        why   = faq ? "plain decode: f16 is 1-2% faster where it fits"
                    : "no AI_FA_Q on this GPU: quantised attention takes ggml's path (2.6-3.1x slower per cell), so f16 "
                      "where it fits";
    }
    if (cfg_.verbose) {
        fprintf(stderr, "[ai] kv auto: VRAM free after weights %.2f GiB of %.2f%s, margin %zu MiB | %lld stream(s), "
                        "%lld row(s) per round, chunk %lld | AI_FA_Q %s\n",
                free_b / 1073741824.0, total_b / 1073741824.0, forced ? " (AI_FORCE_VRAM_MB)" : "", margin >> 20,
                (long long) B, (long long) (K + 1), (long long) cfg_.chunk, faq ? "on" : "off");
    }
    // largest context (multiple of 256) that fits with `type`, up to `cap`
    auto max_fit = [&](int type, int64_t cap) -> int64_t {
        int64_t lo = 0, hi = cap / 256;
        while (lo < hi) {
            const int64_t mid = (lo + hi + 1) / 2;
            if (need(type, mid * 256) <= (int64_t) avail) lo = mid; else hi = mid - 1;
        }
        return lo * 256;
    };
    int     type = -1;
    int64_t ctx  = cfg_.max_ctx;
    if (auto_ctx) {
        const int64_t cap = prog_.spec().n_ctx_train > 0 ? prog_.spec().n_ctx_train : 262144;
        for (int t : order) {
            const int64_t m = std::min(max_fit(t, cap), cap);
            if (cfg_.verbose) {
                fprintf(stderr, "[ai] kv auto:   %-4s largest context that fits %lld cells (cap %lld)\n", kv_name(t),
                        (long long) m, (long long) cap);
            }
            if (m >= 256) { type = t; ctx = m; break; }
        }
    } else {
        for (int t : order) {
            const int64_t n  = need(t, ctx);
            const bool    ok = n <= (int64_t) avail;
            if (cfg_.verbose) {
                fprintf(stderr, "[ai] kv auto:   %-4s at %lld cells needs %.2f GiB -> %s\n", kv_name(t), (long long) ctx,
                        n / 1073741824.0, ok ? "fits" : "does not fit");
            }
            if (ok) { type = t; break; }
        }
        // nothing fits the requested context: clamp it with the smallest cell type, unless the workspace is still the
        // derived estimate (several times the real peak) with a calibration pending. Then the estimate, not the cache,
        // is what does not fit, and the bootstrap below keeps the requested context.
        if (type < 0 && !(ws_cal_pending_ && cfg_.ws_bytes == 0)) {
            const int     t = order.back();
            const int64_t m = max_fit(t, ctx);
            if (m >= 256) {
                fprintf(stderr, "[ai] kv auto: WARNING: %lld cells do not fit even with %s; context CLAMPED to %lld\n",
                        (long long) ctx, kv_name(t), (long long) m);
                type = t;
                ctx  = m;
            }
        }
    }
    if (type < 0 && ws_cal_pending_ && cfg_.ws_bytes == 0) {
        // Bootstrap: the derived workspace (several times the real peak) is what does not fit, and there is no
        // measured peak yet. Keep the requested context (the calibration must sweep its envelope) and give the
        // workspace what remains, but never less than derived / 2.5, which is above every observed peak. If that
        // floor does not fit at the requested context, the context shrinks until it does. The run calibrates in
        // place; no restart.
        auto ws_derived = [&](int64_t c) -> int64_t {
            const int64_t region = (c + PagedKv::kPage - 1) / PagedKv::kPage * PagedKv::kPage;
            return (int64_t) prog_.workspace_bytes_for(cfg_.chunk, std::max<int64_t>(16, B * (K + 1)), region * B, (int) B) *
                   (K > 0 ? 2 : 1);
        };
        int64_t c = auto_ctx ? std::min<int64_t>(prog_.spec().n_ctx_train > 0 ? prog_.spec().n_ctx_train : 262144, 32768)
                             : ctx;
        // Try the requested context in every candidate type first, and only then shrink the context.
        // AI_WS_BOOT_RESERVE_MB (default 1024): the workspace below takes everything left after the cache, so the
        // backend's own pool (ggml's VMM pool, which grows during the calibration run) would have only the kv margin;
        // a CUDA OOM there aborts before the calibration is saved, so every later start would bootstrap and abort
        // again. Hold this much back during a bootstrap only; if the floor then does not fit, the context shrinks for
        // this start and the next start recalibrates at the full context. AI_WS_* is planning-only (not in the
        // calibration key). The reserve is adaptive: try it, then half of it, then none, and take the first that
        // holds any context, so a card with little room after the weights still starts.
        static const int64_t boot_reserve = (int64_t) (getenv("AI_WS_BOOT_RESERVE_MB") ? atol(getenv("AI_WS_BOOT_RESERVE_MB")) : 1024) << 20;
        const int64_t c_req = c;
        int64_t avail_boot = 0;
        int t = -1;
        for (const int64_t r : { boot_reserve, boot_reserve / 2, (int64_t) 0 }) {
            avail_boot = std::max<int64_t>((int64_t) avail - r, 0);
            auto fits = [&](int tt, int64_t cc) { return need(tt, cc, false) + ws_derived(cc) * 2 / 5 <= avail_boot; };
            c = c_req;
            t = -1;
            for (int tt : order) {
                if (fits(tt, c)) { t = tt; break; }
            }
            if (t < 0) {
                t = order.back();
                while (c >= 256 && !fits(t, c)) c = (c / 2) / 256 * 256;
            }
            if (c >= 256) {
                if (r != boot_reserve) {
                    fprintf(stderr, "[ai] ws auto: BOOTSTRAP reserve %lld MiB (the default %lld MiB holds no context here)\n",
                            (long long) (r >> 20), (long long) (boot_reserve >> 20));
                }
                break;
            }
        }
        if (c >= 256) {
            cfg_.ws_bytes = (size_t) (avail_boot - need(t, c, false)) / al * al;
            type          = t;
            if (c < ctx || auto_ctx) {
                fprintf(stderr, "[ai] ws auto: BOOTSTRAP context %lld cells%s\n", (long long) c,
                        auto_ctx ? " (auto; later starts size it from the measured workspace)" : " (CLAMPED for this first start)");
            }
            ctx           = c;
            ws_bootstrap_ = true;
            fprintf(stderr, "[ai] ws auto: BOOTSTRAP -- the derived workspace does not fit and this configuration has no "
                            "measurement yet: running on the remaining %.0f MiB of workspace and calibrating in place\n",
                    cfg_.ws_bytes / 1048576.0);
        }
    }
    if (type < 0) {
        fprintf(stderr, "[ai] kv auto: nothing fits: %.2f GiB free after the weights (margin %zu MiB) does not hold "
                        "256 cells of any KV type. Free VRAM, lower --chunk/--parallel, or offload experts.\n",
                free_b / 1073741824.0, margin >> 20);
        return false;
    }
    cfg_.kv_type   = type;
    cfg_.max_ctx   = ctx;
    kv_plan_bytes_ = need(type, ctx);
    if (cfg_.verbose) {
        fprintf(stderr, "[ai] kv auto: CHOSE %s, %lld cells per stream (%.2f GiB planned) -- %s%s\n", kv_name(type),
                (long long) ctx, kv_plan_bytes_ / 1073741824.0, why,
                type == Q4_0 ? " [q4_0: +0.4% PPL, chosen only because nothing finer fits]" : "");
    }
    return true;
}

// Measured workspace size. The derived estimate (Qwen35Hybrid::workspace_bytes) is several times the real peak and
// is the largest single item in the post-weights budget, so it, not the cache, would decide how much context fits.
// The real peak barely depends on the context; it scales with the shapes (chunk, streams, drafts), so it is a
// property of (model, binary, GPU, chunk, streams, drafts, offload, env switches), measurable in one short run at
// the real shapes. First start of a configuration: the derived size (safe) plus a calibration run -- prefill chunks
// and a few decode/verify rounds through the public generate_batch -- whose high-water mark is cached in
// $AI_WS_CACHE (default ~/.cache/alphainfer/ws_peaks.tsv). Later starts size the workspace at peak * 1.25 + 32 MiB
// (+ the analytic AI_FA_Q bound), and the KV planner gets the difference. Any change to the model file, the binary,
// the GPU, the shapes or an AI_*/ALPHA_*/GGML_* switch is a new key and recalibrates.
// AI_WS_AUTO=0 keeps the derived size; --ws-mib always wins. The overflow abort remains the guard.
static std::string ws_cache_path() {
    if (const char * p = getenv("AI_WS_CACHE")) return p;
    const char * home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/alphainfer/ws_peaks.tsv";
}

static std::string ws_file_id(const std::string & path) {
    struct stat st {};
    if (path.empty() || stat(path.c_str(), &st) != 0) return path + ":?";
    return path + ":" + std::to_string((long long) st.st_size) + ":" + std::to_string((long long) st.st_mtime);
}

static std::string ws_env_hash() {   // every switch that can change which code paths (and buffers) run
    std::vector<std::string> v;
    for (char ** e = environ; e && *e; ++e) {
        const std::string s = *e;
        const bool ours = s.rfind("AI_", 0) == 0 || s.rfind("ALPHA_", 0) == 0 || s.rfind("GGML_", 0) == 0;
        const bool planning_only = s.rfind("AI_WS_", 0) == 0 || s.rfind("AI_FORCE_VRAM_MB=", 0) == 0 ||
                                   s.rfind("AI_KV_MARGIN_MB=", 0) == 0;
        if (ours && !planning_only) v.push_back(s);
    }
    std::sort(v.begin(), v.end());
    uint64_t h = 1469598103934665603ULL;   // FNV-1a
    for (const auto & s : v) {
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
        h ^= 0xff; h *= 1099511628211ULL;
    }
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) h);
    return buf;
}

void Engine::ws_auto_lookup() {
    const char * a = getenv("AI_WS_AUTO");
    if (cfg_.ws_bytes != 0 || (a && atoi(a) == 0)) return;
    char exe[4096] = { 0 };
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) exe[n] = 0;
    cudaDeviceProp pr {};
    cudaGetDeviceProperties(&pr, cfg_.device);
    // key version: bump when the calibration procedure changes so old peaks are not reused
    ws_key_ = "v5|" + ws_file_id(cfg_.model_path) + "|" + ws_file_id(exe) + "|" + pr.name + "|sm" +
              std::to_string(pr.major * 10 + pr.minor) + "|chunk" + std::to_string(cfg_.chunk) + "|B" +
              std::to_string(cfg_.max_batch) + "|K" + std::to_string(std::max(0, cfg_.mtp_k)) + "|df2:" +
              ws_file_id(cfg_.dflash2_path) + "|slots" + std::to_string(cfg_.expert_slots) + "|eager" +
              std::to_string((int) cfg_.eager) + "|adapt" + std::to_string((int) cfg_.mtp_adapt) + "|env" + ws_env_hash();
    long long peak = -1;
    std::ifstream in(ws_cache_path());
    std::string line;
    while (std::getline(in, line)) {   // "key \t peak_bytes \t region_cells"; the last entry for the key wins
        const size_t t1 = line.find('\t');
        if (t1 == std::string::npos || line.compare(0, t1, ws_key_) != 0 || t1 != ws_key_.size()) continue;
        peak = atoll(line.c_str() + t1 + 1);
        const size_t t2 = line.find('\t', t1 + 1);
        ws_cal_region_ = t2 == std::string::npos ? 0 : atoll(line.c_str() + t2 + 1);
    }
    if (peak > 0) {
        const auto & sp = prog_.spec();
        const int64_t faq = kern::fa_decode_q_enabled() && sp.n_head_kv > 0
                                ? kern::fa_decode_q_ws_bound(sp.n_head, sp.n_head_kv, cfg_.max_batch) : 0;
        const size_t ws = (size_t) ((double) peak * 1.25) + (32ULL << 20) + (size_t) faq;
        const int64_t K = std::max(0, cfg_.mtp_k);
        const size_t derived = prog_.workspace_bytes_for(cfg_.chunk, std::max<int64_t>(16, cfg_.max_batch * (K + 1)),
                                                         cfg_.max_ctx > 0 ? cfg_.max_ctx * cfg_.max_batch : 0,
                                                         cfg_.max_batch) * (K > 0 ? 2 : 1);
        cfg_.ws_bytes = ws;
        if (cfg_.verbose) {
            fprintf(stderr, "[ai] ws auto: measured peak %.1f MiB for this configuration -> workspace %.0f MiB "
                            "(peak x1.25 + 32 MiB%s; derived estimate %.0f MiB)\n",
                    peak / 1048576.0, ws / 1048576.0, faq ? " + AI_FA_Q bound" : "", derived / 1048576.0);
        }
    } else {
        ws_cal_pending_ = true;
        if (cfg_.verbose) {
            fprintf(stderr, "[ai] ws auto: no measurement for this configuration yet -> derived workspace now; "
                            "calibrating once after init (cache %s)\n", ws_cache_path().c_str());
        }
    }
}

void Engine::ws_auto_calibrate() {
    // The peak is set in several places:
    //  (1) Prefill grows with the chunk index up to about the fourth chunk, so every stream prefills four full
    //      chunks (+3), then a few real rounds.
    //  (2) A decode / verify round's workspace is allocated when its graph is built and grows with the attention
    //      envelope F. A short calibration never reaches a large F, so every F the configured context can reach
    //      is run once below.
    //  (3) Prefill chunks at the deepest position (see below).
    // The x1.25 margin must not be what saves a run. Content does not matter (synthetic ids and real text give
    // identical peaks); ids are spread over the common vocabulary anyway.
    const int64_t K = std::max(0, cfg_.mtp_k);
    const int64_t n_cal = 3 * (K + 1) + 4;
    // capped to what generate() accepts; the cap only binds when 4 chunks exceed the context, where (1) cannot
    // reach chunk 4 in real use either
    const int64_t n_prompt = std::min<int64_t>(4 * cfg_.chunk + 3, prog_.kv().region_cells() - n_cal - 2 * K - 4);
    std::vector<std::vector<int32_t>> prompts((size_t) cfg_.max_batch);
    uint32_t x = 12345;
    for (auto & p : prompts) {
        for (int64_t i = 0; i < n_prompt; ++i) {
            x = x * 1664525u + 1013904223u;
            p.push_back((int32_t) (1000 + (x >> 8) % 60000));
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    // The engine must come back exactly as fresh: callers (the C API's "decode before prefill is EINVAL")
    // read "no stream open" as "no prefill yet", and the adaptive-k estimates must not carry calibration
    // tokens into real traffic. Covered by tools/api-tests/ai_api_decode_test.c in auto mode.
    const int  saved_active = active_batch_;
    const int  saved_live   = live_streams_;
    const int  saved_k_in   = k_in_;
    const auto saved_acc    = acc_;
    Timing tm;
    ws_calibrating_ = true;
    generate_batch(prompts, n_cal, [](int, int32_t) { return true; }, &tm);
    ws_calibrating_ = false;
    // (2) build one round at every envelope F up to the region, with the round's real width (k_in = k_out = K)
    {
        const int64_t B = active_batch_, L = K + 1, T = L * B;
        const bool    mtp = K > 0;
        const int64_t R = prog_.kv().region_cells();
        std::vector<int32_t> zeros((size_t) L, 0);
        for (int64_t f = 256;; f *= 2) {
            const int64_t F = std::min(f, R);
            for (int b = 0; b < B; ++b) {
                stage_rows(main_in_, b, b * L, T, zeros.data(), L, 0, F, false);
                if (mtp && !df2_) stage_rows(head_in_, b, b * L, T, nullptr, L, 1, F, true);
            }
            main_in_.upload(dev_, L, B, F);
            const auto vin = main_in_.view(L, B, F, 0);
            Qwen35Hybrid::Inputs hin;
            if (mtp && !df2_) {
                head_in_.upload(dev_, L, B, F);
                hin = head_in_.view(L, B, F, 0);
            }
            kern::q8_stage_clear();
            dev_.ws_reset(0);
            {
                // built and launched: building allocates most of the workspace, but a kernel that falls back at
                // launch (e.g. to ggml's attention at a deep envelope) takes scratch only then, so a record-only
                // pass under-measures. Launching writes only the calibration streams, which are wiped below.
                op::Round r(dev_);
                r.begin();
                if (df2_) build_df2(vin);
                else if (mtp) build_mtp(vin, hin, (int) K);
                else build_plain(vin);
                r.end();
                r.launch();
                dev_.synchronize();
            }
            dev_.ws_reset(0);
            if (f >= R) break;
        }
        dev_.synchronize();
    }
    // (3) Prefill chunks at the deepest position the region allows, launched (ggml's prefill attention takes part
    // of its scratch at launch). A chunk attends over visible_cells(off, T), so its workspace grows with depth, and
    // (1) only prefills the first four chunks. The last chunk of a prompt can be any width 1..chunk (prompt length
    // mod chunk), and ggml's attention switches kernels with the query count: a short deep last chunk takes a
    // split-KV path whose scratch grows with the KV length. So: one full non-last chunk, then a last chunk at every
    // width where a kernel choice can flip (powers of two and their neighbours). Writes only the calibration stream,
    // which is wiped below. Skipped for eager chunks (expert offload / --eager), which launch as they build.
    if (!chunk_eager()) {
        const int64_t R   = prog_.kv().region_cells();
        const int64_t C   = std::min<int64_t>(cfg_.chunk, R);
        const bool    mtp = K > 0 && !df2_;
        std::vector<int64_t> widths;
        for (int64_t w = 1; w <= C; w *= 2) {
            for (int64_t v : { w - 1, w, w + 1 }) {
                if (v >= 1 && v <= C && (widths.empty() || widths.back() < v)) widths.push_back(v);
            }
        }
        if (widths.back() != C) widths.push_back(C);
        std::vector<int32_t> ids((size_t) C + 1, 1000);
        auto deep_chunk = [&](int64_t T, bool last) {
            const int64_t off = std::max<int64_t>(0, R - T - 2 * K - 4);   // the deepest start a real chunk can have
            kern::q8_stage_clear();
            dev_.ws_reset(0);
            dev_.ws_tag(last ? "calibration: deepest LAST prefill chunk" : "calibration: deepest prefill chunk");
            auto in = stage_chunk(0, ids.data(), T, off, prog_.visible_cells(off, T));
            op::Round r(dev_);
            r.begin();
            Tensor hidden = prog_.forward(dev_, in, mtp || last);
            if (mtp && !last) {
                const int64_t nkv = prog_.visible_cells(off + 1, T);
                stage_rows(head_in_, 0, 0, T, ids.data() + 1, T, off + 1, nkv, true);
                head_in_.upload(dev_, T, 1, nkv);
                auto hin = head_in_.view(T, 1, nkv, 0);
                if (!head_in_.mask_fits(nkv, T)) {   // AI_HEAD_MASK_DEV, as prefill_mtp does
                    hin.mask = main_in_.mask_view(nkv, T, 1);
                    op::head_causal_mask(dev_, hin.mask, T, nkv, off + 1);
                }
                prog_.mtp(dev_, hin.ids, hidden, hin, Qwen35Hybrid::HeadRows::None);
            } else if (last) {
                const int64_t Eh = hidden.ne[0], Th = hidden.ne[1];
                const Tensor  lr = hidden.view(Eh, 1, 1, 1, hidden.nb[1], hidden.nb[1], hidden.nb[1],
                                               hidden.nb[1] * (size_t) (Th - 1));
                op::argmax(dev_, prog_.logits(dev_, lr));
            }
            r.end();
            r.launch();
            dev_.synchronize();
            dev_.ws_reset(0);
        };
        const size_t before = dev_.ws_high_water();
        deep_chunk(C, false);
        for (int64_t T : widths) deep_chunk(T, true);
        if (cfg_.verbose) {
            fprintf(stderr, "[ai] ws auto: deepest prefill chunks (%zu last-chunk widths up to %lld, at ~%lld cells): peak "
                            "%.1f -> %.1f MiB\n", widths.size(), (long long) C, (long long) R, before / 1048576.0,
                    dev_.ws_high_water() / 1048576.0);
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const size_t peak = dev_.ws_high_water();
    reset_streams((int) cfg_.max_batch);   // zero every slot's KV length and recurrent state ...
    prog_.kv().free_all();                 // ... then close them, as a fresh engine has
    streams_.clear();
    active_batch_ = saved_active;
    live_streams_ = saved_live;
    k_in_         = saved_k_in;
    acc_          = saved_acc;
    const std::string path = ws_cache_path();
    const size_t slash = path.rfind('/');
    if (slash != std::string::npos) {
        const std::string dir = path.substr(0, slash);
        mkdir(dir.substr(0, dir.rfind('/')).c_str(), 0755);
        mkdir(dir.c_str(), 0755);
    }
    std::ofstream out(path, std::ios::app);
    out << ws_key_ << '\t' << peak << '\t' << prog_.kv().region_cells() << '\n';
    if (cfg_.verbose) {
        fprintf(stderr, "[ai] ws auto: calibrated in %.2f s: peak %.1f MiB (in \"%s\") of a %.0f MiB workspace -> %s; the "
                        "next start of this configuration uses %.0f MiB\n",
                sec, peak / 1048576.0, dev_.ws_high_tag(), dev_.ws_capacity() / 1048576.0, out ? "saved" : "NOT SAVED (cache unwritable)",
                (peak * 1.25 + (32 << 20)) / 1048576.0);
    }
}

bool Engine::init(const EngineConfig & cfg) {
    cfg_ = cfg;
    // AI_CAPTURE = own (default) | backend | none
    const char * cap = getenv("AI_CAPTURE");
    const std::string capture = cap ? cap : "own";
    op::set_capture(capture == "backend" ? op::Capture::Backend
                    : capture == "none"  ? op::Capture::None
                                         : op::Capture::Own);
    if (cfg_.mtp_k > kMaxK) {
        fprintf(stderr, "[ai] --mtp: at most %d drafts\n", kMaxK);
        return false;
    }
    cfg_.max_batch = std::max(1, cfg_.max_batch);
    // AI_CHUNK: prefill rows per forward, for callers that cannot set cfg.chunk (the C API). Larger chunks prefill
    // faster but the workspace grows about linearly with the chunk.
    if (const char * e = getenv("AI_CHUNK")) {
        if (atoll(e) > 0) cfg_.chunk = atoll(e);
    }
    // q8_1 staging (producers emit the q8_1 copy of an activation so its consumers skip their quantise) is off by
    // default: on this tree's ggml a staged q8_1 is not bit-identical to the consumer's own quantise, and T = 1 and
    // verify rounds stage different sets, so exact and plain decode would diverge. Off costs nothing measurable at one
    // stream. ALPHA_Q8_STAGE=1 opts back in.
    if (cfg_.expert_slots == 0) setenv("ALPHA_Q8_STAGE", "0", 0);
    // The registry's entries live for a round, not a backend pass (a round is many passes, with producers and
    // consumers in different ones; MTP rounds stage across passes). Set here, not only in the app, so every entry
    // point, including the C API, gets it.
    setenv("ALPHA_Q8_STAGE_HOST_LIFETIME", "1", 0);
    if (!dev_.init(cfg.device)) {
        return false;
    }
    if (cfg_.expert_slots > 0) {
        // Offload: prefill chunks page experts in groups and run eagerly; decode rounds record,
        // with the per-layer routing read as a host step between captured spans.
        setenv("GGML_CUDA_REGISTER_HOST", "1", 0);   // host banks registered for async copies
    }
    if (!cfg_.dflash2_path.empty()) {
        // DFlash2 drafts from the target's taps: load it first so its tap layers can switch the taps on
        // (Qwen35Hybrid reads AI_TAPS at load). Batch 1, fixed k, no MTP head involved.
        if (cfg_.max_batch != 1 || cfg_.mtp_adapt || cfg_.expert_slots > 0) {
            fprintf(stderr, "[ai] --dflash2: batch 1, fixed k, resident experts only\n");
            return false;
        }
        df2_ = std::make_unique<DFlash2>();
        if (!df2_->load(dev_, cfg_.dflash2_path, cfg.verbose)) {
            return false;
        }
        std::string layers;
        for (int il : df2_->target_layers()) {
            layers += (layers.empty() ? "" : ",") + std::to_string(il);
        }
        setenv("AI_TAPS", layers.c_str(), 1);
        // With AI_LOOKUP on, a larger --mtp keeps its meaning as the widest verify (lookup drafts use it);
        // the DFlash2 block itself stays block_size() - 1 drafts (decode_round caps k_out there)
        // (the CLI forbids --dflash2 with --mtp, so AI_LOOKUP_K=<widest verify> sets it there)
        const int want = getenv("AI_LOOKUP_K") ? atoi(getenv("AI_LOOKUP_K")) : cfg_.mtp_k;
        cfg_.mtp_k = df2_->block_size() - 1;
        if (lookup_cfg().on && want > cfg_.mtp_k) cfg_.mtp_k = std::min(want, kMaxK);
    }
    if (!prog_.load(dev_, cfg.model_path, cfg.verbose, cfg_.expert_slots > 0)) {
        return false;
    }
    if (df2_ && (!prog_.taps().valid() || prog_.taps().ne[0] != df2_->tap_width())) {
        fprintf(stderr, "[ai] --dflash2: the target did not set up taps of width %lld\n",
                (long long) df2_->tap_width());
        return false;
    }
    if (cfg_.mtp_k > 0 && !df2_ && !prog_.has_mtp()) {
        fprintf(stderr, "[ai] --mtp: model has no next-n block\n");
        return false;
    }
    ws_auto_lookup();   // sets cfg_.ws_bytes from a measured peak, or marks a calibration run
    if (!plan_kv()) {
        return false;
    }
    if (!ws_key_.empty() && !ws_cal_pending_) {
        // a round's workspace grows with its envelope (built per F up to the region), so a peak measured up to a
        // smaller region does not cover this one: recalibrate (the x1.25 margin covers the run meanwhile).
        // Compared by power-of-two bucket: the envelope ladder is powers of two and growth within one bucket is
        // negligible, so context-auto's few-percent VRAM noise does not recalibrate every start.
        const int64_t region = (cfg_.max_ctx + PagedKv::kPage - 1) / PagedKv::kPage * PagedKv::kPage;
        auto pow2 = [](int64_t v) { int64_t p = 256; while (p < v) p *= 2; return p; };
        if (pow2(region) > pow2(ws_cal_region_)) {
            ws_cal_pending_ = true;
            if (cfg_.verbose) {
                fprintf(stderr, "[ai] ws auto: measured up to %lld cells, this run reaches %lld -> recalibrating\n",
                        (long long) ws_cal_region_, (long long) region);
            }
        }
    }
    const int64_t B = cfg_.max_batch;
    const int64_t K = std::max(0, cfg_.mtp_k);
    kv_alloc_before_ = dev_.total_allocated();
    prog_.alloc_state(dev_, cfg_.max_ctx, (int) B, cfg_.kv_type);
    // the MTP chain's widest round stays unscoped; wider (lookup) rounds run scoped with a record arena
    prog_.set_unscoped_rows(B * (std::min<int64_t>(K, chain_cap((int) K)) + 1));
    if (cfg_.expert_slots > 0) {
        int slots = cfg_.expert_slots;
        if (cfg_.expert_slots_auto) {
            // the expert slots get what is left after everything else this engine will allocate:
            // the workspace (the measured or derived size init_workspace will take), the ingress masks (main + head),
            // and a margin for CUDA graphs, cuBLAS and fragmentation (AI_DOOR_MARGIN_MIB, default 768)
            const int64_t Ra      = prog_.kv().region_cells();
            const int64_t Ta      = std::max(cfg_.chunk, B * (K + 1));
            const size_t  ws_est  = cfg_.ws_bytes ? cfg_.ws_bytes
                                                  : prog_.workspace_bytes(cfg_.chunk, std::max<int64_t>(16, B * (K + 1))) *
                                                        (K > 0 ? 2 : 1);
            const size_t  ingress = ((size_t) (2 * Ta * Ra) + (size_t) Ta * 64) * (K > 0 ? 2 : 1);
            const size_t  margin  = (size_t) (getenv("AI_DOOR_MARGIN_MIB") ? atoi(getenv("AI_DOOR_MARGIN_MIB")) : 768) << 20;
            size_t freeb = 0, totb = 0;
            dev_.memory_info(&freeb, &totb);
            const size_t reserve = ws_est + ingress + margin;
            const size_t budget  = freeb > reserve ? freeb - reserve : 0;
            slots = prog_.doorbell_slots_for(budget, cfg_.mtp_k > 0);
            if (slots < 0) {
                fprintf(stderr, "[ai] --expert-slots auto needs the doorbell offload (AI_OFFLOAD=doorbell, AI_DOORBELL build)\n");
                return false;
            }
            fprintf(stderr, "[ai] expert slots auto: VRAM free %.2f GiB - workspace %.0f MiB - ingress %.0f MiB - margin %.0f MiB "
                            "= %.2f GiB for experts -> %d slots per layer\n",
                    freeb / 1073741824.0, ws_est / 1048576.0, ingress / 1048576.0, margin / 1048576.0,
                    budget / 1073741824.0, slots);
            cfg_.expert_slots = std::max(1, slots);   // stays > 0: the offload flag for the rest of the engine
        }
        // the doorbell's captured-round width: B streams x (widest verify + 1); with prompt lookup the verify is the
        // full mtp_k (lookup drafts use it), otherwise the MTP chain (chain_cap); AI_DOOR_TMAX still overrides
        const int64_t widest = lookup_cfg().on ? K : std::min<int64_t>(K, chain_cap((int) K));
        prog_.alloc_expert_pool(dev_, slots, cfg_.mtp_k > 0, B * (widest + 1));
    }
    const int64_t R = prog_.kv().region_cells();

    // ingress: sized for the largest round (a prefill chunk, or B streams x (k+1) verify rows)
    const int64_t T = std::max(cfg_.chunk, B * (K + 1));
    // AI_HOST_EMBD: token_embd may be host-kept to recover VRAM (it is only ever gathered, one row
    // per token). Two configurations cannot be served, and both refuse here rather than silently
    // falling back to a different configuration.
    // a mapped host table is read by the Ops directly (Qwen35Hybrid::embd_mapped): no ingress staging, MTP allowed
    const bool host_embd = prog_.weights().is_host("token_embd.weight") && !prog_.embd_mapped();
    if (host_embd) {
        if (K > 0) {
            fprintf(stderr, "[ai] host-kept token_embd is not compatible with MTP: the draft head "
                            "embeds device-produced ids (the verify targets), and gathering those on "
                            "the host would need a mid-round sync that breaks the round's capture\n");
            return false;
        }
        if (!prog_.has_separate_output()) {
            fprintf(stderr, "[ai] host-kept token_embd is not compatible with tied embeddings: with no "
                            "output.weight the LM head aliases the table, and the head is a full GEMV "
                            "rather than a gather\n");
            return false;
        }
    }
    main_in_.alloc(dev_, T, T * R, B, "ingress");
    if (host_embd) {
        main_in_.alloc_embd(dev_, prog_.tok_embd(), "ingress.embd");
    }
    h_out_ = (int32_t *) dev_.alloc_pinned((size_t) (B + 16) * 4);
    if (K > 0) {
        head_in_.mask_on_device = head_mask_dev();
        head_in_.alloc(dev_, T, (head_in_.mask_on_device ? std::min<int64_t>(T, B * (K + 1)) : T) * R, B, "ingress.head");
        h_g_      = (int32_t *) dev_.alloc_pinned((size_t) (B * (K + 1)) * 4);
        h_m_      = (int32_t *) dev_.alloc_pinned((size_t) B * 4);
        h_drafts_ = (int32_t *) dev_.alloc_pinned((size_t) (B * K) * 4);
        h_conf_   = (float *) dev_.alloc_pinned((size_t) (B * K) * 4);
        acc_.fill(0.7);
    }
    if (df2_) {
        df2_->alloc(dev_, cfg_.max_ctx);
    }
    // AI_MM_ROWS=1 (set by a caller that will send image prompts, e.g. a server with --mmproj): f32 [E, chunk]
    // staging for the image rows of one prefill chunk (prefill_mm). Allocated here because device buffers are fixed at
    // finalize(). Without it prefill_mm with image rows refuses.
    if (getenv("AI_MM_ROWS") && atoi(getenv("AI_MM_ROWS")) != 0 && !df2_) {
        const int64_t E = prog_.spec().n_embd, C = cfg_.chunk;
        mm_buf_  = dev_.alloc_buffer((size_t) (E * C * 4) + dev_.alignment(), "mm.rows");
        size_t off = 0;
        mm_dev_  = carve(dev_, mm_buf_, off, F32, E * C);
        mm_host_ = (float *) dev_.alloc_pinned((size_t) (E * C * 4));
    }
    size_t ws = cfg_.ws_bytes ? cfg_.ws_bytes
                              : prog_.workspace_bytes(cfg_.chunk, std::max<int64_t>(16, B * (K + 1))) * (K > 0 ? 2 : 1);
    dev_.init_workspace(ws);
    dev_.finalize();
    if (kv_plan_bytes_ >= 0 && cfg.verbose) {
        // the fit model, checked on every run: what the plan expected vs what alloc_state..finalize took
        const double got = (double) (dev_.total_allocated() - kv_alloc_before_);
        fprintf(stderr, "[ai] kv auto check: planned %.0f MiB, allocated %.0f MiB (%+.1f%%)%s\n",
                kv_plan_bytes_ / 1048576.0, got / 1048576.0, 100.0 * (got / (double) kv_plan_bytes_ - 1.0),
                got > (double) kv_plan_bytes_ * 1.05 ? "  <-- UNDER-PLANNED: the fit model is missing an allocation" : "");
    }

    size_t free_b = 0, total_b = 0;
    dev_.memory_info(&free_b, &total_b);
    if (cfg.verbose) {
        const auto & s = prog_.spec();
        fprintf(stderr,
                "[ai] %s: %d layers, E=%lld, V=%lld, experts %d/%d | weights %.2f GiB in %.1f s | "
                "engine memory %.2f GiB (workspace %.0f MiB) | VRAM free %.2f / %.2f GiB | mode %s | "
                "streams %lld x %lld cells\n",
                s.arch.c_str(), s.n_layer, (long long) s.n_embd, (long long) s.n_vocab, s.n_expert_used,
                s.n_expert, prog_.weights().bytes() / 1073741824.0, prog_.weights().load_seconds(),
                dev_.total_allocated() / 1073741824.0, ws / 1048576.0, free_b / 1073741824.0,
                total_b / 1073741824.0,
                cfg_.eager ? "eager" : cfg_.expert_slots > 0 ? "rounds, streamed MoE (eager prefill)" : "rounds",
                (long long) B, (long long) R);
#if defined(AI_DOORBELL)
        if (const auto * door = prog_.doorbell()) {
            fprintf(stderr, "[ai] expert offload (doorbell): %.2f GiB on device, banks %.2f GiB in host memory\n",
                    door->device_bytes() / 1073741824.0, prog_.weights().host_bytes() / 1073741824.0);
        }
#endif
        if (const auto * pool = prog_.expert_pool()) {
            fprintf(stderr, "[ai] expert offload: %d slots/layer, pool %.2f GiB on device, banks %.2f GiB in host memory\n",
                    pool->n_slots(), pool->device_bytes() / 1073741824.0, prog_.weights().host_bytes() / 1073741824.0);
        }
    }
    if (ws_cal_pending_) {
        ws_auto_calibrate();
        ws_calibrated_now_ = true;
    }
    return true;
}

void Engine::reset_streams(int B) {
    prog_.kv().free_all();
    for (int b = 0; b < B; ++b) {
        prog_.kv().new_seq_at(b);
        prog_.slots().zero(dev_, b);
    }
    // A slot is live only between its prefill and its end/reset. Open slots start idle (done), so a round never
    // stages a slot that holds no request.
    streams_.assign((size_t) B, Stream{});
    for (auto & st : streams_) st.done = true;
    // the first round verifies what the drafter produced at prefill: the DFlash2 block (mtp_k may exceed it for lookup)
    // and for MTP the chain's width, not the lookup cap, so a lookup engine whose lookup never arms takes the same
    // width sequence as a plain MTP engine
    k_in_         = df2_ ? std::min(cfg_.mtp_k, (int) df2_->block_size() - 1) : chain_cap(cfg_.mtp_k);
    in_decode_    = false;
    live_streams_ = 0;
}

void Engine::reset_stream(int stream) {
    if (stream < 0 || stream >= (int) streams_.size()) {
        throw std::runtime_error("reset_stream: slot outside the active batch");
    }
    // free_seq is one vector element (kv.cpp) and the region is addressed as seq * region_, so
    // this cannot disturb another slot's cells. Stale cell contents past the new length are never
    // read: attention is bounded by the sequence length and the mask, not by what the cells hold.
    prog_.kv().free_seq(stream);
    prog_.kv().new_seq_at(stream);
    prog_.slots().zero(dev_, stream);
    prog_.set_rope_delta(stream, 0);   // image input
    prog_.set_pos_override(stream, nullptr, 0);
    streams_[(size_t) stream]      = Stream{};
    streams_[(size_t) stream].done = true;   // idle until its next prefill
    // Recomputed rather than adjusted. live_streams_ is exactly the count of streams not done, and an
    // incremental ++ here would drift the first time a slot is reset twice, or reset while still live.
    live_streams_ = 0;
    for (const auto & s : streams_) {
        live_streams_ += s.done ? 0 : 1;
    }
}

void Engine::open_streams(int B) {
    if (B < 1 || B > cfg_.max_batch) {
        throw std::runtime_error("open_streams: stream count outside 1..max_batch");
    }
    reset_streams(B);
    active_batch_ = B;
}

int32_t Engine::prefill_one(int stream, const std::vector<int32_t> & prompt, Timing & tm) {
    return prefill_stream(stream, prompt, tm);
}

int32_t Engine::prefill_mm(int stream, const std::vector<int32_t> & prompt, const int32_t * pos4, const float * embd,
                           const int32_t * embd_row, int64_t n_rows, Timing & tm) {
    if (df2_) {
        throw std::runtime_error("prefill_mm: image input is not available with DFlash2");
    }
    if (!pos4 || n_rows < 0 || (n_rows > 0 && (!embd || !embd_row))) {
        throw std::runtime_error("prefill_mm: pos4 is required; embd and embd_row when n_rows > 0");
    }
    const int64_t n = (int64_t) prompt.size();
    for (int64_t r = 0; r < n_rows; ++r) {
        if (embd_row[r] < 0 || embd_row[r] >= n || (r > 0 && embd_row[r] <= embd_row[r - 1])) {
            throw std::runtime_error("prefill_mm: embd_row must be strictly increasing prompt rows");
        }
    }
    if (n_rows > 0 && mm_buf_ < 0) {
        throw std::runtime_error("prefill_mm: image rows need the engine created with AI_MM_ROWS=1 (staging is fixed at init)");
    }
    mm_ = { pos4, embd, embd_row, n_rows };
    try {
        const int32_t first = prefill_stream(stream, prompt, tm);
        mm_ = {};
        return first;
    } catch (...) {
        mm_ = {};
        throw;
    }
}

// the image rows of prompt rows [off, off + T): staged into mm_dev_ and listed as contiguous runs on `in`
void Engine::mm_attach(Qwen35Hybrid::Inputs & in, int64_t off, int64_t T) {
    in.img_runs.clear();
    if (!mm_.embd || mm_.n_rows == 0) return;
    const int64_t E = prog_.spec().n_embd;
    const int32_t * lo = std::lower_bound(mm_.row, mm_.row + mm_.n_rows, (int32_t) off);
    int64_t r = lo - mm_.row, k = 0;
    for (; r < mm_.n_rows && mm_.row[r] < off + T; ++r, ++k) {
        memcpy(mm_host_ + (size_t) k * (size_t) E, mm_.embd + (size_t) r * (size_t) E, (size_t) E * 4);
        const int64_t row = mm_.row[r] - off;
        if (!in.img_runs.empty() && in.img_runs.back()[0] + in.img_runs.back()[1] == row) {
            in.img_runs.back()[1] += 1;
        } else {
            in.img_runs.push_back({ row, 1, k });
        }
    }
    if (k == 0) return;
    in.img = as_shape(mm_dev_, F32, E, k);
    dev_.upload_async(in.img, mm_host_, 0, (size_t) (E * k * 4));
}

int32_t Engine::prefill_feed(int stream, const int32_t * tokens, int64_t n, bool last, Timing & tm) {
    if (stream < 0 || stream >= (int) streams_.size()) {
        throw std::runtime_error("prefill_feed: slot outside the active batch");
    }
    if (df2_) {
        throw std::runtime_error("prefill_feed: chunked prefill is not available with DFlash2 -- use prefill_one");
    }
    if (n < 0 || (n > 0 && tokens == nullptr)) {
        throw std::runtime_error("prefill_feed: tokens must be non-null when n > 0");
    }
    if (!streams_[(size_t) stream].prefilling) {
        reset_stream(stream);   // a new request starts from a zeroed slot, exactly as prefill_stream starts one
        request_start();
        streams_[(size_t) stream].prefilling = true;
    }
    Stream & s = streams_[(size_t) stream];
    s.pf.insert(s.pf.end(), tokens, tokens + n);
    if (last && s.pf.empty()) {
        reset_stream(stream);
        throw std::runtime_error("prefill_feed: empty prompt");
    }
    dev_.ws_tag("prefill");
    if (cfg_.mtp_k > 0) {
        prefill_mtp_range(stream, s.pf, s.pf_off, last, s);
    } else {
        const int32_t t = prefill_range(stream, s.pf, s.pf_off, last);
        if (last) {
            s.pend = t;
        }
    }
    if (!last) {
        return -1;
    }
    // prefill_stream's epilogue
    s.pos  = (int64_t) s.pf.size();
    s.done = false;
    lk_begin(s, s.pf);
    live_streams_ = 0;
    for (const auto & x : streams_) live_streams_ += x.done ? 0 : 1;
    tm.prompt_n += s.pos;
    s.prefilling = false;
    std::vector<int32_t>().swap(s.pf);
    s.pf_off = 0;
    return s.pend;
}

const std::vector<int32_t> & Engine::prefill_buffer(int stream) const {
    if (stream < 0 || stream >= (int) streams_.size()) {
        throw std::runtime_error("prefill_buffer: slot outside the active batch");
    }
    return streams_[(size_t) stream].pf;
}

void Engine::request_start() {
    // default: on when the output is meant to be exact (plain decode, or the EXACT verify), off for FAST drafting,
    // whose tokens differ from plain anyway and which keeps the residency the history built (saves the restore)
    const char * env = getenv("AI_DOOR_RESET_PER_REQUEST");
    const bool   on  = env ? atoi(env) != 0 : (cfg_.mtp_k == 0 || cfg_.exact_verify);
    if (on && streams_.size() == 1) {
        prog_.offload_reset_residency();
    }
}

void Engine::reset_one(int stream) {
    reset_stream(stream);
}

int64_t Engine::stream_position(int stream) const {
    if (stream < 0 || stream >= (int) streams_.size()) {
        throw std::runtime_error("stream_position: slot outside the active batch");
    }
    return streams_[(size_t) stream].pos + prog_.rope_delta(stream);   // the RoPE position (== cells for text)
}

bool Engine::stream_done(int stream) const {
    if (stream < 0 || stream >= (int) streams_.size()) {
        throw std::runtime_error("stream_done: slot outside the active batch");
    }
    return streams_[(size_t) stream].done;
}

int Engine::warm_profiles(Timing & tm) {
    const int B = active_batch_;
    if ((int) streams_.size() != B || B < 1) {
        throw std::runtime_error("warm_profiles: no open batch -- call open_streams() first");
    }
    if (cfg_.mtp_k > 0) {
        // An MTP round's key carries k_in/k_out, which come from the adaptive schedule at run time.
        // Enumerating them here would guess a schedule rather than record the one that will run.
        throw std::runtime_error("warm_profiles: not supported with mtp_k > 0 -- an MTP profile key "
                                 "depends on the adaptive k schedule, which is not known at init");
    }
    const int64_t R = prog_.kv().region_cells();
    // Exactly the values frontier() can return: powers of two from 256, clamped to the region.
    std::vector<int64_t> ladder;
    for (int64_t f = 256;; f *= 2) {
        const int64_t v = std::min(f, R);
        if (ladder.empty() || ladder.back() != v) {
            ladder.push_back(v);
        }
        if (f >= R) {
            break;
        }
    }
    const int64_t before = tm.rounds_recorded;
    // the full-width round, plus (multi-slot, compaction on) the 1-wide round on each slot's region that
    // decode_round uses while exactly one stream is live -- the common case for a 2-slot server
    std::vector<std::pair<int, int>> shapes = { { B, 0 } };
    const bool compact = !getenv("AI_COMPACT") || atoi(getenv("AI_COMPACT")) != 0;
    if (compact && B > 1 && !df2_) {
        for (int r = 0; r < B; ++r) shapes.emplace_back(1, r);
    }
    for (size_t i = 0; i < ladder.size(); ++i) {
        const int64_t F = ladder[i];
        const int64_t L = 1;            // plain decode stages one row per stream
        for (const auto & [W, R0] : shapes) {
            const int64_t T = L * W;
            for (int b = 0; b < W; ++b) {
                int32_t tok = 0;
                // A large attention envelope over a short sequence is the ordinary decode shape -- the
                // envelope is a launch promise and the live length is carried by the mask -- so this
                // records the same graph a real round at this F will replay.
                stage_rows(main_in_, R0 + b, b * L, T, &tok, L, 0, F, false);
            }
            main_in_.upload(dev_, L, W, F);
            const Qwen35Hybrid::Inputs vin = main_in_.view(L, W, F, R0);
            bool warm = false;
            step_round(ProfileKey{ F, W, 0, 0, R0 }, [&]() { return build_plain(vin); }, tm, &warm);
            dev_.synchronize();
        }
    }
    const int recorded = (int) (tm.rounds_recorded - before);
    // Required: staging reserved KV to F on every slot and wrote garbage recurrent state. Clearing
    // here means a caller cannot observe a warmed slot, so prefill-before-decode stays a guarantee.
    reset_streams(B);
    active_batch_ = B;
    if (cfg_.verbose) {
        fprintf(stderr, "[ai] warm_profiles: %d profile(s) recorded over %zu F bucket(s), B=%d; "
                        "all slots reset\n", recorded, ladder.size(), B);
    }
    return recorded;
}

int Engine::warm_mtp_profiles(int64_t pos_lo, int64_t pos_hi, Timing & tm) {
    const int K = chain_cap(cfg_.mtp_k);   // the MTP chain's widths (lookup's wider buckets are captured on first use)
    if (K <= 0 || df2_ || cfg_.mtp_adapt) {
        throw std::runtime_error("warm_mtp_profiles: needs MTP (mtp_k > 0) without --mtp-adapt or DFlash2");
    }
    if ((int) streams_.size() != active_batch_ || active_batch_ != 1) {
        throw std::runtime_error("warm_mtp_profiles: needs an open batch of exactly one stream (open_streams(1))");
    }
    // A round at maxpos p asks for F = frontier(p + L + k_out + 2), L = k_in + 1 (decode_round). The envelope set for
    // positions [pos_lo, pos_hi] is every bucket from the narrowest round at pos_lo to the widest at pos_hi.
    const int64_t R  = prog_.kv().region_cells();
    const int64_t lo = std::max<int64_t>(0, pos_lo);
    const int64_t hi = std::min<int64_t>(pos_hi, R - (2 * K + 3));
    if (hi < lo) {
        return 0;
    }
    std::vector<int64_t> Fs;
    const int64_t f_last = frontier(hi + (K + 1) + K + 2);
    for (int64_t f = frontier(lo + 1 + K + 2);;) {
        Fs.push_back(f);
        if (f >= f_last || f >= R) break;
        f = frontier(f + 1);
    }
    static const int graph_max = getenv("AI_GRAPH_MAX") ? atoi(getenv("AI_GRAPH_MAX")) : 16;
    if ((int) Fs.size() * (K + 1) > graph_max) {
        fprintf(stderr, "[ai] warm_mtp_profiles: %zu envelope buckets x %d widths exceed the %d-profile graph budget; "
                        "warming only the first %d bucket(s)\n", Fs.size(), K + 1, graph_max, graph_max / (K + 1));
        Fs.resize((size_t) std::max(1, graph_max / (K + 1)));
    }
    // state a synthetic round would disturb (the calibration saves the same set)
    const int  saved_k_in = k_in_;
    const auto saved_acc  = acc_;
    const auto saved_step = step_ms_;
    const long long saved_cold_n  = g_cold_rounds;
    const double    saved_cold_ms = g_cold_ms;
    Timing scratch;
    int warmed = 0, missing = 0;
    ws_calibrating_ = true;   // no gate, no compaction, no AI_TF dump / conf log for these rounds
    try {
        // the widths a round can take: the MTP chain's 0..K, plus the AI_LOOKUP buckets above it (lk_bucket)
        std::vector<int> ks;
        for (int k = 0; k <= K; ++k) ks.push_back(k);
        if (lookup_cfg().on && !cfg_.exact_verify) {
            for (int b : { 8, 16, 32, 64, 128 }) {
                const int w = std::min(b, cfg_.mtp_k);
                if (w > K && std::find(ks.begin(), ks.end(), w) == ks.end()) ks.push_back(w);
            }
        }
        for (const int64_t F : Fs) {
            for (const int k : ks) {
                // the position whose round lands exactly in bucket F at verify width k (frontier(F) == F)
                const int64_t p = F - (k + 1) - K - 2;
                if (p < 0) continue;
                for (int rep = 0; rep < 3; ++rep) {   // launch 1 direct, launch 2 captures, launch 3 replays
                    Stream & s = streams_[0];
                    s.done = false;
                    s.prefilling = false;
                    s.out.clear();
                    s.pend = 0;
                    s.drafts.assign((size_t) std::max(K, k), 0);
                    s.pos = p;
                    live_streams_ = 1;
                    k_in_ = k;
                    in_decode_ = true;
                    decode_round([](int, int32_t) { return true; }, scratch);
                }
                const auto * it = rounds_.find(ProfileKey{ F, 1, k, K, 0 });
                if (it != nullptr && it->launches > 2) {
                    ++warmed;
                } else {
                    ++missing;   // the live key differs from the one predicted here: report, never hide it
                }
            }
        }
    } catch (...) {
        ws_calibrating_ = false;
        reset_streams(1);
        active_batch_ = 1;
        throw;
    }
    ws_calibrating_ = false;
    // required, as in warm_profiles: the rounds wrote garbage KV and recurrent state into slot 0
    reset_streams(1);
    active_batch_ = 1;
    k_in_         = saved_k_in;
    acc_          = saved_acc;
    step_ms_      = saved_step;
    g_cold_rounds = saved_cold_n;
    g_cold_ms     = saved_cold_ms;
    tm.rounds_recorded += scratch.rounds_recorded;
    fprintf(stderr, "[ai] warm_mtp_profiles: %d profile(s) captured (%zu envelope bucket(s) F %lld..%lld x k_in 0..%d, "
                    "positions %lld..%lld)%s\n", warmed, Fs.size(), (long long) Fs.front(), (long long) Fs.back(), K,
            (long long) lo, (long long) hi, missing ? " -- KEY MISMATCH: some predicted profiles were not found" : "");
    return missing ? -missing : warmed;
}

void Engine::decode_step(const std::vector<DecodeIn> & in, std::vector<std::pair<int, int32_t>> & out,
                         Timing & tm) {
    const int B = active_batch_;
    if ((int) streams_.size() != B || B < 1) {
        throw std::runtime_error("decode_step: no open batch -- call open_streams() first");
    }
    char msg[256];
    std::vector<int> seen((size_t) B, 0);
    for (const auto & d : in) {
        if (d.stream < 0 || d.stream >= B) {
            snprintf(msg, sizeof msg, "decode_step: stream %d outside the open batch of %d", d.stream, B);
            throw std::runtime_error(msg);
        }
        if (seen[(size_t) d.stream]++) {
            snprintf(msg, sizeof msg, "decode_step: stream %d named twice in one step", d.stream);
            throw std::runtime_error(msg);
        }
    }
    // Batch-rigid: every slot is staged every round, so an unnamed open stream would be advanced with
    // whatever its previous pending token was. Refuse instead of producing a silent wrong answer.
    for (int b = 0; b < B; ++b) {
        if (!streams_[(size_t) b].done && !seen[(size_t) b]) {
            snprintf(msg, sizeof msg,
                     "decode_step: a round is batch-rigid and stream %d is open but was not named; "
                     "every open stream must appear in one step", b);
            throw std::runtime_error(msg);
        }
    }
    for (const auto & d : in) {
        Stream & s = streams_[(size_t) d.stream];
        if (s.done) {
            snprintf(msg, sizeof msg, "decode_step: stream %d is idle (not prefilled since open/reset)", d.stream);
            throw std::runtime_error(msg);
        }
        // The error names both values; one number alone is not actionable.
        const int64_t have = s.pos + prog_.rope_delta(d.stream);   // callers count RoPE positions
        if (d.position != have) {
            snprintf(msg, sizeof msg,
                     "decode_step: position mismatch on stream %d -- engine has %lld, caller sent %lld",
                     d.stream, (long long) have, (long long) d.position);
            throw std::runtime_error(msg);
        }
        s.pend = d.token;
    }
    in_decode_ = true;
    // The sink always returns true: in caller-driven decode the caller owns stopping, so the engine
    // must not mark a stream done. s.out still accumulates per slot and is cleared by reset_one(),
    // which bounds it to one request's length.
    const std::function<bool(int, int32_t)> sink = [&out](int b, int32_t tok) {
        out.emplace_back(b, tok);
        return true;
    };
    decode_round(sink, tm);
}

bool Engine::emit_token(int stream, int32_t tok, const std::function<bool(int, int32_t)> & sink, Timing & tm) {
    Stream & s = streams_[(size_t) stream];
    s.out.push_back(tok);
    tm.decode_tokens += in_decode_ ? 1 : 0;
    // The sink runs after the push, so a caller counting out.size() sees the token just emitted.
    // generate_batch's n_gen rule depends on that ordering.
    const bool more = sink(stream, tok);
    if (!more) {
        s.done = true;
        --live_streams_;
    }
    return more;
}

// AI_LOOKUP: a request's lookup state from its whole prompt + the pending token (MTP or DFlash2). Called by both
// prefill paths (prefill_stream and prefill_feed, the C API's ai_engine_prefill_chunk).
void Engine::lk_begin(Stream & s, const std::vector<int32_t> & prompt) {
    s.lk_k = 0;
    if (lookup_cfg().on && cfg_.mtp_k > 0) {
        s.lk_armed = lookup_cfg().start >= 2 || (lookup_cfg().start == 1 && df2_on());   // AI_LOOKUP_START
        s.hist = prompt;
        s.hist.push_back(s.pend);
        s.lk.clear();
        s.lk_done = 0;
    }
}

int32_t Engine::prefill_stream(int stream, const std::vector<int32_t> & prompt, Timing & tm) {
    dev_.ws_tag("prefill");
    if (stream < 0 || stream >= (int) streams_.size()) {
        throw std::runtime_error("prefill_stream: slot outside the active batch");
    }
    if (prompt.empty()) {
        throw std::runtime_error("prefill_stream: empty prompt");
    }
    // Every request starts from a zeroed slot: an idle slot may have ridden along in full-width rounds since its reset
    // (3+ slots), and a caller that prefills a still-live slot without ai_engine_reset_stream would otherwise continue
    // from that slot's stale recurrent state. The reset is a KV free + one state memset.
    reset_stream(stream);
    request_start();
    Stream & s = streams_[(size_t) stream];
    if (mm_.pos4) {   // image input: this prefill's rows carry explicit RoPE positions; later rows run at cell + delta
        const int64_t n = (int64_t) prompt.size();
        // the next position after the prompt: one past its largest t / y / x component (llama.cpp's mtmd advances by
        // max(nx, ny) per image, which is exactly this when the prompt ends with an image)
        int32_t       tmax = -1;
        for (int64_t i = 0; i < 3 * n; ++i) tmax = std::max(tmax, mm_.pos4[i]);
        prog_.set_pos_override(stream, mm_.pos4, n);
        prog_.set_rope_delta(stream, (int64_t) tmax + 1 - n);
    }
    if (df2_on()) {
        prefill_df2(stream, prompt, s);
    } else if (cfg_.mtp_k > 0) {
        prefill_mtp(stream, prompt, s);
    } else {
        s.pend = prefill(stream, prompt);
    }
    s.pos  = (int64_t) prompt.size();
    prog_.set_pos_override(stream, nullptr, 0);   // the caller's arrays are not ours past this call
    s.done = false;   // the slot is live from its prefill (open/reset leave it idle)
    if (df2_) {   // AI_DF2_ROUTE: the first round verifies what this request's drafter produced
        k_in_ = df2_on() ? std::min(cfg_.mtp_k, (int) df2_->block_size() - 1) : chain_cap(cfg_.mtp_k);
    }
    lk_begin(s, prompt);
    if (FILE * dt = draft_trace_file(); dt != nullptr && !ws_calibrating_) {
        fprintf(dt, "Q %d %zu", stream, prompt.size());
        for (int32_t t : prompt) fprintf(dt, " %d", t);
        fprintf(dt, "\nF %d %lld %d |", stream, (long long) s.pos, s.pend);
        for (int32_t d : s.drafts) fprintf(dt, " %d", d);
        fprintf(dt, "\n");
        fflush(dt);
    }
    live_streams_ = 0;
    for (const auto & x : streams_) live_streams_ += x.done ? 0 : 1;
    tm.prompt_n += s.pos;
    return s.pend;
}

int64_t Engine::frontier(int64_t cells) const {
    // Decode profiles: powers of two from 256 up to the region. The attention envelope is a launch
    // promise; the live length is carried by the mask and costs nothing to change.
    const int64_t R = prog_.kv().region_cells();
    if (cells > R) {
        throw std::runtime_error("frontier: sequence outgrew its KV region (raise --ctx)");
    }
    // AI_F_MIN: the smallest envelope a round may take (rounded up to the ladder). Fewer envelope buckets = fewer
    // distinct round profiles to record and cache, so fewer re-records at the profile cap.
    static const int64_t f_min = getenv("AI_F_MIN") ? atoll(getenv("AI_F_MIN")) : 0;
    int64_t f = 256;
    while (f < cells || f < f_min) {
        f *= 2;
    }
    return std::min(f, R);
}

// ------------------------------------------------------------------ staging

void Engine::stage_rows(Ingress & g, int seq, int64_t row0, int64_t T, const int32_t * tokens, int64_t L,
                        int64_t pos0, int64_t n_kv, bool head) {
    if (row0 + L > T || T > g.T_cap || (!g.mask_on_device && !g.mask_fits(n_kv, T))) {
        throw std::runtime_error("stage_rows: round larger than its buffers");
    }
    if (head) {
        prog_.fill_head_inputs(seq, L, pos0, n_kv, row0, T, g.h_pos, g.h_cells,
                               (g.mask_fits(n_kv, T) && !g.mask_dev_now) ? g.h_mask : nullptr);
        if (tokens) {
            memcpy(g.h_ids + row0, tokens, (size_t) L * 4);
        }
    } else {
        prog_.fill_inputs(seq, tokens, L, pos0, n_kv, row0, T, g.h_ids, g.h_pos, g.h_cells,
                          g.mask_dev_now ? nullptr : g.h_mask);
    }
    const int64_t b = row0 / L;
    g.h_last[b] = (int32_t) (row0 + L - 1);
    g.h_base[b] = (int32_t) pos0;
    g.h_delta[b] = (int32_t) prog_.rope_delta(seq);
}

Qwen35Hybrid::Inputs Engine::stage_chunk(int stream, const int32_t * tokens, int64_t T, int64_t pos0, int64_t n_kv) {
    stage_rows(main_in_, stream, 0, T, tokens, T, pos0, n_kv, false);
    main_in_.upload(dev_, T, 1, n_kv);
    Qwen35Hybrid::Inputs in = main_in_.view(T, 1, n_kv, stream);
    in.chunk = true;   // AI_CHUNK_SCOPE: scoped at any width
    return in;
}

// ------------------------------------------------------------------ prefill / score

int32_t Engine::prefill(int stream, const std::vector<int32_t> & prompt) {
    int64_t off = 0;
    return prefill_range(stream, prompt, off, true);
}

// The body of the whole-prompt prefill, resumable (AI_PREFILL_CHUNK): it runs the chunks from `off` (advanced in
// place) at the same boundaries as a one-shot prefill (multiples of cfg_.chunk from 0). When !final it runs only
// chunks that end before the buffer does, so the last chunk -- the one that takes the logits -- is always run by the
// final call. Same chunks, same arithmetic: a chunked prefill is a one-shot prefill split across calls.
int32_t Engine::prefill_range(int stream, const std::vector<int32_t> & prompt, int64_t & off, bool final) {
    const int64_t n = (int64_t) prompt.size();
    int32_t tok = -1;
    for (; off < n; off += cfg_.chunk) {
        const int64_t T    = std::min(cfg_.chunk, n - off);
        if (!final && off + T >= n) {
            break;
        }
        const bool    last = off + T == n;
        dev_.ws_reset(0);
        auto in = stage_chunk(stream, prompt.data() + off, T, off, prog_.visible_cells(off, T));
        mm_attach(in, off, T);
        static char tag[96];
        snprintf(tag, sizeof(tag), "prefill T=%lld off=%lld%s", (long long) T, (long long) off, last ? " last" : "");
        dev_.ws_tag(tag);

        std::unique_ptr<op::Round> round;
        if (!chunk_eager()) {
            round = std::make_unique<op::Round>(dev_);
            round->begin();
        }
        // The last chunk keeps every row through the final layer and takes the logits of its last row, as MTP-mode
        // prefill does (engine_mtp.cpp prefill_mtp). Pruning the final layer to one row would run that row through
        // different (M=1) kernels, so the first generated token could differ between plain and MTP runs. Cost: one
        // layer of one chunk.
        Tensor hidden = prog_.forward(dev_, in, last);
        Tensor am;
        if (last) {
            const int64_t Eh = hidden.ne[0], Th = hidden.ne[1];
            const Tensor  lr = hidden.view(Eh, 1, 1, 1, hidden.nb[1], hidden.nb[1], hidden.nb[1], hidden.nb[1] * (size_t) (Th - 1));
            am = op::argmax(dev_, prog_.logits(dev_, lr));
        }
        if (round) {
            round->end();
            round->launch();
        }
        if (last) {
            dev_.download_async(am, h_out_, 0, 4);
            dev_.synchronize();
            tok = h_out_[0];
        }
    }
    return tok;
}

void Engine::score(const std::vector<int32_t> & tokens,
                   const std::function<void(int64_t, const float *)> & sink, Timing * t) {
    reset_streams(1);
    const int64_t n     = (int64_t) tokens.size();
    const int64_t V     = prog_.spec().n_vocab;
    const int64_t E     = prog_.spec().n_embd;
    const int64_t slice = 16;
    std::vector<float> host((size_t) (slice * V));

    const auto t0 = clock_t_::now();
    for (int64_t off = 0; off < n; off += cfg_.chunk) {
        const int64_t T = std::min(cfg_.chunk, n - off);
        dev_.ws_reset(0);
        auto in = stage_chunk(0, tokens.data() + off, T, off, prog_.visible_cells(off, T));
        Tensor hidden;
        if (chunk_eager()) {
            hidden = prog_.forward(dev_, in, true);
        } else {
            op::Round round(dev_);
            round.begin();
            hidden = prog_.forward(dev_, in, true);
            round.end();
            round.launch();
        }
        // the logits slice sits on top of the forward's live workspace, which the calibration measured with one logit
        // row per chunk: size the slice to the room left (a multi-row slice of a large vocab can overflow it)
        const size_t  room = dev_.ws_capacity() > dev_.ws_mark() ? dev_.ws_capacity() - dev_.ws_mark() : 0;
        const size_t  row  = (size_t) V * 4 + (size_t) E * 8 + 4096;
        const int64_t fit  = std::max<int64_t>(1, std::min<int64_t>(slice, (int64_t) (room > (1u << 20) ? (room - (1u << 20)) / row : 0)));
        for (int64_t s = 0; s < T; s += fit) {
            const int64_t m = std::min(fit, T - s);
            Device::Scope sc(dev_);
            Tensor part   = hidden.view(E, m, 1, 1, hidden.nb[1], hidden.nb[1] * (size_t) m,
                                        hidden.nb[1] * (size_t) m, hidden.nb[1] * (size_t) s);
            Tensor logits = prog_.logits(dev_, part);
            dev_.download(logits, host.data(), 0, (size_t) (m * V) * 4);
            for (int64_t r = 0; r < m; ++r) {
                sink(off + s + r, host.data() + r * V);
            }
        }
    }
    if (t) {
        t->prompt_n  = n;
        t->prefill_s = ms_between(t0, clock_t_::now()) / 1e3;
    }
}

// ------------------------------------------------------------------ decode

Engine::StepOut Engine::build_plain(const Qwen35Hybrid::Inputs & in) {
    dev_.ws_tag("plain decode round");
    StepOut o;
    o.logits = prog_.logits(dev_, prog_.forward(dev_, in, false));
    o.token  = op::argmax(dev_, o.logits);
    return o;
}

const Engine::StepOut & Engine::step_round(const ProfileKey & key, const std::function<StepOut()> & build,
                                           Timing & tm, bool * warm) {
    // offload: a round records only if all its routed experts fit the slots at once
    const int64_t rows = (int64_t) std::get<1>(key) * (std::get<2>(key) + 1);
    const bool    streamed_eager = (cfg_.expert_slots > 0 && !prog_.streamed_fixed_ok(std::max(rows, (int64_t) 1))) ||
                                   (prog_.cpu_ffn_tmax() > 0 && rows > prog_.cpu_ffn_tmax());   // AI_FFN_CPU
    if (cfg_.eager || streamed_eager) {
        dev_.ws_reset(0);
        eager_out_ = build();
        *warm = true;
        return eager_out_;
    }
    return rounds_.launch(dev_, key, build, warm, cfg_.verbose, [&](const ProfileKey & k, const auto & e) {
        tm.rounds_recorded++;
        tm.round_nodes = std::max(tm.round_nodes, e.round->n_nodes());
        // AI_PROFILE: one line per recorded profile, naming the key and this profile's own node count.
        // tm.round_nodes above is a running max across profiles, which hides a node count that is not
        // monotone in F. ggml's flash attention allocates a fixup buffer only when
        // ntiles_dst % blocks_num.x != 0, a modular test, so nodes can appear and disappear as F doubles.
        size_t vfree = 0, vtot = 0;
        dev_.memory_info(&vfree, &vtot);
        fprintf(stderr, "[ai-profile] recorded #%lld  F=%lld B=%d k_in=%d k_out=%d r0=%d  nodes=%d  free %.0f MiB\n",
                (long long) tm.rounds_recorded, (long long) std::get<0>(k), std::get<1>(k),
                std::get<2>(k), std::get<3>(k), std::get<4>(k), e.round->n_nodes(),
                vfree / 1048576.0);
    });
}

std::vector<int32_t> Engine::generate(const std::vector<int32_t> & prompt, int64_t n_gen,
                                      const std::function<bool(int32_t)> & on_token, Timing * t) {
    auto outs = generate_batch({ prompt }, n_gen, [&](int, int32_t tok) { return on_token(tok); }, t);
    return outs.at(0);
}

int Engine::decode_round(const std::function<bool(int, int32_t)> & sink, Timing & tm) {
    const int  BA  = active_batch_;
    const int  K   = cfg_.mtp_k;
    const bool mtp = K > 0;
    if ((int) streams_.size() != BA) {
        throw std::runtime_error("decode_round: hoisted stream state does not match the active batch");
    }
    // Compaction: a round is B streams wide and computes every one of them. With one live stream in a multi-slot
    // engine (a 2-slot server serving one request), the round runs 1 wide on that stream's own KV region and state
    // slot (R0) instead of 2 wide with an idle passenger, which roughly doubles that user's speed. The region is
    // baked into a captured graph, so it is part of the profile key. AI_COMPACT=0 restores the rigid batch.
    int R0 = 0, B = BA;
    static const bool compact = !getenv("AI_COMPACT") || atoi(getenv("AI_COMPACT")) != 0;
    if (compact && BA > 1 && !df2_on() && !ws_calibrating_) {
        int live = 0, last = -1;
        for (int b = 0; b < BA; ++b) {
            if (!streams_[(size_t) b].done) { ++live; last = b; }
        }
        if (live == 1) { R0 = last; B = 1; }
    }
    // AI_PREFILL_CHUNK: a round computes every stream in its width (idle ones ride along with frozen inputs), which
    // would advance a mid-prefill stream's recurrent state and write its KV, corrupting later text. A 2-slot
    // engine never gets here (one live stream compacts to 1 wide); a 3+-slot engine or AI_COMPACT=0 must not mix
    // chunked prefill with decode, and is told so instead of being corrupted.
    for (int b = R0; b < R0 + B; ++b) {
        if (streams_[(size_t) b].prefilling) {
            char m[320];
            snprintf(m, sizeof m,
                     "decode_round: stream %d is mid-prefill (chunked) and this round is %d wide from slot %d, so it would "
                     "compute that stream and corrupt its state; chunked prefill needs compaction (AI_COMPACT on) with at "
                     "most one live stream besides the one prefilling (n_streams <= 2), or finish the prefill first",
                     b, B, R0);
            throw std::runtime_error(m);
        }
    }
    // Reference aliases keep the round body's original local names.
    // st[b] is the round's b-th stream: slot R0 + b (R0 = 0 unless compacted)
    struct RoundStreams {
        std::vector<Stream> & v; int r0; int n;
        Stream & operator[](size_t b) const { return v[(size_t) r0 + b]; }
        Stream * begin() const { return v.data() + r0; }
        Stream * end() const { return v.data() + r0 + n; }
    } st{ streams_, R0, B };
    int                  & k_in    = k_in_;
    std::vector<int32_t> & vt      = vt_;
    std::vector<int>     & acc_now = acc_now_;
    auto emit = [&](int b, int32_t tok) { return emit_token(R0 + b, tok, sink, tm); };
    vt.resize((size_t) K + 1);
    for (int b = 0; b < B; ++b) {
        // A slot reset by reset_stream() has no drafts yet. Staging would read past the end of the
        // vector and build a round out of garbage ids, a silent wrong answer. Refuse at
        // the staging site and name the contract instead of degrading.
        Stream & sb = st[(size_t) b];
        if (sb.done) {
            // an idle or finished slot riding along in a full-width round (3+ slots): give it valid ids so the
            // round reads in-range embeddings; its outputs are discarded and prefill re-zeroes the slot
            if (sb.pend < 0) sb.pend = 0;
            if ((int) sb.drafts.size() < K) sb.drafts.resize((size_t) K, sb.pend);
            continue;
        }
        if ((int64_t) sb.drafts.size() < k_in) {
            throw std::runtime_error("decode_round: slot has fewer drafts than the round's k_in "
                                     "(reset_stream must be followed by prefill_stream)");
        }
    }

        const auto r0    = clock_t_::now();
        // Exact verify at 2+ streams: exact verify holds up to 4 verify columns; at 5+ ggml's MMVQ halves its warp
        // split (compile-time pin only). A round's columns are streams x (k_in + 1), so with B live streams the verify
        // keeps k_in <= 4/B - 1 (2 streams verify 1 draft each, 3+ verify none) and drafts only that many. One live
        // stream compacts to B = 1 and gets the full K back.
        // The same cap also helps fast mode: a wide multi-stream verify reads about twice the distinct experts.
        // AI_MTP_BCAP=0 restores full width in fast mode (exact keeps the cap: it is correctness).
        // Whether the cap applies is the policy's (cfg_.stream_cap, engine/mtp_policy.cpp): MoE fast and every exact.
        int k_cap = K;
        if (mtp && B > 1 && !df2_on() && cfg_.stream_cap &&
            !(cfg_.exact_verify && getenv("AI_EXACT_ANY_K") && atoi(getenv("AI_EXACT_ANY_K")) != 0)) {
            k_cap = std::max(0, 4 / B - 1);
            k_in  = std::min(k_in, k_cap);
        }
        // the MTP chain drafts at most chain_cap(K) (<= 8, AI_MTP_CHAIN lower) while K stays the widest verify
        const int  kc_d  = df2_on() ? std::min(K, (int) df2_->block_size() - 1) : chain_cap(K);   // DFlash2: its full block
        // (not the MTP chain cap, which AI_DF2_ROUTE may set below the DFlash2 block)
        const int  k_out = mtp ? std::max(1, std::min(std::min(k_cap, kc_d),
                                                      cfg_.mtp_adapt ? choose_k(k_in) : K)) : 0;
        const int64_t L  = k_in + 1;
        int64_t maxpos = 0;
        for (const auto & s : st) {
            maxpos = std::max(maxpos, s.pos);
        }
        const int64_t F = frontier(maxpos + L + k_out + 2);
        const int64_t T = L * B;

        static const bool tf_force = getenv("AI_TF_FORCE") != nullptr;
        if (tf_force && mtp && B == 1) {
            Stream & s = st[0];
            for (int j = 0; j < k_in; ++j) {
                int32_t t;
                if (tf_force_token(s.pos + 1 + j, &t)) s.drafts[(size_t) j] = t;
            }
        }
        // AI_DECODE_MASK_DEV (default on): the captured round writes its masks (main [F, L, B] and the MTP head's)
        // from the per-stream base positions it already receives, byte-identical to fill_inputs / fill_head_inputs,
        // instead of filling them on the host and uploading F x L x 2 bytes each round. The AI_MTP_PROF and
        // AI_NO_ROUND debug paths keep the host masks.
        static const bool mask_dev_env = getenv("AI_DECODE_MASK_DEV") ? atoi(getenv("AI_DECODE_MASK_DEV")) != 0 : true;
        static const bool mask_prof = getenv("AI_MTP_PROF") != nullptr, mask_noround = getenv("AI_NO_ROUND") != nullptr;
        const bool mask_dev = mask_dev_env && !(mtp && mask_prof && !df2_on()) && !(!mtp && mask_noround);
        main_in_.mask_dev_now = mask_dev;
        head_in_.mask_dev_now = mask_dev;
        for (int b = 0; b < B; ++b) {
            Stream & s = st[(size_t) b];
            vt[0] = s.pend;
            std::copy(s.drafts.begin(), s.drafts.begin() + k_in, vt.begin() + 1);
            stage_rows(main_in_, R0 + b, b * L, T, vt.data(), L, s.pos, F, false);
            if (mtp && !df2_on()) {
                stage_rows(head_in_, R0 + b, b * L, T, nullptr, L, s.pos + 1, F, true);
            }
        }
        main_in_.upload(dev_, L, B, F);
        const auto vin = main_in_.view(L, B, F, R0);
        Qwen35Hybrid::Inputs hin;
        if (mtp && !df2_on()) {
            head_in_.upload(dev_, L, B, F);
            hin = head_in_.view(L, B, F, R0);
        }
        main_in_.mask_dev_now = false;   // every other user of the ingress (prefill, calibration) keeps host masks
        head_in_.mask_dev_now = false;

        bool warm = false;
        // The staging registry's entries describe workspace memory that this round is about to
        // recycle, so drop them here rather than relying on same-address replacement. Host-owned
        // lifetime (ALPHA_Q8_STAGE_HOST_LIFETIME) makes the backend's own per-pass reset a no-op:
        // a round is many passes, and its producer and consumers live in different ones.
        kern::q8_stage_clear();
        static const bool mtp_prof = getenv("AI_MTP_PROF") != nullptr;
        // AI_NO_ROUND=1: run the step without a Recorder, so the per-layer AI_DUMP_DIR tap (which is
        // inert while recording) fires on decode. Debug only; it costs the round's capture.
        static const bool no_round = getenv("AI_NO_ROUND") != nullptr;
        StepOut prof_out;
        if (!mtp && no_round) {
            dev_.ws_reset(0);
            prof_out = build_plain(vin);
            dev_.synchronize();
            warm     = true;
        } else if (mtp && mtp_prof && !df2_on()) {
            dev_.ws_reset(0);
            prof_out = profile_mtp(vin, hin, k_out, tm);
            warm = true;
        }
        const StepOut & o = ((mtp && mtp_prof && !df2_on()) || (!mtp && no_round))
                                ? prof_out
                                : step_round(ProfileKey{ F, B, k_in, k_out, R0 },
                                             [&]() {
                                                 if (mask_dev) {   // AI_DECODE_MASK_DEV: first ops of the round
                                                     op::decode_causal_mask(dev_, vin.mask, main_in_.base_view(B), L,
                                                                            B, F, false);
                                                     if (mtp && !df2_on()) {
                                                         op::decode_causal_mask(dev_, hin.mask, head_in_.base_view(B),
                                                                                L, B, F, true);
                                                     }
                                                 }
                                                 return df2_on() ? build_df2(vin)
                                                        : mtp ? build_mtp(vin, hin, k_out) : build_plain(vin);
                                             },
                                             tm, &warm);
        if (mtp) {
            dev_.download_async(o.g, h_g_, 0, (size_t) T * 4);
            dev_.download_async(o.m, h_m_, 0, (size_t) B * 4);
            for (int s = 0; s < (int) o.drafts.size(); ++s) {   // DFlash2 drafts after the sync instead
                dev_.download_async(o.drafts[(size_t) s], h_drafts_ + s * B, 0, (size_t) B * 4);
            }
            for (int s = 0; s < (int) o.confs.size(); ++s) {
                dev_.download_async(o.confs[(size_t) s], h_conf_ + s * B, 0, (size_t) B * 4);
            }
        } else {
            dev_.download_async(o.token, h_out_, 0, (size_t) B * 4);
        }
        const auto r1 = clock_t_::now();
        dev_.synchronize();
        const auto r2 = clock_t_::now();
        prog_.offload_after_round();   // doorbell admission between rounds (no-op without it)

        static const bool tf_dump = getenv("AI_TF_DUMP") != nullptr;
        if (tf_dump && B == 1 && o.logits.valid() && !st[0].done && !ws_calibrating_) {
            const int64_t V    = o.logits.ne[0];
            const int64_t rows = mtp ? std::min<int64_t>(h_m_[0], k_in) + 1 : 1;
            std::vector<float> lg((size_t) (V * rows));
            dev_.download(o.logits, lg.data(), 0, lg.size() * 4);
            const Stream & s = st[0];
            for (int64_t l = 0; l < rows; ++l) {
                const int32_t fed = l == 0 ? s.pend : s.drafts[(size_t) (l - 1)];
                const int32_t am  = mtp ? h_g_[l] : h_out_[0];
                tf_write(s.pos + l, fed, am, lg.data() + l * V, V);
            }
            fflush(tf_state().dump);
        }
        acc_now.clear();
        for (int b = 0; b < B; ++b) {
            Stream & s = st[(size_t) b];
            if (s.done) {
                // finished streams ride along with frozen inputs until the batch ends
                s.drafts.assign((size_t) k_out, s.pend);
                continue;
            }
            if (!mtp) {
                s.pend = h_out_[b];
                s.pos += 1;
                emit(b, s.pend);
                continue;
            }
            const int m = h_m_[b];
            if (m < 0 || m > k_in) {
                throw std::runtime_error("decode: device accept out of range");
            }
            if (FILE * cf = mtp_conf_file(); cf != nullptr && B == 1 && !ws_calibrating_) {
                fprintf(cf, "M %d %d\n", m, k_in);
            }
            acc_now.push_back(m);
            tm.mtp_drafted  += k_in;
            tm.mtp_accepted += m;
            for (int j = 0; j < k_in && j < (int) tm.mtp_verified_by_pos.size(); ++j) {
                tm.mtp_verified_by_pos[(size_t) j] += 1;
                tm.mtp_acc_by_pos[(size_t) j] += j < m ? 1 : 0;
            }
            const int32_t * g = h_g_ + b * L;
            for (int i = 0; i <= m && !s.done; ++i) {
                emit(b, g[i]);
            }
            const int64_t pos0 = s.pos;
            s.pend = g[m];
            s.pos += m + 1;
            if (df2_on() && o.drafts.empty()) {
                // AI_DF2_HOST: the verify's committed rows (the pending token + m accepted drafts) join the
                // drafter's context; the new pending token anchors the next block
                if (!s.done) {
                    df2_advance(s, m + 1, pos0, s.pend, s.pos);
                } else {
                    s.drafts.assign((size_t) k_out, s.pend);
                }
                if (!lookup_cfg().on) continue;   // else fall through: AI_LOOKUP may replace the block
            } else {
                s.drafts.resize((size_t) k_out);
                for (int j = 0; j < k_out; ++j) {
                    s.drafts[(size_t) j] = h_drafts_[j * B + b];
                }
            }
            if (const LookupCfg & lc = lookup_cfg(); lc.on && B == 1 && !ws_calibrating_ && !s.hist.empty()) {
                const int l0 = std::max(1, lc.l0);
                if (s.lk_len <= 0) s.lk_len = l0;
                if (s.lk_k > 0) {   // this round verified lookup drafts: slow start grows on a full accept, resets on a miss
                    g_lk_rounds++;
                    g_lk_drafted += k_in;
                    g_lk_acc += m;
                    // lk_full: consecutive fully accepted rounds at the current length (reset whenever the length moves)
                    s.lk_full = m >= s.lk_n ? s.lk_full + 1 : 0;
                    if (m < s.lk_n) { s.lk_len = l0; s.lk_full = 0; }
                    else if (2 * s.lk_len <= lc.top || s.lk_full >= lc.hold) {
                        const int nl = std::min(K, 2 * s.lk_len);
                        if (nl != s.lk_len) s.lk_full = 0;
                        s.lk_len = nl;
                    }
                    // disarm after AI_LOOKUP_DISARM (2) rounds that accepted under a quarter of their real lookup drafts
                    if (m >= s.lk_n) s.lk_miss = 0;
                    else if (m < s.lk_n / 4 && ++s.lk_miss >= lc.disarm) { s.lk_armed = false; s.lk_run = 0; s.lk_miss = 0; }
                }
                g_lk_all_rounds++;
                for (int i = 0; i <= m; ++i) s.hist.push_back(g[i]);
                lk_extend(s, lc.nmin, lc.nmax);
                static thread_local std::vector<int32_t> pr;
                // AI_LOOKUP_ARM shadow: advance the tracked continuation over this round's emitted tokens; a mismatch ends
                // the run. A fresh continuation is taken once the old one is spent. Arms at lc.arm matched tokens.
                const int arm_n = df2_on() && lc.arm_df2 >= 0 ? lc.arm_df2 : lc.arm;
                if (arm_n > 0 && !s.lk_armed) {
                    for (int i = 0; i <= m; ++i) {
                        if (s.lk_shi < (int) s.lk_sh.size() && s.lk_sh[(size_t) s.lk_shi] == g[i]) { s.lk_shi++; s.lk_run++; }
                        else { s.lk_sh.clear(); s.lk_shi = 0; s.lk_run = 0; break; }
                    }
                    if (s.lk_run >= arm_n) s.lk_armed = true;
                    if (s.lk_shi >= (int) s.lk_sh.size()) {
                        s.lk_shi = 0;
                        if (lk_propose(s, lc.nmin, lc.nmax, 64, pr) > 0) s.lk_sh = pr; else s.lk_sh.clear();
                        if (s.lk_sh.empty()) s.lk_run = 0;
                    }
                }
                const bool armed = arm_n <= 0 || s.lk_armed;
                // exact verify holds to 4 columns: lookup drafts stay within K (<= 3 there). AI_EXACT_ANY_K=1 (a
                // batch-invariant ggml build, MMVQ pinned to 1..8 columns) lifts that to K, the widest exact verify.
                static const bool exact_any_k = getenv("AI_EXACT_ANY_K") && atoi(getenv("AI_EXACT_ANY_K")) != 0;
                const int cap = cfg_.exact_verify ? std::min(K, exact_any_k ? K : 3) : std::min(K, s.lk_len);
                s.lk_k = 0;
                if (armed && lk_propose(s, lc.nmin, lc.nmax, cap, pr) > 0 && (int) pr.size() >= lc.minlen &&
                    [&]() {   // AI_LOOKUP_AGREE=N: the first N lookup tokens must equal the drafter's (0 = off)
                        const int n = std::min<int>({ lc.agree, (int) pr.size(), (int) s.drafts.size() });
                        return lc.agree <= 0 || (n > 0 && std::equal(pr.begin(), pr.begin() + n, s.drafts.begin()));
                    }()) {
                    // bucket the width (few captured profiles); pad with the last proposal (a padded draft is rejected)
                    const int w = cfg_.exact_verify ? std::max((int) pr.size(), k_out) : lk_bucket((int) pr.size(), k_out, K);
                    s.lk_n = (int) pr.size();
                    while ((int) pr.size() < w) pr.push_back(pr.back());
                    s.drafts.assign(pr.begin(), pr.begin() + w);
                    s.lk_k = w;
                }
            }
            if (FILE * dt = draft_trace_file(); dt != nullptr && !ws_calibrating_) {
                fprintf(dt, "R %d %lld %d", R0 + b, (long long) pos0, m);
                for (int i = 0; i <= m; ++i) fprintf(dt, " %d", g[i]);
                fprintf(dt, " |");
                for (int j = 0; j < k_out; ++j) fprintf(dt, " %d", s.drafts[(size_t) j]);
                fprintf(dt, "\n");
            }
        }
        tm.decode_rounds++;
        tm.k_steps[(size_t) k_out]++;
        tm.host_ms_per_round += ms_between(r0, r1);
        if (mtp && cfg_.mtp_adapt) {
            observe(acc_now, k_in, k_out, warm ? ms_between(r0, r2) : -1.0);
        }
        k_in = k_out;
        // AI_MTP_GATE=tau: confidence-gated verify width. The next round verifies the leading drafts while their
        // cumulative confidence stays >= tau; the chain still drafts k_out. Applies at any batch (the widest gated
        // width over the live streams). Every verify width <= 4 is exact on the default build, so this does not
        // affect --mtp-exact's guarantee.
        static const double k_gate     = getenv("AI_MTP_GATE") ? atof(getenv("AI_MTP_GATE")) : 0.0;
        static const int    k_gate_min = getenv("AI_MTP_GATE_MIN") ? atoi(getenv("AI_MTP_GATE_MIN")) : 0;
        if (mtp && !warm) {   // a round whose profile was recorded or captured inside it (decode-timer cost)
            g_cold_rounds++;
            g_cold_ms += ms_between(r0, r2);
            tm.cold_rounds++;
            tm.cold_ms += ms_between(r0, r2);
        }
        static const bool k_gate_b1 = getenv("AI_MTP_GATE_B1") != nullptr;   // gate at batch 1 only
        if (mtp && k_gate > 0.0 && (B == 1 || !k_gate_b1) && !df2_on() && (int) o.confs.size() >= k_out &&
            !ws_calibrating_) {
            // A round is batch-rigid: every stream verifies the same k_in columns. So the round takes the widest
            // gated width over the live streams; a column is cut only when no stream expects to use it.
            int w = 0;
            for (int b = 0; b < B; ++b) {
                if (st[(size_t) b].done) continue;
                // a lookup round's drafts are not the MTP head's: its confidences say nothing about them,
                // so the gate keeps the lookup width (exact mode caps it at 3 already)
                if (st[(size_t) b].lk_k > 0) { w = std::max(w, k_out); continue; }
                double p  = 1.0;
                int    wb = 0;
                for (int j = 0; j < k_out; ++j) {
                    p *= (double) h_conf_[j * B + b];
                    if (p < k_gate) break;
                    ++wb;
                }
                w = std::max(w, wb);
            }
            w = std::max(std::min(k_gate_min, k_out), w);
            k_in = w;
            // counts live in a function static: the engine is gone by the time atexit runs
            static long long hist[9] = {0};
            hist[k_in]++;
            static bool armed = false;
            if (!armed) {
                armed = true;
                atexit([]() {
                    long long n = 0;
                    for (long long c : hist) n += c;
                    fprintf(stderr, "[mtp-gate] tau %.2f: verify width k_in 0..8 =", k_gate);
                    for (int i = 0; i <= 8; ++i) if (hist[i]) fprintf(stderr, " %d:%lld", i, hist[i]);
                    fprintf(stderr, " (rounds %lld; cold rounds %lld = %.2f ms)\n", n, g_cold_rounds, g_cold_ms);
                });
            }
        }
        // AI_LOOKUP: a lookup proposal replaces MTP's drafts for the next round: verify all of them (no gate)
        if (mtp && B == 1 && st[0].lk_k > 0) {
            k_in = st[0].lk_k;
            static bool armed = false;
            if (!armed) {
                armed = true;
                atexit([]() {
                    fprintf(stderr, "[lookup] rounds %lld of %lld verified lookup drafts: drafted %lld, accepted %lld (%.1f%%)\n",
                            g_lk_rounds, g_lk_all_rounds, g_lk_drafted, g_lk_acc,
                            g_lk_drafted ? 100.0 * (double) g_lk_acc / (double) g_lk_drafted : 0.0);
                });
            }
        }
        return live_streams_;
}

std::vector<std::vector<int32_t>> Engine::generate_batch(const std::vector<std::vector<int32_t>> & prompts,
                                                         int64_t n_gen,
                                                         const std::function<bool(int, int32_t)> & on_token,
                                                         Timing * t) {
    const int B = (int) prompts.size();
    if (B < 1 || B > cfg_.max_batch) {
        throw std::runtime_error("generate: stream count outside 1..max_batch");
    }
    const int     K   = cfg_.mtp_k;
    const int64_t R   = prog_.kv().region_cells();
    for (const auto & p : prompts) {
        if (p.empty()) {
            throw std::runtime_error("generate: empty prompt");
        }
        if ((int64_t) p.size() + n_gen + 2 * K + 4 > R) {
            throw std::runtime_error("generate: prompt + n_gen + draft window exceeds the per-stream context");
        }
    }
    reset_streams(B);
    active_batch_ = B;
    step_ms_.clear();
    Timing local;
    Timing & tm = t ? *t : local;
    tm.batch = B;
    // decode_round takes a sink and does not own the stopping rule, so generate_batch imposes n_gen
    // here. emit_token pushes the token before calling the sink, so out.size() counts the token just
    // emitted.
    const std::function<bool(int, int32_t)> sink = [&](int b, int32_t tok) {
        return on_token(b, tok) && (int64_t) streams_[(size_t) b].out.size() < n_gen;
    };

    const auto t0 = clock_t_::now();
    for (int b = 0; b < B; ++b) {
        prefill_stream(b, prompts[(size_t) b], tm);
    }
    const auto t1 = clock_t_::now();
    for (int b = 0; b < B; ++b) {
        emit_token(b, streams_[(size_t) b].pend, sink, tm);
    }
    in_decode_ = true;

    const uint64_t block0 = dev_.blocking_calls();
    while (live_streams_ > 0) {
        decode_round(sink, tm);
    }

    const auto t2 = clock_t_::now();

    std::vector<std::vector<int32_t>> outs;
    for (auto & s : streams_) {
        tm.generated_n += (int64_t) s.out.size();
        outs.push_back(std::move(s.out));
    }
    tm.prefill_s             = ms_between(t0, t1) / 1e3;
    tm.decode_s              = ms_between(t1, t2) / 1e3;
    tm.decode_blocking_calls = dev_.blocking_calls() - block0;
    if (tm.decode_rounds > 0) {
        tm.host_ms_per_round /= (double) tm.decode_rounds;
    }
    return outs;
}

}  // namespace ai
