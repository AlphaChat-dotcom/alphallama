// Layer-major prefill for long prompts in doorbell mode.
//
// The doorbell path runs rounds of <= kDoorbellT rows, so a long prompt takes thousands of rounds. Token-major large
// chunks do not fix it: every chunk touches nearly all experts of every layer, so each would re-stream all the
// non-resident ones. Here a super-chunk of up to AI_Q4X_PREFILL_SUPER tokens is processed layer by layer: the
// residual streams of all its rows stay on the device (R_all), and each layer runs its sub-chunks (AI_Q4X_PREFILL_SUB
// rows) in order. A non-resident expert is copied into the layer's scratch bank the first time a sub-chunk routes to
// it, so it crosses PCIe at most once per layer per super-chunk. The MoE then runs as two masked mul_mat_id passes --
// resident slot bank and scratch bank -- plus the shared expert (moe_streamed's arithmetic). Attention / GDN / PLE state
// advances sub-chunk by sub-chunk in token order inside each layer, which is all the recurrences need.
//
// FAST mode: the MoE runs through the wide grouped expert kernel (or ggml's MUL_MAT_ID with AI_Q4X_PF_MMQ), not the
// doorbell's per-round kernel + CPU, so it is not bitwise equal to the kDoorbellT-row path; validate by greedy-token
// agreement and paired NLL (--score uses it too).
#include "models/qwen4exp_run.h"

#include "ggml.h"
#include "kernels/kernels.h"
#include "ops/adapter.h"
#include "ops/q4x_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ai {

static int64_t env_i64(const char * k, int64_t d) { const char * e = getenv(k); return e ? atoll(e) : d; }

// sum_j exps[:, j, t] * w[j, t] + shared as one native kernel (kern::moe_wsum, shared with decode) instead of ggml's
// mul + (k - 1) adds + add, which is k + 2 launches per slice, each streaming the full activation.
// AI_Q4X_PF_WSUM_NATIVE=0 restores the ggml chain (default 1).
static Tensor pf_wsum(Device & d, const Tensor & exps, const Tensor & w, const Tensor & shared, int k) {
    static const bool native = !getenv("AI_Q4X_PF_WSUM_NATIVE") || atoi(getenv("AI_Q4X_PF_WSUM_NATIVE")) != 0;
    if (!native || !exps.contiguous() || !w.contiguous() || !shared.contiguous()) return op::q4x::moe_weighted_sum(d, exps, w, shared, k);
    const int64_t N = exps.ne[0], T = exps.ne[2];
    Tensor        out = d.ws_alloc(F32, N, T);
    const float * ep = (const float *) exps.data, * we = (const float *) w.data, * sh = (const float *) shared.data;
    float *       op_ = (float *) out.data;
    Device *      dv = &d;
    adapter::native(d, [=]() { kern::moe_wsum(adapter::stream(*dv), ep, we, sh, op_, N, k, T); });
    return out;
}

// AI_Q4X_PF_TIMING=1: host wall time of the layer-major prefill by phase (s, printed at exit). "route sync" is the
// sel read-back: it also waits for the GPU work queued before it.
namespace {
struct PfTiming {
    bool   on = getenv("AI_Q4X_PF_TIMING") && atoi(getenv("AI_Q4X_PF_TIMING")) != 0;
    double inputs = 0, ple = 0, embed = 0, mixer = 0, route_sync = 0, plan = 0, copies = 0, moe_enq = 0, tail = 0, mtp = 0, logits = 0;
    ~PfTiming() {
        if (!on) return;
        fprintf(stderr, "[q4x-pf-timing] s: inputs %.3f | ple %.3f | embed %.3f | mixer enqueue %.3f | route sync %.3f | "
                        "plan+uploads %.3f | copy issue %.3f | moe enqueue %.3f | scatter+copy %.3f | mtp fill %.3f | logits %.3f%c",
                inputs, ple, embed, mixer, route_sync, plan, copies, moe_enq, tail, mtp, logits, 10);
    }
};
PfTiming g_pf;
struct PfClock {
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    void lap(double & acc) {
        const auto n = std::chrono::steady_clock::now();
        if (g_pf.on) acc += std::chrono::duration<double>(n - t).count();
        t = n;
    }
};
}  // namespace

// Persistent device buffers for one super-chunk of up to S rows (allocated on first use): residual streams, PLE rows,
// token ids, imrope positions, write cells and the per-sub-chunk causal masks; plus the expert scratch bank sized for
// the largest layer's (E - n_slots) experts.
bool Qwen4ExpRun::prefill_alloc(int64_t S, int64_t sub) {
    if (pf_S_ >= S && pf_sub_ == sub) return true;
    const Qwen4ExpSpec & s = prog_.spec();
    const int64_t HW = s.hc_dim(), nsub = (S + sub - 1) / sub;
    size_t bytes = 0;
    auto take = [&](size_t b) { const size_t o = bytes; bytes = (bytes + b + 255) / 256 * 256; return o; };
    pf_off_R_    = take((size_t) HW * S * 4);
    pf_off_ple_  = take(s.ple_layers.empty() ? 0 : (size_t) s.n_embd * S * 4);
    pf_off_ids_  = take((size_t) S * 4);
    pf_off_pos4_ = take((size_t) 4 * S * 4);
    pf_off_cell_ = take((size_t) S * 8);
    // One [n_kv, sub] f16 causal mask, rewritten on the device before each attention sub-chunk (q4x_causal_mask). A
    // mask per sub-chunk would cost max_ctx * sub * 2 B each, i.e. up to a GiB at long contexts.
    pf_off_mask_ = take((size_t) max_ctx_ * sub * 2 + 256);
    (void) nsub;
    size_t scr = 0;
    const int E = s.n_expert;
    for (size_t il = 0; il < banks_.size(); ++il) {
        const auto & m = prog_.layers()[il].moe;
        const int    cap = E - banks_[il].n_slots;
        scr = std::max(scr, (size_t) cap * (m.gate_exps.nb[2] + m.up_exps.nb[2] + m.down_exps.nb[2]) + 3 * 256);
    }
    int b1 = -1, b2 = -1;   // Device::alloc_buffer throws on OOM: catch it so the fallback below is real
    try { b1 = dev_.alloc_buffer(bytes, "q4x.prefill"); } catch (const std::exception &) { b1 = -1; }
    if (b1 >= 0) { try { b2 = dev_.alloc_buffer(scr, "q4x.prefill_scratch"); } catch (const std::exception &) { b2 = -1; } }
    if (b1 < 0 || b2 < 0) {
        dev_.free_buffer(b1);   // a half-made pair is released (the caller may retry with a smaller super-chunk)
        fprintf(stderr, "[q4x-prefill] device buffers (%.0f + %.0f MiB) do not fit: a smaller super-chunk, else %lld-row rounds\n",
                bytes / 1048576.0, scr / 1048576.0, (long long) kDoorbellT);
        return false;
    }
    pf_buf_ = b1; pf_scr_ = b2; pf_S_ = S; pf_sub_ = sub;
    cudaMemset(dev_.buffer_base(b2), 0, scr);   // never-written scratch slots read as zero weights (finite, masked)
    // AI_Q4X_PF_STREAM (default 1): copy each layer's whole non-resident set on a separate copy stream, overlapping the
    // mixer phase. AI_Q4X_PF_STREAM2=1 adds a second scratch buffer for double buffering; without room, one buffer.
    static const bool want_stream = !getenv("AI_Q4X_PF_STREAM") || atoi(getenv("AI_Q4X_PF_STREAM")) != 0;
    static const bool want_double = getenv("AI_Q4X_PF_STREAM2") && atoi(getenv("AI_Q4X_PF_STREAM2")) != 0;
    if (want_stream && pf_cstream_ == nullptr) {
        if (want_double) {
            try { pf_scr2_ = dev_.alloc_buffer(scr, "q4x.prefill_scratch2"); } catch (const std::exception &) { pf_scr2_ = -1; }
            if (pf_scr2_ < 0) fprintf(stderr, "[q4x-prefill] no room for the second expert scratch (%.0f MiB): one buffer%c", scr / 1048576.0, 10);
        }
        {
            cudaStream_t cs;
            cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
            pf_cstream_ = cs;
            for (int i = 0; i < 2; ++i) {
                cudaEvent_t a, f;
                cudaEventCreateWithFlags(&a, cudaEventDisableTiming);
                cudaEventCreateWithFlags(&f, cudaEventDisableTiming);
                pf_ev_ready_[i] = a;
                pf_ev_free_[i]  = f;
            }
            pf_stream_on_ = true;
        }
    }
    fprintf(stderr, "[q4x-prefill] layer-major prefill: super-chunk %lld rows, sub-chunk %lld; %.0f MiB state + %.0f MiB "
                    "expert scratch%s%c", (long long) S, (long long) sub, bytes / 1048576.0, scr / 1048576.0,
            pf_stream_on_ ? (pf_scr2_ >= 0 ? " x2 (layer stream, double buffer)" : " (layer stream under the mixer phase)") : "", 10);
    return true;
}

// The scratch bank of layer il over buffer `buf`: gate | up | down, (E - n_slots) experts each, 256-byte aligned.
void Qwen4ExpRun::pf_bank(int il, int buf, Tensor & sg, Tensor & su, Tensor & sd) const {
    const auto & m   = prog_.layers()[(size_t) il].moe;
    const int    cap = prog_.spec().n_expert - banks_[(size_t) il].n_slots;
    char *       sb  = (char *) dev_.buffer_base(buf);
    size_t       off = 0;
    auto bank = [&](const Tensor & h) {
        Tensor t = h;
        t.buf    = buf;
        t.ne[2]  = cap;
        set_contiguous_strides(t);
        off    = (off + 255) / 256 * 256;
        t.data = sb + off;
        off   += t.nbytes();
        return t;
    };
    sg = bank(m.gate_exps);
    su = bank(m.up_exps);
    sd = bank(m.down_exps);
}

// Layer stream: queue every non-resident expert of layer il into scratch buffer il % 2 on the copy stream, after the
// compute stream is done with that buffer's previous layer; ready[il % 2] marks the copies done.
void Qwen4ExpRun::pf_stream_layer(int il) {
    const int b = pf_scr2_ >= 0 ? (il & 1) : 0;
    cudaStream_t cs = (cudaStream_t) pf_cstream_, st = (cudaStream_t) adapter::stream(dev_);
    cudaEventRecord((cudaEvent_t) pf_ev_free_[b], st);   // everything queued so far (the previous user of buffer b)
    cudaStreamWaitEvent(cs, (cudaEvent_t) pf_ev_free_[b], 0);
    Tensor sg, su, sd;
    pf_bank(il, b ? pf_scr2_ : pf_scr_, sg, su, sd);
    const auto &   m = prog_.layers()[(size_t) il].moe;
    const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
    const Tensor * dst[3] = { &sg, &su, &sd };
    static const bool pf_nocopy = getenv("AI_Q4X_PF_NOCOPY") && atoi(getenv("AI_Q4X_PF_NOCOPY")) != 0;
    const auto & nr = pf_nonres_[(size_t) il];
    for (size_t i = 0; i < nr.size() && !pf_nocopy; ++i)
        for (int r = 0; r < 3; ++r)
            cudaMemcpyAsync((char *) dst[r]->data + i * dst[r]->nb[2], (const char *) src[r]->data + (size_t) nr[i] * src[r]->nb[2],
                            src[r]->nb[2], cudaMemcpyHostToDevice, cs);
    pf_streamed_ += nr.size();
    cudaEventRecord((cudaEvent_t) pf_ev_ready_[b], cs);
    pf_buf_layer_[b] = il;
}

// At the start of layer il: single buffer -- queue layer il now (the buffer is free once the previous layer's MoE,
// already queued, has run); double -- layer il if not yet queued, then layer il + 1 into the other buffer (last used by
// layer il - 1).
void Qwen4ExpRun::pf_layer_begin(int il) {
    const int nl = (int) prog_.layers().size();
    if (pf_scr2_ < 0) {
        pf_stream_layer(il);
        return;
    }
    if (pf_buf_layer_[il & 1] != il) pf_stream_layer(il);
    if (il + 1 < nl) pf_stream_layer(il + 1);
}

// One layer's MoE over a big batch of Tb rows (fm f32 [N, Tb], stream mode's scratch; see AI_Q4X_PF_MOE_BIG): routing,
// the shared expert and the weighted sum run per slice of `slice` rows (the same op shapes as the per-slice path, so the
// same bits), while gate/up runs once over all Tb rows for each bank, so an owner's weight tiles are decoded once for
// all its ~k * Tb / E columns. Down runs per slice into an N x k x slice buffer.
Tensor Qwen4ExpRun::moe_prefill_big(int il, const Tensor & fm, int64_t slice) {
    const Qwen4ExpSpec &  s = prog_.spec();
    const Qwen4ExpLayer & L = prog_.layers()[(size_t) il];
    const SlotBank &      B = banks_[(size_t) il];
    const int     k = s.n_expert_used, E = s.n_expert;
    const int64_t N = fm.ne[0], Tb = fm.ne[1], nsl = (Tb + slice - 1) / slice;
    op::MoeParams mp{ E, k };
    PfClock ck;
    Tensor sg, su, sd;
    const int b = pf_scr2_ >= 0 ? (il & 1) : 0;
    pf_bank(il, b ? pf_scr2_ : pf_scr_, sg, su, sd);
    void * st = adapter::stream(dev_);
    if (pf_layer_ != il) {
        pf_layer_ = il;
        pf_slot_of_.assign((size_t) E, -1);
        const auto & nr = pf_nonres_[(size_t) il];
        for (size_t i = 0; i < nr.size(); ++i) pf_slot_of_[(size_t) nr[i]] = (int) i;
        pf_n_ = (int) nr.size();
        if (pf_buf_layer_[b] != il) pf_stream_layer(il);
        cudaStreamWaitEvent((cudaStream_t) st, (cudaEvent_t) pf_ev_ready_[b], 0);
    }
    auto slice_of = [&](const Tensor & t, int64_t s0, int64_t Ts) {   // columns [s0, s0 + Ts) of a contiguous [N, Tb]
        return t.view(t.ne[0], Ts, 1, 1, t.nb[1], t.nb[1] * (size_t) Ts, t.nb[1] * (size_t) Ts, t.nb[1] * (size_t) s0);
    };
    std::vector<op::MoeRoute> routes;
    for (int64_t q = 0; q < nsl; ++q) {
        const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
        routes.push_back(op::moe_route(dev_, slice_of(fm, s0, Ts), L.moe.router, mp));
    }
    ck.lap(g_pf.moe_enq);
    std::vector<int32_t> sel((size_t) (k * Tb));
    for (int64_t q = 0; q < nsl; ++q) {
        const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
        dev_.download(routes[(size_t) q].sel, sel.data() + (size_t) (s0 * k), 0, (size_t) (Ts * k) * 4);
    }
    ck.lap(g_pf.route_sync);
    // owner plans: gate/up over every column of the batch, down per slice (records split at the kernel's column cap)
    const int rec = kern::q4x_experts_wide_rec(), wcols = kern::q4x_experts_wide_cols();
    const int cap = E - B.n_slots;
    auto build = [&](int64_t c0, int64_t c1, std::vector<int32_t> & pr, int & nr_, std::vector<int32_t> & ps, int & ns_) {
        std::vector<int32_t> own_r((size_t) B.n_slots, -1), own_s((size_t) cap, -1);
        auto add = [&](std::vector<int32_t> & plan, std::vector<int32_t> & own, int & n, int slot, int col) {
            if (own[(size_t) slot] >= 0 && plan[(size_t) own[(size_t) slot] * rec + 1] >= wcols) own[(size_t) slot] = -1;
            if (own[(size_t) slot] < 0) {
                own[(size_t) slot] = n++;
                plan.resize((size_t) n * rec, 0);
                plan[(size_t) own[(size_t) slot] * rec] = slot;
            }
            int32_t * r = plan.data() + (size_t) own[(size_t) slot] * rec;
            r[2 + r[1]++] = col;
        };
        nr_ = ns_ = 0;
        for (int64_t cc = c0; cc < c1; ++cc) {
            const int32_t e = sel[(size_t) cc];
            const int32_t r = B.table_h[(size_t) e];
            if (r >= 0) add(pr, own_r, nr_, r, (int) cc);
            else        add(ps, own_s, ns_, pf_slot_of_[(size_t) e], (int) cc);
        }
    };
    std::vector<int32_t> gr, gs;
    int ngr = 0, ngs = 0;
    build(0, k * Tb, gr, ngr, gs, ngs);
    std::vector<std::vector<int32_t>> dr((size_t) nsl), ds((size_t) nsl);
    std::vector<int> ndr((size_t) nsl), nds((size_t) nsl);
    for (int64_t q = 0; q < nsl; ++q) {
        const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
        build(s0 * k, (s0 + Ts) * k, dr[(size_t) q], ndr[(size_t) q], ds[(size_t) q], nds[(size_t) q]);
    }
    auto up_plan = [&](const std::vector<int32_t> & v) {
        Tensor t = dev_.ws_alloc(I32, std::max<int64_t>(1, (int64_t) v.size()));
        if (!v.empty()) dev_.upload(t, v.data(), 0, v.size() * 4);
        return t;
    };
    const Tensor tgr = up_plan(gr), tgs = up_plan(gs);
    std::vector<Tensor> tdr, tds;
    for (int64_t q = 0; q < nsl; ++q) { tdr.push_back(up_plan(dr[(size_t) q])); tds.push_back(up_plan(ds[(size_t) q])); }
    ck.lap(g_pf.plan);
    const int64_t K = fm.ne[0], F = B.up.ne[1];
    Tensor scr = dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(K, F, k, Tb) / 4) + 64, 1);
    Tensor Y   = dev_.ws_alloc(F32, N, Tb);
    Tensor ex  = dev_.ws_alloc(F32, N, k, std::min(slice, Tb));
    struct Bank { int tg, td; const void * u, * g, * d; int64_t rbg, ebg, rbd, ebd; };
    const Bank br{ (int) B.up.type, (int) B.down.type, B.up.data, B.gate.data, B.down.data, (int64_t) B.up.nb[1],
                   (int64_t) B.up.nb[2], (int64_t) B.down.nb[1], (int64_t) B.down.nb[2] };
    const Bank bs{ (int) su.type, (int) sd.type, su.data, sg.data, sd.data, (int64_t) su.nb[1], (int64_t) su.nb[2],
                   (int64_t) sd.nb[1], (int64_t) sd.nb[2] };
    const float * px = (const float *) fm.data;
    void *        psc = scr.data;
    Device *      dv  = &dev_;
    const int32_t * pgr = (const int32_t *) tgr.data, * pgs = (const int32_t *) tgs.data;
    adapter::native(dev_, [=]() {
        void * stv = adapter::stream(*dv);
        for (int pass = 0; pass < 2; ++pass) {
            const Bank & bk = pass == 0 ? br : bs;
            if (!kern::q4x_experts_wide_gu(stv, bk.tg, bk.u, bk.g, bk.rbg, bk.ebg, bk.td, bk.d, bk.rbd, bk.ebd, px, K, pass == 0,
                                           pass == 0 ? pgr : pgs, pass == 0 ? ngr : ngs, K, F, N, k, Tb, psc)) {
                fprintf(stderr, "[q4x-prefill] big-batch gate/up declined a shape%c", 10);
                abort();
            }
        }
        kern::q4x_experts_wide_qact(stv, K, F, k, Tb, psc);
    });
    float * pex = (float *) ex.data;
    for (int64_t q = 0; q < nsl; ++q) {
        const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
        const int32_t * pdr = (const int32_t *) tdr[(size_t) q].data, * pds = (const int32_t *) tds[(size_t) q].data;
        const int ndr_q = ndr[(size_t) q], nds_q = nds[(size_t) q];
        const int64_t col_lo = s0 * k;
        adapter::native(dev_, [=]() {
            void * stv = adapter::stream(*dv);
            for (int pass = 0; pass < 2; ++pass) {
                const Bank & bk = pass == 0 ? br : bs;
                if (!kern::q4x_experts_wide_dn(stv, bk.tg, bk.u, bk.g, bk.rbg, bk.ebg, bk.td, bk.d, bk.rbd, bk.ebd,
                                               pass == 0 ? pdr : pds, pass == 0 ? ndr_q : nds_q, K, F, N, k, Tb, psc, pex, col_lo)) {
                    fprintf(stderr, "[q4x-prefill] big-batch down declined a shape%c", 10);
                    abort();
                }
            }
        });
        Device::Scope sc(dev_);   // the slice's temporaries end with it (Y, ex and the routes live outside)
        const Tensor xs  = slice_of(fm, s0, Ts);
        const Tensor exs = ex.view(N, k, Ts, 1, ex.nb[1], ex.nb[2], ex.nb[2] * (size_t) Ts, 0);
        const Tensor sh  = op::moe_shared(dev_, xs, L.moe);
        const Tensor y   = pf_wsum(dev_, exs, routes[(size_t) q].weights, sh, k);
        op::copy_into(dev_, y, slice_of(Y, s0, Ts), "q4x pf big moe out");
    }
    ck.lap(g_pf.moe_enq);
    return Y;
}

// One layer's MoE for one sub-chunk: route (host read-back), stream any non-resident routed expert into the layer's
// scratch bank, then resident pass + scratch pass (masked, distinct fillers in each row) + the shared expert.
Tensor Qwen4ExpRun::moe_prefill(int il, const Tensor & x) {
    const Qwen4ExpSpec &  s = prog_.spec();
    const Qwen4ExpLayer & L = prog_.layers()[(size_t) il];
    const SlotBank &      B = banks_[(size_t) il];
    const int     k = s.n_expert_used, E = s.n_expert;
    const int64_t T = x.ne[1];
    op::MoeParams mp{ E, k };
    PfClock ck;
    op::MoeRoute  route = op::moe_route(dev_, x, L.moe.router, mp);
    ck.lap(g_pf.moe_enq);
    std::vector<int32_t> sel((size_t) (k * T));
    dev_.download(route.sel, sel.data(), 0, sel.size() * 4);
    ck.lap(g_pf.route_sync);
    // the scratch bank: this layer's tensors over pf_scr_ (stream mode: buffer il % 2), (E - n_slots) experts each
    const auto & m   = L.moe;
    const int    cap = E - B.n_slots;
    Tensor sg, su, sd;
    pf_bank(il, pf_stream_now_ && pf_scr2_ >= 0 && (il & 1) ? pf_scr2_ : pf_scr_, sg, su, sd);
    void * st = adapter::stream(dev_);
    const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
    const Tensor * dst[3] = { &sg, &su, &sd };
    if (pf_stream_now_) {
        // Layer stream: the whole non-resident set of this layer is (being) copied into buffer il % 2 by the copy
        // stream; slots are fixed (pf_nonres_ order). The first sub-chunk of the layer makes the compute stream wait for
        // it, and starts the next layer's copies into the other buffer (free once the previous layer's MoE is done).
        if (pf_layer_ != il) {
            pf_layer_ = il;
            pf_slot_of_.assign((size_t) E, -1);
            const auto & nr = pf_nonres_[(size_t) il];
            for (size_t i = 0; i < nr.size(); ++i) pf_slot_of_[(size_t) nr[i]] = (int) i;
            pf_n_ = (int) nr.size();
            const int b = pf_scr2_ >= 0 ? (il & 1) : 0;
            if (pf_buf_layer_[b] != il) pf_stream_layer(il);   // not queued at the layer's start (should not happen)
            cudaStreamWaitEvent((cudaStream_t) st, (cudaEvent_t) pf_ev_ready_[b], 0);
        }
    } else if (pf_layer_ != il) {   // a new layer: its scratch starts empty
        pf_layer_ = il;
        pf_slot_of_.assign((size_t) E, -1);
        pf_n_ = 0;
        // The scratch still holds the previous layer's experts in that layer's types: a filler column (weight 0)
        // reading them can decode garbage scales -> inf/NaN, and 0 * NaN poisons the sum. Fillers only use slots
        // < max(pf_n_, k), so zeroing the first k slots in this layer's layout keeps every filler finite.
        for (int r = 0; r < 3; ++r) cudaMemsetAsync(dst[r]->data, 0, (size_t) k * dst[r]->nb[2], (cudaStream_t) st);
    }
    // AI_Q4X_PF_NOCOPY=1 (profiling only, wrong output): assign scratch slots but skip the expert copies, which
    // exposes the prefill's compute + host-sync floor without the PCIe stream
    static const bool pf_nocopy = getenv("AI_Q4X_PF_NOCOPY") && atoi(getenv("AI_Q4X_PF_NOCOPY")) != 0;
    for (int32_t e : sel) {
        if (pf_stream_now_) break;
        if (B.table_h[(size_t) e] >= 0 || pf_slot_of_[(size_t) e] >= 0) continue;
        if (pf_n_ >= cap) throw std::runtime_error("q4x prefill: scratch bank full");
        const int sl = pf_n_++;
        pf_slot_of_[(size_t) e] = sl;
        for (int r = 0; r < 3 && !pf_nocopy; ++r) {
            cudaMemcpyAsync((char *) dst[r]->data + (size_t) sl * dst[r]->nb[2],
                            (const char *) src[r]->data + (size_t) e * src[r]->nb[2], src[r]->nb[2],
                            cudaMemcpyHostToDevice, (cudaStream_t) st);
        }
        pf_streamed_++;
    }
    ck.lap(g_pf.copies);
    // Per row: the resident pass takes hits, the scratch pass the rest; each fills its other columns with distinct
    // unused slots of its own bank at weight 0 (ggml-cuda's MUL_MAT_ID MMQ faults on repeated ids in a row).
    static const bool pf_mmq = getenv("AI_Q4X_PF_MMQ") && atoi(getenv("AI_Q4X_PF_MMQ")) != 0;
    const bool use_mmq = pf_mmq && B.up.type != GGML_TYPE_IQ2_XS;
    std::vector<int32_t> s_res((size_t) (k * T)), s_scr((size_t) (k * T));
    std::vector<float>   m_res((size_t) (k * T)), m_scr((size_t) (k * T));
    std::vector<char>    used_r((size_t) B.n_slots), used_s((size_t) std::max(cap, k));
    const int scr_valid = std::max(pf_n_, k);   // slots 0..k-1 exist (zeroed) even before k experts were streamed
    // Only the MMQ path reads these, so they are built only there: the loop is host time spent with the GPU idle
    // behind the route read-back.
    for (int64_t row = 0; row < (use_mmq ? T : 0); ++row) {
        const size_t r0 = (size_t) (row * k);
        std::fill(used_r.begin(), used_r.end(), 0);
        std::fill(used_s.begin(), used_s.end(), 0);
        for (int j = 0; j < k; ++j) {
            const int32_t e = sel[r0 + (size_t) j];
            if (B.table_h[(size_t) e] >= 0) used_r[(size_t) B.table_h[(size_t) e]] = 1;
            else                            used_s[(size_t) pf_slot_of_[(size_t) e]] = 1;
        }
        int nr = 0, ns = 0;
        for (int j = 0; j < k; ++j) {
            const int32_t e   = sel[r0 + (size_t) j];
            const bool    res = B.table_h[(size_t) e] >= 0;
            if (res) {
                s_res[r0 + (size_t) j] = B.table_h[(size_t) e]; m_res[r0 + (size_t) j] = 1.0f;
                while (ns < scr_valid && used_s[(size_t) ns]) ++ns;
                s_scr[r0 + (size_t) j] = ns; used_s[(size_t) ns] = 1; m_scr[r0 + (size_t) j] = 0.0f;
            } else {
                s_scr[r0 + (size_t) j] = pf_slot_of_[(size_t) e]; m_scr[r0 + (size_t) j] = 1.0f;
                while (nr < B.n_slots && used_r[(size_t) nr]) ++nr;
                s_res[r0 + (size_t) j] = nr; used_r[(size_t) nr] = 1; m_res[r0 + (size_t) j] = 0.0f;
            }
        }
    }
    // Both passes run the grouped expert kernel over the whole sub-chunk (q4x_experts_wide, kernels/q4x_experts.cu) with
    // an owner plan built here from the host routing, not ggml's MUL_MAT_ID: above 8 columns that is MMQ, and on Ampere
    // the IQ2_XS MUL_MAT_ID MMQ path can return non-finite values for a real routed expert (MMVQ at T <= 8 is clean).
    // Per (row, column) the grouped kernel computes what MMVQ does; one launch per pass reads each expert tile once,
    // where narrow slices would re-read them per slice. Each pass writes only its own columns of a shared buffer.
    // AI_Q4X_PF_MMQ=1 (experimental, default 0): layers whose gate/up type is not IQ2_XS keep ggml's MUL_MAT_ID (MMQ
    // above 8 columns, tensor cores) -- two raw passes with distinct fillers, then a select takes each column from the
    // bank holding its expert. Types other than IQ2_XS have not been validated on this path, hence the default.
    if (use_mmq) {
        std::vector<int32_t> use_r((size_t) (k * T));
        for (size_t c = 0; c < use_r.size(); ++c) use_r[c] = m_res[c] > 0.5f ? 1 : 0;
        Tensor tr = dev_.ws_alloc(I32, k, T), ts = dev_.ws_alloc(I32, k, T), ur2 = dev_.ws_alloc(I32, k * T);
        dev_.upload(tr, s_res.data(), 0, s_res.size() * 4);
        dev_.upload(ts, s_scr.data(), 0, s_scr.size() * 4);
        dev_.upload(ur2, use_r.data(), 0, use_r.size() * 4);
        const Tensor  a2 = op::q4x::moe_experts_raw(dev_, x, B.up, B.gate, B.down, tr);
        const Tensor  b2 = op::q4x::moe_experts_raw(dev_, x, su, sg, sd, ts);
        const int64_t N2 = a2.ne[0];
        Tensor        e2 = dev_.ws_alloc(F32, N2, k, T);
        const float * qa = (const float *) a2.data, * qb = (const float *) b2.data;
        const int32_t * qu = (const int32_t *) ur2.data;
        float *       qe = (float *) e2.data;
        Device *      dv2 = &dev_;
        adapter::native(dev_, [=]() { kern::select_cols(adapter::stream(*dv2), qa, qb, qu, qe, N2, (int64_t) k * T); });
        return pf_wsum(dev_, e2, route.weights, op::moe_shared(dev_, x, L.moe), k);
    }
    const int rec = kern::q4x_experts_wide_rec();
    const int wcols = kern::q4x_experts_wide_cols();
    std::vector<int32_t> use_res((size_t) (k * T));
    std::vector<int32_t> own_r((size_t) B.n_slots, -1), own_s((size_t) cap, -1);   // slot -> owner record index
    std::vector<int32_t> plan_r, plan_s;
    int                  n_r = 0, n_s = 0;
    auto add = [&](std::vector<int32_t> & plan, std::vector<int32_t> & own, int & n, int slot, int col) {
        // a slot with more than wcols columns (a popular expert over a big MoE batch) continues in a new record
        if (own[(size_t) slot] >= 0 && plan[(size_t) own[(size_t) slot] * rec + 1] >= wcols) own[(size_t) slot] = -1;
        if (own[(size_t) slot] < 0) {
            own[(size_t) slot] = n++;
            plan.resize((size_t) n * rec, 0);
            plan[(size_t) own[(size_t) slot] * rec] = slot;
        }
        int32_t * r = plan.data() + (size_t) own[(size_t) slot] * rec;
        r[2 + r[1]++] = col;
    };
    for (int64_t c = 0; c < k * T; ++c) {   // ascending columns: each owner's list comes out sorted
        const int32_t e = sel[(size_t) c];
        const int32_t r = B.table_h[(size_t) e];
        use_res[(size_t) c] = r >= 0 ? 1 : 0;
        if (r >= 0) add(plan_r, own_r, n_r, r, (int) c);
        else        add(plan_s, own_s, n_s, pf_slot_of_[(size_t) e], (int) c);
    }
    (void) s_res; (void) s_scr; (void) m_res; (void) m_scr;
    Tensor pr_t = dev_.ws_alloc(I32, std::max<int64_t>(1, (int64_t) plan_r.size()));
    Tensor ps_t = dev_.ws_alloc(I32, std::max<int64_t>(1, (int64_t) plan_s.size()));
    Tensor ur   = dev_.ws_alloc(I32, k * T);
    if (!plan_r.empty()) dev_.upload(pr_t, plan_r.data(), 0, plan_r.size() * 4);
    if (!plan_s.empty()) dev_.upload(ps_t, plan_s.data(), 0, plan_s.size() * 4);
    dev_.upload(ur, use_res.data(), 0, use_res.size() * 4);
    const int64_t K = x.ne[0], F = B.up.ne[1], N = B.down.ne[1];
    if (!x.contiguous()) throw std::runtime_error("q4x prefill: MoE input must be contiguous");
    ck.lap(g_pf.plan);
    // both passes write only their own columns of one buffer (no per-bank outputs + select: 3x the memory at big batches)
    Tensor ex = dev_.ws_alloc(F32, N, k, T);
    Tensor scr = dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(K, F, k, T) / 4) + 64, 1);
    struct Bank { int tg, td; const void * u, * g, * d; int64_t rbg, ebg, rbd, ebd; };
    const Bank br{ (int) B.up.type, (int) B.down.type, B.up.data, B.gate.data, B.down.data, (int64_t) B.up.nb[1],
                   (int64_t) B.up.nb[2], (int64_t) B.down.nb[1], (int64_t) B.down.nb[2] };
    const Bank bs{ (int) su.type, (int) sd.type, su.data, sg.data, sd.data, (int64_t) su.nb[1], (int64_t) su.nb[2],
                   (int64_t) sd.nb[1], (int64_t) sd.nb[2] };
    const float *   px  = (const float *) x.data;
    const int32_t * ppr = (const int32_t *) pr_t.data, * pps = (const int32_t *) ps_t.data, * pu = (const int32_t *) ur.data;
    float *         pe  = (float *) ex.data;
    void *          psc = scr.data;
    Device *        dv  = &dev_;
    const int       nr = n_r, ns = n_s;
    adapter::native(dev_, [=]() {
        void * stv = adapter::stream(*dv);
        for (int pass = 0; pass < 2; ++pass) {
            const Bank & bk = pass == 0 ? br : bs;
            if (!kern::q4x_experts_wide(stv, bk.tg, bk.u, bk.g, bk.rbg, bk.ebg, bk.td, bk.d, bk.rbd, bk.ebd, px, K,
                                        pass == 0 ? ppr : pps, pass == 0 ? nr : ns, K, F, N, k, T, psc, pe)) {
                fprintf(stderr, "[q4x-prefill] wide grouped kernel declined a shape\n");
                abort();
            }
        }
    });
    (void) pu;
    const Tensor sh = op::moe_shared(dev_, x, L.moe);
    Tensor ret = pf_wsum(dev_, ex, route.weights, sh, k);
    ck.lap(g_pf.moe_enq);
    return ret;
}

Tensor Qwen4ExpRun::prefill(const int32_t * tokens, int64_t n, std::vector<float> * all_logits) {
    const Qwen4ExpSpec & s = prog_.spec();
    admit_drain();   // the previous request's last admissions: every slot mapped again before the scratch is sized
    // AI_Q4X_PREFILL_SUPER (default 16384): rows per super-chunk. Larger is better while it fits, since every
    // super-chunk streams the non-resident experts again. AI_Q4X_PREFILL_SUB (default 256): rows per mixer sub-chunk.
    // A super-chunk that does not fit halves (down to 1024) before falling back to kDoorbellT-row rounds.
    int64_t       S   = std::max<int64_t>(8, env_i64("AI_Q4X_PREFILL_SUPER", 16384));
    const int64_t sub = std::max<int64_t>(8, env_i64("AI_Q4X_PREFILL_SUB", 256));
    S = std::min<int64_t>(S, std::max<int64_t>(sub, (n + sub - 1) / sub * sub));   // no bigger than the prompt
    bool fit = doorbell_ && n > kDoorbellT && prefill_alloc(S, sub);
    while (!fit && doorbell_ && n > kDoorbellT && S > 1024 && pf_buf_ < 0) {
        S /= 2;
        fit = prefill_alloc(S, sub);
    }
    if (!fit) {   // short prompt (or no room): kDoorbellT-row rounds
        Tensor lg;
        for (int64_t off = 0; off < n; off += kDoorbellT) {
            const int64_t T = std::min<int64_t>(kDoorbellT, n - off);
            lg = chunk(tokens + off, T);
            if (all_logits) {
                const int64_t V = s.n_vocab;
                const size_t  o = all_logits->size();
                all_logits->resize(o + (size_t) (V * T));
                dev_.download(lg, all_logits->data() + o, 0, (size_t) (V * T) * 4);
            }
        }
        return lg;
    }
    const int64_t HW = s.hc_dim(), V = s.n_vocab;
    char * pb = (char *) dev_.buffer_base(pf_buf_);
    auto raw = [&](size_t off, int type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        Tensor t; t.buf = pf_buf_; t.type = type; t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2;
        set_contiguous_strides(t); t.data = pb + off; return t;
    };
    const auto & layers = prog_.layers();
    Tensor lg;
    for (int64_t s0 = 0; s0 < n; s0 += S) {
        const int64_t SC = std::min(S, n - s0), nsub = (SC + sub - 1) / sub;
        mtp_flush();
        const int64_t pos0 = (int64_t) hist_.size();
        if (pos0 + SC > max_ctx_) throw std::runtime_error("Qwen4ExpRun::prefill: context full");
        hist_.insert(hist_.end(), tokens + s0, tokens + s0 + SC);
        dev_.synchronize();
        dev_.ws_reset(0);
        PfClock pc;
        // host inputs for every sub-chunk, uploaded once
        std::vector<int32_t>  pos4((size_t) (4 * SC));
        std::vector<int64_t>  cells((size_t) SC);
        for (int64_t c = 0; c < nsub; ++c) {
            const int64_t c0 = c * sub, T = std::min(sub, SC - c0), p0 = pos0 + c0;
            for (int64_t i = 0; i < T; ++i) {
                const int32_t p = (int32_t) (p0 + i);
                pos4[(size_t) (4 * c0 + i)] = p; pos4[(size_t) (4 * c0 + T + i)] = p;
                pos4[(size_t) (4 * c0 + 2 * T + i)] = p; pos4[(size_t) (4 * c0 + 3 * T + i)] = 0;
                cells[(size_t) (c0 + i)] = p0 + i;
            }
        }
        uint16_t * const mask_dev = (uint16_t *) (pb + pf_off_mask_);
        Device * const   dvm      = &dev_;
        cudaMemcpy(pb + pf_off_ids_, tokens + s0, (size_t) SC * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pb + pf_off_pos4_, pos4.data(), pos4.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pb + pf_off_cell_, cells.data(), cells.size() * 8, cudaMemcpyHostToDevice);
        pc.lap(g_pf.inputs);
        if (!s.ple_layers.empty()) {
            std::vector<float> rows((size_t) (s.n_embd * SC));
            ple_rows(pos0, SC, rows.data());
            cudaMemcpy(pb + pf_off_ple_, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice);
        }
        pc.lap(g_pf.ple);
        const Tensor R_all = raw(pf_off_R_, F32, s.n_embd, s.hc, SC);
        auto R_of = [&](int64_t c) {
            const int64_t c0 = c * sub, T = std::min(sub, SC - c0);
            return R_all.view(s.n_embd, s.hc, T, 1, (size_t) s.n_embd * 4, (size_t) HW * 4, (size_t) HW * 4 * T,
                              (size_t) (HW * c0) * 4);
        };
        for (int64_t c = 0; c < nsub; ++c) {   // the hc streams start as copies of the token embedding
            Device::Scope sc(dev_);
            const int64_t c0 = c * sub, T = std::min(sub, SC - c0);
            const Tensor ids = raw(pf_off_ids_ + (size_t) c0 * 4, I32, T);
            op::copy_into(dev_, op::q4x::embed_hc(dev_, prog_.tok_embd(), ids, s.hc), R_of(c), "q4x pf hc_init");
        }
        pc.lap(g_pf.embed);
        pf_layer_ = -1;
        // stream whole layers only for big chunks (a short one routes to few of the non-resident experts)
        static const int64_t stream_min = env_i64("AI_Q4X_PF_STREAM_MIN", 1024);
        pf_stream_now_ = pf_stream_on_ && SC >= stream_min;
        if (pf_stream_now_) {
            pf_nonres_.assign(layers.size(), {});
            for (size_t l = 0; l < layers.size(); ++l) {
                for (int e = 0; e < s.n_expert; ++e)
                    if (banks_[l].table_h[(size_t) e] < 0) pf_nonres_[l].push_back(e);
                if ((int) pf_nonres_[l].size() > s.n_expert - banks_[l].n_slots) pf_stream_now_ = false;   // scratch too small
            }
            pf_buf_layer_[0] = pf_buf_layer_[1] = -1;
            if (pf_stream_now_ && pf_scr2_ >= 0) pf_stream_layer(0);
        }
        // Debug: AI_Q4X_PF_NANCHK=1 on sub-chunk 0 of each layer, report the first stage whose output is non-finite
        static const bool nanchk = getenv("AI_Q4X_PF_NANCHK") && atoi(getenv("AI_Q4X_PF_NANCHK")) != 0;
        bool nan_said = false;
        auto chk = [&](const char * what, int il, int64_t c, const Tensor & t) {
            if (!nanchk || nan_said || c != 0 || t.type != F32 || !t.contiguous()) return;
            std::vector<float> h((size_t) t.numel());
            dev_.download(t, h.data(), 0, h.size() * 4);
            size_t bad = 0;
            for (float v : h) bad += !std::isfinite(v);
            if (bad) {
                nan_said = true;
                fprintf(stderr, "[q4x-prefill-nan] layer %d (%s) %s: %zu of %zu values non-finite (T = %lld)\n", il,
                        layers[(size_t) il].attn ? "attn" : "gdn", what, bad, h.size(), (long long) t.ne[t.ne[2] > 1 ? 2 : 1]);
            }
        };
        for (int il = 0; il < (int) layers.size(); ++il) {
            const Qwen4ExpLayer & L = layers[(size_t) il];
            // Layer stream: the layer's non-resident experts start copying now (single buffer: after the previous
            // layer's MoE; double: the next layer as well) -- under the mixer phase below
            if (pf_stream_now_) pf_layer_begin(il);
            // phase A: the mixer of every sub-chunk (attention / GDN state advances in token order; the MoE of this
            // layer reads none of it until phase B)
            for (int64_t c = 0; c < nsub; ++c) {
                Device::Scope sc(dev_);
                const int64_t c0 = c * sub, T = std::min(sub, SC - c0), n_kv = pos0 + c0 + T;
                op::AttnInputs ain;
                ain.write_cells  = raw(pf_off_cell_ + (size_t) c0 * 8, I64, T);
                ain.pos4         = raw(pf_off_pos4_ + (size_t) (4 * c0) * 4, I32, 4 * T);
                if (L.attn) {   // in stream order: the previous sub-chunk's attention has read the buffer already
                    const int64_t mp0 = pos0 + c0;
                    adapter::native(dev_, [=]() { kern::q4x_causal_mask(adapter::stream(*dvm), mask_dev, n_kv, T, mp0); });
                }
                ain.mask         = raw(pf_off_mask_, F16, n_kv, T).reshape(n_kv, T, 1, 1);
                ain.n_kv         = n_kv;
                ain.B            = 1;
                ain.region_cells = max_ctx_;
                ain.first_cell   = 0;
                Tensor R = R_of(c);
                chk("layer input R", il, c, R);
                if (L.ple) {
                    const Tensor ple_t = raw(pf_off_ple_ + (size_t) (s.n_embd * c0) * 4, F32, s.n_embd, T);
                    R = op::q4x::ple_block(dev_, R, ple_t, L.plew, pp_, ple_[(size_t) il], false, nullptr);
                    chk("after PLE", il, c, R);
                }
                op::q4x::HcRead r = op::q4x::hc_read(dev_, R, L.hc_attn, s.rms_eps);
                chk("hc_read(attn) mixed", il, c, r.mixed);
                Tensor out = L.attn
                    ? op::q4x::attn_qsa_block(dev_, r.mixed, L.at, ap_, kv_[(size_t) il], ain, L.qsa, L.qsa_p, qsa_[(size_t) il])
                    : op::q4x::gdn_block_sigz(dev_, r.mixed, L.gdn, gp_, gdn_[(size_t) il], nullptr);
                chk("mixer out", il, c, out);
                R = op::q4x::hc_scatter(dev_, R, out, r.inject);
                chk("hc_scatter(attn)", il, c, R);
                op::copy_into(dev_, R, R_of(c), "q4x pf layer mid");
                pc.lap(g_pf.mixer);
            }
            // phase B: the MoE over batches of AI_Q4X_PF_MOE_ROWS rows (default 1024), independent of the mixer's
            // sub-chunk: each expert's weight tiles are decoded once per batch, and a small sub-chunk gives each expert
            // only ~k * T / E rows, so its weights would be decoded many times per layer
            static const int64_t moe_rows = std::max<int64_t>(sub, env_i64("AI_Q4X_PF_MOE_ROWS", 1024));
            // AI_Q4X_PF_MOE_BIG (default 4096, 0 = off): gate/up over batches of this many rows; HC read, routing,
            // shared expert, down and scatter stay per moe_rows slice (moe_prefill_big), so the output is bit-identical
            // to the per-slice path. Needs the layer stream.
            static const int64_t moe_big = env_i64("AI_Q4X_PF_MOE_BIG", 4096);
            const SlotBank & Bl = banks_[(size_t) il];
            const bool big_ok = pf_stream_now_ && moe_big > moe_rows &&
                                kern::q4x_experts_wide_split_ok((int) Bl.up.type, Bl.up.data, Bl.gate.data, (int64_t) Bl.up.nb[1],
                                                                (int64_t) Bl.up.nb[2], (int) Bl.down.type, Bl.down.data,
                                                                (int64_t) Bl.down.nb[1], (int64_t) Bl.down.nb[2], s.n_embd,
                                                                Bl.up.ne[1], Bl.down.ne[1], std::min(moe_big, SC));
            for (int64_t m0 = 0; big_ok && m0 < SC; m0 += moe_big) {
                Device::Scope sc(dev_);
                const int64_t Tb = std::min(moe_big, SC - m0);
                Tensor fm  = dev_.ws_alloc(F32, s.n_embd, Tb);
                Tensor ijb = dev_.ws_alloc(F32, s.hc, Tb);
                auto col = [](const Tensor & t, int64_t s0, int64_t Ts) {
                    return t.view(t.ne[0], Ts, 1, 1, t.nb[1], t.nb[1] * (size_t) Ts, t.nb[1] * (size_t) Ts, t.nb[1] * (size_t) s0);
                };
                auto Rrows = [&](int64_t r0, int64_t Tr) {
                    return R_all.view(s.n_embd, s.hc, Tr, 1, (size_t) s.n_embd * 4, (size_t) HW * 4, (size_t) HW * 4 * Tr,
                                      (size_t) (HW * r0) * 4);
                };
                for (int64_t h0 = 0; h0 < Tb; h0 += moe_rows) {
                    Device::Scope sh(dev_);
                    const int64_t Th = std::min(moe_rows, Tb - h0);
                    op::q4x::HcRead f = op::q4x::hc_read(dev_, Rrows(m0 + h0, Th), L.hc_ffn, s.rms_eps);
                    op::copy_into(dev_, f.mixed, col(fm, h0, Th), "q4x pf big fm");
                    op::copy_into(dev_, f.inject, col(ijb, h0, Th), "q4x pf big inject");
                }
                pc.lap(g_pf.mixer);
                const Tensor Y = moe_prefill_big(il, fm, moe_rows);
                pc.t = std::chrono::steady_clock::now();
                for (int64_t h0 = 0; h0 < Tb; h0 += moe_rows) {
                    Device::Scope sh(dev_);
                    const int64_t Th = std::min(moe_rows, Tb - h0);
                    const Tensor Rb = Rrows(m0 + h0, Th);
                    const Tensor R  = op::q4x::hc_scatter(dev_, Rb, col(Y, h0, Th), col(ijb, h0, Th));
                    op::copy_into(dev_, R, Rb, "q4x pf layer out");
                }
                pc.lap(g_pf.tail);
            }
            for (int64_t m0 = 0; !big_ok && m0 < SC; m0 += moe_rows) {
                Device::Scope sc(dev_);
                const int64_t c = m0 / sub, Tm = std::min(moe_rows, SC - m0);
                const Tensor Rb = R_all.view(s.n_embd, s.hc, Tm, 1, (size_t) s.n_embd * 4, (size_t) HW * 4, (size_t) HW * 4 * Tm,
                                             (size_t) (HW * m0) * 4);
                Tensor R = Rb;
                op::q4x::HcRead f = op::q4x::hc_read(dev_, R, L.hc_ffn, s.rms_eps);
                chk("hc_read(ffn) mixed", il, c, f.mixed);
                pc.lap(g_pf.mixer);
                Tensor y = moe_prefill(il, f.mixed);
                pc.t = std::chrono::steady_clock::now();
                chk("moe out", il, c, y);
                R = op::q4x::hc_scatter(dev_, R, y, f.inject);
                chk("hc_scatter(ffn)", il, c, R);
                op::copy_into(dev_, R, Rb, "q4x pf layer out");
                pc.lap(g_pf.tail);
            }
        }
        pc.t = std::chrono::steady_clock::now();
        // logits: the last sub-chunk always (the caller reads its last row); every sub-chunk when scoring
        for (int64_t c = all_logits ? 0 : nsub - 1; c < nsub; ++c) {
            const int64_t T = std::min(sub, SC - c * sub);
            if (c == nsub - 1) {   // outside a scope: the returned logits stay live until the next reset
                lg = op::linear(dev_, prog_.output(), op::q4x::hc_head(dev_, R_of(c), prog_.hc_head(), s.rms_eps));
                if (all_logits) {
                    const size_t o = all_logits->size();
                    all_logits->resize(o + (size_t) (V * T));
                    dev_.download(lg, all_logits->data() + o, 0, (size_t) (V * T) * 4);
                }
                continue;
            }
            Device::Scope sc(dev_);
            const Tensor l = op::linear(dev_, prog_.output(), op::q4x::hc_head(dev_, R_of(c), prog_.hc_head(), s.rms_eps));
            const size_t o = all_logits->size();
            all_logits->resize(o + (size_t) (V * T));
            dev_.download(l, all_logits->data() + o, 0, (size_t) (V * T) * 4);
        }
        pc.lap(g_pf.logits);
        if (mtp_on_) {   // the MTP head's K/V for every prompt row, in kDoorbellT-row fills (mtp_H_ holds 1 + kDoorbellT rows)
            for (int64_t r0 = 0; r0 < SC; r0 += kDoorbellT) {
                mtp_flush();
                const int64_t m = std::min<int64_t>(kDoorbellT, SC - r0);
                const Tensor src = R_all.view(HW, m, 1, 1, (size_t) HW * 4, (size_t) (HW * m) * 4, (size_t) (HW * m) * 4,
                                              (size_t) (HW * r0) * 4);
                const Tensor dst = mtp_H_.view(HW, m, 1, 1, (size_t) HW * 4, (size_t) (HW * m) * 4, (size_t) (HW * m) * 4,
                                               (size_t) HW * 4);
                op::copy_into(dev_, src, dst, "q4x pf mtp H");
                mtp_fill_p0_ = pos0 + r0;
                mtp_fill_m_  = m;
            }
        }
        pc.lap(g_pf.mtp);
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[q4x-prefill] %lld rows layer-major; %llu expert copies streamed into scratch\n", (long long) n,
                (unsigned long long) pf_streamed_);
    }
    // The prefill's device buffers (state, masks, expert scratch: around a GB) are released before decode allocates its
    // own (verify records, round graphs), since with the MTP head and its draft rows resident both may not fit. The
    // next prefill allocates them again. AI_Q4X_PF_KEEP=1 keeps them.
    static const bool keep = getenv("AI_Q4X_PF_KEEP") && atoi(getenv("AI_Q4X_PF_KEEP")) != 0;
    if (!keep) {
        dev_.free_buffer(pf_buf_);
        dev_.free_buffer(pf_scr_);
        dev_.free_buffer(pf_scr2_);
        pf_buf_ = pf_scr_ = pf_scr2_ = -1;
        pf_S_ = 0;
        pf_buf_layer_[0] = pf_buf_layer_[1] = -1;
    }
    return lg;
}

}  // namespace ai
