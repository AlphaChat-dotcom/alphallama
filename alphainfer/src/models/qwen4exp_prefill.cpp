// SPDX-License-Identifier: Apache-2.0
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
#include <thread>

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
    double inputs = 0, ple = 0, embed = 0, mixer = 0, route_sync = 0, plan = 0, copies = 0, moe_enq = 0, tail = 0, mtp = 0, logits = 0, share_wait = 0;
    ~PfTiming() {
        if (!on) return;
        fprintf(stderr, "[q4x-pf-timing] s: inputs %.3f | ple %.3f | embed %.3f | mixer enqueue %.3f | route sync %.3f | "
                        "plan+uploads %.3f | copy issue %.3f | moe enqueue %.3f | scatter+copy %.3f | mtp fill %.3f | logits %.3f | "
                        "cpu share wait %.3f%c",
                inputs, ple, embed, mixer, route_sync, plan, copies, moe_enq, tail, mtp, logits, share_wait, 10);
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
// prefill_alloc's state buffer for a super-chunk of S rows (the same takes, in the same order)
size_t Qwen4ExpRun::pf_state_bytes(int64_t S, int64_t sub) const {
    const Qwen4ExpSpec & s = prog_.spec();
    size_t bytes = 0;
    auto take = [&](size_t b) { bytes = (bytes + b + 255) / 256 * 256; };
    take((size_t) s.hc_dim() * S * 4);
    take(s.ple_layers.empty() ? 0 : (size_t) s.n_embd * S * 4);
    take((size_t) S * 4);
    take((size_t) 4 * S * 4);
    take((size_t) S * 8);
    take((size_t) max_ctx_ * sub * 2 + 256);
    return bytes;
}

int Qwen4ExpRun::pf_cap(int il) const {
    const SlotBank & B = banks_[(size_t) il];
    return prog_.spec().n_expert - (lent_layer(il) ? 0 : B.n_slots);
}

// [AI_Q4X_LEND] Lend: every lent layer's experts become non-resident (table -1, owners kept in lend_owner_) before the
// prefill writes its state and scratch over their banks. Ordered on the compute stream after the decode work that read
// them; admissions were drained by the caller (prefill's start), so no copy is landing in a lent bank.
void Qwen4ExpRun::lend_begin(size_t used) {
    if (lent_ || lend_buf_ < 0) return;
    lend_publish(true);   // [AI_Q4X_LEND_ASYNC] the previous give-back has landed and is mapped before lending again
    admit_drain();
    const int E  = prog_.spec().n_expert;
    void *    st = adapter::stream(dev_);
    // [AI_Q4X_LEND_PART, default 1] lend only the layers whose banks the prefill overwrites ([0, used) of the lent
    // buffer; banks are laid out in layer order): a short prompt's single scratch takes ~2 layers, a long prompt's
    // 3-buffer ring ~6-8. Lending all of them made every prefill stream (and give back) all lent layers' experts:
    // 5090 set A (30-token prompts) TTFT 0.19 -> 0.22 s when LEND2 grew the lent range from 4 to 6 layers.
    static const bool part = !getenv("AI_Q4X_LEND_PART") || atoi(getenv("AI_Q4X_LEND_PART")) != 0;
    const int  n_all = (int) banks_.size() - lend_from_;
    int        n     = n_all;
    if (part && used != (size_t) -1) {
        const char * lb = (const char *) dev_.buffer_base(lend_buf_);
        n = 0;
        for (int il = lend_from_; il < (int) banks_.size(); ++il)
            if ((size_t) ((const char *) banks_[(size_t) il].gate.data - lb) < used) n = il - lend_from_ + 1;
        n = std::max(1, n);
    }
    lend_n_ = n;
    lend_owner_.assign((size_t) lend_n_, {});
    for (size_t il = (size_t) lend_from_; il < (size_t) (lend_from_ + lend_n_); ++il) {
        SlotBank & B = banks_[il];
        lend_owner_[il - (size_t) lend_from_] = B.owner;
        std::fill(B.owner.begin(), B.owner.end(), -1);
        std::fill(B.table_h.begin(), B.table_h.end(), -1);
        std::fill(B.score.begin(), B.score.end(), 0.0f);
        int32_t * pin = table_pin_ + il * (size_t) E;
        memcpy(pin, B.table_h.data(), (size_t) E * 4);
        cudaMemcpyAsync(B.table.data, pin, (size_t) E * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
    lent_ = true;
}

// Give back: once every prefill kernel and scratch copy has finished, the lent banks get the same experts again (fillers
// included: slots >= n_slots hold experts 0..k-1), then the tables. Ordered before the first decode round.
void Qwen4ExpRun::lend_end() {
    if (!lent_) return;
    cudaDeviceSynchronize();   // the prefill's compute and copy streams are done with the lent memory
    const auto & layers = prog_.layers();
    const int    E  = prog_.spec().n_expert;
    void *       st = adapter::stream(dev_);
    size_t       moved = 0;
    // [AI_Q4X_LEND_ASYNC=1, default 0 until measured] the copies on a stream of their own, tables left unmapped (owners
    // -1, so no admission takes these slots) until lend_publish: the prefill returns its logits -- the first token --
    // without waiting for ~2 GB (3090: R TTFT +0.12-0.2 s per prompt over the same config without lending)
    static const bool async_back = getenv("AI_Q4X_LEND_ASYNC") && atoi(getenv("AI_Q4X_LEND_ASYNC")) != 0;
    if (async_back) {
        if (!lend_stream_) {
            cudaStreamCreateWithFlags((cudaStream_t *) &lend_stream_, cudaStreamNonBlocking);
            cudaEventCreateWithFlags((cudaEvent_t *) &lend_ev_, cudaEventDisableTiming);
        }
        for (size_t il = (size_t) lend_from_; il < (size_t) (lend_from_ + lend_n_); ++il) {
            SlotBank &                   B   = banks_[il];
            const std::vector<int32_t> & own = lend_owner_[il - (size_t) lend_from_];
            const auto &                 m   = layers[il].moe;
            const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
            const Tensor * dst[3] = { &B.gate, &B.up, &B.down };
            const int      nb     = (int) B.gate.ne[2];
            for (int sl = 0; sl < nb; ++sl) {
                const int32_t e = sl < B.n_slots ? own[(size_t) sl] : sl - B.n_slots;
                if (e < 0) continue;
                for (int r = 0; r < 3; ++r) {
                    cudaMemcpyAsync((char *) dst[r]->data + (size_t) sl * dst[r]->nb[2], (const char *) src[r]->data + (size_t) e * src[r]->nb[2],
                                    src[r]->nb[2], cudaMemcpyHostToDevice, (cudaStream_t) lend_stream_);
                    moved += src[r]->nb[2];
                }
            }
        }
        cudaEventRecord((cudaEvent_t) lend_ev_, (cudaStream_t) lend_stream_);
        lend_back_ = true;
        lent_      = false;
        static bool said_a = false;
        if (!said_a) {
            said_a = true;
            fprintf(stderr, "[q4x] lend: %d layers' experts copied back ASYNC after the prefill (%.0f MiB; mapped when landed)%c",
                    lend_n_, moved / 1048576.0, 10);
        }
        return;
    }
    for (size_t il = (size_t) lend_from_; il < (size_t) (lend_from_ + lend_n_); ++il) {
        SlotBank &                   B   = banks_[il];
        const std::vector<int32_t> & own = lend_owner_[il - (size_t) lend_from_];
        const auto &                 m   = layers[il].moe;
        const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
        const Tensor * dst[3] = { &B.gate, &B.up, &B.down };
        const int      nb     = (int) B.gate.ne[2];
        for (int sl = 0; sl < nb; ++sl) {
            const int32_t e = sl < B.n_slots ? own[(size_t) sl] : sl - B.n_slots;
            if (e < 0) continue;
            for (int r = 0; r < 3; ++r) {
                cudaMemcpyAsync((char *) dst[r]->data + (size_t) sl * dst[r]->nb[2], (const char *) src[r]->data + (size_t) e * src[r]->nb[2],
                                src[r]->nb[2], cudaMemcpyHostToDevice, (cudaStream_t) st);
                moved += src[r]->nb[2];
            }
            if (sl < B.n_slots) { B.owner[(size_t) sl] = e; B.table_h[(size_t) e] = sl; }
        }
        int32_t * pin = table_pin_ + il * (size_t) E;
        memcpy(pin, B.table_h.data(), (size_t) E * 4);
        cudaMemcpyAsync(B.table.data, pin, (size_t) E * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
    lent_ = false;
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[q4x] lend: %d layers' experts copied back after the prefill (%.0f MiB)%c", lend_n_,
                moved / 1048576.0, 10);
    }
}

// [AI_Q4X_LEND_ASYNC] map the lent layers again once their give-back copies have landed (wait = block for them); the
// tables go up on the compute stream, ordered before the next round
void Qwen4ExpRun::lend_publish(bool wait) {
    if (!lend_back_) return;
    if (wait) cudaEventSynchronize((cudaEvent_t) lend_ev_);
    else if (cudaEventQuery((cudaEvent_t) lend_ev_) != cudaSuccess) return;
    const int E  = prog_.spec().n_expert;
    void *    st = adapter::stream(dev_);
    for (size_t il = (size_t) lend_from_; il < (size_t) (lend_from_ + lend_n_); ++il) {
        SlotBank &                   B   = banks_[il];
        const std::vector<int32_t> & own = lend_owner_[il - (size_t) lend_from_];
        for (int sl = 0; sl < B.n_slots; ++sl) {
            const int32_t e = own[(size_t) sl];
            if (e < 0) continue;
            B.owner[(size_t) sl] = e;
            B.table_h[(size_t) e] = sl;
        }
        int32_t * pin = table_pin_ + il * (size_t) E;
        memcpy(pin, B.table_h.data(), (size_t) E * 4);
        cudaMemcpyAsync(B.table.data, pin, (size_t) E * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
    lend_back_ = false;
}

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
    size_t scr = 0, lscr = 0;   // lscr: with the lent layers' experts all non-resident
    for (size_t il = 0; il < banks_.size(); ++il) {
        const auto & m   = prog_.layers()[il].moe;
        const size_t per = (size_t) (m.gate_exps.nb[2] + m.up_exps.nb[2] + m.down_exps.nb[2]);
        scr  = std::max(scr, (size_t) (s.n_expert - banks_[il].n_slots) * per + 3 * 256);
        lscr = std::max(lscr, (size_t) ((int) il >= lend_from_ ? s.n_expert : s.n_expert - banks_[il].n_slots) * per + 3 * 256);
    }
    const size_t soff = (bytes + 255) / 256 * 256;
    // ring size this super-chunk will use (the layer stream double/triple-buffers only from the stream minimum on)
    static const int  nbuf_env = getenv("AI_Q4X_PF_NBUF") ? atoi(getenv("AI_Q4X_PF_NBUF")) : 2;
    static const bool lend2_env = !getenv("AI_Q4X_LEND2") || atoi(getenv("AI_Q4X_LEND2")) != 0;
    const bool    stream_env = (!getenv("AI_Q4X_PF_STREAM") || atoi(getenv("AI_Q4X_PF_STREAM")) != 0) &&
                               (!getenv("AI_Q4X_PF_STREAM2") || atoi(getenv("AI_Q4X_PF_STREAM2")) != 0);
    const int64_t smin_env = env_i64("AI_Q4X_PF_STREAM_MIN", env_i64("AI_Q4X_PF_SHARE", 1) != 0 && miss_ ? 3072 : 1024);
    const size_t  lscr_a   = (lscr + 255) / 256 * 256;
    int nb_want = lend2_env && stream_env && S >= smin_env ? std::max(1, std::min(3, nbuf_env)) : 1;
    while (nb_want > 1 && soff + (size_t) nb_want * lscr_a > lend_bytes_) --nb_want;
    if (lend_buf_ >= 0 && soff + lscr <= lend_bytes_) {
        // [AI_Q4X_LEND] the state and the scratch over the lent layers' banks (lend_end copies their experts back)
        lend_begin(soff + (size_t) nb_want * lscr_a);
        pf_buf_ = pf_scr_ = lend_buf_; pf_buf_off_ = 0; pf_scr_off_ = soff; pf_S_ = S; pf_sub_ = sub;
        scr = lscr;
        dev_.mem_report("prefill_alloc (lent)");
        cudaMemsetAsync((char *) dev_.buffer_base(lend_buf_) + soff, 0, scr, (cudaStream_t) adapter::stream(dev_));
        // [AI_Q4X_LEND2] the second scratch of the double-buffered layer stream behind the first, when the lent buffer holds it
        static const bool lend2 = !getenv("AI_Q4X_LEND2") || atoi(getenv("AI_Q4X_LEND2")) != 0;
        const size_t soff2 = (soff + lscr + 255) / 256 * 256;
        if (lend2 && nb_want >= 2 && pf_scr2_ < 0 && soff2 + lscr <= lend_bytes_) {
            pf_scr2_     = lend_buf_;
            pf_scr2_off_ = soff2;
            pf_nb_       = 2;
            cudaMemsetAsync((char *) dev_.buffer_base(lend_buf_) + soff2, 0, lscr, (cudaStream_t) adapter::stream(dev_));
            static const int nbuf = getenv("AI_Q4X_PF_NBUF") ? atoi(getenv("AI_Q4X_PF_NBUF")) : 2;
            const size_t soff3 = (soff2 + lscr + 255) / 256 * 256;
            if (nbuf >= 3 && nb_want >= 3 && soff3 + lscr <= lend_bytes_) {
                pf_scr3_off_ = soff3;
                pf_nb_       = 3;
                cudaMemsetAsync((char *) dev_.buffer_base(lend_buf_) + soff3, 0, lscr, (cudaStream_t) adapter::stream(dev_));
            }
        }
    } else {
        int b1 = -1, b2 = -1;   // Device::alloc_buffer throws on OOM: catch it so the fallback below is real
        try { b1 = dev_.alloc_buffer(bytes, "q4x.prefill"); } catch (const std::exception &) { b1 = -1; }
        if (b1 >= 0) { try { b2 = dev_.alloc_buffer(scr, "q4x.prefill_scratch"); } catch (const std::exception &) { b2 = -1; } }
        if (b1 < 0 || b2 < 0) {
            dev_.free_buffer(b1);   // a half-made pair is released (the caller may retry with a smaller super-chunk)
            fprintf(stderr, "[q4x-prefill] device buffers (%.0f + %.0f MiB) do not fit: a smaller super-chunk, else %lld-row rounds\n",
                    bytes / 1048576.0, scr / 1048576.0, (long long) kDoorbellT);
            return false;
        }
        pf_buf_ = b1; pf_scr_ = b2; pf_buf_off_ = pf_scr_off_ = 0; pf_S_ = S; pf_sub_ = sub;
        dev_.mem_report("prefill_alloc");
        cudaMemset(dev_.buffer_base(b2), 0, scr);   // never-written scratch slots read as zero weights (finite, masked)
    }
    // AI_Q4X_PF_STREAM (default 1): copy each layer's whole non-resident set on a separate copy stream, overlapping the
    // mixer phase. AI_Q4X_PF_STREAM2=1 adds a second scratch buffer for double buffering; without room, one buffer.
    static const bool want_stream = !getenv("AI_Q4X_PF_STREAM") || atoi(getenv("AI_Q4X_PF_STREAM")) != 0;
    // AI_Q4X_PF_STREAM2 (default 1): the second scratch buffer, allocated per prefill and only when it fits with
    // AI_Q4X_PF_STREAM2_MARGIN_MB (default 768) of VRAM left. The auto slot count does not reserve it, so a long prompt's
    // bigger state can leave no room (one buffer then). With the plans pulled from mapped memory, the next layer's
    // copies run under this layer's MoE; the output bits do not change.
    static const bool    want_double = !getenv("AI_Q4X_PF_STREAM2") || atoi(getenv("AI_Q4X_PF_STREAM2")) != 0;
    static const int64_t s2_margin   = env_i64("AI_Q4X_PF_STREAM2_MARGIN_MB", 768) << 20;
    // (the staged walk, below the stream minimum, never reads it)
    const int64_t s2_min = env_i64("AI_Q4X_PF_STREAM_MIN", env_i64("AI_Q4X_PF_SHARE", 1) != 0 && miss_ ? 3072 : 1024);
    const int     E = s.n_expert;
    if (want_stream && want_double && pf_scr2_ < 0 && pf_spare_P_ == 0 && S >= s2_min) {
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        if ((int64_t) fb >= (int64_t) scr + s2_margin) {
            try { pf_scr2_ = dev_.alloc_buffer(scr, "q4x.prefill_scratch2"); } catch (const std::exception &) { pf_scr2_ = -1; }
            if (pf_scr2_ >= 0) pf_nb_ = 2;
        }
        if (pf_scr2_ < 0) {
            // AI_Q4X_PF_SPARE (default 1): no room for a second full buffer, so two spare buffers of P experts each (P
            // from the free VRAM, leaving AI_Q4X_PF_SPARE_MARGIN_MB, default 256). A layer's first P non-resident experts go to spare
            // il % 2, copied while the previous layer's MoE runs; the rest to the main buffer after it (the link is idle
            // under the MoE otherwise). The MoE reads three banks: resident, main, spare.
            static const bool    want_spare = !getenv("AI_Q4X_PF_SPARE") || atoi(getenv("AI_Q4X_PF_SPARE")) != 0;
            static const int64_t sp_margin  = env_i64("AI_Q4X_PF_SPARE_MARGIN_MB", 256) << 20;
            size_t per = 0;
            for (size_t il = 0; il < banks_.size(); ++il) {
                const auto & m = prog_.layers()[il].moe;
                per = std::max(per, (size_t) (m.gate_exps.nb[2] + m.up_exps.nb[2] + m.down_exps.nb[2]));
            }
            cudaMemGetInfo(&fb, &tb);
            const int64_t room = (int64_t) fb - sp_margin;
            int P = room > 0 && per > 0 ? (int) std::min<int64_t>(room / (int64_t) (2 * per + 2048), E) : 0;
            if (want_spare && P >= 16) {
                const size_t sb = (size_t) P * per + 3 * 256;
                try {
                    pf_spare_[0] = dev_.alloc_buffer(sb, "q4x.prefill_spare0");
                    pf_spare_[1] = dev_.alloc_buffer(sb, "q4x.prefill_spare1");
                    pf_spare_P_  = P;
                    cudaMemset(dev_.buffer_base(pf_spare_[0]), 0, sb);
                    cudaMemset(dev_.buffer_base(pf_spare_[1]), 0, sb);
                } catch (const std::exception &) {
                    dev_.free_buffer(pf_spare_[0]); dev_.free_buffer(pf_spare_[1]);
                    pf_spare_[0] = pf_spare_[1] = -1; pf_spare_P_ = 0;
                }
            }
            fprintf(stderr, "[q4x-prefill] no room for the second expert scratch (%.0f MiB, %.0f MiB free): %s%c", scr / 1048576.0,
                    fb / 1048576.0, pf_spare_P_ > 0 ? "two spare buffers instead" : "one buffer", 10);
            if (pf_spare_P_ > 0) fprintf(stderr, "[q4x-prefill] spare buffers: %d experts each (%.0f MiB)%c", pf_spare_P_,
                                         (double) pf_spare_P_ * per / 1048576.0, 10);
        }
    }
    if (want_stream && pf_cstream_ == nullptr) {
        {
            cudaStream_t cs;
            cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking);
            pf_cstream_ = cs;
            for (int i = 0; i < 3; ++i) {
                cudaEvent_t a, f;
                cudaEventCreateWithFlags(&a, cudaEventDisableTiming);
                cudaEventCreateWithFlags(&f, cudaEventDisableTiming);
                pf_ev_ready_[i] = a;
                pf_ev_free_[i]  = f;
            }
            for (int i = 0; i < 3; ++i) cudaEventCreateWithFlags((cudaEvent_t *) &pf_ev_spfree_[i], cudaEventDisableTiming);
            pf_stream_on_ = true;
        }
    }
    fprintf(stderr, "[q4x-prefill] layer-major prefill: super-chunk %lld rows, sub-chunk %lld; %.0f MiB state + %.0f MiB "
                    "expert scratch%s%c", (long long) S, (long long) sub, bytes / 1048576.0, scr / 1048576.0,
            pf_stream_on_ ? (pf_nb_ == 3 ? " x3 (layer stream, 3-buffer ring)" : pf_scr2_ >= 0 ? " x2 (layer stream, double buffer)" : pf_spare_P_ > 0 ? " + 2 spares (layer stream)"
                                                                     : " (layer stream under the mixer phase)") : "", 10);
    return true;
}

// The scratch bank of layer il over buffer `buf`: gate | up | down, (E - n_slots) experts each (n > 0: n experts),
// 256-byte aligned.
void Qwen4ExpRun::pf_bank_b(int il, int b, Tensor & sg, Tensor & su, Tensor & sd) const {
    pf_bank(il, b == 0 ? pf_scr_ : b == 1 ? pf_scr2_ : lend_buf_, sg, su, sd, 0, b);
}

void Qwen4ExpRun::pf_bank(int il, int buf, Tensor & sg, Tensor & su, Tensor & sd, int n, int which) const {
    const auto & m   = prog_.layers()[(size_t) il].moe;
    const int    cap = n > 0 ? n : pf_cap(il);
    char *       sb  = (char *) dev_.buffer_base(buf) + (buf >= 0 && buf == lend_buf_ ? (which == 2 ? pf_scr3_off_ : which == 1 ? pf_scr2_off_ : pf_scr_off_) : 0);
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

// Queue experts nr[i0..i1) of the three tensors into destination slots 0..i1-i0 on cs. AI_Q4X_PF_COPY_RUNS (default 1)
// makes one copy per run of consecutive expert ids (contiguous in the source and in the scratch), so the copy
// stream's queue does not fill with one call per expert and block the host inside cudaMemcpyAsync.
static void pf_copy_experts(const Tensor * const src[3], const Tensor * const dst[3], const std::vector<int32_t> & nr,
                            size_t i0, size_t i1, cudaStream_t cs) {
    static const bool runs = !getenv("AI_Q4X_PF_COPY_RUNS") || atoi(getenv("AI_Q4X_PF_COPY_RUNS")) != 0;
    for (int r = 0; r < 3; ++r) {
        const size_t eb = src[r]->nb[2];
        const bool   merge = runs && dst[r]->nb[2] == eb;
        for (size_t i = i0; i < i1;) {
            size_t j = i + 1;
            if (merge) while (j < i1 && nr[j] == nr[j - 1] + 1) ++j;
            cudaMemcpyAsync((char *) dst[r]->data + (i - i0) * dst[r]->nb[2], (const char *) src[r]->data + (size_t) nr[i] * eb,
                            (j - i) * eb, cudaMemcpyHostToDevice, cs);
            i = j;
        }
    }
}

// Layer stream: queue every non-resident expert of layer il into scratch buffer il % 2 on the copy stream, after the
// compute stream is done with that buffer's previous layer; ready[il % 2] marks the copies done.
// AI_Q4X_PF_SPARE: part 0 = the layer's first P non-resident experts into spare il % 2 (free once layer il - 2's MoE,
// queued before this call, has run); part 1 = the rest into the main buffer (free once layer il - 1's MoE has run);
// ready[0] after part 1 (the copy stream is in order: part 0 was queued earlier).
void Qwen4ExpRun::pf_stream_part(int il, int part) {
    cudaStream_t cs = (cudaStream_t) pf_cstream_, st = (cudaStream_t) adapter::stream(dev_);
    const int    e  = part == 0 ? 1 + (il & 1) : 0;   // free-event slot: 0 main, 1 / 2 spares
    cudaEventRecord((cudaEvent_t) pf_ev_spfree_[e], st);
    cudaStreamWaitEvent(cs, (cudaEvent_t) pf_ev_spfree_[e], 0);
    Tensor sg, su, sd;
    if (part == 0) pf_bank(il, pf_spare_[il & 1], sg, su, sd, pf_spare_P_);
    else           pf_bank(il, pf_scr_, sg, su, sd);
    const auto &   m = prog_.layers()[(size_t) il].moe;
    const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
    const Tensor * dst[3] = { &sg, &su, &sd };
    const auto &   nr = pf_nonres_[(size_t) il];
    const size_t   P  = std::min(nr.size(), (size_t) pf_spare_P_);
    const size_t   i0 = part == 0 ? 0 : P, i1 = part == 0 ? P : nr.size();
    pf_copy_experts(src, dst, nr, i0, i1, cs);
    pf_streamed_ += i1 - i0;
    if (part == 1) {
        cudaEventRecord((cudaEvent_t) pf_ev_ready_[0], cs);
        pf_buf_layer_[0] = il;
    } else {
        pf_spare_layer_[il & 1] = il;
    }
}

void Qwen4ExpRun::pf_stream_layer(int il) {
    const int b = pf_b(il);
    cudaStream_t cs = (cudaStream_t) pf_cstream_, st = (cudaStream_t) adapter::stream(dev_);
    cudaEventRecord((cudaEvent_t) pf_ev_free_[b], st);   // everything queued so far (the previous user of buffer b)
    cudaStreamWaitEvent(cs, (cudaEvent_t) pf_ev_free_[b], 0);
    Tensor sg, su, sd;
    pf_bank_b(il, b, sg, su, sd);
    const auto &   m = prog_.layers()[(size_t) il].moe;
    const Tensor * src[3] = { &m.gate_exps, &m.up_exps, &m.down_exps };
    const Tensor * dst[3] = { &sg, &su, &sd };
    static const bool pf_nocopy = getenv("AI_Q4X_PF_NOCOPY") && atoi(getenv("AI_Q4X_PF_NOCOPY")) != 0;
    const auto & nr = pf_nonres_[(size_t) il];
    if (!pf_nocopy) pf_copy_experts(src, dst, nr, 0, nr.size(), cs);
    pf_streamed_ += nr.size();
    cudaEventRecord((cudaEvent_t) pf_ev_ready_[b], cs);
    pf_buf_layer_[b] = il;
}

// At the start of layer il: single buffer -- queue layer il now (the buffer is free once the previous layer's MoE,
// already queued, has run); double -- layer il if not yet queued, then layer il + 1 into the other buffer (last used by
// layer il - 1).
void Qwen4ExpRun::pf_layer_begin(int il) {
    const int nl = (int) prog_.layers().size();
    if (pf_spare_P_ > 0) {   // this layer's main part (its spare part went with the previous layer's), the next one's spare part
        if (pf_spare_layer_[il & 1] != il) pf_stream_part(il, 0);
        pf_stream_part(il, 1);
        if (il + 1 < nl) pf_stream_part(il + 1, 0);
        return;
    }
    if (pf_scr2_ < 0) {
        pf_stream_layer(il);
        return;
    }
    // ring of pf_nb_ buffers: layer il, then the next pf_nb_ - 1 layers (layer il + j reuses the buffer of layer
    // il + j - pf_nb_, whose MoE is queued before this point)
    for (int j = 0; j < pf_nb_ && il + j < nl; ++j)
        if (pf_buf_layer_[pf_b(il + j)] != il + j) pf_stream_layer(il + j);
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
    Tensor sg, su, sd, pg, pu, pd;
    const bool spare = pf_spare_P_ > 0;
    const int  b     = pf_b(il);
    pf_bank_b(il, b, sg, su, sd);
    if (spare) pf_bank(il, pf_spare_[il & 1], pg, pu, pd, pf_spare_P_);
    void * st = adapter::stream(dev_);
    if (pf_layer_ != il) {
        pf_layer_ = il;
        pf_slot_of_.assign((size_t) E, -1);
        pf_spare_of_.assign((size_t) E, -1);
        const auto & nr = pf_nonres_[(size_t) il];
        const size_t P  = spare ? std::min(nr.size(), (size_t) pf_spare_P_) : 0;
        for (size_t i = 0; i < nr.size(); ++i) {
            if (i < P) pf_spare_of_[(size_t) nr[i]] = (int) i;
            else       pf_slot_of_[(size_t) nr[i]] = (int) (i - P);
        }
        pf_n_ = (int) nr.size();
        if (spare) {
            if (pf_buf_layer_[0] != il) {
                if (pf_spare_layer_[il & 1] != il) pf_stream_part(il, 0);
                pf_stream_part(il, 1);
            }
        } else if (pf_buf_layer_[b] != il) {
            pf_stream_layer(il);
        }
        cudaStreamWaitEvent((cudaStream_t) st, (cudaEvent_t) pf_ev_ready_[spare ? 0 : b], 0);
    }
    auto slice_of = [&](const Tensor & t, int64_t s0, int64_t Ts) {   // columns [s0, s0 + Ts) of a contiguous [N, Tb]
        return t.view(t.ne[0], Ts, 1, 1, t.nb[1], t.nb[1] * (size_t) Ts, t.nb[1] * (size_t) Ts, t.nb[1] * (size_t) s0);
    };
    std::vector<op::MoeRoute> routes;
    for (int64_t q = 0; q < nsl; ++q) {
        const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
        routes.push_back(op::moe_route(dev_, slice_of(fm, s0, Ts), L.moe.router, mp));
    }
    static const bool overlap = !getenv("AI_Q4X_PF_ROUTE_OVERLAP") || atoi(getenv("AI_Q4X_PF_ROUTE_OVERLAP")) != 0;
    Tensor Y = dev_.ws_alloc(F32, N, Tb);
    // [AI_Q4X_PF_DEVPLAN=1, default 0 until gated] the owner plans built ON THE DEVICE from the routing (kern::q4x_pf_plan):
    // no route read-back, so the host never waits on the GPU inside the layer loop and enqueues the next layer's mixer
    // while this layer's MoE runs. 5090 nsys: ~48 x 2.3 ms per 2.6K R prompt of GPU idle between a layer's
    // MoE tail and the next layer's mixer, waiting for the host; Strata reads its plan from mapped memory. The same
    // columns go to the same kernels (records in slot order): identical bits. Only the default prefill kernels
    // (PF_GU 4 + PF_GU_MMQ 1, PF_DN 1) skip ncols-0 records, so the knob needs them.
    static const bool devplan = getenv("AI_Q4X_PF_DEVPLAN") && atoi(getenv("AI_Q4X_PF_DEVPLAN")) != 0 &&
                                env_i64("AI_Q4X_PF_GU", 4) == 4 && env_i64("AI_Q4X_PF_GU_MMQ", 1) != 0 && env_i64("AI_Q4X_PF_DN", 1) == 1;
    Tensor tgr, tgs, tgp;
    std::vector<Tensor> tdr, tds, tdp;
    int ngr = 0, ngs = 0, ngp = 0;
    std::vector<int> ndr, nds, ndp;
    const int rec_w = kern::q4x_experts_wide_rec(), wcols_w = kern::q4x_experts_wide_cols();
    if (devplan) {
        if (il == 0) cudaStreamSynchronize((cudaStream_t) st);   // a super-chunk's start: the previous one's plans were read
        // the shared expert (routing-independent) into Y first, as the overlap path
        for (int64_t q = 0; q < nsl; ++q) {
            Device::Scope sc(dev_);
            const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
            op::copy_into(dev_, op::moe_shared(dev_, slice_of(fm, s0, Ts), L.moe), slice_of(Y, s0, Ts), "q4x pf big shared");
        }
        ck.lap(g_pf.moe_enq);
        // the batch's routing contiguous on the device
        Tensor selt = dev_.ws_alloc(I32, k * Tb);
        for (int64_t q = 0; q < nsl; ++q) {
            const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
            cudaMemcpyAsync((int32_t *) selt.data + s0 * k, routes[(size_t) q].sel.data, (size_t) (Ts * k) * 4,
                            cudaMemcpyDeviceToDevice, (cudaStream_t) st);
        }
        // this layer's scratch / spare slot maps in a mapped pinned region of its own (the plan kernel reads them there)
        const int ncap = pf_cap(il);
        if (!pf_dmap_pin_) {
            pf_dmap_n_ = (int) prog_.layers().size();
            cudaHostAlloc((void **) &pf_dmap_pin_, (size_t) pf_dmap_n_ * 2 * E * 4, cudaHostAllocMapped);
            cudaHostGetDevicePointer((void **) &pf_dmap_dev_, pf_dmap_pin_, 0);
            cudaMalloc((void **) &pf_dplan_ovf_, 4);
            cudaMemset(pf_dplan_ovf_, 0, 4);
        }
        int32_t * mp_h = pf_dmap_pin_ + (size_t) il * 2 * E;
        memcpy(mp_h, pf_slot_of_.data(), (size_t) E * 4);
        if (spare) memcpy(mp_h + E, pf_spare_of_.data(), (size_t) E * 4);
        const int32_t * scr_d = pf_dmap_dev_ + (size_t) il * 2 * E, * spr_d = spare ? scr_d + E : nullptr;
        auto caps = [&](int64_t cols, int nslots) { return (int) std::min<int64_t>(nslots, cols) + (int) ((cols + wcols_w - 1) / wcols_w) + 1; };
        auto alloc_plan = [&](int cap) { return dev_.ws_alloc(I32, std::max<int64_t>(1, (int64_t) cap * rec_w)); };
        auto plan_range = [&](int64_t c0, int64_t c1, Tensor & pr, int & cr, Tensor & pp, int & cp, Tensor & ps, int & cs) {
            const int64_t cols = c1 - c0;
            cr = caps(cols, B.n_slots); cp = spare ? caps(cols, pf_spare_P_) : 0; cs = caps(cols, ncap);
            pr = alloc_plan(cr); pp = alloc_plan(std::max(cp, 1)); ps = alloc_plan(cs);
            if (!kern::q4x_pf_plan(st, (const int32_t *) selt.data, c0, c1, (const int32_t *) B.table.data, spr_d, scr_d,
                                   B.n_slots, pf_spare_P_, ncap, (int32_t *) pr.data, cr, (int32_t *) pp.data, cp,
                                   (int32_t *) ps.data, cs, pf_dplan_ovf_)) {
                fprintf(stderr, "[q4x-prefill] device plan declined (slots > 1024)%c", 10);
                abort();
            }
        };
        plan_range(0, k * Tb, tgr, ngr, tgp, ngp, tgs, ngs);
        tdr.resize((size_t) nsl); tds.resize((size_t) nsl); tdp.resize((size_t) nsl);
        ndr.assign((size_t) nsl, 0); nds.assign((size_t) nsl, 0); ndp.assign((size_t) nsl, 0);
        for (int64_t q = 0; q < nsl; ++q) {
            const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
            plan_range(s0 * k, (s0 + Ts) * k, tdr[(size_t) q], ndr[(size_t) q], tdp[(size_t) q], ndp[(size_t) q],
                       tds[(size_t) q], nds[(size_t) q]);
        }
        static bool said_dp = false;
        if (!said_dp) { said_dp = true; fprintf(stderr, "[q4x-prefill] owner plans built on the device (AI_Q4X_PF_DEVPLAN): no route read-back%c", 10); }
        ck.lap(g_pf.plan);
    } else {
        std::vector<int32_t> sel((size_t) (k * Tb));
        if (overlap) {
            // the routes are recorded; the shared expert (routing-independent) goes on the stream behind them, into Y, and
            // the host reads the routes on a side stream as soon as they land, building the plans while the GPU computes.
            // 3090, 2 runs, tokens identical: 16K card 9.12-9.16 -> 9.01-9.06 s
            if (!pf_rstream_) {
                cudaStream_t rs;
                cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking);
                pf_rstream_ = rs;
                cudaEvent_t ev;
                cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
                pf_ev_route_ = ev;
            }
            if (pf_sel_cap_ < sel.size()) {
                if (pf_sel_pin_) cudaFreeHost(pf_sel_pin_);
                cudaHostAlloc((void **) &pf_sel_pin_, sel.size() * 4, cudaHostAllocDefault);
                pf_sel_cap_ = sel.size();
            }
            cudaEventRecord((cudaEvent_t) pf_ev_route_, (cudaStream_t) st);
            cudaStreamWaitEvent((cudaStream_t) pf_rstream_, (cudaEvent_t) pf_ev_route_, 0);
            for (int64_t q = 0; q < nsl; ++q) {
                const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
                cudaMemcpyAsync(pf_sel_pin_ + s0 * k, routes[(size_t) q].sel.data, (size_t) (Ts * k) * 4, cudaMemcpyDeviceToHost,
                                (cudaStream_t) pf_rstream_);
            }
            for (int64_t q = 0; q < nsl; ++q) {
                Device::Scope sc(dev_);
                const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
                op::copy_into(dev_, op::moe_shared(dev_, slice_of(fm, s0, Ts), L.moe), slice_of(Y, s0, Ts), "q4x pf big shared");
            }
            ck.lap(g_pf.moe_enq);
            cudaStreamSynchronize((cudaStream_t) pf_rstream_);
            memcpy(sel.data(), pf_sel_pin_, sel.size() * 4);
        } else {
            ck.lap(g_pf.moe_enq);
            for (int64_t q = 0; q < nsl; ++q) {
                const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
                dev_.download(routes[(size_t) q].sel, sel.data() + (size_t) (s0 * k), 0, (size_t) (Ts * k) * 4);
            }
        }
        ck.lap(g_pf.route_sync);
        // owner plans: gate/up over every column of the batch, down per slice (records split at the kernel's column cap)
        const int rec = kern::q4x_experts_wide_rec(), wcols = kern::q4x_experts_wide_cols();
        const int cap = pf_cap(il);
        auto build = [&](int64_t c0, int64_t c1, std::vector<int32_t> & pr, int & nr_, std::vector<int32_t> & ps, int & ns_,
                         std::vector<int32_t> & pp, int & np_) {
            std::vector<int32_t> own_r((size_t) B.n_slots, -1), own_s((size_t) cap, -1), own_p((size_t) std::max(1, pf_spare_P_), -1);
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
            nr_ = ns_ = np_ = 0;
            for (int64_t cc = c0; cc < c1; ++cc) {
                const int32_t e = sel[(size_t) cc];
                const int32_t r = B.table_h[(size_t) e];
                if (r >= 0)                                          add(pr, own_r, nr_, r, (int) cc);
                else if (spare && pf_spare_of_[(size_t) e] >= 0)     add(pp, own_p, np_, pf_spare_of_[(size_t) e], (int) cc);
                else                                                 add(ps, own_s, ns_, pf_slot_of_[(size_t) e], (int) cc);
            }
        };
        std::vector<int32_t> gr, gs, gp;
        ngr = ngs = ngp = 0;
        build(0, k * Tb, gr, ngr, gs, ngs, gp, ngp);
        std::vector<std::vector<int32_t>> dr((size_t) nsl), ds((size_t) nsl), dp((size_t) nsl);
        ndr.assign((size_t) nsl, 0); nds.assign((size_t) nsl, 0); ndp.assign((size_t) nsl, 0);
        for (int64_t q = 0; q < nsl; ++q) {
            const int64_t s0 = q * slice, Ts = std::min(slice, Tb - s0);
            build(s0 * k, (s0 + Ts) * k, dr[(size_t) q], ndr[(size_t) q], ds[(size_t) q], nds[(size_t) q], dp[(size_t) q], ndp[(size_t) q]);
        }
        // with the overlap, kernels behind the routes are still running over workspace this call may reuse: the plans go up
        // ORDERED on the stream (Device::upload is not), from pinned staging that the next call rewrites only after its own
        // route event, i.e. after these copies ran. The staging is MAPPED and a kernel pulls it (kern::q4x_copy_i32): a
        // cudaMemcpyAsync H2D waited on the copy engine behind the stream's queued expert blobs, so the expert kernels sat
        // behind the next layer's copies (4080 4K card: kernels 1.24 s, link 2.67 s, overlapping only 0.58 s).
        size_t pin_need = 0, pin_at = 0;
        if (overlap) {
            pin_need = gr.size() + gs.size() + gp.size();
            for (int64_t q = 0; q < nsl; ++q) pin_need += dr[(size_t) q].size() + ds[(size_t) q].size() + dp[(size_t) q].size();
            if (pf_plan_cap_ < pin_need) {
                cudaStreamSynchronize((cudaStream_t) st);
                if (pf_plan_pin_) cudaFreeHost(pf_plan_pin_);
                pf_plan_cap_ = pin_need + pin_need / 4 + 1024;
                cudaHostAlloc((void **) &pf_plan_pin_, pf_plan_cap_ * 4, cudaHostAllocMapped);
                cudaHostGetDevicePointer((void **) &pf_plan_dev_, pf_plan_pin_, 0);
            }
        }
        auto up_plan = [&](const std::vector<int32_t> & v) {
            Tensor t = dev_.ws_alloc(I32, std::max<int64_t>(1, (int64_t) v.size()));
            if (v.empty()) return t;
            if (overlap) {
                memcpy(pf_plan_pin_ + pin_at, v.data(), v.size() * 4);
                kern::q4x_copy_i32(st, pf_plan_dev_ + pin_at, (int32_t *) t.data, (int64_t) v.size());
                pin_at += v.size();
            } else {
                dev_.upload(t, v.data(), 0, v.size() * 4);
            }
            return t;
        };
        tgr = up_plan(gr); tgs = up_plan(gs); tgp = up_plan(gp);
        for (int64_t q = 0; q < nsl; ++q) {
            tdr.push_back(up_plan(dr[(size_t) q])); tds.push_back(up_plan(ds[(size_t) q])); tdp.push_back(up_plan(dp[(size_t) q]));
        }
        ck.lap(g_pf.plan);
    }
    const int64_t K = fm.ne[0], F = B.up.ne[1];
    Tensor scr = dev_.ws_alloc(F32, (int64_t) (kern::q4x_experts_scratch_bytes(K, F, k, Tb) / 4) + 64, 1);
    Tensor ex  = dev_.ws_alloc(F32, N, k, std::min(slice, Tb));
    struct Bank { int tg, td; const void * u, * g, * d; int64_t rbg, ebg, rbd, ebd; };
    const Bank br{ (int) B.up.type, (int) B.down.type, B.up.data, B.gate.data, B.down.data, (int64_t) B.up.nb[1],
                   (int64_t) B.up.nb[2], (int64_t) B.down.nb[1], (int64_t) B.down.nb[2] };
    const Bank bs{ (int) su.type, (int) sd.type, su.data, sg.data, sd.data, (int64_t) su.nb[1], (int64_t) su.nb[2],
                   (int64_t) sd.nb[1], (int64_t) sd.nb[2] };
    const Bank bp = spare ? Bank{ (int) pu.type, (int) pd.type, pu.data, pg.data, pd.data, (int64_t) pu.nb[1], (int64_t) pu.nb[2],
                                  (int64_t) pd.nb[1], (int64_t) pd.nb[2] }
                          : bs;
    const int     npass = spare ? 3 : 2;
    const float * px = (const float *) fm.data;
    void *        psc = scr.data;
    Device *      dv  = &dev_;
    const int32_t * pgr = (const int32_t *) tgr.data, * pgs = (const int32_t *) tgs.data, * pgp = (const int32_t *) tgp.data;
    adapter::native(dev_, [=]() {
        void * stv = adapter::stream(*dv);
        for (int pass = 0; pass < npass; ++pass) {
            const Bank & bk = pass == 0 ? br : pass == 1 ? bs : bp;
            if (!kern::q4x_experts_wide_gu(stv, bk.tg, bk.u, bk.g, bk.rbg, bk.ebg, bk.td, bk.d, bk.rbd, bk.ebd, px, K, pass == 0,
                                           pass == 0 ? pgr : pass == 1 ? pgs : pgp, pass == 0 ? ngr : pass == 1 ? ngs : ngp,
                                           K, F, N, k, Tb, psc)) {
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
        const int32_t * pdp = (const int32_t *) tdp[(size_t) q].data;
        const int ndr_q = ndr[(size_t) q], nds_q = nds[(size_t) q], ndp_q = ndp[(size_t) q];
        const int64_t col_lo = s0 * k;
        adapter::native(dev_, [=]() {
            void * stv = adapter::stream(*dv);
            for (int pass = 0; pass < npass; ++pass) {
                const Bank & bk = pass == 0 ? br : pass == 1 ? bs : bp;
                if (!kern::q4x_experts_wide_dn(stv, bk.tg, bk.u, bk.g, bk.rbg, bk.ebg, bk.td, bk.d, bk.rbd, bk.ebd,
                                               pass == 0 ? pdr : pass == 1 ? pds : pdp, pass == 0 ? ndr_q : pass == 1 ? nds_q : ndp_q,
                                               K, F, N, k, Tb, psc, pex, col_lo)) {
                    fprintf(stderr, "[q4x-prefill] big-batch down declined a shape%c", 10);
                    abort();
                }
            }
        });
        Device::Scope sc(dev_);   // the slice's temporaries end with it (Y, ex and the routes live outside)
        const Tensor xs  = slice_of(fm, s0, Ts);
        const Tensor exs = ex.view(N, k, Ts, 1, ex.nb[1], ex.nb[2], ex.nb[2] * (size_t) Ts, 0);
        const Tensor sh  = overlap || devplan ? slice_of(Y, s0, Ts) : op::moe_shared(dev_, xs, L.moe);
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
    // AI_Q4X_PF_SHARE (default 1 when the CPU pool exists; 0 = off): the CPU share of a staged (short-prompt, <= 3072
    // rows) batch. The decode CPU pool, idle while a prompt is read, computes the non-resident experts that few of the
    // batch's tokens route to instead of copying each over PCIe; the GPU copies and computes the rest, then adds the
    // CPU's weighted rows. Every column is still computed: the same math in other kernels.
    // The split is a cost model fitted online: the CPU costs A per decode pass (a pass serves up to 4 of an expert's
    // columns; A from the pool's job times), the GPU G0 + C per copied expert (C = the expert's bytes over
    // AI_Q4X_PF_SHARE_GBS, default 12 GB/s; G0 from timing events around the GPU part, read at the next batch, when they
    // have completed). The non-resident experts sorted by column count, the CPU takes the prefix that minimises the
    // larger of the two. AI_Q4X_PF_SHARE_CMAX = n: a fixed rule instead (the CPU takes every expert with <= n columns).
    static const int    share_on  = env_i64("AI_Q4X_PF_SHARE", 1) != 0;
    static const int    share_fix = (int) env_i64("AI_Q4X_PF_SHARE_CMAX", 0);
    static const double share_bw  = (double) env_i64("AI_Q4X_PF_SHARE_GBS", 12) * 1e9;
    const bool          share     = share_on && miss_ && !pf_stream_now_ && x.contiguous();
    std::vector<char> cpu_e;
    std::vector<int32_t> cj_tok, cj_ids;
    std::vector<float>   cj_w, xh, yph, wh;
    int                  n_gpu_copy = 0, cpu_groups = 0;
    if (share) {
        if (!pf_ev_s0_) {
            cudaEventCreate((cudaEvent_t *) &pf_ev_s0_);
            cudaEventCreate((cudaEvent_t *) &pf_ev_s1_);
        }
        const size_t nl = prog_.layers().size();
        if (pf_share_Al_.size() != nl) { pf_share_Al_.assign(nl, -1.0); pf_share_G0l_.assign(nl, -1.0); }
        if (pf_share_pending_) {   // the previous shared batch's GPU part (complete: the read-back above synced)
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, (cudaEvent_t) pf_ev_s0_, (cudaEvent_t) pf_ev_s1_) == cudaSuccess) {
                const double g0 = std::max(0.0, ms * 1e-3 - pf_share_C_ * pf_share_ngpu_);
                double &     gl = pf_share_G0l_[(size_t) pf_share_il_];
                gl           = gl < 0 ? g0 : 0.6 * gl + 0.4 * g0;
                pf_share_G0_ = pf_share_G0_ < 0 ? g0 : 0.8 * pf_share_G0_ + 0.2 * g0;
            }
            pf_share_pending_ = false;
        }
        const double C = (double) (L.moe.gate_exps.nb[2] + L.moe.up_exps.nb[2] + L.moe.down_exps.nb[2]) / share_bw;
        pf_share_C_ = C;
        std::vector<int> cnt((size_t) E, 0);
        for (int32_t e : sel) ++cnt[(size_t) e];
        std::vector<std::pair<int, int>> cand;   // (columns, expert) of the routed non-resident experts
        for (int e = 0; e < E; ++e)
            if (cnt[(size_t) e] > 0 && B.table_h[(size_t) e] < 0) cand.push_back({ cnt[(size_t) e], e });
        std::sort(cand.begin(), cand.end());
        size_t take = 0;
        if (share_fix > 0) {
            while (take < cand.size() && cand[take].first <= share_fix) ++take;
        } else {
            // per layer (the layers' expert types differ, IQ2 vs IQ3 decode ~1.5x), seeded from the running global ones;
            // the CPU's estimate is biased up 15%: a CPU overrun idles the GPU at the add
            const double G0 = pf_share_G0l_[(size_t) il] >= 0 ? pf_share_G0l_[(size_t) il] : pf_share_G0_ < 0 ? 0.004 : pf_share_G0_;
            const double A  = 1.15 * (pf_share_Al_[(size_t) il] > 0 ? pf_share_Al_[(size_t) il] : pf_share_A_);
            double best = G0 + C * (double) cand.size(), cpu = 0.0;
            for (size_t i = 0; i < cand.size(); ++i) {
                cpu += A * (double) ((cand[i].first + 3) / 4);
                const double t = std::max(cpu, G0 + C * (double) (cand.size() - i - 1));
                if (t < best) { best = t; take = i + 1; }
            }
        }
        cpu_e.assign((size_t) E, 0);
        for (size_t i = 0; i < take; ++i) {
            cpu_e[(size_t) cand[i].second] = 1;
            cpu_groups += (cand[i].first + 3) / 4;
        }
        n_gpu_copy = (int) (cand.size() - take);
        wh.resize((size_t) (k * T));
        dev_.download(route.weights, wh.data(), 0, wh.size() * 4);
        for (int64_t c = 0; c < k * T; ++c) {
            const int32_t e = sel[(size_t) c];
            if (!cpu_e[(size_t) e]) continue;
            cj_tok.push_back((int32_t) (c / k));
            cj_ids.push_back(e);
            cj_w.push_back(wh[(size_t) c]);
            wh[(size_t) c] = 0.0f;   // the GPU's weighted sum leaves the CPU's columns out
        }
        if (cj_ids.empty()) {
            cpu_e.clear();
        } else {
            xh.resize((size_t) (x.ne[0] * T));
            yph.resize(xh.size());
            dev_.download(x, xh.data(), 0, xh.size() * 4);
            miss_->host_job_begin(il, xh.data(), (int) T, cj_tok.data(), cj_ids.data(), cj_w.data(), (int) cj_ids.size(),
                                  yph.data());
            pf_shared_cols_ += cj_ids.size();
        }
    }
    const bool shared = !cpu_e.empty();
    if (shared) cudaEventRecord((cudaEvent_t) pf_ev_s0_, (cudaStream_t) adapter::stream(dev_));
    ck.lap(g_pf.route_sync);
    // the scratch bank: this layer's tensors over pf_scr_ (stream mode: buffer il % 2), (E - n_slots) experts each
    const auto & m   = L.moe;
    const int    cap = pf_cap(il);
    Tensor sg, su, sd;
    pf_bank_b(il, pf_stream_now_ ? pf_b(il) : 0, sg, su, sd);
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
            const int b = pf_b(il);
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
    // [AI_Q4X_PF_COPY_OVERLAP=1, default 0 until measured] the staged walk's expert copies on a stream of their own: the
    // resident pass and the shared expert run while they land, and only the scratch pass waits for them. Serial on the
    // compute stream they were 0.36 s of a 1.85 s 2K-token prefill with the GPU idle (3090 nsys, R w_engine, 10-31).
    // The copy stream first waits for the compute stream (the new layer's scratch zeroing above; the route sync left
    // everything before it complete, so no earlier kernel still reads the scratch).
    static const bool ovl = getenv("AI_Q4X_PF_COPY_OVERLAP") && atoi(getenv("AI_Q4X_PF_COPY_OVERLAP")) != 0;
    void * cst = st;
    if (ovl && !pf_stream_now_) {
        if (!pf_ostream_) {
            cudaStreamCreateWithFlags((cudaStream_t *) &pf_ostream_, cudaStreamNonBlocking);
            cudaEventCreateWithFlags((cudaEvent_t *) &pf_ev_ocopy_, cudaEventDisableTiming);
            cudaEventCreateWithFlags((cudaEvent_t *) &pf_ev_omain_, cudaEventDisableTiming);
        }
        cudaEventRecord((cudaEvent_t) pf_ev_omain_, (cudaStream_t) st);
        cudaStreamWaitEvent((cudaStream_t) pf_ostream_, (cudaEvent_t) pf_ev_omain_, 0);
        cst = pf_ostream_;
    }
    // With the overlap the copies are only RECORDED here and issued after the plan uploads and the resident pass are
    // queued: Device::upload is a synchronous pageable copy, and on the 3090's in-order H2D engine it waited behind every
    // expert copy already issued ("plan+uploads" 0.65 s of a 1.86 s 1K-token prefill, unchanged by a second stream).
    struct DCopy { void * d; const void * s; size_t n; };
    std::vector<DCopy> dcopies;
    for (int32_t e : sel) {
        if (pf_stream_now_) break;
        if (B.table_h[(size_t) e] >= 0 || pf_slot_of_[(size_t) e] >= 0) continue;
        if (shared && cpu_e[(size_t) e]) continue;   // the CPU computes it
        if (pf_n_ >= cap) throw std::runtime_error("q4x prefill: scratch bank full");
        const int sl = pf_n_++;
        pf_slot_of_[(size_t) e] = sl;
        for (int r = 0; r < 3 && !pf_nocopy; ++r) {
            void *       dp = (char *) dst[r]->data + (size_t) sl * dst[r]->nb[2];
            const void * sp = (const char *) src[r]->data + (size_t) e * src[r]->nb[2];
            if (cst != st) dcopies.push_back({ dp, sp, (size_t) src[r]->nb[2] });
            else cudaMemcpyAsync(dp, sp, src[r]->nb[2], cudaMemcpyHostToDevice, (cudaStream_t) cst);
        }
        pf_streamed_++;
    }
    void * ocopy_ev = cst != st ? pf_ev_ocopy_ : nullptr;
    auto issue_copies = [&]() {
        for (const DCopy & c : dcopies) cudaMemcpyAsync(c.d, c.s, c.n, cudaMemcpyHostToDevice, (cudaStream_t) cst);
        cudaEventRecord((cudaEvent_t) pf_ev_ocopy_, (cudaStream_t) cst);
    };
    ck.lap(g_pf.copies);
    // Per row: the resident pass takes hits, the scratch pass the rest; each fills its other columns with distinct
    // unused slots of its own bank at weight 0 (ggml-cuda's MUL_MAT_ID MMQ faults on repeated ids in a row).
    static const bool pf_mmq = getenv("AI_Q4X_PF_MMQ") && atoi(getenv("AI_Q4X_PF_MMQ")) != 0;
    const bool use_mmq = pf_mmq && B.up.type != GGML_TYPE_IQ2_XS && !lent_layer(il);   // lent: its bank is the scratch
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
        if (shared && cpu_e[(size_t) e]) continue;   // the CPU's column: no GPU owner (its ex column is zeroed below)
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
    Tensor wz;
    if (shared) {   // the CPU's columns: zero outputs and zero weights in the GPU's sum (uploads are safe: the stream is idle)
        cudaMemsetAsync(ex.data, 0, ex.nbytes(), (cudaStream_t) adapter::stream(dev_));
        wz = dev_.ws_alloc(F32, k, T);
        dev_.upload(wz, wh.data(), 0, wh.size() * 4);
    }
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
    auto run_pass = [=](int pass) {
        adapter::native(dev_, [=]() {
            void * stv = adapter::stream(*dv);
            const Bank & bk = pass == 0 ? br : bs;
            if (pass == 1 && ocopy_ev) cudaStreamWaitEvent((cudaStream_t) stv, (cudaEvent_t) ocopy_ev, 0);
            if (!kern::q4x_experts_wide(stv, bk.tg, bk.u, bk.g, bk.rbg, bk.ebg, bk.td, bk.d, bk.rbd, bk.ebd, px, K,
                                        pass == 0 ? ppr : pps, pass == 0 ? nr : ns, K, F, N, k, T, psc, pe)) {
                fprintf(stderr, "[q4x-prefill] wide grouped kernel declined a shape\n");
                abort();
            }
        });
    };
    (void) pu;
    Tensor sh;
    if (ocopy_ev) {   // [AI_Q4X_PF_COPY_OVERLAP] resident pass, then the copies, the shared expert, the scratch pass after them
        run_pass(0);
        issue_copies();
        sh = op::moe_shared(dev_, x, L.moe);
        run_pass(1);
    } else {
        run_pass(0);
        run_pass(1);
        sh = op::moe_shared(dev_, x, L.moe);
    }
    Tensor ret = pf_wsum(dev_, ex, shared ? wz : route.weights, sh, k);
    ck.lap(g_pf.moe_enq);
    if (shared) {   // the CPU's rows, added once its job is done (an ordered copy: kernels above may still run)
        cudaEventRecord((cudaEvent_t) pf_ev_s1_, (cudaStream_t) adapter::stream(dev_));
        pf_share_pending_ = true;
        pf_share_ngpu_    = n_gpu_copy;
        pf_share_il_      = il;
        miss_->host_job_wait();
        if (cpu_groups > 0) {   // the CPU's cost per decode pass, from this job's time
            const double a  = (double) miss_->host_job_ns() * 1e-9 / (double) cpu_groups;
            double &     al = pf_share_Al_[(size_t) il];
            al          = al < 0 ? a : 0.6 * al + 0.4 * a;
            pf_share_A_ = 0.8 * pf_share_A_ + 0.2 * a;
        }
        ck.lap(g_pf.share_wait);
        Tensor yp = dev_.ws_alloc(F32, x.ne[0], T);
        cudaMemcpyAsync(yp.data, yph.data(), yph.size() * 4, cudaMemcpyHostToDevice, (cudaStream_t) adapter::stream(dev_));
        ret = op::add(dev_, ret, yp);
    }
    return ret;
}

Tensor Qwen4ExpRun::prefill(const int32_t * tokens, int64_t n, std::vector<float> * all_logits) {
    const Qwen4ExpSpec & s = prog_.spec();
    pin_launch_thread();   // [AI_Q4X_RESERVE_CORES] the thread that decodes owns the launch core
    fold_join();     // [AI_Q4X_FOLD_SIDE] the last commit's fold before the prompt advances the state
    ple_gate_idle();
    admit_drain();   // the previous request's last admissions: every slot mapped again before the scratch is sized
    // AI_Q4X_PREFILL_SUPER (default 16384): rows per super-chunk. Larger is better while it fits, since every
    // super-chunk streams the non-resident experts again. AI_Q4X_PREFILL_SUB (default 256): rows per mixer sub-chunk.
    // A super-chunk that does not fit halves (down to 1024) before falling back to kDoorbellT-row rounds.
    int64_t       S   = std::max<int64_t>(8, env_i64("AI_Q4X_PREFILL_SUPER", 16384));
    bells_live_ = false;   // the bells still hold the previous request's last round
    const int64_t sub = std::max<int64_t>(8, env_i64("AI_Q4X_PREFILL_SUB", 256));
    // a prompt a little past the super-chunk takes one bigger chunk rather than a tiny second chunk that would walk
    // every layer again and re-stream most of the non-resident experts
    if (n > S && n <= S + S / 8) S = (n + sub - 1) / sub * sub;
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
    char * pb = (char *) dev_.buffer_base(pf_buf_) + pf_buf_off_;
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
        // A cudaMemcpy from PAGEABLE memory returns once the data is staged, before its DMA lands, and the backend stream
        // that reads these buffers next is a non-blocking stream, not ordered against the default stream the copy runs on:
        // the first kernels could read stale cells. An event recorded on the default stream after the copies, and waited on
        // by the backend stream, orders them with no host stall. (Found by AI_Q4X_PF_HASH on the PLE rows, 4080 2026-10-28:
        // fresh starts diverged in layer 1, the first PLE layer, until a download between the upload and its reader.)
        static cudaEvent_t pf_up_ev = [] { cudaEvent_t e; cudaEventCreateWithFlags(&e, cudaEventDisableTiming); return e; }();
        auto order_uploads = [&]() {
            cudaEventRecord(pf_up_ev, 0);
            cudaStreamWaitEvent((cudaStream_t) adapter::stream(dev_), pf_up_ev, 0);
        };
        cudaMemcpy(pb + pf_off_ids_, tokens + s0, (size_t) SC * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pb + pf_off_pos4_, pos4.data(), pos4.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(pb + pf_off_cell_, cells.data(), cells.size() * 8, cudaMemcpyHostToDevice);
        order_uploads();
        pc.lap(g_pf.inputs);
        // AI_Q4X_PLE_OVERLAP (default 1): the super-chunk's PLE rows are gathered on a helper thread while the GPU embeds
        // and runs the layers before the first PLE layer (layer 1 here); joined and uploaded just before that layer
        // reads them. The PLE region of pf_buf_ is read by no earlier layer, so the synchronous upload cannot race the
        // queued kernels. Done serially, the gather leaves the GPU idle (AI_Q4X_PF_TIMING reports it as "ple").
        static const bool  ple_overlap = !getenv("AI_Q4X_PLE_OVERLAP") || atoi(getenv("AI_Q4X_PLE_OVERLAP")) != 0;
        std::vector<float> ple_host;
        std::thread        ple_th;
        struct Joiner {   // an exception between the start and the join must not leave a joinable thread behind
            std::thread & t;
            ~Joiner() { if (t.joinable()) t.join(); }
        } ple_joiner{ ple_th };
        bool ple_pending = false;
        auto ple_upload  = [&]() {
            if (!ple_pending) return;
            if (ple_th.joinable()) ple_th.join();
            cudaMemcpy(pb + pf_off_ple_, ple_host.data(), ple_host.size() * 4, cudaMemcpyHostToDevice);
            order_uploads();   // the PLE block's kernels on the backend stream must see the landed rows
            ple_pending = false;
        };
        if (!s.ple_layers.empty()) {
            ple_host.resize((size_t) (s.n_embd * SC));
            ple_pending = true;
            if (ple_overlap) ple_th = std::thread([this, pos0, SC, &ple_host]() { ple_rows(pos0, SC, ple_host.data()); });
            else { ple_rows(pos0, SC, ple_host.data()); ple_upload(); }
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
        // with the CPU share on, batches up to 3072 rows take the staged walk so they can hand the CPU its share;
        // above that every expert streams
        static const int64_t stream_min = env_i64("AI_Q4X_PF_STREAM_MIN", env_i64("AI_Q4X_PF_SHARE", 1) != 0 && miss_ ? 3072 : 1024);
        pf_stream_now_ = pf_stream_on_ && SC >= stream_min;
        if (pf_stream_now_) {
            pf_nonres_.assign(layers.size(), {});
            for (size_t l = 0; l < layers.size(); ++l) {
                for (int e = 0; e < s.n_expert; ++e)
                    if (banks_[l].table_h[(size_t) e] < 0) pf_nonres_[l].push_back(e);
                if ((int) pf_nonres_[l].size() > pf_cap((int) l)) pf_stream_now_ = false;   // scratch too small
            }
            pf_buf_layer_[0] = pf_buf_layer_[1] = pf_buf_layer_[2] = -1;
            pf_spare_layer_[0] = pf_spare_layer_[1] = -1;
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
        // [debug] AI_Q4X_PF_HASH=1: FNV-1a of the whole residual R after the embedding and after each layer's mixer and MoE
        // phases (synchronising downloads: slow), for finding where two starts' prefills first differ
        static const bool pf_hash = getenv("AI_Q4X_PF_HASH") && atoi(getenv("AI_Q4X_PF_HASH")) != 0;
        auto hash_R = [&](const char * what, int il) {
            if (!pf_hash) return;
            std::vector<uint32_t> h((size_t) (HW * SC));
            dev_.download(R_all, h.data(), 0, h.size() * 4);
            uint64_t x = 1469598103934665603ull;
            for (uint32_t v : h) { x ^= v; x *= 1099511628211ull; }
            fprintf(stderr, "[q4x-pf-hash] L%02d %-5s %016llx\n", il, what, (unsigned long long) x);
        };
        auto hash_t = [&](const char * what, int il, const Tensor & t) {   // any contiguous f32 tensor
            if (!pf_hash || !t.contiguous()) return;
            std::vector<uint32_t> h((size_t) t.numel());
            dev_.download(t, h.data(), 0, h.size() * 4);
            uint64_t x = 1469598103934665603ull;
            for (uint32_t v : h) { x ^= v; x *= 1099511628211ull; }
            fprintf(stderr, "[q4x-pf-hash] L%02d %-5s %016llx\n", il, what, (unsigned long long) x);
        };
        hash_R("embed", -1);
        for (int il = 0; il < (int) layers.size(); ++il) {
            const Qwen4ExpLayer & L = layers[(size_t) il];
            if (L.ple) ple_upload();   // the helper's rows, before the first layer that reads them
            // Layer stream: the layer's non-resident experts start copying now (single buffer: after the previous
            // layer's MoE; double: the next layer as well) -- under the mixer phase below
            if (pf_stream_now_) pf_layer_begin(il);
            // phase A: the mixer of every sub-chunk (attention / GDN state advances in token order; the MoE of this
            // layer reads none of it until phase B)
            // AI_Q4X_PF_HC_GROUP (default 2048 rows; 0 = per sub-chunk): the attention-side HC read and scatter over
            // groups of sub-chunks. A token's read needs only the layer's input (after its PLE), and its scatter only its
            // own mixer output, so the low-rank GEMMs run on G-row batches instead of small stream-k MMQ tiles. The mixer
            // itself still runs per sub-chunk in token order.
            static const int64_t hc_group = env_i64("AI_Q4X_PF_HC_GROUP", 2048);
            const int64_t        G        = hc_group >= 2 * sub ? hc_group / sub * sub : 0;
            for (int64_t g0 = 0; G > 0 && g0 < SC; g0 += G) {
                Device::Scope sg(dev_);
                const int64_t Gb = std::min(G, SC - g0), c_lo = g0 / sub, c_hi = (g0 + Gb + sub - 1) / sub;
                const Tensor  Rg = R_all.view(s.n_embd, s.hc, Gb, 1, (size_t) s.n_embd * 4, (size_t) HW * 4, (size_t) HW * 4 * Gb,
                                              (size_t) (HW * g0) * 4);
                if (L.ple) {   // the PLE in token order, written back: the read and the scatter both start from it
                    if (pf_hash) {
                        uint64_t x = 1469598103934665603ull;
                        for (float f : ple_host) { uint32_t v; memcpy(&v, &f, 4); x ^= v; x *= 1099511628211ull; }
                        fprintf(stderr, "[q4x-pf-hash] L%02d plhst %016llx\n", il, (unsigned long long) x);
                        hash_t("pldev", il, raw(pf_off_ple_, F32, s.n_embd, SC));
                    }
                    for (int64_t c = c_lo; c < c_hi; ++c) {
                        Device::Scope sc(dev_);
                        const int64_t c0 = c * sub, T = std::min(sub, SC - c0);
                        const Tensor  ple_t = raw(pf_off_ple_ + (size_t) (s.n_embd * c0) * 4, F32, s.n_embd, T);
                        const Tensor  R = op::q4x::ple_block(dev_, R_of(c), ple_t, L.plew, pp_, ple_[(size_t) il], false, nullptr);
                        op::copy_into(dev_, R, R_of(c), "q4x pf ple");
                    }
                }
                if (L.ple) hash_R("ple", il);
                const op::q4x::HcRead r = op::q4x::hc_read(dev_, Rg, L.hc_attn, s.rms_eps);
                hash_t("hcmix", il, r.mixed);
                Tensor                outg = dev_.ws_alloc(F32, s.n_embd, Gb);
                for (int64_t c = c_lo; c < c_hi; ++c) {
                    Device::Scope sc(dev_);
                    const int64_t c0 = c * sub, T = std::min(sub, SC - c0), n_kv = pos0 + c0 + T, o0 = c0 - g0;
                    op::AttnInputs ain;
                    ain.write_cells  = raw(pf_off_cell_ + (size_t) c0 * 8, I64, T);
                    ain.pos4         = raw(pf_off_pos4_ + (size_t) (4 * c0) * 4, I32, 4 * T);
                    if (L.attn) {
                        const int64_t mp0 = pos0 + c0;
                        adapter::native(dev_, [=]() { kern::q4x_causal_mask(adapter::stream(*dvm), mask_dev, n_kv, T, mp0); });
                    }
                    ain.mask         = raw(pf_off_mask_, F16, n_kv, T).reshape(n_kv, T, 1, 1);
                    ain.n_kv         = n_kv;
                    ain.B            = 1;
                    ain.region_cells = max_ctx_;
                    ain.first_cell   = 0;
                    const Tensor mx  = r.mixed.view(s.n_embd, T, 1, 1, r.mixed.nb[1], r.mixed.nb[1] * (size_t) T,
                                                    r.mixed.nb[1] * (size_t) T, r.mixed.nb[1] * (size_t) o0);
                    // the mixers take contiguous rows: mixed is [N, Gb] contiguous, so a column range is contiguous too
                    const Tensor out = L.attn
                        ? op::q4x::attn_qsa_block(dev_, mx, L.at, ap_, kv_[(size_t) il], ain, L.qsa, L.qsa_p, qsa_[(size_t) il])
                        : op::q4x::gdn_block_sigz(dev_, mx, L.gdn, gp_, gdn_[(size_t) il], nullptr);
                    op::copy_into(dev_, out, outg.view(s.n_embd, T, 1, 1, outg.nb[1], outg.nb[1] * (size_t) T,
                                                       outg.nb[1] * (size_t) T, outg.nb[1] * (size_t) o0), "q4x pf mixer out");
                    pc.lap(g_pf.mixer);
                }
                hash_t("mixer", il, outg);
                op::q4x::hc_scatter_into(dev_, Rg, outg, r.inject);
                pc.lap(g_pf.mixer);
            }
            for (int64_t c = 0; G == 0 && c < nsub; ++c) {
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
            hash_R("mix", il);
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
                    op::q4x::hc_scatter_into(dev_, Rb, col(Y, h0, Th), col(ijb, h0, Th));
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
                op::q4x::hc_scatter_into(dev_, Rb, y, f.inject);
                chk("hc_scatter(ffn)", il, c, Rb);
                pc.lap(g_pf.tail);
            }
            hash_R("moe", il);
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
        // AI_Q4X_MTP_FILL_ROWS (default 256): the MTP head's K/V for the prompt. The first rows go through the pending
        // fill (its H comes from the carried slot), the rest in wide chunks straight from R_all (row r's H is R_all row
        // r - 1) on the prefill-width attention with a device mask, instead of one call per kDoorbellT rows.
        static const int64_t fill_rows = env_i64("AI_Q4X_MTP_FILL_ROWS", 256);
        if (mtp_on_ && fill_rows > kDoorbellT && SC > kDoorbellT) {
            mtp_flush();
            const int64_t m0 = kDoorbellT;
            {
                const Tensor src = R_all.view(HW, m0, 1, 1, (size_t) HW * 4, (size_t) (HW * m0) * 4, (size_t) (HW * m0) * 4, 0);
                const Tensor dst = mtp_H_.view(HW, m0, 1, 1, (size_t) HW * 4, (size_t) (HW * m0) * 4, (size_t) (HW * m0) * 4,
                                               (size_t) HW * 4);
                op::copy_into(dev_, src, dst, "q4x pf mtp H");
                mtp_fill_p0_ = pos0;
                mtp_fill_m_  = m0;
                mtp_flush();
            }
            for (int64_t r0 = m0; r0 < SC; r0 += fill_rows) {
                const int64_t m = std::min<int64_t>(fill_rows, SC - r0);
                const Tensor  H = R_all.view(HW, m, 1, 1, (size_t) HW * 4, (size_t) (HW * m) * 4, (size_t) (HW * m) * 4,
                                             (size_t) (HW * (r0 - 1)) * 4);
                mtp_layer(H, hist_.data() + pos0 + r0, pos0 + r0, m, false, nullptr, true);
            }
            const Tensor last  = R_all.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, (size_t) (HW * (SC - 1)) * 4);
            const Tensor slot0 = mtp_H_.view(HW, 1, 1, 1, (size_t) HW * 4, (size_t) HW * 4, (size_t) HW * 4, 0);
            op::copy_into(dev_, last, slot0, "q4x pf mtp Hcarry");
            mtp_fill_m_ = 0;
        } else if (mtp_on_) {   // the MTP head's K/V for every prompt row, in kDoorbellT-row fills (mtp_H_ holds 1 + kDoorbellT rows)
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
    if (pf_dplan_ovf_) {   // [AI_Q4X_PF_DEVPLAN] a bank needed more records than its cap: the output is not trustworthy
        int ovf = 0;
        cudaMemcpy(&ovf, pf_dplan_ovf_, 4, cudaMemcpyDeviceToHost);
        if (ovf) { fprintf(stderr, "[q4x-prefill] FATAL: device plan overflow (a bank's records exceeded its cap)%c", 10); abort(); }
    }
    static bool said = false;
    if (!said) {
        said = true;
        fprintf(stderr, "[q4x-prefill] %lld rows layer-major; %llu expert copies streamed into scratch; %llu columns on the CPU share\n",
                (long long) n, (unsigned long long) pf_streamed_, (unsigned long long) pf_shared_cols_);
    }
    // The prefill's device buffers (state, masks, expert scratch: around a GB) are released before decode allocates its
    // own (verify records, round graphs), since with the MTP head and its draft rows resident both may not fit. The
    // next prefill allocates them again. AI_Q4X_PF_KEEP=1 keeps them.
    static const bool keep = getenv("AI_Q4X_PF_KEEP") && atoi(getenv("AI_Q4X_PF_KEEP")) != 0;
    if (lent_) {   // [AI_Q4X_LEND] the state and scratch were the lent banks: give them back (nothing to free or keep)
        lend_end();
        if (pf_buf_ == lend_buf_) pf_buf_ = -1;
        if (pf_scr_ == lend_buf_) pf_scr_ = -1;
        if (pf_scr2_ == lend_buf_) { pf_scr2_ = -1; pf_nb_ = 1; }
        pf_buf_off_ = pf_scr_off_ = pf_scr2_off_ = pf_scr3_off_ = 0;
        pf_S_ = 0;
        pf_buf_layer_[0] = pf_buf_layer_[1] = pf_buf_layer_[2] = -1;
    }
    if (!keep) {
        dev_.free_buffer(pf_buf_);
        dev_.free_buffer(pf_scr_);
        dev_.free_buffer(pf_scr2_);
        dev_.free_buffer(pf_spare_[0]);
        dev_.free_buffer(pf_spare_[1]);
        pf_buf_ = pf_scr_ = pf_scr2_ = -1;
        pf_nb_ = 1;
        pf_spare_[0] = pf_spare_[1] = -1;
        pf_spare_P_ = 0;
        pf_spare_layer_[0] = pf_spare_layer_[1] = -1;
        pf_S_ = 0;
        pf_buf_layer_[0] = pf_buf_layer_[1] = pf_buf_layer_[2] = -1;
    }
    return lg;
}

}  // namespace ai
