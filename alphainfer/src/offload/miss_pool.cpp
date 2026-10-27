#include "offload/miss_pool.h"
#include "core/host_mem.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "offload/gsq_cpu.h"

#include <algorithm>
#include <cstdlib>

#include <cuda_runtime.h>

#include <chrono>
#include <climits>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
static inline void cpu_relax() { _mm_pause(); }
#else
static inline void cpu_relax() {}
#endif

namespace ai::offload {

static constexpr int64_t kGuRows   = 64;    // gate/up rows per work item
static constexpr int64_t kDownRows = 128;   // down rows per work item

// AI_Q4X_HANDOFF_STATS: host stamps on the device globaltimer's epoch (CLOCK_REALTIME)
static bool hs_on() {
    static const bool on = getenv("AI_Q4X_HANDOFF_STATS") && atoi(getenv("AI_Q4X_HANDOFF_STATS")) != 0;
    return on;
}
static inline long long rt_ns() {
    return (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static thread_local unsigned long long tl_items = 0;   // AI_Q4X_HANDOFF_STATS: items this thread ran in the current job

static void pin_to(int cpu) {
#if defined(__linux__)
    if (cpu < 0) return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(cpu, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#else
    (void) cpu;
#endif
}

MissPool::MissPool(const std::vector<ExpertBanks> & layers, int64_t N, int k, int64_t t_max, int threads,
                   const std::vector<int> & cpus)
    : layers_(layers), N_(N), t_max_(t_max), k_(k), cpus_(cpus) {
    ggml_cpu_init();
    for (const auto & L : layers_) {
        if (!L.gate.valid()) continue;
        F_ = std::max<int64_t>(F_, L.gate.ne[1]);
    }
    // one mapped block: per layer {seq, done, n_tok} (64 B), miss [k*Tmax], x [N*Tmax], y [N*k*Tmax]
    const size_t hdr = 64, miss = (size_t) k * t_max * 4, xb = (size_t) N * t_max * 4, yb = (size_t) N * k * t_max * 4;
    const size_t wb = (size_t) k * t_max * 4, ypb = (size_t) N * t_max * 4;
    auto r256 = [](size_t v) { return (v + 255) / 256 * 256; };
    const size_t o_x = r256(hdr + miss), o_y = r256(o_x + xb), o_w = r256(o_y + yb), o_yp = r256(o_w + wb);
    const size_t per = r256(o_yp + ypb);
    // huge-page pinned + mapped (core/host_mem.h: 4 KiB pinned pages draw compaction's TLB-shootdown storms)
    mapped_mem_ = pinned_host_alloc(per * layers_.size(), /*register_cuda=*/ true);
    if (!mapped_mem_.ptr) throw std::runtime_error("MissPool: pinned mapped allocation failed");
    mapped_ = mapped_mem_.ptr;
    memset(mapped_, 0, per * layers_.size());
    char * dev_base = nullptr;
    if (cudaHostGetDevicePointer((void **) &dev_base, mapped_, 0) != cudaSuccess) {
        throw std::runtime_error("MissPool: cudaHostGetDevicePointer failed");
    }
    bells_.resize(layers_.size());
    for (size_t l = 0; l < layers_.size(); ++l) {
        char * h = (char *) mapped_ + per * l;
        char * d = dev_base + per * l;
        Bell & b = bells_[l];
        b.seq   = (volatile unsigned *) h;          b.d_seq   = (unsigned *) d;
        b.done  = (volatile unsigned *) (h + 4);    b.d_done  = (unsigned *) (d + 4);
        b.n_tok = (int32_t *) (h + 8);              b.d_n_tok = (int32_t *) (d + 8);
        b.miss  = (int32_t *) (h + hdr);            b.d_miss  = (int32_t *) (d + hdr);
        b.x     = (float *) (h + o_x);              b.d_x     = (float *) (d + o_x);
        b.y     = (float *) (h + o_y);              b.d_y     = (float *) (d + o_y);
        b.w     = (float *) (h + o_w);              b.d_w     = (float *) (d + o_w);
        b.yp    = (float *) (h + o_yp);             b.d_yp    = (float *) (d + o_yp);
        b.ring_ts = (volatile unsigned long long *) (h + 16);
        b.wend_ts = (volatile unsigned long long *) (h + 24);
        b.wbeg_ts = (volatile unsigned long long *) (h + 32);
    }
    xq_.resize((size_t) t_max * (size_t) N * 4 + 1024);
    h_.resize((size_t) k * t_max * (size_t) F_ * 2);
    hq_.resize((size_t) k * t_max * (size_t) F_ * 4 + 1024);
    cols_.reserve((size_t) k * t_max);
    groups_.reserve((size_t) k * t_max);
    prep_row_ = (ggml_cpu_q2_0_prep_size((int) F_) + 63) / 64 * 64;
    prep_.resize((size_t) k * t_max * prep_row_ + 64);
    n_workers_ = std::max(0, (cpus_.empty() ? threads : (int) cpus_.size()) - 1);
    worker_done_ = std::vector<std::atomic<int>>((size_t) n_workers_);
    for (auto & w : worker_done_) w.store(0);
    hs_wstart_ = std::vector<std::atomic<long long>>((size_t) n_workers_);
    for (auto & w : hs_wstart_) w.store(0);
    hs_wi_ = std::vector<std::atomic<unsigned long long>>((size_t) 3 * (size_t) (n_workers_ + 1));
    hs_wn_ = std::vector<std::atomic<unsigned long long>>((size_t) 3 * (size_t) (n_workers_ + 1));
    for (auto & w : hs_wi_) w.store(0);
    for (auto & w : hs_wn_) w.store(0);
    hs_done_.assign(layers_.size(), 0);
    hs_wend_.assign(layers_.size(), 0);
    if (hs_on()) hs_.reserve(1 << 20);
    for (int i = 0; i < n_workers_; ++i) threads_.emplace_back([this, i] { work_loop(i); });
    threads_.emplace_back([this] { poll_loop(); });
}

MissPool::~MissPool() {
    stop_.store(true);
    for (auto & t : threads_) t.join();
    if (hs_on()) hs_report();
    pinned_host_free(mapped_mem_);
}

// One job = the gate/up phase (items: expert group x row chunk of F) then the down phase (group x row chunk of N).
// A group is the missed columns of one expert (the round's tokens that all routed to it, <= 4 at a time): with the
// decode-once kernels (gsq_cpu.cpp) each weight row is decoded once for the whole group. AI_GSQ_CPU=0 turns them off
// (ggml-cpu vec_dot per column).
void MissPool::compute_share(int part, int n_parts) {
    static const bool k_gsq = [] { const char * e = getenv("AI_GSQ_CPU"); return !e || atoi(e) != 0; }();
    const int           l = job_layer_;
    const ExpertBanks & L = layers_[(size_t) l];
    const Bell &        b = bells_[(size_t) l];
    const int64_t K = L.gate.ne[0], F = L.gate.ne[1];
    const auto *  cg = ggml_get_type_traits_cpu((ggml_type) L.gate.type);
    const auto *  cd = ggml_get_type_traits_cpu((ggml_type) L.down.type);
    const size_t  rg = ggml_row_size((ggml_type) L.gate.type, K), rd = ggml_row_size((ggml_type) L.down.type, F);
    const size_t  xq_row = ggml_row_size(cg->vec_dot_type, K), hq_row = ggml_row_size(cd->vec_dot_type, F);
    const int     ph = job_phase_;
    // dynamic claiming balances a heterogeneous pool (P-cores ~2x the E-cores); static shares wait for the
    // slowest participant every phase
    static const bool steal = !getenv("AI_Q4X_POOL_STEAL") || atoi(getenv("AI_Q4X_POOL_STEAL")) != 0;   // default on
    if (ph == 2) {   // SwiGLU per missed column (silu(gate) * up), quantised for the down type -- one item per column
        const auto * th = ggml_get_type_traits_cpu(cd->vec_dot_type);
        for (int ci = steal ? next_item_.fetch_add(1) : part; ci < n_items_;
             ci = steal ? next_item_.fetch_add(1) : ci + n_parts) {
            ++tl_items;
            const float * gu = h_.data() + (size_t) ci * 2 * F_;
            thread_local std::vector<float> hv;   // FFN width: up to 17408 for a dense 27B
            if ((int64_t) hv.size() < F) hv.resize((size_t) F);
            float * hh = hv.data();
            for (int64_t r = 0; r < F; ++r) hh[r] = gu[r] / (1.0f + std::exp(-gu[r])) * gu[F_ + r];
            th->from_float(hh, hq_.data() + (size_t) ci * hq_row, F);
            if (L.down.type == GGML_TYPE_Q2_0) {   // the decode-once Q2_0 kernel reads a prepared activation
                ggml_cpu_q2_0_prepare(hq_.data() + (size_t) ci * hq_row, (int) F, prep_.data() + (size_t) ci * prep_row_);
            }
        }
        return;
    }
    const int64_t per_grp = ph == 0 ? (F + kGuRows - 1) / kGuRows : (N_ + kDownRows - 1) / kDownRows;
    for (int it = steal ? next_item_.fetch_add(1) : part; it < n_items_;
         it = steal ? next_item_.fetch_add(1) : it + n_parts) {
        ++tl_items;
        const Group & gr = groups_[(size_t) (it / (int) per_grp)];
        const int64_t ch = it % per_grp;
        const int32_t e  = b.miss[cols_[(size_t) gr.c0]];
        if (ph == 0) {
            const char *  G  = (const char *) L.gate.data + (size_t) e * L.gate.nb[2];
            const char *  U  = (const char *) L.up.data + (size_t) e * L.up.nb[2];
            const int64_t r0 = ch * kGuRows, r1 = std::min(F, (ch + 1) * kGuRows);
            if (k_gsq && gsq_dot_preferred(L.gate.type, gr.n) && gsq_dot_preferred(L.up.type, gr.n)) {
                const void * y[4];
                for (int i = 0; i < gr.n; ++i) y[i] = xq_.data() + (size_t) (cols_[(size_t) (gr.c0 + i)] / k_) * xq_row;
                // the group's columns are consecutive in cols_, so their h rows are 2*F_ apart
                float * gu = h_.data() + (size_t) gr.c0 * 2 * F_;
                gsq_dot(L.gate.type, (int) K, G, rg, r0, r1, y, gr.n, gu, 2 * F_);
                gsq_dot(L.up.type, (int) K, U, rg, r0, r1, y, gr.n, gu + F_, 2 * F_);
            } else {
                for (int i = 0; i < gr.n; ++i) {
                    const int    ci = gr.c0 + i;
                    const char * xq = xq_.data() + (size_t) (cols_[(size_t) ci] / k_) * xq_row;
                    float *      gu = h_.data() + (size_t) ci * 2 * F_;
                    for (int64_t r = r0; r < r1; ++r) {
                        cg->vec_dot((int) K, &gu[r], 0, G + (size_t) r * rg, 0, xq, 0, 1);
                        cg->vec_dot((int) K, &gu[F_ + r], 0, U + (size_t) r * rg, 0, xq, 0, 1);
                    }
                }
            }
        } else {
            const char *  D  = (const char *) L.down.data + (size_t) e * L.down.nb[2];
            const int64_t r0 = ch * kDownRows, r1 = std::min(N_, (ch + 1) * kDownRows);
            if (k_gsq && gsq_dot_preferred(L.down.type, gr.n)) {
                const bool   prep = L.down.type == GGML_TYPE_Q2_0;
                const void * y[4];
                for (int i = 0; i < gr.n; ++i) {
                    y[i] = prep ? (const void *) (prep_.data() + (size_t) (gr.c0 + i) * prep_row_)
                                : (const void *) (hq_.data() + (size_t) (gr.c0 + i) * hq_row);
                }
                float tmp[4 * kDownRows];   // the columns' y rows are not evenly spaced: compute, then scatter
                gsq_dot(L.down.type, (int) F, D + (size_t) r0 * rd, rd, 0, r1 - r0, y, gr.n, tmp, kDownRows);
                for (int i = 0; i < gr.n; ++i) {
                    memcpy(b.y + (size_t) cols_[(size_t) (gr.c0 + i)] * N_ + r0, tmp + (size_t) i * kDownRows,
                           (size_t) (r1 - r0) * sizeof(float));
                }
            } else {
                for (int i = 0; i < gr.n; ++i) {
                    const int    ci = gr.c0 + i;
                    const char * hq = hq_.data() + (size_t) ci * hq_row;
                    float *      y  = b.y + (size_t) cols_[(size_t) ci] * N_;
                    for (int64_t r = r0; r < r1; ++r) {
                        cd->vec_dot((int) F, &y[r], 0, D + (size_t) r * rd, 0, hq, 0, 1);
                    }
                }
            }
        }
    }
}

void MissPool::work_loop(int id) {
    if (!cpus_.empty()) pin_to(cpus_[(size_t) id + 1]);
    int seen = 0;
    while (!stop_.load(std::memory_order_relaxed)) {
        const int g = job_gen_.load(std::memory_order_acquire);
        if (g == seen) { cpu_relax(); continue; }
        seen = g;
        if (hs_on()) {
            const long long t = rt_ns();
            hs_wstart_[(size_t) id].store(t, std::memory_order_relaxed);
            const int ph = job_phase_, T = *bells_[(size_t) job_layer_].n_tok;
            tl_items = 0;
            compute_share(id + 1, n_workers_ + 1);
            if (T <= 8) hs_worker_add(id + 1, ph, tl_items, rt_ns() - t);
        } else {
            compute_share(id + 1, n_workers_ + 1);
        }
        worker_done_[(size_t) id].store(g, std::memory_order_release);
    }
}

void MissPool::poll_loop() {
    if (!cpus_.empty()) pin_to(cpus_[0]);
    std::vector<unsigned> last(layers_.size(), 0);
    int gen = 0;
    const bool hs = hs_on();
    long long  hs_lag = 0;   // the current ring's worst worker start lag over its phases
    auto run_phase = [&](int l, int phase, int items) {
        n_items_   = items;
        next_item_.store(0, std::memory_order_relaxed);   // all workers left the previous phase (done gen waited)
        job_phase_ = phase;
        job_layer_ = l;
        const long long pub = hs ? rt_ns() : 0;
        job_gen_.store(++gen, std::memory_order_release);
        tl_items = 0;
        compute_share(0, n_workers_ + 1);
        if (hs && *bells_[(size_t) l].n_tok <= 8) hs_worker_add(0, phase, tl_items, rt_ns() - pub);
        for (auto & w : worker_done_) {
            while (w.load(std::memory_order_acquire) != gen) cpu_relax();
        }
        if (hs) {
            for (auto & w : hs_wstart_) hs_lag = std::max(hs_lag, w.load(std::memory_order_relaxed) - pub);
        }
    };
    while (!stop_.load(std::memory_order_relaxed)) {
        bool any = false;
        for (size_t l = 0; l < layers_.size(); ++l) {
            const Bell &   b = bells_[l];
            const unsigned s = *b.seq;
            if (s == last[l]) continue;
            any = true;
            std::atomic_thread_fence(std::memory_order_acquire);
            const auto t0 = std::chrono::steady_clock::now();
            HsSample   hsm{};
            if (hs) {
                const long long seen_h = rt_ns();
                if (hs_reset_.exchange(false)) hs_.clear();
                hsm.layer = (int) l;
                hsm.T     = *b.n_tok;
                hsm.a     = seen_h - (long long) *b.ring_ts;
                hsm.busy  = seen_h;   // seen; turned into busy at done
                // the device's wait on this layer's previous answer finished before it rang again
                const unsigned long long we = *b.wend_ts;
                const unsigned long long wb = *b.wbeg_ts;
                if (we != hs_wend_[l] && hs_done_[l] != 0) {
                    hsm.c      = (long long) we - hs_done_[l];
                    hsm.arrive = (long long) wb - hs_done_[l];
                } else {
                    hsm.c = LLONG_MIN;
                }
                hs_wend_[l] = we;
                hs_lag = 0;
            }
            const ExpertBanks & L = layers_[l];
            const int T = *b.n_tok;
            cols_.clear();
            // Fast-mode route-tail skip (needs AI_Q4X_PARTIAL, where the device zeroes every missed column, so a
            // skipped one contributes 0): AI_Q4X_TAIL_RANK=R skips missed slots j >= R; AI_Q4X_TAIL_W=x skips missed
            // columns whose routing weight is below x.
            // AI_Q4X_TAIL_RANK defaults to 8 (10 or more disables): the router's top-k is in descending weight, so this
            // drops only missed experts at ranks 8-9, the two lightest of ten (resident ones are still computed).
            // Paired NLL is neutral within noise, for about 25% more decode speed when misses dominate.
            static const int   tail_rank = getenv("AI_Q4X_TAIL_RANK") ? atoi(getenv("AI_Q4X_TAIL_RANK")) : 8;
            static const float tail_w    = getenv("AI_Q4X_TAIL_W") ? (float) atof(getenv("AI_Q4X_TAIL_W")) : 0.0f;
            static const bool  partial_on = !getenv("AI_Q4X_PARTIAL") || atoi(getenv("AI_Q4X_PARTIAL")) != 0;
            skip_.assign((size_t) (k_ * T), 0);   // host-only: the bell's miss list stays intact for admission
            // AI_Q4X_TAIL_WIN=R (fast mode; default 7, 0 = off): skip a missed expert only when every row of the window
            // that routes it does so at rank >= R (rank by routing weight within the row, 0 = heaviest). Applied on top
            // of TAIL_RANK; paired NLL stays neutral within noise.
            static const int tail_win = getenv("AI_Q4X_TAIL_WIN") ? atoi(getenv("AI_Q4X_TAIL_WIN")) : 7;
            if (tail_win > 0 && partial_on) {
                std::vector<int> rank((size_t) (k_ * T));
                for (int t = 0; t < T; ++t) {
                    for (int j = 0; j < k_; ++j) {
                        int rnk = 0;
                        const float wj = b.w[t * k_ + j];
                        for (int i = 0; i < k_; ++i) rnk += b.w[t * k_ + i] > wj || (b.w[t * k_ + i] == wj && i < j);
                        rank[(size_t) (t * k_ + j)] = rnk;
                    }
                }
                for (int c = 0; c < k_ * T; ++c) {
                    if (b.miss[c] < 0) continue;
                    int best = k_;
                    for (int d = 0; d < k_ * T; ++d) if (b.miss[d] == b.miss[c]) best = std::min(best, rank[(size_t) d]);
                    if (best >= tail_win) skip_[(size_t) c] = 1;
                }
            }
            // Bound measurement, wrong output: AI_Q4X_SKIP_FRAC=p skips every missed expert whose id hashes below p (all
            // its columns), i.e. the CPU computes a fraction 1 - p of the misses -- the round time if a share moved off
            // the CPU
            static const double skip_frac = getenv("AI_Q4X_SKIP_FRAC") ? atof(getenv("AI_Q4X_SKIP_FRAC")) : 0.0;
            auto frac_skip = [&](int32_t e) {
                return skip_frac > 0.0 && (double) (((uint32_t) e * 2654435761u) >> 8) / (double) (1u << 24) < skip_frac;
            };
            for (int c = 0; c < k_ * T; ++c) {
                if (b.miss[c] < 0) continue;
                if (partial_on && (skip_[(size_t) c] || (c % k_) >= tail_rank || b.w[c] < tail_w || frac_skip(b.miss[c]))) {
                    stats_.skipped.fetch_add(1, std::memory_order_relaxed);
                    skip_[(size_t) c] = 1;   // contributes nothing to the partial sum
                    continue;
                }
                cols_.push_back(c);
            }
            // group the columns by expert (stable: token order inside a group), at most 4 per group -- the decode-once
            // GSQ kernels take 4 columns. Types they never take run ggml vec_dot over 64-row chunks that stay in L2, so
            // a group of up to 16 still reads each weight row from DRAM once (dense FFN on the CPU: a 6-row MTP5
            // verify would take two DRAM passes at 4)
            const bool gsq_any = gsq_dot_preferred(L.gate.type, 1) || gsq_dot_preferred(L.gate.type, 4) ||
                                 gsq_dot_preferred(L.up.type, 1) || gsq_dot_preferred(L.up.type, 4) ||
                                 gsq_dot_preferred(L.down.type, 1) || gsq_dot_preferred(L.down.type, 4);
            const int  gcap    = gsq_any ? 4 : 16;
            std::stable_sort(cols_.begin(), cols_.end(), [&](int a, int c) { return b.miss[a] < b.miss[c]; });
            groups_.clear();
            for (int c = 0; c < (int) cols_.size();) {
                int n = 1;
                while (c + n < (int) cols_.size() && n < gcap && b.miss[cols_[(size_t) (c + n)]] == b.miss[cols_[(size_t) c]]) ++n;
                groups_.push_back({ c, n });
                c += n;
            }
            // Debug bound: AI_Q4X_SKIP_MISS=1 answers every ring without computing (wrong output) -- the round's
            // floor with free CPU misses
            static const bool skip_miss = getenv("AI_Q4X_SKIP_MISS") && atoi(getenv("AI_Q4X_SKIP_MISS")) != 0;
            if (!cols_.empty() && !skip_miss) {
                const int64_t K = L.gate.ne[0], F = L.gate.ne[1];
                const auto *  cg = ggml_get_type_traits_cpu((ggml_type) L.gate.type);
                const auto *  cd = ggml_get_type_traits_cpu((ggml_type) L.down.type);
                const auto *  tx = ggml_get_type_traits_cpu(cg->vec_dot_type);
                const auto *  th = ggml_get_type_traits_cpu(cd->vec_dot_type);
                const size_t  xq_row = ggml_row_size(cg->vec_dot_type, K), hq_row = ggml_row_size(cd->vec_dot_type, F);
                using clk = std::chrono::steady_clock;
                auto ns = [](clk::time_point a, clk::time_point b) {
                    return (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
                };
                const auto p0 = clk::now();
                for (int t = 0; t < T; ++t) tx->from_float(b.x + (size_t) t * N_, xq_.data() + (size_t) t * xq_row, K);
                const int nc = (int) cols_.size();
                const int ng = (int) groups_.size();
                stats_.groups.fetch_add((unsigned long long) ng, std::memory_order_relaxed);
                for (const Group & gr : groups_) if (gr.n > 1) stats_.grouped_cols.fetch_add((unsigned long long) gr.n, std::memory_order_relaxed);
                const auto p1 = clk::now();
                run_phase((int) l, 0, ng * (int) ((F + kGuRows - 1) / kGuRows));
                const auto p2 = clk::now();
                // SwiGLU + quantise for the down type, spread over the pool (AI_Q4X_SWIGLU_PAR=0 runs it serially on
                // the poller)
                static const bool mid_par = !getenv("AI_Q4X_SWIGLU_PAR") || atoi(getenv("AI_Q4X_SWIGLU_PAR")) != 0;
                if (mid_par) {
                    run_phase((int) l, 2, nc);
                } else {
                    for (int ci = 0; ci < nc; ++ci) {
                        const float * gu = h_.data() + (size_t) ci * 2 * F_;
                        thread_local std::vector<float> hv;
                        if ((int64_t) hv.size() < F) hv.resize((size_t) F);
                        float * hh = hv.data();
                        for (int64_t r = 0; r < F; ++r) hh[r] = gu[r] / (1.0f + std::exp(-gu[r])) * gu[F_ + r];
                        th->from_float(hh, hq_.data() + (size_t) ci * hq_row, F);
                        if (L.down.type == GGML_TYPE_Q2_0) {
                            ggml_cpu_q2_0_prepare(hq_.data() + (size_t) ci * hq_row, (int) F, prep_.data() + (size_t) ci * prep_row_);
                        }
                    }
                }
                const auto p3 = clk::now();
                run_phase((int) l, 1, ng * (int) ((N_ + kDownRows - 1) / kDownRows));
                const auto p4 = clk::now();
                stats_.ns_x.fetch_add(ns(p0, p1), std::memory_order_relaxed);
                stats_.ns_gu.fetch_add(ns(p1, p2), std::memory_order_relaxed);
                stats_.ns_mid.fetch_add(ns(p2, p3), std::memory_order_relaxed);
                stats_.ns_down.fetch_add(ns(p3, p4), std::memory_order_relaxed);
                if (hs) {
                    hsm.x = (long long) ns(p0, p1); hsm.gu = (long long) ns(p1, p2);
                    hsm.mid = (long long) ns(p2, p3); hsm.down = (long long) ns(p3, p4);
                    hsm.ncols = nc;
                }
                stats_.misses.fetch_add((unsigned long long) nc, std::memory_order_relaxed);
            }
            static const bool partial = !getenv("AI_Q4X_PARTIAL") || atoi(getenv("AI_Q4X_PARTIAL")) != 0;   // default on
            const long long tail0 = hs ? rt_ns() : 0;
            if (partial) {   // one weighted sum per token: the device reads N floats per token, not per column
                for (int t = 0; t < T; ++t) {
                    float * yp  = b.yp + (size_t) t * N_;
                    bool    any = false;
                    for (int j = 0; j < k_; ++j) {
                        const int c = t * k_ + j;
                        if (b.miss[c] < 0 || skip_[(size_t) c]) continue;
                        const float   wc = b.w[c];
                        const float * yc = b.y + (size_t) c * N_;
                        if (!any) for (int64_t n = 0; n < N_; ++n) yp[n] = wc * yc[n];
                        else      for (int64_t n = 0; n < N_; ++n) yp[n] += wc * yc[n];
                        any = true;
                    }
                    if (!any) for (int64_t n = 0; n < N_; ++n) yp[n] = 0.0f;   // all skipped (or none missed)
                }
            }
            // Debug: AI_Q4X_POOL_DELAY_US answers every ring this much later (simulates a slow CPU; race test)
            static const int delay_us = getenv("AI_Q4X_POOL_DELAY_US") ? atoi(getenv("AI_Q4X_POOL_DELAY_US")) : 0;
            if (delay_us > 0) {
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(delay_us);
                while (std::chrono::steady_clock::now() < until) cpu_relax();
            }
            std::atomic_thread_fence(std::memory_order_release);
            *b.done = s;
            last[l] = s;
            if (hs) {
                const long long done_h = rt_ns();
                hsm.tail    = done_h - tail0;
                hsm.busy    = done_h - hsm.busy;
                hsm.lag     = hs_lag;
                hs_done_[l] = done_h;
                if (hs_.size() < hs_.capacity()) hs_.push_back(hsm);
            }
            stats_.rings.fetch_add(1, std::memory_order_relaxed);
            stats_.busy_ns.fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
        }
        if (!any) cpu_relax();
    }
}

// AI_Q4X_HANDOFF_STATS: distributions over the rings since the last reset_stats (decode only when the caller resets
// after the prefill). Device - host differences carry the globaltimer / CLOCK_REALTIME offset; the "- min" columns
// remove a constant offset, and mean A + mean C is offset-free.
void MissPool::hs_report() const {
    if (hs_.empty()) { fprintf(stderr, "[q4x-handoff] no samples\n"); return; }
    auto dist = [](std::vector<long long> v, const char * name, long long sub) {
        if (v.empty()) { fprintf(stderr, "[q4x-handoff] %-28s n 0\n", name); return; }
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (long long x : v) sum += (double) (x - sub);
        auto q = [&](double f) { return (double) (v[(size_t) (f * (double) (v.size() - 1))] - sub) / 1e3; };
        fprintf(stderr, "[q4x-handoff] %-28s n %7zu  mean %8.2f  p10 %8.2f  p50 %8.2f  p90 %8.2f  p99 %8.2f us\n",
                name, v.size(), sum / (double) v.size() / 1e3, q(0.1), q(0.5), q(0.9), q(0.99));
    };
    std::vector<long long> a, c, waited, late, busy, x, gu, mid, down, tail, lag, other, cols;
    long long amin = LLONG_MAX, cmin = LLONG_MAX;
    // decode / verify rounds only (the prefill rings T > 8 have another cost structure)
    std::vector<HsSample> hv;
    for (const auto & h : hs_) if (h.T <= 8) hv.push_back(h);
    if (hv.empty()) { fprintf(stderr, "[q4x-handoff] no decode samples\n"); return; }
    for (const auto & h : hv) {
        amin = std::min(amin, h.a);
        if (h.c != LLONG_MIN) cmin = std::min(cmin, h.c);
    }
    for (const auto & h : hv) {
        a.push_back(h.a);
        busy.push_back(h.busy); x.push_back(h.x); gu.push_back(h.gu); mid.push_back(h.mid); down.push_back(h.down);
        tail.push_back(h.tail); lag.push_back(h.lag); cols.push_back((long long) h.ncols * 1000);
        other.push_back(h.busy - h.x - h.gu - h.mid - h.down - h.tail);
        if (h.c != LLONG_MIN) {
            c.push_back(h.c);
            if (h.arrive < 0) waited.push_back(h.c); else late.push_back(h.c - h.arrive);
        }
    }
    double ma = 0, mc = 0;
    for (long long v : a) ma += (double) v;
    for (long long v : c) mc += (double) v;
    ma /= (double) a.size();
    if (!c.empty()) mc /= (double) c.size();
    fprintf(stderr, "[q4x-handoff] rings %zu | mean A + mean C (offset-free: ring->seen + done->wait-end) %.2f us | "
                    "raw mean A %.2f C %.2f us | device spun on %zu, arrived after done on %zu\n",
            hv.size(), (ma + mc) / 1e3, ma / 1e3, mc / 1e3, waited.size(), late.size());
    dist(a, "A ring->seen (-min)", amin);
    dist(c, "C done->wait-end (-min)", cmin);
    dist(waited, "C | device spun (-minC)", cmin);
    dist(late, "wait end-start | arrived late", 0);
    dist(busy, "busy seen->done", 0);
    dist(x, "  x quantise", 0);
    dist(gu, "  gate/up phase", 0);
    dist(mid, "  swiglu phase", 0);
    dist(down, "  down phase", 0);
    dist(tail, "  partial sum + done", 0);
    dist(other, "  other (route/sort)", 0);
    dist(lag, "worker start lag (max)", 0);
    dist(cols, "missed cols (x1000)", 0);
    const int np = n_workers_ + 1;
    const char * pn[3] = { "gate/up", "down", "swiglu" };
    for (int ph = 0; ph < 2; ++ph) {
        unsigned long long tot = 0;
        for (int p = 0; p < np; ++p) tot += hs_wi_[(size_t) (ph * np + p)].load();
        std::string line;
        for (int p = 0; p < np; ++p) {
            const unsigned long long it = hs_wi_[(size_t) (ph * np + p)].load(), ns = hs_wn_[(size_t) (ph * np + p)].load();
            char buf[64];
            snprintf(buf, sizeof buf, " c%d:%.1f%%/%.0f", cpus_.empty() ? p : cpus_[(size_t) p], tot ? 100.0 * it / tot : 0.0,
                     ns ? it / (ns / 1e6) : 0.0);
            line += buf;
        }
        fprintf(stderr, "[q4x-handoff] %s items: share%%/items-per-ms by cpu (poller first):%s\n", pn[ph], line.c_str());
    }
}

}  // namespace ai::offload
