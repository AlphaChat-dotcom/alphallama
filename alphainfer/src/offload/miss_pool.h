// The host half of the doorbell offload MoE: per-layer doorbells in mapped host memory and a CPU worker
// pool that computes the routed experts the device residency table missed (gate/up -> SwiGLU -> down with ggml-cpu's
// own vec_dot per type, activations quantised to each type's vec_dot_type). The device half is kernels/offload.cu.
//
// Protocol per (window, layer): the device writes miss ids, x and the token count into the layer's bell, then seq.
// The poller sees seq change, the pool computes every missed (token, slot) column into y, then the poller writes
// done = seq. The device spins on done before splicing y into its expert outputs. No driver call on either side.
#pragma once

#include "core/host_mem.h"
#include "core/tensor.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace ai::offload {

struct Bell {   // host view of one layer's mapped doorbell
    volatile unsigned * seq   = nullptr;
    volatile unsigned * done  = nullptr;
    int32_t *           n_tok = nullptr;
    int32_t *           miss  = nullptr;   // [k * Tmax]   expert id per (t, j), -1 = resident
    float *             x     = nullptr;   // [N * Tmax]
    float *             y     = nullptr;   // [N * k * Tmax] unweighted expert outputs per column t*k + j
    float *             w     = nullptr;   // [k * Tmax]   routing weight per column (device -> host)
    float *             yp    = nullptr;   // [N * Tmax]   AI_Q4X_PARTIAL: sum_j w * y over a token's missed columns
    // AI_Q4X_HANDOFF_STATS: device globaltimer stamps written into the header (bytes 16 / 24) under AI_Q4X_WAIT_STATS
    volatile unsigned long long * ring_ts = nullptr;   // the ring (resolve) time
    volatile unsigned long long * wend_ts = nullptr;   // the end of the wait on this layer's last answer
    volatile unsigned long long * wbeg_ts = nullptr;   // its start (written before wend_ts)
    // the same buffers as the device sees them
    unsigned * d_seq = nullptr, * d_done = nullptr;
    int32_t *  d_n_tok = nullptr, * d_miss = nullptr;
    float *    d_x = nullptr, * d_y = nullptr, * d_w = nullptr, * d_yp = nullptr;
};

struct ExpertBanks {   // one MoE layer's routed-expert banks in host memory ([K, F, E] gate/up, [F, K, E] down)
    Tensor gate, up, down;
};

class MissPool {
public:
    // N = n_embd, k = experts per token, t_max = rows per window. cpus: logical cpus to pin the workers to (empty =
    // `threads` unpinned workers). The poller is one more thread (pinned to cpus[0] when given, the rest compute).
    MissPool(const std::vector<ExpertBanks> & layers, int64_t N, int k, int64_t t_max, int threads,
             const std::vector<int> & cpus = {});
    ~MissPool();

    const Bell & bell(int layer) const { return bells_[(size_t) layer]; }

    // Prefill CPU share: compute `ncols` routed columns of layer `layer` for a prompt batch on the pool. The poller
    // serves the job between rings, so jobs are published from one thread only (prefill and decode never overlap).
    // x: host f32 [T][N]; per listed column i: tok[i] its token, ids[i] its expert, w[i] its routing weight, listed in
    // ascending (token, slot) order. yp: host f32 [T][N], set to the sum over each token's listed columns of w * expert
    // output (0 for a token with none). Every listed column is computed (no tail rules). The pointers stay valid until
    // host_job_wait returns.
    void host_job_begin(int layer, const float * x, int T, const int32_t * tok, const int32_t * ids, const float * w,
                        int ncols, float * yp);
    void host_job_wait();
    long long host_job_ns() const { return hj_ns_; }   // the last host job's duration on the pool (after host_job_wait)

    struct Stats {
        std::atomic<unsigned long long> rings{ 0 }, misses{ 0 }, busy_ns{ 0 }, skipped{ 0 };
        // profile: expert groups computed, columns that shared a group (n > 1), and the poller's per-phase time
        std::atomic<unsigned long long> groups{ 0 }, grouped_cols{ 0 };
        std::atomic<unsigned long long> ns_x{ 0 }, ns_gu{ 0 }, ns_mid{ 0 }, ns_down{ 0 }, ns_tail{ 0 };
    };
    const Stats & stats() const { return stats_; }
    void          reset_stats() {   // e.g. after the prefill, so per-round numbers are decode-only
        for (auto * a : { &stats_.rings, &stats_.misses, &stats_.busy_ns, &stats_.skipped, &stats_.groups,
                          &stats_.grouped_cols, &stats_.ns_x, &stats_.ns_gu, &stats_.ns_mid, &stats_.ns_down,
                          &stats_.ns_tail }) a->store(0);
        hs_reset_.store(true);
    }

private:
    void poll_loop();
    void work_loop(int id);
    void compute_share(int part, int n_parts);   // static share `part` of the current job's items

    std::vector<ExpertBanks> layers_;
    std::vector<Bell>        bells_;
    void *                   mapped_ = nullptr;   // one pinned + mapped block for every bell (mapped_mem_)
    PinnedHost               mapped_mem_;
    int64_t                  N_ = 0, F_ = 0, t_max_ = 0;
    int                      k_ = 0;
    int                      n_workers_ = 0;
    std::vector<int>         cpus_;

    // The current job (one phase of one layer), handed from the poller to the workers. The poller writes the fields,
    // then publishes job_gen_; each worker runs its static share (items part, part + n_parts, ...) and stores the gen
    // into its done slot. The poller changes the fields only after every worker has reported, so no worker ever reads
    // a half-written job (a work-stealing counter shared across phases could hand a late worker a new phase's item).
    std::atomic<int>      job_gen_{ 0 };
    std::atomic<int>      next_item_{ 0 };   // AI_Q4X_POOL_STEAL: items claimed dynamically (P/E-core balance)
    int                   job_layer_ = -1, job_phase_ = 0, n_items_ = 0;
    std::vector<std::atomic<int>> worker_done_;
    std::vector<char>     skip_;         // route-tail skipped columns of the current job (host only)
    std::vector<int>      cols_;         // missed columns t*k + j of the current job, sorted by expert id
    // the current job's column maps: expert of column c, its token (nullptr: c / k), its output row (y + c * N)
    const int32_t *       jmiss_ = nullptr;
    const int32_t *       jtok_  = nullptr;
    float *               jy_    = nullptr;
    int                   jtok(int c) const { return jtok_ ? jtok_[c] : c / k_; }
    // the host job (prefill CPU share)
    std::atomic<bool>     hj_pending_{ false }, hj_done_{ true };
    int                   hj_layer_ = 0, hj_T_ = 0, hj_ncols_ = 0;
    const float *         hj_x_ = nullptr, * hj_w_ = nullptr;
    const int32_t *       hj_tok_ = nullptr, * hj_ids_ = nullptr;
    float *               hj_yp_ = nullptr;
    long long             hj_ns_ = 0;
    std::vector<float>    hy_;           // its compact outputs, one row of N per listed column
    std::vector<char>     hneed_;        // its tokens with at least one listed column
    struct Group { int c0, n; };         // columns [c0, c0 + n) of cols_ share one expert (n <= 4)
    std::vector<Group>    groups_;
    std::vector<char>     prep_;         // Q2_0 down: each column's h prepared once (ggml_cpu_q2_0_prepare)
    size_t                prep_row_ = 0;
    std::vector<char>     xq_;           // x quantised per token for the gate/up type  [Tmax][row bytes]
    std::vector<float>    h_;            // SwiGLU per missed column [F] each
    std::vector<char>     hq_;           // h quantised for the down type
    std::atomic<bool>     stop_{ false };
    std::vector<std::thread> threads_;
    Stats                 stats_;
    // AI_Q4X_HANDOFF_STATS=1 (with AI_Q4X_WAIT_STATS=1): per ring, written by the poller only, printed at destruction:
    // A = host_seen - device_ring, C = device_wait_end - host_done (each carries the host/device clock offset with
    // opposite sign: A + C is offset-free = notice + answer latency; A - min A and C - min C split it), busy = done - seen
    // Host times are system_clock ns (CLOCK_REALTIME, the same epoch as the device globaltimer up to its drift).
    struct HsSample {
        long long a;        // host saw seq - device ring
        long long c;        // device wait end - host done (previous ring of the layer); LLONG_MIN = no wait stamp
        long long arrive;   // device wait start - host done (< 0: the device waited for the answer)
        long long busy, x, gu, mid, down, tail;   // host seen -> done, and its phases
        long long lag;      // max over the job's phases of the slowest worker's start - publish
        int       layer, ncols, T;
    };
    std::vector<HsSample> hs_;
    std::vector<long long> hs_done_;   // per layer: host done time of its last ring
    std::vector<unsigned long long> hs_wend_;   // per layer: the last wait-end stamp seen
    std::vector<std::atomic<long long>> hs_wstart_;   // per worker: system_clock ns it picked up the current gen
    std::atomic<bool>     hs_reset_{ false };
    // per participant (0 = poller, i + 1 = worker i) and phase (0 gate/up, 1 down, 2 swiglu): items run and ns in the
    // job, decode rings only; a straggler shows as a low item share (work stealing), a uniform slowdown as low rates
    std::vector<std::atomic<unsigned long long>> hs_wi_, hs_wn_;
    void hs_worker_add(int part, int ph, unsigned long long items, long long ns) {
        const size_t i = (size_t) ph * (size_t) (n_workers_ + 1) + (size_t) part;
        if (i >= hs_wi_.size()) return;
        hs_wi_[i].fetch_add(items, std::memory_order_relaxed);
        hs_wn_[i].fetch_add((unsigned long long) ns, std::memory_order_relaxed);
    }
    void                  hs_report() const;
};

}  // namespace ai::offload
