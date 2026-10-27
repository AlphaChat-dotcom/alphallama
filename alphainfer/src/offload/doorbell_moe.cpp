#include "offload/doorbell_moe.h"

#include "kernels/doorbell.h"
#include "offload/cpu_topo.h"
#include "offload/miss_pool.h"
#include "ops/adapter.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ai::offload {

static int env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

size_t DoorbellMoe::expert_bytes(const op::MoeWeights & w) {
    return w.gate_exps.nb[2] + w.up_exps.nb[2] + w.down_exps.nb[2];
}

// "DREP" u32 n_layers u64 n_tokens, then per layer i32 blk, i32 n, u32 uses[n]. Reads records until the file ends
// (a header's n_layers may count a layer that has no record).
static std::vector<std::vector<uint32_t>> read_drep(const std::string & path, int n_banks, int E) {
    std::vector<std::vector<uint32_t>> prof((size_t) n_banks);
    std::ifstream f(path, std::ios::binary);
    uint32_t magic = 0, nl = 0;
    uint64_t nt = 0;
    f.read((char *) &magic, 4);
    f.read((char *) &nl, 4);
    f.read((char *) &nt, 8);
    if (!f || magic != 0x50455244u) {
        fprintf(stderr, "[door] profile %s: not a DREP file, residency falls back to experts 0..slots-1\n", path.c_str());
        return prof;
    }
    int got = 0;
    for (;;) {
        int32_t blk = 0, n = 0;
        if (!f.read((char *) &blk, 4) || !f.read((char *) &n, 4) || n <= 0 || n > 65536) break;
        std::vector<uint32_t> u((size_t) n);
        if (!f.read((char *) u.data(), (std::streamsize) u.size() * 4)) break;
        if (blk >= 0 && blk < n_banks && n == E) { prof[(size_t) blk] = std::move(u); ++got; }
    }
    fprintf(stderr, "[door] profile %s: %d of %d banks ranked\n", path.c_str(), got, n_banks);
    return prof;
}

DoorbellMoe::DoorbellMoe(Device & dev, const std::vector<op::MoeWeights> & banks, const std::vector<int> & slots,
                         int n_expert, int k, int64_t N, const DoorbellConfig & cfg)
    : dev_(dev), cfg_(cfg), E_(n_expert), k_(k), N_(N) {
    if (banks.empty() || banks.size() != slots.size()) throw std::runtime_error("DoorbellMoe: one slot count per bank");
    if (!getenv("AI_Q4X_PARTIAL") ? false : atoi(getenv("AI_Q4X_PARTIAL")) == 0) {
        throw std::runtime_error("DoorbellMoe: needs the CPU pool's partial sums (AI_Q4X_PARTIAL must not be 0)");
    }
    const size_t nbk = banks.size();
    for (const auto & w : banks) {
        // MissPool quantises x once for the gate type and uses it for up too, and keeps the SwiGLU row on the stack
        if (w.gate_exps.type != w.up_exps.type) {
            throw std::runtime_error(std::string("DoorbellMoe: gate and up experts of different types (") +
                                     dtype_name(w.gate_exps.type) + " / " + dtype_name(w.up_exps.type) +
                                     ") are not supported by the CPU miss pool yet");
        }
        if (w.gate_exps.ne[1] > 2048) throw std::runtime_error("DoorbellMoe: expert FFN width > 2048 (miss pool limit)");
        if (w.gate_exps.ne[2] != E_) throw std::runtime_error("DoorbellMoe: bank expert count != n_expert");
    }
    std::vector<std::vector<uint32_t>> prof((size_t) nbk);
    if (!cfg_.profile.empty()) prof = read_drep(cfg_.profile, (int) nbk, E_);

    // staging groups: banks with identical types and row shapes share one prefill staging bank
    banks_.assign(nbk, {});
    const int pf = cfg_.pf_slots > 0 ? std::min(cfg_.pf_slots, E_) : std::min(E_, 64);
    std::vector<int> stage_rep;   // representative bank of each staging group
    for (size_t b = 0; b < nbk; ++b) {
        const auto & w = banks[b];
        int g = -1;
        for (size_t s = 0; s < stage_rep.size() && g < 0; ++s) {
            const auto & r = banks[(size_t) stage_rep[s]];
            bool same = true;
            for (const auto pr : { std::make_pair(&w.gate_exps, &r.gate_exps), std::make_pair(&w.up_exps, &r.up_exps),
                                   std::make_pair(&w.down_exps, &r.down_exps) }) {
                same = same && pr.first->type == pr.second->type && pr.first->ne[0] == pr.second->ne[0] &&
                       pr.first->ne[1] == pr.second->ne[1] && pr.first->nb[2] == pr.second->nb[2];
            }
            if (same) g = (int) s;
        }
        if (g < 0) { g = (int) stage_rep.size(); stage_rep.push_back((int) b); }
        banks_[b].stage = g;
    }

    // one device buffer: slot banks + tables, staging banks, bell counters
    auto bank_of = [&](const Tensor & h, int n) {
        Tensor t = h;
        t.ne[2]  = n;
        set_contiguous_strides(t);
        return t;
    };
    auto pad = [&](size_t v) { return (v + 255) / 256 * 256; };
    size_t total = 4096;
    for (size_t b = 0; b < nbk; ++b) {
        const auto & w  = banks[b];
        const int    ns = std::max(0, std::min(slots[b], E_));
        banks_[b].n_slots = ns;
        for (const Tensor * h : { &w.gate_exps, &w.up_exps, &w.down_exps }) total += pad(dev_.alloc_size(bank_of(*h, ns + k_)));
        total += pad((size_t) E_ * 4);
    }
    for (int rep : stage_rep) {
        const auto & w = banks[(size_t) rep];
        for (const Tensor * h : { &w.gate_exps, &w.up_exps, &w.down_exps }) total += pad(dev_.alloc_size(bank_of(*h, pf)));
    }
    total += pad(nbk * 4);
    const int buf  = dev_.alloc_buffer(total, "door.slots");
    char *    base = (char *) dev_.buffer_base(buf);
    size_t    off  = 0;
    auto carve = [&](const Tensor & h, int n) {
        Tensor t = bank_of(h, n);
        t.buf    = buf;
        off      = pad(off);
        t.data   = base + off;
        off     += dev_.alloc_size(t);
        return t;
    };
    auto copy_expert = [](const Tensor & dst, int slot, const Tensor & src, int e) {
        cudaMemcpy((char *) dst.data + (size_t) slot * dst.nb[2], (const char *) src.data + (size_t) e * src.nb[2],
                   src.nb[2], cudaMemcpyHostToDevice);
    };
    double resident = 0;
    int    ranked   = 0;
    size_t slot_bytes = 0;
    std::vector<ExpertBanks> host;
    for (size_t b = 0; b < nbk; ++b) {
        const auto & w = banks[b];
        Bank &       B = banks_[b];
        const int    ns = B.n_slots, nb = ns + k_;
        B.gate = carve(w.gate_exps, nb);
        B.up   = carve(w.up_exps, nb);
        B.down = carve(w.down_exps, nb);
        B.host_gate = w.gate_exps;
        B.host_up   = w.up_exps;
        B.host_down = w.down_exps;
        slot_bytes += (size_t) nb * expert_bytes(w);
        off = pad(off);
        B.table.buf   = buf;
        B.table.type  = I32;
        B.table.ne[0] = E_;
        set_contiguous_strides(B.table);
        B.table.data = base + off;
        off += (size_t) E_ * 4;
        std::vector<int32_t> order((size_t) E_);
        for (int e = 0; e < E_; ++e) order[(size_t) e] = e;
        if (!prof[b].empty()) {
            const auto & pr = prof[b];
            std::stable_sort(order.begin(), order.end(), [&](int a, int c) { return pr[(size_t) a] > pr[(size_t) c]; });
            uint64_t all = 0, top = 0;
            for (int e = 0; e < E_; ++e) all += pr[(size_t) e];
            for (int i = 0; i < ns; ++i) top += pr[(size_t) order[(size_t) i]];
            resident += all ? (double) top / (double) all : 0.0;
            ++ranked;
        }
        B.table_h.assign((size_t) E_, -1);
        B.owner.assign((size_t) nb, -1);
        B.score.assign((size_t) E_, 0.0f);
        for (int sl = 0; sl < nb; ++sl) {
            const int e = sl < ns ? order[(size_t) sl] : sl - ns;   // fillers: copies of experts 0..k-1
            if (sl < ns) { B.table_h[(size_t) e] = sl; B.owner[(size_t) sl] = e; }
            copy_expert(B.gate, sl, w.gate_exps, e);
            copy_expert(B.up, sl, w.up_exps, e);
            copy_expert(B.down, sl, w.down_exps, e);
        }
        cudaMemcpy(B.table.data, B.table_h.data(), (size_t) E_ * 4, cudaMemcpyHostToDevice);
        host.push_back({ w.gate_exps, w.up_exps, w.down_exps });
    }
    for (int rep : stage_rep) {   // every staging slot always holds valid weights (it can serve as a weight-0 filler)
        const auto & w = banks[(size_t) rep];
        Stage st;
        st.slots = pf;
        st.gate  = carve(w.gate_exps, pf);
        st.up    = carve(w.up_exps, pf);
        st.down  = carve(w.down_exps, pf);
        for (int sl = 0; sl < pf; ++sl) {
            copy_expert(st.gate, sl, w.gate_exps, sl);
            copy_expert(st.up, sl, w.up_exps, sl);
            copy_expert(st.down, sl, w.down_exps, sl);
        }
        stages_.push_back(st);
    }
    off       = pad(off);
    counters_ = (unsigned *) (base + off);
    cudaMemset(counters_, 0, nbk * 4);
    off += nbk * 4;
    if (off > total) throw std::runtime_error("DoorbellMoe: device carve overran its buffer");
    device_bytes_ = total;

    // CPU pool placement: explicit list, else the topology plan (the launch thread owns its physical core)
    std::vector<int> cpus = cfg_.cpus;
    int threads = cfg_.threads > 0 ? cfg_.threads : 8;
    if (cpus.empty()) {
        CpuPlan plan = plan_cpus(cfg_.threads > 0 ? cfg_.threads : 0);
        // Default: one worker per free physical core. Spinning workers on both SMT threads of a core make the pool
        // slower and erratic, and starve the admission worker and the driver threads. AI_DOOR_THREADS=N overrides.
        if (cfg_.threads <= 0 && plan.n_phys > 0 && (int) plan.pool.size() > plan.n_phys) plan.pool.resize((size_t) plan.n_phys);
        if (!plan.pool.empty()) {
            pin_this_thread(plan.launch);
            cpus = plan.pool;
        }
        fprintf(stderr, "[door] cpu placement: %s -> launch %s, pool %s\n", plan.why.c_str(),
                cpu_list_str(plan.launch).c_str(), cpus.empty() ? "(unpinned)" : cpu_list_str(cpus).c_str());
    }
    pool_ = std::make_unique<MissPool>(host, N_, k_, cfg_.t_max, threads, cpus);

    // admission: a copy stream, pinned staging buffers, a pinned table mirror, the worker
    if (cfg_.admit > 0) {
        cudaStreamCreateWithFlags((cudaStream_t *) &copy_stream_, cudaStreamNonBlocking);
        size_t unit = 0;
        for (const auto & w : banks) unit = std::max(unit, expert_bytes(w));
        const int n_stage = std::max(cfg_.admit, env_int("AI_DOOR_ADMIT_STAGE", 2 * cfg_.admit));
        for (int i = 0; i < n_stage; ++i) {
            char * p = nullptr;
            cudaHostAlloc((void **) &p, unit, cudaHostAllocDefault);
            staging_.push_back(p);
            cudaEvent_t ev;
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            staging_ev_.push_back((void *) ev);
        }
        recorded_.reset(new std::atomic<int64_t>[(size_t) n_stage]);
        for (int i = 0; i < n_stage; ++i) recorded_[(size_t) i].store(-1);
        cudaHostAlloc((void **) &table_pin_, nbk * (size_t) E_ * 4, cudaHostAllocDefault);
        for (size_t b = 0; b < nbk; ++b) memcpy(table_pin_ + b * (size_t) E_, banks_[b].table_h.data(), (size_t) E_ * 4);
        admit_thr_ = std::thread([this] { admit_worker(); });
    }
    int min_s = E_, max_s = 0;
    for (const auto & B : banks_) { min_s = std::min(min_s, B.n_slots); max_s = std::max(max_s, B.n_slots); }
    fprintf(stderr, "[door] doorbell offload: %zu banks, %d..%d resident + %d filler slots per bank of %d experts "
                    "(%.2f GiB slots, %zu staging group(s) x %d slots, %.2f GiB device total), t_max %lld, CPU pool %d "
                    "threads%s, admission %d/round; profile mass resident %s\n",
            nbk, min_s, max_s, k_, E_, slot_bytes / 1073741824.0, stages_.size(), pf, total / 1073741824.0,
            (long long) cfg_.t_max, cpus.empty() ? threads : (int) cpus.size(), cpus.empty() ? "" : " (pinned)",
            cfg_.admit, ranked ? (std::to_string(100.0 * resident / ranked).substr(0, 5) + "%").c_str() : "n/a");
}

DoorbellMoe::~DoorbellMoe() {
    if (admit_thr_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(admit_mu_);
            admit_stop_ = true;
        }
        admit_cv_.notify_all();
        admit_thr_.join();
    }
    fprintf(stderr, "[door] %s\n", stats_line().c_str());
    pool_.reset();
    for (char * p : staging_) cudaFreeHost(p);
    for (void * e : staging_ev_) cudaEventDestroy((cudaEvent_t) e);
    if (table_pin_) cudaFreeHost(table_pin_);
    if (h_up_) cudaFreeHost(h_up_);
    if (copy_stream_) cudaStreamDestroy((cudaStream_t) copy_stream_);
}

Tensor DoorbellMoe::moe(int bank, const op::MoeWeights & w, const op::MoeParams & p, const Tensor & x) {
    const int64_t T = x.ne[1];
    if (T < 1 || T > cfg_.t_max) throw std::runtime_error("DoorbellMoe::moe: rows out of 1..t_max");
    if (x.type != F32 || !x.contiguous() || x.ne[0] != N_) throw std::runtime_error("DoorbellMoe::moe: x must be f32 [N, T] contiguous");
    if (p.n_used != k_) throw std::runtime_error("DoorbellMoe::moe: expert_used_count mismatch");
    const Bank &   B  = banks_.at((size_t) bank);
    const Bell &   bl = pool_->bell(bank);
    op::MoeRoute route = op::moe_route(dev_, x, w.router, p);
    Tensor slots = dev_.ws_alloc(I32, k_, T);
    Tensor wm    = dev_.ws_alloc(F32, 1, k_, T);
    const int32_t * tab  = (const int32_t *) B.table.data;
    const int32_t * sel  = (const int32_t *) route.sel.data;
    const float *   wp   = (const float *) route.weights.data;
    const float *   xp   = (const float *) x.data;
    int32_t *       sp   = (int32_t *) slots.data;
    float *         wmp  = (float *) wm.data;
    unsigned *      ctr  = counters_ + bank;
    const int32_t   f0   = B.n_slots;
    const int       k    = k_;
    const int64_t   N    = N_;
    Device *        dev  = &dev_;
    adapter::native(dev_, [=]() {
        kern::door_resolve_ring(adapter::stream(*dev), tab, sel, wp, xp, N, k, T, f0, sp, wmp, bl.d_miss, bl.d_w,
                                bl.d_x, bl.d_n_tok, bl.d_seq, ctr);
    });
    Tensor shared = op::moe_shared(dev_, x, w);   // overlaps the CPU's work on the misses
    if (shared.type != F32 || !shared.contiguous()) throw std::runtime_error("DoorbellMoe::moe: shared expert output not f32 contiguous");
    Tensor routed = op::moe_experts(dev_, x, B.up, B.gate, B.down, slots, wm, k_);
    float * shp = (float *) shared.data;
    adapter::native(dev_, [=]() {
        kern::door_wait_add(adapter::stream(*dev), bl.d_done, ctr, sp, f0, bl.d_yp, shp, N, k, T);
    });
    return op::add(dev_, routed, shared);
}

Tensor DoorbellMoe::moe_bulk(int bank, const op::MoeWeights & w, const op::MoeParams & p, const Tensor & x) {
    if (op::recording()) throw std::runtime_error("DoorbellMoe::moe_bulk: host-synced path inside a recorded round");
    if (p.n_used != k_) throw std::runtime_error("DoorbellMoe::moe_bulk: expert_used_count mismatch");
    const int64_t T = x.ne[1];
    const int64_t n = (int64_t) k_ * T;
    Bank &        B = banks_.at((size_t) bank);
    const Stage & S = stages_.at((size_t) B.stage);
    op::MoeRoute route = op::moe_route(dev_, x, w.router, p);
    // pinned regions: sel, then (slots i32 + mask f32) per pass; grown here, at a synchronised point
    const int    max_pass = 1 + (E_ + S.slots - 1) / S.slots;
    const size_t need = (size_t) n * 4 * (1 + 2 * (size_t) max_pass) + 4096;
    dev_.synchronize();
    if (need > h_up_bytes_) {
        if (h_up_) cudaFreeHost(h_up_);
        cudaHostAlloc((void **) &h_up_, need, cudaHostAllocDefault);
        h_up_bytes_ = need;
        h_sel_      = (int32_t *) h_up_;
    }
    dev_.download_async(route.sel, h_sel_, 0, (size_t) n * 4);
    dev_.synchronize();
    char * region = h_up_ + (size_t) n * 4;
    int    pass   = 0;
    auto upload_pass = [&](const std::vector<int32_t> & sl, const std::vector<float> & mk, Tensor & st, Tensor & m) {
        int32_t * hs = (int32_t *) (region + (size_t) pass * 2 * n * 4);
        float *   hm = (float *) (region + (size_t) pass * 2 * n * 4 + (size_t) n * 4);
        memcpy(hs, sl.data(), (size_t) n * 4);
        memcpy(hm, mk.data(), (size_t) n * 4);
        st = dev_.ws_alloc(I32, k_, T);
        m  = dev_.ws_alloc(F32, 1, k_, T);
        dev_.upload_async(st, hs, 0, (size_t) n * 4);
        dev_.upload_async(m, hm, 0, (size_t) n * 4);
        ++pass;
    };
    std::vector<int32_t> sl((size_t) n);
    std::vector<float>   mk((size_t) n);
    std::vector<char>    seen((size_t) E_, 0);
    std::vector<int32_t> missed;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = h_sel_[i];
        const int32_t s = B.table_h[(size_t) e];
        sl[(size_t) i] = s >= 0 ? s : B.n_slots + (int32_t) (i % k_);
        mk[(size_t) i] = s >= 0 ? 1.0f : 0.0f;
        if (s < 0 && !seen[(size_t) e]) { seen[(size_t) e] = 1; missed.push_back(e); }
    }
    n_bulk_cols_ += (uint64_t) n;
    // Every pass accumulates into one buffer and rolls its own intermediates back; kept alive to the layer's end,
    // several passes of a large chunk would overflow the workspace the model sizes for one MoE block.
    Tensor acc   = dev_.ws_alloc(F32, N_, T);
    bool   first = true;
    auto run_pass = [&](const Tensor & up, const Tensor & gate, const Tensor & down) {
        Device::Scope sc(dev_);
        Tensor st, m;
        upload_pass(sl, mk, st, m);
        Tensor part = op::moe_experts(dev_, x, up, gate, down, st, op::mul(dev_, route.weights, m), k_);
        if (first) op::copy_into(dev_, part, acc, "door bulk acc");
        else       op::copy_into(dev_, op::add(dev_, acc, part), acc, "door bulk acc");
        first = false;
    };
    run_pass(B.up, B.gate, B.down);
    void * stream = adapter::stream(dev_);
    std::vector<int32_t> stage_of((size_t) E_, -1);
    std::vector<char>    used((size_t) S.slots, 0);
    for (size_t g0 = 0; g0 < missed.size(); g0 += (size_t) S.slots) {
        const size_t g1 = std::min(missed.size(), g0 + (size_t) S.slots);
        // the copies are ordered on the compute stream after the previous pass's kernels, which read the staging bank
        for (size_t i = g0; i < g1; ++i) {
            const int32_t e  = missed[i];
            const int     ss = (int) (i - g0);
            stage_of[(size_t) e] = ss;
            const Tensor * src[3] = { &B.host_gate, &B.host_up, &B.host_down };
            const Tensor * dst[3] = { &S.gate, &S.up, &S.down };
            for (int r = 0; r < 3; ++r) {
                cudaMemcpyAsync((char *) dst[r]->data + (size_t) ss * dst[r]->nb[2],
                                (const char *) src[r]->data + (size_t) e * src[r]->nb[2], src[r]->nb[2],
                                cudaMemcpyHostToDevice, (cudaStream_t) stream);
                n_bulk_bytes_ += src[r]->nb[2];
            }
        }
        for (int64_t t = 0; t < T; ++t) {
            std::fill(used.begin(), used.end(), 0);
            for (int j = 0; j < k_; ++j) {
                const int32_t e = h_sel_[t * k_ + j];
                if (stage_of[(size_t) e] >= 0) used[(size_t) stage_of[(size_t) e]] = 1;
            }
            int next = 0;
            for (int j = 0; j < k_; ++j) {
                const int64_t i = t * k_ + j;
                const int32_t e = h_sel_[i];
                if (stage_of[(size_t) e] >= 0) {
                    sl[(size_t) i] = stage_of[(size_t) e];
                    mk[(size_t) i] = 1.0f;
                    n_bulk_miss_++;
                } else {   // a distinct staging slot of this row at weight 0 (every staging slot holds valid weights)
                    while (next < S.slots && used[(size_t) next]) ++next;
                    if (next == S.slots) throw std::runtime_error("DoorbellMoe::moe_bulk: no distinct staging filler");
                    used[(size_t) next] = 1;
                    sl[(size_t) i] = next;
                    mk[(size_t) i] = 0.0f;
                }
            }
        }
        run_pass(S.up, S.gate, S.down);
        for (size_t i = g0; i < g1; ++i) stage_of[(size_t) missed[i]] = -1;
    }
    return op::add(dev_, acc, op::moe_shared(dev_, x, w));
}

void DoorbellMoe::admit_worker() {
    for (;;) {
        Pending job;
        {
            std::unique_lock<std::mutex> lk(admit_mu_);
            admit_cv_.wait(lk, [&] { return admit_stop_ || !admit_q_.empty(); });
            if (admit_stop_) return;
            job = admit_q_.front();
            admit_q_.pop_front();
        }
        const size_t   si = (size_t) (job.job % (int64_t) staging_.size());
        const Bank &   B  = banks_[(size_t) job.bank];
        char *         st = staging_[si];
        size_t         o  = 0;
        const Tensor * src[3] = { &B.host_gate, &B.host_up, &B.host_down };
        const Tensor * dst[3] = { &B.gate, &B.up, &B.down };
        for (int r = 0; r < 3; ++r) {
            memcpy(st + o, (const char *) src[r]->data + (size_t) job.expert * src[r]->nb[2], src[r]->nb[2]);
            cudaMemcpyAsync((char *) dst[r]->data + (size_t) job.slot * dst[r]->nb[2], st + o, src[r]->nb[2],
                            cudaMemcpyHostToDevice, (cudaStream_t) copy_stream_);
            o += src[r]->nb[2];
        }
        cudaEventRecord((cudaEvent_t) staging_ev_[si], (cudaStream_t) copy_stream_);
        recorded_[si].store(job.job, std::memory_order_release);
    }
}

// Between rounds (the previous round has completed). 1) publish the copies queued at the previous boundary (waiting
// for them if needed, so residency is a function of the token history alone); 2) count every bell that rang since
// the last call; 3) admit the hottest missed experts into the coldest slots; 4) changed tables -> device, ordered on
// the compute stream before the next round (a victim is unmapped there, before the round can read its slot).
void DoorbellMoe::after_round() {
    const size_t nbk = banks_.size();
    std::vector<char> dirty(nbk, 0);
    for (const Pending & p : pending_) {
        const size_t si = (size_t) (p.job % (int64_t) staging_.size());
        while (recorded_[si].load(std::memory_order_acquire) != p.job) std::this_thread::yield();
        cudaEventSynchronize((cudaEvent_t) staging_ev_[si]);
        Bank & B = banks_[(size_t) p.bank];
        B.table_h[(size_t) p.expert] = p.slot;
        B.owner[(size_t) p.slot]     = p.expert;
        dirty[(size_t) p.bank]       = 1;
        n_publish_++;
    }
    pending_.clear();
    struct Cand { float score; int bank, expert; };
    std::vector<Cand> cand;
    for (size_t b = 0; b < nbk; ++b) {
        Bank &         B  = banks_[b];
        const Bell &   bl = pool_->bell((int) b);
        const unsigned s  = *bl.seq;
        if (s == B.last_seq) continue;   // not rung since the last boundary: its miss list is stale
        B.last_seq = s;
        for (auto & v : B.score) v *= 0.97f;
        const int T = *bl.n_tok;
        for (int c = 0; c < k_ * T; ++c) {
            const int32_t m = bl.miss[c];
            const int32_t e = m >= 0 ? m : -(m + 2);
            if (e < 0 || e >= E_) continue;
            n_cols_++;
            B.score[(size_t) e] += 1.0f;
            if (m >= 0) {
                n_miss_++;
                if (B.table_h[(size_t) e] < 0) cand.push_back({ B.score[(size_t) e], (int) b, e });
            }
        }
    }
    if (cfg_.admit > 0 && !cand.empty()) {
        std::sort(cand.begin(), cand.end(), [](const Cand & a, const Cand & c) { return a.score > c.score; });
        int issued = 0;
        for (const Cand & cd : cand) {
            if (issued >= cfg_.admit || pending_.size() >= staging_.size()) break;
            if (cd.score < 2.0f) break;
            Bank & B = banks_[(size_t) cd.bank];
            if (B.table_h[(size_t) cd.expert] >= 0) continue;
            bool in_flight = false;
            for (const Pending & p : pending_) in_flight = in_flight || (p.bank == cd.bank && p.expert == cd.expert);
            if (in_flight) continue;
            int   victim = -1;
            float vs     = 1e30f;
            for (int sl = 0; sl < B.n_slots; ++sl) {
                const int32_t e = B.owner[(size_t) sl];
                if (e < 0) continue;
                if (B.score[(size_t) e] < vs) { vs = B.score[(size_t) e]; victim = sl; }
            }
            if (victim < 0 || cd.score < 1.5f * vs + 1e-6f) continue;
            const int32_t old = B.owner[(size_t) victim];
            B.table_h[(size_t) old]  = -1;
            B.owner[(size_t) victim] = -1;
            dirty[(size_t) cd.bank]  = 1;
            Pending job{ cd.bank, victim, cd.expert, (int64_t) n_admit_ };
            pending_.push_back(job);
            {
                std::lock_guard<std::mutex> lk(admit_mu_);
                admit_q_.push_back(job);
            }
            n_admit_++;
            issued++;
        }
        if (issued) admit_cv_.notify_one();
    }
    if (!table_pin_) return;
    void * st = adapter::stream(dev_);
    for (size_t b = 0; b < nbk; ++b) {
        if (!dirty[b]) continue;
        int32_t * pin = table_pin_ + b * (size_t) E_;
        memcpy(pin, banks_[b].table_h.data(), (size_t) E_ * 4);
        cudaMemcpyAsync(banks_[b].table.data, pin, (size_t) E_ * 4, cudaMemcpyHostToDevice, (cudaStream_t) st);
    }
}

std::string DoorbellMoe::stats_line() const {
    const auto & ps = pool_->stats();
    char buf[512];
    snprintf(buf, sizeof buf,
             "decode columns %llu, missed %llu (hit %.1f%%), admitted %llu, published %llu | CPU pool rings %llu, "
             "computed %llu, busy %.1f ms | prefill columns %llu, staged %llu (%.2f GiB copied)",
             (unsigned long long) n_cols_, (unsigned long long) n_miss_,
             n_cols_ ? 100.0 * (double) (n_cols_ - n_miss_) / (double) n_cols_ : 0.0, (unsigned long long) n_admit_,
             (unsigned long long) n_publish_, (unsigned long long) ps.rings.load(), (unsigned long long) ps.misses.load(),
             ps.busy_ns.load() / 1e6, (unsigned long long) n_bulk_cols_, (unsigned long long) n_bulk_miss_,
             n_bulk_bytes_ / 1073741824.0);
    return buf;
}

void DoorbellMoe::reset_stats() {
    n_cols_ = n_miss_ = n_bulk_cols_ = n_bulk_miss_ = n_bulk_bytes_ = 0;
}

}  // namespace ai::offload
