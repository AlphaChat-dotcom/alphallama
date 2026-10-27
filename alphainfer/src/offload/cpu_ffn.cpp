#include "offload/cpu_ffn.h"

#include "kernels/doorbell.h"
#include "offload/cpu_topo.h"
#include "offload/miss_pool.h"
#include "ops/adapter.h"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace ai::offload {

CpuFfn::CpuFfn(Device & dev, const std::vector<op::MoeWeights> & layers, int64_t N, int64_t t_max, int threads)
    : dev_(dev), N_(N), t_max_(t_max) {
    if (layers.empty()) throw std::runtime_error("CpuFfn: no layers");
    std::vector<ExpertBanks> banks;
    size_t host = 0;
    for (const auto & w : layers) {
        if (w.sh_gate.buf >= 0 || w.sh_up.buf >= 0 || w.sh_down.buf >= 0) throw std::runtime_error("CpuFfn: weights must be host-kept");
        if (w.sh_gate.type != w.sh_up.type) throw std::runtime_error("CpuFfn: gate and up of different types");
        banks.push_back({ w.sh_gate, w.sh_up, w.sh_down });
        for (const Tensor * t : { &w.sh_gate, &w.sh_up, &w.sh_down }) stage_bytes_ = std::max(stage_bytes_, dev_.alloc_size(*t));
        host += w.sh_gate.nbytes() + w.sh_up.nbytes() + w.sh_down.nbytes();
    }
    // device: constants + per-layer bell counters, and the prefill staging buffer (the largest single matrix)
    const size_t nl = layers.size(), cbytes = 256 + (size_t) t_max * 8 + nl * 4 + 1024;
    const int    cb = dev_.alloc_buffer(cbytes, "cpu_ffn.consts");
    char *       base = (char *) dev_.buffer_base(cb);
    table_    = (int32_t *) base;
    sel_      = (int32_t *) (base + 256);
    w_        = (float *) (base + 256 + (size_t) t_max * 4);
    counters_ = (unsigned *) (base + 256 + (size_t) t_max * 8 + 256);
    const int32_t minus1 = -1;
    cudaMemcpy(table_, &minus1, 4, cudaMemcpyHostToDevice);
    cudaMemset(sel_, 0, (size_t) t_max * 4);
    std::vector<float> ones((size_t) t_max, 1.0f);
    cudaMemcpy(w_, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(counters_, 0, nl * 4);
    stage_buf_ = dev_.alloc_buffer(stage_bytes_ + 4096, "cpu_ffn.stage");
    stage_     = dev_.buffer_base(stage_buf_);

    // CPU pool: one worker per free physical core by default (as in doorbell_moe.cpp: SMT siblings slow the pool down)
    CpuPlan plan = plan_cpus(threads > 0 ? threads : 0);
    if (threads <= 0 && plan.n_phys > 0 && (int) plan.pool.size() > plan.n_phys) plan.pool.resize((size_t) plan.n_phys);
    std::vector<int> cpus;
    if (!plan.pool.empty()) {
        pin_this_thread(plan.launch);
        cpus = plan.pool;
    }
    pool_ = std::make_unique<MissPool>(banks, N_, 1, t_max_, threads > 0 ? threads : 8, cpus);
    fprintf(stderr, "[ffn-cpu] %zu dense FFN layer(s) in host memory (%.2f GiB, no VRAM), decode rows <= %lld on %zu CPU "
                    "threads (%s); prefill staged through %.0f MiB of VRAM\n",
            nl, host / 1073741824.0, (long long) t_max_, cpus.size(), cpu_list_str(cpus).c_str(), stage_bytes_ / 1048576.0);
}

CpuFfn::~CpuFfn() {
    fprintf(stderr, "[ffn-cpu] %s\n", stats_line().c_str());
    pool_.reset();
}

Tensor CpuFfn::ffn(int idx, const op::MoeWeights & w, const Tensor & x) {
    const int64_t T = x.ne[1];
    if (x.type != F32 || !x.contiguous() || x.ne[0] != N_) throw std::runtime_error("CpuFfn::ffn: x must be f32 [N, T] contiguous");
    Device * dev = &dev_;
    if (T <= t_max_) {
        const Bell & bl  = pool_->bell(idx);
        Tensor       sl  = dev_.ws_alloc(I32, 1, T);
        Tensor       wm  = dev_.ws_alloc(F32, 1, 1, T);
        Tensor       out = dev_.ws_alloc(F32, N_, T);
        const int32_t * tab = table_, * sel = sel_;
        const float *   wp  = w_, * xp = (const float *) x.data;
        int32_t *       sp  = (int32_t *) sl.data;
        float *         wmp = (float *) wm.data, * op_ = (float *) out.data;
        unsigned *      ctr = counters_ + idx;
        const int64_t   N   = N_;
        adapter::native(dev_, [=]() {
            void * st = adapter::stream(*dev);
            kern::door_resolve_ring(st, tab, sel, wp, xp, N, 1, T, 0, sp, wmp, bl.d_miss, bl.d_w, bl.d_x, bl.d_n_tok, bl.d_seq, ctr);
            cudaMemsetAsync(op_, 0, (size_t) (N * T) * 4, (cudaStream_t) st);
            kern::door_wait_add(st, bl.d_done, ctr, sp, 0, bl.d_yp, op_, N, 1, T);
        });
        n_decode_++;
        return out;
    }
    if (op::recording()) throw std::runtime_error("CpuFfn::ffn: a recorded round wider than t_max (raise AI_FFN_CPU_TMAX)");
    // prefill: up, gate, down in turn through the staging buffer (each copy is ordered after the previous kernel)
    auto staged = [&](const Tensor & h) {
        const void * src = h.data;
        void *       dst = stage_;
        const size_t n   = h.nbytes();
        adapter::native(dev_, [=]() {
            cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t) adapter::stream(*dev));
        });
        Tensor t = h;
        t.buf    = stage_buf_;
        t.data   = stage_;
        return t;
    };
    const Tensor u = op::linear(dev_, staged(w.sh_up), x);
    const Tensor g = op::linear(dev_, staged(w.sh_gate), x);
    Tensor h;
    {
        adapter::Call c(dev_);
        h = c.run(ggml_swiglu_split(c.ctx(), c.in(g, "ffn_cpu_gate"), c.in(u, "ffn_cpu_up")));
    }
    n_staged_++;
    return op::linear(dev_, staged(w.sh_down), h);
}

std::string CpuFfn::stats_line() const {
    const auto & ps = pool_->stats();
    char buf[256];
    snprintf(buf, sizeof buf, "decode calls %llu (CPU rings %llu, busy %.1f ms), staged prefill calls %llu",
             (unsigned long long) n_decode_, (unsigned long long) ps.rings.load(), ps.busy_ns.load() / 1e6,
             (unsigned long long) n_staged_);
    return buf;
}

}  // namespace ai::offload
