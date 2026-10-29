// SPDX-License-Identifier: Apache-2.0
// ALPHA_DRE_PF_CPU=K (included at the end of dre.cu): for prompt chunks under ALPHA_DRE_PF_PIPE, a non-resident routed
// expert that only K or fewer of the chunk's tokens route to is not copied over PCIe; the CPU computes it from RAM
// instead, in parallel with the GPU's resident and copied experts. For so few tokens the CPU work is cheaper than
// copying the whole expert:
//   routing sync     the scheduler hands the ids and the MoE input (ggml_cuda_dre_prefill_x)
//   pipe issue       cold experts are left out of the copies; a gather kernel copies their tokens' input rows into
//                    mapped host memory and the job goes to the CPU thread
//   CPU thread       gate, up, SwiGLU, down per (expert, token) with the decode path's dot kernels, on the CPU pool
//   after the down   a fill kernel waits for the job's flag and writes the CPU rows over the GEMM's (garbage) rows
// The rows are exact in the CPU's arithmetic (as decode's CPU misses are), not in the GPU's.

namespace {

struct dre_pfc_entry { int e; int row; int slot; };   // expert, gathered token row, output slot (token * n_used + j)

struct dre_pfc_state {
    // pending at the routing sync
    bool valid = false;
    int li = -1, ne0 = 0, ne1 = 0;
    std::vector<int32_t> ids;            // [ne1][ne0]
    const char * x = nullptr; int64_t x_tok = 0; int n_embd = 0;
    // the issued job
    int job_li = -1;                     // the job awaiting its fill (reset by the fill)
    int cpu_li = -1;                     // the job the CPU thread computes (never reset under it)
    std::vector<dre_pfc_entry> ent;      // sorted by expert
    std::vector<int> e_first;            // per cold expert: first entry index (and a sentinel)
    std::vector<int> e_id;
    int n_rows = 0;
    // mapped pinned buffers
    float * xb_h = nullptr; float * xb_d = nullptr; size_t xb_cap = 0;        // gathered input rows
    float * yb_h = nullptr; float * yb_d = nullptr; size_t yb_cap = 0;        // output rows, one per entry
    int32_t * rows_h = nullptr; int32_t * rows_d = nullptr; size_t rows_cap = 0;   // gather: token index per row
    int32_t * slot_h = nullptr; int32_t * slot_d = nullptr; size_t slot_cap = 0;   // fill: output slot per entry
    volatile uint32_t * done_h = nullptr; uint32_t * done_d = nullptr;
    uint32_t seq = 0;
    cudaEvent_t ev_x = nullptr;
    // CPU thread
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    bool has_job = false;
    int device = 0;
    long long n_jobs = 0, n_experts = 0, n_entries = 0, cpu_us = 0;
} g_pfc;

static double dre_pfc_ms() { static const auto t0 = std::chrono::steady_clock::now(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

int dre_pfc_k() {
    static const int v = dre_env_int("ALPHA_DRE_PF_CPU", 8);
    // ALPHA_DRE_PF_CPU_DIV=d: the threshold grows with the chunk: max(K, n_tokens / d). A long chunk's coldest experts
    // carry almost none of its tokens, so the CPU can take more of them without falling behind the GPU
    static const int div = dre_env_int("ALPHA_DRE_PF_CPU_DIV", 0);
    if (v <= 0) return v;
    return div > 0 ? std::max(v, g_dre_pf_ntok / div) : v;
}

template <typename T> static void dre_pfc_grow(T ** h, T ** d, size_t * cap, size_t need) {
    if (*cap >= need) return;
    if (*h) CUDA_CHECK(cudaFreeHost(*h));
    const size_t n = std::max(need, *cap * 2);
    CUDA_CHECK(cudaHostAlloc((void **) h, n * sizeof(T), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void **) d, *h, 0));
    *cap = n;
}

static __global__ void dre_pfc_gather_k(const char * __restrict__ x, const long long x_tok, const int32_t * __restrict__ rows,
                                        const int n_rows, const int n_embd, float * __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= n_rows) return;
    const float * src = (const float *) (x + (long long) rows[r] * x_tok);
    float * dst = out + (size_t) r * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) dst[i] = src[i];
}

static __global__ void dre_pfc_fill_k(const volatile uint32_t * done, const uint32_t want, const float * __restrict__ yb,
                                      const int32_t * __restrict__ slots, const int n_ent, const int n_out,
                                      float * __restrict__ dst, const long long slot_stride) {
    if (threadIdx.x == 0) {
        while (*done != want) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(1000);  // sm_70+ only (see dre.cu)
#endif
        }
    }
    __syncthreads();
    __threadfence_system();
    for (int i = blockIdx.x; i < n_ent; i += gridDim.x) {
        const int s = slots[i];
        const float4 * src = (const float4 *) (yb + (size_t) i * n_out);
        float4 * d = (float4 *) (dst + (long long) s * slot_stride);
        for (int j = threadIdx.x; j < n_out / 4; j += blockDim.x) d[j] = __ldcv(src + j);
    }
}

// the CPU work of one job: every cold expert in parallel over the pool
void dre_pfc_compute() {
    const dre_layer & L = g_dre.layers[g_pfc.cpu_li];
    const dre_tensor & G = g_dre.tens[L.ti[0]];
    const dre_tensor & U = g_dre.tens[L.ti[1]];
    const dre_tensor & D = g_dre.tens[L.ti[2]];
    const int K  = (int) G.t->ne[0];
    const int F  = (int) G.t->ne[1];
    const int NO = (int) D.t->ne[1];
    const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(G.t->type);
    const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(U.t->type);
    const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(D.t->type);
    const bool same_q = tu->vec_dot_type == tg->vec_dot_type;
    const size_t qg_row = ggml_row_size(tg->vec_dot_type, K), qu_row = ggml_row_size(tu->vec_dot_type, K);
    const size_t qa_row = ggml_row_size(td->vec_dot_type, F);
    const bool q2p = D.t->type == GGML_TYPE_Q2_0 && td->vec_dot_type == GGML_TYPE_Q8_0 && ggml_cpu_q2_0_prepared_available();
    const size_t q2psz = q2p ? (ggml_cpu_q2_0_prep_size(F) + 63) & ~(size_t) 63 : 0;
    const int ne = (int) g_pfc.e_id.size();
    // its own threads, not the decode pool: pool.run is a single-dispatcher protocol and this runs beside it
    static const int nthr = std::max(1, dre_env_int("ALPHA_DRE_PF_CPU_THREADS", 16));
    std::atomic<int> next_item{0};
    auto body = [&](int ie) {
        const int e = g_pfc.e_id[ie];
        const int i0 = g_pfc.e_first[ie], i1 = g_pfc.e_first[ie + 1], m = i1 - i0;
        thread_local std::vector<char> qx_g, qx_u, qa, prep;
        thread_local std::vector<float> gb, ub, ab;
        qx_g.resize(qg_row * m); qx_u.resize(qu_row * m); qa.resize(qa_row * m);
        gb.resize((size_t) F * m); ub.resize((size_t) F * m); ab.resize((size_t) F);
        if (q2p) prep.resize(q2psz * m + 64);
        for (int i = 0; i < m; i++) {
            const float * xr = g_pfc.xb_h + (size_t) g_pfc.ent[i0 + i].row * K;
            ggml_get_type_traits_cpu(tg->vec_dot_type)->from_float(xr, qx_g.data() + i * qg_row, K);
            if (!same_q) ggml_get_type_traits_cpu(tu->vec_dot_type)->from_float(xr, qx_u.data() + i * qu_row, K);
        }
        const char * qxu = same_q ? qx_g.data() : qx_u.data();
        const size_t qxu_row = same_q ? qg_row : qu_row;
        const char * wg = dre_src(G, e), * wu = dre_src(U, e), * wd = dre_src(D, e);
        const size_t rg = G.t->nb[1], ru = U.t->nb[1], rd = D.t->nb[1];
        for (int r = 0; r < F; r++) {
            for (int i = 0; i < m; i++) {
                tg->vec_dot(K, &gb[(size_t) i * F + r], 0, wg + (size_t) r * rg, 0, qx_g.data() + i * qg_row, 0, 1);
                tu->vec_dot(K, &ub[(size_t) i * F + r], 0, wu + (size_t) r * ru, 0, qxu + i * qxu_row, 0, 1);
            }
        }
        char * pbase = q2p ? (char *) (((uintptr_t) prep.data() + 63) & ~(uintptr_t) 63) : nullptr;
        for (int i = 0; i < m; i++) {
            const float * g = gb.data() + (size_t) i * F, * u = ub.data() + (size_t) i * F;
            for (int r = 0; r < F; r++) ab[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
            ggml_get_type_traits_cpu(td->vec_dot_type)->from_float(ab.data(), qa.data() + i * qa_row, F);
            if (q2p) ggml_cpu_q2_0_prepare(qa.data() + i * qa_row, F, pbase + (size_t) i * q2psz);
        }
        for (int i = 0; i < m; i++) {
            float * y = g_pfc.yb_h + (size_t) (i0 + i) * NO;
            if (q2p) {
                ggml_cpu_q2_0_dot_prepared(F, NO, wd, rd, pbase + (size_t) i * q2psz, y);
            } else {
                for (int r = 0; r < NO; r++) td->vec_dot(F, &y[r], 0, wd + (size_t) r * rd, 0, qa.data() + i * qa_row, 0, 1);
            }
        }
    };
    auto worker = [&]() {
        int ie;
        while ((ie = next_item.fetch_add(1)) < ne) body(ie);
    };
    std::vector<std::thread> th;
    const int nt = std::min(nthr, ne);
    for (int t = 1; t < nt; t++) th.emplace_back(worker);
    worker();
    for (auto & x : th) x.join();
}

void dre_pfc_thread() {
    CUDA_CHECK(cudaSetDevice(g_pfc.device));
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g_pfc.m);
            g_pfc.cv.wait(lk, [] { return g_pfc.has_job; });
            g_pfc.has_job = false;
        }
        static const bool dbg = dre_env_int("ALPHA_DRE_PF_CPU_DBG", 0) != 0;
        if (dbg) fprintf(stderr, "[pfc %.1f] thread: job seq %u li %d, waiting for rows\n", dre_pfc_ms(), g_pfc.seq, g_pfc.cpu_li);
        CUDA_CHECK(cudaEventSynchronize(g_pfc.ev_x));   // the gathered input rows are in host memory
        if (dbg) fprintf(stderr, "[pfc %.1f] thread: computing %zu experts / %zu entries\n", dre_pfc_ms(), g_pfc.e_id.size(), g_pfc.ent.size());
        const auto t0 = std::chrono::steady_clock::now();
        dre_pfc_compute();
        g_pfc.cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *g_pfc.done_h = g_pfc.seq;
        if (dbg) fprintf(stderr, "[pfc %.1f] thread: done seq %u in %.1f ms\n", dre_pfc_ms(), g_pfc.seq,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
}

// called from dre_pipe_issue: drop the layer's cold non-resident experts from the copies and start their CPU job.
// Returns true when a job was started (g_pp_used then excludes the cold experts).
bool dre_pfc_issue(int li, cudaStream_t compute_st, const uint8_t * spec) {
    const int k = dre_pfc_k();
    if (k <= 0 || !g_pfc.valid || g_pfc.li != li) return false;
    g_pfc.valid = false;
    const dre_layer & L = g_dre.layers[li];
    const int n_used = g_pfc.ne0, n_tok = g_pfc.ne1;
    std::vector<int> cnt((size_t) L.n_exp, 0);
    for (int i = 0; i < n_tok * n_used; i++) { const int e = g_pfc.ids[i]; if (e >= 0 && e < L.n_exp) cnt[e]++; }
    std::vector<uint8_t> cold((size_t) L.n_exp, 0);
    int n_cold = 0;
    for (int e = 0; e < L.n_exp; e++) if (cnt[e] > 0 && cnt[e] <= k && L.slot[e] < 0 && !(spec && spec[e])) { cold[e] = 1; n_cold++; }
    if (n_cold == 0) return false;
    // entries per cold expert, token rows deduplicated
    g_pfc.ent.clear(); g_pfc.e_first.clear(); g_pfc.e_id.clear();
    std::vector<int> row_of((size_t) n_tok, -1);
    std::vector<int32_t> rows;
    std::vector<std::vector<std::pair<int,int>>> per((size_t) L.n_exp);
    for (int t = 0; t < n_tok; t++) for (int j = 0; j < n_used; j++) {
        const int e = g_pfc.ids[t * n_used + j];
        if (e < 0 || e >= L.n_exp || !cold[e]) continue;
        if (row_of[t] < 0) { row_of[t] = (int) rows.size(); rows.push_back(t); }
        per[e].push_back({ row_of[t], t * n_used + j });
    }
    for (int e = 0; e < L.n_exp; e++) {
        if (!cold[e]) continue;
        g_pfc.e_id.push_back(e);
        g_pfc.e_first.push_back((int) g_pfc.ent.size());
        for (auto & pr : per[e]) g_pfc.ent.push_back({ e, pr.first, pr.second });
        g_pp_used[(size_t) e] = 0;   // not copied
    }
    g_pfc.e_first.push_back((int) g_pfc.ent.size());
    g_pfc.n_rows = (int) rows.size();
    const int NO = (int) g_dre.tens[L.ti[2]].t->ne[1];
    dre_pfc_grow(&g_pfc.xb_h, &g_pfc.xb_d, &g_pfc.xb_cap, (size_t) g_pfc.n_rows * g_pfc.n_embd);
    dre_pfc_grow(&g_pfc.yb_h, &g_pfc.yb_d, &g_pfc.yb_cap, g_pfc.ent.size() * (size_t) NO);
    dre_pfc_grow(&g_pfc.rows_h, &g_pfc.rows_d, &g_pfc.rows_cap, rows.size());
    dre_pfc_grow(&g_pfc.slot_h, &g_pfc.slot_d, &g_pfc.slot_cap, g_pfc.ent.size());
    std::memcpy(g_pfc.rows_h, rows.data(), rows.size() * sizeof(int32_t));
    for (size_t i = 0; i < g_pfc.ent.size(); i++) g_pfc.slot_h[i] = g_pfc.ent[i].slot;
    if (!g_pfc.done_h) {
        CUDA_CHECK(cudaHostAlloc((void **) &g_pfc.done_h, sizeof(uint32_t), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &g_pfc.done_d, (void *) g_pfc.done_h, 0));
        *g_pfc.done_h = 0;
        CUDA_CHECK(cudaEventCreateWithFlags(&g_pfc.ev_x, cudaEventDisableTiming));
        CUDA_CHECK(cudaGetDevice(&g_pfc.device));
        dre_cpu_start();
        g_pfc.th = std::thread(dre_pfc_thread);
        g_pfc.th.detach();
    }
    dre_pfc_gather_k<<<g_pfc.n_rows, 256, 0, compute_st>>>(g_pfc.x, g_pfc.x_tok, g_pfc.rows_d, g_pfc.n_rows, g_pfc.n_embd, g_pfc.xb_d);
    CUDA_CHECK(cudaEventRecord(g_pfc.ev_x, compute_st));
    g_pfc.seq++;
    g_pfc.job_li = li;
    g_pfc.cpu_li = li;
    { static const bool dbg = dre_env_int("ALPHA_DRE_PF_CPU_DBG", 0) != 0; if (dbg) fprintf(stderr, "[pfc %.1f] issue li %d seq %u\n", dre_pfc_ms(), li, g_pfc.seq); }
    g_pfc.n_jobs++; g_pfc.n_experts += n_cold; g_pfc.n_entries += (long long) g_pfc.ent.size();
    {
        std::lock_guard<std::mutex> lk(g_pfc.m);
        g_pfc.has_job = true;
    }
    g_pfc.cv.notify_one();
    return true;
}
}   // namespace

// the scheduler's prompt-layer hook: keep the routed ids and the MoE input until the layer's copies are issued
GGML_BACKEND_API_C void ggml_cuda_dre_prefill_x(void * backend_ptr, const struct ggml_tensor * x, const void * weight,
                                                const int32_t * ids, int ne0, int ne1, int s0, int s1) {
    GGML_UNUSED(backend_ptr);
    g_pfc.valid = false;
    if (dre_pfc_k() <= 0 || !x || x->type != GGML_TYPE_F32) return;
    auto it = g_dre.by_data.find(weight);
    if (it == g_dre.by_data.end() || it->second < 0) return;
    const dre_tensor & d = g_dre.tens[it->second];
    if (d.role == 2) return;   // the down projection's input is not the layer's MoE input
    const int64_t n_tok = x->ne[2] == ne1 ? x->ne[2] : x->ne[1];
    if (n_tok != ne1 || x->ne[0] != (int64_t) g_dre.tens[g_dre.layers[d.layer].ti[0]].t->ne[0]) return;
    g_pfc.ids.resize((size_t) ne0 * ne1);
    for (int t = 0; t < ne1; t++) for (int j = 0; j < ne0; j++) g_pfc.ids[(size_t) t * ne0 + j] = ids[t * s1 + j * s0];
    g_pfc.ne0 = ne0; g_pfc.ne1 = ne1;
    g_pfc.x = (const char *) x->data;
    g_pfc.x_tok = x->ne[2] == ne1 ? (int64_t) x->nb[2] : (int64_t) x->nb[1];
    g_pfc.n_embd = (int) x->ne[0];
    g_pfc.li = d.layer;
    g_pfc.valid = true;
}

// after the down GEMM of a pipelined prompt layer (mmq.cu): the CPU's rows over the GEMM's rows for the cold experts
bool ggml_cuda_dre_pfc_fill(const void * staged_down, float * dst, int64_t n_out, int64_t slot_stride, cudaStream_t stream) {
    static const bool dbg = dre_env_int("ALPHA_DRE_PF_CPU_DBG", 0) != 0;
    if (dbg) fprintf(stderr, "[pfc %.1f] fill? job_li %d ent %zu staged %p slot %p issued %d\n", dre_pfc_ms(), g_pfc.job_li, g_pfc.ent.size(),
                     staged_down, (void *) g_pfs[2][g_pp_slot].p, g_pp_issued_li);
    if (g_pfc.job_li < 0 || g_pfc.ent.empty()) return false;
    if (staged_down != g_pfs[2][g_pp_slot].p || g_pp_issued_li != g_pfc.job_li) return false;
    if (n_out % 4 != 0 || slot_stride % 4 != 0) return false;
    dre_pfc_fill_k<<<std::min<int>(256, (int) g_pfc.ent.size()), 256, 0, stream>>>(g_pfc.done_d, g_pfc.seq, g_pfc.yb_d,
        g_pfc.slot_d, (int) g_pfc.ent.size(), (int) n_out, dst, (long long) slot_stride);
    g_pfc.job_li = -1;
    return true;
}

// one line per prompt: what the CPU took off the copy engine (printed and reset at the pipe release)
void dre_pfc_report() {
    if (g_pfc.n_jobs) {
        fprintf(stderr, "[dre] prefill CPU experts: %lld layers, %lld experts, %lld token rows, %.1f ms of CPU\n",
                g_pfc.n_jobs, g_pfc.n_experts, g_pfc.n_entries, g_pfc.cpu_us / 1000.0);
    }
    g_pfc.n_jobs = g_pfc.n_experts = g_pfc.n_entries = g_pfc.cpu_us = 0;
}
