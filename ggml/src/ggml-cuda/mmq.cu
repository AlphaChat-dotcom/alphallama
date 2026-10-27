#include "common.cuh"
#include "mmq.cuh"
#include "dre.cuh"

#include <cstdio>
#include <cstdlib>
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdlib>
#include <vector>

// ALPHA_MMQ_SPLIT_PROF (default off): split MMQ's cost into quantise vs GEMM.
// ALPHA_OP_PROF cannot answer this: quantize_mmq_q8_1_cuda runs inside ggml_cuda_mul_mat_q, so both
// halves land in one MUL_MAT node and the op table shows a single number.
//
// Event-based, like ALPHA_OP_PROF: a cudaDeviceSynchronize per call would serialise the pipelining
// under measurement. Events are recorded on the compute stream and tallied every _EVERY calls.
// Per-event overhead inflates absolute ms; the share (quantise vs gemm) is the usable quantity.
struct alpha_mmq_split_prof {
    std::vector<cudaEvent_t> qb, qe, gb, ge;
    size_t n = 0;
    double q_us = 0.0, g_us = 0.0;
    long long calls = 0, reports = 0;
};
static alpha_mmq_split_prof g_mmq_sp;

static bool alpha_mmq_sp_on() {
    static const bool v = []() {
        const char * e = getenv("ALPHA_MMQ_SPLIT_PROF");
        return e && atoi(e) != 0;
    }();
    return v;
}
static int alpha_mmq_sp_every() {
    static const int v = []() {
        const char * e = getenv("ALPHA_MMQ_SPLIT_PROF_EVERY");
        const int x = e ? atoi(e) : 512;
        return x < 1 ? 1 : x;
    }();
    return v;
}
// Grow the event pool to cover slot n. Events are reused across reports.
static void alpha_mmq_sp_reserve(size_t n) {
    alpha_mmq_split_prof & p = g_mmq_sp;
    while (p.qb.size() <= n) {
        cudaEvent_t a, b, c, d;
        cudaEventCreateWithFlags(&a, 0); cudaEventCreateWithFlags(&b, 0);
        cudaEventCreateWithFlags(&c, 0); cudaEventCreateWithFlags(&d, 0);
        p.qb.push_back(a); p.qe.push_back(b); p.gb.push_back(c); p.ge.push_back(d);
    }
}
static void alpha_mmq_sp_flush(cudaStream_t st) {
    alpha_mmq_split_prof & p = g_mmq_sp;
    if (p.n == 0) return;
    cudaStreamSynchronize(st);
    for (size_t i = 0; i < p.n; i++) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, p.qb[i], p.qe[i]) == cudaSuccess) p.q_us += ms * 1000.0;
        if (cudaEventElapsedTime(&ms, p.gb[i], p.ge[i]) == cudaSuccess) p.g_us += ms * 1000.0;
    }
    p.n = 0;
    p.reports++;
    const double tot = p.q_us + p.g_us;
    fprintf(stderr, "[mmq-split] calls=%lld  quantise=%.3f ms (%.1f%%)  gemm=%.3f ms (%.1f%%)  "
                    "per-call: q=%.4f g=%.4f ms\n",
            (long long) p.calls,
            p.q_us / 1000.0, tot > 0 ? 100.0 * p.q_us / tot : 0.0,
            p.g_us / 1000.0, tot > 0 ? 100.0 * p.g_us / tot : 0.0,
            p.calls ? p.q_us / 1000.0 / (double) p.calls : 0.0,
            p.calls ? p.g_us / 1000.0 / (double) p.calls : 0.0);
}

// Pool-direct MoE offset table (defined in ggml-backend.cpp / ggml-cuda.cu), mirrors mmvq.cu.
GGML_API_C void * ggml_backend_get_tensor_offsets(const void * tensor_ptr);
struct pcache_offset_info {
    void *    pool_base;
    int64_t * d_offsets;
};

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_q_case<GGML_TYPE_IQ1_M>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// ALPHA_MMQ_JBOUND=0 disables the per-tile column bound of the expert GEMM (see mul_mat_q_process_tile)
static bool mmid_bound_cols() {
    static const int v = []() {
        const char * e = getenv("ALPHA_MMQ_JBOUND");
        return e ? atoi(e) : 1;
    }();
    return v != 0;
}

static bool mmid_use_stream_k(bool default_sk) {
    static const int v = []() {
        const char * e = getenv("ALPHA_MMID_STREAMK");
        return e ? atoi(e) : 1;
    }();
    return v != 0 && default_sk;
}

// ALPHA_MMID_CACHE: see mmq.cuh. Persistent raw device buffers (grown on demand; a regrow invalidates).
// ALPHA_MMID_CACHE=0 disables; ALPHA_MMID_CACHE_STATS=N prints counters every N graphs.
struct alpha_mmid_cache {
    int32_t *    ids_buf = nullptr;  size_t ids_cap = 0;        // [ids_src1 | ids_dst | expert_bounds]
    const void * ids_data = nullptr; size_t ids_bytes = 0;
    int64_t      k_ne02 = 0, k_ne12 = 0, k_nused = 0, k_ne11 = 0, k_si1 = 0, k_sis1 = 0;
    bool         ids_valid = false;
    char *       q8_buf = nullptr;   size_t q8_cap = 0;
    const void * src1_data = nullptr; size_t src1_bytes = 0;
    int64_t      q_ne10 = 0, q_s11 = 0, q_s12 = 0, q_s13 = 0;
    int          q_type = -1;
    bool         q8_valid = false;
    bool         fail = false;
    int          dev = -1;   // the buffers belong to one device; any other device bypasses the cache
    long long    ids_hit = 0, ids_miss = 0, q8_hit = 0, q8_miss = 0, graphs = 0;
};
static alpha_mmid_cache g_mmidc;

static bool alpha_mmid_cache_on() {
    static const bool v = []() {
        const char * e = getenv("ALPHA_MMID_CACHE");
        return e ? atoi(e) != 0 : true;
    }();
    return v;
}

void ggml_cuda_mmq_id_cache_begin_graph() {
    g_mmidc.ids_valid = false;
    g_mmidc.q8_valid  = false;
    g_mmidc.graphs++;
    static const long long stats = []() {
        const char * e = getenv("ALPHA_MMID_CACHE_STATS");
        return e ? atoll(e) : 0LL;
    }();
    if (stats > 0 && g_mmidc.graphs % stats == 0) {
        fprintf(stderr, "[mmid-cache] graphs=%lld ids hit/miss=%lld/%lld q8 hit/miss=%lld/%lld\n",
                g_mmidc.graphs, g_mmidc.ids_hit, g_mmidc.ids_miss, g_mmidc.q8_hit, g_mmidc.q8_miss);
    }
}

void ggml_cuda_mmq_id_cache_note_write(const void * data, size_t nbytes) {
    if ((!g_mmidc.ids_valid && !g_mmidc.q8_valid) || data == nullptr || nbytes == 0) {
        return;
    }
    const char * a = (const char *) data;
    auto hits = [&](const void * d, size_t n) {
        const char * b = (const char *) d;
        return a < b + n && b < a + nbytes;
    };
    if (g_mmidc.ids_valid && hits(g_mmidc.ids_data, g_mmidc.ids_bytes)) {
        g_mmidc.ids_valid = false;
        g_mmidc.q8_valid  = false;   // the gathered rows follow the ids
    }
    if (g_mmidc.q8_valid && hits(g_mmidc.src1_data, g_mmidc.src1_bytes)) {
        g_mmidc.q8_valid = false;
    }
}

// grow a raw device buffer; false on allocation failure (caller falls back to the pool path)
static bool alpha_mmid_cache_reserve(void ** buf, size_t * cap, size_t need) {
    if (*buf != nullptr && *cap >= need) {
        return true;
    }
    if (*buf != nullptr) {
        cudaFree(*buf);
        *buf = nullptr;
        *cap = 0;
    }
    size_t want = need + need / 4;
    void * b = nullptr;
    if (cudaMalloc(&b, want) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    *buf = b;
    *cap = want;
    return true;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k =
#ifdef GGML_CUDA_BATCH_INVARIANT
        // Batch-invariant: stream-K splits the K-contraction across SMs and recombines with a
        // += fixup whose split point = f(batch M) -> non-reproducible across batch widths.
        // Force the full-K-per-block path (mul_mat_q, direct write_back, no fixup) instead.
        false;
#else
        (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);
#endif

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        const bool sp_on = alpha_mmq_sp_on();
        size_t sp_slot = 0;
        if (sp_on) {
            sp_slot = g_mmq_sp.n;
            alpha_mmq_sp_reserve(sp_slot);
        }
        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (sp_on) cudaEventRecord(g_mmq_sp.qb[sp_slot], stream);
            if (use_native_mxfp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            if (sp_on) cudaEventRecord(g_mmq_sp.qe[sp_slot], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1, nullptr, false, nullptr};
        if (sp_on) cudaEventRecord(g_mmq_sp.gb[sp_slot], stream);
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        if (sp_on) {
            cudaEventRecord(g_mmq_sp.ge[sp_slot], stream);
            g_mmq_sp.n++;
            g_mmq_sp.calls++;
            if ((int) g_mmq_sp.n >= alpha_mmq_sp_every()) alpha_mmq_sp_flush(stream);
        }
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    int32_t * ids_src1_p = nullptr, * ids_dst_p = nullptr, * expert_bounds_p = nullptr;

    // ALPHA_MMID_CACHE: prefill-sized batches only, never inside a CUDA graph capture
    const int si1_c  = (int) (ids->nb[1] / ggml_element_size(ids));
    const int sis1_c = (int) (nb12 / nb11);
    bool use_cache = false;
    {
        cudaStreamCaptureStatus cap_st = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &cap_st));
        if (cap_st != cudaStreamCaptureStatusNone) {
            g_mmidc.ids_valid = false;
            g_mmidc.q8_valid  = false;
        } else if (alpha_mmid_cache_on() && !g_mmidc.fail && ne12 >= 32 &&
                   (g_mmidc.dev < 0 || g_mmidc.dev == ggml_cuda_get_device())) {
            g_mmidc.dev = ggml_cuda_get_device();
            void * vb = g_mmidc.ids_buf;
            const bool had = vb != nullptr;
            const size_t need = (size_t) (2 * ne_get_rows + ne02 + 1) * sizeof(int32_t);
            if (alpha_mmid_cache_reserve(&vb, &g_mmidc.ids_cap, need)) {
                if (!had || vb != (void *) g_mmidc.ids_buf) { g_mmidc.ids_valid = false; g_mmidc.q8_valid = false; }
                g_mmidc.ids_buf = (int32_t *) vb;
                use_cache = true;
            } else {
                g_mmidc.fail = true;
            }
        }
    }
    bool ids_from_cache = false;
    if (use_cache) {
        ids_src1_p      = g_mmidc.ids_buf;
        ids_dst_p       = g_mmidc.ids_buf + ne_get_rows;
        expert_bounds_p = g_mmidc.ids_buf + 2 * ne_get_rows;
        // ids_dst / expert_bounds depend only on the ids; ids_src1 = it*sis1 + iex % ne11 also on the src1 layout
        const bool base_hit = g_mmidc.ids_valid && g_mmidc.ids_data == ids->data && g_mmidc.k_ne02 == ne02 &&
            g_mmidc.k_ne12 == ne12 && g_mmidc.k_nused == n_expert_used && g_mmidc.k_si1 == si1_c;
        if (base_hit && g_mmidc.k_ne11 == ne11 && g_mmidc.k_sis1 == sis1_c) {
            ids_from_cache = true;
        } else if (base_hit && ne11 == n_expert_used && sis1_c == n_expert_used) {
            // one src1 row per (token, slot), e.g. the down projection's input: ids_src1 == ids_dst exactly
            ids_src1_p     = ids_dst_p;
            ids_from_cache = true;
        }
    } else {
        ids_src1.alloc(ne_get_rows);
        ids_dst.alloc(ne_get_rows);
        expert_bounds.alloc(ne02 + 1);
        ids_src1_p      = ids_src1.get();
        ids_dst_p       = ids_dst.get();
        expert_bounds_p = expert_bounds.get();
    }
    // ALPHA_MMID_PROF=1: synchronous event split of the MoE MMQ path (ids helper / q8_1 gather-quantize / MMQ)
    static const bool mmid_prof = getenv("ALPHA_MMID_PROF") != nullptr;
    static double mmid_ms[3] = {0.0, 0.0, 0.0};
    static int    mmid_calls = 0;
    cudaEvent_t   mmid_ev[4] = {nullptr, nullptr, nullptr, nullptr};
    if (mmid_prof) {
        for (int i = 0; i < 4; i++) { CUDA_CHECK(cudaEventCreate(&mmid_ev[i])); }
        CUDA_CHECK(cudaEventRecord(mmid_ev[0], stream));
    }

    if (ids_from_cache) {
        g_mmidc.ids_hit++;
    } else {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1_p, ids_dst_p, expert_bounds_p,
            ne02, ne12, n_expert_used, ne11, si1_c, sis1_c, stream);
        CUDA_CHECK(cudaGetLastError());
        if (use_cache) {
            g_mmidc.ids_miss++;
            g_mmidc.ids_data = ids->data; g_mmidc.ids_bytes = ggml_nbytes(ids);
            g_mmidc.k_ne02 = ne02; g_mmidc.k_ne12 = ne12; g_mmidc.k_nused = n_expert_used;
            g_mmidc.k_ne11 = ne11; g_mmidc.k_si1 = si1_c; g_mmidc.k_sis1 = sis1_c;
            g_mmidc.ids_valid = true;
            g_mmidc.q8_valid  = false;
        }
    }
    if (mmid_prof) { CUDA_CHECK(cudaEventRecord(mmid_ev[1], stream)); }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    char * src1_q8_p = nullptr;

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        bool q8_cached = false;
        if (use_cache) {
            void * qb = g_mmidc.q8_buf;
            const bool had = qb != nullptr;
            if (alpha_mmid_cache_reserve(&qb, &g_mmidc.q8_cap, nbytes_src1_q8_1)) {
                if (!had || qb != (void *) g_mmidc.q8_buf) { g_mmidc.q8_valid = false; }
                g_mmidc.q8_buf = (char *) qb;
                src1_q8_p = g_mmidc.q8_buf;
                q8_cached = true;
            } else {
                g_mmidc.fail = true;
            }
        }
        if (!q8_cached) {
            src1_q8_1.alloc(nbytes_src1_q8_1);
            src1_q8_p = src1_q8_1.get();
        }
        // the gathered layout depends on the weight type (block_q8_1_mmq layouts), so key on it
        const bool q8_hit = q8_cached && g_mmidc.q8_valid && g_mmidc.ids_valid && g_mmidc.src1_data == src1->data &&
            g_mmidc.q_ne10 == ne10 && g_mmidc.q_s11 == s11 && g_mmidc.q_s12 == s12 && g_mmidc.q_s13 == s13 &&
            g_mmidc.q_type == (int) src0->type;
        if (q8_hit) {
            g_mmidc.q8_hit++;
        } else {
            if (use_native_mxfp4) {
                quantize_mmq_mxfp4_cuda(src1_d, ids_src1_p, src1_q8_p, src0->type, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            } else {
                quantize_mmq_q8_1_cuda(src1_d, ids_src1_p, src1_q8_p, src0->type, ne10, s11, s12, s13,
                                       ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
            CUDA_CHECK(cudaGetLastError());
            if (q8_cached) {
                g_mmidc.q8_miss++;
                g_mmidc.src1_data = src1->data; g_mmidc.src1_bytes = ggml_nbytes(src1);
                g_mmidc.q_ne10 = ne10; g_mmidc.q_s11 = s11; g_mmidc.q_s12 = s12; g_mmidc.q_s13 = s13;
                g_mmidc.q_type = (int) src0->type;
                g_mmidc.q8_valid = g_mmidc.ids_valid;
            }
        }
    }
    if (mmid_prof) { CUDA_CHECK(cudaEventRecord(mmid_ev[2], stream)); }

    const int64_t s12 = use_native_mxfp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                                           ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Pool-direct MoE: if the pcache flush published a per-expert offset table for this
    // tensor, read weights straight from the pool base at expert_offsets[expert] instead of
    // the contiguous per-expert stride. MMQ runs gate/up as separate MUL_MAT_ID ops, so each
    // call keys the table on its own src0 (no dual gate table here). nullptr => unchanged
    // contiguous-stride behavior.
    const char    * x_base             = src0_d;
    const int64_t * expert_offsets_ptr = nullptr;
    static const bool mmq_offset_disabled = getenv("ALPHA_NO_OFFSET_TABLE") != nullptr;
    // Single-token only, mirroring mmvq.cu (`ncols_dst == 1`). On a multi-token pass the host
    // computes pool_direct=false and writes the real weights into input_cpy via a deferred D2D,
    // but upload_table still publishes a table. Reading it would take the live pool instead of
    // that snapshot; the pool keeps churning during a long prefill (misses for later layers
    // insert and the scratch ring wraps) while these kernels are only enqueued, so the bytes
    // would move under them and select wrong-but-in-bounds experts. ggml-cuda.cu asserts the
    // same invariant ("Multi-token passes never use tables"). ids->ne[1] is the token count the
    // pcache itself gates on (ggml_cuda_expert_pcache_set_ntokens).
    const bool mmq_table_ok = ids && ids->ne[1] == 1;
    if (!mmq_offset_disabled && mmq_table_ok) {
        pcache_offset_info * info = (pcache_offset_info *) ggml_backend_get_tensor_offsets(src0);
        if (info && info->d_offsets && info->pool_base) {
            x_base             = (const char *) info->pool_base;
            expert_offsets_ptr = info->d_offsets;
        }
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        x_base, src0->type, (const int *) src1_q8_p, ids_dst_p, expert_bounds_p, dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        // ALPHA_MMID_STREAMK: stream-k partitions the (tile x k-block) space uniformly across the
        // persistent blocks, but for MUL_MAT_ID the tile space is sparse: ntx is derived from the
        // whole batch while each expert fills only ~ne12*n_used/n_expert rows, so on a large
        // prefill most tiles are empty and the real work lands on whichever blocks happen to own
        // the non-empty ranges. ALPHA_MMID_STREAMK=0 uses the plain xy grid instead (one block per
        // tile, empties exit at once, hardware load balancing).
        mmid_use_stream_k(use_stream_k), ne12, expert_offsets_ptr, mmid_bound_cols(), nullptr};

    // ALPHA_DRE_PF_PIPE: the experts are still landing in expert groups: run the GEMM one group at a time,
    // each behind its own copy event (a slice of the expert range: shifted x, bounds and channel count)
    const int * pipe_end = nullptr; cudaEvent_t const * pipe_ev = nullptr;
    const int pipe_ng = (expert_offsets_ptr == nullptr && x_base == src0_d) ? ggml_cuda_dre_pipe_groups(src0_d, &pipe_end, &pipe_ev) : 0;
    if (pipe_ng > 0) {
        int e0 = 0;
        for (int g = 0; g < pipe_ng; g++) {
            const int e1 = (int) std::min<int64_t>(pipe_end[g], ne02);
            CUDA_CHECK(cudaStreamWaitEvent(stream, pipe_ev[g], 0));
            if (e1 > e0) {
                mmq_args a = args;
                a.x             = x_base + (size_t) e0 * src0->nb[2];
                a.expert_bounds = expert_bounds_p + e0;
                a.nchannels_x   = e1 - e0;
                a.nchannels_y   = e1 - e0;
                ggml_cuda_mul_mat_q_switch_type(ctx, a, stream);
            }
            e0 = e1;
        }
        if (dst->nb[2] == dst->nb[1] * dst->ne[1]) {
            (void) ggml_cuda_dre_pfc_fill(src0_d, dst_d, dst->ne[0], (int64_t) (dst->nb[1] / sizeof(float)), stream);
        }
    } else {
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    }
    if (mmid_prof) {
        CUDA_CHECK(cudaEventRecord(mmid_ev[3], stream));
        CUDA_CHECK(cudaEventSynchronize(mmid_ev[3]));
        for (int i = 0; i < 3; i++) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, mmid_ev[i], mmid_ev[i + 1]));
            mmid_ms[i] += ms;
        }
        for (int i = 0; i < 4; i++) { CUDA_CHECK(cudaEventDestroy(mmid_ev[i])); }
        if (++mmid_calls % 120 == 0) {
            fprintf(stderr, "[mmid-prof] calls=%d ne12=%lld n_used=%lld n_exp=%lld  per-call ms: ids=%.4f quant=%.4f mmq=%.4f\n",
                    mmid_calls, (long long) ne12, (long long) n_expert_used, (long long) ne02,
                    mmid_ms[0] / mmid_calls, mmid_ms[1] / mmid_calls, mmid_ms[2] / mmid_calls);
        }
    }
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k =
#ifdef GGML_CUDA_BATCH_INVARIANT
        false;  // batch-invariant: no stream-K K-split (see note at the MoE launch site above)
#else
        ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
#endif
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols, nullptr, false, nullptr};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:   // expert down projections; otherwise prefill falls back to a per-expert cuBLAS loop
        case GGML_TYPE_IQ1_M:  // via the Q6_K tile (load_tiles_iq1_m); otherwise the same per-expert cuBLAS fallback
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
