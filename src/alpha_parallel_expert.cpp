#include "alpha_parallel_expert.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>

// Detect AVX-VNNI at runtime
static bool detect_vnni() {
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 7, 1);
    return (regs[0] >> 4) & 1;  // EAX bit 4 = AVX-VNNI
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __asm__ __volatile__("cpuid" : "=a"(eax),"=b"(ebx),"=c"(ecx),"=d"(edx) : "a"(7),"c"(1));
    return (eax >> 4) & 1;
#endif
#else
    return false;
#endif
}

void alpha_parallel_expert_init(
        alpha_parallel_expert_ctx * pctx,
        int64_t n_embd, int64_t n_ff) {
    memset(pctx, 0, sizeof(*pctx));

    pctx->n_embd = n_embd;

    // Context size: enough for input tensor + weight views + intermediate + output
    // gate_up matmul produces [n_ff*2], swiglu produces [n_ff], down produces [n_embd]
    // Plus graph overhead
    pctx->ctx_size = 512 * 1024 + (size_t)n_embd * n_ff * 8;

    for (int i = 0; i < 2; i++) {
        struct ggml_init_params params = { pctx->ctx_size, nullptr, false };
        pctx->ctx[i] = ggml_init(params);
        pctx->output[i] = (float *)malloc(n_embd * sizeof(float));
    }

    // Thread allocation: half of physical cores per expert
    // hardware_concurrency() returns logical threads; /2 = physical; /2 = per expert
    int hw = (int)std::thread::hardware_concurrency();
    pctx->total_threads = std::max(hw / 2, 2);          // physical cores
    pctx->threads_per_expert = std::max(pctx->total_threads / 2, 1);  // per expert

    pctx->use_vnni = detect_vnni();

    pctx->total_calls = 0;
    pctx->total_ms = 0;

    fprintf(stderr, "[PEXPERT] init: n_embd=%lld, n_ff=%lld, %d threads/expert (total %d), VNNI=%s\n",
            (long long)n_embd, (long long)n_ff,
            pctx->threads_per_expert, pctx->total_threads,
            pctx->use_vnni ? "yes" : "no");
}

void alpha_parallel_expert_free(alpha_parallel_expert_ctx * pctx) {
    for (int i = 0; i < 2; i++) {
        if (pctx->ctx[i]) ggml_free(pctx->ctx[i]);
        free(pctx->output[i]);
        pctx->ctx[i] = nullptr;
        pctx->output[i] = nullptr;
    }
}

// Compute a single expert FFN in a given ggml context.
// This is the core compute function called by each thread.
static void compute_expert_ffn(
        struct ggml_context * ctx,
        struct ggml_tensor * gate_up_all,   // merged gate+up [n_embd, n_ff*2, n_expert] or nullptr
        struct ggml_tensor * gate_all,      // separate gate [n_embd, n_ff, n_expert] or nullptr
        struct ggml_tensor * up_all,        // separate up [n_embd, n_ff, n_expert] or nullptr
        struct ggml_tensor * down_all,      // down [n_ff, n_embd, n_expert]
        int expert_id,
        size_t slice_bytes_gu,              // bytes per expert for gate_up (or gate)
        size_t slice_bytes_d,               // bytes per expert for down
        const float * hidden_state,
        float * output,
        int64_t n_embd, int64_t n_ff,
        int n_threads) {

    // Reset context for reuse (free all tensors but keep the buffer)
    ggml_free(ctx);
    size_t ctx_size = 512 * 1024 + (size_t)n_embd * n_ff * 8;
    struct ggml_init_params params = { ctx_size, nullptr, false };
    ctx = ggml_init(params);
    if (!ctx) return;

    // Input tensor
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    memcpy(input->data, hidden_state, n_embd * sizeof(float));

    ggml_tensor * result = nullptr;

    if (gate_up_all) {
        // Merged gate+up path
        ggml_tensor * gate_up_w = ggml_view_2d(ctx, gate_up_all,
            gate_up_all->ne[0], gate_up_all->ne[1],
            gate_up_all->nb[1], expert_id * slice_bytes_gu);

        ggml_tensor * gate_up = ggml_mul_mat(ctx, gate_up_w, input);
        ggml_tensor * activated = ggml_swiglu(ctx, gate_up);

        ggml_tensor * down_w = ggml_view_2d(ctx, down_all,
            down_all->ne[0], down_all->ne[1],
            down_all->nb[1], expert_id * slice_bytes_d);

        result = ggml_mul_mat(ctx, down_w, activated);
    } else if (gate_all && up_all) {
        // Separate gate + up path
        size_t slice_bytes_g = slice_bytes_gu;  // reuse for gate
        size_t slice_bytes_u = slice_bytes_gu;  // same size for up

        ggml_tensor * gate_w = ggml_view_2d(ctx, gate_all,
            gate_all->ne[0], gate_all->ne[1],
            gate_all->nb[1], expert_id * slice_bytes_g);
        ggml_tensor * up_w = ggml_view_2d(ctx, up_all,
            up_all->ne[0], up_all->ne[1],
            up_all->nb[1], expert_id * slice_bytes_u);

        ggml_tensor * gate_out = ggml_mul_mat(ctx, gate_w, input);
        ggml_tensor * up_out = ggml_mul_mat(ctx, up_w, input);
        ggml_tensor * activated = ggml_swiglu_split(ctx, gate_out, up_out);

        ggml_tensor * down_w = ggml_view_2d(ctx, down_all,
            down_all->ne[0], down_all->ne[1],
            down_all->nb[1], expert_id * slice_bytes_d);

        result = ggml_mul_mat(ctx, down_w, activated);
    }

    if (result) {
        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, result);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        memcpy(output, result->data, n_embd * sizeof(float));
    }
}

void alpha_parallel_expert_compute(
        alpha_parallel_expert_ctx * pctx,
        // Expert 0
        struct ggml_tensor * gate_up_0, struct ggml_tensor * gate_0,
        struct ggml_tensor * up_0, struct ggml_tensor * down_0,
        int expert_id_0, size_t slice_bytes_gu_0, size_t slice_bytes_d_0,
        // Expert 1
        struct ggml_tensor * gate_up_1, struct ggml_tensor * gate_1,
        struct ggml_tensor * up_1, struct ggml_tensor * down_1,
        int expert_id_1, size_t slice_bytes_gu_1, size_t slice_bytes_d_1,
        // Input
        const float * hidden_state,
        int64_t n_embd, int64_t n_ff) {

    auto t0 = std::chrono::high_resolution_clock::now();

    int tpe = pctx->threads_per_expert;

    // Launch expert 1 on a separate thread (uses pctx->ctx[1])
    std::thread t1([&]{
        compute_expert_ffn(
            pctx->ctx[1],
            gate_up_1, gate_1, up_1, down_1,
            expert_id_1, slice_bytes_gu_1, slice_bytes_d_1,
            hidden_state, pctx->output[1],
            n_embd, n_ff, tpe);
    });

    // Expert 0 on current thread (uses pctx->ctx[0])
    compute_expert_ffn(
        pctx->ctx[0],
        gate_up_0, gate_0, up_0, down_0,
        expert_id_0, slice_bytes_gu_0, slice_bytes_d_0,
        hidden_state, pctx->output[0],
        n_embd, n_ff, tpe);

    t1.join();

    auto t1_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1_end - t0).count();

    pctx->total_calls++;
    pctx->total_ms += ms;

    // Log first few calls and periodic after
    if (pctx->total_calls <= 3 || (pctx->total_calls % 100 == 0)) {
        fprintf(stderr, "[PEXPERT] 2 experts parallel: %.2fms (avg %.2fms, %lld calls, %d threads/expert)\n",
                ms, pctx->total_ms / pctx->total_calls,
                (long long)pctx->total_calls, tpe);
    }
}

void alpha_parallel_expert_print_stats(const alpha_parallel_expert_ctx * pctx) {
    if (pctx->total_calls > 0) {
        fprintf(stderr, "\n=== Parallel Expert Stats ===\n");
        fprintf(stderr, "  Calls: %lld\n", (long long)pctx->total_calls);
        fprintf(stderr, "  Avg time: %.2f ms (2 experts parallel)\n",
                pctx->total_ms / pctx->total_calls);
        fprintf(stderr, "  Total time: %.1f ms\n", pctx->total_ms);
        fprintf(stderr, "  Threads/expert: %d (total %d)\n",
                pctx->threads_per_expert, pctx->total_threads);
        fprintf(stderr, "  VNNI: %s\n", pctx->use_vnni ? "active" : "not available");
        fprintf(stderr, "\n");
    }
}
