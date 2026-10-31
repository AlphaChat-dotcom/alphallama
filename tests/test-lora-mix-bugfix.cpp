// AlphaLlama Bug Fix Verification Tests
//
// Verifies the P1 fix: graph router top_k clamping to n_loras.
// 15 test cases across 4 groups:
//   Group 1: Bug fix boundary verification (5 tests)
//   Group 2: Router → Dispatch integration with clamped top_k (4 tests)
//   Group 3: CPU vs Graph router clamp parity (3 tests)
//   Group 4: Stress / boundary (3 tests)
//
// Build:
//   cmake --build build --config Release --target test-lora-mix-bugfix
//
// Run:
//   ./build/bin/Release/test-lora-mix-bugfix.exe

#include "alpha_router.h"
#include "alpha_lora_mix.h"
#include "alpha_lora_mix_graph.h"

#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// Test utilities
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        tests_failed++; \
        return false; \
    } \
} while(0)

#define TEST_FLOAT_EQ(a, b, tol, msg) do { \
    if (fabsf((a) - (b)) > (tol)) { \
        fprintf(stderr, "  FAIL: %s (line %d): %s (%.6f vs %.6f)\n", \
                __func__, __LINE__, msg, (float)(a), (float)(b)); \
        tests_failed++; \
        return false; \
    } \
} while(0)

static void test_pass(const char * name) {
    fprintf(stderr, "  PASS: %s\n", name);
    tests_passed++;
}

static ggml_context * make_test_ctx(size_t mem_size = 16 * 1024 * 1024) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    return ggml_init(params);
}

// Helper: build a graph router with given config and zero weights
static alpha_lora_mix_graph_router make_graph_router(ggml_context * ctx,
        int hidden_dim, int router_hidden, int n_loras, int top_k) {
    alpha_lora_mix_graph_router r = {};
    r.w_down  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, router_hidden);
    r.w_up    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, router_hidden, n_loras);
    r.b_down  = nullptr;
    r.b_up    = nullptr;
    r.n_loras = n_loras;
    r.top_k   = top_k;
    memset(r.w_down->data, 0, ggml_nbytes(r.w_down));
    memset(r.w_up->data, 0, ggml_nbytes(r.w_up));
    // Set some non-zero weights so softmax produces distinct values
    for (int i = 0; i < hidden_dim * router_hidden; i++) {
        ((float *)r.w_down->data)[i] = 0.1f * ((i % 7) - 3);
    }
    for (int i = 0; i < router_hidden * n_loras; i++) {
        ((float *)r.w_up->data)[i] = 0.2f * ((i % 5) - 2);
    }
    return r;
}

// Helper: write valid GGUF for CPU router tests
static void write_gguf(const char * path, int hidden_dim, int router_hidden,
                       int n_loras, int top_k) {
    gguf_context * gctx = gguf_init_empty();
    gguf_set_val_i32(gctx, "alpha_lora_mix.version", 1);
    gguf_set_val_i32(gctx, "alpha_lora_mix.n_loras", n_loras);
    gguf_set_val_i32(gctx, "alpha_lora_mix.top_k", top_k);
    gguf_set_val_i32(gctx, "alpha_lora_mix.router_hidden", router_hidden);
    gguf_set_val_bool(gctx, "alpha_lora_mix.shared", true);

    size_t mem = (size_t)hidden_dim * router_hidden * sizeof(float)
               + (size_t)router_hidden * n_loras * sizeof(float) + 1024 * 1024;
    ggml_context * tctx = make_test_ctx(std::max(mem, (size_t)(2 * 1024 * 1024)));

    ggml_tensor * w_down = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, hidden_dim, router_hidden);
    ggml_set_name(w_down, "alpha_lora_mix.router.down.weight");
    for (int i = 0; i < hidden_dim * router_hidden; i++)
        ((float *)w_down->data)[i] = 0.1f * ((i % 7) - 3);
    gguf_add_tensor(gctx, w_down);

    ggml_tensor * w_up = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, router_hidden, n_loras);
    ggml_set_name(w_up, "alpha_lora_mix.router.up.weight");
    for (int i = 0; i < router_hidden * n_loras; i++)
        ((float *)w_up->data)[i] = 0.2f * ((i % 5) - 2);
    gguf_add_tensor(gctx, w_up);

    gguf_write_to_file(gctx, path, false);
    gguf_free(gctx);
    ggml_free(tctx);
}

// ============================================================================
// Group 1: Bug Fix Verification — top_k clamping boundaries
// ============================================================================

static bool test_graph_topk_equals_nloras() {
    // TC-BF-01: top_k == n_loras — no clamping needed
    ggml_context * ctx = make_test_ctx();
    auto router = make_graph_router(ctx, 4, 2, 3, 3);  // top_k=3, n_loras=3
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));
    ((float *)hidden->data)[0] = 1.0f;

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "should produce valid indices");
    TEST_ASSERT(route.indices->ne[0] == 3, "top_k=n_loras=3, no clamping → ne[0]==3");
    ggml_free(ctx);
    return true;
}

static bool test_graph_topk_exceeds_nloras_by_1() {
    // TC-BF-02: top_k = n_loras + 1 — should clamp to n_loras
    ggml_context * ctx = make_test_ctx();
    auto router = make_graph_router(ctx, 4, 2, 3, 4);  // top_k=4, n_loras=3
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "should produce valid indices");
    TEST_ASSERT(route.indices->ne[0] == 3, "top_k=4 clamped to n_loras=3 → ne[0]==3");
    ggml_free(ctx);
    return true;
}

static bool test_graph_topk_much_larger() {
    // TC-BF-03: top_k=100, n_loras=2 — should clamp to 2
    ggml_context * ctx = make_test_ctx();
    auto router = make_graph_router(ctx, 4, 2, 2, 100);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "should produce valid indices");
    TEST_ASSERT(route.indices->ne[0] == 2, "top_k=100 clamped to n_loras=2 → ne[0]==2");
    ggml_free(ctx);
    return true;
}

static bool test_graph_topk_zero() {
    // TC-BF-04: top_k=0 — should clamp to 1
    ggml_context * ctx = make_test_ctx();
    auto router = make_graph_router(ctx, 4, 2, 3, 0);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "should produce valid indices");
    TEST_ASSERT(route.indices->ne[0] == 1, "top_k=0 clamped to 1 → ne[0]==1");
    ggml_free(ctx);
    return true;
}

static bool test_graph_topk_negative() {
    // TC-BF-05: top_k=-5 — should clamp to 1
    ggml_context * ctx = make_test_ctx();
    auto router = make_graph_router(ctx, 4, 2, 3, -5);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "should produce valid indices");
    TEST_ASSERT(route.indices->ne[0] == 1, "top_k=-5 clamped to 1 → ne[0]==1");
    ggml_free(ctx);
    return true;
}

// ============================================================================
// Group 2: Router → Dispatch integration with clamped top_k
// ============================================================================

static bool test_router_dispatch_clamped_topk() {
    // TC-BF-06: Router clamps top_k=4 to n_loras=2, then dispatch builds correctly
    ggml_context * ctx = make_test_ctx();
    int in_feat = 8, rh = 4, n_loras = 2, top_k = 4, n_tokens = 2, rank = 2, out_feat = 8;

    auto router = make_graph_router(ctx, in_feat, rh, n_loras, top_k);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "router should produce indices");
    int clamped_k = (int)route.indices->ne[0];
    TEST_ASSERT(clamped_k == 2, "top_k=4 clamped to n_loras=2");

    // Now build dispatch
    alpha_lora_mix_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_feat, rank, n_loras);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, out_feat, n_loras);
    stacked.scale = 1.0f;
    stacked.n_loras = n_loras;
    stacked.rank = rank;
    memset(stacked.stacked_a->data, 0, ggml_nbytes(stacked.stacked_a));
    memset(stacked.stacked_b->data, 0, ggml_nbytes(stacked.stacked_b));

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_feat, n_tokens);
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(base->data, 0, ggml_nbytes(base));
    memset(cur->data, 0, ggml_nbytes(cur));

    ggml_tensor * result = alpha_lora_mix_build_lora_dispatch(ctx, base, cur, &stacked, &route);
    TEST_ASSERT(result != nullptr, "dispatch should build graph");
    TEST_ASSERT(result->ne[0] == out_feat, "output dim0 correct");
    TEST_ASSERT(result->ne[1] == n_tokens, "output dim1 correct");

    ggml_free(ctx);
    return true;
}

static bool test_router_dispatch_topk1_single_lora() {
    // TC-BF-07: Minimal config — 1 LoRA, top_k=1
    ggml_context * ctx = make_test_ctx();
    int in_feat = 4, rh = 2, n_loras = 1, top_k = 1, n_tokens = 1, rank = 2, out_feat = 4;

    auto router = make_graph_router(ctx, in_feat, rh, n_loras, top_k);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(hidden->data, 0, ggml_nbytes(hidden));
    ((float *)hidden->data)[0] = 1.0f;

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "single LoRA router works");
    TEST_ASSERT(route.indices->ne[0] == 1, "top_k=1 produces 1 index");

    alpha_lora_mix_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_feat, rank, n_loras);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, out_feat, n_loras);
    stacked.scale = 1.0f;
    stacked.n_loras = n_loras;
    stacked.rank = rank;
    memset(stacked.stacked_a->data, 0, ggml_nbytes(stacked.stacked_a));
    memset(stacked.stacked_b->data, 0, ggml_nbytes(stacked.stacked_b));

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_feat, n_tokens);
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(base->data, 0, ggml_nbytes(base));
    memset(cur->data, 0, ggml_nbytes(cur));

    ggml_tensor * result = alpha_lora_mix_build_lora_dispatch(ctx, base, cur, &stacked, &route);
    TEST_ASSERT(result != nullptr, "single LoRA dispatch works");
    TEST_ASSERT(result->ne[0] == out_feat && result->ne[1] == n_tokens, "output shape correct");

    ggml_free(ctx);
    return true;
}

static bool test_router_dispatch_topk_equals_nloras() {
    // TC-BF-09: top_k == n_loras — all experts selected
    ggml_context * ctx = make_test_ctx();
    int in_feat = 4, rh = 2, n_loras = 3, top_k = 3, n_tokens = 1, rank = 2, out_feat = 4;

    auto router = make_graph_router(ctx, in_feat, rh, n_loras, top_k);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices->ne[0] == 3, "all 3 experts selected");

    alpha_lora_mix_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_feat, rank, n_loras);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, out_feat, n_loras);
    stacked.scale = 1.0f;
    stacked.n_loras = n_loras;
    stacked.rank = rank;
    memset(stacked.stacked_a->data, 0, ggml_nbytes(stacked.stacked_a));
    memset(stacked.stacked_b->data, 0, ggml_nbytes(stacked.stacked_b));

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_feat, n_tokens);
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(base->data, 0, ggml_nbytes(base));
    memset(cur->data, 0, ggml_nbytes(cur));

    ggml_tensor * result = alpha_lora_mix_build_lora_dispatch(ctx, base, cur, &stacked, &route);
    TEST_ASSERT(result != nullptr, "all-experts dispatch works");

    ggml_free(ctx);
    return true;
}

static bool test_router_dispatch_max_topk_max_loras() {
    // TC-BF-08: Maximum config — n_loras=64, top_k=4
    ggml_context * ctx = make_test_ctx(64 * 1024 * 1024);  // need more memory
    int in_feat = 4, rh = 2, n_loras = 64, top_k = 4, n_tokens = 1, rank = 2, out_feat = 4;

    auto router = make_graph_router(ctx, in_feat, rh, n_loras, top_k);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "max config router works");
    TEST_ASSERT(route.indices->ne[0] == 4, "top_k=4 with 64 LoRAs → ne[0]==4");

    alpha_lora_mix_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_feat, rank, n_loras);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, out_feat, n_loras);
    stacked.scale = 1.0f;
    stacked.n_loras = n_loras;
    stacked.rank = rank;
    memset(stacked.stacked_a->data, 0, ggml_nbytes(stacked.stacked_a));
    memset(stacked.stacked_b->data, 0, ggml_nbytes(stacked.stacked_b));

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_feat, n_tokens);
    ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);
    memset(base->data, 0, ggml_nbytes(base));
    memset(cur->data, 0, ggml_nbytes(cur));

    ggml_tensor * result = alpha_lora_mix_build_lora_dispatch(ctx, base, cur, &stacked, &route);
    TEST_ASSERT(result != nullptr, "max config dispatch works");

    ggml_free(ctx);
    return true;
}

// ============================================================================
// Group 3: CPU vs Graph router clamp parity
// ============================================================================

static bool test_cpu_clamps_topk_to_nloras() {
    // TC-BF-10: CPU router with top_k=10, n_loras=3 → n_selected=3
    const char * path = "test_bf_cpu_clamp.gguf";
    write_gguf(path, 8, 4, 3, 10);

    alpha_router * router = alpha_router_load(path);
    TEST_ASSERT(router != nullptr, "should load");
    // CPU router clamps top_k to min(top_k, MAX_TOP_K=4), then to min(4, n_loras=3) = 3
    TEST_ASSERT(router->top_k <= 4, "GGUF top_k clamped to MAX_TOP_K");

    std::vector<float> hidden(8, 1.0f);
    alpha_route_result result = alpha_router_forward(router, hidden.data(), 10);
    TEST_ASSERT(result.n_selected <= 3, "CPU n_selected clamped to n_loras=3");

    alpha_router_free(router);
    remove(path);
    return true;
}

static bool test_cpu_clamps_topk_to_max_topk() {
    // TC-BF-11: CPU router with top_k=100, n_loras=64 → n_selected=4 (MAX_TOP_K)
    const char * path = "test_bf_cpu_max.gguf";
    write_gguf(path, 4, 2, 10, 100);  // n_loras=10, top_k=100

    alpha_router * router = alpha_router_load(path);
    TEST_ASSERT(router != nullptr, "should load");
    TEST_ASSERT(router->top_k == 4, "GGUF top_k=100 clamped to MAX_TOP_K=4");

    std::vector<float> hidden(4, 1.0f);
    alpha_route_result result = alpha_router_forward(router, hidden.data(), 100);
    TEST_ASSERT(result.n_selected == 4, "CPU n_selected = MAX_TOP_K=4");

    alpha_router_free(router);
    remove(path);
    return true;
}

static bool test_cpu_graph_clamp_parity() {
    // TC-BF-12: Same config through CPU and graph paths → same effective K
    const char * path = "test_bf_parity.gguf";
    int H = 8, R = 4, N = 3, K = 5;  // top_k=5 > n_loras=3
    write_gguf(path, H, R, N, K);

    // CPU path
    alpha_router * cpu_router = alpha_router_load(path);
    TEST_ASSERT(cpu_router != nullptr, "CPU router loads");
    std::vector<float> hidden(H, 1.0f);
    alpha_route_result cpu_result = alpha_router_forward(cpu_router, hidden.data(), K);
    int cpu_k = cpu_result.n_selected;

    // Graph path
    ggml_context * ctx = make_test_ctx();
    auto graph_router = make_graph_router(ctx, H, R, N, K);
    ggml_tensor * hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    memset(hidden_t->data, 0, ggml_nbytes(hidden_t));
    for (int i = 0; i < H; i++) ((float *)hidden_t->data)[i] = 1.0f;

    alpha_lora_mix_route_tensors graph_result = alpha_lora_mix_build_router_graph(ctx, &graph_router, hidden_t);
    int graph_k = (int)graph_result.indices->ne[0];

    // Both should clamp to min(K, N) = min(5, 3) = 3
    // But CPU also clamps to MAX_TOP_K=4, so cpu_k = min(min(5,4), 3) = 3
    // Graph clamps to min(5, 3) = 3
    TEST_ASSERT(cpu_k == graph_k,
        "CPU and graph router should agree on effective K");

    alpha_router_free(cpu_router);
    ggml_free(ctx);
    remove(path);
    return true;
}

// ============================================================================
// Group 4: Stress / Boundary
// ============================================================================

static bool test_graph_router_max_top_k_value() {
    // TC-BF-13: n_loras=64, top_k=4 (MAX_TOP_K) — maximum valid config
    ggml_context * ctx = make_test_ctx(64 * 1024 * 1024);
    auto router = make_graph_router(ctx, 8, 4, 64, 4);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "max valid config works");
    TEST_ASSERT(route.indices->ne[0] == 4, "top_k=4 with 64 LoRAs → ne[0]==4");

    ggml_free(ctx);
    return true;
}

static bool test_graph_router_large_batch_clamped() {
    // TC-BF-14: Large batch (n_tokens=1024) with top_k clamped
    ggml_context * ctx = make_test_ctx(64 * 1024 * 1024);
    int n_tokens = 1024;
    auto router = make_graph_router(ctx, 4, 2, 2, 10);  // top_k=10 > n_loras=2

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, n_tokens);
    memset(hidden->data, 0, ggml_nbytes(hidden));

    alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
    TEST_ASSERT(route.indices != nullptr, "large batch builds");
    TEST_ASSERT(route.indices->ne[0] == 2, "top_k clamped to n_loras=2");
    TEST_ASSERT(route.indices->ne[1] == n_tokens, "all 1024 tokens routed");

    ggml_free(ctx);
    return true;
}

static bool test_graph_router_repeated_builds() {
    // TC-BF-15: Build router graph 100 times in same context — no memory corruption
    ggml_context * ctx = make_test_ctx(256 * 1024 * 1024);  // lots of memory
    auto router = make_graph_router(ctx, 4, 2, 3, 2);

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memset(hidden->data, 0, ggml_nbytes(hidden));
    ((float *)hidden->data)[0] = 1.0f;

    for (int i = 0; i < 100; i++) {
        alpha_lora_mix_route_tensors route = alpha_lora_mix_build_router_graph(ctx, &router, hidden);
        if (!route.indices) {
            fprintf(stderr, "  FAIL: %s: build %d returned null indices\n", __func__, i);
            tests_failed++;
            ggml_free(ctx);
            return false;
        }
    }

    ggml_free(ctx);
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    fprintf(stderr, "\n=== LoRA-mixture Bug Fix Verification Tests ===\n\n");

    fprintf(stderr, "--- Group 1: Top-K Clamping Boundaries ---\n");
    if (test_graph_topk_equals_nloras())         test_pass("test_graph_topk_equals_nloras");
    if (test_graph_topk_exceeds_nloras_by_1())   test_pass("test_graph_topk_exceeds_nloras_by_1");
    if (test_graph_topk_much_larger())           test_pass("test_graph_topk_much_larger");
    if (test_graph_topk_zero())                  test_pass("test_graph_topk_zero");
    if (test_graph_topk_negative())              test_pass("test_graph_topk_negative");

    fprintf(stderr, "\n--- Group 2: Router → Dispatch Integration ---\n");
    if (test_router_dispatch_clamped_topk())      test_pass("test_router_dispatch_clamped_topk");
    if (test_router_dispatch_topk1_single_lora()) test_pass("test_router_dispatch_topk1_single_lora");
    if (test_router_dispatch_topk_equals_nloras()) test_pass("test_router_dispatch_topk_equals_nloras");
    if (test_router_dispatch_max_topk_max_loras()) test_pass("test_router_dispatch_max_topk_max_loras");

    fprintf(stderr, "\n--- Group 3: CPU vs Graph Clamp Parity ---\n");
    if (test_cpu_clamps_topk_to_nloras())        test_pass("test_cpu_clamps_topk_to_nloras");
    if (test_cpu_clamps_topk_to_max_topk())      test_pass("test_cpu_clamps_topk_to_max_topk");
    if (test_cpu_graph_clamp_parity())           test_pass("test_cpu_graph_clamp_parity");

    fprintf(stderr, "\n--- Group 4: Stress / Boundary ---\n");
    if (test_graph_router_max_top_k_value())     test_pass("test_graph_router_max_top_k_value");
    if (test_graph_router_large_batch_clamped()) test_pass("test_graph_router_large_batch_clamped");
    if (test_graph_router_repeated_builds())     test_pass("test_graph_router_repeated_builds");

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n",
            tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
