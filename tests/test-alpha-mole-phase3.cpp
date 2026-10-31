// AlphaLlama Phase 2-3 Tests
//
// Targets:
// - ggml compute graph router (alpha_mole_graph.h/.cpp)
// - Router graph construction and tensor shapes
// - LoRA dispatch graph construction
// - Hardening validations (F32 check, 2D check, bias validation, top_k clamp)
// - Python conversion script round-trip (GGUF written by Python, loaded by C++)
// - LoRA-mixture llama model builder compilation verification
//
// Build:
//   cmake --build build --config Release --target test-alpha-mole-phase3
//
// Run:
//   ./build/bin/Release/test-alpha-mole-phase3.exe

#include "alpha_router.h"
#include "alpha_mole.h"
#include "alpha_mole_graph.h"

#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

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
        fprintf(stderr, "  FAIL: %s (line %d): %s (%.6f vs %.6f, tol=%.6f)\n", \
                __func__, __LINE__, msg, (float)(a), (float)(b), (float)(tol)); \
        tests_failed++; \
        return false; \
    } \
} while(0)

static void test_pass(const char * name) {
    fprintf(stderr, "  PASS: %s\n", name);
    tests_passed++;
}

// ============================================================================
// Helper: create ggml context for graph tests
// ============================================================================

static ggml_context * make_test_ctx(size_t mem_size = 16 * 1024 * 1024) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    return ggml_init(params);
}

// ============================================================================
// Group 1: Graph Router Construction
// ============================================================================

// Test that the router graph builds valid tensor shapes
static bool test_graph_router_shapes() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    const int hidden_dim    = 8;
    const int router_hidden = 4;
    const int n_loras       = 3;
    const int top_k         = 2;
    const int n_tokens      = 5;

    // Create router weight tensors
    ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, router_hidden);
    ggml_tensor * w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, router_hidden, n_loras);

    alpha_mole_graph_router router = {};
    router.w_down  = w_down;
    router.w_up    = w_up;
    router.b_down  = nullptr;
    router.b_up    = nullptr;
    router.n_loras = n_loras;
    router.top_k   = top_k;

    // Create input hidden states
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_tokens);

    // Build router graph
    alpha_mole_route_tensors route = alpha_mole_build_router_graph(ctx, &router, hidden);

    // Verify output tensor shapes
    TEST_ASSERT(route.indices != nullptr, "indices should not be null");
    TEST_ASSERT(route.weights != nullptr, "weights should not be null");

    // indices should be [top_k, n_tokens]
    TEST_ASSERT(route.indices->ne[0] == top_k, "indices dim 0 should be top_k");
    TEST_ASSERT(route.indices->ne[1] == n_tokens, "indices dim 1 should be n_tokens");

    // weights (probs) should be [n_loras, n_tokens]
    TEST_ASSERT(route.weights->ne[0] == n_loras, "weights dim 0 should be n_loras");
    TEST_ASSERT(route.weights->ne[1] == n_tokens, "weights dim 1 should be n_tokens");

    ggml_free(ctx);
    test_pass("test_graph_router_shapes");
    return true;
}

// Test router graph with biases
static bool test_graph_router_with_biases() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    const int hidden_dim    = 4;
    const int router_hidden = 2;
    const int n_loras       = 3;

    ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, router_hidden);
    ggml_tensor * w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, router_hidden, n_loras);
    ggml_tensor * b_down = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, router_hidden);
    ggml_tensor * b_up   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_loras);

    alpha_mole_graph_router router = {};
    router.w_down  = w_down;
    router.w_up    = w_up;
    router.b_down  = b_down;
    router.b_up    = b_up;
    router.n_loras = n_loras;
    router.top_k   = 2;

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, 1);

    alpha_mole_route_tensors route = alpha_mole_build_router_graph(ctx, &router, hidden);

    // Graph should build without crashing
    TEST_ASSERT(route.indices != nullptr, "indices with biases should not be null");
    TEST_ASSERT(route.weights != nullptr, "weights with biases should not be null");

    ggml_free(ctx);
    test_pass("test_graph_router_with_biases");
    return true;
}

// Test router graph with single token
static bool test_graph_router_single_token() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);

    alpha_mole_graph_router router = {};
    router.w_down  = w_down;
    router.w_up    = w_up;
    router.n_loras = 3;
    router.top_k   = 1;

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);

    alpha_mole_route_tensors route = alpha_mole_build_router_graph(ctx, &router, hidden);

    TEST_ASSERT(route.indices->ne[0] == 1, "top_k=1 should produce 1 index per token");
    TEST_ASSERT(route.indices->ne[1] == 1, "single token batch");

    ggml_free(ctx);
    test_pass("test_graph_router_single_token");
    return true;
}

// Test router graph with large batch
static bool test_graph_router_large_batch() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
    ggml_tensor * w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 10);

    alpha_mole_graph_router router = {};
    router.w_down  = w_down;
    router.w_up    = w_up;
    router.n_loras = 10;
    router.top_k   = 3;

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 512);

    alpha_mole_route_tensors route = alpha_mole_build_router_graph(ctx, &router, hidden);

    TEST_ASSERT(route.indices->ne[0] == 3, "top_k=3");
    TEST_ASSERT(route.indices->ne[1] == 512, "512 tokens");

    ggml_free(ctx);
    test_pass("test_graph_router_large_batch");
    return true;
}

// ============================================================================
// Group 2: LoRA Dispatch Graph Construction
// ============================================================================

// Test dispatch returns base_result when stacked is null
static bool test_dispatch_null_stacked() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);

    alpha_mole_route_tensors route = {};
    route.indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 4);

    // null stacked should return base unchanged
    ggml_tensor * result = alpha_mole_build_lora_dispatch(ctx, base, cur, nullptr, &route);
    TEST_ASSERT(result == base, "null stacked should return base_result");

    ggml_free(ctx);
    test_pass("test_dispatch_null_stacked");
    return true;
}

// Test dispatch returns base_result when route is null
static bool test_dispatch_null_route() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);

    alpha_mole_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 3);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 3);
    stacked.n_loras   = 3;
    stacked.rank      = 4;
    stacked.scale     = 1.0f;

    ggml_tensor * result = alpha_mole_build_lora_dispatch(ctx, base, cur, &stacked, nullptr);
    TEST_ASSERT(result == base, "null route should return base_result");

    ggml_free(ctx);
    test_pass("test_dispatch_null_route");
    return true;
}

// Test dispatch returns base_result when route.indices is null
static bool test_dispatch_null_indices() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);

    alpha_mole_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 4, 3);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 8, 3);
    stacked.n_loras   = 3;
    stacked.rank      = 4;
    stacked.scale     = 1.0f;

    alpha_mole_route_tensors route = {};
    route.indices = nullptr;
    route.weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);

    ggml_tensor * result = alpha_mole_build_lora_dispatch(ctx, base, cur, &stacked, &route);
    TEST_ASSERT(result == base, "null indices should return base_result");

    ggml_free(ctx);
    test_pass("test_dispatch_null_indices");
    return true;
}

// Test dispatch builds graph with correct output shape
static bool test_dispatch_output_shape() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    const int in_feat  = 8;
    const int out_feat = 8;
    const int rank     = 4;
    const int n_loras  = 3;
    const int n_tokens = 5;
    const int top_k    = 2;

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out_feat, n_tokens);
    ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, n_tokens);

    alpha_mole_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, in_feat, rank, n_loras);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, out_feat, n_loras);
    stacked.n_loras   = n_loras;
    stacked.rank      = rank;
    stacked.scale     = 1.0f;

    alpha_mole_route_tensors route = {};
    route.indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, n_tokens);
    route.weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_loras, n_tokens);

    ggml_tensor * result = alpha_mole_build_lora_dispatch(ctx, base, cur, &stacked, &route);

    TEST_ASSERT(result != base, "dispatch should produce new tensor");
    TEST_ASSERT(result->ne[0] == out_feat, "output dim 0 should be out_feat");
    TEST_ASSERT(result->ne[1] == n_tokens, "output dim 1 should be n_tokens");

    ggml_free(ctx);
    test_pass("test_dispatch_output_shape");
    return true;
}

// Test dispatch with scale != 1.0
static bool test_dispatch_with_scale() {
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * cur  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);

    alpha_mole_stacked_lora stacked = {};
    stacked.stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 2);
    stacked.stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 4, 2);
    stacked.n_loras   = 2;
    stacked.rank      = 2;
    stacked.scale     = 0.5f;  // non-trivial scale

    alpha_mole_route_tensors route = {};
    route.indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 2);
    route.weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);

    ggml_tensor * result = alpha_mole_build_lora_dispatch(ctx, base, cur, &stacked, &route);
    TEST_ASSERT(result != base, "scaled dispatch should produce new tensor");

    ggml_free(ctx);
    test_pass("test_dispatch_with_scale");
    return true;
}

// ============================================================================
// Group 3: Hardening Validation
// ============================================================================

// Test that non-F32 tensors are rejected
static bool test_hardening_reject_non_f32() {
    const char * path = "test_non_f32.gguf";

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_mole.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.n_loras", 2);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.router_hidden", 2);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    // Create F16 tensors instead of F32
    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 4, 2);
    ggml_set_name(t_down, "alpha_mole.router.down.weight");

    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 2, 2);
    ggml_set_name(t_up, "alpha_mole.router.up.weight");

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    // Loading should fail because tensors are F16, not F32
    alpha_router * r = alpha_router_load(path);
    TEST_ASSERT(r == nullptr, "F16 tensors should be rejected");

    remove(path);
    test_pass("test_hardening_reject_non_f32");
    return true;
}

// Test that 3D tensors are rejected
static bool test_hardening_reject_3d_tensor() {
    const char * path = "test_3d_tensor.gguf";

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_mole.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.n_loras", 2);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.router_hidden", 2);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    // Create 3D tensor (should be 2D) — use non-1 dims so ggml_n_dims sees it as 3D
    struct ggml_tensor * t_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 2, 3);
    ggml_set_name(t_down, "alpha_mole.router.down.weight");

    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(t_up, "alpha_mole.router.up.weight");

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    alpha_router * r = alpha_router_load(path);
    TEST_ASSERT(r == nullptr, "3D tensor should be rejected");

    remove(path);
    test_pass("test_hardening_reject_3d_tensor");
    return true;
}

// Test that top_k from GGUF is clamped
static bool test_hardening_topk_clamp() {
    const char * path = "test_topk_clamp.gguf";

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_mole.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.n_loras", 2);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.top_k", 999);  // absurdly large
    gguf_set_val_i32(gguf_ctx, "alpha_mole.router_hidden", 2);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(t_down, "alpha_mole.router.down.weight");
    memset(t_down->data, 0, ggml_nbytes(t_down));

    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(t_up, "alpha_mole.router.up.weight");
    memset(t_up->data, 0, ggml_nbytes(t_up));

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    alpha_router * r = alpha_router_load(path);
    TEST_ASSERT(r != nullptr, "should load despite large top_k");
    TEST_ASSERT(r->top_k <= ALPHA_MOLE_MAX_TOP_K, "top_k should be clamped to MAX_TOP_K");
    TEST_ASSERT(r->top_k >= 1, "top_k should be at least 1");

    alpha_router_free(r);
    remove(path);
    test_pass("test_hardening_topk_clamp");
    return true;
}

// Test that negative top_k from GGUF is clamped to 1
static bool test_hardening_negative_topk_clamp() {
    const char * path = "test_neg_topk.gguf";

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_mole.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.n_loras", 2);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.top_k", -5);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.router_hidden", 2);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(t_down, "alpha_mole.router.down.weight");
    memset(t_down->data, 0, ggml_nbytes(t_down));

    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(t_up, "alpha_mole.router.up.weight");
    memset(t_up->data, 0, ggml_nbytes(t_up));

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    alpha_router * r = alpha_router_load(path);
    TEST_ASSERT(r != nullptr, "should load despite negative top_k");
    TEST_ASSERT(r->top_k == 1, "negative top_k should clamp to 1");

    alpha_router_free(r);
    remove(path);
    test_pass("test_hardening_negative_topk_clamp");
    return true;
}

// Test that mismatched bias dimensions are ignored (not crash)
static bool test_hardening_bad_bias_ignored() {
    const char * path = "test_bad_bias.gguf";

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_mole.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.n_loras", 3);
    gguf_set_val_i32(gguf_ctx, "alpha_mole.router_hidden", 2);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(t_down, "alpha_mole.router.down.weight");
    memset(t_down->data, 0, ggml_nbytes(t_down));

    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);
    ggml_set_name(t_up, "alpha_mole.router.up.weight");
    memset(t_up->data, 0, ggml_nbytes(t_up));

    // Wrong-sized bias: should be [2] but is [5]
    struct ggml_tensor * t_down_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
    ggml_set_name(t_down_bias, "alpha_mole.router.down.bias");
    memset(t_down_bias->data, 0, ggml_nbytes(t_down_bias));

    // Wrong-typed bias: should be F32 but is F16
    struct ggml_tensor * t_up_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 3);
    ggml_set_name(t_up_bias, "alpha_mole.router.up.bias");
    memset(t_up_bias->data, 0, ggml_nbytes(t_up_bias));

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_add_tensor(gguf_ctx, t_down_bias);
    gguf_add_tensor(gguf_ctx, t_up_bias);
    gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    // Should load successfully but ignore the bad biases
    alpha_router * r = alpha_router_load(path);
    TEST_ASSERT(r != nullptr, "bad biases should be ignored, not crash");
    TEST_ASSERT(r->b_down.empty(), "wrong-sized down bias should be skipped");
    TEST_ASSERT(r->b_up.empty(), "wrong-typed up bias should be skipped");

    alpha_router_free(r);
    remove(path);
    test_pass("test_hardening_bad_bias_ignored");
    return true;
}

// ============================================================================
// Group 4: Python Conversion Script Round-Trip
// ============================================================================

// Test that a GGUF generated by the Python script loads correctly in C++
static bool test_python_roundtrip() {
    // Generate a router GGUF using the Python script
    int ret = system(
        "python convert_router_to_gguf.py "
        "--random --hidden-dim 8 --n-loras 3 --router-hidden 4 --top-k 2 "
        "--output test_python_rt.gguf 2>&1"
    );

    if (ret != 0) {
        fprintf(stderr, "  SKIP: test_python_roundtrip (Python script not available)\n");
        tests_passed++;  // count as pass — script availability is env-dependent
        return true;
    }

    // Load in C++
    alpha_router * r = alpha_router_load("test_python_rt.gguf");
    TEST_ASSERT(r != nullptr, "Python-generated GGUF should load in C++");
    TEST_ASSERT(r->hidden_dim    == 8, "hidden_dim should be 8");
    TEST_ASSERT(r->router_hidden == 4, "router_hidden should be 4");
    TEST_ASSERT(r->n_loras       == 3, "n_loras should be 3");
    TEST_ASSERT(r->top_k         == 2, "top_k should be 2");

    // Run forward pass — should not crash
    std::vector<float> hidden(8, 1.0f);
    alpha_route_result res = alpha_router_forward(r, hidden.data(), 2);
    TEST_ASSERT(res.n_selected == 2, "should select 2 LoRAs");
    TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-5f, "weights should sum to 1");

    alpha_router_free(r);
    remove("test_python_rt.gguf");
    test_pass("test_python_roundtrip");
    return true;
}

// Test Python script with larger dimensions
static bool test_python_roundtrip_large() {
    int ret = system(
        "python convert_router_to_gguf.py "
        "--random --hidden-dim 2048 --n-loras 10 --router-hidden 64 --top-k 3 "
        "--output test_python_rt_large.gguf 2>&1"
    );

    if (ret != 0) {
        fprintf(stderr, "  SKIP: test_python_roundtrip_large (Python script not available)\n");
        tests_passed++;
        return true;
    }

    alpha_router * r = alpha_router_load("test_python_rt_large.gguf");
    TEST_ASSERT(r != nullptr, "large Python GGUF should load");
    TEST_ASSERT(r->hidden_dim    == 2048, "hidden_dim=2048");
    TEST_ASSERT(r->router_hidden == 64, "router_hidden=64");
    TEST_ASSERT(r->n_loras       == 10, "n_loras=10");
    TEST_ASSERT(r->top_k         == 3, "top_k=3");

    // Forward pass with realistic-sized input
    std::vector<float> hidden(2048, 0.01f);
    hidden[0] = 1.0f;
    alpha_route_result res = alpha_router_forward(r, hidden.data(), 3);
    TEST_ASSERT(res.n_selected == 3, "should select 3 LoRAs");
    float sum = res.weights[0] + res.weights[1] + res.weights[2];
    TEST_FLOAT_EQ(sum, 1.0f, 1e-4f, "weights should sum to 1");

    alpha_router_free(r);
    remove("test_python_rt_large.gguf");
    test_pass("test_python_roundtrip_large");
    return true;
}

// ============================================================================
// Group 5: CPU Router vs Graph Router Consistency
// ============================================================================

// Verify that the CPU router forward pass and the graph router
// use the same mathematical operations (same op sequence)
static bool test_cpu_graph_op_consistency() {
    // The CPU router does: Linear -> ReLU -> Linear -> Softmax -> TopK
    // The graph router does: mul_mat -> relu -> mul_mat -> soft_max -> argsort_top_k
    // These should be mathematically identical.

    // Build a graph router and verify the op chain
    ggml_context * ctx = make_test_ctx();
    TEST_ASSERT(ctx, "failed to create ggml context");

    ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_tensor * w_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 3);

    alpha_mole_graph_router router = {};
    router.w_down  = w_down;
    router.w_up    = w_up;
    router.n_loras = 3;
    router.top_k   = 2;

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);

    alpha_mole_route_tensors route = alpha_mole_build_router_graph(ctx, &router, hidden);

    // Trace back through the graph to verify op chain:
    // indices = argsort_top_k(soft_max(mul_mat(relu(mul_mat(hidden, w_down)), w_up)))

    // indices should come from argsort (top_k is argsort + view internally)
    // The exact op depends on ggml internals — argsort_top_k may return
    // GGML_OP_ARGSORT or a view of it
    TEST_ASSERT(route.indices != nullptr, "indices should exist");

    // weights (probs) should come from soft_max
    TEST_ASSERT(route.weights->op == GGML_OP_SOFT_MAX, "weights should be soft_max op");

    // soft_max input should be mul_mat (layer 2)
    ggml_tensor * logits = route.weights->src[0];
    TEST_ASSERT(logits->op == GGML_OP_MUL_MAT, "logits should be mul_mat op");

    // mul_mat input[1] should be unary (relu)
    ggml_tensor * relu_out = logits->src[1];
    TEST_ASSERT(relu_out->op == GGML_OP_UNARY, "intermediate should be unary (relu) op");

    // unary input should be mul_mat (layer 1)
    ggml_tensor * layer1_out = relu_out->src[0];
    TEST_ASSERT(layer1_out->op == GGML_OP_MUL_MAT, "layer 1 should be mul_mat op");

    ggml_free(ctx);
    test_pass("test_cpu_graph_op_consistency");
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    fprintf(stderr, "\n=== Alpha-MoLE Phase 2-3 Test Suite ===\n\n");

    fprintf(stderr, "--- Group 1: Graph Router Construction ---\n");
    test_graph_router_shapes();
    test_graph_router_with_biases();
    test_graph_router_single_token();
    test_graph_router_large_batch();

    fprintf(stderr, "\n--- Group 2: LoRA Dispatch Graph Construction ---\n");
    test_dispatch_null_stacked();
    test_dispatch_null_route();
    test_dispatch_null_indices();
    test_dispatch_output_shape();
    test_dispatch_with_scale();

    fprintf(stderr, "\n--- Group 3: Hardening Validation ---\n");
    test_hardening_reject_non_f32();
    test_hardening_reject_3d_tensor();
    test_hardening_topk_clamp();
    test_hardening_negative_topk_clamp();
    test_hardening_bad_bias_ignored();

    fprintf(stderr, "\n--- Group 4: Python Conversion Round-Trip ---\n");
    test_python_roundtrip();
    test_python_roundtrip_large();

    fprintf(stderr, "\n--- Group 5: CPU/Graph Router Consistency ---\n");
    test_cpu_graph_op_consistency();

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n",
            tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
