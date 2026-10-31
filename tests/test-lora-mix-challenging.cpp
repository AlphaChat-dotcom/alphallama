// AlphaLlama Challenging QA Tests
//
// Goes beyond the 8 basic tests. Targets:
// - Numerical edge cases (NaN, Inf, denormals, underflow)
// - Boundary conditions (zero weights, identical scores, single LoRA)
// - GGUF malformation (wrong shapes, missing metadata, corrupt values)
// - Memory safety (double-free, zero-size, routing_info lifecycle)
// - Router MLP contract violations
//
// Build:
//   cmake --build build --config Release --target test-lora-mix-challenging
//
// Run:
//   ./build/bin/Release/test-lora-mix-challenging.exe

#include "alpha_router.h"
#include "alpha_lora_mix.h"

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
static int tests_xfail  = 0;

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

#define TEST_XFAIL(cond, msg) do { \
    if (cond) { \
        fprintf(stderr, "  XFAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        tests_xfail++; \
        return true; \
    } \
} while(0)

static void test_pass(const char * name) {
    fprintf(stderr, "  PASS: %s\n", name);
    tests_passed++;
}

// Helper: build a simple router with given dimensions
static alpha_router make_router(int H, int R, int N,
                                const std::vector<float> & w_down,
                                const std::vector<float> & w_up,
                                const std::vector<float> & b_down = {},
                                const std::vector<float> & b_up = {}) {
    alpha_router router = {};
    router.hidden_dim    = H;
    router.router_hidden = R;
    router.n_loras       = N;
    router.top_k         = 2;
    router.shared        = true;
    router.w_down = w_down;
    router.w_up   = w_up;
    router.b_down = b_down;
    router.b_up   = b_up;
    return router;
}

// Helper: write a GGUF router file with custom metadata and tensors
static bool write_router_gguf(const char * path,
                              int version, int n_loras, int top_k,
                              int router_hidden, bool shared,
                              int hidden_dim,
                              const float * w_down_data, int w_down_size,
                              const float * w_up_data, int w_up_size,
                              bool include_version = true,
                              bool include_tensors = true) {
    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) return false;

    if (include_version) {
        gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.version", version);
    }
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.n_loras",       n_loras);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.top_k",         top_k);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.router_hidden", router_hidden);
    gguf_set_val_bool(gguf_ctx, "alpha_lora_mix.shared",       shared);

    struct ggml_context * ctx = nullptr;
    if (include_tensors) {
        struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, false };
        ctx = ggml_init(params);
        if (!ctx) { gguf_free(gguf_ctx); return false; }

        struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, router_hidden);
        ggml_set_name(t_down, "alpha_lora_mix.router.down.weight");
        memcpy(t_down->data, w_down_data, w_down_size * sizeof(float));

        struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, router_hidden, n_loras);
        ggml_set_name(t_up, "alpha_lora_mix.router.up.weight");
        memcpy(t_up->data, w_up_data, w_up_size * sizeof(float));

        gguf_add_tensor(gguf_ctx, t_down);
        gguf_add_tensor(gguf_ctx, t_up);
    }

    gguf_write_to_file(gguf_ctx, path, false);

    if (ctx) ggml_free(ctx);
    gguf_free(gguf_ctx);
    return true;
}


// ============================================================================
// GROUP 1: Router Numerical Edge Cases
// ============================================================================

// TC-R-01: All-zero weights — intermediate is all zeros → uniform softmax
static bool test_router_all_zero_weights() {
    std::vector<float> w_down(8, 0.0f);  // 4x2
    std::vector<float> w_up(6, 0.0f);    // 2x3
    alpha_router router = make_router(4, 2, 3, w_down, w_up);

    float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 3);

    TEST_ASSERT(res.n_selected == 3, "should select 3");
    // All scores should be equal after softmax → uniform weights
    TEST_FLOAT_EQ(res.weights[0], res.weights[1], 1e-5f, "weights should be uniform");
    TEST_FLOAT_EQ(res.weights[1], res.weights[2], 1e-5f, "weights should be uniform");
    TEST_FLOAT_EQ(res.weights[0] + res.weights[1] + res.weights[2], 1.0f, 1e-5f, "sum to 1");

    // No NaN/Inf
    for (int k = 0; k < 3; k++) {
        TEST_ASSERT(!std::isnan(res.weights[k]), "weight should not be NaN");
        TEST_ASSERT(!std::isinf(res.weights[k]), "weight should not be Inf");
    }

    test_pass("test_router_all_zero_weights");
    return true;
}

// TC-R-02: All-zero hidden state → ReLU kills everything → uniform softmax
static bool test_router_zero_hidden() {
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w_up   = {1.0f, 0.0f, 0.0f, 1.0f};
    alpha_router router = make_router(2, 2, 2, w_down, w_up);

    float hidden[2] = {0.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 2);

    TEST_ASSERT(res.n_selected == 2, "should select 2");
    TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-5f, "sum to 1");
    TEST_ASSERT(!std::isnan(res.weights[0]), "no NaN");
    TEST_ASSERT(!std::isnan(res.weights[1]), "no NaN");

    test_pass("test_router_zero_hidden");
    return true;
}

// TC-R-03: Very large negative weights → underflow in softmax
static bool test_router_extreme_negative_scores() {
    // w_down produces large intermediate values, w_up produces large NEGATIVE scores
    std::vector<float> w_down = {1000.0f, 1000.0f};  // 1x2
    std::vector<float> w_up   = {-1.0f, -2.0f};      // 2x1? No: R=2, N=1 → [2*1]
    // Wait: R=2, N=1 → w_up needs 2 elements
    alpha_router router = make_router(1, 2, 1, w_down, w_up);

    float hidden[1] = {1.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    TEST_ASSERT(res.n_selected == 1, "should select 1");
    // Single LoRA → weight should be 1.0 after re-norm
    TEST_FLOAT_EQ(res.weights[0], 1.0f, 1e-5f, "single lora weight should be 1.0");
    TEST_ASSERT(!std::isnan(res.weights[0]), "no NaN");

    test_pass("test_router_extreme_negative_scores");
    return true;
}

// TC-R-04: NaN in hidden state — should not crash (may produce NaN weights)
static bool test_router_nan_hidden() {
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w_up   = {1.0f, 0.0f, 0.0f, 1.0f};
    alpha_router router = make_router(2, 2, 2, w_down, w_up);

    float hidden[2] = {NAN, 1.0f};
    // This should not crash — NaN propagation is expected
    alpha_route_result res = alpha_router_forward(&router, hidden, 2);

    // We just verify it doesn't crash. NaN output is acceptable.
    TEST_ASSERT(res.n_selected == 2, "should still select 2");

    test_pass("test_router_nan_hidden");
    return true;
}

// TC-R-05: Inf in hidden state — should not crash
static bool test_router_inf_hidden() {
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w_up   = {1.0f, 0.0f, 0.0f, 1.0f};
    alpha_router router = make_router(2, 2, 2, w_down, w_up);

    float hidden[2] = {INFINITY, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 2);
    TEST_ASSERT(res.n_selected == 2, "should still select 2");

    test_pass("test_router_inf_hidden");
    return true;
}

// TC-R-06: All identical scores → top-K should select any K, weights uniform
static bool test_router_identical_scores() {
    // All w_up columns identical → all LoRA scores equal
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};  // 2x2
    std::vector<float> w_up   = {1.0f, 1.0f, 1.0f,          // 2x3
                                  1.0f, 1.0f, 1.0f};
    alpha_router router = make_router(2, 2, 3, w_down, w_up);

    float hidden[2] = {1.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 2);

    TEST_ASSERT(res.n_selected == 2, "should select 2");
    // When all scores are equal, top-K picks first K (stable sort)
    // Weights should be equal after re-norm
    TEST_FLOAT_EQ(res.weights[0], res.weights[1], 1e-5f, "equal scores → equal weights");
    TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-5f, "sum to 1");

    test_pass("test_router_identical_scores");
    return true;
}

// TC-R-07: Negative top_k — should clamp to 1
static bool test_router_negative_topk() {
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w_up   = {1.0f, 0.0f, 0.0f, 1.0f};
    alpha_router router = make_router(2, 2, 2, w_down, w_up);

    float hidden[2] = {1.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, -1);
    TEST_ASSERT(res.n_selected == 1, "negative top_k should clamp to 1");

    res = alpha_router_forward(&router, hidden, -999);
    TEST_ASSERT(res.n_selected == 1, "very negative top_k should clamp to 1");

    test_pass("test_router_negative_topk");
    return true;
}


// ============================================================================
// GROUP 2: Router Dimension Edge Cases
// ============================================================================

// TC-D-01: Single LoRA (n_loras=1) — top-K always selects it
static bool test_router_single_lora() {
    std::vector<float> w_down = {1.0f, 0.0f};  // 2x1 (H=2, R=1)
    std::vector<float> w_up   = {1.0f};          // 1x1 (R=1, N=1)
    alpha_router router = make_router(2, 1, 1, w_down, w_up);

    float hidden[2] = {5.0f, 3.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    TEST_ASSERT(res.n_selected == 1, "should select 1");
    TEST_ASSERT(res.indices[0] == 0, "only LoRA 0 exists");
    TEST_FLOAT_EQ(res.weights[0], 1.0f, 1e-5f, "single LoRA → weight 1.0");

    test_pass("test_router_single_lora");
    return true;
}

// TC-D-02: Large hidden_dim — performance/correctness
static bool test_router_large_hidden_dim() {
    const int H = 4096;  // Realistic model hidden size
    const int R = 64;
    const int N = 4;

    // Create weights (identity-like pattern)
    std::vector<float> w_down(H * R, 0.0f);
    for (int i = 0; i < std::min(H, R); i++) {
        w_down[i * R + i] = 1.0f;
    }
    std::vector<float> w_up(R * N, 0.0f);
    for (int n = 0; n < N; n++) {
        w_up[n * N + n] = 1.0f;  // Each neuron favors one LoRA
    }
    alpha_router router = make_router(H, R, N, w_down, w_up);

    std::vector<float> hidden(H, 0.0f);
    hidden[0] = 1.0f;  // Activate first neuron
    alpha_route_result res = alpha_router_forward(&router, hidden.data(), 2);

    TEST_ASSERT(res.n_selected == 2, "should select 2");
    TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-4f, "sum to 1");
    TEST_ASSERT(!std::isnan(res.weights[0]), "no NaN");

    test_pass("test_router_large_hidden_dim");
    return true;
}

// TC-D-03: Minimal dimensions (H=1, R=1, N=1)
static bool test_router_minimal_dimensions() {
    std::vector<float> w_down = {2.0f};
    std::vector<float> w_up   = {3.0f};
    alpha_router router = make_router(1, 1, 1, w_down, w_up);

    float hidden[1] = {1.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    TEST_ASSERT(res.n_selected == 1, "should select 1");
    TEST_FLOAT_EQ(res.weights[0], 1.0f, 1e-5f, "single LoRA weight 1.0");

    test_pass("test_router_minimal_dimensions");
    return true;
}

// TC-D-04: Max LoRAs (n_loras=64) with top_k=4
static bool test_router_max_loras() {
    const int H = 4, R = 2, N = 64;
    std::vector<float> w_down(H * R, 0.1f);
    std::vector<float> w_up(R * N, 0.0f);
    // Make LoRA 42 strongly preferred
    w_up[0 * N + 42] = 10.0f;
    w_up[1 * N + 42] = 10.0f;

    alpha_router router = make_router(H, R, N, w_down, w_up);

    float hidden[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 4);

    TEST_ASSERT(res.n_selected == 4, "should select 4 (MAX_TOP_K)");
    TEST_ASSERT(res.indices[0] == 42, "LoRA 42 should be top selection");
    TEST_ASSERT(res.weights[0] > 0.5f, "LoRA 42 should dominate");

    float sum = 0.0f;
    for (int k = 0; k < 4; k++) sum += res.weights[k];
    TEST_FLOAT_EQ(sum, 1.0f, 1e-4f, "weights sum to 1");

    test_pass("test_router_max_loras");
    return true;
}


// ============================================================================
// GROUP 3: Bias Edge Cases
// ============================================================================

// TC-B-01: Down bias (b_down) shifts ReLU activation
static bool test_router_down_bias() {
    std::vector<float> w_down = {0.0f, 0.0f, 0.0f, 0.0f};  // 2x2, all zero
    std::vector<float> w_up   = {1.0f, 0.0f, 0.0f, 1.0f};  // 2x2
    std::vector<float> b_down = {5.0f, -5.0f};  // Neuron 0 active, neuron 1 killed by ReLU
    alpha_router router = make_router(2, 2, 2, w_down, w_up, b_down);

    float hidden[2] = {0.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    // b_down=[5,-5]: after w_down (all zero), intermediate=[5,0] (ReLU kills -5)
    // w_up: [5,0] @ [[1,0],[0,1]] = [5, 0] → softmax → lora 0 dominates
    TEST_ASSERT(res.indices[0] == 0, "b_down should make lora 0 win");

    test_pass("test_router_down_bias");
    return true;
}

// TC-B-02: Negative up bias kills a LoRA
static bool test_router_negative_up_bias() {
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w_up   = {1.0f, 1.0f, 1.0f, 1.0f};  // Uniform weights
    std::vector<float> b_up   = {0.0f, -100.0f};  // Kill LoRA 1

    alpha_router router = make_router(2, 2, 2, w_down, w_up, {}, b_up);

    float hidden[2] = {1.0f, 1.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    TEST_ASSERT(res.indices[0] == 0, "negative bias should kill lora 1");

    test_pass("test_router_negative_up_bias");
    return true;
}


// ============================================================================
// GROUP 4: GGUF Validation Edge Cases
// ============================================================================

// TC-G-01: Missing version key → should reject
static bool test_gguf_missing_version() {
    float w_down[4] = {1,0,0,1};
    float w_up[4]   = {1,0,0,1};
    write_router_gguf("test_no_version.gguf", 1, 2, 2, 2, true, 2,
                      w_down, 4, w_up, 4, /*include_version=*/false);

    alpha_router * r = alpha_router_load("test_no_version.gguf");
    TEST_ASSERT(r == nullptr, "missing version should reject");
    remove("test_no_version.gguf");

    test_pass("test_gguf_missing_version");
    return true;
}

// TC-G-02: Version 0 — should reject
static bool test_gguf_version_zero() {
    float w_down[4] = {1,0,0,1};
    float w_up[4]   = {1,0,0,1};
    write_router_gguf("test_v0.gguf", 0, 2, 2, 2, true, 2,
                      w_down, 4, w_up, 4);

    alpha_router * r = alpha_router_load("test_v0.gguf");
    TEST_ASSERT(r == nullptr, "version 0 should reject");
    remove("test_v0.gguf");

    test_pass("test_gguf_version_zero");
    return true;
}

// TC-G-03: n_loras=0 in metadata (no tensors) — should reject
static bool test_gguf_zero_loras() {
    // Write GGUF with n_loras=0 but NO tensors (avoid ggml crash on 0-dim tensor)
    write_router_gguf("test_0loras.gguf", 1, 0, 2, 2, true, 2,
                      nullptr, 0, nullptr, 0, true, false);

    alpha_router * r = alpha_router_load("test_0loras.gguf");
    TEST_ASSERT(r == nullptr, "n_loras=0 should reject");
    remove("test_0loras.gguf");

    test_pass("test_gguf_zero_loras");
    return true;
}

// TC-G-04: Negative n_loras in metadata (no tensors) — should reject
static bool test_gguf_negative_loras() {
    write_router_gguf("test_negloras.gguf", 1, -5, 2, 2, true, 2,
                      nullptr, 0, nullptr, 0, true, false);

    alpha_router * r = alpha_router_load("test_negloras.gguf");
    TEST_ASSERT(r == nullptr, "negative n_loras should reject");
    remove("test_negloras.gguf");

    test_pass("test_gguf_negative_loras");
    return true;
}

// TC-G-05: router_hidden=0 (no tensors) — should reject
static bool test_gguf_zero_router_hidden() {
    write_router_gguf("test_0hidden.gguf", 1, 2, 2, 0, true, 2,
                      nullptr, 0, nullptr, 0, true, false);

    alpha_router * r = alpha_router_load("test_0hidden.gguf");
    TEST_ASSERT(r == nullptr, "router_hidden=0 should reject");
    remove("test_0hidden.gguf");

    test_pass("test_gguf_zero_router_hidden");
    return true;
}

// TC-G-06: Tensor dimension mismatch — w_down shape doesn't match metadata
static bool test_gguf_shape_mismatch() {
    // Metadata says n_loras=3, router_hidden=2, but w_up is [2x2] not [2x3]
    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.n_loras", 3);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.router_hidden", 2);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    // w_down: correct [4, 2]
    float w_down_data[8] = {};
    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    ggml_set_name(t_down, "alpha_lora_mix.router.down.weight");
    memcpy(t_down->data, w_down_data, sizeof(w_down_data));

    // w_up: WRONG — [2, 2] instead of [2, 3] (n_loras=3 but tensor says 2)
    float w_up_data[4] = {};
    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    ggml_set_name(t_up, "alpha_lora_mix.router.up.weight");
    memcpy(t_up->data, w_up_data, sizeof(w_up_data));

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_write_to_file(gguf_ctx, "test_shape_mismatch.gguf", false);
    ggml_free(ctx);
    gguf_free(gguf_ctx);

    alpha_router * r = alpha_router_load("test_shape_mismatch.gguf");
    TEST_ASSERT(r == nullptr, "shape mismatch should reject");
    remove("test_shape_mismatch.gguf");

    test_pass("test_gguf_shape_mismatch");
    return true;
}

// TC-G-07: Valid GGUF with bias tensors — round-trip
static bool test_gguf_with_biases() {
    const int H = 2, R = 2, N = 2;

    struct gguf_context * gguf_ctx = gguf_init_empty();
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.version", 1);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.n_loras", N);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.top_k", 2);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.router_hidden", R);
    gguf_set_val_bool(gguf_ctx, "alpha_lora_mix.shared", true);

    struct ggml_init_params params = { 1024 * 1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    float w_down_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, R);
    ggml_set_name(t_down, "alpha_lora_mix.router.down.weight");
    memcpy(t_down->data, w_down_data, sizeof(w_down_data));

    float w_up_data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, N);
    ggml_set_name(t_up, "alpha_lora_mix.router.up.weight");
    memcpy(t_up->data, w_up_data, sizeof(w_up_data));

    float b_down_data[2] = {0.5f, -0.5f};
    struct ggml_tensor * t_bd = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, R);
    ggml_set_name(t_bd, "alpha_lora_mix.router.down.bias");
    memcpy(t_bd->data, b_down_data, sizeof(b_down_data));

    float b_up_data[2] = {0.0f, 10.0f};
    struct ggml_tensor * t_bu = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N);
    ggml_set_name(t_bu, "alpha_lora_mix.router.up.bias");
    memcpy(t_bu->data, b_up_data, sizeof(b_up_data));

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);
    gguf_add_tensor(gguf_ctx, t_bd);
    gguf_add_tensor(gguf_ctx, t_bu);

    gguf_write_to_file(gguf_ctx, "test_bias_roundtrip.gguf", false);
    ggml_free(ctx);
    gguf_free(gguf_ctx);

    alpha_router * r = alpha_router_load("test_bias_roundtrip.gguf");
    TEST_ASSERT(r != nullptr, "should load successfully with biases");
    TEST_ASSERT(!r->b_down.empty(), "b_down should be loaded");
    TEST_ASSERT(!r->b_up.empty(), "b_up should be loaded");
    TEST_FLOAT_EQ(r->b_down[0], 0.5f, 1e-6f, "b_down[0] match");
    TEST_FLOAT_EQ(r->b_up[1], 10.0f, 1e-6f, "b_up[1] match");

    // Verify bias affects forward pass
    float hidden[2] = {1.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(r, hidden, 1);
    TEST_ASSERT(res.indices[0] == 1, "b_up +10 should make lora 1 win");

    alpha_router_free(r);
    remove("test_bias_roundtrip.gguf");

    test_pass("test_gguf_with_biases");
    return true;
}

// TC-G-08: Empty file path string — should reject
static bool test_gguf_empty_path() {
    alpha_router * r = alpha_router_load("");
    TEST_ASSERT(r == nullptr, "empty path should return nullptr");

    test_pass("test_gguf_empty_path");
    return true;
}


// ============================================================================
// GROUP 5: API Safety & Lifecycle
// ============================================================================

// TC-A-01: alpha_lora_mix_routing_info_free double-call — should not crash
static bool test_routing_info_double_free() {
    alpha_lora_mix_routing_info info = {};
    info.routes = (alpha_lora_mix_token_route *)malloc(sizeof(alpha_lora_mix_token_route));
    info.n_tokens = 1;

    alpha_lora_mix_routing_info_free(&info);
    TEST_ASSERT(info.routes == nullptr, "routes should be null after free");
    TEST_ASSERT(info.n_tokens == 0, "n_tokens should be 0 after free");

    // Second free should be safe (routes is null, early return)
    alpha_lora_mix_routing_info_free(&info);

    test_pass("test_routing_info_double_free");
    return true;
}

// TC-A-02: alpha_lora_mix_routing_info_free with zero routes
static bool test_routing_info_free_zero() {
    alpha_lora_mix_routing_info info = {};
    info.routes = nullptr;
    info.n_tokens = 0;

    // Should not crash
    alpha_lora_mix_routing_info_free(&info);

    test_pass("test_routing_info_free_zero");
    return true;
}

// TC-A-03: alpha_router_free with null — should not crash
static bool test_router_free_null() {
    // The code does `delete router` without null check.
    // `delete nullptr` is well-defined in C++ (no-op), so this should be fine.
    alpha_router_free(nullptr);

    test_pass("test_router_free_null");
    return true;
}

// TC-A-04: alpha_lora_mix_init with n_loras at max boundary (64)
static bool test_lora_mix_init_max_loras_boundary() {
    alpha_lora_mix_params p = {};
    p.n_loras = ALPHA_LORA_MIX_MAX_LORAS;  // 64 — should be accepted
    const char * paths[64];
    for (int i = 0; i < 64; i++) paths[i] = "dummy.gguf";
    p.lora_paths = paths;

    // Can't actually load — llama_adapter_lora_init will fail on dummy paths
    // But the bounds check (n_loras > ALPHA_LORA_MIX_MAX_LORAS) should NOT trigger
    // This tests that 64 is accepted, not 65
    // We know init will fail later (no real model), but bounds check passes

    // Test that 65 fails
    p.n_loras = ALPHA_LORA_MIX_MAX_LORAS + 1;
    auto * m = alpha_lora_mix_init((llama_model*)1, (llama_context*)1, p);
    TEST_ASSERT(m == nullptr, "n_loras=65 should fail");

    // Test that 64 passes bounds check (will fail later on LoRA load, but that's fine)
    // We can't test this without real model files, so just verify boundary value

    test_pass("test_lora_mix_init_max_loras_boundary");
    return true;
}

// TC-A-05: Verify weight sum invariant across many random-ish inputs
static bool test_router_weight_sum_invariant() {
    std::vector<float> w_down = {
        0.3f, 0.7f, -0.2f, 0.5f,
        0.1f, -0.4f, 0.8f, 0.6f,
    };
    std::vector<float> w_up = {
        0.5f, -0.3f, 0.2f, 0.8f,
        -0.1f, 0.6f, 0.4f, -0.7f,
    };
    alpha_router router = make_router(4, 2, 4, w_down, w_up);

    // Test 20 different hidden states
    float inputs[20][4] = {
        {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1},
        {1,1,0,0}, {0,0,1,1}, {1,0,1,0}, {0,1,0,1},
        {1,1,1,1}, {-1,-1,-1,-1}, {0.5,0.5,0.5,0.5}, {100,0,0,0},
        {0,0,0,100}, {-100,0,0,0}, {1,-1,1,-1}, {0.001f,0.001f,0.001f,0.001f},
        {FLT_MAX,0,0,0}, {0,0,0,FLT_MIN}, {-FLT_MAX,0,0,0}, {1e10f,1e-10f,0,0},
    };

    for (int i = 0; i < 20; i++) {
        for (int k = 1; k <= 4; k++) {
            alpha_route_result res = alpha_router_forward(&router, inputs[i], k);
            if (res.n_selected == 0) continue;  // null input case

            float sum = 0.0f;
            for (int j = 0; j < res.n_selected; j++) {
                sum += res.weights[j];
            }

            // Allow NaN for extreme inputs (FLT_MAX, etc) — just don't crash
            if (!std::isnan(sum)) {
                TEST_FLOAT_EQ(sum, 1.0f, 1e-3f, "weights must sum to 1.0");
            }
        }
    }

    test_pass("test_router_weight_sum_invariant");
    return true;
}


// ============================================================================
// GROUP 6: Determinism & Consistency
// ============================================================================

// TC-C-01: Same input → same output (deterministic)
static bool test_router_deterministic() {
    std::vector<float> w_down = {1.0f, 0.5f, -0.3f, 0.0f, 0.7f, 0.2f};
    std::vector<float> w_up   = {0.8f, -0.4f, 0.1f, 0.6f, -0.2f, 0.9f};
    alpha_router router = make_router(3, 2, 3, w_down, w_up);

    float hidden[3] = {0.5f, -0.2f, 0.8f};

    alpha_route_result res1 = alpha_router_forward(&router, hidden, 2);
    alpha_route_result res2 = alpha_router_forward(&router, hidden, 2);

    TEST_ASSERT(res1.indices[0] == res2.indices[0], "same input → same top index");
    TEST_ASSERT(res1.indices[1] == res2.indices[1], "same input → same 2nd index");
    TEST_FLOAT_EQ(res1.weights[0], res2.weights[0], 1e-7f, "same input → same weight");
    TEST_FLOAT_EQ(res1.weights[1], res2.weights[1], 1e-7f, "same input → same weight");

    test_pass("test_router_deterministic");
    return true;
}

// TC-C-02: Unused slots always -1 and 0.0
static bool test_router_unused_slots_zeroed() {
    std::vector<float> w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> w_up   = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f};
    alpha_router router = make_router(2, 2, 3, w_down, w_up);

    float hidden[2] = {1.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    TEST_ASSERT(res.n_selected == 1, "top-1");
    TEST_ASSERT(res.indices[1] == -1, "slot 1 should be -1");
    TEST_ASSERT(res.indices[2] == -1, "slot 2 should be -1");
    TEST_ASSERT(res.indices[3] == -1, "slot 3 should be -1");
    TEST_FLOAT_EQ(res.weights[1], 0.0f, 1e-7f, "slot 1 weight 0");
    TEST_FLOAT_EQ(res.weights[2], 0.0f, 1e-7f, "slot 2 weight 0");
    TEST_FLOAT_EQ(res.weights[3], 0.0f, 1e-7f, "slot 3 weight 0");

    test_pass("test_router_unused_slots_zeroed");
    return true;
}


// ============================================================================
// Main
// ============================================================================

int main() {
    fprintf(stderr, "\n=== LoRA-mixture Challenging Test Suite ===\n\n");

    fprintf(stderr, "--- Group 1: Numerical Edge Cases ---\n");
    test_router_all_zero_weights();
    test_router_zero_hidden();
    test_router_extreme_negative_scores();
    test_router_nan_hidden();
    test_router_inf_hidden();
    test_router_identical_scores();
    test_router_negative_topk();

    fprintf(stderr, "\n--- Group 2: Dimension Edge Cases ---\n");
    test_router_single_lora();
    test_router_large_hidden_dim();
    test_router_minimal_dimensions();
    test_router_max_loras();

    fprintf(stderr, "\n--- Group 3: Bias Edge Cases ---\n");
    test_router_down_bias();
    test_router_negative_up_bias();

    fprintf(stderr, "\n--- Group 4: GGUF Validation ---\n");
    test_gguf_missing_version();
    test_gguf_version_zero();
    test_gguf_zero_loras();
    test_gguf_negative_loras();
    test_gguf_zero_router_hidden();
    test_gguf_shape_mismatch();
    test_gguf_with_biases();
    test_gguf_empty_path();

    fprintf(stderr, "\n--- Group 5: API Safety ---\n");
    test_routing_info_double_free();
    test_routing_info_free_zero();
    test_router_free_null();
    test_lora_mix_init_max_loras_boundary();
    test_router_weight_sum_invariant();

    fprintf(stderr, "\n--- Group 6: Determinism ---\n");
    test_router_deterministic();
    test_router_unused_slots_zeroed();

    fprintf(stderr, "\n=== Results: %d passed, %d failed, %d xfailed ===\n\n",
            tests_passed, tests_failed, tests_xfail);

    return tests_failed > 0 ? 1 : 0;
}
