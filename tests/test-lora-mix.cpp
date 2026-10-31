#include "alpha_router.h"
#include "alpha_lora_mix.h"

#include "ggml.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
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
// Helper: create a router.gguf test fixture in memory
// ============================================================================

// Create a minimal router GGUF file for testing
// Router config: hidden_dim=4, router_hidden=2, n_loras=3, top_k=2
static bool create_test_router_gguf(const char * path) {
    const int hidden_dim    = 4;
    const int router_hidden = 2;
    const int n_loras       = 3;

    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) return false;

    // Set metadata
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.version",       1);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.n_loras",       n_loras);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.top_k",         2);
    gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.router_hidden", router_hidden);
    gguf_set_val_bool(gguf_ctx, "alpha_lora_mix.shared",       true);

    // Create ggml context for tensors
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        gguf_free(gguf_ctx);
        return false;
    }

    // w_down: [hidden_dim, router_hidden] = [4, 2]
    // Weights chosen so that:
    //   hidden=[1,0,0,0] -> intermediate=[1,0] -> scores biased to lora 0
    //   hidden=[0,0,0,1] -> intermediate=[0,1] -> scores biased to lora 2
    float w_down_data[4 * 2] = {
        1.0f,  0.0f,   // h=0: activates neuron 0
        0.5f,  0.5f,   // h=1: activates both
        0.0f,  0.0f,   // h=2: dead
        0.0f,  1.0f,   // h=3: activates neuron 1
    };

    // w_up: [router_hidden, n_loras] = [2, 3]
    // neuron 0 -> favors lora 0
    // neuron 1 -> favors lora 2
    float w_up_data[2 * 3] = {
        2.0f, 0.0f, 0.0f,   // neuron 0 -> lora 0
        0.0f, 0.0f, 2.0f,   // neuron 1 -> lora 2
    };

    struct ggml_tensor * t_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, router_hidden);
    ggml_set_name(t_down, "alpha_lora_mix.router.down.weight");
    memcpy(t_down->data, w_down_data, sizeof(w_down_data));

    struct ggml_tensor * t_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, router_hidden, n_loras);
    ggml_set_name(t_up, "alpha_lora_mix.router.up.weight");
    memcpy(t_up->data, w_up_data, sizeof(w_up_data));

    gguf_add_tensor(gguf_ctx, t_down);
    gguf_add_tensor(gguf_ctx, t_up);

    gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);
    return true;
}

// ============================================================================
// Test 1: Router MLP forward pass — basic correctness
// ============================================================================

static bool test_router_forward_basic() {
    // Create router manually (no GGUF)
    alpha_router router = {};
    router.hidden_dim    = 4;
    router.router_hidden = 2;
    router.n_loras       = 3;
    router.top_k         = 2;
    router.shared        = true;

    // Same weights as the GGUF fixture
    router.w_down = {
        1.0f, 0.0f,
        0.5f, 0.5f,
        0.0f, 0.0f,
        0.0f, 1.0f,
    };
    router.w_up = {
        2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 2.0f,
    };

    // Test 1a: hidden=[1,0,0,0] should favor lora 0
    {
        float hidden[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        alpha_route_result res = alpha_router_forward(&router, hidden, 2);

        TEST_ASSERT(res.n_selected == 2, "should select 2 LoRAs");
        TEST_ASSERT(res.indices[0] == 0, "top LoRA should be index 0");
        TEST_ASSERT(res.weights[0] > res.weights[1], "top weight should be highest");
        TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-5f, "weights should sum to 1");
    }

    // Test 1b: hidden=[0,0,0,1] should favor lora 2
    {
        float hidden[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        alpha_route_result res = alpha_router_forward(&router, hidden, 2);

        TEST_ASSERT(res.n_selected == 2, "should select 2 LoRAs");
        TEST_ASSERT(res.indices[0] == 2, "top LoRA should be index 2");
        TEST_ASSERT(res.weights[0] > res.weights[1], "top weight should be highest");
        TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-5f, "weights should sum to 1");
    }

    // Test 1c: hidden=[0,0,0,0] (all zeros) — should return uniform-ish scores
    {
        float hidden[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        alpha_route_result res = alpha_router_forward(&router, hidden, 2);

        TEST_ASSERT(res.n_selected == 2, "should select 2 LoRAs");
        TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-5f, "weights should sum to 1");
    }

    test_pass("test_router_forward_basic");
    return true;
}

// ============================================================================
// Test 2: Top-K selection edge cases
// ============================================================================

static bool test_router_topk_edge_cases() {
    alpha_router router = {};
    router.hidden_dim    = 4;
    router.router_hidden = 2;
    router.n_loras       = 3;
    router.top_k         = 2;
    router.shared        = true;
    router.w_down = {1.0f,0.0f, 0.5f,0.5f, 0.0f,0.0f, 0.0f,1.0f};
    router.w_up   = {2.0f,0.0f,0.0f, 0.0f,0.0f,2.0f};

    float hidden[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    // Top-K = 1
    {
        alpha_route_result res = alpha_router_forward(&router, hidden, 1);
        TEST_ASSERT(res.n_selected == 1, "top-1 should select 1");
        TEST_ASSERT(res.indices[0] == 0, "top-1 should pick lora 0");
        TEST_FLOAT_EQ(res.weights[0], 1.0f, 1e-5f, "top-1 weight should be 1.0");
        TEST_ASSERT(res.indices[1] == -1, "unused slot should be -1");
    }

    // Top-K = 3 (all loras)
    {
        alpha_route_result res = alpha_router_forward(&router, hidden, 3);
        TEST_ASSERT(res.n_selected == 3, "top-3 should select 3");
        TEST_FLOAT_EQ(res.weights[0] + res.weights[1] + res.weights[2], 1.0f, 1e-5f,
                      "all weights should sum to 1");
    }

    // Top-K = 0 (clamped to 1)
    {
        alpha_route_result res = alpha_router_forward(&router, hidden, 0);
        TEST_ASSERT(res.n_selected == 1, "top-0 should clamp to 1");
    }

    // Top-K = 100 (clamped to n_loras=3)
    {
        alpha_route_result res = alpha_router_forward(&router, hidden, 100);
        TEST_ASSERT(res.n_selected == 3, "top-100 should clamp to n_loras=3");
    }

    // Null inputs
    {
        alpha_route_result res = alpha_router_forward(nullptr, hidden, 2);
        TEST_ASSERT(res.n_selected == 0, "null router should return empty result");

        res = alpha_router_forward(&router, nullptr, 2);
        TEST_ASSERT(res.n_selected == 0, "null hidden should return empty result");
    }

    test_pass("test_router_topk_edge_cases");
    return true;
}

// ============================================================================
// Test 3: Router with biases
// ============================================================================

static bool test_router_with_biases() {
    alpha_router router = {};
    router.hidden_dim    = 2;
    router.router_hidden = 2;
    router.n_loras       = 2;
    router.top_k         = 2;
    router.shared        = true;

    // Identity-like down weights
    router.w_down = {1.0f, 0.0f, 0.0f, 1.0f};
    // Identity-like up weights
    router.w_up   = {1.0f, 0.0f, 0.0f, 1.0f};

    // Add bias to shift scores: lora 1 gets +10 bias
    router.b_down = {};  // no down bias
    router.b_up   = {0.0f, 10.0f};

    // hidden=[1,0]: without bias -> lora 0 wins. With +10 bias on lora 1 -> lora 1 should win
    float hidden[2] = {1.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 1);

    TEST_ASSERT(res.n_selected == 1, "should select 1");
    TEST_ASSERT(res.indices[0] == 1, "bias should make lora 1 win");

    test_pass("test_router_with_biases");
    return true;
}

// ============================================================================
// Test 4: Router GGUF round-trip (write then load)
// ============================================================================

static bool test_router_gguf_roundtrip() {
    const char * path = "test_router_tmp.gguf";

    // Create test GGUF file
    TEST_ASSERT(create_test_router_gguf(path), "failed to create test router GGUF");

    // Load it back
    alpha_router * router = alpha_router_load(path);
    TEST_ASSERT(router != nullptr, "failed to load router from GGUF");

    TEST_ASSERT(router->hidden_dim    == 4, "hidden_dim mismatch");
    TEST_ASSERT(router->router_hidden == 2, "router_hidden mismatch");
    TEST_ASSERT(router->n_loras       == 3, "n_loras mismatch");
    TEST_ASSERT(router->top_k         == 2, "top_k mismatch");
    TEST_ASSERT(router->shared        == true, "shared mismatch");

    // Verify weights match what we wrote
    TEST_FLOAT_EQ(router->w_down[0], 1.0f, 1e-6f, "w_down[0] mismatch");
    TEST_FLOAT_EQ(router->w_down[1], 0.0f, 1e-6f, "w_down[1] mismatch");
    TEST_FLOAT_EQ(router->w_up[0],   2.0f, 1e-6f, "w_up[0] mismatch");
    TEST_FLOAT_EQ(router->w_up[5],   2.0f, 1e-6f, "w_up[5] mismatch");

    // Run forward pass and verify same results as manual router
    float hidden[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(router, hidden, 2);
    TEST_ASSERT(res.indices[0] == 0, "GGUF-loaded router should favor lora 0 for hidden=[1,0,0,0]");

    float hidden2[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    alpha_route_result res2 = alpha_router_forward(router, hidden2, 2);
    TEST_ASSERT(res2.indices[0] == 2, "GGUF-loaded router should favor lora 2 for hidden=[0,0,0,1]");

    alpha_router_free(router);

    // Cleanup temp file
    remove(path);

    test_pass("test_router_gguf_roundtrip");
    return true;
}

// ============================================================================
// Test 5: Router GGUF validation — bad files
// ============================================================================

static bool test_router_gguf_validation() {
    // Null path
    {
        alpha_router * r = alpha_router_load(nullptr);
        TEST_ASSERT(r == nullptr, "null path should return nullptr");
    }

    // Nonexistent file
    {
        alpha_router * r = alpha_router_load("nonexistent_file_12345.gguf");
        TEST_ASSERT(r == nullptr, "nonexistent file should return nullptr");
    }

    // Create GGUF with wrong version
    {
        struct gguf_context * gguf_ctx = gguf_init_empty();
        gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.version", 99);
        gguf_write_to_file(gguf_ctx, "test_bad_version.gguf", false);
        gguf_free(gguf_ctx);

        alpha_router * r = alpha_router_load("test_bad_version.gguf");
        TEST_ASSERT(r == nullptr, "wrong version should return nullptr");
        remove("test_bad_version.gguf");
    }

    // Create GGUF with missing tensors
    {
        struct gguf_context * gguf_ctx = gguf_init_empty();
        gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.version", 1);
        gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.n_loras", 3);
        gguf_set_val_i32(gguf_ctx, "alpha_lora_mix.router_hidden", 2);
        gguf_write_to_file(gguf_ctx, "test_no_tensors.gguf", false);
        gguf_free(gguf_ctx);

        alpha_router * r = alpha_router_load("test_no_tensors.gguf");
        TEST_ASSERT(r == nullptr, "missing tensors should return nullptr");
        remove("test_no_tensors.gguf");
    }

    test_pass("test_router_gguf_validation");
    return true;
}

// ============================================================================
// Test 6: Softmax numerical stability
// ============================================================================

static bool test_router_softmax_stability() {
    alpha_router router = {};
    router.hidden_dim    = 2;
    router.router_hidden = 1;
    router.n_loras       = 2;
    router.top_k         = 2;
    router.shared        = true;

    // Large weight values that could cause overflow without max-subtraction
    router.w_down = {1000.0f, 0.0f};
    router.w_up   = {1.0f, -1.0f};

    float hidden[2] = {1.0f, 0.0f};
    alpha_route_result res = alpha_router_forward(&router, hidden, 2);

    // Should not produce NaN or Inf
    TEST_ASSERT(!std::isnan(res.weights[0]), "weight[0] should not be NaN");
    TEST_ASSERT(!std::isnan(res.weights[1]), "weight[1] should not be NaN");
    TEST_ASSERT(!std::isinf(res.weights[0]), "weight[0] should not be Inf");
    TEST_ASSERT(!std::isinf(res.weights[1]), "weight[1] should not be Inf");
    TEST_FLOAT_EQ(res.weights[0] + res.weights[1], 1.0f, 1e-4f, "weights should sum to 1");

    // Lora 0 should dominate (positive score vs negative)
    TEST_ASSERT(res.indices[0] == 0, "lora 0 should win with large positive score");
    TEST_ASSERT(res.weights[0] > 0.9f, "lora 0 should have near-1.0 weight");

    test_pass("test_router_softmax_stability");
    return true;
}

// ============================================================================
// Test 7: alpha_lora_mix_params validation
// ============================================================================

static bool test_lora_mix_init_validation() {
    // Null model
    {
        alpha_lora_mix_params p = {};
        p.n_loras = 1;
        const char * paths[] = {"dummy.gguf"};
        p.lora_paths = paths;
        auto * m = alpha_lora_mix_init(nullptr, (llama_context*)1, p);
        TEST_ASSERT(m == nullptr, "null model should fail");
    }

    // Null context
    {
        alpha_lora_mix_params p = {};
        p.n_loras = 1;
        const char * paths[] = {"dummy.gguf"};
        p.lora_paths = paths;
        auto * m = alpha_lora_mix_init((llama_model*)1, nullptr, p);
        TEST_ASSERT(m == nullptr, "null context should fail");
    }

    // Zero loras: valid, the base model alone (RAW strategy)
    {
        alpha_lora_mix_params p = {};
        p.n_loras = 0;
        auto * m = alpha_lora_mix_init((llama_model*)1, (llama_context*)1, p);
        TEST_ASSERT(m != nullptr, "zero loras should give a RAW context");
        TEST_ASSERT(alpha_lora_mix_get_strategy(m) == LORA_MIX_STRATEGY_RAW, "zero loras should select RAW");
        alpha_lora_mix_free(m);
    }

    // Too many loras
    {
        alpha_lora_mix_params p = {};
        p.n_loras = ALPHA_LORA_MIX_MAX_LORAS + 1;
        const char * paths[] = {"dummy.gguf"};
        p.lora_paths = paths;
        auto * m = alpha_lora_mix_init((llama_model*)1, (llama_context*)1, p);
        TEST_ASSERT(m == nullptr, "too many loras should fail");
    }

    // Null paths
    {
        alpha_lora_mix_params p = {};
        p.n_loras = 1;
        p.lora_paths = nullptr;
        auto * m = alpha_lora_mix_init((llama_model*)1, (llama_context*)1, p);
        TEST_ASSERT(m == nullptr, "null paths should fail");
    }

    // Null decode
    {
        llama_batch batch = {};
        int ret = alpha_lora_mix_decode(nullptr, batch);
        TEST_ASSERT(ret == -1, "null mctx should return -1");
    }

    // Null get_routing
    {
        alpha_lora_mix_routing_info info = {};
        int ret = alpha_lora_mix_get_routing(nullptr, &info);
        TEST_ASSERT(ret == -1, "null mctx should return -1");

        alpha_lora_mix_context * mctx = (alpha_lora_mix_context *)1;  // dummy
        ret = alpha_lora_mix_get_routing(mctx, nullptr);
        TEST_ASSERT(ret == -1, "null info should return -1");
    }

    // alpha_lora_mix_free with null (should not crash)
    alpha_lora_mix_free(nullptr);

    // alpha_lora_mix_routing_info_free with null (should not crash)
    alpha_lora_mix_routing_info_free(nullptr);

    test_pass("test_lora_mix_init_validation");
    return true;
}

// ============================================================================
// Test 8: Routing decisions differ for different inputs
// ============================================================================

static bool test_router_different_inputs_different_routes() {
    alpha_router router = {};
    router.hidden_dim    = 4;
    router.router_hidden = 2;
    router.n_loras       = 3;
    router.top_k         = 1;
    router.shared        = true;
    router.w_down = {1.0f,0.0f, 0.5f,0.5f, 0.0f,0.0f, 0.0f,1.0f};
    router.w_up   = {2.0f,0.0f,0.0f, 0.0f,0.0f,2.0f};

    // Two very different hidden states should route to different LoRAs
    float hidden_a[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // should -> lora 0
    float hidden_b[4] = {0.0f, 0.0f, 0.0f, 1.0f};  // should -> lora 2

    alpha_route_result res_a = alpha_router_forward(&router, hidden_a, 1);
    alpha_route_result res_b = alpha_router_forward(&router, hidden_b, 1);

    TEST_ASSERT(res_a.indices[0] != res_b.indices[0],
                "different hidden states should route to different LoRAs");
    TEST_ASSERT(res_a.indices[0] == 0, "hidden_a should route to lora 0");
    TEST_ASSERT(res_b.indices[0] == 2, "hidden_b should route to lora 2");

    test_pass("test_router_different_inputs_different_routes");
    return true;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    fprintf(stderr, "\n=== LoRA-mixture Test Suite ===\n\n");

    // Router unit tests (no model/LoRA files needed)
    fprintf(stderr, "--- Router MLP Tests ---\n");
    test_router_forward_basic();
    test_router_topk_edge_cases();
    test_router_with_biases();
    test_router_softmax_stability();
    test_router_different_inputs_different_routes();

    // GGUF round-trip tests
    fprintf(stderr, "\n--- Router GGUF Tests ---\n");
    test_router_gguf_roundtrip();
    test_router_gguf_validation();

    // API validation tests (no real model needed)
    fprintf(stderr, "\n--- LoRA-mixture API Validation Tests ---\n");
    test_lora_mix_init_validation();

    // Summary
    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n\n",
            tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
