#include "alpha_router.h"
#include "alpha_mole.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

// GGUF metadata keys for router format
static const char * ALPHA_MOLE_KEY_VERSION       = "alpha_mole.version";
static const char * ALPHA_MOLE_KEY_N_LORAS       = "alpha_mole.n_loras";
static const char * ALPHA_MOLE_KEY_TOP_K         = "alpha_mole.top_k";
static const char * ALPHA_MOLE_KEY_ROUTER_HIDDEN = "alpha_mole.router_hidden";
static const char * ALPHA_MOLE_KEY_SHARED        = "alpha_mole.shared";

// Tensor names
static const char * ALPHA_MOLE_TENSOR_DOWN       = "alpha_mole.router.down.weight";
static const char * ALPHA_MOLE_TENSOR_UP         = "alpha_mole.router.up.weight";
static const char * ALPHA_MOLE_TENSOR_DOWN_BIAS  = "alpha_mole.router.down.bias";
static const char * ALPHA_MOLE_TENSOR_UP_BIAS    = "alpha_mole.router.up.bias";

alpha_router * alpha_router_load(const char * path) {
    if (!path) {
        return nullptr;
    }

    struct ggml_context * ggml_ctx = nullptr;

    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ggml_ctx,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "alpha_router: failed to load GGUF from '%s'\n", path);
        return nullptr;
    }

    // Cleanup helper — ensures both contexts are freed on all exit paths
    auto cleanup = [&]() {
        if (ggml_ctx) { ggml_free(ggml_ctx); }
        gguf_free(gguf_ctx);
    };

    // Read metadata
    int version_idx = gguf_find_key(gguf_ctx, ALPHA_MOLE_KEY_VERSION);
    if (version_idx < 0) {
        fprintf(stderr, "alpha_router: missing '%s' in '%s'\n", ALPHA_MOLE_KEY_VERSION, path);
        cleanup();
        return nullptr;
    }

    // Read integer metadata — handle both INT32 and UINT32 types
    auto get_int = [&](int idx) -> int {
        enum gguf_type type = gguf_get_kv_type(gguf_ctx, idx);
        if (type == GGUF_TYPE_UINT32) {
            return (int)gguf_get_val_u32(gguf_ctx, idx);
        }
        return (int)gguf_get_val_i32(gguf_ctx, idx);
    };

    int version = get_int(version_idx);
    if (version != 1) {
        fprintf(stderr, "alpha_router: unsupported version %d (expected 1)\n", version);
        cleanup();
        return nullptr;
    }

    auto get_i32 = [&](const char * key, int default_val) -> int {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? get_int(idx) : default_val;
    };

    auto get_bool = [&](const char * key, bool default_val) -> bool {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? gguf_get_val_bool(gguf_ctx, idx) : default_val;
    };

    int  n_loras       = get_i32(ALPHA_MOLE_KEY_N_LORAS, 0);
    int  top_k         = get_i32(ALPHA_MOLE_KEY_TOP_K, 2);
    int  router_hidden = get_i32(ALPHA_MOLE_KEY_ROUTER_HIDDEN, 64);
    bool shared        = get_bool(ALPHA_MOLE_KEY_SHARED, true);

    if (n_loras <= 0 || router_hidden <= 0) {
        fprintf(stderr, "alpha_router: invalid config (n_loras=%d, router_hidden=%d)\n", n_loras, router_hidden);
        cleanup();
        return nullptr;
    }

    // Clamp loaded top_k to valid range (hardening #3)
    if (top_k < 1) top_k = 1;
    if (top_k > ALPHA_MOLE_MAX_TOP_K) top_k = ALPHA_MOLE_MAX_TOP_K;

    // Find tensors
    struct ggml_tensor * t_down = ggml_get_tensor(ggml_ctx, ALPHA_MOLE_TENSOR_DOWN);
    struct ggml_tensor * t_up   = ggml_get_tensor(ggml_ctx, ALPHA_MOLE_TENSOR_UP);

    if (!t_down || !t_up) {
        fprintf(stderr, "alpha_router: missing required tensors in '%s'\n", path);
        cleanup();
        return nullptr;
    }

    // Validate tensor type is F32 (hardening #1)
    if (t_down->type != GGML_TYPE_F32 || t_up->type != GGML_TYPE_F32) {
        fprintf(stderr, "alpha_router: tensors must be F32 (got down=%d, up=%d)\n",
                t_down->type, t_up->type);
        cleanup();
        return nullptr;
    }

    // Validate tensors are 2D (hardening #5)
    if (ggml_n_dims(t_down) != 2 || ggml_n_dims(t_up) != 2) {
        fprintf(stderr, "alpha_router: tensors must be 2D (got down=%dD, up=%dD)\n",
                ggml_n_dims(t_down), ggml_n_dims(t_up));
        cleanup();
        return nullptr;
    }

    int hidden_dim = (int)t_down->ne[0];

    // Validate dimensions
    if ((int)t_down->ne[1] != router_hidden) {
        fprintf(stderr, "alpha_router: down weight shape mismatch (%lld vs %d)\n",
                (long long)t_down->ne[1], router_hidden);
        cleanup();
        return nullptr;
    }

    if ((int)t_up->ne[0] != router_hidden || (int)t_up->ne[1] != n_loras) {
        fprintf(stderr, "alpha_router: up weight shape mismatch\n");
        cleanup();
        return nullptr;
    }

    // Create router and copy weights (must happen before cleanup frees tensor data)
    auto * router = new alpha_router();
    router->hidden_dim    = hidden_dim;
    router->router_hidden = router_hidden;
    router->n_loras       = n_loras;
    router->top_k         = top_k;
    router->shared        = shared;

    // Use size_t to prevent integer overflow (hardening #4)
    size_t down_size = (size_t)hidden_dim * router_hidden;
    size_t up_size   = (size_t)router_hidden * n_loras;

    router->w_down.resize(down_size);
    router->w_up.resize(up_size);

    memcpy(router->w_down.data(), t_down->data, down_size * sizeof(float));
    memcpy(router->w_up.data(),   t_up->data,   up_size   * sizeof(float));

    // Load biases if present
    struct ggml_tensor * t_down_bias = ggml_get_tensor(ggml_ctx, ALPHA_MOLE_TENSOR_DOWN_BIAS);
    struct ggml_tensor * t_up_bias   = ggml_get_tensor(ggml_ctx, ALPHA_MOLE_TENSOR_UP_BIAS);

    if (t_down_bias) {
        // Validate bias dimensions (hardening #2)
        if (t_down_bias->type != GGML_TYPE_F32 || (int)t_down_bias->ne[0] != router_hidden) {
            fprintf(stderr, "alpha_router: down bias shape/type mismatch, ignoring\n");
        } else {
            router->b_down.resize(router_hidden);
            memcpy(router->b_down.data(), t_down_bias->data, router_hidden * sizeof(float));
        }
    }

    if (t_up_bias) {
        // Validate bias dimensions (hardening #2)
        if (t_up_bias->type != GGML_TYPE_F32 || (int)t_up_bias->ne[0] != n_loras) {
            fprintf(stderr, "alpha_router: up bias shape/type mismatch, ignoring\n");
        } else {
            router->b_up.resize(n_loras);
            memcpy(router->b_up.data(), t_up_bias->data, n_loras * sizeof(float));
        }
    }

    fprintf(stderr, "alpha_router: loaded router from '%s' (hidden_dim=%d, router_hidden=%d, n_loras=%d, top_k=%d)\n",
            path, hidden_dim, router_hidden, n_loras, top_k);

    cleanup();
    return router;
}

alpha_route_result alpha_router_forward(
        const alpha_router * router,
        const float        * hidden,
        int                  top_k) {
    alpha_route_result result = {};

    if (!router || !hidden) {
        return result;
    }

    if (top_k < 1) top_k = 1;
    if (top_k > router->n_loras) top_k = router->n_loras;
    if (top_k > ALPHA_MOLE_MAX_TOP_K) top_k = ALPHA_MOLE_MAX_TOP_K;

    const int H = router->hidden_dim;
    const int R = router->router_hidden;
    const int N = router->n_loras;

    // Layer 1: hidden_state @ w_down -> intermediate [R]
    // w_down is [H x R], stored row-major
    std::vector<float> intermediate(R, 0.0f);
    for (int r = 0; r < R; r++) {
        float sum = 0.0f;
        for (int h = 0; h < H; h++) {
            sum += hidden[h] * router->w_down[h * R + r];
        }
        if (!router->b_down.empty()) {
            sum += router->b_down[r];
        }
        // ReLU
        intermediate[r] = sum > 0.0f ? sum : 0.0f;
    }

    // Layer 2: intermediate @ w_up -> scores [N]
    // w_up is [R x N], stored row-major
    std::vector<float> scores(N, 0.0f);
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int r = 0; r < R; r++) {
            sum += intermediate[r] * router->w_up[r * N + n];
        }
        if (!router->b_up.empty()) {
            sum += router->b_up[n];
        }
        scores[n] = sum;
    }

    // Softmax over all scores
    float max_score = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;
    for (int n = 0; n < N; n++) {
        scores[n] = expf(scores[n] - max_score);
        sum_exp += scores[n];
    }
    for (int n = 0; n < N; n++) {
        scores[n] /= sum_exp;
    }

    // Top-K selection
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&scores](int a, int b) { return scores[a] > scores[b]; });

    // Re-normalize top-K weights
    float top_sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        top_sum += scores[indices[k]];
    }

    result.n_selected = top_k;
    for (int k = 0; k < top_k; k++) {
        result.indices[k] = indices[k];
        result.weights[k] = scores[indices[k]] / top_sum;
    }
    // Zero out remaining slots
    for (int k = top_k; k < 4; k++) {
        result.indices[k] = -1;
        result.weights[k] = 0.0f;
    }

    return result;
}

void alpha_router_free(alpha_router * router) {
    delete router;
}
