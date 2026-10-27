#include "alpha_train_novel.h"

#include <algorithm>

static constexpr int TIER_2_THRESHOLD = 50000;
static constexpr float FINITE_DIFF_EPS = 1e-5f;

// ============================================================
// Initialization
// ============================================================

void alpha_novel_init_layer(alpha_novel_layer & layer) {
    // Auto-select tier
    layer.tier = (layer.n_params < TIER_2_THRESHOLD)
        ? alpha_novel_layer::TIER_2_FINITE_DIFF
        : alpha_novel_layer::TIER_3_LOCAL_MEZO;

    // Initialize params if not provided
    if ((int)layer.params.size() != layer.n_params) {
        layer.params.resize(layer.n_params, 0.0f);
    }

    layer.grad.resize(layer.n_params, 0.0f);
    layer.m.resize(layer.n_params, 0.0f);
    layer.v.resize(layer.n_params, 0.0f);
    layer.params_anchor = layer.params;
    layer.saved_n_tokens = 0;

    if (layer.tier == alpha_novel_layer::TIER_3_LOCAL_MEZO) {
        layer.z.resize(layer.n_params, 0.0f);
        if (layer.mezo_k <= 0) layer.mezo_k = 4;
    }

    fprintf(stderr, "ext_layer: '%s' (pos %d, %d params, mode %d)\n",
            layer.name.c_str(), layer.layer_id, layer.n_params, layer.tier);
}

// ============================================================
// Forward
// ============================================================

void alpha_novel_forward_all(
    std::vector<alpha_novel_layer> & layers,
    int layer_id,
    float * hidden,
    int n_tokens)
{
    for (auto & nl : layers) {
        if (nl.layer_id != layer_id) continue;
        if (!nl.forward_fn) continue;

        int sz = n_tokens * nl.hidden_dim;

        // Save input
        nl.input_saved.resize(sz);
        memcpy(nl.input_saved.data(), hidden, sz * sizeof(float));
        nl.saved_n_tokens = n_tokens;

        // Forward into temp buffer then copy back
        std::vector<float> output(sz);
        nl.forward_fn(output.data(), hidden, nl.params.data(), n_tokens, nl.hidden_dim);
        memcpy(hidden, output.data(), sz * sizeof(float));
    }
}

// ============================================================
// Tier 2: Finite-difference backward
// ============================================================

static void tier2_backward(
    alpha_novel_layer & nl,
    const float * d_output,
    float * d_input,
    int n_tokens)
{
    if (!nl.forward_fn || nl.input_saved.empty()) return;

    int out_size = n_tokens * nl.hidden_dim;

    // Baseline: y = f(x, theta)
    std::vector<float> y_base(out_size);
    nl.forward_fn(y_base.data(), nl.input_saved.data(), nl.params.data(),
                  n_tokens, nl.hidden_dim);

    // Parameter gradients: dL/dtheta_i = sum_j d_output[j] * (f(x, theta + eps*e_i) - f(x, theta))[j] / eps
    std::vector<float> y_pert(out_size);
    std::vector<float> params_tmp = nl.params;

    for (int i = 0; i < nl.n_params; i++) {
        params_tmp[i] = nl.params[i] + FINITE_DIFF_EPS;
        nl.forward_fn(y_pert.data(), nl.input_saved.data(), params_tmp.data(),
                      n_tokens, nl.hidden_dim);

        float g = 0.0f;
        for (int j = 0; j < out_size; j++) {
            g += d_output[j] * (y_pert[j] - y_base[j]) / FINITE_DIFF_EPS;
        }
        nl.grad[i] += g;

        params_tmp[i] = nl.params[i];
    }

    // Input gradient: dL/dx_j = sum_k d_output[k] * (f(x + eps*e_j, theta) - f(x, theta))[k] / eps
    if (d_input) {
        int in_size = n_tokens * nl.hidden_dim;
        std::vector<float> input_tmp = nl.input_saved;

        for (int j = 0; j < in_size; j++) {
            float saved = input_tmp[j];
            input_tmp[j] = saved + FINITE_DIFF_EPS;

            nl.forward_fn(y_pert.data(), input_tmp.data(), nl.params.data(),
                          n_tokens, nl.hidden_dim);

            float g = 0.0f;
            for (int k = 0; k < out_size; k++) {
                g += d_output[k] * (y_pert[k] - y_base[k]) / FINITE_DIFF_EPS;
            }
            d_input[j] = g;

            input_tmp[j] = saved;
        }
    }
}

// ============================================================
// Backward dispatch
// ============================================================

void alpha_novel_backward_all(
    std::vector<alpha_novel_layer> & layers,
    int layer_id,
    const float * d_output,
    float * d_input,
    int n_tokens)
{
    for (auto & nl : layers) {
        if (nl.layer_id != layer_id) continue;

        if (nl.tier == alpha_novel_layer::TIER_2_FINITE_DIFF) {
            tier2_backward(nl, d_output, d_input, n_tokens);
        } else {
            // Tier 3: gradient set via mezo_accumulate, just pass gradient through
            if (d_input && d_output) {
                memcpy(d_input, d_output, n_tokens * nl.hidden_dim * sizeof(float));
            }
        }
    }
}

// ============================================================
// MeZO for Tier 3
// ============================================================

void alpha_novel_mezo_sample_z(std::vector<alpha_novel_layer> & layers, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & nl : layers) {
        if (nl.tier == alpha_novel_layer::TIER_3_LOCAL_MEZO) {
            for (int i = 0; i < nl.n_params; i++) nl.z[i] = dist(rng);
        }
    }
}

void alpha_novel_mezo_perturb(std::vector<alpha_novel_layer> & layers, float sign, float eps) {
    for (auto & nl : layers) {
        if (nl.tier == alpha_novel_layer::TIER_3_LOCAL_MEZO) {
            for (int i = 0; i < nl.n_params; i++) nl.params[i] += sign * eps * nl.z[i];
        }
    }
}

void alpha_novel_mezo_accumulate(std::vector<alpha_novel_layer> & layers, float g_proj) {
    for (auto & nl : layers) {
        if (nl.tier == alpha_novel_layer::TIER_3_LOCAL_MEZO) {
            for (int i = 0; i < nl.n_params; i++) nl.grad[i] += g_proj * nl.z[i];
        }
    }
}

// ============================================================
// Adam update
// ============================================================

void alpha_novel_update_all(
    std::vector<alpha_novel_layer> & layers,
    float lr, float beta1, float beta2,
    float adam_eps, float elastic_lambda,
    int step)
{
    if (step <= 0) return;

    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);

    for (auto & nl : layers) {
        for (int i = 0; i < nl.n_params; i++) {
            float g = nl.grad[i];
            if (elastic_lambda > 0 && !nl.params_anchor.empty()) {
                g += elastic_lambda * (nl.params[i] - nl.params_anchor[i]);
            }

            nl.m[i] = beta1 * nl.m[i] + (1.0f - beta1) * g;
            nl.v[i] = beta2 * nl.v[i] + (1.0f - beta2) * g * g;

            float m_hat = nl.m[i] / bc1;
            float v_hat = nl.v[i] / bc2;
            nl.params[i] -= lr * m_hat / (sqrtf(v_hat) + adam_eps);
        }
        std::fill(nl.grad.begin(), nl.grad.end(), 0.0f);
    }
}

// ============================================================
// Summary
// ============================================================

void alpha_novel_print_summary(const std::vector<alpha_novel_layer> & layers) {
    if (layers.empty()) return;

    fprintf(stderr, "\n=== Extension Layers ===\n");
    int total = 0;
    for (const auto & nl : layers) {
        fprintf(stderr, "  %s: %d params, mode %d\n",
                nl.name.c_str(), nl.n_params, nl.tier);
        total += nl.n_params;
    }
    fprintf(stderr, "  Total: %zu layers, %d params\n", layers.size(), total);
    fprintf(stderr, "========================\n");
}
