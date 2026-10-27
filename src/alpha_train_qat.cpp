#include "alpha_train_qat.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// QAT initialization

static bool is_trainable_type(enum ggml_type type, const alpha_qat_config & cfg) {
    switch (type) {
        case GGML_TYPE_Q2_K: return cfg.train_q2k;
        case GGML_TYPE_Q3_K: return cfg.train_q3k;
        case GGML_TYPE_Q4_K: return cfg.train_q4k;
        case GGML_TYPE_Q5_K: return cfg.train_q5k;
        case GGML_TYPE_Q6_K: return cfg.train_q6k;
        case GGML_TYPE_Q8_0: return cfg.train_q8;
        default: return false;
    }
}

alpha_qat_state * alpha_qat_init(
    struct llama_model * model,
    alpha_qat_config config)
{
    if (!model) return nullptr;

    auto * state = new alpha_qat_state();
    state->total_shadow_bytes = 0;
    state->n_trainable = 0;

    // Iterate all model layers, find quantized weight tensors
    for (int il = 0; il < (int)model->hparams.n_layer; il++) {
        auto & layer = model->layers[il];

        // Collect candidate tensors
        ggml_tensor * candidates[] = {
            layer.wq, layer.wk, layer.wv, layer.wo,
            layer.ffn_gate, layer.ffn_up, layer.ffn_down,
            layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps,
            nullptr
        };

        for (int t = 0; candidates[t]; t++) {
            ggml_tensor * tensor = candidates[t];
            if (!tensor || !tensor->data) continue;
            if (!is_trainable_type(tensor->type, config)) continue;

            alpha_qat_shadow shadow = {};
            shadow.original = tensor;
            shadow.nelements = ggml_nelements(tensor);

            if (config.use_shadow_weights) {
                // Dequantize to FP32 shadow
                shadow.fp32.resize(shadow.nelements);
                shadow.grad.resize(shadow.nelements, 0.0f);

                const auto * traits = ggml_get_type_traits(tensor->type);
                if (traits && traits->to_float) {
                    traits->to_float(tensor->data, shadow.fp32.data(), shadow.nelements);
                }
            }

            state->shadows.push_back(std::move(shadow));
            state->total_shadow_bytes += shadow.nelements * sizeof(float) * 2;  // fp32 + grad
            state->n_trainable++;
        }
    }

    fprintf(stderr, "alpha_qat_init: %d quantized tensors trainable\n", state->n_trainable);
    fprintf(stderr, "  Shadow memory: %.1f GB (FP32 copies + gradients)\n",
            state->total_shadow_bytes / 1e9f);

    return state;
}

void alpha_qat_free(alpha_qat_state * state) {
    delete state;
}

// QAT optimizer step

void alpha_qat_step(
    alpha_qat_state * state,
    alpha_qat_config config,
    int step)
{
    if (!state) return;

    float lr = config.lr;
    if (config.warmup_steps > 0 && step <= config.warmup_steps) {
        lr = config.lr * (float)step / (float)config.warmup_steps;
    }

    // Gradient clipping (global norm)
    if (config.grad_clip > 0) {
        float total_norm_sq = 0.0f;
        for (auto & shadow : state->shadows) {
            for (int64_t i = 0; i < shadow.nelements; i++) {
                total_norm_sq += shadow.grad[i] * shadow.grad[i];
            }
        }
        float total_norm = sqrtf(total_norm_sq);
        if (total_norm > config.grad_clip) {
            float scale = config.grad_clip / total_norm;
            for (auto & shadow : state->shadows) {
                for (int64_t i = 0; i < shadow.nelements; i++) {
                    shadow.grad[i] *= scale;
                }
            }
        }
    }

    // Adam update on shadow weights
    // Note: Adam state (m, v) is stored inline in this simplified version
    // For production, use the adam_state struct from alpha_train_full.cpp
    static std::vector<float> adam_m, adam_v;
    static int adam_t = 0;

    size_t total_n = 0;
    for (auto & s : state->shadows) total_n += s.nelements;

    if (adam_m.size() != total_n) {
        adam_m.resize(total_n, 0.0f);
        adam_v.resize(total_n, 0.0f);
        adam_t = 0;
    }
    adam_t++;

    float beta1_t = 1.0f - powf(config.beta1, (float)adam_t);
    float beta2_t = 1.0f - powf(config.beta2, (float)adam_t);

    size_t offset = 0;
    for (auto & shadow : state->shadows) {
        for (int64_t i = 0; i < shadow.nelements; i++) {
            float g = shadow.grad[i];
            if (config.weight_decay > 0) {
                g += config.weight_decay * shadow.fp32[i];
            }

            adam_m[offset + i] = config.beta1 * adam_m[offset + i] + (1.0f - config.beta1) * g;
            adam_v[offset + i] = config.beta2 * adam_v[offset + i] + (1.0f - config.beta2) * g * g;

            float m_hat = adam_m[offset + i] / beta1_t;
            float v_hat = adam_v[offset + i] / beta2_t;

            shadow.fp32[i] -= lr * m_hat / (sqrtf(v_hat) + config.eps);
        }
        offset += shadow.nelements;

        // Clear gradients for next step
        memset(shadow.grad.data(), 0, shadow.nelements * sizeof(float));
    }

    // Re-quantize if this step is a requantize step
    if (config.requantize_every > 0 && step % config.requantize_every == 0) {
        alpha_qat_requantize(state);
    }
}

// Re-quantize shadow weights back into the model

void alpha_qat_requantize(alpha_qat_state * state) {
    if (!state) return;

    for (auto & shadow : state->shadows) {
        ggml_tensor * tensor = shadow.original;
        if (!tensor || !tensor->data || shadow.fp32.empty()) continue;

        // Quantize FP32 shadow back into the tensor's native type
        // Use ggml's quantization function for the tensor's type
        const auto * traits = ggml_get_type_traits(tensor->type);
        if (traits && traits->from_float_ref) {
            traits->from_float_ref(shadow.fp32.data(), tensor->data, shadow.nelements);
        }
    }
}
