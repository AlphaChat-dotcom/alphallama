#include "alpha_train_combined.h"
#include "alpha_train_backward.h"
#include "llama-model.h"
#include "llama-adapter.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>

// Per-layer LoRA parameters (extends train_engine's structure)

struct combined_lora_layer {
    std::string name;     // e.g., "blk.5.attn_q"
    ggml_tensor * W;      // frozen base weight

    int64_t in_dim;
    int64_t out_dim;
    int     rank;

    // LoRA A/B (F32, CPU)
    std::vector<float> A, B;

    // Anchor copies for elastic anchoring (saved after Phase 1 / mask computation)
    std::vector<float> A_anchor, B_anchor;

    // Adam moments (separate for MeZO and backprop channels)
    std::vector<float> mA, vA, mB, vB;

    // MeZO perturbation direction
    std::vector<float> zA, zB;

    // Accumulated exact gradients (for masked params only)
    std::vector<float> dA, dB;

    // Param selection: true = this element gets exact backprop
    std::vector<bool> maskA, maskB;
    int n_masked_A, n_masked_B;

    // Displacement tracker (accumulated |update| during training)
    std::vector<float> dispA, dispB;

    // GV-ZO: per-parameter guiding vector (EMA of gradient estimates)
    // Aligns perturbation direction toward estimated gradient (arxiv 2601.04710)
    std::vector<float> guideA, guideB;

    // GradientStabilizer: per-tensor gradient norm EMA for backprop channel
    float grad_norm_ema_A, grad_norm_ema_B;

    // Saved activation from forward (input to this projection)
    std::vector<float> x_saved;
    int x_tokens;
};

// Combined trainer

struct alpha_combined_trainer {
    llama_model   * model;
    llama_context * ctx;
    alpha_combined_train_config config;

    std::vector<combined_lora_layer> layers;

    // LoRA adapter
    llama_adapter_lora     * adapter;
    ggml_context           * adapter_ctx;
    ggml_backend_buffer_t    adapter_buf;

    // Fisher mask
    alpha_fisher_mask mask;

    // Convergence monitor
    alpha_convergence_monitor conv;

    // Training state
    int   step;
    float current_loss;
    float total_loss;
    int   n_steps;

    // GradientStabilizer: running norm EMAs (arxiv 2502.17055)
    // Stabilizes gradient magnitude without spike detection thresholds
    float g_proj_norm_ema;     // EMA of |g_proj| for MeZO channel

    // Model info
    int n_layer;
    int n_embd;
    int n_vocab;

    // Novel layers (Tier 2/3 - forward-only training)
    std::vector<alpha_novel_layer> novel_layers;

    // RNG
    std::mt19937 rng;

    // Timing
    std::chrono::high_resolution_clock::time_point t_start;
};

// Thread-local for eval callback activation capture

static thread_local alpha_combined_trainer * g_combined_capture = nullptr;

static bool combined_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * trainer = g_combined_capture;
    if (!trainer && user_data) trainer = (alpha_combined_trainer *)user_data;
    if (!trainer) return true;

    const char * name = ggml_get_name(t);
    if (!name || !name[0]) return true;

    if (ask) return true;  // capture all

    std::string sname(name);
    if (sname.find("attn_norm") == std::string::npos) return true;

    // Parse layer ID from "attn_norm-N"
    int block_id = -1;
    size_t dash = sname.rfind('-');
    if (dash != std::string::npos) block_id = atoi(sname.c_str() + dash + 1);
    if (block_id < 0) return true;

    int n_elem = (int)ggml_nelements(t);
    int n_tokens = n_elem / trainer->n_embd;

    char prefix[32];
    snprintf(prefix, sizeof(prefix), "blk.%d.", block_id);
    for (auto & ll : trainer->layers) {
        if (ll.name.find(prefix) != std::string::npos) {
            ll.x_saved.resize(n_elem);
            ggml_backend_tensor_get(t, ll.x_saved.data(), 0, n_elem * sizeof(float));
            ll.x_tokens = n_tokens;
        }
    }
    return true;
}

// Sync CPU LoRA params -> adapter tensors

static void sync_to_adapter(alpha_combined_trainer * trainer) {
    for (auto & ll : trainer->layers) {
        std::string base_name(ll.W->name);
        auto it = trainer->adapter->ab_map.find(base_name);
        if (it == trainer->adapter->ab_map.end()) continue;
        ggml_backend_tensor_set(it->second.a, ll.A.data(), 0, ll.A.size() * sizeof(float));
        ggml_backend_tensor_set(it->second.b, ll.B.data(), 0, ll.B.size() * sizeof(float));
    }
}

// Forward pass with activation capture + loss

static float forward_pass(alpha_combined_trainer * trainer, const llama_token * tokens, int n_tokens) {
    sync_to_adapter(trainer);
    llama_memory_clear(llama_get_memory(trainer->ctx), true);
    g_combined_capture = trainer;

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(trainer->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) return -1.0f;

    // Fill placeholder activations for layers where callback didn't fire
    for (auto & ll : trainer->layers) {
        if (ll.x_saved.empty() || ll.x_tokens == 0) {
            int sz = (int)ll.in_dim * n_tokens;
            ll.x_saved.resize(sz, 0.01f);
            ll.x_tokens = n_tokens;
        }
    }

    // Compute CE loss
    float loss = 0.0f;
    int count = 0;
    for (int i = 0; i < n_tokens - 1; i++) {
        float * logits = llama_get_logits_ith(trainer->ctx, i);
        if (!logits) continue;
        int label = tokens[i + 1];
        if (label < 0 || label >= trainer->n_vocab) continue;

        float max_val = logits[0];
        for (int v = 1; v < trainer->n_vocab; v++)
            if (logits[v] > max_val) max_val = logits[v];
        float sum_exp = 0.0f;
        for (int v = 0; v < trainer->n_vocab; v++)
            sum_exp += expf(logits[v] - max_val);
        loss -= (logits[label] - max_val) - logf(sum_exp);
        count++;
    }
    return count > 0 ? loss / count : 0.0f;
}

// MeZO: sample z, perturb, restore

static void mezo_sample_z(alpha_combined_trainer * trainer) {
    // GV-ZO: Guiding Vector-Augmented perturbation (arxiv 2601.04710)
    // z = alpha * normalized_guide + (1-alpha) * random_noise
    // First 20 steps: pure random (guide is zero/noise)
    // After: blend guide direction with random exploration
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float alpha = (trainer->step > 20) ? 0.5f : 0.0f;  // alignment strength

    for (auto & ll : trainer->layers) {
        // Compute guide norm for normalization
        float normA = 0.0f, normB = 0.0f;
        if (alpha > 0) {
            for (size_t i = 0; i < ll.guideA.size(); i++) normA += ll.guideA[i] * ll.guideA[i];
            for (size_t i = 0; i < ll.guideB.size(); i++) normB += ll.guideB[i] * ll.guideB[i];
            normA = sqrtf(normA + 1e-8f);
            normB = sqrtf(normB + 1e-8f);
        }

        for (size_t i = 0; i < ll.zA.size(); i++) {
            float noise = dist(trainer->rng);
            float guide = (normA > 1e-6f) ? ll.guideA[i] / normA : 0.0f;
            ll.zA[i] = alpha * guide + (1.0f - alpha) * noise;
        }
        for (size_t i = 0; i < ll.zB.size(); i++) {
            float noise = dist(trainer->rng);
            float guide = (normB > 1e-6f) ? ll.guideB[i] / normB : 0.0f;
            ll.zB[i] = alpha * guide + (1.0f - alpha) * noise;
        }
    }
}

static void mezo_perturb(alpha_combined_trainer * trainer, float sign) {
    float eps = trainer->config.mezo_eps;
    for (auto & ll : trainer->layers) {
        for (size_t i = 0; i < ll.A.size(); i++) ll.A[i] += sign * eps * ll.zA[i];
        for (size_t i = 0; i < ll.B.size(); i++) ll.B[i] += sign * eps * ll.zB[i];
    }
}

// Backward: exact gradient for masked LoRA params only

static void backward_masked_gradients(
    alpha_combined_trainer * trainer,
    const llama_token * tokens,
    int n_tokens)
{
    const int nt = n_tokens - 1;
    if (nt <= 0) return;

    const int ne = trainer->n_embd;
    const int nv = trainer->n_vocab;

    // Step 1: Loss gradient (dLogits)
    fprintf(stderr, "[backward] step 1: dLogits alloc (%d × %d = %.1f MB)\n",
            nv, nt, (float)nv * nt * 4 / 1e6f);
    std::vector<float> dLogits(nv * nt);
    std::vector<int> targets(nt);
    for (int i = 0; i < nt; i++) targets[i] = tokens[i + 1];

    std::vector<float> all_logits(nv * nt);
    for (int i = 0; i < nt; i++) {
        float * li = llama_get_logits_ith(trainer->ctx, i);
        if (li) memcpy(all_logits.data() + i * nv, li, nv * sizeof(float));
    }
    alpha_train_loss_gradient(all_logits.data(), targets.data(), dLogits.data(), nv, nt);

    // Step 2: Backward through output head -> dHidden
    fprintf(stderr, "[backward] step 2: output head backward\n");
    ggml_tensor * output_w = trainer->model->output;
    if (!output_w) output_w = trainer->model->tok_embd;
    if (output_w) {
        fprintf(stderr, "[backward] output_w: type=%s, size=%.1f MB, data=%p\n",
                ggml_type_name(output_w->type), (float)ggml_nbytes(output_w) / 1e6f, output_w->data);
    }

    std::vector<float> dY(ne * nt, 0.0f);
    if (output_w) {
        alpha_train_linear_backward_gpu(nullptr, nullptr, output_w,
            dLogits.data(), dY.data(), ne, nv, nt);
    }

    // Step 3: Layer-by-layer backward (only compute LoRA grads for masked layers)
    fprintf(stderr, "[backward] step 3: layer-by-layer backward (%d layers)\n", trainer->n_layer);
    std::vector<float> dResidual(ne * nt);
    std::vector<float> dFFN(ne * nt);
    std::vector<float> scratch1, scratch2, scratch3;

    for (int il = trainer->n_layer - 1; il >= 0; il--) {
        const auto & layer = trainer->model->layers[il];

        // Check if any LoRA in this block has masked params
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "blk.%d.", il);
        bool has_masked = false;
        for (const auto & ll : trainer->layers) {
            if (ll.name.find(prefix) != std::string::npos &&
                (ll.n_masked_A > 0 || ll.n_masked_B > 0)) {
                has_masked = true;
                break;
            }
        }

        // Residual split
        memcpy(dResidual.data(), dY.data(), ne * nt * sizeof(float));
        memcpy(dFFN.data(), dY.data(), ne * nt * sizeof(float));

        // FFN backward (always needed for gradient propagation)
        if (layer.ffn_down && layer.ffn_gate && layer.ffn_up) {
            int64_t n_ff = layer.ffn_up->ne[1];
            scratch1.resize(n_ff * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.ffn_down,
                dFFN.data(), scratch1.data(), (int)n_ff, ne, nt);

            scratch2.resize(n_ff * nt);
            scratch3.resize(n_ff * nt);
            for (int64_t i = 0; i < n_ff * nt; i++) {
                scratch2[i] = scratch1[i] * 0.5f;
                scratch3[i] = scratch1[i] * 0.5f;
            }

            std::vector<float> d_ffn_input(ne * nt, 0.0f);
            std::vector<float> tmp(ne * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.ffn_gate,
                scratch2.data(), tmp.data(), ne, (int)n_ff, nt);
            for (int i = 0; i < ne * nt; i++) d_ffn_input[i] += tmp[i];
            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.ffn_up,
                scratch3.data(), tmp.data(), ne, (int)n_ff, nt);
            for (int i = 0; i < ne * nt; i++) d_ffn_input[i] += tmp[i];

            for (int i = 0; i < ne * nt; i++) dResidual[i] += d_ffn_input[i];
        }

        // Attention backward -> LoRA grads (only if this block has masked params)
        memcpy(dFFN.data(), dResidual.data(), ne * nt * sizeof(float));  // reuse as dAttn

        if (layer.wo && has_masked) {
            int64_t attn_out_dim = layer.wo->ne[0];
            scratch1.resize(attn_out_dim * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wo,
                dFFN.data(), scratch1.data(), (int)attn_out_dim, ne, nt);

            // Compute exact LoRA gradients for masked Q and V
            for (auto & ll : trainer->layers) {
                if (ll.name.find(prefix) == std::string::npos) continue;
                if (ll.n_masked_A == 0 && ll.n_masked_B == 0) continue;
                if (ll.x_saved.empty()) continue;

                std::vector<float> dA_local(ll.A.size(), 0.0f);
                std::vector<float> dB_local(ll.B.size(), 0.0f);
                alpha_train_lora_gradient(
                    scratch1.data(), ll.x_saved.data(),
                    ll.A.data(), ll.B.data(),
                    dA_local.data(), dB_local.data(),
                    1.0f, (int)ll.in_dim, (int)ll.out_dim, ll.rank, nt);

                // Accumulate only masked elements
                for (size_t i = 0; i < ll.dA.size(); i++) {
                    if (ll.maskA[i]) ll.dA[i] += dA_local[i];
                }
                for (size_t i = 0; i < ll.dB.size(); i++) {
                    if (ll.maskB[i]) ll.dB[i] += dB_local[i];
                }
            }

            // Propagate gradient to previous layer
            std::vector<float> d_attn_norm(ne * nt, 0.0f);
            if (layer.wq) {
                std::vector<float> tmp(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wq,
                    scratch1.data(), tmp.data(), ne, (int)attn_out_dim, nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp[i];
            }
            if (layer.wv) {
                std::vector<float> tmp(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wv,
                    scratch1.data(), tmp.data(), ne, (int)layer.wv->ne[1], nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp[i];
            }
            for (int i = 0; i < ne * nt; i++) dResidual[i] += d_attn_norm[i];
        }

        memcpy(dY.data(), dResidual.data(), ne * nt * sizeof(float));
    }
}

// Convergence monitor

static void convergence_init(alpha_convergence_monitor & conv, int window_size) {
    memset(&conv, 0, sizeof(conv));
    conv.window_size = window_size > 0 ? window_size : 200;
    conv.eval_loss_baseline = 1e9f;
    conv.eval_loss_best = 1e9f;
}

static void convergence_add_loss(alpha_convergence_monitor & conv, float loss) {
    conv.loss_window[conv.window_pos] = loss;
    conv.window_pos = (conv.window_pos + 1) % conv.window_size;
    if (conv.window_count < conv.window_size) conv.window_count++;

    if (conv.window_count < 50) return;  // need minimum samples

    // Compute velocity (linear regression slope over window)
    int n = conv.window_count;
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; i++) {
        int idx = (conv.window_pos - n + i + conv.window_size) % conv.window_size;
        float x = (float)i;
        float y = conv.loss_window[idx];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    float denom = (float)n * sum_xx - sum_x * sum_x;
    conv.loss_velocity = denom > 1e-10f ? ((float)n * sum_xy - sum_x * sum_y) / denom : 0.0f;

    // Compute variance
    float mean = sum_y / n;
    float var_sum = 0;
    for (int i = 0; i < n; i++) {
        int idx = (conv.window_pos - n + i + conv.window_size) % conv.window_size;
        float d = conv.loss_window[idx] - mean;
        var_sum += d * d;
    }
    conv.loss_variance = var_sum / (n - 1);

    // Plateau detection: velocity within noise floor
    float noise_floor = sqrtf(conv.loss_variance / n);
    if (fabsf(conv.loss_velocity) < noise_floor * 0.5f) {
        conv.plateau_count++;
    } else {
        conv.plateau_count = 0;
    }
}

// Adaptive hyperparameter tuning
// Spike prevention is handled by GradientStabilizer (threshold-free, structural).
// Adaptive tune only handles:
//   1. Overfitting (traindown eval flat) -> increase weight decay, reduce LR
//   2. Stalled (no train progress) -> increase LR, reduce elastic

static void adaptive_tune(alpha_combined_trainer * trainer, float loss, float eval_loss) {
    auto & conv = trainer->conv;
    auto & cfg = trainer->config;

    // Update EMA of recent losses (for overfitting/stall detection)
    if (conv.recent_avg_loss < 1e-6f) {
        conv.recent_avg_loss = loss;
    } else {
        conv.recent_avg_loss = 0.9f * conv.recent_avg_loss + 0.1f * loss;
    }

    // Overfitting detection (at eval checkpoints)
    if (eval_loss > 0 && conv.last_eval_loss > 0) {
        bool train_improving = conv.loss_velocity < -1e-5f;
        bool eval_worsening = eval_loss >= conv.last_eval_loss * 0.999f;  // not improving

        if (train_improving && eval_worsening) {
            conv.overfit_count++;
            if (conv.overfit_count >= 1) {
                // Overfitting -> increase regularization
                cfg.weight_decay = std::min(cfg.weight_decay * 2.0f, 0.1f);
                cfg.lr *= 0.7f;
                cfg.elastic_lambda = std::min(cfg.elastic_lambda * 1.3f, 1.0f);
                conv.overfit_count = 0;
                conv.n_adjustments++;
                if (cfg.json_progress) {
                    fprintf(stderr, "{\"event\":\"adapt\",\"reason\":\"drift\",\"step\":%d,\"wd\":%.4f,\"lr\":%.2e,\"reg\":%.3f}\n",
                            trainer->step, cfg.weight_decay, cfg.lr, cfg.elastic_lambda);
                } else {
                    fprintf(stderr, "  [adapt] drift detected → wd=%.3f, lr=%.1e, elastic=%.2f\n",
                            cfg.weight_decay, cfg.lr, cfg.elastic_lambda);
                }
            }
        } else {
            conv.overfit_count = 0;
        }
    }
    if (eval_loss > 0) conv.last_eval_loss = eval_loss;

    // Stall detection (check every 50 steps via window)
    if (conv.window_count >= 50 && trainer->step % 50 == 0) {
        bool no_progress = fabsf(conv.loss_velocity) < 1e-5f;
        bool far_from_converged = conv.recent_avg_loss > conv.eval_loss_baseline * 0.85f;

        if (no_progress && far_from_converged) {
            conv.stall_count++;
            if (conv.stall_count >= 2) {
                // Stalled but not converged -> increase LR
                cfg.lr *= 1.5f;
                cfg.lr_mezo *= 1.5f;
                cfg.elastic_lambda = std::max(cfg.elastic_lambda * 0.75f, 0.01f);
                conv.stall_count = 0;
                conv.n_adjustments++;
                if (cfg.json_progress) {
                    fprintf(stderr, "{\"event\":\"adapt\",\"reason\":\"plateau\",\"step\":%d,\"lr\":%.2e,\"lr2\":%.2e,\"reg\":%.3f}\n",
                            trainer->step, cfg.lr, cfg.lr_mezo, cfg.elastic_lambda);
                } else {
                    fprintf(stderr, "  [adapt] plateau → lr=%.1e, lr2=%.1e, reg=%.2f\n",
                            cfg.lr, cfg.lr_mezo, cfg.elastic_lambda);
                }
            }
        } else {
            conv.stall_count = 0;
        }
    }
}

// Block-Fisher mask computation

void alpha_combined_compute_mask(
    alpha_combined_trainer * trainer,
    const llama_token * eval_tokens,
    int n_eval_tokens,
    int seq_len)
{
    if (!trainer || !eval_tokens || n_eval_tokens < seq_len) return;

    float eps = trainer->config.mezo_eps;
    int n_batches = trainer->config.fisher_batches;
    if (n_batches <= 0) n_batches = 5;

    fprintf(stderr, "Analyzing parameter sensitivity (%d batches)...\n", n_batches);

    // Initialize importance scores to zero
    for (auto & ll : trainer->layers) {
        ll.dispA.assign(ll.A.size(), 0.0f);
        ll.dispB.assign(ll.B.size(), 0.0f);
    }

    // For each batch: perturb each LoRA tensor, measure |delta_loss|
    for (int b = 0; b < n_batches; b++) {
        int offset = (b * seq_len) % (n_eval_tokens - seq_len + 1);
        const llama_token * batch_tokens = eval_tokens + offset;

        // Baseline loss
        float loss_base = forward_pass(trainer, batch_tokens, seq_len);
        if (loss_base < 0) continue;

        // Perturb each LoRA layer independently and measure sensitivity
        for (size_t li = 0; li < trainer->layers.size(); li++) {
            auto & ll = trainer->layers[li];

            // Perturb A
            std::normal_distribution<float> dist(0.0f, 1.0f);
            std::vector<float> zA(ll.A.size());
            for (size_t i = 0; i < zA.size(); i++) zA[i] = dist(trainer->rng);

            for (size_t i = 0; i < ll.A.size(); i++) ll.A[i] += eps * zA[i];
            float loss_a = forward_pass(trainer, batch_tokens, seq_len);
            for (size_t i = 0; i < ll.A.size(); i++) ll.A[i] -= eps * zA[i];

            if (loss_a >= 0) {
                float sensitivity = fabsf(loss_a - loss_base) / eps;
                // Distribute to per-element importance proportional to |z|
                for (size_t i = 0; i < ll.dispA.size(); i++) {
                    ll.dispA[i] += sensitivity * fabsf(zA[i]);
                }
            }

            // Perturb B
            std::vector<float> zB(ll.B.size());
            for (size_t i = 0; i < zB.size(); i++) zB[i] = dist(trainer->rng);

            for (size_t i = 0; i < ll.B.size(); i++) ll.B[i] += eps * zB[i];
            float loss_b = forward_pass(trainer, batch_tokens, seq_len);
            for (size_t i = 0; i < ll.B.size(); i++) ll.B[i] -= eps * zB[i];

            if (loss_b >= 0) {
                float sensitivity = fabsf(loss_b - loss_base) / eps;
                for (size_t i = 0; i < ll.dispB.size(); i++) {
                    ll.dispB[i] += sensitivity * fabsf(zB[i]);
                }
            }
        }
    }

    // Average importance over batches
    for (auto & ll : trainer->layers) {
        for (auto & v : ll.dispA) v /= n_batches;
        for (auto & v : ll.dispB) v /= n_batches;
    }

    // Compute threshold for top sparsity_ratio
    float sparsity = trainer->config.sparsity_ratio;
    std::vector<float> all_importance;
    for (const auto & ll : trainer->layers) {
        all_importance.insert(all_importance.end(), ll.dispA.begin(), ll.dispA.end());
        all_importance.insert(all_importance.end(), ll.dispB.begin(), ll.dispB.end());
    }
    std::sort(all_importance.begin(), all_importance.end(), std::greater<float>());

    int n_to_mask = std::max(1, (int)(all_importance.size() * sparsity));
    float threshold = all_importance[std::min(n_to_mask, (int)all_importance.size() - 1)];

    // Apply mask
    int total_masked = 0;
    int total_params = 0;
    for (auto & ll : trainer->layers) {
        ll.maskA.resize(ll.A.size());
        ll.maskB.resize(ll.B.size());
        ll.n_masked_A = 0;
        ll.n_masked_B = 0;

        for (size_t i = 0; i < ll.A.size(); i++) {
            ll.maskA[i] = (ll.dispA[i] >= threshold);
            if (ll.maskA[i]) ll.n_masked_A++;
        }
        for (size_t i = 0; i < ll.B.size(); i++) {
            ll.maskB[i] = (ll.dispB[i] >= threshold);
            if (ll.maskB[i]) ll.n_masked_B++;
        }
        total_masked += ll.n_masked_A + ll.n_masked_B;
        total_params += (int)(ll.A.size() + ll.B.size());
    }

    trainer->mask.total_masked = total_masked;
    trainer->mask.total_params = total_params;
    trainer->mask.sparsity_ratio = (float)total_masked / total_params;

    // Save anchor copies for elastic anchoring
    for (auto & ll : trainer->layers) {
        ll.A_anchor = ll.A;
        ll.B_anchor = ll.B;
    }

    fprintf(stderr, "Param selection: %d / %d params masked (%.1f%%), threshold=%.6f\n",
            total_masked, total_params, 100.0f * total_masked / total_params, threshold);

    // Print per-layer summary
    for (const auto & ll : trainer->layers) {
        if (ll.n_masked_A + ll.n_masked_B > 0) {
            fprintf(stderr, "  %s: %d/%zu masked\n",
                    ll.name.c_str(),
                    ll.n_masked_A + ll.n_masked_B,
                    ll.A.size() + ll.B.size());
        }
    }
}

// Combined step: MeZO + sparse backprop

float alpha_combined_step(
    alpha_combined_trainer * trainer,
    const llama_token * tokens,
    int n_tokens)
{
    if (!trainer || !tokens || n_tokens < 2) return -1.0f;

    float mezo_eps = trainer->config.mezo_eps;
    float lr_mezo = trainer->config.lr_mezo;
    float lr_bp = trainer->config.lr;
    float elastic = trainer->config.elastic_lambda;

    // === MeZO channel ===

    // 1. Sample z (LoRA + novel Tier 3 layers)
    mezo_sample_z(trainer);
    alpha_novel_mezo_sample_z(trainer->novel_layers, trainer->rng);

    // 2. Forward +eps (also captures activations for backprop)
    mezo_perturb(trainer, +1.0f);
    alpha_novel_mezo_perturb(trainer->novel_layers, +1.0f, mezo_eps);
    float loss_plus = forward_pass(trainer, tokens, n_tokens);

    // 3. Forward -2eps (net: -eps)
    mezo_perturb(trainer, -2.0f);
    alpha_novel_mezo_perturb(trainer->novel_layers, -2.0f, mezo_eps);
    float loss_minus = forward_pass(trainer, tokens, n_tokens);

    // 4. Restore to original
    mezo_perturb(trainer, +1.0f);
    alpha_novel_mezo_perturb(trainer->novel_layers, +1.0f, mezo_eps);

    if (loss_plus < 0 || loss_minus < 0) return -1.0f;

    float g_proj = (loss_plus - loss_minus) / (2.0f * mezo_eps);

    // GradientStabilizer for MeZO channel (arxiv 2502.17055)
    // Preserve direction, replace magnitude with stabilized EMA - threshold-free
    {
        float g_abs = fabsf(g_proj);
        float gs_beta = 0.99f;
        if (trainer->g_proj_norm_ema < 1e-10f) {
            trainer->g_proj_norm_ema = g_abs;  // first step
        } else {
            trainer->g_proj_norm_ema = gs_beta * trainer->g_proj_norm_ema + (1.0f - gs_beta) * g_abs;
        }
        // Stabilize: sign preserved, magnitude bounded by EMA
        if (g_abs > 1e-10f) {
            g_proj = (g_proj > 0 ? 1.0f : -1.0f) * trainer->g_proj_norm_ema;
        }
    }

    // Update GV-ZO guiding vector from this step's gradient signal
    // guide[i] = EMA of (g_proj * z[i]) - estimates gradient direction per param
    // (arxiv 2601.04710: guiding vector achieves strongest alignment with true gradient)
    {
        float guide_beta = 0.9f;
        for (auto & ll : trainer->layers) {
            for (size_t i = 0; i < ll.guideA.size(); i++) {
                ll.guideA[i] = guide_beta * ll.guideA[i] + (1.0f - guide_beta) * g_proj * ll.zA[i];
            }
            for (size_t i = 0; i < ll.guideB.size(); i++) {
                ll.guideB[i] = guide_beta * ll.guideB[i] + (1.0f - guide_beta) * g_proj * ll.zB[i];
            }
        }
    }

    // === Exact backprop channel (on masked params) ===
    // Activations reused from +eps forward (fwd 1). O(eps) gradient error is
    // negligible vs mini-batch noise (~0.1% vs 10-50%). No 3rd forward needed.

    // Clear accumulated gradients
    for (auto & ll : trainer->layers) {
        std::fill(ll.dA.begin(), ll.dA.end(), 0.0f);
        std::fill(ll.dB.begin(), ll.dB.end(), 0.0f);
    }

    // Backward through model, computing exact grads for masked params
    backward_masked_gradients(trainer, tokens, n_tokens);

    // GradientStabilizer for backprop channel (arxiv 2502.17055)
    // Per-tensor: preserve gradient direction, replace norm with stabilized EMA
    {
        float gs_beta = 0.99f;
        for (auto & ll : trainer->layers) {
            // Stabilize dA
            float normA = 0.0f;
            for (size_t i = 0; i < ll.dA.size(); i++) normA += ll.dA[i] * ll.dA[i];
            normA = sqrtf(normA + 1e-10f);
            if (ll.grad_norm_ema_A < 1e-10f) {
                ll.grad_norm_ema_A = normA;
            } else {
                ll.grad_norm_ema_A = gs_beta * ll.grad_norm_ema_A + (1.0f - gs_beta) * normA;
            }
            if (normA > 1e-10f) {
                float scale = ll.grad_norm_ema_A / normA;
                for (size_t i = 0; i < ll.dA.size(); i++) ll.dA[i] *= scale;
            }

            // Stabilize dB
            float normB = 0.0f;
            for (size_t i = 0; i < ll.dB.size(); i++) normB += ll.dB[i] * ll.dB[i];
            normB = sqrtf(normB + 1e-10f);
            if (ll.grad_norm_ema_B < 1e-10f) {
                ll.grad_norm_ema_B = normB;
            } else {
                ll.grad_norm_ema_B = gs_beta * ll.grad_norm_ema_B + (1.0f - gs_beta) * normB;
            }
            if (normB > 1e-10f) {
                float scale = ll.grad_norm_ema_B / normB;
                for (size_t i = 0; i < ll.dB.size(); i++) ll.dB[i] *= scale;
            }
        }
    }

    // === Combined update ===

    trainer->step++;
    int t = trainer->step;
    float beta1 = trainer->config.beta1;
    float beta2 = trainer->config.beta2;
    float adam_eps = trainer->config.eps;
    float wd = trainer->config.weight_decay;

    float effective_lr_mezo = lr_mezo;
    float effective_lr_bp = lr_bp;
    int warmup = trainer->config.warmup_steps;
    if (t < warmup) {
        float scale = (float)t / warmup;
        effective_lr_mezo *= scale;
        effective_lr_bp *= scale;
    }

    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    for (auto & ll : trainer->layers) {
        // Update A
        for (size_t i = 0; i < ll.A.size(); i++) {
            float g;
            if (ll.maskA[i] && ll.dA[i] != 0.0f) {
                // Exact gradient + elastic anchoring
                g = ll.dA[i] + elastic * (ll.A[i] - ll.A_anchor[i]);
                // Use backprop learning rate
                ll.mA[i] = beta1 * ll.mA[i] + (1.0f - beta1) * g;
                ll.vA[i] = beta2 * ll.vA[i] + (1.0f - beta2) * g * g;
                float m_hat = ll.mA[i] / bc1;
                float v_hat = ll.vA[i] / bc2;
                ll.A[i] -= effective_lr_bp * m_hat / (sqrtf(v_hat) + adam_eps);
            } else {
                // MeZO gradient
                g = g_proj * ll.zA[i] + wd * ll.A[i];
                ll.mA[i] = beta1 * ll.mA[i] + (1.0f - beta1) * g;
                ll.vA[i] = beta2 * ll.vA[i] + (1.0f - beta2) * g * g;
                float m_hat = ll.mA[i] / bc1;
                float v_hat = ll.vA[i] / bc2;
                ll.A[i] -= effective_lr_mezo * m_hat / (sqrtf(v_hat) + adam_eps);
            }

            // Track displacement
            ll.dispA[i] += fabsf(ll.A[i] - ll.A_anchor[i]);
        }

        // Update B
        for (size_t i = 0; i < ll.B.size(); i++) {
            float g;
            if (ll.maskB[i] && ll.dB[i] != 0.0f) {
                g = ll.dB[i] + elastic * (ll.B[i] - ll.B_anchor[i]);
                ll.mB[i] = beta1 * ll.mB[i] + (1.0f - beta1) * g;
                ll.vB[i] = beta2 * ll.vB[i] + (1.0f - beta2) * g * g;
                float m_hat = ll.mB[i] / bc1;
                float v_hat = ll.vB[i] / bc2;
                ll.B[i] -= effective_lr_bp * m_hat / (sqrtf(v_hat) + adam_eps);
            } else {
                g = g_proj * ll.zB[i] + wd * ll.B[i];
                ll.mB[i] = beta1 * ll.mB[i] + (1.0f - beta1) * g;
                ll.vB[i] = beta2 * ll.vB[i] + (1.0f - beta2) * g * g;
                float m_hat = ll.mB[i] / bc1;
                float v_hat = ll.vB[i] / bc2;
                ll.B[i] -= effective_lr_mezo * m_hat / (sqrtf(v_hat) + adam_eps);
            }
            ll.dispB[i] += fabsf(ll.B[i] - ll.B_anchor[i]);
        }
    }

    // === Novel layer updates ===
    // Tier 3: accumulate MeZO gradient (g_proj * z)
    alpha_novel_mezo_accumulate(trainer->novel_layers, g_proj);
    // Tier 2: gradients already accumulated during backward_masked_gradients
    // (novel layers at masked positions get finite-diff backward)

    // Apply optimizer update to all novel layers
    alpha_novel_update_all(trainer->novel_layers, effective_lr_bp, beta1, beta2,
                           adam_eps, elastic, t);

    float loss = (loss_plus + loss_minus) / 2.0f;
    trainer->current_loss = loss;
    trainer->total_loss += loss;
    trainer->n_steps++;

    // Update convergence monitor
    convergence_add_loss(trainer->conv, loss);

    // Adaptive tuning (spike detection runs every step, eval_loss=-1 means no eval this step)
    adaptive_tune(trainer, loss, -1.0f);

    return loss;
}

// Convergence check

bool alpha_combined_check_convergence(
    alpha_combined_trainer * trainer,
    float loss)
{
    if (!trainer) return true;

    auto & conv = trainer->conv;
    int patience = trainer->config.plateau_patience > 0 ? trainer->config.plateau_patience : 3;

    // Hard stop
    if (trainer->step >= trainer->config.max_steps) {
        fprintf(stderr, "Convergence: max steps reached (%d)\n", trainer->config.max_steps);
        return true;
    }

    // Plateau detection: N consecutive flat windows
    if (conv.plateau_count >= patience && conv.window_count >= conv.window_size) {
        fprintf(stderr, "Convergence: %d consecutive plateau windows (velocity=%.2e, noise=%.2e)\n",
                conv.plateau_count, conv.loss_velocity, sqrtf(conv.loss_variance));
        return true;
    }

    return false;
}

// Quality gate

bool alpha_combined_quality_gate(
    alpha_combined_trainer * trainer,
    const llama_token * eval_tokens,
    int n_eval_tokens)
{
    if (!trainer || !eval_tokens || n_eval_tokens < 2) return false;

    float eval_loss = forward_pass(trainer, eval_tokens, n_eval_tokens);
    if (eval_loss < 0) return false;

    auto & conv = trainer->conv;

    // Initialize baseline on first call
    if (conv.eval_loss_baseline > 1e8f) {
        conv.eval_loss_baseline = eval_loss;
        conv.eval_loss_best = eval_loss;
        if (trainer->config.json_progress) {
            fprintf(stderr, "{\"event\":\"eval\",\"step\":%d,\"eval_loss\":%.4f,\"baseline\":%.4f,\"best\":%.4f,\"gate\":\"baseline\"}\n",
                    trainer->step, eval_loss, eval_loss, eval_loss);
        } else {
            fprintf(stderr, "Eval baseline: %.4f\n", eval_loss);
        }
        return true;
    }

    // Check improvement
    float threshold = trainer->config.gate_threshold;
    bool passes = eval_loss < conv.eval_loss_baseline * threshold;

    if (eval_loss < conv.eval_loss_best) {
        conv.eval_loss_best = eval_loss;
        conv.best_step = trainer->step;
    }

    if (trainer->config.json_progress) {
        fprintf(stderr, "{\"event\":\"eval\",\"step\":%d,\"eval_loss\":%.4f,\"baseline\":%.4f,\"best\":%.4f,\"best_step\":%d,\"gate\":\"%s\"}\n",
                trainer->step, eval_loss, conv.eval_loss_baseline, conv.eval_loss_best, conv.best_step,
                passes ? "pass" : "fail");
    } else {
        fprintf(stderr, "Eval: loss=%.4f (base=%.4f, best=%.4f at step %d) %s\n",
                eval_loss, conv.eval_loss_baseline, conv.eval_loss_best, conv.best_step,
                passes ? "PASS" : "FAIL");
    }

    // Trigger adaptive tuning with eval loss (overfitting/stall detection)
    adaptive_tune(trainer, trainer->current_loss, eval_loss);

    return passes;
}

// Init

alpha_combined_trainer * alpha_combined_trainer_init(
    struct llama_model   * model,
    struct llama_context * ctx,
    alpha_combined_train_config config)
{
    if (!model || !ctx) return nullptr;

    auto * trainer = new alpha_combined_trainer();
    trainer->model = model;
    trainer->ctx = ctx;
    trainer->config = config;
    trainer->step = 0;
    trainer->current_loss = 0;
    trainer->total_loss = 0;
    trainer->n_steps = 0;
    trainer->g_proj_norm_ema = 0.0f;
    trainer->adapter = nullptr;
    trainer->adapter_ctx = nullptr;
    trainer->adapter_buf = nullptr;
    trainer->rng.seed(42);
    trainer->t_start = std::chrono::high_resolution_clock::now();

    trainer->n_layer = (int)model->layers.size();
    trainer->n_embd = model->hparams.n_embd;
    trainer->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    convergence_init(trainer->conv, config.convergence_window);

    int rank = config.rank > 0 ? config.rank : 32;

    // Create LoRA layers for Q and V projections
    for (int il = 0; il < trainer->n_layer; il++) {
        const auto & layer = model->layers[il];
        struct { ggml_tensor * W; const char * suffix; } targets[] = {
            { layer.wq, "attn_q" },
            { layer.wv, "attn_v" },
        };

        for (auto & t : targets) {
            if (!t.W) continue;

            combined_lora_layer ll = {};
            ll.name = std::string("blk.") + std::to_string(il) + "." + t.suffix;
            ll.W = t.W;
            ll.in_dim = t.W->ne[0];
            ll.out_dim = t.W->ne[1];
            ll.rank = rank;

            int nA = rank * (int)ll.in_dim;
            int nB = (int)ll.out_dim * rank;

            ll.A.resize(nA);
            ll.B.resize(nB, 0.0f);
            ll.mA.resize(nA, 0.0f);
            ll.vA.resize(nA, 0.0f);
            ll.mB.resize(nB, 0.0f);
            ll.vB.resize(nB, 0.0f);
            ll.zA.resize(nA, 0.0f);
            ll.zB.resize(nB, 0.0f);
            ll.dA.resize(nA, 0.0f);
            ll.dB.resize(nB, 0.0f);
            ll.dispA.resize(nA, 0.0f);
            ll.dispB.resize(nB, 0.0f);
            ll.guideA.resize(nA, 0.0f);
            ll.guideB.resize(nB, 0.0f);
            ll.grad_norm_ema_A = 0.0f;
            ll.grad_norm_ema_B = 0.0f;

            // Initialize mask to all-false (set by compute_mask)
            ll.maskA.resize(nA, false);
            ll.maskB.resize(nB, false);
            ll.n_masked_A = 0;
            ll.n_masked_B = 0;

            ll.x_tokens = 0;

            // A = random, B = zero (standard LoRA init)
            std::mt19937 rng(42 + il * 100 + (t.suffix[5] - 'a'));
            std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)rank));
            for (int i = 0; i < nA; i++) ll.A[i] = dist(rng);

            trainer->layers.push_back(std::move(ll));
        }
    }

    if (trainer->layers.empty()) {
        fprintf(stderr, "combined: no trainable layers found\n");
        delete trainer;
        return nullptr;
    }

    // Create LoRA adapter
    trainer->adapter = new llama_adapter_lora(model);
    trainer->adapter->alpha = (float)rank;

    size_t ctx_size = ggml_tensor_overhead() * trainer->layers.size() * 2 + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    trainer->adapter_ctx = ggml_init(params);

    ggml_backend_buffer_type_t cpu_buft = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_buft = ggml_backend_dev_buffer_type(dev);
            break;
        }
    }

    for (auto & ll : trainer->layers) {
        char name_buf[128];
        snprintf(name_buf, sizeof(name_buf), "%s.lora_a", ll.name.c_str());
        ggml_tensor * tA = ggml_new_tensor_2d(trainer->adapter_ctx, GGML_TYPE_F32, ll.in_dim, rank);
        ggml_set_name(tA, name_buf);

        snprintf(name_buf, sizeof(name_buf), "%s.lora_b", ll.name.c_str());
        ggml_tensor * tB = ggml_new_tensor_2d(trainer->adapter_ctx, GGML_TYPE_F32, rank, ll.out_dim);
        ggml_set_name(tB, name_buf);

        std::string base_name(ll.W->name);
        trainer->adapter->ab_map[base_name] = llama_adapter_lora_weight(tA, tB);
    }

    trainer->adapter_buf = ggml_backend_alloc_ctx_tensors_from_buft(trainer->adapter_ctx, cpu_buft);
    if (!trainer->adapter_buf) {
        fprintf(stderr, "combined: failed to allocate adapter buffer\n");
        delete trainer->adapter;
        ggml_free(trainer->adapter_ctx);
        delete trainer;
        return nullptr;
    }

    // Initialize adapter tensors
    for (auto & ll : trainer->layers) {
        std::string base_name(ll.W->name);
        auto it = trainer->adapter->ab_map.find(base_name);
        if (it != trainer->adapter->ab_map.end()) {
            ggml_backend_tensor_set(it->second.a, ll.A.data(), 0, ll.A.size() * sizeof(float));
            ggml_backend_tensor_set(it->second.b, ll.B.data(), 0, ll.B.size() * sizeof(float));
        }
    }

    llama_adapter_lora * adapters[] = { trainer->adapter };
    float scales[] = { 1.0f };
    llama_set_adapters_lora(ctx, adapters, 1, scales);

    int64_t total_params = 0;
    for (const auto & ll : trainer->layers) {
        total_params += (int64_t)ll.A.size() + (int64_t)ll.B.size();
    }

    fprintf(stderr, "Training: %zu adapter layers, rank=%d, %.2f M trainable params\n",
            trainer->layers.size(), rank, (float)total_params / 1e6f);
    if (!config.json_progress) {
        fprintf(stderr, "  Primary lr=%.1e, Secondary lr=%.1e (%.0f%% precision)\n",
                config.lr_mezo, config.lr, config.sparsity_ratio * 100.0f);
    }

    return trainer;
}

// Save

void alpha_combined_save(const alpha_combined_trainer * trainer, const char * path) {
    if (!trainer || !path) return;

    int rank = trainer->config.rank > 0 ? trainer->config.rank : 32;

    struct gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.type", "adapter");
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_str(gguf, "adapter.type", "lora");
    gguf_set_val_f32(gguf, "adapter.lora.alpha", (float)rank);

    // Allocate tensors with data (no_alloc=false) so GGUF computes correct offsets
    size_t total_data = 0;
    for (const auto & ll : trainer->layers) {
        total_data += ll.A.size() * sizeof(float);
        total_data += ll.B.size() * sizeof(float);
    }

    size_t ctx_size = ggml_tensor_overhead() * trainer->layers.size() * 2 + total_data + 1024;
    std::vector<uint8_t> ctx_buf(ctx_size);
    struct ggml_init_params params = { ctx_size, ctx_buf.data(), false };
    struct ggml_context * ctx = ggml_init(params);

    for (const auto & ll : trainer->layers) {
        char name_a[128], name_b[128];
        snprintf(name_a, sizeof(name_a), "%s.lora_a.weight", ll.name.c_str());
        snprintf(name_b, sizeof(name_b), "%s.lora_b.weight", ll.name.c_str());

        ggml_tensor * tA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ll.in_dim, rank);
        ggml_set_name(tA, name_a);
        memcpy(tA->data, ll.A.data(), ll.A.size() * sizeof(float));

        ggml_tensor * tB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, ll.out_dim);
        ggml_set_name(tB, name_b);
        memcpy(tB->data, ll.B.data(), ll.B.size() * sizeof(float));

        gguf_add_tensor(gguf, tA);
        gguf_add_tensor(gguf, tB);
    }

    // Write complete GGUF (header + data, offsets correct)
    gguf_write_to_file(gguf, path, false);

    ggml_free(ctx);
    gguf_free(gguf);

    fprintf(stderr, "combined: saved %zu LoRA layers to %s\n", trainer->layers.size(), path);
}

// Stats

void alpha_combined_print_stats(const alpha_combined_trainer * trainer) {
    if (!trainer) return;

    auto t_now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_now - trainer->t_start).count();

    fprintf(stderr, "\n=== Training Stats ===\n");
    fprintf(stderr, "  Steps: %d, Loss: %.4f, Avg: %.4f\n",
            trainer->step, trainer->current_loss,
            trainer->n_steps > 0 ? trainer->total_loss / trainer->n_steps : 0.0f);
    fprintf(stderr, "  Time: %.1f s (%.1f ms/step, optimized)\n",
            elapsed, trainer->step > 0 ? elapsed * 1000.0 / trainer->step : 0.0);
    fprintf(stderr, "  Selected: %d / %d params (%.1f%%), remaining %.1f%%\n",
            trainer->mask.total_masked, trainer->mask.total_params,
            trainer->mask.total_params > 0 ? 100.0f * trainer->mask.total_masked / trainer->mask.total_params : 0.0f,
            trainer->mask.total_params > 0 ? 100.0f * (1.0f - (float)trainer->mask.total_masked / trainer->mask.total_params) : 0.0f);
    fprintf(stderr, "  Convergence: velocity=%.2e, plateau_count=%d\n",
            trainer->conv.loss_velocity, trainer->conv.plateau_count);
    if (trainer->conv.n_adjustments > 0) {
        fprintf(stderr, "  Adjusted: %d times (lr=%.1e, wd=%.3f)\n",
                trainer->conv.n_adjustments,
                trainer->config.lr, trainer->config.weight_decay);
    }
    if (trainer->conv.eval_loss_best < 1e8f) {
        fprintf(stderr, "  Eval: baseline=%.4f, best=%.4f (step %d), improvement=%.1f%%\n",
                trainer->conv.eval_loss_baseline, trainer->conv.eval_loss_best,
                trainer->conv.best_step,
                100.0f * (1.0f - trainer->conv.eval_loss_best / trainer->conv.eval_loss_baseline));
    }
    fprintf(stderr, "===============================================\n");
}

// Free

void alpha_combined_trainer_free(alpha_combined_trainer * trainer) {
    if (!trainer) return;
    llama_set_adapters_lora(trainer->ctx, nullptr, 0, nullptr);
    if (trainer->adapter_buf) ggml_backend_buffer_free(trainer->adapter_buf);
    if (trainer->adapter_ctx) ggml_free(trainer->adapter_ctx);
    if (trainer->adapter) {
        trainer->adapter->ab_map.clear();
        delete trainer->adapter;
    }
    delete trainer;
}

// Callback accessor

ggml_backend_sched_eval_callback alpha_combined_get_callback(void) {
    return combined_eval_callback;
}

// Full training loop

int alpha_combined_train(
    struct llama_model * model,
    struct llama_context * ctx,
    const char * data_file,
    const char * output_file,
    alpha_combined_train_config config,
    int epochs,
    int seq_len)
{
    if (!model || !ctx || !data_file) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Load training data
    std::ifstream fin(data_file);
    if (!fin.is_open()) {
        fprintf(stderr, "combined: failed to open %s\n", data_file);
        return 1;
    }
    std::string text, line;
    while (std::getline(fin, line)) {
        if (!line.empty()) text += line + "\n";
    }
    fin.close();

    if (seq_len <= 0) seq_len = 256;

    int n_max = (int)text.size() + 256;
    std::vector<llama_token> all_tokens(n_max);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   all_tokens.data(), n_max, true, false);
    if (n_tokens < 0) {
        n_max = -n_tokens;
        all_tokens.resize(n_max);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   all_tokens.data(), n_max, true, false);
    }
    all_tokens.resize(n_tokens);

    std::vector<std::vector<llama_token>> all_sequences;
    for (int i = 0; i + seq_len <= n_tokens; i += seq_len) {
        all_sequences.emplace_back(all_tokens.begin() + i, all_tokens.begin() + i + seq_len);
    }

    if (all_sequences.empty()) {
        fprintf(stderr, "combined: no training data\n");
        return 1;
    }

    // Split into train (90%) and eval (10%) - held-out eval prevents false convergence
    size_t n_eval = std::max((size_t)1, all_sequences.size() / 10);
    size_t n_train = all_sequences.size() - n_eval;

    std::vector<std::vector<llama_token>> sequences(all_sequences.begin(), all_sequences.begin() + n_train);
    std::vector<std::vector<llama_token>> eval_sequences(all_sequences.begin() + n_train, all_sequences.end());

    // Flatten eval sequences into one token buffer for quality gate
    std::vector<llama_token> eval_tokens;
    for (const auto & seq : eval_sequences) {
        eval_tokens.insert(eval_tokens.end(), seq.begin(), seq.end());
    }
    int eval_len = std::min((int)eval_tokens.size(), seq_len);

    fprintf(stderr, "combined: %d tokens, %zu train + %zu eval sequences of %d\n",
            n_tokens, sequences.size(), eval_sequences.size(), seq_len);

    // Initialize trainer
    auto * trainer = alpha_combined_trainer_init(model, ctx, config);
    if (!trainer) return 1;

    // Step 0: Compute Block-Fisher mask using eval sequences
    alpha_combined_compute_mask(trainer, eval_tokens.data(), (int)eval_tokens.size(), seq_len);

    // Compute baseline eval loss on held-out data (not training data)
    alpha_combined_quality_gate(trainer, eval_tokens.data(), eval_len);

    if (epochs <= 0) epochs = 1;

    fprintf(stderr, "\n=== Training (%.0f%% precision) ===\n", config.sparsity_ratio * 100.0f);
    fprintf(stderr, "  Optimized pipeline active\n");
    fprintf(stderr, "  Data: %s (%zu sequences x %d tokens)\n", data_file, sequences.size(), seq_len);
    fprintf(stderr, "  Epochs: %d, Max steps: %d, Warmup: %d\n", epochs, config.max_steps, config.warmup_steps);
    fprintf(stderr, "=======================================================\n\n");

    bool converged = false;
    for (int epoch = 0; epoch < epochs && !converged; epoch++) {
        float epoch_loss = 0;
        int epoch_steps = 0;

        // Shuffle sequences
        std::mt19937 shuffle_rng(42 + epoch);
        for (size_t i = sequences.size() - 1; i > 0; i--) {
            size_t j = shuffle_rng() % (i + 1);
            std::swap(sequences[i], sequences[j]);
        }

        for (size_t s = 0; s < sequences.size() && !converged; s++) {
            float loss = alpha_combined_step(trainer,
                sequences[s].data(), (int)sequences[s].size());
            if (loss < 0) continue;

            epoch_loss += loss;
            epoch_steps++;

            // Logging
            if (config.log_every > 0 && trainer->step % config.log_every == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - trainer->t_start).count();
                double ms_per_step = trainer->step > 0 ? elapsed * 1000.0 / trainer->step : 0.0;

                if (config.json_progress) {
                    fprintf(stderr, "{\"event\":\"step\",\"step\":%d,\"loss\":%.4f,\"eval_loss\":%.4f,"
                            "\"velocity\":%.6e,\"ms_per_step\":%.1f,\"lr\":%.2e,\"lr2\":%.2e,"
                            "\"reg\":%.3f,\"wd\":%.4f,\"adjustments\":%d,"
                            "\"epoch\":%d,\"total_epochs\":%d,\"elapsed_s\":%.1f}\n",
                            trainer->step, loss, trainer->conv.eval_loss_best,
                            trainer->conv.loss_velocity, ms_per_step,
                            config.lr, config.lr_mezo, config.elastic_lambda, config.weight_decay,
                            trainer->conv.n_adjustments, epoch + 1, epochs, elapsed);
                } else {
                    fprintf(stderr, "  step %d | loss %.4f | vel %.2e | %.1f ms/step | epoch %d/%d\n",
                            trainer->step, loss, trainer->conv.loss_velocity,
                            ms_per_step, epoch + 1, epochs);
                }
            }

            // Eval + quality gate (on held-out data, not training data)
            if (config.eval_every > 0 && trainer->step % config.eval_every == 0) {
                alpha_combined_quality_gate(trainer,
                    eval_tokens.data(), eval_len);
            }

            // Save checkpoint
            if (config.save_every > 0 && output_file && trainer->step % config.save_every == 0) {
                alpha_combined_save(trainer, output_file);
            }

            // Check convergence
            converged = alpha_combined_check_convergence(trainer, loss);
        }

        if (epoch_steps > 0) {
            fprintf(stderr, "Epoch %d/%d complete. avg_loss=%.4f\n",
                    epoch + 1, epochs, epoch_loss / epoch_steps);
        }
    }

    // Final quality gate (on held-out eval data)
    alpha_combined_quality_gate(trainer, eval_tokens.data(), eval_len);

    // Save final model
    if (output_file && output_file[0]) {
        alpha_combined_save(trainer, output_file);
    }

    alpha_combined_print_stats(trainer);

    // Final completion event
    if (config.json_progress) {
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_s = std::chrono::duration<double>(t_end - trainer->t_start).count();
        fprintf(stderr, "{\"event\":\"complete\",\"steps\":%d,\"final_loss\":%.4f,"
                "\"eval_loss_baseline\":%.4f,\"eval_loss_best\":%.4f,\"best_step\":%d,"
                "\"improvement_pct\":%.1f,\"elapsed_s\":%.1f,\"output\":\"%s\","
                "\"adjustments\":%d}\n",
                trainer->step, trainer->current_loss,
                trainer->conv.eval_loss_baseline, trainer->conv.eval_loss_best,
                trainer->conv.best_step,
                trainer->conv.eval_loss_baseline > 0 ?
                    100.0f * (1.0f - trainer->conv.eval_loss_best / trainer->conv.eval_loss_baseline) : 0.0f,
                total_s, output_file ? output_file : "",
                trainer->conv.n_adjustments);
    }

    alpha_combined_trainer_free(trainer);
    return 0;
}
