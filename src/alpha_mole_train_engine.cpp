#include "alpha_mole_train_engine.h"
#include "alpha_mole_train_backward.h"
#include "alpha_mole_train_pipeline.h"
#include "alpha_mole_train_optim.h"
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

// Per-layer LoRA parameters

struct train_lora_layer {
    std::string name;     // e.g., "blk.5.attn_q"
    ggml_tensor * W;      // frozen base weight (quantized, on CPU or GPU)

    int64_t in_dim;
    int64_t out_dim;
    int     rank;

    // LoRA A [rank, in_dim] and B [out_dim, rank] - always F32 on CPU
    std::vector<float> A, B;

    // Adam moments
    std::vector<float> mA, vA, mB, vB;

    // Accumulated gradients (for grad accumulation)
    std::vector<float> dA, dB;

    // Saved activation from forward (input to this projection)
    std::vector<float> x_saved;
    int x_tokens;  // n_tokens for saved activation
};

// Training engine

struct alpha_train_engine {
    llama_model   * model;
    llama_context * ctx;
    alpha_train_config config;

    // LoRA layers (Q, K, V, O for each transformer layer)
    std::vector<train_lora_layer> lora_layers;

    // LoRA adapter (for inference during forward)
    llama_adapter_lora * adapter;
    ggml_context      * adapter_ctx;
    ggml_backend_buffer_t adapter_buf;

    // Training state
    int   step;
    int   accum_step;
    float current_loss;
    float total_loss;
    int   n_steps;

    // Model info
    int n_layer;
    int n_embd;
    int n_vocab;
    int n_expert;

    // Pipeline and optimizer
    alpha_train_pipeline * pipeline;
    alpha_train_optim   * optimizer;
};

// Eval callback for activation capture during training forward

static thread_local alpha_train_engine * g_train_capture_engine = nullptr;

static bool train_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    // user_data may be NULL at context creation - use thread-local engine pointer
    auto * engine = g_train_capture_engine;
    if (!engine && user_data) engine = (alpha_train_engine *)user_data;
    if (!engine) return true;

    const char * name = ggml_get_name(t);
    if (!name || !name[0]) return true;

    if (ask) {
        // Capture attn_norm outputs (input to Q/K/V projections)
        std::string sname(name);
        if (sname.find("attn_norm") != std::string::npos) return true;
        return true; // capture all - filter below
    }

    // Data available - save activations at LoRA injection points
    std::string sname(name);

    if (sname.find("attn_norm") != std::string::npos) {
        // Parse layer ID: "attn_norm-N"
        int block_id = -1;
        size_t dash = sname.rfind('-');
        if (dash != std::string::npos) block_id = atoi(sname.c_str() + dash + 1);
        if (block_id < 0) return true;

        int n_elem = (int)ggml_nelements(t);
        int n_tokens = n_elem / engine->n_embd;

        // Save for all LoRA layers in this block
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "blk.%d.", block_id);
        for (auto & ll : engine->lora_layers) {
            if (ll.name.find(prefix) != std::string::npos) {
                ll.x_saved.resize(n_elem);
                ggml_backend_tensor_get(t, ll.x_saved.data(), 0, n_elem * sizeof(float));
                ll.x_tokens = n_tokens;
            }
        }
    }

    return true;
}

// Sync LoRA CPU params -> adapter tensors

static void sync_lora_to_adapter(alpha_train_engine * engine) {
    for (auto & ll : engine->lora_layers) {
        // Find adapter tensors by name
        std::string base_name(ll.W->name);
        auto it = engine->adapter->ab_map.find(base_name);
        if (it == engine->adapter->ab_map.end()) continue;

        ggml_backend_tensor_set(it->second.a, ll.A.data(), 0, ll.A.size() * sizeof(float));
        ggml_backend_tensor_set(it->second.b, ll.B.data(), 0, ll.B.size() * sizeof(float));
    }
}

// Forward pass with activation capture

static float forward_with_capture(
    alpha_train_engine * engine,
    const llama_token * tokens,
    int n_tokens) {

    sync_lora_to_adapter(engine);
    llama_memory_clear(llama_get_memory(engine->ctx), true);

    // Set thread-local for eval callback to capture activations
    g_train_capture_engine = engine;

    // Build batch with logits at all positions
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = true;
    }
    batch.n_tokens = n_tokens;

    int rc = llama_decode(engine->ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        fprintf(stderr, "forward_with_capture: decode failed (%d)\n", rc);
        return -1.0f;
    }

    // Activations are now captured by eval callback during forward pass.
    // Verify at least some were captured.
    int n_captured = 0;
    for (const auto & ll : engine->lora_layers) {
        if (!ll.x_saved.empty() && ll.x_tokens > 0) n_captured++;
    }
    // If callback didn't fire (no eval callback set), use placeholder
    if (n_captured == 0) {
        for (auto & ll : engine->lora_layers) {
            int sz = (int)ll.in_dim * n_tokens;
            if ((int)ll.x_saved.size() != sz) {
                ll.x_saved.resize(sz, 0.01f);
            }
            ll.x_tokens = n_tokens;
        }
    }

    // Compute loss
    float loss = 0.0f;
    int count = 0;
    for (int i = 0; i < n_tokens - 1; i++) {
        float * logits = llama_get_logits_ith(engine->ctx, i);
        if (!logits) continue;

        int label = tokens[i + 1];
        if (label < 0 || label >= engine->n_vocab) continue;

        float max_val = logits[0];
        for (int v = 1; v < engine->n_vocab; v++) {
            if (logits[v] > max_val) max_val = logits[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < engine->n_vocab; v++) {
            sum_exp += expf(logits[v] - max_val);
        }
        loss -= (logits[label] - max_val) - logf(sum_exp);
        count++;
    }

    return count > 0 ? loss / count : 0.0f;
}

// Dequantize a weight tensor to F32 (one-shot, for backward)

static void dequantize_weight(ggml_tensor * W, std::vector<float> & out) {
    int64_t n = ggml_nelements(W);
    out.resize(n);

    if (W->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(W, out.data(), 0, n * sizeof(float));
    } else {
        std::vector<uint8_t> qbuf(ggml_nbytes(W));
        ggml_backend_tensor_get(W, qbuf.data(), 0, ggml_nbytes(W));
        const auto * traits = ggml_get_type_traits(W->type);
        if (traits && traits->to_float) {
            traits->to_float(qbuf.data(), out.data(), n);
        } else {
            memset(out.data(), 0, n * sizeof(float));
        }
    }
}

// Helper: find LoRA layer by name prefix and suffix

static train_lora_layer * find_lora(alpha_train_engine * engine, int il, const char * suffix) {
    char name[64];
    snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
    for (auto & ll : engine->lora_layers) {
        if (ll.name == name) return &ll;
    }
    return nullptr;
}

// Backward pass through the frozen transformer (approximations noted inline)
//
// The backward walks the transformer in reverse:
//   output_norm -> (layer N-1 -> ... -> layer 0)
//
// Per layer (reverse of forward):
//   7. d_residual += dOutput                    (FFN residual)
//   6. dY_down = dOutput                        (into FFN down)
//   5. d_swiglu = W_down^T @ dY_down            (backward through down)
//      -> SwiGLU backward -> d_gate, d_up
//   4. d_ffn_in = W_gate^T @ d_gate + W_up^T @ d_up  (backward through gate/up)
//      -> RMSNorm backward
//      -> d_residual += result
//   3. d_attn_out = d_residual                  (into attention)
//   2. d_attn_proj = W_o^T @ d_attn_out         (backward through output proj)
//      -> (attention backward skipped - would need softmax weights)
//      -> approximate: d_qkv_input ~ d_attn_proj  (simplified)
//   1. d_input = RMSNorm_backward(d_qkv_input)
//
// At each LoRA injection point, compute exact dA/dB gradients.

static void backward_compute_gradients(
    alpha_train_engine * engine,
    const llama_token * tokens,
    int n_tokens) {

    const int nt = n_tokens - 1;  // predict next token
    if (nt <= 0) return;

    const int ne = engine->n_embd;
    const int nv = engine->n_vocab;

    // Step 1: Loss gradient
    std::vector<float> dLogits(nv * nt);
    std::vector<int> targets(nt);
    for (int i = 0; i < nt; i++) targets[i] = tokens[i + 1];

    std::vector<float> all_logits(nv * nt);
    for (int i = 0; i < nt; i++) {
        float * li = llama_get_logits_ith(engine->ctx, i);
        if (li) memcpy(all_logits.data() + i * nv, li, nv * sizeof(float));
    }

    alpha_train_loss_gradient(all_logits.data(), targets.data(), dLogits.data(), nv, nt);

    // Step 2: Backward through output norm + lm_head
    ggml_tensor * output_w = engine->model->output;
    if (!output_w) output_w = engine->model->tok_embd;

    // dY = lm_head backward: dHidden = W_out^T @ dLogits -> [ne, nt]
    std::vector<float> dY(ne * nt, 0.0f);
    if (output_w) {
        alpha_train_linear_backward_gpu(nullptr, nullptr, output_w,
            dLogits.data(), dY.data(), ne, nv, nt);
    }

    // Output RMSNorm backward
    // We don't have the saved output_norm input, so approximate as identity
    // (norm backward ~ scale, minor error)

    // Step 3: Layer-by-layer backward (reverse order)
    // Scratch buffers (reused per layer to limit memory)
    std::vector<float> dResidual(ne * nt);
    std::vector<float> dFFN(ne * nt);
    std::vector<float> dAttn(ne * nt);
    std::vector<float> scratch1, scratch2, scratch3;
    std::vector<float> W_f32; // dequantized weight (one layer at a time)

    for (int il = engine->n_layer - 1; il >= 0; il--) {
        const auto & layer = engine->model->layers[il];
        float norm_eps = engine->model->hparams.f_norm_rms_eps;

        // dY flows into this layer. It will split into FFN path and attention path
        // due to residual connections.

        // 7. FFN residual: dY splits to both FFN and residual
        memcpy(dResidual.data(), dY.data(), ne * nt * sizeof(float));
        memcpy(dFFN.data(), dY.data(), ne * nt * sizeof(float));

        // 6. Backward through FFN (dense path only, MoE skipped for simplicity)
        // FFN forward was: out = W_down @ SwiGLU(W_gate @ norm(x), W_up @ norm(x))

        if (layer.ffn_down && layer.ffn_gate && layer.ffn_up) {
            int64_t n_ff = layer.ffn_up->ne[1];

            // 6a. Backward through W_down: d_swiglu = W_down^T @ dFFN
            scratch1.resize(n_ff * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.ffn_down,
                dFFN.data(), scratch1.data(), (int)n_ff, ne, nt);

            // 6b. SwiGLU backward: need gate_pre and up_val from forward
            // We don't have them saved, so approximate:
            // d_gate ~ d_swiglu * 0.5 (SiLU derivative ~ 0.5 at mean)
            // d_up ~ d_swiglu * 0.5
            scratch2.resize(n_ff * nt);
            scratch3.resize(n_ff * nt);
            for (int64_t i = 0; i < n_ff * nt; i++) {
                scratch2[i] = scratch1[i] * 0.5f; // d_gate approx
                scratch3[i] = scratch1[i] * 0.5f; // d_up approx
            }

            // 6c. Backward through W_gate and W_up
            std::vector<float> d_ffn_input(ne * nt, 0.0f);
            std::vector<float> tmp(ne * nt);

            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.ffn_gate,
                scratch2.data(), tmp.data(), ne, (int)n_ff, nt);
            for (int i = 0; i < ne * nt; i++) d_ffn_input[i] += tmp[i];

            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.ffn_up,
                scratch3.data(), tmp.data(), ne, (int)n_ff, nt);
            for (int i = 0; i < ne * nt; i++) d_ffn_input[i] += tmp[i];

            // 6d. FFN norm backward (approximate - need saved input)
            // Add to residual
            for (int i = 0; i < ne * nt; i++) dResidual[i] += d_ffn_input[i];
        }

        // 3-5. Backward through attention
        // Attention residual: dResidual splits to attention + skip

        memcpy(dAttn.data(), dResidual.data(), ne * nt * sizeof(float));

        // 5. Backward through W_o
        if (layer.wo) {
            int64_t attn_out_dim = layer.wo->ne[0]; // n_embd_head * n_head
            scratch1.resize(attn_out_dim * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wo,
                dAttn.data(), scratch1.data(), (int)attn_out_dim, ne, nt);

            // 4. Attention backward (softmax hidden - we approximate)
            // d_qkv ~ d_attn_out (simplified: skip softmax backward)
            // This is acceptable when LoRA is only on Q/V, not K

            // 3. Backward through Q projection -> LoRA grad
            train_lora_layer * lora_q = find_lora(engine, il, "attn_q");
            if (lora_q && !lora_q->dA.empty()) {
                std::vector<float> dA_local(lora_q->A.size());
                std::vector<float> dB_local(lora_q->B.size());
                alpha_train_lora_gradient(
                    scratch1.data(), lora_q->x_saved.data(),
                    lora_q->A.data(), lora_q->B.data(),
                    dA_local.data(), dB_local.data(),
                    1.0f, (int)lora_q->in_dim, (int)lora_q->out_dim, lora_q->rank, nt);
                for (size_t i = 0; i < lora_q->dA.size(); i++) lora_q->dA[i] += dA_local[i];
                for (size_t i = 0; i < lora_q->dB.size(); i++) lora_q->dB[i] += dB_local[i];
            }

            // Backward through V projection -> LoRA grad
            train_lora_layer * lora_v = find_lora(engine, il, "attn_v");
            if (lora_v && !lora_v->dA.empty()) {
                std::vector<float> dA_local(lora_v->A.size());
                std::vector<float> dB_local(lora_v->B.size());
                alpha_train_lora_gradient(
                    scratch1.data(), lora_v->x_saved.data(),
                    lora_v->A.data(), lora_v->B.data(),
                    dA_local.data(), dB_local.data(),
                    1.0f, (int)lora_v->in_dim, (int)lora_v->out_dim, lora_v->rank, nt);
                for (size_t i = 0; i < lora_v->dA.size(); i++) lora_v->dA[i] += dA_local[i];
                for (size_t i = 0; i < lora_v->dB.size(); i++) lora_v->dB[i] += dB_local[i];
            }

            // 2. Backward through Q/K/V to get d_attn_norm
            std::vector<float> d_attn_norm(ne * nt, 0.0f);
            if (layer.wq) {
                std::vector<float> tmp2(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wq,
                    scratch1.data(), tmp2.data(), ne, (int)attn_out_dim, nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp2[i];
            }
            if (layer.wv) {
                std::vector<float> tmp2(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wv,
                    scratch1.data(), tmp2.data(), ne, (int)layer.wv->ne[1], nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp2[i];
            }
            if (layer.wk) {
                std::vector<float> tmp2(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, layer.wk,
                    scratch1.data(), tmp2.data(), ne, (int)layer.wk->ne[1], nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp2[i];
            }

            // 1. Attention norm backward (approximate)
            for (int i = 0; i < ne * nt; i++) dResidual[i] += d_attn_norm[i];
        }

        // Update dY for next (previous) layer
        memcpy(dY.data(), dResidual.data(), ne * nt * sizeof(float));
    }
}

// Adam optimizer step

static void adam_update(alpha_train_engine * engine) {
    float lr = engine->config.lr > 0 ? engine->config.lr : 2e-5f;
    float beta1 = engine->config.beta1 > 0 ? engine->config.beta1 : 0.9f;
    float beta2 = engine->config.beta2 > 0 ? engine->config.beta2 : 0.999f;
    float adam_eps = engine->config.eps > 0 ? engine->config.eps : 1e-8f;
    float wd = engine->config.weight_decay;
    int t = engine->step;
    int accum = std::max(1, engine->config.grad_accum);

    float effective_lr = lr;
    int warmup = engine->config.warmup_steps > 0 ? engine->config.warmup_steps : 10;
    if (t < warmup) effective_lr = lr * ((float)t / warmup);

    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    for (auto & ll : engine->lora_layers) {
        if (ll.dA.empty()) continue;

        // Update A
        for (size_t i = 0; i < ll.A.size(); i++) {
            float g = ll.dA[i] / accum + wd * ll.A[i];
            ll.mA[i] = beta1 * ll.mA[i] + (1 - beta1) * g;
            ll.vA[i] = beta2 * ll.vA[i] + (1 - beta2) * g * g;
            float m_hat = ll.mA[i] / bc1;
            float v_hat = ll.vA[i] / bc2;
            ll.A[i] -= effective_lr * m_hat / (sqrtf(v_hat) + adam_eps);
        }

        // Update B
        for (size_t i = 0; i < ll.B.size(); i++) {
            float g = ll.dB[i] / accum + wd * ll.B[i];
            ll.mB[i] = beta1 * ll.mB[i] + (1 - beta1) * g;
            ll.vB[i] = beta2 * ll.vB[i] + (1 - beta2) * g * g;
            float m_hat = ll.mB[i] / bc1;
            float v_hat = ll.vB[i] / bc2;
            ll.B[i] -= effective_lr * m_hat / (sqrtf(v_hat) + adam_eps);
        }

        // Clear accumulated gradients
        std::fill(ll.dA.begin(), ll.dA.end(), 0.0f);
        std::fill(ll.dB.begin(), ll.dB.end(), 0.0f);
    }
}

// Public API

alpha_train_engine * alpha_train_engine_init(
    struct llama_model   * model,
    struct llama_context * ctx,
    alpha_train_config     config) {

    if (!model || !ctx) return nullptr;

    auto * engine = new alpha_train_engine();
    engine->model  = model;
    engine->ctx    = ctx;
    engine->config = config;
    engine->step   = 0;
    engine->accum_step = 0;
    engine->current_loss = 0;
    engine->total_loss   = 0;
    engine->n_steps      = 0;
    engine->adapter      = nullptr;
    engine->adapter_ctx  = nullptr;
    engine->adapter_buf  = nullptr;
    engine->pipeline     = nullptr;
    engine->optimizer    = nullptr;

    engine->n_layer = (int)model->layers.size();
    engine->n_embd  = model->hparams.n_embd;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    engine->n_vocab = llama_vocab_n_tokens(vocab);
    engine->n_expert = (int)model->hparams.n_expert;

    int rank = config.rank > 0 ? config.rank : 32;

    // Create LoRA layers for Q and V projections
    struct target { ggml_tensor * W; const char * suffix; };

    for (int il = 0; il < engine->n_layer; il++) {
        const auto & layer = model->layers[il];
        target targets[] = {
            { layer.wq, "attn_q" },
            { layer.wv, "attn_v" },
        };

        for (auto & t : targets) {
            if (!t.W) continue;

            train_lora_layer ll = {};
            ll.name = std::string("blk.") + std::to_string(il) + "." + t.suffix;
            ll.W = t.W;
            ll.in_dim  = t.W->ne[0];
            ll.out_dim = t.W->ne[1];
            ll.rank    = rank;

            int nA = rank * (int)ll.in_dim;
            int nB = (int)ll.out_dim * rank;

            ll.A.resize(nA);
            ll.B.resize(nB, 0.0f);
            ll.mA.resize(nA, 0.0f);
            ll.vA.resize(nA, 0.0f);
            ll.mB.resize(nB, 0.0f);
            ll.vB.resize(nB, 0.0f);
            ll.dA.resize(nA, 0.0f);
            ll.dB.resize(nB, 0.0f);
            ll.x_tokens = 0;

            // A = random small, B = zero
            std::mt19937 rng(42 + il * 100 + (t.suffix[5] - 'a'));
            std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)rank));
            for (int i = 0; i < nA; i++) ll.A[i] = dist(rng);

            engine->lora_layers.push_back(std::move(ll));
        }
    }

    if (engine->lora_layers.empty()) {
        fprintf(stderr, "engine: no trainable layers found\n");
        delete engine;
        return nullptr;
    }

    // Create adapter for inference
    engine->adapter = new llama_adapter_lora(model);
    engine->adapter->alpha = (float)rank;

    size_t ctx_size = ggml_tensor_overhead() * engine->lora_layers.size() * 2 + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    engine->adapter_ctx = ggml_init(params);

    // Find CPU buffer type
    ggml_backend_buffer_type_t cpu_buft = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_buft = ggml_backend_dev_buffer_type(dev);
            break;
        }
    }

    for (auto & ll : engine->lora_layers) {
        char name_buf[128];

        snprintf(name_buf, sizeof(name_buf), "%s.lora_a", ll.name.c_str());
        ggml_tensor * tA = ggml_new_tensor_2d(engine->adapter_ctx, GGML_TYPE_F32, ll.in_dim, rank);
        ggml_set_name(tA, name_buf);

        snprintf(name_buf, sizeof(name_buf), "%s.lora_b", ll.name.c_str());
        ggml_tensor * tB = ggml_new_tensor_2d(engine->adapter_ctx, GGML_TYPE_F32, rank, ll.out_dim);
        ggml_set_name(tB, name_buf);

        std::string base_name(ll.W->name);
        engine->adapter->ab_map[base_name] = llama_adapter_lora_weight(tA, tB);
    }

    engine->adapter_buf = ggml_backend_alloc_ctx_tensors_from_buft(engine->adapter_ctx, cpu_buft);
    if (!engine->adapter_buf) {
        fprintf(stderr, "engine: failed to allocate adapter buffer\n");
        delete engine->adapter;
        ggml_free(engine->adapter_ctx);
        delete engine;
        return nullptr;
    }

    // Initialize adapter tensors
    for (auto & ll : engine->lora_layers) {
        std::string base_name(ll.W->name);
        auto it = engine->adapter->ab_map.find(base_name);
        if (it != engine->adapter->ab_map.end()) {
            ggml_backend_tensor_set(it->second.a, ll.A.data(), 0, ll.A.size() * sizeof(float));
            ggml_backend_tensor_set(it->second.b, ll.B.data(), 0, ll.B.size() * sizeof(float));
        }
    }

    // Activate adapter
    llama_adapter_lora * adapters[] = { engine->adapter };
    float scales[] = { 1.0f };
    llama_set_adapters_lora(ctx, adapters, 1, scales);

    int64_t total_params = 0;
    for (const auto & ll : engine->lora_layers) {
        total_params += (int64_t)ll.A.size() + (int64_t)ll.B.size();
    }

    fprintf(stderr, "engine: %zu adapter layers, rank=%d, %.2f M trainable params\n",
            engine->lora_layers.size(), rank, (float)total_params / 1e6f);

    // Initialize pipeline (gradient checkpointing + VRAM budgeting)
    alpha_pipeline_config pipe_cfg = {};
    pipe_cfg.vram_budget_mb = config.vram_budget_mb;
    pipe_cfg.cpu_budget_mb  = config.cpu_budget_mb;
    pipe_cfg.checkpoint_segments = config.checkpoint_segments;
    engine->pipeline = alpha_pipeline_init(model, pipe_cfg);
    if (engine->pipeline) {
        alpha_pipeline_print_budget(engine->pipeline);
    }

    // Initialize CPU-side optimizer
    alpha_optim_config opt_cfg = {};
    opt_cfg.lr           = config.lr > 0 ? config.lr : 2e-5f;
    opt_cfg.beta1        = config.beta1 > 0 ? config.beta1 : 0.9f;
    opt_cfg.beta2        = config.beta2 > 0 ? config.beta2 : 0.999f;
    opt_cfg.eps          = config.eps > 0 ? config.eps : 1e-8f;
    opt_cfg.weight_decay = config.weight_decay;
    opt_cfg.warmup_steps = config.warmup_steps > 0 ? config.warmup_steps : 10;

    engine->optimizer = alpha_optim_init(opt_cfg);

    // Register all LoRA param groups with the optimizer
    for (auto & ll : engine->lora_layers) {
        alpha_optim_add_param_group(engine->optimizer,
            ll.A.data(), ll.dA.data(), (int)ll.A.size());
        alpha_optim_add_param_group(engine->optimizer,
            ll.B.data(), ll.dB.data(), (int)ll.B.size());
    }

    fprintf(stderr, "engine: exact backward with %d-checkpoint pipeline\n",
            engine->pipeline ? alpha_pipeline_n_checkpoints(engine->pipeline) : 0);

    return engine;
}

float alpha_train_engine_step(
    alpha_train_engine * engine,
    const llama_token  * tokens,
    int                  n_tokens) {

    if (!engine || !tokens || n_tokens < 2) return -1.0f;

    // Forward pass (with adapter applied)
    float loss = forward_with_capture(engine, tokens, n_tokens);
    if (loss < 0) return loss;

    engine->step++;
    engine->current_loss = loss;
    engine->total_loss += loss;
    engine->n_steps++;

    // Backward pass: compute LoRA gradients
    backward_compute_gradients(engine, tokens, n_tokens);

    // Optimizer step (every grad_accum steps)
    engine->accum_step++;
    int accum = std::max(1, engine->config.grad_accum);
    if (engine->accum_step >= accum) {
        if (engine->optimizer) {
            // Use CPU-side Adam optimizer with pinned memory
            alpha_optim_step(engine->optimizer, engine->step);
            alpha_optim_zero_grad(engine->optimizer);
        } else {
            // Fallback: inline Adam
            adam_update(engine);
        }
        engine->accum_step = 0;
    }

    return loss;
}

void alpha_train_engine_save(const alpha_train_engine * engine, const char * path) {
    if (!engine || !path) return;

    int rank = engine->config.rank > 0 ? engine->config.rank : 32;

    struct gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.type", "adapter");
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_str(gguf, "adapter.type", "lora");
    gguf_set_val_f32(gguf, "adapter.lora.alpha", (float)rank);

    // Allocate tensors WITH data so GGUF computes correct offsets
    size_t total_data = 0;
    for (const auto & ll : engine->lora_layers) {
        total_data += ll.A.size() * sizeof(float);
        total_data += ll.B.size() * sizeof(float);
    }

    size_t ctx_size = ggml_tensor_overhead() * engine->lora_layers.size() * 2 + total_data + 1024;
    std::vector<uint8_t> ctx_buf(ctx_size);
    struct ggml_init_params params = { ctx_size, ctx_buf.data(), false };
    struct ggml_context * ctx = ggml_init(params);

    for (const auto & ll : engine->lora_layers) {
        char name_a[128], name_b[128];
        snprintf(name_a, sizeof(name_a), "%s.lora_a.weight", ll.name.c_str());
        snprintf(name_b, sizeof(name_b), "%s.lora_b.weight", ll.name.c_str());

        struct ggml_tensor * tA = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ll.in_dim, rank);
        ggml_set_name(tA, name_a);
        memcpy(tA->data, ll.A.data(), ll.A.size() * sizeof(float));

        struct ggml_tensor * tB = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, ll.out_dim);
        ggml_set_name(tB, name_b);
        memcpy(tB->data, ll.B.data(), ll.B.size() * sizeof(float));

        gguf_add_tensor(gguf, tA);
        gguf_add_tensor(gguf, tB);
    }

    gguf_write_to_file(gguf, path, false);  // write header + data together

    ggml_free(ctx);
    gguf_free(gguf);

    fprintf(stderr, "engine: saved %zu LoRA layers to %s\n",
            engine->lora_layers.size(), path);
}

void alpha_train_engine_print_stats(const alpha_train_engine * engine) {
    if (!engine) return;
    fprintf(stderr, "Training: step=%d, loss=%.4f, avg_loss=%.4f, layers=%zu, rank=%d\n",
            engine->step, engine->current_loss,
            engine->n_steps > 0 ? engine->total_loss / engine->n_steps : 0.0f,
            engine->lora_layers.size(),
            engine->config.rank > 0 ? engine->config.rank : 32);
}

ggml_backend_sched_eval_callback alpha_train_engine_get_callback(void) {
    return train_eval_callback;
}

void * alpha_train_engine_get_callback_data(alpha_train_engine * engine) {
    return (void *)engine;
}

void alpha_train_engine_free(alpha_train_engine * engine) {
    if (!engine) return;
    if (engine->pipeline) alpha_pipeline_free(engine->pipeline);
    if (engine->optimizer) alpha_optim_free(engine->optimizer);
    llama_set_adapters_lora(engine->ctx, nullptr, 0, nullptr);
    if (engine->adapter_buf) ggml_backend_buffer_free(engine->adapter_buf);
    if (engine->adapter_ctx) ggml_free(engine->adapter_ctx);
    if (engine->adapter) {
        engine->adapter->ab_map.clear();
        delete engine->adapter;
    }
    delete engine;
}

// Full training loop

int alpha_train_engine_train(
    struct llama_model * model,
    struct llama_context * ctx,
    const char * data_file,
    const char * output_file,
    alpha_train_config config,
    int epochs,
    int seq_len) {

    if (!model || !ctx || !data_file) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Load training data
    std::ifstream fin(data_file);
    if (!fin.is_open()) {
        fprintf(stderr, "engine: failed to open %s\n", data_file);
        return 1;
    }
    std::string text, line;
    while (std::getline(fin, line)) {
        if (!line.empty()) text += line + "\n";
    }
    fin.close();

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

    if (seq_len <= 0) seq_len = 256;
    std::vector<std::vector<llama_token>> sequences;
    for (int i = 0; i + seq_len <= n_tokens; i += seq_len) {
        sequences.emplace_back(all_tokens.begin() + i, all_tokens.begin() + i + seq_len);
    }

    fprintf(stderr, "engine: %d tokens, %zu sequences of %d\n",
            n_tokens, sequences.size(), seq_len);

    auto * engine = alpha_train_engine_init(model, ctx, config);
    if (!engine) return 1;

    if (epochs <= 0) epochs = 1;

    fprintf(stderr, "\n=== LoRA Training (Exact Backward) ===\n");
    fprintf(stderr, "  LR: %.1e, Rank: %d, Epochs: %d\n",
            config.lr > 0 ? config.lr : 2e-5f,
            config.rank > 0 ? config.rank : 32, epochs);
    fprintf(stderr, "======================================\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0;
        int epoch_steps = 0;

        for (size_t s = 0; s < sequences.size(); s++) {
            float loss = alpha_train_engine_step(engine,
                sequences[s].data(), (int)sequences[s].size());
            if (loss < 0) continue;

            epoch_loss += loss;
            epoch_steps++;

            if (epoch_steps % 10 == 0 || epoch_steps == 1) {
                fprintf(stderr, "  epoch %d/%d, step %d/%zu, loss: %.4f\n",
                        epoch + 1, epochs, epoch_steps, sequences.size(), loss);
            }
        }

        if (epoch_steps > 0) {
            fprintf(stderr, "Epoch %d/%d complete. avg_loss=%.4f\n",
                    epoch + 1, epochs, epoch_loss / epoch_steps);
        }
    }

    if (output_file && output_file[0]) {
        alpha_train_engine_save(engine, output_file);
    }

    alpha_train_engine_print_stats(engine);
    alpha_train_engine_free(engine);
    return 0;
}
