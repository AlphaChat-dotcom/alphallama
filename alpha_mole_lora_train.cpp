#include "alpha_mole_lora_train.h"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Per-layer LoRA parameters (CPU-side, for MeZO optimization)
struct lora_layer {
    int layer_id;
    std::string name;  // e.g., "blk.0.attn_q"

    // CPU-side A/B matrices (the trainable parameters)
    std::vector<float> A;   // [in_dim, rank] — matches GGUF/adapter convention
    std::vector<float> B;   // [rank, out_dim]

    // Adam moments
    std::vector<float> mA, vA;
    std::vector<float> mB, vB;

    // MeZO: saved perturbation direction
    std::vector<float> zA;
    std::vector<float> zB;

    int64_t in_dim;
    int64_t out_dim;
    int     rank;

    // Pointers into the adapter's ggml tensors (for fast set)
    ggml_tensor * adapter_a;
    ggml_tensor * adapter_b;
};

struct alpha_lora_trainer {
    alpha_lora_train_config config;
    llama_model * model;
    llama_context * ctx;

    std::vector<lora_layer> layers;

    // The in-memory LoRA adapter
    llama_adapter_lora * adapter;
    ggml_context * adapter_ctx;
    ggml_backend_buffer_t adapter_buf;

    // Training state
    int step;
    float current_loss;
    float total_loss;
    int n_steps;

    // RNG
    std::mt19937 rng;
};

// Sync CPU A/B matrices into the adapter's ggml tensors
static void sync_to_adapter(alpha_lora_trainer * trainer) {
    for (auto & ll : trainer->layers) {
        ggml_backend_tensor_set(ll.adapter_a, ll.A.data(), 0, ll.A.size() * sizeof(float));
        ggml_backend_tensor_set(ll.adapter_b, ll.B.data(), 0, ll.B.size() * sizeof(float));
    }
}

// MeZO: sample random perturbation direction z ~ N(0,1)
static void mezo_sample_z(alpha_lora_trainer * trainer) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & ll : trainer->layers) {
        for (size_t i = 0; i < ll.zA.size(); i++) ll.zA[i] = dist(trainer->rng);
        for (size_t i = 0; i < ll.zB.size(); i++) ll.zB[i] = dist(trainer->rng);
    }
}

// MeZO: perturb all LoRA parameters by sign * eps * z
static void mezo_perturb(alpha_lora_trainer * trainer, float sign) {
    float eps = trainer->config.mezo_eps > 0 ? trainer->config.mezo_eps : 1e-3f;
    for (auto & ll : trainer->layers) {
        for (size_t i = 0; i < ll.A.size(); i++) ll.A[i] += sign * eps * ll.zA[i];
        for (size_t i = 0; i < ll.B.size(); i++) ll.B[i] += sign * eps * ll.zB[i];
    }
}

// Compute cross-entropy loss from logits
static float compute_loss(
        llama_context * ctx,
        int n_vocab,
        const llama_token * tokens,
        int n_tokens) {
    float loss = 0.0f;
    int count = 0;

    for (int i = 0; i < n_tokens - 1; i++) {
        float * logits = llama_get_logits_ith(ctx, i);
        if (!logits) continue;

        int label = tokens[i + 1];
        if (label < 0 || label >= n_vocab) continue;

        float max_val = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_val) max_val = logits[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            sum_exp += expf(logits[v] - max_val);
        }
        float log_prob = (logits[label] - max_val) - logf(sum_exp);
        loss -= log_prob;
        count++;
    }

    return count > 0 ? loss / count : 0.0f;
}

// Run forward pass with current LoRA adapter applied
static float forward_with_lora(
        alpha_lora_trainer * trainer,
        const llama_token * tokens,
        int n_tokens) {
    // Sync CPU params → adapter tensors
    sync_to_adapter(trainer);

    // Clear memory for clean forward
    llama_memory_clear(llama_get_memory(trainer->ctx), true);

    // Build batch with logits enabled for all positions (needed for loss)
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

    if (rc != 0) {
        fprintf(stderr, "forward_with_lora: decode failed (%d)\n", rc);
        return -1.0f;
    }

    const llama_vocab * vocab = llama_model_get_vocab(trainer->model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    return compute_loss(trainer->ctx, n_vocab, tokens, n_tokens);
}

alpha_lora_trainer * alpha_lora_trainer_init(
        struct llama_model * model,
        struct llama_context * ctx,
        alpha_lora_train_config config) {
    if (!model || !ctx) return nullptr;

    auto * trainer = new alpha_lora_trainer();
    trainer->config = config;
    trainer->model = model;
    trainer->ctx = ctx;
    trainer->step = 0;
    trainer->current_loss = 0;
    trainer->total_loss = 0;
    trainer->n_steps = 0;
    trainer->rng.seed(42);
    trainer->adapter = nullptr;
    trainer->adapter_ctx = nullptr;
    trainer->adapter_buf = nullptr;

    int rank = config.rank > 0 ? config.rank : 32;
    int n_layer = (int)model->layers.size();

    // Identify target tensors for LoRA (Q and V projections)
    struct target_info {
        ggml_tensor * W;
        std::string name;
    };
    std::vector<target_info> targets;

    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model->layers[il];
        struct { ggml_tensor * W; const char * suffix; } tgts[] = {
            { layer.wq, "attn_q" },
            { layer.wv, "attn_v" },
        };
        for (auto & t : tgts) {
            if (!t.W) continue;
            target_info ti;
            ti.W = t.W;
            ti.name = std::string("blk.") + std::to_string(il) + "." + t.suffix;
            targets.push_back(ti);
        }
    }

    if (targets.empty()) {
        fprintf(stderr, "adapter: no trainable layers found\n");
        delete trainer;
        return nullptr;
    }

    // Create adapter and allocate ggml tensors for A/B
    trainer->adapter = new llama_adapter_lora(model);
    trainer->adapter->alpha = (float)rank;

    size_t ctx_size = ggml_tensor_overhead() * targets.size() * 2 + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    trainer->adapter_ctx = ggml_init(params);

    // Find CPU buffer type
    ggml_backend_buffer_type_t cpu_buft = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_buft = ggml_backend_dev_buffer_type(dev);
            break;
        }
    }

    // Create LoRA layers
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)rank));

    for (auto & ti : targets) {
        int64_t in_dim = ti.W->ne[0];
        int64_t out_dim = ti.W->ne[1];

        lora_layer ll = {};
        ll.name = ti.name;
        ll.in_dim = in_dim;
        ll.out_dim = out_dim;
        ll.rank = rank;

        int nA = (int)(in_dim * rank);
        int nB = (int)(rank * out_dim);

        ll.A.resize(nA);
        ll.B.resize(nB, 0.0f);
        ll.mA.resize(nA, 0.0f);
        ll.vA.resize(nA, 0.0f);
        ll.mB.resize(nB, 0.0f);
        ll.vB.resize(nB, 0.0f);
        ll.zA.resize(nA, 0.0f);
        ll.zB.resize(nB, 0.0f);

        // A = random, B = zero
        for (int i = 0; i < nA; i++) ll.A[i] = dist(trainer->rng);

        // Create adapter tensors: a=[in_dim, rank], b=[rank, out_dim]
        char name_buf[128];
        snprintf(name_buf, sizeof(name_buf), "%s.lora_a", ll.name.c_str());
        ll.adapter_a = ggml_new_tensor_2d(trainer->adapter_ctx, GGML_TYPE_F32, in_dim, rank);
        ggml_set_name(ll.adapter_a, name_buf);

        snprintf(name_buf, sizeof(name_buf), "%s.lora_b", ll.name.c_str());
        ll.adapter_b = ggml_new_tensor_2d(trainer->adapter_ctx, GGML_TYPE_F32, rank, out_dim);
        ggml_set_name(ll.adapter_b, name_buf);

        // Register in adapter's ab_map using the base tensor name
        std::string base_name(ti.W->name);
        trainer->adapter->ab_map[base_name] = llama_adapter_lora_weight(ll.adapter_a, ll.adapter_b);

        trainer->layers.push_back(std::move(ll));
    }

    // Allocate buffer for adapter tensors
    trainer->adapter_buf = ggml_backend_alloc_ctx_tensors_from_buft(trainer->adapter_ctx, cpu_buft);
    if (!trainer->adapter_buf) {
        fprintf(stderr, "adapter: failed to allocate adapter buffer\n");
        delete trainer->adapter;
        ggml_free(trainer->adapter_ctx);
        delete trainer;
        return nullptr;
    }

    // Initialize adapter tensors from CPU data
    for (auto & ll : trainer->layers) {
        ggml_backend_tensor_set(ll.adapter_a, ll.A.data(), 0, ll.A.size() * sizeof(float));
        ggml_backend_tensor_set(ll.adapter_b, ll.B.data(), 0, ll.B.size() * sizeof(float));
    }

    // Activate the adapter on the context
    llama_adapter_lora * adapters[] = { trainer->adapter };
    float scales[] = { 1.0f };
    llama_set_adapters_lora(ctx, adapters, 1, scales);

    // Count total trainable params
    int64_t total_params = 0;
    for (const auto & ll : trainer->layers) {
        total_params += (int64_t)ll.A.size() + (int64_t)ll.B.size();
    }

    fprintf(stderr, "adapter: %zu LoRA layers, rank=%d, %.2f M trainable params\n",
            trainer->layers.size(), rank, (float)total_params / 1e6f);
    fprintf(stderr, "adapter: optimizer configured (eps=%.1e, lr=%.1e)\n",
            config.mezo_eps > 0 ? config.mezo_eps : 1e-3f,
            config.lr > 0 ? config.lr : 1e-4f);

    return trainer;
}

float alpha_lora_train_step(
        alpha_lora_trainer * trainer,
        const llama_token * tokens,
        int n_tokens) {
    if (!trainer || !tokens || n_tokens < 2) return -1.0f;

    float eps = trainer->config.mezo_eps > 0 ? trainer->config.mezo_eps : 1e-3f;
    float lr = trainer->config.lr > 0 ? trainer->config.lr : 1e-4f;

    // MeZO step:
    // 1. Sample random direction z ~ N(0,1)
    mezo_sample_z(trainer);

    // 2. Perturb: theta + eps*z, forward, get loss+
    mezo_perturb(trainer, +1.0f);
    float loss_plus = forward_with_lora(trainer, tokens, n_tokens);

    // 3. Perturb: theta - 2*eps*z (net: theta - eps*z), forward, get loss-
    mezo_perturb(trainer, -2.0f);
    float loss_minus = forward_with_lora(trainer, tokens, n_tokens);

    // 4. Restore: theta - eps*z + eps*z = theta
    mezo_perturb(trainer, +1.0f);

    if (loss_plus < 0 || loss_minus < 0) {
        fprintf(stderr, "train step: forward pass failed\n");
        return -1.0f;
    }

    // 5. Projected gradient
    float g_proj = (loss_plus - loss_minus) / (2.0f * eps);

    // 6. Adam update using grad ≈ g_proj * z
    trainer->step++;
    float beta1 = trainer->config.beta1 > 0 ? trainer->config.beta1 : 0.9f;
    float beta2 = trainer->config.beta2 > 0 ? trainer->config.beta2 : 0.999f;
    float adam_eps = trainer->config.eps > 0 ? trainer->config.eps : 1e-8f;
    float wd = trainer->config.weight_decay;
    int t = trainer->step;

    float effective_lr = lr;
    int warmup = trainer->config.warmup_steps > 0 ? trainer->config.warmup_steps : 10;
    if (t < warmup) {
        effective_lr = lr * ((float)t / warmup);
    }

    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);

    for (auto & ll : trainer->layers) {
        for (size_t i = 0; i < ll.A.size(); i++) {
            float g = g_proj * ll.zA[i] + wd * ll.A[i];
            ll.mA[i] = beta1 * ll.mA[i] + (1 - beta1) * g;
            ll.vA[i] = beta2 * ll.vA[i] + (1 - beta2) * g * g;
            float m_hat = ll.mA[i] / bc1;
            float v_hat = ll.vA[i] / bc2;
            ll.A[i] -= effective_lr * m_hat / (sqrtf(v_hat) + adam_eps);
        }
        for (size_t i = 0; i < ll.B.size(); i++) {
            float g = g_proj * ll.zB[i] + wd * ll.B[i];
            ll.mB[i] = beta1 * ll.mB[i] + (1 - beta1) * g;
            ll.vB[i] = beta2 * ll.vB[i] + (1 - beta2) * g * g;
            float m_hat = ll.mB[i] / bc1;
            float v_hat = ll.vB[i] / bc2;
            ll.B[i] -= effective_lr * m_hat / (sqrtf(v_hat) + adam_eps);
        }
    }

    float loss = (loss_plus + loss_minus) / 2.0f;
    trainer->current_loss = loss;
    trainer->total_loss += loss;
    trainer->n_steps++;

    return loss;
}

void alpha_lora_print_stats(const alpha_lora_trainer * trainer) {
    if (!trainer) return;
    fprintf(stderr, "Training: step=%d, loss=%.4f, avg_loss=%.4f, layers=%zu, rank=%d\n",
            trainer->step, trainer->current_loss,
            trainer->n_steps > 0 ? trainer->total_loss / trainer->n_steps : 0.0f,
            trainer->layers.size(), trainer->config.rank > 0 ? trainer->config.rank : 32);
}

void alpha_lora_save_gguf(const alpha_lora_trainer * trainer, const char * path) {
    if (!trainer || !path) return;

    int rank = trainer->config.rank > 0 ? trainer->config.rank : 32;
    float alpha = (float)rank;

    struct gguf_context * gguf = gguf_init_empty();

    gguf_set_val_str(gguf, "general.type", "adapter");
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_str(gguf, "adapter.type", "lora");
    gguf_set_val_f32(gguf, "adapter.lora.alpha", alpha);

    // Create tensor context with data (no_alloc=false so tensors get data pointers)
    // Calculate total buffer needed for all A and B tensors
    size_t total_data = 0;
    for (const auto & ll : trainer->layers) {
        total_data += ll.A.size() * sizeof(float);
        total_data += ll.B.size() * sizeof(float);
    }

    size_t ctx_size = ggml_tensor_overhead() * trainer->layers.size() * 2 + total_data + 1024;
    std::vector<uint8_t> ctx_buf(ctx_size);
    struct ggml_init_params params = { ctx_size, ctx_buf.data(), false };  // no_alloc=false
    struct ggml_context * ctx = ggml_init(params);

    for (const auto & ll : trainer->layers) {
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

    // Write complete GGUF file (header + tensor data, offsets computed correctly)
    gguf_write_to_file(gguf, path, false);  // false = write data too

    ggml_free(ctx);
    gguf_free(gguf);

    fprintf(stderr, "saved %zu adapter layers (rank=%d) to %s\n",
            trainer->layers.size(), rank, path);
}

void alpha_lora_trainer_free(alpha_lora_trainer * trainer) {
    if (!trainer) return;

    // Remove adapter from context
    llama_set_adapters_lora(trainer->ctx, nullptr, 0, nullptr);

    if (trainer->adapter_buf) ggml_backend_buffer_free(trainer->adapter_buf);
    if (trainer->adapter_ctx) ggml_free(trainer->adapter_ctx);
    // Don't delete adapter->ab_map tensors — they're owned by adapter_ctx
    // But we need to clear the map before deleting the adapter
    if (trainer->adapter) {
        trainer->adapter->ab_map.clear();
        delete trainer->adapter;
    }
    delete trainer;
}

// Load training data
static std::vector<std::vector<llama_token>> load_training_data(
        const llama_vocab * vocab,
        const char * path,
        int seq_len) {
    std::vector<std::vector<llama_token>> sequences;

    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "load_training_data: failed to open %s\n", path);
        return sequences;
    }

    std::string text;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        text += line + "\n";
    }
    fin.close();

    if (text.empty()) {
        fprintf(stderr, "load_training_data: empty file %s\n", path);
        return sequences;
    }

    // Tokenize entire text
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

    fprintf(stderr, "load_training_data: %d tokens from %s\n", n_tokens, path);

    // Split into sequences
    for (int i = 0; i + seq_len <= n_tokens; i += seq_len) {
        std::vector<llama_token> seq(all_tokens.begin() + i, all_tokens.begin() + i + seq_len);
        sequences.push_back(std::move(seq));
    }

    int remainder = n_tokens % seq_len;
    if (remainder > 16) {
        int start = n_tokens - remainder;
        std::vector<llama_token> seq(all_tokens.begin() + start, all_tokens.end());
        sequences.push_back(std::move(seq));
    }

    fprintf(stderr, "load_training_data: %zu sequences of length %d\n",
            sequences.size(), seq_len);
    return sequences;
}

int alpha_lora_train(
        struct llama_model * model,
        struct llama_context * ctx,
        const char * data_file,
        const char * output_file,
        alpha_lora_train_config config,
        int epochs,
        int seq_len) {
    if (!model || !ctx || !data_file) return 1;

    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto sequences = load_training_data(vocab, data_file, seq_len > 0 ? seq_len : 256);
    if (sequences.empty()) {
        fprintf(stderr, "train: no training data loaded\n");
        return 1;
    }

    alpha_lora_trainer * trainer = alpha_lora_trainer_init(model, ctx, config);
    if (!trainer) {
        fprintf(stderr, "train: failed to initialize trainer\n");
        return 1;
    }

    if (epochs <= 0) epochs = 1;

    fprintf(stderr, "\n=== Training ===\n");
    fprintf(stderr, "  Data: %s (%zu sequences x %d tokens)\n", data_file, sequences.size(), seq_len);
    fprintf(stderr, "  Epochs: %d\n", epochs);
    fprintf(stderr, "  LR: %.1e, Rank: %d\n", config.lr > 0 ? config.lr : 1e-4f,
            config.rank > 0 ? config.rank : 32);
    fprintf(stderr, "  Output: %s\n", output_file ? output_file : "(none)");
    fprintf(stderr, "============================\n\n");

    for (int epoch = 0; epoch < epochs; epoch++) {
        float epoch_loss = 0;
        int epoch_steps = 0;

        for (size_t s = 0; s < sequences.size(); s++) {
            float loss = alpha_lora_train_step(trainer, sequences[s].data(), (int)sequences[s].size());
            if (loss < 0) {
                fprintf(stderr, "Training step failed at epoch %d, seq %zu\n", epoch, s);
                continue;
            }

            epoch_loss += loss;
            epoch_steps++;

            if (epoch_steps % 10 == 0 || epoch_steps == 1) {
                fprintf(stderr, "  epoch %d/%d, step %d/%zu, loss: %.4f, avg: %.4f\n",
                        epoch + 1, epochs, epoch_steps, sequences.size(),
                        loss, epoch_loss / epoch_steps);
            }
        }

        if (epoch_steps > 0) {
            fprintf(stderr, "Epoch %d/%d complete. avg_loss=%.4f\n",
                    epoch + 1, epochs, epoch_loss / epoch_steps);
        }
    }

    if (output_file && output_file[0]) {
        alpha_lora_save_gguf(trainer, output_file);
    }

    alpha_lora_print_stats(trainer);
    alpha_lora_trainer_free(trainer);
    return 0;
}
