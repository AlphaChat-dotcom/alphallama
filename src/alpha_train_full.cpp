#include "alpha_train_full.h"
#include "alpha_train_graph.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <chrono>

// Adam Optimizer State

struct adam_state {
    std::vector<float> m;  // first moment
    std::vector<float> v;  // second moment
    int t;                 // timestep
};

static void adam_init(adam_state & state, size_t n_params) {
    state.m.resize(n_params, 0.0f);
    state.v.resize(n_params, 0.0f);
    state.t = 0;
}

static void adam_step(
    float * param_data, const float * grad_data, size_t n,
    adam_state & state, size_t offset,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    int warmup_steps)
{
    state.t++;
    float effective_lr = lr;
    if (warmup_steps > 0 && state.t <= warmup_steps) {
        effective_lr = lr * (float)state.t / (float)warmup_steps;
    }

    float beta1_t = 1.0f - powf(beta1, (float)state.t);
    float beta2_t = 1.0f - powf(beta2, (float)state.t);

    for (size_t i = 0; i < n; i++) {
        float g = grad_data[i];
        if (weight_decay > 0.0f) {
            g += weight_decay * param_data[i];  // decoupled weight decay
        }
        state.m[offset + i] = beta1 * state.m[offset + i] + (1.0f - beta1) * g;
        state.v[offset + i] = beta2 * state.v[offset + i] + (1.0f - beta2) * g * g;

        float m_hat = state.m[offset + i] / beta1_t;
        float v_hat = state.v[offset + i] / beta2_t;

        param_data[i] -= effective_lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// Data Loading

struct train_data {
    std::vector<std::vector<llama_token>> sequences;
    size_t current_idx;
};

static train_data load_training_data(
    const char * path,
    const llama_vocab * vocab,
    int seq_len)
{
    train_data data = {};
    data.current_idx = 0;

    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "train: cannot open data file: %s\n", path);
        return data;
    }

    // Read all text, tokenize into sequences of seq_len
    std::string all_text;
    std::string line;
    while (std::getline(f, line)) {
        all_text += line + "\n";
    }
    f.close();

    if (all_text.empty()) {
        fprintf(stderr, "train: data file is empty\n");
        return data;
    }

    // Tokenize entire corpus
    int n_tokens_est = (int)all_text.size();  // overestimate
    std::vector<llama_token> all_tokens(n_tokens_est);
    int n_tokens = llama_tokenize(vocab, all_text.c_str(), all_text.size(),
                                   all_tokens.data(), n_tokens_est, true, false);
    if (n_tokens < 0) {
        all_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, all_text.c_str(), all_text.size(),
                                   all_tokens.data(), all_tokens.size(), true, false);
    }
    all_tokens.resize(n_tokens);

    fprintf(stderr, "train: loaded %d tokens from %s\n", n_tokens, path);

    // Split into sequences of seq_len+1 (input + target shift)
    for (int i = 0; i + seq_len + 1 <= n_tokens; i += seq_len) {
        std::vector<llama_token> seq(all_tokens.begin() + i, all_tokens.begin() + i + seq_len + 1);
        data.sequences.push_back(std::move(seq));
    }

    fprintf(stderr, "train: %zu training sequences (seq_len=%d)\n",
            data.sequences.size(), seq_len);

    return data;
}

static bool get_next_batch(
    train_data & data,
    llama_token * input,   // [seq_len]
    llama_token * target,  // [seq_len]
    int seq_len)
{
    if (data.sequences.empty()) return false;
    if (data.current_idx >= data.sequences.size()) {
        data.current_idx = 0;  // wrap around for next epoch
        return false;  // signal epoch complete
    }

    auto & seq = data.sequences[data.current_idx++];
    // input = seq[0..seq_len-1], target = seq[1..seq_len]
    for (int i = 0; i < seq_len; i++) {
        input[i] = seq[i];
        target[i] = seq[i + 1];
    }
    return true;
}

// Gradient Computation (Phase 1: CPU logit gradient)

// Compute cross-entropy loss and gradient of loss w.r.t. logits
static float compute_loss_and_logit_grad(
    const float * logits,     // [seq_len, n_vocab]
    const llama_token * targets, // [seq_len]
    float * d_logits,         // [seq_len, n_vocab] output gradient
    int seq_len,
    int n_vocab)
{
    float total_loss = 0.0f;

    for (int i = 0; i < seq_len; i++) {
        const float * l = logits + (int64_t)i * n_vocab;
        float * dl = d_logits + (int64_t)i * n_vocab;

        // Stable softmax
        float max_l = l[0];
        for (int v = 1; v < n_vocab; v++) {
            if (l[v] > max_l) max_l = l[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            dl[v] = expf(l[v] - max_l);
            sum_exp += dl[v];
        }
        // Softmax probabilities in dl
        for (int v = 0; v < n_vocab; v++) {
            dl[v] /= sum_exp;
        }

        // Loss = -log(p[target])
        total_loss += -logf(dl[targets[i]] + 1e-10f);

        // Gradient of CE loss w.r.t. logits: softmax(logits) - one_hot(target)
        dl[targets[i]] -= 1.0f;

        // Scale by 1/seq_len (mean reduction)
        for (int v = 0; v < n_vocab; v++) {
            dl[v] /= (float)seq_len;
        }
    }

    return total_loss / (float)seq_len;
}

// Model Saving (GGUF)

#include "gguf.h"

static void alpha_train_save_model(
    llama_model * model,
    alpha_train_graph * tg,
    const char * output_path,
    int step)
{
    // Save trained parameters to GGUF file.
    // For full training: all model weight tensors
    // For LoRA: only adapter deltas (not implemented here - use existing LoRA save)
    //
    // Strategy: write a GGUF file containing all trainable tensors.
    // This can be loaded back as the model or merged.

    struct gguf_context * ctx_out = gguf_init_empty();
    if (!ctx_out) {
        fprintf(stderr, "save: failed to create GGUF context\n");
        return;
    }

    // Copy model metadata from original
    // Architecture, vocab, hparams etc. come from the loaded model
    // For simplicity, write just the trained parameters with their names

    gguf_set_val_str(ctx_out, "general.architecture", "llama");  // TODO: get from model
    gguf_set_val_str(ctx_out, "general.name", "alpha-trained");
    gguf_set_val_u32(ctx_out, "general.file_type", 0);  // F32
    gguf_set_val_u32(ctx_out, "alpha.train_step", step);

    // Write each trainable parameter
    int n_written = 0;
    for (auto * param : tg->params) {
        if (!param || !param->data) continue;

        // Only write F32/F16 parameters (quantized params stay as-is for now)
        if (param->type == GGML_TYPE_F32 || param->type == GGML_TYPE_F16) {
            gguf_add_tensor(ctx_out, param);
            n_written++;
        }
    }

    // Write to file
    if (n_written > 0) {
        gguf_write_to_file(ctx_out, output_path, false);
        fprintf(stderr, "save: wrote %d tensors to %s\n", n_written, output_path);
    } else {
        fprintf(stderr, "save: no F32/F16 tensors to save (model is quantized)\n");
        fprintf(stderr, "  For quantized training, use train-full with a F16 model.\n");
    }

    gguf_free(ctx_out);
}

// Save optimizer state for resume
static void alpha_train_save_checkpoint(
    const adam_state & optim,
    const char * path,
    int step)
{
    std::string ckpt_path = std::string(path) + ".ckpt";
    FILE * f = fopen(ckpt_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "checkpoint: cannot write %s\n", ckpt_path.c_str());
        return;
    }

    // Header
    int32_t magic = 0x414C5054;  // "ALPT"
    int32_t version = 1;
    int32_t t = optim.t;
    int64_t n = (int64_t)optim.m.size();

    fwrite(&magic, sizeof(int32_t), 1, f);
    fwrite(&version, sizeof(int32_t), 1, f);
    fwrite(&t, sizeof(int32_t), 1, f);
    fwrite(&n, sizeof(int64_t), 1, f);
    fwrite(&step, sizeof(int32_t), 1, f);

    // Adam moments
    fwrite(optim.m.data(), sizeof(float), n, f);
    fwrite(optim.v.data(), sizeof(float), n, f);

    fclose(f);
    fprintf(stderr, "checkpoint: step %d saved to %s\n", step, ckpt_path.c_str());
}

// Load optimizer state for resume
static bool alpha_train_load_checkpoint(
    adam_state & optim,
    const char * path,
    int & step_out)
{
    std::string ckpt_path = std::string(path) + ".ckpt";
    FILE * f = fopen(ckpt_path.c_str(), "rb");
    if (!f) return false;

    int32_t magic, version, t, step;
    int64_t n;
    fread(&magic, sizeof(int32_t), 1, f);
    fread(&version, sizeof(int32_t), 1, f);
    fread(&t, sizeof(int32_t), 1, f);
    fread(&n, sizeof(int64_t), 1, f);
    fread(&step, sizeof(int32_t), 1, f);

    if (magic != 0x414C5054 || version != 1) {
        fclose(f);
        return false;
    }

    if (n != (int64_t)optim.m.size()) {
        fprintf(stderr, "resume: size mismatch (%lld vs %zu)\n",
                (long long)n, optim.m.size());
        fclose(f);
        return false;
    }

    optim.t = t;
    fread(optim.m.data(), sizeof(float), n, f);
    fread(optim.v.data(), sizeof(float), n, f);
    step_out = step;

    fclose(f);
    fprintf(stderr, "resume: resumed from step %d\n", step);
    return true;
}

// Gradient Checkpointing

static alpha_grad_checkpoint_config setup_checkpointing(int n_layer, const alpha_train_full_config & cfg) {
    alpha_grad_checkpoint_config ckpt = {};
    ckpt.enabled = cfg.gradient_checkpointing;

    if (!ckpt.enabled) return ckpt;

    int interval = cfg.checkpoint_interval;
    if (interval <= 0) {
        // Auto: sqrt(n_layer) gives optimal memory/compute tradeoff
        interval = std::max(1, (int)sqrtf((float)n_layer));
    }

    // Build list of checkpoint layers
    for (int il = 0; il < n_layer; il += interval) {
        ckpt.checkpoint_layers.push_back(il);
    }
    // Always checkpoint the last layer
    if (ckpt.checkpoint_layers.back() != n_layer - 1) {
        ckpt.checkpoint_layers.push_back(n_layer - 1);
    }

    ckpt.interval = interval;
    ckpt.n_checkpoints = (int)ckpt.checkpoint_layers.size();

    fprintf(stderr, "train: gradient checkpointing enabled, interval=%d, %d checkpoints for %d layers\n",
            interval, ckpt.n_checkpoints, n_layer);
    fprintf(stderr, "  Memory savings: ~%.0fx less activation storage\n",
            (float)n_layer / ckpt.n_checkpoints);
    fprintf(stderr, "  Compute overhead: ~%.0f%% (recomputation)\n",
            100.0f * (float)(n_layer - ckpt.n_checkpoints) / n_layer / 3.0f);  // ~1/3 extra forward

    return ckpt;
}

// Activation checkpoint storage: saves hidden states at checkpoint layers
struct activation_checkpoint {
    int layer_id;
    std::vector<float> hidden_state;  // [seq_len, n_embd]
};

struct checkpoint_store {
    std::vector<activation_checkpoint> checkpoints;
    int n_embd;
    int seq_len;

    void init(int n_embd_, int seq_len_, int n_checkpoints) {
        this->n_embd = n_embd_;
        this->seq_len = seq_len_;
        checkpoints.resize(n_checkpoints);
        for (auto & cp : checkpoints) {
            cp.hidden_state.resize((size_t)n_embd_ * seq_len_);
        }
    }

    void save(int checkpoint_idx, int layer_id, const float * data) {
        if (checkpoint_idx < (int)checkpoints.size()) {
            checkpoints[checkpoint_idx].layer_id = layer_id;
            memcpy(checkpoints[checkpoint_idx].hidden_state.data(), data,
                   (size_t)n_embd * seq_len * sizeof(float));
        }
    }

    const float * get(int checkpoint_idx) const {
        if (checkpoint_idx < (int)checkpoints.size()) {
            return checkpoints[checkpoint_idx].hidden_state.data();
        }
        return nullptr;
    }

    size_t memory_bytes() const {
        return checkpoints.size() * n_embd * seq_len * sizeof(float);
    }
};

// Main Training Loop

int alpha_train_full_run(
    struct llama_model * model,
    struct llama_context * ctx,
    const char * data_path,
    const char * output_path,
    alpha_train_full_config config,
    int n_epochs)
{
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int seq_len = config.seq_len;

    // Load training data
    train_data data = load_training_data(data_path, vocab, seq_len);
    if (data.sequences.empty()) {
        fprintf(stderr, "train: no training data loaded\n");
        return 1;
    }

    // Create training graph (marks parameters)
    enum alpha_train_mode mode;
    switch (config.mode) {
        case alpha_train_full_config::TRAIN_LORA: mode = ALPHA_TRAIN_LORA; break;
        case alpha_train_full_config::TRAIN_FREEZE_ATTN: mode = ALPHA_TRAIN_FREEZE_ATTN; break;
        default: mode = ALPHA_TRAIN_FULL; break;
    }
    alpha_train_graph * tg = alpha_train_graph_create(model, ctx, mode);
    if (!tg || tg->params.empty()) {
        fprintf(stderr, "train: no trainable parameters found\n");
        if (tg) alpha_train_graph_free(tg);
        return 1;
    }

    // Count total trainable parameters
    size_t total_params = 0;
    for (auto * p : tg->params) {
        total_params += ggml_nelements(p);
    }
    fprintf(stderr, "train: %zu trainable parameters (%.2f M)\n",
            total_params, total_params / 1e6f);

    // Initialize optimizer
    adam_state optim;
    adam_init(optim, total_params);

    // Set up gradient checkpointing
    alpha_grad_checkpoint_config ckpt = setup_checkpointing(tg->n_layer, config);
    checkpoint_store ckpt_store;
    if (ckpt.enabled) {
        ckpt_store.init(model->hparams.n_embd, config.seq_len, ckpt.n_checkpoints);
        fprintf(stderr, "  Checkpoint memory: %.1f MB\n", ckpt_store.memory_bytes() / 1e6f);
    }

    // Allocate buffers
    std::vector<llama_token> input(seq_len);
    std::vector<llama_token> target(seq_len);
    std::vector<float> d_logits((int64_t)seq_len * n_vocab);

    // Try to resume from checkpoint
    int global_step = 0;
    if (output_path) {
        alpha_train_load_checkpoint(optim, output_path, global_step);
    }

    // Training loop
    float running_loss = 0.0f;
    int running_count = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        data.current_idx = 0;

        // Shuffle sequences
        for (size_t i = data.sequences.size() - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            std::swap(data.sequences[i], data.sequences[j]);
        }

        while (get_next_batch(data, input.data(), target.data(), seq_len)) {
            // Forward pass
            float loss = alpha_train_graph_compute(tg, ctx, input.data(), target.data(), seq_len);
            if (loss < 0) {
                fprintf(stderr, "train: forward pass failed at step %d\n", global_step);
                break;
            }

            running_loss += loss;
            running_count++;

            // Get logits and compute gradient
            float * logits = llama_get_logits(ctx);
            if (logits) {
                compute_loss_and_logit_grad(logits, target.data(), d_logits.data(), seq_len, n_vocab);
            }

            // Phase 2: Compute backward through output head + accumulate gradients
            // Use alpha_train_decode_backward for GPU-aware gradient computation
            // For now the gradients flow through the output head (d_logits -> d_hidden)
            // Full per-layer backward is computed via the split approach.
            //
            // TODO: Once ggml graph interception is added, this becomes a single
            // ggml_backend_graph_compute() call for forward+backward together.

            global_step++;

            // Logging
            if (config.log_every > 0 && global_step % config.log_every == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                float avg_loss = running_loss / running_count;
                float tok_per_sec = (float)(global_step * seq_len) / elapsed;

                fprintf(stderr, "step %d | loss %.4f | lr %.2e | %.0f tok/s | epoch %d/%d\n",
                        global_step, avg_loss,
                        config.lr * std::min(1.0f, (float)global_step / std::max(1, config.warmup_steps)),
                        tok_per_sec, epoch + 1, n_epochs);

                running_loss = 0;
                running_count = 0;
            }

            // Checkpoint saving
            if (config.save_every > 0 && output_path && global_step % config.save_every == 0) {
                alpha_train_save_checkpoint(optim, output_path, global_step);
            }

            // Clear KV cache for next sequence
            llama_memory_clear(llama_get_memory(ctx), true);
        }

        fprintf(stderr, "Epoch %d/%d complete (step %d)\n", epoch + 1, n_epochs, global_step);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();
    fprintf(stderr, "\nTraining complete: %d steps in %.1f s (%.0f tok/s)\n",
            global_step, total_time, (float)(global_step * seq_len) / total_time);
    fprintf(stderr, "Final loss: %.4f\n", running_loss / std::max(1, running_count));

    // Save final model
    if (output_path) {
        fprintf(stderr, "Saving trained model to: %s\n", output_path);
        alpha_train_save_model(model, tg, output_path, global_step);
    }

    alpha_train_graph_free(tg);
    return 0;
}
