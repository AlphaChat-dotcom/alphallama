#include "alpha_mole_train_pipeline.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

struct alpha_train_pipeline {
    llama_model * model;
    alpha_pipeline_config config;

    int n_layer;
    int n_embd;
    int n_checkpoints;       // sqrt(N) segments

    // Gradient checkpoint storage (CPU pinned memory)
    // Only checkpoint_indices layers are saved; others are recomputed
    std::vector<int> checkpoint_indices;  // which layers to checkpoint
    std::vector<std::vector<float>> checkpoints; // [n_layer] (sparse)

    // VRAM budget tracking
    size_t vram_per_layer;   // bytes per layer's weights (Q4)
    size_t vram_lora;        // bytes for all LoRA params
    size_t vram_grad;        // bytes for gradient workspace
    size_t vram_activations; // bytes for checkpoint activations in VRAM
    size_t vram_total;       // total VRAM needed
};

alpha_train_pipeline * alpha_pipeline_init(
    struct llama_model * model,
    alpha_pipeline_config config) {

    if (!model) return nullptr;

    auto * pipe = new alpha_train_pipeline();
    pipe->model  = model;
    pipe->config = config;
    pipe->n_layer = (int)model->layers.size();
    pipe->n_embd  = model->hparams.n_embd;

    // Compute layer sizes for VRAM budgeting
    pipe->vram_per_layer = 0;
    if (pipe->n_layer > 0) {
        const auto & layer = model->layers[0];
        // Sum all weight tensors in one layer
        ggml_tensor * tensors[] = {
            layer.wq, layer.wk, layer.wv, layer.wo,
            layer.ffn_gate, layer.ffn_up, layer.ffn_down,
            layer.attn_norm, layer.ffn_norm,
        };
        for (auto * t : tensors) {
            if (t) pipe->vram_per_layer += ggml_nbytes(t);
        }
    }

    // Gradient checkpointing: save activations at sqrt(N) boundaries
    int n_seg = config.checkpoint_segments;
    if (n_seg <= 0) {
        n_seg = std::max(1, (int)sqrtf((float)pipe->n_layer));
    }
    pipe->n_checkpoints = n_seg;

    // Determine which layers to checkpoint
    int interval = std::max(1, pipe->n_layer / n_seg);
    for (int i = 0; i < pipe->n_layer; i += interval) {
        pipe->checkpoint_indices.push_back(i);
    }
    // Always checkpoint the last layer
    if (pipe->checkpoint_indices.empty() ||
        pipe->checkpoint_indices.back() != pipe->n_layer - 1) {
        pipe->checkpoint_indices.push_back(pipe->n_layer - 1);
    }

    pipe->checkpoints.resize(pipe->n_layer);

    // VRAM budget analysis
    pipe->vram_lora = 0; // computed later when LoRA is created
    pipe->vram_grad = pipe->n_embd * 256 * sizeof(float); // one layer gradient workspace
    pipe->vram_activations = pipe->checkpoint_indices.size() * pipe->n_embd * 256 * sizeof(float);
    pipe->vram_total = 2 * pipe->vram_per_layer  // double-buffered
                     + pipe->vram_grad
                     + pipe->vram_activations
                     + 50 * 1024 * 1024; // 50 MB scratch

    fprintf(stderr, "pipeline: %d layers, %d checkpoints (every %d layers)\n",
            pipe->n_layer, (int)pipe->checkpoint_indices.size(), interval);
    fprintf(stderr, "  per-layer weights: %.1f MB\n", (float)pipe->vram_per_layer / (1024*1024));
    fprintf(stderr, "  VRAM estimate: %.1f MB (2 layers + grads + checkpoints + scratch)\n",
            (float)pipe->vram_total / (1024*1024));

    return pipe;
}

int alpha_pipeline_n_checkpoints(const alpha_train_pipeline * pipe) {
    return pipe ? pipe->n_checkpoints : 0;
}

bool alpha_pipeline_is_checkpoint(const alpha_train_pipeline * pipe, int layer) {
    if (!pipe) return false;
    for (int idx : pipe->checkpoint_indices) {
        if (idx == layer) return true;
    }
    return false;
}

void alpha_pipeline_save_checkpoint(
    alpha_train_pipeline * pipe,
    int layer,
    const float * data,
    int n_elements) {
    if (!pipe || layer < 0 || layer >= pipe->n_layer || !data) return;

    pipe->checkpoints[layer].resize(n_elements);
    memcpy(pipe->checkpoints[layer].data(), data, n_elements * sizeof(float));
}

const float * alpha_pipeline_load_checkpoint(
    alpha_train_pipeline * pipe,
    int layer) {
    if (!pipe || layer < 0 || layer >= pipe->n_layer) return nullptr;
    if (pipe->checkpoints[layer].empty()) return nullptr;
    return pipe->checkpoints[layer].data();
}

void alpha_pipeline_print_budget(const alpha_train_pipeline * pipe) {
    if (!pipe) return;

    fprintf(stderr, "\n=== Training Pipeline VRAM Budget ===\n");
    fprintf(stderr, "  Double-buffered layers: %.1f MB (2 × %.1f MB)\n",
            2.0f * pipe->vram_per_layer / (1024*1024),
            (float)pipe->vram_per_layer / (1024*1024));
    fprintf(stderr, "  Gradient workspace:     %.1f MB\n",
            (float)pipe->vram_grad / (1024*1024));
    fprintf(stderr, "  Activation checkpoints: %.1f MB (%d saves)\n",
            (float)pipe->vram_activations / (1024*1024),
            (int)pipe->checkpoint_indices.size());
    fprintf(stderr, "  Scratch:                50.0 MB\n");
    fprintf(stderr, "  TOTAL:                  %.1f MB\n",
            (float)pipe->vram_total / (1024*1024));

    size_t budget = pipe->config.vram_budget_mb * 1024 * 1024;
    if (budget > 0) {
        if (pipe->vram_total <= budget) {
            fprintf(stderr, "  Status: FITS in %.0f MB budget ✓\n",
                    (float)budget / (1024*1024));
        } else {
            fprintf(stderr, "  Status: EXCEEDS %.0f MB budget by %.1f MB ✗\n",
                    (float)budget / (1024*1024),
                    (float)(pipe->vram_total - budget) / (1024*1024));
        }
    }
    fprintf(stderr, "=====================================\n\n");
}

// NVMe tier: spill/load optimizer states

void alpha_pipeline_spill_optim(alpha_train_pipeline * pipe, int layer,
    const float * m_data, const float * v_data, int n_params) {
    if (!pipe || !pipe->config.tier2_path || layer < 0) return;

    char path_m[512], path_v[512];
    snprintf(path_m, sizeof(path_m), "%s/optim_m_layer%d.bin", pipe->config.tier2_path, layer);
    snprintf(path_v, sizeof(path_v), "%s/optim_v_layer%d.bin", pipe->config.tier2_path, layer);

    FILE * fm = fopen(path_m, "wb");
    if (fm) {
        fwrite(m_data, sizeof(float), n_params, fm);
        fclose(fm);
    }
    FILE * fv = fopen(path_v, "wb");
    if (fv) {
        fwrite(v_data, sizeof(float), n_params, fv);
        fclose(fv);
    }
}

bool alpha_pipeline_load_optim(alpha_train_pipeline * pipe, int layer,
    float * m_data, float * v_data, int n_params) {
    if (!pipe || !pipe->config.tier2_path || layer < 0) return false;

    char path_m[512], path_v[512];
    snprintf(path_m, sizeof(path_m), "%s/optim_m_layer%d.bin", pipe->config.tier2_path, layer);
    snprintf(path_v, sizeof(path_v), "%s/optim_v_layer%d.bin", pipe->config.tier2_path, layer);

    FILE * fm = fopen(path_m, "rb");
    if (!fm) return false;
    size_t read_m = fread(m_data, sizeof(float), n_params, fm);
    fclose(fm);

    FILE * fv = fopen(path_v, "rb");
    if (!fv) return false;
    size_t read_v = fread(v_data, sizeof(float), n_params, fv);
    fclose(fv);

    return (int)read_m == n_params && (int)read_v == n_params;
}

void alpha_pipeline_free(alpha_train_pipeline * pipe) {
    if (!pipe) return;
    delete pipe;
}
