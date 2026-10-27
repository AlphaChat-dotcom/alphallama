#include "alpha_train_pipeline.h"
#include "llama-model.h"

#include <cstdio>
#include <cmath>
#include <algorithm>

alpha_train_pipeline * alpha_train_pipeline_create(
    struct llama_model * model,
    const struct alpha_hw_plan * hw_plan,
    alpha_train_pipeline_config config)
{
    if (!model || !hw_plan) return nullptr;

    auto * pipe = new alpha_train_pipeline();
    pipe->n_layers_total = model->hparams.n_layer;
    pipe->n_embd = model->hparams.n_embd;

    // Determine number of stages
    int n_stages = config.n_stages;
    if (n_stages <= 0) {
        // Auto: one stage per GPU device in the hw plan
        n_stages = 0;
        for (int i = 0; i < hw_plan->n_devices; i++) {
            if (hw_plan->devices[i]) {
                enum ggml_backend_dev_type dt = ggml_backend_dev_type(hw_plan->devices[i]);
                if (dt != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    n_stages++;
                }
            }
        }
        if (n_stages == 0) n_stages = 1;  // CPU-only: single stage
    }

    // Assign layers to stages proportional to device VRAM
    std::vector<size_t> stage_vram(n_stages, 0);
    size_t total_vram = 0;
    int gpu_idx = 0;
    for (int i = 0; i < hw_plan->n_devices && gpu_idx < n_stages; i++) {
        if (!hw_plan->devices[i]) break;
        enum ggml_backend_dev_type dt = ggml_backend_dev_type(hw_plan->devices[i]);
        if (dt == GGML_BACKEND_DEVICE_TYPE_CPU) continue;

        size_t free = 0, total = 0;
        ggml_backend_dev_memory(hw_plan->devices[i], &free, &total);
        stage_vram[gpu_idx] = free / (1024 * 1024);  // MB
        total_vram += stage_vram[gpu_idx];
        gpu_idx++;
    }

    if (total_vram == 0) {
        // CPU-only fallback
        stage_vram[0] = 1;
        total_vram = 1;
    }

    // Distribute layers proportionally
    int layers_assigned = 0;
    pipe->stages.resize(n_stages);
    gpu_idx = 0;
    for (int s = 0; s < n_stages; s++) {
        pipe->stages[s].device_idx = s;
        pipe->stages[s].layer_start = layers_assigned;
        pipe->stages[s].vram_budget_mb = stage_vram[s];
        pipe->stages[s].backend = nullptr;  // set during execution

        int n_layers_this_stage;
        if (s == n_stages - 1) {
            // Last stage gets remaining layers
            n_layers_this_stage = pipe->n_layers_total - layers_assigned;
        } else {
            float proportion = (float)stage_vram[s] / (float)total_vram;
            n_layers_this_stage = std::max(1, (int)(pipe->n_layers_total * proportion));
        }
        pipe->stages[s].layer_end = layers_assigned + n_layers_this_stage - 1;
        layers_assigned += n_layers_this_stage;
    }

    // Ensure all layers are covered
    pipe->stages[n_stages - 1].layer_end = pipe->n_layers_total - 1;

    // Allocate communication buffers
    int micro_batch_tokens = config.micro_batch_size > 0 ? config.micro_batch_size : 1;
    size_t buf_size = (size_t)micro_batch_tokens * pipe->n_embd;

    pipe->fwd_buffers.resize(n_stages - 1);
    pipe->bwd_buffers.resize(n_stages - 1);
    for (int i = 0; i < n_stages - 1; i++) {
        pipe->fwd_buffers[i].resize(buf_size, 0.0f);
        pipe->bwd_buffers[i].resize(buf_size, 0.0f);
    }

    // Pipeline schedule: 1F1B
    int n_micro = config.n_micro_batches > 0 ? config.n_micro_batches : n_stages * 2;
    pipe->n_warmup = n_stages - 1;
    pipe->n_steady = n_micro - n_stages + 1;
    pipe->n_cooldown = n_stages - 1;

    return pipe;
}

void alpha_train_pipeline_free(alpha_train_pipeline * pipe) {
    delete pipe;
}

void alpha_train_pipeline_print(const alpha_train_pipeline * pipe) {
    if (!pipe) return;

    fprintf(stderr, "\n=== Training Pipeline ===\n");
    fprintf(stderr, "  Stages: %zu, Layers: %d, Embd: %d\n",
            pipe->stages.size(), pipe->n_layers_total, pipe->n_embd);

    for (int s = 0; s < (int)pipe->stages.size(); s++) {
        auto & st = pipe->stages[s];
        int n_layers = st.layer_end - st.layer_start + 1;
        fprintf(stderr, "  Stage %d: GPU[%d] layers %d-%d (%d layers, %zu MB VRAM)\n",
                s, st.device_idx, st.layer_start, st.layer_end, n_layers, st.vram_budget_mb);
    }

    fprintf(stderr, "  Schedule: warmup=%d, steady=%d, cooldown=%d\n",
            pipe->n_warmup, pipe->n_steady, pipe->n_cooldown);

    size_t buf_bytes = 0;
    for (auto & b : pipe->fwd_buffers) buf_bytes += b.size() * sizeof(float);
    buf_bytes *= 2;  // fwd + bwd
    fprintf(stderr, "  Communication buffers: %.1f MB\n", buf_bytes / 1e6f);
    fprintf(stderr, "\n");
}

float alpha_train_pipeline_step(
    alpha_train_pipeline * pipe,
    struct llama_model * model,
    struct llama_context * ctx,
    const llama_token * tokens,
    const llama_token * targets,
    int total_tokens)
{
    if (!pipe || pipe->stages.empty()) return -1.0f;

    // For single-stage pipeline (1 GPU): just run the normal training path
    if (pipe->stages.size() == 1) {
        // Delegate to standard training (no pipeline overhead)
        // This is handled by the caller (alpha_train_full_run)
        return 0.0f;
    }

    // Multi-stage 1F1B pipeline execution:
    //
    // Timeline (3 stages, 6 micro-batches):
    //   Stage 0: F0 F1 F2  B3 F3 B4 F4 B5 F5  B0 B1 B2
    //   Stage 1:    F0 F1  B3 F2 B4 F3 B5 F4      B0 B1
    //   Stage 2:       F0  B3 F1 B4 F2 B5 F3         B0
    //               warmup |--- steady ---|  cooldown
    //
    // Each "F" = forward through this stage's layers
    // Each "B" = backward through this stage's layers
    // Between stages: send/recv hidden state or gradient via DMA buffer

    // TODO: implement the 1F1B schedule. It requires:
    // 1. Splitting the model's forward pass to execute only a subset of layers
    // 2. Exposing per-layer hidden states for inter-stage transfer
    // 3. Async DMA between GPU buffers (ggml_backend_tensor_copy_async)
    // 4. Synchronization barriers at stage boundaries

    fprintf(stderr, "pipeline: multi-GPU execution (%zu stages) — executing sequentially\n",
            pipe->stages.size());

    // Stages are not executed here yet.
    (void)tokens;
    (void)targets;
    (void)total_tokens;
    (void)model;
    (void)ctx;

    return 0.0f;
}
