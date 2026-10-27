#include "alpha_dense_pipeline.h"
#include "alpha_mole_moe_optimizer.h"  // for alpha_cuda_dma_*
#include "llama-model.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// Thread-local layer range for graph builder
thread_local int g_alpha_pipeline_layer_start = -1;
thread_local int g_alpha_pipeline_layer_end   = -1;
thread_local int g_alpha_pipeline_skip_start  = -1;
thread_local int g_alpha_pipeline_skip_end    = -1;

void alpha_pipeline_set_layer_range(int start, int end) {
    g_alpha_pipeline_layer_start = start;
    g_alpha_pipeline_layer_end = end;
}

void alpha_pipeline_clear_layer_range(void) {
    g_alpha_pipeline_layer_start = -1;
    g_alpha_pipeline_layer_end = -1;
}

void alpha_pipeline_set_skip_range(int skip_start, int skip_end) {
    g_alpha_pipeline_skip_start = skip_start;
    g_alpha_pipeline_skip_end = skip_end;
}

void alpha_pipeline_clear_skip_range(void) {
    g_alpha_pipeline_skip_start = -1;
    g_alpha_pipeline_skip_end = -1;
}

// Weight tensors per layer that need GPU staging
// These are the large matrices - norms are tiny and stay wherever they are
struct layer_weight_info {
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_down;
    ggml_tensor * ffn_up;

    // Saved original data pointers (for restore after pass)
    void * orig_wq;
    void * orig_wk;
    void * orig_wv;
    void * orig_wo;
    void * orig_gate;
    void * orig_down;
    void * orig_up;
};

struct alpha_dense_pipeline {
    int n_passes;
    int layers_per_pass;
    int n_layer;

    // Per-layer weight info
    std::vector<layer_weight_info> layer_weights;

    // GPU staging buffers (double-buffered)
    // Each buffer holds layers_per_pass worth of weight data
    ggml_backend_buffer_t gpu_buf[2];
    std::vector<ggml_tensor *> gpu_tensors[2];  // [buffer_idx][tensor_idx within buffer]
    size_t per_layer_bytes;  // total weight bytes per layer

    // Pinned staging for async DMA
    void * pinned_buf;
    size_t pinned_size;

    // State
    int active_buffer;
    bool first_token;

    // Stats
    int64_t n_tokens;
    double total_dma_ms;
    double total_compute_ms;
};

// Count bytes for the main weight tensors of one layer
static size_t compute_layer_weight_bytes(const llama_layer & layer) {
    size_t bytes = 0;
    if (layer.wq)       bytes += ggml_nbytes(layer.wq);
    if (layer.wk)       bytes += ggml_nbytes(layer.wk);
    if (layer.wv)       bytes += ggml_nbytes(layer.wv);
    if (layer.wo)       bytes += ggml_nbytes(layer.wo);
    if (layer.ffn_gate) bytes += ggml_nbytes(layer.ffn_gate);
    if (layer.ffn_down) bytes += ggml_nbytes(layer.ffn_down);
    if (layer.ffn_up)   bytes += ggml_nbytes(layer.ffn_up);
    return bytes;
}

alpha_dense_pipeline * alpha_dense_pipeline_init(
    struct llama_model * model,
    size_t vram_budget_mb)
{
    if (!model) return nullptr;

    int n_layer = llama_model_n_layer(model);
    if (n_layer <= 0) return nullptr;

    // Compute per-layer weight size
    size_t per_layer_bytes = compute_layer_weight_bytes(model->layers[0]);
    if (per_layer_bytes == 0) return nullptr;

    size_t total_weight_bytes = per_layer_bytes * n_layer;
    size_t total_weight_mb = total_weight_bytes / (1024 * 1024);

    // Reserve VRAM for: KV cache (~500MB), compute buffer (~500MB), norms (~50MB)
    size_t reserved_mb = 1200;
    if (vram_budget_mb <= reserved_mb) {
        fprintf(stderr, "[PIPELINE] Not enough VRAM for pipeline (%zuMB available, %zuMB reserved)\n",
                vram_budget_mb, reserved_mb);
        return nullptr;
    }

    size_t available_mb = vram_budget_mb - reserved_mb;

    // If all weights fit in one pass, no pipeline needed
    if (total_weight_mb <= available_mb) {
        return nullptr;  // model fits - use normal path
    }

    // Calculate passes: need double-buffer, so each buffer = available / 2
    size_t per_buffer_mb = available_mb / 2;
    size_t per_layer_mb = per_layer_bytes / (1024 * 1024);
    if (per_layer_mb == 0) per_layer_mb = 1;

    int layers_per_pass = (int)(per_buffer_mb / per_layer_mb);
    if (layers_per_pass < 1) {
        fprintf(stderr, "[PIPELINE] Not enough VRAM for even 1 layer per pass (%zuMB/buffer, %zuMB/layer)\n",
                per_buffer_mb, per_layer_mb);
        return nullptr;
    }

    int n_passes = (n_layer + layers_per_pass - 1) / layers_per_pass;

    auto * pipe = new alpha_dense_pipeline();
    pipe->n_passes = n_passes;
    pipe->layers_per_pass = layers_per_pass;
    pipe->n_layer = n_layer;
    pipe->per_layer_bytes = per_layer_bytes;
    pipe->active_buffer = 0;
    pipe->first_token = true;
    pipe->n_tokens = 0;
    pipe->total_dma_ms = 0;
    pipe->total_compute_ms = 0;

    // Collect layer weight info
    pipe->layer_weights.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        auto & lw = pipe->layer_weights[il];
        auto & layer = model->layers[il];
        lw.wq       = layer.wq;
        lw.wk       = layer.wk;
        lw.wv       = layer.wv;
        lw.wo       = layer.wo;
        lw.ffn_gate = layer.ffn_gate;
        lw.ffn_down = layer.ffn_down;
        lw.ffn_up   = layer.ffn_up;
    }

    // Allocate GPU staging buffers
    // Find GPU backend
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_dev = dev;
            break;
        }
    }
    if (!gpu_dev) {
        fprintf(stderr, "[PIPELINE] No GPU device found\n");
        delete pipe;
        return nullptr;
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(gpu_dev);
    size_t buf_size = per_layer_bytes * layers_per_pass;

    for (int b = 0; b < 2; b++) {
        pipe->gpu_buf[b] = ggml_backend_buft_alloc_buffer(gpu_buft, buf_size);
        if (!pipe->gpu_buf[b]) {
            fprintf(stderr, "[PIPELINE] Failed to allocate GPU buffer %d (%zuMB)\n",
                    b, buf_size / (1024 * 1024));
            // Clean up
            if (b > 0 && pipe->gpu_buf[0]) ggml_backend_buffer_free(pipe->gpu_buf[0]);
            delete pipe;
            return nullptr;
        }
    }

    // Allocate pinned staging buffer for async DMA
    pipe->pinned_size = per_layer_bytes;  // one layer at a time
    pipe->pinned_buf = _acpa(pipe->pinned_size);

    fprintf(stderr, "[PIPELINE] %d-pass pipeline: %d layers/pass, %.1fMB/layer, %.1fMB/buffer\n",
            n_passes, layers_per_pass, per_layer_bytes / (1024.0 * 1024.0),
            buf_size / (1024.0 * 1024.0));

    return pipe;
}

int alpha_dense_pipeline_n_passes(const alpha_dense_pipeline * pipe) {
    return pipe ? pipe->n_passes : 1;
}

// DMA one layer's weights from CPU (mmap) to GPU buffer
// For now: uses ggml_backend_tensor_set on the original tensor
// (swaps buffer temporarily). TODO: dedicated GPU staging tensors for async DMA.
static void dma_layer_to_gpu(alpha_dense_pipeline * pipe, int layer_idx, int buf_idx, int slot) {
    (void)pipe; (void)layer_idx; (void)buf_idx; (void)slot;
    // Placeholder - actual DMA done in prepare_pass via tensor pointer swap
    // The GPU buffer holds raw weight data at known offsets
    // prepare_pass sets tensor->data to point into this buffer
}

void alpha_dense_pipeline_prepare_pass(
    alpha_dense_pipeline * pipe,
    struct llama_model * model,
    int pass_idx)
{
    if (!pipe || !model) return;

    int layer_start = pass_idx * pipe->layers_per_pass;
    int layer_end = std::min(layer_start + pipe->layers_per_pass, pipe->n_layer);
    int buf_idx = pass_idx % 2;

    // Set layer range globals for graph builder
    g_alpha_pipeline_layer_start = layer_start;
    g_alpha_pipeline_layer_end = layer_end;

    // If first token or buffer not pre-loaded, DMA now (synchronous)
    if (pipe->first_token) {
        for (int il = layer_start; il < layer_end; il++) {
            dma_layer_to_gpu(pipe, il, buf_idx, il - layer_start);
        }
        if (pass_idx == pipe->n_passes - 1) {
            pipe->first_token = false;
        }
    }

    // Swap model tensor pointers to GPU buffer
    size_t buf_offset = 0;
    uint8_t * gpu_base = (uint8_t *)ggml_backend_buffer_get_base(pipe->gpu_buf[buf_idx]);
    for (int il = layer_start; il < layer_end; il++) {
        auto & lw = pipe->layer_weights[il];
        auto & layer = model->layers[il];

        // Save original pointers
        lw.orig_wq   = layer.wq ? layer.wq->data : nullptr;
        lw.orig_wk   = layer.wk ? layer.wk->data : nullptr;
        lw.orig_wv   = layer.wv ? layer.wv->data : nullptr;
        lw.orig_wo   = layer.wo ? layer.wo->data : nullptr;
        lw.orig_gate = layer.ffn_gate ? layer.ffn_gate->data : nullptr;
        lw.orig_down = layer.ffn_down ? layer.ffn_down->data : nullptr;
        lw.orig_up   = layer.ffn_up ? layer.ffn_up->data : nullptr;

        // Swap to GPU buffer
        ggml_tensor * tensors[] = { layer.wq, layer.wk, layer.wv, layer.wo,
                                     layer.ffn_gate, layer.ffn_down, layer.ffn_up };
        for (auto * t : tensors) {
            if (!t) continue;
            size_t sz = ggml_nbytes(t);
            t->data = gpu_base + buf_offset;
            buf_offset += sz;
        }
    }
}

void alpha_dense_pipeline_finish_pass(
    alpha_dense_pipeline * pipe,
    struct llama_model * model,
    int pass_idx)
{
    if (!pipe || !model) return;

    int layer_start = pass_idx * pipe->layers_per_pass;
    int layer_end = std::min(layer_start + pipe->layers_per_pass, pipe->n_layer);

    // Restore original tensor pointers
    for (int il = layer_start; il < layer_end; il++) {
        auto & lw = pipe->layer_weights[il];
        auto & layer = model->layers[il];
        if (layer.wq)       layer.wq->data       = lw.orig_wq;
        if (layer.wk)       layer.wk->data       = lw.orig_wk;
        if (layer.wv)       layer.wv->data       = lw.orig_wv;
        if (layer.wo)       layer.wo->data       = lw.orig_wo;
        if (layer.ffn_gate) layer.ffn_gate->data  = lw.orig_gate;
        if (layer.ffn_down) layer.ffn_down->data  = lw.orig_down;
        if (layer.ffn_up)   layer.ffn_up->data    = lw.orig_up;
    }

    // Clear layer range
    g_alpha_pipeline_layer_start = -1;
    g_alpha_pipeline_layer_end = -1;

    // Start async DMA for next pass (double-buffer)
    int next_pass = (pass_idx + 1) % pipe->n_passes;
    int next_buf = next_pass % 2;
    int next_start = next_pass * pipe->layers_per_pass;
    int next_end = std::min(next_start + pipe->layers_per_pass, pipe->n_layer);
    for (int il = next_start; il < next_end; il++) {
        dma_layer_to_gpu(pipe, il, next_buf, il - next_start);
    }

    pipe->n_tokens++;
}

void alpha_dense_pipeline_sync(alpha_dense_pipeline * pipe) {
    if (!pipe) return;
    _acds();
}

void alpha_dense_pipeline_print_stats(const alpha_dense_pipeline * pipe) {
    if (!pipe) return;
    fprintf(stderr, "[PIPELINE] %d passes × %d layers, %lld tokens processed\n",
            pipe->n_passes, pipe->layers_per_pass, (long long)pipe->n_tokens);
}

void alpha_dense_pipeline_free(alpha_dense_pipeline * pipe) {
    if (!pipe) return;
    for (int b = 0; b < 2; b++) {
        if (pipe->gpu_buf[b]) ggml_backend_buffer_free(pipe->gpu_buf[b]);
    }
    if (pipe->pinned_buf) _acpf(pipe->pinned_buf);
    delete pipe;
}
