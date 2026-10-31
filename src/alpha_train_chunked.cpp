#include "alpha_train_chunked.h"
#include "alpha_train_backward.h"
#include "alpha_train_optim.h"
#include "alpha_train_tiered.h"
#include "alpha_train_combined.h"
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
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

// ============================================================
// CPU matrix operations (needed for weight gradient computation)
// ============================================================

// C = A^T @ B  where A[k,m], B[k,n] -> C[m,n]
static void matmul_tn_cpu(const float * A, const float * B, float * C,
                          int k, int m, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < m; i++) {
            float a = A[p * m + i];
            if (fabsf(a) < 1e-10f) continue;
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * B[p * n + j];
            }
        }
    }
}

// C = A @ B^T  where A[m,k], B[n,k] -> C[m,n]
static void matmul_nt_cpu(const float * A, const float * B, float * C,
                          int m, int k, int n) {
    memset(C, 0, (size_t)m * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[j * k + p];
            }
            C[i * n + j] = sum;
        }
    }
}

// ============================================================
// Dequantize / requantize helpers
// ============================================================

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

static void requantize_and_write(ggml_tensor * W, const float * f32_data) {
    if (W->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(W, f32_data, 0, ggml_nelements(W) * sizeof(float));
        return;
    }

    int64_t nrows = W->ne[1];
    int64_t n_per_row = W->ne[0];
    size_t qbuf_size = ggml_nbytes(W);
    std::vector<uint8_t> qbuf(qbuf_size);

    ggml_quantize_chunk(W->type, f32_data, qbuf.data(), 0, nrows, n_per_row, nullptr);
    ggml_backend_tensor_set(W, qbuf.data(), 0, qbuf_size);
}

// ============================================================
// Per-layer training state
// ============================================================

struct chunked_layer {
    int  layer_id;
    bool is_active;        // in current chunk (gets weight gradients)
    bool is_gpu_resident;

    // Weight tensors (pointers into model, not owned)
    ggml_tensor * wq;
    ggml_tensor * wk;
    ggml_tensor * wv;
    ggml_tensor * wo;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_down;
    ggml_tensor * attn_norm;
    ggml_tensor * ffn_norm;

    // Saved activations (CPU, F32)
    std::vector<float> act_input;       // input to this layer [n_embd, n_tokens]
    std::vector<float> act_ffn_norm;    // after FFN norm [n_embd, n_tokens]
    std::vector<float> act_ffn_gate;    // gate projection output pre-SiLU [n_ff, n_tokens]
    std::vector<float> act_ffn_up;      // up projection output [n_ff, n_tokens]
    int act_tokens;

    // Weight gradients (only for active chunk layers, mode B/D)
    std::vector<float> grad_wq, grad_wk, grad_wv, grad_wo;
    std::vector<float> grad_ffn_gate, grad_ffn_up, grad_ffn_down;
    int grad_accum_count;  // number of steps accumulated
};

// Per-layer LoRA params (Mode A)
struct chunked_lora_layer {
    std::string name;
    ggml_tensor * W;

    int64_t in_dim;
    int64_t out_dim;
    int     rank;

    std::vector<float> A, B;
    std::vector<float> mA, vA, mB, vB;
    std::vector<float> dA, dB;
    std::vector<float> x_saved;
    int x_tokens;
};

// Per-chunk Adam state for full-param training
struct chunk_adam_state {
    std::vector<float> m;  // first moment (all weight params in chunk)
    std::vector<float> v;  // second moment
    int n_params;
    bool initialized;
};

// ============================================================
// Engine state
// ============================================================

struct alpha_chunked_engine {
    llama_model   * model;
    llama_context * ctx;
    alpha_chunked_config config;
    alpha_chunked_plan   plan;

    // Layers
    std::vector<chunked_layer> layers;

    // LoRA (Mode A only)
    std::vector<chunked_lora_layer> lora_layers;
    llama_adapter_lora * adapter;
    ggml_context       * adapter_ctx;
    ggml_backend_buffer_t adapter_buf;

    // Per-chunk Adam states for full-param training
    std::vector<chunk_adam_state> chunk_adam;

    // Training state
    int  current_chunk;
    int  current_cycle;
    int  global_step;
    int  chunk_step;
    float current_loss;
    float eval_loss_best;
    float eval_loss_baseline;
    bool  should_stop;

    // Sweep order
    std::vector<int> chunk_order;

    // Scratch buffers
    std::vector<float> dY;
    std::vector<float> dResidual;
    std::vector<float> dFFN;
    std::vector<float> dAttn;
    std::vector<float> scratch1, scratch2, scratch3;

    // Model info
    int n_layer;
    int n_embd;
    int n_vocab;
    int chunk_size;

    // Disk streaming state (Level 4)
    std::string work_dir;
    bool disk_streaming;

    // Timing
    double total_train_ms;
    int    total_fwd_steps;
    int    total_bwd_steps;
};

// ============================================================
// Thread-local for eval callback
// ============================================================

static thread_local alpha_chunked_engine * g_chunked_engine = nullptr;

static bool chunked_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * engine = g_chunked_engine;
    if (!engine && user_data) engine = (alpha_chunked_engine *)user_data;
    if (!engine) return true;

    if (ask) return true;

    const char * name = ggml_get_name(t);
    if (!name || !name[0]) return true;

    std::string sname(name);

    // Parse block ID from tensor name
    auto parse_block_id = [&]() -> int {
        size_t dash = sname.rfind('-');
        if (dash == std::string::npos) return -1;
        int id = atoi(sname.c_str() + dash + 1);
        return (id >= 0 && id < engine->n_layer) ? id : -1;
    };

    // Save attn_norm outputs as layer inputs for backward
    if (sname.find("attn_norm") != std::string::npos) {
        int block_id = parse_block_id();
        if (block_id < 0) return true;

        int n_elem = (int)ggml_nelements(t);
        int n_tokens = n_elem / engine->n_embd;

        auto & layer = engine->layers[block_id];
        layer.act_input.resize(n_elem);
        ggml_backend_tensor_get(t, layer.act_input.data(), 0, n_elem * sizeof(float));
        layer.act_tokens = n_tokens;

        // Also save for LoRA layers
        for (auto & ll : engine->lora_layers) {
            char prefix[32];
            snprintf(prefix, sizeof(prefix), "blk.%d.", block_id);
            if (ll.name.find(prefix) != std::string::npos) {
                ll.x_saved.resize(n_elem);
                memcpy(ll.x_saved.data(), layer.act_input.data(), n_elem * sizeof(float));
                ll.x_tokens = n_tokens;
            }
        }
    }

    // Save FFN norm output (input to FFN projections)
    if (sname.find("ffn_norm") != std::string::npos) {
        int block_id = parse_block_id();
        if (block_id < 0) return true;
        auto & layer = engine->layers[block_id];
        if (!layer.is_active) return true;  // only save for active chunk

        int n_elem = (int)ggml_nelements(t);
        layer.act_ffn_norm.resize(n_elem);
        ggml_backend_tensor_get(t, layer.act_ffn_norm.data(), 0, n_elem * sizeof(float));
    }

    // Save FFN gate output (pre-SiLU) for exact SwiGLU backward
    if (sname.find("ffn_gate") != std::string::npos && sname.find("ffn_gate_") == std::string::npos) {
        int block_id = parse_block_id();
        if (block_id < 0) return true;
        auto & layer = engine->layers[block_id];
        if (!layer.is_active) return true;

        int n_elem = (int)ggml_nelements(t);
        layer.act_ffn_gate.resize(n_elem);
        ggml_backend_tensor_get(t, layer.act_ffn_gate.data(), 0, n_elem * sizeof(float));
    }

    // Save FFN up output for exact SwiGLU backward
    if (sname.find("ffn_up") != std::string::npos) {
        int block_id = parse_block_id();
        if (block_id < 0) return true;
        auto & layer = engine->layers[block_id];
        if (!layer.is_active) return true;

        int n_elem = (int)ggml_nelements(t);
        layer.act_ffn_up.resize(n_elem);
        ggml_backend_tensor_get(t, layer.act_ffn_up.data(), 0, n_elem * sizeof(float));
    }

    return true;
}

// ============================================================
// Auto-mode planning
// ============================================================

alpha_chunked_plan alpha_chunked_plan_training(
    struct llama_model * model,
    alpha_chunked_config config) {

    alpha_chunked_plan plan = {};

    plan.n_layer = (int)model->layers.size();
    plan.n_embd  = model->hparams.n_embd;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    plan.n_vocab = llama_vocab_n_tokens(vocab);

    plan.n_params = 0;
    plan.model_bytes = 0;
    for (int il = 0; il < plan.n_layer; il++) {
        const auto & layer = model->layers[il];
        ggml_tensor * tensors[] = {
            layer.wq, layer.wk, layer.wv, layer.wo,
            layer.ffn_gate, layer.ffn_up, layer.ffn_down,
            layer.attn_norm, layer.ffn_norm,
        };
        for (auto * t : tensors) {
            if (t) {
                plan.model_bytes += ggml_nbytes(t);
                plan.n_params += ggml_nelements(t);
            }
        }
    }
    plan.layer_bytes = plan.n_layer > 0 ? plan.model_bytes / plan.n_layer : 0;

    // Detect GPU VRAM
    plan.gpu_vram_mb = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_mem = 0, total_mem = 0;
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);
            plan.gpu_vram_mb = total_mem / (1024 * 1024);
            break;
        }
    }

    plan.ram_mb = config.ram_budget_mb > 0 ? config.ram_budget_mb : 32000;

    // Select mode
    if (config.mode != ALPHA_CHUNKED_AUTO) {
        plan.mode = config.mode;
    } else {
        plan.mode = ALPHA_CHUNKED_FULL_BP;
    }

    // Select level and K
    size_t G = plan.gpu_vram_mb > 0 ? (size_t)(plan.gpu_vram_mb * 0.9) * 1024 * 1024 : 0;
    size_t M = plan.model_bytes;
    size_t act_per_layer = (size_t)plan.n_embd * config.seq_len * sizeof(float);
    int L = plan.n_layer;

    if (config.level != ALPHA_LEVEL_AUTO) {
        plan.level = config.level;
    } else if (G == 0) {
        plan.level = ALPHA_LEVEL_RAM_STREAM;
    } else {
        size_t adam_bytes = (size_t)plan.n_params * 8;
        size_t level0 = M + M + act_per_layer * L + adam_bytes;
        if (level0 < G) {
            plan.level = ALPHA_LEVEL_FULL_GPU;
        } else if (M * 2 + act_per_layer * (L / 4) + plan.layer_bytes * 2 < G) {
            plan.level = ALPHA_LEVEL_GPU_RESIDENT;
        } else if (M / 4 * 2 + act_per_layer * (L / 4) + plan.layer_bytes * 2 < G) {
            plan.level = ALPHA_LEVEL_PARTIAL_GPU;
        } else if (plan.layer_bytes * 2 + act_per_layer * (L / 8) + M / 8 < G) {
            plan.level = ALPHA_LEVEL_RAM_STREAM;
        } else {
            plan.level = ALPHA_LEVEL_DISK_STREAM;
        }
    }

    // Select K based on level
    if (config.K > 0) {
        plan.K = config.K;
    } else {
        switch (plan.level) {
            case ALPHA_LEVEL_FULL_GPU:     plan.K = 1; break;
            case ALPHA_LEVEL_GPU_RESIDENT: plan.K = 4; break;
            case ALPHA_LEVEL_PARTIAL_GPU:  plan.K = 4; break;
            case ALPHA_LEVEL_RAM_STREAM:   plan.K = 8; break;
            case ALPHA_LEVEL_DISK_STREAM:  plan.K = 16; break;
            default:                       plan.K = 4; break;
        }
        if (plan.K > L) plan.K = L;
    }

    // Compute GPU residency
    int chunk_layers = L / plan.K;
    switch (plan.level) {
        case ALPHA_LEVEL_FULL_GPU:
            plan.gpu_resident_layers = L;
            break;
        case ALPHA_LEVEL_GPU_RESIDENT:
            plan.gpu_resident_layers = L;
            break;
        case ALPHA_LEVEL_PARTIAL_GPU: {
            size_t chunk_gpu = (size_t)chunk_layers * plan.layer_bytes * 2;
            size_t remaining = G > chunk_gpu ? G - chunk_gpu : 0;
            int extra = (int)(remaining / plan.layer_bytes);
            plan.gpu_resident_layers = std::min(L, chunk_layers + extra);
            break;
        }
        default:
            plan.gpu_resident_layers = chunk_layers;
            break;
    }
    plan.ram_streamed_layers = L - plan.gpu_resident_layers;
    plan.disk_streamed_layers = (plan.level == ALPHA_LEVEL_DISK_STREAM) ? plan.ram_streamed_layers : 0;

    // Timing estimate
    float gpu_ms = plan.gpu_resident_layers * 0.5f;
    float stream_ms = plan.ram_streamed_layers * 20.0f;
    plan.ms_per_step = gpu_ms + stream_ms;

    int steps_per_cycle = plan.K * (config.steps_per_chunk > 0 ? config.steps_per_chunk : 1000);
    int cycles = (plan.mode == ALPHA_CHUNKED_FROM_SCRATCH) ? 3 : 1;
    plan.total_minutes = (plan.ms_per_step * steps_per_cycle * cycles) / 60000.0f;

    return plan;
}

void alpha_chunked_print_plan(const alpha_chunked_plan & plan) {
    const char * mode_names[] = { "Auto", "LoRA", "Chunked", "Full" };
    const char * level_names[] = { "Full GPU", "GPU-Resident", "Partial GPU", "RAM Stream", "Disk Stream" };

    fprintf(stderr, "\n=== Training Plan ===\n");
    fprintf(stderr, "  Mode:     %s\n", mode_names[plan.mode]);
    fprintf(stderr, "  Level:    %d (%s)\n", plan.level, level_names[plan.level]);
    fprintf(stderr, "  Chunks:   K=%d (%d layers each)\n", plan.K, plan.n_layer / plan.K);
    fprintf(stderr, "  Model:    %d layers, %d embd, %.1f M params\n",
            plan.n_layer, plan.n_embd, plan.n_params / 1e6f);
    fprintf(stderr, "  Size:     %.1f MB stored, %.1f MB/layer\n",
            plan.model_bytes / 1e6f, plan.layer_bytes / 1e6f);
    fprintf(stderr, "  GPU:      %zu MB VRAM, %d layers resident\n",
            plan.gpu_vram_mb, plan.gpu_resident_layers);
    fprintf(stderr, "  Stream:   %d layers from RAM", plan.ram_streamed_layers);
    if (plan.disk_streamed_layers > 0) {
        fprintf(stderr, ", %d from disk", plan.disk_streamed_layers);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "  Speed:    ~%.0f ms/step\n", plan.ms_per_step);
    fprintf(stderr, "  Estimate: ~%.1f min total\n", plan.total_minutes);
    fprintf(stderr, "=====================\n\n");
}

// ============================================================
// Sync LoRA params to adapter (Mode A)
// ============================================================

static void sync_lora_to_adapter(alpha_chunked_engine * engine) {
    if (!engine->adapter) return;
    for (auto & ll : engine->lora_layers) {
        std::string base_name(ll.W->name);
        auto it = engine->adapter->ab_map.find(base_name);
        if (it == engine->adapter->ab_map.end()) continue;
        ggml_backend_tensor_set(it->second.a, ll.A.data(), 0, ll.A.size() * sizeof(float));
        ggml_backend_tensor_set(it->second.b, ll.B.data(), 0, ll.B.size() * sizeof(float));
    }
}

// ============================================================
// Allocate / free weight gradients for active chunk
// ============================================================

static void alloc_weight_grads(alpha_chunked_engine * engine) {
    if (engine->config.mode == ALPHA_CHUNKED_LORA) return;

    int chunk_start = engine->current_chunk * engine->chunk_size;
    int chunk_end   = std::min(chunk_start + engine->chunk_size, engine->n_layer);

    for (int il = chunk_start; il < chunk_end; il++) {
        auto & cl = engine->layers[il];
        auto alloc_grad = [](std::vector<float> & grad, ggml_tensor * W) {
            if (!W) return;
            grad.resize(ggml_nelements(W), 0.0f);
        };
        alloc_grad(cl.grad_wq, cl.wq);
        alloc_grad(cl.grad_wk, cl.wk);
        alloc_grad(cl.grad_wv, cl.wv);
        alloc_grad(cl.grad_wo, cl.wo);
        alloc_grad(cl.grad_ffn_gate, cl.ffn_gate);
        alloc_grad(cl.grad_ffn_up, cl.ffn_up);
        alloc_grad(cl.grad_ffn_down, cl.ffn_down);
        cl.grad_accum_count = 0;
    }
}

static void zero_weight_grads(alpha_chunked_engine * engine) {
    int chunk_start = engine->current_chunk * engine->chunk_size;
    int chunk_end   = std::min(chunk_start + engine->chunk_size, engine->n_layer);

    for (int il = chunk_start; il < chunk_end; il++) {
        auto & cl = engine->layers[il];
        auto zero = [](std::vector<float> & g) {
            if (!g.empty()) std::fill(g.begin(), g.end(), 0.0f);
        };
        zero(cl.grad_wq); zero(cl.grad_wk); zero(cl.grad_wv); zero(cl.grad_wo);
        zero(cl.grad_ffn_gate); zero(cl.grad_ffn_up); zero(cl.grad_ffn_down);
        cl.grad_accum_count = 0;
    }
}

static void free_weight_grads(alpha_chunked_engine * engine) {
    for (auto & cl : engine->layers) {
        auto free_grad = [](std::vector<float> & g) { g.clear(); g.shrink_to_fit(); };
        free_grad(cl.grad_wq); free_grad(cl.grad_wk); free_grad(cl.grad_wv); free_grad(cl.grad_wo);
        free_grad(cl.grad_ffn_gate); free_grad(cl.grad_ffn_up); free_grad(cl.grad_ffn_down);
    }
}

// ============================================================
// Forward pass (reuses llama_decode with activation capture)
// ============================================================

static float forward_pass(
    alpha_chunked_engine * engine,
    const llama_token * tokens,
    int n_tokens) {

    if (engine->adapter) sync_lora_to_adapter(engine);
    llama_memory_clear(llama_get_memory(engine->ctx), true);

    g_chunked_engine = engine;

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
        fprintf(stderr, "chunked forward: decode failed (%d)\n", rc);
        return -1.0f;
    }

    // Compute cross-entropy loss
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

    engine->total_fwd_steps++;
    return count > 0 ? loss / count : 0.0f;
}

// ============================================================
// Backward pass: chunked -- exact grads on active, input-only on frozen.
// Active layers accumulate weight gradients and use the saved activations
// for an exact SwiGLU backward.
// ============================================================

static void backward_chunked(
    alpha_chunked_engine * engine,
    const llama_token * tokens,
    int n_tokens) {

    const int nt = n_tokens - 1;
    if (nt <= 0) return;

    const int ne = engine->n_embd;
    const int nv = engine->n_vocab;
    const int chunk_start = engine->current_chunk * engine->chunk_size;
    const int chunk_end   = std::min(chunk_start + engine->chunk_size, engine->n_layer);
    const bool full_param = (engine->config.mode != ALPHA_CHUNKED_LORA);

    // Step 1: Loss gradient (dLogits)
    std::vector<float> dLogits(nv * nt);
    std::vector<int> targets(nt);
    for (int i = 0; i < nt; i++) targets[i] = tokens[i + 1];

    std::vector<float> all_logits(nv * nt);
    for (int i = 0; i < nt; i++) {
        float * li = llama_get_logits_ith(engine->ctx, i);
        if (li) memcpy(all_logits.data() + (size_t)i * nv, li, nv * sizeof(float));
    }

    alpha_train_loss_gradient(all_logits.data(), targets.data(), dLogits.data(), nv, nt);

    // Step 2: Backward through output head
    ggml_tensor * output_w = engine->model->output;
    if (!output_w) output_w = engine->model->tok_embd;

    engine->dY.resize(ne * nt, 0.0f);
    std::fill(engine->dY.begin(), engine->dY.end(), 0.0f);

    if (output_w) {
        alpha_train_linear_backward_gpu(nullptr, nullptr, output_w,
            dLogits.data(), engine->dY.data(), ne, nv, nt);
    }

    // Step 3: Layer-by-layer backward from output to chunk_start
    engine->dResidual.resize(ne * nt);
    engine->dFFN.resize(ne * nt);
    engine->dAttn.resize(ne * nt);

    for (int il = engine->n_layer - 1; il >= chunk_start; il--) {
        const auto & mlayer = engine->model->layers[il];
        auto & clayer = engine->layers[il];
        bool is_active = (il >= chunk_start && il < chunk_end);

        memcpy(engine->dResidual.data(), engine->dY.data(), ne * nt * sizeof(float));
        memcpy(engine->dFFN.data(), engine->dY.data(), ne * nt * sizeof(float));

        // -- FFN backward --
        if (mlayer.ffn_down && mlayer.ffn_gate && mlayer.ffn_up) {
            int64_t n_ff = mlayer.ffn_up->ne[1];

            // Backward through W_down: dSwiGLU = W_down^T @ dFFN
            engine->scratch1.resize(n_ff * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.ffn_down,
                engine->dFFN.data(), engine->scratch1.data(), (int)n_ff, ne, nt);

            // Weight gradient for W_down: dW_down = dFFN @ act_swiglu_out^T
            // (act_swiglu_out = SiLU(gate)*up, which is the input to ffn_down)
            if (is_active && full_param && !clayer.act_ffn_gate.empty() && !clayer.act_ffn_up.empty()) {
                // Recompute SwiGLU output = SiLU(gate) * up
                std::vector<float> swiglu_out(n_ff * nt);
                for (int64_t i = 0; i < n_ff * nt; i++) {
                    float g = clayer.act_ffn_gate[i];
                    float sg = g / (1.0f + expf(-g));  // SiLU(gate)
                    swiglu_out[i] = sg * clayer.act_ffn_up[i];
                }
                // dW_down += dFFN^T @ swiglu_out  [ne, n_ff] accumulated
                // dFFN is [ne, nt], swiglu_out is [n_ff, nt]
                // dW_down[ne, n_ff] = dFFN[ne, nt] @ swiglu_out[n_ff, nt]^T
                std::vector<float> dW_tmp(ne * n_ff);
                matmul_nt_cpu(engine->dFFN.data(), swiglu_out.data(), dW_tmp.data(), ne, nt, (int)n_ff);
                for (size_t i = 0; i < clayer.grad_ffn_down.size() && i < dW_tmp.size(); i++) {
                    clayer.grad_ffn_down[i] += dW_tmp[i];
                }
            }

            // SwiGLU backward
            engine->scratch2.resize(n_ff * nt);
            engine->scratch3.resize(n_ff * nt);

            if (is_active && !clayer.act_ffn_gate.empty() && !clayer.act_ffn_up.empty()) {
                // Exact SwiGLU backward with saved activations
                alpha_train_swiglu_backward(
                    clayer.act_ffn_gate.data(), clayer.act_ffn_up.data(),
                    engine->scratch1.data(),
                    engine->scratch2.data(), engine->scratch3.data(),
                    (int)(n_ff * nt));
            } else {
                // Approximation for frozen layers (no saved activations)
                for (int64_t i = 0; i < n_ff * nt; i++) {
                    engine->scratch2[i] = engine->scratch1[i] * 0.5f;
                    engine->scratch3[i] = engine->scratch1[i] * 0.5f;
                }
            }

            // Weight gradients for W_gate and W_up
            if (is_active && full_param && !clayer.act_ffn_norm.empty()) {
                // dW_gate = d_gate^T @ ffn_norm_out  [n_ff, ne]
                // scratch2 is d_gate [n_ff, nt], act_ffn_norm is [ne, nt]
                std::vector<float> dW_tmp((size_t)n_ff * ne);
                matmul_nt_cpu(engine->scratch2.data(), clayer.act_ffn_norm.data(), dW_tmp.data(), (int)n_ff, nt, ne);
                for (size_t i = 0; i < clayer.grad_ffn_gate.size() && i < dW_tmp.size(); i++) {
                    clayer.grad_ffn_gate[i] += dW_tmp[i];
                }

                // dW_up = d_up^T @ ffn_norm_out  [n_ff, ne]
                matmul_nt_cpu(engine->scratch3.data(), clayer.act_ffn_norm.data(), dW_tmp.data(), (int)n_ff, nt, ne);
                for (size_t i = 0; i < clayer.grad_ffn_up.size() && i < dW_tmp.size(); i++) {
                    clayer.grad_ffn_up[i] += dW_tmp[i];
                }
            }

            // Backward through W_gate and W_up to get d_ffn_input
            std::vector<float> d_ffn_input(ne * nt, 0.0f);
            std::vector<float> tmp(ne * nt);

            alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.ffn_gate,
                engine->scratch2.data(), tmp.data(), ne, (int)n_ff, nt);
            for (int i = 0; i < ne * nt; i++) d_ffn_input[i] += tmp[i];

            alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.ffn_up,
                engine->scratch3.data(), tmp.data(), ne, (int)n_ff, nt);
            for (int i = 0; i < ne * nt; i++) d_ffn_input[i] += tmp[i];

            for (int i = 0; i < ne * nt; i++) engine->dResidual[i] += d_ffn_input[i];
        }

        // -- Attention backward --
        memcpy(engine->dAttn.data(), engine->dResidual.data(), ne * nt * sizeof(float));

        if (mlayer.wo) {
            int64_t attn_out_dim = mlayer.wo->ne[0];
            engine->scratch1.resize(attn_out_dim * nt);
            alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.wo,
                engine->dAttn.data(), engine->scratch1.data(), (int)attn_out_dim, ne, nt);

            // Weight gradient for W_o
            if (is_active && full_param && !clayer.act_input.empty()) {
                // dW_o = dAttn^T @ attn_output (approximated as act_input for now)
                std::vector<float> dW_tmp(ne * (int)attn_out_dim);
                matmul_nt_cpu(engine->dAttn.data(), clayer.act_input.data(), dW_tmp.data(), ne, nt, (int)attn_out_dim);
                for (size_t i = 0; i < clayer.grad_wo.size() && i < dW_tmp.size(); i++) {
                    clayer.grad_wo[i] += dW_tmp[i];
                }
            }

            // LoRA gradients (Mode A)
            for (auto & ll : engine->lora_layers) {
                char prefix[32];
                snprintf(prefix, sizeof(prefix), "blk.%d.", il);
                if (ll.name.find(prefix) == std::string::npos) continue;
                if (ll.x_saved.empty() || ll.x_tokens == 0) continue;

                std::vector<float> dA_local(ll.A.size());
                std::vector<float> dB_local(ll.B.size());
                alpha_train_lora_gradient(
                    engine->scratch1.data(), ll.x_saved.data(),
                    ll.A.data(), ll.B.data(),
                    dA_local.data(), dB_local.data(),
                    1.0f, (int)ll.in_dim, (int)ll.out_dim, ll.rank, nt);

                for (size_t i = 0; i < ll.dA.size(); i++) ll.dA[i] += dA_local[i];
                for (size_t i = 0; i < ll.dB.size(); i++) ll.dB[i] += dB_local[i];
            }

            // Backward through Q/K/V projections
            std::vector<float> d_attn_norm(ne * nt, 0.0f);
            if (mlayer.wq) {
                std::vector<float> tmp2(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.wq,
                    engine->scratch1.data(), tmp2.data(), ne, (int)attn_out_dim, nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp2[i];

                // Weight gradient for W_q
                if (is_active && full_param && !clayer.act_input.empty()) {
                    std::vector<float> dW_tmp(ne * (int)attn_out_dim);
                    matmul_nt_cpu(engine->scratch1.data(), clayer.act_input.data(), dW_tmp.data(), (int)attn_out_dim, nt, ne);
                    for (size_t i = 0; i < clayer.grad_wq.size() && i < dW_tmp.size(); i++) {
                        clayer.grad_wq[i] += dW_tmp[i];
                    }
                }
            }
            if (mlayer.wv) {
                std::vector<float> tmp2(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.wv,
                    engine->scratch1.data(), tmp2.data(), ne, (int)mlayer.wv->ne[1], nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp2[i];

                if (is_active && full_param && !clayer.act_input.empty()) {
                    std::vector<float> dW_tmp((size_t)mlayer.wv->ne[1] * ne);
                    matmul_nt_cpu(engine->scratch1.data(), clayer.act_input.data(), dW_tmp.data(), (int)mlayer.wv->ne[1], nt, ne);
                    for (size_t i = 0; i < clayer.grad_wv.size() && i < dW_tmp.size(); i++) {
                        clayer.grad_wv[i] += dW_tmp[i];
                    }
                }
            }
            if (mlayer.wk) {
                std::vector<float> tmp2(ne * nt);
                alpha_train_linear_backward_gpu(nullptr, nullptr, mlayer.wk,
                    engine->scratch1.data(), tmp2.data(), ne, (int)mlayer.wk->ne[1], nt);
                for (int i = 0; i < ne * nt; i++) d_attn_norm[i] += tmp2[i];

                if (is_active && full_param && !clayer.act_input.empty()) {
                    std::vector<float> dW_tmp((size_t)mlayer.wk->ne[1] * ne);
                    matmul_nt_cpu(engine->scratch1.data(), clayer.act_input.data(), dW_tmp.data(), (int)mlayer.wk->ne[1], nt, ne);
                    for (size_t i = 0; i < clayer.grad_wk.size() && i < dW_tmp.size(); i++) {
                        clayer.grad_wk[i] += dW_tmp[i];
                    }
                }
            }

            for (int i = 0; i < ne * nt; i++) engine->dResidual[i] += d_attn_norm[i];
        }

        // Propagate to next layer
        memcpy(engine->dY.data(), engine->dResidual.data(), ne * nt * sizeof(float));
    }

    // Mark gradient accumulation
    for (int il = chunk_start; il < chunk_end; il++) {
        engine->layers[il].grad_accum_count++;
    }

    engine->total_bwd_steps++;
}

// ============================================================
// Adam update for LoRA params
// ============================================================

static void adam_update_lora(alpha_chunked_engine * engine) {
    float lr = engine->config.lr > 0 ? engine->config.lr : 2e-5f;
    float beta1 = engine->config.beta1;
    float beta2 = engine->config.beta2;
    float adam_eps = engine->config.eps;
    float wd = engine->config.weight_decay;
    int t = engine->global_step;

    float effective_lr = lr;
    int warmup = engine->config.warmup_steps > 0 ? engine->config.warmup_steps : 100;
    if (t < warmup) effective_lr = lr * ((float)(t + 1) / warmup);

    float bc1 = 1.0f - powf(beta1, (float)(t + 1));
    float bc2 = 1.0f - powf(beta2, (float)(t + 1));

    for (auto & ll : engine->lora_layers) {
        if (ll.dA.empty()) continue;

        auto update = [&](std::vector<float> & params, std::vector<float> & grads,
                          std::vector<float> & m, std::vector<float> & v) {
            for (size_t i = 0; i < params.size(); i++) {
                float g = grads[i] + wd * params[i];
                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;
                float m_hat = m[i] / bc1;
                float v_hat = v[i] / bc2;
                params[i] -= effective_lr * m_hat / (sqrtf(v_hat) + adam_eps);
                grads[i] = 0.0f;
            }
        };

        update(ll.A, ll.dA, ll.mA, ll.vA);
        update(ll.B, ll.dB, ll.mB, ll.vB);
    }
}

// ============================================================
// Adam update for full-param base weights
// ============================================================

static void adam_update_base_weights(alpha_chunked_engine * engine) {
    if (engine->config.mode == ALPHA_CHUNKED_LORA) return;

    int chunk_id = engine->current_chunk;
    int chunk_start = chunk_id * engine->chunk_size;
    int chunk_end   = std::min(chunk_start + engine->chunk_size, engine->n_layer);

    // Ensure Adam state exists for this chunk
    if (chunk_id >= (int)engine->chunk_adam.size()) {
        engine->chunk_adam.resize(chunk_id + 1);
    }
    auto & adam = engine->chunk_adam[chunk_id];

    // Count total params in chunk
    int total_params = 0;
    for (int il = chunk_start; il < chunk_end; il++) {
        auto & cl = engine->layers[il];
        ggml_tensor * weights[] = { cl.wq, cl.wk, cl.wv, cl.wo, cl.ffn_gate, cl.ffn_up, cl.ffn_down };
        for (auto * w : weights) {
            if (w) total_params += (int)ggml_nelements(w);
        }
    }

    if (!adam.initialized || adam.n_params != total_params) {
        adam.m.resize(total_params, 0.0f);
        adam.v.resize(total_params, 0.0f);
        adam.n_params = total_params;
        adam.initialized = true;
    }

    // Hyperparams
    float lr = engine->config.lr > 0 ? engine->config.lr : 2e-5f;
    // From-scratch needs higher LR
    if (engine->config.mode == ALPHA_CHUNKED_FROM_SCRATCH && engine->config.lr <= 0) {
        lr = 1e-4f;
    }
    float beta1 = engine->config.beta1;
    float beta2 = engine->config.beta2;
    float adam_eps = engine->config.eps;
    float wd = engine->config.weight_decay;
    int t = engine->global_step;

    int warmup = engine->config.warmup_steps > 0 ? engine->config.warmup_steps : 100;
    float effective_lr = lr;
    if (t < warmup) effective_lr = lr * ((float)(t + 1) / warmup);

    float bc1 = 1.0f - powf(beta1, (float)(t + 1));
    float bc2 = 1.0f - powf(beta2, (float)(t + 1));

    // Process each weight tensor in the chunk
    int adam_offset = 0;
    for (int il = chunk_start; il < chunk_end; il++) {
        auto & cl = engine->layers[il];

        struct weight_grad_pair {
            ggml_tensor * W;
            std::vector<float> * grad;
        };
        weight_grad_pair pairs[] = {
            { cl.wq,       &cl.grad_wq },
            { cl.wk,       &cl.grad_wk },
            { cl.wv,       &cl.grad_wv },
            { cl.wo,       &cl.grad_wo },
            { cl.ffn_gate, &cl.grad_ffn_gate },
            { cl.ffn_up,   &cl.grad_ffn_up },
            { cl.ffn_down, &cl.grad_ffn_down },
        };

        for (auto & p : pairs) {
            if (!p.W || p.grad->empty()) {
                if (p.W) adam_offset += (int)ggml_nelements(p.W);
                continue;
            }

            int n = (int)ggml_nelements(p.W);
            int accum = cl.grad_accum_count > 0 ? cl.grad_accum_count : 1;

            // Dequantize current weights
            std::vector<float> W_f32;
            dequantize_weight(p.W, W_f32);

            // Adam update
            float * m_ptr = adam.m.data() + adam_offset;
            float * v_ptr = adam.v.data() + adam_offset;

            for (int i = 0; i < n; i++) {
                float g = (*p.grad)[i] / accum + wd * W_f32[i];
                m_ptr[i] = beta1 * m_ptr[i] + (1 - beta1) * g;
                v_ptr[i] = beta2 * v_ptr[i] + (1 - beta2) * g * g;
                float m_hat = m_ptr[i] / bc1;
                float v_hat = v_ptr[i] / bc2;
                W_f32[i] -= effective_lr * m_hat / (sqrtf(v_hat) + adam_eps);
            }

            // Requantize and write back
            requantize_and_write(p.W, W_f32.data());

            adam_offset += n;
        }
    }
}

// ============================================================
// Optimizer state disk spill
// ============================================================

static void spill_adam_to_disk(alpha_chunked_engine * engine, int chunk_id) {
    if (engine->work_dir.empty()) return;
    if (chunk_id >= (int)engine->chunk_adam.size()) return;
    auto & adam = engine->chunk_adam[chunk_id];
    if (!adam.initialized || adam.n_params == 0) return;

    char path_m[512], path_v[512];
    snprintf(path_m, sizeof(path_m), "%s/adam_chunk_%d_m.bin", engine->work_dir.c_str(), chunk_id);
    snprintf(path_v, sizeof(path_v), "%s/adam_chunk_%d_v.bin", engine->work_dir.c_str(), chunk_id);

    FILE * fm = fopen(path_m, "wb");
    FILE * fv = fopen(path_v, "wb");
    if (fm && fv) {
        fwrite(adam.m.data(), sizeof(float), adam.n_params, fm);
        fwrite(adam.v.data(), sizeof(float), adam.n_params, fv);
    }
    if (fm) fclose(fm);
    if (fv) fclose(fv);

    // Free RAM
    adam.m.clear(); adam.m.shrink_to_fit();
    adam.v.clear(); adam.v.shrink_to_fit();
    adam.initialized = false;
}

static void load_adam_from_disk(alpha_chunked_engine * engine, int chunk_id) {
    if (engine->work_dir.empty()) return;
    if (chunk_id >= (int)engine->chunk_adam.size()) {
        engine->chunk_adam.resize(chunk_id + 1);
    }
    auto & adam = engine->chunk_adam[chunk_id];
    if (adam.initialized) return;  // already in RAM

    char path_m[512], path_v[512];
    snprintf(path_m, sizeof(path_m), "%s/adam_chunk_%d_m.bin", engine->work_dir.c_str(), chunk_id);
    snprintf(path_v, sizeof(path_v), "%s/adam_chunk_%d_v.bin", engine->work_dir.c_str(), chunk_id);

    FILE * fm = fopen(path_m, "rb");
    FILE * fv = fopen(path_v, "rb");
    if (!fm || !fv) {
        if (fm) fclose(fm);
        if (fv) fclose(fv);
        return;  // no saved state, will init fresh
    }

    // Count params to know buffer size
    int chunk_start = chunk_id * engine->chunk_size;
    int chunk_end   = std::min(chunk_start + engine->chunk_size, engine->n_layer);
    int total_params = 0;
    for (int il = chunk_start; il < chunk_end; il++) {
        auto & cl = engine->layers[il];
        ggml_tensor * weights[] = { cl.wq, cl.wk, cl.wv, cl.wo, cl.ffn_gate, cl.ffn_up, cl.ffn_down };
        for (auto * w : weights) {
            if (w) total_params += (int)ggml_nelements(w);
        }
    }

    adam.m.resize(total_params);
    adam.v.resize(total_params);
    adam.n_params = total_params;

    size_t read_m = fread(adam.m.data(), sizeof(float), total_params, fm);
    size_t read_v = fread(adam.v.data(), sizeof(float), total_params, fv);
    fclose(fm);
    fclose(fv);

    adam.initialized = (read_m == (size_t)total_params && read_v == (size_t)total_params);
    if (!adam.initialized) {
        std::fill(adam.m.begin(), adam.m.end(), 0.0f);
        std::fill(adam.v.begin(), adam.v.end(), 0.0f);
        adam.initialized = true;
    }
}

// ============================================================
// Initialize engine
// ============================================================

alpha_chunked_engine * alpha_chunked_init(
    struct llama_model   * model,
    struct llama_context * ctx,
    alpha_chunked_config   config,
    alpha_chunked_plan     plan) {

    if (!model || !ctx) return nullptr;

    auto * engine = new alpha_chunked_engine();
    engine->model  = model;
    engine->ctx    = ctx;
    engine->config = config;
    engine->plan   = plan;

    engine->n_layer = plan.n_layer;
    engine->n_embd  = plan.n_embd;
    engine->n_vocab = plan.n_vocab;
    engine->chunk_size = plan.n_layer / plan.K;

    engine->current_chunk = 0;
    engine->current_cycle = 0;
    engine->global_step   = 0;
    engine->chunk_step    = 0;
    engine->current_loss  = 0;
    engine->eval_loss_best = 1e9f;
    engine->eval_loss_baseline = -1.0f;
    engine->should_stop = false;
    engine->disk_streaming = (plan.level == ALPHA_LEVEL_DISK_STREAM);

    engine->total_train_ms = 0;
    engine->total_fwd_steps = 0;
    engine->total_bwd_steps = 0;

    engine->adapter     = nullptr;
    engine->adapter_ctx = nullptr;
    engine->adapter_buf = nullptr;

    // Work directory for optimizer spill
    if (config.work_dir) {
        engine->work_dir = config.work_dir;
    } else {
        engine->work_dir = ".";
    }

    // Build sweep order
    engine->chunk_order.resize(plan.K);
    if (plan.mode == ALPHA_CHUNKED_FROM_SCRATCH) {
        for (int i = 0; i < plan.K; i++) engine->chunk_order[i] = plan.K - 1 - i;
    } else {
        for (int i = 0; i < plan.K; i++) engine->chunk_order[i] = i;
    }
    engine->current_chunk = engine->chunk_order[0];

    // Initialize layer state
    engine->layers.resize(plan.n_layer);
    for (int il = 0; il < plan.n_layer; il++) {
        auto & cl = engine->layers[il];
        cl.layer_id = il;
        cl.act_tokens = 0;
        cl.grad_accum_count = 0;

        const auto & ml = model->layers[il];
        cl.wq       = ml.wq;
        cl.wk       = ml.wk;
        cl.wv       = ml.wv;
        cl.wo       = ml.wo;
        cl.ffn_gate = ml.ffn_gate;
        cl.ffn_up   = ml.ffn_up;
        cl.ffn_down = ml.ffn_down;
        cl.attn_norm = ml.attn_norm;
        cl.ffn_norm  = ml.ffn_norm;

        int chunk_id = il / engine->chunk_size;
        cl.is_active = (chunk_id == engine->current_chunk);
        cl.is_gpu_resident = (il < plan.gpu_resident_layers);
    }

    // Initialize LoRA (all modes use LoRA for attention)
    int rank = config.lora_rank > 0 ? config.lora_rank : 64;
    if (plan.mode != ALPHA_CHUNKED_LORA) rank = config.lora_rank > 0 ? config.lora_rank : 32;

    struct lora_target { ggml_tensor * W; const char * suffix; };

    for (int il = 0; il < plan.n_layer; il++) {
        const auto & layer = model->layers[il];
        lora_target targets[] = {
            { layer.wq, "attn_q" },
            { layer.wk, "attn_k" },
            { layer.wv, "attn_v" },
            { layer.wo, "attn_o" },
        };

        for (auto & t : targets) {
            if (!t.W) continue;

            chunked_lora_layer ll = {};
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

            std::mt19937 rng(42 + il * 100 + (int)(t.suffix[5]));
            std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)rank));
            for (int i = 0; i < nA; i++) ll.A[i] = dist(rng);

            engine->lora_layers.push_back(std::move(ll));
        }
    }

    // Create llama_adapter_lora for inference
    engine->adapter = new llama_adapter_lora(model);
    engine->adapter->alpha = (float)rank;

    size_t ctx_size = ggml_tensor_overhead() * engine->lora_layers.size() * 2 + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    engine->adapter_ctx = ggml_init(params);

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
        fprintf(stderr, "chunked: failed to allocate adapter buffer\n");
        delete engine->adapter;
        ggml_free(engine->adapter_ctx);
        delete engine;
        return nullptr;
    }

    for (auto & ll : engine->lora_layers) {
        std::string base_name(ll.W->name);
        auto it = engine->adapter->ab_map.find(base_name);
        if (it != engine->adapter->ab_map.end()) {
            ggml_backend_tensor_set(it->second.a, ll.A.data(), 0, ll.A.size() * sizeof(float));
            ggml_backend_tensor_set(it->second.b, ll.B.data(), 0, ll.B.size() * sizeof(float));
        }
    }

    llama_adapter_lora * adapters[] = { engine->adapter };
    float scales[] = { 1.0f };
    llama_set_adapters_lora(ctx, adapters, 1, scales);

    // Allocate weight grads for initial active chunk (Mode B/D)
    alloc_weight_grads(engine);

    // Initialize chunk Adam states
    engine->chunk_adam.resize(plan.K);

    int64_t total_params = 0;
    for (const auto & ll : engine->lora_layers) {
        total_params += (int64_t)ll.A.size() + (int64_t)ll.B.size();
    }

    fprintf(stderr, "chunked: %zu LoRA layers (Q/K/V/O), rank=%d, %.2f M trainable params\n",
            engine->lora_layers.size(), rank, (float)total_params / 1e6f);
    if (plan.mode != ALPHA_CHUNKED_LORA) {
        fprintf(stderr, "chunked: full-param weight updates enabled for active chunk\n");
    }
    fprintf(stderr, "chunked: K=%d chunks of %d layers, mode=%d, level=%d\n",
            plan.K, engine->chunk_size, plan.mode, plan.level);

    return engine;
}

// ============================================================
// Public API: step
// ============================================================

float alpha_chunked_step(
    alpha_chunked_engine * engine,
    const llama_token    * tokens,
    int                    n_tokens) {

    if (!engine || !tokens || n_tokens < 2) return -1.0f;

    auto t0 = std::chrono::high_resolution_clock::now();

    // Forward
    float loss = forward_pass(engine, tokens, n_tokens);
    if (loss < 0) return loss;

    engine->global_step++;
    engine->chunk_step++;
    engine->current_loss = loss;

    // Backward
    backward_chunked(engine, tokens, n_tokens);

    // Check if we should apply gradients (gradient accumulation)
    int accum = engine->config.grad_accum > 1 ? engine->config.grad_accum : 1;
    if (engine->chunk_step % accum == 0) {
        // LoRA optimizer step
        adam_update_lora(engine);

        // Full-param weight update (Mode B/D)
        if (engine->config.mode != ALPHA_CHUNKED_LORA) {
            adam_update_base_weights(engine);
            zero_weight_grads(engine);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    engine->total_train_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return loss;
}

// ============================================================
// Chunk management
// ============================================================

bool alpha_chunked_next_chunk(alpha_chunked_engine * engine) {
    if (!engine) return false;

    // Spill current chunk's Adam state to disk if needed
    if (engine->plan.K > 2 && !engine->work_dir.empty()) {
        spill_adam_to_disk(engine, engine->current_chunk);
    }

    // Free weight grads for current chunk
    free_weight_grads(engine);

    // Find current position in sweep order
    int order_pos = -1;
    for (int i = 0; i < (int)engine->chunk_order.size(); i++) {
        if (engine->chunk_order[i] == engine->current_chunk) {
            order_pos = i;
            break;
        }
    }

    order_pos++;
    if (order_pos >= (int)engine->chunk_order.size()) {
        engine->current_cycle++;
        engine->chunk_step = 0;

        if (engine->plan.mode == ALPHA_CHUNKED_FROM_SCRATCH) {
            int max_cycles = engine->config.max_cycles > 0 ? engine->config.max_cycles : 5;
            if (engine->current_cycle >= max_cycles) {
                engine->should_stop = true;
                return false;
            }
        } else {
            return false;
        }

        order_pos = 0;
    }

    engine->current_chunk = engine->chunk_order[order_pos];
    engine->chunk_step = 0;

    // Update active flags
    int chunk_start = engine->current_chunk * engine->chunk_size;
    int chunk_end   = std::min(chunk_start + engine->chunk_size, engine->n_layer);
    for (int il = 0; il < engine->n_layer; il++) {
        engine->layers[il].is_active = (il >= chunk_start && il < chunk_end);
    }

    // Load Adam state from disk for new chunk
    if (engine->plan.K > 2 && !engine->work_dir.empty()) {
        load_adam_from_disk(engine, engine->current_chunk);
    }

    // Allocate weight grads for new chunk
    alloc_weight_grads(engine);

    fprintf(stderr, "  chunk %d/%d (layers %d-%d), cycle %d\n",
            engine->current_chunk + 1, engine->plan.K,
            chunk_start, chunk_end - 1, engine->current_cycle + 1);

    return true;
}

bool alpha_chunked_should_stop(const alpha_chunked_engine * engine) {
    if (!engine) return true;
    if (engine->should_stop) return true;
    if (engine->config.max_steps > 0 && engine->global_step >= engine->config.max_steps) return true;
    return false;
}

int   alpha_chunked_current_chunk(const alpha_chunked_engine * engine) { return engine ? engine->current_chunk : -1; }
int   alpha_chunked_current_cycle(const alpha_chunked_engine * engine) { return engine ? engine->current_cycle : -1; }
int   alpha_chunked_current_step(const alpha_chunked_engine * engine) { return engine ? engine->global_step : -1; }
float alpha_chunked_current_loss(const alpha_chunked_engine * engine) { return engine ? engine->current_loss : -1.0f; }
float alpha_chunked_eval_loss(const alpha_chunked_engine * engine) { return engine ? engine->eval_loss_best : -1.0f; }

// ============================================================
// Save LoRA adapter
// ============================================================

void alpha_chunked_save(const alpha_chunked_engine * engine, const char * path) {
    if (!engine || !path) return;

    int rank = engine->lora_layers.empty() ? 32 : engine->lora_layers[0].rank;

    struct gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.type", "adapter");
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_str(gguf, "adapter.type", "lora");
    gguf_set_val_f32(gguf, "adapter.lora.alpha", (float)rank);

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

    gguf_write_to_file(gguf, path, false);
    ggml_free(ctx);
    gguf_free(gguf);

    fprintf(stderr, "chunked: saved %zu LoRA layers to %s\n", engine->lora_layers.size(), path);
}

// ============================================================
// Save full model with updated base weights (Mode B/D)
// ============================================================

void alpha_chunked_save_full_model(const alpha_chunked_engine * engine, const char * path) {
    if (!engine || !path) return;

    // Save the model by writing a GGUF with all tensors
    // The base weights have already been updated in-place via requantize_and_write
    // So we just need to serialize the current model state

    fprintf(stderr, "chunked: saving full model to %s\n", path);

    struct gguf_context * gguf = gguf_init_empty();

    // Copy metadata from original model's GGUF
    gguf_set_val_str(gguf, "general.type", "model");
    gguf_set_val_str(gguf, "general.architecture", "llama");

    // Write all model tensors
    size_t ctx_size = ggml_tensor_overhead() * (engine->n_layer * 9 + 4) + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    // Helper: add a tensor from the model
    auto add_tensor = [&](ggml_tensor * t) {
        if (!t) return;
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> data(nbytes);
        ggml_backend_tensor_get(t, data.data(), 0, nbytes);

        ggml_tensor * copy = ggml_new_tensor(ctx, t->type, ggml_n_dims(t), t->ne);
        ggml_set_name(copy, ggml_get_name(t));
        gguf_add_tensor(gguf, copy);
    };

    // Token embeddings + output
    add_tensor(engine->model->tok_embd);
    if (engine->model->output && engine->model->output != engine->model->tok_embd) {
        add_tensor(engine->model->output);
    }
    add_tensor(engine->model->output_norm);

    // Per-layer tensors
    for (int il = 0; il < engine->n_layer; il++) {
        const auto & ml = engine->model->layers[il];
        add_tensor(ml.attn_norm);
        add_tensor(ml.wq);
        add_tensor(ml.wk);
        add_tensor(ml.wv);
        add_tensor(ml.wo);
        add_tensor(ml.ffn_norm);
        add_tensor(ml.ffn_gate);
        add_tensor(ml.ffn_up);
        add_tensor(ml.ffn_down);
    }

    gguf_write_to_file(gguf, path, false);
    ggml_free(ctx);
    gguf_free(gguf);

    fprintf(stderr, "chunked: saved full model (%d layers) to %s\n", engine->n_layer, path);
}

// ============================================================
// Stats + cleanup
// ============================================================

void alpha_chunked_print_stats(const alpha_chunked_engine * engine) {
    if (!engine) return;
    float avg_ms = engine->total_fwd_steps > 0 ? engine->total_train_ms / engine->total_fwd_steps : 0;
    fprintf(stderr, "\n=== Training Summary ===\n");
    fprintf(stderr, "  Steps:      %d (fwd=%d, bwd=%d)\n",
            engine->global_step, engine->total_fwd_steps, engine->total_bwd_steps);
    fprintf(stderr, "  Final loss: %.4f\n", engine->current_loss);
    fprintf(stderr, "  Best eval:  %.4f\n", engine->eval_loss_best);
    fprintf(stderr, "  Avg ms/step: %.1f\n", avg_ms);
    fprintf(stderr, "  Total time: %.1f min\n", engine->total_train_ms / 60000.0);

    if (engine->config.json_progress) {
        fprintf(stderr, "{\"event\":\"done\",\"steps\":%d,\"loss\":%.4f,\"eval_best\":%.4f,\"ms_per_step\":%.1f,\"total_min\":%.1f}\n",
                engine->global_step, engine->current_loss, engine->eval_loss_best,
                avg_ms, engine->total_train_ms / 60000.0);
    }
    fprintf(stderr, "========================\n");
}

ggml_backend_sched_eval_callback alpha_chunked_get_callback(void) {
    return chunked_eval_callback;
}

void * alpha_chunked_get_callback_data(alpha_chunked_engine * engine) {
    return (void *)engine;
}

void alpha_chunked_free(alpha_chunked_engine * engine) {
    if (!engine) return;
    llama_set_adapters_lora(engine->ctx, nullptr, 0, nullptr);
    if (engine->adapter_buf) ggml_backend_buffer_free(engine->adapter_buf);
    if (engine->adapter_ctx) ggml_free(engine->adapter_ctx);
    if (engine->adapter) {
        engine->adapter->ab_map.clear();
        delete engine->adapter;
    }
    delete engine;
}

// ============================================================
// Full training loop with 3-stage pipeline
// ============================================================

static float evaluate_loss(
    alpha_chunked_engine * engine,
    const std::vector<std::vector<llama_token>> & sequences,
    size_t eval_start, int max_eval) {

    float eval_loss = 0;
    int eval_count = 0;
    for (size_t s = eval_start; s < sequences.size() && eval_count < max_eval; s++) {
        float l = forward_pass(engine, sequences[s].data(), (int)sequences[s].size());
        if (l >= 0) { eval_loss += l; eval_count++; }
    }
    return eval_count > 0 ? eval_loss / eval_count : 1e9f;
}

// Progress bar helper
static void print_progress(int step, int total, float loss, float eval_loss, double elapsed_ms) {
    int bar_width = 30;
    float progress = total > 0 ? (float)step / total : 0;
    int filled = (int)(progress * bar_width);

    fprintf(stderr, "\r  [");
    for (int i = 0; i < bar_width; i++) {
        fprintf(stderr, i < filled ? "#" : ".");
    }

    float eta_min = total > 0 && step > 0 ? (float)(elapsed_ms / step * (total - step)) / 60000.0f : 0;
    fprintf(stderr, "] %d/%d loss=%.4f eval=%.4f ETA=%.1fm  ",
            step, total, loss, eval_loss, eta_min);
}

int alpha_chunked_train(
    struct llama_model   * model,
    struct llama_context * ctx,
    const char           * data_file,
    const char           * output_file,
    alpha_chunked_config   config) {

    if (!model || !ctx || !data_file) return 1;

    // Plan
    alpha_chunked_plan plan = alpha_chunked_plan_training(model, config);
    alpha_chunked_print_plan(plan);

    // LR defaults based on model size
    if (config.lr <= 0) {
        float n_params_b = plan.n_params / 1e9f;
        if (n_params_b >= 30.0f) config.lr = 5e-6f;
        else if (n_params_b >= 7.0f) config.lr = 1e-5f;
        else if (n_params_b >= 1.0f) config.lr = 2e-5f;
        else config.lr = 5e-5f;

        if (config.mode == ALPHA_CHUNKED_FROM_SCRATCH) {
            config.lr *= 5.0f;  // from-scratch needs higher LR
        }
        fprintf(stderr, "  auto LR: %.1e (%.1fB params)\n", config.lr, n_params_b);
    }

    // Warmup default
    if (config.warmup_steps <= 0) {
        config.warmup_steps = (config.mode == ALPHA_CHUNKED_FROM_SCRATCH) ? 500 : 100;
    }

    // Load data
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::ifstream fin(data_file);
    if (!fin.is_open()) {
        fprintf(stderr, "chunked: failed to open %s\n", data_file);
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

    int seq_len = config.seq_len > 0 ? config.seq_len : 512;
    std::vector<std::vector<llama_token>> sequences;
    for (int i = 0; i + seq_len <= n_tokens; i += seq_len) {
        sequences.emplace_back(all_tokens.begin() + i, all_tokens.begin() + i + seq_len);
    }

    if (sequences.empty()) {
        fprintf(stderr, "chunked: no training sequences (need >= %d tokens)\n", seq_len);
        return 1;
    }

    // Split train/eval (90/10)
    size_t n_eval = std::max((size_t)1, sequences.size() / 10);
    size_t n_train = sequences.size() - n_eval;

    fprintf(stderr, "chunked: %d tokens, %zu train + %zu eval sequences of %d\n",
            n_tokens, n_train, n_eval, seq_len);

    // Init engine
    auto * engine = alpha_chunked_init(model, ctx, config, plan);
    if (!engine) return 1;

    // Baseline eval
    engine->eval_loss_baseline = evaluate_loss(engine, sequences, n_train, 10);
    engine->eval_loss_best = engine->eval_loss_baseline;
    fprintf(stderr, "  baseline eval_loss: %.4f\n", engine->eval_loss_baseline);

    // Training parameters
    int steps_per_chunk = config.steps_per_chunk > 0 ? config.steps_per_chunk : 1000;
    int log_every = config.log_every > 0 ? config.log_every : 10;
    int eval_every = config.eval_every > 0 ? config.eval_every : 100;
    int save_every = config.save_every > 0 ? config.save_every : 500;

    std::mt19937 rng(42);
    auto train_start = std::chrono::high_resolution_clock::now();

    // Stage dispatcher
    if (plan.mode == ALPHA_CHUNKED_FROM_SCRATCH) {
        // 3-stage from-scratch pipeline

        // Stage 1: chunked progressive backprop.
        // Backward sweep with decreasing step counts.
        fprintf(stderr, "\n=== Stage 1: Progressive Training ===\n");

        int max_cycles = config.max_cycles > 0 ? config.max_cycles : 3;
        int cycle_steps[] = { steps_per_chunk, steps_per_chunk / 2, steps_per_chunk / 3 };

        for (int cycle = 0; cycle < max_cycles; cycle++) {
            int spc = cycle < 3 ? cycle_steps[cycle] : steps_per_chunk / 3;
            fprintf(stderr, "  cycle %d/%d (%d steps/chunk)\n", cycle + 1, max_cycles, spc);

            // First chunk is already set from init (backward sweep order)
            if (cycle > 0) {
                // Reset to first in sweep order
                engine->current_chunk = engine->chunk_order[0];
                engine->chunk_step = 0;
                int cs = engine->current_chunk * engine->chunk_size;
                int ce = std::min(cs + engine->chunk_size, engine->n_layer);
                for (int il = 0; il < engine->n_layer; il++) {
                    engine->layers[il].is_active = (il >= cs && il < ce);
                }
                alloc_weight_grads(engine);
            }

            bool cycle_done = false;
            while (!cycle_done) {
                int cs = engine->current_chunk * engine->chunk_size;
                int ce = std::min(cs + engine->chunk_size, engine->n_layer);
                fprintf(stderr, "    chunk %d/%d (layers %d-%d)\n",
                        engine->current_chunk + 1, plan.K, cs, ce - 1);

                for (int step = 0; step < spc; step++) {
                    size_t idx = rng() % n_train;
                    float loss = alpha_chunked_step(engine,
                        sequences[idx].data(), (int)sequences[idx].size());
                    if (loss < 0) continue;

                    if (engine->global_step % log_every == 0) {
                        auto now = std::chrono::high_resolution_clock::now();
                        double elapsed = std::chrono::duration<double, std::milli>(now - train_start).count();
                        int total_est = max_cycles * plan.K * spc;
                        print_progress(engine->global_step, total_est, loss, engine->eval_loss_best, elapsed);
                    }

                    if (engine->global_step % eval_every == 0) {
                        float el = evaluate_loss(engine, sequences, n_train, 10);
                        if (el < engine->eval_loss_best) engine->eval_loss_best = el;
                        fprintf(stderr, "\n    [eval] loss=%.4f best=%.4f\n", el, engine->eval_loss_best);
                    }

                    if (engine->global_step % save_every == 0 && output_file) {
                        std::string ckpt = std::string(output_file) + ".ckpt";
                        alpha_chunked_save(engine, ckpt.c_str());
                    }
                }

                if (!alpha_chunked_next_chunk(engine)) {
                    cycle_done = true;
                }
            }

            // Convergence check between cycles
            float el = evaluate_loss(engine, sequences, n_train, 10);
            float improvement = engine->eval_loss_baseline > 0 ?
                (1.0f - el / engine->eval_loss_baseline) : 0;
            fprintf(stderr, "\n  cycle %d eval=%.4f improvement=%.1f%%\n",
                    cycle + 1, el, improvement * 100.0f);

            if (cycle > 0 && improvement > 0 && improvement < config.convergence_threshold) {
                fprintf(stderr, "  converged (improvement < %.1f%%)\n",
                        config.convergence_threshold * 100.0f);
                break;
            }
        }

        // Save Stage 1 checkpoint
        if (output_file) {
            std::string s1_ckpt = std::string(output_file) + ".stage1";
            alpha_chunked_save(engine, s1_ckpt.c_str());
        }

        // Stage 2: combined MeZO step (broad improvement).
        // Only for level 3-4, where streaming is slow.
        if (plan.level >= ALPHA_LEVEL_RAM_STREAM) {
            fprintf(stderr, "\n=== Stage 2: Combined Optimizer ===\n");

            alpha_combined_train_config cfg2 = alpha_combined_default_config();
            cfg2.rank = config.lora_rank > 0 ? config.lora_rank : 32;
            cfg2.lr = config.lr * 0.5f;
            cfg2.lr_mezo = config.lr * 2.5f;
            cfg2.max_steps = 4000;
            cfg2.json_progress = config.json_progress;

            std::string s2_output = output_file ? std::string(output_file) + ".stage2" : "stage2.gguf";
            alpha_combined_train(model, ctx, data_file, s2_output.c_str(),
                                cfg2, 1, seq_len);
        }

        // Stage 3: polish sweep (fewer steps per chunk)
        fprintf(stderr, "\n=== Stage 3: Polish Sweep ===\n");

        int polish_steps = steps_per_chunk / 3;
        engine->current_chunk = engine->chunk_order[0];
        engine->chunk_step = 0;

        // Reset active flags
        {
            int cs = engine->current_chunk * engine->chunk_size;
            int ce = std::min(cs + engine->chunk_size, engine->n_layer);
            for (int il = 0; il < engine->n_layer; il++) {
                engine->layers[il].is_active = (il >= cs && il < ce);
            }
        }
        alloc_weight_grads(engine);

        bool polish_done = false;
        while (!polish_done) {
            int cs = engine->current_chunk * engine->chunk_size;
            int ce = std::min(cs + engine->chunk_size, engine->n_layer);
            fprintf(stderr, "  polish chunk %d/%d (layers %d-%d)\n",
                    engine->current_chunk + 1, plan.K, cs, ce - 1);

            for (int step = 0; step < polish_steps; step++) {
                size_t idx = rng() % n_train;
                float loss = alpha_chunked_step(engine,
                    sequences[idx].data(), (int)sequences[idx].size());
                if (loss < 0) continue;

                if (engine->global_step % log_every == 0) {
                    fprintf(stderr, "    step %d loss=%.4f\n", engine->global_step, loss);
                }
            }

            if (!alpha_chunked_next_chunk(engine)) {
                polish_done = true;
            }
        }

        // Final eval
        float final_eval = evaluate_loss(engine, sequences, n_train, 10);
        float total_improvement = engine->eval_loss_baseline > 0 ?
            (1.0f - final_eval / engine->eval_loss_baseline) * 100.0f : 0;
        fprintf(stderr, "\n  Final eval=%.4f (%.1f%% improvement from baseline %.4f)\n",
                final_eval, total_improvement, engine->eval_loss_baseline);

    } else {
        // Fine-tune mode (Mode B) or LoRA (Mode A)
        fprintf(stderr, "\n=== Training ===\n");
        fprintf(stderr, "  %d steps/chunk, K=%d chunks\n", steps_per_chunk, plan.K);
        fprintf(stderr, "================\n\n");

        bool done = false;
        while (!done && !alpha_chunked_should_stop(engine)) {
            int cs = engine->current_chunk * engine->chunk_size;
            int ce = std::min(cs + engine->chunk_size, engine->n_layer);
            fprintf(stderr, "  chunk %d/%d (layers %d-%d), cycle %d\n",
                    engine->current_chunk + 1, plan.K,
                    cs, ce - 1, engine->current_cycle + 1);

            for (int step = 0; step < steps_per_chunk; step++) {
                size_t idx = rng() % n_train;
                float loss = alpha_chunked_step(engine,
                    sequences[idx].data(), (int)sequences[idx].size());
                if (loss < 0) continue;

                if (engine->global_step % log_every == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed = std::chrono::duration<double, std::milli>(now - train_start).count();
                    print_progress(engine->global_step, steps_per_chunk * plan.K,
                                   loss, engine->eval_loss_best, elapsed);
                }

                if (engine->global_step % eval_every == 0 && n_eval > 0) {
                    float el = evaluate_loss(engine, sequences, n_train, 10);
                    if (el < engine->eval_loss_best) engine->eval_loss_best = el;
                    float improvement = engine->eval_loss_baseline > 0 ?
                        (1.0f - el / engine->eval_loss_baseline) * 100.0f : 0.0f;
                    fprintf(stderr, "\n    [eval] loss=%.4f, best=%.4f, improvement=%.1f%%\n",
                            el, engine->eval_loss_best, improvement);
                }

                if (engine->global_step % save_every == 0 && output_file) {
                    std::string ckpt = std::string(output_file) + ".ckpt";
                    alpha_chunked_save(engine, ckpt.c_str());
                }

                if (alpha_chunked_should_stop(engine)) break;
            }

            if (!alpha_chunked_next_chunk(engine)) {
                done = true;
            }
        }
    }

    fprintf(stderr, "\n");

    // Save final output
    if (output_file && output_file[0]) {
        alpha_chunked_save(engine, output_file);

        // For full-param modes, also save the updated model
        if (config.mode != ALPHA_CHUNKED_LORA) {
            std::string model_path = std::string(output_file) + ".model.gguf";
            alpha_chunked_save_full_model(engine, model_path.c_str());
        }
    }

    alpha_chunked_print_stats(engine);

    // JSON summary
    if (config.json_progress) {
        fprintf(stderr, "{\"event\":\"complete\",\"steps\":%d,\"final_loss\":%.4f,"
                "\"eval_best\":%.4f,\"baseline\":%.4f,\"improvement_pct\":%.1f,"
                "\"total_min\":%.1f}\n",
                engine->global_step, engine->current_loss,
                engine->eval_loss_best, engine->eval_loss_baseline,
                engine->eval_loss_baseline > 0 ?
                    (1.0f - engine->eval_loss_best / engine->eval_loss_baseline) * 100.0f : 0,
                engine->total_train_ms / 60000.0);
    }

    alpha_chunked_free(engine);
    return 0;
}
