#include "alpha_train_backward.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>

// Loss gradient: cross-entropy backward

float alpha_train_loss_gradient(
    const float * logits,
    const int   * targets,
    float       * dLogits,
    int n_vocab,
    int n_tokens) {

    float loss = 0.0f;
    int count = 0;

    for (int t = 0; t < n_tokens; t++) {
        const float * row = logits + t * n_vocab;
        float * drow = dLogits + t * n_vocab;
        int label = targets[t];

        if (label < 0 || label >= n_vocab) {
            memset(drow, 0, n_vocab * sizeof(float));
            continue;
        }

        // Numerically stable softmax
        float max_val = row[0];
        for (int v = 1; v < n_vocab; v++) {
            if (row[v] > max_val) max_val = row[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            sum_exp += expf(row[v] - max_val);
        }

        // dLogits = softmax(logits) - one_hot(target)
        // This is the standard cross-entropy gradient
        float log_sum = logf(sum_exp);
        for (int v = 0; v < n_vocab; v++) {
            float p = expf(row[v] - max_val - log_sum);
            drow[v] = p;
        }
        drow[label] -= 1.0f;

        // Scale by 1/n_tokens for mean loss
        for (int v = 0; v < n_vocab; v++) {
            drow[v] /= n_tokens;
        }

        // Loss = -log(softmax[label])
        float log_prob = (row[label] - max_val) - log_sum;
        loss -= log_prob;
        count++;
    }

    return count > 0 ? loss / count : 0.0f;
}

// RMSNorm backward

void alpha_train_rmsnorm_backward(
    const float * x,
    const float * d_out,
    const float * gamma,
    float       * d_input,
    int n,
    int m,
    float eps) {

    for (int j = 0; j < m; j++) {
        const float * xj = x + j * n;
        const float * dj = d_out + j * n;
        float * dij = d_input + j * n;

        // Forward: rms = sqrt(mean(x^2) + eps), x_hat = x / rms, out = gamma * x_hat
        float sum_sq = 0.0f;
        for (int i = 0; i < n; i++) sum_sq += xj[i] * xj[i];
        float rms = sqrtf(sum_sq / n + eps);
        float rms_inv = 1.0f / rms;

        // d_x_hat = d_out * gamma
        // d_rms = -sum(d_x_hat * x) / rms^2
        // d_input = (d_x_hat - x_hat * mean(x_hat * d_x_hat)) / rms
        float dot = 0.0f;
        for (int i = 0; i < n; i++) {
            float x_hat = xj[i] * rms_inv;
            float dx_hat = dj[i] * (gamma ? gamma[i] : 1.0f);
            dot += x_hat * dx_hat;
        }
        dot /= n;

        for (int i = 0; i < n; i++) {
            float x_hat = xj[i] * rms_inv;
            float dx_hat = dj[i] * (gamma ? gamma[i] : 1.0f);
            dij[i] = (dx_hat - x_hat * dot) * rms_inv;
        }
    }
}

// SwiGLU backward

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float silu_deriv(float x) {
    float s = 1.0f / (1.0f + expf(-x));
    return s * (1.0f + x * (1.0f - s));
}

void alpha_train_swiglu_backward(
    const float * gate_pre,
    const float * up_val,
    const float * d_out,
    float       * d_gate,
    float       * d_up,
    int n) {

    // Forward: out = SiLU(gate) * up
    // Backward: d_gate = d_out * up * SiLU'(gate)
    //           d_up   = d_out * SiLU(gate)
    for (int i = 0; i < n; i++) {
        float g = gate_pre[i];
        float u = up_val[i];
        float dout = d_out[i];

        d_gate[i] = dout * u * silu_deriv(g);
        d_up[i]   = dout * silu(g);
    }
}

// CPU matrix operations (for LoRA gradients, small matrices)

// C = A @ B  where A[m,k], B[k,n] -> C[m,n]
static void matmul_cpu(const float * A, const float * B, float * C,
                       int m, int k, int n) {
    memset(C, 0, m * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            float a = A[i * k + p];
            if (fabsf(a) < 1e-10f) continue;
            for (int j = 0; j < n; j++) {
                C[i * n + j] += a * B[p * n + j];
            }
        }
    }
}

// C = A^T @ B  where A[k,m], B[k,n] -> C[m,n]
static void matmul_tn(const float * A, const float * B, float * C,
                      int k, int m, int n) {
    memset(C, 0, m * n * sizeof(float));
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
static void matmul_nt(const float * A, const float * B, float * C,
                      int m, int k, int n) {
    memset(C, 0, m * n * sizeof(float));
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

// LoRA gradient computation

void alpha_train_lora_gradient(
    const float * dY,
    const float * x,
    const float * A,
    const float * B,
    float       * dA,
    float       * dB,
    float scale,
    int in_dim,
    int out_dim,
    int rank,
    int n_tokens) {

    // Forward: y_lora = scale * B @ (A @ x) = scale * B @ z
    // where z = A @ x  [rank, n_tokens]

    // Compute z = A @ x
    std::vector<float> z(rank * n_tokens);
    matmul_cpu(A, x, z.data(), rank, in_dim, n_tokens);

    // dB = scale * dY @ z^T  -> [out_dim, rank]
    matmul_nt(dY, z.data(), dB, out_dim, n_tokens, rank);
    for (int i = 0; i < out_dim * rank; i++) dB[i] *= scale;

    // dA = scale * B^T @ dY @ x^T  -> [rank, in_dim]
    // First: tmp = B^T @ dY  -> [rank, n_tokens]
    std::vector<float> tmp(rank * n_tokens);
    matmul_tn(B, dY, tmp.data(), out_dim, rank, n_tokens);

    // Then: dA = tmp @ x^T  -> [rank, in_dim]
    matmul_nt(tmp.data(), x, dA, rank, n_tokens, in_dim);
    for (int i = 0; i < rank * in_dim; i++) dA[i] *= scale;
}

// Linear backward through frozen quantized weight (GPU-accelerated)

void alpha_train_linear_backward_gpu(
    struct ggml_context * /*ctx_unused*/,
    struct ggml_backend * /*backend_unused*/,
    struct ggml_tensor  * W,
    const float         * dY_data,
    float               * dX_data,
    int in_dim,
    int out_dim,
    int n_tokens) {

    // Backward of Y = ggml_mul_mat(W, X):
    //   W has ne[0]=in_dim, ne[1]=out_dim
    //   Forward: mul_mat(W, X) computes C[i,j] = sum_k W[k,i]*X[k,j] (ggml convention)
    //   Backward for X: dX[k,j] = sum_i W[k,i] * dY[i,j]
    //   This is mul_mat(W_transposed, dY) where W_transposed has ne[0]=out_dim
    //
    // GPU path: dequantize W to F32, transpose on CPU, upload, GPU mul_mat.
    // The matmul is the expensive part; dequant+transpose is O(n) overhead.

    // Lazy-init GPU backend
    static ggml_backend_t s_gpu_backend = nullptr;
    static bool s_gpu_checked = false;
    if (!s_gpu_checked) {
        s_gpu_checked = true;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                s_gpu_backend = ggml_backend_dev_init(dev, nullptr);
                break;
            }
        }
    }

    // F32 with host data: use direct CPU matmul (no copy needed)
    if (W->type == GGML_TYPE_F32 && W->data) {
        matmul_tn((const float *)W->data, dY_data, dX_data, out_dim, in_dim, n_tokens);
        return;
    }

    size_t w_bytes = ggml_nbytes(W);
    bool gpu_feasible = s_gpu_backend != nullptr;

    // Skip GPU for tensors > 1GB (VRAM pressure)
    if (w_bytes > 1024ULL * 1024 * 1024) {
        gpu_feasible = false;
    }

    if (gpu_feasible) {
        // Dequantize W to F32 on CPU
        int64_t n_elements = (int64_t)in_dim * out_dim;
        std::vector<float> W_f32(n_elements);

        if (W->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(W, W_f32.data(), 0, n_elements * sizeof(float));
        } else {
            std::vector<uint8_t> qbuf(ggml_nbytes(W));
            ggml_backend_tensor_get(W, qbuf.data(), 0, ggml_nbytes(W));
            const auto * traits = ggml_get_type_traits(W->type);
            if (traits && traits->to_float) {
                traits->to_float(qbuf.data(), W_f32.data(), n_elements);
            } else {
                gpu_feasible = false;
            }
        }

        if (gpu_feasible) {
            // Transpose W: from ggml [in_dim, out_dim] to [out_dim, in_dim]
            // W_f32 layout: out_dim rows of in_dim each -> W_f32[i*in_dim + k]
            // W_t layout: in_dim rows of out_dim each -> W_t[k*out_dim + i]
            std::vector<float> W_t(n_elements);
            for (int i = 0; i < out_dim; i++) {
                for (int k = 0; k < in_dim; k++) {
                    W_t[(int64_t)k * out_dim + i] = W_f32[(int64_t)i * in_dim + k];
                }
            }

            ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(s_gpu_backend);

            size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
            struct ggml_init_params params = { ctx_size, nullptr, true };
            struct ggml_context * gctx = ggml_init(params);
            if (!gctx) goto cpu_fallback;

            {
                // W_t: [out_dim, in_dim] F32 - ne[0]=out_dim matches dY's ne[0]
                struct ggml_tensor * Wt_ref = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, out_dim, in_dim);
                ggml_set_name(Wt_ref, "bwd_Wt");

                struct ggml_tensor * dY_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, out_dim, n_tokens);
                ggml_set_name(dY_t, "bwd_dY");

                // mul_mat(Wt, dY): ne[0]=out_dim match, result=[in_dim, n_tokens]
                struct ggml_tensor * dX_t = ggml_mul_mat(gctx, Wt_ref, dY_t);
                ggml_set_name(dX_t, "bwd_dX");

                struct ggml_cgraph * graph = ggml_new_graph(gctx);
                ggml_build_forward_expand(graph, dX_t);

                ggml_backend_t backends[] = { s_gpu_backend };
                ggml_backend_buffer_type_t bufts[] = { gpu_buft };
                ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, 1, 64, false, false);

                if (sched) {
                    ggml_backend_sched_alloc_graph(sched, graph);

                    // Upload transposed F32 weights
                    ggml_backend_tensor_set(Wt_ref, W_t.data(), 0, n_elements * sizeof(float));

                    // Upload dY
                    ggml_backend_tensor_set(dY_t, dY_data, 0, (size_t)out_dim * n_tokens * sizeof(float));

                    // GPU matmul
                    ggml_backend_sched_graph_compute(sched, graph);

                    // Download result
                    ggml_backend_tensor_get(dX_t, dX_data, 0, (size_t)in_dim * n_tokens * sizeof(float));

                    ggml_backend_sched_free(sched);
                    ggml_free(gctx);
                    return;
                }

                ggml_free(gctx);
            }
        }
    }

cpu_fallback:
    // CPU fallback: dequantize W to F32 then matmul
    {
        int64_t n_elements = (int64_t)in_dim * out_dim;

        // For F32 weights on host memory, use data pointer directly (no copy needed)
        if (W->type == GGML_TYPE_F32 && W->data) {
            matmul_tn((const float *)W->data, dY_data, dX_data, out_dim, in_dim, n_tokens);
            return;
        }

        // For backend-resident F32 or quantized: download + dequantize
        std::vector<float> W_f32(n_elements);

        if (W->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(W, W_f32.data(), 0, n_elements * sizeof(float));
        } else {
            std::vector<uint8_t> qbuf(ggml_nbytes(W));
            ggml_backend_tensor_get(W, qbuf.data(), 0, ggml_nbytes(W));

            const auto * type_traits = ggml_get_type_traits(W->type);
            if (type_traits && type_traits->to_float) {
                type_traits->to_float(qbuf.data(), W_f32.data(), n_elements);
            } else {
                fprintf(stderr, "unsupported type: %s\n", ggml_type_name(W->type));
                memset(dX_data, 0, (size_t)in_dim * n_tokens * sizeof(float));
                return;
            }
        }

        matmul_tn(W_f32.data(), dY_data, dX_data, out_dim, in_dim, n_tokens);
    }
}

// Full one-layer backward

void alpha_train_layer_backward(
    struct alpha_train_engine * engine,
    int layer_id,
    const float * dOutput,
    float       * dInput,
    alpha_train_layer_grads * grads,
    const alpha_train_layer_state * saved) {

    // TODO: implement full layer backward
    // This will be wired up by the train engine which manages the
    // layer streaming pipeline and provides the model layer weights
    //
    // The backward follows the forward in reverse:
    //
    // 7. d_residual += dOutput (FFN residual)
    // 6. d_ffn_down = dOutput -> backward through down projection
    // 5. d_swiglu -> SwiGLU backward -> d_gate, d_up
    // 4. d_ffn_norm -> backward through gate/up projections + RMSNorm
    // 3. d_residual += d_ffn_norm (attention residual)
    // 2. d_attn -> backward through attention (Q/K/V/O projections)
    // 1. d_input -> backward through attention norm

    (void)engine;
    (void)layer_id;

    // Placeholder until the train engine orchestrates the layer-by-layer
    // backward with weight streaming
    if (dOutput && dInput && saved) {
        // Copy through as identity for testing
        int n = saved->input.ne[0] * saved->input.ne[1];
        memcpy(dInput, dOutput, n * sizeof(float));
    }

    (void)grads;
}
