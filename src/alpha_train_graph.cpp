#include "alpha_train_graph.h"
#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <cmath>

// Training graph helpers. A single forward+backward graph (weights flagged GGML_TENSOR_FLAG_PARAM,
// the loss flagged GGML_TENSOR_FLAG_LOSS, ggml_build_backward_expand() on the result) is not possible
// yet because llama_decode() builds and computes the graph immediately. Instead this uses a split
// approach:
// 1. Do the forward pass normally (llama_decode)
// 2. Get logits from the context
// 3. Compute loss + gradients of loss w.r.t. logits (cross-entropy backward) on CPU
// 4. Use the logit gradients as the seed and propagate backward through the model

alpha_train_graph * alpha_train_graph_create(
    llama_model * model,
    llama_context * ctx,
    enum alpha_train_mode mode)
{
    auto * tg = new alpha_train_graph();
    tg->loss = nullptr;
    tg->n_layer = model->hparams.n_layer;
    tg->n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    tg->ctx = nullptr;
    tg->sched = nullptr;
    tg->graph = nullptr;

    // Collect trainable parameters from model
    // For FULL mode: all weight tensors
    // For LORA mode: only LoRA adapter tensors
    // For FREEZE_ATTN: only FFN weight tensors

    // Get all tensors from model layers
    for (int il = 0; il < tg->n_layer; il++) {
        auto & layer = model->layers[il];

        bool train_attn = (mode != ALPHA_TRAIN_FREEZE_ATTN);

        if (mode == ALPHA_TRAIN_FULL || mode == ALPHA_TRAIN_FREEZE_ATTN) {
            // FFN parameters (always trained unless LORA mode)
            if (layer.ffn_gate) {
                layer.ffn_gate->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_gate);
            }
            if (layer.ffn_up) {
                layer.ffn_up->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_up);
            }
            if (layer.ffn_down) {
                layer.ffn_down->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_down);
            }

            // MoE expert parameters
            if (layer.ffn_gate_exps) {
                layer.ffn_gate_exps->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_gate_exps);
            }
            if (layer.ffn_up_exps) {
                layer.ffn_up_exps->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_up_exps);
            }
            if (layer.ffn_down_exps) {
                layer.ffn_down_exps->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_down_exps);
            }

            // Attention parameters (if not frozen)
            if (train_attn) {
                if (layer.wq) {
                    layer.wq->flags |= GGML_TENSOR_FLAG_PARAM;
                    tg->params.push_back(layer.wq);
                }
                if (layer.wk) {
                    layer.wk->flags |= GGML_TENSOR_FLAG_PARAM;
                    tg->params.push_back(layer.wk);
                }
                if (layer.wv) {
                    layer.wv->flags |= GGML_TENSOR_FLAG_PARAM;
                    tg->params.push_back(layer.wv);
                }
                if (layer.wo) {
                    layer.wo->flags |= GGML_TENSOR_FLAG_PARAM;
                    tg->params.push_back(layer.wo);
                }
            }

            // Norm parameters
            if (layer.attn_norm) {
                layer.attn_norm->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.attn_norm);
            }
            if (layer.ffn_norm) {
                layer.ffn_norm->flags |= GGML_TENSOR_FLAG_PARAM;
                tg->params.push_back(layer.ffn_norm);
            }
        }
    }

    // Output head
    if (mode == ALPHA_TRAIN_FULL) {
        if (model->output) {
            model->output->flags |= GGML_TENSOR_FLAG_PARAM;
            tg->params.push_back(model->output);
        }
        if (model->output_norm) {
            model->output_norm->flags |= GGML_TENSOR_FLAG_PARAM;
            tg->params.push_back(model->output_norm);
        }
        if (model->tok_embd) {
            model->tok_embd->flags |= GGML_TENSOR_FLAG_PARAM;
            tg->params.push_back(model->tok_embd);
        }
    }

    fprintf(stderr, "graph: %zu trainable parameters marked (mode=%d)\n",
            tg->params.size(), (int)mode);

    return tg;
}

void alpha_train_graph_free(alpha_train_graph * tg) {
    if (!tg) return;

    // Unmark parameters
    for (auto * p : tg->params) {
        p->flags &= ~GGML_TENSOR_FLAG_PARAM;
    }

    if (tg->ctx) ggml_free(tg->ctx);
    delete tg;
}

float alpha_train_graph_compute(
    alpha_train_graph * tg,
    llama_context * ctx,
    const llama_token * tokens,
    const llama_token * targets,
    int seq_len)
{
    // Forward via llama_decode, loss on the CPU. The backward pass is not GPU-accelerated.

    // 1. Forward pass: decode tokens to get logits at all positions
    //    Must use llama_batch_init (not _get_one) to request per-position logits
    llama_batch batch = llama_batch_init(seq_len, 0, 1);
    for (int i = 0; i < seq_len; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;  // need logits at every position for loss
    }
    batch.n_tokens = seq_len;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "graph: forward pass failed\n");
        llama_batch_free(batch);
        return -1.0f;
    }

    // 2. Get logits from context - now has [seq_len, n_vocab] thanks to per-position logits
    float * logits = llama_get_logits(ctx);
    if (!logits) {
        fprintf(stderr, "graph: no logits available\n");
        llama_batch_free(batch);
        return -1.0f;
    }

    // 3. Compute cross-entropy loss
    float total_loss = 0.0f;
    int n_vocab = tg->n_vocab;

    // Compute loss for each position (logits at position i predict target[i])
    for (int i = 0; i < seq_len; i++) {
        float * pos_logits = logits + i * n_vocab;

        // Numerically stable softmax + cross-entropy
        float max_logit = pos_logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (pos_logits[v] > max_logit) max_logit = pos_logits[v];
        }

        float sum_exp = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            sum_exp += expf(pos_logits[v] - max_logit);
        }

        float log_sum_exp = max_logit + logf(sum_exp);
        float loss_i = log_sum_exp - pos_logits[targets[i]];
        total_loss += loss_i;
    }

    total_loss /= seq_len;

    llama_batch_free(batch);

    // 4. Only the loss value is returned; gradient computation uses the existing train engine.

    return total_loss;
}

ggml_tensor * alpha_train_graph_get_grad(alpha_train_graph * tg, int param_idx) {
    if (!tg || !tg->graph || param_idx < 0 || param_idx >= (int)tg->params.size()) {
        return nullptr;
    }
    if (param_idx < (int)tg->grad_accs.size()) {
        return tg->grad_accs[param_idx];
    }
    return nullptr;
}

// Forward on GPU, loss and logit gradients on CPU, backward through the output head

#include "llama-context.h"
#include "llama-graph.h"
#include "llama-batch.h"

float alpha_train_decode_backward(
    alpha_train_graph * tg,
    struct llama_model * model,
    struct llama_context * ctx,
    const llama_token * tokens,
    const llama_token * targets,
    int seq_len)
{
    if (!tg || !model || !ctx || seq_len <= 0) return -1.0f;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Step 1: forward pass with all logits enabled (gradients are needed at every position)
    llama_batch batch = llama_batch_init(seq_len, 0, 1);
    for (int i = 0; i < seq_len; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;  // need logits at all positions for loss
    }
    batch.n_tokens = seq_len;

    // Split approach: llama_decode runs the forward pass (building and computing the graph), then
    // the loss and logit gradients (softmax - one_hot) are computed on the CPU. A single
    // forward+backward graph would need llama_context::process_ubatch to build without computing;
    // after llama_decode the intermediate tensors may already be freed, so the backward pass
    // through the layers cannot reuse them.

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return -1.0f;
    }

    // Step 2: Get logits and compute loss
    float * logits = llama_get_logits(ctx);
    if (!logits) {
        llama_batch_free(batch);
        return -1.0f;
    }

    // Step 3: Cross-entropy loss + gradient w.r.t. logits
    float total_loss = 0.0f;

    // Allocate gradient buffer for logits
    size_t logit_grad_size = (size_t)seq_len * n_vocab;
    std::vector<float> d_logits(logit_grad_size);

    for (int i = 0; i < seq_len; i++) {
        float * l = logits + (int64_t)i * n_vocab;
        float * dl = d_logits.data() + (int64_t)i * n_vocab;

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
        for (int v = 0; v < n_vocab; v++) {
            dl[v] /= sum_exp;
        }

        total_loss += -logf(dl[targets[i]] + 1e-10f);

        // Gradient: softmax - one_hot
        dl[targets[i]] -= 1.0f;

        // Mean reduction
        for (int v = 0; v < n_vocab; v++) {
            dl[v] /= (float)seq_len;
        }
    }

    total_loss /= (float)seq_len;

    // Step 4: Backward through model
    // The logit gradients (d_logits) propagate back through:
    //   output_head (linear) -> output_norm -> layers[N-1..0]
    // Only the output head is done here.
    {
        // Get output head weight tensor
        ggml_tensor * output_w = model->output;   // [n_vocab, n_embd]
        ggml_tensor * output_norm_w = model->output_norm; // [n_embd]

        if (output_w && output_norm_w) {
            // d_logits is [seq_len, n_vocab]
            // output head: logits = output_w @ hidden  ->  d_hidden = output_w^T @ d_logits
            // d_hidden shape: [seq_len, n_embd]

            int n_embd = model->hparams.n_embd;

            // Compute d_hidden = d_logits @ output_w (matmul: [seq, vocab] @ [vocab, embd] = [seq, embd])
            // output_w is [n_vocab, n_embd] in ggml row-major, so d_hidden = d_logits @ output_w
            std::vector<float> d_hidden(seq_len * n_embd, 0.0f);

            // Dequantize output weight to FP32 for matmul
            std::vector<float> output_w_f32(n_vocab * n_embd);
            if (output_w->type != GGML_TYPE_F32) {
                // Dequantize
                ggml_get_type_traits(output_w->type)->to_float(
                    (const void *)output_w->data, output_w_f32.data(), n_vocab * n_embd);
            } else {
                memcpy(output_w_f32.data(), output_w->data, n_vocab * n_embd * sizeof(float));
            }

            // d_hidden[s, e] = sum_v d_logits[s, v] * output_w[v, e]
            for (int s = 0; s < seq_len; s++) {
                for (int e = 0; e < n_embd; e++) {
                    float sum = 0.0f;
                    const float * dl_row = d_logits.data() + (int64_t)s * n_vocab;
                    for (int v = 0; v < n_vocab; v++) {
                        sum += dl_row[v] * output_w_f32[(int64_t)v * n_embd + e];
                    }
                    d_hidden[s * n_embd + e] = sum;
                }
            }

            // d_hidden is where the backward pass would continue through the layers, with the same
            // pattern per layer: dequantize weight -> matmul for input grad -> accumulate weight grad.

            // Store the output head gradient (d_output_w) for optimizer
            // d_output_w[v, e] = sum_s d_logits[s, v] * hidden[s, e]
            // (requires saved hidden states from forward - available via activation capture)

            (void)d_hidden;  // used by optimizer in full implementation
        }
    }

    llama_batch_free(batch);

    // Store gradient info in tg for the optimizer
    tg->graph = nullptr;  // no unified graph (split mode)
    tg->loss = nullptr;

    return total_loss;
}
