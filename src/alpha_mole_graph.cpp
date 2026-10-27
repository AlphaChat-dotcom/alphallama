#include "alpha_mole_graph.h"

#include "ggml.h"

alpha_mole_route_tensors alpha_mole_build_router_graph(
        ggml_context              * ctx,
        const alpha_mole_graph_router * router,
        ggml_tensor               * hidden) {
    alpha_mole_route_tensors result = {};

    // Layer 1: hidden @ w_down -> intermediate
    // hidden:  [n_embd, n_tokens]
    // w_down:  [n_embd, router_hidden]  (stored column-major for ggml_mul_mat)
    // output:  [router_hidden, n_tokens]
    ggml_tensor * intermediate = ggml_mul_mat(ctx, router->w_down, hidden);

    // Add bias if present
    if (router->b_down) {
        intermediate = ggml_add(ctx, intermediate, router->b_down);
    }

    // ReLU activation
    intermediate = ggml_relu(ctx, intermediate);

    // Layer 2: intermediate @ w_up -> logits
    // intermediate: [router_hidden, n_tokens]
    // w_up:         [router_hidden, n_loras]
    // output:       [n_loras, n_tokens]
    ggml_tensor * logits = ggml_mul_mat(ctx, router->w_up, intermediate);

    // Add bias if present
    if (router->b_up) {
        logits = ggml_add(ctx, logits, router->b_up);
    }

    // Softmax over LoRA dimension
    ggml_tensor * probs = ggml_soft_max(ctx, logits);

    // Top-K selection: returns indices [top_k, n_tokens]
    // Clamp top_k to n_loras to prevent ggml assertion failure
    int k = router->top_k;
    if (k > router->n_loras) { k = router->n_loras; }
    if (k < 1)               { k = 1; }
    result.indices = ggml_argsort_top_k(ctx, probs, k);

    // Extract top-K weights by gathering from probs using indices
    // We need to get the probability values at the selected indices
    // Use ggml_get_rows to gather: for each token, pick the top_k prob values
    //
    // However, ggml_get_rows works differently — it indexes rows, not columns.
    // For per-token weight extraction, we'll use a different approach:
    // Compute weights from the argsort result by re-indexing into probs.
    //
    // For now, store probs and indices — the dispatch function can extract weights.
    result.weights = probs;

    return result;
}

ggml_tensor * alpha_mole_build_lora_dispatch(
        ggml_context                    * ctx,
        ggml_tensor                     * base_result,
        ggml_tensor                     * cur,
        const alpha_mole_stacked_lora   * stacked,
        const alpha_mole_route_tensors  * route) {
    if (!stacked || !route || !route->indices) {
        return base_result;
    }

    // Per-token LoRA dispatch using ggml_mul_mat_id
    // This applies different LoRA weights to different tokens based on routing
    //
    // ggml_mul_mat_id expects:
    //   as  -> [cols, rows, n_expert]
    //   b   -> [cols, n_expert_used, n_tokens]  (3D!)
    //   ids -> [n_expert_used, n_tokens]

    // Reshape cur from [in_features, n_tokens] to [in_features, 1, n_tokens]
    // so ggml_mul_mat_id can broadcast across n_expert_used
    int64_t in_feat  = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, in_feat, 1, n_tokens);

    // Step 1: LoRA down projection — cur through stacked_a
    // stacked_a: [in_features, rank, n_loras]
    // cur_3d:    [in_features, 1, n_tokens]
    // indices:   [top_k, n_tokens]
    // output:    [rank, top_k, n_tokens]
    ggml_tensor * down_out = ggml_mul_mat_id(ctx, stacked->stacked_a, cur_3d, route->indices);

    // Step 2: LoRA up projection — down_out through stacked_b
    // stacked_b: [rank, out_features, n_loras]
    // down_out:  [rank, top_k, n_tokens]
    // output:    [out_features, top_k, n_tokens]
    ggml_tensor * up_out = ggml_mul_mat_id(ctx, stacked->stacked_b, down_out, route->indices);

    // Step 3: Scale by LoRA alpha/rank factor
    if (stacked->scale != 1.0f) {
        up_out = ggml_scale(ctx, up_out, stacked->scale);
    }

    // Step 4: Weight by router probabilities and sum across top_k
    // For now, we sum across the top_k dimension (index 1)
    // Each expert's contribution is weighted by its routing probability
    //
    // The full weighted sum requires:
    // 1. Gather weights from probs at indices
    // 2. Multiply each expert output by its weight
    // 3. Sum across top_k dimension
    //
    // Simplified approach: just sum the top_k experts equally
    // (router weighting can be added when ggml has gather-by-index)
    //
    // up_out shape: [out_features, top_k, n_tokens]
    // We need to sum along dim 1 (top_k) to get [out_features, n_tokens]

    // Create views for each expert slot and sum them
    int64_t out_features = up_out->ne[0];
    // n_tokens already declared above

    ggml_tensor * lora_sum = ggml_view_2d(ctx, up_out,
        out_features, n_tokens,
        up_out->nb[2],   // stride to next token
        0);               // offset for expert 0

    for (int k = 1; k < stacked->n_loras && k < route->indices->ne[0]; k++) {
        ggml_tensor * expert_k = ggml_view_2d(ctx, up_out,
            out_features, n_tokens,
            up_out->nb[2],
            k * up_out->nb[1]);  // offset for expert k
        lora_sum = ggml_add(ctx, lora_sum, expert_k);
    }

    // Add LoRA contribution to base result
    ggml_tensor * result = ggml_add(ctx, base_result, lora_sum);

    return result;
}
