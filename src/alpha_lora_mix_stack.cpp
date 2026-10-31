#include "alpha_lora_mix_stack.h"
#include "alpha_lora_mix_graph.h"
#include "llama-adapter.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <set>

alpha_lora_mix_stacked_weights * alpha_lora_mix_stack_loras(
        llama_adapter_lora ** loras,
        int                   n_loras,
        float               * scales) {
    if (!loras || n_loras <= 0) {
        return nullptr;
    }

    // Collect all tensor names that have LoRA weights in any adapter
    std::set<std::string> all_names;
    for (int i = 0; i < n_loras; i++) {
        if (!loras[i]) return nullptr;
        for (const auto & kv : loras[i]->ab_map) {
            all_names.insert(kv.first);
        }
    }

    if (all_names.empty()) {
        fprintf(stderr, "stack: no LoRA weights found\n");
        return nullptr;
    }

    // Validate uniform rank across all LoRAs and all tensors
    int uniform_rank = -1;
    for (int i = 0; i < n_loras; i++) {
        for (const auto & kv : loras[i]->ab_map) {
            int rank = (int)kv.second.b->ne[0];
            if (uniform_rank < 0) {
                uniform_rank = rank;
            } else if (rank != uniform_rank) {
                fprintf(stderr, "stack: rank mismatch (lora %d tensor '%s' has rank %d, expected %d)\n",
                        i, kv.first.c_str(), rank, uniform_rank);
                return nullptr;
            }
        }
    }

    if (uniform_rank <= 0) {
        fprintf(stderr, "stack: invalid rank %d\n", uniform_rank);
        return nullptr;
    }

    // Compute combined scale (use first LoRA's alpha as reference)
    float combined_scale = 1.0f;
    if (loras[0]->alpha != 0.0f) {
        combined_scale = loras[0]->alpha / (float)uniform_rank;
    }

    // Estimate memory needed for all stacked tensors
    size_t total_mem = 0;
    for (const auto & name : all_names) {
        // Find a representative weight to get dimensions
        for (int i = 0; i < n_loras; i++) {
            auto it = loras[i]->ab_map.find(name);
            if (it != loras[i]->ab_map.end()) {
                int64_t a_elems = ggml_nelements(it->second.a);
                int64_t b_elems = ggml_nelements(it->second.b);
                // stacked_a: [dim0, dim1, n_loras] + stacked_b: same pattern
                total_mem += (size_t)(a_elems * n_loras + b_elems * n_loras) * sizeof(float);
                break;
            }
        }
    }
    // Add overhead for ggml tensor metadata
    total_mem += all_names.size() * 2 * ggml_tensor_overhead() + 1024 * 1024;

    // Create ggml context for stacked tensors
    struct ggml_init_params params = {
        /*.mem_size   =*/ total_mem,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "stack: failed to create ggml context (%zu bytes)\n", total_mem);
        return nullptr;
    }

    auto * sw = new alpha_lora_mix_stacked_weights();
    sw->ctx     = ctx;
    sw->n_loras = n_loras;
    sw->rank    = uniform_rank;
    sw->scale   = combined_scale;

    // Stack weights for each tensor name
    for (const auto & name : all_names) {
        // Find representative dimensions from first LoRA that has this tensor
        int64_t a_ne0 = 0, a_ne1 = 0;  // a: [in_features, rank]
        int64_t b_ne0 = 0, b_ne1 = 0;  // b: [rank, out_features]
        bool found = false;

        for (int i = 0; i < n_loras; i++) {
            auto it = loras[i]->ab_map.find(name);
            if (it != loras[i]->ab_map.end()) {
                a_ne0 = it->second.a->ne[0];
                a_ne1 = it->second.a->ne[1];
                b_ne0 = it->second.b->ne[0];
                b_ne1 = it->second.b->ne[1];
                found = true;
                break;
            }
        }

        if (!found) continue;

        // Create 3D stacked tensors
        // stacked_a: [a_ne0, a_ne1, n_loras]  e.g., [in_features, rank, n_loras]
        ggml_tensor * stacked_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, a_ne0, a_ne1, n_loras);
        // stacked_b: [b_ne0, b_ne1, n_loras]  e.g., [rank, out_features, n_loras]
        ggml_tensor * stacked_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, b_ne0, b_ne1, n_loras);

        if (!stacked_a || !stacked_b) {
            fprintf(stderr, "stack: failed to allocate stacked tensor for '%s'\n", name.c_str());
            ggml_free(ctx);
            delete sw;
            return nullptr;
        }

        // Zero-fill (LoRAs missing this tensor get zero contribution)
        memset(stacked_a->data, 0, ggml_nbytes(stacked_a));
        memset(stacked_b->data, 0, ggml_nbytes(stacked_b));

        // Copy each LoRA's weights into the appropriate slice
        for (int i = 0; i < n_loras; i++) {
            auto it = loras[i]->ab_map.find(name);
            if (it == loras[i]->ab_map.end()) {
                // This LoRA doesn't target this tensor — leave as zeros
                continue;
            }

            const ggml_tensor * src_a = it->second.a;
            const ggml_tensor * src_b = it->second.b;

            // Validate dimensions match
            if (src_a->ne[0] != a_ne0 || src_a->ne[1] != a_ne1 ||
                src_b->ne[0] != b_ne0 || src_b->ne[1] != b_ne1) {
                fprintf(stderr, "stack: dimension mismatch for '%s' in lora %d\n",
                        name.c_str(), i);
                ggml_free(ctx);
                delete sw;
                return nullptr;
            }

            // Copy into slice i of the 3D tensor
            size_t a_slice_bytes = (size_t)a_ne0 * a_ne1 * sizeof(float);
            size_t b_slice_bytes = (size_t)b_ne0 * b_ne1 * sizeof(float);

            float * a_dst = (float *)stacked_a->data + (size_t)i * a_ne0 * a_ne1;
            float * b_dst = (float *)stacked_b->data + (size_t)i * b_ne0 * b_ne1;

            // TODO: handle non-F32 source tensors (dequantize if needed)
            memcpy(a_dst, src_a->data, a_slice_bytes);
            memcpy(b_dst, src_b->data, b_slice_bytes);

            // Apply per-adapter scale if provided
            if (scales && scales[i] != 1.0f) {
                for (size_t j = 0; j < (size_t)a_ne0 * a_ne1; j++) a_dst[j] *= scales[i];
                for (size_t j = 0; j < (size_t)b_ne0 * b_ne1; j++) b_dst[j] *= scales[i];
            }
        }

        alpha_lora_mix_stacked_lora stack = {};
        stack.stacked_a = stacked_a;
        stack.stacked_b = stacked_b;
        stack.scale     = combined_scale;
        stack.n_loras   = n_loras;
        stack.rank      = uniform_rank;

        sw->stacks[name] = stack;
    }

    fprintf(stderr, "stack: stacked %zu tensor groups, %d LoRAs, rank %d\n",
            sw->stacks.size(), n_loras, uniform_rank);

    return sw;
}

void alpha_lora_mix_stacked_weights_free(alpha_lora_mix_stacked_weights * sw) {
    if (!sw) return;
    if (sw->ctx) {
        ggml_free(sw->ctx);
    }
    delete sw;
}
