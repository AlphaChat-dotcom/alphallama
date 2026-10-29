// SPDX-License-Identifier: Apache-2.0
#include "alpha_mole_expert_stacker.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Per-expert tensor name patterns.
// Maps stacked tensor type to individual expert naming:
//   ffn_gate_exps -> ffn.experts.%d.w1 (gate/SwiGLU gate)
//   ffn_down_exps -> ffn.experts.%d.w2 (down projection)
//   ffn_up_exps   -> ffn.experts.%d.w3 (up projection)
struct expert_pattern {
    const char * stacked_suffix;   // e.g., "ffn_gate_exps"
    const char * per_expert_fmt;   // e.g., "ffn.experts.%d.w1"
    int layer_tensor_type;         // 0=gate, 1=up, 2=down
};

static const expert_pattern expert_patterns[] = {
    { "ffn_gate_exps",    "ffn.experts.%d.w1", 0 },
    { "ffn_down_exps",    "ffn.experts.%d.w2", 2 },
    { "ffn_up_exps",      "ffn.experts.%d.w3", 1 },
    { "ffn_gate_up_exps", "ffn.experts.%d.w13", 3 },  // merged gate+up (if any converter uses this)
};

bool alpha_mole_has_per_expert_tensors(const struct llama_model * model) {
    if (!model) return false;

    // Check if layer 0 has per-expert tensors by looking at the expert tensor
    // If the expert tensor is NULL or single-expert sized, check for per-expert pattern
    // This is a heuristic; real detection needs the GGUF metadata

    // Check if standard stacked tensors exist
    ggml_tensor * gate_exps = model->layers[0].ffn_gate_exps;
    ggml_tensor * down_exps = model->layers[0].ffn_down_exps;

    if (gate_exps && ggml_n_dims(gate_exps) >= 3 && gate_exps->ne[2] > 1) {
        return false;  // already stacked
    }
    if (down_exps && ggml_n_dims(down_exps) >= 3 && down_exps->ne[2] > 1) {
        return false;  // already stacked
    }

    // If stacked tensors are missing or single-expert, per-expert layout is likely
    return true;
}

bool alpha_mole_stack_experts(struct llama_model * model, const char * model_path) {
    if (!model || !model_path) return false;

    // Open GGUF to find per-expert tensor offsets
    struct gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ nullptr,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "stacker: failed to open GGUF\n");
        return false;
    }

    // Check for per-expert tensors: look for "blk.0.ffn.experts.0.w1.weight"
    char probe_name[256];
    snprintf(probe_name, sizeof(probe_name), "blk.0.ffn.experts.0.w1.weight");
    int probe_idx = gguf_find_tensor(gguf_ctx, probe_name);
    if (probe_idx < 0) {
        // No per-expert tensors found, nothing to stack
        gguf_free(gguf_ctx);
        return false;
    }

    fprintf(stderr, "stacker: detected per-expert tensor layout\n");

    int n_layer = (int)model->layers.size();
    int n_expert = model->hparams.n_expert;

    // Open model file for reading expert data
    FILE * f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "stacker: failed to open model file\n");
        gguf_free(gguf_ctx);
        return false;
    }

    int stacked_count = 0;

    for (int il = 0; il < n_layer; il++) {
        for (int pat_idx = 0; pat_idx < 3; pat_idx++) {  // gate, down, up
            const expert_pattern & pat = expert_patterns[pat_idx];

            // Get the stacked tensor from the model layer
            ggml_tensor * stacked = nullptr;
            switch (pat.layer_tensor_type) {
                case 0: stacked = model->layers[il].ffn_gate_exps; break;
                case 1: stacked = model->layers[il].ffn_up_exps; break;
                case 2: stacked = model->layers[il].ffn_down_exps; break;
                case 3: stacked = model->layers[il].ffn_gate_up_exps; break;
            }

            if (!stacked || !stacked->data) continue;

            size_t expert_slice_bytes = ggml_nbytes(stacked) / (stacked->ne[2] > 1 ? stacked->ne[2] : n_expert);

            // Read each individual expert tensor and copy into the stacked tensor
            for (int eid = 0; eid < n_expert; eid++) {
                char expert_name[256];
                snprintf(expert_name, sizeof(expert_name), "blk.%d.%s.weight", il, "");
                // Build the per-expert tensor name
                char full_name[256];
                snprintf(full_name, sizeof(full_name), "blk.%d.", il);
                char expert_suffix[128];
                snprintf(expert_suffix, sizeof(expert_suffix), pat.per_expert_fmt, eid);
                snprintf(full_name, sizeof(full_name), "blk.%d.%s.weight", il, expert_suffix);

                int tensor_idx = gguf_find_tensor(gguf_ctx, full_name);
                if (tensor_idx < 0) {
                    // Try without .weight suffix
                    snprintf(full_name, sizeof(full_name), "blk.%d.%s", il, expert_suffix);
                    tensor_idx = gguf_find_tensor(gguf_ctx, full_name);
                }
                if (tensor_idx < 0) continue;

                // Read expert data from GGUF file
                size_t data_offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
                size_t dst_offset = (size_t)eid * expert_slice_bytes;

                // Read from file and write to stacked tensor
                std::vector<uint8_t> expert_data(expert_slice_bytes);
                fseek(f, (long)data_offset, SEEK_SET);
                size_t read = fread(expert_data.data(), 1, expert_slice_bytes, f);
                if (read != expert_slice_bytes) {
                    fprintf(stderr, "stacker: short read for %s\n", full_name);
                    continue;
                }

                // Write into stacked tensor at the correct offset
                if (stacked->buffer) {
                    // GPU or managed buffer: use backend API
                    ggml_backend_tensor_set(stacked, expert_data.data(), dst_offset, expert_slice_bytes);
                } else if (stacked->data) {
                    // CPU buffer: direct memcpy
                    memcpy((char *)stacked->data + dst_offset, expert_data.data(), expert_slice_bytes);
                }
            }
            stacked_count++;
        }
    }

    fclose(f);
    gguf_free(gguf_ctx);

    fprintf(stderr, "stacker: stacked %d expert tensors across %d layers\n",
            stacked_count, n_layer);
    return stacked_count > 0;
}
