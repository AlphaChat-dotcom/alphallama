#include "alpha_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

// Profile file format (binary):
//   Header: "AMEP" (4 bytes) + version (uint32) + n_layers + n_experts + n_hot
//   Model hash: 64 bytes (SHA256 hex string of model file)
//   Per layer: n_hot expert IDs (int32 each, sorted by frequency desc)
//   Per layer: n_experts frequency counts (int32 each)

#define AMEP_MAGIC "AMEP"
#define AMEP_VERSION 1

struct alpha_expert_profile {
    int n_layers;
    int n_experts;
    int n_expert_used;
    int n_hot;

    char model_hash[65];  // SHA256 hex + null

    // Per-layer hot expert IDs [n_layers][n_hot]
    std::vector<std::vector<int>> hot_experts;

    // Per-layer expert activation frequency [n_layers][n_experts]
    std::vector<std::vector<int>> freq;

    // Total tokens used for calibration
    int calibration_tokens;
};

alpha_expert_profile * alpha_expert_profile_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return nullptr;

    // Read header
    char magic[4];
    uint32_t version, n_layers, n_experts, n_hot;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, AMEP_MAGIC, 4) != 0) {
        fclose(f);
        return nullptr;
    }
    if (fread(&version, 4, 1, f) != 1 || version != AMEP_VERSION) {
        fclose(f);
        return nullptr;
    }
    if (fread(&n_layers, 4, 1, f) != 1) { fclose(f); return nullptr; }
    if (fread(&n_experts, 4, 1, f) != 1) { fclose(f); return nullptr; }
    if (fread(&n_hot, 4, 1, f) != 1) { fclose(f); return nullptr; }

    auto * p = new alpha_expert_profile();
    p->n_layers = (int)n_layers;
    p->n_experts = (int)n_experts;
    p->n_hot = (int)n_hot;
    p->n_expert_used = 0;
    p->calibration_tokens = 0;

    // Read model hash
    if (fread(p->model_hash, 1, 64, f) != 64) {
        delete p;
        fclose(f);
        return nullptr;
    }
    p->model_hash[64] = '\0';

    // Read hot experts per layer
    p->hot_experts.resize(p->n_layers);
    for (int il = 0; il < p->n_layers; il++) {
        p->hot_experts[il].resize(p->n_hot);
        if (fread(p->hot_experts[il].data(), sizeof(int), p->n_hot, f) != (size_t)p->n_hot) {
            delete p;
            fclose(f);
            return nullptr;
        }
    }

    // Read frequency counts per layer
    p->freq.resize(p->n_layers);
    for (int il = 0; il < p->n_layers; il++) {
        p->freq[il].resize(p->n_experts);
        if (fread(p->freq[il].data(), sizeof(int), p->n_experts, f) != (size_t)p->n_experts) {
            delete p;
            fclose(f);
            return nullptr;
        }
    }

    fclose(f);
    fprintf(stderr, "alpha_expert_profile: loaded profile from %s (%d layers, %d experts, %d hot)\n",
            path, p->n_layers, p->n_experts, p->n_hot);
    return p;
}

bool alpha_expert_profile_save(const alpha_expert_profile * p, const char * path) {
    if (!p || !path) return false;

    FILE * f = fopen(path, "wb");
    if (!f) return false;

    // Write header
    fwrite(AMEP_MAGIC, 1, 4, f);
    uint32_t version = AMEP_VERSION;
    uint32_t n_layers = (uint32_t)p->n_layers;
    uint32_t n_experts = (uint32_t)p->n_experts;
    uint32_t n_hot = (uint32_t)p->n_hot;
    fwrite(&version, 4, 1, f);
    fwrite(&n_layers, 4, 1, f);
    fwrite(&n_experts, 4, 1, f);
    fwrite(&n_hot, 4, 1, f);

    // Write model hash
    fwrite(p->model_hash, 1, 64, f);

    // Write hot experts
    for (int il = 0; il < p->n_layers; il++) {
        fwrite(p->hot_experts[il].data(), sizeof(int), p->n_hot, f);
    }

    // Write frequencies
    for (int il = 0; il < p->n_layers; il++) {
        fwrite(p->freq[il].data(), sizeof(int), p->n_experts, f);
    }

    fclose(f);
    fprintf(stderr, "alpha_expert_profile: saved profile to %s\n", path);
    return true;
}

alpha_expert_profile * alpha_expert_profile_generate(
        struct llama_model   * model,
        struct llama_context * ctx,
        const char * calibration_text,
        int n_tokens) {
    if (!model || !ctx || !calibration_text) return nullptr;

    int n_layers = llama_model_n_layer(model);
    int n_expert = 0;
    int n_expert_used = 0;

    // Read expert count from model metadata
    char buf[64];
    for (int i = 0; i < llama_model_meta_count(model); i++) {
        char key[128] = {};
        llama_model_meta_key_by_index(model, i, key, sizeof(key));
        if (strstr(key, "expert_count")) {
            llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
            n_expert = atoi(buf);
        } else if (strstr(key, "expert_used_count")) {
            llama_model_meta_val_str_by_index(model, i, buf, sizeof(buf));
            n_expert_used = atoi(buf);
        }
    }

    if (n_expert <= 1) {
        fprintf(stderr, "alpha_expert_profile: not a MoE model\n");
        return nullptr;
    }

    auto * p = new alpha_expert_profile();
    p->n_layers = n_layers;
    p->n_experts = n_expert;
    p->n_expert_used = n_expert_used;
    p->n_hot = std::min(n_expert, std::max(n_expert_used * 2, 4)); // heuristic: 2x top-K or 4
    p->calibration_tokens = n_tokens;
    memset(p->model_hash, '0', 64);
    p->model_hash[64] = '\0';

    // Initialize frequency counters
    p->freq.resize(n_layers);
    for (auto & v : p->freq) v.assign(n_expert, 0);
    p->hot_experts.resize(n_layers);
    for (auto & v : p->hot_experts) v.resize(p->n_hot, 0);

    fprintf(stderr, "alpha_expert_profile: generating profile with %d calibration tokens...\n", n_tokens);

    // TODO: Run calibration tokens through the model and record expert activations
    // This requires intercepting the argsort output per layer per token.
    // For now: use a uniform frequency as placeholder
    for (int il = 0; il < n_layers; il++) {
        for (int e = 0; e < n_expert; e++) {
            // Placeholder: assume power-law distribution (expert 0 most active)
            p->freq[il][e] = n_tokens / (e + 1);
        }
        // Sort by frequency to get hot experts
        std::vector<int> indices(n_expert);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return p->freq[il][a] > p->freq[il][b]; });
        for (int k = 0; k < p->n_hot; k++) {
            p->hot_experts[il][k] = indices[k];
        }
    }

    fprintf(stderr, "alpha_expert_profile: generated (%d layers, %d experts, %d hot per layer)\n",
            n_layers, n_expert, p->n_hot);
    return p;
}

const int * alpha_expert_profile_get_hot(const alpha_expert_profile * p, int layer) {
    if (!p || layer < 0 || layer >= p->n_layers) return nullptr;
    return p->hot_experts[layer].data();
}

float alpha_expert_profile_hit_rate(const alpha_expert_profile * p, int K) {
    if (!p || K <= 0) return 0.0f;

    // Estimate hit rate: what fraction of expert activations are covered by top-K cached
    float total_hits = 0;
    float total_activations = 0;

    for (int il = 0; il < p->n_layers; il++) {
        // Sum frequency of top-K experts
        std::vector<int> sorted_freq = p->freq[il];
        std::sort(sorted_freq.begin(), sorted_freq.end(), std::greater<int>());

        int layer_total = 0;
        int layer_hits = 0;
        for (int e = 0; e < p->n_experts; e++) {
            layer_total += sorted_freq[e];
            if (e < K) layer_hits += sorted_freq[e];
        }
        if (layer_total > 0) {
            total_hits += (float)layer_hits;
            total_activations += (float)layer_total;
        }
    }

    return total_activations > 0 ? total_hits / total_activations : 0.0f;
}

const char * alpha_expert_profile_model_hash(const alpha_expert_profile * p) {
    return p ? p->model_hash : nullptr;
}

void alpha_expert_profile_free(alpha_expert_profile * p) {
    delete p;
}
