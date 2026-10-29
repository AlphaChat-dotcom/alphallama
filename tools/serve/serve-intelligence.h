// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaServe Intelligence Layer
//
// Pure data processing - NO llama.cpp dependency.
// Works with any backend (AlphaServe, SGLang, vLLM).
//
// Pre-inference:  BM25 retrieval + NER expansion + Redis + experience injection
// Post-inference: experience recording, analytics (stub)

#include "server-bm25.h"
#include "server-entity.h"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <functional>

using json = nlohmann::ordered_json;

// Forward declarations for optional components
#ifdef ALPHA_HAS_REDIS
class redis_index;
#endif

namespace alphaserve {

struct context_budget {
    int max_code_chars = 10000;
    int max_ner_chars = 2000;
    int max_experience_chars = 5000;
    int max_experience_count = 3;
};

struct intelligence_ctx {
    // Retrieval (required for augmentation)
    bm25_index * bm25 = nullptr;

    // NER (optional - enables extended-code mode)
    entity_index * entities = nullptr;
    entity_linker * linker = nullptr;
    code_entity_extractor * extractor = nullptr;

    // Redis (primary structured retrieval, BM25 as fallback)
#ifdef ALPHA_HAS_REDIS
    redis_index * redis = nullptr;
#endif

    // Config
    int ctx_size = 8192;       // backend context window
    bool use_ner = true;       // enable NER summary + multihop
    bool use_redis = true;     // Redis is primary index (BM25 fallback)
    std::string corpus_dir;    // for experience persistence
};

// Compute context budget based on available context window
context_budget compute_budget(int ctx_size, bool has_tools, int n_history_msgs);

// Pre-inference: augment request JSON with retrieved context
// Mutates body["messages"] in-place (prepends system message)
void intelligence_augment(intelligence_ctx & ctx, json & body);

// Post-inference: record experiences, analytics (stub for now)
void intelligence_post_process(intelligence_ctx & ctx,
                                const json & request,
                                const json & response);

// Initialize intelligence context from corpus directory
// Scans files, builds BM25 index + NER entities
bool intelligence_init_corpus(intelligence_ctx & ctx, const std::string & corpus_dir);

} // namespace alphaserve
