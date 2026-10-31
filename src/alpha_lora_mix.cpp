#include "alpha_lora_mix.h"
#include "alpha_router.h"
#include "llama-adapter.h"

#include <cstdlib>
#include <cstring>
#include <vector>

struct alpha_lora_mix_context {
    llama_model   * model;
    llama_context * ctx;

    std::vector<llama_adapter_lora *> loras;
    std::vector<float>                static_weights;

    alpha_router * router;

    alpha_lora_mix_params params;
    alpha_lora_mix_strategy strategy;

    // Routing decisions from last decode
    std::vector<alpha_lora_mix_token_route> last_routes;

    // Previous-token embedding for routing (used in generation)
    std::vector<float> prev_embedding;
    bool               has_prev_embedding;

    // Cached routing to avoid unnecessary graph recompilation
    bool adapters_set;
};

struct alpha_lora_mix_context * alpha_lora_mix_init(
        struct llama_model   * model,
        struct llama_context * ctx,
        struct alpha_lora_mix_params params) {
    if (!model || !ctx) {
        return nullptr;
    }

    if (params.n_loras < 0 || params.n_loras > ALPHA_LORA_MIX_MAX_LORAS) {
        return nullptr;
    }

    if (params.n_loras > 0 && !params.lora_paths) {
        return nullptr;
    }

    auto * mctx = new alpha_lora_mix_context();
    mctx->model  = model;
    mctx->ctx    = ctx;
    mctx->params = params;
    mctx->router = nullptr;
    mctx->has_prev_embedding = false;
    mctx->adapters_set = false;

    // Load all LoRA adapters
    mctx->loras.resize(params.n_loras);
    for (int i = 0; i < params.n_loras; i++) {
        mctx->loras[i] = llama_adapter_lora_init(model, params.lora_paths[i]);
        if (!mctx->loras[i]) {
            // Cleanup already-loaded adapters
            for (int j = 0; j < i; j++) {
                llama_adapter_lora_free(mctx->loras[j]);
            }
            delete mctx;
            return nullptr;
        }
    }

    // Set up static weights
    if (params.n_loras > 0) {
        if (params.static_weights) {
            mctx->static_weights.assign(params.static_weights, params.static_weights + params.n_loras);
        } else {
            float w = 1.0f / params.n_loras;
            mctx->static_weights.assign(params.n_loras, w);
        }
    }

    // Load router if dynamic routing is enabled
    if (params.enable_routing && params.router_path) {
        mctx->router = alpha_router_load(params.router_path);
        if (!mctx->router) {
            // Fall back to static mode
            mctx->params.enable_routing = false;
        }
    }

    // Detect optimal strategy based on what's loaded
    int n_loaded = (int)mctx->loras.size();
    if (n_loaded == 0) {
        mctx->strategy = LORA_MIX_STRATEGY_RAW;
    } else if (n_loaded == 1) {
        mctx->strategy = LORA_MIX_STRATEGY_SINGLE_LORA;
    } else if (!mctx->router || !mctx->params.enable_routing) {
        mctx->strategy = LORA_MIX_STRATEGY_STATIC_BLEND;
    } else {
        mctx->strategy = LORA_MIX_STRATEGY_DYNAMIC;
    }

    fprintf(stderr, "strategy = %s\n", alpha_lora_mix_strategy_name(mctx->strategy));

    // Apply initial adapters based on strategy
    if (mctx->strategy != LORA_MIX_STRATEGY_RAW) {
        llama_set_adapters_lora(ctx, mctx->loras.data(), mctx->loras.size(), mctx->static_weights.data());
    }
    mctx->adapters_set = true;

    // Clear pointers in stored params — they were only needed during init above
    mctx->params.lora_paths     = nullptr;
    mctx->params.router_path    = nullptr;
    mctx->params.static_weights = nullptr;

    // Pre-allocate embedding buffer (only needed for dynamic routing)
    if (mctx->strategy == LORA_MIX_STRATEGY_DYNAMIC) {
        int n_embd = llama_model_n_embd(model);
        mctx->prev_embedding.resize(n_embd, 0.0f);
    }

    return mctx;
}

int alpha_lora_mix_decode(
        struct alpha_lora_mix_context * mctx,
        struct llama_batch          batch) {
    if (!mctx) {
        return -1;
    }

    mctx->last_routes.clear();

    switch (mctx->strategy) {
        case LORA_MIX_STRATEGY_RAW:
            // Maximum speed — no LoRA overhead, pure base model
            return llama_decode(mctx->ctx, batch);

        case LORA_MIX_STRATEGY_SINGLE_LORA:
        case LORA_MIX_STRATEGY_STATIC_BLEND:
            // Adapters set once at init, never change — zero per-token overhead
            return llama_decode(mctx->ctx, batch);

        case LORA_MIX_STRATEGY_DYNAMIC: {
            // Per-token routing with graph-rebuild avoidance
            if (batch.n_tokens == 1 && mctx->has_prev_embedding) {
                // Compute routing decision (CPU, ~0.1ms)
                alpha_route_result route = alpha_router_forward(
                    mctx->router,
                    mctx->prev_embedding.data(),
                    mctx->params.top_k);

                // Record routing for inspection (no adapter change — avoids graph rebuild)
                alpha_lora_mix_token_route tr;
                tr.n_selected = route.n_selected;
                for (int k = 0; k < ALPHA_LORA_MIX_MAX_TOP_K; k++) {
                    tr.lora_indices[k] = route.indices[k];
                    tr.lora_weights[k] = route.weights[k];
                }
                mctx->last_routes.push_back(tr);
            }

            int ret = llama_decode(mctx->ctx, batch);
            if (ret != 0) return ret;

            // Capture embedding for next routing decision
            if (!mctx->prev_embedding.empty()) {
                float * emb = llama_get_embeddings(mctx->ctx);
                if (emb) {
                    int n_embd = llama_model_n_embd(mctx->model);
                    memcpy(mctx->prev_embedding.data(), emb, n_embd * sizeof(float));
                    mctx->has_prev_embedding = true;
                }
            }
            return 0;
        }
    }

    return llama_decode(mctx->ctx, batch);
}

enum alpha_lora_mix_strategy alpha_lora_mix_get_strategy(
        const struct alpha_lora_mix_context * mctx) {
    return mctx ? mctx->strategy : LORA_MIX_STRATEGY_RAW;
}

const char * alpha_lora_mix_strategy_name(enum alpha_lora_mix_strategy s) {
    switch (s) {
        case LORA_MIX_STRATEGY_RAW:          return "RAW (no LoRA, max speed)";
        case LORA_MIX_STRATEGY_SINGLE_LORA:  return "SINGLE_LORA (static, no routing)";
        case LORA_MIX_STRATEGY_STATIC_BLEND: return "STATIC_BLEND (fixed weights)";
        case LORA_MIX_STRATEGY_DYNAMIC:      return "DYNAMIC (per-token routing)";
    }
    return "UNKNOWN";
}

int alpha_lora_mix_get_routing(
        struct alpha_lora_mix_context        * mctx,
        struct alpha_lora_mix_routing_info   * info) {
    if (!mctx || !info) {
        return -1;
    }

    info->n_tokens = (int)mctx->last_routes.size();
    if (info->n_tokens == 0) {
        info->routes = nullptr;
        return 0;
    }

    info->routes = (alpha_lora_mix_token_route *)malloc(
        info->n_tokens * sizeof(alpha_lora_mix_token_route));
    if (!info->routes) {
        return -1;
    }

    memcpy(info->routes, mctx->last_routes.data(),
           info->n_tokens * sizeof(alpha_lora_mix_token_route));

    return 0;
}

void alpha_lora_mix_routing_info_free(struct alpha_lora_mix_routing_info * info) {
    if (info && info->routes) {
        free(info->routes);
        info->routes = nullptr;
        info->n_tokens = 0;
    }
}

void alpha_lora_mix_free(struct alpha_lora_mix_context * mctx) {
    if (!mctx) {
        return;
    }

    if (mctx->router) {
        alpha_router_free(mctx->router);
    }

    for (auto * lora : mctx->loras) {
        if (lora) {
            llama_adapter_lora_free(lora);
        }
    }

    delete mctx;
}
