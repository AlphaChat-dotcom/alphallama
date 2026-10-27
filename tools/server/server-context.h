#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-bm25.h"
#include "server-entity.h"
#include "server-metadata.h"

// Corpus context for tool registration - defined here to avoid circular deps
struct corpus_tool_context {
    bm25_index * bm25 = nullptr;
    entity_index * entities = nullptr;
    entity_linker * linker = nullptr;
    code_entity_extractor * extractor = nullptr;
    std::string model_name;  // for model-tagged experiences
};

#include <nlohmann/json_fwd.hpp>

#include <cstddef>
#include <memory>
#include <set>

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    json json_webui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
};

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // async pipeline loop for multi-user throughput (--mode pipeline)
    void start_pipeline_loop();

    // terminate main loop (will unblock start_loop or start_pipeline_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // check if MoE LUT-direct mode is active (for two-pass refinement)
    bool is_moe_lut_active() const;
    // get portfolio K estimate and VRAM (for feature gating)
    int get_moe_K_estimate() const;
    size_t get_moe_gpu_vram_mb() const;

    // register a callback to be called when sleeping state changes
    // must be set before load_model() is called
    void on_sleeping_changed(std::function<void(bool)> callback);

    // get corpus tool context for tool registration (BM25, NER, entity index)
    corpus_tool_context get_corpus_tool_context() const;

};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t get_api_show;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_ollama_chat;      // Ollama /api/chat (native format)
    server_http_context::handler_t post_ollama_generate;  // Ollama /api/generate
    server_http_context::handler_t get_ollama_ps;         // Ollama /api/ps (running models)
    server_http_context::handler_t post_ollama_pull;      // Ollama /api/pull (stub)
    server_http_context::handler_t delete_ollama_model;   // Ollama /api/delete (stub)
    server_http_context::handler_t post_ollama_copy;      // Ollama /api/copy (stub)
    server_http_context::handler_t post_ollama_embed;     // Ollama /api/embed (stub)
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;

    // KV-chunk endpoints for long-context system
    server_http_context::handler_t post_prefill_chunk;
    server_http_context::handler_t post_activate_chunks;
    server_http_context::handler_t post_decode_step;
    server_http_context::handler_t post_kv_extract_centroids;
    server_http_context::handler_t post_kv_save;
    server_http_context::handler_t post_kv_load;
    server_http_context::handler_t post_kv_index;   // BM25 chunk indexing
    server_http_context::handler_t post_kv_route;   // BM25 query routing
    server_http_context::handler_t post_expert_routing; // Expert routing data for profiling
    server_http_context::handler_t post_oracle;          // 0.8B error oracle for user help
#ifdef ALPHA_DELTANET
    server_http_context::handler_t post_deltanet_load;   // Load DeltaNet checkpoint dynamically
#endif
    server_http_context::handler_t post_records_index;   // POST /api/records/index - define + ingest structured records
    server_http_context::handler_t post_records_aggregate; // POST /api/records/aggregate - FT.AGGREGATE over all records
    server_http_context::handler_t post_dense_index;     // POST /api/dense/index - embed + store chunks (dense-L0)
    server_http_context::handler_t post_dense_build_ann; // POST /api/dense/build_ann -- (re)build the IVF index (dense-L0)
    server_http_context::handler_t post_dense_search;    // POST /api/dense/search - semantic top-k (dense-L0)

    // KV pool endpoints
    server_http_context::handler_t get_kv_pool_stats;    // GET /api/kv_pool/stats
    server_http_context::handler_t post_kv_pool_save;    // POST /api/kv_pool/save (session persist)
    server_http_context::handler_t post_kv_pool_load;    // POST /api/kv_pool/load (session restore)

    // Batch inference endpoint
    server_http_context::handler_t post_batch_completions;  // POST /api/batch (multi-prompt)

    void init_kv_chunk_routes();
    void corpus_augment_messages(json & body);  // auto-retrieve for extended-code chat
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    struct server_tools * tools_ptr = nullptr;  // set after tool registration for auto-tool-call

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    const server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);
};
