// Pipeline server implementation is included from server-context.cpp
// because server_context_impl is defined there (not in a header).
// This file is intentionally empty - the implementation lives in
// server-context.cpp after server_context_impl is defined.
//
// See: server-pipeline.h for the pipeline_server struct definition.

#if 0  // Implementation moved to server-context.cpp

void pipeline_server::start(server_context_impl & ctx_server, llama_context * ctx, int n_slots) {
    if (running.load()) return;
    running.store(true);

    // Initialize backend sampler chains per slot
    // These run ON GPU as part of the llama_decode() compute graph
    for (int i = 0; i < n_slots; i++) {
        llama_sampler_chain_params cparams = llama_sampler_chain_default_params();
        cparams.no_perf = true;
        llama_sampler * chain = llama_sampler_chain_init(cparams);
        // Default sampling: temperature 0.7 + top-k 40 + top-p 0.95
        // Individual requests can override via llama_set_sampler() later
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(42 + i));

        if (!llama_set_sampler(ctx, (llama_seq_id)i, chain)) {
            SRV_WRN("pipeline: failed to set backend sampler for slot %d\n", i);
            llama_sampler_free(chain);
        } else {
            backend_samplers.push_back(chain);
        }
    }

    SRV_INF("pipeline: %d backend samplers attached to GPU context\n", (int)backend_samplers.size());

    // Initialize batch buffers
    for (int b = 0; b < 2; b++) {
        buffers[b].batch = llama_batch_init(2048, 0, n_slots);
        buffers[b].slot_indices.reserve(n_slots);
        buffers[b].sampled.reserve(n_slots);
    }

    // Start threads
    gpu_thread = std::thread([this, ctx]() { gpu_loop(ctx); });
    scheduler_thread = std::thread([this, &ctx_server, ctx]() { scheduler_loop(ctx_server, ctx); });

    SRV_INF("pipeline: started (scheduler + GPU threads, %d slots)\n", n_slots);
}

void pipeline_server::stop() {
    if (!running.load()) return;
    running.store(false);

    // Wake up blocked threads
    cv_batch_ready.notify_all();
    cv_decode_done.notify_all();

    if (scheduler_thread.joinable()) scheduler_thread.join();
    if (gpu_thread.joinable()) gpu_thread.join();

    // Free batch buffers
    for (int b = 0; b < 2; b++) {
        llama_batch_free(buffers[b].batch);
    }

    // Free backend samplers
    for (auto * s : backend_samplers) {
        llama_sampler_free(s);
    }
    backend_samplers.clear();

    SRV_INF("%s", "pipeline: stopped\n");
}

// GPU Thread: decode + backend sampling
// Minimal - just waits for batch, decodes, reads sampled tokens

void pipeline_server::gpu_loop(llama_context * ctx) {
    SRV_INF("%s", "pipeline GPU thread started\n");

    while (running.load()) {
        int buf_idx = gpu_buf.load();
        auto & buf = buffers[buf_idx];

        // Wait for scheduler to prepare a batch
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_batch_ready.wait(lock, [&]() {
                return buf.ready.load() || !running.load();
            });
        }

        if (!running.load()) break;
        if (buf.batch.n_tokens == 0) {
            buf.ready.store(false);
            buf.decoded.store(true);
            cv_decode_done.notify_one();
            continue;
        }

        // Decode (GPU + backend sampling in one pass)
        int ret = llama_decode(ctx, buf.batch);
        if (ret != 0) {
            SRV_ERR("pipeline: llama_decode failed, ret = %d\n", ret);
        }

        // Read pre-sampled tokens (4 bytes each, fast)
        buf.sampled.resize(buf.slot_indices.size());
        for (int i = 0; i < (int)buf.slot_indices.size(); i++) {
            // Backend sampler ran on GPU - just read the result
            llama_token tok = llama_get_sampled_token_ith(ctx, i);
            buf.sampled[i] = tok;
        }

        // Signal scheduler that decode is done
        buf.ready.store(false);
        buf.decoded.store(true);
        cv_decode_done.notify_one();
    }

    SRV_INF("%s", "pipeline GPU thread stopped\n");
}

// Scheduler Thread: builds batches + processes results
// Alternates between:
//   1. Processing results from completed GPU buffer
//   2. Building next batch for GPU

void pipeline_server::scheduler_loop(server_context_impl & ctx_server, llama_context * ctx) {
    SRV_INF("%s", "pipeline scheduler thread started\n");

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    while (running.load()) {
        // Phase 1: Process completed decode results
        int done_idx = gpu_buf.load();
        auto & done_buf = buffers[done_idx];

        if (done_buf.decoded.load()) {
            // Process sampled tokens from completed buffer
            for (int i = 0; i < (int)done_buf.slot_indices.size(); i++) {
                int slot_id = done_buf.slot_indices[i];
                if (slot_id < 0 || slot_id >= (int)ctx_server.slots.size()) continue;

                auto & slot = ctx_server.slots[slot_id];
                if (slot.state != SLOT_STATE_GENERATING) continue;

                llama_token tok = done_buf.sampled[i];
                if (tok == LLAMA_TOKEN_NULL) {
                    // Backend sampler didn't fire - fall back to CPU sampling
                    tok = common_sampler_sample(slot.smpl.get(), ctx, i);
                }

                common_sampler_accept(slot.smpl.get(), tok, true);
                slot.n_decoded += 1;
                slot.prompt.tokens.push_back(tok);

                // TODO: process_token, send_partial_response, stop detection
                // For now, just track the token
            }

            done_buf.decoded.store(false);
            done_buf.slot_indices.clear();
        }

        // Phase 2: Build next batch
        int prep_idx = prep_buf.load();
        auto & prep = buffers[prep_idx];

        common_batch_clear(prep.batch);
        prep.slot_indices.clear();
        prep.n_generating = 0;

        for (auto & slot : ctx_server.slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.n_decoded > 0) {
                // Add the last sampled token to the batch
                auto tokens = slot.prompt.tokens.get_tokens();
                if (!tokens.empty()) {
                    llama_token last_tok = tokens.back();
                    common_batch_add(prep.batch, last_tok,
                                     slot.prompt.tokens.pos_next() - 1,
                                     { slot.id }, true);
                    prep.slot_indices.push_back(slot.id);
                    prep.n_generating++;
                }
            }
        }

        if (prep.batch.n_tokens > 0) {
            // Signal GPU thread
            prep.ready.store(true);
            cv_batch_ready.notify_one();

            // Swap buffers
            int old_gpu = gpu_buf.exchange(prep_idx);
            prep_buf.store(old_gpu);

            // Wait for GPU to finish
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_decode_done.wait_for(lock, std::chrono::milliseconds(100), [&]() {
                    return buffers[gpu_buf.load()].decoded.load() || !running.load();
                });
            }
        } else {
            // No work - brief sleep to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    SRV_INF("%s", "pipeline scheduler thread stopped\n");
}
#endif  // if 0 - implementation in server-context.cpp
