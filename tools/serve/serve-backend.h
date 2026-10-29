// SPDX-License-Identifier: Apache-2.0
#pragma once

// AlphaServe Backend Abstraction
//
// Backends, one interface:
//   LOCAL:      AlphaServe pipeline (llama_decode on main thread)
//   SGLANG:     HTTP proxy to a remote SGLang server (OpenAI-compatible)
//   VLLM:       HTTP proxy to a remote vLLM server (OpenAI-compatible)
//   FLASHINFER: native FlashInfer GPU engine (created in serve-main.cpp)
//
// SGLANG and VLLM share one OpenAI-compatible HTTP proxy (same /v1/chat/completions
// + /health endpoints); they differ only in the reported backend name.
//
// The intelligence proxy calls backend->complete() or backend->stream()
// after augmenting the request. Backend is interchangeable.

#include <nlohmann/json.hpp>
#include <string>
#include <functional>
#include <memory>

using json = nlohmann::ordered_json;

namespace alphaserve {

enum class BackendType { LOCAL, SGLANG, VLLM, FLASHINFER };

struct BackendConfig {
    BackendType type = BackendType::LOCAL;
    std::string remote_url;      // e.g., "http://localhost:30000"
    std::string remote_api_key;  // optional Bearer token
};

// Abstract backend interface
class backend {
public:
    virtual ~backend() = default;

    // Non-streaming: send request, wait for full response
    virtual json complete(const json & request) = 0;

    // Streaming: send request, call on_chunk for each SSE data line,
    // return assembled final response
    virtual json stream(const json & request,
                        std::function<void(const std::string & sse_line)> on_chunk) = 0;

    // Health check
    virtual bool is_healthy() = 0;

    virtual const char * name() const = 0;
};

// Create backend from config
std::unique_ptr<backend> create_backend(const BackendConfig & config);

} // namespace alphaserve
