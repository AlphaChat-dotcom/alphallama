#include "serve-backend.h"

#define CPPHTTPLIB_NO_EXCEPTIONS
#include <cpp-httplib/httplib.h>

#include <sstream>
#include <cstdio>

namespace alphaserve {

// OpenAI-compatible HTTP proxy backend
// Forwards OpenAI-compatible requests to a remote inference server via HTTP.
// Used for both SGLang and vLLM - they expose identical /v1/chat/completions +
// /health endpoints, so one implementation serves both (differing only in name_).
// Streaming: reads SSE chunks and relays them to the caller.

class openai_proxy_backend : public backend {
public:
    openai_proxy_backend(const std::string & display_name,
                         const std::string & base_url, const std::string & api_key)
        : name_(display_name), base_url_(base_url), api_key_(api_key) {
        // Parse URL into host + port
        std::string url = base_url;
        if (url.substr(0, 7) == "http://") url = url.substr(7);
        else if (url.substr(0, 8) == "https://") url = url.substr(8);

        auto colon = url.find(':');
        if (colon != std::string::npos) {
            host_ = url.substr(0, colon);
            port_ = std::stoi(url.substr(colon + 1));
        } else {
            host_ = url;
            port_ = 80;
        }
        // Strip trailing slash
        while (!host_.empty() && host_.back() == '/') host_.pop_back();

        fprintf(stderr, "[backend] %s backend: %s:%d\n", name_.c_str(), host_.c_str(), port_);
    }

    json complete(const json & request) override {
        httplib::Client cli(host_, port_);
        cli.set_read_timeout(120, 0);  // 2 min timeout
        cli.set_write_timeout(10, 0);

        // Ensure stream is false
        json req = request;
        req["stream"] = false;

        httplib::Headers headers = {{"Content-Type", "application/json"}};
        if (!api_key_.empty()) {
            headers.emplace("Authorization", "Bearer " + api_key_);
        }

        auto res = cli.Post("/v1/chat/completions", headers, req.dump(), "application/json");
        if (!res) {
            return json{{"error", name_ + " connection failed"}};
        }
        if (res->status != 200) {
            return json{{"error", name_ + " returned " + std::to_string(res->status)},
                        {"body", res->body}};
        }

        try {
            return json::parse(res->body);
        } catch (...) {
            return json{{"error", "Failed to parse " + name_ + " response"}};
        }
    }

    json stream(const json & request,
                std::function<void(const std::string & sse_line)> on_chunk) override {
        httplib::Client cli(host_, port_);
        cli.set_read_timeout(120, 0);
        cli.set_write_timeout(10, 0);

        json req = request;
        req["stream"] = true;

        httplib::Headers headers = {{"Content-Type", "application/json"}};
        if (!api_key_.empty()) {
            headers.emplace("Authorization", "Bearer " + api_key_);
        }

        // Assemble full response text from streamed chunks
        std::string full_text;
        std::string model_name;

        auto res = cli.Post("/v1/chat/completions", headers, req.dump(), "application/json",
            [&](const char * data, size_t len) -> bool {
                std::string chunk(data, len);

                // SSE can deliver multiple events in one chunk
                std::istringstream ss(chunk);
                std::string line;
                while (std::getline(ss, line)) {
                    // Strip \r
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;

                    if (line == "data: [DONE]") {
                        on_chunk("data: [DONE]\n\n");
                        continue;
                    }

                    if (line.substr(0, 6) == "data: ") {
                        // Pass through to client
                        on_chunk(line + "\n\n");

                        // Extract text for post-processing
                        try {
                            auto j = json::parse(line.substr(6));
                            if (j.contains("model")) model_name = j["model"];
                            if (j.contains("choices") && !j["choices"].empty()) {
                                auto & c = j["choices"][0];
                                if (c.contains("delta")) {
                                    if (c["delta"].contains("content")) {
                                        full_text += c["delta"]["content"].get<std::string>();
                                    }
                                }
                            }
                        } catch (...) {
                            // Non-JSON line, ignore
                        }
                    }
                }
                return true;  // continue reading
            });

        // Build assembled response for post-processing
        return json{
            {"model", model_name},
            {"choices", json::array({json{
                {"message", json{{"role", "assistant"}, {"content", full_text}}},
                {"finish_reason", "stop"},
            }})},
        };
    }

    bool is_healthy() override {
        httplib::Client cli(host_, port_);
        cli.set_read_timeout(5, 0);
        auto res = cli.Get("/health");
        return res && res->status == 200;
    }

    const char * name() const override { return name_.c_str(); }

private:
    std::string name_;
    std::string base_url_;
    std::string api_key_;
    std::string host_;
    int port_ = 80;
};

// Factory

std::unique_ptr<backend> create_backend(const BackendConfig & config) {
    switch (config.type) {
        case BackendType::SGLANG:
            return std::make_unique<openai_proxy_backend>("sglang", config.remote_url, config.remote_api_key);
        case BackendType::VLLM:
            return std::make_unique<openai_proxy_backend>("vllm", config.remote_url, config.remote_api_key);
        case BackendType::LOCAL:
        default:
            // Local + FlashInfer backends are created by serve-main.cpp directly
            // (need the AlphaServe server + response router, which are main-specific)
            return nullptr;
    }
}

} // namespace alphaserve
