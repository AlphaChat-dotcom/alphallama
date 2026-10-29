// SPDX-License-Identifier: Apache-2.0
// AlphaServe Dual-Backend Intelligence Proxy
//
// Architecture:
//   HTTP thread:       accept connections, parse JSON
//   Main thread:       GPU decode loop (local) OR idle (sglang)
//   Scheduler thread:  batch building (local only)
//   Intelligence:      BM25 + NER + Redis augmentation (any backend)
//
// Backends:
//   --backend local   : AlphaServe pipeline (default, needs -m)
//   --backend fi       : native FlashInfer GPU engine (needs -m)
//   --backend sglang  : Forward to a remote SGLang server (needs --backend-url)
//   --backend vllm    : Forward to a remote vLLM server  (needs --backend-url)
//
// Usage:
//   alphallama-serve -m model.gguf --backend local [--corpus-dir /path]
//   alphallama-serve --backend sglang --backend-url http://localhost:30000 [--corpus-dir /path]
//   alphallama-serve --backend vllm   --backend-url http://localhost:8000  [--corpus-dir /path]

#include "alphaserve.h"
#include "alpha_env_audit.h"
#include "build-info.h"
#include <cstring>
#include "serve-response.h"
#include "serve-backend.h"
#include "serve-intelligence.h"

#ifdef ALPHA_HAS_FLASHINFER
#include "serve-forward.h"
#include "serve-paged-kv.h"
#include "serve-fi-engine.h"
#include "llama-model.h"
#endif
#include "serve-ai-engine.h"   // empty unless ALPHA_HAS_ALPHAINFER (independent of FlashInfer)
#include "common.h"
#include "chat.h"
#include "serve-mm.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef ALPHA_DELTANET
#include "../../src/alpha_deltanet_state.h" // DeltaNet prescan state load (serving parity with alphallama)
#endif
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

// Marlin INT4 Tensor Core hook for ggml Q4_K decode
extern "C" void marlin_hook_enable();
extern "C" void marlin_reset_locks(void *);

// Alpha-MoLE: persistent expert cache for GPU-accelerated MoE offloading
typedef bool   (*ggml_expert_pcache_fn_t)(void*, void*, const void*, size_t, int64_t, int64_t, size_t);
typedef void   (*ggml_expert_pcache_flush_fn_t)(void*);
typedef void * (*ggml_expert_pcache_redirect_fn_t)(void*, void*);
// ggml-cuda entry points; without the CUDA backend they are no-ops from serve-cuda-stubs.cpp, declared
// extern "C" there because a dllimport declaration would not pull the stub object out of the static library.
#ifndef ALPHA_CUDA_API_C
#  ifdef GGML_USE_CUDA
#    define ALPHA_CUDA_API_C GGML_BACKEND_API_C
#  else
#    define ALPHA_CUDA_API_C extern "C"
#  endif
#endif
ALPHA_CUDA_API_C bool   ggml_cuda_expert_pcache_copy(void*, void*, const void*, size_t, int64_t, int64_t, size_t);
ALPHA_CUDA_API_C void   ggml_cuda_expert_pcache_flush(void*);
ALPHA_CUDA_API_C void * ggml_cuda_expert_pcache_get_extra(void);
GGML_API_C void ggml_backend_set_expert_cache_copy_fn(ggml_expert_pcache_fn_t);
GGML_API_C void ggml_backend_set_expert_cache_flush_fn(ggml_expert_pcache_flush_fn_t);
GGML_API_C void ggml_backend_set_expert_cache_redirect_fn(ggml_expert_pcache_redirect_fn_t);
typedef void * (*ggml_expert_pcache_extra_fn_t)(void);
GGML_API_C void ggml_backend_set_expert_cache_extra_fn(ggml_expert_pcache_extra_fn_t);
typedef void (*ggml_expert_pcache_ntokens_fn_t)(int, int, int, int);
ALPHA_CUDA_API_C void ggml_cuda_expert_pcache_set_ntokens(int, int, int, int);
GGML_API_C void ggml_backend_set_expert_cache_ntokens_fn(ggml_expert_pcache_ntokens_fn_t);
typedef void (*ggml_expert_pcache_pass_begin_fn_t)(void);
ALPHA_CUDA_API_C void ggml_cuda_expert_pcache_pass_begin(void);
GGML_API_C void ggml_backend_set_expert_cache_pass_begin_fn(ggml_expert_pcache_pass_begin_fn_t);
ALPHA_CUDA_API_C void ggml_cuda_expert_pcache_prealloc(void);
ALPHA_CUDA_API_C void ggml_cuda_set_pool_reserve_bytes(size_t);   // KV/compute-aware pool reserve
typedef void (*ggml_expert_pcache_ids_fn_t)(const int32_t *, int, int, int, int);
ALPHA_CUDA_API_C void ggml_cuda_expert_pcache_set_ids(const int32_t *, int, int, int, int);
GGML_API_C void ggml_backend_set_expert_cache_ids_fn(ggml_expert_pcache_ids_fn_t);
// Layer-fuse (ALPHA_LAYER_FUSE): MoE layer boundary, so the cache can fetch a whole layer's
// expert misses in one flush instead of one flush per expert tensor.
typedef void (*ggml_expert_pcache_group_fn_t)(const void *, int);
ALPHA_CUDA_API_C void ggml_cuda_expert_pcache_set_group(const void *, int);
GGML_API_C void ggml_backend_set_expert_cache_group_fn(ggml_expert_pcache_group_fn_t);

// ALPHA_DRE (device-resolved experts): the scheduler leaves decode expert weights in place
typedef bool (*ggml_sched_direct_weight_fn_t)(const struct ggml_tensor *, const struct ggml_tensor *);
GGML_API_C void ggml_backend_sched_set_direct_weight_fn(ggml_sched_direct_weight_fn_t);
ALPHA_CUDA_API_C bool ggml_cuda_dre_direct_weight(const struct ggml_tensor *, const struct ggml_tensor *);
ALPHA_CUDA_API_C bool ggml_cuda_dre_prefill_copy(void *, void *, const void *, size_t, int64_t, int64_t, size_t);
ALPHA_CUDA_API_C void ggml_cuda_dre_prefill_flush(void *);
ALPHA_CUDA_API_C void * ggml_cuda_dre_prefill_redirect(void *, void *);
ALPHA_CUDA_API_C bool ggml_cuda_dre_prefill_stream_whole(void *, const void *, int);
ALPHA_CUDA_API_C void ggml_cuda_dre_prefill_ids(const int32_t *, int, int, int, int);
ALPHA_CUDA_API_C void ggml_cuda_dre_prefill_x(void *, const struct ggml_tensor *, const void *, const int32_t *, int, int, int, int);
ALPHA_CUDA_API_C bool ggml_cuda_dre_tier_busy(void);   // ALPHA_DRE_TIER, dre.cu
ALPHA_CUDA_API_C void ggml_cuda_dre_tier_defer(void);
ALPHA_CUDA_API_C void ggml_cuda_dre_tier_start(void);
GGML_API_C void ggml_backend_set_expert_cache_pfx_fn(void (*)(void *, const struct ggml_tensor *, const void *, const int32_t *, int, int, int, int));
extern "C" void ggml_backend_set_expert_cache_stream_fn(bool (*fn)(void *, const void *, int));
ALPHA_CUDA_API_C void ggml_cuda_dre_preregister(const struct ggml_tensor *);

// Multi-user expert cache partitioning
ALPHA_CUDA_API_C void ggml_cuda_init_expert_users(int n_users);
ALPHA_CUDA_API_C void ggml_cuda_set_expert_ctx(void * ctx);

// Auto serve-mode: programmatic CPU-inline override (enable<0/split<0 = leave unchanged).
ALPHA_CUDA_API_C void ggml_cuda_expert_pcache_set_cpu_inline(int enable, int split_pct);

#define CPPHTTPLIB_NO_EXCEPTIONS
// The default httplib worker pool (~hardware_concurrency) caps simultaneous long-lived connections,
// which caps the decode batch width. Raise it so 64-128 concurrent users (each holding one blocking
// POST) can all be in flight at once.
#define CPPHTTPLIB_THREAD_POOL_COUNT 320
#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <csignal>
#include <memory>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_set>   // /api/corpus/query union-mode dedup
#include <cctype>          // /api/corpus/query span selector
// POSIX-only. execinfo (backtrace) and dirent have no MSVC equivalent; on Windows the directory
// scan below uses std::filesystem instead.
#if defined(_WIN32)
#  include <io.h>
#  include <filesystem>
#else
#  include <execinfo.h>
#  include <unistd.h>
#  include <dirent.h>
#endif
#include <sys/stat.h>
// Image input (--mmproj):
// OpenAI chat content parts {"type":"image_url","image_url":{"url":"data:image/...;base64,..."}} (also "input_image" with
// "image_url" as a plain string) become the projector's media marker in the text and a bitmap each, in order.
// Reasoning budget:
// Server default --reasoning-budget N / ALPHA_REASONING_BUDGET (-1 = unlimited); per request "reasoning_budget" (alias
// "thinking_budget", also inside chat_template_kwargs) and "reasoning_budget_message" (the line written before
// </think>; "" for none). Works on /v1/chat/completions (text and image) and /v1/completions.
static int         g_reasoning_budget = -1;
static std::string g_reasoning_budget_msg = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n";
template <typename J>
static void alpha_reasoning_budget(const J & body, const std::string & prompt, const llama_vocab * vocab,
                                   alphaserve::InferenceRequest & ireq) {
    int budget = g_reasoning_budget;
    std::string msg = g_reasoning_budget_msg;
    auto pick = [&](const J & o) {
        for (const char * k : { "reasoning_budget", "thinking_budget" })
            if (o.contains(k) && o[k].is_number_integer()) budget = o[k].template get<int>();
        if (o.contains("reasoning_budget_message") && o["reasoning_budget_message"].is_string())
            msg = o["reasoning_budget_message"].template get<std::string>();
    };
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) pick(body["chat_template_kwargs"]);
    pick(body);
    if (budget < 0) return;
    ireq.reasoning_budget = budget;
    // template-opened thinking: the generation prompt ends with "<think>" (a model that writes <think> itself is
    // detected when it does)
    const size_t o = prompt.rfind("<think>"), c = prompt.rfind("</think>");
    ireq.think_open = o != std::string::npos && (c == std::string::npos || c < o) && prompt.size() - o <= 16;
    ireq.think_close = common_tokenize(vocab, msg + "</think>\n\n", false, true);
}

static std::string alpha_b64_decode(const std::string & in) {
    static int T[256]; static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; i++) T[i] = -1;
        const char * a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) T[(unsigned char) a[i]] = i;
        T[(unsigned char) '-'] = 62; T[(unsigned char) '_'] = 63;   // url-safe alphabet too
        init = true;
    }
    std::string out; out.reserve(in.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        const int d = T[c];
        if (d < 0) continue;   // whitespace / newlines
        val = (val << 6) + d; bits += 6;
        if (bits >= 0) { out.push_back((char) ((val >> bits) & 0xFF)); bits -= 8; }
    }
    return out;
}
struct alpha_mm_bitmaps {
    std::vector<mtmd_bitmap *> v;
    ~alpha_mm_bitmaps() { for (auto * b : v) mtmd_bitmap_free(b); }
};
static bool alpha_mm_has_images(const nlohmann::json & msgs) {
    if (!msgs.is_array()) return false;
    for (const auto & m : msgs) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (const auto & part : m["content"]) {
            const std::string t = part.is_object() ? part.value("type", std::string()) : std::string();
            if (t == "image_url" || t == "input_image") return true;
        }
    }
    return false;
}
// rewrites image parts into marker text parts; false + err on a part it cannot read
static bool alpha_mm_extract(nlohmann::json & msgs, alpha_mm_bitmaps & out, std::string & err) {
    for (auto & m : msgs) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (auto & part : m["content"]) {
            const std::string t = part.is_object() ? part.value("type", std::string()) : std::string();
            if (t != "image_url" && t != "input_image") continue;
            std::string url;
            if (part.contains("image_url")) {
                const auto & iu = part["image_url"];
                if (iu.is_string()) url = iu.get<std::string>();
                else if (iu.is_object() && iu.contains("url") && iu["url"].is_string()) url = iu["url"].get<std::string>();
            }
            if (url.rfind("data:", 0) != 0) { err = "image_url must be a data: URL (data:image/...;base64,...); remote URLs are not fetched"; return false; }
            const size_t comma = url.find(',');
            if (comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos) { err = "image_url data: URL must be base64-encoded"; return false; }
            const std::string bytes = alpha_b64_decode(url.substr(comma + 1));
            mtmd_bitmap * bm = bytes.empty() ? nullptr :
                mtmd_helper_bitmap_init_from_buf(alphaserve::g_serve_mtmd, (const unsigned char *) bytes.data(), bytes.size());
            if (!bm) { err = "could not decode image " + std::to_string(out.v.size() + 1) + " (png / jpeg / bmp / gif expected)"; return false; }
            out.v.push_back(bm);
            part = nlohmann::json{ { "type", "text" }, { "text", mtmd_default_marker() } };
        }
    }
    return true;
}

// setenv() is POSIX; MSVC provides _putenv_s with the arguments reversed and no overwrite flag.
#if defined(_WIN32)
// POSIX setenv() on top of _putenv_s, with the overwrite flag honoured
static inline int setenv(const char * k, const char * v, int overwrite) {
    if (!overwrite && getenv(k)) return 0;
    return _putenv_s(k, v);
}
#endif

static inline int alpha_setenv(const char * k, const char * v) {
    return setenv(k, v, 1);
}

// Free/total VRAM of the current device; false (and zeros) on a build without CUDA.
static bool alpha_vram_info(size_t * free_b, size_t * total_b) {
#ifdef GGML_USE_CUDA
    if (cudaMemGetInfo(free_b, total_b) == cudaSuccess) {
        return true;
    }
    cudaGetLastError();
#endif
    *free_b = *total_b = 0;
    return false;
}

// MSVC defines _S_IFDIR but not the POSIX S_ISDIR test macro.
#if defined(_WIN32) && !defined(S_ISDIR)
#  define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

// Diagnostic: print a C++ backtrace on SIGSEGV. Enabled via ALPHA_SEGV_TRACE=1.
#if !defined(_WIN32)   // backtrace()/backtrace_symbols_fd() are glibc-only
static void alpha_segv_handler(int sig) {
    void * bt[64];
    int n = backtrace(bt, 64);
    const char * msg = "\n=== ALPHA SIGSEGV BACKTRACE ===\n";
    ssize_t rc = write(2, msg, strlen(msg)); (void)rc;
    backtrace_symbols_fd(bt, n, 2);
    _exit(139);
}
#endif
#include <string>
#include <atomic>
#include <chrono>

static int s_n_expert_peek = 0;   // expert_count from the startup GGUF peek (0 = dense / unknown)

using json = nlohmann::ordered_json;

static std::atomic<bool> g_running{true};

static void signal_handler(int) {
    g_running.store(false);
}

// JSON helpers

static json make_completion_response(int request_id, const std::string & text, int n_tokens,
                                     const std::string & model_name, const std::string & finish_reason) {
    return json{
        {"id",      "cmpl-" + std::to_string(request_id)},
        {"object",  "text_completion"},
        {"created", (int64_t)time(nullptr)},
        {"model",   model_name},
        {"choices", json::array({json{
            {"text",          text},
            {"index",         0},
            {"logprobs",      nullptr},
            {"finish_reason", finish_reason},
        }})},
        {"usage", json{
            {"prompt_tokens",     0},
            {"completion_tokens", n_tokens},
            {"total_tokens",      n_tokens},
        }},
    };
}

// Length of the longest prefix of `s` that consists only of complete UTF-8 codepoints.
// If `s` ends in the middle of a multi-byte sequence (a lead byte without all its
// continuation bytes yet), the returned length excludes that trailing partial sequence.
// Used by the streaming path to hold incomplete bytes back until the next token piece,
// so a codepoint split across two pieces (CJK/emoji) is emitted whole. Malformed (not
// just incomplete) input is left in place for make_sse_chunk's replace-on-dump to sanitize.
static size_t utf8_complete_prefix_len(const std::string & s) {
    size_t n = s.size();
    if (n == 0) return 0;
    // walk back over continuation bytes (10xxxxxx) to find the last lead byte
    size_t i = n;
    while (i > 0 && (((unsigned char)s[i - 1] & 0xC0) == 0x80)) i--;
    if (i == 0) return n;                       // all continuation bytes: malformed, don't hold
    unsigned char lead = (unsigned char)s[i - 1];
    size_t need;
    if      ((lead & 0x80) == 0x00) need = 1;   // ASCII
    else if ((lead & 0xE0) == 0xC0) need = 2;
    else if ((lead & 0xF0) == 0xE0) need = 3;
    else if ((lead & 0xF8) == 0xF0) need = 4;
    else return n;                              // invalid lead byte: malformed, don't hold
    size_t have = n - (i - 1);                  // bytes present from the lead byte to the end
    return (have < need) ? (i - 1) : n;         // incomplete trailing sequence -> hold it back
}

// Per-request no-progress timeout. A request whose decode never makes progress (dropped intake
// or a stalled decode) releases its httplib worker thread after this window instead of blocking
// forever, which would eventually exhaust the pool and hang the server, /health included. Resets
// on every token; must exceed worst-case prefill.
static std::chrono::milliseconds request_idle_timeout() {
    static const int s = [] {
        const char * e = getenv("ALPHA_REQ_IDLE_TIMEOUT_S");
        int v = e ? atoi(e) : 120;
        return v > 0 ? v : 120;
    }();
    return std::chrono::seconds(s);
}

// Time-to-first-token (prefill) budget, separate from the inter-token idle window above. No token
// is emitted during prefill, so a large prompt (100K+ tokens, especially with expert offload) can
// legitimately go minutes without progress.
static std::chrono::milliseconds request_prefill_timeout() {
    static const int s = [] {
        const char * e = getenv("ALPHA_REQ_PREFILL_TIMEOUT_S");
        int v = e ? atoi(e) : 900;
        return v > 0 ? v : 900;
    }();
    return std::chrono::seconds(s);
}

// Reasoning-channel stripper (chat "content").
// Some instruct models emit a "thinking" channel before the final answer:
//   gemma4-A4B:  "<|channel>thought\n<channel|>ANSWER"   (answer follows the last <channel|>)
//   R1-style:    "<think>...</think>ANSWER"
// OpenAI-chat "content" should be the final answer only, so this drops the reasoning preamble.
// A no-op unless a known channel-close delimiter is present.
// Disable with ALPHA_NO_STRIP_THINK=1.
static std::string strip_reasoning_channel(const std::string & in) {
    static const bool disabled = getenv("ALPHA_NO_STRIP_THINK") != nullptr;
    if (disabled) return in;
    std::string s = in;
    auto after_last = [&](const std::string & delim) {
        size_t p = s.rfind(delim);
        if (p != std::string::npos) s = s.substr(p + delim.size());
    };
    after_last("<channel|>");   // gemma4 thinking-channel close (answer follows)
    after_last("</think>");     // DeepSeek-R1-style reasoning close
    // Scrub any residual stray channel/thought markers.
    for (const std::string & m : { std::string("<|channel>"), std::string("<channel|>"),
                                   std::string("<|thought"),  std::string("<|message|>") }) {
        size_t p;
        while ((p = s.find(m)) != std::string::npos) s.erase(p, m.size());
    }
    // Trim leading whitespace/newlines left by the removed preamble.
    size_t b = s.find_first_not_of(" \t\r\n");
    s = (b == std::string::npos) ? std::string() : s.substr(b);
    return s;
}

static std::string make_sse_chunk(int request_id, const std::string & text,
                                  const std::string & model_name, bool is_chat, bool is_done,
                                  const char * finish_reason = "stop") {
    json delta;
    if (is_done) {
        delta = json{
            {"id",      (is_chat ? "chatcmpl-" : "cmpl-") + std::to_string(request_id)},
            {"object",  is_chat ? "chat.completion.chunk" : "text_completion"},
            {"created", (int64_t)time(nullptr)},
            {"model",   model_name},
            {"choices", json::array({json{
                {"index",         0},
                {"delta",         json::object()},
                {"finish_reason", finish_reason},
            }})},
        };
    } else {
        json content = is_chat
            ? json{{"role", "assistant"}, {"content", text}}
            : json{{"text", text}};
        delta = json{
            {"id",      (is_chat ? "chatcmpl-" : "cmpl-") + std::to_string(request_id)},
            {"object",  is_chat ? "chat.completion.chunk" : "text_completion"},
            {"created", (int64_t)time(nullptr)},
            {"model",   model_name},
            {"choices", json::array({json{
                {"index",         0},
                {"delta",         content},
                {"finish_reason", nullptr},
            }})},
        };
    }
    // error_handler_t::replace: never throw on invalid or partial UTF-8 (emit U+FFFD instead).
    // A throw here would happen on the httplib worker thread and terminate the whole server. The
    // hold-back in the streaming lambdas keeps split codepoints intact; this is the last line of
    // defense.
    return "data: " + delta.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

// A chat stream delta carrying reasoning text (the DeepSeek / llama.cpp "reasoning_content" field).
static std::string make_sse_reasoning_chunk(int request_id, const std::string & text, const std::string & model_name) {
    json delta = json{
        {"id",      "chatcmpl-" + std::to_string(request_id)},
        {"object",  "chat.completion.chunk"},
        {"created", (int64_t)time(nullptr)},
        {"model",   model_name},
        {"choices", json::array({json{
            {"index",         0},
            {"delta",         json{{"role", "assistant"}, {"reasoning_content", text}}},
            {"finish_reason", nullptr},
        }})},
    };
    return "data: " + delta.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

// Parse the model's native chat output into an OpenAI assistant message (content, reasoning_content, tool_calls)
// with the common_chat parser of the request's template. The streaming and the non-streaming paths both use it, so a
// reply comes back the same way either way. Malformed output degrades to plain content, never an error.
static void parse_chat_reply(const common_chat_params & cchat, const std::string & text, int request_id,
                             json & message, std::string & finish_reason) {
    message = json{{"role", "assistant"}};
    finish_reason = "stop";
    common_chat_parser_params pp(cchat);   // format + generation_prompt from the template
    pp.parse_tool_calls = true;
    if (!cchat.parser.empty()) pp.parser.load(cchat.parser);   // the PEG parser spec
    common_chat_msg parsed;
    bool ok = true;
    try { parsed = common_chat_parse(text, /*is_partial=*/false, pp); }
    catch (...) { ok = false; }
    if (ok && (!parsed.tool_calls.empty() || !parsed.content.empty())) {
        std::string clean = strip_reasoning_channel(parsed.content);  // drop empty <think></think> marker
        message["content"] = clean.empty() ? json(nullptr) : json(clean);
        if (!parsed.reasoning_content.empty()) message["reasoning_content"] = parsed.reasoning_content;
        if (!parsed.tool_calls.empty()) {
            json tcs = json::array();
            int ti = 0;
            for (auto & tc : parsed.tool_calls) {
                tcs.push_back(json{
                    {"id", tc.id.empty() ? ("call_" + std::to_string(request_id) + "_" + std::to_string(ti)) : tc.id},
                    {"type", "function"},
                    {"function", json{{"name", tc.name}, {"arguments", tc.arguments}}}});
                ti++;
            }
            message["tool_calls"] = tcs;
            finish_reason = "tool_calls";
        }
    } else {
        message["content"] = strip_reasoning_channel(text);  // parse gave nothing -> raw
    }
}

// Streaming form of parsed tool calls: one chunk whose delta carries every call with its index (OpenAI chunk shape).
static std::string make_sse_tool_calls_chunk(int request_id, const json & tool_calls, const std::string & model_name) {
    json tcs = json::array();
    for (size_t i = 0; i < tool_calls.size(); i++) {
        json tc = tool_calls[i];
        tc["index"] = (int) i;
        tcs.push_back(tc);
    }
    json delta = json{
        {"id",      "chatcmpl-" + std::to_string(request_id)},
        {"object",  "chat.completion.chunk"},
        {"created", (int64_t)time(nullptr)},
        {"model",   model_name},
        {"choices", json::array({json{
            {"index",         0},
            {"delta",         json{{"role", "assistant"}, {"content", nullptr}, {"tool_calls", tcs}}},
            {"finish_reason", nullptr},
        }})},
    };
    return "data: " + delta.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

// Main

// ALPHA_AUTO: one decision for the whole local engine, from the model file and the GPU.
// A bare `alphallama-serve -m model.gguf` picks the arrangement a tuned launcher would, for any model and any
// card: everything in VRAM when it fits, device-resolved expert offload when a MoE does not, and MTP when the
// model carries a draft head (in the file, or a sidecar next to it). It decides from model metadata and free
// VRAM only, never from a model name, and expresses the decision as the same ALPHA_* variables the engine
// already reads, set with overwrite=0 so an explicit setting always wins. ALPHA_AUTO=0 disables it.
static void alpha_auto_profile(const char * model_path, int n_slots, bool np_explicit, bool ctx_explicit, char ** argv) {
    const char * ae = getenv("ALPHA_AUTO");
    if (ae && atoi(ae) == 0) { fprintf(stderr, "[auto] ALPHA_AUTO=0: no automatic profile\n"); return; }
    // second pass after the re-exec below: the decision is already in the environment
    if (getenv("ALPHA_AUTO_APPLIED")) { fprintf(stderr, "[auto] profile applied (process re-executed with it)\n"); return; }
    if (!model_path) return;

    struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * gc = gguf_init_from_file(model_path, gp);
    if (!gc) return;
    std::string arch;
    { int64_t k = gguf_find_key(gc, "general.architecture"); if (k >= 0) arch = gguf_get_val_str(gc, k); }
    auto u32 = [&](const std::string & key) -> int64_t {
        int64_t k = gguf_find_key(gc, key.c_str());
        if (k < 0) return -1;
        switch (gguf_get_kv_type(gc, k)) {
            case GGUF_TYPE_UINT32: return gguf_get_val_u32(gc, k);
            case GGUF_TYPE_INT32:  return gguf_get_val_i32(gc, k);
            default: return -1;
        }
    };
    const int64_t n_expert = u32(arch + ".expert_count");
    const int64_t nextn    = u32(arch + ".nextn_predict_layers");
    const bool recurrent   = gguf_find_key(gc, (arch + ".ssm.state_size").c_str()) >= 0 ||
                             gguf_find_key(gc, (arch + ".ssm.conv_kernel").c_str()) >= 0;
    bool mtp_tensors = false;
    for (int64_t i = 0; i < gguf_get_n_tensors(gc) && !mtp_tensors; i++) {
        const char * n = gguf_get_tensor_name(gc, i);
        mtp_tensors = strstr(n, "nextn") != nullptr || strstr(n, ".mtp.") != nullptr;
    }
    // token_embd stays in host memory whatever -ngl says, so it does not count against VRAM
    size_t embd_bytes = 0;
    { const int64_t te = gguf_find_tensor(gc, "token_embd.weight"); if (te >= 0) embd_bytes = gguf_get_tensor_size(gc, te); }
    // routed experts' mean bits per weight, by size over every *_exps tensor in this file (per-layer quants may be
    // mixed); prompt chunking below keys on it. 0 when this shard has no expert tensors.
    double exp_bpw = 0.0;
    {
        double bytes = 0.0, elems = 0.0;
        for (int64_t i = 0; i < gguf_get_n_tensors(gc); i++) {
            if (!strstr(gguf_get_tensor_name(gc, i), "_exps")) continue;
            const ggml_type t = gguf_get_tensor_type(gc, i);
            const double sz = (double) gguf_get_tensor_size(gc, i);
            bytes += sz;
            elems += sz / (double) ggml_type_size(t) * (double) ggml_blck_size(t);
        }
        if (elems > 0.0) exp_bpw = 8.0 * bytes / elems;
    }
    gguf_free(gc);
    const bool is_moe = n_expert > 0;

    // model bytes, across split shards (<prefix>-00001-of-0000N.gguf)
    size_t model_bytes = 0;
    {
        std::string mp(model_path);
        struct stat st;
        size_t pos = mp.rfind("-00001-of-");
        if (pos != std::string::npos && mp.size() >= pos + 15) {
            const int n_shards = atoi(mp.c_str() + pos + 10);
            const std::string prefix = mp.substr(0, pos), suffix = mp.substr(pos + 15);
            for (int s = 1; s <= n_shards; s++) {
                char shard[1024];
                snprintf(shard, sizeof(shard), "%s-%05d-of-%05d%s", prefix.c_str(), s, n_shards, suffix.c_str());
                if (stat(shard, &st) == 0) model_bytes += (size_t) st.st_size;
            }
        } else if (stat(model_path, &st) == 0) {
            model_bytes = (size_t) st.st_size;
        }
    }

    size_t free_vram = 0, total_vram = 0;
#ifdef GGML_USE_CUDA
    alpha_vram_info(&free_vram, &total_vram);
#endif
    if (total_vram == 0) { fprintf(stderr, "[auto] no CUDA device: no automatic GPU profile\n"); return; }
    { const char * fv = getenv("ALPHA_FORCE_VRAM_MB"); if (fv && atol(fv) > 0) total_vram = (size_t) atol(fv) * 1024 * 1024; }

    // Same fit rule as the fused-infer auto below (model + 2 GB for KV/compute vs the card).
    const size_t GB = 1024ull * 1024 * 1024;
    const bool fits = model_bytes + 2 * GB <= total_vram;
    // Expert-pool fraction, same estimate as pick_serve_mode: how much of the expert bytes the VRAM pool can hold.
    double pool_frac = 1.0;
    if (is_moe && !fits) {
        const int backbone_pct = n_expert <= 8 ? 30 : n_expert <= 64 ? 12 : n_expert <= 128 ? 8 : 4;
        const size_t expert_bytes = model_bytes * (100 - backbone_pct) / 100;
        const size_t non_expert   = model_bytes - expert_bytes;
        const size_t usable = total_vram > non_expert + 4 * GB ? total_vram - non_expert - 4 * GB : 0;
        pool_frac = expert_bytes ? (double) usable / (double) expert_bytes : 1.0;
    }
    // Device-resolved experts (decode reads experts through a device table; misses are computed by CPU workers and
    // joined through a doorbell) for every MoE that does not fit: faster than the expert cache at every pool size
    // tried.
    const bool dre = is_moe && !fits;
    // A dense model that misses the 2 GB margin but whose GPU-side weights still fit with ~0.6 GB for a short KV and
    // the compute buffers: put every layer on the GPU (MTP off, 4K context) rather than partial offload. Each layer
    // left on the CPU costs ~17%, and MTP buys nothing once layers are on the CPU.
    const bool tight = !is_moe && !fits && model_bytes > embd_bytes &&
                       model_bytes - embd_bytes + 600ull * 1024 * 1024 <= free_vram;

    // MTP draft head: in the file, or a sidecar GGUF: next to the model with "mtp" in its name, or any GGUF in MTP/.
    std::string sidecar;
    const bool head_in_file = nextn > 0 || mtp_tensors;
#if !defined(_WIN32)
    if (!head_in_file && !getenv("ALPHA_EXTRA_GGUF")) {
        std::string mp(model_path), dir = ".";
        size_t sl = mp.find_last_of('/');
        if (sl != std::string::npos) dir = mp.substr(0, sl);
        for (const std::string & d : { dir, dir + "/MTP" }) {
            DIR * dd = opendir(d.c_str());
            if (!dd) continue;
            while (struct dirent * e = readdir(dd)) {
                std::string n = e->d_name, lower = n;
                for (auto & c : lower) c = (char) tolower((unsigned char) c);
                if (lower.size() < 5 || lower.compare(lower.size() - 5, 5, ".gguf") != 0) continue;
                // in <dir>/MTP the directory names the role: any file there qualifies; next to the model the
                // name must say "mtp"
                if (d == dir && lower.find("mtp") == std::string::npos) continue;
                const std::string full = d + "/" + n;
                if (full == mp) continue;
                // a sidecar is a small file holding only the draft head, never another full model whose name
                // happens to contain "mtp" (e.g. a 27B-MTP file next to the plain 27B)
                struct stat sst;
                if (stat(full.c_str(), &sst) != 0 || (size_t) sst.st_size * 8 > model_bytes) continue;
                struct gguf_context * sc = gguf_init_from_file(full.c_str(), gp);
                if (!sc) continue;
                std::string sarch;
                { int64_t k = gguf_find_key(sc, "general.architecture"); if (k >= 0) sarch = gguf_get_val_str(sc, k); }
                bool head = false;
                for (int64_t t = 0; t < gguf_get_n_tensors(sc) && !head; t++) head = strstr(gguf_get_tensor_name(sc, t), "nextn") != nullptr;
                gguf_free(sc);
                // the model's own architecture, or a sidecar variant of it ("qwen4exp-mtp-sidecar"), carrying the head
                const bool arch_ok = sarch == arch || (sarch.rfind(arch + "-", 0) == 0 && sarch.find("mtp") != std::string::npos);
                // prefer a q8 sidecar when several match (the head is small; precision is cheap)
                if (arch_ok && head && (sidecar.empty() || lower.find("q8") != std::string::npos)) sidecar = full;
            }
            closedir(dd);
        }
    }
#endif
    const bool has_head = head_in_file || !sidecar.empty() || getenv("ALPHA_EXTRA_GGUF") != nullptr;
    // MTP on for resident dense models and for DRE offload. Off for a resident MoE (neutral to slower on this
    // engine) and for expert-cache offload, where every verify column re-stalls on misses. Also off for a dense
    // model that only fits tightly or is partly on the CPU: no gain there, while its head and KV cost VRAM that
    // could hold another layer.
    const bool mtp = has_head && ((!is_moe && fits) || dre);

    int changed = 0;
    auto set = [&changed](const char * k, const char * v) { if (!getenv(k)) { setenv(k, v, 1); changed++; } };
    if (!sidecar.empty() && mtp) set("ALPHA_EXTRA_GGUF", sidecar.c_str());
    if (dre) {
        set("ALPHA_DRE", "1"); set("ALPHA_DRE_CPU", "1"); set("ALPHA_DRE_GHOST", "0"); set("ALPHA_DRE_PF_STREAM", "1");
        // decode graphs are reused across the per-token expert table changes
        set("ALPHA_GRAPH_SEMCMP", "1"); set("ALPHA_GRAPH_CACHE_MAX", "1024"); set("ALPHA_GRAPH_SHAPE_KEY", "3");
        // several slots: the reserve for the grown prefill micro-batch assumes every slot's sequence, and prompt
        // staging then runs out of memory. A 1024-token prefill micro-batch keeps the reserve small.
        if (np_explicit && n_slots > 1) set("ALPHA_DRE_PREFILL_UB", "1024");
        // One slot on a 16 GB-class card or larger: an 11520-token prompt micro-batch instead of the 4096 default, so a
        // long prompt streams the experts in 2 chunks instead of 4-6 (each chunk touches nearly every expert). 8 GB
        // cards keep 4096 (the grown micro-batch costs ~1 GB of compute buffer at init).
        // 16 GB-class card with larger experts (mean >= 2.6 bits per weight, see exp_bpw): 18432, so a 16-18K prompt is
        // one chunk. With larger experts the copies, not the compute, set the pace; with smaller experts the copies hide
        // under the compute and the bigger borrow's restore is pure cost, so they keep 11520. 24576 runs out of memory
        // at ~23K on 16 GB. Larger cards keep 11520.
        else if (total_vram >= 15ull * GB && total_vram < 20ull * GB && exp_bpw >= 2.6) set("ALPHA_DRE_PREFILL_UB", "18432");
        else if (total_vram >= 15ull * GB) set("ALPHA_DRE_PREFILL_UB", "11520");
#if !defined(_WIN32)
        // a routing profile shipped beside the model (<dir>/dre-profile*.bin or <dir>/MTP/dre-profile*.bin, the format
        // ALPHA_DRE_PROFILE_OUT writes) seeds the expert pool with the most-used experts. A profile from another
        // model only misplaces the seed (its blk/expert counts are checked); ALPHA_DRE_PROFILE= disables.
        if (!getenv("ALPHA_DRE_PROFILE")) {
            std::string mp(model_path), dir = ".", found;
            size_t sl = mp.find_last_of('/');
            if (sl != std::string::npos) dir = mp.substr(0, sl);
            for (const std::string & d : { dir, dir + "/MTP" }) {
                DIR * dd = opendir(d.c_str());
                if (!dd) continue;
                while (struct dirent * e = readdir(dd)) {
                    const std::string n = e->d_name;
                    if (n.rfind("dre-profile", 0) == 0 && n.size() > 4 && n.compare(n.size() - 4, 4, ".bin") == 0) { found = d + "/" + n; break; }
                }
                closedir(dd);
                if (!found.empty()) break;
            }
            if (!found.empty()) set("ALPHA_DRE_PROFILE", found.c_str());
        }
#endif
    }
    if (mtp) {
        set("ALPHA_MTP", "1");
        set("ALPHA_MTP_LOOP", "32");
        if (recurrent) set("ALPHA_REPLAYSSM", "1");
        // depth: 2 under DRE (each verify column re-stalls on CPU misses; deeper is slower), 3 for dense
        set("ALPHA_MTP_DRAFT_N", dre ? "2" : "3");
        if (dre) set("ALPHA_MTP_NO_SNAP", "1");
    }
    // Serving shape when no -np / -c is given: every configuration above (MTP, DRE, recurrent state, prefix reuse)
    // is validated at one slot with 16K context; some sparse-attention paths assert with several streams.
    // Carried through the re-exec in ALPHA_AUTO_NP / ALPHA_AUTO_CTX; main() applies them.
    if (!np_explicit && (mtp || dre || recurrent || tight)) { set("ALPHA_AUTO_NP", "1"); n_slots = 1; }
    if (tight) { set("ALPHA_AUTO_NGL", "999"); if (!ctx_explicit) set("ALPHA_AUTO_CTX", "4096"); }
    if (!ctx_explicit) set("ALPHA_AUTO_CTX", "16384");
    // Recurrent-checkpoint prefix reuse: byte-identical output, much lower multi-turn TTFT (declines itself where unsafe).
    if (recurrent && n_slots == 1) set("ALPHA_HYBRID_PREFIX", "1");

    // report what is effective (an explicit setting may have overridden the decision)
    auto on = [](const char * k) { const char * v = getenv(k); return v && atoi(v) != 0; };
    const bool dre_eff = on("ALPHA_DRE"), mtp_eff = on("ALPHA_MTP"), pfx_eff = on("ALPHA_HYBRID_PREFIX");
    const double GBd = (double) GB;
    if (getenv("ALPHA_AUTO_NP") || getenv("ALPHA_AUTO_CTX"))
        fprintf(stderr, "[auto] serving shape: %s slot(s) x %s context (pass -np / -c to choose)\n",
                getenv("ALPHA_AUTO_NP") ? getenv("ALPHA_AUTO_NP") : "default", getenv("ALPHA_AUTO_CTX") ? getenv("ALPHA_AUTO_CTX") : "default");
    fprintf(stderr, "[auto] %s: %s%s, %.1f GB on a %.1f GB GPU -> %s%s%s%s%s  (ALPHA_AUTO=0 disables; any ALPHA_* you set wins)\n",
            arch.c_str(), is_moe ? "MoE" : "dense", recurrent ? " (recurrent)" : "",
            model_bytes / GBd, total_vram / GBd,
            fits ? "all in VRAM" : getenv("ALPHA_AUTO_NGL") ? "all layers in VRAM (tight fit: MTP off, short context)" : (dre_eff ? "expert offload, device-resolved (DRE)" : (is_moe ? "expert offload, expert cache" : "partial layer offload")),
            (is_moe && !fits) ? (std::string(", experts in pool ~") + std::to_string((int) (pool_frac * 100)) + "%").c_str() : "",
            mtp_eff ? ", MTP on" : (has_head ? (mtp ? ", MTP off (operator)" : ", MTP off (no gain here)") : ""),
            (sidecar.empty() || !mtp_eff) ? "" : (std::string(", MTP sidecar ") + sidecar).c_str(),
            pfx_eff ? ", prefix reuse on" : "");
#if defined(__linux__)
    // Several engine settings are read by global initializers (e.g. the MTP switches in serve-gpu-engine.cpp), which
    // run before main(), so a setenv() here comes too late for them. When the profile changed anything, restart this
    // binary once with it.
    if (changed > 0) {
        setenv("ALPHA_AUTO_APPLIED", "1", 1);
        fflush(stderr); fflush(stdout);
        execv("/proc/self/exe", argv);
        fprintf(stderr, "[auto] re-exec failed; settings read before main() keep their defaults\n");
    }
#endif
}

#ifdef ALPHA_HAS_ALPHAINFER
// ALPHA_AUTO: pick the AlphaInfer backend for a bare -m, but only inside the scope it is validated for.
// Auto picks it only for a validated model file (by SHA-256, cached per path/size/mtime so the hash of a large
// file is paid once) on the matching GPU class. Everything else stays on the built-in engine. ALPHA_AUTO=0 or
// an explicit --backend disables it.
// Each row is a validated (model file, GPU class, serving shape). A row is added only after token-exactness vs
// plain through the C API, a soak at the shape, and a check of VRAM headroom after deep context.
// ALPHA_SPEED_MODE=1 selects the best-speed mode for benchmarks (production keeps exact): fast MTP at the row's
// fast_k. Fast verify is not token-identical to plain greedy. An explicit AI_MTP_MODE / ALPHA_AI_MTP_K still wins.
struct alpha_ai_validated { const char * sha256; long long size; int sm_major, sm_minor; int n_streams; int ctx_total;
                            const char * mtp_mode;   // AI_MTP_MODE for this row (nullptr = the engine policy default)
                            int fast_k;              // MTP depth in ALPHA_SPEED_MODE (fast verify), best for this row
                            // ALPHA_SPEED_MODE drafter "<dir>/<file>": the DFlash2 drafter in that exact sibling directory of the
                            // model directory (or <file> beside the model), or ALPHA_AI_DRAFTER. AlphaInfer runs it through
                            // AI_DFLASH2: one stream only, always fast verify. nullptr = MTP is this row's best speed.
                            const char * speed_drafter;
                            // ALPHA_SPEED_MODE extra library env, "K=V K=V ..." (setenv without overwrite: an explicit env
                            // wins), and the stream count it was tuned at (0 = keep the row's).
                            const char * speed_env;
                            int speed_streams;
                            // KV type pinned for this row (ALPHA_AI_KV, no overwrite), or nullptr = auto. Auto picks it from
                            // the VRAM left after MTP, so MTP on/off could land on different types and differ in text.
                            const char * kv_type;
                            // routed experts offloaded to host memory (ALPHA_AI_EXPERT_SLOTS, no overwrite): "auto" = the
                            // library's doorbell offload sized from the VRAM left (AI_OFFLOAD=doorbell); nullptr = resident
                            const char * expert_slots;
                            // library env this row always serves with, "K=V K=V ..." (setenv without overwrite), e.g. the
                            // fit settings of a model that misses the card by a little; nullptr = none
                            const char * env;
                            const char * what; };
static const alpha_ai_validated k_ai_validated[] = {
    // speed mode: fast MTP k64 + n-gram lookup K64, 1 stream at 32768 cells (1 x 98304 with K64 verify widths runs out
    // of memory on 24 GB once the workspace is calibrated)
    { "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b", 22663387424LL, 8, 6, 2, 98304, "exact", 64, nullptr,
      "AI_LOOKUP=1 AI_LOOKUP_ARM=32 AI_LOOKUP_NMIN=4 AI_MTP_CHAIN=3 AI_GRAPH_MAX=32 ALPHA_AI_CTX=32768", 1,
      // pin q4_0 so MTP on and off give identical text (auto would choose q4_0 with MTP, f16 without)
      "q4_0", nullptr, nullptr, "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M (RTX 3090)" },
    // Exact MTP (token-identical to plain greedy). Identity at depth needs libalphainfer 583c749+ on cc < 12: older
    // libraries sized the q4_0/q8_0 decode-attention split (AI_FA_Q) from the round's KV envelope, so exact and plain
    // split differently where the envelope grows; newer ones size it from the stream's KV region (constant per engine).
    // 2 x 131072 fits a 24 GB card with q4_0 KV. Speed drafter: DFlash2 q5_K.
    { "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482", 16464440224LL, 8, 6, 2, 262144, "exact", 3,
      "Qwen3.8-27B-DFlash2/dflash2-q5_K.gguf",
      // speed mode: the DFlash2 drafter as a router plus n-gram lookup K128. The exact default is AI_WIDE_EXACT on a
      // batch-invariant library (AI_BI_FORK=ON), which the library enables by itself for dense exact rows.
      "AI_DF2_ROUTE=1 AI_DF2_HOST=1 AI_LOOKUP=1 AI_LOOKUP_ARM=8 AI_LOOKUP_NMIN=4 AI_LOOKUP_K=128 AI_GRAPH_MAX=32", 1,
      nullptr, nullptr, nullptr, "Qwen3.8-27B-UD-Q4_K_M (RTX 3090)" },
    // RTX 5090 (sm_120, libalphainfer built for sm_120a).
    { "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b", 22663387424LL, 12, 0, 2, 98304, "exact", 3, nullptr,
      nullptr, 0, nullptr, nullptr, nullptr, "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M (RTX 5090)" },
    // 27B on sm_120 is not added: exact == plain through the server fails there, likely the plain path's unpinned FA
    // split on sm_12x (dense models get no plain pins). Re-gate before adding.
    // RTX 4080 (sm_89, 16 GB): the models do not fit; both rows need libalphainfer built with AI_DOORBELL=ON.
    // 35B: routed experts in host memory, the doorbell offload sized from the VRAM left (expert_slots "auto"). Fast:
    // the CPU computes the missed experts (MTP is fast verify under offload).
    // MTP drafts 3 deep plus prompt-lookup drafts up to 32 wide (shadow-armed at the MoE default of 32 matched tokens);
    // a lookup verify round is 33 rows, so the doorbell's captured-round width AI_DOOR_TMAX must cover it.
    { "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b", 22663387424LL, 8, 9, 1, 32768, nullptr, 3, nullptr,
      nullptr, 0, "q4_0", "auto", "ALPHA_AI_MTP_K=32 AI_MTP_CHAIN=3 AI_LOOKUP=1 AI_LOOKUP_NMIN=4 AI_DOOR_TMAX=33",
      "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M (RTX 4080, doorbell offload, FAST)" },
    // 27B: original weights; 4 dense FFN layers in host memory computed by the CPU (AI_FFN_CPU), token_embd in mapped
    // host memory, the draft head on the shared output rows, 128-row prefill chunks and a 640 MiB workspace. A
    // load-time requant would be faster but fails the per-language NLL gate, so the weights stay as shipped. Fast MTP
    // drafting 5 deep plus prompt-lookup drafts up to 16 wide (AI_LOOKUP, shadow-armed at 8 matched tokens, so
    // non-copy text runs exactly as MTP-5). Lookup at 32 wide is no better than 16.
    { "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482", 16464440224LL, 8, 9, 1, 8192, "fast", 16, nullptr,
      nullptr, 0, "q4_0", nullptr,
      "AI_HOST_EMBD=1 AI_FFN_CPU=4 AI_MTP_DRAFT_Q=off AI_KV_MARGIN_MB=128 AI_CHUNK=128 ALPHA_AI_WS_MIB=640 ALPHA_AI_MTP_K=16 "
      "AI_MTP_CHAIN=5 AI_LOOKUP=1 AI_LOOKUP_NMIN=4",
      "Qwen3.8-27B-UD-Q4_K_M (RTX 4080, 4 FFN layers on the CPU, FAST)" },
};

// The row's speed drafter "<dir>/<file>": ALPHA_AI_DRAFTER, else <file> beside the model, else exactly <parent>/<dir>/<file>
// (/data/models/Qwen3.8-27B-GGUF/<model> -> /data/models/Qwen3.8-27B-DFlash2/dflash2-q5_K.gguf). No directory scan: a
// generic drafter file name in another model's directory must never be picked. "" = none.
static std::string alpha_ai_find_drafter(const char * model_path, const char * rel) {
    if (const char * e = getenv("ALPHA_AI_DRAFTER")) return e;
    if (!rel || !model_path) return "";
    struct stat st;
    const std::string mp(model_path), r(rel);
    const size_t s1 = mp.rfind('/');
    const std::string dir = s1 == std::string::npos ? "." : mp.substr(0, s1);
    const size_t rs = r.rfind('/');
    const std::string file = rs == std::string::npos ? r : r.substr(rs + 1);
    if (stat((dir + "/" + file).c_str(), &st) == 0 && S_ISREG(st.st_mode)) return dir + "/" + file;
    if (rs != std::string::npos) {
        const size_t s2 = dir.rfind('/');
        const std::string cand = (s2 == std::string::npos ? std::string(".") : dir.substr(0, s2)) + "/" + r;
        if (stat(cand.c_str(), &st) == 0 && S_ISREG(st.st_mode)) return cand;
    }
    return "";
}

namespace {
struct sha256_ctx { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; };
static const uint32_t k_sha256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void sha256_block(sha256_ctx & c, const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i+1] << 16 | (uint32_t) p[4*i+2] << 8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
        const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = c.h[0], b = c.h[1], cc = c.h[2], d = c.h[3], e = c.h[4], f = c.h[5], g = c.h[6], h = c.h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k_sha256[i] + w[i];
        const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c.h[0] += a; c.h[1] += b; c.h[2] += cc; c.h[3] += d; c.h[4] += e; c.h[5] += f; c.h[6] += g; c.h[7] += h;
}
static std::string sha256_file(const char * path) {
    sha256_ctx c = { { 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 }, 0, {}, 0 };
    FILE * f = fopen(path, "rb");
    if (!f) return "";
    std::vector<uint8_t> buf(1 << 22);
    size_t r;
    while ((r = fread(buf.data(), 1, buf.size(), f)) > 0) {
        c.len += r;
        size_t i = 0;
        if (c.n) { while (c.n < 64 && i < r) c.buf[c.n++] = buf[i++]; if (c.n == 64) { sha256_block(c, c.buf); c.n = 0; } }
        for (; i + 64 <= r; i += 64) sha256_block(c, buf.data() + i);
        while (i < r) c.buf[c.n++] = buf[i++];
    }
    fclose(f);
    const uint64_t bits = c.len * 8;
    c.buf[c.n++] = 0x80;
    if (c.n > 56) { while (c.n < 64) c.buf[c.n++] = 0; sha256_block(c, c.buf); c.n = 0; }
    while (c.n < 56) c.buf[c.n++] = 0;
    for (int i = 7; i >= 0; i--) c.buf[c.n++] = (uint8_t) (bits >> (8 * i));
    sha256_block(c, c.buf);
    char hex[65];
    for (int i = 0; i < 8; i++) snprintf(hex + 8 * i, 9, "%08x", c.h[i]);
    return std::string(hex, 64);
}
} // namespace

// SHA-256 of a model file, cached in ~/.cache/alphallama/sha256.tsv keyed on path, size and mtime.
static std::string alpha_model_sha256_cached(const char * path) {
    struct stat st;
    if (stat(path, &st) != 0) return "";
    const char * home = getenv("HOME");
    const std::string dir = std::string(home ? home : "/tmp") + "/.cache/alphallama";
    const std::string cache = dir + "/sha256.tsv";
    const std::string key = std::string(path) + "\t" + std::to_string((long long) st.st_size) + "\t" + std::to_string((long long) st.st_mtime);
    {
        std::ifstream in(cache);
        std::string line;
        while (std::getline(in, line)) {
            if (line.size() > key.size() + 1 && line.compare(0, key.size(), key) == 0 && line[key.size()] == '\t') return line.substr(key.size() + 1);
        }
    }
    fprintf(stderr, "[auto] hashing %s once to check it against the AlphaInfer-validated file (cached afterwards)...\n", path);
    const std::string h = sha256_file(path);
    if (!h.empty()) {
        mkdir((std::string(home ? home : "/tmp") + "/.cache").c_str(), 0755);
        mkdir(dir.c_str(), 0755);
        std::ofstream out(cache, std::ios::app);
        out << key << "\t" << h << "\n";
    }
    return h;
}

static const alpha_ai_validated * alpha_auto_alphainfer_row(const char * model_path) {
    const char * ae = getenv("ALPHA_AUTO");
    if ((ae && atoi(ae) == 0) || !model_path) return nullptr;
#ifndef GGML_USE_CUDA
    return nullptr;
#else
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    struct stat st;
    if (stat(model_path, &st) != 0) return nullptr;
    for (const auto & r : k_ai_validated) {
        if (r.sm_major != prop.major || r.sm_minor != prop.minor || r.size != (long long) st.st_size) continue;   // cheap filters
        if (alpha_model_sha256_cached(model_path) == r.sha256) return &r;
    }
    return nullptr;
#endif
}
#endif // ALPHA_HAS_ALPHAINFER

int main(int argc, char ** argv) {
    // Engine identity banner: alphallama and alphallama-serve are different engines with different speed and
    // endpoints; announce which one is running.
    fprintf(stderr,
        "==================================================================\n"
        " AlphaServe  ·  high-throughput serving engine  (tools/serve)\n"
        "   Fused-infer / ALL_GPU MoE decode — THIS is the fast serving path.\n"
        "   (decode speed is the '[alphaserve] N tokens ... tok/s' log line)\n"
        "   Also serves the corpus /api/corpus/query retrieval endpoint.\n"
        "   'alphallama' is the retrieval/dev twin (LUT_DIRECT MoE).\n"
        "==================================================================\n");
    fprintf(stderr, "%s (%s), build %s\n", alphallama_version_name(), alphallama_version(), llama_build_info());
#if !defined(_WIN32)
    if (getenv("ALPHA_SEGV_TRACE")) signal(SIGSEGV, alpha_segv_handler);
#endif
    // A write to a socket the client already closed raises SIGPIPE, whose default action silently
    // terminates the process. The vendored cpp-httplib does not use MSG_NOSIGNAL, and async clients
    // tear connections down aggressively, so ignore it; writes then fail with EPIPE and are handled
    // per request.
#if !defined(_WIN32)
    signal(SIGPIPE, SIG_IGN);   // Windows has no SIGPIPE; a closed socket surfaces as a send() error
#endif
    // Parse args
    const char * model_path = nullptr;
    const char * host = "0.0.0.0";
    const char * backend_str = "local";
    const char * backend_url = nullptr;
    const char * backend_key = nullptr;
    const char * corpus_dir = nullptr;
    const char * redis_kv = nullptr;  // --redis-kv HOST:PORT for KV cache persistence
    int port = 8080;
    int n_gpu_layers = 99;
    bool ngl_explicit = false;   // true once -ngl is parsed; gates the dense auto-fit below
    int n_slots = 4;
    int n_ctx = 2048;
    bool np_explicit = false, ctx_explicit = false;   // ALPHA_AUTO: the AlphaInfer pick applies its validated shape only when unset
    int n_ubatch = 512;       // --ubatch: prefill chunk + context ubatch
    bool ubatch_explicit = false; // true once --ubatch is parsed; gates the auto-fit ubatch shrink
    int ubatch_prefill = 0;   // --ubatch-prefill: runtime-grow target for long prompts
    const char * override_tensor = nullptr;  // --override-tensor 'pattern=CPU'
    { const char * rb = getenv("ALPHA_REASONING_BUDGET"); if (rb && *rb) g_reasoning_budget = atoi(rb); }   // --reasoning-budget overrides
    const char * mmproj_path = getenv("ALPHA_MMPROJ");   // --mmproj FILE: image input through this projector (built-in engine)
#ifdef ALPHA_DELTANET
    const char * deltanet_state = nullptr;      // --deltanet-state PATH (load a prebuilt recurrent state)
#endif
    // -kvu/--kv-unified: one KV buffer shared across sequences instead of a per-sequence stream.
    // Required by anything that puts two sequences in one batch over a shared prefix (tree verify,
    // OR-verify, seq-fork drafting); llama-batch rejects the sequential split with "coupled
    // sequences in the input batch".
    bool kv_unified = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) host = argv[++i];
        else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) mmproj_path = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) { n_gpu_layers = atoi(argv[++i]); ngl_explicit = true; }
        else if ((strcmp(argv[i], "-np") == 0 || strcmp(argv[i], "--parallel") == 0) && i + 1 < argc) { n_slots = atoi(argv[++i]); np_explicit = true; }
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) { n_ctx = atoi(argv[++i]); ctx_explicit = true; }
        else if (strcmp(argv[i], "-kvu") == 0 || strcmp(argv[i], "--kv-unified") == 0) kv_unified = true;
        else if (strcmp(argv[i], "-no-kvu") == 0 || strcmp(argv[i], "--no-kv-unified") == 0) kv_unified = false;
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) backend_str = argv[++i];
        else if (strcmp(argv[i], "--backend-url") == 0 && i + 1 < argc) backend_url = argv[++i];
        else if (strcmp(argv[i], "--backend-key") == 0 && i + 1 < argc) backend_key = argv[++i];
        else if (strcmp(argv[i], "--corpus-dir") == 0 && i + 1 < argc) corpus_dir = argv[++i];
        else if (strcmp(argv[i], "--redis-kv") == 0 && i + 1 < argc) redis_kv = argv[++i];
        else if ((strcmp(argv[i], "--override-tensor") == 0 || strcmp(argv[i], "-ot") == 0) && i + 1 < argc)
            override_tensor = argv[++i];
#ifdef ALPHA_DELTANET
        else if (strcmp(argv[i], "--deltanet-state") == 0 && i + 1 < argc) deltanet_state = argv[++i];
#endif
        else if ((strcmp(argv[i], "--ubatch") == 0 || strcmp(argv[i], "-ub") == 0) && i + 1 < argc) {
            // -ub is the llama.cpp spelling; --ubatch is also accepted. Note this loop has no
            // unknown-flag branch, so a misspelled flag is silently ignored.
            // Larger ubatch = larger prefill chunks (fewer llama_decode calls for long prompts).
            // Decode is unaffected (1-8 tokens fit any ubatch). Cost: compute-buffer VRAM scales
            // with n_ubatch; on VRAM-tight MoE offload configs this displaces expert-pool budget.
            n_ubatch = atoi(argv[++i]);
            if (n_ubatch < 128)  n_ubatch = 128;
            if (n_ubatch > 8192) n_ubatch = 8192;
            ubatch_explicit = true;
        }
        else if (strcmp(argv[i], "--reasoning-budget") == 0 && i + 1 < argc) {
            g_reasoning_budget = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--reasoning-budget-message") == 0 && i + 1 < argc) {
            g_reasoning_budget_msg = argv[++i];
        }
        else if (strcmp(argv[i], "--ubatch-prefill") == 0 && i + 1 < argc) {
            // Runtime grow: the context starts at the small ubatch; prompts >1024 tokens get
            // N-sized prefill chunks and the GPU engine grows the compute buffers on demand
            // (llama_set_n_ubatch, rollback-safe). The big buffer is only allocated if a long
            // prompt actually arrives.
            ubatch_prefill = atoi(argv[++i]);
            if (ubatch_prefill < 0)    ubatch_prefill = 0;
            if (ubatch_prefill > 8192) ubatch_prefill = 8192;
        }
        // --marlin-dir is parsed separately (see marlin_dir variable below)
    }

    enum BackendMode { MODE_LOCAL, MODE_SGLANG, MODE_VLLM, MODE_AUTO, MODE_FLASHINFER, MODE_ALPHAINFER };
    BackendMode mode = MODE_LOCAL;
    if (strcmp(backend_str, "local") == 0 || strcmp(backend_str, "ggml") == 0) mode = MODE_LOCAL;
    else if (strcmp(backend_str, "sglang") == 0) mode = MODE_SGLANG;
    else if (strcmp(backend_str, "vllm") == 0) mode = MODE_VLLM;
    else if (strcmp(backend_str, "auto") == 0) mode = MODE_AUTO;
    else if (strcmp(backend_str, "flashinfer") == 0 || strcmp(backend_str, "fi") == 0) mode = MODE_FLASHINFER;
    else if (strcmp(backend_str, "alphainfer") == 0 || strcmp(backend_str, "ai") == 0) {
#ifdef ALPHA_HAS_ALPHAINFER
        mode = MODE_ALPHAINFER;
#else
        fprintf(stderr, "[serve] --backend alphainfer: this binary was built without AlphaInfer "
                        "(configure with -DALPHA_ALPHAINFER=ON -DALPHAINFER_ROOT=<bundle>)\n");
        return 1;
#endif
    }

    // SGLang and vLLM are both remote OpenAI-compatible HTTP proxies with identical wiring.
    auto is_remote_proxy = [](BackendMode m) { return m == MODE_SGLANG || m == MODE_VLLM; };

    int auto_threshold = 2;  // auto mode: use local for <= N active, sglang for > N

    // Parse --auto-threshold if present
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--auto-threshold") == 0 && i + 1 < argc) {
            auto_threshold = atoi(argv[++i]);
        }
    }

    // Backend auto-routing per model type and slot count:
    //   MoE (n_expert>0)          -> local fused-infer/ALL_GPU (batches well; FlashInfer MoE is slower)
    //   dense FP16/BF16, >1 slot  -> FlashInfer (batched decode)
    //   dense quantized           -> local (the FlashInfer Q4_K dequant path is not supported)
    //   dense, single slot        -> local (fastest single-stream decode)
    // Applies to the default backend and to `--backend auto` (unless a remote --backend-url is set,
    // which keeps the sglang/vllm proxy path). `--backend fi` and `--backend local` still force
    // their path.
    bool explicit_backend = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backend") == 0) { explicit_backend = true; break; }
    }
    bool have_remote_url = (backend_url && backend_url[0]);

    // Arch-first format routing: AWQ checkpoint dirs go to an external engine (SGLang first,
    // ALPHA_ROUTE_AWQ_BACKEND=vllm flips). Architecture dominates format: external engines reject
    // some target archs, and their offload is naive per-forward weight streaming, so an AWQ model
    // that does not fit is never routed out (GGUF + fused-infer is the offload path).
    //   GGUF (any arch)                        -> local (the router below)
    //   AWQ dir + loadable arch + fits + URL   -> SGLang (or vLLM) proxy
    //   AWQ dir + blocked arch                 -> error: serve the GGUF equivalent
    //   AWQ dir + does not fit VRAM            -> error: GGUF + fused-infer instead
    if ((!explicit_backend || mode == MODE_AUTO) && model_path) {
        struct stat mst;
        if (stat(model_path, &mst) == 0 && S_ISDIR(mst.st_mode)) {
            json awq_cfg(json::value_t::discarded);
            {
                std::ifstream cf(std::string(model_path) + "/config.json");
                if (cf.good()) awq_cfg = json::parse(cf, nullptr, false);
            }
            std::string quant, hf_arch;
            if (!awq_cfg.is_discarded() && awq_cfg.is_object()) {
                hf_arch = awq_cfg.value("model_type", "");
                if (awq_cfg.contains("quantization_config") && awq_cfg["quantization_config"].is_object()) {
                    quant = awq_cfg["quantization_config"].value("quant_method", "");
                }
            }
            if (quant == "awq") {
                // arch guard: external engines reject these families; env-extendable
                const char * be = getenv("ALPHA_ROUTE_BLOCKED_ARCHS");
                std::string blocked = be ? be : "qwen3_5,qwen35,qwen3_next,gemma4,deepseek_v4,deepseek4";
                bool arch_blocked = false;
                {
                    std::stringstream ss(blocked); std::string tok;
                    while (std::getline(ss, tok, ',')) {
                        if (!tok.empty() && hf_arch.find(tok) != std::string::npos) { arch_blocked = true; break; }
                    }
                }
                if (arch_blocked) {
                    fprintf(stderr, "[serve] arch-first routing: AWQ arch '%s' is externally UNSUPPORTED "
                            "(blocklist: %s) and the native Marlin path is pending the bf16 fix. "
                            "Serve the GGUF equivalent instead (-m model.gguf).\n",
                            hf_arch.c_str(), blocked.c_str());
                    return 1;
                }
                // fit guard: weights + ~2GB KV/workspace vs free VRAM
                size_t weight_bytes = 0;
#if defined(_WIN32)
                {   // std::filesystem: portable replacement for the opendir/readdir scan
                    std::error_code ec;
                    for (const auto & de : std::filesystem::directory_iterator(model_path, ec)) {
                        const std::string n = de.path().filename().string();
                        if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) {
                            std::error_code fec;
                            const auto sz = std::filesystem::file_size(de.path(), fec);
                            if (!fec) weight_bytes += (size_t) sz;
                        }
                    }
                }
#else
                if (DIR * d = opendir(model_path)) {
                    while (struct dirent * e = readdir(d)) {
                        const char * n = e->d_name; size_t l = strlen(n);
                        if (l > 12 && strcmp(n + l - 12, ".safetensors") == 0) {
                            struct stat fs;
                            if (stat((std::string(model_path) + "/" + n).c_str(), &fs) == 0) {
                                weight_bytes += (size_t) fs.st_size;
                            }
                        }
                    }
                    closedir(d);
                }
#endif
                size_t free_vram = 0, total_vram = 0;
                alpha_vram_info(&free_vram, &total_vram);
                { const char * fv = getenv("ALPHA_FORCE_VRAM_MB"); if (fv) { long v = atol(fv);
                    if (v > 0) free_vram = (size_t) v * 1024 * 1024; } }   // dry-run knob for other card sizes
                const bool fits = weight_bytes > 0 && weight_bytes + (2ull << 30) <= free_vram;
                if (!fits) {
                    fprintf(stderr, "[serve] arch-first routing: AWQ '%s' (%.1f GB weights) does not fit "
                            "%.1f GB free VRAM. External engines only do naive per-forward weight "
                            "streaming (~2 t/s) — do NOT route out; serve the GGUF equivalent on the "
                            "local engine (fused-infer is the offload-capable path).\n",
                            hf_arch.c_str(), weight_bytes / 1073741824.0, free_vram / 1073741824.0);
                    return 1;
                }
                // target selection: SGLang preferred (vLLM's AWQ prefill on Ampere has high TTFT); env flips
                const char * pref    = getenv("ALPHA_ROUTE_AWQ_BACKEND");
                const bool prefer_vllm = pref && strcmp(pref, "vllm") == 0;
                const char * sgl_url = getenv("ALPHA_SGLANG_URL");
                const char * vll_url = getenv("ALPHA_VLLM_URL");
                const char * url = nullptr; BackendMode m2 = MODE_SGLANG;
                if (have_remote_url)                            { url = backend_url; m2 = prefer_vllm ? MODE_VLLM : MODE_SGLANG; }
                else if (!prefer_vllm && sgl_url && sgl_url[0]) { url = sgl_url;     m2 = MODE_SGLANG; }
                else if (vll_url && vll_url[0])                 { url = vll_url;     m2 = MODE_VLLM; }
                else if (sgl_url && sgl_url[0])                 { url = sgl_url;     m2 = MODE_SGLANG; }
                if (!url) {
                    fprintf(stderr, "[serve] arch-first routing: AWQ '%s' fits and is externally loadable, "
                            "but no engine URL is configured. Set ALPHA_SGLANG_URL (preferred) or "
                            "ALPHA_VLLM_URL, or pass --backend-url.\n", hf_arch.c_str());
                    return 1;
                }
                mode = m2; backend_url = url; have_remote_url = true;
                fprintf(stderr, "[serve] arch-first routing: AWQ/%s, %.1f GB weights, fits %.1f GB free "
                        "→ %s proxy %s\n", hf_arch.c_str(), weight_bytes / 1073741824.0,
                        free_vram / 1073741824.0, m2 == MODE_VLLM ? "vllm" : "sglang", url);
            }
        }
    }

#ifdef ALPHA_HAS_ALPHAINFER
    // An explicit `--backend alphainfer` still takes the validated row's mode / speed mode / shape; otherwise an
    // "exact" row would get the library's dense default (fast). Any other explicit backend skips the row.
    if (const alpha_ai_validated * air = (explicit_backend && mode != MODE_ALPHAINFER) ? nullptr : alpha_auto_alphainfer_row(model_path)) {
        mode = MODE_ALPHAINFER;
        if (!np_explicit)  n_slots = air->n_streams;   // the validated shape for this model
        if (!ctx_explicit) n_ctx   = air->ctx_total;
        const char * sm = getenv("ALPHA_SPEED_MODE");
        if (sm && atoi(sm) != 0) {
            setenv("AI_MTP_MODE", "fast", /*overwrite=*/ 0);
            if (air->speed_env) {   // the row's speed config; an explicit env wins
                std::istringstream kv(air->speed_env);
                std::string item;
                while (kv >> item) {
                    const size_t eq = item.find('=');
                    if (eq == std::string::npos) continue;
                    if (ctx_explicit && item.compare(0, eq, "ALPHA_AI_CTX") == 0) continue;   // an explicit -c wins over the row's speed context
                    setenv(item.substr(0, eq).c_str(), item.substr(eq + 1).c_str(), /*overwrite=*/ 0);
                }
                fprintf(stderr, "[auto] ALPHA_SPEED_MODE: row speed env %s\n", air->speed_env);
            }
            if (air->speed_streams > 0 && !np_explicit) {
                if (!ctx_explicit) n_ctx = air->ctx_total / std::max(1, air->n_streams) * air->speed_streams;
                n_slots = air->speed_streams;
            }
            const std::string drafter = getenv("AI_DFLASH2") ? std::string(getenv("AI_DFLASH2")) : alpha_ai_find_drafter(model_path, air->speed_drafter);
            if (!drafter.empty() && (!np_explicit || n_slots == 1)) {
                // DFlash2 holds one drafter context: one stream, at the per-stream context of the validated shape
                setenv("AI_DFLASH2", drafter.c_str(), /*overwrite=*/ 0);
                n_slots = 1;
                if (!ctx_explicit) n_ctx = air->ctx_total / std::max(1, air->n_streams);
                fprintf(stderr, "[auto] ALPHA_SPEED_MODE: best-speed DFlash2 drafter %s, 1 stream -- output is NOT token-identical to plain greedy\n",
                        drafter.c_str());
            } else {
                if (!drafter.empty()) fprintf(stderr, "[auto] ALPHA_SPEED_MODE: DFlash2 needs 1 stream (-np %d given) -- using fast MTP\n", n_slots);
                unsetenv("AI_DFLASH2");   // an inherited one would make the library refuse create at n_streams != 1
                if (air->fast_k > 0) setenv("ALPHA_AI_MTP_K", std::to_string(air->fast_k).c_str(), /*overwrite=*/ 0);
                fprintf(stderr, "[auto] ALPHA_SPEED_MODE: best-speed MTP (%s, k=%s) -- output is NOT token-identical to plain greedy\n",
                        getenv("AI_MTP_MODE"), getenv("ALPHA_AI_MTP_K") ? getenv("ALPHA_AI_MTP_K") : "default");
            }
        }
        else if (getenv("AI_DFLASH2"))   // the library reads it directly: an operator's (or inherited) one overrides exact
            fprintf(stderr, "[auto] WARNING: AI_DFLASH2 is set -- DFlash2 FAST verify, NOT this row's %s mode (output not "
                            "token-identical); /health reports it\n", air->mtp_mode ? air->mtp_mode : "default");
        if (air->mtp_mode) setenv("AI_MTP_MODE", air->mtp_mode, /*overwrite=*/ 0);   // the validated mode; an explicit env wins
        if (air->kv_type) setenv("ALPHA_AI_KV", air->kv_type, /*overwrite=*/ 0);     // the row's pinned KV type
        if (air->expert_slots) {   // the row serves with routed experts in host memory (doorbell offload)
            setenv("ALPHA_AI_EXPERT_SLOTS", air->expert_slots, /*overwrite=*/ 0);
            setenv("AI_OFFLOAD", "doorbell", /*overwrite=*/ 0);
        }
        if (air->env) {   // the row's always-on library env; an explicit env wins
            std::istringstream kv(air->env);
            std::string item;
            while (kv >> item) {
                const size_t eq = item.find('=');
                if (eq != std::string::npos) setenv(item.substr(0, eq).c_str(), item.substr(eq + 1).c_str(), /*overwrite=*/ 0);
            }
            fprintf(stderr, "[auto] row env %s\n", air->env);
        }
        fprintf(stderr, "[auto] validated for AlphaInfer: %s\n", air->what);
        fprintf(stderr, "[auto] AlphaInfer-validated model on this GPU -> AlphaInfer backend, %d streams x %d cells%s (--backend local or ALPHA_AUTO=0 keeps the built-in engine)\n",
                n_slots, n_ctx / (n_slots > 0 ? n_slots : 1), getenv("AI_DFLASH2") ? ", DFlash2 drafter" : "");
    }
#endif
    // `--backend auto` with no remote URL resolves to the smart local/FI selection below.
    bool want_auto_route = (!explicit_backend) || (mode == MODE_AUTO && !have_remote_url);
    if (mode == MODE_AUTO && !have_remote_url) mode = MODE_LOCAL;

    bool auto_routed = false;
    if (want_auto_route && mode == MODE_LOCAL && model_path && !override_tensor) {
        const char * mp = model_path;
        // Filename substring match, kept only as a fallback for GGUFs with no general.file_type.
        // It cannot see a quant whose name does not contain "qN": NVFP4 and MXFP4 read as
        // unquantised (they contain "P4", not "Q4").
        bool is_quant = (strstr(mp, "q4") || strstr(mp, "Q4") || strstr(mp, "q5") || strstr(mp, "Q5") ||
                         strstr(mp, "q6") || strstr(mp, "Q6") || strstr(mp, "q8") || strstr(mp, "Q8") ||
                         strstr(mp, "q2") || strstr(mp, "Q2") || strstr(mp, "q3") || strstr(mp, "Q3"));
        // Peek n_expert from the GGUF header so MoE never routes to FI (local batches better for MoE).
        // Same idiom as the auto-fit / auto-fused-infer peeks below.
        int n_expert_peek = 0;
        {
            struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
            struct gguf_context * gc = gguf_init_from_file(mp, gp);
            if (gc) {
                int64_t ak = gguf_find_key(gc, "general.architecture");
                if (ak >= 0) {
                    std::string arch = gguf_get_val_str(gc, ak);
                    int64_t ek = gguf_find_key(gc, (arch + ".expert_count").c_str());
                    if (ek >= 0) n_expert_peek = (int) gguf_get_val_u32(gc, ek);
                }
                gguf_free(gc);
            }
        }
        bool is_moe = (n_expert_peek > 0);
        s_n_expert_peek = n_expert_peek;   // file-scope copy for the MoE ubatch default below
        // Authoritative quantisation check: ask the GGUF, not the filename. Anything that is not
        // F32/F16/BF16 is quantised and stays on the local backend. Falls back to the filename
        // heuristic above when the key is absent.
        {
            struct gguf_init_params gp2 = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
            struct gguf_context * gc2 = gguf_init_from_file(mp, gp2);
            if (gc2) {
                int64_t fk = gguf_find_key(gc2, "general.file_type");
                if (fk >= 0) {
                    const uint32_t ft = gguf_get_val_u32(gc2, fk);
                    is_quant = !(ft == (uint32_t) LLAMA_FTYPE_ALL_F32 ||
                                 ft == (uint32_t) LLAMA_FTYPE_MOSTLY_F16 ||
                                 ft == (uint32_t) LLAMA_FTYPE_MOSTLY_BF16);
                }
                gguf_free(gc2);
            }
        }
        // FlashInfer FP16/BF16 batched decode for dense models; MoE and quantized dense stay on
        // local. --backend fi forces FI regardless.
        if (is_moe) {
            fprintf(stderr, "[serve] auto: MoE (n_expert=%d) → local fused-infer/ALL_GPU (batches; FI-MoE slower)\n", n_expert_peek);
        } else if (!is_quant && n_slots > 1) {
            mode = MODE_FLASHINFER;
            auto_routed = true;
            fprintf(stderr, "[serve] auto: dense FP16/BF16, %d slots → FlashInfer (batched decode)\n", n_slots);
        } else {
            fprintf(stderr, "[serve] auto: %s → local (fastest here; FI Q4_K dequant path n/a). --backend fi to force FI.\n",
                    is_quant ? "quantized dense" : "single-slot dense");
        }
    }

    bool need_local = (mode == MODE_LOCAL || mode == MODE_AUTO);
    bool need_remote = (is_remote_proxy(mode) || mode == MODE_AUTO);
    bool need_flashinfer = (mode == MODE_FLASHINFER);
    bool need_model = (need_local || need_flashinfer || mode == MODE_ALPHAINFER);

    if (need_model && !model_path) {
        fprintf(stderr,
            "Usage:\n"
            "  Single user:  alphallama-serve -m model.gguf\n"
            "  Multi user:   alphallama-serve -m model.gguf -np 8\n"
            "  Marlin INT4:  alphallama-serve -m fp16.gguf --marlin-dir /path -np 8\n"
            "\n"
            "Backends (auto-selected by model format + slot count):\n"
            "  Any GGUF + 1 slot        → ggml native (fastest single-user)\n"
            "  FP16/BF16 GGUF + 2+ slots → FlashInfer (batched paged attention; NEOX rope;\n"
            "                             CUDA graphs gated to single-seq decode)\n"
            "  Quantized GGUF + 2+ slots → local (ggml) — FI Q4_K dequant not yet fixed\n"
            "  AWQ INT4 (--marlin-dir)  → Marlin W4A16 (Tensor Core INT4)\n"
            "\n"
            "  Override: --backend ggml|fi|sglang|vllm|auto   (--backend ggml forces local)\n"
            "\n"
            "Flags:\n"
            "  -m MODEL              Model file (GGUF)\n"
            "  -np N                 Number of slots/users (default: 4)\n"
            "  --backend TYPE        Override: ggml, fi, sglang, vllm, auto\n"
            "  --marlin-dir DIR      AWQ INT4 weights (Marlin format)\n"
            "  --backend-url URL     Remote server URL (for sglang/vllm/auto)\n"
            "\n"
            "Arch-first AWQ routing (-m <hf-awq-dir> with default/auto backend):\n"
            "  loadable arch + fits VRAM  → SGLang proxy (set ALPHA_SGLANG_URL / ALPHA_VLLM_URL;\n"
            "                               ALPHA_ROUTE_AWQ_BACKEND=vllm to prefer vLLM)\n"
            "  blocked arch or no-fit     → refuses with guidance (use the GGUF on local engine;\n"
            "                               blocklist via ALPHA_ROUTE_BLOCKED_ARCHS)\n"
            "  --backend-key KEY     Optional API key for remote backend\n"
            "  --auto-threshold N    Auto mode: local for <=N active slots (default: 2)\n"
            "  --corpus-dir DIR      Enable intelligence layer (BM25 + NER retrieval)\n"
            "  --redis-kv HOST:PORT  Enable Redis KV cache (cross-session prefix reuse)\n"
            "  --port PORT           HTTP port (default: 8080)\n"
            "  --host HOST           Bind address (default: 0.0.0.0)\n"
            "  -np N                 Number of local slots (default: 4)\n"
            "  -ngl N                GPU layers (default: 99)\n"
            "  -c N                  Context per slot (default: 2048)\n"
        );
        return 1;
    }

    if (need_remote && !backend_url) {
        fprintf(stderr, "[serve] Error: --backend sglang/vllm/auto requires --backend-url\n");
        return 1;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Initialize intelligence layer
    alphaserve::intelligence_ctx intel_ctx;
    intel_ctx.ctx_size = n_ctx;

    if (corpus_dir) {
        fprintf(stderr, "[serve] Initializing intelligence layer from: %s\n", corpus_dir);
        alphaserve::intelligence_init_corpus(intel_ctx, corpus_dir);
    }

    // Initialize backend
    const char * model_name = "alphallama";
    common_chat_templates_ptr chat_tmpls;  // initialized after model load

    // Local backend state (only used if --backend local)
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    std::unique_ptr<alphaserve::server> local_srv;
    std::unique_ptr<alphaserve::response_router> local_router;

    // Remote backend
    std::unique_ptr<alphaserve::backend> remote_backend;

    // Initialize local backend
    if (need_local) {
        fprintf(stderr, "[serve] Backend: local (AlphaServe)\n");
        fprintf(stderr, "[serve] Loading model: %s\n", model_path);
        alpha_auto_profile(model_path, n_slots, np_explicit, ctx_explicit, argv);   // ALPHA_AUTO: before anything reads an ALPHA_* variable
        // ALPHA_AUTO: the profile's serving shape, only where none was given
        if (!np_explicit  && getenv("ALPHA_AUTO_NP"))  n_slots = std::max(1, atoi(getenv("ALPHA_AUTO_NP")));
        if (!ctx_explicit && getenv("ALPHA_AUTO_CTX")) n_ctx   = std::max(256, atoi(getenv("ALPHA_AUTO_CTX")));
        // qwen4exp serves one sequence: its sparse attention (build_attn_qsa) has no multi-stream KV form and a
        // 2-sequence batch would assert. Refuse at start with the reason instead of crashing inside a request.
        if (n_slots > 1) {
            gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
            if (struct gguf_context * gc = gguf_init_from_file(model_path, gp)) {
                const int64_t ak = gguf_find_key(gc, "general.architecture");
                const bool qwen4exp = ak >= 0 && strcmp(gguf_get_val_str(gc, ak), "qwen4exp") == 0;
                gguf_free(gc);
                if (qwen4exp) {
                    fprintf(stderr, "[serve] ERROR: this model (qwen4exp / Flash-Next) serves one sequence: its sparse attention "
                                    "has no multi-stream KV yet. Use -np 1 (got -np %d).\n", n_slots);
                    return 1;
                }
            }
        }
        // tight dense fit: request every layer; the dense auto-fit below still caps it if the weights cannot fit
        if (!ngl_explicit && getenv("ALPHA_AUTO_NGL")) { n_gpu_layers = atoi(getenv("ALPHA_AUTO_NGL")); ngl_explicit = true; }

        // Dense auto-fit ngl (anti-OOM safety net).
        // Without -ngl the default of 99 offloads all layers, which runs out of memory on a card too
        // small for the model. When -ngl is not given and the model is dense and does not fit free
        // VRAM, pack as many layers as fit (partial offload). The per-slot reserve is deliberately
        // conservative (KV + compute + recurrent-state buffers, which grow with -np): the goal is
        // "runs", not "optimal". MoE is skipped (fused-infer / -ot handles its placement); an
        // explicit -ngl bypasses this entirely.
        {
            size_t free_vram = 0, total_vram = 0;
            if (alpha_vram_info(&free_vram, &total_vram) && free_vram > 0) {
                // Same override as the planner (alpha_hw_planner.cpp): dry-run the auto-fit for any
                // card size.
                { const char * fv = getenv("ALPHA_FORCE_VRAM_MB"); if (fv) { long v = atol(fv);
                    if (v > 0) free_vram = (size_t) v * 1024 * 1024; } }
                struct stat st; size_t model_bytes = 0;
                if (stat(model_path, &st) == 0) model_bytes = (size_t)st.st_size;
                int n_layer = 0, n_expert = 0;
                size_t emb_out_fixed_mb = 0, non_rep_mb = 0;  // GPU-fixed lm_head + non-repeating bytes
                // attention geometry for the per-layer KV term (all optional; 0/empty = absent)
                uint32_t kvh_scalar = 0, key_len = 0, val_len = 0, key_len_swa = 0, val_len_swa = 0;
                uint32_t swa_window = 0, full_attn_interval = 0;
                std::vector<uint32_t> kvh_arr;      // per-layer head_count_kv (gemma4-style)
                std::vector<int8_t>  swa_pattern;   // per-layer is-SWA flags (true = sliding-window layer)
                struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
                struct gguf_context * gc = gguf_init_from_file(model_path, gp);
                if (gc) {
                    int64_t ak = gguf_find_key(gc, "general.architecture");
                    if (ak >= 0) {
                        std::string arch = gguf_get_val_str(gc, ak);
                        int64_t bk = gguf_find_key(gc, (arch + ".block_count").c_str());
                        if (bk >= 0) n_layer = (int)gguf_get_val_u32(gc, bk);
                        int64_t ek = gguf_find_key(gc, (arch + ".expert_count").c_str());
                        if (ek >= 0) n_expert = (int)gguf_get_val_u32(gc, ek);
                        // scalar-or-array head_count_kv, key/value lengths (+_swa variants),
                        // sliding window + per-layer pattern, and the hybrid full-attention interval.
                        auto get_u32 = [&](const char * suffix) -> uint32_t {
                            int64_t k = gguf_find_key(gc, (arch + suffix).c_str());
                            if (k < 0) return 0;
                            switch (gguf_get_kv_type(gc, k)) {   // writers emit u32 or i32
                                case GGUF_TYPE_UINT32: return gguf_get_val_u32(gc, k);
                                case GGUF_TYPE_INT32:  { int32_t v = gguf_get_val_i32(gc, k);
                                                         return v > 0 ? (uint32_t)v : 0; }
                                default:               return 0;
                            }
                        };
                        key_len            = get_u32(".attention.key_length");
                        val_len            = get_u32(".attention.value_length");
                        key_len_swa        = get_u32(".attention.key_length_swa");
                        val_len_swa        = get_u32(".attention.value_length_swa");
                        swa_window         = get_u32(".attention.sliding_window");
                        full_attn_interval = get_u32(".full_attention_interval");
                        int64_t hk = gguf_find_key(gc, (arch + ".attention.head_count_kv").c_str());
                        if (hk >= 0) {
                            if (gguf_get_kv_type(gc, hk) == GGUF_TYPE_ARRAY) {
                                const enum gguf_type at = gguf_get_arr_type(gc, hk);
                                if (at == GGUF_TYPE_UINT32 || at == GGUF_TYPE_INT32) {  // gemma4 writes i32
                                    const uint32_t * d = (const uint32_t *) gguf_get_arr_data(gc, hk);
                                    kvh_arr.assign(d, d + gguf_get_arr_n(gc, hk));
                                }
                            } else kvh_scalar = get_u32(".attention.head_count_kv");
                        }
                        int64_t pk = gguf_find_key(gc, (arch + ".attention.sliding_window_pattern").c_str());
                        if (pk >= 0 && gguf_get_kv_type(gc, pk) == GGUF_TYPE_ARRAY &&
                            gguf_get_arr_type(gc, pk) == GGUF_TYPE_BOOL) {
                            const int8_t * d = (const int8_t *) gguf_get_arr_data(gc, pk);
                            swa_pattern.assign(d, d + gguf_get_arr_n(gc, pk));
                        }
                    }
                    // embd/output sizes (mirrors alpha_hw_planner): output/lm_head is GPU-fixed; token_embd
                    // is on the CPU under partial offload but non-repeating. Exclude both from the per-layer
                    // divisor so big-vocab models don't OOM and lighter-per-layer ones auto-pack.
                    int64_t oid = gguf_find_tensor(gc, "output.weight");
                    int64_t eid = gguf_find_tensor(gc, "token_embd.weight");
                    size_t ob = oid >= 0 ? gguf_get_tensor_size(gc, oid) : 0;
                    size_t eb = eid >= 0 ? gguf_get_tensor_size(gc, eid) : 0;
                    emb_out_fixed_mb = (ob > 0 ? ob : eb) / (1024 * 1024);
                    non_rep_mb       = (ob + eb) / (1024 * 1024);
                    gguf_free(gc);
                }
                size_t model_mb = model_bytes / (1024 * 1024);
                size_t free_mb  = free_vram   / (1024 * 1024);
                // dense (n_expert==0), known geometry, and the whole model won't fit
                if (n_expert == 0 && n_layer > 0 && model_mb > 0 && model_mb + 768 > free_mb) {
                    // Shrink the prefill ubatch unless the user set it. The prefill compute buffer
                    // scales with n_ubatch, for some architectures super-linearly (gemma4's
                    // sliding-window attention graph reserves ~2.6 GiB at ubatch 256 but ~22 MiB at
                    // ubatch 64). We only get here when the model does not fit, decode speed does not
                    // depend on ubatch, and a small ubatch keeps the compute buffer within the
                    // n_ubatch-MB reserve proxy below for every arch. The cost is extra prefill
                    // chunks for long prompts; --ubatch overrides.
                    if (!ubatch_explicit && n_ubatch > 64) n_ubatch = 64;
                    // per-repeating-layer cost excludes the non-repeating output+token_embd (see above).
                    size_t per_layer = (model_mb > non_rep_mb ? model_mb - non_rep_mb : model_mb) / (size_t)n_layer;
                    // Average per-layer KV-cache MB from metadata. The KV of a GPU-resident layer lives
                    // in VRAM; without this term auto-fit over-picks layers on models with a large KV
                    // (e.g. gemma4-31B, ~20MB/layer at -c 4096) and fails in the attention alloc.
                    // Hybrids (full_attention_interval > 1) have KV on only every Nth layer; the
                    // recurrent layers' state is per-slot and already inside the 320MB/slot term.
                    double kv_mb_avg = 0.0;
                    if ((kvh_scalar > 0 || !kvh_arr.empty()) && key_len > 0) {
                        if (val_len == 0) val_len = key_len;
                        const size_t tok_full = (size_t)n_ctx * n_slots;   // ctx_params.n_ctx = n_ctx * n_slots
                        const size_t tok_swa_c = ((size_t)swa_window + (size_t)n_ubatch) * n_slots;
                        const size_t tok_swa  = (swa_window > 0 && tok_swa_c < tok_full) ? tok_swa_c : tok_full;
                        double total_mb = 0.0;
                        for (int l = 0; l < n_layer; l++) {
                            if (full_attn_interval > 1 && ((l + 1) % (int)full_attn_interval != 0))
                                continue;   // recurrent layer: no KV
                            const uint32_t kvh = !kvh_arr.empty() ? kvh_arr[l % kvh_arr.size()] : kvh_scalar;
                            const bool swa = !swa_pattern.empty() && swa_pattern[l % swa_pattern.size()] != 0;
                            const uint32_t kl = (swa && key_len_swa) ? key_len_swa : key_len;
                            const uint32_t vl = (swa && val_len_swa) ? val_len_swa : val_len;
                            total_mb += (double)(swa ? tok_swa : tok_full) * kvh * (kl + vl)
                                        * 2 /*f16*/ / (1024.0 * 1024.0);
                        }
                        kv_mb_avg = total_mb / n_layer;
                    }
                    const size_t per_layer_cost = per_layer + (size_t)kv_mb_avg;
                    // reserve = GPU-fixed lm_head + per-slot recurrent-state/compute (grows with -np;
                    // 320MB/slot unified with the HW planner) + prefill compute buffer (~n_ubatch MB).
                    // Per-layer KV is in per_layer_cost above, not here.
                    size_t reserve = emb_out_fixed_mb + (size_t)n_slots * 320 + (size_t)n_ubatch;
                    int fit = (free_mb > reserve && per_layer_cost > 0)
                            ? (int)((free_mb - reserve) / per_layer_cost) : 0;
                    if (fit < 0) fit = 0;
                    if (fit > n_layer) fit = n_layer;
                    // Cap ngl to what fits (only ever reduce). Without -ngl this always applies. With
                    // it (e.g. the common -ngl 99 = "all layers"), cap only if the requested layers'
                    // weights alone exceed free VRAM, so an over-large -ngl degrades to partial
                    // offload instead of a cudaMalloc crash while a tuned -ngl that fits is untouched.
                    int  req       = (n_gpu_layers > n_layer) ? n_layer : n_gpu_layers;
                    bool would_oom = ((size_t) req * per_layer_cost + reserve > free_mb);
                    if ((!ngl_explicit || would_oom) && fit < n_gpu_layers) {
                        fprintf(stderr,
                            "[serve] dense auto-fit: model %zuMB, free %zuMB, %d slots, ubatch %d, "
                            "kv/layer %.0fMB, requested ngl=%d%s -> ngl=%d/%d (reserve %zuMB). "
                            "Pass -ngl / --ubatch to override.\n",
                            model_mb, free_mb, n_slots, n_ubatch, kv_mb_avg, n_gpu_layers,
                            ngl_explicit ? " (would OOM)" : "", fit, n_layer, reserve);
                        n_gpu_layers = fit;
                    }
                }
            }
        }

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = n_gpu_layers;

        // --override-tensor: put specific tensors on CPU (e.g., MoE expert weights)
        // Format: 'pattern=CPU', regex matching (same as alphachat)
        static std::string ot_pattern;
        static llama_model_tensor_buft_override ot_overrides[3] = {};   // up to 2 patterns + terminator
        bool fused_infer_active = false;

        if (override_tensor) {
            // Explicit -ot flag: use as given
            std::string ot(override_tensor);
            auto eq = ot.find('=');
            if (eq != std::string::npos) {
                ot_pattern = ot.substr(0, eq);
                ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
                ot_overrides[0] = {ot_pattern.c_str(), cpu_buft};
                ot_overrides[1] = {nullptr, nullptr};
                model_params.tensor_buft_overrides = ot_overrides;
                fprintf(stderr, "[serve] Override tensor: '%s' → CPU\n", ot_pattern.c_str());
                fused_infer_active = true;
            }
        }
#ifdef GGML_USE_CUDA
        // The GGUF peek runs for both paths. It drives auto-offload when -ot is not given, and
        // either way it drives the fused-infer sub-mode pick below, so an explicit
        // `-ot exps=CPU` also gets the sub-mode suited to its pool size.
        {
            // Pick the fused-infer sub-mode (pool-direct vs CPU-inline split) from how much of
            // the expert working set the VRAM pool can hold. Shared by both paths.
            auto pick_serve_mode = [](uint32_t n_expert, size_t model_size, size_t total_vram) {
                // Auto serve-mode: pool-direct vs CPU-inline split, by miss-heaviness.
                // The fused-infer sub-mode depends on how much of the expert working set fits the
                // VRAM pool (hit rate), which tracks expert_bytes vs usable pool:
                //   hit-rich   (pool holds a good chunk) -> pool-direct is fastest; adding the
                //              CPU pipe only injects fence stall.
                //   miss-heavy (pool small vs experts) -> the CPU pipe (compute misses from RAM in
                //              parallel with GPU DMA) adds ~10%.
                //   starved    (pool is a sliver) -> the GPU pipe is a straggler at any share: RAM
                //              (~50 GB/s) so dominates PCIe (~12 GB/s) that whatever the GPU takes
                //              finishes last and sets the token time.
                // The risk is asymmetric: wrongly enabling CPU-inline on a hit-rich model costs
                // ~35%, wrongly skipping it ~8%. So default to pool-direct and enable CPU-inline
                // only when the pool is clearly starved. ALPHA_CPU_INLINE always wins.
                int backbone_pct = n_expert <= 8 ? 30 : n_expert <= 64 ? 12 :
                                   n_expert <= 128 ? 8 : 4;
                size_t expert_bytes = model_size * (100 - backbone_pct) / 100;
                size_t non_expert   = model_size - expert_bytes;
                // Usable expert pool ~ VRAM - non-expert weights - (KV + compute + context).
                size_t reserve = (size_t)4 * 1024*1024*1024;
                size_t usable_pool = total_vram > non_expert + reserve
                                   ? total_vram - non_expert - reserve : 0;
                double pool_frac = expert_bytes > 0
                                 ? (double)usable_pool / (double)expert_bytes : 1.0;
                const bool op_set_inline = getenv("ALPHA_CPU_INLINE") != nullptr;
                if (pool_frac < 0.20 && !op_set_inline) {
                    // Clearly starved -> CPU-inline split (% of misses kept on the CPU).
                    // Two regimes, because the curve is a step, not a ramp: the GPU pipe's fixed
                    // per-token cost (miss staging + DMA setup + fence) is paid in full as soon as
                    // any miss is routed to it, so it only pays off when its share is large
                    // enough to amortize that. Once the pool is a sliver it never is, hence the
                    // jump to all-CPU.
                    int split;
                    if (pool_frac < 0.15) {
                        split = 100;                                   // starved -> all-CPU
                    } else {
                        // 15-20%: the GPU pipe still earns its keep; ramp 90 -> 50.
                        split = 50 + (int)((0.20 - pool_frac) * 800.0);
                    }
                    if (split < 50)  split = 50;
                    if (split > 100) split = 100;
                    ggml_cuda_expert_pcache_set_cpu_inline(1, split);
                    fprintf(stderr, "[serve] Auto serve-mode: %s (pool≈%.0f%% of experts) "
                            "→ CPU-inline split (ALPHA_INLINE_SPLIT=%d). "
                            "Disable with ALPHA_CPU_INLINE=0.\n",
                            split >= 100 ? "STARVED" : "MISS-HEAVY", pool_frac*100.0, split);
                } else if (!op_set_inline) {
                    fprintf(stderr, "[serve] Auto serve-mode: HIT-RICH (pool≈%.0f%% of experts) "
                            "→ pool-direct (default). Very miss-heavy models can try "
                            "ALPHA_CPU_INLINE=1 ALPHA_INLINE_SPLIT=~50 (see scripts/serve_3090_122b.sh).\n",
                            pool_frac*100.0);
                }
            };

            // Auto fused-infer: peek at GGUF metadata to detect MoE models that won't fit in VRAM.
            // If n_expert > 0 and model_size > free VRAM: auto-offload expert weights to CPU.
            struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
            struct gguf_context * gf = gguf_init_from_file(model_path, gp);
            if (!gf) {
                fprintf(stderr, "[serve] GGUF peek failed for '%s' — skipping auto fused-infer\n", model_path);
            }
            if (gf) {
                uint32_t n_expert = 0;
                // Try common architecture prefixes for n_expert key
                for (const char * prefix : {"qwen35moe", "qwen2moe", "mixtral", "deepseek2",
                                             "dbrx", "grok", "jamba", "olmoe", "granite"}) {
                    char key[128];
                    snprintf(key, sizeof(key), "%s.expert_count", prefix);
                    int64_t kid = gguf_find_key(gf, key);
                    if (kid >= 0) { n_expert = gguf_get_val_u32(gf, kid); break; }
                }
                // Also check the generic general.* namespace
                if (n_expert == 0) {
                    int64_t kid = gguf_find_key(gf, "general.expert_count");
                    if (kid >= 0) n_expert = gguf_get_val_u32(gf, kid);
                }
                if (n_expert > 0) { s_n_expert_peek = (int) n_expert; }   // MoE ubatch default (this peek runs on every backend path)
                // Also try architecture-specific key from general.architecture
                if (n_expert == 0) {
                    int64_t arch_kid = gguf_find_key(gf, "general.architecture");
                    if (arch_kid >= 0) {
                        const char * arch = gguf_get_val_str(gf, arch_kid);
                        char key[128];
                        snprintf(key, sizeof(key), "%s.expert_count", arch);
                        int64_t kid = gguf_find_key(gf, key);
                        if (kid >= 0) n_expert = gguf_get_val_u32(gf, kid);
                    }
                }

                fprintf(stderr, "[serve] GGUF peek: n_expert=%u, n_tensors=%lld\n",
                        n_expert, (long long)gguf_get_n_tensors(gf));

                if (n_expert > 1) {
                    size_t free_vram = 0, total_vram = 0;
                    alpha_vram_info(&free_vram, &total_vram);
                    // ALPHA_FORCE_VRAM_MB overrides the card VRAM the tier picker sees, so one machine can
                    // dry-run the auto tier decision (ALL_GPU / pool-direct / CPU-inline split) for any card
                    // size. Same knob the external-routing fit guard and the dense auto-fit honor.
                    { const char * fv = getenv("ALPHA_FORCE_VRAM_MB"); if (fv) { long v = atol(fv);
                        if (v > 0) { total_vram = (size_t) v * 1024 * 1024; free_vram = total_vram; } } }
                    size_t model_size = 0;
                    // Sum all tensor sizes in the GGUF metadata
                    int64_t n_tensors = gguf_get_n_tensors(gf);
                    for (int64_t i = 0; i < n_tensors; i++) {
                        model_size += gguf_get_tensor_size(gf, i);
                    }
                    // Split GGUF: n_tensors=0 (tensor data in other shards).
                    // Estimate model size from file system.
                    if (n_tensors == 0) {
                        // Try to get file size of all shards
                        std::string path(model_path);
                        auto dash = path.rfind("-00001-of-");
                        if (dash != std::string::npos) {
                            // Split GGUF pattern: *-00001-of-NNNNN.gguf
                            std::string prefix = path.substr(0, dash);
                            std::string suffix = path.substr(path.rfind('.'));
                            int n_shards = 0;
                            sscanf(path.c_str() + dash + 10, "%d", &n_shards);
                            for (int s = 1; s <= n_shards; s++) {
                                char shard[512];
                                snprintf(shard, sizeof(shard), "%s-%05d-of-%05d%s",
                                         prefix.c_str(), s, n_shards, suffix.c_str());
                                FILE * f = fopen(shard, "rb");
                                if (f) {
                                    fseek(f, 0, SEEK_END);
                                    model_size += (size_t)ftell(f);
                                    fclose(f);
                                }
                            }
                            fprintf(stderr, "[serve] Split GGUF: %d shards, total %.0f MB\n",
                                    n_shards, (float)model_size/(1024*1024));
                        }
                    }

                    const size_t vram_reserve = 2ULL * 1024 * 1024 * 1024;  // 2 GB for KV + compute
                    if (fused_infer_active) {
                        // Explicit -ot already put the experts on CPU: don't touch the overrides,
                        // just pick the sub-mode the auto path would have picked.
                        pick_serve_mode(n_expert, model_size, total_vram);
                    } else if (model_size + vram_reserve > total_vram) {
                        // Model + overhead exceeds GPU capacity: auto-enable fused-infer
                        ot_pattern = "ffn_(gate|up|down)_exps";
                        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
                        ot_overrides[0] = {ot_pattern.c_str(), cpu_buft};
                        ot_overrides[1] = {nullptr, nullptr};
                        model_params.tensor_buft_overrides = ot_overrides;
                        fprintf(stderr, "[serve] Auto fused-infer: MoE model (n_expert=%u, model=%.0f MB, "
                                "GPU=%.0f MB) → expert weights offloaded to CPU\n",
                                n_expert, (float)model_size/(1024*1024), (float)total_vram/(1024*1024));
                        fused_infer_active = true;

                        pick_serve_mode(n_expert, model_size, total_vram);
                    } else {
                        // Whole model (weights + reserve) fits VRAM -> keep experts on the GPU.
                        // This is the fastest tier (no PCIe/CPU pipe at all).
                        fprintf(stderr, "[serve] Auto: whole model fits VRAM (model=%.0f MB + 2GB "
                                "reserve <= GPU=%.0f MB) → ALL_GPU, experts stay on GPU (fastest, "
                                "no offload)\n",
                                (float)model_size/(1024*1024), (float)total_vram/(1024*1024));
                    }
                }
                gguf_free(gf);
            }
        }
#endif

        // CPU lm_head: the output matrix moves to host memory; the scheduler ships the final
        // hidden state D2H and runs the head GEMV on CPU threads (same math, different device).
        // The freed VRAM is picked up by the expert-pool prealloc below (it sizes from free
        // VRAM after model load). Pass 2's lm_head host copy detects the host-resident tensor
        // and reads it in place.
        //
        // Default off (ALPHA_CPU_LMHEAD=1 enables): the CPU head GEMV + D2H + sync adds
        // ~13 ms/token on the critical path. That hides under the DMA wall on plain decode but
        // nearly doubles the token time on the LUT draft path, which has no DMA.
        {
            static const char * lmhead_pat = "^output\\.weight$";
            const char * clh = getenv("ALPHA_CPU_LMHEAD");
            if (fused_infer_active && (clh && atoi(clh) != 0)) {
                int oi = 0;
                while (oi < 2 && ot_overrides[oi].pattern) oi++;
                if (oi < 2) {
                    ot_overrides[oi]     = { lmhead_pat, ggml_backend_cpu_buffer_type() };
                    ot_overrides[oi + 1] = { nullptr, nullptr };
                    model_params.tensor_buft_overrides = ot_overrides;
                    fprintf(stderr, "[serve] CPU lm_head ENABLED (ALPHA_CPU_LMHEAD=1): output.weight -> host (frees ~2GB VRAM; costs ~13ms/tok on the LUT draft path)\n");
                }
            }
        }

        const bool dre_on = fused_infer_active && getenv("ALPHA_DRE") && atoi(getenv("ALPHA_DRE")) != 0;
        if (dre_on) {
            // ALPHA_DRE: decode resolves experts on the device (ggml-cuda/dre.cu); prefill keeps the
            // scheduler's plain copy path. The expert cache is not registered: both would own the pool.
            ggml_backend_sched_set_direct_weight_fn(ggml_cuda_dre_direct_weight);
            // prefill expert staging: VRAM->VRAM for pool-resident experts, and the prompt's usage for warm admission
            if (!getenv("ALPHA_DRE_PF") || atoi(getenv("ALPHA_DRE_PF")) != 0) {
                ggml_backend_set_expert_cache_copy_fn(ggml_cuda_dre_prefill_copy);
                ggml_backend_set_expert_cache_flush_fn(ggml_cuda_dre_prefill_flush);
                ggml_backend_set_expert_cache_redirect_fn(ggml_cuda_dre_prefill_redirect);
                ggml_backend_set_expert_cache_stream_fn(ggml_cuda_dre_prefill_stream_whole);
                if (getenv("ALPHA_DRE_IDS_HIST")) ggml_backend_set_expert_cache_ids_fn(ggml_cuda_dre_prefill_ids);
                ggml_backend_set_expert_cache_pfx_fn(ggml_cuda_dre_prefill_x);   // ALPHA_DRE_PF_CPU (a no-op unless set)
            }
            fprintf(stderr, "[serve] ALPHA_DRE: device-resolved experts for decode (expert cache not registered)\n");
        }
        if (fused_infer_active && !dre_on) {
            // Register the persistent expert cache
            ggml_backend_set_expert_cache_copy_fn(ggml_cuda_expert_pcache_copy);
            ggml_backend_set_expert_cache_flush_fn(ggml_cuda_expert_pcache_flush);
            ggml_backend_set_expert_cache_extra_fn(ggml_cuda_expert_pcache_get_extra);
            ggml_backend_set_expert_cache_ntokens_fn(ggml_cuda_expert_pcache_set_ntokens);
            ggml_backend_set_expert_cache_pass_begin_fn(ggml_cuda_expert_pcache_pass_begin);
            ggml_backend_set_expert_cache_ids_fn(ggml_cuda_expert_pcache_set_ids);
            ggml_backend_set_expert_cache_group_fn(ggml_cuda_expert_pcache_set_group);
            fprintf(stderr, "[serve] Fused-infer expert cache registered (VMM + D2D fallback)\n");
        }

        model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            fprintf(stderr, "[serve] Failed to load model\n");
            return 1;
        }

        vocab = llama_model_get_vocab(model);

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx * n_slots;
        // iswa models (Gemma sliding-window attention): llama_context_default_params() sets
        // swa_full=true, which sizes the sliding-window KV cache to the full n_ctx instead of the
        // window, so a long-context Gemma runs out of memory during warmup. Bound it by default so
        // the SWA layers cost their window and only the few global layers scale with n_ctx.
        // ALPHA_SWA_FULL=1 restores the full cache (needed only if a feature branches sequences on
        // an SWA model and must retain the tail).
        ctx_params.swa_full = getenv("ALPHA_SWA_FULL") != nullptr;
        ctx_params.kv_unified = kv_unified;
        if (kv_unified) {
            fprintf(stderr, "[serve] kv_unified: ON (shared KV buffer across sequences)\n");
        }
        // ALPHA_REF_DUMP: dump per-layer reference activations (L2 norm) for comparing
        // the hand-written FlashInfer engine against the known-good ggml graph.
        if (getenv("ALPHA_REF_DUMP")) {
            ctx_params.cb_eval = [](struct ggml_tensor * t, bool ask, void * /*ud*/) -> bool {
                if (ask) {
                    const char * n = t->name;
                    return strstr(n, "linear_attn_out") || strstr(n, "l_out-") ||
                           strstr(n, "attn_post_norm") || strstr(n, "final_output") ||
                           strstr(n, "attn_norm") || strstr(n, "ffn_out") || strstr(n, "attn_residual") ||
                           strstr(n, "inp_embd") || strstr(n, "ffn_inp");
                }
                if (t->type != GGML_TYPE_F32) return true;
                int64_t ne = ggml_nelements(t);
                if (ne > 4200000) return true;
                std::vector<float> h(ne);
                ggml_backend_tensor_get(t, h.data(), 0, ggml_nbytes(t));
                double ss = 0; float mx = 0;
                for (int64_t i = 0; i < ne; i++) { float v = h[i]; ss += (double)v * v; float a = fabsf(v); if (a > mx) mx = a; }
                fprintf(stderr, "[REF] %-26s ne=%lld l2=%.5g maxabs=%.5g [0]=%.5g [1]=%.5g [2]=%.5g\n",
                        t->name, (long long)ne, sqrt(ss), mx, h[0], ne > 1 ? h[1] : 0.f, ne > 2 ? h[2] : 0.f);
                return true;
            };
        }
        // ALPHA_PATCH_DUMP=<file>: capture per-layer residual ("ffn_out") + final logits
        // ("result_output") for the first few decode tokens, so the hidden-state drift between two
        // runs can be diffed offline.
        if (const char * pdpath = getenv("ALPHA_PATCH_DUMP")) {
            static FILE * pdf = fopen(pdpath, "wb");
            static int pd_step = -1;   // decode-step index (bumped on each result_output)
            static const int PD_MAX_STEPS = 4;
            ctx_params.cb_eval = [](struct ggml_tensor * t, bool ask, void * /*ud*/) -> bool {
                if (ask) {
                    const char * n = t->name;
                    return strstr(n, "ffn_out") || strstr(n, "result_output");
                }
                if (!pdf) return true;
                if (t->type != GGML_TYPE_F32) return true;
                const int64_t ne0 = t->ne[0];
                const int64_t nt  = t->ne[1];         // tokens in this ubatch
                if (nt != 1) return true;             // decode only (skip prefill chunks)
                const bool is_logits = strstr(t->name, "result_output");
                if (is_logits) pd_step++;             // one result_output per decode step
                if (pd_step < 0 || pd_step >= PD_MAX_STEPS) return true;
                std::vector<float> h((size_t)ne0);
                ggml_backend_tensor_get(t, h.data(), 0, ne0 * sizeof(float));
                // record: name[24] | il(i32) | ne0(i32) | step(i32) | floats
                char nm[24]; memset(nm, 0, sizeof(nm));
                strncpy(nm, t->name, sizeof(nm) - 1);
                int32_t il = 0; { const char * dash = strrchr(t->name, '-'); if (dash) il = atoi(dash + 1); }
                int32_t ne0_i = (int32_t)ne0, step_i = (int32_t)pd_step;
                fwrite(nm, 1, sizeof(nm), pdf);
                fwrite(&il, sizeof(il), 1, pdf);
                fwrite(&ne0_i, sizeof(ne0_i), 1, pdf);
                fwrite(&step_i, sizeof(step_i), 1, pdf);
                fwrite(h.data(), sizeof(float), (size_t)ne0, pdf);
                fflush(pdf);
                return true;
            };
            fprintf(stderr, "[serve] ALPHA_PATCH_DUMP active → %s (first %d decode steps)\n",
                    pdpath, PD_MAX_STEPS);
        }
        // ALPHA_ROUTE_DUMP=<file>: capture per-(decode-token,layer) routed expert IDs
        // (ffn_moe_topk, weight-ordered) + their normalized gate weights (ffn_moe_weights_norm),
        // for offline routing-locality analysis: are the cold experts (rarely routed, so not
        // pool-resident) also low-weight (high argsort rank, safely droppable like ALPHA_GEN_TOPK),
        // or do they include high-weight experts? Runs in the coherent ALL_GPU path (no pool), all
        // decode tokens. Record: char tag('i'/'w') | i32 il | i32 nexp | payload(nexp * int32 ids
        // or nexp * float weights).
        if (const char * rdpath = getenv("ALPHA_ROUTE_DUMP")) {
            static FILE * rdf = fopen(rdpath, "wb");
            ctx_params.cb_eval = [](struct ggml_tensor * t, bool ask, void * /*ud*/) -> bool {
                if (ask) {
                    return strstr(t->name, "ffn_moe_topk") || strstr(t->name, "ffn_moe_weights_norm");
                }
                if (!rdf) return true;
                if (t->ne[1] != 1) return true;   // decode tokens only (ubatch of 1)
                const bool is_ids = strstr(t->name, "ffn_moe_topk");
                const int32_t nexp = (int32_t) t->ne[0];
                int32_t il = 0; { const char * dash = strrchr(t->name, '-'); if (dash) il = atoi(dash + 1); }
                char tag;
                if (is_ids) {
                    if (t->type != GGML_TYPE_I32) return true;
                    tag = 'i';
                    std::vector<int32_t> v((size_t) nexp);
                    ggml_backend_tensor_get(t, v.data(), 0, nexp * sizeof(int32_t));
                    fwrite(&tag, 1, 1, rdf); fwrite(&il, sizeof(il), 1, rdf); fwrite(&nexp, sizeof(nexp), 1, rdf);
                    fwrite(v.data(), sizeof(int32_t), (size_t) nexp, rdf);
                } else {
                    if (t->type != GGML_TYPE_F32) return true;
                    tag = 'w';
                    std::vector<float> v((size_t) nexp);
                    ggml_backend_tensor_get(t, v.data(), 0, nexp * sizeof(float));
                    fwrite(&tag, 1, 1, rdf); fwrite(&il, sizeof(il), 1, rdf); fwrite(&nexp, sizeof(nexp), 1, rdf);
                    fwrite(v.data(), sizeof(float), (size_t) nexp, rdf);
                }
                fflush(rdf);
                return true;
            };
            fprintf(stderr, "[serve] ALPHA_ROUTE_DUMP active -> %s (ffn_moe_topk + weights, all decode tokens)\n", rdpath);
        }
        // ALPHA_MOE_UBATCH: MoE prefill amortises a per-call fixed term in the fused expert GEMM, so a
        // resident MoE grows its prefill micro-batch to 2048 (a large prefill gain, decode unchanged;
        // dense models gain little, so this is MoE-only). It uses the runtime-grow path (rollback-safe
        // compute buffers), so a card that cannot fit the bigger buffer degrades to 512-token chunks
        // instead of failing at load.
        // ALPHA_MOE_UBATCH_PREFILL=N overrides the target (0 = off); explicit --ubatch / --ubatch-prefill win.
        if (s_n_expert_peek > 0 && !fused_infer_active && !ubatch_explicit && ubatch_prefill == 0) {
            const char * e = getenv("ALPHA_MOE_UBATCH_PREFILL");
            const int target = e ? atoi(e) : 2048;
            if (target > n_ubatch) {
                ubatch_prefill = target;
                fprintf(stderr, "[serve] MoE resident: prefill micro-batch grows to %d on demand (base ubatch %d; "
                                "ALPHA_MOE_UBATCH_PREFILL=0 disables, --ubatch N pins)\n", target, n_ubatch);
            }
        }

        // Runtime grow is forced off under fused-infer: growing the compute buffer before/after the
        // expert pool carve displaces pool budget, and the smaller pool costs more decode than the
        // grow gains in prefill.
        // ALPHA_DRE: device-resolved experts borrow the grown buffer from the expert pool for the prompt and
        // return it before decode (ggml_cuda_dre_borrow/return), so the grow costs decode nothing.
        const bool dre_borrow = fused_infer_active && getenv("ALPHA_DRE") && atoi(getenv("ALPHA_DRE")) != 0 &&
                                (!getenv("ALPHA_DRE_PREFILL_UB") || atoi(getenv("ALPHA_DRE_PREFILL_UB")) > 0);
        if (dre_borrow && !ubatch_explicit && ubatch_prefill == 0) {
            const char * e = getenv("ALPHA_DRE_PREFILL_UB");
            // with the pp logits reservation capped (ALPHA_RESERVE_PP_OUTPUTS) a micro-batch token costs ~0.15 MB of
            // compute buffer, so one micro-batch covers most prompts: every expert then crosses PCIe once per prompt
            ubatch_prefill = e ? atoi(e) : 4096;
            if (ubatch_prefill > 32768) ubatch_prefill = 32768;
            fprintf(stderr, "[serve] ALPHA_DRE: prefill micro-batch grows to %d, borrowing the buffer from the expert pool\n",
                    ubatch_prefill);
            // one pinned block for the scheduler's host buffer of every long prompt (see ggml_cuda_host_malloc):
            // ~0.18 MB per micro-batch token (KQ mask, QSA and PLE inputs), with the 50% growth headroom
            if (!getenv("ALPHA_CUDA_HOST_PREWARM_MB")) {
                const int mb = std::max(256, (int) ((double) ubatch_prefill * 0.18));
                alpha_setenv("ALPHA_CUDA_HOST_PREWARM_MB", std::to_string(mb).c_str());
            }
        }
        if (ubatch_prefill > 0 && fused_infer_active && !dre_borrow) {
            fprintf(stderr, "[serve] --ubatch-prefill disabled (fused-infer: expert pool owns the VRAM headroom)\n");
            ubatch_prefill = 0;
        }
        alphaserve::g_serve_ubatch_prefill = ubatch_prefill;

        // n_batch must cover the largest prefill chunk we may ever send;
        // n_ubatch starts small; the engine grows it on demand (rollback-safe).
        ctx_params.n_batch = std::max(n_ubatch, ubatch_prefill);
        ctx_params.n_ubatch = n_ubatch;
        alphaserve::g_serve_prompt_chunk = n_ubatch;  // prefill chunks fill exactly one ubatch
        alphaserve::g_serve_n_batch = ctx_params.n_batch;  // scheduler token budget (assert n_tokens_all <= n_batch)
        // Generation length cap: llama gives each sequence n_ctx_seq = -c positions.
        // Stop cleanly (finish_reason=length) before the KV runs out; past it,
        // llama_decode fails mid-generation.
        alphaserve::g_serve_ctx_per_slot = n_ctx;
        ctx_params.n_seq_max = n_slots;

        // ALPHA_FORCE_KV_Q4: force Q4_0 KV cache to free VRAM for a larger expert pool
        // (fused-infer: more experts cached -> higher hit -> faster). Attention KV only;
        // the recurrent/DeltaNet state is unaffected. Small quality cost on the GQA layers.
        bool kv_forced_q4 = false;
        if (getenv("ALPHA_FORCE_KV_Q4")) {
            ctx_params.type_k = GGML_TYPE_Q4_0;
            ctx_params.type_v = GGML_TYPE_Q4_0;
            kv_forced_q4 = true;
            fprintf(stderr, "[serve] ALPHA_FORCE_KV_Q4: forcing Q4_0 KV cache (frees VRAM for expert pool)\n");
        }
        // ALPHA_FORCE_KV_F16: pin the KV cache to f16 (skip the auto-Q4 downgrade below).
        // CUDA flash-attn cannot read Q4_0 KV at some head geometries (e.g. gemma4 SWA).
        // Auto KV cache type. llama_model_kv_size_bytes sums the attention layers with their real GQA
        // widths, so the decision is made before any allocation:
        //   f16 KV + compute headroom for n_batch fits free VRAM  -> f16
        //   else Q4_0 KV (and if even that plus the grown micro-batch does not fit, the grow is disabled)
        // A try-then-fallback stays as a backstop for anything the estimate misses.
        // ALPHA_FORCE_KV_F16 / ALPHA_FORCE_KV_Q4 still force; ALPHA_KV_TRY_F16=1 forces the blind try.
        bool kv_try_f16 = false;
#ifdef GGML_USE_CUDA
        if (!kv_forced_q4 && !getenv("ALPHA_FORCE_KV_F16")) {
            const char * e_try = getenv("ALPHA_KV_TRY_F16");
            const bool blind_try = e_try && atoi(e_try) != 0;
            size_t free_vram = 0, total_vram = 0;
            alpha_vram_info(&free_vram, &total_vram);
            const uint32_t n_cells = ctx_params.n_ctx;
            const size_t kv_f16 = llama_model_kv_size_bytes(model, n_cells, GGML_TYPE_F16, GGML_TYPE_F16);
            const size_t kv_q4  = llama_model_kv_size_bytes(model, n_cells, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
            // Compute-buffer envelope, ~n_embd x 480 B per micro-batch token plus a fixed term; logits live in a
            // host buffer and do not count. Keep this close to reality: an overestimate picks Q4_0 KV where f16
            // fits, which changes MTP's text and slows its verify pass.
            const int64_t n_embd_c = llama_model_n_embd(model);
            auto compute_bytes = [&](int nb) {
                return (size_t) 128 * 1024 * 1024 + (size_t) nb * n_embd_c * 512;
            };
            const size_t cb_base = compute_bytes(n_ubatch);                 // allocated by the context itself
            const size_t cb_grow = ubatch_prefill > 0 ? compute_bytes(ubatch_prefill) : cb_base;
            // ALPHA_MTP: budget what MTP allocates after the context: the device mirror of the recurrent state
            // every window snapshots for verify rollback, the head's own F32 KV, and the CUDA graph instances of
            // each verify width. Unbudgeted, the mirror's cudaMalloc can fail every window on a tight card, so the
            // window bails after drafting and MTP silently runs as slower plain decode.
            // Policy, exactness first: MTP output equals plain greedy only with f16 KV (Q4_0 KV breaks it), so the
            // reserve never silently demotes the KV. If f16 + reserve does not fit, MTP is disabled up front
            // (plain decode, exact, no mid-run graph OOM); ALPHA_MTP_KV_Q4=1 trades that for Q4_0 KV.
            {
                const char * em = getenv("ALPHA_MTP");
                const size_t rs = (em && atoi(em) != 0) ? llama_model_rs_size_bytes(model, ctx_params.n_seq_max) : 0;
                if (rs > 0) {
                    const char * er = getenv("ALPHA_REPLAYSSM");
                    const char * ed = getenv("ALPHA_MTP_DRAFT_N");
                    const bool deep = (er && atoi(er) != 0) || (ed && atoi(ed) > 1);
                    // slack: head KV (~33 MB at 4K) + graph instances. ReplaySSM or depth > 1 needs ~170+ MB for
                    // cudaGraphInstantiate; depth 1 fits in 64 MB.
                    const size_t slack = (size_t) (deep ? 192 : 64) * 1024 * 1024;
                    const size_t res   = rs + slack;
                    const char * eq = getenv("ALPHA_MTP_KV_Q4");
                    const bool allow_q4 = eq && atoi(eq) != 0;
                    const double MBr = 1024.0 * 1024.0;
                    if (kv_f16 + cb_base + res <= free_vram) {
                        fprintf(stderr, "[serve] MTP reserve: %.0f MB (state mirror %.0f + %.0f slack) fits with f16 KV\n",
                                res / MBr, rs / MBr, slack / MBr);
                        free_vram -= res;
                    } else if (allow_q4) {
                        fprintf(stderr, "[serve] MTP reserve: %.0f MB held back (ALPHA_MTP_KV_Q4=1): KV may drop to Q4_0, and "
                                        "MTP output is then NOT byte-identical to plain greedy\n", res / MBr);
                        free_vram = free_vram > res ? free_vram - res : 0;
                    } else {
                        alphaserve::mtp_disable("f16 KV + compute + MTP reserve does not fit");
                        fprintf(stderr, "[serve] MTP DISABLED: f16 KV %.0f + compute %.0f + MTP reserve %.0f MB = %.0f MB > free %.0f MB. "
                                        "Decoding plain (exact). Lower -c, or set ALPHA_MTP_KV_Q4=1 to run MTP on Q4_0 KV.\n",
                                kv_f16 / MBr, cb_base / MBr, res / MBr, (kv_f16 + cb_base + res) / MBr, free_vram / MBr);
                    }
                }
            }
            // Runtime slack for the grow. The grow aborts hard on OOM, and the peak during a request exceeds the
            // compute buffer reserved at start. compute_bytes() already over-reserves the allocator by ~185 MiB at
            // ubatch 1536, so the 331 MiB margin here gives ~516 MiB of real slack.
            // ALPHA_MOE_GROW_MARGIN_MB overrides (negative values clamp to 0).
            const size_t grow_margin = []() {
                const char * e = getenv("ALPHA_MOE_GROW_MARGIN_MB");
                const int mb = e ? atoi(e) : 331;
                return (size_t) (mb > 0 ? mb : 0) * 1024 * 1024;
            }();
            const double MB = 1024.0 * 1024.0;
            auto disable_grow = [&](const char * why) {
                if (ubatch_prefill > 0) {
                    fprintf(stderr, "[serve] micro-batch grow to %d disabled: %s (grown compute %.0f MB + %.0f MB margin, free %.0f MB); staying at ubatch %d\n",
                            ubatch_prefill, why, cb_grow / MB, grow_margin / MB, free_vram / MB, n_ubatch);
                    ubatch_prefill = 0;
                    alphaserve::g_serve_ubatch_prefill = 0;
                    ctx_params.n_batch = n_ubatch;
                    alphaserve::g_serve_n_batch = ctx_params.n_batch;
                }
            };
            // Priority: f16 KV (~+17-20% decode) over the micro-batch grow (~-15-24% prefill) when only one fits.
            size_t kv_chosen = kv_f16;
            if (blind_try) {
                kv_try_f16 = true;
            } else if (kv_f16 + cb_base <= free_vram) {
                fprintf(stderr, "[serve] Auto KV cache: f16 (exact %.0f MB + compute %.0f MB at ubatch %d, free %.0f MB)\n",
                        kv_f16 / MB, cb_base / MB, n_ubatch, free_vram / MB);
                kv_try_f16 = true;   // context creation fails softly -> the retry below still covers an estimate miss
            } else {
                ctx_params.type_k = GGML_TYPE_Q4_0;
                ctx_params.type_v = GGML_TYPE_Q4_0;
                kv_chosen = kv_q4;
                fprintf(stderr, "[serve] Auto KV cache: Q4_0 (f16 would need %.0f MB + compute %.0f MB at ubatch %d, free %.0f MB; Q4_0 is %.0f MB)\n",
                        kv_f16 / MB, cb_base / MB, n_ubatch, free_vram / MB, kv_q4 / MB);
                if (kv_q4 + cb_base > free_vram) {
                    fprintf(stderr, "[serve] WARNING: even Q4_0 KV + base compute may not fit (need %.0f MB, free %.0f MB)\n",
                            (kv_q4 + cb_base) / MB, free_vram / MB);
                }
            }
            // Step the grow target down instead of all-or-nothing: when the full grow does not fit next to
            // f16 KV, a smaller one usually does and keeps most of the prefill gain. Rungs: target, 3/4, 5/8,
            // 1/2 (256-aligned). The 5/8 rung keeps a small drop in free VRAM between boots from costing two
            // notches instead of one.
            // ALPHA_DRE: a borrowed micro-batch takes its buffer from the expert pool at prefill time (and
            // llama_set_n_ubatch rolls back on failure), so free VRAM at startup is not its budget
            if (!blind_try && ubatch_prefill > 0 && !dre_borrow) {
                const int want = ubatch_prefill;
                const int rungs[4] = { want, (want * 3 / 4) / 256 * 256, (want * 5 / 8) / 256 * 256, (want / 2) / 256 * 256 };
                int pick = 0;
                for (int r : rungs) {
                    if (r <= n_ubatch) continue;
                    if (kv_chosen + compute_bytes(r) + grow_margin <= free_vram) { pick = r; break; }
                }
                if (pick == 0) {
                    disable_grow("no ladder rung fits next to the KV");
                } else {
                    fprintf(stderr, "[serve] micro-batch grow ladder: rung %d of target %d (KV %.0f MB + compute %.0f MB + margin %.0f MB = %.0f MB, free %.0f MB)\n",
                            pick, want, kv_chosen / MB, compute_bytes(pick) / MB, grow_margin / MB,
                            (kv_chosen + compute_bytes(pick) + grow_margin) / MB, free_vram / MB);
                    if (pick != want) {
                        ubatch_prefill = pick;
                        alphaserve::g_serve_ubatch_prefill = pick;
                        ctx_params.n_batch = std::max(n_ubatch, pick);
                        alphaserve::g_serve_n_batch = ctx_params.n_batch;
                    }
                }
            }
        }
#endif

        // MTP identity: with one slot, plain decode never runs 2..16-token batches except short prefills, so
        // pinning those widths to the 1-token kernel geometry costs nothing and makes MTP greedy text equal
        // plain greedy text on prompts of <= 16 tokens too. Co-batched serving (-np > 1) keeps the stock
        // kernels unless ALPHA_MTP is on. Explicit settings win. Set before the context exists: the kernels
        // read these once, on first use.
        if (n_slots == 1) {
            int n_set = 0;
            for (const char * k : { "ALPHA_FA_BI_MAXQ", "ALPHA_TOPK_FUSE_MAXROWS", "ALPHA_MMID_SPLIT", "ALPHA_MMVF_BI_MAXQ" }) {
                if (!getenv(k)) { alpha_setenv(k, "16"); n_set++; }
            }
            if (n_set > 0) {
                fprintf(stderr, "[serve] single slot: small-batch kernels pinned to the 1-token geometry (%d of 4 set; MTP == plain greedy)\n", n_set);
            }
        }

        // image input: the projector loads before the context, because the expert pool sizes itself from the free
        // VRAM during context creation (sched reserve) and a projector loaded after it would find no VRAM left
        if (mmproj_path && *mmproj_path) {
            mtmd_context_params mp = mtmd_context_params_default();
            mp.use_gpu = true;
            mp.print_timings = false;
            mp.n_threads = 8;
            mp.warmup = true;
            alphaserve::g_serve_mtmd = mtmd_init_from_file(mmproj_path, model, mp);
            if (!alphaserve::g_serve_mtmd) {
                fprintf(stderr, "[serve] --mmproj %s: failed to load the projector\n", mmproj_path);
                llama_model_free(model);
                return 1;
            }
            fprintf(stderr, "[serve] image input ON: %s (vision %s, M-RoPE %s)\n", mmproj_path,
                    mtmd_support_vision(alphaserve::g_serve_mtmd) ? "yes" : "no",
                    mtmd_decode_use_mrope(alphaserve::g_serve_mtmd) ? "yes" : "no");
        }
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx && kv_try_f16) {
            fprintf(stderr, "[serve] Auto KV cache: f16 context did not fit, retrying with Q4_0 KV\n");
            ctx_params.type_k = GGML_TYPE_Q4_0;
            ctx_params.type_v = GGML_TYPE_Q4_0;
            ctx = llama_init_from_model(model, ctx_params);
        } else if (ctx && kv_try_f16) {
            fprintf(stderr, "[serve] Auto KV cache: f16 allocated (exact-size decision confirmed by the allocator)\n");
        }
        if (!ctx) {
            fprintf(stderr, "[serve] Failed to create context\n");
            llama_model_free(model);
            return 1;
        }
#ifdef ALPHA_DELTANET
        // DeltaNet: load a prebuilt recurrent state. The prescan build stays offline in alphallama
        // (it needs the BM25 corpus); AlphaServe only loads the finished state. The state-prior
        // mechanism (build_rs) carries it into the forward pass.
        if (deltanet_state && *deltanet_state) {
            uint64_t dn_toks = alpha_deltanet_state_load(deltanet_state, ctx);
            fprintf(stderr, "[serve] DeltaNet: loaded state '%s' (%llu tokens)\n",
                    deltanet_state, (unsigned long long) dn_toks);
        }
#endif

        // Fused-infer: claim the expert-pool VRAM now, while the allocation picture is deterministic
        // (weights + KV + configured-ubatch compute buffers). Otherwise a long first request grows
        // the prefill compute buffers before the lazy pool init and the pool permanently gets the
        // scraps. A later ubatch grow simply fails and rolls back to chunked prefill.
        if (fused_infer_active) {
            // Dynamic KV/compute-aware pool reserve (ALPHA_DYNAMIC_RESERVE, default off). The fixed 1GB
            // reserve over-provisions tight configs (starving the expert pool) yet under-provisions big
            // models. This computes the reserve from the actual model width + prefill batch. KV for n_ctx
            // is already in free_mem; this covers the transient compute working set. It errs conservative
            // (assumes non-flash attention unless FA is explicitly enabled) so it never under-reserves.
            // TODO: calibrate the constants (activations in flight, margin).
            if (getenv("ALPHA_DYNAMIC_RESERVE")) {
                const size_t n_embd  = (size_t) llama_model_n_embd(model);
                const size_t n_head  = (size_t) llama_model_n_head(model);
                // The compute buffer is sized by the physical ubatch (the graph runs one ubatch
                // at a time), not the logical n_batch. KV for n_ctx is already subtracted from
                // free_mem before this reserve is applied, so only the transient graph is covered.
                const size_t nub     = (size_t) llama_n_ubatch(ctx);
                // Per-sequence context: a single prefill ubatch attends only within its own
                // sequence, so the KQ buffer is nub x n_ctx_seq, not the multi-seq total n_ctx
                // (the total over-reserves ~n_seq_max-fold and needlessly starves the pool).
                const size_t n_ctx   = (size_t) llama_n_ctx_seq(ctx);
                const bool   fa_on   = (ctx_params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
                size_t act  = nub * n_embd * sizeof(float) * 8;                 // ~8 full-width activations in flight
                // Without flash-attn the last prefill ubatch materializes a KQ scores tensor of
                // [n_ubatch x n_ctx_seq] per query head, the dominant term for long-context non-FA;
                // it scales with n_ctx, not nub^2.
                size_t attn = fa_on ? 0 : (nub * n_ctx * n_head * sizeof(float));
                size_t reserve = act + attn + (size_t)384*1024*1024;           // + safety margin for fragmentation
                if (reserve < (size_t)512*1024*1024) reserve = (size_t)512*1024*1024;  // floor
                ggml_cuda_set_pool_reserve_bytes(reserve);
                fprintf(stderr, "[serve] dynamic pool reserve = %.0f MB (act=%.0f attn=%.0f MB; "
                        "n_ubatch=%zu n_ctx=%zu n_embd=%zu n_head=%zu fa=%d) [ALPHA_DYNAMIC_RESERVE]\n",
                        (double)reserve/(1024*1024), (double)act/(1024*1024), (double)attn/(1024*1024),
                        nub, n_ctx, n_embd, n_head, (int)fa_on);
            }
            // ALPHA_MTP_KV_PREALLOC (default on; =0 restores the lazy path): allocate the MTP head's KV (2 x n_ctx x
            // kv_row f32, 128 MB at 32K) now. Allocated lazily, it can fail while a prefill holds the VRAM, leaving
            // the first request without drafting. With MTP on, the DRE reserve defaults to 2 GB (dre.cu) to cover it.
            if (getenv("ALPHA_MTP") && atoi(getenv("ALPHA_MTP")) != 0 &&
                (!getenv("ALPHA_MTP_KV_PREALLOC") || atoi(getenv("ALPHA_MTP_KV_PREALLOC")) != 0)) {   // default ON (reserve covers it)
                const bool ok = alpha_mtp_kv_prealloc(ctx);
                fprintf(stderr, "[serve] MTP head KV preallocated: %s\n", ok ? "yes" : "no (no MTP head or no VRAM)");
            }
            ggml_cuda_expert_pcache_prealloc();
        }
        // ALPHA_DRE: pin every host expert buffer now (the MTP sidecar's included: no warmup reads it). This must
        // come after the pool claim; pinning before the context runs out of memory on the first decode graph
        // instantiate under the prefill-streaming path.
        if (fused_infer_active && getenv("ALPHA_DRE") && atoi(getenv("ALPHA_DRE")) != 0) {
            alpha_model_foreach_tensor(model, [](const ggml_tensor * t, void *) { ggml_cuda_dre_preregister(t); }, nullptr);
        }

        // Marlin INT4 Tensor Core: only for pre-converted AWQ/GPTQ models (--marlin-dir)
        // Q4_K runtime conversion disabled: too slow for MoE (768 expert weights x 1s each)

        // Multi-user expert cache partitioning for MoE models
        if (n_slots > 1) {
            ggml_cuda_init_expert_users(n_slots);
            fprintf(stderr, "[serve] Expert cache: %d user partitions for MoE models\n", n_slots);
        }

        local_srv = std::make_unique<alphaserve::server>();
        local_srv->init(ctx, n_slots);

        // Detect M-RoPE: slot prefix matching causes position crash on these models.
        // The cached KV position (X) from the previous request exceeds the new prefix
        // start position (Y), violating M-RoPE's X < Y constraint.
        bool is_mrope = false;
        if (model) {
            auto rtype = llama_model_rope_type(model);
            is_mrope = (rtype == LLAMA_ROPE_TYPE_MROPE || rtype == LLAMA_ROPE_TYPE_IMROPE);
        }

        // ALPHA_HYBRID_PREFIX: on hybrid/recurrent models llama_memory_hybrid::seq_rm tries the recurrent half
        // first and, when that refuses, returns false without truncating attention, so old KV positions survive
        // above the new prefill (the cause behind the M-RoPE and offload guards below). Checkpoint-based reuse
        // (snapshot the recurrent state at the last turn boundary, restore it before truncating) removes it.
        // MTP: the checkpoint also carries the head's boundary row h_{ckpt-1}, so the head-KV column at the
        // checkpoint is rebuilt with the new request's token (verify keeps output exact regardless; the row
        // protects acceptance). Declined for ALPHA_DRAFT_MODEL: the external drafter's mirror is reset only at
        // position 0 and would draft from the old continuation.
        const bool model_hybrid = model && (llama_model_is_hybrid(model) || llama_model_is_recurrent(model));
        const bool hyb_prefix_env = []() {
            const char * e = getenv("ALPHA_HYBRID_PREFIX");
            return e && e[0] && atoi(e) != 0;
        }();
        const bool mtp_env = []() {
            const char * e = getenv("ALPHA_MTP");
            return e && e[0] && atoi(e) != 0;
        }();
        const bool draft_model_env = []() { const char * e = getenv("ALPHA_DRAFT_MODEL"); return e && e[0]; }();
        const char * hyb_decline = nullptr;
        if (mtp_env && draft_model_env) hyb_decline = "ALPHA_DRAFT_MODEL mirror is not checkpoint-aware";
        if (model_hybrid && hyb_prefix_env && !hyb_decline) {
            local_srv->enable_hybrid_snapshot();
            fprintf(stderr, "[serve] Hybrid prefix reuse ON (ALPHA_HYBRID_PREFIX): recurrent checkpoint at the last "
                            "turn boundary, restore-then-truncate (supersedes the M-RoPE/offload guards)%s\n",
                    mtp_env ? "; MTP head boundary row carried" : "");
        } else if (model_hybrid && !is_mrope && !override_tensor && !fused_infer_active) {
            // Prefix caching here would hit the seq_rm no-op above: stale recurrent state and drifting
            // output. No reuse is strictly better than broken reuse.
            fprintf(stderr, "[serve] Prefix caching disabled (hybrid/recurrent model: partial truncation is "
                            "refused by recurrent memory; set ALPHA_HYBRID_PREFIX=1 for checkpoint reuse)%s\n",
                    (hyb_prefix_env && hyb_decline) ? " -- ALPHA_HYBRID_PREFIX declined (see next line)" : "");
            if (hyb_prefix_env && hyb_decline) fprintf(stderr, "[serve] ALPHA_HYBRID_PREFIX declined: %s\n", hyb_decline);
            local_srv->disable_slot_prefix_match();
        } else if (!override_tensor && !fused_infer_active && !is_mrope) {
            local_srv->set_prefix_caching(true);
        } else {
            if (hyb_prefix_env && hyb_decline && model_hybrid) {
                fprintf(stderr, "[serve] ALPHA_HYBRID_PREFIX declined: %s\n", hyb_decline);
            }
            if (is_mrope) {
                fprintf(stderr, "[serve] Prefix caching + slot prefix match disabled (M-RoPE model)\n");
                local_srv->disable_slot_prefix_match();
            } else {
                fprintf(stderr, "[serve] Prefix caching disabled (MoE expert offload active)\n");
            }
        }
        local_srv->start();

        local_router = std::make_unique<alphaserve::response_router>();

        // Initialize chat template from model
        if (!chat_tmpls && model) {
            chat_tmpls = common_chat_templates_init(model, "");
        }
    }

    // Initialize remote backend (SGLang or vLLM; MODE_AUTO's remote leg defaults to SGLang)
    if (need_remote) {
        const bool is_vllm = (mode == MODE_VLLM);
        fprintf(stderr, "[serve] Backend: %s at %s\n", is_vllm ? "vLLM" : "SGLang", backend_url);
        alphaserve::BackendConfig cfg;
        cfg.type = is_vllm ? alphaserve::BackendType::VLLM : alphaserve::BackendType::SGLANG;
        cfg.remote_url = backend_url;
        if (backend_key) cfg.remote_api_key = backend_key;
        remote_backend = alphaserve::create_backend(cfg);
    }

    // FlashInfer backend state
#ifdef ALPHA_HAS_FLASHINFER
    std::unique_ptr<alphaserve::PagedKVPool> fi_kv_pool;
    std::unique_ptr<alphaserve::ForwardEngine> fi_engine;
    std::unique_ptr<alphaserve::fi_gpu_engine> fi_gpu;

    // Check if --marlin-dir was specified (model_path reused for dir path)
    bool use_marlin_dir = false;
    const char * marlin_dir = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--marlin-dir") == 0 && i + 1 < argc) {
            marlin_dir = argv[i + 1];
            use_marlin_dir = true;
        }
    }

    if (need_flashinfer && use_marlin_dir) {
        // Marlin INT4 path: load weights from binary dir
        fprintf(stderr, "[serve] Backend: FlashInfer + Marlin INT4\n");
        fprintf(stderr, "[serve] Loading Marlin weights: %s\n", marlin_dir);

        // Still need llama_model for vocab: load the AWQ FP16 GGUF if available
        if (model_path) {
            llama_model_params model_params = llama_model_default_params();
            model_params.n_gpu_layers = 0;  // CPU only, just for vocab
            model = llama_model_load_from_file(model_path, model_params);
            if (model) vocab = llama_model_get_vocab(model);
        }

        fi_engine = std::make_unique<alphaserve::ForwardEngine>();

        // Init paged KV using config from Marlin weights
        // First do a partial init to get config
        fi_kv_pool = std::make_unique<alphaserve::PagedKVPool>();

        if (!fi_engine->init_marlin(marlin_dir, nullptr)) {
            fprintf(stderr, "[serve] Failed to load Marlin weights\n");
            return 1;
        }

        // Init KV pool with config from engine, then set on engine
        auto & cfg = fi_engine->config();
        if (!fi_kv_pool->init(cfg.n_head_kv, cfg.head_dim, cfg.n_layers, n_slots, n_ctx)) {
            fprintf(stderr, "[serve] Failed to init paged KV cache\n");
            return 1;
        }
        fi_engine->set_kv_pool(fi_kv_pool.get());

        // Scheduler + GPU engine (same as normal FlashInfer path)
        local_srv = std::make_unique<alphaserve::server>();
        local_srv->init_scheduler_only(vocab, n_slots);
        local_srv->start();

        fi_gpu = std::make_unique<alphaserve::fi_gpu_engine>();
        int fi_max_batch = std::max(512, n_slots * 2);
        // Honor ALPHA_FI_MIN_BATCH (caps the fi-engine's d_logits = batch x vocab x 4, ~508 MiB at 512):
        // keep it consistent with the ForwardEngine workspace cap so a big MoE fits on 24 GiB.
        if (const char * e = getenv("ALPHA_FI_MIN_BATCH")) fi_max_batch = std::max(16, atoi(e));
        if (!fi_gpu->init(fi_engine.get(), fi_kv_pool.get(), vocab,
                          local_srv->gpu_in_queue(), local_srv->gpu_out_queue(),
                          fi_max_batch)) {
            fprintf(stderr, "[serve] Failed to init FlashInfer GPU engine\n");
            return 1;
        }
        local_router = std::make_unique<alphaserve::response_router>();
        fprintf(stderr, "[serve] Marlin INT4 + FlashInfer ready (%d slots)\n", n_slots);

    } else if (need_flashinfer) {
        fprintf(stderr, "[serve] Backend: FlashInfer (native, scheduler-batched)\n");
        fprintf(stderr, "[serve] Loading model: %s\n", model_path);

        llama_model_params model_params = llama_model_default_params();
        // For auto-routed quantized models: keep ggml weights on CPU to avoid
        // double GPU allocation (FlashInfer engine dequants to its own GPU buffers)
        model_params.n_gpu_layers = auto_routed ? 0 : n_gpu_layers;
        // CPU_REPACK rewrites q4_K/q4_0 CPU tensors into an interleaved AVX layout.
        // The FI engine reads raw tensor bytes (dequant + quant-resident GEMV), so
        // repacked data silently garbles the output. Disable repacking for FI loads.
        model_params.use_extra_bufts = false;
        if (auto_routed) fprintf(stderr, "[serve] CPU-only model load (FlashInfer manages GPU weights)\n");
        model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            fprintf(stderr, "[serve] Failed to load model\n");
            return 1;
        }
        vocab = llama_model_get_vocab(model);

        if (!alphaserve::ForwardEngine::is_supported(model)) {
            fprintf(stderr, "[serve] Model not supported by FlashInfer backend\n");
            llama_model_free(model);
            return 1;
        }

        // Initialize paged KV cache. Auto-fit the pool to free VRAM (vLLM-style) so a fixed -np x -c does
        // not OOM: reserve for the FI weight copy (~model size, allocated in fi_engine->init below) + ~2 GiB
        // forward/logits headroom, then cap n_slots to what fits. n_slots is read back so the scheduler
        // (init_scheduler_only) uses the same capped value.
        const auto & hp = model->hparams;
        // Reserve = FI weight copy (~model size) + forward/dequant workspace + safety. The FI engine's fixed
        // GPU cost is ~1.74x model_size: the extra is a ~0.8x workspace (dequant scratch, lm_head fp16,
        // activations) with a floor for small models.
        const size_t GiB = 1024ull * 1024 * 1024;
        size_t model_bytes = (size_t) llama_model_size(model);
        size_t workspace = std::max(model_bytes * 4 / 5, (size_t)4 * GiB);
        // ALPHA_NO_FUSED drops the fused-weight duplication (wqkv/gate_up/dn_proj copies, ~0.43x model).
        // The default 0.8x workspace budgets for that duplication, so without it shrink the workspace to
        // 0.45x model and let auto-fit hand the freed bytes to KV slots, which raises FI's concurrency cap
        // on memory-bound models. Stays conservative (0.45x, 3 GiB floor) to cover the larger activation
        // buffers at the higher batch.
        if (getenv("ALPHA_NO_FUSED")) {
            workspace = std::max(model_bytes * 9 / 20, (size_t)3 * GiB);
            fprintf(stderr, "[serve] ALPHA_NO_FUSED: shrinking FI KV reserve workspace %zu→%zu MiB "
                    "(no fused-weight duplication) → more KV slots\n",
                    (size_t)(std::max(model_bytes*4/5,(size_t)4*GiB)/(1024*1024)), workspace/(1024*1024));
        }
        size_t fi_reserve = model_bytes + workspace + 1 * GiB;
        // ALPHA_FI_RESERVE_GB overrides the KV-pool reserve directly. The default 0.8x model
        // workspace assumes F16 weight copies (dense BF16 models); a quant-resident MoE keeps
        // weights in GGUF format, so the true FI footprint is ~model_size, not ~1.8x. Set this to
        // the GPU weight footprint (+ a little) so KV auto-fit can allocate real slots for
        // multi-user on a big MoE.
        if (const char * e = getenv("ALPHA_FI_RESERVE_GB")) {
            fi_reserve = (size_t)(atof(e) * GiB);
            fprintf(stderr, "[serve] ALPHA_FI_RESERVE_GB=%s → fi_reserve=%.2f GiB\n",
                    e, (double)fi_reserve / GiB);
        }
        fi_kv_pool = std::make_unique<alphaserve::PagedKVPool>();
        if (!fi_kv_pool->init(hp.n_head_kv(), hp.n_embd_head_k(),
                               hp.n_layer, n_slots, n_ctx, fi_reserve)) {
            fprintf(stderr, "[serve] Failed to init paged KV cache\n");
            llama_model_free(model);
            return 1;
        }
        if (fi_kv_pool->max_seqs != n_slots) {
            fprintf(stderr, "[serve] KV auto-fit: n_slots %d → %d\n", n_slots, fi_kv_pool->max_seqs);
            n_slots = fi_kv_pool->max_seqs;   // scheduler + fi_gpu must use the capped count
        }

        // Initialize forward engine
        fi_engine = std::make_unique<alphaserve::ForwardEngine>();
        if (!fi_engine->init(model, fi_kv_pool.get())) {
            fprintf(stderr, "[serve] Failed to init forward engine\n");
            llama_model_free(model);
            return 1;
        }

        // Use AlphaServe scheduler for request batching
        local_srv = std::make_unique<alphaserve::server>();
        local_srv->init_scheduler_only(vocab, n_slots);
        local_srv->start();  // starts scheduler thread

        // Wire fi_gpu_engine to scheduler's SPSC queues
        fi_gpu = std::make_unique<alphaserve::fi_gpu_engine>();
        // max_batch must cover prefill chunks (up to 512 tokens) + decode tokens
        int fi_max_batch = std::max(512, n_slots * 2);
        // Honor ALPHA_FI_MIN_BATCH (caps the fi-engine's d_logits = batch x vocab x 4, ~508 MiB at 512):
        // keep it consistent with the ForwardEngine workspace cap so a big MoE fits on 24 GiB.
        if (const char * e = getenv("ALPHA_FI_MIN_BATCH")) fi_max_batch = std::max(16, atoi(e));
        if (!fi_gpu->init(fi_engine.get(), fi_kv_pool.get(), vocab,
                          local_srv->gpu_in_queue(), local_srv->gpu_out_queue(),
                          fi_max_batch)) {
            fprintf(stderr, "[serve] Failed to init FlashInfer GPU engine\n");
            return 1;
        }

        // Redis KV cache for cross-session prefix reuse
        static alphaserve::KVCacheManager kv_cache_mgr;
        if (redis_kv) {
            char redis_host[256] = "localhost";
            int redis_port = 6379;
            if (sscanf(redis_kv, "%[^:]:%d", redis_host, &redis_port) >= 1) {
                if (kv_cache_mgr.connect(redis_host, redis_port)) {
                    kv_cache_mgr.set_model_hash(model_path ? model_path : "default");
                    fi_gpu->set_kv_cache(&kv_cache_mgr);
                }
            }
        }

        local_router = std::make_unique<alphaserve::response_router>();

        // Initialize chat template from model
        if (!chat_tmpls && model) {
            chat_tmpls = common_chat_templates_init(model, "");
        }

        // Enable prefix caching only for FlashInfer (ggml can't handle partial prefill), and not for
        // recurrent/hybrid (DeltaNet/SSM) models: their recurrent state can't be partially reused, so a
        // slot-reuse prefix match would reprocess only the suffix and leave the prefix's recurrent state
        // stale from the previous request, and output would drift across rounds.
        bool model_recurrent = model && (llama_model_is_recurrent(model) || llama_model_is_hybrid(model));
        if (model_recurrent) {
            local_srv->disable_slot_prefix_match();
            fprintf(stderr, "[serve] FlashInfer backend ready — prefix caching DISABLED (recurrent/hybrid model: state can't be partially reused)\n");
        } else {
            local_srv->set_prefix_caching(true);
            fprintf(stderr, "[serve] FlashInfer backend ready (scheduler-batched, prefix caching ON)\n");
        }
    }
#endif

#ifdef ALPHA_HAS_ALPHAINFER
    // AlphaInfer backend: AlphaServe's HTTP/scheduler/slots in front of AlphaInfer's engine.
    // This side loads only the GGUF vocabulary (tokenizer + chat template); the weights, KV and every kernel are
    // AlphaInfer's. n_streams = -np, ctx per stream = -c / -np (ALPHA_AI_CTX overrides, -1 = largest that fits),
    // mtp_k = ALPHA_AI_MTP_K (default 3), KV auto, workspace auto.
    std::unique_ptr<alphaserve::ai_gpu_engine> ai_gpu;
    if (mode == MODE_ALPHAINFER) {
        fprintf(stderr, "[serve] Backend: AlphaInfer\n");
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        // No GPU device on this side: a vocabulary load needs none, and enumerating CUDA0 for its free memory would
        // create a CUDA context and load the ggml-cuda module (~256 MiB) out of AlphaInfer's headroom.
        static ggml_backend_dev_t ai_no_devices[] = { nullptr };
        mp.devices = ai_no_devices;
        mp.n_gpu_layers = 0;
        model = llama_model_load_from_file(model_path, mp);
        if (!model) { fprintf(stderr, "[serve] failed to load the vocabulary of %s\n", model_path); return 1; }
        vocab = llama_model_get_vocab(model);
        // image input on the AlphaInfer path: the projector (mtmd) encodes on our side and the adapter hands the
        // embeddings + M-RoPE positions to ai_engine_prefill_mm. Loaded before the engine so its auto sizing (context,
        // expert slots) sees the projector's VRAM. ALPHA_MMPROJ_CPU=1 encodes on the CPU instead (no VRAM; an image
        // then takes seconds). The vocab-only model still carries the hparams mtmd checks the projector's width against.
        if (mmproj_path && *mmproj_path) {
            mtmd_context_params mmp = mtmd_context_params_default();
            mmp.use_gpu = !(getenv("ALPHA_MMPROJ_CPU") && atoi(getenv("ALPHA_MMPROJ_CPU")) != 0);
            mmp.print_timings = false;
            mmp.n_threads = 8;
            mmp.warmup = true;
            alphaserve::g_serve_mtmd = mtmd_init_from_file(mmproj_path, model, mmp);
            if (!alphaserve::g_serve_mtmd) {
                fprintf(stderr, "[serve] --mmproj %s: failed to load the projector\n", mmproj_path);
                llama_model_free(model);
                return 1;
            }
            setenv("AI_MM_ROWS", "1", /*overwrite=*/ 1);   // the engine stages image rows in buffers fixed at create
            fprintf(stderr, "[serve] image input ON (AlphaInfer): %s on the %s (vision %s, M-RoPE %s)\n", mmproj_path,
                    mmp.use_gpu ? "GPU" : "CPU", mtmd_support_vision(alphaserve::g_serve_mtmd) ? "yes" : "no",
                    mtmd_decode_use_mrope(alphaserve::g_serve_mtmd) ? "yes" : "no");
        }

        ai_engine_params ap = {};
        ap.model_path = model_path;
        ap.n_streams = n_slots;
        const char * ctx_env = getenv("ALPHA_AI_CTX");
        ap.ctx_cells_per_stream = ctx_env ? atoll(ctx_env) : (n_ctx > 2048 ? (int64_t) n_ctx / std::max(1, n_slots) : -1);
        // ALPHA_AI_EXPERT_SLOTS=auto|N keeps the routed experts in host memory: "auto" (-1) sizes the library's doorbell
        // offload from the VRAM left (needs AI_OFFLOAD=doorbell and a library built with AI_DOORBELL), N = N slots/layer
        ap.expert_slots = 0;
        if (const char * es = getenv("ALPHA_AI_EXPERT_SLOTS")) {
            ap.expert_slots = std::string(es) == "auto" ? -1 : atoi(es);
            if (ap.expert_slots != 0) setenv("AI_OFFLOAD", "doorbell", /*overwrite=*/ 0);
            fprintf(stderr, "[serve] ALPHA_AI_EXPERT_SLOTS=%s -> routed experts offloaded to host memory (%s, AI_OFFLOAD=%s)\n",
                    es, ap.expert_slots < 0 ? "auto-sized" : "fixed slots", getenv("AI_OFFLOAD") ? getenv("AI_OFFLOAD") : "");
        }
        // ALPHA_AI_WS_MIB=N: a fixed workspace (0 = the library's auto size); a row whose config is tight on VRAM pins
        // its size instead of a first-start bootstrap
        ap.ws_mib = getenv("ALPHA_AI_WS_MIB") ? atoi(getenv("ALPHA_AI_WS_MIB")) : 0;
        ap.host_embd = 0;
        ap.kv_type_is_q4_0 = -1;   // auto
        // ALPHA_AI_KV=f16|q8_0|q4_0 pins the KV type. Auto picks it from the VRAM left after the MTP head and verify
        // memory, so MTP on and off can land on different KV types and then their text differs. An exact == plain
        // comparison must pin the same type in both arms.
        if (const char * kv_env = getenv("ALPHA_AI_KV")) {
            const std::string kv(kv_env);
            ap.kv_type_is_q4_0 = kv == "q4_0" ? 1 : kv == "q8_0" ? 2 : kv == "f16" ? 0 : -1;
            fprintf(stderr, "[serve] ALPHA_AI_KV=%s -> KV type %s\n", kv_env, ap.kv_type_is_q4_0 < 0 ? "auto (unrecognised value)" : kv_env);
        }
        const char * k_env = getenv("ALPHA_AI_MTP_K");
        ap.mtp_k = k_env ? atoi(k_env) : 3;
        ap.verbose = getenv("ALPHA_AI_VERBOSE") ? 1 : 0;

        local_srv = std::make_unique<alphaserve::server>();
        local_srv->init_scheduler_only(vocab, n_slots);
        // prompts must reach the engine whole: no slot prefix reuse (AlphaInfer prefills the full prompt itself)
        local_srv->disable_slot_prefix_match();
        // the adapter buffers prompt chunks anyway; large chunks just mean fewer scheduler round trips
        alphaserve::g_serve_n_batch = 8192;
        // ...and the prompt goes to the engine whole: the slot chunk would default to n_ubatch (a llama.cpp
        // compute-buffer size that means nothing to AlphaInfer), and even chunks would cost extra scheduler round
        // trips and prefill calls. While another slot is generating the piece is still capped by
        // ALPHA_PREFILL_PIECE_DECODING (1024), so long prompts keep interleaving with decode.
        alphaserve::g_serve_prompt_chunk = 8192;
        // decode-only and prompt-only batches alternate with a decode time share (serve-scheduler ALPHA_DECODE_SHARE), so a
        // reply in progress keeps streaming while another slot's long prompt prefills in pieces (ai_engine_prefill_chunk).
        // Mixed batches stall the streaming reply for seconds; a 0.75 share keeps it near full speed with little
        // effect on the prefilling stream's TTFT.
        setenv("ALPHA_PREFILL_ALONE", "1", 0);
        setenv("ALPHA_DECODE_SHARE", "0.75", 0);
        // admission rejects prompts longer than a stream region (400 context_length_exceeded) instead of the engine
        // refusing the prefill and the client getting an empty 200
        if (ap.ctx_cells_per_stream > 0) alphaserve::g_serve_ctx_per_slot = (int) ap.ctx_cells_per_stream;
        local_srv->start();

        // ALPHA_AI_PAUSE_S=N: hold N seconds before AlphaInfer allocates anything, so nvidia-smi can read what this
        // side of the process holds on the GPU (it should be ~nothing: only the vocabulary is loaded)
        if (const char * ps = getenv("ALPHA_AI_PAUSE_S")) {
            fprintf(stderr, "[serve] ALPHA_AI_PAUSE_S: pausing %s s before ai_engine_create (pid %d)\n", ps, (int) getpid());
            std::this_thread::sleep_for(std::chrono::seconds(atoi(ps)));
        }
        ai_gpu = std::make_unique<alphaserve::ai_gpu_engine>();
        if (!ai_gpu->init(ap, vocab, local_srv->gpu_in_queue(), local_srv->gpu_out_queue(), 8192)) {
            fprintf(stderr, "[serve] AlphaInfer engine refused to start: %s\n", ai_gpu->last_error().c_str());
            return 1;
        }
        local_router = std::make_unique<alphaserve::response_router>();
        if (!chat_tmpls) chat_tmpls = common_chat_templates_init(model, "");
        fprintf(stderr, "[serve] AlphaInfer backend ready (%d streams, %lld cells/stream)\n",
                n_slots, (long long) ap.ctx_cells_per_stream);
    }
#endif

    if (mode == MODE_AUTO) {
        fprintf(stderr, "[serve] Auto mode: local for <=%d active, SGLang for >%d\n",
                auto_threshold, auto_threshold);
    }

    std::atomic<int> next_request_id{1};

    // HTTP server
    httplib::Server http;
    http.set_payload_max_length((size_t) 128 << 20);   // base64 images

    // ALPHA_SERVE_WARMUP: /health reports "loading" (503) until the startup warm-up request has finished
    static std::atomic<bool> g_warm_ready{true};

    // Health check
    http.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        if (!g_warm_ready.load()) {
            res.status = 503;
            res.set_content(json{{"status", "loading"}}.dump(), "application/json");
            return;
        }
        const char * mode_str = (mode == MODE_AUTO) ? "auto" : (mode == MODE_SGLANG) ? "sglang" : (mode == MODE_ALPHAINFER) ? "alphainfer" :
                                (mode == MODE_VLLM) ? "vllm" : "local";
        json body = {
            {"status", "ok"},
            {"backend", mode_str},
            {"intelligence", corpus_dir ? true : false},
        };
        if (local_srv) {
            body["slots_active"] = local_srv->active_slots();
            body["slots_total"] = n_slots;
        }
        if (mode == MODE_AUTO) {
            body["auto_threshold"] = auto_threshold;
        }
#ifdef ALPHA_HAS_ALPHAINFER
        // A graph-capture failure degrades to direct launches and returns rc == 0, so the engine loses speed while
        // every other signal still says success, and the log is not a reliable channel. capture_state is
        // process-sticky, so once any round has degraded the adapter fails every subsequent batch; without this
        // the server would sit in a failure loop while /health still answered "ok".
        // The payload is a contract (launch scripts check for "ok"), so capture state is an added field, and
        // "status" drops to "degraded" only when the AlphaInfer backend is live and degraded. The local backend
        // is untouched.
        if (auto * ai = alphaserve::ai_engine_live()) {
            body["capture"]       = ai->capture_state_str();   // "own" | "degraded" | "none"
            body["free_vram_mib"] = ai->free_vram_mib();
            body["mtp_mode"]      = ai->mtp_mode_str();   // "exact" | "fast" | "plain": fast is not token-identical
            body["mtp_k"]         = ai->mtp_k();
            if (getenv("AI_DFLASH2")) body["drafter"] = "dflash2";   // DFlash2 speculation (fast verify), not the model's MTP head
            if (ai->calibration_state() >= 0) {   // newer libalphainfer: the calibration signal for launchers
                body["ai_calibration_state"] = ai->calibration_state();
                body["ai_ctx_cells"]         = ai->ctx_cells();
            }
            if (ai->is_degraded()) {
                body["status"] = "degraded";   // deliberately not "ok": batches are failing
                body["error"]  = ai->last_error();
            }
        }
#endif
        res.set_content(body.dump(), "application/json");
    });

    // llama.cpp-compatible discovery endpoints. Clients (e.g. coding agents) size their requests from these, so the
    // context reported is per slot (the most one request can use), not the total -c across slots.
    auto ctx_per_slot = [&]() -> int {
        if (alphaserve::g_serve_ctx_per_slot > 0) return alphaserve::g_serve_ctx_per_slot;
        return n_slots > 0 ? n_ctx / n_slots : n_ctx;
    };
    http.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        json body = {
            {"default_generation_settings", json{{"n_ctx", ctx_per_slot()}}},
            {"total_slots", n_slots},
            {"model_path", model_path ? model_path : ""},
            {"n_ctx", ctx_per_slot()},
        };
        res.set_content(body.dump(), "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json body = {
            {"object", "list"},
            {"data", json::array({json{{"id", model_name}, {"object", "model"}, {"owned_by", "alphallama"},
                                       {"meta", json{{"n_ctx", ctx_per_slot()}}}}})},
        };
        res.set_content(body.dump(), "application/json");
    });

    // Corpus query (explicit retrieval API over the BM25 intelligence layer)
    // Body: {"query":str, "k":int=50, "span":bool=false, "union":bool=false}. Returns ranked chunks with
    // source/provenance and text or a query-aware span (offset). union = merge a rare-bigram-first pass.
    http.Post("/api/corpus/query", [&](const httplib::Request & req, httplib::Response & res) {
        if (!intel_ctx.bm25 || intel_ctx.bm25->N == 0) {
            res.status = 503;
            res.set_content("{\"error\":\"no corpus loaded (start with --corpus-dir DIR)\"}", "application/json");
            return;
        }
        std::string q; int k = 50; bool want_span = false, want_union = false;
        // span_chars: length of the returned query-anchored span. The default of 330 keeps about a quarter of
        // the text at ~2% cosine loss, fitting ~1500 spans in a 128K window. Opt-in via "span": true.
        int span_chars = 330;
        try { json b = json::parse(req.body); q = b.value("query", ""); k = b.value("k", 50);
              want_span = b.value("span", false); want_union = b.value("union", false);
              span_chars = b.value("span_chars", 330); } catch (...) {}
        if (q.empty()) { res.status = 400; res.set_content("{\"error\":\"missing 'query'\"}", "application/json"); return; }
        if (span_chars < 32)   span_chars = 32;
        if (span_chars > 4096) span_chars = 4096;
        std::vector<int> ids = intel_ctx.bm25->top_k(q, k);
        if (want_union) {
            std::unordered_set<int> seen(ids.begin(), ids.end());
            for (int cid : intel_ctx.bm25->top_k(q, k, /*bf_override=*/1)) if (seen.insert(cid).second) ids.push_back(cid);
        }
        json results = json::array();
        for (int cid : ids) {
            // Provenance: full source (path@byteoffset), clean parent doc path, parsed byte offset
            // into the source file, and article attribution when the corpus carries article headers.
            std::string source = intel_ctx.bm25->get_source(cid);
            json r = { {"chunk_id", cid}, {"source", source},
                       {"doc", intel_ctx.bm25->get_parent_source(cid)} };
            { size_t at = source.rfind('@');
              if (at != std::string::npos && at + 1 < source.size()) {
                  bool alldig = true;
                  for (size_t i = at + 1; i < source.size(); i++) if (!isdigit((unsigned char) source[i])) { alldig = false; break; }
                  if (alldig) r["byte_offset"] = (long long) atoll(source.c_str() + at + 1);
              } }
            auto am = intel_ctx.bm25->get_article_meta(cid);
            if (!am.title.empty() || !am.article_id.empty()) {
                r["article_title"] = am.title;
                r["article_id"]    = am.article_id;
                r["article_pos"]   = cid - intel_ctx.bm25->get_parent_id(cid); // sub-chunk index within article
            }
            std::string text = intel_ctx.bm25->get_text(cid);
            if (want_span) {
                std::string ql; for (char c : q) ql += (char) std::tolower((unsigned char) c);
                size_t pos = std::string::npos;
                { std::string tl; tl.reserve(text.size()); for (char c : text) tl += (char) std::tolower((unsigned char) c);
                  std::istringstream is(ql); std::string w;
                  while (is >> w) if (w.size() > 2) { size_t p = tl.find(w); if (p != std::string::npos && p < pos) pos = p; } }
                int lead = span_chars / 4; if (lead > 120) lead = 120;   // keep some context before the hit
                size_t start = (pos == std::string::npos) ? 0 : (pos > (size_t) lead ? pos - lead : 0);
                r["span_start"] = (int) start;                            // byte offset within the chunk text
                r["span_chars"] = span_chars;
                r["span"] = text.substr(start, (size_t) span_chars);
            } else {
                r["text"] = text;
            }
            results.push_back(r);
        }
        json out = { {"query", q}, {"count", (int) ids.size()}, {"union", want_union}, {"results", results} };
        res.set_content(out.dump(), "application/json");
    });

#ifdef ALPHA_DELTANET
    // DeltaNet: load a prebuilt recurrent state at runtime (parity with alphallama's /api/deltanet/load).
    http.Post("/api/deltanet/load", [&](const httplib::Request & req, httplib::Response & res) {
        std::string path;
        try { json b = json::parse(req.body); path = b.value("path", ""); } catch (...) {}
        if (path.empty()) { res.status = 400; res.set_content("{\"error\":\"missing 'path'\"}", "application/json"); return; }
        if (!ctx)        { res.status = 503; res.set_content("{\"error\":\"no local context\"}", "application/json"); return; }
        uint64_t toks = alpha_deltanet_state_load(path, ctx);
        json out = {{"loaded_tokens", toks}, {"path", path}};
        res.set_content(out.dump(), "application/json");
    });
#endif

    // Chat completions handler
    // Works with both backends. Intelligence augmentation runs first.
    http.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        // ALPHA_SERVE_TIMING=1: one line per request showing where the time goes before the scheduler sees it
        static const bool serve_timing = getenv("ALPHA_SERVE_TIMING") != nullptr;
        const int64_t t_in = ggml_time_us();
        int64_t t_parsed = 0, t_tmpl = 0, t_tok = 0;
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }

        int req_id = next_request_id.fetch_add(1);
        bool stream = body.value("stream", false);
        t_parsed = ggml_time_us();

        // Intelligence augmentation
        if (intel_ctx.bm25 && intel_ctx.bm25->N > 0) {
            alphaserve::intelligence_augment(intel_ctx, body);
        }

        // Backend routing
        // Auto mode: route based on current load
        bool route_to_remote = is_remote_proxy(mode);
        if (mode == MODE_AUTO && local_srv && remote_backend) {
            int active = local_srv->active_slots();
            route_to_remote = (active >= auto_threshold);
            if (route_to_remote) {
                fprintf(stderr, "[serve] auto: %d active >= %d → %s\n", active, auto_threshold,
                        remote_backend->name());
            }
        }

        if (route_to_remote && remote_backend) {
            // Remote proxy backend (SGLang/vLLM): forward request
            if (!stream) {
                auto resp = remote_backend->complete(body);
                if (resp.contains("error")) {
                    res.status = 502;
                }
                // Post-process
                alphaserve::intelligence_post_process(intel_ctx, body, resp);
                res.set_content(resp.dump(), "application/json");
            } else {
                res.set_header("Cache-Control", "no-cache");
                res.set_chunked_content_provider("text/event-stream",
                    [&, body, req_id](size_t, httplib::DataSink & sink) -> bool {
                        json req_copy = body;
                        auto resp = remote_backend->stream(req_copy,
                            [&](const std::string & sse_line) {
                                sink.write(sse_line.c_str(), sse_line.size());
                            });
                        alphaserve::intelligence_post_process(intel_ctx, body, resp);
                        sink.done();
                        return true;
                    });
            }
        } else if (local_srv && local_router && vocab) {
            // Local backend: AlphaServe pipeline.
            // Build the prompt from messages via the model's own Jinja chat template, with the same library
            // (common_chat) the standard alphallama server uses. This renders `tools`, honors
            // `chat_template_kwargs.enable_thinking`, and handles assistant `tool_calls` + `role:"tool"`
            // + `content:null` messages. `cchat.format` is kept so the response can be parsed back into
            // tool_calls.
            std::string prompt;
            common_chat_params cchat;   // format defaults to CONTENT_ONLY (used to parse the response)
            bool req_tools = false;     // the request offered tools: the reply may carry tool calls to parse
            alpha_mm_bitmaps mm_bitmaps;   // image input, in prompt order
            if (body.contains("messages") && body["messages"].is_array() && chat_tmpls) {
                try {
                    common_chat_templates_inputs inputs;
                    nlohmann::json msgs_in = body.at("messages");
                    if (alpha_mm_has_images(msgs_in)) {
                        if (!alphaserve::g_serve_mtmd) {
                            res.status = 400;
                            res.set_content("{\"error\":{\"message\":\"this server has no image input: start it with --mmproj "
                                            "<projector.gguf> (built-in engine)\",\"type\":\"invalid_request_error\","
                                            "\"code\":\"images_not_supported\"}}", "application/json");
                            return;
                        }
                        std::string mm_err;
                        if (!alpha_mm_extract(msgs_in, mm_bitmaps, mm_err)) {
                            res.status = 400;
                            res.set_content(nlohmann::json{ { "error", { { "message", mm_err }, { "type", "invalid_request_error" } } } }.dump(),
                                            "application/json");
                            return;
                        }
                    }
                    inputs.messages = common_chat_msgs_parse_oaicompat(msgs_in);
                    if (body.contains("tools") && body.at("tools").is_array())
                        inputs.tools = common_chat_tools_parse_oaicompat(body.at("tools"));
                    inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(body.value("tool_choice", std::string("auto")));
                    // OpenAI's default is true. common_chat's is false, and the parser built from a false flag accepts
                    // exactly one <tool_call> block, so a reply with several calls would fail to parse and come back
                    // as plain content.
                    inputs.parallel_tool_calls = body.value("parallel_tool_calls", true);
                    inputs.use_jinja = true;
                    inputs.add_generation_prompt = true;
                    bool enable_thinking = true;
                    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
                        for (auto & el : body["chat_template_kwargs"].items()) {
                            inputs.chat_template_kwargs[el.key()] = el.value().is_string()
                                ? el.value().get<std::string>() : el.value().dump();
                        }
                        auto & ck = body["chat_template_kwargs"];
                        if (ck.contains("enable_thinking")) {
                            if (ck["enable_thinking"].is_boolean())      enable_thinking = ck["enable_thinking"].get<bool>();
                            else if (ck["enable_thinking"].is_string())  enable_thinking = (ck["enable_thinking"].get<std::string>() != "false");
                        }
                    }
                    if (body.contains("think") && body["think"].is_boolean()) enable_thinking = body["think"].get<bool>();
                    inputs.enable_thinking = enable_thinking;
                    cchat  = common_chat_templates_apply(chat_tmpls.get(), inputs);
                    prompt = cchat.prompt;
                    // tool_choice=required/none returns a parser that assumes a grammar forces the shape, so it
                    // does not extract the <tool_call> the model still emits. AlphaServe has no grammar, so parse
                    // with the auto apply's format+parser instead (the prompt above keeps the requested tool_choice).
                    // The same holds for parallel calls: with no grammar the model may emit several calls whatever the
                    // request said, so parse with a parser that accepts any number of them; dropping calls the model
                    // actually made would return a wrong answer.
                    if ((inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_AUTO || !inputs.parallel_tool_calls) && !inputs.tools.empty()) {
                        common_chat_templates_inputs ai = inputs;
                        ai.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
                        ai.parallel_tool_calls = true;
                        common_chat_params ac = common_chat_templates_apply(chat_tmpls.get(), ai);
                        cchat.format = ac.format;
                        cchat.parser = ac.parser;
                        cchat.grammar_triggers = ac.grammar_triggers;   // where a call starts (streaming hold-back)
                    }
                    req_tools = !inputs.tools.empty();
                } catch (const std::exception & e) {
                    res.status = 400;
                    res.set_content(std::string("{\"error\":{\"message\":\"chat template error: ")
                                    + e.what() + "\",\"type\":\"invalid_request_error\"}}", "application/json");
                    return;
                }
            } else if (body.contains("messages") && body["messages"].is_array()) {
                // Fallback (no chat template loaded): plain concat, null-safe.
                for (auto & msg : body["messages"]) {
                    std::string role = msg.value("role", std::string("user"));
                    std::string content = (msg.contains("content") && msg["content"].is_string())
                                          ? msg["content"].get<std::string>() : std::string();
                    if (role == "system") prompt += "System: " + content + "\n";
                    else if (role == "user") prompt += "User: " + content + "\n";
                    else if (role == "assistant") prompt += "Assistant: " + content + "\n";
                }
                prompt += "Assistant:";
            }

            t_tmpl = ggml_time_us();
            std::vector<llama_token> tokens;
            std::shared_ptr<alphaserve::mm_prompt> mmp;
            if (!mm_bitmaps.v.empty()) {
                static std::mutex mm_tok_mtx;   // mtmd is not thread-safe; HTTP threads tokenize one at a time
                std::lock_guard<std::mutex> lk(mm_tok_mtx);
                mmp = std::make_shared<alphaserve::mm_prompt>();
                mmp->chunks = mtmd_input_chunks_init();
                mtmd_input_text txt{ prompt.c_str(), /*add_special=*/true, /*parse_special=*/true };
                std::vector<const mtmd_bitmap *> bms(mm_bitmaps.v.begin(), mm_bitmaps.v.end());
                const int32_t r = mtmd_tokenize(alphaserve::g_serve_mtmd, mmp->chunks, &txt, bms.data(), bms.size());
                if (r != 0) {
                    res.status = 400;
                    res.set_content(std::string("{\"error\":{\"message\":\"image tokenization failed (") +
                                    (r == 1 ? "image count does not match the media markers" : "image preprocessing error") +
                                    ")\",\"type\":\"invalid_request_error\"}}", "application/json");
                    return;
                }
                mmp->n_images = (int) bms.size();
                mmp->n_tokens = (int) mtmd_helper_get_n_tokens(mmp->chunks);
                mmp->n_pos    = (int) mtmd_helper_get_n_pos(mmp->chunks);
                tokens.reserve(mmp->n_tokens);
                for (size_t ci = 0; ci < mtmd_input_chunks_size(mmp->chunks); ci++) {
                    const mtmd_input_chunk * ck = mtmd_input_chunks_get(mmp->chunks, ci);
                    if (mtmd_input_chunk_get_type(ck) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                        size_t nt = 0;
                        const llama_token * tt = mtmd_input_chunk_get_tokens_text(ck, &nt);
                        tokens.insert(tokens.end(), tt, tt + nt);
                    } else {
                        tokens.insert(tokens.end(), mtmd_input_chunk_get_n_tokens(ck), (llama_token) alphaserve::MM_PLACEHOLDER_TOKEN);
                    }
                }
            } else {
                tokens = common_tokenize(vocab, prompt, true, true);
            }
            t_tok = ggml_time_us();
            const int n_tok_timing = (int) tokens.size();

            // Context overflow: reject with a clear error instead of silently returning an empty
            // completion. Past n_ctx the engine hits "decode error at pos 0" and finishes the
            // request with 0 tokens, indistinguishable from empty generation to the client.
            if (alphaserve::g_serve_ctx_per_slot > 0 &&
                (int) tokens.size() >= alphaserve::g_serve_ctx_per_slot) {
                res.status = 400;
                res.set_content(std::string("{\"error\":{\"message\":\"prompt is ") +
                    std::to_string(tokens.size()) + " tokens, exceeding the context size of " +
                    std::to_string(alphaserve::g_serve_ctx_per_slot) +
                    " (raise -c / n_ctx, or shorten the prompt)\",\"type\":\"invalid_request_error\","
                    "\"code\":\"context_length_exceeded\"}}", "application/json");
                return;
            }

            alphaserve::InferenceRequest ireq;
            ireq.request_id = req_id;
            ireq.prompt_tokens = std::move(tokens);
            ireq.mm = mmp;
            alpha_reasoning_budget(body, prompt, vocab, ireq);
            ireq.max_tokens = body.value("max_tokens", 200);
            ireq.temperature = body.value("temperature", 0.7f);
            ireq.top_k = body.value("top_k", 40);
            ireq.top_p = body.value("top_p", 0.95f);
            ireq.seed = body.value("seed", -1);
            #ifdef ALPHA_HAS_ALPHAINFER
            // AlphaInfer is greedy-only: absent sampling fields mean greedy (not the 0.7 / 40 / 0.95 defaults above), and an
            // explicit temperature > 0 is a clear 400 up front, never a truncated 200 from a refused batch.
            if (mode == MODE_ALPHAINFER) {
                if (!body.contains("temperature") || body["temperature"].is_null()) ireq.temperature = 0.0f;
                // ALPHA_AI_LENIENT=1 (opt-in): a temperature > 0 is served greedy instead of refused, and response_format
                // json_object is accepted without enforcement (the output is whatever greedy text the prompt asks for), for
                // clients that always send them. json_schema, grammar and logit_bias still refuse.
                static const bool ai_lenient = getenv("ALPHA_AI_LENIENT") && atoi(getenv("ALPHA_AI_LENIENT")) != 0;
                if (ai_lenient && ireq.temperature > 0.0f) { ireq.temperature = 0.0f; res.set_header("X-Alpha-Sampling", "greedy"); }
                if (ireq.temperature > 0.0f) {
                    res.status = 400;
                    res.set_content("{\"error\":{\"message\":\"this server (AlphaInfer backend) is greedy-only: send temperature 0 "
                                    "or omit it\",\"type\":\"invalid_request_error\",\"code\":\"unsupported_sampling\"}}", "application/json");
                    return;
                }
                // constrained decoding needs a logits mask, which the engine refuses (ENOTSUP): say so instead of
                // returning unconstrained text
                const bool constrained = body.contains("grammar") || body.contains("json_schema") || body.contains("logit_bias") ||
                    (body.contains("response_format") && body["response_format"].is_object() &&
                     body["response_format"].value("type", std::string("text")) != "text" &&
                     !(ai_lenient && body["response_format"].value("type", std::string("text")) == "json_object"));
                if (constrained) {
                    res.status = 400;
                    res.set_content("{\"error\":{\"message\":\"this server (AlphaInfer backend) does not support grammar, "
                                    "json_schema / response_format or logit_bias\",\"type\":\"invalid_request_error\","
                                    "\"code\":\"unsupported_constraint\"}}", "application/json");
                    return;
                }
                ireq.top_k = 1; ireq.top_p = 1.0f;   // irrelevant at temperature 0
            }
            #endif
            ireq.stream = stream;
            if (body.contains("stop")) {
                if (body["stop"].is_array()) {
                    for (auto & s : body["stop"]) ireq.stop_words.push_back(s.get<std::string>());
                } else if (body["stop"].is_string()) {
                    ireq.stop_words.push_back(body["stop"].get<std::string>());
                }
            }

            auto pr = local_router->create(req_id, stream);
            if (!local_srv->submit(std::move(ireq))) {
                local_router->remove(req_id);
                res.status = 503;
                res.set_content("{\"error\":\"server busy\"}", "application/json");
                return;
            }
            if (serve_timing) {
                const int64_t t_sub = ggml_time_us();
                fprintf(stderr, "[serve-timing] req %d: parse %.2f ms | template %.2f ms | tokenize %.2f ms (%zu chars -> %d tok) | submit %.2f ms | total pre-engine %.2f ms\n",
                        req_id, (t_parsed - t_in) / 1e3, (t_tmpl - t_parsed) / 1e3, (t_tok - t_tmpl) / 1e3, prompt.size(),
                        n_tok_timing, (t_sub - t_tok) / 1e3, (t_sub - t_in) / 1e3);
            }

            if (!stream) {
                // Poll the socket while waiting: a dead client must cancel the decode, not just
                // abandon it; otherwise the slot burns its full generation budget for nobody and
                // the next client times out in the queue.
                const int wres = pr->wait_done_poll(request_idle_timeout(), request_prefill_timeout(),
                    std::chrono::milliseconds(1000),
                    [&req]() { return !req.is_connection_closed(); });
                if (wres != 1) {
                    local_srv->cancel(req_id);   // free the slot + KV pages
                    local_router->remove(req_id);
                    if (wres == -1) return;      // peer gone: nothing to send
                    res.status = 504;
                    res.set_content("{\"error\":{\"message\":\"request timed out with no progress "
                                    "(decode stalled or not scheduled)\",\"type\":\"timeout\"}}",
                                    "application/json");
                    return;
                }
                // Parse the model's native output into content + tool_calls via the same common_chat
                // parser the standard server uses (format from the template above). Malformed output
                // degrades to plain content, never a 500.
                json message;
                std::string finish_reason;
                parse_chat_reply(cchat, pr->text, req_id, message, finish_reason);
                if (pr->error) finish_reason = "error";   // decode error: the reply is truncated, so say so
                else if (pr->length && finish_reason == "stop") finish_reason = "length";   // cut by max_tokens / the context
                auto resp = json{
                    {"id",      "chatcmpl-" + std::to_string(req_id)},
                    {"object",  "chat.completion"},
                    {"created", (int64_t)time(nullptr)},
                    {"model",   model_name},
                    {"choices", json::array({json{
                        {"index", 0},
                        {"message", message},
                        {"finish_reason", finish_reason},
                    }})},
                    {"usage", json{
                        {"prompt_tokens", n_tok_timing},
                        {"completion_tokens", pr->n_tokens},
                        {"total_tokens", n_tok_timing + pr->n_tokens},
                    }},
                };
                res.set_content(resp.dump(-1, ' ', false, json::error_handler_t::replace)   /* generated text may end mid-UTF-8: never throw */, "application/json");
            } else {
                res.set_header("Cache-Control", "no-cache");
                // Tool calls in a stream: text before a call streams as it comes; from the first tool-call trigger
                // word on (e.g. "<tool_call>") the output is held, and at the end the whole reply goes through the same
                // parser as the non-streaming path and the calls are sent as a tool_calls delta. Without a plain-word
                // trigger (e.g. tool_choice=required: the whole reply is calls) everything is held.
                std::vector<std::string> trig_words;
                bool trig_hold_all = false;
                if (req_tools) {
                    for (const auto & t : cchat.grammar_triggers) {
                        if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD && !t.value.empty()) trig_words.push_back(t.value);
                        else trig_hold_all = true;
                    }
                    if (trig_words.empty()) trig_hold_all = true;
                }
                res.set_chunked_content_provider("text/event-stream",
                    [&, pr, req_id, cchat, req_tools, trig_words, trig_hold_all, n_prompt = n_tok_timing,
                     usage_in_stream = (body.contains("stream_options") && body["stream_options"].is_object() &&
                                        body["stream_options"].value("include_usage", false))](size_t, httplib::DataSink & sink) -> bool {
                        std::string chunk, utf8_tail;
                        int wr;
                        // Reasoning-channel strip for streaming (matches the non-streaming path).
                        // state: 0 = buffering head (undecided), 1 = inside a channel preamble,
                        // 2 = pass-through. Buffer only while the head could be a channel opener,
                        // so normal answers stream with no added delay.
                        // A <think> preamble streams as it is generated, as reasoning_content deltas (the
                        // non-streaming reply carries it in reasoning_content too); content starts after </think>.
                        int strip_state = getenv("ALPHA_NO_STRIP_THINK") ? 2 : 0;
                        std::string head;
                        bool think_stream = false;  // state 1 on a <think> opener: stream the reasoning
                        size_t think_pos = 0;       // head offset of the first reasoning byte not yet sent
                        bool peer_dead = false;    // sink.write fails once the client is gone
                        std::string sent;          // content already sent (tool mode: not sent again after the parse)
                        auto emit = [&](const std::string & t) {
                            if (t.empty() || peer_dead) return;
                            sent += t;
                            auto sse = make_sse_chunk(req_id, t, model_name, true, false);
                            if (!sink.write(sse.c_str(), sse.size())) peer_dead = true;
                        };
                        auto emit_reasoning = [&](const std::string & t) {
                            if (t.empty() || peer_dead) return;
                            auto sse = make_sse_reasoning_chunk(req_id, t, model_name);
                            if (!sink.write(sse.c_str(), sse.size())) peer_dead = true;
                        };
                        // the unsent reasoning of a <think> preamble that never closed (the reply hit its limit)
                        auto flush_think = [&]() {
                            if (think_pos < head.size()) emit_reasoning(head.substr(think_pos));
                            head.clear(); think_pos = 0;
                        };
                        auto ltrim = [](const std::string & x) {
                            size_t b = x.find_first_not_of(" \t\r\n");
                            return b == std::string::npos ? std::string() : x.substr(b);
                        };
                        static const char * const openers[] = { "<|channel>", "<|thought", "<think>" };
                        auto feed = [&](const std::string & out) {   // reasoning strip, then emit
                            if (out.empty()) return;
                            if (strip_state == 2) { emit(out); return; }
                            head += out;
                            if (strip_state == 0) {            // classify the head
                                std::string t = ltrim(head);
                                if (t.empty()) return;         // only whitespace so far
                                bool is_chan = false, maybe = false;
                                for (auto o : openers) {
                                    size_t ol = std::string(o).size();
                                    if (t.size() >= ol) { if (t.compare(0, ol, o) == 0) is_chan = true; }
                                    else if (std::string(o).compare(0, t.size(), t) == 0) maybe = true;
                                }
                                if (is_chan) {
                                    strip_state = 1;
                                    if (t.compare(0, 7, "<think>") == 0) { think_stream = true; think_pos = head.find("<think>") + 7; }
                                }
                                else if (maybe && head.size() < 256) { return; }  // still could be an opener
                                else { strip_state = 2; emit(head); head.clear(); return; }  // normal content
                            }
                            if (strip_state == 1 && think_stream) {   // stream the reasoning, hold back a possible close
                                static const std::string close = "</think>";
                                size_t c = head.find(close, think_pos);
                                if (c != std::string::npos) {
                                    emit_reasoning(head.substr(think_pos, c - think_pos));
                                    std::string rest = ltrim(head.substr(c + close.size()));
                                    head.clear(); think_pos = 0; strip_state = 2;
                                    emit(rest);
                                    return;
                                }
                                size_t keep = 0;               // longest tail of head that starts the close tag
                                for (size_t k = std::min(close.size() - 1, head.size() - think_pos); k > 0; k--)
                                    if (head.compare(head.size() - k, k, close, 0, k) == 0) { keep = k; break; }
                                size_t end = head.size() - keep;
                                if (end > think_pos) { emit_reasoning(head.substr(think_pos, end - think_pos)); think_pos = end; }
                                return;
                            }
                            if (strip_state == 1) {            // buffer the preamble until its close
                                if (head.rfind("<channel|>") != std::string::npos ||
                                    head.rfind("</think>")   != std::string::npos ||
                                    head.size() > 16384) {
                                    std::string body = strip_reasoning_channel(head);
                                    head.clear(); strip_state = 2;
                                    emit(body);
                                }
                            }
                        };
                        // tool mode: raw = the whole reply (for the parser); pend = text not yet streamed that may be
                        // the start of a trigger word; held = a trigger was seen (or no word trigger): stream no more
                        std::string raw, pend;
                        bool held = req_tools && trig_hold_all;
                        while (!peer_dead && (wr = pr->wait_chunk_for(chunk, request_idle_timeout(), request_prefill_timeout())) == 1) {
                            std::string piece = utf8_tail; piece += chunk;
                            size_t good = utf8_complete_prefix_len(piece);
                            utf8_tail = piece.substr(good);   // hold incomplete trailing bytes for next piece
                            if (good == 0) continue;           // no complete codepoint yet
                            std::string out = piece.substr(0, good);
                            if (req_tools) {
                                raw += out;
                                if (held) continue;
                                pend += out;
                                size_t cut = std::string::npos;
                                for (const auto & w : trig_words) cut = std::min(cut, pend.find(w));
                                if (cut != std::string::npos) {
                                    out = pend.substr(0, cut); pend.clear(); held = true;
                                } else {                       // keep back a tail that could begin a trigger word
                                    size_t keep = 0;
                                    for (const auto & w : trig_words)
                                        for (size_t k = std::min(w.size() - 1, pend.size()); k > keep; k--)
                                            if (pend.compare(pend.size() - k, k, w, 0, k) == 0) { keep = k; break; }
                                    out = pend.substr(0, pend.size() - keep);
                                    pend.erase(0, pend.size() - keep);
                                }
                            }
                            feed(out);
                        }
                        if (wr < 0 || peer_dead) {   // stalled or client gone: cancel the decode, free slot + thread
                            local_srv->cancel(req_id);
                            local_router->remove(req_id);
                        }
                        const char * finish = pr->error ? "error" : pr->length ? "length" : "stop";   // truncated: error / limit
                        if (req_tools) {
                            raw += utf8_tail; utf8_tail.clear();
                            if (!held) feed(pend);             // no call started: the kept-back tail is plain text
                            else {
                                json message; std::string fr;
                                parse_chat_reply(cchat, raw, req_id, message, fr);
                                if (message.contains("tool_calls")) {
                                    // content the parser found beyond what already streamed (e.g. text after the calls)
                                    std::string c = (message.contains("content") && message["content"].is_string())
                                                    ? message["content"].get<std::string>() : std::string();
                                    std::string s = ltrim(sent), ct = ltrim(c);
                                    while (!s.empty() && isspace((unsigned char) s.back())) s.pop_back();
                                    if (ct.size() > s.size() && ct.compare(0, s.size(), s) == 0) feed(ct.substr(s.size()));
                                    if (strip_state != 2 && think_stream) flush_think();
                                    else if (strip_state != 2 && !head.empty()) { emit(strip_reasoning_channel(head)); head.clear(); }
                                    if (!peer_dead) {
                                        auto tsse = make_sse_tool_calls_chunk(req_id, message["tool_calls"], model_name);
                                        if (!sink.write(tsse.c_str(), tsse.size())) peer_dead = true;
                                    }
                                    finish = "tool_calls";
                                } else {
                                    // no call parsed: send the held text as plain text. It starts at the first
                                    // trigger word (nothing earlier can hold one: the stream cut there)
                                    size_t cut = std::string::npos;
                                    for (const auto & w : trig_words) cut = std::min(cut, raw.find(w));
                                    feed(raw.substr((trig_hold_all || cut == std::string::npos) ? 0 : cut));
                                }
                            }
                        }
                        if (strip_state != 2 && think_stream) flush_think();   // reasoning cut off by the limit
                        else if (strip_state != 2 && !head.empty()) emit(strip_reasoning_channel(head));  // flush buffered head
                        if (!utf8_tail.empty()) {              // flush any leftover (dump sanitizes it)
                            auto tsse = make_sse_chunk(req_id, utf8_tail, model_name, true, false);
                            sink.write(tsse.c_str(), tsse.size());
                        }
                        auto sse = make_sse_chunk(req_id, "", model_name, true, true, finish);
                        sink.write(sse.c_str(), sse.size());
                        if (usage_in_stream) {   // OpenAI stream_options.include_usage: one chunk with empty choices + usage
                            json u = {
                                {"id", "chatcmpl-" + std::to_string(req_id)}, {"object", "chat.completion.chunk"},
                                {"created", (int64_t) time(nullptr)}, {"model", model_name}, {"choices", json::array()},
                                {"usage", {{"prompt_tokens", n_prompt}, {"completion_tokens", pr->n_tokens},
                                           {"total_tokens", n_prompt + pr->n_tokens}}},
                            };
                            std::string us = "data: " + u.dump() + "\n\n";
                            sink.write(us.c_str(), us.size());
                        }
                        std::string done = "data: [DONE]\n\n";
                        sink.write(done.c_str(), done.size());
                        sink.done();
                        return true;
                    });
            }
        }
        // FlashInfer mode routes through local_srv (scheduler-batched), so it uses the same
        // handler as the local backend
    });

    // POST /v1/completions
    http.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        if (is_remote_proxy(mode)) {
            // Forward directly to the remote proxy (SGLang/vLLM)
            json body;
            try { body = json::parse(req.body); } catch (...) {
                res.status = 400;
                res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
                return;
            }
            auto resp = remote_backend->complete(body);
            res.set_content(resp.dump(-1, ' ', false, json::error_handler_t::replace)   /* generated text may end mid-UTF-8: never throw */, "application/json");
            return;
        }

        // Local backend
        json body;
        try { body = json::parse(req.body); } catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }

        std::string prompt = body.value("prompt", "");
        int req_id = next_request_id.fetch_add(1);
        bool stream = body.value("stream", false);

        auto tokens = common_tokenize(vocab, prompt, true, true);

        // Context overflow: clear error instead of a silent empty completion (see above).
        if (alphaserve::g_serve_ctx_per_slot > 0 &&
            (int) tokens.size() >= alphaserve::g_serve_ctx_per_slot) {
            res.status = 400;
            res.set_content(std::string("{\"error\":{\"message\":\"prompt is ") +
                std::to_string(tokens.size()) + " tokens, exceeding the context size of " +
                std::to_string(alphaserve::g_serve_ctx_per_slot) +
                " (raise -c / n_ctx, or shorten the prompt)\",\"type\":\"invalid_request_error\","
                "\"code\":\"context_length_exceeded\"}}", "application/json");
            return;
        }

        alphaserve::InferenceRequest ireq;
        ireq.request_id = req_id;
        ireq.prompt_tokens = std::move(tokens);
        alpha_reasoning_budget(body, prompt, vocab, ireq);
        ireq.max_tokens = body.value("max_tokens", 200);
        ireq.temperature = body.value("temperature", 0.7f);
        ireq.top_k = body.value("top_k", 40);
        ireq.top_p = body.value("top_p", 0.95f);
        ireq.seed = body.value("seed", -1);
        #ifdef ALPHA_HAS_ALPHAINFER
        // AlphaInfer is greedy-only: absent sampling fields mean greedy (not the 0.7 / 40 / 0.95 defaults above), and an
        // explicit temperature > 0 is a clear 400 up front, never a truncated 200 from a refused batch.
        if (mode == MODE_ALPHAINFER) {
            if (!body.contains("temperature") || body["temperature"].is_null()) ireq.temperature = 0.0f;
            // ALPHA_AI_LENIENT=1 (opt-in): a temperature > 0 is served greedy instead of refused, and response_format
            // json_object is accepted without enforcement (the output is whatever greedy text the prompt asks for), for
            // clients that always send them. json_schema, grammar and logit_bias still refuse.
            static const bool ai_lenient = getenv("ALPHA_AI_LENIENT") && atoi(getenv("ALPHA_AI_LENIENT")) != 0;
            if (ai_lenient && ireq.temperature > 0.0f) { ireq.temperature = 0.0f; res.set_header("X-Alpha-Sampling", "greedy"); }
            if (ireq.temperature > 0.0f) {
                res.status = 400;
                res.set_content("{\"error\":{\"message\":\"this server (AlphaInfer backend) is greedy-only: send temperature 0 "
                                "or omit it\",\"type\":\"invalid_request_error\",\"code\":\"unsupported_sampling\"}}", "application/json");
                return;
            }
            // constrained decoding needs a logits mask, which the engine refuses (ENOTSUP): say so instead of
            // returning unconstrained text
            const bool constrained = body.contains("grammar") || body.contains("json_schema") || body.contains("logit_bias") ||
                (body.contains("response_format") && body["response_format"].is_object() &&
                 body["response_format"].value("type", std::string("text")) != "text" &&
                 !(ai_lenient && body["response_format"].value("type", std::string("text")) == "json_object"));
            if (constrained) {
                res.status = 400;
                res.set_content("{\"error\":{\"message\":\"this server (AlphaInfer backend) does not support grammar, "
                                "json_schema / response_format or logit_bias\",\"type\":\"invalid_request_error\","
                                "\"code\":\"unsupported_constraint\"}}", "application/json");
                return;
            }
            ireq.top_k = 1; ireq.top_p = 1.0f;   // irrelevant at temperature 0
        }
        #endif
        ireq.stream = stream;

        auto pr = local_router->create(req_id, stream);
        if (!local_srv->submit(std::move(ireq))) {
            local_router->remove(req_id);
            res.status = 503;
            res.set_content("{\"error\":\"server busy\"}", "application/json");
            return;
        }

        if (!stream) {
            // Poll the socket while waiting: a dead client cancels the decode
            const int wres = pr->wait_done_poll(request_idle_timeout(), request_prefill_timeout(),
                std::chrono::milliseconds(1000),
                [&req]() { return !req.is_connection_closed(); });
            if (wres != 1) {
                local_srv->cancel(req_id);
                local_router->remove(req_id);
                if (wres == -1) return;      // peer gone: nothing to send
                res.status = 504;
                res.set_content("{\"error\":{\"message\":\"request timed out with no progress "
                                "(decode stalled or not scheduled)\",\"type\":\"timeout\"}}",
                                "application/json");
                return;
            }
            auto resp = make_completion_response(req_id, pr->text, pr->n_tokens, model_name, pr->error ? "error" : pr->length ? "length" : "stop");
            res.set_content(resp.dump(-1, ' ', false, json::error_handler_t::replace)   /* generated text may end mid-UTF-8: never throw */, "application/json");
        } else {
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider("text/event-stream",
                [&, pr, req_id](size_t, httplib::DataSink & sink) -> bool {
                    std::string chunk, utf8_tail;
                    int wr;
                    bool peer_dead = false;
                    while (!peer_dead && (wr = pr->wait_chunk_for(chunk, request_idle_timeout(), request_prefill_timeout())) == 1) {
                        std::string piece = utf8_tail; piece += chunk;
                        size_t good = utf8_complete_prefix_len(piece);
                        utf8_tail = piece.substr(good);   // hold incomplete trailing bytes for next piece
                        if (good == 0) continue;           // no complete codepoint yet
                        auto sse = make_sse_chunk(req_id, piece.substr(0, good), model_name, false, false);
                        if (!sink.write(sse.c_str(), sse.size())) peer_dead = true;
                    }
                    if (wr < 0 || peer_dead) {   // stalled or client gone: cancel decode, free slot + thread
                        local_srv->cancel(req_id);
                        local_router->remove(req_id);
                    }
                    if (!utf8_tail.empty()) {              // flush any leftover (dump sanitizes it)
                        auto tsse = make_sse_chunk(req_id, utf8_tail, model_name, false, false);
                        sink.write(tsse.c_str(), tsse.size());
                    }
                    auto sse = make_sse_chunk(req_id, "", model_name, false, true, pr->error ? "error" : pr->length ? "length" : "stop");
                    sink.write(sse.c_str(), sse.size());
                    std::string done = "data: [DONE]\n\n";
                    sink.write(done.c_str(), done.size());
                    sink.done();
                    return true;
                });
        }
    });

    // ALPHA_ENV_AUDIT: last point before blocking on listen(). Every backend shared object is loaded
    // by now, so a flag string missing here is missing for good. Warns on any ALPHA_* var that is set
    // but that no loaded code can read (silent no-op).
    // Announce the source identity that produced this binary: a number is only reproducible if the
    // source behind it can be named.
    {
        const char * bid = llama_commit();
        fprintf(stderr, "[serve] build %s (%s)\n", bid, llama_build_target());
        if (strncmp(bid, "src:", 4) == 0) {
            fprintf(stderr, "[serve] WARNING: no git identity -- this binary is named by a SOURCE CONTENT HASH.\n");
            fprintf(stderr, "[serve]          Numbers from it cannot be reproduced from a commit. Archive the\n");
            fprintf(stderr, "[serve]          source alongside them, or deploy from git rather than a tarball.\n");
        } else if (strstr(bid, "-dirty:") != nullptr) {
            fprintf(stderr, "[serve] WARNING: built from a DIRTY worktree -- the commit alone does NOT describe\n");
            fprintf(stderr, "[serve]          this binary; the -dirty: suffix is a hash of the actual source.\n");
        }
    }

    alpha_env_audit();

    // Start HTTP server in background thread
    std::thread http_thread([&]() {
        fprintf(stderr, "[serve] HTTP server listening on %s:%d\n", host, port);
        http.listen(host, port);
    });

    // ALPHA_SERVE_WARMUP (default on in local mode; =0 disables): one short greedy request through the real engine
    // before /health turns ok. It registers the device-resolved expert layers, runs the profile fill, allocates the
    // MTP head's KV and captures the decode / draft / verify CUDA graphs, all of which would otherwise land inside
    // the first client request and slow it down. Nothing here is shaped by any client prompt.
    std::thread warm_thread;
    {
        const char * we = getenv("ALPHA_SERVE_WARMUP");
        const bool warm = (we ? atoi(we) != 0 : true) && !is_remote_proxy(mode) && mode != MODE_FLASHINFER;
        if (warm) {
            g_warm_ready = false;
            ggml_cuda_dre_tier_defer();   // ALPHA_DRE_TIER: filled after the warm-up request, not inside it
            warm_thread = std::thread([port]() {
                const int64_t t0 = ggml_time_us();
                httplib::Client cli("127.0.0.1", port);
                cli.set_read_timeout(600, 0);
                cli.set_write_timeout(60, 0);
                for (int i = 0; i < 200 && !cli.Get("/health"); i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                const char * wt = getenv("ALPHA_SERVE_WARMUP_TOKENS");
                json req = {
                    {"messages", json::array({ json{{"role", "user"}, {"content",
                        "Write a short Python function that merges two sorted lists, then explain its complexity."}} })},
                    {"max_tokens", wt ? atoi(wt) : 64},
                    {"temperature", 0},
                    {"chat_template_kwargs", json{{"enable_thinking", false}}},
                };
                auto r = cli.Post("/v1/chat/completions", req.dump(), "application/json");
                // ALPHA_DRE_TIER: fill the RAM hot tier now; nothing decodes and /health is still "loading" until it is done
                ggml_cuda_dre_tier_start();
                for (int i = 0; i < 6000 && ggml_cuda_dre_tier_busy(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(200));
                fprintf(stderr, "[serve] warm-up request %s in %.0f ms -- /health is ok from now on\n",
                        r && r->status == 200 ? "done" : "FAILED", (ggml_time_us() - t0) / 1000.0);
                g_warm_ready = true;
            });
        }
    }

    const char * mode_label = (mode == MODE_FLASHINFER) ? "flashinfer" : (mode == MODE_ALPHAINFER) ? "alphainfer" : (mode == MODE_AUTO) ? "auto" :
                              (mode == MODE_SGLANG) ? "sglang" : (mode == MODE_VLLM) ? "vllm" : "local";
    fprintf(stderr, "[serve] Backend: %s | Intelligence: %s | Port: %d\n",
            mode_label, corpus_dir ? "enabled" : "disabled", port);

    // Main loop
    if (is_remote_proxy(mode)) {
        // Pure proxy mode: main thread idle
        fprintf(stderr, "[serve] %s mode — main thread idle\n", mode_label);
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else if (mode == MODE_ALPHAINFER) {
#ifdef ALPHA_HAS_ALPHAINFER
        // AlphaInfer mode: main thread drives the adapter (one engine round per batch) + event routing
        fprintf(stderr, "[serve] alphainfer mode — engine rounds on main thread\n");
        while (g_running.load()) {
            bool did_work = ai_gpu->process_one();
            alphaserve::TokenEvent event;
            while (local_srv->poll_event(event)) {
                local_router->route_event(event);
            }
            if (!did_work) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
#endif
    } else if (mode == MODE_FLASHINFER) {
#ifdef ALPHA_HAS_FLASHINFER
        // FlashInfer mode: main thread drives fi_gpu_engine + event routing
        fprintf(stderr, "[serve] flashinfer mode — batched GPU on main thread\n");
        while (g_running.load()) {
            bool did_work = fi_gpu->process_one();

            alphaserve::TokenEvent event;
            while (local_srv->poll_event(event)) {
                local_router->route_event(event);
            }

            if (!did_work) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
#endif
    } else {
        // Local or auto mode: main thread drives GPU decode + event routing
        fprintf(stderr, "[serve] %s mode — GPU decode on main thread\n", mode_label);

        // Decode watchdog: a separate thread that logs if no token is produced for a while
        // while requests are pending, so a genuine stall (or a hung llama_decode freezing this
        // main thread) is visible in the log and distinguishable from a merely slow decode.
        // Log-only; never touches the GPU.
        std::atomic<int64_t> last_token_us{ ggml_time_us() };
        const int watchdog_s = [] {
            const char * e = getenv("ALPHA_DECODE_WATCHDOG_S");
            return e ? atoi(e) : 60;   // 0 disables
        }();
        std::thread watchdog;
        if (watchdog_s > 0) {
            watchdog = std::thread([&] {
                bool warned = false;
                while (g_running.load()) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    int pending = local_router->pending_count();
                    double idle_s = (ggml_time_us() - last_token_us.load()) / 1e6;
                    if (pending > 0 && idle_s > watchdog_s) {
                        if (!warned) {
                            fprintf(stderr, "[watchdog] decode: no token produced in %.0fs with %d "
                                    "request(s) pending — possible stall\n", idle_s, pending);
                            warned = true;
                        }
                    } else {
                        warned = false;
                    }
                }
            });
        }

        {
            static const bool pre_flush = [] { const char * e = getenv("ALPHA_SERVE_PRE_BATCH_FLUSH"); return !e || atoi(e) != 0; }();
            if (pre_flush) {
                local_srv->set_pre_batch_hook([&]() {
                    alphaserve::TokenEvent ev;
                    bool any = false;
                    while (local_srv->poll_event(ev)) { local_router->route_event(ev); any = true; }
                    if (any) last_token_us.store(ggml_time_us());
                });
            }
        }
        while (g_running.load()) {
            bool did_work = local_srv->step();

            alphaserve::TokenEvent event;
            bool routed = false;
            while (local_srv->poll_event(event)) {
                local_router->route_event(event);
                routed = true;
            }
            if (routed) last_token_us.store(ggml_time_us());

            if (!did_work) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
        if (watchdog.joinable()) watchdog.join();
    }

    // Shutdown
    fprintf(stderr, "\n[serve] Shutting down...\n");
    http.stop();
    if (warm_thread.joinable()) warm_thread.join();
    if (http_thread.joinable()) http_thread.join();

    if (local_srv) local_srv->stop();
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);

    fprintf(stderr, "[serve] Done.\n");
    return 0;
}
