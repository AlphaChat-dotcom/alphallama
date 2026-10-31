#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"
#include "server-help.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <atomic>
#include <clocale>
#include <cerrno>
#include <cstring>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if !defined(_WIN32)
#include <unistd.h> // execv / access for the MoE auto-route to AlphaServe
#include <climits>
#include <cstdlib>
#if defined(__APPLE__)
#include <mach-o/dyld.h> // _NSGetExecutablePath
#endif

// Directory of this executable, for finding alphallama-serve beside it. argv[0] alone is not enough: started from
// PATH it has no directory, and through a symlink it names the link's directory. Resolves the real file (Linux
// /proc/self/exe, macOS _NSGetExecutablePath + realpath), else argv[0]'s directory.
static std::string alpha_self_dir(const char * argv0) {
    std::string self;
#if defined(__linux__)
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        self = buf;
    }
#elif defined(__APPLE__)
    char     buf[PATH_MAX];
    uint32_t sz = sizeof(buf);
    char     real[PATH_MAX];
    if (_NSGetExecutablePath(buf, &sz) == 0 && realpath(buf, real)) {
        self = real;
    }
#endif
    if (self.empty()) {
        self = argv0 ? argv0 : "";
    }
    const size_t slash = self.find_last_of('/');
    return slash == std::string::npos ? std::string(".") : self.substr(0, slash);
}
#endif

#include "../../src/alpha_hw_planner.h" // alpha_hw_plan_create: detect MoE-won't-fit for auto-route
#include "../../src/alpha_env_audit.h" // alpha_env_audit: warn on ALPHA_* vars nothing reads

#if defined(_WIN32)
#include <windows.h>

// Directory of this executable (GetModuleFileNameW), for finding alphallama-serve.exe beside it.
static std::wstring alpha_self_dir_w() {
    wchar_t buf[4 * MAX_PATH];
    const DWORD cap = (DWORD) (sizeof(buf) / sizeof(buf[0]));
    const DWORD n = GetModuleFileNameW(nullptr, buf, cap);
    const std::wstring self = (n > 0 && n < cap) ? std::wstring(buf, n) : std::wstring();
    const size_t slash = self.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring(L".") : self.substr(0, slash);
}

// Windows has no execv: run alphallama-serve.exe with this process's own arguments (the command line after the
// program name, so the user's quoting is kept), wait for it, and hand back its exit code. Ctrl+C reaches the child
// through the shared console; this process ignores it while it waits. False if the child could not be started.
static bool alpha_win_run_serve(const std::wstring & serve, const std::wstring & extra_args, int & rc) {
    const wchar_t * cl = GetCommandLineW();
    // skip the program name: a quoted one runs to the closing quote, an unquoted one to the first space or tab
    if (*cl == L'"') {
        cl++;
        while (*cl && *cl != L'"') cl++;
        if (*cl) cl++;
    } else {
        while (*cl && *cl != L' ' && *cl != L'\t') cl++;
    }
    std::wstring cmd = L"\"" + serve + L"\"" + cl + extra_args;
    // pass this process's standard handles on explicitly: without STARTF_USESTDHANDLES a child of a process whose
    // output is redirected (a log file, a service) writes to the console instead, and its output is lost
    STARTUPINFOW si = {};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    for (HANDLE h : { si.hStdInput, si.hStdOutput, si.hStdError }) {
        if (h && h != INVALID_HANDLE_VALUE) {
            SetHandleInformation(h, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT);
        }
    }
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(serve.c_str(), &cmd[0], nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        return false;
    }
    SetConsoleCtrlHandler(nullptr, TRUE);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    rc = (int) code;
    return true;
}
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // Engine identity banner (the two engines have near-identical names)
    fprintf(stderr,
        "==================================================================\n"
        " AlphaLlama  ·  llama-server engine  (tools/server)\n"
        "   Corpus/RAG + dev endpoints.  MoE decode = LUT_DIRECT.\n"
        "   A MoE model that does not fit VRAM auto-routes to 'alphallama-serve'\n"
        "   (AlphaServe: fused-infer / ALL_GPU, ~8x faster). ALPHA_NO_AUTOROUTE=1\n"
        "   forces this engine.\n"
        "==================================================================\n");
    fprintf(stderr, "%s (%s), build %s\n", alphallama_version_name(), alphallama_version(), llama_build_info());

    // own arguments required by this example
    common_params params;

    common_init();

    // Reconstruct command line for error assistant context
    std::string cmdline;
    for (int i = 0; i < argc; ++i) { cmdline += argv[i]; cmdline += " "; }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        fprintf(stderr, "%s\n", alpha_error_explain(
            "Failed to parse command line arguments",
            cmdline).c_str());
        return 1;
    }

    // No model given: print how to start a model instead of starting llama.cpp's router mode, which would listen
    // with no model loaded. The router still starts when it is asked for (--models-dir / --models-preset).
    if (params.model.path.empty() && params.model.hf_repo.empty() && params.model.url.empty() &&
        params.models_dir.empty() && params.models_preset.empty()) {
        fprintf(stderr,
            "\nUsage:\n"
            "  alphallama -m model.gguf                 OpenAI-compatible server, retrieval and dev endpoints\n"
            "  alphallama-serve -m model.gguf           the fast serving engine (MoE offload, AlphaInfer, multi-user)\n"
            "  alphallama -hf <user>/<repo>[:quant]     download a GGUF from Hugging Face and serve it\n"
            "  alphallama --models-dir DIR              llama.cpp router mode: serve the models in DIR on demand\n"
            "  alphallama --help                        all options\n\n");
        return 1;
    }

    // Auto-route to AlphaServe for MoE models that don't fit VRAM.
    // This engine serves a MoE that spills VRAM through the slow LUT_DIRECT offload path, while
    // AlphaServe's fused-infer / pool-direct path is several times faster on the same model. When
    // the HW planner classifies the run as MoE offload (LUT_DIRECT / SELF_SPEC / CPU_MOE) and the
    // invocation uses none of the corpus/RAG/dev features this engine owns, exec the
    // alphallama-serve binary next to this one. Its arg parser ignores unknown flags, so argv
    // passes through, plus --host/--port when absent (on Windows, which has no exec, it runs as a child and its exit code is returned).
    // ALPHA_NO_AUTOROUTE=1 forces this engine.
    if (!getenv("ALPHA_NO_AUTOROUTE") && !params.model.path.empty()) {
        bool corpus_or_dev = false;
        for (int i = 1; i < argc; i++) {
            const char * a = argv[i];
            if (strstr(a, "corpus") || strcmp(a, "--context") == 0 ||
                strcmp(a, "--tools") == 0 || strcmp(a, "--mode") == 0) {
                corpus_or_dev = true; // retrieval/dev job: this engine owns it, don't route
                break;
            }
        }
        // alphallama-serve does not implement API keys or TLS: handing off would silently drop them
        // (params reflects both the flags and their environment variables; the scan below also catches an empty key file)
        bool auth_or_tls = !params.api_keys.empty() || !params.ssl_file_key.empty() || !params.ssl_file_cert.empty();
        for (const char * e : { "LLAMA_API_KEY", "LLAMA_ARG_API_KEY", "LLAMA_ARG_API_KEY_FILE",
                                "LLAMA_ARG_SSL_KEY_FILE", "LLAMA_ARG_SSL_CERT_FILE" }) {
            if (getenv(e)) {
                auth_or_tls = true;
            }
        }
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--api-key") == 0 || strcmp(argv[i], "--api-key-file") == 0 ||
                strcmp(argv[i], "--ssl-key-file") == 0 || strcmp(argv[i], "--ssl-cert-file") == 0) {
                auth_or_tls = true;
            }
        }
        if (!corpus_or_dev) {
            alpha_hw_plan plan = alpha_hw_plan_create(params.model.path.c_str(), -1);
            const bool moe_offload =
                plan.strategy == ALPHA_HW_LUT_DIRECT ||
                plan.strategy == ALPHA_HW_SELF_SPEC  ||
                plan.strategy == ALPHA_HW_CPU_MOE;
            if (moe_offload) {
#if defined(_WIN32)
                const std::wstring serve = alpha_self_dir_w() + L"\\alphallama-serve.exe";
                const bool have_serve = GetFileAttributesW(serve.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
                const std::string dir = alpha_self_dir(argv[0]);
                std::string serve = dir + "/alphallama-serve";
                const bool have_serve = access(serve.c_str(), X_OK) == 0;
#endif
                if (have_serve && auth_or_tls) {
                    fprintf(stderr, "[alphallama] not handing off to alphallama-serve: it would not apply "
                                    "--api-key / --api-key-file / --ssl-key-file / --ssl-cert-file; serving here.\n");
                } else if (have_serve) {
                    fprintf(stderr,
                        "==================================================================\n"
                        "[alphallama] MoE model does not fit VRAM (HW planner strategy: %s).\n"
                        "  Serving it here would use the slow LUT_DIRECT offload path.\n"
                        "  Auto-routing to AlphaServe (fused-infer / pool-direct) for ~8x\n"
                        "  higher decode throughput.  Set ALPHA_NO_AUTOROUTE=1 to force this\n"
                        "  (llama-server / corpus) engine instead.\n"
                        "==================================================================\n",
                        plan.strategy_name ? plan.strategy_name : "MoE-offload");
                    fflush(stderr);
                    // alphallama-serve defaults to 0.0.0.0:8080; pass this engine's bind address and port
                    // (loopback / 18080 unless set) when the command line does not name them
                    bool have_host = false;
                    bool have_port = false;
                    for (int i = 1; i < argc; i++) {
                        have_host = have_host || strcmp(argv[i], "--host") == 0;
                        have_port = have_port || strcmp(argv[i], "--port") == 0;
                    }
                    std::vector<std::string> extra;
                    if (!have_host) {
                        extra.push_back("--host");
                        extra.push_back(params.hostname);
                    }
                    if (!have_port) {
                        extra.push_back("--port");
                        extra.push_back(std::to_string(params.port));
                    }
#if defined(_WIN32)
                    std::wstring extra_w;
                    for (const auto & s : extra) {
                        extra_w += L" \"" + std::wstring(s.begin(), s.end()) + L"\"";
                    }
                    int rc = 0;
                    if (alpha_win_run_serve(serve, extra_w, rc)) {
                        return rc;
                    }
                    fprintf(stderr, "[alphallama] could not start alphallama-serve.exe (error %lu) — continuing in this engine.\n",
                            (unsigned long) GetLastError());
#else
                    std::vector<char *> serve_argv;
                    serve_argv.push_back(const_cast<char *>(serve.c_str()));
                    for (int i = 1; i < argc; i++) {
                        serve_argv.push_back(argv[i]);
                    }
                    for (auto & s : extra) {
                        serve_argv.push_back(&s[0]);
                    }
                    serve_argv.push_back(nullptr);
                    execv(serve.c_str(), serve_argv.data());
                    // execv only returns on failure; fall through and run in-process.
                    fprintf(stderr, "[alphallama] exec of '%s' failed (%s) — continuing in this engine.\n",
                            serve.c_str(), strerror(errno));
#endif
                }
            }
        }
    }

    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params.n_batch, params.n_ubatch);
        LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("build_info: %s\n", llama_build_info());
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    routes.init_kv_chunk_routes();
    server_tools tools;

    bool is_router_server = params.model.path.empty();
    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to initialize router models: %s\n", __func__, e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.get_api_show                = models_routes->proxy_get;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.post("/api/show",                 ex_wrapper(routes.get_api_show));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/api/tags",                 ex_wrapper(routes.get_models)); // ollama specific endpoint. public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat",                 ex_wrapper(routes.post_ollama_chat));       // ollama native format
    ctx_http.post("/api/generate",             ex_wrapper(routes.post_ollama_generate));   // ollama text completion
    ctx_http.get ("/api/ps",                   ex_wrapper(routes.get_ollama_ps));          // ollama running models
    ctx_http.post("/api/pull",                 ex_wrapper(routes.post_ollama_pull));       // ollama model pull (stub)
    ctx_http.post("/api/delete",               ex_wrapper(routes.delete_ollama_model));    // ollama model delete (stub, uses POST)
    ctx_http.post("/api/copy",                 ex_wrapper(routes.post_ollama_copy));       // ollama model copy (stub)
    ctx_http.post("/api/embed",                ex_wrapper(routes.post_ollama_embed));      // ollama embed (stub)
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    // KV-chunk endpoints for long-context system
    ctx_http.post("/api/prefill_chunk",              ex_wrapper(routes.post_prefill_chunk));
    ctx_http.post("/api/activate_chunks",             ex_wrapper(routes.post_activate_chunks));
    ctx_http.post("/api/decode_step",                 ex_wrapper(routes.post_decode_step));
    ctx_http.post("/api/kv_cache/extract_centroids",  ex_wrapper(routes.post_kv_extract_centroids));
    ctx_http.post("/api/kv_cache/save",               ex_wrapper(routes.post_kv_save));
    ctx_http.post("/api/kv_cache/load",               ex_wrapper(routes.post_kv_load));
    ctx_http.post("/api/corpus/index",                ex_wrapper(routes.post_kv_index));
    ctx_http.post("/api/corpus/finalize",             ex_wrapper(routes.post_corpus_finalize));
    ctx_http.post("/api/corpus/query",                ex_wrapper(routes.post_kv_route));
    ctx_http.post("/api/kv_chunk/index",              ex_wrapper(routes.post_kv_index));  // legacy alias
    ctx_http.post("/api/kv_chunk/route",              ex_wrapper(routes.post_kv_route));  // legacy alias
#ifdef ALPHA_INTERNAL
    ctx_http.post("/api/expert_routing",              ex_wrapper(routes.post_expert_routing));
#endif
    ctx_http.post("/api/oracle",                       ex_wrapper(routes.post_oracle));
#ifdef ALPHA_DELTANET
    ctx_http.post("/api/deltanet/load",                ex_wrapper(routes.post_deltanet_load));
#endif
    ctx_http.post("/api/records/index",                ex_wrapper(routes.post_records_index));
    ctx_http.post("/api/records/aggregate",            ex_wrapper(routes.post_records_aggregate));
    ctx_http.post("/api/dense/index",                  ex_wrapper(routes.post_dense_index));
    ctx_http.post("/api/dense/search",                 ex_wrapper(routes.post_dense_search));
    ctx_http.post("/api/dense/build_ann",              ex_wrapper(routes.post_dense_build_ann));
    // KV pool endpoints
    ctx_http.get ("/api/kv_pool/stats",               ex_wrapper(routes.get_kv_pool_stats));
#ifdef ALPHA_INTERNAL
    ctx_http.post("/api/kv_pool/save",                ex_wrapper(routes.post_kv_pool_save));
    ctx_http.post("/api/kv_pool/load",                ex_wrapper(routes.post_kv_pool_load));
#endif
    // Batch inference
    ctx_http.post("/api/batch",                       ex_wrapper(routes.post_batch_completions));

    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));
    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    }
    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        corpus_tool_context empty_corpus;
        tools.setup(params.server_tools, empty_corpus);
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        LOG_INF("%s: starting router server, no model will be loaded in this process\n", __func__);

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
            return 1;
        }

        // load the model
        if (params.model.path.empty()) {
            clean_up();
            LOG_ERR("%s: no model path specified (use --model <path>)\n", __func__);
            fprintf(stderr, "%s\n", alpha_error_explain(
                "No model path specified. Use --model <path> to load a GGUF model.",
                cmdline).c_str());
            return 1;
        }
        LOG_INF("%s: loading model\n", __func__);

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            LOG_ERR("%s: exiting due to model loading error\n", __func__);
            fprintf(stderr, "%s\n", alpha_error_explain(
                "Failed to load model. Check that the file exists and is a valid GGUF.",
                "",
                std::to_string(params.n_gpu_layers) + " GPU layers").c_str());
            return 1;
        }

        routes.update_meta(ctx_server);

        // Re-register tools with corpus context now that model + corpus are loaded
        if (!params.server_tools.empty()) {
            tools.corpus_dir = params.corpus_dir;
            tools.setup(params.server_tools, ctx_server.get_corpus_tool_context());
            routes.tools_ptr = &tools;
        }

        ctx_http.is_ready.store(true);

        LOG_INF("%s: model loaded\n", __func__);

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        LOG_INF("%s: router server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
        LOG_INF("%s: NOTE: router mode is experimental\n", __func__);
        LOG_INF("%s:       it is not recommended to use this mode in untrusted environments\n", __func__);
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        LOG_INF("%s: server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
        LOG_INF("%s: starting the main loop...\n", __func__);
        // Everything (incl. lazily dlopen-ed backends) is loaded by now, so
        // a flag string absent here is absent for good.
        alpha_env_audit();

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            monitor_thread = server_models::setup_child_server(shutdown_handler);
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        if (params.alpha_mode == ALPHA_MODE_PIPELINE) {
            LOG_INF("Starting async pipeline loop (--mode pipeline)\n");
            ctx_server.start_pipeline_loop();
        } else {
            ctx_server.start_loop();
        }

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            llama_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
