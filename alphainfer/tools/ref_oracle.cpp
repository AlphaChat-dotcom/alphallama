// SPDX-License-Identifier: Apache-2.0
// ai-ref: the reference forward of this repository's libllama, used as the correctness oracle for alphainfer.
//
//   ai-ref -m M.gguf --tokens-in prompt.i32 -n 64 --gen-out ref_gen.i32
//   ai-ref -m M.gguf --tokens-in prompt.i32 --score ref.logits
//
// Same chunking as alphainfer (n_batch = n_ubatch = --chunk), flash attention on, f16 KV,
// all layers on the GPU, llama's MoE ubatch-grow disabled so chunk boundaries match.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// --dump-dir: capture llama's per-layer tensors by name for the layer-diff tool.
static std::string g_dump_dir;
static bool dump_cb(ggml_tensor * t, bool ask, void *) {
    // qwen35 names, then qwen4exp's (src/models/qwen4exp.cpp: hc_init, ffn_out-N, l_last-N, result_norm)
    static const char * keep[] = {"attn_norm-", "linear_attn_out-", "attn_output-", "attn_residual-",
                                  "ffn_moe_out-", "ffn_shexp_gated-", "ffn_out-", "l_out-", "result_norm",
                                  "hc_init", "l_last-"};
    const std::string name = ggml_get_name(t);
    bool want = false;
    for (const char * k : keep) {
        if (name.rfind(k, 0) == 0) {
            want = true;
        }
    }
    if (ask) {
        return want;
    }
    if (!want || t->type != GGML_TYPE_F32) {
        return true;
    }
    if (!ggml_is_contiguous(t)) {
        return true;
    }
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    const int64_t rows = ggml_nelements(t) / t->ne[0];
    FILE * f = fopen((g_dump_dir + "/" + name + ".bin").c_str(), "wb");
    int64_t hdr[2] = {t->ne[0], rows};
    fwrite(hdr, 8, 2, f);
    fwrite(buf.data(), 4, buf.size(), f);
    fclose(f);
    return true;
}

static std::vector<int32_t> read_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> t(raw.size() / 4);
    memcpy(t.data(), raw.data(), t.size() * 4);
    return t;
}

int main(int argc, char ** argv) {
    setenv("ALPHA_MOE_UBATCH_PREFILL", "0", 0);

    std::string model, tokens_in, gen_out, score_out;
    int64_t n_gen = 64, ctx = 8192, chunk = 512;
    bool    cpu_experts = false;
    std::string prompt_file, tokens_out;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "-m") model = v;
        else if (a == "--tokens-in") tokens_in = v;
        else if (a == "-n") n_gen = atoll(v.c_str());
        else if (a == "--gen-out") gen_out = v;
        else if (a == "--score") score_out = v;
        else if (a == "--ctx") ctx = atoll(v.c_str());
        else if (a == "--chunk") chunk = atoll(v.c_str());
        else if (a == "--dump-dir") g_dump_dir = v;
        else if (a == "--cpu-experts") cpu_experts = atoi(v.c_str()) != 0;
        else if (a == "--prompt-file") prompt_file = v;   // tokenize with the model's own vocab (no chat template)
        else if (a == "--tokens-out") tokens_out = v;     // write the prompt ids (i32), e.g. for alphainfer --tokens-in
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    std::vector<int32_t> ids;
    if (prompt_file.empty()) ids = read_tokens(tokens_in);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    // --cpu-experts 1: routed experts and the PLE table stay in CPU memory (llama tensor buffer
    // overrides, as `-ot exps=CPU`) so a model larger than the card still runs; ggml computes those on the CPU backend
    static llama_model_tensor_buft_override ovr[3];
    if (cpu_experts) {
        ovr[0] = { "_exps\\.", ggml_backend_cpu_buffer_type() };
        ovr[1] = { "per_layer_token_embd", ggml_backend_cpu_buffer_type() };
        ovr[2] = { nullptr, nullptr };
        mp.tensor_buft_overrides = ovr;
    }
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (!m) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(m);
    if (!prompt_file.empty()) {
        std::ifstream pf(prompt_file, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
        ids.resize(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), ids.data(), (int32_t) ids.size(),
                                     /*add_special=*/ true, /*parse_special=*/ true);
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        ids.resize((size_t) n);
        if (!tokens_out.empty()) {
            FILE * tf = fopen(tokens_out.c_str(), "wb");
            fwrite(ids.data(), 4, ids.size(), tf);
            fclose(tf);
        }
        fprintf(stderr, "[ai-ref] prompt: %zu tokens\n", ids.size());
    }
    const int32_t V = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) ctx;
    cp.n_batch         = (uint32_t) chunk;
    cp.n_ubatch        = (uint32_t) chunk;
    cp.n_seq_max       = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k          = GGML_TYPE_F16;
    cp.type_v          = GGML_TYPE_F16;
    if (!g_dump_dir.empty()) {
        cp.cb_eval           = dump_cb;
        cp.cb_eval_user_data = nullptr;
    }
    llama_context * lc = llama_init_from_model(m, cp);
    if (!lc) { fprintf(stderr, "context init failed\n"); return 1; }

    const int64_t n = (int64_t) ids.size();
    using clk = std::chrono::steady_clock;

    if (!score_out.empty()) {
        FILE * f = fopen(score_out.c_str(), "wb");
        int64_t hn = n, hv = V;
        fwrite(&hn, 8, 1, f);
        fwrite(&hv, 8, 1, f);
        const auto t0 = clk::now();
        for (int64_t off = 0; off < n; off += chunk) {
            const int64_t T = std::min(chunk, n - off);
            llama_batch b = llama_batch_init((int32_t) T, 0, 1);
            for (int64_t i = 0; i < T; ++i) {
                b.token[i] = ids[(size_t) (off + i)];
                b.pos[i] = (llama_pos) (off + i);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = 0;
                b.logits[i] = 1;
            }
            b.n_tokens = (int32_t) T;
            if (llama_decode(lc, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            for (int64_t i = 0; i < T; ++i) {
                fwrite(llama_get_logits_ith(lc, (int32_t) i), 4, (size_t) V, f);
            }
            llama_batch_free(b);
        }
        fclose(f);
        double s = std::chrono::duration<double>(clk::now() - t0).count();
        fprintf(stderr, "[ref] scored %lld positions in %.2f s\n", (long long) n, s);
        return 0;
    }

    std::vector<int32_t> out;
    const auto t0 = clk::now();
    for (int64_t off = 0; off < n; off += chunk) {
        const int64_t T = std::min(chunk, n - off);
        llama_batch b = llama_batch_init((int32_t) T, 0, 1);
        for (int64_t i = 0; i < T; ++i) {
            b.token[i] = ids[(size_t) (off + i)];
            b.pos[i] = (llama_pos) (off + i);
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (off + i == n - 1);
        }
        b.n_tokens = (int32_t) T;
        if (llama_decode(lc, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        llama_batch_free(b);
    }
    auto argmax = [&](int32_t i) {
        const float * l = llama_get_logits_ith(lc, i);
        return (int32_t) (std::max_element(l, l + V) - l);
    };
    int32_t tok = argmax(-1);
    const auto t1 = clk::now();
    int64_t pos = n;
    for (int64_t i = 0; i < n_gen; ++i) {
        out.push_back(tok);
        if (llama_vocab_is_eog(vocab, tok) || i + 1 == n_gen) break;
        llama_batch b = llama_batch_get_one(&tok, 1);
        (void) pos;
        if (llama_decode(lc, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
        tok = argmax(-1);
        ++pos;
    }
    const auto t2 = clk::now();
    double ps = std::chrono::duration<double>(t1 - t0).count();
    double ds = std::chrono::duration<double>(t2 - t1).count();
    fprintf(stderr, "[ref] prefill %lld tok %.3f s = %.1f tok/s | decode %zu tok %.3f s = %.2f tok/s\n",
            (long long) n, ps, n / ps, out.size(), ds, out.size() > 1 ? (out.size() - 1) / ds : 0.0);
    if (!gen_out.empty()) {
        std::ofstream f(gen_out, std::ios::binary);
        f.write((const char *) out.data(), (std::streamsize) (out.size() * 4));
    }
    llama_free(lc);
    llama_model_free(m);
    return 0;
}
