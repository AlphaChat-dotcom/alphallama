// ai-api-gen: greedy generation through the C API only (api/ai_engine.h), the calls AlphaServe's adapter makes:
// create -> per request { prefill (or prefill_chunk pieces) -> decode_mtp / decode rounds -> reset_stream } -> destroy.
// Writes each request's tokens (the first token included, as ai-q4x-run --gen-out) so a served stream can be checked
// against the CLI's.
//
//   ai-api-gen -m model.gguf --tokens-in p.i32[,q.i32,...] [-n 256] [--ctx 4096] [--slots N|-1] [--mtp-k K] [--requests R]
//              [--chunk-feed C] [--stop a,b] [--gen-out prefix]   -> prefix.<r>.i32 per request
#include "api/ai_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::string model, tokens_in, gen_out, chat_turn;
    int         n_gen = 256, slots = -1, mtp_k = 0, requests = 1, chunk_feed = 0;
    int64_t     ctx = 4096;
    std::vector<int32_t> stop;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "-m") model = v;
        else if (a == "--tokens-in") tokens_in = v;
        else if (a == "-n") n_gen = atoi(v.c_str());
        else if (a == "--ctx") ctx = atoll(v.c_str());
        else if (a == "--slots") slots = atoi(v.c_str());
        else if (a == "--mtp-k") mtp_k = atoi(v.c_str());
        else if (a == "--requests") requests = atoi(v.c_str());
        else if (a == "--chunk-feed") chunk_feed = atoi(v.c_str());
        else if (a == "--gen-out") gen_out = v;
        else if (a == "--chat-turn") chat_turn = v;   // request r > 0: previous prompt + its answer + these tokens
        else if (a == "--stop") {
            for (const char * q = v.c_str(); *q;) {
                char *     e  = nullptr;
                const long id = strtol(q, &e, 10);
                if (e == q) break;
                stop.push_back((int32_t) id);
                q = *e == ',' ? e + 1 : e;
            }
        } else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    // --tokens-in a.i32[,b.i32,...]: request r prefills prompt r mod n (several first-seen prompts on one engine, e.g. a
    // warm-up and then the timed ones, as a long-lived server sees them)
    std::vector<std::vector<int32_t>> prompts;
    for (size_t at = 0; at <= tokens_in.size();) {
        size_t c = tokens_in.find(',', at);
        if (c == std::string::npos) c = tokens_in.size();
        std::ifstream     tf(tokens_in.substr(at, c - at), std::ios::binary);
        std::vector<char> raw((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
        std::vector<int32_t> pr(raw.size() / 4);
        memcpy(pr.data(), raw.data(), pr.size() * 4);
        prompts.push_back(std::move(pr));
        at = c + 1;
    }
    std::vector<int32_t> prompt = prompts.empty() ? std::vector<int32_t>() : prompts[0];
    if (model.empty() || prompt.empty()) { fprintf(stderr, "need -m and a non-empty --tokens-in\n"); return 2; }
    std::vector<int32_t> turn;
    if (!chat_turn.empty()) {
        std::ifstream     cf(chat_turn, std::ios::binary);
        std::vector<char> cr((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        turn.resize(cr.size() / 4);
        memcpy(turn.data(), cr.data(), turn.size() * 4);
    }

    ai_engine_params p{};
    p.model_path           = model.c_str();
    p.ctx_cells_per_stream = ctx;
    p.n_streams            = 1;
    p.expert_slots         = slots;
    p.mtp_k                = mtp_k;
    p.verbose              = 1;
    const char * err = nullptr;
    const auto   tc  = std::chrono::steady_clock::now();
    ai_engine *  e   = ai_engine_create(&p, &err);
    if (!e) { fprintf(stderr, "[api-gen] create failed: %s\n", err ? err : "?"); return 1; }
    int k = 0, exact = 0;
    ai_engine_mtp_info(e, &k, &exact);
    const int cap = ai_engine_max_tokens_per_round(e);
    fprintf(stderr, "[api-gen] created in %.1f s: has_mtp %d, mtp_k %d, exact %d, round tokens <= %d, ctx %lld, capture %d, "
                    "calibration %d, free VRAM %lld MiB\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count(), ai_engine_has_mtp(e), k, exact, cap,
            (long long) ai_engine_ctx_cells(e), (int) ai_engine_capture_state(e), ai_engine_calibration_state(e),
            (long long) ai_engine_free_vram_mib(e));
    auto is_stop = [&](int32_t t) { return std::find(stop.begin(), stop.end(), t) != stop.end(); };
    std::vector<int32_t> ot((size_t) cap), os((size_t) cap);
    int rc_all = 0;
    for (int r = 0; r < requests; ++r) {
        if (prompts.size() > 1) prompt = prompts[(size_t) r % prompts.size()];
        const auto tp    = std::chrono::steady_clock::now();
        int32_t    first = -1;
        int        rc    = AI_ENGINE_OK;
        if (chunk_feed > 0) {
            for (size_t off = 0; off < prompt.size() && rc == AI_ENGINE_OK; off += (size_t) chunk_feed) {
                const int  nn   = (int) std::min(prompt.size() - off, (size_t) chunk_feed);
                const bool last = off + (size_t) nn >= prompt.size();
                rc = ai_engine_prefill_chunk(e, 0, prompt.data() + off, nn, last ? 1 : 0, &first);
            }
        } else {
            rc = ai_engine_prefill(e, 0, prompt.data(), (int) prompt.size(), &first);
        }
        if (rc != AI_ENGINE_OK) { fprintf(stderr, "[api-gen] prefill rc %d: %s\n", rc, ai_engine_last_error(e)); rc_all = 1; break; }
        const double pf = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
        std::vector<int32_t> out{ first };
        int32_t pending = first;
        int32_t pos     = (int32_t) prompt.size();
        int     rounds  = 0;
        bool    stopped = is_stop(first);
        const auto tg = std::chrono::steady_clock::now();
        while ((int) out.size() < n_gen && !stopped) {
            const int32_t sid = 0;
            int           n   = 0;
            if (k > 0) rc = ai_engine_decode_mtp(e, &pending, &pos, &sid, 1, ot.data(), os.data(), cap, &n);
            else rc = ai_engine_decode(e, &pending, &pos, &sid, 1, nullptr, ot.data(), &n);
            if (rc != AI_ENGINE_OK) { fprintf(stderr, "[api-gen] decode rc %d: %s\n", rc, ai_engine_last_error(e)); rc_all = 1; break; }
            ++rounds;
            for (int i = 0; i < n && (int) out.size() < n_gen && !stopped; ++i) {
                out.push_back(ot[(size_t) i]);
                stopped = is_stop(ot[(size_t) i]);
            }
            pos += n;
            pending = ot[(size_t) n - 1];
        }
        const double sg = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
        fprintf(stderr, "[api-gen] request %d: prompt %zu tokens prefilled in %.2f s; %zu tokens in %.2f s (%.2f tok/s), "
                        "%d rounds, %.2f tokens/round; free VRAM %lld MiB\n",
                r, prompt.size(), pf, out.size(), sg, (out.size() - 1) / std::max(sg, 1e-9), rounds,
                (double) (out.size() - 1) / std::max(1, rounds), (long long) ai_engine_free_vram_mib(e));
        if (!gen_out.empty()) {
            const std::string fn = gen_out + "." + std::to_string(r) + ".i32";
            FILE *            gf = fopen(fn.c_str(), "wb");
            if (gf) { fwrite(out.data(), 4, out.size(), gf); fclose(gf); }
            const std::string fp = gen_out + "." + std::to_string(r) + ".prompt.i32";   // to replay on a fresh engine
            FILE *            pf2 = fopen(fp.c_str(), "wb");
            if (pf2) { fwrite(prompt.data(), 4, prompt.size(), pf2); fclose(pf2); }
        }
        if (!turn.empty()) {   // the next request resends the conversation: this prompt, this answer, a new turn
            prompt.insert(prompt.end(), out.begin(), out.end());
            prompt.insert(prompt.end(), turn.begin(), turn.end());
        }
        if (ai_engine_reset_stream(e, 0) != AI_ENGINE_OK) { fprintf(stderr, "[api-gen] reset: %s\n", ai_engine_last_error(e)); rc_all = 1; break; }
        if (rc_all) break;
    }
    fprintf(stderr, "[api-gen] capture state at exit %d\n", (int) ai_engine_capture_state(e));
    ai_engine_destroy(e);
    return rc_all;
}
