// alphainfer CLI.
//
//   generate:  alphainfer -m M.gguf -p "text" [--chat] [-n 64]
//   parity:    alphainfer -m M.gguf --tokens-in prompt.i32 -n 64 --gen-out ai_gen.i32
//   scoring:   alphainfer -m M.gguf --tokens-in prompt.i32 --score ai.logits
//
// Token files are raw little-endian int32. Logit files: int64 n, int64 V, then n*V float32.
// NLL files (--score-nll): int64 n, int64 V, then (n-1) float32 -- the NLL of the actual next token
// at each position, since position i predicts token i+1 and the last position has no target. Same
// header as --score, so a reader can assert the file size and never assume a vocab size.
#include "core/weights.h"
#include "engine/draft_vocab.h"
#include "engine/engine.h"
#include "engine/mtp_policy.h"
#include "engine/tokenizer.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// An input that cannot be opened is an error, not an empty prompt: otherwise a missing -f file runs a bare chat
// template and prints a plausible speed.
static void must_open(const std::ifstream & f, const std::string & path, const char * what) {
    if (!f) {
        fprintf(stderr, "cannot open %s '%s'\n", what, path.c_str());
        exit(2);
    }
}

static std::vector<int32_t> read_tokens(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    must_open(f, path, "--tokens-in file");
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<int32_t> t(raw.size() / 4);
    memcpy(t.data(), raw.data(), t.size() * 4);
    return t;
}

static void write_tokens(const std::string & path, const std::vector<int32_t> & t) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) t.data(), (std::streamsize) (t.size() * 4));
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    must_open(f, path, "prompt file");
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void usage() {
    fprintf(stderr,
            "usage: alphainfer -m model.gguf (-p text | -f file | --tokens-in file) [options]\n"
            "  -n N              tokens to generate (default 64)\n"
            "  --chat            wrap the prompt as a ChatML user turn (thinking off)\n"
            "  --think           with --chat, leave thinking on\n"
            "  --ctx N|auto      max context (default 8192; auto = largest that fits)\n"
            "  --kv-type T       KV cell type: auto (default: fit + measured speed rules) | f16 | q8_0 | q4_0\n"
            "  --chunk N         prefill rows per forward (default 512)\n"
            "  --ws-mib N        workspace MiB (default: derived)\n"
            "  --dump-tokens F   write the prompt token ids\n"
            "  --gen-out F       write generated token ids\n"
            "  --score F         teacher-forced logits for every prompt position -> F\n"
            "  --score-n N       limit scored positions\n"
            "  --score-nll F     teacher-forced NLL of the actual next token -> F: one float per\n"
            "                    position instead of a full logit row (248320x smaller)\n"
            "  --eager           launch every op separately (no captured rounds)\n"
            "  --mtp K           MTP self-speculation, K drafts per step (up to K with --mtp-adapt)\n"
            "  --mtp-adapt       choose drafts per step from measured acceptance and step time\n"
            "  --dflash2 GGUF    DFlash2 block-diffusion drafter (tools/convert_dflash2.py), 7 drafts per step\n"
            "  --mtp-exact       MTP verify bit-identical to plain decode at any K (default on MoE models; on a\n"
            "                    dense model it also keeps the verify off the small-M tiers)\n"
            "  --mtp-fast        MoE: the old fast verify (output can drift from plain decode)\n"
            "  --warm-profiles   record every decode profile at startup, so none is built mid-run\n"
            "  --parallel N      decode N streams in shared rounds (the prompt N times, or --prompt-list)\n"
            "  --prompt-list F   one prompt per line (with --chat each is wrapped); streams = lines\n"
            "  --repeat N        run the same request N times on one warm engine (reports each)\n"
            "  --expert-slots N  keep routed experts in host memory; N device slots per layer\n"
            "                    (auto: from the VRAM left, with AI_OFFLOAD=doorbell)\n"
            "  --no-graphs       no round capture (AI_CAPTURE=none; also own|backend)\n"
            "  -q                quiet (no streamed text)\n");
}

int main(int argc, char ** argv) {
    ai::EngineConfig cfg;
    std::string prompt, prompt_file, tokens_in, dump_tokens, gen_out, score_out, prompt_list;
    std::string score_nll_out;
    int64_t n_gen = 64, score_n = -1;
    bool    mtp_exact = false;
    bool    mtp_fast  = false;
    int     parallel = 1;
    int     repeat   = 1;
    bool chat = false, think = false, quiet = false;
    bool warm_profiles = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (a == "-m") cfg.model_path = next();
        else if (a == "-p") prompt = next();
        else if (a == "-f") prompt_file = next();
        else if (a == "--tokens-in") tokens_in = next();
        else if (a == "-n") n_gen = atoll(next().c_str());
        else if (a == "--chat") chat = true;
        else if (a == "--think") think = true;
        else if (a == "--ctx") {
            const std::string v = next();
            cfg.max_ctx = v == "auto" ? 0 : atoll(v.c_str());
        }
        else if (a == "--kv-type") {
            const std::string t = next();
            if      (t == "auto") cfg.kv_type = ai::kKvAuto;
            else if (t == "f16")  cfg.kv_type = ai::F16;
            else if (t == "q8_0") cfg.kv_type = ai::Q8_0;
            else if (t == "q4_0") cfg.kv_type = ai::Q4_0;
            else { fprintf(stderr, "unknown --kv-type '%s' (auto|f16|q8_0|q4_0)\n", t.c_str()); return 2; }
        }
        else if (a == "--chunk") cfg.chunk = atoll(next().c_str());
        else if (a == "--ws-mib") cfg.ws_bytes = (size_t) atoll(next().c_str()) << 20;
        else if (a == "--dump-tokens") dump_tokens = next();
        else if (a == "--gen-out") gen_out = next();
        else if (a == "--score") score_out = next();
        else if (a == "--score-nll") score_nll_out = next();
        else if (a == "--score-n") score_n = atoll(next().c_str());
        else if (a == "-q") quiet = true;
        else if (a == "--eager") cfg.eager = true;
        else if (a == "--mtp") cfg.mtp_k = atoi(next().c_str());
        else if (a == "--mtp-adapt") cfg.mtp_adapt = true;
        else if (a == "--mtp-exact") mtp_exact = true;
        else if (a == "--mtp-fast") mtp_fast = true;
        else if (a == "--dflash2") cfg.dflash2_path = next();
        else if (a == "--parallel") parallel = atoi(next().c_str());
        else if (a == "--repeat") repeat = atoi(next().c_str());
        else if (a == "--prompt-list") prompt_list = next();
        else if (a == "--expert-slots") {
            const std::string v = next();
            cfg.expert_slots_auto = v == "auto";
            cfg.expert_slots      = cfg.expert_slots_auto ? 1 : atoi(v.c_str());
        }
        else if (a == "--no-graphs") setenv("AI_CAPTURE", "none", 1);
        else if (a == "--warm-profiles") warm_profiles = true;
        else { usage(); return 2; }
    }
    if (!cfg.dflash2_path.empty()) {
        // DFlash2 verifies [t, d1..d7] exactly like MTP k=7, so it takes MTP's verify configuration (the dense
        // small-M tier and the FA/MMVF pins below); the engine sets k from the drafter's block size.
        if (cfg.mtp_k > 0 || cfg.mtp_adapt) {
            fprintf(stderr, "[ai] --dflash2 and --mtp are alternatives\n");
            return 2;
        }
        cfg.mtp_k = 7;
    }
    if (cfg.model_path.empty() ||
        (prompt.empty() && prompt_file.empty() && tokens_in.empty() && prompt_list.empty())) {
        usage();
        return 2;
    }

    ai::Tokenizer tok;
    if (!tok.load(cfg.model_path)) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }
    std::vector<std::vector<int32_t>> batch;
    if (!prompt_list.empty()) {
        std::ifstream f(prompt_list);
        must_open(f, prompt_list, "--prompt-list file");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) {
                continue;
            }
            batch.push_back(tok.encode(chat ? ai::Tokenizer::chat_prompt(line, think) : line));
        }
        if (batch.empty()) {
            fprintf(stderr, "--prompt-list: no prompts\n");
            return 2;
        }
        parallel = (int) batch.size();
    }
    std::vector<int32_t> ids;
    if (!prompt_list.empty()) {
        ids = batch[0];
    } else if (!tokens_in.empty()) {
        ids = read_tokens(tokens_in);
    } else {
        std::string text = prompt.empty() ? read_file(prompt_file) : prompt;
        if (chat) {
            text = ai::Tokenizer::chat_prompt(text, think);
        }
        ids = tok.encode(text);
    }
    if (score_n > 0 && (int64_t) ids.size() > score_n) {
        ids.resize((size_t) score_n);
    }
    if (!dump_tokens.empty()) {
        write_tokens(dump_tokens, ids);
    }

    if (batch.empty()) {
        batch.assign((size_t) std::max(1, parallel), ids);
    }
    cfg.max_batch = (int) batch.size();
    // The verify configuration (exact/fast, pins, K clamp, gate default) lives in engine/mtp_policy.cpp, shared
    // with the C API so the two cannot drift.
    // AI_DF2_ROUTE=1 with --dflash2: route each request by script -- DFlash2 for Latin-dominant prompts, the MTP head
    // otherwise (DFlash2 is faster on English, slower on other languages). The MTP chain is held at 5 so its graph
    // profiles never share a key with DFlash2's 7-draft block.
    const bool df2_route = !cfg.dflash2_path.empty() && getenv("AI_DF2_ROUTE") && atoi(getenv("AI_DF2_ROUTE")) != 0;
    if (df2_route) setenv("AI_MTP_CHAIN", "5", 0);
    ai::apply_mtp_policy(cfg, mtp_exact ? ai::kMtpModeExact : mtp_fast ? ai::kMtpModeFast : ai::kMtpModeDefault);
    ai::Engine eng;
    if (!eng.init(cfg)) {
        fprintf(stderr, "engine init failed\n");
        return 1;
    }
    if (warm_profiles) {
        // Record every profile the F ladder can ask for before any generation, so nothing is
        // instantiated mid-run. Composes with generate_batch because reset_streams() does not clear
        // rounds_: these profiles are replayed, not rebuilt. warm_profiles() resets every slot before
        // returning (staging reserves KV to F), and generate_batch resets again on entry anyway.
        ai::Timing wtm;
        try {
            eng.open_streams(cfg.max_batch);
            const int nw = eng.warm_profiles(wtm);
            fprintf(stderr, "[ai] --warm-profiles: %d profile(s) recorded at startup (B=%d)\n",
                    nw, cfg.max_batch);
            if (nw < 1) {
                // Zero recorded is the silent-degrade case this flag exists to rule out.
                fprintf(stderr, "[ai] --warm-profiles: recorded 0 profiles; capture is not happening\n");
                return 1;
            }
        } catch (const std::exception & ex) {
            fprintf(stderr, "[ai] --warm-profiles failed: %s\n", ex.what());
            return 1;
        }
    }

    ai::Timing t;
    if (!score_nll_out.empty()) {
        // A quality gate needs one number per position, not a 248320-float row. Full logits for a
        // properly powered paired gate (~82 chunks x 8 languages x 3 arms) would be ~500 GB of I/O;
        // reducing in the callback makes it ~100 KB. Same self-describing header as --score.
        FILE * f = fopen(score_nll_out.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "[ai] cannot open %s\n", score_nll_out.c_str());
            return 1;
        }
        const int64_t n = (int64_t) ids.size();
        const int64_t V = eng.program().spec().n_vocab;
        fwrite(&n, 8, 1, f);
        fwrite(&V, 8, 1, f);
        int64_t written = 0;
        eng.score(ids, [&](int64_t pos, const float * row) {
            if (pos + 1 >= n) return;                      // the final position has no target
            const int32_t tgt = ids[(size_t) (pos + 1)];
            if (tgt < 0 || (int64_t) tgt >= V) {
                throw std::runtime_error("--score-nll: token id outside the vocab");
            }
            // Accumulate the logsumexp in double. Float accumulation over 248320 terms drifts, and
            // that drift would differ between two arms scoring the same text -- which is exactly the
            // paired difference a quality gate exists to measure. Max-subtraction keeps exp() in range.
            float m = row[0];
            for (int64_t j = 1; j < V; ++j) {
                if (row[j] > m) m = row[j];
            }
            double sum = 0.0;
            for (int64_t j = 0; j < V; ++j) {
                sum += std::exp((double) row[j] - (double) m);
            }
            const float nll = (float) (((double) m + std::log(sum)) - (double) row[tgt]);
            fwrite(&nll, 4, 1, f);
            ++written;
        }, &t);
        fclose(f);
        // Print the count so a caller can assert it: a silently short file is how a broken reader
        // reports a plausible wrong number instead of an error.
        fprintf(stderr, "[ai] score-nll: %lld positions, %lld NLLs in %.2f s | ws high-water %.0f MiB\n",
                (long long) n, (long long) written, t.prefill_s,
                eng.device().ws_high_water() / 1048576.0);
        if (written != n - 1) {
            fprintf(stderr, "[ai] score-nll: expected %lld NLLs, wrote %lld\n",
                    (long long) (n - 1), (long long) written);
            return 1;
        }
        return 0;
    }
    if (!score_out.empty()) {
        FILE * f = fopen(score_out.c_str(), "wb");
        const int64_t n = (int64_t) ids.size();
        const int64_t V = eng.program().spec().n_vocab;
        fwrite(&n, 8, 1, f);
        fwrite(&V, 8, 1, f);
        eng.score(ids, [&](int64_t, const float * row) { fwrite(row, 4, (size_t) V, f); }, &t);
        fclose(f);
        fprintf(stderr, "[ai] scored %lld positions in %.2f s (%.1f tok/s incl. logit download) | ws high-water %.0f MiB\n",
                (long long) n, t.prefill_s, n / t.prefill_s, eng.device().ws_high_water() / 1048576.0);
        return 0;
    }

    const bool multi = batch.size() > 1;
    if (!quiet) {
        fprintf(stderr, "[ai] prompt: %zu tokens%s\n", ids.size(),
                multi ? (" (x" + std::to_string(batch.size()) + " streams)").c_str() : "");
    }
    // AI_DRAFT_VOCAB: pick the MTP draft rows from the prompt's scripts (default rows for Latin prompts)
    // MTP and DFlash2 share the draft head, so both pick its rows by script
    if (cfg.mtp_k > 0 && eng.draft_vocab_rows() > 0 && (!getenv("AI_DRAFT_VOCAB") || atoi(getenv("AI_DRAFT_VOCAB")) != 0)) {
        const std::vector<uint8_t> cls  = ai::classify_vocab(tok, tok.n_vocab());
        std::string                why;
        const std::vector<int32_t> rows = ai::draft_rows_for(cls, ids, eng.draft_vocab_rows(), &why);
        const bool                 ok   = eng.set_draft_vocab(rows);
        fprintf(stderr, "[draft-vocab] %s%s\n", why.c_str(), ok ? "" : " (unavailable)");
    }
    if (df2_route && eng.dflash2()) {
        const std::vector<uint8_t> cls   = ai::classify_vocab(tok, tok.n_vocab());
        const bool                 latin = ai::prompt_is_latin(cls, ids);
        const double               en    = ai::prompt_english_share(tok, ids);
        static const double        thr   = getenv("AI_DF2_ROUTE_EN") ? atof(getenv("AI_DF2_ROUTE_EN")) : 0.08;
        // AI_DF2_ROUTE_EDIT=1 (default off): a code-edit request goes to the MTP head. Opt-in because it was a net loss
        // over a mix of edit kinds.
        static const bool          edit_on = getenv("AI_DF2_ROUTE_EDIT") && atoi(getenv("AI_DF2_ROUTE_EDIT")) != 0;
        const bool                 edit  = edit_on && latin && en >= thr && ai::prompt_is_code_edit(tok, ids);
        const bool                 df2   = latin && en >= thr && !edit;   // DFlash2 wins on English only
        eng.set_df2_active(df2);
        fprintf(stderr, "[ai] df2 route: %s, English function words %.1f%%%s -> %s\n", latin ? "Latin" : "non-Latin",
                100.0 * en, edit ? ", code edit" : "", df2 ? "DFlash2" : "MTP head");
    }
    // AI_WARM_MTP (default on for a single-stream MTP run): capture every MTP round profile this generation can reach
    // before the decode timer starts: the envelope buckets for positions prompt..prompt+n_gen x verify widths 0..K.
    // This moves the capture cost out of the decode window to startup; it does not remove it. AI_WARM_MTP=0 disables.
    if (eng.mtp_k() > 0 && !eng.dflash2() && !cfg.mtp_adapt && !multi && score_nll_out.empty() && score_out.empty() &&
        (!getenv("AI_WARM_MTP") || atoi(getenv("AI_WARM_MTP")) != 0)) {
        ai::Timing wtm;
        const auto w0 = std::chrono::steady_clock::now();
        try {
            eng.open_streams(1);
            const int64_t p0 = (int64_t) ids.size();
            const int     nw = eng.warm_mtp_profiles(p0, p0 + n_gen, wtm);
            fprintf(stderr, "[ai] AI_WARM_MTP: %d profile(s) warm in %.1f ms (outside the decode timer)\n", nw,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count());
        } catch (const std::exception & ex) {
            fprintf(stderr, "[ai] AI_WARM_MTP failed (continuing cold): %s\n", ex.what());
        }
    }
    ai::ExpertPool::Stats             pool0;
    ai::Qwen35Hybrid::OffloadProf     prof0;
    for (int run = 0; run < std::max(1, repeat); ++run) {
    t = ai::Timing{};
    if (repeat > 1) {
        fprintf(stderr, "[ai] run %d/%d\n", run + 1, repeat);
    }
    auto outs = eng.generate_batch(batch, n_gen, [&](int b, int32_t id) {
        if (!quiet && !multi) {
            std::cout << tok.piece(id) << std::flush;
        }
        return !tok.is_eog(id);
    }, &t);
    if (!quiet) {
        if (multi) {
            for (size_t b = 0; b < outs.size(); ++b) {
                std::cout << "=== stream " << b << " (" << outs[b].size() << " tokens) ===\n";
                for (int32_t id : outs[b]) {
                    std::cout << tok.piece(id);
                }
                std::cout << "\n";
            }
        }
        std::cout << std::endl;
    }
    // AI_SHORTLIST_PROBE: how often is a generated token outside a context-adaptive draft shortlist
    // S_N = {id < N} U {every id already seen in the prompt or the generation}? A draft head restricted to
    // S_N can never propose such a token, so this bounds the acceptance a shortlist could lose.
    if (getenv("AI_SHORTLIST_PROBE") && !outs.empty()) {
        const int Ns[] = { 4096, 8192, 16384, 32768, 65536, 131072 };
        std::vector<char> seen(1 << 18, 0);
        for (int32_t p : ids) {
            if (p >= 0 && p < (1 << 18)) seen[(size_t) p] = 1;
        }
        long long miss_ctx[6] = { 0 }, miss_static[6] = { 0 }, n = 0;
        for (int32_t g : outs[0]) {
            for (int k = 0; k < 6; ++k) {
                const bool low = g < Ns[k];
                miss_static[k] += !low;
                miss_ctx[k] += !low && !(g >= 0 && g < (1 << 18) && seen[(size_t) g]);
            }
            if (g >= 0 && g < (1 << 18)) seen[(size_t) g] = 1;
            ++n;
        }
        fprintf(stderr, "[shortlist-probe] %lld generated tokens; miss%% static(id<N) / context(id<N or seen):", n);
        for (int k = 0; k < 6; ++k) {
            fprintf(stderr, " N=%d %.1f/%.1f;", Ns[k], 100.0 * miss_static[k] / std::max(1LL, n),
                    100.0 * miss_ctx[k] / std::max(1LL, n));
        }
        fprintf(stderr, "\n");
    }
    if (!gen_out.empty()) {
        write_tokens(gen_out, outs[0]);
        for (size_t b = 1; b < outs.size(); ++b) {
            write_tokens(gen_out + "." + std::to_string(b), outs[b]);
        }
    }
    // decode tokens per second (summed over streams; each stream's first token comes from prefill)
    const double dec_rate = t.decode_s > 0 ? t.decode_tokens / t.decode_s : 0.0;
    fprintf(stderr,
            "[ai] prefill %lld tok in %.3f s = %.1f tok/s | decode %lld tokens in %lld rounds, %.3f s = %.2f tok/s "
            "(%.2f ms/tok, %d streams = %.2f tok/s per stream) | blocking calls/round %.2f | host ms/round %.3f | "
            "rounds recorded %d (max %d nodes) | ws high-water %.0f MiB\n",
            (long long) t.prompt_n, t.prefill_s, t.prompt_n / t.prefill_s, (long long) t.decode_tokens,
            (long long) t.decode_rounds, t.decode_s, dec_rate, dec_rate > 0 ? 1000.0 / dec_rate : 0.0, t.batch,
            dec_rate / t.batch, t.decode_rounds ? (double) t.decode_blocking_calls / t.decode_rounds : 0.0,
            t.host_ms_per_round, t.rounds_recorded, t.round_nodes, eng.device().ws_high_water() / 1048576.0);
    if (t.mtp_drafted > 0) {
        std::string ks;
        for (size_t k = 1; k < t.k_steps.size(); ++k) {
            if (t.k_steps[k]) {
                ks += " k" + std::to_string(k) + ":" + std::to_string(t.k_steps[k]);
            }
        }
        fprintf(stderr, "[ai] mtp k<=%d%s: %lld steps, acceptance %.1f%% (%lld/%lld drafts), %.2f tokens/step/stream |"
                        " steps by k:%s\n",
                cfg.mtp_k, cfg.mtp_adapt ? " adaptive" : "", (long long) t.decode_rounds,
                100.0 * (double) t.mtp_accepted / (double) t.mtp_drafted, (long long) t.mtp_accepted,
                (long long) t.mtp_drafted, (double) t.decode_tokens / (double) t.decode_rounds / t.batch, ks.c_str());
        std::string acc, ver;
        for (size_t j = 0; j < t.mtp_verified_by_pos.size() && t.mtp_verified_by_pos[j] > 0; ++j) {
            acc += (j ? "," : "") + std::to_string(t.mtp_acc_by_pos[j]);
            ver += (j ? "," : "") + std::to_string(t.mtp_verified_by_pos[j]);
        }
        // The gate can cut the verify width, so not every draft is verified; the verified count per position is
        // printed beside the accepted count.
        fprintf(stderr, "[ai] mtp accepted by pos %s | verified by pos %s | rounds %lld | cold rounds %lld = %.2f ms "
                        "(inside the decode timer)\n", acc.c_str(), ver.c_str(), (long long) t.decode_rounds,
                (long long) t.cold_rounds, t.cold_ms);
    }
    if (t.mtp_prof_steps > 0) {
        const double n   = (double) t.mtp_prof_steps;
        const double tot = t.mtp_phase_ms[0] + t.mtp_phase_ms[1] + t.mtp_phase_ms[2] + t.mtp_phase_ms[3];
        fprintf(stderr, "[ai] mtp step split over %lld steps (uncaptured, one sync per phase): "
                        "verify %.3f ms (%.0f%%) | accept+fold %.3f (%.0f%%) | head fill %.3f (%.0f%%) | "
                        "chain %.3f (%.0f%%) | total %.3f ms\n",
                (long long) t.mtp_prof_steps, t.mtp_phase_ms[0] / n, 100 * t.mtp_phase_ms[0] / tot,
                t.mtp_phase_ms[1] / n, 100 * t.mtp_phase_ms[1] / tot, t.mtp_phase_ms[2] / n,
                100 * t.mtp_phase_ms[2] / tot, t.mtp_phase_ms[3] / n, 100 * t.mtp_phase_ms[3] / tot, tot / n);
    }
    const ai::Qwen35Hybrid::OffloadProf & op = eng.program().offload_prof();
    if (op.steps > prof0.steps) {
        const double wait = op.wait_ms - prof0.wait_ms, cpu = op.cpu_ms - prof0.cpu_ms;
        const double ov_all = (double) (op.overlap_all - prof0.overlap_all);
        fprintf(stderr, "[ai] streamed MoE host steps: %llu, blocked on device %.1f ms, host cpu %.1f ms "
                        "(per decode token: %.2f / %.3f ms) | experts reused from the layer's previous step %.1f%%\n",
                (unsigned long long) (op.steps - prof0.steps), wait, cpu,
                t.decode_tokens ? wait / t.decode_tokens : 0.0, t.decode_tokens ? cpu / t.decode_tokens : 0.0,
                ov_all > 0 ? 100.0 * (double) (op.overlap_hit - prof0.overlap_hit) / ov_all : 0.0);
        prof0 = op;
    }
    if (const auto * pool = eng.program().expert_pool()) {
        const auto & st = pool->stats();
        const double lk = (double) (st.lookups - pool0.lookups), ms = (double) (st.misses - pool0.misses);
        fprintf(stderr, "[ai] expert pool: %.0f lookups, %.0f misses (hit rate %.2f%%), %.2f GiB paged in\n",
                lk, ms, lk > 0 ? 100.0 * (1.0 - ms / lk) : 0.0, (st.bytes - pool0.bytes) / 1073741824.0);
        if (st.pf_issued > pool0.pf_issued || getenv("AI_PF_PRED")) {
            const auto & ps = eng.program().pred_stats();
            fprintf(stderr, "[ai] prefetch: %llu issued, %llu used, %llu wasted, %.2f GiB | predictor (cumulative): "
                            "%llu steps, %.2f cold candidates/step, top-%s caught %.1f%% of true cold experts\n",
                    (unsigned long long) (st.pf_issued - pool0.pf_issued),
                    (unsigned long long) (st.pf_used - pool0.pf_used),
                    (unsigned long long) (st.pf_wasted - pool0.pf_wasted), (st.pf_bytes - pool0.pf_bytes) / 1073741824.0,
                    (unsigned long long) ps.steps, ps.steps ? (double) ps.cand / (double) ps.steps : 0.0,
                    getenv("AI_PF_MAX") ? getenv("AI_PF_MAX") : "2",
                    ps.cold_true ? 100.0 * (double) ps.cold_pred / (double) ps.cold_true : 0.0);
        }
        pool0 = st;
        // AI_POOL_CENSUS=1: per-layer lookups, misses and distinct experts touched in this run
        if (getenv("AI_POOL_CENSUS")) {
            std::string line = "[ai] pool census (layer:distinct/misses)";
            int over = 0;
            uint64_t miss_over = 0, miss_all = 0;
            for (int l = 0; l < pool->n_layers(); ++l) {
                const auto & L = pool->layer(l);
                int distinct = 0;
                for (uint64_t c : L.seen) {
                    distinct += c > 0 ? 1 : 0;
                }
                miss_all += L.misses;
                if (distinct > pool->n_slots()) {
                    ++over;
                    miss_over += L.misses;
                }
                line += " " + std::to_string(l) + ":" + std::to_string(distinct) + "/" + std::to_string(L.misses);
            }
            fprintf(stderr, "%s\n", line.c_str());
            fprintf(stderr, "[ai] pool census: %d layers touch more experts than the %d slots; they hold %llu of %llu misses\n",
                    over, pool->n_slots(), (unsigned long long) miss_over, (unsigned long long) miss_all);
            pool->reset_layer_counters();
        }
    }
    }
    return 0;
}
