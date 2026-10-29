// SPDX-License-Identifier: Apache-2.0
// ai-q4x-run: correctness harness for Qwen4ExpRun. Prefills a token file through Qwen4ExpRun (one chunk or --chunk N)
// and writes teacher-forced logits in ai-ref's --score format (i64 n, i64 V, then n x V f32), plus per-layer dumps
// under the oracle's names (--dump-dir), so tools/q4x_compare.py and layerdiff.py compare against the alphallama
// oracle (ai-ref built on alphallama's libllama, --cpu-experts 1) on the same tokens.
//
//   ai-q4x-run -m <model-00001-of-0000N.gguf> --tokens-in p.i32 --score out.logits [--chunk 512] [--slots 64]
//              [--dump-dir D] [--ctx 4096]
#include "core/device.h"
#include "engine/draft_vocab.h"
#include "models/qwen4exp_run.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace ai;

int main(int argc, char ** argv) {
    std::string model, tokens_in, score_out, dump_dir, gen_out;
    int64_t n_gen = 0;
    int64_t chunk = 512, ctx = 4096;
    int     slots = 64;
    std::string spec_ref;        // drafts from this token file (the continuation), optionally corrupted
    int     spec_k = 0;          // drafts per verify round (0 = plain decode); T = 1 + spec_k <= kDoorbellT (16)
    int     spec_agree = 0;      // --spec-agree 1: the MTP drafts every round and a suffix copy is taken only when its first
                                 // token equals the MTP's first draft
    int     spec_combine = 0;    // --spec-combine 1: no suffix copy -> MTP drafts, then retry the copy on history + drafts
    double  spec_noise = 0.0;    // probability a reference draft is replaced by a wrong token
    int     spec_ngram = 0;      // > 0: prompt/history n-gram lookup drafter (longest match of this many tokens)
    bool    spec_mtp = false;    // MTP head drafter (AI_Q4X_MTP = the sidecar GGUF)
    float   spec_tau = 0.0f;     // > 0: MTP drafts up to --spec-k while the cumulative top-1 probability >= tau
    int     spec_hybrid = 0;     // > 0: a prompt/history suffix match of >= this many tokens drafts up to --spec-k copied
                                 // tokens; otherwise the MTP head drafts --spec-mtp-k (needs --spec-mtp 1)
    int     spec_mtp_k = 2;
    int     spec_auto = 0;   // 1: cost-aware MTP depth -- tau each round from the measured verify cost per width
    std::vector<int32_t> stop_ids;   // --stop a,b,...: generation ends after emitting one of these (EOS / end of turn)
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "-m") model = v;
        else if (a == "--tokens-in") tokens_in = v;
        else if (a == "--score") score_out = v;
        else if (a == "--chunk") chunk = atoll(v.c_str());
        else if (a == "--ctx") ctx = atoll(v.c_str());
        else if (a == "--slots") slots = atoi(v.c_str());
        else if (a == "--dump-dir") dump_dir = v;
        else if (a == "-n") n_gen = atoll(v.c_str());       // greedy decode after the prompt (T = 1 chunks)
        else if (a == "--gen-out") gen_out = v;
        else if (a == "--spec-ref") spec_ref = v;
        else if (a == "--spec-k") spec_k = atoi(v.c_str());
        else if (a == "--spec-combine") spec_combine = atoi(v.c_str());
        else if (a == "--spec-agree") spec_agree = atoi(v.c_str());
        else if (a == "--spec-noise") spec_noise = atof(v.c_str());
        else if (a == "--spec-ngram") spec_ngram = atoi(v.c_str());
        else if (a == "--spec-mtp") spec_mtp = atoi(v.c_str()) != 0;
        else if (a == "--spec-tau") spec_tau = (float) atof(v.c_str());
        else if (a == "--spec-hybrid") spec_hybrid = atoi(v.c_str());
        else if (a == "--spec-mtp-k") spec_mtp_k = atoi(v.c_str());
        else if (a == "--spec-auto") spec_auto = atoi(v.c_str());
        else if (a == "--stop") {
            for (const char * q = v.c_str(); *q;) {
                char * e = nullptr;
                const long id = strtol(q, &e, 10);
                if (e == q) break;
                stop_ids.push_back((int32_t) id);
                q = *e == ',' ? e + 1 : e;
            }
        }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    std::ifstream tf(tokens_in, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
    std::vector<int32_t> ids(raw.size() / 4);
    memcpy(ids.data(), raw.data(), ids.size() * 4);
    if (ids.empty()) { fprintf(stderr, "no tokens in %s\n", tokens_in.c_str()); return 1; }

    // Own capture (the adapter's default) wraps the backend's launches, so ggml's own CUDA graphs must be off
    // (set_capture sets GGML_CUDA_DISABLE_GRAPHS); otherwise ggml's per-key warmup can complete on a call made inside
    // one of our captures and cudaStreamBeginCapture fails. Same call as Engine (engine.cpp), before the Device is
    // initialised.
    op::set_capture(op::Capture::Own);
    Device dev;
    if (!dev.init(0)) return 1;
    Qwen4Exp prog;
    try {
        if (!prog.load(dev, model, true)) return 1;
        if (spec_mtp) {
            const char * mp = getenv("AI_Q4X_MTP");
            if (!mp || !prog.load_mtp(dev, mp, true)) { fprintf(stderr, "q4x-run: --spec-mtp needs AI_Q4X_MTP=<sidecar>\n"); return 1; }
        }
        Qwen4ExpRun run(prog, dev, ctx, slots);
        if (spec_mtp) run.mtp_enable();
        // AI_DRAFT_VOCAB (default 1): the draft head holds AI_Q4X_DRAFT_ROWS (131072) rows chosen from the prompt's
        // scripts (engine/draft_vocab: Latin prompts -> common + Latin first, others -> their scripts); 3 = the first
        // 131072 ids (a view, no copy); 0 = the full 248320-row head. A smaller draft head makes each draft step
        // cheaper. Drafts only propose: the verify head is full, so the text never changes.
        if (spec_mtp) {
            static const int     dv   = getenv("AI_DRAFT_VOCAB") ? atoi(getenv("AI_DRAFT_VOCAB")) : 1;
            static const int64_t rows = getenv("AI_Q4X_DRAFT_ROWS") ? atoll(getenv("AI_Q4X_DRAFT_ROWS")) : 131072;
            if (dv != 0 && rows > 0) {
                const std::vector<uint8_t> cls = classify_gguf_tokens(prog.weights().get_str_array("tokenizer.ggml.tokens"));
                std::string                why;
                const std::vector<int32_t> sel = cls.empty() ? std::vector<int32_t>{} : draft_rows_for(cls, ids, rows, &why);
                const bool ok = run.set_draft_rows(sel, rows);
                fprintf(stderr, "[draft-vocab] %s%s\n", cls.empty() ? "no tokenizer.ggml.tokens: first rows" : why.c_str(),
                        ok ? (sel.empty() ? " (first-rows view)" : " (compact copy)") : " (UNAVAILABLE: full head)");
            }
        }
        // the activation workspace (nothing else sizes it here); AI_Q4X_WS_MB overrides, after the weights and state
        {
            // doorbell rounds (chunk <= 8 rows) peak around 18 MiB (see AI_WS_REPORT), so 256 MiB; every MiB not
            // spent here is VRAM for expert slots (84 MiB per slot across the 48 layers). Large-chunk scoring keeps 4096.
            const char * e  = getenv("AI_Q4X_WS_MB");
            const bool   db = getenv("AI_Q4X_DOORBELL") && atoi(getenv("AI_Q4X_DOORBELL")) != 0;
            // the layer-major prefill (AI_Q4X_PREFILL_LM) runs 128-row sub-chunks: scoring keeps every row's logits
            // (248320 x 128 x 4 B = 127 MB per sub-chunk), which does not fit in 256 MiB
            // the generation's prompt takes the layer-major prefill by default; scoring only when asked (see pf_lm below)
            const char * lme = getenv("AI_Q4X_PREFILL_LM");
            const bool   lm  = score_out.empty() ? (!lme || atoi(lme) != 0) : (lme && atoi(lme) != 0);
            // layer-major generation prompt: 640 MiB (MoE batches of 1024 rows into one output buffer; 768 leaves no
            // room for the MTP head's buffers); scoring 1536
            const size_t mb = e ? (size_t) atoll(e) : lm ? (score_out.empty() ? 640 : 1536) : (db && chunk <= 8 ? 256 : 4096);
            dev.init_workspace(mb << 20);
            fprintf(stderr, "[ai] q4x workspace %zu MiB\n", mb);
        }
        run.set_dump_dir(dump_dir);
        FILE * f = score_out.empty() ? nullptr : fopen(score_out.c_str(), "wb");
        const int64_t n = (int64_t) ids.size(), V = prog.spec().n_vocab;
        if (f) { fwrite(&n, 8, 1, f); fwrite(&V, 8, 1, f); }
        std::vector<float> row((size_t) V);
        const auto t0 = std::chrono::steady_clock::now();
        // AI_Q4X_TF_VERIFY=1: score through verify + commit(T) (record + fold) instead of chunk -- same chunking, so
        // any difference is the ReplaySSM fold
        static const bool tf_verify = getenv("AI_Q4X_TF_VERIFY") && atoi(getenv("AI_Q4X_TF_VERIFY"));
        // The generation's prompt goes through Qwen4ExpRun::prefill (layer-major, doorbell mode) by default
        // (AI_Q4X_PREFILL_LM=0: 8-row rounds); it matches 8-row scoring in NLL and is much faster.
        // Scoring keeps the 8-row doorbell rounds unless AI_Q4X_PREFILL_LM=1 is set explicitly: NLL checks of
        // decode-path changes (tail skip, CPU misses, residency) must score through the path those changes are on.
        const char *      pf_lm_e = getenv("AI_Q4X_PREFILL_LM");
        static const bool pf_lm   = pf_lm_e && atoi(pf_lm_e);                  // scoring
        static const bool pf_lm_g = !pf_lm_e || atoi(pf_lm_e) != 0;           // the generation's prompt
        auto prompt = [&]() {
            const auto tp = std::chrono::steady_clock::now();
            Tensor l;
            if (pf_lm_g) l = run.prefill(ids.data(), n);
            else for (int64_t off = 0; off < n; off += chunk) l = run.chunk(ids.data() + off, std::min(chunk, n - off));
            dev.synchronize();
            fprintf(stderr, "[q4x-run] prompt %lld tokens prefilled in %.2f s (%s)\n", (long long) n,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count(), pf_lm_g ? "layer-major" : "chunked");
            return l;
        };
        if (f && pf_lm) {   // scoring through the layer-major prefill: every row's logits
            std::vector<float> all;
            run.prefill(ids.data(), n, &all);
            fwrite(all.data(), 4, all.size(), f);
        }
        for (int64_t off = 0; (f && !pf_lm) || n_gen == 0 ? off < n : false; off += chunk) {   // generation re-prefills itself
            const int64_t T = std::min(chunk, n - off);
            Tensor logits = tf_verify ? run.verify(ids.data() + off, T) : run.chunk(ids.data() + off, T);
            if (tf_verify) run.commit(T);
            if (f) {
                std::vector<float> all((size_t) (V * T));
                dev.download(logits, all.data(), 0, all.size() * 4);
                fwrite(all.data(), 4, all.size(), f);
            }
        }
        if (f) fclose(f);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (n_gen > 0 && spec_k > 0) {   // speculative greedy: verify (pending + drafts), accept, commit
            std::vector<int32_t> ref;
            if (!spec_ref.empty()) {
                std::ifstream rf(spec_ref, std::ios::binary);
                std::vector<char> rr((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
                ref.resize(rr.size() / 4);
                memcpy(ref.data(), rr.data(), ref.size() * 4);
            }
            run.reset();
            Tensor lg = prompt();
            dev.download(lg, row.data(), (size_t) (lg.ne[1] - 1) * (size_t) V * 4, (size_t) V * 4);
            int32_t pending = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
            std::vector<int32_t> out{ pending }, ctx(ids);
            uint64_t rng = 0x9E3779B97F4A7C15ull, rounds = 0, drafted = 0, accepted = 0;
            double   draft_s = 0, verify_s = 0;   // host wall time in mtp_draft / in verify + its logits read
            uint64_t n_copied_rounds = 0;   // --spec-hybrid rounds drafted by the suffix match
            uint64_t n_combined_rounds = 0; // --spec-combine rounds whose MTP drafts were extended by a copy
            uint64_t n_vetoed_rounds = 0;   // --spec-agree rounds whose copy disagreed with the MTP's first draft
            dev.synchronize();
            run.reset_pool_stats();         // per-round pool numbers below are decode-only
            std::vector<float> all;
            const auto tg = std::chrono::steady_clock::now();
            auto is_stop = [&](int32_t t) { return std::find(stop_ids.begin(), stop_ids.end(), t) != stop_ids.end(); };
            bool stopped = is_stop(pending);
            // --spec-auto: verify cost per width T (ms, EMA), seeded with a typical measured curve, and the
            // tokens-per-ms rate. Drafting one row deeper pays when its cumulative acceptance probability p beats
            // rate * (cost[T+1] - cost[T]) (plus a draft step), so tau = that product, recomputed every round.
            double vcost[17] = { 0, 15, 20, 23.2, 31, 40, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74 };
            double rate_ema = 0.1, draft_step_ms = 1.3;
            uint64_t n_auto_deep = 0;
            if (spec_auto) run.set_mtp_keep(2);   // depth 2 is the measured optimum floor; tau decides rows 3+
            while ((int64_t) out.size() < n_gen && !stopped) {
                std::vector<int32_t> toks{ pending };
                if (!ref.empty()) {
                    for (int j = 0; j < spec_k && out.size() + (size_t) j < ref.size(); ++j) {
                        int32_t d = ref[out.size() + (size_t) j];
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        if ((double) (rng % 1000000) / 1e6 < spec_noise) d = (d + 1 + (int32_t) (rng % 1000)) % (int32_t) V;
                        toks.push_back(d);
                    }
                } else if (spec_mtp) {
                    bool copied = false;
                    if (spec_hybrid > 0) {   // longest suffix match (>= spec_hybrid tokens, up to 16) in prompt + output
                        std::vector<int32_t> h(ctx);
                        h.insert(h.end(), out.begin(), out.end());
                        for (int g = std::min<int>(16, (int) h.size() - 1); g >= spec_hybrid && !copied; --g) {
                            for (int64_t p = (int64_t) h.size() - g - 1; p >= 0; --p) {
                                if (!std::equal(h.end() - g, h.end(), h.begin() + p)) continue;
                                for (int j = 0; j < spec_k && p + g + j < (int64_t) h.size(); ++j) toks.push_back(h[(size_t) (p + g + j)]);
                                copied = toks.size() > 1;
                                break;
                            }
                        }
                        if (copied) n_copied_rounds++;
                    }
                    const auto td0 = std::chrono::steady_clock::now();
                    float tau = spec_tau;
                    if (spec_auto && !copied) {   // marginal value of the 3rd+ row at the current rate
                        const double dc = std::max(0.5, vcost[4] - vcost[3]) + draft_step_ms;
                        tau = (float) std::min(0.99, std::max(0.05, rate_ema * dc));
                    }
                    std::vector<int32_t> d;
                    if (!copied || spec_agree) d = run.mtp_draft(pending, spec_hybrid > 0 ? spec_mtp_k : spec_k, tau);
                    if (copied && spec_agree) {   // keep the copy only where the MTP agrees on its first token
                        if (d.empty() || d[0] != toks[1]) { toks.resize(1); copied = false; n_copied_rounds--; n_vetoed_rounds++; }
                        else d.clear();
                    }
                    if (spec_auto && !copied && (int) d.size() > 2) n_auto_deep++;
                    toks.insert(toks.end(), d.begin(), d.end());
                    // combined drafting: the MTP drafts may start a span that already occurred -- retry the
                    // longest suffix match (>= spec_hybrid) on history + drafts and append its continuation, up to
                    // spec_k drafts in all
                    if (spec_combine && spec_hybrid > 0 && !copied && !d.empty() && (int) toks.size() - 1 < spec_k) {
                        std::vector<int32_t> h(ctx);
                        h.insert(h.end(), out.begin(), out.end());
                        const int64_t hn = (int64_t) h.size();   // matches must start inside the true history
                        h.insert(h.end(), d.begin(), d.end());
                        bool ext = false;
                        for (int g = std::min<int>(16, (int) h.size() - 1); g >= spec_hybrid && !ext; --g) {
                            for (int64_t q = hn - g - 1; q >= 0; --q) {
                                if (!std::equal(h.end() - g, h.end(), h.begin() + q)) continue;
                                for (int64_t j = q + g; j < hn && (int) toks.size() - 1 < spec_k; ++j) toks.push_back(h[(size_t) j]);
                                ext = true;
                                break;
                            }
                        }
                        if (ext) n_combined_rounds++;
                    }
                    draft_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
                } else if (spec_ngram > 0) {   // longest suffix match (spec_ngram..1 tokens) in prompt + output
                    std::vector<int32_t> h(ctx);
                    h.insert(h.end(), out.begin(), out.end());
                    for (int g = spec_ngram; g >= 1 && toks.size() == 1; --g) {
                        if ((int64_t) h.size() <= g) continue;
                        for (int64_t p = (int64_t) h.size() - g - 1; p >= 0; --p) {
                            if (!std::equal(h.end() - g, h.end(), h.begin() + p)) continue;
                            for (int j = 0; j < spec_k && p + g + j < (int64_t) h.size(); ++j) toks.push_back(h[(size_t) (p + g + j)]);
                            break;
                        }
                    }
                }
                const int64_t T = (int64_t) toks.size();
                const auto tv0 = std::chrono::steady_clock::now();
                lg = run.verify(toks.data(), T);
                all.resize((size_t) (V * T));
                dev.download(lg, all.data(), 0, all.size() * 4);
                const double vms = 1e3 * std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
                verify_s += vms / 1e3;
                if (spec_auto && T <= 16) vcost[T] = 0.8 * vcost[T] + 0.2 * vms;
                std::vector<int32_t> gq((size_t) T);
                for (int64_t t = 0; t < T; ++t) {
                    const float * r = all.data() + (size_t) t * (size_t) V;
                    gq[(size_t) t] = (int32_t) (std::max_element(r, r + V) - r);
                }
                int64_t a = 0;
                while (a < T - 1 && toks[(size_t) (a + 1)] == gq[(size_t) a]) ++a;
                run.commit(a + 1);
                for (int64_t t = 0; t <= a && (int64_t) out.size() < n_gen && !stopped; ++t) {
                    out.push_back(gq[(size_t) t]);
                    stopped = is_stop(gq[(size_t) t]);
                }
                pending = gq[(size_t) a];
                rounds++; drafted += (uint64_t) (T - 1); accepted += (uint64_t) a;
                if (spec_auto) rate_ema = 0.9 * rate_ema + 0.1 * ((double) (a + 1) / std::max(1.0, vcost[T] + draft_step_ms * (double) (T - 1)));
            }
            const double sg = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
            fprintf(stderr, "[q4x-run] spec greedy %lld tokens in %.2f s (%.2f tok/s): %llu rounds, %.2f tokens/round, "
                            "accepted %llu of %llu drafts\n", (long long) out.size(), sg, (out.size() - 1) / std::max(sg, 1e-9),
                    (unsigned long long) rounds, (double) (out.size() - 1) / std::max<double>(1, (double) rounds),
                    (unsigned long long) accepted, (unsigned long long) drafted);
            if (spec_auto) fprintf(stderr, "[q4x-run] spec-auto: %llu MTP rounds deeper than 2; verify cost ms T2..8 %.1f %.1f %.1f %.1f %.1f %.1f %.1f; rate %.3f tok/ms\n",
                                   (unsigned long long) n_auto_deep, vcost[2], vcost[3], vcost[4], vcost[5], vcost[6], vcost[7], vcost[8], rate_ema);
            fprintf(stderr, "[q4x-run] spec per round: draft %.2f ms, verify %.2f ms, rest %.2f ms; suffix-copied rounds %llu, MTP+copy combined rounds %llu, agree-vetoed rounds %llu\n",
                    1e3 * draft_s / std::max<double>(1, (double) rounds), 1e3 * verify_s / std::max<double>(1, (double) rounds),
                    1e3 * (sg - draft_s - verify_s) / std::max<double>(1, (double) rounds), (unsigned long long) n_copied_rounds,
                    (unsigned long long) n_combined_rounds, (unsigned long long) n_vetoed_rounds);
            if (const auto * mp = run.miss_pool()) {
                const auto & st = mp->stats();
                fprintf(stderr, "[q4x-run] spec CPU miss pool: %.1f misses and %.2f ms busy per ROUND (incl. prefill), admissions %llu (tensor copies: %llu from the RAM tier, %llu staged), tail-skipped %llu\n",
                        (double) st.misses.load() / (double) std::max<uint64_t>(1, rounds),
                        st.busy_ns.load() / 1e6 / (double) std::max<uint64_t>(1, rounds), (unsigned long long) run.admitted(),
                        (unsigned long long) run.admit_direct(), (unsigned long long) run.admit_staged(), (unsigned long long) st.skipped.load());
                const double rr = (double) std::max<uint64_t>(1, rounds);
                fprintf(stderr, "[q4x-run] spec pool profile per ROUND: %.1f groups (%.1f of %.1f columns grouped); ms: quantise-x %.2f, "
                                "gate/up %.2f, swiglu %.2f, down %.2f; rest of busy %.2f\n",
                        st.groups.load() / rr, st.grouped_cols.load() / rr, st.misses.load() / rr, st.ns_x.load() / 1e6 / rr,
                        st.ns_gu.load() / 1e6 / rr, st.ns_mid.load() / 1e6 / rr, st.ns_down.load() / 1e6 / rr,
                        (st.busy_ns.load() - st.ns_x.load() - st.ns_gu.load() - st.ns_mid.load() - st.ns_down.load()) / 1e6 / rr);
            }
            if (!ref.empty()) {
                size_t same = 0;
                while (same < out.size() && same < ref.size() && out[same] == ref[same]) ++same;
                fprintf(stderr, "[q4x-run] spec output matches the reference for the first %zu of %zu tokens\n", same, out.size());
            }
            if (!gen_out.empty()) {
                FILE * gf = fopen(gen_out.c_str(), "wb");
                fwrite(out.data(), 4, out.size(), gf);
                fclose(gf);
            }
        } else if (n_gen > 0) {   // greedy: the prompt's last row, then one token per round
            run.reset();
            Tensor lg = prompt();
            std::vector<int32_t> out;
            const auto tg = std::chrono::steady_clock::now();
            for (int64_t i = 0; i < n_gen; ++i) {
                dev.download(lg, row.data(), (size_t) (lg.ne[1] - 1) * (size_t) V * 4, (size_t) V * 4);
                const int32_t tok = (int32_t) (std::max_element(row.begin(), row.end()) - row.begin());
                out.push_back(tok);
                if (i + 1 < n_gen) lg = getenv("AI_Q4X_CAPTURE") && atoi(getenv("AI_Q4X_CAPTURE")) ? run.decode(tok) : run.chunk(&tok, 1);
            }
            const double sg = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();
            fprintf(stderr, "[q4x-run] greedy %lld tokens in %.2f s (%.2f tok/s)\n", (long long) n_gen,
                    sg, (n_gen - 1) / std::max(sg, 1e-9));
            if (const auto * mp = run.miss_pool()) {
                const auto & st = mp->stats();
                fprintf(stderr, "[q4x-run] admissions: %llu\n", (unsigned long long) run.admitted());
                fprintf(stderr, "[q4x-run] CPU miss pool: %llu rings, %llu misses (%.1f per token), busy %.2f ms per token\n",
                        (unsigned long long) st.rings.load(), (unsigned long long) st.misses.load(),
                        (double) st.misses.load() / (double) (n + n_gen), st.busy_ns.load() / 1e6 / (double) (n + n_gen));
            }
            if (!gen_out.empty()) {
                FILE * gf = fopen(gen_out.c_str(), "wb");
                fwrite(out.data(), 4, out.size(), gf);
                fclose(gf);
            }
        }
        const auto & st = run.pool().stats();
        fprintf(stderr, "[q4x-run] %lld positions in %.2f s; pool lookups %llu misses %llu (%.1f GB paged)\n",
                (long long) n, s, (unsigned long long) st.lookups, (unsigned long long) st.misses, st.bytes / 1e9);
    } catch (const std::exception & e) {
        fprintf(stderr, "q4x-run: %s\n", e.what());
        return 1;
    }
    return 0;
}
