// See api/q4x_serve.h.
#include "api/q4x_serve.h"

#include "core/device.h"
#include "engine/draft_vocab.h"
#include "models/qwen4exp.h"
#include "models/qwen4exp_run.h"
#include "ops/ops.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai {

// the u32 values of those keys that the GGUF has (token ids), duplicates dropped
static std::vector<int32_t> gguf_token_ids(const std::string & path, std::initializer_list<const char *> keys) {
    std::vector<int32_t> ids;
    gguf_init_params     gp{};
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return ids;
    for (const char * key : keys) {
        const int64_t k = gguf_find_key(g, key);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_UINT32) continue;
        const int32_t v = (int32_t) gguf_get_val_u32(g, k);
        if (std::find(ids.begin(), ids.end(), v) == ids.end()) ids.push_back(v);
    }
    gguf_free(g);
    return ids;
}

std::string gguf_arch(const std::string & path) {
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = nullptr;
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) return "";
    std::string a;
    const int64_t k = gguf_find_key(g, "general.architecture");
    if (k >= 0) a = gguf_get_val_str(g, k);
    gguf_free(g);
    return a;
}

namespace {

int env_int(const char * n, int d) {
    const char * v = getenv(n);
    return v && *v ? atoi(v) : d;
}

void env_default(const char * n, const char * v) {
    setenv(n, v, /*overwrite=*/0);
}

// the only .gguf in <model dir>/MTP/ ("" when there is none or more than one)
std::string find_sidecar(const std::string & model) {
    const size_t      sl  = model.find_last_of('/');
    const std::string dir = (sl == std::string::npos ? std::string(".") : model.substr(0, sl)) + "/MTP";
    DIR *             d   = opendir(dir.c_str());
    if (!d) return "";
    std::vector<std::string> found;
    while (dirent * e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) found.push_back(dir + "/" + n);
    }
    closedir(d);
    return found.size() == 1 ? found[0] : "";
}

}  // namespace

struct Q4xServe::Impl {
    Device                       dev;
    Qwen4Exp                     prog;
    std::unique_ptr<Qwen4ExpRun> run;
    int64_t                      ctx = 0, V = 0;
    bool                         mtp = false;
    int                          mtp_k = 0, copy_min = 0, copy_k = 15;
    bool                         agree = true;
    std::vector<uint8_t>         vocab_cls;   // per-token script classes for the draft rows (empty = first rows)
    std::vector<int32_t>         hist;        // prompt + every emitted token (the suffix copy's history)
    std::vector<float>           logits;
    std::vector<int32_t>         stop_ids;    // end-of-turn tokens: a decode round commits nothing past one

    int32_t argmax_row(const Tensor & lg, int64_t row) {
        const Tensor r = lg.view(V, 1, 1, 1, (size_t) V * 4, (size_t) V * 4, (size_t) V * 4, (size_t) row * (size_t) V * 4);
        return argmax_cols(r, 1)[0];
    }

    // Greedy token of each of the T logit columns, reduced on the device (op::argmax, bit-exact with ties): T ids cross
    // PCIe instead of T x n_vocab floats, and no host scan runs with the GPU idle each round.
    // AI_Q4X_HOST_ARGMAX=1 downloads the logits and takes the argmax on the host instead (default 0).
    std::vector<int32_t> argmax_cols(const Tensor & lg, int64_t T) {
        static const bool host = env_int("AI_Q4X_HOST_ARGMAX", 0) != 0;
        std::vector<int32_t> g((size_t) T);
        if (host || !lg.contiguous()) {
            logits.resize((size_t) (V * T));
            dev.download(lg, logits.data(), 0, logits.size() * 4);
            for (int64_t a = 0; a < T; ++a) {
                const float * r = logits.data() + (size_t) a * (size_t) V;
                g[(size_t) a]   = (int32_t) (std::max_element(r, r + V) - r);
            }
            return g;
        }
        const Tensor idx = op::argmax(dev, lg);
        dev.download(idx, g.data(), 0, (size_t) T * 4);
        return g;
    }

    // the longest earlier occurrence (>= copy_min tokens, up to 16) of the history's suffix: its continuation
    bool suffix_copy(std::vector<int32_t> & toks, int * match = nullptr) const {
        const int64_t n = (int64_t) hist.size();
        for (int g = (int) std::min<int64_t>(16, n - 1); g >= copy_min; --g) {
            for (int64_t p = n - g - 1; p >= 0; --p) {
                if (!std::equal(hist.end() - g, hist.end(), hist.begin() + p)) continue;
                for (int j = 0; j < copy_k && p + g + j < n; ++j) toks.push_back(hist[(size_t) (p + g + j)]);
                if (match) *match = g;
                return toks.size() > 1;
            }
        }
        return false;
    }

    // Draft policy (AI_Q4X_SPEC_POLICY, default 1, used with the suffix copy): take a copied window only when its
    // expected committed tokens per ms beat the MTP window's by 3%. MTP side: the EMA of tokens an MTP round committed,
    // over the observed round time of its width. Copy side: E(k) = 1 + q + .. + q^k, q = the decayed acceptance of
    // copied drafts whose match was about as long (buckets < 6, < 12, >= 12 tokens; priors 0.75 / 0.88 / 0.93), over
    // the observed round time of width k + 1; the best k is taken. Widths not observed yet are scaled from observed
    // ones by a linear cost model (shape(): a fixed round cost plus a per-row verify cost) and probed a few times when
    // q >= 0.85. Without the policy a copy is taken whenever a match of copy_min tokens exists.
    struct Policy {
        double cost[kDoorbellT + 1] = {}, cost_n[kDoorbellT + 1] = {};
        double mtp_tok = 0, mtp_n = 0;
        double ok[3] = {}, bad[3] = {};
        static int    bucket(int m) { return m < 6 ? 0 : m < 12 ? 1 : 2; }
        static double shape(int t) { return 17.0 + 3.3 * t; }
        double rate(int m) const {
            static const double prior[3] = { 0.75, 0.88, 0.93 };
            const int b = bucket(m);
            return (ok[b] + 4.0 * prior[b]) / (ok[b] + bad[b] + 4.0);
        }
        double cost_ms(int t) const {
            t = std::clamp(t, 1, (int) kDoorbellT);
            if (cost_n[t] > 0) return cost[t];
            double num = 0, den = 0;
            for (int u = 1; u <= (int) kDoorbellT; ++u)
                if (cost_n[u] > 0) {
                    const double w = std::min(cost_n[u], 20.0);
                    num += w * cost[u] * shape(t) / shape(u);
                    den += w;
                }
            return den > 0 ? num / den : shape(t);
        }
        // the copy window to take (1 + drafts), or 0 for the MTP window of t_mtp rows
        int choose(int t_mtp, int k_avail, int match) const {
            const double base = (mtp_n > 0 ? mtp_tok : 1.0 + 0.7 * (t_mtp - 1)) / cost_ms(t_mtp);
            const double q    = rate(match);
            double       e = 1.0, qi = 1.0, best = 0.0;
            int          best_t = 0;
            for (int k = 1; k <= k_avail && k + 1 <= (int) kDoorbellT; ++k) {
                qi *= q;
                e += qi;
                const double r = e / cost_ms(k + 1);
                if (r > best) { best = r; best_t = k + 1; }
            }
            if (best_t > 0 && best > base * 1.03) return best_t;
            const int t_full = std::min(k_avail + 1, (int) kDoorbellT);
            if (q >= 0.85 && cost_n[t_full] < 3) return t_full;   // probe a width whose cost is only guessed
            return 0;
        }
        void observe(bool copy, int t, int accepted, int match, double ms) {
            t = std::clamp(t, 1, (int) kDoorbellT);
            cost[t] = cost_n[t] > 0 ? 0.9 * cost[t] + 0.1 * ms : ms;
            cost_n[t] += 1;
            if (copy) {
                const int b = bucket(match);
                ok[b]  = 0.97 * ok[b] + accepted;
                bad[b] = 0.97 * bad[b] + (accepted < t - 1 ? 1.0 : 0.0);
            } else {
                mtp_tok = mtp_n > 0 ? 0.95 * mtp_tok + 0.05 * (accepted + 1) : accepted + 1;
                mtp_n += 1;
            }
        }
    } policy;
};

Q4xServe::Q4xServe(const Config & c) : p_(std::make_unique<Impl>()) {
    // the runtime reads these at construction; the CLI's serving configuration is the default, any value already in
    // the environment wins
    env_default("AI_Q4X_DOORBELL", "1");
    env_default("AI_Q4X_CAPTURE", "1");
    env_default("AI_Q4X_RAM_TIER_GB", "auto");
    env_default("AI_Q4X_ADMIT", "64");
    if (!getenv("AI_Q4X_PROFILE") && getenv("ALPHA_DRE_PROFILE")) env_default("AI_Q4X_PROFILE", getenv("ALPHA_DRE_PROFILE"));

    Impl & m = *p_;
    m.ctx      = c.ctx > 0 ? c.ctx : 8192;
    m.copy_min = std::max(0, env_int("AI_Q4X_SPEC_HYBRID", 3));   // the suffix copy is on by default (min match 3; 0 = off)
    m.copy_k   = std::clamp(env_int("AI_Q4X_SPEC_COPY_K", 15), 1, (int) kDoorbellT - 1);
    m.agree    = env_int("AI_Q4X_SPEC_AGREE", 1) != 0;
    // AI_Q4X_STOP_IDS=a,b,... overrides the GGUF's eos / eot / eom ids
    if (const char * sv = getenv("AI_Q4X_STOP_IDS")) {
        for (const char * q = sv; *q;) {
            char *     e  = nullptr;
            const long id = strtol(q, &e, 10);
            if (e == q) break;
            m.stop_ids.push_back((int32_t) id);
            q = *e == ',' ? e + 1 : e;
        }
    } else {
        m.stop_ids = gguf_token_ids(c.model_path, { "tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                                    "tokenizer.ggml.eom_token_id" });
    }
    {
        std::string sl;
        for (int32_t t : m.stop_ids) sl += (sl.empty() ? "" : ",") + std::to_string(t);
        fprintf(stderr, "[ai-api] q4x end-of-turn ids (a decode round stops at one): %s\n", sl.empty() ? "none" : sl.c_str());
    }

    // own round capture: ggml's CUDA graphs off, before the device is initialised (as the CLI and Engine)
    op::set_capture(op::Capture::Own);
    if (!m.dev.init(0)) throw std::runtime_error("q4x: device init failed");
    if (!m.prog.load(m.dev, c.model_path, c.verbose)) throw std::runtime_error("q4x: model load failed: " + c.model_path);
    m.V = m.prog.spec().n_vocab;
    if (c.mtp_k > 0) {
        std::string sc = getenv("AI_Q4X_MTP") ? getenv("AI_Q4X_MTP") : find_sidecar(c.model_path);
        if (sc.empty()) throw std::runtime_error("q4x: mtp_k > 0 needs the MTP sidecar (AI_Q4X_MTP, or one .gguf in <model dir>/MTP/)");
        if (!m.prog.load_mtp(m.dev, sc, c.verbose)) throw std::runtime_error("q4x: MTP sidecar load failed: " + sc);
        m.mtp   = true;
        m.mtp_k = std::clamp(c.mtp_k, 1, 8);
        fprintf(stderr, "[ai-api] q4x MTP sidecar %s, depth %d\n", sc.c_str(), m.mtp_k);
    }

    // expert slots per MoE layer: given, or what the VRAM left holds after the reserve (workspace, KV and state,
    // the prefill's buffers, the draft head)
    const auto &         layers = m.prog.layers();
    const Qwen4ExpSpec & sp     = m.prog.spec();
    size_t               per_slot = 0, per_expert_max = 0;   // all layers' bytes of one slot; the largest layer's
    for (const auto & l : layers) {
        const size_t b = l.moe.gate_exps.nb[2] + l.moe.up_exps.nb[2] + l.moe.down_exps.nb[2];
        per_slot += b;
        per_expert_max = std::max(per_expert_max, b);
    }
    const int    k = (int) sp.n_expert_used, E = (int) sp.n_expert;
    const size_t ws    = (size_t) (c.ws_mib > 0 ? c.ws_mib : env_int("AI_Q4X_WS_MB", 640));
    int          slots = c.expert_slots;
    if (slots <= 0) {
        size_t fr = 0, tot = 0;
        m.dev.memory_info(&fr, &tot);
        // Kept free, besides the slots: the workspace; ~30 KiB per context cell (KV, indexer and state); the layer-major
        // prefill's buffers at the full super-chunk, allocated per prompt -- its state (Qwen4ExpRun::prefill_alloc's
        // sizes) and its expert scratch, which holds one layer's non-resident experts and so grows as the slot count
        // falls; and 830 MiB for what the first request leaves behind (captured rounds, verify records, the draft head)
        // plus a margin. The reserve tracks the scratch's growth so a later long prompt does not have to shrink its
        // super-chunk.
        const int64_t S   = std::max<int64_t>(8, (int64_t) env_int("AI_Q4X_PREFILL_SUPER", 16384));
        const int64_t sub = std::max<int64_t>(8, (int64_t) env_int("AI_Q4X_PREFILL_SUB", 256));
        const size_t pf_state = (size_t) sp.hc_dim() * S * 4 + (sp.ple_layers.empty() ? 0 : (size_t) sp.n_embd * S * 4) +
                                (size_t) S * 28 + (size_t) m.ctx * sub * 2 + 256;   // one device-written mask
        const bool   flat  = getenv("AI_Q4X_VRAM_RESERVE_MB") != nullptr;   // a flat reserve, as given
        const size_t fixed = flat ? (size_t) env_int("AI_Q4X_VRAM_RESERVE_MB", 0) << 20
                                  : ((ws + 830 + (size_t) (m.ctx * 30 / 1024)) << 20) + pf_state + (size_t) E * per_expert_max;
        // slots s fit when fr - s * per_slot >= fixed - s * per_expert_max (the scratch is (E - s) experts)
        const size_t per = flat ? per_slot : per_slot - per_expert_max;
        slots = per > 0 && fr > fixed ? (int) ((fr - fixed) / per) : k;
        slots = std::clamp(slots, k, E);
        fprintf(stderr, "[ai-api] q4x expert slots auto: %zu MiB free; kept free %.0f MiB (prefill state %.0f + expert "
                        "scratch %.0f MiB at this count); %.1f MiB per slot -> %d\n",
                fr >> 20, (flat ? fixed : fixed - (size_t) slots * per_expert_max) / 1048576.0, pf_state / 1048576.0,
                (double) (E - slots) * per_expert_max / 1048576.0, per_slot / 1048576.0, slots);
    }
    slots = std::clamp(slots, k, E);
    m.run = std::make_unique<Qwen4ExpRun>(m.prog, m.dev, m.ctx, slots);
    if (m.mtp) {
        m.run->mtp_enable();
        if (env_int("AI_DRAFT_VOCAB", 1) != 0) m.vocab_cls = classify_gguf_tokens(m.prog.weights().get_str_array("tokenizer.ggml.tokens"));
    }
    m.dev.init_workspace(ws << 20);
    fprintf(stderr, "[ai-api] q4x engine: 1 stream, ctx %lld, %d expert slots, workspace %zu MiB, %s; FAST decode "
                    "(not token-identical to plain decode)\n", (long long) m.ctx, slots, ws,
            !m.mtp ? "plain decode" : m.copy_min > 0 ? "MTP + suffix copy" : "MTP drafts");
}

Q4xServe::~Q4xServe() = default;

bool    Q4xServe::has_mtp() const { return p_->mtp; }
int     Q4xServe::mtp_k() const { return !p_->mtp ? 0 : p_->copy_min > 0 ? std::max(p_->mtp_k, p_->copy_k) : p_->mtp_k; }
int64_t Q4xServe::ctx_cells() const { return p_->ctx; }
int64_t Q4xServe::n_embd() const { return p_->prog.spec().n_embd; }
int64_t Q4xServe::position() const { return p_->run->position(); }

int64_t Q4xServe::free_vram_mib() const {
    size_t fr = 0, tot = 0;
    p_->dev.memory_info(&fr, &tot);
    return (int64_t) (fr >> 20);
}

int32_t Q4xServe::prefill(const int32_t * tokens, int n) {
    Impl & m = *p_;
    if (n < 1) throw std::runtime_error("q4x prefill: empty prompt");
    if ((int64_t) n + 1 >= m.ctx) throw std::runtime_error("q4x prefill: the prompt does not fit the context");
    feed_.clear();
    feed_open_ = false;
    // AI_Q4X_PREFIX_REUSE (default 1): a chat resends its history: continue from the committed tokens when the prompt
    // extends them (the previous prompt + answer), else rewind to the last prompt's checkpoint when the prompt still
    // extends that (a regenerate / an edited last turn), else start over. At least one new token is always prefilled
    // (its logits give the first token). The reused state was built by the earlier prefill + decode rounds, so the
    // text can differ from a fresh prefill's by FAST-mode rounding, as any prompt cache.
    static const bool reuse = env_int("AI_Q4X_PREFIX_REUSE", 1) != 0;
    int64_t           start = 0;
    if (reuse) {
        const std::vector<int32_t> & h = m.run->history();
        const int64_t                hn = (int64_t) h.size();
        int64_t                      common = 0;
        while (common < hn && common < n && h[(size_t) common] == tokens[common]) ++common;
        const int64_t ck = m.run->checkpoint_len();
        if (ck > 0 && n == ck && common >= ck && ck_first_ >= 0 && m.run->restore_checkpoint()) {
            // the same prompt again (a regenerate): its end state and its first token are the checkpoint's
            m.hist.assign(tokens, tokens + n);
            m.hist.push_back(ck_first_);
            reused_    = n;
            prefilled_ = true;
            fprintf(stderr, "[ai-api] q4x prefix reuse: the same %d-token prompt, nothing prefilled\n", n);
            return ck_first_;
        }
        if (hn > 0 && common == hn && n > hn) {
            start = hn;
        } else if (ck > 0 && common >= ck && n > ck && m.run->restore_checkpoint()) {
            start = ck;
        }
    }
    if (start == 0) m.run->reset();
    reused_ = start;
    if (m.mtp && env_int("AI_DRAFT_VOCAB", 1) != 0) {   // the draft head's rows from this prompt's scripts
        const int64_t              rows = env_int("AI_Q4X_DRAFT_ROWS", 131072);
        const std::vector<int32_t> ids(tokens, tokens + n);
        std::string                why;
        const std::vector<int32_t> sel = m.vocab_cls.empty() ? std::vector<int32_t>{} : draft_rows_for(m.vocab_cls, ids, rows, &why);
        m.run->set_draft_rows(sel, rows);
    }
    const Tensor  lg    = m.run->prefill(tokens + start, n - start);
    const int32_t first = m.argmax_row(lg, lg.ne[1] - 1);
    if (reuse) {   // the state at the end of this prompt: the rewind point of the next request
        m.run->checkpoint();
        ck_first_ = first;
    }
    if (start > 0) fprintf(stderr, "[ai-api] q4x prefix reuse: %lld of %d prompt tokens kept, %lld prefilled\n",
                           (long long) start, n, (long long) (n - start));
    m.hist.assign(tokens, tokens + n);
    m.hist.push_back(first);
    prefilled_ = true;
    return first;
}

int32_t Q4xServe::prefill_feed(const int32_t * tokens, int n, bool is_last) {
    if (!feed_open_) {
        feed_.clear();
        feed_open_ = true;
        prefilled_ = false;   // a new request: the slot is not decodable until its prompt is in
    }
    if (n > 0) feed_.insert(feed_.end(), tokens, tokens + n);
    if (!is_last) return -1;
    std::vector<int32_t> prompt;
    prompt.swap(feed_);
    feed_open_ = false;
    return prefill(prompt.data(), (int) prompt.size());
}

int32_t Q4xServe::decode_plain(int32_t pending) {
    Impl & m = *p_;
    if (m.run->position() + 1 >= m.ctx) throw std::runtime_error("q4x decode: context full");
    const Tensor  lg  = m.run->decode(pending);
    const int32_t tok = m.argmax_row(lg, 0);
    m.hist.push_back(tok);
    return tok;
}

void Q4xServe::decode_spec(int32_t pending, std::vector<int32_t> & out) {
    Impl & m = *p_;
    out.clear();
    static const bool use_policy = env_int("AI_Q4X_SPEC_POLICY", 1) != 0;
    const auto        t0         = std::chrono::steady_clock::now();
    std::vector<int32_t> toks{ pending };
    int  match  = 0;
    bool copied = m.copy_min > 0 && m.suffix_copy(toks, &match);
    std::vector<int32_t> d;
    if (!copied || m.agree) d = m.run->mtp_draft(pending, m.mtp_k);
    if (copied && m.agree) {   // keep the copy only where the MTP agrees on its first token
        if (d.empty() || d[0] != toks[1]) { toks.resize(1); copied = false; }
    }
    // The policy judges only short matches (< 6 tokens): spurious 3-5 token repeats in fresh text lose when always
    // copied, while the long matches of edits and quotes pay off even where the policy would price them out.
    if (copied && use_policy && match < 6) {   // the copy only where it beats the MTP window on expected tokens per ms
        const int t = m.policy.choose(1 + (int) d.size(), (int) toks.size() - 1, match);
        if (t == 0) { toks.resize(1); copied = false; }
        else toks.resize((size_t) t);
    }
    if (copied) d.clear();
    toks.insert(toks.end(), d.begin(), d.end());
    // never past the context: the verify writes a KV cell per row
    const int64_t room = m.ctx - 1 - m.run->position();
    if (room < 1) throw std::runtime_error("q4x decode: context full");
    if ((int64_t) toks.size() > room) toks.resize((size_t) room);
    const int64_t T  = (int64_t) toks.size();
    const Tensor  lg = m.run->verify(toks.data(), T);
    const std::vector<int32_t> gq = m.argmax_cols(lg, T);
    int64_t a = 0;
    for (; a < T; ++a) {
        const int32_t g = gq[(size_t) a];
        out.push_back(g);
        // an end-of-turn token ends the round: nothing past it is committed, so the kept state is exactly what a chat
        // resends next (prompt + answer through the stop) and the next prefill extends it. Otherwise accepted drafts
        // past the stop would be committed, the next prompt would no longer extend the history, and the previous
        // answer would be prefilled again.
        if (std::find(m.stop_ids.begin(), m.stop_ids.end(), g) != m.stop_ids.end()) break;
        if (a + 1 >= T || toks[(size_t) (a + 1)] != g) break;   // the next draft is accepted only if it equals g
    }
    m.run->commit((int64_t) out.size());
    m.hist.insert(m.hist.end(), out.begin(), out.end());
    if (use_policy)
        m.policy.observe(copied, (int) T, (int) out.size() - 1, match,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void Q4xServe::reset() {
    feed_.clear();
    feed_open_ = false;
    prefilled_ = false;
    p_->hist.clear();
    // With AI_Q4X_PREFIX_REUSE the state is kept for the next prompt (prefill decides what of it to reuse); a stream is
    // still not decodable until that prefill
    if (env_int("AI_Q4X_PREFIX_REUSE", 1) == 0) p_->run->reset();
}

}  // namespace ai
