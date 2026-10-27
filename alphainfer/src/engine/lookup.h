// Prompt-lookup drafting core, shared by every Program's engine (also used by the Flash-Next suffix drafter).
// A stream type S provides: hist (std::vector<int32_t>), lk (std::vector<std::unordered_map<uint64_t, int32_t>>)
// and lk_done (int64_t).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace ai {

// Prompt-lookup drafting beside MTP, enabled by AI_LOOKUP=1 (default 0). After each round the stream's own history
// (prompt + emitted tokens) is searched for the most recent earlier occurrence of its longest suffix n-gram
// (AI_LOOKUP_NMAX down to AI_LOOKUP_NMIN tokens); what followed it becomes the next round's drafts, verified like
// MTP's, so exact mode stays exact (its width cap applies). No match: MTP's drafts as before. A single hit suffices;
// requiring a quorum was slower. Batch 1 only for now.
//   AI_LOOKUP_AGREE=N (default 3): take the lookup only when its first N tokens equal the drafter's (MTP or DFlash2).
//                    On chat a weak match would otherwise displace better MTP drafts.
//   slow start:      the proposal length starts at AI_LOOKUP_L0 (8) and doubles after every fully accepted lookup
//                    round, up to K; a partial accept resets it.
//   AI_LOOKUP_ARM=N (default 8): shadow mode -- lookup drafts are used only after a tracked (unused) lookup continuation
//                    matched N emitted tokens in a row; rounds accepting < 1/4 disarm it. Off-copy text then runs
//                    exactly as the plain drafter. 0 = always armed.
//   AI_LOOKUP_ARM_DF2=N (default -1 = AI_LOOKUP_ARM): the arm under DFlash2. Arm 0 helps copy-heavy prompts but hurts
//                    non-copy outputs, so it is not the default.
//   AI_LOOKUP_DISARM=N (default 2): disarm after N rounds that accepted under a quarter of their real lookup drafts
//                    (counted since the last fully accepted lookup round). The value is not sensitive.
//   AI_LOOKUP_START=N (default 0): a request starts armed (1 = under DFlash2 only, 2 = both drafters). Unlike arm 0 the
//                    disarm still works and re-arming takes the normal shadow run, so the start of a copy is caught
//                    without staying armed on non-copy output.
//   AI_LOOKUP_HOLD=N (default 1 = plain doubling), AI_LOOKUP_TOP=W (64): growing the proposal length past W needs N
//                    consecutive fully accepted lookup rounds, since a partially accepted wide round is expensive.
// Widths are bucketed {chain, 8, 16, 32, 64, 128} (<= K) so few graph profiles are captured.
struct LookupCfg {
    int on, nmin, nmax, minlen, agree, l0, arm, arm_df2, disarm, start, top, hold;
};
inline const LookupCfg & lookup_cfg() {
    static const LookupCfg c = []() {
        auto iv = [](const char * k, int d) { const char * e = getenv(k); return e ? atoi(e) : d; };
        return LookupCfg{ iv("AI_LOOKUP", 0), iv("AI_LOOKUP_NMIN", 2), iv("AI_LOOKUP_NMAX", 6), iv("AI_LOOKUP_MINLEN", 1),
                          iv("AI_LOOKUP_AGREE", 3), iv("AI_LOOKUP_L0", 8),
                          iv("AI_LOOKUP_ARM", 8), iv("AI_LOOKUP_ARM_DF2", -1),
                          iv("AI_LOOKUP_DISARM", 2), iv("AI_LOOKUP_START", 0),
                          iv("AI_LOOKUP_TOP", 64), iv("AI_LOOKUP_HOLD", 1) };
    }();
    return c;
}
inline int lk_bucket(int n, int chain, int K) {
    if (n <= chain) return chain;
    for (int b : { 8, 16, 32, 64, 128 }) {
        if (n <= b) return std::min(b, K);
    }
    return K;
}
inline uint64_t ngram_hash(const int32_t * p, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) {
        h ^= (uint32_t) p[i];
        h *= 1099511628211ull;
        h ^= h >> 29;
    }
    return h;
}
// index every n-gram ending at a position whose successor is known (end <= size - 2)
template <class S> inline void lk_extend(S & s, int nmin, int nmax) {
    if (s.lk.size() != (size_t) (nmax + 1)) {
        s.lk.assign((size_t) (nmax + 1), {});
        s.lk_done = 0;
    }
    const int64_t L = (int64_t) s.hist.size();
    for (int64_t e = s.lk_done; e <= L - 2; ++e) {
        for (int n = nmin; n <= nmax && e - n + 1 >= 0; ++n) {
            s.lk[(size_t) n][ngram_hash(&s.hist[(size_t) (e - n + 1)], n)] = (int32_t) e;   // most recent wins
        }
    }
    s.lk_done = std::max<int64_t>(s.lk_done, L - 1);
}
// the continuation of the longest suffix match, up to maxlen tokens; returns the match length (0 = none)
template <class S> inline int lk_propose(const S & s, int nmin, int nmax, int maxlen, std::vector<int32_t> & out) {
    out.clear();
    const int64_t L = (int64_t) s.hist.size();
    for (int n = (int) std::min<int64_t>(nmax, L - 1); n >= nmin; --n) {
        const int32_t * key = &s.hist[(size_t) (L - n)];
        const auto      it  = s.lk[(size_t) n].find(ngram_hash(key, n));
        if (it == s.lk[(size_t) n].end()) continue;
        const int64_t e = it->second;
        if (memcmp(&s.hist[(size_t) (e - n + 1)], key, (size_t) n * sizeof(int32_t)) != 0) continue;   // collision
        for (int64_t j = e + 1; j < L && (int) out.size() < maxlen; ++j) out.push_back(s.hist[(size_t) j]);
        if (!out.empty()) return n;
    }
    return 0;
}

}  // namespace ai
