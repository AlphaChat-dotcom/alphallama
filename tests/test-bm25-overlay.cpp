// Post-finalize overlay test for tools/server/server-bm25.h (bm25_index).
//
// Documents added after the corpus is finalized (df dictionary frozen, and in inverted mode the postings
// immutable) go to the overlay. This test builds a synthetic corpus in each ingest mode the server uses,
// finalizes it, adds documents afterwards, and checks:
//   (a) a post-finalize document is the top hit for a term only it contains, and its text reads back;
//   (b) rankings and scores equal those of a fresh index built with every document from the start;
//   (c) with no post-finalize adds, scores are bit-identical to a reference BM25 that mirrors the scorer
//       formula, so the empty overlay changes nothing;
//   (d) 4 reader threads query while 1 writer adds 200 docs post-finalize: no crash, well-formed reader
//       results, and the final state equals the same adds done sequentially.
//
// Standalone CPU program (POSIX). Build from the repo root:
//   g++ -std=c++17 -O2 -ffp-contract=off -pthread -Itools/server tests/test-bm25-overlay.cpp -o test-bm25-overlay
// -ffp-contract=off keeps the reference scorer and the index from contracting to FMA differently.

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <thread>
#include <atomic>
#include <unistd.h>

#include "server-bm25.h"

static int g_fail = 0;

static void check(bool ok, const std::string & name) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) g_fail++;
}

// Synthetic corpus

static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd() {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}

static const int VOCAB = 400;
static std::string vocab_word(int i) {
    // letters only, so no stemming / hyphen splitting surprises; distinct for distinct i
    std::string w = "w";
    int x = i;
    do { w += (char)('a' + x % 26); x /= 26; } while (x > 0);
    return w + "x";
}
// Skewed draw: low ids are common, high ids are rare.
static int draw_word() {
    uint32_t r = rnd() % 1000000;
    double u = r / 1000000.0;
    return (int)(u * u * VOCAB) % VOCAB;
}
static std::string make_doc(int len) {
    std::string s;
    for (int i = 0; i < len; i++) {
        if (i) s += ' ';
        s += vocab_word(draw_word());
    }
    return s;
}

// Index builders (mirror server-context.cpp)

enum class mode { RESIDENT_INCREMENTAL, CORPUS_FORWARD, CORPUS_FORWARD_MAPDF, CORPUS_INVERTED, CORPUS_INVERTED_SPILL };

static const char * mode_name(mode m) {
    switch (m) {
        case mode::RESIDENT_INCREMENTAL:  return "resident-incremental";
        case mode::CORPUS_FORWARD:        return "corpus-dir forward (streaming df)";
        case mode::CORPUS_FORWARD_MAPDF:  return "corpus-dir forward (map df)";
        case mode::CORPUS_INVERTED:       return "corpus-dir inverted (resident inversion)";
        case mode::CORPUS_INVERTED_SPILL: return "corpus-dir inverted (external spill)";
    }
    return "?";
}

static std::string tmp_prefix(const char * tag) {
    static int seq = 0;
    const char * td = getenv("TMPDIR"); std::string tmpdir = (td && *td) ? td : "/tmp";
    return tmpdir + "/alpha_bm25_ovtest_" + std::to_string((long)getpid()) + "_" + tag + "_" + std::to_string(seq++);
}

using doc_list = std::vector<std::pair<std::string, std::string>>;   // (text, source)

// Build an index over `docs` the way the given mode does at startup, finalized. chunk_ids are 0..n-1.
static std::unique_ptr<bm25_index> build_corpus(mode m, const doc_list & docs) {
    std::unique_ptr<bm25_index> idx(new bm25_index());
    if (m == mode::RESIDENT_INCREMENTAL) {
        // HTTP-only server: no --corpus-dir, nothing is ever finalized.
        for (size_t i = 0; i < docs.size(); i++) idx->add_document((int)i, docs[i].first, docs[i].second);
        return idx;
    }
    const std::string pfx = tmp_prefix("c");
    idx->open_text_store(pfx + ".text.bin");
    if (m == mode::CORPUS_INVERTED_SPILL) idx->open_postings_spill(pfx + ".spill");
    if (m != mode::CORPUS_FORWARD_MAPDF) idx->enable_df_streaming();
    // A few single adds first (the experience-file path), then batches, as the ingest loop does.
    size_t i = 0;
    for (; i < 3 && i < docs.size(); i++) idx->add_document((int)i, docs[i].first, docs[i].second);
    while (i < docs.size()) {
        size_t n = std::min<size_t>(37, docs.size() - i);
        doc_list batch(docs.begin() + i, docs.begin() + i + n);
        idx->add_documents_batch((int)i, batch);
        i += n;
    }
    idx->finalize_text_store();
    idx->finalize_text_offsets();
    idx->finalize_df();
    if (m == mode::CORPUS_INVERTED || m == mode::CORPUS_INVERTED_SPILL) idx->build_inverted(pfx + ".post.bin");
    return idx;
}

// Rankings: normalised to (score desc, chunk_id asc) so tie order does not depend on the sort used.
using ranking = std::vector<std::pair<float, int>>;
static ranking normalise(ranking r) {
    std::sort(r.begin(), r.end(), [](const std::pair<float,int> & a, const std::pair<float,int> & b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });
    return r;
}
// Matches only (score > 0): the forward scorer also returns score-0 docs, the inverted scorer does not.
static ranking matches(const ranking & r) {
    ranking out;
    for (auto & p : r) if (p.first > 0.0f) out.push_back(p);
    return out;
}
static bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
static bool same_ranking(const ranking & a, const ranking & b, size_t topn) {
    size_t n = std::min(topn, std::max(a.size(), b.size()));
    if (a.size() < n || b.size() < n) return false;
    for (size_t i = 0; i < n; i++)
        if (a[i].second != b[i].second || !same_bits(a[i].first, b[i].first)) return false;
    return true;
}

// Reference BM25 (c)
// Mirrors the scorer formula term for term, from the raw texts, with df counted independently. Without
// post-finalize adds the index must produce exactly these scores.
static ranking reference_scores(const bm25_index & idx, const doc_list & docs, const std::string & query,
                                bool only_matches) {
    const float k1 = idx.k1, b = idx.b;
    std::vector<std::unordered_map<uint64_t, int>> tf(docs.size());
    std::vector<int> nwords(docs.size());
    std::unordered_map<uint64_t, int> df;
    long long total_words = 0;
    for (size_t d = 0; d < docs.size(); d++) {
        auto words = bm25_index::tokenize(docs[d].first);
        nwords[d] = (int)words.size();
        total_words += nwords[d];
        for (auto & w : words) tf[d][idx.cipher.hash_term(w)]++;
        for (auto & kv : tf[d]) df[kv.first]++;
    }
    const int N = (int)docs.size();
    const float avgdl = (float)total_words / std::max(1, N);
    struct qt_t { uint64_t h; float idf; bool valid; };
    std::vector<qt_t> qts;
    for (auto & w : bm25_index::tokenize(query, nullptr, true)) {
        uint64_t h = idx.cipher.hash_term(w);
        auto it = df.find(h);
        int dfc = it != df.end() ? it->second : 0;
        float idf = dfc > 0 ? std::log((N - dfc + 0.5f) / (dfc + 0.5f) + 1.0f) : 0.0f;
        qts.push_back({h, idf, dfc > 0});
    }
    ranking out;
    for (size_t d = 0; d < docs.size(); d++) {
        float s = 0; bool hit = false;
        for (auto & q : qts) {
            if (!q.valid) continue;
            auto it = tf[d].find(q.h);
            if (it == tf[d].end()) continue;
            int t = it->second;
            float tf_norm = (t * (k1 + 1)) / (t + k1 * (1 - b + b * nwords[d] / avgdl));
            s += q.idf * tf_norm * 1.0f;
            hit = true;
        }
        if (!only_matches || hit) out.push_back({s, (int)d});
    }
    return normalise(out);
}

static std::vector<std::string> make_queries(int n) {
    std::vector<std::string> q;
    for (int i = 0; i < n; i++) {
        int len = 1 + (int)(rnd() % 4);
        std::string s;
        for (int j = 0; j < len; j++) {
            if (j) s += ' ';
            // mid-to-rare vocabulary so the top-10 is discriminative, plus some common words
            int id = (j == 0) ? (int)(rnd() % VOCAB) : draw_word();
            s += vocab_word(id);
        }
        q.push_back(s);
    }
    return q;
}

int main() {
    // The flags are function-local statics read on first use: set them before any bm25_index call.
    // ALPHA_BM25_INVERTED only gates build_inverted(), so forward modes simply never call it.
    setenv("ALPHA_BM25_INVERTED", "1", 1);
    unsetenv("ALPHA_BM25_BIGRAM_FIRST");
    unsetenv("ALPHA_BM25_P1");
    unsetenv("ALPHA_BM25_STEM");
    unsetenv("ALPHA_BM25_DICT_RESIDENT");     // inverted spill mode mmaps the dictionary (the default)

    // Base corpus: 200 docs. Post-finalize docs: 5 docs with a unique term each, plus shared vocabulary.
    doc_list base;
    for (int i = 0; i < 200; i++)
        base.push_back({make_doc(20 + (int)(rnd() % 41)), "file" + std::to_string(i / 4) + ".txt@" + std::to_string((i % 4) * 1000)});
    doc_list added;
    const char * uniq[5] = { "zyxquuxa", "zyxquuxb", "zyxquuxc", "zyxquuxd", "zyxquuxe" };
    for (int i = 0; i < 5; i++) {
        std::string t = make_doc(15 + (int)(rnd() % 20));
        t += " " + std::string(uniq[i]) + " " + vocab_word(i * 37 % VOCAB) + " " + std::string(uniq[i]);
        added.push_back({t, "added" + std::to_string(i) + ".txt"});
    }
    doc_list all = base;
    all.insert(all.end(), added.begin(), added.end());
    const std::vector<std::string> queries = make_queries(40);

    const mode modes[] = { mode::RESIDENT_INCREMENTAL, mode::CORPUS_FORWARD, mode::CORPUS_FORWARD_MAPDF,
                           mode::CORPUS_INVERTED, mode::CORPUS_INVERTED_SPILL };
    for (mode m : modes) {
        const std::string mn = mode_name(m);
        const bool inv = (m == mode::CORPUS_INVERTED || m == mode::CORPUS_INVERTED_SPILL);
        printf("\n== %s ==\n", mn.c_str());

        // (c) no post-finalize adds: bit-identical to the reference, overlay empty
        {
            auto idx = build_corpus(m, base);
            if (inv) check(idx->inverted_active, mn + ": inverted index active");
            check(idx->ov_df.empty() && idx->ov_chunk.empty(), mn + ": (c) overlay empty without post-finalize adds");
            check(idx->N == (int)base.size(), mn + ": (c) N unchanged");
            bool all_same = true; int bad = 0;
            for (auto & q : queries) {
                ranking got = normalise(idx->score(q, 0));
                ranking ref = reference_scores(*idx, base, q, inv);
                if (got.size() != ref.size() || !same_ranking(got, ref, got.size())) { all_same = false; bad++; }
            }
            check(all_same, mn + ": (c) scores bit-identical to reference on " + std::to_string(queries.size()) +
                            " queries (" + std::to_string(bad) + " differ)");
        }

        // Post-finalize adds: 3 single adds (HTTP path) + one batch of 2.
        auto idx = build_corpus(m, base);
        const int first = (int)base.size();
        for (int i = 0; i < 3; i++) idx->add_document(first + i, added[i].first, added[i].second);
        idx->add_documents_batch(first + 3, doc_list(added.begin() + 3, added.end()));
        if (m != mode::RESIDENT_INCREMENTAL) {
            check(!idx->ov_df.empty(), mn + ": overlay df populated after post-finalize adds");
            check(inv ? idx->ov_chunk.size() == added.size() : idx->ov_chunk.empty(),
                  mn + (inv ? ": overlay holds the added docs' tf" : ": forward mode keeps added docs in the forward arrays"));
        }
        check(idx->N == (int)all.size(), mn + ": N includes post-finalize docs");

        // (a) unique term -> the added doc is the top (and only) hit; text and source read back
        for (int i = 0; i < 5; i++) {
            auto top = idx->top_k(uniq[i], 10);
            check(top.size() == 1 && top[0] == first + i,
                  mn + ": (a) unique term '" + uniq[i] + "' -> chunk " + std::to_string(first + i) +
                  " (got " + (top.empty() ? std::string("none") : std::to_string(top[0])) + ", " + std::to_string(top.size()) + " hits)");
            check(idx->df_of(idx->cipher.hash_term(uniq[i])) == 1, mn + ": (a) df of '" + uniq[i] + "' is 1");
        }
        {
            bool texts_ok = true;
            for (int i = 0; i < 5; i++)
                if (idx->get_text(first + i) != added[i].first || idx->get_source(first + i) != added[i].second) texts_ok = false;
            for (int i : {0, 1, 63, 64, 128, 199})
                if (idx->get_text(i) != base[i].first || idx->get_source(i) != base[i].second) texts_ok = false;
            check(texts_ok, mn + ": (a) text/source of added and base docs read back");
        }
        // unique term + shared words: the added doc still wins
        {
            auto top = idx->top_k(std::string(uniq[2]) + " " + vocab_word(3) + " " + vocab_word(10), 5);
            check(!top.empty() && top[0] == first + 2, mn + ": (a) unique+shared query ranks the added doc first");
        }

        // (b) same rankings and scores as a fresh index with every doc from the start
        {
            auto fresh = build_corpus(m, all);
            check(std::fabs(fresh->avgdl - idx->avgdl) == 0.0f && fresh->N == idx->N, mn + ": (b) N and avgdl equal fresh index");
            int bad10 = 0, badall = 0, bad_bf = 0;
            std::vector<std::string> qs = queries;
            for (int i = 0; i < 5; i++) qs.push_back(added[i].first.substr(0, 40));   // queries hitting added docs
            for (auto & q : qs) {
                ranking a = matches(normalise(idx->score(q, 0)));
                ranking f = matches(normalise(fresh->score(q, 0)));
                if (!same_ranking(a, f, 10)) bad10++;
                if (a.size() != f.size() || !same_ranking(a, f, a.size())) badall++;
                ranking ab = idx->score(q, 1), fb = fresh->score(q, 1);       // bigram-first (anchored) arm
                // anchored order is (anchor, score, chunk_id) in inverted mode; forward ties are unordered,
                // so compare the matches as sets of (score, chunk_id) plus the leader
                ranking abm = matches(ab), fbm = matches(fb);
                bool bf_ok = abm.size() == fbm.size()
                          && same_ranking(normalise(abm), normalise(fbm), abm.size())
                          && (abm.empty() || (same_bits(abm[0].first, fbm[0].first)));
                if (inv) bf_ok = bf_ok && same_ranking(abm, fbm, abm.size());
                if (!bf_ok) bad_bf++;
            }
            check(bad10 == 0, mn + ": (b) top-10 order and scores equal fresh index (" + std::to_string(bad10) + "/" +
                              std::to_string(qs.size()) + " differ)");
            check(badall == 0, mn + ": (b) full match list equal fresh index (" + std::to_string(badall) + " differ)");
            check(bad_bf == 0, mn + ": (b) bigram-first arm equal fresh index (" + std::to_string(bad_bf) + " differ)");
        }
    }

    // (d) concurrency: 4 reader threads query while 1 writer adds 200 docs post-finalize. No crash, every
    // reader result is well-formed, and the final state equals the same adds done sequentially.
    for (mode m : { mode::CORPUS_INVERTED, mode::CORPUS_INVERTED_SPILL, mode::CORPUS_FORWARD }) {
        const std::string mn = std::string(mode_name(m)) + " [concurrent]";
        printf("\n== %s ==\n", mn.c_str());
        doc_list late;
        std::vector<std::string> late_uniq;
        for (int i = 0; i < 200; i++) {
            std::string u = "zq";
            int x = i; do { u += (char)('a' + x % 26); x /= 26; } while (x > 0);
            u += "qz";
            late_uniq.push_back(u);
            late.push_back({make_doc(10 + (int)(rnd() % 30)) + " " + u, "late" + std::to_string(i) + ".txt"});
        }
        const int first = (int)base.size();
        const int last_cid = first + (int)late.size();
        // Writer: singles and batches of 8, alternating, in chunk-id order.
        auto do_adds = [&](bm25_index & ix) {
            int i = 0;
            while (i < (int)late.size()) {
                if ((i / 8) % 2 == 0) {
                    for (int j = 0; j < 8 && i < (int)late.size(); j++, i++)
                        ix.add_document(first + i, late[i].first, late[i].second);
                } else {
                    int n = std::min(8, (int)late.size() - i);
                    ix.add_documents_batch(first + i, doc_list(late.begin() + i, late.begin() + i + n));
                    i += n;
                }
            }
        };

        auto idx = build_corpus(m, base);
        std::atomic<bool> done{false};
        std::atomic<int>  bad{0};
        std::atomic<long> reads{0};
        std::vector<std::thread> readers;
        for (int t = 0; t < 4; t++) {
            readers.emplace_back([&, t]() {
                uint32_t r = 12345u + (uint32_t)t;
                auto nx = [&]() { r = r * 1664525u + 1013904223u; return r >> 8; };
                while (!done.load()) {
                    const std::string & q = queries[nx() % queries.size()];
                    ranking s = idx->score(q, (int)(nx() % 2));
                    for (auto & p : s) if (!(p.first >= 0.0f) || p.second < 0 || p.second >= last_cid) bad++;
                    // a late doc's unique term finds exactly that doc, or nothing if it is not added yet
                    int li = (int)(nx() % late.size());
                    auto top = idx->top_k(late_uniq[li], 5);
                    if (!(top.empty() || (top.size() == 1 && top[0] == first + li))) bad++;
                    if (!top.empty() && idx->get_text(first + li) != late[li].first) bad++;
                    if (!top.empty() && idx->get_source(first + li) != late[li].second) bad++;
                    int bi = (int)(nx() % base.size());
                    if (idx->get_text(bi) != base[bi].first) bad++;
                    (void)idx->df_of(idx->cipher.hash_term(late_uniq[li]));
                    (void)idx->get_idf_keywords(bi, 5);
                    (void)idx->chunk_id_at(nx() % (size_t)last_cid);
                    reads++;
                }
            });
        }
        std::thread writer([&]() { do_adds(*idx); done = true; });
        writer.join();
        for (auto & th : readers) th.join();
        check(bad.load() == 0, mn + ": readers saw only well-formed results during adds (" +
                               std::to_string(reads.load()) + " read rounds, " + std::to_string(bad.load()) + " bad)");

        auto seq = build_corpus(m, base);
        do_adds(*seq);
        check(idx->N == seq->N && idx->avgdl == seq->avgdl, mn + ": final N/avgdl equal sequential adds");
        int diff = 0;
        std::vector<std::string> qs = queries;
        for (int i = 0; i < 200; i += 17) qs.push_back(late_uniq[i] + " " + late[i].first.substr(0, 30));
        for (auto & q : qs) {
            ranking a = matches(normalise(idx->score(q, 0))), s = matches(normalise(seq->score(q, 0)));
            if (a.size() != s.size() || !same_ranking(a, s, a.size())) diff++;
        }
        bool texts = true;
        for (int i = 0; i < 200; i++)
            if (idx->get_text(first + i) != late[i].first || idx->top_k(late_uniq[i], 5) != std::vector<int>{first + i}) texts = false;
        check(diff == 0, mn + ": final rankings equal sequential adds (" + std::to_string(diff) + "/" +
                         std::to_string(qs.size()) + " differ)");
        check(texts, mn + ": every late doc found by its unique term and its text reads back");
    }

    printf("\n%s: %d failure(s)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
