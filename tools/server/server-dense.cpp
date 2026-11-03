// Dense-L0 semantic retrieval backend implementation (behind ALPHA_DENSE_L0).

#ifdef ALPHA_DENSE_L0

#include "server-dense.h"

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>

// Split text into overlapping sentence windows (~target_chars each). Each window is a
// focused unit: embedding a >512-token chunk whole gives a truncated average, so each
// window is embedded and mapped back to the chunk id. Falls back to a single (capped)
// window when there are no sentence breaks.
static std::vector<std::string> sentence_windows(const std::string & text,
                                                 int target_chars = 400, int max_windows = 16) {
    std::vector<std::string> sents;
    size_t start = 0;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if ((c == '.' || c == '!' || c == '?') &&
            (i + 1 >= text.size() || text[i + 1] == ' ' || text[i + 1] == '\n' || text[i + 1] == '\t')) {
            sents.push_back(text.substr(start, i - start + 1));
            start = i + 1;
        }
    }
    if (start < text.size()) sents.push_back(text.substr(start));
    if (sents.empty()) return { text };

    std::vector<std::string> wins;
    size_t i = 0;
    while (i < sents.size() && (int) wins.size() < max_windows) {
        std::string w; size_t j = i;
        while (j < sents.size() && (int) w.size() < target_chars) { w += sents[j]; j++; }
        size_t a = w.find_first_not_of(" \n\t");
        if (a != std::string::npos) { w = w.substr(a); if (!w.empty()) wins.push_back(w); }
        i = (j > i + 1) ? (j - 1) : j;   // 1-sentence overlap
    }
    if (wins.empty()) wins.push_back(text.substr(0, (size_t) target_chars));
    return wins;
}

dense_index::~dense_index() {
    if (emb_ctx)   { llama_free(emb_ctx);         emb_ctx   = nullptr; }
    if (emb_model) { llama_model_free(emb_model); emb_model = nullptr; }
}

bool dense_index::load_embedder(const std::string & model_path, int n_gpu_layers) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    emb_model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!emb_model) {
        fprintf(stderr, "dense-l0: failed to load embedding model '%s'\n", model_path.c_str());
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_CLS;   // bge-family default
    // n_ctx is the total token budget per decode; many sequences (each <= n_seq_pos) are packed
    // into one decode for ingest throughput. n_ubatch = n_ctx so a whole group pools in one ubatch.
    cparams.n_ctx        = n_ctx;
    cparams.n_batch      = n_ctx;
    cparams.n_ubatch     = n_ctx;
    cparams.n_seq_max    = n_seq_max;
    // Embed threads: llama's default is 4. With no GPU device (CPU-only build or no visible GPU) a
    // bge-base query embed takes 22.7 ms at 4 threads and 14.5 ms at 8, so use min(cores, 8) there.
    // A GPU-resident embedder is unaffected by the thread count and keeps the default.
    // ALPHA_DENSE_L0_EMBED_THREADS overrides both.
    if (const char * ethreads = getenv("ALPHA_DENSE_L0_EMBED_THREADS")) {
        const int t = atoi(ethreads);
        if (t > 0) {
            cparams.n_threads       = t;
            cparams.n_threads_batch = t;
            fprintf(stderr, "[embed-threads] n_threads = %d (ALPHA_DENSE_L0_EMBED_THREADS)\n", t);
        } else {
            fprintf(stderr, "[embed-threads] ignoring ALPHA_DENSE_L0_EMBED_THREADS=%s (not a positive integer)\n",
                    ethreads);
        }
    } else if (!llama_supports_gpu_offload()) {
        const int hw = (int) std::thread::hardware_concurrency();
        const int t  = hw > 0 ? std::min(hw, 8) : 4;
        cparams.n_threads       = t;
        cparams.n_threads_batch = t;
        fprintf(stderr, "[embed-threads] no GPU device: n_threads = %d (min(cores=%d, 8))\n", t, hw);
    }

    emb_ctx = llama_init_from_model(emb_model, cparams);
    if (!emb_ctx) {
        fprintf(stderr, "dense-l0: failed to create embedding context\n");
        llama_model_free(emb_model);
        emb_model = nullptr;
        return false;
    }

    n_embd = llama_model_n_embd_out(emb_model);
    // identity of the embedder file for the persisted store (one sequential read, ~0.5 s per 200 MB)
    {
        FILE * mf = fopen(model_path.c_str(), "rb");
        uint64_t h0 = 0xcbf29ce484222325ULL, h1 = 0x84222325cbf29ce4ULL, nb = 0;
        if (mf) {
            std::vector<unsigned char> buf(1 << 22);
            size_t r;
            while ((r = fread(buf.data(), 1, buf.size(), mf)) > 0) {
                for (size_t i = 0; i < r; i++) {
                    h0 = (h0 ^ buf[i]) * 0x100000001b3ULL;
                    h1 = (h1 ^ (unsigned char) (buf[i] + 0x5b)) * 0x100000001b3ULL;
                }
                nb += r;
            }
            fclose(mf);
        }
        model_hash[0] = h0; model_hash[1] = h1; model_bytes = nb;
    }
    fprintf(stderr, "dense-l0: embedder loaded (dim=%d, batch_ctx=%d, max_seq=%d, id %016llx%016llx) from %s\n",
            n_embd, n_ctx, n_seq_max, (unsigned long long) model_hash[0], (unsigned long long) model_hash[1],
            model_path.c_str());
    return true;
}

// Truncate a tokenised text to the model's max sequence length without losing the terminal special token the
// tokenizer appended ([SEP] for bge/BERT, or EOS). Without it the CLS-pooled vector of a long document is badly off.
// This matches how sentence-transformers truncates.
static void dense_truncate(std::vector<llama_token> & toks, int n_max, const llama_vocab * vocab) {
    if ((int) toks.size() <= n_max || n_max < 2) return;
    llama_token tail = -1;
    if (llama_vocab_get_add_sep(vocab))      tail = llama_vocab_sep(vocab);
    else if (llama_vocab_get_add_eos(vocab)) tail = llama_vocab_eos(vocab);
    if (tail >= 0 && toks.back() == tail) { toks.resize(n_max - 1); toks.push_back(tail); }
    else                                  { toks.resize(n_max); }
}

std::vector<float> dense_index::embed(const std::string & text, bool is_query) const {
    if (!emb_ctx || n_embd <= 0) return {};
    // One decode on the shared context; it must be serialised for correctness.
    std::lock_guard<std::mutex> emb_lk(emb_mu);

    const std::string in = is_query ? (std::string(DENSE_QUERY_PREFIX) + text) : text;

    std::vector<llama_token> tokens = common_tokenize(emb_ctx, in, /*add_special*/ true, /*parse_special*/ false);
    if (tokens.empty()) return {};
    dense_truncate(tokens, n_seq_pos, llama_model_get_vocab(emb_model));   // model max seq len, terminal [SEP] kept

    // Each embed is an independent sequence; clear the pooled/KV memory first.
    llama_memory_clear(llama_get_memory(emb_ctx), true);

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        // logits=true on the final token so the pooled sequence embedding is produced.
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, i + 1 == tokens.size());
    }

    std::vector<float> out;
    if (llama_decode(emb_ctx, batch) == 0) {
        const float * emb = llama_get_embeddings_seq(emb_ctx, 0);
        if (emb) {
            out.resize(n_embd);
            common_embd_normalize(emb, out.data(), n_embd, /*embd_norm L2*/ 2);
        }
    } else {
        fprintf(stderr, "dense-l0: llama_decode failed while embedding\n");
    }

    llama_batch_free(batch);
    return out;
}

std::vector<std::pair<uint32_t, float>> dense_index::search_semantic(const std::string & query, int k) const {
    return search_vec(embed(query, /*is_query*/ true), k);
}

void dense_index::reserve(size_t n) {
    std::unique_lock<std::shared_mutex> lk(store_mu);
    ids.reserve(n);
    if (n_embd > 0) vecs.reserve(n * (size_t) n_embd);
}

// On-disk format (little-endian, fixed width, no compression):
//   "ALPHADV2" | int32 n_embd | uint64 count
//     | uint64 model_hash[2] | uint64 model_bytes | int32 pooling | int32 n_seq_pos | uint32 flags | uint64 prefix_hash
//     | uint32 ids[count] | float vecs[count * n_embd]
// flags: bit0 L2-normalised, bit1 sentence-window. V1 ("ALPHADV1", no identity block) is refused unless
// ALPHA_DENSE_L0_ACCEPT_V1=1: a store's vectors are only meaningful under the exact embedder and config that wrote
// them, and a mismatched store cannot be detected at query time (cosine statistics look normal while retrieval
// quality collapses).
static const char ALPHA_DV_MAGIC[8]  = { 'A','L','P','H','A','D','V','2' };
static const char ALPHA_DV1_MAGIC[8] = { 'A','L','P','H','A','D','V','1' };

namespace {
struct dv_identity {
    uint64_t model_hash[2];
    uint64_t model_bytes;
    int32_t  pooling;
    int32_t  n_seq_pos;
    uint32_t flags;
    uint64_t prefix_hash;
};
uint64_t fnv64(const char * s) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (; *s; ++s) h = (h ^ (unsigned char) *s) * 0x100000001b3ULL;
    return h;
}
}  // namespace

bool dense_index::save_vectors(const std::string & path) const {
    std::shared_lock<std::shared_mutex> lk(store_mu);   // a torn snapshot would persist corruption
    if (n_embd <= 0 || ids.empty()) return false;
    if (vecs.size() != ids.size() * (size_t) n_embd) return false;   // never persist a torn store
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const int32_t  dim = (int32_t)  n_embd;
    const uint64_t cnt = (uint64_t) ids.size();
    const dv_identity id = { { model_hash[0], model_hash[1] }, model_bytes, (int32_t) LLAMA_POOLING_TYPE_CLS,
                             (int32_t) n_seq_pos, 1u | (windows ? 2u : 0u), fnv64(DENSE_QUERY_PREFIX) };
    bool ok = fwrite(ALPHA_DV_MAGIC, 1, 8, f) == 8
           && fwrite(&dim, sizeof(dim), 1, f) == 1
           && fwrite(&cnt, sizeof(cnt), 1, f) == 1
           && fwrite(id.model_hash, sizeof(uint64_t), 2, f) == 2
           && fwrite(&id.model_bytes, sizeof(id.model_bytes), 1, f) == 1
           && fwrite(&id.pooling, sizeof(id.pooling), 1, f) == 1
           && fwrite(&id.n_seq_pos, sizeof(id.n_seq_pos), 1, f) == 1
           && fwrite(&id.flags, sizeof(id.flags), 1, f) == 1
           && fwrite(&id.prefix_hash, sizeof(id.prefix_hash), 1, f) == 1
           && fwrite(ids.data(),  sizeof(uint32_t), ids.size(),  f) == ids.size()
           && fwrite(vecs.data(), sizeof(float),    vecs.size(), f) == vecs.size();
    if (fclose(f) != 0) ok = false;
    if (!ok) { remove(tmp.c_str()); return false; }
    // rename is atomic on the same filesystem, so a reader never sees a partial file
    if (rename(tmp.c_str(), path.c_str()) != 0) { remove(tmp.c_str()); return false; }
    fprintf(stderr, "dense-l0: persisted %zu vectors (dim %d, %.1f MB) -> %s\n",
            ids.size(), n_embd,
            (8 + 4 + 8 + 44 + ids.size() * 4.0 + vecs.size() * 4.0) / 1e6, path.c_str());
    return true;
}

bool dense_index::load_vectors(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8] = {0};
    int32_t  dim = 0;
    uint64_t cnt = 0;
    if (fread(magic, 1, 8, f) != 8) {
        fprintf(stderr, "dense-l0: %s is not a dense store (short file); ignoring\n", path.c_str());
        fclose(f); return false;
    }
    const bool v1 = memcmp(magic, ALPHA_DV1_MAGIC, 8) == 0;
    if (!v1 && memcmp(magic, ALPHA_DV_MAGIC, 8) != 0) {
        fprintf(stderr, "dense-l0: %s is not a dense store (bad magic); ignoring\n", path.c_str());
        fclose(f); return false;
    }
    if (v1) {
        const char * acc = getenv("ALPHA_DENSE_L0_ACCEPT_V1");
        if (!(acc && atoi(acc) != 0)) {
            fprintf(stderr, "dense-l0: %s is a V1 store with NO embedder identity; REFUSING to load (it would be served "
                            "even if written by a different model -- re-embed, or set ALPHA_DENSE_L0_ACCEPT_V1=1 only if "
                            "you know this store came from this exact embedder and config)\n", path.c_str());
            fclose(f); return false;
        }
        fprintf(stderr, "dense-l0: WARNING loading V1 store %s WITHOUT identity check (ALPHA_DENSE_L0_ACCEPT_V1=1)\n",
                path.c_str());
    }
    if (fread(&dim, sizeof(dim), 1, f) != 1 || fread(&cnt, sizeof(cnt), 1, f) != 1) {
        fprintf(stderr, "dense-l0: %s header truncated; ignoring\n", path.c_str());
        fclose(f); return false;
    }
    if (!v1) {
        dv_identity got{};
        const bool hok = fread(got.model_hash, sizeof(uint64_t), 2, f) == 2
                      && fread(&got.model_bytes, sizeof(got.model_bytes), 1, f) == 1
                      && fread(&got.pooling, sizeof(got.pooling), 1, f) == 1
                      && fread(&got.n_seq_pos, sizeof(got.n_seq_pos), 1, f) == 1
                      && fread(&got.flags, sizeof(got.flags), 1, f) == 1
                      && fread(&got.prefix_hash, sizeof(got.prefix_hash), 1, f) == 1;
        if (!hok) {
            fprintf(stderr, "dense-l0: %s identity block truncated; REFUSING to load\n", path.c_str());
            fclose(f); return false;
        }
        const uint32_t want_flags = 1u | (windows ? 2u : 0u);
        const char * bad = nullptr;
        char detail[200] = {0};
        if (got.model_hash[0] != model_hash[0] || got.model_hash[1] != model_hash[1] || got.model_bytes != model_bytes) {
            bad = "embedder model";
            snprintf(detail, sizeof(detail), "store %016llx%016llx (%llu B) vs loaded %016llx%016llx (%llu B)",
                     (unsigned long long) got.model_hash[0], (unsigned long long) got.model_hash[1],
                     (unsigned long long) got.model_bytes, (unsigned long long) model_hash[0],
                     (unsigned long long) model_hash[1], (unsigned long long) model_bytes);
        } else if (got.pooling != (int32_t) LLAMA_POOLING_TYPE_CLS) {
            bad = "pooling";
            snprintf(detail, sizeof(detail), "store %d vs %d", got.pooling, (int) LLAMA_POOLING_TYPE_CLS);
        } else if (got.n_seq_pos != (int32_t) n_seq_pos) {
            bad = "n_seq_pos (truncation length)";
            snprintf(detail, sizeof(detail), "store %d vs %d", got.n_seq_pos, n_seq_pos);
        } else if ((got.flags & 1u) != (want_flags & 1u)) {
            bad = "L2 normalisation";
            snprintf(detail, sizeof(detail), "store flags %u vs %u", got.flags, want_flags);
        } else if ((got.flags & 2u) != (want_flags & 2u)) {
            bad = "sentence-window mode (ALPHA_DENSE_L0_SW)";
            snprintf(detail, sizeof(detail), "store %s vs %s", (got.flags & 2u) ? "on" : "off", windows ? "on" : "off");
        } else if (got.prefix_hash != fnv64(DENSE_QUERY_PREFIX)) {
            bad = "query prefix";
            snprintf(detail, sizeof(detail), "prefix hash differs");
        }
        if (bad) {
            fprintf(stderr, "dense-l0: %s was written under a different %s (%s); REFUSING to load -- its vectors would be "
                            "served as if they came from this embedder\n", path.c_str(), bad, detail);
            fclose(f); return false;
        }
    }
    // The embedder must already be loaded, so a store written under a different embedder is refused
    // rather than reinterpreted. n_embd == 0 means no embedder yet -> refuse, do not guess.
    if (n_embd <= 0 || dim != (int32_t) n_embd) {
        fprintf(stderr, "dense-l0: %s dim %d != embedder dim %d; REFUSING to load\n",
                path.c_str(), (int) dim, n_embd);
        fclose(f); return false;
    }
    if (cnt == 0 || cnt > (uint64_t) 1e9) {
        fprintf(stderr, "dense-l0: %s implausible count %llu; ignoring\n",
                path.c_str(), (unsigned long long) cnt);
        fclose(f); return false;
    }
    // Read into locals, so a truncated file leaves the live store untouched.
    std::vector<uint32_t> nids((size_t) cnt);
    std::vector<float>    nvecs((size_t) cnt * (size_t) dim);
    const bool ok = fread(nids.data(),  sizeof(uint32_t), nids.size(),  f) == nids.size()
                 && fread(nvecs.data(), sizeof(float),    nvecs.size(), f) == nvecs.size();
    // A trailing byte means the file is not what the header describes.
    const bool at_end = (fgetc(f) == EOF);
    fclose(f);
    if (!ok || !at_end) {
        fprintf(stderr, "dense-l0: %s truncated or oversized vs its header; REFUSING to load\n",
                path.c_str());
        return false;
    }
    {
        std::unique_lock<std::shared_mutex> lk(store_mu);
        ids.swap(nids);
        vecs.swap(nvecs);
        restored = true;
        ann_active = false;
        ann_built_n = 0;   // ids/vecs were swapped wholesale: every ann_lists index is now meaningless
    }
    // The IVF centroids are not persisted. Clear ann_active so the next search brute-forces
    // instead of consulting an index that was never built for these vectors.
    ann_active = false;
    fprintf(stderr, "dense-l0: restored %zu vectors (dim %d) from %s -- NO re-embed needed\n",
            ids.size(), n_embd, path.c_str());
    return true;
}

bool dense_index::has_vec(uint32_t chunk_id) const {
    std::shared_lock<std::shared_mutex> lk(store_mu);
    for (size_t i = 0; i < ids.size(); i++) if (ids[i] == chunk_id) return true;
    return false;
}

bool dense_index::get_vec(uint32_t chunk_id, std::vector<float> & out) const {
    std::shared_lock<std::shared_mutex> lk(store_mu);
    if (n_embd <= 0) return false;
    for (size_t i = 0; i < ids.size(); i++) {
        if (ids[i] != chunk_id) continue;
        const size_t off = i * (size_t) n_embd;
        if (off + (size_t) n_embd > vecs.size()) return false;      // store mid-append; treat as miss
        out.assign(vecs.begin() + off, vecs.begin() + off + n_embd);
        return true;
    }
    return false;
}

void dense_index::add_vec_locked(uint32_t chunk_id, const std::vector<float> & v) {
    if ((int) v.size() != n_embd || n_embd <= 0) return;
    if (!ids.empty() && chunk_id <= ids.back()) ids_monotonic = false;
    ids.push_back(chunk_id);
    vecs.insert(vecs.end(), v.begin(), v.end());

    // An append does not invalidate the IVF: ann_lists holds indices into these append-only buffers,
    // so everything built stays indexed and this vector joins the un-indexed tail, which search_ann
    // scans exhaustively. The tail is only cheap while it is small, so past a bound the index is
    // marked stale and search falls back to brute force.
    if (ann_active) {
        size_t cap = std::max((size_t) 65536, ann_built_n / 8);   // 12.5% of the built size
        if (const char * e = getenv("ALPHA_DENSE_L0_ANN_DELTA_MAX")) {
            long v = atol(e);
            if (v > 0) cap = (size_t) v;
        }
        if (ids.size() - ann_built_n > cap) {
            ann_active = false;   // tail too large to scan: index is stale, rebuild needed
        }
    }
}

void dense_index::add_vec(uint32_t chunk_id, const std::vector<float> & v) {
    // Exclusive: push_back/insert reallocate, and search_vec indexes into these same buffers.
    std::unique_lock<std::shared_mutex> lk(store_mu);
    add_vec_locked(chunk_id, v);
}

void dense_index::add(uint32_t chunk_id, const std::string & text) {
    std::vector<float> v = embed(text, /*is_query*/ false);
    if (!v.empty()) add_vec(chunk_id, v);
}

std::vector<std::vector<float>> dense_index::embed_batch(const std::vector<std::string> & texts, bool is_query) const {
    std::vector<std::vector<float>> out(texts.size());
    if (!emb_ctx || n_embd <= 0 || texts.empty()) return out;

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    size_t i = 0;
    while (i < texts.size()) {
        // Lock per group (= per decode), not per call. Holding emb_mu for a whole 2048-document
        // ingest batch would stall every concurrent query for seconds; per group the worst a query
        // waits is one decode. The scope ends at the bottom of this loop body, including on the
        // `continue` paths below.
        std::lock_guard<std::mutex> emb_lk(emb_mu);
        // Build one packed group of sequences (bounded by token budget n_ctx and n_seq_max).
        llama_memory_clear(llama_get_memory(emb_ctx), true);
        common_batch_clear(batch);
        std::vector<size_t> group;      // original indices, in seq_id order
        int cur_tok = 0;
        while (i < texts.size() && (int) group.size() < n_seq_max) {
            const std::string in = is_query ? (std::string(DENSE_QUERY_PREFIX) + texts[i]) : texts[i];
            std::vector<llama_token> toks = common_tokenize(emb_ctx, in, /*add_special*/ true, /*parse_special*/ false);
            if (toks.empty()) { i++; continue; }                 // out[i] stays empty
            dense_truncate(toks, n_seq_pos, llama_model_get_vocab(emb_model));
            if (cur_tok + (int) toks.size() > n_ctx) {
                if (group.empty()) toks.resize((size_t) std::max(1, n_ctx));  // one oversized text
                else break;                                       // flush; retry this text next group
            }
            const int seq = (int) group.size();
            for (int t = 0; t < (int) toks.size(); t++)
                common_batch_add(batch, toks[t], (llama_pos) t, { seq }, t + 1 == (int) toks.size());
            cur_tok += (int) toks.size();
            group.push_back(i);
            i++;
        }
        if (group.empty()) continue;
        if (llama_decode(emb_ctx, batch) != 0) continue;         // whole group failed -> leave empty
        for (int s = 0; s < (int) group.size(); s++) {
            const float * emb = llama_get_embeddings_seq(emb_ctx, s);
            if (emb) {
                out[group[s]].resize(n_embd);
                common_embd_normalize(emb, out[group[s]].data(), n_embd, /*L2*/ 2);
            }
        }
    }
    llama_batch_free(batch);
    return out;
}

void dense_index::add_batch(const std::vector<std::pair<uint32_t, std::string>> & items) {
    // A restored store already holds these vectors. Without this the corpus pass would embed the whole
    // corpus again on top of the restored one and double the store, and the duplicates would look
    // like a healthy larger index.
    if (restored) return;
    if (items.empty()) return;
    // Expand to (window_text, chunk_id) pairs (sentence-window mode) or 1:1.
    std::vector<std::string> texts;
    std::vector<uint32_t>     owner;
    texts.reserve(items.size());
    owner.reserve(items.size());
    for (const auto & it : items) {
        if (windows) {
            for (auto & w : sentence_windows(it.second)) { texts.push_back(std::move(w)); owner.push_back(it.first); }
        } else {
            texts.push_back(it.second); owner.push_back(it.first);
        }
    }
    // This local `vecs` shadows the member of the same name. embed_batch takes emb_mu per
    // decode internally and has released it before the store lock is taken below.
    std::vector<std::vector<float>> vecs = embed_batch(texts, /*is_query*/ false);
    {
        // One exclusive acquisition for the whole batch rather than one per vector.
        std::unique_lock<std::shared_mutex> lk(store_mu);
        for (size_t k = 0; k < texts.size(); k++)
            if ((int) vecs[k].size() == n_embd) add_vec_locked(owner[k], vecs[k]);
    }
}

// Dot product with EIGHT independent accumulators.
//
// The obvious `for (j) d += q[j]*r[j]` is a single dependency chain -- each add waits on the
// previous (~4 cycles) -- so the ALU stalls no matter how wide the ISA is, which is why
// -mavx2/-mfma measured under 1% on this loop. Eight partial sums let the pipeline overlap:
// 276.3 -> 124.3 ns/vector at n=57,600, a 2.22x win, plateauing at 8 exactly as FP-add latency
// predicts (16 is no better). A single-threaded BLAS GEMV reaches 89.0 ns/vector on the same core,
// so there is headroom left here in CODE, not in bytes.
//
// \u26a0 The summation ORDER differs from the one-accumulator form, so results differ in the last
// bits. Validated by nDCG@10 and a top-k set-change count, not assumed harmless.
//
// n_embd is whatever the embedder reports -- never assumed to be a multiple of 8.
static inline float alpha_dot8(const float * __restrict a, const float * __restrict b, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f, s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    int j = 0;
    const int n8 = n & ~7;
    for (; j < n8; j += 8) {
        s0 += a[j + 0] * b[j + 0];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
        s4 += a[j + 4] * b[j + 4];
        s5 += a[j + 5] * b[j + 5];
        s6 += a[j + 6] * b[j + 6];
        s7 += a[j + 7] * b[j + 7];
    }
    float tail = 0.0f;
    for (; j < n; ++j) tail += a[j] * b[j];
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)) + tail;
}

std::vector<std::pair<uint32_t, float>> dense_index::search_vec(const std::vector<float> & qv, int k) const {
    std::vector<std::pair<uint32_t, float>> out;
    // Shared: concurrent queries do not block each other, but an append cannot realloc underneath.
    std::shared_lock<std::shared_mutex> lk(store_mu);
    if ((int) qv.size() != n_embd || n_embd <= 0 || ids.empty()) return out;

    if (ann_active) return search_ann(qv, k);   // sub-linear IVF path past ~1M vectors
    // A store at or above the ANN threshold being brute-forced is almost always a forgotten build (HTTP ingest without
    // POST /api/dense/build_ann): every query then pays a full linear scan. Warn once.
    if (ids.size() >= (size_t) ann_min()) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            fprintf(stderr, "dense-l0: WARNING brute-force search over %zu vectors (>= ANN threshold %d): no IVF index is "
                            "built -- after an HTTP ingest call POST /api/dense/build_ann. Warned once.\n",
                    ids.size(), ann_min());
        }
    }

    // Best cosine per chunk_id: in sentence-window mode many vectors share a chunk id,
    // so keep the strongest window per chunk and return distinct chunk ids.
    const size_t n = ids.size();
    std::vector<std::pair<uint32_t, float>> scored;
    if (ids_monotonic) {
        // Every id is distinct, so there is no "strongest window per chunk" to resolve: score
        // straight into the result vector and skip the per-document hash lookup entirely.
        scored.resize(n);
        for (size_t d = 0; d < n; ++d) {
            // vectors are L2-normalized -> dot == cosine
            scored[d] = { ids[d], alpha_dot8(qv.data(), &vecs[d * (size_t) n_embd], n_embd) };
        }
    } else {
        std::unordered_map<uint32_t, float> best;
        best.reserve(n);
        for (size_t d = 0; d < n; ++d) {
            // vectors are L2-normalized -> dot == cosine
            const float dot = alpha_dot8(qv.data(), &vecs[d * (size_t) n_embd], n_embd);
            auto it = best.find(ids[d]);
            if (it == best.end()) best.emplace(ids[d], dot);
            else if (dot > it->second) it->second = dot;
        }
        scored.assign(best.begin(), best.end());
    }

    // A TOTAL order: (cosine desc, chunk_id asc). Score alone is not a total order, and neither
    // std::sort nor std::partial_sort is stable, so with score-only comparison a cosine tie makes
    // the returned top-k arbitrary -- and at the k boundary it changes the SET, not just the
    // order. Measured on identical vectors: 35 of 50 positions differed between two correct
    // implementations. bm25_index::score_inverted already sorts (score desc, chunk_id asc).
    const auto cmp = [](const std::pair<uint32_t, float> & a, const std::pair<uint32_t, float> & b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    };
    if (k <= 0 || (size_t) k >= scored.size()) {
        std::sort(scored.begin(), scored.end(), cmp);
        return scored;
    }
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(), cmp);
    out.assign(scored.begin(), scored.begin() + k);
    return out;
}

std::vector<std::pair<uint32_t, float>> dense_index::search(const std::string & query, int k) const {
    return search_vec(embed(query, /*is_query*/ true), k);
}

// IVF ANN (coarse inverted file): keeps dense search sub-linear past ~1M vectors

int dense_index::ann_min() {
    // 1,048,576 vectors; brute force is fine below this. The default is not a measured crossover;
    // ALPHA_DENSE_L0_ANN_MIN overrides it.
    static const int v = [] {
        const char * e = getenv("ALPHA_DENSE_L0_ANN_MIN");
        const int x = e ? atoi(e) : 0;
        return x > 0 ? x : (1 << 20);
    }();
    return v;
}

void dense_index::build_ann(int nlist_hint) {
    const int ANN_MIN = ann_min();
    std::unique_lock<std::shared_mutex> lk(store_mu);   // writes the IVF arrays, reads vecs
    const size_t n = ids.size();
    ann_active = false; centroids.clear(); ann_lists.clear(); ann_nlist = 0;
    if (n_embd <= 0 || (int) n < ANN_MIN) return;
    if (const char * np = getenv("ALPHA_DENSE_L0_NPROBE")) { int v = atoi(np); if (v > 0) ann_nprobe = v; }

    int nlist = nlist_hint > 0 ? nlist_hint : (int) std::sqrt((double) n);
    // ALPHA_DENSE_L0_NLIST overrides the list count (recall tracks the number of lists probed, so more, smaller
    // lists buy recall at the same scan)
    if (nlist_hint <= 0) {
        if (const char * nl = getenv("ALPHA_DENSE_L0_NLIST")) { int v = atoi(nl); if (v > 0) nlist = v; }
    }
    nlist = std::max(64, std::min(nlist, 65536));

    // Centroids start as evenly-strided sampled vectors (already L2-normalized) and are then refined by
    // the training iterations below.
    // IVF search is approximate: recall@10 against exhaustive search rises with nprobe and, at a fixed number of
    // vectors examined, falls as n grows. ALPHA_DENSE_L0_NPROBE is the quality/speed knob.
    centroids.assign((size_t) nlist * n_embd, 0.0f);
    for (int c = 0; c < nlist; c++) {
        size_t src = (size_t) ((double) c * (double) n / (double) nlist);
        const float * row = &vecs[src * (size_t) n_embd];
        std::copy(row, row + n_embd, &centroids[(size_t) c * n_embd]);
    }

    // Assign every vector to its nearest centroid (multithreaded; this is the build cost, O(n*nlist*d)).
    std::vector<int> assign(n, 0);
    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4; if (nthreads > 32) nthreads = 32;
    auto worker = [&](size_t lo, size_t hi) {
        for (size_t d = lo; d < hi; d++) {
            const float * v = &vecs[d * (size_t) n_embd];
            int bestc = 0; float bests = -2.0f;
            for (int c = 0; c < nlist; c++) {
                const float * cen = &centroids[(size_t) c * n_embd];
                float s = 0.0f;
                for (int j = 0; j < n_embd; j++) s += v[j] * cen[j];
                if (s > bests) { bests = s; bestc = c; }
            }
            assign[d] = bestc;
        }
    };
    std::vector<std::thread> pool;
    size_t chunk = (n + nthreads - 1) / nthreads;
    for (unsigned t = 0; t < nthreads; t++) {
        size_t lo = (size_t) t * chunk, hi = std::min(n, lo + chunk);
        if (lo < hi) pool.emplace_back(worker, lo, hi);
    }
    for (auto & th : pool) th.join();

    // Refine the centroids with spherical Lloyd. The sampled centroids above are only an initialisation;
    // a few iterations improve recall substantially and it saturates within ~6. Each iteration is one more
    // assignment pass, so build cost is (iters+1)x that step. With a fixed nprobe and nlist = sqrt(n) the
    // examined share shrinks as the corpus grows, so training matters more for large stores.
    // ALPHA_DENSE_L0_TRAIN=0 skips training.
    int train_iters = 3;
    if (const char * ti = getenv("ALPHA_DENSE_L0_TRAIN")) train_iters = std::max(0, atoi(ti));
    for (int it = 0; it < train_iters; it++) {
        std::vector<float>  sum((size_t) nlist * n_embd, 0.0f);
        std::vector<size_t> cnt(nlist, 0);
        for (size_t d = 0; d < n; d++) {
            const float * v = &vecs[d * (size_t) n_embd];
            float * acc = &sum[(size_t) assign[d] * n_embd];
            for (int j = 0; j < n_embd; j++) acc[j] += v[j];
            cnt[assign[d]]++;
        }
        for (int c = 0; c < nlist; c++) {
            if (cnt[c] == 0) continue;                       // an empty list keeps its old centroid
            float * cen = &centroids[(size_t) c * n_embd];
            const float * acc = &sum[(size_t) c * n_embd];
            float nrm = 0.0f;
            for (int j = 0; j < n_embd; j++) nrm += acc[j] * acc[j];
            if (nrm <= 1e-12f) continue;
            nrm = 1.0f / std::sqrt(nrm);                     // re-normalise: this is cosine space
            for (int j = 0; j < n_embd; j++) cen[j] = acc[j] * nrm;
        }
        const std::vector<int> prev = assign;
        pool.clear();
        for (unsigned t = 0; t < nthreads; t++) {
            size_t lo = (size_t) t * chunk, hi = std::min(n, lo + chunk);
            if (lo < hi) pool.emplace_back(worker, lo, hi);
        }
        for (auto & th : pool) th.join();
        if (assign == prev) break;                           // converged
    }

    // Log the effective configuration.
    fprintf(stderr, "dense-l0: IVF centroid training: %d iteration(s) requested\n", train_iters);

    ann_lists.assign(nlist, {});
    for (size_t d = 0; d < n; d++) ann_lists[assign[d]].push_back((uint32_t) d);
    ann_nlist = nlist;
    ann_built_n = n;      // everything beyond this index is the un-indexed tail (see search_ann)
    ann_active = true;
    fprintf(stderr, "dense-l0: IVF ANN built (%zu vecs, %d lists, nprobe=%d)\n", n, nlist, ann_nprobe);
}

std::vector<std::pair<uint32_t, float>> dense_index::search_ann(const std::vector<float> & qv, int k) const {
    std::vector<std::pair<uint32_t, float>> out;
    if ((int) qv.size() != n_embd || !ann_active) return out;

    // 1) rank centroids by cosine, keep the nearest nprobe.
    std::vector<std::pair<float, int>> cs; cs.reserve(ann_nlist);
    for (int c = 0; c < ann_nlist; c++) {
        const float * cen = &centroids[(size_t) c * n_embd];
        float s = 0.0f;
        for (int j = 0; j < n_embd; j++) s += qv[j] * cen[j];
        cs.emplace_back(s, c);
    }
    int probe = std::min(ann_nprobe, ann_nlist);
    std::partial_sort(cs.begin(), cs.begin() + probe, cs.end(),
                      [](const auto & a, const auto & b){ return a.first > b.first; });

    // 2) scan only the probed lists, best cosine per chunk_id.
    std::unordered_map<uint32_t, float> best;
    for (int p = 0; p < probe; p++) {
        for (uint32_t d : ann_lists[cs[p].second]) {
            const float * row = &vecs[(size_t) d * n_embd];
            float dot = 0.0f;
            for (int j = 0; j < n_embd; j++) dot += qv[j] * row[j];
            auto it = best.find(ids[d]);
            if (it == best.end()) best.emplace(ids[d], dot);
            else if (dot > it->second) it->second = dot;
        }
    }

    // 3) the un-indexed tail, scanned exhaustively. Vectors appended after build_ann() are in no
    // list, so without this they would be silently unfindable. Same arithmetic as the brute-force
    // path, just over a contiguous suffix.
    for (size_t d = ann_built_n; d < ids.size(); ++d) {
        const float * row = &vecs[d * (size_t) n_embd];
        float dot = 0.0f;
        for (int j = 0; j < n_embd; j++) dot += qv[j] * row[j];
        auto it = best.find(ids[d]);
        if (it == best.end()) best.emplace(ids[d], dot);
        else if (dot > it->second) it->second = dot;
    }
    std::vector<std::pair<uint32_t, float>> scored(best.begin(), best.end());
    if (k <= 0 || (size_t) k >= scored.size()) {
        std::sort(scored.begin(), scored.end(), [](const auto & a, const auto & b){ return a.second > b.second; });
        return scored;
    }
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto & a, const auto & b){ return a.second > b.second; });
    out.assign(scored.begin(), scored.begin() + k);
    return out;
}

#endif // ALPHA_DENSE_L0
