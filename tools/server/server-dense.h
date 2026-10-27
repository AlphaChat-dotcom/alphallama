#pragma once

// Dense-L0 semantic retrieval backend (behind ALPHA_DENSE_L0).
//
// Covers paraphrase queries that lexical BM25 misses: a low-echo query ("fraud-like
// language") shares few words with its answer. This module owns a dedicated embedding
// llama_context (separate from the main generation model) plus a contiguous
// normalized-vector store with brute-force or IVF cosine search.
//
// Works with any GGUF embedding model (bge-base by default). Off by default.

#ifdef ALPHA_DENSE_L0

#include <string>
#include <vector>
#include <cstdint>
#include <utility>
#include <mutex>
#include <shared_mutex>   // C++17

struct llama_model;
struct llama_context;

class dense_index {
public:
    dense_index() = default;
    ~dense_index();

    dense_index(const dense_index &) = delete;
    dense_index & operator=(const dense_index &) = delete;

    // Load the embedding model (e.g. bge-base-en-v1.5 GGUF) as a dedicated embedding
    // context (embeddings=true, pooling=CLS by default). Returns false on failure.
    bool load_embedder(const std::string & model_path, int n_gpu_layers = 999);
    bool ready()  const { return emb_model && emb_ctx; }
    int  dim()    const { return n_embd; }
    // Takes the shared lock: this is read by the query path's `dense.size() > 0` guard while an
    // ingest may be appending. No dense_index method calls it internally, so it cannot recurse.
    size_t size() const { std::shared_lock<std::shared_mutex> lk(store_mu); return ids.size(); }

    // Semantic search used by the RAG union: query-embedding hits.
    std::vector<std::pair<uint32_t, float>> search_semantic(const std::string & query, int k) const;

    // Embed one text (pooled, L2-normalized). `is_query` prepends the bge retrieval
    // instruction (queries only; passages raw). Returns empty vector on failure.
    std::vector<float> embed(const std::string & text, bool is_query) const;

    // Add a chunk: embed the passage and store its normalized vector under chunk_id.
    void add(uint32_t chunk_id, const std::string & text);
    // Add a precomputed normalized vector (bulk/threaded ingest); dim must match.
    void add_vec(uint32_t chunk_id, const std::vector<float> & v);

    // Read back a stored vector by chunk id. The class otherwise exposes only search(), so a caller
    // cannot rescore a candidate list it already has, which query-time dense reranking over a
    // lexical candidate pool needs. Linear scan over `ids`; a caller doing this per candidate per
    // query should build its own id->slot map from ids() once.
    bool has_vec(uint32_t chunk_id) const;
    bool get_vec(uint32_t chunk_id, std::vector<float> & out) const;

    // Persist / restore the vector store so a restart does not re-embed the whole corpus.
    // save_vectors writes <path>.tmp then renames, so an interrupted save cannot leave a file a
    // later boot would load. load_vectors refuses a bad magic, a dim mismatch, a truncated file, or
    // a size that disagrees with the header, and leaves the store untouched on refusal. A
    // successful load clears ann_active: the IVF centroids are not persisted, so the next search
    // falls back to brute force.
    bool save_vectors(const std::string & path) const;
    bool load_vectors(const std::string & path);
    // True once load_vectors() has succeeded. add_batch() declines while this is set, so the corpus
    // pass does not re-embed on top of a restored store and double it. Singular add() is unaffected,
    // so incremental HTTP additions still work.
    bool restored_from_disk() const { return restored; }
    void reserve(size_t n);

    // Embed many texts in as few llama_decode calls as possible; packs multiple
    // sequences per decode (up to the token/seq budget). One normalized vector per input.
    std::vector<std::vector<float>> embed_batch(const std::vector<std::string> & texts, bool is_query) const;
    // Batched ingest: embed all items at once and store. When sentence-window mode is on,
    // each chunk is split into sentence windows and each window is embedded under the chunk's
    // id; a >512-token chunk otherwise embeds only its truncated average. search dedups back
    // to one id per chunk.
    void add_batch(const std::vector<std::pair<uint32_t, std::string>> & items);

    // Sentence-window mode. Off = one whole-chunk embedding per chunk.
    void set_windows(bool w) { windows = w; }
    bool windows_enabled() const { return windows; }

    // Search: embed the query, cosine (== dot of normalized vectors) vs all stored
    // vectors, return top-k (chunk_id, score) sorted descending.
    std::vector<std::pair<uint32_t, float>> search(const std::string & query, int k) const;
    std::vector<std::pair<uint32_t, float>> search_vec(const std::vector<float> & qv, int k) const;

    // Build an IVF (inverted-file) ANN index so a query scans only the nearest `nprobe`
    // clusters instead of all N vectors, keeping dense search sub-linear past ~1M vectors.
    // Approximate (tune recall/speed via ALPHA_DENSE_L0_NPROBE). No-op below ANN_MIN.
    // Called after ingest; invalidated by any later add_vec (forces rebuild).
    void build_ann(int nlist = 0);
    bool ann_ready()  const { return ann_active; }
    // True when the IVF covers the whole store (no un-indexed tail): a rebuild would change nothing.
    bool ann_current() const { std::shared_lock<std::shared_mutex> lk(store_mu); return ann_active && ann_built_n == ids.size(); }
    // The size below which build_ann() is a no-op (brute force). ALPHA_DENSE_L0_ANN_MIN overrides the default 1<<20.
    static int ann_min();
    int  ann_nlists() const { return ann_nlist; }

private:
    llama_model *   emb_model = nullptr;
    llama_context * emb_ctx   = nullptr;
    int n_embd     = 0;
    bool restored  = false;  // set by load_vectors(); makes add_batch() decline (see the header note)
    // Embedder identity, written into a persisted store and checked on load: a 128-bit content hash of the
    // GGUF actually loaded (two FNV-1a-64 streams over the whole file) and its byte size. dim alone identifies nothing:
    // 768 is shared by bge-base, e5-base, gte-base and all-mpnet-base-v2.
    uint64_t model_hash[2] = { 0, 0 };
    uint64_t model_bytes   = 0;
    int n_seq_pos  = 512;    // bge-base max sequence length (per-text truncation)
    int n_ctx      = 2048;   // total token budget per decode (packs a few sequences)
    int n_seq_max  = 16;     // max sequences packed per decode
    bool windows   = false;  // sentence-window chunking, opt-in: coherent passages that fit the
                             // embedder window match a whole-passage paraphrase better than fragments.
                             // Helps only when the retrieval unit dilutes the answer (large multi-topic chunks).

    // Concurrency: every dense entry point runs on an httplib worker thread, so all state below is shared.
    //   emb_mu   guards emb_ctx. A llama_context cannot be used concurrently at all: embed() and
    //            embed_batch() both llama_memory_clear() it and then llama_decode(), so without this
    //            one thread wipes another's in-flight batch and reads back the wrong sequence.
    //   store_mu guards vecs/ids and the IVF arrays. Shared so concurrent queries do not serialise
    //            against each other; appends take it exclusively. Required for memory safety, not
    //            just freshness: ids.push_back/vecs.insert reallocate while search_vec is indexing
    //            into the same buffers.
    // Lock order: emb_mu is always released before store_mu is taken; never held together.
    mutable std::mutex        emb_mu;
    mutable std::shared_mutex store_mu;

    // Append without taking store_mu -- the caller must already hold it exclusively. Exists so
    // add_batch can append a whole batch under one lock instead of one per vector.
    void add_vec_locked(uint32_t chunk_id, const std::vector<float> & v);

    std::vector<float>    vecs;   // size() * n_embd, row-major, L2-normalized
    std::vector<uint32_t> ids;

    // IVF ANN index (optional; built by build_ann when size() >= ANN_MIN).
    std::vector<float>                 centroids;   // ann_nlist * n_embd (L2-normalized samples)
    std::vector<std::vector<uint32_t>> ann_lists;   // per centroid: indices into vecs/ids
    int  ann_nlist  = 0;
    // Lists probed per query (ALPHA_DENSE_L0_NPROBE overrides). A small nprobe makes crossing ANN_MIN cost
    // noticeable recall versus brute force; 128 keeps recall@10 around 0.97 of exhaustive at ~4x the speed.
    // A fixed nprobe keeps the scan at 128*sqrt(n), so the speedup grows with n at the price of a slow recall
    // drift (about 0.5 pp per doubling). nlist/8 would hold recall but scan n/8, i.e. linear like brute force.
    int  ann_nprobe = 128;
    bool ann_active = false;
    // Store size when the IVF was built. ann_lists holds indices into vecs/ids and the store is
    // append-only, so [0, ann_built_n) stays exactly indexed and [ann_built_n, ids.size()) is the
    // un-indexed tail, which search_ann scans exhaustively. See add_vec_locked/search_ann.
    size_t ann_built_n = 0;
    // Does not lock: only called from search_vec(), which already holds store_mu shared.
    // Taking it again would deadlock, and a recursive shared acquisition on one thread is UB.
    std::vector<std::pair<uint32_t, float>> search_ann(const std::vector<float> & qv, int k) const;
};

// bge-family retrieval instruction (queries only).
static constexpr const char * DENSE_QUERY_PREFIX =
    "Represent this sentence for searching relevant passages: ";

#endif // ALPHA_DENSE_L0
