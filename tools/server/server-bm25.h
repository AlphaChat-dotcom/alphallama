#pragma once

// Lightweight BM25 index for KV-chunk routing.
// Pure C++ (no external deps), Windows/Linux compatible.
// Used by the serve mode to route queries to relevant chunks.
//
// Security: all chunk data encrypted in memory.
//   - doc.text: ChaCha20 stream cipher (RFC 7539), per-chunk nonce
//   - doc.tf / df: SipHash-2-4 keyed hash replaces plaintext term strings
//   - Session key: 256-bit, random per server startup, mlock'd + DONTDUMP
//   - Memory dump reveals: encrypted blobs + opaque uint64 hashes + counts

#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <array>
#include <random>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <queue>
#include <thread>       // parallel batch ingest
#include <atomic>       // parallel batch ingest (work-stealing counter)
#include <mutex>        // add vs query lock
#include <shared_mutex> // add vs query lock

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>   // getrusage: build-peak RSS profiling (ALPHA_BUILD_MEMPROF)
#endif

// ChaCha20 (RFC 7539)
// Inline implementation, no dependencies. Used by TLS 1.3, WireGuard, OpenSSH.

namespace chacha20_detail {

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

static inline void quarter_round(uint32_t & a, uint32_t & b, uint32_t & c, uint32_t & d) {
    a += b; d ^= a; d = rotl32(d, 16);
    c += d; b ^= c; b = rotl32(b, 12);
    a += b; d ^= a; d = rotl32(d, 8);
    c += d; b ^= c; b = rotl32(b, 7);
}

// Produce 64 bytes of keystream from key(32), nonce(12), counter(4).
static inline void chacha20_block(const uint8_t key[32], uint32_t counter,
                                  const uint8_t nonce[12], uint8_t out[64]) {
    uint32_t s[16];
    // "expand 32-byte k"
    s[0] = 0x61707865; s[1] = 0x3320646e; s[2] = 0x79622d32; s[3] = 0x6b206574;
    // Key (little-endian)
    for (int i = 0; i < 8; i++) {
        std::memcpy(&s[4 + i], key + i * 4, 4);
    }
    s[12] = counter;
    for (int i = 0; i < 3; i++) {
        std::memcpy(&s[13 + i], nonce + i * 4, 4);
    }

    uint32_t w[16];
    std::memcpy(w, s, sizeof(w));

    // 20 rounds (10 double-rounds)
    for (int i = 0; i < 10; i++) {
        // Column rounds
        quarter_round(w[0], w[4], w[8],  w[12]);
        quarter_round(w[1], w[5], w[9],  w[13]);
        quarter_round(w[2], w[6], w[10], w[14]);
        quarter_round(w[3], w[7], w[11], w[15]);
        // Diagonal rounds
        quarter_round(w[0], w[5], w[10], w[15]);
        quarter_round(w[1], w[6], w[11], w[12]);
        quarter_round(w[2], w[7], w[8],  w[13]);
        quarter_round(w[3], w[4], w[9],  w[14]);
    }

    for (int i = 0; i < 16; i++) {
        w[i] += s[i];
    }
    std::memcpy(out, w, 64);
}

} // namespace chacha20_detail

// SipHash-2-4
// Keyed 64-bit hash. Standard for hash tables (Python, Rust, Redis).
// Resistant to hash-flooding DoS. Not reversible to original term.

namespace siphash_detail {

static inline uint64_t rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static inline void sipround(uint64_t & v0, uint64_t & v1, uint64_t & v2, uint64_t & v3) {
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
}

static inline uint64_t siphash_2_4(const uint8_t * data, size_t len, uint64_t k0, uint64_t k1) {
    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

    const size_t blocks = len / 8;
    for (size_t i = 0; i < blocks; i++) {
        uint64_t m;
        std::memcpy(&m, data + i * 8, 8);
        v3 ^= m;
        sipround(v0, v1, v2, v3);
        sipround(v0, v1, v2, v3);
        v0 ^= m;
    }

    // Last block with length byte
    uint64_t last = (uint64_t)len << 56;
    const uint8_t * tail = data + blocks * 8;
    switch (len & 7) {
        case 7: last |= (uint64_t)tail[6] << 48; [[fallthrough]];
        case 6: last |= (uint64_t)tail[5] << 40; [[fallthrough]];
        case 5: last |= (uint64_t)tail[4] << 32; [[fallthrough]];
        case 4: last |= (uint64_t)tail[3] << 24; [[fallthrough]];
        case 3: last |= (uint64_t)tail[2] << 16; [[fallthrough]];
        case 2: last |= (uint64_t)tail[1] << 8;  [[fallthrough]];
        case 1: last |= (uint64_t)tail[0];        break;
        case 0: break;
    }

    v3 ^= last;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    v0 ^= last;

    v2 ^= 0xff;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

} // namespace siphash_detail

// Chunk cipher: ChaCha20 encryption + SipHash term hashing

struct chunk_cipher {
    alignas(64) uint8_t key[32] = {};  // 256-bit session key, cache-line aligned
    // The per-session key keys both the ChaCha20 text store and the SipHash term hashes, so the lexical index
    // (df dictionary, postings, text) cannot be persisted across restarts: every boot re-ingests the corpus, and
    // df arrays from two runs never compare equal (check merge changes in-process with ALPHA_BM25_DF_VERIFY).
    // The dense store does persist (ALPHADV2), so a cold boot is a dense restore plus a full lexical re-ingest.
    // This is deliberate: the random key makes the text unreadable once the process exits. Persisting the
    // dictionary would need a fixed term key, and predictable hashes would reveal which terms are indexed.

    void init() {
        std::random_device rd;
        for (int i = 0; i < 8; i++) {
            uint32_t r = rd();
            std::memcpy(key + i * 4, &r, 4);
        }

        // Lock key in RAM to prevent swap/hibernation exposure
#ifdef _WIN32
        VirtualLock(key, sizeof(key));
#else
        mlock(key, sizeof(key));
#ifdef MADV_DONTDUMP
        // Exclude from core dumps
        madvise(key, sizeof(key), MADV_DONTDUMP);
#endif
#endif
    }

    ~chunk_cipher() {
        // Secure-zero the key before freeing
        volatile uint8_t * p = reinterpret_cast<volatile uint8_t *>(key);
        for (size_t i = 0; i < sizeof(key); i++) p[i] = 0;

#ifdef _WIN32
        VirtualUnlock(key, sizeof(key));
#else
        munlock(key, sizeof(key));
#endif
    }

    // Non-copyable (key must not be duplicated in memory)
    chunk_cipher(const chunk_cipher &) = delete;
    chunk_cipher & operator=(const chunk_cipher &) = delete;
    chunk_cipher(chunk_cipher &&) = delete;
    chunk_cipher & operator=(chunk_cipher &&) = delete;
    chunk_cipher() = default;

    // Encrypt or decrypt chunk text with ChaCha20.
    // Nonce derived from chunk_id, so each chunk gets a unique keystream.
    std::string transform(const std::string & input, int chunk_id) const {
        // Build 12-byte nonce from chunk_id (zero-padded)
        uint8_t nonce[12] = {};
        uint32_t cid = (uint32_t)chunk_id;
        std::memcpy(nonce, &cid, 4);

        std::string output = input;
        uint8_t block[64];
        size_t pos = 0;
        uint32_t counter = 0;

        while (pos < output.size()) {
            chacha20_detail::chacha20_block(key, counter++, nonce, block);
            size_t n = std::min((size_t)64, output.size() - pos);
            for (size_t i = 0; i < n; i++) {
                output[pos + i] ^= (char)block[i];
            }
            pos += n;
        }

        // Zero the last keystream block from stack
        volatile uint8_t * bp = block;
        for (int i = 0; i < 64; i++) bp[i] = 0;

        return output;
    }

    // SipHash-2-4 keyed hash for term strings.
    // Deterministic (same term -> same hash) for BM25 lookup.
    // Not reversible and resistant to hash flooding.
    uint64_t hash_term(const std::string & term) const {
        uint64_t k0, k1;
        std::memcpy(&k0, key,     8);
        std::memcpy(&k1, key + 8, 8);
        return siphash_detail::siphash_2_4(
            reinterpret_cast<const uint8_t *>(term.data()), term.size(), k0, k1);
    }
};

// BM25 index

struct bm25_index {
    float k1 = 1.5f;
    float b  = 0.75f;

    chunk_cipher cipher;

    // Add vs query lock. HTTP adds (add_document / add_documents_batch, after the corpus is built) take it
    // exclusively; the public readers that query threads call (score, top_k, top_k_scored, get_text,
    // get_source, get_parent_source, get_idf_keywords, df_of, chunk_id_at, resident) take it shared. A
    // method that holds the lock calls only the *_nolock helpers, never another locking method, so the
    // lock is never taken twice by one thread. The build path (open_*, finalize_*, build_inverted*,
    // register_parent_range, parent_source/article_meta) runs during startup ingest, before the server
    // serves queries, and is not locked. Direct reads of N are a plain aligned int and are not locked.
    mutable std::shared_mutex mu;

    // Per-document data
    struct doc_info {
        int chunk_id;
        int n_words;
        // Ciphertext is addressed by (offset, text_len). The offset is not stored here: the disk store is
        // written sequentially, so a doc's offset is the prefix sum of text_len, reconstructed in get_text
        // from a sparse checkpoint (text_off_ckpt) plus a short sum when text_off_reconstruct is set;
        // otherwise it comes from the doc_text_off[] side array. Keeping out the int64 holds doc_info at
        // 16 bytes with no alignment padding. text_len <= 0 means no text.
        int32_t text_len = 0;   // ciphertext length
        int32_t source_id = -1; // index into source_tab; sub-chunks of one file share an interned source
        // Term frequencies are stored outside doc_info, as sorted parallel arrays (term hash -> count) in
        // doc_tf_terms/doc_tf_counts, which is several times smaller than a per-doc hash map and allows a
        // binary-search lookup in score(). In spill mode tf goes to shards and the side arrays stay empty.
    };
    std::vector<doc_info> docs;
    // Documents whose raw text was non-empty but which tokenized to zero searchable terms. tokenize()
    // classifies byte-wise with ASCII-only isalpha, so every UTF-8 byte >= 0x80 is a separator and
    // non-Latin scripts (zh/ja/ko/ru) yield nothing. Such a document is still indexed (the ingest guard
    // tests !text.empty(), not the term count), so this counter is the only signal. Atomic because
    // add_documents_batch ingests on many threads. Diagnostic only; never read by the scorer.
    mutable std::atomic<size_t> n_docs_no_terms{0};
    // Per-doc sorted tf (term hash -> count), index-aligned with docs[]. Empty in spill mode.
    std::vector<std::vector<uint64_t>> doc_tf_terms;   // ascending SipHash(term)
    std::vector<std::vector<int32_t>>  doc_tf_counts;  // parallel counts

    // Source dedup: sub-chunks of one file share a source path, so it is interned once. source_tab is the
    // persistent id -> path table (read by get_source); source_intern is the ingest-only dedup map (freed
    // at finalize). doc.source_id indexes source_tab.
    std::vector<std::string>                source_tab;
    // Keyed by the path hash rather than a copy of the string, so each path is stored once. Collisions are
    // resolved by comparing against source_tab (the multimap chains them).
    std::unordered_multimap<uint64_t, int32_t> source_intern;
    int32_t intern_source(const std::string & s) {
        const uint64_t h = std::hash<std::string>{}(s);
        auto range = source_intern.equal_range(h);
        for (auto it = range.first; it != range.second; ++it)
            if (source_tab[it->second] == s) return it->second;   // verified match (guards hash collisions)
        int32_t id = (int32_t)source_tab.size();
        source_tab.push_back(s);
        source_intern.emplace(h, id);
        return id;
    }

    // Per-sub-chunk byte offset into the parent file, index-aligned with docs[]. The ingest source is
    // "<path>@<offset>"; only the path is interned (one entry per file) and get_source() rejoins
    // path + "@" + offset, reproducing the original string exactly.
    std::vector<int32_t> doc_source_off;   // -1 = no "@<digits>" suffix; the source is stored verbatim
    // Split "<path>@<byteoffset>": intern the path, return the offset in off_out (-1 if there is no
    // all-digit "@N" suffix, in which case the whole string is interned).
    int32_t intern_source_path(const std::string & tagged, int32_t & off_out) {
        off_out = -1;
        size_t at = tagged.rfind('@');
        if (at != std::string::npos && at + 1 < tagged.size() && tagged.size() - at - 1 <= 15) {
            bool alldig = true;
            for (size_t k = at + 1; k < tagged.size(); k++)
                if (tagged[k] < '0' || tagged[k] > '9') { alldig = false; break; }
            if (alldig) {
                long long v = atoll(tagged.c_str() + at + 1);
                // Only split if re-stringifying reproduces the exact suffix (rejects leading-zero forms
                // like "@00"), so get_source is lossless.
                if (v >= 0 && v <= 0x7fffffffLL && std::to_string(v) == tagged.substr(at + 1)) {
                    off_out = (int32_t)v; return intern_source(tagged.substr(0, at));
                }
            }
        }
        return intern_source(tagged);   // no valid offset suffix: intern verbatim, off stays -1
    }

    // Disk-backed text store: rather than keeping each doc's ChaCha20 ciphertext resident, stream it to
    // a private file during ingest and mmap it read-only; get_text() reads the slice and decrypts. The
    // data at rest is still ciphertext. Enabled by open_text_store() (batch --corpus-dir ingest); the
    // incremental API path keeps text resident.
    bool        text_on_disk = false;
    std::string text_path;
    FILE *      text_fp = nullptr;   // open during ingest (append), closed at finalize
    int64_t     text_woff = 0;       // running write offset
    const unsigned char * text_mm = nullptr;
    size_t      text_mm_len = 0;
#ifndef _WIN32
    int         text_fd = -1;
#endif
    std::string text_fallback;       // holds the store in-memory if the file/mmap path is unavailable
    // Resident ciphertext blob for non-disk docs (resident mode, or the disk-full fallback). Docs point
    // in via a negative text_off = -(byte_offset + 1). Empty in pure disk mode.
    std::string resident_text;

    // On-disk inverted index (ALPHA_BM25_INVERTED)
    // The forward index (per-doc tf arrays) is resident and forces an O(N) scan per query; both grow
    // linearly with the corpus. The inverted index moves postings to a varint-compressed mmap'd file and
    // turns a query into posting-list traversal over only the docs that contain a query term.
    // build_inverted() runs once at finalize: it inverts the resident forward arrays, writes and mmaps the
    // postings, and frees the per-doc tf arrays; score() then routes to score_inverted(). The term
    // dictionary is the sorted df_terms; post_off/post_len are parallel to it.
    bool                  inverted_active = false;
    // Warn once when a query hits an unfinalized index in inverted mode (see score()). An owner that never
    // finalizes by design (a small resident corpus with no finalize endpoint) clears it.
    bool                  warn_unfinalized = true;
    std::vector<uint64_t> post_off;   // byte offset of term t's postings blob (parallel to df_terms)
    std::vector<uint32_t> post_len;   // byte length of term t's postings blob
    std::vector<int32_t>  doc_nwords; // n_words indexed by chunk_id (length norm; survives freeing tf)
    std::string           post_path;
    const unsigned char * post_mm = nullptr;
    size_t                post_mm_len = 0;
#ifndef _WIN32
    int                   post_fd = -1;
#endif
    std::string           post_fallback;  // resident postings if mmap unavailable (Windows / mmap fail)
    // On by default: measured 5.9-11.7x faster per query than the forward scan on four BEIR corpora
    // (scifact, nfcorpus, scidocs, fiqa) with nDCG@10 unchanged (three identical to four decimals, one
    // +0.0001); scores are bit-identical by construction. ALPHA_BM25_INVERTED=0 keeps the forward index.
    // The index is built at finalize (corpus-dir ingest at start, or POST /api/corpus/finalize after
    // HTTP ingest); until then, or if the build fails, queries use the forward scan. Documents added after
    // build_inverted() (e.g. HTTP /api/kv_chunk/index after a corpus-dir ingest) cannot join the immutable
    // postings; they go to the post-finalize overlay (see below), which score_inverted() scans alongside
    // the postings.
    static bool bm25_inverted() {
        static const bool v = []() {
            const char * e = getenv("ALPHA_BM25_INVERTED");
            return e == nullptr || atoi(e) != 0;
        }();
        return v;
    }

    // External (streaming) inversion
    // The resident inversion at finalize still holds the whole forward index in RAM during the build.
    // With spilling, ingest writes each doc's postings as fixed 16-byte (hash, doc, tf) records into
    // SPILL_SHARDS files keyed by the top bits of the hash, and clears the doc's tf arrays. At finalize,
    // one shard at a time is loaded, sorted by (hash, doc) and grouped into varint postings, so peak RAM
    // is one shard regardless of corpus size. Shards are in hash-prefix order and each is fully sorted,
    // so df_terms stays globally sorted and score_inverted() works unchanged.
    static constexpr int SPILL_BITS  = 8;                 // 256 shards
    static constexpr int SPILL_SHARDS = 1 << SPILL_BITS;
    bool                  spill_active = false;
    std::string           spill_prefix;                   // shard files: <prefix>.<k>
    std::vector<FILE*>    spill_fp;                        // one append handle per shard (ingest)
    uint64_t              spill_records = 0;               // total postings spilled (stat)
    struct spill_rec { uint64_t hash; uint32_t doc; uint32_t tf; };  // 16 B, packed by layout
    void open_postings_spill(const std::string & prefix) {
        if (spill_active) return;
        spill_prefix = prefix;
        spill_fp.assign(SPILL_SHARDS, nullptr);
        bool ok = true;
        for (int k = 0; k < SPILL_SHARDS && ok; k++) {
            const std::string p = prefix + "." + std::to_string(k);
            // w+b: write during ingest, then rewind and read the same handle at finalize, with no
            // reopen by path. On POSIX the file is unlinked immediately so a SIGKILL mid-ingest cannot
            // orphan it; the OS reclaims the space when the fd closes. The spill is several times the
            // size of the final index, so an orphaned spill can fill the disk.
            spill_fp[k] = fopen(p.c_str(), "w+b");
            if (!spill_fp[k]) { ok = false; break; }
#ifndef _WIN32
            unlink(p.c_str());     // mapping/handle stays valid; nothing left on disk by name
#endif
        }
        if (!ok) {                                        // couldn't open shards: stay on the resident path
            for (auto f : spill_fp) if (f) fclose(f);
            spill_fp.clear();
            spill_cleanup();       // remove any named shards (Windows path / partial open)
            return;
        }
        spill_active = true;
    }
    // Spill one doc's postings (its sorted tf arrays) to the shard files. Called from the serial merge
    // of add_documents_batch or from add_document, never concurrently, so the FILE* appends are safe.
    void spill_postings(const std::vector<uint64_t> & tf_terms, const std::vector<int32_t> & tf_counts, uint32_t chunk_id) {
        const size_t m = tf_terms.size();
        for (size_t i = 0; i < m; i++) {
            spill_rec r{ tf_terms[i], chunk_id, (uint32_t)tf_counts[i] };
            int k = (int)(r.hash >> (64 - SPILL_BITS));
            fwrite(&r, sizeof(r), 1, spill_fp[k]);
        }
        spill_records += m;
    }

    // Off-RAM term dictionary
    // With postings on disk, the largest resident structure is the term dictionary: df_terms(8) +
    // df_counts(4) + post_off(8) + post_len(4) = 24 bytes/term. These are POD parallel arrays, so they
    // are mmap'd from a single file and binary-searched in place. The dict accessors read the mmap when
    // active, else the resident vectors. On by default in inverted mode; ALPHA_BM25_DICT_RESIDENT=1 keeps the
    // dictionary resident. Rankings are identical either way.
    static bool dict_mmap_enabled() {
        static const bool v = [] { const char * e = getenv("ALPHA_BM25_DICT_RESIDENT"); return !e || atoi(e) == 0; }();
        return v;
    }
    bool                  dict_mm_active = false;
    size_t                dict_n = 0;                      // term count when the dict is mmap'd
    std::string           dict_path;
    const unsigned char * dict_mm = nullptr;
    size_t                dict_mm_len = 0;
#ifndef _WIN32
    int                   dict_fd = -1;
#endif
    std::string           dict_fallback;
    // Section pointers into the mmap: [terms V*8][poff V*8][dfc V*4][plen V*4].
    const uint64_t *      dterms_mm = nullptr;
    const int32_t  *      ddfc_mm   = nullptr;
    const uint64_t *      dpoff_mm  = nullptr;
    const uint32_t *      dplen_mm  = nullptr;
    // Dictionary accessors: one code path for query, backed by mmap or resident vectors.
    size_t   dict_V()          const { return dict_mm_active ? dict_n     : df_terms.size(); }
    uint64_t dict_term(size_t i) const { return dict_mm_active ? dterms_mm[i] : df_terms[i]; }
    int32_t  dict_dfc(size_t i)  const { return dict_mm_active ? ddfc_mm[i]   : df_counts[i]; }
    uint64_t dict_poff(size_t i) const { return dict_mm_active ? dpoff_mm[i]  : post_off[i]; }
    uint32_t dict_plen(size_t i) const { return dict_mm_active ? dplen_mm[i]  : post_len[i]; }

    // Global stats
    std::unordered_map<uint64_t, int> df;  // SipHash(term) -> document count (build-time map, incremental path only)
    // Packed df: after batch ingest, finalize_df() produces sorted parallel arrays (12 bytes/entry).
    // df_of() binary-searches when finalized, else falls back to the map (incremental API path).
    std::vector<uint64_t> df_terms;
    std::vector<int32_t>  df_counts;
    bool                  df_finalized = false;
    // Streaming df (batch): df is derivable from the per-doc sorted tf_terms, so in streaming mode
    // add_document does not build the df map and finalize_df() k-way-merges the docs' tf_terms into the
    // df arrays. This avoids a large global map that would also fragment the heap.
    bool                  df_streaming = false;
    void enable_df_streaming() { df_streaming = true; }

    // Post-finalize overlay
    // Once the corpus is finalized the df dictionary is frozen (packed arrays, possibly mmap'd) and, in
    // inverted mode, the postings are immutable and the forward tf arrays are gone. A document added after
    // that point is recorded here instead:
    //   - ov_df counts its terms; df_of() returns dictionary df + ov_df, so a term only a later document
    //     introduces still has df > 0 and is scored.
    //   - in inverted mode its sorted tf arrays and length go to ov_chunk/ov_nwords/ov_tf_*, which
    //     score_inverted() scores by brute force next to the postings. In forward mode the document
    //     still lands in doc_tf_terms/doc_tf_counts (score() scans those), so only ov_df is used.
    // N, total_words and avgdl include overlay documents in both modes (add_document maintains them as
    // before), and the text store path is unchanged. The overlay is empty unless a document is added after
    // finalize, so a corpus without such adds scores exactly as without it. There is no save/load of the
    // index; the overlay lives as long as the process, like the rest of the index.
    std::unordered_map<uint64_t, int32_t> ov_df;          // term hash -> count over overlay-era adds
    std::vector<int32_t>                  ov_chunk;       // inverted mode: chunk_id per overlay doc
    std::vector<int32_t>                  ov_nwords;      // inverted mode: n_words per overlay doc
    std::vector<std::vector<uint64_t>>    ov_tf_terms;    // inverted mode: ascending term hashes
    std::vector<std::vector<int32_t>>     ov_tf_counts;   // inverted mode: parallel counts
    bool post_finalize() const { return df_finalized || inverted_active; }
    // Record one post-finalize document's terms. Inverted mode keeps the tf arrays in the overlay (taking
    // them from the caller) and returns true; forward mode returns false and the caller stores them in
    // doc_tf_terms/doc_tf_counts as usual.
    bool overlay_add(int chunk_id, int n_words, std::vector<uint64_t> & tf_terms, std::vector<int32_t> & tf_counts) {
        for (uint64_t h : tf_terms) ov_df[h]++;
        if (!inverted_active) return false;
        ov_chunk.push_back(chunk_id);
        ov_nwords.push_back(n_words);
        ov_tf_terms.push_back(std::move(tf_terms));
        ov_tf_counts.push_back(std::move(tf_counts));
        return true;
    }

    int df_of(uint64_t hash) const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return df_of_nolock(hash);
    }
    int df_of_nolock(uint64_t hash) const {
        int d;
        if (df_finalized) {
            int t = term_index(hash);
            d = t < 0 ? 0 : dict_dfc((size_t)t);
        } else {
            auto it = df.find(hash);
            d = it != df.end() ? it->second : 0;
        }
        if (!ov_df.empty()) {
            auto it = ov_df.find(hash);
            if (it != ov_df.end()) d += it->second;
        }
        return d;
    }

    // Produce the packed df arrays. Streaming mode derives df from the per-doc tf_terms (no map);
    // otherwise it collapses the incremental-path map. Call once after a batch ingest.
    void finalize_df() {
        if (df_finalized) return;
        std::unordered_multimap<uint64_t,int32_t>().swap(source_intern);  // ingest-only dedup map (hash-keyed)
        // External inversion mode: the tf arrays were spilled and cleared during ingest, so there is
        // nothing to k-way-merge here. df is built by build_inverted_external() from the shards.
        if (spill_active) return;
        if (df_streaming) {
            // k-way merge across all docs' ascending tf_terms; each doc contributes a term once, so
            // the number of cursors sharing a hash = that term's document frequency.
            struct cur { uint64_t h; int d; int pos; };
            struct gt { bool operator()(const cur & a, const cur & b) const { return a.h > b.h; } };

            // Parallel range-partitioned merge (ALPHA_BM25_DF_THREADS)
            // Each doc's tf_terms is sorted ascending, so the hash space splits into disjoint [lo, hi)
            // ranges that merge independently. A worker binary-searches lower_bound(lo) in each doc and
            // touches only its own slice, so total work is not duplicated. SipHash output is uniform, so
            // equal-width ranges balance. Ranges concatenate in order and the merge within a range is the
            // serial algorithm, so the output is identical to the serial merge.
            //
            // Default min(hardware threads, 8); returns diminish beyond that. ALPHA_BM25_DF_THREADS=N
            // overrides (1 = serial, 0 = all hardware threads). ALPHA_BM25_DF_VERIFY checks the result
            // against the serial merge.
            unsigned df_threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
            if (const char * e = getenv("ALPHA_BM25_DF_THREADS")) {
                int v = atoi(e);
                if (v < 0) v = 0;
                df_threads = (v == 0) ? std::max(1u, std::thread::hardware_concurrency()) : (unsigned) v;
            }
            const size_t ND_df = doc_tf_terms.size();
            if (df_threads > 1 && ND_df > 0) {
                const unsigned R = df_threads;
                std::vector<std::vector<uint64_t>> r_terms(R);
                std::vector<std::vector<int32_t>>  r_counts(R);
                auto worker = [&](unsigned r) {
                    // [lo, hi) over the full uint64 hash space; the last range absorbs the remainder.
                    const uint64_t span = (uint64_t) (~0ULL / R);
                    const uint64_t lo = span * r;
                    const uint64_t hi = (r + 1 == R) ? ~0ULL : (span * (r + 1));
                    std::priority_queue<cur, std::vector<cur>, gt> pq;
                    for (size_t d = 0; d < ND_df; d++) {
                        const std::vector<uint64_t> & tt = doc_tf_terms[d];
                        if (tt.empty()) continue;
                        // start at this range's first term in this doc; skip the doc otherwise
                        size_t p0 = (size_t) (std::lower_bound(tt.begin(), tt.end(), lo) - tt.begin());
                        if (p0 >= tt.size()) continue;
                        if (!(r + 1 == R) && tt[p0] >= hi) continue;
                        pq.push({tt[p0], (int) d, (int) p0});
                    }
                    while (!pq.empty()) {
                        const uint64_t h = pq.top().h;
                        if (!(r + 1 == R) && h >= hi) break;    // past this range: the rest belongs to r+1
                        int cnt = 0;
                        while (!pq.empty() && pq.top().h == h) {
                            cur cc = pq.top(); pq.pop();
                            cnt++;
                            const std::vector<uint64_t> & tt = doc_tf_terms[cc.d];
                            if (cc.pos + 1 < (int) tt.size()) pq.push({tt[cc.pos + 1], cc.d, cc.pos + 1});
                        }
                        r_terms[r].push_back(h);
                        r_counts[r].push_back(cnt);
                    }
                };
                std::vector<std::thread> pool;
                for (unsigned r = 0; r < R; r++) pool.emplace_back(worker, r);
                for (auto & th : pool) th.join();
                size_t tot = 0;
                for (unsigned r = 0; r < R; r++) tot += r_terms[r].size();
                df_terms.reserve(tot); df_counts.reserve(tot);
                for (unsigned r = 0; r < R; r++) {               // concatenate in range order
                    df_terms.insert(df_terms.end(), r_terms[r].begin(), r_terms[r].end());
                    df_counts.insert(df_counts.end(), r_counts[r].begin(), r_counts[r].end());
                }
                // ALPHA_BM25_DF_VERIFY: also run the serial merge in this process and compare byte for byte.
                // It has to be in-process: term hashes use a random per-session key, so two runs never
                // share df_terms.
                if (getenv("ALPHA_BM25_DF_VERIFY")) {
                    std::vector<uint64_t> s_terms; std::vector<int32_t> s_counts;
                    std::priority_queue<cur, std::vector<cur>, gt> spq;
                    for (int d = 0; d < (int) ND_df; d++)
                        if (!doc_tf_terms[d].empty()) spq.push({doc_tf_terms[d][0], d, 0});
                    while (!spq.empty()) {
                        const uint64_t h = spq.top().h;
                        int cnt = 0;
                        while (!spq.empty() && spq.top().h == h) {
                            cur cc = spq.top(); spq.pop();
                            cnt++;
                            if (cc.pos + 1 < (int) doc_tf_terms[cc.d].size())
                                spq.push({doc_tf_terms[cc.d][cc.pos + 1], cc.d, cc.pos + 1});
                        }
                        s_terms.push_back(h); s_counts.push_back(cnt);
                    }
                    const bool same = s_terms.size() == df_terms.size() && s_counts.size() == df_counts.size()
                        && std::memcmp(s_terms.data(), df_terms.data(), s_terms.size() * sizeof(uint64_t)) == 0
                        && std::memcmp(s_counts.data(), df_counts.data(), s_counts.size() * sizeof(int32_t)) == 0;
                    fprintf(stderr, "bm25: df VERIFY parallel(%u threads) vs serial: %s (%zu vs %zu terms)\n",
                            R, same ? "BYTE-IDENTICAL" : "DIFFERENT", df_terms.size(), s_terms.size());
                }
                df_finalized = true;
                return;
            }

            std::priority_queue<cur, std::vector<cur>, gt> pq;
            for (int d = 0; d < (int)doc_tf_terms.size(); d++)   // empty in spill mode (df comes from shards)
                if (!doc_tf_terms[d].empty()) pq.push({doc_tf_terms[d][0], d, 0});
            while (!pq.empty()) {
                uint64_t h = pq.top().h;
                int cnt = 0;
                while (!pq.empty() && pq.top().h == h) {
                    cur c = pq.top(); pq.pop();
                    cnt++;
                    if (c.pos + 1 < (int)doc_tf_terms[c.d].size())
                        pq.push({doc_tf_terms[c.d][c.pos + 1], c.d, c.pos + 1});
                }
                df_terms.push_back(h);
                df_counts.push_back(cnt);
            }
            df_terms.shrink_to_fit();    // push_back doubled capacity; reclaim the slack
            df_counts.shrink_to_fit();
            df_finalized = true;
            return;
        }
        if (df.empty()) return;
        std::vector<std::pair<uint64_t,int>> tmp(df.begin(), df.end());
        std::sort(tmp.begin(), tmp.end(), [](const auto&a, const auto&b){ return a.first < b.first; });
        df_terms.reserve(tmp.size()); df_counts.reserve(tmp.size());
        for (auto & p : tmp) { df_terms.push_back(p.first); df_counts.push_back((int32_t)p.second); }
        std::unordered_map<uint64_t,int>().swap(df);   // free the map
        df_finalized = true;
    }

    // varint (unsigned LEB128): delta doc ids + tf compress to ~2-3 bytes/posting
    static void put_varint(std::string & out, uint32_t v) {
        while (v >= 0x80) { out.push_back((char)(v | 0x80)); v >>= 7; }
        out.push_back((char)v);
    }
    static uint32_t get_varint(const unsigned char * p, size_t & i) {
        uint32_t v = 0; int shift = 0; unsigned char b;
        do { b = p[i++]; v |= (uint32_t)(b & 0x7F) << shift; shift += 7; } while (b & 0x80);
        return v;
    }

    // Index of a term hash in the sorted dictionary, or -1. The postings dictionary is the df
    // dictionary. Binary search goes through the accessor so it works whether the dictionary is a
    // resident vector or an mmap'd file.
    int term_index(uint64_t h) const {
        size_t lo = 0, hi = dict_V();
        while (lo < hi) {
            size_t mid = lo + ((hi - lo) >> 1);
            uint64_t v = dict_term(mid);
            if (v < h) lo = mid + 1;
            else       hi = mid;
        }
        if (lo < dict_V() && dict_term(lo) == h) return (int)lo;
        return -1;
    }

    // Build the on-disk inverted index from the resident forward arrays. Must be called after
    // finalize_df() (needs df_terms as the dictionary) and after the ingest loop.
    // Postings for term t are the docs containing t in ascending chunk_id, which follows from iterating
    // docs in chunk order, so no per-term sort is needed. Delta + varint encoded. After writing and
    // mmapping, the per-doc tf arrays are freed.
    void build_inverted(const std::string & path) {
        if (!bm25_inverted() || inverted_active) return;
        // If ingest spilled postings to shards, invert externally; otherwise invert the resident
        // forward arrays below.
        if (spill_active) { build_inverted_external(path); return; }
        finalize_df();                                  // idempotent; guarantees df_terms
        const size_t V = df_terms.size();
        if (V == 0 || docs.empty()) return;

        // doc_nwords[chunk_id] survives the freeing of the docs' tf arrays; scoring's length norm
        // needs it. chunk_ids are dense 0..N-1 in the batch path, but size by max to be safe.
        int max_cid = 0;
        for (auto & d : docs) max_cid = std::max(max_cid, d.chunk_id);
        doc_nwords.assign((size_t)max_cid + 1, 0);
        for (auto & d : docs) doc_nwords[d.chunk_id] = d.n_words;

        // Counting sort docs -> per-term contiguous slices. pfx[t] = start of term t's postings.
        std::vector<uint64_t> pfx(V + 1, 0);
        for (size_t t = 0; t < V; t++) pfx[t + 1] = pfx[t] + (uint64_t)df_counts[t];
        const uint64_t total = pfx[V];
        std::vector<int32_t> pdoc(total);
        std::vector<int32_t> ptf(total);
        std::vector<uint64_t> cursor(pfx.begin(), pfx.begin() + V);
        for (size_t d = 0; d < docs.size(); d++) {
            const doc_info & doc = docs[d];
            const size_t m = doc_tf_terms[d].size();
            for (size_t j = 0; j < m; j++) {
                int t = term_index(doc_tf_terms[d][j]);
                if (t < 0) continue;                    // cannot happen (df built from these terms)
                uint64_t c = cursor[t]++;
                pdoc[c] = doc.chunk_id;
                ptf[c]  = doc_tf_counts[d][j];
            }
        }

        // Encode each term's slice into the postings file. Slices are already ascending in doc.
        post_off.assign(V, 0);
        post_len.assign(V, 0);
        post_path = path;
        FILE * fp = fopen(path.c_str(), "wb");
        bool ok = (fp != nullptr);
        uint64_t woff = 0;
        std::string buf;
        if (ok) {
            for (size_t t = 0; t < V && ok; t++) {
                buf.clear();
                int32_t prev = 0;
                for (uint64_t i = pfx[t]; i < pfx[t + 1]; i++) {
                    put_varint(buf, (uint32_t)(pdoc[i] - prev));
                    prev = pdoc[i];
                    put_varint(buf, (uint32_t)ptf[i]);
                }
                post_off[t] = woff;
                post_len[t] = (uint32_t)buf.size();
                if (!buf.empty() && fwrite(buf.data(), 1, buf.size(), fp) != buf.size()) ok = false;
                woff += buf.size();
            }
            fflush(fp); fclose(fp); fp = nullptr;
        }
        std::vector<int32_t>().swap(pdoc);              // free the inversion scratch
        std::vector<int32_t>().swap(ptf);
        if (!ok) { post_off.clear(); post_len.clear(); return; }   // stay on the forward path

        // mmap the postings read-only (mirror finalize_text_store). Fallback: slurp resident.
        bool mapped = false;
#ifndef _WIN32
        post_fd = open(path.c_str(), O_RDONLY);
        if (post_fd >= 0) {
            struct stat st;
            if (fstat(post_fd, &st) == 0 && st.st_size > 0) {
                void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, post_fd, 0);
                if (p != MAP_FAILED) {
                    post_mm = (const unsigned char*)p; post_mm_len = (size_t)st.st_size;
                    unlink(path.c_str());               // mapping stays valid; no on-disk residue
                    mapped = true;
                }
            }
            if (!mapped) { close(post_fd); post_fd = -1; }
        }
#endif
        if (!mapped) {
            FILE * rf = fopen(path.c_str(), "rb");
            if (!rf) { post_off.clear(); post_len.clear(); return; }
            fseek(rf, 0, SEEK_END); long sz = ftell(rf); fseek(rf, 0, SEEK_SET);
            post_fallback.resize(sz > 0 ? sz : 0);
            if (sz > 0) { size_t rd = fread(&post_fallback[0], 1, sz, rf); (void)rd; }
            fclose(rf);
            post_mm = (const unsigned char*)post_fallback.data(); post_mm_len = post_fallback.size();
#ifndef _WIN32
            unlink(path.c_str());
#endif
        }

        // Free the forward tf arrays; the postings now live on disk.
        std::vector<std::vector<uint64_t>>().swap(doc_tf_terms);
        std::vector<std::vector<int32_t>>().swap(doc_tf_counts);
        inverted_active = true;
    }

    // External build: postings were spilled to shard files during ingest (see spill_postings), so the
    // forward index was never accumulated. Process one shard at a time: read it, sort by (hash, doc),
    // group into varint postings, and build df_terms/df_counts and post_off/post_len. Peak RAM is the
    // largest single shard, independent of corpus size.
    void build_inverted_external(const std::string & path) {
        // Do not close spill_fp up front: on POSIX the shards were unlinked at open and have no name to
        // reopen. Each retained handle is flushed, rewound and read below, then closed, which also
        // reclaims its disk space.

        std::unordered_multimap<uint64_t,int32_t>().swap(source_intern);  // ingest-only dedup map (hash-keyed)

        // doc_nwords[chunk_id] for the length norm (the tf arrays are already gone).
        int max_cid = 0;
        for (auto & d : docs) max_cid = std::max(max_cid, d.chunk_id);
        doc_nwords.assign((size_t)max_cid + 1, 0);
        for (auto & d : docs) doc_nwords[d.chunk_id] = d.n_words;

        log_build_mem("ingest-end");                     // docs full, tf spilled/cleared, dict not yet built
        df_terms.clear(); df_counts.clear();
        post_off.clear(); post_len.clear();
        FILE * fp = fopen(path.c_str(), "wb");
        if (!fp) { spill_cleanup(); return; }            // no postings file: cannot invert
        post_path = path;
        uint64_t woff = 0;
        std::string buf;
        std::vector<spill_rec> recs;

        // Stream the dict to disk during the loop instead of building df_terms/post_off/df_counts/post_len
        // in RAM. Terms are emitted globally sorted, so each section is appended to its own temp file and
        // the files are concatenated and mmap'd at the end. Only done when the dict would be mmap'd anyway
        // (dict_mmap_enabled); if any temp file fails to open, fall back to the resident vectors.
        const std::string dpath = path + ".dict";
        const std::string ft = dpath + ".t", fpo = dpath + ".po", fdc = dpath + ".dc", fpl = dpath + ".pl";
        bool stream_dict = dict_mmap_enabled();
        FILE *sd_t=nullptr,*sd_po=nullptr,*sd_dc=nullptr,*sd_pl=nullptr;
        if (stream_dict) {
            sd_t=fopen(ft.c_str(),"wb"); sd_po=fopen(fpo.c_str(),"wb");
            sd_dc=fopen(fdc.c_str(),"wb"); sd_pl=fopen(fpl.c_str(),"wb");
            if (!(sd_t && sd_po && sd_dc && sd_pl)) {     // couldn't open temps: resident fallback
                if(sd_t)fclose(sd_t); if(sd_po)fclose(sd_po); if(sd_dc)fclose(sd_dc); if(sd_pl)fclose(sd_pl);
                sd_t=sd_po=sd_dc=sd_pl=nullptr; stream_dict=false;
                remove(ft.c_str()); remove(fpo.c_str()); remove(fdc.c_str()); remove(fpl.c_str());
            }
        }
        size_t Vs = 0;                                    // streamed term count
        bool stream_ok = stream_dict;

        // Shards in ascending prefix order, so df_terms comes out globally sorted (within-shard sort by
        // full hash; shard k holds exactly the hashes whose top SPILL_BITS == k).
        for (int k = 0; k < SPILL_SHARDS; k++) {
            FILE * sf = (k < (int)spill_fp.size()) ? spill_fp[k] : nullptr;
            if (!sf) continue;
            fflush(sf);
            fseek(sf, 0, SEEK_END); long sz = ftell(sf); fseek(sf, 0, SEEK_SET);
            size_t n = sz > 0 ? (size_t)sz / sizeof(spill_rec) : 0;
            if (n == 0) { fclose(sf); spill_fp[k] = nullptr; continue; }
            recs.resize(n);
            size_t rd = fread(recs.data(), sizeof(spill_rec), n, sf);
            fclose(sf); spill_fp[k] = nullptr;           // closing reclaims the (already unlinked) space
            if (rd != n) recs.resize(rd);

            std::sort(recs.begin(), recs.end(), [](const spill_rec & a, const spill_rec & b) {
                if (a.hash != b.hash) return a.hash < b.hash;
                return a.doc < b.doc;
            });
            for (size_t i = 0; i < recs.size(); ) {
                const uint64_t h = recs[i].hash;
                size_t j = i; buf.clear(); uint32_t prev = 0;
                while (j < recs.size() && recs[j].hash == h) {
                    put_varint(buf, recs[j].doc - prev);  // ascending within a term: non-negative delta
                    prev = recs[j].doc;
                    put_varint(buf, recs[j].tf);
                    j++;
                }
                const int32_t  dc = (int32_t)(j - i);
                const uint32_t pl = (uint32_t)buf.size();
                if (stream_dict && stream_ok) {          // append to the 4 streamed sections
                    stream_ok = stream_ok
                        && fwrite(&h,   8, 1, sd_t)  == 1
                        && fwrite(&woff,8, 1, sd_po) == 1
                        && fwrite(&dc,  4, 1, sd_dc) == 1
                        && fwrite(&pl,  4, 1, sd_pl) == 1;
                    Vs++;
                } else {                                  // resident dict (or a streamed write failed)
                    df_terms.push_back(h);
                    df_counts.push_back(dc);
                    post_off.push_back(woff);
                    post_len.push_back(pl);
                }
                if (!buf.empty() && fwrite(buf.data(), 1, buf.size(), fp) != buf.size()) {
                    fclose(fp);
                    if(sd_t)fclose(sd_t); if(sd_po)fclose(sd_po); if(sd_dc)fclose(sd_dc); if(sd_pl)fclose(sd_pl);
                    remove(ft.c_str()); remove(fpo.c_str()); remove(fdc.c_str()); remove(fpl.c_str());
                    spill_cleanup(); return;
                }
                woff += buf.size();
                i = j;
            }
        }
        fflush(fp); fclose(fp);
        if (sd_t){fflush(sd_t);fclose(sd_t);} if(sd_po){fflush(sd_po);fclose(sd_po);}
        if (sd_dc){fflush(sd_dc);fclose(sd_dc);} if(sd_pl){fflush(sd_pl);fclose(sd_pl);}
        if (!stream_dict) {
            df_terms.shrink_to_fit(); df_counts.shrink_to_fit();
            post_off.shrink_to_fit(); post_len.shrink_to_fit();
        }
        df_finalized = true;                             // df built here, not in finalize_df()
        std::vector<spill_rec>().swap(recs);

        // mmap the postings (mirror finalize_text_store); fallback = slurp resident.
        bool mapped = false;
#ifndef _WIN32
        post_fd = open(path.c_str(), O_RDONLY);
        if (post_fd >= 0) {
            struct stat st;
            if (fstat(post_fd, &st) == 0 && st.st_size > 0) {
                void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, post_fd, 0);
                if (p != MAP_FAILED) {
                    post_mm = (const unsigned char*)p; post_mm_len = (size_t)st.st_size;
                    unlink(path.c_str()); mapped = true;
                }
            }
            if (!mapped) { close(post_fd); post_fd = -1; }
        }
#endif
        if (!mapped) {
            FILE * rf = fopen(path.c_str(), "rb");
            if (rf) {
                fseek(rf, 0, SEEK_END); long sz = ftell(rf); fseek(rf, 0, SEEK_SET);
                post_fallback.resize(sz > 0 ? sz : 0);
                if (sz > 0) { size_t rd = fread(&post_fallback[0], 1, sz, rf); (void)rd; }
                fclose(rf);
                post_mm = (const unsigned char*)post_fallback.data(); post_mm_len = post_fallback.size();
#ifndef _WIN32
                unlink(path.c_str());
#endif
            }
        }
        // all shard handles were closed in the loop; drop any stragglers and clear.
        for (auto f : spill_fp) if (f) fclose(f);
        spill_fp.clear();
        spill_active = false;
        inverted_active = (post_mm != nullptr || post_mm_len == 0);

        // Move the term dictionary off-RAM: write the 4 parallel POD arrays to one file, mmap it, and
        // free the resident vectors. term_index/df_of/score_inverted then binary-search the mmap via the
        // dict accessors. Skipped with ALPHA_BM25_DICT_RESIDENT=1 (dictionary stays resident).
        log_build_mem("dict-built");                     // peak composition: docs + dict

        // Streamed dict: concatenate the 4 temp sections into the single-file layout
        // [terms V*8][poff V*8][dfc V*4][plen V*4], then mmap. On any failure, read the temps back into
        // the resident vectors so the index stays correct.
        if (stream_dict && inverted_active) {
            bool ok = stream_ok && Vs > 0;
            if (ok) {
                FILE * out = fopen(dpath.c_str(), "wb");
                ok = out != nullptr;
                if (ok) {
                    ok = append_file(out, ft) && append_file(out, fpo)
                      && append_file(out, fdc) && append_file(out, fpl);
                    fflush(out); fclose(out);
                }
                if (ok) ok = mmap_dict(dpath, Vs);       // sets accessors + dict_mm_active
                if (!ok) remove(dpath.c_str());
            }
            if (!ok) {                                   // recover: rebuild resident dict from the temps
                read_all_into(ft, df_terms);  read_all_into(fpo, post_off);
                read_all_into(fdc, df_counts); read_all_into(fpl, post_len);
                if (dict_mmap_enabled() && !df_terms.empty()) build_dict_mmap(dpath);  // retry mmap resident
            }
            remove(ft.c_str()); remove(fpo.c_str()); remove(fdc.c_str()); remove(fpl.c_str());
        } else if (inverted_active && dict_mmap_enabled() && !df_terms.empty()) {
            build_dict_mmap(dpath);                       // dict was built resident; move it to mmap
        }
        log_build_mem("final");                          // dict off-RAM (mmap); docs still resident
    }

    // Serialize df_terms|df_counts|post_off|post_len to <dpath> as 4 contiguous sections, mmap it,
    // repoint the accessors at the mapping, and free the resident vectors.
    void build_dict_mmap(const std::string & dpath) {
        const size_t V = df_terms.size();
        FILE * fp = fopen(dpath.c_str(), "wb");
        if (!fp) return;                                   // keep the resident dict on failure
        // Order the two 8-byte arrays first so both are naturally 8-aligned in the mapping regardless
        // of parity of V (avoids an unaligned uint64 read on strict-alignment targets).
        bool ok = true;
        ok = ok && fwrite(df_terms.data(),  8, V, fp) == V;   // terms  @ 0
        ok = ok && fwrite(post_off.data(),  8, V, fp) == V;   // poff   @ V*8
        ok = ok && fwrite(df_counts.data(), 4, V, fp) == V;   // dfc    @ V*16
        ok = ok && fwrite(post_len.data(),  4, V, fp) == V;   // plen   @ V*16 + V*4
        fflush(fp); fclose(fp);
        if (!ok) { remove(dpath.c_str()); return; }
        mmap_dict(dpath, V);                                // mmap + repoint accessors + free resident
    }

    // mmap the finalized dict file <dpath> (layout [terms V*8][poff V*8][dfc V*4][plen V*4]), repoint the
    // accessors at the mapping, and free the resident dict vectors. Shared by the resident-build path
    // (build_dict_mmap) and the streamed-dict path in build_inverted_external. Returns success.
    bool mmap_dict(const std::string & dpath, size_t V) {
        dict_path = dpath;
        bool mapped = false;
#ifndef _WIN32
        dict_fd = open(dpath.c_str(), O_RDONLY);
        if (dict_fd >= 0) {
            struct stat st;
            if (fstat(dict_fd, &st) == 0 && st.st_size > 0) {
                void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, dict_fd, 0);
                if (p != MAP_FAILED) { dict_mm = (const unsigned char*)p; dict_mm_len = (size_t)st.st_size;
                                       unlink(dpath.c_str()); mapped = true; }
            }
            if (!mapped) { close(dict_fd); dict_fd = -1; }
        }
#endif
        if (!mapped) {
            FILE * rf = fopen(dpath.c_str(), "rb");
            if (!rf) return false;                         // keep resident dict
            fseek(rf, 0, SEEK_END); long sz = ftell(rf); fseek(rf, 0, SEEK_SET);
            dict_fallback.resize(sz > 0 ? sz : 0);
            if (sz > 0) { size_t rd = fread(&dict_fallback[0], 1, sz, rf); (void)rd; }
            fclose(rf);
            dict_mm = (const unsigned char*)dict_fallback.data(); dict_mm_len = dict_fallback.size();
#ifndef _WIN32
            unlink(dpath.c_str());
#endif
        }
        // Section pointers: [terms V*8][poff V*8][dfc V*4][plen V*4].
        dterms_mm = (const uint64_t*)(dict_mm);
        dpoff_mm  = (const uint64_t*)(dict_mm + V * 8);
        ddfc_mm   = (const int32_t *)(dict_mm + V * 16);
        dplen_mm  = (const uint32_t*)(dict_mm + V * 16 + V * 4);
        dict_n = V;
        dict_mm_active = true;
        // Free the resident dictionary vectors (already empty if the dict was streamed).
        std::vector<uint64_t>().swap(df_terms);
        std::vector<int32_t>().swap(df_counts);
        std::vector<uint64_t>().swap(post_off);
        std::vector<uint32_t>().swap(post_len);
        return true;
    }

    // Streamed-dict helpers for build_inverted_external(): append a temp section to the dict file, and
    // read a temp section back into a resident vector when the streamed path fails.
    static bool append_file(FILE * out, const std::string & src) {
        FILE * in = fopen(src.c_str(), "rb"); if (!in) return false;
        std::vector<char> b(1 << 20); bool ok = true; size_t n;
        while ((n = fread(b.data(), 1, b.size(), in)) > 0)
            if (fwrite(b.data(), 1, n, out) != n) { ok = false; break; }
        fclose(in); return ok;
    }
    template<class T> static void read_all_into(const std::string & src, std::vector<T> & v) {
        FILE * in = fopen(src.c_str(), "rb"); if (!in) return;
        fseek(in, 0, SEEK_END); long sz = ftell(in); fseek(in, 0, SEEK_SET);
        if (sz > 0) { v.resize((size_t)sz / sizeof(T)); size_t rd = fread(v.data(), sizeof(T), v.size(), in); (void)rd; }
        fclose(in);
    }
    void spill_cleanup() {
        // close retained handles (POSIX: they were unlinked at open, so this reclaims the space),
        // and best-effort remove any named shards (Windows, or a partial open before unlink).
        for (auto f : spill_fp) if (f) fclose(f);
        spill_fp.clear();
        if (!spill_prefix.empty())
            for (int k = 0; k < SPILL_SHARDS; k++)
                remove((spill_prefix + "." + std::to_string(k)).c_str());
        spill_active = false;
    }

    float avgdl = 0;
    long long total_words = 0;  // running sum of doc lengths (keeps avgdl O(1) per add)
    int N = 0;

    // Sub-chunk -> parent mapping (for 1K sub-chunk indexing).
    // When BM25 finds a sub-chunk, map back to the parent file for full context retrieval.
    // A file registers a contiguous range [start, start+len) of sub-chunk ids, all mapping to
    // `start` (its first chunk id). Files are ingested in chunk-id order (the chunk_id counter is
    // monotonic and never resets, even across --append), so the ranges are strictly increasing and
    // non-overlapping and are stored as per-file (start, len) pairs in sorted parallel vectors.
    // get_parent_id() returns the largest start <= cid, if cid is inside that file's range.
    // Invariant: register_parent_range() must be called in non-decreasing `start` order
    // (guaranteed by sequential ingest); upper_bound relies on it.
    std::vector<int32_t> parent_starts;                 // sorted, one per file (== parent/first chunk id)
    std::vector<int32_t> parent_lens;                   // parallel: n_sub for that file
    std::unordered_map<int, std::string> parent_source; // parent_id -> file path (no @offset)
    void register_parent_range(int start, int len) {
        parent_starts.push_back(start);
        parent_lens.push_back(len);
    }
    // Index of the file range containing cid, or -1 if cid is standalone (before the first file,
    // or in a gap between file ranges, e.g. a non-sub-chunk doc). Shared by the getters below.
    int parent_range_of(int cid) const {
        auto it = std::upper_bound(parent_starts.begin(), parent_starts.end(), cid);
        if (it == parent_starts.begin()) return -1;
        size_t i = (size_t)(it - parent_starts.begin()) - 1;
        return (cid < parent_starts[i] + parent_lens[i]) ? (int)i : -1;
    }

    // Article-boundary metadata (from split_wiki_xml.py "[Article: T] [ID: I]" headers).
    // parent_id (first sub-chunk of a file) -> {title, article_id}; empty for corpora
    // without article headers. A sub-chunk's position within its article is
    // (chunk_id - parent_id). Enables per-chunk article attribution and citation.
    struct article_meta_t { std::string title; std::string article_id; int n_chunks = 0; };
    std::unordered_map<int, article_meta_t> article_meta; // parent_id -> article metadata

    // Article metadata for any (sub-)chunk; empty title if the corpus has no headers.
    article_meta_t get_article_meta(int chunk_id) const {
        int pid = get_parent_id(chunk_id);
        auto it = article_meta.find(pid);
        return it != article_meta.end() ? it->second : article_meta_t{};
    }

    // Get parent file source for a sub-chunk (strips @offset). This applies to every sub-chunk
    // of a file, including the first (cid == parent_id), so it range-checks rather than
    // shortcutting on pid == cid.
    std::string get_parent_source(int chunk_id) const {
        std::shared_lock<std::shared_mutex> lk(mu);
        int i = parent_range_of(chunk_id);
        if (i >= 0) {
            auto ps = parent_source.find(parent_starts[i]);
            if (ps != parent_source.end()) return ps->second;
        }
        // Not a sub-chunk (or parent has no recorded source), return regular source
        return get_source_nolock(chunk_id);
    }

    // Get parent chunk ID for deduplication
    int get_parent_id(int chunk_id) const {
        int i = parent_range_of(chunk_id);
        return i >= 0 ? parent_starts[i] : chunk_id;
    }

    bm25_index() {
        cipher.init();
    }

    ~bm25_index() {
#ifndef _WIN32
        if (text_mm && text_mm_len && text_fd >= 0) munmap((void*)text_mm, text_mm_len);
        if (text_fd >= 0) close(text_fd);
        if (post_mm && post_mm_len && post_fd >= 0) munmap((void*)post_mm, post_mm_len);
        if (post_fd >= 0) close(post_fd);
        if (dict_mm && dict_mm_len && dict_fd >= 0) munmap((void*)dict_mm, dict_mm_len);
        if (dict_fd >= 0) close(dict_fd);
#endif
        if (text_fp) fclose(text_fp);
        for (auto f : spill_fp) if (f) fclose(f);   // POSIX: unlinked at open, so close reclaims space
    }

    // Enable disk-backed text before a batch ingest. `path` is a private file (unlinked after mmap,
    // so nothing remains on disk). If the file can't be opened, stays in resident mode (a no-op).
    void open_text_store(const std::string & path) {
        if (text_fp || text_mm) return;                // already active
        text_path = path;
        text_fp = fopen(path.c_str(), "wb");
        text_on_disk = (text_fp != nullptr);
        text_woff = 0;
    }

    // Close the write handle and mmap the store read-only. Call once, after the batch ingest loop.
    void finalize_text_store() {
        if (text_fp) { fflush(text_fp); fclose(text_fp); text_fp = nullptr; }
        if (!text_on_disk || text_woff == 0 || text_path.empty()) return;  // nothing was written
#ifndef _WIN32
        text_fd = open(text_path.c_str(), O_RDONLY);
        if (text_fd >= 0) {
            struct stat st;
            if (fstat(text_fd, &st) == 0 && st.st_size > 0) {
                void * p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, text_fd, 0);
                if (p != MAP_FAILED) {
                    text_mm = (const unsigned char*)p; text_mm_len = (size_t)st.st_size;
                    unlink(text_path.c_str());          // mapping stays valid; no residue
                    return;
                }
            }
            close(text_fd); text_fd = -1;
        }
#endif
        // mmap unavailable (or Windows): read the file into memory so get_text still works.
        FILE * rf = fopen(text_path.c_str(), "rb");
        if (rf) { fseek(rf,0,SEEK_END); long sz=ftell(rf); fseek(rf,0,SEEK_SET);
                  text_fallback.resize(sz>0?sz:0); if(sz>0){ size_t rd=fread(&text_fallback[0],1,sz,rf); (void)rd; } fclose(rf);
                  text_mm=(const unsigned char*)text_fallback.data(); text_mm_len=text_fallback.size();
#ifndef _WIN32
                  unlink(text_path.c_str());
#endif
        } else { text_on_disk = false; }               // give up: resident mode (docs kept their text)
    }

    // Text offsets. The disk ciphertext store is written sequentially, so a doc's byte offset is the
    // prefix sum of text_len. During ingest offsets are recorded in doc_text_off. At finalize, if every
    // doc is a pure sequential disk append, only a sparse checkpoint (one per TEXT_OFF_CKPT docs) is
    // kept and the per-doc array is freed; get_text reconstructs. Resident or mixed (disk-full
    // fallback) corpora keep the explicit per-doc array.
    static constexpr size_t TEXT_OFF_CKPT = 64;
    std::vector<int64_t> doc_text_off;    // per-doc offset (>= 0 disk, < 0 = -(blob+1)); freed if reconstructable
    std::vector<int64_t> text_off_ckpt;   // sparse checkpoint of doc_text_off (disk-sequential mode only)
    bool text_off_reconstruct = false;    // true: doc_text_off freed; reconstruct from ckpt + text_len prefix
    // Docs covered by the checkpoint in reconstruct mode. Docs added after finalize (resident text) append
    // their explicit offsets to the freed doc_text_off, so doc_text_off[pos - text_off_base_n] holds them.
    size_t text_off_base_n = 0;
    void finalize_text_offsets() {
        // Reconstruct only if every doc is a pure sequential disk append: off[i] == sum(text_len[0..i-1]),
        // all >= 0. Any resident blob (< 0) or gap (disk-full fallback) keeps explicit offsets.
        bool seq = text_on_disk && text_mm && doc_text_off.size() == docs.size() && !docs.empty();
        if (seq) {
            int64_t exp = 0;
            for (size_t i = 0; i < doc_text_off.size(); i++) {
                if (doc_text_off[i] != exp) { seq = false; break; }
                exp += docs[i].text_len;
            }
        }
        if (seq) {
            const size_t n = doc_text_off.size();
            text_off_ckpt.assign((n + TEXT_OFF_CKPT - 1) / TEXT_OFF_CKPT, 0);
            for (size_t c = 0; c < text_off_ckpt.size(); c++)
                text_off_ckpt[c] = doc_text_off[c * TEXT_OFF_CKPT];
            std::vector<int64_t>().swap(doc_text_off);   // free the per-doc array
            text_off_base_n = n;
            text_off_reconstruct = true;
        } else {
            text_off_reconstruct = false;                // keep doc_text_off explicit
        }
    }
    // The doc's ciphertext offset: reconstructed (ckpt + short text_len sum) or explicit. `pos` = index in docs[].
    int64_t text_off_at(size_t pos) const {
        if (!text_off_reconstruct) return pos < doc_text_off.size() ? doc_text_off[pos] : 0;
        if (pos >= text_off_base_n) {                    // added after finalize: explicit offset
            const size_t j = pos - text_off_base_n;
            return j < doc_text_off.size() ? doc_text_off[j] : 0;
        }
        int64_t off = text_off_ckpt[pos / TEXT_OFF_CKPT];
        for (size_t j = (pos / TEXT_OFF_CKPT) * TEXT_OFF_CKPT; j < pos; j++) off += docs[j].text_len;
        return off;   // sequential disk mode: always >= 0
    }

    // ALPHA_BM25_STEM: conservative English stemmer (default off)
    // Paraphrased queries differ from documents morphologically ("increase" vs "increased"); without
    // stemming these are different terms and the matching unigram and bigrams are lost.
    //
    // Level 1 applies Porter steps 1a, 1b, 1c and 5 only. Steps 2/3/4 (the -ational/-iveness/-ement
    // families) are the aggressive ones that collide unrelated words (university->univers,
    // organization->organ), which costs precision on a mixed prose + code corpus. Step 5 is needed:
    // without 5a, increase/increased do not unify. With 1a/1b/1c/5, increase/increased/increases/
    // increasing all collapse to "increas".
    //
    // Tokens containing a digit or hyphen, or shorter than 4 chars, are never stemmed, which protects
    // identifiers (OPAL-742, file extensions, marker tokens) from being mangled.
    //
    // Off by default: stemming raises query/document term coverage but does not reliably improve
    // ranking. When the gap between query and document is semantic, no suffix rule bridges it, and
    // merging variants only reshuffles the competing documents. It also inflates df, which pushes
    // borderline bigrams out of the rare-anchor gate; if enabled, recalibrate ALPHA_BM25_ANCHOR_DF_PPM.
    static bool stem_enabled() {
        static const bool v = []() {
            const char * e = getenv("ALPHA_BM25_STEM");
            return e && atoi(e) != 0;
        }();
        return v;
    }
    // Stem level:
    //   0 / unset -> no stemming
    //   1         -> Porter steps 1a/1b/1c/5 only
    //   2         -> full Porter, adding steps 2/3/4 (the -ion/-ive/-ational/-ity families)
    // Level 2 helps graded-relevance tasks (e.g. BEIR scifact, nfcorpus) slightly over level 1.
    static int stem_level() {
        static const int v = []() {
            const char * e = getenv("ALPHA_BM25_STEM");
            return e ? atoi(e) : 0;
        }();
        return v;
    }
    static bool st_is_cons(const std::string & w, int i) {
        char c = w[i];
        if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u') return false;
        if (c=='y') return (i==0) ? true : !st_is_cons(w, i-1);
        return true;
    }
    static int st_measure(const std::string & w) {           // count of VC sequences
        int n = 0; size_t i = 0; const size_t L = w.size();
        while (i < L && st_is_cons(w,(int)i)) i++;            // leading consonants
        while (i < L) {
            while (i < L && !st_is_cons(w,(int)i)) i++;       // vowels
            if (i >= L) break;
            n++;
            while (i < L && st_is_cons(w,(int)i)) i++;        // consonants
        }
        return n;
    }
    static bool st_has_vowel(const std::string & w) {
        for (size_t i = 0; i < w.size(); i++) if (!st_is_cons(w,(int)i)) return true;
        return false;
    }
    static bool st_double_cons_end(const std::string & w) {
        size_t L = w.size();
        return L >= 2 && w[L-1] == w[L-2] && st_is_cons(w,(int)L-1);
    }
    static bool st_cvc(const std::string & w) {               // cvc where final c is not w,x,y
        size_t L = w.size();
        if (L < 3) return false;
        if (!st_is_cons(w,(int)L-1) || st_is_cons(w,(int)L-2) || !st_is_cons(w,(int)L-3)) return false;
        char c = w[L-1];
        return !(c=='w' || c=='x' || c=='y');
    }
    static bool st_ends(const std::string & w, const char * suf, size_t & stem_len) {
        size_t sl = strlen(suf);
        if (w.size() < sl) return false;
        if (w.compare(w.size()-sl, sl, suf) != 0) return false;
        stem_len = w.size() - sl;
        return true;
    }
    static std::string stem_word(const std::string & in) {
        if (in.size() < 4) return in;
        for (char c : in) if (!std::isalpha((unsigned char)c)) return in;   // digits/hyphens: leave alone
        std::string w = in;
        size_t sl;
        // --- step 1a: plurals
        if      (st_ends(w,"sses",sl)) w = w.substr(0,sl) + "ss";
        else if (st_ends(w,"ies", sl)) w = w.substr(0,sl) + "i";
        else if (st_ends(w,"ss",  sl)) { /* keep */ }
        else if (st_ends(w,"s",   sl)) w = w.substr(0,sl);
        // --- step 1b: -eed / -ed / -ing
        bool step1b_stripped = false;
        if (st_ends(w,"eed",sl)) {
            if (st_measure(w.substr(0,sl)) > 0) w = w.substr(0,sl) + "ee";
        } else if (st_ends(w,"ed",sl) && st_has_vowel(w.substr(0,sl))) {
            w = w.substr(0,sl); step1b_stripped = true;
        } else if (st_ends(w,"ing",sl) && st_has_vowel(w.substr(0,sl))) {
            w = w.substr(0,sl); step1b_stripped = true;
        }
        if (step1b_stripped) {
            if      (st_ends(w,"at",sl) || st_ends(w,"bl",sl) || st_ends(w,"iz",sl)) w += "e";
            else if (st_double_cons_end(w)) {
                char c = w[w.size()-1];
                if (c!='l' && c!='s' && c!='z') w.erase(w.size()-1);
            }
            else if (st_measure(w) == 1 && st_cvc(w)) w += "e";
        }
        // --- step 1c: y -> i
        if (st_ends(w,"y",sl) && st_has_vowel(w.substr(0,sl))) w = w.substr(0,sl) + "i";
        // --- steps 2/3/4: full Porter, level 2 only. Standard Porter order is 1a,1b,1c,2,3,4,5, so
        // these sit exactly where the level-1 path jumps straight from 1c to 5. Each table entry is
        // applied only when the measure of the remaining stem clears the step's threshold, which is
        // what keeps "university" (m=0 stem for -iti) from collapsing to "univers".
        if (stem_level() >= 2) {
            struct Rule { const char * suf; const char * rep; };
            // step 2: m > 0
            static const Rule S2[] = {
                {"ational","ate"}, {"tional","tion"}, {"enci","ence"}, {"anci","ance"},
                {"izer","ize"},    {"abli","able"},   {"alli","al"},   {"entli","ent"},
                {"eli","e"},       {"ousli","ous"},   {"ization","ize"},{"ation","ate"},
                {"ator","ate"},    {"alism","al"},    {"iveness","ive"},{"fulness","ful"},
                {"ousness","ous"}, {"aliti","al"},    {"iviti","ive"}, {"biliti","ble"},
            };
            for (const Rule & r : S2) {
                if (st_ends(w, r.suf, sl) && st_measure(w.substr(0,sl)) > 0) {
                    w = w.substr(0,sl) + r.rep; break;
                }
            }
            // step 3: m > 0
            static const Rule S3[] = {
                {"icate","ic"}, {"ative",""}, {"alize","al"}, {"iciti","ic"},
                {"ical","ic"},  {"ful",""},   {"ness",""},
            };
            for (const Rule & r : S3) {
                if (st_ends(w, r.suf, sl) && st_measure(w.substr(0,sl)) > 0) {
                    w = w.substr(0,sl) + r.rep; break;
                }
            }
            // step 4: m > 1. "ion" is special-cased: only after s or t, per Porter.
            static const char * S4[] = {
                "al","ance","ence","er","ic","able","ible","ant","ement","ment","ent",
                "ou","ism","ate","iti","ous","ive","ize",
            };
            bool done4 = false;
            if (st_ends(w,"ion",sl) && sl > 0 && (w[sl-1]=='s' || w[sl-1]=='t')
                && st_measure(w.substr(0,sl)) > 1) {
                w = w.substr(0,sl); done4 = true;
            }
            if (!done4) {
                for (const char * suf : S4) {
                    if (st_ends(w, suf, sl) && st_measure(w.substr(0,sl)) > 1) {
                        w = w.substr(0,sl); break;
                    }
                }
            }
        }
        // --- step 5: final -e, and -ll (without 5a, increase/increased never unify)
        if (st_ends(w,"e",sl)) {
            int m = st_measure(w.substr(0,sl));
            if (m > 1 || (m == 1 && !st_cvc(w.substr(0,sl)))) w = w.substr(0,sl);
        }
        if (st_measure(w) > 1 && st_double_cons_end(w) && w[w.size()-1]=='l') w.erase(w.size()-1);
        return w.empty() ? in : w;
    }

    // Tokenize text into lowercase words (>= 2 chars)
    // Also produces bigrams and preserves hyphenated compound tokens (e.g., OPAL-742).
    // `surfaces` (optional): parallel vector of the unstemmed surface form for each emitted term.
    // get_idf_keywords() needs it because its output is shown to a model, and "temperatur" is not
    // a useful keyword. Index/score paths use the stemmed term; only display uses the surface.
    // `want_bigrams` (default true): emit the forward bigrams. Indexing always passes true, which
    // the rare-bigram anchor path and union mode's bigram arm rely on. The plain query arm passes
    // false: bigrams in a plain query only add spurious partial matches (the dl/avgdl length norm
    // cancels, since avgdl comes from the same inflated counts), and dropping them improves nDCG.
    static std::vector<std::string> tokenize(const std::string & text,
                                             std::vector<std::string> * surfaces = nullptr,
                                             bool want_bigrams = true) {
        // Pass 1: split into raw tokens (alpha + digits + hyphens kept together)
        std::vector<std::string> raw_tokens;
        std::string tok;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            bool is_alnum = std::isalpha((unsigned char)c) || std::isdigit((unsigned char)c);
            // Keep hyphens inside tokens: "OPAL-742" stays together
            bool is_inner_hyphen = (c == '-' && !tok.empty() && i + 1 < text.size() &&
                                    (std::isalnum((unsigned char)text[i + 1])));
            if (is_alnum || is_inner_hyphen) {
                tok += (char)std::tolower((unsigned char)c);
            } else {
                if (tok.size() >= 2) raw_tokens.push_back(tok);
                tok.clear();
            }
        }
        if (tok.size() >= 2) raw_tokens.push_back(tok);

        // Pass 1b: stemmed parallel of raw_tokens. Stems are computed once per raw token and
        // reused for the unigram and both bigram positions, so `temperature increased` and
        // `temperature increase` produce the same bigram `temperatur_increas`. Stemming after
        // bigram assembly would not achieve that.
        const bool do_stem = stem_enabled();
        std::vector<std::string> stem_tokens;
        if (do_stem) {
            stem_tokens.reserve(raw_tokens.size());
            for (auto & t : raw_tokens) stem_tokens.push_back(stem_word(t));
        }
        auto S = [&](size_t i) -> const std::string & {
            return do_stem ? stem_tokens[i] : raw_tokens[i];
        };

        // Pass 2: build output = unigrams + bigrams + compound tokens
        std::vector<std::string> words;
        words.reserve(raw_tokens.size() * 2);
        if (surfaces) { surfaces->clear(); surfaces->reserve(raw_tokens.size() * 2); }
        auto emit = [&](const std::string & term, const std::string & surface) {
            words.push_back(term);
            if (surfaces) surfaces->push_back(surface);
        };

        for (size_t i = 0; i < raw_tokens.size(); i++) {
            auto & t = raw_tokens[i];

            // Unigram: always add the full token
            emit(S(i), t);

            // If token contains hyphen (compound like "opal-742"), also add parts as unigrams.
            // Parts are not stemmed: hyphenated tokens are identifiers, which stem_word() already
            // leaves alone.
            if (t.find('-') != std::string::npos) {
                size_t pos = 0, next;
                while ((next = t.find('-', pos)) != std::string::npos) {
                    std::string part = t.substr(pos, next - pos);
                    if (part.size() >= 2) emit(part, part);
                    pos = next + 1;
                }
                std::string last = t.substr(pos);
                if (last.size() >= 2) emit(last, last);
            }

            // Bigram: consecutive token pair (e.g., "activation" + "code" -> "activation_code")
            if (want_bigrams && i + 1 < raw_tokens.size()) {
                emit(S(i) + "_" + S(i + 1), raw_tokens[i] + "_" + raw_tokens[i + 1]);
            }
        }
        return words;
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lk(mu);
        docs.clear();
        doc_tf_terms.clear(); doc_tf_counts.clear();   // side arrays aligned with docs[]
        doc_source_off.clear();
        doc_text_off.clear(); text_off_ckpt.clear(); text_off_reconstruct = false; text_off_base_n = 0;
        df.clear();
        df_terms.clear(); df_counts.clear(); df_finalized = false;
        ov_df.clear(); ov_chunk.clear(); ov_nwords.clear(); ov_tf_terms.clear(); ov_tf_counts.clear();
        avgdl = 0;
        N = 0;
        total_words = 0;
        // reset the disk text store
#ifndef _WIN32
        if (text_mm && text_mm_len && text_fd >= 0) munmap((void*)text_mm, text_mm_len);
        if (text_fd >= 0) close(text_fd);
        text_fd = -1;
#endif
        if (text_fp) fclose(text_fp);
        text_fp = nullptr; text_mm = nullptr; text_mm_len = 0; text_woff = 0;
        text_on_disk = false; text_fallback.clear(); text_path.clear();
        // reset the inverted-index postings store
#ifndef _WIN32
        if (post_mm && post_mm_len && post_fd >= 0) munmap((void*)post_mm, post_mm_len);
        if (post_fd >= 0) close(post_fd);
        post_fd = -1;
#endif
        post_mm = nullptr; post_mm_len = 0; post_fallback.clear(); post_path.clear();
        post_off.clear(); post_len.clear(); doc_nwords.clear(); inverted_active = false;
        // reset the spill state (close any open shard handles, drop unconsumed shard files)
        for (auto f : spill_fp) if (f) fclose(f);
        spill_fp.clear();
        if (spill_active) spill_cleanup();
        spill_active = false; spill_records = 0; spill_prefix.clear();
        // reset the off-RAM dictionary
#ifndef _WIN32
        if (dict_mm && dict_mm_len && dict_fd >= 0) munmap((void*)dict_mm, dict_mm_len);
        if (dict_fd >= 0) close(dict_fd);
        dict_fd = -1;
#endif
        dict_mm = nullptr; dict_mm_len = 0; dict_fallback.clear(); dict_path.clear();
        dterms_mm = nullptr; ddfc_mm = nullptr; dpoff_mm = nullptr; dplen_mm = nullptr;
        dict_mm_active = false; dict_n = 0;
    }

    // Retrieve and decrypt stored text for a chunk ID
    std::string get_text(int chunk_id) const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return get_text_nolock(chunk_id);
    }
    std::string get_text_nolock(int chunk_id) const {
        for (size_t i = 0; i < docs.size(); i++) {
            const doc_info & doc = docs[i];
            if (doc.chunk_id == chunk_id) {
                if (doc.text_len <= 0) return "";
                const int64_t toff = text_off_at(i);         // reconstructed or explicit
                std::string ct;
                if (toff >= 0) {                             // disk store (mmap'd, or the in-memory fallback)
                    if (text_mm && toff + doc.text_len <= (int64_t)text_mm_len)
                        ct.assign((const char*)text_mm + toff, (size_t)doc.text_len);
                    else return "";
                } else {                                     // resident blob at -(toff+1)
                    size_t off = (size_t)(-toff - 1);
                    if (off + (size_t)doc.text_len <= resident_text.size())
                        ct.assign(resident_text.data() + off, (size_t)doc.text_len);
                    else return "";
                }
                return cipher.transform(ct, chunk_id);
            }
        }
        return "";
    }

    // Resident-RAM breakdown (bytes). A node is ~40 bytes for <u64,int> maps (next ptr + pair + malloc
    // header); the bucket array is bucket_count * 8. Used to profile where serve RSS goes at scale.
    struct resident_breakdown { size_t tf=0, df=0, source=0, docstruct=0, maps=0, text=0, inverted=0; };
    resident_breakdown resident() const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return resident_nolock();
    }
    resident_breakdown resident_nolock() const {
        resident_breakdown r;
        r.df = df_finalized ? (df_terms.capacity()*8 + df_counts.capacity()*4)
                            : (df.bucket_count()*8 + (size_t)df.size()*40);
        if (!ov_df.empty()) r.df += ov_df.bucket_count()*8 + (size_t)ov_df.size()*40;   // post-finalize overlay df
        for (size_t oi = 0; oi < ov_tf_terms.size(); oi++)          // post-finalize overlay tf (inverted mode)
            r.tf += ov_tf_terms[oi].capacity()*8 + ov_tf_counts[oi].capacity()*4;
        // Inverted mode: postings live in the mmap'd file (not counted here); the resident cost is the
        // per-term offset table plus the doc-length array. Freed tf reads 0 below.
        r.inverted = post_off.capacity()*8 + post_len.capacity()*4 + doc_nwords.capacity()*4;
        // doc_text_off is freed at finalize when offsets are reconstructable; then only the sparse
        // checkpoint stays resident.
        r.docstruct = docs.size() * sizeof(doc_info)
                    + doc_text_off.capacity()*8 + text_off_ckpt.capacity()*8;
        // tf lives in the side arrays (empty in spill mode).
        for (size_t di = 0; di < doc_tf_terms.size(); di++)
            r.tf += doc_tf_terms[di].capacity()*8 + doc_tf_counts[di].capacity()*4;
        r.tf += doc_tf_terms.capacity()*sizeof(std::vector<uint64_t>)
              + doc_tf_counts.capacity()*sizeof(std::vector<int32_t>);
        r.text = resident_text.capacity();          // one blob (empty in disk mode)
        // source = the interned table (persistent) + the ingest-only dedup map.
        for (const auto & s : source_tab) r.source += sizeof(std::string) + s.capacity();
        r.source += source_intern.bucket_count()*8 + (size_t)source_intern.size()*56;
        r.source += doc_source_off.capacity()*4;   // 4 bytes/sub-chunk offset; path shared per file
        // sub-chunk -> parent: two int32 vectors, one entry per file.
        r.maps = parent_starts.capacity()*4 + parent_lens.capacity()*4
               + parent_source.bucket_count()*8  + (size_t)parent_source.size()*56
               + article_meta.bucket_count()*8   + (size_t)article_meta.size()*88;
        return r;
    }

    // Build-peak profiler (ALPHA_BUILD_MEMPROF)
    // Prints peak RSS (getrusage) and the resident component breakdown at a named build phase, plus
    // per-doc and per-term byte costs, so the peak for a large corpus can be projected from a small ingest.
    void log_build_mem(const char * phase) const {
        if (!getenv("ALPHA_BUILD_MEMPROF")) return;
        double peak_gb = 0.0;
#ifndef _WIN32
        struct rusage ru; if (getrusage(RUSAGE_SELF, &ru) == 0) peak_gb = ru.ru_maxrss / 1048576.0; // KB -> GB
#endif
        resident_breakdown r = resident_nolock();
        const size_t nd = docs.size(), nt = df_terms.size();
        auto G = [](size_t b){ return b / 1e9; };
        fprintf(stderr,
            "[buildmem %-10s] peakRSS=%.2fG | docs=%zu terms=%zu || docstruct=%.2fG source=%.2fG text=%.2fG "
            "tf=%.2fG dict=%.2fG inverted=%.2fG maps=%.2fG || perdoc=%zuB perterm=%zuB\n",
            phase, peak_gb, nd, nt, G(r.docstruct), G(r.source), G(r.text), G(r.tf), G(r.df), G(r.inverted),
            G(r.maps),
            nd ? (r.docstruct + r.source + r.text + r.tf + r.maps) / nd : 0,
            nt ? (r.df + r.inverted) / std::max<size_t>(nt,1) : 0);
        fflush(stderr);
    }

    // Retrieve source filename for a chunk ID
    std::string get_source(int chunk_id) const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return get_source_nolock(chunk_id);
    }
    std::string get_source_nolock(int chunk_id) const {
        for (size_t i = 0; i < docs.size(); i++) {
            if (docs[i].chunk_id != chunk_id) continue;
            int sid = docs[i].source_id;
            if (sid < 0 || sid >= (int)source_tab.size()) return "";
            const std::string & path = source_tab[sid];
            int32_t off = (i < doc_source_off.size()) ? doc_source_off[i] : -1;
            return off < 0 ? path : path + "@" + std::to_string(off);   // rejoin the original string
        }
        return "";
    }

    // Chunk id of the i-th document in insertion order, or -1 if out of range. For callers that walk the
    // document list while adds may be running (they must not hold references into docs[]).
    int chunk_id_at(size_t i) const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return i < docs.size() ? docs[i].chunk_id : -1;
    }

    void add_document(int chunk_id, const std::string & text, const std::string & source = "") {
        std::unique_lock<std::shared_mutex> lk(mu);
        doc_info doc;
        doc.chunk_id = chunk_id;
        std::string enc = cipher.transform(text, chunk_id);  // ChaCha20 encrypt
        doc.text_len = (int32_t)enc.size();
        int64_t toff;                                        // offset recorded in doc_text_off
        if (text_on_disk && text_fp) {
            // Stream the ciphertext to the disk store; keep only (offset,len) resident.
            toff = text_woff;                                // >= 0: disk store
            if (!enc.empty() && fwrite(enc.data(), 1, enc.size(), text_fp) != enc.size()) {
                // write failed (e.g. disk full): stop writing; this and later docs go to resident_text.
                // Docs written before the failure keep their disk offsets and are served from the mmap.
                fclose(text_fp); text_fp = nullptr;
                toff = -(int64_t)resident_text.size() - 1;   // < 0: resident blob
                resident_text.append(enc);
            } else {
                text_woff += doc.text_len;
            }
        } else {
            toff = -(int64_t)resident_text.size() - 1;       // resident mode: < 0 = resident blob
            resident_text.append(enc);
        }
        int32_t soff;                                        // intern the path, keep the offset
        doc.source_id = intern_source_path(source, soff);

        // Tokenize plaintext, SipHash terms -> sorted parallel (term, count) arrays.
        auto words = tokenize(text);
        doc.n_words = (int)words.size();
        if (words.empty() && !text.empty()) {
            // Non-empty text, no searchable terms: almost always a non-Latin script. Warn once, then
            // just count, so a 1M-document CJK corpus does not produce 1M log lines.
            if (n_docs_no_terms.fetch_add(1) == 0) {
                fprintf(stderr, "bm25: WARNING chunk %d has non-empty text but ZERO searchable terms. "
                                "tokenize() is ASCII-only, so non-Latin scripts (Chinese/Japanese/"
                                "Korean/Cyrillic) index as empty and are NOT retrievable by BM25. "
                                "Further occurrences counted, not logged.\n", chunk_id);
            }
        }

        std::vector<uint64_t> hashes;
        hashes.reserve(words.size());
        for (auto & w : words) hashes.push_back(cipher.hash_term(w));
        std::sort(hashes.begin(), hashes.end());

        size_t ndistinct = 0;
        for (size_t i = 0; i < hashes.size(); ) {
            size_t j = i; while (j < hashes.size() && hashes[j] == hashes[i]) j++;
            ndistinct++; i = j;
        }
        // Build into local arrays; they are moved into the side arrays (or spilled) below.
        std::vector<uint64_t> tf_terms;  tf_terms.reserve(ndistinct);
        std::vector<int32_t>  tf_counts; tf_counts.reserve(ndistinct);
        const bool post_fin = post_finalize();             // df is frozen: count terms in the overlay
        for (size_t i = 0; i < hashes.size(); ) {
            size_t j = i; while (j < hashes.size() && hashes[j] == hashes[i]) j++;
            tf_terms.push_back(hashes[i]);
            tf_counts.push_back((int32_t)(j - i));
            if (!df_streaming && !post_fin) df[hashes[i]]++;   // df built here only in incremental mode;
                                                               // streaming mode derives df from tf_terms in
                                                               // finalize_df()
            i = j;
        }

        // External inversion: spill this doc's postings to the shards (not stored resident);
        // df is built at finalize from shards. After finalize the doc goes to the overlay (tf arrays
        // too in inverted mode). Otherwise keep tf in the side arrays, aligned with docs[].
        const bool spilled = spill_active;
        if (spilled) spill_postings(tf_terms, tf_counts, (uint32_t)doc.chunk_id);
        const bool in_overlay = !spilled && post_fin && overlay_add(doc.chunk_id, doc.n_words, tf_terms, tf_counts);

        total_words += doc.n_words;
        docs.push_back(std::move(doc));
        doc_source_off.push_back(soff);                  // always resident (provenance), aligned with docs[]
        doc_text_off.push_back(toff);                    // build transient; freed at finalize if reconstructable
        if (!spilled && !in_overlay) {
            doc_tf_terms.push_back(std::move(tf_terms));
            doc_tf_counts.push_back(std::move(tf_counts));
        }
        N = (int)docs.size();

        // avgdl via running total, so each add is O(1).
        avgdl = (float)total_words / std::max(1, N);
    }

    // Parallel batch ingest
    // The expensive parts of ingest are pure and run in parallel:
    //   parallel : cipher.transform() (locals only, keyed by chunk_id), tokenize() (static),
    //              hash_term(), the sort, and the run-length collapse into tf_terms/tf_counts.
    //   serial   : the text-store fwrite (offsets must be sequential), docs.push_back, and the
    //              total_words accumulator.
    // The df map is not a contention point: batch ingest calls enable_df_streaming(), so
    // add_document's `df[...]++` is skipped and df is derived in finalize_df().
    //
    // Results are written into a pre-sized vector indexed by position and merged strictly in order,
    // so chunk_ids, text offsets and docs[] are identical to the serial path for any thread count.
    // ALPHA_INGEST_THREADS=N sets the thread count (default: hardware threads, max 64); 1 = serial.
    static int ingest_threads() {
        static const int v = []() {
            const char * e = getenv("ALPHA_INGEST_THREADS");
            int hw = (int)std::thread::hardware_concurrency();
            if (hw <= 0) hw = 4;
            int d = e ? atoi(e) : hw;              // default: all hardware threads
            if (d < 1) d = 1;
            if (d > 64) d = 64;
            return d;
        }();
        return v;
    }

    // Build the per-doc payload for one sub-chunk. Touches no member state, so it is safe to call
    // concurrently.
    void build_doc(doc_info & doc, std::vector<uint64_t> & tf_terms, std::vector<int32_t> & tf_counts,
                   int chunk_id, const std::string & text,
                   const std::string & source, std::string & enc_out) const {
        doc.chunk_id = chunk_id;
        enc_out = cipher.transform(text, chunk_id);      // ChaCha20; locals only
        // Source is not set here: build_doc is const/concurrent and interning touches member
        // state. The serial merge in add_documents_batch assigns doc.source_id via intern_source().
        (void)source;

        auto words = tokenize(text);                     // static member fn
        doc.n_words = (int)words.size();

        std::vector<uint64_t> hashes;
        hashes.reserve(words.size());
        for (auto & w : words) hashes.push_back(cipher.hash_term(w));
        std::sort(hashes.begin(), hashes.end());

        size_t ndistinct = 0;
        for (size_t i = 0; i < hashes.size(); ) {
            size_t j = i; while (j < hashes.size() && hashes[j] == hashes[i]) j++;
            ndistinct++; i = j;
        }
        tf_terms.reserve(ndistinct);
        tf_counts.reserve(ndistinct);
        for (size_t i = 0; i < hashes.size(); ) {
            size_t j = i; while (j < hashes.size() && hashes[j] == hashes[i]) j++;
            tf_terms.push_back(hashes[i]);
            tf_counts.push_back((int32_t)(j - i));
            i = j;
        }
    }

    // items: (text, source) pairs. chunk_ids are assigned first_chunk_id + index, so the caller
    // gets exactly the ids the serial loop would have produced.
    void add_documents_batch(int first_chunk_id,
                             const std::vector<std::pair<std::string, std::string>> & items) {
        const size_t n = items.size();
        if (n == 0) return;
        std::vector<doc_info>   built(n);
        std::vector<std::string> encs(n);
        std::vector<std::vector<uint64_t>> built_tf_terms(n);   // tf output per doc
        std::vector<std::vector<int32_t>>  built_tf_counts(n);

        const int nthr = std::min<int>(ingest_threads(), (int)n);
        if (nthr <= 1) {
            for (size_t i = 0; i < n; i++)
                build_doc(built[i], built_tf_terms[i], built_tf_counts[i],
                          first_chunk_id + (int)i, items[i].first, items[i].second, encs[i]);
        } else {
            std::vector<std::thread> pool;
            std::atomic<size_t> next{0};
            for (int t = 0; t < nthr; t++) {
                pool.emplace_back([&]() {
                    for (;;) {
                        size_t i = next.fetch_add(1);
                        if (i >= n) break;
                        build_doc(built[i], built_tf_terms[i], built_tf_counts[i],
                                  first_chunk_id + (int)i, items[i].first, items[i].second, encs[i]);
                    }
                });
            }
            for (auto & th : pool) th.join();
        }

        // Strictly ordered merge: identical to what the serial path would write. Only the merge mutates the
        // index, so only it holds the lock; the parallel build above reads nothing but the cipher key.
        std::unique_lock<std::shared_mutex> lk(mu);
        for (size_t i = 0; i < n; i++) {
            doc_info & doc = built[i];
            int32_t soff;                                     // path interning (serial, not in build_doc)
            doc.source_id = intern_source_path(items[i].second, soff);
            doc.text_len = (int32_t)encs[i].size();
            int64_t toff;                                     // offset recorded in doc_text_off
            if (text_on_disk && text_fp) {
                toff = text_woff;                             // >= 0: disk store
                if (!encs[i].empty() &&
                    fwrite(encs[i].data(), 1, encs[i].size(), text_fp) != encs[i].size()) {
                    fclose(text_fp); text_fp = nullptr;      // disk full: fall back to resident_text
                    toff = -(int64_t)resident_text.size() - 1;
                    resident_text.append(encs[i]);
                } else {
                    text_woff += doc.text_len;
                }
            } else {
                toff = -(int64_t)resident_text.size() - 1;    // resident mode: < 0 = resident blob
                resident_text.append(encs[i]);
            }
            // When not streaming, df is built here in the serial merge, as add_document does (one
            // increment per distinct term); build_doc leaves df untouched so it stays thread-safe.
            // No-op on the corpus_auto_ingest path, which enables df streaming before batching.
            // After finalize the df dictionary is frozen, so the terms are counted in the overlay instead.
            const bool post_fin = post_finalize();
            if (!df_streaming && !post_fin) {
                for (uint64_t term : built_tf_terms[i]) df[term]++;
            }
            // External inversion: spill this doc's postings to the shards (serial, so the FILE*
            // appends are safe). After finalize the doc goes to the overlay (tf arrays too in inverted
            // mode). Otherwise keep tf in the side arrays, index-aligned with docs[].
            const bool spilled = spill_active;
            if (spilled) spill_postings(built_tf_terms[i], built_tf_counts[i], (uint32_t)doc.chunk_id);
            const bool in_overlay = !spilled && post_fin
                && overlay_add(doc.chunk_id, doc.n_words, built_tf_terms[i], built_tf_counts[i]);
            total_words += doc.n_words;
            docs.push_back(std::move(doc));
            doc_source_off.push_back(soff);                // always resident (provenance), aligned with docs[]
            doc_text_off.push_back(toff);                  // build transient; freed at finalize if reconstructable
            if (!spilled && !in_overlay) {
                doc_tf_terms.push_back(std::move(built_tf_terms[i]));
                doc_tf_counts.push_back(std::move(built_tf_counts[i]));
            }
        }
        N = (int)docs.size();
        avgdl = (float)total_words / std::max(1, N);
    }

    // ALPHA_BM25_BIGRAM_FIRST (default off): documents matching a rare query bigram are ranked ahead
    // of unigram-only matches, instead of merely out-weighing them additively.
    //
    // BM25 is a sum, so many weak common-term matches can out-score one strong match, and the best
    // coincidental score among competitors grows with corpus size. A rare conjunction's candidate pool
    // does not grow that way, so conjunction precision holds up as the corpus scales.
    //
    // The rarity gate matters: there is no stopword removal, so queries also produce common bigrams
    // (`the_ancient`, `how_many`) that are typically orders of magnitude more frequent than the
    // discriminative ones. If the gate is too loose, every document already ahead of the target also
    // anchors and tiering is a no-op. The threshold is therefore sub-percent and set in parts per
    // million: anchor iff df <= N * ALPHA_BM25_ANCHOR_DF_PPM / 1e6 (default 500 ppm = 0.05%).
    //
    // This changes selection order only and never rescales a score, so within each tier the ranking
    // is unchanged. (An explicit IDF multiplier on bigrams breaks partial compound matching, e.g.
    // BERYL-858 vs BERYL-858-UPP.)
    //
    // Off by default because it can displace the score leader when the anchors are rare but
    // meaningless: generic code queries produce bigrams like `the_limit` or `not_none` that clear the
    // df gate, and paraphrased prose queries often share no discriminative bigram with the relevant
    // document, so the anchors promote some other document. It helps when queries quote their target
    // verbatim; enable per deployment there. Making it a default would need evaluation on labelled
    // relevance judgments.
    static bool bigram_first_enabled() {
        static const bool v = []() {
            const char * e = getenv("ALPHA_BM25_BIGRAM_FIRST");
            return e && atoi(e) != 0;       // absent => off
        }();
        return v;
    }

    // ALPHA_BM25_P1: query-bigram suppression (default off). ALPHA_BM25_P1=0 also leaves it off.
    //
    // Improves nDCG@10 on prose retrieval benchmarks (BEIR scifact, nfcorpus) but hurts precision@1
    // on code and identifier search: removing query bigrams removes adjacent-token phrase evidence,
    // which in code is what puts the exact chunk ahead of its near-identical siblings. Enable it per
    // deployment for prose corpora; leave it off for code, identifiers, and mixed corpora.
    static bool p1_enabled() {
        static const bool v = []() {
            const char * e = getenv("ALPHA_BM25_P1");
            return e && atoi(e) != 0;       // absent => off
        }();
        return v;
    }
    static int anchor_df_ppm() {
        static const int v = []() {
            const char * e = getenv("ALPHA_BM25_ANCHOR_DF_PPM");
            int x = e ? atoi(e) : 500;                 // 500ppm = 0.05% of the corpus
            return x < 1 ? 1 : (x > 1000000 ? 1000000 : x);
        }();
        return v;
    }
    // Minimum anchor count (ALPHA_BM25_MIN_ANCHORS, default 2): guards against a single incidental
    // anchor hijacking the ranking. A query built from ubiquitous keywords, e.g. "if the pointer is
    // null or the size is zero return false", may have exactly one bigram under the gate (`null_or`),
    // a meaningless function-word pair; every document containing it would outrank every unanchored
    // document regardless of score. The candidate set is small in that case, so a size-based trigger
    // would not catch it. With fewer than min_anchors anchors the query falls back to score-only
    // ordering. Queries with real content phrases typically have several anchors.
    static int min_anchors() {
        static const int v = []() {
            const char * e = getenv("ALPHA_BM25_MIN_ANCHORS");
            int x = e ? atoi(e) : 2;
            return x < 1 ? 1 : x;
        }();
        return v;
    }
    // Constituent gate (ALPHA_BM25_ANCHOR_MAXCONST_PPM): anchor quality, which the anchor count
    // does not capture.
    //
    // A bigram is only a concept if both of its words are themselves discriminative. IDF cannot
    // tell: `the_limit` is rare as a pairing, so it has high IDF, yet it means nothing. What
    // separates it from `ancient_settlement` is that it contains a near-ubiquitous word. A bigram
    // therefore anchors only if neither constituent unigram exceeds this df fraction, which rejects
    // rare pairings of common words while keeping real phrases.
    //
    // Default 300000 ppm = 30%: rejects true stopwords (the/is/or/not/and are far above) while
    // keeping moderately common content words such as `many`. ALPHA_BM25_ANCHOR_DEBUG prints the
    // constituent df to help tune it.
    static int anchor_max_const_ppm() {
        static const int v = []() {
            const char * e = getenv("ALPHA_BM25_ANCHOR_MAXCONST_PPM");
            int x = e ? atoi(e) : 300000;
            return x < 1 ? 1 : (x > 1000000 ? 1000000 : x);
        }();
        return v;
    }

    // Inverted-index scoring: same BM25 math and anchor logic as the forward score() below, but it
    // visits only the docs on a query term's posting list instead of scanning all N docs. Partial
    // scores accumulate into a dense array keyed by chunk_id; a doc scores nonzero iff it contains a
    // query term, so the top-k matches the forward scan (which also returns the zero-score docs,
    // sorted to the bottom). Ties break by (score desc, chunk_id asc) for determinism.
    // Caller holds mu (shared); only score() calls it.
    std::vector<std::pair<float, int>> score_inverted(const std::string & query_text, int bf_override) const {
        const bool bigram_first = (bf_override >= 0) ? (bf_override != 0) : bigram_first_enabled();
        // With P1 enabled, query bigrams are only kept for the bigram-first arm, which needs them
        // (an anchor is a low-df query bigram). With P1 off (default) bigrams are always emitted.
        const bool want_bg = !p1_enabled() || bigram_first;
        auto q_words = tokenize(query_text, nullptr, want_bg);
        const int  anchor_df_max = std::max(1, (int)((long long)N * anchor_df_ppm() / 1000000));

        // Same two passes as score(): df/idf per term, then the anchor verdict from constituent df.
        struct query_term { uint64_t hash; float boost; float idf; bool valid; bool anchor; int tidx; };
        std::vector<query_term> q_terms; q_terms.reserve(q_words.size());
        int n_anchor = 0;
        std::unordered_map<std::string, int> uni_df;
        std::vector<int> df_of_term; df_of_term.reserve(q_words.size());
        for (auto & w : q_words) {
            uint64_t h = cipher.hash_term(w);
            int dfc = df_of_nolock(h);
            float idf = dfc > 0 ? std::log((N - dfc + 0.5f) / (dfc + 0.5f) + 1.0f) : 0.0f;
            if (w.find('_') == std::string::npos) uni_df[w] = dfc;
            df_of_term.push_back(dfc);
            q_terms.push_back({h, 1.0f, idf, dfc > 0, false, dfc > 0 ? term_index(h) : -1});
        }
        const int const_df_max = std::max(1, (int)((long long)N * anchor_max_const_ppm() / 1000000));
        for (size_t i = 0; i < q_words.size(); i++) {
            const std::string & w = q_words[i];
            const int dfc = df_of_term[i];
            const size_t us = w.find('_');
            const bool is_bigram = us != std::string::npos;
            bool const_ok = true;
            if (is_bigram) {
                auto ia = uni_df.find(w.substr(0, us));
                auto ib = uni_df.find(w.substr(us + 1));
                int dfa = ia != uni_df.end() ? ia->second : -1;
                int dfb = ib != uni_df.end() ? ib->second : -1;
                const_ok = (dfa > 0 && dfa <= const_df_max) && (dfb > 0 && dfb <= const_df_max);
            }
            const bool anchor = bigram_first && is_bigram && dfc > 0 && dfc <= anchor_df_max && const_ok;
            if (anchor) n_anchor++;
            q_terms[i].anchor = anchor;
        }

        const bool use_anchor = bigram_first && n_anchor >= min_anchors();
        const size_t ND = doc_nwords.size();
        std::vector<float>   acc(ND, 0.0f);
        std::vector<uint8_t> anch;  if (use_anchor) anch.assign(ND, 0);
        std::vector<int>     touched;

        for (auto & qt : q_terms) {
            if (!qt.valid || qt.tidx < 0) continue;
            const unsigned char * p = post_mm + dict_poff((size_t)qt.tidx);
            const size_t len = dict_plen((size_t)qt.tidx);
            size_t i = 0; int prev = 0;
            while (i < len) {
                int doc = prev + (int)get_varint(p, i);
                prev = doc;
                int tf = (int)get_varint(p, i);
                if ((size_t)doc >= ND) continue;               // guard (dense array bound)
                float tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_nwords[doc] / avgdl));
                if (acc[doc] == 0.0f) touched.push_back(doc);
                acc[doc] += qt.idf * tf_norm * qt.boost;
                if (use_anchor && qt.anchor) anch[doc] = 1;
            }
        }

        // Post-finalize overlay docs are not in the postings: score each one by brute force over its sorted
        // tf arrays with the same terms, idf, length norm and accumulation order as the postings loop. A doc
        // is a candidate iff it contains a valid query term, as for the postings. Overlay docs are kept as
        // separate entries (not folded into acc[] by chunk_id), since an added chunk_id may repeat or fall
        // outside the dense range; the forward scorer also lists every doc separately.
        struct ov_hit { float s; int cid; uint8_t a; };
        std::vector<ov_hit> ov_hits;
        for (size_t o = 0; o < ov_chunk.size(); o++) {
            const std::vector<uint64_t> & tt = ov_tf_terms[o];
            const std::vector<int32_t>  & tc = ov_tf_counts[o];
            const int nwords = ov_nwords[o];
            float s = 0.0f; bool hit = false; uint8_t a = 0;
            for (auto & qt : q_terms) {
                if (!qt.valid) continue;
                auto lo = std::lower_bound(tt.begin(), tt.end(), qt.hash);
                if (lo == tt.end() || *lo != qt.hash) continue;
                int tf = tc[lo - tt.begin()];
                float tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * nwords / avgdl));
                s += qt.idf * tf_norm * qt.boost;
                hit = true;
                if (use_anchor && qt.anchor) a = 1;
            }
            if (hit) ov_hits.push_back({s, ov_chunk[o], a});
        }

        std::vector<std::pair<float, int>> results;
        results.reserve(touched.size() + ov_hits.size());
        if (use_anchor) {
            // Anchored first, then (score desc, chunk_id asc). The anchor flag travels with each entry
            // because overlay chunk_ids are not indexes into anch[].
            std::vector<ov_hit> tmp;
            tmp.reserve(touched.size() + ov_hits.size());
            for (int doc : touched) tmp.push_back({acc[doc], doc, anch[doc]});
            tmp.insert(tmp.end(), ov_hits.begin(), ov_hits.end());
            std::sort(tmp.begin(), tmp.end(), [](const ov_hit & a, const ov_hit & b) {
                if (a.a != b.a) return a.a > b.a;               // anchored first
                if (a.s != b.s) return a.s > b.s;
                return a.cid < b.cid;
            });
            for (auto & t : tmp) results.push_back({t.s, t.cid});
        } else {
            for (int doc : touched) results.push_back({acc[doc], doc});
            for (auto & h : ov_hits) results.push_back({h.s, h.cid});
            std::sort(results.begin(), results.end(), [](const auto & a, const auto & b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
        }
        return results;
    }

    // Score all documents against query, return sorted (score, chunk_id) pairs.
    // There is no explicit compound/bigram boost: compound and bigram IDF is naturally higher because
    // those terms are rarer, and an explicit multiplier breaks partial compound matching
    // (BERYL-858 vs BERYL-858-UPP).
    // `bf_override`: -1 = use the ALPHA_BM25_BIGRAM_FIRST env (normal), 0/1 = force off/on for this
    // call, so both settings can be compared on one live index without a restart.
    std::vector<std::pair<float, int>> score(const std::string & query_text, int bf_override = -1) const {
        // A query that tokenizes to nothing scores every document 0, so the caller gets an arbitrary
        // ranking with no indication anything went wrong. Warn once. Placed above the dispatch so it
        // also covers score_inverted(), whose only caller is score().
        if (!query_text.empty() && tokenize(query_text).empty()) {
            static std::atomic<bool> warned{false};
            bool expected = false;
            if (warned.compare_exchange_strong(expected, true)) {
                fprintf(stderr, "bm25: WARNING query produced ZERO searchable terms, so every document "
                                "scores 0 and the returned order is arbitrary. tokenize() is ASCII-only "
                                "-- non-Latin queries cannot match. Warned once per process.\n");
            }
        }
        std::shared_lock<std::shared_mutex> lk(mu);
        if (inverted_active) return score_inverted(query_text, bf_override);
        // Inverted mode, but the index was never built: a corpus ingested over HTTP and not finalized.
        // The forward scan still answers correctly, only slower; say once how to build the index.
        // (Not finalized automatically: a pause in ingest is not the end of it.)
        if (bm25_inverted() && warn_unfinalized && N > 0) {
            static std::atomic<bool> warned_unfinalized{false};
            bool expected = false;
            if (warned_unfinalized.compare_exchange_strong(expected, true)) {
                fprintf(stderr, "bm25: the corpus (%d documents) has no inverted index yet, so queries use the "
                                "slower forward scan. Call POST /api/corpus/finalize once ingest is complete "
                                "to build it. Warned once per process.\n", (int) N);
            }
        }
        return score_forward(query_text, bf_override);
    }

    // Forward-index scoring (the O(N) scan). Caller holds mu (shared); only score() calls it.
    std::vector<std::pair<float, int>> score_forward(const std::string & query_text, int bf_override) const {
        const bool bigram_first = (bf_override >= 0) ? (bf_override != 0) : bigram_first_enabled();
        // See score_inverted(): with ALPHA_BM25_P1 set, query bigrams are kept for the bigram-first
        // arm only; by default they are always emitted.
        const bool want_bg = !p1_enabled() || bigram_first;
        auto q_words = tokenize(query_text, nullptr, want_bg);
        const int  anchor_df_max = std::max(1, (int)((long long)N * anchor_df_ppm() / 1000000));

        // SipHash query terms and precompute df/idf once per term (O(q), not O(N*q)).
        struct query_term { uint64_t hash; float boost; float idf; bool valid; bool anchor; };
        std::vector<query_term> q_terms;
        q_terms.reserve(q_words.size());
        // Pass 1: df/idf for every query term, and record each unigram's df so a bigram's anchor
        // verdict can consult its constituents (tokenize always emits both words of a bigram as
        // unigrams in the same query, so every constituent is available here).
        int n_anchor = 0;
        std::unordered_map<std::string, int> uni_df;
        std::vector<int> df_of_term; df_of_term.reserve(q_words.size());
        for (auto & w : q_words) {
            uint64_t h = cipher.hash_term(w);
            int dfc = df_of_nolock(h);
            float idf = dfc > 0 ? std::log((N - dfc + 0.5f) / (dfc + 0.5f) + 1.0f) : 0.0f;
            if (w.find('_') == std::string::npos) uni_df[w] = dfc;
            df_of_term.push_back(dfc);
            q_terms.push_back({h, 1.0f, idf, dfc > 0, false});
        }

        // Pass 2: anchor decision. Needs pass 1 complete because it consults constituent df.
        const int const_df_max = std::max(1, (int)((long long)N * anchor_max_const_ppm() / 1000000));
        static const bool adbg = getenv("ALPHA_BM25_ANCHOR_DEBUG") != nullptr;
        for (size_t i = 0; i < q_words.size(); i++) {
            const std::string & w = q_words[i];
            const int dfc = df_of_term[i];
            const size_t us = w.find('_');
            const bool is_bigram = us != std::string::npos;
            int dfa = -1, dfb = -1;
            bool const_ok = true;
            if (is_bigram) {
                auto ia = uni_df.find(w.substr(0, us));
                auto ib = uni_df.find(w.substr(us + 1));
                dfa = ia != uni_df.end() ? ia->second : -1;
                dfb = ib != uni_df.end() ? ib->second : -1;
                // An unknown constituent (should not happen) fails closed: the anchor is refused.
                const_ok = (dfa > 0 && dfa <= const_df_max) && (dfb > 0 && dfb <= const_df_max);
            }
            const bool anchor = bigram_first && is_bigram && dfc > 0 && dfc <= anchor_df_max && const_ok;
            if (anchor) n_anchor++;
            q_terms[i].anchor = anchor;
            if (adbg) {
                static int printed = 0;
                if (printed < 400) {
                    fprintf(stderr, "[anchor-dbg] N=%d max=%d cmax=%d %-26s bigram=%d df=%-8d "
                                    "cdf=%d/%d const_ok=%d anchor=%d\n",
                            N, anchor_df_max, const_df_max, w.c_str(), (int)is_bigram, dfc,
                            dfa, dfb, (int)const_ok, (int)anchor);
                    printed++;
                }
            }
        }

        // Score one document. Returns the BM25 sum; sets `anchored` if any rare-bigram anchor hit.
        auto score_doc = [&](size_t d, bool & anchored) -> float {
            const std::vector<uint64_t> & tt = doc_tf_terms[d];
            const std::vector<int32_t>  & tc = doc_tf_counts[d];
            const int nwords = docs[d].n_words;
            float s = 0;
            for (auto & qt : q_terms) {
                if (!qt.valid) continue;
                // binary search the sorted per-doc tf arrays
                auto lo = std::lower_bound(tt.begin(), tt.end(), qt.hash);
                if (lo == tt.end() || *lo != qt.hash) continue;
                int tf = tc[lo - tt.begin()];
                if (tf == 0) continue;

                float tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * nwords / avgdl));
                s += qt.idf * tf_norm * qt.boost;
                if (qt.anchor) anchored = true;
            }
            return s;
        };

        // Fast path (the default): feature off, or fewer than min_anchors rare bigrams. Build
        // `results` directly, 8 bytes per doc, with no intermediate copy.
        if (!bigram_first || n_anchor < min_anchors()) {
            std::vector<std::pair<float, int>> results;
            results.reserve(docs.size());
            for (size_t d = 0; d < docs.size(); d++) {
                bool ignored = false;
                results.push_back({score_doc(d, ignored), docs[d].chunk_id});
            }
            // (score desc, chunk_id asc), the inverted scorer's order, so both paths rank ties alike
            // (std::sort is not stable: a score-only comparator left tie order unspecified)
            std::sort(results.begin(), results.end(), [](auto & a, auto & b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });
            return results;
        }

        // Anchored path: lexicographic, anchored docs first, then by score. The extra byte per doc
        // and the second pass are paid only here, when the feature is actually doing something.
        struct scored { float s; int cid; bool anchored; };
        std::vector<scored> tmp;
        tmp.reserve(docs.size());
        for (size_t d = 0; d < docs.size(); d++) {
            bool anchored = false;
            float s = score_doc(d, anchored);
            tmp.push_back({s, docs[d].chunk_id, anchored});
        }
        std::sort(tmp.begin(), tmp.end(), [](const scored & a, const scored & b) {
            if (a.anchored != b.anchored) return a.anchored;
            if (a.s != b.s) return a.s > b.s;
            return a.cid < b.cid;
        });

        std::vector<std::pair<float, int>> results;
        results.reserve(tmp.size());
        for (auto & t : tmp) results.push_back({t.s, t.cid});
        return results;
    }

    // Get top-N highest-IDF terms from a specific chunk.
    // Decrypts text, re-tokenizes, scores each word by global IDF.
    // Used by the digest pipeline (layer 1: BM25 IDF extraction).
    std::vector<std::string> get_idf_keywords(int chunk_id, int max_terms = 15) const {
        std::shared_lock<std::shared_mutex> lk(mu);
        std::string text = get_text_nolock(chunk_id);
        if (text.empty()) return {};

        // Score by the stemmed term (what df and the index are keyed on) but return the surface
        // form, since this output is fed to a model as keywords.
        std::vector<std::string> surfaces;
        auto words = tokenize(text, &surfaces);

        // Deduplicate and score by IDF
        std::unordered_map<std::string, float> word_idf;
        for (size_t wi = 0; wi < words.size(); wi++) {
            const std::string & w   = words[wi];
            const std::string & disp = wi < surfaces.size() ? surfaces[wi] : w;
            if (word_idf.count(disp)) continue;  // first occurrence only
            uint64_t h = cipher.hash_term(w);
            int dfc = df_of_nolock(h);
            if (dfc == 0) continue;
            float idf = std::log((N - dfc + 0.5f) / (dfc + 0.5f) + 1.0f);
            word_idf[disp] = idf;
        }

        // Sort by IDF descending
        std::vector<std::pair<float, std::string>> ranked;
        ranked.reserve(word_idf.size());
        for (auto & [w, idf] : word_idf) {
            ranked.push_back({idf, w});
        }
        std::sort(ranked.begin(), ranked.end(), [](auto & a, auto & b) {
            return a.first > b.first;
        });

        // Return top-N terms
        std::vector<std::string> result;
        for (int i = 0; i < std::min(max_terms, (int)ranked.size()); i++) {
            result.push_back(ranked[i].second);
        }
        return result;
    }

    // Get IDF keywords as a single space-separated string (for digest building)
    std::string get_idf_keywords_str(int chunk_id, int max_terms = 15) const {
        auto keywords = get_idf_keywords(chunk_id, max_terms);
        std::string result;
        for (auto & kw : keywords) {
            if (!result.empty()) result += " ";
            result += kw;
        }
        return result;
    }

    // Scored variant of top_k, for score-aware fusion (top_k() drops the score). Same cost: one
    // score() call. The truncation must match top_k() exactly, including k <= 0 (an empty list).
    std::vector<std::pair<float, int>> top_k_scored(const std::string & query, int k,
                                                    int bf_override = -1) const {
        auto ranked = score(query, bf_override);
        std::vector<std::pair<float, int>> out;
        const int n = std::min(k, (int) ranked.size());
        // Only documents that match a query term (score > 0). The forward scorer sorts all N docs, so
        // without this a query matching fewer than k documents would be padded with score-0 documents
        // in arbitrary order; the inverted scorer returns matches only. Both paths use the same rule.
        for (int i = 0; i < n && ranked[i].first > 0.0f; i++) out.push_back(ranked[i]);
        return out;
    }

    // Get top-k chunk IDs
    std::vector<int> top_k(const std::string & query, int k, int bf_override = -1) const {
        auto ranked = score(query, bf_override);
        std::vector<int> ids;
        for (int i = 0; i < std::min(k, (int)ranked.size()) && ranked[i].first > 0.0f; i++) {   // matches only (see above)
            ids.push_back(ranked[i].second);
        }
        return ids;
    }
};

// Routing strategy based on model size
enum class kv_chunk_routing {
    BM25_ONLY,       // <4B models: pure BM25 (no expansion)
    BM25_EXPAND,     // >=4B models: BM25 + query expansion via model
};

// Determine routing strategy from model parameter count
inline kv_chunk_routing select_routing_strategy(uint64_t n_params) {
    // Models <4B generate poor synonym expansions that hurt routing
    if (n_params < 4000000000ULL) {
        return kv_chunk_routing::BM25_ONLY;
    }
    return kv_chunk_routing::BM25_EXPAND;
}

// Max extended context (1M tokens)
static constexpr int KV_CHUNK_MAX_CONTEXT = 1048576;
