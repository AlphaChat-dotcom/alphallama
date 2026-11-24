#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>
#include <atomic>
#include <unordered_map>
#include <vector>

// DeepSeek-V4.1 Engram tables, kept on disk.
//
// A table is a [head_dim, n_rows] tensor of a GGUF file that is never loaded: the rows a ubatch needs are read with
// pread (after a POSIX_FADV_WILLNEED hint for all of them) and converted to F32. Two layouts are accepted:
//   - upstream llama.cpp: one tensor "per_layer_token_embd.weight" holding the tables of all Engram layers back to
//     back; the row ids of the hash (head offset + bucket) index it directly
//   - Project Maya shards: one tensor "blk.<il>.engram_embd.weight" per Engram layer; the row id is relative to the
//     first head offset of that layer
// The tensors may be in the model files themselves or in extra files (--shard / --ple, env ALPHA_ENGRAM_SHARDS).

struct llama_engram_table {
    struct source {
        std::string path;
        int         fd       = -1;
        uint64_t    offset   = 0;  // file offset of the tensor data
        ggml_type   type     = GGML_TYPE_F32;
        int64_t     n_rows   = 0;
        size_t      row_size = 0;  // bytes per row
        int64_t     row_base = 0;  // absolute row id of the first row
        mutable int fd_direct = -2; // O_DIRECT descriptor (ALPHA_ENGRAM_DIRECT), -2 = not opened yet
    };

    int64_t head_dim = 0;

    // one per Engram module (the same source for all of them in the single table layout)
    std::vector<source> modules;

    // statistics
    mutable std::atomic<uint64_t> n_rows_read{0};
    mutable std::atomic<uint64_t> t_read_us{0};

    // ALPHA_ENGRAM_CACHE_MB: rows read so far (raw bytes) per module (modules are read concurrently); filled until full
    struct module_cache {
        std::unordered_map<uint32_t, uint32_t> row_cache;   // row id -> slot
        std::vector<uint8_t>                   row_arena;   // slot * row_size
        size_t                                 slots_max = 0;
    };
    mutable std::vector<module_cache> caches;

    llama_engram_table() = default;
    llama_engram_table(const llama_engram_table &) = delete;
    llama_engram_table & operator=(const llama_engram_table &) = delete;
    ~llama_engram_table();

    bool empty() const { return modules.empty(); }

    // looks for the tables of the Engram layers (model layers `layers`) in `paths`; row_base[e] / n_rows[e] is the
    // absolute row range of module e. Throws std::runtime_error when a table is missing or does not fit.
    void open(const std::vector<std::string> & paths, const std::vector<int32_t> & layers,
              const std::vector<int64_t> & row_base, const std::vector<int64_t> & n_rows, int64_t head_dim);

    // rows [n] absolute row ids of module e -> dst [n * head_dim] F32
    void read_rows(int e, const int32_t * rows, size_t n, float * dst) const;

    // ai4: raw bytes of rows [n] (row_size each) of module e; false on a read error
    bool read_raw(int e, const int32_t * rows, size_t n, uint8_t * out) const;

    // readahead hint for rows that read_rows will ask for
    void prefetch_rows(int e, const int32_t * rows, size_t n) const;
};
