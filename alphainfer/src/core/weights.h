// SPDX-License-Identifier: Apache-2.0
// GGUF -> resident WeightSet. GGUF is the only artifact.
//
// Loads once, before finalize(): every tensor is packed into device buffers of at most
// `max_buffer_bytes` (with the kernel padding the device reports), then streamed from the file
// through a pinned staging buffer. Metadata is exposed by key for the family's ModelSpec.
#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

struct gguf_context;

namespace ai {

class WeightSet {
public:
    ~WeightSet();

    bool open(const std::string & path);                         // metadata only; a split GGUF's first shard opens all

    // Tensors for which `host` returns true are read into one registered host block instead of
    // device memory (their Tensor has buf == -1 and a host data pointer).
    using HostFilter = std::function<bool(const std::string &)>;
    // Each group's tensors are placed back to back (no gap) so a fused projection can alias them.
    using Groups = std::vector<std::vector<std::string>>;
    // Load-time conversion: for a device tensor, `convert` returns the type to hold it in (or -1 = as stored).
    // Supported: BF16 -> Q8_0 (rows of whole blocks), via ggml_bf16_to_fp32_row + ggml_quantize_chunk -- the same two
    // calls alphallama's loader makes for its ALPHA_HC_Q8 hc projections, so the bytes are identical.
    using ConvertFilter = std::function<int(const std::string &, const Tensor &)>;
    // host_mmap: host-kept tensors point into a read-only mapping of their shard instead of a copied, registered
    // block, for models whose host side (e.g. 43 GB of experts + a 29 GB PLE table) would not fit in RAM as a copy.
    // The page cache holds them; device copies from them are pageable (the mapped hot tier is the fast path).
    bool load(Device & dev, size_t max_buffer_bytes, bool verbose, const HostFilter & host = nullptr,
              const Groups & groups = {}, const ConvertFilter & convert = nullptr, bool host_mmap = false);

    // host_mmap only: copy n bytes at a pointer into the shard mappings to dst with `threads` parallel reads of the file
    // (O_DIRECT through an aligned bounce buffer when the filesystem allows, so the file is not cached twice) -- the
    // RAM tier's fill (qwen4exp_run.cpp). Returns false on a read error.
    bool      copy_from_mapping(const void * src, void * dst, size_t n, int threads, bool * direct = nullptr) const;
    // host_mmap only: many copies at once, straight from the file into place: each job's dst must sit at the same offset
    // modulo 4096 as its source's file offset, with dst - (that offset) .. round_up(dst + n, 4096) writable -- then
    // every chunk is one O_DIRECT read into the destination (no bounce, no page-cache copy). Falls back to a buffered
    // pread per chunk where O_DIRECT is refused. One pool of `threads` over all jobs' 64 MiB chunks.
    struct ReadJob {
        const void * src;
        char *       dst;
        size_t       n;
    };
    bool      read_into(const std::vector<ReadJob> & jobs, int threads, bool * direct = nullptr) const;
    // the offset of a pointer into the shard mappings within its file, modulo 4096 (SIZE_MAX when not mapped)
    size_t    file_page_offset(const void * p) const;
    // host_mmap only: read n bytes at a pointer into the shard mappings with pread() instead of dereferencing the
    // mapping. The page-cache page is then never mapped into this process, so kernel memory compaction migrating it
    // sends no TLB-shootdown IPIs to the spinning pool cpus (which can slow every pool phase ~2x). Thread-safe.
    bool      host_pread(const void * p, void * dst, size_t n) const;
    // host_mmap only: start reading n bytes at a pointer into the shard mappings into the page cache (POSIX_FADV_WILLNEED:
    // asynchronous readahead, nothing mapped); false when p is not in a mapping
    bool      host_prefetch(const void * p, size_t n) const;
    // host_mmap only: drop this process's page-table entries over the whole shard mappings (MADV_DONTNEED: the pages stay
    // in the page cache; a later access refaults them) -- after the RAM tier took the hot tensors, so compaction
    // migrating the rest cannot IPI the pool. Returns the bytes covered.
    size_t    drop_mapping_ptes();

    // One tensor spanning adjacent same-type, same-width tensors along ne[1] (rows), or an
    // invalid tensor when they are not adjacent.
    Tensor span(const std::vector<std::string> & names) const;

    bool     is_host(const std::string & name) const;
    uint64_t host_bytes() const { return host_bytes_; }

    std::string arch() const;
    bool        has_key(const std::string & key) const;
    int64_t     get_int(const std::string & key) const;           // throws if missing
    int64_t     get_int(const std::string & key, int64_t def) const;
    double      get_float(const std::string & key) const;
    std::vector<int64_t> get_int_array(const std::string & key) const;
    std::vector<std::string> get_str_array(const std::string & key) const;   // empty when missing

    bool          has(const std::string & name) const { return tensors_.count(name) != 0; }
    const Tensor & get(const std::string & name) const;
    const Tensor * find(const std::string & name) const;
    std::vector<std::string> names() const;
    // every tensor's name, type, shape and byte size after open() -- metadata only, no load() needed
    // (find() and names() see loaded tensors only)
    struct TensorMeta {
        std::string name;
        Tensor      t;
        size_t      nbytes = 0;
    };
    std::vector<TensorMeta> meta() const;

    size_t   n_tensors() const { return tensors_.size(); }
    uint64_t bytes() const { return bytes_; }
    double   load_seconds() const { return load_s_; }

private:
    struct Entry {
        std::string name;
        Tensor      t;
        uint64_t    file_offset;
        size_t      nbytes;
        int         file = 0;   // shard index into paths_ (split GGUF)
        int         src_type   = -1;   // ConvertFilter: the stored type when converted at load (t.type is the held one)
        size_t      src_nbytes = 0;
    };

    std::string                   path_;
    std::vector<std::string>      paths_;          // every shard of a split GGUF (paths_[0] == path_)
    std::vector<gguf_context *>   extra_gguf_;     // shards 2..N (metadata of shard 1 is gguf_)
    gguf_context *                gguf_ = nullptr;
    std::vector<Entry>            entries_;
    std::map<std::string, Tensor> tensors_;
    uint64_t                      bytes_  = 0;
    uint64_t                      host_bytes_ = 0;
    char *                        host_block_ = nullptr;
    std::vector<std::pair<void *, size_t>> maps_;   // host_mmap: one read-only mapping per shard
    std::vector<int>              fds_;            // host_pread: one O_RDONLY fd per shard, opened with the mappings
    bool                          host_registered_ = false;
    double                        load_s_ = 0;
};

// True only when the GGUF header opens and declares no experts (no <arch>.expert_count). Reads metadata
// only. An unreadable file is "not known dense" -> false, so callers fall back to their MoE/exact default.
bool gguf_is_dense(const std::string & path);

}  // namespace ai
