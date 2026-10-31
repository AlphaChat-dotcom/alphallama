// SPDX-License-Identifier: Apache-2.0
#include "core/weights.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <stdexcept>
#include <thread>

namespace ai {

WeightSet::~WeightSet() {
    if (host_block_) {
        if (host_registered_) {
            ggml_backend_cuda_unregister_host_buffer(host_block_);
        }
        free(host_block_);
    }
    for (auto & m : maps_) {
        munmap(m.first, m.second);
    }
    for (int fd : fds_) ::close(fd);
    for (int fd : dfds_) if (fd >= 0) ::close(fd);
    if (gguf_) {
        gguf_free(gguf_);
    }
    for (gguf_context * g : extra_gguf_) {
        gguf_free(g);
    }
}

bool WeightSet::open(const std::string & path) {
    path_ = path;
    ggml_context * meta = nullptr;
    gguf_init_params p  = { /*no_alloc=*/ true, &meta };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if (!gguf_) {
        fprintf(stderr, "[ai] cannot open gguf %s\n", path.c_str());
        return false;
    }
    const size_t data0 = gguf_get_data_offset(gguf_);
    const int64_t n    = gguf_get_n_tensors(gguf_);
    entries_.reserve((size_t) n);
    for (int64_t i = 0; i < n; ++i) {
        const char *  name = gguf_get_tensor_name(gguf_, i);
        ggml_tensor * mt   = ggml_get_tensor(meta, name);
        Entry e;
        e.name   = name;
        e.t.type = mt->type;
        for (int d = 0; d < 4; ++d) {
            e.t.ne[d] = mt->ne[d];
        }
        set_contiguous_strides(e.t);
        e.file_offset = data0 + gguf_get_tensor_offset(gguf_, i);
        e.nbytes      = ggml_nbytes(mt);
        entries_.push_back(e);
    }
    ggml_free(meta);
    paths_ = { path };
    // split GGUF (llama.cpp gguf-split: "split.count" in every shard, files "<base>-%05d-of-%05d.gguf"):
    // the first shard holds the model metadata; shards 2..N add tensors with their own data offsets.
    const int n_split = (int) get_int("split.count", 1);
    if (n_split > 1) {
        char tail[64];
        snprintf(tail, sizeof tail, "-%05d-of-%05d.gguf", 1, n_split);
        const size_t tl = strlen(tail);
        if (path.size() <= tl || path.compare(path.size() - tl, tl, tail) != 0) {
            fprintf(stderr, "[ai] %s declares split.count %d but is not named *%s\n", path.c_str(), n_split, tail);
            return false;
        }
        const std::string base = path.substr(0, path.size() - tl);
        for (int k = 2; k <= n_split; ++k) {
            char sfx[64];
            snprintf(sfx, sizeof sfx, "-%05d-of-%05d.gguf", k, n_split);
            const std::string sp = base + sfx;
            ggml_context * m2 = nullptr;
            gguf_init_params p2 = { /*no_alloc=*/ true, &m2 };
            gguf_context * g2 = gguf_init_from_file(sp.c_str(), p2);
            if (!g2) {
                fprintf(stderr, "[ai] cannot open gguf shard %s\n", sp.c_str());
                return false;
            }
            extra_gguf_.push_back(g2);
            paths_.push_back(sp);
            const size_t d2 = gguf_get_data_offset(g2);
            for (int64_t i = 0; i < gguf_get_n_tensors(g2); ++i) {
                const char *  name = gguf_get_tensor_name(g2, i);
                ggml_tensor * mt   = ggml_get_tensor(m2, name);
                Entry e;
                e.name   = name;
                e.t.type = mt->type;
                for (int d = 0; d < 4; ++d) {
                    e.t.ne[d] = mt->ne[d];
                }
                set_contiguous_strides(e.t);
                e.file_offset = d2 + gguf_get_tensor_offset(g2, i);
                e.nbytes      = ggml_nbytes(mt);
                e.file        = k - 1;
                entries_.push_back(e);
            }
            ggml_free(m2);
        }
    }
    return true;
}

bool WeightSet::load(Device & dev, size_t max_buffer_bytes, bool verbose, const HostFilter & host,
                     const Groups & groups, const ConvertFilter & convert, bool host_mmap) {
    const auto t0    = std::chrono::steady_clock::now();
    const size_t aln = dev.alignment();

    // ConvertFilter: decide conversions first, since placement must see the held size
    if (convert) {
        size_t n_conv = 0, saved = 0;
        for (auto & e : entries_) {
            if (host && host(e.name)) continue;   // device tensors only
            const int to = convert(e.name, e.t);
            if (to < 0 || to == e.t.type) continue;
            // BF16 -> Q8_0 or F16 (lossless for weights inside F16's range: it puts small matmuls on ggml's F16
            // matrix-vector kernels instead of cutlass BF16), or (AI_REQUANT) any type ggml can dequantise -> a type that needs no importance matrix
            // (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K), whole blocks only
            const bool to_ok   = to == Q8_0 || to == F16 || to == Q4_0 || to == Q4_K || to == Q5_K || to == Q6_K;
            const bool from_ok = ggml_get_type_traits((ggml_type) e.t.type)->to_float != nullptr;
            if (!to_ok || !from_ok || e.t.ne[0] % ggml_blck_size((ggml_type) to) != 0 ||
                ggml_quantize_requires_imatrix((ggml_type) to) || !e.t.contiguous()) {
                throw std::runtime_error("WeightSet: unsupported load-time conversion for " + e.name);
            }
            e.src_type   = e.t.type;
            e.src_nbytes = e.nbytes;
            e.t.type     = to;
            set_contiguous_strides(e.t);
            e.nbytes = e.t.nbytes();
            saved += e.src_nbytes - e.nbytes;
            n_conv++;
        }
        if (verbose && n_conv) {
            fprintf(stderr, "[ai] load-time conversion: %zu tensors requantised (%.2f GiB less)\n", n_conv,
                    saved / 1073741824.0);
        }
    }

    // host-kept tensors: one page-aligned block, registered with the device for async copies
    std::vector<size_t> host_off(entries_.size(), SIZE_MAX);
    size_t host_total = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (host && host(entries_[i].name)) {
            host_off[i] = host_total;
            host_total += (entries_[i].nbytes + 4095) / 4096 * 4096;
        }
    }
    if (host_total && host_mmap) {
        for (const std::string & p : paths_) {
            const int fd = ::open(p.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("cannot open for mmap: " + p);
            }
            struct stat st;
            if (fstat(fd, &st) != 0) {
                ::close(fd);
                throw std::runtime_error("cannot stat: " + p);
            }
            void * m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
            fds_.push_back(fd);   // kept for host_pread
#if defined(POSIX_FADV_RANDOM)
            posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);   // host_pread reads scattered rows: no readahead
#endif
            if (m == MAP_FAILED) {
                throw std::runtime_error("mmap failed: " + p);
            }
            maps_.push_back({ m, (size_t) st.st_size });
        }
        host_bytes_ = host_total;
    } else if (host_total) {
        if (posix_memalign((void **) &host_block_, 4096, host_total) != 0) {
            throw std::runtime_error("host block allocation failed");
        }
        host_bytes_ = host_total;
    }

    // placement order: file order, except that a group is placed whole where its first member
    // appears, members back to back (only the last carries the kernel padding)
    std::map<std::string, size_t> index;
    for (size_t i = 0; i < entries_.size(); ++i) {
        index[entries_[i].name] = i;
    }
    std::map<size_t, size_t> group_of;   // entry -> group
    for (size_t gi = 0; gi < groups.size(); ++gi) {
        bool ok = groups[gi].size() > 1;
        for (const auto & n : groups[gi]) {
            auto it = index.find(n);
            ok = ok && it != index.end() && host_off[it->second] == SIZE_MAX;
        }
        if (ok) {
            for (const auto & n : groups[gi]) {
                group_of[index[n]] = gi;
            }
        }
    }
    std::vector<std::vector<size_t>> runs;   // each run is placed contiguously
    std::vector<char> placed(entries_.size(), 0);
    for (size_t i = 0; i < entries_.size(); ++i) {
        if (placed[i] || host_off[i] != SIZE_MAX) {
            continue;
        }
        std::vector<size_t> run;
        auto g = group_of.find(i);
        if (g != group_of.end()) {
            for (const auto & n : groups[g->second]) {
                run.push_back(index[n]);
            }
        } else {
            run.push_back(i);
        }
        for (size_t e : run) {
            placed[e] = 1;
        }
        runs.push_back(run);
    }

    // pack runs into buffers of at most max_buffer_bytes
    struct Placement { int buf; size_t off; };
    std::vector<Placement> place(entries_.size(), Placement{-1, 0});
    std::vector<size_t>    buf_sizes;
    size_t cur = 0;
    for (const auto & run : runs) {
        size_t total = 0;
        for (size_t r = 0; r < run.size(); ++r) {
            total += r + 1 < run.size() ? entries_[run[r]].nbytes : dev.alloc_size(entries_[run[r]].t);
        }
        total = (total + aln - 1) / aln * aln;
        if (buf_sizes.empty() || cur + total > max_buffer_bytes) {
            buf_sizes.push_back(0);
            cur = 0;
        }
        size_t off = cur;
        for (size_t e : run) {
            place[e] = { (int) buf_sizes.size() - 1, off };
            off += entries_[e].nbytes;
        }
        cur += total;
        buf_sizes.back() = cur;
    }
    std::vector<int> buf_ids;
    for (size_t b = 0; b < buf_sizes.size(); ++b) {
        buf_ids.push_back(dev.alloc_buffer(buf_sizes[b], "weights." + std::to_string(b)));
    }

    // pinned staging
    const size_t stage_bytes = 128u << 20;
    ggml_backend_buffer_t stage = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), stage_bytes);
    if (!stage) {
        throw std::runtime_error("pinned staging allocation failed");
    }
    char * staging = (char *) ggml_backend_buffer_get_base(stage);

    std::vector<FILE *> files;
    for (const std::string & p : paths_) {
        FILE * fp = fopen(p.c_str(), "rb");
        if (!fp) {
            throw std::runtime_error("cannot reopen " + p);
        }
        files.push_back(fp);
    }

    for (size_t i = 0; i < entries_.size(); ++i) {
        Entry & e = entries_[i];
        FILE *  f = files[(size_t) e.file];
        if (host_off[i] != SIZE_MAX && host_mmap) {
            e.t.buf  = -1;
            e.t.data = (char *) maps_[(size_t) e.file].first + e.file_offset;
            tensors_[e.name] = e.t;
            continue;
        }
        if (host_off[i] != SIZE_MAX) {
            e.t.buf  = -1;
            e.t.data = host_block_ + host_off[i];
            if (fseeko(f, (off_t) e.file_offset, SEEK_SET) != 0 ||
                fread(e.t.data, 1, e.nbytes, f) != e.nbytes) {
                throw std::runtime_error("host read failed for " + e.name);
            }
            tensors_[e.name] = e.t;
            continue;
        }
        e.t.buf     = buf_ids[place[i].buf];
        void * addr = (char *) dev.buffer_base(e.t.buf) + place[i].off;

        // bind through the backend so the kernel padding is initialised
        ggml_tensor g{};
        g.type = (ggml_type) e.t.type;
        for (int d = 0; d < 4; ++d) {
            g.ne[d] = e.t.ne[d];
            g.nb[d] = e.t.nb[d];
        }
        if (ggml_backend_tensor_alloc(dev.gbuf(e.t.buf), &g, addr) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("tensor_alloc failed for " + e.name);
        }
        e.t.data = addr;

        if (fseeko(f, (off_t) e.file_offset, SEEK_SET) != 0) {
            throw std::runtime_error("seek failed for " + e.name);
        }
        if (e.src_type >= 0) {   // ConvertFilter: stored as src_type in the file, held as t.type
            const int64_t ne0 = e.t.ne[0], nrows = e.t.ne[1] * e.t.ne[2] * e.t.ne[3];
            std::vector<uint8_t> src(e.src_nbytes);
            if (fread(src.data(), 1, e.src_nbytes, f) != e.src_nbytes) {
                throw std::runtime_error("short read for " + e.name);
            }
            std::vector<float>   f32((size_t) (ne0 * nrows));
            std::vector<uint8_t> q(e.nbytes);
            const auto * tt = ggml_get_type_traits((ggml_type) e.src_type);
            const size_t rs = ggml_row_size((ggml_type) e.src_type, ne0);
            // dequantise + requantise in row blocks over the hardware threads (a 27B's Q5_K FFN is ~3 GB)
            const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(32, (int64_t) std::thread::hardware_concurrency()));
            const int64_t per = (nrows + nt - 1) / nt;
            std::vector<std::thread> th;
            for (int64_t w = 0; w < nt; ++w) {
                const int64_t r0 = w * per, r1 = std::min(nrows, r0 + per);
                if (r0 >= r1) break;
                th.emplace_back([&, r0, r1]() {
                    for (int64_t r = r0; r < r1; ++r) tt->to_float(src.data() + (size_t) r * rs, f32.data() + r * ne0, ne0);
                    ggml_quantize_chunk((ggml_type) e.t.type, f32.data(), q.data(), r0 * ne0, r1 - r0, ne0, nullptr);
                });
            }
            for (auto & t : th) t.join();
            ggml_backend_tensor_set(&g, q.data(), 0, e.nbytes);
            tensors_[e.name] = e.t;
            bytes_ += e.nbytes;
            continue;
        }
        size_t done = 0;
        while (done < e.nbytes) {
            const size_t n = std::min(stage_bytes, e.nbytes - done);
            if (fread(staging, 1, n, f) != n) {
                throw std::runtime_error("short read for " + e.name);
            }
            ggml_backend_tensor_set(&g, staging, done, n);
            done += n;
        }
        tensors_[e.name] = e.t;
        bytes_ += e.nbytes;
        if (verbose && (i % 100 == 0 || i + 1 == entries_.size())) {
            fprintf(stderr, "[ai] load %zu/%zu tensors, %.2f GiB\n", i + 1, entries_.size(), bytes_ / 1073741824.0);
        }
    }
    for (FILE * fp : files) {
        fclose(fp);
    }
    ggml_backend_buffer_free(stage);
    if (host_block_) {
        host_registered_ = ggml_backend_cuda_register_host_buffer(host_block_, host_bytes_);
        if (verbose) {
            fprintf(stderr, "[ai] host-kept tensors: %.2f GiB (%s)\n", host_bytes_ / 1073741824.0,
                    host_registered_ ? "registered for async copies" : "NOT registered: copies are pageable");
        }
    }
    load_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

std::string WeightSet::arch() const {
    int64_t id = gguf_find_key(gguf_, "general.architecture");
    if (id < 0) {
        return "";
    }
    return gguf_get_val_str(gguf_, id);
}

bool WeightSet::has_key(const std::string & key) const {
    return gguf_find_key(gguf_, key.c_str()) >= 0;
}

int64_t WeightSet::get_int(const std::string & key, int64_t def) const {
    int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(gguf_, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(gguf_, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(gguf_, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(gguf_, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf_, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf_, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(gguf_, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(gguf_, id);
        case GGUF_TYPE_BOOL:   return gguf_get_val_bool(gguf_, id);
        default: throw std::runtime_error("gguf key is not an integer: " + key);
    }
}

int64_t WeightSet::get_int(const std::string & key) const {
    if (!has_key(key)) {
        throw std::runtime_error("gguf key missing: " + key);
    }
    return get_int(key, 0);
}

double WeightSet::get_float(const std::string & key) const {
    int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) {
        throw std::runtime_error("gguf key missing: " + key);
    }
    switch (gguf_get_kv_type(gguf_, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(gguf_, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(gguf_, id);
        default:                return (double) get_int(key);
    }
}

std::vector<int64_t> WeightSet::get_int_array(const std::string & key) const {
    int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0) {
        throw std::runtime_error("gguf key missing: " + key);
    }
    if (gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY) {
        return { get_int(key) };
    }
    const size_t n  = gguf_get_arr_n(gguf_, id);
    const auto   at = gguf_get_arr_type(gguf_, id);
    const void * p  = gguf_get_arr_data(gguf_, id);
    std::vector<int64_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        switch (at) {
            case GGUF_TYPE_INT32:  out[i] = ((const int32_t *) p)[i]; break;
            case GGUF_TYPE_UINT32: out[i] = ((const uint32_t *) p)[i]; break;
            case GGUF_TYPE_INT64:  out[i] = ((const int64_t *) p)[i]; break;
            case GGUF_TYPE_UINT64: out[i] = (int64_t) ((const uint64_t *) p)[i]; break;   // bit-cast (qwen4exp PLE)
            default: throw std::runtime_error("unsupported array type for " + key);
        }
    }
    return out;
}

std::vector<std::string> WeightSet::get_str_array(const std::string & key) const {
    const int64_t id = gguf_find_key(gguf_, key.c_str());
    if (id < 0 || gguf_get_kv_type(gguf_, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gguf_, id) != GGUF_TYPE_STRING) {
        return {};
    }
    const size_t n = gguf_get_arr_n(gguf_, id);
    std::vector<std::string> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = gguf_get_arr_str(gguf_, id, i);
    return out;
}

const Tensor & WeightSet::get(const std::string & name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return it->second;
}

Tensor WeightSet::span(const std::vector<std::string> & names) const {
    const Tensor * first = find(names.at(0));
    if (!first || first->buf < 0) {
        return Tensor{};
    }
    int64_t rows = 0;
    const char * expect = (const char *) first->data;
    for (const auto & n : names) {
        const Tensor * t = find(n);
        if (!t || t->buf != first->buf || t->type != first->type || t->ne[0] != first->ne[0] ||
            t->ne[2] != 1 || t->ne[3] != 1 || (const char *) t->data != expect) {
            return Tensor{};
        }
        expect += t->nbytes();
        rows   += t->ne[1];
    }
    Tensor out = *first;
    out.ne[1] = rows;
    set_contiguous_strides(out);
    return out;
}

bool WeightSet::is_host(const std::string & name) const {
    const Tensor * t = find(name);
    return t && t->buf < 0;
}

bool WeightSet::copy_from_mapping(const void * src, void * dst, size_t n, int threads, bool * direct) const {
    constexpr size_t pg = 4096, kChunk = 64ull << 20;
    int    map = -1;
    size_t off = 0;
    for (size_t i = 0; i < maps_.size() && i < paths_.size(); ++i) {
        const uintptr_t mb = (uintptr_t) maps_[i].first;
        if ((uintptr_t) src >= mb && (uintptr_t) src + n <= mb + maps_[i].second) { map = (int) i; off = (uintptr_t) src - mb; break; }
    }
    if (map < 0) return false;
    std::atomic<size_t> next{ 0 };
    std::atomic<int>    failed{ 0 }, buffered{ 0 };
    const size_t        n_chunks = (n + kChunk - 1) / kChunk;
    auto work = [&]() {
        int    fd_d = ::open(paths_[(size_t) map].c_str(), O_RDONLY | O_DIRECT);
        int    fd_b = -1;
        char * bounce = nullptr;
        if (fd_d >= 0 && posix_memalign((void **) &bounce, pg, kChunk + 2 * pg) != 0) bounce = nullptr;
        for (size_t c; (c = next.fetch_add(1)) < n_chunks;) {
            const size_t o0 = c * kChunk, len = std::min(kChunk, n - o0);
            const size_t fo = off + o0;                       // file offset of this chunk
            bool ok = false;
            if (fd_d >= 0 && bounce) {                        // aligned superset into the bounce buffer, then copy
                const size_t a0 = fo & ~(pg - 1), a1 = (fo + len + pg - 1) & ~(pg - 1);
                size_t got = 0;
                while (got < a1 - a0) {
                    const ssize_t r = pread(fd_d, bounce + got, a1 - a0 - got, (off_t) (a0 + got));
                    if (r <= 0) break;
                    got += (size_t) r;
                }
                if (got >= fo - a0 + len) { memcpy((char *) dst + o0, bounce + (fo - a0), len); ok = true; }
            }
            if (!ok) {                                        // no O_DIRECT (or a short read at EOF): plain pread
                if (fd_b < 0) fd_b = ::open(paths_[(size_t) map].c_str(), O_RDONLY);
                size_t got = 0;
                while (fd_b >= 0 && got < len) {
                    const ssize_t r = pread(fd_b, (char *) dst + o0 + got, len - got, (off_t) (fo + got));
                    if (r <= 0) break;
                    got += (size_t) r;
                }
                if (got < len) failed++;
                buffered++;
            }
        }
        free(bounce);
        if (fd_d >= 0) ::close(fd_d);
        if (fd_b >= 0) ::close(fd_b);
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < std::max(1, threads); ++i) pool.emplace_back(work);
    for (auto & t : pool) t.join();
    if (direct) *direct = buffered.load() == 0;
    return failed.load() == 0;
}

size_t WeightSet::file_page_offset(const void * p) const {
    for (const auto & m : maps_) {
        const uintptr_t mb = (uintptr_t) m.first;
        if ((uintptr_t) p >= mb && (uintptr_t) p < mb + m.second) return ((uintptr_t) p - mb) & 4095;
    }
    return SIZE_MAX;
}

bool WeightSet::read_into(const std::vector<ReadJob> & jobs, int threads, bool * direct) const {
    constexpr size_t pg = 4096, kChunk = 64ull << 20;
    struct Chunk { int map; uint64_t fo; size_t len; char * dst; };   // file range [fo, fo + len) -> dst
    std::vector<Chunk> chunks;
    for (const ReadJob & j : jobs) {
        int map = -1;
        for (size_t i = 0; i < maps_.size() && i < paths_.size(); ++i) {
            const uintptr_t mb = (uintptr_t) maps_[i].first;
            if ((uintptr_t) j.src >= mb && (uintptr_t) j.src + j.n <= mb + maps_[i].second) { map = (int) i; break; }
        }
        if (map < 0) return false;
        const uint64_t fo = (uintptr_t) j.src - (uintptr_t) maps_[(size_t) map].first;
        if ((((uintptr_t) j.dst) & (pg - 1)) != (fo & (pg - 1))) return false;   // caller broke the alignment contract
        // split at 64 MiB file-offset boundaries; every chunk starts page-aligned except the job's first
        for (uint64_t a = fo; a < fo + j.n;) {
            const uint64_t b = std::min<uint64_t>(fo + j.n, (a / kChunk + 1) * kChunk);
            chunks.push_back({ map, a, (size_t) (b - a), j.dst + (a - fo) });
            a = b;
        }
    }
    std::atomic<size_t> next{ 0 };
    std::atomic<int>    failed{ 0 }, buffered{ 0 };
    auto work = [&]() {
        std::vector<int> fd_d(paths_.size(), -2), fd_b(paths_.size(), -2);
        for (size_t c; (c = next.fetch_add(1)) < chunks.size();) {
            const Chunk & k = chunks[c];
            int & fd = fd_d[(size_t) k.map];
            if (fd == -2) fd = ::open(paths_[(size_t) k.map].c_str(), O_RDONLY | O_DIRECT);
            // aligned superset: [a0, a1) of the file into [dst - (fo - a0), ...) -- the slack lies in the caller's padding
            const uint64_t a0 = k.fo & ~(uint64_t) (pg - 1), a1 = (k.fo + k.len + pg - 1) & ~(uint64_t) (pg - 1);
            char * const   t0 = k.dst - (k.fo - a0);
            size_t got = 0;
            bool   ok  = false;
            if (fd >= 0) {
                while (got < a1 - a0) {
                    const ssize_t r = pread(fd, t0 + got, a1 - a0 - got, (off_t) (a0 + got));
                    if (r <= 0) break;
                    got += (size_t) r;
                }
                ok = got >= (k.fo - a0) + k.len;   // a short read at EOF still covers the data
            }
            if (!ok) {
                int & fb = fd_b[(size_t) k.map];
                if (fb == -2) fb = ::open(paths_[(size_t) k.map].c_str(), O_RDONLY);
                size_t g2 = 0;
                while (fb >= 0 && g2 < k.len) {
                    const ssize_t r = pread(fb, k.dst + g2, k.len - g2, (off_t) (k.fo + g2));
                    if (r <= 0) break;
                    g2 += (size_t) r;
                }
                if (g2 < k.len) failed++;
                buffered++;
            }
        }
        for (int f : fd_d) if (f >= 0) ::close(f);
        for (int f : fd_b) if (f >= 0) ::close(f);
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < std::max(1, threads); ++i) pool.emplace_back(work);
    for (auto & t : pool) t.join();
    if (direct) *direct = buffered.load() == 0;
    return failed.load() == 0;
}

bool WeightSet::host_pread(const void * p, void * dst, size_t n) const {
    const uintptr_t a = (uintptr_t) p;
    for (size_t i = 0; i < maps_.size() && i < fds_.size(); ++i) {
        const uintptr_t mb = (uintptr_t) maps_[i].first;
        if (a < mb || a + n > mb + maps_[i].second) continue;
        size_t done = 0;
        while (done < n) {
            const ssize_t r = pread(fds_[i], (char *) dst + done, n - done, (off_t) (a - mb + done));
            if (r <= 0) return false;
            done += (size_t) r;
        }
        return true;
    }
    return false;
}

bool WeightSet::host_pread_direct(const void * p, void * dst, size_t n) const {
    const uintptr_t a = (uintptr_t) p;
    for (size_t i = 0; i < maps_.size() && i < paths_.size(); ++i) {
        const uintptr_t mb = (uintptr_t) maps_[i].first;
        if (a < mb || a + n > mb + maps_[i].second) continue;
        int fd;
        {
            std::lock_guard<std::mutex> lk(dfd_mu_);
            if (dfds_.empty()) dfds_.assign(paths_.size(), -2);
            if (dfds_[i] == -2) dfds_[i] = ::open(paths_[i].c_str(), O_RDONLY | O_DIRECT);
            fd = dfds_[i];
        }
        if (fd < 0) return false;
        const uint64_t off = a - mb, b0 = off & ~(uint64_t) 4095, b1 = (off + n + 4095) & ~(uint64_t) 4095;
        const size_t   len = (size_t) (b1 - b0);
        thread_local void * blk = nullptr;
        thread_local size_t cap = 0;
        if (cap < len) {
            free(blk);
            blk = nullptr;
            cap = 0;
            if (posix_memalign(&blk, 4096, len) != 0) return false;
            cap = len;
        }
        size_t done = 0;
        while (done < len) {
            const ssize_t r = pread(fd, (char *) blk + done, len - done, (off_t) (b0 + done));
            if (r <= 0) break;   // EOF (the last block of the file) or an error
            done += (size_t) r;
        }
        if (done < (size_t) (off - b0) + n) return false;
        memcpy(dst, (const char *) blk + (off - b0), n);
        return true;
    }
    return false;
}

int WeightSet::host_pread_nowait(const void * p, void * dst, size_t n) const {
#if defined(RWF_NOWAIT)
    const uintptr_t a = (uintptr_t) p;
    for (size_t i = 0; i < maps_.size() && i < fds_.size(); ++i) {
        const uintptr_t mb = (uintptr_t) maps_[i].first;
        if (a < mb || a + n > mb + maps_[i].second) continue;
        struct iovec iov = { dst, n };
        const ssize_t r = preadv2(fds_[i], &iov, 1, (off_t) (a - mb), RWF_NOWAIT);
        if (r == (ssize_t) n) return 1;
        if (r >= 0 || errno == EAGAIN) return 0;   // short: part of the range is not cached
        return -1;
    }
#else
    (void) p; (void) dst; (void) n;
#endif
    return -1;
}

bool WeightSet::host_prefetch(const void * p, size_t n) const {
#if defined(POSIX_FADV_WILLNEED)
    const uintptr_t a = (uintptr_t) p;
    for (size_t i = 0; i < maps_.size() && i < fds_.size(); ++i) {
        const uintptr_t mb = (uintptr_t) maps_[i].first;
        if (a < mb || a + n > mb + maps_[i].second) continue;
        return posix_fadvise(fds_[i], (off_t) (a - mb), (off_t) n, POSIX_FADV_WILLNEED) == 0;
    }
#else
    (void) p; (void) n;
#endif
    return false;
}

size_t WeightSet::drop_mapping_ptes() {
    size_t covered = 0;
    for (const auto & m : maps_) {
        madvise(m.first, m.second, MADV_DONTNEED);
        covered += m.second;
    }
#if defined(POSIX_FADV_DONTNEED)
    // AI_Q4X_DROP_CACHE (default 1): also evict the model files' clean page cache. The pinned RAM tier plus the cached
    // GGUF pages can exceed system memory, and the resulting reclaim and compaction during decode cause TLB shootdown
    // storms on the spinning pool CPUs. Nothing here reads the cache again except host_pread's scattered PLE rows,
    // which re-cache only the pages they touch.
    static const bool drop = !getenv("AI_Q4X_DROP_CACHE") || atoi(getenv("AI_Q4X_DROP_CACHE")) != 0;
    if (drop) for (int fd : fds_) if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
    return covered;
}

std::vector<std::string> WeightSet::names() const {
    std::vector<std::string> out;
    for (const auto & kv : tensors_) {
        out.push_back(kv.first);
    }
    return out;
}

std::vector<WeightSet::TensorMeta> WeightSet::meta() const {
    std::vector<TensorMeta> out;
    out.reserve(entries_.size());
    for (const auto & e : entries_) {
        out.push_back({ e.name, e.t, e.nbytes });
    }
    return out;
}

const Tensor * WeightSet::find(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool gguf_is_dense(const std::string & path) {
    gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context *   g = gguf_init_from_file(path.c_str(), p);
    if (g == nullptr) {
        return false;   // unknown is not dense
    }
    bool moe = false;
    for (int64_t i = 0; i < gguf_get_n_kv(g) && !moe; ++i) {
        const char * k = gguf_get_key(g, i);
        const size_t n = strlen(k);
        moe = n > 13 && strcmp(k + n - 13, ".expert_count") == 0;
    }
    gguf_free(g);
    return !moe;
}

}  // namespace ai
