#include "llama-engram.h"

#include "llama-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>

// ALPHA_ENGRAM_DIRECT (default off, Linux): batches of at least 64 rows are read with O_DIRECT by ALPHA_ENGRAM_THREADS
// (default 128) threads, each reading the aligned block that holds a row; anything else uses the buffered pread path
static bool engram_direct_on() {
#if defined(__linux__)
    // default off: 16 threads of O_DIRECT reads took 4.4 s per module for an 8K prompt (f45), slower than the
    // buffered reads after one WILLNEED hint for every row
    static const bool v = [] { const char * e = getenv("ALPHA_ENGRAM_DIRECT"); return e && atoi(e) != 0; }();
    return v;
#else
    return false;
#endif
}
static double engram_io_gb() {   // this process's storage reads so far (Linux /proc/self/io), GB
#if defined(__linux__)
    FILE * f = fopen("/proc/self/io", "r");
    if (!f) {
        return 0.0;
    }
    char line[128];
    unsigned long long v = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "read_bytes: %llu", &v) == 1) {
            break;
        }
    }
    fclose(f);
    return v / 1e9;
#else
    return 0.0;
#endif
}
static int engram_threads() {
    static const int v = [] { const char * e = getenv("ALPHA_ENGRAM_THREADS"); return e ? std::max(1, atoi(e)) : 128; }();   // random 4 KB reads scale with queue depth (f71)
    return v;
}
static int engram_cvt_threads() {
    static const int v = [] { const char * e = getenv("ALPHA_ENGRAM_CVT_THREADS"); return e ? std::max(1, atoi(e)) : 4; }();
    return v;
}
static size_t engram_cache_bytes() {
    static const size_t v = [] { const char * e = getenv("ALPHA_ENGRAM_CACHE_MB"); return (size_t) (e ? atoll(e) : 512) << 20; }();
    return v;
}
static bool engram_log() {
    static const bool v = getenv("ALPHA_ENGRAM_LOG") && atoi(getenv("ALPHA_ENGRAM_LOG")) != 0;
    return v;
}

llama_engram_table::~llama_engram_table() {
    std::vector<int> closed;
    for (auto & m : modules) {
        if (m.fd < 0) {
            continue;
        }
        bool done = false;
        for (int fd : closed) {
            done = done || fd == m.fd;
        }
        if (!done) {
#if defined(_WIN32)
            _close(m.fd);
#else
            close(m.fd);
#endif
            closed.push_back(m.fd);
        }
    }
}

void llama_engram_table::open(const std::vector<std::string> & paths, const std::vector<int32_t> & layers,
                              const std::vector<int64_t> & row_base, const std::vector<int64_t> & n_rows, int64_t head_dim_) {
    head_dim = head_dim_;
    modules.assign(layers.size(), source());

    for (const auto & path : paths) {
        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ nullptr,
        };
        gguf_context * g = gguf_init_from_file(path.c_str(), params);
        if (!g) {
            throw std::runtime_error(format("engram: failed to read the GGUF header of %s", path.c_str()));
        }

        int fd = -1;
        const int64_t n_tensors = gguf_get_n_tensors(g);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const std::string name = gguf_get_tensor_name(g, i);

            int e_first = -1;
            int e_last  = -1;
            bool single = false;
            if (name == "per_layer_token_embd.weight") {
                e_first = 0;
                e_last  = (int) layers.size() - 1;
                single  = true;
            } else {
                for (size_t e = 0; e < layers.size(); ++e) {
                    if (name == format("blk.%d.engram_embd.weight", layers[e])) {
                        e_first = e_last = (int) e;
                    }
                }
            }
            if (e_first < 0) {
                continue;
            }

            const ggml_type type     = gguf_get_tensor_type(g, i);
            const size_t    size     = gguf_get_tensor_size(g, i);
            const size_t    row_size = ggml_row_size(type, head_dim);
            const auto *    traits   = ggml_get_type_traits(type);
            if (head_dim % ggml_blck_size(type) != 0 || row_size == 0 || size % row_size != 0 ||
                    (type != GGML_TYPE_F32 && traits->to_float == nullptr)) {
                gguf_free(g);
                throw std::runtime_error(format("engram: tensor %s in %s has an unsupported type or row size (%s)",
                        name.c_str(), path.c_str(), ggml_type_name(type)));
            }
            const int64_t rows_have = (int64_t) (size/row_size);
            const int64_t rows_need = single ? (row_base[e_last] + n_rows[e_last]) : n_rows[e_first];
            if (rows_have < rows_need) {
                gguf_free(g);
                throw std::runtime_error(format("engram: tensor %s in %s has %" PRId64 " rows, the hash needs %" PRId64,
                        name.c_str(), path.c_str(), rows_have, rows_need));
            }

            if (fd < 0) {
#if defined(_WIN32)
                fd = _open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
                fd = ::open(path.c_str(), O_RDONLY);
#endif
                if (fd < 0) {
                    gguf_free(g);
                    throw std::runtime_error(format("engram: cannot open %s", path.c_str()));
                }
#if defined(POSIX_FADV_RANDOM)
                {
                    static const bool fadv = [] { const char * v = getenv("ALPHA_ENGRAM_FADV_RANDOM"); return v && atoi(v) != 0; }();
                    if (fadv) {
                        posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
                    }
                }
#endif
            }

            for (int e = e_first; e <= e_last; ++e) {
                source & m = modules[e];
                if (m.fd >= 0) {
                    LLAMA_LOG_INFO("%s: table of layer %d overridden by %s\n", __func__, layers[e], path.c_str());
                }
                m.path     = path;
                m.fd       = fd;
                m.offset   = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i);
                m.type     = type;
                m.n_rows   = rows_have;
                m.row_size = row_size;
                m.row_base = single ? 0 : row_base[e];
            }

            LLAMA_LOG_INFO("%s: %s: %s [%" PRId64 " x %" PRId64 "] %s, %.2f GiB, read on demand (%zu bytes per row)\n", __func__,
                    path.c_str(), name.c_str(), head_dim, rows_have, ggml_type_name(type), size/1024.0/1024.0/1024.0, row_size);
        }

        gguf_free(g);
    }

    caches.resize(modules.size());   // ai4: one row cache per module
    for (size_t e = 0; e < modules.size(); ++e) {
        if (modules[e].fd < 0) {
            throw std::runtime_error(format(
                    "engram: no table for layer %d: give the GGUF file(s) holding per_layer_token_embd.weight or "
                    "blk.%d.engram_embd.weight with --shard / --ple (or env ALPHA_ENGRAM_SHARDS, comma separated)",
                    layers[e], layers[e]));
        }
    }
}

bool llama_engram_table::read_raw(int e, const int32_t * rows, size_t n, uint8_t * out) const {
    const source & m = modules[e];
    std::atomic<bool> bad{false};
#if defined(__linux__)
    if (engram_direct_on() && n >= 64) {
        if (m.fd_direct == -2) {
            m.fd_direct = ::open(m.path.c_str(), O_RDONLY | O_DIRECT);
        }
    }
    if (engram_direct_on() && n >= 64 && m.fd_direct >= 0) {
        const int nt = (int) std::min<size_t>((size_t) engram_threads(), (n + 15) / 16);
        auto work = [&](int t) {
            void * blk = nullptr;
            if (posix_memalign(&blk, 4096, 8192) != 0) {
                bad = true;
                return;
            }
            const size_t i0 = n * (size_t) t / (size_t) nt;
            const size_t i1 = n * (size_t) (t + 1) / (size_t) nt;
            for (size_t i = i0; i < i1 && !bad; ++i) {
                const int64_t r = (int64_t) rows[i] - m.row_base;
                if (r < 0 || r >= m.n_rows) {
                    bad = true;
                    break;
                }
                const uint64_t off = m.offset + (uint64_t) r*m.row_size;
                const uint64_t a   = off & ~(uint64_t) 4095;
                const size_t   in  = (size_t) (off - a);
                const size_t   len = in + m.row_size <= 4096 ? 4096 : 8192;
                const ssize_t  got = pread(m.fd_direct, blk, len, (off_t) a);
                if (got < (ssize_t) (in + m.row_size)) {
                    bad = true;
                    break;
                }
                memcpy(out + i * m.row_size, (const uint8_t *) blk + in, m.row_size);
            }
            free(blk);
        };
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) {
            th.emplace_back(work, t);
        }
        work(0);
        for (auto & x : th) {
            x.join();
        }
        if (!bad) {
            return true;
        }
        bad = false;   // reread below
    }
#endif
#if !defined(_WIN32)
    // buffered pread, the rows split over threads (page-cache hits are cheap; misses get queue depth)
    if (n >= 256) {
        const int nt = (int) std::min<size_t>((size_t) engram_threads(), n / 64);
        auto work = [&](int t) {
            const size_t a = n * (size_t) t / (size_t) nt, b = n * (size_t) (t + 1) / (size_t) nt;
            for (size_t i = a; i < b && !bad; ++i) {
                const int64_t r = (int64_t) rows[i] - m.row_base;
                if (r < 0 || r >= m.n_rows) {
                    bad = true;
                    break;
                }
                const uint64_t off = m.offset + (uint64_t) r*m.row_size;
                if (pread(m.fd, out + i * m.row_size, m.row_size, (off_t) off) != (ssize_t) m.row_size) {
                    bad = true;
                    break;
                }
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < nt; ++t) {
            th.emplace_back(work, t);
        }
        work(0);
        for (auto & x : th) {
            x.join();
        }
        return !bad;
    }
#endif
    for (size_t i = 0; i < n; ++i) {
        const int64_t r = (int64_t) rows[i] - m.row_base;
        if (r < 0 || r >= m.n_rows) {
            return false;
        }
        const uint64_t off = m.offset + (uint64_t) r*m.row_size;
#if defined(_WIN32)
        _lseeki64(m.fd, (int64_t) off, SEEK_SET);
        const int64_t got = _read(m.fd, out + i * m.row_size, (unsigned) m.row_size);
#else
        const int64_t got = pread(m.fd, out + i * m.row_size, m.row_size, (off_t) off);
#endif
        if (got != (int64_t) m.row_size) {
            return false;
        }
    }
    return true;
}

void llama_engram_table::prefetch_rows(int e, const int32_t * rows, size_t n) const {
    // ALPHA_ENGRAM_PREFETCH (default off): one WILLNEED per row cost ~74K syscalls per module for an 8K prompt; the
    // reader now reads the distinct missing rows from several threads instead
    static const bool hint = [] { const char * v = getenv("ALPHA_ENGRAM_PREFETCH"); return v && atoi(v) != 0; }();
    if (!hint || (engram_direct_on() && n >= 64)) {
        return;
    }
#if defined(POSIX_FADV_WILLNEED)
    const source & m = modules[e];
    for (size_t i = 0; i < n; ++i) {
        const int64_t r = (int64_t) rows[i] - m.row_base;
        if (r < 0 || r >= m.n_rows) {
            continue;
        }
        if ((size_t) e < caches.size() && caches[e].row_cache.count((uint32_t) rows[i])) {
            continue;
        }
        posix_fadvise(m.fd, (off_t) (m.offset + (uint64_t) r*m.row_size), (off_t) m.row_size, POSIX_FADV_WILLNEED);
    }
#else
    (void) e; (void) rows; (void) n;
#endif
}

void llama_engram_table::read_rows(int e, const int32_t * rows, size_t n, float * dst) const {
    const source & m = modules[e];
    const auto * traits = ggml_get_type_traits(m.type);

    const int64_t t0 = ggml_time_us();
    const double io0 = engram_log() ? engram_io_gb() : 0.0;

    // ai4: row cache (see ALPHA_ENGRAM_CACHE_MB): cached rows straight from memory, distinct misses read once, the
    // conversion to F32 in ALPHA_ENGRAM_CVT_THREADS slices
    if (engram_cache_bytes() > 0 && m.row_size > 0 && m.row_size <= 4096 && n > 0 && (size_t) e < caches.size()) {
        module_cache & C = caches[e];
        if (C.slots_max == 0) {
            C.slots_max = engram_cache_bytes() / modules.size() / m.row_size;
        }
        std::vector<int32_t> miss_rows;
        std::unordered_map<uint32_t, size_t> miss_idx;   // row id -> index in miss_rows
        for (size_t i = 0; i < n; ++i) {
            const uint32_t k = (uint32_t) rows[i];
            if (C.row_cache.count(k) || miss_idx.count(k)) {
                continue;
            }
            miss_idx.emplace(k, miss_rows.size());
            miss_rows.push_back(rows[i]);
        }
        std::vector<uint8_t> miss_bytes(miss_rows.size() * m.row_size);
        bool ok = true;
        if (!miss_rows.empty()) {
            ok = read_raw(e, miss_rows.data(), miss_rows.size(), miss_bytes.data());
        }
        if (ok) {
            for (size_t j = 0; j < miss_rows.size() && C.row_cache.size() < C.slots_max; ++j) {
                const uint32_t slot = (uint32_t) C.row_cache.size();
                if (C.row_arena.size() < (size_t) (slot + 1) * m.row_size) {
                    C.row_arena.resize(std::min(C.slots_max, std::max<size_t>(2*(size_t) slot + 1, 65536)) * m.row_size);
                }
                memcpy(C.row_arena.data() + (size_t) slot * m.row_size, miss_bytes.data() + j * m.row_size, m.row_size);
                C.row_cache.emplace((uint32_t) miss_rows[j], slot);
            }
            // resolve each row's source first (no map lookups in the parallel part)
            std::vector<const uint8_t *> src(n);
            for (size_t i = 0; i < n; ++i) {
                const uint32_t k = (uint32_t) rows[i];
                auto it = C.row_cache.find(k);
                src[i] = it != C.row_cache.end() ? C.row_arena.data() + (size_t) it->second * m.row_size
                                                 : miss_bytes.data() + miss_idx.at(k) * m.row_size;
            }
            const int nt = n >= 4096 ? engram_cvt_threads() : 1;
            auto cvt = [&](int t) {
                const size_t a = n * (size_t) t / (size_t) nt, b = n * (size_t) (t + 1) / (size_t) nt;
                for (size_t i = a; i < b; ++i) {
                    if (m.type == GGML_TYPE_F32) {
                        memcpy(dst + i*head_dim, src[i], m.row_size);
                    } else {
                        traits->to_float(src[i], dst + i*head_dim, head_dim);
                    }
                }
            };
            std::vector<std::thread> th;
            for (int t = 1; t < nt; ++t) {
                th.emplace_back(cvt, t);
            }
            cvt(0);
            for (auto & x : th) {
                x.join();
            }
            n_rows_read += miss_rows.size();
            t_read_us   += (uint64_t) (ggml_time_us() - t0);
            if (engram_log()) {
                fprintf(stderr, "[engram] module %d: %zu rows, %zu distinct misses read, %zu cached rows, %.1f ms, storage read %.3f GB\n", e, n,
                        miss_rows.size(), C.row_cache.size(), (ggml_time_us() - t0) / 1000.0, engram_io_gb() - io0);
            }
            return;
        }
    }

#if defined(__linux__)
    if (engram_direct_on() && n >= 64 && m.row_size <= 4096) {
        if (m.fd_direct == -2) {
            m.fd_direct = ::open(m.path.c_str(), O_RDONLY | O_DIRECT);
        }
        if (m.fd_direct >= 0) {
            const int nt = (int) std::min<size_t>((size_t) engram_threads(), (n + 31) / 32);
            std::atomic<bool> bad{false};
            auto work = [&](int t) {
                void * blk = nullptr;
                if (posix_memalign(&blk, 4096, 8192) != 0) {
                    bad = true;
                    return;
                }
                const size_t i0 = n * (size_t) t / (size_t) nt;
                const size_t i1 = n * (size_t) (t + 1) / (size_t) nt;
                for (size_t i = i0; i < i1 && !bad; ++i) {
                    const int64_t r = (int64_t) rows[i] - m.row_base;
                    if (r < 0 || r >= m.n_rows) {
                        bad = true;
                        break;
                    }
                    const uint64_t off = m.offset + (uint64_t) r*m.row_size;
                    const uint64_t a   = off & ~(uint64_t) 4095;
                    const size_t   in  = (size_t) (off - a);
                    const size_t   len = in + m.row_size <= 4096 ? 4096 : 8192;
                    const ssize_t  got = pread(m.fd_direct, blk, len, (off_t) a);
                    if (got < (ssize_t) (in + m.row_size)) {
                        bad = true;
                        break;
                    }
                    const uint8_t * src = (const uint8_t *) blk + in;
                    if (m.type == GGML_TYPE_F32) {
                        memcpy(dst + i*head_dim, src, m.row_size);
                    } else {
                        traits->to_float(src, dst + i*head_dim, head_dim);
                    }
                }
                free(blk);
            };
            std::vector<std::thread> th;
            for (int t = 1; t < nt; ++t) {
                th.emplace_back(work, t);
            }
            work(0);
            for (auto & x : th) {
                x.join();
            }
            if (!bad) {
                n_rows_read += n;
                t_read_us   += (uint64_t) (ggml_time_us() - t0);
                if (engram_log()) {
                    fprintf(stderr, "[engram] module %d: %zu rows in %.1f ms (O_DIRECT, %d threads)\n", e, n,
                            (ggml_time_us() - t0) / 1000.0, nt);
                }
                return;
            }
            // any failure: the buffered path below rereads every row
        }
    }
#endif

    std::vector<uint8_t> buf(m.row_size);
    for (size_t i = 0; i < n; ++i) {
        const int64_t r = (int64_t) rows[i] - m.row_base;
        if (r < 0 || r >= m.n_rows) {
            throw std::runtime_error(format("engram: row %d of module %d is outside its table", rows[i], e));
        }
        const uint64_t off = m.offset + (uint64_t) r*m.row_size;
        void * out = m.type == GGML_TYPE_F32 ? (void *) (dst + i*head_dim) : (void *) buf.data();
#if defined(_WIN32)
        _lseeki64(m.fd, (int64_t) off, SEEK_SET);
        const int64_t got = _read(m.fd, out, (unsigned) m.row_size);
#else
        const int64_t got = pread(m.fd, out, m.row_size, (off_t) off);
#endif
        if (got != (int64_t) m.row_size) {
            throw std::runtime_error(format("engram: short read in %s", m.path.c_str()));
        }
        if (m.type != GGML_TYPE_F32) {
            traits->to_float(buf.data(), dst + i*head_dim, head_dim);
        }
    }

    n_rows_read += n;
    t_read_us   += (uint64_t) (ggml_time_us() - t0);
    if (engram_log()) {
        fprintf(stderr, "[engram] module %d: %zu rows in %.1f ms (buffered)\n", e, n, (ggml_time_us() - t0) / 1000.0);
    }
}
