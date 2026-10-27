// Device-resolved experts for offloaded-MoE decode. Enabled by ALPHA_DRE=1 (default off).
//
// The fused-infer expert cache resolves expert placement on the host: for every MoE layer of every decode
// token the scheduler synchronises, reads the router ids, issues the miss DMAs and only then launches the
// expert GEMVs, leaving the GPU idle in those per-layer round trips. Here placement lives in per-tensor
// device tables instead:
//
//   table[e] = byte offset from the tensor's host base to expert e's bytes in a VRAM pool slot, or, when
//              the expert is not resident, either its own pinned host copy (zero-copy read over PCIe) or
//              PCACHE_SLOT_SKIP when the CPU computes missed experts (ALPHA_DRE_CPU=1).
//
// The MMVQ kernel reads table[ids[...]] itself (mmvq.cu, tag bit 0 = byte offsets), so a decode graph has
// no expert inputs to stage, no host sync, and runs as one CUDA graph. Residency is per expert unit (the
// gate, up and down rows of one expert in one layer, kept together in one pool slot) and changes only
// between tokens (graph_begin): the previous token's ids are read back from pinned memory, misses are
// admitted into LRU-evicted slots, and a slot is published to the tables only after its copy has landed
// (and unpublished before it is overwritten), so no running kernel sees a slot change under it.
//
// CPU miss path (ALPHA_DRE_CPU=1): before a layer's gate/up GEMV a one-block kernel copies the layer input
// and ids to mapped host memory and rings a per-layer doorbell; a host worker computes the layer's
// non-resident experts (gate, up, SwiGLU, down) from RAM with the CPU type traits while the GPU computes
// the resident ones, and after the down GEMV a one-block kernel waits for the worker's answer and writes
// the missed experts' outputs into the down result. Everything is a kernel, so it replays in the graph.
//
// Scope: single-token MUL_MAT_ID through MMVQ (ids->ne[1] == 1); SwiGLU experts named blk.N.ffn_{gate,up,
// down}_exps. Multi-token graphs (prefill) keep the scheduler's copy path. The pool reuses the VRAM the
// expert cache preallocates at load; the expert cache itself is not registered when DRE is on.

#include "dre.cuh"
#include "ggml-cpu.h"
// NVTX at file scope: included inside the anonymous namespace below, CUDA 12.0 rejects its weak symbols
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DRE_HAS_NVTX 1
#endif
#endif
#include "ggml-backend-impl.h"

#include <atomic>
#include <chrono>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>   // _mm_pause
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sched.h>
#include <unistd.h>
#endif

GGML_BACKEND_API_C void * ggml_cuda_expert_pool_take(size_t * bytes);   // ggml-cuda.cu
size_t ggml_cuda_trim_pools(ggml_backend_cuda_context & ctx);              // ggml-cuda.cu
static ggml_backend_cuda_context * g_dre_ctx = nullptr;                    // the decode backend, for the trim

bool ggml_cuda_dre_enabled() {
    static const bool v = [] { const char * e = getenv("ALPHA_DRE"); return e && atoi(e) != 0; }();
    return v;
}

static int dre_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

void ggml_cuda_outside_capture(void (*fn)(void *), void * arg);   // ggml-cuda.cu

namespace {

constexpr int     DRE_MAX_IDS  = 160;         // ids per layer per graph (n_expert_used * tokens)
constexpr int     DRE_MAX_TOK  = 8;           // tokens per graph (decode = 1, an MTP verify window = 1 + drafts)
constexpr int64_t DRE_SKIP     = INT64_MIN;   // == PCACHE_SLOT_SKIP (mmvq.cu)
constexpr int     DRE_MAX_EMBD = 16384;      // model width
constexpr int     DRE_MAX_X    = DRE_MAX_EMBD * DRE_MAX_TOK;

bool dre_cpu_mode() {
    static const bool v = dre_env_int("ALPHA_DRE_CPU", 0) != 0;
    return v;
}

struct dre_tensor {
    const ggml_tensor * t     = nullptr;
    const char *        host  = nullptr;   // t->data (device-readable after registration)
    size_t              esize = 0;         // nb[2]
    int                 layer = -1;        // index into g_dre.layers
    int                 role  = -1;        // 0 gate, 1 up, 2 down
    int64_t *           d_tab = nullptr;   // device table
    int64_t *           h_tab = nullptr;   // pinned mirror (uploaded when dirty)
    bool                dirty = false;
    // ALPHA_DRE_TIER: per-expert host source when the buffer is too big to pin or copy whole: the RAM hot tier's
    // arena copy for the hottest experts, the file mapping (pageable) for the rest. nullptr = host + e * esize.
    const char **       hp    = nullptr;
    uint8_t *           hpin  = nullptr;   // 1 = hp[e] is pinned (DMA / zero-copy readable)
    // every hp[e] == hp[0] + e * esize inside the pinned arena: the prefill copy paths (gather kernel, merged H2D runs)
    // may treat hp[0] as a contiguous host base again. Set once, after the arena is registered (dre_tier_fill).
    bool                contig = false;
};
static inline const char * dre_pipe_host(const dre_tensor & d, const char * file_base) { return d.contig ? d.hp[0] : file_base; }

static inline const char * dre_src(const dre_tensor & d, int64_t e) { return d.hp ? d.hp[e] : d.host + (size_t) e * d.esize; }
static inline bool dre_src_pinned(const dre_tensor & d, int64_t e) { return !d.hp || d.hpin[e]; }

// A copy from pageable host memory (an expert the RAM hot tier does not hold, or a tier piece not yet pinned) issued
// from a worker thread must not overlap a CUDA graph capture on the decode thread: the driver's staged pageable copy
// makes the decode graph's cudaGraphExecUpdate fail ("invalid argument"). Pinned sources take the plain path.
static bool dre_host_pinned(const void * p) {
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    return a.type == cudaMemoryTypeHost;
}
static void dre_h2d_worker(void * dst, const void * src, size_t n, cudaStream_t st) {
    if (dre_host_pinned(src)) { CUDA_CHECK(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, st)); return; }
    struct arg { void * d; const void * s; size_t n; cudaStream_t st; cudaError_t r; } a = { dst, src, n, st, cudaSuccess };
    ggml_cuda_outside_capture([](void * x) {
        arg * q = (arg *) x;
        q->r = cudaMemcpyAsync(q->d, q->s, q->n, cudaMemcpyHostToDevice, q->st);
    }, &a);
    CUDA_CHECK(a.r);
}
// experts [e0, e1) of d as runs whose host sources are contiguous: fn(dst offset, src, bytes) per run (one run when untiered)
template <typename F> static void dre_src_runs(const dre_tensor & d, int e0, int e1, F && fn) {
    int a = e0;
    while (a < e1) {
        const char * src = dre_src(d, a);
        int b = a + 1;
        while (b < e1 && dre_src(d, b) == src + (size_t) (b - a) * d.esize) b++;
        fn((size_t) a * d.esize, src, (size_t) (b - a) * d.esize);
        a = b;
    }
}

// per-layer doorbell block in mapped host memory, written by the GPU
struct dre_bell {
    volatile uint32_t seq;
    int32_t           n_ids;               // n_tok * n_used, token-major
    int32_t           n_tok;
    int32_t           ids[DRE_MAX_IDS];
    alignas(16) float x[DRE_MAX_X];        // n_tok rows of the layer input (16-byte aligned: float4 writes)
};
// per-layer answer block in mapped host memory, written by the CPU worker
struct dre_answer {
    volatile uint32_t done;
    int32_t           n_miss;
    int32_t           slot[DRE_MAX_IDS];   // which output (token * n_used + k) each answer row fills
};

struct dre_layer {
    int  blk = -1;
    int  ti[3] = { -1, -1, -1 };
    int  n_exp = 0;
    int  cls   = -1;
    size_t sub[3] = { 0, 0, 0 };           // offset of each role inside a unit slot
    std::vector<int>      slot;            // expert -> slot, -1 not resident, -2 pending
    std::vector<uint32_t> last_use, last_miss;
    std::vector<uint32_t> uses;            // lifetime use count (ALPHA_DRE_PROFILE_OUT)
    std::vector<float>    score;           // decayed use count (ALPHA_DRE_LFU), valid at token score_tok[e]
    std::vector<float>    prof;            // profile uses per token (ALPHA_DRE_PROFILE), empty without one
    std::vector<uint32_t> score_tok;
    std::vector<uint32_t> pf_count;        // prefill: how many ubatch copies asked for each expert (warm admission)
    bool pinned = false;                   // ALPHA_DRE_PIN_BLK: keep every expert resident (e.g. the MTP block)
    int  prev_miss[64]; int n_prev_miss = 0;   // unique experts the previous CPU job of this layer computed
    int32_t * ids_pin = nullptr;           // D2H ids readback (zero-copy mode)
    int       n_ids   = 0;
    // CPU mode
    dre_bell *   bell = nullptr;    dre_bell *   bell_d = nullptr;
    dre_answer * ans  = nullptr;    dre_answer * ans_d  = nullptr;
    float *      y    = nullptr;    float *      y_d    = nullptr;   // [DRE_MAX_IDS][n_out]
    uint32_t *   ctr_d = nullptr;   // device-side signal counter
    uint32_t     seen  = 0;         // worker: last seq handled
    bool complete() const { return ti[0] >= 0 && ti[1] >= 0 && ti[2] >= 0; }
};

struct dre_slot {
    char * ptr  = nullptr;   // device address of the unit
    int    chunk = -1;
    int    li   = -1;   // owning layer (-1 free)
    int    e    = -1;
    int    prev = -1, next = -1;   // LRU list (head = most recent)
    bool   pending = false;
    bool   pinned  = false;   // never evicted (ALPHA_DRE_PIN_BLK)
};

struct dre_class {
    size_t           size = 0;
    std::vector<int> free;
    int              head = -1, tail = -1;
};

struct dre_state {
    bool        init_done = false;
    bool        failed    = false;
    char *      pool      = nullptr;
    size_t      pool_size = 0;   // bytes in live chunks
    size_t      pool_used = 0;   // bytes carved from live chunks
    // the pool is a list of chunks so a long prefill can borrow some of it back (ggml_cuda_dre_borrow)
    struct chunk { char * base = nullptr; size_t size = 0, used = 0; bool alive = false; };
    std::vector<chunk> chunks;
    size_t      borrowed = 0;
    cudaStream_t dma      = nullptr;
    cudaEvent_t  ev_tok   = nullptr;
    bool         ev_tok_valid = false;
    std::vector<dre_tensor> tens;
    std::vector<dre_layer>  layers;
    std::unordered_map<const void *, int> by_data;
    std::unordered_map<int, int> layer_by_blk;
    std::unordered_map<const void *, bool> registered_buf;
    std::vector<dre_slot>  slots;
    std::vector<dre_class> classes;
    struct pacer_job;
    struct batch { cudaEvent_t ev; std::vector<int> slots; std::shared_ptr<pacer_job> pj; };
    std::vector<batch> batches;
    std::vector<cudaEvent_t> ev_free;
    uint32_t tok = 0;
    long long n_tok = 0, n_use = 0, n_hit = 0, n_admit = 0, n_evict = 0, n_publish = 0;
    long long n_pf_d2d = 0, n_pf_h2d = 0, n_pf_warm = 0;
    bool      pf_pending = false;          // a prefill recorded usage: warm-admit at the next decode
    std::atomic<long long> n_cpu{0}, n_cpu_u{0}, n_cpu_skip{0};
    int admit_per_tok = 1024;
    int ghost_w = 16;
};

dre_state g_dre;
int64_t * g_tab_d = nullptr;   // all device tables (arena)
int64_t * g_tab_h = nullptr;   // pinned host mirror of the arena
size_t    g_tab_cap = 0, g_tab_used = 0;
const ggml_tensor * g_dre_last_ids = nullptr;   // dedupe within one graph build: first call of a layer
const ggml_tensor * g_dre_last_bell_ids = nullptr;
unsigned long long * g_dre_wait_ns_d = nullptr;   // GPU time spent waiting for the CPU answers (stats)
std::atomic<long long> g_dre_cpu_ns{0};           // CPU worker time computing missed experts (stats)
// idle iterations before a worker thread naps: a decode token rings one doorbell per MoE layer a fraction of a ms
// apart, so a thread that naps between layers pays its wake-up latency on every miss (ALPHA_DRE_SPIN).
// A spinning thread must yield its core's issue slots: the host thread that launches the next token may be its
// hyperthread sibling.
static inline void dre_pause() {
#if defined(_MSC_VER) && defined(_M_X64)
    _mm_pause();
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}
const long g_dre_spin = [] { const char * e = getenv("ALPHA_DRE_SPIN"); return e ? atol(e) : 50000000L; }();

void dre_init() {
    if (g_dre.init_done) return;
    g_dre.init_done = true;
    size_t bytes = 0;
    void * base = ggml_cuda_expert_pool_take(&bytes);
    if (!base) {
        size_t free_mem = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total));
        // VRAM left free (ALPHA_DRE_RESERVE_MB, default 1280): covers a short prompt at the base micro-batch (it
        // borrows like any prefill, see serve-slot.h) and, with MTP on, the head KV preallocated after this claim
        size_t reserve = (size_t) dre_env_int("ALPHA_DRE_RESERVE_MB", 1280) << 20;
        // With ALPHA_MTP, this claim runs inside context creation (the first graph reservation), before the server can
        // budget anything: MTP then still needs the draft head's KV, the recurrent-state mirror and room for more
        // captured CUDA graphs. Default 0; the prefill KV fill is handled by ALPHA_MTP_KV_PREALLOC.
        if (dre_env_int("ALPHA_MTP", 0) != 0) {
            reserve += (size_t) std::max(0, dre_env_int("ALPHA_MTP_RESERVE_MB", 0)) << 20;
        }
        bytes = free_mem > reserve ? free_mem - reserve : 0;
        if (bytes < (256u << 20) || cudaMalloc(&base, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            g_dre.failed = true;
            fprintf(stderr, "[dre] no pool VRAM -- DRE disabled\n");
            return;
        }
    }
    // re-carve the load-time claim into chunks (same total) so prefill can borrow and return pieces
    {
        const size_t CH = (size_t) dre_env_int("ALPHA_DRE_CHUNK_MB", 512) << 20;
        CUDA_CHECK(cudaFree(base));
        size_t left = bytes;
        while (left >= (64u << 20)) {
            const size_t sz = std::min(CH, left);
            void * b = nullptr;
            if (cudaMalloc(&b, sz) != cudaSuccess) { (void) cudaGetLastError(); break; }
            dre_state::chunk c; c.base = (char *) b; c.size = sz; c.alive = true;
            g_dre.chunks.push_back(c);
            g_dre.pool_size += sz;
            left -= sz;
        }
        bytes = g_dre.pool_size;
    }
    g_dre.pool = g_dre.chunks.empty() ? nullptr : g_dre.chunks[0].base;
    CUDA_CHECK(cudaStreamCreateWithFlags(&g_dre.dma, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&g_dre.ev_tok, cudaEventDisableTiming));
    g_dre.admit_per_tok = dre_env_int("ALPHA_DRE_ADMIT", 96);
    g_dre.ghost_w       = dre_env_int("ALPHA_DRE_GHOST", 16);
    fprintf(stderr, "[dre] device-resolved experts ON: pool %.0f MB, admit <= %d units/token, misses %s\n",
            bytes / 1048576.0, g_dre.admit_per_tok, dre_cpu_mode() ? "computed on the CPU" : "read zero-copy");
}

// ALPHA_DRE_ARENA (default on when RAM allows): copy a registered host expert buffer out of the file mapping into an
// anonymous transparent-huge-page arena and serve every DRE host read (CPU miss workers, admission DMA, prefill staging)
// from it. The mapping is 4 KB page cache, which streams random experts noticeably slower than anonymous THP memory,
// and the CPU miss path is RAM-bound.
struct dre_arena_map { char * lo = nullptr; size_t len = 0; char * arena = nullptr; };
static std::vector<dre_arena_map> g_dre_arenas;
const char * dre_host_ptr(const void * p) {
    for (const auto & a : g_dre_arenas) {
        const char * c = (const char *) p;
        if (c >= a.lo && c < a.lo + a.len) return a.arena + (c - a.lo);
    }
    return (const char *) p;
}
static char * dre_make_arena(char * lo, size_t len) {
#if defined(__linux__)
    // default ON when the copy fits in available RAM with headroom (MemAvailable counts the reclaimable page
    // cache the copy displaces); ALPHA_DRE_ARENA=0 disables, =1 forces
    const int mode = dre_env_int("ALPHA_DRE_ARENA", -1);
    if (mode == 0) return nullptr;
    if (mode < 0) {
        long long avail_kb = 0;
        if (FILE * mf = fopen("/proc/meminfo", "r")) {
            char line[256];
            while (fgets(line, sizeof(line), mf)) if (sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
            fclose(mf);
        }
        const long long need_kb = (long long) (len >> 10) + (8LL << 20);   // + 8 GB headroom
        if (avail_kb < need_kb) {
            fprintf(stderr, "[dre] arena: skipped (%.1f GB available, %.1f GB needed)\n", avail_kb / 1048576.0, need_kb / 1048576.0);
            return nullptr;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    void * a = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a == MAP_FAILED) { fprintf(stderr, "[dre] arena: mmap of %.1f GB failed\n", len / 1073741824.0); return nullptr; }
    madvise(a, len, MADV_HUGEPAGE);
    // the model file behind [lo, lo+len) (from /proc/self/maps): its page cache is dropped for the range before the
    // arena is faulted (the memory the huge pages need) and again for each piece once copied. Any reader may do this.
    // ALPHA_DRE_ARENA_FADVISE (default 0): dropping the cache did not raise the huge-page share and made every later
    // start re-read the model from disk.
    int fdf = -1; long long foff = 0;
    FILE * mf = dre_env_int("ALPHA_DRE_ARENA_FADVISE", 0) != 0 ? fopen("/proc/self/maps", "r") : nullptr;
    if (mf) {
        char line[4096];
        while (fgets(line, sizeof(line), mf)) {
            unsigned long long s0 = 0, s1 = 0, off = 0; char path[3072] = { 0 };
            if (sscanf(line, "%llx-%llx %*s %llx %*s %*s %3071[^\n]", &s0, &s1, &off, path) == 4 &&
                (uintptr_t) lo >= s0 && (uintptr_t) lo < s1 && path[0] == '/') {
                fdf = open(path, O_RDONLY);
                foff = (long long) (off + ((uintptr_t) lo - s0));
                break;
            }
        }
        fclose(mf);
    }
    if (fdf >= 0) posix_fadvise(fdf, foff, (off_t) len, POSIX_FADV_DONTNEED);
    const int nt = std::max(1, dre_env_int("ALPHA_DRE_ARENA_THREADS", 16));
    std::vector<std::thread> th;
    const size_t step = ((len / nt) + (2u << 20) - 1) & ~(size_t) ((2u << 20) - 1);
    // ALPHA_DRE_ARENA_PREFAULT (default 1): fault the whole arena in as huge pages before the file is read. Copying
    // straight from the mapped file fills the page cache while the arena is being faulted, and the kernel then falls
    // back to 4 KB pages for most of it (the CPU expert path then slows with the TLB misses).
    if (dre_env_int("ALPHA_DRE_ARENA_PREFAULT", 1) != 0) {
        for (int t = 0; t < nt; t++) {
            th.emplace_back([=]() {
                const size_t b0 = std::min(len, (size_t) t * step), b1 = std::min(len, b0 + step);
                for (size_t o = b0; o < b1; o += (2u << 20)) ((volatile char *) a)[o] = 0;
            });
        }
        for (auto & x : th) x.join();
        th.clear();
    }
    for (int t = 0; t < nt; t++) {
        th.emplace_back([=]() {
            const size_t b0 = std::min(len, (size_t) t * step), b1 = std::min(len, b0 + step);
            // in 256 MB pieces, each file range paged out once copied: MADV_DONTNEED only unmaps it, and the page
            // cache left behind competes with the arena for memory
            for (size_t p0 = b0; p0 < b1; p0 += ((size_t) 256 << 20)) {
                const size_t p1 = std::min(b1, p0 + ((size_t) 256 << 20));
                memcpy((char *) a + p0, lo + p0, p1 - p0);
                madvise(lo + p0, p1 - p0, MADV_DONTNEED);   // our mapping of it ...
                if (fdf >= 0) posix_fadvise(fdf, foff + (off_t) p0, (off_t) (p1 - p0), POSIX_FADV_DONTNEED);   // ... and its cache
            }
        });
    }
    for (auto & x : th) x.join();
    if (fdf >= 0) close(fdf);
    fprintf(stderr, "[dre] arena: copied %.1f GB of host experts into anonymous huge pages in %.1f s\n", len / 1073741824.0,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return (char *) a;
#else
    (void) lo; (void) len;
    return nullptr;
#endif
}

static std::unordered_map<ggml_backend_buffer_t, bool> g_dre_tier_bufs;   // ALPHA_DRE_TIER: buffers served via hp[]
bool dre_register_buffer(const ggml_tensor * w) {
    ggml_backend_buffer_t buf = w->buffer;
    auto it = g_dre.registered_buf.find(buf);
    if (it != g_dre.registered_buf.end()) return it->second;
    char * b = (char *) ggml_backend_buffer_get_base(buf);
    size_t sz = ggml_backend_buffer_get_size(buf);
#if defined(__linux__)
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
#else
    const uintptr_t page = 4096;
#endif
    char * lo = (char *) ((uintptr_t) b & ~(page - 1));
    size_t len = (size_t) (((uintptr_t) b + sz + page - 1) & ~(page - 1)) - (size_t) (uintptr_t) lo;
    // ALPHA_DRE_ARENA_MAX_FRAC (default 0.6): a buffer bigger than this share of RAM goes straight to the hot tier.
    // Copied or pinned whole, the first shard of a multi-shard model takes the RAM and leaves the next shard's experts
    // on the SSD; tiered, the hottest units of every buffer share one RAM budget.
    bool tier_big = false;
#if defined(__linux__)
    {
        static const double frac = [] { const char * e = getenv("ALPHA_DRE_ARENA_MAX_FRAC"); return e ? atof(e) : 0.6; }();
        static const long long total_kb = [] {
            long long kb = 0;
            if (FILE * mf = fopen("/proc/meminfo", "r")) {
                char line[256];
                while (fgets(line, sizeof(line), mf)) if (sscanf(line, "MemTotal: %lld kB", &kb) == 1) break;
                fclose(mf);
            }
            return kb;
        }();
        tier_big = frac > 0 && total_kb > 0 && dre_cpu_mode() && dre_env_int("ALPHA_DRE_TIER", 1) != 0 &&
                   (double) len > frac * (double) (total_kb << 10);
    }
#endif
    if (char * ar = tier_big ? nullptr : dre_make_arena(lo, len)) {
        g_dre_arenas.push_back({ lo, len, ar });
        b  = ar + (b - lo);
        lo = ar;
    }
    // ALPHA_DRE_TIER_FORCE (test): treat every expert buffer as too big to pin, to exercise the hot tier where it fits
    static const bool tier_force = dre_env_int("ALPHA_DRE_TIER_FORCE", 0) != 0 && dre_cpu_mode() && g_dre_arenas.empty();
    cudaError_t err = (tier_force || tier_big) ? cudaErrorMemoryAllocation
                                 : cudaHostRegister(lo, len, cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        (void) cudaGetLastError();
        err = cudaSuccess;
    }
    bool ok = err == cudaSuccess;
    if (ok) {
        void * dptr = nullptr;
        if (cudaHostGetDevicePointer(&dptr, b, 0) != cudaSuccess || dptr != b) {
            (void) cudaGetLastError();
            ok = false;
            fprintf(stderr, "[dre] host buffer %p is not identity-mapped on the device -- not using DRE for it\n", b);
        }
    } else {
        (void) cudaGetLastError();
        // ALPHA_DRE_TIER (default 1, CPU-miss mode): too big for RAM to pin or copy whole: keep DRE on, serve the experts
        // from the file mapping and copy the hottest ones into a pinned RAM hot tier once every layer is known
        static const bool tier = dre_env_int("ALPHA_DRE_TIER", 1) != 0;
        if (tier && dre_cpu_mode()) {
            g_dre_tier_bufs[buf] = true;
            ok = true;
            fprintf(stderr, "[dre] tier: %.1f GB host expert buffer cannot be pinned whole (%s) -- file mapping + RAM hot tier\n",
                    len / 1073741824.0, cudaGetErrorString(err));
        } else {
            // zero-copy misses (ALPHA_DRE_CPU=0) read every expert straight from pinned host memory. Skipping one buffer
            // would leave the experts of a multi-shard model half-registered (a layer's gate/up in a DRE buffer, its down
            // in this one), which decodes NaN. All or nothing: DRE goes off for the whole model.
            g_dre.failed = true;
            fprintf(stderr, "[dre] cudaHostRegister of %.1f GB failed (%s) -- zero-copy misses need EVERY expert buffer "
                            "pinned: DRE OFF for this model (ALPHA_DRE_CPU=1, the default, serves it from a RAM hot tier)\n",
                    len / 1073741824.0, cudaGetErrorString(err));
        }
    }
    if (ok && !g_dre_tier_bufs.count(buf)) fprintf(stderr, "[dre] pinned host expert buffer: %.1f GB\n", len / 1073741824.0);
    g_dre.registered_buf[buf] = ok;
    return ok;
}

int dre_class_for(size_t size) {
    for (size_t i = 0; i < g_dre.classes.size(); i++) if (g_dre.classes[i].size == size) return (int) i;
    dre_class c; c.size = size;
    g_dre.classes.push_back(c);
    return (int) g_dre.classes.size() - 1;
}

// ALPHA_DRE_ZC_PCT: share of non-resident experts the GPU reads zero-copy over PCIe inside the same MMVQ launch
// instead of the CPU computing them: a second miss pipe in parallel with the CPU's RAM-bandwidth-bound one.
// Fixed per (layer, expert) by a hash, so the GPU table and the CPU worker agree without any per-token exchange.
// 0 = every miss on the CPU.
bool dre_zc(const dre_layer & L, int e) {
    static const int pct = dre_env_int("ALPHA_DRE_ZC_PCT", 0);
    if (pct <= 0) return false;
    const uint32_t h = (uint32_t) e * 2654435761u ^ (uint32_t) (L.blk + 1) * 40503u;
    return (int) ((h >> 8) % 100u) < pct;
}

int64_t dre_miss_entry(const dre_layer & L, const dre_tensor & d, int e) {
    if (dre_cpu_mode() && L.complete() && (!dre_zc(L, e) || !dre_src_pinned(d, e))) return DRE_SKIP;
    static const bool skip = dre_env_int("ALPHA_DRE_SKIP_MISS", 0) != 0;   // diagnostic, wrong output
    // bit 1 tags a zero-copy entry so the MMVQ stages the expert's rows through shared memory (coalesced PCIe reads)
    const int64_t off = (int64_t) (dre_src(d, e) - d.host);
    return skip ? DRE_SKIP : ((d.esize & 3) == 0 ? (off | 2) : off);
}

void dre_set_entries(dre_layer & L, int e) {
    for (int r = 0; r < 3; r++) {
        if (L.ti[r] < 0) continue;
        dre_tensor & d = g_dre.tens[L.ti[r]];
        const int s = L.slot[e];
        d.h_tab[e] = s >= 0 ? (int64_t) ((g_dre.slots[s].ptr + L.sub[r]) - d.host) : dre_miss_entry(L, d, e);
        d.dirty = true;
    }
}

void dre_layer_completed(dre_layer & L) {
    // the unit slot holds all three roles back to back
    size_t off = 0;
    for (int r = 0; r < 3; r++) {
        L.sub[r] = off;
        off += (g_dre.tens[L.ti[r]].esize + 255) & ~(size_t) 255;
    }
    L.cls = dre_class_for(off);
    if (dre_cpu_mode()) {
        const ggml_tensor * dn = g_dre.tens[L.ti[2]].t;
        const ggml_tensor * gt = g_dre.tens[L.ti[0]].t;
        GGML_ASSERT(gt->ne[0] <= DRE_MAX_EMBD && dn->ne[1] <= DRE_MAX_EMBD && "DRE CPU path: model too wide");
        CUDA_CHECK(cudaHostAlloc(&L.bell, sizeof(dre_bell), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&L.ans, sizeof(dre_answer), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostAlloc(&L.y, sizeof(float) * DRE_MAX_IDS * dn->ne[1], cudaHostAllocMapped));   // one row per (token, slot)
        memset((void *) L.bell, 0, sizeof(dre_bell));
        memset((void *) L.ans, 0, sizeof(dre_answer));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &L.bell_d, L.bell, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &L.ans_d, L.ans, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &L.y_d, L.y, 0));
        CUDA_CHECK(cudaMalloc(&L.ctr_d, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(L.ctr_d, 0, sizeof(uint32_t)));
    }
    for (int e = 0; e < L.n_exp; e++) dre_set_entries(L, e);
}

int dre_register(const ggml_tensor * w) {
    auto it = g_dre.by_data.find(w->data);
    if (it != g_dre.by_data.end()) return it->second;
    dre_init();
    int blk = -1;
    const char * nm = w->name;
    const int role = strstr(nm, "_gate_exps") ? 0 : strstr(nm, "_up_exps") ? 1 : strstr(nm, "_down_exps") ? 2 : -1;
    if (g_dre.failed || sscanf(nm, "blk.%d.", &blk) != 1 || role < 0 || !dre_register_buffer(w)) {
        g_dre.by_data[w->data] = -1;
        return -1;
    }
    auto lit = g_dre.layer_by_blk.find(blk);
    int li;
    if (lit == g_dre.layer_by_blk.end()) {
        li = (int) g_dre.layers.size();
        g_dre.layers.emplace_back();
        dre_layer & L = g_dre.layers.back();
        L.blk = blk;
        L.n_exp = (int) w->ne[2];
        L.slot.assign(L.n_exp, -1);
        L.last_use.assign(L.n_exp, 0);
        L.score.assign(L.n_exp, 0.0f);
        L.uses.assign(L.n_exp, 0);
        L.score_tok.assign(L.n_exp, 0);
        L.last_miss.assign(L.n_exp, 0);
        L.pf_count.assign(L.n_exp, 0);
        {
            static const int pin_blk = dre_env_int("ALPHA_DRE_PIN_BLK", -1);
            L.pinned = (blk == pin_blk);
        }
        CUDA_CHECK(cudaMallocHost(&L.ids_pin, DRE_MAX_IDS * sizeof(int32_t)));
        g_dre.layer_by_blk[blk] = li;
    } else {
        li = lit->second;
    }
    if (g_dre.layers[li].n_exp != (int) w->ne[2] || g_dre.layers[li].ti[role] >= 0) {
        g_dre.by_data[w->data] = -1;
        return -1;
    }
    dre_tensor d;
    d.t     = w;
    d.host  = dre_host_ptr(w->data);
    d.esize = w->nb[2];
    d.layer = li;
    d.role  = role;
    if (g_dre_tier_bufs.count(w->buffer)) {
        d.hp   = new const char * [w->ne[2]];
        d.hpin = new uint8_t[w->ne[2]]();
        for (int64_t e = 0; e < w->ne[2]; e++) d.hp[e] = (const char *) w->data + (size_t) e * d.esize;
    }
    const int n_exp = g_dre.layers[li].n_exp;
    // every table lives in one device arena with one pinned host mirror, so a graph's dirty tables upload as one
    // copy instead of many small ones ahead of every graph
    if (!g_tab_d) {
        g_tab_cap = (size_t) dre_env_int("ALPHA_DRE_TAB_CAP", 1 << 19);
        CUDA_CHECK(cudaMalloc(&g_tab_d, g_tab_cap * sizeof(int64_t)));
        CUDA_CHECK(cudaMallocHost(&g_tab_h, g_tab_cap * sizeof(int64_t)));
    }
    GGML_ASSERT(g_tab_used + (size_t) n_exp <= g_tab_cap && "ALPHA_DRE_TAB_CAP too small");
    d.d_tab = g_tab_d + g_tab_used;
    d.h_tab = g_tab_h + g_tab_used;
    g_tab_used += (size_t) ((n_exp + 1) & ~1);   // keep 16-byte alignment
    for (int e = 0; e < n_exp; e++) d.h_tab[e] = (int64_t) e * (int64_t) d.esize;
    const int ti = (int) g_dre.tens.size();
    g_dre.tens.push_back(d);
    g_dre.by_data[w->data] = ti;
    dre_layer & L = g_dre.layers[li];
    L.ti[role] = ti;
    if (L.complete()) dre_layer_completed(L);
    for (int r = 0; r < 3; r++) {
        if (L.ti[r] < 0) continue;
        dre_tensor & dd = g_dre.tens[L.ti[r]];
        CUDA_CHECK(cudaMemcpy(dd.d_tab, dd.h_tab, n_exp * sizeof(int64_t), cudaMemcpyHostToDevice));
        dd.dirty = false;
    }
    return ti;
}

// ---- LRU list per class
void lru_unlink(dre_class & c, int s) {
    dre_slot & x = g_dre.slots[s];
    if (x.prev >= 0) g_dre.slots[x.prev].next = x.next; else if (c.head == s) c.head = x.next;
    if (x.next >= 0) g_dre.slots[x.next].prev = x.prev; else if (c.tail == s) c.tail = x.prev;
    x.prev = x.next = -1;
}
void lru_push_front(dre_class & c, int s) {
    dre_slot & x = g_dre.slots[s];
    x.prev = -1; x.next = c.head;
    if (c.head >= 0) g_dre.slots[c.head].prev = s;
    c.head = s;
    if (c.tail < 0) c.tail = s;
}

// ALPHA_DRE_LFU (default on, half-life ALPHA_DRE_LFU_HALF=1024 tokens): frequency-aware replacement. LRU keeps what the
// last token used, which is the wrong set for an MTP verify: the draft token's predecessor was never decoded, so its
// experts were never admitted and verify graphs miss about 2x per token. A decayed use count keeps the generally
// popular experts resident instead: the victim is the least-scored of the class's ALPHA_DRE_LFU_WIN least-recent units,
// and a candidate is admitted only if it scores higher than that victim.
float dre_score(const dre_layer & L, int e) {
    static const float decay = (float) (dre_env_int("ALPHA_DRE_LFU_HALF", 1024) > 0 ? std::pow(0.5, 1.0 / dre_env_int("ALPHA_DRE_LFU_HALF", 1024)) : 0.997);
    const uint32_t dt = g_dre.tok - L.score_tok[e];
    return L.score[e] * std::pow(decay, (float) dt);
}
void dre_score_hit(dre_layer & L, int e) {
    // ALPHA_DRE_BOOST_BLK=b, ALPHA_DRE_BOOST=w: a use in block b counts w times (e.g. the MTP draft head, whose experts
    // are used less often per window than each main layer's, so plain LFU starves it and every draft waits on CPU
    // misses). A soft preference, unlike ALPHA_DRE_PIN_BLK.
    static const int   boost_blk = dre_env_int("ALPHA_DRE_BOOST_BLK", -1);
    static const float boost_w   = (float) dre_env_int("ALPHA_DRE_BOOST", 1);
    L.score[e] = dre_score(L, e) + (L.blk == boost_blk ? boost_w : 1.0f);
    L.score_tok[e] = g_dre.tok;
}
bool dre_lfu_on() {
    static const bool v = dre_env_int("ALPHA_DRE_LFU", 1) != 0;
    return v;
}
// returns a slot for the candidate (free, new, or an evicted low-score victim), or -1 to skip admission
int dre_take_slot_lfu(int ci, float cand_score) {
    dre_class & c = g_dre.classes[ci];
    if (!c.free.empty()) { int s = c.free.back(); c.free.pop_back(); return s; }
    for (int ch = 0; ch < (int) g_dre.chunks.size(); ch++) {
        auto & k = g_dre.chunks[ch];
        if (!k.alive || k.used + c.size > k.size) continue;
        dre_slot x; x.ptr = k.base + k.used; x.chunk = ch;
        k.used += c.size;
        g_dre.pool_used += c.size;
        g_dre.slots.push_back(x);
        return (int) g_dre.slots.size() - 1;
    }
    static const int win = std::max(1, dre_env_int("ALPHA_DRE_LFU_WIN", 32));
    int best = -1; float best_score = 0.0f; int seen = 0;
    for (int s = c.tail; s >= 0 && seen < win; s = g_dre.slots[s].prev) {
        dre_slot & x = g_dre.slots[s];
        if (x.pending || x.pinned) continue;
        dre_layer & V = g_dre.layers[x.li];
        if (V.last_use[x.e] >= g_dre.tok) continue;
        seen++;
        const float sc = dre_score(V, x.e);
        if (best < 0 || sc < best_score) { best = s; best_score = sc; }
    }
    if (best < 0 || best_score >= cand_score) return -1;
    dre_slot & x = g_dre.slots[best];
    dre_layer & V = g_dre.layers[x.li];
    V.slot[x.e] = -1;
    dre_set_entries(V, x.e);
    lru_unlink(c, best);
    x.li = -1; x.e = -1;
    g_dre.n_evict++;
    return best;
}

int dre_take_slot(int ci) {
    dre_class & c = g_dre.classes[ci];
    if (!c.free.empty()) { int s = c.free.back(); c.free.pop_back(); return s; }
    for (int ch = 0; ch < (int) g_dre.chunks.size(); ch++) {
        auto & k = g_dre.chunks[ch];
        if (!k.alive || k.used + c.size > k.size) continue;
        dre_slot x; x.ptr = k.base + k.used; x.chunk = ch;
        k.used += c.size;
        g_dre.pool_used += c.size;
        g_dre.slots.push_back(x);
        return (int) g_dre.slots.size() - 1;
    }
    for (int s = c.tail; s >= 0; s = g_dre.slots[s].prev) {
        dre_slot & x = g_dre.slots[s];
        if (x.pending || x.pinned) continue;
        dre_layer & V = g_dre.layers[x.li];
        if (V.last_use[x.e] >= g_dre.tok) continue;   // used by the token that just ran: keep
        // unpublish the victim before its slot is overwritten (the upload is ordered ahead of the copy)
        V.slot[x.e] = -1;
        dre_set_entries(V, x.e);
        lru_unlink(c, s);
        x.li = -1; x.e = -1;
        g_dre.n_evict++;
        return s;
    }
    return -1;
}

// ALPHA_DRE_TAB_KERNEL (default 1): the table upload is a small kernel reading the pinned host table directly. As a
// cudaMemcpyAsync on the compute stream it would queue on the copy engine behind the prompt's speculative expert copies
// (hundreds of MB), and the compute stream would wait for them.
static __global__ void dre_tab_copy_k(int64_t * __restrict__ dst, const int64_t * __restrict__ src, const size_t n) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) dst[i] = src[i];
}
void dre_upload_dirty(cudaStream_t st) {
    // one copy spanning every dirty table (clean tables inside the span are rewritten with identical bytes)
    int64_t * lo = nullptr, * hi = nullptr;
    for (dre_tensor & d : g_dre.tens) {
        if (!d.dirty) continue;
        int64_t * a = d.h_tab, * b = d.h_tab + g_dre.layers[d.layer].n_exp;
        if (!lo || a < lo) lo = a;
        if (!hi || b > hi) hi = b;
        d.dirty = false;
    }
    if (lo) {
        static const bool tab_k = dre_env_int("ALPHA_DRE_TAB_KERNEL", 1) != 0;
        const size_t n = (size_t) (hi - lo);
        int64_t * lo_d = nullptr;
        if (tab_k && ggml_backend_sched_alpha_async_inputs_on() && cudaHostGetDevicePointer((void **) &lo_d, lo, 0) == cudaSuccess && lo_d) {   // prompts only
            const int nb = (int) std::min<size_t>(256, (n + 255) / 256);
            dre_tab_copy_k<<<nb, 256, 0, st>>>(g_tab_d + (lo - g_tab_h), lo_d, n);
        } else {
            (void) cudaGetLastError();
            CUDA_CHECK(cudaMemcpyAsync(g_tab_d + (lo - g_tab_h), lo, n * sizeof(int64_t), cudaMemcpyHostToDevice, st));
        }
    }
}

// CPU miss path

static __global__ void dre_signal_k(const float * __restrict__ x, int nx, const int32_t * __restrict__ ids, int nids,
                                    int ntok, int ids_stride, dre_bell * bell, uint32_t * ctr) {
    const int n_used = nids / ntok;
    if ((nx & 3) == 0 && ((uintptr_t) x & 15) == 0 && ((uintptr_t) bell->x & 15) == 0) {   // 16-byte posted writes over PCIe
        const float4 * x4 = (const float4 *) x;
        float4 * b4 = (float4 *) bell->x;
        for (int i = threadIdx.x; i < nx/4; i += blockDim.x) b4[i] = x4[i];
    } else {
        for (int i = threadIdx.x; i < nx; i += blockDim.x) bell->x[i] = x[i];
    }
    // ids is a view of the router's argsort rows, so tokens sit ids_stride apart
    for (int i = threadIdx.x; i < nids; i += blockDim.x) bell->ids[i] = ids[(i / n_used) * ids_stride + i % n_used];
    __syncthreads();
    if (threadIdx.x == 0) {
        bell->n_ids = nids;
        bell->n_tok = ntok;
        __threadfence_system();
        const uint32_t s = *ctr + 1;
        *ctr = s;
        bell->seq = s;
        __threadfence_system();
    }
}

static __device__ __forceinline__ uint64_t dre_gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

static __global__ void dre_wait_fill_k(const uint32_t * __restrict__ ctr, const dre_answer * ans, const float * y,
                                       float * __restrict__ dst, int n_out, int64_t slot_stride, int n_used,
                                       int64_t tok_stride, unsigned long long * wait_ns) {
    // every block waits for the answer, then the blocks copy the missed experts' rows out of mapped host memory
    // together: one block reading 4-byte words over PCIe is latency-bound
    __shared__ int n_miss;
    if (threadIdx.x == 0) {
        const uint32_t want = *ctr;
        const volatile uint32_t * done = &ans->done;
        const uint64_t t0 = blockIdx.x == 0 ? dre_gtimer() : 0;
        while (*done != want) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(200);   // sm_70+ only; older archs (default PTX targets 50/61) spin without the yield
#endif
        }
        if (blockIdx.x == 0) atomicAdd(wait_ns, (unsigned long long) (dre_gtimer() - t0));
        __threadfence_system();
        n_miss = ((const volatile int32_t *) &ans->n_miss)[0];
    }
    __syncthreads();
    const bool vec = (n_out & 3) == 0 && (slot_stride & 3) == 0 && (tok_stride & 3) == 0 && ((uintptr_t) dst & 15) == 0;
    if (vec) {
        const int n4 = n_out / 4;
        const int total = n_miss * n4;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
            const int m = idx / n4, i = idx - m * n4;
            const int tk = ((const volatile int32_t *) ans->slot)[m];
            const float4 v = __ldcv((const float4 *) (y + (size_t) m * n_out) + i);
            float * d = dst + (int64_t) (tk / n_used) * tok_stride + (int64_t) (tk % n_used) * slot_stride;
            ((float4 *) d)[i] = v;
        }
    } else {
        const int total = n_miss * n_out;
        for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
            const int m = idx / n_out, i = idx - m * n_out;
            const int tk = ((const volatile int32_t *) ans->slot)[m];
            const volatile float * src = y + (size_t) m * n_out;
            float * d = dst + (int64_t) (tk / n_used) * tok_stride + (int64_t) (tk % n_used) * slot_stride;
            d[i] = src[i];
        }
    }
}

struct dre_pool_threads {
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};
    std::atomic<int>      next{0};
    std::atomic<int>      done{0};
    int                   n_items = 0;
    std::function<void(int)> fn;
    std::atomic<bool>     stop{false};

    void start(int n, const std::vector<int> & cpus) {
        for (int t = 0; t < n; t++) {
            th.emplace_back([this, t, cpus] {
#if defined(__linux__)
                if (t + 1 < (int) cpus.size()) {
                    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpus[t + 1], &cs);
                    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
                }
#endif
                uint64_t seen = 0;
                int idle = 0;
                while (!stop.load(std::memory_order_relaxed)) {
                    const uint64_t g = gen.load(std::memory_order_acquire);
                    if (g == seen) {
                        if (++idle > g_dre_spin) { std::this_thread::sleep_for(std::chrono::microseconds(50)); }
                        else dre_pause();
                        continue;
                    }
                    idle = 0;
                    seen = g;
                    work();
                }
            });
        }
    }
    void work() {
        int i;
        while ((i = next.fetch_add(1, std::memory_order_relaxed)) < n_items) {
            fn(i);
            done.fetch_add(1, std::memory_order_release);
        }
    }
    void run(int n, std::function<void(int)> f) {
        fn = std::move(f);
        n_items = n;
        done.store(0, std::memory_order_relaxed);
        next.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_acq_rel);
        work();
        while (done.load(std::memory_order_acquire) < n) { dre_pause(); }
    }
};

struct dre_cpu_worker {
    std::thread       th;
    std::atomic<bool> stop{false};
    dre_pool_threads  pool;
    bool              started = false;
};
dre_cpu_worker g_cpu;

// One layer: compute every non-resident routed expert (gate, up, SwiGLU, down) for the token in the bell.
void dre_cpu_layer_impl(dre_layer & L);
// per-token-count CPU stats: [T] = { layers, ns, misses, unique }
long long g_dre_tstat[DRE_MAX_TOK + 1][4];
std::chrono::steady_clock::time_point g_dre_stage_q, g_dre_stage_c;   // ALPHA_DRE_CPU_TIMERS stage marks
double g_dre_stage_acc[4]; long long g_dre_stage_n[2];
long long g_dre_pred[2];   // [0] unique misses the previous job of the same layer also missed, [1] unique misses
std::atomic<uint64_t> g_dre_pf_sink{0};
int g_dre_last_nm = 0, g_dre_last_nu = 0;
void dre_cpu_layer(dre_layer & L) {
    const auto t0 = std::chrono::steady_clock::now();
    g_dre_stage_q = g_dre_stage_c = t0;
    dre_cpu_layer_impl(L);
    const auto t1 = std::chrono::steady_clock::now();
    const long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    // ALPHA_DRE_CPU_TIMERS: stages of jobs that had misses: setup+quantise / compute / answer, and empty jobs
    if (g_dre_last_nm > 0) {
        g_dre_stage_acc[0] += std::chrono::duration<double, std::micro>(g_dre_stage_q - t0).count();
        g_dre_stage_acc[1] += std::chrono::duration<double, std::micro>(g_dre_stage_c - g_dre_stage_q).count();
        g_dre_stage_acc[2] += std::chrono::duration<double, std::micro>(t1 - g_dre_stage_c).count();
        g_dre_stage_n[0]++;
    } else {
        g_dre_stage_acc[3] += std::chrono::duration<double, std::micro>(t1 - t0).count();
        g_dre_stage_n[1]++;
    }
    g_dre_cpu_ns += ns;
    const int T = std::min(DRE_MAX_TOK, std::max(1, (int) L.bell->n_tok));
    g_dre_tstat[T][0]++; g_dre_tstat[T][1] += ns; g_dre_tstat[T][2] += g_dre_last_nm; g_dre_tstat[T][3] += g_dre_last_nu;
    // ALPHA_DRE_OVERLAP_STATS (diagnostic): how often tokens of one multi-token graph route to the same expert: the
    // bound on what a kernel reading each expert once for all its tokens could save. Per token count T: (token,
    // expert) pairs, distinct experts, and the same restricted to pool-resident experts (the GPU-read part).
    static const bool ov = dre_env_int("ALPHA_DRE_OVERLAP_STATS", 0) != 0;
    if (ov && T >= 2) {
        static long long st[DRE_MAX_TOK + 1][5];   // layers, pairs, distinct, resident pairs, resident distinct
        int seen[DRE_MAX_IDS]; int ns2 = 0, nr = 0, nrd = 0;
        const int n = std::min((int) L.bell->n_ids, DRE_MAX_IDS);
        for (int i = 0; i < n; i++) {
            const int e = L.bell->ids[i];
            const bool res = e >= 0 && e < L.n_exp && L.slot[e] >= 0;
            if (res) nr++;
            bool dup = false;
            for (int j = 0; j < ns2; j++) if (seen[j] == e) { dup = true; break; }
            if (!dup) { seen[ns2++] = e; if (res) nrd++; }
        }
        st[T][0]++; st[T][1] += n; st[T][2] += ns2; st[T][3] += nr; st[T][4] += nrd;
        static long long nlog = 0;
        if ((++nlog % 20000) == 0) {
            for (int t = 2; t <= DRE_MAX_TOK; t++) {
                if (!st[t][0]) continue;
                fprintf(stderr, "[dre-overlap] T=%d: %lld layer graphs, %.1f pairs -> %.1f distinct experts (%.1f%% duplicate); "
                        "resident %.1f pairs -> %.1f distinct (%.1f%% duplicate)\n", t, st[t][0],
                        (double) st[t][1] / st[t][0], (double) st[t][2] / st[t][0], 100.0 * (1.0 - (double) st[t][2] / st[t][1]),
                        (double) st[t][3] / st[t][0], (double) st[t][4] / st[t][0],
                        st[t][3] ? 100.0 * (1.0 - (double) st[t][4] / st[t][3]) : 0.0);
            }
        }
    }
}
void dre_cpu_layer_impl(dre_layer & L) {
    const dre_tensor & G = g_dre.tens[L.ti[0]];
    const dre_tensor & U = g_dre.tens[L.ti[1]];
    const dre_tensor & D = g_dre.tens[L.ti[2]];
    const int n_ids = L.bell->n_ids;
    const int n_tok = std::max(1, (int) L.bell->n_tok);
    const int n_used = n_ids / n_tok;
    const int K  = (int) G.t->ne[0];    // model width
    const int F  = (int) G.t->ne[1];    // expert FFN width
    const int NO = (int) D.t->ne[1];    // output width
    int miss_e[DRE_MAX_IDS], miss_k[DRE_MAX_IDS], miss_t[DRE_MAX_IDS], nm = 0;
    g_dre_last_nm = 0; g_dre_last_nu = 0;
    // ALPHA_DRE_CPU_NOOP: timing probe only, wrong output. Answers every doorbell at once with no misses, so a run
    // measures the GPU-only floor of a decode / verify graph (what the CPU miss path costs on top).
    static const bool cpu_noop = dre_env_int("ALPHA_DRE_CPU_NOOP", 0) != 0;
    for (int k = 0; k < n_ids && k < DRE_MAX_IDS && !cpu_noop; k++) {
        const int e = L.bell->ids[k];
        if (e >= 0 && e < L.n_exp && L.slot[e] < 0 && !dre_zc(L, e)) { miss_e[nm] = e; miss_k[nm] = k; miss_t[nm] = k / std::max(1, n_used); nm++; }
    }
    {   // how many of this job's (unique) misses the previous job of the same layer also missed (the
        // predictor ALPHA_DRE_CPU_PREFETCH uses), then remember this job's set for the next one
        int uniq[64], nuq = 0;
        for (int m = 0; m < nm; m++) {
            bool dup = false;
            for (int u = 0; u < nuq; u++) if (uniq[u] == miss_e[m]) { dup = true; break; }
            if (!dup && nuq < 64) uniq[nuq++] = miss_e[m];
        }
        for (int u = 0; u < nuq; u++) {
            for (int q = 0; q < L.n_prev_miss; q++) if (L.prev_miss[q] == uniq[u]) { g_dre_pred[0]++; break; }
        }
        g_dre_pred[1] += nuq;
        if (nuq > 0) { std::memcpy(L.prev_miss, uniq, sizeof(int) * nuq); L.n_prev_miss = nuq; }
    }
    // ALPHA_DRE_APPROX_BLK=b: the layer of block b (the MTP draft head) answers its misses with zero rows instead
    // of computing them, so the missed experts drop out of that layer's weighted sum. Only for a layer whose output
    // is a guess: a draft is checked by an exact verify, so this can change acceptance, never the output.
    static const int approx_blk = dre_env_int("ALPHA_DRE_APPROX_BLK", -1);
    if (nm > 0 && L.blk == approx_blk) {
        std::memset(L.y, 0, sizeof(float) * (size_t) nm * NO);
        g_dre.n_cpu_skip += nm;
    } else if (nm > 0) {
        const ggml_type_traits_cpu * tg = ggml_get_type_traits_cpu(G.t->type);
        const ggml_type_traits_cpu * tu = ggml_get_type_traits_cpu(U.t->type);
        const ggml_type_traits_cpu * td = ggml_get_type_traits_cpu(D.t->type);
        static std::vector<char> qx_g, qx_u, qa;
        static std::vector<float> gbuf, ubuf, abuf;
        const size_t qg_row = ggml_row_size(tg->vec_dot_type, K);
        const size_t qu_row = ggml_row_size(tu->vec_dot_type, K);
        qx_g.resize(qg_row * n_tok);
        qx_u.resize(qu_row * n_tok);
        const bool same_q = tu->vec_dot_type == tg->vec_dot_type;
        for (int t = 0; t < n_tok; t++) {
            ggml_get_type_traits_cpu(tg->vec_dot_type)->from_float(L.bell->x + (size_t) t * K, qx_g.data() + t * qg_row, K);
            if (!same_q) ggml_get_type_traits_cpu(tu->vec_dot_type)->from_float(L.bell->x + (size_t) t * K, qx_u.data() + t * qu_row, K);
        }
        g_dre_stage_q = std::chrono::steady_clock::now();
        const char * qxg = qx_g.data();
        const char * qxu = same_q ? qx_g.data() : qx_u.data();
        const size_t qxu_row = same_q ? qg_row : qu_row;
        gbuf.resize((size_t) nm * F);
        ubuf.resize((size_t) nm * F);
        // group the misses by expert: in a multi-token (verify) graph several tokens often miss the same expert,
        // and each weight row is then read from RAM once and dotted against every token that needs it
        int ug_first[DRE_MAX_IDS], ug_next[DRE_MAX_IDS], nu = 0;
        {
            int last_of[DRE_MAX_IDS];
            for (int m = 0; m < nm; m++) {
                ug_next[m] = -1;
                int u = 0;
                while (u < nu && miss_e[ug_first[u]] != miss_e[m]) u++;
                if (u == nu) { ug_first[nu] = m; last_of[nu] = m; nu++; }
                else { ug_next[last_of[u]] = m; last_of[u] = m; }
            }
        }
        // gate + up rows, chunked across the pool
        static const int RCH = std::max(8, dre_env_int("ALPHA_DRE_CPU_RCH", 32));   // gate/up rows per work item (down: 4x)
        const int nch_f = (F + RCH - 1) / RCH;
        float * gb = gbuf.data(); float * ub = ubuf.data();
        abuf.resize((size_t) nm * F);
        const size_t qa_row = ggml_row_size(td->vec_dot_type, F);
        qa.resize(qa_row * nm);
        const int OCH = RCH * 4;
        const int nch_o = (NO + OCH - 1) / OCH;
        float * yb = L.y;
        const char * qab = qa.data();
        auto gate_up_item = [&](int it) {
            const int which = it % 2;   // 0 gate, 1 up
            const int rest  = it / 2;
            const int u = rest / nch_f, c = rest % nch_f;
            const dre_tensor & W = which ? U : G;
            const ggml_type_traits_cpu * tt = which ? tu : tg;
            const char * base = dre_src(W, miss_e[ug_first[u]]);
            const size_t rb = W.t->nb[1];
            const int r1 = std::min(F, (c + 1) * RCH);
            for (int m = ug_first[u]; m >= 0; m = ug_next[m]) {
                const char * q = which ? qxu + miss_t[m] * qxu_row : qxg + miss_t[m] * qg_row;
                float * out = (which ? ub : gb) + (size_t) m * F;
                for (int r = c * RCH; r < r1; r++) tt->vec_dot(K, &out[r], 0, base + (size_t) r * rb, 0, q, 0, 1);
            }
        };
        // ALPHA_DRE_CPU_Q2P (default on): Q2_0 down rows against a once-prepared activation (bit-plane layout,
        // hoisted sums), about 2x ggml_vec_dot_q2_0_q8_0 per core; this path is compute-bound, not RAM-bound.
        static const bool q2p_env = dre_env_int("ALPHA_DRE_CPU_Q2P", 1) != 0;
        const bool q2p = q2p_env && D.t->type == GGML_TYPE_Q2_0 && td->vec_dot_type == GGML_TYPE_Q8_0 &&
                         ggml_cpu_q2_0_prepared_available();
        static std::vector<char> q2prep;
        const size_t q2psz = q2p ? (ggml_cpu_q2_0_prep_size(F) + 63) & ~(size_t) 63 : 0;
        if (q2p) q2prep.resize(q2psz * nm + 64);
        char * q2pbase = q2p ? (char *) (((uintptr_t) q2prep.data() + 63) & ~(uintptr_t) 63) : nullptr;
        auto swiglu_q = [&](int m) {
            float * a = abuf.data() + (size_t) m * F;
            const float * g = gbuf.data() + (size_t) m * F;
            const float * u = ubuf.data() + (size_t) m * F;
            for (int r = 0; r < F; r++) a[r] = (g[r] / (1.0f + expf(-g[r]))) * u[r];
            ggml_get_type_traits_cpu(td->vec_dot_type)->from_float(a, qa.data() + m * qa_row, F);
            if (q2p) ggml_cpu_q2_0_prepare(qa.data() + m * qa_row, F, q2pbase + (size_t) m * q2psz);
        };
        auto down_item = [&](int it) {
            const int u = it / nch_o, c = it % nch_o;
            const char * base = dre_src(D, miss_e[ug_first[u]]);
            const size_t rb = D.t->nb[1];
            const int r0 = c * OCH, r1 = std::min(NO, r0 + OCH);
            for (int m = ug_first[u]; m >= 0; m = ug_next[m]) {
                if (q2p) {
                    ggml_cpu_q2_0_dot_prepared(F, r1 - r0, base + (size_t) r0 * rb, rb, q2pbase + (size_t) m * q2psz,
                                               &yb[(size_t) m * NO + r0]);
                    continue;
                }
                for (int r = r0; r < r1; r++) td->vec_dot(F, &yb[(size_t) m * NO + r], 0, base + (size_t) r * rb, 0, qab + m * qa_row, 0, 1);
            }
        };
        // ALPHA_DRE_CPU_ONEPASS (default on): one pool dispatch per layer. The thread that completes an expert's
        // last gate/up chunk runs its SwiGLU + quantise and releases that expert's down chunks, so there is no
        // second barrier and no serial step on the dispatching thread between the two halves.
        static const bool onepass = dre_env_int("ALPHA_DRE_CPU_ONEPASS", 1) != 0;
        if (onepass) {
            static std::atomic<int> gu_left[DRE_MAX_IDS];
            static std::atomic<int> ready[DRE_MAX_IDS];
            for (int u = 0; u < nu; u++) { gu_left[u].store(nch_f * 2, std::memory_order_relaxed); ready[u].store(0, std::memory_order_relaxed); }
            const int n_gu = nu * nch_f * 2;
            g_cpu.pool.run(n_gu + nu * nch_o, [&](int it) {
                if (it < n_gu) {
                    gate_up_item(it);
                    const int u = (it / 2) / nch_f;
                    if (gu_left[u].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        for (int m = ug_first[u]; m >= 0; m = ug_next[m]) swiglu_q(m);
                        ready[u].store(1, std::memory_order_release);
                    }
                } else {
                    const int d = it - n_gu;
                    const int u = d / nch_o;
                    while (!ready[u].load(std::memory_order_acquire)) dre_pause();
                    down_item(d);
                }
            });
        } else {
            g_cpu.pool.run(nu * nch_f * 2, gate_up_item);
            for (int m = 0; m < nm; m++) swiglu_q(m);
            g_cpu.pool.run(nu * nch_o, down_item);
        }
        g_dre_stage_c = std::chrono::steady_clock::now();
        g_dre.n_cpu += nm;
        g_dre.n_cpu_u += nu;
        g_dre_last_nm = nm; g_dre_last_nu = nu;
    }
    for (int m = 0; m < nm; m++) L.ans->slot[m] = miss_k[m];
    L.ans->n_miss = nm;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    L.ans->done = L.bell->seq;
}

// ALPHA_DRE_CPU_AUTO (default when neither ALPHA_DRE_CPU_LIST nor ALPHA_DRE_CPU_THREADS is set): the CPU miss pool from
// the machine's topology: one thread per physical performance core plus every efficiency core but the last two (left to
// the GPU-launch and dispatch threads); on a non-hybrid CPU every physical core but one. Within the process's affinity mask.
static std::vector<int> dre_parse_cpu_list(const char * l) {
    std::vector<int> v;
    for (const char * p = l; p && *p; ) {
        char * end = nullptr;
        const long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        if (*end == '-') { p = end + 1; b = strtol(p, &end, 10); }
        for (long c = a; c <= b; c++) v.push_back((int) c);
        p = (*end == ',') ? end + 1 : end;
        while (*p == '\n' || *p == ' ') p++;
    }
    return v;
}
static std::vector<int> dre_cpu_auto_list() {
    std::vector<int> out;
#if defined(__linux__)
    auto read_list = [](const char * path) {
        std::vector<int> v;
        if (FILE * f = fopen(path, "r")) {
            char buf[1024] = {};
            if (fgets(buf, sizeof(buf), f)) v = dre_parse_cpu_list(buf);
            fclose(f);
        }
        return v;
    };
    cpu_set_t aff; CPU_ZERO(&aff);
    const bool have_aff = sched_getaffinity(0, sizeof(aff), &aff) == 0;
    auto allowed = [&](int c) { return !have_aff || CPU_ISSET(c, &aff); };
    std::vector<int> pcores = read_list("/sys/devices/cpu_core/cpus");
    std::vector<int> ecores = read_list("/sys/devices/cpu_atom/cpus");
    const bool hybrid = !pcores.empty() && !ecores.empty();
    if (!hybrid) {
        pcores.clear(); ecores.clear();
        const long n = sysconf(_SC_NPROCESSORS_ONLN);
        for (int c = 0; c < n; c++) pcores.push_back(c);
    }
    // one logical cpu per physical core: the lowest of its SMT siblings
    std::vector<int> prim;
    for (int c : pcores) {
        if (!allowed(c)) continue;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        std::vector<int> sib = read_list(path);
        const int first = sib.empty() ? c : *std::min_element(sib.begin(), sib.end());
        if (first == c) prim.push_back(c);
    }
    std::vector<int> eff;
    for (int c : ecores) if (allowed(c)) eff.push_back(c);
    if (hybrid && eff.size() >= 4) eff.resize(eff.size() - 2);
    else if (!prim.empty() && prim.size() > 2) prim.pop_back();
    out = prim;
    out.insert(out.end(), eff.begin(), eff.end());
#endif
    return out;
}

void dre_cpu_start() {
    if (g_cpu.started) return;
    g_cpu.started = true;
    const int nthr = std::max(1, dre_env_int("ALPHA_DRE_CPU_THREADS", 8));
    // one thread per physical P-core on a hybrid Intel part: logical 0,2,4,... by default
    std::vector<int> cpus;
    const int stride = dre_env_int("ALPHA_DRE_CPU_STRIDE", 2), first = dre_env_int("ALPHA_DRE_CPU_FIRST", 0);
    for (int t = 0; t < nthr; t++) cpus.push_back(first + t * stride);
    // ALPHA_DRE_CPU_LIST=0,2,4,...: explicit logical cpus (e.g. every P-core plus every E-core on a hybrid part);
    // overrides THREADS/STRIDE. Work is pulled from a shared counter, so slower cores just take fewer chunks.
    if (const char * l = getenv("ALPHA_DRE_CPU_LIST")) {
        cpus = dre_parse_cpu_list(l);
    } else if (!getenv("ALPHA_DRE_CPU_THREADS") && dre_env_int("ALPHA_DRE_CPU_AUTO", 1) != 0) {
        std::vector<int> a = dre_cpu_auto_list();
        if (a.size() >= 2) cpus = a;
    }
    const int nthr_eff = cpus.empty() ? nthr : (int) cpus.size();
    g_cpu.pool.start(nthr_eff - 1, cpus);
    g_cpu.th = std::thread([cpus] {
#if defined(__linux__)
        if (!cpus.empty()) {
            cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpus[0], &cs);
            pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
        }
#endif
        int idle = 0;
        while (!g_cpu.stop.load(std::memory_order_relaxed)) {
            bool any = false;
            for (size_t li = 0; li < g_dre.layers.size(); li++) {
                dre_layer & L = g_dre.layers[li];
                if (!L.bell) continue;
                const uint32_t s = L.bell->seq;
                if (s != L.seen) {
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    L.seen = s;
                    dre_cpu_layer(L);
                    any = true;
                    // ALPHA_DRE_CPU_PREFETCH (opt-in): while the GPU runs the next layer's attention, pull into the
                    // CPU caches the experts the next layer missed last time (still non-resident): its miss compute
                    // then reads L3 instead of DRAM for the ones that repeat.
                    static const bool pf = dre_env_int("ALPHA_DRE_CPU_PREFETCH", 0) != 0;
                    if (pf && li + 1 < g_dre.layers.size()) {
                        dre_layer & N = g_dre.layers[li + 1];
                        if (N.complete() && N.n_prev_miss > 0) {
                            static std::vector<std::pair<const char *, size_t>> rng;   // dispatcher-written, pool-read
                            rng.clear();
                            for (int q = 0; q < N.n_prev_miss; q++) {
                                const int e = N.prev_miss[q];
                                if (e < 0 || e >= N.n_exp || N.slot[e] >= 0) continue;
                                for (int r = 0; r < 3; r++) {
                                    const dre_tensor & W = g_dre.tens[N.ti[r]];
                                    const char * base = dre_src(W, e);
                                    for (size_t o = 0; o < W.esize; o += 65536) rng.push_back({ base + o, std::min<size_t>(65536, W.esize - o) });
                                }
                            }
                            if (!rng.empty()) {
                                g_cpu.pool.run((int) rng.size(), [](int it) {
                                    const char * p = rng[it].first; const size_t n = rng[it].second;
                                    uint64_t acc = 0;
                                    for (size_t o = 0; o < n; o += 64) acc += *(const volatile uint64_t *) (p + o);
                                    g_dre_pf_sink += acc;
                                });
                            }
                        }
                    }
                }
            }
            if (any) idle = 0;
            else if (++idle > g_dre_spin) std::this_thread::sleep_for(std::chrono::microseconds(20));
            else dre_pause();
        }
    });
    g_cpu.th.detach();
    {
        std::string lst;
        for (size_t i = 0; i < cpus.size(); i++) { if (i) lst += ','; lst += std::to_string(cpus[i]); }
        fprintf(stderr, "[dre] CPU miss worker: %d threads (logical cpus %s)%s\n", nthr_eff, lst.c_str(),
                getenv("ALPHA_DRE_CPU_LIST") ? " [ALPHA_DRE_CPU_LIST]" : getenv("ALPHA_DRE_CPU_THREADS") ? "" : " [auto]");
    }
#if defined(__linux__)
    // ALPHA_DRE_HOST_PIN (default 1): this is the decode thread (dre_cpu_start runs inside its first expert graph): keep it
    // on the performance-core logical cpus the miss workers do not use (hybrid parts; elsewhere every cpu they do not
    // use). Unpinned, the scheduler moves it onto E-cores and busy worker cores, and the host work between the window's
    // graphs (draft launches, the fold, the verify's inputs and launch) runs slower.
    if (dre_env_int("ALPHA_DRE_HOST_PIN", 1) != 0 && !cpus.empty()) {
        auto read_list = [](const char * path) {
            std::vector<int> v;
            if (FILE * f = fopen(path, "r")) {
                char buf[1024] = {};
                if (fgets(buf, sizeof(buf), f)) v = dre_parse_cpu_list(buf);
                fclose(f);
            }
            return v;
        };
        cpu_set_t aff; CPU_ZERO(&aff);
        const bool have_aff = sched_getaffinity(0, sizeof(aff), &aff) == 0;
        std::vector<int> cand = read_list("/sys/devices/cpu_core/cpus");   // hybrid: the P-cores only
        if (cand.empty() || read_list("/sys/devices/cpu_atom/cpus").empty()) {
            cand.clear();
            const long n = sysconf(_SC_NPROCESSORS_ONLN);
            for (int c = 0; c < n; c++) cand.push_back(c);
        }
        cpu_set_t hs; CPU_ZERO(&hs);
        std::string hl; int nh = 0;
        for (int c : cand) {
            if (have_aff && !CPU_ISSET(c, &aff)) continue;
            if (std::find(cpus.begin(), cpus.end(), c) != cpus.end()) continue;
            CPU_SET(c, &hs); nh++;
            if (!hl.empty()) hl += ',';
            hl += std::to_string(c);
        }
        if (nh > 0 && pthread_setaffinity_np(pthread_self(), sizeof(hs), &hs) == 0) {
            fprintf(stderr, "[dre] decode thread pinned to logical cpus %s\n", hl.c_str());
        }
    }
#endif
}

} // namespace

// Pin (and arena-copy) the host buffer of an expert tensor now instead of on the first graph that reads it. Without
// this the MTP sidecar's experts, read by no warmup, pay a multi-second fault-in + copy inside the first request that
// drafts. Buffer-level only: no pool or layer state, so it is safe right after model load.
GGML_BACKEND_API_C void ggml_cuda_dre_preregister(const ggml_tensor * w) {
    if (!ggml_cuda_dre_enabled() || !w || !w->buffer || !ggml_backend_buffer_is_host(w->buffer) || w->ne[2] < 2 ||
        !ggml_is_quantized(w->type)) {
        return;
    }
    (void) dre_register_buffer(w);
}

GGML_BACKEND_API_C bool ggml_cuda_dre_direct_weight(const ggml_tensor * node, const ggml_tensor * w) {
    if (!ggml_cuda_dre_enabled() || g_dre.failed || node->op != GGML_OP_MUL_MAT_ID || node->src[0] != w) return false;
    const ggml_tensor * ids = node->src[2];
    // register on any graph (a prefill comes first): the prefill copy hook below needs the tables too
    if (w->buffer && ggml_backend_buffer_is_host(w->buffer) && w->ne[2] >= 2 && ggml_is_quantized(w->type)) {
        (void) dre_register(w);
    }
    // decode and small verify windows only; prefill keeps the scheduler's copy path
    static const int max_tok = std::min(DRE_MAX_TOK, std::max(1, dre_env_int("ALPHA_DRE_MAX_TOK", DRE_MAX_TOK)));
    if (!ids || ids->ne[1] < 1 || ids->ne[1] > max_tok || ids->ne[0] * ids->ne[1] > DRE_MAX_IDS) return false;
    if (!w->buffer || !ggml_backend_buffer_is_host(w->buffer) || w->ne[2] < 2 || !ggml_is_quantized(w->type)) return false;
    return dre_register(w) >= 0;
}

// Prefill expert staging (the scheduler's per-expert copy hook, multi-token graphs only): a resident expert is copied
// VRAM->VRAM from its pool slot instead of over PCIe, and every request is counted so the next decode can warm-admit
// what the prompt actually used (a new prompt otherwise starts decode on a cold pool). Host copies are coalesced: a
// prefill asks for nearly every expert of a tensor in id order, and one small cudaMemcpyAsync per expert runs well
// below link speed.
namespace {
struct dre_pf_run {
    cudaStream_t st = nullptr;
    char * dst = nullptr; const char * src = nullptr;   // base of the run
    size_t bytes = 0, pad = 0;
    const void * tensor = nullptr; int64_t next_eid = -1;
} g_pf_run;
void dre_pf_flush() {
    if (g_pf_run.bytes) {
        // the MMQ padding past the run is the next expert's first bytes: contiguous in the file, but elsewhere (or absent)
        // when the tensor is tiered (ALPHA_DRE_TIER), where copying the run's own tail would overwrite that expert's start
        const char * pad_src = g_pf_run.src + g_pf_run.bytes;
        if (g_pf_run.pad) {
            auto it = g_dre.by_data.find(g_pf_run.tensor);
            if (it != g_dre.by_data.end() && it->second >= 0 && g_dre.tens[it->second].hp) {
                const dre_tensor & d = g_dre.tens[it->second];
                pad_src = g_pf_run.next_eid < d.t->ne[2] ? dre_src(d, g_pf_run.next_eid) : nullptr;
            }
        }
        if (pad_src == g_pf_run.src + g_pf_run.bytes) {
            CUDA_CHECK(cudaMemcpyAsync(g_pf_run.dst, g_pf_run.src, g_pf_run.bytes + g_pf_run.pad, cudaMemcpyHostToDevice, g_pf_run.st));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(g_pf_run.dst, g_pf_run.src, g_pf_run.bytes, cudaMemcpyHostToDevice, g_pf_run.st));
            if (pad_src) CUDA_CHECK(cudaMemcpyAsync(g_pf_run.dst + g_pf_run.bytes, pad_src, g_pf_run.pad, cudaMemcpyHostToDevice, g_pf_run.st));
        }
    }
    g_pf_run = dre_pf_run();
}
}

// ALPHA_DRE_PF_STREAM (default off): layer-ahead expert streaming for prefill. The scheduler synchronises on each
// MoE layer's router before it asks for that layer's experts, so the plain path is strictly
//     route L -> copy L -> compute L -> route L+1 -> copy L+1 ...
// and copies never overlap compute. Here the first request for a tensor of layer L also starts streaming the same
// role of layer L+1 (every expert, since a big micro-batch uses nearly all of them) into a DRE-owned buffer on a
// separate stream, so it lands while L computes. The scheduler's staging tensor is then redirected to that buffer
// (no device copy). Weights never change, so a buffer stays valid for its tensor across micro-batches. Two buffers
// per role (gate/up/down).
namespace {
// The largest staging buffer any layer needs for role r (gate/up/down): models can mix expert quant types per layer,
// and a buffer sized for the current layer would be freed and re-allocated (a stream sync) at every larger-expert layer.
size_t dre_role_max_need(int r) {
    static size_t mx[3] = { 0, 0, 0 };
    if (mx[r] == 0) {
        for (const auto & L : g_dre.layers) {
            if (!L.complete()) continue;
            mx[r] = std::max(mx[r], g_dre.tens[L.ti[r]].esize * (size_t) L.n_exp + 512);
        }
    }
    return mx[r];
}
struct dre_pfs_buf { char * p = nullptr; size_t cap = 0; int tens = -1; cudaEvent_t ready = nullptr, free = nullptr; };
// ALPHA_DRE_PF_DEPTH=d (default 1): how many layers ahead each role streams. With 1, layer L+1's copy starts when
// layer L asks for its experts, and the scheduler's per-layer router sync leaves the copy engine idle between layers.
// d layers ahead keep it busy; slot = blk % (d+1).
constexpr int DRE_PFS_MAX = 4;
dre_pfs_buf g_pfs[3][DRE_PFS_MAX];
int dre_pfs_depth() {
    static const int v = std::max(1, std::min(DRE_PFS_MAX - 1, dre_env_int("ALPHA_DRE_PF_DEPTH", 1)));
    return v;
}
cudaStream_t g_pfs_stream = nullptr;
// ALPHA_DRE_PIPE_RSTREAM (default 1): the routed (critical) copies of a layer go on their own high-priority stream.
// On the one copy stream they would queue behind the speculative copies of the next layers, so the deeper the
// speculation the longer each layer's MoE would wait for its last experts.
cudaStream_t g_pfr_stream = nullptr;
cudaEvent_t g_sp_ev[DRE_PFS_MAX] = {};   // per slot: recorded on the spec stream after that slot's speculative copies
// ALPHA_DRE_PIPE_NOCOPY: diagnostic only (wrong output): skip the pipe's expert copies, to time the prompt as if PCIe were free
bool dre_pipe_nocopy() {
    static const bool v = dre_env_int("ALPHA_DRE_PIPE_NOCOPY", 0) != 0;
    return v;
}
bool dre_pipe_rstream() {
    static const bool v = dre_env_int("ALPHA_DRE_PIPE_RSTREAM", 1) != 0;
    return v;
}
void dre_pipe_streams() {
    if (!g_pfs_stream) CUDA_CHECK(cudaStreamCreateWithFlags(&g_pfs_stream, cudaStreamNonBlocking));
    if (dre_pipe_rstream() && !g_pfr_stream) {
        int lo = 0, hi = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
        CUDA_CHECK(cudaStreamCreateWithPriority(&g_pfr_stream, cudaStreamNonBlocking, hi));
        for (auto & e : g_sp_ev) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    }
}
int g_pfs_cur_tens = -1, g_pfs_cur_role = -1, g_pfs_cur_slot = -1;
long long g_pfs_bytes_h2d = 0, g_pfs_bytes_d2d = 0, g_pfs_issued = 0, g_pfs_waits = 0;

int g_dre_pf_ntok = 0;   // tokens of the prompt chunk being decoded (ggml_cuda_dre_set_prefill_tokens)
bool dre_pfs_on() {
    // ALPHA_DRE_PF_STREAM_MIN (default 2048): whole-tensor layer-ahead streaming only for chunks of at least this
    // many tokens. A short chunk routes to a fraction of the experts, and copying only those (the scheduler's
    // routed path) beats prefetching all of them.
    static const bool v = dre_env_int("ALPHA_DRE_PF_STREAM", 0) != 0;
    static const int mn = dre_env_int("ALPHA_DRE_PF_STREAM_MIN", dre_env_int("ALPHA_DRE_PF_PIPE", 1) != 0 ? (1 << 30) : 2048);   // the pipe takes every size when it is on
    return v && (g_dre_pf_ntok == 0 || g_dre_pf_ntok >= mn);
}

// Stream every expert of tensor ti into a buffer slot of its role (not `avoid`); returns the slot.
int dre_pfs_issue(int ti, int avoid, cudaStream_t compute_st) {
    const dre_tensor & d = g_dre.tens[ti];
    const dre_layer & L = g_dre.layers[d.layer];
    const int r = d.role;
    const int nb = dre_pfs_depth() + 1;
    for (int k = 0; k < nb; k++) if (g_pfs[r][k].tens == ti) return k;
    const int k = nb == 2 ? (avoid == 0 ? 1 : 0) : (L.blk % nb);
    dre_pfs_buf & b = g_pfs[r][k];
    const size_t need = d.esize * (size_t) L.n_exp + 512;
    const size_t need_alloc = std::max(need, dre_role_max_need(r));   // every layer fits: no re-allocation mid-prompt
    if (!g_pfs_stream) CUDA_CHECK(cudaStreamCreateWithFlags(&g_pfs_stream, cudaStreamNonBlocking));
    if (!b.ready) { CUDA_CHECK(cudaEventCreateWithFlags(&b.ready, cudaEventDisableTiming)); CUDA_CHECK(cudaEventCreateWithFlags(&b.free, cudaEventDisableTiming)); }
    // whatever last read this buffer has been enqueued on the compute stream before now
    CUDA_CHECK(cudaEventRecord(b.free, compute_st));
    CUDA_CHECK(cudaStreamWaitEvent(g_pfs_stream, b.free, 0));
    if (b.cap < need) {
        CUDA_CHECK(cudaStreamSynchronize(g_pfs_stream));
        if (b.p) CUDA_CHECK(cudaFree(b.p));
        b.p = nullptr; b.cap = 0;
        if (cudaMalloc(&b.p, need_alloc) != cudaSuccess) { (void) cudaGetLastError(); b.p = nullptr; b.tens = -1; return -1; }
        b.cap = need_alloc;
    }
    const char * host = d.host;
    // resident experts come VRAM->VRAM from their pool slot, the rest as coalesced H2D runs from the (arena) host copy
    int run0 = -1;
    auto flush_run = [&](int e_end) {
        if (run0 < 0) return;
        dre_src_runs(d, run0, e_end, [&](size_t off, const char * src, size_t n) {
            CUDA_CHECK(cudaMemcpyAsync(b.p + off, src, n, cudaMemcpyHostToDevice, g_pfs_stream));
        });
        g_pfs_bytes_h2d += (long long) (e_end - run0) * (long long) d.esize;
        run0 = -1;
    };
    (void) host;
    for (int e = 0; e < L.n_exp; e++) {
        const int sl = L.slot[e];
        if (sl >= 0 && L.complete()) {
            flush_run(e);
            CUDA_CHECK(cudaMemcpyAsync(b.p + (size_t) e * d.esize, g_dre.slots[sl].ptr + L.sub[r], d.esize, cudaMemcpyDeviceToDevice, g_pfs_stream));
            g_pfs_bytes_d2d += (long long) d.esize;
        } else if (run0 < 0) {
            run0 = e;
        }
    }
    flush_run(L.n_exp);
    CUDA_CHECK(cudaMemsetAsync(b.p + d.esize * (size_t) L.n_exp, 0, 512, g_pfs_stream));   // MMQ reads past the end
    CUDA_CHECK(cudaEventRecord(b.ready, g_pfs_stream));
    b.tens = ti;
    g_pfs_issued++;
    return k;
}

int dre_tensor_next_layer(int ti) {
    const dre_tensor & d = g_dre.tens[ti];
    auto it = g_dre.layer_by_blk.find(g_dre.layers[d.layer].blk + 1);
    if (it == g_dre.layer_by_blk.end()) return -1;
    const dre_layer & N = g_dre.layers[it->second];
    return N.complete() ? N.ti[d.role] : -1;
}
}   // namespace

void dre_pipe_release();
namespace { void dre_spec_join(); }
// release the streaming buffers (prefill is over; decode wants the VRAM back)
void dre_pfs_release() {
    dre_spec_join();
    if (g_pfr_stream) CUDA_CHECK(cudaStreamSynchronize(g_pfr_stream));
    if (g_pfs_stream) CUDA_CHECK(cudaStreamSynchronize(g_pfs_stream));
    for (auto & rr : g_pfs) for (auto & b : rr) { if (b.p) CUDA_CHECK(cudaFree(b.p)); b.p = nullptr; b.cap = 0; b.tens = -1; }
    if (g_pfs_issued) {
        fprintf(stderr, "[dre] prefill streaming: %lld tensors, %.1f GB H2D + %.1f GB from the pool, %lld waits\n",
                g_pfs_issued, g_pfs_bytes_h2d / 1e9, g_pfs_bytes_d2d / 1e9, g_pfs_waits);
    }
    g_pfs_issued = g_pfs_bytes_h2d = g_pfs_bytes_d2d = g_pfs_waits = 0;
    g_pfs_cur_tens = -1;
    dre_pipe_release();
}

// ALPHA_DRE_PF_PIPE: routed prefill streaming, pipelined with the expert GEMMs (chunks below
// ALPHA_DRE_PF_STREAM_MIN, where whole-tensor streaming copies experts nobody routes to). The scheduler's per-expert
// copy calls only record which experts the layer routes to; the first flush of a layer issues the gate, up and down
// copies of exactly those experts on the streaming stream into a full-layout buffer per role (resident experts
// VRAM->VRAM from their pool slot), recording an event every ALPHA_DRE_PIPE_GROUPS-th of the expert range. The MMQ
// (mmq.cu) then runs the expert GEMM one expert group at a time behind those events, so PCIe never waits for
// compute and compute starts on the first group instead of after the last copy.
namespace {
constexpr int DRE_PIPE_MAXG = 32;
struct dre_pipe_t {
    const void * buf = nullptr;
    int ng = 0;
    int e_end[DRE_PIPE_MAXG];
    cudaEvent_t ev[DRE_PIPE_MAXG] = {};
};
dre_pipe_t g_pipe[3][DRE_PFS_MAX];
std::vector<uint8_t> g_pp_used;       // experts the collecting layer routes to
int g_pp_collect_li = -1;             // layer whose copy calls are being collected
int g_pp_issued_li  = -1;             // layer whose copies are in flight
int g_pp_slot = 0, g_pp_cur_role = -1;
cudaEvent_t g_pp_free = nullptr;
long long g_pp_bytes_h2d = 0, g_pp_bytes_d2d = 0, g_pp_layers = 0;
long long g_pp_issue_us = 0;
// ALPHA_DRE_PIPE_SPEC=N: while layer L computes, the next layer's routing is unknown and the copy engine sits idle
// (attention, GDN, the router, the host sync). Fill that gap with layer L+1's N hottest non-resident experts (by the
// prompts' past usage, then the decode LFU score), copied into L+1's slot; its issue then copies only the rest.
// ALPHA_DRE_PIPE_SLOTS=n (default 2, max 4): staging slots; with n > 2 the speculative copies run n-1 layers ahead, so
// the copy engine never waits for a router.
std::vector<uint8_t> g_sp_mask[DRE_PFS_MAX];   // per slot: the experts already copied there for layer g_sp_li[slot]
int g_sp_li[DRE_PFS_MAX] = { -1, -1, -1, -1 };
int g_sp_last[DRE_PFS_MAX] = { 0, 0, 0, 0 };
long long g_pp_spec_n = 0, g_pp_spec_hit = 0;
int g_pp_spec_cur = -1;   // ALPHA_DRE_PIPE_SPEC_ADAPT: the current speculative count
int dre_pipe_slots() {
    static const int v = std::max(2, std::min(DRE_PFS_MAX, dre_env_int("ALPHA_DRE_PIPE_SLOTS", 2)));
    return v;
}

bool dre_pipe_wanted() {
    static const bool v = dre_env_int("ALPHA_DRE_PF_PIPE", 1) != 0;   // default on (beats whole-tensor streaming at every size)
    return v && g_dre_pf_ntok > DRE_MAX_TOK && !dre_pfs_on();
}
bool dre_pipe_on() { return dre_pipe_wanted() && g_dre.borrowed > 0; }   // the buffers come out of a borrow

// ALPHA_DRE_PIPE_GATHER (default 1): one copy kernel per expert group instead of a cudaMemcpyAsync per run. A thousand
// or more copy calls per layer keep the host busy for milliseconds after each layer's routing sync, with the GPU and
// PCIe both idle. The kernel reads pinned host memory (or a pool slot) through UVA at full PCIe rate.
struct dre_gjob { const char * src; long long dst_off; };
static __global__ void dre_gather_k(const dre_gjob * __restrict__ jobs, const int nj, char * __restrict__ dst, const long long n16) {
    const long long total = (long long) nj * n16;
    const long long stride = (long long) gridDim.x * blockDim.x;
    for (long long base = (long long) blockIdx.x * blockDim.x + threadIdx.x; base < total; base += 4 * stride) {
        uint4 v[4]; uint4 * d[4]; int c = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const long long i = base + j * stride;
            if (i >= total) break;
            const int jb = (int) (i / n16);
            const long long o = i - (long long) jb * n16;
            const dre_gjob jj = jobs[jb];
            v[j] = ((const uint4 *) jj.src)[o];
            d[j] = (uint4 *) (dst + jj.dst_off) + o;
            c = j + 1;
        }
        for (int j = 0; j < c; j++) *d[j] = v[j];
    }
}
constexpr int DRE_GJ_SLOTS = 4;
dre_gjob * g_gj_h[DRE_GJ_SLOTS] = {};   // mapped pinned job lists, one per issue slot
dre_gjob * g_gj_d[DRE_GJ_SLOTS] = {};
cudaEvent_t g_gj_done[DRE_GJ_SLOTS] = {};
int g_gj_slot = 0, g_gj_slot_last = 0;

// ALPHA_DRE_PIPE_SPEC_THREAD (default 0): the speculative copies are cudaMemcpyAsync runs issued by a worker thread, so
// the many copy calls per layer do not delay the next graph launch (a copy kernel instead would hold SMs for the whole
// PCIe transfer). The stream order is kept by joining the worker before anything else is enqueued on the copy stream.
struct dre_cpy { char * dst; const char * src; size_t n; };
struct dre_spec_worker {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<dre_cpy> jobs;
    bool busy = false, quit = false;
    std::thread th;
    void run() {
        CUDA_CHECK(cudaSetDevice(0));
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return busy || quit; });
            if (quit) return;
            lk.unlock();   // never hold the lock across the copy calls (the issuing thread would block on it)
            for (const dre_cpy & c : jobs) {
                CUDA_CHECK(cudaMemcpyAsync(c.dst, c.src, c.n, cudaMemcpyHostToDevice, g_pfs_stream));
            }
            lk.lock();
            busy = false;
            cv.notify_all();
        }
    }
    void join() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    void submit(std::vector<dre_cpy> && j) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
        jobs = std::move(j);
        busy = true;
        if (!th.joinable()) {
            int dev = 0;
            CUDA_CHECK(cudaGetDevice(&dev));
            (void) dev;
            th = std::thread([this] { run(); });
            th.detach();
        }
        cv.notify_all();
    }
};
dre_spec_worker * g_spec_w = nullptr;
void dre_spec_join() { if (g_spec_w) g_spec_w->join(); }

bool dre_pfc_issue(int li, cudaStream_t compute_st, const uint8_t * spec);   // dre_pfcpu.inc

// layer nli's n_spec hottest non-resident experts into slot kn. defer == nullptr: enqueued now on the spec stream and
// the slot's event recorded (the routed stream waits on it); otherwise the copy runs are appended to *defer
// ALPHA_DRE_PIPE_SPEC_GBLOCKS=n (default 0 = cudaMemcpyAsync runs; n > 0 is slower since the kernel holds SMs): the
// speculative copies are one gather kernel per tensor with n blocks. Thousands of queued cudaMemcpyAsync runs fill the
// stream's queue, so every further call blocks the issuing thread for about one copy's duration and the host can no
// longer run ahead of the GPU. A few blocks keep PCIe busy and leave the SMs free.
dre_gjob * g_sj_h[DRE_PFS_MAX] = {}, * g_sj_d[DRE_PFS_MAX] = {};   // mapped job lists, one per slot (3 roles each)
// ALPHA_DRE_NVTX (diagnostic): NVTX ranges per layer issue / speculative copy (nsys -t nvtx)
struct dre_nvtx {
    bool on;
    dre_nvtx(const char * what, int li) : on(false) {
#ifdef DRE_HAS_NVTX
        static const bool en = dre_env_int("ALPHA_DRE_NVTX", 0) != 0;
        if (en) { char b[64]; snprintf(b, sizeof(b), "%s %d", what, li); nvtxRangePushA(b); on = true; }
#else
        (void) what; (void) li;
#endif
    }
    ~dre_nvtx() {
#ifdef DRE_HAS_NVTX
        if (on) nvtxRangePop();
#endif
    }
};
void dre_spec_one(int nli, int kn, int n_spec, std::vector<dre_cpy> * defer) {
    dre_nvtx nv_("spec", nli);
    static const int sgb = dre_env_int("ALPHA_DRE_PIPE_SPEC_GBLOCKS", 0);
    std::vector<dre_cpy> local;
    std::vector<dre_cpy> & spec_jobs = defer ? *defer : local;
    const dre_layer & N = g_dre.layers[nli];
    std::vector<std::pair<float,int>> rank;
    for (int e = 0; e < N.n_exp; e++) {
        if (N.slot[e] >= 0) continue;
        rank.push_back({ (float) N.pf_count[e] + dre_score(N, e), e });
    }
    if (g_pp_spec_cur < 0) g_pp_spec_cur = n_spec;
    const int ns = std::min<int>(g_pp_spec_cur, (int) rank.size());
    std::partial_sort(rank.begin(), rank.begin() + ns, rank.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    std::vector<int> ids(ns);
    for (int i = 0; i < ns; i++) ids[i] = rank[i].second;
    std::sort(ids.begin(), ids.end());
    for (int r = 0; r < 3; r++) {
        const dre_tensor & d = g_dre.tens[N.ti[r]];
        dre_pfs_buf & b = g_pfs[r][kn];
        const size_t need = d.esize * (size_t) N.n_exp + 512;
        const size_t need_alloc = std::max(need, dre_role_max_need(r));   // every layer fits: no re-allocation mid-prompt
        if (b.cap < need) {
            CUDA_CHECK(cudaStreamSynchronize(g_pfs_stream));
            if (b.p) CUDA_CHECK(cudaFree(b.p));
            CUDA_CHECK(cudaMalloc(&b.p, need_alloc));
            CUDA_CHECK(cudaMemset(b.p, 0, need_alloc));
            b.cap = need_alloc;
        }
        const char * host = dre_pipe_host(d, dre_host_ptr(d.t->data));
        if (!defer && sgb > 0 && (!d.hp || d.contig) && !(d.esize & 15) && !((uintptr_t) host & 15) && N.n_exp <= 4096) {
            if (!g_sj_h[kn]) {
                CUDA_CHECK(cudaHostAlloc(&g_sj_h[kn], sizeof(dre_gjob) * 3 * 4096, cudaHostAllocMapped));
                CUDA_CHECK(cudaHostGetDevicePointer((void **) &g_sj_d[kn], g_sj_h[kn], 0));
            } else if (r == 0) {
                CUDA_CHECK(cudaEventSynchronize(g_sp_ev[kn]));   // this slot's previous list was read long ago
            }
            dre_gjob * jh = g_sj_h[kn] + (size_t) r * 4096;
            for (int i = 0; i < ns; i++) jh[i] = { host + (size_t) ids[i] * d.esize, (long long) ((size_t) ids[i] * d.esize) };
            if (ns > 0 && !dre_pipe_nocopy()) {
                dre_gather_k<<<sgb, 256, 0, g_pfs_stream>>>(g_sj_d[kn] + (size_t) r * 4096, ns, b.p, (long long) (d.esize / 16));
            }
            g_pp_bytes_h2d += (long long) d.esize * ns;
            continue;
        }
        // contiguous runs of the chosen experts: one copy per run
        for (int i = 0; i < ns; ) {
            int j = i + 1;
            while (j < ns && ids[j] == ids[j - 1] + 1) j++;
            dre_src_runs(d, ids[i], ids[j - 1] + 1, [&](size_t off, const char * src, size_t n) {
                spec_jobs.push_back({ b.p + off, src, n });
            });
            i = j;
        }
        g_pp_bytes_h2d += (long long) d.esize * ns;
    }
    g_sp_mask[kn].assign((size_t) N.n_exp, 0);
    for (int i = 0; i < ns; i++) g_sp_mask[kn][(size_t) ids[i]] = 1;
    if (!defer) {
        if (!dre_pipe_nocopy()) for (const dre_cpy & c : spec_jobs) CUDA_CHECK(cudaMemcpyAsync(c.dst, c.src, c.n, cudaMemcpyHostToDevice, g_pfs_stream));
        spec_jobs.clear();
        CUDA_CHECK(cudaEventRecord(g_sp_ev[kn], g_pfs_stream));
    }
    g_sp_li[kn] = nli; g_sp_last[kn] = ns;
    g_pp_spec_n += ns;
}

void dre_pipe_issue(int li, cudaStream_t compute_st) {
    dre_nvtx nv_("issue", li);
    struct pp_tm { int64_t t0; ~pp_tm() { g_pp_issue_us += ggml_time_us() - t0; } } pp_tmr { ggml_time_us() };
    const dre_layer & L = g_dre.layers[li];
    static const int ng_env = std::max(1, std::min(DRE_PIPE_MAXG, dre_env_int("ALPHA_DRE_PIPE_GROUPS", 8)));
    if (!g_pfs_stream) CUDA_CHECK(cudaStreamCreateWithFlags(&g_pfs_stream, cudaStreamNonBlocking));
    if (!g_pp_free) CUDA_CHECK(cudaEventCreateWithFlags(&g_pp_free, cudaEventDisableTiming));
    dre_spec_join();   // the previous layer's speculative copies are enqueued first
    const int nsl = dre_pipe_slots();
    g_pp_slot = (g_pp_slot + 1) % nsl;
    const int k = g_pp_slot;
    const bool spec_here = g_sp_li[k] == li && (int) g_sp_mask[k].size() == L.n_exp;
    (void) dre_pfc_issue(li, compute_st, spec_here ? g_sp_mask[k].data() : nullptr);   // ALPHA_DRE_PF_CPU
    if (spec_here) {   // already in this slot: not copied again (and never sent to the CPU)
        int hits = 0;
        for (int e = 0; e < L.n_exp; e++) if (g_sp_mask[k][(size_t) e] && g_pp_used[(size_t) e]) { g_pp_used[(size_t) e] = 0; g_pp_spec_hit++; hits++; }
        // ALPHA_DRE_PIPE_SPEC_ADAPT (default 1): grow the speculative count while nearly all of it is routed (long
        // prompts route to nearly every expert), shrink it when much of it was wasted (short prompts)
        static const bool adapt = dre_env_int("ALPHA_DRE_PIPE_SPEC_ADAPT", 1) != 0;
        if (adapt && g_sp_last[k] > 0) {
            const double ratio = (double) hits / g_sp_last[k];
            // ALPHA_DRE_PIPE_SPEC_HI/LO (percent, default 98/95): the hit ratios that grow / shrink the count. A big
            // chunk keeps 95/85: its hit ratio sits near 94%, and shrinking the prefetch there would move the copies
            // behind the router.
            static const int big_tok = dre_env_int("ALPHA_DRE_PIPE_SPEC_BIG_TOK", 6144);
            const bool big = g_dre_pf_ntok >= big_tok;
            static const double hi_s = dre_env_int("ALPHA_DRE_PIPE_SPEC_HI", 98) / 100.0;
            static const double lo_s = dre_env_int("ALPHA_DRE_PIPE_SPEC_LO", 95) / 100.0;
            static const double hi_b = dre_env_int("ALPHA_DRE_PIPE_SPEC_HI_BIG", 95) / 100.0;
            static const double lo_b = dre_env_int("ALPHA_DRE_PIPE_SPEC_LO_BIG", 85) / 100.0;
            const double hi = big ? hi_b : hi_s, lo = big ? lo_b : lo_s;
            if (ratio > hi)      g_pp_spec_cur = std::min(L.n_exp, g_pp_spec_cur * 5 / 4 + 8);
            else if (ratio < lo) g_pp_spec_cur = std::max(16, g_pp_spec_cur * 4 / 5);
        }
    }
    g_sp_li[k] = -1;
    // whatever the compute stream has enqueued so far (the reads of this slot two layers ago included) comes first
    CUDA_CHECK(cudaEventRecord(g_pp_free, compute_st));
    CUDA_CHECK(cudaStreamWaitEvent(g_pfs_stream, g_pp_free, 0));
    const bool two = dre_pipe_rstream();
    dre_pipe_streams();
    cudaStream_t rs = two ? g_pfr_stream : g_pfs_stream;
    if (two) {
        CUDA_CHECK(cudaStreamWaitEvent(rs, g_pp_free, 0));
        // this slot's speculative copies land before the group events below (the MMQ waits only on those)
        if (spec_here) CUDA_CHECK(cudaStreamWaitEvent(rs, g_sp_ev[k], 0));
    }
    const int ne = L.n_exp;
    const int gs = (ne + ng_env - 1) / ng_env;
    static const bool gather_env = dre_env_int("ALPHA_DRE_PIPE_GATHER", 1) != 0;
    static const int gblocks = std::max(1, dre_env_int("ALPHA_DRE_PIPE_GBLOCKS", 48));
    bool gather = gather_env;
    for (int r = 0; r < 3 && gather; r++) {
        const dre_tensor & d = g_dre.tens[L.ti[r]];
        if ((d.hp && !d.contig) || (d.esize & 15) || ((uintptr_t) dre_pipe_host(d, dre_host_ptr(d.t->data)) & 15) || (L.sub[r] & 15)) gather = false;
    }
    dre_gjob * jh = nullptr, * jd = nullptr;
    if (gather) {
        const int js = g_gj_slot; g_gj_slot = (g_gj_slot + 1) % DRE_GJ_SLOTS;
        if (!g_gj_h[js]) {
            CUDA_CHECK(cudaHostAlloc(&g_gj_h[js], sizeof(dre_gjob) * 6 * 4096, cudaHostAllocMapped));   // 3 routed + 3 spec
            CUDA_CHECK(cudaHostGetDevicePointer((void **) &g_gj_d[js], g_gj_h[js], 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&g_gj_done[js], cudaEventDisableTiming));
        } else {
            CUDA_CHECK(cudaEventSynchronize(g_gj_done[js]));   // the kernels that read this list four issues ago
        }
        GGML_ASSERT(ne <= 4096);
        jh = g_gj_h[js]; jd = g_gj_d[js];
        g_gj_slot_last = js;
    }
    for (int r = 0; r < 3; r++) {
        const dre_tensor & d = g_dre.tens[L.ti[r]];
        dre_pfs_buf & b = g_pfs[r][k];
        const size_t need = d.esize * (size_t) ne + 512;
        const size_t need_alloc = std::max(need, dre_role_max_need(r));   // every layer fits: no re-allocation mid-prompt
        if (b.cap < need) {
            CUDA_CHECK(cudaStreamSynchronize(g_pfs_stream));
            if (two) CUDA_CHECK(cudaStreamSynchronize(rs));
            if (b.p) CUDA_CHECK(cudaFree(b.p));
            CUDA_CHECK(cudaMalloc(&b.p, need_alloc));
            CUDA_CHECK(cudaMemset(b.p, 0, need_alloc));   // the MMQ reads past an expert's end: never NaN garbage
            b.cap = need_alloc;
        }
        b.tens = -1;   // not a whole-tensor stream
        dre_pipe_t & P = g_pipe[r][k];
        P.buf = b.p;
        const char * host = dre_pipe_host(d, dre_host_ptr(d.t->data));
        int g = 0;
        if (gather) {
            dre_gjob * rj_h = jh + (size_t) r * 4096, * rj_d = jd + (size_t) r * 4096;
            int nj_all = 0;
            for (int e0 = 0; e0 < ne; e0 += gs, g++) {
                const int e1 = std::min(ne, e0 + gs);
                const int j0 = nj_all;
                // ALPHA_DRE_PIPE_H2D_CE (default 1, with the routed stream): this layer's host experts go through the copy
                // engine on the spec stream, ahead of the next speculative copies. Read by the gather kernel they would
                // share PCIe with those copies and each layer's MoE would wait for a whole future-layer transfer.
                static const bool h2d_ce = dre_env_int("ALPHA_DRE_PIPE_H2D_CE", 1) != 0;
                const bool ce = two && h2d_ce;
                int run0 = -1, n_h = 0;
                auto flush_h = [&](int e_end) {
                    if (run0 < 0) return;
                    const size_t off = (size_t) run0 * d.esize, n = (size_t) (e_end - run0) * d.esize;
                    if (!dre_pipe_nocopy()) CUDA_CHECK(cudaMemcpyAsync(b.p + off, host + off, n, cudaMemcpyHostToDevice, g_pfs_stream));
                    run0 = -1; n_h++;
                };
                for (int e = e0; e < e1; e++) {
                    if (!g_pp_used[e]) { if (ce) flush_h(e); continue; }
                    const int sl = L.slot[e];
                    if (ce && sl < 0) {
                        if (run0 < 0) run0 = e;
                        g_pp_bytes_h2d += (long long) d.esize;
                        continue;
                    }
                    if (ce) flush_h(e);
                    const char * src = sl >= 0 ? g_dre.slots[sl].ptr + L.sub[r] : host + (size_t) e * d.esize;
                    rj_h[nj_all++] = { src, (long long) ((size_t) e * d.esize) };
                    if (sl >= 0) g_pp_bytes_d2d += (long long) d.esize; else g_pp_bytes_h2d += (long long) d.esize;
                }
                if (ce) flush_h(e1);
                if (n_h > 0) {
                    static cudaEvent_t hev[3][DRE_PIPE_MAXG] = {};
                    if (!hev[r][g]) CUDA_CHECK(cudaEventCreateWithFlags(&hev[r][g], cudaEventDisableTiming));
                    CUDA_CHECK(cudaEventRecord(hev[r][g], g_pfs_stream));
                    CUDA_CHECK(cudaStreamWaitEvent(rs, hev[r][g], 0));
                }
                if (nj_all > j0 && !dre_pipe_nocopy()) {
                    dre_gather_k<<<gblocks, 256, 0, rs>>>(rj_d + j0, nj_all - j0, b.p, (long long) (d.esize / 16));
                }
                if (!P.ev[g]) CUDA_CHECK(cudaEventCreateWithFlags(&P.ev[g], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(P.ev[g], rs));
                P.e_end[g] = e1;
            }
            P.ng = g;
            continue;
        }
        for (int e0 = 0; e0 < ne; e0 += gs, g++) {
            const int e1 = std::min(ne, e0 + gs);
            int run0 = -1;
            auto flush_run = [&](int e_end) {
                if (run0 < 0) return;
                dre_src_runs(d, run0, e_end, [&](size_t off, const char * src, size_t n) {
                    CUDA_CHECK(cudaMemcpyAsync(b.p + off, src, n, cudaMemcpyHostToDevice, rs));
                });
                g_pp_bytes_h2d += (long long) (e_end - run0) * (long long) d.esize;
                run0 = -1;
            };
            for (int e = e0; e < e1; e++) {
                const int sl = L.slot[e];
                if (!g_pp_used[e]) { flush_run(e); continue; }
                if (sl >= 0) {
                    flush_run(e);
                    CUDA_CHECK(cudaMemcpyAsync(b.p + (size_t) e * d.esize, g_dre.slots[sl].ptr + L.sub[r], d.esize,
                                               cudaMemcpyDeviceToDevice, rs));
                    g_pp_bytes_d2d += (long long) d.esize;
                } else if (run0 < 0) {
                    run0 = e;
                }
            }
            flush_run(e1);
            if (!P.ev[g]) CUDA_CHECK(cudaEventCreateWithFlags(&P.ev[g], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(P.ev[g], rs));
            P.e_end[g] = e1;
        }
        P.ng = g;
    }
    g_pp_issued_li = li;
    g_pp_layers++;

    static const int n_spec = dre_env_int("ALPHA_DRE_PIPE_SPEC", 512);
    std::vector<dre_cpy> spec_jobs;
    // layer nli's hottest non-resident experts into slot kn (last read by a layer whose MoE precedes this router)
    auto issue_spec = [&](int nli, int kn) { dre_spec_one(nli, kn, n_spec, two ? nullptr : &spec_jobs); };
    if (n_spec > 0) {
        int ti = L.ti[0];
        for (int dd = 1; dd < nsl; dd++) {
            ti = dre_tensor_next_layer(ti);
            if (ti < 0) break;
            const int nli = g_dre.tens[ti].layer;
            const int kn = (k + dd) % nsl;
            if (g_sp_li[kn] == nli) continue;   // issued by an earlier layer
            issue_spec(nli, kn);
        }
        if (!spec_jobs.empty()) {
            static const bool spec_thread = dre_env_int("ALPHA_DRE_PIPE_SPEC_THREAD", 0) != 0;   // off: the worker is starved by the spinning CPU-expert threads
            if (spec_thread) {
                if (!g_spec_w) g_spec_w = new dre_spec_worker();
                g_spec_w->submit(std::move(spec_jobs));
            } else {
                for (const dre_cpy & c : spec_jobs) CUDA_CHECK(cudaMemcpyAsync(c.dst, c.src, c.n, cudaMemcpyHostToDevice, g_pfs_stream));
            }
        }
    }
    if (gather) CUDA_CHECK(cudaEventRecord(g_gj_done[g_gj_slot_last], rs));   // (the routed stream reads the lists)
}
}   // namespace

// ALPHA_DRE_PIPE_PRESTART (default 1): called once the prompt's micro-batch is reserved: the first MoE layers'
// speculative copies start now, behind the embedding, instead of at the first layer's router
GGML_BACKEND_API_C void ggml_cuda_dre_pipe_prestart(void) {
    static const bool on = dre_env_int("ALPHA_DRE_PIPE_PRESTART", 1) != 0;
    static const int n_spec = dre_env_int("ALPHA_DRE_PIPE_SPEC", 512);
    // (also at the start of every later chunk of the prompt: the previous chunk's graph is complete by then, so its
    // slots are free, and the first layers' copies overlap the host's input preparation for the chunk)
    if (!on || n_spec <= 0 || !dre_pipe_on() || !dre_pipe_rstream()) return;
    dre_pipe_streams();
    const int nsl = dre_pipe_slots();
    int li = -1;
    for (int i = 0; i < (int) g_dre.layers.size(); i++) if (g_dre.layers[i].complete()) { li = i; break; }
    if (li < 0) return;
    int ti = g_dre.layers[li].ti[0];
    for (int dd = 1; dd < nsl && ti >= 0; dd++) {   // slot g_pp_slot+dd is the one the dd-th issue toggles to
        const int nli = g_dre.tens[ti].layer;
        const int kn = (g_pp_slot + dd) % nsl;
        if (g_sp_li[kn] != nli) dre_spec_one(nli, kn, n_spec, nullptr);
        ti = dre_tensor_next_layer(ti);
    }
}

void dre_pfc_report();   // dre_pfcpu.inc
void dre_pipe_release() {
    dre_spec_join();
    dre_pfc_report();
    for (auto & rr : g_pipe) for (auto & P : rr) { P.buf = nullptr; P.ng = 0; }
    if (g_pp_spec_n) fprintf(stderr, "[dre] prefill pipe spec: %lld experts prefetched, %lld of them routed (last count %d)\n", g_pp_spec_n, g_pp_spec_hit, g_pp_spec_cur);
    g_pp_spec_n = g_pp_spec_hit = 0; g_pp_spec_cur = -1;
    for (int q = 0; q < DRE_PFS_MAX; q++) { g_sp_li[q] = -1; g_sp_last[q] = 0; }
    g_pp_collect_li = g_pp_issued_li = -1; g_pp_cur_role = -1;
    if (g_pp_layers) {
        fprintf(stderr, "[dre] prefill pipe: %lld layer issues, %.1f GB H2D + %.1f GB from the pool, issue host %lld us\n", g_pp_layers,
                g_pp_bytes_h2d / 1e9, g_pp_bytes_d2d / 1e9, g_pp_issue_us);
    }
    g_pp_layers = g_pp_bytes_h2d = g_pp_bytes_d2d = g_pp_issue_us = 0;
}

// ALPHA_DRE_PF_PIPE: for mmq.cu, the expert groups of a pipelined staging buffer (0 = not one)
int ggml_cuda_dre_pipe_groups(const void * buf, const int ** e_end, cudaEvent_t const ** ev) {
    if (!buf) return 0;
    for (int r = 0; r < 3; r++) for (int k = 0; k < DRE_PFS_MAX; k++) {
        const dre_pipe_t & P = g_pipe[r][k];
        if (P.buf == buf && P.ng > 0) { *e_end = P.e_end; *ev = P.ev; return P.ng; }
    }
    return 0;
}

// ALPHA_DRE_IDS_HIST (diagnostic): tokens-per-expert histogram of prompt chunks (printed at the pool return)
static long long g_ih[9] = {0}, g_ih_tok[9] = {0};
GGML_BACKEND_API_C void ggml_cuda_dre_prefill_ids(const int32_t * ids, int ne0, int ne1, int s0, int s1) {
    static const bool on = dre_env_int("ALPHA_DRE_IDS_HIST", 0) != 0;
    if (!on || ne1 <= DRE_MAX_TOK) return;
    static std::vector<int> cnt;
    cnt.assign(4096, 0);
    int mx = 0;
    for (int t = 0; t < ne1; t++) for (int j = 0; j < ne0; j++) { const int e = ids[t * s1 + j * s0]; if (e >= 0 && e < 4096) { cnt[e]++; mx = std::max(mx, e + 1); } }
    for (int e = 0; e < mx; e++) {
        const int c = cnt[e];
        const int b = c == 0 ? 0 : c == 1 ? 1 : c <= 2 ? 2 : c <= 4 ? 3 : c <= 8 ? 4 : c <= 16 ? 5 : c <= 32 ? 6 : c <= 64 ? 7 : 8;
        g_ih[b]++; g_ih_tok[b] += c;
    }
}
static void dre_ids_hist_print() {
    long long n = 0; for (long long v : g_ih) n += v;
    if (!n) return;
    static const char * lab[9] = {"0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"};
    fprintf(stderr, "[dre] tokens/expert (expert-tensors, %% / token share):");
    long long tt = 0; for (long long v : g_ih_tok) tt += v;
    for (int b = 0; b < 9; b++) fprintf(stderr, " %s:%.1f%%/%.1f%%", lab[b], 100.0 * g_ih[b] / n, tt ? 100.0 * g_ih_tok[b] / tt : 0.0);
    fprintf(stderr, "\n");
    for (int b = 0; b < 9; b++) g_ih[b] = g_ih_tok[b] = 0;
}

GGML_BACKEND_API_C void ggml_cuda_dre_set_prefill_tokens(int n) {
    g_dre_pf_ntok = n;
    g_pp_collect_li = -1;
}

GGML_BACKEND_API_C void ggml_cuda_dre_prefill_flush(void * backend_ptr) {
    if (dre_pipe_on() && g_pp_collect_li >= 0) {
        // the layer's routed set is complete at its first flush: issue gate, up and down; later flushes of the
        // same layer find it in flight. The compute stream does not wait here; the MMQ waits per expert group.
        if (g_pp_issued_li != g_pp_collect_li) {
            ggml_backend_t backend = (ggml_backend_t) backend_ptr;
            dre_pipe_issue(g_pp_collect_li, ((ggml_backend_cuda_context *) backend->context)->stream());
        }
        return;
    }
    if (dre_pfs_on() && g_pfs_cur_tens >= 0 && g_pfs_cur_slot >= 0) {
        ggml_backend_t backend = (ggml_backend_t) backend_ptr;
        cudaStream_t st = ((ggml_backend_cuda_context *) backend->context)->stream();
        CUDA_CHECK(cudaStreamWaitEvent(st, g_pfs[g_pfs_cur_role][g_pfs_cur_slot].ready, 0));
        g_pfs_waits++;
        return;
    }
    GGML_UNUSED(backend_ptr);
    dre_pf_flush();
}

// ALPHA_DRE_PF_STREAM: tells the scheduler a prefill expert tensor is streamed whole (no routing-ids read needed)
GGML_BACKEND_API_C bool ggml_cuda_dre_prefill_stream_whole(void * backend_ptr, const void * cpu_src, int n_tokens) {
    GGML_UNUSED(backend_ptr);
    static const bool on = dre_env_int("ALPHA_DRE_PF_NOSYNC", 0) != 0;   // opt-in, no measurable gain
    if (!on || !dre_pfs_on() || n_tokens <= DRE_MAX_TOK) return false;
    auto it = g_dre.by_data.find(cpu_src);
    return it != g_dre.by_data.end() && it->second >= 0;
}

// the scheduler's staging tensor for the current expert input is redirected to the streamed buffer
GGML_BACKEND_API_C void * ggml_cuda_dre_prefill_redirect(void * backend_ptr, void * gpu_dst) {
    GGML_UNUSED(backend_ptr); GGML_UNUSED(gpu_dst);
    if (dre_pipe_on() && g_pp_cur_role >= 0 && g_pp_issued_li == g_pp_collect_li) {
        void * p = g_pfs[g_pp_cur_role][g_pp_slot].p;
        g_pp_cur_role = -1;
        return p;
    }
    if (!dre_pfs_on() || g_pfs_cur_tens < 0 || g_pfs_cur_slot < 0) return nullptr;
    void * p = g_pfs[g_pfs_cur_role][g_pfs_cur_slot].p;
    g_pfs_cur_tens = -1;   // the next tensor starts fresh
    return p;
}

GGML_BACKEND_API_C bool ggml_cuda_dre_prefill_copy(void * backend_ptr, void * gpu_dst, const void * cpu_src,
                                                   size_t expert_size, int64_t n_expert, int64_t expert_id,
                                                   size_t copy_size) {
    GGML_UNUSED(n_expert);
    ggml_backend_t backend = (ggml_backend_t) backend_ptr;
    cudaStream_t st = ((ggml_backend_cuda_context *) backend->context)->stream();
    char *       dst = (char *) gpu_dst + (size_t) expert_id * expert_size;
    const char * src = dre_host_ptr(cpu_src) + (size_t) expert_id * expert_size;   // the arena copy when there is one
    {
        auto ts = g_dre.by_data.find(cpu_src);
        if (ts != g_dre.by_data.end() && ts->second >= 0) {
            const dre_tensor & dt = g_dre.tens[ts->second];
            if (dt.hp && expert_id >= 0 && expert_id < dt.t->ne[2]) src = dre_src(dt, expert_id);
        }
    }
    if (dre_pipe_on()) {
        auto itp = g_dre.by_data.find(cpu_src);
        if (itp != g_dre.by_data.end() && itp->second >= 0) {
            const dre_tensor & d = g_dre.tens[itp->second];
            dre_layer & Lp = g_dre.layers[d.layer];
            if (Lp.complete() && expert_id >= 0 && expert_id < Lp.n_exp) {
                if (g_pp_collect_li != d.layer) {   // a new layer: start its routed set
                    g_pp_collect_li = d.layer;
                    g_pp_used.assign((size_t) Lp.n_exp, 0);
                }
                g_pp_used[(size_t) expert_id] = 1;
                g_pp_cur_role = d.role;
                Lp.pf_count[expert_id]++; g_dre.pf_pending = true;
                return true;
            }
        }
    }
    if (dre_pfs_on()) {
        auto itp = g_dre.by_data.find(cpu_src);
        if (itp != g_dre.by_data.end() && itp->second >= 0) {
            const int ti = itp->second;
            dre_layer & Lp = g_dre.layers[g_dre.tens[ti].layer];
            if (expert_id >= 0 && expert_id < Lp.n_exp) { Lp.pf_count[expert_id]++; g_dre.pf_pending = true; }
            if (ti != g_pfs_cur_tens) {
                const int r = g_dre.tens[ti].role;
                int k = -1;
                for (int q = 0; q < dre_pfs_depth() + 1; q++) if (g_pfs[r][q].tens == ti) k = q;
                if (k < 0) k = dre_pfs_issue(ti, -1, st);           // not prefetched (first layer): stream it now
                if (k >= 0) {
                    g_pfs_cur_tens = ti; g_pfs_cur_role = r; g_pfs_cur_slot = k;
                    int nt = ti;                                     // the next d layers stream while this one computes
                    for (int a = 0; a < dre_pfs_depth(); a++) {
                        nt = dre_tensor_next_layer(nt);
                        if (nt < 0) break;
                        (void) dre_pfs_issue(nt, k, st);
                    }
                } else {
                    g_pfs_cur_tens = -1;
                }
            }
            if (g_pfs_cur_tens == ti) return true;
        }
    }
    auto host_copy = [&]() {
        if (g_pf_run.bytes && g_pf_run.tensor == cpu_src && g_pf_run.next_eid == expert_id && g_pf_run.st == st &&
            g_pf_run.src + g_pf_run.bytes == src) {
            g_pf_run.bytes += expert_size;            // extends the run; the padding moves to its new end
            g_pf_run.pad = copy_size - expert_size;
        } else {
            dre_pf_flush();
            g_pf_run.st = st; g_pf_run.dst = dst; g_pf_run.src = src;
            g_pf_run.bytes = expert_size; g_pf_run.pad = copy_size - expert_size;
            g_pf_run.tensor = cpu_src;
        }
        g_pf_run.next_eid = expert_id + 1;
    };
    auto it = g_dre.by_data.find(cpu_src);
    if (it != g_dre.by_data.end() && it->second >= 0) {
        const dre_tensor & d = g_dre.tens[it->second];
        dre_layer & L = g_dre.layers[d.layer];
        if (expert_id >= 0 && expert_id < L.n_exp) {
            L.pf_count[expert_id]++;
            g_dre.pf_pending = true;
            const int s = L.slot[expert_id];
            if (s >= 0 && L.complete()) {
                dre_pf_flush();
                CUDA_CHECK(cudaMemcpyAsync(dst, g_dre.slots[s].ptr + L.sub[d.role], expert_size,
                                           cudaMemcpyDeviceToDevice, st));
                if (copy_size > expert_size) {   // the MMQ padding the plain path copies from the next expert
                    const char * pad_src = !d.hp ? src + expert_size
                                                 : (expert_id + 1 < d.t->ne[2] ? dre_src(d, expert_id + 1) : nullptr);
                    if (pad_src) CUDA_CHECK(cudaMemcpyAsync(dst + expert_size, pad_src, copy_size - expert_size,
                                                            cudaMemcpyHostToDevice, st));
                }
                g_dre.n_pf_d2d++;
                return true;
            }
        }
    }
    host_copy();
    g_dre.n_pf_h2d++;
    return true;
}

// DMA pacer (ALPHA_DRE_PACER, default on). Queued all at once, admission copies (tens of MB per graph) would delay
// every graph's own input uploads, which share the copy engine and are served in submission order. The pacer
// thread feeds the same copies in small chunks with at most ALPHA_DRE_DMA_DEPTH chunks outstanding, so anything
// else submitted to the engine waits for at most one chunk.
struct dre_state::pacer_job {
    struct piece { char * dst; const char * src; size_t n; };
    std::vector<piece> pieces;
    cudaEvent_t        ev_ready = nullptr;   // compute stream: the unpublish upload this batch must follow
    cudaEvent_t        ev_done  = nullptr;   // dma stream: recorded after the last piece
    std::atomic<bool>  issued{false};
};
namespace {
struct dre_pacer_t {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::shared_ptr<dre_state::pacer_job>> q;
    std::atomic<int> pending{0};
    bool started = false;
    int  device = 0;
} g_pacer;

bool dre_pacer_on() {
    static const bool v = dre_env_int("ALPHA_DRE_PACER", 1) != 0;
    return v;
}

void dre_pacer_loop() {
    CUDA_CHECK(cudaSetDevice(g_pacer.device));
    const size_t chunk = (size_t) std::max(16, dre_env_int("ALPHA_DRE_DMA_CHUNK_KB", 256)) << 10;
    const int depth = std::max(1, std::min(8, dre_env_int("ALPHA_DRE_DMA_DEPTH", 2)));
    std::vector<cudaEvent_t> ring((size_t) depth);
    for (auto & e : ring) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    std::vector<bool> used((size_t) depth, false);
    int ri = 0;
    auto wait_ev = [](cudaEvent_t e) {
        for (;;) {
            const cudaError_t r = cudaEventQuery(e);
            if (r == cudaSuccess) return;
            if (r != cudaErrorNotReady) { CUDA_CHECK(r); return; }
            std::this_thread::sleep_for(std::chrono::microseconds(4));
        }
    };
    for (;;) {
        std::shared_ptr<dre_state::pacer_job> j;
        {
            std::unique_lock<std::mutex> lk(g_pacer.m);
            g_pacer.cv.wait(lk, [] { return !g_pacer.q.empty(); });
            j = g_pacer.q.front();
            g_pacer.q.pop_front();
        }
        CUDA_CHECK(cudaStreamWaitEvent(g_dre.dma, j->ev_ready, 0));
        for (const auto & pc : j->pieces) {
            for (size_t off = 0; off < pc.n; off += chunk) {
                if (used[(size_t) ri]) wait_ev(ring[(size_t) ri]);
                const size_t n = std::min(chunk, pc.n - off);
                dre_h2d_worker(pc.dst + off, pc.src + off, n, g_dre.dma);
                CUDA_CHECK(cudaEventRecord(ring[(size_t) ri], g_dre.dma));
                used[(size_t) ri] = true;
                ri = (ri + 1) % depth;
            }
        }
        CUDA_CHECK(cudaEventRecord(j->ev_done, g_dre.dma));
        j->issued.store(true, std::memory_order_release);
        g_pacer.pending.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void dre_pacer_submit(const std::shared_ptr<dre_state::pacer_job> & j) {
    if (!g_pacer.started) {
        g_pacer.started = true;
        CUDA_CHECK(cudaGetDevice(&g_pacer.device));
        std::thread(dre_pacer_loop).detach();
    }
    g_pacer.pending.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard<std::mutex> lk(g_pacer.m);
        g_pacer.q.push_back(j);
    }
    g_pacer.cv.notify_one();
}

// every queued batch has been handed to the engine (its ev_done recorded)
void dre_pacer_drain() {
    while (g_pacer.pending.load(std::memory_order_acquire) > 0) std::this_thread::sleep_for(std::chrono::microseconds(20));
}

bool dre_batch_done(const dre_state::batch & bt) {
    const cudaEvent_t e = bt.pj ? bt.pj->ev_done : bt.ev;
    if (bt.pj && !bt.pj->issued.load(std::memory_order_acquire)) return false;
    if (cudaEventQuery(e) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    return true;
}

void dre_batch_free(dre_state::batch & bt) {
    g_dre.ev_free.push_back(bt.ev);
    if (bt.pj) { g_dre.ev_free.push_back(bt.pj->ev_done); bt.pj.reset(); }
}
}

// ALPHA_DRE_RESTORE (default on): put back exactly the experts the borrow evicted, most recently used first, as one
// bulk copy before the first decode graph. Without it the pool comes back empty and decode crawls for tens of tokens
// while paced admission refills it; a short blocking copy here is far cheaper than that cold start.
static std::vector<std::pair<int,int>> g_dre_restore;
// ALPHA_DRE_TRACE: the routing trace (written in graph_begin) also marks each prefill borrow (u32 0xFFFFFFFF, u32 n,
// then n x (u16 layer, u16 expert) evicted) and return (u32 0xFFFFFFFE), so an offline replay sees the pool holes
static FILE * dre_trace_file() {
    static FILE * tf = [] { const char * e = getenv("ALPHA_DRE_TRACE"); return e ? fopen(e, "wb") : (FILE *) nullptr; }();
    return tf;
}
static int dre_take_free_slot(int ci) {
    dre_class & c = g_dre.classes[ci];
    if (!c.free.empty()) { int s = c.free.back(); c.free.pop_back(); return s; }
    for (int ch = 0; ch < (int) g_dre.chunks.size(); ch++) {
        auto & k = g_dre.chunks[ch];
        if (!k.alive || k.used + c.size > k.size) continue;
        dre_slot x; x.ptr = k.base + k.used; x.chunk = ch;
        k.used += c.size;
        g_dre.pool_used += c.size;
        g_dre.slots.push_back(x);
        return (int) g_dre.slots.size() - 1;
    }
    return -1;
}
// the restore's copies in flight on g_dre.dma (ggml_cuda_dre_return_begin issues them, _finish publishes the slots)
struct dre_rs_piece { char * dst; const char * src; size_t n; };
static struct { bool active = false; std::vector<int> done; size_t bytes = 0, n_want = 0;
                std::chrono::steady_clock::time_point t0; std::thread th;
                // copy queue fed by dre_restore_pass, drained by the worker (or issued inline without one)
                std::mutex mu; std::condition_variable cv; std::deque<dre_rs_piece> q; bool closed = true, threaded = false; } g_dre_rs_pend;
static void dre_restore_finish() {
    if (!g_dre_rs_pend.active) return;
    g_dre_rs_pend.active = false;
    const auto t1 = std::chrono::steady_clock::now();
    if (g_dre_rs_pend.th.joinable()) g_dre_rs_pend.th.join();
    CUDA_CHECK(cudaStreamSynchronize(g_dre.dma));
    for (int s : g_dre_rs_pend.done) {
        dre_slot & x = g_dre.slots[s];
        dre_layer & L = g_dre.layers[x.li];
        x.pending = false;
        L.slot[x.e] = s;
        dre_set_entries(L, x.e);
        lru_push_front(g_dre.classes[L.cls], s);
    }
    dre_upload_dirty(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t2 = std::chrono::steady_clock::now();
    fprintf(stderr, "[dre] restored %zu of %zu borrowed expert units (%.0f MB) in %.1f ms (%.1f ms waited at the finish)\n",
            g_dre_rs_pend.done.size(), g_dre_rs_pend.n_want, g_dre_rs_pend.bytes / 1048576.0,
            std::chrono::duration<double, std::milli>(t2 - g_dre_rs_pend.t0).count(),
            std::chrono::duration<double, std::milli>(t2 - t1).count());
    g_dre_rs_pend.done.clear();
}
static bool dre_restore_begin() {
    // ALPHA_DRE_RESTORE (default on, synchronous): with the return at the end of the prompt (serve-gpu-engine.cpp,
    // ALPHA_DRE_RETURN_AT_PREFILL) a sync restore puts the pre-prompt hot set back before the first token. The async
    // (paced) form blocks the decode's own admissions behind one big job and stays opt-in.
    static const bool on = dre_env_int("ALPHA_DRE_RESTORE", 1) != 0;
    if (!on || g_dre_restore.empty()) { g_dre_restore.clear(); return false; }
    const auto t0 = std::chrono::steady_clock::now();
    // hottest first: by the LFU score when LFU is on (the replacement policy decode actually uses), else by recency
    std::sort(g_dre_restore.begin(), g_dre_restore.end(), [](const std::pair<int,int> & a, const std::pair<int,int> & b) {
        if (dre_lfu_on()) return dre_score(g_dre.layers[a.first], a.second) > dre_score(g_dre.layers[b.first], b.second);
        return g_dre.layers[a.first].last_use[a.second] > g_dre.layers[b.first].last_use[b.second];
    });
    // ALPHA_DRE_RESTORE_PCT (default 100): restore only the hottest share of the borrowed units; the rest of the
    // returned slots stay free for the decode's own misses (the restore is prompt time)
    {
        static const int pct = std::max(0, std::min(100, dre_env_int("ALPHA_DRE_RESTORE_PCT", 100)));
        // ALPHA_DRE_RESTORE_PCT_SMALL (default 50) for a short prompt chunk, decided by the chunk's token count
        // (ALPHA_DRE_RESTORE_SMALL_TOK, default 6144). A byte threshold would also catch long prompts once the borrow
        // per token is small, and their decode would lose.
        static const int pct_small = std::max(0, std::min(100, dre_env_int("ALPHA_DRE_RESTORE_PCT_SMALL", 50)));
        static const int small_tok = dre_env_int("ALPHA_DRE_RESTORE_SMALL_TOK", 6144);
        const int p = (g_dre_pf_ntok > 0 && g_dre_pf_ntok <= small_tok) ? std::min(pct, pct_small) : pct;
        if (p < 100) g_dre_restore.resize(g_dre_restore.size() * (size_t) p / 100);
    }
    std::vector<int> done;
    size_t bytes = 0;
    // ALPHA_DRE_RESTORE_ASYNC (opt-in, needs the pacer): the restore is an ordinary paced admission batch, so the
    // first decode does not wait for it (it would count in time-to-first-token); the slots are published by the
    // decode loop once their copies land, like every other admission
    static const bool async_restore = dre_env_int("ALPHA_DRE_RESTORE_ASYNC", 0) != 0;   // default off: the return happens at the end of the prompt
    if (async_restore && dre_pacer_on()) {
        for (const auto & u : g_dre_restore) {
            dre_layer & L = g_dre.layers[u.first];
            if (L.slot[u.second] != -1) continue;
            const int s = dre_take_free_slot(L.cls);
            if (s < 0) continue;
            dre_slot & x = g_dre.slots[s];
            x.li = u.first; x.e = u.second; x.pending = true; x.pinned = L.pinned;
            L.slot[u.second] = -2;
            done.push_back(s);
        }
        if (!done.empty()) {
            cudaEvent_t ev;
            if (!g_dre.ev_free.empty()) { ev = g_dre.ev_free.back(); g_dre.ev_free.pop_back(); }
            else CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(ev, g_dre.dma));
            auto pj = std::make_shared<dre_state::pacer_job>();
            pj->ev_ready = ev;
            if (!g_dre.ev_free.empty()) { pj->ev_done = g_dre.ev_free.back(); g_dre.ev_free.pop_back(); }
            else CUDA_CHECK(cudaEventCreateWithFlags(&pj->ev_done, cudaEventDisableTiming));
            for (int s : done) {
                const dre_slot & x = g_dre.slots[s];
                const dre_layer & L = g_dre.layers[x.li];
                for (int r = 0; r < 3; r++) {
                    const dre_tensor & d = g_dre.tens[L.ti[r]];
                    pj->pieces.push_back({ x.ptr + L.sub[r], dre_src(d, x.e), d.esize });
                    bytes += d.esize;
                }
            }
            g_dre.batches.push_back({ ev, done, pj });
            dre_pacer_submit(pj);
        }
        fprintf(stderr, "[dre] restoring %zu of %zu borrowed expert units (%.0f MB) in the background\n", done.size(),
                g_dre_restore.size(), bytes / 1048576.0);
        g_dre_restore.clear();
        return false;
    }
    g_dre_rs_pend.done.clear();
    g_dre_rs_pend.bytes = 0;
    g_dre_rs_pend.t0 = t0;
    g_dre_rs_pend.q.clear();
    g_dre_rs_pend.closed = false;
    // ALPHA_DRE_RESTORE_THREAD (default 1): the copies are issued from a worker thread. Each cudaMemcpyAsync from
    // the arena blocks its caller for about the copy's duration, so issuing them inline would serialise the restore
    // with the host work after it (the chunk re-allocations, the scheduler reserve).
    static const bool use_thread = dre_env_int("ALPHA_DRE_RESTORE_THREAD", 1) != 0;
    g_dre_rs_pend.threaded = use_thread;
    if (use_thread) {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        g_dre_rs_pend.th = std::thread([dev]() {
            CUDA_CHECK(cudaSetDevice(dev));
            // ALPHA_DRE_RESTORE_STREAMS (default 1): copies round-robin over this many streams
            static const int nst = std::max(1, std::min(4, dre_env_int("ALPHA_DRE_RESTORE_STREAMS", 1)));
            static cudaStream_t extra[3] = {};
            for (int k = 0; k < nst - 1; k++) if (!extra[k]) CUDA_CHECK(cudaStreamCreateWithFlags(&extra[k], cudaStreamNonBlocking));
            auto & P = g_dre_rs_pend;
            size_t rr = 0;
            for (;;) {
                std::deque<dre_rs_piece> batch;
                {
                    std::unique_lock<std::mutex> lk(P.mu);
                    P.cv.wait(lk, [&] { return !P.q.empty() || P.closed; });
                    if (P.q.empty() && P.closed) break;
                    batch.swap(P.q);
                }
                for (const auto & q : batch) {
                    const int k = (int) (rr++ % (size_t) nst);
                    dre_h2d_worker(q.dst, q.src, q.n, k == 0 ? g_dre.dma : extra[k - 1]);
                }
            }
            for (int k = 0; k < nst - 1; k++) CUDA_CHECK(cudaStreamSynchronize(extra[k]));
            CUDA_CHECK(cudaStreamSynchronize(g_dre.dma));
        });
    }
    return true;
}
// one assignment pass: every not-yet-placed unit of the restore list that a free slot (or the room left in an alive
// chunk) now fits gets the slot and its copies queued; units that do not fit yet stay for the next pass
static void dre_restore_pass() {
    std::vector<dre_rs_piece> pieces;
    for (auto & u : g_dre_restore) {
        if (u.first < 0) continue;   // placed by an earlier pass
        dre_layer & L = g_dre.layers[u.first];
        if (L.slot[u.second] != -1) { u.first = -1; continue; }
        const int s = dre_take_free_slot(L.cls);
        if (s < 0) continue;   // this class is full; a smaller class may still fit, or a later chunk
        dre_slot & x = g_dre.slots[s];
        x.li = u.first; x.e = u.second; x.pending = true; x.pinned = L.pinned;
        L.slot[u.second] = -2;
        for (int r = 0; r < 3; r++) {
            const dre_tensor & d = g_dre.tens[L.ti[r]];
            pieces.push_back({ x.ptr + L.sub[r], dre_src(d, x.e), d.esize });
            g_dre_rs_pend.bytes += d.esize;
        }
        g_dre_rs_pend.done.push_back(s);
        u.first = -1;
    }
    if (pieces.empty()) return;
    if (g_dre_rs_pend.threaded) {
        { std::lock_guard<std::mutex> lk(g_dre_rs_pend.mu); g_dre_rs_pend.q.insert(g_dre_rs_pend.q.end(), pieces.begin(), pieces.end()); }
        g_dre_rs_pend.cv.notify_one();
    } else {
        for (const auto & q : pieces) CUDA_CHECK(cudaMemcpyAsync(q.dst, q.src, q.n, cudaMemcpyHostToDevice, g_dre.dma));
    }
}
static void dre_restore_end(bool defer) {
    g_dre_rs_pend.n_want = g_dre_restore.size();
    { std::lock_guard<std::mutex> lk(g_dre_rs_pend.mu); g_dre_rs_pend.closed = true; }
    g_dre_rs_pend.cv.notify_one();
    g_dre_rs_pend.active = true;
    g_dre_restore.clear();
    if (!defer) dre_restore_finish();
}
static void dre_restore_borrowed(bool defer) {
    if (!dre_restore_begin()) return;
    dre_restore_pass();
    dre_restore_end(defer);
}

// Prefill borrow-and-refill: free whole pool chunks so a long prompt can run in big micro-batches (every expert
// crosses PCIe once per micro-batch, so prefill speed is roughly linear in its size), then re-allocate them after the
// prompt and let warm admission refill them. Only between graphs.
GGML_BACKEND_API_C size_t ggml_cuda_dre_borrow(size_t want) {
    if (!ggml_cuda_dre_enabled() || g_dre.failed || g_dre.chunks.empty() || want == 0) return 0;
    // headroom on top of the compute buffer: the CUDA temp pools grow with the micro-batch and keep what they grew
    want += (size_t) dre_env_int("ALPHA_DRE_PREFILL_EXTRA_MB", 512) << 20;
    if (dre_pfs_on() || dre_pipe_wanted()) {
        size_t per_layer = 0;   // one buffer set (gate+up+down of a layer) per slot: depth+1 slots
        for (const auto & L : g_dre.layers) {
            if (!L.complete()) continue;
            size_t b = 0;
            for (int r = 0; r < 3; r++) b += g_dre.tens[L.ti[r]].esize * (size_t) L.n_exp + 512;
            per_layer = std::max(per_layer, b);
        }
        // ALPHA_DRE_PF_STREAM_MB overrides. With ALPHA_DRE_PF_EARLY the buffers exist before the scheduler reserves, so the
        // borrow must cover them; otherwise 768 MB plus the slack of the extra headroom has always been enough
        const char * mb = getenv("ALPHA_DRE_PF_STREAM_MB");
        static const bool early = dre_env_int("ALPHA_DRE_PF_EARLY", 0) != 0;
        size_t stage = mb ? (size_t) atoll(mb) << 20
                          : early ? per_layer * (size_t) (dre_pfs_depth() + 1)                         // allocated before the reserve
                                  : ((size_t) 768 << 20) + per_layer * (size_t) (dre_pfs_depth() - 1);   // late: slack covers the rest
        if (dre_pipe_wanted() && !mb) stage += per_layer * (size_t) (dre_pipe_slots() - 2);   // ALPHA_DRE_PIPE_SLOTS
        // every staging buffer set the prompt allocates (each sized for the largest layer): 768 MB fits IQ2-class
        // layers, but a 3-bit file's layer set is larger and its prompt staging would hit OOM in dre_spec_one. The
        // borrow only takes what is not already free, so the exact figure costs nothing where VRAM is spare.
        if (!mb) stage = std::max(stage, per_layer * (size_t) std::max(dre_pfs_depth() + 1, dre_pipe_wanted() ? dre_pipe_slots() : 0));
        want += stage;
    }
    {
        size_t free_mem = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total));
        want = want > free_mem ? want - free_mem : 0;   // only what is not already free
        if (want == 0) return 1;
    }
    static const bool bt_on = dre_env_int("ALPHA_DRE_RETURN_TIMERS", 0) != 0;
    const int64_t bt0 = ggml_time_us();
    dre_pacer_drain();                     // every paced admission is on the engine...
    CUDA_CHECK(cudaDeviceSynchronize());   // ...and done; no kernel reads a table or a slot
    const int64_t bt1 = ggml_time_us();
    // land every in-flight admission first, so each slot is either resident or free
    for (auto & bt : g_dre.batches) {
        for (int sl : bt.slots) {
            dre_slot & x = g_dre.slots[sl];
            if (x.li < 0) continue;
            dre_layer & L = g_dre.layers[x.li];
            x.pending = false;
            L.slot[x.e] = sl;
            dre_set_entries(L, x.e);
            lru_push_front(g_dre.classes[L.cls], sl);
        }
        dre_batch_free(bt);
    }
    g_dre.batches.clear();
    // ALPHA_DRE_BORROW_MIGRATE (default on): the borrow frees whole chunks, i.e. evicts whatever happens to live in
    // them (a large share of the pool for a long prompt), and decode would then restart on a gutted pool while the
    // restore refills it over PCIe. Before freeing, move the hottest units of the chunks being freed into the coldest
    // (or free) slots of the chunks being kept, VRAM to VRAM, so the borrow evicts cold experts and decode restarts warm.
    {
        static const bool migrate = dre_env_int("ALPHA_DRE_BORROW_MIGRATE", 1) != 0;
        std::vector<char> freeing(g_dre.chunks.size(), 0);
        size_t f = 0;
        for (int ch = (int) g_dre.chunks.size() - 1; ch >= 0 && f < want; ch--) {
            if (!g_dre.chunks[ch].alive) continue;
            freeing[ch] = 1; f += g_dre.chunks[ch].size;
        }
        size_t moved = 0; int n_moved = 0;
        if (migrate) {
            for (int ci = 0; ci < (int) g_dre.classes.size(); ci++) {
                dre_class & c = g_dre.classes[ci];
                std::vector<std::pair<float, int>> hot, cold;   // (score, slot)
                for (int sl : c.free) {
                    const dre_slot & x = g_dre.slots[sl];
                    if (x.chunk >= 0 && !freeing[x.chunk]) cold.push_back({ -1.0f, sl });   // free kept slots first
                }
                for (int sl = 0; sl < (int) g_dre.slots.size(); sl++) {
                    const dre_slot & x = g_dre.slots[sl];
                    if (x.li < 0 || x.chunk < 0 || x.pinned || x.pending) continue;
                    const dre_layer & L = g_dre.layers[x.li];
                    if (L.cls != ci) continue;
                    const float sc = dre_score(L, x.e);
                    if (freeing[x.chunk]) hot.push_back({ sc, sl }); else cold.push_back({ sc, sl });
                }
                std::sort(hot.begin(), hot.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
                std::stable_sort(cold.begin(), cold.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
                size_t j = 0;
                for (size_t i = 0; i < hot.size() && j < cold.size() && hot[i].first > cold[j].first; i++, j++) {
                    const int ss = hot[i].second, ds = cold[j].second;
                    dre_slot & src = g_dre.slots[ss];
                    dre_slot & dst = g_dre.slots[ds];
                    if (dst.li >= 0) {   // a cold resident unit gives up its slot: it joins the restore list
                        dre_layer & D = g_dre.layers[dst.li];
                        D.slot[dst.e] = -1;
                        dre_set_entries(D, dst.e);
                        g_dre_restore.push_back({ dst.li, dst.e });
                        lru_unlink(c, ds);
                    } else {
                        c.free.erase(std::remove(c.free.begin(), c.free.end(), ds), c.free.end());
                    }
                    CUDA_CHECK(cudaMemcpyAsync(dst.ptr, src.ptr, c.size, cudaMemcpyDeviceToDevice, g_dre.dma));
                    dre_layer & L = g_dre.layers[src.li];
                    dst.li = src.li; dst.e = src.e; dst.pinned = false; dst.pending = false;
                    L.slot[dst.e] = ds;
                    dre_set_entries(L, dst.e);
                    lru_unlink(c, ss);
                    lru_push_front(c, ds);
                    src.li = -1; src.e = -1;   // nothing left to evict in the freed chunk for this slot
                    moved += c.size; n_moved++;
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(g_dre.dma));
            if (n_moved) fprintf(stderr, "[dre] borrow: kept the %d hottest units of the freed chunks (%.0f MB moved in VRAM)\n",
                                 n_moved, moved / 1048576.0);
        }
    }
    const int64_t bt2 = ggml_time_us();
    size_t freed = 0;
    for (int ch = (int) g_dre.chunks.size() - 1; ch >= 0 && freed < want; ch--) {
        auto & k = g_dre.chunks[ch];
        if (!k.alive) continue;
        for (int sl = 0; sl < (int) g_dre.slots.size(); sl++) {
            dre_slot & x = g_dre.slots[sl];
            if (x.chunk != ch) continue;
            if (x.li >= 0) {
                dre_layer & L = g_dre.layers[x.li];
                L.slot[x.e] = -1;
                dre_set_entries(L, x.e);
                lru_unlink(g_dre.classes[L.cls], sl);
                g_dre_restore.push_back({ x.li, x.e });
            }
            x.li = -1; x.e = -1; x.chunk = -2; x.pinned = false;   // dead
        }
        g_dre.pool_used -= k.used;
        CUDA_CHECK(cudaFree(k.base));
        k.base = nullptr; k.used = 0; k.alive = false;
        freed += k.size;
    }
    for (auto & c : g_dre.classes) {   // drop dead slots from the free lists
        c.free.erase(std::remove_if(c.free.begin(), c.free.end(), [](int sl) { return g_dre.slots[sl].chunk < 0; }), c.free.end());
    }
    dre_upload_dirty(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (FILE * tf = dre_trace_file()) {
        const uint32_t mk = 0xFFFFFFFFu, n = (uint32_t) g_dre_restore.size();
        fwrite(&mk, 4, 1, tf); fwrite(&n, 4, 1, tf);
        for (const auto & u : g_dre_restore) { const uint16_t h[2] = { (uint16_t) u.first, (uint16_t) u.second }; fwrite(h, 2, 2, tf); }
        fflush(tf);
    }
    g_dre.borrowed += freed;
    fprintf(stderr, "[dre] prefill borrowed %.0f MB of the expert pool\n", freed / 1048576.0);
    if (bt_on) fprintf(stderr, "[dre-rt] borrow: drain+sync %.1f ms, migrate %.1f ms, free chunks %.1f ms\n",
                       (bt1 - bt0) / 1e3, (bt2 - bt1) / 1e3, (ggml_time_us() - bt2) / 1e3);
    // ALPHA_DRE_PF_EARLY (default 0): start streaming the first MoE layers' experts now, while the host still has the
    // scheduler reserve, the graph build and the inputs ahead of it and the copy engine would sit idle
    static const bool pf_early = dre_env_int("ALPHA_DRE_PF_EARLY", 0) != 0;   // opt-in, no measurable gain
    if (pf_early && dre_pfs_on() && g_dre_ctx && !g_dre.layer_by_blk.empty()) {
        const int li0 = g_dre.layer_by_blk.begin()->second;
        const dre_layer & L0 = g_dre.layers[li0];
        if (L0.complete()) {
            for (int r = 0; r < 3; r++) {
                int ti = L0.ti[r];
                const int k0 = dre_pfs_issue(ti, -1, g_dre_ctx->stream());
                for (int a = 0; a < dre_pfs_depth() && k0 >= 0; a++) {
                    ti = dre_tensor_next_layer(ti);
                    if (ti < 0) break;
                    (void) dre_pfs_issue(ti, k0, g_dre_ctx->stream());
                }
            }
        }
    }
    return freed;
}

GGML_BACKEND_API_C void ggml_cuda_dre_return_begin(size_t keep_extra);
GGML_BACKEND_API_C void ggml_cuda_dre_return_finish(void) {
    if (!ggml_cuda_dre_enabled()) return;
    dre_restore_finish();
}
GGML_BACKEND_API_C void ggml_cuda_dre_return(void) {
    ggml_cuda_dre_return_begin(0);
    ggml_cuda_dre_return_finish();
}

// ALPHA_DRE_RETURN_OVERLAP: the pool return in two halves: _begin re-allocates the borrowed chunks (leaving keep_extra
// bytes free for a compute buffer allocated before _finish) and issues the restore copies on the DMA stream; _finish
// waits for them and publishes the slots. No expert graph may run in between.
GGML_BACKEND_API_C void ggml_cuda_dre_return_begin(size_t keep_extra) {
    if (!ggml_cuda_dre_enabled()) return;
    // ALPHA_DRE_RETURN_TIMERS: host phases of the pool return
    static const bool rt = dre_env_int("ALPHA_DRE_RETURN_TIMERS", 0) != 0;
    const int64_t rt0 = ggml_time_us();
    dre_pfs_release();
    const int64_t rt1 = ggml_time_us();
    if (g_dre.borrowed == 0) return;
    // ALPHA_DRE_RETURN_TRIM (default 1): unmapping the temp pools frees a few hundred MB but drops every captured
    // CUDA graph, so the first decode windows after each prompt run uncaptured and re-capture (0 keeps them)
    static const bool ret_trim = dre_env_int("ALPHA_DRE_RETURN_TRIM", 1) != 0;
    if (g_dre_ctx && ret_trim) {
        const size_t t = ggml_cuda_trim_pools(*g_dre_ctx);
        if (t) fprintf(stderr, "[dre] trimmed %.0f MB of idle CUDA temp pools\n", t / 1048576.0);
    }
    if (rt) fprintf(stderr, "[dre-rt] trim %.1f ms\n", (ggml_time_us() - rt1) / 1e3);
    size_t back = 0;
    const size_t keep_free = ((size_t) dre_env_int("ALPHA_DRE_RETURN_KEEP_MB", 768) << 20) + keep_extra;   // decode headroom
    // ALPHA_DRE_RETURN_PIPE (default 1): the restore starts with the first re-allocated chunk instead of after the
    // last one: each chunk's units are placed and their copies queued while the next chunk is allocated
    static const bool ret_pipe = dre_env_int("ALPHA_DRE_RETURN_PIPE", 1) != 0;
    const bool piped = ret_pipe && !(dre_env_int("ALPHA_DRE_RESTORE_ASYNC", 0) != 0 && dre_pacer_on()) && dre_restore_begin();
    if (piped) dre_restore_pass();   // free slots in the chunks that stayed
    for (auto & k : g_dre.chunks) {
        if (k.alive) continue;
        size_t free_mem = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total));
        if (free_mem < k.size + keep_free) break;
        void * b = nullptr;
        if (cudaMalloc(&b, k.size) != cudaSuccess) { (void) cudaGetLastError(); continue; }
        k.base = (char *) b; k.used = 0; k.alive = true;
        back += k.size;
        if (piped) dre_restore_pass();
    }
    g_dre.borrowed -= std::min(g_dre.borrowed, back);
    if (rt) fprintf(stderr, "[dre-rt] staging release %.1f ms, trim + chunk allocs %.1f ms\n", (rt1 - rt0) / 1e3, (ggml_time_us() - rt1) / 1e3);
    dre_ids_hist_print();
    {
        static long long last_h2d = 0, last_d2d = 0;
        fprintf(stderr, "[dre] prompt routed expert copies: h2d %lld d2d %lld (expert tensors)\n",
                g_dre.n_pf_h2d - last_h2d, g_dre.n_pf_d2d - last_d2d);
        last_h2d = g_dre.n_pf_h2d; last_d2d = g_dre.n_pf_d2d;
    }
    if (FILE * tf = dre_trace_file()) { const uint32_t mk = 0xFFFFFFFEu; fwrite(&mk, 4, 1, tf); fflush(tf); }
    fprintf(stderr, "[dre] prefill returned %.0f MB to the expert pool (%.0f MB still out)\n",
            back / 1048576.0, g_dre.borrowed / 1048576.0);
    if (piped) dre_restore_end(true); else dre_restore_borrowed(true);
}

bool ggml_cuda_dre_lookup(const ggml_tensor * w, const ggml_tensor * ids, cudaStream_t stream,
                          const void ** base, const int64_t ** tagged_table) {
    if (!ggml_cuda_dre_enabled() || !w || !w->data) return false;
    auto it = g_dre.by_data.find(w->data);
    if (it == g_dre.by_data.end() || it->second < 0) return false;
    dre_tensor & d = g_dre.tens[it->second];
    dre_layer & L = g_dre.layers[d.layer];
    if (ids && !(dre_cpu_mode() && L.complete())) {
        const int n = (int) (ids->ne[0] * ids->ne[1]);
        if (n <= DRE_MAX_IDS && ids->nb[0] == sizeof(int32_t) && ids != g_dre_last_ids) {
            // one readback per MoE layer: gate, up and down are consecutive nodes over the same ids tensor
            // (a view of the router's argsort rows: 2-D copy, tokens ids->nb[1] apart)
            CUDA_CHECK(cudaMemcpy2DAsync(L.ids_pin, ids->ne[0] * sizeof(int32_t), ids->data, ids->nb[1],
                                         ids->ne[0] * sizeof(int32_t), ids->ne[1], cudaMemcpyDeviceToHost, stream));
            L.n_ids = n;
            g_dre_last_ids = ids;
        }
    }
    *base = d.host;
    *tagged_table = (const int64_t *) ((uintptr_t) d.d_tab | 1);
    return true;
}

void ggml_cuda_dre_pre(const ggml_tensor * w, const ggml_tensor * src1, const ggml_tensor * ids, cudaStream_t stream) {
    if (!dre_cpu_mode() || !ids || !w) return;
    auto it = g_dre.by_data.find(w->data);
    if (it == g_dre.by_data.end() || it->second < 0) return;
    dre_tensor & d = g_dre.tens[it->second];
    dre_layer & L = g_dre.layers[d.layer];
    if (!L.complete() || !L.bell || d.role == 2 || ids == g_dre_last_bell_ids) return;   // once, before gate/up
    g_dre_last_bell_ids = ids;
    dre_cpu_start();
    // gate/up input: [K, 1, n_tok] (broadcast over the experts), contiguous
    const int ntok = (int) ids->ne[1];
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1) && src1->ne[0] <= DRE_MAX_EMBD &&
                src1->ne[1] == 1 && src1->ne[2] == ntok && ntok <= DRE_MAX_TOK);
    const int n = (int) (ids->ne[0] * ids->ne[1]);
    GGML_ASSERT(ids->nb[0] == sizeof(int32_t) && n <= DRE_MAX_IDS);
    L.n_ids = n;
    dre_signal_k<<<1, 256, 0, stream>>>((const float *) src1->data, (int) (src1->ne[0] * ntok), (const int32_t *) ids->data, n,
                                        ntok, (int) (ids->nb[1] / sizeof(int32_t)), L.bell_d, L.ctr_d);
}

void ggml_cuda_dre_post(const ggml_tensor * w, ggml_tensor * dst, cudaStream_t stream) {
    if (!dre_cpu_mode() || !w) return;
    auto it = g_dre.by_data.find(w->data);
    if (it == g_dre.by_data.end() || it->second < 0) return;
    dre_tensor & d = g_dre.tens[it->second];
    dre_layer & L = g_dre.layers[d.layer];
    if (!L.complete() || !L.bell || d.role != 2) return;
    // dst: [n_out, n_expert_used, 1]; slot k lives at k*nb[1]
    if (!g_dre_wait_ns_d) {
        CUDA_CHECK(cudaMalloc(&g_dre_wait_ns_d, (DRE_MAX_TOK + 1) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(g_dre_wait_ns_d, 0, (DRE_MAX_TOK + 1) * sizeof(unsigned long long)));
    }
    // dst: [n_out, n_used, n_tok]
    static const int fill_blocks = std::max(1, dre_env_int("ALPHA_DRE_FILL_BLOCKS", 1));
    dre_wait_fill_k<<<fill_blocks, 256, 0, stream>>>(L.ctr_d, L.ans_d, L.y_d, (float *) dst->data, (int) dst->ne[0],
                                           (int64_t) (dst->nb[1] / sizeof(float)), (int) dst->ne[1],
                                           (int64_t) (dst->nb[2] / sizeof(float)),
                                           g_dre_wait_ns_d + std::min<int64_t>(DRE_MAX_TOK, std::max<int64_t>(1, dst->ne[2])));
}

static std::vector<std::pair<int,int>> g_dre_warm_backlog;   // prompt-hot experts still to admit (paced)

// ALPHA_DRE_PROFILE: expert-popularity profile, so the pool does not start empty and track only recency.
// File: "DREP" u32 n_layers u64 n_tokens, then per layer: i32 blk, i32 n_exp, u32 uses[n_exp].
// ALPHA_DRE_PROFILE_OUT=path writes the running counts at every stats line; ALPHA_DRE_PROFILE=path bulk-fills the
// pool with the most-used experts once, before the first decode, and seeds the LFU scores.
static long long g_dre_tok_counted = 0;
static void dre_profile_save() {
    static const char * path = getenv("ALPHA_DRE_PROFILE_OUT");
    if (!path) return;
    FILE * f = fopen(path, "wb");
    if (!f) return;
    const uint32_t magic = 0x50455244u;   // "DREP"
    const uint32_t nl = (uint32_t) g_dre.layers.size();
    const uint64_t nt = (uint64_t) g_dre_tok_counted;
    fwrite(&magic, 4, 1, f); fwrite(&nl, 4, 1, f); fwrite(&nt, 8, 1, f);
    for (const auto & L : g_dre.layers) {
        const int32_t blk = L.blk, ne = L.n_exp;
        fwrite(&blk, 4, 1, f); fwrite(&ne, 4, 1, f);
        fwrite(L.uses.data(), 4, (size_t) ne, f);
    }
    fclose(f);
}
static bool g_dre_tier_done = false;
static std::atomic<bool> g_dre_tier_entries_dirty { false };   // set by the background fill when it ends
static std::atomic<bool> g_dre_tier_busy { false };            // a background fill is copying / pinning
// ALPHA_DRE_TIER: the serve's warm-up keeps /health at "loading" while this is true, so no request runs during the fill
GGML_BACKEND_API_C bool ggml_cuda_dre_tier_busy(void) { return g_dre_tier_busy.load(); }
// ALPHA_DRE_TIER: a server with a startup warm-up defers the fill out of the first decode (where it would run inside a
// request long enough for the server's idle timeout to cancel the warm-up) and starts it with ggml_cuda_dre_tier_start()
// once the warm-up request is done: nothing decodes during the fill and /health is still "loading"
static std::atomic<bool> g_dre_tier_deferred { false };
static void dre_tier_fill();
GGML_BACKEND_API_C void ggml_cuda_dre_tier_defer(void) { g_dre_tier_deferred.store(true); }
GGML_BACKEND_API_C void ggml_cuda_dre_tier_start(void) {
    if (!g_dre_tier_deferred.exchange(false) || !ggml_cuda_dre_enabled() || !g_dre_ctx) return;
    CUDA_CHECK(cudaSetDevice(g_dre_ctx->device));
    dre_tier_fill();   // the zero-copy entries are refreshed by the next graph (g_dre_tier_entries_dirty)
}
// ALPHA_DRE_TIER_GB (default: MemAvailable - 12 GB): RAM budget of the hot tier. Units (a layer's gate+up+down of one
// expert) are taken hottest first by the routing profile; the rest stay on the file mapping (page cache / SSD).
static void dre_tier_fill() {
#if defined(__linux__)
    if (g_dre_tier_done || g_dre_tier_bufs.empty()) return;
    for (const auto & L : g_dre.layers) if (!L.complete()) return;
    g_dre_tier_done = true;
    struct unit { float f; int li, e; size_t bytes; };
    std::vector<unit> units;
    size_t tier_total = 0;
    bool have_prof = false;
    for (int li = 0; li < (int) g_dre.layers.size(); li++) {
        const dre_layer & L = g_dre.layers[li];
        if (!g_dre.tens[L.ti[0]].hp) continue;
        size_t ub = 0;
        for (int r = 0; r < 3; r++) ub += g_dre.tens[L.ti[r]].esize;
        if (!L.prof.empty()) have_prof = true;
        for (int e = 0; e < L.n_exp; e++) {
            units.push_back({ L.prof.empty() ? 0.0f : L.prof[e], li, e, ub });
            tier_total += ub;
        }
    }
    if (units.empty()) return;
    if (!have_prof) fprintf(stderr, "[dre] tier: no routing profile (ALPHA_DRE_PROFILE) -- hot set chosen by expert index\n");
    std::stable_sort(units.begin(), units.end(), [](const unit & a, const unit & b) { return a.f > b.f; });
    long long avail_kb = 0;
    if (FILE * mf = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), mf)) if (sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
        fclose(mf);
    }
    const char * gb = getenv("ALPHA_DRE_TIER_GB");
    const long long auto_b = (avail_kb << 10) - (12LL << 30);
    size_t budget = gb ? (size_t) (atof(gb) * 1073741824.0) : (size_t) std::max(0LL, auto_b);
    budget = std::min(budget, tier_total);
    size_t n_units = 0, bytes = 0;
    while (n_units < units.size() && bytes + units[n_units].bytes <= budget) bytes += units[n_units++].bytes;
    if (n_units == 0) { fprintf(stderr, "[dre] tier: no RAM budget (%.1f GB available) -- every expert from the file mapping\n", avail_kb / 1048576.0); return; }
    // every copy already issued from the old sources (the profile's bulk pool fill, paced admissions) lands first
    dre_pacer_drain();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    const size_t alloc = bytes + 4096;   // slack: a prefill copy may read a little past its last expert
    void * a = mmap(nullptr, alloc, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a == MAP_FAILED) { fprintf(stderr, "[dre] tier: mmap of %.1f GB failed -- every expert from the file mapping\n", alloc / 1073741824.0); return; }
    // ALPHA_DRE_TIER_THP (default 0): transparent huge pages for the tier arena. With defrag=madvise every huge-page
    // fault compacts synchronously, which makes the fill several times slower; decode does not benefit (CPU misses read
    // 4 KB pages fine).
    if (dre_env_int("ALPHA_DRE_TIER_THP", 0) != 0) madvise(a, alloc, MADV_HUGEPAGE);
    // destinations, then a parallel copy; each copied source range is dropped from our mapping (it is in the arena now)
    struct piece { char * dst; const char * src; size_t n; int ti; int e; };
    std::vector<piece> pcs;
    pcs.reserve(n_units * 3);
    char * w = (char *) a;
    if (n_units == units.size()) {
        // every expert fits: one contiguous block per tensor in expert order, so a layer's routed experts copy as a few
        // long runs and the prefill gather path works on the arena (a unit-interleaved layout would cut every copy at
        // each expert, and thousands of cudaMemcpyAsync calls per layer cost seconds of host issue per prompt)
        for (int li = 0; li < (int) g_dre.layers.size(); li++) {
            const dre_layer & L = g_dre.layers[li];
            if (!g_dre.tens[L.ti[0]].hp) continue;
            for (int r = 0; r < 3; r++) {
                const dre_tensor & d = g_dre.tens[L.ti[r]];
                for (int e = 0; e < L.n_exp; e++) { pcs.push_back({ w, d.hp[e], d.esize, L.ti[r], e }); w += d.esize; }
            }
        }
    } else {
        for (size_t i = 0; i < n_units; i++) {
            const dre_layer & L = g_dre.layers[units[i].li];
            for (int r = 0; r < 3; r++) {
                const dre_tensor & d = g_dre.tens[L.ti[r]];
                pcs.push_back({ w, d.hp[units[i].e], d.esize, L.ti[r], units[i].e });
                w += d.esize;
            }
        }
    }
    static const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    // ALPHA_DRE_TIER_DROP (default 0): drop each copied source range from our mapping. Only safe for a file mapping
    // (pages re-read from the file); on anonymous memory it zeroes the source. Clean file pages are reclaimed anyway.
    static const bool drop = dre_env_int("ALPHA_DRE_TIER_DROP", 0) != 0;
    static const bool nopin  = dre_env_int("ALPHA_DRE_TIER_NOPIN", 0) != 0;    // diagnostic: leave the arena pageable
    static const bool noswap = dre_env_int("ALPHA_DRE_TIER_NOSWAP", 0) != 0;   // diagnostic: keep every source on the mapping
    // each expert is repointed as soon as its piece lands: until then it is read from the file mapping (same bytes)
    struct job { std::vector<piece> pcs; std::vector<size_t> order; size_t n_units, n_all, bytes, total; long long avail_kb; void * a; size_t alloc; };
    auto J = std::make_shared<job>();
    J->pcs = std::move(pcs);
    J->order.resize(J->pcs.size());
    for (size_t k = 0; k < J->order.size(); k++) J->order[k] = k;
    // source order, each thread a contiguous slice: hottest-first order would read the file at random 4 KB faults
    std::sort(J->order.begin(), J->order.end(), [&](size_t x, size_t y) { return J->pcs[x].src < J->pcs[y].src; });
    J->n_units = n_units; J->n_all = units.size(); J->bytes = bytes; J->total = tier_total; J->avail_kb = avail_kb;
    J->a = a; J->alloc = alloc;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    // ALPHA_DRE_TIER_BG (default 0): copy in a background thread while decode runs. The overlapped decode starves (its
    // CPU misses page-fault the file from the disk the fill saturates) and a fill can race a graph update. /health stays
    // "loading" for the fill either way, so the foreground fill (inside the warm-up) costs no client anything.
    static const bool bg = dre_env_int("ALPHA_DRE_TIER_BG", 0) != 0;
    auto run = [J, t0, dev]() {
        if (bg) {
            (void) cudaSetDevice(dev);
            cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
            (void) cudaThreadExchangeStreamCaptureMode(&mode);
        }
        const int nt = std::max(1, dre_env_int("ALPHA_DRE_ARENA_THREADS", 16));
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) {
            th.emplace_back([&, t]() {
                const auto & order = J->order;
                const size_t k0 = order.size() * (size_t) t / (size_t) nt, k1 = order.size() * (size_t) (t + 1) / (size_t) nt;
                for (size_t i = k0; i < k1; i++) {
                    const piece & q = J->pcs[order[i]];
                    if (i + 1 < k1) {
                        const piece & nx = J->pcs[order[i + 1]];
                        madvise((void *) ((uintptr_t) nx.src & ~(page - 1)), nx.n + page, MADV_WILLNEED);
                    }
                    memcpy(q.dst, q.src, q.n);
                    if (!noswap) {   // published pageable; pinned below once the whole arena is registered
                        dre_tensor & d = g_dre.tens[q.ti];
                        __atomic_store_n(&d.hp[q.e], (const char *) q.dst, __ATOMIC_RELEASE);
                    }
                    if (drop) {
                        const uintptr_t lo = ((uintptr_t) q.src + page - 1) & ~(page - 1);
                        const uintptr_t hi = ((uintptr_t) q.src + q.n) & ~(page - 1);
                        if (hi > lo) madvise((void *) lo, hi - lo, MADV_DONTNEED);
                    }
                }
            });
        }
        for (auto & x : th) x.join();
        // pin after the copy, off the request path: registering a large not-yet-faulted region up front (on a page cache
        // full of the model file) can block the warm-up request past its timeout.
        // Registered with no graph capture in flight (ggml_cuda_outside_capture): a register that overlaps a capture
        // makes cudaGraphExecUpdate fail. One registration: a copy whose source spans two separate registrations fails
        // with "invalid argument" (merged expert runs cross any piece boundary).
        struct reg_arg { void * p; size_t n; cudaError_t r; } ra = { J->a, J->alloc, cudaSuccess };
        const auto tp0 = std::chrono::steady_clock::now();
        if (!nopin) ggml_cuda_outside_capture([](void * x) {
            reg_arg * r = (reg_arg *) x;
            r->r = cudaHostRegister(r->p, r->n, cudaHostRegisterPortable | cudaHostRegisterMapped);
        }, &ra);
        const bool pinned = !nopin && ra.r == cudaSuccess;
        if (!pinned) (void) cudaGetLastError();
        fprintf(stderr, "[dre] tier: arena %s in %.1f s\n", pinned ? "pinned" : "left pageable",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count());
        if (pinned && !noswap) for (const piece & q : J->pcs) __atomic_store_n(&g_dre.tens[q.ti].hpin[q.e], (uint8_t) 1, __ATOMIC_RELEASE);
        if (pinned && !noswap) {   // tensors whose every expert is now one pinned contiguous arena block
            int nc = 0;
            for (auto & d : g_dre.tens) {
                if (!d.hp || d.contig) continue;
                const int ne = (int) d.t->ne[2];
                bool ok = true;
                for (int e = 0; e < ne && ok; e++) ok = d.hpin[e] && d.hp[e] == d.hp[0] + (size_t) e * d.esize;
                if (ok) { __atomic_store_n(&d.contig, true, __ATOMIC_RELEASE); nc++; }
            }
            fprintf(stderr, "[dre] tier: %d expert tensors contiguous in the pinned arena (prefill gather / merged runs)\n", nc);
        }
        if (!drop && dre_env_int("ALPHA_DRE_TIER_VERIFY", 0) != 0) {   // diagnostic: every arena piece vs its source
            size_t bad = 0;
            for (const piece & q : J->pcs) bad += memcmp(q.dst, q.src, q.n) != 0;
            fprintf(stderr, "[dre] tier verify: %zu of %zu pieces differ from their source\n", bad, J->pcs.size());
        }
        g_dre_tier_entries_dirty.store(true);   // zero-copy miss entries may now point into the arena (main thread)
        fprintf(stderr, "[dre] tier: %zu of %zu expert units (%.1f of %.1f GB) in a %s RAM hot tier in %.1f s (%.1f GB available)\n",
                J->n_units, J->n_all, J->bytes / 1073741824.0, J->total / 1073741824.0, pinned ? "pinned" : "pageable (pin failed)",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), J->avail_kb / 1048576.0);
        g_dre_tier_busy.store(false);
    };
    g_dre_tier_busy.store(true);
    if (bg) {
        fprintf(stderr, "[dre] tier: filling %zu expert units (%.1f GB) in the background\n", n_units, bytes / 1073741824.0);
        std::thread(run).detach();
    } else {
        run();
    }
#endif
}

static bool g_dre_profile_done = false;
static void dre_profile_fill() {
    static const char * path = getenv("ALPHA_DRE_PROFILE");
    if (!path || g_dre_profile_done) return;
    for (const auto & L : g_dre.layers) if (!L.complete()) return;   // wait until every MoE layer is registered
    g_dre_profile_done = true;
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[dre] profile %s: cannot open\n", path); return; }
    uint32_t magic = 0, nl = 0; uint64_t nt = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x50455244u || fread(&nl, 4, 1, f) != 1 || fread(&nt, 8, 1, f) != 1 || nt == 0) {
        fprintf(stderr, "[dre] profile %s: bad header\n", path); fclose(f); return;
    }
    struct ent { float f; int li, e; };
    std::vector<ent> v;
    for (uint32_t i = 0; i < nl; i++) {
        int32_t blk = 0, ne = 0;
        if (fread(&blk, 4, 1, f) != 1 || fread(&ne, 4, 1, f) != 1 || ne <= 0 || ne > 4096) break;
        std::vector<uint32_t> u((size_t) ne);
        if (fread(u.data(), 4, (size_t) ne, f) != (size_t) ne) break;
        auto it = g_dre.layer_by_blk.find(blk);
        if (it == g_dre.layer_by_blk.end()) continue;
        dre_layer & L = g_dre.layers[it->second];
        if (L.n_exp != ne) continue;
        static const double H = (double) dre_env_int("ALPHA_DRE_LFU_HALF", 1024);
        L.prof.assign((size_t) ne, 0.0f);
        for (int e = 0; e < ne; e++) {
            const float fr = (float) u[e] / (float) nt;   // uses per token
            L.prof[e] = fr;
            // ALPHA_DRE_PROFILE_SEED: scale of the seeded score (default 0). A full seed makes every profile expert look
            // as hot as its long-run rate, so the first request's own misses cannot evict them until it decays
            static const float seed_k = [] { const char * e2 = getenv("ALPHA_DRE_PROFILE_SEED"); return e2 ? (float) atof(e2) : 0.0f; }();   // default 0: a cold first request decodes faster unseeded
            L.score[e] = (float) (seed_k * fr * H / 0.693);   // the decayed score a steady rate fr converges to
            L.score_tok[e] = g_dre.tok;
            if (u[e] > 0) v.push_back({ fr, it->second, e });
        }
    }
    fclose(f);
    std::sort(v.begin(), v.end(), [](const ent & a, const ent & b) { return a.f > b.f; });
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> done;
    size_t bytes = 0;
    for (const ent & x : v) {
        dre_layer & L = g_dre.layers[x.li];
        if (L.slot[x.e] != -1) continue;
        const int s = dre_take_free_slot(L.cls);
        if (s < 0) continue;
        dre_slot & sl = g_dre.slots[s];
        sl.li = x.li; sl.e = x.e; sl.pending = true; sl.pinned = L.pinned;
        L.slot[x.e] = -2;
        for (int r = 0; r < 3; r++) {
            const dre_tensor & d = g_dre.tens[L.ti[r]];
            CUDA_CHECK(cudaMemcpyAsync(sl.ptr + L.sub[r], dre_src(d, sl.e), d.esize, cudaMemcpyHostToDevice, g_dre.dma));
            bytes += d.esize;
        }
        done.push_back(s);
    }
    CUDA_CHECK(cudaStreamSynchronize(g_dre.dma));
    for (int s : done) {
        dre_slot & sl = g_dre.slots[s];
        dre_layer & L = g_dre.layers[sl.li];
        sl.pending = false;
        L.slot[sl.e] = s;
        dre_set_entries(L, sl.e);
        lru_push_front(g_dre.classes[L.cls], s);
    }
    fprintf(stderr, "[dre] profile %s: filled %zu expert units (%.0f MB) of %zu ranked in %.0f ms\n", path, done.size(),
            bytes / 1048576.0, v.size(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

void ggml_cuda_dre_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (ggml_cuda_dre_enabled()) g_dre_ctx = &ctx;
    if (!ggml_cuda_dre_enabled() || g_dre.tens.empty() || g_dre.failed) return;
    bool has_dre = false;
    for (int i = 0; i < cgraph->n_nodes && !has_dre; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_MUL_MAT_ID && n->src[0] && g_dre.by_data.count(n->src[0]->data)) has_dre = true;
    }
    if (!has_dre) return;
    g_dre_last_ids = nullptr;
    g_dre_last_bell_ids = nullptr;
    cudaStream_t st = ctx.stream();
    if (g_dre.ev_tok_valid) CUDA_CHECK(cudaEventSynchronize(g_dre.ev_tok));
    std::atomic_thread_fence(std::memory_order_seq_cst);

    g_dre.tok++;
    dre_profile_fill();
    if (!g_dre_tier_deferred.load()) dre_tier_fill();   // ALPHA_DRE_TIER: after the profile, whose frequencies pick the hot set
    if (g_dre_tier_entries_dirty.exchange(false)) {   // the background fill ended: refresh zero-copy miss entries here, between graphs
        for (auto & L : g_dre.layers) for (int e = 0; e < L.n_exp; e++) if (L.slot[e] < 0) dre_set_entries(L, e);
    }
    g_dre_tok_counted++;
    // ALPHA_DRE_TRACE=path: append the previous graph's routing: u32 n_layers, then per layer u16 layer, u16 n_tok,
    // u16 n_ids, u16 ids[n_ids], for offline cache-policy simulation
    {
        FILE * tf = dre_trace_file();
        static bool tf_hdr = false;
        if (tf && !tf_hdr) {   // u32 0xFFFFFFFD, u32 n, then n x (i32 blk, i32 n_exp, i32 pool slots of its class): layer index -> blk
            tf_hdr = true;
            const uint32_t mk = 0xFFFFFFFDu, n = (uint32_t) g_dre.layers.size();
            fwrite(&mk, 4, 1, tf); fwrite(&n, 4, 1, tf);
            for (const auto & L : g_dre.layers) {
                const int32_t h[3] = { L.blk, L.n_exp, L.complete() ? (int32_t) (g_dre.pool_size / std::max<size_t>(1, g_dre.classes[L.cls].size)) : 0 };
                fwrite(h, 4, 3, tf);
            }
        }
        if (tf) {
            uint32_t nl = 0;
            for (const auto & L : g_dre.layers) if (L.complete() && L.n_ids > 0) nl++;
            fwrite(&nl, 4, 1, tf);
            for (int li = 0; li < (int) g_dre.layers.size(); li++) {
                const dre_layer & L = g_dre.layers[li];
                if (!L.complete() || L.n_ids <= 0) continue;
                const int32_t * idv = (dre_cpu_mode() && L.bell) ? L.bell->ids : L.ids_pin;
                const uint16_t h[3] = { (uint16_t) li, (uint16_t) (L.bell ? L.bell->n_tok : 1), (uint16_t) L.n_ids };
                fwrite(h, 2, 3, tf);
                for (int k = 0; k < L.n_ids; k++) { const uint16_t v = (uint16_t) idv[k]; fwrite(&v, 2, 1, tf); }
            }
            fflush(tf);
        }
    }
    // 1) account the last token's routing; collect misses (second chance while the pool is full)
    std::vector<std::pair<int,int>> cand;
    for (int li = 0; li < (int) g_dre.layers.size(); li++) {
        dre_layer & L = g_dre.layers[li];
        if (!L.complete()) continue;
        const int32_t * idv = (dre_cpu_mode() && L.bell) ? L.bell->ids : L.ids_pin;
        // n_ids persists: a replayed graph refreshes the ids without running host code
        for (int k = 0; k < L.n_ids; k++) {
            const int e = idv[k];
            if (e < 0 || e >= L.n_exp) continue;
            g_dre.n_use++;
            L.last_use[e] = g_dre.tok;
            L.uses[e]++;
            if (dre_lfu_on()) dre_score_hit(L, e);
            const int s = L.slot[e];
            if (s >= 0) {
                g_dre.n_hit++;
                dre_class & c = g_dre.classes[L.cls];
                lru_unlink(c, s); lru_push_front(c, s);
            } else if (s == -1) {
                const bool room = g_dre.pool_used + g_dre.classes[L.cls].size <= g_dre.pool_size - g_dre.borrowed ||
                                  !g_dre.classes[L.cls].free.empty();
                if (room || g_dre.ghost_w <= 0 || (L.last_miss[e] != 0 && g_dre.tok - L.last_miss[e] <= (uint32_t) g_dre.ghost_w)) {
                    cand.emplace_back(li, e);
                }
                L.last_miss[e] = g_dre.tok;
            }
        }
    }
    // 1a) pinned layers: everything not yet resident goes first
    for (int li = 0; li < (int) g_dre.layers.size(); li++) {
        dre_layer & L = g_dre.layers[li];
        if (!L.pinned || !L.complete()) continue;
        for (int e = 0; e < L.n_exp; e++) if (L.slot[e] == -1) cand.insert(cand.begin(), { li, e });
    }
    // 1b) a prefill ran since the last decode: queue the experts the prompt used most, hottest first
    // ALPHA_DRE_PF_WARM (default 0): cap on the prompt's hottest experts queued for admission. Prompt routing is not
    // decode routing, so warm-admitting prompt-hot experts churns the decode pool; a small cap can still help the first
    // request after a start on short prompts.
    static const int pf_warm_max = dre_env_int("ALPHA_DRE_PF_WARM", 0);
    if (g_dre.pf_pending && pf_warm_max > 0) {
        g_dre.pf_pending = false;
        std::vector<std::pair<uint32_t, std::pair<int,int>>> hot;
        for (int li = 0; li < (int) g_dre.layers.size(); li++) {
            dre_layer & L = g_dre.layers[li];
            if (!L.complete()) { std::fill(L.pf_count.begin(), L.pf_count.end(), 0); continue; }
            for (int e = 0; e < L.n_exp; e++) {
                if (L.pf_count[e] > 0 && L.slot[e] == -1) hot.push_back({ L.pf_count[e], { li, e } });
                L.pf_count[e] = 0;
            }
        }
        std::sort(hot.begin(), hot.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        const size_t keep = std::min(hot.size(), (size_t) pf_warm_max);
        // the prompt's hot set is a backlog drained at the paced admission rate below, behind the decode's own misses
        g_dre_warm_backlog.clear();
        for (size_t i = 0; i < keep; i++) g_dre_warm_backlog.push_back(hot[i].second);
        g_dre.n_pf_warm += (long long) keep;
    }
    // Pacing: H2D admissions share the copy engine with every graph's input upload, and an input queued behind
    // ~100 MB of expert copies stalls the next graph for milliseconds. So issue a new batch only when at most
    // ALPHA_DRE_INFLIGHT batches are still copying, and at most ALPHA_DRE_ADMIT units per batch; the warm backlog
    // waits its turn.
    static const int max_inflight = dre_env_int("ALPHA_DRE_INFLIGHT", 1);
    size_t n_inflight = 0;
    for (auto & bt : g_dre.batches) {
        if (!dre_batch_done(bt)) n_inflight++;
    }
    const bool may_admit = (int) n_inflight < max_inflight;
    if (may_admit) {
        for (const auto & w : g_dre_warm_backlog) cand.push_back(w);
    }
    // 2) publish admissions whose copies have landed
    for (size_t b = 0; b < g_dre.batches.size(); ) {
        auto & bt = g_dre.batches[b];
        if (!dre_batch_done(bt)) { b++; continue; }
        for (int s : bt.slots) {
            dre_slot & x = g_dre.slots[s];
            dre_layer & L = g_dre.layers[x.li];
            x.pending = false;
            L.slot[x.e] = s;
            dre_set_entries(L, x.e);
            lru_push_front(g_dre.classes[L.cls], s);
            g_dre.n_publish++;
        }
        dre_batch_free(bt);
        g_dre.batches.erase(g_dre.batches.begin() + b);
    }
    // 3) admit misses (evicting unpublishes victims, so the table upload below precedes any overwrite)
    std::vector<int> newslots;
    const int admit_cap = may_admit ? g_dre.admit_per_tok : 0;
    size_t n_backlog_used = 0;
    const size_t n_own = cand.size() - (may_admit ? g_dre_warm_backlog.size() : 0);
    for (size_t ci = 0; ci < cand.size(); ci++) {
        auto & ce = cand[ci];
        if ((int) newslots.size() >= admit_cap) break;
        if (ci >= n_own) n_backlog_used = ci - n_own + 1;
        dre_layer & L = g_dre.layers[ce.first];
        if (L.slot[ce.second] != -1) continue;
        // pinned layers and the prompt's warm backlog are admitted unconditionally
        const bool lfu = dre_lfu_on() && !L.pinned && ci < n_own;
        const int s = lfu ? dre_take_slot_lfu(L.cls, dre_score(L, ce.second)) : dre_take_slot(L.cls);
        if (s < 0) { if (lfu) continue; break; }
        dre_slot & x = g_dre.slots[s];
        x.li = ce.first; x.e = ce.second; x.pending = true;
        x.pinned = L.pinned;
        L.slot[ce.second] = -2;
        newslots.push_back(s);
    }
    if (n_backlog_used) g_dre_warm_backlog.erase(g_dre_warm_backlog.begin(), g_dre_warm_backlog.begin() + n_backlog_used);
    dre_upload_dirty(st);
    if (!newslots.empty()) {
        cudaEvent_t ev;
        if (!g_dre.ev_free.empty()) { ev = g_dre.ev_free.back(); g_dre.ev_free.pop_back(); }
        else CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev, st));               // after the unpublish upload
        if (dre_pacer_on()) {
            auto pj = std::make_shared<dre_state::pacer_job>();
            pj->ev_ready = ev;
            if (!g_dre.ev_free.empty()) { pj->ev_done = g_dre.ev_free.back(); g_dre.ev_free.pop_back(); }
            else CUDA_CHECK(cudaEventCreateWithFlags(&pj->ev_done, cudaEventDisableTiming));
            for (int s : newslots) {
                const dre_slot & x = g_dre.slots[s];
                const dre_layer & L = g_dre.layers[x.li];
                for (int r = 0; r < 3; r++) {
                    const dre_tensor & d = g_dre.tens[L.ti[r]];
                    pj->pieces.push_back({ x.ptr + L.sub[r], dre_src(d, x.e), d.esize });
                }
            }
            g_dre.batches.push_back({ ev, newslots, pj });
            dre_pacer_submit(pj);
        } else {
        CUDA_CHECK(cudaStreamWaitEvent(g_dre.dma, ev, 0));
        for (int s : newslots) {
            const dre_slot & x = g_dre.slots[s];
            const dre_layer & L = g_dre.layers[x.li];
            for (int r = 0; r < 3; r++) {
                const dre_tensor & d = g_dre.tens[L.ti[r]];
                CUDA_CHECK(cudaMemcpyAsync(x.ptr + L.sub[r], dre_src(d, x.e), d.esize,
                                           cudaMemcpyHostToDevice, g_dre.dma));
            }
        }
        CUDA_CHECK(cudaEventRecord(ev, g_dre.dma));        // re-recorded: now marks this batch's copies done
        g_dre.batches.push_back({ ev, newslots, nullptr });
        }
        g_dre.n_admit += (long long) newslots.size();
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    g_dre.n_tok++;
    static const int every = dre_env_int("ALPHA_DRE_STATS", 256);
    if (every > 0 && g_dre.n_tok % every == 0) {
        unsigned long long wns = 0, wt[DRE_MAX_TOK + 1] = {0};
        if (g_dre_wait_ns_d) {
            CUDA_CHECK(cudaMemcpy(wt, g_dre_wait_ns_d, sizeof(wt), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemset(g_dre_wait_ns_d, 0, sizeof(wt)));
            for (int T = 0; T <= DRE_MAX_TOK; T++) wns += wt[T];
        }
        const long long cns = g_dre_cpu_ns.exchange(0);
        fprintf(stderr, "[dre] tok %lld: hit %.1f%% (window) | pool %.0f/%.0f MB, %zu unit slots | admitted %lld evicted %lld "
                        "published %lld | cpu experts %lld (unique %lld) | in flight %zu | per token: gpu waits %.2f ms, cpu busy %.2f ms"
                        " | prefill d2d %lld h2d %lld warm %lld\n",
                g_dre.n_tok, g_dre.n_use ? 100.0 * g_dre.n_hit / g_dre.n_use : 0.0,
                g_dre.pool_used / 1048576.0, g_dre.pool_size / 1048576.0, g_dre.slots.size(),
                g_dre.n_admit, g_dre.n_evict, g_dre.n_publish, g_dre.n_cpu.load(), g_dre.n_cpu_u.load(), g_dre.batches.size(),
                wns / 1e6 / every, cns / 1e6 / every, g_dre.n_pf_d2d, g_dre.n_pf_h2d, g_dre.n_pf_warm);
        if (g_dre_stage_n[0] > 0) {
            fprintf(stderr, "[dre]   cpu job stages (us): setup+quant %.2f | compute %.2f | answer %.2f  (%lld jobs) ; empty jobs %.2f (%lld)\n",
                    g_dre_stage_acc[0] / g_dre_stage_n[0], g_dre_stage_acc[1] / g_dre_stage_n[0], g_dre_stage_acc[2] / g_dre_stage_n[0],
                    g_dre_stage_n[0], g_dre_stage_n[1] ? g_dre_stage_acc[3] / g_dre_stage_n[1] : 0.0, g_dre_stage_n[1]);
            for (auto & v : g_dre_stage_acc) v = 0; g_dre_stage_n[0] = g_dre_stage_n[1] = 0;
        }
        if (g_dre_pred[1] > 0) {
            fprintf(stderr, "[dre]   miss repeat: %.1f%% of unique CPU misses were also missed by the previous job of that layer\n",
                    100.0 * g_dre_pred[0] / g_dre_pred[1]);
            g_dre_pred[0] = g_dre_pred[1] = 0;
        }
        for (int T = 1; T <= DRE_MAX_TOK; T++) {
            long long * a = g_dre_tstat[T];
            if (a[0] == 0) continue;
            fprintf(stderr, "[dre]   T=%d: %lld layer-jobs, %.1f us/layer, %.2f misses/layer (%.2f unique), %.1f us/miss | GPU waited %.1f us/layer\n", T, a[0],
                    a[1] / 1e3 / a[0], (double) a[2] / a[0], (double) a[3] / a[0], a[2] ? a[1] / 1e3 / a[2] : 0.0,
                    wt[T] / 1e3 / a[0]);
            a[0] = a[1] = a[2] = a[3] = 0;
        }
        g_dre.n_use = g_dre.n_hit = 0;
        dre_profile_save();
    }
}

void ggml_cuda_dre_graph_end(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    GGML_UNUSED(cgraph);
    if (!ggml_cuda_dre_enabled() || g_dre.tens.empty() || g_dre.failed) return;
    CUDA_CHECK(cudaEventRecord(g_dre.ev_tok, ctx.stream()));
    g_dre.ev_tok_valid = true;
}

#include "dre_pfcpu.inc"
