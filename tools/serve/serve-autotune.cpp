// SPDX-License-Identifier: Apache-2.0
// See serve-autotune.h.
#include "serve-autotune.h"

#include "gguf.h"

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/stat.h>
#endif

#if defined(_WIN32)
// POSIX setenv() on top of _putenv_s, with the overwrite flag honoured (as in serve-main.cpp)
static inline int setenv(const char * k, const char * v, int overwrite) {
    if (!overwrite && getenv(k)) return 0;
    return _putenv_s(k, v);
}
#endif

namespace alphaserve {

static std::string read_first_line(const std::string & path) {
    std::ifstream f(path);
    std::string   s;
    if (f) std::getline(f, s);
    return s;
}

static std::string cpu_model() {
#if defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string   line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            const size_t c = line.find(':');
            if (c != std::string::npos) return line.substr(c + 2);
        }
    }
#endif
    return "unknown";
}

// "0-3,8,10-11" -> {0,1,2,3,8,10,11}
static std::vector<int> parse_cpu_list(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string       part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) continue;
        const size_t d = part.find('-');
        const int    a = atoi(part.c_str());
        const int    b = d == std::string::npos ? a : atoi(part.c_str() + d + 1);
        for (int i = a; i <= b; ++i) out.push_back(i);
    }
    return out;
}

static std::string cpu_list_str(const std::vector<int> & v) {
    std::string out;
    for (size_t i = 0; i < v.size();) {
        size_t j = i;
        while (j + 1 < v.size() && v[j + 1] == v[j] + 1) ++j;
        if (!out.empty()) out += ",";
        out += std::to_string(v[i]);
        if (j > i) out += "-" + std::to_string(v[j]);
        i = j + 1;
    }
    return out;
}

// The online CPUs grouped by last-level (L3) cache domain, each domain sorted.
static std::vector<std::vector<int>> l3_domains() {
    std::vector<std::vector<int>> doms;
#if defined(__linux__)
    std::set<std::string> seen;
    for (int cpu : parse_cpu_list(read_first_line("/sys/devices/system/cpu/online"))) {
        const std::string l = read_first_line("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                              "/cache/index3/shared_cpu_list");
        if (l.empty() || !seen.insert(l).second) continue;
        std::vector<int> d = parse_cpu_list(l);
        std::sort(d.begin(), d.end());
        doms.push_back(d);
    }
#endif
    return doms;
}

static std::string cache_path() {
    if (const char * p = getenv("ALPHA_AUTOTUNE_CACHE")) return p;
    const char * home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/alphainfer/autotune.tsv";
}

// pinned host -> device copies of 256 MiB, best of 3 (GB/s); 0 when no CUDA device answers
static double measure_pcie_gbs() {
#ifdef GGML_USE_CUDA
    const size_t n = 256ull << 20;
    void * h = nullptr;
    void * d = nullptr;
    double best = 0;
    if (cudaMallocHost(&h, n) == cudaSuccess && cudaMalloc(&d, n) == cudaSuccess) {
        memset(h, 1, n);
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);   // warm-up
        for (int i = 0; i < 3; ++i) {
            cudaEventRecord(a);
            cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            if (ms > 0) best = std::max(best, (double) n / (ms * 1e-3) / 1e9);
        }
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    if (d) cudaFree(d);
    if (h) cudaFreeHost(h);
    cudaGetLastError();
    return best;
#else
    return 0;
#endif
}

// host read bandwidth with one thread per physical core (half the logical CPUs), 512 MiB, best of 2 (GB/s)
static double measure_dram_gbs() {
    const size_t n_words = (512ull << 20) / sizeof(uint64_t);
    std::vector<uint64_t> buf(n_words, 1);
    const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
    const unsigned nt = std::max(1u, hc / 2);
    double best = 0;
    for (int rep = 0; rep < 2; ++rep) {
        std::vector<uint64_t> sums(nt, 0);
        std::vector<std::thread> th;
        const auto t0 = std::chrono::steady_clock::now();
        for (unsigned t = 0; t < nt; ++t) {
            th.emplace_back([&, t] {
                const size_t a = n_words * t / nt, b = n_words * (t + 1) / nt;
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                size_t   i = a;
                for (; i + 4 <= b; i += 4) { s0 += buf[i]; s1 += buf[i + 1]; s2 += buf[i + 2]; s3 += buf[i + 3]; }
                for (; i < b; ++i) s0 += buf[i];
                sums[t] = s0 + s1 + s2 + s3;
            });
        }
        for (auto & x : th) x.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        uint64_t total = 0;
        for (uint64_t v : sums) total += v;
        if (s > 0 && total == n_words) best = std::max(best, (double) (n_words * sizeof(uint64_t)) / s / 1e9);
    }
    return best;
}

static std::string machine_key() {
    std::string gpu = "nogpu";
#ifdef GGML_USE_CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        int drv = 0;
        cudaDriverGetVersion(&drv);
        gpu = std::string(prop.name) + "|cc" + std::to_string(prop.major) + std::to_string(prop.minor) + "|drv" +
              std::to_string(drv) + "|pci" + std::to_string(prop.pciBusID);
    } else {
        cudaGetLastError();
    }
#endif
    return gpu + "|" + cpu_model() + "|ncpu" + std::to_string(std::thread::hardware_concurrency());
}

autotune_rates autotune_measure() {
    autotune_rates r;
    r.l3_domains = (int) l3_domains().size();
    const std::string key  = machine_key();
    const std::string path = cache_path();
    const char * mode = getenv("ALPHA_AUTOTUNE");
    const bool   remeasure = mode && strcmp(mode, "remeasure") == 0;
    if (!remeasure) {
        std::ifstream in(path);
        std::string   line;
        while (std::getline(in, line)) {   // "key \t pcie_gbs \t dram_gbs"; the last entry for the key wins
            const size_t t1 = line.find('\t');
            if (t1 == std::string::npos || line.compare(0, t1, key) != 0 || t1 != key.size()) continue;
            double p = 0, dr = 0;
            if (sscanf(line.c_str() + t1 + 1, "%lf\t%lf", &p, &dr) == 2 && p > 0) {
                r.pcie_gbs = p;
                r.dram_gbs = dr;
                r.cached   = true;
            }
        }
        if (r.cached) return r;
    }
    const auto t0 = std::chrono::steady_clock::now();
    r.pcie_gbs = measure_pcie_gbs();
    r.dram_gbs = measure_dram_gbs();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "[auto-tune] calibrated in %.1f s (cached in %s)\n", s, path.c_str());
#if defined(__linux__)
    const size_t sl = path.rfind('/');
    if (sl != std::string::npos) {   // mkdir -p of the cache directory
        const std::string dir = path.substr(0, sl);
        for (size_t p = 1; p <= dir.size(); ++p) {
            if (p == dir.size() || dir[p] == '/') mkdir(dir.substr(0, p).c_str(), 0755);
        }
    }
#endif
    if (r.pcie_gbs > 0) {
        std::ofstream out(path, std::ios::app);
        if (out) out << key << '\t' << r.pcie_gbs << '\t' << r.dram_gbs << '\n';
    }
    return r;
}

// bytes of one routed expert (gate + up + down) of the first MoE layer found in the model's (first) GGUF; 0 if unknown
static double expert_bytes(const char * model_path) {
    struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    struct gguf_context * gc = gguf_init_from_file(model_path, gp);
    if (!gc) return 0;
    double bytes = 0;
    const int64_t ak = gguf_find_key(gc, "general.architecture");
    const std::string arch = ak >= 0 ? gguf_get_val_str(gc, ak) : "";
    const int64_t ek = gguf_find_key(gc, (arch + ".expert_count").c_str());
    const int64_t n_expert = ek >= 0 ? (int64_t) gguf_get_val_u32(gc, ek) : 0;
    for (int il = 0; il < 8 && n_expert > 0 && bytes == 0; ++il) {
        double sum = 0;
        int    found = 0;
        for (const char * part : { "gate", "up", "down" }) {
            const std::string name = "blk." + std::to_string(il) + ".ffn_" + part + "_exps.weight";
            const int64_t     ti   = gguf_find_tensor(gc, name.c_str());
            if (ti >= 0) { sum += (double) gguf_get_tensor_size(gc, ti); ++found; }
        }
        if (found == 3) bytes = sum / (double) n_expert;
    }
    gguf_free(gc);
    return bytes;
}

static std::set<std::string> g_profile_keys;   // knobs the VRAM-class profile set (the auto-tune leaves them)

static void set_default(const char * key, const std::string & value, const std::string & reason) {
    if (const char * cur = getenv(key)) {
        fprintf(stderr, "[auto-tune] %s=%s (%s; auto-tune would pick %s: %s)\n", key, cur,
                g_profile_keys.count(key) ? "class profile" : "set by you", value.c_str(), reason.c_str());
        return;
    }
    setenv(key, value.c_str(), 0);
    fprintf(stderr, "[auto-tune] %s=%s: %s\n", key, value.c_str(), reason.c_str());
}

// Measured best Flash-Next configurations per VRAM class (decode A/V/R vs Strata 0.2.8 on the same GPU, 2 passes reversed). A class
// profile is applied before the auto-tune: an explicit env wins over it, it wins over the measured-rate rules.
// ALPHA_CLASS_PROFILE=0 skips it. No profile lowers AI_Q4X_TAIL_RANK below the engine's 8: tail rank 6 measured KL 0.0057 vs
// 0.0031 against the fp32 reference (top-1 98.47 vs 98.83%, worse on 24 of 25 prompts) for +1-2% decode. A profile only holds what was measured on that class: AI_Q4X_PF_SHARE_GBS is the
// prefill cost model's price of one copied expert, not a link speed (48 on the 3090 with PF_COPY_OVERLAP, tuned for
// TTFT; the measured link's 25 cost R TTFT 1.83 vs 1.46 s).
struct class_profile {
    const char * name;
    double       min_gib, max_gib;
    const char * evidence;
    std::vector<std::pair<const char *, const char *>> env;
};

static const std::vector<class_profile> & class_profiles() {
    static const std::vector<class_profile> p = {
        // AI_Q4X_PF_STREAM2=0 (one prefill expert buffer, not two): with the double buffer, 2.0.3 at bare -m on an RTX 4080
        // ran out of VRAM within 3 chat requests (captured graphs + a 1.1 GB second prefill scratch); with one, 37 requests
        // ran clean (2026-11-08). The 4080 production launcher has pinned it since 2026-11-03.
        // AI_GRAPH_MAX=8 (captured-round profiles, default 16): RTX 4080 soak of 62 chat requests (thinking on, max_tokens
        // 3072, two 8K prompts): with 16, 5 low-memory-valve evictions; with 8, none, at 102.4 vs 103.4 tok/s (noise);
        // 0 OOM / decode errors / aborts either way (2026-11-08).
        { "16 GB", 14.0, 20.0, "RTX 4080: 1.12x Strata 0.2.8 (2026-10-30)",
          { { "AI_Q4X_LEND", "1" }, { "ALPHA_AI_MTP_K", "3" }, { "AI_Q4X_MTP_MINP", "0.5" }, { "AI_Q4X_PF_STREAM2", "0" },
            { "AI_GRAPH_MAX", "8" } } },
        { "24 GB", 20.0, 28.0, "RTX 3090: 1.14-1.18x Strata 0.2.8 (2026-11-01, two runs)",
          { { "ALPHA_AI_MTP_K", "3" }, { "AI_Q4X_LEND", "1" }, { "AI_Q4X_MTP_MINP", "0.5" },
            { "AI_Q4X_PLE_GATE", "1" }, { "AI_Q4X_EXP_FORK", "2" }, { "AI_Q4X_ROUTE_RING", "1" },
            { "AI_Q4X_STREAM_SPLIT", "1" }, { "AI_Q4X_PF_COPY_OVERLAP", "1" }, { "AI_Q4X_PF_SHARE_GBS", "48" },
            { "AI_Q4X_PF_STREAM_MIN", "1024" }, { "AI_Q4X_PF_NBUF", "2" }, { "AI_Q4X_RESERVE_EXTRA_MB", "400" },
            { "AI_Q4X_ADMIT", "256" }, { "AI_Q4X_ADMIT_STAGE", "640" }, { "AI_Q4X_ADMIT_THR", "1.0" },
            { "AI_Q4X_ADMIT_RATIO", "1.1" }, { "AI_Q4X_LEND_ASYNC", "0" } } },
        // the 5090's best measured run minus two knobs: LEND_ASYNC (the first request after a lent prefill is not repeatable) and
        // TAIL_RANK 6 (+1-2%, pending the KL / lm-eval check). Draft min-p 0 measured +2.6% V over 0.5 here.
        { "32 GB", 28.0, 1e9, "RTX 5090: 1.10-1.12x Strata 0.2.8 (2026-11-01)",
          { { "ALPHA_AI_MTP_K", "3" }, { "AI_Q4X_MTP_MINP", "0" }, { "AI_Q4X_LEND", "1" }, { "AI_Q4X_LEND_ASYNC", "0" },
            { "AI_Q4X_PLE_GATE", "1" }, { "AI_Q4X_PF_COPY_OVERLAP", "1" }, { "AI_Q4X_PF_SHARE", "0" },
            { "AI_Q4X_ROUTE_RING", "1" }, { "AI_Q4X_STREAM_SPLIT", "1" }, { "AI_Q4X_EXP_FORK", "1" },
            { "AI_Q4X_ADMIT", "256" }, { "AI_Q4X_ADMIT_STAGE", "640" }, { "AI_Q4X_ADMIT_THR", "1.0" },
            { "AI_Q4X_ADMIT_RATIO", "1.1" } } },
    };
    return p;
}

static void apply_class_profile(size_t vram_total) {
    const char * mode = getenv("ALPHA_CLASS_PROFILE");
    if ((mode && strcmp(mode, "0") == 0) || vram_total == 0) return;
    const double gib = vram_total / 1073741824.0;
    for (const auto & cp : class_profiles()) {
        if (gib < cp.min_gib || gib >= cp.max_gib) continue;
        fprintf(stderr, "[profile] %s class (%.1f GiB): %s\n", cp.name, gib, cp.evidence);
        for (const auto & kv : cp.env) {
            const char * cur = getenv(kv.first);
            if (cur && strcmp(cur, kv.second) != 0) {
                fprintf(stderr, "[profile] %s=%s (set by you; the profile has %s)\n", kv.first, cur, kv.second);
                continue;
            }
            setenv(kv.first, kv.second, 1);
            g_profile_keys.insert(kv.first);
            fprintf(stderr, "[profile] %s=%s\n", kv.first, kv.second);
        }
        return;
    }
}

bool autotune_apply_qwen4exp(const char * model_path, size_t vram_total) {
    apply_class_profile(vram_total);
    const char * mode = getenv("ALPHA_AUTOTUNE");
    if (mode && strcmp(mode, "0") == 0) return false;
    const autotune_rates r = autotune_measure();
    fprintf(stderr, "[auto-tune] machine: PCIe host->GPU %.1f GB/s, DRAM read %.1f GB/s, %d L3 domain(s), VRAM %.1f GiB%s\n",
            r.pcie_gbs, r.dram_gbs, r.l3_domains, vram_total / 1073741824.0, r.cached ? " (cached measurement)" : "");
    char buf[160];

    // prefill: the CPU-share split prices a streamed expert at the link speed
    if (r.pcie_gbs > 0) {
        const long gbs = std::max(1L, std::lround(r.pcie_gbs));
        snprintf(buf, sizeof buf, "the prefill CPU-share split prices expert streaming at the measured PCIe speed "
                                  "(default assumes 12 GB/s)");
        set_default("AI_Q4X_PF_SHARE_GBS", std::to_string(gbs), buf);
        // on a fast link streaming every expert beats handing the CPU its share from 1024 rows up
        if (r.pcie_gbs >= 30) {
            set_default("AI_Q4X_PF_STREAM_MIN", "1024", "the link is fast (>= 30 GB/s): batches from 1024 rows stream "
                                                        "every expert instead of taking the CPU-share walk");
        }
    }

    // decode admission: the per-round copy budget = link speed x round time x 30%, in experts
    const double eb      = expert_bytes(model_path);
    const double round_s = (getenv("ALPHA_AUTOTUNE_ROUND_MS") ? atof(getenv("ALPHA_AUTOTUNE_ROUND_MS")) : 27.0) * 1e-3;   // fitted: 5090 (52 GB/s) -> ~225, 3090 (~25 GB/s) -> ~108, 4080 (12.9) -> ~56
    if (r.pcie_gbs > 0 && eb > 0) {
        const int by_link = std::clamp((int) std::lround(r.pcie_gbs * 1e9 * round_s * 0.3 / eb), 32, 256);
        // 20 GB and larger cards have the slots to take the full budget: the 3090 vs Strata measured 1.10x at
        // 256 against ~1.00x at the link's 107; the 5090's link already gives ~225. 16 GB cards keep the link value.
        const bool big   = vram_total >= (20ull << 30);
        const int  admit = big ? 256 : by_link;
        if (big && by_link < 256) {
            snprintf(buf, sizeof buf, "%d GB-class card: the full budget (the link alone gives %d)",
                     (int) std::lround(vram_total / 1073741824.0), by_link);
        } else {
            snprintf(buf, sizeof buf, "%.1f GB/s x %.0f ms round x 30%% / %.2f MB per expert", r.pcie_gbs, round_s * 1e3,
                     eb / 1e6);
        }
        set_default("AI_Q4X_ADMIT", std::to_string(admit), buf);
        set_default("AI_Q4X_ADMIT_STAGE", std::to_string((int) std::lround(admit * 2.5)), "2.5 x the admission budget");
        // a large budget can admit on weaker evidence: misses cost the critical path, admissions run async
        const char * thr = admit >= 192 ? "1.0" : admit >= 96 ? "1.5" : "2.0";
        const char * rat = admit >= 192 ? "1.1" : admit >= 96 ? "1.2" : "1.5";
        snprintf(buf, sizeof buf, "admission budget %d: %s", admit,
                 admit >= 96 ? "cheap async copies, admit on weaker evidence" : "the engine's default evidence");
        set_default("AI_Q4X_ADMIT_THR", thr, buf);
        set_default("AI_Q4X_ADMIT_RATIO", rat, buf);
    }

    // prefill slot lending: measured to pay on 16 GB-class and 32 GB-class cards, not on 24 GB (3090: +1-3% decode
    // for 20-40% more TTFT). The engine's own verify-width reserve is enough on both classes (a 400 MiB extra cost 3
    // slots per layer on the 4080 and the 5090, which never OOMed without it); the OOM that needs AI_Q4X_RESERVE_EXTRA_MB
    // was a 24 GB card with lending forced on, which auto-tune does not do.
    const bool card16 = vram_total > 0 && vram_total < (20ull << 30);
    const bool card32 = vram_total >= (28ull << 30);
    if (card16 || card32) {
        set_default("AI_Q4X_LEND", "1", card16 ? "16 GB-class card: lending buys ~20 more resident slots per layer"
                                               : "32 GB-class card: lending measured faster");
    }

    // CPU pool placement: on a multi-L3 CPU keep the first core of every L3 domain free for the launch and driver
    // threads (cross-domain traffic otherwise lands on the cores the pool spins on)
    const auto doms = l3_domains();
    if (doms.size() >= 2) {
        std::vector<int> keep;
        for (const auto & d : doms) {
            const std::string sib = read_first_line("/sys/devices/system/cpu/cpu" + std::to_string(d[0]) +
                                                    "/topology/thread_siblings_list");
            const std::vector<int> first = parse_cpu_list(sib.empty() ? std::to_string(d[0]) : sib);
            for (int c : d) {
                if (std::find(first.begin(), first.end(), c) == first.end()) keep.push_back(c);
            }
        }
        std::sort(keep.begin(), keep.end());
        snprintf(buf, sizeof buf, "%zu L3 domains: the first core of each stays free for launch/driver threads", doms.size());
        set_default("AI_Q4X_CPU_LIST", cpu_list_str(keep), buf);
    }

    // MTP draft min-p until the engine picks it per round: 0 on Blackwell (5090 V +2.6%), 0.5 elsewhere (3090, 4080)
    if (getenv("ALPHA_AI_MTP_K") && atoi(getenv("ALPHA_AI_MTP_K")) > 0) {
        int cc_major = 0;
#ifdef GGML_USE_CUDA
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) cc_major = prop.major; else cudaGetLastError();
#endif
        set_default("AI_Q4X_MTP_MINP", cc_major >= 12 ? "0" : "0.5",
                    cc_major >= 12 ? "Blackwell: cheap verify widths, keep every draft" : "verify widths cost more than "
                                                                                         "low-confidence drafts return");
    }
    return true;
}

}  // namespace alphaserve
