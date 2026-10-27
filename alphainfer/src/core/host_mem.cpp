#include "core/host_mem.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/mman.h>

namespace ai {

static constexpr size_t kHuge = 2ull << 20;

double thp_fraction(const void * p, size_t n) {
    std::ifstream f("/proc/self/smaps");
    if (!f) return 0;
    const uintptr_t a = (uintptr_t) p, b = a + n;
    std::string line;
    bool        in = false;
    size_t      vma_size = 0, huge = 0, total = 0;
    while (std::getline(f, line)) {
        uintptr_t lo = 0, hi = 0;
        if (sscanf(line.c_str(), "%lx-%lx ", &lo, &hi) == 2 && line.find(':') > line.find(' ')) {
            in = lo < b && hi > a;   // a VMA overlapping the range (ours is one VMA unless the kernel split it)
            vma_size = hi - lo;
            if (in) total += vma_size;
            continue;
        }
        if (in && line.rfind("AnonHugePages:", 0) == 0) {
            unsigned long kb = 0;
            sscanf(line.c_str() + 14, "%lu", &kb);
            huge += (size_t) kb * 1024;
        }
    }
    (void) vma_size;
    return total ? (double) huge / (double) total : 0.0;
}

bool pinned_host_register(PinnedHost & m) {
    if (!m.ptr || m.registered) return m.ptr != nullptr;
    if (cudaHostRegister(m.ptr, m.span, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    m.registered = true;
    m.thp        = thp_fraction(m.ptr, m.span);
    return true;
}

PinnedHost pinned_host_reserve(size_t bytes, bool thp_hint) {
    PinnedHost m;
    if (bytes == 0) return m;
    const size_t span = (bytes + kHuge - 1) / kHuge * kHuge;
    char * raw = (char *) mmap(nullptr, span + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return m;
    char * p = (char *) (((uintptr_t) raw + kHuge - 1) & ~(uintptr_t) (kHuge - 1));
    if (p > raw) munmap(raw, (size_t) (p - raw));
    char * end = raw + span + kHuge;
    if (end > p + span) munmap(p + span, (size_t) (end - (p + span)));
    static const bool env_thp = getenv("AI_PINNED_THP") && atoi(getenv("AI_PINNED_THP")) != 0;
    if (thp_hint && env_thp) madvise(p, span, MADV_HUGEPAGE);
    m.ptr   = p;
    m.bytes = bytes;
    m.span  = span;
    return m;
}

PinnedHost pinned_host_alloc(size_t bytes, bool register_cuda, bool thp_hint) {
    PinnedHost m;
    if (bytes == 0) return m;
    const size_t span = (bytes + kHuge - 1) / kHuge * kHuge;
    // over-map by one huge page, then trim to a 2 MiB-aligned span: THP needs aligned 2 MiB ranges inside the VMA
    char * raw = (char *) mmap(nullptr, span + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return m;
    char * p = (char *) (((uintptr_t) raw + kHuge - 1) & ~(uintptr_t) (kHuge - 1));
    if (p > raw) munmap(raw, (size_t) (p - raw));
    char * end = raw + span + kHuge;
    if (end > p + span) munmap(p + span, (size_t) (end - (p + span)));
    static const bool env_thp = getenv("AI_PINNED_THP") && atoi(getenv("AI_PINNED_THP")) != 0;   // opt-in (see host_mem.h)
    if (thp_hint && env_thp) madvise(p, span, MADV_HUGEPAGE);
    for (size_t o = 0; o < span; o += 4096) p[o] = 0;   // fault in now (a huge page per 2 MiB where the kernel can)
    m.ptr   = p;
    m.bytes = bytes;
    m.span  = span;
    m.thp   = thp_fraction(p, span);
    if (register_cuda) {
        if (cudaHostRegister(p, span, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            (void) cudaGetLastError();
            munmap(p, span);
            return PinnedHost{};
        }
        m.registered = true;
    }
    return m;
}

void pinned_host_free(PinnedHost & m) {
    if (!m.ptr) return;
    if (m.registered) cudaHostUnregister(m.ptr);
    munmap(m.ptr, m.span);
    m = PinnedHost{};
}

}  // namespace ai
