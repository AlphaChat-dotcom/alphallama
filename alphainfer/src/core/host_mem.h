// SPDX-License-Identifier: Apache-2.0
// Pinned + mapped host memory for the offload path (RAM tier, admission staging, doorbells).
//
// Compaction caveat (seen on Linux 5.15 with proactive compaction on): pinned host memory of any kind -- cudaHostAlloc,
// or anonymous + cudaHostRegister -- is fuel for futile compaction. While the kernel is compaction-active it keeps trying
// to migrate pinned pages, fails on the extra reference, and every attempt sends TLB-shootdown IPIs to all cpus running
// this process, which can make every miss-pool phase ~2x slower. This allocator (separate anonymous mapping + register)
// is the mildest kind. MADV_HUGEPAGE on pinned memory is the worst (khugepaged keeps trying to collapse its 4 KiB
// parts), so AI_PINNED_THP=1 exists for experiments only. The robust fix is vm.compaction_proactiveness=0 (needs root);
// otherwise keep the pinned set no larger than it must be.
#pragma once

#include <cstddef>

namespace ai {

struct PinnedHost {
    void * ptr   = nullptr;
    size_t bytes = 0;      // requested size
    size_t span  = 0;      // mapped size (2 MiB multiple)
    double thp   = 0;      // fraction of the span backed by huge pages (from /proc/self/smaps)
    bool   registered = false;
};

// 2 MiB-aligned anonymous memory, faulted in at once (so its cost is paid at load), then cudaHostRegister(Portable |
// Mapped) unless register_cuda is false. AI_PINNED_THP=1 (and thp_hint) adds MADV_HUGEPAGE -- see above, experiments
// only. Returns ptr == nullptr on failure.
PinnedHost pinned_host_alloc(size_t bytes, bool register_cuda = true, bool thp_hint = true);
void       pinned_host_free(PinnedHost & m);
// Two-step form for a large fill: reserve (mapped, not faulted in -- the filler's reads fault it, in parallel), fill,
// then register. Registering after the fill pins pages that already exist instead of faulting 40 GB twice.
PinnedHost pinned_host_reserve(size_t bytes, bool thp_hint = true);
bool       pinned_host_register(PinnedHost & m);

// Fraction of [p, p + n) backed by AnonHugePages, from /proc/self/smaps (0 when unknown).
double     thp_fraction(const void * p, size_t n);

}  // namespace ai
