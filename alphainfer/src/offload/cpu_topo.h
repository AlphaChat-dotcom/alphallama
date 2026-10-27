// Default CPU placement for the doorbell miss pool, from the sysfs topology.
//
// The launch thread owns its whole physical core, and no spinning worker may share it: a worker on the launch
// core's SMT sibling roughly halves decode speed, and leaving the pool unpinned is far worse. On a part with several
// L3 domains (Zen CCDs) the launch thread's whole L3 is kept free as well; on a single-L3 part (Intel hybrid) only
// its core is.
#pragma once

#include <string>
#include <vector>

namespace ai::offload {

struct CpuPlan {
    std::vector<int> launch;   // cpus the launch thread is pinned to (its physical core: all SMT siblings)
    std::vector<int> pool;     // pool cpus, poller first; empty = topology unknown (caller falls back)
    std::string      why;      // one line for the startup log
    int              n_phys = 0;   // pool candidates that are the first SMT thread of their core (one per core)
};

// cap: most pool threads (0 = default 32). Considers only the cpus the process may run on (sched_getaffinity).
CpuPlan plan_cpus(int cap);

// Pin the calling thread to `cpus` (no-op when empty or not Linux).
void pin_this_thread(const std::vector<int> & cpus);

// Move every existing thread of the process (the CUDA runtime's, the admission worker, the caller) off the pool cpus
// onto the rest of the allowed set; threads created later inherit their creator's affinity. Call before the pool
// starts (its threads pin themselves). Returns the threads moved. A stray thread landing on a spinning pool cpu
// makes every pool phase straggle, so run-to-run speed varies widely without this.
int confine_threads_off(const std::vector<int> & pool);

// "0-3,8" from a cpu list
std::string cpu_list_str(const std::vector<int> & cpus);

}  // namespace ai::offload
