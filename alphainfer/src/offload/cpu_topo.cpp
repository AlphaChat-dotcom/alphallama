// SPDX-License-Identifier: Apache-2.0
#include "offload/cpu_topo.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#if defined(__linux__)
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <cstdlib>
#endif

namespace ai::offload {

static int read_int(const std::string & path, int def) {
    std::ifstream f(path);
    int v = def;
    if (!(f >> v)) return def;
    return v;
}

std::string cpu_list_str(const std::vector<int> & cpus) {
    std::vector<int> c(cpus);
    std::sort(c.begin(), c.end());
    std::string s;
    for (size_t i = 0; i < c.size();) {
        size_t j = i;
        while (j + 1 < c.size() && c[j + 1] == c[j] + 1) ++j;
        if (!s.empty()) s += ",";
        s += std::to_string(c[i]);
        if (j > i) s += "-" + std::to_string(c[j]);
        i = j + 1;
    }
    return s;
}

void pin_this_thread(const std::vector<int> & cpus) {
#if defined(__linux__)
    if (cpus.empty()) return;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int c : cpus) CPU_SET(c, &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
#else
    (void) cpus;
#endif
}

int confine_threads_off(const std::vector<int> & pool) {
#if defined(__linux__)
    if (pool.empty()) return 0;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0) return 0;
    for (int c : pool) CPU_CLR(c, &cs);
    if (CPU_COUNT(&cs) == 0) return 0;   // the pool is every allowed cpu: nowhere else to put them
    int n = 0;
    if (DIR * d = opendir("/proc/self/task")) {
        while (dirent * e = readdir(d)) {
            const int tid = atoi(e->d_name);
            if (tid > 0 && sched_setaffinity(tid, sizeof(cs), &cs) == 0) ++n;
        }
        closedir(d);
    }
    return n;
#else
    (void) pool;
    return 0;
#endif
}

int confine_threads_onto(const std::vector<int> & onto) {
#if defined(__linux__)
    if (onto.empty()) return 0;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int c : onto) CPU_SET(c, &cs);
    int n = 0;
    if (DIR * d = opendir("/proc/self/task")) {
        while (dirent * e = readdir(d)) {
            const int tid = atoi(e->d_name);
            if (tid > 0 && sched_setaffinity(tid, sizeof(cs), &cs) == 0) ++n;
        }
        closedir(d);
    }
    return n;
#else
    (void) onto;
    return 0;
#endif
}

CpuPlan plan_cpus(int cap) {
    CpuPlan plan;
#if defined(__linux__)
    if (cap <= 0) cap = 32;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { plan.why = "sched_getaffinity failed"; return plan; }
    struct Cpu { int id, core, l3, sib; };
    std::vector<Cpu> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        const std::string b = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/";
        const int pkg  = read_int(b + "topology/physical_package_id", 0);
        const int core = read_int(b + "topology/core_id", -1);
        if (core < 0) { plan.why = "no sysfs topology"; return plan; }
        const int l3 = read_int(b + "cache/index3/id", -1);
        cpus.push_back({ c, pkg * 100000 + core, l3 < 0 ? pkg : pkg * 100000 + l3, 0 });
    }
    if (cpus.size() < 2) { plan.why = "fewer than 2 allowed cpus"; return plan; }
    // SMT sibling rank inside each physical core (0 = first thread of the core)
    std::map<int, int> seen;
    for (auto & c : cpus) c.sib = seen[c.core]++;
    std::set<int> l3s;
    for (const auto & c : cpus) l3s.insert(c.l3);
    const Cpu &  first   = cpus.front();
    const bool   multi   = l3s.size() > 1;
    for (const auto & c : cpus) if (c.core == first.core) plan.launch.push_back(c.id);
    // [AI_Q4X_RESERVE_CORES] host cores for everything but the launch thread and the pool: from the launch L3 when the
    // pool skips it (multi-L3), else the next physical cores, which then leave the pool
    const int reserve = getenv("AI_Q4X_RESERVE_CORES") ? std::max(0, atoi(getenv("AI_Q4X_RESERVE_CORES"))) : 1;
    std::set<int> host_cores;
    for (const auto & c : cpus) {
        if ((int) host_cores.size() >= reserve) break;
        if (c.core == first.core || host_cores.count(c.core)) continue;
        if (multi && c.l3 != first.l3) continue;
        host_cores.insert(c.core);
    }
    if (multi && (int) host_cores.size() < reserve) {   // the launch L3 is one core: take the next ones anywhere
        for (const auto & c : cpus) {
            if ((int) host_cores.size() >= reserve) break;
            if (c.core != first.core) host_cores.insert(c.core);
        }
    }
    for (const auto & c : cpus) if (host_cores.count(c.core)) plan.host.push_back(c.id);
    std::vector<Cpu> cand;
    for (const auto & c : cpus) {
        if (c.core == first.core || host_cores.count(c.core)) continue;
        if (multi && c.l3 == first.l3) continue;
        cand.push_back(c);
    }
    // one thread per physical core first, the L3 domains interleaved, then the SMT siblings
    std::map<int, std::vector<Cpu>> by_l3;
    for (const auto & c : cand) by_l3[c.l3].push_back(c);
    std::vector<Cpu> ordered;
    for (int sib = 0; sib < 8; ++sib) {
        bool any = true;
        for (size_t k = 0; any; ++k) {
            any = false;
            for (auto & kv : by_l3) {
                std::vector<Cpu> lvl;
                for (const auto & c : kv.second) if (c.sib == sib) lvl.push_back(c);
                if (k < lvl.size()) { ordered.push_back(lvl[k]); any = true; }
            }
        }
    }
    for (size_t i = 0; i < ordered.size() && (int) plan.pool.size() < cap; ++i) plan.pool.push_back(ordered[i].id);
    for (const auto & c : cand) plan.n_phys += c.sib == 0 ? 1 : 0;
    char buf[256];
    snprintf(buf, sizeof buf, "%zu allowed cpus, %zu L3 domain(s): launch core %s, %s excluded, host threads on %s, pool %zu of %zu "
                              "candidates (cap %d)",
             cpus.size(), l3s.size(), cpu_list_str(plan.launch).c_str(), multi ? "its L3" : "only its core",
             plan.host.empty() ? "the launch core" : cpu_list_str(plan.host).c_str(), plan.pool.size(), cand.size(), cap);
    plan.why = buf;
#else
    (void) cap;
    plan.why = "not Linux";
#endif
    return plan;
}

}  // namespace ai::offload
