// SPDX-License-Identifier: Apache-2.0
// RoundCache: decode rounds recorded once per profile, captured on their second launch and replayed after that, under
// a graph budget. Model-agnostic: every Program's engine keeps its rounds here, and the caller still decides when a
// round runs eagerly instead (offload, --eager).
#pragma once

#include "core/device.h"
#include "ops/ops.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>

namespace ai {

template <class Key, class Out> class RoundCache {
public:
    struct Entry {
        std::unique_ptr<op::Round> round;
        Out                        out;
        int                        launches = 0;
        uint64_t                   last_use = 0;   // LRU clock of its last launch (graph budget)
    };

    // Launch `key`'s round, recording it from `build` on first use (from a fixed workspace origin). *warm = it had
    // already been launched twice (the first launch runs direct, the second captures). `on_record` runs once per new
    // profile, after recording (the caller's timing and log line).
    const Out & launch(Device & dev, const Key & key, const std::function<Out()> & build, bool * warm, bool verbose,
                       const std::function<void(const Key &, const Entry &)> & on_record) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            budget(dev, key, verbose);
            dev.ws_reset(0);
            Entry r;
            r.round = std::make_unique<op::Round>(dev);
            r.round->begin();
            r.out = build();
            r.round->end();
            dev.ws_reset(0);
            it = map_.emplace(key, std::move(r)).first;
            if (on_record) on_record(key, it->second);
        }
        Entry & r = it->second;
        if (r.launches == 1) {
            budget(dev, key, verbose);   // this launch captures and instantiates: make room first
        }
        r.last_use = ++clock_;
        r.round->launch();
        *warm = ++r.launches > 2;
        return r.out;
    }

    const Entry * find(const Key & key) const {
        const auto it = map_.find(key);
        return it == map_.end() ? nullptr : &it->second;
    }
    size_t    size() const { return map_.size(); }
    long long evicted() const { return evicted_; }

private:
    // Graph budget. Every captured round profile holds its instantiated graph (~14 MiB on a 35B MoE), and the set of
    // profiles grows with every (envelope bucket, width, verify width, region) a server meets; unbounded, a long-running
    // server can run cudaGraphInstantiate out of memory. The bound is a count, least-recently-launched first
    // (AI_GRAPH_MAX, default 16). It is not a free-VRAM bound because destroyed graphs do not show up again in
    // cudaMemGetInfo (the memory stays cached below it and is reused by later graphs), so free VRAM would never recover
    // and every profile would be evicted. Free VRAM is used only as a safety valve (AI_GRAPH_RESERVE_MB, default 32)
    // that never evicts below 8 profiles. An evicted profile is re-recorded on its next use (~2 cold rounds).
    void budget(Device & dev, const Key & keep, bool verbose) {
        static const size_t max_graphs = getenv("AI_GRAPH_MAX") ? (size_t) atol(getenv("AI_GRAPH_MAX")) : 16;
        static const size_t reserve =
            (size_t) (getenv("AI_GRAPH_RESERVE_MB") ? atol(getenv("AI_GRAPH_RESERVE_MB")) : 32) << 20;
        size_t vfree = 0, vtot = 0;
        dev.memory_info(&vfree, &vtot);
        // the profile about to be recorded or captured counts against the cap
        const bool   low    = vfree < reserve;
        const size_t target =
            low ? std::min(max_graphs, std::max<size_t>(8, map_.size() > 0 ? map_.size() - 1 : 0)) : max_graphs;
        if (map_.size() < target || map_.empty()) {
            return;
        }
        dev.synchronize();   // nothing in flight may still reference a graph we destroy
        while (map_.size() >= target) {
            auto lru = map_.end();
            for (auto i = map_.begin(); i != map_.end(); ++i) {
                if (i->first == keep) continue;
                if (lru == map_.end() || i->second.last_use < lru->second.last_use) lru = i;
            }
            if (lru == map_.end()) {
                break;
            }
            map_.erase(lru);
            evicted_++;
        }
        static long long said = 0;
        if (evicted_ > said && (said == 0 || evicted_ >= 2 * said) && verbose) {   // 1, 2, 4, 8, ...
            said = evicted_;
            dev.memory_info(&vfree, &vtot);
            fprintf(stderr, "[graph-budget] evicted %lld profile(s) so far; %zu kept (cap %zu), free %.0f MiB%s\n",
                    evicted_, map_.size(), max_graphs, vfree / 1048576.0, low ? " (low-memory valve)" : "");
        }
    }

    std::map<Key, Entry> map_;
    uint64_t             clock_   = 0;
    long long            evicted_ = 0;
};

}  // namespace ai
