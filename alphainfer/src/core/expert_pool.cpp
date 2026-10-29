// SPDX-License-Identifier: Apache-2.0
#include "core/expert_pool.h"

#include "kernels/kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <tuple>

namespace ai {

static Tensor pool_like(Device & dev, int buf, size_t & off, const Tensor & host, int slots) {
    Tensor t;
    t.type  = host.type;
    t.ne[0] = host.ne[0];
    t.ne[1] = host.ne[1];
    t.ne[2] = slots;
    set_contiguous_strides(t);
    t.buf  = buf;
    t.data = (char *) dev.buffer_base(buf) + off;
    off += (dev.alloc_size(t) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
    if (off > dev.buffer_size(buf)) {
        throw std::runtime_error("ExpertPool: slot bank overruns its buffer");
    }
    return t;
}

static size_t bank_bytes(Device & dev, const Tensor & host, int slots) {
    Tensor probe = host;
    probe.ne[2] = slots;
    set_contiguous_strides(probe);
    return (dev.alloc_size(probe) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
}

void ExpertPool::alloc(Device & dev, const std::vector<std::vector<Tensor>> & banks, int n_expert, int n_slots) {
    n_slots_  = n_slots;
    n_expert_ = n_expert;
    const char *  es     = getenv("AI_POOL_SHARE");
    const bool    share  = !es || atoi(es) != 0;
    const char *  em     = getenv("AI_POOL_SHARE_MIB");
    const size_t  cap_b  = (size_t) (em ? atoll(em) : 1024) << 20;

    // key = (types, row shapes) of up, gate, down
    using Key = std::tuple<int, int, int, int64_t, int64_t, int64_t, int64_t>;
    std::map<Key, std::vector<int>> by_key;
    for (size_t l = 0; l < banks.size(); ++l) {
        const auto & b = banks[l];
        if (b.size() != 3) {
            throw std::runtime_error("ExpertPool: each layer needs up, gate, down");
        }
        Key k{ b[0].type, b[1].type, b[2].type, b[0].ne[0], b[0].ne[1], b[2].ne[0], b[2].ne[1] };
        if (b[1].ne[0] != b[0].ne[0] || b[1].ne[1] != b[0].ne[1]) {
            k = Key{ b[0].type, b[1].type, b[2].type, b[0].ne[0], b[0].ne[1], -1 - (int64_t) l, 0 };
        }
        by_key[k].push_back((int) l);
    }

    layers_.assign(banks.size(), Layer{});
    for (auto & kv : by_key) {
        const std::vector<int> & ls = kv.second;
        const auto & b0 = banks[(size_t) ls[0]];
        const size_t per_layer_max = std::max({ bank_bytes(dev, b0[0], n_slots), bank_bytes(dev, b0[1], n_slots),
                                                bank_bytes(dev, b0[2], n_slots) });
        const size_t per_group = share ? std::max<size_t>(1, cap_b / std::max<size_t>(1, per_layer_max)) : 1;
        const size_t n_groups  = (ls.size() + per_group - 1) / per_group;
        std::vector<std::vector<int>> members(n_groups);
        for (size_t i = 0; i < ls.size(); ++i) {
            members[i % n_groups].push_back(ls[i]);   // round-robin spreads neighbouring hot layers
        }
        for (auto & mem : members) {
            const int gi    = (int) groups_.size();
            const int slots = n_slots * (int) mem.size();
            size_t bytes = 0;
            for (const Tensor & h : b0) {
                bytes += bank_bytes(dev, h, slots);
            }
            const int buf = dev.alloc_buffer(bytes, "expert_pool.g" + std::to_string(gi));
            size_t off = 0;
            const Tensor up   = pool_like(dev, buf, off, b0[0], slots);
            const Tensor gate = pool_like(dev, buf, off, b0[1], slots);
            const Tensor down = pool_like(dev, buf, off, b0[2], slots);
            device_bytes_ += bytes;

            Group G;
            G.slots = slots;
            G.owner.assign((size_t) slots, -1);
            G.last_use.assign((size_t) slots, 0);
            G.prev.resize((size_t) slots);
            G.next.resize((size_t) slots);
            for (int s = 0; s < slots; ++s) {
                G.prev[(size_t) s] = s - 1;
                G.next[(size_t) s] = s + 1 < slots ? s + 1 : -1;
            }
            G.prefetched.assign((size_t) slots, 0);
            G.head   = 0;
            G.tail   = slots - 1;
            G.layers = mem;
            groups_.push_back(std::move(G));

            for (int l : mem) {
                const auto & b = banks[(size_t) l];
                Layer & L = layers_[(size_t) l];
                L.up    = { b[0], up };
                L.gate  = { b[1], gate };
                L.down  = { b[2], down };
                L.group = gi;
                L.slot_of.assign((size_t) n_expert, -1);
                L.seen.assign((size_t) n_expert, 0);
            }
        }
    }
    fprintf(stderr, "[ai] expert pool: %zu layers in %zu slot bank groups (%s)\n", layers_.size(), groups_.size(),
            share ? "shared across layers" : "one per layer");
}

void ExpertPool::reset_layer_counters() const {
    for (const Layer & L : layers_) {
        L.lookups = L.misses = 0;
        std::fill(L.seen.begin(), L.seen.end(), 0);
    }
}

void ExpertPool::upload(Device & dev, const Bank & b, int expert, int slot) {
    const size_t span = b.host.nb[2];
    dev.upload_async(b.pool, (char *) b.host.data + span * (size_t) expert, span * (size_t) slot, span);
    stats_.bytes += span;
}

int ExpertPool::evict_one(Group & G, uint64_t now, int l, int e) {
    const int victim = G.head;   // least recently used; pinned slots all sit behind it
    if (victim < 0 || G.last_use[(size_t) victim] == now) {
        throw std::runtime_error("ExpertPool: no evictable slot");
    }
    const int32_t old = G.owner[(size_t) victim];
    if (old >= 0) {
        layers_[(size_t) (old / n_expert_)].slot_of[(size_t) (old % n_expert_)] = -1;
    }
    if (G.prefetched[(size_t) victim]) {
        G.prefetched[(size_t) victim] = 0;
        ++stats_.pf_wasted;
    }
    G.owner[(size_t) victim]    = l * n_expert_ + e;
    G.last_use[(size_t) victim] = now;
    layers_[(size_t) l].slot_of[(size_t) e] = victim;
    touch(G, victim);
    return victim;
}

void ExpertPool::wait_pending(Device & dev, Group & G) {
    if (G.pending) {
        kern::stream_wait_event(dev.compute_stream(), G.event);
        G.pending = false;
    }
}

void ExpertPool::prefetch(Device & dev, int l, const std::vector<int32_t> & experts, int max) {
    Layer & L = layers_.at((size_t) l);
    Group & G = groups_[(size_t) L.group];
    if (!copy_stream_) {
        copy_stream_ = kern::copy_stream_create();
        if (!copy_stream_) {
            throw std::runtime_error("ExpertPool: copy stream creation failed");
        }
    }
    if (!G.event) {
        G.event = kern::event_create();
    }
    // a still-running earlier prefetch into this group must finish before slots are reassigned
    wait_pending(dev, G);
    const uint64_t now = ++tick_;
    int issued = 0;
    for (int32_t e : experts) {
        if (issued >= max) {
            break;
        }
        if (L.slot_of[(size_t) e] >= 0) {
            continue;
        }
        const int s = evict_one(G, now, l, e);
        G.prefetched[(size_t) s] = 1;
        for (const Bank * b : { &L.up, &L.gate, &L.down }) {
            const size_t span = b->host.nb[2];
            if (!kern::copy_h2d_async(copy_stream_, (char *) b->pool.data + span * (size_t) s,
                                      (const char *) b->host.data + span * (size_t) e, span)) {
                throw std::runtime_error("ExpertPool: prefetch copy failed");
            }
            stats_.pf_bytes += span;
        }
        ++issued;
        ++stats_.pf_issued;
    }
    if (issued > 0) {
        kern::event_record(G.event, copy_stream_);
        G.pending = true;
    }
}

void ExpertPool::touch(Group & G, int s) {
    if (G.tail == s) {
        return;
    }
    // unlink
    const int p = G.prev[(size_t) s], n = G.next[(size_t) s];
    if (p >= 0) {
        G.next[(size_t) p] = n;
    } else {
        G.head = n;
    }
    if (n >= 0) {
        G.prev[(size_t) n] = p;
    }
    // append at the tail (most recently used)
    G.prev[(size_t) s]      = G.tail;
    G.next[(size_t) s]      = -1;
    G.next[(size_t) G.tail] = s;
    G.tail                  = s;
}

void ExpertPool::ensure(Device & dev, int l, const std::vector<int32_t> & experts) {
    Layer & L = layers_.at((size_t) l);
    Group & G = groups_[(size_t) L.group];
    if ((int) experts.size() > std::min(n_slots_, G.slots)) {
        throw std::runtime_error("ExpertPool::ensure: more experts than slots");
    }
    wait_pending(dev, G);   // prefetch copies land before any read or overwrite of the bank
    const uint64_t now = ++tick_;
    // pin the requested set first (move to the tail) so eviction never takes one of them
    for (int32_t e : experts) {
        const int32_t s = L.slot_of[(size_t) e];
        if (s >= 0) {
            G.last_use[(size_t) s] = now;
            touch(G, s);
            if (G.prefetched[(size_t) s]) {
                G.prefetched[(size_t) s] = 0;
                ++stats_.pf_used;
            }
        }
    }
    for (int32_t e : experts) {
        ++stats_.lookups;
        ++L.lookups;
        ++L.seen[(size_t) e];
        if (L.slot_of[(size_t) e] >= 0) {
            continue;
        }
        ++stats_.misses;
        ++L.misses;
        const int victim = evict_one(G, now, l, e);
        upload(dev, L.up, e, victim);
        upload(dev, L.gate, e, victim);
        upload(dev, L.down, e, victim);
    }
}

}  // namespace ai
