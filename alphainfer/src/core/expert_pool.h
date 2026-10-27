// ExpertPool: device slots of routed-expert weights.
//
// The full expert banks live in registered host memory. Device slot banks have the same row layout
// as the host banks but hold `slots` experts instead of n_expert; an expert occupies one slot's
// row span. The Op reads a slot bank through the same kernel contract as a resident bank; only the
// expert index changes (expert -> slot).
//
// Shared banks: MoE layers whose {up, gate, down} banks have identical types and row shapes share
// one slot bank per group, with one exact LRU across the group. A layer that touches many experts
// borrows room from layers that touch few, while the pool bytes stay the same (n_slots per layer on
// average). Groups are capped in bytes (AI_POOL_SHARE_MIB, default 1024 per bank tensor), and a
// type key with more layers than one group holds is split round-robin across several groups so
// neighbouring hot layers spread out. AI_POOL_SHARE=0 restores one bank per layer.
#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ai {

class ExpertPool {
public:
    struct Bank {
        Tensor host;     // [.., .., n_expert] host-kept source (this layer)
        Tensor pool;     // [.., .., group slots] device slots (shared by the group)
    };
    struct Group {
        int                   slots = 0;
        std::vector<int32_t>  owner;       // slot -> layer*n_expert + expert, -1 if free
        std::vector<uint64_t> last_use;    // slot -> tick
        std::vector<int32_t>  prev, next;  // LRU list, head = least recently used
        int                   head = -1, tail = -1;
        std::vector<int>      layers;
        std::vector<char>     prefetched;   // slot filled by prefetch, not yet used
        void *                event   = nullptr;   // end of the group's latest prefetch copies
        bool                  pending = false;
    };
    struct Layer {
        Bank                  up, gate, down;
        int                   group = -1;
        std::vector<int32_t>  slot_of;     // expert -> slot in the group bank, -1 if absent
        mutable uint64_t              lookups = 0, misses = 0;   // per-layer, resettable
        mutable std::vector<uint64_t> seen;                      // expert -> lookups (census)
    };
    struct Stats {
        uint64_t lookups = 0, misses = 0, bytes = 0;
        uint64_t pf_issued = 0, pf_used = 0, pf_wasted = 0, pf_bytes = 0;
    };

    // `banks[l]` = {up, gate, down} host tensors of MoE layer l; n_slots = average slots per layer.
    void alloc(Device & dev, const std::vector<std::vector<Tensor>> & banks, int n_expert, int n_slots);

    int  n_slots() const { return n_slots_; }   // average per layer (and the per-call ensure cap)
    const Layer & layer(int l) const { return layers_.at((size_t) l); }
    const Group & group(int g) const { return groups_.at((size_t) g); }
    int           n_layers() const { return (int) layers_.size(); }
    int           n_groups() const { return (int) groups_.size(); }
    // Slots in layer l's bank, and whether a slot holds valid weights (for zero-weight fillers).
    int           bank_slots(int l) const { return groups_[(size_t) layers_[(size_t) l].group].slots; }
    bool          occupied(int l, int slot) const {
        return groups_[(size_t) layers_[(size_t) l].group].owner[(size_t) slot] >= 0;
    }
    void          reset_layer_counters() const;

    // Make `experts` resident for layer l (|experts| <= n_slots); fills slot_of. Slots pinned by
    // this call are never evicted by it; other layers' slots in the group may be.
    void ensure(Device & dev, int l, const std::vector<int32_t> & experts);

    // Speculatively make up to `max` of `experts` (in rank order) resident for layer l by copying
    // them on a separate copy stream, overlapping whatever the compute stream runs next. The next
    // ensure() on the group makes the compute stream wait for those copies first.
    void prefetch(Device & dev, int l, const std::vector<int32_t> & experts, int max);

    const Stats & stats() const { return stats_; }
    size_t        device_bytes() const { return device_bytes_; }

private:
    void upload(Device & dev, const Bank & b, int expert, int slot);
    void touch(Group & g, int slot);

    int  evict_one(Group & G, uint64_t now, int l, int e);
    void wait_pending(Device & dev, Group & G);

    std::vector<Layer> layers_;
    void *             copy_stream_ = nullptr;
    std::vector<Group> groups_;
    int                n_slots_  = 0;
    int                n_expert_ = 0;
    uint64_t           tick_     = 0;
    size_t             device_bytes_ = 0;
    Stats              stats_;
};

}  // namespace ai
