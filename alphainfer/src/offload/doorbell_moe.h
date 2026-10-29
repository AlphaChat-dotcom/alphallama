// SPDX-License-Identifier: Apache-2.0
// DoorbellMoe: model-independent MoE expert offload, for any model whose MoE layers are op::MoeWeights (qwen35moe and
// others).
//
// Per bank (one MoE layer, or the MTP head's), `slots` experts live in device slot banks plus k filler slots (copies
// of experts 0..k-1: valid weights that never hold a routed expert). A device table maps expert -> slot or -1.
//
// moe()  (T <= t_max rows; capturable, no host synchronisation): route on the device; the resolve kernel maps each
//        routed column to its slot (or a filler with weight 0) and rings the bank's doorbell in mapped host memory
//        with the misses; the routed experts run over the slot bank while the CPU pool (offload/miss_pool) computes
//        the misses; a wait kernel adds the CPU's weighted sum into the shared expert's output. Fast mode: the CPU
//        rounds differently from the GPU, and its sum is added in a different order.
// moe_bulk() (T > t_max rows: prefill chunks; eager only): host-synced. Resident columns run over the slot bank;
//        missed experts are copied, a staging group at a time, into a small staging bank and run from there.
// after_round(): admission between rounds (needs the previous round complete): counts every routed expert the
//        bells saw, copies the hottest missed ones into the coldest slots on a copy stream, publishes them in the
//        table at the next boundary (swap when score >= 2 and >= 1.5x the victim's; scores decay by 0.97).
#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/ops.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ai::offload {

class MissPool;

struct DoorbellConfig {
    int64_t          t_max     = 8;    // rows per doorbell call (decode / verify widths)
    std::string      profile;          // DREP routing profile (alphallama's format) for the initial residency
    int              threads   = 0;    // CPU pool threads (0 = from the topology plan, else 8)
    std::vector<int> cpus;             // explicit pool cpus (empty = plan_cpus)
    int              admit     = 32;   // admissions per round (0 = static residency)
    int              pf_slots  = 0;    // prefill staging slots per type group (0 = min(n_expert, 64))
};

class DoorbellMoe {
public:
    // banks[b]: bank b's host-kept MoeWeights (only up/gate/down_exps are read); slots[b]: resident slots of bank b
    // (clamped to n_expert). Every device buffer is allocated here (before Device::finalize).
    DoorbellMoe(Device & dev, const std::vector<op::MoeWeights> & banks, const std::vector<int> & slots, int n_expert,
                int k, int64_t N, const DoorbellConfig & cfg);
    ~DoorbellMoe();

    int64_t t_max() const { return cfg_.t_max; }
    int     n_banks() const { return (int) banks_.size(); }
    size_t  device_bytes() const { return device_bytes_; }

    // routed + shared expert outputs f32 [N, T] (not the residual)
    Tensor moe(int bank, const op::MoeWeights & w, const op::MoeParams & p, const Tensor & x);
    Tensor moe_bulk(int bank, const op::MoeWeights & w, const op::MoeParams & p, const Tensor & x);

    void after_round();
    // Back to the initial residency (the profile fill), scores cleared, before a new request (needs no round in
    // flight). The CPU computes missed experts with different rounding than the GPU, so with admission on, a
    // request's output depends on which experts earlier requests left resident; after a reset it depends on the
    // request alone. Only the slots that changed since the fill are copied again.
    void reset_residency();
    std::string stats_line() const;
    void        reset_stats();

    // bytes of one expert (gate + up + down) of bank b on the device
    static size_t expert_bytes(const op::MoeWeights & w);

private:
    struct Bank {
        Tensor gate, up, down, table;        // device: slot banks [.., .., n_slots + k], table i32 [E]
        Tensor host_gate, host_up, host_down;
        int    n_slots = 0;
        int    stage   = -1;                 // staging group (same types and row shapes)
        std::vector<int32_t> table_h;        // expert -> slot or -1
        std::vector<int32_t> owner;          // slot -> expert (-1 free / pending)
        std::vector<int32_t> owner0;         // owner after the initial fill (reset_residency)
        std::vector<float>   score;          // expert -> decayed use count
        unsigned             last_seq = 0;   // the bell's seq at the previous after_round
    };
    struct Stage { Tensor gate, up, down; int slots = 0; };
    struct Pending { int bank, slot, expert; int64_t job; };

    void admit_worker();

    Device &          dev_;
    DoorbellConfig    cfg_;
    int               E_ = 0, k_ = 0;
    int64_t           N_ = 0;
    std::vector<Bank> banks_;
    std::vector<Stage> stages_;
    std::unique_ptr<MissPool> pool_;
    unsigned *        counters_ = nullptr;   // device: one sequence counter per bell
    size_t            device_bytes_ = 0;

    // bulk path: pinned slot/mask upload regions (grown at a host-synchronised point)
    int32_t *         h_sel_ = nullptr;
    char *            h_up_  = nullptr;
    size_t            h_up_bytes_ = 0;

    // admission
    void *                                  copy_stream_ = nullptr;
    std::vector<char *>                     staging_;
    std::vector<void *>                     staging_ev_;
    std::unique_ptr<std::atomic<int64_t>[]> recorded_;
    int32_t *                               table_pin_ = nullptr;
    std::vector<Pending>                    pending_;
    std::thread                             admit_thr_;
    std::mutex                              admit_mu_;
    std::condition_variable                 admit_cv_;
    std::deque<Pending>                     admit_q_;
    bool                                    admit_stop_ = false;

    // stats (rounds seen by after_round)
    uint64_t n_cols_ = 0, n_miss_ = 0, n_admit_ = 0, n_publish_ = 0, n_bulk_cols_ = 0, n_bulk_miss_ = 0;
    uint64_t n_bulk_bytes_ = 0;
    uint64_t n_resets_ = 0, n_restored_ = 0;
};

}  // namespace ai::offload
