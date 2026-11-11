// SPDX-License-Identifier: Apache-2.0
// KV store and recurrent-state slot pool.
//
// PagedKv: f16 cells per attention layer, laid out as fixed per-stream regions: stream r owns
// cells [r*R, (r+1)*R). Attention reads a stream's region as a strided view (ne3 = streams),
// so B streams share one attention launch. Regions are page-aligned (64 cells).
// TODO: a page table per region; it would change only `cell()`.
//
// StateSlots: per recurrent layer, fixed slots of conv history and recurrent state. A batch of B
// streams uses the consecutive slot block [s0, s0+B) as one [.., B] tensor.
#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ai {

class PagedKv {
public:
    static constexpr int kPage = 64;

    // `type` is the KV cell type, F16 by default. A quantised type trades cell precision for
    // capacity: 98304 cells x 10 attention layers is 3840 MiB at F16 against 1080 MiB at Q4_0.
    // The serve tool auto-downgrades for the same reason (serve-main.cpp: "Auto KV cache").
    void alloc(Device & dev, int n_layers, int64_t row_width, int64_t region_cells, int n_regions,
               int type = F16);

    int     new_seq();                     // lowest free region
    void    new_seq_at(int region);        // claim a specific region (batches use 0..B-1)
    void    free_seq(int seq);
    void    free_all();
    void    reserve(int seq, int64_t len); // checks len against the region
    int64_t length(int seq) const { return seqs_.at((size_t) seq).len; }
    void    set_length(int seq, int64_t len) { seqs_.at((size_t) seq).len = len; }
    int64_t cell(int seq, int64_t pos) const { return first_cell(seq) + pos; }
    int64_t first_cell(int seq) const { return (int64_t) seq * region_; }
    int64_t region_cells() const { return region_; }
    int     n_regions() const { return (int) seqs_.size(); }
    int64_t capacity_cells() const { return region_ * (int64_t) seqs_.size(); }

    const Tensor & k(int layer) const { return k_.at((size_t) layer); }
    const Tensor & v(int layer) const { return v_.at((size_t) layer); }

private:
    struct Seq { int64_t len = 0; bool live = false; };
    int64_t              region_ = 0;
    std::vector<Tensor>  k_, v_;
    std::vector<Seq>     seqs_;
};

class StateSlots {
public:
    void alloc(Device & dev, int n_layers, int64_t conv_bytes_per_slot, int64_t ssm_elems_per_slot, int n_slots);
    void zero(Device & dev, int slot);

    Tensor conv(int layer, int slot, int n = 1) const;   // f32 [(K-1)*C, n]
    Tensor ssm(int layer, int slot, int n = 1) const;    // f32 [Sv*Sv*Hv, n]
    int    n_slots() const { return n_slots_; }
    int    n_layers() const { return (int) conv_.size(); }
    size_t slot_bytes() const { return (size_t) (conv_n_ + ssm_n_) * 4 * conv_.size(); }   // one slot's state, all layers

private:
    int                 n_slots_ = 0;
    int64_t             conv_n_  = 0;
    int64_t             ssm_n_   = 0;
    std::vector<Tensor> conv_, ssm_;
};

}  // namespace ai
