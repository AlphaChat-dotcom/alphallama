#include "core/kv.h"
#include "kernels/kernels.h"

#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ai {

static Tensor carve(Device & dev, int buf, size_t & off, int type, int64_t n0, int64_t n1) {
    Tensor t;
    t.type  = type;
    t.ne[0] = n0;
    t.ne[1] = n1;
    set_contiguous_strides(t);
    const size_t a = (dev.alloc_size(t) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
    t.buf  = buf;
    t.data = (char *) dev.buffer_base(buf) + off;
    off += a;
    return t;
}

void PagedKv::alloc(Device & dev, int n_layers, int64_t row_width, int64_t region_cells, int n_regions,
                    int type) {
    region_ = (region_cells + kPage - 1) / kPage * kPage;
    const int64_t cells = region_ * n_regions;
    // A quantised row must be a whole number of blocks, or every per-cell view is misaligned and
    // the failure surfaces far from here as wrong numbers rather than an error. Refuse instead.
    const int64_t blk = (int64_t) block_size(type);
    if (blk > 1 && row_width % blk != 0) {
        throw std::runtime_error("PagedKv: row_width " + std::to_string(row_width) +
                                 " is not a multiple of the block size " + std::to_string(blk) +
                                 " for KV type " + std::to_string(type));
    }
    Tensor probe;
    probe.type  = type;
    probe.ne[0] = row_width;
    probe.ne[1] = cells;
    set_contiguous_strides(probe);
    const size_t per = (dev.alloc_size(probe) + dev.alignment() - 1) / dev.alignment() * dev.alignment();
    const int buf = dev.alloc_buffer(per * 2 * (size_t) n_layers, "kv");
    size_t off = 0;
    for (int l = 0; l < n_layers; ++l) {
        k_.push_back(carve(dev, buf, off, type, row_width, cells));
        v_.push_back(carve(dev, buf, off, type, row_width, cells));
    }
    seqs_.assign((size_t) n_regions, Seq{});
    // Log the type and the bytes actually allocated, so a type parameter that fails to take effect
    // is visible here.
    fprintf(stderr,
            "[ai] kv: type=%d, %d layers x 2 x [%lld,%lld] = %.1f MiB total (%.0f B/cell/layer)\n",
            type, n_layers, (long long) row_width, (long long) cells,
            (double) (per * 2 * (size_t) n_layers) / 1048576.0,
            n_layers ? (double) (per * 2) / (double) cells : 0.0);
}

int PagedKv::new_seq() {
    for (size_t i = 0; i < seqs_.size(); ++i) {
        if (!seqs_[i].live) {
            seqs_[i] = Seq{};
            seqs_[i].live = true;
            return (int) i;
        }
    }
    throw std::runtime_error("PagedKv: no free region");
}

void PagedKv::new_seq_at(int region) {
    Seq & s = seqs_.at((size_t) region);
    if (s.live) {
        throw std::runtime_error("PagedKv: region already in use");
    }
    s = Seq{};
    s.live = true;
}

void PagedKv::free_seq(int seq) {
    seqs_.at((size_t) seq) = Seq{};
}

void PagedKv::free_all() {
    for (auto & s : seqs_) {
        s = Seq{};
    }
}

void PagedKv::reserve(int seq, int64_t len) {
    if (!seqs_.at((size_t) seq).live) {
        throw std::runtime_error("PagedKv: reserve on a free region");
    }
    if (len > region_) {
        throw std::runtime_error("PagedKv: sequence longer than its region (raise --ctx)");
    }
}

void StateSlots::alloc(Device & dev, int n_layers, int64_t conv_n, int64_t ssm_n, int n_slots) {
    n_slots_ = n_slots;
    conv_n_  = conv_n;
    ssm_n_   = ssm_n;
    const size_t per = (size_t) (conv_n + ssm_n) * 4 * (size_t) n_slots + 4 * dev.alignment();
    const int buf = dev.alloc_buffer(per * (size_t) n_layers, "state");
    size_t off = 0;
    for (int l = 0; l < n_layers; ++l) {
        conv_.push_back(carve(dev, buf, off, F32, conv_n, n_slots));
        ssm_.push_back(carve(dev, buf, off, F32, ssm_n, n_slots));
    }
}

Tensor StateSlots::conv(int layer, int slot, int n) const {
    if (slot < 0 || n < 1 || slot + n > n_slots_) {
        throw std::runtime_error("StateSlots: slot block out of range");
    }
    const Tensor & t = conv_.at((size_t) layer);
    const size_t   r = t.nb[1] * (size_t) n;
    return t.view(conv_n_, n, 1, 1, t.nb[1], r, r, t.nb[1] * (size_t) slot);
}

Tensor StateSlots::ssm(int layer, int slot, int n) const {
    if (slot < 0 || n < 1 || slot + n > n_slots_) {
        throw std::runtime_error("StateSlots: slot block out of range");
    }
    const Tensor & t = ssm_.at((size_t) layer);
    const size_t   r = t.nb[1] * (size_t) n;
    return t.view(ssm_n_, n, 1, 1, t.nb[1], r, r, t.nb[1] * (size_t) slot);
}

// A device-side fill on the compute stream, ordered after any round still reading the slot. Uploading host zeros
// instead moves ~157 MB per reset on the 27B (48 layers x (3 MiB ssm + 120 KiB conv)), and a request resets twice,
// which dominates a short prompt's prefill on a slow PCIe link. Zero bits are 0.0f, so the result is identical.
// AI_ZERO_UPLOAD=1 uses the blocking upload instead (not ordered against the compute stream).
void StateSlots::zero(Device & dev, int slot) {
    static const bool upload = getenv("AI_ZERO_UPLOAD") && atoi(getenv("AI_ZERO_UPLOAD")) != 0;
    void * st = upload ? nullptr : dev.compute_stream();
    std::vector<float> zc, zs;
    for (size_t l = 0; l < conv_.size(); ++l) {
        const Tensor c = conv((int) l, slot), s = ssm((int) l, slot);
        if (st && kern::memset_async(st, c.data, (size_t) conv_n_ * 4) && kern::memset_async(st, s.data, (size_t) ssm_n_ * 4)) {
            continue;
        }
        if (zc.empty()) {
            zc.assign((size_t) conv_n_, 0.0f);
            zs.assign((size_t) ssm_n_, 0.0f);
        }
        dev.upload(c, zc.data(), 0, zc.size() * 4);
        dev.upload(s, zs.data(), 0, zs.size() * 4);
    }
}

}  // namespace ai
