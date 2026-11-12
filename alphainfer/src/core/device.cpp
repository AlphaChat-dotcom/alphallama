// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include <utility>
#include <algorithm>
#include "core/device.h"

#include "kernels/kernels.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ai {

// A throwaway descriptor for backend calls that take a ggml_tensor.
static ggml_tensor make_desc(const Tensor & t, ggml_backend_buffer * b) {
    ggml_tensor g{};
    g.type = (ggml_type) t.type;
    for (int i = 0; i < 4; ++i) {
        g.ne[i] = t.ne[i];
        g.nb[i] = t.nb[i];
    }
    g.data   = t.data;
    g.buffer = b;
    return g;
}

Device::~Device() {
    // AI_WS_REPORT: peak workspace actually used vs the capacity allocated.
    // Printed from the destructor so it covers the whole process; separate prefill from decode by
    // choosing the run (long prompt with -n 0 vs one-token prompt with -n 300), not by more code.
    if (getenv("AI_WS_REPORT") != nullptr && ws_cap_ > 0) {
        fprintf(stderr,
                "[ai-ws] workspace peak %.1f / %.1f MiB (%.1f%% used, %.1f MiB never touched) set in [%s]\n",
                (double) ws_high_ / 1048576.0, (double) ws_cap_ / 1048576.0,
                100.0 * (double) ws_high_ / (double) ws_cap_,
                (double) (ws_cap_ - ws_high_) / 1048576.0, ws_high_tag_.c_str());
    }
    if (ws_trace_on_ && !ws_peak_live_.empty()) {
        // the allocations live at the peak, largest first (the bump arena's peak = their extent)
        std::vector<WsRec> v = ws_peak_live_;
        std::sort(v.begin(), v.end(), [](const WsRec & a, const WsRec & b) { return a.bytes > b.bytes; });
        fprintf(stderr, "[ai-ws-trace] %zu allocations live at the peak (%.1f MiB) in [%s]:\n", v.size(),
                (double) ws_high_ / 1048576.0, ws_high_tag_.c_str());
        for (const WsRec & r : v) {
            fprintf(stderr, "[ai-ws-trace] %9.2f MiB @%8.1f type %2d ne [%lld %lld %lld %lld] %s\n", (double) r.bytes / 1048576.0,
                    (double) r.off / 1048576.0, r.type, (long long) r.ne[0], (long long) r.ne[1], (long long) r.ne[2],
                    (long long) r.ne[3], r.tag.c_str());
        }
    }
    for (auto * b : bufs_) {
        ggml_backend_buffer_free(b);
    }
    for (auto * b : pinned_) {
        ggml_backend_buffer_free(b);
    }
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

bool Device::init(int device_index) {
    device_  = device_index;
    backend_ = ggml_backend_cuda_init(device_index);
    if (!backend_) {
        fprintf(stderr, "[ai] ggml_backend_cuda_init(%d) failed\n", device_index);
        return false;
    }
    alignment_ = ggml_backend_buft_get_alignment(ggml_backend_cuda_buffer_type(device_index));
    return true;
}

void Device::finalize() {
    finalized_ = true;
    mem_report("finalize");
}

void Device::mem_report(const char * when) {
    // AI_MEM_REPORT: per-tag device footprint (=1: by tag prefix; =2: by full tag). Every buffer is
    // already tagged, so this is a report and not an estimate. Aggregated by tag prefix: the
    // per-buffer list is noise, the per-role totals are the answer.
    if (getenv("AI_MEM_REPORT") != nullptr) {
        std::vector<std::pair<std::string, size_t>> roll;
        size_t sum = 0;
        for (size_t i = 0; i < bufs_.size(); ++i) {
            const size_t b = bufs_[i] ? ggml_backend_buffer_get_size(bufs_[i]) : 0;
            sum += b;
            std::string k = tags_[i];
            const size_t dot = k.find('.');
            if (dot != std::string::npos && atoi(getenv("AI_MEM_REPORT")) < 2) { k = k.substr(0, dot); }
            bool hit = false;
            for (auto & e : roll) { if (e.first == k) { e.second += b; hit = true; break; } }
            if (!hit) { roll.emplace_back(k, b); }
        }
        std::sort(roll.begin(), roll.end(),
                  [](const std::pair<std::string, size_t> & a,
                     const std::pair<std::string, size_t> & b) { return a.second > b.second; });
        fprintf(stderr, "[ai-mem] %s: device buffers by role (%zu buffers, tracked total %.1f MiB):\n", when,
                bufs_.size(), (double) sum / 1048576.0);
        for (const auto & e : roll) {
            fprintf(stderr, "[ai-mem]   %-14s %9.1f MiB  %5.1f%%\n", e.first.c_str(),
                    (double) e.second / 1048576.0, sum ? 100.0 * (double) e.second / (double) sum : 0.0);
        }
        size_t fb = 0, tb = 0;
        memory_info(&fb, &tb);
        fprintf(stderr, "[ai-mem]   %-14s %9.1f MiB   <- total_ counter, for cross-check\n",
                "total_", (double) total_ / 1048576.0);
        fprintf(stderr, "[ai-mem]   VRAM free %.2f / %.2f GiB; anything above tracked total is CUDA "
                        "context + cuBLAS workspace + the allocator's own slack\n",
                (double) fb / 1073741824.0, (double) tb / 1073741824.0);
    }
}

int Device::alloc_buffer(size_t bytes, const std::string & tag) {
    if (finalized_) {
        fprintf(stderr, "[ai] FATAL: buffer allocation '%s' after finalize()\n", tag.c_str());
        abort();
    }
    ggml_backend_buffer * b = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device_), bytes);
    if (!b) {
        throw std::runtime_error("device buffer allocation failed: " + tag + " (" +
                                 std::to_string(bytes >> 20) + " MiB)");
    }
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bufs_.push_back(b);
    tags_.push_back(tag);
    total_ += bytes;
    return (int) bufs_.size() - 1;
}

void Device::free_buffer(int buf) {
    if (buf < 0 || buf >= (int) bufs_.size() || bufs_[(size_t) buf] == nullptr) return;
    synchronize();   // nothing queued may still use it
    total_ -= ggml_backend_buffer_get_size(bufs_[(size_t) buf]);
    ggml_backend_buffer_free(bufs_[(size_t) buf]);
    bufs_[(size_t) buf] = nullptr;
}

int Device::alloc_host_mapped_buffer(size_t bytes, const std::string & tag) {
    if (finalized_) {
        fprintf(stderr, "[ai] FATAL: buffer allocation '%s' after finalize()\n", tag.c_str());
        abort();
    }
    ggml_backend_buffer * b = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), bytes);
    if (!b) {
        throw std::runtime_error("pinned host buffer allocation failed: " + tag + " (" + std::to_string(bytes >> 20) +
                                 " MiB)");
    }
    ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bufs_.push_back(b);
    tags_.push_back(tag);
    return (int) bufs_.size() - 1;
}

void * Device::buffer_base(int buf) const {
    return ggml_backend_buffer_get_base(bufs_.at(buf));
}

size_t Device::buffer_size(int buf) const {
    return ggml_backend_buffer_get_size(bufs_.at(buf));
}

int Device::buf_id(const ggml_backend_buffer * b) const {
    for (size_t i = 0; i < bufs_.size(); ++i) {
        if (bufs_[i] == b) {
            return (int) i;
        }
    }
    return -1;
}

size_t Device::alloc_size(const Tensor & t) const {
    ggml_tensor g = make_desc(t, nullptr);
    return ggml_backend_buft_get_alloc_size(ggml_backend_cuda_buffer_type(device_), &g);
}

void Device::ws_trace_note(int type, const int64_t * ne, const std::string & label) {
    if (!ws_trace_on_ || ws_live_.empty()) return;
    WsRec & r = ws_live_.back();
    r.type = type;
    for (int i = 0; i < 4; ++i) r.ne[i] = ne[i];
    r.tag = label + " | " + r.tag;
    if (!ws_peak_live_.empty() && ws_peak_live_.back().off == r.off) ws_peak_live_.back() = r;
}

void Device::ws_trace_pop(size_t mark) {
    while (!ws_live_.empty() && ws_live_.back().off >= mark) ws_live_.pop_back();
}

void Device::init_workspace(size_t bytes) {
    ws_trace_on_ = getenv("AI_WS_TRACE") != nullptr;
    ws_buf_ = alloc_buffer(bytes, "workspace");
    ws_cap_ = bytes;
    ws_off_ = 0;
}

void * Device::ws_alloc_raw(size_t bytes, int * buf_out) {
    size_t off = (ws_off_ + alignment_ - 1) / alignment_ * alignment_;
    if (ws_measure_) {   // round-fit probe: count, never abort; past the capacity the pointer aliases the start
        ws_off_  = off + bytes;
        ws_need_ = std::max(ws_need_, ws_off_);
        *buf_out = ws_buf_;
        return (char *) buffer_base(ws_buf_) + ws_base_off_ + (ws_off_ <= ws_cap_ ? off : 0);
    }
    if (off + bytes > ws_cap_) {
        fprintf(stderr, "[ai] FATAL: workspace overflow: need %zu bytes at offset %zu MiB, capacity %zu MiB, phase \"%s\" "
                        "(peak so far %zu MiB in \"%s\")\n",
                bytes, off >> 20, ws_cap_ >> 20, ws_tag_ ? ws_tag_ : "?", ws_high_ >> 20, ws_high_tag_.c_str());
        abort();
    }
    ws_off_ = off + bytes;
    if (ws_trace_on_) ws_live_.push_back(WsRec{off, bytes, -1, {0, 0, 0, 0}, ws_tag_ ? ws_tag_ : ""});
    if (ws_off_ > ws_high_) {
        ws_high_      = ws_off_;
        ws_high_tag_  = ws_tag_;
        if (ws_trace_on_) ws_peak_live_ = ws_live_;
    }
    *buf_out = ws_buf_;
    return (char *) buffer_base(ws_buf_) + ws_base_off_ + off;
}

void Device::ws_redirect(int buf, size_t off, size_t cap) {
    if (ws_redirected()) throw std::runtime_error("ws_redirect: already redirected");
    if (off % alignment_ != 0) throw std::runtime_error("ws_redirect: borrowed range not aligned");
    ws_saved_ = WsSaved{ ws_buf_, ws_cap_, ws_off_, ws_high_, ws_base_off_, ws_high_tag_ };
    ws_buf_ = buf;
    ws_base_off_ = off;
    ws_cap_ = cap;
    ws_off_ = 0;
    ws_high_ = 0;
    ws_live_.clear();
}

void Device::ws_restore() {
    if (!ws_redirected()) return;
    ws_buf_      = ws_saved_.buf;
    ws_cap_      = ws_saved_.cap;
    ws_off_      = ws_saved_.off;
    ws_high_     = ws_saved_.high;
    ws_base_off_ = ws_saved_.base_off;
    ws_high_tag_ = ws_saved_.high_tag;
    ws_saved_    = WsSaved{};
    ws_live_.clear();
}

Tensor Device::ws_alloc(int type, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    Tensor t;
    t.type  = type;
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3;
    set_contiguous_strides(t);
    t.data = ws_alloc_raw(alloc_size(t), &t.buf);
    if (ws_trace_on_ && !ws_live_.empty()) {
        WsRec & r = ws_live_.back();
        r.type = type; r.ne[0] = n0; r.ne[1] = n1; r.ne[2] = n2; r.ne[3] = n3;
        if (!ws_peak_live_.empty() && ws_peak_live_.back().off == r.off) ws_peak_live_.back() = r;
    }
    return t;
}

void Device::upload(const Tensor & dst, const void * src, size_t offset, size_t n) {
    ++n_block_;
    ggml_tensor g = make_desc(dst, bufs_.at(dst.buf));
    ggml_backend_tensor_set(&g, src, offset, n);
}

void Device::download(const Tensor & src, void * dst, size_t offset, size_t n) {
    ++n_block_;
    ggml_backend_synchronize(backend_);
    ggml_tensor g = make_desc(src, bufs_.at(src.buf));
    ggml_backend_tensor_get(&g, dst, offset, n);
}

void Device::synchronize() {
    ++n_block_;
    ggml_backend_synchronize(backend_);
}

void * Device::alloc_pinned(size_t bytes) {
    if (finalized_) {
        fprintf(stderr, "[ai] FATAL: pinned allocation after finalize()\n");
        abort();
    }
    ggml_backend_buffer * b = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), bytes);
    if (!b) {
        throw std::runtime_error("pinned host allocation failed");
    }
    pinned_.push_back(b);
    return ggml_backend_buffer_get_base(b);
}

void Device::upload_async(const Tensor & dst, const void * src, size_t offset, size_t n) {
    ggml_tensor g = make_desc(dst, bufs_.at(dst.buf));
    ggml_backend_tensor_set_async(backend_, &g, src, offset, n);
}

void Device::download_async(const Tensor & src, void * dst, size_t offset, size_t n) {
    ggml_tensor g = make_desc(src, bufs_.at(src.buf));
    ggml_backend_tensor_get_async(backend_, &g, dst, offset, n);
}

void * Device::compute_stream() const {
    return kern::backend_stream(backend_);
}

void Device::memory_info(size_t * free_b, size_t * total_b) const {
    ggml_backend_cuda_get_device_memory(device_, free_b, total_b);
}

}  // namespace ai
