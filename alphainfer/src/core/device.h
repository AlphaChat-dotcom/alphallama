// AlphaInfer device: owns every byte of device memory the engine uses.
//
// Rules:
//  * all device memory is allocated by ResourceManager-level code before `finalize()`;
//    after finalize, any buffer allocation aborts.
//  * activations come from a bump workspace with scoped rollback (no allocator on the hot path).
//  * host<->device copies go through here and are the only synchronising calls.
//
// Buffers are ggml CUDA backend buffers so the adapter Ops can hand them to ggml's kernels.
#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <string>
#include <vector>

struct ggml_backend;
struct ggml_backend_buffer;

namespace ai {

class Device {
public:
    ~Device();

    bool   init(int device_index);
    void   finalize();                       // no buffer allocation after this point
    bool   finalized() const { return finalized_; }

    int    alloc_buffer(size_t bytes, const std::string & tag);
    void   free_buffer(int buf);   // release a buffer early (its id stays reserved; tensors on it are invalid after)
    // pinned host memory as a buffer id the Ops can read: kernels reach it through the unified address space (over
    // PCIe, no VRAM). Not counted in total_allocated(). For tensors read a few rows at a time (an embedding table).
    int    alloc_host_mapped_buffer(size_t bytes, const std::string & tag);
    void * buffer_base(int buf) const;
    size_t buffer_size(int buf) const;
    size_t alignment() const { return alignment_; }
    size_t total_allocated() const { return total_; }

    // bytes a tensor needs in a device buffer (quantised types carry kernel padding)
    size_t alloc_size(const Tensor & t) const;

    // Workspace: one buffer, bump allocation, rollback by mark.
    void   init_workspace(size_t bytes);
    Tensor ws_alloc(int type, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1);
    void * ws_alloc_raw(size_t bytes, int * buf_out);
    size_t ws_mark() const { return ws_off_; }
    void   ws_reset(size_t mark) { ws_off_ = mark; if (ws_trace_on_) ws_trace_pop(mark); }
    size_t ws_high_water() const { return ws_high_; }
    // AI_WS_REPORT: the phase that set the high-water mark (the engine tags its phases; build time, not replay)
    void         ws_tag(const char * t) { ws_tag_ = t; }
    const char * ws_high_tag() const { return ws_high_tag_.c_str(); }
    size_t ws_capacity() const { return ws_cap_; }
    void * ws_base() const { return ws_buf_ >= 0 ? buffer_base(ws_buf_) : nullptr; }
    // AI_WS_TRACE: name the most recent raw allocation (adapter graph nodes: op and shape)
    void   ws_trace_note(int type, const int64_t * ne, const std::string & label);
    bool   ws_trace_on() const { return ws_trace_on_; }

    struct Scope {
        Device & d;
        size_t   mark;
        explicit Scope(Device & dev) : d(dev), mark(dev.ws_mark()) {}
        ~Scope() { d.ws_reset(mark); }
    };

    // Copies. `upload`/`download` synchronise the stream and count as blocking calls.
    void mem_report(const char * when);   // AI_MEM_REPORT: tracked device buffers + VRAM free (no-op when unset)
    void upload(const Tensor & dst, const void * src, size_t offset, size_t n);
    void download(const Tensor & src, void * dst, size_t offset, size_t n);
    void synchronize();

    // Pinned host memory for ingress/egress (allocated before finalize) and non-blocking copies
    // on the compute stream. Async copies from pinned memory never synchronise.
    void * alloc_pinned(size_t bytes);
    void   upload_async(const Tensor & dst, const void * src, size_t offset, size_t n);
    void   download_async(const Tensor & src, void * dst, size_t offset, size_t n);

    // Every call that blocks the host on the device (synchronize, blocking copies).
    uint64_t blocking_calls() const { return n_block_; }

    ggml_backend *        backend() const { return backend_; }
    void *                compute_stream() const;   // the backend's stream (opaque)
    ggml_backend_buffer * gbuf(int buf) const { return bufs_.at(buf); }
    int                   buf_id(const ggml_backend_buffer * b) const;

    // VRAM report (free, total) in bytes
    void   memory_info(size_t * free_b, size_t * total_b) const;

private:
    ggml_backend *                     backend_ = nullptr;
    std::vector<ggml_backend_buffer *> bufs_;
    std::vector<std::string>           tags_;
    size_t                             alignment_ = 32;
    size_t                             total_     = 0;
    bool                               finalized_ = false;
    int                                device_    = 0;

    uint64_t                           n_block_ = 0;
    std::vector<ggml_backend_buffer *> pinned_;

    int    ws_buf_  = -1;
    size_t ws_cap_  = 0;
    size_t ws_off_  = 0;
    size_t ws_high_ = 0;
    const char * ws_tag_ = "";
    std::string  ws_high_tag_;
    // AI_WS_TRACE: the live allocations at the moment the high-water mark was set (offset, bytes, shape)
    struct WsRec { size_t off, bytes; int type; int64_t ne[4]; std::string tag; };
    bool               ws_trace_on_ = false;
    std::vector<WsRec> ws_live_, ws_peak_live_;
    void ws_trace_pop(size_t mark);
};

}  // namespace ai
