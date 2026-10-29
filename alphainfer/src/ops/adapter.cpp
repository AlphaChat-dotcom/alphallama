// SPDX-License-Identifier: Apache-2.0
#include "ops/adapter.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"   // ggml_cgraph::uid (backend-capture mode)

#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace ai::adapter {

static constexpr size_t kCallMetaBytes  = 4 << 20;
static constexpr int    kCallGraphNodes = 512;
static constexpr size_t kRoundMetaBytes = 128 << 20;
static constexpr int    kSegmentNodes   = 4096;

static thread_local Recorder * g_active = nullptr;
static std::atomic<uint64_t>   g_uid{1ULL << 40};
static CaptureMode             g_mode = CaptureMode::Own;
// Sticky: set wherever capture_ok_ is cleared, so a server can refuse to serve after a
// round has degraded. Process-wide rather than per-Recorder because that is the question a
// /health endpoint asks, and inferring it from the replay counters cannot distinguish "no rounds
// yet" from "capture failed".
static std::atomic<bool>       g_cap_degraded{false};

void set_capture_mode(CaptureMode m) {
    g_mode = m;
    // Own capture wraps the backend's launches; the backend must not capture underneath it.
    if (m != CaptureMode::Backend) {
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    }
}

CaptureMode capture_mode() {
    return g_mode;
}

bool capture_degraded() {
    return g_cap_degraded.load();
}

void * stream(Device & dev) {
    return kern::backend_stream(dev.backend());
}

// ---- Recorder ---------------------------------------------------------------------------------

Recorder::Recorder(Device & dev) : dev_(dev), meta_(new char[kRoundMetaBytes]) {
    ggml_init_params p = { kRoundMetaBytes, meta_.get(), /*no_alloc=*/ true };
    ctx_ = ggml_init(p);
    if (!ctx_) {
        throw std::runtime_error("adapter: round ggml_init failed");
    }
    uid_ = g_uid.fetch_add(1 << 12);
}

Recorder::~Recorder() {
    if (g_active == this) {
        g_active = nullptr;
    }
    free_captures();
    ggml_free(ctx_);
}

void Recorder::free_captures() {
    for (kern::Capture & c : caps_) {
        kern::capture_free(c);
    }
    caps_.clear();
    spans_.clear();
}

void Recorder::begin() {
    if (g_active) {
        throw std::runtime_error("adapter: a round is already being recorded");
    }
    g_active = this;
}

void Recorder::end() {
    if (g_active == this) {
        g_active = nullptr;
    }
    // AI_ROUND_STATS=1: launchable node census per recorded round (pre-fusion upper bound)
    static const bool stats = getenv("AI_ROUND_STATS") != nullptr;
    if (stats) {
        std::map<std::string, int> cnt;
        int n_native = 0, n_host = 0, total = 0;
        for (const Segment & sg : segs_) {
            if (!sg.gf) {
                (sg.host ? n_host : n_native)++;
                continue;
            }
            for (int i = 0; i < ggml_graph_n_nodes(sg.gf); ++i) {
                const ggml_tensor * t = ggml_graph_node(sg.gf, i);
                if (t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
                    t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
                    continue;
                }
                std::string k = ggml_op_desc(t);
                if (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) {
                    k += std::string("(") + (t->src[0]->name[0] ? t->src[0]->name : "?") + ")";
                }
                cnt[k]++;
                total++;
            }
        }
        std::vector<std::pair<int, std::string>> v;
        for (auto & kv : cnt) {
            v.push_back({ kv.second, kv.first });
        }
        std::sort(v.rbegin(), v.rend());
        std::string line;
        for (auto & e : v) {
            line += " " + e.second + ":" + std::to_string(e.first);
        }
        fprintf(stderr, "[round-stats] %d ggml nodes, %d native, %d host steps |%s\n", total, n_native, n_host,
                line.c_str());
    }
    uint64_t u = uid_;
    for (Segment & s : segs_) {
        if (s.gf) {
            s.gf->uid = ++u;
        }
    }
}

Recorder::Segment & Recorder::ggml_segment() {
    if (segs_.empty() || !segs_.back().gf) {
        Segment s;
        s.gf = ggml_new_graph_custom(ctx_, kSegmentNodes, false);
        segs_.push_back(s);
    }
    return segs_.back();
}

void Recorder::add_native(std::function<void()> fn) {
    Segment s;
    s.native = std::move(fn);
    segs_.push_back(std::move(s));
}

void Recorder::add_host(std::function<void()> fn) {
    Segment s;
    s.native = std::move(fn);
    s.host   = true;
    segs_.push_back(std::move(s));
}

void Recorder::run_range(size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
        Segment & s = segs_[i];
        if (s.gf) {
            if (ggml_backend_graph_compute_async(dev_.backend(), s.gf) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("adapter: round segment compute failed");
            }
        } else {
            s.native();
        }
    }
}

// Liveness counters for the capture mode, printed at exit. Blocking calls per round and rounds
// recorded are the same across own|backend|none, so they cannot show whether AI_CAPTURE took
// effect; these can: a replayed round goes through capture_launch, an unreplayed one through run_range.
static std::atomic<long long> g_cap_replay{0}, g_cap_runrange{0};
struct ai_cap_report_t {
    ~ai_cap_report_t() {
        if (g_cap_replay.load() || g_cap_runrange.load()) {
            fprintf(stderr, "[ai-capture] mode=%s rounds replayed-from-graph=%lld run_range=%lld\n",
                    g_mode == CaptureMode::Own ? "own" : g_mode == CaptureMode::Backend ? "backend" : "none",
                    g_cap_replay.load(), g_cap_runrange.load());
        }
    }
};
static ai_cap_report_t ai_cap_report_inst;

void Recorder::launch() {
    const size_t n  = segs_.size();
    void *       st = stream(dev_);
    if (g_mode != CaptureMode::Own || !capture_ok_) {
        g_cap_runrange.fetch_add(1);
        run_range(0, n);
        return;
    }
    if (captured_) {
        g_cap_replay.fetch_add(1);
    }
    if (captured_) {
        size_t pos = 0;
        for (size_t c = 0; c < spans_.size(); ++c) {
            run_range(pos, spans_[c].first);   // host steps before this span
            if (!kern::capture_launch(caps_[c], st)) {
                throw std::runtime_error("adapter: captured round launch failed");
            }
            pos = spans_[c].second;
        }
        run_range(pos, n);
        return;
    }
    if (launches_++ == 0) {
        run_range(0, n);   // warm launch: pools and handles settle before capture
        return;
    }
    size_t i = 0;
    while (i < n) {
        if (segs_[i].host) {
            segs_[i].native();
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n && !segs_[j].host) {
            ++j;
        }
        if (!kern::capture_begin(st)) {
            // fall back to direct launches for good, and say so: the slowdown is otherwise invisible
            fprintf(stderr, "[capture] begin failed; falling back to direct launches for this engine\n");
            g_cap_degraded.store(true);
            capture_ok_ = false;
            free_captures();
            run_range(i, n);
            return;
        }
        run_range(i, j);
        kern::Capture c;
        if (!kern::capture_end(st, c)) {
            // Same fallback as a failed capture_begin rather than aborting (capture_end can fail at
            // large contexts). Capture records rather than executes, so the ops in [i, j) did not
            // run: recovery re-runs from i, as the begin path does.
            fprintf(stderr, "[capture] end failed; falling back to direct launches for this engine\n");
            g_cap_degraded.store(true);
            capture_ok_ = false;
            free_captures();
            run_range(i, n);
            return;
        }
        caps_.push_back(c);
        spans_.emplace_back(i, j);
        if (!kern::capture_launch(c, st)) {
            throw std::runtime_error("adapter: captured round launch failed");
        }
        i = j;
    }
    captured_ = true;
}

int Recorder::n_nodes() const {
    int n = 0;
    for (const Segment & s : segs_) {
        n += s.gf ? ggml_graph_n_nodes(s.gf) : (s.host ? 0 : 1);
    }
    return n;
}

Recorder * Recorder::active() {
    return g_active;
}

void native(Device & dev, std::function<void()> fn) {
    if (g_active) {
        g_active->add_native(std::move(fn));
    } else {
        fn();
    }
    (void) dev;
}

void host(Device & dev, std::function<void()> fn) {
    if (g_active) {
        g_active->add_host(std::move(fn));
    } else {
        fn();
    }
    (void) dev;
}

// ---- Call -------------------------------------------------------------------------------------

Call::Call(Device & dev) : dev_(dev), rec_(g_active) {
    if (rec_) {
        ctx_ = rec_->ctx_;
        return;
    }
    static thread_local std::vector<char> meta(kCallMetaBytes);
    ggml_init_params p = { kCallMetaBytes, meta.data(), /*no_alloc=*/ true };
    ctx_ = ggml_init(p);
    if (!ctx_) {
        throw std::runtime_error("adapter: ggml_init failed");
    }
}

Call::~Call() {
    if (!rec_) {
        ggml_free(ctx_);   // metadata buffer is ours; ggml_free does not release it
    }
}

ggml_tensor * Call::in(const Tensor & t, const char * name) {
    if (!t.data || t.buf < 0) {
        throw std::runtime_error(std::string("adapter: input without storage ") + (name ? name : "") + " " + t.shape_str());
    }
    ggml_tensor * g = ggml_new_tensor_4d(ctx_, (ggml_type) t.type, t.ne[0], t.ne[1], t.ne[2], t.ne[3]);
    for (int i = 0; i < 4; ++i) {
        g->nb[i] = t.nb[i];
    }
    g->data   = t.data;
    g->buffer = dev_.gbuf(t.buf);
    if (name) {
        ggml_set_name(g, name);
    }
    return g;
}

void Call::effect(ggml_tensor * node) {
    effects_.push_back(node);
}

void Call::late(ggml_tensor * node) {
    late_.push_back(node);
}

static int assign_memory(Device & dev, ggml_cgraph * gf, int from) {
    const int n = ggml_graph_n_nodes(gf);
    for (int i = from; i < n; ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->view_src) {
            if (!node->buffer) {
                if (ggml_backend_view_init(node) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("adapter: view_init failed");
                }
            }
            continue;
        }
        if (node->data) {
            continue;
        }
        Tensor shape;
        shape.type = node->type;
        for (int d = 0; d < 4; ++d) {
            shape.ne[d] = node->ne[d];
        }
        set_contiguous_strides(shape);
        int    buf  = -1;
        void * addr = dev.ws_alloc_raw(dev.alloc_size(shape), &buf);
        if (dev.ws_trace_on()) {
            dev.ws_trace_note(node->type, node->ne, std::string(ggml_op_desc(node)) + " " + node->name);
        }
        if (ggml_backend_tensor_alloc(dev.gbuf(buf), node, addr) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("adapter: tensor_alloc failed");
        }
    }
    return n;
}

void Call::launch(ggml_tensor * out) {
    std::vector<ggml_tensor *> order = effects_;
    if (out) {
        order.push_back(out);
    }
    order.insert(order.end(), late_.begin(), late_.end());

    if (rec_) {
        Recorder::Segment & seg = rec_->ggml_segment();
        // Op outputs are read by later Ops through aliasing leaves; mark them so the backend's
        // fusion passes never treat them as disposable intermediates.
        for (ggml_tensor * t : order) {
            ggml_set_output(t);
            ggml_build_forward_expand(seg.gf, t);
        }
        seg.allocated = assign_memory(dev_, seg.gf, seg.allocated);
        return;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx_, kCallGraphNodes, false);
    for (ggml_tensor * t : order) {
        ggml_build_forward_expand(gf, t);
    }
    assign_memory(dev_, gf, 0);
    if (ggml_backend_graph_compute_async(dev_.backend(), gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("adapter: graph compute failed");
    }
}

Tensor Call::to_tensor(ggml_tensor * g) const {
    Tensor t;
    t.type = g->type;
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = g->ne[i];
        t.nb[i] = g->nb[i];
    }
    t.data = g->data;
    const ggml_backend_buffer * b = g->view_src ? g->view_src->buffer : g->buffer;
    t.buf = dev_.buf_id(b);
    if (t.buf < 0) {
        throw std::runtime_error("adapter: output in an unknown buffer");
    }
    return t;
}

Tensor Call::run(ggml_tensor * out) {
    launch(out);
    return to_tensor(out);
}

void Call::run_effects_only() {
    launch(nullptr);
}

}  // namespace ai::adapter
