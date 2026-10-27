// Adapter: runs closed Ops as ggml expressions on ggml's CUDA kernels, interleaved with
// AlphaInfer's own native kernels.
//
// Private to src/ops (plus the Round API in ops.h). The Program and Engine never see a ggml type.
// Every input is an ai::Tensor over engine-owned memory; every node output is carved from the
// Device workspace; nothing is planned, cached or allocated by ggml.
//
// Modes:
//  * eager: each Op call builds a small expression and launches it at once.
//  * recording: while a Recorder is active, Op calls append to one round, an ordered list
//    of segments (a ggml subgraph, or one native kernel launch). Relaunching an unchanged round
//    replays it: the Recorder captures the whole segment list on the backend's stream after one
//    warm launch and replays that capture from then on (AI_OWN_CAPTURE=0 falls back to the
//    backend's own per-graph capture, which cannot include native kernels).
//    A round may also hold host segments (e.g. the streamed-offload routing read). They split the
//    round into device spans; each span is captured on its own and the host steps run between
//    the span launches, every launch.
#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "kernels/kernels.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;

namespace ai::adapter {

enum class CaptureMode { Own, Backend, None };
void        set_capture_mode(CaptureMode m);   // before the Device is initialised
CaptureMode capture_mode();
bool        capture_degraded();   // sticky: a round has fallen back to direct launches

class Recorder {
public:
    explicit Recorder(Device & dev);
    ~Recorder();
    Recorder(const Recorder &) = delete;
    Recorder & operator=(const Recorder &) = delete;

    void begin();          // make this the active recorder
    void end();            // stop recording (the round stays launchable)
    void launch();         // one asynchronous launch of the whole round

    int  n_nodes() const;
    int  n_segments() const { return (int) segs_.size(); }
    bool captured() const { return captured_; }
    static Recorder * active();

    void add_native(std::function<void()> fn);
    void add_host(std::function<void()> fn);

private:
    friend class Call;
    struct Segment {
        ggml_cgraph *         gf = nullptr;
        int                   allocated = 0;
        std::function<void()> native;
        bool                  host = false;
    };
    Segment & ggml_segment();
    void      run_range(size_t from, size_t to);
    void      free_captures();

    Device &                dev_;
    std::unique_ptr<char[]> meta_;
    ggml_context *          ctx_ = nullptr;
    std::vector<Segment>    segs_;
    uint64_t                uid_ = 0;
    int                     launches_ = 0;
    bool                    captured_ = false;
    bool                    capture_ok_ = true;
    std::vector<kern::Capture>             caps_;
    std::vector<std::pair<size_t, size_t>> spans_;   // device spans, one capture each
};

// Launch (eager) or record (inside a round) one native kernel call.
void   native(Device & dev, std::function<void()> fn);
void * stream(Device & dev);
// Run a host step now (eager) or at this point of every launch (inside a round).
void   host(Device & dev, std::function<void()> fn);

class Call {
public:
    explicit Call(Device & dev);
    ~Call();
    Call(const Call &) = delete;
    Call & operator=(const Call &) = delete;

    ggml_context * ctx() const { return ctx_; }

    // Descriptor over an engine tensor (a leaf).
    ggml_tensor * in(const Tensor & t, const char * name = nullptr);

    // Side-effect or extra-output nodes launched before the main output, in order.
    void effect(ggml_tensor * node);
    // Extra-output nodes launched after the main output, in order.
    void late(ggml_tensor * node);

    // Assign memory to every new node, launch (eager) or append (recording), and return `out`.
    Tensor run(ggml_tensor * out);
    void   run_effects_only();

    // An engine tensor for a node already launched by this call.
    Tensor to_public(ggml_tensor * g) const { return to_tensor(g); }

private:
    void   launch(ggml_tensor * out);
    Tensor to_tensor(ggml_tensor * g) const;

    Device &                   dev_;
    Recorder *                 rec_ = nullptr;
    ggml_context *             ctx_ = nullptr;
    std::vector<ggml_tensor *> effects_;
    std::vector<ggml_tensor *> late_;
};

}  // namespace ai::adapter
