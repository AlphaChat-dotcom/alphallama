// Stream access and whole-round capture. Includes the ggml CUDA backend's context definition to
// reach its compute stream.
#include "kernels/kernels.h"

#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

namespace ai::kern {

void * backend_stream(ggml_backend * backend) {
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return ctx->stream();
}

void * copy_stream_create() {
    cudaStream_t s = nullptr;
    return cudaStreamCreate(&s) == cudaSuccess ? (void *) s : nullptr;
}

bool copy_h2d_async(void * stream, void * dst, const void * src, size_t n) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t) stream) == cudaSuccess;
}

bool memset_async(void * stream, void * dst, size_t n) {
    return cudaMemsetAsync(dst, 0, n, (cudaStream_t) stream) == cudaSuccess;
}

void * event_create() {
    cudaEvent_t e = nullptr;
    return cudaEventCreate(&e, cudaEventDisableTiming) == cudaSuccess ? (void *) e : nullptr;
}

void event_record(void * event, void * stream) {
    cudaEventRecord((cudaEvent_t) event, (cudaStream_t) stream);
}

void stream_wait_event(void * stream, void * event) {
    cudaStreamWaitEvent((cudaStream_t) stream, (cudaEvent_t) event, 0);
}

bool capture_begin(void * stream) {
    return cudaStreamBeginCapture((cudaStream_t) stream, cudaStreamCaptureModeRelaxed) == cudaSuccess;
}

bool capture_end(void * stream, Capture & c) {
    cudaGraph_t g = nullptr;
    if (cudaStreamEndCapture((cudaStream_t) stream, &g) != cudaSuccess) {
        return false;
    }
    c.graph = g;
    cudaGraphExec_t e = nullptr;
    if (cudaGraphInstantiate(&e, g, nullptr, nullptr, 0) != cudaSuccess) {
        return false;
    }
    c.exec = e;
    return true;
}

bool capture_launch(const Capture & c, void * stream) {
    return cudaGraphLaunch((cudaGraphExec_t) c.exec, (cudaStream_t) stream) == cudaSuccess;
}

void capture_free(Capture & c) {
    if (c.exec) {
        cudaGraphExecDestroy((cudaGraphExec_t) c.exec);
    }
    if (c.graph) {
        cudaGraphDestroy((cudaGraph_t) c.graph);
    }
    c = Capture{};
}

}  // namespace ai::kern
