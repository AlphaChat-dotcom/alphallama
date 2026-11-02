// SPDX-License-Identifier: Apache-2.0
// Second compute stream for the ggml CUDA backend (AI_MOE_CONCURRENT).
//
// Small GEMVs lose a fixed few microseconds per kernel to ramp and drain, independent of grid size, which only
// concurrency hides. The MoE block has two branches that meet only at the merge -- the shared
// expert and the routed experts -- but ggml runs them one after the other on one stream. The backend
// already keeps per-stream contexts and pools (upstream's concurrency support); this file only lets the
// engine choose which of those streams the next graph launches on. The fork and join are event
// record/wait pairs, which a stream capture records as parallel branches of the round's graph.
#include "kernels/kernels.h"

#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

namespace ai::kern {

void backend_set_stream_no(ggml_backend * backend, int n) {
    auto * ctx          = (ggml_backend_cuda_context *) backend->context;
    ctx->curr_stream_no = n;
}

void * backend_stream_no(ggml_backend * backend, int n) {
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return ctx->stream(ctx->device, n);
}

namespace {
__global__ void copy_spans_k(const CopySpans s) {
    const int i = blockIdx.y;
    const float * src = s.src[i];
    float *       dst = s.dst[i];
    const long long n = s.n[i];
    for (long long j = (long long) blockIdx.x * blockDim.x + threadIdx.x; j < n; j += (long long) gridDim.x * blockDim.x) dst[j] = src[j];
}
}   // namespace

void copy_spans(void * stream, const CopySpans & s) {
    if (s.count <= 0) return;
    long long mx = 0;
    for (int i = 0; i < s.count; ++i) mx = s.n[i] > mx ? s.n[i] : mx;
    const unsigned gx = (unsigned) ((mx + 255) / 256 < 64 ? (mx + 255) / 256 : 64);
    copy_spans_k<<<dim3(gx > 0 ? gx : 1, (unsigned) s.count), 256, 0, (cudaStream_t) stream>>>(s);
}

}   // namespace ai::kern
