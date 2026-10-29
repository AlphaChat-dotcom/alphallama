// SPDX-License-Identifier: Apache-2.0
// Bridge from AlphaInfer's native producers to ggml's q8_1 staging registry.
//
// The registry lives in the CUDA backend (ggml-cuda/q8stage.cuh) and its consumer side is the GEMV's
// lookup. ggml's own producers are ggml nodes and reach it through the fusion-args struct; these are
// native kernels with no ggml node, so they call the registry directly. That is why the interface is
// a raw device pointer plus geometry rather than a ggml Q8_1 src1, which would mean changing ggml core.
//
// Registration is a host call made while a round is being captured, in graph order, from inside the
// native segment that also launches the producing kernel. The GEMV's lookup happens in the same
// capture pass and bakes the staged pointer into the recorded kernel node, so every replay reads it
// with no lookup at all. A cache that decides per call (like ALPHA_Q8_CACHE) cannot work under graph
// replay, because no decision runs on replay.
#include "kernels/kernels.h"

#include "ggml-cuda/q8stage.cuh"
#include "ggml-cuda/quantize.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ai::kern {

void q8_stage_clear() {
    // The registry's stats line is printed from ggml_cuda_q8_stage_reset(), which is called from
    // ggml-cuda.cu's graph compute path rather than from AlphaInfer's rounds, so report it from the
    // per-round clear instead. ALPHA_Q8_STAGE_STATS=N still controls the frequency.
    ggml_cuda_q8_stage_report();
    ggml_cuda_q8_stage_clear();
}

void q8_stage_invalidate(const void * data, size_t nbytes) {
    // node -2: no entry of ours carries it (ours are produced_at = -1), so every overlapping entry goes
    ggml_cuda_q8_stage_invalidate(data, nbytes, -2);
}

void q8_stage_put(const void * src1_f32, const void * q8, int64_t ne10_padded, int64_t ne11, int64_t ne12,
                  int64_t ne13, int64_t s11, int64_t s12, int64_t s13) {
    alpha_q8_staged e;
    e.src1_data = src1_f32;
    e.q8        = (const char *) q8;
    e.ne10      = ne10_padded;   // the lookup is keyed on the padded row length
    e.ne11      = ne11;
    e.ne12      = ne12;
    e.ne13      = ne13;
    e.s11       = s11;
    e.s12       = s12;
    e.s13       = s13;
    e.produced_at = -1;   // no ggml node produced this; nothing can mistake a node for its producer
    ggml_cuda_q8_stage_put(e);
}

// AI_Q8_STAGE=3: run the reference quantiser over the same activation and compare byte for byte.
// This checks that the staged copy is what the consumer would have produced itself, which the
// emit-only mode cannot show because nothing reads the copy there.
void q8_stage_verify(void * stream, const float * x, const void * mine, int64_t ncols, int64_t rows) {
    const int64_t bytes = ncols / 32 * rows * 36;
    void *        ref   = nullptr;
    if (cudaMalloc(&ref, (size_t) bytes) != cudaSuccess) {
        return;
    }
    quantize_row_q8_1_cuda(x, nullptr, (char *) ref, GGML_TYPE_Q8_0, ncols, ncols, ncols * rows, ncols * rows,
                           ncols, rows, 1, 1, (cudaStream_t) stream);
    cudaStreamSynchronize((cudaStream_t) stream);
    std::vector<unsigned char> a((size_t) bytes), b((size_t) bytes);
    cudaMemcpy(a.data(), mine, (size_t) bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), ref, (size_t) bytes, cudaMemcpyDeviceToHost);
    int64_t diff = 0, first = -1;
    for (int64_t i = 0; i < bytes; ++i) {
        if (a[(size_t) i] != b[(size_t) i]) {
            if (first < 0) {
                first = i;
            }
            diff++;
        }
    }
    // Print the first mismatches wherever they occur, not the first few calls: a mismatch can appear
    // only on graph replays, after a capture-time emission that matched.
    static long long checked = 0, bad = 0;
    checked++;
    if (diff != 0) {
        bad++;
    }
    static int printed = 0;
    if ((diff != 0 && printed < 6) || (checked % 500) == 0) {
        printed++;
        fprintf(stderr, "[q8-verify] call %lld, mismatching calls so far %lld\n", checked, bad);
        fprintf(stderr, "[q8-verify] ncols=%lld rows=%lld bytes=%lld differing=%lld first=%lld (block %lld, offset %lld: mine %u ref %u)\n",
                (long long) ncols, (long long) rows, (long long) bytes, (long long) diff, (long long) first,
                first < 0 ? -1 : (long long) (first / 36), first < 0 ? -1 : (long long) (first % 36),
                first < 0 ? 0u : a[(size_t) first], first < 0 ? 0u : b[(size_t) first]);
    }
    cudaFree(ref);
}

// AI_Q8_STAGE=4: fill the staged buffer with the reference quantiser instead of our epilogue,
// keeping the registration and the consumer path identical. If output is correct under 4 and wrong
// under 1, the fault is our emission at replay; if wrong under both, it is the plumbing.
void q8_stage_fill_reference(void * stream, const float * x, void * dst, int64_t ncols, int64_t rows) {
    quantize_row_q8_1_cuda(x, nullptr, (char *) dst, GGML_TYPE_Q8_0, ncols, ncols, ncols * rows, ncols * rows,
                           ncols, rows, 1, 1, (cudaStream_t) stream);
}

bool q8_stage_enabled() {
    return ggml_cuda_q8_stage_enabled();
}

const void * q8_stage_get(const void * src1_f32, int64_t ne10_padded, int64_t ne11, int64_t ne12, int64_t ne13,
                          int64_t s11, int64_t s12, int64_t s13) {
    if (!ggml_cuda_q8_stage_enabled()) {
        return nullptr;
    }
    return ggml_cuda_q8_stage_get(src1_f32, ne10_padded, ne11, ne12, ne13, s11, s12, s13);
}

}   // namespace ai::kern
