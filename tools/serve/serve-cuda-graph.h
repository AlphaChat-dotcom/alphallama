// SPDX-License-Identifier: Apache-2.0
#pragma once

// CUDA Graph Capture for FlashInfer Forward Pass
//
// Pre-captures the forward pass kernel sequence as a CUDA graph.
// Subsequent calls with the same batch size replay the graph
// instead of re-launching individual kernels.
//
// Piecewise capture at power-of-2 batch sizes: [1, 2, 4, 8, 16, ...]
// Non-power-of-2 batches pad to the next power of 2.

#include <cuda_runtime.h>
#include <unordered_map>
#include <cstdint>
#include <cstdio>

namespace alphaserve {

class CUDAGraphPool {
public:
    CUDAGraphPool() = default;
    ~CUDAGraphPool() { destroy(); }

    // Check if a graph exists for the given batch size
    bool has_graph(int batch_size) const {
        int bucket = next_pow2(batch_size);
        return graphs_.count(bucket) > 0;
    }

    // Capture a graph from the given stream
    // The caller must have already recorded the kernel sequence on the stream
    bool capture(int batch_size, cudaStream_t stream);

    // Replay a previously captured graph
    bool replay(int batch_size, cudaStream_t stream);

    void destroy();

    static int next_pow2(int n) {
        if (n <= 1) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    }

private:
    struct GraphEntry {
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
    };

    std::unordered_map<int, GraphEntry> graphs_;
};

} // namespace alphaserve
