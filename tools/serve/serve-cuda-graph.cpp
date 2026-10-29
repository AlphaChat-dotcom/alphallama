// SPDX-License-Identifier: Apache-2.0
#include "serve-cuda-graph.h"
#include <cstdlib>   // getenv
#include <cstdio>    // snprintf, fprintf

namespace alphaserve {

bool CUDAGraphPool::capture(int batch_size, cudaStream_t stream) {
    int bucket = next_pow2(batch_size);

    // End capture
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess || !graph) {
        fprintf(stderr, "[cuda-graph] capture failed for batch=%d: %s\n",
                bucket, cudaGetErrorString(err));
        return false;
    }

    // Instantiate
    cudaGraphExec_t exec = nullptr;
    err = cudaGraphInstantiate(&exec, graph, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cuda-graph] instantiate failed: %s\n", cudaGetErrorString(err));
        cudaGraphDestroy(graph);
        return false;
    }
    // Dump node/edge structure for offline inspection (ALPHA_GRAPH_DOT=1). Edges are the graph's
    // dependencies; two GEMM/splitK nodes with no path between them run concurrently under
    // cudaGraphLaunch and, if they share lt_workspace_, clobber each other.
    if (getenv("ALPHA_GRAPH_DOT")) {
        char path[80];
        snprintf(path, sizeof path, "/tmp/graph_b%d.dot", bucket);
        cudaGraphDebugDotPrint(graph, path, cudaGraphDebugDotFlagsVerbose);
        fprintf(stderr, "[cuda-graph] dumped %s (nodes+edges)\n", path);
    }

    // Store (replace if exists)
    auto it = graphs_.find(bucket);
    if (it != graphs_.end()) {
        cudaGraphExecDestroy(it->second.exec);
        cudaGraphDestroy(it->second.graph);
    }

    graphs_[bucket] = {graph, exec};
    fprintf(stderr, "[cuda-graph] captured graph for batch=%d (bucket=%d)\n", batch_size, bucket);
    return true;
}

bool CUDAGraphPool::replay(int batch_size, cudaStream_t stream) {
    int bucket = next_pow2(batch_size);
    auto it = graphs_.find(bucket);
    if (it == graphs_.end()) return false;

    cudaError_t err = cudaGraphLaunch(it->second.exec, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[cuda-graph] launch failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}

void CUDAGraphPool::destroy() {
    for (auto & [bucket, entry] : graphs_) {
        if (entry.exec) cudaGraphExecDestroy(entry.exec);
        if (entry.graph) cudaGraphDestroy(entry.graph);
    }
    graphs_.clear();
}

} // namespace alphaserve
