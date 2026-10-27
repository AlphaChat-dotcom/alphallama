// CpuFfn: dense FFN layers whose weights stay in host memory, for a dense model that misses the card by a little
// (Qwen3.8-27B UD-Q4_K_M on a 16 GB card). No weight is changed.
//
// Decode rows (T <= t_max; capturable, no host synchronisation): the CPU miss pool (offload/miss_pool) computes the
// whole FFN as one always-missed "expert" -- gate/up rows, SwiGLU, down -- through a mapped doorbell, with the same
// kernels as the doorbell MoE (kernels/doorbell.cu: the table maps expert 0 to "not resident", the weight is 1).
// Prefill chunks (T > t_max; eager only): each of the layer's three matrices is copied, in turn, into one VRAM staging
// buffer and run on the GPU (stream-ordered: the next copy waits for the previous matrix's kernel).
#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "ops/ops.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ai::offload {

class MissPool;

class CpuFfn {
public:
    // layers[i]: the i-th offloaded layer's weights (sh_gate / sh_up / sh_down, host tensors). threads: CPU pool workers
    // (0 = one per free physical core). Allocates its device buffers here (before Device::finalize).
    CpuFfn(Device & dev, const std::vector<op::MoeWeights> & layers, int64_t N, int64_t t_max, int threads);
    ~CpuFfn();

    int64_t t_max() const { return t_max_; }
    // FFN output f32 [N, T] (no residual) of offloaded layer `idx` for x f32 [N, T]
    Tensor  ffn(int idx, const op::MoeWeights & w, const Tensor & x);
    std::string stats_line() const;

private:
    Device &                  dev_;
    int64_t                   N_ = 0, t_max_ = 0;
    std::unique_ptr<MissPool> pool_;
    int32_t *                 table_ = nullptr;   // device i32 [1] = -1 (expert 0 never resident)
    int32_t *                 sel_   = nullptr;   // device i32 [t_max] = 0
    float *                   w_     = nullptr;   // device f32 [t_max] = 1
    unsigned *                counters_ = nullptr;   // device u32 [layers]
    int                       stage_buf_ = -1;
    void *                    stage_ = nullptr;
    size_t                    stage_bytes_ = 0;
    uint64_t                  n_decode_ = 0, n_staged_ = 0;
};

}  // namespace ai::offload
