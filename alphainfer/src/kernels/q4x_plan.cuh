// The decode grouped experts' ownership PLAN (q4x_experts.cu) and the one-block routine that builds it, shared by
// plan_k (q4x_experts.cu) and the fused route + ring kernel (q4x_route.cu, AI_Q4X_ROUTE_RING: its last block builds the
// plan, so the experts call skips plan_k). Device code only; each TU gets its own copy.
#pragma once

#include <cstdint>

namespace q4xplan {

constexpr int kMaxCols  = 160;   // k * T <= 160 (= q4x_experts.cu's kMaxCols)
constexpr int kPlanCols = 16;    // columns one owner can serve (= the most window rows sharing a slot; T <= 16)
struct Own { int ch, ne, cs[kPlanCols]; };
struct Plan { int count, pad[3]; Own own[kMaxCols]; };

// The window's plan, built by ONE block (blockDim.x >= n): the lowest column routed to each slot in [lo, hi) owns it;
// plan->own[i] = { slot - lo, ncols, the columns in ascending order } for i < plan->count. Columns outside [lo, hi)
// (missed: slot >= filler0) own nothing. The owner ORDER is whatever the atomic gives -- each column's result does not
// depend on it. ids[(c % k) + (c / k) * ids_stride] is column c's slot. sid / cnt: the caller's shared memory.
__device__ __forceinline__ void plan_build(const int32_t * ids, int ids_stride, int k, int n, int lo, int hi,
                                           Plan * __restrict__ plan, int * sid, int & cnt) {
    const int c = threadIdx.x;
    if (c == 0) cnt = 0;
    if (c < n) sid[c] = __ldcg(ids + (c % k) + (c / k) * ids_stride);   // L2: other blocks of the ring kernel wrote them
    __syncthreads();
    if (c < n) {
        const int ch = sid[c];
        bool owner = ch >= lo && ch < hi;
        for (int d = 0; d < c && owner; ++d) owner = sid[d] != ch;
        if (owner) {
            const int i = atomicAdd(&cnt, 1);
            Own &     o = plan->own[i];
            int       ne = 0;
            for (int d = c; d < n && ne < kPlanCols; ++d) if (sid[d] == ch) o.cs[ne++] = d;
            o.ch = ch - lo;
            o.ne = ne;
        }
    }
    __syncthreads();
    if (c == 0) plan->count = cnt;
}

}  // namespace q4xplan
