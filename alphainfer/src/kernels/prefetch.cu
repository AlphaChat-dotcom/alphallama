// L2 prefetch of expert weight spans.
//
// One kernel thread per (bank, slot) pair issues `prefetch.global.L2::evict_normal` on every
// 128-byte cache line of that expert's row span, so the SMs that later run the expert kernel find
// its weights in L2 instead of streaming them from HBM. Slots < 0 are skipped.
#include "kernels/kernels.h"

#include <cuda_runtime.h>

#include <cstdlib>

static constexpr long long kLine = 128;

__global__ static void l2_prefetch_k(const char * b0, const char * b1, const char * b2, long long s0, long long s1,
                                     long long s2, const int32_t * slots, long long n, long long stride) {
    const long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 3 * n) {
        return;
    }
    const long long j    = t % n;
    const int       bank = (int) (t / n);
    const int32_t   slot = slots[j];
    if (slot < 0) {
        return;
    }
    const char *    base = bank == 0 ? b0 : bank == 1 ? b1 : b2;
    const long long span = bank == 0 ? s0 : bank == 1 ? s1 : s2;
    const char *    p    = base + span * (long long) slot;
    if (stride <= 0) {   // control: same launch, no prefetch
        volatile char c = p[0];
        (void) c;
        return;
    }
    for (long long off = 0; off < span; off += kLine * stride) {
        asm volatile ("prefetch.global.L2::evict_normal [%0];" ::"l"(p + off) :);
    }
}

namespace ai::kern {

void l2_prefetch_experts(void * stream, const void * up, const void * gate, const void * down, int64_t span_up,
                         int64_t span_gate, int64_t span_down, const int32_t * slots, int64_t n) {
    static const long long stride = getenv("AI_L2_PF_STRIDE") ? atoll(getenv("AI_L2_PF_STRIDE")) : 1;
    const long long threads = 3 * n;
    l2_prefetch_k<<<1, (unsigned) threads, 0, (cudaStream_t) stream>>>(
        (const char *) up, (const char *) gate, (const char *) down, (long long) span_up, (long long) span_gate,
        (long long) span_down, slots, (long long) n, stride);
}

}  // namespace ai::kern
