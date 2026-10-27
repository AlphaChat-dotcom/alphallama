// Generic doorbell offload kernels (offload/doorbell_moe.h): the device half of a model-independent MoE offload.
// Unlike kernels/offload.cu (qwen4exp's window protocol), every bell keeps its own sequence counter in device memory:
// a ring advances it, the wait reads it back, so a captured graph replays with fresh numbers and a bell may ring any
// number of times per round (the MTP head's chained calls), with no window-end step.
#pragma once

#include <cstdint>

namespace ai::kern {

// One block. For every routed column i = t*k + j: slots[i] = table[sel[i]] when resident, else filler slot
// filler0 + j (distinct within a row, always valid weights); w_masked[i] = w[i] for resident columns, 0 for missed
// ones (their filler outputs drop out of the weighted sum); miss_bell[i] = expert id (missed) or -(id+2) (resident);
// w_bell[i] = w[i]; x copied to the bell. Then ctr = ++*counter, n_tok = T, seq = ctr.
void door_resolve_ring(void * stream, const int32_t * table, const int32_t * sel, const float * w, const float * x,
                       int64_t N, int k, int64_t T, int32_t filler0, int32_t * slots, float * w_masked,
                       int32_t * miss_bell, float * w_bell, float * x_bell, int32_t * n_tok_bell, unsigned * seq_bell,
                       unsigned * counter);

// One block per token: if any of the token's columns missed, wait for done >= *counter, then out[:, t] += yp[:, t]
// (the CPU's weighted sum over the token's missed columns).
void door_wait_add(void * stream, const unsigned * done_bell, const unsigned * counter, const int32_t * slots,
                   int32_t filler0, const float * yp_bell, float * out, int64_t N, int k, int64_t T);

}  // namespace ai::kern
