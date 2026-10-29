// SPDX-License-Identifier: Apache-2.0
// No-op stand-ins for the ggml-cuda expert-cache / DRE entry points that alphallama-serve calls, for a build
// without the CUDA backend (CMake adds this file only when GGML_CUDA is OFF). Without a GPU none of these paths
// does anything: every query reports "nothing cached / not handled", every setter is ignored, and the
// pointer-returning hooks return nullptr, which their callers treat as "no redirect".

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

extern "C" {

void   ggml_cuda_batch_mu_count_step(void) {}
void   ggml_cuda_batch_mu_report(void) {}
void   ggml_cuda_batch_mu_set_active(int) {}

size_t ggml_cuda_dre_borrow(size_t) { return 0; }
bool   ggml_cuda_dre_direct_weight(const ggml_tensor *, const ggml_tensor *) { return false; }
void   ggml_cuda_dre_pipe_prestart(void) {}
bool   ggml_cuda_dre_prefill_copy(void *, void *, const void *, size_t, int64_t, int64_t, size_t) { return false; }
void   ggml_cuda_dre_prefill_flush(void *) {}
void   ggml_cuda_dre_prefill_ids(const int32_t *, int, int, int, int) {}
void * ggml_cuda_dre_prefill_redirect(void *, void *) { return nullptr; }
bool   ggml_cuda_dre_prefill_stream_whole(void *, const void *, int) { return false; }
void   ggml_cuda_dre_prefill_x(void *, const ggml_tensor *, const void *, const int32_t *, int, int, int, int) {}
void   ggml_cuda_dre_preregister(const ggml_tensor *) {}
void   ggml_cuda_dre_return(void) {}
void   ggml_cuda_dre_return_begin(size_t) {}
void   ggml_cuda_dre_return_finish(void) {}
void   ggml_cuda_dre_set_prefill_tokens(int) {}
bool   ggml_cuda_dre_tier_busy(void) { return false; }
void   ggml_cuda_dre_tier_defer(void) {}
void   ggml_cuda_dre_tier_start(void) {}

bool   ggml_cuda_expert_pcache_copy(void *, void *, const void *, size_t, int64_t, int64_t, size_t) { return false; }
int    ggml_cuda_expert_pcache_dropped_count(void) { return 0; }
void   ggml_cuda_expert_pcache_dropped_get(int, int * seq, long long * eid, const void ** base, size_t * off, size_t * esz) {
    if (seq)  { *seq  = 0; }
    if (eid)  { *eid  = 0; }
    if (base) { *base = nullptr; }
    if (off)  { *off  = 0; }
    if (esz)  { *esz  = 0; }
}
void   ggml_cuda_expert_pcache_dropped_reset(void) {}
void   ggml_cuda_expert_pcache_flush(void *) {}
void * ggml_cuda_expert_pcache_get_extra(void) { return nullptr; }
size_t ggml_cuda_expert_pcache_get_stride(void) { return 0; }
void   ggml_cuda_expert_pcache_hitmiss(long long * hits, long long * misses) {
    if (hits)   { *hits   = 0; }
    if (misses) { *misses = 0; }
}
void   ggml_cuda_expert_pcache_maxlayer_get(int * max_misses, long long * max_bytes, long long * pool_bytes) {
    if (max_misses) { *max_misses = 0; }
    if (max_bytes)  { *max_bytes  = 0; }
    if (pool_bytes) { *pool_bytes = 0; }
}
void   ggml_cuda_expert_pcache_maxlayer_reset(void) {}
void   ggml_cuda_expert_pcache_pass_begin(void) {}
void   ggml_cuda_expert_pcache_prealloc(void) {}
void   ggml_cuda_expert_pcache_prefetch_union(int * out_unique, int * out_admitted, int * out_overflow) {
    if (out_unique)   { *out_unique   = 0; }
    if (out_admitted) { *out_admitted = 0; }
    if (out_overflow) { *out_overflow = 0; }
}
void * ggml_cuda_expert_pcache_redirect(void *, void *) { return nullptr; }
void   ggml_cuda_expert_pcache_set_cold_skip(int) {}
void   ggml_cuda_expert_pcache_set_cpu_inline(int, int) {}
void   ggml_cuda_expert_pcache_set_draft_norecord(int) {}
void   ggml_cuda_expert_pcache_set_group(const void *, int) {}
void   ggml_cuda_expert_pcache_set_hidden(const float *, int, int, int) {}
void   ggml_cuda_expert_pcache_set_ids(const int32_t *, int, int, int, int) {}
void   ggml_cuda_expert_pcache_set_ntokens(int, int, int, int) {}
void   ggml_cuda_expert_pcache_set_weights(const float *, int) {}
void   ggml_cuda_expert_pcache_set_window_admit(int) {}

void   ggml_cuda_init_expert_users(int) {}
void   ggml_cuda_pregate_set_gates(const float *, int, int, int) {}
void   ggml_cuda_pregate_set_norms(const float *, int, int, float) {}
void   ggml_cuda_set_expert_ctx(void *) {}
void   ggml_cuda_set_expert_user(int) {}
void   ggml_cuda_set_pool_reserve_bytes(size_t) {}
void   ggml_cuda_stream_tail_begin_forward(void) {}
bool   ggml_cuda_stream_tail_copy(void *, void *, const void *, size_t, const char *) { return false; }

}
