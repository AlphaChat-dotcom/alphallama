#pragma once
#include "common.cuh"

// Device-resolved experts for offloaded-MoE decode (ALPHA_DRE). See dre.cu.

// true when ALPHA_DRE=1 (read once)
bool ggml_cuda_dre_enabled();

// MMVQ hook: if `w` (a host-resident expert weight) has a DRE table, return its base pointer and the
// tagged (bit 0 = byte offsets) device table, and enqueue the ids readback used for residency decisions.
// `ids` may be null (the fused-gate lookup passes null: the paired tensor records them).
bool ggml_cuda_dre_lookup(const ggml_tensor * w, const ggml_tensor * ids, cudaStream_t stream,
                          const void ** base, const int64_t ** tagged_table);

// CPU miss path (ALPHA_DRE_CPU=1): ring the layer's doorbell before its gate/up GEMV (copies the input and
// ids to mapped memory), and after its down GEMV wait for the CPU's answer and write the missed experts'
// outputs into dst. No-ops otherwise.
void ggml_cuda_dre_pre(const ggml_tensor * w, const ggml_tensor * src1, const ggml_tensor * ids, cudaStream_t stream);
void ggml_cuda_dre_post(const ggml_tensor * w, ggml_tensor * dst, cudaStream_t stream);

// Called at the start of every CUDA graph_compute: once per token, publishes finished admissions and
// starts new ones (between tokens only, so a running kernel never sees a slot change under it).
void ggml_cuda_dre_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);
// Called after the graph is launched: records the event the next begin() waits on before it reads
// the ids readback.
void ggml_cuda_dre_graph_end(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph);

// ALPHA_DRE_PF_PIPE: for a routed prefill staging buffer filled in expert groups, returns the group count (0 = not one),
// e_end[g] = one past group g's last expert, ev[g] = its copies landed (record on the streaming stream)
int ggml_cuda_dre_pipe_groups(const void * buf, const int ** e_end, cudaEvent_t const ** ev);

// ALPHA_DRE_PF_CPU: after a pipelined prompt layer's down GEMM, write the CPU-computed rows of the cold experts over the
// GEMM's output (waits for the CPU job on the device). dst is [n_out, n_used, n_tok]; slot_stride in floats.
bool ggml_cuda_dre_pfc_fill(const void * staged_down, float * dst, int64_t n_out, int64_t slot_stride, cudaStream_t stream);
