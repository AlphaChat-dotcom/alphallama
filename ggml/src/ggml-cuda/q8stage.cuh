#pragma once

#include "common.cuh"

// Producer-side q8_1 staging. Enabled by ALPHA_Q8_STAGE.
//
// At decode sizes quantize_q8_1 is dominated by launch latency rather than work, so the gain comes from
// not launching it at all.
//
// A producer that already holds the activation in registers (rms_norm, GLU, a gated multiply) can emit
// the q8_1 copy as an epilogue for a few extra instructions. The consuming GEMV then skips its own
// quantise. Unlike ALPHA_Q8_CACHE this is not a runtime side-channel: the staged buffer is written by a
// kernel that is part of the captured CUDA graph, so it is produced on every replay, not once at
// capture. One producer can serve several consumers of the same activation (the post-attention norm
// feeds the router, the shared expert and the routed experts).
//
// Lifetime: entries are registered during a single graph-evaluation pass and cleared at its start. The
// backing allocation must outlive the pass -- see the owner vector in ggml_cuda_graph_evaluate_and_capture.
// A buffer whose lifetime ended before its reader would be reused by ggml's allocator; this buffer is
// owned here, not by ggml, and is held for the whole pass so that cannot happen.

struct alpha_q8_staged {
    const void * src1_data   = nullptr;   // the f32 activation this q8_1 copy corresponds to
    const char * q8          = nullptr;   // block_q8_1 buffer, MMVQ layout
    int64_t      ne10        = 0;         // logical row length
    int64_t      ne11 = 0, ne12 = 0, ne13 = 0;
    int64_t      s11 = 0, s12 = 0, s13 = 0;   // f32 strides the copy was made with
    int          produced_at = -1;        // node index of the producer, so its own write is not mistaken
                                          // for an overwrite
};

// Clear the registry. Call once per graph-evaluation pass, before the node loop.
void ggml_cuda_q8_stage_reset();

// Drop all entries without touching the counters. Call this whenever the memory the entries describe is
// recycled -- e.g. an engine that resets a workspace arena at each layer boundary must call it there, or
// layer N's entry will still be sitting at the address layer N+1 reuses, with identical geometry, and the
// consumer will silently read the wrong layer's activation. Registration already replaces a same-address
// entry, so this is for recycling that happens with no registration in between.
void ggml_cuda_q8_stage_clear();

// Register a staged q8_1 copy of src1_data. Silently ignored if the registry is full.
void ggml_cuda_q8_stage_put(const alpha_q8_staged & e);

// Look up a staged copy matching this activation and its shape/strides exactly. Returns nullptr on
// any mismatch -- a partial match must never be used, since the GEMV would read the wrong layout.
const char * ggml_cuda_q8_stage_get(const void * src1_data,
                                    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
                                    int64_t s11, int64_t s12, int64_t s13);

// Drop any entry whose activation overlaps [data, data+nbytes) -- the bytes have been handed to another
// node, so the staged copy no longer corresponds to the activation. `node_idx` is the index of the node
// doing the writing; an entry produced at that node is left alone, since a producer writing its own
// output is the point rather than a hazard.
void ggml_cuda_q8_stage_invalidate(const void * data, size_t nbytes, int node_idx);

bool ggml_cuda_q8_stage_enabled();

// Liveness counters: a fold that never fires is reported as such (hits == 0) rather than silently
// timing the unfused path.
void ggml_cuda_q8_stage_note_hit();
void ggml_cuda_q8_stage_note_staged();
void ggml_cuda_q8_stage_report();

// Record the shape of a GEMV that had to quantise for itself. The report then says exactly which
// activations are still paying for a quantise launch, i.e. where another producer epilogue would help.
void ggml_cuda_q8_stage_note_miss(int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13);

// Record a producer site the hook reached, and separately a staging refused because the arena was full,
// so "the hook never ran", "the plan declined" and "the arena filled" can be told apart.
void ggml_cuda_q8_stage_note_producer(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
void ggml_cuda_q8_stage_note_arena_full();

// Name the consumer that took a staged copy. A bare "consumed" count cannot tell an intended consumer
// from one that hit a stale entry at a reused address.
void ggml_cuda_q8_stage_note_hit_named(const char * consumer);
